//! Typeck stage of HIR.

use crate::error::Error;
use crate::error::Error::TypeConflict;
use crate::infer::ScopeKind;
use crate::lower::get_ident_from_ref;
use crate::typed::{
    BinaryIntIntrinsic, BoolIntrinsic, Constraint, Expr, IntIntrinsic, LocalEnv, Relation, Ty,
    TypedExpr, UnaryIntIntrinsic,
};
use crate::{
    error::Result,
    infer::{InferMetadata, TypeLowerer},
    Hir, IntSign, Intrinsic, ModuleId, Node, Op, Pattern, PrimitiveTy, Scope, ScopeId,
};
use common::span::{Spanned, SpannedExt};
use std::collections::VecDeque;

const BOOL: Ty = Ty::Primitive(PrimitiveTy::Bool);

/// Performs any remaining type checking and desugaring with the knowledge of the types of all
/// expressions.
pub struct TypeChecker<'a> {
    /// The original type lowerer.
    pub lower: &'a mut TypeLowerer,
}

impl<'a> TypeChecker<'a> {
    /// Create a new type checker from the given TypeLowerer.
    pub fn from_lowerer(lower: &'a mut TypeLowerer) -> Self {
        Self { lower }
    }

    pub const fn thir(&self) -> &Hir<InferMetadata> {
        &self.lower.thir
    }

    pub fn thir_mut(&mut self) -> &mut Hir<InferMetadata> {
        &mut self.lower.thir
    }

    pub fn take_substitutions(&mut self) -> VecDeque<Ty> {
        self.lower.table.substitutions.drain(..).collect()
    }

    #[inline]
    fn register_scope(&mut self, scope: Scope<InferMetadata>) -> ScopeId {
        let scopes = &mut self.thir_mut().scopes;
        let scope_id = ScopeId(scopes.len() + 1);
        scopes.insert(scope_id, scope);
        scope_id
    }

    #[inline]
    fn anon_scope_from_expr(&mut self, module: ModuleId, expr: Spanned<TypedExpr>) -> ScopeId {
        let span = expr.span();
        self.register_scope(Scope::new(
            module,
            None,
            Spanned(vec![Spanned(Node::ImplicitReturn(expr), span)], span),
        ))
    }

    /// Desugar logical operators into if statements.
    ///
    /// For example, `a || b` becomes:
    /// ```text
    /// {
    ///     let __lhs = a;
    ///     if __lhs to bool { __lhs } else { b }
    /// }
    /// ```
    ///
    /// Similarly, `a && b` becomes:
    /// ```text
    /// {
    ///     let __lhs = a;
    ///     if __lhs to bool { b } else { __lhs }
    /// }
    #[inline]
    pub fn lower_logical_into_if_stmt(
        &mut self,
        module: ModuleId,
        lhs: Spanned<TypedExpr>,
        rhs: Spanned<TypedExpr>,
        ty: Ty,
        is_or: bool,
    ) -> Spanned<TypedExpr> {
        let span = lhs.span().merge(rhs.span());
        let binding = Spanned(get_ident_from_ref("__lhs"), lhs.span());

        let Spanned(TypedExpr(lhs_expr, lhs_ty), lhs_span) = lhs;
        let binding_expr = TypedExpr(
            Expr::Local(binding, None, LocalEnv::Internal),
            lhs_ty.clone(),
        )
        .spanned(lhs_span);

        // Generate portions of the if-statement
        let cond = Box::new(Spanned(
            TypedExpr(Expr::Cast(Box::new(binding_expr.clone()), BOOL), BOOL),
            lhs_span,
        ));
        let then = self.anon_scope_from_expr(module, binding_expr);
        let els = self.anon_scope_from_expr(module, rhs);
        let stmt = if is_or {
            Expr::If(cond, then, Some(els))
        } else {
            Expr::If(cond, els, Some(then))
        };

        let spanned = |node| Spanned(node, span);
        let block = self.register_scope(Scope::new(
            module,
            None,
            vec![
                // Store the LHS in a temporary binding
                spanned(Node::Let {
                    pat: Spanned(
                        Pattern::Ident {
                            ident: binding,
                            mut_kw: None,
                        },
                        lhs_span,
                    ),
                    ty: lhs_ty.clone(),
                    ty_span: Some(lhs_span),
                    value: Some(TypedExpr(lhs_expr, lhs_ty).spanned(lhs_span)),
                }),
                spanned(Node::ImplicitReturn(Spanned(
                    TypedExpr(stmt, ty.clone()),
                    span,
                ))),
            ]
            .spanned(span),
        ));
        Spanned(TypedExpr(Expr::Block(block), ty), span)
    }

    fn lower_logical(&mut self, module_id: ModuleId, intrinsic: BoolIntrinsic) -> (Expr, Ty) {
        let (is_or, lhs, rhs) = match intrinsic {
            BoolIntrinsic::And(lhs, rhs) => (false, *lhs, *rhs),
            BoolIntrinsic::Or(lhs, rhs) => (true, *lhs, *rhs),
            // SAFETY: this is a private function only called when the intrinsic is a logical
            _ => unsafe { std::hint::unreachable_unchecked() },
        };
        let lhs_ty = lhs.value().1.clone();
        let rhs_ty = rhs.value().1.clone();
        // Determine the diverging type
        let ty = match lhs_ty.relation_to(&rhs_ty) {
            // Favor the LHS type if it is less or equally as specific
            Relation::Eq | Relation::Super => lhs_ty,
            // Otherwise, favor the RHS type
            Relation::Sub => rhs_ty,
            // If the types are unrelated, this is a type error
            Relation::Unrelated => {
                self.lower.err_nonfatal(TypeConflict {
                    actual: crate::Ty::from(rhs_ty.clone()).spanned(rhs.span()),
                    expected: (crate::Ty::from(lhs_ty.clone()), Some(lhs.span())),
                    constraint: Constraint(lhs_ty, rhs_ty.clone()),
                });
                // Assume the RHS type
                rhs_ty
            }
        };

        if ty.relation_to(&BOOL).compatible() {
            let intrinsic = match is_or {
                true => BoolIntrinsic::Or(Box::new(lhs), Box::new(rhs)),
                false => BoolIntrinsic::And(Box::new(lhs), Box::new(rhs)),
            };
            (Expr::BoolIntrinsic(intrinsic), ty)
        } else {
            // If the types are not bool, desugar into an if-statement
            let desugared = self
                .lower_logical_into_if_stmt(module_id, lhs, rhs, ty, is_or)
                .into_value();
            (desugared.0, desugared.1)
        }
    }

    #[inline]
    const fn op_to_binary_int_intrinsic(op: Op) -> Option<BinaryIntIntrinsic> {
        type Intrinsic = BinaryIntIntrinsic;

        Some(match op {
            Op::Add => Intrinsic::Add,
            Op::Sub => Intrinsic::Sub,
            Op::Mul => Intrinsic::Mul,
            Op::Div => Intrinsic::Div,
            Op::Mod => Intrinsic::Mod,
            Op::Shl => Intrinsic::Shl,
            Op::Shr => Intrinsic::Shr,
            Op::BitAnd => Intrinsic::BitAnd,
            Op::BitOr => Intrinsic::BitOr,
            Op::BitXor => Intrinsic::BitXor,
            Op::Lt => Intrinsic::Lt,
            Op::Gt => Intrinsic::Gt,
            Op::Le => Intrinsic::Le,
            Op::Ge => Intrinsic::Ge,
            Op::Eq => Intrinsic::Eq,
            _ => return None,
        })
    }

    /// Lowers an op call into a potential intrinsic.
    pub fn lower_op(
        op: Op,
        target: Spanned<TypedExpr>,
        arg1: Option<Spanned<TypedExpr>>,
        expr: &mut Expr,
        ty: &mut Ty,
    ) {
        use PrimitiveTy::Int;
        use Ty::Primitive as P;

        let int_intrinsic = Self::op_to_binary_int_intrinsic(op);

        let arg1_ty = arg1.as_ref().map(|arg| &arg.value().1);
        match (op, int_intrinsic, &target.value().1, arg1_ty) {
            // int op int -> (widest, least specific) int
            (_, Some(intr), P(Int(lsign, lw)), Some(P(Int(rsign, rw)))) => {
                let sign = match (lsign, rsign) {
                    (IntSign::Unsigned, IntSign::Unsigned) => IntSign::Unsigned,
                    _ => IntSign::Signed,
                };
                let width = *lw.max(rw);
                *expr = Expr::IntIntrinsic(
                    IntIntrinsic::Binary(intr, Box::new(target), Box::new(arg1.unwrap())),
                    sign,
                    width,
                );
                *ty = P(Int(sign, width));
            }
            // op int -> int
            (Op::BitNot | Op::Neg, _, &P(Int(sign, width)), None) => {
                let intr = match op {
                    Op::BitNot => UnaryIntIntrinsic::BitNot,
                    Op::Neg => UnaryIntIntrinsic::Neg,
                    // SAFETY: this is only called when the op is BitNot or Neg
                    _ => unsafe { std::hint::unreachable_unchecked() },
                };
                *expr =
                    Expr::IntIntrinsic(IntIntrinsic::Unary(intr, Box::new(target)), sign, width);
                *ty = P(Int(sign, width));
            }
            // a: int ** b: int -> (a to float64) powi b -> float64
            (Op::Pow, _, P(Int(sign, width)), Some(P(Int(..)))) => todo!(),
            _ => (),
        }
    }

    /// Lower the given expression, performing any remaining type checking and desugaring.
    pub fn lower_expr(&mut self, module: ModuleId, expr: &mut Spanned<TypedExpr>) {
        let TypedExpr(expr, ty) = expr.value_mut();
        match expr {
            Expr::BoolIntrinsic(intrinsic) => {
                // TODO: intrinsic does not need to be cloned at all
                let (new_expr, new_ty) = self.lower_logical(module, intrinsic.clone());
                *expr = new_expr;
                *ty = new_ty;
            }
            Expr::If(_, then, Some(els)) => {
                // TODO: lots of repetitive code here, this could be refactored.
                let scopes = &self.thir().scopes;
                let then_span = scopes.get(then).expect("missing scope").children.span();
                let els_span = scopes.get(els).expect("missing scope").children.span();

                let Some(then) = self
                    .lower
                    .raw_resolution_lookup
                    .get(then) else { return };
                let Some(els) = self
                    .lower
                    .raw_resolution_lookup
                    .get(els) else { return };
                // SAFETY: since self.lower is a valid reference to the lowerer, and the lowerer
                //   owns the type resolution table, this is safe.
                let (then, els) = unsafe { (&mut **then, &mut **els) };

                match then.relation_to(&els) {
                    // Favor the LHS type if it is less or equally as specific
                    Relation::Eq | Relation::Super => *ty = then.clone(),
                    // Otherwise, favor the RHS type
                    Relation::Sub => *ty = els.clone(),
                    // If the types are unrelated, this is a type error
                    Relation::Unrelated => {
                        self.lower.err_nonfatal(TypeConflict {
                            actual: crate::Ty::from(els.clone()).spanned(els_span),
                            expected: (crate::Ty::from(then.clone()), Some(then_span)),
                            constraint: Constraint(then.clone(), els.clone()),
                        });
                        // Assume the RHS type
                        *ty = els.clone();
                    }
                }
            }
            Expr::CallOp(op, target, args) => {
                Self::lower_op(*op, (**target).clone(), args.first().cloned(), expr, ty);
            }
            _ => (),
        }
    }

    pub fn substitute_expr(
        &mut self,
        module: ModuleId,
        typed_expr: &mut Spanned<TypedExpr>,
        substitutions: &VecDeque<Ty>,
    ) {
        typed_expr.value_mut().1.apply(substitutions);
        match &mut typed_expr.value_mut().0 {
            Expr::Local(_, Some(tys), _) => tys
                .value_mut()
                .iter_mut()
                .for_each(|ty| ty.apply(substitutions)),
            Expr::Type(ty) => ty.value_mut().apply(substitutions),
            Expr::Tuple(values) | Expr::Array(values) | Expr::Intrinsic(_, values) => values
                .iter_mut()
                .for_each(|value| value.value_mut().1.apply(substitutions)),
            Expr::Call { args, kwargs, .. } => args
                .iter_mut()
                .chain(kwargs.iter_mut().map(|(_, value)| value))
                .for_each(|value| value.value_mut().1.apply(substitutions)),
            Expr::CallOp(_, target, operands) => operands
                .iter_mut()
                .chain(std::iter::once(target.as_mut()))
                .for_each(|value| value.value_mut().1.apply(substitutions)),
            Expr::CallStaticOp(_, ty, operands) => {
                ty.apply(substitutions);
                operands
                    .iter_mut()
                    .for_each(|value| value.value_mut().1.apply(substitutions))
            }
            Expr::Cast(value, ty) => {
                value.value_mut().1.apply(substitutions);
                ty.apply(substitutions);
            }
            Expr::GetAttr(value, _) | Expr::Assign(_, value) => {
                value.value_mut().1.apply(substitutions)
            }
            Expr::SetAttr(value, _, new_value) => {
                value.value_mut().1.apply(substitutions);
                new_value.value_mut().1.apply(substitutions);
            }
            Expr::Block(scope) => self.substitute_scope(module, *scope, substitutions),
            Expr::If(cond, then, els) => {
                let cond_ty = &mut cond.value_mut().1;
                cond_ty.apply(substitutions);
                if cond_ty != &BOOL {
                    self.lower.err_nonfatal(Error::ConditionNotBool(Spanned(
                        cond_ty.clone().into(),
                        cond.span(),
                    )));
                }
                self.substitute_scope(module, *then, substitutions);
                if let Some(els) = els {
                    self.substitute_scope(module, *els, substitutions);
                }
            }
            Expr::Loop(scope) => self.substitute_scope(module, *scope, substitutions),
            Expr::AssignPtr(pointee, value) => {
                pointee.value_mut().1.apply(substitutions);
                value.value_mut().1.apply(substitutions);
            }
            _ => (),
        }
    }

    pub fn pass_over_expr(
        &mut self,
        module: ModuleId,
        typed_expr: &mut Spanned<TypedExpr>,
        substitutions: &VecDeque<Ty>,
    ) {
        self.substitute_expr(module, typed_expr, substitutions);
        self.lower_expr(module, typed_expr);
        // Substitute again to give fresh types
        self.substitute_expr(module, typed_expr, substitutions);
    }

    /// Performs a shallow type substitution over the scope.
    pub fn substitute_scope(
        &mut self,
        module: ModuleId,
        scope_id: ScopeId,
        substitutions: &VecDeque<Ty>,
    ) {
        // Remove the scope out of THIR temporarily to avoid accidentally mutating it later
        let mut scope = self
            .thir_mut()
            .scopes
            .remove(&scope_id)
            .expect("scope not found");

        for child in scope.children.0.iter_mut() {
            match child.value_mut() {
                Node::Expr(expr)
                | Node::Break(_, Some(expr))
                | Node::Return(Some(expr))
                | Node::ImplicitReturn(expr) => {
                    self.pass_over_expr(module, expr, substitutions);
                }
                Node::Let { ty, value, .. } => {
                    ty.apply(substitutions);
                    if let Some(value) = value {
                        self.pass_over_expr(module, value, substitutions);
                    }
                }
                _ => (),
            }
        }
        if let Some(ty) = self.lower.raw_resolution_lookup.get_mut(&scope_id) {
            // SAFETY: since self.lower is a valid reference to the lowerer, and the lowerer
            //   owns the type resolution table, this is safe.
            unsafe {
                (**ty).apply(substitutions);
            }
        }
        // Put the scope back into THIR
        self.thir_mut().scopes.insert(scope_id, scope);
    }

    /// Performs a typeck lowering over the module.
    pub fn check_module(&mut self, module_id: ModuleId, substitutions: &VecDeque<Ty>) {
        let scope_id = *self.thir().modules.get(&module_id).unwrap();
        self.substitute_scope(module_id, scope_id, substitutions);
    }
}
