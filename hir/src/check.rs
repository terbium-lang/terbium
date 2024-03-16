//! Typeck stage of HIR.

use crate::{
    error::Error,
    infer::{InferMetadata, TypeLowerer},
    lower::get_ident_from_ref,
    typed::{
        BinaryIntIntrinsic, BoolIntrinsic, Constraint, Expr, IntIntrinsic, LocalEnv, Relation, Ty,
        TypedExpr, UnaryIntIntrinsic, UnificationTable,
    },
    Hir, IntSign, Lookup, ModuleId, Node, Op, Pattern, PrimitiveTy, Scope, ScopeId,
};
use common::span::{Spanned, SpannedExt};

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

    pub fn take_table(&mut self) -> UnificationTable {
        std::mem::take(&mut self.lower.table)
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
    ) -> (Expr, Ty) {
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
        let stmt =
            if is_or {
                Expr::If(cond, then, Some(els))
            } else {
                Expr::If(cond, els, Some(then))
            };

        let spanned = |node| Spanned(node, span);
        let block =
            self.register_scope(Scope::new(
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
                    spanned(Node::ImplicitReturn(Spanned(TypedExpr(stmt, ty.clone()), span))),
                ]
                .spanned(span),
            ));
        (Expr::Block(block), ty)
    }

    fn lower_logical(
        &mut self,
        module_id: ModuleId,
        intrinsic: BoolIntrinsic,
        table: &mut UnificationTable,
    ) -> (Expr, Ty) {
        let (is_or, lhs, rhs) = match intrinsic {
            BoolIntrinsic::And(lhs, rhs) => (false, *lhs, *rhs),
            BoolIntrinsic::Or(lhs, rhs) => (true, *lhs, *rhs),
            // SAFETY: this is a private function only called when the intrinsic is a logical
            _ => unsafe { std::hint::unreachable_unchecked() },
        };
        let mut lhs_ty = lhs.value().1.clone();
        let mut rhs_ty = rhs.value().1.clone();

        // Unify the types of the LHS and RHS
        if let Some(conflict) = table.unify_constraint(Constraint(lhs_ty.clone(), rhs_ty.clone())) {
            self.lower.err_nonfatal(Error::TypeConflict {
                actual: rhs_ty.clone().spanned(rhs.span()),
                expected: (lhs_ty.clone(), Some(lhs.span())),
                constraint: conflict,
            });
        }
        lhs_ty.apply(&table.substitutions);
        rhs_ty.apply(&table.substitutions);

        // Determine the diverging type
        let ty =
            match lhs_ty.relation_to(&rhs_ty) {
                // Favor the LHS type if it is less or equally as specific
                Relation::Eq | Relation::Super => lhs_ty,
                // Otherwise, favor the RHS type
                Relation::Sub => rhs_ty,
                // If the types are unrelated, this is a type error
                Relation::Unrelated => {
                    // error: mismatched types
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
            self.lower_logical_into_if_stmt(module_id, lhs, rhs, ty, is_or)
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
            Op::Eq => Intrinsic::Eq,
            Op::Ne => Intrinsic::Ne,
            Op::Lt => Intrinsic::Lt,
            Op::Gt => Intrinsic::Gt,
            Op::Le => Intrinsic::Le,
            Op::Ge => Intrinsic::Ge,
            _ => return None,
        })
    }

    /// Lowers an op call into a potential intrinsic.
    pub fn lower_op(
        Spanned(op, op_span): Spanned<Op>,
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
            // int cmp_op int -> bool
            // int op int -> (widest, least specific) int
            (_, Some(intr), P(Int(lsign, lw)), Some(P(Int(rsign, rw)))) => {
                let sign = match (lsign, rsign) {
                    (IntSign::Unsigned, IntSign::Unsigned) => IntSign::Unsigned,
                    _ => IntSign::Signed,
                };
                let width = *lw.max(rw);
                *expr =
                    Expr::IntIntrinsic(
                        IntIntrinsic::Binary(intr, Box::new(target), Box::new(arg1.unwrap())),
                        sign,
                        width,
                    );
                *ty = if matches!(op, Op::Lt | Op::Gt | Op::Le | Op::Ge | Op::Eq | Op::Ne) {
                    BOOL
                } else {
                    P(Int(sign, width))
                };
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
            // pos a: int -> a
            (Op::Pos, _, P(Int(..)), None) => {
                let TypedExpr(new_expr, new_ty) = target.into_value();
                *expr = new_expr;
                *ty = new_ty;
            }
            // a: int ** b: int -> (a to float64) powi b -> float64
            (Op::Pow, _, P(Int(sign, width)), Some(P(Int(..)))) => todo!(),
            _ => (),
        }
    }

    /// Lower the given expression, performing any remaining type checking and desugaring.
    pub fn lower_expr(
        &mut self,
        module: ModuleId,
        Spanned(TypedExpr(expr, ty), span): &mut Spanned<TypedExpr>,
        table: &mut UnificationTable,
    ) {
        let initial = ty.clone();
        match expr {
            Expr::BoolIntrinsic(intrinsic) => {
                // TODO: intrinsic does not need to be cloned at all
                let (new_expr, new_ty) = self.lower_logical(module, intrinsic.clone(), table);
                *expr = new_expr;
                *ty = new_ty;
            }
            Expr::If(_, then_id, Some(else_id)) => {
                // TODO: lots of repetitive code here, this could be refactored.
                let scopes = &self.thir().scopes;
                let then_span = scopes.get(then_id).expect("missing scope").children.span();
                let els_span = scopes.get(else_id).expect("missing scope").children.span();

                let Some((mut then, tr_span)) = self.lower.resolution_lookup.remove(then_id) else {
                    return;
                };
                let Some((mut els, er_span)) = self.lower.resolution_lookup.remove(else_id) else {
                    return;
                };

                if let Some(conflict) =
                    table.unify_constraint(Constraint(then.clone(), els.clone()))
                {
                    self.lower.err_nonfatal(Error::TypeConflict {
                        actual: conflict.1.clone().spanned(els_span),
                        expected: (conflict.0.clone(), Some(then_span)),
                        constraint: conflict,
                    });
                }
                then.apply(&table.substitutions);
                els.apply(&table.substitutions);

                match then.relation_to(&els) {
                    // Favor the LHS type if it is less or equally as specific
                    Relation::Eq | Relation::Super => *ty = then.clone(),
                    // Otherwise, favor the RHS type
                    Relation::Sub => *ty = els.clone(),
                    // If the types are unrelated, this is a type error
                    Relation::Unrelated => {
                        // err(self, Constraint(then.clone(), els.clone()));
                        // Assume the RHS type
                        *ty = els.clone();
                    }
                }
                // Replace the then and else scopes with the new types
                self.lower
                    .resolution_lookup
                    .insert(*then_id, (then, tr_span));
                self.lower
                    .resolution_lookup
                    .insert(*else_id, (els, er_span));
            }
            Expr::CallOp(op, target, args) => {
                Self::lower_op(*op, (**target).clone(), args.first().cloned(), expr, ty);
            }
            _ => (),
        }
        // Unify the new type with the old type
        if let Some(conflict) = table.unify_constraint(Constraint(initial, ty.clone())) {
            self.lower.err_nonfatal(Error::TypeConflict {
                expected: (conflict.0.clone(), None),
                actual: conflict.1.clone().spanned(*span),
                constraint: conflict,
            });
        }
    }

    pub fn substitute_expr(
        &mut self,
        module: ModuleId,
        typed_expr: &mut Spanned<TypedExpr>,
        table: &mut UnificationTable,
    ) {
        match &mut typed_expr.value_mut().0 {
            Expr::Local(_, Some(tys), _) => tys
                .value_mut()
                .iter_mut()
                .for_each(|ty| ty.apply(&table.substitutions)),
            Expr::Type(ty) => ty.value_mut().apply(&table.substitutions),
            Expr::Tuple(values) | Expr::Array(values) | Expr::Intrinsic(_, values) => values
                .iter_mut()
                .for_each(|value| self.pass_over_expr(module, value, table)),
            Expr::CallFunc {
                parent,
                args,
                kwargs,
                ..
            } => {
                args.iter_mut()
                    .chain(kwargs.iter_mut().map(|(_, value)| value))
                    .for_each(|value| self.pass_over_expr(module, value, table));

                // Apply the substitutions to the parent type
                if let Some(ty) = parent.as_mut() {
                    ty.apply(&table.substitutions);
                }
            }
            Expr::CallOp(_, target, operands) => operands
                .iter_mut()
                .chain(std::iter::once(target.as_mut()))
                .for_each(|value| self.pass_over_expr(module, value, table)),
            Expr::CallStaticOp(_, ty, operands) => {
                operands
                    .iter_mut()
                    .for_each(|value| self.pass_over_expr(module, value, table));
                ty.apply(&table.substitutions);
            }
            Expr::Cast(value, ty) => {
                self.pass_over_expr(module, value, table);
                ty.apply(&table.substitutions);
            }
            Expr::GetField(value, _) | Expr::Assign(_, value) => {
                self.pass_over_expr(module, value, table);
            }
            Expr::SetField(value, _, new_value) => {
                self.pass_over_expr(module, value, table);
                self.pass_over_expr(module, new_value, table);
            }
            Expr::Block(scope) => self.substitute_scope(module, *scope, table),
            Expr::If(cond, then, els) => {
                self.pass_over_expr(module, cond, table);

                let cond_ty = &cond.value().1;
                if cond_ty != &BOOL {
                    self.lower.err_nonfatal(
                        Error::ConditionNotBool(Spanned(cond_ty.clone().into(), cond.span()))
                    );
                }
                self.substitute_scope(module, *then, table);
                if let Some(els) = els {
                    self.substitute_scope(module, *els, table);
                }
            }
            Expr::Loop(scope) => self.substitute_scope(module, *scope, table),
            Expr::AssignPtr(pointee, value) => {
                pointee.value_mut().1.apply(&table.substitutions);
                value.value_mut().1.apply(&table.substitutions);
            }
            _ => (),
        }
        typed_expr.value_mut().1.apply(&table.substitutions);
    }

    pub fn pass_over_expr(
        &mut self,
        module: ModuleId,
        typed_expr: &mut Spanned<TypedExpr>,
        table: &mut UnificationTable,
    ) {
        self.substitute_expr(module, typed_expr, table);
        self.lower_expr(module, typed_expr, table);
        // Substitute again to give fresh types
        self.substitute_expr(module, typed_expr, table);
    }

    /// Performs a shallow type substitution over the scope.
    pub fn substitute_scope(
        &mut self,
        module: ModuleId,
        scope_id: ScopeId,
        table: &mut UnificationTable,
    ) {
        // Remove the scope out of THIR temporarily to avoid accidentally mutating it later
        let mut scope = self
            .thir_mut()
            .scopes
            .remove(&scope_id)
            .expect("scope not found");

        // Substitute over all functions in the scope
        for (_, &Lookup(_, id)) in &scope.items {
            let scope = self.thir_mut().funcs[&id].body;
            self.substitute_scope(module, scope, table);

            let func = self.thir_mut().funcs.get_mut(&id).unwrap();
            func.header.ret_ty.apply(&table.substitutions);
        }

        // Substitute over the scope
        for child in scope.children.0.iter_mut() {
            match child.value_mut() {
                Node::Expr(expr)
                | Node::Break(_, Some(expr))
                | Node::Return(Some(expr))
                | Node::ImplicitReturn(expr) => {
                    self.pass_over_expr(module, expr, table);
                }
                Node::Let { ty, value, .. } => {
                    if let Some(value) = value {
                        self.pass_over_expr(module, value, table);
                    }
                    ty.apply(&table.substitutions);
                }
                _ => (),
            }
        }
        if let Some((ty, _)) = self.lower.resolution_lookup.get_mut(&scope_id) {
            ty.apply(&table.substitutions);
        }
        // Put the scope back into THIR
        self.thir_mut().scopes.insert(scope_id, scope);
    }

    /// Performs a typeck lowering over the module.
    pub fn check_module(&mut self, module_id: ModuleId, table: &mut UnificationTable) {
        let scope_id = *self.thir().modules.get(&module_id).unwrap();
        self.substitute_scope(module_id, scope_id, table);

        for (i, subst) in table.substitutions.iter().enumerate() {
            println!("${} => {}", i, subst);
        }
    }
}
