//! Typeck stage of HIR.

use crate::error::Error;
use crate::lower::get_ident_from_ref;
use crate::typed::{Expr, Relation, Ty, TypedExpr};
use crate::{
    error::Result,
    infer::{InferMetadata, TypeLowerer},
    typed::UnificationTable,
    Hir, Intrinsic, ModuleId, Node, Pattern, PrimitiveTy, Scope, ScopeId,
};
use common::span::{Spanned, SpannedExt};
use std::collections::VecDeque;

const BOOL: Ty = Ty::Primitive(PrimitiveTy::Bool);

/// Performs any remaining type checking and desugaring with the knowledge of the types of all
/// expressions.
pub struct TypeChecker {
    /// The original type lowerer.
    lower: TypeLowerer,
}

impl TypeChecker {
    /// Create a new type checker from the given TypeLowerer.
    pub fn from_lowerer(lower: TypeLowerer) -> Self {
        Self { lower }
    }

    pub const fn thir(&self) -> &Hir<InferMetadata> {
        &self.lower.thir
    }

    pub fn thir_mut(&mut self) -> &mut Hir<InferMetadata> {
        &mut self.lower.thir
    }

    pub const fn substitutions(&self) -> &VecDeque<Ty> {
        &self.lower.table.substitutions
    }

    #[inline]
    fn register_scope(&mut self, scope: Scope<InferMetadata>) -> ScopeId {
        let scopes = &mut self.thir_mut().scopes;
        let scope_id = ScopeId(scopes.len());
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
    ) -> Result<Spanned<TypedExpr>> {
        let span = lhs.span().merge(rhs.span());
        let binding = Spanned(get_ident_from_ref("__lhs"), lhs.span());

        let TypedExpr(lhs_expr, lhs_ty) = lhs.into_value();
        let binding_expr =
            TypedExpr(Expr::Ident(binding, None), lhs_ty.clone()).spanned(lhs.span());

        // Generate portions of the if-statement
        let cond = Box::new(Spanned(
            TypedExpr(Expr::Cast(Box::new(binding_expr.clone()), BOOL), BOOL),
            lhs.span(),
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
                    pat: lhs.as_ref().map(|_| Pattern::Ident {
                        ident: binding,
                        mut_kw: None,
                    }),
                    ty: lhs_ty.clone(),
                    ty_span: Some(lhs.span()),
                    value: Some(TypedExpr(lhs_expr, lhs_ty).spanned(lhs.span())),
                }),
                spanned(Node::ImplicitReturn(Spanned(
                    TypedExpr(stmt, ty.clone()),
                    span,
                ))),
            ]
            .spanned(span),
        ));
        Ok(Spanned(TypedExpr(Expr::Block(block), ty), span))
    }

    pub fn lower_logical(
        &mut self,
        intrinsic: Intrinsic,
        lhs: Spanned<TypedExpr>,
        rhs: Spanned<TypedExpr>,
    ) -> Result<TypedExpr> {
        let lhs_ty = lhs.0.1;
        let rhs_ty = rhs.0.1;
        // Determine the diverging type
        let ty = match lhs_ty.relation_to(&rhs_ty) {
            // Favor the LHS type if it is less or equally as specific
            Relation::Eq | Relation::Super => lhs_ty.clone(),
            // Otherwise, favor the RHS type
            Relation::Sub => rhs_ty.clone(),
            // If the types are unrelated, this is a type error
            Relation::Unrelated => {

            }
        }


    }

    /// Lower the given expression, performing any remaining type checking and desugaring.
    pub fn lower_expr(&mut self, expr: &mut Spanned<TypedExpr>) -> Result<()> {
        let TypedExpr(expr, ty) = expr.value_mut();
        match expr {
            Expr::Intrinsic(intrinsic @ (Intrinsic::BoolAnd | Intrinsic::BoolOr), args) => {
                // These intrinsics might need to be desugared
                let lhs = args.remove(0);
                let rhs = args.remove(0);
                let TypedExpr(new_expr, new_ty) = self.lower_logical(*intrinsic, lhs, rhs)?;
                *expr = new_expr;
                *ty = new_ty;
            }

            _ => (),
        }
        Ok(())
    }

    pub fn substitute_expr(
        &mut self,
        TypedExpr(expr, ty): &mut TypedExpr,
        substitutions: &VecDeque<Ty>,
    ) {
        ty.apply(substitutions);
        match expr {
            Expr::Ident(_, Some(tys)) => tys
                .value_mut()
                .iter_mut()
                .for_each(|ty| ty.apply(substitutions)),
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
            Expr::Block(scope) => self.substitute_scope(*scope, substitutions),
            Expr::If(cond, then, els) => {
                let cond_ty = &mut cond.value_mut().1;
                cond_ty.apply(substitutions);
                if cond_ty != &BOOL {
                    self.lower.err_nonfatal(Error::ConditionNotBool(Spanned(
                        cond_ty.clone().into(),
                        cond.span(),
                    )));
                }
                self.substitute_scope(*then, substitutions);
                if let Some(els) = els {
                    self.substitute_scope(*els, substitutions);
                }
            }
            Expr::Loop(scope) => self.substitute_scope(*scope, substitutions),
            Expr::AssignPtr(pointee, value) => {
                pointee.value_mut().1.apply(substitutions);
                value.value_mut().1.apply(substitutions);
            }
            _ => (),
        }
    }

    /// Performs a shallow type substitution over the scope.
    pub fn substitute_scope(&mut self, scope: ScopeId, substitutions: &VecDeque<Ty>) {
        let scope = self
            .thir_mut()
            .scopes
            .get_mut(&scope)
            .expect("scope not found");

        for child in scope.children.0.iter_mut() {
            match child.value_mut() {
                Node::Expr(Spanned(expr, _))
                | Node::Break(_, Some(Spanned(expr, _)))
                | Node::Return(Some(Spanned(expr, _)))
                | Node::ImplicitReturn(Spanned(expr, _)) => {
                    self.substitute_expr(expr, substitutions);
                }
                Node::Let { ty, value, .. } => {
                    ty.apply(substitutions);
                    if let Some(Spanned(value, _)) = value {
                        self.substitute_expr(value, substitutions);
                    }
                }
                _ => (),
            }
        }
    }
}
