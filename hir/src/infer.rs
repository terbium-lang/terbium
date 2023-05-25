use crate::{
    error::{Error, Result},
    typed::{self, Constraint, InvalidTypeCause, Ty, TypedExpr, UnificationTable},
    warning::Warning,
    Expr, FloatWidth, Hir, Ident, IntSign, IntWidth, ItemId, Literal, Metadata, ModuleId, Node,
    Pattern, PrimitiveTy, TyParam,
};
use common::span::{Span, Spanned};
use std::{borrow::Cow, collections::HashMap};

impl UnificationTable {
    pub fn substitute(&mut self, a: usize, b: Ty) {
        // Add a => b to our substitution
        self.substitutions[a] = b.clone();
        // Replace further unifications of a with b
        for Constraint(x, y) in self.constraints.iter_mut() {
            x.substitute(a, b.clone());
            y.substitute(a, b.clone());
        }
    }

    /// Unify the constraint a = b. Return failed constraint on conflict.
    pub fn unify_constraint(&mut self, constraint: Constraint) -> Option<Constraint> {
        match constraint {
            // If both a and b are unknown, replace one of them with the other.
            Constraint(Ty::Unknown(i), b @ Ty::Unknown(_)) => {
                self.substitute(i, b);
            }
            // If a is unknown and b is known, replace a with b. Similarly, if b is unknown and a is
            // known, replace b with a.
            Constraint(Ty::Unknown(i), mut b) | Constraint(mut b, Ty::Unknown(i)) => {
                // Do not allow recursive unification (i = a<i>)
                if b.has_unknown(i) {
                    b = Ty::Invalid(InvalidTypeCause::CyclicTypeReference);
                }
                self.substitute(i, b);
            }
            // If both a and b are known, check if they are the same general type (e.g. A<...> = A<...>)
            // and if so, unify their type arguments.
            Constraint(a, b) if b.has_same_outer_type(&a) => {
                for (a, b) in a.into_inner_unifications().zip(b.into_inner_unifications()) {
                    if let c @ Some(_) = self.unify_constraint(Constraint(a, b)) {
                        return c;
                    }
                }
            }
            // If both a and b are known but they are not the same general type, this is a type
            // mismatch
            _ => return Some(constraint),
        }
        None
    }

    /// Unifies all constraints in the unifier. Return erroneous constraint if unification fails.
    pub fn unify_all(&mut self) -> Option<Constraint> {
        while let Some(constraint) = self.constraints.pop_front() {
            if let constraint @ Some(_) = self.unify_constraint(constraint) {
                return constraint;
            }
        }
        None
    }
}

struct Binding {
    pub def_span: Span,
    pub ty: Ty,
    pub mutable: Option<Span>,
}

#[derive(Clone, Debug)]
pub struct Local {
    pub def_span: Span,
    pub ty: Ty,
    pub mutable: Option<Span>,
    // analysis checks
    pub used: bool,
    pub mutated: bool,
}

impl Local {
    #[inline]
    pub const fn new(def_span: Span, ty: Ty, mutable: Option<Span>) -> Self {
        Self {
            def_span,
            ty,
            mutable,
            used: false,
            mutated: false,
        }
    }

    #[inline]
    fn from_binding(binding: Binding) -> Self {
        Self {
            def_span: binding.def_span,
            ty: binding.ty,
            mutable: binding.mutable,
            used: false,
            mutated: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Scope {
    module_id: ModuleId,
    ty_params: Vec<TyParam>,
    locals: HashMap<Ident, Local>,
}

#[derive(Clone, Debug)]
pub struct InferMetadata;
impl Metadata for InferMetadata {
    type Expr = TypedExpr;
    type Ty = Ty;
}

/// Lowers types of expressions and performs type inference and one pass of typeck.
#[derive(Debug)]
pub struct TypeLowerer {
    scopes: Vec<Scope>,
    table: UnificationTable,
    /// The HIR that is being lowered.
    pub hir: Hir,
    /// The HIR that is being generated.
    pub thir: Hir<InferMetadata>,
    /// Warnings that occurred during type inference.
    pub warnings: Vec<Warning>,
    /// Non-fatal errors that occurred during type inference.
    pub errors: Vec<Error>,
}

impl TypeLowerer {
    /// Creates a new type lowerer over the given HIR.
    pub fn new(hir: Hir) -> Self {
        Self {
            scopes: Vec::new(),
            table: UnificationTable::default(),
            hir,
            thir: Hir::default(),
            warnings: Vec::new(),
            errors: Vec::new(),
        }
    }

    #[inline]
    pub fn scope(&self) -> &Scope {
        self.scopes.last().expect("no scopes entered?")
    }

    #[inline]
    pub fn scope_mut(&mut self) -> &mut Scope {
        self.scopes.last_mut().expect("no scopes entered?")
    }

    pub fn err_nonfatal(&mut self, err: impl Into<Error>) {
        self.errors.push(err.into());
    }

    pub fn enter_scope(&mut self, module: ModuleId) -> &mut Self {
        self.scopes.push(Scope {
            module_id: module,
            ty_params: Vec::new(),
            locals: HashMap::new(),
        });
        self
    }

    pub fn exit_scope(&mut self) {
        let scope = self.scopes.pop().expect("no scopes entered?");
        for (ident, local) in scope.locals {
            if !local.used {
                self.warnings.push(Warning::UnusedVariable(Spanned(
                    ident.to_string(),
                    local.def_span,
                )))
            }
            if !local.mutated && let Some(mutable) = local.mutable {
                self.warnings.push(Warning::UnusedMut(Spanned(ident.to_string(), local.def_span), mutable))
            }
        }
    }

    pub fn lower_literal(lit: &Literal) -> Ty {
        match lit {
            Literal::Void => Ty::Primitive(PrimitiveTy::Void),
            Literal::Bool(_) => Ty::Primitive(PrimitiveTy::Bool),
            Literal::Int(i) => {
                // Fit the integer into the smallest possible type (at least 32 bits)
                let width = if i >= &i32::MIN.into() && i <= &i32::MAX.into() {
                    IntWidth::Int32
                } else if i >= &i64::MIN.into() && i <= &i64::MAX.into() {
                    IntWidth::Int64
                } else {
                    IntWidth::Int128
                };
                Ty::Primitive(PrimitiveTy::Int(IntSign::Signed, width))
            }
            Literal::UInt(u) => {
                let width = if u <= &u32::MAX.into() {
                    IntWidth::Int32
                } else if u <= &u64::MAX.into() {
                    IntWidth::Int64
                } else {
                    IntWidth::Int128
                };
                Ty::Primitive(PrimitiveTy::Int(IntSign::Unsigned, width))
            }
            Literal::Float(_) => Ty::Primitive(PrimitiveTy::Float(FloatWidth::Float64)),
            Literal::Char(_) => Ty::Primitive(PrimitiveTy::Char),
            Literal::String(_) => unimplemented!(),
        }
    }

    #[inline]
    fn lower_hir_ty(&mut self, ty: crate::Ty) -> Ty {
        Ty::from_ty(ty, &mut self.table)
    }

    #[inline]
    fn lower_ident_ty(
        &mut self,
        ident: &Spanned<Ident>,
        args: &Option<Spanned<Vec<crate::Ty>>>,
    ) -> Result<Ty> {
        if let Some(args) = args {
            // TODO: function pointers, first-class types
            return Err(Error::ExplicitTypeArgumentsNotAllowed(args.span()));
        }

        // Search for the local variable
        for scope in self.scopes.iter_mut().rev() {
            if let Some(local) = scope.locals.get_mut(ident.value()) {
                local.used = true;
                return Ok(local.ty.clone());
            }
        }

        // Does a constant with this name exist?
        let item = ItemId(self.scope().module_id, *ident.value());
        if let Some(cnst) = self.hir.consts.get(&item) {
            return Ok(self.lower_hir_ty(cnst.ty.clone()));
        }

        Err(Error::UnresolvedIdentifier(ident.map(|i| i.to_string())))
    }

    /// Lowers the type of an expression.
    pub fn lower_expr_ty(&mut self, expr: &Spanned<Expr>) -> Result<Ty> {
        Ok(match expr.value() {
            Expr::Literal(lit) => Self::lower_literal(lit),
            Expr::Ident(ident, args) => self.lower_ident_ty(ident, args)?,
            Expr::Tuple(exprs) => {
                let mut tys = Vec::with_capacity(exprs.len());
                for expr in exprs {
                    tys.push(self.lower_expr_ty(expr)?);
                }
                Ty::Tuple(tys)
            }
            Expr::Cast(_, ty) => self.lower_hir_ty(ty.clone()),
            Expr::If(cond, left, right) => {
                let cond_ty = match self.lower_expr_ty(cond)? {
                    // If the type is not known, constrain it to bool, this will be checked
                    // one more time at the typeck stage.
                    ty @ Ty::Unknown(_) => {
                        self.table
                            .constraints
                            .push_back(Constraint(ty.clone(), Ty::Primitive(PrimitiveTy::Bool)));
                        ty
                    }
                    // If the type is known, check that it is bool
                    ty @ Ty::Primitive(PrimitiveTy::Bool) => ty,
                    // Otherwise, error
                    ty => {
                        return Err(Error::ConditionNotBool(Spanned(ty.into(), cond.span())));
                    }
                };
                // "Evenness" check will be done again at the typeck stage
                todo!()
            }
            // TODO
            _ => self.lower_hir_ty(crate::Ty::Unknown),
        })
    }

    pub fn lower_expr(&mut self, expr: Spanned<Expr>) -> Result<Spanned<TypedExpr>> {
        let ty = self.lower_expr_ty(&expr)?;
        let Spanned(expr, span) = expr;

        let expr = match expr {
            Expr::Literal(lit) => typed::Expr::Literal(lit),
            Expr::Ident(ident, args) => typed::Expr::Ident(
                ident,
                args.map(|args| {
                    args.map(|args| args.into_iter().map(|ty| self.lower_hir_ty(ty)).collect())
                }),
            ),
            Expr::Tuple(exprs) => typed::Expr::Tuple(
                exprs
                    .into_iter()
                    .map(|expr| self.lower_expr(expr))
                    .collect::<Result<Vec<_>>>()?,
            ),
            Expr::Cast(expr, ty) => {
                typed::Expr::Cast(Box::new(self.lower_expr(*expr)?), self.lower_hir_ty(ty))
            }
            _ => unimplemented!(),
        };
        Ok(Spanned(TypedExpr(expr, ty), span))
    }

    fn bind_pattern_to_ty(
        pat: &Spanned<Pattern>,
        ty: Ty,
        expr: Option<&Spanned<TypedExpr>>,
        bindings: &mut Vec<(Ident, Binding)>,
    ) -> Result<()> {
        match (pat.value(), ty) {
            (_, Ty::Unknown(_) | Ty::Invalid(_)) => {}
            (Pattern::Ident { ident, mut_kw }, ty) => {
                bindings.push((
                    ident.0,
                    Binding {
                        def_span: ident.1,
                        ty,
                        mutable: mut_kw.clone(),
                    },
                ));
            }
            (Pattern::Tuple(pats), Ty::Tuple(tys)) => {
                if pats.len() != tys.len() {
                    return Err(Error::PatternMismatch {
                        pat: Cow::Owned(format!("tuple of length {}", pats.len())),
                        pat_span: pat.span(),
                        value: format!("tuple of length {}", tys.len()),
                        value_span: expr.map(|expr| expr.span()),
                    });
                }
                // Can we destructure the tuple further?
                if let Some(Spanned(TypedExpr(typed::Expr::Tuple(ref exprs), _), _)) = expr {
                    debug_assert_eq!(pats.len(), exprs.len());
                    for ((pat, ty), expr) in pats.iter().zip(tys).zip(exprs) {
                        Self::bind_pattern_to_ty(pat, ty, Some(expr), bindings)?;
                    }
                } else {
                    for (pat, ty) in pats.iter().zip(tys) {
                        Self::bind_pattern_to_ty(pat, ty, expr, bindings)?;
                    }
                }
            }
            // TODO: tuple structs/enum variants?
            (Pattern::Tuple(_), ty) => {
                return Err(Error::PatternMismatch {
                    pat: Cow::Borrowed("tuple"),
                    pat_span: pat.span(),
                    value: format!("value of type `{}`", crate::Ty::from(ty)),
                    value_span: expr.map(|expr| expr.span()),
                })
            }
        }
        Ok(())
    }

    /// Runs type inference through a single node.
    ///
    /// Return (node, exit_ty?)
    pub fn lower_node(&mut self, node: Node) -> Result<(Node<InferMetadata>, Option<Spanned<Ty>>)> {
        let mut bindings = Vec::new();
        let res = match node {
            Node::Expr(expr) => (Node::Expr(self.lower_expr(expr)?), None),
            Node::Let {
                pat,
                ty,
                ty_span,
                value,
            } => {
                // Lower the type of the binding
                let node = if let Some(value) = value {
                    let mut expr = self.lower_expr(value)?;
                    let mut lower_ty = self.lower_hir_ty(ty.clone());

                    self.table
                        .constraints
                        .push_back(Constraint(lower_ty.clone(), expr.value().1.clone()));
                    let conflict = self.table.unify_all();

                    lower_ty.apply(&self.table.substitutions);
                    expr.value_mut().1.apply(&self.table.substitutions);

                    if let Some(conflict) = conflict {
                        self.err_nonfatal(Error::TypeConflict {
                            expected: (ty, ty_span),
                            actual: expr.as_ref().map(|expr| expr.1.clone().into()),
                            constraint: conflict,
                        })
                    }

                    Self::bind_pattern_to_ty(&pat, lower_ty.clone(), Some(&expr), &mut bindings)?;
                    Node::Let {
                        pat,
                        ty: lower_ty,
                        ty_span,
                        value: Some(expr),
                    }
                } else {
                    let ty = self.lower_hir_ty(ty);
                    Self::bind_pattern_to_ty(&pat, ty.clone(), None, &mut bindings)?;

                    Node::Let {
                        pat,
                        ty,
                        ty_span,
                        value: None,
                    }
                };
                (node, None)
            }
            Node::ImplicitReturn(expr) => {
                let expr = self.lower_expr(expr)?;
                let exit_ty = expr.as_ref().map(|expr| expr.1.clone());
                (Node::ImplicitReturn(expr), Some(exit_ty))
            }
            _ => todo!(),
        };
        for (ident, binding) in bindings {
            self.scope_mut()
                .locals
                .insert(ident, Local::from_binding(binding));
        }
        Ok(res)
    }
}
