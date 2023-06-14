use crate::typed::Relation;
use crate::{
    error::{Error, Result},
    typed::{self, Constraint, InvalidTypeCause, Ty, TypedExpr, UnificationTable},
    warning::Warning,
    Expr, FloatWidth, Hir, Ident, IntSign, IntWidth, ItemId, Literal, Metadata, ModuleId, Node,
    Pattern, PrimitiveTy, ScopeId, TyParam,
};
use common::span::{Span, Spanned, SpannedExt};
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
    #[must_use = "unification returns Some(conflict) if there is a conflict"]
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
    id: ScopeId,
    module_id: ModuleId,
    kind: ScopeKind,
    ty_params: Vec<TyParam>,
    locals: HashMap<Ident, Local>,
    label: Option<Spanned<Ident>>,
    resolution: Ty,
    exited: Option<ExitAction>,
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
    // raw pointers are used here because:
    // 1. it can be guaranteed that the pointers are valid for the lifetime of the type lowerer
    // 2. it's faster
    raw_resolution_lookup: HashMap<ScopeId, *mut Ty>,
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
            raw_resolution_lookup: HashMap::new(),
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

    pub fn enter_scope(
        &mut self,
        scope_id: ScopeId,
        module: ModuleId,
        kind: ScopeKind,
        ty_params: Vec<TyParam>,
        label: Option<Spanned<Ident>>,
    ) -> Ty {
        let resolution = Ty::from_ty(crate::Ty::Unknown, &mut self.table);
        let scope = Scope {
            id: scope_id,
            module_id: module,
            kind,
            ty_params,
            locals: HashMap::new(),
            label,
            resolution: resolution.clone(),
            exited: None,
        };
        self.scopes.push(scope);
        // SAFETY: the scope has just been pushed
        self.raw_resolution_lookup.insert(scope_id, unsafe {
            let last_index = self.scopes.len() - 1;
            &mut self.scopes.get_unchecked_mut(last_index).resolution as *mut _
        });
        resolution
    }

    pub fn exit_scope(&mut self) -> Scope {
        let scope = self.scopes.pop().expect("no scopes entered?");
        for (ident, local) in &scope.locals {
            if local.ty.has_any_unknown() {
                self.err_nonfatal(Error::CouldNotInferType(Spanned(*ident, local.def_span)))
            }
            if ident.to_string().chars().next() != Some('_') && !local.used {
                self.warnings.push(Warning::UnusedVariable(Spanned(
                    ident.to_string(),
                    local.def_span,
                )))
            }
            if !local.mutated && let Some(mutable) = local.mutable {
                self.warnings.push(Warning::UnusedMut(Spanned(ident.to_string(), local.def_span), mutable))
            }
        }
        scope
    }

    pub fn mark_scopes(&mut self, f: impl Fn(&Scope) -> bool, action: &ExitAction) -> Option<(usize, ScopeId, ScopeKind)> {
        let target_scope = self.scopes
            .iter()
            .rev()
            .enumerate()
            .find_map(|(i, scope)| f(scope).then_some((i, scope.id, scope.kind)));

        let len = self.scopes.len();
        if let Some((i, ..)) = target_scope {
            (1..=i + 1).for_each(|i| self.scopes[len - i].exited = Some(action.clone()));
        }
        self.exit_scope();
        target_scope
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

    fn lower_local(&mut self, ident: &Spanned<Ident>) -> Result<&mut Local> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(local) = scope.locals.get_mut(ident.value()) {
                local.used = true;
                return Ok(local);
            }
        }
        Err(Error::UnresolvedIdentifier(ident.map(|i| i.to_string())))
    }

    fn lower_ident_binding(
        &mut self,
        ident: &Spanned<Ident>,
        args: &Option<Spanned<Vec<crate::Ty>>>,
    ) -> Result<Binding> {
        if let Some(args) = args {
            // TODO: function pointers, first-class types
            return Err(Error::ExplicitTypeArgumentsNotAllowed(args.span()));
        }

        // Search for the local variable
        if let Ok(local) = self.lower_local(ident) {
            return Ok(Binding {
                def_span: local.def_span,
                ty: local.ty.clone(),
                mutable: local.mutable,
            });
        }

        // Does a constant with this name exist?
        let item = ItemId(self.scope().module_id, *ident.value());
        if let Some(cnst) = self.hir.consts.get(&item) {
            return Ok(Binding {
                def_span: cnst.name.span(),
                ty: self.lower_hir_ty(cnst.ty.clone()),
                mutable: None,
            });
        }

        Err(Error::UnresolvedIdentifier(ident.map(|i| i.to_string())))
    }

    #[inline]
    fn lower_exit_in_context(&mut self, scope_id: ScopeId) -> Result<Ty> {
        let label = self.hir.scopes.get(&scope_id).unwrap().label.clone();
        Ok(
            match self.lower_scope(scope_id, ScopeKind::Block, Vec::new())? {
                // Does it exit *just* itself? If so, return the type of the expression
                ExitAction::FromBlock(None, ty, _) => ty,
                ExitAction::FromBlock(Some(lbl), ty, _)
                    if label.is_some_and(|label| label.0 == lbl.0) =>
                {
                    ty
                }
                // Otherwise, return an exit type
                action => Ty::Exit(Box::new(action)),
            },
        )
    }

    /// Lowers an expression.
    pub fn lower_expr(&mut self, expr: Spanned<Expr>) -> Result<Spanned<TypedExpr>> {
        let Spanned(expr, span) = expr;

        let expr = match expr {
            Expr::Literal(lit) => {
                let ty = Self::lower_literal(&lit);
                TypedExpr(typed::Expr::Literal(lit), ty)
            }
            Expr::Ident(ident, args) => {
                let ty = self.lower_ident_binding(&ident, &args)?.ty;
                TypedExpr(
                    typed::Expr::Ident(
                        ident,
                        args.map(|args| {
                            args.map(|args| {
                                args.into_iter().map(|ty| self.lower_hir_ty(ty)).collect()
                            })
                        }),
                    ),
                    ty,
                )
            }
            Expr::Tuple(exprs) => {
                let exprs = exprs
                    .into_iter()
                    .map(|expr| self.lower_expr(expr))
                    .collect::<Result<Vec<_>>>()?;
                let tys = exprs.iter().map(|expr| expr.value().1.clone()).collect();

                TypedExpr(typed::Expr::Tuple(exprs), Ty::Tuple(tys))
            }
            Expr::Cast(expr, ty) => TypedExpr(
                typed::Expr::Cast(
                    Box::new(self.lower_expr(*expr)?),
                    self.lower_hir_ty(ty.clone()),
                ),
                self.lower_hir_ty(ty),
            ),
            Expr::Block(scope_id) => TypedExpr(
                typed::Expr::Block(scope_id),
                self.lower_exit_in_context(scope_id)?,
            ),
            Expr::If(cond, left, right) => {
                let cond = self.lower_expr(*cond)?;
                match cond.value().1.clone() {
                    // If the type is not known, constrain it to bool, this will be checked
                    // one more time at the typeck stage.
                    ty @ Ty::Unknown(_) => {
                        self.table
                            .constraints
                            .push_back(Constraint(ty, Ty::Primitive(PrimitiveTy::Bool)));
                    }
                    // If the type is known, check that it is bool
                    Ty::Primitive(PrimitiveTy::Bool) => (),
                    // Otherwise, error
                    ty => {
                        return Err(Error::ConditionNotBool(Spanned(ty.into(), cond.span())));
                    }
                };
                // "Evenness" check will be done at the typeck stage
                let left_ty = self.lower_exit_in_context(left)?;
                let ty = match right
                    .map(|right| self.lower_exit_in_context(right))
                    .transpose()?
                {
                    Some(right_ty) => {
                        #[allow(unused_parens)]
                        if !left_ty.known()
                            || matches!(
                                left_ty.relation_to(&right_ty),
                                (Relation::Eq | Relation::Super)
                            )
                        {
                            left_ty
                        } else {
                            // Again, if left_ty != right_ty, check at the typeck stage.
                            // The types may still be incompatible, but we don't know that yet.
                            right_ty
                        }
                    }
                    None => Ty::Primitive(PrimitiveTy::Void),
                };
                TypedExpr(typed::Expr::If(Box::new(cond), left, right), ty)
            }
            Expr::Assign(target, value) => {
                let lowered = self.lower_expr(*value)?;
                let ty = lowered.value().1.clone();

                let mut bindings = Vec::new();
                Self::bind_pattern_to_ty(&target, ty.clone(), Some(&lowered), &mut bindings)?;
                for (ident, binding) in bindings {
                    // SAFETY: the local should live as long as &mut self
                    let target = unsafe {
                        let target =
                            self.lower_local(&Spanned(ident, binding.def_span))? as *mut Local;
                        (*target).mutated = true;

                        // Unify and update
                        self.table
                            .constraints
                            .push_back(Constraint(binding.ty.clone(), (*target).ty.clone()));

                        let target = &mut *target;
                        if let Some(conflict) = self.table.unify_all() {
                            self.err_nonfatal(Error::TypeConflict {
                                constraint: conflict,
                                expected: (target.ty.clone().into(), None),
                                actual: lowered.as_ref().map(|expr| expr.1.clone().into()),
                            });
                        }

                        target.ty.apply(&self.table.substitutions);
                        Binding {
                            def_span: target.def_span,
                            ty: target.ty.clone(),
                            mutable: target.mutable,
                        }
                    };
                    if target.mutable.is_none() {
                        self.err_nonfatal(Error::ReassignmentToImmutable(
                            binding.def_span,
                            Spanned(ident, target.def_span),
                        ));
                    }
                }

                TypedExpr(typed::Expr::Assign(target, Box::new(lowered)), ty)
            }
            Expr::Loop(scope_id) => {
                let label = self.hir.scopes.get(&scope_id).unwrap().label.clone();
                let ty = match self.lower_scope(scope_id, ScopeKind::Loop, Vec::new())? {
                    // Are we exiting from just the loop?
                    ExitAction::FromBlock(Some(lbl), ty, _) if label.is_some_and(|l| l == lbl) => {
                        ty
                    }
                    ExitAction::FromNearestLoop(ty, _) => ty,
                    // Otherwise we are exiting further out from the loop.
                    r => Ty::Exit(Box::new(r)),
                };
                TypedExpr(typed::Expr::Loop(scope_id), ty)
            }
            _ => unimplemented!(),
        };
        Ok(Spanned(expr, span))
    }

    fn bind_pattern_to_ty(
        pat: &Spanned<Pattern>,
        ty: Ty,
        expr: Option<&Spanned<TypedExpr>>,
        bindings: &mut Vec<(Ident, Binding)>,
    ) -> Result<()> {
        match (pat.value(), ty) {
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
    pub fn lower_node(
        &mut self,
        node: Spanned<Node>,
    ) -> Result<(Spanned<Node<InferMetadata>>, Option<ExitAction>)> {
        let Spanned(node, span) = node;

        let mut action = None;
        let node = match node {
            Node::Expr(expr) => {
                let expr = self.lower_expr(expr)?;
                if let Ty::Exit(exit_action) = &expr.value().1 {
                    action = Some(exit_action.as_ref().clone());
                }
                Node::Expr(expr)
            }
            Node::Let {
                pat,
                ty,
                ty_span,
                value,
            } => {
                let mut bindings = Vec::new();
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
                for (ident, binding) in bindings {
                    self.scope_mut()
                        .locals
                        .insert(ident, Local::from_binding(binding));
                }
                node
            }
            Node::ImplicitReturn(expr) => {
                let expr = self.lower_expr(expr)?;
                let exit_ty = expr.as_ref().map(|expr| expr.1.clone());
                action = Some(ExitAction::FromBlock(None, exit_ty.into_value(), span));
                Node::ImplicitReturn(expr)
            }
            Node::Continue(label) => {
                action = Some(ExitAction::ContinueLoop(label, span));
                Node::Continue(label)
            }
            Node::Break(label, value) => {
                let value = value.map(|value| self.lower_expr(value)).transpose()?;
                let exit_ty = value
                    .as_ref()
                    .map(|value| value.value().1.clone())
                    .unwrap_or(Ty::Primitive(PrimitiveTy::Void));

                action = Some(match label {
                    Some(label) => ExitAction::FromBlock(Some(label), exit_ty, span),
                    None => ExitAction::FromNearestLoop(exit_ty, span),
                });
                Node::Break(label, value)
            }
            Node::Return(value) => {
                let value = value.map(|value| self.lower_expr(value)).transpose()?;
                let exit_ty = value
                    .as_ref()
                    .map(|value| value.value().1.clone())
                    .unwrap_or(Ty::Primitive(PrimitiveTy::Void));
                action = Some(ExitAction::FromFunc(exit_ty, span));

                Node::Return(value)
            }
            _ => todo!(),
        };
        Ok((Spanned(node, span), action))
    }

    /// Unifies the resolution type of a scope with a given type.
    #[inline]
    fn unify_scope(&mut self, scope_id: ScopeId, ty: Ty, span: Span) {
        let Some(tgt) = self.raw_resolution_lookup.get(&scope_id) else { return };
        // SAFETY: raw resolution guarantees all pointers are self-referential
        // and non-dropping, so we can safely dereference them as long as `self` is alive
        let tgt = unsafe { &mut **tgt };
        // unify the return type of the function with the type of the return expression
        self.table
            .constraints
            .push_back(Constraint(ty.clone(), tgt.clone()));
        if let Some(conflict) = self.table.unify_all() {
            self.err_nonfatal(Error::TypeConflict {
                expected: (tgt.clone().into(), None),
                actual: Spanned(ty.into(), span),
                constraint: conflict,
            });
        }
        tgt.apply(&self.table.substitutions);
    }

    /// Runs type inference over a scope.
    /// *Moves* the scope into the new typed HIR and returns a tuple
    /// (ScopeId, exit type of the scope).
    ///
    /// The exit type of the scope is how the scope itself exits, not how it exits its outer scope.
    /// `FromBlock(None, ..)` means the scope exits from itself.
    pub fn lower_scope(
        &mut self,
        scope_id: ScopeId,
        kind: ScopeKind,
        ty_params: Vec<TyParam>,
    ) -> Result<ExitAction> {
        let mut scope = self.hir.scopes.remove(&scope_id).expect("scope not found");
        let label = std::mem::replace(&mut scope.label, None);
        self.enter_scope(scope_id, scope.module_id, kind, ty_params, label);

        let mut exit_action = None;
        let full_span = scope.children.span();
        let mut lowered = Vec::with_capacity(scope.children.value().len());
        let mut children = scope.children.value_mut().drain(..);

        while let Some(node) = children.next() {
            let (node, mut et) = self.lower_node(node)?;
            lowered.push(node);

            if let Some(action) = &self.scope().exited {
                et = Some(action.clone());
            }
            if let Some(et) = et {
                match et.clone() {
                    ExitAction::FromFunc(ty, span) => {
                        match self.mark_scopes(|scope| scope.kind == ScopeKind::Func, &et) {
                            Some((_, scope, _)) => self.unify_scope(scope, ty, span),
                            None => self.err_nonfatal(Error::InvalidReturn(span)),
                        }
                    }
                    ExitAction::FromBlock(Some(ref label), ty, span) => {
                        match self.mark_scopes(|scope| scope.label.is_some_and(|lbl| lbl.0 == label.0), &et) {
                            Some((_, scope, _)) => self.unify_scope(scope, ty, span),
                            // TODO: label not found can be non-fatal but it currently will emit multiple times
                            None => return Err(Error::LabelNotFound(label.clone())),
                        }
                    }
                    ExitAction::FromNearestLoop(ty, span) => {
                        match self.mark_scopes(|scope| scope.kind == ScopeKind::Loop, &et) {
                            Some((_, scope, _)) => self.unify_scope(scope, ty, span),
                            None => self.err_nonfatal(Error::InvalidBreak(span, None)),
                        }
                    }
                    ExitAction::ContinueLoop(None, span) => {
                        if self
                            .mark_scopes(|scope| scope.kind == ScopeKind::Loop, &et)
                            .is_none()
                        {
                            self.err_nonfatal(Error::InvalidBreak(span, None));
                        }
                    }
                    ExitAction::ContinueLoop(Some(ref label), span) => {
                        match self
                            .mark_scopes(|scope| scope.label.is_some_and(|lbl| lbl.0 == label.0), &et)
                        {
                            Some((_, _, kind)) if kind == ScopeKind::Loop => {}
                            Some(_) => {
                                self.err_nonfatal(Error::InvalidBreak(span, Some(label.clone())))
                            }
                            None => return Err(Error::LabelNotFound(label.clone())),
                        }
                    }
                    ExitAction::FromBlock(None, ty, span) => {
                        let scope = self.exit_scope();
                        self.unify_scope(scope.id, ty, span);
                    }
                    ExitAction::NeverReturn => {}
                }
                exit_action = Some(et);
                break;
            }
        }
        self.thir.scopes.insert(
            scope_id,
            crate::Scope {
                module_id: scope.module_id,
                label: scope.label,
                children: lowered.spanned(full_span),
            },
        );
        let exit_action = exit_action.unwrap_or_else(|| {
            self.unify_scope(scope_id, Ty::Primitive(PrimitiveTy::Void), full_span);
            match kind {
                ScopeKind::Loop => ExitAction::NeverReturn,
                _ => ExitAction::FromBlock(None, Ty::Primitive(PrimitiveTy::Void), full_span),
            }
        });
        // If there are any remaining nodes, they are unreachable
        if let Some(first) = children.next() {
            self.warnings.push(Warning::UnreachableCode(
                exit_action.span(),
                first
                    .span()
                    .merge_opt(children.last().map(|last| last.span())),
            ));
        }
        Ok(exit_action)
    }

    /// Lowers a module into a typed HIR module.
    pub fn lower_module(&mut self, module_id: ModuleId) -> Result<()> {
        let scope_id = *self.hir.modules.get(&module_id).unwrap();

        // TODO: exit action can be non-standard in things like REPLs
        match self.lower_scope(scope_id, ScopeKind::Block, Vec::new())? {
            _ => (), // TODO: check exit type is void
        }

        self.thir.modules.insert(module_id, scope_id);
        Ok(())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ScopeKind {
    Block,
    Loop,
    Func,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExitAction {
    /// return or implicit return from nearest function
    FromFunc(Ty, Span),
    /// break with label or return from block
    FromBlock(Option<Spanned<Ident>>, Ty, Span),
    /// continue
    ContinueLoop(Option<Spanned<Ident>>, Span),
    /// break w/o label
    FromNearestLoop(Ty, Span),
    /// never return
    NeverReturn,
}

impl ExitAction {
    pub const fn span(&self) -> Option<Span> {
        match self {
            Self::FromFunc(_, span)
            | Self::FromBlock(_, _, span)
            | Self::ContinueLoop(_, span)
            | Self::FromNearestLoop(_, span) => Some(*span),
            Self::NeverReturn => None,
        }
    }
}
