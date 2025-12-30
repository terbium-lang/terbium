use crate::{
    Expr, FloatWidth, Func, FuncHeader, FuncParam, Hir, Ident, IntSign, IntWidth, ItemId, ItemKind,
    Literal, LogicalOp, LookupId, Metadata, ModuleId, Node, Pattern, PrimitiveTy, ScopeId, TyParam,
    error::{Error, Result},
    typed::{
        self, BoolIntrinsic, Constraint, InvalidTypeCause, LocalEnv, Relation, Ty, TypedExpr,
        UnificationTable,
    },
    warning::Warning,
};
use common::span::{Span, Spanned, SpannedExt};
use std::{borrow::Cow, collections::HashMap};

impl UnificationTable {
    /// Create a fresh type variable.
    pub fn new_unknown(&mut self) -> Ty {
        Ty::from_ty(crate::Ty::Unknown, self)
    }

    /// Substitute the unknown constraint a => b. Return failed constraint on conflict.
    pub fn substitute(&mut self, a: usize, b: Ty) {
        // Add a => b to our substitution
        self.substitutions[a] = b.clone();
        // Replace further unifications of a with b
        for Constraint(x, y) in self.constraints.iter_mut() {
            x.substitute(a, b.clone());
            y.substitute(a, b.clone());
        }
    }

    /// Unify the constraint a == b. Return failed constraint on conflict.
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BindingKind {
    Var,
    Param,
    Func(LookupId),
}

pub struct Binding {
    pub def_span: Span,
    pub ty: Ty,
    pub mutable: Option<Span>,
    pub initialized: bool,
    pub kind: BindingKind,
}

#[derive(Clone, Debug)]
pub struct Local {
    pub def_span: Span,
    pub ty: Ty,
    pub mutable: Option<Span>,
    // analysis checks
    pub initialized: bool,
    pub used: bool,
    pub mutated: bool,
}

impl Local {
    #[inline]
    pub const fn new(def_span: Span, ty: Ty, mutable: Option<Span>, initialized: bool) -> Self {
        Self {
            def_span,
            ty,
            mutable,
            initialized,
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
            initialized: binding.initialized,
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
    ty_params: Vec<TyParam<Ty>>,
    locals: HashMap<(Ident, LocalEnv), Local>,
    funcs: HashMap<ItemId, (LookupId, FuncHeader<InferMetadata>)>,
    label: Option<Spanned<Ident>>,
    exited: Option<ExitAction>,
}

#[derive(Clone, Debug)]
pub struct InferMetadata;
impl Metadata for InferMetadata {
    type Expr = TypedExpr;
    type Ty = Ty;
}

/// Type resolution of a scope
pub type ScopeResolution = (Ty, Option<Span>);

/// Lowers types of expressions and performs type inference and one pass of typeck.
#[derive(Debug)]
pub struct TypeLowerer {
    local_env: LocalEnv,
    pub(crate) scopes: Vec<Scope>,
    pub(crate) table: UnificationTable,
    /// Scope resolution lookup.
    pub(crate) resolution_lookup: HashMap<ScopeId, ScopeResolution>,
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
        let mut thir = Hir::default();
        thir.func_ids = hir.func_ids.clone();
        Self {
            local_env: LocalEnv::Standard,
            scopes: Vec::new(),
            table: UnificationTable::default(),
            resolution_lookup: HashMap::new(),
            hir,
            thir,
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
        bindings: Vec<(Ident, Binding)>,
        ty_params: Vec<TyParam<Ty>>,
        label: Option<Spanned<Ident>>,
        resolution: Option<ScopeResolution>,
    ) {
        let resolution = resolution.unwrap_or_else(|| (self.table.new_unknown(), None));
        let scope = Scope {
            id: scope_id,
            module_id: module,
            kind,
            ty_params,
            locals: bindings
                .into_iter()
                .map(|(ident, binding)| ((ident, self.local_env), Local::from_binding(binding)))
                .collect(),
            funcs: HashMap::new(),
            label,
            exited: None,
        };
        self.scopes.push(scope);
        self.resolution_lookup.insert(scope_id, resolution);
    }

    pub fn exit_scope_if_exists(&mut self) -> Option<Scope> {
        let mut scope = self.scopes.pop()?;

        for ((ident, env), local) in &mut scope.locals {
            // Only check locals in the current environment
            if *env != self.local_env {
                continue;
            }
            // Apply substitutions to the type one final time
            local.ty.apply(&mut self.table.substitutions);

            if ident.to_string().chars().next() != Some('_') && !local.used {
                self.warnings.push(Warning::UnusedVariable(Spanned(
                    ident.to_string(),
                    local.def_span,
                )))
            }
            if !local.mutated
                && let Some(mutable) = local.mutable
            {
                self.warnings.push(Warning::UnusedMut(
                    Spanned(ident.to_string(), local.def_span),
                    mutable,
                ))
            }
        }
        Some(scope)
    }

    pub fn exit_scope(&mut self) -> Scope {
        self.exit_scope_if_exists().expect("no scopes entered?")
    }

    pub fn mark_scopes(
        &mut self,
        f: impl Fn(&Scope) -> bool,
        action: Option<&ExitAction>,
    ) -> Option<(usize, ScopeId, ScopeKind)> {
        let target_scope = self
            .scopes
            .iter()
            .rev()
            .enumerate()
            .find_map(|(i, scope)| f(scope).then_some((i, scope.id, scope.kind)));

        let len = self.scopes.len();
        if let Some(action) = action
            && let Some((i, ..)) = target_scope
        {
            (1..=i + 1).for_each(|i| {
                let _ = self.scopes[len - i].exited.insert(action.clone());
            });
        }
        self.exit_scope();
        target_scope
    }

    pub fn lower_literal(lit: &Literal) -> Ty {
        match lit {
            Literal::Void => Ty::VOID,
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
            Literal::String(_) => Ty::Primitive(PrimitiveTy::String),
            Literal::Bytes(_) => Ty::Array(
                Box::new(Ty::Primitive(PrimitiveTy::Int(
                    IntSign::Unsigned,
                    IntWidth::Int8,
                ))),
                None,
            ),
        }
    }

    #[inline]
    fn lower_hir_ty(&mut self, ty: crate::Ty) -> Ty {
        Ty::from_ty(ty, &mut self.table)
    }

    fn lower_local(&mut self, ident: &Spanned<Ident>) -> Result<&mut Local> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(local) = scope.locals.get_mut(&(*ident.value(), self.local_env)) {
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
        // Search for the local variable
        if let Ok(local) = self.lower_local(ident) {
            if let Some(args) = args {
                return Err(Error::ExplicitTypeArgumentsNotAllowed(args.span()));
            }
            return Ok(Binding {
                def_span: local.def_span,
                ty: local.ty.clone(),
                mutable: local.mutable,
                initialized: local.initialized,
                kind: BindingKind::Var,
            });
        }

        let item = ItemId(self.scope().module_id, *ident.value());
        // Does an item with this name exist?
        for scope in self.scopes.iter().rev() {
            if let Some(cnst) = self.hir.scopes.get(&scope.id).and_then(|scope| {
                scope
                    .items
                    .get(&(ItemKind::Const, item))
                    .and_then(|id| self.hir.consts.get(id))
            }) {
                if let Some(args) = args {
                    return Err(Error::ExplicitTypeArgumentsNotAllowed(args.span()));
                }
                return Ok(Binding {
                    def_span: cnst.name.span(),
                    ty: self.lower_hir_ty(cnst.ty.clone()),
                    mutable: None,
                    initialized: true,
                    kind: BindingKind::Var,
                });
            }

            if let Some((id, header)) = scope.funcs.get(&item).cloned() {
                let ty_args = args.as_ref().map(|args| {
                    args.value()
                        .iter()
                        .cloned()
                        .map(|ty| self.lower_hir_ty(ty))
                        .collect::<Vec<_>>()
                });
                if let Some(ty_args) = &ty_args {
                    if ty_args.len() != header.ty_params.len() {
                        return Err(Error::IncorrectTypeArgumentCount {
                            span: args.as_ref().expect("checked above").span(),
                            ty: header.name.as_ref().map(ToString::to_string),
                            expected: header.ty_params.len(),
                            actual: ty_args.len(),
                        });
                    }
                }
                let subs = ty_args
                    .as_ref()
                    .map(|ty_args| {
                        header
                            .ty_params
                            .iter()
                            .map(|param| param.name)
                            .zip(ty_args.iter().cloned())
                            .collect::<HashMap<_, _>>()
                    })
                    .unwrap_or_default();
                let param_tys = header
                    .params
                    .iter()
                    .chain(header.kw_params.iter())
                    .map(|param| param.ty.substitute_generics(&subs))
                    .collect::<Vec<_>>();
                let ret_ty = header.ret_ty.substitute_generics(&subs);
                return Ok(Binding {
                    def_span: header.name.span(),
                    ty: Ty::Func(param_tys, Box::new(ret_ty)),
                    mutable: None,
                    initialized: true,
                    kind: BindingKind::Func(id),
                });
            }
        }

        Err(Error::UnresolvedIdentifier(ident.map(|i| i.to_string())))
    }

    #[inline]
    fn lower_exit_in_context(&mut self, scope_id: ScopeId, divergent: bool) -> Result<Ty> {
        let label = self.hir.scopes.get(&scope_id).unwrap().label.clone();
        Ok(
            match self.lower_scope(
                scope_id,
                ScopeKind::Block,
                Vec::new(),
                Vec::new(),
                divergent,
                None,
            )? {
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
                let binding = self.lower_ident_binding(&ident, &args)?;
                if !binding.initialized {
                    self.err_nonfatal(Error::UseOfUninitializedVariable(
                        ident.span(),
                        Spanned(ident.0, binding.def_span),
                    ));
                }
                let args = args.map(|args| {
                    args.map(|args| args.into_iter().map(|ty| self.lower_hir_ty(ty)).collect())
                });

                match binding.kind {
                    BindingKind::Var | BindingKind::Param => {
                        TypedExpr(typed::Expr::Local(ident, args, self.local_env), binding.ty)
                    }
                    BindingKind::Func(id) => {
                        TypedExpr(typed::Expr::Func(ident, args, id), binding.ty)
                    }
                }
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
                self.lower_exit_in_context(scope_id, true)?,
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
                let left_ty = self.lower_exit_in_context(left, false)?;
                let ty = match right
                    .map(|right| self.lower_exit_in_context(right, true))
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
                    None => Ty::VOID,
                };
                TypedExpr(typed::Expr::If(Box::new(cond), left, right), ty)
            }
            Expr::CallLogicalOp(op, lhs, rhs) => {
                // lower to an intrinsic and desugar from there in typeck stage
                let lhs = Box::new(self.lower_expr(*lhs)?);
                let rhs = Box::new(self.lower_expr(*rhs)?);

                let ty = {
                    let left_ty = &lhs.value().1;
                    let right_ty = &rhs.value().1;

                    #[allow(unused_parens)]
                    if !left_ty.known()
                        || matches!(
                            left_ty.relation_to(&right_ty),
                            (Relation::Eq | Relation::Super)
                        )
                    {
                        left_ty.clone()
                    } else {
                        // Like `If`, if left_ty != right_ty, check at the typeck stage.
                        right_ty.clone()
                    }
                };
                let intrinsic = match op.value() {
                    LogicalOp::And => BoolIntrinsic::And(lhs, rhs),
                    LogicalOp::Or => BoolIntrinsic::Or(lhs, rhs),
                };
                TypedExpr(typed::Expr::BoolIntrinsic(intrinsic), ty)
            }
            Expr::Assign(target, value) => {
                let lowered = self.lower_expr(*value)?;
                let ty = lowered.value().1.clone();

                let mut bindings = Vec::new();
                self.bind_pattern_to_ty(&target, ty.clone(), Some(&lowered), &mut bindings)?;
                for (ident, binding) in bindings {
                    // SAFETY: the local should live as long as &mut self
                    let target = unsafe {
                        let target =
                            self.lower_local(&Spanned(ident, binding.def_span))? as *mut Local;

                        // Unify and update
                        self.table
                            .constraints
                            .push_back(Constraint(binding.ty.clone(), (*target).ty.clone()));

                        let target = &mut *target;
                        if target.initialized {
                            target.mutated = true;
                        }
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
                            initialized: std::mem::replace(&mut target.initialized, true),
                            kind: BindingKind::Var,
                        }
                    };
                    if target.initialized && target.mutable.is_none() {
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
                let ty = match self.lower_scope(
                    scope_id,
                    ScopeKind::Loop,
                    Vec::new(),
                    Vec::new(),
                    true,
                    None,
                )? {
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
            Expr::CallOp(op, lhs, rhs) => TypedExpr(
                typed::Expr::CallOp(
                    op,
                    Box::new(self.lower_expr(*lhs)?),
                    rhs.into_iter()
                        .map(|expr| self.lower_expr(expr))
                        .collect::<Result<Vec<_>>>()?,
                ),
                self.table.new_unknown(),
            ),
            Expr::Call {
                callee,
                args,
                kwargs,
            } => {
                let callee = self.lower_expr(*callee)?;
                let callee_ty = callee.value().1.clone();

                let mut args = args
                    .into_iter()
                    .map(|expr| self.lower_expr(expr))
                    .collect::<Result<Vec<_>>>()?;
                let mut kwargs = kwargs
                    .into_iter()
                    .map(|(name, expr)| Ok((name, self.lower_expr(expr)?)))
                    .collect::<Result<Vec<_>>>()?;

                let (expr, return_ty) = match callee.value().0.clone() {
                    typed::Expr::Func(_, ty_args, item) => {
                        let header = self.thir.funcs[&item].header.clone();
                        let ty_args = ty_args.map(|args| args.into_value());
                        if let Some(ty_args) = &ty_args {
                            if ty_args.len() != header.ty_params.len() {
                                self.err_nonfatal(Error::IncorrectTypeArgumentCount {
                                    span,
                                    ty: header.name.as_ref().map(ToString::to_string),
                                    expected: header.ty_params.len(),
                                    actual: ty_args.len(),
                                });
                            }
                        }
                        let subs = ty_args
                            .as_ref()
                            .map(|ty_args| {
                                header
                                    .ty_params
                                    .iter()
                                    .map(|param| param.name)
                                    .zip(ty_args.iter().cloned())
                                    .collect::<HashMap<_, _>>()
                            })
                            .unwrap_or_default();
                        let param_tys = header
                            .params
                            .iter()
                            .map(|param| param.ty.substitute_generics(&subs))
                            .collect::<Vec<_>>();
                        let kw_param_tys = header
                            .kw_params
                            .iter()
                            .map(|param| param.ty.substitute_generics(&subs))
                            .collect::<Vec<_>>();
                        let ret_ty = header.ret_ty.substitute_generics(&subs);

                        if args.len() != param_tys.len() {
                            self.err_nonfatal(Error::IncorrectArgumentCount {
                                span,
                                expected: param_tys.len(),
                                actual: args.len(),
                            });
                        }

                        let mut kw_map = HashMap::new();
                        for (name, arg) in kwargs.drain(..) {
                            if kw_map.insert(name, arg).is_some() {
                                self.err_nonfatal(Error::DuplicateKeywordArgument(span, name));
                            }
                        }

                        let mut ordered_args = args;
                        for param in &header.kw_params {
                            let Some(name) = param.pat.value().ident() else {
                                continue;
                            };
                            if let Some(arg) = kw_map.remove(&name) {
                                ordered_args.push(arg);
                            } else {
                                self.err_nonfatal(Error::MissingKeywordArgument(span, name));
                            }
                        }
                        for (name, _arg) in kw_map {
                            self.err_nonfatal(Error::UnknownKeywordArgument(span, name));
                        }

                        let expected = param_tys.len() + kw_param_tys.len();
                        if ordered_args.len() != expected {
                            self.err_nonfatal(Error::IncorrectArgumentCount {
                                span,
                                expected,
                                actual: ordered_args.len(),
                            });
                        }

                        for (arg, ty) in ordered_args
                            .iter()
                            .zip(param_tys.into_iter().chain(kw_param_tys))
                        {
                            self.table
                                .constraints
                                .push_back(Constraint(arg.value().1.clone(), ty.clone()));

                            if let Some(conflict) = self.table.unify_all() {
                                self.err_nonfatal(Error::TypeConflict {
                                    expected: (ty.into(), None),
                                    actual: arg.as_ref().map(|expr| expr.1.clone().into()),
                                    constraint: conflict,
                                });
                            }
                        }

                        for arg in &mut ordered_args {
                            arg.value_mut().1.apply(&self.table.substitutions);
                        }

                        (
                            typed::Expr::CallFunc {
                                parent: None,
                                func: item,
                                args: ordered_args,
                                kwargs: Vec::new(),
                                ty_args,
                            },
                            ret_ty,
                        )
                    }
                    _ => {
                        if !kwargs.is_empty() {
                            self.err_nonfatal(Error::UnknownKeywordArgument(span, kwargs[0].0));
                        }

                        let (arg_tys, mut return_ty) = match callee_ty {
                            Ty::Func(args, ret_ty) => (Some(args), *ret_ty),
                            Ty::Unknown(i) => {
                                let arg_tys = args
                                    .iter()
                                    .map(|expr| expr.value().1.clone())
                                    .collect::<Vec<_>>();
                                let return_ty = self.table.new_unknown();
                                self.table.substitute(
                                    i,
                                    Ty::Func(arg_tys.clone(), Box::new(return_ty.clone())),
                                );
                                (Some(arg_tys), return_ty)
                            }
                            _ => {
                                self.err_nonfatal(Error::NotCallable(
                                    span,
                                    Spanned(callee_ty.into(), callee.span()),
                                ));
                                (None, self.table.new_unknown())
                            }
                        };

                        if let Some(arg_tys) = arg_tys {
                            if args.len() != arg_tys.len() {
                                self.err_nonfatal(Error::IncorrectArgumentCount {
                                    span,
                                    expected: arg_tys.len(),
                                    actual: args.len(),
                                });
                            }

                            for (arg, ty) in args.iter().zip(arg_tys) {
                                self.table
                                    .constraints
                                    .push_back(Constraint(arg.value().1.clone(), ty.clone()));

                                if let Some(conflict) = self.table.unify_all() {
                                    self.err_nonfatal(Error::TypeConflict {
                                        expected: (ty.into(), None),
                                        actual: arg.as_ref().map(|expr| expr.1.clone().into()),
                                        constraint: conflict,
                                    });
                                }
                            }

                            for arg in &mut args {
                                arg.value_mut().1.apply(&self.table.substitutions);
                            }
                        }

                        return_ty.apply(&self.table.substitutions);

                        (typed::Expr::CallIndirect(Box::new(callee), args), return_ty)
                    }
                };
                TypedExpr(expr, return_ty)
            }
            _ => unimplemented!(),
        };
        Ok(Spanned(expr, span))
    }

    fn bind_pattern_to_ty(
        &mut self,
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
                        initialized: expr.is_some(),
                        kind: BindingKind::Var,
                    },
                ));
            }
            (Pattern::Tuple(pats), Ty::Tuple(tys)) => {
                if pats.len() != tys.len() {
                    return Err(pat_errors::tuple_len_mismatch(
                        pats.len().spanned(pat.span()),
                        tys.len(),
                        expr.map(|expr| expr.span()),
                    ));
                }
                // Can we destructure the tuple further?
                if let Some(Spanned(TypedExpr(typed::Expr::Tuple(exprs), _), _)) = expr {
                    debug_assert_eq!(pats.len(), exprs.len());
                    for ((pat, ty), expr) in pats.iter().zip(tys).zip(exprs) {
                        self.bind_pattern_to_ty(pat, ty, Some(expr), bindings)?;
                    }
                } else {
                    for (pat, ty) in pats.iter().zip(tys) {
                        self.bind_pattern_to_ty(pat, ty, expr, bindings)?;
                    }
                }
            }
            (Pattern::Tuple(pats), Ty::Struct(item_id, args)) => {
                if let Some(field_tys) = self.struct_field_types(item_id, &args) {
                    if pats.len() != field_tys.len() {
                        return Err(pat_errors::tuple_len_mismatch(
                            pats.len().spanned(pat.span()),
                            field_tys.len(),
                            expr.map(|expr| expr.span()),
                        ));
                    }
                    for (pat, ty) in pats.iter().zip(field_tys) {
                        self.bind_pattern_to_ty(pat, ty, expr, bindings)?;
                    }
                } else {
                    return Err(pat_errors::tuple_mismatch(
                        pat.span(),
                        Ty::Struct(item_id, args),
                        expr.map(|expr| expr.span()),
                    ));
                }
            }
            (Pattern::Tuple(_), ty) => {
                return Err(pat_errors::tuple_mismatch(
                    pat.span(),
                    ty,
                    expr.map(|expr| expr.span()),
                ));
            }
        }
        Ok(())
    }

    fn struct_field_types(&mut self, item: ItemId, args: &[Ty]) -> Option<Vec<Ty>> {
        let scope_id = self.hir.modules.get(&item.0)?;
        let scope = self.hir.scopes.get(scope_id)?;
        let id = scope.items.get(&(ItemKind::Struct, item))?;
        let sct = self.hir.structs.get(id)?.clone();

        let subs = sct
            .ty_params
            .iter()
            .map(|param| param.name)
            .zip(args.iter().cloned())
            .collect::<HashMap<_, _>>();

        Some(
            sct.fields
                .iter()
                .map(|field| {
                    let ty = self.lower_hir_ty(field.ty.clone());
                    ty.substitute_generics(&subs)
                })
                .collect(),
        )
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
                let (lower_ty, expr) = if let Some(value) = value {
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
                            expected: (lower_ty.clone(), ty_span),
                            actual: expr.as_ref().map(|expr| expr.1.clone().into()),
                            constraint: conflict,
                        });
                    }
                    (lower_ty, Some(expr))
                } else {
                    (self.lower_hir_ty(ty), None)
                };
                self.bind_pattern_to_ty(&pat, lower_ty.clone(), expr.as_ref(), &mut bindings)?;
                let node = Node::Let {
                    pat,
                    ty: lower_ty,
                    ty_span,
                    value: expr,
                };

                for (ident, binding) in bindings {
                    let local_env = self.local_env;
                    self.scope_mut()
                        .locals
                        .insert((ident, local_env), Local::from_binding(binding));
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
                    .unwrap_or(Ty::VOID);

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
                    .unwrap_or(Ty::VOID);
                action = Some(ExitAction::FromFunc(exit_ty, span));

                Node::Return(value)
            }
        };
        Ok((Spanned(node, span), action))
    }

    /// Unifies the resolution type of a scope with a given type.
    #[inline]
    fn unify_scope(&mut self, scope_id: ScopeId, ty: Ty, span: Span) {
        let Some((mut tgt, tgt_span)) = self.resolution_lookup.remove(&scope_id) else {
            return;
        };

        // unify the return type of the function with the type of the return expression
        self.table
            .constraints
            .push_back(Constraint(tgt.clone(), ty.clone()));

        if let Some(conflict) = self.table.unify_all() {
            self.err_nonfatal(Error::TypeConflict {
                expected: (tgt.clone().into(), tgt_span.clone()),
                actual: Spanned(ty.into(), span),
                constraint: conflict,
            });
        }
        tgt.apply(&self.table.substitutions);

        // insert the unified type back into the resolution lookup
        self.resolution_lookup.insert(scope_id, (tgt, tgt_span));
    }

    fn handle_exit(&mut self, et: &ExitAction, divergent: bool) -> Result<()> {
        let action = divergent.then_some(et);
        match et.clone() {
            ExitAction::FromFunc(ty, span) => {
                match self.mark_scopes(|scope| scope.kind == ScopeKind::Func, action) {
                    Some((_, scope, _)) => self.unify_scope(scope, ty, span),
                    None => self.err_nonfatal(Error::InvalidReturn(span)),
                }
            }
            ExitAction::FromBlock(Some(ref label), ty, span) => {
                match self.mark_scopes(
                    |scope| scope.label.is_some_and(|lbl| lbl.0 == label.0),
                    action,
                ) {
                    Some((_, scope, _)) => self.unify_scope(scope, ty, span),
                    // TODO: label not found can be non-fatal but it currently will emit multiple times
                    None => return Err(Error::LabelNotFound(label.clone())),
                }
            }
            ExitAction::FromNearestLoop(ty, span) => {
                match self.mark_scopes(|scope| scope.kind == ScopeKind::Loop, action) {
                    Some((_, scope, _)) => self.unify_scope(scope, ty, span),
                    None => self.err_nonfatal(Error::InvalidBreak(span, None)),
                }
            }
            ExitAction::ContinueLoop(None, span) => {
                if self
                    .mark_scopes(|scope| scope.kind == ScopeKind::Loop, action)
                    .is_none()
                {
                    self.err_nonfatal(Error::InvalidBreak(span, None));
                }
            }
            ExitAction::ContinueLoop(Some(ref label), span) => {
                match self.mark_scopes(
                    |scope| scope.label.is_some_and(|lbl| lbl.0 == label.0),
                    action,
                ) {
                    Some((_, _, kind)) if kind == ScopeKind::Loop => {}
                    Some(_) => self.err_nonfatal(Error::InvalidBreak(span, Some(label.clone()))),
                    None => return Err(Error::LabelNotFound(label.clone())),
                }
            }
            ExitAction::FromBlock(None, ty, span) => {
                let scope = self.exit_scope();
                self.unify_scope(scope.id, ty, span);
            }
            ExitAction::NeverReturn => {}
        }
        Ok(())
    }

    /// Runs type inference over a function.
    pub fn lower_func_header(&mut self, header: FuncHeader) -> Result<FuncHeader<InferMetadata>> {
        Ok(FuncHeader {
            name: header.name,
            ty_params: header
                .ty_params
                .into_iter()
                .map(|param| {
                    Ok(TyParam {
                        name: param.name,
                        infer: param.infer,
                        bound: param
                            .bound
                            .map(|bound| self.lower_hir_ty(*bound))
                            .map(Box::new),
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            params: header
                .params
                .into_iter()
                .map(|param| {
                    Ok(FuncParam {
                        pat: param.pat,
                        ty: self.lower_hir_ty(param.ty),
                        default: param
                            .default
                            .map(|expr| self.lower_expr(expr))
                            .transpose()?,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            kw_params: header
                .kw_params
                .into_iter()
                .map(|param| {
                    Ok(FuncParam {
                        pat: param.pat,
                        ty: self.lower_hir_ty(param.ty),
                        default: param
                            .default
                            .map(|expr| self.lower_expr(expr))
                            .transpose()?,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            ret_ty: self.lower_hir_ty(header.ret_ty),
            ret_ty_span: header.ret_ty_span,
        })
    }

    pub fn lower_func_scope(&mut self, func: &Func<InferMetadata>) -> Result<Ty> {
        let mut bindings = Vec::new();
        for param in func
            .header
            .params
            .iter()
            .chain(func.header.kw_params.iter())
        {
            if let Err(why) = flatten_param(&param.pat, param.ty.clone(), &mut bindings) {
                self.err_nonfatal(why);
            }
        }
        let Some(body) = func.body else {
            return Ok(func.header.ret_ty.clone());
        };
        self.lower_scope(
            body,
            ScopeKind::Func,
            bindings,
            func.header.ty_params.clone(),
            true,
            Some((func.header.ret_ty.clone(), func.header.ret_ty_span)),
        )?;
        Ok(self.resolution_lookup[&body].0.clone())
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
        bindings: Vec<(Ident, Binding)>,
        ty_params: Vec<TyParam<Ty>>,
        divergent: bool,
        resolution: Option<ScopeResolution>,
    ) -> Result<ExitAction> {
        let mut scope = self.hir.scopes.remove(&scope_id).expect("scope not found");
        let label = std::mem::replace(&mut scope.label, None);
        self.enter_scope(
            scope_id,
            scope.module_id,
            kind,
            bindings,
            ty_params,
            label,
            resolution,
        );

        let mut lowering = Vec::with_capacity(scope.items.len());
        let mut items = HashMap::with_capacity(scope.items.len());
        let mut func_aliases: HashMap<LookupId, Vec<ItemId>> =
            HashMap::with_capacity(scope.items.len());

        for ((kind, name), id) in &scope.items {
            if *kind != ItemKind::Func {
                continue;
            }
            func_aliases.entry(*id).or_default().push(*name);
        }

        for (id, aliases) in func_aliases {
            let primary = self.hir.func_ids.get(&id).copied().unwrap_or_else(|| {
                aliases
                    .iter()
                    .copied()
                    .find(|item| item.0 != scope.module_id)
                    .unwrap_or_else(|| aliases[0])
            });

            let header = if let Some(func) = self.thir.funcs.get(&id) {
                func.header.clone()
            } else {
                let func = self.hir.funcs.remove(&id).expect("func not found");
                let header = self.lower_func_header(func.header)?;
                let func = Func {
                    vis: func.vis,
                    kind: func.kind,
                    header: header.clone(),
                    body: func.body,
                };
                self.thir.funcs.insert(id, func.clone());
                lowering.push((id, func));
                header
            };

            for name in aliases {
                self.scope_mut().funcs.insert(name, (id, header.clone()));
            }
            items.insert((ItemKind::Func, primary), id);
        }

        for (id, func) in lowering {
            let ty = self.lower_func_scope(&func)?;
            let old = &mut self.thir.funcs.get_mut(&id).unwrap().header.ret_ty;
            self.table
                .constraints
                .push_back(Constraint(old.clone(), ty.clone()));
            let _ = self.table.unify_all();
            *old = ty;
        }

        let mut exit_action = None;
        let full_span = scope.children.span();
        let mut lowered = Vec::with_capacity(scope.children.value().len());
        let mut children = scope.children.value_mut().drain(..).peekable();

        while let Some(node) = children.next() {
            let (node, mut et) = self.lower_node(node)?;
            lowered.push(node);

            if let Some(action) = &self.scope().exited {
                et = Some(action.clone());
            }
            if let Some(et) = et {
                self.handle_exit(&et, divergent)?;
                exit_action = Some(et);
                break;
            }
        }
        self.thir.scopes.insert(
            scope_id,
            crate::Scope {
                module_id: scope.module_id,
                label,
                decorators: scope.decorators,
                children: lowered.spanned(full_span),
                items,
            },
        );
        let exit_action = exit_action.unwrap_or_else(|| {
            self.unify_scope(scope_id, Ty::VOID, full_span);
            self.exit_scope();
            match kind {
                ScopeKind::Loop => ExitAction::NeverReturn,
                _ => ExitAction::FromBlock(None, Ty::VOID, full_span),
            }
        });

        let first_span = children.peek().map(|child| child.span());
        let mut last_span = first_span;
        for node in children {
            last_span = Some(node.span());
            self.lower_node(node)?;
        }
        // If there are any remaining nodes, they are unreachable
        if let Some(first_span) = first_span {
            self.warnings.push(Warning::UnreachableCode(
                exit_action.span(),
                // SAFETY: first_span is Some if children is not empty
                first_span.merge(unsafe { last_span.unwrap_unchecked() }),
            ));
        }
        Ok(exit_action)
    }

    /// Lowers a module into a typed HIR module.
    pub fn lower_module(&mut self, module_id: ModuleId) -> Result<()> {
        let scope_id = *self.hir.modules.get(&module_id).unwrap();

        // TODO: exit action can be non-standard in things like REPLs
        match self.lower_scope(
            scope_id,
            ScopeKind::Block,
            Vec::new(),
            Vec::new(),
            true,
            None,
        )? {
            _ => (), // TODO: For packaged code, check exit type is void
        }
        self.exit_scope_if_exists();
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

/// Flattens a pattern parameter into a list of bindings to prepare for MIR lowering.
///
/// # Example
/// ```terbium
/// func sum_tuple((a, b): (int, int)) = a + b;
///
/// // Lowered to:
/// func sum_tuple(a: int, b: int) = a + b;
/// ```
pub fn flatten_param(
    pat: &Spanned<Pattern>,
    ty: Ty,
    bindings: &mut Vec<(Ident, Binding)>,
) -> Result<()> {
    match (pat.value(), ty) {
        (Pattern::Ident { ident, mut_kw }, ty) => bindings.push((
            ident.0,
            Binding {
                def_span: ident.1,
                ty,
                mutable: mut_kw.clone(),
                initialized: true,
                kind: BindingKind::Param,
            },
        )),
        (Pattern::Tuple(pats), Ty::Tuple(tys)) => {
            if pats.len() != tys.len() {
                return Err(pat_errors::tuple_len_mismatch(
                    pats.len().spanned(pat.span()),
                    tys.len(),
                    None,
                ));
            }
            for (pat, ty) in pats.iter().zip(tys) {
                flatten_param(pat, ty, bindings)?;
            }
        }
        (Pattern::Tuple(_), ty) => {
            return Err(pat_errors::tuple_mismatch(pat.span(), ty, None));
        }
    }
    Ok(())
}

pub mod pat_errors {
    use super::*;

    #[inline]
    pub fn tuple_len_mismatch(
        pat_len: Spanned<usize>,
        ty_len: usize,
        ty_span: Option<Span>,
    ) -> Error {
        Error::PatternMismatch {
            pat: Cow::Owned(format!("{}-element tuple", pat_len.value())),
            pat_span: pat_len.span(),
            value: format!("{}-element tuple", ty_len),
            value_span: ty_span,
        }
    }

    #[inline]
    pub fn tuple_mismatch(pat_span: Span, ty: Ty, ty_span: Option<Span>) -> Error {
        Error::PatternMismatch {
            pat: Cow::Borrowed("tuple"),
            pat_span,
            value: format!("value of type `{}`", crate::Ty::from(ty)),
            value_span: ty_span,
        }
    }
}
