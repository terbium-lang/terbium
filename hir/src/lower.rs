use crate::{
    error::{Error, Result},
    Const, Decorator, Expr, FieldVisibility, FloatWidth, Func, FuncHeader, FuncParam, Hir, Ident,
    IntSign, IntWidth, ItemId, Literal, LogicalOp, ModuleId, Node, Op, Pattern, PrimitiveTy, Scope,
    ScopeId, StructField, StructTy, Ty, TyDef, TyParam,
};
use common::span::{Span, Spanned, SpannedExt};
use grammar::{
    ast::{
        self, AssignmentOperator, AssignmentTarget, Atom, GenericTyApp, StructDef, TypeExpr,
        TypePath, While,
    },
    token::IntLiteralInfo,
};
use internment::Intern;
use std::collections::HashMap;

type NamedTyDef = (ItemId, TyDef);

/// A temporary state used when lowering an AST to HIR.
#[must_use = "the HIR is only constructed when this is used"]
pub struct AstLowerer {
    /// Lookup of module ASTs given a module ID
    module_nodes: HashMap<ModuleId, Vec<Spanned<ast::Node>>>,
    /// During the lowering of structs to typerefs, this keeps a list of structs that need to
    /// inherit fields.
    sty_needs_field_resolution: HashMap<ItemId, (Span, ItemId, Spanned<TypePath>, Vec<Ty>)>,
    /// Build up outer decorators to apply to the next item.
    outer_decorators: Vec<Spanned<Decorator>>,
    /// The HIR being constructed.
    pub hir: Hir,
    /// Non-fatal errors that occurred during lowering.
    pub errors: Vec<Error>,
}

/// A cumulative context used when lowering an AST to HIR.
pub struct Ctx<'a> {
    pub scope: &'a Scope,
    pub parent_scope: Option<ScopeId>,
    pub ty_params: Vec<TyParam>,
}

impl<'a> Ctx<'a> {
    fn new(scope: &'a Scope) -> Self {
        Self {
            scope,
            parent_scope: None,
            ty_params: Vec::new(),
        }
    }

    fn module(&self) -> ModuleId {
        self.scope.module_id
    }
}

#[inline]
pub fn get_ident(ident: String) -> Ident {
    Ident(Intern::new(ident))
}

#[inline]
pub fn get_ident_from_ref(ident: impl AsRef<str>) -> Ident {
    Ident(Intern::from_ref(ident.as_ref()))
}

#[inline]
fn ty_params_into_ty(ty_params: &[ast::TyParam]) -> Vec<Ty> {
    ty_params
        .iter()
        .map(|tp| Ty::Generic(get_ident_from_ref(tp.name.value())))
        .collect()
}

#[inline]
fn ty_params_into_unbounded_ty_param(ty_params: &[ast::TyParam]) -> Vec<TyParam> {
    ty_params
        .iter()
        .map(|tp| TyParam {
            name: get_ident_from_ref(tp.name.value()),
            bound: None,
            infer: false,
        })
        .collect()
}

impl AstLowerer {
    /// Creates a new AST lowerer.
    pub fn new(root: Vec<Spanned<ast::Node>>) -> Self {
        Self {
            module_nodes: HashMap::from([(ModuleId::root(), root)]),
            sty_needs_field_resolution: HashMap::new(),
            outer_decorators: Vec::new(),
            hir: Hir::default(),
            errors: Vec::new(),
        }
    }

    #[inline]
    fn register_scope(&mut self, scope: Scope) -> ScopeId {
        let scope_id = ScopeId(self.hir.scopes.len());
        self.hir.scopes.insert(scope_id, scope);
        scope_id
    }

    #[inline]
    fn err_nonfatal(&mut self, err: impl Into<Error>) {
        self.errors.push(err.into());
    }

    #[inline]
    fn propagate_nonfatal<T>(&mut self, result: Result<T>) -> Option<T> {
        match result {
            Ok(t) => Some(t),
            Err(err) => {
                self.err_nonfatal(err);
                None
            }
        }
    }

    /// Asserts the item ID is unique.
    #[inline]
    pub fn assert_item_unique(
        &self,
        scope: &Scope,
        item: &ItemId,
        src: Spanned<String>,
    ) -> Result<()> {
        let occupied = if let Some(occupied) = scope.structs.get(item) {
            Some(occupied.name.span())
        } else if let Some(occupied) = scope.consts.get(item) {
            Some(occupied.name.span())
        } else {
            None
        };
        if let Some(occupied) = occupied {
            return Err(Error::NameConflict(occupied, src));
        }
        Ok(())
    }

    /// Creates a new context for the given scope.
    #[inline]
    pub fn scope_ctx(&self, scope: ScopeId) -> Ctx {
        Ctx::new(self.hir.scopes.get(&scope).unwrap())
    }

    /// Creates a new context for the given module.
    #[inline]
    pub fn module_ctx(&self, module: ModuleId) -> Ctx {
        let scope_id = self.hir.modules.get(&module).unwrap();
        self.scope_ctx(*scope_id)
    }

    /// Completely performs a lowering pass over a module.
    pub fn resolve_module(&mut self, module: ModuleId, span: Span) -> Result<()> {
        // SAFETY: `children` is set later in this function.
        let mut scope = Scope::from_module_id(module);
        self.resolve_types(module, &mut scope)?;
        self.resolve_consts(module, &mut scope)?;
        self.resolve_funcs(module, &mut scope)?;

        let ctx = Ctx::new(&scope);
        let nodes = std::mem::replace(self.module_nodes.get_mut(&module).unwrap(), Vec::new());
        scope.children = self.lower_ast_nodes(&ctx, nodes)?.spanned(span);

        let scope_id = self.register_scope(scope);
        self.hir.modules.insert(module, scope_id);

        Ok(())
    }

    /// Perform a pass over the AST to simply resolve all top-level types.
    pub fn resolve_types(&mut self, module: ModuleId, scope: &mut Scope) -> Result<()> {
        let nodes = self.module_nodes.get(&module).expect("module not found");

        // Do a pass over all types to identify them
        for node in nodes {
            if let Some((item_id, ty_def)) = self.pass_over_ty_def(module, node)? {
                scope.types.insert(item_id, ty_def);
            }
        }

        // Do a second pass to register and resolve types
        for Spanned(node, _) in nodes.clone() {
            match node {
                ast::Node::Struct(sct) => {
                    let sct_name = sct.name.clone();
                    let ident = get_ident(sct_name.0.clone());
                    let item_id = ItemId(module, ident);
                    let sty = self.lower_struct_def_into_ty(module, sct.clone(), scope)?;

                    // Update type parameters with their bounds
                    if let Some(ty_def) = scope.types.get_mut(&item_id) {
                        ty_def.ty_params = sty.ty_params.clone();
                    }
                    self.propagate_nonfatal(self.assert_item_unique(scope, &item_id, sct_name));
                    scope.structs.insert(item_id, sty);
                }
                _ => (),
            }
        }

        let mut sty_parents =
            std::mem::replace(&mut self.sty_needs_field_resolution, HashMap::new());

        // Do a pass over all structs to resolve parent fields
        while !sty_parents.is_empty() {
            let mut seen = HashMap::<_, (Spanned<Ident>, Spanned<TypePath>, ItemId)>::with_capacity(
                sty_parents.len(),
            );
            // Grab the next struct item ID to resolve
            let mut key = unsafe {
                // SAFETY: sty_parents is guaranteed to have elements
                *sty_parents.keys().next().unwrap_unchecked()
            };

            let mut removed = Vec::with_capacity(sty_parents.len());
            // Walk up in the FRO (field resolution order) tree, checking if we encounter
            // a seen type again, and if so, this is a circular type reference.
            while let Some((sid, (src_span, pid, dest, args))) = sty_parents.remove_entry(&key) {
                // Has the destination type been seen?
                if seen.contains_key(&pid) {
                    // If so, this is a circular type reference
                    let mut cycle = Vec::with_capacity(seen.len());

                    let mut cur = pid;
                    while let Some((src, dest, pid)) = seen.remove(&cur) {
                        cycle.push((src, dest));
                        cur = pid;
                    }
                    cycle.push((Spanned(sid.1, src_span), dest));
                    return Err(Error::CircularTypeReference(cycle));
                }

                let fields = scope
                    .structs
                    .get(&pid)
                    .cloned()
                    .expect("struct not found, this is a bug")
                    .into_adhoc_struct_ty_with_applied_ty_params(Some(dest.span()), args)?
                    .fields;
                removed.push(sid);

                for child in &removed {
                    let sty = scope
                        .structs
                        .get_mut(&child)
                        .expect("struct not found, this is a bug");

                    let mut fields = fields.clone();
                    fields.append(&mut sty.fields);
                    sty.fields = fields;
                }

                key = pid;
                seen.insert(sid, (Spanned(sid.1, src_span), dest, pid));
            }
        }

        self.desugar_inferred_types_in_structs(scope);
        Ok(())
    }

    /// Perform a pass over the AST to resolve all top-level constants
    pub fn resolve_consts(&mut self, module: ModuleId, scope: &mut Scope) -> Result<()> {
        let nodes = self.module_nodes.get(&module).expect("module not found");

        // TODO: Cloning may not be necessary here...
        for node in nodes.clone() {
            if let ast::Node::Const {
                vis,
                name,
                ty,
                value,
                ..
            } = node.into_value()
            {
                let ctx = Ctx::new(scope);
                let ident = name.as_ref().map(get_ident_from_ref);
                let cnst = Const {
                    vis,
                    name: ident,
                    ty: self.lower_ty_or_infer(&ctx, ty.map(Spanned::into_value))?,
                    value: self.lower_expr(&ctx, value)?,
                };
                let item = ItemId(module, *ident.value());
                self.propagate_nonfatal(self.assert_item_unique(scope, &item, name));
                scope.consts.insert(item, cnst);
            }
        }
        Ok(())
    }

    #[inline]
    fn pass_over_ty_def(
        &self,
        module: ModuleId,
        Spanned(node, _span): &Spanned<ast::Node>,
    ) -> Result<Option<NamedTyDef>> {
        Ok(match node {
            ast::Node::Struct(sct) => Some(self.pass_over_struct_def(module, sct)?),
            _ => None,
        })
    }

    fn pass_over_struct_def(&self, module: ModuleId, sct: &StructDef) -> Result<NamedTyDef> {
        let sct_name = sct.name.clone();
        let ident = get_ident(sct_name.0.clone());
        let item_id = ItemId(module, ident);

        Ok((
            item_id,
            TyDef {
                name: sct_name.map(|ident| get_ident(ident)),
                ty: Ty::Struct(item_id, ty_params_into_ty(&sct.ty_params)),
                // This is only a pass over the struct def, so we don't actually care about
                // the type bounds
                ty_params: ty_params_into_unbounded_ty_param(&sct.ty_params),
            },
        ))
    }

    pub fn lower_ty_param(&mut self, ctx: &Ctx, param: ast::TyParam) -> Result<TyParam> {
        Ok(TyParam {
            name: get_ident(param.name.into_value()),
            bound: param
                .bound
                .map(|bound| self.lower_ty(&ctx, bound.into_value()))
                .transpose()?
                .map(Box::new),
            infer: false,
        })
    }

    pub fn lower_struct_def_into_ty(
        &mut self,
        module: ModuleId,
        struct_def: StructDef,
        scope: &Scope,
    ) -> Result<StructTy> {
        // Accumulate all generic type parameters
        let mut ctx = Ctx::new(scope);
        for param in struct_def.ty_params {
            let param = self.lower_ty_param(&ctx, param)?;
            ctx.ty_params.push(param);
        }

        // Acknowledge fields from parent struct, we will resolve them later
        if let Some(parent) = struct_def.extends {
            if let TypeExpr::Path(path) = parent.value() {
                let ty = self.lower_ty_path(&ctx, path.clone())?;

                if let Ty::Struct(sid, args) = ty {
                    // Defer the field resolution to later
                    self.sty_needs_field_resolution.insert(
                        ItemId(module, get_ident_from_ref(struct_def.name.value())),
                        (struct_def.name.span(), sid, path.clone(), args),
                    );
                } else {
                    return Err(Error::CannotExtendFieldsFromType(parent));
                }
            } else {
                return Err(Error::CannotExtendFieldsFromType(parent));
            }
        }

        // Lower the struct fields
        let fields = struct_def
            .fields
            .into_iter()
            .map(|Spanned(field, _)| {
                Ok(StructField {
                    vis: FieldVisibility::from_ast(field.vis)?,
                    name: get_ident(field.name.into_value()),
                    ty: self.lower_ty(&ctx, field.ty.into_value())?,
                    // TODO: resolution of this expr should be resolved later
                    default: field
                        .default
                        .map(|d| self.lower_expr(&ctx, d))
                        .transpose()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(StructTy {
            vis: struct_def.vis,
            name: struct_def
                .name
                .as_ref()
                .map(|name| get_ident_from_ref(name)),
            ty_params: ctx.ty_params,
            fields,
        })
    }

    /// Desugar inferred types in structs to generics, e.g.:
    ///
    /// ```text
    /// struct A { a: _ }
    /// ```
    ///
    /// Desugars to:
    ///
    /// ```text
    /// struct A<__0> { a: __0 }
    /// ```
    fn desugar_inferred_types_in_structs(&mut self, scope: &mut Scope) {
        for sty in scope.structs.values_mut() {
            // Desugar inference type into generics that will be inferred anyways
            for (i, ty) in sty
                .fields
                .iter_mut()
                .flat_map(|field| field.ty.iter_unknown_types())
                .enumerate()
            {
                let name = get_ident(format!("__{i}"));
                sty.ty_params.push(TyParam {
                    name,
                    bound: None,
                    infer: true,
                });
                *ty = Ty::Generic(name);
            }
        }
    }

    #[inline]
    fn lower_ty_or_infer(&self, ctx: &Ctx, ty: Option<TypeExpr>) -> Result<Ty> {
        if let Some(ty) = ty {
            self.lower_ty(ctx, ty)
        } else {
            Ok(Ty::Unknown)
        }
    }

    /// Lowers a type expression into an HIR type.
    pub fn lower_ty(&self, ctx: &Ctx, ty: TypeExpr) -> Result<Ty> {
        match ty {
            TypeExpr::Infer => Ok(Ty::Unknown),
            TypeExpr::Path(path) => self.lower_ty_path(ctx, path),
            TypeExpr::Tuple(tys) => {
                let tys = tys
                    .into_iter()
                    .map(|ty| self.lower_ty(ctx, ty.into_value()))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Ty::Tuple(tys))
            }
            _ => todo!(),
        }
    }

    /// Tries resolving a type from a type path
    pub fn lower_ty_path(&self, ctx: &Ctx, path: Spanned<TypePath>) -> Result<Ty> {
        let (TypePath(segments), full_span) = path.into_inner();
        // Add as many segments as possible to the module path
        let mut ty_module = Vec::with_capacity(segments.len());
        // SAFETY: we have at least one segment, so this will be
        let mut span_parts = Vec::with_capacity(segments.len());
        let mut application = None;
        let segments = segments.into_iter();

        for Spanned(segment, _) in segments {
            // Always favor primitive types over anything else
            if let Some(ty) = Self::lower_ty_ident_into_primitive(segment.0.value()) {
                if let Some(Spanned(_, span)) = &segment.1 {
                    return Err(Error::ScalarTypeWithArguments(ty, segment.0.span(), *span));
                }
                return Ok(Ty::Primitive(ty));
            }

            let (ident, span) = segment.0.into_inner();
            ty_module.push(ident);
            span_parts.push(span);
            if let Some(app) = segment.1 {
                application.replace(app);
                break;
            }
            let interned = Intern::from_ref(ty_module.as_slice());
            if !self.hir.modules.contains_key(&ModuleId(interned)) {
                break;
            }
        }

        // UNWRAP: We know that we have at least one segment
        let tail = ty_module.pop().unwrap();
        let ident = get_ident(tail.clone());
        let span = span_parts.pop().unwrap();
        let mid = if ty_module.is_empty() {
            // Check if the tail is a type parameter
            if let Some(param) = ctx.ty_params.iter().find(|&param| param.name == ident) {
                return Ok(Ty::Generic(param.name));
            }
            // If we have no segments, then we have an empty path. Look in the current module.
            ctx.module()
        } else {
            ModuleId(Intern::from_ref(ty_module.as_slice()))
        };

        let lookup = ItemId(mid, ident);
        let ty_def = ctx
            .scope
            .types
            .get(&lookup)
            .cloned()
            .ok_or(Error::TypeNotFound(full_span, Spanned(tail, span), mid))?;

        let ty_params = match application {
            Some(app) => app
                .0
                .args
                .into_iter()
                .map(|ty| self.lower_ty(ctx, ty.into_value()))
                .collect::<Result<Vec<_>>>()?,
            None => Vec::new(),
        };
        ty_def.apply_params(span, ty_params)
    }

    pub fn lower_ty_ident_into_primitive(s: &str) -> Option<PrimitiveTy> {
        macro_rules! int_ty {
            ($sign:ident $w:ident) => {{
                Some(PrimitiveTy::Int(IntSign::$sign, IntWidth::$w))
            }};
        }

        match s {
            "int" => int_ty!(Signed Unknown),
            "int8" => int_ty!(Signed Int8),
            "int16" => int_ty!(Signed Int16),
            "int32" => int_ty!(Signed Int32),
            "int64" => int_ty!(Signed Int64),
            "int128" => int_ty!(Signed Int128),
            "uint" => int_ty!(Unsigned Unknown),
            "uint8" => int_ty!(Unsigned Int8),
            "uint16" => int_ty!(Unsigned Int16),
            "uint32" => int_ty!(Unsigned Int32),
            "uint64" => int_ty!(Unsigned Int64),
            "uint128" => int_ty!(Unsigned Int128),
            "float" => Some(PrimitiveTy::Float(FloatWidth::Unknown)),
            "float32" => Some(PrimitiveTy::Float(FloatWidth::Float32)),
            "float64" => Some(PrimitiveTy::Float(FloatWidth::Float64)),
            "bool" => Some(PrimitiveTy::Bool),
            "char" => Some(PrimitiveTy::Char),
            "void" => Some(PrimitiveTy::Void),
            _ => None,
        }
    }

    /// Lowers a top-level function declaration into an HIR node.
    pub fn resolve_funcs(&mut self, module: ModuleId, scope: &mut Scope) -> Result<()> {
        let nodes = self.module_nodes.get(&module).expect("module not found");

        for node in nodes.clone() {
            if let ast::Node::Func {
                vis,
                name,
                ty_params,
                params,
                kw_params: _, // TODO
                ret,
                body,
            } = node.into_value()
            {
                let mut ctx = Ctx::new(&scope);
                let ident = name.as_ref().map(get_ident_from_ref);
                for ty_param in ty_params {
                    ctx.ty_params.push(self.lower_ty_param(&ctx, ty_param)?);
                }
                let header = FuncHeader {
                    name: ident,
                    ty_params: ctx.ty_params.clone(),
                    params: params
                        .into_iter()
                        .map(|Spanned(ast::FuncParam { pat, ty, default }, _)| {
                            Ok(FuncParam {
                                pat: self.lower_pat(pat),
                                ty: self.lower_ty(&ctx, ty.0)?,
                                default: default
                                    .map(|expr| self.lower_expr(&ctx, expr))
                                    .transpose()?,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?,
                    ret_ty_span: ret.as_ref().map(|ret| ret.span()),
                    ret_ty: self.lower_ty_or_infer(&ctx, ret.map(Spanned::into_value))?,
                };
                let func = Func {
                    vis,
                    header,
                    body: self.lower_body(&ctx, &None, body)?,
                };
                let item = ItemId(module, *ident.value());
                self.propagate_nonfatal(self.assert_item_unique(ctx.scope, &item, name));
                scope.funcs.insert(item, func);
            }
        }
        Ok(())
    }

    /// Desugars a conditional control flow expression into an if-statement, e.g.
    /// `break if COND` becomes `if COND { break; }`
    #[inline]
    fn desugar_conditional_ctrl(
        &mut self,
        ctx: &Ctx,
        cond: Option<Spanned<ast::Expr>>,
        base: Spanned<Node>,
    ) -> Result<Node> {
        Ok(if let Some(cond @ Spanned(_, span)) = cond {
            Node::Expr(Spanned(
                Expr::If(
                    Box::new(self.lower_expr(ctx, cond)?),
                    self.register_scope(Scope::new(ctx.module(), None, Spanned(vec![base], span))),
                    None,
                ),
                span,
            ))
        } else {
            base.into_value()
        })
    }

    /// Lowers a statement into an HIR node.
    pub fn lower_stmt(
        &mut self,
        ctx: &Ctx,
        node: Spanned<ast::Node>,
    ) -> Result<Option<Spanned<Node>>> {
        let Spanned(node, span) = node;
        let node = match node {
            ast::Node::Expr(expr) => Node::Expr(self.lower_expr(ctx, expr)?),
            ast::Node::Let { pat, ty, value, .. } => {
                let ty_span = ty.as_ref().map(|ty| ty.span());
                Node::Let {
                    pat: self.lower_pat(pat),
                    ty: self.lower_ty_or_infer(ctx, ty.map(Spanned::into_value))?,
                    ty_span,
                    value: value.map(|value| self.lower_expr(ctx, value)).transpose()?,
                }
            }
            ast::Node::Break {
                label, value, cond, ..
            } => {
                let base = Node::Break(
                    label.map(|label| label.map(get_ident)),
                    value.map(|value| self.lower_expr(ctx, value)).transpose()?,
                );
                self.desugar_conditional_ctrl(ctx, cond, Spanned(base, span))?
            }
            ast::Node::Continue { label, cond, .. } => self.desugar_conditional_ctrl(
                ctx,
                cond,
                Spanned(
                    Node::Continue(label.map(|label| label.map(get_ident))),
                    span,
                ),
            )?,
            ast::Node::Return { value, cond, .. } => {
                let base =
                    Node::Return(value.map(|value| self.lower_expr(ctx, value)).transpose()?);
                self.desugar_conditional_ctrl(ctx, cond, Spanned(base, span))?
            }
            ast::Node::ImplicitReturn(expr) => Node::ImplicitReturn(self.lower_expr(ctx, expr)?),
            ast::Node::OuterDecorator(decorator) => {
                self.lower_decorator(decorator)
                    .map(|decorator| self.outer_decorators.push(decorator.spanned(span)));
                return Ok(None);
            }
            _ => return Ok(None),
        };
        Ok(Some(Spanned(node, span)))
    }

    /// Lowers a decorator.
    pub fn lower_decorator(&mut self, decorator: ast::Decorator) -> Option<Decorator> {
        let Spanned(Some((name, loc)), path_span) = decorator.path.as_ref().map(|dec| dec.split_last()) else { return None };

        let mut assert_no_args = |out: Decorator| {
            if let Some(Spanned(_, span)) = &decorator.args {
                self.err_nonfatal(Error::BareDecoratorWithArguments(
                    name.value().clone().spanned(path_span),
                    *span,
                ));
            }
            Some(out)
        };

        match (loc, name.value().as_str()) {
            // third-party decorators cannot be named any built-in decorators,
            // so check for a built-in decorator first
            ([], "inline") => assert_no_args(Decorator::Inline),
            ([], "always_inline") => assert_no_args(Decorator::AlwaysInline),
            ([], "never_inline") => assert_no_args(Decorator::NeverInline),
            ([], "rarely_called") => assert_no_args(Decorator::RarelyCalled),
            ([], "frequently_called") => assert_no_args(Decorator::FrequentlyCalled),
            ([], "suppress") => todo!(),
            _ => todo!(),
        }
    }

    /// Lowers a pattern into an HIR pattern.
    pub fn lower_pat(&mut self, Spanned(pat, span): Spanned<ast::Pattern>) -> Spanned<Pattern> {
        let pat = match pat {
            ast::Pattern::Ident { ident, mut_kw } => Pattern::Ident {
                ident: ident.map(get_ident),
                mut_kw,
            },
            ast::Pattern::Tuple(_variant, pats) => {
                Pattern::Tuple(pats.into_iter().map(|pat| self.lower_pat(pat)).collect())
            }
            _ => todo!(),
        };
        Spanned(pat, span)
    }

    #[inline]
    pub fn lower_ast_nodes(
        &mut self,
        ctx: &Ctx,
        nodes: Vec<Spanned<ast::Node>>,
    ) -> Result<Vec<Spanned<Node>>> {
        nodes
            .into_iter()
            .filter_map(|node| self.lower_stmt(ctx, node).transpose())
            .collect()
    }

    pub fn lower_body(
        &mut self,
        ctx: &Ctx,
        label: &Option<Spanned<String>>,
        nodes: Spanned<Vec<Spanned<ast::Node>>>,
    ) -> Result<ScopeId> {
        let children = nodes
            .map(|nodes| self.lower_ast_nodes(ctx, nodes))
            .transpose()?;

        Ok(self.register_scope(Scope::new(
            ctx.module(),
            label.as_ref().map(|l| l.as_ref().map(get_ident_from_ref)),
            children,
        )))
    }

    /// Lowers an expression into an HIR node.
    pub fn lower_expr(&mut self, ctx: &Ctx, expr: Spanned<ast::Expr>) -> Result<Spanned<Expr>> {
        use ast::Expr as E;
        use ast::UnaryOp as U;

        let (expr, span) = expr.into_inner();
        let expr = match expr {
            E::Atom(atom) => return self.lower_atom(Spanned(atom, span)),
            E::Ident(ident, tys) => return self.lower_ident_expr(ctx, ident.map(get_ident), tys),
            E::UnaryOp { op, expr } => Expr::CallOp(
                op.map(|op| match op {
                    U::Plus => Op::Pos,
                    U::Minus => Op::Neg,
                    U::BitNot => Op::BitNot,
                    U::Not => Op::Not,
                }),
                Box::new(self.lower_expr(ctx, *expr)?),
                Vec::new(),
            ),
            E::BinaryOp { left, op, right } => {
                if let Some(inner) = Self::lower_bin_op(*op.value()) {
                    Expr::CallOp(
                        inner.spanned(op.span()),
                        Box::new(self.lower_expr(ctx, *left)?),
                        vec![self.lower_expr(ctx, *right)?],
                    )
                } else {
                    Expr::CallLogicalOp(
                        op.map(|op| match op {
                            ast::BinaryOp::LogicalAnd => LogicalOp::And,
                            ast::BinaryOp::LogicalOr => LogicalOp::Or,
                            _ => unimplemented!("operator is not a logical operator"),
                        }),
                        Box::new(self.lower_expr(ctx, *left)?),
                        Box::new(self.lower_expr(ctx, *right)?),
                    )
                }
            }
            E::Tuple(exprs) => Expr::Tuple(
                exprs
                    .into_iter()
                    .map(|e| self.lower_expr(ctx, e))
                    .collect::<Result<Vec<_>>>()?,
            ),
            E::Array(exprs) => Expr::Array(
                exprs
                    .into_iter()
                    .map(|e| self.lower_expr(ctx, e))
                    .collect::<Result<Vec<_>>>()?,
            ),
            E::Block { label, body } => Expr::Block(self.lower_body(ctx, &label, body)?),
            E::If {
                label,
                cond,
                body,
                else_if_bodies,
                mut else_body,
                ..
            } => {
                // Desugar all else-if bodies into `else`
                else_body =
                    else_if_bodies
                        .into_iter()
                        .fold(else_body, |else_body, (cond, body)| {
                            let span = cond.span().merge(body.span());

                            Some(ast::expr_as_block(Spanned(
                                ast::Expr::If {
                                    label: None,
                                    cond: Box::new(cond),
                                    body,
                                    else_if_bodies: Vec::new(),
                                    else_body,
                                    ternary: false,
                                },
                                span,
                            )))
                        });

                Expr::If(
                    Box::new(self.lower_expr(ctx, *cond)?),
                    self.lower_body(ctx, &label, body)?,
                    else_body
                        .map(|body| self.lower_body(ctx, &label, body))
                        .transpose()?,
                )
            }
            E::While(stmt) => self.lower_while_loop(ctx, stmt)?,
            E::Loop { label, body } => Expr::Loop(self.lower_body(ctx, &label, body)?),
            E::Call { func, args, kwargs } => Expr::Call {
                callee: Box::new(self.lower_expr(ctx, *func)?),
                args: args
                    .into_iter()
                    .map(|arg| self.lower_expr(ctx, arg))
                    .collect::<Result<Vec<_>>>()?,
                kwargs: kwargs
                    .into_iter()
                    .map(|(name, arg)| Ok((get_ident(name), self.lower_expr(ctx, arg)?)))
                    .collect::<Result<Vec<_>>>()?,
            },
            E::Assign {
                target,
                op: raw_op,
                value,
            } => {
                match Self::lower_assign_op_into_op(*raw_op.value()) {
                    Some(op) => Expr::CallOp(
                        op.spanned(raw_op.span()),
                        Box::new(self.lower_assignment_target_into_expr(ctx, target)?),
                        vec![self.lower_expr(ctx, *value)?],
                    ),
                    None => match raw_op.value() {
                        // Desugar a ||= b or a &&= b to `a = a || b` or `a = a && b`
                        op @ (AssignmentOperator::LogicalOrAssign
                        | AssignmentOperator::LogicalAndAssign) => {
                            let rhs = Expr::CallLogicalOp(
                                Spanned(
                                    match op {
                                        AssignmentOperator::LogicalAndAssign => LogicalOp::And,
                                        AssignmentOperator::LogicalOrAssign => LogicalOp::Or,
                                        _ => unimplemented!("operator is not a logical operator"),
                                    },
                                    raw_op.span(),
                                ),
                                Box::new(self.lower_expr(
                                    ctx,
                                    Self::lower_assignment_target_into_ast_expr(target.clone())?,
                                )?),
                                Box::new(self.lower_expr(ctx, *value)?),
                            );
                            self.lower_assignment(ctx, target, rhs.spanned(span))?
                        }
                        AssignmentOperator::Assign => {
                            let value = self.lower_expr(ctx, *value)?;
                            self.lower_assignment(ctx, target, value)?
                        }
                        _ => unimplemented!(),
                    },
                }
            }
            E::Attr { subject, attr, .. } => Expr::GetAttr(
                Box::new(self.lower_expr(ctx, *subject)?),
                attr.map(get_ident),
            ),
            E::Index { subject, index } => Expr::CallOp(
                Op::Index.spanned(index.span()),
                Box::new(self.lower_expr(ctx, *subject)?),
                vec![self.lower_expr(ctx, *index)?],
            ),
            E::Cast { expr, ty } => Expr::Cast(
                Box::new(self.lower_expr(ctx, *expr)?),
                self.lower_ty(&ctx, ty.into_value())?,
            ),
            _ => todo!(),
        };
        Ok(Spanned(expr, span))
    }

    #[inline]
    fn lower_assignment(
        &mut self,
        ctx: &Ctx,
        lhs: Spanned<AssignmentTarget>,
        rhs: Spanned<Expr>,
    ) -> Result<Expr> {
        let (lhs, span) = lhs.into_inner();

        Ok(match lhs {
            AssignmentTarget::Pattern(pat) => {
                Expr::Assign(self.lower_pat(Spanned(pat, span)), Box::new(rhs))
            }
            AssignmentTarget::Attr { subject, attr } => Expr::SetAttr(
                Box::new(self.lower_expr(ctx, *subject)?),
                attr.map(get_ident),
                Box::new(rhs),
            ),
            AssignmentTarget::Index { subject, index } => Expr::AssignPtr(
                Box::new(Spanned(
                    Expr::CallOp(
                        Op::IndexMut.spanned(index.span()),
                        Box::new(self.lower_expr(ctx, *subject)?),
                        vec![self.lower_expr(ctx, *index)?],
                    ),
                    span,
                )),
                Box::new(rhs),
            ),
            AssignmentTarget::Pointer(ptr) => {
                Expr::AssignPtr(Box::new(self.lower_expr(ctx, *ptr)?), Box::new(rhs))
            }
        })
    }

    #[inline]
    fn lower_assign_op_into_op(op: AssignmentOperator) -> Option<Op> {
        Some(match op {
            AssignmentOperator::AddAssign => Op::AddAssign,
            AssignmentOperator::SubAssign => Op::SubAssign,
            AssignmentOperator::MulAssign => Op::MulAssign,
            AssignmentOperator::DivAssign => Op::DivAssign,
            AssignmentOperator::ModAssign => Op::ModAssign,
            AssignmentOperator::PowAssign => Op::PowAssign,
            AssignmentOperator::BitAndAssign => Op::BitAndAssign,
            AssignmentOperator::BitOrAssign => Op::BitOrAssign,
            AssignmentOperator::BitXorAssign => Op::BitXorAssign,
            AssignmentOperator::ShlAssign => Op::ShlAssign,
            AssignmentOperator::ShrAssign => Op::ShrAssign,
            _ => return None,
        })
    }

    fn lower_assignment_target_into_ast_expr(
        tgt: Spanned<AssignmentTarget>,
    ) -> Result<Spanned<ast::Expr>> {
        let (tgt, span) = tgt.into_inner();
        match tgt {
            AssignmentTarget::Pattern(p) => match p {
                ast::Pattern::Ident {
                    ident: ident @ Spanned(_, span),
                    ..
                } => Ok(Spanned(ast::Expr::Ident(ident, None), span)),
                pat => Err(Error::InvalidAssignmentTarget(span, pat.name())),
            },
            AssignmentTarget::Attr { subject, attr } => Ok(Spanned(
                ast::Expr::Attr {
                    subject,
                    dot: attr.span(),
                    attr,
                },
                span,
            )),
            AssignmentTarget::Index { subject, index } => {
                Ok(Spanned(ast::Expr::Index { subject, index }, span))
            }
            AssignmentTarget::Pointer(subject) => Ok(*subject),
        }
    }

    fn lower_assignment_target_into_expr(
        &mut self,
        ctx: &Ctx,
        tgt: Spanned<AssignmentTarget>,
    ) -> Result<Spanned<Expr>> {
        let (tgt, span) = tgt.into_inner();
        match tgt {
            AssignmentTarget::Pattern(p) => match p {
                ast::Pattern::Ident { ident, .. } => {
                    self.lower_ident_expr(ctx, ident.map(get_ident), None)
                }
                pat => Err(Error::InvalidAssignmentTarget(span, pat.name())),
            },
            AssignmentTarget::Attr { subject, attr } => Ok(Spanned(
                Expr::GetAttr(
                    Box::new(self.lower_expr(ctx, *subject)?),
                    attr.map(|a| get_ident(a)),
                ),
                span,
            )),
            AssignmentTarget::Index { subject, index } => Ok(Spanned(
                Expr::CallOp(
                    Op::IndexMut.spanned(index.span()),
                    Box::new(self.lower_expr(ctx, *subject)?),
                    vec![self.lower_expr(ctx, *index)?],
                ),
                span,
            )),
            AssignmentTarget::Pointer(subject) => Ok(self.lower_expr(ctx, *subject)?),
        }
    }

    /// Lowers a while-loop into a loop block.
    ///
    /// For example, `while COND { STMT; }` desugars to `loop { if COND { STMT; } else { break; } }`
    /// Note that `while COND { STMT; } else { STMT; }` desugars to `loop { ... else { break { STMT; } } }`
    pub fn lower_while_loop(&mut self, ctx: &Ctx, stmt: While) -> Result<Expr> {
        let label = stmt
            .label
            .as_ref()
            .map(|s| s.as_ref().map(get_ident_from_ref));
        let cond = self.lower_expr(ctx, *stmt.cond)?;
        let body_span = stmt
            .body
            .span()
            .merge_opt(stmt.else_body.as_ref().map(Spanned::span));

        let children = stmt
            .body
            .map(|body| self.lower_ast_nodes(ctx, body))
            .transpose()?;
        let then = self.register_scope(Scope::new(ctx.module(), None, children));
        let else_body = stmt
            .else_body
            .map(|body| {
                let children = body
                    .map(|body| self.lower_ast_nodes(ctx, body))
                    .transpose()?;
                let span = children.span();

                let block = self.register_scope(Scope::new(ctx.module(), None, children));
                Ok(self.register_scope(Scope::new(
                    ctx.module(),
                    None,
                    vec![Node::Break(label, Some(Spanned(Expr::Block(block), span))).spanned(span)]
                        .spanned(span),
                )))
            })
            .transpose()?
            // this is the `else { break; }` portion
            .unwrap_or_else(|| {
                self.register_scope(Scope::new(
                    ctx.module(),
                    None,
                    vec![Node::Break(label, None).spanned(body_span)].spanned(body_span),
                ))
            });

        let scope_id = self.register_scope(Scope::new(
            ctx.module(),
            stmt.label
                .as_ref()
                .map(|l| l.as_ref().map(get_ident_from_ref)),
            vec![
                Node::Expr(Expr::If(Box::new(cond), then, Some(else_body)).spanned(body_span))
                    .spanned(body_span),
            ]
            .spanned(body_span),
        ));
        Ok(Expr::Loop(scope_id))
    }

    #[inline]
    pub fn lower_ident_expr(
        &mut self,
        ctx: &Ctx,
        ident: Spanned<Ident>,
        app: Option<GenericTyApp>,
    ) -> Result<Spanned<Expr>> {
        let item_id = ItemId(ctx.module(), *ident.value());
        Ok(if let Some(cnst) = ctx.scope.consts.get(&item_id) {
            // TODO: true const-eval instead of inline (this will be replaced by `alias`)
            if let Some(app) = app {
                return Err(Error::ExplicitTypeArgumentsNotAllowed(app.span()));
            }
            cnst.value.clone()
        } else {
            Expr::Ident(
                ident,
                app.map(|app| {
                    app.map(|app| {
                        app.into_iter()
                            .map(|ty| self.lower_ty(ctx, ty.into_value()))
                            .collect::<Result<_>>()
                    })
                    .transpose()
                })
                .transpose()?,
            )
            .spanned(ident.span())
        })
    }

    #[inline]
    pub const fn lower_bin_op(op: ast::BinaryOp) -> Option<Op> {
        use ast::BinaryOp as B;

        match op {
            B::Add => Some(Op::Add),
            B::Sub => Some(Op::Sub),
            B::Mul => Some(Op::Mul),
            B::Div => Some(Op::Div),
            B::Mod => Some(Op::Mod),
            B::Pow => Some(Op::Pow),
            B::BitAnd => Some(Op::BitAnd),
            B::BitOr => Some(Op::BitOr),
            B::BitXor => Some(Op::BitXor),
            B::Shl => Some(Op::Shl),
            B::Shr => Some(Op::Shr),
            B::Eq => Some(Op::Eq),
            B::Ne => Some(Op::Ne),
            B::Lt => Some(Op::Lt),
            B::Le => Some(Op::Le),
            B::Gt => Some(Op::Gt),
            B::Ge => Some(Op::Ge),
            _ => None,
        }
    }

    /// Lowers an atom into an HIR literal expression.
    pub fn lower_atom(&mut self, atom: Spanned<Atom>) -> Result<Spanned<Expr>> {
        let (atom, span) = atom.into_inner();
        Ok(Spanned(
            match atom {
                Atom::Int(int, IntLiteralInfo { unsigned, .. }) => Expr::Literal(if unsigned {
                    Literal::UInt(
                        int.parse()
                            .map_err(|_| Error::IntegerLiteralOverflow(span, int))?,
                    )
                } else {
                    Literal::Int(
                        int.parse()
                            .map_err(|_| Error::IntegerLiteralOverflow(span, int))?,
                    )
                }),
                Atom::Float(f) => {
                    Expr::Literal(Literal::Float(f.parse::<f64>().unwrap().to_bits()))
                }
                Atom::Bool(b) => Expr::Literal(Literal::Bool(b)),
                Atom::Char(c) => Expr::Literal(Literal::Char(c)),
                Atom::String(s) => Expr::Literal(Literal::String(s)),
                Atom::Void => Expr::Literal(Literal::Void),
            },
            span,
        ))
    }
}
