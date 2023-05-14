use crate::{
    error::{Error, Result},
    Const, Expr, FieldVisibility, FloatWidth, Hir, Ident, IntSign, IntWidth, ItemId, Literal,
    ModuleId, Node, Op, Pattern, PrimitiveTy, Scope, ScopeId, StructField, StructTy, Ty, TyDef,
    TyParam,
};
use common::span::{Span, Spanned};
use grammar::{
    ast::{self, AssignmentOperator, AssignmentTarget, Atom, StructDef, TypeExpr, TypePath, While},
    token::IntLiteralInfo,
};
use internment::Intern;
use std::collections::HashMap;

const BOOL: Ty = Ty::Primitive(PrimitiveTy::Bool);

type NamedTyDef = (ItemId, TyDef);

/// A temporary state used when lowering an AST to HIR.
#[must_use = "the HIR is only constructed when this is used"]
pub struct AstLowerer {
    /// Synchronously increasing scope ID
    scope_id: ScopeId,
    /// Lookup of module ASTs given a module ID
    module_nodes: HashMap<ModuleId, Vec<Spanned<ast::Node>>>,
    /// During the lowering of structs to typerefs, this keeps a list of structs that need to
    /// inherit fields.
    sty_needs_field_resolution: HashMap<ItemId, (Span, ItemId, Spanned<TypePath>, Vec<Ty>)>,
    /// The HIR being constructed.
    pub hir: Hir,
}

/// A cumulative context used when lowering an AST to HIR.
pub struct Ctx {
    pub module: ModuleId,
    pub ty_params: Vec<TyParam>,
}

impl Ctx {
    fn new(module: ModuleId) -> Self {
        Self {
            module,
            ty_params: Vec::new(),
        }
    }
}

#[inline]
fn get_ident(ident: String) -> Ident {
    Ident(Intern::new(ident))
}

#[inline]
fn get_ident_from_ref(ident: impl AsRef<str>) -> Ident {
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
            scope_id: ScopeId(0),
            module_nodes: HashMap::from([(ModuleId::root(), root)]),
            sty_needs_field_resolution: HashMap::new(),
            hir: Hir::default(),
        }
    }

    #[inline]
    fn register_scope(&mut self, scope: Scope) -> ScopeId {
        let scope_id = self.scope_id;
        self.hir.scopes.insert(scope_id, scope);
        self.scope_id = self.scope_id.next();
        scope_id
    }

    #[inline]
    fn anon_scope_from_expr(&mut self, expr: Expr) -> ScopeId {
        self.register_scope(Scope {
            label: None,
            children: vec![Node::ImplicitReturn(expr)],
        })
    }

    /// Asserts the item ID is unique.
    #[inline]
    pub fn assert_item_unique(&self, item: &ItemId, src: Spanned<String>) -> Result<()> {
        let occupied = if let Some(occupied) = self.hir.structs.get(item) {
            Some(occupied.name.span())
        } else if let Some(occupied) = self.hir.consts.get(item) {
            Some(occupied.name.span())
        } else {
            None
        };
        if let Some(occupied) = occupied {
            return Err(Error::NameConflict(occupied, src));
        }
        Ok(())
    }

    /// Completely performs a lowering pass over a module.
    pub fn resolve_module(&mut self, module: ModuleId) -> Result<()> {
        self.resolve_top_level_types(module)?;
        self.resolve_top_level_consts(module)?;

        let ctx = Ctx::new(module);
        let nodes = std::mem::replace(self.module_nodes.get_mut(&module).unwrap(), Vec::new());
        let children = self.lower_ast_nodes(&ctx, nodes)?;
        let scope_id = self.register_scope(Scope {
            label: None,
            children,
        });
        self.hir.modules.insert(module, scope_id);

        Ok(())
    }

    /// Perform a pass over the AST to simply resolve all top-level types.
    pub fn resolve_top_level_types(&mut self, module: ModuleId) -> Result<()> {
        let nodes = self.module_nodes.get(&module).expect("module not found");

        // Do a pass over all types to identify them
        for node in nodes {
            if let Some((item_id, ty_def)) = self.pass_over_ty_def(module, node)? {
                self.hir.types.insert(item_id, ty_def);
            }
        }

        // Do a second pass to register and resolve types
        for Spanned(node, _) in nodes.clone() {
            match node {
                ast::Node::Struct(sct) => {
                    let sct_name = sct.name.clone();
                    let ident = get_ident(sct_name.0.clone());
                    let item_id = ItemId(module, ident);
                    let sty = self.lower_struct_def_into_ty(module, sct.clone())?;

                    // Update type parameters with their bounds
                    if let Some(ty_def) = self.hir.types.get_mut(&item_id) {
                        ty_def.ty_params = sty.ty_params.clone();
                    }
                    self.assert_item_unique(&item_id, sct_name)?;
                    self.hir.structs.insert(item_id, sty);
                }
                _ => (),
            }
        }

        let mut sty_parents =
            std::mem::replace(&mut self.sty_needs_field_resolution, HashMap::new());

        // Do a pass over all structs to resolve parent fields
        while !sty_parents.is_empty() {
            let mut seen = HashMap::with_capacity(sty_parents.len());
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
                if let Some(&circular_at) = seen.get(&pid) {
                    // If so, this is a circular type reference
                    return Err(Error::CircularTypeReference {
                        src: Spanned(sid.1, src_span),
                        dest,
                        circular_at,
                    });
                }

                let fields = self
                    .hir
                    .structs
                    .get(&pid)
                    .cloned()
                    .expect("struct not found, this is a bug")
                    .into_adhoc_struct_ty_with_applied_ty_params(Some(dest.span()), args)?
                    .fields;
                removed.push(sid);

                for child in &removed {
                    let sty = self
                        .hir
                        .structs
                        .get_mut(&child)
                        .expect("struct not found, this is a bug");

                    let mut fields = fields.clone();
                    fields.append(&mut sty.fields);
                    sty.fields = fields;
                }

                key = pid;
                seen.insert(sid, src_span);
            }
        }

        self.desugar_inferred_types_in_structs();
        Ok(())
    }

    /// Perform a pass over the AST to resolve all top-level constants
    pub fn resolve_top_level_consts(&mut self, module: ModuleId) -> Result<()> {
        let nodes = self.module_nodes.get(&module).expect("module not found");
        let ctx = Ctx::new(module);

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
                let ident = name.as_ref().map(get_ident_from_ref);
                let cnst = Const {
                    vis,
                    name: ident,
                    ty: self.lower_ty_or_infer(&ctx, ty)?,
                    value: self.lower_expr(&ctx, value)?,
                };
                let item = ItemId(module, *ident.value());
                self.assert_item_unique(&item, name)?;
                self.hir.consts.insert(item, cnst);
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

    pub fn lower_struct_def_into_ty(
        &mut self,
        module: ModuleId,
        struct_def: StructDef,
    ) -> Result<StructTy> {
        // Accumulate all generic type parameters
        let mut ctx = Ctx::new(module);
        for param in struct_def.ty_params {
            let param = TyParam {
                name: get_ident(param.name.into_value()),
                bound: param
                    .bound
                    .map(|bound| self.lower_ty(&ctx, bound.into_value()))
                    .transpose()?
                    .map(Box::new),
                infer: false,
            };
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
    fn desugar_inferred_types_in_structs(&mut self) {
        for sty in self.hir.structs.values_mut() {
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
    fn lower_ty_or_infer(&self, ctx: &Ctx, ty: Option<Spanned<TypeExpr>>) -> Result<Ty> {
        if let Some(ty) = ty {
            self.lower_ty(ctx, ty.into_value())
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
            ctx.module
        } else {
            ModuleId(Intern::from_ref(ty_module.as_slice()))
        };

        let lookup = ItemId(mid, ident);
        let ty_def = self
            .hir
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

    /// Desugars a conditional control flow expression into an if-statement, e.g.
    /// `break if COND` becomes `if COND { break; }`
    #[inline]
    fn desugar_conditional_ctrl(
        &mut self,
        ctx: &Ctx,
        cond: Option<Spanned<ast::Expr>>,
        base: Node,
    ) -> Result<Node> {
        Ok(if let Some(cond) = cond {
            Node::Expr(Expr::If(
                Box::new(self.lower_expr(ctx, cond)?),
                self.register_scope(Scope {
                    label: None,
                    children: vec![base],
                }),
                None,
            ))
        } else {
            base
        })
    }

    /// Lowers a statement into an HIR node.
    pub fn lower_stmt(&mut self, ctx: &Ctx, node: Spanned<ast::Node>) -> Result<Option<Node>> {
        let node = match node.into_value() {
            ast::Node::Expr(expr) => Node::Expr(self.lower_expr(ctx, expr)?),
            ast::Node::Let { pat, ty, value, .. } => Node::Let {
                pat: self.lower_pat(pat.into_value()),
                ty: self.lower_ty_or_infer(ctx, ty)?,
                value: value.map(|value| self.lower_expr(ctx, value)).transpose()?,
            },
            ast::Node::Break {
                label, value, cond, ..
            } => {
                let base = Node::Break(
                    label.map(|label| get_ident(label.into_value())),
                    value.map(|value| self.lower_expr(ctx, value)).transpose()?,
                );
                self.desugar_conditional_ctrl(ctx, cond, base)?
            }
            ast::Node::Continue { label, cond, .. } => self.desugar_conditional_ctrl(
                ctx,
                cond,
                Node::Continue(label.map(|label| get_ident(label.into_value()))),
            )?,
            ast::Node::Return { value, cond, .. } => {
                let base =
                    Node::Return(value.map(|value| self.lower_expr(ctx, value)).transpose()?);
                self.desugar_conditional_ctrl(ctx, cond, base)?
            }
            ast::Node::ImplicitReturn(expr) => Node::ImplicitReturn(self.lower_expr(ctx, expr)?),
            _ => return Ok(None),
        };
        Ok(Some(node))
    }

    /// Lowers a pattern into an HIR pattern.
    pub fn lower_pat(&mut self, pat: ast::Pattern) -> Pattern {
        match pat {
            ast::Pattern::Ident { ident, mut_kw } => Pattern::Ident {
                ident: get_ident(ident.into_value()),
                is_mut: mut_kw.is_some(),
            },
            _ => todo!(),
        }
    }

    #[inline]
    pub fn lower_ast_nodes(
        &mut self,
        ctx: &Ctx,
        nodes: Vec<Spanned<ast::Node>>,
    ) -> Result<Vec<Node>> {
        nodes
            .into_iter()
            .filter_map(|node| self.lower_stmt(ctx, node).transpose())
            .collect()
    }

    pub fn lower_body(
        &mut self,
        ctx: &Ctx,
        label: &Option<Spanned<String>>,
        nodes: Vec<Spanned<ast::Node>>,
    ) -> Result<ScopeId> {
        let children = self.lower_ast_nodes(ctx, nodes)?;

        Ok(self.register_scope(Scope {
            label: label.as_ref().map(|l| get_ident_from_ref(l.value())),
            children,
        }))
    }

    /// Lowers an expression into an HIR node.
    pub fn lower_expr(&mut self, ctx: &Ctx, expr: Spanned<ast::Expr>) -> Result<Expr> {
        use ast::Expr as E;
        use ast::UnaryOp as U;

        let (expr, span) = expr.into_inner();
        Ok(match expr {
            E::Atom(atom) => self.lower_atom(ctx, Spanned(atom, span))?,
            E::UnaryOp { op, expr } => Expr::CallOp(
                match *op.value() {
                    U::Plus => Op::Pos,
                    U::Minus => Op::Neg,
                    U::BitNot => Op::BitNot,
                    U::Not => Op::Not,
                },
                Box::new(self.lower_expr(ctx, *expr)?),
                Vec::new(),
            ),
            E::BinaryOp { left, op, right } => {
                if let Some(op) = Self::lower_bin_op(*op.value()) {
                    Expr::CallOp(
                        op,
                        Box::new(self.lower_expr(ctx, *left)?),
                        vec![self.lower_expr(ctx, *right)?],
                    )
                } else {
                    self.lower_logical_into_if_stmt(ctx, *left, *right, op.into_value())?
                }
            }
            E::Tuple(exprs) => Expr::Tuple(
                exprs
                    .into_iter()
                    .map(|e| self.lower_expr(ctx, e))
                    .collect::<Result<Vec<_>>>()?,
            ),
            E::Block { label, body } => {
                Expr::Block(self.lower_body(ctx, &label, body.into_value())?)
            }
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
                    self.lower_body(ctx, &label, body.into_value())?,
                    else_body
                        .map(|body| self.lower_body(ctx, &label, body.into_value()))
                        .transpose()?,
                )
            }
            E::While(stmt) => self.lower_while_loop(ctx, stmt)?,
            E::Loop { label, body } => {
                Expr::Loop(self.lower_body(ctx, &label, body.into_value())?)
            }
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
            E::Assign { target, op, value } => {
                match Self::lower_assign_op_into_op(*op.value()) {
                    Some(op) => Expr::CallOp(
                        op,
                        Box::new(self.lower_assignment_target_into_expr(ctx, target)?),
                        vec![self.lower_expr(ctx, *value)?],
                    ),
                    None => match op.value() {
                        // Desugar a ||= b or a &&= b to `a = a || b` or `a = a && b`
                        op @ (AssignmentOperator::LogicalOrAssign
                        | AssignmentOperator::LogicalAndAssign) => {
                            let rhs = self.lower_logical_into_if_stmt(
                                ctx,
                                Self::lower_assignment_target_into_ast_expr(target.clone())?,
                                *value,
                                match op {
                                    AssignmentOperator::LogicalOrAssign => ast::BinaryOp::LogicalOr,
                                    AssignmentOperator::LogicalAndAssign => {
                                        ast::BinaryOp::LogicalAnd
                                    }
                                    _ => unreachable!(),
                                },
                            )?;
                            self.lower_assignment(ctx, target, rhs)?
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
                Op::Index,
                Box::new(self.lower_expr(ctx, *subject)?),
                vec![self.lower_expr(ctx, *index)?],
            ),
            _ => todo!(),
        })
    }

    #[inline]
    fn lower_assignment(
        &mut self,
        ctx: &Ctx,
        lhs: Spanned<AssignmentTarget>,
        rhs: Expr,
    ) -> Result<Expr> {
        Ok(match lhs.into_value() {
            AssignmentTarget::Pattern(pat) => Expr::Assign(self.lower_pat(pat), Box::new(rhs)),
            AssignmentTarget::Attr { subject, attr } => Expr::SetAttr(
                Box::new(self.lower_expr(ctx, *subject)?),
                attr.map(get_ident),
                Box::new(rhs),
            ),
            AssignmentTarget::Index { subject, index } => Expr::AssignPtr(
                Box::new(Expr::CallOp(
                    Op::IndexMut,
                    Box::new(self.lower_expr(ctx, *subject)?),
                    vec![self.lower_expr(ctx, *index)?],
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
                    ident: Spanned(ident, span),
                    ..
                } => Ok(Spanned(ast::Expr::Atom(Atom::Ident(ident)), span)),
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
    ) -> Result<Expr> {
        let (tgt, span) = tgt.into_inner();
        match tgt {
            AssignmentTarget::Pattern(p) => match p {
                ast::Pattern::Ident { ident, .. } => {
                    Ok(self.lower_ident_expr(ctx, get_ident(ident.into_value())))
                }
                pat => Err(Error::InvalidAssignmentTarget(span, pat.name())),
            },
            AssignmentTarget::Attr { subject, attr } => Ok(Expr::GetAttr(
                Box::new(self.lower_expr(ctx, *subject)?),
                attr.map(|a| get_ident(a)),
            )),
            AssignmentTarget::Index { subject, index } => Ok(Expr::CallOp(
                Op::IndexMut,
                Box::new(self.lower_expr(ctx, *subject)?),
                vec![self.lower_expr(ctx, *index)?],
            )),
            AssignmentTarget::Pointer(subject) => Ok(self.lower_expr(ctx, *subject)?),
        }
    }

    /// Lowers a while-loop into a loop block.
    ///
    /// For example, `while COND { STMT; }` desugars to `loop { if COND { STMT; } else { break; } }`
    /// Note that `while COND { STMT; } else { STMT; }` desugars to `loop { ... else { break { STMT; } } }`
    pub fn lower_while_loop(&mut self, ctx: &Ctx, stmt: While) -> Result<Expr> {
        let label = stmt.label.as_ref().map(|l| get_ident_from_ref(l.value()));
        let cond = self.lower_expr(ctx, *stmt.cond)?;

        let children = self.lower_ast_nodes(ctx, stmt.body.into_value())?;
        let then = self.register_scope(Scope {
            label: None,
            children,
        });
        let else_body = stmt
            .else_body
            .map(|body| {
                let children = self.lower_ast_nodes(ctx, body.into_value())?;
                let block = self.register_scope(Scope {
                    label: None,
                    children,
                });
                Ok(self.register_scope(Scope {
                    label: None,
                    children: vec![Node::Break(label, Some(Expr::Block(block)))],
                }))
            })
            .transpose()?
            // this is the `else { break; }` portion
            .unwrap_or_else(|| {
                self.register_scope(Scope {
                    label: None,
                    children: vec![Node::Break(label, None)],
                })
            });

        let scope_id = self.register_scope(Scope {
            label: stmt.label.as_ref().map(|l| get_ident_from_ref(l.value())),
            children: vec![Node::Expr(Expr::If(Box::new(cond), then, Some(else_body)))],
        });
        Ok(Expr::Loop(scope_id))
    }

    #[inline]
    pub fn lower_ident_expr(&mut self, ctx: &Ctx, ident: Ident) -> Expr {
        let item_id = ItemId(ctx.module, ident);
        if let Some(cnst) = self.hir.consts.get(&item_id) {
            return cnst.value.clone();
        }
        Expr::Ident(ident)
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
            B::Lt => Some(Op::Lt),
            B::Le => Some(Op::Le),
            B::Gt => Some(Op::Gt),
            B::Ge => Some(Op::Ge),
            _ => None,
        }
    }

    /// Desugar logical operators into if statements.
    ///
    /// For example, `a || b` becomes:
    /// ```text
    /// {
    ///     let __lhs = a;
    ///     if __lhs { __lhs } else { b }
    /// }
    /// ```
    #[inline]
    pub fn lower_logical_into_if_stmt(
        &mut self,
        ctx: &Ctx,
        lhs: Spanned<ast::Expr>,
        rhs: Spanned<ast::Expr>,
        op: ast::BinaryOp,
    ) -> Result<Expr> {
        let lhs = self.lower_expr(ctx, lhs)?;
        let rhs = self.lower_expr(ctx, rhs)?;
        let binding = get_ident_from_ref("__lhs");
        let cond = Box::new(Expr::Cast(Box::new(Expr::Ident(binding)), BOOL));

        let then = self.anon_scope_from_expr(Expr::Ident(binding));
        let els = self.anon_scope_from_expr(rhs);
        let stmt = match op {
            ast::BinaryOp::LogicalOr => Expr::If(cond, then, Some(els)),
            ast::BinaryOp::LogicalAnd => Expr::If(cond, then, Some(els)),
            _ => unimplemented!("logical op is not implemented for this operator"),
        };

        let block = self.register_scope(Scope {
            label: None,
            children: vec![
                // Store the LHS in a temporary binding
                Node::Let {
                    pat: Pattern::Ident {
                        ident: binding,
                        is_mut: false,
                    },
                    ty: Ty::Unknown,
                    value: Some(lhs),
                },
                Node::ImplicitReturn(stmt),
            ],
        });
        Ok(Expr::Block(block))
    }

    /// Lowers an atom into an HIR literal expression.
    pub fn lower_atom(&mut self, ctx: &Ctx, atom: Spanned<Atom>) -> Result<Expr> {
        let (atom, span) = atom.into_inner();
        Ok(match atom {
            Atom::Ident(ident) => self.lower_ident_expr(ctx, get_ident(ident)),
            Atom::Int(int, IntLiteralInfo { unsigned, .. }) => Expr::Literal(if unsigned {
                Literal::UInt(
                    int.parse()
                        .map_err(|_| Error::IntegerLiteralOverflow(span))?,
                )
            } else {
                Literal::Int(
                    int.parse()
                        .map_err(|_| Error::IntegerLiteralOverflow(span))?,
                )
            }),
            Atom::Float(f) => Expr::Literal(Literal::Float(
                f.parse().map_err(|_| Error::FloatLiteralOverflow(span))?,
            )),
            Atom::Bool(b) => Expr::Literal(Literal::Bool(b)),
            Atom::Char(c) => Expr::Literal(Literal::Char(c)),
            Atom::String(s) => Expr::Literal(Literal::String(s)),
            Atom::Void => Expr::Literal(Literal::Void),
        })
    }
}