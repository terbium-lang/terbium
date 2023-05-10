use crate::{
    error::{AstLoweringError as Error, Result},
    Expr, FieldVisibility, FloatWidth, Hir, Ident, IntSign, IntWidth, ItemId, ItemVisibility,
    Literal, ModuleId, Node, Op, Pattern, PrimitiveTy, Scope, ScopeId, StructField, StructTy, Ty,
    TyDef, TyParam,
};
use common::span::Spanned;
use grammar::ast::{StructDef, TypeExpr, TypePath};
use grammar::{ast, token::IntLiteralInfo};
use internment::Intern;
use std::collections::HashMap;

const BOOL: Ty = Ty::Primitive(PrimitiveTy::Bool);

type NamedTyDef = (ItemId, TyDef);

/// A temporary state used when lowering an AST to HIR.
#[must_use = "the HIR is only constructed when this is used"]
pub struct AstLowerer {
    scope_id: ScopeId,
    module_nodes: HashMap<ModuleId, Vec<Spanned<ast::Node>>>,
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

    fn with_ty_params(self, ty_params: Vec<TyParam>) -> Self {
        Self { ty_params, ..self }
    }
}

#[inline]
fn ty_params_into_ty(ty_params: &[TyParam]) -> Vec<Ty> {
    ty_params.iter().map(|tp| Ty::Generic(tp.name)).collect()
}

impl AstLowerer {
    /// Creates a new AST lowerer.
    pub fn new(root: Vec<Spanned<ast::Node>>) -> Self {
        Self {
            scope_id: ScopeId(1),
            module_nodes: HashMap::from([(ModuleId::root(), root)]),
            hir: Hir::default(),
        }
    }

    #[inline]
    fn get_ident(&self, ident: String) -> Ident {
        Ident(Intern::new(ident))
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
            children: vec![Node::Expr(expr)],
        })
    }

    /// Perform a pass over the AST to simply resolve all top-level types.
    pub fn resolve_top_level_types(&mut self, module: ModuleId) -> Result<()> {
        for node in self
            .module_nodes
            .get(&module)
            .cloned()
            .expect("module not found")
        {
            if let Some((item_id, ty_def)) = self.register_ty_def(module, &node)? {
                self.hir.types.insert(item_id, ty_def);
            }
        }
        Ok(())
    }

    #[inline]
    fn resolve_ty_def(
        &self,
        module: ModuleId,
        Spanned(node, _span): &Spanned<ast::Node>,
    ) -> Result<Option<NamedTyDef>> {
        // SAFETY: this is not safe at all
        let self_mut = unsafe { &mut *(self as *const Self as *mut Self) };
        Ok(match node {
            ast::Node::Struct(sct) => Some(self_mut.resolve_struct_def(module, sct)?.1),
            _ => None,
        })
    }

    #[inline]
    fn register_ty_def(
        &mut self,
        module: ModuleId,
        Spanned(node, _span): &Spanned<ast::Node>,
    ) -> Result<Option<NamedTyDef>> {
        Ok(match node {
            ast::Node::Struct(sct) => {
                let sct_name = sct.name.clone();
                let (sty, def) = self.resolve_struct_def(module, sct)?;

                if let Some(occupied) = self.hir.structs.insert(def.0, sty) {
                    return Err(Error::NameConflict(occupied.name.span(), sct_name));
                }
                Some(def)
            }
            _ => None,
        })
    }

    fn resolve_struct_def(
        &mut self,
        module: ModuleId,
        sct: &StructDef,
    ) -> Result<(StructTy, NamedTyDef)> {
        let sct_name = sct.name.clone();
        let ident = self.get_ident(sct_name.0.clone());
        let item_id = ItemId(module, ident);

        let sty = self.lower_struct_def_into_ty(module, sct.clone())?;
        let ty_params = sty.ty_params.clone();
        let gen_params = ty_params_into_ty(&ty_params);

        let def = (
            item_id,
            TyDef {
                name: sct_name.map(|ident| self.get_ident(ident)),
                ty: Ty::Struct(item_id, gen_params),
                ty_params,
            },
        );
        Ok((sty, def))
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
                name: self.get_ident(param.name.into_value()),
                bound: param
                    .bound
                    .map(|bound| self.lower_ty(&ctx, bound.into_value()))
                    .transpose()?
                    .map(Box::new),
            };
            ctx.ty_params.push(param);
        }

        // Lower all parent fields
        let mut fields = if let Some(parent) = struct_def.extends {
            if let TypeExpr::Path(path) = parent.value() {
                let path_span = path.span();
                let ty = self.lower_ty_path(&ctx, path.clone())?;
                if let Ty::Struct(sid, args) = ty {
                    let applied = self
                        .hir
                        .structs
                        .get(&sid)
                        .cloned()
                        .expect("struct not found, this is a bug")
                        .into_adhoc_struct_ty_with_applied_ty_params(Some(path_span), args)?;
                    applied.fields
                } else {
                    return Err(Error::CannotExtendFieldsFromType(parent));
                }
            } else {
                return Err(Error::CannotExtendFieldsFromType(parent));
            }
        } else {
            Vec::new()
        };

        // Lower the struct fields
        fields.extend(
            struct_def
                .fields
                .into_iter()
                .map(|Spanned(field, _)| {
                    Ok(StructField {
                        vis: FieldVisibility::from_ast(field.vis),
                        name: self.get_ident(field.name.into_value()),
                        ty: self.lower_ty(&ctx, field.ty.into_value())?,
                        default: field
                            .default
                            .map(|d| self.lower_expr(&ctx, d))
                            .transpose()?,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        );

        Ok(StructTy {
            vis: ItemVisibility::from_ast(struct_def.vis),
            name: struct_def.name.map(|name| self.get_ident(name)),
            ty_params: ctx.ty_params,
            fields,
        })
    }

    #[inline]
    fn lower_ty_or_infer(&mut self, ctx: &Ctx, ty: Option<Spanned<TypeExpr>>) -> Result<Ty> {
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
        let ident = self.get_ident(tail.clone());
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
            .map(Ok)
            // TODO: fix this line, which should allow for forward declarations (i.e. top-level structs that are declared after they are used)
            // .or_else(|| self.find_ty(lookup, span.end).transpose())
            .ok_or(Error::TypeNotFound(full_span, Spanned(tail, span), mid))??;

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

    /// Does a linear search for a type that matches the lookup.
    pub fn find_ty(&self, item_id: ItemId, offset: usize) -> Result<Option<TyDef>> {
        let module = item_id.0;
        for node in self.module_nodes.get(&module).unwrap() {
            // Skip nodes that are before the offset
            if node.span().start < offset {
                continue;
            }
            if let Ok(Some((item, def))) = self.resolve_ty_def(module, node)
                && item == item_id
            {
                return Ok(Some(def));
            }
        }
        Ok(None)
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

    /// Lowers a node into an HIR node.
    pub fn lower_node(&mut self, ctx: &Ctx, node: ast::Node) -> Result<Node> {
        use ast::Node as N;

        match node {
            N::Expr(expr) => Ok(Node::Expr(self.lower_expr(ctx, expr)?)),
            N::Let { pat, ty, value, .. } => Ok(Node::Let {
                pat: self.lower_pat(pat.into_value()),
                ty: self.lower_ty_or_infer(ctx, ty)?,
                value: value.map(|value| self.lower_expr(ctx, value)).transpose()?,
            }),
            _ => todo!(),
        }
    }

    /// Lowers a pattern into an HIR pattern.
    pub fn lower_pat(&mut self, pat: ast::Pattern) -> Pattern {
        match pat {
            ast::Pattern::Ident { ident, mut_kw } => Pattern::Ident {
                ident: self.get_ident(ident.into_value()),
                is_mut: mut_kw.is_some(),
            },
            _ => todo!(),
        }
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
            E::Block { label, body } => Expr::Block(self.register_scope(Scope {
                label: label.map(|l| self.get_ident(l.into_value())),
                children: todo!(),
            })),
            _ => todo!(),
        })
    }

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
    /// For example, `a || b` becomes `if a::bool { a } else { b }`.
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
        let cond = Box::new(Expr::Cast(Box::new(lhs.clone()), BOOL));

        let lhs = self.anon_scope_from_expr(lhs);
        let rhs = self.anon_scope_from_expr(rhs);

        Ok(match op {
            ast::BinaryOp::LogicalOr => Expr::If(cond, lhs, rhs),
            ast::BinaryOp::LogicalAnd => Expr::If(cond, rhs, lhs),
            _ => unimplemented!("logical op is not implemented for this operator"),
        })
    }

    /// Lowers an atom into an HIR literal expression.
    pub fn lower_atom(&mut self, ctx: &Ctx, atom: Spanned<ast::Atom>) -> Result<Expr> {
        use ast::Atom as A;

        let (atom, span) = atom.into_inner();
        Ok(match atom {
            A::Ident(ident) => self.lower_ident_expr(ctx, self.get_ident(ident)),
            A::Int(int, IntLiteralInfo { unsigned, .. }) => Expr::Literal(if unsigned {
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
            A::Float(f) => Expr::Literal(Literal::Float(
                f.parse().map_err(|_| Error::FloatLiteralOverflow(span))?,
            )),
            A::Bool(b) => Expr::Literal(Literal::Bool(b)),
            A::Char(c) => Expr::Literal(Literal::Char(c)),
            A::String(s) => Expr::Literal(Literal::String(s)),
            A::Void => Expr::Literal(Literal::Void),
        })
    }
}
