use crate::{Expr, FloatWidth, Hir, Ident, IntSign, IntWidth, ItemId, Literal, ModuleId, Node, Op, Pattern, PrimitiveTy, Scope, ScopeId, StructTy, Ty, error::{AstLoweringError as Error, Result}};
use grammar::{ast, token::IntLiteralInfo};
use grammar::ast::{StructDef, TypeExpr, TypePath};
use internment::Intern;
use common::span::Spanned;

const BOOL: Ty = Ty::Primitive(PrimitiveTy::Bool);

/// A temporary state used when lowering an AST to HIR.
#[must_use = "the HIR is only constructed when this is used"]
pub struct AstLowerer {
    scope_id: ScopeId,
    /// The HIR being constructed.
    pub hir: Hir,
}

impl AstLowerer {
    /// Creates a new AST lowerer.
    pub fn new() -> Self {
        Self {
            scope_id: ScopeId(0),
            hir: Hir::default(),
        }
    }

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
    pub fn resolve_top_level_types(&mut self, module: ModuleId, nodes: Vec<ast::Node>) {
        for node in nodes {
            match node {
                ast::Node::Struct(sct) => {
                    let ident = self.get_ident(sct.name.into_value());
                    let item_id = ItemId(module, ident);

                }
            }
        }
    }

    pub fn lower_struct_def_into_ty(&mut self, struct_def: StructDef) -> Result<StructTy> {
        let mut fields = if let Some(ref parent) = struct_def.extends {
            if let TypeExpr::Type(name, params) = parent.value() {

            } else {
                return Err(Error::StructNotDeclared())
            }
        }
    }

    /// Lowers a node into an HIR node.
    pub fn lower_node(&mut self, node: ast::Node) -> Node {
        use ast::Node as N;

        match node {
            N::Expr(expr) => Node::Expr(self.lower_expr(expr.into_value())),
            N::Let { pat, ty, value, .. } => Node::Let {
                pat: self.lower_pat(pat.into_value()),
                ty: self.lower_ty(ty.into_value()),
            },
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

    /// Lowers a type expression into an HIR type.
    pub fn lower_ty(&mut self, ty: TypeExpr) -> Ty {
        match ty {
            TypeExpr::Infer => Ty::Unknown,
            TypeExpr::Path(path) => self.lower_ty_path(path),
        }
    }

    pub fn lower_ty_path(&mut self, module: ModuleId, TypePath(segments): TypePath) -> Result<Ty> {
        // Add as many segments as possible to the module path
        let mut ty_module = Vec::with_capacity(segments.len());
        let segments = segments.into_iter();

        for Spanned(segment, _) in segments {
            // Always favor primitive types over anything else
            if let Some(ty) = Self::lower_ty_ident_into_primitive(segment.0.value()) {
                return Ok(Ty::Primitive(ty));
            }

            ty_module.push(segment.0.into_value());
            if segment.1.is_some() {
                break;
            }
            let interned = Intern::new(ty_module.clone());
            if !self.hir.modules.contains_key(&ModuleId(interned)) {
                break;
            }
        }

        // UNWRAP: We know that we have at least one segment
        let ident = ty_module.pop().unwrap();
        let mid = if ty_module.is_empty() {
            // If we have no segments, then we have an empty path. Look in the current module.
            module
        } else {
            ModuleId(Intern::new(ty_module))
        };
        let lookup = ItemId(mid, self.get_ident(ident));
        self.hir.types.get(&lookup).cloned().ok_or(Error::TypeNotFound(lookup))
    }

    pub const fn lower_ty_ident_into_primitive(s: &str) -> Option<PrimitiveTy> {
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

    /// Lowers an expression into an HIR node.
    pub fn lower_expr(&mut self, expr: Spanned<ast::Expr>) -> Result<Expr> {
        use ast::Expr as E;
        use ast::UnaryOp as U;

        let (expr, span) = expr.into_inner();
        Ok(match expr {
            E::Atom(atom) => self.lower_atom(Spanned(atom, span))?,
            E::UnaryOp { op, expr } => Expr::CallOp(
                match *op.value() {
                    U::Plus => Op::Pos,
                    U::Minus => Op::Neg,
                    U::BitNot => Op::BitNot,
                    U::Not => Op::Not,
                },
                Box::new(self.lower_expr(*expr)?),
                Vec::new(),
            ),
            E::BinaryOp { left, op, right } => {
                if let Some(op) = Self::lower_bin_op(*op.value()) {
                    Expr::CallOp(
                        op,
                        Box::new(self.lower_expr(*left)?),
                        vec![self.lower_expr(*right)?],
                    )
                } else {
                    self.lower_logical_into_if_stmt(*left, *right, op.into_value())?
                }
            }
            E::Tuple(exprs) => Expr::Tuple(
                exprs
                    .into_iter()
                    .map(|e| self.lower_expr(e))
                    .collect::<Result<Vec<_>>>()?,
            ),
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
        lhs: Spanned<ast::Expr>,
        rhs: Spanned<ast::Expr>,
        op: ast::BinaryOp,
    ) -> Result<Expr> {
        let lhs = self.lower_expr(lhs)?;
        let rhs = self.lower_expr(rhs)?;
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
    pub fn lower_atom(&mut self, atom: Spanned<ast::Atom>) -> Result<Expr> {
        use ast::Atom as A;

        let (atom, span) = atom.into_inner();
        Ok(match atom {
            A::Ident(ident) => Expr::Ident(self.get_ident(ident)),
            A::Int(int, IntLiteralInfo { unsigned, .. }) => Expr::Literal(if unsigned {
                Literal::UInt(int.parse().map_err(|_| Error::IntegerLiteralOverflow(span))?)
            } else {
                Literal::Int(int.parse().map_err(|_| Error::IntegerLiteralOverflow(span))?)
            }),
            A::Float(f) => Expr::Literal(Literal::Float(f.parse().map_err(|_| Error::FloatLiteralOverflow(span))?)),
            A::Bool(b) => Expr::Literal(Literal::Bool(b)),
            A::Char(c) => Expr::Literal(Literal::Char(c)),
            A::String(s) => Expr::Literal(Literal::String(s)),
            A::Void => Expr::Literal(Literal::Void),
        })
    }
}
