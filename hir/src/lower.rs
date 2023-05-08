use crate::{Expr, Hir, Literal, Node};
use grammar::{ast, token::IntLiteralInfo};
use internment::Intern;

/// A temporary state used when lowering an AST to HIR.
pub struct AstLowerer {
    /// The HIR being constructed.
    pub hir: Hir,
    /// The current scope.
    pub scope: ScopeId,
}

impl AstLowerer {
    fn get_ident(&self, ident: String) -> Ident {
        Intern::new(ident)
    }

    /// Lowers a node into an HIR node.
    pub fn lower_node(&mut self, node: ast::Node) -> Node {
        use ast::Node as N;

        match node {
            N::Expr(expr) => Node::Expr(self.lower_expr(expr)),
        }
    }

    /// Lowers an expression into an HIR node.
    pub fn lower_expr(&mut self, expr: ast::Expr) -> Expr {
        use ast::Expr as E;

        match expr {
            E::Atom(atom) => self.lower_atom(atom),
            E::UnaryOp(op, expr) =>
        }
    }

    /// Lowers an atom into an HIR literal expression.
    pub fn lower_atom(&mut self, atom: ast::Atom) -> Expr {
        use ast::Atom as A;

        match atom {
            A::Ident(ident) => Expr::Ident(self.get_ident(ident)),
            // For ints and floats, we use `.unwrap_or(0)` because:
            // - the integer literal has already been validated by the tokenizer
            // - to avoid panicking if for some bizzare reason, the literal is invalid
            A::Int(int, IntLiteralInfo { unsigned, .. }) => {
                Expr::Literal(if unsigned {
                    Literal::UInt(int.parse().unwrap_or(0))
                } else {
                    Literal::Int(int.parse().unwrap_or(0))
                })
            }
            A::Float(f) => Expr::Literal(Literal::Float(f.parse().unwrap_or(0.0))),
            A::Bool(b) => Expr::Literal(Literal::Bool(b)),
            A::Char(c) => Expr::Literal(Literal::Char(c)),
            A::String(s) => Expr::Literal(Literal::String(s)),
            A::Void => Expr::Literal(Literal::Void),
        }
    }
}
