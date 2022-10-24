//! Models representing the abstract syntax tree.

use super::Span;

/// A compound of a span and a value.
pub struct Spanned<T>(pub T, pub Span);

impl<T> Spanned<T> {
    /// Returns the value.
    #[must_use]
    pub fn value(&self) -> &T {
        &self.0
    }

    /// Returns the span.
    #[must_use]
    pub fn span(&self) -> Span {
        self.1
    }

    /// Returns a tuple (value, span).
    #[must_use]
    pub fn into_inner(self) -> (T, Span) {
        (self.0, self.1)
    }

    /// Returns a tuple (&value, span).
    #[must_use]
    pub fn as_inner(&self) -> (&T, Span) {
        (&self.0, self.1)
    }
}

/// An enumeration of possible unary operators.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    /// The `+` operator.
    Plus,
    /// The `-` operator.
    Minus,
    /// The `!` operator.
    Not,
}

/// An enumeration of possible binary operators.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    /// The `+` operator.
    Add,
    /// The `-` operator.
    Sub,
    /// The `*` operator.
    Mul,
    /// The `/` operator.
    Div,
    /// The `%` operator.
    Mod,
    /// The `==` operator.
    Eq,
    /// The `!=` operator.
    Ne,
    /// The `<` operator.
    Lt,
    /// The `<=` operator.
    Le,
    /// The `>` operator.
    Gt,
    /// The `>=` operator.
    Ge,
    /// The `&&` operator.
    LogicalAnd,
    /// The `||` operator.
    LogicalOr,
    /// The `&` operator.
    BitAnd,
    /// The `|` operator.
    BitOr,
    /// The `^` operator.
    BitXor,
    /// The `<<` operator.
    Shl,
    /// The `>>` operator.
    Shr,
}

/// An expression that can be evaluated to a value.
pub enum Expr {
    /// An integer.
    Int(String),
    /// A floating-point number.
    Float(String),
    /// A string. For string literals, this is after resolving escape sequences.
    String(String),
    /// A boolean.
    Bool(bool),
    /// A non-keyword identifier.
    Ident(String),
    /// A tuple of expressions.
    Tuple(Vec<Spanned<Expr>>),
    /// An array of expressions.
    Array(Vec<Spanned<Expr>>),
    /// A unary operation.
    UnaryOp {
        /// The operator.
        op: Spanned<UnaryOp>,
        /// The operand.
        expr: Box<Spanned<Expr>>,
    },
    /// A binary operation.
    BinaryOp {
        /// The left operand.
        left: Box<Spanned<Expr>>,
        /// The operator.
        op: Spanned<BinaryOp>,
        /// The right operand.
        right: Box<Spanned<Expr>>,
    },
    /// Attribute access via dot notation.
    Attr {
        /// The object being accessed.
        subject: Box<Spanned<Expr>>,
        /// The span of the dot.
        dot: Span,
        /// The attribute being accessed.
        attr: String,
    },
}

/// A node in the abstract syntax tree. These are all statements.
pub enum Node {
    /// An expression.
    Expr(Spanned<Expr>),
}
