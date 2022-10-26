//! Models representing the abstract syntax tree.

use super::{Radix, Span};
use std::fmt::{self, Formatter};

/// A compound of a span and a value.
#[derive(Clone, Debug)]
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

    /// Consumes and maps the inner value.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Spanned<U> {
        Spanned(f(self.0), self.1)
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

impl<T: fmt::Display> fmt::Display for Spanned<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
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
    /// The `~` operator.
    BitNot,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plus => write!(f, "+"),
            Self::Minus => write!(f, "-"),
            Self::Not => write!(f, "!"),
            Self::BitNot => write!(f, "~"),
        }
    }
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

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Add => write!(f, "+"),
            Self::Sub => write!(f, "-"),
            Self::Mul => write!(f, "*"),
            Self::Div => write!(f, "/"),
            Self::Mod => write!(f, "%"),
            Self::Eq => write!(f, "=="),
            Self::Ne => write!(f, "!="),
            Self::Lt => write!(f, "<"),
            Self::Le => write!(f, "<="),
            Self::Gt => write!(f, ">"),
            Self::Ge => write!(f, ">="),
            Self::LogicalAnd => write!(f, "&&"),
            Self::LogicalOr => write!(f, "||"),
            Self::BitAnd => write!(f, "&"),
            Self::BitOr => write!(f, "|"),
            Self::BitXor => write!(f, "^"),
            Self::Shl => write!(f, "<<"),
            Self::Shr => write!(f, ">>"),
        }
    }
}

/// Represents a type of delimiter.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Delimiter {
    /// The `(` delimiter.
    Paren,
    /// The `[` delimiter.
    Bracket,
    /// The `{` delimiter.
    Brace,
    /// The `<` delimiter.
    Angle,
}

impl Delimiter {
    /// Returns the opening delimiter.
    #[must_use]
    pub fn open(self) -> char {
        match self {
            Self::Paren => '(',
            Self::Bracket => '[',
            Self::Brace => '{',
            Self::Angle => '<',
        }
    }

    /// Returns the closing delimiter.
    #[must_use]
    pub fn close(self) -> char {
        match self {
            Self::Paren => ')',
            Self::Bracket => ']',
            Self::Brace => '}',
            Self::Angle => '>',
        }
    }
}

impl fmt::Display for Delimiter {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Paren => write!(f, "("),
            Self::Bracket => write!(f, "["),
            Self::Brace => write!(f, "{{"),
            Self::Angle => write!(f, "<"),
        }
    }
}

/// An atom, which is an expression that cannot be further decomposed into other expressions.
///
/// For example, the literal integer 1 is an atom, but the binary operation 1 + 1 is not, since
/// it is composed of two expressions.
#[derive(Clone, Debug)]
pub enum Atom {
    /// An integer.
    Int(String, Radix),
    /// A floating-point number.
    Float(String),
    /// A string. For string literals, this is after resolving escape sequences.
    String(String),
    /// A boolean.
    Bool(bool),
    /// A non-keyword identifier.
    Ident(String),
}

impl fmt::Display for Atom {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(s, radix) => write!(
                f,
                "{}{}",
                s,
                match radix {
                    Radix::Decimal => "",
                    Radix::Hexadecimal => "0x",
                    Radix::Octal => "0o",
                    Radix::Binary => "0b",
                }
            ),
            Self::Float(s) | Self::Ident(s) => write!(f, "{}", s),
            Self::String(s) => write!(f, "{:?}", s),
            Self::Bool(b) => write!(f, "{}", b),
        }
    }
}

/// An expression that can be evaluated to a value.
#[derive(Clone, Debug)]
pub enum Expr {
    /// An atom represented as an expression.
    Atom(Atom),
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

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Atom(a) => write!(f, "{}", a),
            Self::Tuple(items) | Self::Array(items) => {
                if matches!(self, Self::Tuple(_)) && items.len() == 1 {
                    return write!(f, "({},)", items[0]);
                }

                let (open, close) = if matches!(self, Self::Tuple(_)) {
                    ("(", ")")
                } else {
                    ("[", "]")
                };
                write!(f, "{}", open)?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "{}", close)
            }
            Self::UnaryOp { op, expr } => write!(f, "({}{})", op, expr),
            Self::BinaryOp { left, op, right } => write!(f, "({} {} {})", left, op, right),
            Self::Attr { subject, attr, .. } => write!(f, "({}.{})", subject, attr),
        }
    }
}

/// A node in the abstract syntax tree. These are all statements.
pub enum Node {
    /// An expression represented as a statement.
    Expr(Spanned<Expr>),
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Expr(e) => write!(f, "{};", e),
        }
    }
}
