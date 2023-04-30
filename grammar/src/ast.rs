//! Models representing the abstract syntax tree.

use super::{Radix, Span};
use crate::TokenInfo;
use std::fmt::{self, Display, Formatter};

/// A compound of a span and a value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Spanned<T>(pub T, pub Span);

impl<T> Spanned<T> {
    /// Returns the value.
    #[must_use]
    pub const fn value(&self) -> &T {
        &self.0
    }

    /// Returns the span.
    #[must_use]
    pub const fn span(&self) -> Span {
        self.1
    }

    /// Consumes and maps the inner value.
    #[must_use]
    #[allow(
        clippy::missing_const_for_fn,
        reason = "closure is not supposed to be const"
    )]
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Spanned<U> {
        Spanned(f(self.0), self.1)
    }

    /// Returns a tuple (value, span).
    #[must_use]
    #[allow(clippy::missing_const_for_fn, reason = "destructors can't be const")]
    pub fn into_inner(self) -> (T, Span) {
        (self.0, self.1)
    }

    /// Returns a tuple (&value, span).
    #[must_use]
    pub const fn as_inner(&self) -> (&T, Span) {
        (&self.0, self.1)
    }
}

impl<T: Display> Display for Spanned<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// An enumeration of possible unary operators.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
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

impl Display for UnaryOp {
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
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
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
    /// The `**` operator.
    Pow,
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

impl Display for BinaryOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Mod => "%",
            Self::Pow => "**",
            Self::Eq => "==",
            Self::Ne => "!=",
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Gt => ">",
            Self::Ge => ">=",
            Self::LogicalAnd => "&&",
            Self::LogicalOr => "||",
            Self::BitAnd => "&",
            Self::BitOr => "|",
            Self::BitXor => "^",
            Self::Shl => "<<",
            Self::Shr => ">>",
        })
    }
}

/// Represents a type of delimiter.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
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
    pub const fn open(self) -> char {
        match self {
            Self::Paren => '(',
            Self::Bracket => '[',
            Self::Brace => '{',
            Self::Angle => '<',
        }
    }

    /// Returns the closing delimiter.
    #[must_use]
    pub const fn close(self) -> char {
        match self {
            Self::Paren => ')',
            Self::Bracket => ']',
            Self::Brace => '}',
            Self::Angle => '>',
        }
    }

    /// Returns the opening delimiter as a token.
    #[must_use]
    pub const fn open_token(self) -> TokenInfo {
        match self {
            Self::Paren => TokenInfo::LeftParen,
            Self::Bracket => TokenInfo::LeftBracket,
            Self::Brace => TokenInfo::LeftBrace,
            Self::Angle => TokenInfo::Lt,
        }
    }

    /// Returns the closing delimiter as a token.
    #[must_use]
    pub const fn close_token(self) -> TokenInfo {
        match self {
            Self::Paren => TokenInfo::RightParen,
            Self::Bracket => TokenInfo::RightBracket,
            Self::Brace => TokenInfo::RightBrace,
            Self::Angle => TokenInfo::Gt,
        }
    }
}

impl Display for Delimiter {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Paren => write!(f, "parenthesis"),
            Self::Bracket => write!(f, "square bracket"),
            Self::Brace => write!(f, "curly bracket"),
            Self::Angle => write!(f, "angle bracket"),
        }
    }
}

/// Parameter of a function type.
#[derive(Clone, Debug)]
pub struct FuncTypeParam {
    /// The name of the parameter if one is given.
    pub name: Option<String>,
    /// The type of the parameter.
    pub ty: TypeExpr,
    /// Whether the parameter is optional.
    pub optional: bool,
}

impl Display for FuncTypeParam {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if let Some(name) = &self.name {
            f.write_str(name)?;
            if self.optional {
                f.write_str("?")?;
            }
            f.write_str(": ")?;
        }
        write!(f, "{}", self.ty)?;
        Ok(())
    }
}

/// A type expression.
#[derive(Clone, Debug)]
pub enum TypeExpr {
    /// A type identifier, e.g. `Type`.
    Ident(String),
    /// A type path access, e.g. `Type.Member` or `module.Type`.
    Member(Box<TypeExpr>, String),
    /// A tuple type, e.g. `(Type, Type)`.
    Tuple(Vec<TypeExpr>),
    /// An array type, e.g. `Type[]`, `Type[5]`, or `Type[CONSTANT]`.
    Array(Box<TypeExpr>, Option<Box<Expr>>),
    /// A function type, e.g. `(Type, Type) -> Type`.
    Func {
        /// Positional parameters.
        args: Vec<FuncTypeParam>,
        /// Keyword-only parameters, represented as (name, type_expr, is_optional).
        kwargs: Vec<FuncTypeParam>,
        /// The return type.
        ret: Option<Box<TypeExpr>>,
    },
    /// A type union, e.g. `Type | Type`.
    Union(Vec<TypeExpr>),
    /// A product ("and") type, e.g. `Type & Type`.
    And(Vec<TypeExpr>),
    /// A negation ("not") type, e.g. `!Type`.
    Not(Box<TypeExpr>),
    /// The `to` modifier which allows any type that has a cast function to the given type,
    /// e.g. `to Type` or `to string`.
    To(Box<TypeExpr>),
    /// A generic type application, e.g. `Type<A, B>`.
    Application {
        /// The type being applied to.
        ty: Box<TypeExpr>,
        /// The standard type arguments.
        args: Vec<TypeExpr>,
        /// The associated type (keyword) arguments.
        kwargs: Vec<(String, TypeExpr)>,
    },
}

impl Display for TypeExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ident(s) => write!(f, "{s}"),
            Self::Member(ty, s) => write!(f, "{ty}.{s}"),
            Self::Tuple(tys) => {
                f.write_str("(")?;
                let tys = tys
                    .iter()
                    .map(|ty| ty.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");

                write!(f, "{tys})")
            }
            Self::Array(ty, size) => {
                write!(
                    f,
                    "{ty}[{}]",
                    size.map_or_else(|| "".to_string(), |e| e.to_string())
                )
            }
            Self::Func { args, kwargs, ret } => {
                f.write_str("(")?;

                let args = args
                    .iter()
                    .chain(kwargs.iter())
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "{args})")?;

                if let Some(ret) = ret {
                    write!(f, " -> {ret}")?;
                }
                Ok(())
            }
            Self::Union(tys) => {
                let tys = tys
                    .iter()
                    .map(|ty| ty.to_string())
                    .collect::<Vec<_>>()
                    .join(" | ");
                write!(f, "{tys}")
            }
            Self::And(tys) => {
                let tys = tys
                    .iter()
                    .map(|ty| ty.to_string())
                    .collect::<Vec<_>>()
                    .join(" & ");
                write!(f, "{tys}")
            }
            Self::Not(ty) => write!(f, "!{ty}"),
            Self::To(ty) => write!(f, "to {ty}"),
            Self::Application { ty, args, kwargs } => {
                write!(f, "{ty}<")?;

                let args = args
                    .iter()
                    .map(|ty| ty.to_string())
                    .chain(kwargs.iter().map(|(name, ty)| format!("{name} = {ty}")))
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "{args}>")
            }
        }
    }
}

/// An atom, which is an expression that cannot be further decomposed into other expressions.
///
/// For example, the literal integer 1 is an atom, but the binary operation 1 + 1 is not, since
/// it is composed of two expressions.
#[derive(Clone, Debug, Hash)]
pub enum Atom {
    /// An integer litereal.
    Int(String, Radix),
    /// A floating-point number literal.
    Float(String),
    /// A string literal. This is after resolving escape sequences.
    String(String),
    /// A boolean literal.
    Bool(bool),
    /// A non-keyword identifier.
    Ident(String),
}

impl Display for Atom {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(s, radix) => write!(
                f,
                "{s}{}",
                match radix {
                    Radix::Decimal => "",
                    Radix::Hexadecimal => "0x",
                    Radix::Octal => "0o",
                    Radix::Binary => "0b",
                }
            ),
            Self::Float(s) | Self::Ident(s) => write!(f, "{s}"),
            Self::String(s) => write!(f, "{s:?}"),
            Self::Bool(b) => write!(f, "{b}"),
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
    /// An explicit cast to a type.
    Cast {
        /// The expression being cast.
        expr: Box<Spanned<Expr>>,
        /// The type being cast to.
        ty: Box<Spanned<TypeExpr>>,
    },
    /// A function call.
    Call {
        /// The function being called.
        func: Box<Spanned<Expr>>,
        /// The positional arguments to the function.
        args: Vec<Spanned<Expr>>,
        /// The keyword arguments to the function.
        kwargs: Vec<(String, Spanned<Expr>)>,
    },
    /// A range expression.
    ///
    /// A range expression can be represented by one of the following:
    /// * `start..end` -> `[start, end)`
    /// * `start..=end` -> `[start, end]`
    /// * `..end` -> `[?, end)` (where `?` varies based on context)
    /// * `..=end` -> `[?, end]` (where `?` varies based on context)
    /// * `start..` -> `[start, ?)` (where `?` varies based on context)
    /// * `start..=` -> `[start, ?]` (where `?` varies based on context)
    /// * `..` -> `[?, ?)` (where `?` varies based on context)
    /// * `..=` -> `[?, ?]` (where `?` varies based on context)
    Range {
        /// The start value of the range. There might not be a start.
        start: Option<Box<Spanned<Expr>>>,
        /// The span of the `..` or `..=` token.
        sep: Span,
        /// Whether the range is inclusive.
        inclusive: bool,
        /// The end value of the range. There might not be an end.
        end: Option<Box<Spanned<Expr>>>,
    },
}

impl Display for Expr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Atom(a) => write!(f, "{a}"),
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
                    write!(f, "{item}")?;
                }
                write!(f, "{close}")
            }
            Self::UnaryOp { op, expr } => write!(f, "({op}{expr})"),
            Self::BinaryOp { left, op, right } => write!(f, "({left} {op} {right})"),
            Self::Attr { subject, attr, .. } => write!(f, "({subject}.{attr})"),
            Self::Cast { expr, ty } => write!(f, "({expr}::{ty})"),
            Self::Call { func, args, kwargs } => write!(
                f,
                "{func}({})",
                args.iter()
                    .map(ToString::to_string)
                    .chain(kwargs.iter().map(|(k, v)| format!("{k}: {v}")))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::Range {
                start,
                inclusive,
                end,
                ..
            } => {
                write!(
                    f,
                    "({}..{}{})",
                    start.as_ref().map(ToString::to_string).unwrap_or_default(),
                    if *inclusive { "=" } else { "" },
                    end.as_ref().map(ToString::to_string).unwrap_or_default()
                )
            }
        }
    }
}

/// A node in the abstract syntax tree. These are all statements.
pub enum Node {
    /// An expression represented as a statement.
    Expr(Spanned<Expr>),
    /// A variable declaration.
    Decl {
        /// The span of the declaration keyword (i.e. let, const).
        kw: Span,
        /// Whether the declaration is a const declaration.
        is_const: bool,
        /// If the declaration is mutable, this is the span of the mut keyword.
        mut_kw: Option<Span>,
        /// The identifier being declared. (TODO: allow destructuring)
        ident: Spanned<String>,
        /// The type of the variable, if it is specified.
        ty: Option<Box<Spanned<TypeExpr>>>,
        /// The value of the variable, if it is specified.
        /// This is always specified for const declarations.
        value: Option<Box<Spanned<Expr>>>,
    },
}

impl Display for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Expr(e) => write!(f, "{e};"),
            Self::Decl {
                is_const,
                mut_kw,
                ident,
                ty,
                value,
                ..
            } => {
                write!(f, "{}", if *is_const { "const" } else { "let" })?;
                if let Some(mut_kw) = mut_kw {
                    write!(f, " {}", mut_kw)?;
                }
                write!(f, " {ident}")?;
                if let Some(ty) = ty {
                    write!(f, ": {ty}")?;
                }
                if let Some(value) = value {
                    write!(f, " = {value}")?;
                }
                write!(f, ";")
            }
        }
    }
}
