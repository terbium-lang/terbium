//! Models representing the abstract syntax tree.

use crate::{
    span::{Span, Spanned},
    token::{Radix, TokenInfo},
};
use std::fmt::{self, Display, Formatter};

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
#[derive(Clone, Debug, PartialEq, Eq)]
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
#[derive(Clone, Debug, PartialEq, Eq)]
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
        params: Vec<FuncTypeParam>,
        /// Keyword-only parameters, represented as (name, type_expr, is_optional).
        kw_params: Vec<FuncTypeParam>,
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
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");

                write!(f, "{tys})")
            }
            Self::Array(ty, size) => {
                write!(
                    f,
                    "{ty}[{}]",
                    size.as_ref().map_or_else(String::new, ToString::to_string)
                )
            }
            Self::Func {
                params,
                kw_params,
                ret,
            } => {
                f.write_str("(")?;

                let params = params
                    .iter()
                    .chain(kw_params.iter())
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "{params})")?;

                if let Some(ret) = ret {
                    write!(f, " -> {ret}")?;
                }
                Ok(())
            }
            Self::Union(tys) => {
                let tys = tys
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(" | ");
                write!(f, "{tys}")
            }
            Self::And(tys) => {
                let tys = tys
                    .iter()
                    .map(ToString::to_string)
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
                    .map(ToString::to_string)
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
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
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
                "{}{s}",
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

/// The assignment target of an assignment expression.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AssignmentTarget {
    /// A pattern match target.
    Pattern(Pattern),
    /// An attribute access. Note that this overrides variant patterns such as `Enum.Variant`;
    /// that will parse as Attr { subject: Enum, attr: "Variant" }.
    Attr {
        /// The subject of the attribute access.
        subject: Box<Spanned<Expr>>,
        /// The name of the attribute.
        attr: Spanned<String>,
    },
    /// An index access.
    Index {
        /// The subject of the index access.
        subject: Box<Spanned<Expr>>,
        /// The index of the index access.
        index: Box<Spanned<Expr>>,
    },
}

/// A pattern-match target used when matching values, including in function parameters and in
/// declaration bindings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pattern {
    /// An identifier binding, likely a variable.
    Ident(String),
    /// A tuple of patterns, matching any tuples.
    ///
    /// When accessing tuple enum variants, a variant name must be specified, e.g.
    /// `let Enum.Variant(a) = Enum.Variant(a)`.
    Tuple(Vec<Spanned<String>>, Vec<Spanned<Self>>),
    /// A destructuring of a field-based object, e.g. `let {a, b} = Struct {a: 1, b: 2}`.
    ///
    /// They may also bind to a new pattern, e.g. `let {a: b} = Struct {a: 1}` binds `b` to `1`,
    /// and `let {a: (b, c)} = Struct {a: (1, 2)}` binds `b` to `1` and `c` to `2`.
    ///
    /// Field-patterns may only bind to field-based types. If the fields are enclosed in an enum
    /// variant, the enum variant must be specified, e.g.
    /// `let Enum.Variant {a} = Enum.Variant {a: 1}`.
    Fields(
        Vec<Spanned<String>>,
        Vec<(Spanned<String>, Option<Box<Spanned<Self>>>)>,
    ),
    /// An array (like) of patterns, matching any array-like objects
    /// (such as arrays, but also lists).
    ///
    /// If the size of the array is known, all elements must be present. If not, there must
    /// be a wildcard pattern to match unmatched elements in the array.
    Array(Vec<Spanned<Self>>),
    /// A binding to any pattern match. For example, `let (a, b) as c = (1, 2)` binds `c` to
    /// `(1, 2)`. This is represented as `As(lhs, rhs)`.
    ///
    /// This may also be extended further, e.g. `let (a, (b, c) as d) = (1, (2, 3))` binds
    /// `(2, 3)` to `d`.
    ///
    /// This is useful for binding to a pattern match while also binding to the pattern match's
    /// value.
    As(Box<Spanned<Self>>, Box<Spanned<Self>>),
    /// The wildcard pattern `*`, which matches anything. An optional pattern can be provided after
    /// to the wildcard to bind further.
    ///
    /// For example, `let (a, *) = (1, 2, 3)` binds `a` to `1` and ignores the rest of the tuple.
    /// `let (a, *b) = (1, 2, 3)` binds `a` to `1` and `b` to `(2, 3)`.
    ///
    /// `let (a, *(b, c)) = (1, 2, 3)` is the equivalent of `let (a, b, c) = (1, 2, 3)` because
    /// `(b, c)` is binded to the rest of the tuple `(2, 3)`.
    ///
    /// Wildcard patterns are only supported in tuple, array, and field patterns.
    Wildcard(Option<Box<Spanned<Self>>>),
}

/// An extended pattern used specifically in a match expression. The extra bindings here
/// deliberately match values but do not bind, so it wouldn't make sense to have in declarations
/// `let` (e.g. `let 1 = 1` is just dumb). This is a superset of the `Pattern` enum.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatchPattern {
    /// A standard pattern. The `Pattern::As` variant is overriden by the `MatchPattern::As`
    /// variant and will never be used.
    Pattern(Pattern),
    /// An enum variant, e.g. `Enum.Variant`. This is never a binding; in assignment targets this
    /// is always overridden by an attribute access.
    Variant(Vec<Spanned<String>>),
    /// A literal atom, e.g. `1`, `"hello"`, or `true`.
    Atom(Atom),
    /// A pattern that matches within a range, e.g. `1..=10`. The left and right bounds are
    /// optional and can be omitted, e.g. `..=10` or `1..`. The bounds can only be atoms.
    Range {
        /// The left bound of the range.
        start: Option<Spanned<Atom>>,
        /// The span of the `..` or `..=` token.
        sep: Span,
        /// Whether the range is inclusive.
        inclusive: bool,
        /// The right bound of the range.
        end: Option<Box<Atom>>,
    },
    /// A binding to any pattern match; see `Pattern::As` but matches the extend variants here.
    /// For example, matching `5` against `1..10 as x` will succeed and bind `x` to `5`.
    ///
    /// Note that the actual binding can only within be the standard subset of [`Pattern`] because
    /// the extended variants here are not valid in declarations. (`1..10 as 5` does not make sense)
    As(Box<Spanned<Self>>, Spanned<Pattern>),
    /// A pattern that matches one of a set of patterns, e.g. `1 | 2 | 3`.
    /// If the patterns have bindings, all bindings must be the same and have the same type.
    ///
    /// For example, given an `(int, string, int)`, the pattern `(a, *) | (*, a)` is valid, but:
    /// * `(a, *) | (*, b)` is invalid because the bindings have different names.
    /// * `(a, *) | (*, a, *)` is invalid because the bindings have different types.
    /// * `(a, *) | *` is invalid because the binding `a` only exists in one of the patterns.
    ///
    /// Note that the `|` operator takes lower precedence over `as`, so `a | b as c` is equivalent
    /// to `a | (b as c)` (which is obviously invalid).
    OneOf(Vec<Spanned<Self>>),
}

/// An expression that can be evaluated to a value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expr {
    /// An atom represented as an expression.
    Atom(Atom),
    /// A tuple of expressions.
    Tuple(Vec<Spanned<Self>>),
    /// An array of expressions.
    Array(Vec<Spanned<Self>>),
    /// A unary operation.
    UnaryOp {
        /// The operator.
        op: Spanned<UnaryOp>,
        /// The operand.
        expr: Box<Spanned<Self>>,
    },
    /// A binary operation.
    BinaryOp {
        /// The left operand.
        left: Box<Spanned<Self>>,
        /// The operator.
        op: Spanned<BinaryOp>,
        /// The right operand.
        right: Box<Spanned<Self>>,
    },
    /// Attribute access via dot notation.
    Attr {
        /// The object being accessed.
        subject: Box<Spanned<Self>>,
        /// The span of the dot.
        dot: Span,
        /// The attribute being accessed.
        attr: Spanned<String>,
    },
    /// An explicit cast to a type.
    Cast {
        /// The expression being cast.
        expr: Box<Spanned<Self>>,
        /// The type being cast to.
        ty: Spanned<TypeExpr>,
    },
    /// A function call.
    Call {
        /// The function being called.
        func: Box<Spanned<Self>>,
        /// The positional arguments to the function.
        args: Vec<Spanned<Self>>,
        /// The keyword arguments to the function.
        kwargs: Vec<(String, Spanned<Self>)>,
    },
    /// An index, sometimes called a "subscript", e.g. `a[b]`.
    Index {
        /// The object being indexed.
        subject: Box<Spanned<Self>>,
        /// The index being accessed.
        index: Box<Spanned<Self>>,
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
        start: Option<Box<Spanned<Self>>>,
        /// The span of the `..` or `..=` token.
        sep: Span,
        /// Whether the range is inclusive.
        inclusive: bool,
        /// The end value of the range. There might not be an end.
        end: Option<Box<Spanned<Self>>>,
    },
    /// An if-statement. This is an expression in an AST because divergent if-statements
    /// are expressions.
    If {
        /// The condition of the if-statement.
        cond: Box<Spanned<Self>>,
        /// The body of the if-statement.
        body: Vec<Spanned<Node>>,
        /// The body of the else block. This is `None` if there is no else block.
        else_body: Option<Vec<Spanned<Node>>>,
        /// Whether the if-statement is a ternary-if expression.
        /// (i.e. an inline "if-then-else" expression).
        ///
        /// If this is `true`, then:
        /// * `else_body` will always be `Some`
        /// * `body` will always have exactly one `Expr` node
        /// * `else_body` will always have exactly one `Expr` node
        ternary: bool,
    },
    /// A while-loop. This is an expression in an AST because divergent while-loops
    /// are expressions.
    While {
        /// The condition of the while-loop.
        cond: Box<Spanned<Self>>,
        /// The body of the while-loop.
        body: Vec<Spanned<Node>>,
        /// The body of the else block. This is `None` if there is no else block.
        /// The else-block is executed if the while-loop finishes execution without a break.
        ///
        /// While-loops without else-blocks are considered non-diverging and should not be
        /// considered as expressions.
        else_body: Option<Vec<Spanned<Node>>>,
    },
    /// A loop expression. Loop expressions either never diverge due to infinite loops,
    /// or always diverge due to a break statement. Therefore, they can also be considered as
    /// expressions.
    Loop(Vec<Spanned<Node>>),
    /// A when expression. When expressions can simplify long if-else if chains into a mapping of
    /// conditions to corresponding values.
    ///
    /// For example, `when { x == 0 -> 0, x < 0 -> -1, else 1 }` is a valid expression and is
    /// equivalent to `if x == 0 then 0 else if x < 0 then -1 else 1`.
    When {
        /// The condition-value pairs of the when expression.
        arms: Vec<(Spanned<Self>, Spanned<Self>)>,
        /// The else value of the when expression. This is `None` if there is no else value.
        else_value: Option<Box<Spanned<Self>>>,
    },
}

trait Indent {
    /// Indent all lines of the string by 4 spaces.
    fn write_indent(&self, f: &mut Formatter<'_>) -> fmt::Result;
}

impl<T: ToString> Indent for T {
    fn write_indent(&self, f: &mut Formatter<'_>) -> fmt::Result {
        const INDENT: &str = "    ";
        let s = self.to_string();

        for (i, line) in s.lines().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{INDENT}{line}")?;
        }
        Ok(())
    }
}

impl Display for Expr {
    #[allow(clippy::too_many_lines)]
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
            Self::Index { subject, index } => write!(f, "({subject}[{index}])"),
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
            Self::If {
                cond,
                body,
                else_body,
                ternary,
            } => {
                write!(f, "if {cond} ")?;
                if *ternary {
                    f.write_str("then ")?;
                    for node in body {
                        write!(f, "{node}")?;
                    }
                    f.write_str(" else ")?;
                    for node in else_body.as_ref().expect("ternary if must have else body") {
                        write!(f, "{node}")?;
                    }
                    return Ok(());
                }

                f.write_str("{\n")?;

                for node in body {
                    node.write_indent(f)?;
                }
                f.write_str("\n}")?;

                if let Some(else_body) = else_body {
                    f.write_str(" else ")?;
                    if !*ternary {
                        f.write_str("{\n")?;
                    }
                    for node in else_body {
                        node.write_indent(f)?;
                    }
                    f.write_str("\n}")?;
                }
                Ok(())
            }
            Self::While {
                cond,
                body,
                else_body,
            } => {
                writeln!(f, "while {cond} {{")?;
                for node in body {
                    node.write_indent(f)?;
                }

                if let Some(else_body) = else_body {
                    f.write_str(" else {\n")?;
                    for node in else_body {
                        node.write_indent(f)?;
                    }
                }
                Ok(())
            }
            Self::Loop(body) => {
                writeln!(f, "loop {{")?;
                for node in body {
                    node.write_indent(f)?;
                }
                f.write_str("\n}")
            }
            Self::When { arms, else_value } => {
                f.write_str("when {\n")?;
                for (cond, value) in arms {
                    format!("{cond} -> {value}").write_indent(f)?;
                }
                if let Some(else_value) = else_value {
                    format!("else -> {else_value}").write_indent(f)?;
                }
                f.write_str("\n}")
            }
        }
    }
}

/// Information about a function parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FuncParam {
    /// The name of the parameter. (TODO: destructuring)
    pub name: Spanned<String>,
    /// The type of the parameter.
    pub ty: Spanned<TypeExpr>,
    /// The default value of the parameter, if it is specified.
    pub default: Option<Box<Spanned<Expr>>>,
}

impl Display for FuncParam {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.ty)?;
        if let Some(default) = &self.default {
            write!(f, " = {default}")?;
        }
        Ok(())
    }
}

/// A node in the abstract syntax tree.
#[derive(Clone, Debug, PartialEq, Eq)]
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
        ty: Option<Spanned<TypeExpr>>,
        /// The value of the variable, if it is specified.
        /// This is always specified for const declarations.
        value: Option<Spanned<Expr>>,
    },
    /// An explicit return statement. These return from the closest function.
    Return {
        /// The span of the return keyword.
        kw: Span,
        /// The value being returned, if it is specified.
        value: Option<Spanned<Expr>>,
        /// The condition on whether to return.
        /// This is only specified for `return if` statements.
        cond: Option<Spanned<Expr>>,
    },
    /// An implicit return statement. These return from the closest block.
    ImplicitReturn(Spanned<Expr>),
    /// Break statement.
    Break {
        /// The span of the break keyword.
        kw: Span,
        /// The value to break with, if specified.
        value: Option<Spanned<Expr>>,
        /// The condition on whether to break.
        /// This is only specified for `break if` statements.
        cond: Option<Spanned<Expr>>,
    },
    /// Continue statement.
    Continue {
        /// The span of the continue keyword.
        kw: Span,
        /// The condition on whether to continue.
        /// This is only specified for `continue if` statements.
        cond: Option<Spanned<Expr>>,
    },
    /// A named function declaration.
    Func {
        /// The span of the "func" keyword.
        kw: Span,
        /// The name of the function.
        name: Spanned<String>,
        /// The positional parameters of the function.
        params: Vec<FuncParam>,
        /// The keyword parameters of the function.
        kw_params: Vec<FuncParam>,
        /// The return type of the function, if it is specified.
        ret: Option<Spanned<TypeExpr>>,
        /// The body of the function.
        body: Vec<Self>,
    },
}

impl Display for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        fn write_control_flow_stmt(
            f: &mut Formatter,
            kw: &str,
            value: &Option<Spanned<Expr>>,
            cond: &Option<Spanned<Expr>>,
        ) -> fmt::Result {
            f.write_str(kw)?;
            if let Some(value) = value {
                write!(f, " {value}")?;
            }
            if let Some(cond) = cond {
                write!(f, " if {cond}")?;
            }
            write!(f, ";")
        }

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
                if mut_kw.is_some() {
                    f.write_str(" mut")?;
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
            Self::Return { value, cond, .. } => write_control_flow_stmt(f, "return", value, cond),
            Self::ImplicitReturn(e) => write!(f, "{e}"),
            Self::Break { value, cond, .. } => write_control_flow_stmt(f, "break", value, cond),
            Self::Continue { cond, .. } => {
                f.write_str("continue")?;
                if let Some(cond) = cond {
                    write!(f, " if {cond}")?;
                }
                write!(f, ";")
            }
            Self::Func {
                name,
                params,
                kw_params,
                ret,
                body,
                ..
            } => {
                write!(f, "func {name}(")?;

                let params = params
                    .iter()
                    .map(ToString::to_string)
                    .chain(kw_params.is_empty().then(|| "*".to_string()))
                    .chain(kw_params.iter().map(ToString::to_string))
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "{params})")?;

                if let Some(ret) = ret {
                    write!(f, " -> {ret}")?;
                }
                writeln!(f, " {{")?;
                for node in body {
                    node.write_indent(f)?;
                }
                f.write_str("\n}")
            }
        }
    }
}
