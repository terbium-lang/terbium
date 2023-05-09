//! Models representing the abstract syntax tree.

use crate::{
    error::Error,
    span::{Span, Spanned},
    token::{IntLiteralInfo, TokenInfo},
};
use std::fmt::{self, Debug, Display, Formatter};

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
    pub ty: Spanned<TypeExpr>,
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

/// A generic type application, e.g. `Type<A, B>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeApplication {
    /// The standard type arguments.
    pub args: Vec<Spanned<TypeExpr>>,
    /// The associated type (keyword) arguments.
    pub kwargs: Vec<(Spanned<String>, Spanned<TypeExpr>)>,
}

impl Display for TypeApplication {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let args = self
            .args
            .iter()
            .map(ToString::to_string)
            .chain(
                self.kwargs
                    .iter()
                    .map(|(name, ty)| format!("{name} = {ty}")),
            )
            .collect::<Vec<_>>()
            .join(", ");

        write!(f, "<{args}>")
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypePathSeg(pub Spanned<String>, pub Option<Spanned<TypeApplication>>);

impl Display for TypePathSeg {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)?;
        if let Some(app) = &self.1 {
            write!(f, "{}", app)?;
        }
        Ok(())
    }
}

/// A type path, e.g. `Type`, `Type.Member`, or `module.Type`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypePath(pub Vec<Spanned<TypePathSeg>>);

impl Display for TypePath {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            self.0
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(".")
        )
    }
}

/// A type qualification, i.e. the `<Type: Trait>` in `<Type: Trait>.AssociatedType`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(unused)] // TODO: implement qualified types
pub struct TypeQual {
    /// The type being qualified.
    pub ty: Spanned<Box<TypeExpr>>,
    /// The target trait.
    pub r#trait: Spanned<TypePath>,
}

/// A type expression.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypeExpr {
    /// An inference placeholder, i.e. `_`.
    Infer,
    /// A standalone type path, e.g. `Type`, `Type.Member`, or `module.Type`.
    /// This may include a type application, e.g. `Type<A, B>` or `module.Type<A>.AssociatedType`.
    Path(Spanned<TypePath>),
    /// A tuple type, e.g. `(Type, Type)`.
    Tuple(Vec<Spanned<TypeExpr>>),
    /// An array type, e.g. `[Type]`, `[Type; 5]`, or `[Type; CONSTANT]`.
    Array(Box<Spanned<TypeExpr>>, Option<Spanned<Atom>>),
    /// A function type, e.g. `(Type, Type) -> Type`.
    Func {
        /// Positional parameters.
        params: Vec<FuncTypeParam>,
        /// Keyword-only parameters, represented as (name, type_expr, is_optional).
        kw_params: Vec<FuncTypeParam>,
        /// The return type.
        ret: Option<Box<Spanned<TypeExpr>>>,
    },
    /// A type union, e.g. `Type | Type`.
    Union(Vec<Spanned<TypeExpr>>),
    /// A product ("and") type, e.g. `Type & Type`.
    And(Vec<Spanned<TypeExpr>>),
    /// A negation ("not") type, e.g. `!Type`.
    Not(Box<Spanned<TypeExpr>>),
    /// The `mut` modifier which allows the type to be mutable and have access to mutable methods,
    /// e.g. `mut Type`.
    Mut(Box<Spanned<TypeExpr>>),
    /// The `to` modifier which allows any type that has a cast function to the given type,
    /// e.g. `to Type` or `to string`.
    To(Box<Spanned<TypeExpr>>),
}

impl Display for TypeExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Infer => f.write_str("_"),
            Self::Path(path) => write!(f, "{path}"),
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
            Self::Mut(ty) => write!(f, "mut {ty}"),
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
    Int(String, IntLiteralInfo),
    /// A floating-point number literal.
    Float(String),
    /// A string literal. This is after resolving escape sequences.
    String(String),
    /// A char literal. This is after resolving escape sequences.
    Char(char),
    /// A boolean literal.
    Bool(bool),
    /// The `void` keyword, which represents the absence of a value.
    Void,
    /// A non-keyword identifier.
    Ident(String),
}

impl Display for Atom {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(s, info) => write!(
                f,
                "{}{s}{}",
                info.radix.prefix(),
                if info.unsigned { "u" } else { "" },
            ),
            Self::Float(s) | Self::Ident(s) => write!(f, "{s}"),
            Self::String(s) => write!(f, "{s:?}"),
            Self::Char(c) => write!(f, "c{c:?}"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::Void => write!(f, "void"),
        }
    }
}

/// A visibility specifier of a top-level item.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ItemVisibility {
    /// This item is visible to everything.
    Public,
    /// This item is visible only to the current library.
    Lib,
    /// This item is visible only to the parent module.
    Super,
    /// This item is visible only to the current module. This is equivalent to `private`.
    #[default]
    Mod,
}

impl Display for ItemVisibility {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Public => write!(f, "public"),
            Self::Lib => write!(f, "public(lib)"),
            Self::Super => write!(f, "public(super)"),
            Self::Mod => write!(f, "private"),
        }
    }
}

/// A visibility specifier of a type member.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemberVisibility {
    /// This item is visible to everything.
    Public = 5,
    /// This item is visible only to the current library.
    Lib = 4,
    /// This item is visible only to the parent module.
    Super = 3,
    /// This item is visible only to the current module.
    #[default]
    Mod = 2,
    /// This item is visible only within the definition implementation and extensions.
    /// (i.e. self + subclasses)
    Sub = 1,
    /// This item is visible only within the definition implementation. (i.e. self)
    Private = 0,
}

impl MemberVisibility {
    /// Returns the inner visibility specifier, if any.
    #[must_use]
    pub const fn inner_visibility(&self) -> Option<&'static str> {
        Some(match self {
            Self::Public => return None,
            Self::Lib => "lib",
            Self::Super => "super",
            Self::Mod => "mod",
            Self::Sub => "sub",
            Self::Private => "private",
        })
    }
}

impl Display for MemberVisibility {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Public => write!(f, "public"),
            Self::Lib => write!(f, "public(lib)"),
            Self::Super => write!(f, "public(super)"),
            Self::Mod => write!(f, "public(mod)"),
            Self::Sub => write!(f, "public(sub)"),
            Self::Private => write!(f, "private"),
        }
    }
}

/// A visibility specifier for struct or class fields, which allows for finer visibility control
/// over getters and setters of the field. Fully expanded, these are written as:
/// `public(VIS get, VIS set)`, where `VIS` is any visibility specifier.
///
/// If `VIS` is not specified, full `public` visibility is assumed. If `get` or `set` is not
/// specified, the visibility of the getter or setter is assumed to be `private`.
///
/// The visibility of the setter must be equal or more private than the visibility of the getter.
///
/// Examples:
/// ```text
/// public -> public(get, set)
/// public(VIS) -> public(VIS get, VIS set)
///     where VIS is any visibility specifier
///     for example, public(lib) -> public(lib get, lib set)
/// public(get) -> public(get, private set)
/// public(VIS get) -> public(VIS get, private set)
///     where VIS is any visibility specifier
///    for example, public(lib get) -> public(lib get, private set)
/// public(get, VIS set) -> public(get, VIS set)
///     where VIS is any visibility specifier
///     for example, public(get, lib set) -> public(get, lib set)
/// public(GET_VIS get, SET_VIS set) -> public(VIS get, VIS set)
///     where GET_VIS and SET_VIS are any visibility specifiers
///     for example, public(lib get, mod set) -> public(lib get, mod set)
/// private -> public(private get, private set)
///     this is equivalent to public(private)
/// ```
///
/// Invalid examples, where the visibility of the setter is more public than the getter:
/// ```text
/// public(set) expands to public(private get, set)
/// public(lib set) expands to public(private get, lib set)
/// public(mod get, lib set)
/// ```
///
/// As a general rule of thumb:
/// ```text
/// For public((GET_VIS get) as GET, (SET_VIS set) as SET):
/// * GET_VIS <= SET_VIS
/// * if GET_VIS is not specified, GET_VIS = public
/// * if SET_VIS is not specified, SET_VIS = public
/// * if GET is not specified, GET = private
/// * if SET is not specified, SET = private
///
/// Expand public(VIS) to public(VIS get, VIS set)
/// Expand public to public(get, set)
/// Expand private to public(private get, private set)
/// Explicitly disallow public()
/// When visibility is not specified, default to:
///     public(mod) expands to public(mod get, mod set)
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FieldVisibility {
    /// The visibility of the getter.
    pub get: (MemberVisibility, Option<Span>),
    /// The visibility of the setter.
    pub set: (MemberVisibility, Option<Span>),
}

impl Display for FieldVisibility {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("public(")?;
        if let Some(vis) = self.get.0.inner_visibility() {
            write!(f, "{} ", vis)?;
        }
        f.write_str("get, ")?;

        if let Some(vis) = self.set.0.inner_visibility() {
            write!(f, "{} ", vis)?;
        }
        f.write_str("set)")
    }
}

/// The assignment operator of an assignment expression.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AssignmentOperator {
    /// The `=` operator.
    Assign,
    /// The `+=` operator.
    AddAssign,
    /// The `-=` operator.
    SubAssign,
    /// The `*=` operator.
    MulAssign,
    /// The `/=` operator.
    DivAssign,
    /// The `%=` operator.
    ModAssign,
    /// The `&=` operator.
    BitAndAssign,
    /// The `|=` operator.
    BitOrAssign,
    /// The `^=` operator.
    BitXorAssign,
    /// The `<<=` operator.
    ShlAssign,
    /// The `>>=` operator.
    ShrAssign,
    /// The `**=` operator.
    PowAssign,
    /// The `||=` operator.
    LogicalOrAssign,
    /// The `&&=` operator.
    LogicalAndAssign,
}

impl Display for AssignmentOperator {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Assign => write!(f, "="),
            Self::AddAssign => write!(f, "+="),
            Self::SubAssign => write!(f, "-="),
            Self::MulAssign => write!(f, "*="),
            Self::DivAssign => write!(f, "/="),
            Self::ModAssign => write!(f, "%="),
            Self::BitAndAssign => write!(f, "&="),
            Self::BitOrAssign => write!(f, "|="),
            Self::BitXorAssign => write!(f, "^="),
            Self::ShlAssign => write!(f, "<<="),
            Self::ShrAssign => write!(f, ">>="),
            Self::PowAssign => write!(f, "**="),
            Self::LogicalOrAssign => write!(f, "||="),
            Self::LogicalAndAssign => write!(f, "&&="),
        }
    }
}

/// The assignment target of an assignment expression.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AssignmentTarget {
    /// A pattern match target.
    Pattern(Pattern),
    /// A dereference of a mutable reference, directing the reassignment to what the reference is
    /// pointing to instead of the reference itself, e.g. `&x = 1;`.
    Pointer(Box<Spanned<Expr>>),
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

impl Display for AssignmentTarget {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pattern(p) => write!(f, "{p}"),
            Self::Pointer(p) => write!(f, "&{p}"),
            Self::Attr { subject, attr } => write!(f, "{subject}.{attr}"),
            Self::Index { subject, index } => write!(f, "{subject}[{index}]"),
        }
    }
}

/// A pattern-match target used when matching values, including in function parameters and in
/// declaration bindings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pattern {
    /// A binding.
    Ident {
        /// The name of the identifier.
        ident: Spanned<String>,
        /// The span of the `mut` keyword if the identifier is specified as mutable, otherwise
        /// `None`.
        mut_kw: Option<Span>,
    },
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
    #[allow(clippy::type_complexity)]
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

impl Display for Pattern {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut fmt_path = |path: &[Spanned<_>]| {
            write!(
                f,
                "{}",
                path.iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(".")
            )
        };

        match self {
            Self::Ident { ident, mut_kw } => {
                if mut_kw.is_some() {
                    f.write_str("mut ")?;
                }
                write!(f, "{ident}")
            }
            Self::Tuple(path, pats) => {
                fmt_path(path)?;
                write!(
                    f,
                    "({})",
                    pats.iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            Self::Array(pats) => {
                write!(
                    f,
                    "[{}]",
                    pats.iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            Self::Fields(path, fields) => {
                fmt_path(path)?;
                write!(
                    f,
                    "{{{}}}",
                    fields
                        .iter()
                        .map(|(name, pat)| {
                            pat.as_ref()
                                .map_or_else(|| name.to_string(), |pat| format!("{name}: {pat}"))
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            Self::As(lhs, rhs) => write!(f, "{lhs} as {rhs}"),
            Self::Wildcard(pat) => {
                f.write_str("*")?;
                pat.as_ref().map_or(Ok(()), |pat| write!(f, "{pat}"))
            }
        }
    }
}

impl Pattern {
    /// Walks the pattern, ensuring that all bindings are not mutable. This is used during syntax
    /// checking on assignment patterns.
    pub fn assert_immutable_bindings(&self) -> Result<(), Error> {
        match self {
            Self::Ident { ident, mut_kw } => {
                if let Some(span) = mut_kw {
                    return Err(Error::mutable_assignment_target(*span, ident.clone()));
                }
            }
            Self::Tuple(_, pats) | Self::Array(pats) => {
                for pat in pats {
                    pat.value().assert_immutable_bindings()?;
                }
            }
            Self::Fields(_, fields) => {
                for (_, pat) in fields {
                    if let Some(pat) = pat {
                        pat.value().assert_immutable_bindings()?;
                    }
                }
            }
            Self::As(lhs, rhs) => {
                lhs.value().assert_immutable_bindings()?;
                rhs.value().assert_immutable_bindings()?;
            }
            Self::Wildcard(pat) => {
                if let Some(pat) = pat {
                    pat.value().assert_immutable_bindings()?;
                }
            }
        }
        Ok(())
    }
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

/// Creates a new block that immediately returns a value.
#[must_use]
pub fn expr_as_block(expr: Spanned<Expr>) -> Spanned<Vec<Spanned<Node>>> {
    let span = expr.span();
    Spanned(vec![Spanned(Node::ImplicitReturn(expr), span)], span)
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
    /// Specifies the value to be an immutable reference, e.g. `ref x`.
    ///
    /// Note that immutable references are inferred by default, so this is only needed for
    /// cases where you want to explicitly specify.
    Ref(Span, Box<Spanned<Self>>),
    /// Specifies the value to be a mutable reference, e.g. `mut x`.
    Mut(Span, Box<Spanned<Self>>),
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
        /// The block label given to the if-statement.
        label: Option<Spanned<String>>,
        /// The condition of the if-statement.
        cond: Box<Spanned<Self>>,
        /// The body of the if-statement.
        body: Spanned<Vec<Spanned<Node>>>,
        /// The body of the else block. This is `None` if there is no else block.
        else_body: Option<Spanned<Vec<Spanned<Node>>>>,
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
        /// The block label given to the while-loop.
        label: Option<Spanned<String>>,
        /// The condition of the while-loop.
        cond: Box<Spanned<Self>>,
        /// The body of the while-loop.
        body: Spanned<Vec<Spanned<Node>>>,
        /// The body of the else block. This is `None` if there is no else block.
        /// The else-block is executed if the while-loop finishes execution without a break.
        ///
        /// While-loops without else-blocks are considered non-diverging and should not be
        /// considered as expressions.
        else_body: Option<Spanned<Vec<Spanned<Node>>>>,
    },
    /// A loop expression. Loop expressions either never diverge due to infinite loops,
    /// or always diverge due to a break statement. Therefore, they can also be considered as
    /// expressions.
    Loop {
        /// The block label given to the loop.
        label: Option<Spanned<String>>,
        /// The body of the loop.
        body: Spanned<Vec<Spanned<Node>>>,
    },
    /// A when expression. When expressions can simplify long if-else if chains into a mapping of
    /// conditions to corresponding values.
    ///
    /// For example, `when { x == 0 -> 0, x < 0 -> -1, else 1 }` is a valid expression and is
    /// equivalent to `if x == 0 then 0 else if x < 0 then -1 else 1`.
    When {
        /// The block label given to the when block.
        label: Option<Spanned<String>>,
        /// The condition-value pairs of the when expression.
        arms: Vec<(Spanned<Self>, Spanned<Self>)>,
        /// The else value of the when expression. This is `None` if there is no else value.
        else_value: Option<Box<Spanned<Self>>>,
    },
    /// A block expression. Block expressions enclose a new lexical scope.
    Block {
        /// The block label given to the block.
        label: Option<Spanned<String>>,
        /// The body of the block.
        body: Spanned<Vec<Spanned<Node>>>,
    },
    /// An assignment expression. Assignment expressions are used to assign values to declared
    /// variables.
    ///
    /// These are experssions that return the value that was assigned. For example, `a = 1`
    /// evaluates to `1`, and so something like `let (a, b); a = b = 0;` can be written.
    Assign {
        /// The target of the assignment.
        target: Spanned<AssignmentTarget>,
        /// The assignment operator.
        op: Spanned<AssignmentOperator>,
        /// The value being assigned.
        value: Box<Spanned<Self>>,
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

        for line in s.lines() {
            writeln!(f, "{INDENT}{line}")?;
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
            Self::Ref(_, subject) => write!(f, "(ref {subject})"),
            Self::Mut(_, subject) => write!(f, "(mut {subject})"),
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
                label,
                cond,
                body,
                else_body,
                ..
            } => {
                if let Some(label) = label {
                    write!(f, ":{label} ")?;
                }
                write!(f, "if {cond} ")?;
                f.write_str("{\n")?;

                for node in body.value() {
                    node.write_indent(f)?;
                }
                f.write_str("}")?;

                if let Some(else_body) = else_body {
                    f.write_str(" else {\n")?;
                    for node in else_body.value() {
                        node.write_indent(f)?;
                    }
                    f.write_str("}")?;
                }
                Ok(())
            }
            Self::While {
                label,
                cond,
                body,
                else_body,
            } => {
                if let Some(label) = label {
                    write!(f, ":{label} ")?;
                }
                writeln!(f, "while {cond} {{")?;
                for node in body.value() {
                    node.write_indent(f)?;
                }
                f.write_str("}")?;

                if let Some(else_body) = else_body {
                    f.write_str(" else {\n")?;
                    for node in else_body.value() {
                        node.write_indent(f)?;
                    }
                    f.write_str("}")?;
                }
                Ok(())
            }
            Self::Loop { label, body } => {
                if let Some(label) = label {
                    write!(f, ":{label} ")?;
                }
                writeln!(f, "loop {{")?;
                for node in body.value() {
                    node.write_indent(f)?;
                }
                f.write_str("}")
            }
            Self::When {
                label,
                arms,
                else_value,
            } => {
                if let Some(label) = label {
                    write!(f, ":{label} ")?;
                }
                f.write_str("when {\n")?;
                for (cond, value) in arms {
                    format!("{cond} -> {value}").write_indent(f)?;
                }
                if let Some(else_value) = else_value {
                    format!("else -> {else_value}").write_indent(f)?;
                }
                f.write_str("}")
            }
            Self::Block { label, body } => {
                if let Some(label) = label {
                    write!(f, ":{label} ")?;
                }
                f.write_str("{\n")?;
                for node in body.value() {
                    node.write_indent(f)?;
                }
                f.write_str("}")
            }
            Self::Assign { target, op, value } => {
                write!(f, "{target} {op} {value}")
            }
        }
    }
}

/// Information about a function parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FuncParam {
    /// The pattern of the parameter.
    pub pat: Spanned<Pattern>,
    /// The type of the parameter.
    pub ty: Spanned<TypeExpr>,
    /// The default value of the parameter, if it is specified.
    pub default: Option<Spanned<Expr>>,
}

impl Display for FuncParam {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.pat, self.ty)?;
        if let Some(default) = &self.default {
            write!(f, " = {default}")?;
        }
        Ok(())
    }
}

/// A generic type parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TyParam {
    /// The name of the type parameter.
    pub name: Spanned<String>,
    /// The bound of the type parameter, if any.
    pub bound: Option<Spanned<TypeExpr>>,
}

impl Display for TyParam {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)?;
        if let Some(bound) = &self.bound {
            write!(f, ": {}", bound)?;
        }
        Ok(())
    }
}

/// A field in a struct.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructField {
    /// The visibility of the field.
    pub vis: FieldVisibility,
    /// The name of the field.
    pub name: Spanned<String>,
    /// The type of the field.
    pub ty: Spanned<TypeExpr>,
    /// The default value of the field, if it is specified.
    pub default: Option<Spanned<Expr>>,
}

impl Display for StructField {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}: {}", self.vis, self.name, self.ty)?;
        if let Some(default) = &self.default {
            write!(f, " = {default}")?;
        }
        Ok(())
    }
}

/// A struct-like type definition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructDef {
    /// The visibility of the struct.
    pub vis: ItemVisibility,
    /// The name of the struct.
    pub name: Spanned<String>,
    /// The generic type parameters of the struct.
    pub ty_params: Vec<TyParam>,
    /// The fields of the struct.
    pub fields: Vec<Spanned<StructField>>,
    /// The struct this struct inherits its fields from, if any.
    pub extends: Option<Spanned<TypeExpr>>,
}

impl Display for StructDef {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} struct {}", self.vis, self.name)?;
        if !self.ty_params.is_empty() {
            write!(
                f,
                "<{}>",
                self.ty_params
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            )?;
        }
        write!(f, " {{\n")?;
        for field in &self.fields {
            writeln!(f, "    {field},")?;
        }
        f.write_str("}")
    }
}

/// A node in the abstract syntax tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Node {
    /// An expression represented as a statement.
    Expr(Spanned<Expr>),
    /// A variable declaration using `let`.
    Let {
        /// The span of the declaration keyword (i.e. let, const).
        kw: Span,
        /// The binding pattern of the declaration.
        pat: Spanned<Pattern>,
        /// The type of the variable, if it is specified.
        ty: Option<Spanned<TypeExpr>>,
        /// The value of the variable, if it is specified.
        /// This is always specified for const declarations.
        value: Option<Spanned<Expr>>,
    },
    /// A constant declaration.
    Const {
        /// The visibility of the constant.
        vis: ItemVisibility,
        /// The span of the const keyword.
        kw: Span,
        /// The name of the constant.
        name: Spanned<String>,
        /// The type of the variable, if it is specified.
        ty: Option<Spanned<TypeExpr>>,
        /// The value of the constant.
        value: Spanned<Expr>,
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
        /// The block label to break out of. If this is `None`, then the closest **loop** is broken
        /// out of.
        label: Option<Spanned<String>>,
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
        /// The block label to continue. If this is `None`, then the closest loop is continued.
        /// If this label is a non-loop block, this will be a compile-error.
        label: Option<Spanned<String>>,
        /// The condition on whether to continue.
        /// This is only specified for `continue if` statements.
        cond: Option<Spanned<Expr>>,
    },
    /// A named function declaration.
    Func {
        /// The visibility of the function.
        vis: ItemVisibility,
        /// The name of the function.
        name: Spanned<String>,
        /// The positional parameters of the function.
        params: Vec<Spanned<FuncParam>>,
        /// The keyword parameters of the function.
        kw_params: Vec<Spanned<FuncParam>>,
        /// The return type of the function, if it is specified.
        ret: Option<Spanned<TypeExpr>>,
        /// The body of the function.
        body: Spanned<Vec<Spanned<Node>>>,
    },
    /// A struct declaration.
    Struct(StructDef),
}

impl Display for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        fn write_control_flow_stmt(
            f: &mut Formatter,
            kw: &str,
            label: Option<&Spanned<String>>,
            value: &Option<Spanned<Expr>>,
            cond: &Option<Spanned<Expr>>,
        ) -> fmt::Result {
            f.write_str(kw)?;
            if let Some(label) = label {
                write!(f, " :{label}")?;
            }
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
            Self::Let { pat, ty, value, .. } => {
                write!(f, "let {pat}")?;
                if let Some(ty) = ty {
                    write!(f, ": {ty}")?;
                }
                if let Some(value) = value {
                    write!(f, " = {value}")?;
                }
                write!(f, ";")
            }
            Self::Const {
                name, ty, value, ..
            } => {
                write!(f, "const {name}")?;
                if let Some(ty) = ty {
                    write!(f, ": {ty}")?;
                }
                write!(f, " = {value};")
            }
            Self::Return { value, cond, .. } => {
                write_control_flow_stmt(f, "return", None, value, cond)
            }
            Self::ImplicitReturn(e) => write!(f, "{e}"),
            Self::Break {
                label, value, cond, ..
            } => write_control_flow_stmt(f, "break", label.as_ref(), value, cond),
            Self::Continue { label, cond, .. } => {
                f.write_str("continue")?;
                if let Some(label) = label {
                    write!(f, " :{label}")?;
                }
                if let Some(cond) = cond {
                    write!(f, " if {cond}")?;
                }
                write!(f, ";")
            }
            Self::Func {
                vis,
                name,
                params,
                kw_params,
                ret,
                body,
            } => {
                write!(f, "{vis} func {name}(")?;

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
                for node in body.value() {
                    node.write_indent(f)?;
                }
                f.write_str("}")
            }
            Self::Struct(s) => write!(f, "{s}"),
        }
    }
}
