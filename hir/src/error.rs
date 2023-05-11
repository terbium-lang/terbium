use crate::{Ident, ModuleId};
use ariadne::{ColorGenerator, Label, Report, ReportKind};
use common::span::{ProviderCache, Span, Spanned};
use grammar::ast::{TypeExpr, TypePath};
use std::fmt::{Display, Formatter};
use std::io::Write;

pub type Result<T> = std::result::Result<T, AstLoweringError>;

/// Represents an error that occurred during the lowering of AST to HIR.
#[derive(Clone, Debug)]
pub enum AstLoweringError {
    /// The provided type is not a concrete struct. Structs may only extend fields from other
    /// structs.
    CannotExtendFieldsFromType(Spanned<TypeExpr>),
    /// Another item with the same name already exists in the current module.
    NameConflict(Span, Spanned<String>),
    /// Integer literal will overflow 128 bits. If the integer is not unsigned, making it unsigned
    /// may increase the maximum value before an overflow such as this error occurs.
    IntegerLiteralOverflow(Span),
    /// Float literal will overflow 64 bits.
    FloatLiteralOverflow(Span),
    /// Type with the given path was not found. Represented as (type_span, name_span, module_id)
    TypeNotFound(Span, Spanned<String>, ModuleId),
    /// The module with the given path was not found.
    ModuleNotFound(Spanned<String>),
    /// Type was provided with the incorrect number of type arguments.
    IncorrectTypeArgumentCount {
        /// The span of the full type expression with the incorrect number of type arguments.
        span: Span,
        /// The type declaration.
        ty: Spanned<String>,
        /// The expected number of type arguments.
        expected: usize,
        /// The actual number of type arguments.
        actual: usize,
    },
    /// Encountered a circular type reference.
    CircularTypeReference {
        /// The type reference.
        src: Spanned<Ident>,
        /// The type being referenced.
        dest: Spanned<TypePath>,
        /// The span where the type being referenced references the source type again.
        circular_at: Span,
    },
}

impl Display for AstLoweringError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CannotExtendFieldsFromType(ty) => {
                write!(f, "cannot extend fields from type `{ty}`")
            }
            Self::NameConflict(_, name) => {
                write!(f, "name conflict for `{name}`")
            }
            Self::IntegerLiteralOverflow(_) => {
                write!(f, "integer literal will overflow 128 bits")
            }
            Self::FloatLiteralOverflow(_) => {
                write!(f, "float literal will overflow 64 bits")
            }
            Self::TypeNotFound(_, name, module) => {
                write!(f, "could not find type `{name}` in module `{module}`")
            }
            Self::ModuleNotFound(name) => {
                write!(f, "could not find module `{name}`")
            }
            Self::IncorrectTypeArgumentCount {
                ty,
                expected,
                actual,
                ..
            } => {
                write!(
                    f,
                    "incorrect number of type arguments for `{ty}`: expected {expected} type arguments, found {actual}",
                )
            }
            Self::CircularTypeReference { src, dest, .. } => {
                write!(
                    f,
                    "circular type `{src}` references `{dest}`, which references `{src}` again"
                )
            }
        }
    }
}

impl std::error::Error for AstLoweringError {}

impl AstLoweringError {
    pub const fn simple_message(&self) -> &'static str {
        match self {
            Self::CannotExtendFieldsFromType(_) => "cannot extend fields from type",
            Self::NameConflict(..) => "item name conflict",
            Self::IntegerLiteralOverflow(_) => "integer literal will overflow 128 bits",
            Self::FloatLiteralOverflow(_) => "float literal will overflow 64 bits",
            Self::TypeNotFound(..) => "could not resolve type",
            Self::ModuleNotFound(..) => "could not resolve module",
            Self::IncorrectTypeArgumentCount { .. } => "incorrect number of type arguments",
            Self::CircularTypeReference { .. } => "encountered circular type reference",
        }
    }

    pub const fn span(&self) -> Span {
        match self {
            Self::CannotExtendFieldsFromType(ty) => ty.span(),
            Self::NameConflict(span, def) => span.merge(def.span()),
            Self::IntegerLiteralOverflow(span)
            | Self::FloatLiteralOverflow(span)
            | Self::TypeNotFound(span, ..) => *span,
            Self::ModuleNotFound(name) => name.span(),
            Self::IncorrectTypeArgumentCount { span, ty, .. } => span.merge(ty.span()),
            Self::CircularTypeReference {
                src,
                dest,
                circular_at,
            } => src.span().merge(dest.span()).merge(*circular_at),
        }
    }

    pub const fn error_code(&self) -> usize {
        match self {
            Self::CannotExtendFieldsFromType(_) => 100,
            Self::NameConflict(..) => 101,
            Self::IntegerLiteralOverflow(_) => 102,
            Self::FloatLiteralOverflow(_) => 103,
            Self::TypeNotFound(..) => 104,
            Self::ModuleNotFound(..) => 105,
            Self::IncorrectTypeArgumentCount { .. } => 106,
            Self::CircularTypeReference { .. } => 107,
        }
    }

    /// Writes the error as a diagnostic to the given writer.
    pub fn write(self, cache: &ProviderCache, writer: impl Write) -> std::io::Result<()> {
        let mut colors = ColorGenerator::new();
        let primary = colors.next();

        let span = self.span();
        let mut report = Report::build(ReportKind::Error, span.src, span.start)
            .with_code(format!("E{:03}", self.error_code()))
            .with_message(self.simple_message());

        report = match self {
            Self::CannotExtendFieldsFromType(ty) => report
                .with_label(Label::new(ty.span())
                    .with_message(format!("cannot extend fields from `{ty}`"))
                    .with_color(primary),
                )
                .with_help("you can only extend fields from concrete struct types, nothing else"),
            Self::NameConflict(def_span, src) => report
                .with_label(Label::new(def_span)
                    .with_message(format!("item with name {src} defined here"))
                    .with_color(primary)
                )
                .with_label(Label::new(src.span())
                    .with_message("but it is also defined here")
                    .with_color(colors.next())
                )
                .with_help("try renaming to something else"),
            Self::IntegerLiteralOverflow(span) => report
                .with_label(Label::new(span)
                    .with_message("this integer literal here")
                    .with_color(primary)
                )
                .with_help("represent the integer as a string, and then convert to a type that can fit this large of an integer at runtime"),
            Self::FloatLiteralOverflow(span) => report
                .with_label(Label::new(span)
                    .with_message("this float literal here")
                    .with_color(primary)
                )
                .with_help("represent the float as a string, and then convert to a type that can fit this large of an float at runtime"),
            Self::TypeNotFound(_, name, module) => report
                .with_label(Label::new(name.span())
                    .with_message(format!("could not find type `{name}` in {module}"))
                    .with_color(primary)
                )
                .with_help("check the spelling of the type"), // TODO: maybe you meant...
            Self::ModuleNotFound(name) => report
                .with_label(Label::new(name.span())
                    .with_message(format!("could not find module `{name}`"))
                    .with_color(primary)
                )
                .with_message("check the spelling of the module"),
            Self::IncorrectTypeArgumentCount { span, ty, expected, actual } => report
                .with_label(Label::new(span)
                    .with_message(format!("provided {actual} type arguments to {ty}"))
                    .with_color(primary)
                )
                .with_label(Label::new(ty.span())
                    .with_message(format!("...but it expects {expected} type arguments"))
                    .with_color(colors.next())
                )
                .with_help(r#"try specifying the required number of type arguments. if you do not want to specify types in full, you can specify the inference type `_` instead"#),
            Self::CircularTypeReference { src, dest, circular_at } => report
                .with_label(Label::new(src.span())
                    .with_message(format!("source type `{src}` defined here"))
                    .with_color(primary)
                )
                .with_label(Label::new(dest.span())
                    .with_message(format!("reference to type `{dest}` found here"))
                    .with_color(colors.next())
                )
                .with_label(Label::new(circular_at)
                    .with_message(format!("the type `{dest}` references `{src}` here, causing a circular reference"))
                    .with_color(colors.next())
                )
                .with_help("try adding a level of indirection or removing the circular type completely"),
        };

        report.finish().write(cache, writer)
    }
}
