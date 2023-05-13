use crate::{Ident, ModuleId};
use ariadne::{ColorGenerator, Label, Report, ReportKind};
use common::{
    pluralize,
    span::{ProviderCache, Span, Spanned},
};
use grammar::ast::{self, TypeExpr, TypePath};
use std::{
    fmt::{Display, Formatter},
    io::Write,
};

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
    /// Visibility for field access is less than the visibility for field mutation.
    GetterLessVisibleThanSetter(Spanned<ast::FieldVisibility>),
    /// Invalid assignment target. Operator-assigns desugar to convert the assignment target to a
    /// valid expression, hence many forms of patterns are disallowed.
    InvalidAssignmentTarget(Span, &'static str),
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
            Self::GetterLessVisibleThanSetter(vis) => {
                write!(
                    f,
                    "getter is less visible than setter ({} < {})",
                    vis.0.get.0, vis.0.set.0
                )
            }
            Self::InvalidAssignmentTarget(_, subject) => {
                write!(f, "cannot use {subject} as an assignment target")
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
            Self::GetterLessVisibleThanSetter { .. } => "getter is less visible than setter",
            Self::InvalidAssignmentTarget(..) => "invalid assignment target",
        }
    }

    pub const fn span(&self) -> Span {
        match self {
            Self::CannotExtendFieldsFromType(ty) => ty.span(),
            Self::NameConflict(span, def) => span.merge(def.span()),
            Self::IntegerLiteralOverflow(span)
            | Self::FloatLiteralOverflow(span)
            | Self::TypeNotFound(span, ..)
            | Self::InvalidAssignmentTarget(span, _) => *span,
            Self::ModuleNotFound(name) => name.span(),
            Self::IncorrectTypeArgumentCount { span, ty, .. } => span.merge(ty.span()),
            Self::CircularTypeReference {
                src,
                dest,
                circular_at,
            } => src.span().merge(dest.span()).merge(*circular_at),
            Self::GetterLessVisibleThanSetter(vis) => vis.span(),
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
            Self::GetterLessVisibleThanSetter(_) => 108,
            Self::InvalidAssignmentTarget(..) => 109,
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
            Self::CannotExtendFieldsFromType(ty) => {
                report
                    .with_label(
                        Label::new(ty.span())
                            .with_message(format!("cannot extend fields from `{ty}`"))
                            .with_color(primary),
                    )
                    .with_help("you can only extend fields from concrete struct types, nothing else")
            },
            Self::NameConflict(def_span, src) => {
                report
                    .with_label(Label::new(def_span)
                        .with_message(format!("item with name {src} defined here"))
                        .with_color(primary)
                        .with_order(0)
                    )
                    .with_label(Label::new(src.span())
                        .with_message("but it is also defined here")
                        .with_color(colors.next())
                        .with_order(1)
                    )
                    .with_help("try renaming to something else")
            },
            Self::IntegerLiteralOverflow(span) => {
                report
                    .with_label(Label::new(span)
                        .with_message("this integer literal here")
                        .with_color(primary)
                    )
                    .with_help("represent the integer as a string, and then convert to a type that can fit this large of an integer at runtime")
            },
            Self::FloatLiteralOverflow(span) => {
                report
                    .with_label(Label::new(span)
                        .with_message("this float literal here")
                        .with_color(primary)
                    )
                    .with_help("represent the float as a string, and then convert to a type that can fit this large of an float at runtime")
            },
            Self::TypeNotFound(_, name, module) => {
                report
                    .with_label(Label::new(name.span())
                        .with_message(format!("could not find type `{name}` in {module}"))
                        .with_color(primary)
                    )
                    .with_help("check the spelling of the type") // TODO: maybe you meant...
            },
            Self::ModuleNotFound(name) => {
                report
                    .with_label(Label::new(name.span())
                        .with_message(format!("could not find module `{name}`"))
                        .with_color(primary)
                    )
                    .with_message("check the spelling of the module")
            },
            Self::IncorrectTypeArgumentCount { span, ty, expected, actual } => {
                report
                    .with_label(Label::new(span)
                        .with_message(format!("provided {actual} type {} to {ty}", pluralize(actual, "argument", "arguments")))
                        .with_color(primary)
                        .with_order(0)
                    )
                    .with_label(Label::new(ty.span())
                        .with_message(format!("{ty} expects {expected} type {}", pluralize(actual, "argument", "arguments")))
                        .with_color(colors.next())
                        .with_order(1)
                    )
                    .with_help(r#"try specifying the required number of type arguments. if you do not want to specify types in full, you can specify the inference type `_` instead"#)
            },
            Self::CircularTypeReference { src, dest, circular_at } => {
                report
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
                    .with_help("try adding a level of indirection or removing the circular type completely")
            }
            Self::GetterLessVisibleThanSetter(Spanned(vis, _)) => {
                if let Some(get_span) = vis.get.1 {
                    report.add_label(Label::new(get_span)
                        .with_message(format!("visibility of getter defined as {} here", vis.get.0))
                        .with_color(primary)
                    );
                    report.set_help("try generalizing visibility to both get and set");
                } else {
                    report.set_note(format!("visibility of getter is implied to be {}", vis.get.0));
                }
                if let Some(set_span) = vis.set.1 {
                    report.add_label(Label::new(set_span)
                        .with_message(format!("visibility of setter defined as {} here", vis.set.0))
                        .with_color(colors.next())
                    );
                    report.set_help("try unspecifying the setter visibility");
                } else {
                    report.set_note(format!("visibility of setter is implied to be {}", vis.set.0));
                }
                report
            }
            Self::InvalidAssignmentTarget(span, subject) => {
                report
                    .with_label(Label::new(span)
                        .with_message(format!("cannot use {subject} as an assignment target"))
                        .with_color(primary)
                    )
                    .with_help("assign to an identifier, field, or index instead")
            }
        };

        report.finish().write(cache, writer)
    }
}
