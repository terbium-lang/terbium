use crate::{Ident, ModuleId, Ty};
use common::{
    pluralize,
    span::{Span, Spanned},
};
use diagnostics::{Action, Diagnostic, Fix, Label, Section, Severity};
use grammar::ast::{self, TypeExpr, TypePath};
use std::{
    fmt::{Display, Formatter},
    io::Write,
};

pub type Result<T> = std::result::Result<T, Error>;

/// Represents an error that occurred during the lowering of AST to HIR.
#[derive(Clone, Debug)]
pub enum Error {
    /// The provided type is not a concrete struct. Structs may only extend fields from other
    /// structs.
    CannotExtendFieldsFromType(Spanned<TypeExpr>),
    /// Another item with the same name already exists in the current module.
    NameConflict(Span, Spanned<String>),
    /// Integer literal will overflow 128 bits. If the integer is not unsigned, making it unsigned
    /// may increase the maximum value before an overflow such as this error occurs.
    IntegerLiteralOverflow(Span, String),
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
    /// Encountered a circular type reference. Each cycle is represented as `(def_src, def_dest)`.
    CircularTypeReference(Vec<(Spanned<Ident>, Spanned<TypePath>)>),
    /// Visibility for field access is less than the visibility for field mutation.
    GetterLessVisibleThanSetter(Spanned<ast::FieldVisibility>),
    /// Invalid assignment target. Operator-assigns desugar to convert the assignment target to a
    /// valid expression, hence many forms of patterns are disallowed.
    InvalidAssignmentTarget(Span, &'static str),
    /// Cannot solve cyclic type constraint.
    CyclicTypeConstraint {
        /// The span of the type constraint.
        span: Span,
        /// The right-hand side of the type constraint that was cyclic.
        rhs: Ty,
    },
    /// Type mismatch.
    TypeMismatch {
        /// The expected type with an optional span.
        expected: (Ty, Option<Span>),
        /// The actual type.
        actual: Spanned<Ty>,
    },
    /// Explicit generic type arguments are not allowed in this context.
    ExplicitTypeArgumentsNotAllowed(Span),
    /// Unresolved identifier.
    UnresolvedIdentifier(Spanned<String>),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CannotExtendFieldsFromType(ty) => {
                write!(f, "cannot extend fields from type `{ty}`")
            }
            Self::NameConflict(_, name) => {
                write!(f, "name conflict for `{name}`")
            }
            Self::IntegerLiteralOverflow(..) => {
                write!(f, "integer literal will overflow 128 bits")
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
            Self::CircularTypeReference(cycle) => {
                let src = &cycle.first().expect("cycle cannot be empty").0;
                let dest = &cycle.last().unwrap().1;
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
            Self::CyclicTypeConstraint { rhs, .. } => {
                write!(f, "cannot solve cyclic type constraint `_ = {rhs}`",)
            }
            Self::TypeMismatch {
                expected: (expected, _),
                actual: Spanned(actual, _),
            } => {
                write!(f, "type mismatch, expected `{expected}`, found `{actual}`")
            }
            Self::ExplicitTypeArgumentsNotAllowed(_) => {
                write!(
                    f,
                    "explicit generic type arguments are not allowed in this context"
                )
            }
            Self::UnresolvedIdentifier(name) => {
                write!(f, "unresolved identifier `{name}`")
            }
        }
    }
}

impl std::error::Error for Error {}

impl Error {
    pub const fn simple_message(&self) -> &'static str {
        match self {
            Self::CannotExtendFieldsFromType(_) => "cannot extend fields from this type",
            Self::NameConflict(..) => "item name conflict",
            Self::IntegerLiteralOverflow(..) => "integer literal will overflow 128 bits",
            Self::TypeNotFound(..) => "could not resolve type",
            Self::ModuleNotFound(..) => "could not resolve module",
            Self::IncorrectTypeArgumentCount { .. } => "incorrect number of type arguments",
            Self::CircularTypeReference { .. } => "encountered circular type reference",
            Self::GetterLessVisibleThanSetter { .. } => "getter is less visible than setter",
            Self::InvalidAssignmentTarget(..) => "invalid assignment target",
            Self::CyclicTypeConstraint { .. } => "cannot solve cyclic type constraint",
            Self::TypeMismatch { .. } => "type mismatch",
            Self::ExplicitTypeArgumentsNotAllowed(..) => {
                "explicit generic type arguments are not allowed in this context"
            }
            Self::UnresolvedIdentifier(..) => "unresolved identifier",
        }
    }

    pub const fn error_code(&self) -> usize {
        match self {
            Self::CannotExtendFieldsFromType(_) => 100,
            Self::NameConflict(..) => 101,
            Self::IntegerLiteralOverflow(..) => 102,
            // NOTE: error code 103 is open
            Self::TypeNotFound(..) => 104,
            Self::ModuleNotFound(..) => 105,
            Self::IncorrectTypeArgumentCount { .. } => 106,
            Self::CircularTypeReference { .. } => 107,
            Self::GetterLessVisibleThanSetter(_) => 108,
            Self::InvalidAssignmentTarget(..) => 109,
            Self::CyclicTypeConstraint { .. } => 110,
            Self::TypeMismatch { .. } => 111,
            Self::ExplicitTypeArgumentsNotAllowed(_) => 112,
            Self::UnresolvedIdentifier(..) => 113,
        }
    }

    /// Converst the error as a diagnostic.
    pub fn into_diagnostic(self) -> Diagnostic {
        let mut diagnostic =
            Diagnostic::new(Severity::Error(self.error_code()), self.simple_message());
        diagnostic = match self {
            Self::CannotExtendFieldsFromType(ty) => {
                diagnostic
                    .with_section(Section::new()
                        .with_label(Label::at(ty.span())
                            .with_message(format!("cannot extend fields from type `{ty}`"))
                        )
                        .with_note(format!("`{ty}` is not a concrete struct type"))
                    )
                    .with_help("you can only extend fields from concrete struct types, nothing else")
            }
            Self::NameConflict(def_span, src) => {
                diagnostic
                    .with_section(Section::new()
                        .with_label(Label::at(def_span)
                            .with_message(format!("item with name `{src}` defined here"))
                        )
                        .with_label(Label::at(src.span())
                            .with_message(format!("but it is also defined here"))
                        )
                    )
                    .with_fix(
                        Fix::new(Action::Replace(src.span(), "<new_name>".to_string()))
                            .with_message("try renaming to something else")
                    )
            }
            Self::IntegerLiteralOverflow(span, attempt) => {
                let diagnostic  = diagnostic
                    .with_section(Section::new()
                        .with_label(Label::at(span).with_message("this integer literal here"))
                    );

                // Can it still be represented as a uint?
                if attempt.parse::<u128>().is_ok() {
                    diagnostic.with_fix(
                        Fix::new(Action::InsertAfter(span, "u".to_string()))
                            .with_message("making this an unsigned integer will fix the overflow")
                            .with_label("suffix with `u` to make this an unsigned integer"),
                    )
                } else {
                    diagnostic.with_fix(
                        Fix::new(Action::InsertAfter(span, ".0".to_string()))
                            .with_message("making this a float will fix the overflow at the expense of precision")
                            .with_label("add `.0` to make this a float"),
                    )
                }
            }
            Self::TypeNotFound(_, name, module) => {
                // TODO: maybe you meant...
                diagnostic
                    .with_section(Section::new()
                        .with_label(Label::at(name.span())
                            .with_message(format!("could not find type `{name}` in {module}"))
                        )
                    )
                    .with_help("check the spelling of the type")
            }
            Self::ModuleNotFound(module) => {
                diagnostic
                    .with_section(Section::new()
                        .with_label(Label::at(module.span())
                            .with_message(format!("could not find module `{module}`"))
                        )
                    )
                    .with_help("check the spelling of the module")
            }
            Self::IncorrectTypeArgumentCount { span, ty, expected, actual } => {
                let expected_args = pluralize(expected, "argument", "arguments");
                let actual_args = pluralize(actual, "argument", "arguments");
                diagnostic
                    .with_section(Section::new()
                        .with_label(Label::at(span)
                            .with_message(format!("provided {actual} type {actual_args} to `{ty}`"))
                        )
                    )
                    .with_section(Section::new()
                        .with_header(format!("`{ty}` is defined here:"))
                        .with_label(Label::at(ty.span())
                            .with_message(format!("`{ty}` expects {expected} type {expected_args}, not {actual}"))
                        )
                    )
                    .with_help(format!(
                        "specify the required number of type arguments. \
                        if you do not want to specify types in full, you can specify the \
                        inference type `_` instead"
                    ))
            }
            Self::CircularTypeReference(cycle) => {
                let last = cycle.len() - 2;
                let mut cycle = cycle.into_iter();

                let (src, dest) = cycle.next().expect("cycle cannot be empty");
                let mut diagnostic = diagnostic
                    .with_section(Section::new()
                        .with_label(Label::at(src.span())
                            .with_message(format!("source type `{src}` defined here"))
                        )
                        .with_label(Label::at(dest.span())
                            .with_message(format!("reference to type `{dest}` defined here"))
                        )
                    );

                for (i, (src, dest)) in cycle.enumerate() {
                    diagnostic = diagnostic.with_section(Section::new()
                        .with_header(if i == 0 {
                            format!("however, `{src}` also references `{dest}`:")
                        } else {
                            format!("...which also references `{dest}`")
                        })
                        .with_label(Label::at(dest.span())
                            .with_context_span(src.span().merge(dest.span()))
                            .with_message(format!("the type `{src}` references `{dest}` here{}", if i == last {
                                ", leading to a circular type reference"
                            } else { "" }))
                        )
                        .with_note("circular type references occur when a type references itself, either directly or indirectly")
                    );
                }

                diagnostic.with_help("try adding a level of indirection or removing the circular type completely")
            }
            Self::GetterLessVisibleThanSetter(Spanned(vis, span)) => {
                let mut section = Section::new();
                let mut fix = None;
                // We can guarantee that this error is only emitted when at least one of the getter
                // or setter visibility was explicitly provided
                if let Some(get_span) = vis.get.1 {
                    section = section.with_label(Label::at(get_span)
                        .with_message(format!("visibility of getter defined as {} here", vis.get.0))
                    );
                    // If *only* the getter was provided, (note that if the setter was also provided,
                    // this condition is overwritten), we can provide a fix to make the getter public
                    fix = Some(Fix::new(Action::Replace(span, "public".to_string()))
                        .with_message("try generalizing the visibility:")
                        .with_label("make the getter public")
                    );
                } else {
                    section = section.with_note(format!("visibility of getter is implied to be {}", vis.get.0));
                }
                if let Some(set_span) = vis.set.1 {
                    section = section.with_label(Label::at(set_span)
                        .with_message(format!("visibility of setter defined as {} here", vis.set.0))
                    );
                    fix = Some(Fix::new(Action::Remove(set_span))
                        .with_message("try unspecifying the setter visibility:")
                        .with_label("removing this will imply `private set`")
                    );
                } else {
                    section = section.with_note(format!("visibility of setter is implied to be {}", vis.set.0));
                }
                diagnostic = diagnostic.with_section(section);
                if let Some(fix) = fix {
                    diagnostic = diagnostic.with_fix(fix);
                }
                diagnostic
                    .with_note("the visibility of the getter must be at least as visible as the setter")
            }
            Self::InvalidAssignmentTarget(span, subject) => {
                diagnostic
                    .with_section(Section::new()
                        .with_label(Label::at(span)
                            .with_message(format!("cannot use {subject} as an assignment target"))
                        )
                        .with_note("you can only assign to identifiers, pointers, fields, or indices")
                    )
            }
            Self::CyclicTypeConstraint { span, rhs } => {
                diagnostic
                    .with_section(Section::new()
                        .with_label(Label::at(span)
                            .with_message(format!("the type in this context cannot be constrained to `{rhs}`"))
                        )
                        .with_note("such a constraint is not allowed because it causes a cyclic type constraint"),
                    )
                    .with_help("try explicitly specifying types to remove ambiguity")
                    .with_note(format!("tried solving the constraint `_ = {rhs}`"))
            }
            Self::TypeMismatch { expected: (expected, span), actual } => {
                let mut section = Section::new();
                if let Some(span) = span {
                    section = section.with_label(Label::at(span)
                        .with_message(format!("expected type `{expected}` here"))
                    );
                }

                diagnostic
                    .with_section(section
                        .with_label(Label::at(actual.span())
                            .with_message(format!("found `{actual}` here"))
                        )
                    )
                    // TODO: only show this fix if a cast can actually be performed
                    .with_fix(Fix::new(Action::InsertAfter(actual.span(), format!(" to {expected}")))
                        .with_message("try casting the value to the expected type if possible")
                    )
                    .with_help("try changing the value to match the expected type")
            }
            Self::ExplicitTypeArgumentsNotAllowed(span) => {
                diagnostic
                    .with_section(Section::new()
                        .with_label(Label::at(span)
                            .with_message("explicit type arguments are not allowed in this context")
                        )
                    )
                    .with_fix(Fix::new(Action::Remove(span))
                        .with_message("try removing the explicit type arguments")
                        .with_label("remove these type arguments")
                    )
            }
            Self::UnresolvedIdentifier(Spanned(name, span)) => {
                // TODO: suggest similar identifiers
                diagnostic
                    .with_section(Section::new()
                        .with_label(Label::at(span)
                            .with_message(format!("cannot find `{name}` in this scope"))
                        )
                    )
                    .with_help("check the spelling of the identifier")
            }
        };
        diagnostic
    }
}
