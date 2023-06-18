use crate::{typed::Constraint, Ident, ModuleId, PrimitiveTy, Ty};
use common::{
    pluralize,
    span::{Span, Spanned},
};
use diagnostics::{Action, Diagnostic, Fix, Label, Section, Severity};
use grammar::ast::{self, TypeExpr, TypePath};
use std::{
    borrow::Cow,
    fmt::{Display, Formatter},
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
    /// A scalar type like `int` was provided with type arguments. Represented as
    /// `(ty, type_span, application_span)`.
    ScalarTypeWithArguments(PrimitiveTy, Span, Span),
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
    /// Type conflict when trying to unify types.
    TypeConflict {
        /// The expected type with an optional span.
        expected: (Ty, Option<Span>),
        /// The actual type.
        actual: Spanned<Ty>,
        /// The constraint that caused the type conflict.
        constraint: Constraint,
    },
    /// Explicit generic type arguments are not allowed in this context.
    ExplicitTypeArgumentsNotAllowed(Span),
    /// Unresolved identifier.
    UnresolvedIdentifier(Spanned<String>),
    /// Cannot bind pattern to the given value.
    PatternMismatch {
        /// The description of the pattern attempted to be bound to.
        pat: Cow<'static, str>,
        /// The span of the pattern.
        pat_span: Span,
        /// The description of the value attempted to be bound.
        value: String,
        /// The span of the value, if applicable.
        value_span: Option<Span>,
    },
    /// Type of a condition is not `bool`.
    ConditionNotBool(Spanned<Ty>),
    /// Cannot reassign to an immutable variable. Represented as (reass_span, def)
    ReassignmentToImmutable(Span, Spanned<Ident>),
    /// Could not infer the type of a variable.
    CouldNotInferType(Spanned<Ident>),
    /// There is no loop to break or continue from.
    InvalidBreak(Span, Option<Spanned<Ident>>),
    /// There is no function to return from.
    InvalidReturn(Span),
    /// Block label not found.
    LabelNotFound(Spanned<Ident>),
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
            Self::ScalarTypeWithArguments(ty, _, _) => {
                write!(f, "scalar type `{ty}` cannot have type arguments")
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
            Self::TypeConflict {
                expected: (expected, _),
                actual: Spanned(actual, _),
                ..
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
            Self::PatternMismatch { pat, value, .. } => {
                write!(f, "cannot bind {value} to {pat}")
            }
            Self::ConditionNotBool(ty) => {
                write!(f, "expected condition to be `bool`, got `{ty}` instead")
            }
            Self::ReassignmentToImmutable(_, def) => {
                write!(f, "cannot reassign to immutable variable `{def}`")
            }
            Self::CouldNotInferType(def) => {
                write!(f, "could not infer type for `{def}`")
            }
            Self::InvalidBreak(..) => {
                write!(f, "cannot break from this context")
            }
            Self::InvalidReturn(_) => {
                write!(f, "cannot return from this context")
            }
            Self::LabelNotFound(name) => {
                write!(f, "cannot find label `{name}`")
            }
        }
    }
}

impl std::error::Error for Error {}

impl Error {
    /// A simple message describing the error.
    pub const fn simple_message(&self) -> &'static str {
        match self {
            Self::CannotExtendFieldsFromType(_) => "cannot extend fields from this type",
            Self::NameConflict(..) => "item name conflict",
            Self::IntegerLiteralOverflow(..) => "integer literal will overflow 128 bits",
            Self::TypeNotFound(..) => "could not resolve type",
            Self::ModuleNotFound(..) => "could not resolve module",
            Self::IncorrectTypeArgumentCount { .. } => "incorrect number of type arguments",
            Self::ScalarTypeWithArguments(..) => "scalar type cannot have type arguments",
            Self::CircularTypeReference { .. } => "encountered circular type reference",
            Self::GetterLessVisibleThanSetter { .. } => "getter is less visible than setter",
            Self::InvalidAssignmentTarget(..) => "invalid assignment target",
            Self::CyclicTypeConstraint { .. } => "cannot solve cyclic type constraint",
            Self::TypeConflict { .. } => "type conflict",
            Self::ExplicitTypeArgumentsNotAllowed(..) => {
                "explicit generic type arguments are not allowed in this context"
            }
            Self::UnresolvedIdentifier(..) => "unresolved identifier",
            Self::PatternMismatch { .. } => "pattern mismatch",
            Self::ConditionNotBool(..) => "condition is not `bool`",
            Self::ReassignmentToImmutable(..) => "cannot reassign to immutable variable",
            Self::CouldNotInferType(..) => "could not infer type",
            Self::InvalidBreak(..) => "cannot break from this context",
            Self::InvalidReturn(_) => "cannot return from this context",
            Self::LabelNotFound(_) => "cannot find label",
        }
    }

    /// The error code used to identify this error in the error index.
    pub const fn error_code(&self) -> usize {
        match self {
            Self::CannotExtendFieldsFromType(_) => 100,
            Self::NameConflict(..) => 101,
            Self::IntegerLiteralOverflow(..) => 102,
            // NOTE: error code 103 was left open which was eventually taken by
            // `ScalarTypeWithArguments`.
            Self::ScalarTypeWithArguments(..) => 103,
            Self::TypeNotFound(..) => 104,
            Self::ModuleNotFound(..) => 105,
            Self::IncorrectTypeArgumentCount { .. } => 106,
            Self::CircularTypeReference { .. } => 107,
            Self::GetterLessVisibleThanSetter(_) => 108,
            Self::InvalidAssignmentTarget(..) => 109,
            Self::CyclicTypeConstraint { .. } => 110,
            Self::TypeConflict { .. } => 111,
            Self::ExplicitTypeArgumentsNotAllowed(_) => 112,
            Self::UnresolvedIdentifier(..) => 113,
            Self::PatternMismatch { .. } => 114,
            Self::ConditionNotBool(_) => 115,
            Self::ReassignmentToImmutable(..) => 116,
            Self::CouldNotInferType(_) => 117,
            Self::InvalidBreak(..) => 118,
            Self::InvalidReturn(_) => 119,
            Self::LabelNotFound(_) => 120,
        }
    }

    /// Converts the error into a diagnostic.
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
            Self::ScalarTypeWithArguments(ty, ty_span, app_span) => {
                diagnostic
                    .with_section(Section::new()
                        .with_label(Label::at(app_span)
                            .with_context_span(ty_span.merge(app_span))
                            .with_message(format!("scalar type `{ty}` does not take any type arguments"))
                        )
                    )
                    .with_fix(
                        Fix::new(Action::Remove(app_span))
                            .with_message(format!("remove the type application provided to `{ty}`"))
                    )
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
            Self::TypeConflict {
                expected: (expected, span),
                actual,
                constraint: Constraint(a, b),
            } => {
                let mut section = Section::new();
                let mut message = format!("found `{actual}` here");
                if let Some(span) = span {
                    section = section.with_label(Label::at(span)
                        .with_message(format!("expected type `{expected}` here"))
                    );
                } else {
                    message.push_str(&*format!(", but expected `{expected}`"))
                }

                diagnostic
                    .with_section(section
                        .with_label(Label::at(actual.span()).with_message(message))
                        .with_note(format!("type `{actual}` is not compatible with `{expected}`"))
                    )
                    // TODO: only show this fix if a cast can actually be performed
                    .with_fix(Fix::new(Action::InsertAfter(actual.span(), format!(" to {expected}")))
                        .with_message("try casting the value to the expected type if possible")
                    )
                    .with_note(format!(
                        "failed solving constraint: {} != {}",
                        Ty::from(a),
                        Ty::from(b),
                    ))
                    .with_help("try changing the value to match the expected type")
            }
            Self::ExplicitTypeArgumentsNotAllowed(span) => {
                diagnostic
                    .with_section(Section::new()
                        .with_label(Label::at(span)
                            .with_message("cannot use explicit type arguments here")
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
            Self::PatternMismatch { pat, pat_span, value, value_span } => {
                let mut section = Section::new()
                    .with_label(Label::at(pat_span)
                        .with_message(format!("{pat} expected here"))
                    )
                    .with_note(format!("cannot bind {value} to {pat}"));

                if let Some(value_span) = value_span {
                    section = section.with_label(Label::at(value_span)
                        .with_message(format!("tried binding {value} to {pat} here"))
                    );
                }

                diagnostic.with_section(section)
                    .with_fix(Fix::new(Action::Replace(pat_span, "value".to_string()))
                        .with_message(
                            "try changing the pattern to match the type. \
                            you can also try binding to an identifier instead:",
                        )
                        .with_note("identifiers can be used to bind values of any type")
                    )
            }
            Self::ConditionNotBool(ty) => {
                diagnostic
                    .with_section(Section::new()
                        .with_label(Label::at(ty.span())
                            .with_message(format!("expected `bool`, found `{ty}` instead"))
                        )
                        .with_note("conditions must be of type `bool`")
                    )
                    // TODO: only show this fix if a cast can actually be performed
                    .with_fix(Fix::new(Action::InsertAfter(ty.span(), " to bool".to_string()))
                        .with_message("try casting the value to `bool` if possible")
                    )
            }
            Self::ReassignmentToImmutable(reass_span, def) => {
                diagnostic
                    .with_section(Section::new()
                        .with_label(Label::at(reass_span)
                            .with_message(format!("cannot reassign to immutable variable `{def}`"))
                        )
                    )
                    .with_section(Section::new()
                        .with_header(format!("`{def}` was defined as immutable here:"))
                        .with_label(Label::at(def.span())
                            .with_message(format!("`{def}` defined as immutable here"))
                        )
                    )
                    .with_fix(Fix::new(Action::InsertBefore(def.span(), "mut ".to_string()))
                        .with_message("add `mut` to make the binding mutable")
                    )
            }
            Self::CouldNotInferType(def) => {
                diagnostic
                    .with_section(Section::new()
                        .with_label(Label::at(def.span())
                            .with_message(format!("cannot infer type for variable `{def}`"))
                        )
                        .with_note("all types must be known at compile-time")
                    )
                    .with_help("try specifying the type explicitly")
            }
            Self::InvalidBreak(span, label) => {
                let label = if let Some(label) = label {
                    Label::at(label.span())
                        .with_context_span(span)
                        .with_message(format!("cannot continue from block `{label}` because it is not a loop"))
                } else {
                    Label::at(span).with_message("cannot break or continue here")
                };
                diagnostic
                    .with_section(Section::new()
                        .with_label(label)
                        .with_note("you may only break or continue from loops")
                    )
            }
            Self::InvalidReturn(span) => {
                diagnostic
                    .with_section(Section::new()
                        .with_label(Label::at(span)
                            .with_message("cannot return here")
                        )
                        .with_note("you may only return from functions")
                    )
            }
            Self::LabelNotFound(Spanned(label, span)) => {
                diagnostic
                    .with_section(Section::new()
                        .with_label(Label::at(span)
                            .with_message(format!("cannot find block label `{label}` from this scope"))
                        )
                    )
                    .with_help("check the spelling of the label")
            }
        };
        diagnostic
    }
}
