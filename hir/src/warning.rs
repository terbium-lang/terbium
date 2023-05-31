use common::span::{Span, Spanned};
use diagnostics::{Action, Diagnostic, Fix, Label, Section, Severity};

/// A warning that may be emitted by the compiler. Warnings are not fatal errors and may be ignored,
/// but may indicate a problem with the code.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Warning {
    /// An item is unused (e.g. a type or function).
    Unused(Spanned<String>),
    /// A variable is never used before it is dropped.
    UnusedVariable(Spanned<String>),
    /// A variable is declared as mutable, but it is never mutated.
    /// The second span is the span of the `mut` keyword.
    UnusedMut(Spanned<String>, Span),
    /// The name of a variable, function, module, or member (field or method) is not written in
    /// snake_case.
    NotSnakeCase(Spanned<String>),
    /// The name of a constant is not written in SCREAMING_SNAKE_CASE.
    NotScreamingSnakeCase(Spanned<String>),
    /// The name of a type or enum variant is not written in PascalCase or UpperCamelCase.
    NotPascalCase(Spanned<String>),
    /// The name of an item contains non-ASCII characters.
    NonAsciiName(Spanned<String>),
    /// A top-level variable is declared as mutable (global mutable state).
    GlobalMutableState(Spanned<String>),
    /// A complex function was marked as inline.
    InlineOnComplexFunction(Spanned<String>),
    /// Code is unreachable. Represented as (break_span, unreachable_span).
    UnreachableCode(Option<Span>, Span),
}

impl Warning {
    /// A human-readable message describing the warning.
    pub const fn message(&self) -> &'static str {
        match self {
            Self::Unused(_) => "unused item",
            Self::UnusedVariable(_) => "unused variable",
            Self::UnusedMut(_, _) => "variable marked as mutable is never mutated",
            Self::NotSnakeCase(_) => "name is not snake_case",
            Self::NotScreamingSnakeCase(_) => {
                "name of constant or alias is not SCREAMING_SNAKE_CASE"
            }
            Self::NotPascalCase(_) => {
                "name of type or enum variant is not PascalCase or UpperCamelCase"
            }
            Self::NonAsciiName(_) => "name contains non-ASCII characters",
            Self::GlobalMutableState(_) => "top-level variable is declared as mutable",
            Self::InlineOnComplexFunction(_) => "inline on complex function",
            Self::UnreachableCode(..) => "unreachable code",
        }
    }

    /// The code identifying the warning in the warning index. This code is used when specifying
    /// warnings to suppress.
    ///
    /// # Example
    /// ```text
    /// @suppress(unreachable_code) // Since we suppress `unreachable_code`...
    /// func sample() {
    ///     return;
    ///     println("This code is unreachable"); // ...no warning is emitted here.
    /// }
    /// ```
    pub const fn warning_code(&self) -> &'static str {
        match self {
            Self::Unused(_) => "unused",
            Self::UnusedVariable(_) => "unused_variables",
            Self::UnusedMut(_, _) => "unused_mut",
            Self::NotSnakeCase(_) => "not_snake_case",
            Self::NotScreamingSnakeCase(_) => "not_screaming_snake_case",
            Self::NotPascalCase(_) => "not_pascal_case",
            Self::NonAsciiName(_) => "non_ascii_name",
            Self::GlobalMutableState(_) => "global_mutable_state",
            Self::InlineOnComplexFunction(_) => "inline_on_complex_function",
            Self::UnreachableCode(..) => "unreachable_code",
        }
    }

    pub fn into_diagnostic(self) -> Diagnostic {
        let mut diagnostic =
            Diagnostic::new(Severity::Warning(self.warning_code()), self.message());
        diagnostic = match self {
            Self::Unused(name) => diagnostic
                .with_section(
                    Section::new()
                        .with_label(
                            Label::at(name.span())
                                .with_message(format!("item `{name}` defined here")),
                        )
                        .with_note(format!("`{name}` is never used")),
                )
                .with_fix(
                    Fix::new(Action::InsertBefore(name.span(), '_'.to_string()))
                        .with_message("prefix ignored items with `_`"),
                ),
            Self::UnusedVariable(name) => diagnostic
                .with_section(
                    Section::new()
                        .with_label(
                            Label::at(name.span())
                                .with_message(format!("variable `{name}` declared here")),
                        )
                        .with_note(format!("`{name}` is never used")),
                )
                .with_fix(
                    Fix::new(Action::InsertBefore(name.span(), '_'.to_string()))
                        .with_message("prefix ignored variables with `_`"),
                ),
            Self::UnusedMut(name, mut_span) => diagnostic
                .with_section(
                    Section::new()
                        .with_label(
                            Label::at(mut_span)
                                .with_message(format!("variable `{name}` declared as mutable here"))
                                .with_context_span(mut_span.merge(name.span())),
                        )
                        .with_note(format!("`{name}` is never mutated")),
                )
                .with_fix(
                    Fix::new(Action::Remove(mut_span))
                        .with_message("mark the variable as immutable by removing the `mut`"),
                ),
            Self::UnreachableCode(break_span, span) => {
                let mut section = Section::new();
                if let Some(break_span) = break_span {
                    section = section.with_label(
                        Label::at(break_span)
                            .with_message("block exits here")
                            .with_context_span(break_span),
                    );
                }
                diagnostic.with_section(
                    section.with_label(
                        Label::at(span).with_message("this code will never be executed"),
                    ),
                )
            }
            _ => diagnostic, // TODO
        };
        diagnostic
    }
}
