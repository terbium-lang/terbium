use common::span::{Span, Spanned};

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
}
