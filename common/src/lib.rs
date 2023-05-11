//! Common utilities for the compiler.

#![feature(lint_reasons)]
#![feature(const_trait_impl)]

pub mod span;

/// Pluralizes the given string
#[inline]
pub fn pluralize<'a>(count: usize, singular: &'a str, plural: &'a str) -> &'a str {
    if count == 1 {
        singular
    } else {
        plural
    }
}
