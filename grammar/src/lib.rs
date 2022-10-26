#![allow(
    clippy::uninlined_format_args,
    reason = "editor will complain about unused variables otherwise"
)]
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    reason = "not really necessary"
)]
#![allow(
    clippy::module_name_repetitions,
    reason = "items are not usually accessed directly"
)]
#![feature(const_trait_impl)]
#![feature(box_syntax)]
#![feature(lint_reasons)]
#![feature(never_type)]

pub mod ast;
pub mod parser;
mod token;

pub use ast::Spanned;
pub use parser::Parser;
pub use token::{
    Error as TokenizationError, Radix, StringLiteralFlags, Token, TokenInfo, TokenReader,
};

/// Represents the span of a token or a node in the AST. Can be represented as [start, end).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Span {
    /// The index of the first byte of the span.
    pub start: usize,
    /// One more than the index of the last byte of the span.
    pub end: usize,
}

mod sealed {
    use std::ops::{Range, RangeInclusive};

    pub trait RangeInclusiveExt {
        fn to_range(self) -> RangeInclusive<usize>;
    }

    impl RangeInclusiveExt for RangeInclusive<usize> {
        fn to_range(self) -> RangeInclusive<usize> {
            self
        }
    }

    impl RangeInclusiveExt for Range<usize> {
        #[allow(clippy::range_minus_one, reason = "required for conversion")]
        fn to_range(self) -> RangeInclusive<usize> {
            self.start..=(self.end - 1)
        }
    }
}

#[allow(
    clippy::len_without_is_empty,
    reason = "semantically incorrect to include is_empty method"
)]
impl Span {
    /// Creates a new span from the given start, end, and source.
    #[must_use]
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Creates a new span from the given range and source.
    #[must_use]
    pub fn from_range<R: sealed::RangeInclusiveExt>(range: R) -> Self {
        let range = range.to_range();
        Self::new(*range.start(), *range.end() + 1)
    }

    /// Creates a single-length span.
    #[must_use]
    pub const fn single(index: usize) -> Self {
        Self::new(index, index + 1)
    }

    /// Returns the length of the span.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.end - self.start
    }

    /// Merges this span with another.
    #[must_use]
    pub const fn merge(self, other: Self) -> Self {
        Self::new(self.start.min(other.start), self.end.max(other.end))
    }

    /// Merges an iterator of spans.
    ///
    /// # Panics
    /// * If the iterator is empty.
    #[must_use]
    pub fn from_spans<I: IntoIterator<Item = Self>>(spans: I) -> Self {
        spans
            .into_iter()
            .reduce(Self::merge)
            .expect("Cannot create a span from an empty iterator")
    }
}

impl<R: sealed::RangeInclusiveExt> From<R> for Span {
    fn from(range: R) -> Self {
        Self::from_range(range)
    }
}
