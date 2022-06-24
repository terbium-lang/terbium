#![feature(trait_alias)]

pub mod ast;
pub mod error;
pub mod token;

pub use crate::ast::{Body, Expr, Node, Param, ParseInterface, Target, TypeExpr};
pub use crate::error::*;
pub use crate::token::{get_lexer as tokenizer, Operator, Token};
pub use chumsky::Parser as ChumskyParser;
pub use chumsky::Stream as ChumskyStream;

use std::{
    cmp::{Eq, PartialEq},
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    ops::{Deref, DerefMut, Range},
    path::{Path, PathBuf},
};

use ariadne::Span as AriadneSpan;
use chumsky::Span as ChumskySpan;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Source(Vec<String>);

impl Source {
    #[must_use]
    pub fn repl() -> Self {
        Self(vec!["<repl>".to_string()])
    }

    #[must_use]
    pub fn from_path<P>(path: P) -> Self
    where
        P: AsRef<Path>,
    {
        Self(
            path.as_ref()
                .iter()
                .map(|c| c.to_string_lossy().into_owned())
                .collect(),
        )
    }

    #[must_use]
    pub fn to_path(&self) -> PathBuf {
        self.0.iter().map(ToString::to_string).collect()
    }
}

impl Default for Source {
    fn default() -> Self {
        Self(vec!["<unknown>".to_string()])
    }
}

impl Display for Source {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{}", self.0.clone().join("/"))
    }
}

impl Debug for Source {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{}", self)
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct Span {
    pub(crate) source: Source,
    pub range: (usize, usize),
}

impl Span {
    #[must_use]
    pub const fn single(source: Source, loc: usize) -> Self {
        #[allow(clippy::range_plus_one)] // From arnge takes Range instead of RangeInclusive
        Self::from_range(source, loc..loc + 1)
    }

    #[must_use]
    pub const fn from_range(source: Source, range: Range<usize>) -> Self {
        Self {
            source,
            range: (range.start, range.end),
        }
    }

    #[must_use]
    pub fn src(&self) -> Source {
        self.source.clone()
    }

    #[must_use]
    pub const fn range(&self) -> Range<usize> {
        self.range.0..self.range.1
    }

    /// Merge the span with other span.
    ///
    /// # Panics
    /// * Panic when other have a different source than self.
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        assert_eq!(
            self.source, other.source,
            "cannot merge spans with different sources"
        );

        let merged = Self {
            source: self.source,
            range: (
                self.range.0.min(other.range.0),
                self.range.1.max(other.range.1),
            ),
        };
        drop(other);

        merged
    }

    #[must_use]
    pub const fn start(&self) -> usize {
        self.range.0
    }

    #[must_use]
    pub const fn end(&self) -> usize {
        self.range.1
    }
}

impl Default for Span {
    fn default() -> Self {
        Self::from_range(Source::default(), 0..0)
    }
}

impl Debug for Span {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{:?}:{:?}", self.source, self.range())
    }
}

impl ChumskySpan for Span {
    type Context = Source;
    type Offset = usize;

    fn new(source: Self::Context, range: Range<usize>) -> Self {
        Self {
            source,
            range: (range.start, range.end),
        }
    }

    fn context(&self) -> Self::Context {
        self.source.clone()
    }

    fn start(&self) -> Self::Offset {
        self.range.0
    }

    fn end(&self) -> Self::Offset {
        self.range.1
    }
}

impl AriadneSpan for Span {
    type SourceId = Source;

    fn source(&self) -> &Self::SourceId {
        &self.source
    }

    fn start(&self) -> usize {
        self.range.0
    }

    fn end(&self) -> usize {
        self.range.1
    }
}

#[derive(Clone)]
pub struct Spanned<T> {
    pub(crate) inner: Box<T>,
    span: Span,
}

impl<T> Spanned<T> {
    #[must_use]
    pub fn new(inner: T, span: Span) -> Self {
        Self {
            inner: Box::new(inner),
            span,
        }
    }

    #[must_use]
    pub const fn node(&self) -> &T {
        &self.inner
    }

    #[must_use]
    pub fn node_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    #[must_use]
    pub fn into_node(self) -> T {
        *self.inner
    }

    #[must_use]
    pub fn span(&self) -> Span {
        self.span.clone()
    }

    #[must_use]
    pub fn map<U, F>(self, f: F) -> Spanned<U>
    where
        F: FnOnce(T) -> U,
    {
        Spanned::new(f(*self.inner), self.span)
    }

    #[must_use]
    pub fn map_span<F: FnOnce(Span) -> Span>(self, f: F) -> Self {
        Self {
            inner: self.inner,
            span: f(self.span),
        }
    }

    #[must_use]
    pub fn span_mut(&mut self) -> &mut Span {
        &mut self.span
    }

    #[must_use]
    pub const fn node_span(&self) -> (&T, &Span) {
        (&self.inner, &self.span)
    }

    #[must_use]
    pub fn into_node_span(self) -> (T, Span) {
        (*self.inner, self.span)
    }
}

impl<T> Deref for Spanned<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.node()
    }
}

impl<T> DerefMut for Spanned<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.node_mut()
    }
}

impl<T: PartialEq> PartialEq for Spanned<T> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<T: Debug> Debug for Spanned<T> {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        if f.alternate() {
            write!(f, "{:#?} @ {:?}", self.inner, self.span)
        } else {
            write!(f, "{:?} @ {:?}", self.inner, self.span)
        }
    }
}
