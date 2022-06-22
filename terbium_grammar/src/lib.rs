#![feature(trait_alias)]

pub mod ast;
pub mod error;
pub mod token;

pub use crate::ast::{Body, Expr, Node, ParseInterface, TypeExpr};
pub use crate::error::*;
pub use crate::token::{get_lexer as tokenizer, Operator, Token};

use std::{
    cmp::{Eq, PartialEq},
    fmt::{Debug, Display, Formatter, Result as FmtResult, Write},
    ops::{Deref, DerefMut, Range},
    path::{PathBuf, Path},
};

use chumsky::Span as ChumskySpan;
use ariadne::Span as AriadneSpan;

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Source(Vec<String>);

impl Source {
    pub fn repl() -> Self {
        Self(vec!["<repl>".to_string()])
    }

    pub fn from_path<P>(path: P) -> Self
    where
        P: AsRef<Path>
    {
        Self(
            path
                .as_ref()
                .iter()
                .map(|c| c.to_string_lossy().into_owned())
                .collect()
        )
    }

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

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Span {
    pub(crate) source: Source,
    pub(crate) range: (usize, usize),
}

impl Span {
    pub fn single(source: Source, loc: usize) -> Self {
        Self::new(source, loc..loc + 1)
    }

    pub fn src(&self) -> Source { self.source.clone() }

    pub fn range(&self) -> Range<usize> { self.start()..self.end() }

    pub fn merge(self, other: Self) -> Self {
        assert_eq!(self.source, other.source, "cannot merge spans with different sources");

        Self {
            source: self.source,
            range: (self.start().min(other.start()), self.end().max(other.end())),
        }
    }
}

impl Default for Span {
    fn default() -> Self {
        Self::new(Source::default(), 0..0)
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
        Self { source, range: (range.start, range.end) }
    }

    fn context(&self) -> Self::Context { self.source }

    fn start(&self) -> Self::Offset { self.range.0 }

    fn end(&self) -> Self::Offset { self.range.1 }
}

impl AriadneSpan for Span {
    type SourceId = Source;

    fn source(&self) -> &Self::SourceId {
        &self.source
    }

    fn start(&self) -> usize { self.range.0 }

    fn end(&self) -> usize { self.range.1 }
}

#[derive(Clone, Debug)]
pub struct Spanned<T> {
    inner: Box<T>,
    span: Span,
}

impl<T> Spanned<T> {
    pub fn new(inner: T, span: Span) -> Self {
        Self { inner: Box::new(inner), span }
    }

    pub fn node(&self) -> &T {
        &self.inner
    }

    pub fn node_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    pub fn into_node(self) -> T {
        *self.inner
    }

    pub fn span(&self) -> Span { self.span }

    pub fn map<U, F>(self, f: F) -> Self<U>
    where
        F: FnOnce(T) -> U,
    {
        Self::new(f(*self.inner), self.span)
    }

    pub fn map_span<F: FnOnce(Span) -> Span>(self, f: F) -> Self {
        Self {
            inner: self.inner,
            span: f(self.span),
        }
    }

    pub fn span_mut(&mut self) -> &mut Span {
        &mut self.span
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
        write!(f, "{:?} @ {:?}", self.inner, self.span)
    }
}
