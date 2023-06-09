use internment::Intern;
use std::{
    borrow::Cow,
    fmt::{self, Debug, Display, Formatter},
    ops::Range,
    path::{Path, PathBuf},
};

/// Represents the span of a token or a node in the AST. Can be represented as [start, end).
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Span {
    /// The source of the span.
    pub src: Src,
    /// The index of the first byte of the span.
    pub start: usize,
    /// One more than the index of the last byte of the span.
    pub end: usize,
}

impl Display for Span {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}..{}", self.src, self.start, self.end)
    }
}

impl Debug for Span {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <Self as Display>::fmt(self, f)
    }
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
    pub const fn new(src: Src, start: usize, end: usize) -> Self {
        Self { src, start, end }
    }

    /// Creates a new span from the given range and source.
    #[must_use]
    pub fn from_range<R: sealed::RangeInclusiveExt>(src: Src, range: R) -> Self {
        let range = range.to_range();
        Self::new(src, *range.start(), *range.end() + 1)
    }

    /// Creates a single-length span.
    #[must_use]
    pub const fn single(src: Src, index: usize) -> Self {
        Self::new(src, index, index + 1)
    }

    /// Returns the length of the span.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.end - self.start
    }

    /// Converts this span into a range.
    #[must_use]
    pub fn range(&self) -> Range<usize> {
        self.start..self.end
    }

    /// Merges this span with another.
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        Self::new(
            self.src,
            self.start.min(other.start),
            self.end.max(other.end),
        )
    }

    /// Merges this span with another, or leaves the span unchanged if the other span is `None`.
    #[must_use]
    pub fn merge_opt(self, other: Option<Self>) -> Self {
        match other {
            Some(other) => self.merge(other),
            None => self,
        }
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

    /// Extends the span one character to the left.
    #[must_use]
    pub const fn extend_back(mut self) -> Self {
        self.start -= 1;
        self
    }

    /// Creates a new span from the same source.
    #[must_use]
    pub const fn get_span(&self, start: usize, end: usize) -> Self {
        Self::new(self.src, start, end)
    }

    /// Gets the last index of the span as another span.
    #[must_use]
    pub const fn last_span(&self) -> Self {
        self.get_span(self.end - 1, self.end)
    }
}

impl chumsky::Span for Span {
    type Context = Src;
    type Offset = usize;

    fn new(src: Self::Context, range: Range<Self::Offset>) -> Self {
        Self::from_range(src, range)
    }

    fn context(&self) -> Self::Context {
        self.src
    }

    fn start(&self) -> Self::Offset {
        self.start
    }

    fn end(&self) -> Self::Offset {
        self.end
    }
}

/// A compound of a span and a value.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Spanned<T>(pub T, pub Span);

impl<T> Spanned<T> {
    /// Returns the value.
    #[must_use]
    pub const fn value(&self) -> &T {
        &self.0
    }

    /// Returns the value as a mutable reference.
    #[must_use]
    pub fn value_mut(&mut self) -> &mut T {
        &mut self.0
    }

    /// Consumes this span and returns the inner value.
    #[must_use]
    #[allow(clippy::missing_const_for_fn, reason = "destructors can't be const")]
    pub fn into_value(self) -> T {
        self.0
    }

    /// Returns the span.
    #[must_use]
    pub const fn span(&self) -> Span {
        self.1
    }

    /// Converts from `Spanned<T>` to `Spanned<&T>`.
    #[must_use]
    pub const fn as_ref(&self) -> Spanned<&T> {
        Spanned(&self.0, self.1)
    }

    /// Consumes and maps the inner value.
    #[must_use]
    #[allow(
        clippy::missing_const_for_fn,
        reason = "closure is not supposed to be const"
    )]
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Spanned<U> {
        Spanned(f(self.0), self.1)
    }

    /// Consumes and maps the fallible inner value, mapping the span if the inner value is `Ok`
    /// and propagating the error otherwise.
    ///
    /// # Errors
    /// If the inner value is `Err`, the error is propagated.
    #[allow(
        clippy::missing_const_for_fn,
        reason = "closure is not supposed to be const"
    )]
    pub fn try_map<U, E>(self, f: impl FnOnce(T) -> Result<U, E>) -> Result<Spanned<U>, E> {
        Ok(Spanned(f(self.0)?, self.1))
    }

    /// Returns a tuple (value, span).
    #[must_use]
    #[allow(clippy::missing_const_for_fn, reason = "destructors can't be const")]
    pub fn into_inner(self) -> (T, Span) {
        (self.0, self.1)
    }

    /// Returns a tuple (&value, span).
    #[must_use]
    pub const fn as_inner(&self) -> (&T, Span) {
        (&self.0, self.1)
    }
}

impl<T, E> Spanned<Result<T, E>> {
    /// Converts from `Spanned<Result<T, E>>` to `Result<Spanned<T>, E>`,
    /// mapping the span if the inner value is `Ok` and propagating the error otherwise.
    ///
    /// # Errors
    /// If the inner value is `Err`, the error is propagated.
    pub fn transpose(self) -> Result<Spanned<T>, E> {
        self.0.map(|v| Spanned(v, self.1))
    }
}

impl<T: Display> Display for Spanned<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T: Debug> Debug for Spanned<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)?;
        write!(f, " @ {}", self.1)
    }
}

/// The source of a span.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Src {
    /// Unknown source, or source unapplicable.
    #[default]
    None,
    /// Read-eval-print loop.
    Repl,
    /// Path to file.
    Path(Intern<Vec<String>>),
}

impl Display for Src {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Self::None => f.write_str("<unknown>"),
            Self::Repl => f.write_str("<repl>"),
            Self::Path(path) => {
                if path.is_empty() {
                    f.write_str("<file>")
                } else {
                    write!(f, "{}", path.clone().join("/"))
                }
            }
        }
    }
}

impl Src {
    /// Creates a source from the provided path.
    #[must_use = "this returns the created Src"]
    pub fn from_path(path: impl AsRef<Path>) -> Self {
        Self::Path(Intern::new(
            path.as_ref()
                .iter()
                .map(|chunk| chunk.to_string_lossy().into_owned())
                .collect(),
        ))
    }

    /// Returns this source as a [`PathBuf`].
    pub fn as_path(&self) -> PathBuf {
        match self {
            Self::Path(path) => path.iter().map(ToString::to_string).collect(),
            _ => self.to_string().into(),
        }
    }
}

/// A complete source provider, storing the [`Src`] and its content.
#[derive(Clone)]
pub struct Provider<'a>(pub Src, pub Cow<'a, str>);

impl<'a> Provider<'a> {
    /// Creates a new provider with the given source and content.
    pub fn new(src: Src, content: impl Into<Cow<'a, str>>) -> Self {
        Self(src, content.into())
    }

    /// The source of the provider.
    #[must_use]
    pub const fn src(&self) -> Src {
        self.0
    }

    /// The content of the provider.
    #[must_use]
    pub fn content(&'a self) -> &'a str {
        self.1.as_ref()
    }

    /// The "eof" span of the provider content.
    #[must_use]
    pub fn eof(&self) -> Span {
        Span::single(self.src(), self.1.chars().count())
    }
}

impl Provider<'static> {
    /// Resolves a provider from a file path.
    ///
    /// # Errors
    /// * If the file cannot be read.
    pub fn read_from_file(path: impl AsRef<Path>) -> std::io::Result<Self> {
        Ok(Self(
            Src::from_path(path.as_ref()),
            Cow::Owned(std::fs::read_to_string(path)?),
        ))
    }
}

/// Resolves a provider from a file path at compile-time.
#[macro_export]
macro_rules! include_provider {
    ($path:literal $(,)?) => {{
        Provider(Src::from_path($path), include_str!($path))
    }};
}

pub use include_provider;
