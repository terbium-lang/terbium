use super::*;
use chumsky::prelude::*;

use std::fmt::{Display, Formatter, Result as FmtResult};
use std::ops::Range;

type Span = Range<usize>;

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum TargetKind {
    Char(char),
    Token(Token),
    Literal,
    Ident,
    End,
}

impl From<char> for TargetKind {
    fn from(c: char) -> Self {
        Self::Char(c)
    }
}

impl From<Token> for TargetKind {
    fn from(t: Token) -> Self {
        Self::Token(t)
    }
}

impl Display for TargetKind {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        match self {
            TargetKind::Token(token) => write!(f, "{}", token),
            TargetKind::Char(c) => write!(f, "{:?}", c),
            TargetKind::Literal => write!(f, "literal"),
            TargetKind::Ident => write!(f, "identifier"),
            TargetKind::End => write!(f, "end"),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum ErrorKind {
    UnexpectedEnd,
    Unexpected(TargetKind),
    Unclosed {
        start: TargetKind,
        span: Span,
        before: Option<TargetKind>,
    },
}

#[derive(Debug, PartialEq)]
pub struct Error {
    pub kind: ErrorKind,
    pub span: Span,
    pub expected: Vec<ErrorKind>,
    pub label: Option<&'static str>,
}

impl<T: Into<TargetKind>> chumsky::Error<T> for Error {
    type Span = Span;
    type Label = &'static str;

    fn expected_input_found<Iter: IntoIterator<Item = Option<T>>>(
        span: Self::Span,
        expected: Iter,
        found: Option<T>,
    ) -> Self {
        Self {
            kind: found
                .map(Into::into)
                .map(ErrorKind::Unexpected)
                .unwrap_or(ErrorKind::UnexpectedEnd),
            span,
            expected: expected
                .into_iter()
                .map(|x| x.map(Into::into).unwrap_or(TargetKind::End))
                .collect(),
            label: None,
        }
    }

    fn unclosed_delimiter(
        span: Self::Span,
        start: T,
        before_span: Self::Span,
        expected: T,
        before: Option<T>,
    ) -> Self {
        Self {
            kind: ErrorKind::Unclosed {
                start: start.into(),
                span: before_span,
                before: before.map(Into::into),
            },
            span,
            expected: std::iter::once(expected.into()).collect(),
            label: None,
        }
    }

    fn with_label(mut self, label: Self::Label) -> Self {
        self.label.get_or_insert(label);
        self
    }

    fn merge(self, other: Self) -> Self {
        Error::merge(self, other)
    }
}
