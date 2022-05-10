use super::*;

use std::collections::HashSet;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::ops::Range;

type Span = Range<usize>;

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum TargetKind {
    Char(char),
    Token(Token),
    Literal,
    Identifier,
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
            TargetKind::Identifier => write!(f, "identifier"),
            TargetKind::End => write!(f, "end"),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ErrorKind {
    Custom,
    UnexpectedEnd,
    Unexpected(TargetKind),
    Unclosed {
        start: TargetKind,
        span: Span,
        before: Option<TargetKind>,
    },
}

#[derive(Debug, PartialEq, Eq)]
pub struct Error {
    pub kind: ErrorKind,
    pub span: Span,
    pub expected: HashSet<TargetKind>,
    pub label: Option<&'static str>,
    pub message: String,
}

impl Error {
    pub fn placeholder() -> Self {
        Self {
            kind: ErrorKind::Custom,
            span: 0..0,
            expected: HashSet::new(),
            label: None,
            message: String::new(),
        }
    }

    pub fn custom(span: Span, message: impl Display) -> Self {
        Self {
            kind: ErrorKind::Custom,
            span,
            expected: HashSet::new(),
            label: None,
            message: message.to_string(),
        }
    }

    pub fn unexpected_token(span: Span, token: Token) -> Self {
        Self {
            kind: ErrorKind::Unexpected(TargetKind::Token(token.clone())),
            span,
            expected: HashSet::new(),
            label: None,
            message: format!("unexpected token {}", token),
        }
    }

    pub fn print(&self) {
        // TODO: more comprehensive error messages
        eprintln!("[{:?}] {}", self.span, self.message);
    }
}

impl<T: Into<TargetKind> + Clone> chumsky::Error<T> for Error {
    type Span = Span;
    type Label = &'static str;

    fn expected_input_found<I: IntoIterator<Item = Option<T>>>(
        span: Self::Span,
        expected: I,
        found: Option<T>,
    ) -> Self {
        let expected = expected
            .into_iter()
            .map(|x| x.map(Into::into).unwrap_or(TargetKind::End))
            .collect::<HashSet<TargetKind>>();

        let expected_message = expected
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>();

        let joined;
        let expected_message = if expected_message.len() == 1 {
            expected_message.first().unwrap().as_str()
        } else {
            joined = expected_message.join(" or ");
            joined.as_str()
        };

        Self {
            kind: found
                .clone()
                .map(Into::into)
                .map(ErrorKind::Unexpected)
                .unwrap_or(ErrorKind::UnexpectedEnd),
            span,
            expected,
            label: None,
            message: format!(
                "expected {}{}",
                expected_message,
                if let Some(found) = found {
                    found.into().to_string()
                } else {
                    String::new()
                }
            ),
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
            expected: std::iter::once(expected.clone().into()).collect(),
            label: None,
            message: format!("unclosed delimiter: expected {}", {
                expected.clone().into().to_string()
            }),
        }
    }

    fn with_label(mut self, label: Self::Label) -> Self {
        self.label.get_or_insert(label);
        self
    }

    fn merge(mut self, other: Self) -> Self {
        for expected in other.expected {
            self.expected.insert(expected);
        }

        Self {
            kind: self.kind,
            span: self.span.start..other.span.end,
            expected: self.expected,
            label: self.label,
            message: self.message,
        }
    }
}
