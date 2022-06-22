use super::{Token, Source, Span};

use std::collections::HashSet;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::io::Write;
use std::ops::Range;
use std::string::ToString;

use ariadne;

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
            Self::Token(token) => write!(f, "{}", token),
            Self::Char(c) => write!(f, "{:?}", c),
            Self::Literal => write!(f, "literal"),
            Self::Identifier => write!(f, "identifier"),
            Self::End => write!(f, "end"),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
#[allow(clippy::module_name_repetitions)]
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

#[derive(Copy, Clone, Debug)]
pub enum HintAction {
    Replace(String),
    Insert(String),
    Remove(String),
    None,
}

#[derive(Debug)]
pub struct Hint {
    pub message: String,
    pub action: HintAction,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Error {
    pub kind: ErrorKind,
    pub span: Span,
    pub expected: HashSet<TargetKind>,
    pub label: Option<&'static str>,
    pub message: String,
    pub hint: Option<Hint>,
}

impl Error {
    #[must_use]
    pub fn placeholder() -> Self {
        Self {
            kind: ErrorKind::Custom,
            span: Span::default(),
            expected: HashSet::new(),
            label: None,
            message: String::new(),
            hint: None,
        }
    }

    #[must_use]
    pub fn custom(span: Span, message: impl Display) -> Self {
        Self {
            kind: ErrorKind::Custom,
            span,
            expected: HashSet::new(),
            label: None,
            message: message.to_string(),
            hint: None,
        }
    }

    #[must_use]
    pub fn unexpected_token(span: Span, token: &Token) -> Self {
        Self {
            kind: ErrorKind::Unexpected(TargetKind::Token(token.clone())),
            span,
            expected: HashSet::new(),
            label: None,
            message: format!("unexpected token {}", token),
            hint: None,
        }
    }

    #[must_use]
    pub fn no_const_mut(span: Span) -> Self {
        Self {
            kind: ErrorKind::Custom,
            span,
            expected: HashSet::new(),
            label: None,
            message: format!("cannot declare as 'const mut'"),
            hint: Some(Hint {
                message: "consider using 'let mut' instead".to_string(),
                action: HintAction::Replace("let mut".to_string()),
            }),
        }
    }

    pub fn write<C>(self, cache: C, writer: impl Write)
    where
        C: ariadne::Cache<Source>,
    {
        use ariadne::{Report, ReportKind, Label, ColorGenerator, Fmt};

        let mut colors = ColorGenerator::new();
        let primary = colors.next();

        let report = Report::build(ReportKind::Error, self.span.source, self.span.range.start())
            .with_code(0)
            .with_message(self.message)
            .with_label(Label::new(self.span.clone())
                .with_message("error occurred here")
                .with_color(primary)
            );

        let report = if let Some(hint) = self.hint {
            report.with_help(hint.message)
        } else {
            report
        };

        report
            .finish()
            .write(cache, writer)
            .unwrap();
    }

    pub fn print(self) {
        self.write()
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
            .map(|x| x.map_or(TargetKind::End, Into::into))
            .collect::<HashSet<TargetKind>>();

        let expected_message = expected
            .iter()
            .map(ToString::to_string)
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
                .map_or(ErrorKind::UnexpectedEnd, ErrorKind::Unexpected),
            span,
            expected,
            label: None,
            message: format!(
                "expected {}, found {} instead",
                expected_message,
                found.map_or_else(String::new, |found| found.into().to_string())
            ),
            hint: None,
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
                expected.into().to_string()
            }),
            hint: Some(Hint {
                message: "add the missing delimiter".to_string(),
                action: HintAction::Insert(expected.into().to_string()),
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
            span: self.span.merge(other.span),
            expected: self.expected,
            label: self.label,
            message: self.message,
            hint: self.hint.or(other.hint),
        }
    }
}
