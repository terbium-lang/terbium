use crate::{
    ast::Delimiter,
    span::{Span, Spanned, Src},
    token::{Error as TokenizationError, TokenInfo},
};
use std::ops::Range;
use std::{
    fmt::{self, Display, Formatter, Result as FmtResult},
    string::ToString,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TargetKind {
    Nothing,
    Char(char),
    Token(TokenInfo),
    Keyword(&'static str),
    OpeningDelimiter(Delimiter),
    ClosingDelimiter(Delimiter),
    Literal,
    Identifier,
    Expression,
    Unknown,
    End,
    OneOf(Vec<Self>),
}

impl TargetKind {
    #[must_use]
    pub fn hint(&self) -> String {
        match self {
            Self::Nothing => String::new(),
            Self::Char(c) => c.to_string(),
            Self::Token(token) => token.to_string(),
            Self::Keyword(k) => k.to_string(),
            Self::OpeningDelimiter(d) => d.open().to_string(),
            Self::ClosingDelimiter(d) => d.close().to_string(),
            Self::Literal => "\"example\"".to_string(),
            Self::Identifier => "example".to_string(),
            Self::Expression => "value".to_string(),
            Self::Unknown => "<unknown>".to_string(),
            Self::End => "<EOF>".to_string(),
            Self::OneOf(targets) => targets[0].hint(),
        }
    }
}

impl From<char> for TargetKind {
    fn from(c: char) -> Self {
        Self::Char(c)
    }
}

impl From<TokenInfo> for TargetKind {
    fn from(t: TokenInfo) -> Self {
        macro_rules! delim {
            ($variant:ident $delim:ident) => {{
                Self::$variant(Delimiter::$delim)
            }};
        }

        match t {
            TokenInfo::LeftParen => delim!(OpeningDelimiter Paren),
            TokenInfo::RightParen => delim!(ClosingDelimiter Paren),
            TokenInfo::LeftBrace => delim!(OpeningDelimiter Brace),
            TokenInfo::RightBrace => delim!(ClosingDelimiter Brace),
            TokenInfo::LeftBracket => delim!(OpeningDelimiter Bracket),
            TokenInfo::RightBracket => delim!(ClosingDelimiter Bracket),
            _ => Self::Token(t),
        }
    }
}

impl<T: Into<Self>> From<Spanned<T>> for TargetKind {
    fn from(t: Spanned<T>) -> Self {
        t.0.into()
    }
}

impl From<Delimiter> for TargetKind {
    fn from(d: Delimiter) -> Self {
        Self::ClosingDelimiter(d)
    }
}

impl Display for TargetKind {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        match self {
            Self::Nothing => f.write_str("nothing"),
            Self::Token(token) => write!(f, "{token}"),
            Self::Char(c) => write!(f, "{c:?}"),
            Self::Keyword(k) => write!(f, "{k}"),
            Self::OpeningDelimiter(d) => write!(f, "opening {d}"),
            Self::ClosingDelimiter(d) => write!(f, "closing {d}"),
            Self::Literal => write!(f, "literal"),
            Self::Identifier => write!(f, "identifier"),
            Self::Expression => write!(f, "expression"),
            Self::Unknown => write!(f, "unknown"),
            Self::End => write!(f, "end of file"),
            Self::OneOf(targets) => {
                if targets.len() == 1 {
                    return write!(f, "{}", unsafe {
                        // SAFETY: `targets` is guaranteed to be non-empty.
                        targets.get_unchecked(0)
                    });
                }

                let mut targets = targets.iter();
                write!(f, "one of {}", targets.next().unwrap())?;
                for target in targets {
                    write!(f, ", {target}")?;
                }
                Ok(())
            }
        }
    }
}

/// Represents information about an error that occured during parsing.
#[derive(Debug, PartialEq, Eq)]
pub enum ErrorInfo {
    /// An error occured during tokenization.
    Tokenization(TokenizationError),
    /// An unexpected token or item was encountered.
    Unexpected {
        /// The span where the unexpected item was encountered.
        span: Span,
        /// The item that was expected.
        expected: TargetKind,
        /// The item that was found.
        found: TargetKind,
    },
    /// Encountered the end of the input.
    UnexpectedEof,
    /// An unmatched closing delimiter was encountered.
    UnmatchedDelimiter {
        /// The delimiter that was not closed.
        start: TargetKind,
        /// The span of the opening delimiter.
        opening_span: Span,
        /// The delimiter that was expected.
        expected: TargetKind,
        /// The full span of the unmatched item being delimited.
        span: Span,
    },
    /// Unknown escape sequence encountered.
    UnknownEscapeSequence(char, Span),
    /// Invalid hex escape sequence encountered.
    InvalidHexEscapeSequence(String, Span),
    /// Encountered a positional argument after a named argument.
    UnexpectedPositionalArgument(Span),
}

/// An action to take to fix an error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HintAction {
    /// Replace the span with the given string.
    Replace(Span, String),
    /// Insert the string before the span.
    InsertBefore(Span, String),
    /// Insert the string after the span.
    InsertAfter(Span, String),
    /// Remove the contents at the span.
    Remove(Span),
    /// Do nothing.
    None,
}

/// A hint for how to fix the error.
#[derive(Debug, PartialEq, Eq)]
pub struct Hint {
    /// The message to display to the user.
    pub message: String,
    /// The action to take to fix the error.
    pub action: HintAction,
}

impl Display for ErrorInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tokenization(err) => err.fmt(f),
            Self::Unexpected {
                span,
                expected,
                found,
            } => {
                write!(f, "{span}: expected {expected}, found {found}")
            }
            Self::UnexpectedEof => write!(f, "unexpected end of file"),
            Self::UnmatchedDelimiter { start, span, .. } => {
                write!(f, "{span}: unmatched delimiter {:?}", start.hint())
            }
            Self::UnknownEscapeSequence(c, span) => {
                write!(f, "{span}: unknown escape sequence \\{c}")
            }
            Self::InvalidHexEscapeSequence(s, span) => {
                write!(f, "{span}: invalid hex escape sequence \\x{s}")
            }
            Self::UnexpectedPositionalArgument(span) => {
                write!(f, "{span}: positional argument found after named argument")
            }
        }
    }
}

impl std::error::Error for ErrorInfo {}

/// An error that occurred during parsing.
#[derive(Debug, PartialEq, Eq)]
pub struct Error {
    /// The kind and information of the error that occurred.
    pub info: ErrorInfo,
    /// The span of the error.
    pub span: Span,
    /// A label for the error.
    pub label: Option<&'static str>,
    /// A hint for how to fix the error.
    pub hint: Option<Hint>,
    /// Additional notes for the error.
    pub notes: Vec<String>,
}

impl Error {
    /// Unexpected token or item.
    #[must_use]
    pub fn unexpected(
        span: Span,
        expected: Option<impl Into<TargetKind>>,
        found: &(impl Into<TargetKind> + Clone),
    ) -> Self {
        Self {
            info: ErrorInfo::Unexpected {
                span,
                expected: expected.map_or(TargetKind::Unknown, Into::into),
                found: found.clone().into(),
            },
            span,
            label: None,
            hint: Some(Hint {
                action: HintAction::Remove(span),
                message: "remove this".to_owned(),
            }),
            notes: Vec::new(),
        }
    }

    /// Unexpected EOF.
    #[must_use]
    pub fn unexpected_eof(
        eof: Span,
        insert_hint: Option<(impl Into<TargetKind>, impl ToString)>,
    ) -> Self {
        Self {
            info: ErrorInfo::UnexpectedEof,
            span: eof,
            label: None,
            hint: insert_hint.map(|(hint, msg)| Hint {
                action: HintAction::InsertAfter(eof, hint.into().hint()),
                message: msg.to_string(),
            }),
            notes: Vec::new(),
        }
    }

    /// Unknown escape sequence.
    #[must_use]
    pub fn unknown_escape_sequence(c: char, span: Span) -> Self {
        Self {
            info: ErrorInfo::UnknownEscapeSequence(c, span),
            span,
            label: None,
            hint: Some(escape_the_escape_sequence(span)),
            notes: Vec::new(),
        }
    }

    /// Invalid hex escape sequence.
    #[must_use]
    pub fn invalid_hex_escape_sequence(s: String, span: Span) -> Self {
        Self {
            info: ErrorInfo::InvalidHexEscapeSequence(s, span),
            span,
            label: None,
            hint: Some(escape_the_escape_sequence(span)),
            notes: Vec::new(),
        }
    }

    /// Unexpected positional argument.
    #[must_use]
    pub fn unexpected_positional_argument(span: Span) -> Self {
        Self {
            info: ErrorInfo::UnexpectedPositionalArgument(span),
            span,
            label: None,
            hint: Some(Hint {
                action: HintAction::InsertBefore(span, "name: ".to_string()),
                message: "add the name of the parameter here".to_string(),
            }),
            notes: Vec::new(),
        }
    }

    /// Adds a note to the error.
    #[must_use]
    pub fn note<S: ToString>(mut self, note: S) -> Self {
        self.notes.push(note.to_string());
        self
    }
}

#[inline]
fn escape_the_escape_sequence(span: Span) -> Hint {
    Hint {
        action: HintAction::InsertBefore(span, "\\".to_string()),
        message: "escape the escape sequence".to_string(),
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
            .collect::<Vec<_>>();
        let count = expected.len();

        let expected = match count {
            0 => TargetKind::Nothing,
            // SAFETY: `expected` is not empty.
            1 => unsafe { expected.into_iter().next().unwrap_unchecked() },
            _ => TargetKind::OneOf(expected),
        };
        let hint = match count {
            _ if found.is_none() && count > 0 => Hint {
                action: HintAction::InsertAfter(span, expected.hint()),
                message: "add required tokens".to_string(),
            },
            0 => Hint {
                action: HintAction::Remove(span),
                message: "remove this".to_string(),
            },
            _ => Hint {
                action: HintAction::Replace(span, expected.hint()),
                message: "replace with this".to_string(),
            },
        };

        Self {
            info: found
                .map(Into::into)
                .map_or(ErrorInfo::UnexpectedEof, |target| ErrorInfo::Unexpected {
                    span,
                    expected,
                    found: target,
                }),
            span,
            label: None,
            hint: Some(hint),
            notes: Vec::new(),
        }
    }

    fn unclosed_delimiter(
        span: Self::Span,
        start: T,
        before_span: Self::Span,
        expected: T,
        _before: Option<T>,
    ) -> Self {
        let expected = expected.into();
        let action = HintAction::InsertAfter(span, expected.hint());

        Self {
            info: ErrorInfo::UnmatchedDelimiter {
                start: start.into(),
                opening_span: before_span,
                expected,
                span,
            },
            span,
            label: None,
            hint: Some(Hint {
                action,
                message: "add the missing delimiter".to_string(),
            }),
            notes: Vec::new(),
        }
    }

    fn with_label(mut self, label: Self::Label) -> Self {
        self.label.get_or_insert(label);
        self
    }

    fn merge(self, other: Self) -> Self {
        Self {
            info: self.info,
            span: self.span.merge(other.span),
            label: self.label.or(other.label),
            hint: self.hint.or(other.hint),
            notes: self.notes.into_iter().chain(other.notes).collect(),
        }
    }
}
