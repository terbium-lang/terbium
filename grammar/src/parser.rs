use super::{ast::*, Span, Token, TokenInfo, TokenReader, TokenizationError};
use crate::parser::Error::UnmatchedDelimiter;
use crate::StringLiteralFlags;
use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};
use std::result::Result as StdResult;

type TokenResult = StdResult<Token, TokenizationError>;
type Result<T> = StdResult<T, Error>;

/// Represents an error that occured during parsing.
#[derive(Clone, Debug)]
pub enum Error {
    /// An error occured during tokenization.
    Tokenization(TokenizationError),
    /// An unexpected token was encountered.
    UnexpectedToken(Token),
    /// Encountered the end of the input.
    UnexpectedEof,
    /// An unmatched closing parenthesis was encountered.
    /// The span is the span of the starting delimiter.
    UnmatchedDelimiter(Delimiter, Span),
    /// Unknown escape sequence encountered.
    UnknownEscapeSequence(char, Span),
    /// Invalid hex escape sequence encountered.
    InvalidHexEscapeSequence(String, Span),
}

/// A token peeker that saves all token peeks.
#[derive(Clone)]
pub struct Peeker<I: Iterator + Clone>
where
    I::Item: Clone,
{
    /// The iterator that is being peeked.
    iter: I,
    /// A clone of the original iterator to peek tokens.
    peeker: I,
    /// The peeked tokens.
    peeked: VecDeque<I::Item>,
}

impl<I: Iterator + Clone> Peeker<I>
where
    I::Item: Clone,
{
    /// Creates a new peeker from the given iterator.
    pub fn new(iter: I) -> Self {
        Self {
            peeker: iter.clone(),
            peeked: VecDeque::new(),
            iter,
        }
    }

    /// Peeks the next token.
    pub fn peek(&mut self) -> Option<I::Item> {
        self.peeked.get(0).cloned().or_else(|| {
            let value = self.peeker.next()?;
            self.peeked.push_back(value.clone());
            Some(value)
        })
    }

    /// Peeks the nth token, where for n = 1 this is the same as peek().
    pub fn peek_nth(&mut self, n: usize) -> Option<I::Item> {
        if n == 1 {
            return self.peek();
        }
        if self.peeked.len() >= n {
            return self.peeked.get(n - 1).cloned();
        }
        for _ in 0..n - self.peeked.len() {
            self.peeked.push_back(self.peeker.next()?);
        }
        self.peeked.back().cloned()
    }
}

impl<I: Iterator + Clone> Iterator for Peeker<I>
where
    I::Item: Clone,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.peeked.pop_front().or_else(|| self.iter.next())
    }
}

impl<I: Iterator + Clone> Deref for Peeker<I>
where
    I::Item: Clone,
{
    type Target = I;

    fn deref(&self) -> &Self::Target {
        &self.iter
    }
}

impl<I: Iterator + Clone> DerefMut for Peeker<I>
where
    I::Item: Clone,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.iter
    }
}

/// Parses a token stream into an AST.
#[derive(Clone)]
pub struct Parser<I: Iterator<Item = TokenResult> + Clone> {
    tokens: Peeker<I>,
}

trait ResultExt<T> {
    fn map_token<U>(self, f: impl FnOnce(T, Span) -> U) -> Result<U>;
    fn and_then_token<U>(self, f: impl FnOnce(T, Span) -> Result<U>) -> Result<U>;
}

impl ResultExt<TokenInfo> for Result<Token> {
    fn map_token<U>(self, f: impl FnOnce(TokenInfo, Span) -> U) -> Result<U> {
        self.map(|token| f(token.info, token.span))
    }

    fn and_then_token<U>(self, f: impl FnOnce(TokenInfo, Span) -> Result<U>) -> Result<U> {
        self.and_then(|token| f(token.info, token.span))
    }
}

impl<T> ResultExt<T> for Result<Spanned<T>> {
    fn map_token<U>(self, f: impl FnOnce(T, Span) -> U) -> Result<U> {
        self.map(|Spanned(value, span)| f(value, span))
    }

    fn and_then_token<U>(self, f: impl FnOnce(T, Span) -> Result<U>) -> Result<U> {
        self.and_then(|Spanned(value, span)| f(value, span))
    }
}

macro_rules! consume_token {
    (@no_ws $self:expr, $p:pat) => {{
        loop {
            match $self.tokens.peek() {
                Some(Ok(t @ Token { info: $p, .. })) => {
                    $self.tokens.next();
                    break Ok(t);
                }
                Some(Ok(other)) => break Err(Error::UnexpectedToken(other)),
                Some(Err(e)) => break Err(Error::Tokenization(e)),
                None => break Err(Error::UnexpectedEof),
            }
        }
    }};
    ($self:expr, $p:pat) => {{
        loop {
            match $self.tokens.peek() {
                Some(Ok(Token {
                    info: TokenInfo::Whitespace,
                    ..
                })) => {
                    $self.tokens.next();
                }
                Some(Ok(t @ Token { info: $p, .. })) => {
                    $self.tokens.next();
                    break Ok(t);
                }
                Some(Ok(other)) => break Err(Error::UnexpectedToken(other)),
                Some(Err(e)) => break Err(Error::Tokenization(e)),
                None => break Err(Error::UnexpectedEof),
            }
        }
    }};
}

macro_rules! assert_token {
    ($info:ident: $p:pat => $out:expr) => {{
        match $info {
            $p => $out,
            _ => unreachable!("should not have encountered this token"),
        }
    }};
    (@unsafe $info:ident: $p:pat => $out:expr) => {{
        match $info {
            $p => $out,
            // SAFETY: upheld by the user
            _ => unsafe { std::hint::unreachable_unchecked() },
        }
    }};
}

impl<I: Iterator<Item = TokenResult> + Clone> Parser<I> {
    /// Skips all whitespace tokens.
    pub fn skip_ws(&mut self) {
        while let Some(Ok(Token {
            info: TokenInfo::Whitespace,
            ..
        })) = self.tokens.peek()
        {
            self.tokens.next();
        }
    }

    /// Resolves the string after escaping all escape sequences.
    pub fn resolve_string(
        content: String,
        flags: StringLiteralFlags,
        span: Span,
    ) -> Result<String> {
        if flags.is_raw() {
            return Ok(content);
        }

        let mut result = String::with_capacity(content.len());
        let mut chars = content.chars();
        let mut pos = span.start;

        while let Some(mut c) = chars.next() {
            if c == '\\' {
                pos += 1;

                macro_rules! hex_sequence {
                    ($length:literal) => {{
                        let sequence = chars.by_ref().take($length).collect::<String>();
                        let value = u32::from_str_radix(&sequence, 16).map_err(|_| {
                            Error::InvalidHexEscapeSequence(
                                sequence.clone(),
                                Span::new(pos - 1, pos + 1 + $length),
                            )
                        })?;

                        pos += $length;
                        char::from_u32(value).ok_or(Error::InvalidHexEscapeSequence(
                            sequence,
                            Span::new(pos - 1, pos + 1 + $length),
                        ))?
                    }};
                }

                c = match chars.next().ok_or(Error::UnexpectedEof)? {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    'b' => '\x08',
                    'f' => '\x0c',
                    '0' => '\0',
                    '\'' => '\'',
                    '"' => '"',
                    '\\' => '\\',
                    'x' => hex_sequence!(2),
                    'u' => hex_sequence!(4),
                    'U' => hex_sequence!(8),
                    c => return Err(Error::UnknownEscapeSequence(c, Span::new(pos - 1, pos + 1))),
                };
            }

            result.push(c);
            pos += 1;
        }

        Ok(result)
    }

    /// Parses and consumes the next atom. An atom is the most basic unit of an expression that
    /// cannot be broken down into other expressions any further.
    ///
    /// For example, 1 is an atom, as is "hello" - but 1 + 1 is not, since that can be further
    /// broken down into two expressions.
    pub fn consume_atom(&mut self) -> Result<Spanned<Atom>> {
        consume_token!(self, TokenInfo::IntLiteral(..)).map_token(|info, span| {
            Spanned(
                {
                    let (val, radix) =
                        assert_token!(@unsafe info: TokenInfo::IntLiteral(i, radix) => (i, radix));
                    Atom::Int(val, radix)
                },
                span,
            )
        })
        .or_else(|_| {
            consume_token!(self, TokenInfo::FloatLiteral(_))
                .map_token(|info, span| Spanned(
                    Atom::Float(assert_token!(@unsafe info: TokenInfo::FloatLiteral(i) => i)),
                    span,
                ))
        })
        .or_else(|_| {
            consume_token!(self, TokenInfo::Ident(..))
                .map_token(|info, span| Spanned(
                    match assert_token!(@unsafe info: TokenInfo::Ident(i) => i).as_str() {
                        "true" => Atom::Bool(true),
                        "false" => Atom::Bool(false),
                        i => Atom::Ident(i.to_string()),
                    },
                    span,
                ))
        })
        .or_else(|_| {
            consume_token!(self, TokenInfo::StringLiteral(..))
                .and_then_token(|info, span| Ok(Spanned(
                    Atom::String(assert_token!(
                        @unsafe info: TokenInfo::StringLiteral(s, flags, content_span)
                            => Self::resolve_string(s, flags, content_span)?
                    )),
                    span,
                )))
        })
    }

    /// Parses and consumes the next unary expression.
    pub fn consume_unary(&mut self) -> Result<Spanned<Expr>> {
        #[allow(unused_parens, reason = "parenthesis are needed here")]
        consume_token!(self, (TokenInfo::Plus | TokenInfo::Minus | TokenInfo::Not)).and_then_token(
            |info, span| {
                Ok(Spanned(
                    Expr::UnaryOp {
                        op: Spanned(
                            match info {
                                TokenInfo::Plus => UnaryOp::Plus,
                                TokenInfo::Minus => UnaryOp::Minus,
                                TokenInfo::Not => UnaryOp::Not,
                                // SAFETY: checked above in macro call
                                _ => unsafe { std::hint::unreachable_unchecked() },
                            },
                            span,
                        ),
                        expr: box self.consume_expr()?,
                    },
                    span,
                ))
            },
        )
    }

    /// Parses and consumes the next expression.
    pub fn consume_expr(&mut self) -> Result<Spanned<Expr>> {
        // Parenthesized expression
        consume_token!(self, TokenInfo::LeftParen)
            .and_then_token(|_, span| {
                let expr = self.consume_expr()?;
                consume_token!(self, TokenInfo::RightParen)
                    .map_err(|_| UnmatchedDelimiter(Delimiter::Paren, span))?;

                Ok(expr)
            })
            // Unary expression
            .or_else(|_| self.consume_unary())
            // Atom
            .or_else(|_| {
                self.consume_atom()
                    .map_token(|atom, span| Spanned(Expr::Atom(atom), span))
            })
    }

    /// Parses and consumes the next node.
    pub fn consume_node(&mut self) -> Result<Spanned<Node>> {
        self.skip_ws();
        todo!()
    }
}

impl<'a> Parser<TokenReader<'a>> {
    /// Creates a new parser from a string slice.
    #[must_use]
    pub fn from_str(source: &'a str) -> Self {
        Self::from(TokenReader::new(source))
    }
}

impl<I: IntoIterator<Item = TokenResult>> From<I> for Parser<I::IntoIter>
where
    I::IntoIter: Clone,
{
    fn from(tokens: I) -> Self {
        Self {
            tokens: Peeker::new(tokens.into_iter()),
        }
    }
}
