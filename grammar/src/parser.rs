use super::{
    ast::{Atom, BinaryOp, Delimiter, Expr, Node, Spanned, UnaryOp},
    Span, StringLiteralFlags, Token, TokenInfo, TokenReader, TokenizationError,
};
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
    iter: I, // Honestly, immediately having a Vec of tokens might become more efficient than this in the long run.
    /// A clone of the original iterator to peek tokens.
    peeker: I,
    /// The peeked tokens.
    peeked: VecDeque<I::Item>,
    // /// A `Vec` of items starting from the previous checkpoint.
    // checkpoint: Option<VecDeque<I::Item>>,
    // /// Whether the peeker is actively consuming the checkpoint.
    // consuming_checkpoint: bool,
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
            // checkpoint: None,
            // consuming_checkpoint: false,
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

    // /// Sets the current position as a checkpoint.
    // pub fn checkpoint(&mut self) {
    //     self.checkpoint = Some(VecDeque::new());
    // }
    //
    // /// Reverts to the last checkpoint.
    // pub fn revert(&mut self) {
    //     self.consuming_checkpoint = true;
    // }
}

impl<I: Iterator + Clone> Iterator for Peeker<I>
where
    I::Item: Clone,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.peeked.pop_front().or_else(|| self.iter.next())

        // if let Some(checkpoint) = &mut self.checkpoint {
        //     if self.consuming_checkpoint {
        //         if let Some(value) = checkpoint.pop_front() {
        //             return Some(value);
        //         } else {
        //             self.consuming_checkpoint = false;
        //             self.checkpoint = None;
        //         }
        //     }
        // }
        //
        // let value = self.peeked.pop_front().or_else(|| self.iter.next())?;
        // if !self.consuming_checkpoint {
        //     if let Some(checkpoint) = &mut self.checkpoint {
        //         checkpoint.push_back(value.clone());
        //     }
        // }
        //
        // Some(value)
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
    fn map_span<U>(self, f: impl FnOnce(T, Span) -> U) -> Result<U>;
    fn and_then_span<U>(self, f: impl FnOnce(T, Span) -> Result<U>) -> Result<U>;
}

impl<T> ResultExt<T> for Result<Spanned<T>> {
    fn map_span<U>(self, f: impl FnOnce(T, Span) -> U) -> Result<U> {
        self.map(|Spanned(value, span)| f(value, span))
    }

    fn and_then_span<U>(self, f: impl FnOnce(T, Span) -> Result<U>) -> Result<U> {
        self.and_then(|Spanned(value, span)| f(value, span))
    }
}

macro_rules! consume_token {
    (@no_ws $self:expr, $p:pat) => {{
        loop {
            match $self.tokens.peek() {
                Some(Ok(t @ Spanned($p, _))) => {
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
                Some(Ok(Spanned(TokenInfo::Whitespace, _))) => {
                    $self.tokens.next();
                }
                Some(Ok(t @ Spanned($p, _))) => {
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
        while let Some(Ok(Spanned(TokenInfo::Whitespace, _))) = self.tokens.peek() {
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
        consume_token!(self, TokenInfo::IntLiteral(..)).map_span(|info, span| {
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
                .map_span(|info, span| Spanned(
                    Atom::Float(assert_token!(@unsafe info: TokenInfo::FloatLiteral(i) => i)),
                    span,
                ))
        })
        .or_else(|_| {
            consume_token!(self, TokenInfo::Ident(..))
                .map_span(|info, span| Spanned(
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
                .and_then_span(|info, span| Ok(Spanned(
                    Atom::String(assert_token!(
                        @unsafe info: TokenInfo::StringLiteral(s, flags, content_span)
                            => Self::resolve_string(s, flags, content_span)?
                    )),
                    span,
                )))
        })
    }

    /// Parses and consumes the next expression that does not have to be orderly disambiguated
    /// against.
    pub fn consume_unambiguous_expr(&mut self) -> Result<Spanned<Expr>> {
        // Parenthesized expression
        consume_token!(self, TokenInfo::LeftParen)
            .and_then_span(|_, span| {
                let expr = self.consume_expr()?;
                consume_token!(self, TokenInfo::RightParen)
                    .map_err(|_| Error::UnmatchedDelimiter(Delimiter::Paren, span))?;

                Ok(expr)
            })
            // Atom
            .or_else(|_| {
                self.consume_atom()
                    .map_span(|atom, span| Spanned(Expr::Atom(atom), span))
            })
    }

    /// Parses the next attribute access expression.
    pub fn consume_attr_access(&mut self) -> Result<Spanned<Expr>> {
        let original = self.consume_unambiguous_expr()?;

        Ok(std::iter::repeat_with(|| {
            consume_token!(self, TokenInfo::Dot).and_then_span(|_, dot| {
                consume_token!(self, TokenInfo::Ident(_)).map_span(|token, span| (token, span, dot))
            })
        })
        .map_while(Result::ok)
        .fold(original, |current, (attr, span, dot)| {
            let span = current.span().merge(span);

            Spanned(
                Expr::Attr {
                    subject: box current,
                    dot,
                    attr: assert_token!(@unsafe attr: TokenInfo::Ident(i) => i),
                },
                span,
            )
        }))
    }

    /// Parses and consumes the next unary expression.
    pub fn consume_unary(&mut self) -> Result<Spanned<Expr>> {
        #[allow(clippy::needless_collect, reason = "see comment below")]
        #[allow(unused_parens, reason = "parentheses are required as per macro rules")]
        let tokens = std::iter::repeat_with(|| {
            consume_token!(
                self,
                (TokenInfo::Plus | TokenInfo::Minus | TokenInfo::Not | TokenInfo::Tilde)
            )
        })
        .map_while(Result::ok)
        .collect::<Vec<_>>();

        // We must collect then re-iterate over the tokens since otherwise there will be two
        // mutable borrows of self.
        Ok(tokens
            .into_iter()
            .rfold(self.consume_attr_access()?, |expr, op| {
                let (op, op_span) = op.into_inner();
                let op = match op {
                    TokenInfo::Plus => UnaryOp::Plus,
                    TokenInfo::Minus => UnaryOp::Minus,
                    TokenInfo::Not => UnaryOp::Not,
                    TokenInfo::Tilde => UnaryOp::BitNot,
                    // SAFETY: checked when op token was consumed
                    _ => unsafe { std::hint::unreachable_unchecked() },
                };
                let (expr, span) = expr.into_inner();

                Spanned(
                    Expr::UnaryOp {
                        op: Spanned(op, op_span),
                        expr: box Spanned(expr, span),
                    },
                    op_span.merge(span),
                )
            }))
    }

    /// Parses and consumes the next binary power expression.
    pub fn consume_pow(&mut self) -> Result<Spanned<Expr>> {
        #[allow(clippy::needless_collect, reason = "see comment in consume_unary")]
        // This case is a bit special, since this operator is right-associative.
        let tokens = std::iter::repeat_with(|| {
            // TODO: this recovery system is inefficient
            let fallback = self.tokens.clone();

            // To conform to right-associtivity, we must immediately begin consuming tokens
            self.consume_unary()
                .and_then(|expr| {
                    let op_span = consume_token!(self, TokenInfo::Asterisk)?.span();
                    Ok((expr, op_span))
                })
                .and_then(|o| consume_token!(@no_ws self, TokenInfo::Asterisk).map(|_| o))
                // ...however, if there is a problem consuming, reset the token stream to the
                // previous position.
                .map_err(|e| {
                    self.tokens = fallback;
                    e
                })
        })
        .map_while(Result::ok)
        .collect::<Vec<_>>();

        Ok(tokens
            .into_iter()
            .rfold(self.consume_unary()?, |current, (expr, mut op_span)| {
                let (expr, expr_span) = expr.into_inner();
                let (current, current_span) = current.into_inner();
                // This accounts for the second asterisk
                op_span.end += 1;

                Spanned(
                    Expr::BinaryOp {
                        left: box Spanned(expr, expr_span),
                        op: Spanned(BinaryOp::Pow, op_span),
                        right: box Spanned(current, current_span),
                    },
                    current_span.merge(expr_span),
                )
            }))
    }

    /// Parses and consumes the next expression.
    pub fn consume_expr(&mut self) -> Result<Spanned<Expr>> {
        self.consume_pow()
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
    pub fn new(source: &'a str) -> Self {
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
