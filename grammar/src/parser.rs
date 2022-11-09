use super::{
    ast::{Atom, BinaryOp, Delimiter, Expr, Node, Spanned, UnaryOp},
    Span, StringLiteralFlags, Token, TokenInfo, TokenReader, TokenizationError,
};
use std::result::Result as StdResult;

type Result<T> = StdResult<T, Error>;

/// Represents an error that occured during parsing.
#[derive(Clone, Debug)]
pub enum Error {
    /// An error occured during tokenization.
    Tokenization(TokenizationError),
    /// An unexpected token was encountered.
    UnexpectedToken { expected: String, found: Token },
    /// Encountered the end of the input.
    UnexpectedEof,
    /// An unmatched closing delimiter was encountered.
    /// The span is the span of the starting delimiter.
    UnmatchedDelimiter(Delimiter, Span),
    /// Unknown escape sequence encountered.
    UnknownEscapeSequence(char, Span),
    /// Invalid hex escape sequence encountered.
    InvalidHexEscapeSequence(String, Span),
}

/// A cursor over a collection of tokens.
#[derive(Clone, Default)]
pub struct TokenCursor {
    tokens: Vec<Token>,
    pos: usize,
}

impl TokenCursor {
    fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            ..Default::default()
        }
    }

    /// Peeks at the next token.
    #[must_use]
    pub fn peek(&self) -> Option<Token> {
        self.tokens.get(self.pos).cloned()
    }
}

impl Iterator for TokenCursor {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        let token = self.tokens.get(self.pos)?;
        self.pos += 1;
        Some(token.clone())
    }
}

/// Parses a token stream into an AST.
#[derive(Clone)]
pub struct Parser {
    tokens: TokenCursor,
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
    (@no_ws $self:ident, $p:pat, $expecting:expr) => {{
        match $self.tokens.peek() {
            Some(t @ Spanned($p, _)) => {
                $self.tokens.next();
                Ok(t)
            }
            Some(other) => Err(Error::UnexpectedToken {
                expected: $expecting.to_string(),
                found: other,
            }),
            None => Err(Error::UnexpectedEof),
        }
    }};
    ($self:ident, $p:pat, $expecting:expr) => {{
        loop {
            match $self.tokens.peek() {
                Some(Spanned(TokenInfo::Whitespace, _)) => {
                    $self.tokens.next();
                }
                Some(t @ Spanned($p, _)) => {
                    $self.tokens.next();
                    break Ok(t);
                }
                Some(other) => {
                    break Err(Error::UnexpectedToken {
                        expected: $expecting.to_string(),
                        found: other,
                    })
                }
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

macro_rules! expr_infix_impl {
    (
        $self:ident,
        $consume_expr:expr,
        $consume_op:expr,
        $op_ident_name:ident => $transform_op:expr
    ) => {{
        let original = $consume_expr?;

        #[allow(unused_parens, reason = "parentheses are required as per macro rules")]
        Ok(
            std::iter::repeat_with(||
                {
                    let fallback = $self.tokens.pos;
                    $consume_op.and_then(|op| Ok(
                        (
                            $self.consume_unary()
                                .or_else(|_| $self.consume_logical_not())
                                .or_else(|_| $consume_expr)?,
                            op,
                        ),
                    ))
                    .map_err(|e| {
                        $self.tokens.pos = fallback;
                        e
                    })
                }
            )
            .map_while(Result::ok)
            .fold(original, |current, (expr, $op_ident_name)| {
                let span = current.span().merge(expr.span());
                let (op, op_span) = $transform_op;

                Spanned(
                    Expr::BinaryOp {
                        left: box current,
                        op: Spanned(op, op_span),
                        right: box expr,
                    },
                    span,
                )
            }),
        )
    }};
    (
        $self:ident,
        $consume_expr:expr,
        @single $consume_op:pat,
        $expecting:expr,
        $op_ident_name:ident => $transform_op:expr
    ) => {
        expr_infix_impl!(
            $self,
            $consume_expr,
            consume_token!($self, $consume_op, $expecting),
            op => {
                let ($op_ident_name, op_span) = op.into_inner();
                ($transform_op, op_span)
            }
        )
    }
}

macro_rules! fallback {
    ($self:ident, $consume:expr) => {{
        let fallback = $self.tokens.pos;
        $consume.map_err(|e| {
            $self.tokens.pos = fallback;
            e
        })
    }};
}

impl Parser {
    /// Skips all whitespace tokens.
    pub fn skip_ws(&mut self) {
        while let Some(Spanned(TokenInfo::Whitespace, _)) = self.tokens.peek() {
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

    /// Parses expressions separated by the given separator.
    pub fn separated_by<T>(
        &mut self,
        mut consumer: impl FnMut(&mut Self) -> Result<T>,
        separator: &TokenInfo,
    ) -> Result<(Vec<T>, bool)> {
        let mut result = Vec::new();
        let mut is_empty = true;

        loop {
            if let Ok(res) = fallback!(self, consumer(self)) {
                result.push(res);
            } else {
                return Ok((result, is_empty));
            }

            self.skip_ws();
            match self.tokens.peek() {
                Some(token) if &token.0 == separator => {
                    is_empty = false;
                    self.tokens.next();
                }
                _ => return Ok((result, false)),
            }
        }
    }

    /// Parses expressions delimited by the given delimiter. Fallbacks for this must be handled
    /// by the caller.
    fn delimited_by<T>(
        &mut self,
        consumer: impl FnOnce(&mut Self) -> Result<T>,
        delimiter: Delimiter,
    ) -> Result<(Span, T, Span)> {
        self.skip_ws();
        let open_span = match self.tokens.peek() {
            Some(Spanned(token, span)) if token == delimiter.open_token() => {
                self.tokens.next();
                span
            }
            Some(found) => {
                return Err(Error::UnexpectedToken {
                    expected: format!("opening delimiter `{}`", delimiter.open()),
                    found,
                })
            }
            None => return Err(Error::UnexpectedEof),
        };

        self.skip_ws();
        let result = consumer(self)?;

        self.skip_ws();
        let close_span = match self.tokens.peek() {
            Some(Spanned(token, span)) if token == delimiter.close_token() => {
                self.tokens.next();
                span
            }
            Some(found) => {
                return Err(Error::UnexpectedToken {
                    expected: format!("closing delimiter `{}`", delimiter.close()),
                    found,
                })
            }
            None => return Err(Error::UnmatchedDelimiter(delimiter, open_span)),
        };

        Ok((open_span, result, close_span))
    }

    /// Parses and consumes the next atom. An atom is the most basic unit of an expression that
    /// cannot be broken down into other expressions any further.
    ///
    /// For example, 1 is an atom, as is "hello" - but 1 + 1 is not, since that can be further
    /// broken down into two expressions.
    pub fn consume_atom(&mut self) -> Result<Spanned<Atom>> {
        consume_token!(self, TokenInfo::IntLiteral(..), "integer literal").map_span(|info, span| {
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
            consume_token!(self, TokenInfo::FloatLiteral(_), "float literal")
                .map_span(|info, span| Spanned(
                    Atom::Float(assert_token!(@unsafe info: TokenInfo::FloatLiteral(i) => i)),
                    span,
                ))
        })
        .or_else(|_| {
            consume_token!(self, TokenInfo::Ident(..), "identifier")
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
            consume_token!(self, TokenInfo::StringLiteral(..), "string literal")
                .and_then_span(|info, span| Ok(Spanned(
                    Atom::String(assert_token!(
                        @unsafe info: TokenInfo::StringLiteral(s, flags, content_span)
                            => Self::resolve_string(s, flags, content_span)?
                    )),
                    span,
                )))
        })
    }

    /// Parses the next comma-separated delimited expression.
    pub fn consume_comma_separated_delimited(
        &mut self,
        delimiter: Delimiter,
        wrapper: impl FnOnce(Vec<Spanned<Expr>>) -> Expr,
    ) -> Result<Spanned<Expr>> {
        let res = self.delimited_by(
            |parser| {
                parser
                    .separated_by(Self::consume_expr, &TokenInfo::Comma)
                    .map(|res| res.0)
            },
            delimiter,
        );
        let (open_span, elements, close_span) = fallback!(self, res)?;
        Ok(Spanned(wrapper(elements), open_span.merge(close_span)))
    }

    /// Parses and consumes the next expression that does not have to be orderly disambiguated
    /// against.
    pub fn consume_unambiguous_expr(&mut self) -> Result<Spanned<Expr>> {
        // Parenthesized expression
        fallback!(
            self,
            self.delimited_by(Self::consume_expr, Delimiter::Paren)
        )
        .map(|(_, expr, _)| expr)
        // Tuple
        .or_else(|_| self.consume_comma_separated_delimited(Delimiter::Paren, Expr::Tuple))
        // Array
        .or_else(|_| self.consume_comma_separated_delimited(Delimiter::Bracket, Expr::Array))
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
            consume_token!(self, TokenInfo::Dot, ".").and_then_span(|_, dot| {
                consume_token!(self, TokenInfo::Ident(_), "identifier")
                    .map_span(|token, span| (token, span, dot))
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

    /// Parses and consumes the next binary power expression.
    pub fn consume_pow(&mut self) -> Result<Spanned<Expr>> {
        #[allow(clippy::needless_collect, reason = "see comment in consume_unary")]
        // This case is a bit special, since this operator is right-associative.
        let tokens = std::iter::repeat_with(|| {
            let fallback = self.tokens.pos;

            // To conform to right-associtivity, we must immediately begin consuming tokens
            self.consume_attr_access()
                .and_then(|expr| {
                    let op_span = consume_token!(self, TokenInfo::Asterisk, "*")?.span();
                    Ok((expr, op_span))
                })
                .and_then(|o| consume_token!(@no_ws self, TokenInfo::Asterisk, "**").map(|_| o))
                // ...however, if there is a problem consuming, reset the token stream to the
                // previous position.
                .map_err(|e| {
                    self.tokens.pos = fallback;
                    e
                })
        })
        .map_while(Result::ok)
        .collect::<Vec<_>>();

        Ok(tokens.into_iter().rfold(
            self.consume_attr_access()?,
            |current, (expr, mut op_span)| {
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
            },
        ))
    }

    /// Parses and consumes the next unary expression. This does not include the logical not
    /// operator, that comes lower in precedence.
    pub fn consume_unary(&mut self) -> Result<Spanned<Expr>> {
        #[allow(clippy::needless_collect, reason = "see comment below")]
        #[allow(unused_parens, reason = "parentheses are required as per macro rules")]
        let tokens = std::iter::repeat_with(|| {
            consume_token!(
                self,
                (TokenInfo::Plus | TokenInfo::Minus | TokenInfo::Tilde),
                "unary operator (+, -, ~)"
            )
        })
        .map_while(Result::ok)
        .collect::<Vec<_>>();

        // We must collect then re-iterate over the tokens since otherwise there will be two
        // mutable borrows of self.
        Ok(tokens.into_iter().rfold(self.consume_pow()?, |expr, op| {
            let (op, op_span) = op.into_inner();
            let op = match op {
                TokenInfo::Plus => UnaryOp::Plus,
                TokenInfo::Minus => UnaryOp::Minus,
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

    /// Parses and consumes the next binary multiplication, division, or modulus expression.
    pub fn consume_product(&mut self) -> Result<Spanned<Expr>> {
        expr_infix_impl!(
            self,
            self.consume_unary(),
            @single (TokenInfo::Asterisk | TokenInfo::Divide | TokenInfo::Modulus),
            "multiplication, division, or modulus operator",
            op => match op {
                TokenInfo::Asterisk => BinaryOp::Mul,
                TokenInfo::Divide => BinaryOp::Div,
                TokenInfo::Modulus => BinaryOp::Mod,
                // SAFETY: checked when op token was consumed
                _ => unsafe { std::hint::unreachable_unchecked() },
            }
        )
    }

    /// Parses and consumes the next binary addition or subtraction expression.
    pub fn consume_sum(&mut self) -> Result<Spanned<Expr>> {
        expr_infix_impl!(
            self,
            self.consume_product(),
            @single (TokenInfo::Plus | TokenInfo::Minus),
            "addition or subtraction operator",
            op => match op {
                TokenInfo::Plus => BinaryOp::Add,
                TokenInfo::Minus => BinaryOp::Sub,
                // SAFETY: checked when op token was consumed
                _ => unsafe { std::hint::unreachable_unchecked() },
            }
        )
    }

    /// Parses and consumes the next bitwise shift expression.
    pub fn consume_bitwise_shift(&mut self) -> Result<Spanned<Expr>> {
        expr_infix_impl!(
            self,
            self.consume_sum(),
            consume_token!(self, TokenInfo::Lt, "<")
                .and_then(|_| consume_token!(@no_ws self, TokenInfo::Lt, "<<"))
                .map_span(|_, span| (BinaryOp::Shl, span))
                .or_else(|_| {
                    consume_token!(self, TokenInfo::Gt, ">")
                        .and_then(|_| consume_token!(@no_ws self, TokenInfo::Gt, ">>"))
                        .map_span(|_, span| (BinaryOp::Shr, span))
                })
                .map(|(op, span)| (op, span.extend_back())),
            op => op
        )
    }

    /// Parses and consumes the next bitwise AND expression.
    pub fn consume_bitwise_and(&mut self) -> Result<Spanned<Expr>> {
        expr_infix_impl!(
            self,
            self.consume_bitwise_shift(),
            @single TokenInfo::And,
            "bitwise and operator",
            _op => BinaryOp::BitAnd
        )
    }

    /// Parses and consumes the next bitwise XOR expression.
    pub fn consume_bitwise_xor(&mut self) -> Result<Spanned<Expr>> {
        expr_infix_impl!(
            self,
            self.consume_bitwise_and(),
            @single TokenInfo::Caret,
            "bitwise xor operator",
            _op => BinaryOp::BitXor
        )
    }

    /// Parses and consumes the next bitwise OR expression.
    pub fn consume_bitwise_or(&mut self) -> Result<Spanned<Expr>> {
        expr_infix_impl!(
            self,
            self.consume_bitwise_xor(),
            @single TokenInfo::Or,
            "bitwise or operator",
            _op => BinaryOp::BitOr
        )
    }

    /// Parses and consumes the next logical comparison expression.
    pub fn consume_logical_comparison(&mut self) -> Result<Spanned<Expr>> {
        expr_infix_impl!(
            self,
            self.consume_bitwise_or(),
            consume_token!(self, TokenInfo::Equals, "=")
                .and_then(|_| consume_token!(@no_ws self, TokenInfo::Equals, "=="))
                .map_span(|_, span| (BinaryOp::Eq, span.extend_back()))
                .or_else(|_| {
                    consume_token!(self, TokenInfo::Not, "!")
                        .and_then(|_| consume_token!(@no_ws self, TokenInfo::Equals, "!="))
                        .map_span(|_, span| (BinaryOp::Ne, span.extend_back()))
                })
                .or_else(|_| {
                    consume_token!(self, TokenInfo::Lt, "<")
                        .and_then_span(|_, lt_span| {
                            consume_token!(@no_ws self, TokenInfo::Equals, "<=")
                                .map_span(|_, eq_span| (BinaryOp::Le, eq_span.extend_back()))
                                .or_else(|_| Ok((BinaryOp::Lt, lt_span)))
                        })
                })
                .or_else(|_| {
                    consume_token!(self, TokenInfo::Gt, ">")
                        .and_then_span(|_, gt_span| {
                            consume_token!(@no_ws self, TokenInfo::Equals, ">=")
                                .map_span(|_, eq_span| (BinaryOp::Ge, eq_span.extend_back()))
                                .or_else(|_| Ok((BinaryOp::Gt, gt_span)))
                        })
                }),
            op => op
        )
    }

    /// Parses and consumes the next logical not expression.
    pub fn consume_logical_not(&mut self) -> Result<Spanned<Expr>> {
        #[allow(clippy::needless_collect, reason = "see comment below")]
        #[allow(unused_parens, reason = "parentheses are required as per macro rules")]
        let tokens = std::iter::repeat_with(|| consume_token!(self, TokenInfo::Not, "!"))
            .map_while(Result::ok)
            .collect::<Vec<_>>();

        // We must collect then re-iterate over the tokens since otherwise there will be two
        // mutable borrows of self.
        Ok(tokens
            .into_iter()
            .rfold(self.consume_logical_comparison()?, |expr, op| {
                let op_span = op.span();
                let (expr, span) = expr.into_inner();

                Spanned(
                    Expr::UnaryOp {
                        op: Spanned(UnaryOp::Not, op_span),
                        expr: box Spanned(expr, span),
                    },
                    op_span.merge(span),
                )
            }))
    }

    /// Parses and consumes the next logical and expression.
    pub fn consume_logical_and(&mut self) -> Result<Spanned<Expr>> {
        expr_infix_impl!(
            self,
            self.consume_logical_not(),
            consume_token!(self, TokenInfo::And, "&")
                .and_then(|_| consume_token!(@no_ws self, TokenInfo::And, "&&"))
                .map_span(|_, span| (BinaryOp::LogicalAnd, span.extend_back())),
            op => op
        )
    }

    /// Parses and consumes the next logical or expression.
    pub fn consume_logical_or(&mut self) -> Result<Spanned<Expr>> {
        expr_infix_impl!(
            self,
            self.consume_logical_and(),
            consume_token!(self, TokenInfo::Or, "|")
                .and_then(|_| consume_token!(@no_ws self, TokenInfo::Or, "||"))
                .map_span(|_, span| (BinaryOp::LogicalOr, span.extend_back())),
            op => op
        )
    }

    /// Parses and consumes the next expression.
    pub fn consume_expr(&mut self) -> Result<Spanned<Expr>> {
        self.consume_logical_or()
    }

    /// Parses and consumes the next node.
    pub fn consume_node(&mut self) -> Result<Spanned<Node>> {
        self.skip_ws();
        todo!()
    }
}

impl Parser {
    /// Creates a new parser from a string slice.
    pub fn new(source: &str) -> StdResult<Self, TokenizationError> {
        Ok(Self {
            tokens: TokenCursor::new(TokenReader::new(source).collect::<StdResult<Vec<_>, _>>()?),
        })
    }
}
