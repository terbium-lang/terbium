use crate::ast::Node;
use crate::error::TargetKind;
use crate::{
    ast::{Atom, BinaryOp, Expr, TypeExpr, UnaryOp},
    error::Error,
    span::{Provider, Span, Spanned},
    token::{ChumskyTokenStreamer, StringLiteralFlags, TokenInfo, TokenReader},
};
use chumsky::{
    combinator::{IgnoreThen, Repeated, ThenIgnore},
    error::Error as _,
    prelude::{
        choice, end, filter_map, just, recursive, select, Parser as ChumskyParser, Recursive,
    },
    primitive::Just,
    stream::Stream,
};
use std::result::Result as StdResult;

pub type Result<T> = StdResult<T, Error>;
pub type RecursiveParser<'a, T> = Recursive<'a, TokenInfo, T, Error>;

type JustToken = Just<TokenInfo, TokenInfo, Error>;
type RepeatedToken = Repeated<JustToken>;
type PadWsTy<T, O> =
    ThenIgnore<IgnoreThen<RepeatedToken, T, Vec<TokenInfo>, O>, RepeatedToken, O, Vec<TokenInfo>>;

trait WsPadExt<T, O> {
    fn pad_ws(self) -> PadWsTy<T, O>;
}

impl<O, T: ChumskyParser<TokenInfo, O, Error = Error>> WsPadExt<T, O> for T {
    #[inline] // Maybe inline is a bad idea
    fn pad_ws(self) -> PadWsTy<T, O> {
        self.padded_by(just(TokenInfo::Whitespace).repeated())
    }
}

macro_rules! lparen {
    () => {{
        just(TokenInfo::LeftParen).pad_ws()
    }};
}
macro_rules! rparen {
    () => {{
        just(TokenInfo::RightParen).pad_ws()
    }};
}

macro_rules! kw {
    ($kw:literal) => {{
        just(TokenInfo::Ident($kw.to_string())).map_with_span(Spanned)
    }};
    ($kw:ident) => {{
        kw!(stringify!($kw))
    }};
    (@pad $kw:tt) => {{
        kw!($kw).pad_ws()
    }};
}

/// Resolves escape sequences in a string.
fn resolve_string(content: String, flags: StringLiteralFlags, span: Span) -> Result<String> {
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
                        Error::invalid_hex_escape_sequence(
                            sequence.clone(),
                            span.get_span(pos - 1, pos + 1 + $length),
                        )
                    })?;

                    pos += $length;
                    char::from_u32(value).ok_or(Error::invalid_hex_escape_sequence(
                        sequence,
                        span.get_span(pos - 1, pos + 1 + $length),
                    ))?
                }};
            }

            c = match chars.next().ok_or_else(|| {
                Error::unexpected_eof(
                    Span::single(span.src, pos),
                    Some(('n', "insert an escape sequence")),
                )
            })? {
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
                c => {
                    return Err(Error::unknown_escape_sequence(
                        c,
                        span.get_span(pos - 1, pos + 1),
                    ))
                }
            };
        }

        result.push(c);
        pos += 1;
    }

    Ok(result)
}

pub fn type_expr_parser<'a>() -> RecursiveParser<'a, Spanned<TypeExpr>> {
    recursive(|ty| {
        let ident = select! {
            TokenInfo::Ident(name) => TypeExpr::Ident(name),
        }
        .map_with_span(Spanned)
        .pad_ws();

        ident
    })
}

pub fn body_parser<'a>() -> RecursiveParser<'a, Vec<Spanned<Node>>> {
    recursive(|body: Recursive<TokenInfo, Vec<Spanned<Node>>, _>| {
        let expr = expr_parser(body.clone());
        let ty = type_expr_parser();

        let ident = select! {
            TokenInfo::Ident(name) => name,
        }
        .map_with_span(Spanned)
        .pad_ws();

        let expression = expr
            .clone()
            .then_ignore(just(TokenInfo::Semicolon).pad_ws())
            .map(Node::Expr)
            .map_with_span(Spanned)
            .labelled("expression");

        // Declaration
        let decl = kw!("let")
            .map(|spanned| spanned.map(|_| false))
            .or(kw!("const").map(|spanned| spanned.map(|_| true)))
            .then(kw!("mut").map(|spanned| spanned.span()).pad_ws().or_not())
            .then(ident)
            .then(
                just(TokenInfo::Colon)
                    .pad_ws()
                    .ignore_then(ty.clone())
                    .or_not(),
            )
            .then_ignore(just(TokenInfo::Equals))
            .then(expr.clone().or_not())
            .then_ignore(just(TokenInfo::Semicolon))
            .try_map(
                |((((Spanned(is_const, kw_span), mut_kw), ident), ty), expr), span| {
                    if is_const {
                        if let Some(mut_kw) = mut_kw {
                            let keyword = TargetKind::Keyword("mut");
                            return Err(Error::unexpected(mut_kw, None::<TargetKind>, &keyword)
                                .note("constants cannot be mutable"));
                        }
                        if expr.is_none() {
                            return Err(Error::expected_input_found(
                                span,
                                std::iter::once(Some(TargetKind::Expression)),
                                None,
                            ));
                        }
                    }
                    Ok(Spanned(
                        Node::Decl {
                            kw: kw_span,
                            mut_kw,
                            ident,
                            ty,
                            value: expr.map(Box::new),
                            is_const,
                        },
                        span,
                    ))
                },
            );

        choice((decl, expression)).pad_ws().repeated()
    })
}

#[allow(clippy::too_many_lines)]
pub fn expr_parser<'a>(
    body: Recursive<TokenInfo, Vec<Spanned<Node>>, Error>,
) -> RecursiveParser<'a, Spanned<Expr>> {
    let ty = type_expr_parser();

    recursive(|expr: Recursive<TokenInfo, Spanned<Expr>, _>| {
        fn bin_foldl(
            lhs: Spanned<Expr>,
            (op, rhs): (Spanned<BinaryOp>, Spanned<Expr>),
        ) -> Spanned<Expr> {
            let span = lhs.span().merge(rhs.span());
            Spanned(
                Expr::BinaryOp {
                    left: Box::new(lhs),
                    op,
                    right: Box::new(rhs),
                },
                span,
            )
        }

        // An identifier
        let ident = select! {
            TokenInfo::Ident(name) => Expr::Atom(match name.as_str() {
                "true" => Atom::Bool(true),
                "false" => Atom::Bool(false),
                _ => Atom::Ident(name),
            })
        }
        .map_with_span(Spanned)
        .pad_ws();

        // Parses and consumes the next atom. An atom is the most basic unit of an expression that
        // cannot be broken down into other expressions any further.
        //
        // For example, 1 is an atom, as is "hello" - but 1 + 1 is not, since that can be further
        // broken down into two expressions.
        let atom = filter_map(|span, token| {
            Ok(Spanned(
                Expr::Atom(match token {
                    TokenInfo::IntLiteral(val, radix) => Atom::Int(val, radix),
                    TokenInfo::FloatLiteral(val) => Atom::Float(val),
                    TokenInfo::StringLiteral(content, flags, inner_span) => {
                        Atom::String(resolve_string(content, flags, inner_span)?)
                    }
                    _ => return Err(Error::expected_input_found(span, None, Some(token))),
                }),
                span,
            ))
        })
        .pad_ws()
        .or(ident.clone())
        .labelled("atom");

        // Intermediate parser to consume comma-separated sequences, e.g. 1, 2, 3
        let comma_separated = expr
            .clone()
            .separated_by(just(TokenInfo::Comma).pad_ws())
            .allow_trailing();

        // Parses expressions that do not have to be orderly disambiguated against
        let unambiguous = choice((
            expr.clone().delimited_by(lparen!(), rparen!()),
            comma_separated
                .clone()
                .delimited_by(lparen!(), rparen!())
                .map_with_span(|exprs, span| Spanned(Expr::Tuple(exprs), span))
                .labelled("tuple"),
            comma_separated
                .delimited_by(
                    just(TokenInfo::LeftBracket).pad_ws(),
                    just(TokenInfo::RightBracket).pad_ws(),
                )
                .map_with_span(|exprs, span| Spanned(Expr::Array(exprs), span))
                .labelled("array"),
            atom.clone(),
        ))
        .labelled("unambiguous expression")
        .boxed();

        // Attribute access: a.b.c
        let attr = unambiguous
            .clone()
            .then(
                just(TokenInfo::Dot)
                    .map_with_span(|_, span: Span| span)
                    .then(ident)
                    .repeated(),
            )
            .foldl(|expr, (dot, ident)| {
                let span = expr.span();

                Spanned(
                    Expr::Attr {
                        subject: Box::new(expr),
                        dot,
                        attr: ident.to_string(),
                    },
                    span.merge(ident.span()),
                )
            })
            .labelled("attribute access")
            .boxed();

        // Function call: a(b, c)
        let call = attr
            .clone()
            .map(Ok)
            .then(
                select!(TokenInfo::Ident(ident) => ident)
                    .then_ignore(just(TokenInfo::Colon))
                    .or_not()
                    .pad_ws()
                    .then(expr.clone())
                    .separated_by(just(TokenInfo::Comma).pad_ws())
                    .allow_trailing()
                    .delimited_by(lparen!(), rparen!())
                    .map_with_span(Spanned)
                    .repeated(),
            )
            .foldl(|lhs, Spanned(mut args, span)| {
                let lhs = lhs?;
                let partition = args
                    .iter()
                    .position(|(name, _)| name.is_some())
                    .unwrap_or(args.len());
                let kwargs = args.split_off(partition);
                let span = lhs.span().merge(span);

                Ok(Spanned(
                    Expr::Call {
                        func: Box::new(lhs),
                        args: args.into_iter().map(|(_, arg)| arg).collect(),
                        kwargs: kwargs
                            .into_iter()
                            .map(|(name, arg)| {
                                Ok((
                                    name.ok_or_else(|| {
                                        Error::unexpected_positional_argument(arg.span())
                                    })?,
                                    arg,
                                ))
                            })
                            .collect::<Result<Vec<_>>>()?,
                    },
                    span,
                ))
            })
            .try_map(|e, _span| e)
            .labelled("function call")
            .boxed();

        // Prefix unary operators: -a, +a, !a
        let unary = just(TokenInfo::Minus)
            .to(UnaryOp::Minus)
            .or(just(TokenInfo::Plus).to(UnaryOp::Plus))
            .or(just(TokenInfo::Not).to(UnaryOp::Not))
            .map_with_span(Spanned)
            .pad_ws()
            .repeated()
            .then(call.clone())
            .foldr(|op, expr| {
                let span = op.span().merge(expr.span());

                Spanned(
                    Expr::UnaryOp {
                        op,
                        expr: Box::new(expr),
                    },
                    span,
                )
            })
            .labelled("unary expression")
            .boxed();

        // Type cast, e.g. a::b
        let cast = unary
            .clone()
            .then(
                just(TokenInfo::Colon)
                    .ignore_then(just(TokenInfo::Colon))
                    .pad_ws()
                    .ignore_then(ty.clone())
                    .repeated(),
            )
            .foldl(|target, ty| {
                let span = target.span().merge(ty.span());

                Spanned(
                    Expr::Cast {
                        expr: Box::new(target),
                        ty,
                    },
                    span,
                )
            })
            .labelled("type cast")
            .boxed();

        // Power operator: a ** b
        // Note that this is right-associative, so a ** b ** c is a ** (b ** c)
        let pow = cast
            .clone()
            .then(
                just(TokenInfo::Asterisk)
                    .ignore_then(just(TokenInfo::Asterisk))
                    .to(BinaryOp::Pow)
                    .map_with_span(Spanned)
                    .pad_ws(),
            )
            .repeated()
            .then(cast)
            .foldr(|(lhs, op), rhs| {
                let span = lhs.span().merge(rhs.span());

                Spanned(
                    Expr::BinaryOp {
                        left: Box::new(lhs),
                        op,
                        right: Box::new(rhs),
                    },
                    span,
                )
            })
            .labelled("pow")
            .boxed();

        // Product operators: a * b, a / b, a % b
        let prod = pow
            .clone()
            .then(
                just(TokenInfo::Asterisk)
                    .to(BinaryOp::Mul)
                    .or(just(TokenInfo::Divide).to(BinaryOp::Div))
                    .or(just(TokenInfo::Modulus).to(BinaryOp::Mod))
                    .map_with_span(Spanned)
                    .pad_ws()
                    .then(pow)
                    .repeated(),
            )
            .foldl(bin_foldl)
            .labelled("product")
            .boxed();

        // Sum operators: a + b, a - b
        let sum = prod
            .clone()
            .then(
                just(TokenInfo::Plus)
                    .to(BinaryOp::Add)
                    .or(just(TokenInfo::Minus).to(BinaryOp::Sub))
                    .pad_ws()
                    .map_with_span(Spanned)
                    .then(prod)
                    .repeated(),
            )
            .foldl(bin_foldl)
            .labelled("sum")
            .boxed();

        macro_rules! compound {
            ($ident1:ident $ident2:ident => $to:expr) => {{
                just(TokenInfo::$ident1)
                    .ignore_then(just(TokenInfo::$ident2))
                    .to($to)
            }};
        }

        // Comparison operators: a == b, a != b, a < b, a > b, a <= b, a >= b
        let cmp = sum
            .clone()
            .then(
                compound!(Equals Equals => BinaryOp::Eq)
                    .or(compound!(Not Equals => BinaryOp::Ne))
                    .or(compound!(Lt Equals => BinaryOp::Le))
                    .or(compound!(Gt Equals => BinaryOp::Ge))
                    .or(just(TokenInfo::Lt).to(BinaryOp::Lt))
                    .or(just(TokenInfo::Gt).to(BinaryOp::Gt))
                    .map_with_span(Spanned)
                    .pad_ws()
                    .then(sum)
                    .repeated(),
            )
            .foldl(bin_foldl)
            .labelled("comparison")
            .boxed();

        // Logical AND: a && b
        let logical_and = cmp
            .clone()
            .then(
                compound!(And And => BinaryOp::LogicalAnd)
                    .map_with_span(Spanned)
                    .pad_ws()
                    .then(cmp)
                    .repeated(),
            )
            .foldl(bin_foldl)
            .labelled("logical and")
            .boxed();

        // Logical OR: a || b
        let logical_or = logical_and
            .clone()
            .then(
                compound!(Or Or => BinaryOp::LogicalOr)
                    .map_with_span(Spanned)
                    .pad_ws()
                    .then(logical_and)
                    .repeated(),
            )
            .foldl(bin_foldl)
            .labelled("logical or")
            .boxed();

        // Bitwise operators: a & b, a | b, a ^ b
        let bitwise = logical_or
            .clone()
            .then(
                just(TokenInfo::And)
                    .to(BinaryOp::BitAnd)
                    .or(just(TokenInfo::Or).to(BinaryOp::BitOr))
                    .or(just(TokenInfo::Caret).to(BinaryOp::BitXor))
                    .map_with_span(Spanned)
                    .pad_ws()
                    .then(logical_or)
                    .repeated(),
            )
            .foldl(bin_foldl)
            .labelled("bitwise")
            .boxed();

        bitwise
    })
}

/// Parses a token stream into an AST.
#[must_use = "parser will only parse if you call its provided methods"]
pub struct Parser<'a> {
    tokens: ChumskyTokenStreamer<'a>,
    eof: Span,
}

impl<'a> Parser<'a> {
    /// Creates a new parser over the provided source provider.
    pub fn from_provider(provider: &'a Provider<'a>) -> Self {
        Self {
            tokens: ChumskyTokenStreamer(TokenReader::new(provider)),
            eof: provider.eof(),
        }
    }

    #[inline]
    fn stream(&mut self) -> Stream<TokenInfo, Span, &mut ChumskyTokenStreamer<'a>> {
        Stream::from_iter(self.eof, &mut self.tokens)
    }

    /// Consumes the next expression in the token stream.
    pub fn next_expr(&mut self) -> StdResult<Spanned<Expr>, Vec<Error>> {
        let body_parser = body_parser();
        expr_parser(body_parser).parse(self.stream())
    }

    /// Consumes the entire token tree as an expression.
    pub fn consume_expr_until_end(&mut self) -> StdResult<Spanned<Expr>, Vec<Error>> {
        let body_parser = body_parser();
        expr_parser(body_parser)
            .then_ignore(end())
            .parse(self.stream())
    }

    /// Consumes the entire token tree as a body.
    pub fn consume_body_until_end(&mut self) -> StdResult<Vec<Spanned<Node>>, Vec<Error>> {
        body_parser().then_ignore(end()).parse(self.stream())
    }
}
