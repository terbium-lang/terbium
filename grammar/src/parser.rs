use crate::ast::{Atom, BinaryOp, Expr, TypeExpr, UnaryOp};
use crate::error::{Error, TargetKind};
use crate::{Span, Spanned, StringLiteralFlags, Token, TokenInfo};
use chumsky::combinator::{IgnoreThen, OrNot, Repeated, ThenIgnore};
use chumsky::prelude::*;
use chumsky::primitive::Just;

pub type Result<T> = std::result::Result<T, Error>;
pub type RecursiveParser<'a, T> = Recursive<'a, TokenInfo, T, Error>;

type JustToken = Just<TokenInfo, TokenInfo, Error>;
type RepeatedToken = Repeated<JustToken>;
type PadWsTy<T, O> = ThenIgnore<
    IgnoreThen<RepeatedToken, T, Vec<TokenInfo>, O>,
    RepeatedToken,
    O,
    Vec<TokenInfo>,
>;

trait WsPadExt<T, O> {
    fn pad_ws(self) -> PadWsTy<T, O>;
}

impl<O, T: Parser<TokenInfo, O, Error = Error>> WsPadExt<T, O> for T {
    #[inline] // Maybe inline is a bad idea
    fn pad_ws(self) -> PadWsTy<T, O> {
        self.padded_by(just(TokenInfo::Whitespace).repeated())
    }
}

macro_rules! lparen {
    () => {{ just(TokenInfo::LeftParen).pad_ws() }};
}
macro_rules! rparen {
    () => {{ just(TokenInfo::RightParen).pad_ws() }};
}

// Parses a string.
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
                            Span::new(pos - 1, pos + 1 + $length),
                        )
                    })?;

                    pos += $length;
                    char::from_u32(value).ok_or(Error::invalid_hex_escape_sequence(
                        sequence,
                        Span::new(pos - 1, pos + 1 + $length),
                    ))?
                }};
            }

            c = match chars.next().ok_or_else(|| {
                Error::unexpected_eof(pos, Some(('n', "insert an escape sequence")))
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
                        Span::new(pos - 1, pos + 1),
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
        .pad_ws()
        .map_with_span(Spanned);

        ident
    })
}

pub fn expr_parser<'a>() -> RecursiveParser<'a, Spanned<Expr>> {
    let ty = type_expr_parser();

    recursive(|expr: Recursive<TokenInfo, Spanned<Expr>, _>| {
        // An identifier
        let ident = select! {
            TokenInfo::Ident(name) => Expr::Atom(match name.as_str() {
                "true" => Atom::Bool(true),
                "false" => Atom::Bool(false),
                _ => Atom::Ident(name),
            })
        }
        .map_with_span(Spanned);

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
        .or(ident)
        .pad_ws()
        .labelled("atom");

        // Intermediate parser to consume comma-separated sequences, e.g. 1, 2, 3
        let comma_separated = expr
            .clone()
            .separated_by(just(TokenInfo::Comma).pad_ws())
            .allow_trailing();

        // Parses expressions that do not have to be orderly disambiguated against
        let unambiguous = expr
            .delimited_by(lparen!(), rparen!())
            .or(comma_separated
                .clone()
                .delimited_by(lparen!(), rparen!())
                .map_with_span(|exprs, span| Spanned(Expr::Tuple(exprs), span)))
            .labelled("tuple")
            .or(comma_separated
                .delimited_by(just(TokenInfo::LeftBracket).pad_ws(), just(TokenInfo::RightBracket).pad_ws())
                .map_with_span(|exprs, span| Spanned(Expr::Array(exprs), span)))
            .labelled("array")
            .or(atom.clone())
            .labelled("unambiguous expression");

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
            .labelled("attribute access");

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
                let mut partition = args
                    .iter()
                    .position(|(name, _)| name.is_some())
                    .unwrap_or_else(|| args.len());
                let kwargs = args.split_off(partition);

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
                    lhs.span().merge(span),
                ))
            })
            .try_map(|e, span| e)
            .labelled("function call");

        // Prefix unary operators: -a, +a, !a
        let unary = just(TokenInfo::Minus)
            .to(UnaryOp::Minus)
            .or(just(TokenInfo::Plus).to(UnaryOp::Plus))
            .or(just(TokenInfo::Not).to(UnaryOp::Not))
            .pad_ws()
            .map_with_span(Spanned)
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
            .labelled("unary expression");

        // Type cast, e.g. a::b
        let cast = unary
            .clone()
            .then(just(TokenInfo::Colon)
                .ignore_then(just(TokenInfo::Colon))
                .pad_ws()
                .ignore_then(ty.clone())
                .repeated()
            )
            .foldl(|target, ty| {
                let span = target.span().merge(ty.span());

                Spanned(
                    Expr::Cast {
                        expr: Box::new(target),
                        ty: Box::new(ty),
                    },
                    span,
                )
            })
            .labelled("type cast");

        // Power operator: a ** b
        // Note that this is right-associative, so a ** b ** c is a ** (b ** c)
        let pow = cast
            .clone()
            .then(just(TokenInfo::Asterisk)
                .ignore_then(just(TokenInfo::Asterisk))
                .to(BinaryOp::Pow)
                .pad_ws()
                .map_with_span(Spanned)
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
            .labelled("pow");

        fn bin_foldl(lhs: Spanned<Expr>, (op, rhs): (Spanned<BinaryOp>, Spanned<Expr>)) -> Spanned<Expr> {
            let span = lhs.span().merge(rhs.span());
            Spanned(Expr::BinaryOp { left: Box::new(lhs), op, right: Box::new(rhs) }, span)
        }

        // Product operators: a * b, a / b, a % b
        let prod = pow
            .clone()
            .then(
                just(TokenInfo::Asterisk).to(BinaryOp::Mul)
                    .or(just(TokenInfo::Divide).to(BinaryOp::Div))
                    .or(just(TokenInfo::Modulus).to(BinaryOp::Mod))
                    .pad_ws()
                    .map_with_span(Spanned)
                    .then(pow)
                    .repeated()
            )
            .foldl(bin_foldl)
            .labelled("product");

        // Sum operators: a + b, a - b
        let sum = prod
            .clone()
            .then(
                just(TokenInfo::Plus).to(BinaryOp::Add)
                    .or(just(TokenInfo::Minus).to(BinaryOp::Sub))
                    .pad_ws()
                    .map_with_span(Spanned)
                    .then(prod)
                    .repeated()
            )
            .foldl(bin_foldl)
            .labelled("sum");

        macro_rules! compound {
            ($ident1:ident $ident2:ident => $to:expr) => {{
                just(TokenInfo::$ident).ignore_then(just(TokenInfo::$ident)).to($expr)
            }}
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
                    .pad_ws()
                    .map_with_span(Spanned)
                    .then(sum)
                    .repeated()
            )
            .foldl(bin_foldl)
            .labelled("comparison");

        1
    })
}
