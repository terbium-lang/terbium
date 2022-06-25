use super::token::{get_lexer, Bracket, Keyword, Literal, Operator, StringLiteral, Token};
use super::{Error, Source, Span, Spanned};

use std::path::Path;

use chumsky::prelude::*;
use chumsky::Stream;

pub type SpannedExpr = Spanned<Expr>;
pub type SpannedOperator = Spanned<Operator>;
pub type SpannedNode = Spanned<Node>;
pub type SpannedBody = Spanned<Body>;
pub type SpannedTarget = Spanned<Target>;
pub type SpannedParam = Spanned<Param>;
pub type SpannedTypeExpr = Spanned<TypeExpr>;

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Integer(u128),
    Float(String), // See token.rs for why this is a String
    String(String),
    Bool(bool),
    Ident(String),
    Array(Vec<SpannedExpr>),
    Cast(SpannedExpr, SpannedExpr),
    UnaryExpr {
        operator: SpannedOperator,
        value: SpannedExpr,
    },
    BinaryExpr {
        operator: SpannedOperator,
        lhs: SpannedExpr,
        rhs: SpannedExpr,
    },
    Attr(SpannedExpr, String),
    Call {
        value: SpannedExpr,
        args: Vec<SpannedExpr>,
        kwargs: Vec<(String, SpannedExpr)>,
    },
    If {
        condition: SpannedExpr,
        body: SpannedBody,
        else_if_bodies: Vec<(SpannedExpr, SpannedBody)>,
        else_body: Option<SpannedBody>,
    },
    While {
        condition: SpannedExpr,
        body: Vec<SpannedNode>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum TypeExpr {
    /// Given T, this becomes Ident("T")
    Ident(String),
    /// Given mod.Type, this becomes Attr(Ident(mod), "type")
    Attr(SpannedTypeExpr, String),
    /// Given Type<A, B>, this becomes Generic(Ident(Type), [Ident(A), Ident(B)])
    Generic(SpannedTypeExpr, Vec<SpannedTypeExpr>),
    /// Given A | B, this becomes Union(Ident(A), Ident(B))
    ///
    /// Terbium handles Unions wide-to-narrow. This means given type A | B,
    /// something compatible with either type A or B will be compatible with it,
    /// but only fields/operations that exist on **both** A and B will exist on the type.
    Union(SpannedTypeExpr, SpannedTypeExpr),
    /// Given A & B, this becomes And(Ident(A), Ident(B))
    ///
    /// Terbium handles And narrow-to-wide. This means given type A & B,
    /// something is only compatible with it if it is also compatible with type A
    /// **and** type B. This means the fields/operations of this type will be a
    /// combination of those from A and B.
    ///
    /// This is useful for requiring types that implement a multitude of traits,
    /// i.e. the type ``Iterator & Joinable``.
    And(SpannedTypeExpr, SpannedTypeExpr),
    /// Given ?T, this becomes Nullable(Ident(T))
    /// Equivalent to T | null.
    Nullable(SpannedTypeExpr),
    /// Given !T, this becomes Not(Ident(T))
    ///
    /// Only types that are not compatible with T will be compatible with !T.
    Not(SpannedTypeExpr),
    /// Given T[], this becomes Array(Ident(T), None).
    /// Given T[n], where n is u32, this becomes Array(Ident(T), Some(n)).
    ///
    /// This represents an Array of T, and if n is specified, an array with
    /// such capacity.
    Array(SpannedTypeExpr, Option<u32>),
    /// Given [A, B], this becomes Tuple([Ident(A), Ident(B)]).
    ///
    /// A tuple is an array with an exact number of elements but with
    /// varying types throughout each element.
    Tuple(Vec<SpannedTypeExpr>),
    /// The constant null type. Only `null` will be compatible with this,
    /// nothing else.
    Null,
    /// The `auto` type, which is the default type and means that
    /// the type of the object is inferred.
    ///
    /// Given the type cannot be inferred and Terbium was configured to not
    /// raise an error given such scenario happens, this will default to
    /// the `any` type.
    Auto,
    /// The `any` type. It is compatible with any other type and any other type is
    /// compatible with it, including `null`.
    Any,
}

pub trait ParseInterface {
    // Parses the given tokens using this parser.
    ///
    /// # Errors
    /// * The tokens does not match Terbium grammar.
    fn parse(tokens: Vec<(Token, Span)>) -> Result<Self, Vec<Error>>
    where
        Self: Sized;

    /// Tokenizes source code and parses it using this parser.
    ///
    /// # Errors
    /// * The string does not match Terbium grammar.
    fn from_string(source: Source, s: String) -> Result<Self, Vec<Error>>
    where
        Self: Sized,
    {
        let tokens = get_lexer().parse(Stream::<_, Span, _>::from_iter(
            Span::single(source.clone(), s.chars().count()),
            s.chars()
                .enumerate()
                .map(|(i, c)| (c, Span::single(source.clone(), i))),
        ))?;

        Self::parse(tokens)
    }

    /// Reads and tokenizes the specified source file.
    ///
    /// # Errors
    /// * The source file's content does not match Terbium grammar.
    fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Vec<Error>>
    where
        Self: Sized,
    {
        let content = std::fs::read_to_string(path.as_ref()).expect("file is not valid utf-8");

        Self::from_string(Source::from_path(path.as_ref()), content)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Target {
    // Could represent a variable or a parameter. Supports destructuring.
    Ident(String),
    Array(Vec<SpannedTarget>),
    Attr(SpannedTarget, String), // Invalid as a parameter in a declaration statement or parameter.
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    target: SpannedTarget,
    ty: SpannedTypeExpr,
    default: Option<SpannedExpr>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Module(Vec<SpannedNode>),
    Func {
        name: String,
        params: Vec<SpannedParam>,
        body: Vec<SpannedNode>,
        return_last: bool,
        return_ty: SpannedTypeExpr,
    },
    Expr(SpannedExpr),
    // e.g. x.y = z becomes Assign { target: Attr(Ident("x"), "y"), value: Ident("z"), .. }
    Declare {
        targets: Vec<SpannedTarget>,
        ty: SpannedTypeExpr,
        value: SpannedExpr,
        r#mut: bool,
        r#const: bool,
    },
    Assign {
        targets: Vec<SpannedTarget>,
        value: SpannedExpr,
    },
    Return(Option<SpannedExpr>),
    Require(Vec<String>), // TODO: require y from x; require * from x
}

#[derive(Clone, Debug, PartialEq)]
pub struct Body(pub Vec<SpannedNode>, pub bool); // body, return_last

impl ParseInterface for Vec<(Token, Span)> {
    fn parse(tokens: Vec<(Token, Span)>) -> Result<Self, Vec<Error>>
    where
        Self: Sized,
    {
        Ok(tokens)
    }
}

impl ParseInterface for Expr {
    fn parse(mut tokens: Vec<(Token, Span)>) -> Result<Self, Vec<Error>>
    where
        Self: Sized,
    {
        let (last, span) = tokens.last().unwrap();
        let span = span.clone();

        if last != &Token::Semicolon {
            tokens.push((Token::Semicolon, Span::single(span.src(), span.end())));
        }

        get_body_parser()
            .then_ignore(end())
            .map(|b| {
                let Body(body, _) = b.into_node();

                if let Node::Expr(e) = body.get(0).unwrap().node() {
                    e.clone().into_node()
                } else {
                    unreachable!();
                }
            })
            .parse(Stream::<_, Span, _>::from_iter(
                Span::single(span.src(), span.end() + 1),
                tokens.into_iter(),
            ))
    }
}

impl ParseInterface for Node {
    fn parse(tokens: Vec<(Token, Span)>) -> Result<Self, Vec<Error>>
    where
        Self: Sized,
    {
        let (_, span) = tokens.last().unwrap();

        get_body_parser()
            .then_ignore(end())
            .map(|body| Self::Module(body.into_node().0))
            .parse(Stream::<_, Span, _>::from_iter(
                Span::single(span.src(), span.end()),
                tokens.into_iter(),
            ))
    }
}

impl ParseInterface for Body {
    fn parse(tokens: Vec<(Token, Span)>) -> Result<Self, Vec<Error>>
    where
        Self: Sized,
    {
        let (_, span) = tokens.last().unwrap();

        get_body_parser()
            .then_ignore(end())
            .parse(Stream::<_, Span, _>::from_iter(
                Span::single(span.src(), span.end()),
                tokens.into_iter(),
            ))
            .map(Spanned::into_node)
    }
}

pub trait CommonParser<T> = Parser<Token, T, Error = Error> + Clone;
pub type RecursiveParser<'a, T> = Recursive<'a, Token, T, Error>;

pub fn nested_parser<'a, T: 'a>(
    parser: impl CommonParser<T> + 'a,
    delimiter: Bracket,
    f: impl Fn(Span) -> T + Clone + 'a,
) -> impl CommonParser<T> + 'a {
    parser
        .delimited_by(
            just(Token::StartBracket(delimiter)),
            just(Token::EndBracket(delimiter)),
        )
        .recover_with(nested_delimiters(
            Token::StartBracket(delimiter),
            Token::EndBracket(delimiter),
            [
                (
                    Token::StartBracket(Bracket::Paren),
                    Token::EndBracket(Bracket::Paren),
                ),
                (
                    Token::StartBracket(Bracket::Bracket),
                    Token::EndBracket(Bracket::Bracket),
                ),
                (
                    Token::StartBracket(Bracket::Brace),
                    Token::EndBracket(Bracket::Brace),
                ),
            ],
            f,
        ))
        .boxed()
}

#[must_use]
#[allow(clippy::too_many_lines, clippy::missing_panics_doc)]
pub fn get_body_parser<'a>() -> RecursiveParser<'a, SpannedBody> {
    recursive(|body: Recursive<Token, SpannedBody, Error>| {
        let ty = recursive(|ty: Recursive<Token, SpannedTypeExpr, Error>| {
            let ident = select! {
                Token::Identifier(s) => match s.as_str() {
                    "any" => TypeExpr::Any,
                    "auto" => TypeExpr::Auto,
                    "null" => TypeExpr::Null,
                    _ => TypeExpr::Ident(s),
                },
            }
            .map_with_span(Spanned::new)
            .labelled("type identifier");

            let array = ty
                .clone()
                .then(
                    select!(Token::Literal(Literal::Integer(i)) => i)
                        .or_not()
                        .delimited_by(
                            just(Token::StartBracket(Bracket::Bracket)),
                            just(Token::StartBracket(Bracket::Bracket)),
                        ),
                )
                .try_map(|(t, n), span| {
                    Ok(Spanned::new(
                        TypeExpr::Array(
                            t,
                            if let Some(n) = n {
                                Some(n.try_into().map_err(|_| {
                                    Error::custom(span.clone(), "fixed array length is too large")
                                })?)
                            } else {
                                None
                            },
                        ),
                        span,
                    ))
                });

            let tuple = ty
                .clone()
                .separated_by(just::<_, Token, Error>(Token::Comma))
                .allow_trailing()
                .delimited_by(
                    just(Token::StartBracket(Bracket::Bracket)),
                    just(Token::StartBracket(Bracket::Bracket)),
                )
                .map_with_span(|a, span| Spanned::new(TypeExpr::Tuple(a), span));

            let atom = choice((
                ident,
                array,
                tuple,
                ty.clone()
                    .delimited_by(
                        just(Token::StartBracket(Bracket::Paren)),
                        just(Token::EndBracket(Bracket::Paren)),
                    )
                    .boxed(),
            ));

            let attr = atom
                .clone()
                .then(
                    just::<_, Token, _>(Token::Dot)
                        .ignore_then(
                            select!(Token::Identifier(s) => s)
                                .map_with_span(Spanned::<String>::new),
                        )
                        .repeated(),
                )
                .foldl(|t, ident| {
                    let span = t.span().merge(ident.span());

                    Spanned::new(TypeExpr::Attr(t, ident.into_node()), span)
                })
                .boxed();

            let generic = attr
                .clone()
                .then(
                    ty.clone()
                        .separated_by(just::<_, Token, _>(Token::Comma))
                        .allow_trailing()
                        .delimited_by(
                            just(Token::Operator(Operator::Lt)),
                            just(Token::Operator(Operator::Gt)),
                        )
                        .or_not(),
                )
                .map_with_span(|(subject, params), span| match params {
                    Some(params) => Spanned::new(TypeExpr::Generic(subject, params), span),
                    None => subject,
                })
                .boxed();

            let nullable = just(Token::Question)
                .map_with_span(Spanned::<Token>::new)
                .repeated()
                .then(generic.clone())
                .foldr(|op, t| {
                    let span = op.span().merge(t.span());

                    Spanned::new(TypeExpr::Nullable(t), span)
                })
                .boxed();

            let not = just(Token::Operator(Operator::Not))
                .map_with_span(Spanned::<Token>::new)
                .repeated()
                .then(nullable.clone())
                .foldr(|op, t| {
                    let span = op.span().merge(t.span());

                    Spanned::new(TypeExpr::Not(t), span)
                })
                .boxed();

            let and = not
                .clone()
                .then(
                    just(Token::Operator(Operator::BitAnd))
                        .ignore_then(not)
                        .repeated(),
                )
                .foldl(|lhs, rhs| {
                    let span = lhs.span().merge(rhs.span());

                    Spanned::new(TypeExpr::And(lhs, rhs), span)
                })
                .boxed();

            and.clone()
                .then(
                    just(Token::Operator(Operator::BitOr))
                        .ignore_then(and)
                        .repeated(),
                )
                .foldl(|lhs, rhs| {
                    let span = lhs.span().merge(rhs.span());

                    Spanned::new(TypeExpr::Union(lhs, rhs), span)
                })
                .boxed()
        });

        let e = recursive(|e: Recursive<Token, SpannedExpr, Error>| {
            let literal = select! {
                Token::Literal(lit) => match lit {
                    Literal::Integer(i) => Expr::Integer(i),
                    Literal::Float(f) => Expr::Float(f),
                    Literal::String(s) => match s {
                        StringLiteral::String(s) => Expr::String(s),
                        _ => unreachable!(),
                    },
                }
            }
            .map_with_span(SpannedExpr::new)
            .labelled("literal");

            let ident = select! {
                Token::Identifier(s) => match s.as_str() {
                    "true" => Expr::Bool(true),
                    "false" => Expr::Bool(false),
                    _ => Expr::Ident(s),
                },
            }
            .map_with_span(SpannedExpr::new)
            .labelled("identifier");

            let array = e
                .clone()
                .separated_by(just::<_, Token, Error>(Token::Comma))
                .allow_trailing()
                .delimited_by(
                    just(Token::StartBracket(Bracket::Bracket)),
                    just(Token::EndBracket(Bracket::Bracket)),
                )
                .map_with_span(|a, span| Spanned::new(Expr::Array(a), span));

            let if_stmt = just::<_, Token, Error>(Token::Keyword(Keyword::If))
                .ignore_then(e.clone())
                .then(body.clone().delimited_by(
                    just(Token::StartBracket(Bracket::Brace)),
                    just(Token::EndBracket(Bracket::Brace)),
                ))
                .then(
                    just::<_, Token, _>(Token::Keyword(Keyword::Else))
                        .ignore_then(just(Token::Keyword(Keyword::If)))
                        .ignore_then(e.clone())
                        .then(body.clone().delimited_by(
                            just(Token::StartBracket(Bracket::Brace)),
                            just(Token::EndBracket(Bracket::Brace)),
                        ))
                        .repeated(),
                )
                .then(
                    just::<_, Token, _>(Token::Keyword(Keyword::Else))
                        .ignore_then(body.clone().delimited_by(
                            just(Token::StartBracket(Bracket::Brace)),
                            just(Token::EndBracket(Bracket::Brace)),
                        ))
                        .or_not(),
                )
                .map_with_span(|(((condition, body), else_if), else_body), span| {
                    SpannedExpr::new(
                        Expr::If {
                            condition,
                            body,
                            else_if_bodies: else_if,
                            else_body,
                        },
                        span,
                    )
                });

            let while_stmt = just::<_, Token, Error>(Token::Keyword(Keyword::While))
                .ignore_then(e.clone())
                .then(body.clone().delimited_by(
                    just(Token::StartBracket(Bracket::Brace)),
                    just(Token::EndBracket(Bracket::Brace)),
                ))
                .map_with_span(|(condition, body), span| {
                    SpannedExpr::new(
                        Expr::While {
                            condition,
                            body: body.into_node().0,
                        },
                        span,
                    )
                });

            let atom = choice((
                literal,
                ident,
                e.clone()
                    .delimited_by(
                        just(Token::StartBracket(Bracket::Paren)),
                        just(Token::EndBracket(Bracket::Paren)),
                    )
                    .boxed(),
                if_stmt,
                while_stmt,
                array,
            ))
            .boxed();

            let attr = atom
                .clone()
                .then(
                    just::<_, Token, Error>(Token::Dot)
                        .ignore_then(ident)
                        .repeated(),
                )
                .foldl(|expr, ident| {
                    let span = expr.span();

                    SpannedExpr::new(
                        Expr::Attr(
                            expr,
                            match ident.node() {
                                Expr::Ident(s) => s.to_string(),
                                Expr::Bool(b) => b.to_string(),
                                _ => unreachable!(),
                            },
                        ),
                        span.merge(ident.span()),
                    )
                })
                .boxed();

            let call = attr
                .clone()
                .then(
                    e.clone()
                        .separated_by(just::<_, Token, Error>(Token::Comma))
                        .allow_trailing()
                        .delimited_by(
                            just(Token::StartBracket(Bracket::Paren)),
                            just(Token::EndBracket(Bracket::Paren)),
                        )
                        .or_not(),
                )
                .map_with_span(|(expr, args), span| match args {
                    Some(args) => SpannedExpr::new(
                        Expr::Call {
                            value: expr,
                            args,
                            kwargs: vec![],
                        },
                        span,
                    ),
                    None => expr,
                })
                .boxed();

            let spanned_op = |o: Token, span: Span| -> SpannedOperator {
                match o {
                    Token::Operator(op) => SpannedOperator::new(op, span),
                    _ => unreachable!(),
                }
            };

            let unary = just(Token::Operator(Operator::Sub))
                .or(just(Token::Operator(Operator::Add)))
                .or(just(Token::Operator(Operator::Not)))
                .or(just(Token::Operator(Operator::BitNot)))
                .map_with_span(spanned_op)
                .repeated()
                .then(call.clone())
                .foldr(|operator, expr| {
                    let span = operator.span().merge(expr.span());

                    SpannedExpr::new(
                        Expr::UnaryExpr {
                            operator,
                            value: expr,
                        },
                        span,
                    )
                })
                .boxed();

            let binary_cast = unary
                .clone()
                .then(just(Token::Cast).ignore_then(unary).repeated())
                .foldl(|subject, ty| {
                    let span = subject.span().merge(ty.span());

                    SpannedExpr::new(Expr::Cast(subject, ty), span)
                })
                .boxed();

            let binary_pow = binary_cast
                .clone()
                .then(just(Token::Operator(Operator::Pow)).map_with_span(spanned_op))
                .repeated()
                .then(binary_cast)
                .foldr(|(lhs, operator), rhs| {
                    let span = lhs.span().merge(rhs.span());

                    SpannedExpr::new(Expr::BinaryExpr { operator, lhs, rhs }, span)
                })
                .boxed();

            let op = just(Token::Operator(Operator::Mul))
                .or(just(Token::Operator(Operator::Div)))
                .or(just(Token::Operator(Operator::Mod)))
                .map_with_span(spanned_op);
            let binary_product = binary_pow
                .clone()
                .then(op.then(binary_pow).repeated())
                .foldl(|lhs, (operator, rhs)| {
                    let span = lhs.span().merge(rhs.span());

                    Spanned::new(Expr::BinaryExpr { operator, lhs, rhs }, span)
                })
                .boxed();

            let op = just(Token::Operator(Operator::Add))
                .or(just(Token::Operator(Operator::Sub)))
                .map_with_span(spanned_op);
            let binary_sum = binary_product
                .clone()
                .then(op.then(binary_product).repeated())
                .foldl(|lhs, (operator, rhs)| {
                    let span = lhs.span().merge(rhs.span());

                    Spanned::new(Expr::BinaryExpr { operator, lhs, rhs }, span)
                })
                .boxed();

            let op = just(Token::Operator(Operator::Eq))
                .or(just(Token::Operator(Operator::Ne)))
                .or(just(Token::Operator(Operator::Lt)))
                .or(just(Token::Operator(Operator::Gt)))
                .or(just(Token::Operator(Operator::Le)))
                .or(just(Token::Operator(Operator::Ge)))
                .map_with_span(spanned_op);
            let binary_cmp = binary_sum
                .clone()
                .then(op.then(binary_sum).repeated())
                .foldl(|lhs, (operator, rhs)| {
                    let span = lhs.span().merge(rhs.span());

                    Spanned::new(Expr::BinaryExpr { operator, lhs, rhs }, span)
                })
                .boxed();

            let binary_logical_and = binary_cmp
                .clone()
                .then(
                    just(Token::Operator(Operator::And))
                        .map_with_span(spanned_op)
                        .then(binary_cmp)
                        .repeated(),
                )
                .foldl(|lhs, (operator, rhs)| {
                    let span = lhs.span().merge(rhs.span());

                    Spanned::new(Expr::BinaryExpr { operator, lhs, rhs }, span)
                })
                .boxed();

            let binary_logical_or = binary_logical_and
                .clone()
                .then(
                    just(Token::Operator(Operator::Or))
                        .map_with_span(spanned_op)
                        .then(binary_logical_and)
                        .repeated(),
                )
                .foldl(|lhs, (operator, rhs)| {
                    let span = lhs.span().merge(rhs.span());

                    Spanned::new(Expr::BinaryExpr { operator, lhs, rhs }, span)
                })
                .boxed();

            let op = just(Token::Operator(Operator::BitAnd))
                .or(just(Token::Operator(Operator::BitOr)))
                .or(just(Token::Operator(Operator::BitXor)))
                .map_with_span(spanned_op);
            binary_logical_or
                .clone()
                .then(op.then(binary_logical_or).repeated())
                .foldl(|lhs, (operator, rhs)| {
                    let span = lhs.span().merge(rhs.span());

                    Spanned::new(Expr::BinaryExpr { operator, lhs, rhs }, span)
                })
                .boxed()
        });

        let require = just::<_, Token, _>(Token::Keyword(Keyword::Require))
            .ignore_then(
                select! {
                    Token::Identifier(i) => i,
                }
                .separated_by(just::<_, Token, _>(Token::Comma))
                .allow_trailing()
                .at_least(1),
            )
            .then_ignore(just::<_, Token, _>(Token::Semicolon))
            .map_with_span(|n, span| Spanned::new(Node::Require(n), span));

        // TODO: support more targets
        let target = recursive(|t| {
            select!(
                Token::Identifier(i) => Target::Ident(i),
            )
            .map_with_span(Spanned::new)
            .or(t
                .separated_by(just::<_, Token, _>(Token::Comma))
                .allow_trailing()
                .at_least(1)
                .delimited_by(
                    just(Token::StartBracket(Bracket::Bracket)),
                    just(Token::EndBracket(Bracket::Bracket)),
                )
                .map_with_span(|a, span| Spanned::new(Target::Array(a), span)))
        });

        let declare = just::<_, Token, _>(Token::Keyword(Keyword::Let))
            .or(just(Token::Keyword(Keyword::Const)))
            .then(just(Token::Keyword(Keyword::Mut)).or_not())
            .map_with_span(Spanned::<(Token, Option<Token>)>::new)
            .then(
                target
                    .clone()
                    .then(
                        just(Token::Colon)
                            .ignore_then(ty.clone())
                            .or_not()
                            .map_with_span(|t, span| {
                                t.unwrap_or_else(|| Spanned::new(TypeExpr::Auto, span))
                            }),
                    )
                    .then_ignore(just::<_, Token, _>(Token::Assign)),
            )
            .then(e.clone())
            .then_ignore(just::<_, Token, _>(Token::Semicolon))
            .try_map(|((modifiers, (target, ty)), expr), span| {
                let (modifier, is_mut) = modifiers.node();

                let r#mut = is_mut.is_some();
                let r#const = matches!(modifier, Token::Keyword(Keyword::Const));

                if r#mut && r#const {
                    return Err(Error::no_const_mut(modifiers.span()));
                }

                Ok(Spanned::new(
                    Node::Declare {
                        targets: vec![target],
                        ty,
                        value: expr,
                        r#mut,
                        r#const,
                    },
                    span,
                ))
            });

        let assign = target
            .clone()
            .then_ignore(just::<_, Token, _>(Token::Assign))
            .repeated()
            .at_least(1)
            .then(e.clone())
            .then_ignore(just::<_, Token, _>(Token::Semicolon))
            .map_with_span(|(targets, expr), span| {
                Spanned::new(
                    Node::Assign {
                        targets,
                        value: expr,
                    },
                    span,
                )
            });

        let param = target
            .clone()
            .then(
                just(Token::Colon)
                    .ignore_then(ty.clone())
                    .or_not()
                    .map_with_span(|t, span| {
                        t.unwrap_or_else(|| Spanned::new(TypeExpr::Auto, span))
                    }),
            )
            .then(
                just::<_, Token, _>(Token::Assign)
                    .ignore_then(e.clone())
                    .or_not(),
            )
            .map_with_span(|((target, ty), default), span| {
                Spanned::new(
                    Param {
                        target,
                        ty,
                        default,
                    },
                    span,
                )
            });

        let func = just::<_, Token, _>(Token::Keyword(Keyword::Func))
            .ignore_then(select! {
                Token::Identifier(i) => i,
            })
            .then(
                param
                    .separated_by(just::<_, Token, _>(Token::Comma))
                    .allow_trailing()
                    .delimited_by(
                        just(Token::StartBracket(Bracket::Paren)),
                        just(Token::EndBracket(Bracket::Paren)),
                    ),
            )
            .then(
                just(Token::Arrow)
                    .ignore_then(ty)
                    .or_not()
                    .map_with_span(|t, span| {
                        t.unwrap_or_else(|| Spanned::new(TypeExpr::Auto, span))
                    }),
            )
            .then(body.clone().delimited_by(
                just(Token::StartBracket(Bracket::Brace)),
                just(Token::EndBracket(Bracket::Brace)),
            ))
            .map_with_span(|(((name, params), return_ty), body), span| {
                let Body(body, return_last) = body.into_node();

                Spanned::new(
                    Node::Func {
                        name,
                        params,
                        body,
                        return_last,
                        return_ty,
                    },
                    span,
                )
            });

        let r#return = just::<_, Token, _>(Token::Keyword(Keyword::Return))
            .ignore_then(e.clone().or_not())
            .then_ignore(just::<_, Token, _>(Token::Semicolon))
            .map_with_span(|e, span| Spanned::new(Node::Return(e), span));

        let expr = e
            .clone()
            .then_ignore(just::<_, Token, _>(Token::Semicolon))
            .or(e
                .clone()
                .try_map(|e, _| match e.node() {
                    Expr::If { .. } | Expr::While { .. } => Ok(e),
                    _ => Err(Error::placeholder()),
                })
                .then_ignore(none_of(Token::EndBracket(Bracket::Brace)).rewind()))
            .map_with_span(|e, span| Spanned::new(Node::Expr(e), span));

        choice((func, declare, assign, r#return, require, expr))
            .repeated()
            .then(
                e.clone()
                    .or_not()
                    .map_with_span(|o, span| o.map(|o| Spanned::new(Node::Expr(o), span))),
            )
            .recover_with(skip_then_retry_until([]))
            .map_with_span(|(mut nodes, last), span| {
                Spanned::new(
                    {
                        let return_last = last.is_some();
                        if let Some(last) = last {
                            nodes.push(last);
                        }
                        Body(nodes, return_last)
                    },
                    span,
                )
            })
    })
}

// TODO: write tests
