use super::Error;
use super::token::{get_lexer, Bracket, Keyword, Literal, Operator, StringLiteral, Token};

use chumsky::prelude::*;
use chumsky::primitive::FilterMap;

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Integer(u128),
    Float(String), // See token.rs for why this is a String
    String(String),
    Bool(bool),
    Ident(String),
    Array(Vec<Expr>),
    Cast(Box<Expr>, Box<Expr>),
    UnaryExpr {
        operator: Operator,
        value: Box<Expr>,
    },
    BinaryExpr {
        operator: Operator,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Attr(Box<Expr>, String),
    Call {
        value: Box<Expr>,
        args: Vec<Expr>,
        kwargs: Vec<(String, Expr)>,
    },
    If {
        condition: Box<Expr>,
        body: Vec<Node>,
        else_if_bodies: Vec<(Expr, Body)>,
        else_body: Option<Body>,
        return_last: bool,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum TypeExpr {
    /// Given T, this becomes Ident("T")
    Ident(String),
    /// Given mod.Type, this becomes Attr(Ident(mod), "type")
    Attr(Box<TypeExpr>, String),
    /// Given Type<A, B>, this becomes Generic(Ident(Type), [Ident(A), Ident(B)])
    Generic(Box<TypeExpr>, Vec<TypeExpr>),
    /// Given A | B, this becomes Union(Ident(A), Ident(B))
    ///
    /// Terbium handles Unions wide-to-narrow. This means given type A | B,
    /// something compatible with either type A or B will be compatible with it,
    /// but only fields/operations that exist on **both** A and B will exist on the type.
    Union(Vec<TypeExpr>),
    /// Given A & B, this becomes And(Ident(A), Ident(B))
    ///
    /// Terbium handles And narrow-to-wide. This means given type A & B,
    /// something is only compatible with it if it is also compatible with type A
    /// **and** type B. This means the fields/operations of this type will be a
    /// combination of those from A and B.
    ///
    /// This is useful for requiring types that implement a multitude of traits,
    /// i.e. the type ``Iterator & Joinable``.
    And(Vec<TypeExpr>),
    /// Given ?T, this becomes Nullable(Ident(T))
    /// Equivalent to T | null.
    Nullable(Box<TypeExpr>),
    /// Given !T, this becomes Not(Ident(T))
    ///
    /// Only types that are not compatible with T will be compatible with !T.
    Not(Box<TypeExpr>),
    /// Given T[], this becomes Array(Ident(T), None).
    /// Given T[n], where n is u32, this becomes Array(Ident(T), Some(n)).
    ///
    /// This represents an Array of T, and if n is specified, an array with
    /// such capacity.
    Array(Box<TypeExpr>, Option<u32>),
    /// Given [A, B], this becomes Tuple([Ident(A), Ident(B)]).
    ///
    /// A tuple is an array with an exact number of elements but with
    /// varying types throughout each element.
    Tuple(Vec<TypeExpr>),
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
    fn parse(tokens: Vec<Token>) -> (Option<Self>, Vec<Error>)
    where
        Self: Sized;

    #[must_use]
    fn from_tokens(tokens: Vec<Token>) -> (Self, Vec<Error>)
    where
        Self: Sized,
    {
        let (expr, errors) = Self::parse(tokens);

        (expr.unwrap(), errors)
    }

    fn from_string(s: String) -> Result<(Self, Vec<Error>), Vec<Error>>
    where
        Self: Sized,
    {
        let tokens = get_lexer().parse(s.as_str())?;

        Ok(Self::from_tokens(tokens))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Target {
    // Could represent a variable or a parameter. Supports destructuring.
    Ident(String),
    Array(Vec<Target>),
    Attr(Box<Target>, String), // Invalid as a parameter or when let/immut is used.
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    // TODO: typing
    target: Target,
    default: Option<Expr>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Module(Vec<Node>),
    Func {
        name: String,
        params: Vec<Param>,
        body: Vec<Node>,
        return_last: bool,
    },
    Expr(Expr),
    // e.g. x.y = z becomes Assign { target: Attr(Ident("x"), "y"), value: Ident("z"), .. }
    Assign {
        targets: Vec<Target>,
        value: Expr,
        r#let: bool,
        immut: bool,
        r#const: bool,
    },
    Return(Option<Expr>),
    Require(Vec<String>), // TODO: require y from x; require * from x
}

#[derive(Clone, Debug, PartialEq)]
pub struct Body(pub Vec<Node>, pub bool); // body, return_last

impl ParseInterface for Expr {
    fn parse(mut tokens: Vec<Token>) -> (Option<Self>, Vec<Error>)
    where
        Self: Sized,
    {
        if tokens.last().unwrap() != &Token::Semicolon {
            tokens.push(Token::Semicolon);
        }

        get_body_parser()
            .map(|Body(body, _)| {
                if let Node::Expr(e) = body.get(0).unwrap() {
                    e.to_owned()
                } else {
                    unreachable!();
                }
            })
            .parse_recovery(tokens)
    }
}

impl ParseInterface for Node {
    fn parse(tokens: Vec<Token>) -> (Option<Self>, Vec<Error>)
    where
        Self: Sized,
    {
        get_body_parser()
            .map(|Body(b, _)| Self::Module(b))
            .parse_recovery(tokens)
    }
}

impl ParseInterface for Body {
    fn parse(tokens: Vec<Token>) -> (Option<Self>, Vec<Error>)
    where
        Self: Sized,
    {
        get_body_parser().parse_recovery(tokens)
    }
}

pub trait CommonParser<T> = Parser<Token, T, Error = Error> + Clone;
pub type RecursiveParser<'a, T> = Recursive<'a, Token, T, Error>;

#[must_use]
pub fn get_body_parser<'a>() -> RecursiveParser<'a, Body> {
    recursive(|body: Recursive<Token, Body, Error>| {
        let e = recursive(|e: Recursive<Token, Expr, Error>| {
            let literal: FilterMap<_, Error> = select! {
                Token::Literal(lit) => match lit {
                    Literal::Integer(i) => Expr::Integer(i),
                    Literal::Float(f) => Expr::Float(f),
                    Literal::String(s) => match s {
                        StringLiteral::String(s) => Expr::String(s),
                        _ => unreachable!(),
                    },
                }
            };

            let ident = select! {
                Token::Identifier(s) => match s.as_str() {
                    "true" => Expr::Bool(true),
                    "false" => Expr::Bool(false),
                    _ => Expr::Ident(s),
                }
            };

            let array = e
                .clone()
                .separated_by(just::<_, Token, _>(Token::Comma))
                .allow_trailing()
                .delimited_by(
                    just(Token::StartBracket(Bracket::Bracket)),
                    just(Token::EndBracket(Bracket::Bracket)),
                )
                .map(Expr::Array);

            let if_stmt = just::<_, Token, _>(Token::Keyword(Keyword::If))
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
                .map(
                    |(((condition, Body(body, return_last)), else_if), else_body)| Expr::If {
                        condition: Box::new(condition),
                        body,
                        else_if_bodies: else_if,
                        else_body,
                        return_last,
                    },
                );

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
                array,
            ))
            .boxed();

            let attr = atom
                .clone()
                .then(
                    just::<_, Token, _>(Token::Dot)
                        .ignore_then(ident)
                        .repeated(),
                )
                .foldl(|expr, ident| {
                    Expr::Attr(
                        Box::new(expr),
                        match ident {
                            Expr::Ident(s) => s,
                            Expr::Bool(b) => b.to_string(),
                            _ => unreachable!(),
                        },
                    )
                })
                .boxed();

            let call = attr
                .clone()
                .then(
                    e.clone()
                        .separated_by(just::<_, Token, _>(Token::Comma))
                        .allow_trailing()
                        .delimited_by(
                            just(Token::StartBracket(Bracket::Paren)),
                            just(Token::EndBracket(Bracket::Paren)),
                        )
                        .or_not(),
                )
                .map(|(expr, args)| match args {
                    Some(args) => Expr::Call {
                        value: Box::new(expr),
                        args,
                        kwargs: vec![],
                    },
                    None => expr,
                })
                .boxed();

            let unary = just(Token::Operator(Operator::Sub))
                .or(just(Token::Operator(Operator::Add)))
                .or(just(Token::Operator(Operator::Not)))
                .or(just(Token::Operator(Operator::BitNot)))
                .repeated()
                .then(call.clone())
                .foldr(|operator, expr| match operator {
                    Token::Operator(operator) => Expr::UnaryExpr {
                        operator,
                        value: Box::new(expr),
                    },
                    _ => unreachable!(),
                })
                .boxed();

            let binary_cast = unary
                .clone()
                .then(just(Token::Cast).ignore_then(unary).repeated())
                .foldl(|subject, ty| Expr::Cast(
                    Box::new(subject),
                    Box::new(ty)
                ))
                .boxed();

            let binary_pow = binary_cast
                .clone()
                .then(just(Token::Operator(Operator::Pow)).map(|o| match o {
                    Token::Operator(op) => op,
                    _ => unreachable!(),
                }))
                .repeated()
                .then(binary_cast)
                .foldr(|(lhs, operator), rhs| Expr::BinaryExpr {
                    operator,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                })
                .boxed();

            let op = just(Token::Operator(Operator::Mul))
                .or(just(Token::Operator(Operator::Div)))
                .or(just(Token::Operator(Operator::Mod)))
                .map(|o| match o {
                    Token::Operator(op) => op,
                    _ => unreachable!(),
                });
            let binary_product = binary_pow
                .clone()
                .then(op.then(binary_pow).repeated())
                .foldl(|lhs, (operator, rhs)| Expr::BinaryExpr {
                    operator,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                })
                .boxed();

            let op = just(Token::Operator(Operator::Add))
                .or(just(Token::Operator(Operator::Sub)))
                .map(|o| match o {
                    Token::Operator(op) => op,
                    _ => unreachable!(),
                });
            let binary_sum = binary_product
                .clone()
                .then(op.then(binary_product).repeated())
                .foldl(|lhs, (operator, rhs)| Expr::BinaryExpr {
                    operator,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                })
                .boxed();

            let op = just(Token::Operator(Operator::Eq))
                .or(just(Token::Operator(Operator::Ne)))
                .or(just(Token::Operator(Operator::Lt)))
                .or(just(Token::Operator(Operator::Gt)))
                .or(just(Token::Operator(Operator::Le)))
                .or(just(Token::Operator(Operator::Ge)))
                .map(|o| match o {
                    Token::Operator(op) => op,
                    _ => unreachable!(),
                });
            let binary_cmp = binary_sum
                .clone()
                .then(op.then(binary_sum).repeated())
                .foldl(|lhs, (operator, rhs)| Expr::BinaryExpr {
                    operator,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                })
                .boxed();

            let binary_logical_and = binary_cmp
                .clone()
                .then(
                    just(Token::Operator(Operator::And))
                        .map(|o| match o {
                            Token::Operator(op) => op,
                            _ => unreachable!(),
                        })
                        .then(binary_cmp)
                        .repeated(),
                )
                .foldl(|lhs, (operator, rhs)| Expr::BinaryExpr {
                    operator,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                })
                .boxed();

            let binary_logical_or = binary_logical_and
                .clone()
                .then(
                    just(Token::Operator(Operator::Or))
                        .map(|o| match o {
                            Token::Operator(op) => op,
                            _ => unreachable!(),
                        })
                        .then(binary_logical_and)
                        .repeated(),
                )
                .foldl(|lhs, (operator, rhs)| Expr::BinaryExpr {
                    operator,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                })
                .boxed();

            let op = just(Token::Operator(Operator::BitAnd))
                .or(just(Token::Operator(Operator::BitOr)))
                .or(just(Token::Operator(Operator::BitXor)))
                .map(|o| match o {
                    Token::Operator(op) => op,
                    _ => unreachable!(),
                });
            binary_logical_or
                .clone()
                .then(op.then(binary_logical_or).repeated())
                .foldl(|lhs, (operator, rhs)| Expr::BinaryExpr {
                    operator,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
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
            .map(Node::Require);

        let assign = just::<_, Token, _>(Token::Keyword(Keyword::Let))
            .or_not()
            .then(just::<_, Token, _>(Token::Keyword(Keyword::Immut)).or_not())
            .or(just(Token::Keyword(Keyword::Const))
                .map(|_| (None, Some(Token::Keyword(Keyword::Const)))))
            .or_not()
            .then(select! {
                // TODO: target parser (only supports raw idents)
                Token::Identifier(i) => i,
            })
            .then(
                just::<_, Token, _>(Token::Assign)
                    .ignore_then(select! { Token::Identifier(i) => i })
                    .repeated(),
            )
            .then_ignore(just::<_, Token, _>(Token::Assign))
            .then(e.clone())
            .then_ignore(just::<_, Token, _>(Token::Semicolon))
            .map(|(((modifiers, first_target), targets), expr)| {
                let (r#let, immut, r#const) = match modifiers {
                    Some((r#let, immut_or_const)) => match r#let {
                        Some(_) => (true, immut_or_const.is_some(), false),
                        None => match immut_or_const {
                            Some(Token::Keyword(Keyword::Const)) => (false, false, true),
                            Some(Token::Keyword(Keyword::Immut)) => (false, true, false),
                            Some(_) => unreachable!(),
                            None => (false, false, false),
                        },
                    },
                    None => (false, false, false),
                };

                let targets = vec![first_target]
                    .into_iter()
                    .chain(targets)
                    .map(Target::Ident)
                    .collect::<Vec<_>>();

                Node::Assign {
                    targets,
                    value: expr,
                    r#let,
                    immut,
                    r#const,
                }
            });

        let param = select! {
            Token::Identifier(i) => Target::Ident(i),
        } // TODO: type annotations
        .or(select! {
            Token::Identifier(i) => Target::Ident(i),
        }
        .separated_by(just::<_, Token, _>(Token::Comma))
        .allow_trailing()
        .at_least(1)
        .delimited_by(
            just::<_, Token, _>(Token::StartBracket(Bracket::Bracket)),
            just(Token::EndBracket(Bracket::Bracket)),
        )
        .map(Target::Array))
        .then(
            just::<_, Token, _>(Token::Assign)
                .ignore_then(e.clone())
                .or_not(),
        )
        .map(|(target, default)| Param { target, default });

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
            ) // TODO: return type annotation
            .then(body.clone().delimited_by(
                just(Token::StartBracket(Bracket::Brace)),
                just(Token::EndBracket(Bracket::Brace)),
            ))
            .map(|((name, params), Body(body, return_last))| Node::Func {
                name,
                params,
                body,
                return_last,
            });

        let r#return = just::<_, Token, _>(Token::Keyword(Keyword::Return))
            .ignore_then(e.clone().or_not())
            .then_ignore(just::<_, Token, _>(Token::Semicolon))
            .map(Node::Return);

        let expr = e
            .clone()
            .then_ignore(just::<_, Token, _>(Token::Semicolon))
            .or(e
                .clone()
                .try_map(|e, _| match e {
                    Expr::If { .. } => Ok(e),
                    _ => Err(Error::placeholder()),
                })
                .then_ignore(none_of(Token::EndBracket(Bracket::Brace)).rewind()))
            .map(Node::Expr);

        choice((func, assign, r#return, require, expr))
            .repeated()
            .then(e.clone().or_not().map(|o| o.map(Node::Expr)))
            .map(|(mut nodes, last)| {
                let return_last = last.is_some();
                if let Some(last) = last {
                    nodes.push(last);
                }
                Body(nodes, return_last)
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Expr::*;

    #[test]
    fn test_expr_parser() {
        let code = "-1 + 2 * (5 - [2, a.b() - (c + -d), e(5, f())])";
        let (tree, errors) = Expr::from_string(code.to_string()).unwrap();

        assert_eq!(
            tree,
            BinaryExpr {
                operator: Operator::Add,
                lhs: Box::new(UnaryExpr {
                    operator: Operator::Sub,
                    value: Box::new(Integer(1)),
                }),
                rhs: Box::new(BinaryExpr {
                    operator: Operator::Mul,
                    lhs: Box::new(Integer(2)),
                    rhs: Box::new(BinaryExpr {
                        operator: Operator::Sub,
                        lhs: Box::new(Integer(5)),
                        rhs: Box::new(Array(vec![
                            Integer(2),
                            BinaryExpr {
                                operator: Operator::Sub,
                                lhs: Box::new(Call {
                                    value: Box::new(Attr(
                                        Box::new(Ident("a".to_string())),
                                        "b".to_string()
                                    )),
                                    args: vec![],
                                    kwargs: vec![],
                                }),
                                rhs: Box::new(BinaryExpr {
                                    operator: Operator::Add,
                                    lhs: Box::new(Ident("c".to_string())),
                                    rhs: Box::new(UnaryExpr {
                                        operator: Operator::Sub,
                                        value: Box::new(Ident("d".to_string())),
                                    }),
                                }),
                            },
                            Call {
                                value: Box::new(Ident("e".to_string())),
                                args: vec![
                                    Integer(5),
                                    Call {
                                        value: Box::new(Ident("f".to_string())),
                                        args: vec![],
                                        kwargs: vec![],
                                    },
                                ],
                                kwargs: vec![],
                            },
                        ])),
                    }),
                }),
            }
        );
        assert_eq!(errors.len(), 0);
    }

    #[test]
    fn test_body_parser() {
        use crate::ast::Node::*;

        let code = r#"
            require std;

            func foo() {
                std.println("Hello, world!");

                if 1 + 1 == 2 {
                    let x = 5;
                }

                "10"::int
            }
        "#;
        let (tree, errors) = Body::from_string(code.to_string()).unwrap();

        assert_eq!(
            tree,
            Body(
                vec![
                    Require(vec!["std".to_string()]),
                    Func {
                        name: "foo".to_string(),
                        params: vec![],
                        body: vec![
                            Expr(Call {
                                value: Box::new(Attr(
                                    Box::new(Ident("std".to_string())),
                                    "println".to_string()
                                ),),
                                args: vec![String("Hello, world!".to_string()),],
                                kwargs: vec![],
                            }),
                            Expr(If {
                                condition: Box::new(BinaryExpr {
                                    operator: Operator::Eq,
                                    lhs: Box::new(BinaryExpr {
                                        operator: Operator::Add,
                                        lhs: Box::new(Integer(1)),
                                        rhs: Box::new(Integer(1)),
                                    }),
                                    rhs: Box::new(Integer(2)),
                                }),
                                body: vec![Assign {
                                    targets: vec![Target::Ident("x".to_string()),],
                                    value: Integer(5),
                                    r#let: true,
                                    immut: false,
                                    r#const: false,
                                },],
                                else_if_bodies: vec![],
                                else_body: None,
                                return_last: false,
                            }),
                            Expr(Cast(
                                Box::new(String("10".to_string())),
                                Box::new(Ident("int".to_string())),
                            )),
                        ],
                        return_last: true,
                    }
                ],
                false
            )
        );

        assert_eq!(errors.len(), 0);
    }
}
