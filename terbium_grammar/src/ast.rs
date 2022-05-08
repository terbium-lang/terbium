use super::token::{Literal, Operator, StringLiteral, Token};
use crate::token::{get_lexer, Bracket};
use crate::Error;

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
}

impl Expr {
    pub fn from_tokens(tokens: Vec<Token>) -> (Self, Vec<Error>) {
        let (expr, errors) = get_expr_parser().parse_recovery(tokens);

        (expr.unwrap(), errors)
    }

    pub fn from_string(s: String) -> (Self, Vec<Error>) {
        Self::from_tokens(get_lexer().parse(s.as_str()).unwrap())
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
pub enum Node {
    Module(Vec<Node>),
    Func {
        name: String,
        params: Vec<Node>,
        body: Vec<Node>,
    },
    Expr(Expr),
    // e.g. x.y = z becomes Assign { target: Attr(Ident("x"), "y"), value: Ident("z"), .. }
    Assign {
        target: Target,
        value: Expr,
        r#let: bool,
        immut: bool,
    },
    Return(Option<Expr>),
}

pub trait CommonParser<T> = Parser<Token, T, Error = Error> + Clone;

pub fn get_expr_parser() -> impl CommonParser<Expr> {
    recursive(|e: Recursive<Token, Expr, Error>| {
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

        let atom = choice((
            literal,
            ident,
            e.clone()
                .delimited_by(
                    just(Token::StartBracket(Bracket::Paren)),
                    just(Token::EndBracket(Bracket::Paren)),
                )
                .boxed(),
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
            .foldl(|a, b| {
                Expr::Attr(
                    Box::new(a),
                    match b {
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

        let binary_pow = unary
            .clone()
            .then(
                just(Token::Operator(Operator::Pow))
                    .map(|o| match o {
                        Token::Operator(op) => op,
                        _ => unreachable!(),
                    })
                    .then(unary)
                    .repeated(),
            )
            .foldl(|lhs, (operator, rhs)| Expr::BinaryExpr {
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Expr::*;

    #[test]
    fn test_expr_parser() {
        let code = "-1 + 2 * (5 - [2, a.b() - (c + -d), e(5, f())])";
        let (tree, errors) = Expr::from_string(code.to_string());

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
}
