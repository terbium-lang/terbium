use crate::Error;
use crate::token::{Bracket, get_lexer};
use super::token::{Literal, Operator, StringLiteral, Token};

use chumsky::Error as _;
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
    pub fn from_tokens(tokens: Vec<Token>) -> Self {
        let (expr, _) = get_expr_parser().parse_recovery(tokens);

        expr.unwrap()
    }

    pub fn from_string(s: String) -> Self {
        Self::from_tokens(get_lexer().parse(s.as_str()).unwrap())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Target { // Could represent a variable or a parameter. Supports destructuring.
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
    // e.g. x.y = z becomes Assign { lhs: Attr(Ident("x"), "y"), rhs: Ident("z") }
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
        let literal: FilterMap<_, Expr> = select! {
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
            .separated_by(just::<_, Token, _>(Token::Comma))
            .delimited_by(
                just(Token::StartBracket(Bracket::Bracket)),
                just(Token::EndBracket(Bracket::Bracket)),
            )
            .padded()
            .map(Expr::Array);

        let unary = filter(|token: &Token| match token {
            Token::Operator(op) => op.supports_unary(),
            _ => false,
        })
            .map(select! { Token::Operator(op) => op })
            .then(e.clone())
            .padded()
            .map(|(operator, value)| Expr::UnaryExpr { operator, value: Box::new(value) });

        let binary = e
            .then(
                filter(|token: &Token| match token {
                    Token::Operator(op) => op.supports_binary(),
                    _ => false,
                }).map(select! {
                    Token::Operator(op) => op,
                }),
            )
            .then(e.clone())
            .padded()
            .map(|((lhs, operator), rhs)| Expr::BinaryExpr {
                operator,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            });

        let attr = e
            .clone()
            .then_ignore(Token::Dot)
            .then(filter(|token: &Token| match token {
                Token::Identifier(_) => true,
                _ => false,
            })
            .map(|i| -> String { i.0 }))
            .padded()
            .map(|(obj, value)| Expr::Attr(Box::new(obj), value));

        let call = e
            .then(e
                .clone()
                .separated_by(just::<_, Token, _>(Token::Comma))
                .delimited_by(
                    just(Token::StartBracket(Bracket::Paren)),
                    just(Token::EndBracket(Bracket::Paren)),
                )
                .padded()
            )
            .padded()
            .map(|(val, args)| Expr::Call {
                value: Box::new(val),
                args,
                kwargs: vec![],
            });

        choice((literal, ident, array, unary, binary, attr, call))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expr_parser() {
        let code = "-1 + 2 * (5 - [2, a.b() - (c + -d)])";
        let tree = Expr::from_string(code.to_string());

        println!("{:#?}", tree);
    }
}