use grammar::{
    ast::{Atom, BinaryOp, Expr, UnaryOp},
    parser::Parser,
    span::{Provider, Span, Spanned, Src},
    token::{IntLiteralInfo, Radix},
};

pub const NORMAL: IntLiteralInfo = IntLiteralInfo {
    radix: Radix::Decimal,
    unsigned: false,
};

#[test]
fn test_expr_parser() {
    let provider = Provider::new(Src::None, "5 + (4 * x) - f(!y, 1, [2, 3], a: 5, b: 10)");

    let mut parser = Parser::from_provider(&provider);
    let expr = parser.consume_expr_until_end();
    let span = |start, end| Span::new(Src::None, start, end);

    assert_eq!(
        expr,
        Ok(Spanned(
            Expr::BinaryOp {
                left: Box::new(Spanned(
                    Expr::BinaryOp {
                        left: Box::new(Spanned(
                            Expr::Atom(Atom::Int("5".to_string(), NORMAL)),
                            span(0, 1),
                        )),
                        op: Spanned(BinaryOp::Add, span(2, 4),),
                        right: Box::new(Spanned(
                            Expr::BinaryOp {
                                left: Box::new(Spanned(
                                    Expr::Atom(Atom::Int("4".to_string(), NORMAL)),
                                    span(5, 6),
                                )),
                                op: Spanned(BinaryOp::Mul, span(7, 8),),
                                right: Box::new(Spanned(
                                    Expr::Ident(Spanned("x".to_string(), span(9, 10)), None),
                                    span(9, 10),
                                )),
                            },
                            span(5, 10),
                        )),
                    },
                    span(0, 10),
                )),
                op: Spanned(BinaryOp::Sub, span(12, 14),),
                right: Box::new(Spanned(
                    Expr::Call {
                        func: Box::new(Spanned(
                            Expr::Ident(Spanned("f".to_string(), span(14, 15)), None),
                            span(14, 15),
                        )),
                        args: vec![
                            Spanned(
                                Expr::UnaryOp {
                                    op: Spanned(UnaryOp::Not, span(16, 17)),
                                    expr: Box::new(Spanned(
                                        Expr::Ident(Spanned("y".to_string(), span(17, 18)), None),
                                        span(17, 18),
                                    )),
                                },
                                span(16, 18),
                            ),
                            Spanned(Expr::Atom(Atom::Int("1".to_string(), NORMAL)), span(20, 21),),
                            Spanned(
                                Expr::Array(vec![
                                    Spanned(
                                        Expr::Atom(Atom::Int("2".to_string(), NORMAL)),
                                        span(24, 25),
                                    ),
                                    Spanned(
                                        Expr::Atom(Atom::Int("3".to_string(), NORMAL)),
                                        span(27, 28),
                                    ),
                                ],),
                                span(23, 29),
                            ),
                        ],
                        kwargs: vec![
                            (
                                "a".to_string(),
                                Spanned(
                                    Expr::Atom(Atom::Int("5".to_string(), NORMAL)),
                                    span(34, 35),
                                ),
                            ),
                            (
                                "b".to_string(),
                                Spanned(
                                    Expr::Atom(Atom::Int("10".to_string(), NORMAL)),
                                    span(40, 42),
                                ),
                            ),
                        ],
                    },
                    span(14, 43),
                )),
            },
            span(0, 43),
        ))
    )
}

#[test]
fn test_body_parser() {
    let provider = Provider::new(Src::None, "let x = if y then 2 else 3;");
    let reader = grammar::token::TokenReader::new(&provider);
    println!("{:#?}", reader.collect::<Vec<_>>());

    let mut parser = Parser::from_provider(&provider);
    let nodes = parser.consume_body_until_end();

    if let Ok(ref nodes) = nodes {
        for node in nodes {
            println!("{node}");
        }
    }
    println!("{nodes:#?}");
}
