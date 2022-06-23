use terbium::grammar::token::*;
use terbium::grammar::{Span, Source};

#[test]
fn test_lexer() {
    let raw = r#"
        func main() {
            std.println("Hello, world!");
            if 1 + 2 == 3 {
                // Here is a comment
            }
        }
    "#;

    let (tokens, errors) = get_lexer().parse_recovery(
        Stream::<_, Span, _>::from_iter(
            Span::single(Source::default(), raw.chars().count()),
            raw.chars().enumerate().map(|(i, c)| (c, Span::single(Source::default(), i))),
        ),
    );

    assert_eq!(
        tokens.map(|t| t
            .into_iter()
            .map(|t| t.0)
            .collect::<Vec<_>>()
        ),
        Some(vec![
            Token::Keyword(Keyword::Func),
            Token::Identifier("main".to_string()),
            Token::StartBracket(Bracket::Paren),
            Token::EndBracket(Bracket::Paren),
            Token::StartBracket(Bracket::Brace),
            Token::Identifier("std".to_string()),
            Token::Dot,
            Token::Identifier("println".to_string()),
            Token::StartBracket(Bracket::Paren),
            Token::Literal(Literal::String(StringLiteral::String(
                "Hello, world!".to_string()
            ))),
            Token::EndBracket(Bracket::Paren),
            Token::Semicolon,
            Token::Keyword(Keyword::If),
            Token::Literal(Literal::Integer(1)),
            Token::Operator(Operator::Add),
            Token::Literal(Literal::Integer(2)),
            Token::Operator(Operator::Eq),
            Token::Literal(Literal::Integer(3)),
            Token::StartBracket(Bracket::Brace),
            Token::EndBracket(Bracket::Brace),
            Token::EndBracket(Bracket::Brace),
        ]),
    );
    assert_eq!(errors, vec![]);
}
