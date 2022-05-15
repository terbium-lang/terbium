use super::Error;

use chumsky::prelude::*;
use chumsky::text::Character;

use std::fmt::Display;
use std::hash::Hash;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Operator {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    // Logical
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Or,
    And,
    Not,
    // Bitwise
    BitOr,
    BitXor,
    BitAnd,
    BitNot,
    // Programmatic
    Range,
}

impl Operator {
    pub fn supports_unary(&self) -> bool {
        match self {
            // TODO: &ident could retrieve memory address of the object
            Self::Add | Self::Sub | Self::Not | Self::BitNot => true,
            _ => false,
        }
    }

    pub fn supports_binary(&self) -> bool {
        match self {
            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Div
            | Self::Mod
            | Self::Pow
            | Self::Eq
            | Self::Ne
            | Self::Lt
            | Self::Le
            | Self::Gt
            | Self::Ge
            | Self::Or
            | Self::And
            | Self::Not
            | Self::BitOr
            | Self::BitXor
            | Self::BitAnd
            | Self::Range => true,
            _ => false,
        }
    }
}

impl Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Mod => "%",
            Self::Pow => "**",
            Self::Eq => "==",
            Self::Ne => "!=",
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Gt => ">",
            Self::Ge => ">=",
            Self::Or => "||",
            Self::And => "&&",
            Self::Not => "!",
            Self::BitOr => "|",
            Self::BitXor => "^",
            Self::BitAnd => "&",
            Self::BitNot => "~",
            Self::Range => "..",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum StringLiteral {
    String(String),
    ByteString(String),
    RawString(String),
    InterpolatedString(String),
}

impl Display for StringLiteral {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(
            match self {
                Self::String(s) => format!("\"{}\"", s),
                Self::ByteString(s) => format!("~\"{}\"", s),
                Self::RawString(s) => format!("r\"{}\"", s),
                Self::InterpolatedString(s) => format!("$\"{}\"", s),
            }
            .as_str(),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Literal {
    String(StringLiteral),
    Integer(u128), // This can be unsigned since unary minus is parsed separate from Literal
    Float(String), // Rust floats are not hashable, additionally we want to avoid as much floating point precision loss as possible
}

impl Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(
            match self {
                Self::String(s) => s.to_string(),
                Self::Integer(i) => i.to_string(),
                Self::Float(f) => f.clone(),
            }
            .as_str(),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Keyword {
    Func,
    Class,
    // Modules
    Require,
    Export,
    // Variables
    Let,
    Const,
    Immut,
    // Attributes
    Private,
    // Control flow
    If,
    Else,
    Match,
    For,
    In,
    While,
    Break,
    Continue,
    Return,
    With,
    Throws,
}

impl Display for Keyword {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(match self {
            Self::Func => "func",
            Self::Class => "class",
            Self::Require => "require",
            Self::Export => "export",
            Self::Let => "let",
            Self::Const => "const",
            Self::Immut => "immut",
            Self::Private => "private",
            Self::If => "if",
            Self::Else => "else",
            Self::Match => "match",
            Self::For => "for",
            Self::In => "in",
            Self::While => "while",
            Self::Break => "break",
            Self::Continue => "continue",
            Self::Return => "return",
            Self::With => "with",
            Self::Throws => "throws",
        })
    }
}

impl Keyword {
    pub fn is_soft(&self) -> bool {
        match self {
            Self::Func
            | Self::Class
            | Self::Let
            | Self::Const
            | Self::Immut
            | Self::If
            | Self::Else
            | Self::For
            | Self::In
            | Self::While
            | Self::Break
            | Self::Continue
            | Self::Return
            | Self::With => false,
            _ => true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Bracket {
    Paren,   // ()
    Bracket, // []
    Brace,   // {}
    Angle,   // <>
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Token {
    Invalid(char),
    Operator(Operator),
    Literal(Literal),
    Keyword(Keyword),
    Identifier(String),
    StartBracket(Bracket),
    EndBracket(Bracket),
    Comma,
    Dot,
    Cast, // ::
    Question,
    Semicolon,
    Assign, // =
}

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let s: String;

        f.write_str(
            match self {
                Self::Invalid(c) => {
                    s = c.to_string();
                    s.as_str()
                }
                Self::Operator(o) => {
                    s = o.to_string();
                    s.as_str()
                }
                Self::Literal(l) => {
                    s = l.to_string();
                    s.as_str()
                }
                Self::Keyword(k) => {
                    s = k.to_string();
                    s.as_str()
                }
                Self::Identifier(s) => s.as_str(),
                Self::StartBracket(b) => match b {
                    Bracket::Paren => "(",
                    Bracket::Bracket => "[",
                    Bracket::Brace => "{",
                    Bracket::Angle => "<",
                },
                Self::EndBracket(b) => match b {
                    Bracket::Paren => ")",
                    Bracket::Bracket => "]",
                    Bracket::Brace => "}",
                    Bracket::Angle => ">",
                },
                Self::Comma => ",",
                Self::Dot => ".",
                Self::Cast => "::",
                Self::Question => "?",
                Self::Semicolon => ";",
                Self::Assign => "=",
            }
            .clone(),
        )
    }
}

macro_rules! escape_hex {
    ($c:expr, $l:expr) => {{
        just($c).ignore_then(
            filter(|c: &char| c.is_digit(16))
                .repeated()
                .exactly($l)
                .collect::<String>()
                .validate(|digits, span, emit| {
                    char::from_u32(u32::from_str_radix(&digits, 16).unwrap()).unwrap_or_else(|| {
                        emit(Error::custom(
                            span,
                            format!("invalid unicode character \\u{}", digits),
                        ));
                        '\u{FFFD}' // unicode replacement character
                    })
                }),
        )
    }};
}

pub fn get_lexer() -> impl Parser<char, Vec<Token>, Error = Error> {
    let integer = text::int::<_, Error>(10)
        .from_str::<u128>()
        .unwrapped()
        .map(Literal::Integer)
        .map(Token::Literal)
        .labelled("integer literal");

    let float = text::int::<_, Error>(10)
        .chain::<char, _, _>(just('.').chain(text::digits(10)).or_not().flatten())
        .or(just('.').chain::<char, _, _>(text::digits(10)))
        .collect::<String>()
        .map(Literal::Float)
        .map(Token::Literal)
        .labelled("float literal");

    let escape = just::<_, _, Error>('\\')
        .ignore_then(
            just('\\')
                .or(just('"'))
                .or(just('\''))
                .or(just('b').to('\x08'))
                .or(just('f').to('\x0C'))
                .or(just('n').to('\n'))
                .or(just('r').to('\r'))
                .or(just('t').to('\t'))
                .or(escape_hex!('x', 2))
                .or(escape_hex!('u', 4))
                .or(escape_hex!('U', 8)),
        )
        .labelled("escape sequence");

    let string = just::<_, _, Error>('"')
        .ignore_then(
            filter(|c: &char| *c != '\\' && *c != '"')
                .or(escape)
                .repeated(),
        )
        .then_ignore(just::<_, char, _>('"'))
        .or(just('\'')
            .ignore_then(
                filter(|c: &char| *c != '\\' && *c != '\'')
                    .or(escape)
                    .repeated(),
            )
            .then_ignore(just::<_, char, _>('\'')))
        .collect::<String>()
        .map(|s| Token::Literal(Literal::String(StringLiteral::String(s))))
        .labelled("string literal");

    let ident_or_keyword = text::ident().map(|s: String| match s.as_str() {
        "func" => Token::Keyword(Keyword::Func),
        "class" => Token::Keyword(Keyword::Class),
        "require" => Token::Keyword(Keyword::Require),
        "export" => Token::Keyword(Keyword::Export),
        "let" => Token::Keyword(Keyword::Let),
        "const" => Token::Keyword(Keyword::Const),
        "immut" => Token::Keyword(Keyword::Immut),
        "private" => Token::Keyword(Keyword::Private),
        "if" => Token::Keyword(Keyword::If),
        "else" => Token::Keyword(Keyword::Else),
        "match" => Token::Keyword(Keyword::Match),
        "for" => Token::Keyword(Keyword::For),
        "in" => Token::Keyword(Keyword::In),
        "while" => Token::Keyword(Keyword::While),
        "break" => Token::Keyword(Keyword::Break),
        "continue" => Token::Keyword(Keyword::Continue),
        "return" => Token::Keyword(Keyword::Return),
        "with" => Token::Keyword(Keyword::With),
        "throws" => Token::Keyword(Keyword::Throws),
        _ => Token::Identifier(s),
    });

    let single_line = just::<_, _, Error>("//")
        .then(take_until(text::newline()))
        .ignored();

    let multi_line = just::<_, _, Error>("/*")
        .then(take_until(just("*/")))
        .ignored();

    let comment = single_line.or(multi_line).or_not();

    let symbol = choice::<_, Error>((
        just(',').to(Token::Comma),
        just(';').to(Token::Semicolon),
        just('?').to(Token::Question),
        just("::").to(Token::Cast),
        just("..").map(|_| Token::Operator(Operator::Range)),
        just('.').to(Token::Dot),
        just('+').map(|_| Token::Operator(Operator::Add)),
        just('-').map(|_| Token::Operator(Operator::Sub)),
        just("**").map(|_| Token::Operator(Operator::Pow)),
        just('*').map(|_| Token::Operator(Operator::Mul)),
        just('/').map(|_| Token::Operator(Operator::Div)),
        just('%').map(|_| Token::Operator(Operator::Mod)),
    ))
    .or(choice((
        // Weird split-off as chumsky only supports choices up to 26-length tuples. Maybe it would be better to separate them based off of category
        just("==").map(|_| Token::Operator(Operator::Eq)),
        just("!=").map(|_| Token::Operator(Operator::Ne)),
        just('!').map(|_| Token::Operator(Operator::Not)), // Conflicts with !=
        just('=').to(Token::Assign),                       // Conflicts with ==
        just("<=").map(|_| Token::Operator(Operator::Le)),
        just(">=").map(|_| Token::Operator(Operator::Ge)),
        just('<').map(|_| Token::Operator(Operator::Lt)),
        just('>').map(|_| Token::Operator(Operator::Gt)),
        just("||").map(|_| Token::Operator(Operator::Or)),
        just("&&").map(|_| Token::Operator(Operator::And)),
        just('|').map(|_| Token::Operator(Operator::BitOr)),
        just('^').map(|_| Token::Operator(Operator::BitXor)),
        just('&').map(|_| Token::Operator(Operator::BitAnd)),
        just('~').map(|_| Token::Operator(Operator::BitNot)),
    )));

    let brackets = choice::<_, Error>((
        just('(').map(|_| Token::StartBracket(Bracket::Paren)),
        just('[').map(|_| Token::StartBracket(Bracket::Bracket)),
        just('{').map(|_| Token::StartBracket(Bracket::Brace)),
        just(')').map(|_| Token::EndBracket(Bracket::Paren)),
        just(']').map(|_| Token::EndBracket(Bracket::Bracket)),
        just('}').map(|_| Token::EndBracket(Bracket::Brace)),
    ));

    choice::<_, Error>((symbol, brackets, ident_or_keyword, string, integer, float))
        .or(any().map(Token::Invalid).validate(|token, span, emit| {
            emit(Error::unexpected_token(span, &token));
            token
        }))
        // .map_with_span(move |token, span| (token, span)) Could be useful for debugging
        .padded()
        .recover_with(skip_then_retry_until([]))
        .padded_by(comment)
        .repeated()
        .padded()
        .then_ignore(end())
}

#[cfg(test)]
mod tests {
    use crate::token::*;
    use chumsky::Parser;

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

        let (tokens, errors) = get_lexer().parse_recovery(raw);

        assert_eq!(
            tokens,
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
}
