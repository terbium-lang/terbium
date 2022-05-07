use chumsky::prelude::*;

use std::fmt::Display;
use crate::Token::Operator;

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
            Self::Add | Self::Sub | Self::BitNot => true,
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
        f.write_str(match self {
            Self::String(s) => format!("\"{}\"", s),
            Self::ByteString(s) => format!("~\"{}\"", s),
            Self::RawString(s) => format!("r\"{}\"", s),
            Self::InterpolatedString(s) => format!("$\"{}\"", s),
        }
            .as_str()
        )
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Literal {
    String(StringLiteral),
    Integer(u128), // This can be unsigned since unary minus is parsed separate from Literal
    Float(f64),
}

impl Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(match self {
            Self::String(s) => s,
            Self::Integer(i) => i,
            Self::Float(f) => f,
        }
            .to_string()
            .as_str()
        )
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Bracket {
    Paren,   // ()
    Bracket, // []
    Brace,   // {}
    Angle,   // <>
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
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
        f.write_str(match self {
            Self::Invalid(c) => c,
            Self::Operator(o) => o,
            Self::Literal(l) => l,
            Self::Keyword(k) => k,
            Self::Identifier(s) => s,
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
            .to_string()
            .as_str()
        )
    }
}

macro_rules! escape_hex {
    ($c:expr, $radix:expr) => {{
        just($c).ignore_then(
            filter(|c: &char| c.is_digit(16))
                .repeated()
                .exactly($radix)
                .collect::<String>()
                .validate(|digits, span, emit| {
                    char::from_u32(u32::from_str_radix(&digits, $radix).unwrap())
                        .unwrap_or_else(|| {
                            emit(Simple::custom(span, format!(
                                "invalid unicode character {}",
                                digits,
                            )));
                            '\u{FFFD}' // unicode replacement character
                        })
                }),
        )
    }}
}

pub fn get_lexer() -> impl Parser<char, Vec<Token>, Error = Simple<String>> {
    let integer = text::int(10).map(Literal::Int)
        .labelled("integer literal");

    let float = text::int(10)
        .chain(just('.'))
        .chain::<char, _, _>(text::digits(10).or_not().flatten())
        .or(
            just('.')
                .chain::<char, _, _>(text::digits(10))
        )
        .collect::<String>()
        .map(Literal::Float)
        .labelled("float literal");

    let escape = just('\\').ignore_then(
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

    let string = just('"')
        .ignore_then(filter(|c: &char| *c != '\\' && *c != '"').or(escape).repeated())
        .then_ignore(just::<_, char, _>('"'))
        .or(
            just('\'')
            .ignore_then(filter(|c: &char| *c != '\\' && *c != '\'').or(escape).repeated())
            .then_ignore(just::<_, char, _>('\''))
        )
        .collect::<String>()
        .map(Literal::String)
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

    let comment = just("//")
        .then_ignore(none_of('\n').ignored().repeated())
        .padded()
        .or(just("/*").ignore_then(none_of("*/").ignored().repeated()))
        .ignored()
        .repeated();

    let symbol = choice((
        just('+').map(|_| Token::Operator(Operator::Add)),
        just('-').map(|_| Token::Operator(Operator::Sub)),
        just("**").map(|_| Token::Operator(Operator::Pow)),
        just('*').map(|_| Token::Operator(Operator::Mul)),
        just('/').map(|_| Token::Operator(Operator::Div)),
        just('%').map(|_| Token::Operator(Operator::Mod)),
        just('=').to(Token::Assign),
        just("==").map(|_| Token::Operator(Operator::Eq)),
        just("!=").map(|_| Token::Operator(Operator::Ne)),
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
        just("..").map(|_| Token::Operator(Operator::Range)),
    ));
}