use chumsky::prelude::*;

use std::fmt::Display;

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
        write!(f, match self {
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum StringLiteral {
    String(String),
    ByteString(String),
    RawString(String),
    InterpolatedString(String),
}

impl Display for StringLiteral {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, match self {
            Self::String(s) => format!("\"{}\"", s),
            Self::ByteString(s) => format!("~\"{}\"", s),
            Self::RawString(s) => format!("r\"{}\"", s),
            Self::InterpolatedString(s) => format!("$\"{}\"", s),
        })
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
        write!(f, match self {
            Self::String(s) => s,
            Self::Integer(i) => i,
            Self::Float(f) => f,
        })
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
        write!(f, match self {
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
}

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, match self {
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
        })
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
        .ignore_then(filter(|c| *c != '\\' && *c != '"').or(escape).repeated())
        .then_ignore(just('"'))
        .or(
            just('\'')
            .ignore_then(filter(|c| *c != '\\' && *c != '\'').or(escape).repeated())
            .then_ignore(just('\''))
        )
        .collect::<String>()
        .map(Literal::String)
        .labelled("string literal");
}