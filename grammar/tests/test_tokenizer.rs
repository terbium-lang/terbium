use grammar::{
    span::{Provider, Span, Spanned, Src},
    token::{IntLiteralInfo, Radix, StringLiteralFlags, Token, TokenReader},
};

pub const NORMAL: IntLiteralInfo = IntLiteralInfo {
    radix: Radix::Decimal,
    unsigned: false,
};

pub struct WhitespaceDiscarder<'a> {
    inner: TokenReader<'a>,
}

impl Iterator for WhitespaceDiscarder<'_> {
    type Item = Spanned<Token>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.inner.next() {
                Some(Spanned(Token::Whitespace, _)) => continue,
                other => return other,
            }
        }
    }
}

macro_rules! assert_tokens {
    ($tokens:expr, $($info:expr => $start:literal .. $end:literal),* $(,)?) => {
        let provider = Provider(Src::None, ::std::borrow::Cow::from($tokens));
        let mut tokens = WhitespaceDiscarder { inner: TokenReader::new(&provider) };
        $(
            assert_eq!(tokens.next(), Some(Spanned($info, Span::new(Src::None, $start, $end))));
        )*
        assert_eq!(tokens.next(), None);
    };
}

#[test]
fn test_tokenizer_simple() {
    assert_tokens! {
        "func a() -> b { 1 + (2 * 3) }",
        Token::Ident("func".into(), false) => 0..4,
        Token::Ident("a".into(), false) => 5..6,
        Token::LeftParen => 6..7,
        Token::RightParen => 7..8,
        Token::Minus => 9..10,
        Token::Gt => 10..11,
        Token::Ident("b".into(), false) => 12..13,
        Token::LeftBrace => 14..15,
        Token::IntLiteral("1".into(), NORMAL) => 16..17,
        Token::Plus => 18..19,
        Token::LeftParen => 20..21,
        Token::IntLiteral("2".into(), NORMAL) => 21..22,
        Token::Asterisk => 23..24,
        Token::IntLiteral("3".into(), NORMAL) => 25..26,
        Token::RightParen => 26..27,
        Token::RightBrace => 28..29,
    }
}

#[test]
fn test_tokenizer_integer_radix() {
    let radix = |radix| IntLiteralInfo {
        radix,
        unsigned: false,
    };

    assert_tokens! {
        "0xAC + 0o123 + 0b1010",
        Token::IntLiteral("AC".into(), radix(Radix::Hexadecimal)) => 0..4,
        Token::Plus => 5..6,
        Token::IntLiteral("123".into(), radix(Radix::Octal)) => 7..12,
        Token::Plus => 13..14,
        Token::IntLiteral("1010".into(), radix(Radix::Binary)) => 15..21,
    }
}

#[test]
fn test_tokenizer_strings() {
    let span = |start, end| Span::new(Src::None, start, end);

    assert_tokens! {
        r####"
            'single-quoted string'
            "double-quoted string"
            "'nested' string"
            "\"escaped quotes\""
            "escaped backslash \\"
            #"multi-line string"#
            ~"raw string"
            ~"raw \ string"
            ~#"raw multi-line string"#
            #"multi-line "quotes inside" string 123"#
            ###"multiple #"hashes"#"###
            $"interpolated string"
            $#"interpolated multi-line string"#
            ~$"raw interpolated string"
            ~$#"raw interpolated multi-line string"#
        "####,
        Token::StringLiteral("single-quoted string".into(), StringLiteralFlags(0), span(14, 34)) => 13..35,
        Token::StringLiteral("double-quoted string".into(), StringLiteralFlags(0), span(49, 69)) => 48..70,
        Token::StringLiteral("'nested' string".into(), StringLiteralFlags(0), span(84, 99)) => 83..100,
        Token::StringLiteral("\\\"escaped quotes\\\"".into(), StringLiteralFlags(0), span(114, 132)) => 113..133,
        Token::StringLiteral("escaped backslash \\\\".into(), StringLiteralFlags(0), span(147, 167)) => 146..168,
        Token::StringLiteral("multi-line string".into(), StringLiteralFlags(0), span(183, 200)) => 181..202,
        Token::StringLiteral("raw string".into(), StringLiteralFlags(1), span(217, 227)) => 215..228,
        Token::StringLiteral("raw \\ string".into(), StringLiteralFlags(1), span(243, 255)) => 241..256,
        Token::StringLiteral("raw multi-line string".into(), StringLiteralFlags(1), span(272, 293)) => 269..295,
        Token::StringLiteral("multi-line \"quotes inside\" string 123".into(), StringLiteralFlags(0), span(310, 347)) => 308..349,
        Token::StringLiteral("multiple #\"hashes\"#".into(), StringLiteralFlags(0), span(366, 385)) => 362..389,
        Token::StringLiteral("interpolated string".into(), StringLiteralFlags(2), span(404, 423)) => 402..424,
        Token::StringLiteral("interpolated multi-line string".into(), StringLiteralFlags(2), span(440, 470)) => 437..472,
        Token::StringLiteral("raw interpolated string".into(), StringLiteralFlags(3), span(488, 511)) => 485..512,
        Token::StringLiteral("raw interpolated multi-line string".into(), StringLiteralFlags(3), span(529, 563)) => 525..565,
    }
}

#[test]
fn test_tokenizer_float_disambiguation() {
    assert_tokens! {
        "0.5 + 1.0e5 + 1.e5 + 1..2.0 + 0.0..1.0 + 1...2 + 0..1. + 1.0e5.test",
        // 0.5 -> (0.5)
        Token::FloatLiteral("0.5".into()) => 0..3,
        Token::Plus => 4..5,
        // 1.0e5 -> (1.0e5)
        Token::FloatLiteral("1.0e5".into()) => 6..11,
        Token::Plus => 12..13,
        // 1.e5 -> (1).(e5)
        Token::IntLiteral("1".into(), NORMAL) => 14..15,
        Token::Dot => 15..16,
        Token::Ident("e5".into(), false) => 16..18,
        Token::Plus => 19..20,
        // 1..2.0 -> (1)..(2.0)
        Token::IntLiteral("1".into(), NORMAL) => 21..22,
        Token::Dot => 22..23,
        Token::Dot => 23..24,
        Token::FloatLiteral("2.0".into()) => 24..27,
        Token::Plus => 28..29,
        // 0.0..1.0 -> (0.0)..(1.0)
        Token::FloatLiteral("0.0".into()) => 30..33,
        Token::Dot => 33..34,
        Token::Dot => 34..35,
        Token::FloatLiteral("1.0".into()) => 35..38,
        Token::Plus => 39..40,
        // 1...2 -> (1) (.) (.) (.) (2)
        Token::IntLiteral("1".into(), NORMAL) => 41..42,
        Token::Dot => 42..43,
        Token::Dot => 43..44,
        Token::Dot => 44..45,
        Token::IntLiteral("2".into(), NORMAL) => 45..46,
        Token::Plus => 47..48,
        // 0..1. -> (0)..(1.)
        Token::IntLiteral("0".into(), NORMAL) => 49..50,
        Token::Dot => 50..51,
        Token::Dot => 51..52,
        Token::FloatLiteral("1".into()) => 52..54,
        Token::Plus => 55..56,
        // 1.0e5.test -> (1.0e5).(test)
        Token::FloatLiteral("1.0e5".into()) => 57..62,
        Token::Dot => 62..63,
        Token::Ident("test".into(), false) => 63..67,
    }
}
