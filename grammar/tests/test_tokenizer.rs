use grammar::{Radix, Span, Spanned, StringLiteralFlags, Token, TokenInfo, TokenReader};

pub struct WhitespaceDiscarder<'a> {
    inner: TokenReader<'a>,
}

impl Iterator for WhitespaceDiscarder<'_> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.inner.next() {
                Some(Ok(Spanned(TokenInfo::Whitespace, _))) => continue,
                other => return other.map(|r| r.unwrap()),
            }
        }
    }
}

macro_rules! assert_tokens {
    ($tokens:expr, $($info:expr => $start:literal .. $end:literal),* $(,)?) => {
        let mut tokens = WhitespaceDiscarder { inner: TokenReader::new($tokens) };
        $(
            assert_eq!(tokens.next(), Some(Spanned($info, Span::new($start, $end))));
        )*
        assert_eq!(tokens.next(), None);
    };
}

#[test]
fn test_tokenizer_simple() {
    assert_tokens! {
        "func a() -> b { 1 + (2 * 3) }",
        TokenInfo::Ident("func".into()) => 0..4,
        TokenInfo::Ident("a".into()) => 5..6,
        TokenInfo::LeftParen => 6..7,
        TokenInfo::RightParen => 7..8,
        TokenInfo::Minus => 9..10,
        TokenInfo::Gt => 10..11,
        TokenInfo::Ident("b".into()) => 12..13,
        TokenInfo::LeftBrace => 14..15,
        TokenInfo::IntLiteral("1".into(), Radix::Decimal) => 16..17,
        TokenInfo::Plus => 18..19,
        TokenInfo::LeftParen => 20..21,
        TokenInfo::IntLiteral("2".into(), Radix::Decimal) => 21..22,
        TokenInfo::Asterisk => 23..24,
        TokenInfo::IntLiteral("3".into(), Radix::Decimal) => 25..26,
        TokenInfo::RightParen => 26..27,
        TokenInfo::RightBrace => 28..29,
    }
}

#[test]
fn test_tokenizer_integer_radix() {
    assert_tokens! {
        "0xAC + 0o123 + 0b1010",
        TokenInfo::IntLiteral("AC".into(), Radix::Hexadecimal) => 0..4,
        TokenInfo::Plus => 5..6,
        TokenInfo::IntLiteral("123".into(), Radix::Octal) => 7..12,
        TokenInfo::Plus => 13..14,
        TokenInfo::IntLiteral("1010".into(), Radix::Binary) => 15..21,
    }
}

#[test]
fn test_tokenizer_strings() {
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
        TokenInfo::StringLiteral("single-quoted string".into(), StringLiteralFlags(0), Span::new(14, 34)) => 13..35,
        TokenInfo::StringLiteral("double-quoted string".into(), StringLiteralFlags(0), Span::new(49, 69)) => 48..70,
        TokenInfo::StringLiteral("'nested' string".into(), StringLiteralFlags(0), Span::new(84, 99)) => 83..100,
        TokenInfo::StringLiteral("\\\"escaped quotes\\\"".into(), StringLiteralFlags(0), Span::new(114, 132)) => 113..133,
        TokenInfo::StringLiteral("escaped backslash \\\\".into(), StringLiteralFlags(0), Span::new(147, 167)) => 146..168,
        TokenInfo::StringLiteral("multi-line string".into(), StringLiteralFlags(0), Span::new(183, 200)) => 181..202,
        TokenInfo::StringLiteral("raw string".into(), StringLiteralFlags(1), Span::new(217, 227)) => 215..228,
        TokenInfo::StringLiteral("raw \\ string".into(), StringLiteralFlags(1), Span::new(243, 255)) => 241..256,
        TokenInfo::StringLiteral("raw multi-line string".into(), StringLiteralFlags(1), Span::new(272, 293)) => 269..295,
        TokenInfo::StringLiteral("multi-line \"quotes inside\" string 123".into(), StringLiteralFlags(0), Span::new(310, 347)) => 308..349,
        TokenInfo::StringLiteral("multiple #\"hashes\"#".into(), StringLiteralFlags(0), Span::new(366, 385)) => 362..389,
        TokenInfo::StringLiteral("interpolated string".into(), StringLiteralFlags(2), Span::new(404, 423)) => 402..424,
        TokenInfo::StringLiteral("interpolated multi-line string".into(), StringLiteralFlags(2), Span::new(440, 470)) => 437..472,
        TokenInfo::StringLiteral("raw interpolated string".into(), StringLiteralFlags(3), Span::new(488, 511)) => 485..512,
        TokenInfo::StringLiteral("raw interpolated multi-line string".into(), StringLiteralFlags(3), Span::new(529, 563)) => 525..565,
    }
}

#[test]
fn test_tokenizer_float_disambiguation() {
    assert_tokens! {
        "0.5 + 1.0e5 + 1.e5 + 1..2.0 + 0.0..1.0 + 1...2 + 0..1. + 1.0e5.test",
        // 0.5 -> (0.5)
        TokenInfo::FloatLiteral("0.5".into()) => 0..3,
        TokenInfo::Plus => 4..5,
        // 1.0e5 -> (1.0e5)
        TokenInfo::FloatLiteral("1.0e5".into()) => 6..11,
        TokenInfo::Plus => 12..13,
        // 1.e5 -> (1).(e5)
        TokenInfo::IntLiteral("1".into(), Radix::Decimal) => 14..15,
        TokenInfo::Dot => 15..16,
        TokenInfo::Ident("e5".into()) => 16..18,
        TokenInfo::Plus => 19..20,
        // 1..2.0 -> (1)..(2.0)
        TokenInfo::IntLiteral("1".into(), Radix::Decimal) => 21..22,
        TokenInfo::Dot => 22..23,
        TokenInfo::Dot => 23..24,
        TokenInfo::FloatLiteral("2.0".into()) => 24..27,
        TokenInfo::Plus => 28..29,
        // 0.0..1.0 -> (0.0)..(1.0)
        TokenInfo::FloatLiteral("0.0".into()) => 30..33,
        TokenInfo::Dot => 33..34,
        TokenInfo::Dot => 34..35,
        TokenInfo::FloatLiteral("1.0".into()) => 35..38,
        TokenInfo::Plus => 39..40,
        // 1...2 -> (1) (.) (.) (.) (2)
        TokenInfo::IntLiteral("1".into(), Radix::Decimal) => 41..42,
        TokenInfo::Dot => 42..43,
        TokenInfo::Dot => 43..44,
        TokenInfo::Dot => 44..45,
        TokenInfo::IntLiteral("2".into(), Radix::Decimal) => 45..46,
        TokenInfo::Plus => 47..48,
        // 0..1. -> (0)..(1.)
        TokenInfo::IntLiteral("0".into(), Radix::Decimal) => 49..50,
        TokenInfo::Dot => 50..51,
        TokenInfo::Dot => 51..52,
        TokenInfo::FloatLiteral("1".into()) => 52..54,
        TokenInfo::Plus => 55..56,
        // 1.0e5.test -> (1.0e5).(test)
        TokenInfo::FloatLiteral("1.0e5".into()) => 57..62,
        TokenInfo::Dot => 62..63,
        TokenInfo::Ident("test".into()) => 63..67,
    }
}
