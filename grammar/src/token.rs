use super::Span;
use std::str::Chars;

#[derive(Copy, Clone, Debug)]
pub struct StringLiteralFlags(pub u8);

impl StringLiteralFlags {
    pub const RAW: Self = Self(1 << 0);
    pub const INTERPOLATED: Self = Self(1 << 1);
    pub const RAW_INTERPOLATED: Self = Self(Self::RAW.bits() | Self::INTERPOLATED.bits());

    pub const fn bits(self) -> u8 {
        self.0
    }

    pub const fn is_raw(self) -> bool {
        self.bits() & Self::RAW.bits() != 0
    }

    pub const fn is_interpolated(self) -> bool {
        self.bits() & Self::INTERPOLATED.bits() != 0
    }
}

/// Represents information about a lexical token in the source code.
#[derive(Clone, Debug)]
pub enum TokenInfo {
    /// Any sequence of whitespace.
    Whitespace,
    /// A documentation comment. These are comments that begin with `///` or `//!`.
    DocComment {
        /// The trimmed contents of the comment.
        content: String,
        /// `true` if the comment begins with `//!`.
        is_inner: bool,
    },
    /// An identifier, such as "x". This includes keywords, since some keywords can also be used
    /// as identifiers.
    Ident(String),
    /// A string literal, such as "hello". This only includes the raw contents of the string,
    /// before any escape sequences are processed or any interpolation is done.
    ///
    /// This accounts for all strings, i.e. "normal", ~"raw", #"multiline"#, $"interpolated",
    /// and even ~$#"combinations of them all"#.
    StringLiteral(String, StringLiteralFlags),
    /// An integer literal, such as 123, 0, or 0x123.
    IntLiteral(String),
    /// A float literal, such as 1.23, 0.0, or 1e-10.
    FloatLiteral(String),
    /// Left parenthesis, `(`.
    LeftParen,
    /// Right parenthesis, `)`.
    RightParen,
    /// Left square bracket, `[`.
    LeftBracket,
    /// Right square bracket, `]`.
    RightBracket,
    /// Left curly brace, `{`.
    LeftBrace,
    /// Right curly brace, `}`.
    RightBrace,
    /// Comma, `,`.
    Comma,
    /// Semicolon, `;`.
    Semicolon,
    /// Colon, `:`.
    Colon,
    /// Period, `.`.
    Period,
    /// Less than, `<`.
    LessThan,
    /// Greater than, `>`.
    GreaterThan,
    /// Equals, `=`.
    Equals,
    /// Not, `!`.
    Not,
    /// Plus, `+`.
    Plus,
    /// Minus or hyphen, `-`.
    Minus,
    /// Asterisk, `*`.
    Asterisk,
    /// Forward slash, `/`, usually only used for division.
    Divide,
    /// Backslash, `\`.
    Backslash,
    /// Caret, `^`.
    Caret,
    /// And, `&`.
    And,
    /// Or, `|`.
    Or,
    /// Modulus, `%`.
    Modulus,
}

/// Represents a lexical token in the source code.
#[derive(Clone, Debug)]
pub struct Token {
    /// Information and classification of the token.
    pub info: TokenInfo,
    /// The span of the token in the source code.
    pub span: Span,
}

impl Token {
    /// Splits this token into a tuple (info, span).
    pub fn split(self) -> (TokenInfo, Span) {
        (self.info, self.span)
    }
}

/// A wrapped iterator over a `Char` iterator.
#[derive(Clone)]
struct Cursor<'a> {
    inner: Chars<'a>,
    pos: usize,
    len: usize,
}

impl<'a> Cursor<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            inner: source.chars(),
            pos: 0,
            len: source.len(),
        }
    }

    /// Returns the current position of the cursor.
    #[must_use]
    const fn pos(&self) -> usize {
        self.pos
    }

    /// Peeks at the next character in the iterator.
    #[must_use]
    fn peek(&mut self) -> Option<char> {
        // more efficient than `.nth(0)`
        self.inner.clone().next()
    }

    /// Peeks at the second character in the iterator.
    #[must_use]
    fn peek_second(&mut self) -> Option<char> {
        // more efficient than `.nth(1)`
        let mut iter = self.inner.clone();
        iter.next()?;
        iter.next()
    }

    /// Advances to the next character.
    fn advance(&mut self) -> Option<char> {
        self.pos += 1;
        self.inner.next()
    }

    /// Whether the cursor is at the end of the source.
    fn is_eof(&self) -> bool {
        self.pos >= self.len
    }
}

/// A reader over a source string that can be used to lex tokens.
pub struct TokenReader<'a> {
    cursor: Cursor<'a>,
}

// Copied from rustc-lexer
fn is_whitespace(c: char) -> bool {
    matches!(
        c,
        // Usual ASCII suspects
        '\u{0009}'   // \t
        | '\u{000A}' // \n
        | '\u{000B}' // vertical tab
        | '\u{000C}' // form feed
        | '\u{000D}' // \r
        | '\u{0020}' // space

        // NEXT LINE from latin1
        | '\u{0085}'

        // Bidi markers
        | '\u{200E}' // LEFT-TO-RIGHT MARK
        | '\u{200F}' // RIGHT-TO-LEFT MARK

        // Dedicated whitespace characters from Unicode
        | '\u{2028}' // LINE SEPARATOR
        | '\u{2029}' // PARAGRAPH SEPARATOR
    )
}

impl<'a> TokenReader<'a> {
    /// Creates a token reader over the given source string.
    pub fn new(source: &'a str) -> Self {
        Self {
            cursor: Cursor::new(source),
        }
    }

    /// Returns the current position of the cursor.
    #[must_use]
    pub const fn pos(&self) -> usize {
        self.cursor.pos()
    }

    /// Consumes whitespace until the next non-whitespace character, returning `true` if whitespace
    /// was encountered. This wil leave the cursor pointing at the whitespace character directly
    /// before the next non-whitespace character.
    fn consume_whitespace(&mut self) -> bool {
        let mut ws = false;

        while let Some(c) = self.cursor.peek() {
            if is_whitespace(c) {
                ws = true;
                self.cursor.advance();
                continue;
            }
            break;
        }
        ws
    }

    /// Puts the cursor right before the first character on the next line.
    fn discard_line(&mut self) {
        while let Some(c) = self.cursor.advance() {
            if c == '\n' {
                break;
            }
        }
    }

    /// Consumes the contents of a line. Used for documentation comments.
    fn consume_line(&mut self) -> String {
        let mut line = String::new();

        while let Some(c) = self.cursor.advance() {
            if c == '\n' {
                break;
            }
            line.push(c);
        }
        line
    }

    /// Discards characters until it reaches */. Currently this is not recursive and works similarly
    /// to strings.
    fn discard_block_comment(&mut self) {
        while let Some(c) = self.cursor.advance() {
            if c == '*' && self.cursor.peek() == Some('/') {
                self.cursor.advance();
                break;
            }
        }
    }

    /// Consumes a division token or a comment. Returns Some if a division token or doc comment was
    /// found, and None if a normal comment was found.
    fn consume_comment_or_divide(&mut self) -> Option<Token> {
        // Assume the cursor is on the first '/'
        let start = self.cursor.pos();

        match self.cursor.peek() {
            Some('/') => {
                self.cursor.advance();

                if let Some(c @ ('/' | '!')) = self.cursor.peek() {
                    self.cursor.advance();
                    let line = self.consume_line();
                    Some(Token {
                        info: TokenInfo::DocComment {
                            content: line,
                            is_inner: c == '!',
                        },
                        span: Span::new(start, self.cursor.pos()),
                    })
                } else {
                    self.discard_line();
                    None
                }
            }
            Some('*') => {
                self.cursor.advance();
                self.discard_block_comment();
                None
            }
            _ => Some(Token {
                info: TokenInfo::Divide,
                span: Span::single(start),
            }),
        }
    }

    /// Assuming that the cursor is on " or ', consume the raw contents of a string.
    /// If the string is invalid, return None.
    fn consume_string_content(&mut self, hashes: u8, target: char) -> Option<String> {

    }

    /// Possibly consumes a string literal. Returns None if no string was found.
    fn consume_string_literal(&mut self) -> Option<Token> {
        let next = self.cursor.peek()?;
        if !matches!(next, '"' | '\'' | '$' | '~' | '#') {
            return None;
        }

        let (is_raw, is_interpolated) = match (next, self.cursor.peek_second()?) {
            ('$', '~') | ('~', '$') => (true, true),
            ('~', _) => (true, false),
            ('$', _) => (false, true),
            _ => (false, false),
        };

        // Capture the current state so we can still recover later tokens.
        // Note that when propagating .advance()? or .peek()? we don't need to restore the cursor
        // since there is nothing else for the cursor to read nevertheless.
        let mut original = self.cursor.clone();
        // Advance the cursor to avoid infinite recursion
        original.advance();

        let start = self.cursor.pos() + 1;
        // Advance to hashes or quote
        if is_raw {
            self.cursor.advance();
        }
        if is_interpolated {
            self.cursor.advance();
        }

        let mut hashes = 0;
        while self.cursor.peek() == Some('#') {
            hashes += 1;
            self.cursor.advance();
        }

        let quote = self.cursor.advance()?;
        if !matches!(quote, '"' | '\'') {
            self.cursor = original;
            return None;
        }

        let content = match self.consume_string_content(hashes, quote) {
            Some(content) => content,
            None => {
                self.cursor = original;
                return None;
            }
        };

        let flags = StringLiteralFlags(0);
    }
}

impl Iterator for TokenReader<'_> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        let start = self.pos();

        if self.consume_whitespace() {
            return Some(Token {
                info: TokenInfo::Whitespace,
                span: Span::new(start, self.pos()),
            });
        }

        macro_rules! token {
            ($info:expr) => {{
                return Some(Token {
                    info: $info,
                    span: Span::new(start, self.pos()),
                });
            }};
        }

        match self.cursor.advance()? {
            '(' => token!(TokenInfo::LeftParen),
            ')' => token!(TokenInfo::RightParen),
            '[' => token!(TokenInfo::LeftBracket),
            ']' => token!(TokenInfo::RightBracket),
            '{' => token!(TokenInfo::LeftBrace),
            '}' => token!(TokenInfo::RightBrace),
            ',' => token!(TokenInfo::Comma),
            ';' => token!(TokenInfo::Semicolon),
            ':' => token!(TokenInfo::Colon),
            '.' => token!(TokenInfo::Period),
            '<' => token!(TokenInfo::LessThan),
            '>' => token!(TokenInfo::GreaterThan),
            '=' => token!(TokenInfo::Equals),
            '!' => token!(TokenInfo::Not),
            '+' => token!(TokenInfo::Plus),
            '-' => token!(TokenInfo::Minus),
            '*' => token!(TokenInfo::Asterisk),
            '/' => return self.consume_comment_or_divide().or_else(|| self.next()),
            '\\' => token!(TokenInfo::Backslash),
            '^' => token!(TokenInfo::Caret),
            '&' => token!(TokenInfo::And),
            '|' => token!(TokenInfo::Or),
            '%' => token!(TokenInfo::Modulus),
            '0'..='9' => {

            },
            'a'..='z' | 'A'..='Z' | '_' => {

            },
            _ => todo!(),
        }

        None
    }
}
