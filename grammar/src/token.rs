use super::Span;
use std::str::Chars;
use unicode_xid::UnicodeXID;

/// Bitflags representing extra flags about a string (i.e. interpolated, raw).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct StringLiteralFlags(pub u8);

impl StringLiteralFlags {
    /// The string is a raw string - do not unescape any characters.
    pub const RAW: Self = Self(1 << 0);
    /// The string is an interpolated string.
    pub const INTERPOLATED: Self = Self(1 << 1);

    /// The bitflags represented as bits.
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// If the first bit is set.
    pub const fn is_raw(self) -> bool {
        self.bits() & Self::RAW.bits() != 0
    }

    /// If the second bit is set.
    pub const fn is_interpolated(self) -> bool {
        self.bits() & Self::INTERPOLATED.bits() != 0
    }
}

impl const std::ops::BitOr for StringLiteralFlags {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        Self(self.bits() | other.bits())
    }
}

impl std::ops::BitOrAssign for StringLiteralFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

/// Radix of an integer literal.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum Radix {
    /// Normal, base 10 integer.
    #[default]
    Decimal = 10,
    /// Represented in hexadecimal: 0x1a3d.
    Hexadecimal = 16,
    /// Represented in octal: 0o1234.
    Octal = 8,
    /// Represented in binary: 0b1010.
    Binary = 2,
}

/// Represents information about a lexical token in the source code.
#[derive(Clone, Debug, PartialEq, Eq)]
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
    StringLiteral(String, StringLiteralFlags, Span),
    /// An integer literal, such as 123, 0, or 0x123.
    IntLiteral(String, Radix),
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
    Dot,
    /// Less than, `<`.
    Lt,
    /// Greater than, `>`.
    Gt,
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
#[derive(Clone, Debug, PartialEq, Eq)]
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

/// Information about a tokenization error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// Unexpected character.
    UnexpectedCharacter(char),
}

/// An error that occured during tokenization.
#[derive(Clone, Debug)]
pub struct Error {
    /// The span of the error in the source code.
    pub span: Span,
    /// The kind of error this is.
    pub kind: ErrorKind,
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
#[derive(Clone)]
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

// Modified function to allow underscores
fn is_ident_start(c: char) -> bool {
    c == '_' || c.is_xid_start()
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
    fn consume_string_content(
        &mut self,
        hashes: u8,
        target: char,
        raw: bool,
    ) -> Option<(String, Span)> {
        let mut content = String::new();
        let start = self.cursor.pos();

        while let Some(c) = self.cursor.advance() {
            if c == '\\' && !raw {
                content.push('\\');
                content.push(self.cursor.advance()?);
                continue;
            }

            if c == target {
                let end = self.cursor.pos() - 1;
                let mut ending = 0;
                // Check if hashes match starting hashes
                while self.cursor.peek() == Some('#') && ending < hashes {
                    ending += 1;
                    self.cursor.advance();
                }

                if ending != hashes {
                    // Re-add the quote and hashes to the contents of the string
                    content.push(target);
                    std::iter::repeat('#')
                        .take(ending as usize)
                        .for_each(|c| content.push(c));
                    continue;
                }

                return Some((content, Span::new(start, end)));
            }

            content.push(c);
        }

        None
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

        let start = self.cursor.pos();
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

        let (content, content_span) = match self.consume_string_content(hashes, quote, is_raw) {
            Some(content) => content,
            None => {
                self.cursor = original;
                return None;
            }
        };

        let mut flags = StringLiteralFlags::default();
        if is_raw {
            flags |= StringLiteralFlags::RAW;
        }
        if is_interpolated {
            flags |= StringLiteralFlags::INTERPOLATED;
        }

        Some(Token {
            info: TokenInfo::StringLiteral(content, flags, content_span),
            span: Span::new(start, self.pos()),
        })
    }

    /// Consumes a number (integer/float literal), returning None if no number was found.
    /// This assumes the cursor is before the first digit.
    fn consume_number(&mut self) -> Option<Token> {
        let next = self.cursor.peek()?;
        // For now, let's not allow float literals to start with . due to ambiguities. Allowing .
        // to come first will result in weird ambiguities during tokenizing, even though the syntax
        // is invalid. Take the range expression 1..2, should this be tokenized as (1) (..) (2) or
        // should it be tokenized as (1.) (.2)? Obviously, we prefer the first one, however if we
        // were to allow . to come first, the current implementation will actually output the second
        // result, which is invaid syntax.
        if !matches!(next, '0'..='9') {
            return None;
        }

        let start = self.cursor.pos();
        let mut content = String::new();

        macro_rules! advance_then {
            ($p:pat => $e:expr) => {{
                self.cursor.advance();
                self.cursor.advance();

                loop {
                    match self.cursor.peek() {
                        Some('_') => {
                            self.cursor.advance();
                        }
                        Some($p) => {
                            content.push(self.cursor.advance()?);
                        }
                        _ => break,
                    }
                }

                $e
            }};
        }

        let radix = if next == '0' {
            match self.cursor.peek_second()? {
                'x' | 'X' => advance_then!('0'..='9' | 'a'..='f' | 'A'..='F' => Radix::Hexadecimal),
                'o' | 'O' => advance_then!('0'..='7' => Radix::Octal),
                'b' | 'B' => advance_then!('0' | '1' => Radix::Binary),
                _ => Radix::Decimal,
            }
        } else {
            Radix::Decimal
        };

        let mut is_float = false;

        if radix == Radix::Decimal {
            macro_rules! consume_ignore_underscore {
                ($p:pat) => {{
                    while let Some(c @ $p) = self.cursor.peek() {
                        self.cursor.advance();
                        if c != '_' {
                            content.push(c);
                        }
                    }
                }};
            }

            let mut try_exponent = true;
            // Consume digits until the first .
            while let Some(c @ ('0'..='9' | '_' | '.')) = self.cursor.peek() {
                if c == '.' {
                    match self.cursor.peek_second() {
                        // After the ., only allow digits 0 through 9. Not even underscores, since
                        // that could be an identifier.
                        Some('0'..='9') => {
                            // Consume the . and any digits after it
                            is_float = true;
                            content.push(self.cursor.advance()?);
                            consume_ignore_underscore!('0'..='9' | '_');
                        }
                        // On the event of a consecutive dot, this is probably a range operator, so
                        // treat this as an integer. On the event of an identifier starter, this is
                        // probably an attribute access.
                        Some(c) if c == '.' || is_ident_start(c) => try_exponent = false,
                        // Otherwise, this is a float with a trailing dot, consume that dot and
                        // move on.
                        _ => {
                            self.cursor.advance();
                            is_float = true;
                            // Don't allow E notation since this could conflict with attribute
                            // access.
                            try_exponent = false;
                        }
                    }
                    break;
                }

                self.cursor.advance();
                if c != '_' {
                    content.push(c);
                }
            }

            // Attempt to consume an exponent if it is safe to do so
            if try_exponent {
                if let Some('e' | 'E') = self.cursor.peek() {
                    match self.cursor.peek_second() {
                        Some('+' | '-') => {
                            content.push(self.cursor.advance()?);
                            content.push(self.cursor.advance()?);
                        }
                        _ => {
                            content.push(self.cursor.advance()?);
                        }
                    }

                    consume_ignore_underscore!('0'..='9' | '_');
                    is_float = true;
                }
            }
        }

        Some(if is_float {
            Token {
                info: TokenInfo::FloatLiteral(content),
                span: Span::new(start, self.pos()),
            }
        } else {
            Token {
                info: TokenInfo::IntLiteral(content, radix),
                span: Span::new(start, self.pos()),
            }
        })
    }

    /// Consumes the next identifier and returns it as a token, but returns None if no identifier
    /// was found immediately after the cursor.
    fn consume_ident(&mut self) -> Option<Token> {
        if !is_ident_start(self.cursor.peek()?) {
            return None;
        }

        let start = self.cursor.pos();
        let mut content = String::from(self.cursor.advance()?);

        while let Some(c) = self.cursor.peek() {
            if c.is_xid_continue() {
                self.cursor.advance();
                content.push(c);
            } else {
                break;
            }
        }

        Some(Token {
            info: TokenInfo::Ident(content),
            span: Span::new(start, self.pos()),
        })
    }
}

impl Iterator for TokenReader<'_> {
    type Item = Result<Token, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        let start = self.pos();

        if self.consume_whitespace() {
            return Some(Ok(Token {
                info: TokenInfo::Whitespace,
                span: Span::new(start, self.pos()),
            }));
        }

        macro_rules! consider {
            ($e:expr) => {{
                if let Some(token) = $e {
                    return Some(Ok(token));
                }
            }};
            ($($e:expr),*) => {{
                $(consider!($e);)*
            }};
        }

        consider! {
            self.consume_string_literal(),
            self.consume_number(),
            self.consume_ident()
        }

        let info = match self.cursor.advance()? {
            '(' => TokenInfo::LeftParen,
            ')' => TokenInfo::RightParen,
            '[' => TokenInfo::LeftBracket,
            ']' => TokenInfo::RightBracket,
            '{' => TokenInfo::LeftBrace,
            '}' => TokenInfo::RightBrace,
            ',' => TokenInfo::Comma,
            ';' => TokenInfo::Semicolon,
            ':' => TokenInfo::Colon,
            '.' => TokenInfo::Dot,
            '<' => TokenInfo::Lt,
            '>' => TokenInfo::Gt,
            '=' => TokenInfo::Equals,
            '!' => TokenInfo::Not,
            '+' => TokenInfo::Plus,
            '-' => TokenInfo::Minus,
            '*' => TokenInfo::Asterisk,
            '/' => {
                return self
                    .consume_comment_or_divide()
                    .map(Ok)
                    .or_else(|| self.next())
            }
            '\\' => TokenInfo::Backslash,
            '^' => TokenInfo::Caret,
            '&' => TokenInfo::And,
            '|' => TokenInfo::Or,
            '%' => TokenInfo::Modulus,
            c => {
                return Some(Err(Error {
                    span: Span::new(start, self.pos()),
                    kind: ErrorKind::UnexpectedCharacter(c),
                }))
            }
        };

        Some(Ok(Token {
            info,
            span: Span::new(start, self.pos()),
        }))
    }
}
