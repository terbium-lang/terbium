use super::{Token, TokenReader, TokenizationError};

type TokenResult = Result<Token, TokenizationError>;

/// Parses a token stream into an AST.
#[derive(Clone)]
pub struct Parser<I: Iterator<Item = TokenResult> + Clone> {
    tokens: I,
}

impl<I: Iterator<Item = TokenResult> + Clone> Parser<I> {}

impl<'a> Parser<TokenReader<'a>> {
    /// Creates a new parser from a string slice.
    #[must_use]
    pub fn from_str(source: &'a str) -> Self {
        Self::from(TokenReader::new(source))
    }
}

impl<I: IntoIterator<Item = TokenResult>> From<I> for Parser<I::IntoIter>
where
    I::IntoIter: Clone,
{
    fn from(tokens: I) -> Self {
        Self {
            tokens: tokens.into_iter(),
        }
    }
}
