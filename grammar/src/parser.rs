use super::{ast::Expr, Token, TokenReader, TokenizationError};
use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};

type TokenResult = std::result::Result<Token, TokenizationError>;
type Result<T> = std::result::Result<T, Error>;

/// Represents an error that occured during parsing.
#[derive(Clone, Debug)]
pub enum Error {
    /// An error occured during tokenization.
    Tokenization(TokenizationError),
    /// An unexpected token was encountered.
    UnexpectedToken(Token),
}

/// A token peeker that saves all token peeks.
#[derive(Clone)]
pub struct Peeker<I: Iterator + Clone>
where
    I::Item: Clone,
{
    /// The iterator that is being peeked.
    iter: I,
    /// A clone of the original iterator to peek tokens.
    peeker: I,
    /// The peeked tokens.
    peeked: VecDeque<I::Item>,
}

impl<I: Iterator + Clone> Peeker<I>
where
    I::Item: Clone,
{
    /// Creates a new peeker from the given iterator.
    pub fn new(iter: I) -> Self {
        Self {
            peeker: iter.clone(),
            peeked: VecDeque::new(),
            iter,
        }
    }

    /// Peeks the next token.
    pub fn peek(&mut self) -> Option<I::Item> {
        self.peeked.get(0).cloned().or_else(|| {
            let value = self.peeker.next()?;
            self.peeked.push_back(value.clone());
            Some(value)
        })
    }

    /// Peeks the nth token, where for n = 1 this is the same as peek().
    pub fn peek_nth(&mut self, n: usize) -> Option<I::Item> {
        if n == 1 {
            return self.peek();
        }
        if self.peeked.len() >= n {
            return self.peeked.get(n - 1).cloned();
        }
        for _ in 0..n - self.peeked.len() {
            self.peeked.push_back(self.peeker.next()?);
        }
        self.peeked.back().cloned()
    }
}

impl<I: Iterator + Clone> Iterator for Peeker<I>
where
    I::Item: Clone,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.peeked.pop_front().or_else(|| self.iter.next())
    }
}

impl<I: Iterator + Clone> Deref for Peeker<I>
where
    I::Item: Clone,
{
    type Target = I;

    fn deref(&self) -> &Self::Target {
        &self.iter
    }
}

impl<I: Iterator + Clone> DerefMut for Peeker<I>
where
    I::Item: Clone,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.iter
    }
}

/// Parses a token stream into an AST.
#[derive(Clone)]
pub struct Parser<I: Iterator<Item = TokenResult> + Clone> {
    tokens: Peeker<I>,
}

impl<I: Iterator<Item = TokenResult> + Clone> Parser<I> {
    /// Peeks at the next token.
    fn peek(&mut self) -> Option<Token> {
        self.tokens.clone().next().transpose().unwrap()
    }

    /// Peeks at the next nth token, where when n = 1, it is the same as peek().
    fn peek_nth(&mut self, n: usize) -> Option<Token> {
        self.tokens.clone().skip(n - 1).next().transpose().unwrap()
    }

    /// Consumes the next integer literal.
    pub fn consume_int_literal(&mut self) -> Option<Token> {
        todo!()
    }

    /// Parses and returns the next expression.
    pub fn consume_expr(&mut self) -> Result<Expr> {
        todo!()
    }
}

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
            tokens: Peeker::new(tokens.into_iter()),
        }
    }
}
