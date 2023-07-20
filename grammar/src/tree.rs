//! Delimited token tree parsing interface, used in metaprogramming via macros and decorators.

use crate::{
    ast::{Delimiter, TokenTree},
    token::Token,
};
use common::span::{Span, Spanned};

/// An error that occurs trying to parse a meta signature (macros, decorators, other token-trees).
pub struct MetaSignatureError {
    /// The entity that was expected.
    expected: (String, Option<Span>),
    /// The actual entity provided.
    actual: Spanned<String>,
}

pub trait ConsumerVec: Clone + Default + Sized {
    type Consumer;

    fn inner(&mut self) -> &mut Vec<(Option<String>, Self::Consumer)>;

    /// Creates a new consumer vec with the given consumers.
    fn new(f: impl FnOnce(&mut Self) -> &mut Self) -> Self {
        let mut slf = Self::default();
        f(&mut slf);
        slf
    }

    /// Chain a consumer to the consumer tree.
    fn then(&mut self, consumer: Self::Consumer) -> &mut Self {
        self.inner().push((None, consumer));
        self
    }

    /// Chain a named consumer to the consumer tree.
    fn then_named(&mut self, name: impl Into<String>, consumer: Self::Consumer) -> &mut Self {
        self.inner().push((Some(name.into()), consumer));
        self
    }
}

#[derive(Clone, Debug, Default)]
pub struct UnspannedConsumerVec(Vec<(Option<String>, Consumer<Self>)>);

impl ConsumerVec for UnspannedConsumerVec {
    type Consumer = Consumer<Self>;

    fn inner(&mut self) -> &mut Vec<(Option<String>, Self::Consumer)> {
        &mut self.0
    }
}

#[derive(Clone, Debug, Default)]
pub struct SpannedConsumerVec(Vec<(Option<String>, Spanned<Consumer<Self>>)>);

impl ConsumerVec for SpannedConsumerVec {
    type Consumer = Spanned<Consumer<Self>>;

    fn inner(&mut self) -> &mut Vec<(Option<String>, Self::Consumer)> {
        &mut self.0
    }
}

#[derive(Clone, Debug)]
pub struct TokenTreeCursor {
    pub tree: Vec<TokenTree>,
    pub cursor: usize,
}

impl TokenTreeCursor {
    pub fn new(tree: Vec<TokenTree>) -> Self {
        let mut cursor = Self { tree, cursor: 0 };
        cursor.consume_ws(); // discard any ws padding
        cursor
    }

    /// Returns the token at the cursor, but doesn't advance it.
    pub fn peek(&self) -> Option<&TokenTree> {
        self.tree.get(self.cursor)
    }

    /// Returns the token at the cursor, then advances the cursor.
    pub fn advance(&mut self) -> Option<&TokenTree> {
        self.tree.get(self.cursor).map(|res| {
            self.cursor += 1;
            res
        })
    }

    fn consume_ws(&mut self) -> bool {
        matches!(self.peek(), Some(TokenTree::Token(Token::Whitespace)))
            .then(|| self.advance())
            .is_some()
    }
}

/// A consumer of tokens.
#[derive(Clone, Debug)]
pub enum Consumer<C: ConsumerVec> {
    /// Consume just this token and then any amount of whitespace.
    Token(Token),
    /// Consume just this token, but ensure there is no whitespace after it.
    TokenNoWs(Token),
    /// Consume either a token or delimited token tree.
    TokenTree,
    /// Consume any identifier or raw identifier.
    Ident,
    /// Consume any raw path. (e.g. a, a.b, a.b.c)
    Path,
    /// Consume any literal (string literals, numeric litereals, boolean literals, void literal).
    Literal,
    /// Consume a string literal.
    StringLiteral,
    /// Consume an integer literal.
    IntLiteral,
    /// Consume a float literal.
    FloatLiteral,
    /// Consume a boolean literal.
    BoolLiteral,
    /// Consume a block expression.
    Block,
    /// Consume an expression.
    Expr,
    /// Consume a type expression.
    TypeExpr,
    /// Consume any item declaration, excluding decorators or doc comments.
    Item,
    /// Consume a single decorator or a single line of documentation.
    Decorator,
    /// Consume a visibility modifier.
    Visibility,
    /// Consume a match pattern.
    Pattern,
    /// Consume a token tree delimited by the given delimiter.
    Delimited(Delimiter, C),
    /// Try running the consumers, but don't consume anything if it fails.
    Maybe(C),
    /// Run the consumer as many times as possible, including zero times, optionally separated by the given token.
    ZeroOrMore(C, Option<Token>),
    /// Run the consumer at least once and then as many times as possible, optionally separated by the given token.
    OneOrMore(C, Option<Token>),
}

impl<C: ConsumerVec> Consumer<C> {
    pub fn consume(
        self,
        tokens: &mut TokenTreeCursor,
        consumer_span: Option<Span>,
    ) -> Result<Vec<Spanned<TokenTree>>, MetaSignatureError> {
        match self {
            Self::Token(token) => todo!(),
            _ => todo!(),
        }
    }
}

/// A token tree parser.
#[derive(Clone, Debug, Default)]
pub struct TokenTreeParser<C: ConsumerVec = UnspannedConsumerVec> {
    consumers: C,
}

pub type SpannedTokenTreeParser = TokenTreeParser<SpannedConsumerVec>;

impl<C: ConsumerVec> TokenTreeParser<C> {
    /// Creates a new token tree parser with the given consumers.
    pub fn new(consumers: C) -> Self {
        Self { consumers }
    }
}

macro_rules! consumer {
    // Base case: when no more consumer needs to be chained
    (@consumer $tree:expr $(,)?) => { $tree };

    // When a named consumer is present
    (@consumer $tree:expr, $name:literal : $consumer:expr $(, $($rest:tt)*)?) => {
        consumer!(@consumer $tree.then_named($name, $consumer), $($($rest)*)?)
    };

    // When an optional consumer is present
    (@consumer $tree:expr, maybe $consumer:expr $(, $($rest:tt)*)?) => {
        consumer!(
            @consumer
            $tree.then(Consumer::Maybe($consumer)),
            $($($rest)*)?
        )
    };

    // When a zero or more consumer is present
    (@consumer $tree:expr, zero_or_more $consumer:expr => $sep:expr $(, $($rest:tt)*)?) => {
        consumer!(
            @consumer
            $tree.then(Consumer::ZeroOrMore($consumer, $sep)),
            $($($rest)*)?
        )
    };

    // When a one or more consumer is present
    (@consumer $tree:expr, one_or_more $consumer:expr => $sep:expr $(, $($rest:tt)*)?) => {
        consumer!(
            @consumer
            $tree.then(Consumer::OneOrMore($consumer, $sep)),
            $($($rest)*)?
        )
    };

    // When an unnamed consumer is present
    (@consumer $tree:expr, $consumer:expr $(, $($rest:tt)*)?) => {
        consumer!(@consumer $tree.then($consumer), $($($rest)*)?)
    };

    // Entry point with initial consumer
    ($($tt:tt)*) => {
        UnspannedConsumerVec::new(|tree| consumer!(@consumer tree, $($tt)*))
    };
}

impl TokenTreeParser {
    /// Parser for the `@suppress` and `@forbid` decorators.
    pub fn path_list_with_reason() -> Self {
        // #(#name:path),+ #(, #(reason: #reason:string_literal #(,)?)?)?
        Self::new(consumer! {
            one_or_more consumer!("path": Consumer::Path) => Some(Token::Comma),
            maybe consumer! {
                Consumer::Token(Token::Comma),
                maybe consumer! {
                    "reason": Consumer::StringLiteral,
                    maybe consumer!(Consumer::Token(Token::Comma))
                }
            }
        })
    }

    /// Parser for the `@derive` decorator.
    pub fn path_list() -> Self {
        // #(#name:path),+ #(,)?
        Self::new(consumer! {
            one_or_more consumer!("path": Consumer::Path) => Some(Token::Comma),
            maybe consumer!(Consumer::Token(Token::Comma)),
        })
    }
}
