pub mod util;

use util::to_snake_case;
use terbium_grammar::{ParseInterface, Token, tokenizer};

use chumsky::Parser;
use std::io::Write;

#[repr(u16)]
#[derive(Clone, Debug, PartialEq)]
pub enum WarningType {
    NonSnakeCase(Token), // the Ident token triggering this
    NonAscii(Token),
}

impl WarningType {
    /// Return a number 0 to 5 (inclusive) representing
    /// the servity of this specific type of warning.
    ///
    /// This can be used to ignore errors lower than a specific severity,
    /// or exit the analysis stage all together when a warning with a higher
    /// serverity is encountered.
    ///
    /// A higher number means a more severe warning.
    /// By default, the analyzer is set to ignore no errors and stop
    /// analysis at only level 5.
    pub fn severity(&self) -> u8 {
        match self {
            Self::NonSnakeCase(_) => 1,
            Self::NonAscii(_) => 2,
        }
    }
}

#[repr(u16)]
#[derive(Clone, Debug, PartialEq)]
pub enum ErrorType {
    UnresolvableIdentifier(Token), // the Ident token triggering this
}

#[derive(Clone, Debug, PartialEq)]
pub enum AnalyzerMessageKind {
    Info,
    Warning(WarningType),
    Error(ErrorType),
}

#[derive(Clone, Debug, PartialEq)]
pub struct AnalyzerMessage {
    kind: AnalyzerMessageKind,
    message: String,
    // TODO: span
}

/// Represents a single analyzer which attempts to anaylize or optimize
/// a specific pattern of code.
pub trait Analyzer<P, E = &'static str> {
    /// The config/flag form of this, e.g. UnusedVariables -> "unused-variables"
    /// as in `// @trb:allow unused-variables`
    const ID: &'static str;

    fn from_raw_tokens(tokens: impl IntoIterator<Item = Token>) -> P
    where
        P: ParseInterface,
    {
        P::from_tokens(tokens.collect()).0
    }

    fn analyze(input: P, messages: &mut Vec<AnalyzerMessage>) -> Result<(), E>;
}

// TODO: currently analyzers are completely independent from other analyzers, which degrades performance.
// TODO: (cont.) for example, NonSnakeCaseAnalyzer and NonAsciiAnalyzer both only check identifiers
// TODO: (cont.) and analysis for both could be done for each iteration of tokens.
// TODO: (cont.) however, these are currently independent and the Vec of tokens will
// TODO: (cont.) have to be cloned in order to perform analysis.
pub struct NonSnakeCaseAnalyzer;
impl Analyzer<Vec<Token>> for NonSnakeCaseAnalyzer {
    const ID: &'static str = "non-snake-case";

    fn analyze(input: Vec<Token>, messages: &mut Vec<AnalyzerMessage>) -> Result<(), &'static str> {
        for tk in input {
            if let Token::Identifier(i) = tk {
                let snake = to_snake_case(i.as_str());
                // TODO: account for constants, which should be SCREAMING_SNAKE_CASE
                if i.to_ascii_lowercase() == snake { continue; }

                messages.push(AnalyzerMessage {
                    kind: AnalyzerMessageKind::Warning(WarningType::NonSnakeCase(tk.to_owned())),
                    message: "convention is to use snake_case casing for identifiers".to_string(),
                })
            }
        }

        Ok(())
    }
}

pub struct NonAsciiAnalyzer;
impl Analyzer<Vec<Token>> for NonAsciiAnalyzer {
    const ID: &'static str = "non-ascii";

    fn analyze(input: Vec<Token>, messages: &mut Vec<AnalyzerMessage>) -> Result<(), &'static str> {
        for tk in input {
            if let Token::Identifier(i) = tk {
                if i.is_ascii() { continue; }

                messages.push(AnalyzerMessage {
                    kind: AnalyzerMessageKind::Warning(WarningType::NonAscii(tk.to_owned())),
                    message: "convention is to exclude non-ascii characters in identifier names"
                        .to_string(),
                })
            }
        }

        Ok(())
    }
}

pub(crate) fn run_analyzer(id: String, tokens: Vec<Token>, messages: &mut Vec<AnalyzerMessage>) -> Result<(), &'static str> {
    match id.as_str() {
        "non-snake-case" => NonSnakeCaseAnalyzer::analyze(tokens, messages),
        "non-ascii" => NonAsciiAnalyzer::analyze(tokens, messages),
        _ => Err(format!("unknown analyzer with id {:?}", id).as_str())
    }
}

#[derive(Debug)]
pub struct BulkAnalyzer {
    pub analyzers: Vec<String>, // Vec of analyzers by ID
    pub messages: Vec<AnalyzerMessage>,
}

impl BulkAnalyzer {
    pub fn new() -> Self {
        Self {
            analyzers: Vec::new(),
            messages: Vec::new(),
        }
    }

    pub fn analyze_tokens(&mut self, tokens: Vec<Token>) {
        self.analyzers.for_each(|a| run_analyzer(a, tokens, &mut self.messages));
    }

    pub fn analyze_string(&mut self, s: String) {
        let tokens = tokenizer().parse(s.as_str()).unwrap_or_else(|_| todo!());

        self.analyze_tokens(tokens)
    }

    pub fn write(&self, writer: impl Write) {
        self.messages.for_each(|m| writeln!(writer, "{:?}", m));
    }
}

#[cfg(test)]
mod tests {
    use super::BulkAnalyzer;

    #[test]
    fn test_analysis() {
        let mut a = BulkAnalyzer::new();

        a.analyze_string(String::from("
            func camelCase() {
                let notSnakeCase = 5;
                let non_ascii_identifier_\u{2022} = 5;
            }
        "));

        a.write(std::io::stdout());
    }
}