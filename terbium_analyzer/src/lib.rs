pub mod util;

use terbium_grammar::{ParseInterface, Token, Source, Span};
use util::to_snake_case;

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
    #[must_use]
    pub const fn severity(&self) -> u8 {
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
    /// The config/flag form of this, e.g. `UnusedVariables` -> "unused-variables"
    /// as in `// @trb:allow unused-variables`
    const ID: &'static str;

    fn from_raw_tokens(tokens: impl IntoIterator<Item = (Token, Span)>) -> P
    where
        P: ParseInterface,
    {
        P::parse(tokens.into_iter().collect()).unwrap_or_else(|_| todo!())
    }

    #[allow(clippy::missing_errors_doc)]
    fn analyze(input: &P, messages: &mut Vec<AnalyzerMessage>) -> Result<(), E>;
}

// TODO: currently analyzers are completely independent from other analyzers, which degrades performance.
// TODO: (cont.) for example, NonSnakeCaseAnalyzer and NonAsciiAnalyzer both only check identifiers
// TODO: (cont.) and analysis for both could be done for each iteration of tokens.
// TODO: (cont.) however, these are currently independent and the Vec of tokens will
// TODO: (cont.) have to be cloned in order to perform analysis.
pub struct NonSnakeCaseAnalyzer;
impl Analyzer<Vec<(Token, Span)>> for NonSnakeCaseAnalyzer {
    const ID: &'static str = "non-snake-case";

    fn analyze(
        input: &Vec<(Token, Span)>,
        messages: &mut Vec<AnalyzerMessage>,
    ) -> Result<(), &'static str> {
        for (tk, _span) in input {
            if let Token::Identifier(ref i) = tk {
                let snake = to_snake_case(i.as_str());
                // TODO: account for constants, which should be SCREAMING_SNAKE_CASE
                if i.to_ascii_lowercase() == snake {
                    continue;
                }

                messages.push(AnalyzerMessage {
                    kind: AnalyzerMessageKind::Warning(WarningType::NonSnakeCase(tk.clone())),
                    message: "convention is to use snake_case casing for identifiers".to_string(),
                });
            }
        }

        Ok(())
    }
}

pub struct NonAsciiAnalyzer;
impl Analyzer<Vec<(Token, Span)>> for NonAsciiAnalyzer {
    const ID: &'static str = "non-ascii";

    fn analyze(
        input: &Vec<(Token, Span)>,
        messages: &mut Vec<AnalyzerMessage>,
    ) -> Result<(), &'static str> {
        for (tk, _span) in input {
            if let Token::Identifier(ref i) = tk {
                if i.is_ascii() {
                    continue;
                }

                messages.push(AnalyzerMessage {
                    kind: AnalyzerMessageKind::Warning(WarningType::NonAscii(tk.clone())),
                    message: "convention is to exclude non-ascii characters in identifier names"
                        .to_string(),
                });
            }
        }

        Ok(())
    }
}

#[allow(clippy::ptr_arg)]
pub(crate) fn run_analyzer(
    id: &str,
    tokens: &Vec<(Token, Span)>,
    messages: &mut Vec<AnalyzerMessage>,
) -> Result<(), String> {
    match id {
        "non-snake-case" => NonSnakeCaseAnalyzer::analyze(tokens, messages),
        "non-ascii" => NonAsciiAnalyzer::analyze(tokens, messages),
        _ => return Err(format!("unknown analyzer with id {:?}", id)),
    }
    .map_err(ToString::to_string)
}

#[derive(Debug)]
pub struct BulkAnalyzer {
    pub analyzers: Vec<String>, // Vec of analyzers by ID
    pub messages: Vec<AnalyzerMessage>,
}

impl BulkAnalyzer {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            analyzers: Vec::new(),
            messages: Vec::new(),
        }
    }

    #[must_use]
    pub fn new_with_analyzers(analyzers: Vec<String>) -> Self {
        Self {
            analyzers,
            messages: Vec::new(),
        }
    }

    #[allow(clippy::needless_pass_by_value, clippy::missing_panics_doc)]
    pub fn analyze_tokens(&mut self, tokens: Vec<(Token, Span)>) {
        for a in &self.analyzers {
            run_analyzer(a.as_str(), &tokens, &mut self.messages).expect("error trying to analyze");
        }
    }

    #[allow(clippy::needless_pass_by_value, clippy::missing_panics_doc)]
    pub fn analyze_string(&mut self, source: Source, s: String) {
        let tokens = Vec::<(Token, Span)>::from_string(source, s).unwrap_or_else(|_| todo!());

        self.analyze_tokens(tokens);
    }

    pub fn write(&self, mut writer: impl Write) {
        for m in &self.messages {
            writeln!(writer, "{:?}", m).expect("failed to write");
        }
    }
}

impl Default for BulkAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::BulkAnalyzer;
    use terbium_grammar::Source;

    #[test]
    fn test_analysis() {
        let mut a = BulkAnalyzer::new_with_analyzers(
            vec!["non-snake-case"]
                .iter()
                .map(ToString::to_string)
                .collect(),
        );

        a.analyze_string(Source::default(), String::from(
            "
            func camelCase() {
                let notSnakeCase = 5;
            }
        ",
        ));

        a.write(std::io::stdout());
    }
}
