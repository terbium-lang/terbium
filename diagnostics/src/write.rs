use crate::span::{Line, LineSrc};
use crate::{Diagnostic, LabelKind, Section, Severity};
use common::span::{Provider, Span, Src};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::io::{self, Write};
use yansi::{Color, Paint};

// constants are named by the position of their corner, for example `L` would be said to be `BOTTOM_LEFT`
const TOP_LEFT: char = '\u{250c}';
const BOTTOM_RIGHT: char = '\u{2518}';
const BOTTOM_LEFT: char = '\u{2514}';
const HORIZONTAL: char = '\u{2500}';
const VERTICAL: char = '\u{2502}';
const VERTICAL_OUT: char = '\u{251c}';

const ERROR_COLOR: Color = Color::Red;
const WARNING_COLOR: Color = Color::Yellow;
const INFO_COLOR: Color = Color::Fixed(79);
const MARGIN_COLOR: Color = Color::Fixed(246);
const EXTRA_COLOR: Color = Color::Fixed(43);

#[derive(Clone)]
enum Colored<T: ToString> {
    Colored(Paint<T>),
    Uncolored(String),
}

impl<T: ToString> Colored<T> {
    fn fg(self, color: Color) -> Self {
        if let Colored::Colored(paint) = self {
            return Self::Colored(paint.fg(color));
        }
        self
    }

    fn bold(self) -> Self {
        if let Colored::Colored(paint) = self {
            return Self::Colored(paint.bold());
        }
        self
    }
}

impl<T: Display> Display for Colored<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Colored(paint) => write!(f, "{paint}"),
            Self::Uncolored(string) => write!(f, "{string}"),
        }
    }
}

#[derive(Clone)]
pub struct DiagnosticWriter {
    cache: HashMap<Src, LineSrc>,
    ansi: bool,
}

impl DiagnosticWriter {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            ansi: true,
        }
    }

    fn color<T: ToString>(&self, text: T) -> Colored<T> {
        if self.ansi {
            Colored::Colored(Paint::new(text))
        } else {
            Colored::Uncolored(text.to_string())
        }
    }

    pub fn add_provider(&mut self, provider: Provider) {
        self.cache
            .insert(provider.src(), LineSrc::from(provider.content()));
    }

    pub fn highlight_span(&self, line: Line, span: Span, color: Color) -> String {
        let before = &line.chars[..span.start - line.offset];
        let inner = &line.chars[span.start - line.offset..span.end - line.offset];
        let after = &line.chars[span.end - line.offset..];

        format!("{before}{}{after}", self.color(inner).fg(color))
    }

    pub fn write_severity<W: Write>(&self, mut w: W, severity: Severity) -> io::Result<()> {
        match severity {
            Severity::Error(code) => write!(
                w,
                "{} (E{code:0>3})",
                self.color("error").fg(ERROR_COLOR).bold()
            ),
            Severity::Warning(code) => write!(
                w,
                "{} (W{code:0>3})",
                self.color("warning").fg(WARNING_COLOR).bold()
            ),
            Severity::Info => write!(w, "{}", self.color("info").fg(INFO_COLOR).bold()),
        }
    }

    pub fn compute_labels(&self, labels: Vec<LabelKind>) -> ComputedLabels {
        let mut labels = ComputedLabels {
            inline: Vec::new(),
            multiline: Vec::new(),
        };

        labels
    }

    pub fn write_section<W: Write>(
        &self,
        mut w: W,
        cached: CachedSection,
        max_line_num_len: usize,
    ) -> io::Result<()> {
        let blanks = vec![b' '; max_line_num_len + 1];
        w.write(&blanks)?;

        let start_span = cached
            .section
            .span
            .unwrap_or_else(|| cached.section.labels.first().unwrap().context_span());
        let (src, start) = (start_span.src, start_span.start);
        let (_, start_line, start_col) = self
            .cache
            .get(&src)
            .expect("missing source")
            .get_offset_line(start)
            .unwrap();

        writeln!(
            w,
            "{}{}{src}:{start_line}:{start_col}{}",
            self.color(TOP_LEFT).fg(MARGIN_COLOR),
            self.color('[').fg(MARGIN_COLOR),
            self.color(']').fg(MARGIN_COLOR),
        )?;

        // SAFETY: `blanks` is a vector of spaces, so it is valid utf8
        let blanks = unsafe { String::from_utf8_unchecked(blanks) };
        writeln!(w, "{blanks}{}", self.color(VERTICAL).fg(MARGIN_COLOR))?;

        let mut prev = usize::MAX;
        for (line_num, line) in cached.lines {
            // If we're skipping lines, `:` could act as a "vertical ellipsis" to show that we are
            // in fact skipping lines.
            if line_num != prev + 1 {
                writeln!(w, "{blanks}{}", self.color(':').fg(MARGIN_COLOR))?;
            }
            prev = line_num;

            writeln!(
                w,
                "{: >width$} {} {}",
                self.color(line_num + 1).fg(Color::Fixed(250)),
                self.color(VERTICAL).fg(MARGIN_COLOR),
                line.chars,
                width = max_line_num_len,
            )?;
        }

        // Give an extra line of padding if there is a note
        if cached.section.note.is_some() {
            writeln!(w, "{blanks}{}", self.color(VERTICAL).fg(MARGIN_COLOR))?;
        }
        // Finish the section
        let blanks = vec![b' '; max_line_num_len];
        w.write(&blanks)?;
        write!(
            w,
            "{}{}",
            self.color(HORIZONTAL).fg(MARGIN_COLOR),
            self.color(BOTTOM_RIGHT).fg(MARGIN_COLOR),
        )?;
        if let Some(note) = cached.section.note {
            writeln!(w, " {} {note}", self.color("note:").fg(EXTRA_COLOR).bold())
        } else {
            writeln!(w)
        }
    }

    pub fn write_diagnostic<W: Write>(&self, mut w: W, diagnostic: Diagnostic) -> io::Result<()> {
        // header
        self.write_severity(&mut w, diagnostic.severity)?;
        writeln!(w, ": {}", self.color(diagnostic.message).bold())?;

        let sections = diagnostic
            .sections
            .into_iter()
            .map(|sect| {
                // Only keep lines that are labelled
                let lines = sect
                    .span
                    .map(|span| {
                        // If an explicit span was provided, use that
                        let source = self.cache.get(&span.src).expect("missing source");

                        source
                            .get_line_range(span)
                            .map(|line| (line, source.lines[line].clone()))
                            .collect::<Vec<_>>()
                    })
                    .or_else(|| {
                        // If no span was provided, use the labels
                        let source = self
                            .cache
                            .get(&sect.labels.first()?.context_span().src)
                            .expect("missing source");
                        Some(
                            sect.labels
                                .iter()
                                .flat_map(|label| {
                                    debug_assert!(label.context_span().len() >= label.span().len());

                                    source
                                        .get_line_range(label.context_span())
                                        .map(|line| (line, source.lines[line].clone()))
                                })
                                .collect(),
                        )
                    })
                    .expect("section without labels?");

                CachedSection {
                    lines,
                    section: sect,
                }
            })
            .collect::<Vec<_>>();

        // grab the largest section line number
        let max_line_len = sections
            .iter()
            .flat_map(|sect| sect.lines.iter().rev().map(|(line, _)| line))
            .max()
            .copied()
            .map(|value| if value == 0 { 1 } else { value })
            .unwrap_or(1)
            .ilog10()
            + 1;

        for section in sections {
            self.write_section(&mut w, section, max_line_len as _)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct InlineLabel {
    pub span: Span,
    pub context_span: Span,
    pub color: Color,
    pub label: String,
    pub char: char,
}

#[derive(Clone)]
pub struct MultilineLabel {
    pub span: Span,
    pub color: Color,
    pub label: String,
    pub endpoint: Option<char>,
}

#[derive(Clone)]
pub struct ComputedLabels {
    pub inline: Vec<Vec<InlineLabel>>,
    pub multiline: Vec<MultilineLabel>,
}

#[derive(Clone)]
pub struct CachedSection {
    section: Section,
    lines: Vec<(usize, Line)>,
}
