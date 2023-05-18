use crate::span::{Line, LineSrc};
use crate::{Action, Diagnostic, Fix, Label, Section, SectionKind, Severity};
use common::span::{Provider, Span, Src};
use std::{
    collections::HashMap,
    fmt::{Display, Formatter},
    io::{self, Write},
};
use yansi::{Color, Paint};

// constants are named by the position of their corner, for example `L` would be said to be `BOTTOM_LEFT`
const TOP_LEFT: char = '\u{250c}';
const BOTTOM_RIGHT: char = '\u{2518}';
const BOTTOM_LEFT: char = '\u{2514}';
const HORIZONTAL: char = '\u{2500}';
const VERTICAL: char = '\u{2502}';
const VERTICAL_OUT: char = '\u{251c}';
const DOTTED_VERTICAL: char = ':';
const CROSS: char = '\u{2713}';

const ERROR_COLOR: Color = Color::Red;
const WARNING_COLOR: Color = Color::Yellow;
const INFO_COLOR: Color = Color::Fixed(79);
const MARGIN_COLOR: Color = Color::Fixed(246);
const EXTRA_COLOR: Color = Color::Fixed(43);

const LABEL_COLORS: [Color; 4] = [
    Color::Cyan,
    Color::Magenta,
    Color::Yellow,
    Color::Fixed(250),
];

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

    pub fn tidy_chunks(&self, chars: Vec<(char, Option<Color>)>) -> String {
        // Merge the characters into strings of similar colors
        // For example, [('a', red), ('b', red), ('c', blue)] becomes [("ab", red), ("c", blue)]
        let mut colored = String::new();
        let mut current = (String::new(), None);

        macro_rules! extend {
            () => {{
                if !current.0.is_empty() {
                    colored.extend(
                        if let Some(color) = current.1 {
                            self.color(&current.0).fg(color).to_string()
                        } else {
                            current.0.clone()
                        }
                        .chars(),
                    );
                }
            }};
        }

        for (c, color) in chars {
            if color != current.1 {
                extend!();
                current = (String::new(), color);
            }
            current.0.push(c);
        }
        extend!();
        colored
    }

    pub fn highlight_spans(&self, line: &Line, spans: Vec<(Span, Color)>) -> String {
        let mut chars = line.chars().map(|c| (c, None)).collect::<Vec<_>>();

        // "Color" the spans
        for (span, color) in spans {
            for i in span.range() {
                chars[i - line.offset].1 = Some(color);
            }
        }
        self.tidy_chunks(chars)
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

    pub fn compute_section_labels<'a>(&self, labels: &'a [Label]) -> ComputedLabels<'a> {
        let mut computed = ComputedLabels {
            inline: HashMap::new(),
            multiline: Vec::new(),
        };

        for (position, label) in labels.into_iter().enumerate() {
            let color = LABEL_COLORS[position % LABEL_COLORS.len()];
            let span = label.context_span.unwrap_or(label.span);
            let line_range = self
                .cache
                .get(&span.src)
                .expect("missing source")
                .get_line_range(span);

            if line_range.len() > 1 {
                computed.multiline.push(MultilineLabel {
                    label: &label.message,
                    endpoint: '>',
                    span,
                    color,
                    position,
                });
            } else {
                computed
                    .inline
                    .entry(line_range.start)
                    .or_default()
                    .push(InlineLabel {
                        context_span: span,
                        span: label.span,
                        color,
                        label: &label.message,
                        char: '^',
                        position,
                    });
            }
        }

        computed
    }

    pub fn compute_fix_labels<'a>(
        &self,
        fix: &'a Fix,
        lookup: &mut HashMap<usize, Line>,
    ) -> ComputedLabels<'a> {
        let mut computed = ComputedLabels {
            inline: HashMap::new(),
            multiline: Vec::new(),
        };

        let src = self
            .cache
            .get(&fix.action.span().src)
            .expect("missing source");
        // TODO: multiline fix labels
        let (line, num, _) = src.get_offset_line(fix.action.span().start).unwrap();
        let mut line = line.clone();

        let span = match &fix.action {
            Action::InsertBefore(span, content) => {
                line.chars.insert_str(span.start - line.offset, content);
                span.get_span(span.start, span.start + content.len())
            }
            Action::InsertAfter(span, content) => {
                line.chars.insert_str(span.end - line.offset, content);
                span.get_span(span.end, span.end + content.len())
            }
            Action::Replace(span, content) => {
                line.chars
                    .replace_range(span.start - line.offset..span.end - line.offset, content);
                span.get_span(span.start, span.start + content.len())
            }
            // we want to show the user the span will be removed, but we don't want to actually
            // remove it from the source
            Action::Remove(span) => *span,
        };
        lookup.insert(num, line);

        let (color, char) = match fix.action {
            Action::InsertBefore(..) | Action::InsertAfter(..) => (Color::Green, '+'),
            Action::Replace(..) => (Color::Yellow, '~'),
            Action::Remove(_) => (Color::Red, '-'),
        };
        computed.inline.insert(
            num,
            vec![InlineLabel {
                context_span: span,
                span,
                color,
                label: &fix.label,
                char,
                position: 0,
            }],
        );
        computed
    }

    pub fn write_section<W: Write>(
        &self,
        mut w: W,
        cached: CachedSection,
        max_line_num_len: usize,
    ) -> io::Result<()> {
        let blanks = vec![b' '; max_line_num_len + 1];
        w.write(&blanks)?;

        let full_span = cached.section.full_span();
        let (src, start) = (full_span.src, full_span.start);
        let (_, start_line, start_col) = self
            .cache
            .get(&src)
            .expect("missing source")
            .get_offset_line(start)
            .unwrap();

        writeln!(
            w,
            "{}{}{src}:{}:{}{}",
            self.color(TOP_LEFT).fg(MARGIN_COLOR),
            self.color('[').fg(MARGIN_COLOR),
            start_line + 1,
            start_col + 1,
            self.color(']').fg(MARGIN_COLOR),
        )?;

        // SAFETY: `blanks` is a vector of spaces, so it is valid utf8
        let blanks = unsafe { String::from_utf8_unchecked(blanks) };
        writeln!(w, "{blanks}{}", self.color(VERTICAL).fg(MARGIN_COLOR))?;

        let mut prev = None;
        let mut line_lookup = HashMap::new();
        let mut computed = match cached.section {
            SectionKind::Standard(ref sect) => self.compute_section_labels(&sect.labels),
            SectionKind::Fix(ref fix) => self.compute_fix_labels(fix, &mut line_lookup),
        };
        for (line_num, line) in cached.lines {
            // If we're skipping lines, `:` could act as a "vertical ellipsis" to show that we are
            // in fact skipping lines.
            if let Some(prev) = prev && line_num != prev + 1 {
                writeln!(w, "{blanks}{}", self.color(DOTTED_VERTICAL).fg(MARGIN_COLOR))?;
            }
            prev = Some(line_num);

            let mut labels = computed.inline.remove(&line_num).unwrap_or_default();
            writeln!(
                w,
                "{} {} {}",
                self.color(format!("{:width$}", line_num + 1, width = max_line_num_len))
                    .fg(Color::Fixed(250)),
                self.color(VERTICAL).fg(MARGIN_COLOR),
                self.highlight_spans(
                    line_lookup.get(&line_num).unwrap_or(&line),
                    labels.iter().map(|lbl| (lbl.span, lbl.color)).collect()
                ),
            )?;

            // For single-line labels, the label can be written inline with the arrow
            if labels.len() == 1 {
                let label = labels[0];
                let arrow = label
                    .context_span
                    .range()
                    .map(|idx| {
                        if label.span.range().contains(&idx) {
                            label.char
                        } else {
                            '-'
                        }
                    })
                    .collect::<String>();

                let offset = " ".repeat(label.context_span.start - line.offset);
                let arrow = self.color(arrow).fg(label.color);
                writeln!(
                    w,
                    "{blanks}{} {offset}{} {}",
                    self.color(VERTICAL).fg(MARGIN_COLOR),
                    arrow,
                    label.label,
                )?;
                continue;
            }

            // Otherwise, write overlapping arrows
            //
            // For example,
            // L │ let sample = a + b;
            //   │     ^^^^^^   --^--
            //   │       └────────│──── first label
            //   │                └──── second label

            // All arrow endpoints will be rendered on a single line
            let mut endpoints = vec![(' ', None); line.len];
            for label in &labels {
                // this initial check is an optimization
                if label.context_span != label.span {
                    for i in label.context_span.range() {
                        endpoints[i - line.offset] = ('-', Some(label.color));
                    }
                }
                for i in label.span.range() {
                    endpoints[i - line.offset] = (label.char, Some(label.color));
                }
            }
            // Write the endpoints
            writeln!(
                w,
                "{blanks}{} {}",
                self.color(VERTICAL).fg(MARGIN_COLOR),
                self.tidy_chunks(endpoints),
            )?;

            // From now on, labels should be sorted by the position they were defined in
            labels.sort_by_key(|lbl| lbl.position);
            // Create a running list of segments, which will be overwritten by further segments
            let mut segments = vec![vec![(' ', None); line.len]; labels.len()];

            for (idx, label) in labels.iter().enumerate() {
                // find the center of the primary span
                let center = label.span.start + label.span.len() / 2 - line.offset;
                // draw the vertical part of the arrow
                for i in 0..idx {
                    segments[i][center] = (VERTICAL, Some(label.color));
                }
                segments[idx][center] = (BOTTOM_LEFT, Some(label.color));
                // draw the horizontal part of the arrow. it should span until the end of the line.
                for i in center + 1..line.len {
                    segments[idx][i] = (HORIZONTAL, Some(label.color));
                }
            }

            for (segment, label) in segments.into_iter().zip(labels) {
                writeln!(
                    w,
                    "{blanks}{} {} {}",
                    self.color(VERTICAL).fg(MARGIN_COLOR),
                    self.tidy_chunks(segment),
                    label.label,
                )?;
            }
        }

        // Give an extra line of padding if there is a note
        if cached.section.note().is_some() {
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
        if let Some(note) = cached.section.note() {
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
                let mut lines = sect
                    .explicit_span()
                    .map(|span| {
                        // If an explicit span was provided, use that
                        let source = self.cache.get(&span.src).expect("missing source");

                        source
                            .get_line_range(span)
                            .map(|line| (line, source.lines[line].clone()))
                            .collect::<Vec<_>>()
                    })
                    .or_else(|| {
                        let labels = sect.inner_context_spans();
                        // If no span was provided, use the labels
                        let source = self
                            .cache
                            .get(&labels.first()?.src)
                            .expect("missing source");
                        Some(
                            labels
                                .iter()
                                .flat_map(|span| {
                                    source
                                        .get_line_range(*span)
                                        .map(|line| (line, source.lines[line].clone()))
                                })
                                .collect(),
                        )
                    })
                    .expect("section without labels?");
                lines.dedup_by_key(|(line, _)| *line);

                CachedSection {
                    lines,
                    section: sect,
                }
            })
            .collect::<Vec<_>>();

        // grab the largest section line number
        let max_line_len = sections
            .iter()
            .flat_map(|sect| sect.lines.iter().rev().map(|(line, _)| line + 1))
            .max()
            .map(|value| if value == 0 { 1 } else { value })
            .unwrap_or(1)
            .ilog10()
            + 1;

        for section in sections {
            match section.section {
                SectionKind::Standard(Section {
                    header: Some(ref header),
                    ..
                }) => {
                    writeln!(w, "\n{header}")?;
                }
                SectionKind::Fix(ref fix) => {
                    writeln!(
                        w,
                        "\n{} {}",
                        self.color("fix:").fg(EXTRA_COLOR).bold(),
                        fix.message
                    )?;
                }
                _ => {}
            }
            self.write_section(&mut w, section, max_line_len as _)?;
        }

        for (label, message) in diagnostic.end {
            write!(
                w,
                "{} {} ",
                " ".repeat(max_line_len as _),
                self.color('=').fg(MARGIN_COLOR),
            )?;

            let mut indent = 4;
            if let Some(label) = label {
                indent += label.len() + 2;
                write!(w, "{} ", self.color(label + ":").fg(EXTRA_COLOR).bold())?;
            }

            let mut lines = message.lines();
            if let Some(first) = lines.next() {
                writeln!(w, "{first}")?;
            }
            for line in lines {
                writeln!(w, "{}{line}", " ".repeat(indent))?;
            }
        }
        Ok(())
    }
}

#[derive(Copy, Clone)]
pub struct InlineLabel<'a> {
    pub span: Span,
    pub context_span: Span,
    pub color: Color,
    pub label: &'a str,
    pub char: char,
    pub position: usize,
}

#[derive(Copy, Clone)]
pub struct MultilineLabel<'a> {
    pub span: Span,
    pub color: Color,
    pub label: &'a str,
    pub endpoint: char,
    pub position: usize,
}

#[derive(Clone)]
pub struct ComputedLabels<'a> {
    pub inline: HashMap<usize, Vec<InlineLabel<'a>>>,
    pub multiline: Vec<MultilineLabel<'a>>,
}

impl ComputedLabels<'_> {
    pub fn finalize(&mut self) {
        self.inline.values_mut().for_each(|labels| {
            labels.sort_unstable_by_key(|label| -(label.context_span.len() as isize));
        })
    }
}

#[derive(Clone)]
pub struct CachedSection {
    section: SectionKind,
    lines: Vec<(usize, Line)>,
}
