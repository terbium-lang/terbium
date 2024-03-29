//! Terbium diagnostic reporting.

#![feature(let_chains)]

use common::span::Span;

pub mod span;
pub mod write;

/// The severity of a diagnostic.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum Severity {
    Error(usize),
    Warning(&'static str),
    #[default]
    Info,
}

/// A diagnostic report.
#[derive(Clone, Debug, Default)]
pub struct Diagnostic {
    /// The severity of the diagnostic.
    pub severity: Severity,
    /// The message of the diagnostic.
    pub message: String,
    /// The diagnostic sections.
    pub sections: Vec<SectionKind>,
    /// Extra lines at the end of the diagnostic.
    pub end: Vec<(Option<String>, String)>,
    /// If `Some`, the diagnostic will be wrapped to this number of columns.
    pub wrap: Option<usize>,
}

impl Diagnostic {
    pub fn new(severity: Severity, message: impl ToString) -> Self {
        Self {
            severity,
            message: message.to_string(),
            sections: Vec::new(),
            end: Vec::new(),
            wrap: None,
        }
    }

    pub fn with_section(mut self, section: Section) -> Self {
        self.sections.push(SectionKind::Standard(section));
        self
    }

    pub fn with_fix(mut self, fix: Fix) -> Self {
        self.sections.push(SectionKind::Fix(fix));
        self
    }

    pub fn with_end(mut self, label: Option<&str>, message: impl ToString) -> Self {
        self.end
            .push((label.map(ToString::to_string), message.to_string()));
        self
    }

    pub fn with_help(self, help: impl ToString) -> Self {
        self.with_end(Some("help"), help.to_string())
    }

    pub fn with_note(self, note: impl ToString) -> Self {
        self.with_end(Some("note"), note.to_string())
    }

    pub fn wrap_to(mut self, cols: usize) -> Self {
        self.wrap = Some(cols);
        self
    }
}

#[derive(Clone, Debug)]
pub enum SectionKind {
    Standard(Section),
    Fix(Fix),
}

impl SectionKind {
    pub fn full_span(&self) -> Span {
        match self {
            Self::Standard(s) => s.span.unwrap_or_else(|| {
                Span::from_spans(
                    s.labels
                        .iter()
                        .map(|lbl| lbl.context_span.unwrap_or(lbl.span)),
                )
            }),
            Self::Fix(f) => f.action.span(),
        }
    }

    pub const fn explicit_span(&self) -> Option<Span> {
        match self {
            Self::Standard(s) => s.span,
            Self::Fix(f) => f.span,
        }
    }

    pub fn note(&self) -> Option<&str> {
        match self {
            Self::Standard(s) => s.note.as_deref(),
            Self::Fix(f) => f.note.as_deref(),
        }
    }

    pub fn inner_context_spans(&self) -> Vec<Span> {
        match self {
            Self::Standard(s) => s.labels.iter().map(Label::full_span).collect(),
            Self::Fix(f) => vec![f.action.span()],
        }
    }
}

/// A diagnostic section, which previews a section of code.
#[derive(Clone, Debug)]
pub struct Section {
    /// The header of the section.
    pub header: Option<String>,
    /// The span of the section. If no span is provided, the lines will be automatically determined
    /// by the labels.
    pub span: Option<Span>,
    /// Labels added to the section.
    pub labels: Vec<Label>,
    /// A note added at the end of the section.
    pub note: Option<String>,
}

impl Section {
    pub const fn new() -> Self {
        Self {
            header: None,
            span: None,
            labels: Vec::new(),
            note: None,
        }
    }

    pub fn over(span: Span) -> Self {
        Self {
            span: Some(span),
            ..Self::new()
        }
    }

    pub fn with_header(mut self, header: impl ToString) -> Self {
        self.header = Some(header.to_string());
        self
    }

    pub fn with_label(mut self, label: Label) -> Self {
        self.labels.push(label);
        self
    }

    pub fn with_note(mut self, note: impl ToString) -> Self {
        self.note = Some(note.to_string());
        self
    }
}

/// A label added to a section.
#[derive(Clone, Debug)]
pub struct Label {
    /// Where this label points to.
    pub span: Span,
    /// The span of relevant context to this label.
    pub context_span: Option<Span>,
    /// The message of the label.
    pub message: String,
    /// The character used to underline the label. Defaults to `^`.
    pub underline: char,
    /// The character used to underline the context. Defaults to `-`.
    pub context_underline: char,
}

impl Label {
    pub fn at(span: Span) -> Self {
        Self {
            span,
            context_span: None,
            message: String::new(),
            underline: '^',
            context_underline: '-',
        }
    }

    pub fn with_context_span(mut self, span: Span) -> Self {
        debug_assert!(
            span.len() >= self.span.len(),
            "context span cannot span less than the actual span",
        );
        self.context_span = Some(span);
        self
    }

    pub fn with_message(mut self, message: impl ToString) -> Self {
        self.message = message.to_string();
        self
    }

    pub fn with_underline(mut self, underline: char) -> Self {
        self.underline = underline;
        self
    }

    pub fn with_context_underline(mut self, underline: char) -> Self {
        self.context_underline = underline;
        self
    }

    pub fn full_span(&self) -> Span {
        self.context_span.unwrap_or(self.span)
    }
}

/// The action to take to fix a diagnostic.
#[derive(Clone, Debug)]
pub enum Action {
    /// Replace the content at the span with the given context.
    Replace(Span, String),
    /// Delete the content at the span.
    Remove(Span),
    /// Insert the given content before the span.
    InsertBefore(Span, String),
    /// Insert the given content after the span.
    InsertAfter(Span, String),
}

impl Action {
    pub const fn span(&self) -> Span {
        let (Self::Replace(span, _)
        | Self::Remove(span)
        | Self::InsertBefore(span, _)
        | Self::InsertAfter(span, _)) = self;
        *span
    }
}

/// A fix section.
#[derive(Clone, Debug)]
pub struct Fix {
    /// The full span that should be displayed, if any.
    pub span: Option<Span>,
    /// The action to take to fix the diagnostic.
    pub action: Action,
    /// The message of the fix, displayed at the beginning of the section.
    pub message: String,
    /// The label message of the fix, displayed within the source code.
    pub label: String,
    /// A note displayed at the end of the fix.
    pub note: Option<String>,
}

impl Fix {
    pub fn new(action: Action) -> Self {
        Self {
            span: None,
            action,
            message: String::new(),
            label: String::new(),
            note: None,
        }
    }

    pub fn over(span: Span, action: Action) -> Self {
        Self {
            span: Some(span),
            ..Self::new(action)
        }
    }

    pub fn with_message(mut self, message: impl ToString) -> Self {
        self.message = message.to_string();
        self
    }

    pub fn with_label(mut self, label: impl ToString) -> Self {
        self.label = label.to_string();
        self
    }

    pub fn with_note(mut self, note: impl ToString) -> Self {
        self.note = Some(note.to_string());
        self
    }
}
