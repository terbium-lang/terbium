//! Terbium diagnostic reporting.

use common::span::Span;

pub mod span;
pub mod write;

/// The severity of a diagnostic.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum Severity {
    Error(usize),
    Warning(usize),
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
    pub sections: Vec<Section>,
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
        self.sections.push(section);
        self
    }

    pub fn with_end(mut self, label: Option<&str>, message: impl ToString) -> Self {
        self.end.push((label.map(ToString::to_string), message.to_string()));
        self
    }

    pub fn wrap_to(mut self, cols: usize) -> Self {
        self.wrap = Some(cols);
        self
    }
}

#[derive(Clone, Debug)]
pub enum LabelKind {
    Label(Label),
    Fix(Fix),
}

impl LabelKind {
    pub const fn span(&self) -> Span {
        match self {
            Self::Label(l) => l.span,
            Self::Fix(f) => f.action.span(),
        }
    }

    pub fn context_span(&self) -> Span {
        match self {
            Self::Label(l) => l.context_span.unwrap_or(l.span),
            Self::Fix(f) => f.action.span(),
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
    pub labels: Vec<LabelKind>,
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
        self.labels.push(LabelKind::Label(label));
        self
    }

    pub fn with_fix(mut self, fix: Fix) -> Self {
        self.labels.push(LabelKind::Fix(fix));
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
}

impl Label {
    pub fn at(span: Span) -> Self {
        Self {
            span,
            context_span: None,
            message: String::new(),
        }
    }

    pub fn with_context_span(mut self, span: Span) -> Self {
        self.context_span = Some(span);
        self
    }

    pub fn with_message(mut self, message: impl ToString) -> Self {
        self.message = message.to_string();
        self
    }
}

/// The action to take to fix a diagnostic.
#[derive(Clone, Debug)]
pub enum Action {
    /// Replace the content at the span with the given context.
    Replace(Span, String),
    /// Delete the content at the span.
    Delete(Span),
    /// Insert the given content before the span.
    InsertBefore(Span, String),
    /// Insert the given content after the span.
    InsertAfter(Span, String),
}

impl Action {
    pub const fn span(&self) -> Span {
        match self {
            Self::Replace(span, _) => *span,
            Self::Delete(span) => *span,
            Self::InsertBefore(span, _) => *span,
            Self::InsertAfter(span, _) => *span,
        }
    }
}

/// A fix section.
#[derive(Clone, Debug)]
pub struct Fix {
    /// The action to take to fix the diagnostic.
    pub action: Action,
    /// The message of the fix, displayed at the beginning of the section.
    pub message: String,
    /// The label message of the fix, displayed within the source code.
    pub label: String,
}

impl Fix {
    pub fn new(action: Action) -> Self {
        Self {
            action,
            message: String::new(),
            label: String::new(),
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
}
