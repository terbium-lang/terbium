#![feature(lint_reasons)]

pub mod util;

use std::collections::{HashMap, HashSet};
use std::fmt::Formatter;
use terbium_grammar::error::{Hint, HintAction};
use terbium_grammar::{Expr, Node, ParseInterface, Source, Span, Spanned, Target, Token};
use util::to_snake_case;

use crate::util::get_levenshtein_distance;
use ariadne::{sources, Cache};
use std::io::Write;
use std::str::FromStr;

#[derive(Clone, Debug, PartialEq)]
pub enum AnalyzerMessageKind {
    Info,
    Alert(AnalyzerKind),
}

#[derive(Debug, PartialEq)]
pub struct AnalyzerMessage {
    pub kind: AnalyzerMessageKind,
    message: String,
    label: Option<String>,
    span: Span,
    hint: Option<Hint>,
}

impl AnalyzerMessage {
    #[must_use]
    pub fn non_snake_case(name: &str, counterpart: String, span: Span) -> Self {
        Self {
            kind: AnalyzerMessageKind::Alert(AnalyzerKind::NonSnakeCase),
            message: "non-type identifier names should be snake_case".to_string(),
            label: Some(format!("{:?} is not snake_case", name)),
            span,
            hint: Some(Hint {
                message: format!("rename to {:?}", counterpart),
                action: HintAction::Replace(counterpart),
            }),
        }
    }

    pub fn unnecessary_mut_variable(name: String, span: Span) -> Self {
        Self {
            kind: AnalyzerMessageKind::Alert(AnalyzerKind::UnnecessaryMutVariables),
            message: "variable was unneedingly declared as mutable".to_string(),
            label: Some(format!(
                "variable {:?} is declared mutable here, but it is never mutated",
                name
            )),
            span,
            hint: Some(Hint {
                message: "make variable immutable by declaring with `let` instead".to_string(),
                action: HintAction::None,
            }),
        }
    }

    pub fn unused_variable(name: String, span: Span) -> Self {
        Self {
            kind: AnalyzerMessageKind::Alert(AnalyzerKind::UnusedVariables),
            message: "variable is declared but never used".to_string(),
            label: Some(format!(
                "variable {:?} is declared here, but it is never used",
                name
            )),
            span,
            hint: Some(Hint {
                message: format!(
                    "remove the declaration, or prefix with `_`: {:?}",
                    "_".to_string() + &*name
                ),
                action: HintAction::None,
            }),
        }
    }

    pub fn global_mutable_variable(name: String, span: Span) -> Self {
        Self {
            kind: AnalyzerMessageKind::Alert(AnalyzerKind::GlobalMutableVariables),
            message: "variable is declared as mutable in the global scope".to_string(),
            label: Some(format!(
                "variable {:?} is declared as mutable here, but it is accessible to the entire program",
                name
            )),
            span,
            hint: Some(Hint {
                message: "declare as immutable instead, or move the declaration into a non-global context such as inside of a function"
                    .to_string(),
                action: HintAction::None,
            }),
        }
    }

    #[must_use]
    pub fn unresolved_identifier(name: &str, close_match: Option<String>, span: Span) -> Self {
        Self {
            kind: AnalyzerMessageKind::Alert(AnalyzerKind::UnresolvedIdentifiers),
            message: "identifier could not be resolved".to_string(),
            label: Some(format!("variable {:?} not found in this scope", name)),
            span,
            hint: close_match.map(|m| Hint {
                message: format!("perhaps you meant {:?}", m),
                action: HintAction::Replace(m),
            }),
        }
    }

    #[must_use]
    pub fn redeclared_const_variable(name: &str, span: Span) -> Self {
        Self {
            kind: AnalyzerMessageKind::Alert(AnalyzerKind::RedeclaredConstVariables),
            message: "cannot redeclare variable declared as `const`".to_string(),
            label: Some(format!("variable {:?} was declared as `const`", name)),
            span,
            hint: Some(Hint {
                message: "declare with `let` instead".to_string(),
                action: HintAction::None,
            }),
        }
    }

    #[must_use]
    pub fn reassigned_immutable_variable(name: &str, span: Span, was_const: bool) -> Self {
        Self {
            kind: AnalyzerMessageKind::Alert(AnalyzerKind::ReassignedImmutableVariables),
            message: format!(
                "cannot reassign to {}",
                if was_const {
                    "variable declared as `const`"
                } else {
                    "immutable variable"
                }
            ),
            label: Some(format!(
                "attempted to reassign to variable {:?}, but it was {}",
                name,
                if was_const {
                    "declared as `const`"
                } else {
                    "never declared as `mut`"
                }
            )),
            span,
            hint: Some(Hint {
                message: "make variable mutable by declaring with `let mut` instead".to_string(),
                action: HintAction::None,
            }),
        }
    }

    /// Write error to specified writer.
    /// 
    /// # Panics
    /// * Panic when writing to writer failed.
    pub fn write<C: Cache<Source>>(self, cache: C, writer: impl Write) {
        use ariadne::{Color, Label, Report, ReportKind};

        #[expect(
            clippy::match_wildcard_for_single_variants, 
            reason = "Nothing should reach this arm"
        )]
        let color = match self.kind {
            AnalyzerMessageKind::Info => Color::Blue,
            AnalyzerMessageKind::Alert(k) if k.is_warning() => Color::Yellow,
            AnalyzerMessageKind::Alert(k) if k.is_error() => Color::Red,
            _ => unreachable!(),
        };

        let report = Report::build(
            #[allow(
                clippy::match_wildcard_for_single_variants, 
                reason = "Nothing should reach this arm"
            )]
            match self.kind {
                AnalyzerMessageKind::Info => ReportKind::Advice,
                AnalyzerMessageKind::Alert(k) if k.is_warning() => ReportKind::Warning,
                AnalyzerMessageKind::Alert(k) if k.is_error() => ReportKind::Error,
                _ => unreachable!(),
            },
            self.span.src(),
            self.span.start(),
        );

        let report = match self.kind {
            AnalyzerMessageKind::Alert(k) => report.with_code(k.code()),
            AnalyzerMessageKind::Info => report,
        }
        .with_message(self.message);

        let report = if let Some(label) = self.label {
            report.with_label(Label::new(self.span).with_message(label).with_color(color))
        } else {
            report
        };

        let report = if let Some(hint) = self.hint {
            report.with_help(hint.message)
        } else {
            report
        };

        let report = if let AnalyzerMessageKind::Alert(k) = self.kind {
            report.with_note(format!(
               "view this {} in the error index: https://github.com/TerbiumLang/standard/blob/main/error_index.md#{}{:0>3}",
               if k.is_error() { "error" } else { "warning" },
               if k.is_error() { "E" } else { "W" },
               k.code(),
           ))
        } else {
            report
        };

        report.finish().write(cache, writer).unwrap();
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ScopeEntryModifier {
    None,
    Mut,
    Const,
}

#[derive(Clone, Debug)]
pub struct MockScopeEntry {
    pub name: String,
    pub ty: (), // TODO
    pub modifier: ScopeEntryModifier,
    pub used: bool,
    pub mutated: bool,
    pub span: Span,
}

impl MockScopeEntry {
    pub fn new(name: String, ty: (), modifier: ScopeEntryModifier, span: Span) -> Self {
        Self {
            name,
            ty,
            modifier,
            used: false,
            mutated: false,
            span,
        }
    }

    #[must_use]
    pub fn is_let(&self) -> bool {
        self.modifier == ScopeEntryModifier::None
    }

    #[must_use]
    pub fn is_mut(&self) -> bool {
        self.modifier == ScopeEntryModifier::Mut
    }

    #[must_use]
    pub fn is_const(&self) -> bool {
        self.modifier == ScopeEntryModifier::Const
    }

    #[must_use]
    pub const fn is_mutated(&self) -> bool {
        self.mutated
    }
}

#[derive(Debug)]
pub struct MockScope(pub HashMap<String, MockScopeEntry>);

impl MockScope {
    #[must_use]
    pub fn new() -> Self { Self(HashMap::new()) }

    #[must_use]
    pub fn lookup(&self, name: &str) -> Option<&MockScopeEntry> {
        self.0.get(name)
    }
}

impl Default for MockScope {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct Context {
    pub tokens: Vec<(Token, Span)>,
    pub ast: Node,
    pub messages: Vec<AnalyzerMessage>,
    pub scopes: Vec<MockScope>,
    pub cache: Vec<(Source, String)>,
}

impl Context {
    #[must_use]
    pub fn from_tokens(cache: Vec<(Source, String)>, tokens: Vec<(Token, Span)>) -> Self {
        let ast = Node::parse(tokens.clone()).unwrap_or_else(|e| {
            for error in e {
                error.write(sources(cache.clone()), std::io::stderr());
            }

            std::process::exit(-1)
        });

        Self {
            tokens,
            ast,
            messages: Vec::new(),
            scopes: vec![MockScope::new()],
            cache,
        }
    }

    #[must_use]
    pub fn cache(&self) -> impl Cache<Source> {
        sources(self.cache.clone())
    }

    #[must_use]
    pub fn is_top_level(&self) -> bool {
        self.scopes.len() == 1
    }

    pub fn locals(&self) -> &MockScope {
        self.scopes.last().unwrap_or_else(|| unreachable!())
    }

    #[must_use]
    pub fn locals_mut(&mut self) -> &mut MockScope {
        self.scopes.last_mut().unwrap_or_else(|| unreachable!())
    }

    pub fn store_var(&mut self, name: String, entry: MockScopeEntry) {
        self.locals_mut().0.insert(name, entry);
    }

    #[must_use]
    pub fn lookup_var(&self, name: &String) -> Option<&MockScopeEntry> {
        for scope in self.scopes.iter().rev() {
            if let Some(entry) = scope.0.get(name) {
                return Some(entry);
            }
        }

        None
    }

    #[must_use]
    pub fn lookup_var_mut(&mut self, name: &String) -> Option<&mut MockScopeEntry> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(entry) = scope.0.get_mut(name) {
                return Some(entry);
            }
        }

        None
    }

    #[must_use]
    pub fn close_var_match(&self, name: &str) -> Option<String> {
        #[expect(
            clippy::cast_sign_loss,
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            reason = "Is not possible for indentifer to be this large"
        )]
        let threshold = (name.chars().count() as f64 * 0.14)
            .round()
            .max(2_f64) as usize;

        for scope in self.scopes.iter().rev() {
            for sample in scope.0.keys() {
                if get_levenshtein_distance(name, sample.as_str()) <= threshold {
                    return Some(sample.clone());
                }
            }
        }

        None
    }

    pub fn enter_scope(&mut self) {
        self.scopes.push(MockScope::new());
    }

    pub fn exit_scope(&mut self, analyzers: &AnalyzerSet, messages: &mut Vec<AnalyzerMessage>) {
        let scope = self.scopes.pop().unwrap();

        for entry in scope.0.into_values() {
            if analyzers.contains(&AnalyzerKind::UnnecessaryMutVariables)
                && entry.is_mut()
                && !entry.is_mutated()
            {
                messages.push(AnalyzerMessage::unnecessary_mut_variable(
                    entry.name.clone(),
                    entry.span.clone(),
                ));
            }

            if analyzers.contains(&AnalyzerKind::UnusedVariables)
                && !entry.used
                && !entry.name.starts_with("_")
            {
                messages.push(AnalyzerMessage::unused_variable(entry.name, entry.span));
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AnalyzerKind {
    /// [W000] Non-type identifier names should be snake_case
    NonSnakeCase,
    /// [W001] Type identifier names, such as classes or traits should be PascalCase
    NonPascalCase,
    /// [W002] Identifier names should contain only ASCII characters
    NonAscii,
    /// [W003] A variable or parameter was declared but never used
    UnusedVariables,
    /// [W004] A variable or parameter was declared as mutable, but is never mutated
    UnnecessaryMutVariables,
    /// [W005] Global mutable variables are highly discouraged
    GlobalMutableVariables,
    /// [E001] An identifier (e.g. a variable) could not be found in the current scope
    UnresolvedIdentifiers,
    /// [E002] A variable declared as `const` was redeclared later on
    RedeclaredConstVariables,
    /// [E003] An immutable variable was reassigned to
    ReassignedImmutableVariables,
}

impl AnalyzerKind {
    /// Returns a number 1 to 5 (inclusive) representing
    /// the servity of this specific type of warning.
    ///
    /// This can be used to ignore errors lower than a specific severity,
    /// or exit the analysis stage all together when a warning with a higher
    /// serverity is encountered.
    ///
    /// A higher number means a more severe warning.
    /// By default, the analyzer is set to ignore no errors and stop
    /// analysis at only level 5.
    ///
    /// If this is an error, return 0.
    #[must_use]
    pub const fn severity(&self) -> u8 {
        match self {
            Self::NonSnakeCase |
            Self::NonPascalCase |
            Self::NonAscii => 1,
            Self::UnusedVariables |
            Self::UnnecessaryMutVariables => 2,
            Self::UnresolvedIdentifiers |
            Self::RedeclaredConstVariables |
            Self::ReassignedImmutableVariables => 0,
            Self::GlobalMutableVariables => 4,
        }
    }

    /// References the error index
    #[must_use]
    pub const fn code(&self) -> u8 {
        match self {
            Self::NonSnakeCase => 0,
            Self::NonPascalCase | Self::UnresolvedIdentifiers => 1,
            Self::NonAscii | Self::RedeclaredConstVariables => 2,
            Self::UnusedVariables | Self::ReassignedImmutableVariables => 3,
            Self::UnnecessaryMutVariables => 4,
            Self::GlobalMutableVariables => 5,
        }
    }

    #[must_use]
    pub const fn is_warning(&self) -> bool {
        self.severity() != 0
    }

    #[must_use]
    pub const fn is_error(&self) -> bool {
        self.severity() == 0
    }

    pub const fn warn_level(&self) -> Result<u8, ()> {
        match self.severity() {
            0 => Err(()),
            n => Ok(n),
        }
    }
}

impl FromStr for AnalyzerKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "non-snake-case" => Self::NonSnakeCase,
            "non-pascal_case" => Self::NonPascalCase,
            "non-ascii" => Self::NonAscii,
            "unused-variables" => Self::UnusedVariables,
            "unnecessary-mut-variables" => Self::UnnecessaryMutVariables,
            "unresolved-identifiers" => Self::UnresolvedIdentifiers,
            "redeclared-const-variables" => Self::RedeclaredConstVariables,
            "reassigned-immutable-variables" => Self::ReassignedImmutableVariables,
            "global-mutable-variables" => Self::GlobalMutableVariables,
            _ => return Err(format!("invalid analyzer {:?}", s)),
        })
    }
}

impl std::fmt::Display for AnalyzerKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NonSnakeCase => "non-snake-case",
            Self::NonPascalCase => "non-pascal_case",
            Self::NonAscii => "non-ascii",
            Self::UnusedVariables => "unused-variables",
            Self::UnnecessaryMutVariables => "unnecessary-mut-variables",
            Self::UnresolvedIdentifiers => "unresolved-identifiers",
            Self::RedeclaredConstVariables => "redeclared-const-variables",
            Self::ReassignedImmutableVariables => "reassigned-immutable-variables",
            Self::GlobalMutableVariables => "global-mutable-variables",
        })
    }
}

#[derive(Clone, Debug)]
pub struct AnalyzerSet(pub HashSet<AnalyzerKind>);

impl AnalyzerSet {
    #[must_use]
    pub fn contains(&self, member: &AnalyzerKind) -> bool {
        self.0.contains(member)
    }

    #[must_use]
    pub fn none() -> Self {
        Self(HashSet::new())
    }

    #[must_use]
    pub fn all() -> Self {
        type A = AnalyzerKind;

        Self(HashSet::from([
            A::NonSnakeCase,
            A::NonPascalCase,
            A::NonAscii,
            A::UnusedVariables,
            A::UnnecessaryMutVariables,
            A::UnresolvedIdentifiers,
            A::RedeclaredConstVariables,
            A::ReassignedImmutableVariables,
            A::GlobalMutableVariables,
        ]))
    }

    pub fn from_disabled(disabled: HashSet<AnalyzerKind>) -> Self {
        Self(
            Self::default()
                .0
                .difference(&disabled)
                .map(|a| a.clone())
                .collect(),
        )
    }

    pub fn from_allowed_disabled(
        allowed: HashSet<AnalyzerKind>,
        disabled: HashSet<AnalyzerKind>,
    ) -> Self {
        Self(
            Self::default()
                .0
                .union(&allowed)
                .collect::<HashSet<_>>()
                .difference(&disabled.iter().collect())
                .map(|a| *a.clone())
                .collect::<HashSet<_>>(),
        )
    }
}

impl Default for AnalyzerSet {
    fn default() -> Self {
        Self::all()
    }
}

#[allow(unused_variables, reason = "`analyzers` will be used later")]
pub fn visit_expr(
    analyzers: &AnalyzerSet,
    ctx: &mut Context,
    messages: &mut Vec<AnalyzerMessage>,
    expr: Spanned<Expr>,
) -> Result<(), &'static str> {
    let span = expr.span();
    let expr = expr.into_node();

    match expr {
        Expr::Ident(s) => match ctx.lookup_var_mut(&s) {
            Some(e) => {
                e.used = true;
            }
            None => {
                let close_match = ctx.close_var_match(&s);

                messages.push(AnalyzerMessage::unresolved_identifier(
                    &s,
                    close_match,
                    span,
                ))
            }
        },
        _ => unimplemented!(),
    }

    Ok(())
}

pub fn visit_node(
    analyzers: &AnalyzerSet,
    ctx: &mut Context,
    messages: &mut Vec<AnalyzerMessage>,
    node: Spanned<Node>,
) -> Result<(), &'static str> {
    let span = node.span();
    let node = node.into_node();

    match node {
        Node::Module(m) => {
            for node in m {
                visit_node(analyzers, ctx, messages, node)?;
            }
        }
        Node::Declare {
            targets,
            r#mut,
            r#const,
            ..
        } => {
            type DeferEntry = (String, (), Span);

            fn recur(
                ctx: &Context,
                messages: &mut Vec<AnalyzerMessage>,
                target: Target,
                span: Span,
                tgt_span: Span,
                deferred: &mut Vec<DeferEntry>,
            ) {
                match target {
                    Target::Ident(s) => {
                        if let Some(entry) = ctx.lookup_var(&s) {
                            if entry.is_const() {
                                messages.push(AnalyzerMessage::redeclared_const_variable(
                                    &s,
                                    span.clone(),
                                ));
                            }
                        }

                        deferred.push((s, (), tgt_span));
                    }
                    Target::Array(targets) => {
                        for target in targets {
                            let (target, tgt_span) = target.into_node_span();

                            recur(
                                ctx,
                                messages,
                                target,
                                span.clone(),
                                tgt_span.clone(),
                                deferred,
                            );
                        }
                    }
                    _ => todo!(),
                };
            }

            // Assume there can only be one target
            let (target, tgt_span) = targets
                .first()
                .ok_or("multiple declaration targets unsupported")?
                .node_span();

            let modifier = match (r#mut, r#const) {
                (true, false) => ScopeEntryModifier::Mut,
                (false, true) => ScopeEntryModifier::Const,
                (false, false) => ScopeEntryModifier::None,
                (true, true) => unreachable!(),
            };

            let mut deferred = Vec::<DeferEntry>::new();

            recur(ctx, messages, target.clone(), span.clone(), tgt_span.clone(), &mut deferred);

            for (name, ty, tgt_span) in deferred {
                if analyzers.contains(&AnalyzerKind::NonSnakeCase) {
                    let snake = to_snake_case(&*name);

                    if name != snake {
                        messages.push(AnalyzerMessage::non_snake_case(
                            &name,
                            snake,
                            tgt_span.clone(),
                        ));
                    }
                }

                if analyzers.contains(&AnalyzerKind::GlobalMutableVariables)
                    && ctx.is_top_level()
                    && modifier == ScopeEntryModifier::Mut
                {
                    messages.push(AnalyzerMessage::global_mutable_variable(
                        name.clone(),
                        span.clone(),
                    ));
                }

                ctx.store_var(
                    name.clone(),
                    MockScopeEntry::new(name, ty, modifier, span.clone()),
                );
            }
        }
        Node::Assign { targets, .. } => {
            // Assume there can only be one target
            let (target, tgt_span) = targets
                .first()
                .ok_or("multiple assignment targets unsupported")?
                .node_span();

            match target {
                Target::Ident(s) => {
                    let entry = ctx.lookup_var_mut(s);

                    match entry {
                        Some(entry) => {
                            entry.mutated = true;

                            if entry.is_const() || !entry.is_mut() {
                                messages.push(AnalyzerMessage::reassigned_immutable_variable(
                                    s,
                                    span,
                                    entry.is_const(),
                                ));
                            }
                        }
                        None => {
                            let close_match = ctx.close_var_match(s);

                            messages.push(AnalyzerMessage::unresolved_identifier(
                                s,
                                close_match,
                                tgt_span.clone(),
                            ));
                            return Ok(());
                        }
                    }
                }
                Target::Array(_) => return Err("array assignments unsupported"),
                _ => todo!(),
            }
        }
        Node::Expr(expr) => visit_expr(analyzers, ctx, messages, expr)?,
        _ => unimplemented!(),
    }

    Ok(())
}

pub fn run_analysis(
    analyzers: AnalyzerSet,
    mut ctx: Context,
) -> Result<Vec<AnalyzerMessage>, &'static str> {
    let mut messages = Vec::new();
    let ast = std::mem::replace(&mut ctx.ast, Node::Module(Vec::new()));

    visit_node(
        &analyzers,
        &mut ctx,
        &mut messages,
        Spanned::new(
            ast,
            Span::default(), // Guaranteed to be a module, this is a placeholder
        ),
    )?;

    ctx.exit_scope(&analyzers, &mut messages);
    Ok(messages)
}
