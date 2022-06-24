pub mod util;

use std::collections::{HashMap, HashSet};
use std::fmt::Formatter;
use terbium_grammar::{Expr, Node, ParseInterface, Source, Span, Spanned, Target, Token};
use terbium_grammar::error::{Hint, HintAction};
use util::to_snake_case;

use std::io::Write;
use std::str::FromStr;
use ariadne::{Cache, sources};
use chumsky::chain::Chain;
use crate::util::get_levenshtein_distance;

#[derive(Clone, Debug, PartialEq)]
pub enum AnalyzerMessageKind {
    Info,
    Alert(AnalyzerKind),
}

#[derive(Debug, PartialEq)]
pub struct AnalyzerMessage {
    kind: AnalyzerMessageKind,
    message: String,
    label: Option<String>,
    span: Span,
    hint: Option<Hint>,
}

impl AnalyzerMessage {
    pub fn non_snake_case(name: String, counterpart: String, span: Span) -> Self {
        Self {
            kind: AnalyzerMessageKind::Alert(AnalyzerKind::NonSnakeCase),
            message: "non-type identifier names should be snake_case".to_string(),
            label: Some(format!("{:?} is not snake_case", name)),
            span,
            hint: Some(Hint {
                message: format!("rename to {:?}", counterpart),
                action: HintAction::Replace(counterpart),
            })
        }
    }

    pub fn unresolved_identifier(name: String, close_match: Option<String>, span: Span) -> Self {
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

    pub fn redeclared_const_variable(name: String, span: Span) -> Self {
        Self {
            kind: AnalyzerMessageKind::Alert(AnalyzerKind::RedeclaredConstVariables),
            message: "cannot redeclare variable declared as `const`".to_string(),
            label: Some(format!("variable {:?} was declared as `const`", name)),
            span,
            hint: Some(Hint {
                message: "declare with `let` instead".to_string(),
                action: HintAction::None,
            })
        }
    }

    pub fn reassigned_immutable_variable(name: String, span: Span, was_const: bool) -> Self {
        Self {
            kind: AnalyzerMessageKind::Alert(AnalyzerKind::ReassignedImmutableVariables),
            message: format!("cannot reassign to {}", if was_const {
                "variable declared as `const`"
            } else {
                "immutable variable"
            }),
            label: Some(format!("attempted to reassign to variable {:?}, but it was {}", name, if was_const {
                "declared as `const`"
            } else {
                "never declared as `mut`"
            })),
            span,
            hint: Some(Hint {
                message: "make variable mutable by declaring with `let mut` instead".to_string(),
                action: HintAction::None,
            })
        }
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
}

impl MockScopeEntry {
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
}

#[derive(Debug)]
pub struct MockScope(pub HashMap<String, MockScopeEntry>);

impl MockScope {
    pub fn new() -> Self { Self(HashMap::new()) }

    pub fn lookup(&self, name: String) -> Option<&MockScopeEntry> {
        self.0.get(&name)
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

    pub fn cache(&self) -> impl Cache<Source> {
        sources(self.cache.clone())
    }

    pub fn locals(&self) -> &MockScope {
        self.scopes.last().unwrap()
    }

    pub fn locals_mut(&mut self) -> &mut MockScope {
        self.scopes.last_mut().unwrap()
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

    pub fn lookup_var_mut(&mut self, name: &String) -> Option<&mut MockScopeEntry> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(entry) = scope.0.get_mut(name) {
                return Some(entry);
            }
        }

        None
    }

    pub fn close_var_match(&self, name: &String) -> Option<String> {
        let threshold = (name.chars().len() as f64 * 0.14)
            .round()
            .max(2_f64) as usize;

        for scope in self.scopes.iter().rev() {
            for sample in scope.0.keys() {
                if get_levenshtein_distance(name.as_str(), sample.as_str()) <= threshold {
                    return Some(sample.clone());
                }
            }
        }

        None
    }

    pub fn enter_scope(&mut self) {
        self.scopes.push(MockScope::new());
    }

    pub fn exit_scope(&mut self) {
        self.scopes.pop();
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
    /// [E001] An identifier (e.g. a variable) could not be found in the current scope
    UnresolvedIdentifiers,
    /// [E002] A variable declared as `const` was redeclared later on
    RedeclaredConstVariables,
    /// [E003] An immutable variable was reassigned to
    ReassignedImmutableVariables,
    /// [W005] Global mutable variables are highly discouraged
    GlobalMutableVariables,
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
            Self::NonSnakeCase => 1,
            Self::NonPascalCase => 1,
            Self::NonAscii => 1,
            Self::UnusedVariables => 2,
            Self::UnnecessaryMutVariables => 2,
            Self::UnresolvedIdentifiers => 0,
            Self::RedeclaredConstVariables => 0,
            Self::ReassignedImmutableVariables => 0,
            Self::GlobalMutableVariables => 4,
        }
    }

    #[must_use]
    pub const fn is_warning(&self) -> bool {
        self.severity() == 0
    }

    #[must_use]
    pub const fn is_error(&self) -> bool {
        self.severity() != 0
    }

    pub fn warn_level(&self) -> Result<u8, ()> {
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
    pub fn contains(&self, member: &AnalyzerKind) -> bool {
        self.0.contains(member)
    }

    pub fn none() -> Self {
        Self(HashSet::new())
    }

    pub fn all() -> Self {
        type A = AnalyzerKind;

        Self(HashSet::from([
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
        Self(Self::default().0.difference(&disabled).map(|a| a.clone()).collect())
    }

    pub fn from_allowed_disabled(allowed: HashSet<AnalyzerKind>, disabled: HashSet<AnalyzerKind>) -> Self {
        Self(
            Self::default().0
                .union(&allowed)
                .collect::<HashSet<_>>()
                .difference(&disabled.iter().collect())
                .map(|a| *a.clone())
                .collect::<HashSet<_>>()
        )
    }
}

impl Default for AnalyzerSet {
    fn default() -> Self {
        Self::all()
    }
}

pub fn visit_expr(
    analyzers: &AnalyzerSet,
    ctx: &mut Context,
    messages: &mut Vec<AnalyzerMessage>,
    expr: Spanned<Expr>,
) -> Result<(), &'static str> {
    let span = expr.span();
    let expr = expr.into_node();

    match expr {
        _ => unimplemented!(),
    }
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
        Node::Module(m) => for node in m {
            visit_node(analyzers, ctx, messages, node)?;
        },
        Node::Declare { targets, r#mut, r#const, .. } => {
            // Assume there can only be one target
            let (target, span) = targets
                .first()
                .ok_or("multiple declaration targets unsupported")?
                .node_span();

            let modifier = match (r#mut, r#const) {
                (true, false) => ScopeEntryModifier::Mut,
                (false, true) => ScopeEntryModifier::Const,
                (false, false) => ScopeEntryModifier::None,
                (true, true) => unreachable!(),
            };

            type DeferEntry = (String, (), Span);
            let mut deferred = Vec::<DeferEntry>::new();

            fn recur(
                ctx: &Context,
                messages: &mut Vec<AnalyzerMessage>,
                target: Target,
                span: Span,
                deferred: &mut Vec<DeferEntry>,
            ) {
                match target {
                    Target::Ident(s) => {
                        if let Some(entry) = ctx.lookup_var(&s) {
                            if entry.is_const() {
                                messages.push(AnalyzerMessage::redeclared_const_variable(
                                    s.clone(),
                                    span.clone(),
                                ));
                            }
                        }

                        deferred.push((s, (), span));
                    },
                    Target::Array(targets) => for target in targets {
                        let (target, span) = target.into_node_span();

                        recur(ctx, messages, target, span, deferred);
                    }
                    _ => todo!(),
                };
            }

            recur(ctx, messages, target.clone(), span.clone(), &mut deferred);

            for (name, ty, span) in deferred {
                if analyzers.contains(&AnalyzerKind::NonSnakeCase) {
                    let snake = to_snake_case(&*name);

                    if name != snake {
                        messages.push(AnalyzerMessage::non_snake_case(
                            name.clone(),
                            snake,
                            span.clone(),
                        ));
                    }
                }

                ctx.store_var(name.clone(), MockScopeEntry { name, ty, modifier });
            }
        }
        Node::Assign { targets, .. } => {
            // Assume there can only be one target
            let (target, span) = targets
                .first()
                .ok_or("multiple assignment targets unsupported")?
                .node_span();

            match target {
                Target::Ident(s) => {
                    let entry = ctx.lookup_var(s);

                    match entry {
                        Some(entry) => {
                            if entry.is_const() || !entry.is_mut() {
                                messages.push(AnalyzerMessage::reassigned_immutable_variable(
                                    s.to_string(),
                                    span.clone(),
                                    entry.is_const(),
                                ));
                            }
                        }
                        None => {
                            let close_match = ctx.close_var_match(s);

                            messages.push(AnalyzerMessage::unresolved_identifier(
                                s.clone(),
                                close_match,
                                span.clone(),
                            ));
                            return Ok(());
                        }
                    }
                },
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

    visit_node(&analyzers, &mut ctx, &mut messages, Spanned::new(
        ast,
        Span::default(), // Guaranteed to be a module, this is a placeholder
    ))?;

    Ok(messages)
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

        a.analyze_string(
            Source::default(),
            String::from(
                "
            func camelCase() {
                let notSnakeCase = 5;
            }
        ",
            ),
        );

        a.write(std::io::stdout());
    }
}
