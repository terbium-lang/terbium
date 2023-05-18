use common::span::{Provider, Span, Src};
use diagnostics::write::DiagnosticWriter;
use diagnostics::{Action, Diagnostic, Fix, Label, Section, Severity};

#[test]
fn test_diagnostics() -> std::io::Result<()> {
    let provider = Provider::read_from_file("/Users/jay3332/Projects/Terbium/test.trb")?;
    let span = |start, end| Span::new(provider.src(), start, end);
    let diagnostic = Diagnostic::new(Severity::Error(3), "mismatched types")
        .with_section(
            Section::new()
                .with_label(Label::at(span(4, 10)).with_message("label 1"))
                .with_label(Label::at(span(15, 16)).with_message("label 2"))
                .with_label(Label::at(span(13, 14)).with_message("label 3"))
                .with_label(Label::at(span(18, 19)).with_message("label 4"))
                .with_note("sample note"),
        )
        .with_fix(
            Fix::new(Action::Replace(span(4, 10), "new_string".to_string()))
                .with_message("rename the variable")
                .with_label("renamed here"),
        )
        .with_end(Some("note"), "expected `string`\n   found `int`");

    let mut writer = DiagnosticWriter::new();
    writer.add_provider(provider);

    writer.write_diagnostic(&mut std::io::stderr(), diagnostic)?;

    Ok(())
}
