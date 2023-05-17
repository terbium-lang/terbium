use common::span::{Provider, Span, Src};
use diagnostics::{Diagnostic, Label, Section, Severity};
use diagnostics::write::DiagnosticWriter;

#[test]
fn test_diagnostics() -> std::io::Result<()> {
    let provider = Provider::read_from_file("../test.trb")?;
    let span = |start, end| Span::new(provider.src(), start, end);
    let diagnostic = Diagnostic::new(Severity::Error(3), "unknown variable")
        .with_section(Section::over(span(8, 9)).with_label(
            Label::at(span(8, 9)).with_message("unknown variable `a`")
        ).with_note("sample note"))
        .with_end(Some("note"), "sample");

    let mut writer = DiagnosticWriter::new();
    writer.add_provider(provider);

    writer.write_diagnostic(&mut std::io::stderr(), diagnostic)?;

    Ok(())
}
