use terbium::analyzer::BulkAnalyzer;
use terbium::grammar::Source;

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
