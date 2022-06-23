use terbium::analyzer::BulkAnalyzer;

#[test]
fn test_analysis() {
    let mut a = BulkAnalyzer::new_with_analyzers(
        vec!["non-snake-case"]
            .iter()
            .map(ToString::to_string)
            .collect(),
    );

    a.analyze_string(String::from(
        "
        func camelCase() {
            let notSnakeCase = 5;
        }
    ",
    ));

    a.write(std::io::stdout());
}
