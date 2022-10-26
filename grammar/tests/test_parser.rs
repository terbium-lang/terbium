use grammar::parser::Parser;

#[test]
fn test_parser() {
    for token in grammar::TokenReader::new(r#"~5 ~"1""#) {
        println!("{:?}", token);
    }
    let mut p = Parser::from_str(r##"-!1"##);
    let expr = p.consume_expr().unwrap();
    println!("{0:?} => {0}", expr);
}
