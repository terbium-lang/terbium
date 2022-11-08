use grammar::parser::Parser;

#[test]
fn test_parser() {
    let mut p = Parser::new(r##"-6"##).unwrap();
    let expr = p.consume_expr().unwrap();
    println!("{0:?} => {0}", expr);
}
