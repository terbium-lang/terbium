use grammar::parser::Parser;

#[test]
fn test_parser() {
    let mut p = Parser::from_str("10");
    let expr = p.consume_expr().unwrap();
    println!("{0:?} => {0}", expr);
}
