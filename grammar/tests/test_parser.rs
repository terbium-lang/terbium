use grammar::parser::Parser;

#[test]
fn test_parser() {
    let mut p = Parser::new(r##"1 % 5 ** 2 ** (a.b ** 3) * 2 / 3"##).unwrap();
    let expr = p.consume_expr().unwrap();
    println!("{0:?} => {0}", expr);
}
