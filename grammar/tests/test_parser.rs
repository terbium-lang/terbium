use grammar::parser::Parser;

#[test]
fn test_parser() {
    let mut p = Parser::new(r##"-(6 + [1, 2, 3, 5, 4], 5)"##).unwrap();
    let expr = p.consume_expr();

    if let Ok(ref e) = expr { println!("{}", e); }
    println!("{:#?}", expr);
    println!("{:#?}", p.errors);
}
