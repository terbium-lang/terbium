use grammar::{
    parser::Parser,
    span::{Provider, Src},
};

#[test]
fn test_parser() {
    let provider = Provider::new(Src::None, r##"-[6 + (1, 2, 3, 5, 4, 5)]"##);

    let mut parser = Parser::from_provider(&provider);
    let expr = parser.to_expr();

    match expr {
        Ok(ref e) => {
            println!("{}", e);
        }
        Err(ref err) => println!("{:?}", err),
    }
    println!("{:#?}", expr);
}
