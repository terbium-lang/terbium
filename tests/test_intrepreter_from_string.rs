use terbium::interpreter::TerbiumObject;

mod interpreter;
use interpreter::interpret;

#[test]
fn test_interpreter_from_string() {
    let res = interpret(
        r#"
        if 1 + 1 == 3 {
            0
        } else if 1 + 1 == 2 {
            1
        } else {
            2
        }
    "#,
    );

    assert_eq!(res, TerbiumObject::Integer(1));
}
