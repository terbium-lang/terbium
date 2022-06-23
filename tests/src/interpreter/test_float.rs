use terbium::TerbiumObject;

use super::interpret;

#[test]
fn test_float() {
    let res = interpret("2.0");

    assert_eq!(res, &TerbiumObject::Float(2.0))
}
