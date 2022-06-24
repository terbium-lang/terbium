use std::collections::HashMap;
use terbium::bytecode::EqComparableFloat;

#[test]
pub fn test_float_eq() {
    assert_eq!(EqComparableFloat(0.1), EqComparableFloat(0.1));

    let mut sample = HashMap::<EqComparableFloat, u8>::with_capacity(1);
    sample.insert(EqComparableFloat(0.1), 0);

    assert_eq!(sample.get(&EqComparableFloat(0.1)), Some(&0_u8));
}
