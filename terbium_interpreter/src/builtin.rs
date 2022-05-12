use super::{Interpreter, NativeFunctionResult, WrappedTerbiumObject};

pub(crate) fn int_op_add(
    interpreter: &mut Interpreter,
    args: Vec<WrappedTerbiumObject>,
) -> NativeFunctionResult {
    if interpreter.is_instance(args[0], interpreter.clone().singletons.int_type) {}
}
