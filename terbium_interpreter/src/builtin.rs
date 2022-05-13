use super::{Interpreter, NativeFunctionResult, TerbiumExceptionType, TerbiumType, WrappedTerbiumObject};

macro_rules! bin_op {
    ($op:literal, $name:ident, $t:expr, $subject:pat => $body:block) => {
        pub(crate) fn $name(interpreter: &mut Interpreter, args: Vec<WrappedTerbiumObject>) -> NativeFunctionResult {
            if args.len() != 2 {
                Err(TerbiumExceptionType::SignatureError(
                    format!("{} op takes exactly 2 arguments", $op),
                ))?;
            }

            if interpreter.is_instance(&args[0], $t) && interpreter.is_instance(&args[1], $t) {
                return Ok(match (args[0].read().unwrap().clone().ty, args[1].read.unwrap().clone().ty) {
                    $subject => $body,
                });
            }

            Err(TerbiumExceptionType::TypeError(
                format!("expected {}", $t.name),
            ))
        }
    }
}

bin_op!(
    "add",
    int_op_add,
    interpreter.singletons.int_type,
    (TerbiumType::Int(a), TerbiumType::Int(b)) => interpreter.get_int(a + b)
);