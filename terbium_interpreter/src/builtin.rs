use super::{Interpreter, NativeFunctionResult, TerbiumExceptionType, TerbiumType, WrappedTerbiumObject};

pub(crate) fn int_op_add(
    interpreter: &mut Interpreter,
    args: Vec<WrappedTerbiumObject>,
) -> NativeFunctionResult {
    if args.len() != 2 {
        Err(TerbiumExceptionType::SignatureError("add op takes exactly 2 arguments"))?;
    }

    let int_type = interpreter.clone().singletons.int_type;
    if interpreter.is_instance(&args[0], &int_type)
        && interpreter.is_instance(&args[1], &int_type) 
    {
        return Ok(match (args[0].read().unwrap().clone().ty, args[1].read().unwrap().clone().ty) {
            (TerbiumType::Int(a), TerbiumType::Int(b)) => interpreter.get_int(a + b),
            _ => unreachable!(),
        });
    }

    Err(TerbiumExceptionType::TypeError("expected int"))
}
