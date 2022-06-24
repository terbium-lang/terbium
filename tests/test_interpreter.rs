use terbium::bytecode::{Instruction, Program};
use terbium::interpreter::{DefaultInterpreter, TerbiumObject};

#[test]
fn test_interpreter() {
    let program = Program::from_iter([
        Instruction::LoadInt(1).into(),
        Instruction::LoadInt(1).into(),
        Instruction::BinOpAdd.into(),
        Instruction::Halt.into(),
    ]);
    let mut interpreter = DefaultInterpreter::default();
    interpreter.run_bytecode(&program);

    assert_eq!(interpreter.ctx.pop(), &TerbiumObject::Integer(2));
}
