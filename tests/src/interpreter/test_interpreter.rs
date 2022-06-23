use terbium::interpreter::{DefaultInterpreter, TerbiumObject};
use terbium::bytecode::{Instruction, Program};
use terbium::grammar::Body;

#[test]
fn test_interpreter() {
    let program = Program::from_iter([
        Instruction::LoadInt(1),
        Instruction::LoadInt(1),
        Instruction::BinOpAdd,
        Instruction::Halt,
    ]);
    let mut interpreter = DefaultInterpreter::default();
    interpreter.run_bytecode(&program);

    assert_eq!(interpreter.ctx.pop(), &TerbiumObject::Integer(2));
}
