use terbium::bytecode::Interpreter as Transformer;
use terbium::grammar::{Body, ParseInterface, Source};
use terbium::interpreter::{DefaultInterpreter, TerbiumObject};

pub fn interpret(code: &str) -> TerbiumObject {
    let body = Body::from_string(Source::default(), code.to_string()).unwrap_or_else(|e| {
        panic!("tokenization error: {:?}", e);
    });
    let mut transformer = Transformer::default();
    transformer.interpret_body(None, body);

    let mut program = transformer.program();
    program.resolve();

    let mut interpreter = DefaultInterpreter::default();
    interpreter.run_bytecode(&program);

    interpreter.ctx.pop().clone()
}
