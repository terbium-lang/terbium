use inkwell::context::Context;
use inkwell::passes::PassManager;
use terbium_grammar::{Source, ParseInterface, Spanned, Span, Body};
use terbium::Compiler;

#[test]
fn test_compiler() {
    let sample = String::from("-1");
    let body = Body::from_string(Source::default(), sample).unwrap();

    let ctx = Context::create();
    let module = ctx.create_module("tmp");
    let builder = ctx.create_builder();

    let fpm = PassManager::create(&module);

    fpm.add_instruction_combining_pass();
    fpm.add_reassociate_pass();
    fpm.add_gvn_pass();
    fpm.add_cfg_simplification_pass();
    fpm.add_basic_alias_analysis_pass();
    fpm.add_promote_memory_to_register_pass();
    fpm.add_instruction_combining_pass();
    fpm.add_reassociate_pass();
    fpm.initialize();

    let func = Compiler::compile(
        &ctx,
        &builder,
        &fpm,
        &module,
        body,
    ).unwrap();

    func.print_to_stderr();
}
