//! LLVM IR codegen from Terbium MIR.

mod aot;

pub use inkwell::{
    context::Context,
    module::Module,
    targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine},
    values::FunctionValue,
    OptimizationLevel,
};

use inkwell::passes::PassManager;
use mir::{Func, LookupId};
use std::collections::HashMap;

pub fn compile_llvm(context: &Context, functions: HashMap<LookupId, Func>) -> Module {
    let module = context.create_module("root");
    let builder = context.create_builder();

    // Create FPM
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

    aot::Compiler::compile(&context, &builder, &fpm, &module, functions);
    module
}
