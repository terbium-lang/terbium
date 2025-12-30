//! LLVM IR codegen from Terbium MIR.

mod aot;

pub use inkwell::{
    context::Context,
    module::Module,
    passes::PassBuilderOptions,
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
    fpm.initialize();

    aot::Compiler::compile(&context, &builder, &fpm, &module, functions);
    module
}
