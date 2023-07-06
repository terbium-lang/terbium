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
use mir::{Mir, ModuleId};

pub fn compile_llvm<'ctx>(
    context: &'ctx Context,
    mir: &Mir,
    module_id: ModuleId, /*, options: CompileOptions*/
) -> Module<'ctx> {
    let module = context.create_module(&module_id.to_string());
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

    for func in mir
        .functions
        .iter()
        .filter_map(|(id, func)| id.0.eq(&module_id).then_some(func))
    {
        aot::Compiler::compile(&context, &builder, &fpm, &module, func);
    }
    module
}
