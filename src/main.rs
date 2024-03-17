use codegen::{
    compile_llvm, CodeModel, FileType, InitializationConfig, OptimizationLevel, RelocMode, Target,
    TargetMachine,
};
use common::span::{Span, Src};
use diagnostics::write::DiagnosticWriter;
use grammar::parser::Parser;
use grammar::span::Provider;
use hir::check::TypeChecker;
use hir::infer::TypeLowerer;
use hir::lower::AstLowerer;
use hir::ModuleId;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::panic::set_hook(Box::new(|info| {
        eprintln!("{}", info);
    }));

    let interval = std::time::Duration::from_millis(500);
    let mut previous = std::fs::metadata("test.trb")?.modified()?;

    loop {
        std::thread::sleep(interval);
        let new_time = std::fs::metadata("test.trb")?.modified()?;
        if new_time <= previous {
            continue;
        }

        println!("===== changes detected... =====");

        let provider = Provider::read_from_file("test.trb")?;
        let start = std::time::Instant::now();
        let mut full = std::time::Duration::default();
        let mut parser = Parser::from_provider(&provider);
        let nodes = parser.consume_body_until_end();
        full += start.elapsed();
        println!("parse: {:?}", start.elapsed());
        // writeln!(file)?;

        let mut dwriter = DiagnosticWriter::new();
        dwriter.add_provider(provider.clone());

        match nodes {
            Ok(nodes) => {
                // println!("{nodes:#?}");
                // for node in &nodes {
                //     println!("{node}");
                //     // for line in node.to_string().lines() {
                //     //     writeln!(file, "// {line}")?;
                //     // }
                // }

                let mut lowerer = AstLowerer::new(nodes);

                let start = std::time::Instant::now();
                match lowerer.resolve_module(
                    ModuleId::from(Src::None),
                    provider.eof().merge(Span::begin(Src::None)),
                ) {
                    Ok(_) => {
                        full += start.elapsed();
                        // println!(
                        //     "=== [ HIR ({:?} to lower) ] ===\n\n{}",
                        //     start.elapsed(),
                        //     lowerer.hir
                        // );
                        println!("hir: {:?}", start.elapsed());

                        let start = std::time::Instant::now();
                        let mut ty_lowerer = TypeLowerer::new(lowerer.hir.clone());
                        match ty_lowerer.lower_module(ModuleId::from(Src::None)) {
                            Ok(_) => {
                                full += start.elapsed();
                                // println!(
                                //     "=== [ THIR ({:?} to type) ] ===\n\n{}",
                                //     start.elapsed(),
                                //     ty_lowerer.thir
                                // );
                                println!("thir: {:?}", start.elapsed());

                                let start = std::time::Instant::now();
                                let mut typeck = TypeChecker::from_lowerer(&mut ty_lowerer);
                                let mut table = typeck.take_table();

                                typeck.check_module(ModuleId::from(Src::None), &mut table);

                                full += start.elapsed();
                                println!(
                                    "=== [ THIR ({:?} to check) ] ===\n\n{}",
                                    start.elapsed(),
                                    typeck.lower.thir
                                );
                                println!("typeck: {:?}", start.elapsed());
                                for error in typeck.lower.errors.drain(..) {
                                    dwriter.write_diagnostic(
                                        &mut std::io::stdout(),
                                        error.into_diagnostic(),
                                    )?;
                                }

                                let start = std::time::Instant::now();
                                let mut mir_lowerer =
                                    mir::Lowerer::from_thir(typeck.lower.thir.clone());
                                mir_lowerer.lower_module(ModuleId::from(Src::None));

                                full += start.elapsed();
                                // println!(
                                //     "=== [ MIR ({:?} to lower) ] ===\n\n{}",
                                //     start.elapsed(),
                                //     mir_lowerer.mir
                                // );
                                println!("mir: {:?}", start.elapsed());
                                for error in mir_lowerer.errors.drain(..) {
                                    dwriter.write_diagnostic(
                                        &mut std::io::stdout(),
                                        error.into_diagnostic(),
                                    )?;
                                }

                                let start = std::time::Instant::now();

                                let ctx = codegen::Context::create();
                                let module = compile_llvm(&ctx, mir_lowerer.mir.functions);

                                full += start.elapsed();
                                // println!("=== [ LLVM IR ({:?} to compile) ] ===", start.elapsed());
                                // println!("{}", module.to_string());
                                println!("llvm: {:?}", start.elapsed());
                                println!("total cmptime: {full:?}");
                                type F = unsafe extern "C" fn() -> i32;
                                let engine = module
                                    .create_jit_execution_engine(OptimizationLevel::Aggressive)?;
                                if let Ok(f) = unsafe { engine.get_function::<F>("test") } {
                                    println!("evaluating test()...");
                                    println!("-> {}", unsafe { f.call() });
                                }

                                module.write_bitcode_to_path(&*PathBuf::from("out.bc"));

                                //let engine = module.create_execution_engine()?;
                                //let entrypoint = module.get_function("__root").unwrap();

                                let start = std::time::Instant::now();
                                Target::initialize_native(&InitializationConfig::default())?;
                                let target = Target::get_first().unwrap();
                                let machine = target
                                    .create_target_machine(
                                        &TargetMachine::get_default_triple(),
                                        &TargetMachine::get_host_cpu_name().to_string(),
                                        &TargetMachine::get_host_cpu_features().to_string(),
                                        OptimizationLevel::Aggressive,
                                        RelocMode::Default,
                                        CodeModel::Default,
                                    )
                                    .unwrap();

                                let buffer =
                                    machine.write_to_memory_buffer(&module, FileType::Assembly)?;
                                let assembly = buffer.as_slice().to_vec();

                                full += start.elapsed();
                                // println!(
                                //     "=== {} ASM ({:?} to compile, {:?} total) ===",
                                //     TargetMachine::get_default_triple()
                                //         .as_str()
                                //         .to_string_lossy(),
                                //     start.elapsed(),
                                //     full
                                // );
                                // println!("{}", String::from_utf8_lossy(&assembly));
                                machine.write_to_file(
                                    &module,
                                    FileType::Object,
                                    &*PathBuf::from("out.o"),
                                )?;
                            }
                            Err(error) => {
                                dwriter.write_diagnostic(
                                    &mut std::io::stdout(),
                                    error.into_diagnostic(),
                                )?;
                            }
                        }

                        for warning in ty_lowerer.warnings {
                            dwriter.write_diagnostic(
                                &mut std::io::stdout(),
                                warning.into_diagnostic(),
                            )?;
                        }
                        for error in ty_lowerer.errors {
                            dwriter.write_diagnostic(
                                &mut std::io::stdout(),
                                error.into_diagnostic(),
                            )?;
                        }
                    }
                    Err(error) => {
                        dwriter
                            .write_diagnostic(&mut std::io::stdout(), error.into_diagnostic())?;
                    }
                };
            }
            Err(errors) => {
                for error in errors {
                    dwriter.write_diagnostic(&mut std::io::stdout(), error.into_diagnostic())?;
                    // error.write(&cache, &mut std::io::stdout())?;
                    // writeln!(file, "// {}", error.info)?;
                }
            }
        }

        // file.flush()?;
        previous = std::fs::metadata("test.trb")?.modified()?;
    }
}
