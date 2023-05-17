use common::span::{ProviderCache, Src};
use grammar::parser::Parser;
use grammar::span::Provider;
use hir::infer::TypeLowerer;
use hir::lower::AstLowerer;
use hir::Expr::Ident;
use hir::{Hir, ItemId, ModuleId};

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
        let mut parser = Parser::from_provider(&provider);
        let nodes = parser.consume_body_until_end();
        println!("parse: {:?}", start.elapsed());
        // writeln!(file)?;

        let cache = ProviderCache::from_providers([&provider]);

        match nodes {
            Ok(nodes) => {
                println!("{nodes:#?}");
                for node in &nodes {
                    println!("{node}");
                    // for line in node.to_string().lines() {
                    //     writeln!(file, "// {line}")?;
                    // }
                }

                let mut lowerer = AstLowerer::new(nodes);

                let start = std::time::Instant::now();
                match lowerer.resolve_module(ModuleId::from(Src::None)) {
                    Ok(_) => {
                        println!(
                            "=== [ HIR ({:?} to lower) ] ===\n\n{}",
                            start.elapsed(),
                            lowerer.hir
                        );

                        let mut ty_lowerer = TypeLowerer::new(lowerer.hir);
                        for node in ty_lowerer.hir.scopes.clone().into_values() {
                            for mut child in node.children {
                                match ty_lowerer.lower_node(&mut child) {
                                    Ok(()) => (),
                                    Err(error) => error.write(&cache, &mut std::io::stdout())?,
                                }
                            }
                        }
                    }
                    Err(error) => {
                        error.write(&cache, &mut std::io::stdout())?;
                    }
                };
            }
            Err(errors) => {
                for error in errors {
                    error.write(&cache, &mut std::io::stdout())?;
                    // writeln!(file, "// {}", error.info)?;
                }
            }
        }

        // file.flush()?;
        previous = std::fs::metadata("test.trb")?.modified()?;
    }
}
