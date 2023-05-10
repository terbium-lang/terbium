use common::span::{ProviderCache, Src};
use grammar::parser::Parser;
use grammar::span::Provider;
use hir::lower::AstLowerer;
use hir::{Hir, ModuleId};

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
        let mut parser = Parser::from_provider(&provider);
        let nodes = parser.consume_body_until_end();
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
                lowerer.resolve_top_level_types(ModuleId::from(Src::None))?;

                println!("HIR: {:#?}", lowerer.hir);
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
