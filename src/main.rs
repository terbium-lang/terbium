use common::span::ProviderCache;
use grammar::parser::Parser;
use grammar::span::Provider;

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
            Ok(ref nodes) => {
                println!("{nodes:#?}");
                for node in nodes {
                    println!("{node}");
                    // for line in node.to_string().lines() {
                    //     writeln!(file, "// {line}")?;
                    // }
                }
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
