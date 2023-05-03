use grammar::parser::Parser;
use grammar::span::{Provider, Src};
use std::fs::OpenOptions;

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
        let mut file = OpenOptions::new().read(true).open("test.trb")?;

        let mut contents = String::new();
        std::io::Read::read_to_string(&mut file, &mut contents)?;

        let provider = Provider::new(Src::None, &contents);
        let mut parser = Parser::from_provider(&provider);
        let nodes = parser.consume_body_until_end();
        // writeln!(file)?;

        println!("> {contents}");
        match nodes {
            Ok(ref nodes) => {
                for node in nodes {
                    println!("{node}");
                    // for line in node.to_string().lines() {
                    //     writeln!(file, "// {line}")?;
                    // }
                }
            }
            Err(ref errors) => {
                for error in errors {
                    eprintln!("[error] {}", error.info);
                    // writeln!(file, "// {}", error.info)?;
                }
            }
        }

        // file.flush()?;
        drop(file);
        previous = std::fs::metadata("test.trb")?.modified()?;
    }
}
