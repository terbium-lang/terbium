use clap::Parser;
use terbium::run;

/// Terbium CLI.
#[derive(Debug, Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {}

fn main() {
    let _ = Args::parse(); // TODO: Handle CLI arguments.

    println!("你好曹先生");

    run(); // TODO: Pass in options from CLI;
}
