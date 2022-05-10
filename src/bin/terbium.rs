use terbium::{AstError, AstNode, AstParseInterface};

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[clap(name = "terbium")]
#[clap(version, about, long_about = None)]
/// The Terbium command-line interface.
struct Args {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Transforms the Terbium source code into a formatted abstract syntax tree.
    #[clap(arg_required_else_help = true)]
    Ast {
        /// The input file containing Terbium source code.
        file: Option<String>,

        /// The direct source code to parse. Cannot be used with the file argument.
        #[clap(short, long)]
        code: Option<String>,

        /// Whether to output the AST in a "prettier" and more readable format.
        #[clap(short, long)]
        pretty: bool,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    match args.command {
        Command::Ast { file, code, pretty } => {
            let ast = match (file, code) {
                (Some(file), None) => std::fs::read_to_string(file)?,
                (None, Some(code)) => code,
                (Some(_), Some(_)) => Err("must provide only one of file or code")?,
                (None, None) => Err("must provide one of file or code")?,
            };

            let (node, errors) = match AstNode::from_string(ast) {
                Ok(t) => t,
                Err(e) => {
                    e.iter().for_each(AstError::print);
                    return Err("syntax does not match grammar".into());
                }
            };
            if pretty {
                println!("{:#?}", node);
                println!("Errors: {:#?}", errors);
            } else {
                println!("{:?}", node);
                println!("Errors: {:?}", errors);
            }
        }
    }

    Ok(())
}
