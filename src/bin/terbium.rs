use std::io::{stderr, Write};
use std::path::PathBuf;
use std::process::exit;

use ariadne::sources;
use clap::{Parser, Subcommand};
use terbium::{AstNode, BcTransformer};
use terbium_grammar::{ParseInterface, Source};
use terbium_interpreter::DefaultInterpreter;

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
        #[clap(parse(from_os_str))]
        file: Option<PathBuf>,

        /// The direct source code to parse. Cannot be used with the file argument.
        #[clap(short, long)]
        code: Option<String>,

        /// Whether to output the AST in a "prettier" and more readable format.
        #[clap(short, long)]
        pretty: bool,
    },
    /// Disassembles the Terbium source code into Terbium bytecode.
    #[clap(arg_required_else_help = true)]
    Dis {
        /// The input file containing Terbium source code.
        #[clap(parse(from_os_str))]
        file: Option<PathBuf>,

        /// The direct source code to parse. Cannot be used with the file argument.
        #[clap(short, long)]
        code: Option<String>,

        /// Whether to output the bytecode in its raw bytes format.
        /// Not including this flag will output the bytecode in a readable format.
        ///
        /// Useful if you want to compile Terbium source code into bytecode and save
        /// it in a separate file.
        #[clap(short, long)]
        raw: bool,
    },
    /// Interprets the Terbium source code expression, pops the last object on the stack,
    /// and writes the object represented in repr/debug form into standard output.
    ///
    /// Errors encountered during evaluation will be written to standard error.
    #[clap(arg_required_else_help = true)]
    Eval {
        /// The input file containing Terbium source code.
        #[clap(parse(from_os_str))]
        file: Option<PathBuf>,

        /// The direct source code to parse. Cannot be used with the file argument.
        #[clap(short, long)]
        code: Option<String>,
    },
}

fn run_ast<N>(file: Option<PathBuf>, code: Option<String>) -> Result<N, Box<dyn std::error::Error>>
where
    N: ParseInterface,
{
    let (code, src) = match (file, code) {
        (Some(file), None) => (
            std::fs::read_to_string(file.clone())?,
            Source::from_path(file),
        ),
        (None, Some(code)) => (code, Source::default()),
        (Some(_), Some(_)) => return Err("must provide only one of file or code".into()),
        (None, None) => return Err("must provide one of file or code".into()),
    };

    Ok(
        N::from_string(src.clone(), code.clone()).unwrap_or_else(|e| {
            for error in e {
                let cache = sources::<Source, String, _>(vec![(src.clone(), code.clone())]);

                error.write(cache, stderr());
            }

            exit(-1)
        }),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    match args.command {
        Command::Ast { file, code, pretty } => {
            let node = run_ast::<AstNode>(file, code)?;

            if pretty {
                println!("{:#?}", node);
            } else {
                println!("{:?}", node);
            }
        }
        Command::Dis { code, file, raw } => {
            let body = run_ast(file, code)?;

            let mut transformer = BcTransformer::default();
            transformer.interpret_body(None, body);

            let mut program = transformer.program();
            program.resolve();

            let mut stdout = std::io::stdout();
            if raw {
                let bytes = program.bytes();
                stdout.write_all(bytes.as_slice())?;
            } else {
                program.dis(&mut stdout)?;
            }
        }
        Command::Eval { code, file } => {
            let body = run_ast(file, code)?;

            let mut transformer = BcTransformer::default();
            transformer.interpret_body(None, body);

            let mut program = transformer.program();
            program.resolve();

            let mut interpreter = DefaultInterpreter::default();
            interpreter.run_bytecode(&program);

            let popped = interpreter.ctx.pop_ref();
            let popped = interpreter.ctx.store.resolve(popped);
            println!("{}", interpreter.get_object_repr(popped));
        }
    }

    Ok(())
}
