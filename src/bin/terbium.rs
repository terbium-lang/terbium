use std::io::Write;
use std::path::PathBuf;
use terbium::{AstBody, AstError, AstNode, AstParseInterface, BcTransformer};

use clap::{Parser, Subcommand};
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    match args.command {
        Command::Ast { file, code, pretty } => {
            let ast = match (file, code) {
                (Some(file), None) => std::fs::read_to_string(file)?,
                (None, Some(code)) => code,
                (Some(_), Some(_)) => return Err("must provide only one of file or code".into()),
                (None, None) => return Err("must provide one of file or code".into()),
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
                if !errors.is_empty() {
                    eprintln!("Errors: {:#?}", errors);
                }
            } else {
                println!("{:?}", node);
                if !errors.is_empty() {
                    eprintln!("Errors: {:?}", errors);
                }
            }
        }
        Command::Dis { code, file, raw } => {
            let code = match (file, code) {
                (Some(file), None) => std::fs::read_to_string(file)?,
                (None, Some(code)) => code,
                (Some(_), Some(_)) => return Err("must provide only one of file or code".into()),
                (None, None) => return Err("must provide one of file or code".into()),
            };

            let (body, errors) = match AstBody::from_string(code) {
                Ok(b) => b,
                Err(e) => {
                    e.iter().for_each(AstError::print);
                    return Err("syntax does not match grammar".into());
                }
            };

            let mut transformer = BcTransformer::new();
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

            if !errors.is_empty() {
                eprintln!("Errors: {:#?}", errors);
            }
        }
        Command::Eval { code, file } => {
            let code = match (file, code) {
                (Some(file), None) => std::fs::read_to_string(file)?,
                (None, Some(code)) => code,
                (Some(_), Some(_)) => return Err("must provide only one of file or code".into()),
                (None, None) => return Err("must provide one of file or code".into()),
            };

            let (body, errors) = match AstBody::from_string(code) {
                Ok(b) => b,
                Err(e) => {
                    e.iter().for_each(AstError::print);
                    return Err("syntax does not match grammar".into());
                }
            };

            let mut transformer = BcTransformer::new();
            transformer.interpret_body(None, body);

            let mut program = transformer.program();
            program.resolve();

            let mut interpreter = DefaultInterpreter::new();
            interpreter.run_bytecode(program);

            let popped = interpreter.ctx.pop_ref();
            let popped = interpreter.ctx.store.resolve(popped);
            println!("{}", interpreter.get_object_repr(popped));

            if !errors.is_empty() {
                eprintln!("Errors: {:#?}", errors);
            }
        }
    }

    Ok(())
}
