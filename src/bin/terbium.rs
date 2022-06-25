#![feature(lint_reasons)]

use std::io::{stderr, Write};
use std::path::PathBuf;
use std::process::exit;
use std::time::Instant;

use ariadne::sources;
use clap::{Parser, Subcommand};
use terbium::{AstNode, AstToken, BcTransformer};
use terbium_analyzer::{run_analysis, AnalyzerMessageKind, AnalyzerSet, Context};
use terbium_grammar::{ParseInterface, Source, Span};
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
    #[clap(aliases(&["evaluate", "e"]))]
    Eval {
        /// The input file containing Terbium source code.
        #[clap(parse(from_os_str))]
        file: Option<PathBuf>,

        /// The direct source code to parse. Cannot be used with the file argument.
        #[clap(short, long)]
        code: Option<String>,
    },
    /// Analyzes the Terbium source code and checks for any potential runtime errors.
    #[clap(arg_required_else_help = true)]
    #[clap(alias("analyze"))]
    Check {
        /// The input file containing Terbium source code.
        #[clap(parse(from_os_str))]
        file: Option<PathBuf>,

        /// The direct source code to analyze. Cannot be used with the file argument.
        #[clap(short, long)]
        code: Option<String>,
    },
}

type ParseError = Vec<(Source, String)>;

fn run_ast<N>(
    file: Option<PathBuf>,
    code: Option<String>,
) -> Result<(N, ParseError), Box<dyn std::error::Error>>
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

    let s = vec![(src.clone(), code.clone())];

    Ok((
        N::from_string(src, code).unwrap_or_else(|e| {
            for error in e {
                let cache = sources::<Source, String, _>(s.clone());

                error.write(cache, stderr());
            }

            exit(-1)
        }),
        s,
    ))
}

fn analyze<N>(file: Option<PathBuf>, code: Option<String>) -> Result<N, Box<dyn std::error::Error>>
where
    N: ParseInterface,
{
    let (tokens, src) = run_ast::<Vec<(AstToken, Span)>>(file, code)?;

    let ctx = Context::from_tokens(src.clone(), tokens.clone());
    let analyzers = AnalyzerSet::default();

    let instant = Instant::now();
    let messages = run_analysis(&analyzers, ctx)?;
    #[allow(
        clippy::cast_precision_loss,
        reason = "The timing is provided on a best-effort basis, precision loss is tolerable."
    )]
    let elapsed = instant.elapsed().as_micros() as f64 / 1000_f64;

    let mut should_exit = false;

    let count = messages.len();
    let mut info_count = 0;
    let mut warning_count = 0;
    let mut error_count = 0;

    for message in messages {
        match message.kind {
            AnalyzerMessageKind::Info => info_count += 1,
            AnalyzerMessageKind::Alert(k) => {
                if k.is_error() {
                    should_exit = true;
                    error_count += 1;
                } else {
                    warning_count += 1;
                }
            }
        }

        message.write(sources(src.clone()), stderr());
    }

    eprintln!("completed analysis in {} ms", elapsed);
    eprintln!(
        "{} message{} ({} info, {} warning{}, {} error{})\n",
        count,
        if count == 1 { "" } else { "s" },
        info_count,
        warning_count,
        if warning_count == 1 { "" } else { "s" },
        error_count,
        if error_count == 1 { "" } else { "s" },
    );

    if should_exit {
        exit(-1);
    }

    // we can unwrap here since analysis unwraps for us
    Ok(N::parse(tokens).unwrap())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    match args.command {
        Command::Ast { file, code, pretty } => {
            let node = run_ast::<AstNode>(file, code)?.0;

            if pretty {
                println!("{:#?}", node);
            } else {
                println!("{:?}", node);
            }
        }
        Command::Dis { file, code, raw } => {
            let body = analyze(file, code)?;

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
        Command::Eval { file, code } => {
            let body = analyze(file, code)?;

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
        Command::Check { code, file } => {
            println!("analyzing... (analysis will be streamed into stderr)");
            analyze::<Vec<(AstToken, Span)>>(file, code)?;
        }
    }

    Ok(())
}
