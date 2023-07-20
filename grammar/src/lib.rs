//! Defines and parses the grammar of Terbium.
//!
//! # Example
//! ```
//! use common::span::{Src, Provider};
//! use grammar::Parser;
//!
//! // Define the source code to parse
//! let provider = Provider::new(Src::None, r#"func main() { println("Hello, world!"); }"#);
//!
//! // Create a parser over the provider
//! let mut parser = Parser::from_provider(&provider);
//!
//! // Parse the source code into an AST
//! let ast = parser.consume_body_until_end();
//! ```

#![allow(
    clippy::uninlined_format_args,
    reason = "editor will complain about unused variables otherwise"
)]
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    reason = "not really necessary"
)]
#![allow(
    clippy::module_name_repetitions,
    reason = "items are not usually accessed directly"
)]
#![feature(const_trait_impl)]
#![feature(lint_reasons)]
#![feature(never_type)]
#![feature(let_chains)]
#![feature(trait_alias)]
#![feature(lazy_cell)]
#![feature(unboxed_closures)]
#![feature(fn_traits)]

pub mod ast;
pub mod error;
pub mod parser;
pub mod token;
pub mod tree;

pub use common::span;
pub use parser::Parser;
