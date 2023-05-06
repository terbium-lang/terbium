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

pub mod ast;
pub mod error;
pub mod parser;
pub mod token;

pub use common::span;
