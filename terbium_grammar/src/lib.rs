#![feature(trait_alias)]

pub mod ast;
pub mod error;
pub mod token;

pub use crate::token::Token;
pub use crate::error::*;
