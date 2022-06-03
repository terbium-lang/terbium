//! The interpreter for Terbium.

use crate::mem::{BlockAllocError, BlockSize, Heap, Mark, Memory, Mutator, MutatorView, ScopedPtr};
use terbium_grammar::Expr;

mod mem;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TerbiumType {
    Int,
    Float,
    Bool,
    String,
    Function,
    Array,
    Null,
    CallFrameList,
    Thread,
    Symbol,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TerbiumObjectHeader {
    mark: Mark,
    ty: TerbiumType,
    size_class: BlockSize,
    size: u32,
}

#[derive(Clone, Copy)]
pub enum TerbiumValue<'s> {
    Int(ScopedPtr<'s, u128>),
    Float(ScopedPtr<'s, f64>),
    String(ScopedPtr<'s, String>),
    Bool(ScopedPtr<'s, bool>),
    Function(ScopedPtr<'s, TerbiumFunction>), // TODO
    Array(ScopedPtr<'s, Vec<TerbiumValue<'s>>>),
    Null,
}

#[derive(Debug)]
pub struct Interpreter {
    pub memory: Memory,
    mutator: Mutat,
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            memory: Memory::new(),
        }
    }

    pub fn eval(&self, expr: Expr) -> TerbiumValue {
        match expr {
            Expr::Integer(i) => TerbiumValue::Int(ScopedPtr::new()),
        }
    }
}

impl Mutator<()> for Interpreter {
    fn run(&self, mem: &MutatorView, input: I) -> Result<O, BlockAllocError> {}
}
