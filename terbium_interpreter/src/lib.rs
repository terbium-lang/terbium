//! The interpreter for Terbium.

use crate::mem::{BlockSize, Mark, ScopedPtr};

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
