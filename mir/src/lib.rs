//! MIR, or Mid-level Intermediate Representation, is a typed, control-flow graph-based
//! representation of a Terbium program.
//!
//! From the typeck stage, type-validated THIR is lowered into MIR. During MIR analysis,
//! Terbium-specific optimizations (that could not otherwise be done by LLVM or other codegen) are
//! performed. Pattern validations and exhaustiveness checks are also performed during this stage.
//!
//! MIR is the last stage of the compiler that preserves all span, "type", and debuginfo. MIR can be
//! transformed from and to MIR bytecode, and can be interpreted using the `interpreter` crate,
//! usually referred to as the MIR Interpreter or "MIRI". Bytecode can also be used to aid
//! incremental compilation as prior lowerings and parsing stages can be skipped.
//!
//! If a lower-level representation is requested, MIR can be lowered into a lower-level IR such as
//! LLVM IR to be compiled into machine code.

#![feature(map_try_insert)]

mod lower;

use std::{collections::HashMap, fmt::{self, Display, Formatter}};
use common::span::Spanned;
use hir::{
    typed::{BinaryIntIntrinsic, LocalEnv, UnaryIntIntrinsic},
    IntWidth, IntSign, Ident, FloatWidth, ItemId, ScopeId,
};

pub type TypedHir = hir::Hir<hir::infer::InferMetadata>;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlockId(pub Ident);

impl From<ScopeId> for BlockId {
    fn from(value: ScopeId) -> Self {
        Self(format!("_bb{}", value.0).into())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LocalId(pub Ident, pub LocalEnv);

impl Display for LocalId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "%{}.{}", self.1, self.0)
    }
}

pub type BlockMap = HashMap<BlockId, Vec<Spanned<Node>>>;

/// The MIR representation of a Terbium program.
#[derive(Clone, Debug, Default)]
pub struct Mir {
    /// All procedures, including the top-level and main procedures.
    pub functions: HashMap<ItemId, Func>,
}

#[derive(Clone, Debug)]
pub struct Func {
    pub params: Vec<Ident>,
    pub blocks: BlockMap,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Constant {
    Never,
    Void,
    Int(u128, IntSign, IntWidth),
    Float(u64 /* raw bits */, FloatWidth),
    Bool(bool),
    Char(char),
    Tuple(Vec<Self>),
    Array(Vec<Self>),
}

impl Display for Constant {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        #[inline]
        fn write_comma_sep<'a>(f: &mut Formatter, iter: impl Iterator<Item = &'a Constant>) -> fmt::Result {
            let mut iter = iter.peekable();
            while let Some(c) = iter.next() {
                write!(f, "{}", c)?;
                if iter.peek().is_some() {
                    write!(f, ", ")?;
                }
            }
            Ok(())
        }

        match self {
            Self::Never => write!(f, "never"),
            Self::Void => write!(f, "void"),
            Self::Int(i, sign, width) => write!(f, "({}{}) {i}", sign.type_name(), *width as usize),
            Self::Float(fl, width) => {
                write!(f, "({}) ", *width as usize)?;
                match width {
                    FloatWidth::Float32 => write!(f, "{}", f32::from_bits(*fl as u32)),
                    _ => write!(f, "{}", f64::from_bits(*fl)),
                }
            },
            Self::Bool(b) => write!(f, "{b}"),
            Self::Char(c) => write!(f, "c'{c}'"),
            Self::Tuple(t) => {
                write!(f, "(")?;
                write_comma_sep(f, t.iter())?;
                write!(f, ")")
            },
            Self::Array(a) => {
                write!(f, "[")?;
                write_comma_sep(f, a.iter())?;
                write!(f, "]")
            },
        }
    }
}

#[derive(Clone, Debug)]
pub enum IntIntrinsic {
    Unary(UnaryIntIntrinsic, Box<Spanned<Expr>>),
    Binary(BinaryIntIntrinsic, Box<Spanned<Expr>>, Box<Spanned<Expr>>),
}

#[derive(Clone, Debug)]
pub enum BoolIntrinsic {
    Not(Box<Spanned<Expr>>),
    And(Box<Spanned<Expr>>, Box<Spanned<Expr>>),
    Or(Box<Spanned<Expr>>, Box<Spanned<Expr>>),
    Xor(Box<Spanned<Expr>>, Box<Spanned<Expr>>),
}

#[derive(Clone, Debug)]
pub enum Expr {
    Constant(Constant),
    Local(LocalId),
    Assign(LocalId, Box<Spanned<Self>>),
    IntIntrinsic(IntIntrinsic, IntSign, IntWidth),
    BoolIntrinsic(BoolIntrinsic),
    Call(ItemId, Vec<Spanned<Self>>),
}

#[derive(Clone, Debug)]
pub enum Node {
    Expr(Spanned<Expr>),
    // declare initialized with a constant (no alloca needed)
    Declare(LocalId, Spanned<Constant>),
    // initialize local
    Init(LocalId),
    // llvm: br label %block
    Jump(BlockId),
    // llvm: br i1 %cond, label %true, label %false
    Branch(Spanned<Expr>, BlockId, BlockId),
    // llvm for Some(expr): ret ty expr
    // llvm for None: ret void
    Return(Option<Spanned<Expr>>),
}
