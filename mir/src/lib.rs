//! MIR, or Mid-level Intermediate Representation, is a typed, control-flow graph-based
//! representation of a Terbium program.
//!
//! From the typeck stage, type-validated THIR is lowered into MIR. During MIR analysis,
//! Terbium-specific optimizations (that could not otherwise be done by LLVM or other codegen) are
//! performed. Pattern validations and exhaustiveness checks are also performed during this stage.
//!
//! MIR is the last stage of the compiler that preserves all span, type, and debuginfo. MIR can be
//! transformed from and to MIR bytecode, and can be interpreted using the `interpreter` crate,
//! usually referred to as the MIR Interpreter or "MIRI". Bytecode can also be used to aid
//! incremental compilation as prior lowerings and parsing stages can be skipped. MIR also conforms
//! to *single static assignment* (SSA) form, similar to LLVM IR.
//!
//! You might have noticed MIR closely resembles LLVM IR. This is because MIR is designed to be
//! as close to LLVM IR as possible while still maintaining Terbium-specific details.
//!
//! If a lower-level representation is requested, MIR can be lowered into a lower-level IR such as
//! LLVM IR to be compiled into machine code.

#![feature(map_try_insert)]

mod lower;

pub use hir::{
    typed::{
        BinaryIntIntrinsic, LocalEnv, Ty, /* TODO: monomorphize types */
        UnaryIntIntrinsic,
    },
    IntSign, IntWidth, ModuleId, PrimitiveTy,
};
pub use lower::Lowerer;

use common::span::Spanned;
use hir::{FloatWidth, Ident, ItemId, ScopeId};
use indexmap::IndexMap;
use std::{
    collections::HashMap,
    fmt::{self, Display, Formatter},
};

#[inline]
fn write_comma_sep<T: Display>(f: &mut Formatter, iter: impl Iterator<Item = T>) -> fmt::Result {
    let mut iter = iter.peekable();
    while let Some(c) = iter.next() {
        write!(f, "{c}")?;
        if iter.peek().is_some() {
            write!(f, ", ")?;
        }
    }
    Ok(())
}

pub type TypedHir = hir::Hir<hir::infer::InferMetadata>;
pub type HirFunc = hir::Func<hir::infer::InferMetadata>;
pub type BlockMap = IndexMap<BlockId, Vec<Spanned<Node>>>;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlockId(pub Ident);

impl BlockId {
    /// The entry block of a function.
    pub fn entry() -> Self {
        Self("entry".into())
    }
}

impl Display for BlockId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

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

impl LocalId {
    pub fn name(&self) -> String {
        format!("{}.{}", self.1, self.0)
    }
}

/// The MIR representation of a Terbium program.
#[derive(Clone, Debug, Default)]
pub struct Mir {
    /// All procedures, including the top-level and main procedures.
    pub functions: HashMap<ItemId, Func>,
}

impl Display for Mir {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.functions
            .values()
            .map(|func| writeln!(f, "{func}"))
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct Func {
    pub name: ItemId,
    pub params: Vec<(Ident, Ty)>,
    pub ret_ty: Ty,
    pub blocks: BlockMap,
}

impl Display for Func {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "func {}(", self.name)?;
        write_comma_sep(
            f,
            self.params
                .iter()
                .map(|(p, ty)| format!("%local.{p}: {ty}")),
        )?;
        writeln!(f, ") -> {} {{", self.ret_ty)?;
        for (id, block) in &self.blocks {
            writeln!(f, "{id}:")?;
            for node in block {
                writeln!(f, "    {node}")?;
            }
        }
        write!(f, "}}")
    }
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
            }
            Self::Bool(b) => write!(f, "{b}"),
            Self::Char(c) => write!(f, "c{c:?}"),
            Self::Tuple(t) => {
                write!(f, "(")?;
                write_comma_sep(f, t.iter())?;
                write!(f, ")")
            }
            Self::Array(a) => {
                write!(f, "[")?;
                write_comma_sep(f, a.iter())?;
                write!(f, "]")
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntIntrinsic {
    Unary(UnaryIntIntrinsic, Box<Spanned<Expr>>),
    Binary(BinaryIntIntrinsic, Box<Spanned<Expr>>, Box<Spanned<Expr>>),
}

impl Display for IntIntrinsic {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unary(op, target) => write!(f, "({op}{target})"),
            Self::Binary(op, lhs, rhs) => write!(f, "({lhs} {op} {rhs})"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BoolIntrinsic {
    Not(Box<Spanned<Expr>>),
    And(Box<Spanned<Expr>>, Box<Spanned<Expr>>),
    Or(Box<Spanned<Expr>>, Box<Spanned<Expr>>),
    Xor(Box<Spanned<Expr>>, Box<Spanned<Expr>>),
}

impl Display for BoolIntrinsic {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Not(target) => write!(f, "(!{target})"),
            Self::And(lhs, rhs) => write!(f, "({lhs} && {rhs})"),
            Self::Or(lhs, rhs) => write!(f, "({lhs} || {rhs})"),
            Self::Xor(lhs, rhs) => write!(f, "({lhs} ^ {rhs})"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expr {
    Constant(Constant),
    Local(LocalId),
    IntIntrinsic(IntIntrinsic, IntSign, IntWidth),
    BoolIntrinsic(BoolIntrinsic),
    Call(ItemId, Vec<Spanned<Self>>),
}

impl Display for Expr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Constant(c) => write!(f, "{c}"),
            Self::Local(l) => write!(f, "{l}"),
            Self::IntIntrinsic(i, sign, width) => {
                write!(f, "<{}{}>{i}", sign.type_name(), *width as usize)
            }
            Self::BoolIntrinsic(b) => write!(f, "{b}"),
            Self::Call(i, args) => {
                write!(f, "{}(", i)?;
                write_comma_sep(f, args.iter())?;
                write!(f, ")")
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum Node {
    Expr(Spanned<Expr>),
    // declare temporary register
    Register(LocalId, Spanned<Expr>, Ty),
    // initialize stack-allocated local (alloca)
    // this is always done for mutable or late-init locals, and can usually be optimized away
    // with the "memory to register promotion" pass.
    Local(LocalId, Ty),
    // store to local
    Store(LocalId, Box<Spanned<Expr>>),
    // llvm: br label %block
    Jump(BlockId),
    // llvm: br i1 %cond, label %true, label %false
    Branch(Spanned<Expr>, BlockId, BlockId),
    // llvm for Some(expr): ret ty expr
    // llvm for None: ret void
    Return(Option<Spanned<Expr>>),
}

impl Node {
    #[inline]
    pub const fn is_terminator(&self) -> bool {
        matches!(self, Self::Jump(_) | Self::Branch(..) | Self::Return(_))
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Expr(e) => write!(f, "{e}"),
            Self::Register(l, e, ty) => write!(f, "register {l}: {ty} = {e}"),
            Self::Local(l, ty) => write!(f, "alloca {l}: {ty}"),
            Self::Store(l, e) => write!(f, "store {l} = {e}"),
            Self::Jump(b) => write!(f, "jump {b}"),
            Self::Branch(cond, then, els) => write!(f, "branch {cond} {then} {els}"),
            Self::Return(e) => write!(
                f,
                "return {}",
                e.as_ref()
                    .map(ToString::to_string)
                    .unwrap_or("void".to_string())
            ),
        }
    }
}
