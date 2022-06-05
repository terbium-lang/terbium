//! Crate for Terbium byte-code/intermediate represenation.
//!
//! Because Terbium can be directly interpreted or be compiled into LLVM IR,
//! there must be a representation of Terbium code that can provide an easy
//! entrypoint to both.

mod interpreter;
mod util;

use std::mem::size_of;
use std::ops::Add;
pub use util::EqComparableFloat;

pub type Addr = u32;
pub type ObjectPtr = u32;

#[derive(Clone, Debug)]
pub enum Instruction {
    // Constants mapped to a lookup table
    LoadInt(i128), // TODO: this takes 16 bytes in raw bytecode representation, not the best
    LoadFloat(EqComparableFloat),
    LoadString(String),
    LoadBool(bool),
    Load(ObjectPtr),

    // Operations
    UnOpPos,
    UnOpNeg,

    BinOpAdd,
    BinOpSub,
    BinOpMul,
    BinOpDiv,
    BinOpTrueDiv, // Div keeps type, while TrueDiv doesn't. e.g. Div(5, 2) -> 2 but TrueDiv(5, 2) -> 2.5
    BinOpPow,
    
    BinOpBitOr,
    BinOpBitXor,
    BinOpBitAnd,
    BinOpBitNot, // Unary
    
    // Logical Operations
    OpEq,
    OpNe,
    OpLt,
    OpLe,
    OpGt,
    OpGe,
    OpLogicalOr,
    OpLogicalAnd,
    OpLogicalNot, // Unary

    // Variables
    PushLocal, // Push the item on the stack to locals
    PopLocal(usize), // Pop and don't push to stack
    LoadLocal(usize), // "Duplicate" (as a reference) and push to stack

    // Functions
    MakeFunc(usize), // Field 0 is the amount of items to take from the stack as parameters
    CallFunc,

    Jump(Addr),
    JumpIf(Addr),

    Pop,
    Ret,
    Halt,
}

impl Instruction {
    /// Return a i8 representing the change in the count of elements in the stack.
    pub fn stack_diff(&self) -> i8 {
        match self {
            Self::LoadInt(_) => 1,
            Self::Pop => -1,

            _ => todo!(),
        }
    }
    
    pub fn size(&self) -> u8 {
        1_u8 + match self {
            Self::LoadInt(_) => size_of::<i128>(),
            Self::LoadFloat(_) => size_of::<f64>(),
            Self::LoadString(s) => s.len(),
            Self::LoadBool(_) => 1,
            Self::LoadLocal(_) | Self::PopLocal(_) | Self::MakeFunc(_) => size_of::<usize>(),
            Self::Load(_) => size_of::<ObjectPtr>(),
            Self::Jump(_) | Self::JumpIf(_) => size_of::<Addr>(),
            _ => 0,
        }
    }
    
    pub fn to_instr_id(&self) -> u8 {
        match self {
            Self::LoadInt(_) => 0,
            Self::LoadFloat(_) => 1,
            Self::LoadString(_) => 2,
            Self::LoadBool(_) => 3,
            Self::Load(_) => 4,
            Self::UnOpPos => 5,
            Self::UnOpNeg => 6, 
            Self::BinOpAdd => 7,
            Self::BinOpSub => 8,
            Self::BinOpMul => 9,
            Self::BinOpDiv => 10,
            Self::BinOpTrueDiv => 11,
            Self::BinOpPow => 12,
            Self::BinOpBitOr => 13,
            Self::BinOpBitXor => 14,
            Self::BinOpBitAnd => 15,
            Self::BinOpBitNot => 16,
            Self::OpEq => 17,
            Self::OpNe => 18,
            Self::OpLt => 19,
            Self::OpLe => 20,
            Self::OpGt => 21,
            Self::OpGe => 22,
            Self::OpLogicalOr => 23,
            Self::OpLogicalAnd => 24,
            Self::OpLogicalNot => 25,
            Self::PushLocal => 26,
            Self::PopLocal(_) => 27,
            Self::LoadLocal(_) => 28,
            Self::MakeFunc(_) => 29,
            Self::CallFunc => 30,
            Self::Jump(_) => 31,
            Self::JumpIf(_) => 32,
            Self::Pop => 33,
            Self::Ret => 34,
            Self::Halt => 35,
        }
    }

    pub fn from_instr_id(id: u8) -> Self {
        todo!()
    }
}

#[derive(Debug)]
pub struct Program {
    inner: Vec<Instruction>,
}

macro_rules! consume {
    ($split:expr => $($i:ident),* => $then:expr) => {{
        $(
            let $i = $split.next().ok_or("invalid bytecode".to_string())?;
        )*
        $then
    }}
}

macro_rules! parse {
    ($e:expr) => {{
        $e.parse().map_err(|_| "invalid number")?
    }};
    ($e:expr, $msg:literal) => {{
        $e.parse().map_err(|_| $msg)?
    }};
}

impl Program {
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    pub fn inner(&self) -> impl Iterator<Item = &Instruction> {
        self.inner.iter()
    }
    
    pub fn bytes(&self) -> Vec<u8> {
        type I = Instruction;
        let mut bytes = Vec::new();
        
        for instr in self.inner() {
            bytes.push(instr.to_instr_id());

            bytes.extend_from_slice(match instr {
                I::LoadInt(i) => &i.to_ne_bytes(),
                I::LoadFloat(f) => &f.0.to_ne_bytes(),
                I::LoadString(s) => &s.as_bytes(),
                I::LoadBool(b) => &[if b { 0 } else { 1 }],
                I::LoadLocal(i) | I::PopLocal(i) | I::MakeFunc(i) => &i.to_ne_bytes(),
                I::Jump(a) | I::JumpIf(a) => &a.to_ne_bytes(),
                _ => &[],
            })
        }
        
        bytes
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        todo!()
    }
}

impl FromIterator<Instruction> for Program {
    fn from_iter<I: IntoIterator<Item = Instruction>>(iter: I) -> Self {
        Self {
            inner: iter.into_iter().collect(),
        }
    }
}
