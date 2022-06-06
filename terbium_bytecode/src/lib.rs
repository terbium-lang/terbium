//! Crate for Terbium byte-code/intermediate represenation.
//!
//! Because Terbium can be directly interpreted or be compiled into LLVM IR,
//! there must be a representation of Terbium code that can provide an easy
//! entrypoint to both.

mod interpreter;
mod util;

use std::collections::HashMap;
use std::mem::size_of;
use std::ops::Add;
pub use util::EqComparableFloat;

pub type ObjectPtr = u32;
pub type AddrRepr = u32;

#[derive(Copy, Clone, Debug, PartialOrd, PartialEq, Hash)]
pub enum Addr {
    Absolute(AddrRepr),
    Procedure(AddrRepr),
    Offset(AddrRepr, AddrRepr),
}

#[derive(Clone, Debug)]
pub enum Instruction {
    // Constants mapped to a lookup table
    LoadInt(i128), // TODO: this takes 16 bytes in raw bytecode representation, not the best
    LoadFloat(EqComparableFloat),
    LoadString(String),
    LoadBool(bool),
    Load(ObjectPtr),
    LoadFrame(ObjectPtr),

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
    CallFunc(usize), // Field 0 is the number of items

    Jump(Addr),
    JumpIf(Addr),
    JumpIfElse(Addr, Addr),

    Pop,
    Ret,
    RetNull,
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
            Self::Load(_) | Self::LoadFrame(_) => size_of::<ObjectPtr>(),
            Self::Jump(_) | Self::JumpIf(_) => size_of::<AddrRepr>(),
            Self::JumpIfElse(_, _) => size_of::<AddrRepr>() * 2,
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
            Self::CallFunc(_) => 30,
            Self::Jump(_) => 31,
            Self::JumpIf(_) => 32,
            Self::JumpIfElse(_, _) => 33,
            Self::Pop => 34,
            Self::Ret => 35,
            Self::RetNull => 36,
            Self::Halt => 37,
            Self::LoadFrame(_) => 38,
        }
    }

    pub fn from_instr_id(id: u8) -> Self {
        todo!()
    }
}

#[derive(Debug)]
pub struct Program {
    inner: Vec<Instruction>,
    procedures: Vec<Vec<Instruction>>,
}

impl Program {
    pub fn new() -> Self {
        Self { inner: Vec::new(), procedures: Vec::new() }
    }

    pub fn inner(&self) -> impl Iterator<Item = &Instruction> {
        self.inner.iter()
    }

    pub fn create_procedure(&mut self) -> AddrRepr {
        self.procedures.push(Vec::new());

        (self.procedures.len() - 1).into()
    }

    pub fn push(&mut self, procedure: Option<AddrRepr>, instr: Instruction) {
        if let Some(procedure) = procedure {
            return self.procedures
                .get_mut(procedure)
                .expect("no procedure there")
                .push(instr);
        }

        self.inner.push(instr);
    }

    pub fn current_addr(&self, procedure: Option<AddrRepr>) -> Addr {
        match self.next_addr(procedure) {
            Addr::Absolute(i) => Addr::Absolute(i - 1),
            Addr::Offset(p, i) => Addr::Offset(p, i - 1),
            _ => unreachable!(),
        }
    }

    pub fn next_addr(&self, procedure: Option<AddrRepr>) -> Addr {
        if let Some(proc) = procedure {
            return Addr::Offset(
                proc,
                self.procedures
                    .get(proc)
                    .expect("procedure not found")
                    .len()
                    .into()
            );
        }

        Addr::Absolute(self.inner.len().into())
    }

    /// Resolves all procedures to their absolute address
    pub fn resolve(&mut self) -> &Self {
        if self.inner.last() != Some(&Instruction::Halt) {
            self.inner.push(Instruction::Halt);
        }

        // Lookup of proc -> absolute
        let mut lookup: HashMap<AddrRepr, AddrRepr> = HashMap::new();
        for (i, mut proc) in self.procedures.iter().enumerate() {
            lookup.insert(i, self.inner.len() as AddrRepr);

            self.inner.extend(proc);
        }

        for instr in self.inner {
            if let Instruction::Jump(addr) | Instruction::JumpIf(addr) = instr {
                match addr {
                    Addr::Procedure(proc) => *addr = Addr::Absolute(
                        lookup.get(&proc).expect("unknown procedure").clone()
                    ),
                    Addr::Offset(proc, offset) => *addr = Addr.Absolute(
                        lookup.get(&proc).expect("unknown procedure").clone() + offset
                    ),
                    _ => (),
                }
            }
        }

        self
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
                I::Jump(a) | I::JumpIf(a) => match a {
                    Addr::Absolute(p) => &p.to_ne_bytes(),
                    _ => panic!("procedures must be resolved prior to conversion"),
                },
                I::JumpIfElse(a, b) => match (a, b) {
                    (Addr::Absolute(a), Addr::Absolute(b)) => {
                        &[a.to_ne_bytes(), b.to_ne_bytes()].concat()
                    }
                    _ => panic!("procedures must be resolved prior to conversion"),
                }
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
            procedures: Vec::new(),
        }
    }
}
