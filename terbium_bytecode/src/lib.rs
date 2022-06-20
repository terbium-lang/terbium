//! Crate for Terbium byte-code/intermediate represenation.
//!
//! Because Terbium can be directly interpreted or be compiled into LLVM IR,
//! there must be a representation of Terbium code that can provide an easy
//! entrypoint to both.

mod interpreter;
mod util;

use std::collections::HashMap;
use std::io::Write;
use std::mem::size_of;

pub use interpreter::Interpreter;
pub use util::EqComparableFloat;

pub type AddrRepr = u32;

#[derive(Copy, Clone, Debug, PartialOrd, PartialEq, Hash)]
pub enum Addr {
    Absolute(AddrRepr),
    Procedure(AddrRepr),
    Offset(AddrRepr, AddrRepr),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Instruction {
    // Constants mapped to a lookup table
    LoadInt(i128), // TODO: this takes 16 bytes in raw bytecode representation, not the best
    LoadFloat(EqComparableFloat),
    LoadString(String),
    LoadBool(bool),
    Load(usize),
    LoadFrame(usize),

    // Scope
    EnterScope,
    ExitScope,

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
    UnOpBitNot, // Unary

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
    Store(usize),
    StoreVar(usize),
    StoreMutVar(usize),
    StoreConstVar(usize),
    LoadVar(usize),

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
            Self::LoadVar(_)
            | Self::StoreVar(_)
            | Self::StoreMutVar(_)
            | Self::StoreConstVar(_)
            | Self::MakeFunc(_)
            | Self::Load(_)
            | Self::Store(_)
            | Self::LoadFrame(_) => size_of::<usize>(),
            Self::Jump(_) | Self::JumpIf(_) => size_of::<AddrRepr>(),
            Self::JumpIfElse(_, _) => size_of::<AddrRepr>() * 2,
            _ => 0,
        } as u8
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
            Self::UnOpBitNot => 16,
            Self::OpEq => 17,
            Self::OpNe => 18,
            Self::OpLt => 19,
            Self::OpLe => 20,
            Self::OpGt => 21,
            Self::OpGe => 22,
            Self::OpLogicalOr => 23,
            Self::OpLogicalAnd => 24,
            Self::OpLogicalNot => 25,
            Self::LoadVar(_) => 26,
            Self::StoreVar(_) => 27,
            Self::StoreMutVar(_) => 28,
            Self::StoreConstVar(_) => 29,
            Self::MakeFunc(_) => 30,
            Self::CallFunc(_) => 31,
            Self::Jump(_) => 32,
            Self::JumpIf(_) => 33,
            Self::JumpIfElse(_, _) => 34,
            Self::Pop => 35,
            Self::Ret => 36,
            Self::RetNull => 37,
            Self::Halt => 38,
            Self::LoadFrame(_) => 39,
            Self::Store(_) => 40,
            Self::EnterScope => 41,
            Self::ExitScope => 42,
        }
    }
}

#[derive(Debug)]
pub struct Program {
    inner: Vec<Instruction>,
    procedures: Vec<Vec<Instruction>>,
}

fn read_ne_i128(input: &mut &[u8]) -> i128 {
    let (int_bytes, rest) = input.split_at(std::mem::size_of::<i128>());
    *input = rest;
    i128::from_ne_bytes(int_bytes.try_into().unwrap())
}

fn read_ne_f64(input: &mut &[u8]) -> f64 {
    let (int_bytes, rest) = input.split_at(std::mem::size_of::<f64>());
    *input = rest;
    f64::from_ne_bytes(int_bytes.try_into().unwrap())
}

fn read_ne_usize(input: &mut &[u8]) -> usize {
    let (int_bytes, rest) = input.split_at(std::mem::size_of::<usize>());
    *input = rest;
    usize::from_ne_bytes(int_bytes.try_into().unwrap())
}

fn read_ne_addr(input: &mut &[u8]) -> AddrRepr {
    let (int_bytes, rest) = input.split_at(std::mem::size_of::<AddrRepr>());
    *input = rest;
    AddrRepr::from_ne_bytes(int_bytes.try_into().unwrap())
}

macro_rules! progress {
    ($ptr:ident, $e:expr) => {{
        $ptr += 1;
        $e
    }};
}

impl Program {
    pub fn new() -> Self {
        Self {
            inner: Vec::new(),
            procedures: Vec::new(),
        }
    }

    pub fn inner(&self) -> impl Iterator<Item = &Instruction> {
        self.inner.iter()
    }

    pub fn create_procedure(&mut self) -> AddrRepr {
        self.procedures.push(Vec::new());

        (self.procedures.len() - 1) as AddrRepr
    }

    pub fn pop_procedure(&mut self) {
        std::mem::drop(self.procedures.pop());
    }

    pub fn push(&mut self, procedure: Option<AddrRepr>, instr: Instruction) {
        if let Some(procedure) = procedure {
            return self
                .procedures
                .get_mut(procedure as usize)
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
                    .get(proc as usize)
                    .expect("procedure not found")
                    .len() as AddrRepr,
            );
        }

        Addr::Absolute(self.inner.len() as AddrRepr)
    }

    fn resolve_addr(lookup: &HashMap<AddrRepr, AddrRepr>, addr: Addr) -> Addr {
        match addr {
            Addr::Procedure(proc) => {
                Addr::Absolute(lookup.get(&proc).expect("unknown procedure").clone())
            }
            Addr::Offset(proc, offset) => {
                Addr::Absolute(lookup.get(&proc).expect("unknown procedure").clone() + offset)
            }
            o => o,
        }
    }

    /// Resolves all procedures to their absolute address
    pub fn resolve(&mut self) -> &Self {
        if self.inner.last() != Some(&Instruction::Halt) {
            self.inner.push(Instruction::Halt);
        }

        // Lookup of proc -> absolute
        let mut lookup: HashMap<AddrRepr, AddrRepr> = HashMap::new();
        for (i, proc) in self.procedures.iter().enumerate() {
            lookup.insert(i as AddrRepr, self.inner.len() as AddrRepr);

            // TODO: don't clone here
            self.inner.extend(proc.clone());
        }

        self.inner = self
            .inner
            .iter()
            .map(|instr| {
                match instr {
                    Instruction::Jump(addr) => {
                        Instruction::Jump(Self::resolve_addr(&lookup, addr.clone()))
                    }
                    Instruction::JumpIf(addr) => {
                        Instruction::JumpIf(Self::resolve_addr(&lookup, addr.clone()))
                    }
                    Instruction::JumpIfElse(a, b) => Instruction::JumpIfElse(
                        Self::resolve_addr(&lookup, a.clone()),
                        Self::resolve_addr(&lookup, b.clone()),
                    ),
                    o => o.clone(), // TODO: don't clone
                }
            })
            .collect();

        self
    }

    pub fn bytes(&self) -> Vec<u8> {
        type I = Instruction;
        let mut bytes = Vec::new();

        for instr in self.inner() {
            bytes.push(instr.to_instr_id());

            match instr {
                I::LoadInt(i) => bytes.extend_from_slice(&i.to_ne_bytes()),
                I::LoadFloat(f) => bytes.extend_from_slice(&f.0.to_ne_bytes()),
                I::LoadString(s) => {
                    bytes.extend_from_slice(&s.len().to_ne_bytes());
                    bytes.extend_from_slice(s.as_bytes())
                }
                I::LoadBool(b) => bytes.extend_from_slice(&[if *b { 0 } else { 1 }]),
                I::LoadVar(i)
                | I::Load(i)
                | I::Store(i)
                | I::StoreMutVar(i)
                | I::StoreConstVar(i)
                | I::StoreVar(i)
                | I::MakeFunc(i) => {
                    bytes.extend_from_slice(&i.to_ne_bytes())
                }
                I::Jump(a) | I::JumpIf(a) => match a {
                    Addr::Absolute(p) => bytes.extend_from_slice(&p.to_ne_bytes()),
                    _ => panic!("procedures must be resolved prior to conversion"),
                },
                I::JumpIfElse(a, b) => match (a, b) {
                    (Addr::Absolute(a), Addr::Absolute(b)) => {
                        bytes.extend_from_slice(&[a.to_ne_bytes(), b.to_ne_bytes()].concat())
                    }
                    _ => panic!("procedures must be resolved prior to conversion"),
                },
                _ => (),
            }
        }

        bytes
    }

    pub fn dis(&self, w: &mut impl Write) -> std::io::Result<()> {
        type I = Instruction;
        let pad_length = self.inner.len().saturating_sub(1).to_string().len();

        for (j, instr) in self.inner().enumerate() {
            write!(w, "{:01$} | ", j, pad_length)?;
            match instr {
                I::LoadInt(i) => writeln!(w, "load_int {}", i)?,
                I::LoadFloat(f) => writeln!(w, "load_float {}", f.0)?,
                I::LoadString(s) => writeln!(w, "load_string {:?}", s)?,
                I::LoadBool(b) => writeln!(w, "load_bool {:?}", b)?,
                I::Load(i) => writeln!(w, "load {}", i)?,
                I::LoadVar(i) => writeln!(w, "load_var {}", i)?,
                I::Store(i) => writeln!(w, "store {}", i)?,
                I::StoreVar(i) => writeln!(w, "store_var {}", i)?,
                I::StoreMutVar(i) => writeln!(w, "store_mut_var {}", i)?,
                I::StoreConstVar(i) => writeln!(w, "store_const_var {}", i)?,
                I::Jump(Addr::Absolute(addr)) => writeln!(w, "jump {}", addr)?,
                I::JumpIf(Addr::Absolute(addr)) => writeln!(w, "jump_if {}", addr)?,
                I::JumpIfElse(Addr::Absolute(a), Addr::Absolute(b)) => {
                    writeln!(w, "jump_if_else {} {}", a, b)?
                }
                I::BinOpAdd => writeln!(w, "bin_add")?,
                I::BinOpSub => writeln!(w, "bin_sub")?,
                I::BinOpMul => writeln!(w, "bin_mul")?,
                I::BinOpDiv => writeln!(w, "bin_div")?,
                I::BinOpTrueDiv => writeln!(w, "bin_truediv")?,
                I::BinOpPow => writeln!(w, "bin_pow")?,
                I::BinOpBitOr => writeln!(w, "bin_bit_or")?,
                I::BinOpBitXor => writeln!(w, "bin_bit_xor")?,
                I::UnOpBitNot => writeln!(w, "bin_bit_not")?,
                I::BinOpBitAnd => writeln!(w, "bin_bit_and")?,
                I::OpEq => writeln!(w, "bin_eq")?,
                I::OpNe => writeln!(w, "bin_ne")?,
                I::OpLt => writeln!(w, "bin_lt")?,
                I::OpLe => writeln!(w, "bin_le")?,
                I::OpGt => writeln!(w, "bin_gt")?,
                I::OpGe => writeln!(w, "bin_ge")?,
                I::OpLogicalOr => writeln!(w, "log_or")?,
                I::OpLogicalAnd => writeln!(w, "log_and")?,
                I::OpLogicalNot => writeln!(w, "log_not")?,
                I::Pop => writeln!(w, "pop")?,
                I::Ret => writeln!(w, "ret")?,
                I::RetNull => writeln!(w, "ret_null")?,
                I::Halt => writeln!(w, "halt")?,
                I::EnterScope => writeln!(w, "enter_scope")?,
                I::ExitScope => writeln!(w, "exit_scope")?,
                _ => unimplemented!(),
            }
        }

        Ok(())
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        type I = Instruction;

        let mut ptr = 0;
        let mut instructions = Vec::<Instruction>::new();

        loop {
            instructions.push(match bytes.get(ptr) {
                Some(b) => match b {
                    0 => {
                        ptr += 1 + size_of::<i128>();
                        I::LoadInt(read_ne_i128(&mut &bytes[(ptr - 16)..ptr]))
                    }
                    1 => {
                        ptr += 1 + size_of::<f64>();
                        I::LoadFloat(read_ne_f64(&mut &bytes[(ptr - 8)..ptr]).into())
                    }
                    2 => {
                        let size = size_of::<usize>();
                        ptr += 1 + size;
                        let len = read_ne_usize(&mut &bytes[(ptr - size)..ptr]);

                        ptr += 1 + len;
                        I::LoadString(
                            String::from_utf8(Vec::from(&bytes[(ptr - len)..ptr])).unwrap(),
                        )
                    }
                    3 => {
                        ptr += 1 + size_of::<bool>();
                        I::LoadBool(if bytes[ptr - 1] == 0 { true } else { false })
                    }
                    4 => {
                        ptr += 1 + size_of::<usize>();
                        I::Load(read_ne_usize(&mut &bytes[(ptr - size_of::<usize>())..ptr]).into())
                    },
                    5 => progress!(ptr, I::UnOpPos),
                    6 => progress!(ptr, I::UnOpNeg),
                    7 => progress!(ptr, I::BinOpAdd),
                    8 => progress!(ptr, I::BinOpSub),
                    9 => progress!(ptr, I::BinOpMul),
                    10 => progress!(ptr, I::BinOpDiv),
                    11 => progress!(ptr, I::BinOpTrueDiv),
                    12 => progress!(ptr, I::BinOpPow),
                    13 => progress!(ptr, I::BinOpBitOr),
                    14 => progress!(ptr, I::BinOpBitXor),
                    15 => progress!(ptr, I::BinOpBitAnd),
                    16 => progress!(ptr, I::UnOpBitNot),
                    17 => progress!(ptr, I::OpEq),
                    18 => progress!(ptr, I::OpNe),
                    19 => progress!(ptr, I::OpLt),
                    20 => progress!(ptr, I::OpLe),
                    21 => progress!(ptr, I::OpGt),
                    22 => progress!(ptr, I::OpGe),
                    23 => progress!(ptr, I::OpLogicalOr),
                    24 => progress!(ptr, I::OpLogicalAnd),
                    25 => progress!(ptr, I::OpLogicalNot),
                    26 => {
                        ptr += 1 + size_of::<usize>();
                        I::LoadVar(read_ne_usize(&mut &bytes[(ptr - size_of::<usize>())..ptr]).into())
                    },
                    27 => {
                        ptr += 1 + size_of::<usize>();
                        I::StoreVar(read_ne_usize(&mut &bytes[(ptr - size_of::<usize>())..ptr]).into())
                    },
                    28 => {
                        ptr += 1 + size_of::<usize>();
                        I::StoreMutVar(read_ne_usize(&mut &bytes[(ptr - size_of::<usize>())..ptr]).into())
                    },
                    29 => {
                        ptr += 1 + size_of::<usize>();
                        I::StoreConstVar(read_ne_usize(&mut &bytes[(ptr - size_of::<usize>())..ptr]).into())
                    },
                    30 => unimplemented!(),
                    31 => unimplemented!(),
                    32 => {
                        ptr += 1 + size_of::<AddrRepr>();
                        I::Jump(Addr::Absolute(read_ne_addr(
                            &mut &bytes[(ptr - size_of::<AddrRepr>())..ptr],
                        )))
                    }
                    33 => {
                        ptr += 1 + size_of::<AddrRepr>();
                        I::JumpIf(Addr::Absolute(read_ne_addr(
                            &mut &bytes[(ptr - size_of::<AddrRepr>())..ptr],
                        )))
                    }
                    34 => {
                        ptr += 1 + size_of::<AddrRepr>();
                        let first = Addr::Absolute(read_ne_addr(
                            &mut &bytes[(ptr - size_of::<AddrRepr>())..ptr],
                        ));

                        ptr += size_of::<AddrRepr>();
                        I::JumpIfElse(
                            first,
                            Addr::Absolute(read_ne_addr(
                                &mut &bytes[(ptr - size_of::<AddrRepr>())..ptr],
                            )),
                        )
                    }
                    35 => progress!(ptr, I::Pop),
                    36 => progress!(ptr, I::Ret),
                    37 => progress!(ptr, I::RetNull),
                    38 => progress!(ptr, I::Halt),
                    39 => unimplemented!(),
                    40 => {
                        ptr += 1 + size_of::<usize>();
                        I::Store(read_ne_usize(&mut &bytes[(ptr - size_of::<usize>())..ptr]).into())
                    }
                    41 => progress!(ptr, I::EnterScope),
                    42 => progress!(ptr, I::ExitScope),
                    b => panic!("invalid byte 0x{:0x} at position {}", b, ptr),
                },
                None => break,
            });
        }

        Program {
            inner: instructions,
            procedures: Vec::new(),
        }
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
