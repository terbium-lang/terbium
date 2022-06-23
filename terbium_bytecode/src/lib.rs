//! Crate for Terbium byte-code/intermediate represenation.
//!
//! Because Terbium can be directly interpreted or be compiled into LLVM IR,
//! there must be a representation of Terbium code that can provide an easy
//! entrypoint to both.
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]

mod interpreter;
mod util;

use std::collections::HashMap;
use std::io::Write;
use std::mem::size_of;
use std::path::PathBuf;
use std::str::FromStr;

use terbium_grammar::{Source, Span};
pub use interpreter::Interpreter;
pub use util::EqComparableFloat;

pub type AddrRepr = usize;

#[derive(Copy, Clone, Debug, PartialOrd, PartialEq, Eq, Hash)]
pub enum Addr {
    Absolute(AddrRepr),
    Procedure(AddrRepr),
    Offset(AddrRepr, AddrRepr),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Instruction {
    // Constants mapped to a lookup table
    LoadInt(u128), // TODO: this takes 16 bytes in raw bytecode representation, not the best
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
    AssignVar(usize),
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
    #[must_use]
    pub fn stack_diff(&self) -> i8 {
        match self {
            Self::LoadInt(_) => 1,
            Self::Pop => -1,

            _ => todo!(),
        }
    }

    #[must_use]
    pub fn size(&self) -> usize {
        1_usize + match self {
            Self::LoadInt(_) => size_of::<u128>(),
            Self::LoadFloat(_) => size_of::<f64>(),
            Self::LoadString(s) => s.len(), // FIXME: String might exceeds 255
            Self::LoadBool(_) => 1,
            Self::LoadVar(_)
            | Self::StoreVar(_)
            | Self::StoreMutVar(_)
            | Self::StoreConstVar(_)
            | Self::MakeFunc(_)
            | Self::Load(_)
            | Self::Store(_)
            | Self::LoadFrame(_)
            | Self::AssignVar(_) => size_of::<usize>(),
            Self::Jump(_) | Self::JumpIf(_) => size_of::<AddrRepr>(),
            Self::JumpIfElse(_, _) => size_of::<AddrRepr>() * 2,
            _ => 0,
        }
    }

    #[must_use]
    pub const fn to_instr_id(&self) -> u8 {
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
            Self::AssignVar(_) => 43,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RichInstruction {
    inner: Instruction,
    span: Option<Span>,
    name: Option<String>,
}

impl From<Instruction> for RichInstruction {
    fn from(instr: Instruction) -> Self {
        Self {
            inner: instr,
            span: None,
            name: None,
        }
    }
}

impl RichInstruction {
    pub fn spanned(instr: Instruction, span: Span) -> Self {
        Self { inner: instr, span: Some(span), name: None }
    }

    pub fn instr(&self) -> &Instruction {
        &self.inner
    }

    pub fn into_instr(self) -> Instruction {
        self.inner
    }

    pub fn span(&self) -> Option<Span> {
        self.span.clone()
    }

    pub fn name(&self) -> &Option<String> {
        &self.name
    }
}

#[derive(Debug)]
pub struct Program {
    inner: Vec<RichInstruction>,
    procedures: Vec<Vec<RichInstruction>>,
}

pub(crate) fn read_ne_u128(input: &mut &[u8]) -> u128 {
    let (int_bytes, rest) = input.split_at(std::mem::size_of::<u128>());
    *input = rest;
    u128::from_ne_bytes(int_bytes.try_into().unwrap())
}

pub(crate) fn read_ne_f64(input: &mut &[u8]) -> f64 {
    let (int_bytes, rest) = input.split_at(std::mem::size_of::<f64>());
    *input = rest;
    f64::from_ne_bytes(int_bytes.try_into().unwrap())
}

pub(crate) fn read_ne_usize(input: &mut &[u8]) -> usize {
    let (int_bytes, rest) = input.split_at(std::mem::size_of::<usize>());
    *input = rest;
    usize::from_ne_bytes(int_bytes.try_into().unwrap())
}

macro_rules! progress {
    ($ptr:ident, $e:expr) => {{
        $ptr += 1;
        $e
    }};
}

macro_rules! parse_usize {
    ($ptr:ident, $bytes:expr, $member:ident) => {{
        $ptr += 1 + size_of::<usize>();
        Instruction::$member(
            $crate::read_ne_usize(&mut &$bytes[($ptr - ::std::mem::size_of::<usize>())..$ptr])
                .into(),
        )
    }};
}

impl Program {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inner: Vec::new(),
            procedures: Vec::new(),
        }
    }

    pub fn inner(&self) -> impl Iterator<Item = &RichInstruction> {
        self.inner.iter()
    }

    pub fn create_procedure(&mut self) -> AddrRepr {
        self.procedures.push(Vec::new());

        self.procedures.len() - 1
    }

    pub fn pop_procedure(&mut self) {
        std::mem::drop(self.procedures.pop());
    }

    pub fn push(&mut self, procedure: Option<AddrRepr>, instr: RichInstruction) {
        if let Some(procedure) = procedure {
            return self
                .procedures
                .get_mut(procedure)
                .expect("no procedure there")
                .push(instr);
        }

        self.inner.push(instr);
    }

    #[must_use]
    #[allow(clippy::match_wildcard_for_single_variants)]
    pub fn current_addr(&self, procedure: Option<AddrRepr>) -> Addr {
        match self.next_addr(procedure) {
            Addr::Absolute(i) => Addr::Absolute(i - 1),
            Addr::Offset(p, i) => Addr::Offset(p, i - 1),
            _ => unreachable!(),
        }
    }

    #[must_use]
    pub fn next_addr(&self, procedure: Option<AddrRepr>) -> Addr {
        if let Some(proc) = procedure {
            return Addr::Offset(
                proc,
                self.procedures
                    .get(proc)
                    .expect("procedure not found")
                    .len(),
            );
        }

        Addr::Absolute(self.inner.len())
    }

    #[allow(clippy::match_wildcard_for_single_variants)]
    fn resolve_addr(lookup: &HashMap<AddrRepr, AddrRepr>, addr: Addr) -> Addr {
        match addr {
            Addr::Procedure(proc) => Addr::Absolute(*lookup.get(&proc).expect("unknown procedure")),
            Addr::Offset(proc, offset) => {
                Addr::Absolute(lookup.get(&proc).expect("unknown procedure") + offset)
            }
            o => o,
        }
    }

    /// Resolves all procedures to their absolute address
    pub fn resolve(&mut self) -> &Self {
        if self.inner.last().map(RichInstruction::instr) != Some(&Instruction::Halt) {
            self.inner.push(Instruction::Halt.into());
        }

        // Lookup of proc -> absolute
        let mut lookup: HashMap<AddrRepr, AddrRepr> = HashMap::new();
        for (i, proc) in self.procedures.iter().enumerate() {
            lookup.insert(i, self.inner.len());

            // TODO: don't clone here
            self.inner.extend(proc.clone());
        }

        self.inner = self
            .inner
            .iter()
            .map(|RichInstruction { inner: instr, span, name }| RichInstruction {
                inner: match instr {
                    Instruction::Jump(addr) => {
                        Instruction::Jump(Self::resolve_addr(&lookup, *addr))
                    }
                    Instruction::JumpIf(addr) => {
                        Instruction::JumpIf(Self::resolve_addr(&lookup, *addr))
                    }
                    Instruction::JumpIfElse(a, b) => Instruction::JumpIfElse(
                        Self::resolve_addr(&lookup, *a),
                        Self::resolve_addr(&lookup, *b),
                    ),
                    o => o.clone(), // TODO: don't clone
                },
                span: span.clone(),
                name: name.clone(),
            })
            .collect();

        self
    }

    #[must_use]
    pub fn bytes(&self) -> Vec<u8> {
        type I = Instruction;
        let mut bytes = Vec::new();

        for RichInstruction { inner: instr, span, name } in self.inner() {
            bytes.push(instr.to_instr_id());

            match instr {
                I::LoadInt(i) => bytes.extend_from_slice(&i.to_ne_bytes()),
                I::LoadFloat(f) => bytes.extend_from_slice(&f.0.to_ne_bytes()),
                I::LoadString(s) => {
                    bytes.extend_from_slice(&s.len().to_ne_bytes());
                    bytes.extend_from_slice(s.as_bytes());
                }
                I::LoadBool(b) => bytes.extend_from_slice(&[if *b { 0 } else { 1 }]),
                I::LoadVar(i)
                | I::Load(i)
                | I::Store(i)
                | I::StoreMutVar(i)
                | I::StoreConstVar(i)
                | I::StoreVar(i)
                | I::AssignVar(i)
                | I::MakeFunc(i) => bytes.extend_from_slice(&i.to_ne_bytes()),
                I::Jump(a) | I::JumpIf(a) => match a {
                    Addr::Absolute(p) => bytes.extend_from_slice(&p.to_ne_bytes()),
                    _ => panic!("procedures must be resolved prior to conversion"),
                },
                I::JumpIfElse(a, b) => match (a, b) {
                    (Addr::Absolute(a), Addr::Absolute(b)) => {
                        bytes.extend_from_slice(&[a.to_ne_bytes(), b.to_ne_bytes()].concat());
                    }
                    _ => panic!("procedures must be resolved prior to conversion"),
                },
                _ => (),
            }

            let span_bytes = span.clone().map(
                |s| {
                    let path = s.src().to_path();
                    let lossy = path.to_string_lossy();
                    let src_bytes = &*lossy.as_bytes();

                    [
                        [1_u8].as_slice(),
                        s.start().to_ne_bytes().as_slice(),
                        s.end().to_ne_bytes().as_slice(),
                        src_bytes.len().to_ne_bytes().as_slice(),
                        src_bytes,
                    ].concat()
                }
            );
            bytes.extend_from_slice(&*span_bytes.unwrap_or(vec![0_u8]));

            let name_bytes = name.clone().map(|s| s.as_bytes().to_vec());
            bytes.extend_from_slice(&name_bytes
                .clone()
                .map(|n| n.len())
                .unwrap_or(0)
                .to_ne_bytes()
            );
            bytes.extend_from_slice(&*name_bytes.unwrap_or(vec![]));
        }

        bytes
    }

    pub fn dis(&self, w: &mut impl Write) -> std::io::Result<()> {
        type I = Instruction;
        let pad_length = self.inner.len().saturating_sub(1).to_string().len();

        for (j, RichInstruction{ inner: instr, name, .. }) in self.inner().enumerate() {
            write!(w, "{:01$} | ", j, pad_length)?;
            match instr {
                I::LoadInt(i) => write!(w, "load_int {}", i)?,
                I::LoadFloat(f) => write!(w, "load_float {}", f.0)?,
                I::LoadString(s) => write!(w, "load_string {:?}", s)?,
                I::LoadBool(b) => write!(w, "load_bool {:?}", b)?,
                I::Load(i) => write!(w, "load {}", i)?,
                I::LoadVar(i) => write!(w, "load_var {}", i)?,
                I::Store(i) => write!(w, "store {}", i)?,
                I::StoreVar(i) => write!(w, "store_var {}", i)?,
                I::StoreMutVar(i) => write!(w, "store_mut_var {}", i)?,
                I::StoreConstVar(i) => write!(w, "store_const_var {}", i)?,
                I::AssignVar(i) => write!(w, "assign_var {}", i)?,
                I::Jump(Addr::Absolute(addr)) => write!(w, "jump {}", addr)?,
                I::JumpIf(Addr::Absolute(addr)) => write!(w, "jump_if {}", addr)?,
                I::JumpIfElse(Addr::Absolute(a), Addr::Absolute(b)) => {
                    write!(w, "jump_if_else {} {}", a, b)?;
                }
                I::BinOpAdd => write!(w, "bin_add")?,
                I::BinOpSub => write!(w, "bin_sub")?,
                I::BinOpMul => write!(w, "bin_mul")?,
                I::BinOpDiv => write!(w, "bin_div")?,
                I::BinOpTrueDiv => write!(w, "bin_truediv")?,
                I::BinOpPow => write!(w, "bin_pow")?,
                I::BinOpBitOr => write!(w, "bin_bit_or")?,
                I::BinOpBitXor => write!(w, "bin_bit_xor")?,
                I::UnOpBitNot => write!(w, "bin_bit_not")?,
                I::BinOpBitAnd => write!(w, "bin_bit_and")?,
                I::OpEq => write!(w, "bin_eq")?,
                I::OpNe => write!(w, "bin_ne")?,
                I::OpLt => write!(w, "bin_lt")?,
                I::OpLe => write!(w, "bin_le")?,
                I::OpGt => write!(w, "bin_gt")?,
                I::OpGe => write!(w, "bin_ge")?,
                I::OpLogicalOr => write!(w, "log_or")?,
                I::OpLogicalAnd => write!(w, "log_and")?,
                I::OpLogicalNot => write!(w, "log_not")?,
                I::Pop => write!(w, "pop")?,
                I::Ret => write!(w, "ret")?,
                I::RetNull => write!(w, "ret_null")?,
                I::Halt => write!(w, "halt")?,
                I::EnterScope => write!(w, "enter_scope")?,
                I::ExitScope => write!(w, "exit_scope")?,
                _ => unimplemented!(),
            }

            if let Some(name) = name {
                write!(w, " ({})", name)?;
            }

            writeln!(w)?;
        }

        Ok(())
    }

    #[must_use]
    #[allow(clippy::match_same_arms)] // Remove when unimplemented!() are all implemented
    pub fn from_bytes(bytes: &[u8]) -> Self {
        type I = Instruction;

        let mut ptr = 0;
        let mut instructions = Vec::<RichInstruction>::new();

        loop {
            let instr = match bytes.get(ptr) {
                Some(b) => match b {
                    0 => {
                        ptr += 1 + size_of::<u128>();
                        I::LoadInt(read_ne_u128(&mut &bytes[(ptr - 16)..ptr]))
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
                        I::LoadBool(bytes[ptr - 1] == 0)
                    }
                    4 => parse_usize!(ptr, bytes, Load),
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
                    26 => parse_usize!(ptr, bytes, LoadVar),
                    27 => parse_usize!(ptr, bytes, StoreVar),
                    28 => parse_usize!(ptr, bytes, StoreMutVar),
                    29 => parse_usize!(ptr, bytes, StoreConstVar),
                    30 => unimplemented!(),
                    31 => unimplemented!(),
                    32 => {
                        ptr += 1 + size_of::<AddrRepr>();
                        I::Jump(Addr::Absolute(read_ne_usize(
                            &mut &bytes[(ptr - size_of::<AddrRepr>())..ptr],
                        )))
                    }
                    33 => {
                        ptr += 1 + size_of::<AddrRepr>();
                        I::JumpIf(Addr::Absolute(read_ne_usize(
                            &mut &bytes[(ptr - size_of::<AddrRepr>())..ptr],
                        )))
                    }
                    34 => {
                        ptr += 1 + size_of::<AddrRepr>();
                        let first = Addr::Absolute(read_ne_usize(
                            &mut &bytes[(ptr - size_of::<AddrRepr>())..ptr],
                        ));

                        ptr += size_of::<AddrRepr>();
                        I::JumpIfElse(
                            first,
                            Addr::Absolute(read_ne_usize(
                                &mut &bytes[(ptr - size_of::<AddrRepr>())..ptr],
                            )),
                        )
                    }
                    35 => progress!(ptr, I::Pop),
                    36 => progress!(ptr, I::Ret),
                    37 => progress!(ptr, I::RetNull),
                    38 => progress!(ptr, I::Halt),
                    39 => unimplemented!(),
                    40 => parse_usize!(ptr, bytes, Store),
                    41 => progress!(ptr, I::EnterScope),
                    42 => progress!(ptr, I::ExitScope),
                    43 => parse_usize!(ptr, bytes, AssignVar),
                    b => panic!("invalid byte 0x{:0x} at position {}", b, ptr),
                },
                None => break,
            };

            let size = size_of::<usize>();

            let span = if bytes[ptr] == 1 {
                ptr += 1 + size;
                let start = read_ne_usize(&mut &bytes[(ptr - size)..ptr]);

                ptr += 1 + size;
                let end = read_ne_usize(&mut &bytes[(ptr - size)..ptr]);

                ptr += 1 + size;
                let src_len = read_ne_usize(&mut &bytes[(ptr - size)..ptr]);

                ptr += 1 + src_len;
                let path = PathBuf::from_str(&String::from_utf8_lossy(
                    &bytes[(ptr - src_len)..ptr]
                )).unwrap();

                let src = path.as_path();
                let src = Source::from_path(src);

                Some(Span::from_range(src, start..end))
            } else {
                None
            };

            ptr += 1 + size;
            let string_len = read_ne_usize(&mut &bytes[(ptr - size)..ptr]);

            ptr += 1 + string_len;
            let name = if string_len > 0 {
                Some(String::from_utf8(Vec::from(&bytes[(ptr - string_len)..ptr])).unwrap())
            } else {
                None
            };

            instructions.push(RichInstruction {
                inner: instr,
                span,
                name,
            })
        }

        Self {
            inner: instructions,
            procedures: Vec::new(),
        }
    }
}

impl FromIterator<RichInstruction> for Program {
    fn from_iter<I: IntoIterator<Item = RichInstruction>>(iter: I) -> Self {
        Self {
            inner: iter.into_iter().collect(),
            procedures: Vec::new(),
        }
    }
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}
