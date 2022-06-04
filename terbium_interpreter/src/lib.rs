//! The interpreter for Terbium.

mod interner;

use terbium_bytecode::{EqComparableFloat, Instruction, Program};

pub use interner::Interner;
use interner::StringId;

#[derive(Clone, Debug, PartialEq)]
pub enum TerbiumObject {
    Integer(i128),
    Float(EqComparableFloat),
    String(StringId),
    Bool(bool),
}

#[derive(Debug)]
pub struct Stack<const STACK_SIZE: usize = 512> {
    pub(crate) inner: [TerbiumObject; STACK_SIZE],
    pub(crate) ptr: usize,
}

impl Stack {
    pub fn new() -> Self {
        Self {
            inner: unsafe { std::mem::zeroed() },
            ptr: 0,
        }
    }

    pub fn push(&mut self, o: TerbiumObject) {
        self.inner[self.ptr] = o;
        self.ptr += 1;
    }

    pub fn pop(&mut self) -> TerbiumObject {
        self.ptr = self.ptr.checked_sub(1).expect("stack ptr already at 0");

        std::mem::replace(&mut self.inner[self.ptr], unsafe { std::mem::zeroed() })
    }
}

#[derive(Debug)]
pub struct Interpreter<const STACK_SIZE: usize = 512> {
    stack: Stack<STACK_SIZE>,
    string_interner: Interner,
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            stack: Stack::new(),
            // TODO: string length capacity to be interned could be configurable
            string_interner: Interner::with_capacity(128),
        }
    }

    pub fn run_bytecode(&mut self, code: Program) {
        for instr in code.inner() {
            match instr.to_owned() {
                Instruction::LoadInt(i) => self.stack.push(TerbiumObject::Integer(i)),
                Instruction::LoadString(s) => self.stack.push(TerbiumObject::String(
                    self.string_interner.intern(s.as_str()),
                )),
                Instruction::LoadFloat(f) => self.stack.push(TerbiumObject::Float(f)),
                Instruction::AddInt => match (self.stack.pop(), self.stack.pop()) {
                    (TerbiumObject::Integer(rhs), TerbiumObject::Integer(lhs)) => {
                        self.stack.push(TerbiumObject::Integer(lhs + rhs))
                    }
                    _ => panic!("bytecode triggered AddInt on non-integers"),
                },
                Instruction::Pop => {
                    self.stack.pop();
                }
                _ => todo!(),
            }
        }
    }
}
