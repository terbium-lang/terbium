//! The interpreter for Terbium.

mod interner;

use terbium_bytecode::{EqComparableFloat, Instruction, Program};

pub use interner::Interner;
use interner::StringId;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TerbiumObject {
    Null,
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

impl<const STACK_SIZE: usize> Stack<STACK_SIZE> {
    pub fn new() -> Self {
        Self {
            inner: [TerbiumObject::Null; STACK_SIZE],
            ptr: 0,
        }
    }

    /// Push an object to the stack.
    pub fn push(&mut self, o: TerbiumObject) {
        self.inner[self.ptr] = o;
        self.incr_ptr();
    }

    /// Increments the ptr, panicking if it goes above STACK_SIZE
    pub fn incr_ptr(&mut self) {
        self.ptr += 1;

        if self.ptr >= STACK_SIZE {
            panic!("stack overflow (surpassed stack size of {})", STACK_SIZE);
        }
    }

    /// Decrements the ptr, panicking if it goes below 0
    pub fn decr_ptr(&mut self) {
        self.ptr = self.ptr.checked_sub(1).expect("stack ptr already at 0");
    }

    /// Pop the previous object in the stack and move the pointer there
    pub fn pop(&mut self) -> TerbiumObject {
        self.decr_ptr();

        std::mem::replace(&mut self.inner[self.ptr], TerbiumObject::Null)
    }

    /// Get a cloned version of the previous object in the stack,
    /// but also move the pointer there
    pub fn pop_cloned(&mut self) -> TerbiumObject {
        self.decr_ptr();

        self.inner[self.ptr].clone()
    }

    /// Retrieve a reference to the next free slot
    pub fn next_free(&self) -> &TerbiumObject {
        &self.inner[self.ptr]
    }

    /// Retrieve a mutable reference to the free slot
    pub fn next_free_mut(&mut self) -> &mut TerbiumObject {
        &mut self.inner[self.ptr]
    }
}

#[derive(Debug)]
pub struct Interpreter<const STACK_SIZE: usize = 512> {
    stack: Stack<STACK_SIZE>,
    string_interner: Interner,
}

macro_rules! pat_num_ops {
    ($stack:expr, $lhs:ident, $rhs:ident; $ii:expr, $ff:expr, $if:expr, $fi:expr; $($pat:pat => $result:expr),*) => {
        match ($stack.pop(), $stack.pop()) {
            (TerbiumObject::Integer($rhs), TerbiumObject::Integer($lhs)) => {
                $stack.push($ii)
            }
            (TerbiumObject::Float($rhs), TerbiumObject::Float($lhs)) => {
                $stack.push($ff)
            }
            (TerbiumObject::Integer($rhs), TerbiumObject::Float($lhs)) => {
                $stack.push($fi)
            }
            (TerbiumObject::Float($rhs), TerbiumObject::Integer($lhs)) => {
                $stack.push($if)
            }
            $($pat => $result),*
        }
    }
}

impl<const STACK_SIZE: usize> Interpreter<STACK_SIZE> {
    pub fn new() -> Self {
        Self {
            stack: Stack::new(),
            // TODO: string length capacity to be interned could be configurable
            string_interner: Interner::with_capacity(128),
        }
    }

    pub fn stack(&mut self) -> &mut Stack<STACK_SIZE> {
        &mut self.stack
    }

    pub fn run_bytecode(&mut self, code: Program) {
        for instr in code.inner() {
            match instr.to_owned() {
                Instruction::LoadInt(i) => self.stack.push(TerbiumObject::Integer(i)),
                Instruction::LoadString(s) => self.stack.push(TerbiumObject::String(
                    self.string_interner.intern(s.as_str()),
                )),
                Instruction::LoadFloat(f) => self.stack.push(TerbiumObject::Float(f)),
                Instruction::UnOpPos => match self.stack.pop() {
                    o @ TerbiumObject::Integer(_) => self.stack.push(o),
                    o @ TerbiumObject::Float(_) => self.stack.push(o),
                    _ => todo!(),
                }
                Instruction::UnOpNeg => match self.stack.pop() {
                    TerbiumObject::Integer(i) => self.stack.push(
                        TerbiumObject::Integer(-i),
                    ),
                    TerbiumObject::Float(f) => self.stack.push(
                        TerbiumObject::Float((-f.0).into())
                    ),
                    _ => todo!(),
                }
                Instruction::BinOpAdd => pat_num_ops!(
                    self.stack, lhs, rhs;
                    TerbiumObject::Integer(lhs + rhs),
                    TerbiumObject::Float((lhs.0 + rhs.0).into()),
                    TerbiumObject::Float((lhs as f64 + rhs.0).into()),
                    TerbiumObject::Float((lhs.0 + rhs as f64).into());
                    (TerbiumObject::String(rhs), TerbiumObject::String(lhs)) => {
                        self.stack.push(TerbiumObject::String(
                            self.string_interner.intern((
                                self.string_interner.lookup(lhs).to_owned()
                                + self.string_interner.lookup(rhs)
                            ).as_str())
                        ))
                    },
                    _ => {
                        // TODO: Call op function, raise error if not found
                    }
                ),
                Instruction::BinOpSub => pat_num_ops!(
                    self.stack, lhs, rhs;
                    TerbiumObject::Integer(lhs - rhs),
                    TerbiumObject::Float((lhs.0 - rhs.0).into()),
                    TerbiumObject::Float((lhs as f64 - rhs.0).into()),
                    TerbiumObject::Float((lhs.0 - rhs as f64).into());
                    _ => {
                        // TODO
                    }
                ),
                Instruction::Pop => {
                    self.stack.pop();
                }
                _ => todo!(),
            }
        }
    }
}

pub type DefaultInterpreter = Interpreter<512>;

#[cfg(test)]
mod tests {
    use terbium_bytecode::{Instruction, Program};
    use super::{DefaultInterpreter, TerbiumObject};

    #[test]
    fn test_interpreter() {
        let program = Program::from_iter([
            Instruction::LoadInt(1),
            Instruction::LoadInt(1),
            Instruction::BinOpAdd,
        ]);
        let mut interpreter = DefaultInterpreter::new();
        interpreter.run_bytecode(program);

        assert_eq!(interpreter.stack().pop(), TerbiumObject::Integer(2));
    }
}
