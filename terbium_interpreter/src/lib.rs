//! The interpreter for Terbium.

#![feature(box_patterns)]

mod interner;

use std::collections::HashMap;
use terbium_bytecode::{Addr, AddrRepr, EqComparableFloat, Instruction, Program};

pub use interner::Interner;
use interner::StringId;

pub type ObjectRef = usize;

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
    pub(crate) inner: [ObjectRef; STACK_SIZE],
    pub(crate) ptr: usize,
}

impl<const STACK_SIZE: usize> Stack<STACK_SIZE> {
    pub fn new() -> Self {
        Self {
            inner: [0; STACK_SIZE],
            ptr: 0,
        }
    }

    /// Push an object to the stack.
    pub fn push(&mut self, o: ObjectRef) {
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
    pub fn pop(&mut self) -> ObjectRef {
        self.decr_ptr();

        std::mem::replace(&mut self.inner[self.ptr], 0)
    }

    /// Get a cloned version of the previous object in the stack,
    /// but also move the pointer there
    pub fn pop_cloned(&mut self) -> ObjectRef {
        self.decr_ptr();

        self.inner[self.ptr].clone()
    }

    /// Retrieve a reference to the next free slot
    pub fn next_free(&self) -> &ObjectRef {
        &self.inner[self.ptr]
    }

    /// Retrieve a mutable reference to the free slot
    pub fn next_free_mut(&mut self) -> &mut ObjectRef {
        &mut self.inner[self.ptr]
    }
}

#[derive(Debug)]
pub struct ObjectStore(pub(crate) HashMap<ObjectRef, TerbiumObject>);

impl ObjectStore {
    pub fn new() -> Self {
        let mut inner = HashMap::new();
        inner.insert(0 as ObjectRef, TerbiumObject::Null);

        Self(inner)
    }

    pub fn resolve(&self, loc: ObjectRef) -> &TerbiumObject {
        self.0.get(&loc).expect(&*format!("no object at location {:0x}", loc))
    }

    pub fn resolve_or_null(&self, loc: ObjectRef) -> &TerbiumObject {
        self.0.get(&loc).unwrap_or_else(|| &TerbiumObject::Null)
    }
}

#[derive(Debug)]
pub struct ScopeEntry {
    pub loc: ObjectRef,
    r#mut: bool,
    r#const: bool,
}

impl ScopeEntry {
    pub fn is_const(&self) -> bool { self.r#const }

    pub fn is_mut(&self) -> bool { self.r#mut }
}

impl Into<ObjectRef> for ScopeEntry {
    fn into(self) -> ObjectRef {
        self.loc
    }
}

#[derive(Debug)]
pub struct Scope {
    pub locals: HashMap<usize, ScopeEntry>,
}

impl Scope {
    pub fn new() -> Self {
        Self { locals: HashMap::new() }
    }
}

#[derive(Debug)]
pub struct Context<const STACK_SIZE: usize = 512> {
    pub(crate) store: ObjectStore,
    pub(crate) stack: Stack<STACK_SIZE>,
    scopes: Vec<Scope>,
    integer_lookup: HashMap<i128, ObjectRef>,
    bool_lookup: [ObjectRef; 2],
}

impl<const STACK_SIZE: usize> Context<STACK_SIZE> {
    pub fn new() -> Self {
        Self {
            store: ObjectStore::new(),
            stack: Stack::new(),
            scopes: vec![Scope::new()],
            integer_lookup: HashMap::new(),
            bool_lookup: [0, 0],
        }
    }

    pub fn push(&mut self, o: ObjectRef) {
        self.stack.push(o)
    }

    pub fn pop_ref(&mut self) -> ObjectRef {
        self.stack.pop()
    }
    
    pub fn pop_detailed(&mut self) -> (ObjectRef, &TerbiumObject) {
        let loc = self.stack.pop();
        
        (loc, self.store.resolve(loc))
    }
    
    pub fn pop(&mut self) -> &TerbiumObject {
        let loc = self.pop_ref();

        self.store.resolve(loc)
    }

    pub fn pop_or_null(&mut self) -> &TerbiumObject {
        let loc = self.pop_ref();

        self.store.resolve_or_null(loc)
    }

    pub fn pop_cloned(&mut self) -> TerbiumObject {
        self.store.resolve(self.stack.pop_cloned()).clone()
    }

    pub fn store(&mut self, loc: ObjectRef, o: TerbiumObject) -> ObjectRef {
        self.store.0.insert(loc, o);

        loc.clone()
    }

    pub fn store_auto(&mut self, o: TerbiumObject) -> ObjectRef {
        // TODO: this is O(n), not really the best
        let key = self.store.0.keys().max().unwrap_or(&0) + 1;
        self.store(key, o);

        key
    }

    pub fn load_int(&mut self, i: i128) -> ObjectRef {
        let loc = self.store_auto(TerbiumObject::Integer(i));

        *self.integer_lookup.entry(i).or_insert(loc)
    }

    pub fn load_bool(&mut self, b: bool) -> ObjectRef {
        let index = if b { 1 } else { 0 } as usize;
        let ptr = self.bool_lookup[index];

        match ptr {
            0 => {
                let loc = self.store_auto(TerbiumObject::Bool(b));
                self.bool_lookup[index] = loc;
                loc
            }
            _ => ptr,
        }
    }

    pub fn locals(&self) -> &Scope {
        self.scopes.last().unwrap()
    }

    pub fn locals_mut(&mut self) -> &mut Scope {
        self.scopes.last_mut().unwrap()
    }

    pub fn store_var(&mut self, key: usize, entry: ScopeEntry) {
        self.locals_mut().locals.insert(key, entry);
    }

    pub fn lookup_var(&self, key: usize) -> Option<&ScopeEntry> {
        for scope in self.scopes.iter().rev() {
            if let Some(entry) = scope.locals.get(&key) {
                return Some(entry);
            }
        }

        None
    }

    pub fn enter_scope(&mut self) {
        self.scopes.push(Scope::new());
    }

    pub fn exit_scope(&mut self) {
        self.scopes.pop();
    }
}

#[derive(Debug)]
pub struct Interpreter<const STACK_SIZE: usize = 512> {
    pub ctx: Context<STACK_SIZE>,
    string_interner: Interner,
}

macro_rules! pat_num_ops {
    ($ctx:expr, $lhs:ident, $rhs:ident; $ii:expr, $ff:expr, $if:expr, $fi:expr; $($pat:pat => $result:expr),*) => {{
        let first = $ctx.pop_ref();
        let second = $ctx.pop_ref();

        match ($ctx.store.resolve(first), $ctx.store.resolve(second)) {
            (TerbiumObject::Integer($rhs), TerbiumObject::Integer($lhs)) => {
                let result = $ii;
                $ctx.push(result)
            }
            (TerbiumObject::Float($rhs), TerbiumObject::Float($lhs)) => {
                let result = $ff;
                $ctx.push(result)
            }
            (TerbiumObject::Integer($rhs), TerbiumObject::Float($lhs)) => {
                let result = $fi;
                $ctx.push(result)
            }
            (TerbiumObject::Float($rhs), TerbiumObject::Integer($lhs)) => {
                let result = $if;
                $ctx.push(result)
            }
            $($pat => $result),*
        }
    }}
}

macro_rules! deferred_method {
    ($ctx:expr, $meth:ident, $e:expr) => {{
        let subject = $e;
        
        $ctx.$meth(subject)
    }}
}

macro_rules! store_auto {
    ($ctx:expr, $e:expr) => {
        deferred_method!($ctx, store_auto, $e)
    }
}

macro_rules! load_int {
    ($ctx:expr, $e:expr) => {
        deferred_method!($ctx, load_int, $e)
    }
}

macro_rules! load_bool {
    ($ctx:expr, $e:expr) => {
        deferred_method!($ctx, load_bool, $e)
    }
}

macro_rules! push {
    ($ctx:expr, $e:expr) => {
        deferred_method!($ctx, push, $e)
    }
}

impl<const STACK_SIZE: usize> Interpreter<STACK_SIZE> {
    pub fn new() -> Self {
        Self {
            ctx: Context::new(),
            // TODO: string length capacity to be interned could be configurable
            string_interner: Interner::with_capacity(128),
        }
    }

    pub fn stack(&mut self) -> &mut Stack<STACK_SIZE> {
        &mut self.ctx.stack
    }
    
    pub fn string_lookup(&self, id: StringId) -> &str {
        self.string_interner.lookup(id)
    }

    pub fn is_truthy(&self, o: &TerbiumObject) -> bool {
        match o {
            TerbiumObject::Bool(b) => b.clone(),
            TerbiumObject::Integer(i) => *i != 0,
            TerbiumObject::Float(EqComparableFloat(f)) => *f != 0_f64,
            TerbiumObject::String(s) => self.string_interner.lookup(*s).len() > 0,
            TerbiumObject::Null => false,
        }
    }

    pub fn get_bool_object(&mut self, o: &TerbiumObject) -> ObjectRef {
        load_bool!(self.ctx, self.is_truthy(o))
    }

    pub fn run_bytecode(&mut self, code: Program) {
        let mut pos: AddrRepr = 0;
        let mut jump_history: Vec<AddrRepr> = Vec::new();
        let instructions = code.inner().collect::<Vec<_>>();

        loop {
            let instr = instructions[pos as usize];

            match instr.to_owned() {
                Instruction::LoadInt(i) => push!(self.ctx, load_int!(self.ctx, i)),
                Instruction::LoadString(s) => push!(self.ctx, store_auto!(self.ctx,
                    TerbiumObject::String(
                        self.string_interner.intern(s.as_str()),
                    )
                )),
                Instruction::LoadFloat(f) => push!(self.ctx,
                    store_auto!(self.ctx, TerbiumObject::Float(f))
                ),
                Instruction::LoadBool(b) => push!(self.ctx, load_bool!(self.ctx, b)),
                Instruction::UnOpPos => match self.ctx.pop_detailed() {
                    (o, TerbiumObject::Integer(_)) => self.ctx.push(o),
                    (o, TerbiumObject::Float(_)) => self.ctx.push(o),
                    _ => todo!(),
                },
                Instruction::UnOpNeg => match self.ctx.pop() {
                    TerbiumObject::Integer(i) => push!(self.ctx, load_int!(self.ctx, -*i)),
                    TerbiumObject::Float(f) => push!(self.ctx,
                        store_auto!(self.ctx, TerbiumObject::Float((-f.0).into()))
                    ),
                    _ => todo!(),
                },
                Instruction::BinOpAdd => pat_num_ops!(
                    self.ctx, lhs, rhs;
                    load_int!(self.ctx, *lhs + *rhs),
                    store_auto!(self.ctx, TerbiumObject::Float((lhs.0 + rhs.0).into())),
                    store_auto!(self.ctx, TerbiumObject::Float((*lhs as f64 + rhs.0).into())),
                    store_auto!(self.ctx, TerbiumObject::Float((lhs.0 + *rhs as f64).into()));
                    (TerbiumObject::String(rhs), TerbiumObject::String(lhs)) => {
                        let loc = store_auto!(self.ctx, TerbiumObject::String(
                            self.string_interner.intern((
                                self.string_interner.lookup(*lhs).to_owned()
                                + self.string_interner.lookup(*rhs)
                            ).as_str())
                        ));

                        self.ctx.push(loc)
                    },
                    _ => {
                        // TODO: Call op function, raise error if not found
                    }
                ),
                Instruction::BinOpSub => pat_num_ops!(
                    self.ctx, lhs, rhs;
                    load_int!(self.ctx, *lhs - *rhs),
                    store_auto!(self.ctx, TerbiumObject::Float((lhs.0 - rhs.0).into())),
                    store_auto!(self.ctx, TerbiumObject::Float((*lhs as f64 - rhs.0).into())),
                    store_auto!(self.ctx, TerbiumObject::Float((lhs.0 - *rhs as f64).into()));
                    _ => {
                        // TODO
                    }
                ),
                Instruction::BinOpMul => pat_num_ops!(
                    self.ctx, lhs, rhs;
                    load_int!(self.ctx, *lhs * *rhs),
                    store_auto!(self.ctx, TerbiumObject::Float((lhs.0 * rhs.0).into())),
                    store_auto!(self.ctx, TerbiumObject::Float((*lhs as f64 - rhs.0).into())),
                    store_auto!(self.ctx, TerbiumObject::Float((lhs.0 - *rhs as f64).into()));
                    (TerbiumObject::Integer(rhs), TerbiumObject::String(lhs)) => {
                        let loc = store_auto!(self.ctx, TerbiumObject::String(
                            self.string_interner.intern(
                                self.string_interner.lookup(*lhs).repeat(*rhs as usize).as_str(),
                            )
                        ));
                        self.ctx.push(loc)
                    },
                    _ => {
                        // TODO
                    }
                ),
                #[allow(unused_parens)]
                Instruction::OpEq => pat_num_ops!(
                    self.ctx, lhs, rhs;
                    load_bool!(self.ctx, *lhs == *rhs),
                    load_bool!(self.ctx, lhs == rhs),
                    load_bool!(self.ctx, rhs.eq(&(*lhs as f64))),
                    load_bool!(self.ctx, lhs.eq(&(*rhs as f64)));
                    (TerbiumObject::String(rhs), TerbiumObject::String(lhs)) => {
                        let b = load_bool!(self.ctx,
                            self.string_interner.lookup(*lhs)
                            == self.string_interner.lookup(*rhs)
                        );
                        self.ctx.push(b)
                    },
                    (TerbiumObject::Bool(rhs), TerbiumObject::Bool(lhs)) => {
                        let b = load_bool!(self.ctx, lhs == rhs);
                        self.ctx.push(b)
                    },
                    (TerbiumObject::Null, TerbiumObject::Null) => {
                        push!(self.ctx, load_bool!(self.ctx, true))
                    },
                    ((TerbiumObject::Null, _) | (_, TerbiumObject::Null)) => {
                        push!(self.ctx, load_bool!(self.ctx, false))
                    },
                    _ => {
                        // TODO
                    }
                ),
                #[allow(unused_parens)]
                Instruction::OpNe => pat_num_ops!(
                    self.ctx, lhs, rhs;
                    load_bool!(self.ctx, *lhs != *rhs),
                    load_bool!(self.ctx, lhs != rhs),
                    load_bool!(self.ctx, !rhs.eq(&(*lhs as f64))),
                    load_bool!(self.ctx, !lhs.eq(&(*rhs as f64)));
                    (TerbiumObject::String(rhs), TerbiumObject::String(lhs)) => {
                        let b = load_bool!(self.ctx,
                            self.string_interner.lookup(*lhs)
                            != self.string_interner.lookup(*rhs)
                        );
                        self.ctx.push(b)
                    },
                    (TerbiumObject::Bool(rhs), TerbiumObject::Bool(lhs)) => {
                        let b = load_bool!(self.ctx, lhs != rhs);
                        self.ctx.push(b)
                    },
                    (TerbiumObject::Null, TerbiumObject::Null) => {
                        push!(self.ctx, load_bool!(self.ctx, false))
                    },
                    ((TerbiumObject::Null, _) | (_, TerbiumObject::Null)) => {
                        push!(self.ctx, load_bool!(self.ctx, true))
                    },
                    _ => {
                        // TODO
                    }
                ),
                Instruction::OpLogicalNot => {
                    let subject = self.ctx.pop_ref();
                    let subject = self.ctx.store.resolve(subject);

                    match subject {
                        TerbiumObject::Bool(b) => push!(self.ctx, load_bool!(self.ctx, !*b)),
                        // TODO: support custom not operation
                        o => push!(self.ctx, load_bool!(self.ctx, !self.is_truthy(o))),
                    }
                },
                Instruction::Pop => {
                    self.ctx.pop_ref();
                }
                Instruction::Jump(addr) => match addr {
                    Addr::Absolute(a) => {
                        jump_history.push(pos.clone());
                        pos = a;
                        continue;
                    }
                    _ => panic!("attempted to run unresolved bytecode"),
                },
                Instruction::JumpIf(addr) => match addr {
                    Addr::Absolute(a) => {
                        let popped = self.ctx.pop_ref();
                        let popped = self.ctx.store.resolve(popped);

                        if self.is_truthy(popped) {
                            jump_history.push(pos.clone());
                            pos = a;
                            continue;
                        }
                    }
                    _ => panic!("attempted to run unresolved bytecode"),
                },
                Instruction::JumpIfElse(then, fb) => match (then, fb) {
                    (Addr::Absolute(then), Addr::Absolute(fb)) => {
                        jump_history.push(pos.clone());
                        let popped = self.ctx.pop_ref();
                        let popped = self.ctx.store.resolve(popped);

                        pos = if self.is_truthy(popped) { then } else { fb };
                        continue;
                    }
                    _ => panic!("attempted to run unresolved bytecode"),
                },
                Instruction::Ret => {
                    if let Some(back) = jump_history.pop() {
                        pos = back;
                    } else {
                        break;
                    }
                }
                Instruction::RetNull => {
                    self.ctx.push(0); // 0 is null
                    if let Some(back) = jump_history.pop() {
                        pos = back;
                    } else {
                        break;
                    }
                }
                Instruction::Halt => {
                    break;
                }
                Instruction::EnterScope => self.ctx.enter_scope(),
                Instruction::ExitScope => self.ctx.exit_scope(),
                Instruction::LoadVar(key) => {
                    push!(self.ctx, match self.ctx.lookup_var(key) {
                        Some(ScopeEntry { loc, .. }) => *loc,
                        None => {
                            todo!(); // TODO error if variable is not found
                        }
                    });
                }
                ref instr @ (
                    Instruction::StoreVar(key)
                    | Instruction::StoreMutVar(key)
                    | Instruction::StoreConstVar(key)
                ) => {
                    let loc = self.ctx.pop_ref();

                    self.ctx.store_var(key, ScopeEntry {
                        loc,
                        r#mut: matches!(instr, Instruction::StoreMutVar(_)),
                        r#const: matches!(instr, Instruction::StoreConstVar(_)),
                    })
                }
                _ => todo!(),
            }

            pos += 1;
        }
    }
}

pub type DefaultInterpreter = Interpreter<512>;

#[cfg(test)]
mod tests {
    use super::{DefaultInterpreter, TerbiumObject};
    use terbium_bytecode::{Instruction, Interpreter as Transformer, Program};
    use terbium_grammar::{Body, ParseInterface};

    #[test]
    fn test_interpreter() {
        let program = Program::from_iter([
            Instruction::LoadInt(1),
            Instruction::LoadInt(1),
            Instruction::BinOpAdd,
        ]);
        let mut interpreter = DefaultInterpreter::new();
        interpreter.run_bytecode(program);

        assert_eq!(interpreter.ctx.pop(), &TerbiumObject::Integer(2));
    }

    #[test]
    fn test_interpreter_from_string() {
        let code = r#"
            if 1 + 1 == 3 {
                0
            } else if 1 + 1 == 2 {
                1
            } else {
                2
            }
        "#;

        let (body, _errors) = Body::from_string(code.to_string()).unwrap_or_else(|e| {
            panic!("tokenization error: {:?}", e);
        });
        let mut transformer = Transformer::new();
        transformer.interpret_body(None, body);

        let mut program = transformer.program();
        program.resolve();

        let mut interpreter = DefaultInterpreter::new();
        interpreter.run_bytecode(program);

        assert_eq!(interpreter.ctx.pop(), &TerbiumObject::Integer(1));
    }
}
