//! The interpreter for Terbium.

#![feature(box_patterns)]

mod interner;

use std::collections::HashMap;
use terbium_bytecode::{Addr, AddrRepr, EqComparableFloat, Instruction, Program};

pub use interner::Interner;
use interner::StringId;

/// The integer type of the location of a `TerbiumObject`.
pub type ObjectRef = usize;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
/// The internal Terbium object model. These are created during the interpreter runtime.
pub enum TerbiumObject {
    Null,
    Integer(i128),
    Float(EqComparableFloat),
    String(StringId),
    Bool(bool),
}

#[derive(Debug)]
/// Represents stack wrapper around an array.
pub struct Stack<const STACK_SIZE: usize = 512> {
    pub(crate) inner: [ObjectRef; STACK_SIZE],
    pub(crate) ptr: usize,
}

impl<const STACK_SIZE: usize> Stack<STACK_SIZE> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inner: [0; STACK_SIZE],
            ptr: 0,
        }
    }

    /// Pushes the given object to the stack.
    pub fn push(&mut self, o: ObjectRef) {
        self.inner[self.ptr] = o;
        self.incr_ptr();
    }

    /// Increments `ptr` by 1.
    ///
    /// # Panics
    /// - The pointer has surpassed `STACK_SIZE`
    pub fn incr_ptr(&mut self) {
        self.ptr += 1;

        assert!(
            self.ptr < STACK_SIZE,
            "stack overflow (surpassed stack size of {})",
            STACK_SIZE
        );
    }

    /// Decrements `ptr` by 1.
    ///
    /// # Panics
    /// - The pointer is below 0
    pub fn decr_ptr(&mut self) {
        self.ptr = self.ptr.checked_sub(1).expect("stack ptr already at 0");
    }

    /// Pops the previous object in the stack and moves the pointer there.
    pub fn pop(&mut self) -> ObjectRef {
        self.decr_ptr();

        std::mem::replace(&mut self.inner[self.ptr], 0)
    }

    /// Gets a cloned version of the previous object in the stack,
    /// but also moves the pointer there.
    pub fn pop_cloned(&mut self) -> ObjectRef {
        self.decr_ptr();

        self.inner[self.ptr]
    }

    /// Retrieves a reference to the next free slot.
    #[must_use]
    pub const fn next_free(&self) -> &ObjectRef {
        &self.inner[self.ptr]
    }

    /// Retrieves a mutable reference to the free slot.
    pub fn next_free_mut(&mut self) -> &mut ObjectRef {
        &mut self.inner[self.ptr]
    }
}

impl Default for Stack {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
/// A wrapper around a `HashMap` that maps object locations to their actual objects.
/// This owns all objects throughout the interpreter runtime.
///
/// # Note
/// This was designed as a temporary solution; this struct may be removed in the future.
pub struct ObjectStore(pub(crate) HashMap<ObjectRef, TerbiumObject>);

impl ObjectStore {
    #[must_use]
    pub fn new() -> Self {
        let mut inner = HashMap::with_capacity(1);
        inner.insert(0 as ObjectRef, TerbiumObject::Null);

        Self(inner)
    }

    #[must_use]
    /// Resolves an object pointer into a reference of the object its pointing to.
    ///
    /// # Panics
    /// - There is no object stored at `loc`
    pub fn resolve(&self, loc: ObjectRef) -> &TerbiumObject {
        self.0
            .get(&loc)
            .expect(&*format!("no object at location {:0x}", loc))
    }

    #[must_use]
    /// Resolves an object pointer into a reference of the object its pointing to,
    /// but returns `&TerbiumObejct::Null` if no object is stored at `loc`.
    pub fn resolve_or_null(&self, loc: ObjectRef) -> &TerbiumObject {
        self.0.get(&loc).unwrap_or(&TerbiumObject::Null)
    }
}

impl Default for ObjectStore {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
/// Represents an entry in a scope.
///
/// Apart from which object this entry references,
/// it also stores metadata such was whether or not this
/// entry is a constant or if it is mutable.
pub struct ScopeEntry {
    pub loc: ObjectRef,
    r#mut: bool,
    r#const: bool,
}

impl ScopeEntry {
    #[must_use]
    /// Whether or not this entry is a constant.
    ///
    /// Constant objects cannot be overwritten in its own scope,
    /// reassigned to nor mutated.
    pub const fn is_const(&self) -> bool {
        self.r#const
    }

    #[must_use]
    /// Whether or not this entry is mutable.
    ///
    /// Objects which are **not** mutable are immutable,
    /// and cannot be reassigned or mutated.
    ///
    /// The difference between an immutable object and a constant
    /// is that an immutable object can be overwritten in its own scope,
    /// while a constant cannot be. In other words, Redefinition is
    /// allowed with immutable variables but not with constants.
    pub const fn is_mut(&self) -> bool {
        self.r#mut
    }
}

impl From<ScopeEntry> for ObjectRef {
    fn from(s: ScopeEntry) -> Self {
        s.loc
    }
}

#[derive(Debug)]
/// Represents a scope of identifiers.
pub struct Scope {
    pub locals: HashMap<usize, ScopeEntry>,
}

impl Scope {
    #[must_use]
    pub fn new() -> Self {
        Self {
            locals: HashMap::new(),
        }
    }
}

impl Default for Scope {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
/// Represents an interpreter's context during runtime.
pub struct Context<const STACK_SIZE: usize = 512> {
    pub store: ObjectStore,
    pub(crate) stack: Stack<STACK_SIZE>,
    scopes: Vec<Scope>,
    integer_lookup: HashMap<i128, ObjectRef>,
    bool_lookup: [ObjectRef; 2],
}

impl<const STACK_SIZE: usize> Context<STACK_SIZE> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            store: ObjectStore::default(),
            stack: Stack::new(),
            scopes: vec![Scope::default()],
            integer_lookup: HashMap::new(),
            bool_lookup: [0, 0],
        }
    }

    /// Pushes an object by location to the stack.
    pub fn push(&mut self, o: ObjectRef) {
        self.stack.push(o);
    }

    /// Pops the last object from the stack and returns its location.
    pub fn pop_ref(&mut self) -> ObjectRef {
        self.stack.pop()
    }

    /// Pops the last object from the stack and returns a tuple containing
    /// the location of the object in field 0 and a reference to the resolved
    /// object in field 1.
    pub fn pop_detailed(&mut self) -> (ObjectRef, &TerbiumObject) {
        let loc = self.stack.pop();

        (loc, self.store.resolve(loc))
    }

    /// Pops the last object from the stack and returns a reference to it.
    pub fn pop(&mut self) -> &TerbiumObject {
        let loc = self.pop_ref();

        self.store.resolve(loc)
    }

    /// Pops the last object from the stack and returns a reference to it.
    ///
    /// If any of the following happen, `&TerbiumObject::Null` is returned instead:
    ///
    /// - Nothing is on the stack
    /// - What was popped from the stack could not be resolved into an object
    pub fn pop_or_null(&mut self) -> &TerbiumObject {
        let loc = self.pop_ref();

        self.store.resolve_or_null(loc)
    }

    /// Pops the last object from the stack, clones it, and returns it.
    pub fn pop_cloned(&mut self) -> TerbiumObject {
        *self.store.resolve(self.stack.pop_cloned())
    }

    /// Stores the object at the given location.
    pub fn store(&mut self, loc: ObjectRef, o: TerbiumObject) -> ObjectRef {
        self.store.0.insert(loc, o);

        loc
    }

    /// Stores the object at a pre-determined location.
    pub fn store_auto(&mut self, o: TerbiumObject) -> ObjectRef {
        // TODO: this is O(n), not really the best
        let key = self.store.0.keys().max().unwrap_or(&0) + 1;
        self.store(key, o);

        key
    }

    /// Loads the given integer and returns its location.
    pub fn load_int(&mut self, i: i128) -> ObjectRef {
        let loc = self.store_auto(TerbiumObject::Integer(i));

        *self.integer_lookup.entry(i).or_insert(loc)
    }

    /// Loads the given bool and returns its location.
    pub fn load_bool(&mut self, b: bool) -> ObjectRef {
        let index = usize::from(b);
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

    #[must_use]
    /// Returns a reference to the current local scope.
    pub fn locals(&self) -> &Scope {
        self.scopes.last().unwrap_or_else(|| unreachable!())
    }

    /// Returns a mutable reference to the current local scope.
    pub fn locals_mut(&mut self) -> &mut Scope {
        self.scopes.last_mut().unwrap_or_else(|| unreachable!())
    }

    /// Stores the `ScopeEntry` in the given `key`
    pub fn store_var(&mut self, key: usize, entry: ScopeEntry) {
        self.locals_mut().locals.insert(key, entry);
    }

    /// Reassigns the value at the given location to the given `key`.
    pub fn assign_var(&mut self, key: usize, value: ObjectRef) {
        let entry = self.lookup_var_mut(key).unwrap_or_else(|| {
            // TODO: variable not found, error
            unimplemented!();
        });

        if entry.is_const() || !entry.is_mut() {
            // TODO: variable is immutable or const, error
            unimplemented!();
        }

        entry.loc = value;
    }

    #[must_use]
    pub fn lookup_var(&self, key: usize) -> Option<&ScopeEntry> {
        for scope in self.scopes.iter().rev() {
            if let Some(entry) = scope.locals.get(&key) {
                return Some(entry);
            }
        }

        None
    }

    pub fn lookup_var_mut(&mut self, key: usize) -> Option<&mut ScopeEntry> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(entry) = scope.locals.get_mut(&key) {
                return Some(entry);
            }
        }

        None
    }

    pub fn enter_scope(&mut self) {
        self.scopes.push(Scope::default());
    }

    pub fn exit_scope(&mut self) {
        self.scopes.pop();
    }
}

impl<const STACK_SIZE: usize> Default for Context<STACK_SIZE> {
    fn default() -> Self {
        Self::new()
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
    }};
}

macro_rules! store_auto {
    ($ctx:expr, $e:expr) => {
        deferred_method!($ctx, store_auto, $e)
    };
}

macro_rules! load_int {
    ($ctx:expr, $e:expr) => {
        deferred_method!($ctx, load_int, $e)
    };
}

macro_rules! load_bool {
    ($ctx:expr, $e:expr) => {
        deferred_method!($ctx, load_bool, $e)
    };
}

macro_rules! push {
    ($ctx:expr, $e:expr) => {
        deferred_method!($ctx, push, $e)
    };
}

impl<const STACK_SIZE: usize> Interpreter<STACK_SIZE> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            ctx: Context::new(),
            // TODO: string length capacity to be interned could be configurable
            string_interner: Interner::with_capacity(128),
        }
    }

    #[must_use]
    pub fn stack(&mut self) -> &mut Stack<STACK_SIZE> {
        &mut self.ctx.stack
    }

    #[must_use]
    pub fn string_lookup(&self, id: StringId) -> &str {
        self.string_interner.lookup(id)
    }

    #[must_use]
    pub fn is_truthy(&self, o: &TerbiumObject) -> bool {
        match o {
            TerbiumObject::Bool(b) => *b,
            TerbiumObject::Integer(i) => *i != 0,
            TerbiumObject::Float(EqComparableFloat(f)) => *f != 0_f64,
            TerbiumObject::String(s) => !self.string_interner.lookup(*s).is_empty(),
            TerbiumObject::Null => false,
        }
    }

    #[must_use]
    pub fn get_bool_object(&mut self, o: &TerbiumObject) -> ObjectRef {
        load_bool!(self.ctx, self.is_truthy(o))
    }

    #[allow(clippy::too_many_lines)]
    #[allow(clippy::missing_panics_doc)] // Remove when todo is done
    #[allow(clippy::cast_possible_wrap)] // Wrap is not possible because it is parsed as i128
    #[allow(clippy::cast_precision_loss)]
    /// Integers with a mantissa exceeding a width of 52 bits will be wrapped to
    /// 340282366920938500000000000000000000000.
    ///
    /// This will cause unexpected behavior because if it is compared with, for example,
    /// 340282366920938463463374607431768211455 (`u128::MAX`), `true` is returned.
    ///
    /// Furthermore, if it is compared with 340282366920938463463374607431768211454 (`u128::MAX` - 1),
    /// true is also returned.
    ///
    /// This is unexpected behavior as the three integers fundamentally do not equal each other,
    /// however the interpreter thinks it is.
    ///
    /// Currently, the solution is to let `terbium_analyzer` catch this as a hard error
    /// to prevent this behavior from being executed.
    pub fn run_bytecode(&mut self, code: &Program) {
        let mut pos: AddrRepr = 0;
        let mut jump_history: Vec<AddrRepr> = Vec::new();
        let instructions = code.inner().collect::<Vec<_>>();

        loop {
            let instr = instructions[pos as usize];

            // let span = instr.span();
            let instr = instr.instr();

            match instr.clone() {
                Instruction::LoadInt(i) => push!(self.ctx, load_int!(self.ctx, i as i128)),
                Instruction::LoadString(s) => push!(
                    self.ctx,
                    store_auto!(
                        self.ctx,
                        TerbiumObject::String(self.string_interner.intern(s.as_str()),)
                    )
                ),
                Instruction::LoadFloat(f) => {
                    push!(self.ctx, store_auto!(self.ctx, TerbiumObject::Float(f)));
                }
                Instruction::LoadBool(b) => push!(self.ctx, load_bool!(self.ctx, b)),
                Instruction::UnOpPos => match self.ctx.pop_detailed() {
                    (o, TerbiumObject::Integer(_) | TerbiumObject::Float(_)) => self.ctx.push(o),
                    _ => todo!(),
                },
                Instruction::UnOpNeg => match self.ctx.pop() {
                    TerbiumObject::Integer(i) => push!(self.ctx, load_int!(self.ctx, -*i)),
                    TerbiumObject::Float(f) => push!(
                        self.ctx,
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

                        self.ctx.push(loc);
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
                                #[allow(clippy::cast_sign_loss)] // Acts as a safety barrier to prevent overflow
                                self.string_interner.lookup(*lhs).repeat(*rhs as usize).as_str(),
                            )
                        ));
                        self.ctx.push(loc);
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
                        self.ctx.push(b);
                    },
                    (TerbiumObject::Bool(rhs), TerbiumObject::Bool(lhs)) => {
                        let b = load_bool!(self.ctx, lhs == rhs);
                        self.ctx.push(b);
                    },
                    (TerbiumObject::Null, TerbiumObject::Null) => {
                        push!(self.ctx, load_bool!(self.ctx, true));
                    },
                    ((TerbiumObject::Null, _) | (_, TerbiumObject::Null)) => {
                        push!(self.ctx, load_bool!(self.ctx, false));
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
                        self.ctx.push(b);
                    },
                    (TerbiumObject::Bool(rhs), TerbiumObject::Bool(lhs)) => {
                        let b = load_bool!(self.ctx, lhs != rhs);
                        self.ctx.push(b);
                    },
                    (TerbiumObject::Null, TerbiumObject::Null) => {
                        push!(self.ctx, load_bool!(self.ctx, false));
                    },
                    ((TerbiumObject::Null, _) | (_, TerbiumObject::Null)) => {
                        push!(self.ctx, load_bool!(self.ctx, true));
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
                }
                Instruction::Pop => {
                    self.ctx.pop_ref();
                }
                Instruction::Jump(addr) => match addr {
                    Addr::Absolute(a) => {
                        jump_history.push(pos);
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
                            jump_history.push(pos);
                            pos = a;
                            continue;
                        }
                    }
                    _ => panic!("attempted to run unresolved bytecode"),
                },
                Instruction::JumpIfElse(then, fb) => match (then, fb) {
                    (Addr::Absolute(then), Addr::Absolute(fb)) => {
                        jump_history.push(pos);
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
                    push!(
                        self.ctx,
                        match self.ctx.lookup_var(key) {
                            Some(ScopeEntry { loc, .. }) => *loc,
                            None => {
                                todo!(); // TODO error if variable is not found
                            }
                        }
                    );
                }
                ref instr @ (Instruction::StoreVar(key)
                | Instruction::StoreMutVar(key)
                | Instruction::StoreConstVar(key)) => {
                    let loc = self.ctx.pop_ref();

                    self.ctx.store_var(
                        key,
                        ScopeEntry {
                            loc,
                            r#mut: matches!(instr, Instruction::StoreMutVar(_)),
                            r#const: matches!(instr, Instruction::StoreConstVar(_)),
                        },
                    );
                }
                Instruction::AssignVar(key) => {
                    let loc = self.ctx.pop_ref();

                    self.ctx.assign_var(key, loc);
                }
                _ => todo!(),
            }

            pos += 1;
        }
    }

    #[must_use]
    pub fn get_object_repr(&self, o: &TerbiumObject) -> String {
        match o {
            TerbiumObject::Integer(i) => i.to_string(),
            TerbiumObject::Float(f) => f.0.to_string(),
            TerbiumObject::String(s_id) => format!("{:?}", self.string_lookup(*s_id)),
            TerbiumObject::Bool(b) => b.to_string(),
            TerbiumObject::Null => "null".to_string(),
        }
    }
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

pub type DefaultInterpreter = Interpreter<512>;
