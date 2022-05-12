//! The interpreter for Terbium.
//!
//! Design issues to address in the future:
//! - Wrapping practically all TerbiumObjects inside a RwLock is probably more inefficient than it should be.
//!   We could just store a Reference ID, but then it comes at the cost of a HashMap lookup every time we want to access that object.
//! - Native function lookup is by strings, not a really good design principle.
//! - The interpreter only uses an Rc and is therefore not Send + Sync. Using Arc comes with performance downgrades, however.

#![feature(map_try_insert)]

mod builtin;
mod stdlib;

use terbium_grammar::ast::Param;
pub(crate) use terbium_grammar::{Body, Expr, Node, Operator};

use std::{
    collections::HashMap,
    hash::Hash,
    rc::Rc,
    sync::atomic::{AtomicU32, Ordering},
    sync::RwLock,
};

pub(crate) type WrappedTerbiumObject = Rc<RwLock<TerbiumObject>>;
pub(crate) type NativeFunctionResult = Result<WrappedTerbiumObject, TerbiumExceptionType>;

#[derive(Debug)]
pub struct TerbiumScope<'s> {
    pub parent: Option<Box<&'s TerbiumScope<'s>>>,
    pub members: HashMap<String, TerbiumObject>,
}

impl TerbiumScope {
    pub fn new() -> Self {
        Self {
            parent: None,
            members: HashMap::new(),
        }
    }

    pub fn scope(&self) -> Self {
        Self {
            parent: Some(Box::new(&self)),
            members: HashMap::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum TerbiumExceptionType {
    AnyError(String),
    CtrlCError,
    ModuleNotFound(String),
    Error(String),
    SyntaxError(String),
    TypeError(String),
    ValueError(String),
    SignatureError(String),
    RuntimeError(String),
    ReferenceError(String),
    AttributeError(String),
}

pub trait AsTerbiumObject {
    fn as_terbium_object(&self) -> TerbiumObject;

    fn alloc_terbium_object(&self, store: &mut TerbiumObjectStore) -> WrappedTerbiumObject {
        store.alloc(self.as_terbium_object())
    }
}

impl<T: AsTerbiumObject> From<T> for TerbiumObject {
    fn from(obj: T) -> Self {
        obj.as_terbium_object()
    }
}

impl AsTerbiumObject for TerbiumExceptionType {
    fn as_terbium_object(&self) -> TerbiumObject {
        TerbiumObject::new()
    }
}

#[derive(Clone, Debug)]
pub enum TerbiumType {
    Int(i128),
    Float(f64),
    Bool(bool),
    String(String),
    Array(Box<TerbiumType>, Vec<WrappedTerbiumObject>), // ItemType, inner_objects
    Null,
    NativeFunction(String),
    Function(Vec<Param>, Body),
    Method {
        cls: Box<Rc<TerbiumType>>,
        name: String,
        params: Vec<Param>,
        body: Body,
        bound: bool,
    },
    Class(Vec<WrappedTerbiumObject>),
    Module,
    NativeType,
    Error(TerbiumExceptionType),
}

impl AsTerbiumObject for TerbiumType {
    fn as_terbium_object(&self) -> TerbiumObject {}
}

#[derive(Clone, Debug)]
pub struct TerbiumObject {
    pub name: String,
    pub attrs: HashMap<String, WrappedTerbiumObject>,
    pub mutable: bool,
    pub ty: TerbiumType,
    pub ops: HashMap<String, WrappedTerbiumObject>,
    pub casts: HashMap<TerbiumType, WrappedTerbiumObject>,
    pub ref_count: u32,
    pub store_id: u32,
}

impl TerbiumObject {
    pub fn default_ops(store: &mut TerbiumObjectStore) -> HashMap<String, WrappedTerbiumObject> {
        let mut ops = HashMap::new();

        ops.insert(
            "repr".to_string(),
            TerbiumObject::alloc_native_function(store, "repr", "repr"),
        );
        ops
    }

    pub fn native(
        name: String,
        attrs: HashMap<String, WrappedTerbiumObject>,
        mutable: bool,
        ty: TerbiumType,
    ) -> Self {
        Self {
            name,
            attrs,
            mutable,
            ty,
            ops: HashMap::new(),
            casts: HashMap::new(),
            ref_count: 0,
            store_id: 0,
        }
    }

    pub fn native_function(name: impl ToString, lookup_name: impl ToString) -> Self {
        Self::native(
            name,
            HashMap::new(),
            false,
            TerbiumType::NativeFunction(lookup_name.to_string()),
        )
    }

    pub fn alloc_native_function(
        store: &mut TerbiumObjectStore,
        name: impl ToString,
        lookup_name: impl ToString,
    ) -> WrappedTerbiumObject {
        store.alloc(Self::native_function(name, lookup_name))
    }

    pub fn new(
        store: &mut TerbiumObjectStore,
        name: String,
        attrs: HashMap<String, WrappedTerbiumObject>,
        mutable: bool,
        ty: TerbiumType,
    ) -> Self {
        Self {
            name,
            attrs,
            mutable,
            ty,
            ops: Self::default_ops(store),
            casts: HashMap::new(),
            ref_count: 0,
            store_id: 0,
        }
    }

    pub fn alloc_new(
        store: &mut TerbiumObjectStore,
        name: String,
        attrs: HashMap<String, WrappedTerbiumObject>,
        mutable: bool,
        ty: TerbiumType,
    ) -> WrappedTerbiumObject {
        store.alloc(Self::new(store, name, attrs, mutable, ty))
    }
}

#[derive(Debug)]
pub struct TerbiumSingletonObjectLookup {
    pub null: WrappedTerbiumObject,
    pub type_object: WrappedTerbiumObject,
    pub int_type: WrappedTerbiumObject,
    pub float_type: WrappedTerbiumObject,
    pub bool_type: WrappedTerbiumObject,
    pub string_type: WrappedTerbiumObject,
    pub array_type: WrappedTerbiumObject,
    pub function_type: WrappedTerbiumObject,
    pub module_type: WrappedTerbiumObject,
    pub r#true: WrappedTerbiumObject,
    pub r#false: WrappedTerbiumObject,
    pub integer_lookup: HashMap<i128, WrappedTerbiumObject>,
}

impl TerbiumSingletonObjectLookup {
    pub fn new(store: &mut TerbiumObjectStore) -> Self {
        let null = TerbiumObject::alloc_new(
            store,
            "null".to_string(),
            HashMap::new(),
            false,
            TerbiumType::Null,
        );

        let type_object = TerbiumObject::alloc_new(
            store,
            "type".to_string(),
            HashMap::new(),
            false,
            TerbiumType::NativeType,
        );

        let int_type = TerbiumObject::alloc_new(
            store, 
            "int".to_string(),
            HashMap::new(), 
            false, 
            TerbiumType::Class(type_object),
        );

        Self {
            null,
            type_object,
            int_type,
        }
    }

    pub fn int(&mut self, store: &mut TerbiumObjectStore, i: i128) -> WrappedTerbiumObject {
        if self.integer_lookup.contains_key(&i) {
            return self.integer_lookup.get(&i);
        }
        
        self.integer_lookup.insert(i, TerbiumObject::alloc_new(
            store,
            "int_o".to_string(),
            HashMap::new(),
            false,
            TerbiumType::Int(i),
        ))
    }
}

#[derive(Debug)]
pub struct TerbiumObjectStore {
    objects: HashMap<u32, WrappedTerbiumObject>,
    pub(crate) increment: AtomicU32,
}

impl TerbiumObjectStore {
    pub fn new() -> Self {
        Self {
            objects: HashMap::new(),
            increment: AtomicU32::new(0),
        }
    }

    pub fn alloc(&mut self, mut o: TerbiumObject) -> WrappedTerbiumObject {
        o.store_id = self.increment.fetch_add(1, Ordering::SeqCst);

        let rc = Rc::new(RwLock::new(o));
        self.objects.insert(rc.read().unwrap().store_id, rc);

        rc.clone()
    }

    pub fn get(&self, id: u32) -> Option<WrappedTerbiumObject> {
        self.objects.get(&id).map(|rc| rc.clone())
    }
}

#[derive(Debug)]
pub struct Interpreter<'s> {
    pub globals: TerbiumScope<'s>,
    pub singletons: TerbiumSingletonObjectLookup,
    pub current_scope: Option<TerbiumScope<'s>>,
    pub objects: TerbiumObjectStore,
}

impl<'s> Interpreter<'s> {
    pub fn new() -> Self {
        let mut objects = TerbiumObjectStore::new();

        Self {
            globals: TerbiumScope::new(),
            singletons: TerbiumSingletonObjectLookup::new(&mut objects),
            current_scope: None,
            objects,
        }
    }

    pub fn load_native_module(
        &mut self,
        name: String,
    ) -> Result<WrappedTerbiumObject, WrappedTerbiumObject> {
        let mut attrs = HashMap::new();

        match name.as_str() {
            "std" => {
                attrs.insert(
                    "println".to_string(),
                    TerbiumObject::native_function("println"),
                );
            }
            _ => Err(TerbiumExceptionType::ModuleNotFound(name).alloc_terbium_object(self))?,
        }

        Ok(TerbiumObject::new(
            &mut self.objects,
            name.clone(),
            attrs,
            true,
            TerbiumType::Module,
        ))
    }

    pub fn get_native_function(&mut self, name: String) -> (TerbiumObject, F)
    where
        F: Fn(&mut Self, Vec<WrappedTerbiumObject>) -> NativeFunctionResult + 'static,
    {
        match name.as_str() {}
    }

    pub fn get_int(&mut self, i: i128) -> WrappedTerbiumObject {
        self.singletons.int(self.objects, i)
    }

    pub fn is_instance(&self, obj: &WrappedTerbiumObject, ty: &WrappedTerbiumObject) -> bool {
        let obj = obj.read().unwrap().clone();
        let ty_store_id = ty.read().unwrap().store_id;

        match obj.ty {
            TerbiumType::Class(bases) => bases
                .iter()
                .any(|base| base.read().unwrap().store_id == ty_store_id),
            TerbiumType::Int(_) => ty_store_id == self.singletons.int_type.store_id,
            TerbiumType::Float(_) => ty_store_id == self.singletons.float_type.store_id
                || ty_store_id == self.singletons.int_type.store_id,
            TerbiumType::String(_) => ty_store_id == self.singletons.string_type.store_id,
            TerbiumType::Bool(_) => ty_store_id == self.singletons.bool_type.store_id,
            _ => todo!(),
        }
    }

    pub fn eval_expr(&mut self, expr: Expr) -> WrappedTerbiumObject {
        match expr {
            Expr::Int(i) => self.get_int(i),
            Expr::BinaryOp { operator, lhs, rhs } => {
                let lhs = self.eval_expr(lhs);
                let rhs = self.eval_expr(rhs);

                match operator {
                    Operator::Add => todo!(),
                    _ => todo!(),
                }
            }
            _ => todo!(),
        }
    }
}
