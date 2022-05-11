mod stdlib;

pub(crate) use terbium_grammar::{Body, Expr, Node};
use terbium_grammar::ast::Param;

use std::collections::HashMap;

#[derive(Debug)]
pub struct TerbiumScope {
    pub parent: Option<Box<TerbiumScope>>,
    pub members: HashMap<String, TerbiumValue>,
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

impl TerbiumExceptionType {
    pub fn as_terbium_object(&self) -> TerbiumObject {
        TerbiumObject::new()
    }
}

#[derive(Clone, Debug)]
pub enum TerbiumType {
    Int(i128),
    Float(f64),
    Bool(bool),
    String(String),
    Array(Box<TerbiumType>, Vec<TerbiumObject>), // ItemType, inner_objects
    Null,
    NativeFunction(String),
    Function(Vec<Param>, Body),
    Method {
        cls: Box<TerbiumType>,
        name: String,
        params: Vec<Param>,
        body: Body,
        bound: bool,
    },
    Class {
        name: String,
        bases: Vec<TerbiumType>,
        attrs: HashMap<String, TerbiumObject>,
        ops: HashMap<String, TerbiumObject>,
        casts: HashMap<TerbiumType, TerbiumObject>,
    },
    Module,
    NativeType,
    Error(TerbiumExceptionType),
}

#[derive(Clone, Debug)]
pub struct TerbiumObject {
    pub name: String,
    pub attrs: HashMap<String, TerbiumObject>,
    pub mutable: bool,
    pub ty: TerbiumType,
    pub ops: HashMap<String, TerbiumObject>,
    pub casts: HashMap<TerbiumType, TerbiumObject>,
}

impl TerbiumObject {
    const NULL: Self = Self::native(
        "null".to_string(),
        HashMap::new(),
        false,
        TerbiumType::Null,
    );

    pub fn default_ops() -> HashMap<String, TerbiumObject> {
        let mut ops = HashMap::new();

        ops.insert("repr".to_string(), TerbiumObject::native_function("repr"));
        ops
    }

    pub fn native(name: String, attrs: HashMap<String, TerbiumObject>, mutable: bool, ty: TerbiumType) -> Self {
        Self {
            name,
            attrs,
            mutable,
            ty,
            ops: HashMap::new(),
            casts: HashMap::new(),
        }
    }

    pub fn native_function(name: impl ToString) -> Self {
        Self::native(name, HashMap::new(), false, TerbiumType::NativeFunction(name.to_string()))
    }

    pub fn new(name: String, attrs: HashMap<String, TerbiumObject>, mutable: bool, ty: TerbiumType) -> Self {
        Self {
            name,
            attrs,
            mutable,
            ty,
            ops: Self::default_ops(),
            casts: HashMap::new(),
        }
    }
}

pub fn load_native_module(name: String) -> Result<TerbiumObject, TerbiumObject> {
    let mut attrs = HashMap::new();

    match name.as_str() {
        "std" => {
            attrs.insert("println".to_string(), TerbiumObject::native_function("println"));
        },
        _ => Err(TerbiumExceptionType::ModuleNotFound(name).as_terbium_object())?,
    }

    Ok(TerbiumObject::new(name.clone(), attrs, true, TerbiumType::Module))
}

pub fn get_native_function(name: String) -> (TerbiumObject, F)
where F:
    Fn(&mut TerbiumScope, Vec<TerbiumObject>) -> Result<TerbiumObject, TerbiumExceptionType> + 'static,
{
    match name.as_str() {

    }
}