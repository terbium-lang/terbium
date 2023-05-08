//! High-level intermediate representation. This IR is used for type analysis, validating code
//! correctness, and desugaring.

pub mod lower;

use internment::Intern;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Ident(Intern<String>);

/// The ID of a module.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModuleId(Intern<Vec<String>>);

/// The ID of a top-level item.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ItemId {
    /// The key of the item, which is unique within the module.
    pub key: Ident,
    /// The module in which the item is defined.
    pub module: ModuleId,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ScopeId(usize);

impl ScopeId {
    pub const fn next(&self) -> Self {
        Self(self.0 + 1)
    }
}

/// HIR of a Terbium program.
pub struct Hir {
    /// A mapping of all top-level functions in the program.
    pub funcs: HashMap<ItemId, Func>,
    /// A mapping of all constants in the program.
    pub consts: HashMap<ItemId, Const>,
    /// A mapping of all lexical scopes within the program.
    pub scopes: HashMap<ScopeId, Scope>,
    /// The root scope of the program.
    pub root: ScopeId,
}

/// An HIR node.
#[derive(Clone, Debug)]
pub enum Node {
    Expr(Expr),
    Let {
        pat: Pattern,
        ty: Ty,
        value: Option<Expr>,
    },
    Const(Const),
    Func(Func),
    Break(Option<Ident>, Option<Expr>),
    Continue(Option<Ident>),
    Return(Option<Expr>),
}

#[derive(Clone, Debug)]
pub struct Scope {
    pub label: Option<Ident>,
    pub parent: Option<ScopeId>,
    pub children: Vec<Node>,
}

#[derive(Clone, Debug)]
pub struct Const {
    pub name: Ident,
    pub ty: Ty,
    pub value: Expr,
}

/// A pattern that can be matched against.
#[derive(Clone, Debug)]
pub enum Pattern {
    Ident { ident: Ident, is_mut: bool },
    Tuple(Vec<Self>),
}

/// Visibility of a top-level item.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ItemVisibility {
    /// The item is visible to all other items in the program.
    Public,
    /// The item is visible to all other items in the library.
    Lib,
    /// The item is visible to all items in the parent module and its submodules.
    Super,
    /// The item is only visible to the current module. This is the default visibility.
    Private,
}

#[derive(Clone, Debug)]
pub struct FuncParam {
    pub pat: Pattern,
    pub ty: Ty,
    pub default: Option<Expr>,
}

/// HIR of a top-level function.
#[derive(Clone, Debug)]
pub struct Func {
    /// The visibility of the item.
    pub visibility: ItemVisibility,
    /// The name of the function.
    pub name: Ident,
    /// The parameters of the function.
    pub params: Vec<FuncParam>,
    /// The return type of the function.
    pub ret_ty: Ty,
    /// The body of the function.
    pub body: ScopeId,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[repr(u8)]
pub enum IntWidth {
    Int8 = 8,
    Int16 = 16,
    Int32 = 32,
    Int64 = 64,
    Int128 = 128,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum IntSign {
    Signed,
    Unsigned,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[repr(u8)]
pub enum FloatWidth {
    Float32 = 32,
    Float64 = 64,
    // Float128 = 128,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PrimitiveTy {
    // integer "signedness" and bit-width are unified as one type since they coerce to each other
    Int(IntSign, IntWidth),
    Float(FloatWidth),
    Bool,
    Char,
    Void,
}

#[derive(Clone, Debug)]
pub enum Ty {
    Unknown,
    Primitive(PrimitiveTy),
    Tuple(Vec<Ty>),
}

#[derive(Clone, Debug)]
pub enum Literal {
    UInt(u128),
    Int(i128),
    Float(f64),
    Bool(bool),
    Char(char),
    String(String),
    Void,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Intrinsic {
    IntAdd,
    IntSub,
    IntMul,
    IntDiv,
    IntMod,
    FloatAdd,
    FloatSub,
    FloatMul,
    FloatDiv,
    FloatMod,
    IntEq,
    IntLt,
    IntLe,
    IntGt,
    IntGe,
    FloatEq,
    FloatLt,
    FloatLe,
    FloatGt,
    FloatGe,
    BoolEq,
}

#[derive(Clone, Debug)]
pub enum Expr {
    Literal(Literal),
    Ident(Ident),
    Tuple(Vec<Self>),
    Intrinsic(Intrinsic, Vec<Self>),
    Call {
        callee: Box<Self>,
        args: Vec<Self>,
        kwargs: Vec<(Ident, Self)>,
    },
    Cast(Box<Self>, Ty),
    GetAttr(Box<Self>, Ident),
    SetAttr(Box<Self>, Ident, Box<Self>),
    GetIndex(Box<Self>, Box<Self>),
    SetIndex(Box<Self>, Box<Self>, Box<Self>),
    Block(ScopeId),
    If(Box<Self>, ScopeId, ScopeId),
    While(Box<Self>, ScopeId, ScopeId),
    Loop(ScopeId),
    Assign(Pattern, Box<Self>),
}
