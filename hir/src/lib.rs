//! High-level intermediate representation. This IR is used for type analysis, validating code
//! correctness, and desugaring.

#![feature(let_chains)]

pub mod error;
pub mod lower;

use common::span::{Span, Spanned, Src};
use grammar::ast;
use internment::Intern;
use std::{
    collections::HashMap,
    fmt::{Display, Formatter},
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Ident(Intern<String>);

impl Display for Ident {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The ID of a module.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModuleId(Intern<Vec<String>>);

impl Display for ModuleId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.0
                .iter()
                .map(AsRef::as_ref)
                .collect::<Vec<_>>()
                .join(".")
        )
    }
}

impl ModuleId {
    pub fn root() -> Self {
        Self(Intern::new(Vec::new()))
    }
}

impl From<Src> for ModuleId {
    fn from(src: Src) -> Self {
        Self(match src {
            Src::None => return Self::root(),
            Src::Repl => Intern::new(vec!["<repl>".to_string()]),
            Src::Path(p) => p,
        })
    }
}

/// The ID of a top-level item.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ItemId(
    /// The module in which the item is defined.
    ModuleId,
    /// The name of the item, which is unique within the module.
    Ident,
);

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ScopeId(usize);

impl ScopeId {
    pub const ROOT: Self = Self(0);

    pub const fn next(&self) -> Self {
        Self(self.0 + 1)
    }
}

/// HIR of a Terbium program.
#[derive(Debug)]
pub struct Hir {
    /// A mapping of all modules within the program.
    pub modules: HashMap<ModuleId, ScopeId>,
    /// A mapping of all top-level functions in the program.
    pub funcs: HashMap<ItemId, Func>,
    /// A mapping of all constants in the program.
    pub consts: HashMap<ItemId, Const>,
    /// A mapping of all raw structs within the program.
    pub structs: HashMap<ItemId, StructTy>,
    /// A mapping of all types within the program.
    pub types: HashMap<ItemId, TyDef>,
    /// A mapping of all lexical scopes within the program.
    pub scopes: HashMap<ScopeId, Scope>,
}

impl Default for Hir {
    fn default() -> Self {
        Self {
            modules: HashMap::from([(ModuleId::root(), ScopeId::ROOT)]),
            funcs: HashMap::new(),
            consts: HashMap::new(),
            structs: HashMap::new(),
            types: HashMap::new(),
            scopes: HashMap::new(),
        }
    }
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
    pub children: Vec<Node>,
}

#[derive(Clone, Debug)]
pub struct Const {
    pub name: Spanned<Ident>,
    pub ty: Ty,
    pub value: Expr,
}

#[inline]
fn assert_equal_params_length(
    span: Span,
    ty_name: Spanned<Ident>,
    ty_params_len: usize,
    ty_args_len: usize,
) -> Result<(), error::AstLoweringError> {
    if ty_args_len != ty_params_len {
        return Err(error::AstLoweringError::IncorrectTypeArgumentCount {
            span,
            ty: ty_name.as_ref().map(ToString::to_string),
            expected: ty_params_len,
            actual: ty_args_len,
        });
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct TyDef {
    pub name: Spanned<Ident>,
    pub ty: Ty,
    pub ty_params: Vec<TyParam>,
}

impl TyDef {
    pub fn apply_params(&self, span: Span, params: Vec<Ty>) -> Result<Ty, error::AstLoweringError> {
        assert_equal_params_length(span, self.name, self.ty_params.len(), params.len())?;

        let mut ty = self.ty.clone();
        for (param, arg) in self.ty_params.iter().zip(params) {
            ty = ty.subst(param, arg);
        }
        Ok(ty)
    }
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

impl ItemVisibility {
    pub fn from_ast(v: ast::ItemVisibility) -> Self {
        match v {
            ast::ItemVisibility::Public => Self::Public,
            ast::ItemVisibility::Lib => Self::Lib,
            ast::ItemVisibility::Super => Self::Super,
            ast::ItemVisibility::Mod => Self::Private,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MemberVisibility {
    Public,
    Lib,
    Super,
    Mod,
    Sub,
    Private,
}

impl MemberVisibility {
    pub fn from_ast(v: ast::MemberVisibility) -> Self {
        match v {
            ast::MemberVisibility::Public => Self::Public,
            ast::MemberVisibility::Lib => Self::Lib,
            ast::MemberVisibility::Super => Self::Super,
            ast::MemberVisibility::Mod => Self::Mod,
            ast::MemberVisibility::Sub => Self::Sub,
            ast::MemberVisibility::Private => Self::Private,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FieldVisibility {
    pub get: MemberVisibility,
    pub set: MemberVisibility,
}

impl FieldVisibility {
    pub fn from_ast(v: ast::FieldVisibility) -> Self {
        Self {
            get: MemberVisibility::from_ast(v.get.0),
            set: MemberVisibility::from_ast(v.set.0),
        }
    }
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
    pub name: Spanned<Ident>,
    /// The parameters of the function.
    pub params: Vec<FuncParam>,
    /// The return type of the function.
    pub ret_ty: Ty,
    /// The body of the function.
    pub body: ScopeId,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub enum IntWidth {
    Int8 = 8,
    Int16 = 16,
    #[default]
    Int32 = 32,
    Int64 = 64,
    Int128 = 128,
    Unknown = !0,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum IntSign {
    Signed,
    Unsigned,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub enum FloatWidth {
    Float32 = 32,
    #[default]
    Float64 = 64,
    Unknown = !0,
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
    Generic(Ident),
    Tuple(Vec<Ty>),
    Struct(ItemId, Vec<Ty>),
}

impl Ty {
    fn subst(self, param: &TyParam, ty: Ty) -> Self {
        match self {
            Self::Generic(p) if p == param.name => ty,
            Self::Tuple(tys) => Self::Tuple(
                tys.into_iter()
                    .map(|t| t.subst(param, ty.clone()))
                    .collect(),
            ),
            Self::Struct(item, tys) => Self::Struct(
                item,
                tys.into_iter()
                    .map(|t| t.subst(param, ty.clone()))
                    .collect(),
            ),
            other => other,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TyParam {
    pub name: Ident,
    pub bound: Option<Box<Ty>>,
}

#[derive(Clone, Debug)]
pub struct StructField {
    pub vis: FieldVisibility,
    pub name: Ident,
    pub ty: Ty,
    pub default: Option<Expr>,
}

#[derive(Clone, Debug)]
pub struct StructTy {
    pub vis: ItemVisibility,
    pub name: Spanned<Ident>,
    pub ty_params: Vec<TyParam>,
    pub fields: Vec<StructField>,
}

impl StructTy {
    pub fn into_adhoc_struct_ty_with_applied_ty_params(
        self,
        span: Option<Span>,
        params: Vec<Ty>,
    ) -> Result<Self, error::AstLoweringError> {
        assert_equal_params_length(
            span.unwrap_or(self.name.span()),
            self.name,
            self.ty_params.len(),
            params.len(),
        )?;

        Ok(StructTy {
            ty_params: Vec::new(),
            fields: {
                let mut fields = self.fields;
                for (param, arg) in self.ty_params.iter().zip(params) {
                    for field in &mut fields {
                        field.ty = field.ty.clone().subst(param, arg.clone());
                    }
                }
                fields
            },
            ..self
        })
    }
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
    IntNeg,
    IntAdd,
    IntSub,
    IntMul,
    IntDiv,
    IntPow,
    IntMod,
    IntBitOr,
    IntBitAnd,
    IntBitNot,
    IntBitXor,
    IntShl,
    IntShr,
    IntEq,
    IntLt,
    IntLe,
    IntGt,
    IntGe,
    FloatPos,
    FloatNeg,
    FloatAdd,
    FloatSub,
    FloatMul,
    FloatDiv,
    FloatPow,
    FloatMod,
    FloatEq,
    FloatLt,
    FloatLe,
    FloatGt,
    FloatGe,
    BoolEq,
    BoolNot,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Op {
    Pos,
    Neg,
    Add,
    AddAssign,
    Sub,
    SubAssign,
    Mul,
    MulAssign,
    Div,
    DivAssign,
    Mod,
    ModAssign,
    Pow,
    PowAssign,
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
    Not,
    BitOr,
    BitOrAssign,
    BitAnd,
    BitAndAssign,
    BitXor,
    BitXorAssign,
    BitNot,
    Shl,
    ShlAssign,
    Shr,
    ShrAssign,
    Index,
    IndexMut,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum StaticOp {
    New,
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
    CallOp(Op, Box<Self>, Vec<Self>),
    CallStaticOp(StaticOp, Ty, Vec<Self>),
    Cast(Box<Self>, Ty),
    GetAttr(Box<Self>, Ident),
    SetAttr(Box<Self>, Ident, Box<Self>),
    Block(ScopeId),
    If(Box<Self>, ScopeId, ScopeId),
    While(Box<Self>, ScopeId, ScopeId),
    Loop(ScopeId),
    Assign(Pattern, Box<Self>),
}
