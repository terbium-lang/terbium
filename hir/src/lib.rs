//! High-level intermediate representation. This IR is used for type analysis, validating code
//! correctness, and desugaring.
//!
//! # Strategy
//! A total of three "sub"-lowerings are performed:
//! * A first lowering from the AST to the HIR, which desugars the AST and performs some basic
//!   validation and module/path resolution. This lowering is performed by the [`AstLowerer`].
//! * The second lowering is from the initially lowered HIR to a typed HIR. During this lowering,
//!   fundamental type inference and type checking is performed. This lowering is performed by
//!  [`TypeLowerer`].
//! * The third and final lowering is a simple pass over the typed HIR, which performs any remaining
//!   type checking and desugaring with the knowledge of the types of all expressions. This lowering
//!   is performed by [`TypeChecker`].

#![feature(more_qualified_paths)]
#![feature(map_try_insert)]
#![feature(try_blocks)]

pub mod check;
pub mod error;
pub mod infer;
pub mod lower;
pub mod typed;
pub mod warning;

use common::span::{Span, Spanned, Src};
pub use error::Error;
use grammar::ast::{self, Indent};
pub use grammar::ast::{ItemVisibility, MemberVisibility};
pub use infer::pat_errors;
use internment::Intern;
use std::{
    collections::HashMap,
    fmt::{self, Debug, Display, Formatter},
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Ident(Intern<String>);

impl Display for Ident {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for Ident {
    fn from(s: String) -> Self {
        Self(Intern::new(s))
    }
}

impl From<&str> for Ident {
    fn from(s: &str) -> Self {
        Self(Intern::from_ref(s))
    }
}

/// The ID of a module.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModuleId(Intern<Vec<String>>);

impl Display for ModuleId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            f.write_str("<root>")
        } else {
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

/// Global item lookup ID.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LookupId(pub usize);

/// The ID of a top-level or order-agnostic item.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ItemId(
    /// The module in which the item is defined.
    pub ModuleId,
    /// The name of the item, which is unique within the module.
    pub Ident,
);

impl Display for ItemId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.0 == ModuleId::root() {
            write!(f, "{}", self.1)
        } else {
            write!(f, "{}.{}", self.0, self.1)
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ScopeId(pub usize);

impl Display for ScopeId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl ScopeId {
    pub const ROOT: Self = Self(0);

    pub const fn next(&self) -> Self {
        Self(self.0 + 1)
    }
}

pub trait Metadata: Clone + Debug {
    type Expr: Clone + Debug;
    type Ty: Clone + Debug;
}

#[derive(Clone, Debug)]
pub struct LowerMetadata;
impl Metadata for LowerMetadata {
    type Expr = Expr;
    type Ty = Ty;
}

/// HIR of a Terbium program.
#[derive(Clone, Debug)]
pub struct Hir<M: Metadata = LowerMetadata> {
    /// A mapping of all modules within the program.
    pub modules: HashMap<ModuleId, ScopeId>,
    /// A mapping of all lexical scopes within the program.
    pub scopes: HashMap<ScopeId, Scope<M>>,
    /// A mapping of all functions in the program.
    pub funcs: HashMap<LookupId, Func<M>>,
    /// A mapping of all function lookup IDs to their defining item IDs.
    pub func_ids: HashMap<LookupId, ItemId>,
    /// A mapping of all aliases in the program.
    pub aliases: HashMap<LookupId, Alias<M::Expr>>,
    /// A mapping of all constants in the program.
    pub consts: HashMap<LookupId, Const<M>>,
    /// A mapping of all raw structs within the program.
    pub structs: HashMap<LookupId, StructTy<M>>,
    /// A mapping of all types within the program.
    pub types: HashMap<LookupId, TyDef<M::Ty>>,
}

impl<M: Metadata> Default for Hir<M> {
    fn default() -> Self {
        Self {
            modules: HashMap::new(),
            scopes: HashMap::new(),
            funcs: HashMap::new(),
            func_ids: HashMap::new(),
            aliases: HashMap::new(),
            consts: HashMap::new(),
            structs: HashMap::new(),
            types: HashMap::new(),
        }
    }
}

#[inline]
fn join_by_comma<'a, T: ToString + 'a>(items: impl Iterator<Item = &'a T> + 'a) -> String {
    items
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

impl<M: Metadata> Display for Hir<M>
where
    for<'a> WithHir<'a, M::Expr, M>: Display,
    for<'a> WithHir<'a, Spanned<Node<M>>, M>: Display,
    for<'a> WithHir<'a, Func<M>, M>: Display,
    for<'a> WithHir<'a, Const<M>, M>: Display,
    for<'a> WithHir<'a, StructTy<M>, M>: Display,
    for<'a> WithHir<'a, Alias<M::Expr>, M>: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for (module, scope) in &self.modules {
            self.write_scope(f, *scope, |f| write!(f, "module {module} "))?;
        }
        Ok(())
    }
}

/// A decorator that may be applied to a scope.
#[derive(Clone, Debug)]
pub enum Decorator {
    /// The `inline` function decorator, which indicates that the function should preferrably be
    /// inlined. Equates to the `inlinehint` LLVM attribute.
    Inline,
    /// The `always_inline` function decorator, which indicates that the function should always be
    /// inlined. Equates to the `alwaysinline` LLVM attribute.
    AlwaysInline,
    /// The `never_inline` function decorator, which indicates the the function should never be
    /// inlined. Equates to the `noinline` LLVM attribute.
    NeverInline,
    /// The `rarely_called` function decorator, which indicates that the function is rarely
    /// called and equates to the `cold` LLVM attribute.
    RarelyCalled,
    /// The `frequently_called` function decorator, which indicates that the function is frequently
    /// called and equates to the `hot` LLVM attribute.
    FrequentlyCalled,
    /// The `suppress` scope decorator, which indicates that all warnings within the scope in
    /// the specified category should be suppressed.
    Suppress(Ident),
    /// The `forbid` scope decorator, which indicates that all warnings within the scope in
    /// the specified category should be turned into hard errors.
    Forbid(Ident),
}

impl Display for Decorator {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inline => write!(f, "inline"),
            Self::AlwaysInline => write!(f, "always_inline"),
            Self::NeverInline => write!(f, "never_inline"),
            Self::RarelyCalled => write!(f, "rarely_called"),
            Self::FrequentlyCalled => write!(f, "frequently_called"),
            Self::Suppress(cat) => write!(f, "suppress({cat})"),
            Self::Forbid(cat) => write!(f, "forbid({cat})"),
        }
    }
}

/// An HIR node.
#[derive(Clone, Debug)]
pub enum Node<M: Metadata = LowerMetadata> {
    Expr(Spanned<M::Expr>),
    Let {
        pat: Spanned<Pattern>,
        ty: M::Ty,
        ty_span: Option<Span>,
        value: Option<Spanned<M::Expr>>,
    },
    Break(Option<Spanned<Ident>>, Option<Spanned<M::Expr>>),
    Continue(Option<Spanned<Ident>>),
    Return(Option<Spanned<M::Expr>>),
    ImplicitReturn(Spanned<M::Expr>),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ItemKind {
    Func,
    Alias,
    Const,
    Struct,
    Type,
}

#[derive(Clone, Debug)]
pub struct Scope<M: Metadata = LowerMetadata> {
    /// The module in which this scope is defined.
    pub module_id: ModuleId,
    /// Decorators that may modify the behavior of this scope.
    pub decorators: Vec<Spanned<Decorator>>,
    /// The label of this scope.
    pub label: Option<Spanned<Ident>>,
    /// The children of this scope.
    pub children: Spanned<Vec<Spanned<Node<M>>>>,
    /// A lookup of all items in the scope.
    pub items: HashMap<(ItemKind, ItemId), LookupId>,
}

impl<M: Metadata> Scope<M> {
    // caller should verify `children` is set before accessing it.
    pub(crate) fn from_module_id(module_id: ModuleId) -> Self {
        Self::new(module_id, None, Spanned(Vec::new(), Span::PLACEHOLDER))
    }

    pub(crate) fn new(
        module_id: ModuleId,
        label: Option<Spanned<Ident>>,
        children: Spanned<Vec<Spanned<Node<M>>>>,
    ) -> Self {
        Self {
            module_id,
            decorators: Vec::new(),
            label,
            children,
            items: HashMap::new(),
        }
    }

    pub(crate) fn get_lookup(&self, kind: ItemKind, id: ItemId) -> Option<LookupId> {
        self.items.get(&(kind, id)).copied()
    }

    pub(crate) fn get_lookup_or_panic(&self, kind: ItemKind, id: ItemId) -> LookupId {
        self.get_lookup(kind, id)
            .expect(&format!("item {id} not found in scope"))
    }
}

#[derive(Clone, Debug)]
pub struct Const<M: Metadata = LowerMetadata> {
    pub vis: ItemVisibility,
    pub name: Spanned<Ident>,
    pub ty: M::Ty,
    pub value: Spanned<M::Expr>,
}

#[derive(Clone, Debug)]
pub struct Alias<Expr> {
    pub vis: ItemVisibility,
    pub name: Spanned<Ident>,
    pub value: Spanned<Expr>,
}

#[inline]
fn ty_params_len<Ty>(params: &[TyParam<Ty>]) -> usize {
    params.iter().position(|p| p.infer).unwrap_or(params.len())
}

#[inline]
fn assert_equal_params_length(
    span: Span,
    ty_name: Spanned<Ident>,
    ty_params_len: usize,
    ty_args_len: usize,
) -> Result<(), Error> {
    if ty_args_len != ty_params_len {
        return Err(Error::IncorrectTypeArgumentCount {
            span,
            ty: ty_name.as_ref().map(ToString::to_string),
            expected: ty_params_len,
            actual: ty_args_len,
        });
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct TyDef<T = Ty> {
    pub name: Spanned<Ident>,
    pub ty: T,
    pub ty_params: Vec<TyParam<T>>,
}

impl<Ty: Clone + SubstituteTyParams<Ty>> TyDef<Ty> {
    pub fn apply_params(&self, span: Span, params: Vec<Ty>) -> Result<Ty, Error> {
        assert_equal_params_length(
            span,
            self.name,
            ty_params_len(&self.ty_params),
            params.len(),
        )?;

        let mut ty = self.ty.clone();
        for (param, arg) in self.ty_params.iter().zip(params) {
            ty = ty.substitute(param, arg);
        }
        Ok(ty)
    }
}

/// A pattern that can be matched against.
#[derive(Clone, Debug)]
pub enum Pattern {
    Ident {
        ident: Spanned<Ident>,
        mut_kw: Option<Span>,
    },
    Tuple(Vec<Spanned<Self>>),
}

impl Display for Pattern {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ident { ident, mut_kw } => {
                if mut_kw.is_some() {
                    f.write_str("mut ")?;
                }
                write!(f, "{ident}")
            }
            Self::Tuple(items) => write!(f, "({})", join_by_comma(items.iter())),
        }
    }
}

impl Pattern {
    pub fn ident(&self) -> Option<Ident> {
        match self {
            Self::Ident { ident, .. } => Some(ident.0),
            _ => None,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FieldVisibility {
    pub get: MemberVisibility,
    pub set: MemberVisibility,
}

impl Display for FieldVisibility {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("public(")?;
        if let Some(vis) = self.get.inner_visibility() {
            write!(f, "{} ", vis)?;
        }
        f.write_str("get, ")?;

        if let Some(vis) = self.set.inner_visibility() {
            write!(f, "{} ", vis)?;
        }
        f.write_str("set)")
    }
}

impl FieldVisibility {
    pub fn from_ast(v: Spanned<ast::FieldVisibility>) -> error::Result<Self> {
        if v.0.get.0 < v.0.set.0 {
            Err(Error::GetterLessVisibleThanSetter(v))
        } else {
            Ok(Self {
                get: v.0.get.0,
                set: v.0.set.0,
            })
        }
    }
}

#[derive(Clone, Debug)]
pub struct FuncParam<M: Metadata = LowerMetadata> {
    pub pat: Spanned<Pattern>,
    pub ty: M::Ty,
    pub default: Option<Spanned<M::Expr>>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FuncKind {
    Normal,
    External(Span),
    Internal(Span),
}

#[derive(Clone, Debug)]
pub struct FuncHeader<M: Metadata = LowerMetadata> {
    /// The name of the function.
    pub name: Spanned<Ident>,
    /// The type parameters of the function.
    pub ty_params: Vec<TyParam<M::Ty>>,
    /// The parameters of the function.
    pub params: Vec<FuncParam<M>>,
    /// The keyword-only parameters of the function.
    pub kw_params: Vec<FuncParam<M>>,
    /// The return type of the function.
    pub ret_ty: M::Ty,
    /// The span of the return type. `None` if the return type was not specified.
    pub ret_ty_span: Option<Span>,
}

/// HIR of a top-level function.
#[derive(Clone, Debug)]
pub struct Func<M: Metadata = LowerMetadata> {
    /// The visibility of the item.
    pub vis: ItemVisibility,
    /// The kind of function declaration.
    pub kind: FuncKind,
    /// The header of the function.
    pub header: FuncHeader<M>,
    /// The body of the function.
    pub body: Option<ScopeId>,
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

impl IntWidth {
    pub fn naturalize(&mut self) {
        if *self == Self::Unknown {
            *self = Self::default();
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum IntSign {
    Signed,
    Unsigned,
}

impl IntSign {
    pub const fn is_signed(self) -> bool {
        matches!(self, Self::Signed)
    }

    pub const fn type_name(&self) -> &str {
        match self {
            Self::Signed => "int",
            Self::Unsigned => "uint",
        }
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub enum FloatWidth {
    Float32 = 32,
    #[default]
    Float64 = 64,
    Unknown = !0,
    // Float128 = 128,
}

impl FloatWidth {
    pub fn naturalize(&mut self) {
        if *self == Self::Unknown {
            *self = Self::default();
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PrimitiveTy {
    // integer "signedness" and bit-width are unified as one type since they coerce to each other
    Int(IntSign, IntWidth),
    Float(FloatWidth),
    Bool,
    Char,
    Void,
    String,
}

impl Display for PrimitiveTy {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(sign, width) => {
                if *sign == IntSign::Unsigned {
                    f.write_str("u")?;
                }
                f.write_str("int")?;
                if *width != IntWidth::Unknown {
                    write!(f, "{}", *width as isize)?;
                }
                Ok(())
            }
            Self::Float(width) => {
                f.write_str("float")?;
                if *width != FloatWidth::Unknown {
                    write!(f, "{}", *width as isize)?;
                }
                Ok(())
            }
            Self::Bool => f.write_str("bool"),
            Self::Char => f.write_str("char"),
            Self::Void => f.write_str("void"),
            Self::String => f.write_str("string"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Ty {
    Unknown,
    Primitive(PrimitiveTy),
    Generic(Ident),
    Tuple(Vec<Self>),
    Array(Box<Self>, Option<usize>),
    Struct(ItemId, Vec<Self>),
    Func(Vec<Self>, Box<Self>),
}

pub trait SubstituteTyParams<Ty> {
    fn substitute(self, param: &TyParam<Ty>, ty: Ty) -> Self;
}

impl SubstituteTyParams<Self> for Ty {
    fn substitute(self, param: &TyParam, ty: Ty) -> Self {
        if param.infer {
            return Self::Unknown;
        }
        match self {
            Self::Generic(p) if p == param.name => ty,
            Self::Tuple(tys) => Self::Tuple(
                tys.into_iter()
                    .map(|t| t.substitute(param, ty.clone()))
                    .collect(),
            ),
            Self::Struct(item, tys) => Self::Struct(
                item,
                tys.into_iter()
                    .map(|t| t.substitute(param, ty.clone()))
                    .collect(),
            ),
            other => other,
        }
    }
}

impl Ty {
    fn iter_unknown_types<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut Self> + 'a> {
        match self {
            Self::Unknown => Box::new(std::iter::once(self)),
            Self::Tuple(tys) => Box::new(tys.iter_mut().flat_map(|ty| ty.iter_unknown_types())),
            Self::Array(ty, _) => Box::new(ty.iter_unknown_types()),
            Self::Struct(_, args) => {
                Box::new(args.iter_mut().flat_map(|ty| ty.iter_unknown_types()))
            }
            _ => Box::new(std::iter::empty()),
        }
    }
}

impl Display for Ty {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown => f.write_str("<unknown>"),
            Self::Primitive(p) => write!(f, "{p}"),
            Self::Generic(i) => write!(f, "{i}"),
            Self::Tuple(tys) => {
                write!(f, "({})", join_by_comma(tys.iter()))
            }
            Self::Array(ty, len) => {
                write!(f, "[{ty}")?;
                if let Some(len) = len {
                    write!(f, "; {len}")?;
                }
                write!(f, "]")
            }
            Self::Struct(sid, args) => {
                write!(f, "{sid}")?;
                if !args.is_empty() {
                    write!(f, "<{}>", join_by_comma(args.iter()))?;
                }
                Ok(())
            }
            Self::Func(params, ret) => {
                write!(f, "({})", join_by_comma(params.iter()))?;
                write!(f, " -> {ret}")
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct TyParam<T = Ty> {
    pub name: Ident,
    pub bound: Option<Box<T>>,
    /// Indicates you cannot explicitly specify this type
    pub infer: bool,
}

impl<Ty: Display> Display for TyParam<Ty> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.infer {
            f.write_str("infer ")?;
        }
        write!(f, "{}", self.name)?;
        if let Some(bound) = self.bound.as_ref() {
            write!(f, ": {bound}")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct StructField<M: Metadata = LowerMetadata> {
    pub vis: FieldVisibility,
    pub name: Ident,
    pub ty: M::Ty,
    pub default: Option<Spanned<M::Expr>>,
}

#[derive(Clone, Debug)]
pub struct StructTy<M: Metadata = LowerMetadata> {
    pub vis: ItemVisibility,
    pub name: Spanned<Ident>,
    pub ty_params: Vec<TyParam<M::Ty>>,
    pub fields: Vec<StructField<M>>,
}

impl<'a, M: Metadata> Display for WithHir<'a, StructField<M>, M>
where
    WithHir<'a, M::Expr, M>: Display,
    M::Ty: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}: {}", self.0.vis, self.0.name, self.0.ty)?;
        if let Some(default) = &self.0.default {
            write!(f, "= {}", self.with(default))?;
        }
        Ok(())
    }
}

impl<'a, M: Metadata> Display for WithHir<'a, StructTy<M>, M>
where
    WithHir<'a, M::Expr, M>: Display,
    WithHir<'a, StructField<M>, M>: Display,
    M::Ty: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} struct {}", self.0.vis, self.0.name)?;
        if !self.0.ty_params.is_empty() {
            write!(
                f,
                "<{}>",
                self.0
                    .ty_params
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            )?;
        }

        write!(f, " {{\n")?;
        for field in &self.0.fields {
            format!("{},", self.with(field)).write_indent(f)?;
        }
        write!(f, "}}")
    }
}

impl StructTy {
    pub fn into_adhoc_struct_ty_with_applied_ty_params(
        self,
        span: Option<Span>,
        params: Vec<Ty>,
    ) -> Result<Self, Error> {
        assert_equal_params_length(
            span.unwrap_or(self.name.span()),
            self.name,
            ty_params_len(&self.ty_params),
            params.len(),
        )?;

        Ok(StructTy {
            ty_params: Vec::new(),
            fields: {
                let mut fields = self.fields;
                for (param, arg) in self.ty_params.iter().zip(params) {
                    for field in &mut fields {
                        field.ty = field.ty.clone().substitute(param, arg.clone());
                    }
                }
                fields
            },
            ..self
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Literal {
    UInt(u128),
    Int(i128),
    Float(u64 /* raw bits */),
    Bool(bool),
    Char(char),
    String(String),
    Bytes(Vec<u8>), // Bytestring literal, e.g. b"..."
    Void,
}

impl Display for Literal {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::UInt(u) => write!(f, "{u}u"),
            Self::Int(i) => write!(f, "{i}"),
            Self::Float(n) => write!(f, "{:?}", f64::from_bits(*n)),
            Self::Bool(b) => write!(f, "{b}"),
            Self::Char(c) => write!(f, "c{c:?}"),
            Self::String(s) => write!(f, "{s:?}"),
            Self::Bytes(b) => write!(f, "b{:?}", String::from_utf8_lossy(b)),
            Self::Void => f.write_str("void"),
        }
    }
}

// TODO: stubs of intrinsic functions should be defined in the core library instead of manually
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Intrinsic {}

impl Intrinsic {
    pub const fn name(&self) -> &'static str {
        match self {
            _ => todo!(),
        }
    }

    pub fn qualified_name(&self) -> String {
        "<intrinsics>.".to_string() + self.name()
    }
}

impl Display for Intrinsic {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.qualified_name())
    }
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
    Ne,
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

impl Display for Op {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Pos => "pos",
            Self::Neg => "neg",
            Self::Add => "add",
            Self::AddAssign => "add_assign",
            Self::Sub => "sub",
            Self::SubAssign => "sub_assign",
            Self::Mul => "mul",
            Self::MulAssign => "mul_assign",
            Self::Div => "div",
            Self::DivAssign => "div_assign",
            Self::Mod => "mod",
            Self::ModAssign => "mod_assign",
            Self::Pow => "pow",
            Self::PowAssign => "pow_assign",
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Lt => "lt",
            Self::Le => "le",
            Self::Gt => "gt",
            Self::Ge => "ge",
            Self::Not => "not",
            Self::BitOr => "bit_or",
            Self::BitOrAssign => "bit_or_assign",
            Self::BitAnd => "bit_and",
            Self::BitAndAssign => "bit_and_assign",
            Self::BitXor => "bit_xor",
            Self::BitXorAssign => "bit_xor_assign",
            Self::BitNot => "bit_not",
            Self::Shl => "shl",
            Self::ShlAssign => "shl_assign",
            Self::Shr => "shr",
            Self::ShrAssign => "shr_assign",
            Self::Index => "index",
            Self::IndexMut => "index_mut",
        })
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum StaticOp {
    New,
}

impl Display for StaticOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::New => "new",
        })
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum LogicalOp {
    And,
    Or,
}

impl Display for LogicalOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::And => "&&",
            Self::Or => "||",
        })
    }
}

#[derive(Clone, Debug)]
pub enum Expr {
    Literal(Literal),
    Ident(Spanned<Ident>, Option<Spanned<Vec<Ty>>>),
    Tuple(Vec<Spanned<Self>>),
    Array(Vec<Spanned<Self>>),
    Intrinsic(Intrinsic, Vec<Spanned<Self>>),
    Call {
        callee: Box<Spanned<Self>>,
        args: Vec<Spanned<Self>>,
        kwargs: Vec<(Ident, Spanned<Self>)>,
    },
    CallOp(Spanned<Op>, Box<Spanned<Self>>, Vec<Spanned<Self>>),
    CallStaticOp(StaticOp, Ty, Vec<Spanned<Self>>),
    CallLogicalOp(Spanned<LogicalOp>, Box<Spanned<Self>>, Box<Spanned<Self>>),
    Cast(Box<Spanned<Self>>, Ty),
    GetAttr(Box<Spanned<Self>>, Spanned<Ident>),
    SetAttr(Box<Spanned<Self>>, Spanned<Ident>, Box<Spanned<Self>>),
    Block(ScopeId),
    If(Box<Spanned<Self>>, ScopeId, Option<ScopeId>),
    Loop(ScopeId),
    Assign(Spanned<Pattern>, Box<Spanned<Self>>),
    AssignPtr(Box<Spanned<Self>>, Box<Spanned<Self>>),
}

pub struct WithHir<'a, T, M: Metadata = LowerMetadata>(&'a T, &'a Hir<M>);

impl<'a, T, M: Metadata> WithHir<'a, T, M> {
    pub fn with<U>(&self, new: &'a U) -> WithHir<'a, U, M> {
        WithHir(new, self.1)
    }
}

impl<'a, M: 'a + Metadata> Hir<M>
where
    WithHir<'a, M::Expr, M>: Display,
    WithHir<'a, Spanned<Node<M>>, M>: Display,
    WithHir<'a, Func<M>, M>: Display,
    WithHir<'a, Const<M>, M>: Display,
    WithHir<'a, StructTy<M>, M>: Display,
    WithHir<'a, Alias<M::Expr>, M>: Display,
{
    #[inline]
    pub fn get_scope(&self, sid: ScopeId) -> &Scope<M> {
        self.scopes
            .get(&sid)
            .expect(&format!("invalid scope id {sid}"))
    }

    pub fn write_scope(
        &'a self,
        f: &mut Formatter,
        sid: ScopeId,
        header: impl FnOnce(&mut Formatter) -> fmt::Result,
    ) -> fmt::Result {
        let scope = self.get_scope(sid);
        if let Some(label) = scope.label {
            write!(f, ":{label} ")?;
        }
        header(f)?;
        write!(f, "[{sid}] ")?;
        self.write_block(f, &scope)
    }

    pub fn write_block(&'a self, f: &mut Formatter, scope: &'a Scope<M>) -> fmt::Result {
        f.write_str("{\n")?;
        for decorator in &scope.decorators {
            format!("@!{decorator}").write_indent(f)?;
        }

        for ((kind, item_id), lookup) in &scope.items {
            writeln!(f, "{item_id}:")?;
            match kind {
                ItemKind::Func => WithHir(&self.funcs[&lookup], self).write_indent(f)?,
                ItemKind::Alias => WithHir(&self.aliases[&lookup], self).write_indent(f)?,
                ItemKind::Const => WithHir(&self.consts[&lookup], self).write_indent(f)?,
                ItemKind::Struct => WithHir(&self.structs[&lookup], self).write_indent(f)?,
                ItemKind::Type => continue,
            }
        }

        for line in scope.children.value() {
            WithHir(line, self).write_indent(f)?;
        }
        f.write_str("}")
    }
}

impl<'a, T, M: Metadata> Display for WithHir<'a, Spanned<T>, M>
where
    WithHir<'a, T, M>: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.with(self.0.value()).fmt(f)
    }
}

impl Display for WithHir<'_, Expr> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let comma_sep = |args: &[Spanned<Expr>]| -> String {
            args.iter()
                .map(|e| self.with(e.value()).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };

        match self.0 {
            Expr::Literal(l) => write!(f, "{l}"),
            Expr::Ident(i, args) => {
                write!(f, "{i}")?;
                if let Some(args) = args {
                    write!(
                        f,
                        "<{}>",
                        args.value()
                            .iter()
                            .map(|ty| ty.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )?;
                }
                Ok(())
            }
            Expr::Tuple(exprs) => write!(f, "({})", comma_sep(exprs)),
            Expr::Array(exprs) => write!(f, "[{}]", comma_sep(exprs)),
            Expr::Intrinsic(intr, args) => {
                write!(f, "{intr}({})", comma_sep(args))
            }
            Expr::Call {
                callee,
                args,
                kwargs,
            } => {
                let args = args
                    .iter()
                    .map(|e| self.with(e).to_string())
                    .chain(
                        kwargs
                            .iter()
                            .map(|(name, e)| format!("{name}: {}", self.with(e))),
                    )
                    .collect::<Vec<_>>()
                    .join(", ");

                write!(f, "{}({args})", self.with(&**callee))
            }
            Expr::CallOp(op, slf, args) => {
                write!(f, "{}.<{op}>({})", self.with(&**slf), comma_sep(args))
            }
            Expr::CallStaticOp(op, ty, args) => {
                write!(f, "({ty}).{op}({})", comma_sep(args))
            }
            Expr::CallLogicalOp(op, lhs, rhs) => {
                write!(f, "({} {op} {})", self.with(&**lhs), self.with(&**rhs))
            }
            Expr::Cast(expr, ty) => {
                write!(f, "({} to {ty})", self.with(&**expr))
            }
            Expr::GetAttr(expr, name) => {
                write!(f, "({}.{name})", self.with(&**expr))
            }
            Expr::SetAttr(expr, name, value) => {
                write!(
                    f,
                    "({}.{name} = {})",
                    self.with(&**expr),
                    self.with(&**value)
                )
            }
            Expr::Block(sid) => self.1.write_scope(f, *sid, |_| Ok(())),
            Expr::If(cond, then, els) => {
                let then = self.1.get_scope(*then);
                if let Some(label) = then.label {
                    write!(f, ":{label} ")?;
                }
                write!(f, "if {} ", self.with(&**cond))?;
                self.1.write_block(f, then)?;

                if let Some(els) = els {
                    let els = self.1.get_scope(*els);
                    write!(f, " else ")?;
                    self.1.write_block(f, els)?;
                }
                Ok(())
            }
            Expr::Loop(sid) => self.1.write_scope(f, *sid, |f| write!(f, "loop ")),
            Expr::Assign(pat, value) => {
                write!(f, "{pat} = {}", self.with(&**value))
            }
            Expr::AssignPtr(ptr, value) => {
                write!(f, "&{} = {}", self.with(&**ptr), self.with(&**value))
            }
        }
    }
}

impl<'a, M: Metadata> Display for WithHir<'a, Node<M>, M>
where
    WithHir<'a, M::Expr, M>: Display,
    M::Ty: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.0 {
            Node::Expr(e) => write!(f, "{};", self.with(e)),
            Node::Let { pat, ty, value, .. } => {
                write!(f, "let {pat}: {ty}")?;
                if let Some(value) = value {
                    write!(f, " = {};", self.with(value))?;
                }
                Ok(())
            }
            Node::Break(label, value) => {
                write!(f, "break")?;
                if let Some(label) = label {
                    write!(f, " :{label}")?;
                }
                if let Some(value) = value {
                    write!(f, " {}", self.with(value))?;
                }
                f.write_str(";")
            }
            Node::Continue(label) => {
                write!(f, "continue")?;
                if let Some(label) = label {
                    write!(f, " :{label}")?;
                }
                f.write_str(";")
            }
            Node::Return(value) => {
                write!(f, "return")?;
                if let Some(value) = value {
                    write!(f, " {}", self.with(value))?;
                }
                f.write_str(";")
            }
            Node::ImplicitReturn(e) => {
                write!(f, "{}", self.with(e))
            }
        }
    }
}

impl<'a, M: Metadata> Display for WithHir<'a, Const<M>, M>
where
    WithHir<'a, M::Expr, M>: Display,
    M::Ty: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} const {}: {} = {};",
            self.0.vis,
            self.0.name,
            self.0.ty,
            self.with(&self.0.value)
        )
    }
}

impl<'a, M: Metadata> Display for WithHir<'a, Alias<M::Expr>, M>
where
    WithHir<'a, M::Expr, M>: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} alias {} = {};",
            self.0.vis,
            self.0.name,
            self.with(&self.0.value)
        )
    }
}

impl<'a, M: Metadata> Display for WithHir<'a, FuncParam<M>, M>
where
    WithHir<'a, M::Expr, M>: Display,
    M::Ty: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.0.pat, self.0.ty)?;
        if let Some(default) = &self.0.default {
            write!(f, " = {}", self.with(default))?;
        }
        Ok(())
    }
}

impl<'a, M: Metadata> Display for WithHir<'a, FuncHeader<M>, M>
where
    WithHir<'a, M::Expr, M>: Display,
    M::Ty: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let ty_params = if self.0.ty_params.is_empty() {
            String::new()
        } else {
            format!(
                "<{}>",
                self.0
                    .ty_params
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        write!(
            f,
            "{}{}({}) -> {}",
            self.0.name,
            ty_params,
            {
                let mut params = self
                    .0
                    .params
                    .iter()
                    .map(|p| self.with(p).to_string())
                    .collect::<Vec<_>>();
                if !self.0.kw_params.is_empty() {
                    params.push("*".to_string());
                    params.extend(self.0.kw_params.iter().map(|p| self.with(p).to_string()));
                }
                params.join(", ")
            },
            self.0.ret_ty
        )
    }
}

impl<'a, M: Metadata> Display for WithHir<'a, Func<M>, M>
where
    WithHir<'a, M::Expr, M>: Display,
    WithHir<'a, Spanned<Node<M>>, M>: Display,
    WithHir<'a, Const<M>, M>: Display,
    WithHir<'a, StructTy<M>, M>: Display,
    WithHir<'a, Alias<M::Expr>, M>: Display,
    M::Ty: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let kind = match self.0.kind {
            FuncKind::Normal => "",
            FuncKind::External(_) => "external ",
            FuncKind::Internal(_) => "internal ",
        };
        write!(
            f,
            "{} {}func {} ",
            self.0.vis,
            kind,
            self.with(&self.0.header)
        )?;
        match self.0.body {
            Some(body) => self.1.write_block(f, self.1.get_scope(body)),
            None => f.write_str(";"),
        }
    }
}
