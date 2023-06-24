use crate::{
    infer::{ExitAction, InferMetadata},
    Ident, IntSign, IntWidth, Intrinsic, ItemId, Literal, Node, Op, Pattern, PrimitiveTy, ScopeId,
    StaticOp, WithHir,
};
use common::span::Spanned;
use std::{
    cmp::Ordering,
    collections::VecDeque,
    fmt::{self, Display, Formatter},
};

/// The kind of the local environment.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum LocalEnv {
    /// The local is a standard lowered local.
    #[default]
    Standard,
    /// The local is lowered through a user-defined macro.
    Macro, // TODO: this would take a macro ID
    /// The local is lowered internally.
    Internal,
}

/// Unary intrinsic.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum UnaryIntIntrinsic {
    Neg,
    BitNot,
}

impl Display for UnaryIntIntrinsic {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            UnaryIntIntrinsic::Neg => write!(f, "-"),
            UnaryIntIntrinsic::BitNot => write!(f, "~"),
        }
    }
}

/// Binary intrinsic.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BinaryIntIntrinsic {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Display for BinaryIntIntrinsic {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            BinaryIntIntrinsic::Add => write!(f, "+"),
            BinaryIntIntrinsic::Sub => write!(f, "-"),
            BinaryIntIntrinsic::Mul => write!(f, "*"),
            BinaryIntIntrinsic::Div => write!(f, "/"),
            BinaryIntIntrinsic::Mod => write!(f, "%"),
            BinaryIntIntrinsic::BitAnd => write!(f, "&"),
            BinaryIntIntrinsic::BitOr => write!(f, "|"),
            BinaryIntIntrinsic::BitXor => write!(f, "^"),
            BinaryIntIntrinsic::Shl => write!(f, "<<"),
            BinaryIntIntrinsic::Shr => write!(f, ">>"),
            BinaryIntIntrinsic::Eq => write!(f, "=="),
            BinaryIntIntrinsic::Lt => write!(f, "<"),
            BinaryIntIntrinsic::Le => write!(f, "<="),
            BinaryIntIntrinsic::Gt => write!(f, ">"),
            BinaryIntIntrinsic::Ge => write!(f, ">="),
        }
    }
}

/// Integer intrinsic.
#[derive(Clone, Debug)]
pub enum IntIntrinsic {
    Unary(UnaryIntIntrinsic, Box<E>),
    Binary(BinaryIntIntrinsic, Box<E>, Box<E>),
}

/// Boolean intrinsic.
#[derive(Clone, Debug)]
pub enum BoolIntrinsic {
    Not(Box<E>),
    And(Box<E>, Box<E>),
    Or(Box<E>, Box<E>),
    Xor(Box<E>, Box<E>),
}

// TODO: This enum and its display impl are almost duplicates of crate::Expr, maybe find a way to unify them?
#[derive(Clone, Debug)]
pub enum Expr {
    Literal(Literal),
    Local(Spanned<Ident>, Option<Spanned<Vec<Ty>>>, LocalEnv),
    Type(Spanned<Ty>),
    Tuple(Vec<E>),
    Array(Vec<E>),
    IntIntrinsic(IntIntrinsic, IntSign, IntWidth),
    BoolIntrinsic(BoolIntrinsic),
    Intrinsic(Intrinsic, Vec<E>),
    Call {
        callee: Box<E>,
        args: Vec<E>,
        kwargs: Vec<(Ident, E)>,
    },
    CallOp(Op, Box<E>, Vec<E>),
    CallStaticOp(StaticOp, Ty, Vec<E>),
    Cast(Box<E>, Ty),
    GetAttr(Box<E>, Spanned<Ident>),
    SetAttr(Box<E>, Spanned<Ident>, Box<E>),
    Block(ScopeId),
    If(Box<E>, ScopeId, Option<ScopeId>),
    Loop(ScopeId),
    Assign(Spanned<Pattern>, Box<E>),
    AssignPtr(Box<E>, Box<E>),
}

impl Display for WithHir<'_, TypedExpr, InferMetadata> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let comma_sep = |args: &[Spanned<TypedExpr>]| -> String {
            args.iter()
                .map(|e| self.with(e.value()).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };

        write!(f, "({}) ", self.0 .1)?;
        match &self.0 .0 {
            Expr::Literal(l) => write!(f, "{l}"),
            Expr::Local(i, args, _) => {
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
            Expr::Type(ty) => write!(f, "{ty}"),
            Expr::Tuple(exprs) => write!(f, "({})", comma_sep(exprs)),
            Expr::Array(exprs) => write!(f, "[{}]", comma_sep(exprs)),
            Expr::IntIntrinsic(intr, ..) => match intr {
                IntIntrinsic::Unary(intr, e) => write!(f, "({intr}{})", self.with(&**e)),
                IntIntrinsic::Binary(intr, lhs, rhs) => {
                    write!(f, "({} {intr} {})", self.with(&**lhs), self.with(&**rhs))
                }
            },
            Expr::BoolIntrinsic(intr) => match intr {
                BoolIntrinsic::Not(e) => write!(f, "(!{})", self.with(&**e)),
                BoolIntrinsic::And(lhs, rhs) => {
                    write!(f, "({} && {})", self.with(&**lhs), self.with(&**rhs))
                }
                BoolIntrinsic::Or(lhs, rhs) => {
                    write!(f, "({} || {})", self.with(&**lhs), self.with(&**rhs))
                }
                BoolIntrinsic::Xor(lhs, rhs) => {
                    write!(f, "({} ^ {})", self.with(&**lhs), self.with(&**rhs))
                }
            },
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
                write!(f, "[{then}] ")?;
                let then = self.1.get_scope(*then);
                if let Some(label) = then.label {
                    write!(f, ":{label} ")?;
                }
                write!(f, "if {} ", self.with(&**cond))?;
                self.1.write_block(f, then)?;

                if let Some(els) = els {
                    write!(f, " [{els}]")?;
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

#[derive(Clone, Debug)]
pub struct TypedExpr(pub Expr, pub Ty);

type E = Spanned<TypedExpr>;
pub type TypedNode = Node<TypedExpr>;

/// A type unification constraint.
#[derive(Clone, Debug)]
pub struct Constraint(pub Ty, pub Ty);

#[derive(Clone, Debug, Default)]
pub struct UnificationTable {
    pub constraints: VecDeque<Constraint>,
    pub substitutions: VecDeque<Ty>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Relation {
    Eq,
    Sub,
    Super,
    Unrelated,
}

impl Relation {
    pub const fn compatible(&self) -> bool {
        matches!(self, Self::Eq | Self::Sub)
    }

    pub const fn merge_covariant(&self, other: Self) -> Self {
        match (self, other) {
            (Self::Eq, Self::Eq) => Self::Eq,
            (Self::Eq, Self::Sub) | (Self::Sub, Self::Eq) | (Self::Sub, Self::Sub) => Self::Sub,
            _ => Self::Unrelated,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TyConst {
    pub value: Literal,
    pub ty: Ty,
}

impl TyConst {
    // TODO: should there be a usize type in terbium?
    pub fn usize(value: usize) -> Self {
        #[cfg(target_pointer_width = "16")]
        const USIZE_WIDTH: IntWidth = IntWidth::Int16;
        #[cfg(target_pointer_width = "32")]
        const USIZE_WIDTH: IntWidth = IntWidth::Int32;
        #[cfg(target_pointer_width = "64")]
        const USIZE_WIDTH: IntWidth = IntWidth::Int64;

        Self {
            value: Literal::UInt(value as _),
            ty: Ty::Primitive(PrimitiveTy::Int(IntSign::Unsigned, USIZE_WIDTH)),
        }
    }

    pub fn to_usize(self) -> usize {
        match self.value {
            Literal::UInt(value) => value as usize,
            Literal::Int(value) => value as usize,
            _ => panic!("expected usize literal"),
        }
    }
}

impl Display for TyConst {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InvalidTypeCause {
    CyclicTypeReference,
    TypeConflict,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ty {
    Invalid(InvalidTypeCause),
    Unknown(usize),
    Exit(Box<ExitAction>),
    Primitive(PrimitiveTy),
    Generic(Ident),
    Tuple(Vec<Self>),
    Array(Box<Self>, Option<Box<Self>>),
    Struct(ItemId, Vec<Self>),
    Func(Vec<Self>, Box<Self>),
    Const(Box<TyConst>),
}

impl Display for Ty {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(_) => write!(f, "<invalid>"),
            Self::Unknown(id) => write!(f, "${id}"),
            Self::Exit(_) => write!(f, "never"),
            Self::Primitive(p) => write!(f, "{p}"),
            Self::Generic(id) => write!(f, "{id}"),
            Self::Tuple(tys) => {
                write!(f, "(")?;
                for (i, ty) in tys.iter().enumerate() {
                    if i != 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{ty}")?;
                }
                write!(f, ")")
            }
            Self::Array(ty, len) => {
                write!(f, "[{ty}")?;
                if let Some(len) = len {
                    write!(f, "; {len}")?;
                }
                write!(f, "]")
            }
            Self::Struct(id, tys) => {
                write!(f, "{id}<")?;
                for (i, ty) in tys.iter().enumerate() {
                    if i != 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{ty}")?;
                }
                write!(f, ">")
            }
            Self::Func(args, ret) => {
                write!(f, "(")?;
                for (i, ty) in args.iter().enumerate() {
                    if i != 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{ty}")?;
                }
                write!(f, ") -> {ret}")
            }
            Self::Const(c) => write!(f, "{c}"),
        }
    }
}

impl Ty {
    /// Example: `u32` is a subtype of `u64` because they coerce.
    pub fn relation_to(&self, other: &Self) -> Relation {
        match (self, other) {
            (Self::Unknown(i), Self::Unknown(j)) if i == j => Relation::Eq,
            (Self::Exit(_), _) | (_, Self::Exit(_)) => Relation::Eq,
            (Self::Primitive(p), Self::Primitive(q)) => prim_ty_relation(*p, *q),
            // tuples are covariant
            (Self::Tuple(tys), Self::Tuple(other_tys)) => {
                if tys.len() != other_tys.len() {
                    return Relation::Unrelated;
                }
                let mut relation = Relation::Eq;
                for (ty, other_ty) in tys.iter().zip(other_tys.iter()) {
                    relation = relation.merge_covariant(ty.relation_to(other_ty));
                    if !relation.compatible() {
                        return Relation::Unrelated;
                    }
                }
                relation
            }
            // arrays are covariant
            (Self::Array(ty, len), Self::Array(oty, olen)) => {
                if len != olen {
                    return Relation::Unrelated;
                }
                ty.relation_to(oty)
            }
            // structs define their own variance, so we just check that they are the same struct
            (Self::Struct(id, _), Self::Struct(oid, _)) if id == oid => Relation::Eq,
            (Self::Generic(a), Self::Generic(b)) if a == b => Relation::Eq,
            (Self::Func(ftys, frt), Self::Func(gtys, grt)) => {
                if ftys.len() != gtys.len() {
                    return Relation::Unrelated;
                }
                let mut relation = Relation::Eq;
                for (fty, gty) in ftys.iter().zip(gtys.iter()) {
                    relation = relation.merge_covariant(fty.relation_to(gty));
                    if !relation.compatible() {
                        return Relation::Unrelated;
                    }
                }
                relation.merge_covariant(frt.relation_to(grt))
            }
            _ => Relation::Unrelated,
        }
    }

    pub fn has_same_outer_type(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Unknown(i), Self::Unknown(j)) => i == j,
            (Self::Primitive(p), Self::Primitive(q)) => prim_ty_relation(*p, *q).compatible(),
            (Self::Tuple(tys), Self::Tuple(other_tys)) => tys.len() == other_tys.len(),
            (Self::Array(_, alen), Self::Array(_, blen)) => alen == blen,
            (Self::Struct(id, _), Self::Struct(oid, _)) => id == oid,
            (Self::Func(..), Self::Func(..)) | (Self::Exit(_), _) | (_, Self::Exit(_)) => true,
            _ => false,
        }
    }

    pub fn into_inner_unifications<'a>(self) -> Box<dyn Iterator<Item = Ty> + 'a> {
        match self {
            Self::Tuple(tys) => Box::new(tys.into_iter()),
            Self::Array(ty, None) => Box::new(std::iter::once(*ty)),
            Self::Array(ty, Some(len)) => Box::new([*ty, *len].into_iter()),
            Self::Struct(_, tys) => Box::new(tys.into_iter()),
            Self::Func(params, ret) => Box::new(params.into_iter().chain(std::iter::once(*ret))),
            _ => Box::new(std::iter::empty()),
        }
    }

    pub const fn known(&self) -> bool {
        !matches!(self, Self::Unknown(_))
    }

    pub fn substitute(&mut self, i: usize, t: Self) {
        match self {
            Self::Unknown(j) if *j == i => *self = t,
            Self::Tuple(tys) => {
                for ty in tys {
                    ty.substitute(i, t.clone());
                }
            }
            Self::Array(ty, len) => {
                ty.substitute(i, t.clone());
                len.as_mut().map(|len| len.substitute(i, t));
            }
            Self::Struct(_, tys) => {
                for ty in tys {
                    ty.substitute(i, t.clone());
                }
            }
            Self::Func(params, ret) => {
                for ty in params {
                    ty.substitute(i, t.clone());
                }
                ret.substitute(i, t);
            }
            _ => {}
        }
    }

    pub fn apply(&mut self, substitutions: &VecDeque<Ty>) {
        match self {
            Self::Unknown(i) => {
                match substitutions.get(*i) {
                    // If this substitution is a reference to another unknown, we need to apply that
                    // substitution as well.
                    Some(Self::Unknown(j)) if *j != *i => {
                        *self = Self::Unknown(*j);
                        self.apply(substitutions);
                    }
                    Some(ty) => *self = ty.clone(),
                    None => (),
                }
            }
            Self::Tuple(tys) => {
                for ty in tys {
                    ty.apply(substitutions);
                }
            }
            Self::Array(ty, len) => {
                ty.apply(substitutions);
                len.as_mut().map(|len| len.apply(substitutions));
            }
            Self::Struct(_, tys) => {
                for ty in tys {
                    ty.apply(substitutions);
                }
            }
            Self::Func(params, ret) => {
                for ty in params {
                    ty.apply(substitutions);
                }
                ret.apply(substitutions);
            }
            _ => {}
        }
    }

    pub fn has_any_unknown(&self) -> bool {
        match self {
            Self::Unknown(_) => true,
            Self::Tuple(tys) => tys.iter().any(|ty| ty.has_any_unknown()),
            Self::Array(ty, len) => {
                ty.has_any_unknown() || len.as_ref().map_or(false, |len| len.has_any_unknown())
            }
            Self::Struct(_, tys) => tys.iter().any(|ty| ty.has_any_unknown()),
            Self::Func(params, ret) => {
                params.iter().any(|ty| ty.has_any_unknown()) || ret.has_any_unknown()
            }
            _ => false,
        }
    }

    pub fn has_unknown(&self, i: usize) -> bool {
        match self {
            Self::Unknown(j) => *j == i,
            Self::Tuple(ts) => ts.iter().any(|ty| ty.has_unknown(i)),
            Self::Array(ty, len) => {
                ty.has_unknown(i) || len.as_ref().map_or(false, |len| len.has_unknown(i))
            }
            Self::Struct(_, tys) => tys.iter().any(|ty| ty.has_unknown(i)),
            Self::Func(params, ret) => {
                params.iter().any(|ty| ty.has_unknown(i)) || ret.has_unknown(i)
            }
            _ => false,
        }
    }

    pub fn from_ty(ty: crate::Ty, table: &mut UnificationTable) -> Self {
        match ty {
            crate::Ty::Unknown => {
                let i = table.substitutions.len();
                table.substitutions.push_back(Self::Unknown(i));
                Self::Unknown(i)
            }
            crate::Ty::Primitive(prim) => Self::Primitive(prim),
            crate::Ty::Tuple(tys) => {
                Self::Tuple(tys.into_iter().map(|ty| Self::from_ty(ty, table)).collect())
            }
            crate::Ty::Array(ty, len) => Self::Array(
                Box::new(Self::from_ty(*ty, table)),
                len.map(|len| Box::new(Self::Const(Box::new(TyConst::usize(len))))),
            ),
            crate::Ty::Struct(id, args) => Self::Struct(
                id,
                args.into_iter()
                    .map(|ty| Self::from_ty(ty, table))
                    .collect(),
            ),
            crate::Ty::Generic(id) => Self::Generic(id),
            crate::Ty::Func(params, ret) => Self::Func(
                params
                    .into_iter()
                    .map(|ty| Self::from_ty(ty, table))
                    .collect(),
                Box::new(Self::from_ty(*ret, table)),
            ),
        }
    }
}

impl From<Ty> for crate::Ty {
    fn from(ty: Ty) -> Self {
        match ty {
            Ty::Invalid(_) | Ty::Unknown(_) | Ty::Exit(_) => Self::Unknown,
            Ty::Primitive(prim) => Self::Primitive(prim),
            Ty::Tuple(tys) => Self::Tuple(tys.into_iter().map(Self::from).collect()),
            Ty::Array(ty, len) => Self::Array(
                Box::new(Self::from(*ty)),
                len.map(|len| match *len {
                    Ty::Const(cnst) => cnst.to_usize(),
                    _ => unreachable!(),
                }),
            ),
            Ty::Struct(id, args) => Self::Struct(id, args.into_iter().map(Self::from).collect()),
            Ty::Generic(id) => Self::Generic(id),
            Ty::Func(params, ret) => Self::Func(
                params.into_iter().map(Self::from).collect(),
                Box::new(Self::from(*ret)),
            ),
            Ty::Const(_) => unimplemented!("const types cannot be converted to Ty"),
        }
    }
}

/// Compute the relation between two primitive types.
#[inline]
fn prim_ty_relation(p: PrimitiveTy, q: PrimitiveTy) -> Relation {
    match (p, q) {
        // Unsigned integers coerce and are a subtype of signed integers with a larger bit width.
        // For example, uint32 is a subtype of int64, but not int32 since uint32 could represent
        // larger values than int32.
        //
        // Integers with the same signedness coerce and are a subtype of other integers with the
        // same signedness and the same or larger bit width.
        (PrimitiveTy::Int(ps, mut pw), PrimitiveTy::Int(qs, mut qw)) => {
            pw.naturalize();
            qw.naturalize();
            // if the signs are equal...
            if ps == qs {
                match pw.cmp(&qw) {
                    // ...and the widths are equal, they are equal
                    Ordering::Equal => Relation::Eq,
                    // ...and the width of the first is less than the width of the second, it is a subtype
                    Ordering::Less => Relation::Sub,
                    // ...otherwise it is a supertype
                    Ordering::Greater => Relation::Super,
                }
            } else if ps == IntSign::Unsigned && pw < qw {
                Relation::Sub
            } else {
                Relation::Super
            }
        }
        (PrimitiveTy::Float(mut pw), PrimitiveTy::Float(mut qw)) => {
            pw.naturalize();
            qw.naturalize();
            match pw.cmp(&qw) {
                Ordering::Equal => Relation::Eq,
                Ordering::Less => Relation::Sub,
                Ordering::Greater => Relation::Super,
            }
        }
        (PrimitiveTy::Bool, PrimitiveTy::Bool)
        | (PrimitiveTy::Char, PrimitiveTy::Char)
        | (PrimitiveTy::Void, PrimitiveTy::Void) => Relation::Eq,
        _ => Relation::Unrelated,
    }
}
