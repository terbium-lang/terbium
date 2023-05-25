use crate::{
    Ident, IntSign, IntWidth, Intrinsic, ItemId, Literal, Node, Op, Pattern, PrimitiveTy, ScopeId,
    StaticOp,
};
use common::span::Spanned;
use std::cmp::Ordering;
use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub enum Expr {
    Literal(Literal),
    Ident(Spanned<Ident>, Option<Spanned<Vec<Ty>>>),
    Tuple(Vec<E>),
    Array(Vec<E>),
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InvalidTypeCause {
    CyclicTypeReference,
    TypeConflict,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ty {
    Invalid(InvalidTypeCause),
    Unknown(usize),
    Primitive(PrimitiveTy),
    Generic(Ident),
    Tuple(Vec<Self>),
    Array(Box<Self>, Option<Box<Self>>),
    Struct(ItemId, Vec<Self>),
    Func(Vec<Self>, Box<Self>),
    Const(Box<TyConst>),
}

impl Ty {
    /// Example: `u32` is a subtype of `u64` because they coerce.
    pub fn relation_to(&self, other: &Self) -> Relation {
        match (self, other) {
            (Self::Unknown(i), Self::Unknown(j)) if i == j => Relation::Eq,
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
            (Self::Func(..), Self::Func(..)) => true,
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
                if let Some(ty) = substitutions.get(*i) {
                    *self = ty.clone();
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
            Ty::Invalid(_) | Ty::Unknown(_) => Self::Unknown,
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
