use crate::error::Error;
use crate::{
    error::Result, Expr, FloatWidth, Hir, Ident, IntSign, IntWidth, ItemId, Literal, ModuleId,
    Node, Pattern, PrimitiveTy, Ty, TyParam,
};
use common::span::{Span, Spanned};
use std::any::{Any, TypeId};
use std::collections::{HashMap, VecDeque};

#[derive(Clone, Debug)]
pub enum UnificationType {
    Unknown(usize),
    Primitive(PrimitiveTy),
    Tuple(Vec<Self>),
    Array(Box<Self>, Option<usize>),
    Struct(ItemId, Vec<Self>),
    Generic(Ident),
    Func(Vec<Self>, Box<Self>),
    // Constant-like types
    ArrayLen(Option<usize>),
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
                if pw == qw {
                    // ...and the widths are equal, they are equal
                    Relation::Eq
                } else if pw < qw {
                    // ...and the width of the first is less than the width of the second, it is a subtype
                    Relation::Sub
                } else {
                    // ...otherwise it is a supertype
                    Relation::Super
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
            if pw == qw {
                Relation::Eq
            } else if pw < qw {
                Relation::Sub
            } else {
                Relation::Super
            }
        }
        (PrimitiveTy::Bool, PrimitiveTy::Bool)
        | (PrimitiveTy::Char, PrimitiveTy::Char)
        | (PrimitiveTy::Void, PrimitiveTy::Void) => Relation::Eq,
        _ => Relation::Unrelated,
    }
}

impl UnificationType {
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
            (Self::ArrayLen(i), Self::ArrayLen(j)) => i == j,
            _ => false,
        }
    }

    pub fn into_inner_unifications<'a>(self) -> Box<dyn Iterator<Item = Self> + 'a> {
        match self {
            Self::Tuple(tys) => Box::new(tys.into_iter()),
            Self::Array(ty, len) => {
                let len = UnificationType::ArrayLen(len);
                Box::new([*ty, len].into_iter())
            }
            Self::Struct(_, tys) => Box::new(tys.into_iter()),
            Self::Func(params, ret) => Box::new(params.into_iter().chain(std::iter::once(*ret))),
            _ => Box::new(std::iter::empty()),
        }
    }

    pub const fn known(&self) -> bool {
        !matches!(self, Self::Unknown(_))
    }

    pub fn subst(&mut self, i: usize, t: Self) {
        match self {
            Self::Unknown(j) if *j == i => *self = t,
            Self::Tuple(tys) => {
                for ty in tys {
                    ty.subst(i, t.clone());
                }
            }
            _ => {}
        }
    }

    pub fn has_unknown(&self, i: usize) -> bool {
        match self {
            Self::Unknown(j) => *j == i,
            Self::Tuple(ts) => ts.iter().any(|ty| ty.has_unknown(i)),
            Self::Array(ty, _) => ty.has_unknown(i),
            Self::Struct(_, tys) => tys.iter().any(|ty| ty.has_unknown(i)),
            Self::Func(params, ret) => {
                params.iter().any(|ty| ty.has_unknown(i)) || ret.has_unknown(i)
            }
            _ => false,
        }
    }
}

impl UnificationType {
    pub fn from_ty(ty: Ty, counter: &mut usize) -> Self {
        match ty {
            Ty::Unknown => {
                let i = *counter;
                *counter += 1;
                UnificationType::Unknown(i)
            }
            Ty::Primitive(prim) => UnificationType::Primitive(prim),
            Ty::Tuple(tys) => UnificationType::Tuple(
                tys.into_iter()
                    .map(|ty| Self::from_ty(ty, counter))
                    .collect(),
            ),
            Ty::Array(ty, len) => {
                UnificationType::Array(Box::new(Self::from_ty(*ty, counter)), len)
            }
            Ty::Struct(id, args) => UnificationType::Struct(
                id,
                args.into_iter()
                    .map(|ty| Self::from_ty(ty, counter))
                    .collect(),
            ),
            Ty::Generic(id) => UnificationType::Generic(id),
            Ty::Func(params, ret) => UnificationType::Func(
                params
                    .into_iter()
                    .map(|ty| Self::from_ty(ty, counter))
                    .collect(),
                Box::new(Self::from_ty(*ret, counter)),
            ),
        }
    }
}

impl From<UnificationType> for Ty {
    fn from(ty: UnificationType) -> Self {
        match ty {
            UnificationType::Unknown(_) => Ty::Unknown,
            UnificationType::Primitive(prim) => Ty::Primitive(prim),
            UnificationType::Tuple(tys) => Ty::Tuple(tys.into_iter().map(Ty::from).collect()),
            UnificationType::Array(ty, len) => Ty::Array(Box::new(Ty::from(*ty)), len),
            UnificationType::Struct(id, args) => {
                Ty::Struct(id, args.into_iter().map(Ty::from).collect())
            }
            UnificationType::Generic(id) => Ty::Generic(id),
            UnificationType::Func(params, ret) => Ty::Func(
                params.into_iter().map(Ty::from).collect(),
                Box::new(Ty::from(*ret)),
            ),
            UnificationType::ArrayLen(_) => unimplemented!("ArrayLen"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Constraint(pub UnificationType, pub UnificationType);

/// A type unifier.
#[derive(Debug)]
pub struct Unifier {
    pub span: Span,
    pub constraints: VecDeque<Constraint>,
    pub substitutions: Vec<(usize, UnificationType)>,
    pub conflicts: Vec<Constraint>,
}

impl Unifier {
    pub fn substitute(&mut self, a: usize, b: UnificationType) {
        // Add a => b to our substitution
        self.substitutions.push((a, b.clone()));
        // Replace further unifications of a with b
        for Constraint(x, y) in self.constraints.iter_mut() {
            x.subst(a, b.clone());
            y.subst(a, b.clone());
        }
    }

    /// Unify the constraint a = b.
    pub fn unify_constraint(&mut self, constraint: Constraint) -> Result<()> {
        match constraint {
            // If both a and b are unknown, replace one of them with the other.
            Constraint(UnificationType::Unknown(i), b @ UnificationType::Unknown(_)) => {
                self.substitute(i, b);
            }
            // If a is unknown and b is known, replace a with b. Similarly, if b is unknown and a is
            // known, replace b with a.
            Constraint(UnificationType::Unknown(i), b)
            | Constraint(b, UnificationType::Unknown(i)) => {
                // Do not allow recursive unification (i = a<i>)
                if b.has_unknown(i) {
                    return Err(Error::CyclicTypeConstraint {
                        span: self.span,
                        rhs: b.into(),
                    });
                }
                self.substitute(i, b);
            }
            // If both a and b are known, check if they are the same general type (e.g. A<...> = A<...>)
            // and if so, unify their type arguments.
            Constraint(a, b) if b.has_same_outer_type(&a) => {
                for (a, b) in a.into_inner_unifications().zip(b.into_inner_unifications()) {
                    self.unify_constraint(Constraint(a, b))?;
                }
            }
            // If both a and b are known but they are not the same general type, this is a type
            // mismatch
            _ => {
                self.conflicts.push(constraint);
            }
        }
        Ok(())
    }

    /// Unifies all constraints in the unifier.
    pub fn unify(&mut self) -> Result<()> {
        while let Some(constraint) = self.constraints.pop_front() {
            self.unify_constraint(constraint)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Scope {
    module_id: ModuleId,
    ty_params: Vec<TyParam>,
    locals: HashMap<Ident, Ty>,
}

/// Lowers types of expressions and performs type inference.
#[derive(Debug)]
pub struct TypeLowerer {
    scopes: Vec<Scope>,
    /// The HIR that is being lowered.
    pub hir: Hir,
}

impl TypeLowerer {
    /// Creates a new type lowerer over the given HIR.
    pub fn new(hir: Hir) -> Self {
        Self {
            scopes: Vec::new(),
            hir,
        }
    }

    #[inline]
    pub fn scope(&self) -> &Scope {
        self.scopes.last().expect("no scopes entered?")
    }

    pub fn lower_literal(lit: &Literal) -> Ty {
        match lit {
            Literal::Void => Ty::Primitive(PrimitiveTy::Void),
            Literal::Bool(_) => Ty::Primitive(PrimitiveTy::Bool),
            Literal::Int(i) => {
                // Fit the integer into the smallest possible type (at least 32 bits)
                let width = if i >= &i32::MIN.into() && i <= &i32::MAX.into() {
                    IntWidth::Int32
                } else if i >= &i64::MIN.into() && i <= &i64::MAX.into() {
                    IntWidth::Int64
                } else {
                    IntWidth::Int128
                };
                Ty::Primitive(PrimitiveTy::Int(IntSign::Signed, width))
            }
            Literal::UInt(u) => {
                let width = if u <= &u32::MAX.into() {
                    IntWidth::Int32
                } else if u <= &u64::MAX.into() {
                    IntWidth::Int64
                } else {
                    IntWidth::Int128
                };
                Ty::Primitive(PrimitiveTy::Int(IntSign::Unsigned, width))
            }
            Literal::Float(_) => Ty::Primitive(PrimitiveTy::Float(FloatWidth::Float64)),
            Literal::Char(_) => Ty::Primitive(PrimitiveTy::Char),
            Literal::String(_) => unimplemented!(),
        }
    }

    #[inline]
    fn resolve_ident_ty(
        &self,
        ident: &Spanned<Ident>,
        args: &Option<Spanned<Vec<Ty>>>,
    ) -> Result<Ty> {
        if let Some(args) = args {
            // TODO: function pointers, first-class types
            return Err(Error::ExplicitTypeArgumentsNotAllowed(args.span()));
        }

        // Search for the local variable
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.locals.get(ident.value()) {
                return Ok(ty.clone());
            }
        }

        // Does a constant with this name exist?
        let item = ItemId(self.scope().module_id, *ident.value());
        if let Some(cnst) = self.hir.consts.get(&item) {
            return Ok(cnst.ty.clone());
        }

        Err(Error::UnresolvedIdentifier(ident.map(|i| i.to_string())))
    }

    /// Lowers the type of an expression.
    pub fn lower_expr_ty(&mut self, expr: &Spanned<Expr>) -> Result<Ty> {
        Ok(match expr.value() {
            Expr::Literal(lit) => Self::lower_literal(lit),
            Expr::Ident(ident, args) => self.resolve_ident_ty(ident, args)?,
            Expr::Tuple(exprs) => {
                let mut tys = Vec::with_capacity(exprs.len());
                for expr in exprs {
                    tys.push(self.lower_expr_ty(expr)?);
                }
                Ty::Tuple(tys)
            }
            Expr::Cast(_, ty) => ty.clone(),
            // TODO
            _ => Ty::Unknown,
        })
    }

    /// Expands a pattern binding to a type of unknowns, for example:
    /// * `(a, b)` becomes `(_, _)`
    /// * `(a, (b, c))` becomes `(_, (_, _))`
    pub fn bind_pattern_to_inference_ty(&self, pat: &Pattern) -> Ty {
        match pat {
            Pattern::Ident { .. } => Ty::Unknown,
            Pattern::Tuple(pats) => Ty::Tuple(
                pats.iter()
                    .map(|pat| self.bind_pattern_to_inference_ty(pat))
                    .collect(),
            ),
        }
    }

    /// Runs type inference through a single node.
    pub fn lower_node(&mut self, node: &mut Node) -> Result<()> {
        match node {
            Node::Expr(expr) => {
                self.lower_expr_ty(expr)?;
            }
            // This is a proof-of-concept implementation of type inference. It is not complete.
            Node::Let {
                pat,
                ty,
                value: Some(value),
            } => {
                let mut counter = 0;
                let expr_ty = self.lower_expr_ty(value)?;
                let value = value.clone();

                let mut uty = UnificationType::from_ty(ty.clone(), &mut counter);
                // constraint: type hint = structure of value
                let ty_constraint =
                    Constraint(uty.clone(), UnificationType::from_ty(expr_ty, &mut counter));

                let mut unifier = Unifier {
                    span: value.span(),
                    constraints: VecDeque::from([ty_constraint]),
                    substitutions: Vec::new(),
                    conflicts: Vec::new(),
                };
                unifier.unify()?;

                for (i, to) in unifier.substitutions {
                    uty.subst(i, to);
                }
                for conflict in unifier.conflicts {
                    println!(
                        "conflict: {} != {}",
                        Ty::from(conflict.0),
                        Ty::from(conflict.1)
                    );
                }
                *ty = Ty::from(uty);
            }
            _ => (), // TODO
        }
        Ok(())
    }
}
