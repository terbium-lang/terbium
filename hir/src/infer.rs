use std::any::{Any, TypeId};
use crate::error::Error;
use crate::{error::Result, Expr, FloatWidth, Hir, Ident, IntSign, IntWidth, ItemId, Literal, ModuleId, Node, Pattern, PrimitiveTy, Ty, TyParam};
use common::span::{Span, Spanned};
use std::collections::{HashMap, VecDeque};

#[derive(Clone, Debug)]
pub enum UnificationType<Ctx> {
    Unknown(usize),
    Primitive(PrimitiveTy),
    Tuple(Vec<Unification<Ctx>>),
    Array(Box<Unification<Ctx>>, Option<usize>),
    Struct(ItemId, Vec<Unification<Ctx>>),
    Generic(Ident),
    Func(Vec<Unification<Ctx>>, Box<Unification<Ctx>>),
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
        (PrimitiveTy::Int(ps, pw), PrimitiveTy::Int(qs, qw)) => {
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
        (PrimitiveTy::Float(pw), PrimitiveTy::Float(qw)) => {
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
            | (PrimitiveTy::Void, PrimitiveTy::Void)
            => Relation::Eq,
        _ => Relation::Unrelated,
    }
}

impl<Ctx: Clone> UnificationType<Ctx> {
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
                    relation = relation.merge_covariant(ty.ty().relation_to(other_ty.ty()));
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
                ty.ty().relation_to(oty.ty())
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
                    relation = relation.merge_covariant(fty.ty().relation_to(gty.ty()));
                    if !relation.compatible() {
                        return Relation::Unrelated;
                    }
                }
                relation.merge_covariant(frt.ty().relation_to(grt.ty()))
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

    pub fn into_inner_unifications<'a>(self) -> Box<dyn Iterator<Item = Unification<Ctx>> + 'a>
    where
        Ctx: 'a,
    {
        match self {
            Self::Tuple(tys) => Box::new(tys.into_iter()),
            Self::Array(ty, len) => {
                let len = Unification(UnificationType::ArrayLen(len), ty.1.clone());
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
                    ty.ty_mut().subst(i, t.clone());
                }
            }
            _ => {}
        }
    }

    pub fn has_unknown(&self, i: usize) -> bool {
        match self {
            Self::Unknown(j) => *j == i,
            Self::Tuple(ts) => ts.iter().any(|ty| ty.ty().has_unknown(i)),
            Self::Array(ty, _) => ty.ty().has_unknown(i),
            Self::Struct(_, tys) => tys.iter().any(|ty| ty.ty().has_unknown(i)),
            Self::Func(params, ret) => {
                params.iter().any(|ty| ty.ty().has_unknown(i)) || ret.ty().has_unknown(i)
            }
            _ => false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Unification<Ctx = ()>(pub UnificationType<Ctx>, pub Ctx);

impl<Ctx> Unification<Ctx> {
    pub fn ty(&self) -> &UnificationType<Ctx> {
        &self.0
    }

    pub fn ty_mut(&mut self) -> &mut UnificationType<Ctx> {
        &mut self.0
    }
    
    pub fn ctx(&self) -> &Ctx {
        &self.1
    }
    
    pub fn map_ctx<U: Clone>(self, f: impl FnOnce(Ctx) -> U) -> Unification<U> {
        let ctx = f(self.1);
        let ctxf = |_| ctx.clone();

        let ty = match self.0 {
            UnificationType::Unknown(i) => UnificationType::Unknown(i),
            UnificationType::Primitive(prim) => UnificationType::Primitive(prim),
            UnificationType::Tuple(tys) => {
                UnificationType::Tuple(tys.into_iter().map(|ty| ty.map_ctx(ctxf)).collect())
            }
            UnificationType::Array(ty, len) => {
                UnificationType::Array(Box::new(ty.map_ctx(ctxf)), len)
            }
            UnificationType::Struct(id, tys) => {
                UnificationType::Struct(id, tys.into_iter().map(|ty| ty.map_ctx(ctxf)).collect())
            }
            UnificationType::Func(params, ret) => {
                UnificationType::Func(
                    params.into_iter().map(|ty| ty.map_ctx(ctxf)).collect(),
                    Box::new(ret.map_ctx(ctxf)),
                )
            }
            UnificationType::Generic(g) => UnificationType::Generic(g),
            UnificationType::ArrayLen(i) => UnificationType::ArrayLen(i),
        };
        Unification(ty, ctx)
    }
}

impl Unification {
    pub fn from_ty(ty: Ty, counter: &mut usize) -> Self {
        let ty = match ty {
            Ty::Unknown => {
                let i = *counter;
                *counter += 1;
                UnificationType::Unknown(i)
            }
            Ty::Primitive(prim) => UnificationType::Primitive(prim),
            Ty::Tuple(tys) => UnificationType::Tuple(
                tys.into_iter()
                    .map(|ty| Unification::from_ty(ty, counter))
                    .collect(),
            ),
            Ty::Array(ty, len) => UnificationType::Array(
                Box::new(Unification::from_ty(*ty, counter)),
                len,
            ),
            Ty::Struct(id, args) => UnificationType::Struct(
                id,
                args.into_iter()
                    .map(|ty| Unification::from_ty(ty, counter))
                    .collect(),
            ),
            Ty::Generic(id) => UnificationType::Generic(id),
            Ty::Func(params, ret) => UnificationType::Func(
                params.into_iter().map(|ty| Unification::from_ty(ty, counter)).collect(),
                Box::new(Unification::from_ty(*ret, counter)),
            ),
        };
        Unification(ty, ())
    }
}

impl Unification<Spanned<Expr>> {
    pub fn from_ty_expr(ty: Ty, expr: Spanned<Expr>, counter: &mut usize) -> Self {
        let ty = match (ty, expr.clone()) {
            (Ty::Unknown, _) => {
                let i = *counter;
                *counter += 1;
                UnificationType::Unknown(i)
            }
            (Ty::Primitive(prim), _) => UnificationType::Primitive(prim),
            (Ty::Tuple(tys), Spanned(Expr::Tuple(exprs), _)) => {
                assert_eq!(tys.len(), exprs.len());
                UnificationType::Tuple(
                    tys.into_iter()
                        .zip(exprs.into_iter())
                        .map(|(ty, expr)| Unification::from_ty_expr(ty, expr, counter))
                        .collect(),
                )
            }
            (Ty::Array(ty, len), Spanned(Expr::Array(exprs), _)) => {
                if let Some(len) = len {
                    assert_eq!(len, exprs.len());
                }
                UnificationType::Array(
                    Box::new(Unification::from_ty_expr(*ty, exprs[0].clone(), counter)),
                    len,
                )
            }
            _ => todo!(),
        };
        Unification(ty, expr)
    }
}

impl<Ctx> From<Unification<Ctx>> for Ty {
    fn from(value: Unification<Ctx>) -> Self {
        value.0.into()
    }
}

impl<Ctx> From<UnificationType<Ctx>> for Ty {
    fn from(ty: UnificationType<Ctx>) -> Self {
        match ty {
            UnificationType::Unknown(_) => Ty::Unknown,
            UnificationType::Primitive(prim) => Ty::Primitive(prim),
            UnificationType::Tuple(tys) => Ty::Tuple(tys.into_iter().map(Ty::from).collect()),
            UnificationType::Array(ty, len) => Ty::Array(Box::new(Ty::from((*ty).0)), len),
            UnificationType::Struct(id, args) => {
                Ty::Struct(id, args.into_iter().map(Ty::from).collect())
            }
            UnificationType::Generic(id) => Ty::Generic(id),
            UnificationType::Func(params, ret) => {
                Ty::Func(params.into_iter().map(Ty::from).collect(), Box::new(Ty::from((*ret).0)))
            }
            UnificationType::ArrayLen(_) => unimplemented!("ArrayLen"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Constraint<Ctx = ()>(Unification<Ctx>, Unification<Ctx>);

/// A type unifier.
#[derive(Debug)]
pub struct Unifier<Ctx = ()> {
    pub span: Span,
    pub constraints: VecDeque<Constraint<Ctx>>,
    pub substitutions: Vec<(usize, UnificationType<Ctx>)>,
}

impl<Ctx: Clone> Unifier<Ctx> {
    pub fn substitute(&mut self, a: usize, b: UnificationType<Ctx>) {
        // Add a => b to our substitution
        self.substitutions.push((a, b.clone()));
        // Replace further unifications of a with b
        for Constraint(x, y) in self.constraints.iter_mut() {
            x.ty_mut().subst(a, b.clone());
            y.ty_mut().subst(a, b.clone());
        }
    }

    /// Unify the constraint a = b.
    pub fn unify_constraint(&mut self, constraint: Constraint<Ctx>) -> Result<()> where Ctx: 'static {
        match constraint {
            // If both a and b are unknown, replace one of them with the other.
            Constraint(
                Unification(UnificationType::Unknown(i), _),
                Unification(b @ UnificationType::Unknown(_), _),
            ) => {
                self.substitute(i, b);
            }
            // If a is unknown and b is known, replace a with b. Similarly, if b is unknown and a is
            // known, replace b with a.
            Constraint(Unification(UnificationType::Unknown(i), _), b)
            | Constraint(b, Unification(UnificationType::Unknown(i), _))
            => {
                // Do not allow recursive unification (i = a<i>)
                if b.ty().has_unknown(i) {
                    return Err(Error::CyclicTypeConstraint {
                        span: self.span,
                        rhs: b.into(),
                    });
                }
                self.substitute(i, b.0);
            }
            // If both a and b are known, check if they are the same general type (e.g. A<...> = A<...>)
            // and if so, unify their type arguments.
            Constraint(a, b) if b.ty().has_same_outer_type(a.ty()) => {
                for (a, b) in a.0.into_inner_unifications().zip(b.0.into_inner_unifications()) {
                    self.unify_constraint(Constraint(a, b))?;
                }
            }
            // If both a and b are known but they are not the same general type, this is a type
            // mismatch
            Constraint(Unification(a, ax), b) => {
                // SAFETY: wtf lol
                let span = if ax.type_id() == TypeId::of::<Spanned<Expr>>() {
                    unsafe { (*(&ax as *const Ctx as *const Spanned<Expr> )).1 }
                } else {
                    self.span
                };
                return Err(Error::TypeMismatch {
                    expected: (a.into(), None),
                    actual: Spanned(b.into(), span),
                });
            }
        }
        Ok(())
    }

    /// Unifies all constraints in the unifier.
    pub fn unify(&mut self) -> Result<()> where Ctx: 'static {
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
            return Ok(cnst.ty.clone())
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
            Pattern::Tuple(pats) => {
                Ty::Tuple(pats.iter().map(|pat| self.bind_pattern_to_inference_ty(pat)).collect())
            }
        }
    }

    /// Runs type inference through a single node.
    pub fn lower_node(&mut self, node: &mut Node) -> Result<()> {
        match node {
            Node::Expr(expr) => {
                self.lower_expr_ty(expr)?;
                Ok(())
            },
            Node::Let { pat, ty, value: Some(value) } => {
                let mut counter = 0;
                let expr_ty = self.lower_expr_ty(value)?;
                let value = value.clone();

                // constraint: structure of pattern = structure of value
                let pat_ty = self.bind_pattern_to_inference_ty(pat);
                let pat_constraint = Constraint(
                    Unification::from_ty_expr(pat_ty, value.clone(), &mut counter),
                    Unification::from_ty_expr(expr_ty.clone(), value.clone(), &mut counter),
                );

                // constraint: type hint = structure of value
                let ty_constraint = Constraint(
                    Unification::from_ty_expr(ty.clone(), value.clone(), &mut counter),
                    Unification::from_ty_expr(expr_ty, value.clone(), &mut counter),
                );

                let mut unifier = Unifier {
                    span: value.span(),
                    constraints: VecDeque::from([pat_constraint, ty_constraint]),
                    substitutions: Vec::new(),
                };
                unifier.unify()?;

                println!("{unifier:#?}");
                Ok(()) // TODO
            },
            _ => todo!(),
        }
    }
}
