//! Typeck stage of HIR.

use crate::{infer::InferMetadata, infer::TypeLowerer, typed::UnificationTable, Hir, ScopeId};

/// Performs any remaining type checking and desugaring with the knowledge of the types of all
/// expressions.
///
/// Promotes the produced THIR from inference back to a standard HIR without all the
/// inference noise.
pub struct TypeChecker {
    table: UnificationTable,
    /// The THIR to type check.
    pub thir: Hir<InferMetadata>,
    /// The HIR being produced.
    pub hir: Hir,
}

impl TypeChecker {
    /// Create a new type checker from the given TypeLowerer.
    pub fn from_lowerer(lower: TypeLowerer) -> Self {
        Self {
            table: lower.table,
            thir: lower.thir,
            hir: Hir::default(),
        }
    }
}
