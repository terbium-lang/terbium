use std::fmt::{Display, Formatter};
use common::span::{Span, Spanned};
use grammar::ast::TypePath;
use crate::ModuleId;

pub type Result<T> = std::result::Result<T, AstLoweringError>;

/// Represents an error that occurred during the lowering of AST to HIR.
#[derive(Clone, Debug)]
pub enum AstLoweringError {
    /// A struct with the given type name was not found.
    ///
    /// Structs must extend their fields from a struct type declared before them. This is to prevent
    /// circular struct definitions.
    StructNotDeclared(Spanned<String>, ModuleId),
    /// The provided type is not a concrete struct. Structs may only extend fields from other
    /// structs.
    CannotExtendFieldsFromType(Spanned<TypePath>),
    /// Integer literal will overflow 128 bits. If the integer is not unsigned, making it unsigned
    /// may increase the maximum value before an overflow such as this error occurs.
    IntegerLiteralOverflow(Span),
    /// Float literal will overflow 64 bits.
    FloatLiteralOverflow(Span),
}

impl Display for AstLoweringError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StructNotDeclared(name, module) => {
                write!(f, "could not find struct `{name}` in module `{module}`")
            }
            Self::CannotExtendFieldsFromType(ty) => {
                write!(f, "cannot extend fields from type `{ty}`")
            }
            Self::IntegerLiteralOverflow(_) => {
                write!(f, "integer literal will overflow 128 bits")
            }
            Self::FloatLiteralOverflow(_) => {
                write!(f, "float literal will overflow 64 bits")
            }
        }
    }
}

impl std::error::Error for AstLoweringError {}
