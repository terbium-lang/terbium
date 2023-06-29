//! Lowers a typed HIR to MIR.

use std::collections::HashMap;
use common::span::{Spanned, SpannedExt};
use hir::{Ident, IntSign, Literal, ModuleId, PrimitiveTy, ScopeId, typed::Ty};
use hir::lower::{get_ident, get_ident_from_ref};
use hir::typed::{LocalEnv, TypedExpr};
use crate::{BlockId, BlockMap, BoolIntrinsic, Constant, Expr, IntIntrinsic, LocalId, Mir, Node, TypedHir};

#[derive(Clone, Debug)]
#[must_use = "the lowerer must be called to produce the MIR"]
pub struct Lowerer {
    /// The typed HIR to lower.
    pub thir: TypedHir,
    /// The MIR being constructed.
    pub mir: Mir,

    // Keep a current stack of blocks
    blocks: Vec<BlockId>,
    // Keep a mapping of labels to blocks
    labels: HashMap<Ident, BlockId>,
}

struct Ctx<'a> {
    blocks: &'a mut BlockMap,
    current: &'a mut Vec<Spanned<Node>>,
}

impl Lowerer {
    pub fn from_thir(thir: TypedHir) -> Self {
        let len = thir.scopes.len();
        Self {
            thir,
            mir: Mir::default(),
            blocks: Vec::with_capacity(len),
            labels: HashMap::with_capacity(len)
        }
    }

    /// Lowers a literal into a constant.
    pub fn lower_literal(&mut self, literal: Literal, ty: &Ty) -> Constant {
        use Ty::Primitive as P;

        match (literal, ty) {
            (Literal::UInt(u), P(PrimitiveTy::Int(_, width))) =>
                Constant::Int(u, IntSign::Unsigned, *width),
            (Literal::Int(i), P(PrimitiveTy::Int(sign, width))) => {
                Constant::Int(i as _, *sign, *width)
            }
            (Literal::Bool(b), _) => Constant::Bool(b),
            (Literal::Char(c), _) => Constant::Char(c),
            (Literal::Void, _) => Constant::Void,
            // TODO: this implementation will probably change in the future
            (Literal::String(s), _) => {
                Constant::Array(s.chars().map(Constant::Char).collect())
            }
            _ => unimplemented!(),
        }
    }

    pub fn lower_expr(
        &mut self,
        ctx: &mut Ctx,
        Spanned(TypedExpr(expr, ty), span): Spanned<TypedExpr>,
    ) -> Spanned<Expr> {
        type HirExpr = hir::typed::Expr;

        match expr {
            HirExpr::Literal(lit) => Expr::Constant(self.lower_literal(lit, &ty)),
            HirExpr::IntIntrinsic(intr, sign, width) => {
                use hir::typed::IntIntrinsic::{Unary, Binary};

                let intr = match intr {
                    Unary(op, expr) => {
                        IntIntrinsic::Unary(op, Box::new(self.lower_expr(ctx, *expr)))
                    },
                    Binary(op, lhs, rhs) => {
                        IntIntrinsic::Binary(
                            op,
                            Box::new(self.lower_expr(ctx, *lhs)),
                            Box::new(self.lower_expr(ctx, *rhs)),
                        )
                    }
                };
                Expr::IntIntrinsic(intr, sign, width)
            }
            HirExpr::BoolIntrinsic(intr) => {
                use hir::typed::BoolIntrinsic::*;

                Expr::BoolIntrinsic(match intr {
                    Not(expr) => BoolIntrinsic::Not(Box::new(self.lower_expr(ctx, *expr))),
                    And(lhs, rhs) | Or(lhs, rhs) | Xor(lhs, rhs) => {
                        let f: fn(_, _) -> _ = match intr {
                            Not(_) => unreachable!(),
                            And(..) => BoolIntrinsic::And,
                            Or(..) => BoolIntrinsic::Or,
                            Xor(..) => BoolIntrinsic::Xor,
                        };
                        f(Box::new(self.lower_expr(ctx, *lhs)), Box::new(self.lower_expr(ctx, *rhs)))
                    }
                })
            }
            HirExpr::Local(local, _, env) => {
                Expr::Local(LocalId(local.into_value(), env))
            }
            // In order to lower a block with a potential result,
            // create a temporary local `result` and jump to a new block
            // that will assign the result to the local and jump to the continuation block.
            //
            // For example:
            // ```
            // let x = { let y = 1 + 1; y };
            // ```
            // will be lowered to:
            //
            // entry:
            //   init %internal.result ; temporary local
            //   jump _bb0 ; jump to the block that will assign the result
            // _bb0:
            //   init %local.y
            //   store %local.y <- 1 + 1
            //   store %internal.result <- %y ; assign the result
            //   jump _bb0.after
            // _bb0.after:
            //   ... ; jump to the continuation block
            HirExpr::Block(scope) => {
                let block_id = BlockId::from(scope);
                let cont_id = BlockId(format!("_bb{}.after", scope.0).into());
                // Create the temporary result local
                let result_local = LocalId(format!("result{}", scope.0).into(), LocalEnv::Internal);
                ctx.current.extend([
                    Node::Init(result_local).spanned(span),
                    // Jump to the block that will assign the result
                    Node::Jump(block_id).spanned(span),
                ]);
                self.lower_scope(ctx.blocks, scope, block_id, |slf, value, current| {
                    let span = value.span();
                    let expr = slf.lower_expr(ctx, value);
                    // Assign the result to the temporary local
                    current.push(Node::Expr(Expr::Assign(result_local, Box::new(expr)).spanned(span)).spanned(span));
                    Node::Jump(cont_id)
                });
                Expr::Local(result_local)
            }
            _ => todo!()
        }
        .spanned(span)
    }

    /// Lowers the scope and returns the entrypoint block.
    pub fn lower_scope(&mut self, blocks: &mut BlockMap, scope: ScopeId, entry_id: BlockId, mut on_implicit_return: impl FnMut(&mut Self, Spanned<TypedExpr>, &mut Vec<Spanned<Node>>) -> Node) -> &mut Vec<Spanned<Node>> {
        let scope = self.thir.scopes.remove(&scope).expect("no such scope");

        let mut entry = Vec::with_capacity(scope.children.value().len());
        let mut ctx = Ctx { blocks, current: &mut entry };

        for Spanned(node, span) in scope.children.into_value() {
            let node = match node {
                hir::Node::Expr(expr) => {
                    Node::Expr(self.lower_expr(&mut ctx, expr))
                }
                hir::Node::ImplicitReturn(value) => on_implicit_return(self, value, &mut ctx.current),
                _ => todo!()
            };
            entry.push(Spanned(node, span));
        }
        ctx.blocks.try_insert(entry_id, entry).expect("block already exists")
    }

    pub fn lower_module(&mut self, module: ModuleId) {

    }
}
