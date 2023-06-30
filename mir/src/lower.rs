//! Lowers a typed HIR to MIR.

use crate::{
    BlockId, BlockMap, BoolIntrinsic, Constant, Expr, Func, IntIntrinsic, LocalId, Mir, Node,
    TypedHir,
};
use common::span::{Spanned, SpannedExt};
use hir::{
    typed::{LocalEnv, Ty, TypedExpr},
    Ident, IntSign, ItemId, Literal, ModuleId, PrimitiveTy, ScopeId,
};
use std::collections::HashMap;

#[derive(Clone, Debug)]
#[must_use = "the lowerer must be called to produce the MIR"]
pub struct Lowerer {
    /// The typed HIR to lower.
    pub thir: TypedHir,
    /// The MIR being constructed.
    pub mir: Mir,
}

pub struct Ctx<'a> {
    blocks: *mut BlockMap,
    current: &'a mut Vec<Spanned<Node>>,
    track: BlockId,
    bctx: *mut BlockCtx<'a>,
}

impl<'a> Ctx<'a> {
    /// Move the context to start writing in the given block.
    pub(crate) fn move_to(&mut self, block_id: BlockId) {
        self.current = unsafe { &mut *self.blocks }
            .try_insert(block_id, Vec::new())
            .expect("block already exists");
        self.track = block_id;
    }
}

pub struct BlockCtx<'a> {
    // nearest loop block
    continue_to: Option<BlockId>,
    // nearest loop continuation block along with the local to store the result into
    break_to: Option<(BlockId, LocalId)>,
    // keep a lookup of labels to their blocks
    label_continue_map: &'a mut HashMap<Ident, BlockId>,
    // keep a lookup of labels to their block continuations
    label_break_map: &'a mut HashMap<Ident, (BlockId, Option<LocalId>)>,
}

impl<'a> BlockCtx<'a> {
    /// Register a label to map to a block and continuation.
    pub(crate) fn store_label(
        &mut self,
        lowerer: &mut Lowerer,
        scope_id: ScopeId,
        block: BlockId,
        cont: BlockId,
        result: Option<LocalId>,
    ) {
        let Some(Spanned(label, _)) = lowerer.thir.scopes[&scope_id].label else { return };

        self.label_continue_map.insert(label, block);
        self.label_break_map.insert(label, (cont, result));
    }
}

impl Lowerer {
    pub fn from_thir(thir: TypedHir) -> Self {
        Self {
            thir,
            mir: Mir::default(),
        }
    }

    /// Lowers a literal into a constant.
    pub fn lower_literal(&mut self, literal: Literal, ty: &Ty) -> Constant {
        use Ty::Primitive as P;

        match (literal, ty) {
            (Literal::UInt(u), P(PrimitiveTy::Int(_, width))) => {
                Constant::Int(u, IntSign::Unsigned, *width)
            }
            (Literal::Int(i), P(PrimitiveTy::Int(sign, width))) => {
                Constant::Int(i as _, *sign, *width)
            }
            (Literal::Bool(b), _) => Constant::Bool(b),
            (Literal::Char(c), _) => Constant::Char(c),
            (Literal::Void, _) => Constant::Void,
            // TODO: this implementation will probably change in the future
            (Literal::String(s), _) => Constant::Array(s.chars().map(Constant::Char).collect()),
            _ => unimplemented!(),
        }
    }

    pub fn lower_expr(
        &mut self,
        ctx: &mut Ctx,
        Spanned(TypedExpr(expr, ty), span): Spanned<TypedExpr>,
    ) -> Spanned<Expr> {
        type HirExpr = hir::typed::Expr;

        // SAFETY: the context is valid for the duration of the function
        let bctx = unsafe { &mut *ctx.bctx };
        match expr {
            HirExpr::Literal(lit) => Expr::Constant(self.lower_literal(lit, &ty)),
            HirExpr::IntIntrinsic(intr, sign, width) => {
                use hir::typed::IntIntrinsic::{Binary, Unary};

                let intr = match intr {
                    Unary(op, expr) => {
                        IntIntrinsic::Unary(op, Box::new(self.lower_expr(ctx, *expr)))
                    }
                    Binary(op, lhs, rhs) => IntIntrinsic::Binary(
                        op,
                        Box::new(self.lower_expr(ctx, *lhs)),
                        Box::new(self.lower_expr(ctx, *rhs)),
                    ),
                };
                Expr::IntIntrinsic(intr, sign, width)
            }
            HirExpr::BoolIntrinsic(intr) => {
                use hir::typed::BoolIntrinsic::*;

                Expr::BoolIntrinsic(match intr {
                    Not(expr) => BoolIntrinsic::Not(Box::new(self.lower_expr(ctx, *expr))),
                    And(..) | Or(..) | Xor(..) => {
                        let (f, lhs, rhs): (fn(_, _) -> _, _, _) = match intr {
                            Not(_) => unreachable!(),
                            And(lhs, rhs) => (BoolIntrinsic::And, lhs, rhs),
                            Or(lhs, rhs) => (BoolIntrinsic::Or, lhs, rhs),
                            Xor(lhs, rhs) => (BoolIntrinsic::Xor, lhs, rhs),
                        };
                        f(
                            Box::new(self.lower_expr(ctx, *lhs)),
                            Box::new(self.lower_expr(ctx, *rhs)),
                        )
                    }
                })
            }
            HirExpr::Local(local, _, env) => Expr::Local(LocalId(local.into_value(), env)),
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
            //   init %local.x
            //   init %internal.result ; temporary local
            //   jump _bb0 ; jump to the block that will assign the result
            // _bb0:
            //   init %local.y
            //   store %local.y <- 1 + 1
            //   store %internal.result <- %y ; assign the result
            //   jump _bb0.after ; jump to the continuation block
            // _bb0.after:
            //   store %local.x <- %internal.result ; now the result can be used here
            HirExpr::Block(scope) => {
                let block_id = BlockId::from(scope);
                let cont_id = BlockId(format!("_bb{}.after", scope.0).into());
                // Create the temporary result local
                let result_local = LocalId(format!("result{}", scope.0).into(), LocalEnv::Internal);
                ctx.current.extend([
                    Node::Local(result_local, ty).spanned(span),
                    // Jump to the block that will assign the result
                    Node::Jump(block_id).spanned(span),
                ]);
                bctx.store_label(self, scope, block_id, cont_id, Some(result_local));
                self.lower_scope(
                    ctx.blocks,
                    ctx.bctx,
                    scope,
                    block_id,
                    |slf, value, ctx| {
                        let expr = slf.lower_expr(ctx, value);
                        // Assign the result to the temporary local
                        Node::Store(result_local, Box::new(expr))
                    },
                    Some(Node::Jump(cont_id).spanned(span)),
                );
                // Modify the current buffer to the continuation scope
                ctx.move_to(cont_id);
                Expr::Local(result_local)
            }
            // When lowering an if-statement without an else branch,
            // jump to the if-true-block if the condition is true then the continuation block,
            // otherwise jump to the continuation block immediately.
            // Since all non-diverging if-expressions return `void`, this will also return `void`.
            HirExpr::If(cond, then, None) => {
                let then_id = BlockId::from(then);
                let cont_id = BlockId(format!("_bb{}.after", then.0).into());

                let cond = self.lower_expr(ctx, *cond);
                ctx.current
                    .push(Node::Branch(cond, then_id, cont_id).spanned(span));
                bctx.store_label(self, then, then_id, cont_id, None);
                self.lower_scope(
                    ctx.blocks,
                    ctx.bctx,
                    then,
                    then_id,
                    |slf, value, ctx| Node::Expr(slf.lower_expr(ctx, value)),
                    Some(Node::Jump(cont_id).spanned(span)),
                );

                ctx.move_to(cont_id);
                Expr::Constant(Constant::Void)
            }
            // Since if-else-expressions may diverge, this will also store a temporary result local
            HirExpr::If(cond, then, Some(els)) => {
                let then_id = BlockId::from(then);
                let else_id = BlockId::from(els);
                let cont_id = BlockId(format!("_bb{}.after", then.0).into());
                // Create the temporary result local
                let result_local =
                    LocalId(format!("result{}", then_id.0).into(), LocalEnv::Internal);

                let cond = self.lower_expr(ctx, *cond);
                ctx.current.extend([
                    Node::Local(result_local, ty).spanned(span),
                    Node::Branch(cond, then_id, else_id).spanned(span),
                ]);
                bctx.store_label(self, then, then_id, cont_id, Some(result_local));
                bctx.store_label(self, els, else_id, cont_id, Some(result_local));

                let implicit_return = |slf: &mut Self, value: Spanned<_>, ctx: &mut Ctx| {
                    let expr = slf.lower_expr(ctx, value);
                    // Assign the result to the temporary local
                    Node::Store(result_local, Box::new(expr))
                };
                let exit_node = Some(Node::Jump(cont_id).spanned(span));
                self.lower_scope(
                    ctx.blocks,
                    ctx.bctx,
                    then,
                    then_id,
                    implicit_return,
                    exit_node.clone(),
                );
                self.lower_scope(
                    ctx.blocks,
                    ctx.bctx,
                    els,
                    else_id,
                    implicit_return,
                    exit_node,
                );

                // Modify the current buffer to the continuation scope
                ctx.move_to(cont_id);
                Expr::Local(result_local)
            }
            // Loops could break with a value, so they also need a temporary result local
            HirExpr::Loop(body) => {
                let body_id = BlockId::from(body);
                let cont_id = BlockId(format!("_bb{}.after", body.0).into());
                // Create the temporary result local
                let result_local =
                    LocalId(format!("result{}", body_id.0).into(), LocalEnv::Internal);

                ctx.current.extend([
                    Node::Local(result_local, ty).spanned(span),
                    Node::Jump(body_id).spanned(span),
                ]);
                bctx.store_label(self, body, body_id, cont_id, Some(result_local));
                bctx.continue_to = Some(body_id);
                bctx.break_to = Some((cont_id, result_local));
                self.lower_scope(
                    ctx.blocks,
                    ctx.bctx,
                    body,
                    body_id,
                    |slf, value, ctx| Node::Expr(slf.lower_expr(ctx, value)),
                    // Loop back to the start of the loop
                    Some(Node::Jump(body_id).spanned(span)),
                );

                // Modify the current buffer to the continuation scope
                ctx.move_to(cont_id);
                Expr::Local(result_local)
            }
            _ => todo!(),
        }
        .spanned(span)
    }

    /// Lowers the scope and returns the entrypoint block.
    pub fn lower_scope<'a, F>(
        &'a mut self,
        blocks: *mut BlockMap,
        bctx: *mut BlockCtx,
        scope: ScopeId,
        entry_id: BlockId,
        mut on_implicit_return: F,
        exit_node: Option<Spanned<Node>>,
    ) where
        F: FnMut(&mut Self, Spanned<TypedExpr>, &mut Ctx) -> Node + 'a,
    {
        let scope = self.thir.scopes.remove(&scope).expect("no such scope");

        // SAFETY: the blocks pointer is valid for the lifetime of the function
        let entry = unsafe { &mut *blocks }
            .try_insert(entry_id, Vec::with_capacity(scope.children.value().len()))
            .expect("block already exists");

        let mut ctx = Ctx {
            blocks,
            current: entry,
            track: entry_id,
            bctx,
        };
        let bctx = unsafe { &*bctx };

        for Spanned(node, span) in scope.children.into_value() {
            let node = match node {
                hir::Node::Expr(expr) => Node::Expr(self.lower_expr(&mut ctx, expr)),
                hir::Node::ImplicitReturn(value) => on_implicit_return(self, value, &mut ctx),
                hir::Node::Return(value) => {
                    let expr = value.map(|value| self.lower_expr(&mut ctx, value));
                    Node::Return(expr)
                }
                hir::Node::Continue(Some(Spanned(label, _))) => {
                    Node::Jump(*bctx.label_continue_map.get(&label).unwrap())
                }
                hir::Node::Continue(None) => Node::Jump(bctx.continue_to.unwrap()),
                hir::Node::Break(label, value) => {
                    let (block, local) = match label {
                        Some(Spanned(label, _)) => *bctx.label_break_map.get(&label).unwrap(),
                        None => {
                            let (block, local) = bctx.break_to.unwrap();
                            (block, Some(local))
                        }
                    };
                    if let (Some(value), Some(local)) = (value, local) {
                        let expr = self.lower_expr(&mut ctx, value);
                        ctx.current
                            .push(expr.nest(|expr| Node::Store(local, Box::new(expr))))
                    }
                    Node::Jump(block)
                }
                _ => todo!(),
            };
            ctx.current.push(Spanned(node, span));
        }
        ctx.current.extend(exit_node);
    }

    pub fn lower_map(&mut self, scope_id: ScopeId) -> BlockMap {
        let mut blocks = BlockMap::default();
        let entry_id = BlockId("entry".into());

        let mut lcm = HashMap::new();
        let mut lbm = HashMap::new();
        let mut bctx = BlockCtx {
            continue_to: None,
            break_to: None,
            label_continue_map: &mut lcm,
            label_break_map: &mut lbm,
        };
        self.lower_scope(
            &mut blocks,
            &mut bctx,
            scope_id,
            entry_id,
            |slf, value, ctx| {
                let expr = slf.lower_expr(ctx, value);
                Node::Return(Some(expr))
            },
            None,
        );
        blocks
    }

    pub fn lower_module(&mut self, module: ModuleId) {
        let scope_id = self.thir.modules.get(&module).expect("no such module");
        let blocks = self.lower_map(*scope_id);
        let name = ItemId(module, "root".into());
        self.mir.functions.insert(
            name,
            Func {
                name,
                params: Vec::new(),
                ret_ty: Ty::VOID,
                blocks,
            },
        );
    }
}
