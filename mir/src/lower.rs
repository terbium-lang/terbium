//! Lowers a typed HIR to MIR.

use crate::{
    BlockId, BlockMap, BoolIntrinsic, Constant, Expr, Func, HirFunc, IntIntrinsic, LocalId, Mir,
    Node, TypedHir,
};
use common::span::{Spanned, SpannedExt};
use hir::{
    infer::flatten_param,
    typed::{self, LocalEnv, Ty, TypedExpr},
    FloatWidth, Ident, IntSign, IntWidth, ItemId, ItemKind, Literal, LookupId, ModuleId, Pattern,
    PrimitiveTy, ScopeId,
};
use std::collections::HashMap;

#[derive(Clone, Debug)]
#[must_use = "the lowerer must be called to produce the MIR"]
pub struct Lowerer {
    /// The typed HIR to lower.
    pub thir: TypedHir,
    /// The MIR being constructed.
    pub mir: Mir,
    /// Non-fatal errors that occurred during lowering.
    pub errors: Vec<hir::Error>,
    /// Cache of how to flatten calls to functions.
    func_pats: HashMap<LookupId, Vec<Spanned<Pattern>>>,
    /// Generic type parameters for functions, used for monomorphization.
    func_ty_params: HashMap<LookupId, Vec<Ident>>,
    /// Monomorphized function lookup IDs.
    monomorphized: HashMap<(LookupId, Vec<String>), LookupId>,
    next_lookup_id: usize,
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
        self.current = unsafe { &mut *self.blocks }.entry(block_id).or_default();
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
        let Some(Spanned(label, _)) = lowerer.thir.scopes[&scope_id].label else {
            return;
        };

        self.label_continue_map.insert(label, block);
        self.label_break_map.insert(label, (cont, result));
    }
}

impl Lowerer {
    pub fn from_thir(thir: TypedHir) -> Self {
        let next_lookup_id = thir
            .funcs
            .keys()
            .map(|id| id.0)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        let func_ty_params = thir
            .funcs
            .iter()
            .map(|(id, func)| {
                (
                    *id,
                    func.header
                        .ty_params
                        .iter()
                        .map(|param| param.name)
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        Self {
            thir,
            mir: Mir::default(),
            errors: Vec::new(),
            func_pats: HashMap::new(),
            func_ty_params,
            monomorphized: HashMap::new(),
            next_lookup_id,
        }
    }

    fn next_mono_id(&mut self) -> LookupId {
        let id = self.next_lookup_id;
        self.next_lookup_id = self.next_lookup_id.saturating_add(1);
        LookupId(id)
    }

    fn mangle_ty(&self, ty: &Ty) -> String {
        ty.to_string()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect()
    }

    fn mangle_item(&self, item: ItemId, ty_args: &[Ty]) -> ItemId {
        let suffix = ty_args
            .iter()
            .map(|ty| self.mangle_ty(ty))
            .collect::<Vec<_>>()
            .join("_");
        let name = format!("{}__{}", item.1, suffix);
        ItemId(item.0, name.as_str().into())
    }

    fn monomorphize_func(&mut self, func: LookupId, ty_args: &[Ty]) -> LookupId {
        if ty_args.is_empty() {
            return func;
        }
        let key = (
            func,
            ty_args.iter().map(ToString::to_string).collect::<Vec<_>>(),
        );
        if let Some(existing) = self.monomorphized.get(&key) {
            return *existing;
        }
        let Some(param_names) = self.func_ty_params.get(&func) else {
            return func;
        };
        if param_names.len() != ty_args.len() {
            return func;
        }
        let subs = param_names
            .iter()
            .cloned()
            .zip(ty_args.iter().cloned())
            .collect::<HashMap<_, _>>();

        let Some(mut cloned) = self.mir.functions.get(&func).cloned() else {
            return func;
        };

        cloned.name = self.mangle_item(cloned.name, ty_args);
        cloned.params = cloned
            .params
            .into_iter()
            .map(|(ident, ty)| (ident, ty.substitute_generics(&subs)))
            .collect();
        cloned.ret_ty = cloned.ret_ty.substitute_generics(&subs);

        if let Some(blocks) = &mut cloned.blocks {
            for block in blocks.values_mut() {
                for node in block.iter_mut() {
                    match node.value_mut() {
                        Node::Local(_, ty) | Node::Register(_, _, ty) => {
                            *ty = ty.substitute_generics(&subs);
                        }
                        _ => {}
                    }
                }
            }
        }

        let new_id = self.next_mono_id();
        self.mir.functions.insert(new_id, cloned);
        if let Some(pats) = self.func_pats.get(&func).cloned() {
            self.func_pats.insert(new_id, pats);
        }
        self.monomorphized.insert(key, new_id);
        new_id
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
            (Literal::Float(bits), ty) => {
                let width = match ty {
                    Ty::Primitive(PrimitiveTy::Float(width)) => *width,
                    _ => FloatWidth::Float64,
                };
                let bits = match width {
                    FloatWidth::Float32 => (f64::from_bits(bits) as f32).to_bits() as u64,
                    _ => bits,
                };
                Constant::Float(bits, width)
            }
            (Literal::Bool(b), _) => Constant::Bool(b),
            (Literal::Char(c), _) => Constant::Char(c),
            (Literal::Void, _) => Constant::Void,
            (Literal::String(s), P(PrimitiveTy::String)) => Constant::String(s),
            (Literal::String(s), _) => {
                // If not explicitly typed as string, treat as [uint8]
                Constant::Slice(
                    s.bytes()
                        .map(|b| Constant::Int(b as _, IntSign::Unsigned, IntWidth::Int8))
                        .collect(),
                )
            }
            (Literal::Bytes(bytes), _) => Constant::Slice(
                bytes
                    .into_iter()
                    .map(|b| Constant::Int(b as _, IntSign::Unsigned, IntWidth::Int8))
                    .collect(),
            ),
            _ => unimplemented!(),
        }
    }

    pub fn lower_expr(
        &mut self,
        ctx: &mut Ctx,
        Spanned(TypedExpr(expr, ty), span): Spanned<TypedExpr>,
    ) -> Spanned<Expr> {
        type HirExpr = typed::Expr;

        // SAFETY: the context is valid for the duration of the function
        let bctx = unsafe { &mut *ctx.bctx };
        match expr {
            HirExpr::Literal(lit) => Expr::Constant(self.lower_literal(lit, &ty)),
            HirExpr::Func(_, ty_args, func) => {
                let func = ty_args
                    .as_ref()
                    .map(|args| self.monomorphize_func(func, args.value()))
                    .unwrap_or(func);
                Expr::FuncRef(func)
            }
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
                let result_local =
                    LocalId(format!("result.{}", scope.0).into(), LocalEnv::Internal);
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
                    move |slf, value, ctx| {
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
                let then_id = BlockId(format!("if.{}.then", then.0).into());
                let else_id = BlockId(format!("if.{}.else", then.0).into());
                let cont_id = BlockId(format!("if.{}.after", then.0).into());
                // Create the temporary result local
                let result_local = LocalId(format!("result.{}", then.0).into(), LocalEnv::Internal);

                let cond = self.lower_expr(ctx, *cond);
                ctx.current.extend([
                    Node::Local(result_local, ty).spanned(span),
                    Node::Branch(cond, then_id, else_id).spanned(span),
                ]);
                bctx.store_label(self, then, then_id, cont_id, Some(result_local));
                bctx.store_label(self, els, else_id, cont_id, Some(result_local));

                let implicit_return = move |slf: &mut Self, value: Spanned<_>, ctx: &mut Ctx| {
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
                let result_local = LocalId(format!("result.{}", body.0).into(), LocalEnv::Internal);

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
            HirExpr::CallFunc {
                func,
                args,
                ty_args,
                ..
            } => {
                let func = ty_args
                    .as_ref()
                    .map(|args| self.monomorphize_func(func, args))
                    .unwrap_or(func);
                let params = &self.func_pats[&func];
                let mut flattened = Vec::new();
                for (pat, arg) in params.iter().zip(args) {
                    apply_arg(pat, arg, &mut flattened).unwrap();
                }

                let args = flattened
                    .into_iter()
                    .map(|arg| self.lower_expr(ctx, arg))
                    .collect();
                Expr::Call(func, args)
            }
            HirExpr::CallIndirect(callee, args) => {
                let callee_ty = callee.value().1.clone();
                let callee = Box::new(self.lower_expr(ctx, *callee));
                let args = args
                    .into_iter()
                    .map(|arg| self.lower_expr(ctx, arg))
                    .collect();
                Expr::CallIndirect(callee, args, callee_ty)
            }
            // Basic implementation for any other expression types
            // This allows the rest of the code to run without implemented cases blocking it
            _ => Expr::Constant(Constant::Void),
        }
        .spanned(span)
    }

    /// Lowers the scope and returns the entrypoint block.
    pub fn lower_scope<'a, F>(
        &mut self,
        blocks: *mut BlockMap,
        bctx: *mut BlockCtx<'a>,
        scope: ScopeId,
        entry_id: BlockId,
        mut on_implicit_return: F,
        exit_node: Option<Spanned<Node>>,
    ) -> Spanned<&'a mut Vec<Spanned<Node>>>
    where
        F: FnMut(&mut Self, Spanned<TypedExpr>, &mut Ctx) -> Node + 'a,
    {
        let scope = self.thir.scopes.remove(&scope).expect("no such scope");

        // First, lower all static items in the scope
        let mut funcs = Vec::new();
        for ((kind, item), id) in scope.items {
            match kind {
                ItemKind::Func => {
                    let func = self.thir.funcs.remove(&id).expect("no such func");
                    self.func_pats.insert(
                        id,
                        func.header
                            .params
                            .iter()
                            .chain(func.header.kw_params.iter())
                            .map(|p| &p.pat)
                            .cloned()
                            .collect(),
                    );
                    funcs.push((id, item, func));
                }
                _ => continue, // TODO
            }
        }
        for (id, item, func) in funcs {
            let func = self.lower_func(item, func);
            self.mir.functions.insert(id, func);
        }

        // SAFETY: the blocks pointer is valid for the lifetime of the function
        let entry = unsafe { &mut *blocks }
            .entry(entry_id)
            .or_insert_with(|| Vec::with_capacity(scope.children.value().len()));

        let mut ctx = Ctx {
            blocks,
            current: entry,
            track: entry_id,
            bctx,
        };
        let bctx = unsafe { &*bctx };

        let Spanned(children, span) = scope.children;
        for Spanned(node, span) in children {
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
                // in this case we can use registers
                hir::Node::Let {
                    pat:
                        Spanned(
                            Pattern::Ident {
                                ident,
                                mut_kw: None,
                            },
                            _,
                        ),
                    ty,
                    value: Some(value),
                    ..
                } => {
                    let value = self.lower_expr(&mut ctx, value);
                    let local = LocalId(*ident.value(), LocalEnv::Standard);
                    Node::Register(local, value, ty)
                }
                _ => todo!(),
            };
            ctx.current.push(Spanned(node, span));
        }
        ctx.current.extend(exit_node);
        ctx.current.spanned(span)
    }

    pub fn lower_map(&mut self, scope_id: ScopeId, terminate: bool) -> BlockMap {
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
        let Spanned(current, span) = self.lower_scope(
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
        // Does it terminate?
        if terminate
            && !current
                .last()
                .map(|node| node.value().is_terminator())
                .unwrap_or(false)
        {
            // If it doesn't, we need to add a return to void
            current.push(Node::Return(None).spanned(span));
        }
        // Cleanup input by removing all unreachable terminators in each block
        for block in blocks.values_mut() {
            let first_term = block.iter().position(|node| node.value().is_terminator());
            if let Some(first_term) = first_term {
                block.truncate(first_term + 1);
            }
        }
        blocks
    }

    pub fn lower_func(
        &mut self,
        item_id: ItemId,
        HirFunc {
            header, body, kind, ..
        }: HirFunc,
    ) -> Func {
        let mut params = Vec::new();
        for param in header.params.iter().chain(header.kw_params.iter()) {
            if let Err(why) = flatten_param(&param.pat, param.ty.clone(), &mut params) {
                self.errors.push(why);
            }
        }
        Func {
            name: item_id,
            params: params.into_iter().map(|(i, b)| (i, b.ty)).collect(),
            ret_ty: header.ret_ty,
            kind,
            blocks: body.map(|body| self.lower_map(body, true)),
        }
    }

    pub fn lower_module(&mut self, module: ModuleId) {
        let scope_id = self.thir.modules.get(&module).expect("no such module");
        let blocks = self.lower_map(*scope_id, true);

        self.mir.functions.insert(
            LookupId(usize::MAX),
            Func {
                name: ItemId(module, "__root".into()),
                params: Vec::new(),
                ret_ty: Ty::VOID,
                kind: hir::FuncKind::Normal,
                blocks: Some(blocks),
            },
        );
    }
}

pub fn apply_arg(
    pat: &Spanned<Pattern>,
    arg: Spanned<TypedExpr>,
    flattened: &mut Vec<Spanned<TypedExpr>>,
) -> hir::error::Result<()> {
    // NOTE: argument count and types have been checked at this point
    match pat.value() {
        Pattern::Ident { .. } => flattened.push(arg),
        Pattern::Tuple(pats) => {
            match arg.value().0 {
                // if we can destruct the tuple, do so
                typed::Expr::Tuple(ref exprs) => {
                    debug_assert_eq!(pats.len(), exprs.len());
                    for (pat, expr) in pats.iter().zip(exprs) {
                        apply_arg(pat, expr.clone(), flattened)?;
                    }
                }
                _ => {
                    let Ty::Tuple(tys) = arg.value().1.clone() else {
                        unreachable!();
                    };
                    for (i, (pat, ty)) in pats.iter().zip(tys).enumerate() {
                        let Spanned(arg, span) = arg.clone();
                        let expr = Spanned(
                            TypedExpr(
                                typed::Expr::GetField(
                                    Box::new(arg.spanned(span)),
                                    Ident::from(i.to_string()).spanned(span),
                                ),
                                ty,
                            ),
                            span,
                        );
                        apply_arg(pat, expr, flattened)?;
                    }
                }
            }
        }
    }

    Ok(())
}
