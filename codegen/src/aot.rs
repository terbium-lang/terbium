use common::span::Spanned;
use inkwell::{
    attributes::Attribute,
    basic_block::BasicBlock,
    builder::Builder,
    context::Context,
    module::Module,
    passes::PassManager,
    types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum, IntType, StringRadix},
    values::{AnyValue, BasicValue, BasicValueEnum, FunctionValue, PointerValue},
    AddressSpace, IntPredicate,
};
use mir::{
    BlockId, Constant, Expr, Func, Ident, IntIntrinsic, IntSign, IntWidth, LocalEnv, LocalId,
    LookupId, Node, PrimitiveTy, Ty, UnaryIntIntrinsic,
};
use std::{collections::HashMap, mem::MaybeUninit, ops::Not};

#[derive(Copy, Clone)]
struct Local<'ctx> {
    value: PointerValue<'ctx>,
    ty: BasicTypeEnum<'ctx>,
}

/// Compiles a function.
pub struct Compiler<'a, 'ctx> {
    pub context: &'ctx Context,
    pub builder: &'a Builder<'ctx>,
    pub fpm: &'a PassManager<FunctionValue<'ctx>>,
    pub module: &'a Module<'ctx>,

    lowering: MaybeUninit<Func>,
    fn_value: MaybeUninit<FunctionValue<'ctx>>,
    functions: HashMap<LookupId, FunctionValue<'ctx>>,
    locals: HashMap<LocalId, Option<Local<'ctx>>>,
    blocks: HashMap<BlockId, BasicBlock<'ctx>>,
    increment: usize,
}

const VOID: Expr = Expr::Constant(Constant::Void);

impl<'a, 'ctx> Compiler<'a, 'ctx> {
    #[inline]
    fn get_int_type(&self, width: IntWidth) -> IntType<'ctx> {
        self.context.custom_width_int_type(width as usize as _)
    }

    #[inline]
    fn lowering_mut(&mut self) -> &mut Func {
        unsafe { self.lowering.assume_init_mut() }
    }

    #[inline]
    const fn fn_value(&self) -> FunctionValue<'ctx> {
        unsafe { self.fn_value.assume_init() }
    }

    /// Lowers a constant value.
    pub fn lower_constant(&mut self, constant: Constant) -> BasicValueEnum<'ctx> {
        match constant {
            Constant::Int(i, sign, width) if width as usize <= 64 => BasicValueEnum::IntValue(
                self.get_int_type(width as _)
                    .const_int(i as u64, sign.is_signed()),
            ),
            Constant::Int(i, sign, width) => {
                let string = match sign.is_signed() {
                    true => (i as i128).to_string(),
                    false => i.to_string(),
                };
                // TODO: we can use the from_arbitrary_precision method instead of converting to string which is much slower
                BasicValueEnum::IntValue(
                    self.get_int_type(width)
                        .const_int_from_string(&string, StringRadix::Decimal)
                        .unwrap(),
                )
            }
            Constant::Bool(b) => {
                BasicValueEnum::IntValue(self.context.bool_type().const_int(b as u64, false))
            }
            Constant::Void => BasicValueEnum::PointerValue(
                self.context.ptr_type(AddressSpace::default()).const_null(),
            ),
            Constant::String(s) => {
                // Create a global string constant
                let string_type = self.context.i8_type().array_type(s.len() as u32);
                let string_value = self.context.const_string(s.as_bytes(), true);
                let global = self
                    .module
                    .add_global(string_type, None, &self.next_increment());
                global.set_initializer(&string_value);
                BasicValueEnum::PointerValue(global.as_pointer_value())
            }
            Constant::Slice(elements) => {
                // Create a global array constant
                let element_type = self.context.i8_type();
                let array_type = element_type.array_type(elements.len() as u32);
                let values = elements
                    .iter()
                    .map(|element| match element {
                        Constant::Int(i, sign, width) if *width as usize <= 64 => {
                            element_type.const_int(*i as u64, sign.is_signed())
                        }
                        Constant::Char(c) => element_type.const_int(*c as u64, false),
                        Constant::Bool(b) => element_type.const_int(*b as u64, false),
                        _ => element_type.const_zero(),
                    })
                    .collect::<Vec<_>>();
                let array_value = element_type.const_array(&values);
                let global = self
                    .module
                    .add_global(array_type, None, &self.next_increment());
                global.set_initializer(&array_value);
                BasicValueEnum::PointerValue(global.as_pointer_value())
            }
            _ => todo!(),
        }
    }

    /// Get the next temporary local name increment.
    #[inline]
    pub fn next_increment(&mut self) -> String {
        let increment = self.increment;
        self.increment += 1;
        format!("tmp.{increment}")
    }

    /// Lowers an integer intrinsic
    pub fn lower_int_intrinsic(
        &mut self,
        intr: IntIntrinsic,
        sign: IntSign,
    ) -> Option<BasicValueEnum<'ctx>> {
        Some(match intr {
            IntIntrinsic::Unary(op, val) => {
                let next = self.next_increment();
                let val = self.lower_expr(*val)?.into_int_value();
                BasicValueEnum::IntValue(match op {
                    UnaryIntIntrinsic::Neg => self.builder.build_int_neg(val, &next).ok()?,
                    UnaryIntIntrinsic::BitNot => self.builder.build_not(val, &next).ok()?,
                })
            }
            IntIntrinsic::Binary(op, lhs, rhs) => {
                use mir::BinaryIntIntrinsic as B;

                let lhs = self.lower_expr(*lhs)?.into_int_value();
                let rhs = self.lower_expr(*rhs)?.into_int_value();
                let next = self.next_increment();
                let int_value = match op {
                    B::Add => self.builder.build_int_add(lhs, rhs, &next),
                    B::Sub => self.builder.build_int_sub(lhs, rhs, &next),
                    B::Mul => self.builder.build_int_mul(lhs, rhs, &next),
                    B::Div if sign.is_signed() => {
                        self.builder.build_int_signed_div(lhs, rhs, &next)
                    }
                    B::Div => self.builder.build_int_unsigned_div(lhs, rhs, &next),
                    B::Mod if sign.is_signed() => {
                        self.builder.build_int_signed_rem(lhs, rhs, &next)
                    }
                    B::Mod => self.builder.build_int_unsigned_rem(lhs, rhs, &next),
                    b if b.is_cmp() => {
                        let cmp = match (b, sign) {
                            (B::Eq, _) => IntPredicate::EQ,
                            (B::Ne, _) => IntPredicate::NE,
                            (B::Lt, IntSign::Signed) => IntPredicate::SLT,
                            (B::Lt, IntSign::Unsigned) => IntPredicate::ULT,
                            (B::Le, IntSign::Signed) => IntPredicate::SLE,
                            (B::Le, IntSign::Unsigned) => IntPredicate::ULE,
                            (B::Gt, IntSign::Signed) => IntPredicate::SGT,
                            (B::Gt, IntSign::Unsigned) => IntPredicate::UGT,
                            (B::Ge, IntSign::Signed) => IntPredicate::SGE,
                            (B::Ge, IntSign::Unsigned) => IntPredicate::UGE,
                            _ => unreachable!(),
                        };
                        self.builder.build_int_compare(cmp, lhs, rhs, &next)
                    }
                    B::BitOr => self.builder.build_or(lhs, rhs, &next),
                    B::BitAnd => self.builder.build_and(lhs, rhs, &next),
                    B::BitXor => self.builder.build_xor(lhs, rhs, &next),
                    B::Shl => self.builder.build_left_shift(lhs, rhs, &next),
                    B::Shr if sign.is_signed() => {
                        self.builder.build_right_shift(lhs, rhs, true, &next)
                    }
                    B::Shr => self.builder.build_right_shift(lhs, rhs, false, &next),
                    _ => unreachable!(),
                };
                BasicValueEnum::IntValue(int_value.ok()?)
            }
        })
    }

    /// Lowers an expression.
    pub fn lower_expr(&mut self, expr: Spanned<Expr>) -> Option<BasicValueEnum<'ctx>> {
        Some(match expr.into_value() {
            Expr::Local(id) => {
                let ptr = self.locals[&id]?;
                self.builder
                    .build_load(ptr.ty, ptr.value, &self.next_increment())
                    .ok()?
            }
            Expr::Constant(c) => self.lower_constant(c),
            Expr::IntIntrinsic(intr, sign, _) => self.lower_int_intrinsic(intr, sign)?,
            Expr::BoolIntrinsic(intr) => {
                use mir::BoolIntrinsic as B;

                macro_rules! lower {
                    ($($tgt:ident),+ => $e:expr) => {{
                        $(let $tgt = self.lower_expr(*$tgt)?.into_int_value();)+
                        $e
                    }}
                }

                let next = self.next_increment();
                let bool_value = match intr {
                    B::And(lhs, rhs) => lower!(lhs, rhs => self.builder.build_and(lhs, rhs, &next)),
                    B::Or(lhs, rhs) => lower!(lhs, rhs => self.builder.build_or(lhs, rhs, &next)),
                    B::Xor(lhs, rhs) => lower!(lhs, rhs => self.builder.build_xor(lhs, rhs, &next)),
                    B::Not(val) => lower!(val => self.builder.build_not(val, &next)),
                };
                BasicValueEnum::IntValue(bool_value.ok()?)
            }
            Expr::FuncRef(id) => {
                let func = self.functions.get(&id)?;
                BasicValueEnum::PointerValue(func.as_global_value().as_pointer_value())
            }
            Expr::Call(func, args) => {
                let args = args
                    .into_iter()
                    .map(|arg| self.lower_expr(arg).unwrap().into())
                    .collect::<Vec<_>>();

                self.builder
                    .build_call(self.functions[&func], &args, &self.next_increment())
                    .ok()?
                    .try_as_basic_value()
                    .left()
                    .unwrap()
            }
            Expr::CallIndirect(callee, args, callee_ty) => {
                let callee = self.lower_expr(*callee)?.into_pointer_value();
                let args = args
                    .into_iter()
                    .map(|arg| self.lower_expr(arg).unwrap().into())
                    .collect::<Vec<_>>();
                let (params, ret) = match callee_ty {
                    Ty::Func(params, ret) => (params, ret),
                    _ => return None,
                };
                let param_tys = params
                    .iter()
                    .map(|ty| self.lower_ty(ty).into())
                    .collect::<Vec<BasicMetadataTypeEnum>>();
                let fn_ty = if ret.is_zst() {
                    self.context.void_type().fn_type(&param_tys, false)
                } else {
                    self.lower_ty(&ret).fn_type(&param_tys, false)
                };
                self.builder
                    .build_indirect_call(fn_ty, callee, &args, &self.next_increment())
                    .ok()?
                    .try_as_basic_value()
                    .left()
                    .unwrap()
            }
        })
    }

    /// Lowers a MIR node.
    pub fn lower_node(&mut self, node: Spanned<Node>) {
        match node.into_value() {
            Node::Store(loc, val) => {
                let Some(val) = self.lower_expr(*val) else {
                    return;
                };
                let loc = self.locals[&loc].as_ref().unwrap().value;
                self.builder.build_store(loc, val);
            }
            Node::Register(loc, expr, ty) => {
                let local = self.lower_expr(expr).map(|expr| {
                    let ty = self.lower_ty(&ty);
                    let ptr = self
                        .builder
                        .build_alloca(ty, &loc.name())
                        .expect("failed to create alloca");
                    self.builder.build_store(ptr, expr);

                    Local { value: ptr, ty }
                });
                self.locals.insert(loc, local);
            }
            Node::Local(id, ty) => {
                let local = ty.is_zst().not().then(|| {
                    let ty = self.lower_ty(&ty);
                    let ptr = self
                        .builder
                        .build_alloca(ty, &id.name())
                        .expect("failed to create alloca");
                    Local { value: ptr, ty }
                });
                self.locals.insert(id, local);
            }
            Node::Return(None | Some(Spanned(VOID, _))) => {
                self.builder.build_return(None);
            }
            Node::Return(Some(expr)) => {
                let expr = self.lower_expr(expr);
                self.builder
                    .build_return(expr.as_ref().map(|expr| expr as &dyn BasicValue));
            }
            Node::Jump(block) => {
                let block = *self.blocks.get(&block).unwrap();
                self.builder.build_unconditional_branch(block);
            }
            Node::Branch(cond, then_block, else_block) => {
                let cond = self.lower_expr(cond).unwrap().into_int_value();
                let then_block = *self.blocks.get(&then_block).unwrap();
                let else_block = *self.blocks.get(&else_block).unwrap();
                self.builder
                    .build_conditional_branch(cond, then_block, else_block);
            }
            Node::Expr(expr) => {
                self.lower_expr(expr);
            }
        }
    }

    /// Lowers a block given its ID.
    pub fn lower_block(&mut self, block_id: BlockId) {
        let blocks = self.lowering_mut().blocks.as_mut().expect("missing blocks");
        let block = blocks.remove(&block_id).unwrap();
        self.builder
            .position_at_end(*self.blocks.get(&block_id).unwrap());

        for node in block {
            self.lower_node(node)
        }
    }

    /// Lowers a THIR type value.
    ///
    /// TODO: THIR type value should be monomorphized in the THIR->MIR lowering stage.
    pub fn lower_ty(&self, ty: &Ty) -> BasicTypeEnum<'ctx> {
        use Ty::Primitive as P;

        match ty {
            P(PrimitiveTy::Int(_, width)) => BasicTypeEnum::IntType(self.get_int_type(*width)),
            P(PrimitiveTy::Bool) => BasicTypeEnum::IntType(self.context.bool_type()),
            P(PrimitiveTy::String) => {
                // String is represented as a pointer to a null-terminated array of i8
                BasicTypeEnum::PointerType(self.context.ptr_type(AddressSpace::default()))
            }
            Ty::Func(params, ret) => {
                let param_tys = params
                    .iter()
                    .map(|ty| self.lower_ty(ty).into())
                    .collect::<Vec<BasicMetadataTypeEnum>>();
                let fn_ty = if ret.is_zst() {
                    self.context.void_type().fn_type(&param_tys, false)
                } else {
                    self.lower_ty(ret).fn_type(&param_tys, false)
                };
                BasicTypeEnum::PointerType(fn_ty.ptr_type(AddressSpace::default()))
            }
            // Default to a generic pointer type for any other primitive types
            _ => BasicTypeEnum::PointerType(self.context.ptr_type(AddressSpace::default())),
        }
    }

    /// Creates a new stack allocation instruction in the entry block of the function.
    fn create_entry_block_alloca(
        &self,
        name: &str,
        ty: impl BasicType<'ctx>,
    ) -> PointerValue<'ctx> {
        let builder = self.context.create_builder();
        let entry = self.fn_value().get_first_basic_block().unwrap();

        match entry.get_first_instruction() {
            Some(first_instr) => builder.position_before(&first_instr),
            None => builder.position_at_end(entry),
        }

        builder
            .build_alloca(ty, name)
            .expect("failed to create alloca")
    }

    #[inline]
    fn get_simple_attribute(&self, attr: &str) -> Attribute {
        self.context
            .create_enum_attribute(Attribute::get_named_enum_kind_id(attr), 0)
    }

    /// Registers the specified function into an LLVM `FunctionValue`.
    fn register_fn(&mut self, id: LookupId, func: &Func) -> Vec<Ident> {
        let (names, param_tys) = func
            .params
            .iter()
            .filter_map(|(name, ty)| {
                ty.is_zst()
                    .not()
                    .then(|| (*name, BasicMetadataTypeEnum::from(self.lower_ty(ty))))
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();

        let fn_ty = match func.ret_ty.is_zst() {
            true => self.context.void_type().fn_type(&param_tys, false),
            false => self.lower_ty(&func.ret_ty).fn_type(&param_tys, false),
        };

        // TODO: qualified name
        let name = func.name.to_string();
        let fn_value = self.module.add_function(&name, fn_ty, None);
        self.functions.insert(id, fn_value);
        names
    }

    /// Compiles the body of the given function.
    fn compile_fn(&mut self, fn_value: FunctionValue<'ctx>, func: Func, names: Vec<Ident>) {
        let mut func = func;
        let blocks = func.blocks.take().expect("missing function body");
        let block_ids = blocks.keys().copied().collect::<Vec<_>>();
        self.lowering = MaybeUninit::new(func);
        self.fn_value.write(fn_value);
        self.locals.clear();
        self.blocks.clear();
        self.increment = 0;

        // Create blocks
        for id in &block_ids {
            let bb = self
                .context
                .append_basic_block(self.fn_value(), &id.to_string());
            self.blocks.insert(*id, bb);
        }
        self.builder.position_at_end(self.blocks[&BlockId::entry()]);

        for (param, ident) in self.fn_value().get_param_iter().zip(names) {
            let name = ident.to_string();
            param.set_name(&name);

            let alloca = self.create_entry_block_alloca(&name, param.get_type());
            self.builder.build_store(alloca, param);

            // Populate locals
            let local = Local {
                value: alloca,
                ty: param.get_type(),
            };
            self.locals
                .insert(LocalId(ident, LocalEnv::Standard), Some(local));
        }

        // Compile body
        self.lowering_mut().blocks = Some(blocks);
        block_ids.into_iter().for_each(|id| self.lower_block(id));
        self.fn_value().print_to_string();
        unsafe { self.lowering.assume_init_drop() };

        // Verify and run optimizations
        if self.fn_value().verify(true) {
            self.fpm.run_on(&self.fn_value());
        } else {
            unsafe {
                self.fn_value().delete();
            }
        }
    }

    fn get_fputs(&self) -> FunctionValue<'ctx> {
        if let Some(func) = self.module.get_function("fputs") {
            return func;
        }
        let i8_ptr = self.context.ptr_type(AddressSpace::default());
        let i32 = self.context.i32_type();
        let ty = i32.fn_type(&[i8_ptr.into(), i8_ptr.into()], false);
        self.module.add_function("fputs", ty, None)
    }

    fn get_stdio_ptr(&self, name: &str) -> PointerValue<'ctx> {
        let i8_ptr = self.context.ptr_type(AddressSpace::default());
        let global = self
            .module
            .get_global(name)
            .unwrap_or_else(|| self.module.add_global(i8_ptr, None, name));
        global.as_pointer_value()
    }

    fn compile_internal_fn(&mut self, fn_value: FunctionValue<'ctx>, func: &Func) {
        let name = func.name.to_string();
        let (stream_name, param_name) = match name.as_str() {
            "core.intrinsics.write_stdout" => ("stdout", "s"),
            "core.intrinsics.write_stderr" => ("stderr", "s"),
            _ => return,
        };

        let entry = self.context.append_basic_block(fn_value, "entry");
        self.builder.position_at_end(entry);

        let param = fn_value.get_first_param().expect("missing arg");
        param.set_name(param_name);

        let fputs = self.get_fputs();
        let stream = self
            .builder
            .build_load(
                self.context.ptr_type(AddressSpace::default()),
                self.get_stdio_ptr(stream_name),
                "stream",
            )
            .expect("failed to load stdio stream")
            .into_pointer_value();

        self.builder
            .build_call(fputs, &[param.into(), stream.into()], "call");
        self.builder.build_return(None);

        if fn_value.verify(true) {
            self.fpm.run_on(&fn_value);
        } else {
            unsafe {
                fn_value.delete();
            }
        }
    }

    /// Compiles the specified `Function` in the given `Context` and using the specified `Builder`, `PassManager`, and `Module`.
    pub fn compile(
        context: &'ctx Context,
        builder: &'a Builder<'ctx>,
        pass_manager: &'a PassManager<FunctionValue<'ctx>>,
        module: &'a Module<'ctx>,
        functions: HashMap<LookupId, Func>,
    ) {
        let mut compiler = Self {
            context,
            builder,
            fpm: pass_manager,
            module,
            functions: HashMap::with_capacity(functions.len()),
            lowering: MaybeUninit::uninit(),
            fn_value: MaybeUninit::uninit(),
            locals: HashMap::new(),
            blocks: HashMap::new(),
            increment: 0,
        };

        let mut names = Vec::with_capacity(functions.len());
        for (id, func) in &functions {
            names.push(compiler.register_fn(*id, func));
        }
        for ((id, func), names) in functions.into_iter().zip(names) {
            if matches!(func.kind, mir::FuncKind::Internal(_)) {
                compiler.compile_internal_fn(compiler.functions[&id], &func);
                continue;
            }
            if func.blocks.is_none() {
                continue;
            };
            compiler.compile_fn(compiler.functions[&id], func, names);
        }
    }
}
