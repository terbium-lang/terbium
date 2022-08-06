#![feature(slice_take)]
#![feature(box_patterns)]

use terbium_grammar::{Body, Expr, Node, Operator, Spanned};

use inkwell::{
    builder::Builder,
    context::Context,
    FloatPredicate,
    IntPredicate,
    module::Module,
    passes::PassManager,
    types::{BasicType, VectorType, FloatMathType},
    values::{AnyValue, BasicValue, BasicValueEnum, FunctionValue, PointerValue},
};

#[derive(Debug)]
pub struct Compiler<'a, 'ctx> {
    pub ctx: &'ctx Context,
    pub builder: &'a Builder<'ctx>,
    pub module: &'a Module<'ctx>,
    pub fpm: &'a PassManager<FunctionValue<'ctx>>,
    fn_val: Option<FunctionValue<'ctx>>,
}

impl<'a, 'ctx> Compiler<'a, 'ctx> {
    const ENTRYPOINT_FN_NAME: &'static str = "__trb_internal_entrypoint";

    pub fn compile(
        ctx: &'ctx Context,
        builder: &'a Builder<'ctx>,
        fpm: &'a PassManager<FunctionValue<'ctx>>,
        module: &'a Module<'ctx>,
        body: Body,
    ) -> Result<FunctionValue<'ctx>, &'static str> {
        let mut compiler = Self {
            ctx,
            builder,
            fpm,
            module,
            fn_val: None,
        };

        compiler.prepare();
        compiler.compile_body_entrypoint(body)?;

        compiler.optimize().map(|_| compiler.fn_value())
    }

    #[inline]
    pub fn fn_value(&self) -> FunctionValue<'ctx> {
        self.fn_val.expect("expected fn_val.is_some()")
    }

    pub fn create_entry_allocator<T>(&self, name: &str, ty: T) -> PointerValue<'ctx>
    where
        T: BasicType<'ctx>
    {
        let builder = self.ctx.create_builder();
        let entry = self.fn_value().get_first_basic_block().unwrap();

        if let Some(first) = &entry.get_first_instruction() {
            builder.position_before(first);
        } else {
            builder.position_at_end(entry);
        }

        builder.build_alloca(ty, name)
    }

    pub fn eval_expr(&mut self, expr: &Spanned<Expr>) -> Result<BasicValueEnum<'ctx>, &'static str> {
        let (expr, span) = expr.node_span();

        Ok(match expr {
            Expr::Integer(i) => {
                // TODO: this wraps the u128, but a `long` type which is an i128 isp planned
                self.ctx.i64_type().const_int(*i as u64, false).as_basic_value_enum()
            }
            Expr::Float(f) => {
                self.ctx.f64_type().const_float_from_string(f).as_basic_value_enum()
            }
            Expr::String(s) => {
                self.ctx.const_string(s.as_bytes(), false).as_basic_value_enum()
            }
            Expr::Bool(b) => {
                if *b {
                    self.ctx.bool_type().const_all_ones().as_basic_value_enum()
                } else {
                    self.ctx.bool_type().const_zero().as_basic_value_enum()
                }
            }
            Expr::UnaryExpr { operator, value } => {
                let (op, span) = operator.node_span();
                let value = self.eval_expr(value)?;

                match op {
                    Operator::Add => match value {
                        BasicValueEnum::IntValue(i) => i.as_basic_value_enum(),
                        BasicValueEnum::FloatValue(f) => f.as_basic_value_enum(),
                        // Custom operator overloading
                        BasicValueEnum::StructValue(_s) => todo!(),
                        // Analyzer should've caught this
                        _ => unreachable!(),
                    }
                    Operator::Sub => match value {
                        BasicValueEnum::IntValue(i) => if i.is_const() {
                            i.const_neg().as_basic_value_enum()
                        } else {
                            self.builder.build_int_neg(i, "tmpneg")
                                .as_basic_value_enum()
                        },
                        BasicValueEnum::FloatValue(f) => if f.is_const() {
                            f.const_neg().as_basic_value_enum()
                        } else {
                            self.builder.build_float_neg(f, "tmpneg")
                                .as_basic_value_enum()
                        },
                        // Custom operator overloading
                        BasicValueEnum::StructValue(_s) => todo!(),
                        // Analyzer should've caught this
                        _ => unreachable!(),
                    }
                    Operator::Not => match value {
                        BasicValueEnum::IntValue(i) => if i.is_const() {
                            i.const_select(
                                self.ctx.bool_type().const_zero(),
                                self.ctx.bool_type().const_all_ones(),
                            )
                            .as_basic_value_enum()
                        } else {
                            self.builder.build_not(i, "tmpnot")
                                .as_basic_value_enum()
                        }
                        BasicValueEnum::FloatValue(f) => if f.is_const() {
                            f.const_compare(FloatPredicate::UEQ, self.ctx.f64_type().const_zero())
                                .const_select(
                                    self.ctx.bool_type().const_zero(),
                                    self.ctx.bool_type().const_all_ones(),
                                )
                                .as_basic_value_enum()
                        } else {
                            self.builder.build_float_compare(
                                FloatPredicate::UEQ,
                                f,
                                self.ctx.f64_type().const_zero(),
                                "tmpnot",
                            )
                            .as_basic_value_enum()
                        }
                        _ => todo!(),
                    }
                    Operator::BitNot => todo!(),
                    _ => unreachable!(),
                }
            }
            Expr::BinaryExpr { operator, lhs, rhs } => {
                let (operator, span) = operator.node_span();

                let left = self.eval_expr(lhs)?;
                let right = self.eval_expr(rhs)?;

                match operator {
                    Operator::Add => {
                        match (left, right) {
                            (BasicValueEnum::IntValue(lhs), BasicValueEnum::IntValue(rhs)) => {
                                self.builder.build_int_add(lhs, rhs, "tmpintadd")
                                    .as_basic_value_enum()
                            }
                            (BasicValueEnum::FloatValue(lhs), BasicValueEnum::FloatValue(rhs)) => {
                                self.builder.build_float_add(lhs, rhs, "tmpfloatadd")
                                    .as_basic_value_enum()
                            }
                            (BasicValueEnum::FloatValue(f), BasicValueEnum::IntValue(i))
                            | (BasicValueEnum::IntValue(i), BasicValueEnum::FloatValue(f)) => {
                                let int = self.builder.build_signed_int_to_float(i, self.ctx.f64_type(), "tmpintfloatconv");

                                self.builder.build_float_add(int, f, "tmpfloatadd")
                                    .as_basic_value_enum()
                            }
                            _ => todo!()
                        }
                    }
                    _ => todo!(),
                }
            }
            _ => todo!(),
        })
    }

    pub fn compile_node(&mut self, node: &Spanned<Node>) -> Result<(), &'static str> {
        let (node, span) = node.node_span();

        match node {
            Node::Module(nodes) => {
                for node in nodes {
                    self.compile_node(node)?;
                }
            }
            Node::Expr(e) => {
                self.eval_expr(e)?;
            },
            _ => todo!(),
        }

        Ok(())
    }

    pub fn compile_body_entrypoint(&mut self, Body(mut nodes, return_last): Body) -> Result<(), &'static str> {
        let e = if return_last {
            if let Some(spanned) = nodes.pop() {
                if let Node::Expr(last_expr) = spanned.into_node() {
                    Some(last_expr)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        for node in nodes {
            self.compile_node(&node)?;
        }

        if let Some(e) = e {
            let e = self.eval_expr(&e)?;

            self.builder.build_return(Some(match &e {
                BasicValueEnum::IntValue(i) => i,
                BasicValueEnum::FloatValue(f) => f,
                _ => todo!(),
            }));
        }
        Ok(())
    }

    pub fn prepare(&mut self) {
        let fn_type = self.ctx.i64_type().fn_type(&[], false);

        self.fn_val = Some(
            self.module.add_function(Self::ENTRYPOINT_FN_NAME, fn_type, None)
        );

        let entry = self.ctx.append_basic_block(self.fn_value(), "entry");
        self.builder.position_at_end(entry);
    }

    pub fn optimize(&mut self) -> Result<(), &'static str> {
        let fn_val = self.fn_value();

        if fn_val.verify(true) {
            self.fpm.run_on(&fn_val);

            Ok(())
        } else {
            unsafe { fn_val.delete(); }

            Err("invalid function")
        }
    }
}

pub type EntrypointFunction = unsafe extern "C" fn() -> i32;
