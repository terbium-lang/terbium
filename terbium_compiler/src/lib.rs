use terbium_grammar::{Expr, Node, Operator, Spanned};

use inkwell::{
    builder::Builder,
    context::Context,
    FloatPredicate,
    IntPredicate,
    module::Module,
    passes::PassManager,
    types::{BasicType, VectorType, FloatMathType},
    values::{AnyValue, AnyValueEnum, FunctionValue, PointerValue},
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
        node: &Spanned<Node>,
    ) -> Result<FunctionValue<'ctx>, &'static str> {
        let mut compiler = Self {
            ctx,
            builder,
            fpm,
            module,
            fn_val: None,
        };

        compiler.prepare();
        compiler.compile_node(node)?;

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

    pub fn eval_expr(&mut self, expr: &Spanned<Expr>) -> Result<AnyValueEnum<'ctx>, &'static str> {
        let (expr, span) = expr.node_span();

        Ok(match expr {
            Expr::Integer(i) => {
                // TODO: this wraps the u128, but a `long` type which is an i128 isp planned
                self.ctx.i64_type().const_int(*i as u64, false).as_any_value_enum()
            }
            Expr::Float(f) => {
                self.ctx.f64_type().const_float_from_string(f).as_any_value_enum()
            }
            Expr::String(s) => {
                self.ctx.const_string(s.as_bytes(), false).as_any_value_enum()
            }
            Expr::Bool(b) => {
                if *b {
                    self.ctx.bool_type().const_all_ones().as_any_value_enum()
                } else {
                    self.ctx.bool_type().const_zero().as_any_value_enum()
                }
            }
            Expr::UnaryExpr { operator, value } => {
                let (op, span) = operator.node_span();
                let value = self.eval_expr(value)?;

                match op {
                    Operator::Add => match value {
                        AnyValueEnum::IntValue(i) => i.as_any_value_enum(),
                        AnyValueEnum::FloatValue(f) => f.as_any_value_enum(),
                        // Custom operator overloading
                        AnyValueEnum::StructValue(_s) => todo!(),
                        // Analyzer should've caught this
                        _ => unreachable!(),
                    }
                    Operator::Sub => match value {
                        AnyValueEnum::IntValue(i) => if i.is_const() {
                            i.const_neg().as_any_value_enum()
                        } else {
                            self.builder.build_int_neg(i, "tmpneg")
                                .as_any_value_enum()
                        },
                        AnyValueEnum::FloatValue(f) => if f.is_const() {
                            f.const_neg().as_any_value_enum()
                        } else {
                            self.builder.build_float_neg(f, "tmpneg")
                                .as_any_value_enum()
                        },
                        // Custom operator overloading
                        AnyValueEnum::StructValue(_s) => todo!(),
                        // Analyzer should've caught this
                        _ => unreachable!(),
                    }
                    Operator::Not => match value {
                        AnyValueEnum::IntValue(i) => if i.is_const() {
                            i.const_select(
                                self.ctx.bool_type().const_zero(),
                                self.ctx.bool_type().const_all_ones(),
                            )
                            .as_any_value_enum()
                        } else {
                            self.builder.build_not(i, "tmpnot")
                                .as_any_value_enum()
                        }
                        AnyValueEnum::FloatValue(f) => if f.is_const() {
                            f.const_compare(FloatPredicate::UEQ, self.ctx.f64_type().const_zero())
                                .const_select(
                                    self.ctx.bool_type().const_zero(),
                                    self.ctx.bool_type().const_all_ones(),
                                )
                                .as_any_value_enum()
                        } else {
                            self.builder.build_float_compare(
                                FloatPredicate::UEQ,
                                f,
                                self.ctx.f64_type().const_zero(),
                                "tmpnot",
                            )
                            .as_any_value_enum()
                        }
                        _ => todo!(),
                    }
                    Operator::BitNot => todo!(),
                    _ => unreachable!(),
                }
            }
            Expr::BinaryExpr{ operator, lhs, rhs } => {
                let (operator, span) = operator.node_span();

                let left = self.eval_expr(lhs)?;
                let right = self.eval_expr(rhs)?;

                match operator {
                    Operator::Add => {
                        match (left.as_any_value_enum(), right.as_any_value_enum()) {
                            (AnyValueEnum::IntValue(lhs), AnyValueEnum::IntValue(rhs)) => {
                                self.builder.build_int_add(lhs, rhs, "tmpintadd")
                                    .as_any_value_enum()
                            }
                            (AnyValueEnum::FloatValue(lhs), AnyValueEnum::FloatValue(rhs)) => {
                                self.builder.build_float_add(lhs, rhs, "tmpfloatadd")
                                    .as_any_value_enum()
                            }
                            (AnyValueEnum::FloatValue(f), AnyValueEnum::IntValue(i))
                            | (AnyValueEnum::IntValue(i), AnyValueEnum::FloatValue(f)) => {
                                let int = self.builder.build_signed_int_to_float(i, self.ctx.f64_type(), "tmpintfloatconv");

                                self.builder.build_float_add(int, f, "tmpfloatadd")
                                    .as_any_value_enum()
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

    pub fn prepare(&mut self) {
        let fn_type = self.ctx.i32_type().fn_type(&[], false);

        self.fn_val = Some(
            self.module.add_function(Self::ENTRYPOINT_FN_NAME, fn_type, None)
        );
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

//#[cfg(tests)]
mod tests {
    use inkwell::context::Context;
    use inkwell::passes::PassManager;
    use terbium_grammar::{Node, Source, ParseInterface, Spanned, Span};
    use crate::Compiler;

    #[test]
    fn test_compiler() {
        let sample = String::from("-1");
        let node = Node::from_string(Source::default(), sample).unwrap();

        let ctx = Context::create();
        let module = ctx.create_module("tmp");
        let builder = ctx.create_builder();

        let fpm = PassManager::create(&module);

        fpm.add_instruction_combining_pass();
        fpm.add_reassociate_pass();
        fpm.add_gvn_pass();
        fpm.add_cfg_simplification_pass();
        fpm.add_basic_alias_analysis_pass();
        fpm.add_promote_memory_to_register_pass();
        fpm.add_instruction_combining_pass();
        fpm.add_reassociate_pass();
        fpm.initialize();

        let func = Compiler::compile(
            &ctx,
            &builder,
            &fpm,
            &module,
            &Spanned::new(node, Span::default()),
        ).unwrap();

        func.print_to_stderr();
    }
}
