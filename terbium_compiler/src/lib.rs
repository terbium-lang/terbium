use std::intrinsics::unreachable;
use terbium_grammar::{Expr, Operator, Spanned};

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
    #[inline]
    fn fn_value(&self) -> FunctionValue<'ctx> {
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

    pub fn eval_expr<T>(&mut self, expr: &Spanned<Expr>) -> Result<T, &'static str>
    where
        T: AnyValue<'ctx>,
    {
        let (expr, span) = expr.node_span();

        Ok(match expr {
            Expr::Integer(i) => {
                // TODO: this wraps the u128, but a `long` type which is an i128 isp planned
                self.ctx.i64_type().const_int(*i as u64, false)
            }
            Expr::Float(f) => {
                self.ctx.f64_type().const_float_from_string(f)
            }
            Expr::String(s) => {
                self.ctx.const_string(s.as_bytes(), false)
            }
            Expr::Bool(b) => {
                if b {
                    self.ctx.bool_type().const_all_ones()
                } else {
                    self.ctx.bool_type().const_zero()
                }
            }
            Expr::UnaryExpr { operator, value } => {
                let (op, span) = operator.node_span();
                let value: impl AnyValue<'ctx> = self.eval_expr(value)?;

                match op {
                    Operator::Add => match value.as_any_value_enum() {
                        AnyValueEnum::IntValue(i) => i,
                        AnyValueEnum::FloatValue(f) => f,
                        // Custom operator overloading
                        AnyValueEnum::StructValue(_s) => todo!(),
                        // Analyzer should've caught this
                        _ => unreachable!(),
                    }
                    Operator::Sub => match value.as_any_value_enum() {
                        AnyValueEnum::IntValue(i) => if i.is_const() {
                            i.const_neg()
                        } else {
                            self.builder.build_int_neg(i, "tmpneg")
                        },
                        AnyValueEnum::FloatValue(f) => if f.is_const() {
                            f.const_neg()
                        } else {
                            self.builder.build_float_neg(f, "tmpneg")
                        },
                        // Custom operator overloading
                        AnyValueEnum::StructValue(_s) => todo!(),
                        // Analyzer should've caught this
                        _ => unreachable!(),
                    }
                    Operator::Not => match value.as_any_value_enum() {
                        AnyValueEnum::IntValue(i) => if i.is_const() {
                            i.const_select(
                                self.ctx.bool_type().const_zero(),
                                self.ctx.bool_type().const_all_ones(),
                            )
                        } else {
                            self.builder.build_int_compare(
                                IntPredicate::EQ,
                                i,
                                self.ctx.bool_type().const_zero(),
                                "tmpnot",
                            )
                        }
                        AnyValueEnum::FloatValue(f) => if f.is_const() {
                            f.const_compare(FloatPredicate::UEQ, self.ctx.f64_type().const_zero())
                                .const_select(
                                    self.ctx.bool_type().const_zero(),
                                    self.ctx.bool_type().const_all_ones(),
                                )
                        } else {
                            self.builder.build_float_compare(
                                FloatPredicate::UEQ,
                                f,
                                self.ctx.f64_type().const_zero(),
                                "tmpnot",
                            )
                        }
                        _ => todo!(),
                    }
                    Operator::BitNot => todo!(),
                    _ => unreachable!(),
                }
            }
            Expr::Array(e) => {
                VectorType::const_vector(
                    e.into_iter().map(
                        |e| self.eval_expr(e)?
                    ).collect::<&[T]>()
                )
            }
            Expr::BinaryExpr{ operator, lhs, rhs } => {
                let left: impl AnyValue<'ctx> = self.eval_expr(lhs)?;
                let right: impl AnyValue<'ctx> = self.eval_expr(rhs)?;

                match operator {
                    Operator::Add => {
                        match (left.as_any_value_enum(), right.as_any_value_enum()) {
                            (AnyValueEnum::IntValue(lhs), AnyValueEnum::IntValue(rhs)) => {
                                self.builder.build_int_add(lhs, rhs, "addint")
                            }
                            (AnyValueEnum::FloatValue(lhs), AnyValueEnum::FloatValue(rhs)) => {
                                self.builder.build_float_add(lhs, rhs, "addfloat")
                            }
                            (AnyValueEnum::FloatValue(f), AnyValueEnum::IntValue(i)) |
                            (AnyValueEnum::IntValue(i), AnyValueEnum::FloatValue(f)) => {
                                let int = self.builder.build_signed_int_to_float(i, FloatMathType, "inttofloat");
                                self.builder.build_float_add(int, f, "addfloat")
                            }
                            _ => todo!()
                        }
                    }
                }
            }
            _ => todo!(),
        })
    }
}
