//! Interprets Terbium AST into Bytecode.

use super::{Instruction, Program};
use terbium_grammar::{Body, Expr, Node, Operator};

pub struct Interpreter {
    ast: Vec<Node>,
    return_last: bool,
    instructions: Vec<Instruction>,
}

impl Interpreter {
    pub fn from_ast(Body(ast, return_last): Body) -> Self {
        Self { ast, return_last, instructions: Vec::new() }
    }

    pub fn push(&mut self, instr: Instruction) {
        self.instructions.push(instr)
    }

    pub fn interpret_expr(&mut self, expr: Expr) {
        match expr {
            Expr::Integer(i) => self.push(Instruction::LoadInt(i as i128)),
            Expr::UnaryExpr { operator, value } => {
                self.interpret_expr(*value);

                match operator {
                    Operator::Add => self.push(Instruction::UnOpPos),
                    Operator::Sub => self.push(Instruction::UnOpNeg),
                    _ => todo!(),
                }
            }
            Expr::BinaryExpr { operator, lhs, rhs } => {
                self.interpret_expr(*lhs);
                self.interpret_expr(*rhs);

                match operator {
                    Operator::Add => self.push(Instruction::BinOpAdd),
                    Operator::Sub => self.push(Instruction::BinOpSub),
                    Operator::Mul => self.push(Instruction::BinOpMul),
                    Operator::Div => self.push(Instruction::BinOpDiv),
                    _ => todo!(),
                }
            }
            _ => todo!(),
        }
    }
}