//! Interprets Terbium AST into Bytecode.

use super::{AddrRepr, Instruction, Program};
use terbium_grammar::{Body, Expr, Node, Operator};
use crate::Addr;

pub struct Interpreter {
    ast: Vec<Node>,
    return_last: bool,
    program: Program,
}

type MaybeProc = Option<AddrRepr>;

impl Interpreter {
    pub fn from_ast(Body(ast, return_last): Body) -> Self {
        Self { ast, return_last, program: Program::new() }
    }

    pub fn push(&mut self, procedure: MaybeProc, instr: Instruction) {
        self.program.push(procedure, instr)
    }

    pub fn interpret_expr(&mut self, proc: MaybeProc, expr: Expr) {
        match expr {
            Expr::Integer(i) => self.push(proc, Instruction::LoadInt(i as i128)),
            Expr::UnaryExpr { operator, value } => {
                self.interpret_expr(proc, *value);

                match operator {
                    Operator::Add => self.push(proc, Instruction::UnOpPos),
                    Operator::Sub => self.push(proc, Instruction::UnOpNeg),
                    _ => todo!(),
                }
            }
            Expr::BinaryExpr { operator, lhs, rhs } => {
                self.interpret_expr(proc, *lhs);
                self.interpret_expr(proc, *rhs);

                match operator {
                    Operator::Add => self.push(proc, Instruction::BinOpAdd),
                    Operator::Sub => self.push(proc, Instruction::BinOpSub),
                    Operator::Mul => self.push(proc, Instruction::BinOpMul),
                    Operator::Div => self.push(proc, Instruction::BinOpDiv),
                    _ => todo!(),
                }
            }
            Expr::If {
                condition,
                body,
                else_if_bodies,
                else_body,
                return_last,
            } => {
                self.interpret_expr(proc, *condition);

                let then_proc = self.program.create_procedure();
                self.interpret_body(Some(then_proc), Body(body, return_last));

                if else_if_bodies.is_empty() {
                    if let Some(else_body) = else_body {
                        let else_proc = self.program.create_procedure();
                        self.interpret_body(Some(else_proc), else_body);
                        self.push(proc, Instruction::JumpIfElse(
                            Addr::Procedure(then_proc),
                            Addr::Procedure(else_proc),
                        ));
                        return;
                    }

                    self.push(proc, Instruction::JumpIf(Addr::Procedure(then_proc)));
                    return;
                }

                // TODO
                let mut last_proc = self.program.create_procedure();
                for (cond, body) in else_if_bodies {

                }
            }
            _ => todo!(),
        }
    }

    pub fn interpret_node(&mut self, proc: MaybeProc, node: Node) {
        match node {
            Node::Expr(e) => self.interpret_expr(proc, e),
            Node::Module(m) => self.interpret_body(proc, Body(m, false)),
            Node::Return(e) => {
                if let Some(e) = e {
                    self.interpret_expr(proc, e);
                }
                self.push(proc, Instruction::Ret);
            }
            _ => todo!(),
        }
    }

    pub fn interpret_body(&mut self, proc: MaybeProc, Body(body, return_last): Body) {
        for node in body {
            self.interpret_node(proc, node);
        }

        self.push(proc, if return_last {
            Instruction::Ret
        } else {
            Instruction::RetNull
        });
    }
}