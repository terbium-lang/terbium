//! Interprets Terbium AST into Bytecode.

use super::{AddrRepr, Instruction, Program};
use crate::Addr;
use terbium_grammar::{Body, Expr, Node, Operator};

pub struct Interpreter {
    program: Program,
}

type MaybeProc = Option<AddrRepr>;

impl Interpreter {
    pub fn new() -> Self {
        Self {
            program: Program::new(),
        }
    }

    pub fn program(self) -> Program {
        self.program
    }

    pub fn push(&mut self, procedure: MaybeProc, instr: Instruction) {
        self.program.push(procedure, instr)
    }

    pub fn push_return(&mut self, procedure: AddrRepr, return_last: bool) {
        self.push(
            Some(procedure),
            if return_last {
                Instruction::Ret
            } else {
                Instruction::RetNull
            },
        );
    }

    pub fn interpret_expr(&mut self, proc: MaybeProc, expr: Expr) {
        match expr {
            Expr::Integer(i) => self.push(proc, Instruction::LoadInt(i as i128)),
            Expr::Bool(b) => self.push(proc, Instruction::LoadBool(b)),
            Expr::String(s) => self.push(proc, Instruction::LoadString(s)),
            Expr::Float(f) => self.push(
                proc,
                Instruction::LoadFloat(f.parse::<f64>().unwrap().into()),
            ),
            Expr::UnaryExpr { operator, value } => {
                self.interpret_expr(proc, *value);

                self.push(
                    proc,
                    match operator {
                        Operator::Add => Instruction::UnOpPos,
                        Operator::Sub => Instruction::UnOpNeg,
                        Operator::Not => Instruction::OpLogicalNot,
                        _ => unimplemented!(),
                    },
                );
            }
            Expr::BinaryExpr { operator, lhs, rhs } => {
                self.interpret_expr(proc, *lhs);
                self.interpret_expr(proc, *rhs);

                self.push(
                    proc,
                    match operator {
                        Operator::Add => Instruction::BinOpAdd,
                        Operator::Sub => Instruction::BinOpSub,
                        Operator::Mul => Instruction::BinOpMul,
                        Operator::Div => Instruction::BinOpDiv,
                        Operator::Eq => Instruction::OpEq,
                        _ => todo!(),
                    },
                );
            }
            Expr::If {
                condition,
                body,
                mut else_if_bodies,
                else_body,
                return_last,
            } => {
                if else_if_bodies.is_empty() {
                    self.interpret_expr(proc, *condition);

                    let then_proc = self.program.create_procedure();
                    self.interpret_body(Some(then_proc), Body(body, return_last));

                    if let Some(else_body) = else_body {
                        let else_proc = self.program.create_procedure();
                        self.interpret_body(Some(else_proc), Body(else_body.0, return_last));

                        self.push(
                            proc,
                            Instruction::JumpIfElse(
                                Addr::Procedure(then_proc),
                                Addr::Procedure(else_proc),
                            ),
                        );
                        return;
                    }

                    self.push(proc, Instruction::JumpIf(Addr::Procedure(then_proc)));
                    return;
                }

                else_if_bodies.insert(0, (*condition, Body(body, return_last)));

                let mut last_parent = proc;
                let mut last_else = self.program.create_procedure();

                for (cond, body) in else_if_bodies {
                    self.interpret_expr(last_parent, cond);

                    let then_proc = self.program.create_procedure();
                    self.interpret_body(Some(then_proc), body);

                    self.push(
                        last_parent,
                        Instruction::JumpIfElse(
                            Addr::Procedure(then_proc),
                            Addr::Procedure(last_else),
                        ),
                    );

                    last_parent = Some(last_else);
                    last_else = self.program.create_procedure();
                }

                if let Some(else_body) = else_body {
                    self.interpret_body(Some(last_else), else_body);
                } else {
                    self.push(Some(last_else), Instruction::RetNull);
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
                    self.push(proc, Instruction::Ret);
                } else {
                    self.push(proc, Instruction::RetNull);
                }
            }
            _ => todo!(),
        }
    }

    pub fn interpret_body(&mut self, proc: MaybeProc, Body(body, return_last): Body) {
        for node in body {
            self.interpret_node(proc, node);
        }

        if let Some(proc) = proc {
            self.push_return(proc, return_last);
        } else {
            self.push(None, Instruction::Halt);
        }
    }
}
