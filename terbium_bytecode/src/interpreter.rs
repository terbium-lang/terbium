//! Interprets Terbium AST into Bytecode.
// TODO: utilize #![feature(box_patterns)] for this module

use std::collections::HashMap;

use super::{Addr, AddrRepr, Instruction, Program};
use terbium_grammar::ast::Target;
use terbium_grammar::{Body, Expr, Node, Operator};

// Contrary to assumption, this does not take into account scope and in reality
// it's just a super basic string-interner, in a way.
//
// This was made so that over multiple Programs, identifier names could be commonly mapped.
#[derive(Debug)]
pub struct IdentLookup {
    inner: HashMap<String, usize>,
    increment: usize,
}

impl IdentLookup {
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
            increment: 0,
        }
    }

    pub fn get(&mut self, ident: String) -> usize {
        *self.inner.entry(ident).or_insert_with(|| {
            let old = self.increment;
            self.increment += 1;

            old
        })
    }
}

pub struct Interpreter {
    program: Program,
    lookup: IdentLookup,
}

type MaybeProc = Option<AddrRepr>;

impl Interpreter {
    pub fn new() -> Self {
        Self {
            program: Program::new(),
            lookup: IdentLookup::new(),
        }
    }

    pub fn reset_program(&mut self) {
        self.program = Program::new();
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

    pub fn push_enter_scope(&mut self, procedure: AddrRepr) {
        self.push(Some(procedure), Instruction::EnterScope);
    }

    pub fn push_exit_scope(&mut self, procedure: AddrRepr) {
        self.push(Some(procedure), Instruction::ExitScope);
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
                        Operator::BitNot => Instruction::UnOpBitNot,
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
                        Operator::Ne => Instruction::OpNe,
                        Operator::Lt => Instruction::OpLt,
                        Operator::Le => Instruction::OpLe,
                        Operator::Gt => Instruction::OpGt,
                        Operator::Ge => Instruction::OpGe,
                        Operator::Or => Instruction::OpLogicalOr,
                        Operator::And => Instruction::OpLogicalAnd,
                        Operator::BitOr => Instruction::BinOpBitOr,
                        Operator::BitXor => Instruction::BinOpBitXor,
                        Operator::BitAnd => Instruction::BinOpBitAnd,
                        Operator::Range => todo!(),
                        _ => unimplemented!(),
                    },
                );
            }
            Expr::Ident(ident) => {
                let var = self.lookup.get(ident);
                self.push(proc, Instruction::LoadVar(var));
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
                    self.interpret_body_scoped(then_proc, Body(body, return_last));

                    if let Some(else_body) = else_body {
                        let else_proc = self.program.create_procedure();
                        self.interpret_body_scoped(else_proc, Body(else_body.0, return_last));

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
                    self.interpret_body_scoped(then_proc, body);

                    self.push(
                        last_parent,
                        Instruction::JumpIfElse(
                            Addr::Procedure(then_proc),
                            Addr::Procedure(last_else),
                        ),
                    );
                    // If last_parent is None, Halt instruction should be inserted automatically.
                    if let Some(p) = last_parent {
                        self.push_return(p, return_last);
                    }

                    last_parent = Some(last_else);
                    last_else = self.program.create_procedure();
                }
                self.program.pop_procedure();

                if let Some(else_body) = else_body {
                    self.interpret_body_scoped(last_parent.unwrap(), else_body);
                } else {
                    self.push(last_parent, Instruction::RetNull);
                }
            }
            Expr::While { condition, body } => {
                let loc = self.program.next_addr(proc);
                self.interpret_expr(proc, *condition);

                let body_proc = self.program.create_procedure();
                self.interpret_body_scoped_no_return(body_proc, body);

                self.push(Some(body_proc), Instruction::Jump(loc));
                self.push(proc, Instruction::JumpIf(Addr::Procedure(body_proc)));
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
            // TODO: maybe we can check const and mut at runtime, but those should be caught by the analyzer
            Node::Declare {
                targets,
                value,
                r#mut,
                r#const,
            } => {
                self.interpret_expr(proc, value);

                // TODO: currently we assume only one target
                let target = targets.first().unwrap();

                match target {
                    Target::Ident(s) => {
                        let key = self.lookup.get(s.clone());

                        self.push(
                            proc,
                            if r#mut {
                                Instruction::StoreMutVar(key)
                            } else if r#const {
                                Instruction::StoreConstVar(key)
                            } else {
                                Instruction::StoreVar(key)
                            },
                        );
                    }
                    _ => (),
                }
            }
            Node::Assign { targets, value } => {
                self.interpret_expr(proc, value);

                // TODO: currently we assume only one target
                let target = targets.first().unwrap();

                match target {
                    Target::Ident(s) => {
                        let key = self.lookup.get(s.clone());

                        self.push(proc, Instruction::AssignVar(key));
                    },
                    _ => (),
                }
            }
            _ => todo!(),
        }
    }

    pub fn interpret_body_no_return(&mut self, proc: MaybeProc, body: Vec<Node>) {
        for node in body {
            self.interpret_node(proc, node);
        }
    }

    pub fn interpret_body(&mut self, proc: MaybeProc, Body(body, return_last): Body) {
        self.interpret_body_no_return(proc, body);

        if let Some(proc) = proc {
            self.push_return(proc, return_last);
        } else {
            self.push(None, Instruction::Halt);
        }
    }

    pub fn interpret_body_scoped_no_return(&mut self, proc: AddrRepr, body: Vec<Node>) {
        self.push_enter_scope(proc);
        self.interpret_body_no_return(Some(proc), body);
        self.push_exit_scope(proc);
    }

    pub fn interpret_body_scoped(&mut self, proc: AddrRepr, Body(body, return_last): Body) {
        self.interpret_body_scoped_no_return(proc, body);
        self.push_return(proc, return_last);
    }
}
