//! Crate for Terbium byte-code/intermediate represenation.
//!
//! Because Terbium can be directly interpreted or be compiled into LLVM IR,
//! there must be a representation of Terbium code that can provide an easy
//! entrypoint to both.

mod util;

pub use util::EqComparableFloat;

pub type ObjectPtr = u32;

#[derive(Clone, Debug)]
pub enum Instruction {
    Invalid(&'static str),
    Pass,
    Break,

    AllocInt(i128),
    AllocByte(u8),
    AllocFloat(EqComparableFloat),
    AllocString(ObjectPtr),
    AllocVar(ObjectPtr),
    Alloc(ObjectPtr),
    AllocObject {
        pos: ObjectPtr,
        ty: ObjectPtr,
    },
    AllocObjectAttr {
        obj: ObjectPtr,
        attr: String,
        value: ObjectPtr,
    },
}

#[derive(Debug)]
pub struct Program {
    inner: Vec<Instruction>,
}

macro_rules! consume {
    ($split:expr => $($i:ident),* => $then:expr) => {{
        $(
            let $i = $split.next().ok_or("invalid bytecode".to_string())?;
        )*
        $then
    }}
}

macro_rules! parse {
    ($e:expr) => {{
        $e.parse().map_err(|_| "invalid number")?
    }};
    ($e:expr, $msg:literal) => {{
        $e.parse().map_err(|_| $msg)?
    }}
}

impl Program {
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    pub fn parse(code: String) -> Result<Self, String> {
        Ok(code
            .lines()
            .map(|line| -> Result<Instruction, String> {
                let mut split = line.split(' ');
                let command = split.next().ok_or("no command".to_string())?;

                Ok(match command {
                    "pass" => consume!(split => => Instruction::Pass),
                    "break" => consume!(split => => Instruction::Break),
                    "alloc" => consume!(split => pos, ty => Instruction::AllocObject {
                        pos: parse!(pos),
                        ty: parse!(ty),
                    }),
                    "alloc_attr" => consume!(split => obj, attr, value =>
                        Instruction::AllocObjectAttr {
                            obj: parse!(obj),
                            attr: attr.to_string(),
                            value: parse!(value),
                        }
                    ),
                    _ => Err(format!("unknown command {}", command))?,
                })
            })
            .filter_map(Result::ok)
            .collect()
        )
    }
}

impl FromIterator<Instruction> for Program {
    fn from_iter<I: IntoIterator<Item = Instruction>>(iter: I) -> Self {
        Self { inner: iter.into_iter().collect() }
    }
}
