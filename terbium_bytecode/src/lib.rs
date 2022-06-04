//! Crate for Terbium byte-code/intermediate represenation.
//!
//! Because Terbium can be directly interpreted or be compiled into LLVM IR,
//! there must be a representation of Terbium code that can provide an easy
//! entrypoint to both.

mod util;

pub use util::EqComparableFloat;

pub type StackIndex = u32;
pub type RelativeStackIndex = i32;
pub type ObjectPtr = u32;

#[derive(Clone, Debug)]
pub enum Instruction {
    // Constants mapped to a lookup table
    LoadInt(i128),
    LoadFloat(EqComparableFloat),
    LoadString(String),
    LoadVar(ObjectPtr),
    Load(ObjectPtr),

    // Operations
    AddInt,
    SubInt,
    MulInt,
    DivInt,
    TrueDivInt, // Div keeps type, while TrueDiv doesn't. e.g. Div(5, 2) -> 2 but TrueDiv(5, 2) -> 2.5
    PowInt,

    Pop,
}

impl Instruction {
    /// Return a i8 representing the change in the count of elements in the stack.
    pub fn stack_diff(&self) -> i8 {
        match self {
            Self::LoadInt(_) => 1,
            Self::Pop => -1,

            _ => todo!(),
        }
    }
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
    }};
}

impl Program {
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    pub fn inner(&self) -> impl Iterator<Item = &Instruction> {
        self.inner.iter()
    }
}

impl FromIterator<Instruction> for Program {
    fn from_iter<I: IntoIterator<Item = Instruction>>(iter: I) -> Self {
        Self {
            inner: iter.into_iter().collect(),
        }
    }
}
