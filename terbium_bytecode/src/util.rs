use std::fmt::{Display, Formatter, Result as FmtResult};
use std::{
    fmt::Debug,
    hash::{Hash, Hasher},
};

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct EqComparableFloat(pub f64);

impl EqComparableFloat {
    fn key(self) -> u64 {
        u64::from_be_bytes(self.0.to_be_bytes())
    }
}

impl From<f64> for EqComparableFloat {
    fn from(f: f64) -> Self {
        Self(f)
    }
}

impl From<EqComparableFloat> for f64 {
    fn from(f: EqComparableFloat) -> Self {
        f.0
    }
}

impl PartialEq for EqComparableFloat {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl PartialEq<f64> for EqComparableFloat {
    fn eq(&self, other: &f64) -> bool {
        self.key() == Self::from(other.to_owned()).key()
    }
}

impl Eq for EqComparableFloat {}

impl Hash for EqComparableFloat {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.key().hash(h);
    }
}

impl Default for EqComparableFloat {
    /// Returns the default value of 0.0
    fn default() -> Self {
        Self(0.0)
    }
}

impl Display for EqComparableFloat {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        std::fmt::Display::fmt(&self.0, f)
    }
}
