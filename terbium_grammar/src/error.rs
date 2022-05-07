use std::fmt::{Display, Result as FmtResult};

pub struct InvalidToken {
    pub start: usize,
    pub end: usize,
}

impl Display for InvalidToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> FmtResult {
        write!(f, "Invalid token: from pos {} to pos {}", self.start, self.end)
    }
}
