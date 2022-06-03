use std::iter::Peekable;

pub fn to_snake_case(s: &str) -> String {
    CamelCaseRemover::new(s.chars()).flat_map(char::to_lowercase).collect()
}

struct CamelCaseRemover<I: Iterator<Item = char>> {
    iter: Peekable<I>,
    br: bool,
}

impl<I: Iterator<Item = char>> CamelCaseRemover<I> {
    fn new(iter: I) -> Self {
        Self {
            iter: iter.peekable(),
            br: false,
        }
    }
}

impl<I: Iterator<Item = char>> Iterator for CamelCaseRemover<I> {
    type Item = char;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.br {
            self.br = false;
            return Some('_');
        }

        match (self.iter.next(), self.iter.peek()) {
            (Some(curr), Some(next)) if curr.is_lowercase() && next.is_uppercase() => {
                self.br = true;
                Some(curr)
            },
            (next, _) => next,
        }
    }
}
