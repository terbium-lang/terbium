use std::iter::Peekable;

#[must_use]
pub fn to_snake_case(s: &str) -> String {
    CamelCaseRemover::new(s.chars())
        .flat_map(char::to_lowercase)
        .collect()
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
            }
            (next, _) => next,
        }
    }
}

#[must_use]
pub fn get_levenshtein_distance(a: &str, b: &str) -> usize {
    let mut result = 0;

    if a == b {
        return result;
    }

    let a_len = a.chars().count();
    let b_len = b.chars().count();

    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }

    let mut cache: Vec<usize> = (1..).take(a_len).collect();
    let mut a_dist;
    let mut b_dist;

    for (b_idx, b_code) in b.chars().enumerate() {
        result = b_idx;
        a_dist = b_idx;

        for (a_idx, a_code) in a.chars().enumerate() {
            b_dist = a_dist + (a_code != b_code) as usize;
            a_dist = cache[a_idx];

            result = if a_dist > result {
                if b_dist > result {
                    result + 1
                } else {
                    b_dist
                }
            } else if b_dist > a_dist {
                a_dist + 1
            } else {
                b_dist
            };

            cache[a_idx] = result;
        }
    }

    result
}
