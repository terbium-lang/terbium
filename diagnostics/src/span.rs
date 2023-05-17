use std::mem::replace;
use std::ops::Range;
use common::span::{Span, Src};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Line {
    pub offset: usize,
    pub len: usize,
    pub chars: String,
}

impl Line {
    pub fn chars(&self) -> impl Iterator<Item = char> + '_ {
        self.chars.chars()
    }
}

#[derive(Clone, Debug)]
pub struct LineSrc {
    pub lines: Vec<Line>,
    pub len: usize,
}

// Taken from https://github.com/zesterer/ariadne/blob/main/src/source.rs#L63-L115
impl<S: AsRef<str>> From<S> for LineSrc {
    fn from(s: S) -> Self {
        let mut offset = 0;
        // (Last line, last line ends with CR)
        let mut last_line: Option<(Line, bool)> = None;
        let mut lines: Vec<Line> = s
            .as_ref()
            .split_inclusive([
                '\r', // Carriage return
                '\n', // Line feed
                '\x0B', // Vertical tab
                '\x0C', // Form feed
                '\u{0085}', // Next line
                '\u{2028}', // Line separator
                '\u{2029}' // Paragraph separator
            ])
            .flat_map(|line| {
                // Returns last line and set `last_line` to current `line`
                // A hack that makes `flat_map` deals with consecutive lines
                if let Some((last, ends_with_cr)) = last_line.as_mut() {
                    if *ends_with_cr && line == "\n" {
                        last.len += 1;
                        offset += 1;
                        return replace(&mut last_line, None).map(|(l, _)| l);
                    }
                }

                let len = line.chars().count();
                let ends_with_cr = line.ends_with('\r');
                let line = Line {
                    offset,
                    len,
                    chars: line.trim_end().to_owned(),
                };
                offset += len;
                replace(&mut last_line, Some((line, ends_with_cr))).map(|(l, _)| l)
            })
            .collect();

        if let Some((l, _)) = last_line {
            lines.push(l);
        }

        Self {
            lines,
            len: offset,
        }
    }
}

// Derived from https://github.com/zesterer/ariadne/blob/main/src/source.rs#L117-L157
impl LineSrc {
    /// Get the length of the total number of characters in the source.
    pub fn len(&self) -> usize { self.len }

    /// Return an iterator over the characters in the source.
    pub fn chars(&self) -> impl Iterator<Item = char> + '_ {
        self.lines.iter().map(|l| l.chars()).flatten()
    }

    /// Get access to a specific, zero-indexed [`Line`].
    pub fn line(&self, idx: usize) -> Option<&Line> { self.lines.get(idx) }

    /// Return an iterator over the [`Line`]s in this source.
    pub fn lines(&self) -> impl ExactSizeIterator<Item = &Line> + '_ { self.lines.iter() }

    /// Get the line that the given offset appears on, and the line/column numbers of the offset.
    ///
    /// Note that the line/column numbers are zero-indexed.
    pub fn get_offset_line(&self, offset: usize) -> Option<(&Line, usize, usize)> {
        if offset <= self.len {
            let idx = self.lines
                .binary_search_by_key(&offset, |line| line.offset)
                .unwrap_or_else(|idx| idx.saturating_sub(1));
            let line = &self.lines[idx];
            assert!(offset >= line.offset, "offset = {}, line.offset = {}", offset, line.offset);
            Some((line, idx, offset - line.offset))
        } else {
            None
        }
    }

    /// Get the range of lines that this span runs across.
    ///
    /// The resulting range is guaranteed to contain valid line indices (i.e: those that can be used for
    /// [`Source::line`]).
    pub fn get_line_range(&self, span: Span) -> Range<usize> {
        let start = self.get_offset_line(span.start).map_or(0, |(_, l, _)| l);
        let end = self.get_offset_line(span.end.saturating_sub(1).max(span.start)).map_or(self.lines.len(), |(_, l, _)| l + 1);
        start..end
    }
}
