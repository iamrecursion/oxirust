//! A hyphen-delimited subtag scanner shared by every parsing entry point.

/// Splits `input` on ASCII hyphens, yielding each subtag together with its end offset
/// (exclusive) within `input`.
///
/// Unlike [`slice::split`], this never silently loses position information: pairing each
/// subtag with its end offset is exactly what the grammar's byte-range bookkeeping needs.
///
/// The hyphen scan is a plain `core`-only byte search: language tags are at most a few dozen
/// bytes long, so a word-at-a-time scanner would never amortise its set-up cost here, and
/// staying dependency-free is what keeps this crate `#![no_std]`.
///
/// An empty `input` yields exactly one empty subtag (matching how `"".split('-')` behaves),
/// which the caller rejects as [`crate::error::TagParseErrorKind::EmptySubtag`]. Leading,
/// trailing, or doubled hyphens likewise surface as empty subtags in the sequence.
pub(crate) struct SubtagIter<'a> {
    remaining: &'a [u8],
    consumed: usize,
    finished: bool,
}

impl<'a> SubtagIter<'a> {
    /// Starts scanning `input` from its first byte.
    #[inline]
    pub(crate) fn new(input: &'a [u8]) -> Self {
        Self {
            remaining: input,
            consumed: 0,
            finished: false,
        }
    }
}

impl<'a> Iterator for SubtagIter<'a> {
    /// A subtag's bytes, paired with its end offset (exclusive) within the original input.
    type Item = (&'a [u8], usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        match self.remaining.iter().position(|byte| *byte == b'-') {
            Some(hyphen_at) => {
                let subtag = &self.remaining[..hyphen_at];
                let end = self.consumed + hyphen_at;
                self.remaining = &self.remaining[hyphen_at + 1..];
                self.consumed = end + 1;
                Some((subtag, end))
            }
            None => {
                self.finished = true;
                let end = self.consumed + self.remaining.len();
                Some((self.remaining, end))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn collect(input: &[u8]) -> Vec<(&[u8], usize)> {
        SubtagIter::new(input).collect()
    }

    #[test]
    fn splits_simple_tags() {
        assert_eq!(
            collect(b"zh-cmn-Hans-CN"),
            vec![
                (&b"zh"[..], 2),
                (&b"cmn"[..], 6),
                (&b"Hans"[..], 11),
                (&b"CN"[..], 14),
            ]
        );
    }

    #[test]
    fn single_subtag_has_no_hyphen() {
        assert_eq!(collect(b"de"), vec![(&b"de"[..], 2)]);
    }

    #[test]
    fn empty_input_yields_one_empty_subtag() {
        assert_eq!(collect(b""), vec![(&b""[..], 0)]);
    }

    #[test]
    fn leading_trailing_and_doubled_hyphens_surface_as_empty_subtags() {
        assert_eq!(
            collect(b"-ab-abc"),
            vec![(&b""[..], 0), (&b"ab"[..], 3), (&b"abc"[..], 7)]
        );
        assert_eq!(
            collect(b"ab-abc-"),
            vec![(&b"ab"[..], 2), (&b"abc"[..], 6), (&b""[..], 7)]
        );
        assert_eq!(
            collect(b"ab--abc"),
            vec![(&b"ab"[..], 2), (&b""[..], 3), (&b"abc"[..], 7)]
        );
    }
}
