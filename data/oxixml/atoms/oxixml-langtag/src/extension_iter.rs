//! Splits an already-validated extension sequence string into one entry per singleton.

/// Iterates over `(singleton, values)` pairs of an extension sequence string, e.g. turns
/// `"a-babble-b-warble"` into `[('a', "babble"), ('b', "warble")]`.
///
/// The input is assumed to already be well-formed — the concatenation of one or more
/// `singleton 1*("-" (2*8alphanum))` groups, as returned by
/// [`LanguageTag::extension`](crate::LanguageTag::extension) — so no validation is repeated
/// here; this is pure re-segmentation of a string a successful parse already accepted.
pub(crate) struct ExtensionSubtags<'a> {
    remainder: &'a str,
}

impl<'a> ExtensionSubtags<'a> {
    /// Starts iterating over `extension`, the full (possibly multi-singleton) extension
    /// sequence string.
    #[inline]
    pub(crate) fn new(extension: &'a str) -> Self {
        Self {
            remainder: extension,
        }
    }
}

impl<'a> Iterator for ExtensionSubtags<'a> {
    type Item = (char, &'a str);

    fn next(&mut self) -> Option<Self::Item> {
        let bytes = self.remainder.as_bytes();
        let singleton = *bytes.first()? as char;

        // Values start right after "<singleton>-". Scan forward token by token (a token is
        // the run of bytes up to the next hyphen, or to the end of the string) until we find
        // one of length 1 — the start of the *next* group's singleton — or run out of tokens.
        let values_start = bytes.len().min(2);
        let mut token_start = values_start;
        let mut next_group_start = None;
        while token_start < bytes.len() {
            let token_len = self.remainder[token_start..]
                .find('-')
                .unwrap_or(bytes.len() - token_start);
            if token_len == 1 {
                next_group_start = Some(token_start - 1); // include the separating hyphen
                break;
            }
            token_start += token_len + 1;
        }

        let group_end = next_group_start.unwrap_or(bytes.len());
        let values = &self.remainder[values_start..group_end];
        self.remainder = match next_group_start {
            Some(start) => &self.remainder[start + 1..],
            None => "",
        };
        Some((singleton, values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn collect(extension: &str) -> Vec<(char, &str)> {
        ExtensionSubtags::new(extension).collect()
    }

    #[test]
    fn single_group_single_value() {
        assert_eq!(collect("u-co-phonebk"), vec![('u', "co-phonebk")]);
    }

    #[test]
    fn single_group_short_value() {
        assert_eq!(collect("a-bb"), vec![('a', "bb")]);
    }

    #[test]
    fn two_groups() {
        assert_eq!(
            collect("a-babble-b-warble"),
            vec![('a', "babble"), ('b', "warble")]
        );
    }

    #[test]
    fn many_groups_with_multi_value_extensions() {
        assert_eq!(
            collect("z-foo-a-bar-f-spam-b-eggs"),
            vec![('z', "foo"), ('a', "bar"), ('f', "spam"), ('b', "eggs"),]
        );
    }

    #[test]
    fn group_with_several_values() {
        assert_eq!(
            collect("r-extended-sequence"),
            vec![('r', "extended-sequence")]
        );
    }

    #[test]
    fn empty_input_yields_nothing() {
        assert_eq!(collect(""), Vec::<(char, &str)>::new());
    }
}
