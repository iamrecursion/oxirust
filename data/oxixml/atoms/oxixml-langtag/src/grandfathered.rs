//! The fixed list of RFC 5646 `grandfathered` tags (the `irregular` and `regular` ABNF
//! productions), and the canonical casing each one normalizes to.
//!
//! These twenty-six tags were registered under the obsolete [RFC 3066] and do not match the
//! `langtag` production (either at all, for `irregular` tags, or not as extended language /
//! variant subtags, for `regular` tags), so they are recognized as a closed list rather than
//! being parsed structurally.
//!
//! [RFC 3066]: https://www.rfc-editor.org/rfc/rfc3066

/// Every grandfathered tag paired with its canonical casing, as listed in
/// [RFC 5646 §2.1](https://www.rfc-editor.org/rfc/rfc5646#section-2.1) (`irregular` and
/// `regular` productions). Lookups are case-insensitive; the second element of each pair is
/// what a matching input normalizes to.
const GRANDFATHERED: &[(&str, &str)] = &[
    // `regular` — these match the `langtag` production syntactically, but their subtags are
    // not actually extended-language or variant subtags.
    ("art-lojban", "art-lojban"),
    ("cel-gaulish", "cel-gaulish"),
    ("no-bok", "no-bok"),
    ("no-nyn", "no-nyn"),
    ("zh-guoyu", "zh-guoyu"),
    ("zh-hakka", "zh-hakka"),
    ("zh-min", "zh-min"),
    ("zh-min-nan", "zh-min-nan"),
    ("zh-xiang", "zh-xiang"),
    // `irregular` — these do not match the `langtag` production at all.
    ("en-gb-oed", "en-GB-oed"),
    ("i-ami", "i-ami"),
    ("i-bnn", "i-bnn"),
    ("i-default", "i-default"),
    ("i-enochian", "i-enochian"),
    ("i-hak", "i-hak"),
    ("i-klingon", "i-klingon"),
    ("i-lux", "i-lux"),
    ("i-mingo", "i-mingo"),
    ("i-navajo", "i-navajo"),
    ("i-pwn", "i-pwn"),
    ("i-tao", "i-tao"),
    ("i-tay", "i-tay"),
    ("i-tsu", "i-tsu"),
    ("sgn-be-fr", "sgn-BE-FR"),
    ("sgn-be-nl", "sgn-BE-NL"),
    ("sgn-ch-de", "sgn-CH-DE"),
];

/// Returns the canonical casing of `candidate` if it is one of the twenty-six grandfathered
/// tags, comparing case-insensitively. Performs no heap allocation.
#[inline]
pub(crate) fn canonical_grandfathered(candidate: &[u8]) -> Option<&'static str> {
    GRANDFATHERED
        .iter()
        .find(|(name, _)| name.as_bytes().eq_ignore_ascii_case(candidate))
        .map(|(_, canonical)| *canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_every_listed_tag_in_any_case() {
        for (name, canonical) in GRANDFATHERED {
            assert_eq!(canonical_grandfathered(name.as_bytes()), Some(*canonical));
            assert_eq!(
                canonical_grandfathered(name.to_uppercase().as_bytes()),
                Some(*canonical)
            );
            assert_eq!(
                canonical_grandfathered(name.to_lowercase().as_bytes()),
                Some(*canonical)
            );
        }
    }

    #[test]
    fn rejects_non_grandfathered_tags() {
        assert_eq!(canonical_grandfathered(b"en-us"), None);
        assert_eq!(canonical_grandfathered(b"i-notexist"), None);
        assert_eq!(canonical_grandfathered(b"zh-min-nan-extra"), None);
        assert_eq!(canonical_grandfathered(b""), None);
    }

    #[test]
    fn list_has_no_duplicate_keys() {
        for (index, (name, _)) in GRANDFATHERED.iter().enumerate() {
            for (other_name, _) in &GRANDFATHERED[index + 1..] {
                assert!(
                    !name.eq_ignore_ascii_case(other_name),
                    "duplicate grandfathered entry: {name}"
                );
            }
        }
    }
}
