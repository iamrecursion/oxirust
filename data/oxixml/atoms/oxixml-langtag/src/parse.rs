//! The well-formedness checker for the `langtag` and `privateuse` ABNF productions of
//! [RFC 5646 §2.1](https://www.rfc-editor.org/rfc/rfc5646#section-2.1):
//!
//! ```text
//! langtag       = language
//!                 ["-" script]
//!                 ["-" region]
//!                 *("-" variant)
//!                 *("-" extension)
//!                 ["-" privateuse]
//!
//! language      = 2*3ALPHA ["-" extlang] / 4ALPHA / 5*8ALPHA
//! extlang       = 3ALPHA *2("-" 3ALPHA)
//! script        = 4ALPHA
//! region        = 2ALPHA / 3DIGIT
//! variant       = 5*8alphanum / (DIGIT 3alphanum)
//! extension     = singleton 1*("-" (2*8alphanum))
//! singleton     = DIGIT / %x41-57 / %x59-5A / %x61-77 / %x79-7A   ; any letter but "x"/"X"
//! privateuse    = "x" 1*("-" (1*8alphanum))
//! ```
//!
//! On top of bare well-formedness this also enforces the two structural exclusions RFC 5646
//! §2.2.9 lists alongside "well-formed" as conditions for a tag being *valid* — no duplicate
//! variant subtags, and no duplicate extension singletons — because a tag that fails either one
//! is ambiguous to interpret (which occurrence of the repeated variant or singleton is
//! authoritative?) even though the bare ABNF does not forbid it syntactically. RFC 5646
//! Appendix A lists `ar-a-aaa-b-bbb-a-ccc` under "Some Invalid Tags" for exactly this reason.

use alloc::vec::Vec;

use crate::error::{LanguageTagParseError, TagParseErrorKind};
use crate::positions::{TagElementsPositions, TagShape};
use crate::subtag_iter::SubtagIter;

/// The parser's position within the `langtag` production, tracked across the subtag scan.
///
/// Every state except [`Start`](State::Start) already reflects a fully consumed primary
/// language subtag, so later match arms never need to special-case "have we even seen a
/// language subtag yet" beyond the initial branch.
#[derive(Debug, Clone, Copy)]
enum State {
    /// No subtag has been consumed yet; the next one must be the primary language.
    Start,
    /// The primary language subtag was 2 or 3 letters, so up to `3 - extlangs_seen` more
    /// extended-language subtags may still follow directly.
    AfterShortLanguage { extlangs_seen: u8 },
    /// Either the primary language subtag was 4-8 letters (no `extlang` is grammatically
    /// possible), or a short language's `extlang` sequence has been closed off by something
    /// else already matching (script, region, or a variant).
    AfterLanguage,
    /// The script subtag has been consumed; at most one is allowed, so it cannot recur.
    AfterScript,
    /// The region subtag has been consumed (at most one allowed), or a variant subtag has
    /// been consumed (any number allowed) — both leave the parser in the same position:
    /// ready for more variants, an extension, or private use.
    AfterRegionOrVariant,
    /// Collecting `2*8alphanum` value subtags for the extension introduced by `singleton`;
    /// `has_value` is `false` only until the first value subtag is seen.
    InExtension { singleton: u8, has_value: bool },
    /// Collecting `1*8alphanum` value subtags for the private-use sequence introduced by `x`;
    /// `has_value` is `false` only until the first value subtag is seen.
    InPrivateUse { has_value: bool },
}

#[inline]
fn is_alpha(subtag: &[u8]) -> bool {
    subtag.iter().all(u8::is_ascii_alphabetic)
}

#[inline]
fn is_digit(subtag: &[u8]) -> bool {
    subtag.iter().all(u8::is_ascii_digit)
}

#[inline]
fn is_alphanumeric(subtag: &[u8]) -> bool {
    subtag.iter().all(u8::is_ascii_alphanumeric)
}

/// A subtag matches the `variant` production when it is either `5*8alphanum` (any starting
/// character) or a `DIGIT` followed by exactly three more alphanumerics (`subtag.len() == 4`
/// and it starts with a digit — note that a *longer*, digit-starting alphanumeric subtag of
/// length 5-8 is still a variant, just via the first alternative rather than the second).
#[inline]
fn is_variant_shape(subtag: &[u8]) -> bool {
    is_alphanumeric(subtag)
        && ((subtag.len() >= 5 && subtag[0].is_ascii_alphabetic())
            || (subtag.len() >= 4 && subtag[0].is_ascii_digit()))
}

/// `true` when `input` begins with the private-use introducer `x-`/`X-` (case-insensitive).
///
/// This alone does not confirm `input` is a *well-formed* private-use tag — a lone `"x"` with
/// no following hyphen does not match, and is instead rejected later as an invalid one-letter
/// primary language subtag, which is the correct outcome per RFC 5646 Appendix A's `a-DE`
/// example (single-letter primary language subtags are never well-formed).
#[inline]
pub(crate) fn starts_privateuse(input: &[u8]) -> bool {
    input.len() >= 2 && input[..2].eq_ignore_ascii_case(b"x-")
}

/// Resolves `input` against the top-level `Language-Tag = langtag / privateuse /
/// grandfathered` choice and validates whichever production it matches.
pub(crate) fn parse_positions(input: &[u8]) -> Result<TagElementsPositions, LanguageTagParseError> {
    if crate::grandfathered::canonical_grandfathered(input).is_some() {
        return Ok(TagElementsPositions::whole_tag(
            TagShape::Grandfathered,
            input.len(),
        ));
    }
    if starts_privateuse(input) {
        return parse_privateuse_only(input);
    }
    parse_langtag(input)
}

/// Validates `input` as a standalone `privateuse = "x" 1*("-" (1*8alphanum))` tag.
///
/// The caller has already confirmed `input` starts with `x-`/`X-`.
pub(crate) fn parse_privateuse_only(
    input: &[u8],
) -> Result<TagElementsPositions, LanguageTagParseError> {
    let values = &input[2..];
    if values.is_empty() {
        return Err(LanguageTagParseError::new(
            TagParseErrorKind::EmptyPrivateUse,
        ));
    }
    for (subtag, _) in SubtagIter::new(values) {
        if subtag.is_empty() {
            return Err(LanguageTagParseError::new(TagParseErrorKind::EmptySubtag));
        }
        if subtag.len() > 8 {
            return Err(LanguageTagParseError::new(TagParseErrorKind::SubtagTooLong));
        }
        if !is_alphanumeric(subtag) {
            return Err(LanguageTagParseError::new(TagParseErrorKind::InvalidSubtag));
        }
    }
    Ok(TagElementsPositions::whole_tag(
        TagShape::PrivateUseOnly,
        input.len(),
    ))
}

/// Validates `input` as a `langtag` production and records the end offset of each component
/// that is present.
pub(crate) fn parse_langtag(input: &[u8]) -> Result<TagElementsPositions, LanguageTagParseError> {
    let mut state = State::Start;
    let mut language_end = 0;
    let mut extlang_end = None;
    let mut script_end = None;
    let mut region_end = None;
    let mut variant_end = None;
    let mut extension_end = None;

    // Case-insensitive membership scratch space, scoped to this single parse: which extension
    // singletons, and which variant subtags, have already been consumed. Kept as plain byte
    // slices/bytes rather than a hash set — real tags carry at most a handful of variants or
    // extensions, so linear scans are both simpler and at least as fast as hashing would be.
    let mut seen_singletons: Vec<u8> = Vec::new();
    let mut seen_variants: Vec<&[u8]> = Vec::new();

    for (subtag, end) in SubtagIter::new(input) {
        if subtag.is_empty() {
            return Err(LanguageTagParseError::new(TagParseErrorKind::EmptySubtag));
        }
        if subtag.len() > 8 {
            return Err(LanguageTagParseError::new(TagParseErrorKind::SubtagTooLong));
        }

        state = if let State::Start = state {
            // The very first subtag is always the primary language, no matter its shape —
            // checked before anything else so that a single-letter or `x`-shaped first subtag
            // (e.g. `a-value`, RFC 5646 Appendix A's `a-DE`) is rejected as an invalid primary
            // language rather than being misread as the start of an extension or private use
            // with no language subtag at all.
            if subtag.len() < 2 || !is_alpha(subtag) {
                return Err(LanguageTagParseError::new(
                    TagParseErrorKind::InvalidLanguage,
                ));
            }
            language_end = end;
            if subtag.len() < 4 {
                // Only a short (2-3 letter) primary language may carry `extlang`.
                State::AfterShortLanguage { extlangs_seen: 0 }
            } else {
                State::AfterLanguage
            }
        } else if let State::InPrivateUse { .. } = state {
            // Once private use has begun, every remaining subtag belongs to it, whatever its
            // shape — `privateuse` is always the last production in `langtag` and consumes the
            // rest of the tag.
            if !is_alphanumeric(subtag) {
                return Err(LanguageTagParseError::new(TagParseErrorKind::InvalidSubtag));
            }
            State::InPrivateUse { has_value: true }
        } else if subtag.eq_ignore_ascii_case(b"x") {
            if let State::InExtension {
                has_value: false, ..
            } = state
            {
                return Err(LanguageTagParseError::new(
                    TagParseErrorKind::EmptyExtension,
                ));
            }
            State::InPrivateUse { has_value: false }
        } else if subtag.len() == 1 && is_alphanumeric(subtag) {
            // A one-character subtag always opens a *new* extension singleton, even while
            // already inside another extension's values — this is what lets
            // `en-r-az-r-qt`-shaped tags (were the singleton not repeated) express two
            // extensions back to back with nothing but the grammar's own subtag-length cue to
            // separate them.
            if let State::InExtension {
                has_value: false, ..
            } = state
            {
                return Err(LanguageTagParseError::new(
                    TagParseErrorKind::EmptyExtension,
                ));
            }
            let singleton = subtag[0].to_ascii_lowercase();
            if seen_singletons.contains(&singleton) {
                return Err(LanguageTagParseError::new(
                    TagParseErrorKind::DuplicateSingleton,
                ));
            }
            seen_singletons.push(singleton);
            State::InExtension {
                singleton,
                has_value: false,
            }
        } else if let State::InExtension { singleton, .. } = state {
            if !is_alphanumeric(subtag) {
                return Err(LanguageTagParseError::new(TagParseErrorKind::InvalidSubtag));
            }
            extension_end = Some(end);
            State::InExtension {
                singleton,
                has_value: true,
            }
        } else {
            match state {
                // `State::Start` is handled above with top priority; it can never reach here.
                State::AfterShortLanguage { extlangs_seen }
                    if subtag.len() == 3 && is_alpha(subtag) =>
                {
                    let extlangs_seen = extlangs_seen + 1;
                    if extlangs_seen > 3 {
                        return Err(LanguageTagParseError::new(
                            TagParseErrorKind::TooManyExtlangs,
                        ));
                    }
                    extlang_end = Some(end);
                    State::AfterShortLanguage { extlangs_seen }
                }
                State::AfterShortLanguage { .. } | State::AfterLanguage
                    if subtag.len() == 4 && is_alpha(subtag) =>
                {
                    script_end = Some(end);
                    State::AfterScript
                }
                State::AfterShortLanguage { .. } | State::AfterLanguage | State::AfterScript
                    if (subtag.len() == 2 && is_alpha(subtag))
                        || (subtag.len() == 3 && is_digit(subtag)) =>
                {
                    region_end = Some(end);
                    State::AfterRegionOrVariant
                }
                State::AfterShortLanguage { .. }
                | State::AfterLanguage
                | State::AfterScript
                | State::AfterRegionOrVariant
                    if is_variant_shape(subtag) =>
                {
                    if seen_variants
                        .iter()
                        .any(|seen| seen.eq_ignore_ascii_case(subtag))
                    {
                        return Err(LanguageTagParseError::new(
                            TagParseErrorKind::DuplicateVariant,
                        ));
                    }
                    seen_variants.push(subtag);
                    variant_end = Some(end);
                    State::AfterRegionOrVariant
                }
                _ => return Err(LanguageTagParseError::new(TagParseErrorKind::InvalidSubtag)),
            }
        };
    }

    if let State::InExtension {
        has_value: false, ..
    } = state
    {
        return Err(LanguageTagParseError::new(
            TagParseErrorKind::EmptyExtension,
        ));
    }
    if let State::InPrivateUse { has_value: false } = state {
        return Err(LanguageTagParseError::new(
            TagParseErrorKind::EmptyPrivateUse,
        ));
    }

    Ok(TagElementsPositions {
        shape: TagShape::Standard,
        language_end,
        extlang_end,
        script_end,
        region_end,
        variant_end,
        extension_end,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(tag: &str) -> TagElementsPositions {
        parse_positions(tag.as_bytes()).unwrap_or_else(|e| panic!("{tag:?} should parse: {e}"))
    }

    fn err(tag: &str) {
        assert!(
            parse_positions(tag.as_bytes()).is_err(),
            "{tag:?} should be rejected"
        );
    }

    #[test]
    fn simple_language() {
        let p = ok("de");
        assert_eq!(p.language_end, 2);
        assert_eq!(p.extlang_end, None);
    }

    #[test]
    fn extlang_chain_up_to_three() {
        let p = ok("sl-abc-def-ghi");
        assert_eq!(p.extlang_end, Some(14));
        err("sl-abc-def-ghi-jkl");
    }

    #[test]
    fn long_primary_language_forbids_extlang() {
        err("abcd-efg");
    }

    #[test]
    fn script_region_variant_chain() {
        let p = ok("zh-Hans-CN-pinyin");
        assert_eq!(p.script_end, Some(7));
        assert_eq!(p.region_end, Some(10));
        assert_eq!(p.variant_end, Some(17));
    }

    #[test]
    fn duplicate_region_is_rejected() {
        err("de-419-DE");
    }

    #[test]
    fn single_letter_primary_language_is_rejected() {
        err("a-DE");
    }

    #[test]
    fn duplicate_singleton_is_rejected() {
        err("ar-a-aaa-b-bbb-a-ccc");
    }

    #[test]
    fn distinct_singletons_are_accepted() {
        ok("ar-a-aaa-b-bbb-c-ccc");
    }

    #[test]
    fn duplicate_variant_is_rejected() {
        err("de-1996-1996");
    }

    #[test]
    fn distinct_variants_are_accepted() {
        let p = ok("sl-rozaj-biske");
        assert_eq!(p.variant_end, Some(14));
    }

    #[test]
    fn empty_extension_is_rejected() {
        err("en-a-x-foo");
        err("en-a");
    }

    #[test]
    fn empty_privateuse_is_rejected() {
        err("en-x");
        err("x-");
    }

    #[test]
    fn privateuse_only_tag() {
        let p = parse_positions(b"x-fr-ch").expect("well-formed private use tag");
        assert_eq!(p.shape, TagShape::PrivateUseOnly);
        assert_eq!(p.language_end, 7);
    }

    #[test]
    fn grandfathered_tag_is_recognized() {
        let p = parse_positions(b"i-klingon").expect("grandfathered tag");
        assert_eq!(p.shape, TagShape::Grandfathered);
    }
}
