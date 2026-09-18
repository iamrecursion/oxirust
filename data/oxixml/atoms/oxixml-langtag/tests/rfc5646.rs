//! Integration tests driven directly off [RFC 5646](https://www.rfc-editor.org/rfc/rfc5646):
//! Appendix A's worked examples, the fixed grandfathered-tag list, well-formedness edge cases
//! from §2.1, and the §2.1.1 case-normalization algorithm.

use oxixml_langtag::LanguageTag;

/// RFC 5646 Appendix A, "Examples of Language Tags" — every tag listed there as well-formed.
#[test]
fn rfc5646_appendix_a_well_formed_examples() {
    let tags = [
        // Simple language subtag.
        "de",
        "fr",
        "ja",
        "i-enochian", // grandfathered
        // Language + script.
        "zh-Hant",
        "zh-Hans",
        "sr-Cyrl",
        "sr-Latn",
        // Extended language subtags and their primary-language-subtag counterparts.
        "zh-cmn-Hans-CN",
        "cmn-Hans-CN",
        "zh-yue-HK",
        "yue-HK",
        // Language-Script-Region.
        "zh-Hans-CN",
        "sr-Latn-RS",
        // Language-Variant.
        "sl-rozaj",
        "sl-rozaj-biske",
        "sl-nedis",
        // Language-Region-Variant.
        "de-CH-1901",
        "sl-IT-nedis",
        // Language-Script-Region-Variant.
        "hy-Latn-IT-arevela",
        // Language-Region.
        "de-DE",
        "en-US",
        "es-419",
        // Private use subtags.
        "de-CH-x-phonebk",
        "az-Arab-x-AZE-derbend",
        // Private use registry values.
        "x-whatever",
        "qaa-Qaaa-QM-x-southern",
        "de-Qaaa",
        "sr-Latn-QM",
        "sr-Qaaa-RS",
        // Tags that use extensions.
        "en-US-u-islamcal",
        "zh-CN-a-myext-x-private",
        "en-a-myext-b-another",
    ];
    for tag in tags {
        LanguageTag::parse(tag).unwrap_or_else(|e| panic!("{tag:?} must be well-formed: {e}"));
    }
}

/// RFC 5646 Appendix A, "Some Invalid Tags" — every tag listed there as invalid.
#[test]
fn rfc5646_appendix_a_invalid_examples() {
    let tags = [
        "de-419-DE",            // two region tags
        "a-DE",                 // single-character subtag in primary position
        "ar-a-aaa-b-bbb-a-ccc", // two extensions with the same single-letter prefix
    ];
    for tag in tags {
        assert!(LanguageTag::parse(tag).is_err(), "{tag:?} must be rejected");
    }
}

/// RFC 5646 §2.1: the 26 `grandfathered` tags (9 `regular`, 17 `irregular`), parsed in their
/// canonical casing, in all-lowercase, and in all-uppercase.
#[test]
fn grandfathered_tags_parse_in_any_case() {
    let tags = [
        "art-lojban",
        "cel-gaulish",
        "en-GB-oed",
        "i-ami",
        "i-bnn",
        "i-default",
        "i-enochian",
        "i-hak",
        "i-klingon",
        "i-lux",
        "i-mingo",
        "i-navajo",
        "i-pwn",
        "i-tao",
        "i-tay",
        "i-tsu",
        "no-bok",
        "no-nyn",
        "sgn-BE-FR",
        "sgn-BE-NL",
        "sgn-CH-DE",
        "zh-guoyu",
        "zh-hakka",
        "zh-min",
        "zh-min-nan",
        "zh-xiang",
    ];
    for tag in tags {
        let parsed = LanguageTag::parse(tag).unwrap_or_else(|e| panic!("{tag:?} must parse: {e}"));
        assert_eq!(parsed.full_language(), tag);
        assert_eq!(parsed.primary_language(), tag);

        let normalized_from_lower = LanguageTag::parse_and_normalize(&tag.to_lowercase())
            .unwrap_or_else(|e| panic!("{tag:?} (lowercase) must normalize: {e}"));
        assert_eq!(normalized_from_lower.as_str(), tag);

        let normalized_from_upper = LanguageTag::parse_and_normalize(&tag.to_uppercase())
            .unwrap_or_else(|e| panic!("{tag:?} (uppercase) must normalize: {e}"));
        assert_eq!(normalized_from_upper.as_str(), tag);
    }
}

/// `i-notexist` matches the `irregular` production's shape (`i-` plus letters) but is not one
/// of the fixed listed grandfathered tags, so it must be rejected rather than silently accepted
/// as some other production (a 1-letter `i` primary language subtag is also invalid).
#[test]
fn unlisted_i_dash_tag_is_rejected() {
    assert!(LanguageTag::parse("i-notexist").is_err());
}

/// Extension sequences: multiple singletons, multi-value singletons, and the extension example
/// straight from RFC 5646 §3.7's discussion of extensions (`en-a-bbb-x-a-ccc`, where the second
/// `a` is inside the private-use tail and is *not* a repeated singleton).
#[test]
fn extension_components() {
    let tag = LanguageTag::parse("en-a-bbb-x-a-ccc").expect("well-formed");
    assert_eq!(tag.extension(), Some("a-bbb"));
    assert_eq!(tag.private_use(), Some("x-a-ccc"));

    let tag = LanguageTag::parse("en-a-babble-b-warble").expect("well-formed");
    assert_eq!(
        tag.extension_subtags().collect::<Vec<_>>(),
        vec![('a', "babble"), ('b', "warble")]
    );

    let tag = LanguageTag::parse("de-DE-u-co-phonebk").expect("well-formed");
    assert_eq!(tag.extension(), Some("u-co-phonebk"));
    assert_eq!(
        tag.extension_subtags().collect::<Vec<_>>(),
        vec![('u', "co-phonebk")]
    );
}

/// Private-use components, both as the tail of an ordinary tag and as a standalone tag.
#[test]
fn private_use_components() {
    let tag = LanguageTag::parse("en-x-US").expect("well-formed");
    assert_eq!(tag.private_use(), Some("x-US"));
    assert_eq!(tag.private_use_subtags().collect::<Vec<_>>(), vec!["US"]);

    let tag = LanguageTag::parse("x-fr-ch").expect("well-formed");
    assert_eq!(tag.primary_language(), "x-fr-ch");
    assert_eq!(tag.private_use(), Some("x-fr-ch"));
    assert_eq!(
        tag.private_use_subtags().collect::<Vec<_>>(),
        vec!["fr", "ch"]
    );

    let tag = LanguageTag::parse("es-x-foobar-AT-007").expect("well-formed");
    assert_eq!(
        tag.private_use_subtags().collect::<Vec<_>>(),
        vec!["foobar", "AT", "007"]
    );
}

/// §2.1.1 case normalization: `mn-Cyrl-MN` regardless of the input casing.
#[test]
fn normalize_mn_cyrl_mn_in_any_case() {
    for input in ["mn-Cyrl-MN", "MN-cYRL-mn", "mN-cYrL-Mn", "mn-cyrl-mn"] {
        let normalized = LanguageTag::parse_and_normalize(input)
            .unwrap_or_else(|e| panic!("{input:?} must normalize: {e}"));
        assert_eq!(normalized.as_str(), "mn-Cyrl-MN");
    }
}

/// A broader normalization sweep: language/extlang lowercase, script Titlecase, region
/// UPPERCASE, and everything after the region (variants, extensions, private use) lowercase.
#[test]
fn normalize_every_component() {
    let cases = [
        ("en-CA-x-ca", "en-CA-x-ca"),
        ("SL-AFB-lATN-005-nEdis", "sl-afb-Latn-005-nedis"),
        ("az-Latn-x-latn", "az-Latn-x-latn"),
        ("AR-ARB-LATN-DE-NEDIS-FOOBAR", "ar-arb-Latn-DE-nedis-foobar"),
        ("ar-arb-latn-de-nedis-foobar", "ar-arb-Latn-DE-nedis-foobar"),
        ("DE-DE-U-CO-PHONEBK", "de-DE-u-co-phonebk"),
        ("ZH-CMN-HANS-CN", "zh-cmn-Hans-CN"),
    ];
    for (input, expected) in cases {
        let normalized = LanguageTag::parse_and_normalize(input)
            .unwrap_or_else(|e| panic!("{input:?} must normalize: {e}"));
        assert_eq!(normalized.as_str(), expected, "normalizing {input:?}");
    }
}

/// `parse` never normalizes: the original casing (and, for `FromStr`/`parse_and_normalize`, the
/// normalized casing) round-trips through `as_str`.
#[test]
fn parse_preserves_casing_parse_and_normalize_does_not() {
    let raw = LanguageTag::parse("EN-us").expect("well-formed");
    assert_eq!(raw.as_str(), "EN-us");

    let normalized = LanguageTag::parse_and_normalize("EN-us").expect("well-formed");
    assert_eq!(normalized.as_str(), "en-US");
}

/// Ill-formed tags beyond the three Appendix A calls out by name: a broader sweep across every
/// grammar production, covering shape violations, length violations, and empty-subtag shapes.
#[test]
fn ill_formed_tags_are_rejected() {
    let tags = [
        "",                     // empty
        "f",                    // primary language too short
        "f-Latn",               // primary language too short
        "fr-Latn-F",            // trailing single-letter subtag opens an extension with no value
        "a-value",              // single-letter primary language
        "tlh-a-b-foo",          // extension singleton 'a' has no value before 'b' starts a new one
        "abcdefghi-012345678",  // subtag over eight characters
        "ab-abc-abc-abc-abc",   // more than three extlang subtags
        "ab-abcd-abc",          // subtag not valid after script
        "ab-ab-abc",            // 2-letter region has no room for a following 3-letter subtag
        "ab-123-abc",           // 3-digit region has no room for a following subtag
        "a1-Hant-ZH",           // primary language must be letters only
        "ab-a-b",               // empty extension value
        "ab-a-x",               // empty extension value before private use
        "ab--ab",               // doubled hyphen
        "ab-abc-",              // trailing hyphen
        "-ab-abc",              // leading hyphen
        "abcd-efg", // 4-letter (reserved) primary language cannot take extlang-shaped subtags
        "de-1996-1996", // duplicate variant
        "ar-a-aaa-b-bbb-a-ccc", // duplicate extension singleton
        "x-",       // private use with no value
        "x-foo-",   // private use with trailing hyphen
        "x--foo",   // private use with doubled hyphen
        "x-abcdefghijkl", // private use value over eight characters
        "en-a-",    // trailing hyphen after an extension singleton
    ];
    for tag in tags {
        assert!(
            LanguageTag::parse(tag).is_err(),
            "{tag:?} should be rejected as ill-formed"
        );
    }
}

/// `LanguageTag<&str>`, `LanguageTag<String>`, and `LanguageTag<Box<str>>` all parse the same
/// input the same way, and compare equal across storage types.
#[test]
fn parses_over_every_deref_str_storage() {
    let borrowed = LanguageTag::parse("en-US").expect("well-formed");
    let owned = LanguageTag::parse("en-US".to_string()).expect("well-formed");
    let boxed = LanguageTag::parse("en-US".to_string().into_boxed_str()).expect("well-formed");
    assert_eq!(borrowed, owned);
    assert_eq!(borrowed.as_str(), boxed.as_str());
}
