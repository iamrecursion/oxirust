//! [RFC 3986 §5.4](https://www.rfc-editor.org/rfc/rfc3986#section-5.4) reference resolution
//! examples, transcribed verbatim from the RFC text: the "normal" examples (§5.4.1) and the
//! "abnormal" examples (§5.4.2), both against the RFC's own base URI
//! `http://a/b/c/d;p?q`.
//!
//! This crate implements the *strict* resolver (§5.3), so `"http:g"` resolves to `"http:g"`
//! rather than the backward-compatible `"http://a/b/c/g"` some legacy parsers produce — this is
//! exactly what the RFC itself says a strict parser must do.
//!
//! Each case is additionally round-tripped through [`Iri::relativize`]: `base.resolve(&
//! base.relativize(&target)?)` must reproduce `target` exactly.

use oxixml_iri::{Iri, IriRef};

const BASE: &str = "http://a/b/c/d;p?q";

/// §5.4.1 "Normal Examples".
const NORMAL: &[(&str, &str)] = &[
    ("g:h", "g:h"),
    ("g", "http://a/b/c/g"),
    ("./g", "http://a/b/c/g"),
    ("g/", "http://a/b/c/g/"),
    ("/g", "http://a/g"),
    ("//g", "http://g"),
    ("?y", "http://a/b/c/d;p?y"),
    ("g?y", "http://a/b/c/g?y"),
    ("#s", "http://a/b/c/d;p?q#s"),
    ("g#s", "http://a/b/c/g#s"),
    ("g?y#s", "http://a/b/c/g?y#s"),
    (";x", "http://a/b/c/;x"),
    ("g;x", "http://a/b/c/g;x"),
    ("g;x?y#s", "http://a/b/c/g;x?y#s"),
    ("", "http://a/b/c/d;p?q"),
    (".", "http://a/b/c/"),
    ("./", "http://a/b/c/"),
    ("..", "http://a/b/"),
    ("../", "http://a/b/"),
    ("../g", "http://a/b/g"),
    ("../..", "http://a/"),
    ("../../", "http://a/"),
    ("../../g", "http://a/g"),
];

/// §5.4.2 "Abnormal Examples".
const ABNORMAL: &[(&str, &str)] = &[
    ("../../../g", "http://a/g"),
    ("../../../../g", "http://a/g"),
    ("/./g", "http://a/g"),
    ("/../g", "http://a/g"),
    ("g.", "http://a/b/c/g."),
    (".g", "http://a/b/c/.g"),
    ("g..", "http://a/b/c/g.."),
    ("..g", "http://a/b/c/..g"),
    ("./../g", "http://a/b/g"),
    ("./g/.", "http://a/b/c/g/"),
    ("g/./h", "http://a/b/c/g/h"),
    ("g/../h", "http://a/b/c/h"),
    ("g;x=1/./y", "http://a/b/c/g;x=1/y"),
    ("g;x=1/../y", "http://a/b/c/y"),
    ("g?y/./x", "http://a/b/c/g?y/./x"),
    ("g?y/../x", "http://a/b/c/g?y/../x"),
    ("g#s/./x", "http://a/b/c/g#s/./x"),
    ("g#s/../x", "http://a/b/c/g#s/../x"),
    ("http:g", "http:g"),
];

#[test]
fn normal_examples_resolve_exactly() -> Result<(), Box<dyn std::error::Error>> {
    let base = Iri::parse(BASE)?;
    for &(reference, expected) in NORMAL {
        let resolved = base.resolve(&IriRef::parse(reference)?)?;
        assert_eq!(
            resolved.as_str(),
            expected,
            "resolving {reference:?} against {BASE:?}"
        );
    }
    Ok(())
}

#[test]
fn abnormal_examples_resolve_exactly() -> Result<(), Box<dyn std::error::Error>> {
    let base = Iri::parse(BASE)?;
    for &(reference, expected) in ABNORMAL {
        let resolved = base.resolve(&IriRef::parse(reference)?)?;
        assert_eq!(
            resolved.as_str(),
            expected,
            "resolving {reference:?} against {BASE:?}"
        );
    }
    Ok(())
}

#[test]
fn all_examples_round_trip_through_relativize() -> Result<(), Box<dyn std::error::Error>> {
    let base = Iri::parse(BASE)?;
    for &(reference, expected) in NORMAL.iter().chain(ABNORMAL) {
        let target = base.resolve(&IriRef::parse(reference)?)?;
        assert_eq!(target.as_str(), expected);
        let relative = base.relativize(&target)?;
        let round_tripped = base.resolve(&relative)?;
        assert_eq!(
            round_tripped, target,
            "round-trip for target {expected:?} via relative {relative:?}"
        );
    }
    Ok(())
}

#[test]
fn strict_parser_does_not_apply_backward_compatible_scheme_leniency()
-> Result<(), Box<dyn std::error::Error>> {
    // A base URI with a "://" authority marker followed by a reference that merely shares the
    // base's scheme (`"http:g"`) is, per the strict RFC 3986 §5.3 algorithm, left completely
    // untouched: it already looks absolute, so the base contributes nothing.
    let base = Iri::parse(BASE)?;
    let resolved = base.resolve(&IriRef::parse("http:g")?)?;
    assert_eq!(resolved.as_str(), "http:g");
    assert_eq!(resolved.scheme(), "http");
    assert_eq!(resolved.authority(), None);
    assert_eq!(resolved.path(), "g");
    Ok(())
}
