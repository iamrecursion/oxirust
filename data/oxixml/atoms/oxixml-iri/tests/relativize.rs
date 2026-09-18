//! [`Iri::relativize`] correctness: a corpus of `(base, target)` pairs, each checked to produce
//! an [`IriRef`] that [`Iri::resolve`]s back to exactly `target` — the contract `relativize`
//! documents. The corpus spans same-document references, shared-prefix path climbs, differing
//! queries/fragments, differing authorities and schemes, IPv6 hosts, percent-encoded and
//! non-ASCII (`ucschar`) content, and the handful of textually-confusable edge cases the
//! algorithm has to special-case (a leading path segment that could be misread as a scheme, a
//! path that could be misread as growing an authority, and so on).

use oxixml_iri::Iri;

/// `(base, target)` pairs. Every pair must round-trip:
/// `Iri::parse(base)?.resolve(&Iri::parse(base)?.relativize(&Iri::parse(target)?)?)? == target`.
const CORPUS: &[(&str, &str)] = &[
    // Identical IRI -> same-document reference.
    ("http://example.com/a/b", "http://example.com/a/b"),
    ("http://example.com/a/b?q", "http://example.com/a/b?q"),
    ("http://example.com/a/b?q#f", "http://example.com/a/b?q#f"),
    // Fragment-only difference.
    ("http://example.com/a/b", "http://example.com/a/b#frag"),
    ("http://example.com/a/b?q", "http://example.com/a/b?q#frag"),
    ("http://example.com/a/b#f1", "http://example.com/a/b#f2"),
    // Query-only difference.
    ("http://example.com/a/b", "http://example.com/a/b?q2"),
    ("http://example.com/a/b?q1", "http://example.com/a/b?q2"),
    ("http://example.com/a/b?q1", "http://example.com/a/b"),
    // Last-segment-only difference (siblings in the same directory).
    ("http://example.com/a/b", "http://example.com/a/c"),
    ("http://example.com/a/b", "http://example.com/a/c?q#f"),
    (
        "http://example.com/dir/file.html",
        "http://example.com/dir/other.html",
    ),
    // Same directory, target is the directory itself ("." case).
    ("http://example.com/a/b", "http://example.com/a/"),
    ("http://example.com/a/b/c", "http://example.com/a/b/"),
    // Deeper vs. shallower paths sharing a prefix.
    ("http://example.com/a/b/c", "http://example.com/a/b/c/d"),
    ("http://example.com/a/b/c/d", "http://example.com/a/b/c"),
    ("http://example.com/a/", "http://example.com/a/b/c/d"),
    ("http://example.com/a/b/c/d", "http://example.com/a/"),
    // Divergent paths deep in a shared prefix (needs "../" climbs).
    ("http://example.com/a/b/c/d", "http://example.com/a/b/x/y"),
    ("http://example.com/a/b/c/", "http://example.com/a/x/"),
    ("http://example.com/x/y/z/w", "http://example.com/x/y/other"),
    // No shared prefix beyond the root.
    ("http://example.com/a/b", "http://example.com/x/y"),
    ("http://example.com/a", "http://example.com/b"),
    // Root-only base or target.
    ("http://example.com/", "http://example.com/a"),
    ("http://example.com/a", "http://example.com/"),
    ("http://example.com/", "http://example.com/"),
    // Empty target path against a non-empty base.
    ("http://example.com/a/b", "http://example.com"),
    ("http://example.com/a/b?q", "http://example.com?q2"),
    // Different authority, same scheme.
    ("http://example.com/a/b", "http://other.example/a/b"),
    ("http://user@example.com/a", "http://example.com/a"),
    ("http://example.com:8080/a", "http://example.com:9090/a"),
    // Different scheme entirely.
    ("http://example.com/a/b", "https://example.com/a/b"),
    ("http://example.com/a", "urn:isbn:0451450523"),
    ("ftp://example.com/a", "http://example.com/a"),
    // Opaque (no-authority) IRIs.
    ("mailto:a@example.com", "mailto:b@example.com"),
    ("urn:isbn:0451450523", "urn:isbn:0451450524"),
    // First path segment that could be confused with a scheme once relativized (contains a
    // `:` before any `/`) -> must fall back to a form that cannot be misread.
    ("http://example.com/a/", "http://example.com/a/x:y"),
    ("http://example.com/", "http://example.com/a:b"),
    // Percent-encoded and reserved-looking path content.
    ("http://example.com/a/b%20c", "http://example.com/a/d%2Fe"),
    ("http://example.com/a/b;p=1", "http://example.com/a/b;p=2"),
    // Non-ASCII (`ucschar`) content, including a Japanese path/query/fragment.
    (
        "http://example.com/\u{65e5}\u{672c}/a",
        "http://example.com/\u{65e5}\u{672c}/b",
    ),
    (
        "http://example.com/dir/",
        "http://example.com/dir/\u{691c}\u{7d22}?\u{7d50}\u{679c}#\u{6ce8}",
    ),
    // IPv6 host authorities.
    ("http://[2001:db8::7]/a/b", "http://[2001:db8::7]/a/c"),
    ("http://[2001:db8::7]/a", "http://[2001:db8::1]/a"),
    // Matrix-parameter-style segments (semicolons are ordinary `pchar`, not separators).
    (
        "http://example.com/a/b;x=1/y",
        "http://example.com/a/b;x=1/z",
    ),
    // Long shared prefixes collapsing to a single climb.
    (
        "http://example.com/1/2/3/4/5/leaf",
        "http://example.com/1/2/3/4/5/other",
    ),
    (
        "http://example.com/1/2/3/4/5/leaf",
        "http://example.com/1/2/other",
    ),
];

#[test]
fn corpus_has_at_least_forty_pairs() {
    assert!(
        CORPUS.len() >= 40,
        "corpus has only {} pairs, want at least 40",
        CORPUS.len()
    );
}

#[test]
fn every_pair_round_trips() -> Result<(), Box<dyn std::error::Error>> {
    for &(base_text, target_text) in CORPUS {
        let base = Iri::parse(base_text)?;
        let target = Iri::parse(target_text)?;
        let relative = base
            .relativize(&target)
            .unwrap_or_else(|e| panic!("relativize({base_text:?}, {target_text:?}) failed: {e}"));
        let resolved = base.resolve(&relative).unwrap_or_else(|e| {
            panic!(
                "resolving {base_text:?} against its own relativize result {relative:?} \
                 (for target {target_text:?}) failed: {e}"
            )
        });
        assert_eq!(
            resolved.as_str(),
            target_text,
            "base={base_text:?} target={target_text:?} relative={relative:?}"
        );
    }
    Ok(())
}

#[test]
fn identical_iri_relativizes_to_a_same_document_reference() -> Result<(), Box<dyn std::error::Error>>
{
    let iri = Iri::parse("http://example.com/a/b?q#f")?;
    let relative = iri.relativize(&iri)?;
    // A same-document reference resolves back to exactly the same IRI without re-deriving the
    // path from scratch.
    assert_eq!(iri.resolve(&relative)?, iri);
    Ok(())
}

#[test]
fn relativize_rejects_targets_with_dot_segments() -> Result<(), Box<dyn std::error::Error>> {
    let base = Iri::parse("http://example.com/a/b")?;
    // A literal (unresolved) ".." or "." segment in the *target* IRI's path cannot be
    // reproduced by any relative reference, because resolution would eat it.
    let target = Iri::parse_unchecked("http://example.com/a/../b");
    assert!(base.relativize(&target).is_err());
    Ok(())
}

#[test]
fn different_scheme_falls_back_to_the_full_target() -> Result<(), Box<dyn std::error::Error>> {
    let base = Iri::parse("http://example.com/a")?;
    let target = Iri::parse("https://example.com/a")?;
    let relative = base.relativize(&target)?;
    assert_eq!(relative.as_str(), "https://example.com/a");
    Ok(())
}
