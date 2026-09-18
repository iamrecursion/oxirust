//! Documentation-pinning tests: this crate is deliberately **non-normalizing**.
//!
//! RDF defines IRI term identity as exact string equality, so neither RFC 3986 §6.2.2
//! syntax-based normalization nor RFC 3987 §3.1 IDNA/ToASCII conversion is applied. These
//! tests exist so that the "What this crate does *not* do" section of the crate docs cannot
//! silently drift away from the implementation.

use oxixml_iri::{Iri, IriParseError, IriRef};

#[test]
fn scheme_and_host_case_is_preserved() -> Result<(), IriParseError> {
    let shouty = Iri::parse("HTTP://EX.COM/A")?;
    assert_eq!(shouty.as_str(), "HTTP://EX.COM/A");
    assert_eq!(shouty.scheme(), "HTTP");
    assert_eq!(shouty.authority(), Some("EX.COM"));

    // RFC 3986 §6.2.2.1 declares these equivalent; this crate keeps them distinct.
    assert_ne!(shouty, Iri::parse("http://ex.com/A")?);
    Ok(())
}

#[test]
fn percent_encoding_is_neither_case_folded_nor_decoded() -> Result<(), IriParseError> {
    assert_eq!(
        Iri::parse("http://ex.com/%2f")?.as_str(),
        "http://ex.com/%2f"
    );
    assert_ne!(
        Iri::parse("http://ex.com/%2f")?,
        Iri::parse("http://ex.com/%2F")?
    );
    // `~` is unreserved, so §6.2.2.2 would decode `%7E`; this crate does not.
    assert_eq!(
        Iri::parse("http://ex.com/%7E")?.as_str(),
        "http://ex.com/%7E"
    );
    assert_ne!(
        Iri::parse("http://ex.com/%7E")?,
        Iri::parse("http://ex.com/~")?
    );
    Ok(())
}

#[test]
fn dot_segments_survive_parsing_and_are_removed_only_by_resolve()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(
        Iri::parse("http://ex.com/a/./b/../c")?.as_str(),
        "http://ex.com/a/./b/../c"
    );

    // `resolve` — and only `resolve` — applies RFC 3986 §5.2.4 `remove_dot_segments`.
    let base = Iri::parse("http://ex.com/a/")?;
    assert_eq!(
        base.resolve(&IriRef::parse("./b/../c")?)?.as_str(),
        "http://ex.com/a/c"
    );
    Ok(())
}

#[test]
fn default_ports_and_empty_paths_are_kept_verbatim() -> Result<(), IriParseError> {
    assert_eq!(
        Iri::parse("http://ex.com:80/")?.as_str(),
        "http://ex.com:80/"
    );
    assert_ne!(
        Iri::parse("http://ex.com:80/")?,
        Iri::parse("http://ex.com/")?
    );
    assert_eq!(Iri::parse("http://ex.com")?.path(), "");
    Ok(())
}

#[test]
fn hosts_are_not_idna_converted() -> Result<(), IriParseError> {
    // No ToASCII: a Unicode host is validated but never Punycode-encoded, …
    let unicode_host = Iri::parse("http://\u{4f8b}.example/")?;
    assert_eq!(unicode_host.as_str(), "http://\u{4f8b}.example/");
    // … and an already-encoded host is never decoded, so the two stay distinct.
    assert_ne!(unicode_host, Iri::parse("http://xn--fsq.example/")?);
    Ok(())
}
