//! Grammar-level parsing and validation tests: valid IRIs (including IPv6/IPvFuture hosts,
//! percent-encoding, and non-ASCII `ucschar`/`iprivate` content) and invalid inputs, checked
//! against the specific [`IriParseError`] each should produce.

use oxixml_iri::{Iri, IriParseError, IriRef};

fn assert_iri_ok(text: &str) -> Iri<&str> {
    match Iri::parse(text) {
        Ok(iri) => iri,
        Err(e) => panic!("expected {text:?} to parse as a valid Iri, got error: {e}"),
    }
}

fn assert_iri_err(text: &str) -> IriParseError {
    match Iri::parse(text) {
        Ok(iri) => panic!("expected {text:?} to be rejected, parsed as {iri:?}"),
        Err(e) => e,
    }
}

fn assert_iri_ref_ok(text: &str) -> IriRef<&str> {
    match IriRef::parse(text) {
        Ok(iri) => iri,
        Err(e) => panic!("expected {text:?} to parse as a valid IriRef, got error: {e}"),
    }
}

fn assert_iri_ref_err(text: &str) -> IriParseError {
    match IriRef::parse(text) {
        Ok(iri) => panic!("expected {text:?} to be rejected, parsed as {iri:?}"),
        Err(e) => e,
    }
}

// ---------------------------------------------------------------------------------------------
// Valid: scheme
// ---------------------------------------------------------------------------------------------

#[test]
fn valid_schemes() {
    for text in [
        "a:b",
        "http://example.com/",
        "https://example.com/",
        "ftp://example.com/",
        "urn:isbn:0451450523",
        "mailto:user@example.com",
        "tag:example.com,2024:foo",
        "x-custom+scheme.v2://host/",
        "A://host/", // scheme case is preserved, not normalized
    ] {
        assert_iri_ok(text);
    }
}

#[test]
fn scheme_case_is_preserved_not_normalized() {
    let iri = assert_iri_ok("hTTp://example.com");
    assert_eq!(iri.scheme(), "hTTp");
}

// ---------------------------------------------------------------------------------------------
// Invalid: scheme
// ---------------------------------------------------------------------------------------------

#[test]
fn empty_scheme_is_rejected() {
    // No `:` at all before path/query/fragment content -> treated as a relative reference,
    // which `Iri::parse` (unlike `IriRef::parse`) always rejects for lacking a scheme.
    let error = assert_iri_err("//example.com/path");
    assert_eq!(error.to_string(), "no scheme found in an absolute IRI");
}

#[test]
fn scheme_must_start_with_a_letter() {
    let error = assert_iri_err("1http://example.com/");
    assert!(error.to_string().contains("scheme"), "{error}");
}

#[test]
fn scheme_rejects_invalid_characters() {
    for text in ["ht tp://example.com/", "ht@tp://example.com/", "h_ttp://x/"] {
        let error = assert_iri_err(text);
        assert!(error.to_string().contains("scheme"), "{text}: {error}");
    }
}

// ---------------------------------------------------------------------------------------------
// Invalid: generic malformed input
// ---------------------------------------------------------------------------------------------

#[test]
fn raw_space_is_rejected() {
    assert_iri_err("http://example.com/a b");
    assert_iri_err("http://example.com/a\tb");
}

#[test]
fn raw_control_characters_are_rejected() {
    assert_iri_err("http://example.com/a\u{0}b");
    assert_iri_err("http://example.com/a\u{7f}b"); // DEL
}

#[test]
fn truncated_percent_escape_is_rejected() {
    let error = assert_iri_err("http://example.com/a%2");
    assert!(error.to_string().contains("percent-encoding"), "{error}");
}

#[test]
fn invalid_percent_hex_digit_is_rejected() {
    let error = assert_iri_err("http://example.com/a%zz");
    assert!(error.to_string().contains("percent-encoding"), "{error}");
}

#[test]
fn unmatched_host_bracket_is_rejected() {
    let error = assert_iri_err("http://[::1/path");
    assert_eq!(
        error.to_string(),
        "'[' bracket must be paired with a closing one in host"
    );
}

// ---------------------------------------------------------------------------------------------
// Valid / invalid: IPv6 and IPvFuture hosts
// ---------------------------------------------------------------------------------------------

#[test]
fn ipv6_hosts_are_accepted() {
    for text in [
        "http://[2001:db8::7]/c=GB?objectClass?one",
        "http://[::1]/",
        "http://[::]/",
        "http://[2001:0db8:0000:0000:0000:ff00:0042:8329]/",
        "http://[2001:db8::7]:8080/path",
        "ldap://[2001:db8::7]/c=GB?objectClass?one",
    ] {
        assert_iri_ok(text);
    }
}

#[test]
fn ipv6_hosts_reject_malformed_addresses() {
    for text in [
        "http://[gggg::1]/",
        "http://[1:2:3:4:5:6:7:8:9]/",
        "http://[12345::1]/",
    ] {
        let error = assert_iri_err(text);
        assert!(
            error.to_string().contains("invalid host IP"),
            "{text}: {error}"
        );
    }
}

#[test]
fn ip_v_future_hosts_are_accepted() {
    for text in ["http://[v1.fe80::1]/", "http://[vAB.some-token:more]/"] {
        assert_iri_ok(text);
    }
}

#[test]
fn ip_v_future_hosts_reject_malformed_syntax() {
    for text in [
        "http://[v.fe80::1]/", // no hex digits before "."
        "http://[v1fe80::1]/", // missing "."
        "http://[v1.]/",       // empty tail
    ] {
        assert_iri_err(text);
    }
}

// ---------------------------------------------------------------------------------------------
// Valid: userinfo, port
// ---------------------------------------------------------------------------------------------

#[test]
fn userinfo_and_port_are_accepted() {
    let iri = assert_iri_ok("http://user:pass@example.com:8080/path");
    assert_eq!(iri.authority(), Some("user:pass@example.com:8080"));
}

#[test]
fn empty_port_is_accepted() {
    // RFC 3986's `port` production is `*DIGIT` (zero or more).
    assert_iri_ok("http://example.com:/path");
}

#[test]
fn non_digit_port_is_rejected() {
    let error = assert_iri_err("http://example.com:abc/path");
    assert!(
        error.to_string().contains("port") || error.to_string().contains("character"),
        "{error}"
    );
}

// ---------------------------------------------------------------------------------------------
// Valid: percent-encoding and non-ASCII (ucschar / iprivate) content
// ---------------------------------------------------------------------------------------------

#[test]
fn percent_encoded_content_is_accepted_everywhere() {
    assert_iri_ok("http://example.com/a%20b?q=%2Fx#%23frag");
    assert_iri_ok("http://user%40name@example.com/");
}

#[test]
fn japanese_text_is_accepted_as_ucschar() {
    // "日本語" (Japanese) directly in the path, query, and fragment, unescaped.
    let iri = assert_iri_ok(
        "http://example.com/\u{65e5}\u{672c}\u{8a9e}?q=\u{691c}\u{7d22}#\u{7d50}\u{679c}",
    );
    assert_eq!(iri.path(), "/\u{65e5}\u{672c}\u{8a9e}");
    assert_eq!(iri.query(), Some("q=\u{691c}\u{7d22}"));
    assert_eq!(iri.fragment(), Some("\u{7d50}\u{679c}"));
}

#[test]
fn japanese_text_is_accepted_in_host_as_ucschar() {
    // A non-ASCII (IDN-style) reg-name host.
    assert_iri_ok("http://\u{4f8b}.example/");
}

#[test]
fn iprivate_range_is_accepted_only_in_query() {
    // U+E000 is in the `iprivate` range: legal, unescaped, in the query...
    assert_iri_ok("http://example.com/path?\u{e000}");
    // ...but *not* directly in the path or fragment (RFC 3987's `ipath`/`ifragment`
    // productions do not include `iprivate`).
    assert_iri_err("http://example.com/\u{e000}");
    assert_iri_err("http://example.com/path#\u{e000}");
}

// ---------------------------------------------------------------------------------------------
// Relative references (IriRef only)
// ---------------------------------------------------------------------------------------------

#[test]
fn iri_ref_accepts_all_reference_forms() {
    for text in [
        "",
        "#frag",
        "?q",
        "//authority/path",
        "/abs/path",
        "rel/path",
        "../rel",
        ".",
    ] {
        assert_iri_ref_ok(text);
    }
}

#[test]
fn iri_ref_is_absolute_matches_scheme_presence() {
    assert!(assert_iri_ref_ok("http://example.com/").is_absolute());
    assert!(!assert_iri_ref_ok("//example.com/").is_absolute());
    assert!(!assert_iri_ref_ok("/path").is_absolute());
    assert!(!assert_iri_ref_ok("").is_absolute());
}

#[test]
fn iri_rejects_relative_references() {
    for text in [
        "",
        "#frag",
        "?q",
        "//authority/path",
        "/abs/path",
        "rel/path",
    ] {
        let error = assert_iri_err(text);
        assert_eq!(error.to_string(), "no scheme found in an absolute IRI");
    }
}

#[test]
fn iri_ref_still_validates_its_components() {
    assert_iri_ref_err("/a b");
    assert_iri_ref_err("?a%2");
    assert_iri_ref_err("#a\u{0}");
}

// ---------------------------------------------------------------------------------------------
// Component accessors
// ---------------------------------------------------------------------------------------------

#[test]
fn component_accessors_split_correctly() -> Result<(), Box<dyn std::error::Error>> {
    let iri = Iri::parse("http://user:pass@example.com:8080/a/b;p?q=1&r=2#frag")?;
    assert_eq!(iri.scheme(), "http");
    assert_eq!(iri.authority(), Some("user:pass@example.com:8080"));
    assert_eq!(iri.path(), "/a/b;p");
    assert_eq!(iri.query(), Some("q=1&r=2"));
    assert_eq!(iri.fragment(), Some("frag"));

    let opaque = Iri::parse("mailto:foo@bar.com")?;
    assert_eq!(opaque.scheme(), "mailto");
    assert_eq!(opaque.authority(), None);
    assert_eq!(opaque.path(), "foo@bar.com");
    assert_eq!(opaque.query(), None);
    assert_eq!(opaque.fragment(), None);
    Ok(())
}

#[test]
fn empty_query_and_fragment_are_distinguished_from_absent() -> Result<(), Box<dyn std::error::Error>>
{
    let no_query_no_frag = Iri::parse("http://example.com/path")?;
    assert_eq!(no_query_no_frag.query(), None);
    assert_eq!(no_query_no_frag.fragment(), None);

    let empty_query = Iri::parse("http://example.com/path?")?;
    assert_eq!(empty_query.query(), Some(""));

    let empty_fragment = Iri::parse("http://example.com/path#")?;
    assert_eq!(empty_fragment.fragment(), Some(""));
    Ok(())
}
