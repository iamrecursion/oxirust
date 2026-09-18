//! The direct (non-delegating, non-`nextCatalog`) resolution algorithm:
//! direct matches, `prefer` scoping, and rewrite/suffix longest-match rules.

use oxixml_catalog::{Catalog, CatalogError, Prefer};

#[test]
fn direct_public_system_and_uri_matches() -> Result<(), CatalogError> {
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
      <public publicId="-//Example//DTD Foo//EN" uri="public-foo.dtd"/>
      <system systemId="urn:example:foo" uri="system-foo.dtd"/>
      <uri name="urn:example:bar" uri="uri-bar.xml"/>
    </catalog>
    "#;
    let catalog = Catalog::parse(xml, "http://example.org/catalog.xml")?;
    assert_eq!(
        catalog
            .resolve_public("-//Example//DTD Foo//EN", None)
            .as_deref(),
        Some("http://example.org/public-foo.dtd")
    );
    assert_eq!(
        catalog.resolve_system("urn:example:foo").as_deref(),
        Some("http://example.org/system-foo.dtd")
    );
    assert_eq!(
        catalog.resolve_uri("urn:example:bar").as_deref(),
        Some("http://example.org/uri-bar.xml")
    );
    assert_eq!(catalog.resolve_system("urn:example:no-such-id"), None);
    Ok(())
}

#[test]
fn public_id_normalization_collapses_whitespace() -> Result<(), CatalogError> {
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
      <public publicId="-//Example//DTD  Foo//EN" uri="foo.dtd"/>
    </catalog>
    "#;
    let catalog = Catalog::parse(xml, "http://example.org/catalog.xml")?;
    // Extra internal whitespace and leading/trailing space must not matter.
    assert_eq!(
        catalog
            .resolve_public("  -//Example//DTD Foo//EN  ", None)
            .as_deref(),
        Some("http://example.org/foo.dtd")
    );
    Ok(())
}

#[test]
fn system_match_wins_even_when_prefer_is_public() -> Result<(), CatalogError> {
    // A direct `system` match always wins over the public path, regardless of
    // `prefer`: prefer only ever governs whether `public`/`delegatePublic`
    // entries are *considered* when a system identifier is present.
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog" prefer="public">
      <public publicId="-//Example//DTD Foo//EN" uri="public-foo.dtd"/>
      <system systemId="urn:example:foo" uri="system-foo.dtd"/>
    </catalog>
    "#;
    let catalog = Catalog::parse(xml, "http://example.org/catalog.xml")?;
    assert_eq!(
        catalog
            .resolve_public("-//Example//DTD Foo//EN", Some("urn:example:foo"))
            .as_deref(),
        Some("http://example.org/system-foo.dtd")
    );
    Ok(())
}

#[test]
fn prefer_system_ignores_public_entry_when_system_id_present() -> Result<(), CatalogError> {
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog" prefer="system">
      <public publicId="-//Example//DTD Foo//EN" uri="public-foo.dtd"/>
    </catalog>
    "#;
    let catalog = Catalog::parse(xml, "http://example.org/catalog.xml")?;
    // No `system` entry matches `urn:example:unmatched`, but because
    // `prefer="system"` and a system identifier *was* supplied, the public
    // entry must be ignored rather than used as a fallback.
    assert_eq!(
        catalog.resolve_public("-//Example//DTD Foo//EN", Some("urn:example:unmatched")),
        None
    );
    // Without a system identifier at all, the public entry is still used.
    assert_eq!(
        catalog
            .resolve_public("-//Example//DTD Foo//EN", None)
            .as_deref(),
        Some("http://example.org/public-foo.dtd")
    );
    Ok(())
}

#[test]
fn prefer_public_uses_public_entry_even_with_unmatched_system_id() -> Result<(), CatalogError> {
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog" prefer="public">
      <public publicId="-//Example//DTD Foo//EN" uri="public-foo.dtd"/>
    </catalog>
    "#;
    let catalog = Catalog::parse(xml, "http://example.org/catalog.xml")?;
    assert_eq!(
        catalog
            .resolve_public("-//Example//DTD Foo//EN", Some("urn:example:unmatched"))
            .as_deref(),
        Some("http://example.org/public-foo.dtd")
    );
    Ok(())
}

#[test]
fn default_prefer_strategy_is_configurable() -> Result<(), CatalogError> {
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
      <public publicId="-//Example//DTD Foo//EN" uri="public-foo.dtd"/>
    </catalog>
    "#;
    let public_default = Catalog::parse(xml, "http://example.org/catalog.xml")?;
    assert_eq!(
        public_default
            .resolve_public("-//Example//DTD Foo//EN", Some("urn:example:unmatched"))
            .as_deref(),
        Some("http://example.org/public-foo.dtd")
    );

    let system_default =
        Catalog::parse_with_prefer(xml, "http://example.org/catalog.xml", Prefer::System)?;
    assert_eq!(
        system_default.resolve_public("-//Example//DTD Foo//EN", Some("urn:example:unmatched")),
        None
    );
    Ok(())
}

#[test]
fn rewrite_system_uses_longest_matching_prefix() -> Result<(), CatalogError> {
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
      <rewriteSystem systemIdStartString="http://example.com/" rewritePrefix="short/"/>
      <rewriteSystem systemIdStartString="http://example.com/sub/" rewritePrefix="long/"/>
    </catalog>
    "#;
    let catalog = Catalog::parse(xml, "http://example.org/catalog.xml")?;
    assert_eq!(
        catalog
            .resolve_system("http://example.com/sub/foo.dtd")
            .as_deref(),
        Some("http://example.org/long/foo.dtd")
    );
    assert_eq!(
        catalog
            .resolve_system("http://example.com/other/foo.dtd")
            .as_deref(),
        Some("http://example.org/short/other/foo.dtd")
    );
    Ok(())
}

#[test]
fn rewrite_uri_uses_longest_matching_prefix() -> Result<(), CatalogError> {
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
      <rewriteURI uriStartString="http://example.com/" rewritePrefix="short/"/>
      <rewriteURI uriStartString="http://example.com/sub/" rewritePrefix="long/"/>
    </catalog>
    "#;
    let catalog = Catalog::parse(xml, "http://example.org/catalog.xml")?;
    assert_eq!(
        catalog
            .resolve_uri("http://example.com/sub/foo.xsd")
            .as_deref(),
        Some("http://example.org/long/foo.xsd")
    );
    Ok(())
}

#[test]
fn direct_system_match_wins_over_rewrite_system() -> Result<(), CatalogError> {
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
      <rewriteSystem systemIdStartString="http://example.com/" rewritePrefix="rewritten/"/>
      <system systemId="http://example.com/exact.dtd" uri="exact.dtd"/>
    </catalog>
    "#;
    let catalog = Catalog::parse(xml, "http://example.org/catalog.xml")?;
    assert_eq!(
        catalog
            .resolve_system("http://example.com/exact.dtd")
            .as_deref(),
        Some("http://example.org/exact.dtd")
    );
    Ok(())
}

#[test]
fn system_suffix_uses_longest_matching_suffix_and_is_a_direct_replacement()
-> Result<(), CatalogError> {
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
      <systemSuffix systemIdSuffix=".dtd" uri="generic.dtd"/>
      <systemSuffix systemIdSuffix="foo.dtd" uri="specific-foo.dtd"/>
    </catalog>
    "#;
    let catalog = Catalog::parse(xml, "http://example.org/catalog.xml")?;
    // Longer suffix wins: the whole `uri` attribute is the replacement, it is
    // not spliced onto anything (unlike `rewriteSystem`).
    assert_eq!(
        catalog.resolve_system("anywhere/at/all/foo.dtd").as_deref(),
        Some("http://example.org/specific-foo.dtd")
    );
    assert_eq!(
        catalog.resolve_system("anywhere/at/all/bar.dtd").as_deref(),
        Some("http://example.org/generic.dtd")
    );
    Ok(())
}

#[test]
fn uri_suffix_uses_longest_matching_suffix() -> Result<(), CatalogError> {
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
      <uriSuffix uriSuffix=".xsd" uri="generic.xsd"/>
      <uriSuffix uriSuffix="/special.xsd" uri="specific.xsd"/>
    </catalog>
    "#;
    let catalog = Catalog::parse(xml, "http://example.org/catalog.xml")?;
    assert_eq!(
        catalog
            .resolve_uri("http://elsewhere.example/special.xsd")
            .as_deref(),
        Some("http://example.org/specific.xsd")
    );
    assert_eq!(
        catalog
            .resolve_uri("http://elsewhere.example/other.xsd")
            .as_deref(),
        Some("http://example.org/generic.xsd")
    );
    Ok(())
}

#[test]
fn uri_normalization_matches_percent_encoded_form() -> Result<(), CatalogError> {
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
      <uri name="http://example.com/a%20b" uri="space.xml"/>
    </catalog>
    "#;
    let catalog = Catalog::parse(xml, "http://example.org/catalog.xml")?;
    // A raw space in the queried URI must normalize to the same %20 form
    // used (already pre-escaped) in the catalog document.
    assert_eq!(
        catalog.resolve_uri("http://example.com/a b").as_deref(),
        Some("http://example.org/space.xml")
    );
    Ok(())
}

// ------------------------------------------------------------- §6.4 URN "unwrapping"

#[test]
fn publicid_urn_system_identifier_is_unwrapped_to_a_public_identifier() -> Result<(), CatalogError>
{
    // OASIS XML Catalogs 1.1 §6.4 / §7.1.1 step 1: a system identifier written as an RFC 3151
    // `urn:publicid:` URN *is* a public identifier and must be matched as one.
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
      <public publicId="-//OASIS//DTD DocBook XML V4.5//EN" uri="docbook.dtd"/>
    </catalog>
    "#;
    let catalog = Catalog::parse(xml, "file:///etc/xml/catalog.xml")?;
    let direct = catalog
        .resolve_public("-//OASIS//DTD DocBook XML V4.5//EN", None)
        .expect("the plain public identifier resolves");
    assert_eq!(direct, "file:///etc/xml/docbook.dtd");

    assert_eq!(
        catalog
            .resolve_system("urn:publicid:-:OASIS:DTD+DocBook+XML+V4.5:EN")
            .as_deref(),
        Some(direct.as_str())
    );
    assert_eq!(
        catalog
            .resolve_uri("urn:publicid:-:OASIS:DTD+DocBook+XML+V4.5:EN")
            .as_deref(),
        Some(direct.as_str())
    );
    Ok(())
}

#[test]
fn publicid_urn_scheme_is_matched_case_insensitively() -> Result<(), CatalogError> {
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
      <public publicId="-//Example//DTD Foo//EN" uri="foo.dtd"/>
    </catalog>
    "#;
    let catalog = Catalog::parse(xml, "http://example.org/catalog.xml")?;
    assert_eq!(
        catalog
            .resolve_system("URN:PUBLICID:-:Example:DTD+Foo:EN")
            .as_deref(),
        Some("http://example.org/foo.dtd")
    );
    Ok(())
}

#[test]
fn publicid_urn_decodes_rfc_3151_escapes() -> Result<(), CatalogError> {
    // RFC 3151: `;` encodes `::`, `%2B` an actual `+`, `%3A` an actual `:` and `%2F` a `/`.
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
      <public publicId="-//Example//TEXT a+b::c:d/e//EN" uri="odd.dtd"/>
    </catalog>
    "#;
    let catalog = Catalog::parse(xml, "http://example.org/catalog.xml")?;
    assert_eq!(
        catalog
            .resolve_system("urn:publicid:-:Example:TEXT+a%2Bb;c%3Ad%2Fe:EN")
            .as_deref(),
        Some("http://example.org/odd.dtd")
    );
    Ok(())
}

#[test]
fn a_urn_that_is_not_a_publicid_urn_stays_a_system_identifier() -> Result<(), CatalogError> {
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
      <system systemId="urn:example:foo" uri="foo.dtd"/>
    </catalog>
    "#;
    let catalog = Catalog::parse(xml, "http://example.org/catalog.xml")?;
    assert_eq!(
        catalog.resolve_system("urn:example:foo").as_deref(),
        Some("http://example.org/foo.dtd")
    );
    Ok(())
}

#[test]
fn publicid_urn_reaches_delegate_public_entries() -> Result<(), CatalogError> {
    // §7.1.2 step 1: once unwrapped, the URN follows the *public* path, `delegatePublic`
    // included — with no loader in hand the delegation simply yields no match, which is still
    // observably different from the URN having been treated as a system identifier.
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
      <delegateSystem systemIdStartString="urn:publicid:" catalog="never.xml"/>
      <public publicId="-//Example//DTD Foo//EN" uri="foo.dtd"/>
    </catalog>
    "#;
    let catalog = Catalog::parse(xml, "http://example.org/catalog.xml")?;
    // The `delegateSystem` entry must not swallow the URN: it is a public identifier.
    assert_eq!(
        catalog
            .resolve_system("urn:publicid:-:Example:DTD+Foo:EN")
            .as_deref(),
        Some("http://example.org/foo.dtd")
    );
    Ok(())
}

#[test]
fn prefer_system_does_not_suppress_an_unwrapped_publicid_urn() -> Result<(), CatalogError> {
    // Once unwrapped there is no system identifier left, so the `prefer="system"` suppression
    // of `public` entries (§7.1.2) does not apply.
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog" prefer="system">
      <public publicId="-//Example//DTD Foo//EN" uri="foo.dtd"/>
    </catalog>
    "#;
    let catalog = Catalog::parse(xml, "http://example.org/catalog.xml")?;
    assert_eq!(
        catalog
            .resolve_system("urn:publicid:-:Example:DTD+Foo:EN")
            .as_deref(),
        Some("http://example.org/foo.dtd")
    );
    Ok(())
}

#[test]
fn a_publicid_urn_prefix_with_a_multibyte_char_at_the_boundary_does_not_panic()
-> Result<(), CatalogError> {
    // `unwrap_publicid` used to slice at a fixed byte offset before checking whether that
    // offset actually landed on a UTF-8 character boundary. A 12-ASCII-byte prefix (missing its
    // trailing `:`) immediately followed by `€` (a 3-byte character) puts the 13th byte inside
    // that character, which must be handled as "not a match", not as a panic.
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
      <public publicId="-//Example//DTD Foo//EN" uri="foo.dtd"/>
    </catalog>
    "#;
    let catalog = Catalog::parse(xml, "http://example.org/catalog.xml")?;
    let hostile = "urn:publicid\u{20ac}rest-of-the-system-id";
    assert_eq!(catalog.resolve_system(hostile), None);
    assert_eq!(catalog.resolve_uri(hostile), None);
    assert_eq!(catalog.resolve_public(hostile, None), None);
    Ok(())
}

// -------------------------------------------------------- xml:base on rewrite entries

#[test]
fn rewrite_entries_honour_xml_base_on_the_entry_element() -> Result<(), CatalogError> {
    // OASIS XML Catalogs 1.1 §4.1 admits `xml:base` on every catalog entry element, and XML
    // Base makes it govern that element's own attributes — `rewritePrefix` included.
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
      <system xml:base="sub/" systemId="http://example.com/d/plain" uri="plain.dtd"/>
      <rewriteSystem xml:base="sub/" systemIdStartString="http://example.com/d/"
                     rewritePrefix="local/"/>
      <rewriteURI xml:base="sub/" uriStartString="http://example.com/u/"
                  rewritePrefix="local/"/>
    </catalog>
    "#;
    let catalog = Catalog::parse(xml, "file:///etc/xml/catalog.xml")?;
    assert_eq!(
        catalog
            .resolve_system("http://example.com/d/plain")
            .as_deref(),
        Some("file:///etc/xml/sub/plain.dtd")
    );
    assert_eq!(
        catalog
            .resolve_system("http://example.com/d/x.dtd")
            .as_deref(),
        Some("file:///etc/xml/sub/local/x.dtd")
    );
    assert_eq!(
        catalog.resolve_uri("http://example.com/u/x.xml").as_deref(),
        Some("file:///etc/xml/sub/local/x.xml")
    );
    Ok(())
}
