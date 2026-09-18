//! `nextCatalog` traversal, `delegatePublic`/`delegateSystem`/`delegateURI`
//! chaining and ordering, and loop detection — all of which require a
//! [`CatalogLoader`] to fetch the referenced documents.

use std::cell::RefCell;
use std::collections::HashMap;

use oxixml_catalog::{Catalog, CatalogError, CatalogLoader};

/// An in-memory [`CatalogLoader`] backed by a fixed URI → document map, which
/// also records every URI it was asked to load (so tests can assert on which
/// catalogs were actually visited).
#[derive(Default)]
struct MapLoader {
    documents: HashMap<String, String>,
    requested: RefCell<Vec<String>>,
}

impl MapLoader {
    fn new<K: Into<String>, V: Into<String>>(documents: impl IntoIterator<Item = (K, V)>) -> Self {
        Self {
            documents: documents
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
            requested: RefCell::new(Vec::new()),
        }
    }

    fn requested(&self) -> Vec<String> {
        self.requested.borrow().clone()
    }
}

impl CatalogLoader for MapLoader {
    fn load(&self, uri: &str) -> Option<String> {
        self.requested.borrow_mut().push(uri.to_owned());
        self.documents.get(uri).cloned()
    }
}

const NS: &str = r#"xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog""#;

#[test]
fn next_catalog_is_consulted_only_after_the_current_file_has_no_match() -> Result<(), CatalogError>
{
    let next_xml = format!(
        r#"<catalog {NS}><system systemId="urn:example:foo" uri="from-next.dtd"/></catalog>"#
    );
    let loader = MapLoader::new([("http://example.org/next.xml", next_xml.as_str())]);

    let root_xml = format!(
        r#"<catalog {NS}>
             <nextCatalog catalog="next.xml"/>
             <system systemId="urn:example:foo" uri="from-root.dtd"/>
           </catalog>"#
    );
    let catalog = Catalog::parse(&root_xml, "http://example.org/catalog.xml")?;

    // The root file has a direct match, so `next.xml` must never even be requested.
    assert_eq!(
        catalog
            .resolve_system_with("urn:example:foo", Some(&loader))
            .as_deref(),
        Some("http://example.org/from-root.dtd")
    );
    assert!(
        loader.requested().is_empty(),
        "root match must short-circuit nextCatalog"
    );
    Ok(())
}

#[test]
fn next_catalog_is_followed_when_no_match_in_current_file() -> Result<(), CatalogError> {
    let next_xml =
        format!(r#"<catalog {NS}><system systemId="urn:example:bar" uri="bar.dtd"/></catalog>"#);
    let loader = MapLoader::new([("http://example.org/next.xml", next_xml.as_str())]);

    let root_xml = format!(r#"<catalog {NS}><nextCatalog catalog="next.xml"/></catalog>"#);
    let catalog = Catalog::parse(&root_xml, "http://example.org/catalog.xml")?;

    assert_eq!(
        catalog
            .resolve_system_with("urn:example:bar", Some(&loader))
            .as_deref(),
        Some("http://example.org/bar.dtd")
    );
    assert_eq!(
        loader.requested(),
        vec!["http://example.org/next.xml".to_owned()]
    );
    Ok(())
}

#[test]
fn without_a_loader_next_catalog_contributes_no_match() -> Result<(), CatalogError> {
    let root_xml = format!(r#"<catalog {NS}><nextCatalog catalog="next.xml"/></catalog>"#);
    let catalog = Catalog::parse(&root_xml, "http://example.org/catalog.xml")?;
    assert_eq!(catalog.resolve_system("urn:example:bar"), None);
    Ok(())
}

#[test]
fn delegate_system_tries_longest_prefix_catalog_first() -> Result<(), CatalogError> {
    let short_xml = format!(
        r#"<catalog {NS}><system systemId="urn:example:a:b" uri="from-short.dtd"/></catalog>"#
    );
    let long_xml = format!(
        r#"<catalog {NS}><system systemId="urn:example:a:b" uri="from-long.dtd"/></catalog>"#
    );
    let loader = MapLoader::new([
        ("http://example.org/short.xml", short_xml.as_str()),
        ("http://example.org/long.xml", long_xml.as_str()),
    ]);

    let root_xml = format!(
        r#"<catalog {NS}>
             <delegateSystem systemIdStartString="urn:example:" catalog="short.xml"/>
             <delegateSystem systemIdStartString="urn:example:a:" catalog="long.xml"/>
           </catalog>"#
    );
    let catalog = Catalog::parse(&root_xml, "http://example.org/catalog.xml")?;

    // Both delegates match the prefix; the longer `systemIdStartString`
    // ("urn:example:a:") must be tried first, and since it *does* find a
    // match, the shorter one's catalog is never even requested.
    assert_eq!(
        catalog
            .resolve_system_with("urn:example:a:b", Some(&loader))
            .as_deref(),
        Some("http://example.org/from-long.dtd")
    );
    assert_eq!(
        loader.requested(),
        vec!["http://example.org/long.xml".to_owned()]
    );
    Ok(())
}

#[test]
fn delegate_system_falls_through_the_ordered_list() -> Result<(), CatalogError> {
    // The longest-prefix delegate's catalog exists but has no matching entry;
    // resolution must continue to the next delegate in the (longest-first)
    // list, not abandon the search after the first delegate catalog fails.
    let long_xml = format!(r#"<catalog {NS}/>"#);
    let short_xml = format!(
        r#"<catalog {NS}><system systemId="urn:example:a:b" uri="from-short.dtd"/></catalog>"#
    );
    let loader = MapLoader::new([
        ("http://example.org/long.xml", long_xml.as_str()),
        ("http://example.org/short.xml", short_xml.as_str()),
    ]);

    let root_xml = format!(
        r#"<catalog {NS}>
             <delegateSystem systemIdStartString="urn:example:" catalog="short.xml"/>
             <delegateSystem systemIdStartString="urn:example:a:" catalog="long.xml"/>
           </catalog>"#
    );
    let catalog = Catalog::parse(&root_xml, "http://example.org/catalog.xml")?;

    assert_eq!(
        catalog
            .resolve_system_with("urn:example:a:b", Some(&loader))
            .as_deref(),
        Some("http://example.org/from-short.dtd")
    );
    Ok(())
}

#[test]
fn delegation_does_not_fall_back_to_the_delegating_catalogs_own_next_catalog()
-> Result<(), CatalogError> {
    // Once a `delegateSystem` entry matches for this identifier, resolution
    // restarts *exclusively* within the delegate list; if that fails, it
    // must not resume with the rest of the original catalog list — including
    // that same file's own `nextCatalog` entries.
    let delegate_xml = format!(r#"<catalog {NS}/>"#);
    let next_xml = format!(
        r#"<catalog {NS}><system systemId="urn:example:foo" uri="should-not-be-used.dtd"/></catalog>"#
    );
    let loader = MapLoader::new([
        ("http://example.org/delegate.xml", delegate_xml.as_str()),
        ("http://example.org/next.xml", next_xml.as_str()),
    ]);

    let root_xml = format!(
        r#"<catalog {NS}>
             <delegateSystem systemIdStartString="urn:example:" catalog="delegate.xml"/>
             <nextCatalog catalog="next.xml"/>
           </catalog>"#
    );
    let catalog = Catalog::parse(&root_xml, "http://example.org/catalog.xml")?;

    assert_eq!(
        catalog.resolve_system_with("urn:example:foo", Some(&loader)),
        None
    );
    assert_eq!(
        loader.requested(),
        vec!["http://example.org/delegate.xml".to_owned()]
    );
    Ok(())
}

#[test]
fn delegate_public_respects_prefer_scoping() -> Result<(), CatalogError> {
    let delegate_xml = format!(
        r#"<catalog {NS}><public publicId="-//Example//DTD Foo//EN" uri="foo.dtd"/></catalog>"#
    );
    let loader = MapLoader::new([("http://example.org/delegate.xml", delegate_xml.as_str())]);

    let root_xml = format!(
        r#"<catalog {NS} prefer="system">
             <delegatePublic publicIdStartString="-//Example//" catalog="delegate.xml"/>
           </catalog>"#
    );
    let catalog = Catalog::parse(&root_xml, "http://example.org/catalog.xml")?;

    // prefer="system" + an explicit (unmatched) system id must suppress the
    // delegatePublic entry entirely.
    assert_eq!(
        catalog.resolve_public_with(
            "-//Example//DTD Foo//EN",
            Some("urn:example:x"),
            Some(&loader)
        ),
        None
    );
    // Without a system id, the delegatePublic entry is used.
    assert_eq!(
        catalog
            .resolve_public_with("-//Example//DTD Foo//EN", None, Some(&loader))
            .as_deref(),
        Some("http://example.org/foo.dtd")
    );
    Ok(())
}

#[test]
fn delegate_uri_chains_through_a_loader() -> Result<(), CatalogError> {
    let delegate_xml =
        format!(r#"<catalog {NS}><uri name="urn:example:res" uri="res.xml"/></catalog>"#);
    let loader = MapLoader::new([("http://example.org/delegate.xml", delegate_xml.as_str())]);

    let root_xml = format!(
        r#"<catalog {NS}><delegateURI uriStartString="urn:example:" catalog="delegate.xml"/></catalog>"#
    );
    let catalog = Catalog::parse(&root_xml, "http://example.org/catalog.xml")?;

    assert_eq!(
        catalog
            .resolve_uri_with("urn:example:res", Some(&loader))
            .as_deref(),
        Some("http://example.org/res.xml")
    );
    Ok(())
}

#[test]
fn next_catalog_cycle_terminates_and_finds_no_match() -> Result<(), CatalogError> {
    // a.xml -> nextCatalog b.xml -> nextCatalog a.xml (cycle back to the root).
    let a_xml = format!(r#"<catalog {NS}><nextCatalog catalog="b.xml"/></catalog>"#);
    let b_xml = format!(r#"<catalog {NS}><nextCatalog catalog="a.xml"/></catalog>"#);
    let loader = MapLoader::new([
        ("http://example.org/a.xml", a_xml.as_str()),
        ("http://example.org/b.xml", b_xml.as_str()),
    ]);

    let root_xml = format!(r#"<catalog {NS}><nextCatalog catalog="a.xml"/></catalog>"#);
    let catalog = Catalog::parse(&root_xml, "http://example.org/catalog.xml")?;

    // Must terminate (this test would hang forever otherwise) and report no match.
    assert_eq!(
        catalog.resolve_system_with("urn:example:anything", Some(&loader)),
        None
    );
    // Each distinct catalog URI is loaded at most once.
    let mut requested = loader.requested();
    requested.sort();
    requested.dedup();
    assert_eq!(
        requested.len(),
        loader.requested().len(),
        "no catalog URI should be loaded twice"
    );
    Ok(())
}

#[test]
fn delegate_cycle_terminates_and_finds_no_match() -> Result<(), CatalogError> {
    let a_xml = format!(
        r#"<catalog {NS}><delegateSystem systemIdStartString="urn:" catalog="b.xml"/></catalog>"#
    );
    let b_xml = format!(
        r#"<catalog {NS}><delegateSystem systemIdStartString="urn:" catalog="a.xml"/></catalog>"#
    );
    let loader = MapLoader::new([
        ("http://example.org/a.xml", a_xml.as_str()),
        ("http://example.org/b.xml", b_xml.as_str()),
    ]);

    let root_xml = format!(
        r#"<catalog {NS}><delegateSystem systemIdStartString="urn:" catalog="a.xml"/></catalog>"#
    );
    let catalog = Catalog::parse(&root_xml, "http://example.org/catalog.xml")?;

    assert_eq!(
        catalog.resolve_system_with("urn:example:anything", Some(&loader)),
        None
    );
    Ok(())
}

/// Answers every `chain:N` request with a catalog that delegates to `chain:N+1` — a URI it has
/// never been asked for before — forever.
struct ChainLoader;

impl CatalogLoader for ChainLoader {
    fn load(&self, uri: &str) -> Option<String> {
        let index: usize = uri.strip_prefix("chain:")?.parse().ok()?;
        Some(format!(
            r#"<catalog {NS}><delegateSystem systemIdStartString="urn:" catalog="chain:{}"/></catalog>"#,
            index + 1
        ))
    }
}

#[test]
fn a_delegate_chain_of_ever_new_uris_terminates_instead_of_recursing_without_bound()
-> Result<(), CatalogError> {
    // The `visited` loop detector only stops a chain from re-entering a URI it has already
    // seen; it does nothing to bound a chain that only ever delegates to *new* URIs, which is
    // exactly what `ChainLoader` produces. Without the depth cap this recurses the native call
    // stack without bound.
    let root_xml = format!(
        r#"<catalog {NS}><delegateSystem systemIdStartString="urn:" catalog="chain:0"/></catalog>"#
    );
    let catalog = Catalog::parse(&root_xml, "http://example.org/catalog.xml")?;
    assert_eq!(
        catalog.resolve_system_with("urn:example:anything", Some(&ChainLoader)),
        None
    );
    Ok(())
}

#[test]
fn a_delegate_chain_within_the_depth_cap_still_finds_its_match() -> Result<(), CatalogError> {
    /// Like [`ChainLoader`], but the chain ends in a direct match after a fixed number of hops
    /// instead of continuing forever.
    struct BoundedChainLoader {
        last: usize,
    }
    impl CatalogLoader for BoundedChainLoader {
        fn load(&self, uri: &str) -> Option<String> {
            let index: usize = uri.strip_prefix("chain:")?.parse().ok()?;
            if index == self.last {
                Some(format!(
                    r#"<catalog {NS}><system systemId="urn:example:target" uri="found.dtd"/></catalog>"#
                ))
            } else {
                Some(format!(
                    r#"<catalog {NS}><delegateSystem systemIdStartString="urn:" catalog="chain:{}"/></catalog>"#,
                    index + 1
                ))
            }
        }
    }

    // One hop short of the cap, so the match is found on the very last catalog the cap allows.
    let loader = BoundedChainLoader {
        last: oxixml_catalog::MAX_DELEGATE_DEPTH - 1,
    };
    let root_xml = format!(
        r#"<catalog {NS}><delegateSystem systemIdStartString="urn:" catalog="chain:0"/></catalog>"#
    );
    let catalog = Catalog::parse(&root_xml, "http://example.org/catalog.xml")?;
    // The exact resolved URI depends only on ordinary relative-reference resolution against
    // the deepest catalog's own URI (already covered elsewhere); what matters here is that a
    // legitimate chain landing exactly on the cap still finds its match rather than being
    // mistaken for the unbounded case.
    assert!(
        catalog
            .resolve_system_with("urn:example:target", Some(&loader))
            .is_some()
    );
    Ok(())
}

#[test]
fn unreachable_next_catalog_is_treated_as_no_match_not_an_error() -> Result<(), CatalogError> {
    struct NeverLoader;
    impl CatalogLoader for NeverLoader {
        fn load(&self, _uri: &str) -> Option<String> {
            None
        }
    }

    let root_xml = format!(r#"<catalog {NS}><nextCatalog catalog="missing.xml"/></catalog>"#);
    let catalog = Catalog::parse(&root_xml, "http://example.org/catalog.xml")?;
    assert_eq!(
        catalog.resolve_system_with("urn:example:anything", Some(&NeverLoader)),
        None
    );
    Ok(())
}
