//! Tests for the namespace-aware [`NsReader`](crate::reader::NsReader).

use crate::XmlVersion;
use crate::errors::Error;
use crate::events::Event;
use crate::name::{Namespace, NamespaceError, PrefixDeclaration, QName, ResolveResult};
use crate::reader::{DEFAULT_MAX_ELEMENT_DEPTH, NsReader};

type TestResult = Result<(), Error>;

/// Reads the next `Start`/`Empty` event, panicking otherwise.
fn expect_start<'i>(reader: &mut NsReader<&'i [u8]>) -> Event<'i> {
    match reader.read_event().expect("event") {
        e @ (Event::Start(_) | Event::Empty(_)) => e,
        other => panic!("expected start/empty, got {:?}", other),
    }
}

#[test]
fn default_namespace_binds_unprefixed_element() -> TestResult {
    let mut reader = NsReader::from_str(r#"<tag xmlns="urn:d">x</tag>"#);
    match reader.read_resolved_event()? {
        (ns, Event::Start(e)) => {
            assert_eq!(ns, ResolveResult::Bound(Namespace(b"urn:d")));
            assert_eq!(e.local_name().as_ref(), b"tag");
        }
        other => panic!("unexpected {:?}", other),
    }
    Ok(())
}

#[test]
fn unprefixed_attribute_is_unbound_even_with_default_ns() -> TestResult {
    let mut reader = NsReader::from_str(r#"<tag xmlns="urn:d" a="1"/>"#);
    let event = expect_start(&mut reader);
    if let Event::Empty(e) = event {
        // Element resolves to the default namespace ...
        assert_eq!(
            reader.resolve_element(e.name()).0,
            ResolveResult::Bound(Namespace(b"urn:d")),
        );
        // ... but the unprefixed attribute never does.
        let attr = e
            .attributes()
            .with_checks(false)
            .find_map(|a| {
                let a = a.expect("attr");
                (a.key.as_ref() == b"a").then_some(a)
            })
            .expect("attr a");
        assert_eq!(reader.resolve_attribute(attr.key).0, ResolveResult::Unbound,);
    } else {
        panic!("expected empty");
    }
    Ok(())
}

#[test]
fn resolve_flag_distinguishes_element_and_attribute() -> TestResult {
    let mut reader = NsReader::from_str(r#"<tag xmlns="urn:d"/>"#);
    let _ = reader.read_event()?;
    // attribute = false -> element semantics -> default namespace applies.
    assert_eq!(
        reader.resolve(QName(b"x"), false).0,
        ResolveResult::Bound(Namespace(b"urn:d")),
    );
    // attribute = true -> attribute semantics -> no default namespace.
    assert_eq!(reader.resolve(QName(b"x"), true).0, ResolveResult::Unbound);
    Ok(())
}

#[test]
fn unknown_prefix_reports_unknown() -> TestResult {
    let mut reader = NsReader::from_str(r#"<p:tag>x</p:tag>"#);
    match reader.read_resolved_event()? {
        (ns, Event::Start(_)) => {
            assert_eq!(ns, ResolveResult::Unknown(b"p".to_vec()));
        }
        other => panic!("unexpected {:?}", other),
    }
    Ok(())
}

#[test]
fn nested_prefixes_resolve_to_nearest_binding() -> TestResult {
    let xml = r#"<x:a xmlns:x="ns1"><x:b xmlns:x="ns2"><x:c/></x:b><x:d/></x:a>"#;
    let mut reader = NsReader::from_str(xml);

    // <x:a> -> ns1
    match reader.read_resolved_event()? {
        (ns, Event::Start(_)) => assert_eq!(ns, ResolveResult::Bound(Namespace(b"ns1"))),
        other => panic!("a: {:?}", other),
    }
    // <x:b> -> ns2 (shadows x=ns1)
    match reader.read_resolved_event()? {
        (ns, Event::Start(_)) => assert_eq!(ns, ResolveResult::Bound(Namespace(b"ns2"))),
        other => panic!("b: {:?}", other),
    }
    // <x:c/> -> ns2
    match reader.read_resolved_event()? {
        (ns, Event::Empty(_)) => assert_eq!(ns, ResolveResult::Bound(Namespace(b"ns2"))),
        other => panic!("c: {:?}", other),
    }
    // </x:b> -> ns2 (still in scope for the end tag)
    match reader.read_resolved_event()? {
        (ns, Event::End(_)) => assert_eq!(ns, ResolveResult::Bound(Namespace(b"ns2"))),
        other => panic!("/b: {:?}", other),
    }
    // <x:d/> -> back to ns1 after </x:b> popped ns2
    match reader.read_resolved_event()? {
        (ns, Event::Empty(_)) => assert_eq!(ns, ResolveResult::Bound(Namespace(b"ns1"))),
        other => panic!("d: {:?}", other),
    }
    // </x:a> -> ns1
    match reader.read_resolved_event()? {
        (ns, Event::End(_)) => assert_eq!(ns, ResolveResult::Bound(Namespace(b"ns1"))),
        other => panic!("/a: {:?}", other),
    }
    assert_eq!(reader.read_resolved_event()?.1, Event::Eof);
    Ok(())
}

#[test]
fn empty_element_scope_is_popped_after_next_read() -> TestResult {
    // The <b/> declares x=ns2; it must not leak into the sibling <x:c/>.
    let xml = r#"<x:a xmlns:x="ns1"><b xmlns:x="ns2"/><x:c/></x:a>"#;
    let mut reader = NsReader::from_str(xml);

    let _ = reader.read_resolved_event()?; // <x:a>
    let _ = reader.read_resolved_event()?; // <b xmlns:x="ns2"/>
    match reader.read_resolved_event()? {
        (ns, Event::Empty(_)) => assert_eq!(ns, ResolveResult::Bound(Namespace(b"ns1"))),
        other => panic!("c: {:?}", other),
    }
    Ok(())
}

#[test]
fn default_namespace_reset_to_empty() -> TestResult {
    let xml = r#"<a xmlns="ns1"><b xmlns=""><c/></b></a>"#;
    let mut reader = NsReader::from_str(xml);

    match reader.read_resolved_event()? {
        (ns, Event::Start(_)) => assert_eq!(ns, ResolveResult::Bound(Namespace(b"ns1"))),
        other => panic!("a: {:?}", other),
    }
    // <b xmlns=""> resets the default namespace.
    match reader.read_resolved_event()? {
        (ns, Event::Start(_)) => assert_eq!(ns, ResolveResult::Unbound),
        other => panic!("b: {:?}", other),
    }
    match reader.read_resolved_event()? {
        (ns, Event::Empty(_)) => assert_eq!(ns, ResolveResult::Unbound),
        other => panic!("c: {:?}", other),
    }
    Ok(())
}

#[test]
fn prefixes_iterator_lists_in_scope_bindings() -> TestResult {
    let mut reader = NsReader::from_str(r#"<tag xmlns="d" xmlns:a="a1" xmlns:b="b1"/>"#);
    let _ = reader.read_event()?;
    let mut prefixes = reader.prefixes().collect::<Vec<_>>();
    prefixes.sort_by(|l, r| l.1.as_ref().cmp(r.1.as_ref()));
    assert_eq!(
        prefixes,
        vec![
            (PrefixDeclaration::Named(b"a"), Namespace(b"a1")),
            (PrefixDeclaration::Named(b"b"), Namespace(b"b1")),
            (PrefixDeclaration::Default, Namespace(b"d")),
        ],
    );
    Ok(())
}

#[test]
fn read_to_end_pops_scope() -> TestResult {
    let xml = r#"<a xmlns:x="ns1"><x:inner/></a><x:after/>"#;
    let mut reader = NsReader::from_str(xml);
    reader.config_mut().allow_unmatched_ends = true;
    reader.config_mut().check_end_names = false;

    let start = match reader.read_event()? {
        Event::Start(e) => e.into_owned(),
        other => panic!("a: {:?}", other),
    };
    reader.read_to_end(start.name())?;
    // The x binding declared on <a> must have been popped, so <x:after/> is unknown.
    match reader.read_resolved_event()? {
        (ns, Event::Empty(_)) => assert_eq!(ns, ResolveResult::Unknown(b"x".to_vec())),
        other => panic!("after: {:?}", other),
    }
    Ok(())
}

#[test]
fn resolver_and_resolver_mut_are_exposed() -> TestResult {
    let mut reader = NsReader::from_str(r#"<tag xmlns:a="a1"/>"#);
    reader.resolver_mut().set_max_declarations_per_element(4);
    assert_eq!(reader.resolver().max_declarations_per_element(), 4);
    let _ = reader.read_event()?;
    assert_eq!(reader.resolver().level(), 1);
    Ok(())
}

#[test]
fn buffered_reader_resolves_like_slice_reader() -> TestResult {
    let xml = r#"<x:tag xmlns:x="urn:x">t</x:tag>"#;
    let mut reader = NsReader::from_reader(xml.as_bytes());
    let mut buf = Vec::new();
    match reader.read_resolved_event_into(&mut buf)? {
        (ns, Event::Start(e)) => {
            assert_eq!(ns, ResolveResult::Bound(Namespace(b"urn:x")));
            assert_eq!(e.local_name().as_ref(), b"tag");
        }
        other => panic!("unexpected {:?}", other),
    }
    Ok(())
}

#[test]
fn too_many_declarations_is_an_error() {
    let mut reader = NsReader::from_str(r#"<tag xmlns:a="1" xmlns:b="2" xmlns:c="3"/>"#);
    reader.resolver_mut().set_max_declarations_per_element(2);
    match reader.read_event() {
        Err(Error::Namespace(_)) => {}
        other => panic!("expected namespace error, got {:?}", other),
    }
}

#[test]
fn deref_exposes_reader_inspection_methods() -> TestResult {
    let mut reader = NsReader::from_str("<tag/>");
    let _ = reader.read_event()?;
    // `buffer_position` comes from the underlying `Reader` through `Deref`.
    assert_eq!(reader.buffer_position(), 6);
    Ok(())
}

/// Builds a document nested `depth` levels deep, each element re-declaring a
/// default namespace so [`NamespaceResolver::push`](crate::name::NamespaceResolver::push)
/// has to push a binding at every level.
fn nested_document(depth: usize) -> String {
    let mut xml = String::new();
    for i in 0..depth {
        xml.push_str(&format!("<e{i} xmlns=\"urn:ns{i}\">"));
    }
    xml.push_str("text");
    for i in (0..depth).rev() {
        xml.push_str(&format!("</e{i}>"));
    }
    xml
}

/// Regression for the namespace-nesting-depth overflow: with the default
/// [`Config::max_element_depth`](crate::reader::Config::max_element_depth), a
/// document nested past the limit must return a clean
/// [`NamespaceError::TooDeeplyNested`] error rather than panicking (debug) or
/// silently wrapping and corrupting namespace scoping (release).
#[test]
fn document_deeper_than_default_limit_errors_cleanly() {
    let depth = DEFAULT_MAX_ELEMENT_DEPTH + 1;
    let xml = nested_document(depth);
    let mut reader = NsReader::from_str(&xml);
    loop {
        match reader.read_event() {
            Ok(Event::Eof) => panic!("expected a TooDeeplyNested error before EOF"),
            Ok(_) => {}
            // `Error` does not implement `PartialEq` (it wraps an `Arc<io::Error>`
            // variant), so match down to the comparable `NamespaceError` payload.
            Err(Error::Namespace(inner)) => {
                assert_eq!(
                    inner,
                    NamespaceError::TooDeeplyNested(DEFAULT_MAX_ELEMENT_DEPTH)
                );
                break;
            }
            Err(other) => panic!("expected a namespace error, got {other:?}"),
        }
    }
}

/// A namespace declaration is an attribute, and the namespace name it binds is the attribute's
/// *normalized* value: [Namespaces in XML §3] takes the namespace name from "the attribute
/// value", which [XML §3.3.3] normalizes before any application sees it. Reading the same
/// declaration two ways — through the resolver and through
/// [`Attribute::normalized_value`](crate::events::attributes::Attribute::normalized_value) —
/// must therefore give one answer, whatever escapes the author wrote.
///
/// The two rules of the algorithm pull in opposite directions and both have to be honoured: a
/// *white space character* becomes a space, while a *character reference* contributes the
/// character it names — so a literal line break in the declaration is a space, and `&#x0A;` is
/// still a line break.
///
/// [Namespaces in XML §3]: https://www.w3.org/TR/xml-names11/#ns-decl
/// [XML §3.3.3]: https://www.w3.org/TR/xml/#AVNormalize
#[test]
fn a_binding_uses_the_normalized_attribute_value() -> TestResult {
    for (document, expected) in [
        (r#"<r xmlns:p="urn:a&#x0A;b"><p:e/></r>"#, "urn:a\nb"),
        (r#"<r xmlns:p="urn:a&#9;b"><p:e/></r>"#, "urn:a\tb"),
        ("<r xmlns:p=\"urn:a\r\nb\"><p:e/></r>", "urn:a b"),
        ("<r xmlns:p=\"urn:a\tb\"><p:e/></r>", "urn:a b"),
        (r#"<r xmlns:p="urn:a&amp;b"><p:e/></r>"#, "urn:a&b"),
        (r#"<r xmlns:p="urn:plain"><p:e/></r>"#, "urn:plain"),
    ] {
        let mut reader = NsReader::from_str(document);
        let root = match expect_start(&mut reader) {
            Event::Start(e) => e,
            other => panic!("unexpected {:?}", other),
        };
        // The declaration read as an ordinary attribute.
        let declared = root
            .attributes()
            .next()
            .expect("the declaration")
            .expect("a well-formed attribute")
            .normalized_value(XmlVersion::Implicit1_0)
            .expect("a normalizable value")
            .into_owned();
        assert_eq!(declared, expected, "in `{document}`");

        // The same declaration read through the prefix it binds.
        match reader.read_resolved_event()? {
            (ResolveResult::Bound(namespace), Event::Empty(_)) => {
                assert_eq!(
                    std::str::from_utf8(namespace.into_inner()).expect("utf-8"),
                    declared,
                    "in `{document}`",
                );
            }
            other => panic!("unexpected {:?}", other),
        }
    }
    Ok(())
}

/// A declaration naming an entity that only the document declares is bound as it stands.
///
/// The resolver knows the five entities the specification predefines and nothing else, so it
/// cannot finish normalizing such a value — and it does not guess: the raw text is bound, and
/// the well-formedness check, which does hold the internal subset, resolves the reference when
/// it compares expanded attribute names. That is what keeps two spellings of one namespace name
/// from admitting two attributes of the same expanded name, which
/// [Namespaces in XML §5.3](https://www.w3.org/TR/xml-names11/#uniqAttrs) forbids.
#[test]
fn a_declaration_using_a_document_entity_is_left_to_the_reader() -> TestResult {
    let mut reader = NsReader::from_str(
        r#"<!DOCTYPE r [<!ENTITY tilde "~">]><r xmlns:p="urn:a&tilde;b"><p:e/></r>"#,
    );
    loop {
        match reader.read_resolved_event()? {
            (ResolveResult::Bound(namespace), Event::Empty(_)) => {
                assert_eq!(namespace, Namespace(b"urn:a&tilde;b"));
                break;
            }
            (_, Event::Eof) => panic!("the prefixed element was never reached"),
            _ => {}
        }
    }

    // Two spellings of one namespace name still collide where it counts.
    let mut checking = NsReader::from_str(
        r#"<!DOCTYPE foo [<!ENTITY tilde "~">]><foo xmlns:a="urn:~w" xmlns:b="urn:&tilde;w"><bar a:attr="1" b:attr="2"/></foo>"#,
    );
    checking.config_mut().check_well_formedness = true;
    let failure = loop {
        match checking.read_resolved_event() {
            Ok((_, Event::Eof)) => panic!("the duplicate expanded name went undetected"),
            Ok(_) => {}
            Err(error) => break error,
        }
    };
    assert!(
        failure.to_string().contains("more than once"),
        "unexpected error: {failure}"
    );
    Ok(())
}

/// Raising [`Config::max_element_depth`](crate::reader::Config::max_element_depth)
/// allows reading documents nested far past the historical `u16::MAX`
/// (65_535) bound without the namespace resolver corrupting prefix
/// resolution: the binding declared by the *deepest* (> 65_536) element must
/// still be the one a same-depth name resolves to.
#[test]
fn raising_the_limit_allows_deep_nesting_without_corruption() -> TestResult {
    let depth: usize = 70_000;
    let xml = nested_document(depth);
    let mut reader = NsReader::from_str(&xml);
    reader.config_mut().max_element_depth = depth + 1;

    let mut opens = 0usize;
    // Owned, not `ResolveResult<'_>`: that borrows the resolver for the
    // duration of one loop iteration, which cannot outlive the loop.
    let mut deepest_namespace: Option<Vec<u8>> = None;
    loop {
        match reader.read_resolved_event()? {
            (ResolveResult::Bound(ns), Event::Start(_)) => {
                opens += 1;
                if opens == depth {
                    deepest_namespace = Some(ns.into_inner().to_vec());
                }
            }
            (_, Event::Start(_)) => opens += 1,
            (_, Event::Eof) => break,
            _ => {}
        }
    }
    assert_eq!(opens, depth);
    assert_eq!(
        deepest_namespace.as_deref(),
        Some(format!("urn:ns{}", depth - 1).as_bytes()),
    );
    // Every scope must have popped back out by EOF.
    assert_eq!(reader.resolver().level(), 0);
    Ok(())
}
