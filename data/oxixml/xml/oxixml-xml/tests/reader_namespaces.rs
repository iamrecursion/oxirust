//! Integration tests for the namespace-aware [`NsReader`].
//!
//! The core resolution cases are ported from quick-xml 0.41
//! `tests/reader-namespaces.rs` (MIT/Apache-2.0, © the quick-xml authors),
//! adapted to the `oxixml_xml` public API and to the COOLJAPAN no-`unwrap()`
//! policy (tests use `?`/`expect`). The remainder are `oxixml`-specific: the
//! full nested / default / undeclared resolution matrices, the element-versus-
//! attribute resolution difference, reserved-prefix handling and the
//! `prefixes()` iterator.
//!
//! [`NsReader`]: oxixml_xml::reader::NsReader

use std::borrow::Cow;

use oxixml_xml::errors::Error;
use oxixml_xml::events::Event::*;
use oxixml_xml::events::attributes::Attribute;
use oxixml_xml::events::{BytesStart, Event};
use oxixml_xml::name::ResolveResult::*;
use oxixml_xml::name::{Namespace, PrefixDeclaration, QName, ResolveResult};
use oxixml_xml::reader::NsReader;

type TestResult = Result<(), Error>;

#[test]
fn namespace() -> TestResult {
    let mut r = NsReader::from_str("<a xmlns:myns='www1'><myns:b>in namespace!</myns:b></a>");
    let expected = vec![(PrefixDeclaration::Named(b"myns"), Namespace(b"www1"))];

    // <a>
    match r.read_resolved_event()? {
        (ns, Start(_)) => assert_eq!(ns, Unbound),
        e => panic!("expecting outer start element with no namespace, got {e:?}"),
    }
    // The iterator is `Clone`, and cloning does not consume the original.
    let it1 = r.resolver().bindings();
    let it2 = it1.clone();
    assert_eq!(it1.size_hint(), (0, Some(1)));
    assert_eq!(it1.collect::<Vec<_>>(), expected);
    assert_eq!(it2.size_hint(), (0, Some(1)));
    assert_eq!(it2.collect::<Vec<_>>(), expected);

    // <myns:b>
    match r.read_resolved_event()? {
        (ns, Start(_)) => assert_eq!(ns, Bound(Namespace(b"www1"))),
        e => panic!("expecting inner start element to resolve to 'www1', got {e:?}"),
    }
    assert_eq!(r.resolver().bindings().collect::<Vec<_>>(), expected);

    // "in namespace!" — text has no namespace.
    match r.read_resolved_event()? {
        (ns, Text(_)) => assert_eq!(ns, Unbound),
        e => panic!("expecting text content with no namespace, got {e:?}"),
    }
    assert_eq!(r.resolver().bindings().collect::<Vec<_>>(), expected);

    // </myns:b>
    match r.read_resolved_event()? {
        (ns, End(_)) => assert_eq!(ns, Bound(Namespace(b"www1"))),
        e => panic!("expecting inner end element to resolve to 'www1', got {e:?}"),
    }
    assert_eq!(r.resolver().bindings().collect::<Vec<_>>(), expected);

    // </a>
    match r.read_resolved_event()? {
        (ns, End(_)) => assert_eq!(ns, Unbound),
        e => panic!("expecting outer end element with no namespace, got {e:?}"),
    }
    // The binding is still visible while the `End` event is the current one.
    assert_eq!(r.resolver().bindings().collect::<Vec<_>>(), expected);
    Ok(())
}

mod default_namespace {
    use super::*;

    #[test]
    fn event_empty() -> TestResult {
        let mut r = NsReader::from_str("<a attr='val' xmlns='ns' />");

        let e = match r.read_resolved_event()? {
            (ns, Empty(e)) => {
                assert_eq!(ns, Bound(Namespace(b"ns")));
                e
            }
            e => panic!("expecting Empty event, got {e:?}"),
        };

        let mut attrs = e
            .attributes()
            .map(|ar| ar.expect("attribute parsing succeeds"))
            // `xmlns` attributes are not interesting for this test.
            .filter(|kv| kv.key.as_namespace_binding().is_none())
            .map(|Attribute { key: name, value }| {
                let (opt_ns, local_name) = r.resolver().resolve_attribute(name);
                (opt_ns, local_name.into_inner(), value)
            });
        assert_eq!(
            attrs.next(),
            Some((Unbound, &b"attr"[..], Cow::Borrowed(&b"val"[..]))),
        );
        assert_eq!(attrs.next(), None);

        let it = r.resolver().bindings();
        assert_eq!(it.size_hint(), (0, Some(1)));
        assert_eq!(
            it.collect::<Vec<_>>(),
            vec![(PrefixDeclaration::Default, Namespace(b"ns"))],
        );
        Ok(())
    }

    #[test]
    fn event_start_end() -> TestResult {
        let mut r = NsReader::from_str(r#"<a ><b xmlns="www1"></b></a>"#);

        // <a>
        match r.read_resolved_event()? {
            (ns, Start(_)) => assert_eq!(ns, Unbound),
            e => panic!("expecting outer start element with no namespace, got {e:?}"),
        }
        let it = r.resolver().bindings();
        assert_eq!(it.size_hint(), (0, Some(0)));
        assert_eq!(it.collect::<Vec<_>>(), vec![]);

        // <b>
        match r.read_resolved_event()? {
            (ns, Start(_)) => assert_eq!(ns, Bound(Namespace(b"www1"))),
            e => panic!("expecting inner start element to resolve to 'www1', got {e:?}"),
        }
        let it = r.resolver().bindings();
        assert_eq!(it.size_hint(), (0, Some(1)));
        assert_eq!(
            it.collect::<Vec<_>>(),
            vec![(PrefixDeclaration::Default, Namespace(b"www1"))],
        );

        // </b>
        match r.read_resolved_event()? {
            (ns, End(_)) => assert_eq!(ns, Bound(Namespace(b"www1"))),
            e => panic!("expecting inner end element to resolve to 'www1', got {e:?}"),
        }
        assert_eq!(
            r.resolver().bindings().collect::<Vec<_>>(),
            vec![(PrefixDeclaration::Default, Namespace(b"www1"))],
        );

        // </a>. Very important: `a` must not be in any namespace — a default
        // namespace only applies to the sub-document it is declared on.
        match r.read_resolved_event()? {
            (ns, End(_)) => assert_eq!(ns, Unbound),
            e => panic!("expecting outer end element with no namespace, got {e:?}"),
        }
        let it = r.resolver().bindings();
        assert_eq!(it.size_hint(), (0, Some(0)));
        assert_eq!(it.collect::<Vec<_>>(), vec![]);
        Ok(())
    }

    #[test]
    fn reset() -> TestResult {
        let mut r = NsReader::from_str(r#"<a xmlns="www1"><b xmlns=""></b></a>"#);

        // <a>
        match r.read_resolved_event()? {
            (ns, Start(_)) => assert_eq!(ns, Bound(Namespace(b"www1"))),
            e => panic!("expecting outer start element to resolve to 'www1', got {e:?}"),
        }
        let it = r.resolver().bindings();
        assert_eq!(it.size_hint(), (0, Some(1)));
        assert_eq!(
            it.collect::<Vec<_>>(),
            vec![(PrefixDeclaration::Default, Namespace(b"www1"))],
        );

        // <b xmlns=""> undeclares the default namespace.
        match r.read_resolved_event()? {
            (ns, Start(_)) => assert_eq!(ns, Unbound),
            e => panic!("expecting inner start element with no namespace, got {e:?}"),
        }
        let it = r.resolver().bindings();
        assert_eq!(it.size_hint(), (0, Some(2)));
        assert_eq!(it.collect::<Vec<_>>(), vec![]);

        // </b>
        match r.read_resolved_event()? {
            (ns, End(_)) => assert_eq!(ns, Unbound),
            e => panic!("expecting inner end element with no namespace, got {e:?}"),
        }
        let it = r.resolver().bindings();
        assert_eq!(it.size_hint(), (0, Some(2)));
        assert_eq!(it.collect::<Vec<_>>(), vec![]);

        // </a>: the outer default namespace is back in scope.
        match r.read_resolved_event()? {
            (ns, End(_)) => assert_eq!(ns, Bound(Namespace(b"www1"))),
            e => panic!("expecting outer end element to resolve to 'www1', got {e:?}"),
        }
        let it = r.resolver().bindings();
        assert_eq!(it.size_hint(), (0, Some(1)));
        assert_eq!(
            it.collect::<Vec<_>>(),
            vec![(PrefixDeclaration::Default, Namespace(b"www1"))],
        );
        Ok(())
    }

    #[test]
    fn shadowing_empty() -> TestResult {
        let src = "<e xmlns='urn:example:o'><e att1='a' xmlns='urn:example:i' /></e>";
        let mut r = NsReader::from_str(src);

        // <e xmlns='urn:example:o'>
        match r.read_resolved_event()? {
            (ns, Start(e)) => {
                assert_eq!(ns, Bound(Namespace(b"urn:example:o")));
                assert_eq!(e.name(), QName(b"e"));
            }
            e => panic!("expected Start event (outer), got {e:?}"),
        }
        let it = r.resolver().bindings();
        assert_eq!(it.size_hint(), (0, Some(1)));
        assert_eq!(
            it.collect::<Vec<_>>(),
            vec![(PrefixDeclaration::Default, Namespace(b"urn:example:o"))],
        );

        // <e att1='a' xmlns='urn:example:i' />
        let e = match r.read_resolved_event()? {
            (ns, Empty(e)) => {
                assert_eq!(ns, Bound(Namespace(b"urn:example:i")));
                assert_eq!(e.name(), QName(b"e"));
                e
            }
            e => panic!("expecting Empty event, got {e:?}"),
        };
        let mut attrs = e
            .attributes()
            .map(|ar| ar.expect("attribute parsing succeeds"))
            .filter(|kv| kv.key.as_namespace_binding().is_none())
            .map(|Attribute { key: name, value }| {
                let (opt_ns, local_name) = r.resolver().resolve_attribute(name);
                (opt_ns, local_name.into_inner(), value)
            });
        // The attribute must *not* have a namespace name: a default namespace
        // never applies to attributes.
        assert_eq!(
            attrs.next(),
            Some((Unbound, &b"att1"[..], Cow::Borrowed(&b"a"[..]))),
        );
        assert_eq!(attrs.next(), None);

        let it = r.resolver().bindings();
        assert_eq!(it.size_hint(), (0, Some(2)));
        assert_eq!(
            it.collect::<Vec<_>>(),
            vec![(PrefixDeclaration::Default, Namespace(b"urn:example:i"))],
        );

        // </e> (outer)
        match r.read_resolved_event()? {
            (ns, End(e)) => {
                assert_eq!(ns, Bound(Namespace(b"urn:example:o")));
                assert_eq!(e.name(), QName(b"e"));
            }
            e => panic!("expected End event (outer), got {e:?}"),
        }
        let it = r.resolver().bindings();
        assert_eq!(it.size_hint(), (0, Some(1)));
        assert_eq!(
            it.collect::<Vec<_>>(),
            vec![(PrefixDeclaration::Default, Namespace(b"urn:example:o"))],
        );
        Ok(())
    }

    #[test]
    fn shadowing_expanded() -> TestResult {
        let src = "<e xmlns='urn:example:o'><e att1='a' xmlns='urn:example:i' /></e>";
        let mut r = NsReader::from_str(src);
        r.config_mut().expand_empty_elements = true;

        // <e xmlns='urn:example:o'>
        match r.read_resolved_event()? {
            (ns, Start(e)) => {
                assert_eq!(ns, Bound(Namespace(b"urn:example:o")));
                assert_eq!(e.name(), QName(b"e"));
            }
            e => panic!("expected Start event (outer), got {e:?}"),
        }

        // <e att1='a' xmlns='urn:example:i' />, expanded into Start + End.
        let e = match r.read_resolved_event()? {
            (ns, Start(e)) => {
                assert_eq!(ns, Bound(Namespace(b"urn:example:i")));
                assert_eq!(e.name(), QName(b"e"));
                e
            }
            e => panic!("expecting Start event (inner), got {e:?}"),
        };
        let mut attrs = e
            .attributes()
            .map(|ar| ar.expect("attribute parsing succeeds"))
            .filter(|kv| kv.key.as_namespace_binding().is_none())
            .map(|Attribute { key: name, value }| {
                let (opt_ns, local_name) = r.resolver().resolve_attribute(name);
                (opt_ns, local_name.into_inner(), value)
            });
        assert_eq!(
            attrs.next(),
            Some((Unbound, &b"att1"[..], Cow::Borrowed(&b"a"[..]))),
        );
        assert_eq!(attrs.next(), None);

        // virtual </e> (inner)
        match r.read_resolved_event()? {
            (ns, End(e)) => {
                assert_eq!(ns, Bound(Namespace(b"urn:example:i")));
                assert_eq!(e.name(), QName(b"e"));
            }
            e => panic!("expected End event (inner), got {e:?}"),
        }
        // </e> (outer)
        match r.read_resolved_event()? {
            (ns, End(e)) => {
                assert_eq!(ns, Bound(Namespace(b"urn:example:o")));
                assert_eq!(e.name(), QName(b"e"));
            }
            e => panic!("expected End event (outer), got {e:?}"),
        }
        Ok(())
    }
}

/// A single empty element with qualified attributes; empty-element expansion is
/// disabled. The namespace-handling code path differs slightly between `Empty`
/// and `Start` + `End`.
#[test]
fn attributes_empty_ns() -> TestResult {
    let mut r = NsReader::from_str("<a att1='a' r:att2='b' xmlns:r='urn:example:r' />");

    let e = match r.read_resolved_event()? {
        (Unbound, Empty(e)) => e,
        e => panic!("expecting Empty event, got {e:?}"),
    };

    let mut attrs = e
        .attributes()
        .map(|ar| ar.expect("attribute parsing succeeds"))
        .filter(|kv| kv.key.as_namespace_binding().is_none())
        .map(|Attribute { key: name, value }| {
            let (opt_ns, local_name) = r.resolver().resolve_attribute(name);
            (opt_ns, local_name.into_inner(), value)
        });
    assert_eq!(
        attrs.next(),
        Some((Unbound, &b"att1"[..], Cow::Borrowed(&b"a"[..]))),
    );
    assert_eq!(
        attrs.next(),
        Some((
            Bound(Namespace(b"urn:example:r")),
            &b"att2"[..],
            Cow::Borrowed(&b"b"[..]),
        )),
    );
    assert_eq!(attrs.next(), None);

    let it = r.resolver().bindings();
    assert_eq!(it.size_hint(), (0, Some(1)));
    assert_eq!(
        it.collect::<Vec<_>>(),
        vec![(PrefixDeclaration::Named(b"r"), Namespace(b"urn:example:r"))],
    );
    Ok(())
}

/// The same document as [`attributes_empty_ns`] with empty-element expansion on.
#[test]
fn attributes_empty_ns_expanded() -> TestResult {
    let mut r = NsReader::from_str("<a att1='a' r:att2='b' xmlns:r='urn:example:r' />");
    r.config_mut().expand_empty_elements = true;

    {
        let e = match r.read_resolved_event()? {
            (Unbound, Start(e)) => e,
            e => panic!("expecting Start event, got {e:?}"),
        };

        let mut attrs = e
            .attributes()
            .map(|ar| ar.expect("attribute parsing succeeds"))
            .filter(|kv| kv.key.as_namespace_binding().is_none())
            .map(|Attribute { key: name, value }| {
                let (opt_ns, local_name) = r.resolver().resolve_attribute(name);
                (opt_ns, local_name.into_inner(), value)
            });
        assert_eq!(
            attrs.next(),
            Some((Unbound, &b"att1"[..], Cow::Borrowed(&b"a"[..]))),
        );
        assert_eq!(
            attrs.next(),
            Some((
                Bound(Namespace(b"urn:example:r")),
                &b"att2"[..],
                Cow::Borrowed(&b"b"[..]),
            )),
        );
        assert_eq!(attrs.next(), None);

        let it = r.resolver().bindings();
        assert_eq!(it.size_hint(), (0, Some(1)));
        assert_eq!(
            it.collect::<Vec<_>>(),
            vec![(PrefixDeclaration::Named(b"r"), Namespace(b"urn:example:r"))],
        );
    }

    match r.read_resolved_event()? {
        (Unbound, End(e)) => assert_eq!(e.name(), QName(b"a")),
        e => panic!("expecting End event, got {e:?}"),
    }
    Ok(())
}

/// Although the XML specification [recommends against] names whose local part
/// starts with the letters `xml` (case-insensitively), it also mandates that
/// processors *must not* treat them as fatal errors, so parsing continues and it
/// is up to the caller to check the constraint later.
///
/// [recommends against]: https://www.w3.org/TR/xml-names11/#xmlReserved
#[test]
fn reserved_name() -> TestResult {
    let mut r =
        NsReader::from_str(r#"<a xmlns-something="reserved attribute name" xmlns="www1"/>"#);

    match r.read_resolved_event()? {
        (ns, Empty(_)) => assert_eq!(ns, Bound(Namespace(b"www1"))),
        e => panic!("expected empty element bound to namespace 'www1', got {e:?}"),
    }
    Ok(())
}

/// The `xml` prefix is always bound, even without a declaration; the `xmlns`
/// prefix resolves to the namespace-declaration namespace. Neither shows up in
/// [`prefixes()`](NsReader::prefixes).
///
/// [`NsReader::prefixes`]: oxixml_xml::reader::NsReader::prefixes
#[test]
fn reserved_prefixes_are_predeclared() -> TestResult {
    let mut r = NsReader::from_str(r#"<a xml:lang="en"/>"#);
    let _ = r.read_event()?;

    assert_eq!(
        r.resolve(QName(b"xml:lang"), true).0,
        Bound(Namespace(b"http://www.w3.org/XML/1998/namespace")),
    );
    assert_eq!(
        r.resolve(QName(b"xmlns:x"), true).0,
        Bound(Namespace(b"http://www.w3.org/2000/xmlns/")),
    );
    assert_eq!(r.prefixes().collect::<Vec<_>>(), vec![]);
    Ok(())
}

/// Re-binding `xml` to its own namespace is a no-op; binding it to anything else
/// — or binding the `xmlns` prefix at all — is an error.
#[test]
fn reserved_prefixes_cannot_be_rebound() {
    let mut r =
        NsReader::from_str(r#"<a xml:base="b" xmlns:xml="http://www.w3.org/XML/1998/namespace"/>"#);
    assert!(matches!(r.read_event(), Ok(Empty(_))));

    let mut r = NsReader::from_str(r#"<a xmlns:xml="urn:not:the:xml:namespace"/>"#);
    assert!(matches!(r.read_event(), Err(Error::Namespace(_))));

    let mut r = NsReader::from_str(r#"<a xmlns:xmlns="http://www.w3.org/2000/xmlns/"/>"#);
    assert!(matches!(r.read_event(), Err(Error::Namespace(_))));

    // Binding the reserved namespaces to a non-reserved prefix is an error too.
    let mut r = NsReader::from_str(r#"<a xmlns:p="http://www.w3.org/XML/1998/namespace"/>"#);
    assert!(matches!(r.read_event(), Err(Error::Namespace(_))));

    let mut r = NsReader::from_str(r#"<a xmlns:p="http://www.w3.org/2000/xmlns/"/>"#);
    assert!(matches!(r.read_event(), Err(Error::Namespace(_))));
}

/// The full element resolution matrix over nested, default, shadowed and
/// undeclared prefixes.
#[test]
fn nested_resolution_matrix() -> TestResult {
    let xml = "\
        <x:a xmlns:x='ns1' xmlns='dflt1'>\
            <b/>\
            <x:c/>\
            <y:d/>\
            <x:e xmlns:x='ns2' xmlns='dflt2'>\
                <f/>\
                <x:g/>\
            </x:e>\
            <x:h/>\
            <i/>\
        </x:a>";
    let mut r = NsReader::from_str(xml);

    /// Reads the next element event and asserts its resolved namespace and local
    /// name.
    fn assert_next(r: &mut NsReader<&[u8]>, expected: ResolveResult<'_>, local: &[u8]) {
        let event = r.read_event().expect("valid event");
        let name = match &event {
            Start(e) | Empty(e) => e.name(),
            End(e) => e.name(),
            other => panic!("expected an element event, got {other:?}"),
        };
        let (ns, actual_local) = r.resolve_element(name);
        assert_eq!(ns, expected, "namespace of {event:?}");
        assert_eq!(actual_local.into_inner(), local, "local name of {event:?}");
    }

    // <x:a> -> ns1
    assert_next(&mut r, Bound(Namespace(b"ns1")), b"a");
    // <b/> -> the default namespace applies to unprefixed elements
    assert_next(&mut r, Bound(Namespace(b"dflt1")), b"b");
    // <x:c/> -> ns1
    assert_next(&mut r, Bound(Namespace(b"ns1")), b"c");
    // <y:d/> -> `y` is never declared
    assert_next(&mut r, Unknown(b"y".to_vec()), b"d");
    // <x:e ...> shadows both `x` and the default namespace
    assert_next(&mut r, Bound(Namespace(b"ns2")), b"e");
    assert_next(&mut r, Bound(Namespace(b"dflt2")), b"f");
    assert_next(&mut r, Bound(Namespace(b"ns2")), b"g");
    // </x:e> is still resolved in its own scope
    assert_next(&mut r, Bound(Namespace(b"ns2")), b"e");
    // ... and the outer bindings are restored afterwards
    assert_next(&mut r, Bound(Namespace(b"ns1")), b"h");
    assert_next(&mut r, Bound(Namespace(b"dflt1")), b"i");
    assert_next(&mut r, Bound(Namespace(b"ns1")), b"a");
    assert_eq!(r.read_event()?, Eof);
    Ok(())
}

/// `resolve(name, attribute)` differs from `resolve_element` / `resolve_attribute`
/// only in how it treats an **absent** prefix.
#[test]
fn element_versus_attribute_resolution() -> TestResult {
    let mut r = NsReader::from_str(r#"<tag xmlns="dflt" xmlns:p="pfx" a="1" p:b="2"/>"#);
    let e = match r.read_event()? {
        Empty(e) => e,
        other => panic!("expected Empty, got {other:?}"),
    };

    // Unprefixed element name: the default namespace applies.
    assert_eq!(r.resolve(QName(b"n"), false).0, Bound(Namespace(b"dflt")));
    assert_eq!(r.resolve_element(QName(b"n")).0, Bound(Namespace(b"dflt")));
    // Unprefixed attribute name: the default namespace never applies.
    assert_eq!(r.resolve(QName(b"n"), true).0, Unbound);
    assert_eq!(r.resolve_attribute(QName(b"n")).0, Unbound);

    // A prefixed name resolves identically in both positions.
    for attribute in [false, true] {
        assert_eq!(
            r.resolve(QName(b"p:n"), attribute).0,
            Bound(Namespace(b"pfx")),
        );
        assert_eq!(
            r.resolve(QName(b"q:n"), attribute).0,
            Unknown(b"q".to_vec()),
        );
    }

    // The same, applied to the real attributes of the element.
    let attrs = e
        .attributes()
        .map(|a| a.expect("attribute parsing succeeds"))
        .filter(|a| a.key.as_namespace_binding().is_none())
        .map(|a| {
            let (ns, local) = r.resolve_attribute(a.key);
            (ns, local.into_inner().to_vec())
        })
        .collect::<Vec<_>>();
    assert_eq!(
        attrs,
        vec![
            (Unbound, b"a".to_vec()),
            (Bound(Namespace(b"pfx")), b"b".to_vec()),
        ],
    );
    Ok(())
}

/// An `xmlns:p=""` declaration undeclares the prefix `p` for the extent of its
/// scope, which is reported as [`Unknown`] rather than [`Unbound`].
#[test]
fn undeclared_prefix() -> TestResult {
    let mut r = NsReader::from_str("<a xmlns:p='ns'><b xmlns:p=''><p:c/></b><p:d/></a>");

    assert_eq!(r.read_resolved_event()?.0, Unbound); // <a>
    assert_eq!(r.read_resolved_event()?.0, Unbound); // <b>
    assert_eq!(r.read_resolved_event()?.0, Unknown(b"p".to_vec())); // <p:c/>
    assert_eq!(r.read_resolved_event()?.0, Unbound); // </b>
    assert_eq!(r.read_resolved_event()?.0, Bound(Namespace(b"ns"))); // <p:d/>
    Ok(())
}

/// [`prefixes()`](oxixml_xml::reader::NsReader::prefixes) yields the in-scope
/// bindings without the reserved ones, with shadowed prefixes reported once.
#[test]
fn prefixes_iterator() -> TestResult {
    let mut r = NsReader::from_str("<a xmlns='d1' xmlns:x='n1'><b xmlns:x='n2' xmlns:y='n3'/></a>");

    let _ = r.read_event()?; // <a>
    let mut prefixes = r.prefixes().collect::<Vec<_>>();
    prefixes.sort_by(|l, r| l.1.as_ref().cmp(r.1.as_ref()));
    assert_eq!(
        prefixes,
        vec![
            (PrefixDeclaration::Default, Namespace(b"d1")),
            (PrefixDeclaration::Named(b"x"), Namespace(b"n1")),
        ],
    );

    let _ = r.read_event()?; // <b>
    let mut prefixes = r.prefixes().collect::<Vec<_>>();
    prefixes.sort_by(|l, r| l.1.as_ref().cmp(r.1.as_ref()));
    assert_eq!(
        prefixes,
        vec![
            (PrefixDeclaration::Default, Namespace(b"d1")),
            // `x` is reported once, with the innermost binding.
            (PrefixDeclaration::Named(b"x"), Namespace(b"n2")),
            (PrefixDeclaration::Named(b"y"), Namespace(b"n3")),
        ],
    );
    Ok(())
}

/// `bindings_of(level)` exposes exactly the declarations made by one element.
#[test]
fn bindings_of_level() -> TestResult {
    let mut r = NsReader::from_str("<a xmlns='d1'><b xmlns:x='n2'/></a>");

    let _ = r.read_event()?; // <a>
    assert_eq!(r.resolver().level(), 1);
    let _ = r.read_event()?; // <b/>
    assert_eq!(r.resolver().level(), 2);

    assert_eq!(
        r.resolver().bindings_of(1).collect::<Vec<_>>(),
        vec![(PrefixDeclaration::Default, Namespace(b"d1"))],
    );
    assert_eq!(
        r.resolver().bindings_of(2).collect::<Vec<_>>(),
        vec![(PrefixDeclaration::Named(b"x"), Namespace(b"n2"))],
    );
    // Level 0 holds the two reserved bindings.
    assert_eq!(
        r.resolver().bindings_of(0).collect::<Vec<_>>(),
        vec![
            (
                PrefixDeclaration::Named(b"xml"),
                Namespace(b"http://www.w3.org/XML/1998/namespace"),
            ),
            (
                PrefixDeclaration::Named(b"xmlns"),
                Namespace(b"http://www.w3.org/2000/xmlns/"),
            ),
        ],
    );
    Ok(())
}

/// The per-element declaration limit is configurable through `resolver_mut`.
#[test]
fn too_many_declarations() -> TestResult {
    let mut r = NsReader::from_str(r#"<tag xmlns:a="1" xmlns:b="2" xmlns:c="3"/>"#);
    assert_eq!(r.resolver().max_declarations_per_element(), 256);
    r.resolver_mut().set_max_declarations_per_element(2);
    assert_eq!(r.resolver().max_declarations_per_element(), 2);
    assert!(matches!(r.read_event(), Err(Error::Namespace(_))));

    // Raising the limit above the number of declarations makes it parse.
    let mut r = NsReader::from_str(r#"<tag xmlns:a="1" xmlns:b="2" xmlns:c="3"/>"#);
    r.resolver_mut().set_max_declarations_per_element(3);
    assert!(matches!(r.read_event(), Ok(Empty(_))));
    Ok(())
}

/// The buffered (`BufRead`) and borrowing (`&str`) readers must agree on the
/// resolved namespace of every event.
#[test]
fn buffered_and_borrowed_readers_agree() -> TestResult {
    let xml = include_str!("documents/namespaces.xml");

    let mut borrowed = NsReader::from_str(xml);
    let mut buffered = NsReader::from_reader(xml.as_bytes());
    let mut buf = Vec::new();
    loop {
        let (lhs_ns, lhs) = borrowed.read_resolved_event()?;
        let lhs = (lhs_ns.clone(), lhs.into_owned());
        buf.clear();
        let (rhs_ns, rhs) = buffered.read_resolved_event_into(&mut buf)?;
        let rhs = (rhs_ns.clone(), rhs.into_owned());
        assert_eq!(lhs, rhs);
        if lhs.1 == Eof {
            break;
        }
    }
    Ok(())
}

/// `Deref` exposes the inspection methods of the underlying [`Reader`], and the
/// namespace reader hands back its input via the usual accessors.
///
/// [`Reader`]: oxixml_xml::reader::Reader
#[test]
fn deref_and_accessors() -> TestResult {
    let mut r = NsReader::from_str("<tag/>");
    assert!(!r.config().expand_empty_elements);
    let _ = r.read_event()?;
    assert_eq!(r.buffer_position(), 6);
    assert_eq!(r.error_position(), 0);

    let mut r = NsReader::from_reader(&b"<tag/>"[..]);
    let mut buf = Vec::new();
    let _ = r.read_event_into(&mut buf)?;
    assert!(r.get_ref().is_empty());
    let _ = r.get_mut();
    assert!(r.into_inner().is_empty());
    Ok(())
}

/// A start tag whose namespace declarations are malformed must not corrupt the
/// resolver: parsing continues and the (unparsable) attributes are ignored.
#[test]
fn malformed_declarations_are_skipped() -> TestResult {
    let mut r = NsReader::from_str("<a xmlns:p='ns' broken><p:b/></a>");
    r.config_mut().check_end_names = false;

    assert_eq!(r.read_resolved_event()?.0, Unbound); // <a>
    assert_eq!(r.read_resolved_event()?.0, Bound(Namespace(b"ns"))); // <p:b/>
    Ok(())
}

/// `resolve_event` is only meaningful for element events; everything else is
/// reported as [`Unbound`].
#[test]
fn non_element_events_are_unbound() -> TestResult {
    let xml = "<a xmlns='ns'><!--c--><?pi?><![CDATA[x]]>t&e;</a>";
    let mut r = NsReader::from_str(xml);

    assert_eq!(r.read_resolved_event()?.0, Bound(Namespace(b"ns")));
    for _ in 0..5 {
        let (ns, event) = r.read_resolved_event()?;
        assert_eq!(ns, Unbound, "unexpected namespace on {event:?}");
        assert!(!matches!(event, Start(_) | Empty(_) | End(_)));
    }
    assert_eq!(r.read_resolved_event()?.0, Bound(Namespace(b"ns"))); // </a>
    Ok(())
}

/// A `BytesStart` can be pushed into a standalone resolver, which is what the
/// `NsReader` does internally.
#[test]
fn manual_resolver_push_and_pop() {
    use oxixml_xml::name::NamespaceResolver;

    let mut resolver = NamespaceResolver::default();
    assert_eq!(resolver.level(), 0);
    assert_eq!(resolver.resolve_element(QName(b"n")).0, Unbound);

    resolver
        .push(&BytesStart::from_content(r#"a xmlns="d" xmlns:p="n""#, 1))
        .expect("push succeeds");
    assert_eq!(resolver.level(), 1);
    assert_eq!(
        resolver.resolve_element(QName(b"n")).0,
        Bound(Namespace(b"d"))
    );
    assert_eq!(
        resolver.resolve_element(QName(b"p:n")).0,
        Bound(Namespace(b"n")),
    );

    resolver.pop();
    assert_eq!(resolver.level(), 0);
    assert_eq!(resolver.resolve_element(QName(b"n")).0, Unbound);
    assert_eq!(
        resolver.resolve_element(QName(b"p:n")).0,
        Unknown(b"p".to_vec()),
    );
}

/// `ResolveResult` converts into `Option<Namespace>`, failing only for `Unknown`.
#[test]
fn resolve_result_conversion() {
    assert_eq!(Option::<Namespace>::try_from(Unbound), Ok(None));
    assert_eq!(
        Option::<Namespace>::try_from(Bound(Namespace(b"ns"))),
        Ok(Some(Namespace(b"ns"))),
    );
    assert!(Option::<Namespace>::try_from(Unknown(b"p".to_vec())).is_err());
}

/// Sanity check that an [`Event`] round-trips through the resolver unchanged.
#[test]
fn resolve_event_preserves_the_event() -> TestResult {
    let mut r = NsReader::from_str("<a xmlns='ns'/>");
    let (ns, event) = r.read_resolved_event()?;
    assert_eq!(ns, Bound(Namespace(b"ns")));
    assert_eq!(
        event,
        Event::Empty(BytesStart::from_content("a xmlns='ns'", 1)),
    );
    Ok(())
}
