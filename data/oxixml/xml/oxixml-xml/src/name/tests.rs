use super::*;
use ResolveResult::*;

#[test]
fn qname_decompose() {
    let q = QName(b"pre:local");
    assert_eq!(q.prefix().map(|p| p.into_inner()), Some(&b"pre"[..]));
    assert_eq!(q.local_name().into_inner(), b"local");
    let (local, prefix) = q.decompose();
    assert_eq!(local.into_inner(), b"local");
    assert_eq!(prefix.map(|p| p.into_inner()), Some(&b"pre"[..]));

    let plain = QName(b"local");
    assert_eq!(plain.prefix(), None);
    assert_eq!(plain.local_name().into_inner(), b"local");
}

#[test]
fn prefix_special() {
    assert!(QName(b"xml:lang").prefix().expect("prefix").is_xml());
    assert!(QName(b"xmlns:a").prefix().expect("prefix").is_xmlns());
    assert!(!QName(b"other:a").prefix().expect("prefix").is_xml());
}

#[test]
fn namespace_binding_detection() {
    assert_eq!(
        QName(b"xmlns").as_namespace_binding(),
        Some(PrefixDeclaration::Default)
    );
    assert_eq!(
        QName(b"xmlns:p").as_namespace_binding(),
        Some(PrefixDeclaration::Named(b"p"))
    );
    assert_eq!(QName(b"xmlnsx").as_namespace_binding(), None);
    assert_eq!(QName(b"other").as_namespace_binding(), None);
}

#[test]
fn resolve_element_vs_attribute() -> Result<(), NamespaceError> {
    let mut resolver = NamespaceResolver::default();
    resolver.push(&BytesStart::from_content(
        "e xmlns=\"default\" xmlns:p=\"urn:p\"",
        1,
    ))?;

    // Unprefixed element name binds to the default namespace...
    assert_eq!(
        resolver.resolve_element(QName(b"name")).0,
        Bound(Namespace(b"default"))
    );
    // ...but an unprefixed attribute name does not.
    assert_eq!(resolver.resolve_attribute(QName(b"name")).0, Unbound);

    // Prefixed names resolve identically for element and attribute.
    assert_eq!(
        resolver.resolve_element(QName(b"p:name")).0,
        Bound(Namespace(b"urn:p"))
    );
    assert_eq!(
        resolver.resolve_attribute(QName(b"p:name")).0,
        Bound(Namespace(b"urn:p"))
    );

    // Unknown prefix.
    assert_eq!(
        resolver.resolve_element(QName(b"q:name")).0,
        Unknown(b"q".to_vec())
    );
    Ok(())
}

#[test]
fn scoping_push_pop() -> Result<(), NamespaceError> {
    let mut resolver = NamespaceResolver::default();
    resolver.push(&BytesStart::from_content("a xmlns=\"a1\"", 1))?;
    assert_eq!(
        resolver.resolve_element(QName(b"x")).0,
        Bound(Namespace(b"a1"))
    );

    resolver.push(&BytesStart::from_content("b xmlns=\"b1\"", 1))?;
    assert_eq!(
        resolver.resolve_element(QName(b"x")).0,
        Bound(Namespace(b"b1"))
    );

    resolver.pop();
    assert_eq!(
        resolver.resolve_element(QName(b"x")).0,
        Bound(Namespace(b"a1"))
    );

    resolver.pop();
    assert_eq!(resolver.resolve_element(QName(b"x")).0, Unbound);
    Ok(())
}

#[test]
fn undeclaring_default() -> Result<(), NamespaceError> {
    let mut resolver = NamespaceResolver::default();
    resolver.add(PrefixDeclaration::Default, Namespace(b"ns"))?;
    assert_eq!(
        resolver.resolve_element(QName(b"x")).0,
        Bound(Namespace(b"ns"))
    );
    // XML 1.1 empty value undeclares the default namespace.
    resolver.add(PrefixDeclaration::Default, Namespace(b""))?;
    assert_eq!(resolver.resolve_element(QName(b"x")).0, Unbound);
    Ok(())
}

#[test]
fn reserved_prefix_rules() {
    let mut resolver = NamespaceResolver::default();
    // `xml` may only be bound to its reserved URI.
    assert!(
        resolver
            .add(PrefixDeclaration::Named(b"xml"), Namespace(b"other"))
            .is_err()
    );
    assert!(
        resolver
            .add(
                PrefixDeclaration::Named(b"xml"),
                Namespace(b"http://www.w3.org/XML/1998/namespace")
            )
            .is_ok()
    );
    // `xmlns` cannot be bound at all.
    assert!(
        resolver
            .add(PrefixDeclaration::Named(b"xmlns"), Namespace(b"x"))
            .is_err()
    );
    // No other prefix can take the reserved URIs.
    assert!(matches!(
        resolver.add(
            PrefixDeclaration::Named(b"p"),
            Namespace(b"http://www.w3.org/XML/1998/namespace")
        ),
        Err(NamespaceError::InvalidPrefixForXml(_))
    ));
}

#[test]
fn xml_prefix_predefined() {
    // The `xml` prefix resolves without any declaration.
    let resolver = NamespaceResolver::default();
    assert_eq!(
        resolver.resolve_element(QName(b"xml:lang")).0,
        Bound(Namespace(b"http://www.w3.org/XML/1998/namespace"))
    );
}

#[test]
fn too_many_declarations() {
    let mut resolver = NamespaceResolver::default();
    resolver.set_max_declarations_per_element(2);
    assert_eq!(
        resolver.push(&BytesStart::from_content(
            "e xmlns:a='' xmlns:b='' xmlns:c=''",
            1
        )),
        Err(NamespaceError::TooManyDeclarations(2))
    );
}

#[test]
fn resolve_result_try_into() {
    let bound: Result<Option<Namespace>, _> = ResolveResult::Bound(Namespace(b"n")).try_into();
    assert_eq!(bound, Ok(Some(Namespace(b"n"))));
    let unbound: Result<Option<Namespace>, _> = ResolveResult::Unbound.try_into();
    assert_eq!(unbound, Ok(None));
    let unknown: Result<Option<Namespace>, _> = ResolveResult::Unknown(b"p".to_vec()).try_into();
    assert_eq!(unknown, Err(NamespaceError::UnknownPrefix(b"p".to_vec())));
}

#[test]
fn resolve_event_only_element_events() {
    let resolver = NamespaceResolver::default();
    let (res, _) = resolver.resolve_event(Event::Text(crate::events::BytesText::new("hi")));
    assert_eq!(res, ResolveResult::Unbound);
}

/// Regression for the `nesting_level` overflow: the counter is a `usize`, not
/// a `u16`, so nothing realistic can overflow it, but [`NamespaceResolver::push`]
/// still guards the increment with `checked_add` for callers that drive the
/// level directly through [`NamespaceResolver::set_level`]. This exercises
/// that guard directly (a real loop of `usize::MAX` pushes is obviously not
/// feasible), proving the counter type itself is now the fix rather than a
/// depth limit masking it.
#[test]
fn push_past_usize_max_reports_an_error_instead_of_overflowing() {
    let mut resolver = NamespaceResolver::default();
    resolver.set_level(usize::MAX);
    assert_eq!(resolver.level(), usize::MAX);
    assert_eq!(
        resolver.push(&BytesStart::from_content("e xmlns=\"ns\"", 1)),
        Err(NamespaceError::TooDeeplyNested(usize::MAX)),
    );
}

/// Regression for the historical `u16` nesting-level overflow (previously:
/// debug builds panicked with "attempt to add with overflow" and release
/// builds silently wrapped, corrupting namespace scoping) — reproduced
/// directly against [`NamespaceResolver`] rather than through a reader so the
/// test runs in milliseconds. Pushes past the old `u16::MAX` (65_535) bound
/// and spot-checks that prefix resolution at depth > 65_536 is still correct
/// (the binding pushed at the deepest level must still be the one that
/// resolves, and popping back must still reach the reserved-only baseline).
#[test]
fn deep_nesting_beyond_former_u16_bound_does_not_corrupt_resolution() {
    let mut resolver = NamespaceResolver::default();
    let depth: usize = 70_000;
    for i in 0..depth {
        let content = format!("e xmlns:p=\"urn:ns{i}\"");
        resolver
            .push(&BytesStart::from_content(content, 1))
            .expect("push below any configured limit must succeed");
    }
    assert_eq!(resolver.level(), depth);

    // The innermost (deepest, > 65_536) binding must be the one in scope.
    assert_eq!(
        resolver.resolve_element(QName(b"p:x")).0,
        Bound(Namespace(format!("urn:ns{}", depth - 1).as_bytes())),
    );

    for _ in 0..depth {
        resolver.pop();
    }
    assert_eq!(resolver.level(), 0);
    // Only the reserved `xml`/`xmlns` bindings are left once everything pops.
    assert_eq!(
        resolver.resolve_element(QName(b"p:x")).0,
        Unknown(b"p".to_vec())
    );
}
