//! Public-API shape tests: this crate is a drop-in replacement for `oxiri`, so these check the
//! ergonomics that downstream code (in particular `oxirs-core`, which re-exports
//! `oxiri::{Iri, IriParseError}`) relies on — generic storage (`&str`/`String`/`Box<str>`),
//! `Deref<Target = str>`, `Display`/`Debug`, `Eq`/`Hash`/`Ord`, `FromStr`, the `From`/`TryFrom`
//! conversions between `Iri` and `IriRef`, and the allocation-free `resolve_into` path.

use std::collections::HashSet;
use std::str::FromStr;

use oxixml_iri::{Iri, IriRef};

#[test]
fn iri_works_over_borrowed_and_owned_storage() -> Result<(), Box<dyn std::error::Error>> {
    let borrowed: Iri<&str> = Iri::parse("http://example.com/a")?;
    let owned: Iri<String> = Iri::parse("http://example.com/a".to_owned())?;
    let boxed: Iri<Box<str>> = Iri::parse("http://example.com/a".into())?;
    assert_eq!(borrowed.as_str(), owned.as_str());
    assert_eq!(borrowed.as_str(), boxed.as_str());
    assert_eq!(borrowed, owned);
    assert_eq!(owned, boxed.as_ref());
    Ok(())
}

#[test]
fn deref_gives_str_methods_for_free() -> Result<(), Box<dyn std::error::Error>> {
    let iri = Iri::parse("http://example.com/a/b")?;
    assert!(iri.starts_with("http://"));
    assert!(iri.ends_with("/b"));
    assert_eq!(iri.len(), "http://example.com/a/b".len());

    let iri_ref = IriRef::parse("../a/b")?;
    assert!(iri_ref.starts_with(".."));
    Ok(())
}

#[test]
fn display_and_debug_round_trip_the_text() -> Result<(), Box<dyn std::error::Error>> {
    let iri = Iri::parse("http://example.com/a")?;
    assert_eq!(iri.to_string(), "http://example.com/a");
    assert_eq!(format!("{iri:?}"), format!("{:?}", "http://example.com/a"));
    Ok(())
}

#[test]
fn eq_and_hash_agree_across_storage_kinds() -> Result<(), Box<dyn std::error::Error>> {
    let a: Iri<&str> = Iri::parse("http://example.com/a")?;
    let b: Iri<String> = Iri::parse("http://example.com/a".to_owned())?;
    let c: Iri<String> = Iri::parse("http://example.com/b".to_owned())?;
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_eq!(a, "http://example.com/a");
    assert_eq!(a, "http://example.com/a".to_owned());

    let mut set: HashSet<Iri<String>> = HashSet::new();
    set.insert(b);
    set.insert(c);
    assert_eq!(set.len(), 2);
    assert!(set.contains(&Iri::parse("http://example.com/a".to_owned())?));
    Ok(())
}

#[test]
fn ord_is_consistent_with_the_underlying_string() -> Result<(), Box<dyn std::error::Error>> {
    let mut iris = [
        Iri::parse("http://example.com/c".to_owned())?,
        Iri::parse("http://example.com/a".to_owned())?,
        Iri::parse("http://example.com/b".to_owned())?,
    ];
    iris.sort();
    assert_eq!(
        iris.iter().map(Iri::as_str).collect::<Vec<_>>(),
        vec![
            "http://example.com/a",
            "http://example.com/b",
            "http://example.com/c"
        ]
    );
    Ok(())
}

#[test]
fn from_str_parses_into_owned_storage() -> Result<(), Box<dyn std::error::Error>> {
    let iri: Iri<String> = "http://example.com/a".parse()?;
    assert_eq!(iri.as_str(), "http://example.com/a");

    let iri_ref: IriRef<String> = "../a".parse()?;
    assert_eq!(iri_ref.as_str(), "../a");

    assert!(Iri::<String>::from_str("not a valid iri: has spaces").is_err());
    Ok(())
}

#[test]
fn iri_converts_into_iri_ref_and_back() -> Result<(), Box<dyn std::error::Error>> {
    let iri: Iri<String> = Iri::parse("http://example.com/a".to_owned())?;
    let iri_ref: IriRef<String> = iri.clone().into();
    assert_eq!(iri_ref.as_str(), "http://example.com/a");

    let round_tripped: Iri<String> = iri_ref.try_into()?;
    assert_eq!(round_tripped, iri);

    let relative_ref = IriRef::parse("../a".to_owned())?;
    let attempt: Result<Iri<String>, _> = relative_ref.try_into();
    assert!(attempt.is_err());
    Ok(())
}

#[test]
fn borrowed_and_owned_convert_into_each_other() -> Result<(), Box<dyn std::error::Error>> {
    let owned: Iri<String> = Iri::parse("http://example.com/a".to_owned())?;
    let borrowed: Iri<&str> = Iri::from(&owned);
    assert_eq!(borrowed.as_str(), owned.as_str());

    let owned_again: Iri<String> = Iri::from(borrowed);
    assert_eq!(owned_again, owned);
    Ok(())
}

#[test]
fn resolve_into_avoids_allocating_a_fresh_buffer() -> Result<(), Box<dyn std::error::Error>> {
    let base = Iri::parse("http://example.com/dir/file")?;
    let mut buffer = String::from("scratch content to be overwritten: ");
    buffer.clear();
    base.resolve_into(&IriRef::parse("../other")?, &mut buffer)?;
    assert_eq!(buffer, "http://example.com/other");

    // The buffer can be reused across multiple resolutions.
    buffer.clear();
    base.resolve_into(&IriRef::parse("sibling")?, &mut buffer)?;
    assert_eq!(buffer, "http://example.com/dir/sibling");
    Ok(())
}

#[test]
fn resolve_unchecked_matches_resolve_for_already_valid_input()
-> Result<(), Box<dyn std::error::Error>> {
    let base = Iri::parse_unchecked("http://example.com/dir/file");
    let reference = IriRef::parse_unchecked("../other?q");
    let checked =
        Iri::parse("http://example.com/dir/file")?.resolve(&IriRef::parse("../other?q")?)?;
    let unchecked = base.resolve_unchecked(&reference);
    assert_eq!(checked, unchecked);
    Ok(())
}

#[test]
fn parse_unchecked_still_computes_correct_component_offsets() {
    // `parse_unchecked` skips character-class validation but must still compute correct
    // component boundaries, since accessors and `resolve`/`relativize` depend on them.
    let iri = Iri::parse_unchecked("http://user@example.com:80/a/b?q#f");
    assert_eq!(iri.scheme(), "http");
    assert_eq!(iri.authority(), Some("user@example.com:80"));
    assert_eq!(iri.path(), "/a/b");
    assert_eq!(iri.query(), Some("q"));
    assert_eq!(iri.fragment(), Some("f"));
}

#[test]
#[should_panic(expected = "an Iri is always absolute and therefore always has a scheme")]
fn parse_unchecked_on_a_scheme_less_string_panics_on_scheme() {
    // `parse_unchecked` documents that every `Iri` is assumed absolute, so feeding it a
    // scheme-less string succeeds at the call site but panics later, the first time
    // absoluteness is relied on -- here, calling `scheme()`.
    let _ = Iri::parse_unchecked("no-colon-here").scheme();
}

#[test]
fn clone_and_copy_are_available_for_borrowed_storage() -> Result<(), Box<dyn std::error::Error>> {
    let a: Iri<&str> = Iri::parse("http://example.com/a")?;
    let b = a; // `Copy`, not a move.
    assert_eq!(a, b);
    Ok(())
}
