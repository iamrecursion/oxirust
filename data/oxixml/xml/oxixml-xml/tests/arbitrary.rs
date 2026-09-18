//! Regression tests for the optional `arbitrary` feature: `Arbitrary` impls on
//! the event types, added to close the parity gap with quick-xml 0.41's
//! optional `arbitrary` dependency (its `Arbitrary` impls on the event types,
//! used by fuzz targets).
//!
//! These exercise both directions of `BytesStart`'s validation (accepting a
//! non-empty alphanumeric name, rejecting everything else) with concrete,
//! deterministic byte buffers, then sweep a wide range of pseudo-random
//! buffers to confirm the invariant holds generally and that `Event::arbitrary`
//! (the `#[derive(arbitrary::Arbitrary)]` on the enum) can produce every
//! variant without ever panicking -- the property a fuzz harness actually
//! relies on.

use arbitrary::{Arbitrary, Unstructured};
use oxixml_xml::events::Event;
use oxixml_xml::events::{
    BytesCData, BytesDecl, BytesEnd, BytesPI, BytesRef, BytesStart, BytesText,
};

/// A deterministic (not `std`-`Rng`-dependent, so this test is fully
/// reproducible) source of pseudo-random bytes, seeded by `seed`.
fn pseudo_random_bytes(seed: u32, len: usize) -> Vec<u8> {
    let mut x = seed.wrapping_mul(2_654_435_761) ^ 0xdead_beef;
    (0..len)
        .map(|_| {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (x >> 16) as u8
        })
        .collect()
}

/// A buffer of all-ASCII-alphanumeric bytes reliably produces a `BytesStart`
/// with a valid, non-empty, alphanumeric name -- exactly the shape
/// `BytesStart::new`'s "must be a valid name" warning requires.
#[test]
fn bytes_start_arbitrary_accepts_an_alphanumeric_buffer() {
    let data: Vec<u8> = (0..64u8).map(|i| b'a' + (i % 26)).collect();
    let mut u = Unstructured::new(&data);
    let start = BytesStart::arbitrary(&mut u).expect("an all-alphanumeric buffer must be accepted");
    let name = start.name();
    let name = std::str::from_utf8(name.as_ref()).expect("valid UTF-8 name");
    assert!(!name.is_empty());
    assert!(name.chars().all(char::is_alphanumeric));
}

/// An empty buffer cannot supply a non-empty name, so it must be rejected
/// with `arbitrary::Error::IncorrectFormat` rather than panicking or
/// producing a `BytesStart` with an empty (invalid) name.
#[test]
fn bytes_start_arbitrary_rejects_an_empty_buffer() {
    let mut u = Unstructured::new(&[]);
    assert_eq!(
        BytesStart::arbitrary(&mut u),
        Err(arbitrary::Error::IncorrectFormat),
    );
}

/// A buffer that only ever decodes to non-alphanumeric characters (here,
/// repeated `<`, which cannot appear in a `Name` anyway) must be rejected
/// the same way.
#[test]
fn bytes_start_arbitrary_rejects_a_non_alphanumeric_buffer() {
    let data = vec![b'<'; 32];
    let mut u = Unstructured::new(&data);
    assert_eq!(
        BytesStart::arbitrary(&mut u),
        Err(arbitrary::Error::IncorrectFormat),
    );
}

/// Sweeping many pseudo-random buffers: every `Ok` result must still satisfy
/// `BytesStart`'s name invariant (this is the property a fuzz harness
/// actually depends on -- it doesn't hand-pick inputs), and both the success
/// and rejection paths are reliably reached so the validation is proven live,
/// not just present in the source.
#[test]
fn bytes_start_arbitrary_invariant_holds_across_many_buffers() {
    let mut ok = 0;
    let mut err = 0;
    for seed in 0u32..2000 {
        let data = pseudo_random_bytes(seed, 40);
        let mut u = Unstructured::new(&data);
        match BytesStart::arbitrary(&mut u) {
            Ok(start) => {
                ok += 1;
                let name = start.name();
                let name = std::str::from_utf8(name.as_ref())
                    .unwrap_or_else(|e| panic!("seed={seed}: non-UTF-8 name: {e}"));
                assert!(!name.is_empty(), "seed={seed}");
                assert!(
                    name.chars().all(char::is_alphanumeric),
                    "seed={seed} name={name:?}",
                );
            }
            Err(arbitrary::Error::IncorrectFormat) => err += 1,
            Err(other) => panic!("seed={seed}: unexpected error {other:?}"),
        }
    }
    assert!(ok > 0, "expected at least one accepted buffer in the sweep");
    assert!(
        err > 0,
        "expected at least one rejected buffer in the sweep"
    );
}

/// The other `Bytes*` types have no validation of their own (mirroring
/// quick-xml 0.41 exactly: only `BytesStart` and `BytesText` reject anything),
/// so `arbitrary` on them never fails.
#[test]
fn unvalidated_bytes_types_always_succeed() {
    for seed in 0u32..64 {
        let data = pseudo_random_bytes(seed, 40);

        assert!(BytesEnd::arbitrary(&mut Unstructured::new(&data)).is_ok());
        assert!(BytesCData::arbitrary(&mut Unstructured::new(&data)).is_ok());
        assert!(BytesPI::arbitrary(&mut Unstructured::new(&data)).is_ok());
        assert!(BytesRef::arbitrary(&mut Unstructured::new(&data)).is_ok());
        assert!(BytesDecl::arbitrary(&mut Unstructured::new(&data)).is_ok());
    }
}

/// `BytesText` rejects a non-alphanumeric buffer just like `BytesStart` does,
/// but (unlike `BytesStart`) accepts an *empty* one -- quick-xml 0.41 does not
/// check for emptiness there, only the alphanumeric charset.
#[test]
fn bytes_text_arbitrary_rejects_non_alphanumeric_but_allows_empty() {
    let punctuation = vec![b'&'; 16];
    assert_eq!(
        BytesText::arbitrary(&mut Unstructured::new(&punctuation)),
        Err(arbitrary::Error::IncorrectFormat),
    );
    assert!(BytesText::arbitrary(&mut Unstructured::new(&[])).is_ok());
}

/// `#[derive(arbitrary::Arbitrary)]` on `Event` must be able to produce every
/// variant -- proving the derive actually reaches each payload type's own
/// `Arbitrary` impl, rather than e.g. only ever selecting `Event::Eof` -- and,
/// across a wide sweep of pseudo-random buffers, must never panic. That
/// "never panics on arbitrary bytes" property is the entire point of the
/// feature: it is what lets a fuzz harness call `Event::arbitrary` directly on
/// raw fuzzer input instead of hand-rolling a generator.
#[test]
fn event_arbitrary_covers_every_variant_without_panicking() {
    use std::collections::BTreeSet;

    let mut kinds = BTreeSet::new();
    let mut produced_any = false;
    for seed in 0u32..2000 {
        let data = pseudo_random_bytes(seed, 64);
        let mut u = Unstructured::new(&data);
        if let Ok(event) = Event::arbitrary(&mut u) {
            produced_any = true;
            kinds.insert(match event {
                Event::Start(_) => "Start",
                Event::End(_) => "End",
                Event::Empty(_) => "Empty",
                Event::Text(_) => "Text",
                Event::CData(_) => "CData",
                Event::Comment(_) => "Comment",
                Event::Decl(_) => "Decl",
                Event::PI(_) => "PI",
                Event::DocType(_) => "DocType",
                Event::GeneralRef(_) => "GeneralRef",
                Event::Eof => "Eof",
            });
        }
    }
    assert!(produced_any);
    assert_eq!(
        kinds,
        BTreeSet::from([
            "Start",
            "End",
            "Empty",
            "Text",
            "CData",
            "Comment",
            "Decl",
            "PI",
            "DocType",
            "GeneralRef",
            "Eof",
        ]),
        "every Event variant must be reachable through Event::arbitrary",
    );
}

/// `size_hint` must return a well-formed range (lower bound, and if there is
/// an upper bound, lower <= upper) so that callers driving `Unstructured`
/// budgets (e.g. `arbitrary::size_hint::recursion_guard`) don't misbehave.
#[test]
fn event_size_hint_is_well_formed() {
    let (lower, upper) = Event::size_hint(0);
    if let Some(upper) = upper {
        assert!(lower <= upper);
    }
}
