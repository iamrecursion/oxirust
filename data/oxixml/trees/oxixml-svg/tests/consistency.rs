//! Cross-module invariants between independently-authored, now-frozen
//! catalogs.
//!
//! [`names::attrs::AttrId`] and [`style::property::PropertyId`] were written
//! against the same specification requirements catalog by different agents
//! in different waves, and each names its own true-set for "is this a
//! presentation (property-mirroring) attribute/property". This file checks
//! the two agree, in both directions, via each type's own `from_name`/
//! `as_str` round-trip. Per this crate's finalization mission, any mismatch
//! found here is **reported, not fixed**: both `names::attrs` and
//! `style::property` are frozen implementation modules outside this file's
//! ownership. A mismatch found is recorded in the assertion message and
//! pinned against an explicit, documented allowlist below, so that:
//!
//! - the currently-known gap keeps passing without this test being
//!   vacuously green, and
//! - a *new*, previously-unseen mismatch (a regression introduced by a
//!   future change to either frozen catalog) still fails the test loudly.
//!
//! `names::elements::ElementId` and `names::attrs::AttrId`'s own exhaustive
//! `from_name(as_str())` round-trips already exist as inline tests in their
//! defining modules; they are not repeated here.

use oxixml_svg::names::AttrId;
use oxixml_svg::style::PropertyId;

/// Attribute names that [`AttrId::is_presentation`] marks as a presentation
/// attribute but that [`PropertyId::from_name`] does not recognize at all.
///
/// Empty: `font-kerning` used to be the one name here — SVG 2's catalog names
/// it only in the inherited-properties prose, never as its own tabulated row
/// with an initial value, so `PropertyId`'s catalog-transcribed rows had no
/// variant for it even though `names::attrs::groups` already modeled
/// `AttrId::FontKerning` as a real, authorable presentation attribute.
/// `style::property` now fills that row in directly from CSS Fonts Module
/// Level 3/4 (see that module's own documentation), so the two catalogs
/// agree again. Kept as an explicit, non-empty-checkable constant (rather
/// than a bare `assert!(empty)`) for the same reason as its sibling below —
/// so a future divergence in either frozen catalog fails this test loudly
/// instead of silently.
const ATTRS_MARKED_PRESENTATION_WITH_NO_PROPERTY: &[&str] = &[];

/// Property names [`PropertyId::presentation_attribute_allowed`] allows as a
/// presentation attribute but that [`AttrId::from_name`] does not recognize,
/// or recognizes without [`AttrId::is_presentation`] being true.
///
/// Empty: every `PropertyId` this crate models has a same-named,
/// `is_presentation`-true `AttrId` counterpart. Kept as an explicit,
/// non-empty-checkable constant (rather than a bare `assert!(empty)`) for
/// the same reason as its sibling above — so a future divergence in either
/// frozen catalog fails this test loudly instead of silently.
const PROPERTIES_WITH_NO_MATCHING_PRESENTATION_ATTR: &[&str] = &[];

#[test]
fn every_attr_id_marked_presentation_has_a_matching_property_id_except_the_documented_gap() {
    let mut mismatches: Vec<&str> = AttrId::ALL
        .iter()
        .copied()
        .filter(|attr| attr.is_presentation())
        .filter_map(|attr| {
            let name = attr.as_str();
            match PropertyId::from_name(name) {
                Some(property) if property.presentation_attribute_allowed() => None,
                _ => Some(name),
            }
        })
        .collect();
    mismatches.sort_unstable();

    let mut expected = ATTRS_MARKED_PRESENTATION_WITH_NO_PROPERTY.to_vec();
    expected.sort_unstable();

    assert_eq!(
        mismatches, expected,
        "AttrId::is_presentation() attributes with no matching PropertyId (or one that \
         disallows a presentation attribute) must exactly match the documented allowlist; \
         found {mismatches:?}, expected {expected:?}"
    );
}

#[test]
fn every_property_id_allowing_a_presentation_attribute_has_a_matching_attr_id() {
    // `PropertyId::presentation_attribute_allowed` is unconditionally `true`
    // for every variant today (see that method's own documentation), so this
    // iterates the whole `PropertyId::ALL` catalog; the `filter` is kept so
    // this test does not silently stop checking anything if that ever
    // changes.
    let mut mismatches: Vec<&str> = PropertyId::ALL
        .iter()
        .copied()
        .filter(|property| property.presentation_attribute_allowed())
        .filter_map(|property| {
            let name = property.as_str();
            match AttrId::from_name(name) {
                Some(attr) if attr.is_presentation() => None,
                _ => Some(name),
            }
        })
        .collect();
    mismatches.sort_unstable();

    let mut expected = PROPERTIES_WITH_NO_MATCHING_PRESENTATION_ATTR.to_vec();
    expected.sort_unstable();

    assert_eq!(
        mismatches, expected,
        "PropertyId properties with no matching, is_presentation() AttrId must exactly match \
         the documented allowlist (expected to be empty); found {mismatches:?}"
    );
}

/// A same-named `AttrId`/`PropertyId` pair should describe the same
/// attribute in both directions: this pins the round-trip
/// `AttrId::from_name(id.as_str()).as_str() == id.as_str()` for every
/// `is_presentation` attribute, i.e. that `as_str`/`from_name` themselves
/// (not just the presentation-flag cross-check above) are consistent for
/// the exact set this file cares about. `AttrId`'s own module already tests
/// this exhaustively over every variant; this is a narrow, documentation-
/// oriented restatement scoped to the presentation subset, kept so a reader
/// of this file does not have to trust that unrelated guarantee on faith.
#[test]
fn presentation_attr_ids_round_trip_through_as_str_and_from_name() {
    for attr in AttrId::ALL
        .iter()
        .copied()
        .filter(|attr| attr.is_presentation())
    {
        assert_eq!(
            AttrId::from_name(attr.as_str()),
            Some(attr),
            "AttrId::{attr:?} does not round-trip through as_str/from_name"
        );
    }
}
