//! End-to-end tests of the public API, using only `oxixml_svg::...` paths a
//! downstream crate would reach for: parsing, typed reads, the cascade,
//! geometry, the builder, the linter, reference resolution, and error paths.

use oxixml_svg::names::{AttrId, ElementId};
use oxixml_svg::tree::refs::{RefError, resolve_use};
use oxixml_svg::tree::{IdIndex, StyleResolver};
use oxixml_svg::values::LengthContext;
use oxixml_svg::{
    Color, Diagnostic, Length, LengthUnit, Linter, Paint, Rect, SvgBuilder, SvgDocument, SvgError,
};

/// Scenario 1: parse → typed attribute read → cascade computed style →
/// shape-equivalent path → bounding box.
#[test]
fn parse_typed_read_cascade_and_bounding_box() {
    let doc = SvgDocument::parse_str(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
              <style>rect { fill: blue; }</style>
              <rect id="r" x="10" y="20" width="30" height="40" style="stroke: green"/>
            </svg>"##,
    )
    .expect("well-formed SVG");

    let rect = doc
        .svg_element()
        .children()
        .find(|element| element.kind() == Some(ElementId::Rect))
        .expect("the document has a rect element");

    // A typed attribute read straight off the element.
    assert_eq!(
        rect.length_attr(AttrId::X),
        Some(Ok(Length::new(10.0, LengthUnit::None)))
    );

    // The cascade: `fill` has no style="" or presentation attribute, so the
    // matched stylesheet rule wins; `stroke` is set inline, which always
    // outranks a stylesheet rule.
    let resolver = StyleResolver::new(&doc);
    assert_eq!(
        resolver.fill(rect),
        Ok(Paint::Color(Color::rgb(0x00, 0x00, 0xFF)))
    );
    assert_eq!(
        resolver.stroke(rect),
        Ok(Paint::Color(Color::rgb(0x00, 0x80, 0x00)))
    );

    // The shape's equivalent path and its exact bounding box.
    let ctx = LengthContext::default();
    let path = oxixml_svg::tree::shapes::equivalent_path(rect, &ctx)
        .expect("rect is a recognized shape")
        .expect("its geometry attributes are all valid");
    let bbox =
        oxixml_svg::geometry::path_ops::bounding_box(&path).expect("a non-empty rect has a bbox");
    assert_eq!(
        bbox,
        Rect::new(10.0, 20.0, 30.0, 40.0).expect("a valid rect")
    );
}

/// Scenario 2: build a document programmatically, serialize it, re-parse
/// it, and confirm the linter reports nothing.
#[test]
fn builder_serialize_reparse_and_lint_clean() {
    let mut builder = SvgBuilder::new(120.0, 80.0);
    builder.rect(10.0, 10.0, 40.0, 20.0);
    builder.fill(Paint::Color(Color::rgb(0x11, 0x22, 0x33)));
    builder.circle(90.0, 40.0, 15.0);
    builder.stroke(Paint::CurrentColor);
    assert!(
        builder.flaws().is_empty(),
        "well-formed calls should not record any flaw: {:?}",
        builder.flaws()
    );

    let xml = builder.to_string();
    let reparsed = SvgDocument::parse_str(&xml).expect("builder output is well-formed SVG");

    let diagnostics = Linter::new().lint(reparsed.document());
    assert!(
        diagnostics.is_empty(),
        "expected a clean lint of builder output, got {diagnostics:?}"
    );
}

/// Scenario 3: the linter catches every problem seeded into a hand-written
/// document — an unknown element, a negative dimension, and a dangling
/// fragment reference — through nothing but [`Linter::lint`].
#[test]
fn lint_catches_a_seeded_broken_document() {
    let doc = SvgDocument::parse_str(
        r##"<svg xmlns="http://www.w3.org/2000/svg">
              <rect width="-5" height="10"/>
              <weirdElement/>
              <use href="#missing"/>
            </svg>"##,
    )
    .expect("well-formed XML, even though the SVG content is invalid");

    let diagnostics = Linter::new().lint(doc.document());
    let codes: Vec<&str> = diagnostics.iter().map(Diagnostic::code).collect();

    assert!(
        codes.contains(&oxixml_svg::lint::codes::NEGATIVE_DIMENSION),
        "expected negative-dimension among {codes:?}"
    );
    assert!(
        codes.contains(&oxixml_svg::lint::codes::UNKNOWN_ELEMENT),
        "expected unknown-element among {codes:?}"
    );
    assert!(
        codes.contains(&oxixml_svg::lint::codes::DANGLING_REFERENCE),
        "expected dangling-reference among {codes:?}"
    );
}

/// Scenario 4: reference-cycle detection through the public
/// [`oxixml_svg::tree::refs`] API — a `<use>` that references itself.
#[test]
fn refs_cycle_detection_through_the_public_api() {
    let doc = SvgDocument::parse_str(
        r##"<svg xmlns="http://www.w3.org/2000/svg">
              <use id="u" href="#u"/>
            </svg>"##,
    )
    .expect("well-formed SVG");

    let index = IdIndex::build(&doc);
    let use_node = index
        .get("u")
        .expect("the use element is indexed by its id");

    let error = resolve_use(doc.document(), &index, use_node)
        .expect_err("a use element that references itself is a cycle");
    assert_eq!(error, RefError::Cycle { at: use_node });
}

/// Scenario 5: the two document-level error paths — a well-formed XML
/// document whose root is not `svg`, and XML that is not well-formed at
/// all. [`SvgError`] derives only [`std::fmt::Debug`], so these are
/// destructured/matched rather than compared with `assert_eq!`.
#[test]
fn error_paths_not_svg_root_and_malformed_xml() {
    let not_svg = SvgDocument::parse_str("<catalog/>").expect_err("catalog is not an svg root");
    match not_svg {
        SvgError::NotSvg { found: Some(name) } => assert_eq!(&*name, "catalog"),
        other => panic!("expected NotSvg {{ found: Some(\"catalog\") }}, got {other:?}"),
    }

    let malformed =
        SvgDocument::parse_str("<svg>").expect_err("an unclosed element is not well-formed XML");
    assert!(
        matches!(malformed, SvgError::Xml(_)),
        "expected SvgError::Xml, got {malformed:?}"
    );
}
