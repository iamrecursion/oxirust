# oxixml-svg

A typed [SVG 1.1 (Second Edition) / SVG 2](https://www.w3.org/TR/SVG11/) document model, microsyntax
parser/serializer, geometry toolkit, programmatic builder, and linter, for the
[OxiXML](https://github.com/cool-japan/oxixml) workspace, built entirely on an `oxixml-dom` tree.

SVG is the one graphics format the rest of the COOLJAPAN ecosystem cannot avoid — dataframe
plotting, CAD export, and document/media pipelines all need to emit or consume it — and each has so
far done so with its own hand-rolled `format!("<svg ...")` emitter: unescaped, untyped, and unable to
answer even a simple "what is this path's bounding box" question. This crate is the shared
foundation instead: parse real-world SVG into an ordinary `oxixml-dom` `Document`, read it back out
through typed accessors, compute exact geometry (Bézier/arc evaluation, bounding boxes, arc length,
flattening, the `viewBox`-to-viewport transform), build new documents through a chainable,
escaping-safe builder, and lint the result against the closed 88-element/216-attribute vocabulary —
all without ever rasterizing a single pixel; turning geometry and paint into actual output pixels is
a downstream renderer's job, not this crate's.

## Install

```toml
[dependencies]
oxixml-svg = "0.1.2"
```

## Usage

```rust
use oxixml_svg::{Length, LengthUnit, Linter, Rect, SvgBuilder, SvgDocument};

fn main() {
    // Parse a document and read a typed presentation attribute.
    let doc = SvgDocument::parse_str(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">
             <rect x="10" y="20" width="50" height="30"/>
           </svg>"#,
    )
    .expect("well-formed SVG");
    assert_eq!(doc.width(), Some(Ok(Length::new(200.0, LengthUnit::None))));

    // Compute the rect's bounding box through the geometry engine: turn it
    // into its equivalent path, then measure that path.
    let rect = doc.svg_element().children().next().expect("the rect element");
    let ctx = oxixml_svg::values::length::LengthContext::default();
    let path = oxixml_svg::tree::shapes::equivalent_path(rect, &ctx)
        .expect("rect is a recognized shape")
        .expect("its geometry attributes are all valid");
    let bbox =
        oxixml_svg::geometry::path_ops::bounding_box(&path).expect("a non-empty rect has a bbox");
    assert_eq!(bbox, Rect::new(10.0, 20.0, 50.0, 30.0).expect("a valid rect"));

    // Lint the document: nothing here is unknown, deprecated, or dangling.
    let diagnostics = Linter::new().lint(doc.document());
    assert!(diagnostics.is_empty());

    // Build a small document and check it serializes and re-parses cleanly.
    let mut built = SvgBuilder::new(20.0, 20.0);
    built.rect(0.0, 0.0, 10.0, 10.0);
    let xml = built.to_string();
    let reparsed = SvgDocument::parse_str(&xml).expect("builder output is well-formed SVG");
    assert_eq!(reparsed.to_xml(), xml, "serialization is a fixed point");
}
```

## What is covered

* **Microsyntax parsers/serializers** — one pair per SVG attribute grammar: `<length>`/`<angle>`
  (`values::length`), `<color>` (named/hex/`rgb()`/`hsl()`, `values::color`), `<paint>` including
  `url(...)` references and fallbacks (`values::paint`), the `transform` attribute's transform-list
  grammar (`values::transform`), `viewBox`/`preserveAspectRatio` (`values::aspect`), full path data —
  every command, arc-flag lexing, SVG's own error-recovery rule, and both a canonical and a
  shortest-round-tripping serializer (`values::path`) — `points` (with its own odd-count recovery
  rule, `values::points`), SMIL clock values and timing specifiers (`values::clock`), and `<funciri>`
  (`values::iri`).
* **Vocabulary catalog** — every SVG element as `names::elements::ElementId` (88 names, spanning
  SVG 1.1 Second Edition and SVG 2, with deprecated/draft/experimental status tracked per element)
  and every attribute as `names::attrs::AttrId` (216 names), plus `names::content`'s tables for which
  children and which attributes each element admits.
* **Style engine** — the presentation-property catalog (`style::property::PropertyId`: initial
  values, inheritance, animatability), the `style=""` attribute parser (`style::attr`), and a
  miniature `<style>` stylesheet engine (`style::css`) covering type/class/id/universal selectors,
  grouping, the descendant combinator, and CSS specificity.
* **Geometry engine** — affine matrices (`geometry::matrix`), cubic/quadratic Bézier evaluation with
  splitting, extrema, arc-length, and adaptive flattening (`geometry::bezier`), elliptical-arc
  endpoint/center parameterization and arc-to-cubic conversion (`geometry::arc`), whole-path
  operations — bounding box, total length, point-at-length, flattening (`geometry::path_ops`) — and
  the `viewBox`-to-viewport transform (`geometry::viewport`).
* **Typed document model** — `SvgDocument`/`SvgElement` over an ordinary `oxixml-dom` tree, an `id`
  index with cycle-checked `use`/gradient/pattern reference resolution (`tree::refs`), a CSS cascade
  resolver for computed style (`tree::cascade::StyleResolver`), and shape-to-equivalent-path
  conversion (`tree::shapes`).
* **Builder** — `SvgBuilder`, a chainable, escaping-safe builder with shape convenience methods,
  gradient/pattern/clip-path/mask/marker construction (`writer::paint_defs`), and `PathBuilder` for
  path data, replacing hand-rolled string formatting; every failure mode (an invalid name, a
  non-finite number, an unbalanced `close()`) is recorded as a `BuildFlaw` instead of panicking.
* **Linter** — `Linter` validates a plain `oxixml-dom::Document` against unknown element/attribute
  names, content-model violations, duplicate/dangling/circular references, deprecated constructs,
  and every microsyntax this crate implements, producing severity-ranked diagnostics with stable
  kebab-case codes.
* **W3C SVG 1.1 corpus harness** — `tests/w3c_svg.rs` parses, round-trips, lints, and
  strictly re-validates every attribute of the vendored W3C SVG 1.1 Second Edition test suite; see
  Conformance below.

## Conformance

Measured against the W3C SVG 1.1 (Second Edition) test suite (525 `.svg` files under `svg/`),
discovered through `OXIXML_SVG_SUITE_DIR` or `OXIXML_XML_SUITES_DIR` (the suite is not vendored, and
the harness skips silently without it):

| check | result |
|---|---|
| parses as an `SvgDocument` | **525 / 525** |
| serialize → re-parse preserves element/attribute shape | **525 / 525** |
| `Linter::lint` completes without panicking | **525 / 525** |
| `d`/`points`/`transform`/`viewBox`/`preserveAspectRatio` strictly parse | **2703 / 2703** (outside a documented, 20-value allowlist) |

The fourth row's allowlist is not a blanket per-file exemption: it names 20 *exact attribute values*
(not whole files) that the corpus deliberately gets wrong, verified by inspecting every strict-parse
failure individually rather than assumed. Two `paths-data-*-f.svg` files misuse elliptical-arc flags
and a bogus `#` on purpose to exercise this crate's path-data error recovery (7 values, re-parsed
with `PathData::parse_recover` and asserted to still report an error); one `shapes-polygon-03-t.svg`
supplies an odd coordinate count on purpose to exercise the `points` odd-count recovery rule (4
values, re-parsed with the lenient `parse_points`); and four files each carry one deliberately
zero-separator `transform` value with no comma-wsp between two transform functions (9 occurrences,
excluded outright — `TransformList` has no recovery mode). Every other value of every one of the
five microsyntaxes, on every SVG-namespace element in the whole corpus, parses strictly. Every
allowlist entry is re-verified on every run: if a value ever stops being rejected, the test fails
instead of the exemption silently going stale.

## License

Licensed under the Apache License, Version 2.0
([LICENSE](https://github.com/cool-japan/oxixml/blob/main/LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).
