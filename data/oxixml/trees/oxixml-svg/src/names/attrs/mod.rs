//! The catalog of SVG attribute names: the union of the SVG 1.1 Second
//! Edition and SVG 2 attribute vocabularies, including every presentation
//! property that can also be authored as an XML attribute.
//!
//! [`AttrId`] enumerates 216 distinct authored attribute names. This module
//! is split across two files purely for size (`ids` holds the enum, the
//! name table, and the exhaustive lookup tests; `groups` holds the
//! category predicates and their tests) — both are re-exported here as a
//! single [`AttrId`] type with one combined `impl` surface, exactly as if
//! they had been written in one file.
//!
//! # `AttrId` identifies an authored name, not a semantic role
//!
//! This is the single invariant every predicate and every consumer of this
//! module must hold to: **one authored attribute name maps to exactly one
//! [`AttrId`] variant**, even when the same name is legitimately reused by
//! unrelated elements with unrelated value grammars. The clearest example
//! is `fill`: [`AttrId::Fill`] is both the presentation property that
//! paints a shape (`<rect fill="red">`) and the SMIL animation attribute
//! that chooses freeze-vs-remove behavior (`<animate fill="freeze">`).
//! These are unrelated grammars on unrelated element families that happen
//! to share four letters; splitting them into two variants would break
//! [`AttrId::from_name`]'s determinism (a lookup by name alone could not
//! know which variant to produce without already knowing the element it is
//! attached to). So there is one [`AttrId::Fill`], and
//! [`crate::names::content::attribute_applies`] reports it as applying to
//! *both* shape/text elements and animation elements — a caller that
//! additionally needs the paint-vs-timing distinction resolves it from the
//! host element, not from the attribute name.
//!
//! The same pattern recurs, at smaller scale, for [`AttrId::Type`] (an
//! enumerated keyword whose valid values differ per host element: `script`,
//! `style`, `feColorMatrix`, `feFuncR`/`feFuncG`/`feFuncB`/`feFuncA`,
//! `feTurbulence`, `animateTransform`, and SVG 2's `a`, all use a `type`
//! attribute with unrelated value sets), [`AttrId::Offset`] (`stop`'s
//! gradient-stop position vs. a `feFuncR`/`feFuncG`/`feFuncB`/`feFuncA`
//! transfer function's additive constant), [`AttrId::Rotate`] (per-glyph
//! rotation on `text`/`tspan` vs. path-orientation on `animateMotion`),
//! [`AttrId::Operator`] (`feComposite`'s Porter-Duff operator vs.
//! `feMorphology`'s erode/dilate choice), and [`AttrId::D`] (the `path`
//! geometry attribute that SVG 2 additionally promotes to a CSS property of
//! the same name). None of these get a second variant.
//!
//! # Sourcing
//!
//! Every variant here is grounded either in the specification requirements
//! catalog compiled for this crate (SVG 1.1 `attindex.html`/`propidx.html`
//! ∪ SVG 2 `attindex.html`/`changes.html` ∪ Filter Effects Level 1 ∪ the SVG
//! Animations module) or, for a small number of well-known SVG 1.1
//! attributes the catalog's own tables happen not to enumerate
//! (`baseProfile`, `version`, `zoomAndPan`, and the `textPath` quartet
//! `startOffset`/`method`/`spacing`/`side`), in this crate's mission brief,
//! which names them explicitly. Each such mission-sourced addition is
//! flagged on its variant's rustdoc. Two ARIA/custom-data attribute
//! families — `aria-*` and `data-*` — are intentionally **not** enumerated:
//! they are open-ended prefixes, not fixed names, and do not fit a closed
//! enum; callers needing them should match on the raw attribute local name
//! instead of going through [`AttrId`].

mod groups;
mod ids;

pub use ids::AttrId;
