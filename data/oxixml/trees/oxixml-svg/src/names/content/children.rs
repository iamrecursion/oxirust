//! [`allows_child`] and [`allows_character_data`]: the per-element content
//! models from specification requirements catalog §1's "Content model
//! class" column.

use crate::names::elements::ElementId;

/// Reports whether `child` may appear as a direct child of `parent`,
/// according to the SVG content models the specification requirements
/// catalog's §1 element inventory describes.
///
/// This is deliberately **not** a single generic "container" rule: the
/// catalog gives several distinct shapes of content model (a handful of
/// broad "general container" elements that accept almost anything; shapes,
/// filter primitives, and most leaves that accept only descriptive and
/// animation children; the `text`/`tspan`/`textPath` character-data chain;
/// gradients restricted to `stop` plus three specific animation elements;
/// `font`'s and the filter helper containers' (`feMerge`,
/// `feComponentTransfer`, lighting primitives) fixed, narrow child sets;
/// and `foreignObject`/`metadata`'s foreign-content models (permissive for
/// ordinary SVG content, but still respecting narrowly-scoped elements) —
/// each is encoded in its own match arm below, sourced from that element's
/// specific catalog row rather than one coarse category.
///
/// One shape is not sourced from a catalog cell: the timed animation
/// elements (`animate`, `set`, `animateColor`, `animateTransform`,
/// `discard`, `mpath`) accept descriptive children only, never a nested
/// animation element (`animateMotion` additionally accepts `mpath`; see its
/// own arm). Catalog §1.12 "Animation elements" has no "Content model
/// class" column at all, unlike every other §1 section, so this rule
/// instead follows the mission brief's explicit contrast between this
/// family and `use` ("animation elements allow descriptive only" vs. "`use`
/// accepts descriptive+animation") together with the fact that SVG's
/// animation module never adopted SMIL's `<par>`/`<seq>`/`<excl>` time
/// containers — nesting one timed animation element inside another has no
/// defined meaning to permit.
#[must_use]
pub fn allows_child(parent: ElementId, child: ElementId) -> bool {
    use ElementId as E;

    match parent {
        // ---- General container content model ----
        // `svg` ("Container: can hold structural/descriptive/shape/
        // gradient/animation/etc. children"), `g`/`defs`/`switch` (plain
        // "Container"), `symbol` ("Container"), `marker`/`pattern`/`mask`
        // ("Same as svg/symbol content model" / "Same as g/svg content
        // model"), and `a` (permissive simplification of "same content
        // model as its SVG parent context" — see the note on
        // `is_general_container_child`) all accept the same broad set:
        // everything except the elements the catalog scopes to one
        // specific, narrower parent (handled in their own arms below) and
        // every filter primitive (valid only inside `filter`/`feMerge`/
        // `feComponentTransfer`/a lighting primitive).
        E::Svg
        | E::G
        | E::Defs
        | E::Switch
        | E::Symbol
        | E::Marker
        | E::Pattern
        | E::Mask
        | E::A => is_general_container_child(child),

        // `foreignObject` ("Foreign-namespace content, typically HTML")
        // and `metadata` ("Foreign-namespace content, e.g. RDF") hold
        // arbitrary foreign XML, which `ElementId` (SVG-namespace only)
        // cannot represent directly; the meaningful question `ElementId`
        // *can* answer is whether an SVG element nested back inside that
        // foreign content (for example an `<svg>` nested inside HTML
        // inside `foreignObject`) would be valid there on its own terms —
        // so this reuses the general-container rule rather than answering
        // `true` unconditionally, which would incorrectly claim
        // `feMergeNode`/light sources/etc. become valid just by virtue of
        // being nested under foreign content.
        E::ForeignObject | E::Metadata => is_general_container_child(child),

        // `text`/`tspan`/`textPath` ("Character data +
        // tspan/tref/textPath/(1.1: altGlyph)/animation" per catalog §1.4,
        // with SVG 2's `a` also permitted inside text content per the
        // mission brief).
        E::Text | E::Tspan | E::TextPath => {
            matches!(child, E::Tspan | E::Tref | E::TextPath | E::A | E::AltGlyph)
                || child.is_descriptive()
                || child.is_animation()
        }

        // `clipPath` ("Shape/text/use children").
        E::ClipPath => {
            child.is_shape()
                || matches!(child, E::Text | E::Use)
                || child.is_descriptive()
                || child.is_animation()
        }

        // `linearGradient`/`radialGradient` ("stop/animation/description
        // children"). Only three of the seven animation elements make
        // sense on a paint-server template — `animateMotion` (nothing to
        // move along a path), `mpath`, and `discard` are excluded.
        E::LinearGradient | E::RadialGradient => {
            matches!(child, E::Stop | E::Animate | E::Set | E::AnimateTransform)
                || child.is_descriptive()
        }

        // `font` ("Container: font-face, missing-glyph, glyph, hkern,
        // vkern, animation").
        E::Font => {
            matches!(
                child,
                E::FontFace | E::MissingGlyph | E::Glyph | E::Hkern | E::Vkern
            ) || child.is_descriptive()
                || child.is_animation()
        }
        // `font-face` ("font-face-src children + descriptive").
        E::FontFace => matches!(child, E::FontFaceSrc) || child.is_descriptive(),
        // `font-face-src` ("font-face-uri/font-face-name children").
        E::FontFaceSrc => matches!(child, E::FontFaceUri | E::FontFaceName),
        // `font-face-uri` ("font-face-format children").
        E::FontFaceUri => matches!(child, E::FontFaceFormat),

        // `altGlyphDef` ("glyphRef/altGlyphItem children").
        E::AltGlyphDef => matches!(child, E::GlyphRef | E::AltGlyphItem),
        // `altGlyphItem` ("glyphRef children").
        E::AltGlyphItem => matches!(child, E::GlyphRef),

        // `use` — the union of SVG 1.1's "descriptive/animation children"
        // baseline and the additional `clipPath`/`mask`/`script`/`style`
        // SVG 2 permits, so both authoring eras round-trip.
        E::Use => {
            child.is_descriptive()
                || child.is_animation()
                || matches!(child, E::ClipPath | E::Mask | E::Script | E::Style)
        }

        // Draft mesh-gradient / hatch paint-server families: parse-and-
        // preserve child chains only, per catalog §1.7/§1.8.
        E::Meshgradient => matches!(child, E::Meshrow) || child.is_descriptive(),
        E::Meshrow => matches!(child, E::Meshpatch),
        // `meshpatch` ("stop-like meshpatch path-side children") — no
        // named child element exists in this catalog pass for that
        // "stop-like" content; permissive to descriptive only.
        E::Meshpatch => child.is_descriptive(),
        E::Hatch => matches!(child, E::Hatchpath) || child.is_descriptive(),

        // `filter` — its primitive chain (the "top-level" filter
        // primitives only: not the `feFunc*`/`feMergeNode`/light-source
        // helpers, which are scoped to their own specific parent) plus
        // descriptive/animation children.
        E::Filter => {
            is_top_level_filter_primitive(child) || child.is_descriptive() || child.is_animation()
        }

        // `feMerge` ("container for feMergeNode").
        E::FeMerge => {
            matches!(child, E::FeMergeNode) || child.is_descriptive() || child.is_animation()
        }
        // `feComponentTransfer` ("container for feFunc*").
        E::FeComponentTransfer => {
            matches!(child, E::FeFuncR | E::FeFuncG | E::FeFuncB | E::FeFuncA)
                || child.is_descriptive()
                || child.is_animation()
        }
        // `feDiffuseLighting`/`feSpecularLighting` accept exactly one
        // light-source child (`feDistantLight`/`fePointLight`/
        // `feSpotLight`), per the catalog's light-source-elements note.
        E::FeDiffuseLighting | E::FeSpecularLighting => {
            matches!(child, E::FeDistantLight | E::FePointLight | E::FeSpotLight)
                || child.is_descriptive()
                || child.is_animation()
        }

        // `animateMotion` additionally accepts an `mpath` child
        // referencing external path geometry.
        E::AnimateMotion => matches!(child, E::Mpath) || child.is_descriptive(),

        // The remaining animation-timing elements (`animate`, `set`,
        // `animateColor`, `animateTransform`, `discard`) plus `mpath`:
        // descriptive children only, never a nested animation element.
        // Catalog §1.12 has no "Content model class" column at all (unlike
        // every other section-1 table), so this is not read off a catalog
        // cell the way the rest of this function's arms are; it instead
        // follows the mission brief's explicit contrast ("animation
        // elements allow descriptive only" vs. "`use` accepts
        // descriptive+animation") and the fact that SVG's animation module
        // never adopted SMIL's `<par>`/`<seq>`/`<excl>` time containers, so
        // one timed animation element nested inside another has no defined
        // meaning to permit. `animateMotion` already gets its own arm above
        // (`mpath`, no `is_animation()`) — this arm makes the other five
        // animation-family elements consistent with it instead of falling
        // through to the general "empty content model" default below, which
        // would silently re-admit `is_animation()` children.
        E::Animate | E::Set | E::AnimateColor | E::AnimateTransform | E::Discard | E::Mpath => {
            child.is_descriptive()
        }

        // `title`/`desc` ("Text content only") and `script`/`style`
        // ("Character data") hold text, not child elements — not even the
        // otherwise-near-universal descriptive/animation children.
        E::Title | E::Desc | E::Script | E::Style => false,

        // Every other element (the 7 basic shapes, `image`, `view`, most
        // filter primitives, the SVG-fonts leaves, `stop`, `color-profile`,
        // `cursor`, ...): the catalog's "Empty" content model class still
        // permits descriptive and animation children almost everywhere, so
        // that is the default rather than refusing every child. The
        // animation-timing elements themselves are handled above, not here.
        _ => child.is_descriptive() || child.is_animation(),
    }
}

/// Reports whether `element`'s content model permits character data
/// (text nodes) as direct content, per catalog §1's per-element content
/// model column.
///
/// True for the text content elements (`text`, `tspan`, `textPath`,
/// `altGlyph`), the two textual descriptive elements (`desc`, `title`),
/// the two character-data-only scripting/styling elements (`style`,
/// `script`), and — permissively, since their content is foreign XML that
/// may itself mix text and elements — `metadata` and `foreignObject`.
/// `tref`'s content model is `Empty`: it *renders* a copy of another
/// element's character data via `xlink:href`, but does not itself hold any.
#[must_use]
pub const fn allows_character_data(element: ElementId) -> bool {
    matches!(
        element,
        ElementId::Text
            | ElementId::Tspan
            | ElementId::TextPath
            | ElementId::AltGlyph
            | ElementId::Desc
            | ElementId::Title
            | ElementId::Style
            | ElementId::Script
            | ElementId::Metadata
            | ElementId::ForeignObject
    )
}

/// True for elements that may appear as a child wherever the "general
/// container" content model applies (`svg`, `g`, `defs`, `switch`,
/// `symbol`, `marker`, `pattern`, `mask`, `a`) — every element **except**
/// the ones the catalog scopes to one specific, narrower parent elsewhere
/// in [`allows_child`], and every filter primitive (valid only inside
/// `filter`/`feMerge`/`feComponentTransfer`/a lighting primitive, never as
/// a bare child of a general container).
fn is_general_container_child(child: ElementId) -> bool {
    use ElementId as E;

    !matches!(
        child,
        // Gradient/paint-server/font children scoped to one parent.
        E::Stop
            | E::Meshrow
            | E::Meshpatch
            | E::Hatchpath
            | E::FontFaceSrc
            | E::FontFaceUri
            | E::FontFaceFormat
            | E::FontFaceName
            | E::FontFace
            | E::MissingGlyph
            | E::Glyph
            | E::Hkern
            | E::Vkern
            // Glyph-substitution helpers scoped to altGlyphDef/altGlyphItem.
            | E::GlyphRef
            | E::AltGlyphItem
            // animateMotion's path reference.
            | E::Mpath
            // Text-content-chain-only elements (bare `text` itself is a
            // valid general-container child; these nested forms are not).
            | E::Tspan
            | E::TextPath
            | E::Tref
            | E::AltGlyph
    ) && !child.is_filter_primitive()
}

/// True for the filter primitives that may appear directly inside a
/// `filter` element's primitive chain: the 25 `fe*` elements minus the
/// four `feFunc*` transfer-function helpers (valid only inside
/// `feComponentTransfer`), `feMergeNode` (valid only inside `feMerge`),
/// and the three light-source elements (valid only inside
/// `feDiffuseLighting`/`feSpecularLighting`) — 17 elements.
fn is_top_level_filter_primitive(id: ElementId) -> bool {
    use ElementId as E;

    id.is_filter_primitive()
        && !matches!(
            id,
            E::FeFuncR
                | E::FeFuncG
                | E::FeFuncB
                | E::FeFuncA
                | E::FeMergeNode
                | E::FeDistantLight
                | E::FePointLight
                | E::FeSpotLight
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::names::elements::ElementId as E;

    #[test]
    fn svg_allows_rect_but_rect_does_not_allow_rect() {
        assert!(allows_child(E::Svg, E::Rect));
        assert!(!allows_child(E::Rect, E::Rect));
    }

    #[test]
    fn fe_merge_allows_fe_merge_node_and_nowhere_else() {
        assert!(allows_child(E::FeMerge, E::FeMergeNode));
        for &parent in E::ALL {
            if parent == E::FeMerge {
                continue;
            }
            assert!(
                !allows_child(parent, E::FeMergeNode),
                "{parent:?} must not allow feMergeNode"
            );
        }
    }

    #[test]
    fn filter_does_not_allow_fe_merge_node_or_light_sources_directly() {
        assert!(!allows_child(E::Filter, E::FeMergeNode));
        assert!(!allows_child(E::Filter, E::FeDistantLight));
        assert!(!allows_child(E::Filter, E::FePointLight));
        assert!(!allows_child(E::Filter, E::FeSpotLight));
        // But it does allow the 17 top-level primitives.
        assert!(allows_child(E::Filter, E::FeGaussianBlur));
        assert!(allows_child(E::Filter, E::FeMerge));
        assert!(allows_child(E::Filter, E::FeComponentTransfer));
    }

    #[test]
    fn lighting_primitives_allow_exactly_their_light_sources() {
        assert!(allows_child(E::FeDiffuseLighting, E::FeDistantLight));
        assert!(allows_child(E::FeDiffuseLighting, E::FePointLight));
        assert!(allows_child(E::FeDiffuseLighting, E::FeSpotLight));
        assert!(allows_child(E::FeSpecularLighting, E::FeSpotLight));
        assert!(!allows_child(E::FeGaussianBlur, E::FeDistantLight));
    }

    #[test]
    fn linear_gradient_allows_stop_and_a_narrow_animation_set() {
        assert!(allows_child(E::LinearGradient, E::Stop));
        assert!(allows_child(E::LinearGradient, E::Animate));
        assert!(allows_child(E::LinearGradient, E::Set));
        assert!(allows_child(E::LinearGradient, E::AnimateTransform));
        assert!(!allows_child(E::LinearGradient, E::AnimateMotion));
        assert!(!allows_child(E::LinearGradient, E::AnimateColor));
        assert!(!allows_child(E::LinearGradient, E::Mpath));
        assert!(allows_child(E::LinearGradient, E::Desc));
    }

    #[test]
    fn text_allows_tspan_but_path_does_not() {
        assert!(allows_child(E::Text, E::Tspan));
        assert!(allows_child(E::Text, E::TextPath));
        assert!(allows_child(E::Text, E::Tref));
        assert!(allows_child(E::Text, E::A));
        assert!(!allows_child(E::Path, E::Tspan));
    }

    #[test]
    fn general_containers_reject_narrowly_scoped_children() {
        for &parent in &[E::Svg, E::G, E::Defs, E::Switch, E::Symbol, E::A] {
            assert!(
                !allows_child(parent, E::Stop),
                "{parent:?} must not allow stop"
            );
            assert!(
                !allows_child(parent, E::FeFuncR),
                "{parent:?} must not allow feFuncR"
            );
            assert!(
                !allows_child(parent, E::Mpath),
                "{parent:?} must not allow mpath"
            );
            assert!(
                !allows_child(parent, E::Tspan),
                "{parent:?} must not allow bare tspan"
            );
            assert!(allows_child(parent, E::Text), "{parent:?} must allow text");
            assert!(allows_child(parent, E::G), "{parent:?} must allow g");
        }
    }

    #[test]
    fn animate_motion_allows_mpath_only_as_its_special_child() {
        assert!(allows_child(E::AnimateMotion, E::Mpath));
        assert!(!allows_child(E::AnimateMotion, E::Animate));
        assert!(allows_child(E::AnimateMotion, E::Desc));
    }

    #[test]
    fn timed_animation_elements_allow_descriptive_children_only() {
        // Mission-brief contrast: "animation elements allow descriptive
        // only" vs. "`use` accepts descriptive+animation" (catalog §1.12
        // itself has no content-model column to read this from). No timed
        // animation element may nest another one: SVG's animation module
        // never adopted SMIL's `<par>`/`<seq>`/`<excl>` time containers, so
        // nesting has no defined meaning.
        for &parent in &[
            E::Animate,
            E::Set,
            E::AnimateColor,
            E::AnimateTransform,
            E::Discard,
            E::Mpath,
        ] {
            assert!(
                allows_child(parent, E::Desc),
                "{parent:?} should allow desc"
            );
            assert!(
                allows_child(parent, E::Title),
                "{parent:?} should allow title"
            );
            for &animation_child in &[
                E::Animate,
                E::Set,
                E::AnimateColor,
                E::AnimateMotion,
                E::AnimateTransform,
                E::Mpath,
                E::Discard,
            ] {
                assert!(
                    !allows_child(parent, animation_child),
                    "{parent:?} must not allow nested {animation_child:?}"
                );
            }
        }
        // `animateMotion` keeps its own, separately-tested exception
        // (`mpath`), which is why it is not in the `parent` list above.
        assert!(allows_child(E::AnimateMotion, E::Mpath));
    }

    #[test]
    fn font_family_content_models_are_fixed() {
        assert!(allows_child(E::Font, E::FontFace));
        assert!(allows_child(E::Font, E::Glyph));
        assert!(allows_child(E::Font, E::MissingGlyph));
        assert!(allows_child(E::Font, E::Hkern));
        assert!(allows_child(E::Font, E::Vkern));
        assert!(!allows_child(E::Font, E::Rect));
        assert!(allows_child(E::FontFace, E::FontFaceSrc));
        assert!(allows_child(E::FontFaceSrc, E::FontFaceUri));
        assert!(allows_child(E::FontFaceSrc, E::FontFaceName));
        assert!(allows_child(E::FontFaceUri, E::FontFaceFormat));
        assert!(!allows_child(E::FontFace, E::FontFaceFormat));
    }

    #[test]
    fn alt_glyph_definitions_are_fixed() {
        assert!(allows_child(E::AltGlyphDef, E::GlyphRef));
        assert!(allows_child(E::AltGlyphDef, E::AltGlyphItem));
        assert!(allows_child(E::AltGlyphItem, E::GlyphRef));
        assert!(!allows_child(E::AltGlyphItem, E::AltGlyphItem));
    }

    #[test]
    fn use_accepts_the_svg1_1_and_svg2_union() {
        assert!(allows_child(E::Use, E::Desc));
        assert!(allows_child(E::Use, E::Animate));
        assert!(allows_child(E::Use, E::ClipPath));
        assert!(allows_child(E::Use, E::Mask));
        assert!(allows_child(E::Use, E::Script));
        assert!(allows_child(E::Use, E::Style));
        assert!(!allows_child(E::Use, E::Rect));
    }

    #[test]
    fn clip_path_accepts_shapes_text_and_use() {
        assert!(allows_child(E::ClipPath, E::Rect));
        assert!(allows_child(E::ClipPath, E::Text));
        assert!(allows_child(E::ClipPath, E::Use));
        assert!(!allows_child(E::ClipPath, E::Image));
    }

    #[test]
    fn foreign_content_hosts_are_fully_permissive() {
        assert!(allows_child(E::ForeignObject, E::Rect));
        assert!(allows_child(E::ForeignObject, E::Svg));
        assert!(allows_child(E::Metadata, E::Rect));
    }

    #[test]
    fn character_data_only_leaves_allow_no_element_children() {
        assert!(!allows_child(E::Title, E::Desc));
        assert!(!allows_child(E::Desc, E::Title));
        assert!(!allows_child(E::Script, E::Desc));
        assert!(!allows_child(E::Style, E::Desc));
    }

    #[test]
    fn shapes_and_leaves_default_to_descriptive_and_animation_only() {
        for &leaf in &[
            E::Rect,
            E::Circle,
            E::Ellipse,
            E::Line,
            E::Polyline,
            E::Polygon,
            E::Path,
            E::Image,
            E::View,
            E::Stop,
            E::FeGaussianBlur,
            E::FeFlood,
            E::ColorProfile,
            E::Cursor,
        ] {
            assert!(allows_child(leaf, E::Desc), "{leaf:?} should allow desc");
            assert!(allows_child(leaf, E::Title), "{leaf:?} should allow title");
            assert!(
                allows_child(leaf, E::Animate),
                "{leaf:?} should allow animate"
            );
            assert!(
                !allows_child(leaf, E::Rect),
                "{leaf:?} should not allow rect"
            );
        }
    }

    #[test]
    fn allows_character_data_matches_the_catalog() {
        for id in [
            E::Text,
            E::Tspan,
            E::TextPath,
            E::AltGlyph,
            E::Desc,
            E::Title,
            E::Style,
            E::Script,
            E::Metadata,
            E::ForeignObject,
        ] {
            assert!(
                allows_character_data(id),
                "{id:?} should allow character data"
            );
        }
        assert!(
            !allows_character_data(E::Tref),
            "tref's content model is Empty: it renders another element's data, it does not hold its own"
        );
        assert!(!allows_character_data(E::Rect));
        assert!(!allows_character_data(E::G));
    }

    #[test]
    fn every_element_has_a_defined_child_answer_against_itself() {
        // Sanity: allows_child must not panic for any (parent, child) pair
        // in the full 88x88 space.
        for &parent in ElementId::ALL {
            for &child in ElementId::ALL {
                let _ = allows_child(parent, child);
            }
        }
    }

    /// Pins `is_general_container_child`'s exact true-set: the 88 elements
    /// minus the 20 named exclusions (elements scoped to one specific,
    /// narrower parent elsewhere in [`allows_child`]) minus the 25 filter
    /// primitives (none of which overlap the 20), leaving 43. A future
    /// edit to either exclusion list will fail this loudly instead of only
    /// showing up as a silently-wrong `true`/`false` on some untested pair
    /// (as the `Metadata`-allows-`feMergeNode` regression this test
    /// replaces did, before it was caught by
    /// `fe_merge_allows_fe_merge_node_and_nowhere_else`).
    #[test]
    fn general_container_child_membership_is_exactly_43_named_elements() {
        let included: std::collections::HashSet<E> = E::ALL
            .iter()
            .copied()
            .filter(|c| is_general_container_child(*c))
            .collect();
        assert_eq!(included.len(), 43);
        let expected = [
            E::Svg,
            E::G,
            E::Defs,
            E::Symbol,
            E::Use,
            E::Switch,
            E::Desc,
            E::Title,
            E::Metadata,
            E::Path,
            E::Rect,
            E::Circle,
            E::Ellipse,
            E::Line,
            E::Polyline,
            E::Polygon,
            E::Text,
            E::AltGlyphDef,
            E::Font,
            E::Image,
            E::ForeignObject,
            E::A,
            E::View,
            E::LinearGradient,
            E::RadialGradient,
            E::Meshgradient,
            E::Solidcolor,
            E::Pattern,
            E::Hatch,
            E::ClipPath,
            E::Mask,
            E::Marker,
            E::ColorProfile,
            E::Cursor,
            E::Filter,
            E::Animate,
            E::Set,
            E::AnimateMotion,
            E::AnimateTransform,
            E::AnimateColor,
            E::Discard,
            E::Script,
            E::Style,
        ];
        assert_eq!(
            expected.len(),
            43,
            "the expected list itself must have 43 entries"
        );
        for id in expected {
            assert!(
                included.contains(&id),
                "{id:?} should be a general-container child"
            );
        }
    }

    #[test]
    fn general_containers_permit_the_two_self_nesting_cases() {
        // Nested `<svg>` (a fresh viewport inside a viewport) is
        // well-established, valid SVG.
        assert!(allows_child(E::Svg, E::Svg));
        // `symbol` shares `svg`'s content model per the catalog's own
        // "Same as svg/symbol content model" phrasing (used to describe
        // `marker`/`pattern`); nesting one `symbol` inside another is not
        // idiomatic (`use` is the normal reuse mechanism) but is not
        // excluded by any catalog text either, so the permissive answer
        // this crate takes on unconfirmed cases applies here too.
        assert!(allows_child(E::Symbol, E::Symbol));
    }
}
