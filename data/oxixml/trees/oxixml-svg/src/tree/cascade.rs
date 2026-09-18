//! Computed styles: the SVG/CSS cascade (specification requirements
//! catalog §6.4) and property inheritance.
//!
//! # A documented simplification: `!important` tiering
//!
//! The full CSS cascade (catalog §6.4) ranks, from highest to lowest
//! precedence: author stylesheet rules marked `!important`, `style=""`
//! declarations marked `!important`, `style=""` normal declarations,
//! author stylesheet rules at normal priority (by specificity + source
//! order), presentation attributes, inheritance, and the initial value.
//!
//! This resolver collapses the first four tiers into a **single-author-
//! origin** model, per this crate's mission brief: every `style=""`
//! declaration — `!important` or not — outranks every stylesheet rule,
//! `!important` or not. Within the stylesheet tier itself, `!important`
//! still outranks normal declarations, and among rules of equal
//! importance, higher [`Specificity`](crate::style::css::Specificity)
//! wins, then later source order. This is a **documented divergence**
//! from the catalog's full cascade, not an oversight: modeling two
//! separate "author stylesheet" vs. "author inline" origins buys little
//! for a foundation crate that performs no rendering, and hand-authored
//! (or even generated) SVG only rarely relies on the
//! stylesheet-`!important`-beats-inline-style edge case this collapses.

use crate::error::ValueError;
use crate::names::elements::ElementId;
use crate::style::attr::Declaration;
use crate::style::css::{ElementFacts, StyleRule, Stylesheet};
use crate::style::property::PropertyId;
use crate::tree::document::SvgDocument;
use crate::tree::typed::SvgElement;
use crate::values::lex::is_wsp;
use crate::values::paint::{self, Paint};

/// Which cascade tier a [`StyleResolver::specified_value`] result came
/// from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StyleOrigin {
    /// The element's own `style=""` attribute.
    InlineStyle,
    /// A matched rule from an aggregated `<style>` stylesheet.
    Stylesheet,
    /// A presentation attribute (an XML attribute with the same name as
    /// the property).
    PresentationAttribute,
}

/// Resolves computed style values for the elements of one [`SvgDocument`],
/// applying the cascade the [module documentation](self) describes.
///
/// Every `{http://www.w3.org/2000/svg}style` element's text content, in
/// document order, is parsed once at construction time into one
/// concatenated, source-ordered rule list. Per-element resolution is done
/// lazily on every call and is not itself cached. Matching *one* element
/// against the stylesheet ([`Self::specified_value`], via the private
/// `best_stylesheet_match`) is `O(depth)` — it rebuilds that element's own
/// ancestor chain to test selectors against — and
/// [`Self::computed_value`] does that once per ancestor it has to walk up
/// through for an inherited property nothing along the way specifies, so
/// the worst case for a single call is `O(depth²)`, not `O(depth)`. The
/// catalog's own recommendation ("cache per-call is fine") was written
/// against the linear cost a naive reading suggests; the quadratic factor
/// is a real, uncached cost of this implementation, not something the
/// catalog anticipated. In practice this stays cheap: the depths
/// [`SvgOptions`](crate::tree::document::SvgOptions)'s default parse-time
/// ceiling allows keep even the worst case well under a millisecond. It
/// only becomes noticeable for a document assembled deeper than that
/// ceiling — necessarily through some path other than the default parse
/// (an unvalidated
/// [`SvgDocument::from_document`](crate::tree::document::SvgDocument::from_document),
/// or a caller-raised
/// [`SvgOptions::with_max_element_depth`](crate::tree::document::SvgOptions::with_max_element_depth)) —
/// where a caller doing that should budget for `O(depth²)`, not `O(depth)`.
#[derive(Debug, Clone)]
pub struct StyleResolver<'a> {
    document: &'a SvgDocument,
    rules: Vec<StyleRule>,
}

impl<'a> StyleResolver<'a> {
    /// Collects every `style` element's text content, in document order,
    /// each parsed with [`Stylesheet::parse`], into one rule list.
    #[must_use]
    pub fn new(document: &'a SvgDocument) -> Self {
        let mut rules = Vec::new();
        for element in document.svg_element().descendants() {
            if element.kind() == Some(ElementId::Style) {
                if let Some(text) = element.text_content() {
                    rules.extend(Stylesheet::parse(&text).rules);
                }
            }
        }
        Self { document, rules }
    }

    /// The document this resolver was built from.
    #[must_use]
    pub const fn document(&self) -> &'a SvgDocument {
        self.document
    }

    /// The specified value of `property` on `element`: the value the
    /// cascade selects *before* resolving a CSS-wide keyword or falling
    /// back to inheritance/the initial value, together with which tier it
    /// came from.
    ///
    /// `None` when nothing at all specifies `property` on `element` —
    /// [`Self::computed_value`] then falls back to inheritance or the
    /// initial value, exactly as it would for an explicit `unset`.
    #[must_use]
    pub fn specified_value(
        &self,
        element: SvgElement<'a>,
        property: PropertyId,
    ) -> Option<(String, StyleOrigin)> {
        let declarations = element.style_declarations();
        if let Some(declaration) = find_declaration(&declarations, property.as_str()) {
            return Some((declaration.value.to_string(), StyleOrigin::InlineStyle));
        }

        if let Some(value) = self.best_stylesheet_match(element, property) {
            return Some((value, StyleOrigin::Stylesheet));
        }

        if let Some(raw) = element.presentation_attr(property) {
            return Some((raw.to_string(), StyleOrigin::PresentationAttribute));
        }

        None
    }

    /// The best-matching stylesheet declaration's value for `property` on
    /// `element`: `!important` rules beat normal ones regardless of
    /// specificity; within one importance tier, higher specificity wins,
    /// and a later source position breaks a specificity tie.
    fn best_stylesheet_match(
        &self,
        element: SvgElement<'a>,
        property: PropertyId,
    ) -> Option<String> {
        let chain = ancestor_chain_root_first(element);
        let names: Vec<&str> = chain.iter().map(|node| node.local_name()).collect();
        let ids: Vec<Option<&str>> = chain.iter().map(|node| node.id()).collect();
        let classes: Vec<Vec<&str>> = chain
            .iter()
            .map(|node| node.class_list().collect())
            .collect();
        let facts: Vec<ElementFacts<'_>> = (0..chain.len())
            .map(|index| ElementFacts {
                name: names[index],
                id: ids[index],
                classes: classes[index].as_slice(),
            })
            .collect();

        self.rules
            .iter()
            .enumerate()
            .filter(|(_, rule)| rule.selector.matches_chain(&facts))
            .filter_map(|(source_order, rule)| {
                find_declaration(&rule.declarations, property.as_str()).map(|declaration| {
                    (
                        declaration,
                        (
                            declaration.important,
                            rule.selector.specificity(),
                            source_order,
                        ),
                    )
                })
            })
            .max_by_key(|(_, key)| *key)
            .map(|(declaration, _)| declaration.value.to_string())
    }
}

impl<'a> StyleResolver<'a> {
    /// The computed value of `property` on `element`: this always resolves
    /// to a concrete string, applying the cascade in the [module
    /// documentation](self) — `style=""` beats a matched stylesheet rule
    /// beats a presentation attribute beats inheritance beats the
    /// property's initial value.
    ///
    /// The CSS-wide keywords `inherit`/`initial`/`unset` are recognized,
    /// case-insensitively, wherever they appear as the specified value:
    /// `inherit` **unconditionally** takes the parent's computed value —
    /// even for a property that is not normally inherited, since that is
    /// the entire point of writing `inherit` explicitly — falling back to
    /// the initial value only at the document root (no parent to inherit
    /// from). `initial` forces [`PropertyId::initial_value`]. `unset` acts
    /// like `inherit` for a property that *is* normally inherited, and like
    /// `initial` otherwise — the same rule "nothing specified at all"
    /// falls back on. Every other value is returned exactly as specified;
    /// this resolver does not itself parse or validate it, beyond the three
    /// keywords above — callers do the rest with [`crate::values`] parsers
    /// (see [`Self::fill`]/[`Self::stroke`]/[`Self::opacity`] for the
    /// short list this type parses itself).
    ///
    /// Walks from `element` up toward the document root with an explicit
    /// loop, moving to the parent whenever the specified value defers to it
    /// (an explicit `inherit`, an `unset` on an inherited property, or
    /// nothing specified at all for an inherited property), rather than by
    /// calling itself once per ancestor: this keeps stack usage at `O(1)`
    /// regardless of how many ancestors it has to walk through. A document
    /// deep enough for that to matter is reachable (see [`Self`]'s own
    /// documentation on this method's `O(depth²)` time cost, which this
    /// walk does not change), so unlike an ordinary bounded-depth tree,
    /// native call-stack recursion here would have grown one frame per
    /// ancestor with no ceiling of its own.
    #[must_use]
    pub fn computed_value(&self, element: SvgElement<'a>, property: PropertyId) -> String {
        let mut current = element;
        loop {
            let defers_to_parent = match self.specified_value(current, property) {
                Some((value, _origin)) => {
                    let trimmed = value.trim_matches(is_wsp);
                    if trimmed.eq_ignore_ascii_case("inherit") {
                        true
                    } else if trimmed.eq_ignore_ascii_case("initial") {
                        return property.initial_value().to_string();
                    } else if trimmed.eq_ignore_ascii_case("unset") {
                        property.inherited()
                    } else {
                        return value;
                    }
                }
                None => property.inherited(),
            };
            if !defers_to_parent {
                return property.initial_value().to_string();
            }
            match current.parent_element() {
                Some(parent) => current = parent,
                None => return property.initial_value().to_string(),
            }
        }
    }

    /// The computed `fill` paint.
    ///
    /// # Errors
    ///
    /// Returns [`ValueError`] if the computed value is not a valid
    /// `<paint>` (possible for invalid author content, or a raw CSS-wide
    /// keyword this resolver does not itself special-case, such as
    /// `revert`).
    pub fn fill(&self, element: SvgElement<'a>) -> Result<Paint, ValueError> {
        Paint::parse(&self.computed_value(element, PropertyId::Fill))
    }

    /// The computed `stroke` paint.
    ///
    /// # Errors
    ///
    /// See [`Self::fill`].
    pub fn stroke(&self, element: SvgElement<'a>) -> Result<Paint, ValueError> {
        Paint::parse(&self.computed_value(element, PropertyId::Stroke))
    }

    /// The computed `opacity`: a `<number>`/`<percentage>` clamped to
    /// `0.0..=1.0`, falling back to the initial value `1.0` when the
    /// computed value fails to parse as one.
    #[must_use]
    pub fn opacity(&self, element: SvgElement<'a>) -> f64 {
        paint::parse_opacity(&self.computed_value(element, PropertyId::Opacity)).unwrap_or(1.0)
    }

    /// Whether the computed `display` value is `none` — the element and
    /// its entire subtree do not render at all.
    #[must_use]
    pub fn display_none(&self, element: SvgElement<'a>) -> bool {
        self.computed_value(element, PropertyId::Display)
            .trim_matches(is_wsp)
            .eq_ignore_ascii_case("none")
    }

    /// Whether the computed `visibility` value is anything other than
    /// `visible` (that is, `hidden` or `collapse`). SVG has no table-row
    /// context where `collapse`'s behavior would differ from `hidden`'s,
    /// so this crate treats both as "not painted".
    #[must_use]
    pub fn visibility_hidden(&self, element: SvgElement<'a>) -> bool {
        !self
            .computed_value(element, PropertyId::Visibility)
            .trim_matches(is_wsp)
            .eq_ignore_ascii_case("visible")
    }
}

/// Finds the effective declaration for `property_name` within one
/// declaration list: prefers an `!important` declaration over a normal
/// one, and the *last* matching declaration within whichever importance
/// tier applies — matching ordinary CSS "a later declaration of the same
/// property overrides an earlier one" behavior within a single
/// declaration block.
fn find_declaration<'d>(
    declarations: &'d [Declaration],
    property_name: &str,
) -> Option<&'d Declaration> {
    let matching: Vec<&Declaration> = declarations
        .iter()
        .filter(|declaration| declaration.name.as_ref() == property_name)
        .collect();
    matching
        .iter()
        .rev()
        .find(|declaration| declaration.important)
        .copied()
        .or_else(|| matching.last().copied())
}

/// The chain of `element` and its SVG-namespace ancestors, root-first
/// (ending with `element` itself as the last, subject entry) — the order
/// [`crate::style::css::Selector::matches_chain`] expects.
fn ancestor_chain_root_first(element: SvgElement<'_>) -> Vec<SvgElement<'_>> {
    let mut chain = Vec::new();
    let mut current = Some(element);
    while let Some(next) = current {
        chain.push(next);
        current = next.parent_element();
    }
    chain.reverse();
    chain
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ns::SVG_NAMESPACE;
    use oxixml_dom::Document;

    const NS: &str = r##"xmlns="http://www.w3.org/2000/svg""##;

    fn doc(xml: &str) -> SvgDocument {
        SvgDocument::parse_str(xml).expect("fixture is a valid SVG document")
    }

    fn by_id<'d>(document: &'d SvgDocument, id: &str) -> SvgElement<'d> {
        let node = document
            .document()
            .traverse(document.root())
            .find(|&n| document.document().get_attribute(n, "id") == Some(id))
            .unwrap_or_else(|| panic!("no element with id {id:?}"));
        document
            .element(node)
            .unwrap_or_else(|| panic!("{id:?} is not an SVG element"))
    }

    #[test]
    fn precedence_chain_style_beats_stylesheet_beats_presentation_beats_inherited_beats_initial() {
        let d = doc(&format!(
            r##"<svg {NS}>
                  <style>.styled {{ fill: from-stylesheet; }}</style>
                  <g id="parent" fill="from-parent-presentation">
                    <rect id="all-four" class="styled" fill="from-presentation"
                          style="fill: from-style"/>
                    <rect id="no-style" class="styled" fill="from-presentation"/>
                    <rect id="no-style-no-sheet" fill="from-presentation"/>
                    <rect id="nothing-at-all"/>
                  </g>
                  <rect id="top-level-nothing"/>
                </svg>"##
        ));
        let resolver = StyleResolver::new(&d);

        assert_eq!(
            resolver.computed_value(by_id(&d, "all-four"), PropertyId::Fill),
            "from-style"
        );
        assert_eq!(
            resolver.computed_value(by_id(&d, "no-style"), PropertyId::Fill),
            "from-stylesheet"
        );
        assert_eq!(
            resolver.computed_value(by_id(&d, "no-style-no-sheet"), PropertyId::Fill),
            "from-presentation"
        );
        assert_eq!(
            resolver.computed_value(by_id(&d, "nothing-at-all"), PropertyId::Fill),
            "from-parent-presentation",
            "inherits the parent's computed value"
        );
        assert_eq!(
            resolver.computed_value(by_id(&d, "top-level-nothing"), PropertyId::Fill),
            "black",
            "no ancestor sets fill anywhere, falls back to the initial value"
        );
    }

    #[test]
    fn a_non_inherited_property_with_nothing_specified_uses_the_initial_value_directly() {
        let d = doc(&format!(
            r##"<svg {NS}><g opacity="0.2"><rect id="child"/></g></svg>"##
        ));
        let resolver = StyleResolver::new(&d);
        // opacity is not inherited, so the parent's 0.2 must not leak down.
        assert_eq!(
            resolver.computed_value(by_id(&d, "child"), PropertyId::Opacity),
            "1"
        );
    }

    #[test]
    fn specified_value_reports_the_correct_origin_or_none() {
        let d = doc(&format!(
            r##"<svg {NS}>
                  <style>#target {{ fill: from-stylesheet; }}</style>
                  <rect id="target" fill="from-presentation"/>
                  <rect id="presentation-only" fill="p"/>
                  <rect id="nothing"/>
                </svg>"##
        ));
        let resolver = StyleResolver::new(&d);
        assert_eq!(
            resolver.specified_value(by_id(&d, "target"), PropertyId::Fill),
            Some(("from-stylesheet".to_string(), StyleOrigin::Stylesheet))
        );
        assert_eq!(
            resolver.specified_value(by_id(&d, "presentation-only"), PropertyId::Fill),
            Some(("p".to_string(), StyleOrigin::PresentationAttribute))
        );
        assert_eq!(
            resolver.specified_value(by_id(&d, "nothing"), PropertyId::Fill),
            None
        );
    }

    #[test]
    fn id_selector_specificity_beats_class_selector_specificity() {
        let d = doc(&format!(
            r##"<svg {NS}>
                  <style>.a {{ fill: from-class; }} #target {{ fill: from-id; }}</style>
                  <rect id="target" class="a"/>
                </svg>"##
        ));
        let resolver = StyleResolver::new(&d);
        assert_eq!(
            resolver.computed_value(by_id(&d, "target"), PropertyId::Fill),
            "from-id"
        );
    }

    #[test]
    fn important_beats_a_higher_specificity_normal_rule() {
        let d = doc(&format!(
            r##"<svg {NS}>
                  <style>
                    #target {{ fill: from-id-normal; }}
                    .imp {{ fill: from-class-important !important; }}
                  </style>
                  <rect id="target" class="imp"/>
                </svg>"##
        ));
        let resolver = StyleResolver::new(&d);
        assert_eq!(
            resolver.computed_value(by_id(&d, "target"), PropertyId::Fill),
            "from-class-important"
        );
    }

    #[test]
    fn later_source_order_breaks_a_specificity_tie() {
        let d = doc(&format!(
            r##"<svg {NS}>
                  <style>.tie {{ fill: first; }} .tie {{ fill: second; }}</style>
                  <rect id="target" class="tie"/>
                </svg>"##
        ));
        let resolver = StyleResolver::new(&d);
        assert_eq!(
            resolver.computed_value(by_id(&d, "target"), PropertyId::Fill),
            "second"
        );
    }

    #[test]
    fn even_a_zero_specificity_universal_rule_beats_a_presentation_attribute() {
        let d = doc(&format!(
            r##"<svg {NS}>
                  <style>* {{ stroke: from-universal-stylesheet; }}</style>
                  <rect id="target" stroke="from-presentation"/>
                </svg>"##
        ));
        let resolver = StyleResolver::new(&d);
        assert_eq!(
            resolver.computed_value(by_id(&d, "target"), PropertyId::Stroke),
            "from-universal-stylesheet"
        );
    }

    #[test]
    fn important_wins_within_one_style_attribute_regardless_of_position() {
        let d = doc(&format!(
            r##"<svg {NS}>
                  <rect id="target" style="fill: important-first !important; fill: red"/>
                </svg>"##
        ));
        let resolver = StyleResolver::new(&d);
        assert_eq!(
            resolver.computed_value(by_id(&d, "target"), PropertyId::Fill),
            "important-first",
            "an earlier !important declaration beats a later normal one in the same list"
        );
    }

    #[test]
    fn descendant_combinator_selector_matches_through_ancestors() {
        let d = doc(&format!(
            r##"<svg {NS}>
                  <style>g rect {{ fill: matched; }}</style>
                  <g><a><rect id="target"/></a></g>
                </svg>"##
        ));
        let resolver = StyleResolver::new(&d);
        assert_eq!(
            resolver.computed_value(by_id(&d, "target"), PropertyId::Fill),
            "matched",
            "rect has a `g` ancestor even though it is not the immediate parent"
        );
    }

    #[test]
    fn inherit_keyword_forces_the_parent_value_even_for_a_non_inherited_property() {
        let d = doc(&format!(
            r##"<svg {NS}>
                  <g id="parent" opacity="0.5">
                    <rect id="child" style="opacity: inherit"/>
                  </g>
                </svg>"##
        ));
        let resolver = StyleResolver::new(&d);
        assert_eq!(
            resolver.computed_value(by_id(&d, "child"), PropertyId::Opacity),
            "0.5"
        );
    }

    #[test]
    fn inherit_keyword_is_case_insensitive_and_falls_back_to_initial_at_the_root() {
        let d = doc(&format!(r##"<svg {NS} style="fill: INHERIT"/>"##));
        let resolver = StyleResolver::new(&d);
        assert_eq!(
            resolver.computed_value(d.svg_element(), PropertyId::Fill),
            "black"
        );
    }

    #[test]
    fn initial_keyword_ignores_an_inheritable_parent_value() {
        let d = doc(&format!(
            r##"<svg {NS}>
                  <g fill="blue">
                    <rect id="child" style="fill: initial"/>
                  </g>
                </svg>"##
        ));
        let resolver = StyleResolver::new(&d);
        assert_eq!(
            resolver.computed_value(by_id(&d, "child"), PropertyId::Fill),
            "black"
        );
    }

    #[test]
    fn unset_acts_as_inherit_for_an_inherited_property_and_initial_otherwise() {
        let d = doc(&format!(
            r##"<svg {NS}>
                  <g fill="blue" opacity="0.5">
                    <rect id="child" style="fill: unset; opacity: unset"/>
                  </g>
                </svg>"##
        ));
        let resolver = StyleResolver::new(&d);
        let child = by_id(&d, "child");
        assert_eq!(
            resolver.computed_value(child, PropertyId::Fill),
            "blue",
            "fill is inherited, so unset behaves like inherit"
        );
        assert_eq!(
            resolver.computed_value(child, PropertyId::Opacity),
            "1",
            "opacity is not inherited, so unset behaves like initial"
        );
    }

    #[test]
    fn convenience_fill_stroke_opacity_display_and_visibility_helpers() {
        let d = doc(&format!(
            r##"<svg {NS}>
                  <rect id="a" fill="red" stroke="url(#g)" opacity="150%"
                        display="none" visibility="hidden"/>
                  <rect id="b" fill="not-a-color"/>
                  <rect id="c"/>
                </svg>"##
        ));
        let resolver = StyleResolver::new(&d);
        let a = by_id(&d, "a");
        assert_eq!(
            resolver.fill(a).expect("valid paint"),
            Paint::Color(crate::values::color::Color::rgb(0xFF, 0x00, 0x00))
        );
        assert_eq!(
            resolver.stroke(a).expect("valid paint"),
            Paint::Iri {
                href: "#g".into(),
                fallback: None
            }
        );
        assert_eq!(resolver.opacity(a), 1.0, "150% clamps to 1.0");
        assert!(resolver.display_none(a));
        assert!(resolver.visibility_hidden(a));

        let b = by_id(&d, "b");
        assert!(
            resolver.fill(b).is_err(),
            "\"not-a-color\" is not a <paint>"
        );

        let c = by_id(&d, "c");
        assert!(!resolver.display_none(c));
        assert!(!resolver.visibility_hidden(c));
        assert_eq!(resolver.opacity(c), 1.0, "initial value, nothing specified");
    }

    #[test]
    fn document_accessor_returns_the_resolver_s_document() {
        let d = doc(&format!(r##"<svg {NS} id="root"/>"##));
        let resolver = StyleResolver::new(&d);
        assert_eq!(resolver.document().svg_element().id(), Some("root"));
    }

    /// `computed_value` used to call itself once per ancestor while walking
    /// up for inheritance, so a document nested deep enough risked a native
    /// stack overflow -- an uncatchable process abort, not a panic this test
    /// could assert against directly (see the module's own documentation on
    /// why it is an explicit loop now instead).
    ///
    /// [`SvgOptions`](crate::tree::document::SvgOptions)'s default parse-time
    /// depth ceiling keeps every document built through
    /// [`SvgDocument::parse_str`] well short of a depth where that would
    /// ever have mattered, so this builds one directly with [`oxixml_dom`]'s
    /// own mutation API and adopts it with
    /// [`SvgDocument::from_document`] -- the one documented, fully public
    /// way to hand this resolver a document deeper than that ceiling, with
    /// no depth validation of its own. `fill` is set once, at the root, so
    /// resolving it at the deepest leaf walks the entire chain: correct
    /// behavior here (rather than a crash) is what pins the fix.
    #[test]
    fn computed_value_walks_a_deeply_nested_inheritance_chain_without_recursing() {
        const DEPTH: usize = 1000;

        let mut raw = Document::new();
        let root = raw
            .create_element_ns(Some(SVG_NAMESPACE), "svg")
            .expect("svg element");
        raw.append_child(raw.root(), root).expect("append root");
        raw.set_attribute(root, "fill", "red")
            .expect("set fill on the root");

        let mut current = root;
        for _ in 0..DEPTH {
            let g = raw
                .create_element_ns(Some(SVG_NAMESPACE), "g")
                .expect("g element");
            raw.append_child(current, g).expect("append g");
            current = g;
        }
        let leaf = current;

        let d = SvgDocument::from_document(raw).expect("root is svg");
        let resolver = StyleResolver::new(&d);
        let leaf_element = d.element(leaf).expect("leaf is an SVG element");

        assert_eq!(
            resolver.computed_value(leaf_element, PropertyId::Fill),
            "red",
            "fill inherits all the way from the root, {DEPTH} levels up"
        );
    }
}
