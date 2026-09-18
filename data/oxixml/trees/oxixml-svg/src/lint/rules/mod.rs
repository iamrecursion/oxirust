//! The SVG linter: [`Linter`], driven by a [`crate::lint::diagnostic::LintOptions`].
//!
//! # Design: decoupled from the typed document model
//!
//! [`Linter::lint`] operates directly on a plain [`oxixml_dom::Document`],
//! finding the SVG root itself — **not** on `crate::tree::SvgDocument`
//! (the typed model built on top of this crate's parser, developed
//! separately). This keeps the linter usable standalone against any DOM
//! tree that happens to hold SVG content, and avoids a dependency between
//! the two halves of the crate while both are under active development. A
//! convenience method that lints a `SvgDocument` directly is expected to
//! land as a thin wrapper once that type is stable; it is out of this
//! module's scope.
//!
//! # Algorithm
//!
//! [`Linter::lint`] requires the document element to be an `svg` element
//! in [`crate::ns::SVG_NAMESPACE`]; if it is not, [`codes::NOT_SVG`] is the
//! only diagnostic returned. Otherwise:
//!
//! 1. An iterative (explicit-stack, no recursion) pre-order walk of every
//!    SVG-namespace element, performing every check that only needs the
//!    element itself and its own attributes/children: unknown
//!    element/attribute, applicability, deprecation, value grammar, the
//!    `style` attribute, `href`/`xlink:href` duplication, content model,
//!    and character-data placement. The walk also builds the document's
//!    `id` → node table (first occurrence wins; a later duplicate is
//!    reported immediately) and collects every reference site (a
//!    `url(#x)`/`href="#x"`-shaped value) for the next step. A subtree
//!    rooted at a non-SVG-namespace element (typically `foreignObject`
//!    content or `metadata`'s RDF payload) is opaque: neither descended
//!    into nor checked.
//! 2. A document-level reference-resolution pass, run once the full `id`
//!    table exists (a reference may legally point *forward*, to an
//!    element documented later than the reference itself): dangling
//!    fragment references, `use` instantiation cycles, and
//!    gradient/pattern/filter `href`-chain cycles.
//! 3. The collected diagnostics are stable-sorted by
//!    [`oxixml_dom::Document::document_order`], so that document order is
//!    primary and, for two diagnostics on the same node, the order they
//!    were discovered in (step 1 before step 2) is preserved.
//! 4. If more diagnostics were found than
//!    [`LintOptions::max_diagnostics_limit`] allows, the list is truncated
//!    to that many and [`codes::TRUNCATED`] is appended.
//!
//! # Deliberately unimplemented attribute-value grammars
//!
//! [`LintOptions::check_values`] validates every attribute family the
//! frozen `crate::values`/`crate::style::property` parsers cover, plus every
//! purely-enumerated-keyword attribute (`gradientUnits`, `calcMode`,
//! `spreadMethod`, and the rest of that shape) against the small, fixed,
//! specification-defined vocabulary `attrs::validate_enum` checks them
//! against (see `attrs::validate_attr_value` for the exhaustive dispatch).
//! What is left unchecked is narrow and structural rather than a missing
//! keyword table: `keyTimes`/`keySplines`/`values`, whose validity is a
//! cross-attribute cardinality check (their item counts must agree with
//! each other and with the animation's other timing attributes) rather than
//! a single value's grammar, are silently not value-checked. This is a
//! documented scope cut, not an oversight: it needs the whole set of an
//! animation element's attributes together, not a per-attribute dispatch.

mod attrs;
mod content;
mod refs;

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use oxixml_dom::{Document, NodeId};

use crate::lint::diagnostic::{Diagnostic, LintOptions, codes};
use crate::names::elements::{ElementId, ElementStatus};
use crate::ns::SVG_NAMESPACE;

/// Validates an [`oxixml_dom::Document`] holding SVG content against the
/// rule families [`LintOptions`] selects.
///
/// See the [module documentation](self) for the algorithm and its
/// deliberate scope cuts.
///
/// # Example
/// ```
/// use oxixml_svg::lint::diagnostic::codes;
/// use oxixml_svg::lint::rules::Linter;
///
/// let doc = oxixml_dom::build::parse_str(
///     r#"<svg xmlns="http://www.w3.org/2000/svg"><rect width="-1"/></svg>"#,
/// )
/// .expect("well-formed XML");
/// let diagnostics = Linter::new().lint(&doc);
/// assert!(diagnostics.iter().any(|d| d.code() == codes::NEGATIVE_DIMENSION));
/// ```
#[derive(Debug, Clone, Default)]
pub struct Linter {
    options: LintOptions,
}

impl Linter {
    /// Creates a linter with every check enabled (see [`LintOptions::new`]).
    #[must_use]
    pub fn new() -> Self {
        Self {
            options: LintOptions::new(),
        }
    }

    /// Creates a linter with the given options.
    #[must_use]
    pub const fn with_options(options: LintOptions) -> Self {
        Self { options }
    }

    /// The options this linter was constructed with.
    #[must_use]
    pub const fn options(&self) -> &LintOptions {
        &self.options
    }

    /// Lints `doc`, returning every diagnostic found, stable-sorted in
    /// document order (see the [module documentation](self)).
    #[must_use]
    pub fn lint(&self, doc: &Document) -> Vec<Diagnostic> {
        let Some(root) = doc.document_element() else {
            return vec![Diagnostic::error(
                codes::NOT_SVG,
                "the document has no element in the SVG namespace",
                NodeId::DOCUMENT,
            )];
        };
        if !is_svg_root(doc, root) {
            let message = match doc.element(root) {
                Some(element) => format!(
                    "the document element `{}` is not an `svg` element in the SVG namespace",
                    element.qualified_name()
                ),
                None => {
                    "the document element is not an `svg` element in the SVG namespace".to_owned()
                }
            };
            return vec![Diagnostic::error(codes::NOT_SVG, message, root)];
        }

        let mut walker = Walker::new(&self.options);
        walker.walk(doc, root);
        let mut diagnostics = walker.diagnostics;

        if self.options.checks_references() {
            refs::resolve_dangling(&walker.id_table, &walker.ref_sites, &mut diagnostics);
            refs::detect_use_cycles(doc, &walker.use_nodes, &walker.id_table, &mut diagnostics);
            refs::detect_chain_cycles(doc, &walker.chain_nodes, &walker.id_table, &mut diagnostics);
        }

        diagnostics.sort_by_key(|d| doc.document_order(d.node()).unwrap_or(u32::MAX));

        let total = diagnostics.len();
        let limit = self.options.max_diagnostics_limit();
        if total > limit {
            diagnostics.truncate(limit);
            diagnostics.push(Diagnostic::info(
                codes::TRUNCATED,
                format!(
                    "{} further diagnostic(s) were suppressed (max_diagnostics = {limit})",
                    total - limit
                ),
                root,
            ));
        }

        diagnostics
    }
}

/// True when `node` is an `svg` element in [`SVG_NAMESPACE`].
fn is_svg_root(doc: &Document, node: NodeId) -> bool {
    doc.element(node).is_some_and(|element| {
        element.namespace_uri() == Some(SVG_NAMESPACE) && element.local_name() == "svg"
    })
}

/// The mutable state threaded through the main element/attribute walk:
/// the diagnostics found so far, the `id` → node table, and the
/// reference/`use`/href-chain sites collected for the document-level pass
/// [`Linter::lint`] runs afterward.
struct Walker<'a> {
    options: &'a LintOptions,
    diagnostics: Vec<Diagnostic>,
    id_table: HashMap<Box<str>, NodeId>,
    ref_sites: Vec<attrs::RefSite>,
    use_nodes: Vec<NodeId>,
    chain_nodes: Vec<NodeId>,
}

impl<'a> Walker<'a> {
    fn new(options: &'a LintOptions) -> Self {
        Self {
            options,
            diagnostics: Vec::new(),
            id_table: HashMap::new(),
            ref_sites: Vec::new(),
            use_nodes: Vec::new(),
            chain_nodes: Vec::new(),
        }
    }

    /// Walks the subtree rooted at `root` (the SVG root element) in an
    /// iterative pre-order (document-order) traversal, using an explicit
    /// stack rather than recursion.
    fn walk(&mut self, doc: &Document, root: NodeId) {
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            let Some(element) = doc.element(node) else {
                continue;
            };
            if element.namespace_uri() != Some(SVG_NAMESPACE) {
                // Opaque foreign-namespace content (foreignObject/metadata
                // payloads, or any other embedded foreign markup): do not
                // descend, do not check.
                continue;
            }
            let local_name = element.local_name();
            let Some(kind) = ElementId::from_name(local_name) else {
                if self.options.checks_unknown() {
                    self.diagnostics.push(Diagnostic::warning(
                        codes::UNKNOWN_ELEMENT,
                        format!("unknown SVG element `<{local_name}>`"),
                        node,
                    ));
                }
                self.note_id_if_present(doc, node);
                push_children_reversed(doc, node, &mut stack);
                continue;
            };

            if self.options.checks_deprecated() && kind.status() == ElementStatus::DeprecatedSvg2 {
                self.diagnostics.push(Diagnostic::warning(
                    codes::DEPRECATED_ELEMENT,
                    deprecated_element_message(kind),
                    node,
                ));
            }

            let mut seen_href = false;
            let mut seen_xlink_href = false;
            for attr_node in doc.attributes(node) {
                attrs::process_attribute(
                    doc,
                    node,
                    kind,
                    attr_node,
                    self.options,
                    &mut self.diagnostics,
                    &mut self.id_table,
                    &mut self.ref_sites,
                    &mut seen_href,
                    &mut seen_xlink_href,
                );
            }
            if seen_href && seen_xlink_href && self.options.checks_references() {
                self.diagnostics.push(Diagnostic::warning(
                    codes::HREF_DUPLICATED,
                    "both `href` and `xlink:href` are present; `href` takes precedence",
                    node,
                ));
            }

            if self.options.checks_references() {
                self.collect_reference_scaffolding(doc, node, kind);
            }

            if self.options.checks_content_model() {
                content::check_content_model(doc, node, kind, &mut self.diagnostics);
            }

            push_children_reversed(doc, node, &mut stack);
        }
    }

    /// Records `<use>`'s empty-reference diagnostic, and collects `<use>`
    /// / gradient / pattern / filter nodes that carry a resolvable `href`
    /// for the document-level cycle-detection pass.
    fn collect_reference_scaffolding(&mut self, doc: &Document, node: NodeId, kind: ElementId) {
        use crate::ns::XLINK_NAMESPACE;
        use crate::values::iri::effective_href;

        let href = doc.get_attribute(node, "href");
        let xlink_href = doc.get_attribute_ns(node, Some(XLINK_NAMESPACE), "href");
        let effective = effective_href(href, xlink_href);

        match kind {
            ElementId::Use => match effective {
                None => self.diagnostics.push(Diagnostic::warning(
                    codes::EMPTY_USE_REFERENCE,
                    "`<use>` has no `href`/`xlink:href` reference",
                    node,
                )),
                Some(value) if value.trim().is_empty() => {
                    self.diagnostics.push(Diagnostic::warning(
                        codes::EMPTY_USE_REFERENCE,
                        "`<use>` has an empty `href`/`xlink:href` reference",
                        node,
                    ))
                }
                Some(_) => self.use_nodes.push(node),
            },
            ElementId::LinearGradient
            | ElementId::RadialGradient
            | ElementId::Pattern
            | ElementId::Filter
                if effective.is_some() =>
            {
                self.chain_nodes.push(node);
            }
            _ => {}
        }
    }

    /// Records `node`'s plain `id` attribute into the id table, reporting
    /// [`codes::DUPLICATE_ID`] for a value already seen earlier in
    /// document order. Used directly for unknown elements (which skip the
    /// rest of attribute processing); known elements record their `id`
    /// from inside [`attrs::process_attribute`] instead, to avoid a second
    /// attribute lookup.
    fn note_id_if_present(&mut self, doc: &Document, node: NodeId) {
        if let Some(value) = doc.get_attribute(node, "id") {
            attrs::note_id(value, node, &mut self.id_table, &mut self.diagnostics);
        }
    }
}

/// Pushes the children of `node` onto `stack` in reverse document order,
/// so that popping the stack (last-in, first-out) visits them in forward
/// document order.
fn push_children_reversed(doc: &Document, node: NodeId, stack: &mut Vec<NodeId>) {
    for child in doc.children(node).rev() {
        stack.push(child);
    }
}

/// The message for a [`codes::DEPRECATED_ELEMENT`] diagnostic, naming a
/// suggested replacement when the specification catalog names an obvious
/// one.
fn deprecated_element_message(kind: ElementId) -> String {
    let replacement: Option<&str> = match kind {
        ElementId::Tref => Some("`use` (or duplicating the referenced text directly)"),
        ElementId::AnimateColor => Some("`animate`, which handles color values directly in SVG 2"),
        ElementId::AltGlyph
        | ElementId::AltGlyphDef
        | ElementId::AltGlyphItem
        | ElementId::GlyphRef => Some("a plain `text`/`tspan` run styled with CSS Fonts"),
        ElementId::Font
        | ElementId::FontFace
        | ElementId::FontFaceSrc
        | ElementId::FontFaceUri
        | ElementId::FontFaceFormat
        | ElementId::FontFaceName
        | ElementId::Glyph
        | ElementId::MissingGlyph
        | ElementId::Hkern
        | ElementId::Vkern => Some("CSS Fonts / WOFF"),
        ElementId::ColorProfile => None,
        ElementId::Cursor => Some("the CSS `cursor` property with an image URL"),
        _ => None,
    };
    match replacement {
        Some(replacement) => format!(
            "`<{}>` is deprecated in SVG 2; consider {replacement} instead",
            kind.as_str()
        ),
        None => format!("`<{}>` is deprecated in SVG 2", kind.as_str()),
    }
}
