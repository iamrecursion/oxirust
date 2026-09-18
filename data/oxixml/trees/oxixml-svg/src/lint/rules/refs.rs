//! Document-level reference resolution: dangling `url(#x)`/`href="#x"`
//! targets ([`resolve_dangling`]), `<use>` instantiation cycles
//! ([`detect_use_cycles`]), and gradient/pattern/filter `href`-template
//! chain cycles ([`detect_chain_cycles`]).
//!
//! These three passes all run after `super::Walker::walk` has finished,
//! because they need the *complete* `id` → node table: a reference may
//! legally point forward, to an element documented later than the
//! reference itself, so no reference can be conclusively judged dangling
//! (or acyclic) from a single left-to-right walk alone.
//!
//! # Two different cycle shapes
//!
//! `<use>` and the gradient/pattern/filter `href` chain are cycle-checked
//! by two genuinely different algorithms, because they are two genuinely
//! different graph shapes:
//!
//! - A gradient/pattern/filter `href` is *template inheritance*: each node
//!   has at most one outgoing edge, so the "would-follow" graph is a
//!   simple function graph (a linked list per starting node). Walking one
//!   node at a time with a visited set local to that walk finds a cycle
//!   exactly when a node repeats.
//! - `<use>` is *content instantiation*: expanding a `<use>` walks its
//!   target's entire subtree looking for further nested `<use>` elements,
//!   each of which recursively expands its own target. This is a real,
//!   branching graph traversal (a `<symbol>` can contain several `<use>`
//!   children, none of which are a cycle with each other), so it needs a
//!   `currently_open` set that is pushed onto when a branch is entered and
//!   popped from when that branch's expansion is fully explored — a
//!   global "ever visited" set would wrongly flag the ordinary, legal case
//!   of the same target being instantiated twice from two independent
//!   `<use>` elements. [`use_reaches_cycle`] implements this as an
//!   iterative (explicit-stack, no recursion) depth-first search.

use std::collections::{HashMap, HashSet};

use oxixml_dom::{Document, NodeId};

use crate::lint::diagnostic::{Diagnostic, codes};
use crate::ns::{SVG_NAMESPACE, XLINK_NAMESPACE};
use crate::values::iri::{effective_href, local_reference};

use super::attrs::RefSite;

/// Reports [`codes::DANGLING_REFERENCE`] for every collected [`RefSite`]
/// whose target `id` is absent from `id_table`.
pub(super) fn resolve_dangling(
    id_table: &HashMap<Box<str>, NodeId>,
    ref_sites: &[RefSite],
    diagnostics: &mut Vec<Diagnostic>,
) {
    for site in ref_sites {
        if !id_table.contains_key(&site.reference) {
            diagnostics.push(Diagnostic::warning(
                codes::DANGLING_REFERENCE,
                format!("reference to nonexistent id `#{}`", site.reference),
                site.node,
            ));
        }
    }
}

/// Resolves `node`'s effective `href` (SVG 2 `href` over legacy
/// `xlink:href`, per [`effective_href`]) to the [`NodeId`] it names, via
/// `id_table`.
///
/// Returns [`None`] when `node` has no `href`/`xlink:href`, the reference
/// is not a same-document fragment (`#id`), or the id it names does not
/// exist — every one of those is simply "nothing further to expand" for
/// cycle-detection purposes (a dangling reference is separately reported
/// by [`resolve_dangling`], not treated as a cycle).
fn resolve_href_target(
    doc: &Document,
    node: NodeId,
    id_table: &HashMap<Box<str>, NodeId>,
) -> Option<NodeId> {
    let href = doc.get_attribute(node, "href");
    let xlink_href = doc.get_attribute_ns(node, Some(XLINK_NAMESPACE), "href");
    let effective = effective_href(href, xlink_href)?;
    let id = local_reference(effective)?;
    id_table.get(id).copied()
}

/// Reports [`codes::CIRCULAR_REFERENCE`] for every `<use>` in `use_nodes`
/// whose instantiation (directly or transitively, through nested `<use>`
/// expansion) reaches itself or one of its own ancestors. One diagnostic
/// per cyclic `<use>`, attached to that `<use>` itself; independent
/// sibling `<use>` elements that are not themselves part of a cycle are
/// unaffected even if they target the same, otherwise-cyclic subtree.
pub(super) fn detect_use_cycles(
    doc: &Document,
    use_nodes: &[NodeId],
    id_table: &HashMap<Box<str>, NodeId>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    for &start in use_nodes {
        if use_reaches_cycle(doc, start, id_table) {
            diagnostics.push(Diagnostic::error(
                codes::CIRCULAR_REFERENCE,
                "`<use>` instantiation forms a reference cycle",
                start,
            ));
        }
    }
}

/// One entry of the explicit DFS stack [`use_reaches_cycle`] drives:
/// `opened` is the target node this frame is currently expanding (already
/// inserted into the walk's `currently_open` set), and `remaining` is the
/// nested `<use>` descendants of `opened` still to visit.
struct Frame {
    opened: NodeId,
    remaining: Vec<NodeId>,
}

/// See the [module documentation](self) for why this needs a
/// `currently_open` set with proper push/pop (open-on-enter,
/// close-on-backtrack) discipline rather than a single global visited set.
fn use_reaches_cycle(doc: &Document, start: NodeId, id_table: &HashMap<Box<str>, NodeId>) -> bool {
    // Instantiating `start` happens lexically inside every one of its own
    // ancestors (and inside itself, from the caller's point of view), so a
    // `<use>` that resolves back to any of those is an immediate cycle —
    // referencing one's own container, not just one's own literal self.
    let mut currently_open: HashSet<NodeId> = doc.ancestors(start).collect();
    currently_open.insert(start);

    let mut frames: Vec<Frame> = Vec::new();
    let mut current = start;

    loop {
        if let Some(target) = resolve_href_target(doc, current, id_table) {
            if currently_open.contains(&target) {
                return true;
            }
            currently_open.insert(target);
            let nested: Vec<NodeId> = doc
                .descendants(target)
                .filter(|&node| is_use_element(doc, node))
                .collect();
            frames.push(Frame {
                opened: target,
                remaining: nested,
            });
        }
        // Advance to the next `<use>` to expand, backtracking (closing
        // exhausted frames) as needed.
        loop {
            match frames.last_mut() {
                None => return false,
                Some(frame) => {
                    if let Some(next) = frame.remaining.pop() {
                        current = next;
                        break;
                    }
                    currently_open.remove(&frame.opened);
                    frames.pop();
                }
            }
        }
    }
}

fn is_use_element(doc: &Document, node: NodeId) -> bool {
    doc.element(node).is_some_and(|element| {
        element.namespace_uri() == Some(SVG_NAMESPACE) && element.local_name() == "use"
    })
}

/// Reports [`codes::CIRCULAR_REFERENCE`] for every gradient/pattern/filter
/// in `chain_nodes` whose `href` template chain loops back on a node
/// already visited earlier in the same chain. One diagnostic per cyclic
/// chain, attached to the starting node of that walk.
pub(super) fn detect_chain_cycles(
    doc: &Document,
    chain_nodes: &[NodeId],
    id_table: &HashMap<Box<str>, NodeId>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    for &start in chain_nodes {
        let mut seen: HashSet<NodeId> = HashSet::new();
        seen.insert(start);
        let mut current = start;
        while let Some(next) = resolve_href_target(doc, current, id_table) {
            if seen.contains(&next) {
                diagnostics.push(Diagnostic::error(
                    codes::CIRCULAR_REFERENCE,
                    "`href` template chain forms a reference cycle",
                    start,
                ));
                break;
            }
            seen.insert(next);
            current = next;
        }
    }
}

#[cfg(test)]
mod tests {
    use oxixml_dom::build::parse_str;

    use super::*;

    /// Builds a document and its full `id` → node table (first-wins, no
    /// duplicate handling needed for these fixtures).
    fn doc_with_ids(svg: &str) -> (Document, HashMap<Box<str>, NodeId>) {
        let doc = parse_str(svg).expect("fixture is well-formed XML");
        let mut table = HashMap::new();
        for node in doc.descendants(doc.root()) {
            if doc.kind(node) == Some(oxixml_dom::NodeKind::Element) {
                if let Some(id) = doc.get_attribute(node, "id") {
                    table.entry(Box::from(id)).or_insert(node);
                }
            }
        }
        (doc, table)
    }

    // ---- resolve_href_target ----

    #[test]
    fn resolve_href_target_finds_the_id_table_entry() {
        let (doc, table) = doc_with_ids(
            r##"<svg xmlns="http://www.w3.org/2000/svg"><use id="u" href="#t"/><rect id="t"/></svg>"##,
        );
        let use_node = *table.get("u").expect("id table has u");
        let target = resolve_href_target(&doc, use_node, &table);
        assert_eq!(target, table.get("t").copied());
    }

    #[test]
    fn resolve_href_target_is_none_without_an_href() {
        let (doc, table) =
            doc_with_ids(r##"<svg xmlns="http://www.w3.org/2000/svg"><rect id="r"/></svg>"##);
        let rect = *table.get("r").expect("id table has r");
        assert_eq!(resolve_href_target(&doc, rect, &table), None);
    }

    #[test]
    fn resolve_href_target_is_none_for_an_external_reference() {
        let (doc, table) = doc_with_ids(
            r##"<svg xmlns="http://www.w3.org/2000/svg"><use id="u" href="other.svg#t"/></svg>"##,
        );
        let use_node = *table.get("u").expect("id table has u");
        assert_eq!(resolve_href_target(&doc, use_node, &table), None);
    }

    #[test]
    fn resolve_href_target_is_none_for_a_dangling_reference() {
        let (doc, table) = doc_with_ids(
            r##"<svg xmlns="http://www.w3.org/2000/svg"><use id="u" href="#nope"/></svg>"##,
        );
        let use_node = *table.get("u").expect("id table has u");
        assert_eq!(resolve_href_target(&doc, use_node, &table), None);
    }

    #[test]
    fn resolve_href_target_prefers_href_over_xlink_href() {
        let (doc, table) = doc_with_ids(
            r##"<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
                 <use id="u" href="#a" xlink:href="#b"/><rect id="a"/><rect id="b"/>
               </svg>"##,
        );
        let use_node = *table.get("u").expect("id table has u");
        assert_eq!(
            resolve_href_target(&doc, use_node, &table),
            table.get("a").copied()
        );
    }

    // ---- use_reaches_cycle ----

    #[test]
    fn use_reaches_cycle_is_true_for_direct_self_reference() {
        let (doc, table) = doc_with_ids(
            r##"<svg xmlns="http://www.w3.org/2000/svg"><use id="u" href="#u"/></svg>"##,
        );
        let use_node = *table.get("u").expect("id table has u");
        assert!(use_reaches_cycle(&doc, use_node, &table));
    }

    #[test]
    fn use_reaches_cycle_is_false_for_a_use_with_no_target() {
        let (doc, table) =
            doc_with_ids(r##"<svg xmlns="http://www.w3.org/2000/svg"><use id="u"/></svg>"##);
        let use_node = *table.get("u").expect("id table has u");
        assert!(!use_reaches_cycle(&doc, use_node, &table));
    }

    #[test]
    fn use_reaches_cycle_is_false_for_a_simple_non_cyclic_target() {
        let (doc, table) = doc_with_ids(
            r##"<svg xmlns="http://www.w3.org/2000/svg"><symbol id="s"><rect/></symbol><use id="u" href="#s"/></svg>"##,
        );
        let use_node = *table.get("u").expect("id table has u");
        assert!(!use_reaches_cycle(&doc, use_node, &table));
    }

    // ---- detect_chain_cycles ----

    #[test]
    fn detect_chain_cycles_reports_a_two_node_cycle() {
        let (doc, table) = doc_with_ids(
            r##"<svg xmlns="http://www.w3.org/2000/svg">
                 <linearGradient id="a" href="#b"/>
                 <linearGradient id="b" href="#a"/>
               </svg>"##,
        );
        let a = *table.get("a").expect("id table has a");
        let b = *table.get("b").expect("id table has b");
        let mut diagnostics = Vec::new();
        detect_chain_cycles(&doc, &[a, b], &table, &mut diagnostics);
        assert_eq!(diagnostics.len(), 2);
        assert!(
            diagnostics
                .iter()
                .all(|d| d.code() == codes::CIRCULAR_REFERENCE)
        );
    }

    #[test]
    fn detect_chain_cycles_is_silent_for_a_plain_chain() {
        let (doc, table) = doc_with_ids(
            r##"<svg xmlns="http://www.w3.org/2000/svg">
                 <linearGradient id="base"/>
                 <linearGradient id="derived" href="#base"/>
               </svg>"##,
        );
        let derived = *table.get("derived").expect("id table has derived");
        let mut diagnostics = Vec::new();
        detect_chain_cycles(&doc, &[derived], &table, &mut diagnostics);
        assert!(diagnostics.is_empty());
    }

    // ---- resolve_dangling ----

    #[test]
    fn resolve_dangling_reports_missing_targets_only() {
        let mut table: HashMap<Box<str>, NodeId> = HashMap::new();
        table.insert("exists".into(), NodeId::from_index(1));
        let sites = vec![
            RefSite {
                node: NodeId::from_index(2),
                reference: "exists".into(),
            },
            RefSite {
                node: NodeId::from_index(3),
                reference: "missing".into(),
            },
        ];
        let mut diagnostics = Vec::new();
        resolve_dangling(&table, &sites, &mut diagnostics);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].code(), codes::DANGLING_REFERENCE);
        assert_eq!(diagnostics[0].node(), NodeId::from_index(3));
    }

    // ---- is_use_element ----

    #[test]
    fn is_use_element_is_true_only_for_svg_namespace_use() {
        let (doc, table) = doc_with_ids(
            r##"<svg xmlns="http://www.w3.org/2000/svg"><use id="u"/><rect id="r"/></svg>"##,
        );
        assert!(is_use_element(&doc, *table.get("u").expect("u")));
        assert!(!is_use_element(&doc, *table.get("r").expect("r")));
    }
}
