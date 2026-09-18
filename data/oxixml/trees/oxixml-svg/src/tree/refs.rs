//! Reference resolution: the document-wide `id` index, `href`/`xlink:href`
//! fragment lookup, and cycle-safe `use`/template-chain following.
//!
//! See the specification requirements catalog §6.1/§6.2: ids must be
//! unique document-wide (first occurrence wins, duplicates are surfaced —
//! not silently overwritten — for the lint layer), and any reference that
//! could recurse back on itself (`use` content instantiation, gradient/
//! pattern `href` template inheritance) must be resolved through a
//! cycle-checked lookup rather than trusted blindly.

use std::collections::{HashMap, HashSet};
use std::fmt;

use oxixml_dom::{Document, NodeId, NodeKind};

use crate::names::elements::ElementId;
use crate::tree::document::SvgDocument;
use crate::tree::typed::SvgElement;
use crate::values::iri::{self, FuncIri};

/// A document-wide index from `id` attribute value to the first element
/// that carries it, built once by a single document-order walk (per the
/// catalog's §6.1 guidance) rather than recomputed on every reference
/// lookup.
///
/// This tracks only the plain, unprefixed `id` attribute — the one SVG's
/// own content model names (catalog §2.1) — not the XML-global `xml:id`.
#[derive(Debug, Clone, Default)]
pub struct IdIndex {
    first: HashMap<Box<str>, NodeId>,
    duplicates: Vec<(Box<str>, NodeId)>,
}

impl IdIndex {
    /// Builds the index by walking `document`'s whole tree, in document
    /// order, starting from its `svg` root.
    #[must_use]
    pub fn build(document: &SvgDocument) -> Self {
        Self::build_from(document.document(), document.root())
    }

    /// Builds the index by walking the subtree rooted at `root`, in
    /// document order — the lower-level entry point for callers that have
    /// a raw [`Document`] and a root [`NodeId`] but no [`SvgDocument`] yet
    /// (or that want to index only a subtree).
    ///
    /// Every element node is considered, regardless of namespace, matching
    /// the document-wide (not SVG-namespace-scoped) uniqueness the XML `id`
    /// concept itself describes; an empty `id=""` value is ignored (not a
    /// well-formed XML name, so not a meaningful identifier to index).
    #[must_use]
    pub fn build_from(document: &Document, root: NodeId) -> Self {
        let mut first = HashMap::new();
        let mut duplicates = Vec::new();
        for node in document.traverse(root) {
            if document.kind(node) != Some(NodeKind::Element) {
                continue;
            }
            let Some(id_value) = document.get_attribute(node, "id") else {
                continue;
            };
            if id_value.is_empty() {
                continue;
            }
            if first.contains_key(id_value) {
                duplicates.push((Box::<str>::from(id_value), node));
            } else {
                first.insert(Box::<str>::from(id_value), node);
            }
        }
        Self { first, duplicates }
    }

    /// The node whose `id` attribute is `id`, if any — the *first*
    /// occurrence in document order when the id was duplicated.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<NodeId> {
        self.first.get(id).copied()
    }

    /// Every duplicate `id` occurrence found while building this index:
    /// every element beyond the first that carried an already-seen `id`
    /// value, paired with that value. Empty when the document had no
    /// duplicate ids. Intended for the lint layer (catalog §8) — this type
    /// itself only resolves references to the first occurrence and does
    /// not otherwise act on duplicates.
    #[must_use]
    pub fn duplicates(&self) -> &[(Box<str>, NodeId)] {
        &self.duplicates
    }

    /// Resolves a parsed `url(...)` reference to the node it targets.
    ///
    /// Only a local, same-document fragment reference (`url(#id)`)
    /// resolves; an absolute or relative external reference (`url(other.svg#id)`,
    /// `url(http://example.com/x.svg#id)`) returns `None`, since this type
    /// has no notion of a document base URI or of fetching another
    /// document.
    #[must_use]
    pub fn resolve_func_iri(&self, iri: &FuncIri) -> Option<NodeId> {
        self.resolve_local(iri.reference())
    }

    /// Resolves a raw `href`/`xlink:href` value to the node it targets, the
    /// same way [`Self::resolve_func_iri`] does for a `url(...)` value.
    #[must_use]
    pub fn resolve_local(&self, href: &str) -> Option<NodeId> {
        let id = iri::local_reference(href)?;
        self.get(id)
    }
}

/// An error from a cycle-checked reference-following operation
/// ([`follow_href_chain`], [`resolve_use`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RefError {
    /// Following the reference graph reached a node that is already on the
    /// active expansion path: the reference (directly or transitively)
    /// points back at itself.
    Cycle {
        /// The node at which the cycle was detected — the node whose own
        /// reference would have re-entered a node still being expanded.
        at: NodeId,
    },
    /// The reference chain exceeded the caller-supplied (or, for
    /// [`resolve_use`], internal) depth limit without completing or
    /// cycling — a guard against a pathological but acyclic chain
    /// exhausting memory.
    TooDeep {
        /// The depth limit that was exceeded.
        limit: usize,
    },
}

impl fmt::Display for RefError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cycle { at } => write!(f, "reference cycle detected at node {at}"),
            Self::TooDeep { limit } => {
                write!(f, "reference chain exceeded the depth limit of {limit}")
            }
        }
    }
}

impl std::error::Error for RefError {}

/// The effective `href`/`xlink:href` value of `node` (SVG 2's bare `href`
/// taking precedence, per [`iri::effective_href`]), or `None` if `node` is
/// not an SVG-namespace element or carries neither attribute.
fn effective_href(doc: &Document, node: NodeId) -> Option<&str> {
    SvgElement::from_node(doc, node).and_then(SvgElement::href)
}

/// Reports whether `node` is `{http://www.w3.org/2000/svg}use`.
fn is_use_element(doc: &Document, node: NodeId) -> bool {
    SvgElement::from_node(doc, node).is_some_and(|element| element.kind() == Some(ElementId::Use))
}

/// Follows the `href`/`xlink:href` chain starting at `start`: the shape
/// gradient/pattern template inheritance (catalog §6.2, third paragraph)
/// and simple `href`-to-`href` hops in general take. Each node's effective
/// `href` resolves (locally) to the next node, until a node with no (or an
/// unresolvable) `href` is reached; that node ends the chain.
///
/// Cycle-checked with a plain visited set: unlike [`resolve_use`], this
/// walk never branches (each node contributes at most one outgoing edge),
/// so a node visited twice on the same walk is unconditionally a cycle —
/// the "currently open, not globally visited" distinction [`resolve_use`]
/// needs for a branching content-instantiation graph does not apply to a
/// plain attribute-inheritance chain.
///
/// # Errors
///
/// Returns [`RefError::Cycle`] if the chain revisits a node, and
/// [`RefError::TooDeep`] if it exceeds `max` nodes without cycling.
pub fn follow_href_chain(
    doc: &Document,
    index: &IdIndex,
    start: NodeId,
    max: usize,
) -> Result<Vec<NodeId>, RefError> {
    let mut chain = Vec::new();
    let mut visited = HashSet::new();
    let mut current = start;
    loop {
        if !visited.insert(current) {
            return Err(RefError::Cycle { at: current });
        }
        chain.push(current);
        if chain.len() > max {
            return Err(RefError::TooDeep { limit: max });
        }
        let Some(href) = effective_href(doc, current) else {
            return Ok(chain);
        };
        let Some(next) = index.resolve_local(href) else {
            return Ok(chain);
        };
        current = next;
    }
}

/// The result of following an `href` chain: an ergonomic wrapper around
/// [`follow_href_chain`]'s `Vec<NodeId>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HrefChain {
    nodes: Vec<NodeId>,
}

impl HrefChain {
    /// Follows the chain starting at `start`, exactly as
    /// [`follow_href_chain`] does, wrapping the result.
    ///
    /// # Errors
    ///
    /// See [`follow_href_chain`].
    pub fn follow(
        doc: &Document,
        index: &IdIndex,
        start: NodeId,
        max: usize,
    ) -> Result<Self, RefError> {
        follow_href_chain(doc, index, start, max).map(|nodes| Self { nodes })
    }

    /// Every node on the chain, in hop order, starting with the node the
    /// chain was followed from.
    #[must_use]
    pub fn nodes(&self) -> &[NodeId] {
        &self.nodes
    }

    /// The number of nodes on the chain. Always at least 1 for a chain
    /// built by [`HrefChain::follow`]/[`follow_href_chain`] (the starting
    /// node itself, even if it has no further `href` to follow).
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the chain has no nodes at all. Always `false` for a chain
    /// built by [`HrefChain::follow`]/[`follow_href_chain`]; kept for
    /// symmetry with [`Self::len`] and to satisfy the standard
    /// `len`/`is_empty` pairing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The node the chain was followed from.
    #[must_use]
    pub fn start(&self) -> Option<NodeId> {
        self.nodes.first().copied()
    }

    /// The last node reached: the end of the `href` chain, where either no
    /// further reference was present or it did not resolve locally.
    #[must_use]
    pub fn target(&self) -> Option<NodeId> {
        self.nodes.last().copied()
    }
}

/// The depth cap [`resolve_use`] applies to its own internal nested-`use`
/// expansion, guarding against pathological (non-cyclic) nesting
/// exhausting memory or recursion depth.
const MAX_USE_DEPTH: usize = 64;

/// Resolves a `use` element's effective `href` to the node it targets,
/// validating — per catalog §6.2 — that instantiating it would not
/// (directly, or transitively through nested `use`/`symbol` instantiation)
/// reference `use_node` itself or one of `use_node`'s own ancestors.
///
/// `Ok(None)` covers both an absent `href` and one that does not resolve
/// locally (dangling or external). `Ok(Some(target))` is the *immediate*
/// target `use_node` references: following a chain of nested `use`s all
/// the way to non-`use` content is [`follow_href_chain`]'s job, not this
/// function's — `resolve_use` answers "is it safe to resolve this `use` at
/// all", not "what does it fully dereference to".
///
/// # Cycle detection scope
///
/// This validates the *whole* subtree `use_node`'s target would
/// instantiate — every `use` reachable inside it, however deeply nested,
/// is itself resolved and checked — using a "currently open" expansion-path
/// set (not a global visited set), so a legitimate DAG (the same symbol
/// instantiated by two different, non-nested `use`s) is never flagged,
/// only a node that is still being expanded when it is reached again. That
/// covers direct self-reference, referencing a direct ancestor, and mutual
/// reference between two or more `use`s whose targets transitively contain
/// each other.
///
/// # Errors
///
/// Returns [`RefError::Cycle`] when resolving would create such a cycle,
/// and [`RefError::TooDeep`] if validating the nested content exceeds this
/// function's internal depth cap (64 levels of nested `use` expansion).
pub fn resolve_use(
    doc: &Document,
    index: &IdIndex,
    use_node: NodeId,
) -> Result<Option<NodeId>, RefError> {
    let Some(href) = effective_href(doc, use_node) else {
        return Ok(None);
    };
    let Some(target) = index.resolve_local(href) else {
        return Ok(None);
    };
    if target == use_node || doc.is_ancestor_of(target, use_node) {
        return Err(RefError::Cycle { at: use_node });
    }
    let mut open = HashSet::new();
    open.insert(use_node);
    walk_use_subtree(doc, index, target, &mut open, 0)?;
    Ok(Some(target))
}

/// Recursively validates every `use` element within `node`'s subtree
/// (`node` itself included, if it is one), maintaining `open` as the set of
/// nodes on the *active* expansion path — see [`resolve_use`]'s "Cycle
/// detection scope" section for why that, and not a global visited set, is
/// the correct structure here.
fn walk_use_subtree(
    doc: &Document,
    index: &IdIndex,
    node: NodeId,
    open: &mut HashSet<NodeId>,
    depth: usize,
) -> Result<(), RefError> {
    if depth > MAX_USE_DEPTH {
        return Err(RefError::TooDeep {
            limit: MAX_USE_DEPTH,
        });
    }
    if !open.insert(node) {
        return Err(RefError::Cycle { at: node });
    }
    for candidate in doc.traverse(node) {
        if !is_use_element(doc, candidate) {
            continue;
        }
        let Some(href) = effective_href(doc, candidate) else {
            continue;
        };
        let Some(target) = index.resolve_local(href) else {
            continue;
        };
        if target == candidate || doc.is_ancestor_of(target, candidate) {
            return Err(RefError::Cycle { at: candidate });
        }
        walk_use_subtree(doc, index, target, open, depth + 1)?;
    }
    open.remove(&node);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_dom::build;

    const NS: &str = r##"xmlns="http://www.w3.org/2000/svg""##;

    fn parse(inner: &str) -> Document {
        build::parse_str(&format!("<svg {NS}>{inner}</svg>")).expect("fixture is well-formed XML")
    }

    fn node_by_id(doc: &Document, id: &str) -> NodeId {
        doc.traverse(doc.document_element().expect("document element"))
            .find(|&n| doc.get_attribute(n, "id") == Some(id))
            .unwrap_or_else(|| panic!("no element with id {id:?}"))
    }

    #[test]
    fn build_from_indexes_every_id_first_occurrence_wins() {
        let doc = parse(r##"<g id="a"/><rect id="b"/><g id="a"/>"##);
        let root = doc.document_element().expect("root");
        let index = IdIndex::build_from(&doc, root);

        let first_a = node_by_id(&doc, "a");
        assert_eq!(index.get("a"), Some(first_a));
        assert_eq!(index.get("b"), Some(node_by_id(&doc, "b")));
        assert_eq!(index.get("nope"), None);

        assert_eq!(index.duplicates().len(), 1);
        assert_eq!(&*index.duplicates()[0].0, "a");
        assert_ne!(
            index.duplicates()[0].1,
            first_a,
            "the duplicate is the *second* node"
        );
    }

    #[test]
    fn build_from_ignores_empty_id_values() {
        let doc = parse(r##"<g id=""/>"##);
        let root = doc.document_element().expect("root");
        let index = IdIndex::build_from(&doc, root);
        assert_eq!(index.get(""), None);
        assert!(index.duplicates().is_empty());
    }

    #[test]
    fn resolve_local_and_func_iri_only_resolve_same_document_fragments() {
        let doc = parse(r##"<rect id="r"/>"##);
        let root = doc.document_element().expect("root");
        let index = IdIndex::build_from(&doc, root);

        assert_eq!(index.resolve_local("#r"), Some(node_by_id(&doc, "r")));
        assert_eq!(index.resolve_local("#nope"), None);
        assert_eq!(
            index.resolve_local("other.svg#r"),
            None,
            "external reference"
        );

        let iri = FuncIri::new("#r");
        assert_eq!(index.resolve_func_iri(&iri), Some(node_by_id(&doc, "r")));
        let external = FuncIri::new("other.svg#r");
        assert_eq!(index.resolve_func_iri(&external), None);
    }

    #[test]
    fn follow_href_chain_walks_a_gradient_template_chain() {
        let doc = parse(
            r##"<linearGradient id="base"><stop offset="0" stop-color="red"/></linearGradient>
                <linearGradient id="mid" href="#base"/>
                <linearGradient id="derived" href="#mid"/>"##,
        );
        let root = doc.document_element().expect("root");
        let index = IdIndex::build_from(&doc, root);
        let derived = node_by_id(&doc, "derived");

        let chain = follow_href_chain(&doc, &index, derived, 16).expect("no cycle");
        assert_eq!(
            chain,
            vec![derived, node_by_id(&doc, "mid"), node_by_id(&doc, "base")]
        );
    }

    #[test]
    fn href_chain_wrapper_matches_the_free_function() {
        let doc =
            parse(r##"<linearGradient id="base"/><linearGradient id="derived" href="#base"/>"##);
        let root = doc.document_element().expect("root");
        let index = IdIndex::build_from(&doc, root);
        let derived = node_by_id(&doc, "derived");

        let chain = HrefChain::follow(&doc, &index, derived, 16).expect("no cycle");
        assert_eq!(chain.len(), 2);
        assert!(!chain.is_empty());
        assert_eq!(chain.start(), Some(derived));
        assert_eq!(chain.target(), Some(node_by_id(&doc, "base")));
        assert_eq!(chain.nodes(), [derived, node_by_id(&doc, "base")]);
    }

    #[test]
    fn follow_href_chain_detects_a_self_cycle() {
        let doc = parse(r##"<linearGradient id="g" href="#g"/>"##);
        let root = doc.document_element().expect("root");
        let index = IdIndex::build_from(&doc, root);
        let g = node_by_id(&doc, "g");

        let error = follow_href_chain(&doc, &index, g, 16).expect_err("self-reference is a cycle");
        assert_eq!(error, RefError::Cycle { at: g });
    }

    #[test]
    fn follow_href_chain_detects_a_two_node_cycle() {
        let doc =
            parse(r##"<linearGradient id="a" href="#b"/><linearGradient id="b" href="#a"/>"##);
        let root = doc.document_element().expect("root");
        let index = IdIndex::build_from(&doc, root);
        let a = node_by_id(&doc, "a");

        let error = follow_href_chain(&doc, &index, a, 16).expect_err("a <-> b is a cycle");
        assert!(matches!(error, RefError::Cycle { .. }));
    }

    #[test]
    fn follow_href_chain_reports_too_deep_before_stack_overflowing() {
        let doc = parse(r##"<linearGradient id="only" href="#missing"/>"##);
        let root = doc.document_element().expect("root");
        let index = IdIndex::build_from(&doc, root);
        let only = node_by_id(&doc, "only");

        // The chain terminates immediately (its href is dangling), so a
        // max of 0 makes even the one-node chain "too deep".
        let error = follow_href_chain(&doc, &index, only, 0).expect_err("exceeds max of 0");
        assert_eq!(error, RefError::TooDeep { limit: 0 });
    }

    #[test]
    fn resolve_use_detects_a_direct_self_reference() {
        let doc = parse(r##"<use id="u" href="#u"/>"##);
        let root = doc.document_element().expect("root");
        let index = IdIndex::build_from(&doc, root);
        let u = node_by_id(&doc, "u");

        let error = resolve_use(&doc, &index, u).expect_err("self-reference is a cycle");
        assert_eq!(error, RefError::Cycle { at: u });
    }

    #[test]
    fn resolve_use_detects_ancestor_containment() {
        let doc = parse(r##"<g id="g"><use id="ug" href="#g"/></g>"##);
        let root = doc.document_element().expect("root");
        let index = IdIndex::build_from(&doc, root);
        let ug = node_by_id(&doc, "ug");

        let error = resolve_use(&doc, &index, ug).expect_err("g is an ancestor of ug");
        assert_eq!(error, RefError::Cycle { at: ug });
    }

    #[test]
    fn resolve_use_detects_a_nested_use_to_use_cycle() {
        let doc = parse(
            r##"<g id="a"><use id="ua" href="#b"/></g>
                <g id="b"><use id="ub" href="#a"/></g>"##,
        );
        let root = doc.document_element().expect("root");
        let index = IdIndex::build_from(&doc, root);
        let ua = node_by_id(&doc, "ua");

        let error = resolve_use(&doc, &index, ua)
            .expect_err("a and b mutually instantiate each other through ua/ub");
        assert!(matches!(error, RefError::Cycle { .. }));
    }

    #[test]
    fn resolve_use_allows_a_legitimate_dag_same_symbol_twice() {
        let doc = parse(
            r##"<symbol id="s"><rect/></symbol>
                <g id="container"><use id="ia" href="#s"/><use id="ib" href="#s"/></g>
                <use id="outer" href="#container"/>"##,
        );
        let root = doc.document_element().expect("root");
        let index = IdIndex::build_from(&doc, root);
        let outer = node_by_id(&doc, "outer");

        let target = resolve_use(&doc, &index, outer)
            .expect("the same symbol used twice, not nested in each other, is not a cycle");
        assert_eq!(target, Some(node_by_id(&doc, "container")));
    }

    #[test]
    fn resolve_use_follows_a_nested_use_chain_to_its_immediate_target() {
        let doc = parse(r##"<rect id="r"/><use id="u2" href="#r"/><use id="u1" href="#u2"/>"##);
        let root = doc.document_element().expect("root");
        let index = IdIndex::build_from(&doc, root);
        let u1 = node_by_id(&doc, "u1");

        let target = resolve_use(&doc, &index, u1).expect("nested use chain, no cycle");
        assert_eq!(
            target,
            Some(node_by_id(&doc, "u2")),
            "resolve_use returns the immediate target, not the fully dereferenced one"
        );
    }

    #[test]
    fn resolve_use_returns_none_for_absent_or_dangling_href() {
        let doc = parse(r##"<use id="no-href"/><use id="dangling" href="#nope"/>"##);
        let root = doc.document_element().expect("root");
        let index = IdIndex::build_from(&doc, root);

        assert_eq!(
            resolve_use(&doc, &index, node_by_id(&doc, "no-href")).expect("ok"),
            None
        );
        assert_eq!(
            resolve_use(&doc, &index, node_by_id(&doc, "dangling")).expect("ok"),
            None
        );
    }

    #[test]
    fn ref_error_display_mentions_the_node_or_limit() {
        let cycle = RefError::Cycle {
            at: NodeId::from_index(3),
        };
        assert!(cycle.to_string().contains('3'));
        let too_deep = RefError::TooDeep { limit: 64 };
        assert!(too_deep.to_string().contains("64"));
    }
}
