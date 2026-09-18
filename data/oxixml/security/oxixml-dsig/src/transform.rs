//! The reference processing pipeline —
//! [XML Signature §4.4.3.2](https://www.w3.org/TR/xmldsig-core1/#sec-ReferenceProcessingModel).
//!
//! A reference is a small dataflow graph. The `URI` is dereferenced to a *data
//! object*, each `<ds:Transform>` consumes that object and produces the next,
//! and whatever comes out the far end is digested. There are exactly two kinds
//! of data object:
//!
//! > The data-type of the result of URI dereferencing or subsequent Transforms
//! > is either an octet stream or an XPath node-set.
//!
//! and the Recommendation fixes what happens when a transform is handed the
//! wrong one:
//!
//! > If the data object is an octet stream and the next transform requires a
//! > node-set, the signature application MUST attempt to parse the octets
//! > yielding the required node-set via XML well-formed processing.
//! >
//! > If the data object is a node-set and the next transform requires octets,
//! > the signature application MUST attempt to convert the node-set to an octet
//! > stream using Canonical XML.
//!
//! Both conversions are implemented in [`Data`], and both are lossy in ways
//! that matter: parsing octets yields a node-set *with* comments, while
//! canonicalizing a node-set drops them. That asymmetry is deliberate in the
//! Recommendation and is reproduced exactly here.
//!
//! # Which transform needs what
//!
//! | transform | input | output |
//! |---|---|---|
//! | canonicalization | node-set | octets |
//! | enveloped signature | node-set | node-set |
//! | base64 | octets *or* node-set | octets |
//! | XPath filtering | node-set | node-set |
//! | XPath Filter 2.0 | node-set | node-set |

use std::collections::{BTreeMap, BTreeSet};

use oxixml_c14n::{CanonicalizationAlgorithm, InclusiveNamespaces};
use oxixml_dom::{Document, NodeId, NodeKind, build};

use crate::algorithm::TransformAlgorithm;
use crate::error::{DsigError, DsigResult};
use crate::idref::IdIndex;
use crate::model::{FilterOperation, Transform, TransformParameters};
use crate::xpath::value::{NamespaceNode, namespace_nodes};
use crate::xpath::{Context, Expression, Focus, NamespaceContext, NodeSet, XNode};

/// The document a node-set lives in: either the one being verified, or one that
/// a transform produced by re-parsing an octet stream.
#[derive(Debug)]
pub(crate) enum Tree<'a> {
    /// The document the signature is in.
    Original(&'a Document),
    /// A document parsed from an intermediate octet stream.
    Parsed(Box<Document>),
}

impl Tree<'_> {
    /// The document itself.
    pub(crate) fn document(&self) -> &Document {
        match self {
            Self::Original(document) => document,
            Self::Parsed(document) => document,
        }
    }

    /// Whether this is still the document the signature is in — which decides
    /// whether the enveloped-signature transform and `here()` can work.
    pub(crate) const fn is_original(&self) -> bool {
        matches!(self, Self::Original(_))
    }
}

/// A data object flowing through the transform chain.
#[derive(Debug)]
pub(crate) enum Data<'a> {
    /// An octet stream.
    Octets(Vec<u8>),
    /// An XPath node-set, together with the tree it indexes.
    Nodes {
        /// The document the node identifiers refer to.
        tree: Tree<'a>,
        /// The selected nodes.
        set: NodeSet,
    },
}

impl<'a> Data<'a> {
    /// Converts to octets with Canonical XML 1.0, as
    /// [§4.4.3.2](https://www.w3.org/TR/xmldsig-core1/#sec-ReferenceProcessingModel)
    /// requires.
    ///
    /// The node-set is canonicalized *as it stands*, comments included if it
    /// still has any. [§4.4.3.3](https://www.w3.org/TR/xmldsig-core1/#sec-ReferenceProcessingModel)
    /// is explicit about this: "when [XML-C14N] is passed a node-set, it
    /// processes the node-set as is: with or without comments. Only when it is
    /// called with an octet stream does it invoke its own XPath expressions".
    /// Whether comments survive was therefore already decided when the `URI`
    /// was dereferenced — which is why `#xpointer(id('x'))` keeps them and
    /// `#x` does not.
    ///
    /// # Errors
    ///
    /// [`DsigError::Canonicalization`] if canonicalization fails.
    pub(crate) fn into_octets(self) -> DsigResult<Vec<u8>> {
        match self {
            Self::Octets(octets) => Ok(octets),
            Self::Nodes { tree, set } => {
                let document = tree.document();
                let subset = to_c14n_node_set(document, &set);
                Ok(oxixml_c14n::c14n_1_0(document, &subset, true)?)
            }
        }
    }

    /// Converts to a node-set, parsing octets as XML when necessary.
    ///
    /// The parsed node-set includes comments, which is what
    /// [§6.6.3](https://www.w3.org/TR/xmldsig-core1/#sec-XPath) prescribes: "the
    /// application MUST convert the octet stream to an XPath node-set suitable
    /// for use by Canonical XML with Comments".
    ///
    /// # Errors
    ///
    /// [`DsigError::Transform`] if the octets are not well-formed XML.
    pub(crate) fn into_nodes(self) -> DsigResult<(Tree<'a>, NodeSet)> {
        match self {
            Self::Nodes { tree, set } => Ok((tree, set)),
            Self::Octets(octets) => {
                let text = String::from_utf8(octets).map_err(|_| {
                    DsigError::Transform(
                        "the octet stream is not UTF-8 and cannot be parsed as XML".to_owned(),
                    )
                })?;
                let document = build::parse_str(&text).map_err(|error| {
                    DsigError::Transform(format!(
                        "the octet stream is not well-formed XML: {error}"
                    ))
                })?;
                let set = crate::xpath::whole_document_node_set(&document);
                Ok((Tree::Parsed(Box::new(document)), set))
            }
        }
    }
}

/// Everything the transforms of one reference need to know about their
/// surroundings.
pub(crate) struct TransformContext<'a> {
    /// The document the `<ds:Signature>` is in.
    pub(crate) document: &'a Document,
    /// The `<ds:Signature>` element, which the enveloped-signature transform
    /// removes.
    pub(crate) signature: NodeId,
    /// The identifier index over [`document`](Self::document).
    pub(crate) ids: &'a IdIndex,
    /// The conventional identifier attribute names, needed to build an index
    /// over a document a transform produced by re-parsing octets.
    pub(crate) id_attributes: &'a [String],
}

impl TransformContext<'_> {
    /// The identifier index for `document`: the pre-built one when this is
    /// still the document the signature is in, a fresh one otherwise.
    fn ids_for(&self, document: &Document) -> Option<IdIndex> {
        if std::ptr::eq(document, self.document) {
            None
        } else {
            Some(IdIndex::build(
                document,
                self.id_attributes,
                self.ids.allows_duplicates(),
            ))
        }
    }
}

/// Runs one transform.
///
/// # Errors
///
/// [`DsigError::UnsupportedAlgorithm`] for XSLT, [`DsigError::Transform`] when
/// the input cannot be coerced to what the transform needs, and whatever the
/// individual transform reports.
pub(crate) fn apply<'a>(
    transform: &Transform,
    data: Data<'a>,
    context: &TransformContext<'a>,
) -> DsigResult<Data<'a>> {
    match transform.algorithm() {
        TransformAlgorithm::Canonicalization(algorithm) => {
            let prefixes = match transform.parameters() {
                TransformParameters::InclusiveNamespaces(prefixes) => prefixes.clone(),
                _ => InclusiveNamespaces::none(),
            };
            canonicalize(data, algorithm, &prefixes).map(Data::Octets)
        }
        TransformAlgorithm::EnvelopedSignature => enveloped_signature(data, context),
        TransformAlgorithm::Base64 => base64_transform(data).map(Data::Octets),
        TransformAlgorithm::XPath => match transform.parameters() {
            TransformParameters::XPath(parameter) => {
                xpath_filter(data, parameter.expression(), parameter.node(), context)
            }
            _ => Err(DsigError::Structure(
                "the XPath transform has no <ds:XPath> parameter".to_owned(),
            )),
        },
        TransformAlgorithm::XPathFilter2 => match transform.parameters() {
            TransformParameters::Filter2(steps) => filter2(data, steps, context),
            _ => Err(DsigError::Structure(
                "the XPath Filter 2.0 transform has no <XPath> parameters".to_owned(),
            )),
        },
        TransformAlgorithm::Xslt => Err(DsigError::UnsupportedAlgorithm(
            crate::algorithm::XSLT_TRANSFORM_URI.to_owned(),
        )),
    }
}

/// A canonicalization algorithm used in transform position.
///
/// An octet-stream input is parsed first, because the canonical form of a
/// document is defined over its parse tree and not over its bytes.
fn canonicalize(
    data: Data<'_>,
    algorithm: CanonicalizationAlgorithm,
    prefixes: &InclusiveNamespaces,
) -> DsigResult<Vec<u8>> {
    let (tree, set) = data.into_nodes()?;
    let document = tree.document();
    let subset = to_c14n_node_set(document, &set);
    // Unlike the implicit conversion, an explicit canonicalization transform
    // names its own algorithm, and that identifier decides whether comments
    // survive.
    Ok(algorithm.apply_with(document, &subset, prefixes)?)
}

/// The enveloped signature transform —
/// [§6.6.4](https://www.w3.org/TR/xmldsig-core1/#sec-EnvelopedSignature).
///
/// > The required transform … removes the entire `Signature` element containing
/// > the transform from the calculation of the signature.
///
/// It only makes sense against the document the signature is in: once an
/// earlier transform has re-parsed an octet stream, "the `Signature` element
/// containing the transform" is no longer a node of the tree being filtered.
fn enveloped_signature<'a>(data: Data<'a>, context: &TransformContext<'a>) -> DsigResult<Data<'a>> {
    let (tree, set) = data.into_nodes()?;
    if !tree.is_original() {
        return Err(DsigError::Transform(
            "the enveloped signature transform cannot run after a transform that re-parsed \
             the data: the <ds:Signature> element is no longer part of the tree"
                .to_owned(),
        ));
    }
    let document = tree.document();
    let excluded = subtree_members(document, context.signature);
    let kept: Vec<XNode> = set
        .nodes()
        .iter()
        .filter(|node| match node {
            XNode::Tree(id) => !excluded.contains(id),
            XNode::Namespace(namespace) => !excluded.contains(&namespace.element),
        })
        .cloned()
        .collect();
    let set = NodeSet::from_nodes(document, kept);
    Ok(Data::Nodes { tree, set })
}

/// The base64 transform — [§6.6.2](https://www.w3.org/TR/xmldsig-core1/#sec-Base-64).
///
/// A node-set input is not canonicalized: the Recommendation says it is
/// converted "by performing operations logically equivalent to 1) applying an
/// XPath transform with expression `self::text()`, then 2) sorting the nodeset
/// by document order, then concatenating the string-value of each of the
/// nodes". That is what strips the element tags around base64 content.
fn base64_transform(data: Data<'_>) -> DsigResult<Vec<u8>> {
    let text = match data {
        Data::Octets(octets) => String::from_utf8(octets)
            .map_err(|_| DsigError::Transform("base64 input is not UTF-8".to_owned()))?,
        Data::Nodes { tree, set } => {
            let document = tree.document();
            set.nodes()
                .iter()
                .filter(|node| {
                    node.as_tree().is_some_and(|id| {
                        matches!(
                            document.kind(id),
                            Some(NodeKind::Text | NodeKind::CdataSection)
                        )
                    })
                })
                .map(|node| node.string_value(document))
                .collect()
        }
    };
    oxixml_text::base64::base64_decode_forgiving(&text).map_err(|error| {
        DsigError::Transform(format!("the base64 transform input is not base64: {error}"))
    })
}

/// The XPath filtering transform —
/// [§6.6.3](https://www.w3.org/TR/xmldsig-core1/#sec-XPath).
///
/// The expression is evaluated once per node of the input node-set with that
/// node as the context node and a context position and size of 1; the node
/// survives when the result converts to `true`.
fn xpath_filter<'a>(
    data: Data<'a>,
    expression: &str,
    parameter_node: NodeId,
    context: &TransformContext<'a>,
) -> DsigResult<Data<'a>> {
    let expression = Expression::parse(expression)?;
    let namespaces = NamespaceContext::from_element(context.document, parameter_node);
    let (tree, set) = data.into_nodes()?;
    let document = tree.document();
    let same_document = std::ptr::eq(document, context.document);
    let rebuilt = context.ids_for(document);
    let ids: &IdIndex = rebuilt.as_ref().unwrap_or(context.ids);
    let evaluation = Context {
        document,
        namespaces,
        here: same_document.then_some(parameter_node),
        ids,
    };

    let mut kept = Vec::new();
    for node in set.nodes() {
        let focus = Focus::single(node.clone());
        if expression.evaluate(&evaluation, &focus)?.to_boolean() {
            kept.push(node.clone());
        }
    }
    let set = NodeSet::from_nodes(document, kept);
    Ok(Data::Nodes { tree, set })
}

/// The XPath Filter 2.0 transform —
/// [XPath Filter 2.0 §3.4](https://www.w3.org/TR/xmldsig-filter2/#sec-ProcessingModel).
///
/// A filter node-set `F` starts as the whole input document; each `<XPath>` is
/// evaluated against the root node, *subtree-expanded*, and combined into `F`
/// by intersection, subtraction or union; the output is `I ∩ F`.
fn filter2<'a>(
    data: Data<'a>,
    steps: &[crate::model::Filter2Step],
    context: &TransformContext<'a>,
) -> DsigResult<Data<'a>> {
    let (tree, input) = data.into_nodes()?;
    let document = tree.document();
    let same_document = std::ptr::eq(document, context.document);
    let rebuilt = context.ids_for(document);
    let ids: &IdIndex = rebuilt.as_ref().unwrap_or(context.ids);

    // F starts as every node of the input document.
    let mut filter: BTreeSet<XNodeKey> = crate::xpath::whole_document_node_set(document)
        .into_nodes()
        .into_iter()
        .map(XNodeKey::from)
        .collect();

    for step in steps {
        let expression = Expression::parse(step.expression())?;
        let namespaces = NamespaceContext::from_element(context.document, step.node());
        let evaluation = Context {
            document,
            namespaces,
            here: same_document.then_some(step.node()),
            ids,
        };
        let selected = expression
            .evaluate_to_node_set(&evaluation, &Focus::single(XNode::Tree(document.root())))?;
        let expanded = subtree_expansion(document, &selected);
        match step.filter() {
            FilterOperation::Intersect => filter.retain(|node| expanded.contains(node)),
            FilterOperation::Subtract => filter.retain(|node| !expanded.contains(node)),
            FilterOperation::Union => filter.extend(expanded),
        }
    }

    let kept: Vec<XNode> = input
        .into_nodes()
        .into_iter()
        .filter(|node| filter.contains(&XNodeKey::from(node.clone())))
        .collect();
    let set = NodeSet::from_nodes(document, kept);
    Ok(Data::Nodes { tree, set })
}

/// A hashable stand-in for an [`XNode`], so that the set operations of
/// XPath Filter 2.0 can use an ordered set.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum XNodeKey {
    Tree(NodeId),
    Namespace(NodeId, String),
}

impl From<XNode> for XNodeKey {
    fn from(node: XNode) -> Self {
        match node {
            XNode::Tree(id) => Self::Tree(id),
            XNode::Namespace(namespace) => Self::Namespace(namespace.element, namespace.prefix),
        }
    }
}

/// "The set of all nodes in the input document that are either present in `S`
/// or that have an ancestor in `S`", where an element's ancestors include the
/// element that owns an attribute or namespace node.
fn subtree_expansion(document: &Document, selected: &NodeSet) -> BTreeSet<XNodeKey> {
    let mut roots: BTreeSet<NodeId> = BTreeSet::new();
    let mut expanded: BTreeSet<XNodeKey> = BTreeSet::new();
    for node in selected.nodes() {
        match node {
            XNode::Tree(id) => {
                roots.insert(*id);
            }
            XNode::Namespace(namespace) => {
                expanded.insert(XNodeKey::Namespace(
                    namespace.element,
                    namespace.prefix.clone(),
                ));
            }
        }
    }
    for root in &roots {
        for member in subtree_members(document, *root) {
            expanded.insert(XNodeKey::Tree(member));
            if document.kind(member) == Some(NodeKind::Element) {
                for namespace in namespace_nodes(document, member) {
                    expanded.insert(XNodeKey::Namespace(member, namespace.prefix));
                }
            }
        }
    }
    expanded
}

/// Every node of the subtree rooted at `root`, attributes included.
pub(crate) fn subtree_members(document: &Document, root: NodeId) -> BTreeSet<NodeId> {
    let mut members = BTreeSet::new();
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if !members.insert(node) {
            continue;
        }
        for attribute in document.attributes(node) {
            members.insert(attribute);
        }
        stack.extend(document.children(node));
    }
    members
}

/// The node-set of a subtree, in the XPath data model.
///
/// `with_comments` implements the last step of
/// [§4.4.3.3](https://www.w3.org/TR/xmldsig-core1/#sec-ReferenceProcessingModel):
/// "if the URI has no fragment identifier or the fragment identifier is a
/// shortname XPointer, then delete all comment nodes".
pub(crate) fn subtree_node_set(document: &Document, root: NodeId, with_comments: bool) -> NodeSet {
    let mut nodes = Vec::new();
    for member in subtree_members(document, root) {
        match document.kind(member) {
            Some(NodeKind::Comment) if !with_comments => {}
            Some(NodeKind::Element) => {
                nodes.push(XNode::Tree(member));
                nodes.extend(
                    namespace_nodes(document, member)
                        .into_iter()
                        .map(XNode::Namespace),
                );
            }
            Some(NodeKind::Attr)
                if !crate::xpath::value::is_namespace_declaration(document, member) =>
            {
                nodes.push(XNode::Tree(member));
            }
            Some(NodeKind::Attr) => {}
            Some(_) => nodes.push(XNode::Tree(member)),
            None => {}
        }
    }
    NodeSet::from_nodes(document, nodes)
}

/// Translates an XPath node-set into the document subset `oxixml-c14n` takes.
///
/// [`oxixml_c14n::NodeSet`] ties an element's namespace nodes to the element by
/// default, which is exactly right for the node-sets XML Signature produces (the
/// `(//. | //@* | //namespace::*)` idiom never separates them). The explicit
/// per-element bookkeeping is only switched on when the node-set at hand
/// *does* separate them — which an arbitrary XPath transform is free to do.
pub(crate) fn to_c14n_node_set(document: &Document, set: &NodeSet) -> oxixml_c14n::NodeSet {
    let mut subset = oxixml_c14n::NodeSet::empty(document);
    let mut present: BTreeMap<NodeId, BTreeSet<String>> = BTreeMap::new();
    let mut elements: Vec<NodeId> = Vec::new();

    for node in set.nodes() {
        match node {
            XNode::Tree(id) => {
                subset.insert(*id);
                if document.kind(*id) == Some(NodeKind::Element) {
                    elements.push(*id);
                }
            }
            XNode::Namespace(namespace) => {
                present
                    .entry(namespace.element)
                    .or_default()
                    .insert(namespace.prefix.clone());
            }
        }
    }

    let in_scope: BTreeMap<NodeId, BTreeSet<String>> = elements
        .iter()
        .map(|element| {
            (
                *element,
                namespace_nodes(document, *element)
                    .into_iter()
                    .map(|node: NamespaceNode| node.prefix)
                    .collect(),
            )
        })
        .collect();

    let implicit_is_right = in_scope.iter().all(|(element, prefixes)| {
        present
            .get(element)
            .is_some_and(|actual| actual == prefixes)
    });
    if !implicit_is_right {
        for (element, prefixes) in &in_scope {
            let actual = present.get(element);
            for prefix in prefixes {
                let included = actual.is_some_and(|set| set.contains(prefix));
                subset.set_namespace_node(document, *element, prefix, included);
            }
        }
    }
    subset
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::DSIG_NAMESPACE;

    fn parse(xml: &str) -> Document {
        build::parse_str(xml).expect("well-formed test input")
    }

    fn id_names() -> Vec<String> {
        crate::idref::DEFAULT_ID_ATTRIBUTES
            .iter()
            .map(|name| (*name).to_owned())
            .collect()
    }

    fn context<'a>(
        document: &'a Document,
        ids: &'a IdIndex,
        names: &'a [String],
        signature: NodeId,
    ) -> TransformContext<'a> {
        TransformContext {
            document,
            signature,
            ids,
            id_attributes: names,
        }
    }

    #[test]
    fn a_node_set_is_canonicalized_as_it_stands() -> DsigResult<()> {
        let document = parse("<a><!-- c --><b x='1'/></a>");
        // A node-set that still holds its comments keeps them …
        let octets = Data::Nodes {
            tree: Tree::Original(&document),
            set: crate::xpath::whole_document_node_set(&document),
        }
        .into_octets()?;
        assert_eq!(octets, br#"<a><!-- c --><b x="1"></b></a>"#);

        // … and one whose comments were dropped when the URI was dereferenced
        // does not.
        let root = document.document_element().expect("a document element");
        let octets = Data::Nodes {
            tree: Tree::Original(&document),
            set: subtree_node_set(&document, root, false),
        }
        .into_octets()?;
        assert_eq!(octets, br#"<a><b x="1"></b></a>"#);
        Ok(())
    }

    #[test]
    fn octets_become_a_node_set_with_comments() -> DsigResult<()> {
        let data = Data::Octets(b"<a><!-- c --><b/></a>".to_vec());
        let (tree, set) = data.into_nodes()?;
        assert!(!tree.is_original());
        let document = tree.document();
        let comments = set
            .nodes()
            .iter()
            .filter(|node| {
                node.as_tree()
                    .is_some_and(|id| document.kind(id) == Some(NodeKind::Comment))
            })
            .count();
        assert_eq!(comments, 1);
        Ok(())
    }

    #[test]
    fn the_enveloped_transform_removes_the_signature() -> DsigResult<()> {
        let document = parse(
            r##"<doc><payload>x</payload><Signature xmlns="http://www.w3.org/2000/09/xmldsig#"><SignedInfo/></Signature></doc>"##,
        );
        let signature = document
            .get_elements_by_tag_name_ns(document.root(), Some(DSIG_NAMESPACE), "Signature")
            .next()
            .expect("a signature");
        let names = id_names();
        let ids = IdIndex::build(&document, &names, false);
        let data = Data::Nodes {
            tree: Tree::Original(&document),
            set: crate::xpath::whole_document_node_set(&document),
        };
        let octets = apply(
            &only_transform(
                r##"<Transform xmlns="http://www.w3.org/2000/09/xmldsig#" Algorithm="http://www.w3.org/2000/09/xmldsig#enveloped-signature"/>"##,
            ),
            data,
            &context(&document, &ids, &names, signature),
        )?
        .into_octets()?;
        assert_eq!(octets, b"<doc><payload>x</payload></doc>");
        Ok(())
    }

    /// Parses a standalone `<ds:Transform>` document and returns the model for it.
    fn only_transform(xml: &str) -> Transform {
        // The document has to outlive the returned model, so leak it: this is a
        // test helper and the leak is bounded by the test binary's lifetime.
        let document: &'static Document = Box::leak(Box::new(parse(xml)));
        let root = document.document_element().expect("a document element");
        Transform::parse(document, root).expect("a valid transform")
    }

    #[test]
    fn the_base64_transform_strips_tags_from_a_node_set() -> DsigResult<()> {
        let document = parse("<o>T3hp\n WE1M</o>");
        let root = document.document_element().expect("a document element");
        let names = id_names();
        let ids = IdIndex::build(&document, &names, false);
        let data = Data::Nodes {
            tree: Tree::Original(&document),
            set: subtree_node_set(&document, root, false),
        };
        let octets = apply(
            &only_transform(
                r##"<Transform xmlns="http://www.w3.org/2000/09/xmldsig#" Algorithm="http://www.w3.org/2000/09/xmldsig#base64"/>"##,
            ),
            data,
            &context(&document, &ids, &names, root),
        )?
        .into_octets()?;
        assert_eq!(octets, b"OxiXML");
        Ok(())
    }

    #[test]
    fn an_xslt_transform_is_reported_as_unsupported() {
        let document = parse("<a/>");
        let root = document.document_element().expect("a document element");
        let names = id_names();
        let ids = IdIndex::build(&document, &names, false);
        let result = apply(
            &only_transform(
                r##"<Transform xmlns="http://www.w3.org/2000/09/xmldsig#" Algorithm="http://www.w3.org/TR/1999/REC-xslt-19991116"/>"##,
            ),
            Data::Octets(Vec::new()),
            &context(&document, &ids, &names, root),
        );
        assert!(matches!(result, Err(DsigError::UnsupportedAlgorithm(_))));
    }

    #[test]
    fn the_xpath_transform_filters_node_by_node() -> DsigResult<()> {
        let document = parse("<doc><keep>1</keep><drop>2</drop></doc>");
        let root = document.document_element().expect("a document element");
        let names = id_names();
        let ids = IdIndex::build(&document, &names, false);
        let data = Data::Nodes {
            tree: Tree::Original(&document),
            set: crate::xpath::whole_document_node_set(&document),
        };
        let octets = apply(
            &only_transform(
                r##"<Transform xmlns="http://www.w3.org/2000/09/xmldsig#" Algorithm="http://www.w3.org/TR/1999/REC-xpath-19991116"><XPath>not(ancestor-or-self::drop)</XPath></Transform>"##,
            ),
            data,
            &context(&document, &ids, &names, root),
        )?
        .into_octets()?;
        assert_eq!(octets, b"<doc><keep>1</keep></doc>");
        Ok(())
    }

    #[test]
    fn filter2_subtracts_a_subtree() -> DsigResult<()> {
        let document = parse(r##"<doc><a Id="x">1</a><b>2</b></doc>"##);
        let root = document.document_element().expect("a document element");
        let names = id_names();
        let ids = IdIndex::build(&document, &names, false);
        let data = Data::Nodes {
            tree: Tree::Original(&document),
            set: crate::xpath::whole_document_node_set(&document),
        };
        let octets = apply(
            &only_transform(
                r##"<Transform xmlns="http://www.w3.org/2000/09/xmldsig#" Algorithm="http://www.w3.org/2002/06/xmldsig-filter2"><XPath xmlns="http://www.w3.org/2002/06/xmldsig-filter2" Filter="subtract">id('x')</XPath></Transform>"##,
            ),
            data,
            &context(&document, &ids, &names, root),
        )?
        .into_octets()?;
        assert_eq!(octets, b"<doc><b>2</b></doc>");
        Ok(())
    }

    #[test]
    fn filter2_intersects_a_subtree() -> DsigResult<()> {
        let document = parse(r##"<doc><a Id="x">1</a><b>2</b></doc>"##);
        let root = document.document_element().expect("a document element");
        let names = id_names();
        let ids = IdIndex::build(&document, &names, false);
        let data = Data::Nodes {
            tree: Tree::Original(&document),
            set: crate::xpath::whole_document_node_set(&document),
        };
        let octets = apply(
            &only_transform(
                r##"<Transform xmlns="http://www.w3.org/2000/09/xmldsig#" Algorithm="http://www.w3.org/2002/06/xmldsig-filter2"><XPath xmlns="http://www.w3.org/2002/06/xmldsig-filter2" Filter="intersect">id('x')</XPath></Transform>"##,
            ),
            data,
            &context(&document, &ids, &names, root),
        )?
        .into_octets()?;
        assert_eq!(octets, br#"<a Id="x">1</a>"#);
        Ok(())
    }

    #[test]
    fn an_exclusive_c14n_transform_honours_its_prefix_list() -> DsigResult<()> {
        let document = parse(r##"<e:env xmlns:e="urn:e" xmlns:u="urn:u"><p>x</p></e:env>"##);
        let root = document.document_element().expect("a document element");
        let payload = document.children(root).next().expect("a child");
        let names = id_names();
        let ids = IdIndex::build(&document, &names, false);
        let data = Data::Nodes {
            tree: Tree::Original(&document),
            set: subtree_node_set(&document, payload, false),
        };
        let octets = apply(
            &only_transform(
                r##"<Transform xmlns="http://www.w3.org/2000/09/xmldsig#" Algorithm="http://www.w3.org/2001/10/xml-exc-c14n#"><InclusiveNamespaces xmlns="http://www.w3.org/2001/10/xml-exc-c14n#" PrefixList="u"/></Transform>"##,
            ),
            data,
            &context(&document, &ids, &names, root),
        )?
        .into_octets()?;
        assert_eq!(octets, br#"<p xmlns:u="urn:u">x</p>"#);
        Ok(())
    }
}
