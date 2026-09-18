//! Dereferencing a `<ds:Reference URI="…">` —
//! [XML Signature §4.4.3.2–§4.4.3.3](https://www.w3.org/TR/xmldsig-core1/#sec-ReferenceProcessingModel).
//!
//! There are four cases, and the Recommendation gives each of them a different
//! answer:
//!
//! | `URI` | result | comments |
//! |---|---|---|
//! | `""` | the whole containing document as a node-set | removed |
//! | `"#name"` (shortname XPointer) | the identified element's subtree | removed |
//! | `"#xpointer(…)"` | the XPointer's result, subtree-expanded | **kept** |
//! | anything else | octets, from a [`UriResolver`] | n/a |
//! | absent | octets, from a [`UriResolver`] | n/a |
//!
//! The comment column is the part everyone gets wrong. §4.4.3.3 spells the rule
//! out: "if the URI has no fragment identifier or the fragment identifier is a
//! shortname XPointer, then delete all comment nodes", and explains why —
//! canonicalization processes a node-set exactly as given, so the only place
//! the default "strip comments" behaviour can be applied is here. Which is also
//! why `#xpointer(/)` and `#xpointer(id('x'))` exist at all: they are the way to
//! sign a document *with* its comments.
//!
//! # XPointer support
//!
//! §4.4.3.2 makes the null URI and shortname XPointers mandatory and
//! `#xpointer(/)` and `#xpointer(id('ID'))` recommended, and discourages
//! anything further. This module implements the two recommended forms as
//! special cases of a general rule — evaluate the text inside `xpointer(…)` as
//! an XPath expression against the containing document, then subtree-expand the
//! result — which reproduces both exactly and happens to accept the wider
//! `xpointer()` scheme too. The `xmlns()` scheme is not supported: a prefix used
//! inside an `xpointer()` here resolves against the declarations in scope on the
//! `<ds:Reference>` element.

use oxixml_dom::{Document, NodeId};

use crate::error::{DsigError, DsigResult};
use crate::idref::IdIndex;
use crate::model::Reference;
use crate::transform::{Data, Tree, subtree_node_set};
use crate::xpath::{Context, Expression, Focus, NamespaceContext, NodeSet, XNode};

/// Fetches the octets a non-same-document `URI` names.
///
/// Nothing in this crate performs I/O: an XML signature is frequently checked on
/// data that arrived from the network, and a verifier that quietly dereferences
/// a `URI` out of an untrusted document is a server-side request forgery
/// waiting to happen. Detached signatures are still perfectly usable — supply a
/// resolver that serves the payloads the application already has.
pub trait UriResolver {
    /// Returns the octets for `uri`, or an error explaining why not.
    ///
    /// `uri` is the `URI` attribute verbatim; [`None`] means the attribute was
    /// absent, which [XML Signature §4.4.3](https://www.w3.org/TR/xmldsig-core1/#sec-Reference)
    /// leaves entirely to the application.
    ///
    /// # Errors
    ///
    /// [`DsigError::Dereference`] when the resolver declines or fails.
    fn resolve(&self, uri: Option<&str>) -> DsigResult<Vec<u8>>;
}

impl<F> UriResolver for F
where
    F: Fn(Option<&str>) -> DsigResult<Vec<u8>>,
{
    fn resolve(&self, uri: Option<&str>) -> DsigResult<Vec<u8>> {
        self(uri)
    }
}

/// A [`UriResolver`] backed by a fixed table, which is what a detached
/// signature over data the application already holds needs.
///
/// # Example
///
/// ```
/// use oxixml_dsig::{StaticResolver, UriResolver};
///
/// let resolver = StaticResolver::new().with("payload.bin", b"contents".to_vec());
/// assert_eq!(resolver.resolve(Some("payload.bin"))?, b"contents");
/// assert!(resolver.resolve(Some("elsewhere")).is_err());
/// # Ok::<(), oxixml_dsig::DsigError>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StaticResolver {
    entries: Vec<(String, Vec<u8>)>,
    absent: Option<Vec<u8>>,
}

impl StaticResolver {
    /// An empty table, which resolves nothing.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
            absent: None,
        }
    }

    /// Maps a `URI` to its octets.
    #[must_use]
    pub fn with(mut self, uri: impl Into<String>, octets: impl Into<Vec<u8>>) -> Self {
        self.entries.push((uri.into(), octets.into()));
        self
    }

    /// Supplies the octets for a `<ds:Reference>` with *no* `URI` attribute.
    #[must_use]
    pub fn with_implicit(mut self, octets: impl Into<Vec<u8>>) -> Self {
        self.absent = Some(octets.into());
        self
    }
}

impl UriResolver for StaticResolver {
    fn resolve(&self, uri: Option<&str>) -> DsigResult<Vec<u8>> {
        match uri {
            None => self.absent.clone().ok_or_else(|| {
                DsigError::Dereference(
                    "the Reference has no URI and the resolver supplies no implicit data"
                        .to_owned(),
                )
            }),
            Some(uri) => self
                .entries
                .iter()
                .find(|(candidate, _)| candidate == uri)
                .map(|(_, octets)| octets.clone())
                .ok_or_else(|| DsigError::Dereference(format!("no data registered for `{uri}`"))),
        }
    }
}

/// Dereferences a reference's `URI`.
///
/// # Errors
///
/// [`DsigError::Dereference`] when a fragment names no element, names more than
/// one, or when an external reference has no resolver;
/// [`DsigError::XPath`] when an `xpointer()` is not a valid expression.
pub(crate) fn dereference<'a>(
    document: &'a Document,
    reference: &Reference,
    ids: &IdIndex,
    resolver: Option<&dyn UriResolver>,
) -> DsigResult<Data<'a>> {
    let Some(uri) = reference.uri() else {
        return external(None, resolver);
    };

    if uri.is_empty() {
        // §4.4.3.3: "dereferencing a null URI (URI="") MUST result in an XPath
        // node-set that includes every non-comment node of the XML document".
        let set = without_comments(document, crate::xpath::whole_document_node_set(document));
        return Ok(Data::Nodes {
            tree: Tree::Original(document),
            set,
        });
    }

    let Some(fragment) = uri.strip_prefix('#') else {
        return external(Some(uri), resolver);
    };

    if let Some(expression) = xpointer_expression(fragment) {
        let set = xpointer(document, reference.node(), expression, ids)?;
        return Ok(Data::Nodes {
            tree: Tree::Original(document),
            set,
        });
    }

    // A shortname XPointer: an element by identifier, comments removed.
    let element = ids
        .get(fragment)
        .ok_or_else(|| match ids.duplicate_count(fragment) {
            Some(count) => DsigError::Dereference(format!(
                "the identifier `{fragment}` names {count} elements; refusing to guess which one \
                 the signature covers"
            )),
            None => DsigError::Dereference(format!("no element has the identifier `{fragment}`")),
        })?;
    Ok(Data::Nodes {
        tree: Tree::Original(document),
        set: subtree_node_set(document, element, false),
    })
}

/// The text inside `xpointer(…)`, if the fragment has that shape.
fn xpointer_expression(fragment: &str) -> Option<&str> {
    fragment
        .strip_prefix("xpointer(")
        .and_then(|rest| rest.strip_suffix(')'))
}

/// Evaluates an `xpointer()` and expands the result to full subtrees, keeping
/// comments.
fn xpointer(
    document: &Document,
    reference_node: NodeId,
    expression: &str,
    ids: &IdIndex,
) -> DsigResult<NodeSet> {
    let expression = Expression::parse(expression)?;
    let context = Context {
        document,
        namespaces: NamespaceContext::from_element(document, reference_node),
        here: Some(reference_node),
        ids,
    };
    let selected =
        expression.evaluate_to_node_set(&context, &Focus::single(XNode::Tree(document.root())))?;

    let mut nodes = Vec::new();
    for node in selected.nodes() {
        match node {
            XNode::Namespace(_) => nodes.push(node.clone()),
            XNode::Tree(id) => {
                nodes.extend(subtree_node_set(document, *id, true).into_nodes());
            }
        }
    }
    Ok(NodeSet::from_nodes(document, nodes))
}

/// Drops every comment node from a node-set.
fn without_comments(document: &Document, set: NodeSet) -> NodeSet {
    let kept: Vec<XNode> = set
        .into_nodes()
        .into_iter()
        .filter(|node| {
            node.as_tree()
                .is_none_or(|id| document.kind(id) != Some(oxixml_dom::NodeKind::Comment))
        })
        .collect();
    NodeSet::from_nodes(document, kept)
}

fn external<'a>(uri: Option<&str>, resolver: Option<&dyn UriResolver>) -> DsigResult<Data<'a>> {
    match resolver {
        Some(resolver) => resolver.resolve(uri).map(Data::Octets),
        None => Err(DsigError::Dereference(format!(
            "`{}` is not a same-document reference and no UriResolver was configured",
            uri.unwrap_or("<no URI>")
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idref::DEFAULT_ID_ATTRIBUTES;
    use oxixml_dom::build;

    fn names() -> Vec<String> {
        DEFAULT_ID_ATTRIBUTES
            .iter()
            .map(|name| (*name).to_owned())
            .collect()
    }

    fn reference(document: &Document, uri: &str) -> Reference {
        let xml = format!(
            r##"<Reference xmlns="http://www.w3.org/2000/09/xmldsig#" URI="{uri}">
                  <DigestMethod Algorithm="http://www.w3.org/2001/04/xmlenc#sha256"/>
                  <DigestValue></DigestValue>
                </Reference>"##
        );
        let reference_document: &'static Document = Box::leak(Box::new(
            build::parse_str(&xml).expect("well-formed test input"),
        ));
        let root = reference_document
            .document_element()
            .expect("a document element");
        let _ = document;
        Reference::parse(reference_document, root).expect("a valid Reference")
    }

    fn octets(document: &Document, uri: &str) -> DsigResult<Vec<u8>> {
        let ids = IdIndex::build(document, &names(), false);
        dereference(document, &reference(document, uri), &ids, None)?.into_octets()
    }

    #[test]
    fn a_null_uri_selects_the_whole_document_without_comments() -> DsigResult<()> {
        let document =
            build::parse_str("<!-- before --><a><!-- inside -->x</a>").expect("well-formed");
        assert_eq!(octets(&document, "")?, b"<a>x</a>");
        Ok(())
    }

    #[test]
    fn a_shortname_xpointer_selects_a_subtree_without_comments() -> DsigResult<()> {
        let document =
            build::parse_str(r##"<r><o Id="k"><!-- c -->v</o><p/></r>"##).expect("well-formed");
        assert_eq!(octets(&document, "#k")?, br#"<o Id="k">v</o>"#);
        Ok(())
    }

    #[test]
    fn xpointer_keeps_comments() -> DsigResult<()> {
        let document =
            build::parse_str(r##"<r><o Id="k"><!-- c -->v</o></r>"##).expect("well-formed");
        assert_eq!(
            octets(&document, "#xpointer(id('k'))")?,
            br#"<o Id="k"><!-- c -->v</o>"#
        );
        assert_eq!(
            octets(&document, "#xpointer(/)")?,
            br#"<r><o Id="k"><!-- c -->v</o></r>"#
        );
        Ok(())
    }

    #[test]
    fn an_unknown_fragment_is_a_dereference_error() {
        let document = build::parse_str("<r/>").expect("well-formed");
        assert!(matches!(
            octets(&document, "#nope"),
            Err(DsigError::Dereference(_))
        ));
    }

    #[test]
    fn an_ambiguous_fragment_names_the_count_in_the_error() {
        let document = build::parse_str(r##"<r><a Id="x"/><b Id="x"/></r>"##).expect("well-formed");
        match octets(&document, "#x") {
            Err(DsigError::Dereference(message)) => {
                assert!(message.contains('2'), "{message}");
            }
            other => panic!("expected a dereference error, got {other:?}"),
        }
    }

    #[test]
    fn an_external_reference_needs_a_resolver() {
        let document = build::parse_str("<r/>").expect("well-formed");
        assert!(matches!(
            octets(&document, "http://example.com/bar.xml"),
            Err(DsigError::Dereference(_))
        ));

        let ids = IdIndex::build(&document, &names(), false);
        let resolver = StaticResolver::new().with("http://example.com/bar.xml", b"data".to_vec());
        let data = dereference(
            &document,
            &reference(&document, "http://example.com/bar.xml"),
            &ids,
            Some(&resolver),
        )
        .expect("the resolver supplies the data");
        assert_eq!(data.into_octets().expect("octets"), b"data");
    }
}
