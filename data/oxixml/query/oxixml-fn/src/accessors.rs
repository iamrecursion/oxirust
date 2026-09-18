//! The accessor functions of
//! [F&O 3.1 §5](https://www.w3.org/TR/xpath-functions-31/#func-accessors): the six functions
//! that expose an XDM accessor to the query language.
//!
//! Every one of them is a thin, total wrapper over
//! [`XdmNode`] or [`Item`]; the interesting
//! behaviour lives in `oxixml-infoset`. What this module adds is the *function* semantics: the
//! empty-sequence handling and the `FO`-coded errors.

use crate::error::{FnError, FnErrorCode, FnResult};
use oxixml_infoset::{AtomicValue, Item, NodeKind, NodeRef, QNameValue, Sequence, XdmNode};

/// [`fn:node-name`](https://www.w3.org/TR/xpath-functions-31/#func-node-name): the node's name
/// as an `xs:QName`, or the empty sequence for a node that has none.
///
/// # Examples
///
/// ```
/// use oxixml_fn::accessors::node_name;
/// use oxixml_infoset::{Tree, XdmNode};
///
/// let tree = Tree::parse_str("<a:root xmlns:a='urn:a'/>")?;
/// let root = tree.document_element().expect("a document element");
/// let name = node_name(&root).expect("an element has a name");
/// assert_eq!(name.namespace(), Some("urn:a"));
/// assert_eq!(name.local_name(), "root");
///
/// // A document node has no name.
/// assert!(node_name(&tree.root()).is_none());
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
#[must_use]
pub fn node_name(node: &NodeRef) -> Option<QNameValue> {
    node.node_name().map(QNameValue::from)
}

/// [`fn:nilled`](https://www.w3.org/TR/xpath-functions-31/#func-nilled): whether an element node
/// is *nilled*, or the empty sequence for every other node kind.
#[must_use]
pub fn nilled(node: &NodeRef) -> Option<bool> {
    if node.node_kind() == NodeKind::Element {
        Some(node.nilled().unwrap_or(false))
    } else {
        None
    }
}

/// [`fn:string`](https://www.w3.org/TR/xpath-functions-31/#func-string): the string value of an
/// item; the empty sequence yields the zero-length string.
///
/// # Errors
///
/// [`FnErrorCode::Foty0014`] when the item is a map or an array, which have no string value.
///
/// # Examples
///
/// ```
/// use oxixml_fn::accessors::string;
/// use oxixml_infoset::{AtomicValue, Item, XdmMap};
///
/// assert_eq!(string(None)?, "");
/// assert_eq!(string(Some(&Item::from(AtomicValue::integer(23_i64))))?, "23");
/// assert!(string(Some(&Item::from(XdmMap::new()))).is_err());
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn string(item: Option<&Item>) -> FnResult<String> {
    match item {
        None => Ok(String::new()),
        Some(item) => Ok(item.string_value()?),
    }
}

/// [`fn:string`](https://www.w3.org/TR/xpath-functions-31/#func-string) applied to a sequence,
/// which must be empty or a singleton.
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] for a sequence of more than one item, [`FnErrorCode::Foty0014`] for
/// a function item.
pub fn string_of(sequence: &Sequence) -> FnResult<String> {
    match sequence.as_slice() {
        [] => Ok(String::new()),
        [item] => string(Some(item)),
        _ => Err(FnError::new(
            FnErrorCode::Xpty0004,
            "fn:string expects at most one item",
        )),
    }
}

/// [`fn:data`](https://www.w3.org/TR/xpath-functions-31/#func-data): the atomized value of a
/// sequence.
///
/// # Errors
///
/// [`FnErrorCode::Foty0013`] when the sequence contains a map, [`FnErrorCode::Foty0012`] when it
/// contains a node with no typed value.
///
/// # Examples
///
/// ```
/// use oxixml_fn::accessors::data;
/// use oxixml_infoset::{Sequence, Tree, XdmNode};
///
/// let tree = Tree::parse_str("<a>text</a>")?;
/// let atomized = data(&Sequence::from(tree.document_element().expect("root")))?;
/// assert_eq!(atomized.len(), 1);
/// assert_eq!(atomized.first().map(ToString::to_string).as_deref(), Some("text"));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn data(sequence: &Sequence) -> FnResult<Sequence> {
    Ok(sequence.atomize()?)
}

/// [`fn:base-uri`](https://www.w3.org/TR/xpath-functions-31/#func-base-uri): the node's base URI
/// as an `xs:anyURI`, or the empty sequence when it has none.
#[must_use]
pub fn base_uri(node: &NodeRef) -> Option<AtomicValue> {
    node.base_uri().map(AtomicValue::any_uri)
}

/// [`fn:document-uri`](https://www.w3.org/TR/xpath-functions-31/#func-document-uri): the absolute
/// URI of the document node, or the empty sequence for any other node or an unnamed document.
#[must_use]
pub fn document_uri(node: &NodeRef) -> Option<AtomicValue> {
    if node.node_kind() != NodeKind::Document {
        return None;
    }
    node.document_uri().map(AtomicValue::any_uri)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::{Tree, XdmMap};

    #[test]
    fn node_name_reports_expanded_names() -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str("<a:root xmlns:a='urn:a'>text</a:root>")?;
        let root = tree.document_element().expect("a document element");
        let name = node_name(&root).expect("an element has a name");
        assert_eq!(name.namespace(), Some("urn:a"));
        assert_eq!(name.local_name(), "root");
        assert!(node_name(&tree.root()).is_none());
        Ok(())
    }

    #[test]
    fn nilled_is_defined_only_for_elements() -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str("<root/>")?;
        assert_eq!(nilled(&tree.document_element().expect("root")), Some(false));
        assert_eq!(nilled(&tree.root()), None);
        Ok(())
    }

    #[test]
    fn string_and_data_agree_with_the_data_model() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(string(None)?, "");
        assert_eq!(
            string(Some(&Item::from(AtomicValue::integer(23_i64))))?,
            "23"
        );
        let error = string(Some(&Item::from(XdmMap::new()))).expect_err("a map has no string");
        assert!(error.is(FnErrorCode::Foty0014));

        let tree = Tree::parse_str("<a>one<b>two</b></a>")?;
        let root = tree.document_element().expect("root");
        assert_eq!(string(Some(&Item::from(root.clone())))?, "onetwo");
        assert_eq!(data(&Sequence::from(root))?.len(), 1);
        Ok(())
    }

    #[test]
    fn document_uri_is_reported_only_for_the_document_node()
    -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str("<a/>")?;
        assert_eq!(document_uri(&tree.document_element().expect("root")), None);
        Ok(())
    }

    #[test]
    fn string_of_rejects_longer_sequences() {
        let two = Sequence::from_items([
            Item::from(AtomicValue::integer(1_i64)),
            Item::from(AtomicValue::integer(2_i64)),
        ]);
        assert!(string_of(&two).is_err());
    }
}
