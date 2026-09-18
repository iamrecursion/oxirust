//! [`deep_equal`] — the `fn:deep-equal` relation of
//! [XPath and XQuery Functions and Operators 3.1 §15.3.1](https://www.w3.org/TR/xpath-functions-31/#func-deep-equal).

use crate::atomic::{AtomicValue, atomic_equal_with_collation, compare_strings};
use crate::error::{ErrorCode, XdmError};
use crate::item::Item;
use crate::map::XdmMap;
use crate::node::{NodeKind, NodeRef, XdmNode};
use crate::sequence::Sequence;
use std::cmp::Ordering;

/// `fn:deep-equal` on two sequences, under the default (codepoint) collation.
///
/// See [`deep_equal_with_collation`] for the rules.
///
/// # Errors
///
/// Propagates whatever [`XdmNode::typed_value`] raises while atomizing an attribute.
pub fn deep_equal(left: &Sequence, right: &Sequence) -> Result<bool, XdmError> {
    deep_equal_with_collation(left, right, None)
}

/// `fn:deep-equal` on two sequences.
///
/// Two sequences are deep-equal when they have the same length and their items are pairwise
/// deep-equal. Two items are deep-equal according to their kinds:
///
/// * **two atomic values** — when `eq` says they are equal, or when both are `NaN`. Values whose
///   types are not comparable are simply *not* deep-equal; unlike `eq`, this function never
///   raises a type error for them. `xs:untypedAtomic` compares as `xs:string`.
/// * **two elements** — when their names match, their attributes match as *sets* (each
///   attribute of one having a same-named, deep-equal counterpart in the other), their `nilled`
///   properties match, and their children — **ignoring comments and processing instructions** —
///   are pairwise deep-equal. Namespace nodes, base URIs, and type annotations are not
///   compared.
/// * **two documents** — like elements, but with no name and no attributes.
/// * **two attributes** — when their names match and their typed values are deep-equal.
/// * **two processing instructions** — when their targets and their contents match.
/// * **two texts or two comments** — when their string values match under the collation.
/// * **two namespace nodes** — when their prefixes match and their URIs match under the
///   collation.
/// * **two arrays** — when they have the same size and their members are pairwise deep-equal.
/// * **two maps** — when they have the same size and every key of one is a key of the other
///   (in the [`same_key`](crate::same_key) sense) bound to a deep-equal value.
/// * **items of different kinds** — never.
///
/// # Errors
///
/// Propagates whatever [`XdmNode::typed_value`] raises while atomizing an attribute — for a
/// schema-aware store, [`ErrorCode::Foty0012`](crate::ErrorCode::Foty0012).
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{deep_equal, Sequence, Tree, XdmNode};
///
/// // Comments and processing instructions are ignored inside elements...
/// let plain = Tree::parse_str("<a><b>x</b></a>")?;
/// let noisy = Tree::parse_str("<a><!--note--><b>x</b><?pi data?></a>")?;
/// assert!(deep_equal(
///     &Sequence::from(plain.document_element().expect("root")),
///     &Sequence::from(noisy.document_element().expect("root")),
/// )?);
///
/// // ...but attribute order is not significant, and attribute values are.
/// let one = Tree::parse_str(r#"<a p="1" q="2"/>"#)?;
/// let two = Tree::parse_str(r#"<a q="2" p="1"/>"#)?;
/// let three = Tree::parse_str(r#"<a q="2" p="9"/>"#)?;
/// assert!(deep_equal(
///     &Sequence::from(one.document_element().expect("root")),
///     &Sequence::from(two.document_element().expect("root")),
/// )?);
/// assert!(!deep_equal(
///     &Sequence::from(one.document_element().expect("root")),
///     &Sequence::from(three.document_element().expect("root")),
/// )?);
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
pub fn deep_equal_with_collation(
    left: &Sequence,
    right: &Sequence,
    collation: Option<&str>,
) -> Result<bool, XdmError> {
    if left.len() != right.len() {
        return Ok(false);
    }
    for (a, b) in left.iter().zip(right.iter()) {
        if !deep_equal_items(a, b, collation)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// `fn:deep-equal` on two single items.
///
/// # Errors
///
/// Propagates whatever [`XdmNode::typed_value`] raises while atomizing an attribute.
pub fn deep_equal_items(
    left: &Item,
    right: &Item,
    collation: Option<&str>,
) -> Result<bool, XdmError> {
    match (left, right) {
        (Item::Atomic(a), Item::Atomic(b)) => Ok(deep_equal_atomic(a, b, collation)),
        (Item::Node(a), Item::Node(b)) => deep_equal_nodes(a, b, collation),
        (Item::Array(a), Item::Array(b)) => {
            if a.len() != b.len() {
                return Ok(false);
            }
            for (x, y) in a.members().zip(b.members()) {
                if !deep_equal_with_collation(x, y, collation)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        (Item::Map(a), Item::Map(b)) => deep_equal_maps(a, b, collation),
        // F&O 3.1 §14.1.4: a function item that is neither a map nor an array has no
        // equality to test, so `fn:deep-equal` raises rather than answering `false`.
        (Item::Function(_), _) | (_, Item::Function(_)) => Err(XdmError::new(
            ErrorCode::Foty0015,
            "fn:deep-equal cannot compare a function item",
        )),
        _ => Ok(false),
    }
}

fn deep_equal_atomic(left: &AtomicValue, right: &AtomicValue, collation: Option<&str>) -> bool {
    if left.is_nan() && right.is_nan() {
        return true;
    }
    // A type error from `eq` means "not comparable", which deep-equal reports as `false`
    // rather than propagating.
    atomic_equal_with_collation(left, right, collation).unwrap_or(false)
}

fn deep_equal_maps(
    left: &XdmMap,
    right: &XdmMap,
    collation: Option<&str>,
) -> Result<bool, XdmError> {
    if left.len() != right.len() {
        return Ok(false);
    }
    for (key, value) in left.entries() {
        let Some(other) = right.get(key) else {
            return Ok(false);
        };
        if !deep_equal_with_collation(value, other, collation)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether a child participates in an element's or document's content comparison: comments and
/// processing instructions are skipped, per the specification.
fn is_significant_child(node: &NodeRef) -> bool {
    !matches!(
        node.node_kind(),
        NodeKind::Comment | NodeKind::ProcessingInstruction
    )
}

fn deep_equal_nodes(
    left: &NodeRef,
    right: &NodeRef,
    collation: Option<&str>,
) -> Result<bool, XdmError> {
    let kind = left.node_kind();
    if kind != right.node_kind() {
        return Ok(false);
    }
    match kind {
        NodeKind::Document => deep_equal_content(left, right, collation),
        NodeKind::Element => {
            if left.node_name() != right.node_name() || left.nilled() != right.nilled() {
                return Ok(false);
            }
            if !deep_equal_attributes(left, right, collation)? {
                return Ok(false);
            }
            deep_equal_content(left, right, collation)
        }
        NodeKind::Attribute => {
            if left.node_name() != right.node_name() {
                return Ok(false);
            }
            deep_equal_with_collation(&left.typed_value()?, &right.typed_value()?, collation)
        }
        NodeKind::ProcessingInstruction | NodeKind::Namespace => {
            if left.node_name() != right.node_name() {
                return Ok(false);
            }
            same_string(&left.string_value(), &right.string_value(), collation)
        }
        NodeKind::Text | NodeKind::Comment => {
            same_string(&left.string_value(), &right.string_value(), collation)
        }
    }
}

/// Compares two string values under `collation`, which is what the specification's
/// "have the same string value" clauses mean.
fn same_string(left: &str, right: &str, collation: Option<&str>) -> Result<bool, XdmError> {
    Ok(compare_strings(left, right, collation)? == Ordering::Equal)
}

fn deep_equal_attributes(
    left: &NodeRef,
    right: &NodeRef,
    collation: Option<&str>,
) -> Result<bool, XdmError> {
    let left_attributes = left.attributes();
    let right_attributes = right.attributes();
    if left_attributes.len() != right_attributes.len() {
        return Ok(false);
    }
    for attribute in &left_attributes {
        let name = attribute.node_name();
        let Some(counterpart) = right_attributes
            .iter()
            .find(|candidate| candidate.node_name() == name)
        else {
            return Ok(false);
        };
        if !deep_equal_with_collation(
            &attribute.typed_value()?,
            &counterpart.typed_value()?,
            collation,
        )? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn deep_equal_content(
    left: &NodeRef,
    right: &NodeRef,
    collation: Option<&str>,
) -> Result<bool, XdmError> {
    let left_children: Vec<NodeRef> = left
        .children()
        .into_iter()
        .filter(is_significant_child)
        .collect();
    let right_children: Vec<NodeRef> = right
        .children()
        .into_iter()
        .filter(is_significant_child)
        .collect();
    if left_children.len() != right_children.len() {
        return Ok(false);
    }
    for (a, b) in left_children.iter().zip(right_children.iter()) {
        if !deep_equal_nodes(a, b, collation)? {
            return Ok(false);
        }
    }
    Ok(true)
}
