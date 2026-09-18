//! Whether the result of a sequence constructor counts as *populated*.
//!
//! [XSLT 3.0 §8.4.1](https://www.w3.org/TR/xslt-30/#where-populated) defines
//! `xsl:where-populated` in terms of a helper predicate: the result of the instruction is
//! `$R[not(deemed-empty(.))]`, where `deemed-empty($item)` is true for
//!
//! * a document or element node with no children — attributes and namespaces do not count, and
//!   a node whose children are themselves empty is *not* deemed empty;
//! * any other node whose string value is zero-length (a whitespace-only text node is not);
//! * an atomic value whose string value is zero-length;
//! * a map of size zero;
//! * an array whose flattening is empty, or every item of which is itself deemed empty.
//!
//! The same predicate decides `xsl:on-empty` and `xsl:on-non-empty`.

use oxixml_infoset::{Item, NodeKind, XdmNode};

/// Whether at least one item of `items` is not *deemed empty* — the condition under which
/// `xsl:where-populated` passes its operand through.
pub(crate) fn is_populated(items: &[Item]) -> bool {
    items.iter().any(|item| !deemed_empty(item))
}

/// The `deemed-empty($item)` predicate of XSLT 3.0 §8.4.1.
pub(crate) fn deemed_empty(item: &Item) -> bool {
    match item {
        Item::Node(node) => match node.node_kind() {
            NodeKind::Document | NodeKind::Element => node.children().is_empty(),
            _ => node.string_value().is_empty(),
        },
        // Only `xs:string`, `xs:anyURI`, `xs:untypedAtomic`, `xs:hexBinary` and
        // `xs:base64Binary` can cast to a zero-length string, but asking the value itself needs
        // no such table.
        Item::Atomic(value) => value.string_value().is_empty(),
        Item::Map(map) => map.is_empty(),
        Item::Array(array) => (0..array.len())
            .filter_map(|index| array.get_at(index))
            .all(|member| member.iter().all(deemed_empty)),
        // A function item is neither a node, an atomic value, a map nor an array, so no clause
        // of the definition deems it empty.
        Item::Function(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::{AtomicValue, Sequence, Tree, XdmArray, XdmMap};

    fn element(xml: &str) -> Item {
        let tree = Tree::parse_str(xml).expect("the fixture parses");
        Item::Node(tree.document_element().expect("a document element"))
    }

    #[test]
    fn a_childless_element_is_deemed_empty_whatever_its_attributes() {
        assert!(deemed_empty(&element("<a/>")));
        assert!(deemed_empty(&element(r#"<a n="1"/>"#)));
        assert!(!deemed_empty(&element("<a><b/></a>")));
    }

    #[test]
    fn an_atomic_value_is_deemed_empty_only_when_its_string_value_is() {
        assert!(deemed_empty(&Item::Atomic(AtomicValue::string(""))));
        assert!(!deemed_empty(&Item::Atomic(AtomicValue::string(" "))));
        assert!(!deemed_empty(&Item::Atomic(AtomicValue::integer(0_i64))));
    }

    /// coco-103's whole point: emptiness reaches inside arrays, recursively.
    #[test]
    fn an_array_is_deemed_empty_when_everything_it_flattens_to_is() {
        let empty = XdmArray::from_members(Vec::<Sequence>::new());
        assert!(deemed_empty(&Item::Array(empty.clone())));
        let nested = XdmArray::from_members(vec![Sequence::from(Item::Array(empty))]);
        assert!(deemed_empty(&Item::Array(nested)));
        let of_empty_string =
            XdmArray::from_members(vec![Sequence::from(Item::Atomic(AtomicValue::string("")))]);
        assert!(deemed_empty(&Item::Array(of_empty_string)));
        let of_42 = XdmArray::from_members(vec![Sequence::from(Item::Atomic(
            AtomicValue::integer(42_i64),
        ))]);
        assert!(!deemed_empty(&Item::Array(of_42)));
        assert!(is_populated(&[Item::Array(XdmArray::from_members(vec![
            Sequence::from(Item::Atomic(AtomicValue::integer(42_i64))),
        ]))]));
    }

    #[test]
    fn an_empty_map_is_deemed_empty() {
        assert!(deemed_empty(&Item::Map(XdmMap::new())));
    }
}
