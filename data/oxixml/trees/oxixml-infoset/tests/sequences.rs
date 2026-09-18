//! The XDM sequence axioms, atomization, and effective boolean value.

mod common;

use common::sample_tree;
use oxixml_infoset::{
    AtomicType, AtomicValue, ErrorCode, Item, Sequence, Tree, XdmArray, XdmError, XdmMap, XdmNode,
};

fn integers(values: &[i64]) -> Sequence {
    values.iter().copied().map(AtomicValue::integer).collect()
}

#[test]
fn a_sequence_cannot_contain_a_sequence() {
    // The axiom is enforced by the type system: `Item` has no `Sequence` variant, so the only
    // way to combine sequences is to concatenate them.
    let left = integers(&[1, 2]);
    let right = integers(&[3]);
    let joined = left.clone().concat(right.clone());

    assert_eq!(joined.len(), 3);
    assert!(joined.iter().all(Item::is_atomic));
    assert_eq!(
        Sequence::from_sequences([left, right, Sequence::empty()]),
        joined,
        "concatenation is associative and the empty sequence is its identity"
    );
}

#[test]
fn there_is_exactly_one_empty_sequence() {
    assert_eq!(Sequence::empty(), Sequence::default());
    assert_eq!(Sequence::empty(), Sequence::from_items([]));
    assert_eq!(Sequence::empty(), integers(&[]));
    assert!(Sequence::empty().is_empty());

    // It is not the same as a sequence containing an empty string, or a zero.
    assert_ne!(Sequence::empty(), Sequence::from(AtomicValue::string("")));
    assert_ne!(
        Sequence::empty(),
        Sequence::from(AtomicValue::integer(0_i64))
    );
}

#[test]
fn a_singleton_sequence_and_its_item_are_interchangeable() {
    let item = Item::from(AtomicValue::integer(7_i64));
    let sequence = Sequence::singleton(item.clone());

    assert!(sequence.is_singleton());
    assert_eq!(Sequence::from(item.clone()), sequence);
    assert_eq!(sequence.clone().into_singleton(), Some(item));
    assert_eq!(integers(&[1, 2]).into_singleton(), None);
}

#[test]
fn order_is_preserved_and_duplicates_are_kept() {
    let sequence = integers(&[3, 1, 3, 2]);
    assert_eq!(
        sequence.iter().map(Item::to_string).collect::<Vec<_>>(),
        ["3", "1", "3", "2"],
        "a sequence is ordered and may contain the same value twice"
    );
}

#[test]
fn node_sequences_normalize_to_document_order_without_duplicates() -> Result<(), XdmError> {
    let tree = sample_tree()?;
    let root = tree.document_element().expect("a document element");
    let children = root.children();

    let mut sequence = Sequence::empty();
    for node in children.iter().rev() {
        sequence.push(node.clone());
    }
    sequence.push(children[0].clone());
    sequence.push(AtomicValue::string("not a node"));

    assert_eq!(
        sequence.nodes_in_document_order(),
        children,
        "sorting drops duplicates and non-node items"
    );
    Ok(())
}

#[test]
fn atomization_flattens_arrays_but_rejects_maps() -> Result<(), XdmError> {
    let nested = XdmArray::from_members([
        integers(&[1, 2]),
        Sequence::singleton(Item::from(XdmArray::from_iter([AtomicValue::integer(
            3_i64,
        )]))),
    ]);
    let atomized = Sequence::singleton(Item::from(nested)).atomize()?;
    assert_eq!(
        atomized.iter().map(Item::to_string).collect::<Vec<_>>(),
        ["1", "2", "3"]
    );

    let with_map = Sequence::singleton(Item::from(XdmMap::new()));
    assert_eq!(
        with_map.atomize().expect_err("maps do not atomize").code(),
        ErrorCode::Foty0013
    );
    Ok(())
}

#[test]
fn atomization_of_a_mixed_sequence_visits_nodes_and_values_in_order() -> Result<(), XdmError> {
    let tree = Tree::parse_str("<a>text</a>")?;
    let sequence = Sequence::from_items([
        Item::from(AtomicValue::integer(1_i64)),
        Item::from(tree.document_element().expect("a document element")),
        Item::from(AtomicValue::string("z")),
    ]);
    assert_eq!(
        sequence
            .atomize()?
            .iter()
            .map(Item::to_string)
            .collect::<Vec<_>>(),
        ["1", "text", "z"]
    );
    Ok(())
}

#[test]
fn effective_boolean_value_covers_every_rule() -> Result<(), XdmError> {
    let tree = Tree::parse_str("<a/>")?;

    // 1. the empty sequence
    assert!(!Sequence::empty().effective_boolean_value()?);
    // 2. a node-headed sequence, however long
    let node_headed = Sequence::from_items([
        Item::from(tree.root()),
        Item::from(AtomicValue::integer(0_i64)),
    ]);
    assert!(node_headed.effective_boolean_value()?);
    // 3. a singleton boolean
    assert!(Sequence::from(AtomicValue::boolean(true)).effective_boolean_value()?);
    assert!(!Sequence::from(AtomicValue::boolean(false)).effective_boolean_value()?);
    // 4. a singleton string-family value
    assert!(Sequence::from(AtomicValue::string("x")).effective_boolean_value()?);
    assert!(!Sequence::from(AtomicValue::any_uri("")).effective_boolean_value()?);
    assert!(Sequence::from(AtomicValue::untyped_atomic("0")).effective_boolean_value()?);
    // 5. a singleton numeric value
    assert!(Sequence::from(AtomicValue::decimal(1_i64)).effective_boolean_value()?);
    assert!(!Sequence::from(AtomicValue::float(0.0_f32)).effective_boolean_value()?);
    assert!(!Sequence::from(AtomicValue::double(f64::NAN)).effective_boolean_value()?);
    // 6. anything else
    for bad in [
        Sequence::from(AtomicValue::parse(AtomicType::Date, "2020-01-01")?),
        integers(&[1, 2]),
        Sequence::singleton(Item::from(XdmArray::new())),
        Sequence::singleton(Item::from(XdmMap::new())),
    ] {
        assert_eq!(
            bad.effective_boolean_value()
                .expect_err("no effective boolean value")
                .code(),
            ErrorCode::Forg0006
        );
    }
    Ok(())
}

#[test]
fn sequences_collect_from_the_three_natural_element_types() -> Result<(), XdmError> {
    let tree = Tree::parse_str("<a><b/><c/></a>")?;
    let children = tree.document_element().expect("root").children();

    let from_nodes: Sequence = children.iter().cloned().collect();
    assert_eq!(from_nodes.len(), 2);
    assert!(from_nodes.iter().all(Item::is_node));

    let from_values: Sequence = [AtomicValue::integer(1_i64)].into_iter().collect();
    assert_eq!(from_values.len(), 1);

    let from_items: Sequence = from_nodes
        .clone()
        .into_iter()
        .chain(from_values.clone())
        .collect();
    assert_eq!(from_items.len(), 3);
    assert_eq!(from_items, from_nodes.concat(from_values));
    Ok(())
}
