//! Maps and arrays: the two XDM 3.1 function items, as first-class sequence members.

use oxixml_infoset::{
    AtomicValue, DuplicateKeys, ErrorCode, Item, Sequence, Tree, XdmArray, XdmError, XdmMap,
};

fn integers(values: &[i64]) -> Sequence {
    values.iter().copied().map(AtomicValue::integer).collect()
}

fn rendered(sequence: &Sequence) -> Vec<String> {
    sequence.iter().map(Item::to_string).collect()
}

#[test]
fn an_array_holds_sequences_while_a_sequence_holds_items() {
    // `[(1, 2), 3]` has two members; `(1, 2, 3)` has three items. That difference is the whole
    // reason arrays exist alongside sequences.
    let array = XdmArray::from_members([integers(&[1, 2]), integers(&[3])]);
    assert_eq!(array.len(), 2);
    assert_eq!(array.flatten().len(), 3);
    assert_eq!(integers(&[1, 2, 3]).len(), 3);

    // An array is an item, so it can sit inside a sequence.
    let sequence = Sequence::singleton(Item::from(array));
    assert_eq!(sequence.len(), 1);
    assert!(sequence.iter().all(Item::is_function));
}

#[test]
fn array_accessors_are_one_based_and_bounds_checked() -> Result<(), XdmError> {
    let array = XdmArray::from_iter([
        AtomicValue::integer(10_i64),
        AtomicValue::integer(20_i64),
        AtomicValue::integer(30_i64),
    ]);

    assert_eq!(rendered(array.get(1)?), ["10"]);
    assert_eq!(rendered(array.get(3)?), ["30"]);
    assert_eq!(rendered(array.head()?), ["10"]);
    assert_eq!(array.tail()?.len(), 2);

    for bad in [0, 4, usize::MAX] {
        assert_eq!(
            array.get(bad).expect_err("out of bounds").code(),
            ErrorCode::Foay0001
        );
    }
    assert!(XdmArray::new().head().is_err());
    assert!(XdmArray::new().tail().is_err());
    Ok(())
}

#[test]
fn array_mutators_are_persistent() -> Result<(), XdmError> {
    let original = XdmArray::from_iter([AtomicValue::integer(1_i64), AtomicValue::integer(3_i64)]);

    let inserted = original.clone().insert_before(2, integers(&[2]))?;
    let appended = original.clone().append(integers(&[9]));
    let replaced = original.clone().put(1, integers(&[0]))?;
    let removed = original.clone().remove(1)?;
    let reversed = original.clone().reverse();
    let joined = XdmArray::join([original.clone(), XdmArray::new(), inserted.clone()]);
    let sliced = inserted.subarray(2, 2)?;

    assert_eq!(original.len(), 2, "the original is untouched");
    assert_eq!(inserted.len(), 3);
    assert_eq!(appended.len(), 3);
    assert_eq!(rendered(replaced.get(1)?), ["0"]);
    assert_eq!(rendered(removed.get(1)?), ["3"]);
    assert_eq!(rendered(reversed.get(1)?), ["3"]);
    assert_eq!(joined.len(), 5);
    assert_eq!(sliced.len(), 2);
    assert_eq!(rendered(sliced.get(1)?), ["2"]);

    assert!(
        original
            .clone()
            .insert_before(0, Sequence::empty())
            .is_err()
    );
    assert!(
        original
            .clone()
            .insert_before(4, Sequence::empty())
            .is_err()
    );
    assert!(original.clone().put(3, Sequence::empty()).is_err());
    assert!(original.subarray(2, 2).is_err());
    Ok(())
}

#[test]
fn map_keys_use_the_same_key_relation() {
    let mut map = XdmMap::new();
    map.insert(
        AtomicValue::integer(1_i64),
        Sequence::from(AtomicValue::string("one")),
    );
    map.insert(
        AtomicValue::string("k"),
        Sequence::from(AtomicValue::string("text")),
    );
    map.insert(
        AtomicValue::double(f64::NAN),
        Sequence::from(AtomicValue::string("nan")),
    );

    // Cross-type numeric keys unify.
    for lookup in [
        AtomicValue::decimal(1_i64),
        AtomicValue::double(1.0),
        AtomicValue::float(1.0_f32),
    ] {
        assert!(map.contains_key(&lookup), "1 should be found as {lookup}");
    }
    // Text-family keys unify.
    assert!(map.contains_key(&AtomicValue::any_uri("k")));
    assert!(map.contains_key(&AtomicValue::untyped_atomic("k")));
    // NaN is a key.
    assert!(map.contains_key(&AtomicValue::double(f64::NAN)));
    // Unrelated types are different keys, never an error.
    assert!(!map.contains_key(&AtomicValue::string("1")));
    assert!(!map.contains_key(&AtomicValue::boolean(true)));
    assert_eq!(map.len(), 3);
}

#[test]
fn map_mutation_and_merge() -> Result<(), XdmError> {
    let base = XdmMap::from_iter([
        (AtomicValue::string("a"), integers(&[1])),
        (AtomicValue::string("b"), integers(&[2])),
    ]);

    let put = base.clone().put(AtomicValue::string("c"), integers(&[3]));
    assert_eq!(base.len(), 2, "put is persistent");
    assert_eq!(put.len(), 3);

    let mut removed = base.clone();
    assert_eq!(
        removed
            .remove(&AtomicValue::string("a"))
            .map(|value| rendered(&value)),
        Some(vec!["1".to_owned()])
    );
    assert_eq!(removed.len(), 1);
    assert!(removed.remove(&AtomicValue::string("a")).is_none());

    let overlay = XdmMap::from_iter([(AtomicValue::string("b"), integers(&[20]))]);
    assert_eq!(
        XdmMap::merge([base.clone(), overlay.clone()], DuplicateKeys::UseFirst)?
            .get(&AtomicValue::string("b"))
            .map(rendered),
        Some(vec!["2".to_owned()])
    );
    assert_eq!(
        XdmMap::merge([base.clone(), overlay.clone()], DuplicateKeys::UseLast)?
            .get(&AtomicValue::string("b"))
            .map(rendered),
        Some(vec!["20".to_owned()])
    );
    assert_eq!(
        XdmMap::merge([base.clone(), overlay.clone()], DuplicateKeys::Combine)?
            .get(&AtomicValue::string("b"))
            .map(rendered),
        Some(vec!["2".to_owned(), "20".to_owned()])
    );
    assert_eq!(
        XdmMap::merge([base, overlay], DuplicateKeys::Reject)
            .expect_err("duplicate keys")
            .code(),
        ErrorCode::Xqdy0137
    );
    Ok(())
}

#[test]
fn maps_and_arrays_can_hold_nodes_and_each_other() -> Result<(), XdmError> {
    let tree = Tree::parse_str("<a><b/></a>")?;
    let root = tree.document_element().expect("a document element");

    let inner = XdmArray::from_members([Sequence::from(root.clone())]);
    let map = XdmMap::from_iter([
        (AtomicValue::string("nodes"), Sequence::from(root.clone())),
        (
            AtomicValue::string("array"),
            Sequence::singleton(Item::from(inner)),
        ),
    ]);
    let outer = XdmArray::from_members([Sequence::singleton(Item::from(map.clone()))]);

    assert_eq!(outer.len(), 1);
    assert_eq!(
        map.get(&AtomicValue::string("nodes"))
            .and_then(Sequence::first)
            .and_then(Item::as_node),
        Some(&root)
    );

    // Atomizing an array of nodes reaches through to their typed values.
    let nodes_only = XdmArray::from_members([Sequence::from(root)]);
    assert_eq!(
        Item::from(nodes_only).atomize()?.len(),
        1,
        "the element atomizes to one xs:untypedAtomic value"
    );

    // Atomizing anything containing a map is an error.
    assert_eq!(
        Item::from(outer)
            .atomize()
            .expect_err("contains a map")
            .code(),
        ErrorCode::Foty0013
    );
    Ok(())
}

#[test]
fn map_equality_ignores_insertion_order_and_array_equality_does_not() {
    let first = XdmMap::from_iter([
        (AtomicValue::string("x"), integers(&[1])),
        (AtomicValue::string("y"), integers(&[2])),
    ]);
    let second = XdmMap::from_iter([
        (AtomicValue::string("y"), integers(&[2])),
        (AtomicValue::string("x"), integers(&[1])),
    ]);
    assert_eq!(first, second);

    let one = XdmArray::from_members([integers(&[1]), integers(&[2])]);
    let other = XdmArray::from_members([integers(&[2]), integers(&[1])]);
    assert_ne!(one, other, "an array is ordered");
}

#[test]
fn function_items_have_no_string_value() {
    for item in [Item::from(XdmMap::new()), Item::from(XdmArray::new())] {
        assert_eq!(
            item.string_value().expect_err("no string value").code(),
            ErrorCode::Foty0014
        );
        assert!(item.is_function());
        assert!(!item.is_atomic() && !item.is_node());
    }
}
