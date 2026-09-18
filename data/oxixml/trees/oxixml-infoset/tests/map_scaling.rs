//! [`XdmMap`] has to stay usable when a query derives one map from another once per key.
//!
//! The shape below is the one W3C QT3 case `op-same-key/same-key-023` evaluates: build a map
//! with `map:merge`, then, for every key, look it up, `map:remove` it and `map:put` it back.
//! Every one of those returns a *new* map, so a representation that copies its entries makes
//! the whole loop quadratic; the case uses 421 875 keys, which a quadratic map cannot finish
//! in any budget. These tests pin the cost with generous wall-clock bounds, and check the
//! clock as they go so that a regression fails in seconds rather than at the bound.

use oxixml_infoset::{AtomicValue, DuplicateKeys, Sequence, XdmMap};
use std::time::{Duration, Instant};

/// Enough keys that a quadratic implementation cannot pass: it needs hours here, where the
/// persistent trie needs a second or two even in an unoptimized build.
const KEYS: usize = 100_000;

/// Wall-clock budget for the churn loop. The measured cost of the trie is ~1.3 s in a debug
/// build, so this leaves more than an order of magnitude of headroom for a loaded machine
/// while still being unreachable for a per-operation copy of the whole map.
const CHURN_BUDGET: Duration = Duration::from_secs(60);

/// How often the churn loop looks at the clock, so that a regression fails fast.
const CHECK_EVERY: usize = 2_000;

fn keys(count: usize) -> Vec<AtomicValue> {
    (0..count)
        .map(|index| AtomicValue::string(format!("key{index}")))
        .collect()
}

fn merged(keys: &[AtomicValue]) -> XdmMap {
    XdmMap::merge(
        keys.iter()
            .map(|key| XdmMap::from_iter([(key.clone(), Sequence::from(key.clone()))])),
        DuplicateKeys::UseFirst,
    )
    .expect("map:merge over distinct keys never rejects")
}

#[test]
fn deriving_a_map_once_per_key_is_not_quadratic() {
    let keys = keys(KEYS);
    let started = Instant::now();
    let map = merged(&keys);
    assert_eq!(map.len(), KEYS);
    println!(
        "built {KEYS} entries in {:.3}s",
        started.elapsed().as_secs_f64()
    );

    for (index, key) in keys.iter().enumerate() {
        assert!(map.get(key).is_some(), "the key must still be bound");

        let mut without = map.clone();
        assert!(without.remove(key).is_some());
        assert!(!without.contains_key(key));

        let replaced = map
            .clone()
            .put(key.clone(), Sequence::from(AtomicValue::string("x")));
        assert_eq!(
            replaced
                .get(key)
                .and_then(Sequence::first)
                .map(ToString::to_string),
            Some("x".to_owned())
        );

        if index % CHECK_EVERY == 0 {
            let elapsed = started.elapsed();
            assert!(
                elapsed < CHURN_BUDGET,
                "{index} of {KEYS} keys took {elapsed:?}, past the {CHURN_BUDGET:?} budget: \
                 deriving a map is no longer O(log n)"
            );
        }
    }

    let elapsed = started.elapsed();
    println!("churned {KEYS} keys in {:.3}s", elapsed.as_secs_f64());
    assert!(elapsed < CHURN_BUDGET, "the churn loop took {elapsed:?}");
    // Every derivation left the original alone.
    assert_eq!(map.len(), KEYS);
}

#[test]
fn cloning_a_map_does_not_copy_its_entries() {
    // A map travels inside `Item`, which is cloned constantly by any evaluator; the clone has
    // to be a reference-count bump rather than a copy of the entries.
    let keys = keys(KEYS);
    let map = merged(&keys);

    let started = Instant::now();
    let mut kept = Vec::with_capacity(1_000);
    for _ in 0..1_000 {
        kept.push(map.clone());
    }
    let elapsed = started.elapsed();
    println!(
        "1000 clones of a {KEYS}-entry map in {:.3}s",
        elapsed.as_secs_f64()
    );
    assert_eq!(kept.len(), 1_000);
    assert!(
        elapsed < Duration::from_secs(5),
        "1000 clones of a {KEYS}-entry map took {elapsed:?}: cloning is copying the entries"
    );
    assert!(kept.iter().all(|clone| clone.len() == KEYS));
}

#[test]
fn insertion_order_survives_a_large_map() {
    let count = 5_000_usize;
    let keys = keys(count);
    let mut map = merged(&keys);

    // Replacing a binding keeps the key's position; removing and re-adding moves it to the end.
    map.insert(
        keys[0].clone(),
        Sequence::from(AtomicValue::string("replaced")),
    );
    map.remove(&keys[1]);
    map.insert(keys[1].clone(), Sequence::empty());

    let mut expected: Vec<String> = keys.iter().map(AtomicValue::string_value).collect();
    let moved = expected.remove(1);
    expected.push(moved);

    let actual: Vec<String> = map.keys().map(AtomicValue::string_value).collect();
    assert_eq!(actual, expected);
    assert_eq!(map.len(), count);
}
