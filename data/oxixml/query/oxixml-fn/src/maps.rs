//! The `map:` functions of
//! [F&O 3.1 §17.1](https://www.w3.org/TR/xpath-functions-31/#map-functions), which live in the
//! namespace `http://www.w3.org/2005/xpath-functions/map`.
//!
//! Key identity is `op:same-key`, which [`oxixml_infoset::XdmMap`] implements: `xs:integer 1`,
//! `xs:decimal 1.0` and `xs:double 1.0E0` are one key, `NaN` is a usable key, and values of
//! unrelated types are simply different keys rather than an error.

use crate::context::FnContext;
use crate::error::{FnError, FnErrorCode, FnResult};
use crate::function::FunctionItem;
use oxixml_infoset::{AtomicValue, DuplicateKeys, Item, Sequence, XdmArray, XdmMap, same_key};

/// The namespace name of the `map:` functions.
pub const MAP_NAMESPACE: &str = "http://www.w3.org/2005/xpath-functions/map";

/// [`map:size`](https://www.w3.org/TR/xpath-functions-31/#func-map-size).
#[must_use]
pub fn size(map: &XdmMap) -> i64 {
    i64::try_from(map.len()).unwrap_or(i64::MAX)
}

/// [`map:keys`](https://www.w3.org/TR/xpath-functions-31/#func-map-keys), in insertion order —
/// the specification leaves the order implementation-dependent.
#[must_use]
pub fn keys(map: &XdmMap) -> Sequence {
    Sequence::from_items(map.keys().cloned().map(Item::Atomic))
}

/// [`map:contains`](https://www.w3.org/TR/xpath-functions-31/#func-map-contains).
#[must_use]
pub fn contains(map: &XdmMap, key: &AtomicValue) -> bool {
    map.contains_key(key)
}

/// [`map:get`](https://www.w3.org/TR/xpath-functions-31/#func-map-get), which answers the empty
/// sequence for an absent key — indistinguishable, by design, from a key bound to `()`.
#[must_use]
pub fn get(map: &XdmMap, key: &AtomicValue) -> Sequence {
    map.get(key).cloned().unwrap_or_default()
}

/// [`map:put`](https://www.w3.org/TR/xpath-functions-31/#func-map-put): a new map with `key`
/// bound to `value`.
#[must_use]
pub fn put(map: &XdmMap, key: AtomicValue, value: Sequence) -> XdmMap {
    map.clone().put(key, value)
}

/// [`map:remove`](https://www.w3.org/TR/xpath-functions-31/#func-map-remove): a new map without
/// the given keys; keys that are absent are ignored.
#[must_use]
pub fn remove(map: &XdmMap, keys: &[AtomicValue]) -> XdmMap {
    let mut result = map.clone();
    for key in keys {
        result.remove(key);
    }
    result
}

/// [`map:entry`](https://www.w3.org/TR/xpath-functions-31/#func-map-entry): the singleton map.
#[must_use]
pub fn entry(key: AtomicValue, value: Sequence) -> XdmMap {
    XdmMap::new().put(key, value)
}

/// How `map:merge`'s `duplicates` option is spelled in the options map.
///
/// # Errors
///
/// [`FnErrorCode::Forg0006`] for an unrecognized option value.
pub fn duplicates_from_name(name: &str) -> FnResult<DuplicateKeys> {
    match name {
        "use-first" => Ok(DuplicateKeys::UseFirst),
        "use-last" => Ok(DuplicateKeys::UseLast),
        "use-any" => Ok(DuplicateKeys::UseAny),
        "combine" => Ok(DuplicateKeys::Combine),
        "reject" => Ok(DuplicateKeys::Reject),
        other => Err(FnError::new(
            FnErrorCode::Forg0006,
            format!("`{other}` is not a value of the map:merge `duplicates` option"),
        )),
    }
}

/// [`map:merge`](https://www.w3.org/TR/xpath-functions-31/#func-map-merge).
///
/// # Errors
///
/// [`FnErrorCode::Fojs0003`] when `duplicates` is [`DuplicateKeys::Reject`] and two maps bind
/// the same key. (The data model reports that clash as `XQDY0137`, the code the *map
/// constructor* of XQuery raises; F&O 3.1 §17.1.5 gives the function its own code.)
///
/// # Examples
///
/// ```
/// use oxixml_fn::maps::{entry, merge};
/// use oxixml_infoset::{AtomicValue, DuplicateKeys, Sequence, XdmMap};
///
/// let first = entry(AtomicValue::integer(0_i64), Sequence::from(AtomicValue::string("no")));
/// let second = entry(AtomicValue::integer(1_i64), Sequence::from(AtomicValue::string("yes")));
/// assert_eq!(merge(&[first.clone(), second.clone()], DuplicateKeys::UseFirst)?.len(), 2);
///
/// let clash = entry(AtomicValue::integer(0_i64), Sequence::from(AtomicValue::string("nope")));
/// let kept = merge(&[first.clone(), clash.clone()], DuplicateKeys::UseFirst)?;
/// assert_eq!(
///     kept.get(&AtomicValue::integer(0_i64)).and_then(Sequence::first).map(ToString::to_string),
///     Some("no".to_owned())
/// );
/// assert!(merge(&[first, clash], DuplicateKeys::Reject).is_err());
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn merge(maps: &[XdmMap], duplicates: DuplicateKeys) -> FnResult<XdmMap> {
    XdmMap::merge(maps.iter().cloned(), duplicates).map_err(|error| {
        let converted = FnError::from(error);
        if converted.is(FnErrorCode::Xqdy0137) {
            return FnError::new(FnErrorCode::Fojs0003, converted.description().to_owned());
        }
        converted
    })
}

/// [`map:for-each`](https://www.w3.org/TR/xpath-functions-31/#func-map-for-each): applies a
/// two-argument function to every `(key, value)` pair, concatenating the results.
///
/// # Errors
///
/// Whatever `action` raises, or [`FnErrorCode::Xpty0004`] when its arity is not two.
///
/// # Examples
///
/// ```
/// use oxixml_fn::{maps::{entry, for_each}, FnContext, FunctionItem};
/// use oxixml_infoset::{AtomicValue, Sequence};
///
/// let context = FnContext::new();
/// let map = entry(AtomicValue::string("a"), Sequence::from(AtomicValue::integer(1_i64)));
/// let just_the_key = FunctionItem::anonymous(2, |_, arguments| Ok(arguments[0].clone()));
/// assert_eq!(for_each(&context, &map, &just_the_key)?.len(), 1);
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn for_each(context: &FnContext, map: &XdmMap, action: &FunctionItem) -> FnResult<Sequence> {
    if action.arity() != 2 {
        return Err(FnError::new(
            FnErrorCode::Xpty0004,
            "map:for-each expects a function of arity 2",
        ));
    }
    let mut out = Sequence::empty();
    for (key, value) in map.entries() {
        out.append(action.call(context, &[Sequence::from(key.clone()), value.clone()])?);
    }
    Ok(out)
}

/// [`map:find`](https://www.w3.org/TR/xpath-functions-31/#func-map-find): searches `input`
/// recursively, through arrays and maps, and returns an array of every value bound to `key`.
///
/// # Examples
///
/// ```
/// use oxixml_fn::maps::{entry, find};
/// use oxixml_infoset::{AtomicValue, Item, Sequence, XdmArray};
///
/// let inner = entry(AtomicValue::integer(1_i64), Sequence::from(AtomicValue::string("deep")));
/// let outer = entry(
///     AtomicValue::integer(0_i64),
///     Sequence::singleton(Item::from(inner)),
/// );
/// let found = find(
///     &Sequence::singleton(Item::from(outer)),
///     &AtomicValue::integer(1_i64),
/// );
/// assert_eq!(found.len(), 1);
/// ```
#[must_use]
pub fn find(input: &Sequence, key: &AtomicValue) -> XdmArray {
    let mut found = Vec::new();
    collect_matches(input, key, &mut found);
    XdmArray::from_members(found)
}

/// One step still owed by the search: a sequence to look through, or a value already found to
/// be under `key` and waiting to be reported in its turn.
enum Search {
    Look(Sequence),
    Report(Sequence),
}

/// Collects, in pre-order, every value the input holds under `key`.
///
/// The walk keeps its own stack on the heap rather than recursing. The structure it descends is
/// the caller's, and nothing bounds how deeply maps and arrays may be nested inside one another
/// — `fold-left(1 to 100000, [], function($a, $b) { [$a] })` builds a hundred thousand levels
/// without any recursion of its own — so a native recursion here would exhaust the thread's
/// stack, which is a process abort rather than an error a host can catch.
fn collect_matches(input: &Sequence, key: &AtomicValue, found: &mut Vec<Sequence>) {
    let mut pending = vec![Search::Look(input.clone())];
    while let Some(step) = pending.pop() {
        let sequence = match step {
            Search::Report(value) => {
                found.push(value);
                continue;
            }
            Search::Look(sequence) => sequence,
        };
        // Everything this sequence owes is collected in order and then pushed reversed, so that
        // popping walks it left to right and finishes each subtree before starting the next.
        let mut owed = Vec::new();
        for item in sequence.iter() {
            match item {
                Item::Map(map) => {
                    for (candidate, value) in map.entries() {
                        if same_key(candidate, key) {
                            owed.push(Search::Report(value.clone()));
                        }
                        owed.push(Search::Look(value.clone()));
                    }
                }
                Item::Array(array) => {
                    for member in array.members() {
                        owed.push(Search::Look(member.clone()));
                    }
                }
                // A node, an atomic value or an opaque function item holds no map entries to
                // search: `fn:map:find` descends only through maps and arrays.
                Item::Node(_) | Item::Atomic(_) | Item::Function(_) => {}
            }
        }
        owed.reverse();
        pending.append(&mut owed);
    }
}

/// Extracts the map an argument declared as `map(*)` must be.
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] when the sequence is not exactly one map.
pub fn require_map(sequence: &Sequence, what: &str) -> FnResult<XdmMap> {
    match sequence.as_slice() {
        [Item::Map(map)] => Ok(map.clone()),
        _ => Err(FnError::new(
            FnErrorCode::Xpty0004,
            format!("{what} expects exactly one map"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> XdmMap {
        XdmMap::new()
            .put(
                AtomicValue::integer(0_i64),
                Sequence::from(AtomicValue::string("no")),
            )
            .put(
                AtomicValue::integer(1_i64),
                Sequence::from(AtomicValue::string("yes")),
            )
    }

    #[test]
    fn size_keys_contains_and_get() {
        let map = sample();
        assert_eq!(size(&map), 2);
        assert_eq!(keys(&map).len(), 2);
        assert!(contains(&map, &AtomicValue::integer(0_i64)));
        // op:same-key crosses the numeric tower.
        assert!(contains(&map, &AtomicValue::double(1.0)));
        assert!(!contains(&map, &AtomicValue::string("1")));
        assert_eq!(
            get(&map, &AtomicValue::integer(1_i64))
                .first()
                .map(ToString::to_string),
            Some("yes".to_owned())
        );
        assert!(get(&map, &AtomicValue::integer(9_i64)).is_empty());
        assert_eq!(size(&XdmMap::new()), 0);
    }

    #[test]
    fn put_remove_and_entry_are_pure() {
        let map = sample();
        let extended = put(
            &map,
            AtomicValue::integer(2_i64),
            Sequence::from(AtomicValue::string("maybe")),
        );
        assert_eq!(size(&extended), 3);
        assert_eq!(size(&map), 2);

        let reduced = remove(
            &map,
            &[AtomicValue::integer(0_i64), AtomicValue::integer(9_i64)],
        );
        assert_eq!(size(&reduced), 1);
        assert_eq!(size(&map), 2);

        let single = entry(
            AtomicValue::string("k"),
            Sequence::from(AtomicValue::integer(1_i64)),
        );
        assert_eq!(size(&single), 1);
    }

    #[test]
    fn merge_honours_every_duplicates_option() -> FnResult<()> {
        let first = entry(
            AtomicValue::integer(0_i64),
            Sequence::from(AtomicValue::string("no")),
        );
        let clash = entry(
            AtomicValue::integer(0_i64),
            Sequence::from(AtomicValue::string("nope")),
        );

        let kept = merge(&[first.clone(), clash.clone()], DuplicateKeys::UseFirst)?;
        assert_eq!(
            get(&kept, &AtomicValue::integer(0_i64))
                .first()
                .map(ToString::to_string),
            Some("no".to_owned())
        );

        let last = merge(&[first.clone(), clash.clone()], DuplicateKeys::UseLast)?;
        assert_eq!(
            get(&last, &AtomicValue::integer(0_i64))
                .first()
                .map(ToString::to_string),
            Some("nope".to_owned())
        );

        let combined = merge(&[first.clone(), clash.clone()], DuplicateKeys::Combine)?;
        assert_eq!(get(&combined, &AtomicValue::integer(0_i64)).len(), 2);

        assert!(
            merge(&[first, clash], DuplicateKeys::Reject)
                .expect_err("duplicate key")
                .is(FnErrorCode::Fojs0003),
            "F&O 3.1 §17.1.5 gives `map:merge` its own code, not the constructor's XQDY0137"
        );
        Ok(())
    }

    #[test]
    fn duplicates_option_names() -> FnResult<()> {
        assert_eq!(duplicates_from_name("use-last")?, DuplicateKeys::UseLast);
        assert_eq!(duplicates_from_name("combine")?, DuplicateKeys::Combine);
        assert!(duplicates_from_name("nonsense").is_err());
        Ok(())
    }

    #[test]
    fn for_each_visits_every_entry() -> FnResult<()> {
        let context = FnContext::new();
        let keys_only = FunctionItem::anonymous(2, |_, arguments| Ok(arguments[0].clone()));
        assert_eq!(for_each(&context, &sample(), &keys_only)?.len(), 2);

        let wrong_arity = FunctionItem::anonymous(1, |_, arguments| Ok(arguments[0].clone()));
        assert!(for_each(&context, &sample(), &wrong_arity).is_err());
        Ok(())
    }

    #[test]
    fn find_searches_recursively() {
        let inner = entry(
            AtomicValue::integer(1_i64),
            Sequence::from(AtomicValue::string("deep")),
        );
        let outer = entry(
            AtomicValue::integer(0_i64),
            Sequence::singleton(Item::from(inner)),
        );
        let array = XdmArray::from_members([Sequence::singleton(Item::from(outer))]);
        let found = find(
            &Sequence::singleton(Item::from(array)),
            &AtomicValue::integer(1_i64),
        );
        assert_eq!(found.len(), 1);

        let missing = find(&Sequence::empty(), &AtomicValue::integer(1_i64));
        assert_eq!(missing.len(), 0);
    }

    /// The search used to descend by recursing, so a value nested deeply enough — which
    /// `fn:fold-left` builds without recursing at all — aborted the process with a stack
    /// overflow. It now walks an explicit stack and simply answers.
    ///
    /// The depth this can reach is not the walk's own limit but the data model's: freeing a
    /// nested [`Sequence`] still recurses, once per level, so the value cannot be *dropped* much
    /// past this even though the search now sails through it. That is also why the case runs on
    /// a thread of its own rather than on whatever stack the harness happens to give a test.
    /// Raise the depth once freeing a sequence has stopped recursing too.
    #[test]
    fn find_walks_a_deeply_nested_value_without_recursing() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let key = AtomicValue::integer(1_i64);
                let mut value = Sequence::from(AtomicValue::string("deep"));
                for _ in 0..5_000 {
                    value = Sequence::singleton(Item::from(XdmArray::from_members([value])));
                }
                // The sought entry sits at the very bottom, under every one of those arrays.
                let nested = entry(key.clone(), value);
                assert_eq!(
                    find(&Sequence::singleton(Item::from(nested)), &key).len(),
                    1
                );
            })
            .expect("the helper thread starts")
            .join()
            .expect("the helper thread finishes");
    }

    /// Matches come back in the pre-order the recursive walk produced, outermost first.
    #[test]
    fn find_reports_matches_in_document_order() {
        let key = AtomicValue::integer(1_i64);
        let inner = entry(key.clone(), Sequence::from(AtomicValue::string("inner")));
        let outer = entry(
            key.clone(),
            Sequence::from_items([Item::from(AtomicValue::string("outer")), Item::from(inner)]),
        );
        let found = find(&Sequence::singleton(Item::from(outer)), &key);
        let members: Vec<&Sequence> = found.members().collect();
        // The outer binding comes first and carries both its items; the binding nested inside it
        // follows. Reversing the two would mean the walk had stopped being pre-order.
        assert_eq!(members.len(), 2);
        assert_eq!(members[0].len(), 2);
        assert_eq!(members[1].len(), 1);
        assert_eq!(
            members[1].first().map(ToString::to_string),
            Some("inner".to_owned())
        );
    }

    #[test]
    fn require_map_rejects_other_shapes() {
        let map = Sequence::singleton(Item::from(sample()));
        assert!(require_map(&map, "map:size").is_ok());
        assert!(require_map(&Sequence::empty(), "map:size").is_err());
        assert!(require_map(&Sequence::from(AtomicValue::integer(1_i64)), "map:size").is_err());
    }
}
