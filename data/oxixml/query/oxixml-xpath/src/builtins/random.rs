//! [`fn:random-number-generator`](https://www.w3.org/TR/xpath-functions-31/#func-random-number-generator),
//! F&O 3.1 §5.7.1.
//!
//! The function answers a *map* whose members are
//!
//! | key | type | meaning |
//! |---|---|---|
//! | `number` | `xs:double` | a value in `[0, 1)` |
//! | `next` | `function() as map(xs:string, item())` | the next generator in the stream |
//! | `permute` | `function($seq as item()*) as item()*` | a random permutation of its argument |
//!
//! Two of the three members are *function items*, which is why this lives here rather than in
//! `oxixml-fn`: only the evaluator has a callable representation. They reach the map through
//! [`XpathFunction::to_xdm`], the data model's opaque function-item handle.
//!
//! # Determinism
//!
//! §5.7.1 makes the result *deterministic*: two calls with the same seed — including two calls
//! with no seed at all, which the specification says behave as if given an
//! implementation-dependent seed that is the same throughout one execution — must produce the
//! same generator. The seed is therefore mapped to a `u64` by hashing its string value, and the
//! stream is a `splitmix64`, which is reproducible and has no hidden state.

use crate::context::DynamicContext;
use crate::error::XPathResult;
use crate::function::XpathFunction;
use crate::value::XpathValue;
use oxixml_infoset::{AtomicValue, Item, Sequence, XdmMap};

/// Mixes one step of `splitmix64`, the stream this generator is built on.
fn splitmix64(state: u64) -> (u64, u64) {
    let next = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = next;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (next, z ^ (z >> 31))
}

/// The FNV-1a hash of a seed's string value. Any total, deterministic string-to-`u64` mapping
/// satisfies §5.7.1; this one is small and has no dependencies.
fn seed_of(seed: Option<&AtomicValue>) -> u64 {
    let text = seed.map(AtomicValue::string_value).unwrap_or_default();
    let mut hash = 0xCBF2_9CE4_8422_2325_u64;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

/// The `xs:double` in `[0, 1)` drawn from `value`: the top 53 bits, which is exactly the
/// precision an `xs:double` has.
#[allow(clippy::cast_precision_loss)]
fn unit_double(value: u64) -> f64 {
    (value >> 11) as f64 / (1_u64 << 53) as f64
}

/// Builds the generator map for `state`.
fn generator(state: u64) -> XdmMap {
    let (advanced, drawn) = splitmix64(state);

    let next_state = advanced;
    let next = XpathFunction::native(None, 0, move |_context, _arguments| {
        Ok(XpathValue::from(Item::Map(generator(next_state))))
    });

    let permute_state = drawn;
    let permute = XpathFunction::native(None, 1, move |_context, arguments| {
        let input = arguments.first().cloned().unwrap_or_default();
        Ok(permute(&input, permute_state))
    });

    XdmMap::new()
        .put(
            AtomicValue::string("number"),
            Sequence::from(AtomicValue::double(unit_double(drawn))),
        )
        .put(
            AtomicValue::string("next"),
            Sequence::from(Item::Function(next.to_xdm())),
        )
        .put(
            AtomicValue::string("permute"),
            Sequence::from(Item::Function(permute.to_xdm())),
        )
}

/// A Fisher–Yates shuffle driven by the same stream, so that the permutation a generator
/// produces is a function of its seed alone.
fn permute(input: &XpathValue, state: u64) -> XpathValue {
    let mut items: Vec<_> = input.iter().cloned().collect();
    let mut state = state;
    let mut index = items.len();
    while index > 1 {
        let (advanced, drawn) = splitmix64(state);
        state = advanced;
        index -= 1;
        let pick = usize::try_from(drawn % (index as u64 + 1)).unwrap_or(0);
        items.swap(index, pick);
    }
    XpathValue::from_items(items)
}

/// Whether this module implements `local` at `arity`.
pub(crate) fn supports(local: &str, arity: usize) -> bool {
    local == "random-number-generator" && arity <= 1
}

/// Evaluates `fn:random-number-generator`.
///
/// # Errors
///
/// Whatever atomizing the seed raises; the seed itself is never rejected, since §5.7.1 accepts
/// any `xs:anyAtomicType?`.
pub(crate) fn call(_context: &DynamicContext, arguments: &[XpathValue]) -> XPathResult<XpathValue> {
    let seed = match arguments.first() {
        Some(value) => value.atomize_at_most_one("the seed of fn:random-number-generator")?,
        None => None,
    };
    Ok(XpathValue::from(Item::Map(generator(seed_of(
        seed.as_ref(),
    )))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_drawn_number_is_a_double_in_the_unit_interval() {
        for state in [0_u64, 1, 42, u64::MAX] {
            let (_, drawn) = splitmix64(state);
            let number = unit_double(drawn);
            assert!((0.0..1.0).contains(&number), "{number} out of range");
        }
    }

    #[test]
    fn the_same_seed_gives_the_same_generator() {
        assert_eq!(seed_of(None), seed_of(Some(&AtomicValue::string(""))));
        assert_ne!(
            seed_of(Some(&AtomicValue::string("0"))),
            seed_of(Some(&AtomicValue::string("1")))
        );
    }

    #[test]
    fn permutation_keeps_every_item_exactly_once() {
        let input = XpathValue::from_iter((1..=100_i64).map(AtomicValue::integer));
        let shuffled = permute(&input, 12345);
        assert_eq!(shuffled.len(), 100);
        let mut seen: Vec<String> = shuffled.iter().map(ToString::to_string).collect();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 100);
    }

    #[test]
    fn an_empty_sequence_permutes_to_itself() {
        assert!(permute(&XpathValue::empty(), 7).is_empty());
    }
}
