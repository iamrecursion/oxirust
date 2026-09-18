//! [`deep_equal_with`] — `fn:deep-equal` evaluated against a *resolved* [`Collation`].
//!
//! `oxixml-infoset` implements the relation too, but it can only be handed a collation *URI*
//! and only knows the two collations every processor must implement, so a UCA request either
//! raised `FOCH0002` on two identical nodes or was silently swallowed into "not equal". This
//! module keeps the same rules and takes the collation object itself, so every collation the
//! [`Collations`](crate::Collations) registry can resolve works here exactly as it does for
//! `fn:compare`, `fn:distinct-values` and `fn:index-of`.

use crate::collation::Collation;
use crate::error::{FnError, FnErrorCode, FnResult};
use oxixml_infoset::{
    AtomicData, AtomicValue, Item, NodeKind, NodeRef, Sequence, XdmMap, XdmNode as _,
    compare_atomic_in_context,
};
use oxixml_xsd::TimezoneOffset;
use std::cell::Cell;
use std::cmp::Ordering;
use std::sync::Arc;

/// How deeply the comparison may descend into a tree, a map or an array.
///
/// The relation is defined recursively and the two values are the caller's, so without a
/// ceiling a pair of deeply nested arrays — or two trees built a hundred thousand elements deep
/// — exhausts the thread's stack, which is a process abort rather than a catchable error. Past
/// the ceiling the comparison raises [`FnErrorCode::Xpdy0130`], the code reserved for an
/// implementation-defined limit.
pub const MAX_DEPTH: usize = 256;

thread_local! {
    /// How many levels of the comparison are open on this thread.
    static DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Holds one level of comparison depth for as long as it is alive, so that the count is right
/// however the comparison returns.
struct DepthGuard;

impl DepthGuard {
    fn enter() -> FnResult<Self> {
        let depth = DEPTH.with(|depth| {
            let next = depth.get() + 1;
            depth.set(next);
            next
        });
        if depth > MAX_DEPTH {
            DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
            return Err(FnError::new(
                FnErrorCode::Xpdy0130,
                format!(
                    "the values compared nest more than {MAX_DEPTH} levels deep, which is this implementation's limit"
                ),
            ));
        }
        Ok(Self)
    }
}

impl Drop for DepthGuard {
    fn drop(&mut self) {
        DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

/// `fn:deep-equal` on two sequences under `collation`.
///
/// The rules are those of
/// [F&O 3.1 §15.3.1](https://www.w3.org/TR/xpath-functions-31/#func-deep-equal); `collation`
/// decides every string comparison — atomic `xs:string`/`xs:anyURI`/`xs:untypedAtomic` values,
/// text and comment contents, processing-instruction contents and namespace URIs. Map *keys*
/// are deliberately excluded: the specification compares them with the collation-independent
/// `op:same-key` relation. `implicit_timezone` is the [dynamic context's implicit
/// timezone](https://www.w3.org/TR/xpath-functions-31/#dt-implicit-timezone), which a date/time
/// value with no timezone of its own is read in before comparing (F&O 3.1 §10.4.4) — atomic
/// values are deep-equal exactly when they are `eq`, and `eq` reads them that way.
///
/// # Errors
///
/// [`FnErrorCode::Foty0015`] for a function item that is neither a map nor an array,
/// [`FnErrorCode::Xpdy0130`] when the values nest more than [`MAX_DEPTH`] levels deep, plus
/// whatever atomizing an attribute raises.
pub fn deep_equal_with(
    left: &Sequence,
    right: &Sequence,
    collation: &Arc<dyn Collation>,
    implicit_timezone: TimezoneOffset,
) -> FnResult<bool> {
    let _guard = DepthGuard::enter()?;
    if left.len() != right.len() {
        return Ok(false);
    }
    for (a, b) in left.iter().zip(right.iter()) {
        if !items_equal(a, b, collation, implicit_timezone)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// `fn:deep-equal` on two single items under `collation`.
///
/// # Errors
///
/// See [`deep_equal_with`].
pub fn items_equal(
    left: &Item,
    right: &Item,
    collation: &Arc<dyn Collation>,
    implicit_timezone: TimezoneOffset,
) -> FnResult<bool> {
    let _guard = DepthGuard::enter()?;
    match (left, right) {
        (Item::Atomic(a), Item::Atomic(b)) => Ok(atomics_equal(a, b, collation, implicit_timezone)),
        (Item::Node(a), Item::Node(b)) => nodes_equal(a, b, collation, implicit_timezone),
        (Item::Array(a), Item::Array(b)) => {
            if a.len() != b.len() {
                return Ok(false);
            }
            for (x, y) in a.members().zip(b.members()) {
                if !deep_equal_with(x, y, collation, implicit_timezone)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        (Item::Map(a), Item::Map(b)) => maps_equal(a, b, collation, implicit_timezone),
        // F&O 3.1 §14.1.4: a function item that is neither a map nor an array has no equality
        // to test, so `fn:deep-equal` raises rather than answering `false`.
        (Item::Function(_), _) | (_, Item::Function(_)) => Err(FnError::new(
            FnErrorCode::Foty0015,
            "fn:deep-equal cannot compare a function item",
        )),
        _ => Ok(false),
    }
}

/// Two atomic values under `collation`, reading a date/time operand with no timezone in
/// `implicit_timezone`.
///
/// Values whose types are not comparable are simply not deep-equal — unlike `eq`, this relation
/// never raises a type error for them — and two `NaN`s *are* deep-equal.
fn atomics_equal(
    left: &AtomicValue,
    right: &AtomicValue,
    collation: &Arc<dyn Collation>,
    implicit_timezone: TimezoneOffset,
) -> bool {
    if left.is_nan() && right.is_nan() {
        return true;
    }
    if let (AtomicData::Text(a), AtomicData::Text(b)) = (left.data(), right.data()) {
        return collation.equal(a, b);
    }
    // No string is involved, so no collation is either.
    matches!(
        compare_atomic_in_context(left, right, None, Some(implicit_timezone)),
        Ok(Some(Ordering::Equal))
    )
}

fn maps_equal(
    left: &XdmMap,
    right: &XdmMap,
    collation: &Arc<dyn Collation>,
    implicit_timezone: TimezoneOffset,
) -> FnResult<bool> {
    if left.len() != right.len() {
        return Ok(false);
    }
    for (key, value) in left.entries() {
        let Some(other) = right.get(key) else {
            return Ok(false);
        };
        if !deep_equal_with(value, other, collation, implicit_timezone)? {
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

fn nodes_equal(
    left: &NodeRef,
    right: &NodeRef,
    collation: &Arc<dyn Collation>,
    implicit_timezone: TimezoneOffset,
) -> FnResult<bool> {
    let kind = left.node_kind();
    if kind != right.node_kind() {
        return Ok(false);
    }
    match kind {
        NodeKind::Document => content_equal(left, right, collation, implicit_timezone),
        NodeKind::Element => {
            if left.node_name() != right.node_name() || left.nilled() != right.nilled() {
                return Ok(false);
            }
            if !attributes_equal(left, right, collation, implicit_timezone)? {
                return Ok(false);
            }
            content_equal(left, right, collation, implicit_timezone)
        }
        NodeKind::Attribute => {
            if left.node_name() != right.node_name() {
                return Ok(false);
            }
            deep_equal_with(
                &left.typed_value()?,
                &right.typed_value()?,
                collation,
                implicit_timezone,
            )
        }
        NodeKind::ProcessingInstruction | NodeKind::Namespace => {
            if left.node_name() != right.node_name() {
                return Ok(false);
            }
            Ok(collation.equal(&left.string_value(), &right.string_value()))
        }
        NodeKind::Text | NodeKind::Comment => {
            Ok(collation.equal(&left.string_value(), &right.string_value()))
        }
    }
}

fn attributes_equal(
    left: &NodeRef,
    right: &NodeRef,
    collation: &Arc<dyn Collation>,
    implicit_timezone: TimezoneOffset,
) -> FnResult<bool> {
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
        if !deep_equal_with(
            &attribute.typed_value()?,
            &counterpart.typed_value()?,
            collation,
            implicit_timezone,
        )? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn content_equal(
    left: &NodeRef,
    right: &NodeRef,
    collation: &Arc<dyn Collation>,
    implicit_timezone: TimezoneOffset,
) -> FnResult<bool> {
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
        if !nodes_equal(a, b, collation, implicit_timezone)? {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collation::Collations;
    use std::str::FromStr;

    fn codepoint() -> Arc<dyn Collation> {
        Collations::new().codepoint()
    }

    /// F&O 3.1 §15.3.1: two atomic values whose types cannot even be compared (`eq` would
    /// raise `XPTY0004`) are simply not deep-equal — `fn:deep-equal` never raises for this,
    /// unlike `eq` itself.
    #[test]
    fn atomically_incomparable_values_are_not_equal_rather_than_an_error() -> FnResult<()> {
        let date =
            AtomicValue::date(oxixml_xsd::Date::from_str("2014-08-25").map_err(|error| {
                FnError::new(FnErrorCode::Fodt0001, format!("bad fixture date: {error}"))
            })?);
        let text = AtomicValue::string("2014-08-25");
        let collation = codepoint();
        assert!(!atomics_equal(
            &date,
            &text,
            &collation,
            TimezoneOffset::UTC
        ));

        let left = Sequence::from(date);
        let right = Sequence::from(text);
        assert!(!deep_equal_with(
            &left,
            &right,
            &collation,
            TimezoneOffset::UTC
        )?);
        Ok(())
    }

    #[test]
    fn incomparable_atomics_nested_in_equal_length_sequences_are_not_equal() -> FnResult<()> {
        // The shape of iterate-027: a sequence produced by `.../number()` compared against a
        // sequence of numeric literals — if any pairwise comparison hits an atomically
        // incomparable pair, the whole `fn:deep-equal` call must still just answer `false`.
        let left = Sequence::from_iter([
            AtomicValue::double(13.24),
            AtomicValue::date(oxixml_xsd::Date::from_str("2014-08-25").map_err(|error| {
                FnError::new(FnErrorCode::Fodt0001, format!("bad fixture date: {error}"))
            })?),
        ]);
        let right = Sequence::from_iter([AtomicValue::double(13.24), AtomicValue::double(8.12)]);
        assert!(!deep_equal_with(
            &left,
            &right,
            &codepoint(),
            TimezoneOffset::UTC
        )?);
        Ok(())
    }

    /// The relation is recursive over both trees and arrays, and both come from the caller, so
    /// a value nested past the ceiling has to come back as an error rather than as a stack
    /// overflow, which would abort the process.
    #[test]
    fn nesting_past_the_ceiling_is_an_error_not_an_overflow() {
        let mut left = Sequence::from(AtomicValue::integer(1_i64));
        for _ in 0..600 {
            left = Sequence::from(Item::Array(oxixml_infoset::XdmArray::from_members([left])));
        }
        let right = left.clone();
        let error = deep_equal_with(&left, &right, &codepoint(), TimezoneOffset::UTC)
            .expect_err("past the nesting ceiling");
        assert!(error.is(FnErrorCode::Xpdy0130));
        assert!(error.to_string().contains(&MAX_DEPTH.to_string()));
    }
}
