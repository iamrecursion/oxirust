//! The FLWOR tuple stream.
//!
//! [XQuery 3.1 §3.12](https://www.w3.org/TR/xquery-31/#id-flwor-expressions) defines a FLWOR as a
//! pipeline: an initial clause produces a stream of *tuples* — one binding per variable in scope
//! — and every later clause transforms that stream, until `return` maps each surviving tuple to a
//! value and the values are concatenated. This module is that pipeline, one function per clause.

use super::program::{
    CompiledFlwor, CompiledForBinding, CompiledGrouping, CompiledLetBinding, CompiledOrderSpec,
    CompiledWindow, CompiledWindowCondition, FlworStep,
};
use super::{Frame, apply_type_check, as_comparable, evaluate};
use crate::error::{XQueryError, XQueryErrorCode, XQueryResult};
use oxixml_fn::Collation;
use oxixml_infoset::{AtomicData, AtomicValue, ExpandedName, compare_atomic_in_context};
use oxixml_xpath::{EvaluationLimits, XpathItem, XpathValue};
use oxixml_xsd::TimezoneOffset;
use std::cmp::Ordering;
use std::sync::Arc;

/// One tuple of the stream: the variables in scope and their values, in binding order.
type Tuple = Vec<(ExpandedName, XpathValue)>;

/// Evaluates a whole FLWOR expression.
pub(crate) fn evaluate_flwor(flwor: &CompiledFlwor, frame: &Frame) -> XQueryResult<XpathValue> {
    let mut tuples: Vec<Tuple> = vec![Vec::new()];
    for step in &flwor.steps {
        tuples = apply_step(step, tuples, frame)?;
    }
    let mut out = Vec::new();
    for tuple in tuples {
        // The `return` clause runs once per surviving tuple, so this is the FLWOR's own
        // per-item loop and takes the same checkpoint the evaluator's do.
        EvaluationLimits::check_deadline()?;
        let inner = extend(frame, &tuple);
        out.extend(evaluate(&flwor.result, &inner)?.into_items());
    }
    Ok(XpathValue::from_items(out))
}

fn extend(frame: &Frame, tuple: &Tuple) -> Frame {
    let mut inner = frame.clone();
    for (name, value) in tuple {
        inner = inner.bind(name, value.clone());
    }
    inner
}

fn apply_step(step: &FlworStep, tuples: Vec<Tuple>, frame: &Frame) -> XQueryResult<Vec<Tuple>> {
    match step {
        FlworStep::For(bindings) => {
            let mut out = Vec::new();
            for tuple in tuples {
                expand_for(bindings, 0, tuple, frame, &mut out)?;
            }
            Ok(out)
        }
        FlworStep::Let(bindings) => {
            let mut out = Vec::with_capacity(tuples.len());
            for mut tuple in tuples {
                for binding in bindings {
                    let inner = extend(frame, &tuple);
                    let value = bind_value(binding, &inner)?;
                    tuple.push((binding.name.clone(), value));
                }
                out.push(tuple);
            }
            Ok(out)
        }
        FlworStep::Where(condition) => {
            let mut out = Vec::new();
            for tuple in tuples {
                let inner = extend(frame, &tuple);
                if evaluate(condition, &inner)?.effective_boolean_value()? {
                    out.push(tuple);
                }
            }
            Ok(out)
        }
        FlworStep::Count(name) => Ok(tuples
            .into_iter()
            .enumerate()
            .map(|(index, mut tuple)| {
                #[allow(clippy::cast_possible_wrap)]
                let position = (index + 1) as i64;
                tuple.push((
                    name.clone(),
                    XpathValue::from(AtomicValue::integer(position)),
                ));
                tuple
            })
            .collect()),
        FlworStep::OrderBy(specs) => order_by(specs, tuples, frame),
        FlworStep::GroupBy(specs) => group_by(specs, tuples, frame),
        FlworStep::Window(window) => windows(window, tuples, frame),
    }
}

fn bind_value(binding: &CompiledLetBinding, frame: &Frame) -> XQueryResult<XpathValue> {
    let value = evaluate(&binding.value, frame)?;
    match &binding.check {
        Some(check) => apply_type_check(check, value, frame),
        None => Ok(value),
    }
}

fn expand_for(
    bindings: &[CompiledForBinding],
    index: usize,
    tuple: Tuple,
    frame: &Frame,
    out: &mut Vec<Tuple>,
) -> XQueryResult<()> {
    let Some(binding) = bindings.get(index) else {
        out.push(tuple);
        return Ok(());
    };
    let inner = extend(frame, &tuple);
    let items = evaluate(&binding.value, &inner)?.into_items();
    if items.is_empty() && binding.allowing_empty {
        let mut extended = tuple.clone();
        let mut empty = XpathValue::default();
        if let Some(check) = &binding.check {
            empty = apply_type_check(check, empty, &inner)?;
        }
        extended.push((binding.name.clone(), empty));
        if let Some(positional) = &binding.positional {
            extended.push((
                positional.clone(),
                XpathValue::from(AtomicValue::integer(0_i64)),
            ));
        }
        return expand_for(bindings, index + 1, extended, frame, out);
    }
    for (position, item) in items.into_iter().enumerate() {
        EvaluationLimits::check_deadline()?;
        let mut value = XpathValue::singleton(item);
        if let Some(check) = &binding.check {
            value = apply_type_check(check, value, &inner)?;
        }
        let mut extended = tuple.clone();
        extended.push((binding.name.clone(), value));
        if let Some(positional) = &binding.positional {
            #[allow(clippy::cast_possible_wrap)]
            let number = (position + 1) as i64;
            extended.push((
                positional.clone(),
                XpathValue::from(AtomicValue::integer(number)),
            ));
        }
        expand_for(bindings, index + 1, extended, frame, out)?;
    }
    Ok(())
}

// --- order by ----------------------------------------------------------------------------------

fn order_by(
    specs: &[CompiledOrderSpec],
    tuples: Vec<Tuple>,
    frame: &Frame,
) -> XQueryResult<Vec<Tuple>> {
    let mut keyed: Vec<(Vec<Option<AtomicValue>>, Tuple)> = Vec::with_capacity(tuples.len());
    for tuple in tuples {
        let inner = extend(frame, &tuple);
        let mut keys = Vec::with_capacity(specs.len());
        for spec in specs {
            keys.push(sort_key(&evaluate(&spec.key, &inner)?)?);
        }
        keyed.push((keys, tuple));
    }
    let default_greatest = frame.options.empty_order == crate::context::EmptyOrder::Greatest;
    // §3.12.5 compares string keys under the collation in force. Resolving it here, through the
    // same registry `fn:compare` consults, is what makes a UCA collation reach `order by`;
    // handing the raw URI to the infoset layer instead made every such comparison "incomparable"
    // and raised a spurious `XPTY0004`.
    let mut collations = Vec::with_capacity(specs.len());
    for spec in specs {
        collations.push(resolve_collation(frame, spec.collation.as_deref()));
    }
    // F&O 3.1 §10.4.4: a date/time key with no timezone is read in the dynamic context's
    // implicit timezone before comparing, exactly as the `eq`/`lt` operators read it.
    let implicit_timezone = frame.context.fn_context().implicit_timezone();
    let incomparable = std::cell::Cell::new(false);
    keyed.sort_by(|left, right| {
        for (index, spec) in specs.iter().enumerate() {
            let a = left.0.get(index).and_then(Option::as_ref);
            let b = right.0.get(index).and_then(Option::as_ref);
            let empty_greatest = spec.empty_greatest.unwrap_or(default_greatest);
            let Some(collation) = collations.get(index) else {
                incomparable.set(true);
                return Ordering::Equal;
            };
            let mut ordering =
                match compare_keys(a, b, empty_greatest, collation, implicit_timezone) {
                    Some(ordering) => ordering,
                    None => {
                        incomparable.set(true);
                        Ordering::Equal
                    }
                };
            if !spec.ascending {
                ordering = ordering.reverse();
            }
            if ordering != Ordering::Equal {
                return ordering;
            }
        }
        Ordering::Equal
    });
    if incomparable.get() {
        return Err(XQueryError::new(
            XQueryErrorCode::Xpty0004,
            "an `order by` clause compared values of incomparable types",
        ));
    }
    Ok(keyed.into_iter().map(|(_, tuple)| tuple).collect())
}

/// An `order by` key: atomized, at most one item, with `NaN` treated as the empty sequence.
fn sort_key(value: &XpathValue) -> XQueryResult<Option<AtomicValue>> {
    let atomized = value.atomize()?;
    match atomized.as_slice() {
        [] => Ok(None),
        [single] => {
            if single.is_nan() {
                return Ok(None);
            }
            Ok(Some(as_comparable(single)))
        }
        _ => Err(XQueryError::new(
            XQueryErrorCode::Xpty0004,
            "an `order by` key must be at most one item",
        )),
    }
}

/// Resolves an `order by`/`group by` collation URI — the clause's own, or the prolog default —
/// through the function library's collation registry.
///
/// A relative URI is resolved against the static base URI first, as §4.4 requires. The compiler
/// has already rejected a URI this processor does not know (`XQST0076`), so a resolution that
/// still fails here falls back on the code point collation rather than raising.
fn resolve_collation(frame: &Frame, uri: Option<&str>) -> Arc<dyn Collation> {
    let uri = uri.unwrap_or(&frame.options.default_collation);
    let collations = frame.context.fn_context().collations();
    let resolved = match frame.options.base_uri.as_deref() {
        Some(base) => oxixml_fn::strings::uri::resolve_uri_against(uri, base)
            .unwrap_or_else(|_| uri.to_owned()),
        None => uri.to_owned(),
    };
    collations
        .resolve(&resolved)
        .or_else(|_| collations.resolve(uri))
        .unwrap_or_else(|_| collations.codepoint())
}

/// Orders two atomic keys, using `collation` whenever both are strings and `implicit_timezone`
/// whenever either is a date/time value with no timezone of its own.
///
/// Only a string-to-string comparison consults a collation, so every other pair goes to the
/// data model's own ordering, which is where `implicit_timezone` applies.
fn compare_collated(
    left: &AtomicValue,
    right: &AtomicValue,
    collation: &Arc<dyn Collation>,
    implicit_timezone: TimezoneOffset,
) -> Option<Ordering> {
    if let (AtomicData::Text(a), AtomicData::Text(b)) = (left.data(), right.data()) {
        return Some(collation.compare(a, b));
    }
    compare_atomic_in_context(left, right, None, Some(implicit_timezone))
        .ok()
        .flatten()
}

fn compare_keys(
    left: Option<&AtomicValue>,
    right: Option<&AtomicValue>,
    empty_greatest: bool,
    collation: &Arc<dyn Collation>,
    implicit_timezone: TimezoneOffset,
) -> Option<Ordering> {
    match (left, right) {
        (None, None) => Some(Ordering::Equal),
        (None, Some(_)) => Some(if empty_greatest {
            Ordering::Greater
        } else {
            Ordering::Less
        }),
        (Some(_), None) => Some(if empty_greatest {
            Ordering::Less
        } else {
            Ordering::Greater
        }),
        (Some(left), Some(right)) => compare_collated(left, right, collation, implicit_timezone),
    }
}

// --- group by ----------------------------------------------------------------------------------

fn group_by(
    specs: &[CompiledGrouping],
    tuples: Vec<Tuple>,
    frame: &Frame,
) -> XQueryResult<Vec<Tuple>> {
    // §3.12.7 processes the grouping specs in order, each one *binding* its variable for the
    // ones that follow. Two specs may therefore name the same variable, in which case the later
    // binding occludes the earlier one and the grouping key has one component for it, not two.
    let mut grouping_names: Vec<ExpandedName> = Vec::new();
    let mut collations: Vec<Arc<dyn Collation>> = Vec::new();
    for spec in specs {
        let collation = resolve_collation(frame, spec.collation.as_deref());
        match grouping_names.iter().position(|name| name == &spec.name) {
            Some(index) => {
                if let Some(slot) = collations.get_mut(index) {
                    *slot = collation;
                }
            }
            None => {
                grouping_names.push(spec.name.clone());
                collations.push(collation);
            }
        }
    }

    // F&O 3.1 §10.4.4: a date/time key with no timezone is read in the dynamic context's
    // implicit timezone before comparing, exactly as the `eq` operator reads it.
    let implicit_timezone = frame.context.fn_context().implicit_timezone();
    let mut groups: Vec<(Vec<Option<AtomicValue>>, Vec<Tuple>)> = Vec::new();
    for tuple in tuples {
        let mut inner = extend(frame, &tuple);
        let mut key: Vec<Option<AtomicValue>> = vec![None; grouping_names.len()];
        for spec in specs {
            let value = match &spec.value {
                Some(program) => evaluate(program, &inner)?,
                None => inner.context.variable(&spec.name).cloned().ok_or_else(|| {
                    XQueryError::new(
                        XQueryErrorCode::Xqst0094,
                        format!(
                            "`group by ${}` names a variable that is not in scope",
                            spec.name
                        ),
                    )
                })?,
            };
            // §3.12.7: "the grouping keys are created by atomizing the values of the grouping
            // variables … If a type declaration is present and the resulting **atomized** value
            // is not an instance of the specified type, a type error is raised [err:XPTY0004]."
            // The declaration therefore constrains the post-atomization value, which is why
            // `group by $k as attribute(*) := $x/@gender` is an error rather than a match.
            let atomized = value.atomize()?;
            let component = match atomized.as_slice() {
                [] => None,
                [single] => Some(as_comparable(single)),
                _ => {
                    return Err(XQueryError::new(
                        XQueryErrorCode::Xpty0004,
                        "a grouping key must be at most one item",
                    ));
                }
            };
            if let Some(check) = &spec.check {
                let key = atomized
                    .iter()
                    .cloned()
                    .map(XpathItem::from)
                    .collect::<Vec<_>>();
                apply_type_check(check, XpathValue::from_items(key), &inner)?;
            }
            inner = inner.bind(
                &spec.name,
                component
                    .clone()
                    .map_or_else(XpathValue::default, XpathValue::from),
            );
            if let Some(index) = grouping_names.iter().position(|name| name == &spec.name) {
                if let Some(slot) = key.get_mut(index) {
                    *slot = component;
                }
            }
        }
        match groups
            .iter_mut()
            .find(|(existing, _)| keys_equal(existing, &key, &collations, implicit_timezone))
        {
            Some((_, members)) => members.push(tuple),
            None => groups.push((key, vec![tuple])),
        }
    }

    let mut out = Vec::with_capacity(groups.len());
    for (key, members) in groups {
        let mut tuple: Tuple = Vec::new();
        let mut names: Vec<ExpandedName> = Vec::new();
        for member in &members {
            for (name, _) in member {
                if !names.contains(name) {
                    names.push(name.clone());
                }
            }
        }
        for name in names {
            if grouping_names.contains(&name) {
                continue;
            }
            let mut collected = Vec::new();
            for member in &members {
                if let Some((_, value)) = member.iter().rev().find(|(bound, _)| *bound == name) {
                    collected.extend(value.iter().cloned());
                }
            }
            tuple.push((name, XpathValue::from_items(collected)));
        }
        for (name, value) in grouping_names.iter().zip(key) {
            tuple.push((
                name.clone(),
                value.map_or_else(XpathValue::default, XpathValue::from),
            ));
        }
        out.push(tuple);
    }
    Ok(out)
}

fn keys_equal(
    left: &[Option<AtomicValue>],
    right: &[Option<AtomicValue>],
    collations: &[Arc<dyn Collation>],
    implicit_timezone: TimezoneOffset,
) -> bool {
    if left.len() != right.len() {
        return false;
    }
    for (index, (a, b)) in left.iter().zip(right).enumerate() {
        let same = match (a, b, collations.get(index)) {
            (None, None, _) => true,
            (None, Some(_), _) | (Some(_), None, _) => false,
            (Some(_), Some(_), None) => false,
            (Some(a), Some(b), Some(collation)) => {
                if a.is_nan() || b.is_nan() {
                    a.is_nan() && b.is_nan()
                } else {
                    compare_collated(a, b, collation, implicit_timezone) == Some(Ordering::Equal)
                }
            }
        };
        if !same {
            return false;
        }
    }
    true
}

// --- window clauses ------------------------------------------------------------------------------

fn windows(window: &CompiledWindow, tuples: Vec<Tuple>, frame: &Frame) -> XQueryResult<Vec<Tuple>> {
    let mut out = Vec::new();
    for tuple in tuples {
        let inner = extend(frame, &tuple);
        let items = evaluate(&window.value, &inner)?.into_items();
        let length = items.len();
        let mut position = 0_usize;
        while position < length {
            if !holds(&window.start, &items, position, &inner, &[])? {
                position += 1;
                continue;
            }
            // The end condition may name the start condition's variables, so they are bound
            // before it is tested and again in the tuple the clause emits.
            let mut started: Tuple = Vec::new();
            bind_condition(&window.start, &items, position, &mut started);
            let (last, complete) = find_end(window, &items, position, &inner, &started)?;
            if complete || !window.only {
                let mut emitted = tuple.clone();
                let slice: Vec<XpathItem> = items
                    .get(position..=last)
                    .map(<[XpathItem]>::to_vec)
                    .unwrap_or_default();
                let mut value = XpathValue::from_items(slice);
                if let Some(check) = &window.check {
                    value = apply_type_check(check, value, &inner)?;
                }
                emitted.push((window.name.clone(), value));
                emitted.extend(started.iter().cloned());
                if let Some(end) = &window.end {
                    bind_condition(end, &items, last, &mut emitted);
                }
                out.push(emitted);
            }
            position = if window.tumbling {
                last + 1
            } else {
                position + 1
            };
        }
    }
    Ok(out)
}

/// Finds where a window that started at `start` ends, and whether its end condition held.
fn find_end(
    window: &CompiledWindow,
    items: &[XpathItem],
    start: usize,
    frame: &Frame,
    started: &Tuple,
) -> XQueryResult<(usize, bool)> {
    let length = items.len();
    if let Some(end) = &window.end {
        let mut candidate = start;
        while candidate < length {
            if holds(end, items, candidate, frame, started)? {
                return Ok((candidate, true));
            }
            candidate += 1;
        }
        return Ok((length.saturating_sub(1), false));
    }
    // A tumbling window with no end condition runs until the next window starts.
    let mut candidate = start + 1;
    while candidate < length {
        if holds(&window.start, items, candidate, frame, &[])? {
            return Ok((candidate - 1, true));
        }
        candidate += 1;
    }
    Ok((length.saturating_sub(1), true))
}

fn holds(
    condition: &CompiledWindowCondition,
    items: &[XpathItem],
    position: usize,
    frame: &Frame,
    extra: &[(ExpandedName, XpathValue)],
) -> XQueryResult<bool> {
    let Some(when) = &condition.when else {
        return Ok(true);
    };
    let mut bindings: Tuple = extra.to_vec();
    bind_condition(condition, items, position, &mut bindings);
    let inner = extend(frame, &bindings);
    Ok(evaluate(when, &inner)?.effective_boolean_value()?)
}

fn bind_condition(
    condition: &CompiledWindowCondition,
    items: &[XpathItem],
    position: usize,
    into: &mut Tuple,
) {
    if let Some(name) = &condition.current {
        let value = items
            .get(position)
            .cloned()
            .map_or_else(XpathValue::default, XpathValue::singleton);
        into.push((name.clone(), value));
    }
    if let Some(name) = &condition.positional {
        #[allow(clippy::cast_possible_wrap)]
        let number = (position + 1) as i64;
        into.push((name.clone(), XpathValue::from(AtomicValue::integer(number))));
    }
    if let Some(name) = &condition.previous {
        let value = position
            .checked_sub(1)
            .and_then(|index| items.get(index).cloned())
            .map_or_else(XpathValue::default, XpathValue::singleton);
        into.push((name.clone(), value));
    }
    if let Some(name) = &condition.next {
        let value = items
            .get(position + 1)
            .cloned()
            .map_or_else(XpathValue::default, XpathValue::singleton);
        into.push((name.clone(), value));
    }
}
