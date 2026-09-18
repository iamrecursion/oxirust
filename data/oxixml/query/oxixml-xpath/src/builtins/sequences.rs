//! The general sequence functions of
//! [F&O 3.1 §14.1](https://www.w3.org/TR/xpath-functions-31/#general-seq-funcs).
//!
//! [`oxixml_fn::sequences`] implements all of these over [`Sequence`](oxixml_infoset::Sequence),
//! and that is the right home for them — but a `Sequence` cannot hold a function item, so
//! `fn:exists(fn:abs#1)`, `fn:count(map:get#2)` and `fn:head(($f, $g))` would be type errors
//! rather than answers. These functions do not look *inside* their argument, so the engine
//! reimplements the dozen of them over [`XpathValue`] and they work on every item kind.

use super::integer;
use crate::error::{XPathError, XPathErrorCode, XPathResult};
use crate::eval::boolean;
use crate::value::XpathValue;

/// Whether `local` is one of the sequence functions implemented here.
pub(crate) fn supports(local: &str, arity: usize) -> bool {
    matches!(
        (local, arity),
        ("empty", 1)
            | ("exists", 1)
            | ("count", 1)
            | ("head", 1)
            | ("tail", 1)
            | ("reverse", 1)
            | ("unordered", 1)
            | ("subsequence", 2 | 3)
            | ("insert-before", 3)
            | ("remove", 2)
            | ("zero-or-one", 1)
            | ("one-or-more", 1)
            | ("exactly-one", 1)
    )
}

/// Invokes one of the sequence functions.
///
/// # Errors
///
/// `FORG0003`, `FORG0004` and `FORG0005` from the cardinality functions, and `XPTY0004` for a
/// non-numeric position argument.
pub(crate) fn call(local: &str, arguments: &[XpathValue]) -> XPathResult<XpathValue> {
    let source = &arguments[0];
    match local {
        "empty" => Ok(boolean(source.is_empty())),
        "exists" => Ok(boolean(!source.is_empty())),
        "count" => Ok(integer(i64::try_from(source.len()).unwrap_or(i64::MAX))),
        "head" => Ok(source
            .first()
            .cloned()
            .map(XpathValue::singleton)
            .unwrap_or_default()),
        "tail" => Ok(XpathValue::from_items(
            source.iter().skip(1).cloned().collect::<Vec<_>>(),
        )),
        "reverse" => {
            let mut items = source.as_slice().to_vec();
            items.reverse();
            Ok(XpathValue::from_items(items))
        }
        "unordered" => Ok(source.clone()),
        "subsequence" => {
            let start = number(&arguments[1], "the $startingLoc of fn:subsequence")?;
            let length = match arguments.get(2) {
                Some(argument) => Some(number(argument, "the $length of fn:subsequence")?),
                None => None,
            };
            Ok(subsequence(source, start, length))
        }
        "insert-before" => {
            let position = number(&arguments[1], "the $position of fn:insert-before")?;
            Ok(insert_before(source, position, &arguments[2]))
        }
        "remove" => {
            let position = number(&arguments[1], "the $position of fn:remove")?;
            Ok(remove(source, position))
        }
        "zero-or-one" => {
            if source.len() > 1 {
                return Err(cardinality(
                    XPathErrorCode::Forg0003,
                    "fn:zero-or-one was given a sequence of more than one item",
                ));
            }
            Ok(source.clone())
        }
        "one-or-more" => {
            if source.is_empty() {
                return Err(cardinality(
                    XPathErrorCode::Forg0004,
                    "fn:one-or-more was given the empty sequence",
                ));
            }
            Ok(source.clone())
        }
        "exactly-one" => {
            if !source.is_singleton() {
                return Err(cardinality(
                    XPathErrorCode::Forg0005,
                    "fn:exactly-one requires exactly one item",
                ));
            }
            Ok(source.clone())
        }
        other => Err(XPathError::type_error(format!(
            "`fn:{other}` is not implemented"
        ))),
    }
}

fn cardinality(code: XPathErrorCode, message: &str) -> XPathError {
    XPathError::new(code, message.to_owned())
}

fn number(value: &XpathValue, what: &str) -> XPathResult<f64> {
    let Some(atomic) = value.atomize_at_most_one(what)? else {
        return Err(XPathError::type_error(format!("{what} requires a number")));
    };
    Ok(oxixml_fn::numeric::cast_to_double(&atomic))
}

/// `fn:subsequence`, defined by the specification as
/// `$seq[round($start) le position() and position() lt round($start) + round($length)]`, with
/// `NaN` bounds selecting nothing and an infinite `$length` selecting everything from `$start`.
fn subsequence(source: &XpathValue, start: f64, length: Option<f64>) -> XpathValue {
    if start.is_nan() || length.is_some_and(f64::is_nan) {
        return XpathValue::empty();
    }
    let first = round_half_up(start);
    let limit = length.map(|length| first + round_half_up(length));
    let mut kept = Vec::new();
    for (index, item) in source.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let position = (index + 1) as f64;
        if position >= first && limit.is_none_or(|limit| position < limit) {
            kept.push(item.clone());
        }
    }
    XpathValue::from_items(kept)
}

/// `fn:round` with the half-up rule the sequence functions use.
fn round_half_up(value: f64) -> f64 {
    if value.is_infinite() {
        return value;
    }
    (value + 0.5).floor()
}

fn insert_before(source: &XpathValue, position: f64, inserted: &XpathValue) -> XpathValue {
    #[allow(clippy::cast_precision_loss)]
    let length = source.len() as f64;
    let at = position.max(1.0).min(length + 1.0);
    let mut items = Vec::with_capacity(source.len() + inserted.len());
    let mut placed = false;
    for (index, item) in source.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let ordinal = (index + 1) as f64;
        if !placed && ordinal >= at {
            items.extend(inserted.iter().cloned());
            placed = true;
        }
        items.push(item.clone());
    }
    if !placed {
        items.extend(inserted.iter().cloned());
    }
    XpathValue::from_items(items)
}

fn remove(source: &XpathValue, position: f64) -> XpathValue {
    let mut items = Vec::with_capacity(source.len());
    for (index, item) in source.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let ordinal = (index + 1) as f64;
        if ordinal != position {
            items.push(item.clone());
        }
    }
    XpathValue::from_items(items)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::AtomicValue;

    fn integers(values: &[i64]) -> XpathValue {
        XpathValue::from_iter(values.iter().map(|value| AtomicValue::integer(*value)))
    }

    fn double(value: f64) -> XpathValue {
        XpathValue::from(AtomicValue::double(value))
    }

    #[test]
    fn the_catalogue_matches_the_specification() {
        assert!(supports("subsequence", 2));
        assert!(supports("subsequence", 3));
        assert!(!supports("subsequence", 1));
        assert!(!supports("sum", 1));
    }

    #[test]
    fn cardinality_functions_raise_their_codes() {
        let two = integers(&[1, 2]);
        assert_eq!(
            call("zero-or-one", std::slice::from_ref(&two))
                .expect_err("two items")
                .code(),
            Some(XPathErrorCode::Forg0003)
        );
        assert_eq!(
            call("one-or-more", &[XpathValue::empty()])
                .expect_err("no items")
                .code(),
            Some(XPathErrorCode::Forg0004)
        );
        assert_eq!(
            call("exactly-one", &[two]).expect_err("two items").code(),
            Some(XPathErrorCode::Forg0005)
        );
    }

    #[test]
    fn subsequence_follows_the_specification_examples() -> XPathResult<()> {
        let source = integers(&[1, 2, 3, 4, 5]);
        assert_eq!(
            call("subsequence", &[source.clone(), double(4.0)])?.to_string(),
            "4 5"
        );
        assert_eq!(
            call("subsequence", &[source.clone(), double(3.0), double(2.0)])?.to_string(),
            "3 4"
        );
        assert_eq!(
            call("subsequence", &[source.clone(), double(-1.0), double(3.0)])?.to_string(),
            "1"
        );
        assert!(call("subsequence", &[source, double(f64::NAN)])?.is_empty());
        Ok(())
    }

    #[test]
    fn insert_and_remove_clamp_their_positions() -> XPathResult<()> {
        let source = integers(&[1, 2, 3]);
        let inserted = integers(&[9]);
        assert_eq!(
            call(
                "insert-before",
                &[source.clone(), double(0.0), inserted.clone()]
            )?
            .to_string(),
            "9 1 2 3"
        );
        assert_eq!(
            call("insert-before", &[source.clone(), double(99.0), inserted])?.to_string(),
            "1 2 3 9"
        );
        assert_eq!(
            call("remove", &[source.clone(), double(2.0)])?.to_string(),
            "1 3"
        );
        assert_eq!(call("remove", &[source, double(9.0)])?.to_string(), "1 2 3");
        Ok(())
    }

    #[test]
    fn the_simple_accessors() -> XPathResult<()> {
        let source = integers(&[1, 2, 3]);
        assert_eq!(
            call("count", std::slice::from_ref(&source))?.to_string(),
            "3"
        );
        assert_eq!(
            call("head", std::slice::from_ref(&source))?.to_string(),
            "1"
        );
        assert_eq!(
            call("tail", std::slice::from_ref(&source))?.to_string(),
            "2 3"
        );
        assert_eq!(
            call("reverse", std::slice::from_ref(&source))?.to_string(),
            "3 2 1"
        );
        assert_eq!(
            call("exists", std::slice::from_ref(&source))?.to_string(),
            "true"
        );
        assert_eq!(call("empty", &[source])?.to_string(), "false");
        assert_eq!(call("empty", &[XpathValue::empty()])?.to_string(), "true");
        Ok(())
    }
}
