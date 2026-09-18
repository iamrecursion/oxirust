//! The higher-order functions of
//! [F&O 3.1 §16](https://www.w3.org/TR/xpath-functions-31/#higher-order-functions):
//! `fn:function-lookup`, `fn:function-name`, `fn:function-arity`, `fn:for-each`, `fn:filter`,
//! `fn:fold-left`, `fn:fold-right`, `fn:for-each-pair`, `fn:apply` and `fn:sort`.
//!
//! The callable they operate on is a [`FunctionItem`], this crate's representation of an XDM
//! function item.

use crate::context::FnContext;
use crate::error::{FnError, FnErrorCode, FnResult};
use crate::function::FunctionItem;
use oxixml_infoset::{ExpandedName, Item, Sequence, XdmArray, compare_atomic_in_context};
use oxixml_xsd::TimezoneOffset;
use std::cmp::Ordering;

/// [`fn:function-name`](https://www.w3.org/TR/xpath-functions-31/#func-function-name).
#[must_use]
pub fn function_name(function: &FunctionItem) -> Option<ExpandedName> {
    function.name().cloned()
}

/// [`fn:function-arity`](https://www.w3.org/TR/xpath-functions-31/#func-function-arity).
#[must_use]
pub fn function_arity(function: &FunctionItem) -> i64 {
    i64::try_from(function.arity()).unwrap_or(i64::MAX)
}

/// [`fn:for-each`](https://www.w3.org/TR/xpath-functions-31/#func-for-each): applies `action` to
/// each item, concatenating the results.
///
/// # Errors
///
/// Whatever `action` raises, or [`FnErrorCode::Xpty0004`] when its arity is not one.
///
/// # Examples
///
/// ```
/// use oxixml_fn::{higher_order::for_each, FnContext, FunctionItem};
/// use oxixml_infoset::{AtomicValue, Sequence};
///
/// let context = FnContext::new();
/// let double = FunctionItem::anonymous(1, |_, arguments| {
///     let value = arguments[0].first().and_then(|item| item.as_atomic())
///         .and_then(|value| value.as_f64()).unwrap_or_default();
///     Ok(Sequence::from(AtomicValue::double(value * 2.0)))
/// });
///
/// let items = Sequence::from_iter([1_i64, 2, 3].map(AtomicValue::integer));
/// let doubled = for_each(&context, &items, &double)?;
/// assert_eq!(doubled.len(), 3);
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn for_each(
    context: &FnContext,
    sequence: &Sequence,
    action: &FunctionItem,
) -> FnResult<Sequence> {
    require_arity(action, 1, "fn:for-each")?;
    let mut out = Sequence::empty();
    for item in sequence.iter() {
        out.append(action.call(context, &[Sequence::singleton(item.clone())])?);
    }
    Ok(out)
}

/// [`fn:filter`](https://www.w3.org/TR/xpath-functions-31/#func-filter): keeps the items for
/// which `predicate` returns `true`.
///
/// # Errors
///
/// Whatever `predicate` raises, [`FnErrorCode::Xpty0004`] when its arity is not one or its
/// result is not a single `xs:boolean`.
pub fn filter(
    context: &FnContext,
    sequence: &Sequence,
    predicate: &FunctionItem,
) -> FnResult<Sequence> {
    require_arity(predicate, 1, "fn:filter")?;
    let mut out = Sequence::empty();
    for item in sequence.iter() {
        let verdict = predicate.call(context, &[Sequence::singleton(item.clone())])?;
        if require_boolean(&verdict, "fn:filter")? {
            out.push(item.clone());
        }
    }
    Ok(out)
}

/// [`fn:fold-left`](https://www.w3.org/TR/xpath-functions-31/#func-fold-left).
///
/// # Errors
///
/// Whatever `action` raises, or [`FnErrorCode::Xpty0004`] when its arity is not two.
///
/// # Examples
///
/// ```
/// use oxixml_fn::{higher_order::fold_left, numeric::{arithmetic, ArithOp}, FnContext, FunctionItem};
/// use oxixml_infoset::{AtomicValue, Sequence};
///
/// let context = FnContext::new();
/// let plus = FunctionItem::anonymous(2, |_, arguments| {
///     let left = arguments[0].first().and_then(|item| item.as_atomic()).cloned()
///         .unwrap_or_else(|| AtomicValue::integer(0_i64));
///     let right = arguments[1].first().and_then(|item| item.as_atomic()).cloned()
///         .unwrap_or_else(|| AtomicValue::integer(0_i64));
///     Ok(Sequence::from(arithmetic(ArithOp::Add, &left, &right)?))
/// });
///
/// let items = Sequence::from_iter([1_i64, 2, 3, 4, 5].map(AtomicValue::integer));
/// let total = fold_left(&context, &items, &Sequence::from(AtomicValue::integer(0_i64)), &plus)?;
/// assert_eq!(total.first().map(ToString::to_string).as_deref(), Some("15"));
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn fold_left(
    context: &FnContext,
    sequence: &Sequence,
    zero: &Sequence,
    action: &FunctionItem,
) -> FnResult<Sequence> {
    require_arity(action, 2, "fn:fold-left")?;
    let mut accumulator = zero.clone();
    for item in sequence.iter() {
        accumulator = action.call(context, &[accumulator, Sequence::singleton(item.clone())])?;
    }
    Ok(accumulator)
}

/// [`fn:fold-right`](https://www.w3.org/TR/xpath-functions-31/#func-fold-right).
///
/// # Errors
///
/// Whatever `action` raises, or [`FnErrorCode::Xpty0004`] when its arity is not two.
pub fn fold_right(
    context: &FnContext,
    sequence: &Sequence,
    zero: &Sequence,
    action: &FunctionItem,
) -> FnResult<Sequence> {
    require_arity(action, 2, "fn:fold-right")?;
    let mut accumulator = zero.clone();
    for item in sequence.iter().rev() {
        accumulator = action.call(context, &[Sequence::singleton(item.clone()), accumulator])?;
    }
    Ok(accumulator)
}

/// [`fn:for-each-pair`](https://www.w3.org/TR/xpath-functions-31/#func-for-each-pair): applies
/// `action` to corresponding items, stopping at the shorter sequence.
///
/// # Errors
///
/// Whatever `action` raises, or [`FnErrorCode::Xpty0004`] when its arity is not two.
pub fn for_each_pair(
    context: &FnContext,
    left: &Sequence,
    right: &Sequence,
    action: &FunctionItem,
) -> FnResult<Sequence> {
    require_arity(action, 2, "fn:for-each-pair")?;
    let mut out = Sequence::empty();
    for (first, second) in left.iter().zip(right.iter()) {
        out.append(action.call(
            context,
            &[
                Sequence::singleton(first.clone()),
                Sequence::singleton(second.clone()),
            ],
        )?);
    }
    Ok(out)
}

/// [`fn:apply`](https://www.w3.org/TR/xpath-functions-31/#func-apply): calls `function` with the
/// members of `arguments` as its argument list.
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] when the array's size is not the function's arity, plus whatever
/// the function itself raises.
pub fn apply(
    context: &FnContext,
    function: &FunctionItem,
    arguments: &XdmArray,
) -> FnResult<Sequence> {
    if arguments.len() != function.arity() {
        return Err(FnError::new(
            FnErrorCode::Xpty0004,
            format!(
                "fn:apply was given {} argument(s) for {}",
                arguments.len(),
                function.describe()
            ),
        ));
    }
    let arguments: Vec<Sequence> = arguments.members().cloned().collect();
    function.call(context, &arguments)
}

/// [`fn:sort`](https://www.w3.org/TR/xpath-functions-31/#func-sort): a stable sort of `sequence`
/// by the atomized value of `key`, or by the items themselves when `key` is `None`.
///
/// # Errors
///
/// Whatever `key` raises, and [`FnErrorCode::Xpty0004`] when two keys cannot be ordered.
///
/// # Examples
///
/// ```
/// use oxixml_fn::{higher_order::sort, FnContext};
/// use oxixml_infoset::{AtomicValue, Sequence};
///
/// let context = FnContext::new();
/// let items = Sequence::from_iter([3_i64, 1, 2].map(AtomicValue::integer));
/// let sorted = sort(&context, &items, None, None)?;
/// assert_eq!(
///     sorted.iter().map(ToString::to_string).collect::<Vec<_>>(),
///     ["1", "2", "3"]
/// );
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn sort(
    context: &FnContext,
    sequence: &Sequence,
    collation: Option<&str>,
    key: Option<&FunctionItem>,
) -> FnResult<Sequence> {
    if let Some(key) = key {
        require_arity(key, 1, "fn:sort")?;
    }
    let collation = context.collation(collation)?;
    let implicit_timezone = context.implicit_timezone();
    let mut keyed: Vec<(Sequence, Item)> = Vec::with_capacity(sequence.len());
    for item in sequence.iter() {
        let raw = match key {
            None => Sequence::singleton(item.clone()),
            Some(key) => key.call(context, &[Sequence::singleton(item.clone())])?,
        };
        keyed.push((raw.atomize()?, item.clone()));
    }

    let mut failure = None;
    keyed.sort_by(|(left, _), (right, _)| {
        match compare_keys(left, right, collation.as_ref(), implicit_timezone) {
            Ok(ordering) => ordering,
            Err(error) => {
                failure.get_or_insert(error);
                Ordering::Equal
            }
        }
    });
    if let Some(error) = failure {
        return Err(error);
    }
    Ok(Sequence::from_items(
        keyed.into_iter().map(|(_, item)| item),
    ))
}

fn compare_keys(
    left: &Sequence,
    right: &Sequence,
    collation: &dyn crate::collation::Collation,
    implicit_timezone: TimezoneOffset,
) -> FnResult<Ordering> {
    let mut left_items = left.iter();
    let mut right_items = right.iter();
    loop {
        match (left_items.next(), right_items.next()) {
            (None, None) => return Ok(Ordering::Equal),
            (None, Some(_)) => return Ok(Ordering::Less),
            (Some(_), None) => return Ok(Ordering::Greater),
            (Some(left), Some(right)) => {
                let (Some(left), Some(right)) = (left.as_atomic(), right.as_atomic()) else {
                    return Err(FnError::new(
                        FnErrorCode::Xpty0004,
                        "fn:sort keys must atomize to atomic values",
                    ));
                };
                let ordering = if let (Some(left), Some(right)) = (left.as_str(), right.as_str()) {
                    collation.compare(left, right)
                } else if left.is_nan() && right.is_nan() {
                    Ordering::Equal
                } else if left.is_nan() {
                    Ordering::Less
                } else if right.is_nan() {
                    Ordering::Greater
                } else {
                    // F&O 3.1 §10.4.4: a date/time key with no timezone is read in the dynamic
                    // context's implicit timezone before comparing.
                    compare_atomic_in_context(left, right, None, Some(implicit_timezone))?
                        .ok_or_else(|| {
                            FnError::new(
                                FnErrorCode::Xpty0004,
                                "fn:sort keys are not ordered with respect to one another",
                            )
                        })?
                };
                if ordering != Ordering::Equal {
                    return Ok(ordering);
                }
            }
        }
    }
}

fn require_arity(function: &FunctionItem, arity: usize, what: &str) -> FnResult<()> {
    if function.arity() == arity {
        Ok(())
    } else {
        Err(FnError::new(
            FnErrorCode::Xpty0004,
            format!(
                "{what} expects a function of arity {arity}, but {} was supplied",
                function.describe()
            ),
        ))
    }
}

/// Reads the `xs:boolean` a predicate is declared to return.
///
/// `fn:filter`'s `$f` is declared `function(item()) as xs:boolean`, so a function supplied for it
/// is *coerced* (XPath 3.1 §3.1.5.2): the function conversion rules are applied to what its body
/// produces, exactly as they are to an argument. That is what lets a predicate return a node —
/// the node is atomized, and the resulting `xs:untypedAtomic` is cast to `xs:boolean`, so
/// `filter(1 to 20, function($i) { <a>{1 - ($i mod 2)}</a> })` keeps the even numbers. A value
/// that survives neither step is `XPTY0004`, and an untyped one whose lexical form is not a
/// boolean is `FORG0001`.
pub(crate) fn require_boolean(result: &Sequence, what: &str) -> FnResult<bool> {
    use crate::signature::{Occurrence, ParamItemType, ParamType, convert_argument};
    use oxixml_infoset::AtomicType;

    const REQUIRED: ParamType =
        ParamType::new(ParamItemType::Atomic(AtomicType::Boolean), Occurrence::One);
    let converted = convert_argument(result, &REQUIRED, what)?;
    match converted.as_slice() {
        [Item::Atomic(value)] => value.as_bool().ok_or_else(|| {
            FnError::new(
                FnErrorCode::Xpty0004,
                format!("{what} expects its predicate to return one xs:boolean"),
            )
        }),
        _ => Err(FnError::new(
            FnErrorCode::Xpty0004,
            format!("{what} expects its predicate to return one xs:boolean"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{ArithOp, arithmetic};
    use oxixml_infoset::AtomicValue;

    fn first_atomic(arguments: &[Sequence], index: usize) -> AtomicValue {
        arguments
            .get(index)
            .and_then(Sequence::first)
            .and_then(Item::as_atomic)
            .cloned()
            .unwrap_or_else(|| AtomicValue::integer(0_i64))
    }

    fn plus() -> FunctionItem {
        FunctionItem::anonymous(2, |_, arguments| {
            let left = first_atomic(arguments, 0);
            let right = first_atomic(arguments, 1);
            Ok(Sequence::from(arithmetic(ArithOp::Add, &left, &right)?))
        })
    }

    fn is_even() -> FunctionItem {
        FunctionItem::anonymous(1, |_, arguments| {
            let value = first_atomic(arguments, 0).as_f64().unwrap_or_default();
            Ok(Sequence::from(AtomicValue::boolean(value % 2.0 == 0.0)))
        })
    }

    fn integers(values: &[i64]) -> Sequence {
        Sequence::from_iter(values.iter().copied().map(AtomicValue::integer))
    }

    fn rendered(sequence: &Sequence) -> String {
        sequence
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }

    #[test]
    fn for_each_examples_from_the_specification() -> FnResult<()> {
        let context = FnContext::new();
        let square = FunctionItem::anonymous(1, |_, arguments| {
            let value = first_atomic(arguments, 0);
            Ok(Sequence::from(arithmetic(
                ArithOp::Multiply,
                &value,
                &value,
            )?))
        });
        assert_eq!(
            rendered(&for_each(&context, &integers(&[1, 2, 3]), &square)?),
            "1,4,9"
        );
        Ok(())
    }

    #[test]
    fn filter_examples_from_the_specification() -> FnResult<()> {
        let context = FnContext::new();
        assert_eq!(
            rendered(&filter(
                &context,
                &integers(&[1, 2, 3, 4, 5, 6]),
                &is_even()
            )?),
            "2,4,6"
        );
        Ok(())
    }

    /// A predicate whose body yields an untyped node: the node is atomized and the resulting
    /// `xs:untypedAtomic` is cast to `xs:boolean`, so `<a>1</a>` keeps the item and `<a>0</a>`
    /// drops it.
    #[test]
    fn filter_accepts_a_predicate_that_returns_an_untyped_node() -> FnResult<()> {
        let context = FnContext::new();
        let keeps_long_words = FunctionItem::anonymous(1, |_, arguments| {
            let flag = usize::from(first_atomic(arguments, 0).string_value().chars().count() > 3);
            let tree = oxixml_infoset::Tree::parse_str(&format!("<a>{flag}</a>"))
                .map_err(|error| FnError::new(FnErrorCode::Xpty0004, error.to_string()))?;
            let root = tree
                .document_element()
                .ok_or_else(|| FnError::new(FnErrorCode::Xpty0004, "no document element"))?;
            Ok(Sequence::from(root))
        });
        let words = Sequence::from_iter(
            ["red", "green", "blue", "on"]
                .into_iter()
                .map(AtomicValue::string),
        );
        assert_eq!(
            rendered(&filter(&context, &words, &keeps_long_words)?),
            "green,blue"
        );
        Ok(())
    }

    /// A predicate whose result is neither an `xs:boolean` nor untyped data is still
    /// `XPTY0004`: the conversion rules promote nothing to a boolean.
    #[test]
    fn filter_rejects_a_predicate_that_returns_a_number() {
        let context = FnContext::new();
        let counts =
            FunctionItem::anonymous(1, |_, _| Ok(Sequence::from(AtomicValue::integer(1_i64))));
        let error = filter(&context, &integers(&[1]), &counts).expect_err("not a boolean");
        assert!(error.is(FnErrorCode::Xpty0004));
    }

    /// `fn-sort-collation-4`/`-5` and `array-sort-collation-4`/`-5` sort mixed-case words under
    /// the case-blind collation named by their explicit `$collation` argument.
    #[test]
    fn sort_honours_an_explicit_caseblind_collation() -> FnResult<()> {
        let context = FnContext::new();
        let words = Sequence::from_iter(
            ["Red", "green", "blUE", "PINK", "ORanGE"]
                .into_iter()
                .map(AtomicValue::string),
        );
        assert_eq!(
            rendered(&sort(
                &context,
                &words,
                Some(crate::collation::CASEBLIND_COLLATION),
                None
            )?),
            "blUE,green,ORanGE,PINK,Red"
        );
        // The default collation is untouched: under code points the capitals still lead.
        assert_eq!(
            rendered(&sort(&context, &words, None, None)?),
            "ORanGE,PINK,Red,blUE,green"
        );
        Ok(())
    }

    #[test]
    fn folds_examples_from_the_specification() -> FnResult<()> {
        let context = FnContext::new();
        let items = integers(&[1, 2, 3, 4, 5]);
        let zero = Sequence::from(AtomicValue::integer(0_i64));
        assert_eq!(
            rendered(&fold_left(&context, &items, &zero, &plus())?),
            "15"
        );
        assert_eq!(
            rendered(&fold_right(&context, &items, &zero, &plus())?),
            "15"
        );

        // fold-right differs from fold-left when the operation is not associative.
        let subtract = FunctionItem::anonymous(2, |_, arguments| {
            let left = first_atomic(arguments, 0);
            let right = first_atomic(arguments, 1);
            Ok(Sequence::from(arithmetic(
                ArithOp::Subtract,
                &left,
                &right,
            )?))
        });
        assert_eq!(
            rendered(&fold_left(
                &context,
                &integers(&[1, 2, 3]),
                &zero,
                &subtract
            )?),
            "-6"
        );
        assert_eq!(
            rendered(&fold_right(
                &context,
                &integers(&[1, 2, 3]),
                &zero,
                &subtract
            )?),
            "2"
        );
        Ok(())
    }

    #[test]
    fn for_each_pair_stops_at_the_shorter_sequence() -> FnResult<()> {
        let context = FnContext::new();
        assert_eq!(
            rendered(&for_each_pair(
                &context,
                &integers(&[1, 2, 3]),
                &integers(&[10, 20]),
                &plus()
            )?),
            "11,22"
        );
        Ok(())
    }

    #[test]
    fn apply_spreads_an_array_over_the_argument_list() -> FnResult<()> {
        let context = FnContext::new();
        let arguments = XdmArray::from_members([
            Sequence::from(AtomicValue::integer(2_i64)),
            Sequence::from(AtomicValue::integer(3_i64)),
        ]);
        assert_eq!(rendered(&apply(&context, &plus(), &arguments)?), "5");

        let wrong = XdmArray::from_members([Sequence::from(AtomicValue::integer(2_i64))]);
        assert!(apply(&context, &plus(), &wrong).is_err());
        Ok(())
    }

    #[test]
    fn arity_mismatches_are_type_errors() {
        let context = FnContext::new();
        assert!(for_each(&context, &integers(&[1]), &plus()).is_err());
        assert!(fold_left(&context, &integers(&[1]), &Sequence::empty(), &is_even()).is_err());
    }

    #[test]
    fn sort_is_stable_and_key_driven() -> FnResult<()> {
        let context = FnContext::new();
        assert_eq!(
            rendered(&sort(&context, &integers(&[3, 1, 2]), None, None)?),
            "1,2,3"
        );

        let negated = FunctionItem::anonymous(1, |_, arguments| {
            let value = first_atomic(arguments, 0);
            Ok(Sequence::from(crate::numeric::unary_minus(&value)?))
        });
        assert_eq!(
            rendered(&sort(
                &context,
                &integers(&[3, 1, 2]),
                None,
                Some(&negated)
            )?),
            "3,2,1"
        );

        let words = Sequence::from_iter(["banana", "apple", "cherry"].map(AtomicValue::string));
        assert_eq!(
            rendered(&sort(&context, &words, None, None)?),
            "apple,banana,cherry"
        );
        Ok(())
    }

    #[test]
    fn function_name_and_arity() {
        let named = FunctionItem::new(
            ExpandedName::new(Some("http://www.w3.org/2005/xpath-functions"), "abs"),
            1,
            |_, arguments| Ok(arguments[0].clone()),
        );
        assert_eq!(function_arity(&named), 1);
        assert_eq!(
            function_name(&named).map(|name| name.local_name().to_owned()),
            Some("abs".to_owned())
        );
        assert!(function_name(&plus()).is_none());
    }
}
