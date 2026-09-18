//! The `array:` functions of
//! [F&O 3.1 §17.3](https://www.w3.org/TR/xpath-functions-31/#array-functions), which live in the
//! namespace `http://www.w3.org/2005/xpath-functions/array`.
//!
//! Array positions are **one-based** throughout, and an out-of-range position raises
//! `FOAY0001` rather than yielding the empty sequence — the difference between an array, which
//! is a function from `1..size`, and a map, which is a partial function.

use crate::context::FnContext;
use crate::error::{FnError, FnErrorCode, FnResult};
use crate::function::FunctionItem;
use crate::higher_order;
use oxixml_infoset::{Item, Sequence, XdmArray};

/// The namespace name of the `array:` functions.
pub const ARRAY_NAMESPACE: &str = "http://www.w3.org/2005/xpath-functions/array";

fn out_of_bounds(position: i64, size: usize, what: &str) -> FnError {
    FnError::new(
        FnErrorCode::Foay0001,
        format!("{what}: position {position} is outside an array of {size} member(s)"),
    )
}

fn checked_position(position: i64, size: usize, what: &str) -> FnResult<usize> {
    usize::try_from(position)
        .ok()
        .filter(|position| (1..=size).contains(position))
        .ok_or_else(|| out_of_bounds(position, size, what))
}

/// [`array:size`](https://www.w3.org/TR/xpath-functions-31/#func-array-size).
#[must_use]
pub fn size(array: &XdmArray) -> i64 {
    i64::try_from(array.len()).unwrap_or(i64::MAX)
}

/// [`array:get`](https://www.w3.org/TR/xpath-functions-31/#func-array-get).
///
/// # Errors
///
/// [`FnErrorCode::Foay0001`] when `position` is outside `1..=size`.
///
/// # Examples
///
/// ```
/// use oxixml_fn::arrays::get;
/// use oxixml_infoset::{AtomicValue, XdmArray};
///
/// let array = XdmArray::from_iter([AtomicValue::string("a"), AtomicValue::string("b")]);
/// assert_eq!(get(&array, 2)?.first().map(ToString::to_string), Some("b".to_owned()));
/// assert!(get(&array, 3).is_err());
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn get(array: &XdmArray, position: i64) -> FnResult<Sequence> {
    let position = checked_position(position, array.len(), "array:get")?;
    Ok(array.get(position)?.clone())
}

/// [`array:put`](https://www.w3.org/TR/xpath-functions-31/#func-array-put).
///
/// # Errors
///
/// [`FnErrorCode::Foay0001`] when `position` is outside `1..=size`.
pub fn put(array: &XdmArray, position: i64, member: Sequence) -> FnResult<XdmArray> {
    let position = checked_position(position, array.len(), "array:put")?;
    Ok(array.clone().put(position, member)?)
}

/// [`array:append`](https://www.w3.org/TR/xpath-functions-31/#func-array-append).
#[must_use]
pub fn append(array: &XdmArray, member: Sequence) -> XdmArray {
    array.clone().append(member)
}

/// [`array:subarray`](https://www.w3.org/TR/xpath-functions-31/#func-array-subarray), the
/// three-argument form; pass `None` for `length` to take everything from `start` onwards.
///
/// # Errors
///
/// [`FnErrorCode::Foay0001`] when `start` is outside `1..=size+1` or the range runs off the end,
/// and [`FnErrorCode::Foay0002`] when `length` is negative.
///
/// # Examples
///
/// ```
/// use oxixml_fn::arrays::subarray;
/// use oxixml_infoset::{AtomicValue, XdmArray};
///
/// let array = XdmArray::from_iter(["a", "b", "c", "d"].map(AtomicValue::string));
/// assert_eq!(subarray(&array, 2, Some(0))?.len(), 0);
/// assert_eq!(subarray(&array, 2, Some(2))?.len(), 2);
/// assert_eq!(subarray(&array, 2, None)?.len(), 3);
/// assert_eq!(subarray(&array, 5, Some(0))?.len(), 0);
/// assert!(subarray(&array, 2, Some(-1)).is_err());
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn subarray(array: &XdmArray, start: i64, length: Option<i64>) -> FnResult<XdmArray> {
    let size = array.len();
    let start_index = usize::try_from(start)
        .ok()
        .filter(|start| (1..=size + 1).contains(start))
        .ok_or_else(|| out_of_bounds(start, size, "array:subarray"))?;
    let available = size - (start_index - 1);
    let length = match length {
        None => available,
        Some(length) if length < 0 => {
            return Err(FnError::new(
                FnErrorCode::Foay0002,
                format!("array:subarray was given the negative length {length}"),
            ));
        }
        Some(length) => {
            let length = usize::try_from(length).unwrap_or(usize::MAX);
            if length > available {
                return Err(out_of_bounds(
                    start + i64::try_from(length).unwrap_or(0),
                    size,
                    "array:subarray",
                ));
            }
            length
        }
    };
    Ok(array.subarray(start_index, length)?)
}

/// [`array:remove`](https://www.w3.org/TR/xpath-functions-31/#func-array-remove): removes the
/// members at the given one-based positions.
///
/// # Errors
///
/// [`FnErrorCode::Foay0001`] when any position is outside `1..=size`.
pub fn remove(array: &XdmArray, positions: &[i64]) -> FnResult<XdmArray> {
    let size = array.len();
    let mut sorted: Vec<usize> = positions
        .iter()
        .map(|position| checked_position(*position, size, "array:remove"))
        .collect::<FnResult<Vec<usize>>>()?;
    sorted.sort_unstable();
    sorted.dedup();
    let mut result = array.clone();
    for position in sorted.into_iter().rev() {
        result = result.remove(position)?;
    }
    Ok(result)
}

/// [`array:insert-before`](https://www.w3.org/TR/xpath-functions-31/#func-array-insert-before).
///
/// # Errors
///
/// [`FnErrorCode::Foay0001`] when `position` is outside `1..=size+1`.
pub fn insert_before(array: &XdmArray, position: i64, member: Sequence) -> FnResult<XdmArray> {
    let size = array.len();
    let position = usize::try_from(position)
        .ok()
        .filter(|position| (1..=size + 1).contains(position))
        .ok_or_else(|| out_of_bounds(position, size, "array:insert-before"))?;
    Ok(array.clone().insert_before(position, member)?)
}

/// [`array:head`](https://www.w3.org/TR/xpath-functions-31/#func-array-head).
///
/// # Errors
///
/// [`FnErrorCode::Foay0001`] for the empty array.
pub fn head(array: &XdmArray) -> FnResult<Sequence> {
    Ok(array.head()?.clone())
}

/// [`array:tail`](https://www.w3.org/TR/xpath-functions-31/#func-array-tail).
///
/// # Errors
///
/// [`FnErrorCode::Foay0001`] for the empty array.
pub fn tail(array: &XdmArray) -> FnResult<XdmArray> {
    Ok(array.tail()?)
}

/// [`array:reverse`](https://www.w3.org/TR/xpath-functions-31/#func-array-reverse).
#[must_use]
pub fn reverse(array: &XdmArray) -> XdmArray {
    array.clone().reverse()
}

/// [`array:join`](https://www.w3.org/TR/xpath-functions-31/#func-array-join).
#[must_use]
pub fn join(arrays: &[XdmArray]) -> XdmArray {
    XdmArray::join(arrays.iter().cloned())
}

/// [`array:flatten`](https://www.w3.org/TR/xpath-functions-31/#func-array-flatten): replaces
/// every array in the input, recursively, by its members.
#[must_use]
pub fn flatten(sequence: &Sequence) -> Sequence {
    let mut out = Sequence::empty();
    for item in sequence.iter() {
        match item {
            Item::Array(array) => out.append(array.flatten()),
            other => out.push(other.clone()),
        }
    }
    out
}

/// [`array:for-each`](https://www.w3.org/TR/xpath-functions-31/#func-array-for-each).
///
/// # Errors
///
/// Whatever `action` raises, or [`FnErrorCode::Xpty0004`] when its arity is not one.
pub fn for_each(
    context: &FnContext,
    array: &XdmArray,
    action: &FunctionItem,
) -> FnResult<XdmArray> {
    require_arity(action, 1, "array:for-each")?;
    let mut members = Vec::with_capacity(array.len());
    for member in array.members() {
        members.push(action.call(context, std::slice::from_ref(member))?);
    }
    Ok(XdmArray::from_members(members))
}

/// [`array:filter`](https://www.w3.org/TR/xpath-functions-31/#func-array-filter).
///
/// # Errors
///
/// Whatever `predicate` raises, or [`FnErrorCode::Xpty0004`] when its arity is not one or its
/// result is not one `xs:boolean`.
pub fn filter(
    context: &FnContext,
    array: &XdmArray,
    predicate: &FunctionItem,
) -> FnResult<XdmArray> {
    require_arity(predicate, 1, "array:filter")?;
    let mut members = Vec::new();
    for member in array.members() {
        let verdict = predicate.call(context, std::slice::from_ref(member))?;
        if boolean_result(&verdict, "array:filter")? {
            members.push(member.clone());
        }
    }
    Ok(XdmArray::from_members(members))
}

/// [`array:fold-left`](https://www.w3.org/TR/xpath-functions-31/#func-array-fold-left).
///
/// # Errors
///
/// Whatever `action` raises, or [`FnErrorCode::Xpty0004`] when its arity is not two.
pub fn fold_left(
    context: &FnContext,
    array: &XdmArray,
    zero: &Sequence,
    action: &FunctionItem,
) -> FnResult<Sequence> {
    require_arity(action, 2, "array:fold-left")?;
    let mut accumulator = zero.clone();
    for member in array.members() {
        accumulator = action.call(context, &[accumulator, member.clone()])?;
    }
    Ok(accumulator)
}

/// [`array:fold-right`](https://www.w3.org/TR/xpath-functions-31/#func-array-fold-right).
///
/// # Errors
///
/// Whatever `action` raises, or [`FnErrorCode::Xpty0004`] when its arity is not two.
pub fn fold_right(
    context: &FnContext,
    array: &XdmArray,
    zero: &Sequence,
    action: &FunctionItem,
) -> FnResult<Sequence> {
    require_arity(action, 2, "array:fold-right")?;
    let mut accumulator = zero.clone();
    for member in array.members().rev() {
        accumulator = action.call(context, &[member.clone(), accumulator])?;
    }
    Ok(accumulator)
}

/// [`array:for-each-pair`](https://www.w3.org/TR/xpath-functions-31/#func-array-for-each-pair),
/// which stops at the shorter array.
///
/// # Errors
///
/// Whatever `action` raises, or [`FnErrorCode::Xpty0004`] when its arity is not two.
pub fn for_each_pair(
    context: &FnContext,
    left: &XdmArray,
    right: &XdmArray,
    action: &FunctionItem,
) -> FnResult<XdmArray> {
    require_arity(action, 2, "array:for-each-pair")?;
    let mut members = Vec::new();
    for (first, second) in left.members().zip(right.members()) {
        members.push(action.call(context, &[first.clone(), second.clone()])?);
    }
    Ok(XdmArray::from_members(members))
}

/// [`array:sort`](https://www.w3.org/TR/xpath-functions-31/#func-array-sort).
///
/// # Errors
///
/// Whatever `key` raises, and [`FnErrorCode::Xpty0004`] when two keys cannot be ordered.
pub fn sort(
    context: &FnContext,
    array: &XdmArray,
    collation: Option<&str>,
    key: Option<&FunctionItem>,
) -> FnResult<XdmArray> {
    // Sorting an array is sorting the sequence of its members, so reuse fn:sort by boxing each
    // member into a one-item array and unwrapping afterwards.
    let boxed = Sequence::from_items(
        array
            .members()
            .map(|member| Item::from(XdmArray::from_members([member.clone()]))),
    );
    let unwrapping_key = match key {
        None => FunctionItem::anonymous(1, |_, arguments| {
            Ok(single_member(&arguments[0]).unwrap_or_default())
        }),
        Some(key) => {
            let key = key.clone();
            FunctionItem::anonymous(1, move |context, arguments| {
                let member = single_member(&arguments[0]).unwrap_or_default();
                key.call(context, &[member])
            })
        }
    };
    let sorted = higher_order::sort(context, &boxed, collation, Some(&unwrapping_key))?;
    let members: Vec<Sequence> = sorted
        .iter()
        .map(|item| single_member(&Sequence::singleton(item.clone())).unwrap_or_default())
        .collect();
    Ok(XdmArray::from_members(members))
}

fn single_member(sequence: &Sequence) -> Option<Sequence> {
    match sequence.as_slice() {
        [Item::Array(array)] => array.get_at(0).cloned(),
        _ => None,
    }
}

fn require_arity(function: &FunctionItem, arity: usize, what: &str) -> FnResult<()> {
    if function.arity() == arity {
        Ok(())
    } else {
        Err(FnError::new(
            FnErrorCode::Xpty0004,
            format!("{what} expects a function of arity {arity}"),
        ))
    }
}

/// `array:filter`'s `$f` is declared `function(item()*) as xs:boolean`, so its result is put
/// through the function conversion rules exactly as `fn:filter`'s is: atomized, and an
/// `xs:untypedAtomic` cast to `xs:boolean`.
fn boolean_result(result: &Sequence, what: &str) -> FnResult<bool> {
    crate::higher_order::require_boolean(result, what)
}

/// Extracts the array an argument declared as `array(*)` must be.
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] when the sequence is not exactly one array.
pub fn require_array(sequence: &Sequence, what: &str) -> FnResult<XdmArray> {
    match sequence.as_slice() {
        [Item::Array(array)] => Ok(array.clone()),
        _ => Err(FnError::new(
            FnErrorCode::Xpty0004,
            format!("{what} expects exactly one array"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::AtomicValue;

    fn letters() -> XdmArray {
        XdmArray::from_iter(["a", "b", "c", "d"].map(AtomicValue::string))
    }

    fn rendered(array: &XdmArray) -> String {
        array
            .members()
            .map(|member| {
                member
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("")
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    #[test]
    fn size_and_get() -> FnResult<()> {
        let array = letters();
        assert_eq!(size(&array), 4);
        assert_eq!(
            get(&array, 2)?.first().map(ToString::to_string),
            Some("b".to_owned())
        );
        assert!(get(&array, 0).expect_err("zero").is(FnErrorCode::Foay0001));
        assert!(
            get(&array, 5)
                .expect_err("past the end")
                .is(FnErrorCode::Foay0001)
        );
        assert_eq!(size(&XdmArray::new()), 0);
        Ok(())
    }

    #[test]
    fn put_append_insert_and_remove() -> FnResult<()> {
        let array = letters();
        assert_eq!(
            rendered(&put(&array, 2, Sequence::from(AtomicValue::string("z")))?),
            "a,z,c,d"
        );
        assert_eq!(
            rendered(&append(&array, Sequence::from(AtomicValue::string("e")))),
            "a,b,c,d,e"
        );
        assert_eq!(
            rendered(&insert_before(
                &array,
                1,
                Sequence::from(AtomicValue::string("z"))
            )?),
            "z,a,b,c,d"
        );
        assert_eq!(
            rendered(&insert_before(
                &array,
                5,
                Sequence::from(AtomicValue::string("z"))
            )?),
            "a,b,c,d,z"
        );
        assert_eq!(rendered(&remove(&array, &[1, 3])?), "b,d");
        assert!(remove(&array, &[9]).is_err());
        assert!(put(&array, 9, Sequence::empty()).is_err());
        Ok(())
    }

    #[test]
    fn subarray_examples_from_the_specification() -> FnResult<()> {
        let array = letters();
        assert_eq!(rendered(&subarray(&array, 2, Some(0))?), "");
        assert_eq!(rendered(&subarray(&array, 2, Some(1))?), "b");
        assert_eq!(rendered(&subarray(&array, 2, Some(2))?), "b,c");
        assert_eq!(rendered(&subarray(&array, 2, None)?), "b,c,d");
        assert_eq!(rendered(&subarray(&array, 5, Some(0))?), "");
        assert!(
            subarray(&array, 2, Some(-1))
                .expect_err("negative")
                .is(FnErrorCode::Foay0002)
        );
        assert!(subarray(&array, 0, Some(1)).is_err());
        assert!(subarray(&array, 2, Some(4)).is_err());
        Ok(())
    }

    #[test]
    fn head_tail_reverse_and_join() -> FnResult<()> {
        let array = letters();
        assert_eq!(
            head(&array)?.first().map(ToString::to_string),
            Some("a".to_owned())
        );
        assert_eq!(rendered(&tail(&array)?), "b,c,d");
        assert_eq!(rendered(&reverse(&array)), "d,c,b,a");
        assert_eq!(rendered(&join(&[])), "");
        assert_eq!(
            rendered(&join(&[array.clone(), XdmArray::new()])),
            "a,b,c,d"
        );
        assert!(head(&XdmArray::new()).is_err());
        Ok(())
    }

    #[test]
    fn flatten_is_recursive() {
        let inner = XdmArray::from_iter([AtomicValue::integer(2_i64)]);
        let outer = XdmArray::from_members([
            Sequence::from(AtomicValue::integer(1_i64)),
            Sequence::singleton(Item::from(inner)),
        ]);
        let flattened = flatten(&Sequence::singleton(Item::from(outer)));
        assert_eq!(flattened.len(), 2);
    }

    #[test]
    fn higher_order_array_functions() -> FnResult<()> {
        let context = FnContext::new();
        let array = XdmArray::from_iter([1_i64, 2, 3, 4].map(AtomicValue::integer));

        let doubled = for_each(
            &context,
            &array,
            &FunctionItem::anonymous(1, |_, arguments| {
                let value = arguments[0]
                    .first()
                    .and_then(Item::as_atomic)
                    .and_then(oxixml_infoset::AtomicValue::as_f64)
                    .unwrap_or_default();
                Ok(Sequence::from(AtomicValue::double(value * 2.0)))
            }),
        )?;
        assert_eq!(doubled.len(), 4);

        let evens = filter(
            &context,
            &array,
            &FunctionItem::anonymous(1, |_, arguments| {
                let value = arguments[0]
                    .first()
                    .and_then(Item::as_atomic)
                    .and_then(oxixml_infoset::AtomicValue::as_f64)
                    .unwrap_or_default();
                Ok(Sequence::from(AtomicValue::boolean(value % 2.0 == 0.0)))
            }),
        )?;
        assert_eq!(rendered(&evens), "2,4");

        let concat = FunctionItem::anonymous(2, |_, arguments| {
            let mut joined = arguments[0].clone();
            joined.append(arguments[1].clone());
            Ok(joined)
        });
        assert_eq!(
            fold_left(&context, &array, &Sequence::empty(), &concat)?.len(),
            4
        );
        assert_eq!(
            fold_right(&context, &array, &Sequence::empty(), &concat)?.len(),
            4
        );
        assert_eq!(
            for_each_pair(&context, &array, &letters(), &concat)?.len(),
            4
        );
        Ok(())
    }

    #[test]
    fn sort_orders_members() -> FnResult<()> {
        let context = FnContext::new();
        let array = XdmArray::from_iter([3_i64, 1, 2].map(AtomicValue::integer));
        assert_eq!(rendered(&sort(&context, &array, None, None)?), "1,2,3");

        let negated = FunctionItem::anonymous(1, |_, arguments| {
            let value = arguments[0]
                .first()
                .and_then(Item::as_atomic)
                .cloned()
                .unwrap_or_else(|| AtomicValue::integer(0_i64));
            Ok(Sequence::from(crate::numeric::unary_minus(&value)?))
        });
        assert_eq!(
            rendered(&sort(&context, &array, None, Some(&negated))?),
            "3,2,1"
        );
        Ok(())
    }

    /// `array-sort-collation-4`/`-5` and `fn-function-lookup-800` sort mixed-case words under
    /// the case-blind collation named by an explicit `$collation` argument.
    #[test]
    fn sort_honours_an_explicit_caseblind_collation() -> FnResult<()> {
        let context = FnContext::new();
        let array = XdmArray::from_iter(
            ["Red", "green", "blUE", "PINK", "ORanGE"]
                .into_iter()
                .map(AtomicValue::string),
        );
        assert_eq!(
            rendered(&sort(
                &context,
                &array,
                Some(crate::collation::CASEBLIND_COLLATION),
                None
            )?),
            "blUE,green,ORanGE,PINK,Red"
        );
        Ok(())
    }

    #[test]
    fn require_array_rejects_other_shapes() {
        let array = Sequence::singleton(Item::from(letters()));
        assert!(require_array(&array, "array:size").is_ok());
        assert!(require_array(&Sequence::empty(), "array:size").is_err());
    }
}
