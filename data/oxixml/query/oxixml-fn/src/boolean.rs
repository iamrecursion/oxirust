//! The boolean functions of
//! [F&O 3.1 §7](https://www.w3.org/TR/xpath-functions-31/#boolean-functions): `fn:true`,
//! `fn:false`, `fn:boolean` and `fn:not`, plus the `op:boolean-*` comparison operators.

use crate::error::FnResult;
use oxixml_infoset::{AtomicValue, Sequence};
use std::cmp::Ordering;

/// [`fn:true`](https://www.w3.org/TR/xpath-functions-31/#func-true).
#[must_use]
pub fn r#true() -> AtomicValue {
    AtomicValue::boolean(true)
}

/// [`fn:false`](https://www.w3.org/TR/xpath-functions-31/#func-false).
#[must_use]
pub fn r#false() -> AtomicValue {
    AtomicValue::boolean(false)
}

/// [`fn:boolean`](https://www.w3.org/TR/xpath-functions-31/#func-boolean): the *effective boolean
/// value* of a sequence.
///
/// # Errors
///
/// [`FnErrorCode::Forg0006`](crate::FnErrorCode::Forg0006) for a sequence that has no effective
/// boolean value — several atomic items, a singleton `xs:date`, a map or an array.
///
/// # Examples
///
/// ```
/// use oxixml_fn::boolean::boolean;
/// use oxixml_infoset::{AtomicValue, Sequence};
///
/// assert!(!boolean(&Sequence::empty())?);
/// assert!(boolean(&Sequence::from(AtomicValue::string("x")))?);
/// assert!(!boolean(&Sequence::from(AtomicValue::string("")))?);
/// assert!(!boolean(&Sequence::from(AtomicValue::double(f64::NAN)))?);
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn boolean(sequence: &Sequence) -> FnResult<bool> {
    Ok(sequence.effective_boolean_value()?)
}

/// [`fn:not`](https://www.w3.org/TR/xpath-functions-31/#func-not): the negation of the effective
/// boolean value.
///
/// # Errors
///
/// See [`boolean`].
///
/// # Examples
///
/// ```
/// use oxixml_fn::boolean::not;
/// use oxixml_infoset::{AtomicValue, Sequence};
///
/// assert!(not(&Sequence::empty())?);
/// assert!(!not(&Sequence::from(AtomicValue::integer(1_i64)))?);
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn not(sequence: &Sequence) -> FnResult<bool> {
    Ok(!boolean(sequence)?)
}

/// `op:boolean-equal`: `false` sorts before `true`.
#[must_use]
pub fn boolean_equal(left: bool, right: bool) -> bool {
    left == right
}

/// `op:boolean-less-than`: `false` is less than `true`.
#[must_use]
pub fn boolean_less_than(left: bool, right: bool) -> bool {
    !left && right
}

/// `op:boolean-greater-than`: `true` is greater than `false`.
#[must_use]
pub fn boolean_greater_than(left: bool, right: bool) -> bool {
    left && !right
}

/// The total order the three `op:boolean-*` operators induce.
#[must_use]
pub fn compare_booleans(left: bool, right: bool) -> Ordering {
    left.cmp(&right)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::{AtomicType, Item, Tree, XdmMap};

    #[test]
    fn constants() {
        assert_eq!(r#true().string_value(), "true");
        assert_eq!(r#false().string_value(), "false");
    }

    #[test]
    fn effective_boolean_value_rules() -> FnResult<()> {
        assert!(!boolean(&Sequence::empty())?);
        assert!(boolean(&Sequence::from(AtomicValue::boolean(true)))?);
        assert!(!boolean(&Sequence::from(AtomicValue::string("")))?);
        assert!(boolean(&Sequence::from(AtomicValue::string("false")))?);
        assert!(!boolean(&Sequence::from(AtomicValue::integer(0_i64)))?);
        assert!(!boolean(&Sequence::from(AtomicValue::double(f64::NAN)))?);
        Ok(())
    }

    #[test]
    fn nodes_are_always_true() -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str("<a/>")?;
        assert!(boolean(&Sequence::from(tree.root()))?);
        Ok(())
    }

    #[test]
    fn values_without_an_effective_boolean_value_raise() -> Result<(), Box<dyn std::error::Error>> {
        let date = AtomicValue::parse(AtomicType::Date, "2020-01-01")?;
        assert!(boolean(&Sequence::from(date)).is_err());
        assert!(boolean(&Sequence::singleton(Item::from(XdmMap::new()))).is_err());
        Ok(())
    }

    #[test]
    fn not_negates() -> FnResult<()> {
        assert!(not(&Sequence::empty())?);
        assert!(!not(&Sequence::from(AtomicValue::integer(1_i64)))?);
        Ok(())
    }

    #[test]
    fn boolean_operators_order_false_before_true() {
        assert!(boolean_equal(true, true));
        assert!(boolean_less_than(false, true));
        assert!(!boolean_less_than(true, false));
        assert!(boolean_greater_than(true, false));
        assert_eq!(compare_booleans(false, true), Ordering::Less);
    }
}
