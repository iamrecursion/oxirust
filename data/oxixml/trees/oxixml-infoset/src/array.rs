//! [`XdmArray`] — the XDM 3.1 array, a function item from a 1-based position to a sequence.

use crate::atomic::AtomicValue;
use crate::error::{ErrorCode, XdmError};
use crate::item::Item;
use crate::sequence::Sequence;
use std::slice::Iter;

/// An XDM 3.1 [array](https://www.w3.org/TR/xpath-datamodel-31/#dt-array): an ordered list of
/// *members*, each of which is a whole [`Sequence`].
///
/// The distinction from a sequence is exactly this nesting: `[(1, 2), 3]` is an array of two
/// members, the first of which is a two-item sequence, whereas `(1, 2, 3)` is a flat sequence of
/// three items. That is why an array is an *item* while a sequence is not.
///
/// Positions are **1-based** throughout, matching `array:get` and friends; the `_at` methods
/// take 0-based indices for Rust-side convenience.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{AtomicValue, Item, Sequence, XdmArray};
///
/// // [(1, 2), 3]
/// let array = XdmArray::from_members([
///     Sequence::from_items([
///         Item::from(AtomicValue::integer(1_i64)),
///         Item::from(AtomicValue::integer(2_i64)),
///     ]),
///     Sequence::from(AtomicValue::integer(3_i64)),
/// ]);
///
/// assert_eq!(array.len(), 2);
/// assert_eq!(array.get(1)?.len(), 2);
/// assert_eq!(array.flatten().len(), 3);
/// assert!(array.get(3).is_err());
///
/// // Every mutator has a persistent counterpart that leaves the original untouched.
/// let appended = array.clone().append(Sequence::from(AtomicValue::integer(4_i64)));
/// assert_eq!(appended.len(), 3);
/// assert_eq!(array.len(), 2);
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct XdmArray {
    members: Vec<Sequence>,
}

impl XdmArray {
    /// The empty array, `[]`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            members: Vec::new(),
        }
    }

    /// Builds an array from its members, in order.
    pub fn from_members(members: impl IntoIterator<Item = Sequence>) -> Self {
        Self {
            members: members.into_iter().collect(),
        }
    }

    /// The number of members.
    #[must_use]
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Whether the array has no members.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// The member at 1-based `position`, as `array:get` returns it.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Foay0001`] if `position` is not in `1..=len`.
    pub fn get(&self, position: usize) -> Result<&Sequence, XdmError> {
        self.member_at(position.checked_sub(1))
            .ok_or_else(|| self.out_of_bounds(position))
    }

    /// The member at 0-based `index`.
    #[must_use]
    pub fn get_at(&self, index: usize) -> Option<&Sequence> {
        self.members.get(index)
    }

    fn member_at(&self, index: Option<usize>) -> Option<&Sequence> {
        self.members.get(index?)
    }

    fn out_of_bounds(&self, position: usize) -> XdmError {
        XdmError::new(
            ErrorCode::Foay0001,
            format!(
                "position {position} is outside the array's 1..={} range",
                self.members.len()
            ),
        )
    }

    /// Iterates over the members, in order.
    pub fn members(&self) -> Iter<'_, Sequence> {
        self.members.iter()
    }

    /// Consumes the array, yielding its members.
    #[must_use]
    pub fn into_members(self) -> Vec<Sequence> {
        self.members
    }

    /// Returns this array with `member` appended, as `array:append` does.
    #[must_use]
    pub fn append(mut self, member: Sequence) -> Self {
        self.members.push(member);
        self
    }

    /// Returns this array with `member` inserted *before* 1-based `position`.
    ///
    /// `position` may be `len + 1`, which appends.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Foay0001`] if `position` is not in `1..=len + 1`.
    pub fn insert_before(mut self, position: usize, member: Sequence) -> Result<Self, XdmError> {
        let index = position
            .checked_sub(1)
            .filter(|index| *index <= self.members.len())
            .ok_or_else(|| {
                XdmError::new(
                    ErrorCode::Foay0001,
                    format!(
                        "position {position} is outside the insertion range 1..={}",
                        self.members.len() + 1
                    ),
                )
            })?;
        self.members.insert(index, member);
        Ok(self)
    }

    /// Returns this array with the member at 1-based `position` removed.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Foay0001`] if `position` is not in `1..=len`.
    pub fn remove(mut self, position: usize) -> Result<Self, XdmError> {
        let index = position
            .checked_sub(1)
            .filter(|index| *index < self.members.len())
            .ok_or_else(|| self.out_of_bounds(position))?;
        self.members.remove(index);
        Ok(self)
    }

    /// Replaces the member at 1-based `position`, as `array:put` does.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Foay0001`] if `position` is not in `1..=len`.
    pub fn put(mut self, position: usize, member: Sequence) -> Result<Self, XdmError> {
        let index = position
            .checked_sub(1)
            .filter(|index| *index < self.members.len())
            .ok_or_else(|| self.out_of_bounds(position))?;
        self.members[index] = member;
        Ok(self)
    }

    /// The sub-array of `length` members starting at 1-based `start`, as `array:subarray` does.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Foay0001`] if `start` is not in `1..=len + 1`, or if `start + length`
    /// exceeds `len + 1`.
    pub fn subarray(&self, start: usize, length: usize) -> Result<Self, XdmError> {
        let begin = start.checked_sub(1).ok_or_else(|| {
            XdmError::new(
                ErrorCode::Foay0001,
                "a subarray start position must be >= 1",
            )
        })?;
        let end = begin
            .checked_add(length)
            .ok_or_else(|| XdmError::new(ErrorCode::Foay0001, "the subarray range overflows"))?;
        if end > self.members.len() {
            return Err(XdmError::new(
                ErrorCode::Foay0001,
                format!(
                    "the subarray {start}..{} runs past the array's {} members",
                    start + length,
                    self.members.len()
                ),
            ));
        }
        Ok(Self {
            members: self.members[begin..end].to_vec(),
        })
    }

    /// The first member, as `array:head` returns it.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Foay0001`] if the array is empty.
    pub fn head(&self) -> Result<&Sequence, XdmError> {
        self.get(1)
    }

    /// Everything but the first member, as `array:tail` returns it.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Foay0001`] if the array is empty.
    pub fn tail(&self) -> Result<Self, XdmError> {
        if self.members.is_empty() {
            return Err(self.out_of_bounds(1));
        }
        self.subarray(2, self.members.len() - 1)
    }

    /// This array with its members in the opposite order.
    #[must_use]
    pub fn reverse(mut self) -> Self {
        self.members.reverse();
        self
    }

    /// Concatenates several arrays, as `array:join` does.
    pub fn join(arrays: impl IntoIterator<Item = Self>) -> Self {
        let mut members = Vec::new();
        for array in arrays {
            members.extend(array.members);
        }
        Self { members }
    }

    /// Flattens the array into a single sequence, recursively replacing every array item by its
    /// members, as `array:flatten` does.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_infoset::{AtomicValue, Item, Sequence, XdmArray};
    ///
    /// // [ [1, 2], 3 ] flattens to (1, 2, 3).
    /// let inner = XdmArray::from_iter([AtomicValue::integer(1_i64), AtomicValue::integer(2_i64)]);
    /// let outer = XdmArray::from_members([
    ///     Sequence::singleton(Item::from(inner)),
    ///     Sequence::from(AtomicValue::integer(3_i64)),
    /// ]);
    /// assert_eq!(outer.flatten().len(), 3);
    /// ```
    #[must_use]
    pub fn flatten(&self) -> Sequence {
        let mut out = Sequence::empty();
        for member in &self.members {
            for item in member {
                match item {
                    Item::Array(nested) => out.append(nested.flatten()),
                    other => out.push(other.clone()),
                }
            }
        }
        out
    }
}

impl FromIterator<Sequence> for XdmArray {
    fn from_iter<T: IntoIterator<Item = Sequence>>(iter: T) -> Self {
        Self::from_members(iter)
    }
}

impl FromIterator<Item> for XdmArray {
    /// Each item becomes a singleton member.
    fn from_iter<T: IntoIterator<Item = Item>>(iter: T) -> Self {
        Self::from_members(iter.into_iter().map(Sequence::singleton))
    }
}

impl FromIterator<AtomicValue> for XdmArray {
    /// Each atomic value becomes a singleton member.
    fn from_iter<T: IntoIterator<Item = AtomicValue>>(iter: T) -> Self {
        Self::from_members(iter.into_iter().map(Sequence::from))
    }
}

impl<'a> IntoIterator for &'a XdmArray {
    type Item = &'a Sequence;
    type IntoIter = Iter<'a, Sequence>;

    fn into_iter(self) -> Self::IntoIter {
        self.members.iter()
    }
}

impl IntoIterator for XdmArray {
    type Item = Sequence;
    type IntoIter = std::vec::IntoIter<Sequence>;

    fn into_iter(self) -> Self::IntoIter {
        self.members.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numbers(values: &[i64]) -> XdmArray {
        XdmArray::from_iter(values.iter().copied().map(AtomicValue::integer))
    }

    fn as_numbers(array: &XdmArray) -> Vec<String> {
        array
            .members()
            .map(|member| {
                member
                    .iter()
                    .map(Item::to_string)
                    .collect::<Vec<_>>()
                    .join("|")
            })
            .collect()
    }

    #[test]
    fn positions_are_one_based_and_bounds_checked() -> Result<(), XdmError> {
        let array = numbers(&[10, 20, 30]);
        assert_eq!(array.get(1)?.len(), 1);
        assert_eq!(array.get_at(0), array.get(1).ok());
        assert_eq!(
            array.get(0).expect_err("0 is not a position").code(),
            ErrorCode::Foay0001
        );
        assert_eq!(
            array.get(4).expect_err("past the end").code(),
            ErrorCode::Foay0001
        );
        assert!(XdmArray::new().head().is_err());
        Ok(())
    }

    #[test]
    fn insert_remove_and_put() -> Result<(), XdmError> {
        let array = numbers(&[1, 3]);
        let inserted = array
            .clone()
            .insert_before(2, Sequence::from(AtomicValue::integer(2_i64)))?;
        assert_eq!(as_numbers(&inserted), ["1", "2", "3"]);

        // `len + 1` appends.
        let appended = array
            .clone()
            .insert_before(3, Sequence::from(AtomicValue::integer(9_i64)))?;
        assert_eq!(as_numbers(&appended), ["1", "3", "9"]);
        assert!(array.clone().insert_before(4, Sequence::empty()).is_err());

        assert_eq!(as_numbers(&inserted.clone().remove(2)?), ["1", "3"]);
        assert!(inserted.clone().remove(0).is_err());
        assert!(inserted.clone().remove(4).is_err());

        let replaced = inserted.put(1, Sequence::from(AtomicValue::integer(7_i64)))?;
        assert_eq!(as_numbers(&replaced), ["7", "2", "3"]);
        Ok(())
    }

    #[test]
    fn subarray_head_tail_reverse_and_join() -> Result<(), XdmError> {
        let array = numbers(&[1, 2, 3, 4]);
        assert_eq!(as_numbers(&array.subarray(2, 2)?), ["2", "3"]);
        assert_eq!(as_numbers(&array.subarray(1, 0)?), Vec::<String>::new());
        assert_eq!(as_numbers(&array.subarray(5, 0)?), Vec::<String>::new());
        assert!(array.subarray(4, 2).is_err());
        assert!(array.subarray(0, 1).is_err());

        assert_eq!(array.head()?.len(), 1);
        assert_eq!(as_numbers(&array.tail()?), ["2", "3", "4"]);
        assert_eq!(as_numbers(&array.clone().reverse()), ["4", "3", "2", "1"]);

        let joined = XdmArray::join([numbers(&[1]), XdmArray::new(), numbers(&[2, 3])]);
        assert_eq!(as_numbers(&joined), ["1", "2", "3"]);
        Ok(())
    }

    #[test]
    fn flatten_is_recursive() {
        let inner = numbers(&[1, 2]);
        let outer = XdmArray::from_members([
            Sequence::singleton(Item::from(inner)),
            Sequence::from(AtomicValue::integer(3_i64)),
        ]);
        let flat = outer.flatten();
        assert_eq!(flat.len(), 3);
        assert_eq!(
            flat.iter().map(Item::to_string).collect::<Vec<_>>(),
            ["1", "2", "3"]
        );
    }

    #[test]
    fn iteration_and_collection() {
        let array = numbers(&[1, 2]);
        assert_eq!((&array).into_iter().count(), 2);
        assert_eq!(array.clone().into_iter().count(), 2);
        let round_tripped: XdmArray = array.clone().into_iter().collect();
        assert_eq!(round_tripped, array);
        assert_eq!(array.clone().into_members().len(), 2);
        assert!(XdmArray::new().is_empty());
        assert_eq!(XdmArray::default(), XdmArray::new());
    }
}
