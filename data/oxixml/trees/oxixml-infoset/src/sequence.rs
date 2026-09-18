//! [`Sequence`] — the ordered collection of items that every XPath and XQuery expression
//! evaluates to.

use crate::atomic::{AtomicData, AtomicValue};
use crate::error::{ErrorCode, XdmError};
use crate::item::Item;
use crate::node::NodeRef;
use std::slice::Iter;

/// An ordered, flat collection of [`Item`]s: the value of every XPath and XQuery expression.
///
/// # The XDM sequence axioms
///
/// [XDM 3.1 §2.6](https://www.w3.org/TR/xpath-datamodel-31/#dt-sequence) fixes three properties
/// that this type enforces structurally rather than by convention:
///
/// * **Sequences never nest.** [`Item`] has no `Sequence` variant, so there is no way to build
///   one that contains another; [`Sequence::concat`] and [`Sequence::from_sequences`]
///   concatenate rather than embed. (An [`XdmArray`](crate::XdmArray) *does* contain sequences —
///   that is precisely what distinguishes an array from a sequence in XDM 3.1.)
/// * **A sequence of one item is that item.** [`Sequence::singleton`] and
///   [`From<Item>`](#impl-From<Item>-for-Sequence) are the same thing, and
///   [`Sequence::into_singleton`] recovers it.
/// * **There is exactly one empty sequence**, and it is not the same as a sequence containing
///   an empty string or a "null": [`Sequence::empty`].
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{AtomicValue, Item, Sequence};
///
/// // Concatenation flattens: there is no such thing as a sequence of sequences.
/// let left = Sequence::from_items([Item::from(AtomicValue::integer(1_i64))]);
/// let right = Sequence::from_items([
///     Item::from(AtomicValue::integer(2_i64)),
///     Item::from(AtomicValue::integer(3_i64)),
/// ]);
/// let joined = left.concat(right);
/// assert_eq!(joined.len(), 3);
/// assert!(joined.iter().all(Item::is_atomic));
///
/// // The empty sequence is distinct from a sequence holding an empty string.
/// assert!(Sequence::empty().is_empty());
/// assert_eq!(Sequence::singleton(Item::from(AtomicValue::string(""))).len(), 1);
///
/// // Effective boolean value follows fn:boolean.
/// assert!(!Sequence::empty().effective_boolean_value()?);
/// assert!(Sequence::from(AtomicValue::string("x")).effective_boolean_value()?);
/// assert!(!Sequence::from(AtomicValue::string("")).effective_boolean_value()?);
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Sequence {
    items: Vec<Item>,
}

impl Sequence {
    /// The empty sequence.
    #[must_use]
    pub const fn empty() -> Self {
        Self { items: Vec::new() }
    }

    /// A sequence of exactly one item.
    #[must_use]
    pub fn singleton(item: Item) -> Self {
        Self { items: vec![item] }
    }

    /// Builds a sequence from items, in order.
    pub fn from_items(items: impl IntoIterator<Item = Item>) -> Self {
        Self {
            items: items.into_iter().collect(),
        }
    }

    /// Concatenates several sequences into one flat sequence.
    ///
    /// This is what a comma expression does, and it is why nested sequences cannot arise.
    pub fn from_sequences(sequences: impl IntoIterator<Item = Self>) -> Self {
        let mut items = Vec::new();
        for sequence in sequences {
            items.extend(sequence.items);
        }
        Self { items }
    }

    /// Appends `item` to the end of the sequence.
    pub fn push(&mut self, item: impl Into<Item>) {
        self.items.push(item.into());
    }

    /// Appends every item of `other` to the end of this sequence.
    pub fn append(&mut self, other: Self) {
        self.items.extend(other.items);
    }

    /// Returns the concatenation of `self` and `other`.
    #[must_use]
    pub fn concat(mut self, other: Self) -> Self {
        self.append(other);
        self
    }

    /// The number of items.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether this is the empty sequence.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Whether this sequence holds exactly one item.
    #[must_use]
    pub fn is_singleton(&self) -> bool {
        self.items.len() == 1
    }

    /// The item at `index`, counting from zero.
    ///
    /// XPath itself counts from one; the conversion belongs to the evaluator, not here.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Item> {
        self.items.get(index)
    }

    /// The first item.
    #[must_use]
    pub fn first(&self) -> Option<&Item> {
        self.items.first()
    }

    /// The last item.
    #[must_use]
    pub fn last(&self) -> Option<&Item> {
        self.items.last()
    }

    /// The items as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[Item] {
        &self.items
    }

    /// Iterates over the items.
    pub fn iter(&self) -> Iter<'_, Item> {
        self.items.iter()
    }

    /// Consumes the sequence, yielding its items.
    #[must_use]
    pub fn into_items(self) -> Vec<Item> {
        self.items
    }

    /// Recovers the single item of a singleton sequence.
    #[must_use]
    pub fn into_singleton(self) -> Option<Item> {
        let mut items = self.items;
        match items.len() {
            1 => items.pop(),
            _ => None,
        }
    }

    /// Atomizes every item, as `fn:data` does.
    ///
    /// # Errors
    ///
    /// Whatever [`Item::atomize`] raises — [`ErrorCode::Foty0013`] for a map, or
    /// [`ErrorCode::Foty0012`] for a node with no typed value.
    pub fn atomize(&self) -> Result<Self, XdmError> {
        let mut out = Vec::with_capacity(self.items.len());
        for item in &self.items {
            out.extend(item.atomize()?.items);
        }
        Ok(Self { items: out })
    }

    /// The sequence's [effective boolean value](https://www.w3.org/TR/xpath-31/#id-ebv), which
    /// is what `fn:boolean` computes and what a conditional or a predicate tests.
    ///
    /// The rules, in order:
    ///
    /// 1. the empty sequence is `false`;
    /// 2. a sequence whose first item is a node is `true`;
    /// 3. a singleton `xs:boolean` is itself;
    /// 4. a singleton string-family or `xs:untypedAtomic` value is `true` when non-empty;
    /// 5. a singleton numeric value is `true` unless it is zero or `NaN`;
    /// 6. anything else is a type error.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Forg0006`] for a sequence that matches none of the rules above — a
    /// multi-item sequence of atomic values, a singleton `xs:date`, a map, an array.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_infoset::{AtomicValue, Sequence, Tree};
    ///
    /// assert!(!Sequence::empty().effective_boolean_value()?);
    /// assert!(Sequence::from(AtomicValue::integer(1_i64)).effective_boolean_value()?);
    /// assert!(!Sequence::from(AtomicValue::integer(0_i64)).effective_boolean_value()?);
    /// assert!(!Sequence::from(AtomicValue::double(f64::NAN)).effective_boolean_value()?);
    ///
    /// // A node-headed sequence is true whatever the node contains.
    /// let tree = Tree::parse_str("<a/>")?;
    /// assert!(Sequence::from(tree.root()).effective_boolean_value()?);
    ///
    /// // A singleton xs:date has no effective boolean value.
    /// let date = AtomicValue::parse(oxixml_infoset::AtomicType::Date, "2020-01-01")?;
    /// assert!(Sequence::from(date).effective_boolean_value().is_err());
    /// # Ok::<(), oxixml_infoset::XdmError>(())
    /// ```
    pub fn effective_boolean_value(&self) -> Result<bool, XdmError> {
        let Some(first) = self.items.first() else {
            return Ok(false);
        };
        if first.is_node() {
            return Ok(true);
        }
        if self.items.len() != 1 {
            return Err(no_effective_boolean_value(
                "a sequence of more than one item whose first item is not a node",
            ));
        }
        let Some(value) = first.as_atomic() else {
            return Err(no_effective_boolean_value("a function item"));
        };
        match value.data() {
            AtomicData::Boolean(boolean) => Ok(bool::from(*boolean)),
            AtomicData::Text(text) if value.atomic_type().is_text() => Ok(!text.is_empty()),
            data if data.is_numeric() => {
                if data.is_nan() {
                    return Ok(false);
                }
                Ok(data.as_f64().is_some_and(|number| number != 0.0))
            }
            _ => Err(no_effective_boolean_value(&format!(
                "a singleton {}",
                value.atomic_type()
            ))),
        }
    }

    /// The nodes of this sequence, sorted into document order with duplicates removed.
    ///
    /// This is the normalization every path expression step and every `union`, `intersect` or
    /// `except` operator applies. Non-node items are dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_infoset::{Sequence, Tree, XdmNode};
    ///
    /// let tree = Tree::parse_str("<a><b/><c/></a>")?;
    /// let a = tree.document_element().expect("a document element");
    /// let children = a.children();
    ///
    /// // Feed the nodes in backwards, with a duplicate.
    /// let mut sequence = Sequence::empty();
    /// sequence.push(children[1].clone());
    /// sequence.push(children[0].clone());
    /// sequence.push(children[1].clone());
    ///
    /// assert_eq!(sequence.nodes_in_document_order(), children);
    /// # Ok::<(), oxixml_infoset::XdmError>(())
    /// ```
    #[must_use]
    pub fn nodes_in_document_order(&self) -> Vec<NodeRef> {
        let mut nodes: Vec<NodeRef> = self
            .items
            .iter()
            .filter_map(|item| item.as_node().cloned())
            .collect();
        nodes.sort();
        nodes.dedup();
        nodes
    }
}

fn no_effective_boolean_value(what: &str) -> XdmError {
    XdmError::new(
        ErrorCode::Forg0006,
        format!("{what} has no effective boolean value"),
    )
}

impl From<Item> for Sequence {
    fn from(item: Item) -> Self {
        Self::singleton(item)
    }
}

impl From<AtomicValue> for Sequence {
    fn from(value: AtomicValue) -> Self {
        Self::singleton(Item::Atomic(value))
    }
}

impl From<NodeRef> for Sequence {
    fn from(node: NodeRef) -> Self {
        Self::singleton(Item::Node(node))
    }
}

impl FromIterator<Item> for Sequence {
    fn from_iter<T: IntoIterator<Item = Item>>(iter: T) -> Self {
        Self::from_items(iter)
    }
}

impl FromIterator<AtomicValue> for Sequence {
    fn from_iter<T: IntoIterator<Item = AtomicValue>>(iter: T) -> Self {
        Self::from_items(iter.into_iter().map(Item::Atomic))
    }
}

impl FromIterator<NodeRef> for Sequence {
    fn from_iter<T: IntoIterator<Item = NodeRef>>(iter: T) -> Self {
        Self::from_items(iter.into_iter().map(Item::Node))
    }
}

impl IntoIterator for Sequence {
    type Item = Item;
    type IntoIter = std::vec::IntoIter<Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<'a> IntoIterator for &'a Sequence {
    type Item = &'a Item;
    type IntoIter = Iter<'a, Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

impl Extend<Item> for Sequence {
    fn extend<T: IntoIterator<Item = Item>>(&mut self, iter: T) {
        self.items.extend(iter);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atomic::AtomicType;
    use crate::map::XdmMap;

    #[test]
    fn construction_axioms() {
        assert!(Sequence::empty().is_empty());
        assert_eq!(Sequence::empty().len(), 0);
        assert_eq!(Sequence::default(), Sequence::empty());

        let one = Sequence::from(AtomicValue::integer(1_i64));
        assert!(one.is_singleton());
        assert_eq!(
            one.clone().into_singleton(),
            Some(Item::from(AtomicValue::integer(1_i64)))
        );
        assert_eq!(Sequence::empty().into_singleton(), None);
    }

    #[test]
    fn concatenation_flattens_rather_than_nesting() {
        let a = Sequence::from(AtomicValue::integer(1_i64));
        let b = Sequence::from_items([
            Item::from(AtomicValue::integer(2_i64)),
            Item::from(AtomicValue::integer(3_i64)),
        ]);
        let joined = Sequence::from_sequences([a.clone(), b.clone(), Sequence::empty()]);
        assert_eq!(joined.len(), 3);
        assert_eq!(joined, a.concat(b));
        assert!(joined.iter().all(Item::is_atomic));
    }

    #[test]
    fn effective_boolean_value_rules() -> Result<(), XdmError> {
        assert!(!Sequence::empty().effective_boolean_value()?);
        assert!(Sequence::from(AtomicValue::boolean(true)).effective_boolean_value()?);
        assert!(!Sequence::from(AtomicValue::boolean(false)).effective_boolean_value()?);
        assert!(Sequence::from(AtomicValue::string("x")).effective_boolean_value()?);
        assert!(!Sequence::from(AtomicValue::string("")).effective_boolean_value()?);
        assert!(Sequence::from(AtomicValue::untyped_atomic("0")).effective_boolean_value()?);
        assert!(Sequence::from(AtomicValue::integer(-1_i64)).effective_boolean_value()?);
        assert!(!Sequence::from(AtomicValue::integer(0_i64)).effective_boolean_value()?);
        assert!(!Sequence::from(AtomicValue::double(0.0)).effective_boolean_value()?);
        assert!(!Sequence::from(AtomicValue::double(f64::NAN)).effective_boolean_value()?);
        assert!(Sequence::from(AtomicValue::double(f64::INFINITY)).effective_boolean_value()?);
        Ok(())
    }

    #[test]
    fn effective_boolean_value_errors() -> Result<(), XdmError> {
        let two = Sequence::from_items([
            Item::from(AtomicValue::integer(1_i64)),
            Item::from(AtomicValue::integer(2_i64)),
        ]);
        assert_eq!(
            two.effective_boolean_value()
                .expect_err("more than one atomic item")
                .code(),
            ErrorCode::Forg0006
        );

        let date = AtomicValue::parse(AtomicType::Date, "2020-01-01")?;
        assert_eq!(
            Sequence::from(date)
                .effective_boolean_value()
                .expect_err("xs:date has no EBV")
                .code(),
            ErrorCode::Forg0006
        );

        assert_eq!(
            Sequence::singleton(Item::from(XdmMap::new()))
                .effective_boolean_value()
                .expect_err("a map has no EBV")
                .code(),
            ErrorCode::Forg0006
        );
        Ok(())
    }

    #[test]
    fn atomize_flattens_nested_arrays() -> Result<(), XdmError> {
        let inner = crate::array::XdmArray::from_iter([AtomicValue::integer(2_i64)]);
        let outer = crate::array::XdmArray::from_members([
            Sequence::from(AtomicValue::integer(1_i64)),
            Sequence::singleton(Item::from(inner)),
        ]);
        let atomized = Sequence::singleton(Item::from(outer)).atomize()?;
        assert_eq!(atomized.len(), 2);
        assert_eq!(
            atomized
                .iter()
                .map(Item::to_string)
                .collect::<Vec<_>>()
                .join(","),
            "1,2"
        );
        Ok(())
    }

    #[test]
    fn iteration_and_collection_round_trip() {
        let values = [AtomicValue::integer(1_i64), AtomicValue::integer(2_i64)];
        let sequence: Sequence = values.iter().cloned().collect();
        assert_eq!(sequence.len(), 2);
        let round_tripped: Sequence = sequence.clone().into_iter().collect();
        assert_eq!(round_tripped, sequence);
        assert_eq!(sequence.iter().count(), 2);
        assert_eq!((&sequence).into_iter().count(), 2);
        assert_eq!(sequence.as_slice().len(), 2);
        assert_eq!(sequence.first(), sequence.get(0));
        assert_eq!(sequence.last(), sequence.get(1));
    }
}
