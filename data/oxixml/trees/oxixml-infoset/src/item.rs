//! [`Item`] — the four things an XDM sequence can contain.

use crate::array::XdmArray;
use crate::atomic::AtomicValue;
use crate::error::{ErrorCode, XdmError};
use crate::function::XdmFunction;
use crate::map::XdmMap;
use crate::node::{NodeRef, XdmNode};
use crate::sequence::Sequence;
use std::fmt;

/// One member of an XDM [`Sequence`].
///
/// [XDM 3.1 §2.8](https://www.w3.org/TR/xpath-datamodel-31/#dt-item) defines an item as a node,
/// an atomic value, or a function item. Maps and arrays are the two function items the data
/// model itself defines, so they get their own variants; every *other* function item — an
/// inline function, a named function reference, a partial application — is held opaquely by
/// [`Item::Function`], whose [`XdmFunction`] handle exposes only the name and arity XDM defines
/// and lets the evaluator that created it downcast back to its own closure. That is what allows
/// `map { "next": function() { … } }` and `array { fn:abs#1 }` to be ordinary XDM values.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{AtomicValue, Item, Tree, XdmArray};
///
/// let atomic = Item::from(AtomicValue::integer(1_i64));
/// assert!(atomic.is_atomic());
/// assert_eq!(atomic.string_value()?, "1");
///
/// let tree = Tree::parse_str("<a>text</a>")?;
/// let node = Item::from(tree.document_element().expect("a document element"));
/// assert!(node.is_node());
/// assert_eq!(node.string_value()?, "text");
///
/// // An array atomizes to the atomization of its members, in order.
/// let array = Item::from(XdmArray::from_iter([
///     AtomicValue::integer(1_i64),
///     AtomicValue::integer(2_i64),
/// ]));
/// assert_eq!(array.atomize()?.len(), 2);
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Item {
    /// A node.
    Node(NodeRef),
    /// An atomic value.
    Atomic(AtomicValue),
    /// A map — a function item from atomic keys to sequences.
    Map(XdmMap),
    /// An array — a function item from a 1-based integer index to a sequence.
    Array(XdmArray),
    /// Any other function item, held as the opaque handle [`XdmFunction`].
    Function(XdmFunction),
}

impl Item {
    /// Whether this item is a node.
    #[must_use]
    pub const fn is_node(&self) -> bool {
        matches!(self, Self::Node(_))
    }

    /// Whether this item is an atomic value.
    #[must_use]
    pub const fn is_atomic(&self) -> bool {
        matches!(self, Self::Atomic(_))
    }

    /// Whether this item is a function item — a map, an array, or an opaque function.
    #[must_use]
    pub const fn is_function(&self) -> bool {
        matches!(self, Self::Map(_) | Self::Array(_) | Self::Function(_))
    }

    /// The opaque function handle, if this item is one.
    ///
    /// A map or an array answers `None` even though XDM calls them function items too: they
    /// have their own variants, and [`Item::as_map`] and [`Item::as_array`] read them.
    #[must_use]
    pub const fn as_function(&self) -> Option<&XdmFunction> {
        match self {
            Self::Function(function) => Some(function),
            _ => None,
        }
    }

    /// The node, if this item is one.
    #[must_use]
    pub const fn as_node(&self) -> Option<&NodeRef> {
        match self {
            Self::Node(node) => Some(node),
            _ => None,
        }
    }

    /// The atomic value, if this item is one.
    #[must_use]
    pub const fn as_atomic(&self) -> Option<&AtomicValue> {
        match self {
            Self::Atomic(value) => Some(value),
            _ => None,
        }
    }

    /// The map, if this item is one.
    #[must_use]
    pub const fn as_map(&self) -> Option<&XdmMap> {
        match self {
            Self::Map(map) => Some(map),
            _ => None,
        }
    }

    /// The array, if this item is one.
    #[must_use]
    pub const fn as_array(&self) -> Option<&XdmArray> {
        match self {
            Self::Array(array) => Some(array),
            _ => None,
        }
    }

    /// The item's string value, as `fn:string` computes it.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Foty0014`] for a map or an array: function items have no string value.
    pub fn string_value(&self) -> Result<String, XdmError> {
        match self {
            Self::Node(node) => Ok(node.string_value()),
            Self::Atomic(value) => Ok(value.string_value()),
            Self::Map(_) | Self::Array(_) | Self::Function(_) => Err(XdmError::new(
                ErrorCode::Foty0014,
                "a function item has no string value",
            )),
        }
    }

    /// The item's atomized value, as `fn:data` computes it.
    ///
    /// A node atomizes to its typed value; an atomic value to itself; an array to the
    /// concatenated atomization of its members; a map cannot be atomized.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Foty0013`] for a map, or whatever
    /// [`XdmNode::typed_value`] raises for a node.
    pub fn atomize(&self) -> Result<Sequence, XdmError> {
        match self {
            Self::Node(node) => node.typed_value(),
            Self::Atomic(value) => Ok(Sequence::singleton(Self::Atomic(value.clone()))),
            Self::Array(array) => {
                let mut out = Vec::new();
                for member in array.members() {
                    out.extend(member.atomize()?);
                }
                Ok(Sequence::from_items(out))
            }
            Self::Map(_) => Err(XdmError::new(
                ErrorCode::Foty0013,
                "a map cannot be atomized",
            )),
            Self::Function(_) => Err(XdmError::new(
                ErrorCode::Foty0013,
                "a function item cannot be atomized",
            )),
        }
    }
}

impl From<NodeRef> for Item {
    fn from(node: NodeRef) -> Self {
        Self::Node(node)
    }
}

impl From<AtomicValue> for Item {
    fn from(value: AtomicValue) -> Self {
        Self::Atomic(value)
    }
}

impl From<XdmMap> for Item {
    fn from(map: XdmMap) -> Self {
        Self::Map(map)
    }
}

impl From<XdmArray> for Item {
    fn from(array: XdmArray) -> Self {
        Self::Array(array)
    }
}

impl From<XdmFunction> for Item {
    fn from(function: XdmFunction) -> Self {
        Self::Function(function)
    }
}

impl fmt::Display for Item {
    /// Formats the item's string value, or a placeholder for function items, which have none.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Node(node) => f.write_str(&node.string_value()),
            Self::Atomic(value) => write!(f, "{value}"),
            Self::Map(map) => write!(f, "map{{ {} entries }}", map.len()),
            Self::Array(array) => write!(f, "array{{ {} members }}", array.len()),
            Self::Function(function) => write!(f, "{function}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atomic::AtomicType;

    #[test]
    fn classification_predicates_are_mutually_exclusive() {
        let atomic = Item::from(AtomicValue::string("x"));
        assert!(atomic.is_atomic() && !atomic.is_node() && !atomic.is_function());

        let map = Item::from(XdmMap::new());
        assert!(map.is_function() && !map.is_atomic() && !map.is_node());

        let array = Item::from(XdmArray::new());
        assert!(array.is_function() && !array.is_atomic() && !array.is_node());
    }

    #[test]
    fn projections_return_none_for_the_wrong_shape() {
        let atomic = Item::from(AtomicValue::string("x"));
        assert!(atomic.as_atomic().is_some());
        assert!(atomic.as_node().is_none());
        assert!(atomic.as_map().is_none());
        assert!(atomic.as_array().is_none());
    }

    #[test]
    fn function_items_have_no_string_value_and_maps_do_not_atomize() {
        let map = Item::from(XdmMap::new());
        assert_eq!(
            map.string_value().expect_err("no string value").code(),
            ErrorCode::Foty0014
        );
        assert_eq!(
            map.atomize().expect_err("not atomizable").code(),
            ErrorCode::Foty0013
        );
    }

    #[test]
    fn arrays_atomize_their_members_in_order() -> Result<(), XdmError> {
        let array = XdmArray::from_iter([
            AtomicValue::integer(1_i64),
            AtomicValue::parse(AtomicType::String, "two")?,
        ]);
        let atomized = Item::from(array).atomize()?;
        assert_eq!(atomized.len(), 2);
        assert_eq!(atomized.get(0).map(Item::to_string).as_deref(), Some("1"));
        assert_eq!(atomized.get(1).map(Item::to_string).as_deref(), Some("two"));
        Ok(())
    }

    #[test]
    fn display_renders_string_values_and_placeholders() {
        assert_eq!(Item::from(AtomicValue::integer(7_i64)).to_string(), "7");
        assert_eq!(
            Item::from(XdmArray::new()).to_string(),
            "array{ 0 members }"
        );
        assert_eq!(Item::from(XdmMap::new()).to_string(), "map{ 0 entries }");
    }
}
