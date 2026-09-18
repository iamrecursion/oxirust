//! [`XpathItem`] and [`XpathValue`] — the sequences this engine computes with.
//!
//! [`oxixml_infoset::Sequence`] is the XDM sequence: nodes, atomic values, maps and arrays. It
//! deliberately has no variant for a *general* function item, because a closure over an
//! expression tree is not part of the data model. XPath 3.0 introduced exactly that, so the
//! evaluator works with [`XpathValue`], which is a [`Sequence`] plus the one extra item kind the
//! language needs.
//!
//! The two convert freely: every [`Sequence`] is an [`XpathValue`], and an [`XpathValue`]
//! converts back whenever it holds no function items — which is why
//! [`CompiledXPath::evaluate`](crate::CompiledXPath::evaluate) can return a plain [`Sequence`]
//! and only fails on the (rare) expression whose *result* is a function.

use crate::error::{XPathError, XPathErrorCode, XPathResult};
use crate::function::XpathFunction;
use oxixml_infoset::{AtomicData, AtomicValue, Item, NodeRef, Sequence, XdmArray, XdmMap, XdmNode};
use std::fmt;

/// One member of an [`XpathValue`]: everything [`Item`] can be, plus a function item.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{AtomicValue, Item};
/// use oxixml_xpath::XpathItem;
///
/// let item = XpathItem::from(AtomicValue::integer(7_i64));
/// assert!(item.is_atomic());
/// assert!(!item.is_function());
/// assert_eq!(item.string_value()?, "7");
/// assert_eq!(item.into_item().map(|item| item.to_string()), Some("7".to_owned()));
/// # Ok::<(), oxixml_xpath::XPathError>(())
/// ```
#[derive(Debug, Clone)]
pub enum XpathItem {
    /// A node, an atomic value, a map or an array — anything the data model itself models.
    Value(Item),
    /// A function item: an inline function, a named function reference, a partial application,
    /// or a map or array viewed as a function.
    Function(XpathFunction),
}

impl XpathItem {
    /// Whether this item is a node.
    #[must_use]
    pub fn is_node(&self) -> bool {
        matches!(self, Self::Value(item) if item.is_node())
    }

    /// Whether this item is an atomic value.
    #[must_use]
    pub fn is_atomic(&self) -> bool {
        matches!(self, Self::Value(item) if item.is_atomic())
    }

    /// Whether this item is a function item — including a map or an array, which XDM defines as
    /// function items.
    #[must_use]
    pub fn is_function(&self) -> bool {
        match self {
            Self::Function(_) => true,
            Self::Value(item) => item.is_function(),
        }
    }

    /// The node, if this item is one.
    #[must_use]
    pub fn as_node(&self) -> Option<&NodeRef> {
        match self {
            Self::Value(item) => item.as_node(),
            Self::Function(_) => None,
        }
    }

    /// The atomic value, if this item is one.
    #[must_use]
    pub fn as_atomic(&self) -> Option<&AtomicValue> {
        match self {
            Self::Value(item) => item.as_atomic(),
            Self::Function(_) => None,
        }
    }

    /// The map, if this item is one.
    #[must_use]
    pub fn as_map(&self) -> Option<&XdmMap> {
        match self {
            Self::Value(item) => item.as_map(),
            Self::Function(_) => None,
        }
    }

    /// The array, if this item is one.
    #[must_use]
    pub fn as_array(&self) -> Option<&XdmArray> {
        match self {
            Self::Value(item) => item.as_array(),
            Self::Function(_) => None,
        }
    }

    /// The data-model item, if this is not a general function item.
    #[must_use]
    pub fn as_item(&self) -> Option<&Item> {
        match self {
            Self::Value(item) => Some(item),
            Self::Function(_) => None,
        }
    }

    /// Consumes the item, yielding the data-model item when there is one.
    #[must_use]
    pub fn into_item(self) -> Option<Item> {
        match self {
            Self::Value(item) => Some(item),
            Self::Function(_) => None,
        }
    }

    /// The item as a callable function: a general function item, or the map or array reading
    /// XDM says is itself a function of one argument.
    #[must_use]
    pub fn as_callable(&self) -> Option<XpathFunction> {
        match self {
            Self::Function(function) => Some(function.clone()),
            Self::Value(Item::Map(map)) => Some(XpathFunction::from_map(map.clone())),
            Self::Value(Item::Array(array)) => Some(XpathFunction::from_array(array.clone())),
            // A function item that has been through the data model — because it was stored in
            // a map, an array or a sequence — comes back as its opaque handle.
            Self::Value(Item::Function(handle)) => XpathFunction::from_xdm(handle),
            Self::Value(_) => None,
        }
    }

    /// The item's string value, as `fn:string` computes it.
    ///
    /// `xs:double` and `xs:float` follow the *casting* rule of
    /// [F&O 3.1 §19.1.3](https://www.w3.org/TR/xpath-functions-31/#casting-to-string) rather
    /// than the XML Schema canonical form, because that is the rule XPath's `fn:string` and
    /// `cast as xs:string` are defined by: a magnitude in `[1e-6, 1e6)` is written without an
    /// exponent, so `2.1E3` reads as `2100` and `1.0E-5` as `0.00001`, and only values outside
    /// that window use scientific notation.
    ///
    /// # Errors
    ///
    /// [`XPathErrorCode::Foty0014`] for a function item, which has no string value.
    pub fn string_value(&self) -> XPathResult<String> {
        match self {
            Self::Value(Item::Atomic(value)) => Ok(atomic_string_value(value)),
            Self::Value(item) => Ok(item.string_value()?),
            Self::Function(function) => Err(XPathError::new(
                XPathErrorCode::Foty0014,
                format!("{} has no string value", function.describe()),
            )),
        }
    }

    /// The item's atomized value, as `fn:data` computes it.
    ///
    /// # Errors
    ///
    /// [`XPathErrorCode::Foty0013`] for a map or a function item, and whatever
    /// [`XdmNode::typed_value`] raises for a node.
    pub fn atomize(&self) -> XPathResult<Vec<AtomicValue>> {
        match self {
            Self::Value(item) => {
                let atomized = item.atomize()?;
                Ok(atomized
                    .into_items()
                    .into_iter()
                    .filter_map(|item| item.as_atomic().cloned())
                    .collect())
            }
            Self::Function(function) => Err(XPathError::new(
                XPathErrorCode::Foty0013,
                format!("{} cannot be atomized", function.describe()),
            )),
        }
    }

    /// A short human-readable description of the item's kind, for diagnostics.
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            Self::Value(Item::Node(node)) => format!("a {} node", node.node_kind()),
            Self::Value(Item::Atomic(value)) => format!("an {}", value.atomic_type()),
            Self::Value(Item::Map(map)) => format!("a map of {} entries", map.len()),
            Self::Value(Item::Array(array)) => format!("an array of {} members", array.len()),
            Self::Value(Item::Function(handle)) => handle.to_string(),
            Self::Function(function) => function.describe(),
        }
    }
}

/// `fn:string` of an atomic value, with the F&O rule for the two floating-point types.
///
/// Every other type's string value is its XML Schema canonical form, which
/// [`AtomicValue::string_value`] already produces.
#[must_use]
pub fn atomic_string_value(value: &AtomicValue) -> String {
    match value.data() {
        AtomicData::Double(double) => format_double(f64::from(*double)),
        AtomicData::Float(float) => format_float(f32::from(*float)),
        _ => value.string_value(),
    }
}

/// [F&O 3.1 §19.1.3](https://www.w3.org/TR/xpath-functions-31/#casting-to-string): `xs:double`
/// to `xs:string`.
fn format_double(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_owned();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-INF"
        } else {
            "INF"
        }
        .to_owned();
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.to_owned();
    }
    let magnitude = value.abs();
    if (1e-6..1e6).contains(&magnitude) {
        return format!("{value}");
    }
    scientific(&format!("{value:e}"))
}

/// The same rule for `xs:float`, formatted from the single-precision value so that the shortest
/// round-tripping decimal is the `f32` one — `1.1` rather than `1.100000023841858`.
fn format_float(value: f32) -> String {
    if value.is_nan() {
        return "NaN".to_owned();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-INF"
        } else {
            "INF"
        }
        .to_owned();
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.to_owned();
    }
    let magnitude = value.abs();
    if (1e-6..1e6).contains(&magnitude) {
        return format!("{value}");
    }
    scientific(&format!("{value:e}"))
}

/// Normalizes Rust's `{:e}` rendering into the specification's form: an uppercase `E`, and a
/// mantissa with at least one digit after the decimal point.
fn scientific(rendered: &str) -> String {
    let Some((mantissa, exponent)) = rendered.split_once('e') else {
        return rendered.to_owned();
    };
    let mantissa = if mantissa.contains('.') {
        mantissa.to_owned()
    } else {
        format!("{mantissa}.0")
    };
    format!("{mantissa}E{exponent}")
}

impl From<Item> for XpathItem {
    /// A data-model item becomes an `XpathItem`, and a function item this engine created is
    /// *normalized back* to [`XpathItem::Function`] so that every call site downstream sees the
    /// callable form rather than the opaque handle. A handle from another producer, which this
    /// engine cannot call, is kept as it is.
    fn from(item: Item) -> Self {
        match item {
            Item::Function(handle) => match XpathFunction::from_xdm(&handle) {
                Some(function) => Self::Function(function),
                None => Self::Value(Item::Function(handle)),
            },
            other => Self::Value(other),
        }
    }
}

impl From<NodeRef> for XpathItem {
    fn from(node: NodeRef) -> Self {
        Self::Value(Item::Node(node))
    }
}

impl From<AtomicValue> for XpathItem {
    fn from(value: AtomicValue) -> Self {
        Self::Value(Item::Atomic(value))
    }
}

impl From<XdmMap> for XpathItem {
    fn from(map: XdmMap) -> Self {
        Self::Value(Item::Map(map))
    }
}

impl From<XdmArray> for XpathItem {
    fn from(array: XdmArray) -> Self {
        Self::Value(Item::Array(array))
    }
}

impl From<XpathFunction> for XpathItem {
    fn from(function: XpathFunction) -> Self {
        Self::Function(function)
    }
}

impl PartialEq for XpathItem {
    /// Structural equality on data-model items; *identity* on function items, since two
    /// closures are never observably equal.
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Value(left), Self::Value(right)) => left == right,
            (Self::Function(left), Self::Function(right)) => left.is_same_function(right),
            _ => false,
        }
    }
}

impl fmt::Display for XpathItem {
    /// Renders the item's string value, using the same F&O rule as
    /// [`XpathItem::string_value`] for `xs:double` and `xs:float`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Value(Item::Atomic(value)) => f.write_str(&atomic_string_value(value)),
            Self::Value(item) => write!(f, "{item}"),
            Self::Function(function) => f.write_str(&function.describe()),
        }
    }
}

/// An ordered, flat sequence of [`XpathItem`]s: the value of every XPath expression.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{AtomicValue, Sequence};
/// use oxixml_xpath::XpathValue;
///
/// let value = XpathValue::from_iter([
///     AtomicValue::integer(1_i64),
///     AtomicValue::integer(2_i64),
/// ]);
/// assert_eq!(value.len(), 2);
/// assert!(value.effective_boolean_value().is_err()); // two atomic items have no EBV
///
/// // Round-trips through the data-model sequence when it holds no function items.
/// let sequence: Sequence = value.clone().into_sequence()?;
/// assert_eq!(sequence.len(), 2);
/// assert_eq!(XpathValue::from(sequence), value);
/// # Ok::<(), oxixml_xpath::XPathError>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct XpathValue {
    items: Vec<XpathItem>,
}

impl XpathValue {
    /// The empty sequence.
    #[must_use]
    pub const fn empty() -> Self {
        Self { items: Vec::new() }
    }

    /// A sequence of exactly one item.
    pub fn singleton(item: impl Into<XpathItem>) -> Self {
        Self {
            items: vec![item.into()],
        }
    }

    /// Builds a value from items, in order.
    pub fn from_items(items: impl IntoIterator<Item = XpathItem>) -> Self {
        Self {
            items: items.into_iter().collect(),
        }
    }

    /// Concatenates several values into one flat value, as the comma operator does.
    pub fn from_values(values: impl IntoIterator<Item = Self>) -> Self {
        let mut items = Vec::new();
        for value in values {
            items.extend(value.items);
        }
        Self { items }
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

    /// Whether this value holds exactly one item.
    #[must_use]
    pub fn is_singleton(&self) -> bool {
        self.items.len() == 1
    }

    /// The item at `index`, counting from zero.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&XpathItem> {
        self.items.get(index)
    }

    /// The first item.
    #[must_use]
    pub fn first(&self) -> Option<&XpathItem> {
        self.items.first()
    }

    /// The items as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[XpathItem] {
        &self.items
    }

    /// Iterates over the items.
    pub fn iter(&self) -> std::slice::Iter<'_, XpathItem> {
        self.items.iter()
    }

    /// Consumes the value, yielding its items.
    #[must_use]
    pub fn into_items(self) -> Vec<XpathItem> {
        self.items
    }

    /// Appends an item.
    pub fn push(&mut self, item: impl Into<XpathItem>) {
        self.items.push(item.into());
    }

    /// Appends every item of `other`.
    pub fn append(&mut self, other: Self) {
        self.items.extend(other.items);
    }

    /// Whether any item is a general function item — one that is neither a map nor an array.
    ///
    /// Such an item *can* appear in a [`Sequence`], as an opaque handle; this predicate is for
    /// callers that need to know whether the engine's own callable representation is present.
    #[must_use]
    pub fn has_function_items(&self) -> bool {
        self.items
            .iter()
            .any(|item| matches!(item, XpathItem::Function(_)))
    }

    /// Converts to the data-model [`Sequence`].
    ///
    /// Function items convert too: they become [`oxixml_infoset::Item::Function`], the opaque
    /// handle the data model carries for them.
    ///
    /// # Errors
    ///
    /// Currently infallible; the fallible signature is kept because it is what every caller
    /// already threads, and because a future item kind may not convert.
    pub fn into_sequence(self) -> XPathResult<Sequence> {
        let mut items = Vec::with_capacity(self.items.len());
        for item in self.items {
            match item {
                XpathItem::Value(item) => items.push(item),
                // A general function item is a data-model item too: XDM 3.1 §2.8.2 puts it in
                // the same place as a node or an atomic value, and `oxixml_infoset::Item`
                // carries it as an opaque handle.
                XpathItem::Function(function) => items.push(Item::Function(function.to_xdm())),
            }
        }
        Ok(Sequence::from_items(items))
    }

    /// The nodes of this value, in document order and without duplicates.
    #[must_use]
    pub fn nodes(&self) -> Vec<NodeRef> {
        let mut nodes: Vec<NodeRef> = self
            .items
            .iter()
            .filter_map(|item| item.as_node().cloned())
            .collect();
        nodes.sort();
        nodes.dedup();
        nodes
    }

    /// Whether every item is a node.
    #[must_use]
    pub fn is_all_nodes(&self) -> bool {
        self.items.iter().all(XpathItem::is_node)
    }

    /// The sequence's [effective boolean value](https://www.w3.org/TR/xpath-31/#id-ebv).
    ///
    /// # Errors
    ///
    /// [`XPathErrorCode::Forg0006`] for a value that matches none of the rules — a multi-item
    /// sequence of atomic values, a singleton `xs:date`, a function item.
    pub fn effective_boolean_value(&self) -> XPathResult<bool> {
        match self.items.first() {
            None => Ok(false),
            Some(item) if item.is_node() => Ok(true),
            Some(XpathItem::Function(function)) => Err(XPathError::new(
                XPathErrorCode::Forg0006,
                format!("{} has no effective boolean value", function.describe()),
            )),
            Some(_) => {
                let sequence = self.clone().into_sequence()?;
                Ok(sequence.effective_boolean_value()?)
            }
        }
    }

    /// Atomizes every item, as `fn:data` does.
    ///
    /// # Errors
    ///
    /// [`XPathErrorCode::Foty0013`] for a map or a function item, and whatever
    /// [`XdmNode::typed_value`] raises for a node.
    pub fn atomize(&self) -> XPathResult<Vec<AtomicValue>> {
        let mut out = Vec::with_capacity(self.items.len());
        for item in &self.items {
            out.extend(item.atomize()?);
        }
        Ok(out)
    }

    /// Atomizes and requires the result to be at most one value.
    ///
    /// # Errors
    ///
    /// [`XPathErrorCode::Xpty0004`] when atomization yields more than one value.
    pub fn atomize_at_most_one(&self, what: &str) -> XPathResult<Option<AtomicValue>> {
        let mut atomized = self.atomize()?;
        match atomized.len() {
            0 => Ok(None),
            1 => Ok(atomized.pop()),
            other => Err(XPathError::type_error(format!(
                "{what} requires at most one value but was given {other}"
            ))),
        }
    }
}

impl From<Sequence> for XpathValue {
    fn from(sequence: Sequence) -> Self {
        Self::from_items(sequence.into_items().into_iter().map(XpathItem::from))
    }
}

impl From<Item> for XpathValue {
    fn from(item: Item) -> Self {
        Self::singleton(item)
    }
}

impl From<NodeRef> for XpathValue {
    fn from(node: NodeRef) -> Self {
        Self::singleton(node)
    }
}

impl From<AtomicValue> for XpathValue {
    fn from(value: AtomicValue) -> Self {
        Self::singleton(value)
    }
}

impl From<XpathItem> for XpathValue {
    fn from(item: XpathItem) -> Self {
        Self::singleton(item)
    }
}

impl FromIterator<XpathItem> for XpathValue {
    fn from_iter<T: IntoIterator<Item = XpathItem>>(iter: T) -> Self {
        Self::from_items(iter)
    }
}

impl FromIterator<AtomicValue> for XpathValue {
    fn from_iter<T: IntoIterator<Item = AtomicValue>>(iter: T) -> Self {
        Self::from_items(iter.into_iter().map(XpathItem::from))
    }
}

impl FromIterator<NodeRef> for XpathValue {
    fn from_iter<T: IntoIterator<Item = NodeRef>>(iter: T) -> Self {
        Self::from_items(iter.into_iter().map(XpathItem::from))
    }
}

impl IntoIterator for XpathValue {
    type Item = XpathItem;
    type IntoIter = std::vec::IntoIter<XpathItem>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<'a> IntoIterator for &'a XpathValue {
    type Item = &'a XpathItem;
    type IntoIter = std::slice::Iter<'a, XpathItem>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

impl fmt::Display for XpathValue {
    /// Renders the items separated by single spaces, which is how a serialized sequence reads.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, item) in self.items.iter().enumerate() {
            if index > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{item}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::Tree;

    #[test]
    fn conversion_to_and_from_the_data_model_sequence() -> XPathResult<()> {
        let value = XpathValue::from_iter([AtomicValue::string("a"), AtomicValue::string("b")]);
        let sequence = value.clone().into_sequence()?;
        assert_eq!(sequence.len(), 2);
        assert_eq!(XpathValue::from(sequence), value);
        Ok(())
    }

    #[test]
    fn a_function_item_round_trips_through_the_data_model() -> XPathResult<()> {
        let function =
            XpathFunction::native(None, 0, |_context, _arguments| Ok(XpathValue::empty()));
        let value = XpathValue::singleton(function);
        assert!(value.has_function_items());
        // It converts — XDM 3.1 §2.8.2 makes a function item an item — and comes back callable.
        let sequence = value.clone().into_sequence()?;
        assert_eq!(sequence.len(), 1);
        assert!(sequence.get(0).is_some_and(Item::is_function));
        let restored = XpathValue::from(sequence);
        assert!(restored.as_slice()[0].as_callable().is_some());
        // But it still has no effective boolean value and cannot be atomized.
        assert!(value.effective_boolean_value().is_err());
        assert!(value.atomize().is_err());
        Ok(())
    }

    #[test]
    fn a_function_item_keeps_its_identity_across_the_data_model() -> XPathResult<()> {
        let function =
            XpathFunction::native(None, 1, |_context, _arguments| Ok(XpathValue::empty()));
        assert_eq!(function.to_xdm(), function.clone().to_xdm());
        Ok(())
    }

    #[test]
    fn effective_boolean_value_follows_the_data_model() -> XPathResult<()> {
        assert!(!XpathValue::empty().effective_boolean_value()?);
        assert!(XpathValue::from(AtomicValue::string("x")).effective_boolean_value()?);
        assert!(!XpathValue::from(AtomicValue::string("")).effective_boolean_value()?);
        let tree = Tree::parse_str("<a/>")?;
        assert!(XpathValue::from(tree.root()).effective_boolean_value()?);
        Ok(())
    }

    #[test]
    fn nodes_are_deduplicated_and_ordered() -> XPathResult<()> {
        let tree = Tree::parse_str("<a><b/><c/></a>")?;
        let root = tree.document_element().expect("a document element");
        let children = root.children();
        let value = XpathValue::from_iter([
            children[1].clone(),
            children[0].clone(),
            children[1].clone(),
        ]);
        assert_eq!(value.nodes(), children);
        assert!(value.is_all_nodes());
        Ok(())
    }

    #[test]
    fn display_joins_items_with_spaces() {
        let value = XpathValue::from_iter([AtomicValue::integer(1_i64), AtomicValue::string("x")]);
        assert_eq!(value.to_string(), "1 x");
    }
}
