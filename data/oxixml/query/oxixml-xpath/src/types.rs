//! The XPath 3.1 type syntax: [`SequenceType`], [`ItemType`], [`KindTest`] and [`NameTest`],
//! together with the *matching* rules `instance of`, `treat as` and a node test apply.
//!
//! Every name in a type has already been resolved against the static context by the parser, so
//! the structures here carry [`ExpandedName`]s and [`AtomicType`]s rather than lexical `QName`s:
//! `element(chapter)` in a context whose default element namespace is `urn:b` is
//! `{urn:b}chapter` by the time it reaches this module, and `xs:integer` is
//! [`AtomicType::Integer`].
//!
//! # What this engine can and cannot check
//!
//! Matching is exact for node kinds, node names, atomic types, maps and arrays. Two checks are
//! necessarily weaker, because this engine has no schema and no static type system:
//!
//! * a *type annotation* (`element(a, xs:string)`) matches only when the node carries that
//!   annotation. An unvalidated tree carries none explicitly, but XDM still gives its elements
//!   `xs:untyped` and its attributes `xs:untypedAtomic`, and those implicit annotations are
//!   honoured — so `element(*, xs:untyped)` and `element(*, xs:anyType)` match, while
//!   `element(a, xs:string)` does not;
//! * `schema-element(N)` and `schema-attribute(N)` match a node whose name is `N`, since without
//!   a schema there are no substitution groups to widen the test with;
//! * a *typed function test* (`function(xs:integer) as xs:integer`) is exact for an item that
//!   declares a signature — an inline function, and a reference to a built-in whose signature
//!   [`oxixml_fn::signature`] records — and falls back to comparing arity alone for one that
//!   does not, since a function item is otherwise an opaque closure. The families that still
//!   declare nothing are the higher-order, resource and `math:` built-ins.

use crate::context::StaticContext;
use crate::error::{XPathError, XPathErrorCode, XPathResult};
use crate::value::{XpathItem, XpathValue};
use oxixml_infoset::{
    AtomicType, AtomicValue, ExpandedName, Item, NodeKind, NodeRef, XSD_NAMESPACE, XdmArray,
    XdmMap, XdmNode,
};
use std::fmt;

/// A node name test: the `NameTest` production, with the namespace already resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NameTest {
    /// `*` — any name.
    Any,
    /// `local`, `prefix:local` or `Q{uri}local` — one exact expanded name.
    Name(ExpandedName),
    /// `prefix:*` or `Q{uri}*` — any local name in one namespace. `Q{}*` names *no* namespace,
    /// and is held here as the empty string.
    InNamespace(String),
    /// `*:local` — one local name in any namespace, including no namespace.
    WithLocal(String),
}

impl NameTest {
    /// Whether `node`'s name satisfies the test. A node with no name never matches anything but
    /// [`NameTest::Any`].
    #[must_use]
    pub fn matches(&self, node: &NodeRef) -> bool {
        let name = node.node_name();
        match (self, name) {
            (Self::Any, _) => true,
            (_, None) => false,
            (Self::Name(expected), Some(actual)) => *expected == actual,
            // XPath 3.1 §3.3.2.1: `Q{}*` is the wildcard for names in *no* namespace, so the
            // empty braced URI matches a node whose expanded name has no namespace at all
            // (`nametest-22`, `eqname-013`, `eqname-016`).
            (Self::InNamespace(uri), Some(actual)) => {
                actual.namespace() == (!uri.is_empty()).then_some(uri.as_str())
            }
            (Self::WithLocal(local), Some(actual)) => actual.local_name() == local,
        }
    }
}

impl fmt::Display for NameTest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Any => f.write_str("*"),
            Self::Name(name) => match name.namespace() {
                Some(uri) => write!(f, "Q{{{uri}}}{}", name.local_name()),
                None => f.write_str(name.local_name()),
            },
            Self::InNamespace(uri) => write!(f, "Q{{{uri}}}*"),
            Self::WithLocal(local) => write!(f, "*:{local}"),
        }
    }
}

/// A `KindTest`: the node tests that select on node kind rather than on name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KindTest {
    /// `node()`
    AnyKind,
    /// `document-node()`, optionally constrained by the test its single element child must
    /// satisfy.
    Document(Option<Box<KindTest>>),
    /// `element()`, `element(N)`, `element(N, T)` and `element(N, T?)`.
    Element {
        /// `None` for `element()` and `element(*)`.
        name: Option<ExpandedName>,
        /// The required type annotation, if the test names one.
        type_name: Option<ExpandedName>,
        /// Whether the trailing `?` allowed a nilled element.
        nillable: bool,
    },
    /// `attribute()`, `attribute(N)` and `attribute(N, T)`.
    Attribute {
        /// `None` for `attribute()` and `attribute(*)`.
        name: Option<ExpandedName>,
        /// The required type annotation, if the test names one.
        type_name: Option<ExpandedName>,
    },
    /// `schema-element(N)`.
    SchemaElement(ExpandedName),
    /// `schema-attribute(N)`.
    SchemaAttribute(ExpandedName),
    /// `processing-instruction()` and `processing-instruction(target)`.
    ProcessingInstruction(Option<String>),
    /// `comment()`
    Comment,
    /// `text()`
    Text,
    /// `namespace-node()`
    NamespaceNode,
}

impl KindTest {
    /// The node kind this test selects, when it selects exactly one — used to skip whole axes
    /// cheaply.
    #[must_use]
    pub const fn node_kind(&self) -> Option<NodeKind> {
        match self {
            Self::AnyKind => None,
            Self::Document(_) => Some(NodeKind::Document),
            Self::Element { .. } | Self::SchemaElement(_) => Some(NodeKind::Element),
            Self::Attribute { .. } | Self::SchemaAttribute(_) => Some(NodeKind::Attribute),
            Self::ProcessingInstruction(_) => Some(NodeKind::ProcessingInstruction),
            Self::Comment => Some(NodeKind::Comment),
            Self::Text => Some(NodeKind::Text),
            Self::NamespaceNode => Some(NodeKind::Namespace),
        }
    }

    /// Whether `node` satisfies the test.
    #[must_use]
    pub fn matches(&self, node: &NodeRef) -> bool {
        match self {
            Self::AnyKind => true,
            Self::Comment => node.node_kind() == NodeKind::Comment,
            Self::Text => node.node_kind() == NodeKind::Text,
            Self::NamespaceNode => node.node_kind() == NodeKind::Namespace,
            Self::ProcessingInstruction(target) => {
                node.node_kind() == NodeKind::ProcessingInstruction
                    && match target {
                        None => true,
                        Some(target) => node
                            .node_name()
                            .is_some_and(|name| name.local_name() == target),
                    }
            }
            Self::Document(inner) => {
                if node.node_kind() != NodeKind::Document {
                    return false;
                }
                let Some(inner) = inner else {
                    return true;
                };
                let mut elements = node
                    .children()
                    .into_iter()
                    .filter(|child| child.node_kind() == NodeKind::Element);
                match (elements.next(), elements.next()) {
                    (Some(only), None) => inner.matches(&only),
                    _ => false,
                }
            }
            Self::Element {
                name,
                type_name,
                nillable,
            } => {
                node.node_kind() == NodeKind::Element
                    && name_matches(node, name.as_ref())
                    && type_matches(node, type_name.as_ref())
                    // The trailing `?` belongs to the type half of the test, and so does the
                    // rule it relaxes: XPath 3.1 §2.5.5.3 excludes a nilled element from
                    // `element(N, T)` but not from `element()`, `element(*)` or `element(N)`,
                    // which "match any single element node, regardless of its name or type
                    // annotation". A tree that was never validated has no nilled element in it
                    // either way.
                    && (type_name.is_none() || *nillable || node.nilled() != Some(true))
            }
            Self::Attribute { name, type_name } => {
                node.node_kind() == NodeKind::Attribute
                    && name_matches(node, name.as_ref())
                    && type_matches(node, type_name.as_ref())
            }
            Self::SchemaElement(name) => {
                node.node_kind() == NodeKind::Element && name_matches(node, Some(name))
            }
            Self::SchemaAttribute(name) => {
                node.node_kind() == NodeKind::Attribute && name_matches(node, Some(name))
            }
        }
    }
}

fn name_matches(node: &NodeRef, expected: Option<&ExpandedName>) -> bool {
    match expected {
        None => true,
        Some(expected) => node.node_name().is_some_and(|actual| actual == *expected),
    }
}

/// Whether a node's type annotation satisfies the `, TypeName` part of an element or attribute
/// test, per [XPath 3.1 §2.5.5.3](https://www.w3.org/TR/xpath-31/#doc-xpath31-ElementTest).
///
/// A node in a tree that was never schema-validated carries *no explicit* annotation, but XDM
/// gives it one all the same: `xs:untyped` for an element and `xs:untypedAtomic` for an
/// attribute. The rule the specification states is "the annotation is `T`, or is derived from
/// `T`", and both of those implicit annotations derive from `xs:anyType` — so
/// `element(*, xs:untyped)` and `element(*, xs:anyType)` both match a constructed element, which
/// is exactly what the suite's `ForExprType` and `ExternalVariablesWith` cases require.
fn type_matches(node: &NodeRef, expected: Option<&ExpandedName>) -> bool {
    let Some(expected) = expected else {
        return true;
    };
    if let Some(actual) = node.type_name() {
        return derives_from(&actual, expected);
    }
    if expected.namespace() != Some(XSD_NAMESPACE) {
        return false;
    }
    let implicit = match node.node_kind() {
        NodeKind::Attribute => "untypedAtomic",
        _ => "untyped",
    };
    expected.local_name() == implicit
        || expected.local_name() == "anyType"
        || (node.node_kind() == NodeKind::Attribute
            && matches!(expected.local_name(), "anySimpleType" | "anyAtomicType"))
}

/// Whether the type named `actual` *is or is derived from* the type named `expected`.
///
/// XPath 3.1 §2.5.5.3 states the element-test rule that way, not as equality: an element whose
/// annotation an XQuery `declare construction strip` prolog pins to `xs:untyped` still satisfies
/// `element(*, xs:anyType)`, because every type derives from `xs:anyType` (`constprolog-5`,
/// `-6`, `-7`). Only the built-in hierarchy is known here — a user-declared type is matched by
/// name — which is all a schema-less processor can decide.
fn derives_from(actual: &ExpandedName, expected: &ExpandedName) -> bool {
    if actual == expected {
        return true;
    }
    if expected.namespace() != Some(XSD_NAMESPACE) || actual.namespace() != Some(XSD_NAMESPACE) {
        return false;
    }
    match expected.local_name() {
        // The root of the whole type hierarchy, complex types included.
        "anyType" => true,
        // Every simple type derives from it; the two complex ones above do not.
        "anySimpleType" => !matches!(actual.local_name(), "anyType" | "untyped"),
        _ => {
            let Some(from) = AtomicType::from_local_name(actual.local_name()) else {
                return false;
            };
            let Some(to) = AtomicType::from_local_name(expected.local_name()) else {
                return false;
            };
            let mut current = Some(from);
            while let Some(step) = current {
                if step == to {
                    return true;
                }
                current = step.base_type();
            }
            false
        }
    }
}

impl fmt::Display for KindTest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AnyKind => f.write_str("node()"),
            Self::Comment => f.write_str("comment()"),
            Self::Text => f.write_str("text()"),
            Self::NamespaceNode => f.write_str("namespace-node()"),
            Self::ProcessingInstruction(None) => f.write_str("processing-instruction()"),
            Self::ProcessingInstruction(Some(target)) => {
                write!(f, "processing-instruction({target})")
            }
            Self::Document(None) => f.write_str("document-node()"),
            Self::Document(Some(inner)) => write!(f, "document-node({inner})"),
            Self::Element {
                name,
                type_name,
                nillable,
            } => {
                f.write_str("element(")?;
                write_name(f, name.as_ref())?;
                if let Some(type_name) = type_name {
                    write!(f, ", {type_name}")?;
                    if *nillable {
                        f.write_str("?")?;
                    }
                }
                f.write_str(")")
            }
            Self::Attribute { name, type_name } => {
                f.write_str("attribute(")?;
                write_name(f, name.as_ref())?;
                if let Some(type_name) = type_name {
                    write!(f, ", {type_name}")?;
                }
                f.write_str(")")
            }
            Self::SchemaElement(name) => write!(f, "schema-element({name})"),
            Self::SchemaAttribute(name) => write!(f, "schema-attribute({name})"),
        }
    }
}

fn write_name(f: &mut fmt::Formatter<'_>, name: Option<&ExpandedName>) -> fmt::Result {
    match name {
        None => f.write_str("*"),
        Some(name) => write!(f, "{name}"),
    }
}

/// An `ItemType`: what a single item must be.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ItemType {
    /// `item()` — anything.
    Item,
    /// An `AtomicOrUnionType`, such as `xs:integer`.
    Atomic(AtomicType),
    /// `xs:numeric`, the union of `xs:double`, `xs:float` and `xs:decimal` that XPath 3.1
    /// defines for use in a `SequenceType`. It is not a schema type and cannot be a cast target.
    Numeric,
    /// A `KindTest`.
    Kind(KindTest),
    /// `function(*)`
    AnyFunction,
    /// `function(T, …) as R`, judged against the item's declared signature when it has one and
    /// on arity alone when it has not — see the module documentation.
    TypedFunction {
        /// The declared parameter types.
        parameters: Vec<SequenceType>,
        /// The declared result type.
        result: Box<SequenceType>,
    },
    /// `map(*)`
    AnyMap,
    /// `map(K, V)`
    TypedMap {
        /// The required key type.
        key: AtomicType,
        /// The required value type.
        value: Box<SequenceType>,
    },
    /// `array(*)`
    AnyArray,
    /// `array(T)`
    TypedArray(Box<SequenceType>),
}

impl ItemType {
    /// Whether `item` matches this type.
    #[must_use]
    pub fn matches(&self, item: &XpathItem) -> bool {
        match self {
            Self::Item => true,
            Self::Atomic(required) => item
                .as_atomic()
                .is_some_and(|value| matches_atomic_type(value, *required)),
            Self::Numeric => item.as_atomic().is_some_and(AtomicValue::is_numeric),
            Self::Kind(test) => item.as_node().is_some_and(|node| test.matches(node)),
            Self::AnyFunction => item.is_function(),
            Self::TypedFunction { parameters, result } => match item {
                // XPath 3.1 §2.5.5.5 judges a function item by its *declared* signature, with
                // the parameters contravariant and the result covariant. An item that declares
                // none — a built-in wrapped from the library — can only be checked for arity,
                // which is what this engine has always done.
                XpathItem::Function(function) => {
                    function.arity() == parameters.len()
                        && function.signature().is_none_or(|signature| {
                            parameters
                                .iter()
                                .zip(signature.parameters.iter())
                                .all(|(required, declared)| required.is_subtype_of(declared))
                                && signature.result.is_subtype_of(result)
                        })
                }
                // XPath 3.1 §2.5.5.6: `map(K, V)` is a subtype of
                // `function(xs:anyAtomicType) as V?` — the `?` because a lookup of a key the map
                // does not hold yields the empty sequence. A required result type that forbids
                // the empty sequence is therefore never matched by a map, however its values are
                // typed (QT3 `MapTest-061`, `-062`, `-066`).
                XpathItem::Value(Item::Map(map)) => {
                    parameters.len() == 1
                        && result.admits_the_empty_sequence()
                        && map
                            .entries()
                            .all(|(_, value)| result.matches(&XpathValue::from(value.clone())))
                }
                // `array(T)` is a subtype of `function(xs:integer) as T`: an in-range lookup
                // always yields a member, so the empty sequence need not be admitted.
                XpathItem::Value(Item::Array(array)) => {
                    parameters.len() == 1
                        && array
                            .members()
                            .all(|member| result.matches(&XpathValue::from(member.clone())))
                }
                XpathItem::Value(_) => false,
            },
            Self::AnyMap => item.as_map().is_some(),
            Self::TypedMap { key, value } => item
                .as_map()
                .is_some_and(|map| matches_typed_map(map, *key, value)),
            Self::AnyArray => item.as_array().is_some(),
            Self::TypedArray(member) => item
                .as_array()
                .is_some_and(|array| matches_typed_array(array, member)),
        }
    }
}

impl ItemType {
    /// Whether every item of this type is also an item of `required`, as far as this
    /// implementation can establish — the item-type half of
    /// [`SequenceType::is_subtype_of`](SequenceType::is_subtype_of).
    ///
    /// Function types are related the way XPath 3.1 §2.5.6.2 requires: the *parameter* types
    /// contravariantly (the required signature's parameters must be subtypes of this one's, so
    /// a function accepting fewer kinds of argument is not a subtype) and the *result* type
    /// covariantly. `map(K, V)` and `array(T)` are also function types, and are related through
    /// the signatures §2.5.5.6 gives them.
    #[must_use]
    pub fn is_subtype_of(&self, required: &Self) -> bool {
        match (self, required) {
            (_, Self::Item) => true,
            (Self::Item, _) => false,
            (Self::Atomic(actual), Self::Atomic(required)) => actual.is_derived_from(*required),
            (Self::Numeric, Self::Numeric) => true,
            (Self::Atomic(actual), Self::Numeric) => actual.is_numeric(),
            // `xs:numeric` is the union of xs:double, xs:float and xs:decimal, whose only common
            // atomic supertype is xs:anyAtomicType — so it is a subtype of that, and of nothing
            // narrower. Without this a `function(xs:anyAtomicType) as R` is wrongly judged not to
            // subsume a `function(xs:numeric) as R`, since the parameter check runs `xs:numeric`
            // against `xs:anyAtomicType` contravariantly.
            (Self::Numeric, Self::Atomic(required)) => *required == AtomicType::AnyAtomicType,
            (Self::Kind(actual), Self::Kind(required)) => actual.is_subtype_of(required),
            (
                Self::AnyFunction
                | Self::TypedFunction { .. }
                | Self::AnyMap
                | Self::TypedMap { .. }
                | Self::AnyArray
                | Self::TypedArray(_),
                Self::AnyFunction,
            ) => true,
            (Self::AnyMap | Self::TypedMap { .. }, Self::AnyMap) => true,
            (Self::AnyArray | Self::TypedArray(_), Self::AnyArray) => true,
            (Self::TypedMap { key, value }, Self::TypedMap { key: k, value: v }) => {
                key.is_derived_from(*k) && value.is_subtype_of(v)
            }
            (Self::TypedArray(member), Self::TypedArray(required)) => {
                member.is_subtype_of(required)
            }
            (actual, Self::TypedFunction { parameters, result }) => match actual.as_signature() {
                Some((actual_parameters, actual_result)) => {
                    actual_parameters.len() == parameters.len()
                        && parameters
                            .iter()
                            .zip(actual_parameters.iter())
                            .all(|(required, actual)| required.is_subtype_of(actual))
                        && actual_result.is_subtype_of(result)
                }
                None => false,
            },
            _ => false,
        }
    }

    /// The signature this item type guarantees, when it is a function type with one.
    ///
    /// `map(K, V)` is `function(xs:anyAtomicType) as V?` and `array(T)` is
    /// `function(xs:integer) as T` (XPath 3.1 §2.5.5.6); `function(*)`, `map(*)` and `array(*)`
    /// guarantee no particular signature and answer [`None`].
    fn as_signature(&self) -> Option<(Vec<SequenceType>, SequenceType)> {
        match self {
            Self::TypedFunction { parameters, result } => {
                Some((parameters.clone(), (**result).clone()))
            }
            Self::TypedMap { value, .. } => Some((
                vec![SequenceType::new(
                    ItemType::Atomic(AtomicType::AnyAtomicType),
                    Occurrence::One,
                )],
                match (**value).clone() {
                    SequenceType::Empty => SequenceType::Empty,
                    SequenceType::Items { item_type, .. } => {
                        SequenceType::new(item_type, Occurrence::ZeroOrMore)
                    }
                },
            )),
            Self::TypedArray(member) => Some((
                vec![SequenceType::new(
                    ItemType::Atomic(AtomicType::Integer),
                    Occurrence::One,
                )],
                (**member).clone(),
            )),
            _ => None,
        }
    }
}

impl KindTest {
    /// Whether every node this test selects is also selected by `required` — the kind-test half
    /// of the `subtype-itemtype` judgement of [XPath 3.1
    /// §2.5.6.2](https://www.w3.org/TR/xpath-31/#id-sequencetype-subtype).
    ///
    /// The specification's list is followed literally, including its two asymmetries. A test that
    /// names no type is *not* a subtype of one that does, however general that type is — so
    /// `element(e)` is not a subtype of `element(e, xs:anyType)` — while `element()` and
    /// `element(*)`, which name neither a name nor a type, subsume every element test there is.
    /// xslt30 `higher-order-functions-034` gates all eight combinations of that at once.
    ///
    /// Two relations need a schema and so are answered conservatively: `schema-element(N)` and
    /// `schema-attribute(N)` are judged subtypes of themselves alone, since deciding anything
    /// more asks which declarations are in a substitution group.
    #[must_use]
    pub fn is_subtype_of(&self, required: &Self) -> bool {
        match (self, required) {
            // `node()` selects every node, so every test is a subtype of it — and, being the
            // only test that selects nodes of every kind, it is a subtype of nothing else. (That
            // second half is the fallback's doing: `node()` equals no other test.)
            (_, Self::AnyKind) => true,
            // `document-node()` subsumes every document test; a constrained one is decided by
            // the test its element child must satisfy.
            (Self::Document(_), Self::Document(None)) => true,
            (Self::Document(Some(actual)), Self::Document(Some(required))) => {
                actual.is_subtype_of(required)
            }
            // `processing-instruction()` subsumes the targeted form; two targeted tests are
            // related only when they name the same target.
            (Self::ProcessingInstruction(_), Self::ProcessingInstruction(None)) => true,
            // `element()`/`element(*)` and `attribute()`/`attribute(*)` subsume every test of
            // their own kind. A schema test is not one of them: `schema-element(N)` is a
            // `SchemaElementTest`, not an `ElementTest`, and the specification relates it to
            // nothing else.
            (
                Self::Element { .. },
                Self::Element {
                    name: None,
                    type_name: None,
                    ..
                },
            )
            | (
                Self::Attribute { .. },
                Self::Attribute {
                    name: None,
                    type_name: None,
                },
            ) => true,
            (
                Self::Element {
                    name: actual,
                    type_name: None,
                    ..
                },
                Self::Element {
                    name: required @ Some(_),
                    type_name: None,
                    ..
                },
            )
            | (
                Self::Attribute {
                    name: actual,
                    type_name: None,
                },
                Self::Attribute {
                    name: required @ Some(_),
                    type_name: None,
                },
            ) => actual == required,
            (
                Self::Element {
                    name: actual,
                    type_name: Some(actual_type),
                    nillable: actual_nillable,
                },
                Self::Element {
                    name: required,
                    type_name: Some(required_type),
                    nillable: required_nillable,
                },
            ) => {
                // `element(N, T)` admits no nilled element, so it is not subsumed by a test that
                // demands one be admitted; `element(N, T?)` is subsumed by either.
                (*required_nillable || !*actual_nillable)
                    && name_subsumes(required.as_ref(), actual.as_ref())
                    && derives_from(actual_type, required_type)
            }
            (
                Self::Attribute {
                    name: actual,
                    type_name: Some(actual_type),
                },
                Self::Attribute {
                    name: required,
                    type_name: Some(required_type),
                },
            ) => {
                name_subsumes(required.as_ref(), actual.as_ref())
                    && derives_from(actual_type, required_type)
            }
            // Everything left is related only to itself: `text()`, `comment()`,
            // `namespace-node()`, the two schema tests, one target-naming
            // `processing-instruction(N)` against another, and every pair of tests over
            // different node kinds.
            _ => self == required,
        }
    }
}

/// Whether the name half of a kind test that names a type accepts every name the other does:
/// `*` (spelled [`None`] here) accepts them all, and any other name only itself.
///
/// Only the forms that name a type use this. `element(N)` and `element()` are related by their
/// own rules, which do not have the wildcard's leniency — `element(e)` is not a subtype of
/// `element(*, xs:anyType)`, because the latter names a type and the former does not.
fn name_subsumes(required: Option<&ExpandedName>, actual: Option<&ExpandedName>) -> bool {
    match required {
        None => true,
        Some(required) => actual == Some(required),
    }
}

fn matches_atomic_type(value: &AtomicValue, required: AtomicType) -> bool {
    value.atomic_type().is_derived_from(required)
}

fn matches_typed_map(map: &XdmMap, key: AtomicType, value: &SequenceType) -> bool {
    map.entries().all(|(entry_key, entry_value)| {
        matches_atomic_type(entry_key, key) && value.matches(&XpathValue::from(entry_value.clone()))
    })
}

fn matches_typed_array(array: &XdmArray, member: &SequenceType) -> bool {
    array
        .members()
        .all(|value| member.matches(&XpathValue::from(value.clone())))
}

impl fmt::Display for ItemType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Item => f.write_str("item()"),
            Self::Atomic(atomic) => write!(f, "xs:{}", atomic.local_name()),
            Self::Numeric => f.write_str("xs:numeric"),
            Self::Kind(test) => write!(f, "{test}"),
            Self::AnyFunction => f.write_str("function(*)"),
            Self::TypedFunction { parameters, result } => {
                f.write_str("function(")?;
                for (index, parameter) in parameters.iter().enumerate() {
                    if index > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{parameter}")?;
                }
                write!(f, ") as {result}")
            }
            Self::AnyMap => f.write_str("map(*)"),
            Self::TypedMap { key, value } => {
                write!(f, "map(xs:{}, {value})", key.local_name())
            }
            Self::AnyArray => f.write_str("array(*)"),
            Self::TypedArray(member) => write!(f, "array({member})"),
        }
    }
}

/// The `OccurrenceIndicator` a `SequenceType` may carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Occurrence {
    /// No indicator: exactly one item.
    #[default]
    One,
    /// `?` — zero or one item.
    ZeroOrOne,
    /// `*` — any number of items.
    ZeroOrMore,
    /// `+` — one or more items.
    OneOrMore,
}

impl Occurrence {
    /// Whether a sequence of `count` items satisfies this indicator.
    #[must_use]
    pub const fn allows(self, count: usize) -> bool {
        match self {
            Self::One => count == 1,
            Self::ZeroOrOne => count <= 1,
            Self::ZeroOrMore => true,
            Self::OneOrMore => count >= 1,
        }
    }

    /// Whether every count this indicator allows is also allowed by `other`.
    #[must_use]
    pub const fn is_subtype_of(self, other: Self) -> bool {
        matches!(
            (self, other),
            (Self::One, _)
                | (Self::ZeroOrOne, Self::ZeroOrOne | Self::ZeroOrMore)
                | (Self::OneOrMore, Self::OneOrMore | Self::ZeroOrMore)
                | (Self::ZeroOrMore, Self::ZeroOrMore)
        )
    }

    /// The indicator's spelling, empty for [`Occurrence::One`].
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::One => "",
            Self::ZeroOrOne => "?",
            Self::ZeroOrMore => "*",
            Self::OneOrMore => "+",
        }
    }
}

/// A `SequenceType`: an item type plus how many items are allowed.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{AtomicValue, Tree};
/// use oxixml_xpath::{ItemType, KindTest, Occurrence, SequenceType, XpathValue};
/// use oxixml_infoset::AtomicType;
///
/// let optional_integer = SequenceType::new(ItemType::Atomic(AtomicType::Integer), Occurrence::ZeroOrOne);
/// assert!(optional_integer.matches(&XpathValue::empty()));
/// assert!(optional_integer.matches(&XpathValue::from(AtomicValue::integer(1_i64))));
/// assert!(!optional_integer.matches(&XpathValue::from(AtomicValue::string("1"))));
///
/// let any_element = SequenceType::new(
///     ItemType::Kind(KindTest::Element { name: None, type_name: None, nillable: false }),
///     Occurrence::ZeroOrMore,
/// );
/// let tree = Tree::parse_str("<a><b/></a>")?;
/// assert!(any_element.matches(&XpathValue::from(tree.document_element().expect("root"))));
/// assert!(!any_element.matches(&XpathValue::from(tree.root())));
/// # Ok::<(), oxixml_xpath::XPathError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SequenceType {
    /// `empty-sequence()` — matches only the empty sequence.
    Empty,
    /// An item type with an occurrence indicator.
    Items {
        /// What each item must be.
        item_type: ItemType,
        /// How many items are allowed.
        occurrence: Occurrence,
    },
}

impl SequenceType {
    /// Parses a `SequenceType` written in the XPath 3.1 type syntax.
    ///
    /// The names inside it — atomic types, element and attribute names, `Q{uri}local` — are
    /// resolved against `context`, exactly as they would be inside an expression. A host
    /// language that declares types away from any expression (XQuery's `declare function`,
    /// `declare variable` and `declare context item`) uses this to obtain the same type this
    /// engine would have parsed from an `instance of`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_xpath::{SequenceType, StaticContext};
    ///
    /// let declared = SequenceType::parse("xs:integer?", &StaticContext::new())?;
    /// assert!(declared.admits_the_empty_sequence());
    /// # Ok::<(), oxixml_xpath::XPathError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// `XPST0003` when the text is not a sequence type, `XPST0051` for an unknown atomic type,
    /// and `XPST0081` for an unbound namespace prefix.
    pub fn parse(text: &str, context: &StaticContext) -> XPathResult<Self> {
        crate::parser::parse_type(text, context)
    }

    /// Whether every value of this type is also a value of `other` — the *subtype* relation of
    /// [XPath 3.1 §2.5.6.1](https://www.w3.org/TR/xpath-31/#id-sequencetype-subtype).
    ///
    /// The answer is conservative: `true` means the containment is certain, and `false` means
    /// only that this implementation cannot establish it. That is exactly what a `function(…)`
    /// item test needs, since an unprovable containment must make the test fail rather than
    /// succeed.
    #[must_use]
    pub fn is_subtype_of(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Empty, _) => other.admits_the_empty_sequence(),
            (Self::Items { .. }, Self::Empty) => false,
            (
                Self::Items {
                    item_type,
                    occurrence,
                },
                Self::Items {
                    item_type: required,
                    occurrence: allowed,
                },
            ) => occurrence.is_subtype_of(*allowed) && item_type.is_subtype_of(required),
        }
    }

    /// Whether the empty sequence matches this type.
    #[must_use]
    pub const fn admits_the_empty_sequence(&self) -> bool {
        match self {
            Self::Empty => true,
            Self::Items { occurrence, .. } => occurrence.allows(0),
        }
    }

    /// Builds an item type with an occurrence indicator.
    #[must_use]
    pub const fn new(item_type: ItemType, occurrence: Occurrence) -> Self {
        Self::Items {
            item_type,
            occurrence,
        }
    }

    /// `item()*`, the type every expression trivially satisfies.
    #[must_use]
    pub const fn any() -> Self {
        Self::new(ItemType::Item, Occurrence::ZeroOrMore)
    }

    /// Whether `value` matches this type.
    #[must_use]
    pub fn matches(&self, value: &XpathValue) -> bool {
        match self {
            Self::Empty => value.is_empty(),
            Self::Items {
                item_type,
                occurrence,
            } => occurrence.allows(value.len()) && value.iter().all(|item| item_type.matches(item)),
        }
    }

    /// Checks `value` against this type, as `treat as` does.
    ///
    /// # Errors
    ///
    /// [`XPathErrorCode::Xpdy0050`] when the value does not match.
    pub fn treat(&self, value: XpathValue) -> XPathResult<XpathValue> {
        if self.matches(&value) {
            return Ok(value);
        }
        Err(XPathError::new(
            XPathErrorCode::Xpdy0050,
            format!(
                "a sequence of {} item(s) does not match `{self}`",
                value.len()
            ),
        ))
    }
}

impl fmt::Display for SequenceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("empty-sequence()"),
            Self::Items {
                item_type,
                occurrence,
            } => write!(f, "{item_type}{}", occurrence.as_str()),
        }
    }
}

/// A `SingleType`: the target of `cast as` and `castable as`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SingleType {
    /// The target atomic type.
    pub atomic_type: AtomicType,
    /// Whether a trailing `?` allowed the empty sequence.
    pub optional: bool,
}

impl SingleType {
    /// Builds a single type.
    #[must_use]
    pub const fn new(atomic_type: AtomicType, optional: bool) -> Self {
        Self {
            atomic_type,
            optional,
        }
    }
}

impl fmt::Display for SingleType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "xs:{}", self.atomic_type.local_name())?;
        if self.optional {
            f.write_str("?")?;
        }
        Ok(())
    }
}

/// One of the three built-in *list* types, as the target of `cast as` and `castable as`.
///
/// [XPath 3.1 §19.3.4](https://www.w3.org/TR/xpath-31/#casting-to-list-types) casts to a list
/// type by taking the input's string value, splitting it on white space, and casting each token
/// to the list's *item* type — so the result is a sequence rather than a single value, which is
/// why it cannot be carried by a [`SingleType`]. `xs:IDREFS`, `xs:NMTOKENS` and `xs:ENTITIES` are
/// the only ones a processor without a schema has, and each is declared with `minLength = 1`, so
/// a value that yields no token at all is not in the type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ListType {
    /// The list's item type: `xs:IDREF`, `xs:NMTOKEN` or `xs:ENTITY`.
    pub member: AtomicType,
    /// Whether a trailing `?` allowed the empty sequence as the *input*.
    pub optional: bool,
}

impl ListType {
    /// Builds a list type from its item type.
    #[must_use]
    pub const fn new(member: AtomicType, optional: bool) -> Self {
        Self { member, optional }
    }

    /// The list type's own name, as XML Schema spells it.
    #[must_use]
    pub const fn local_name(&self) -> &'static str {
        match self.member {
            AtomicType::Idref => "IDREFS",
            AtomicType::Nmtoken => "NMTOKENS",
            AtomicType::Entity => "ENTITIES",
            // No other list type is built in, and the parser builds no other member.
            _ => "anySimpleType",
        }
    }
}

impl fmt::Display for ListType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "xs:{}", self.local_name())?;
        if self.optional {
            f.write_str("?")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::Tree;

    /// XPath 3.1 §2.5.6.2: a function type's parameters are *contravariant* and its result is
    /// covariant.
    #[test]
    fn function_types_relate_contravariantly() {
        let one = |item_type: ItemType| SequenceType::new(item_type, Occurrence::One);
        let string_to_item = ItemType::TypedFunction {
            parameters: vec![one(ItemType::Atomic(AtomicType::String))],
            result: Box::new(one(ItemType::Item)),
        };
        let item_to_item = ItemType::TypedFunction {
            parameters: vec![one(ItemType::Item)],
            result: Box::new(one(ItemType::Item)),
        };
        assert!(
            item_to_item.is_subtype_of(&string_to_item),
            "a function accepting any item may stand in for one accepting a string"
        );
        assert!(
            !string_to_item.is_subtype_of(&item_to_item),
            "but not the other way round"
        );

        let string_to_string = ItemType::TypedFunction {
            parameters: vec![one(ItemType::Item)],
            result: Box::new(one(ItemType::Atomic(AtomicType::String))),
        };
        assert!(
            string_to_string.is_subtype_of(&item_to_item),
            "the result type is covariant"
        );
        assert!(!item_to_item.is_subtype_of(&string_to_string));

        // `map(K, V)` is `function(xs:anyAtomicType) as V*`, and `array(T)` is
        // `function(xs:integer) as T`.
        let map_type = ItemType::TypedMap {
            key: AtomicType::Integer,
            value: Box::new(one(ItemType::Atomic(AtomicType::String))),
        };
        assert!(map_type.is_subtype_of(&ItemType::AnyMap));
        assert!(map_type.is_subtype_of(&ItemType::AnyFunction));
        assert!(!map_type.is_subtype_of(&string_to_item));
        assert!(!ItemType::AnyMap.is_subtype_of(&map_type));
    }

    /// `xs:numeric` spans xs:double, xs:float and xs:decimal, so its only atomic supertype is
    /// `xs:anyAtomicType`. The subtyping relation must reflect that, including through a function
    /// signature's contravariant parameter position.
    #[test]
    fn numeric_is_a_subtype_of_any_atomic_type() {
        assert!(ItemType::Numeric.is_subtype_of(&ItemType::Atomic(AtomicType::AnyAtomicType)));
        assert!(ItemType::Numeric.is_subtype_of(&ItemType::Numeric));
        assert!(ItemType::Numeric.is_subtype_of(&ItemType::Item));
        // But not of any narrower atomic type.
        assert!(!ItemType::Numeric.is_subtype_of(&ItemType::Atomic(AtomicType::Double)));
        assert!(!ItemType::Numeric.is_subtype_of(&ItemType::Atomic(AtomicType::Decimal)));

        // A function taking xs:anyAtomicType therefore subsumes one taking xs:numeric.
        let one = |item_type: ItemType| SequenceType::new(item_type, Occurrence::One);
        let takes_any = ItemType::TypedFunction {
            parameters: vec![one(ItemType::Atomic(AtomicType::AnyAtomicType))],
            result: Box::new(one(ItemType::Item)),
        };
        let takes_numeric = ItemType::TypedFunction {
            parameters: vec![one(ItemType::Numeric)],
            result: Box::new(one(ItemType::Item)),
        };
        assert!(takes_any.is_subtype_of(&takes_numeric));
        assert!(!takes_numeric.is_subtype_of(&takes_any));
    }

    #[test]
    fn occurrence_indicators() {
        assert!(Occurrence::One.allows(1));
        assert!(!Occurrence::One.allows(0));
        assert!(Occurrence::ZeroOrOne.allows(0));
        assert!(!Occurrence::ZeroOrOne.allows(2));
        assert!(Occurrence::ZeroOrMore.allows(9));
        assert!(!Occurrence::OneOrMore.allows(0));
    }

    #[test]
    fn atomic_types_match_their_derivations() {
        let value = XpathValue::from(AtomicValue::integer(1_i64));
        assert!(
            SequenceType::new(ItemType::Atomic(AtomicType::Integer), Occurrence::One)
                .matches(&value)
        );
        assert!(
            SequenceType::new(ItemType::Atomic(AtomicType::Decimal), Occurrence::One)
                .matches(&value)
        );
        assert!(
            SequenceType::new(ItemType::Atomic(AtomicType::AnyAtomicType), Occurrence::One)
                .matches(&value)
        );
        assert!(
            !SequenceType::new(ItemType::Atomic(AtomicType::Double), Occurrence::One)
                .matches(&value)
        );
    }

    #[test]
    fn kind_tests_select_node_kinds() -> XPathResult<()> {
        let tree = Tree::parse_str("<a x='1'><!--c--><?pi go?>text</a>")?;
        let root = tree.document_element().expect("a document element");
        assert!(KindTest::AnyKind.matches(&root));
        assert!(
            KindTest::Element {
                name: Some(ExpandedName::new(None::<String>, "a")),
                type_name: None,
                nillable: false
            }
            .matches(&root)
        );
        assert!(!KindTest::Text.matches(&root));

        let attribute = root.attributes().first().cloned().expect("one attribute");
        assert!(
            KindTest::Attribute {
                name: None,
                type_name: None
            }
            .matches(&attribute)
        );

        let children = root.children();
        assert!(KindTest::Comment.matches(&children[0]));
        assert!(KindTest::ProcessingInstruction(Some("pi".to_owned())).matches(&children[1]));
        assert!(!KindTest::ProcessingInstruction(Some("other".to_owned())).matches(&children[1]));
        assert!(KindTest::Text.matches(&children[2]));

        assert!(KindTest::Document(None).matches(&tree.root()));
        assert!(
            KindTest::Document(Some(Box::new(KindTest::Element {
                name: Some(ExpandedName::new(None::<String>, "a")),
                type_name: None,
                nillable: false
            })))
            .matches(&tree.root())
        );
        Ok(())
    }

    /// xslt30 `higher-order-functions-034`, whole. A stylesheet function declared
    /// `as element(e)?` is tested against eight result types with `instance of`, and the case
    /// records the answer to each; the parameter types are held fixed so that only the result
    /// relation moves. Rows six to eight are the ones worth keeping: a test that names no type
    /// is not a subtype of one that does, however general.
    #[test]
    fn a_declared_result_type_answers_the_suites_eight_instance_of_tests() -> XPathResult<()> {
        let context = StaticContext::new();
        let declared =
            SequenceType::parse("function(xs:long, xs:NCName) as element(e)?", &context)?;
        for (required, expected) in [
            ("element()", false),
            ("element()+", false),
            ("element()?", true),
            ("element()*", true),
            ("element(e)*", true),
            ("element(e, xs:anyType)*", false),
            ("element(*, xs:anyType)?", false),
            ("element(*, xs:untyped)?", false),
        ] {
            let text = format!("function(xs:long, xs:NCName) as {required}");
            let required_type = SequenceType::parse(&text, &context)?;
            assert_eq!(
                declared.is_subtype_of(&required_type),
                expected,
                "`local:f#2 instance of {text}`"
            );
        }
        Ok(())
    }

    /// The rest of XPath 3.1 §2.5.6.2's kind-test list, one assertion per rule.
    #[test]
    fn kind_tests_relate_the_way_the_specification_lists() -> XPathResult<()> {
        let context = StaticContext::new();
        let related = |actual: &str, required: &str| -> XPathResult<bool> {
            Ok(SequenceType::parse(actual, &context)?
                .is_subtype_of(&SequenceType::parse(required, &context)?))
        };
        for (actual, required) in [
            // `node()` subsumes every kind test, and only itself is a supertype of it.
            ("element(e)", "node()"),
            ("namespace-node()", "node()"),
            ("node()", "node()"),
            // A named or typed attribute test against the unnamed one, and by name.
            ("attribute(a)", "attribute()"),
            ("attribute(a, xs:string)", "attribute(*)"),
            ("attribute(a)", "attribute(a)"),
            ("attribute(a, xs:integer)", "attribute(a, xs:decimal)"),
            ("attribute(a, xs:integer)", "attribute(*, xs:decimal)"),
            // A nilled-tolerant element test is subsumed by another that tolerates one; a strict
            // test is subsumed by either.
            ("element(e, xs:integer?)", "element(*, xs:decimal?)"),
            ("element(e, xs:integer)", "element(e, xs:decimal?)"),
            // The document and processing-instruction forms.
            ("document-node(element(e))", "document-node()"),
            ("document-node(element(e))", "document-node(element())"),
            ("processing-instruction(p)", "processing-instruction()"),
            ("processing-instruction(p)", "processing-instruction(p)"),
        ] {
            assert!(related(actual, required)?, "`{actual}` is a `{required}`");
        }
        for (actual, required) in [
            ("node()", "element()"),
            ("node()", "text()"),
            ("element(e)", "attribute(e)"),
            ("text()", "comment()"),
            ("element()", "element(e)"),
            ("attribute()", "attribute(a)"),
            ("attribute(b)", "attribute(a)"),
            ("attribute(a, xs:decimal)", "attribute(a, xs:integer)"),
            ("attribute(a, xs:string)", "attribute(b, xs:string)"),
            // The nilled-tolerant test admits elements the strict one does not.
            ("element(e, xs:integer?)", "element(e, xs:integer)"),
            ("document-node(element(e))", "document-node(element(f))"),
            ("document-node()", "document-node(element(e))"),
            ("processing-instruction()", "processing-instruction(p)"),
            ("processing-instruction(q)", "processing-instruction(p)"),
        ] {
            assert!(!related(actual, required)?, "`{actual}` is no `{required}`");
        }

        // The two schema tests are built rather than parsed — this processor imports no schema,
        // so it rejects the syntax outright — and a substitution group is exactly what deciding
        // anything more about them would need. Each is a subtype of itself, of `node()`, and of
        // nothing else: an `ElementTest` is what `element()` subsumes, and a
        // `SchemaElementTest` is not one.
        let name = |local: &str| ExpandedName::new(None::<String>, local);
        let any_element = KindTest::Element {
            name: None,
            type_name: None,
            nillable: false,
        };
        assert!(
            KindTest::SchemaElement(name("e")).is_subtype_of(&KindTest::SchemaElement(name("e")))
        );
        assert!(KindTest::SchemaElement(name("e")).is_subtype_of(&KindTest::AnyKind));
        assert!(
            !KindTest::SchemaElement(name("e")).is_subtype_of(&KindTest::SchemaElement(name("f")))
        );
        assert!(!KindTest::SchemaElement(name("e")).is_subtype_of(&any_element));
        assert!(!any_element.is_subtype_of(&KindTest::SchemaElement(name("e"))));
        assert!(
            KindTest::SchemaAttribute(name("a"))
                .is_subtype_of(&KindTest::SchemaAttribute(name("a")))
        );
        assert!(
            !KindTest::SchemaAttribute(name("a")).is_subtype_of(&KindTest::Attribute {
                name: None,
                type_name: None,
            })
        );
        Ok(())
    }

    /// XPath 3.1 §2.5.5.3: `element()`, `element(*)` and `element(N)` "match any single element
    /// node, regardless of its name or type annotation" — a nilled element included. Only a test
    /// that names a type excludes one, and only without the trailing `?`.
    #[test]
    fn a_nilled_element_is_excluded_only_by_a_test_that_names_a_type() -> XPathResult<()> {
        let mut builder = oxixml_infoset::TreeBuilder::new();
        builder.start_annotated_element(
            ExpandedName::new(None::<String>, "e"),
            oxixml_infoset::Annotation {
                nilled: Some(true),
                ..oxixml_infoset::Annotation::default()
            },
        )?;
        builder.end_element()?;
        let tree = builder.build()?;
        let element = tree.document_element().expect("a document element");
        assert_eq!(element.nilled(), Some(true));

        let context = StaticContext::new();
        for (test, expected) in [
            ("element()", true),
            ("element(*)", true),
            ("element(e)", true),
            ("element(*, xs:anyType?)", true),
            ("element(*, xs:anyType)", false),
            ("element(e, xs:untyped)", false),
        ] {
            assert_eq!(
                SequenceType::parse(test, &context)?.matches(&XpathValue::from(element.clone())),
                expected,
                "`{test}`"
            );
        }
        Ok(())
    }

    #[test]
    fn name_tests_honour_wildcards() -> XPathResult<()> {
        let tree = Tree::parse_str("<p:a xmlns:p='urn:p'/>")?;
        let root = tree.document_element().expect("a document element");
        assert!(NameTest::Any.matches(&root));
        assert!(NameTest::InNamespace("urn:p".to_owned()).matches(&root));
        assert!(!NameTest::InNamespace("urn:q".to_owned()).matches(&root));
        assert!(NameTest::WithLocal("a".to_owned()).matches(&root));
        assert!(NameTest::Name(ExpandedName::new(Some("urn:p"), "a")).matches(&root));
        assert!(!NameTest::Name(ExpandedName::new(None::<String>, "a")).matches(&root));
        Ok(())
    }

    #[test]
    fn treat_reports_xpdy0050() {
        let required = SequenceType::new(ItemType::Atomic(AtomicType::Integer), Occurrence::One);
        let failure = required
            .treat(XpathValue::empty())
            .expect_err("the empty sequence is not one integer");
        assert_eq!(failure.code(), Some(XPathErrorCode::Xpdy0050));
    }

    #[test]
    fn types_render_back_to_their_syntax() {
        assert_eq!(SequenceType::Empty.to_string(), "empty-sequence()");
        assert_eq!(SequenceType::any().to_string(), "item()*");
        assert_eq!(
            SequenceType::new(ItemType::Atomic(AtomicType::Integer), Occurrence::ZeroOrOne)
                .to_string(),
            "xs:integer?"
        );
        assert_eq!(ItemType::AnyMap.to_string(), "map(*)");
        assert_eq!(
            SingleType::new(AtomicType::String, true).to_string(),
            "xs:string?"
        );
    }

    /// `Q{}*` is the wildcard for names in *no* namespace, and an element test's type argument
    /// matches by derivation rather than by equality.
    #[test]
    fn the_empty_braced_uri_wildcard_and_type_derivation() -> Result<(), Box<dyn std::error::Error>>
    {
        let tree =
            oxixml_infoset::Tree::parse_str("<r xmlns:p='urn:p'><plain/><p:qualified/></r>")?;
        let root = tree.document_element().ok_or("no document element")?;
        let children = root.children();
        let plain = children.first().ok_or("no first child")?;
        let qualified = children.get(1).ok_or("no second child")?;

        let no_namespace = NameTest::InNamespace(String::new());
        assert!(no_namespace.matches(plain));
        assert!(!no_namespace.matches(qualified));
        assert!(NameTest::InNamespace("urn:p".to_owned()).matches(qualified));

        // An unvalidated element is `xs:untyped`, which derives from `xs:anyType`.
        let any_type = KindTest::Element {
            name: None,
            type_name: Some(ExpandedName::new(Some(XSD_NAMESPACE), "anyType")),
            nillable: false,
        };
        assert!(any_type.matches(plain));
        let a_string = KindTest::Element {
            name: None,
            type_name: Some(ExpandedName::new(Some(XSD_NAMESPACE), "string")),
            nillable: false,
        };
        assert!(!a_string.matches(plain));
        Ok(())
    }
}
