//! The declared signatures of the built-in functions, and the
//! [function conversion rules](https://www.w3.org/TR/xpath-31/#dt-function-conversion) of
//! XPath 3.1 §3.1.5 applied to a call's arguments.
//!
//! F&O 3.1 declares every built-in with a `SequenceType` for each parameter, and a processor is
//! required to apply the *function conversion rules* to each supplied argument **before** the
//! function runs: the value is atomized when the parameter is atomic, `xs:untypedAtomic` is cast
//! to the required type, the numeric and `xs:anyURI` promotions are applied, and the result must
//! then match the declared type — cardinality included. An argument that does not is
//! `err:XPTY0004`, and an `xs:untypedAtomic` whose lexical form does not denote a value of the
//! required type is `err:FORG0001`.
//!
//! Doing this centrally is what keeps the individual implementations in [`crate::registry`]
//! honest: without it, `fn:abs("x")` would quietly stringify its way to a number and
//! `fn:substring-before(1, 2)` would silently succeed.
//!
//! # Examples
//!
//! ```
//! use oxixml_fn::signature::{Occurrence, ParamItemType, ParamType, convert_argument};
//! use oxixml_infoset::{AtomicType, AtomicValue, Sequence};
//!
//! let required = ParamType::new(ParamItemType::Atomic(AtomicType::String), Occurrence::ZeroOrOne);
//!
//! // `xs:untypedAtomic` is converted, not rejected.
//! let untyped = Sequence::from(AtomicValue::untyped_atomic("a"));
//! assert_eq!(convert_argument(&untyped, &required, "fn:x")?.len(), 1);
//!
//! // An `xs:integer` is not an `xs:string`, and there is no promotion that makes it one.
//! let integer = Sequence::from(AtomicValue::integer(1_i64));
//! assert!(convert_argument(&integer, &required, "fn:x").is_err());
//! # Ok::<(), oxixml_fn::FnError>(())
//! ```

mod table;

pub use table::{result_type, signature};

use crate::error::{FnError, FnErrorCode, FnResult};
use oxixml_infoset::{AtomicType, AtomicValue, Item, Sequence};

/// The item type a parameter of a built-in declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParamItemType {
    /// `item()` — anything at all, and no atomization.
    Item,
    /// `node()`.
    Node,
    /// `xs:numeric`, the union of `xs:double`, `xs:float`, `xs:decimal` and `xs:integer`.
    ///
    /// F&O 3.1 §1.6.2 gives this union a conversion rule of its own: `xs:untypedAtomic` is cast
    /// to `xs:double`, and every other numeric member is left as it is.
    Numeric,
    /// A named atomic type; a value matches when its type is derived from it.
    Atomic(AtomicType),
    /// `map(*)`.
    Map,
    /// `array(*)`.
    Array,
    /// `function(*)`.
    Function,
}

impl ParamItemType {
    /// Whether a value of this required type is reached by atomizing the argument.
    const fn atomizes(self) -> bool {
        matches!(self, Self::Numeric | Self::Atomic(_))
    }

    fn matches(self, item: &Item) -> bool {
        match self {
            Self::Item => true,
            Self::Node => item.is_node(),
            Self::Map => item.as_map().is_some(),
            Self::Array => item.as_array().is_some(),
            Self::Function => {
                item.is_function() || item.as_map().is_some() || item.as_array().is_some()
            }
            Self::Numeric => item
                .as_atomic()
                .is_some_and(|value| value.atomic_type().is_numeric()),
            Self::Atomic(required) => item
                .as_atomic()
                .is_some_and(|value| value.atomic_type().is_derived_from(required)),
        }
    }
}

/// How many items a parameter accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Occurrence {
    /// Exactly one item.
    One,
    /// `?` — zero or one.
    ZeroOrOne,
    /// `*` — any number.
    ZeroOrMore,
    /// `+` — one or more.
    OneOrMore,
}

impl Occurrence {
    const fn allows(self, count: usize) -> bool {
        match self {
            Self::One => count == 1,
            Self::ZeroOrOne => count <= 1,
            Self::ZeroOrMore => true,
            Self::OneOrMore => count >= 1,
        }
    }
}

/// One parameter's declared `SequenceType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParamType {
    item_type: ParamItemType,
    occurrence: Occurrence,
}

impl ParamType {
    /// Builds a parameter type from its item type and its occurrence indicator.
    #[must_use]
    pub const fn new(item_type: ParamItemType, occurrence: Occurrence) -> Self {
        Self {
            item_type,
            occurrence,
        }
    }

    /// The declared item type.
    #[must_use]
    pub const fn item_type(&self) -> ParamItemType {
        self.item_type
    }

    /// The declared occurrence indicator.
    #[must_use]
    pub const fn occurrence(&self) -> Occurrence {
        self.occurrence
    }
}

fn type_error(what: &str, message: &str) -> FnError {
    FnError::new(FnErrorCode::Xpty0004, format!("{what}: {message}"))
}

/// Applies the function conversion rules to one argument.
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] when the converted value does not match `required`, and
/// [`FnErrorCode::Forg0001`] when an `xs:untypedAtomic` item cannot be cast to the required
/// type.
pub fn convert_argument(value: &Sequence, required: &ParamType, what: &str) -> FnResult<Sequence> {
    let items = if required.item_type.atomizes() {
        let atomized = value.atomize()?;
        let mut converted = Vec::with_capacity(atomized.len());
        for item in atomized.iter() {
            match item.as_atomic() {
                Some(atomic) => {
                    converted.push(Item::from(promote(atomic, required.item_type, what)?))
                }
                None => converted.push(item.clone()),
            }
        }
        Sequence::from_items(converted)
    } else {
        value.clone()
    };
    if !required.occurrence.allows(items.len()) {
        return Err(type_error(
            what,
            &format!(
                "the argument holds {} item(s), which the declared type does not allow",
                items.len()
            ),
        ));
    }
    for item in items.iter() {
        if !required.item_type.matches(item) {
            return Err(type_error(
                what,
                &format!(
                    "the argument type {} is not the declared type",
                    describe(item)
                ),
            ));
        }
    }
    Ok(items)
}

fn describe(item: &Item) -> String {
    match item {
        Item::Atomic(value) => value.atomic_type().to_string(),
        Item::Node(_) => "node()".to_owned(),
        _ => "a function item".to_owned(),
    }
}

/// The promotion half of the conversion rules, applied to one atomic value.
fn promote(value: &AtomicValue, required: ParamItemType, what: &str) -> FnResult<AtomicValue> {
    let source = value.atomic_type();
    let target = match required {
        ParamItemType::Numeric => {
            if source.is_numeric() {
                return Ok(value.clone());
            }
            AtomicType::Double
        }
        ParamItemType::Atomic(target) => {
            if source.is_derived_from(target) {
                return Ok(value.clone());
            }
            target
        }
        _ => return Ok(value.clone()),
    };
    if source == AtomicType::UntypedAtomic {
        // `xs:untypedAtomic` is *cast* to the required type. Casting to `xs:QName` or
        // `xs:NOTATION` is not allowed (XPath 3.1 §19.1), so those stay a type error.
        if target.is_derived_from(AtomicType::QName) || target.is_derived_from(AtomicType::Notation)
        {
            // XPath 3.1 §3.1.5.2 gives the namespace-sensitive types their own code: "if the
            // expected type is a namespace-sensitive type, and the supplied value is of type
            // xs:untypedAtomic, a type error is raised [err:XPTY0117]".
            return Err(FnError::new(
                FnErrorCode::Xpty0117,
                format!("{what}: xs:untypedAtomic cannot be converted to {target}"),
            ));
        }
        return AtomicValue::parse(target, &value.string_value()).map_err(FnError::from);
    }
    // Numeric promotion, and the `xs:anyURI` to `xs:string` promotion.
    if target == AtomicType::Double && source.is_numeric() {
        return Ok(AtomicValue::double(value.as_f64().unwrap_or(f64::NAN)));
    }
    if target == AtomicType::Float && source.is_derived_from(AtomicType::Decimal) {
        #[allow(clippy::cast_possible_truncation)]
        return Ok(AtomicValue::float(value.as_f64().unwrap_or(f64::NAN) as f32));
    }
    if target.is_derived_from(AtomicType::String) && source == AtomicType::AnyUri {
        return Ok(AtomicValue::string(value.string_value()));
    }
    Err(type_error(
        what,
        &format!("an {source} cannot be converted to an {target}"),
    ))
}

/// Applies the function conversion rules to every argument of a call, when the function's
/// signature is known. Arguments of a function with no declared signature pass through.
///
/// # Errors
///
/// Whatever [`convert_argument`] raises.
pub fn convert_arguments(
    namespace: &str,
    local: &str,
    arguments: &[Sequence],
) -> FnResult<Option<Vec<Sequence>>> {
    let Some(parameters) = signature(namespace, local, arguments.len()) else {
        return Ok(None);
    };
    let what = format!("{local}#{}", arguments.len());
    let mut converted = Vec::with_capacity(arguments.len());
    for (index, argument) in arguments.iter().enumerate() {
        match parameters.get(index) {
            Some(required) => converted.push(convert_argument(argument, required, &what)?),
            None => converted.push(argument.clone()),
        }
    }
    Ok(Some(converted))
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::Tree;

    fn string_q() -> ParamType {
        ParamType::new(
            ParamItemType::Atomic(AtomicType::String),
            Occurrence::ZeroOrOne,
        )
    }

    #[test]
    fn an_integer_is_not_a_string() {
        let value = Sequence::from(AtomicValue::integer(1_i64));
        let error = convert_argument(&value, &string_q(), "fn:x").expect_err("a type error");
        assert_eq!(error.code(), Some(FnErrorCode::Xpty0004));
    }

    #[test]
    fn an_any_uri_promotes_to_a_string() -> FnResult<()> {
        let value = Sequence::from(AtomicValue::any_uri("urn:x"));
        let converted = convert_argument(&value, &string_q(), "fn:x")?;
        assert_eq!(
            converted
                .first()
                .and_then(Item::as_atomic)
                .map(AtomicValue::atomic_type),
            Some(AtomicType::String)
        );
        Ok(())
    }

    #[test]
    fn a_node_atomizes_and_casts() -> FnResult<()> {
        let tree = Tree::parse_str("<r>42</r>")?;
        let element = tree.document_element().expect("a document element");
        let value = Sequence::from(element);
        let numeric = ParamType::new(ParamItemType::Numeric, Occurrence::ZeroOrOne);
        let converted = convert_argument(&value, &numeric, "fn:x")?;
        assert_eq!(
            converted
                .first()
                .and_then(Item::as_atomic)
                .map(AtomicValue::atomic_type),
            Some(AtomicType::Double)
        );
        Ok(())
    }

    #[test]
    fn cardinality_is_enforced() {
        let value = Sequence::from_iter([AtomicValue::string("a"), AtomicValue::string("b")]);
        let error = convert_argument(&value, &string_q(), "fn:x").expect_err("a type error");
        assert_eq!(error.code(), Some(FnErrorCode::Xpty0004));
    }

    #[test]
    fn an_uncastable_untyped_atomic_is_forg0001() {
        let value = Sequence::from(AtomicValue::untyped_atomic("nope"));
        let numeric = ParamType::new(ParamItemType::Numeric, Occurrence::ZeroOrOne);
        let error = convert_argument(&value, &numeric, "fn:x").expect_err("a cast error");
        assert_eq!(error.code(), Some(FnErrorCode::Forg0001));
    }
}
