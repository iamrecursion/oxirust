//! The built-in function catalogue: a name-and-arity index of the library, which is what
//! [`fn:function-lookup`](https://www.w3.org/TR/xpath-functions-31/#func-function-lookup) reads
//! and what an evaluator binds a function call to.
//!
//! # What is in the catalogue
//!
//! Every function whose arguments and result are *values* — that is, [`Sequence`]s — is
//! registered here. The higher-order functions of §16 and the `map:`/`array:` functions that
//! take a callback are deliberately **not**: their arguments include a general function item,
//! which [`Item`] cannot hold, so an evaluator calls
//! [`crate::higher_order`], [`crate::maps::for_each`] and [`crate::arrays::for_each`] directly,
//! passing its own closure as a [`FunctionItem`].
//!
//! # Examples
//!
//! ```
//! use oxixml_fn::{registry::{builtins, FN_NAMESPACE}, FnContext};
//! use oxixml_infoset::{AtomicValue, ExpandedName, Sequence};
//!
//! let registry = builtins();
//! let context = FnContext::new();
//!
//! let abs = registry
//!     .lookup(&ExpandedName::new(Some(FN_NAMESPACE), "abs"), 1)
//!     .expect("fn:abs#1 is a built-in");
//! let result = abs.call(&context, &[Sequence::from(AtomicValue::integer(-3_i64))])?;
//! assert_eq!(result.first().map(ToString::to_string).as_deref(), Some("3"));
//!
//! // A name-and-arity pair that is not in the catalogue simply is not found.
//! assert!(registry.lookup(&ExpandedName::new(Some(FN_NAMESPACE), "abs"), 2).is_none());
//! # Ok::<(), oxixml_fn::FnError>(())
//! ```

use crate::arrays;
use crate::context::FnContext;
use crate::datetime;
use crate::error::{FnError, FnErrorCode, FnResult};
use crate::function::FunctionItem;
use crate::json;
use crate::maps;
use crate::nodes;
use crate::numeric;
use crate::qnames;
use crate::sequences;
use crate::strings;
use crate::xmlio;
use oxixml_infoset::{
    AtomicData, AtomicValue, DuplicateKeys, ExpandedName, Item, NodeRef, QNameValue, Sequence,
    XdmArray, XdmMap,
};
use std::collections::HashMap;

/// The namespace name of the `fn:` functions.
pub const FN_NAMESPACE: &str = "http://www.w3.org/2005/xpath-functions";

/// A catalogue of built-in functions, keyed on expanded name and arity.
#[derive(Debug, Clone, Default)]
pub struct FunctionRegistry {
    entries: HashMap<(ExpandedName, usize), FunctionItem>,
}

impl FunctionRegistry {
    /// An empty catalogue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a function under `namespace`, `local` and `arity`.
    ///
    /// When [`crate::signature`] declares the function's parameter types, the arguments of every
    /// call are put through the *function conversion rules* of XPath 3.1 §3.1.5 before
    /// `implementation` sees them, so that an argument of the wrong type is `XPTY0004` rather
    /// than something the implementation improvises. A name the table does not know is
    /// registered exactly as given.
    pub fn add(
        &mut self,
        namespace: &str,
        local: &str,
        arity: usize,
        implementation: impl Fn(&FnContext, &[Sequence]) -> FnResult<Sequence> + Send + Sync + 'static,
    ) {
        let name = ExpandedName::new(Some(namespace), local);
        let function = match crate::signature::signature(namespace, local, arity) {
            Some(parameters) => {
                let what = format!("{local}#{arity}");
                FunctionItem::new(name.clone(), arity, move |context, arguments| {
                    let mut converted = Vec::with_capacity(arguments.len());
                    for (index, argument) in arguments.iter().enumerate() {
                        converted.push(match parameters.get(index) {
                            Some(required) => {
                                crate::signature::convert_argument(argument, required, &what)?
                            }
                            None => argument.clone(),
                        });
                    }
                    implementation(context, &converted)
                })
            }
            None => FunctionItem::new(name.clone(), arity, implementation),
        };
        self.entries.insert((name, arity), function);
    }

    /// [`fn:function-lookup`](https://www.w3.org/TR/xpath-functions-31/#func-function-lookup).
    #[must_use]
    pub fn lookup(&self, name: &ExpandedName, arity: usize) -> Option<FunctionItem> {
        self.entries.get(&(name.clone(), arity)).cloned()
    }

    /// Whether the catalogue holds a function of this name and arity.
    #[must_use]
    pub fn contains(&self, name: &ExpandedName, arity: usize) -> bool {
        self.entries.contains_key(&(name.clone(), arity))
    }

    /// The number of registered `(name, arity)` pairs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the catalogue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Every registered name and arity, in an unspecified order.
    pub fn names(&self) -> impl Iterator<Item = (&ExpandedName, usize)> {
        self.entries.keys().map(|(name, arity)| (name, *arity))
    }
}

// --- argument helpers -------------------------------------------------------------------

fn argument(arguments: &[Sequence], index: usize) -> Sequence {
    arguments.get(index).cloned().unwrap_or_default()
}

fn type_error(what: &str) -> FnError {
    FnError::new(FnErrorCode::Xpty0004, what.to_owned())
}

/// An `xs:string?` argument: the empty sequence becomes the zero-length string.
fn text(arguments: &[Sequence], index: usize) -> FnResult<String> {
    string_argument(&argument(arguments, index))
}

/// An `xs:string?` argument where the empty sequence means *absent*.
fn optional_text(arguments: &[Sequence], index: usize) -> FnResult<Option<String>> {
    let sequence = argument(arguments, index);
    if sequence.is_empty() {
        return Ok(None);
    }
    Ok(Some(string_argument(&sequence)?))
}

/// Applies the function conversion rules of XPath 3.1 §3.1.5 to a parameter declared
/// `xs:string?`.
///
/// Only three atomic types reach an `xs:string` parameter: `xs:string` and its restrictions,
/// `xs:untypedAtomic` (which is *converted*, not rejected), and `xs:anyURI` (which the rules
/// promote). Anything else — an `xs:integer`, an `xs:date`, an `xs:QName` — is `XPTY0004`;
/// stringifying it would silently accept a query the specification rejects.
fn string_argument(sequence: &Sequence) -> FnResult<String> {
    match sequence.as_slice() {
        [] => Ok(String::new()),
        // A node is atomized first, and in a processor with no schema every node atomizes to
        // `xs:untypedAtomic`, which converts to `xs:string`.
        [Item::Node(node)] => Ok(oxixml_infoset::XdmNode::string_value(node)),
        [Item::Atomic(value)] if value.atomic_type().is_text() => Ok(value.string_value()),
        [Item::Atomic(value)] => Err(type_error(&format!(
            "an xs:string argument cannot be an {}",
            value.atomic_type()
        ))),
        [item] => Ok(item.string_value()?),
        _ => Err(type_error(
            "an xs:string? argument must be at most one item",
        )),
    }
}

/// Reads `fn:format-number`'s `$decimal-format-name` argument.
///
/// F&O 3.1 §4.7.3 says the argument "is interpreted as an `EQName`", after leading and trailing
/// whitespace has been stripped, so ` Q{http://a.ns/}test ` and `Q{http://a.ns/}test` name the
/// same decimal format. A `prefix:local` spelling cannot be expanded here — the function library
/// has no namespace bindings — so it is passed on as written and matched against the spellings
/// the host language registered.
fn decimal_format_name(lexical: &str) -> ExpandedName {
    let trimmed = lexical.trim();
    if let Some(rest) = trimmed.strip_prefix("Q{") {
        if let Some((namespace, local)) = rest.split_once('}') {
            return ExpandedName::new(
                (!namespace.is_empty()).then_some(namespace),
                local.trim_start(),
            );
        }
    }
    ExpandedName::new(None::<&str>, trimmed)
}

fn atomic(arguments: &[Sequence], index: usize) -> FnResult<Option<AtomicValue>> {
    let atomized = argument(arguments, index).atomize()?;
    match atomized.as_slice() {
        [] => Ok(None),
        [Item::Atomic(value)] => Ok(Some(value.clone())),
        _ => Err(type_error("an argument must be at most one atomic value")),
    }
}

fn required_atomic(arguments: &[Sequence], index: usize, what: &str) -> FnResult<AtomicValue> {
    atomic(arguments, index)?.ok_or_else(|| type_error(&format!("{what} requires a value")))
}

fn node(arguments: &[Sequence], index: usize) -> FnResult<Option<NodeRef>> {
    match argument(arguments, index).as_slice() {
        [] => Ok(None),
        [Item::Node(node)] => Ok(Some(node.clone())),
        _ => Err(type_error("an argument must be at most one node")),
    }
}

fn required_node(arguments: &[Sequence], index: usize, what: &str) -> FnResult<NodeRef> {
    node(arguments, index)?.ok_or_else(|| type_error(&format!("{what} requires a node")))
}

fn number(arguments: &[Sequence], index: usize) -> FnResult<Option<f64>> {
    Ok(atomic(arguments, index)?.map(|value| numeric::cast_to_double(&value)))
}

fn required_number(arguments: &[Sequence], index: usize, what: &str) -> FnResult<f64> {
    number(arguments, index)?.ok_or_else(|| type_error(&format!("{what} requires a number")))
}

fn integer(arguments: &[Sequence], index: usize) -> FnResult<Option<i64>> {
    let Some(value) = atomic(arguments, index)? else {
        return Ok(None);
    };
    Ok(Some(exact_integer(&value)?))
}

/// Reads an `xs:integer` argument *exactly*.
///
/// Routing an `xs:integer` through `f64` would silently corrupt every magnitude above 2^53, so
/// the integer and decimal value spaces are read from their own representations and only the
/// genuinely floating-point (and lexical) forms go through a double.
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] when the value is not an integer at all, or
/// [`FnErrorCode::Foca0003`] when it is outside the range of the internal `xs:integer`.
fn exact_integer(value: &AtomicValue) -> FnResult<i64> {
    match value.data() {
        AtomicData::Integer(integer) => Ok(i64::from(*integer)),
        AtomicData::Decimal(decimal) => oxixml_xsd::Integer::try_from(*decimal)
            .map(i64::from)
            .map_err(|_| integer_overflow()),
        _ => {
            let number = numeric::cast_to_double(value);
            if number.is_nan() {
                return Err(type_error("an xs:integer argument is required"));
            }
            let truncated = number.trunc();
            // 2^63 is exactly representable, so this brackets the i64 range without rounding.
            if !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&truncated) {
                return Err(integer_overflow());
            }
            Ok(truncated as i64)
        }
    }
}

/// Narrows a `$precision` argument of `fn:round`/`fn:round-half-to-even` to the `i32` the
/// implementation works in, *saturating* rather than wrapping.
///
/// F&O 3.1 §4.4.4 puts no bound on `$precision`, and the two rounding functions are total in
/// it: a precision beyond the width of any representable value simply returns the value
/// unchanged, and one below it returns zero. Truncating `4294967296` to an `i32` would make it
/// `0` and round to the nearest integer instead — the exact failure QT3's
/// `cbcl-round-half-to-even-001` catches.
const fn saturating_precision(precision: i64) -> i32 {
    if precision > i32::MAX as i64 {
        i32::MAX
    } else if precision < i32::MIN as i64 {
        i32::MIN
    } else {
        precision as i32
    }
}

fn integer_overflow() -> FnError {
    FnError::new(
        FnErrorCode::Foca0003,
        "an xs:integer argument is outside the range this processor represents",
    )
}

fn collation(arguments: &[Sequence], index: usize) -> FnResult<Option<String>> {
    optional_text(arguments, index)
}

fn qname(arguments: &[Sequence], index: usize, what: &str) -> FnResult<QNameValue> {
    let value = required_atomic(arguments, index, what)?;
    qnames::as_qname(&value, what)
}

fn map_argument(arguments: &[Sequence], index: usize, what: &str) -> FnResult<XdmMap> {
    maps::require_map(&argument(arguments, index), what)
}

fn array_argument(arguments: &[Sequence], index: usize, what: &str) -> FnResult<XdmArray> {
    arrays::require_array(&argument(arguments, index), what)
}

fn strings_of(sequence: &Sequence) -> FnResult<Vec<String>> {
    let atomized = sequence.atomize()?;
    let mut out = Vec::with_capacity(atomized.len());
    for item in atomized.iter() {
        out.push(item.string_value()?);
    }
    Ok(out)
}

// --- result helpers ---------------------------------------------------------------------

fn as_string(value: String) -> Sequence {
    Sequence::from(AtomicValue::string(value))
}

fn as_boolean(value: bool) -> Sequence {
    Sequence::from(AtomicValue::boolean(value))
}

fn as_integer(value: i64) -> Sequence {
    Sequence::from(AtomicValue::integer(value))
}

fn as_double(value: f64) -> Sequence {
    Sequence::from(numeric::double_value(value))
}

fn as_optional(value: Option<AtomicValue>) -> Sequence {
    value.map(Sequence::from).unwrap_or_default()
}

fn as_nodes(nodes: Vec<NodeRef>) -> Sequence {
    Sequence::from_items(nodes.into_iter().map(Item::Node))
}

/// The catalogue of every value-level built-in this crate implements.
#[must_use]
pub fn builtins() -> FunctionRegistry {
    let mut registry = FunctionRegistry::new();
    register_accessors(&mut registry);
    register_numeric(&mut registry);
    register_strings(&mut registry);
    register_boolean(&mut registry);
    register_datetime(&mut registry);
    register_qnames(&mut registry);
    register_nodes(&mut registry);
    register_sequences(&mut registry);
    register_maps(&mut registry);
    register_arrays(&mut registry);
    register_xmlio(&mut registry);
    register_external_processors(&mut registry);
    registry
}

fn register_accessors(registry: &mut FunctionRegistry) {
    registry.add(FN_NAMESPACE, "node-name", 1, |_, arguments| {
        Ok(match node(arguments, 0)? {
            Some(node) => as_optional(crate::accessors::node_name(&node).map(AtomicValue::qname)),
            None => Sequence::empty(),
        })
    });
    registry.add(FN_NAMESPACE, "nilled", 1, |_, arguments| {
        Ok(match node(arguments, 0)? {
            Some(node) => as_optional(crate::accessors::nilled(&node).map(AtomicValue::boolean)),
            None => Sequence::empty(),
        })
    });
    registry.add(FN_NAMESPACE, "string", 1, |_, arguments| {
        Ok(as_string(crate::accessors::string_of(&argument(
            arguments, 0,
        ))?))
    });
    registry.add(FN_NAMESPACE, "data", 1, |_, arguments| {
        crate::accessors::data(&argument(arguments, 0))
    });
    registry.add(FN_NAMESPACE, "base-uri", 1, |_, arguments| {
        Ok(match node(arguments, 0)? {
            Some(node) => as_optional(crate::accessors::base_uri(&node)),
            None => Sequence::empty(),
        })
    });
    registry.add(FN_NAMESPACE, "document-uri", 1, |_, arguments| {
        Ok(match node(arguments, 0)? {
            Some(node) => as_optional(crate::accessors::document_uri(&node)),
            None => Sequence::empty(),
        })
    });
    registry.add(FN_NAMESPACE, "trace", 1, |context, arguments| {
        Ok(crate::diagnostics::trace_unlabelled(
            context,
            &argument(arguments, 0),
        ))
    });
    registry.add(FN_NAMESPACE, "trace", 2, |context, arguments| {
        let label = text(arguments, 1)?;
        Ok(crate::diagnostics::trace(
            context,
            &argument(arguments, 0),
            &label,
        ))
    });
    for arity in 0..=3 {
        registry.add(FN_NAMESPACE, "error", arity, move |_, arguments| {
            let name = match arity {
                0 => None,
                _ => atomic(arguments, 0)?.and_then(|value| value.as_qname().cloned()),
            };
            let description = if arity >= 2 {
                optional_text(arguments, 1)?
            } else {
                None
            };
            let object = if arity >= 3 {
                Some(argument(arguments, 2))
            } else {
                None
            };
            Err(crate::diagnostics::error(
                name.as_ref(),
                description.as_deref(),
                object,
            ))
        });
    }
}

fn register_numeric(registry: &mut FunctionRegistry) {
    macro_rules! unary_numeric {
        ($local:literal, $function:path) => {
            registry.add(FN_NAMESPACE, $local, 1, |_, arguments| {
                Ok(match atomic(arguments, 0)? {
                    Some(value) => Sequence::from($function(&value)?),
                    None => Sequence::empty(),
                })
            });
        };
    }
    unary_numeric!("abs", numeric::abs);
    unary_numeric!("ceiling", numeric::ceiling);
    unary_numeric!("floor", numeric::floor);

    registry.add(FN_NAMESPACE, "round", 1, |_, arguments| {
        Ok(match atomic(arguments, 0)? {
            Some(value) => Sequence::from(numeric::round(&value, 0)?),
            None => Sequence::empty(),
        })
    });
    registry.add(FN_NAMESPACE, "round", 2, |_, arguments| {
        let precision = integer(arguments, 1)?.unwrap_or(0);
        Ok(match atomic(arguments, 0)? {
            Some(value) => Sequence::from(numeric::round(&value, saturating_precision(precision))?),
            None => Sequence::empty(),
        })
    });
    registry.add(FN_NAMESPACE, "round-half-to-even", 1, |_, arguments| {
        Ok(match atomic(arguments, 0)? {
            Some(value) => Sequence::from(numeric::round_half_to_even(&value, 0)?),
            None => Sequence::empty(),
        })
    });
    registry.add(FN_NAMESPACE, "round-half-to-even", 2, |_, arguments| {
        let precision = integer(arguments, 1)?.unwrap_or(0);
        Ok(match atomic(arguments, 0)? {
            Some(value) => Sequence::from(numeric::round_half_to_even(
                &value,
                saturating_precision(precision),
            )?),
            None => Sequence::empty(),
        })
    });
    registry.add(FN_NAMESPACE, "number", 1, |_, arguments| {
        Ok(as_double(numeric::number_of(&argument(arguments, 0))?))
    });
    registry.add(FN_NAMESPACE, "parse-ietf-date", 1, |_, arguments| {
        let Some(text) = optional_text(arguments, 0)? else {
            return Ok(Sequence::empty());
        };
        Ok(Sequence::from(datetime::parse_ietf_date(&text)?))
    });
    // `fn:format-integer` takes an optional `$language`. F&O 3.1 §4.6.1 says a processor that
    // does not support the language named must fall back on a language it does support rather
    // than raise an error, and English is the one this crate implements, so the argument is
    // validated for its type and then does not change the numbering.
    for arity in 2..=3 {
        registry.add(FN_NAMESPACE, "format-integer", arity, |_, arguments| {
            let picture = text(arguments, 1)?;
            // §4.6.1: an empty `$value` is the zero-length string, not a formatted zero.
            let Some(value) = integer(arguments, 0)? else {
                return Ok(as_string(String::new()));
            };
            Ok(as_string(datetime::format_integer(value, &picture)?))
        });
    }
    registry.add(FN_NAMESPACE, "format-number", 2, |context, arguments| {
        let picture = text(arguments, 1)?;
        let value = atomic(arguments, 0)?;
        Ok(as_string(numeric::format_number(
            value.as_ref(),
            &picture,
            context.default_decimal_format(),
        )?))
    });
    registry.add(FN_NAMESPACE, "format-number", 3, |context, arguments| {
        let picture = text(arguments, 1)?;
        let value = atomic(arguments, 0)?;
        let name = optional_text(arguments, 2)?.map(|name| decimal_format_name(&name));
        let format = context.decimal_format(name.as_ref())?.clone();
        Ok(as_string(numeric::format_number(
            value.as_ref(),
            &picture,
            &format,
        )?))
    });
}

fn register_strings(registry: &mut FunctionRegistry) {
    registry.add(
        FN_NAMESPACE,
        "codepoints-to-string",
        1,
        |context, arguments| {
            let codepoints: Vec<i64> = argument(arguments, 0)
                .atomize()?
                .iter()
                .filter_map(|item| item.as_atomic().and_then(AtomicValue::as_f64))
                .map(|value| value.trunc().clamp(-9.0e18, 9.0e18) as i64)
                .collect();
            // Which code points are characters at all is the processor's XML version, not the
            // call's; the context carries it and defaults to XML 1.0.
            Ok(as_string(strings::codepoints_to_string_in(
                &codepoints,
                context.xml_version(),
            )?))
        },
    );
    registry.add(FN_NAMESPACE, "string-to-codepoints", 1, |_, arguments| {
        let value = text(arguments, 0)?;
        Ok(Sequence::from_iter(
            strings::string_to_codepoints(&value)
                .into_iter()
                .map(AtomicValue::integer),
        ))
    });
    for arity in 2..=3 {
        registry.add(FN_NAMESPACE, "compare", arity, move |context, arguments| {
            let (left, right) = (optional_text(arguments, 0)?, optional_text(arguments, 1)?);
            let (Some(left), Some(right)) = (left, right) else {
                return Ok(Sequence::empty());
            };
            let collation = if arity == 3 {
                collation(arguments, 2)?
            } else {
                None
            };
            Ok(as_integer(strings::compare(
                context,
                &left,
                &right,
                collation.as_deref(),
            )?))
        });
    }
    registry.add(FN_NAMESPACE, "codepoint-equal", 2, |_, arguments| {
        let (left, right) = (optional_text(arguments, 0)?, optional_text(arguments, 1)?);
        Ok(match (left, right) {
            (Some(left), Some(right)) => as_boolean(strings::codepoint_equal(&left, &right)),
            _ => Sequence::empty(),
        })
    });
    for arity in 2..=10 {
        registry.add(FN_NAMESPACE, "concat", arity, |_, arguments| {
            Ok(as_string(strings::concat(arguments)?))
        });
    }
    registry.add(FN_NAMESPACE, "string-join", 1, |_, arguments| {
        Ok(as_string(strings::string_join(
            &argument(arguments, 0),
            "",
        )?))
    });
    registry.add(FN_NAMESPACE, "string-join", 2, |_, arguments| {
        let separator = text(arguments, 1)?;
        Ok(as_string(strings::string_join(
            &argument(arguments, 0),
            &separator,
        )?))
    });
    registry.add(FN_NAMESPACE, "substring", 2, |_, arguments| {
        let value = text(arguments, 0)?;
        let start = required_number(arguments, 1, "fn:substring")?;
        Ok(as_string(strings::substring(&value, start, None)))
    });
    registry.add(FN_NAMESPACE, "substring", 3, |_, arguments| {
        let value = text(arguments, 0)?;
        let start = required_number(arguments, 1, "fn:substring")?;
        let length = required_number(arguments, 2, "fn:substring")?;
        Ok(as_string(strings::substring(&value, start, Some(length))))
    });
    registry.add(FN_NAMESPACE, "string-length", 1, |_, arguments| {
        let value = text(arguments, 0)?;
        Ok(as_integer(
            i64::try_from(strings::string_length(&value)).unwrap_or(i64::MAX),
        ))
    });
    registry.add(FN_NAMESPACE, "normalize-space", 1, |_, arguments| {
        let value = text(arguments, 0)?;
        Ok(as_string(strings::normalize_space(&value)))
    });
    registry.add(FN_NAMESPACE, "normalize-unicode", 1, |_, arguments| {
        let value = text(arguments, 0)?;
        Ok(as_string(strings::normalize_unicode(
            &value,
            strings::NormalizationForm::Nfc,
        )))
    });
    registry.add(FN_NAMESPACE, "normalize-unicode", 2, |_, arguments| {
        let value = text(arguments, 0)?;
        let form = strings::NormalizationForm::parse(&text(arguments, 1)?)?;
        Ok(as_string(strings::normalize_unicode(&value, form)))
    });
    registry.add(FN_NAMESPACE, "upper-case", 1, |_, arguments| {
        Ok(as_string(strings::upper_case(&text(arguments, 0)?)))
    });
    registry.add(FN_NAMESPACE, "lower-case", 1, |_, arguments| {
        Ok(as_string(strings::lower_case(&text(arguments, 0)?)))
    });
    registry.add(FN_NAMESPACE, "translate", 3, |_, arguments| {
        Ok(as_string(strings::translate(
            &text(arguments, 0)?,
            &text(arguments, 1)?,
            &text(arguments, 2)?,
        )))
    });

    macro_rules! substring_predicate {
        ($local:literal, $function:path) => {
            for arity in 2..=3 {
                registry.add(FN_NAMESPACE, $local, arity, move |context, arguments| {
                    let haystack = text(arguments, 0)?;
                    let needle = text(arguments, 1)?;
                    let collation = if arity == 3 {
                        collation(arguments, 2)?
                    } else {
                        None
                    };
                    Ok(as_boolean($function(
                        context,
                        &haystack,
                        &needle,
                        collation.as_deref(),
                    )?))
                });
            }
        };
    }
    substring_predicate!("contains", strings::contains);
    substring_predicate!("starts-with", strings::starts_with);
    substring_predicate!("ends-with", strings::ends_with);

    macro_rules! substring_part {
        ($local:literal, $function:path) => {
            for arity in 2..=3 {
                registry.add(FN_NAMESPACE, $local, arity, move |context, arguments| {
                    let haystack = text(arguments, 0)?;
                    let needle = text(arguments, 1)?;
                    let collation = if arity == 3 {
                        collation(arguments, 2)?
                    } else {
                        None
                    };
                    Ok(as_string($function(
                        context,
                        &haystack,
                        &needle,
                        collation.as_deref(),
                    )?))
                });
            }
        };
    }
    substring_part!("substring-before", strings::substring_before);
    substring_part!("substring-after", strings::substring_after);

    for arity in 2..=3 {
        registry.add(FN_NAMESPACE, "matches", arity, move |_, arguments| {
            let flags = if arity == 3 {
                text(arguments, 2)?
            } else {
                String::new()
            };
            Ok(as_boolean(strings::matches(
                &text(arguments, 0)?,
                &text(arguments, 1)?,
                &flags,
            )?))
        });
    }
    for arity in 3..=4 {
        registry.add(FN_NAMESPACE, "replace", arity, move |_, arguments| {
            let flags = if arity == 4 {
                text(arguments, 3)?
            } else {
                String::new()
            };
            Ok(as_string(strings::replace(
                &text(arguments, 0)?,
                &text(arguments, 1)?,
                &text(arguments, 2)?,
                &flags,
            )?))
        });
    }
    registry.add(FN_NAMESPACE, "tokenize", 1, |_, arguments| {
        let tokens = strings::tokenize_whitespace(&text(arguments, 0)?)?;
        Ok(Sequence::from_iter(
            tokens.into_iter().map(AtomicValue::string),
        ))
    });
    for arity in 2..=3 {
        registry.add(FN_NAMESPACE, "tokenize", arity, move |_, arguments| {
            let flags = if arity == 3 {
                text(arguments, 2)?
            } else {
                String::new()
            };
            let tokens = strings::tokenize(&text(arguments, 0)?, &text(arguments, 1)?, &flags)?;
            Ok(Sequence::from_iter(
                tokens.into_iter().map(AtomicValue::string),
            ))
        });
    }
    for arity in 2..=3 {
        registry.add(
            FN_NAMESPACE,
            "analyze-string",
            arity,
            move |_, arguments| {
                let flags = if arity == 3 {
                    text(arguments, 2)?
                } else {
                    String::new()
                };
                // §5.6.5: an empty sequence for `$input` is treated as the zero-length string.
                let input = optional_text(arguments, 0)?.unwrap_or_default();
                let node = strings::analyze_string(&input, &text(arguments, 1)?, &flags)?;
                Ok(Sequence::from(node))
            },
        );
    }
    registry.add(FN_NAMESPACE, "resolve-uri", 1, |context, arguments| {
        let Some(relative) = optional_text(arguments, 0)? else {
            return Ok(Sequence::empty());
        };
        Ok(Sequence::from(AtomicValue::any_uri(strings::resolve_uri(
            context, &relative,
        )?)))
    });
    registry.add(FN_NAMESPACE, "resolve-uri", 2, |_, arguments| {
        let Some(relative) = optional_text(arguments, 0)? else {
            return Ok(Sequence::empty());
        };
        let base = text(arguments, 1)?;
        Ok(Sequence::from(AtomicValue::any_uri(
            strings::resolve_uri_against(&relative, &base)?,
        )))
    });
    registry.add(FN_NAMESPACE, "encode-for-uri", 1, |_, arguments| {
        Ok(as_string(strings::encode_for_uri(&text(arguments, 0)?)))
    });
    registry.add(FN_NAMESPACE, "iri-to-uri", 1, |_, arguments| {
        Ok(as_string(strings::iri_to_uri(&text(arguments, 0)?)))
    });
    registry.add(FN_NAMESPACE, "escape-html-uri", 1, |_, arguments| {
        Ok(as_string(strings::escape_html_uri(&text(arguments, 0)?)))
    });
    for arity in 2..=3 {
        registry.add(
            FN_NAMESPACE,
            "contains-token",
            arity,
            move |context, arguments| {
                let input = strings_of(&argument(arguments, 0))?;
                let token = text(arguments, 1)?;
                let collation = if arity == 3 {
                    collation(arguments, 2)?
                } else {
                    None
                };
                Ok(as_boolean(strings::contains_token(
                    context,
                    &input,
                    &token,
                    collation.as_deref(),
                )?))
            },
        );
    }
    for arity in 1..=2 {
        registry.add(
            FN_NAMESPACE,
            "collation-key",
            arity,
            move |context, arguments| {
                let key = text(arguments, 0)?;
                let collation = if arity == 2 {
                    collation(arguments, 1)?
                } else {
                    None
                };
                let bytes = strings::collation_key(context, &key, collation.as_deref())?;
                Ok(Sequence::from(AtomicValue::base64_binary(bytes)))
            },
        );
    }
    // `fn:default-language` reports the language the context supplies for `fn:format-integer`
    // and the `fn:format-*` date functions. English is the one this crate's numbering and
    // calendar data implement, and F&O 3.1 §13.3 leaves the value implementation-defined.
    registry.add(FN_NAMESPACE, "default-language", 0, |_, _| {
        Ok(Sequence::from(AtomicValue::parse(
            oxixml_infoset::AtomicType::Language,
            "en",
        )?))
    });
    registry.add(FN_NAMESPACE, "default-collation", 0, |context, _| {
        Ok(as_string(strings::default_collation(context)))
    });
    registry.add(FN_NAMESPACE, "static-base-uri", 0, |context, _| {
        Ok(match strings::static_base_uri(context) {
            Some(uri) => Sequence::from(AtomicValue::any_uri(uri)),
            None => Sequence::empty(),
        })
    });
}

fn register_boolean(registry: &mut FunctionRegistry) {
    registry.add(FN_NAMESPACE, "true", 0, |_, _| {
        Ok(Sequence::from(crate::boolean::r#true()))
    });
    registry.add(FN_NAMESPACE, "false", 0, |_, _| {
        Ok(Sequence::from(crate::boolean::r#false()))
    });
    registry.add(FN_NAMESPACE, "boolean", 1, |_, arguments| {
        Ok(as_boolean(crate::boolean::boolean(&argument(
            arguments, 0,
        ))?))
    });
    registry.add(FN_NAMESPACE, "not", 1, |_, arguments| {
        Ok(as_boolean(crate::boolean::not(&argument(arguments, 0))?))
    });
}

fn register_datetime(registry: &mut FunctionRegistry) {
    macro_rules! component {
        ($local:literal, $function:path) => {
            registry.add(FN_NAMESPACE, $local, 1, |_, arguments| {
                Ok(match atomic(arguments, 0)? {
                    Some(value) => as_integer($function(&value)?),
                    None => Sequence::empty(),
                })
            });
        };
    }
    component!("years-from-duration", datetime::years_from_duration);
    component!("months-from-duration", datetime::months_from_duration);
    component!("days-from-duration", datetime::days_from_duration);
    component!("hours-from-duration", datetime::hours_from_duration);
    component!("minutes-from-duration", datetime::minutes_from_duration);
    component!("year-from-dateTime", datetime::year_from_date_time);
    component!("month-from-dateTime", datetime::month_from_date_time);
    component!("day-from-dateTime", datetime::day_from_date_time);
    component!("hours-from-dateTime", datetime::hours_from_date_time);
    component!("minutes-from-dateTime", datetime::minutes_from_date_time);
    component!("year-from-date", datetime::year_from_date);
    component!("month-from-date", datetime::month_from_date);
    component!("day-from-date", datetime::day_from_date);
    component!("hours-from-time", datetime::hours_from_time);
    component!("minutes-from-time", datetime::minutes_from_time);

    macro_rules! seconds {
        ($local:literal, $function:path) => {
            registry.add(FN_NAMESPACE, $local, 1, |_, arguments| {
                Ok(match atomic(arguments, 0)? {
                    Some(value) => Sequence::from(datetime::decimal_value($function(&value)?)),
                    None => Sequence::empty(),
                })
            });
        };
    }
    seconds!("seconds-from-duration", datetime::seconds_from_duration);
    seconds!("seconds-from-dateTime", datetime::seconds_from_date_time);
    seconds!("seconds-from-time", datetime::seconds_from_time);

    macro_rules! timezone {
        ($local:literal, $function:path) => {
            registry.add(FN_NAMESPACE, $local, 1, |_, arguments| {
                Ok(match atomic(arguments, 0)? {
                    Some(value) => as_optional(datetime::timezone_value($function(&value)?)),
                    None => Sequence::empty(),
                })
            });
        };
    }
    timezone!("timezone-from-dateTime", datetime::timezone_from_date_time);
    timezone!("timezone-from-date", datetime::timezone_from_date);
    timezone!("timezone-from-time", datetime::timezone_from_time);

    macro_rules! adjust {
        ($local:literal, $function:path) => {
            registry.add(FN_NAMESPACE, $local, 1, |context, arguments| {
                Ok(match atomic(arguments, 0)? {
                    Some(value) => Sequence::from($function(
                        context,
                        &value,
                        datetime::TimezoneArgument::Implicit,
                    )?),
                    None => Sequence::empty(),
                })
            });
            registry.add(FN_NAMESPACE, $local, 2, |context, arguments| {
                let Some(value) = atomic(arguments, 0)? else {
                    return Ok(Sequence::empty());
                };
                let offset = atomic(arguments, 1)?;
                let timezone = datetime::TimezoneArgument::from_optional_duration(offset.as_ref())?;
                Ok(Sequence::from($function(context, &value, timezone)?))
            });
        };
    }
    adjust!(
        "adjust-dateTime-to-timezone",
        datetime::adjust_date_time_to_timezone
    );
    adjust!("adjust-date-to-timezone", datetime::adjust_date_to_timezone);
    adjust!("adjust-time-to-timezone", datetime::adjust_time_to_timezone);

    registry.add(FN_NAMESPACE, "dateTime", 2, |_, arguments| {
        let (date, time) = (atomic(arguments, 0)?, atomic(arguments, 1)?);
        let (Some(date), Some(time)) = (date, time) else {
            return Ok(Sequence::empty());
        };
        Ok(Sequence::from(datetime::date_time(&date, &time)?))
    });
    registry.add(FN_NAMESPACE, "current-dateTime", 0, |context, _| {
        Ok(Sequence::from(datetime::current_date_time(context)))
    });
    registry.add(FN_NAMESPACE, "current-date", 0, |context, _| {
        Ok(Sequence::from(datetime::current_date(context)?))
    });
    registry.add(FN_NAMESPACE, "current-time", 0, |context, _| {
        Ok(Sequence::from(datetime::current_time(context)))
    });
    registry.add(FN_NAMESPACE, "implicit-timezone", 0, |context, _| {
        Ok(Sequence::from(datetime::implicit_timezone(context)))
    });

    macro_rules! format_temporal {
        ($local:literal, $function:path) => {
            for arity in [2_usize, 5] {
                registry.add(FN_NAMESPACE, $local, arity, move |context, arguments| {
                    let Some(value) = atomic(arguments, 0)? else {
                        return Ok(Sequence::empty());
                    };
                    let picture = text(arguments, 1)?;
                    let (language, calendar, place) = if arity == 5 {
                        (
                            optional_text(arguments, 2)?,
                            optional_text(arguments, 3)?,
                            optional_text(arguments, 4)?,
                        )
                    } else {
                        (None, None, None)
                    };
                    Ok(as_string($function(
                        context,
                        &value,
                        &picture,
                        language.as_deref(),
                        calendar.as_deref(),
                        place.as_deref(),
                    )?))
                });
            }
        };
    }
    format_temporal!("format-dateTime", datetime::format_date_time);
    format_temporal!("format-date", datetime::format_date);
    format_temporal!("format-time", datetime::format_time);
}

fn register_qnames(registry: &mut FunctionRegistry) {
    registry.add(FN_NAMESPACE, "resolve-QName", 2, |_, arguments| {
        let Some(lexical) = optional_text(arguments, 0)? else {
            return Ok(Sequence::empty());
        };
        let element = required_node(arguments, 1, "fn:resolve-QName")?;
        qnames::require_element(&element, "fn:resolve-QName")?;
        Ok(Sequence::from(AtomicValue::qname(qnames::resolve_qname(
            &lexical, &element,
        )?)))
    });
    registry.add(FN_NAMESPACE, "QName", 2, |_, arguments| {
        let namespace = optional_text(arguments, 0)?;
        let lexical = text(arguments, 1)?;
        Ok(Sequence::from(AtomicValue::qname(qnames::qname(
            namespace.as_deref(),
            &lexical,
        )?)))
    });
    registry.add(FN_NAMESPACE, "prefix-from-QName", 1, |_, arguments| {
        if argument(arguments, 0).is_empty() {
            return Ok(Sequence::empty());
        }
        let value = qname(arguments, 0, "fn:prefix-from-QName")?;
        Ok(match qnames::prefix_from_qname(&value) {
            Some(prefix) => Sequence::from(qnames::ncname_value(&prefix)?),
            None => Sequence::empty(),
        })
    });
    registry.add(FN_NAMESPACE, "local-name-from-QName", 1, |_, arguments| {
        if argument(arguments, 0).is_empty() {
            return Ok(Sequence::empty());
        }
        let value = qname(arguments, 0, "fn:local-name-from-QName")?;
        Ok(Sequence::from(qnames::ncname_value(
            &qnames::local_name_from_qname(&value),
        )?))
    });
    registry.add(
        FN_NAMESPACE,
        "namespace-uri-from-QName",
        1,
        |_, arguments| {
            if argument(arguments, 0).is_empty() {
                return Ok(Sequence::empty());
            }
            let value = qname(arguments, 0, "fn:namespace-uri-from-QName")?;
            Ok(Sequence::from(AtomicValue::any_uri(
                qnames::namespace_uri_from_qname(&value),
            )))
        },
    );
    registry.add(
        FN_NAMESPACE,
        "namespace-uri-for-prefix",
        2,
        |_, arguments| {
            let prefix = optional_text(arguments, 0)?;
            let element = required_node(arguments, 1, "fn:namespace-uri-for-prefix")?;
            qnames::require_element(&element, "fn:namespace-uri-for-prefix")?;
            Ok(
                match qnames::namespace_uri_for_prefix(prefix.as_deref(), &element) {
                    Some(namespace) => Sequence::from(AtomicValue::any_uri(namespace)),
                    None => Sequence::empty(),
                },
            )
        },
    );
    registry.add(FN_NAMESPACE, "in-scope-prefixes", 1, |_, arguments| {
        let element = required_node(arguments, 0, "fn:in-scope-prefixes")?;
        qnames::require_element(&element, "fn:in-scope-prefixes")?;
        Ok(Sequence::from_iter(
            qnames::in_scope_prefixes(&element)
                .into_iter()
                .map(AtomicValue::string),
        ))
    });
}

fn register_nodes(registry: &mut FunctionRegistry) {
    macro_rules! naming {
        ($local:literal, $function:path, $wrap:path) => {
            registry.add(FN_NAMESPACE, $local, 1, |_, arguments| {
                Ok(match node(arguments, 0)? {
                    Some(node) => Sequence::from($wrap($function(&node))),
                    None => Sequence::from($wrap(String::new())),
                })
            });
        };
    }
    naming!("name", nodes::name, AtomicValue::string);
    naming!("local-name", nodes::local_name, AtomicValue::string);
    naming!("namespace-uri", nodes::namespace_uri, AtomicValue::any_uri);

    registry.add(FN_NAMESPACE, "root", 1, |_, arguments| {
        Ok(match node(arguments, 0)? {
            Some(node) => Sequence::from(nodes::root(&node)),
            None => Sequence::empty(),
        })
    });
    registry.add(FN_NAMESPACE, "has-children", 1, |_, arguments| {
        Ok(match node(arguments, 0)? {
            Some(node) => as_boolean(nodes::has_children(&node)),
            None => as_boolean(false),
        })
    });
    registry.add(FN_NAMESPACE, "lang", 2, |_, arguments| {
        let test = text(arguments, 0)?;
        let node = required_node(arguments, 1, "fn:lang")?;
        Ok(as_boolean(nodes::lang(&test, &node)))
    });
    registry.add(FN_NAMESPACE, "generate-id", 1, |_, arguments| {
        Ok(match node(arguments, 0)? {
            Some(node) => as_string(nodes::generate_id(&node)),
            None => as_string(String::new()),
        })
    });
    macro_rules! ids {
        ($local:literal, $function:path) => {
            registry.add(FN_NAMESPACE, $local, 2, |_, arguments| {
                let values = strings_of(&argument(arguments, 0))?;
                let node = required_node(arguments, 1, $local)?;
                Ok(as_nodes($function(&values, &node)?))
            });
        };
    }
    registry.add(FN_NAMESPACE, "path", 1, |_, arguments| {
        Ok(match node(arguments, 0)? {
            Some(node) => as_string(nodes::path(&node)),
            None => Sequence::empty(),
        })
    });
    macro_rules! hierarchy {
        ($local:literal, $function:path) => {
            registry.add(FN_NAMESPACE, $local, 1, |_, arguments| {
                let mut nodes_in = Vec::new();
                for item in argument(arguments, 0).iter() {
                    let node = item.as_node().cloned().ok_or_else(|| {
                        type_error(concat!("fn:", $local, " expects a sequence of nodes"))
                    })?;
                    nodes_in.push(node);
                }
                Ok(as_nodes($function(&nodes_in)))
            });
        };
    }
    hierarchy!("innermost", nodes::innermost);
    hierarchy!("outermost", nodes::outermost);

    ids!("id", nodes::id);
    ids!("idref", nodes::idref);
    ids!("element-with-id", nodes::element_with_id);

    // [XSLT 3.0 §18.9](https://www.w3.org/TR/xslt-30/#func-copy-of): `item()* -> item()*` — a
    // node in the input is replaced by a fresh copy, and every other item passes through
    // unchanged. Only the one-argument form lives here: the zero-argument form (defaulting to
    // the context item) needs the dynamic context this crate's functions do not have access to,
    // so it is registered wherever that context is available — inside `oxixml-xslt`, the only
    // host this processor gives these two functions to.
    macro_rules! copies {
        ($local:literal, $function:path) => {
            registry.add(FN_NAMESPACE, $local, 1, |_, arguments| {
                copy_or_pass_through(&argument(arguments, 0), $function)
            });
        };
    }
    copies!("copy-of", nodes::copy_of);
    copies!("snapshot", nodes::snapshot);
}

/// Replaces every node in `sequence` with `copy(node)`, leaving every other item as it is —
/// the shape `fn:copy-of` and `fn:snapshot` share.
fn copy_or_pass_through(
    sequence: &Sequence,
    copy: impl Fn(&NodeRef) -> FnResult<NodeRef>,
) -> FnResult<Sequence> {
    let mut out = Vec::with_capacity(sequence.len());
    for item in sequence.iter() {
        out.push(match item {
            Item::Node(node) => Item::Node(copy(node)?),
            other => other.clone(),
        });
    }
    Ok(Sequence::from_items(out))
}

fn register_sequences(registry: &mut FunctionRegistry) {
    registry.add(FN_NAMESPACE, "empty", 1, |_, arguments| {
        Ok(as_boolean(sequences::empty(&argument(arguments, 0))))
    });
    registry.add(FN_NAMESPACE, "exists", 1, |_, arguments| {
        Ok(as_boolean(sequences::exists(&argument(arguments, 0))))
    });
    registry.add(FN_NAMESPACE, "head", 1, |_, arguments| {
        Ok(sequences::head(&argument(arguments, 0)))
    });
    registry.add(FN_NAMESPACE, "tail", 1, |_, arguments| {
        Ok(sequences::tail(&argument(arguments, 0)))
    });
    registry.add(FN_NAMESPACE, "reverse", 1, |_, arguments| {
        Ok(sequences::reverse(&argument(arguments, 0)))
    });
    registry.add(FN_NAMESPACE, "unordered", 1, |_, arguments| {
        Ok(sequences::unordered(&argument(arguments, 0)))
    });
    registry.add(FN_NAMESPACE, "count", 1, |_, arguments| {
        Ok(as_integer(sequences::count(&argument(arguments, 0))))
    });
    registry.add(FN_NAMESPACE, "insert-before", 3, |_, arguments| {
        let position = integer(arguments, 1)?.unwrap_or(1);
        Ok(sequences::insert_before(
            &argument(arguments, 0),
            position,
            &argument(arguments, 2),
        ))
    });
    registry.add(FN_NAMESPACE, "remove", 2, |_, arguments| {
        let position = integer(arguments, 1)?.unwrap_or(0);
        Ok(sequences::remove(&argument(arguments, 0), position))
    });
    registry.add(FN_NAMESPACE, "subsequence", 2, |_, arguments| {
        let start = required_number(arguments, 1, "fn:subsequence")?;
        Ok(sequences::subsequence(&argument(arguments, 0), start, None))
    });
    registry.add(FN_NAMESPACE, "subsequence", 3, |_, arguments| {
        let start = required_number(arguments, 1, "fn:subsequence")?;
        let length = required_number(arguments, 2, "fn:subsequence")?;
        Ok(sequences::subsequence(
            &argument(arguments, 0),
            start,
            Some(length),
        ))
    });
    for arity in 2..=3 {
        registry.add(
            FN_NAMESPACE,
            "index-of",
            arity,
            move |context, arguments| {
                let Some(search) = atomic(arguments, 1)? else {
                    return Ok(Sequence::empty());
                };
                let collation = if arity == 3 {
                    collation(arguments, 2)?
                } else {
                    None
                };
                let positions = sequences::index_of(
                    context,
                    &argument(arguments, 0),
                    &search,
                    collation.as_deref(),
                )?;
                Ok(Sequence::from_iter(
                    positions.into_iter().map(AtomicValue::integer),
                ))
            },
        );
    }
    for arity in 1..=2 {
        registry.add(
            FN_NAMESPACE,
            "distinct-values",
            arity,
            move |context, arguments| {
                let collation = if arity == 2 {
                    collation(arguments, 1)?
                } else {
                    None
                };
                sequences::distinct_values(context, &argument(arguments, 0), collation.as_deref())
            },
        );
        registry.add(FN_NAMESPACE, "max", arity, move |context, arguments| {
            let collation = if arity == 2 {
                collation(arguments, 1)?
            } else {
                None
            };
            Ok(as_optional(sequences::max(
                context,
                &argument(arguments, 0),
                collation.as_deref(),
            )?))
        });
        registry.add(FN_NAMESPACE, "min", arity, move |context, arguments| {
            let collation = if arity == 2 {
                collation(arguments, 1)?
            } else {
                None
            };
            Ok(as_optional(sequences::min(
                context,
                &argument(arguments, 0),
                collation.as_deref(),
            )?))
        });
    }
    for arity in 2..=3 {
        registry.add(
            FN_NAMESPACE,
            "deep-equal",
            arity,
            move |context, arguments| {
                let collation = if arity == 3 {
                    collation(arguments, 2)?
                } else {
                    None
                };
                Ok(as_boolean(sequences::deep_equal(
                    context,
                    &argument(arguments, 0),
                    &argument(arguments, 1),
                    collation.as_deref(),
                )?))
            },
        );
    }
    registry.add(FN_NAMESPACE, "zero-or-one", 1, |_, arguments| {
        sequences::zero_or_one(&argument(arguments, 0))
    });
    registry.add(FN_NAMESPACE, "one-or-more", 1, |_, arguments| {
        sequences::one_or_more(&argument(arguments, 0))
    });
    registry.add(FN_NAMESPACE, "exactly-one", 1, |_, arguments| {
        sequences::exactly_one(&argument(arguments, 0))
    });
    registry.add(FN_NAMESPACE, "avg", 1, |_, arguments| {
        Ok(as_optional(sequences::avg(&argument(arguments, 0))?))
    });
    registry.add(FN_NAMESPACE, "sum", 1, |_, arguments| {
        let zero = AtomicValue::integer(0_i64);
        Ok(as_optional(sequences::sum(
            &argument(arguments, 0),
            Some(&zero),
        )?))
    });
    registry.add(FN_NAMESPACE, "sum", 2, |_, arguments| {
        let zero = atomic(arguments, 1)?;
        Ok(as_optional(sequences::sum(
            &argument(arguments, 0),
            zero.as_ref(),
        )?))
    });
}

fn register_maps(registry: &mut FunctionRegistry) {
    let namespace = maps::MAP_NAMESPACE;
    registry.add(namespace, "size", 1, |_, arguments| {
        Ok(as_integer(maps::size(&map_argument(
            arguments, 0, "map:size",
        )?)))
    });
    registry.add(namespace, "keys", 1, |_, arguments| {
        Ok(maps::keys(&map_argument(arguments, 0, "map:keys")?))
    });
    registry.add(namespace, "contains", 2, |_, arguments| {
        let map = map_argument(arguments, 0, "map:contains")?;
        let key = required_atomic(arguments, 1, "map:contains")?;
        Ok(as_boolean(maps::contains(&map, &key)))
    });
    registry.add(namespace, "get", 2, |_, arguments| {
        let map = map_argument(arguments, 0, "map:get")?;
        let key = required_atomic(arguments, 1, "map:get")?;
        Ok(maps::get(&map, &key))
    });
    registry.add(namespace, "put", 3, |_, arguments| {
        let map = map_argument(arguments, 0, "map:put")?;
        let key = required_atomic(arguments, 1, "map:put")?;
        Ok(Sequence::singleton(Item::from(maps::put(
            &map,
            key,
            argument(arguments, 2),
        ))))
    });
    registry.add(namespace, "remove", 2, |_, arguments| {
        let map = map_argument(arguments, 0, "map:remove")?;
        let keys: Vec<AtomicValue> = argument(arguments, 1)
            .atomize()?
            .iter()
            .filter_map(|item| item.as_atomic().cloned())
            .collect();
        Ok(Sequence::singleton(Item::from(maps::remove(&map, &keys))))
    });
    registry.add(namespace, "entry", 2, |_, arguments| {
        let key = required_atomic(arguments, 0, "map:entry")?;
        Ok(Sequence::singleton(Item::from(maps::entry(
            key,
            argument(arguments, 1),
        ))))
    });
    for arity in 1..=2 {
        registry.add(namespace, "merge", arity, move |_, arguments| {
            let maps_in: Vec<XdmMap> = argument(arguments, 0)
                .iter()
                .filter_map(|item| item.as_map().cloned())
                .collect();
            let duplicates = if arity == 2 {
                let options = map_argument(arguments, 1, "map:merge")?;
                match options.get(&AtomicValue::string("duplicates")) {
                    Some(value) => {
                        maps::duplicates_from_name(&crate::accessors::string_of(value)?)?
                    }
                    None => DuplicateKeys::UseFirst,
                }
            } else {
                DuplicateKeys::UseFirst
            };
            Ok(Sequence::singleton(Item::from(maps::merge(
                &maps_in, duplicates,
            )?)))
        });
    }
    registry.add(namespace, "find", 2, |_, arguments| {
        let key = required_atomic(arguments, 1, "map:find")?;
        Ok(Sequence::singleton(Item::from(maps::find(
            &argument(arguments, 0),
            &key,
        ))))
    });
}

fn register_arrays(registry: &mut FunctionRegistry) {
    let namespace = arrays::ARRAY_NAMESPACE;
    registry.add(namespace, "size", 1, |_, arguments| {
        Ok(as_integer(arrays::size(&array_argument(
            arguments,
            0,
            "array:size",
        )?)))
    });
    registry.add(namespace, "get", 2, |_, arguments| {
        let array = array_argument(arguments, 0, "array:get")?;
        let position = integer(arguments, 1)?.unwrap_or(0);
        arrays::get(&array, position)
    });
    registry.add(namespace, "put", 3, |_, arguments| {
        let array = array_argument(arguments, 0, "array:put")?;
        let position = integer(arguments, 1)?.unwrap_or(0);
        Ok(Sequence::singleton(Item::from(arrays::put(
            &array,
            position,
            argument(arguments, 2),
        )?)))
    });
    registry.add(namespace, "append", 2, |_, arguments| {
        let array = array_argument(arguments, 0, "array:append")?;
        Ok(Sequence::singleton(Item::from(arrays::append(
            &array,
            argument(arguments, 1),
        ))))
    });
    for arity in 2..=3 {
        registry.add(namespace, "subarray", arity, move |_, arguments| {
            let array = array_argument(arguments, 0, "array:subarray")?;
            let start = integer(arguments, 1)?.unwrap_or(1);
            let length = if arity == 3 {
                integer(arguments, 2)?
            } else {
                None
            };
            Ok(Sequence::singleton(Item::from(arrays::subarray(
                &array, start, length,
            )?)))
        });
    }
    registry.add(namespace, "remove", 2, |_, arguments| {
        let array = array_argument(arguments, 0, "array:remove")?;
        let positions: Vec<i64> = argument(arguments, 1)
            .atomize()?
            .iter()
            .filter_map(|item| item.as_atomic().and_then(AtomicValue::as_f64))
            .map(|value| value.trunc().clamp(-9.0e18, 9.0e18) as i64)
            .collect();
        Ok(Sequence::singleton(Item::from(arrays::remove(
            &array, &positions,
        )?)))
    });
    registry.add(namespace, "insert-before", 3, |_, arguments| {
        let array = array_argument(arguments, 0, "array:insert-before")?;
        let position = integer(arguments, 1)?.unwrap_or(1);
        Ok(Sequence::singleton(Item::from(arrays::insert_before(
            &array,
            position,
            argument(arguments, 2),
        )?)))
    });
    registry.add(namespace, "head", 1, |_, arguments| {
        arrays::head(&array_argument(arguments, 0, "array:head")?)
    });
    registry.add(namespace, "tail", 1, |_, arguments| {
        Ok(Sequence::singleton(Item::from(arrays::tail(
            &array_argument(arguments, 0, "array:tail")?,
        )?)))
    });
    registry.add(namespace, "reverse", 1, |_, arguments| {
        Ok(Sequence::singleton(Item::from(arrays::reverse(
            &array_argument(arguments, 0, "array:reverse")?,
        ))))
    });
    registry.add(namespace, "join", 1, |_, arguments| {
        let arrays_in: Vec<XdmArray> = argument(arguments, 0)
            .iter()
            .filter_map(|item| item.as_array().cloned())
            .collect();
        Ok(Sequence::singleton(Item::from(arrays::join(&arrays_in))))
    });
    registry.add(namespace, "flatten", 1, |_, arguments| {
        Ok(arrays::flatten(&argument(arguments, 0)))
    });
}

/// The two functions that hand the work to a *separate* processor:
/// [`fn:load-xquery-module`](https://www.w3.org/TR/xpath-functions-31/#func-load-xquery-module)
/// and [`fn:transform`](https://www.w3.org/TR/xpath-functions-31/#func-transform).
///
/// F&O 3.1 makes both optional and gives each the error a processor raises when it cannot serve
/// the request: `FOQM0006` when no XQuery processor is available to compile and run a library
/// module, and `FOXT0004` when the transformation is not carried out. This library evaluates
/// expressions; it neither compiles library modules nor drives a stylesheet from inside one, so
/// every call reports the corresponding error.
///
/// They are registered all the same, because the specification distinguishes a function that is
/// *absent* from one that is *unavailable*: an absent name is `XPST0017` at the call site and
/// makes `fn:function-lookup` answer the empty sequence, whereas these two must be found — and
/// fail only when they are invoked.
///
/// Neither declares its parameter types in [`crate::signature`], so an argument reaches the body
/// unconverted: the unavailability error is what a call reports whatever it was passed, rather
/// than an argument type error raised on the way in. A host that does have an XQuery or XSLT
/// processor to offer replaces these entries by registering its own under the same name.
fn register_external_processors(registry: &mut FunctionRegistry) {
    for arity in 1..=2 {
        registry.add(FN_NAMESPACE, "load-xquery-module", arity, |_, _| {
            Err(FnError::new(
                FnErrorCode::Foqm0006,
                "no XQuery processor is available to fn:load-xquery-module",
            ))
        });
    }
    registry.add(FN_NAMESPACE, "transform", 1, |_, _| {
        Err(FnError::new(
            FnErrorCode::Foxt0004,
            "fn:transform performs no transformation in this processor",
        ))
    });
}

fn register_xmlio(registry: &mut FunctionRegistry) {
    // F&O 3.1 §14.7.1–2 declare `fn:parse-xml` and `fn:parse-xml-fragment` with *one* argument
    // each; the two-argument forms of the working drafts were dropped, and calling one is
    // `XPST0017` (QT3 `parse-xml-002`, `parse-xml-fragment-002`). The base URI of the result is
    // the static base URI of the caller.
    registry.add(FN_NAMESPACE, "parse-xml", 1, |context, arguments| {
        let Some(xml) = optional_text(arguments, 0)? else {
            return Ok(Sequence::empty());
        };
        Ok(Sequence::from(xmlio::parse_xml(
            &xml,
            context.static_base_uri(),
        )?))
    });
    registry.add(
        FN_NAMESPACE,
        "parse-xml-fragment",
        1,
        |context, arguments| {
            let Some(xml) = optional_text(arguments, 0)? else {
                return Ok(Sequence::empty());
            };
            Ok(Sequence::from(xmlio::parse_xml_fragment(
                &xml,
                context.static_base_uri(),
            )?))
        },
    );
    registry.add(FN_NAMESPACE, "serialize", 1, |_, arguments| {
        Ok(as_string(xmlio::serialize(
            &argument(arguments, 0),
            &xmlio::SerializationParameters::default(),
        )?))
    });
    registry.add(FN_NAMESPACE, "serialize", 2, |_, arguments| {
        let parameters = serialization_parameters(&argument(arguments, 1))?;
        Ok(as_string(xmlio::serialize(
            &argument(arguments, 0),
            &parameters,
        )?))
    });
    register_json(registry);
}

/// The JSON functions of F&O §17.5–§17.6. `fn:json-doc` is *not* here: it retrieves a resource,
/// which is an evaluator's business.
fn register_json(registry: &mut FunctionRegistry) {
    for arity in 1..=2 {
        registry.add(FN_NAMESPACE, "parse-json", arity, move |_, arguments| {
            let Some(text) = optional_text(arguments, 0)? else {
                return Ok(Sequence::empty());
            };
            let options = json_options(arguments, arity, json::JsonOptions::default())?;
            json::parse_json(&text, &options)
        });
        registry.add(
            FN_NAMESPACE,
            "json-to-xml",
            arity,
            move |context, arguments| {
                let Some(text) = optional_text(arguments, 0)? else {
                    return Ok(Sequence::empty());
                };
                let defaults = json::JsonOptions {
                    duplicates: json::parser::Duplicates::Retain,
                    function: json::JsonFunction::JsonToXml,
                    ..json::JsonOptions::default()
                };
                let options = json_options(arguments, arity, defaults)?;
                Ok(Sequence::from(json::json_to_xml_with_base(
                    &text,
                    &options,
                    context.static_base_uri(),
                )?))
            },
        );
        registry.add(FN_NAMESPACE, "xml-to-json", arity, move |_, arguments| {
            let Some(node) = node(arguments, 0)? else {
                return Ok(Sequence::empty());
            };
            let options = json_options(arguments, arity, json::JsonOptions::default())?;
            Ok(as_string(json::xml_to_json(&node, &options)?))
        });
    }
}

/// Reads the options map the JSON functions take as their second argument.
fn json_options<'a>(
    arguments: &[Sequence],
    arity: usize,
    defaults: json::JsonOptions<'a>,
) -> FnResult<json::JsonOptions<'a>> {
    if arity < 2 {
        return Ok(defaults);
    }
    match argument(arguments, 1).first() {
        Some(Item::Map(options)) => json::JsonOptions::from_map(options, defaults),
        None => Ok(defaults),
        Some(_) => Err(type_error("the options argument must be a map")),
    }
}

/// Reads `fn:serialize`'s second argument, which XPath 3.1 allows to be a map of serialization
/// parameters.
fn serialization_parameters(sequence: &Sequence) -> FnResult<xmlio::SerializationParameters> {
    let mut parameters = xmlio::SerializationParameters::default();
    // XPath 3.1 gives `fn:serialize` two spellings for its parameters: an option map, and the
    // `output:serialization-parameters` *element* of Serialization 3.1 §3.1.
    if let Some(Item::Node(node)) = sequence.first() {
        use oxixml_infoset::{NodeKind, XdmNode as _};
        let element = if node.node_kind() == NodeKind::Document {
            node.children()
                .into_iter()
                .find(|child| child.node_kind() == NodeKind::Element)
                .unwrap_or_else(|| node.clone())
        } else {
            node.clone()
        };
        return xmlio::parameters_from_element(&element);
    }
    let Some(Item::Map(options)) = sequence.first() else {
        return Ok(parameters);
    };
    for (key, value) in options.entries() {
        // Serialization 3.1 §3.1: a parameter this specification defines is keyed by an
        // `xs:string`; an `xs:QName` key in a namespace names an implementation-defined
        // parameter, and one in *no* namespace names nothing at all. Neither is this
        // specification's parameter, so both are ignored rather than applied by their string
        // value (QT3 `serialize-xml-120`).
        if key.atomic_type() != oxixml_infoset::AtomicType::String {
            continue;
        }
        let name = key.string_value();
        parameters = match name.as_str() {
            "method" => {
                parameters.with_method(output_method(&required_string(value, &name)?, &name)?)
            }
            "json-node-output-method" => parameters.with_json_node_output_method(output_method(
                &required_string(value, &name)?,
                &name,
            )?),
            "indent" => parameters.with_indent(required_boolean(value, &name)?),
            "omit-xml-declaration" => {
                parameters.with_omit_xml_declaration(required_boolean(value, &name)?)
            }
            "allow-duplicate-names" => {
                parameters.with_allow_duplicate_names(required_boolean(value, &name)?)
            }
            "undeclare-prefixes"
            | "byte-order-mark"
            | "escape-uri-attributes"
            | "include-content-type" => {
                // Accepted, not acted on — but the option parameter conventions still apply,
                // so a value of the wrong type is `XPTY0004`.
                required_boolean(value, &name)?;
                parameters
            }
            "version" => parameters.with_version(required_string(value, &name)?),
            "encoding" => parameters.with_encoding(required_string(value, &name)?),
            "standalone" => match value.as_slice() {
                [Item::Atomic(atomic)]
                    if atomic.atomic_type() == oxixml_infoset::AtomicType::String =>
                {
                    if atomic.string_value().trim() == "omit" {
                        parameters.with_standalone(None)
                    } else {
                        return Err(type_error(
                            "the only string value of `standalone` is `omit`",
                        ));
                    }
                }
                _ => parameters.with_standalone(Some(required_boolean(value, &name)?)),
            },
            "item-separator" => {
                parameters.with_item_separator(Some(required_string(value, &name)?))
            }
            "doctype-system" => {
                parameters.with_doctype_system(Some(required_string(value, &name)?))
            }
            "doctype-public" => {
                parameters.with_doctype_public(Some(required_string(value, &name)?))
            }
            "html-version" => match value.as_slice() {
                [Item::Atomic(atomic)] if atomic.atomic_type().is_numeric() => {
                    parameters.with_html_version(atomic.string_value())
                }
                _ => {
                    return Err(type_error("`html-version` needs one xs:decimal"));
                }
            },
            "media-type" | "normalization-form" => {
                required_string(value, &name)?;
                parameters
            }
            "use-character-maps" => parameters.with_character_map(character_map_option(value)?),
            "cdata-section-elements" => {
                parameters.with_cdata_section_elements(xmlio::qnames_from_option(value, &name)?)
            }
            "suppress-indentation" => {
                // Accepted, not acted on — but Serialization 3.1 §3.2 types the entry `xs:QName*`,
                // so a value of the wrong type is `XPTY0004` (QT3 `serialize-xml-108`, `-109`).
                xmlio::qnames_from_option(value, &name)?;
                parameters
            }
            _ => parameters,
        };
    }
    Ok(parameters)
}

/// Parses a serialization `method`/`json-node-output-method` value.
///
/// [Serialization 3.1 §2](https://www.w3.org/TR/xslt-xquery-serialization-31/#serparam-error-codes)
/// mandates `SEPM0016` for a serialization parameter whose *value* is invalid, so the generic
/// `FORG0006` [`xmlio::OutputMethod::parse`] raises is re-labelled here.
fn output_method(value: &str, name: &str) -> FnResult<xmlio::OutputMethod> {
    xmlio::OutputMethod::parse(value).map_err(|_| {
        FnError::new(
            FnErrorCode::Sepm0016,
            format!("`{value}` is not a valid value for the serialization parameter `{name}`"),
        )
    })
}

/// One `xs:string` (or `xs:anyURI`/`xs:untypedAtomic`, which the option parameter conventions
/// convert) — anything else is `XPTY0004`.
fn required_string(value: &Sequence, name: &str) -> FnResult<String> {
    match value.as_slice() {
        [Item::Atomic(atomic)]
            if atomic.atomic_type().is_text()
                || atomic.atomic_type() == oxixml_infoset::AtomicType::AnyUri =>
        {
            Ok(atomic.string_value())
        }
        _ => Err(type_error(&format!(
            "the serialization parameter `{name}` needs one xs:string"
        ))),
    }
}

/// One `xs:boolean`. F&O 3.1 §2.1.1, the option parameter conventions: an `xs:untypedAtomic`
/// value carries no type of its own and is *cast* to the required type, so
/// `xs:untypedAtomic("false")` is the boolean `false`; a genuine `xs:string` is not, and the
/// strings `yes`/`true` are therefore `XPTY0004` (QT3 `serialize-json-134`).
fn required_boolean(value: &Sequence, name: &str) -> FnResult<bool> {
    match value.as_slice() {
        [Item::Atomic(atomic)] if atomic.atomic_type() == oxixml_infoset::AtomicType::Boolean => {
            Ok(atomic.as_bool().unwrap_or_default())
        }
        [Item::Atomic(atomic)]
            if atomic.atomic_type() == oxixml_infoset::AtomicType::UntypedAtomic =>
        {
            match atomic.string_value().trim() {
                "true" | "1" => Ok(true),
                "false" | "0" => Ok(false),
                _ => Err(type_error(&format!(
                    "the serialization parameter `{name}` needs one xs:boolean"
                ))),
            }
        }
        _ => Err(type_error(&format!(
            "the serialization parameter `{name}` needs one xs:boolean"
        ))),
    }
}

/// The `use-character-maps` parameter given as a map, whose keys and values must both be single
/// `xs:string` values — the option parameter conventions are *not* applied recursively to them,
/// so an `xs:untypedAtomic`, a node or an `xs:QName` is `XPTY0004`.
fn character_map_option(value: &Sequence) -> FnResult<Vec<(char, String)>> {
    let [Item::Map(mappings)] = value.as_slice() else {
        return Err(type_error("`use-character-maps` needs one map"));
    };
    let mut out = Vec::new();
    for (key, mapped) in mappings.entries() {
        if key.atomic_type() != oxixml_infoset::AtomicType::String {
            return Err(type_error(
                "every key of `use-character-maps` must be an xs:string",
            ));
        }
        let [Item::Atomic(replacement)] = mapped.as_slice() else {
            return Err(type_error(
                "every value of `use-character-maps` must be one xs:string",
            ));
        };
        if replacement.atomic_type() != oxixml_infoset::AtomicType::String {
            return Err(type_error(
                "every value of `use-character-maps` must be an xs:string",
            ));
        }
        let text = key.string_value();
        let mut characters = text.chars();
        let (Some(single), None) = (characters.next(), characters.next()) else {
            return Err(type_error(
                "every key of `use-character-maps` must be exactly one character",
            ));
        };
        out.push((single, replacement.string_value()));
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
