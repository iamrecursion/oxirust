//! [`XpathFunction`] — the callable item behind `function(…) {…}`, `fn:abs#1`, `$f(…)`, `?:` and
//! partial application.
//!
//! [`oxixml_fn::FunctionItem`] is the function item of the *function library*: a closure from
//! [`Sequence`](oxixml_infoset::Sequence)s to a `Sequence`. That is exactly right for a library
//! function, and exactly not enough for a language: an XPath 3.0 inline function closes over the
//! variables in scope, may take and return other function items, and must be evaluated in the
//! engine's own dynamic context. `XpathFunction` is that richer item, and it wraps a library
//! function item, a map or an array transparently, so `$m("k")`, `$a(1)` and `fn:abs#1` are all
//! one call path.

use crate::context::DynamicContext;
use crate::error::{XPathError, XPathErrorCode, XPathResult};
use crate::types::SequenceType;
use crate::value::{XpathItem, XpathValue};
use oxixml_fn::FunctionItem;
use oxixml_infoset::{AtomicData, ExpandedName, FunctionBody, XdmArray, XdmFunction, XdmMap};
use std::any::Any;
use std::fmt;
use std::sync::{Arc, OnceLock};

/// The parameter and result types a function item declares.
///
/// XPath 3.1 §2.5.5.5 tests a function item against `function(A₁, …, Aₙ) as R` by comparing
/// *declared* types, not by inspecting what the implementation would do — so a function item
/// that declares nothing satisfies only `function(*)` and the arity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionSignature {
    /// The declared parameter types, in order.
    pub parameters: Vec<SequenceType>,
    /// The declared result type.
    pub result: SequenceType,
}

/// The Rust signature every engine-implemented function has.
type NativeImpl =
    Arc<dyn Fn(&DynamicContext, &[XpathValue]) -> XPathResult<XpathValue> + Send + Sync>;

#[derive(Clone)]
enum Kind {
    /// A function implemented directly by this engine: a context-dependent built-in, a
    /// higher-order built-in, an inline function, or a partial application.
    Native(NativeImpl),
    /// A function from the [`oxixml_fn`] library, which neither takes nor returns function
    /// items.
    Library(FunctionItem),
    /// A map, read as the function `$key -> $value` XDM says it is.
    Map(XdmMap),
    /// An array, read as the function `$position -> $member` XDM says it is.
    Array(XdmArray),
}

/// A function item.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::AtomicValue;
/// use oxixml_xpath::{DynamicContext, StaticContext, XpathFunction, XpathValue};
/// use std::sync::Arc;
///
/// let twice = XpathFunction::native(None, 1, |_context, arguments| {
///     let value = arguments[0]
///         .atomize_at_most_one("the argument")?
///         .and_then(|value| value.as_f64())
///         .unwrap_or_default();
///     Ok(XpathValue::from(AtomicValue::double(value * 2.0)))
/// });
/// assert_eq!(twice.arity(), 1);
/// assert!(twice.name().is_none());
///
/// let context = DynamicContext::new(Arc::new(StaticContext::new()));
/// let result = twice.call(&context, &[XpathValue::from(AtomicValue::integer(21_i64))])?;
/// assert_eq!(result.to_string(), "42");
/// # Ok::<(), oxixml_xpath::XPathError>(())
/// ```
#[derive(Clone)]
pub struct XpathFunction {
    name: Option<ExpandedName>,
    arity: usize,
    kind: Kind,
    /// The signature the function *declares*, when it has one.
    ///
    /// An inline function expression declares one (its parameter and result types, each
    /// defaulting to `item()*`), and that is what a `function(…) as …` item test has to be
    /// judged against; a built-in wrapped from the library carries none, and such a test can
    /// then only check the arity.
    signature: Option<Arc<FunctionSignature>>,
    /// The data-model handle for this very function item, built on demand.
    ///
    /// XDM gives function items reference identity, and [`XdmFunction`] honours that with
    /// `Arc::ptr_eq`. Caching the handle here means that putting the *same* `XpathFunction`
    /// into a sequence twice yields the *same* [`XdmFunction`], so identity survives the round
    /// trip through the data model.
    xdm: OnceLock<XdmFunction>,
}

/// The [`FunctionBody`] adapter that lets an [`XpathFunction`] live inside an
/// [`oxixml_infoset::Item`].
///
/// The data model stores it opaquely — it can read only the name and the arity — and this
/// engine downcasts back to it through [`XpathFunction::from_xdm`].
#[derive(Debug)]
struct XpathFunctionBody(XpathFunction);

impl FunctionBody for XpathFunctionBody {
    fn function_name(&self) -> Option<ExpandedName> {
        self.0.name.clone()
    }

    fn arity(&self) -> usize {
        self.0.arity
    }

    fn as_any(&self) -> &(dyn Any + Send + Sync) {
        self
    }
}

impl XpathFunction {
    /// Builds a function implemented by a Rust closure.
    pub fn native(
        name: Option<ExpandedName>,
        arity: usize,
        implementation: impl Fn(&DynamicContext, &[XpathValue]) -> XPathResult<XpathValue>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            name,
            arity,
            kind: Kind::Native(Arc::new(implementation)),
            signature: None,
            xdm: OnceLock::new(),
        }
    }

    /// Declares this function's signature, as an inline function expression does.
    #[must_use]
    pub fn with_signature(mut self, parameters: Vec<SequenceType>, result: SequenceType) -> Self {
        self.signature = Some(Arc::new(FunctionSignature { parameters, result }));
        self
    }

    /// The signature this function declares, if any.
    #[must_use]
    pub fn signature(&self) -> Option<&FunctionSignature> {
        self.signature.as_deref()
    }

    /// Gives this function the name `name`, discarding any it had.
    ///
    /// [XPath 3.1 §3.1.5.2](https://www.w3.org/TR/xpath-31/#id-function-coercion) needs it:
    /// function coercion builds a *new* function item around the one it was given, and that new
    /// item "has the same name and arity" as the original, so `fn:function-name` reports the
    /// original's name through the wrapper.
    ///
    /// The result is a distinct function item: its data-model handle is rebuilt, so identity
    /// (`fn:deep-equal`, `is`) still distinguishes the wrapper from what it wraps.
    #[must_use]
    pub fn with_name(mut self, name: Option<ExpandedName>) -> Self {
        self.name = name;
        self.xdm = OnceLock::new();
        self
    }

    /// Wraps a function item from the [`oxixml_fn`] library.
    #[must_use]
    pub fn from_library(function: FunctionItem) -> Self {
        Self {
            name: function.name().cloned(),
            arity: function.arity(),
            kind: Kind::Library(function),
            signature: None,
            xdm: OnceLock::new(),
        }
    }

    /// Wraps a map as the arity-one function XDM defines it to be.
    #[must_use]
    pub fn from_map(map: XdmMap) -> Self {
        Self {
            name: None,
            arity: 1,
            kind: Kind::Map(map),
            signature: None,
            xdm: OnceLock::new(),
        }
    }

    /// Wraps an array as the arity-one function XDM defines it to be.
    #[must_use]
    pub fn from_array(array: XdmArray) -> Self {
        Self {
            name: None,
            arity: 1,
            kind: Kind::Array(array),
            signature: None,
            xdm: OnceLock::new(),
        }
    }

    /// `fn:function-name`: the function's name, or `None` when it is anonymous.
    #[must_use]
    pub const fn name(&self) -> Option<&ExpandedName> {
        self.name.as_ref()
    }

    /// `fn:function-arity`: the number of arguments the function accepts.
    #[must_use]
    pub const fn arity(&self) -> usize {
        self.arity
    }

    /// Whether this function is the very same function item as `other`.
    ///
    /// Two closures are never observably equal, so identity is compared by pointer for engine
    /// functions and by name and arity for library functions.
    #[must_use]
    pub fn is_same_function(&self, other: &Self) -> bool {
        if self.arity != other.arity {
            return false;
        }
        match (&self.kind, &other.kind) {
            (Kind::Native(left), Kind::Native(right)) => Arc::ptr_eq(left, right),
            (Kind::Library(_), Kind::Library(_)) => self.name.is_some() && self.name == other.name,
            (Kind::Map(left), Kind::Map(right)) => left == right,
            (Kind::Array(left), Kind::Array(right)) => left == right,
            _ => false,
        }
    }

    /// Invokes the function.
    ///
    /// # Errors
    ///
    /// [`XPathErrorCode::Xpty0004`] when the number of arguments is not the function's arity,
    /// [`XPathErrorCode::Xpdy0130`] when the call nests past what
    /// [`EvaluationLimits`](crate::EvaluationLimits) allows, plus whatever the implementation
    /// raises.
    pub fn call(
        &self,
        context: &DynamicContext,
        arguments: &[XpathValue],
    ) -> XPathResult<XpathValue> {
        // A function item's implementation is opaque: an inline function re-enters the
        // evaluator, but a host-supplied one may recurse without ever passing through it, so the
        // call itself takes a level of the shared budget.
        let _guard = crate::eval::CallGuard::enter()?;
        if arguments.len() != self.arity {
            return Err(XPathError::type_error(format!(
                "{} expects {} argument(s) but was given {}",
                self.describe(),
                self.arity,
                arguments.len()
            )));
        }
        match &self.kind {
            Kind::Native(implementation) => implementation(context, arguments),
            Kind::Library(function) => {
                let mut converted = Vec::with_capacity(arguments.len());
                for argument in arguments {
                    converted.push(argument.clone().into_sequence()?);
                }
                // Reaching a library function through a function item is a *dynamic* call, and
                // the call site is published around it for the same reasons dispatch publishes
                // it around a static one — see [`crate::InvocationKind`].
                let result = crate::context::with_call_site(
                    context.focus(),
                    crate::context::InvocationKind::Dynamic,
                    || function.call(context.fn_context(), &converted),
                )?;
                Ok(XpathValue::from(result))
            }
            Kind::Map(map) => {
                let key = require_single_atomic(&arguments[0], "a map key")?;
                Ok(map
                    .get(&key)
                    .cloned()
                    .map(XpathValue::from)
                    .unwrap_or_default())
            }
            Kind::Array(array) => {
                let index = require_single_atomic(&arguments[0], "an array position")?;
                // `array:get`'s parameter is declared `xs:integer`, so the function conversion
                // rules cast an `xs:untypedAtomic` argument to it — which is what an atomized
                // node from an unvalidated document is.
                let index = if index.atomic_type() == oxixml_infoset::AtomicType::UntypedAtomic {
                    oxixml_infoset::AtomicValue::parse(
                        oxixml_infoset::AtomicType::Integer,
                        &index.string_value(),
                    )?
                } else {
                    index
                };
                // F&O 3.1 §17.3: an integer that names no member is *out of bounds*
                // (`FOAY0001`), which is not the same condition as a key that is not an integer
                // at all (`XPTY0004`).
                let AtomicData::Integer(value) = index.data() else {
                    return Err(XPathError::type_error(
                        "an array subscript must be a positive xs:integer",
                    ));
                };
                let integer = i64::from(*value);
                let position = usize::try_from(integer).map_err(|_| {
                    XPathError::new(
                        crate::error::XPathErrorCode::Foay0001,
                        format!("{integer} is not a position in the array"),
                    )
                })?;
                Ok(XpathValue::from(array.get(position)?.clone()))
            }
        }
    }

    /// Builds the partial application of this function.
    ///
    /// `bound` has one entry per parameter: `Some(value)` fixes the argument, `None` leaves a
    /// placeholder. The result's arity is the number of placeholders.
    ///
    /// # Errors
    ///
    /// [`XPathErrorCode::Xpty0004`] when `bound.len()` is not this function's arity.
    pub fn partially_apply(&self, bound: Vec<Option<XpathValue>>) -> XPathResult<Self> {
        if bound.len() != self.arity {
            return Err(XPathError::type_error(format!(
                "{} expects {} argument(s) but the partial application supplies {}",
                self.describe(),
                self.arity,
                bound.len()
            )));
        }
        let placeholders = bound.iter().filter(|slot| slot.is_none()).count();
        let base = self.clone();
        // XPath 3.1 §3.1.6: the function item a partial application produces is a *new*,
        // anonymous one — `fn:function-name(fn:substring(?, 1, 1))` is the empty sequence, not
        // `fn:substring` (`fn-function-name-021`, `-023`). Keeping the original name would also
        // misreport the arity, since the new item takes only the placeholders.
        Ok(Self::native(
            None,
            placeholders,
            move |context, arguments| {
                let mut supplied = arguments.iter();
                let mut full = Vec::with_capacity(bound.len());
                for slot in &bound {
                    match slot {
                        Some(value) => full.push(value.clone()),
                        None => full.push(supplied.next().cloned().unwrap_or_default()),
                    }
                }
                base.call(context, &full)
            },
        ))
    }

    /// This function item as the data model's opaque handle, so that it can be stored in a
    /// sequence, a map or an array.
    ///
    /// The handle is cached, so calling this twice on the same `XpathFunction` (or on clones of
    /// it) yields the same item, which is what XDM's reference identity requires.
    #[must_use]
    pub fn to_xdm(&self) -> XdmFunction {
        if let Some(existing) = self.xdm.get() {
            return existing.clone();
        }
        let candidate = XdmFunction::new(Arc::new(XpathFunctionBody(Self {
            name: self.name.clone(),
            arity: self.arity,
            kind: self.kind.clone(),
            signature: self.signature.clone(),
            xdm: OnceLock::new(),
        })));
        self.xdm.get_or_init(|| candidate).clone()
    }

    /// Recovers the function item from a data-model handle this engine created.
    ///
    /// Answers `None` for a handle some other producer built, which this engine cannot call.
    #[must_use]
    pub fn from_xdm(handle: &XdmFunction) -> Option<Self> {
        handle
            .body()
            .as_any()
            .downcast_ref::<XpathFunctionBody>()
            .map(|body| body.0.clone())
    }

    /// A short human-readable rendering — `fn:abs#1`, or a description of the arity for an
    /// anonymous function — used in diagnostics.
    #[must_use]
    pub fn describe(&self) -> String {
        match (&self.name, &self.kind) {
            (Some(name), _) => format!("{name}#{}", self.arity),
            (None, Kind::Map(map)) => format!("a map of {} entries", map.len()),
            (None, Kind::Array(array)) => format!("an array of {} members", array.len()),
            (None, _) => format!("an anonymous function of arity {}", self.arity),
        }
    }
}

/// The single atomic value an argument of `map:get` or `array:get` reduces to.
///
/// XDM makes a map the function `map:get($map, $key as xs:anyAtomicType)` and an array
/// `array:get($array, $position as xs:integer)`, so the *function conversion rules* apply to the
/// argument — which begins by atomizing it. `$index(isbn)` therefore looks a key up by the
/// value of the node, not by the node (QT3 `UseCaseR31-009`).
fn require_single_atomic(
    value: &XpathValue,
    what: &str,
) -> XPathResult<oxixml_infoset::AtomicValue> {
    if let [XpathItem::Value(oxixml_infoset::Item::Atomic(atomic))] = value.as_slice() {
        return Ok(atomic.clone());
    }
    value.atomize_at_most_one(what)?.ok_or_else(|| {
        XPathError::new(
            XPathErrorCode::Xpty0004,
            format!("{what} must be exactly one atomic value"),
        )
    })
}

impl fmt::Debug for XpathFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("XpathFunction")
            .field("name", &self.name)
            .field("arity", &self.arity)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for XpathFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

impl From<XdmMap> for XpathFunction {
    fn from(map: XdmMap) -> Self {
        Self::from_map(map)
    }
}

impl From<XdmArray> for XpathFunction {
    fn from(array: XdmArray) -> Self {
        Self::from_array(array)
    }
}

impl From<FunctionItem> for XpathFunction {
    fn from(function: FunctionItem) -> Self {
        Self::from_library(function)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::StaticContext;
    use oxixml_infoset::AtomicValue;

    fn context() -> DynamicContext {
        DynamicContext::new(Arc::new(StaticContext::new()))
    }

    #[test]
    fn arity_is_enforced() {
        let function = XpathFunction::native(None, 2, |_, _| Ok(XpathValue::empty()));
        let context = context();
        assert!(function.call(&context, &[XpathValue::empty()]).is_err());
        assert!(
            function
                .call(&context, &[XpathValue::empty(), XpathValue::empty()])
                .is_ok()
        );
    }

    #[test]
    fn partial_application_fixes_arguments() -> XPathResult<()> {
        let concat = XpathFunction::native(None, 2, |_, arguments| {
            let left = arguments[0].to_string();
            let right = arguments[1].to_string();
            Ok(XpathValue::from(AtomicValue::string(format!(
                "{left}{right}"
            ))))
        });
        let partial =
            concat.partially_apply(vec![Some(XpathValue::from(AtomicValue::string("a"))), None])?;
        assert_eq!(partial.arity(), 1);
        let result = partial.call(&context(), &[XpathValue::from(AtomicValue::string("b"))])?;
        assert_eq!(result.to_string(), "ab");
        Ok(())
    }

    #[test]
    fn maps_and_arrays_are_functions() -> XPathResult<()> {
        let map = XdmMap::new().put(
            AtomicValue::string("k"),
            oxixml_infoset::Sequence::from(AtomicValue::integer(1_i64)),
        );
        let function = XpathFunction::from_map(map);
        let hit = function.call(&context(), &[XpathValue::from(AtomicValue::string("k"))])?;
        assert_eq!(hit.to_string(), "1");
        let miss = function.call(&context(), &[XpathValue::from(AtomicValue::string("z"))])?;
        assert!(miss.is_empty());

        let array = XdmArray::from_iter([AtomicValue::string("x"), AtomicValue::string("y")]);
        let function = XpathFunction::from_array(array);
        let second = function.call(&context(), &[XpathValue::from(AtomicValue::integer(2_i64))])?;
        assert_eq!(second.to_string(), "y");
        assert!(
            function
                .call(&context(), &[XpathValue::from(AtomicValue::integer(9_i64))])
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn identity_is_by_closure_not_by_shape() {
        let one = XpathFunction::native(None, 0, |_, _| Ok(XpathValue::empty()));
        let two = XpathFunction::native(None, 0, |_, _| Ok(XpathValue::empty()));
        assert!(one.is_same_function(&one.clone()));
        assert!(!one.is_same_function(&two));
    }
}
