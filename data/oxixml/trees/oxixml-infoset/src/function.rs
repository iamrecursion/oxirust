//! [`XdmFunction`] — the opaque function item the data model carries for an evaluator.
//!
//! # Why the data model needs one at all
//!
//! [XDM 3.1 §2.8.2](https://www.w3.org/TR/xpath-datamodel-31/#function-items) makes a *function
//! item* one of the three kinds of item a sequence can hold, and maps and arrays are defined as
//! function items with extra structure. Everything a function item **is** — a closure over an
//! expression tree, a partial application, a reference to a library function — belongs to an
//! evaluator, not to the data model; but *where it can live* is a data-model question, and the
//! answer is "anywhere an item can": in a sequence, as a map's value, as an array's member.
//! `map { "next": function() { … } }` is a perfectly ordinary XDM map.
//!
//! # The design
//!
//! `XdmFunction` is therefore a **handle**, not an implementation. It stores an
//! `Arc<dyn FunctionBody>`, which tells the data model only the two things XDM itself defines
//! for every function item — its name (`fn:function-name`) and its arity (`fn:function-arity`) —
//! plus an `Any` view so that the evaluator that created it can get its own representation back:
//!
//! ```ignore
//! let concrete = handle.body().as_any().downcast_ref::<MyClosure>();
//! ```
//!
//! This keeps `oxixml-infoset` engine-agnostic: it never calls a function item, never knows what
//! calling one would mean, and has no dependency on any evaluator. An evaluator implements
//! [`FunctionBody`] for its own closure type and can always recover it, because it is the only
//! producer of the handles it consumes.
//!
//! # Identity
//!
//! XDM gives function items *reference* identity: two function items are the same item only if
//! they are the same item. [`XdmFunction`]'s [`PartialEq`] is therefore `Arc::ptr_eq`, and
//! `fn:deep-equal` raises `FOTY0015` rather than comparing them at all.

use oxixml_qname::ExpandedName;
use std::any::Any;
use std::fmt;
use std::sync::Arc;

/// What the data model knows about a function item, and the door back to the evaluator's own
/// representation of it.
///
/// An evaluator implements this for its closure type and wraps it in an [`XdmFunction`].
pub trait FunctionBody: fmt::Debug + Send + Sync {
    /// The function's name, or `None` for an anonymous one — `fn:function-name`.
    fn function_name(&self) -> Option<ExpandedName>;

    /// The number of arguments the function takes — `fn:function-arity`.
    fn arity(&self) -> usize;

    /// The body as [`Any`], so that the evaluator that created it can downcast back to its own
    /// type. Implementations return `self`.
    fn as_any(&self) -> &(dyn Any + Send + Sync);
}

/// A function item held by the data model: an opaque, reference-identified handle.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{FunctionBody, Item, XdmFunction};
/// use oxixml_qname::ExpandedName;
/// use std::any::Any;
/// use std::sync::Arc;
///
/// #[derive(Debug)]
/// struct Doubler;
///
/// impl FunctionBody for Doubler {
///     fn function_name(&self) -> Option<ExpandedName> {
///         Some(ExpandedName::new(None::<&str>, "double"))
///     }
///     fn arity(&self) -> usize {
///         1
///     }
///     fn as_any(&self) -> &(dyn Any + Send + Sync) {
///         self
///     }
/// }
///
/// let item = XdmFunction::new(Arc::new(Doubler));
/// assert_eq!(item.arity(), 1);
/// assert_eq!(item.name().map(|name| name.local_name().to_owned()), Some("double".to_owned()));
/// assert!(item.body().as_any().downcast_ref::<Doubler>().is_some());
///
/// // Identity is by reference: a second handle over a second `Doubler` is a different item.
/// assert_eq!(item, item.clone());
/// assert_ne!(item, XdmFunction::new(Arc::new(Doubler)));
///
/// // And it is an `Item`, so it can live in a sequence, a map or an array.
/// assert!(Item::from(item).is_function());
/// ```
#[derive(Clone)]
pub struct XdmFunction {
    body: Arc<dyn FunctionBody>,
}

impl XdmFunction {
    /// Wraps an evaluator's function body in a data-model handle.
    #[must_use]
    pub fn new(body: Arc<dyn FunctionBody>) -> Self {
        Self { body }
    }

    /// The function's name — `fn:function-name`.
    #[must_use]
    pub fn name(&self) -> Option<ExpandedName> {
        self.body.function_name()
    }

    /// The function's arity — `fn:function-arity`.
    #[must_use]
    pub fn arity(&self) -> usize {
        self.body.arity()
    }

    /// The underlying body, for an evaluator to downcast back to its own type.
    #[must_use]
    pub fn body(&self) -> &Arc<dyn FunctionBody> {
        &self.body
    }

    /// Whether the two handles are the *same* function item, which is the only identity XDM
    /// gives function items.
    #[must_use]
    pub fn is_same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.body, &other.body)
    }
}

impl PartialEq for XdmFunction {
    /// Reference identity — see [`XdmFunction::is_same`].
    fn eq(&self, other: &Self) -> bool {
        self.is_same(other)
    }
}

impl Eq for XdmFunction {}

impl fmt::Debug for XdmFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("XdmFunction")
            .field("name", &self.name())
            .field("arity", &self.arity())
            .finish()
    }
}

impl fmt::Display for XdmFunction {
    /// `name#arity` for a named function, `function(*)#arity` for an anonymous one.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.name() {
            Some(name) => write!(f, "{name}#{}", self.arity()),
            None => write!(f, "function(*)#{}", self.arity()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Anonymous(usize);

    impl FunctionBody for Anonymous {
        fn function_name(&self) -> Option<ExpandedName> {
            None
        }
        fn arity(&self) -> usize {
            self.0
        }
        fn as_any(&self) -> &(dyn Any + Send + Sync) {
            self
        }
    }

    #[test]
    fn identity_is_by_reference_not_by_value() {
        let body = Arc::new(Anonymous(2));
        let one = XdmFunction::new(body.clone());
        let same = XdmFunction::new(body);
        let other = XdmFunction::new(Arc::new(Anonymous(2)));
        assert_eq!(one, same);
        assert_ne!(one, other);
    }

    #[test]
    fn the_body_can_be_recovered_by_its_creator() {
        let function = XdmFunction::new(Arc::new(Anonymous(3)));
        let body = function
            .body()
            .as_any()
            .downcast_ref::<Anonymous>()
            .expect("the creator's own type");
        assert_eq!(body.0, 3);
    }

    #[test]
    fn display_distinguishes_named_from_anonymous() {
        assert_eq!(
            XdmFunction::new(Arc::new(Anonymous(1))).to_string(),
            "function(*)#1"
        );
    }
}
