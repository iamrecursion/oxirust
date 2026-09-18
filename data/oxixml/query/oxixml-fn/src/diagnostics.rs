//! `fn:error` and `fn:trace`, the two functions of
//! [F&O 3.1 §3](https://www.w3.org/TR/xpath-functions-31/#errors-and-diagnostics) that exist to
//! influence a query's *side effects* rather than its value.

use crate::context::FnContext;
use crate::error::{FnError, FnErrorCode};
use oxixml_infoset::{ExpandedName, Item, QNameValue, Sequence};

/// [`fn:error`](https://www.w3.org/TR/xpath-functions-31/#func-error): builds the dynamic error
/// the function raises.
///
/// The zero-argument form raises `err:FOER0000` with no description and no error object; every
/// argument that is supplied replaces one of those three components.
///
/// This function *returns* the error rather than raising it, because `fn:error`'s declared
/// return type is `none`: a caller turns it into a `Result::Err`.
///
/// # Examples
///
/// ```
/// use oxixml_fn::{diagnostics::error, FnErrorCode};
/// use oxixml_infoset::{AtomicValue, ExpandedName, QNameValue, Sequence};
///
/// // The default is FOER0000.
/// assert_eq!(error(None, None, None).code(), Some(FnErrorCode::Foer0000));
///
/// // A caller-supplied name and description come through unchanged.
/// let name = QNameValue::from(ExpandedName::new(Some("urn:example"), "broken"));
/// let raised = error(Some(&name), Some("it broke"), Some(Sequence::from(AtomicValue::integer(7_i64))));
/// assert_eq!(raised.name().local_name(), "broken");
/// assert_eq!(raised.description(), "it broke");
/// assert_eq!(raised.error_object().map(Sequence::len), Some(1));
/// ```
#[must_use]
pub fn error(
    name: Option<&QNameValue>,
    description: Option<&str>,
    error_object: Option<Sequence>,
) -> FnError {
    // The prefix the caller spelt the name with is carried over: `$err:code` is an `xs:QName`,
    // and `fn:prefix-from-QName($err:code)` has to report the prefix `fn:error` was given
    // (QT3 `try-catch-fn-error-8`). It takes no part in the name's identity, so the `catch`
    // clause still matches on `(namespace, local)` alone.
    let expanded = name.map_or_else(
        || FnErrorCode::Foer0000.expanded_name(),
        |name| {
            let expanded = ExpandedName::new(name.namespace(), name.local_name());
            match name.prefix() {
                Some(prefix) => expanded.with_prefix(prefix),
                None => expanded,
            }
        },
    );
    let error = FnError::with_name(expanded, description.unwrap_or_default());
    match error_object {
        Some(object) => error.with_error_object(object),
        None => error,
    }
}

/// [`fn:trace`](https://www.w3.org/TR/xpath-functions-31/#func-trace): reports the value to the
/// context's [`TraceSink`](crate::TraceSink) and returns it unchanged.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use oxixml_fn::{diagnostics::trace, CollectingTrace, FnContext};
/// use oxixml_infoset::{AtomicValue, Item, Sequence};
///
/// let sink = Arc::new(CollectingTrace::new());
/// let context = FnContext::new().with_trace_sink(sink.clone());
///
/// let value = Sequence::from_items([
///     Item::from(AtomicValue::integer(1_i64)),
///     Item::from(AtomicValue::integer(2_i64)),
/// ]);
/// assert_eq!(trace(&context, &value, "xs: "), value);
/// assert_eq!(sink.messages(), vec!["xs: [1]: 1".to_owned(), "xs: [2]: 2".to_owned()]);
/// ```
#[must_use]
pub fn trace(context: &FnContext, value: &Sequence, label: &str) -> Sequence {
    let total = value.len();
    for (index, item) in value.iter().enumerate() {
        let rendered = item.string_value().unwrap_or_else(|_| item.to_string());
        context
            .trace_sink()
            .trace(label, &rendered, index + 1, total);
    }
    value.clone()
}

/// The one-argument form of [`trace`], which uses an empty label.
#[must_use]
pub fn trace_unlabelled(context: &FnContext, value: &Sequence) -> Sequence {
    trace(context, value, "")
}

/// Renders one item the way `fn:trace` does, which is `fn:string` where that is defined and a
/// placeholder for a function item.
#[must_use]
pub fn trace_rendering(item: &Item) -> String {
    item.string_value().unwrap_or_else(|_| item.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CollectingTrace;
    use oxixml_infoset::AtomicValue;
    use std::sync::Arc;

    #[test]
    fn the_default_error_is_foer0000() {
        let raised = error(None, None, None);
        assert_eq!(raised.code(), Some(FnErrorCode::Foer0000));
        assert_eq!(raised.description(), "");
        assert!(raised.error_object().is_none());
    }

    #[test]
    fn a_supplied_name_is_preserved() {
        let name = QNameValue::from(ExpandedName::new(Some("urn:example"), "broken"));
        let raised = error(Some(&name), Some("it broke"), None);
        assert_eq!(raised.name().namespace(), Some("urn:example"));
        assert_eq!(raised.name().local_name(), "broken");
        assert_eq!(raised.description(), "it broke");
    }

    /// The prefix the caller spelt the error name with survives, so that a `catch` clause's
    /// `fn:prefix-from-QName($err:code)` can report it (QT3 `try-catch-fn-error-8`). It is not
    /// part of the name's identity, so matching is unaffected.
    #[test]
    fn a_supplied_name_keeps_its_prefix() {
        let name = QNameValue::new(Some("example"), Some("http://www.example.com/"), "EXER3141");
        let raised = error(Some(&name), Some("Description"), None);
        assert_eq!(raised.name().prefix(), Some("example"));
        assert_eq!(raised.name().namespace(), Some("http://www.example.com/"));
        assert_eq!(raised.name().local_name(), "EXER3141");
        assert_eq!(
            raised.name(),
            &ExpandedName::new(Some("http://www.example.com/"), "EXER3141"),
            "the prefix takes no part in equality, so `catch` still matches"
        );
    }

    #[test]
    fn an_error_code_in_the_error_namespace_is_recognized() {
        let name = QNameValue::from(FnErrorCode::Forg0001.expanded_name());
        assert!(error(Some(&name), None, None).is(FnErrorCode::Forg0001));
    }

    #[test]
    fn trace_is_the_identity_on_values() {
        let sink = Arc::new(CollectingTrace::new());
        let context = FnContext::new().with_trace_sink(sink.clone());
        let value = Sequence::from(AtomicValue::string("hello"));
        assert_eq!(trace(&context, &value, "label: "), value);
        assert_eq!(sink.messages(), vec!["label: hello".to_owned()]);

        sink.clear();
        assert_eq!(
            trace_unlabelled(&context, &Sequence::empty()),
            Sequence::empty()
        );
        assert!(sink.messages().is_empty());
    }
}
