//! The XPath 1.0 function library's *argument conversion* rules.
//!
//! [XPath 1.0 §3.2](https://www.w3.org/TR/xpath-10/#section-Function-Calls) says that "if the
//! argument is not of the expected type it is converted to the expected type as if by a call to
//! the `string`, `number` or `boolean` function". There is no error for passing a node-set of
//! twenty nodes where a string is expected: `string-length(//title)` measures the *first* title.
//!
//! [`oxixml_fn`] implements the XPath 2.0 library, where the same call is a type error, so the
//! 1.0 conversions must be applied before dispatch. This module holds the signature table that
//! drives them, together with the four functions — `string`, `number`, `boolean` and the naming
//! accessors — whose 1.0 behaviour on a node-set has no 2.0 equivalent at all.

use super::{optional, string};
use crate::context::DynamicContext;
use crate::error::XPathResult;
use crate::eval::{compare, effective_boolean_value};
use crate::value::XpathValue;
use oxixml_fn::nodes;
use oxixml_infoset::{AtomicValue, NodeRef};

/// The type an XPath 1.0 parameter expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expected {
    /// `string` — the argument goes through `string()`.
    Text,
    /// `number` — the argument goes through `number()`.
    Number,
    /// `boolean` — the argument goes through `boolean()`.
    Boolean,
    /// `object` or `node-set` — the argument is passed through unchanged.
    Any,
}

/// The parameter types of the XPath 1.0 function library, for the functions whose arguments are
/// not all `object`.
///
/// A signature shorter than the call's arity repeats its last entry, which is what makes
/// `concat`'s variadic string list work.
const SIGNATURES: &[(&str, &[Expected])] = &[
    ("concat", &[Expected::Text]),
    ("starts-with", &[Expected::Text, Expected::Text]),
    ("contains", &[Expected::Text, Expected::Text]),
    ("substring-before", &[Expected::Text, Expected::Text]),
    ("substring-after", &[Expected::Text, Expected::Text]),
    (
        "substring",
        &[Expected::Text, Expected::Number, Expected::Number],
    ),
    ("string-length", &[Expected::Text]),
    ("normalize-space", &[Expected::Text]),
    (
        "translate",
        &[Expected::Text, Expected::Text, Expected::Text],
    ),
    ("not", &[Expected::Boolean]),
    ("lang", &[Expected::Text]),
    ("floor", &[Expected::Number]),
    ("ceiling", &[Expected::Number]),
    ("round", &[Expected::Number]),
    ("count", &[Expected::Any]),
    ("sum", &[Expected::Any]),
    ("id", &[Expected::Any]),
];

/// The functions whose XPath 1.0 behaviour on a node-set differs from the 2.0 function of the
/// same name, and which are therefore implemented here rather than converted and delegated.
///
/// # Errors
///
/// Whatever the conversion raises; in practice these never fail.
pub(crate) fn call(
    context: &DynamicContext,
    local: &str,
    arguments: &[XpathValue],
) -> XPathResult<Option<XpathValue>> {
    let value = match (local, arguments.len()) {
        ("string", 1) => string(compare::string_10(&arguments[0])?),
        ("number", 1) => XpathValue::from(AtomicValue::double(compare::number_10(&arguments[0]))),
        ("boolean", 1) => XpathValue::from(AtomicValue::boolean(effective_boolean_value(
            &arguments[0],
        )?)),
        ("name", 1) => {
            optional(first_node(&arguments[0]).map(|node| AtomicValue::string(nodes::name(&node))))
        }
        ("local-name", 1) => optional(
            first_node(&arguments[0]).map(|node| AtomicValue::string(nodes::local_name(&node))),
        ),
        ("namespace-uri", 1) => optional(
            first_node(&arguments[0]).map(|node| AtomicValue::any_uri(nodes::namespace_uri(&node))),
        ),
        _ => return Ok(None),
    };
    let _ = context;
    // The naming accessors answer the empty string, not the empty sequence, for an empty
    // node-set.
    Ok(Some(if value.is_empty() {
        string(String::new())
    } else {
        value
    }))
}

/// Converts `arguments` to the types the XPath 1.0 signature of `local` expects, or `None` when
/// the function's arguments need no conversion.
///
/// # Errors
///
/// Whatever `string()` raises on a function item.
pub(crate) fn coerce(
    local: &str,
    arguments: &[XpathValue],
) -> XPathResult<Option<Vec<XpathValue>>> {
    let Some((_, signature)) = SIGNATURES.iter().find(|(name, _)| *name == local) else {
        return Ok(None);
    };
    let mut converted = Vec::with_capacity(arguments.len());
    for (index, argument) in arguments.iter().enumerate() {
        let expected = signature
            .get(index)
            .or_else(|| signature.last())
            .copied()
            .unwrap_or(Expected::Any);
        converted.push(match expected {
            Expected::Any => argument.clone(),
            Expected::Text => string(compare::string_10(argument)?),
            Expected::Number => XpathValue::from(AtomicValue::double(compare::number_10(argument))),
            Expected::Boolean => {
                XpathValue::from(AtomicValue::boolean(effective_boolean_value(argument)?))
            }
        });
    }
    Ok(Some(converted))
}

fn first_node(value: &XpathValue) -> Option<NodeRef> {
    value.nodes().first().cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::StaticContext;
    use crate::version::XPathVersion;
    use oxixml_infoset::{Tree, XdmNode};
    use std::sync::Arc;

    fn context() -> DynamicContext {
        DynamicContext::new(Arc::new(StaticContext::new())).with_version(XPathVersion::XPath10)
    }

    #[test]
    fn a_node_set_converts_to_its_first_node() -> XPathResult<()> {
        let tree = Tree::parse_str("<r><a>first</a><a>second</a></r>")?;
        let root = tree.document_element().expect("a document element");
        let nodes = XpathValue::from_iter(root.children());
        let context = context();

        assert_eq!(
            call(&context, "string", std::slice::from_ref(&nodes))?
                .expect("handled")
                .to_string(),
            "first"
        );
        assert_eq!(
            call(&context, "local-name", std::slice::from_ref(&nodes))?
                .expect("handled")
                .to_string(),
            "a"
        );
        let converted = coerce("string-length", &[nodes])?.expect("a signature");
        assert_eq!(converted[0].to_string(), "first");
        Ok(())
    }

    #[test]
    fn concat_repeats_its_last_parameter_type() -> XPathResult<()> {
        let arguments = vec![
            XpathValue::from(AtomicValue::integer(1_i64)),
            XpathValue::from(AtomicValue::integer(2_i64)),
            XpathValue::from(AtomicValue::integer(3_i64)),
        ];
        let converted = coerce("concat", &arguments)?.expect("a signature");
        assert_eq!(converted.len(), 3);
        assert!(converted.iter().all(|value| {
            value
                .first()
                .and_then(|item| item.as_atomic())
                .is_some_and(|atomic| atomic.as_str().is_some())
        }));
        Ok(())
    }

    #[test]
    fn functions_with_no_conversions_are_left_alone() -> XPathResult<()> {
        assert!(coerce("upper-case", &[])?.is_none());
        assert!(call(&context(), "upper-case", &[])?.is_none());
        Ok(())
    }
}
