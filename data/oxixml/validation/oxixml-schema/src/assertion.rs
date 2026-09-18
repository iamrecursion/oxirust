//! Evaluating XSD 1.1 assertions and conditional type assignment.
//!
//! Three constructs of XSD 1.1 carry an XPath expression that has to be
//! *evaluated* — rather than merely matched, as the identity-constraint paths
//! of [`crate::xpath`] are:
//!
//! * `xs:assert` on a complex type
//!   ([Part 1 §3.4.4.2](https://www.w3.org/TR/xmlschema11-1/#cvc-assertion));
//! * the `assertion` facet on a simple type
//!   ([Part 2 §4.3.12](https://www.w3.org/TR/xmlschema11-2/#rf-assertion)),
//!   with `$value` bound to the value being checked; and
//! * `xs:alternative` — conditional type assignment
//!   ([Part 1 §3.3.2](https://www.w3.org/TR/xmlschema11-1/#dcl.elt.altType)).
//!
//! All three go through [`evaluate`], which compiles the expression as
//! **XPath 2.0** — the level the Recommendation names — against a static
//! context built from the namespace bindings in force where the expression was
//! written and from `xpathDefaultNamespace`, and evaluates it over the
//! fabricated XDM instance of [`crate::xdm`].
//!
//! A *dynamic* error is not propagated: §3.13.4.2 makes an assertion whose
//! evaluation raises an error simply unsatisfied, and the caller reports
//! `cvc-assertion`. A *static* error — an expression that does not parse — is
//! likewise treated as unsatisfied at assessment time; whether the schema
//! itself should have been rejected for it is [`check_static`]'s business.

use oxixml_infoset::{Sequence, Tree};
use oxixml_xpath::{StaticContext, XPath, XPathVersion};

use crate::components::Assertion;

/// The name `$value` is bound to for a facet assertion.
const VALUE: &str = "value";

/// Builds the static context an assertion is compiled against.
///
/// The namespace bindings are those in force on the `xs:assert`,
/// `xs:assertion` or `xs:alternative` element, and the default *element*
/// namespace is `xpathDefaultNamespace` — which, unlike an ordinary XPath
/// host, XSD keeps separate from the default namespace of the schema document
/// (§3.13.1).
fn static_context(assertion: &Assertion, with_value: bool) -> StaticContext {
    let mut context = StaticContext::new();
    for (prefix, namespace) in assertion.namespaces.bindings() {
        if namespace.is_empty() {
            continue;
        }
        // The default namespace of the schema document is *not* the default
        // element namespace of the expression: only `xpathDefaultNamespace`
        // sets that.
        if let Some(prefix) = prefix {
            context = context.with_namespace(prefix, namespace);
        }
    }
    if let Some(namespace) = &assertion.xpath_default_namespace {
        context = context.with_default_element_namespace(namespace.clone());
    }
    if with_value {
        context = context.with_variable(VALUE);
    }
    context
}

/// Whether `assertion` holds of `tree`'s document element.
///
/// `declare_value` puts `$value` in scope — §3.13.1 does so for the `assertion`
/// facet and for an `xs:assert` on a type with simple content, but never for an
/// `xs:alternative`, where a reference to `$value` is a static error. `value`
/// is the binding itself; an empty sequence when the element has no simple
/// value after all.
pub(crate) fn evaluate(
    assertion: &Assertion,
    tree: &Tree,
    value: Option<Sequence>,
    declare_value: bool,
) -> bool {
    let context = static_context(assertion, declare_value);
    let Ok(compiled) = XPath::compile(&assertion.test, XPathVersion::XPath20, context) else {
        return false;
    };
    let Some(item) = tree.document_element() else {
        return false;
    };
    let mut dynamic = compiled.context_with(item);
    if declare_value {
        dynamic = dynamic.with_variable(VALUE, value.unwrap_or_default());
    }
    compiled.evaluate_to_boolean_in(&dynamic).unwrap_or(false)
}

/// Whether the `assertion` **facet** of a simple type holds of `value`.
///
/// Datatypes §4.3.13.3 (*Assertions Valid*) evaluates a facet assertion quite
/// differently from an `xs:assert`: "1.2 There is **no context item** for the
/// evaluation of the XPath expression. … As a consequence the expression `.`,
/// or any implicit or explicit reference to the context item, will raise a
/// dynamic error, which will cause the assertion to be treated as false. …
/// 1.3 There is likewise **no value for the context size and the context
/// position**." Only `$value` — "the ·XDM representation· of V under T" — is
/// available (`saxonData/Assert/assert-simple008` … `assert-simple010`).
pub(crate) fn evaluate_facet(assertion: &Assertion, value: Sequence) -> bool {
    let context = static_context(assertion, true);
    let Ok(compiled) = XPath::compile(&assertion.test, XPathVersion::XPath20, context) else {
        return false;
    };
    let dynamic = compiled.context().with_variable(VALUE, value);
    compiled.evaluate_to_boolean_in(&dynamic).unwrap_or(false)
}

/// Whether `assertion`'s expression is a well-formed XPath 2.0 expression in
/// the static context it was written in.
///
/// This is the *schema representation* half of the rule: `src-assert` (and,
/// for a type alternative, `src-type-alternative`) makes a schema document
/// whose `test` does not parse — or whose prefixes do not resolve — an error
/// at assembly time, quite apart from what any instance says.
pub(crate) fn check_static(
    assertion: &Assertion,
    with_value: bool,
    code: &'static str,
) -> Result<(), crate::error::SchemaError> {
    let context = static_context(assertion, with_value);
    XPath::compile(&assertion.test, XPathVersion::XPath20, context)
        .map(|_| ())
        .map_err(|error| {
            crate::error::SchemaError::new(
                code,
                format!(
                    "`{}` is not an XPath 2.0 expression: {error}",
                    assertion.test
                ),
            )
        })
}
