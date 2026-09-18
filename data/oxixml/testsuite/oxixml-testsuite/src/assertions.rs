//! Shared XDM assertion checker for the QT3 ([`crate::qt3`]) and XSLT 3.0 ([`crate::xslt30`])
//! catalogues.
//!
//! Both catalogues describe an expected outcome with (mostly) the same small vocabulary of
//! `<result>` assertion elements — `assert`, `assert-eq`, `assert-count`, `assert-deep-eq`,
//! `assert-type`, `assert-true`, `assert-false`, `assert-empty`, `assert-string-value`, `error`,
//! `all-of`, `any-of`, `not` — evaluated with `$result` bound to the outcome. Rather than
//! re-implementing every comparison in Rust, most of them are turned into a tiny XPath 3.1
//! boolean expression over `$result` and evaluated with a throwaway [`XPath::compile`]; only
//! `error` (a direct code comparison) and `assert-string-value` (whose W3C-suite semantics —
//! concatenating every item's string value with a space — has no one-line XPath spelling) are
//! handled directly in Rust.
//!
//! `assert-xml` is checked by serializing the outcome and comparing it with the expected
//! fragment (leniently, so an equivalent namespace or whitespace spelling still matches), and
//! `serialization-matches` by serializing it with the parameters the case declared and matching
//! the regular expression the assertion carries — see [`check_with_serialization`], which a
//! caller that can reproduce the case's serialization uses in place of [`check`].
//! Assertion kinds this checker does not implement — `assert-serialization`,
//! `assert-serialization-error`, `assert-permutation`, and anything a catalogue driver does not
//! recognize — decide [`Verdict::Skip`] rather than guessing; each catalogue driver is free to
//! special-case them before falling back to [`check`] for the ones that are simple XDM
//! comparisons.

use oxixml_fn::xmlio::{SerializationParameters, serialize};
use oxixml_infoset::{NodeKind, NodeRef, Sequence, Tree, XdmNode, deep_equal};
use oxixml_xpath::{ERR_NAMESPACE, StaticContext, XPath, XPathVersion, XpathValue};

use crate::xdm_util::{attribute, element_children, local_name, text_of};

/// The error half of a normalized case outcome.
#[derive(Debug, Clone)]
pub struct ErrorInfo {
    /// The text an `error/@code` is compared against: a bare local name when the error is in the
    /// catalogue's standard error namespace, or the `Q{namespace}local` form otherwise.
    pub code: String,
    /// A human-readable message, used only in a failure report.
    pub message: String,
}

impl ErrorInfo {
    /// Builds an [`ErrorInfo`] from a bare local name (the shape every QT3 error code is written
    /// in) and a message.
    pub fn bare(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }

    /// Builds an [`ErrorInfo`] for an error raised under an arbitrary qualified name, writing
    /// [`ErrorInfo::code`] in the spelling both catalogues use for an `error/@code`: a bare local
    /// name inside the standard error namespace, and the `Q{namespace}local` `EQName` outside it
    /// — including `Q{}local` for an error in no namespace, which `fn:error` can raise.
    ///
    /// Every error an engine raises comes through here rather than through [`ErrorInfo::bare`]:
    /// `fn:error` takes an arbitrary `xs:QName`, so a raised code's *namespace* is as much a part
    /// of its identity as its local name (`prod-TryCatchExpr/try-catch-fn-error-1` raises
    /// `Q{http://www.example.com/}EXER3141`, whose local name alone would match nothing the
    /// catalogue wrote and, worse, would match a standard code that happened to spell the same).
    pub fn qualified(namespace: Option<&str>, local: &str, message: impl Into<String>) -> Self {
        let code = match namespace {
            Some(ERR_NAMESPACE) => local.to_owned(),
            Some(namespace) => format!("Q{{{namespace}}}{local}"),
            None => format!("Q{{}}{local}"),
        };
        Self::bare(code, message)
    }
}

/// The verdict one `<result>` assertion tree reaches against a case outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// The assertion held.
    Pass,
    /// The assertion did not hold, with a description of the mismatch.
    Fail(String),
    /// This checker cannot decide the assertion — an assertion kind it does not implement, or an
    /// internal error while compiling/evaluating the small helper expression a comparison needed
    /// (which reflects a gap in this harness, not a verdict on the engine under test).
    Skip,
}

/// What an assertion is decided against besides the outcome itself: the focus an XPath-shaped
/// assertion is evaluated with, and the serialization a `serialization-matches` compares.
///
/// Both are absent by default, and an assertion that needs one it was not given decides
/// [`Verdict::Skip`] — the checker was not handed what the assertion is about.
#[derive(Debug, Clone, Copy, Default)]
struct Evidence<'a> {
    /// The focus for an XPath-shaped assertion; see [`check_with_context_item`].
    context_item: Option<&'a NodeRef>,
    /// The parameters the outcome is serialized with; see [`check_with_serialization`].
    serialization: Option<&'a SerializationParameters>,
}

/// Checks `assertion` against `outcome`, recursing through `all-of`/`any-of`/`not`, with no
/// context item — the QT3 (FOTS) catalogue's own contract: `catalog-schema.xsd` documents
/// `<assert>` as "an XPath expression whose effective boolean value must be true; *usually the
/// expression will use the variable `$result`*", and never mentions a focus.
#[must_use]
pub fn check(assertion: &NodeRef, outcome: &Result<XpathValue, ErrorInfo>) -> Verdict {
    check_against(assertion, outcome, Evidence::default())
}

/// The same check as [`check`], but evaluating every XPath-shaped assertion with `context_item`
/// as the focus.
///
/// The XSLT 3.0 catalogue needs this and the QT3 one must not have it: `xslt30-test`'s
/// `admin/catalog-schema.xsd` says of its own `<assert>` that "the XPath expression will be
/// evaluated with the document node of the result document as the context item", which is what
/// lets a case write `count(//td) = 64` or `/out = "…"` instead of naming `$result`.
#[must_use]
pub fn check_with_context_item(
    assertion: &NodeRef,
    outcome: &Result<XpathValue, ErrorInfo>,
    context_item: Option<&NodeRef>,
) -> Verdict {
    check_against(
        assertion,
        outcome,
        Evidence {
            context_item,
            serialization: None,
        },
    )
}

/// The same check as [`check`], with `serialization-matches` decided against the outcome
/// serialized with `serialization`.
///
/// `None` says the harness could not reproduce the serialization the case asks for — an output
/// method or parameter [`oxixml_fn::xmlio`] does not provide — and every `serialization-matches`
/// then decides [`Verdict::Skip`], which is what this checker does for any assertion it cannot
/// reach a verdict on. The QT3 drivers read those parameters out of the case's own query prolog;
/// see the `qt3::serialization` module for how.
#[must_use]
pub fn check_with_serialization(
    assertion: &NodeRef,
    outcome: &Result<XpathValue, ErrorInfo>,
    serialization: Option<&SerializationParameters>,
) -> Verdict {
    check_against(
        assertion,
        outcome,
        Evidence {
            context_item: None,
            serialization,
        },
    )
}

/// The recursion the three public entry points share.
fn check_against(
    assertion: &NodeRef,
    outcome: &Result<XpathValue, ErrorInfo>,
    evidence: Evidence<'_>,
) -> Verdict {
    let context_item = evidence.context_item;
    match local_name(assertion).as_str() {
        "all-of" => {
            let mut saw_skip = false;
            for child in element_children(assertion) {
                match check_against(&child, outcome, evidence) {
                    Verdict::Pass => {}
                    Verdict::Fail(reason) => return Verdict::Fail(reason),
                    Verdict::Skip => saw_skip = true,
                }
            }
            if saw_skip {
                Verdict::Skip
            } else {
                Verdict::Pass
            }
        }
        "any-of" => {
            let mut last_failure = None;
            let mut decided = false;
            for child in element_children(assertion) {
                match check_against(&child, outcome, evidence) {
                    Verdict::Pass => return Verdict::Pass,
                    Verdict::Fail(reason) => {
                        decided = true;
                        last_failure = Some(reason);
                    }
                    Verdict::Skip => {}
                }
            }
            if decided {
                Verdict::Fail(last_failure.unwrap_or_else(|| "no alternative matched".to_owned()))
            } else {
                Verdict::Skip
            }
        }
        "not" => match element_children(assertion).into_iter().next() {
            Some(child) => match check_against(&child, outcome, evidence) {
                Verdict::Pass => Verdict::Fail("the negated assertion held".to_owned()),
                Verdict::Fail(_) => Verdict::Pass,
                Verdict::Skip => Verdict::Skip,
            },
            None => Verdict::Skip,
        },
        "error" => check_error(assertion, outcome),
        "assert-empty" => boolean_expression(
            "empty($result)",
            outcome,
            "the empty sequence",
            context_item,
        ),
        "assert-true" => boolean_expression("boolean($result)", outcome, "true", context_item),
        "assert-false" => {
            boolean_expression("not(boolean($result))", outcome, "false", context_item)
        }
        "assert-count" => {
            let body = text_of(assertion);
            boolean_expression(
                &format!("count($result) = ({body})"),
                outcome,
                &format!("{} item(s)", body.trim()),
                context_item,
            )
        }
        "assert-eq" => {
            let body = text_of(assertion);
            boolean_expression(&format!("$result = ({body})"), outcome, &body, context_item)
        }
        "assert-deep-eq" => {
            let body = text_of(assertion);
            boolean_expression(
                &format!("deep-equal($result, ({body}))"),
                outcome,
                &body,
                context_item,
            )
        }
        "assert-type" => {
            let body = text_of(assertion);
            boolean_expression(
                &format!("$result instance of {}", body.trim()),
                outcome,
                &format!("an instance of {}", body.trim()),
                context_item,
            )
        }
        // `assert-permutation` holds when the result contains the same *items* as the expected
        // sequence, in any order — the assertion the catalogue writes for an expression whose
        // order the specification leaves to the processor (`fn:collection`, `fn:distinct-values`,
        // map and node-set operations). Equality is `fn:deep-equal`'s, which is what makes the
        // comparison work for nodes as well as atomic values, and which the catalogue relies on
        // to treat `NaN` as equal to itself.
        //
        // Two sequences of equal length whose every item occurs in both the same number of times
        // are permutations of one another, so the multiplicity comparison below is exact — a
        // pairwise "every item appears somewhere" test would wrongly accept `(1, 1, 2)` against
        // `(1, 2, 2)`.
        "assert-permutation" => {
            let body = text_of(assertion);
            boolean_expression(
                &format!(
                    "let $expected := ({body}) return count($result) = count($expected) \
                     and (every $item in $result satisfies \
                     count($result[deep-equal(., $item)]) = count($expected[deep-equal(., $item)]))"
                ),
                outcome,
                &format!("a permutation of {body}"),
                context_item,
            )
        }
        "assert" => {
            let body = text_of(assertion);
            boolean_expression(&body, outcome, &format!("`{body}` to hold"), context_item)
        }
        "assert-string-value" => check_string_value(assertion, outcome),
        "assert-xml" => check_xml(assertion, outcome),
        "serialization-matches" => {
            check_serialization_matches(assertion, outcome, evidence.serialization)
        }
        "assert-serialization-error" => {
            check_serialization_error(assertion, outcome, evidence.serialization)
        }
        _ => Verdict::Skip,
    }
}

/// Checks an `assert-serialization-error` assertion: serializing the outcome with the parameters
/// the case declared must raise the error the assertion names.
///
/// The fault such a case describes — an output declaration the data model cannot honour, a
/// character the chosen encoding cannot represent — is one a processor may legitimately detect
/// *before* serialization begins, while compiling the declaration that causes it. An outcome that
/// is already an error is therefore graded by its code exactly as `<error>` is; only a value that
/// serializes without complaint is a failure.
fn check_serialization_error(
    assertion: &NodeRef,
    outcome: &Result<XpathValue, ErrorInfo>,
    serialization: Option<&SerializationParameters>,
) -> Verdict {
    let value = match outcome {
        Err(_) => return check_error(assertion, outcome),
        Ok(value) => value,
    };
    // Without the case's own serialization parameters the assertion is about a serialization this
    // harness cannot reproduce, so it says nothing about the serializer either way.
    let Some(parameters) = serialization else {
        return Verdict::Skip;
    };
    let Ok(sequence) = value.clone().into_sequence() else {
        return Verdict::Skip; // a function item, which the data model cannot serialize
    };
    match serialize(&sequence, parameters) {
        Err(error) => {
            let raised = ErrorInfo::qualified(
                error.name().namespace(),
                error.name().local_name(),
                error.to_string(),
            );
            check_error(assertion, &Err(raised))
        }
        Ok(serialized) => Verdict::Fail(format!(
            "expected the serialization error {}, but the result serialized to `{serialized}`",
            attribute(assertion, "code").unwrap_or_else(|| "*".to_owned())
        )),
    }
}

/// Checks a `serialization-matches` assertion: the outcome, serialized with the parameters the
/// case's query declared, must match the regular expression the assertion carries.
///
/// `catalog-schema.xsd` fixes three things about the comparison, and all three are what
/// `fn:matches` would do: the pattern is an XPath regular expression, the `flags` attribute is
/// passed to it verbatim, and the match is **not** anchored unless the pattern anchors itself.
///
/// A pattern this engine's regular-expression compiler rejects decides [`Verdict::Skip`] rather
/// than a failure — an assertion the harness could not read says nothing about the serializer —
/// and so does a `serialization` the caller could not reproduce. An outcome that cannot be
/// serialized at all *is* a failure: the case asserts that serializing it produces particular
/// text.
fn check_serialization_matches(
    assertion: &NodeRef,
    outcome: &Result<XpathValue, ErrorInfo>,
    serialization: Option<&SerializationParameters>,
) -> Verdict {
    let Some(parameters) = serialization else {
        return Verdict::Skip;
    };
    let pattern = text_of(assertion);
    let flags = attribute(assertion, "flags").unwrap_or_default();
    let Ok(regex) = oxixml_regex::Regex::xpath(&pattern, &flags) else {
        return Verdict::Skip;
    };
    let value = match outcome {
        Ok(value) => value,
        Err(error) => {
            return Verdict::Fail(format!(
                "expected a serialization matching `{pattern}`, raised {} ({})",
                error.code, error.message
            ));
        }
    };
    let Ok(sequence) = value.clone().into_sequence() else {
        return Verdict::Skip; // a function item, which the data model cannot serialize
    };
    match serialize(&sequence, parameters) {
        Ok(serialized) if regex.is_match(&serialized) => Verdict::Pass,
        Ok(serialized) => Verdict::Fail(format!("`{serialized}` does not match `{pattern}`")),
        Err(error) => Verdict::Fail(format!(
            "expected a serialization matching `{pattern}`, but serializing raised {error}"
        )),
    }
}

/// Checks an `assert-xml` assertion: the outcome, serialized as XML, must equal the expected
/// fragment. The comparison is lenient about the ways two equal fragments can be spelled — an
/// exact text match, a `fn:deep-equal` of the two parsed fragments, or a `fn:deep-equal` of the
/// expected fragment against the very nodes the query built — so that a namespace-prefix or
/// whitespace difference introduced by the serializer does not read as a conformance failure.
///
/// An `assert-xml` that points at an external `file` is not reproduced here (there is no result
/// tree to load it against), so it decides [`Verdict::Skip`].
fn check_xml(assertion: &NodeRef, outcome: &Result<XpathValue, ErrorInfo>) -> Verdict {
    if attribute(assertion, "file").is_some() {
        return Verdict::Skip;
    }
    let expected = text_of(assertion);
    match outcome {
        Ok(value) => {
            let Ok(sequence) = value.clone().into_sequence() else {
                return Verdict::Skip;
            };
            let Ok(actual) = serialize(&sequence, &SerializationParameters::default()) else {
                return Verdict::Skip;
            };
            if actual == expected
                || same_xml(&actual, &expected)
                || same_nodes(&sequence, &expected)
            {
                Verdict::Pass
            } else {
                Verdict::Fail(format!("expected XML `{expected}`, got `{actual}`"))
            }
        }
        Err(error) => Verdict::Fail(format!(
            "expected XML `{expected}`, raised {} ({})",
            error.code, error.message
        )),
    }
}

/// Whether two serialized fragments are the same XML, ignoring the incidental formatting the
/// catalogue happens to have wrapped its expected value in. Both are parsed inside a throwaway
/// wrapper element (an `assert-xml` value is a *fragment*, so it may have several top-level
/// nodes) and compared with `fn:deep-equal`.
fn same_xml(actual: &str, expected: &str) -> bool {
    match (parse_fragment(actual), parse_fragment(expected)) {
        (Some(left), Some(right)) => deep_equal(&left, &right).unwrap_or(false),
        _ => false,
    }
}

/// Whether the result sequence *is* the expected XML, compared node by node rather than through
/// the serializer. This is the second chance for a fragment whose serialized spelling differs
/// from the catalogue's only in how a namespace is written: the expected value is parsed and
/// `fn:deep-equal`-compared against the nodes the query actually built.
fn same_nodes(actual: &Sequence, expected: &str) -> bool {
    match parse_fragment(expected) {
        Some(wanted) => deep_equal(actual, &wanted).unwrap_or(false),
        None => false,
    }
}

/// Parses an XML fragment (zero or more top-level nodes) into the sequence of its top-level
/// nodes, by wrapping it in a synthetic element first, and drops the fragment's *boundary*
/// whitespace — see [`strip_boundary_whitespace`].
fn parse_fragment(text: &str) -> Option<Sequence> {
    Tree::parse_str(&format!(
        "<oxixml-fragment-wrapper>{text}</oxixml-fragment-wrapper>"
    ))
    .ok()
    .and_then(|tree| tree.document_element())
    .map(|root| Sequence::from_iter(strip_boundary_whitespace(root.children())))
}

/// Drops a leading and a trailing whitespace-only text node from a fragment's top-level nodes.
///
/// An `assert-xml` value is written as the content of an element in an indented catalogue file,
/// so the newline and indentation that separate `<assert-xml>` from `]]>` are part of the
/// fragment's text — `fn/json-to-xml.xml`'s `json-to-xml-039` expects `<string …>oh dear
/// DEAD</string>` followed by a newline and nine spaces, and `app/Walmsley.xml`'s `d1e74610`
/// ends the same way. That trailing text is the catalogue's own layout, not a node the query is
/// expected to have produced, and no processor could produce it.
///
/// Only the two boundary nodes are dropped, and only when the fragment contains markup at all:
/// whitespace *between* two elements is a text node the query either built or did not, so it
/// still has to compare exactly (`d1e74610`'s expected value is a run of `</product><product>`
/// with no whitespace between them, and the result really has none), and a fragment that is
/// nothing but whitespace is a value in its own right rather than layout around one.
fn strip_boundary_whitespace(mut nodes: Vec<NodeRef>) -> Vec<NodeRef> {
    if !nodes
        .iter()
        .any(|node| node.node_kind() == NodeKind::Element)
    {
        return nodes;
    }
    if nodes.last().is_some_and(is_whitespace_text) {
        nodes.pop();
    }
    if nodes.first().is_some_and(is_whitespace_text) {
        nodes.remove(0);
    }
    nodes
}

/// Whether `node` is a text node whose content is entirely XML whitespace.
fn is_whitespace_text(node: &NodeRef) -> bool {
    node.node_kind() == NodeKind::Text && node.string_value().trim().is_empty()
}

/// Decides an assertion that expects an error, comparing `assertion`'s `code` attribute with the
/// code actually raised (see [`same_error_code`]); `code="*"`, or no `code` at all, accepts any.
///
/// Shared with [`crate::xslt30`], whose `assert-serialization-error` carries the same `code`
/// attribute and has to grade it by the same rule as `<error>` — the difference between the two is
/// *when* the error must be raised, not how its code is spelled.
pub(crate) fn check_error(assertion: &NodeRef, outcome: &Result<XpathValue, ErrorInfo>) -> Verdict {
    let expected = attribute(assertion, "code").unwrap_or_else(|| "*".to_owned());
    match outcome {
        Err(error) => {
            if expected == "*" || same_error_code(&expected, &error.code, assertion) {
                Verdict::Pass
            } else {
                Verdict::Fail(format!(
                    "expected error {expected}, raised {} ({})",
                    error.code, error.message
                ))
            }
        }
        Ok(value) => Verdict::Fail(format!("expected error {expected}, produced `{value}`")),
    }
}

/// Whether the catalogue's `error/@code` names the error that was actually raised.
///
/// Both are `EQName`s, not strings: the catalogue writes a standard code as a bare local name
/// (`FOAR0001`), everything else as `Q{namespace}local` (`Q{http://www.example.com/}EXER3141`,
/// `Q{}USER9999`) and — as the production the schema points at allows — may write a lexical
/// `QName` whose prefix is bound on the assertion element itself. Comparing the two spellings
/// literally would both miss a match and, for a code raised in a namespace of its own, accept
/// one that is not there.
fn same_error_code(expected: &str, raised: &str, scope: &NodeRef) -> bool {
    expected == raised
        || expanded_error_code(expected, Some(scope)) == expanded_error_code(raised, None)
}

/// The `(namespace, local name)` an error-code spelling denotes. A `Q{…}` `EQName` carries its
/// own namespace (an empty one meaning *no* namespace); a lexical `QName` resolves its prefix
/// against `scope`'s in-scope namespaces when there is a scope to resolve against; anything else
/// is a bare code in the standard error namespace, which is how both catalogues write one.
fn expanded_error_code(code: &str, scope: Option<&NodeRef>) -> (Option<String>, String) {
    if let Some(rest) = code.strip_prefix("Q{")
        && let Some((namespace, local)) = rest.split_once('}')
    {
        let namespace = (!namespace.is_empty()).then(|| namespace.to_owned());
        return (namespace, local.to_owned());
    }
    if let Some((prefix, local)) = code.split_once(':')
        && let Some(namespace) = scope.and_then(|node| namespace_for_prefix(node, prefix))
    {
        return (Some(namespace), local.to_owned());
    }
    (Some(ERR_NAMESPACE.to_owned()), code.to_owned())
}

/// The namespace `prefix` is bound to among `node`'s in-scope namespaces.
fn namespace_for_prefix(node: &NodeRef, prefix: &str) -> Option<String> {
    node.namespaces().into_iter().find_map(|namespace| {
        let bound = namespace.node_name()?;
        (bound.local_name() == prefix).then(|| namespace.string_value())
    })
}

fn check_string_value(assertion: &NodeRef, outcome: &Result<XpathValue, ErrorInfo>) -> Verdict {
    let normalize = attribute(assertion, "normalize-space").as_deref() == Some("true");
    let mut expected = text_of(assertion);
    match outcome {
        Ok(value) => {
            let Some(mut actual) = joined_string_value(value) else {
                return Verdict::Skip;
            };
            if normalize {
                actual = collapse(&actual);
                expected = collapse(&expected);
            }
            if actual == expected {
                Verdict::Pass
            } else {
                Verdict::Fail(format!("expected `{expected}`, got `{actual}`"))
            }
        }
        Err(error) => Verdict::Fail(format!(
            "expected `{expected}`, raised {} ({})",
            error.code, error.message
        )),
    }
}

/// The string value the W3C test suites mean by "the string value of `$result`": every item's
/// own string value, joined with a single space. For a one-item sequence (the common case, and
/// always the shape [`crate::xslt30`] passes) this is exactly that item's string value.
///
/// `None` when an item has no string value to give (e.g. a function item).
#[must_use]
pub fn joined_string_value(value: &XpathValue) -> Option<String> {
    let mut joined = String::new();
    for (index, item) in value.iter().enumerate() {
        let text = item.string_value().ok()?;
        if index > 0 {
            joined.push(' ');
        }
        joined.push_str(&text);
    }
    Some(joined)
}

/// Collapses runs of whitespace to a single space and trims the ends — `fn:normalize-space`.
#[must_use]
pub fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Evaluates `expression` (which may reference `$result`) as an XPath 3.1 boolean, with `$result`
/// bound to `outcome`'s value and `context_item`, when there is one, as the focus.
///
/// A `Skip` distinguishes an internal harness problem (the expression this driver built does not
/// even compile, or evaluating it raised a dynamic error) from a genuine `Fail`, matching how
/// `oxixml-xslt`'s own conformance harness treats the same situation for its XPath-shaped
/// assertions.
fn boolean_expression(
    expression: &str,
    outcome: &Result<XpathValue, ErrorInfo>,
    expected_description: &str,
    context_item: Option<&NodeRef>,
) -> Verdict {
    let value = match outcome {
        Ok(value) => value,
        Err(error) => {
            return Verdict::Fail(format!(
                "expected {expected_description}, raised {} ({})",
                error.code, error.message
            ));
        }
    };
    let context = StaticContext::new().with_variable("result");
    let Ok(compiled) = XPath::compile(expression, XPathVersion::XPath31, context) else {
        return Verdict::Skip;
    };
    let mut dynamic = compiled.context().with_variable("result", value.clone());
    if let Some(node) = context_item {
        dynamic = dynamic.with_context_item(node.clone());
    }
    match compiled.evaluate_to_boolean_in(&dynamic) {
        Ok(true) => Verdict::Pass,
        Ok(false) => Verdict::Fail(format!("expected {expected_description}, got `{value}`")),
        Err(_) => Verdict::Skip,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_fn::xmlio::OutputMethod;
    use oxixml_infoset::{AtomicValue, Sequence, Tree};

    fn assertion(xml: &str) -> Tree {
        Tree::parse_str(xml).expect("well-formed fixture")
    }

    fn ok(text: &str) -> Result<XpathValue, ErrorInfo> {
        Ok(XpathValue::from(Sequence::from(AtomicValue::string(text))))
    }

    #[test]
    fn assert_eq_passes_and_fails() {
        let tree = assertion(r#"<assert-eq>"hi"</assert-eq>"#);
        let root = tree.document_element().expect("a document element");
        assert_eq!(check(&root, &ok("hi")), Verdict::Pass);
        assert!(matches!(check(&root, &ok("bye")), Verdict::Fail(_)));
    }

    /// `assert-permutation` is a multiset comparison, not an order-insensitive membership test:
    /// a reordering holds, a differing multiplicity does not, and neither does a differing
    /// length.
    #[test]
    fn assert_permutation_compares_multisets() {
        fn integers(values: &[i64]) -> Result<XpathValue, ErrorInfo> {
            Ok(XpathValue::from(Sequence::from_iter(
                values.iter().copied().map(AtomicValue::integer),
            )))
        }
        let tree = assertion(r"<assert-permutation>1, 2, 2, 3</assert-permutation>");
        let root = tree.document_element().expect("a document element");
        assert_eq!(check(&root, &integers(&[2, 3, 1, 2])), Verdict::Pass);
        assert_eq!(check(&root, &integers(&[1, 2, 2, 3])), Verdict::Pass);
        assert!(
            matches!(check(&root, &integers(&[1, 1, 2, 3])), Verdict::Fail(_)),
            "an item occurring a different number of times is not a permutation"
        );
        assert!(
            matches!(check(&root, &integers(&[1, 2, 3])), Verdict::Fail(_)),
            "a shorter sequence is not a permutation"
        );
        assert!(
            matches!(check(&root, &integers(&[1, 2, 2, 3, 3])), Verdict::Fail(_)),
            "a longer sequence is not a permutation"
        );
    }

    #[test]
    fn error_assertion_compares_the_code() {
        let tree = assertion(r#"<error code="FOAR0001"/>"#);
        let root = tree.document_element().expect("a document element");
        let matching = Err(ErrorInfo::bare("FOAR0001", "division by zero"));
        let other = Err(ErrorInfo::bare("XPTY0004", "type error"));
        assert_eq!(check(&root, &matching), Verdict::Pass);
        assert!(matches!(check(&root, &other), Verdict::Fail(_)));
        assert!(matches!(
            check(&root, &ok("not an error")),
            Verdict::Fail(_)
        ));
    }

    #[test]
    fn wildcard_error_code_matches_anything() {
        let tree = assertion(r#"<error code="*"/>"#);
        let root = tree.document_element().expect("a document element");
        assert_eq!(
            check(&root, &Err(ErrorInfo::bare("ANYTHING", "x"))),
            Verdict::Pass
        );
    }

    /// `serialization-matches` is an unanchored `fn:matches` over the serialized result, with the
    /// assertion's `flags` passed through — `catalog-schema.xsd` says so in as many words.
    #[test]
    fn serialization_matches_compares_the_serialized_result() {
        let defaults = SerializationParameters::default();
        let tree = assertion("<serialization-matches>^HI$</serialization-matches>");
        let root = tree.document_element().expect("a document element");
        assert!(matches!(
            check_with_serialization(&root, &ok("HI"), Some(&defaults)),
            Verdict::Pass
        ));
        assert!(matches!(
            check_with_serialization(&root, &ok("hi"), Some(&defaults)),
            Verdict::Fail(_)
        ));

        let tree = assertion(r#"<serialization-matches flags="i">^hi$</serialization-matches>"#);
        let root = tree.document_element().expect("a document element");
        assert_eq!(
            check_with_serialization(&root, &ok("HI"), Some(&defaults)),
            Verdict::Pass
        );

        // Unanchored, so a pattern naming part of the output is enough.
        let tree = assertion("<serialization-matches>b</serialization-matches>");
        let root = tree.document_element().expect("a document element");
        assert_eq!(
            check_with_serialization(&root, &ok("abc"), Some(&defaults)),
            Verdict::Pass
        );
    }

    /// The serialization a case declares decides what the pattern is matched against.
    /// `ser/method-adaptive.xml`'s `Serialization-adaptive-84` is the case this matters most
    /// for: it serializes `xs:anyURI('"')` under `method="adaptive"`, which Serialization 3.1
    /// §11 writes in the constructor syntax — quoted, with the quotation mark of its own doubled
    /// — while the default `xml` method writes the bare string value.
    #[test]
    fn the_declared_output_method_decides_what_is_matched() {
        let value = Ok(XpathValue::from(Sequence::from(AtomicValue::any_uri("\""))));
        let tree = assertion(r#"<serialization-matches>^""""$</serialization-matches>"#);
        let root = tree.document_element().expect("a document element");
        let adaptive = SerializationParameters::default().with_method(OutputMethod::Adaptive);
        assert_eq!(
            check_with_serialization(&root, &value, Some(&adaptive)),
            Verdict::Pass
        );
        assert!(matches!(
            check_with_serialization(&root, &value, Some(&SerializationParameters::default())),
            Verdict::Fail(_)
        ));
    }

    /// Without a serialization to decide against — an XSLT-catalogue caller, or a query whose
    /// output declarations this harness cannot reproduce — the assertion is undecided, and an
    /// `any-of` around it keeps looking for an alternative that *is* decidable.
    #[test]
    fn serialization_matches_is_undecided_without_parameters() {
        let tree = assertion("<serialization-matches>^HI$</serialization-matches>");
        let root = tree.document_element().expect("a document element");
        assert_eq!(check(&root, &ok("HI")), Verdict::Skip);
        assert_eq!(
            check_with_serialization(&root, &ok("HI"), None),
            Verdict::Skip
        );

        let tree = assertion(
            "<any-of><serialization-matches>^HI$</serialization-matches><assert-eq>\"HI\"</assert-eq></any-of>",
        );
        let root = tree.document_element().expect("a document element");
        assert_eq!(check(&root, &ok("HI")), Verdict::Pass);
    }

    /// A pattern the regular-expression compiler rejects is a gap in the harness, not a verdict
    /// on the serializer; an error outcome, on the other hand, is a genuine failure of an
    /// assertion that asked for serialized text.
    #[test]
    fn an_unreadable_pattern_skips_and_a_raised_error_fails() {
        let defaults = SerializationParameters::default();
        let tree = assertion("<serialization-matches>[unclosed</serialization-matches>");
        let root = tree.document_element().expect("a document element");
        assert_eq!(
            check_with_serialization(&root, &ok("x"), Some(&defaults)),
            Verdict::Skip
        );

        let tree = assertion("<serialization-matches>^x$</serialization-matches>");
        let root = tree.document_element().expect("a document element");
        let raised = Err(ErrorInfo::bare("FORG0001", "invalid value"));
        assert!(matches!(
            check_with_serialization(&root, &raised, Some(&defaults)),
            Verdict::Fail(_)
        ));
    }

    #[test]
    fn all_of_requires_every_child_to_pass() {
        let tree = assertion(r#"<all-of><assert-eq>"hi"</assert-eq><assert-true/></all-of>"#);
        let root = tree.document_element().expect("a document element");
        assert_eq!(check(&root, &ok("hi")), Verdict::Pass);
    }

    #[test]
    fn not_inverts_its_child() {
        let tree = assertion(r#"<not><assert-eq>"nope"</assert-eq></not>"#);
        let root = tree.document_element().expect("a document element");
        assert_eq!(check(&root, &ok("hi")), Verdict::Pass);
    }

    /// `assert-serialization` is the one kind the catalogues use that this shared checker does
    /// not decide: it compares the *serialized* result as text against an expected document, and
    /// only [`crate::xslt30`] — which has the transformation's own serialization and the test-set
    /// directory to resolve the expectation against — can reach it.
    #[test]
    fn unrecognised_assertion_kinds_skip() {
        for fixture in [
            r"<assert-serialization>&lt;a/&gt;</assert-serialization>",
            r"<assert-nothing-of-the-sort/>",
        ] {
            let tree = assertion(fixture);
            let root = tree.document_element().expect("a document element");
            assert_eq!(check(&root, &ok("hi")), Verdict::Skip, "{fixture}");
        }
    }

    /// The nodes an XML fragment's markup denotes, as a result sequence — the shape a query
    /// that built those nodes would hand the checker.
    fn nodes_of(fragment: &str) -> (Tree, Result<XpathValue, ErrorInfo>) {
        let tree = Tree::parse_str(&format!("<wrapper>{fragment}</wrapper>"))
            .expect("well-formed fixture");
        let root = tree.document_element().expect("a document element");
        let value = Ok(XpathValue::from(Sequence::from_iter(root.children())));
        (tree, value)
    }

    /// `fn-json-to-xml/json-to-xml-039`, `app-Walmsley/d1e74610` and their siblings: the
    /// catalogue writes the expected fragment inside an indented element, so it ends with the
    /// newline and indentation before `]]>`. No processor produces that, and it is not part of
    /// what the case asserts.
    #[test]
    fn assert_xml_ignores_the_catalogues_trailing_layout_whitespace() {
        let (_tree, value) = nodes_of(r#"<a b="1"/>"#);
        let tree = assertion("<assert-xml>&lt;a b=\"1\"/&gt;\n         </assert-xml>");
        let node = tree.document_element().expect("a document element");
        assert_eq!(check(&node, &value), Verdict::Pass);
    }

    /// The narrowness of that leniency: whitespace *between* two elements is a text node the
    /// query either built or did not, and it still has to be there.
    #[test]
    fn assert_xml_still_compares_whitespace_between_elements() {
        let (_tree, spaced) = nodes_of("<a/> <b/>");
        let tree = assertion("<assert-xml>&lt;a/&gt;&lt;b/&gt;</assert-xml>");
        let node = tree.document_element().expect("a document element");
        assert!(
            matches!(check(&node, &spaced), Verdict::Fail(_)),
            "a whitespace text node between two elements is significant"
        );

        let (_tree, joined) = nodes_of("<a/><b/>");
        let tree = assertion("<assert-xml>&lt;a/&gt; &lt;b/&gt;</assert-xml>");
        let node = tree.document_element().expect("a document element");
        assert!(
            matches!(check(&node, &joined), Verdict::Fail(_)),
            "and so is its absence"
        );
    }

    /// Whitespace inside an element's own text is content, never layout.
    #[test]
    fn assert_xml_still_compares_element_text_exactly() {
        let (_tree, value) = nodes_of("<a>x</a>");
        let tree = assertion("<assert-xml>&lt;a&gt; x &lt;/a&gt;</assert-xml>");
        let node = tree.document_element().expect("a document element");
        assert!(matches!(check(&node, &value), Verdict::Fail(_)));
    }

    /// `app-Walmsley/d1e74610` compares mixed content (`Our <i>favorite</i> shirt!`), where the
    /// spaces around the nested element are the very thing being asserted.
    #[test]
    fn assert_xml_still_compares_mixed_content_exactly() {
        let (_tree, value) = nodes_of("<a>Our<i>favorite</i>shirt!</a>");
        let tree = assertion(
            "<assert-xml>&lt;a&gt;Our &lt;i&gt;favorite&lt;/i&gt; shirt!&lt;/a&gt;</assert-xml>",
        );
        let node = tree.document_element().expect("a document element");
        assert!(matches!(check(&node, &value), Verdict::Fail(_)));
    }

    /// The guard on the boundary rule: layout is only layout when there is markup for it to sit
    /// around. A fragment that is nothing but whitespace is a value in its own right.
    #[test]
    fn parse_fragment_strips_boundary_whitespace_only_around_markup() {
        let plain = parse_fragment("<a/>\n         ").expect("well-formed fragment");
        assert_eq!(plain.len(), 1, "the trailing layout text is dropped");

        let leading = parse_fragment("\n   <a/>").expect("well-formed fragment");
        assert_eq!(leading.len(), 1, "and so is the leading one");

        let whitespace = parse_fragment("   ").expect("well-formed fragment");
        assert_eq!(
            whitespace.len(),
            1,
            "a fragment with no markup keeps its text"
        );

        let inner = parse_fragment("<a/> <b/>").expect("well-formed fragment");
        assert_eq!(inner.len(), 3, "only the boundary nodes are dropped");
    }

    /// `prod-TryCatchExpr/try-catch-fn-error-1`: `fn:error` raises a code in a namespace of its
    /// own, and the catalogue writes it as an `EQName`. Comparing local names alone would both
    /// miss this match and accept a standard code that happened to spell the same.
    #[test]
    fn error_assertion_compares_expanded_names() {
        let tree = assertion(r#"<error code="Q{http://www.example.com/}EXER3141"/>"#);
        let root = tree.document_element().expect("a document element");
        let raised = Err(ErrorInfo::qualified(
            Some("http://www.example.com/"),
            "EXER3141",
            "raised by fn:error",
        ));
        assert_eq!(check(&root, &raised), Verdict::Pass);

        let standard = Err(ErrorInfo::bare(
            "EXER3141",
            "same local name, standard code",
        ));
        assert!(
            matches!(check(&root, &standard), Verdict::Fail(_)),
            "the namespace is part of the code's identity"
        );
    }

    /// `Q{}local` — the spelling the catalogue uses for an error in no namespace at all.
    #[test]
    fn error_assertion_matches_a_code_in_no_namespace() {
        let tree = assertion(r#"<error code="Q{}USER9999"/>"#);
        let root = tree.document_element().expect("a document element");
        let raised = Err(ErrorInfo::qualified(None, "USER9999", "no namespace"));
        assert_eq!(check(&root, &raised), Verdict::Pass);
        assert!(matches!(
            check(
                &root,
                &Err(ErrorInfo::bare("USER9999", "standard namespace"))
            ),
            Verdict::Fail(_)
        ));
    }

    /// An `EQName` may also be a lexical `QName`, resolved against the assertion element's own
    /// in-scope namespaces.
    #[test]
    fn error_assertion_resolves_a_prefixed_code() {
        let tree = assertion(
            r#"<error xmlns:err="http://www.w3.org/2005/xqt-errors" code="err:FOAR0001"/>"#,
        );
        let root = tree.document_element().expect("a document element");
        assert_eq!(
            check(&root, &Err(ErrorInfo::bare("FOAR0001", "division by zero"))),
            Verdict::Pass
        );
    }

    #[test]
    fn qualified_writes_a_standard_code_bare_and_everything_else_as_an_eqname() {
        assert_eq!(
            ErrorInfo::qualified(Some(ERR_NAMESPACE), "FOAR0001", "x").code,
            "FOAR0001"
        );
        assert_eq!(
            ErrorInfo::qualified(Some("urn:example"), "OOPS", "x").code,
            "Q{urn:example}OOPS"
        );
        assert_eq!(ErrorInfo::qualified(None, "OOPS", "x").code, "Q{}OOPS");
    }

    /// The XSLT 3.0 half of `<assert>`: `xslt30-test/admin/catalog-schema.xsd` requires the
    /// result document node as the context item, which is what lets `function-0701` write
    /// `count(//td) = 64`.
    #[test]
    fn an_assert_can_use_the_result_document_as_its_context_item() {
        let result = Tree::parse_str("<out><td>1</td><td>2</td></out>").expect("well-formed");
        let root = result.root();
        let value = Ok(XpathValue::from(Sequence::from(root.clone())));
        let tree = assertion("<assert>count(//td) = 2</assert>");
        let node = tree.document_element().expect("a document element");
        assert_eq!(
            check_with_context_item(&node, &value, Some(&root)),
            Verdict::Pass
        );

        let tree = assertion(r#"<assert>/out = "12"</assert>"#);
        let node = tree.document_element().expect("a document element");
        assert_eq!(
            check_with_context_item(&node, &value, Some(&root)),
            Verdict::Pass
        );
    }

    /// And the QT3 half, which must stay as it was: the FOTS catalogue documents `<assert>` as
    /// an expression over `$result` and never supplies a focus, so an absolute path has nothing
    /// to start from — a harness problem for that assertion, not a verdict on the engine.
    #[test]
    fn the_qt3_path_binds_no_context_item() {
        let result = Tree::parse_str("<out><td>1</td></out>").expect("well-formed");
        let value = Ok(XpathValue::from(Sequence::from(result.root())));
        let tree = assertion("<assert>count(//td) = 1</assert>");
        let node = tree.document_element().expect("a document element");
        assert_eq!(check(&node, &value), Verdict::Skip);

        let tree = assertion("<assert>count($result//td) = 1</assert>");
        let node = tree.document_element().expect("a document element");
        assert_eq!(check(&node, &value), Verdict::Pass);
    }

    #[test]
    fn assert_xml_compares_the_serialized_result() {
        use oxixml_infoset::Tree as InfosetTree;

        // A single-element result whose serialization matches the expected fragment passes,
        // and one whose serialization differs fails rather than skipping.
        let element = InfosetTree::parse_str("<a b=\"1\"/>").expect("well-formed");
        let root = element.document_element().expect("a document element");
        let value = Ok(XpathValue::from(Sequence::from_iter([root])));

        let tree = assertion(r#"<assert-xml>&lt;a b="1"/&gt;</assert-xml>"#);
        let node = tree.document_element().expect("a document element");
        assert_eq!(check(&node, &value), Verdict::Pass);

        let tree = assertion(r#"<assert-xml>&lt;a b="2"/&gt;</assert-xml>"#);
        let node = tree.document_element().expect("a document element");
        assert!(matches!(check(&node, &value), Verdict::Fail(_)));
    }

    /// `assert-serialization-error` is decided by *serializing* the result: a value the data
    /// model refuses to serialize (a bare attribute node, Serialization 3.1 §2's `SENR0001`)
    /// raises the code the case names, and one that serializes cleanly is a failure. An error
    /// raised before serialization was reached is graded by its code exactly as `<error>` is.
    #[test]
    fn assert_serialization_error_is_decided_by_serializing_the_result() {
        let parameters = SerializationParameters::default();
        let tree = assertion(r#"<assert-serialization-error code="SENR0001"/>"#);
        let node = tree.document_element().expect("a document element");

        let document = Tree::parse_str(r#"<a b="1"/>"#).expect("well-formed");
        let element = document.document_element().expect("a document element");
        let attributes = element.attributes();
        let [attribute_node] = attributes.as_slice() else {
            panic!("the fixture element has exactly one attribute");
        };
        let unserializable = Ok(XpathValue::from(Sequence::from_iter([
            attribute_node.clone()
        ])));
        assert_eq!(
            check_with_serialization(&node, &unserializable, Some(&parameters)),
            Verdict::Pass
        );

        // A result that serializes without complaint is the failure this assertion describes.
        let clean = Ok(XpathValue::from(Sequence::from_iter([element])));
        assert!(matches!(
            check_with_serialization(&node, &clean, Some(&parameters)),
            Verdict::Fail(_)
        ));

        // The same code, raised earlier — while the output declaration was being read — still
        // answers the assertion.
        let raised: Result<XpathValue, ErrorInfo> = Err(ErrorInfo::bare("SENR0001", "no"));
        assert_eq!(
            check_with_serialization(&node, &raised, Some(&parameters)),
            Verdict::Pass
        );

        // Without the case's own parameters there is no serialization to judge.
        assert_eq!(
            check_with_serialization(&node, &clean_value(), None),
            Verdict::Skip
        );
    }

    /// A serializable result, for the branches of the test above that do not care what it is.
    fn clean_value() -> Result<XpathValue, ErrorInfo> {
        ok("text")
    }
}
