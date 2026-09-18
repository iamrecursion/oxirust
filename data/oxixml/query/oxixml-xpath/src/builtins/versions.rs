//! When each built-in function entered the language.
//!
//! A processor may only offer the functions its version defines: `fn:format-number` is an XSLT
//! 2.0 function that F&O *3.0* added to XPath, `fn:sort` arrives with 3.1, and `fn:data#0`,
//! `fn:round#2` and `fn:string-join#1` are arities that did not exist in 2.0. Calling one of
//! them in an expression compiled at an earlier version is `XPST0017` — "no function of that
//! name and arity is in scope" — not a silent success, and QT3 pins the distinction with
//! version-tagged cases such as `K-FunctionCallExpr-15` (`format-number` under `XP20`),
//! `K-RoundFunc-2` (`round#2` under `XP20`) and `K-DataFunc-1` (`data#0` under `XP20`).
//!
//! Only the *additions* are listed: a name and arity that appears in neither table is treated as
//! having always been there (or as belonging to a host-supplied function library, which this
//! table has no business filtering). The XPath 1.0 library — a *subset* of 2.0's — is not
//! modelled here; 1.0 is served by its own evaluation path, which converts arguments rather
//! than rejecting them.

use crate::context::{ARRAY_NAMESPACE, MAP_NAMESPACE, MATH_NAMESPACE};
use crate::version::XPathVersion;
use oxixml_fn::FN_NAMESPACE;
use oxixml_infoset::ExpandedName;

/// A function name, and the arities of it, added by one version.
///
/// An empty arity list means "every arity of this name".
struct Addition {
    local: &'static str,
    arities: &'static [usize],
}

const fn all(local: &'static str) -> Addition {
    Addition {
        local,
        arities: &[],
    }
}

const fn arity(local: &'static str, arities: &'static [usize]) -> Addition {
    Addition { local, arities }
}

/// The `fn:` functions and arities [F&O 3.0](https://www.w3.org/TR/xpath-functions-30/) added.
const ADDED_IN_30: &[Addition] = &[
    all("analyze-string"),
    all("available-environment-variables"),
    all("element-with-id"),
    all("environment-variable"),
    all("filter"),
    all("fold-left"),
    all("fold-right"),
    all("for-each"),
    all("for-each-pair"),
    all("format-date"),
    all("format-dateTime"),
    all("format-integer"),
    all("format-number"),
    all("format-time"),
    all("function-arity"),
    all("function-lookup"),
    all("function-name"),
    all("generate-id"),
    all("has-children"),
    all("head"),
    all("innermost"),
    all("outermost"),
    all("parse-xml"),
    all("parse-xml-fragment"),
    all("path"),
    all("serialize"),
    all("tail"),
    all("unparsed-text"),
    all("unparsed-text-available"),
    all("unparsed-text-lines"),
    all("uri-collection"),
    // Arities the earlier signatures did not have.
    arity("data", &[0]),
    arity("document-uri", &[0]),
    arity("nilled", &[0]),
    arity("node-name", &[0]),
    arity("round", &[2]),
    arity("string-join", &[1]),
    arity("tokenize", &[1]),
];

/// The `fn:` functions and arities [F&O 3.1](https://www.w3.org/TR/xpath-functions-31/) added.
const ADDED_IN_31: &[Addition] = &[
    all("apply"),
    all("collation-key"),
    all("contains-token"),
    all("default-language"),
    all("json-doc"),
    all("json-to-xml"),
    all("load-xquery-module"),
    all("parse-ietf-date"),
    all("parse-json"),
    all("random-number-generator"),
    all("sort"),
    all("transform"),
    all("xml-to-json"),
    arity("trace", &[1]),
];

/// Whether a function of this name and arity exists in `version`.
///
/// Names outside the four namespaces this engine serves are left alone — a host may register a
/// function library of its own, and its availability is the host's business.
pub(crate) fn available_in(
    name: &ExpandedName,
    arity: usize,
    version: XPathVersion,
    xslt_functions: bool,
) -> bool {
    // XSLT 1.0 §12 defines `format-number` and `generate-id` itself, and an XSLT 1.0 stylesheet
    // compiles its expressions as XPath 1.0 — so in that host they are in scope however early
    // the XPath version is. The rest of XSLT's own library (`current`, `key`, `document`,
    // `system-property`, the availability tests) is in neither table below and is therefore
    // never filtered.
    if xslt_functions
        && name.namespace() == Some(FN_NAMESPACE)
        && XSLT_10_LIBRARY.contains(&name.local_name())
    {
        return true;
    }
    let introduced = match name.namespace() {
        // The whole `map:` and `array:` namespaces, and the `?` operator that reads them, are
        // new in 3.1; `math:` arrives with 3.0.
        Some(MAP_NAMESPACE | ARRAY_NAMESPACE) => XPathVersion::XPath31,
        Some(MATH_NAMESPACE) => XPathVersion::XPath30,
        Some(FN_NAMESPACE) => {
            if listed(ADDED_IN_31, name.local_name(), arity) {
                XPathVersion::XPath31
            } else if listed(ADDED_IN_30, name.local_name(), arity) {
                XPathVersion::XPath30
            } else {
                XPathVersion::XPath10
            }
        }
        _ => return true,
    };
    version.at_least(introduced)
}

/// The `fn:`-namespace functions [XSLT 1.0 §12](https://www.w3.org/TR/xslt-19991116#add-func)
/// supplies that F&O did not add to XPath until 3.0.
const XSLT_10_LIBRARY: &[&str] = &["format-number", "generate-id"];

fn listed(additions: &[Addition], local: &str, arity: usize) -> bool {
    additions.iter().any(|addition| {
        addition.local == local
            && (addition.arities.is_empty() || addition.arities.contains(&arity))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn name(namespace: &str, local: &str) -> ExpandedName {
        ExpandedName::new(Some(namespace), local)
    }

    #[test]
    fn a_function_is_unavailable_before_the_version_that_added_it() {
        let format_number = name(FN_NAMESPACE, "format-number");
        assert!(!available_in(
            &format_number,
            2,
            XPathVersion::XPath20,
            false
        ));
        assert!(available_in(
            &format_number,
            2,
            XPathVersion::XPath30,
            false
        ));
        assert!(available_in(
            &format_number,
            2,
            XPathVersion::XPath31,
            false
        ));

        let sort = name(FN_NAMESPACE, "sort");
        assert!(!available_in(&sort, 1, XPathVersion::XPath30, false));
        assert!(available_in(&sort, 1, XPathVersion::XPath31, false));
    }

    #[test]
    fn only_the_added_arities_are_gated() {
        let round = name(FN_NAMESPACE, "round");
        assert!(available_in(&round, 1, XPathVersion::XPath20, false));
        assert!(!available_in(&round, 2, XPathVersion::XPath20, false));
        assert!(available_in(&round, 2, XPathVersion::XPath30, false));

        let data = name(FN_NAMESPACE, "data");
        assert!(available_in(&data, 1, XPathVersion::XPath20, false));
        assert!(!available_in(&data, 0, XPathVersion::XPath20, false));
    }

    /// The table gates 3.0 and 3.1 additions only; the XPath 1.0 library is not modelled, so a
    /// long-standing function stays in scope there.
    #[test]
    fn nothing_older_than_the_tables_is_gated() {
        for version in XPathVersion::ALL {
            assert!(available_in(&name(FN_NAMESPACE, "not"), 1, *version, false));
            assert!(available_in(
                &name(FN_NAMESPACE, "count"),
                1,
                *version,
                false
            ));
        }
    }

    #[test]
    fn the_map_and_array_namespaces_are_31_only() {
        let size = name(MAP_NAMESPACE, "size");
        assert!(!available_in(&size, 1, XPathVersion::XPath30, false));
        assert!(available_in(&size, 1, XPathVersion::XPath31, false));
        assert!(available_in(
            &name("urn:example:host", "extension"),
            1,
            XPathVersion::XPath20,
            false
        ));
    }

    /// XSLT 1.0 §12 supplies `format-number` and `generate-id` itself, so a stylesheet that
    /// declares `version="1.0"` — and therefore compiles XPath 1.0 — may call them; plain XPath
    /// 1.0 and 2.0 may not.
    #[test]
    fn the_xslt_library_is_in_scope_only_for_an_xslt_host() {
        for local in ["format-number", "generate-id"] {
            let function = name(FN_NAMESPACE, local);
            assert!(!available_in(&function, 2, XPathVersion::XPath10, false));
            assert!(available_in(&function, 2, XPathVersion::XPath10, true));
            // `format-number#3` is exactly the arity a named `xsl:decimal-format` needs.
            assert!(available_in(&function, 3, XPathVersion::XPath10, true));
        }
        // The host flag widens nothing else: `fn:sort` is still 3.1-only.
        assert!(!available_in(
            &name(FN_NAMESPACE, "sort"),
            1,
            XPathVersion::XPath30,
            true
        ));
    }
}
