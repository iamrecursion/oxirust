//! The XSLT namespace and the vocabulary of element names in it.

use oxixml_infoset::{NodeRef, XdmNode};

/// The XSLT namespace, `http://www.w3.org/1999/XSL/Transform`.
pub const XSLT_NAMESPACE: &str = "http://www.w3.org/1999/XSL/Transform";

/// The key an expression's XPath static context carries the *default namespace declaration* in
/// scope where it was written under.
///
/// [XSLT 3.0 §5.1](https://www.w3.org/TR/xslt-30/#unprefixed-qnames) expands the argument of
/// `element-available` with "the default namespace of the defining element" — the `xmlns="…"` in
/// scope — which is a different thing from `[xsl:]xpath-default-namespace`, the *default
/// namespace for elements and types* that the XPath static context already holds. The two must
/// travel separately, so the compiler binds the first under this key, which is deliberately not
/// an `NCName`: no prefix a lexical `QName` can name collides with it.
pub(crate) const DEFAULT_NAMESPACE_KEY: &str = "#default";

/// Every element name XSLT 1.0, 2.0 and 3.0 define, as a static table so that the compiler can
/// dispatch on a `&'static str`.
const XSLT_ELEMENTS: &[&str] = &[
    "accept",
    "accumulator",
    "accumulator-rule",
    "analyze-string",
    "apply-imports",
    "apply-templates",
    "assert",
    "attribute",
    "attribute-set",
    "break",
    "call-template",
    "catch",
    "character-map",
    "choose",
    "comment",
    "context-item",
    "copy",
    "copy-of",
    "decimal-format",
    "document",
    "element",
    "evaluate",
    "expose",
    "fallback",
    "for-each",
    "for-each-group",
    "fork",
    "function",
    "global-context-item",
    "if",
    "import",
    "import-schema",
    "include",
    "iterate",
    "key",
    "map",
    "map-entry",
    "matching-substring",
    "merge",
    "merge-action",
    "merge-key",
    "merge-source",
    "message",
    "mode",
    "namespace",
    "namespace-alias",
    "next-iteration",
    "next-match",
    "non-matching-substring",
    "number",
    "on-completion",
    "on-empty",
    "on-non-empty",
    "otherwise",
    "output",
    "output-character",
    "override",
    "package",
    "param",
    "perform-sort",
    "preserve-space",
    "processing-instruction",
    "result-document",
    "sequence",
    "sort",
    "source-document",
    "strip-space",
    "stylesheet",
    "template",
    "text",
    "transform",
    "try",
    "use-package",
    "value-of",
    "variable",
    "when",
    "where-populated",
    "with-param",
];

/// Interns an XSLT element's local name, returning `None` for a name XSLT does not define.
///
/// # Examples
///
/// ```
/// use oxixml_xslt::names::xslt_element;
///
/// assert_eq!(xslt_element("template"), Some("template"));
/// assert_eq!(xslt_element("nonesuch"), None);
/// ```
#[must_use]
pub fn xslt_element(local: &str) -> Option<&'static str> {
    XSLT_ELEMENTS
        .iter()
        .copied()
        .find(|candidate| *candidate == local)
}

/// Whether the node's name is in the XSLT namespace.
#[must_use]
pub fn is_xslt(node: &NodeRef) -> bool {
    node.node_name()
        .is_some_and(|name| name.namespace() == Some(XSLT_NAMESPACE))
}

/// Whether `uri` is one of the namespaces XSLT reserves, in which a stylesheet may not declare
/// a name — [XSLT 3.0 §3.1](https://www.w3.org/TR/xslt-30/#reserved-namespaces).
///
/// # Examples
///
/// ```
/// use oxixml_xslt::names::is_reserved_namespace;
///
/// assert!(is_reserved_namespace("http://www.w3.org/2001/XMLSchema"));
/// assert!(!is_reserved_namespace("http://example.com/"));
/// ```
#[must_use]
pub fn is_reserved_namespace(uri: &str) -> bool {
    matches!(
        uri,
        XSLT_NAMESPACE
            | "http://www.w3.org/XML/1998/namespace"
            | "http://www.w3.org/2001/XMLSchema"
            | "http://www.w3.org/2001/XMLSchema-instance"
            | "http://www.w3.org/2005/xpath-functions"
            | "http://www.w3.org/2005/xpath-functions/math"
            | "http://www.w3.org/2005/xpath-functions/map"
            | "http://www.w3.org/2005/xpath-functions/array"
            | "http://www.w3.org/2005/xqt-errors"
    )
}

/// Whether `local` names an instruction this processor implements — the answer
/// `element-available()` gives.
#[must_use]
pub fn is_available_instruction(local: &str) -> bool {
    matches!(
        local,
        "analyze-string"
            | "apply-imports"
            | "apply-templates"
            | "assert"
            | "attribute"
            | "break"
            | "call-template"
            | "catch"
            | "choose"
            | "comment"
            | "copy"
            | "copy-of"
            | "document"
            | "element"
            | "fallback"
            | "for-each"
            | "for-each-group"
            | "if"
            | "iterate"
            | "map"
            | "map-entry"
            | "message"
            | "namespace"
            | "next-iteration"
            | "next-match"
            | "number"
            | "on-empty"
            | "on-non-empty"
            | "perform-sort"
            | "processing-instruction"
            | "result-document"
            | "sequence"
            | "text"
            | "try"
            | "value-of"
            | "variable"
            | "where-populated"
    )
}

/// Whether `value` is in the lexical space of `xs:language`.
///
/// [XSLT 3.0 §12.3](https://www.w3.org/TR/xslt-30/#number) and §13.1 both take a `lang`
/// attribute of that type, and a value outside it is `XTDE0030` when the attribute is a value
/// template and `XTSE0020` when it is written literally.
#[must_use]
pub(crate) fn is_language(value: &str) -> bool {
    let mut parts = value.split('-');
    let Some(first) = parts.next() else {
        return false;
    };
    if first.is_empty() || first.len() > 8 || !first.chars().all(|c| c.is_ascii_alphabetic()) {
        return false;
    }
    parts.all(|part| {
        !part.is_empty() && part.len() <= 8 && part.chars().all(|c| c.is_ascii_alphanumeric())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::Tree;

    #[test]
    fn the_vocabulary_is_interned() {
        assert_eq!(xslt_element("for-each-group"), Some("for-each-group"));
        assert_eq!(xslt_element("Template"), None);
    }

    #[test]
    fn xslt_elements_are_recognized_by_namespace() -> Result<(), oxixml_infoset::XdmError> {
        let tree =
            Tree::parse_str(r#"<xsl:template xmlns:xsl="http://www.w3.org/1999/XSL/Transform"/>"#)?;
        let element = tree.document_element().expect("an element");
        assert!(is_xslt(&element));

        let other = Tree::parse_str("<template/>")?;
        assert!(!is_xslt(&other.document_element().expect("an element")));
        Ok(())
    }

    #[test]
    fn availability_matches_what_is_implemented() {
        assert!(is_available_instruction("for-each"));
        assert!(!is_available_instruction("merge"));
    }
}
