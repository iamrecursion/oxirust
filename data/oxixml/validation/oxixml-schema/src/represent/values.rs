//! The *attribute value* grammars of the schema for schemas.
//!
//! [XML Schema Part 1 §3.2](https://www.w3.org/TR/xmlschema-1/#normative-schemaSchema)
//! gives every attribute of the schema vocabulary a datatype: `id` is
//! `xs:ID`, `name` is `xs:NCName`, `type` is `xs:QName`, `schemaLocation` is
//! `xs:anyURI`, `minOccurs` is `xs:nonNegativeInteger`, and so on. A schema
//! document whose attribute values are not in those lexical spaces is not a
//! schema document at all, so the values are checked here — before any
//! component is built — and the builders may then read them without
//! re-checking.

use oxixml_qname::is_ncname;

use crate::names::XsdVersion;
use crate::value::{collapse, is_any_uri};

/// The datatype the schema for schemas gives one attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AttrType {
    /// `xs:ID` — an `NCName`, unique across the schema document.
    Id,
    /// `xs:NCName`.
    NcName,
    /// `xs:QName` (checked lexically; the prefix is bound later).
    QName,
    /// A whitespace-separated list of `xs:QName`, as `memberTypes` is.
    QNameList,
    /// `xs:anyURI`.
    AnyUri,
    /// `xs:token` — anything, once collapsed.
    Token,
    /// `xs:boolean`.
    Boolean,
    /// `xs:nonNegativeInteger`.
    NonNegativeInteger,
    /// `xs:positiveInteger`.
    PositiveInteger,
    /// `unbounded` or `xs:nonNegativeInteger`.
    MaxOccurs,
    /// The `namespaceList` of `xs:any`/`xs:anyAttribute`.
    NamespaceList,
    /// The 1.1 `notNamespace` list: no `##any`/`##other`, at least one item.
    NotNamespaceList,
    /// The 1.1 `notQName` list: `##defined`, `##definedSibling` or a `QName`.
    NotQNameList,
    /// A `#all`-or-list derivation set over the given tokens.
    DerivationSet(&'static [&'static str]),
    /// One of a fixed set of tokens.
    Enum(&'static [&'static str]),
    /// An XPath expression — unconstrained here; the subset is checked when
    /// the selector or field is compiled.
    XPath,
    /// `xs:string` — every value is legal, including the empty one.
    Text,
}

/// Checks one attribute value against its declared type.
///
/// Returns the reason the value is not in the lexical space, or [`None`] when
/// it is.
pub(super) fn check(kind: AttrType, raw: &str, version: XsdVersion) -> Option<String> {
    let token = collapse(raw);
    match kind {
        AttrType::Id | AttrType::NcName => {
            if is_ncname(&token) {
                None
            } else {
                Some(format!("`{raw}` is not an NCName"))
            }
        }
        AttrType::QName => (!is_qname(&token)).then(|| format!("`{raw}` is not a QName")),
        AttrType::QNameList => token
            .split(' ')
            .filter(|item| !item.is_empty())
            .find(|item| !is_qname(item))
            .map(|item| format!("`{item}` is not a QName")),
        AttrType::AnyUri => {
            if is_schema_uri(&token, version) {
                None
            } else {
                Some(format!("`{raw}` is not a URI reference"))
            }
        }
        AttrType::Token | AttrType::Text | AttrType::XPath => None,
        AttrType::Boolean => match token.as_str() {
            "true" | "false" | "0" | "1" => None,
            _ => Some(format!("`{raw}` is not a boolean")),
        },
        AttrType::NonNegativeInteger => is_unsigned(&token, false),
        AttrType::PositiveInteger => is_unsigned(&token, true),
        AttrType::MaxOccurs => {
            if token == "unbounded" {
                None
            } else {
                is_unsigned(&token, false)
            }
        }
        AttrType::NamespaceList => {
            if token == "##any" || token == "##other" {
                return None;
            }
            namespace_items(&token)
        }
        AttrType::NotNamespaceList => {
            if token.is_empty() {
                return Some("`notNamespace` needs at least one namespace".to_owned());
            }
            namespace_items(&token)
        }
        AttrType::NotQNameList => token
            .split(' ')
            .filter(|item| !item.is_empty())
            .find(|item| !matches!(*item, "##defined" | "##definedSibling") && !is_qname(item))
            .map(|item| format!("`{item}` is not a QName")),
        AttrType::DerivationSet(allowed) => {
            if token == "#all" {
                return None;
            }
            token
                .split(' ')
                .filter(|item| !item.is_empty())
                .find(|item| !allowed.contains(item))
                .map(|item| format!("`{item}` is not allowed here"))
        }
        AttrType::Enum(allowed) => {
            let _ = version;
            if allowed.contains(&token.as_str()) {
                None
            } else {
                Some(format!("`{raw}` is not one of {}", allowed.join(", ")))
            }
        }
    }
}

/// `xs:QName`: an optionally prefixed `NCName`.
#[must_use]
pub(super) fn is_qname(value: &str) -> bool {
    match value.split_once(':') {
        Some((prefix, local)) => is_ncname(prefix) && is_ncname(local),
        None => is_ncname(value),
    }
}

fn is_unsigned(token: &str, positive: bool) -> Option<String> {
    let digits = token.strip_prefix('+').unwrap_or(token);
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Some(format!("`{token}` is not a non-negative integer"));
    }
    if positive && digits.bytes().all(|byte| byte == b'0') {
        return Some(format!("`{token}` is not a positive integer"));
    }
    None
}

/// Checks the items of a `namespaceList`, which
/// [XML Schema Part 1 §3.10.2](https://www.w3.org/TR/xmlschema11-1/#nsl)
/// writes as `(##any | ##other) | List of (xs:anyURI | ##targetNamespace | ##local)`.
///
/// An item is told from the two `##`-keywords by *being a URI reference*, so
/// this check keeps the URI-reference reading in **both** versions, unlike
/// [`is_schema_uri`]. Reading it as the widened XSD 1.1 `anyURI` — every string
/// — would make a mistyped or misplaced keyword into a namespace name, and the
/// suite says it is not: `namespace="##anyAttribute"`
/// (`msData/wildcards/wildK002`, `wildK006`, `wildN001`) and
/// `namespace="##target"` (`wildC035`, `wildC036`) are `<expected
/// validity="invalid"/>` with no version qualifier at all.
fn namespace_items(token: &str) -> Option<String> {
    token
        .split(' ')
        .filter(|item| !item.is_empty())
        .find(|item| match *item {
            "##targetNamespace" | "##local" => false,
            "##any" | "##other" => true,
            other => !is_schema_uri(other, XsdVersion::V1_0),
        })
        .map(|item| format!("`{item}` is not allowed in a namespace list"))
}

/// The `xs:anyURI` lexical space, as the schema for schemas needs it.
///
/// [`crate::value::is_any_uri`] applies the permissive reading the
/// Recommendation asks for: under 1.0 a string is an `anyURI` when
/// percent-escaping the characters a URI may not carry turns it into a legal
/// URI reference, so `foo>bar` — which escapes to `foo%3Ebar` — is perfectly
/// good, and the test suite says so
/// (`msData/datatypes/Facets/anyURI/anyURI_a005`); under 1.1 every string is
/// one, which is what makes `anyURI_a001`'s
/// `<xs:appinfo source="999…9anyURI:"/>` a valid schema document under 1.1 and
/// not under 1.0.
///
/// Two exclusions are applied on top of that in **both** versions, because they
/// are properties of the schema-for-schemas attribute rather than of the
/// datatype. Whitespace inside the value is excluded: these attributes are
/// `whiteSpace="collapse"`, so an interior space survives collapsing and
/// separates two references — and in a `namespaceList` it really does separate
/// two items. The *reverse solidus* is excluded for the same reason it is not
/// in the 1.0 lexical space at all: [RFC 3986] gives `\` no meaning, so
/// `schemaLocation="//a\b"` names no resource this processor could resolve.
///
/// [RFC 3986]: https://www.rfc-editor.org/rfc/rfc3986#section-2
#[must_use]
pub(super) fn is_schema_uri(value: &str, version: XsdVersion) -> bool {
    if !is_any_uri(value, version) {
        return false;
    }
    !value
        .chars()
        .any(|character| character.is_ascii_control() || matches!(character, ' ' | '\\'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_must_be_ncnames() {
        assert!(check(AttrType::Id, "foo", XsdVersion::V1_0).is_none());
        assert!(check(AttrType::Id, "", XsdVersion::V1_0).is_some());
        assert!(check(AttrType::Id, "25", XsdVersion::V1_0).is_some());
        assert!(check(AttrType::NcName, "a:b", XsdVersion::V1_0).is_some());
    }

    #[test]
    fn namespace_lists_reject_hash_tokens_in_lists() {
        assert!(check(AttrType::NamespaceList, "##any", XsdVersion::V1_0).is_none());
        assert!(check(AttrType::NamespaceList, "##anyAttribute", XsdVersion::V1_0).is_some());
        assert!(check(AttrType::NamespaceList, "urn:a ##local", XsdVersion::V1_0).is_none());
        assert!(check(AttrType::NamespaceList, "urn:a ##other", XsdVersion::V1_0).is_some());
    }

    #[test]
    fn uris_reject_the_unwise_characters() {
        assert!(is_schema_uri("http://example.org/a?b#c", XsdVersion::V1_0));
        assert!(is_schema_uri("foo>bar", XsdVersion::V1_0));
        assert!(!is_schema_uri("//a\\b", XsdVersion::V1_0));
        assert!(!is_schema_uri("a b", XsdVersion::V1_0));
        assert!(!is_schema_uri("a#b#c", XsdVersion::V1_0));
    }

    /// XSD 1.1 widens `xs:anyURI` to every string, so the scheme-shaped
    /// rejections go with it — `msData/datatypes/Facets/anyURI/anyURI_a001`
    /// carries `<xs:appinfo source="999…9anyURI:"/>` and is *invalid* under 1.0,
    /// *valid* under 1.1. The two exclusions this predicate adds for the
    /// schema-for-schemas — an embedded space, which would separate two items
    /// of a list-valued attribute, and the reverse solidus, which names no
    /// resolvable resource — survive the widening.
    #[test]
    fn schema_uris_follow_the_version_but_keep_their_own_exclusions() {
        for value in [
            "9999999999999999999999999999999999999999999999999999999999999anyURI:",
            "1111111111111111111http://foo/bar",
            ":a",
            "%",
            "a#b#c",
        ] {
            assert!(
                !is_schema_uri(value, XsdVersion::V1_0),
                "`{value}` under 1.0"
            );
            assert!(
                is_schema_uri(value, XsdVersion::V1_1),
                "`{value}` under 1.1"
            );
            assert!(check(AttrType::AnyUri, value, XsdVersion::V1_0).is_some());
            assert!(check(AttrType::AnyUri, value, XsdVersion::V1_1).is_none());
        }
        for value in ["a b", "//a\\b", "\u{1}x"] {
            for version in [XsdVersion::V1_0, XsdVersion::V1_1] {
                assert!(
                    !is_schema_uri(value, version),
                    "`{value}` is not usable as a schema-for-schemas URI under {version}"
                );
            }
        }
    }

    /// A `namespaceList` item keeps the URI-reference reading under both
    /// versions, so that XSD 1.1's widening of `xs:anyURI` does not turn a
    /// misspelled `##`-keyword into a namespace name.
    #[test]
    fn namespace_list_items_stay_uri_references_under_both_versions() {
        for version in [XsdVersion::V1_0, XsdVersion::V1_1] {
            for value in ["##anyAttribute", "##target", "##Other", "a##b"] {
                assert!(
                    check(AttrType::NamespaceList, value, version).is_some(),
                    "`{value}` is not a `namespaceList` under {version}"
                );
                assert!(
                    check(AttrType::NotNamespaceList, value, version).is_some(),
                    "`{value}` is not a `notNamespace` list under {version}"
                );
            }
            assert!(check(AttrType::NamespaceList, "##any", version).is_none());
            assert!(
                check(
                    AttrType::NamespaceList,
                    "urn:a ##targetNamespace ##local",
                    version
                )
                .is_none()
            );
        }
    }

    #[test]
    fn occurrences_are_unsigned() {
        assert!(check(AttrType::MaxOccurs, "unbounded", XsdVersion::V1_0).is_none());
        assert!(check(AttrType::MaxOccurs, "3", XsdVersion::V1_0).is_none());
        assert!(check(AttrType::MaxOccurs, "-1", XsdVersion::V1_0).is_some());
        assert!(check(AttrType::PositiveInteger, "0", XsdVersion::V1_0).is_some());
    }
}
