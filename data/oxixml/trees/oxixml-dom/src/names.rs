//! Name and character-data validation shared by the node factories.
//!
//! Every check here restates a production of
//! [XML 1.0 Fifth Edition](https://www.w3.org/TR/xml/),
//! [XML 1.1](https://www.w3.org/TR/xml11/) or
//! [Namespaces in XML 1.0](https://www.w3.org/TR/xml-names/); the character
//! classes themselves come from [`oxixml_qname`], which in turn builds on the
//! hand-written `NameStartChar` / `NameChar` tables of `oxixml-unicode`.
//!
//! The `Name` productions need no version dispatch: XML 1.0 Fifth Edition
//! adopted the XML 1.1 `NameStartChar` / `NameChar` classes verbatim, so the two
//! versions agree on every name. Only the `Char` production still differs, and
//! it differs twice over — once for what a character reference may denote
//! ([`is_char`]) and once for what may be written literally
//! ([`is_literal_char`]).

use oxixml_qname::{ExpandedName, is_name, is_ncname};
use oxixml_xml::XmlVersion;

use crate::error::{DomError, DomResult, NamespaceViolation};

/// `http://www.w3.org/XML/1998/namespace`, the namespace the `xml` prefix is
/// permanently bound to.
pub const XML_NAMESPACE: &str = oxixml_qname::XML_NAMESPACE;

/// `http://www.w3.org/2000/xmlns/`, the namespace of the namespace declaration
/// attributes themselves.
pub const XMLNS_NAMESPACE: &str = oxixml_qname::XMLNS_NAMESPACE;

/// Splits a qualified name into its optional prefix and its local part, checking
/// both against the `NCName` production.
///
/// # Errors
///
/// Returns [`DomError::InvalidCharacter`] when either part is not an `NCName`,
/// and [`DomError::Namespace`] when the name contains more than one colon or an
/// empty part.
pub fn split_qualified_name(qualified_name: &str) -> DomResult<(Option<&str>, &str)> {
    match qualified_name.split_once(':') {
        None => {
            if is_ncname(qualified_name) {
                Ok((None, qualified_name))
            } else {
                Err(DomError::InvalidCharacter {
                    production: "NCName",
                    value: qualified_name.to_owned(),
                })
            }
        }
        Some((prefix, local)) => {
            if !is_ncname(prefix) || !is_ncname(local) {
                return Err(DomError::Namespace(
                    NamespaceViolation::MalformedQualifiedName {
                        qualified_name: qualified_name.to_owned(),
                    },
                ));
            }
            Ok((Some(prefix), local))
        }
    }
}

/// Checks the namespace well-formedness constraints DOM Level 3 Core §1.3.3
/// places on `createElementNS`.
///
/// # Errors
///
/// Returns [`DomError::Namespace`] when a reserved prefix or namespace is
/// misused, or when a prefix is used without a namespace name.
pub fn check_element_ns(namespace: Option<&str>, prefix: Option<&str>) -> DomResult<()> {
    check_common_ns(namespace, prefix)?;
    if prefix == Some("xmlns") {
        return Err(DomError::Namespace(
            NamespaceViolation::XmlnsPrefixOnElement,
        ));
    }
    if namespace == Some(XMLNS_NAMESPACE) {
        return Err(DomError::Namespace(
            NamespaceViolation::ReservedNamespaceXmlns {
                qualified_name: crate::node::qualify(prefix, "*"),
            },
        ));
    }
    Ok(())
}

/// Checks the namespace well-formedness constraints DOM Level 3 Core §1.3.3
/// places on `createAttributeNS` / `setAttributeNS`.
///
/// Unlike elements, attributes *may* live in the `xmlns` namespace — that is
/// exactly how namespace declarations are modelled — but only under the
/// qualified name `xmlns` or under the `xmlns` prefix.
///
/// # Errors
///
/// Returns [`DomError::Namespace`] when a reserved prefix or namespace is
/// misused, or when a prefix is used without a namespace name.
pub fn check_attribute_ns(
    namespace: Option<&str>,
    prefix: Option<&str>,
    local: &str,
) -> DomResult<()> {
    check_common_ns(namespace, prefix)?;

    let is_declaration = prefix == Some("xmlns") || (prefix.is_none() && local == "xmlns");
    match (is_declaration, namespace) {
        (true, Some(XMLNS_NAMESPACE)) => Ok(()),
        (true, other) => Err(DomError::Namespace(
            NamespaceViolation::ReservedPrefixXmlns {
                namespace: other.unwrap_or_default().to_owned(),
            },
        )),
        (false, Some(XMLNS_NAMESPACE)) => Err(DomError::Namespace(
            NamespaceViolation::ReservedNamespaceXmlns {
                qualified_name: crate::node::qualify(prefix, local),
            },
        )),
        (false, _) => Ok(()),
    }
}

fn check_common_ns(namespace: Option<&str>, prefix: Option<&str>) -> DomResult<()> {
    if let Some(prefix) = prefix {
        match namespace {
            None => {
                return Err(DomError::Namespace(
                    NamespaceViolation::PrefixWithoutNamespace {
                        prefix: prefix.to_owned(),
                    },
                ));
            }
            Some("") => {
                return Err(DomError::Namespace(
                    NamespaceViolation::PrefixWithoutNamespace {
                        prefix: prefix.to_owned(),
                    },
                ));
            }
            Some(_) => {}
        }
    }
    if prefix == Some("xml") && namespace != Some(XML_NAMESPACE) {
        return Err(DomError::Namespace(NamespaceViolation::ReservedPrefixXml {
            namespace: namespace.unwrap_or_default().to_owned(),
        }));
    }
    Ok(())
}

/// Validates a name against the XML `Name` production.
///
/// # Errors
///
/// Returns [`DomError::InvalidCharacter`] when the name does not match.
pub fn check_name(name: &str) -> DomResult<()> {
    if is_name(name) {
        Ok(())
    } else {
        Err(DomError::InvalidCharacter {
            production: "Name",
            value: name.to_owned(),
        })
    }
}

/// Validates a processing-instruction target: it must be a `Name` that is not
/// any capitalization of the reserved string `xml`
/// ([XML 1.0 §2.6](https://www.w3.org/TR/xml/#sec-pi)).
///
/// # Errors
///
/// Returns [`DomError::InvalidCharacter`] when the target does not match.
pub fn check_pi_target(target: &str) -> DomResult<()> {
    if !is_name(target) || target.contains(':') {
        return Err(DomError::InvalidCharacter {
            production: "PITarget",
            value: target.to_owned(),
        });
    }
    if target.eq_ignore_ascii_case("xml") {
        return Err(DomError::InvalidCharacter {
            production: "PITarget",
            value: target.to_owned(),
        });
    }
    Ok(())
}

/// Validates processing-instruction data: it must not contain the `?>`
/// delimiter, which would terminate the instruction early.
///
/// # Errors
///
/// Returns [`DomError::InvalidCharacter`] when the data contains `?>`.
pub fn check_pi_data(data: &str, version: XmlVersion) -> DomResult<()> {
    if data.contains("?>") {
        return Err(DomError::InvalidCharacter {
            production: "PI",
            value: data.to_owned(),
        });
    }
    check_literal_chars(data, version)
}

/// Validates comment data against [XML 1.0 §2.5](https://www.w3.org/TR/xml/#sec-comments):
/// it may not contain `--` and may not end with `-`.
///
/// # Errors
///
/// Returns [`DomError::InvalidCharacter`] when the data violates the production.
pub fn check_comment(data: &str, version: XmlVersion) -> DomResult<()> {
    if data.contains("--") || data.ends_with('-') {
        return Err(DomError::InvalidCharacter {
            production: "Comment",
            value: data.to_owned(),
        });
    }
    check_literal_chars(data, version)
}

/// Validates CDATA-section content: it may not contain the `]]>` delimiter.
///
/// # Errors
///
/// Returns [`DomError::InvalidCharacter`] when the content contains `]]>`.
pub fn check_cdata(data: &str, version: XmlVersion) -> DomResult<()> {
    if data.contains("]]>") {
        return Err(DomError::InvalidCharacter {
            production: "CDSect",
            value: data.to_owned(),
        });
    }
    check_literal_chars(data, version)
}

/// Validates character data that has no reference syntax of its own — the
/// content of a comment, a CDATA section or a processing instruction — against
/// the `Char` production of `version` *minus* the XML 1.1 `RestrictedChar`s.
///
/// Text and attribute values go through [`check_chars`] instead: they may hold a
/// restricted character, because they can name one with a character reference.
/// A comment cannot, so a restricted character put into one could never be
/// written back out.
///
/// # Errors
///
/// Returns [`DomError::InvalidCharacter`] for the first character the version
/// does not admit literally.
pub fn check_literal_chars(data: &str, version: XmlVersion) -> DomResult<()> {
    for ch in data.chars() {
        if !is_literal_char(ch, version) {
            return Err(DomError::InvalidCharacter {
                production: "Char",
                value: data.to_owned(),
            });
        }
    }
    Ok(())
}

/// Validates that every character of `data` matches the `Char` production of
/// `version`, i.e. that it can be written to such a document at all.
///
/// The XML 1.1 production is the wider one: it admits the C0 controls other than
/// `#x0`, which a 1.1 document writes as character references. Whether those
/// characters may also appear *literally* is a separate question, answered by
/// [`is_literal_char`].
///
/// # Errors
///
/// Returns [`DomError::InvalidCharacter`] for the first character outside the
/// production.
pub fn check_chars(data: &str, version: XmlVersion) -> DomResult<()> {
    for ch in data.chars() {
        if !is_char(ch, version) {
            return Err(DomError::InvalidCharacter {
                production: "Char",
                value: data.to_owned(),
            });
        }
    }
    Ok(())
}

/// Returns `true` when `ch` matches the XML 1.0 Fifth Edition `Char`
/// production: `#x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] |
/// [#x10000-#x10FFFF]`.
///
/// Surrogate code points cannot occur in a Rust `char`, so only the `#x0`–`#x8`,
/// `#xB`, `#xC`, `#xE`–`#x1F` control characters and `#xFFFE` / `#xFFFF` are
/// rejected.
#[must_use]
pub const fn is_xml_char(ch: char) -> bool {
    matches!(ch, '\u{9}' | '\u{A}' | '\u{D}')
        || matches!(ch, '\u{20}'..='\u{D7FF}' | '\u{E000}'..='\u{FFFD}' | '\u{10000}'..='\u{10FFFF}')
}

/// Returns `true` when `ch` matches the [XML 1.1 `Char`] production:
/// `[#x1-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]`.
///
/// Compared with [`is_xml_char`] this admits every C0 control except `#x0`; a
/// 1.1 document may only spell most of them as character references, which is
/// what [`is_xml11_restricted_char`] separates out.
///
/// [XML 1.1 `Char`]: https://www.w3.org/TR/xml11/#NT-Char
#[must_use]
pub const fn is_xml11_char(ch: char) -> bool {
    matches!(ch, '\u{1}'..='\u{D7FF}' | '\u{E000}'..='\u{FFFD}' | '\u{10000}'..='\u{10FFFF}')
}

/// Returns `true` when `ch` matches the [XML 1.1 `RestrictedChar`] production:
/// `[#x1-#x8] | [#xB-#xC] | [#xE-#x1F] | [#x7F-#x84] | [#x86-#x9F]`.
///
/// Those characters are `Char`s — a 1.1 document may name them with a character
/// reference — but [§2.1] subtracts them from the `document` production, so
/// writing one *literally* is a fatal error:
///
/// ```text
/// document ::= ( prolog element Misc* ) - ( Char* RestrictedChar Char* )
/// ```
///
/// `#x85` and `#x2028` are deliberately absent: XML 1.1 treats them as line
/// endings rather than as restricted characters.
///
/// [XML 1.1 `RestrictedChar`]: https://www.w3.org/TR/xml11/#NT-RestrictedChar
/// [§2.1]: https://www.w3.org/TR/xml11/#sec-well-formed
#[must_use]
pub const fn is_xml11_restricted_char(ch: char) -> bool {
    matches!(ch,
        '\u{1}'..='\u{8}'
        | '\u{B}'..='\u{C}'
        | '\u{E}'..='\u{1F}'
        | '\u{7F}'..='\u{84}'
        | '\u{86}'..='\u{9F}')
}

/// Dispatches [`is_xml_char`] / [`is_xml11_char`] on the document's version.
///
/// This is the predicate for a character that reached the tree through a
/// *character reference*, which is how a 1.1 document names a `RestrictedChar`.
#[must_use]
pub const fn is_char(ch: char, version: XmlVersion) -> bool {
    match version {
        XmlVersion::Explicit1_1 => is_xml11_char(ch),
        _ => is_xml_char(ch),
    }
}

/// Returns whether `ch` may appear *literally* in a document of `version`, i.e.
/// [`is_char`] minus the XML 1.1 `RestrictedChar` production.
#[must_use]
pub const fn is_literal_char(ch: char, version: XmlVersion) -> bool {
    match version {
        XmlVersion::Explicit1_1 => is_xml11_char(ch) && !is_xml11_restricted_char(ch),
        _ => is_xml_char(ch),
    }
}

/// Returns the first character of `text` that `version` forbids to appear
/// literally, if any.
///
/// Under XML 1.0 every `Char` may be written literally, so this only ever
/// reports something for an XML 1.1 document.
#[must_use]
pub fn first_restricted_char(text: &str, version: XmlVersion) -> Option<char> {
    if !matches!(version, XmlVersion::Explicit1_1) {
        return None;
    }
    text.chars()
        .find(|ch| is_char(*ch, version) && !is_literal_char(*ch, version))
}

/// Builds an [`ExpandedName`] from an optional namespace name and a local part.
///
/// An empty namespace name is folded to "no namespace", matching the DOM rule
/// that `""` and `null` are equivalent for `namespaceURI`.
#[must_use]
pub fn expanded(namespace: Option<&str>, local: &str) -> ExpandedName {
    match namespace {
        Some(ns) if !ns.is_empty() => ExpandedName::new(Some(ns), local),
        _ => ExpandedName::new(None::<&str>, local),
    }
}
