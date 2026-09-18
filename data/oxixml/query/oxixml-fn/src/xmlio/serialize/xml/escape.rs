//! Character-level output for the `xml` and `html` methods: markup escaping, CDATA sections and
//! the URI escaping [Serialization 3.1
//! §7.5](https://www.w3.org/TR/xslt-xquery-serialization-31/#HTML_URI-ESCAPING) asks for.

use super::super::params::SerializationParameters;
use super::html::is_uri_attribute;
use std::fmt::Write as _;

/// How an attribute value is written: which of the `html` method's rules apply to it, and whether
/// it is inside a `script` or `style` element, where nothing is escaped at all.
#[derive(Debug, Clone, Copy)]
pub(super) struct AttributeStyle<'a> {
    /// Whether the element carrying the attribute is being written as an HTML element, which
    /// decides both the `&{` rule and whether URI escaping applies.
    pub(super) html: bool,
    /// Whether the attribute is on a descendant of `script` or `style`, whose markup the `html`
    /// method writes as it stands.
    pub(super) raw_text: bool,
    /// The name of the element carrying the attribute, without a prefix.
    pub(super) element: &'a str,
    /// The attribute's name, without a prefix. The pair of the two is what recognises the
    /// URI-valued attributes of [Serialization 3.1
    /// §7.5](https://www.w3.org/TR/xslt-xquery-serialization-31/#HTML_URI-ESCAPING). Both are
    /// empty for a value that is not an attribute at all, such as a namespace URI.
    pub(super) name: &'a str,
}

impl<'a> AttributeStyle<'a> {
    /// A value that is not an attribute of an element — a namespace URI, which never carries an
    /// HTML rule.
    pub(super) const PLAIN: Self = Self {
        html: false,
        raw_text: false,
        element: "",
        name: "",
    };

    /// An ordinary attribute of an element.
    pub(super) const fn attribute(
        html: bool,
        raw_text: bool,
        element: &'a str,
        name: &'a str,
    ) -> Self {
        Self {
            html,
            raw_text,
            element,
            name,
        }
    }
}

/// Writes one text node as a CDATA section, which is what `cdata-section-elements` asks for.
///
/// A CDATA section cannot carry the string `]]>`, so a text node containing it is written as
/// several sections split around that string, `]]]]><![CDATA[>`. A character the character map
/// replaces leaves the section as well: its replacement is markup that must reach the output
/// unescaped, and a CDATA section would escape it (Serialization 3.1 §7.1.4). A character that
/// has to be written as a *reference* leaves it for the same reason — a CDATA section has no way
/// to spell one, so the section is closed around it.
pub(super) fn write_cdata(out: &mut String, value: &str, parameters: &SerializationParameters) {
    let xml_11 = parameters.is_xml_11();
    let mut open = false;
    // How many `]` the open section ends with, so that a following `>` can be recognised as the
    // `]]>` that would close the section early.
    let mut brackets = 0_usize;
    for character in value.chars() {
        let replacement = match parameters.mapped(character) {
            Some(replacement) => Some(replacement.to_owned()),
            None if needs_character_reference(character, xml_11) => {
                let mut reference = String::new();
                write_character_reference(&mut reference, character);
                Some(reference)
            }
            None => None,
        };
        if let Some(replacement) = replacement {
            if open {
                out.push_str("]]>");
                open = false;
            }
            brackets = 0;
            out.push_str(&replacement);
            continue;
        }
        if !open {
            out.push_str("<![CDATA[");
            open = true;
            brackets = 0;
        }
        if character == '>' && brackets >= 2 {
            out.push_str("]]><![CDATA[>");
            brackets = 0;
            continue;
        }
        brackets = if character == ']' { brackets + 1 } else { 0 };
        out.push(character);
    }
    if open {
        out.push_str("]]>");
    }
}

/// Whether `character` is one the `xml` and `html` output methods write as a character reference
/// wherever it occurs, because writing it literally would either change the text on re-parsing
/// or produce a document a reader may not be able to recover.
///
/// [Serialization 3.1 §5.1.1](https://www.w3.org/TR/xslt-xquery-serialization-31/#xml-output)
/// requires it of the C1 controls `#x7F`–`#x9F` — which include `#x85`, the NEL that XML 1.1
/// end-of-line handling would fold into a line feed — and of `#x2028`, LINE SEPARATOR, for the
/// same reason (QT3 `method-xml/K2-Serialization-5`, `-6`, `-9`, `-10`).
///
/// Under `version="1.1"` the same section extends the rule to the **C0** controls `#x1`–`#x1F`:
/// XML 1.1 admits them in content, but only written as a character reference, so a serializer
/// that wrote them literally would produce a document no XML 1.1 parser accepts (the XSLT 3.0
/// suite's `xml-version-007`…`-010`, `-018`, whose expectations are the references themselves).
/// `#x9`, `#xA` and `#xD` are not among them: they are ordinary white space to XML 1.0 and are
/// handled by the escapers above, `#xD` as a reference of its own.
fn needs_character_reference(character: char, xml_11: bool) -> bool {
    if matches!(character, '\u{7F}'..='\u{9F}' | '\u{2028}') {
        return true;
    }
    xml_11 && is_restricted_c0(character)
}

/// Whether `character` is a C0 control that XML 1.0 cannot represent at all and XML 1.1 admits
/// only as a character reference.
pub(crate) const fn is_restricted_c0(character: char) -> bool {
    matches!(character, '\u{1}'..='\u{1F}') && !matches!(character, '\u{9}' | '\u{A}' | '\u{D}')
}

/// Writes `character` as a hexadecimal character reference.
fn write_character_reference(out: &mut String, character: char) {
    let _ = write!(out, "&#x{:X};", u32::from(character));
}

pub(super) fn escape_text(
    out: &mut String,
    value: &str,
    parameters: &SerializationParameters,
    raw_text: bool,
) {
    for character in value.chars() {
        // A character map replaces the character with text that is *not* escaped in turn, which
        // is what makes `use-character-maps` able to inject markup.
        if let Some(replacement) = parameters.mapped(character) {
            out.push_str(replacement);
            continue;
        }
        // Inside `script` and `style` the HTML output method writes the text as it stands: the
        // content is CDATA in HTML's own grammar, so an escape would reach the script verbatim.
        if raw_text {
            out.push(character);
            continue;
        }
        match character {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '\r' => out.push_str("&#xD;"),
            other if needs_character_reference(other, parameters.is_xml_11()) => {
                write_character_reference(out, other);
            }
            other => out.push(other),
        }
    }
}

/// Percent-encodes a URI-valued HTML attribute the way [HTML 4.01 appendix
/// B.2.1](https://www.w3.org/TR/html401/appendix/notes.html#h-B.2.1) recommends and Serialization
/// 3.1 §7.5 adopts: every character outside ASCII is written as the `%HH` of each of its UTF-8
/// bytes, and everything else is left exactly as it stands — a space included, which the QT3 case
/// `Serialization-html-43` pins by keeping `My Documents` unescaped while `bébé` becomes
/// `b%C3%A9b%C3%A9`.
fn percent_encode_non_ascii(value: &str) -> String {
    if value.is_ascii() {
        return value.to_owned();
    }
    let mut escaped = String::with_capacity(value.len());
    let mut buffer = [0_u8; 4];
    for character in value.chars() {
        if character.is_ascii() {
            escaped.push(character);
            continue;
        }
        for byte in character.encode_utf8(&mut buffer).as_bytes() {
            let _ = write!(escaped, "%{byte:02X}");
        }
    }
    escaped
}

pub(super) fn escape_attribute(
    out: &mut String,
    value: &str,
    parameters: &SerializationParameters,
    style: AttributeStyle<'_>,
) {
    // Serialization 3.1 §7.5: the `html` output method percent-encodes a URI-valued attribute
    // before any markup escaping, so that the escaping below sees ASCII alone.
    let escaped;
    let value = if style.html
        && parameters.escape_uri_attributes()
        && is_uri_attribute(style.element, style.name)
        && !value.is_ascii()
    {
        escaped = percent_encode_non_ascii(value);
        escaped.as_str()
    } else {
        value
    };
    let mut characters = value.chars().peekable();
    while let Some(character) = characters.next() {
        if let Some(replacement) = parameters.mapped(character) {
            out.push_str(replacement);
            continue;
        }
        // Serialization 3.1 §7.5: inside `script` and `style` the `html` method escapes nothing —
        // the rule reaches the markup nested in the content, attribute values included, because
        // that markup is script text to an HTML reader (QT3 `Serialization-html-9`, `-10`).
        if style.raw_text {
            out.push(character);
            continue;
        }
        match character {
            // Serialization 3.1 §7.3: the `html` output method leaves an `&` that is immediately
            // followed by `{` alone, because that pair opens a script's own variable reference
            // rather than a character reference (QT3 `Serialization-html-11`).
            '&' if style.html && characters.peek() == Some(&'{') => out.push('&'),
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\t' => out.push_str("&#x9;"),
            '\n' => out.push_str("&#xA;"),
            '\r' => out.push_str("&#xD;"),
            other if needs_character_reference(other, parameters.is_xml_11()) => {
                write_character_reference(out, other);
            }
            other => out.push(other),
        }
    }
}
