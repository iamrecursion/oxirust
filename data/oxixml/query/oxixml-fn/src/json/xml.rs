//! The XML representation of JSON:
//! [`fn:json-to-xml`](https://www.w3.org/TR/xpath-functions-31/#func-json-to-xml) and
//! [`fn:xml-to-json`](https://www.w3.org/TR/xpath-functions-31/#func-xml-to-json).
//!
//! `fn:json-to-xml` writes the representation as XML text and hands it to
//! [`oxixml_infoset::Tree`], which is both the shortest route to a document node and the one
//! that guarantees the result is a well-formed tree. `fn:xml-to-json` walks the tree back,
//! checking every rule §17.6 lays down on the vocabulary.

use super::parser::{self, Duplicates, JsonValue, is_xml_character};
use super::{JSON_XML_NAMESPACE, JsonFunction, JsonOptions};
use crate::error::{FnError, FnErrorCode, FnResult};
use oxixml_infoset::{AtomicType, AtomicValue, NodeKind, NodeRef, ParseOptions, Tree, XdmNode};

/// [`fn:json-to-xml`](https://www.w3.org/TR/xpath-functions-31/#func-json-to-xml): the document
/// node of the XML representation of a JSON text.
///
/// # Errors
///
/// The `FOJS` codes [`parse_json`](super::parse_json) raises, and
/// [`FnErrorCode::Fojs0001`] if the representation the parse produced is somehow not
/// well-formed XML.
///
/// # Examples
///
/// ```
/// use oxixml_fn::json::{json_to_xml, JsonOptions};
/// use oxixml_infoset::XdmNode;
///
/// let document = json_to_xml("{\"a\":1}", &JsonOptions::default())?;
/// assert_eq!(document.string_value(), "1");
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn json_to_xml(json: &str, options: &JsonOptions<'_>) -> FnResult<NodeRef> {
    json_to_xml_with_base(json, options, None)
}

/// [`json_to_xml`] with the static base URI the result's document node must report.
///
/// [F&O 3.1 §17.5.3](https://www.w3.org/TR/xpath-functions-31/#func-json-to-xml) says the base
/// URI of the document node in the result is the static base URI of the function call, which is
/// what `json-to-xml-041` checks; only the caller knows it.
///
/// # Errors
///
/// As [`json_to_xml`].
pub fn json_to_xml_with_base(
    json: &str,
    options: &JsonOptions<'_>,
    base_uri: Option<&str>,
) -> FnResult<NodeRef> {
    let duplicates = match options.function {
        // The caller declared which function it is serving, so `duplicates` is already either
        // `fn:json-to-xml`'s own default — `retain`, which its representation can express — or
        // the explicit value the option map gave. An explicit `use-first` must survive: it keeps
        // the *first* entry with a given key and drops the rest (QT3 `app-Walmsley/d1e78807d`).
        JsonFunction::JsonToXml => options.duplicates,
        // A caller that left `function` alone handed us `fn:parse-json`'s defaults, whose
        // `use-first` is a default rather than a choice; substitute this function's own.
        JsonFunction::ParseJson if options.duplicates == Duplicates::UseFirst => Duplicates::Retain,
        JsonFunction::ParseJson => options.duplicates,
    };
    let value = parser::parse(
        json,
        duplicates,
        &options.string_policy(),
        options.max_depth,
    )?;
    let mut xml = String::new();
    write_element(&mut xml, &value, None, options.escape, true);
    let mut parse_options = ParseOptions::default();
    if let Some(base_uri) = base_uri {
        parse_options = parse_options.document_uri(base_uri);
    }
    let tree = Tree::parse_str_with(&xml, parse_options).map_err(|error| {
        FnError::new(
            FnErrorCode::Fojs0001,
            format!("the XML representation of the JSON text is not well-formed: {error}"),
        )
    })?;
    Ok(tree.root())
}

/// Writes one value as an element of the JSON XML vocabulary.
fn write_element(out: &mut String, value: &JsonValue, key: Option<&str>, escape: bool, root: bool) {
    let name = match value {
        JsonValue::Null => "null",
        JsonValue::Bool(_) => "boolean",
        JsonValue::Number(..) => "number",
        JsonValue::String(_) => "string",
        JsonValue::Array(_) => "array",
        JsonValue::Object(_) => "map",
    };
    out.push('<');
    out.push_str(name);
    if root {
        out.push_str(" xmlns=\"");
        out.push_str(JSON_XML_NAMESPACE);
        out.push('"');
    }
    if let Some(key) = key {
        if escape && key.contains('\\') {
            out.push_str(" escaped-key=\"true\"");
        }
        out.push_str(" key=\"");
        write_attribute_text(out, key);
        out.push('"');
    }
    if let JsonValue::String(text) = value {
        if escape && text.contains('\\') {
            out.push_str(" escaped=\"true\"");
        }
    }
    match value {
        JsonValue::Null => out.push_str("/>"),
        JsonValue::Bool(flag) => {
            out.push('>');
            out.push_str(if *flag { "true" } else { "false" });
            close(out, name);
        }
        JsonValue::Number(_, lexical) => {
            out.push('>');
            out.push_str(lexical);
            close(out, name);
        }
        JsonValue::String(text) => {
            out.push('>');
            write_element_text(out, text);
            close(out, name);
        }
        JsonValue::Array(members) => {
            if members.is_empty() {
                out.push_str("/>");
            } else {
                out.push('>');
                for member in members {
                    write_element(out, member, None, escape, false);
                }
                close(out, name);
            }
        }
        JsonValue::Object(entries) => {
            if entries.is_empty() {
                out.push_str("/>");
            } else {
                out.push('>');
                for (entry_key, member) in entries {
                    write_element(out, member, Some(entry_key), escape, false);
                }
                close(out, name);
            }
        }
    }
}

fn close(out: &mut String, name: &str) {
    out.push_str("</");
    out.push_str(name);
    out.push('>');
}

fn write_element_text(out: &mut String, text: &str) {
    for character in text.chars() {
        match character {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '\r' => out.push_str("&#xD;"),
            _ => out.push(character),
        }
    }
}

fn write_attribute_text(out: &mut String, text: &str) {
    for character in text.chars() {
        match character {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '"' => out.push_str("&quot;"),
            '\t' => out.push_str("&#x9;"),
            '\n' => out.push_str("&#xA;"),
            '\r' => out.push_str("&#xD;"),
            _ => out.push(character),
        }
    }
}

// --- fn:xml-to-json -------------------------------------------------------------------------

/// [`fn:xml-to-json`](https://www.w3.org/TR/xpath-functions-31/#func-xml-to-json): the JSON text
/// an element of the XML representation denotes.
///
/// # Errors
///
/// [`FnErrorCode::Fojs0006`] when `node` is not a valid instance of the vocabulary, or when it
/// nests deeper than [`JsonOptions::max_depth`], and [`FnErrorCode::Fojs0007`] when a string
/// marked `escaped="true"` contains an escape sequence JSON does not define.
///
/// # Examples
///
/// ```
/// use oxixml_fn::json::{json_to_xml, xml_to_json, JsonOptions};
/// use oxixml_infoset::XdmNode;
///
/// let options = JsonOptions::default();
/// let document = json_to_xml("[1,2]", &options)?;
/// let root = document.children().into_iter().next().expect("the root element");
/// assert_eq!(xml_to_json(&root, &options)?, "[1,2]");
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn xml_to_json(node: &NodeRef, options: &JsonOptions<'_>) -> FnResult<String> {
    let element = match node.node_kind() {
        NodeKind::Document => {
            // A document node stands for the single JSON value its element child represents, so
            // it must have exactly one. A temporary tree is free to hold several — nothing in
            // XSLT stops `<xsl:variable>` from building two siblings — and taking the first
            // would answer a question the caller did not ask (`xml-to-json-C001`).
            let mut elements = node
                .children()
                .into_iter()
                .filter(|child| child.node_kind() == NodeKind::Element);
            let first = elements
                .next()
                .ok_or_else(|| invalid("the document has no element child"))?;
            if elements.next().is_some() {
                return Err(invalid(
                    "the document has more than one element child, so it represents no single \
                     JSON value",
                ));
            }
            first
        }
        NodeKind::Element => node.clone(),
        _ => {
            return Err(invalid(
                "fn:xml-to-json expects an element or document node",
            ));
        }
    };
    let mut out = String::new();
    write_json(&mut out, &element, options.indent, 0, options.max_depth)?;
    Ok(out)
}

fn invalid(what: impl Into<String>) -> FnError {
    FnError::new(FnErrorCode::Fojs0006, what.into())
}

/// Writes one element of the JSON XML vocabulary as JSON text.
///
/// `depth` is how many `map` and `array` elements enclose this one, and `limit` is how many may:
/// the walk is recursive, and the element tree comes from the caller, so a tree deep enough to
/// exhaust the stack is rejected rather than allowed to abort the process.
fn write_json(
    out: &mut String,
    element: &NodeRef,
    indent: bool,
    depth: usize,
    limit: usize,
) -> FnResult<()> {
    if depth > limit {
        return Err(invalid(format!(
            "the XML representation nests more than {limit} levels deep, which is this implementation's configurable limit"
        )));
    }
    let Some(name) = element.node_name() else {
        return Err(invalid("an element of the JSON vocabulary was expected"));
    };
    if name.namespace() != Some(JSON_XML_NAMESPACE) {
        return Err(invalid(format!(
            "`{name}` is not in the JSON XML representation namespace"
        )));
    }
    check_attributes(element)?;
    match name.local_name() {
        "null" => {
            require_no_content(element)?;
            out.push_str("null");
        }
        "boolean" => {
            let text = simple_content(element)?;
            match text.trim() {
                "true" | "1" => out.push_str("true"),
                "false" | "0" => out.push_str("false"),
                other => {
                    return Err(invalid(format!("`{other}` is not an xs:boolean")));
                }
            }
        }
        "number" => {
            let text = simple_content(element)?;
            let value = AtomicValue::parse(AtomicType::Double, text.trim())
                .map_err(|_| invalid(format!("`{text}` is not an xs:double")))?;
            let rendered = value.string_value();
            if rendered == "NaN" || rendered.ends_with("INF") {
                return Err(invalid("a JSON number may not be NaN or infinite"));
            }
            out.push_str(&rendered);
        }
        "string" => {
            let text = simple_content(element)?;
            if is_escaped(element, "escaped")? {
                out.push('"');
                out.push_str(&checked_escapes(&text)?);
                out.push('"');
            } else {
                out.push_str(&json_string(&text));
            }
        }
        "array" => {
            let members = json_children(element)?;
            write_group(out, '[', ']', members.len(), indent, depth, |out, index| {
                write_json(out, &members[index], indent, depth + 1, limit)
            })?;
        }
        "map" => {
            let members = json_children(element)?;
            let mut keys: Vec<String> = Vec::with_capacity(members.len());
            for member in &members {
                keys.push(entry_key(member)?);
            }
            // F&O 3.1 §17.6.2: two keys are the same key when they are equal *after* the
            // `escaped-key` unescaping, so `\u0031` and `1` collide while `\"` and `\\\"`
            // do not. The check therefore runs on the unescaped values, not the lexical ones.
            let mut unescaped: Vec<String> = Vec::with_capacity(keys.len());
            for (key, member) in keys.iter().zip(&members) {
                unescaped.push(if is_escaped(member, "escaped-key")? {
                    unescape_json(key)
                } else {
                    key.clone()
                });
            }
            for (index, key) in unescaped.iter().enumerate() {
                if unescaped[..index].contains(key) {
                    return Err(invalid(format!("the key `{key}` appears more than once")));
                }
            }
            write_group(out, '{', '}', members.len(), indent, depth, |out, index| {
                let member = &members[index];
                if is_escaped(member, "escaped-key")? {
                    out.push('"');
                    out.push_str(&checked_escapes(&keys[index])?);
                    out.push('"');
                } else {
                    out.push_str(&json_string(&keys[index]));
                }
                out.push(':');
                if indent {
                    out.push(' ');
                }
                write_json(out, member, indent, depth + 1, limit)
            })?;
        }
        other => {
            return Err(invalid(format!(
                "`{other}` is not an element of the JSON XML representation"
            )));
        }
    }
    Ok(())
}

/// The element children of a `map` or `array`, rejecting any non-whitespace text between them.
fn json_children(element: &NodeRef) -> FnResult<Vec<NodeRef>> {
    let mut children = Vec::new();
    for child in element.children() {
        match child.node_kind() {
            NodeKind::Element => children.push(child),
            NodeKind::Text if !child.string_value().trim().is_empty() => {
                return Err(invalid(
                    "a `map` or `array` element may only contain element children",
                ));
            }
            _ => {}
        }
    }
    Ok(children)
}

fn entry_key(member: &NodeRef) -> FnResult<String> {
    member
        .attributes()
        .into_iter()
        .find(|attribute| {
            attribute
                .node_name()
                .is_some_and(|name| name.namespace().is_none() && name.local_name() == "key")
        })
        .map(|attribute| attribute.string_value())
        .ok_or_else(|| invalid("every child of a `map` needs a `key` attribute"))
}

/// Reads an `escaped` or `escaped-key` attribute.
///
/// F&O 3.1 §17.6.2 types both `xs:boolean`, so a value outside the four lexical forms XSD gives
/// that type is [`FnErrorCode::Fojs0006`] rather than a silent `false` — `xml-to-json-062`,
/// `-063` and `xml-to-json-C-012`/`-014` all turn on exactly that.
fn is_escaped(element: &NodeRef, attribute: &str) -> FnResult<bool> {
    for candidate in element.attributes() {
        let matches_name = candidate
            .node_name()
            .is_some_and(|name| name.namespace().is_none() && name.local_name() == attribute);
        if !matches_name {
            continue;
        }
        return match candidate.string_value().trim() {
            "true" | "1" => Ok(true),
            "false" | "0" => Ok(false),
            other => Err(invalid(format!(
                "`{other}` is not an xs:boolean value for the `{attribute}` attribute"
            ))),
        };
    }
    Ok(false)
}

/// Rejects an attribute the vocabulary does not define.
///
/// F&O 3.1 §17.6.2 allows exactly `key`, `escaped` and `escaped-key` in no namespace, plus any
/// attribute in a namespace *other* than the JSON one; an attribute in the JSON namespace itself
/// is an error (`xml-to-json-069`).
fn check_attributes(element: &NodeRef) -> FnResult<()> {
    for attribute in element.attributes() {
        let Some(name) = attribute.node_name() else {
            continue;
        };
        match name.namespace() {
            Some(JSON_XML_NAMESPACE) => {
                return Err(invalid(format!(
                    "`{name}` is in the JSON XML representation namespace, where no attribute is defined"
                )));
            }
            Some(_) => continue,
            None => {}
        }
        if !matches!(name.local_name(), "key" | "escaped" | "escaped-key") {
            return Err(invalid(format!(
                "`{}` is not an attribute of the JSON XML representation",
                name.local_name()
            )));
        }
    }
    Ok(())
}

/// The text content of a `string`, `number` or `boolean` element.
///
/// F&O 3.1 §17.6.2 gives these three elements simple content, so an element child is
/// [`FnErrorCode::Fojs0006`] — `xml-to-json-040`, `-083` and `xml-to-json-C-008`/`-009` all
/// supply one, and taking the plain string value would silently accept them.
fn simple_content(element: &NodeRef) -> FnResult<String> {
    for child in element.children() {
        if child.node_kind() == NodeKind::Element {
            return Err(invalid(format!(
                "a `{}` element may not have element children",
                element
                    .node_name()
                    .map(|name| name.local_name().to_owned())
                    .unwrap_or_default()
            )));
        }
    }
    Ok(element.string_value())
}

fn require_no_content(element: &NodeRef) -> FnResult<()> {
    for child in element.children() {
        match child.node_kind() {
            NodeKind::Text if child.string_value().trim().is_empty() => {}
            NodeKind::Comment | NodeKind::ProcessingInstruction => {}
            _ => return Err(invalid("a `null` element must be empty")),
        }
    }
    Ok(())
}

/// Decodes the JSON escapes of a string marked `escaped` or `escaped-key`, for comparison only:
/// an escape this function cannot read is left as it stands, because reporting it is
/// [`checked_escapes`]'s job and doing it here would report it in the wrong order.
fn unescape_json(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut characters = text.chars().peekable();
    while let Some(character) = characters.next() {
        if character != '\\' {
            out.push(character);
            continue;
        }
        match characters.next() {
            Some('"') => out.push('"'),
            Some('\\') => out.push('\\'),
            Some('/') => out.push('/'),
            Some('b') => out.push('\u{8}'),
            Some('f') => out.push('\u{c}'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('t') => out.push('\t'),
            Some('u') => {
                let digits: String = (0..4)
                    .map_while(|_| characters.next_if(char::is_ascii_hexdigit))
                    .collect();
                match u32::from_str_radix(&digits, 16)
                    .ok()
                    .and_then(char::from_u32)
                {
                    Some(decoded) => out.push(decoded),
                    None => {
                        out.push_str("\\u");
                        out.push_str(&digits);
                    }
                }
            }
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

/// Checks that a string marked `escaped` uses only the escape sequences JSON defines, leaving
/// them in place.
fn checked_escapes(text: &str) -> FnResult<String> {
    let mut out = String::with_capacity(text.len());
    let mut characters = text.chars().peekable();
    while let Some(character) = characters.next() {
        if character != '\\' {
            // A raw quotation mark still has to be escaped on the way out.
            if character == '"' {
                out.push_str("\\\"");
            } else if character == '/' {
                // The `json` escaping of Serialization 3.1 and F&O 3.1 §17.6.2 spells a raw
                // solidus `\/`, even inside a string that is already marked `escaped`.
                out.push_str("\\/");
            } else if is_control(character) {
                out.push_str(&short_escape(character));
            } else {
                out.push(character);
            }
            continue;
        }
        let Some(marker) = characters.next() else {
            return Err(FnError::new(
                FnErrorCode::Fojs0007,
                "an escaped string ends in a backslash",
            ));
        };
        match marker {
            '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' => {
                out.push('\\');
                out.push(marker);
            }
            'u' => {
                let mut digits = String::new();
                for _ in 0..4 {
                    match characters.next().filter(char::is_ascii_hexdigit) {
                        Some(digit) => digits.push(digit),
                        None => {
                            return Err(FnError::new(
                                FnErrorCode::Fojs0007,
                                "`\\u` must be followed by four hexadecimal digits",
                            ));
                        }
                    }
                }
                out.push_str("\\u");
                out.push_str(&digits);
            }
            other => {
                return Err(FnError::new(
                    FnErrorCode::Fojs0007,
                    format!("`\\{other}` is not a JSON escape sequence"),
                ));
            }
        }
    }
    Ok(out)
}

/// Whether `fn:xml-to-json` writes `character` as an escape sequence because it is a control.
///
/// [F&O 3.1 §17.6.2](https://www.w3.org/TR/xpath-functions-31/#func-xml-to-json) escapes "any
/// other character in the range 1-31, 127-159", the two-character escapes `\b` `\t` `\n` `\f`
/// `\r` covering the five it names separately. The upper run is the C1 controls, which are
/// legal XML 1.0 characters and so reach the writer as themselves rather than being rejected
/// earlier; JSON still has no way to carry them literally.
const fn is_control(character: char) -> bool {
    let code = character as u32;
    code < 0x20 || (0x7f <= code && code <= 0x9f)
}

fn short_escape(character: char) -> String {
    match character {
        '\u{8}' => "\\b".to_owned(),
        '\u{c}' => "\\f".to_owned(),
        '\n' => "\\n".to_owned(),
        '\r' => "\\r".to_owned(),
        '\t' => "\\t".to_owned(),
        other => format!("\\u{:04X}", other as u32),
    }
}

/// The JSON escape for one character, as a *UTF-16* escape: a character above the BMP has no
/// single `\uXXXX` spelling and is written as the surrogate pair JSON defines it by
/// (`U+1D11E` becomes `𝄞`).
pub(crate) fn json_escape(character: char) -> String {
    let code = character as u32;
    if code <= 0xffff {
        return short_escape(character);
    }
    let mut buffer = [0_u16; 2];
    let units = character.encode_utf16(&mut buffer);
    units
        .iter()
        .map(|unit| format!("\\u{unit:04X}"))
        .collect::<String>()
}

/// A JSON string literal, with the escapes JSON requires.
#[must_use]
pub(crate) fn json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            // F&O 3.1 §17.6.2 escapes the solidus, and the C0 and C1 controls with it.
            '/' => out.push_str("\\/"),
            c if is_control(c) || !is_xml_character(c) => {
                out.push_str(&short_escape(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn write_group(
    out: &mut String,
    open: char,
    close: char,
    count: usize,
    indent: bool,
    depth: usize,
    mut member: impl FnMut(&mut String, usize) -> FnResult<()>,
) -> FnResult<()> {
    out.push(open);
    for index in 0..count {
        if index > 0 {
            out.push(',');
        }
        if indent {
            out.push('\n');
            for _ in 0..=depth {
                out.push_str("  ");
            }
        }
        member(out, index)?;
    }
    if indent && count > 0 {
        out.push('\n');
        for _ in 0..depth {
            out.push_str("  ");
        }
    }
    out.push(close);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(json: &str) -> FnResult<String> {
        let options = JsonOptions::default();
        let document = json_to_xml(json, &options)?;
        let root = document
            .children()
            .into_iter()
            .next()
            .expect("the root element");
        xml_to_json(&root, &options)
    }

    #[test]
    fn the_vocabulary_round_trips() -> FnResult<()> {
        assert_eq!(round_trip("{}")?, "{}");
        assert_eq!(round_trip("[]")?, "[]");
        assert_eq!(round_trip("[1,\"a\",true,null]")?, "[1,\"a\",true,null]");
        assert_eq!(round_trip("{\"a\":{\"b\":[]}}")?, "{\"a\":{\"b\":[]}}");
        Ok(())
    }

    #[test]
    fn numbers_take_the_xpath_canonical_form() -> FnResult<()> {
        assert_eq!(round_trip("[1e6]")?, "[1.0E6]");
        assert_eq!(round_trip("[0.001]")?, "[0.001]");
        assert_eq!(round_trip("[23]")?, "[23]");
        Ok(())
    }

    /// F&O 3.1 §17.6.2: "any other character in the range 1-31, 127-159 ... is replaced by
    /// `\\uHHHH`". The C1 controls (x7F-x9F) are legal XML 1.0 characters, so they reach the
    /// writer as themselves rather than being rejected earlier, and JSON still has no way to
    /// carry them literally. QT3 `fn/xml-to-json` `xml-to-json-073` enumerates the whole run and
    /// `xml-to-json-072` the five with two-character escapes; the C0 half needs an XML 1.1
    /// document to write at all, which is the shape `xml-to-json-D015`/`-D017`/`-D018` use.
    #[test]
    fn the_c0_and_c1_controls_are_escaped() -> FnResult<()> {
        const JSON: &str = "http://www.w3.org/2005/xpath-functions";
        let options = JsonOptions::default();
        for (prolog, reference, expected) in [
            ("", "&#144;", r#""-\u0090-""#),
            ("", "&#127;", r#""-\u007F-""#),
            ("", "&#128;", r#""-\u0080-""#),
            ("", "&#159;", r#""-\u009F-""#),
            // The first character past the run stays as it is.
            ("", "&#160;", "\"-\u{a0}-\""),
            (r#"<?xml version="1.1"?>"#, "&#7;", r#""-\u0007-""#),
            (r#"<?xml version="1.1"?>"#, "&#8;", r#""-\b-""#),
            (r#"<?xml version="1.1"?>"#, "&#12;", r#""-\f-""#),
        ] {
            for escaped in ["true", "false"] {
                let source = format!(
                    r#"{prolog}<string xmlns="{JSON}" escaped="{escaped}">-{reference}-</string>"#
                );
                let tree = Tree::parse_str(&source).expect("well-formed");
                let root = tree.document_element().expect("a document element");
                assert_eq!(
                    xml_to_json(&root, &options)?,
                    expected,
                    "{reference}, escaped={escaped}"
                );
            }
        }
        Ok(())
    }

    /// A document node represents the one JSON value its element child does. Two element
    /// children represent none, which is `FOJS0006` rather than a silent choice of the first
    /// (`xml-to-json-C001`). A parsed document can never have that shape; a *fragment* and an
    /// XSLT temporary tree both can.
    #[test]
    fn a_document_with_two_element_children_is_rejected() -> FnResult<()> {
        const JSON: &str = "http://www.w3.org/2005/xpath-functions";
        let options = JsonOptions::default();

        let two = crate::xmlio::parse_xml_fragment(
            &format!(
                r#"<j:string xmlns:j="{JSON}">a</j:string><j:string xmlns:j="{JSON}">b</j:string>"#
            ),
            None,
        )?;
        let error = xml_to_json(&two, &options).expect_err("two element children");
        assert!(error.is(FnErrorCode::Fojs0006));

        // One element child is the ordinary case and still answers.
        let one = crate::xmlio::parse_xml_fragment(
            &format!(r#"<j:string xmlns:j="{JSON}">a</j:string>"#),
            None,
        )?;
        assert_eq!(xml_to_json(&one, &options)?, "\"a\"");
        Ok(())
    }

    #[test]
    fn a_foreign_element_is_rejected() {
        let tree = Tree::parse_str("<thing/>").expect("well-formed");
        let root = tree.document_element().expect("a document element");
        let error =
            xml_to_json(&root, &JsonOptions::default()).expect_err("not the JSON vocabulary");
        assert!(error.is(FnErrorCode::Fojs0006));
    }

    /// F&O 3.1 §17.6.2 constrains the vocabulary far more tightly than "an element in the JSON
    /// namespace": simple content, `xs:boolean`-valued flags, and no attribute of its own
    /// namespace.
    #[test]
    fn the_vocabulary_constraints_are_enforced() {
        const JSON: &str = "http://www.w3.org/2005/xpath-functions";
        for source in [
            // An element child of a simple-content element.
            format!(r#"<string xmlns="{JSON}">ok<null/></string>"#),
            format!(r#"<j:map xmlns:j="{JSON}"><j:string key="v"><j:array/></j:string></j:map>"#),
            // A flag that is not an xs:boolean.
            format!(r#"<string xmlns="{JSON}" escaped="potty">x</string>"#),
            format!(r#"<map xmlns="{JSON}"><string escaped-key="maybe" key="k">v</string></map>"#),
            // An attribute in the JSON namespace itself.
            format!(
                r#"<j:map xmlns:j="{JSON}" j:base="urn:x"><j:string key="k">v</j:string></j:map>"#
            ),
        ] {
            let tree = Tree::parse_str(&source).expect("well-formed");
            let root = tree.document_element().expect("a document element");
            let error = xml_to_json(&root, &JsonOptions::default()).expect_err(&source);
            assert!(error.is(FnErrorCode::Fojs0006), "{source}");
        }

        // An attribute in some *other* namespace is explicitly allowed, and ignored.
        let source =
            format!(r#"<j:string xmlns:j="{JSON}" xmlns:x="urn:x" x:note="ok">v</j:string>"#);
        let tree = Tree::parse_str(&source).expect("well-formed");
        let root = tree.document_element().expect("a document element");
        assert_eq!(
            xml_to_json(&root, &JsonOptions::default()).expect("a foreign attribute is ignored"),
            "\"v\""
        );
    }

    #[test]
    fn an_undefined_escape_is_rejected() {
        assert!(checked_escapes("a\\qb").is_err());
        assert!(checked_escapes("a\\u00").is_err());
        assert!(checked_escapes("a\\u0041b").is_ok());
    }
}
