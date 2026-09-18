//! Character escaping, shared by every canonicalization algorithm.
//!
//! All four Recommendations ([Canonical XML 1.0 §2.3][c14n10],
//! [Canonical XML 1.1 §2.3][c14n11] and
//! [Canonical XML 2.0 §2.3][c14n20]) escape exactly the same characters, and
//! they escape *different* sets in text content and in attribute values:
//!
//! | context         | escaped                                     |
//! |-----------------|---------------------------------------------|
//! | text            | `&`, `<`, `>`, `#xD`                        |
//! | attribute value | `&`, `<`, `"`, `#x9`, `#xA`, `#xD`          |
//!
//! Note what is *not* escaped: `>` survives inside attribute values, `'`
//! survives everywhere, and `#x9`/`#xA` survive inside text content. Character
//! references are written "in uppercase hexadecimal with no leading zeroes".
//!
//! [c14n10]: https://www.w3.org/TR/xml-c14n#ProcessingModel
//! [c14n11]: https://www.w3.org/TR/xml-c14n11/#ProcessingModel
//! [c14n20]: https://www.w3.org/TR/xml-c14n2/#sec-Processing-Model

/// Appends the canonical form of a text node's string value to `out`.
pub(crate) fn escape_text(out: &mut String, text: &str) {
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '\r' => out.push_str("&#xD;"),
            other => out.push(other),
        }
    }
}

/// Appends the canonical form of an attribute (or namespace node) value to
/// `out`, without the surrounding quotation marks.
pub(crate) fn escape_attribute_value(out: &mut String, value: &str) {
    for ch in value.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '"' => out.push_str("&quot;"),
            '\t' => out.push_str("&#x9;"),
            '\n' => out.push_str("&#xA;"),
            '\r' => out.push_str("&#xD;"),
            other => out.push(other),
        }
    }
}

/// Appends ` name="value"` — the rendering shared by attribute nodes and
/// namespace nodes.
pub(crate) fn write_attribute(out: &mut String, name: &str, value: &str) {
    out.push(' ');
    out.push_str(name);
    out.push_str("=\"");
    escape_attribute_value(out, value);
    out.push('"');
}

/// Appends `<?target data?>`, omitting the separating space when the data is
/// empty.
pub(crate) fn write_processing_instruction(out: &mut String, target: &str, data: &str) {
    out.push_str("<?");
    out.push_str(target);
    if !data.is_empty() {
        out.push(' ');
        out.push_str(data);
    }
    out.push_str("?>");
}

/// Appends `<!--data-->`.
pub(crate) fn write_comment(out: &mut String, data: &str) {
    out.push_str("<!--");
    out.push_str(data);
    out.push_str("-->");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_escapes_the_four_characters_and_nothing_else() {
        let mut out = String::new();
        escape_text(&mut out, "a&b<c>d\"e'f\tg\nh\ri");
        assert_eq!(out, "a&amp;b&lt;c&gt;d\"e'f\tg\nh&#xD;i");
    }

    #[test]
    fn attribute_values_keep_the_closing_angle_bracket() {
        let mut out = String::new();
        escape_attribute_value(&mut out, "a&b<c>d\"e'f\tg\nh\ri");
        assert_eq!(out, "a&amp;b&lt;c>d&quot;e'f&#x9;g&#xA;h&#xD;i");
    }

    #[test]
    fn empty_processing_instruction_data_drops_the_space() {
        let mut out = String::new();
        write_processing_instruction(&mut out, "pi-without-data", "");
        assert_eq!(out, "<?pi-without-data?>");
        out.clear();
        write_processing_instruction(&mut out, "target", "a  b");
        assert_eq!(out, "<?target a  b?>");
    }
}
