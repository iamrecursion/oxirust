//! Small character-class helpers shared by the parser and the serializer.
//!
//! They implement the [XML 1.0 (fifth edition)](https://www.w3.org/TR/xml/#NT-NameStartChar)
//! `NameStartChar` / `NameChar` productions. The serializer uses them to split an IRI into a
//! namespace and a local name; the parser uses [`is_nc_name`] to validate `rdf:ID`,
//! `rdf:nodeID` and `rdf:bagID` values.

/// Returns `true` if `name` matches the Namespaces-in-XML `NCName` production, i.e. an XML
/// `Name` (`NameStartChar (NameChar)*`) that contains no colon.
pub fn is_nc_name(name: &str) -> bool {
    oxixml_qname::is_ncname(name)
}

/// Returns `true` if `c` is an XML [`NameStartChar`](https://www.w3.org/TR/xml/#NT-NameStartChar).
///
/// ```text
/// NameStartChar ::= ":" | [A-Z] | "_" | [a-z] | [#xC0-#xD6] | [#xD8-#xF6] | [#xF8-#x2FF]
///                 | [#x370-#x37D] | [#x37F-#x1FFF] | [#x200C-#x200D] | [#x2070-#x218F]
///                 | [#x2C00-#x2FEF] | [#x3001-#xD7FF] | [#xF900-#xFDCF] | [#xFDF0-#xFFFD]
///                 | [#x10000-#xEFFFF]
/// ```
pub fn is_name_start_char(c: char) -> bool {
    matches!(c,
        ':'
        | 'A'..='Z'
        | '_'
        | 'a'..='z'
        | '\u{00C0}'..='\u{00D6}'
        | '\u{00D8}'..='\u{00F6}'
        | '\u{00F8}'..='\u{02FF}'
        | '\u{0370}'..='\u{037D}'
        | '\u{037F}'..='\u{1FFF}'
        | '\u{200C}'..='\u{200D}'
        | '\u{2070}'..='\u{218F}'
        | '\u{2C00}'..='\u{2FEF}'
        | '\u{3001}'..='\u{D7FF}'
        | '\u{F900}'..='\u{FDCF}'
        | '\u{FDF0}'..='\u{FFFD}'
        | '\u{10000}'..='\u{EFFFF}')
}

/// Returns `true` if `c` is an XML [`NameChar`](https://www.w3.org/TR/xml/#NT-NameChar).
///
/// ```text
/// NameChar ::= NameStartChar | "-" | "." | [0-9] | #xB7 | [#x0300-#x036F] | [#x203F-#x2040]
/// ```
pub fn is_name_char(c: char) -> bool {
    is_name_start_char(c)
        || matches!(c,  '-' | '.' | '0'..='9' | '\u{B7}' | '\u{0300}'..='\u{036F}' | '\u{203F}'..='\u{2040}')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_productions() {
        assert!(!is_nc_name("foo:bar"));
        assert!(is_nc_name("foo"));
        assert!(!is_nc_name("1foo"));
        assert!(!is_nc_name(""));
    }

    #[test]
    fn char_classes() {
        assert!(is_name_start_char('a'));
        assert!(is_name_start_char(':'));
        assert!(!is_name_start_char('-'));
        assert!(is_name_char('-'));
        assert!(is_name_char('7'));
        assert!(!is_name_char('/'));
    }
}
