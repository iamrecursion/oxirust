//! A best-effort scanner for the document type declaration and its internal
//! subset, driving [`DtdHandler`] and [`DeclHandler`] callbacks.
//!
//! [`oxixml_xml`] deliberately does not interpret the `<!DOCTYPE ...>`
//! declaration: it hands back its raw text (everything between `<!DOCTYPE` and
//! the closing `>`) as a single [`Event::DocType`](oxixml_xml::events::Event::DocType).
//! This module re-parses that text against the relevant productions of
//! [XML 1.0 §2.8 and §4.2](https://www.w3.org/TR/xml/#dt-doctype) well enough to
//! report the name, external identifier, and — for the internal subset —
//! `<!ELEMENT>`, `<!ATTLIST>`, `<!ENTITY>` and `<!NOTATION>` declarations.
//!
//! This is a best-effort, non-validating scan: it assumes (but does not enforce)
//! that the input is well-formed, and skips constructs it does not recognize
//! (parameter-entity references, unrecognized markup) rather than failing.
//! Parameter entities are not expanded — a `<!ENTITY % name "...">` declaration
//! is simply skipped, matching the fact that classic SAX2 has no callback for
//! them either.

use std::collections::HashMap;

use oxixml_xml::XmlVersion;

use crate::decl_handler::DeclHandler;
use crate::dtd_handler::DtdHandler;
use crate::entity::replacement_text;
use crate::entity_resolver::EntityResolver;
use crate::error::SaxError;

/// The parsed header of a `<!DOCTYPE ...>` declaration: everything up to (but
/// not including) the internal subset's `[`.
pub(crate) struct DoctypeHeader<'a> {
    pub(crate) name: &'a str,
    pub(crate) public_id: Option<String>,
    pub(crate) system_id: Option<String>,
    pub(crate) internal_subset: Option<&'a str>,
}

/// Parses the raw content of an [`Event::DocType`](oxixml_xml::events::Event::DocType)
/// (the text after `<!DOCTYPE` and its leading white space, up to but excluding
/// the final `>`) into a [`DoctypeHeader`].
pub(crate) fn parse_doctype_header(content: &str) -> DoctypeHeader<'_> {
    let mut cursor = Cursor::new(content);
    let name = cursor.read_name().unwrap_or_default();
    cursor.skip_ws();
    let (public_id, system_id) = read_external_id(&mut cursor);
    cursor.skip_ws();
    let internal_subset = if cursor.peek() == Some(b'[') {
        cursor.advance(1);
        let start = cursor.pos;
        let end = start + find_internal_subset_end(&cursor.input[start..]);
        cursor.pos = end;
        Some(&cursor.input[start..end])
    } else {
        None
    };
    DoctypeHeader {
        name,
        public_id,
        system_id,
        internal_subset,
    }
}

/// Scans `subset` (the text strictly between the internal subset's `[` and `]`)
/// for markup declarations, dispatching them to `dtd`, `decl` and `resolver`,
/// and recording every internal general entity in `entities` so that a `&name;`
/// in content can be replaced by the text it stands for.
///
/// Each `handle_*` helper below takes its handler(s) by value and *hands them
/// back* in its `Ok` result, so this loop can simply reassign `dtd`/`decl`/
/// `resolver` from each call rather than reborrowing them: reassignment is
/// ordinary ownership transfer, not a borrow that has to remain valid across a
/// loop back-edge, so it does not run into the borrow-checker limitations that
/// reborrowing `Option<&mut dyn Trait>` at several call sites inside a loop
/// tends to (each such reborrow can end up needing to be "live" for the whole
/// loop, which then conflicts with every other reborrow of the same variable).
pub(crate) fn dispatch_internal_subset(
    subset: &str,
    mut dtd: Option<&mut dyn DtdHandler>,
    mut decl: Option<&mut dyn DeclHandler>,
    mut resolver: Option<&mut dyn EntityResolver>,
    entities: &mut HashMap<String, String>,
    version: XmlVersion,
) -> Result<(), SaxError> {
    let mut cursor = Cursor::new(subset);
    loop {
        cursor.skip_ws();
        if cursor.is_eof() {
            break;
        }
        if cursor.eat("<!--") {
            match cursor.rest().find("-->") {
                Some(i) => cursor.advance(i + 3),
                None => break,
            }
            continue;
        }
        if cursor.starts_with("<?") {
            cursor.advance(2);
            match cursor.rest().find("?>") {
                Some(i) => cursor.advance(i + 2),
                None => break,
            }
            continue;
        }
        if cursor.eat("<!ELEMENT") {
            decl = handle_element_decl(&mut cursor, decl)?;
            continue;
        }
        if cursor.eat("<!ATTLIST") {
            decl = handle_attlist_decl(&mut cursor, decl)?;
            continue;
        }
        if cursor.eat("<!ENTITY") {
            let handed_back =
                handle_entity_decl(&mut cursor, decl, dtd, resolver, entities, version)?;
            decl = handed_back.0;
            dtd = handed_back.1;
            resolver = handed_back.2;
            continue;
        }
        if cursor.eat("<!NOTATION") {
            let handed_back = handle_notation_decl(&mut cursor, dtd, resolver)?;
            dtd = handed_back.0;
            resolver = handed_back.1;
            continue;
        }
        if cursor.peek() == Some(b'%') {
            // A parameter-entity reference used as a `DeclSep`; we do not expand
            // parameter entities, so just skip past it.
            match cursor.rest().find(';') {
                Some(i) => cursor.advance(i + 1),
                None => break,
            }
            continue;
        }
        // Unrecognized markup: skip defensively to the next `>` so a single
        // malformed declaration cannot hang the scanner in an infinite loop.
        match cursor.rest().find('>') {
            Some(i) => cursor.advance(i + 1),
            None => break,
        }
    }
    Ok(())
}

/// Re-borrows `x` with a lifetime tied to this call only, rather than to `x`'s
/// own lifetime parameter.
///
/// Useful when a single handler is consulted at one call site per event (the
/// common case in [`reader`](crate::reader)'s dispatch loop): a direct
/// `x.as_deref_mut()` reborrow works for that, but going through a plain
/// function call keeps the reborrow's lifetime from being tied to `x` itself,
/// which is what a caller wants whenever the *result* needs to outlive further
/// reborrows of `x` taken later at sibling call sites.
pub(crate) fn reborrow<'a, T: ?Sized>(x: &'a mut Option<&mut T>) -> Option<&'a mut T> {
    match x {
        Some(reference) => Some(&mut **reference),
        None => None,
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Declaration handlers
////////////////////////////////////////////////////////////////////////////////////////////////////

/// `[45] elementdecl ::= '<!ELEMENT' S Name S contentspec S? '>'`. The `<!ELEMENT`
/// keyword has already been consumed. Hands `decl` back so the caller can keep
/// using it for the rest of the internal subset.
fn handle_element_decl<'d>(
    cursor: &mut Cursor<'_>,
    decl: Option<&'d mut dyn DeclHandler>,
) -> Result<Option<&'d mut dyn DeclHandler>, SaxError> {
    let name = cursor.read_name().unwrap_or_default().to_string();
    let model = cursor.take_until_byte(b'>').trim().to_string();
    match decl {
        Some(handler) => {
            handler.element_decl(&name, &model)?;
            Ok(Some(handler))
        }
        None => Ok(None),
    }
}

/// `[52] AttlistDecl ::= '<!ATTLIST' S Name AttDef* S? '>'`. The `<!ATTLIST`
/// keyword has already been consumed. Hands `decl` back; see
/// [`handle_element_decl`].
fn handle_attlist_decl<'d>(
    cursor: &mut Cursor<'_>,
    mut decl: Option<&'d mut dyn DeclHandler>,
) -> Result<Option<&'d mut dyn DeclHandler>, SaxError> {
    let element_name = cursor.read_name().unwrap_or_default().to_string();
    loop {
        cursor.skip_ws();
        match cursor.peek() {
            None => break,
            Some(b'>') => {
                cursor.advance(1);
                break;
            }
            _ => {}
        }
        let Some(attr_name) = cursor.read_name() else {
            break;
        };
        let attr_name = attr_name.to_string();
        cursor.skip_ws();

        let attr_type = if cursor.peek() == Some(b'(') {
            cursor.read_paren_group()
        } else if cursor.eat("NOTATION") {
            cursor.skip_ws();
            format!("NOTATION {}", cursor.read_paren_group())
        } else {
            cursor.read_name().unwrap_or("CDATA").to_string()
        };
        cursor.skip_ws();

        let (mode, value) = if cursor.eat("#REQUIRED") {
            (Some("#REQUIRED".to_string()), None)
        } else if cursor.eat("#IMPLIED") {
            (Some("#IMPLIED".to_string()), None)
        } else if cursor.eat("#FIXED") {
            cursor.skip_ws();
            (
                Some("#FIXED".to_string()),
                cursor.read_quoted().map(str::to_string),
            )
        } else {
            (None, cursor.read_quoted().map(str::to_string))
        };

        decl = match decl {
            Some(handler) => {
                handler.attribute_decl(
                    &element_name,
                    &attr_name,
                    &attr_type,
                    mode.as_deref(),
                    value.as_deref(),
                )?;
                Some(handler)
            }
            None => None,
        };
    }
    Ok(decl)
}

/// `[70] EntityDecl ::= GEDecl | PEDecl`. The `<!ENTITY` keyword has already been
/// consumed. An internal general entity is recorded in `entities`, keyed on its
/// name, with the replacement text of
/// [XML 1.0 §4.5](https://www.w3.org/TR/xml/#intern-replacement) as its value.
/// Hands `(decl, dtd, resolver)` back; see [`handle_element_decl`].
#[allow(clippy::type_complexity)]
fn handle_entity_decl<'d, 't, 'r>(
    cursor: &mut Cursor<'_>,
    decl: Option<&'d mut dyn DeclHandler>,
    dtd: Option<&'t mut dyn DtdHandler>,
    resolver: Option<&'r mut dyn EntityResolver>,
    entities: &mut HashMap<String, String>,
    version: XmlVersion,
) -> Result<
    (
        Option<&'d mut dyn DeclHandler>,
        Option<&'t mut dyn DtdHandler>,
        Option<&'r mut dyn EntityResolver>,
    ),
    SaxError,
> {
    cursor.skip_ws();
    let is_parameter_entity = cursor.peek() == Some(b'%');
    if is_parameter_entity {
        cursor.advance(1);
        cursor.skip_ws();
    }
    let name = cursor.read_name().unwrap_or_default().to_string();
    cursor.skip_ws();

    if is_parameter_entity {
        // Parameter entities have no SAX2 declaration callback; skip to `>`.
        cursor.take_until_byte(b'>');
        return Ok((decl, dtd, resolver));
    }

    if matches!(cursor.peek(), Some(b'"') | Some(b'\'')) {
        // `EntityValue`: an internal general entity.
        let value = cursor.read_quoted().unwrap_or_default().to_string();
        cursor.take_until_byte(b'>');
        // XML 1.0 §4.2: when an entity is declared more than once, the *first*
        // declaration binds.
        entities
            .entry(name.clone())
            .or_insert_with(|| replacement_text(&value, version));
        let decl = match decl {
            Some(handler) => {
                handler.internal_entity_decl(&name, &value)?;
                Some(handler)
            }
            None => None,
        };
        return Ok((decl, dtd, resolver));
    }

    // `ExternalID NDataDecl?`: an external general entity, parsed or unparsed.
    let (public_id, system_id) = read_external_id(cursor);
    cursor.skip_ws();
    let notation = if cursor.eat("NDATA") {
        cursor.skip_ws();
        cursor.read_name().map(str::to_string)
    } else {
        None
    };
    cursor.take_until_byte(b'>');

    let system_id = system_id.unwrap_or_default();
    let resolver = match resolver {
        Some(handler) => {
            let _ = handler.resolve_entity(public_id.as_deref(), Some(system_id.as_str()));
            Some(handler)
        }
        None => None,
    };
    match notation {
        Some(notation_name) => {
            let dtd = match dtd {
                Some(handler) => {
                    handler.unparsed_entity_decl(
                        &name,
                        public_id.as_deref(),
                        &system_id,
                        &notation_name,
                    )?;
                    Some(handler)
                }
                None => None,
            };
            Ok((decl, dtd, resolver))
        }
        None => {
            let decl = match decl {
                Some(handler) => {
                    handler.external_entity_decl(&name, public_id.as_deref(), &system_id)?;
                    Some(handler)
                }
                None => None,
            };
            Ok((decl, dtd, resolver))
        }
    }
}

/// `[82] NotationDecl ::= '<!NOTATION' S Name S (ExternalID | PublicID) S? '>'`.
/// The `<!NOTATION` keyword has already been consumed. Hands `(dtd, resolver)`
/// back; see [`handle_element_decl`].
#[allow(clippy::type_complexity)]
fn handle_notation_decl<'t, 'r>(
    cursor: &mut Cursor<'_>,
    dtd: Option<&'t mut dyn DtdHandler>,
    resolver: Option<&'r mut dyn EntityResolver>,
) -> Result<
    (
        Option<&'t mut dyn DtdHandler>,
        Option<&'r mut dyn EntityResolver>,
    ),
    SaxError,
> {
    let name = cursor.read_name().unwrap_or_default().to_string();
    cursor.skip_ws();

    let (public_id, system_id) = if cursor.eat("PUBLIC") {
        cursor.skip_ws();
        let public_id = cursor.read_quoted().map(str::to_string);
        cursor.skip_ws();
        // `PublicID` has no `SystemLiteral`; only consume one if present.
        let system_id = if matches!(cursor.peek(), Some(b'"') | Some(b'\'')) {
            cursor.read_quoted().map(str::to_string)
        } else {
            None
        };
        (public_id, system_id)
    } else if cursor.eat("SYSTEM") {
        cursor.skip_ws();
        (None, cursor.read_quoted().map(str::to_string))
    } else {
        (None, None)
    };
    cursor.take_until_byte(b'>');

    let resolver = match resolver {
        Some(handler) => {
            let _ = handler.resolve_entity(public_id.as_deref(), system_id.as_deref());
            Some(handler)
        }
        None => None,
    };
    let dtd = match dtd {
        Some(handler) => {
            handler.notation_decl(&name, public_id.as_deref(), system_id.as_deref())?;
            Some(handler)
        }
        None => None,
    };
    Ok((dtd, resolver))
}

/// `[75] ExternalID ::= 'SYSTEM' S SystemLiteral | 'PUBLIC' S PubidLiteral S SystemLiteral`.
fn read_external_id(cursor: &mut Cursor<'_>) -> (Option<String>, Option<String>) {
    if cursor.eat("PUBLIC") {
        cursor.skip_ws();
        let public_id = cursor.read_quoted().map(str::to_string);
        cursor.skip_ws();
        let system_id = cursor.read_quoted().map(str::to_string);
        (public_id, system_id)
    } else if cursor.eat("SYSTEM") {
        cursor.skip_ws();
        (None, cursor.read_quoted().map(str::to_string))
    } else {
        (None, None)
    }
}

/// Finds the byte offset (relative to `s`, which starts right after the internal
/// subset's opening `[`) of its closing `]`, skipping over quoted literals and
/// comments so that a `]` inside either does not end the scan early.
fn find_internal_subset_end(s: &str) -> usize {
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut quote: Option<u8> = None;
    while i < bytes.len() {
        let b = bytes[i];
        if let Some(q) = quote {
            if b == q {
                quote = None;
            }
            i += 1;
            continue;
        }
        match b {
            b'"' | b'\'' => {
                quote = Some(b);
                i += 1;
            }
            b'<' if s[i..].starts_with("<!--") => match s[i + 4..].find("-->") {
                Some(end) => i += 4 + end + 3,
                None => return bytes.len(),
            },
            b']' => return i,
            _ => i += 1,
        }
    }
    bytes.len()
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Cursor
////////////////////////////////////////////////////////////////////////////////////////////////////

/// A minimal byte-position cursor over a `&str`.
///
/// All delimiters this scanner looks for (white space, `<`, `>`, `"`, `'`, `(`,
/// `)`, `[`, `]`, `%`) are ASCII, and ASCII bytes never occur inside a multi-byte
/// UTF-8 sequence, so byte-offset scanning never splits a character even though
/// `Name`s may contain non-ASCII characters.
struct Cursor<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Cursor<'a> {
    const fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn rest(&self) -> &'a str {
        &self.input[self.pos..]
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.input.len()
    }

    fn peek(&self) -> Option<u8> {
        self.input.as_bytes().get(self.pos).copied()
    }

    fn advance(&mut self, n: usize) {
        self.pos = (self.pos + n).min(self.input.len());
    }

    fn starts_with(&self, pattern: &str) -> bool {
        self.rest().starts_with(pattern)
    }

    /// Consumes `pattern` (case-sensitive) if the cursor is at it, returning
    /// whether it matched.
    fn eat(&mut self, pattern: &str) -> bool {
        if self.starts_with(pattern) {
            self.advance(pattern.len());
            true
        } else {
            false
        }
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\r' | b'\n')) {
            self.pos += 1;
        }
    }

    /// Reads an XML `Name` token: skips leading white space, then everything up
    /// to the next white space, `>`, `(` or `)`.
    fn read_name(&mut self) -> Option<&'a str> {
        self.skip_ws();
        let start = self.pos;
        let bytes = self.input.as_bytes();
        while let Some(&b) = bytes.get(self.pos) {
            if matches!(b, b' ' | b'\t' | b'\r' | b'\n' | b'>' | b'(' | b')') {
                break;
            }
            self.pos += 1;
        }
        if self.pos == start {
            None
        } else {
            Some(&self.input[start..self.pos])
        }
    }

    /// Reads a single- or double-quoted literal, skipping leading white space.
    /// Returns `None` (without consuming anything but leading white space) if the
    /// next non-space byte is not a quote, or if the literal is never closed.
    fn read_quoted(&mut self) -> Option<&'a str> {
        self.skip_ws();
        let quote = self.peek()?;
        if quote != b'"' && quote != b'\'' {
            return None;
        }
        self.pos += 1;
        let start = self.pos;
        let bytes = self.input.as_bytes();
        while let Some(&b) = bytes.get(self.pos) {
            if b == quote {
                let value = &self.input[start..self.pos];
                self.pos += 1;
                return Some(value);
            }
            self.pos += 1;
        }
        self.pos = self.input.len();
        None
    }

    /// Reads a parenthesized group (an `Enumeration` or `NotationType` list),
    /// including the parentheses, respecting nesting. Returns an empty string if
    /// the cursor is not at `(`.
    fn read_paren_group(&mut self) -> String {
        if self.peek() != Some(b'(') {
            return String::new();
        }
        let start = self.pos;
        self.pos += 1;
        let mut depth = 1usize;
        let bytes = self.input.as_bytes();
        while let Some(&b) = bytes.get(self.pos) {
            self.pos += 1;
            match b {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
        }
        self.input[start..self.pos].to_string()
    }

    /// Consumes and returns everything up to (not including) the next `delim`
    /// byte, consuming `delim` itself too. If `delim` is never found, consumes to
    /// the end of input and returns everything that was left.
    fn take_until_byte(&mut self, delim: u8) -> &'a str {
        let rest = self.rest();
        match rest.as_bytes().iter().position(|&b| b == delim) {
            Some(i) => {
                let s = &rest[..i];
                self.pos += i + 1;
                s
            }
            None => {
                self.pos = self.input.len();
                rest
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_without_external_id_or_subset() {
        let header = parse_doctype_header("html");
        assert_eq!(header.name, "html");
        assert_eq!(header.public_id, None);
        assert_eq!(header.system_id, None);
        assert!(header.internal_subset.is_none());
    }

    #[test]
    fn header_with_system_id() {
        let header = parse_doctype_header(r#"root SYSTEM "root.dtd""#);
        assert_eq!(header.name, "root");
        assert_eq!(header.system_id.as_deref(), Some("root.dtd"));
        assert_eq!(header.public_id, None);
    }

    #[test]
    fn header_with_public_id() {
        let header = parse_doctype_header(
            r#"html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "xhtml1-strict.dtd""#,
        );
        assert_eq!(header.name, "html");
        assert_eq!(
            header.public_id.as_deref(),
            Some("-//W3C//DTD XHTML 1.0 Strict//EN")
        );
        assert_eq!(header.system_id.as_deref(), Some("xhtml1-strict.dtd"));
    }

    #[test]
    fn header_with_internal_subset() {
        let header = parse_doctype_header(r#"root [ <!ENTITY foo "bar"> ]"#);
        assert_eq!(header.name, "root");
        assert_eq!(header.internal_subset, Some(r#" <!ENTITY foo "bar"> "#));
    }

    #[test]
    fn internal_subset_end_skips_quoted_bracket() {
        let subset = r#" <!ENTITY foo "]"> ]"#;
        let end = find_internal_subset_end(subset);
        assert_eq!(&subset[..end], r#" <!ENTITY foo "]"> "#);
    }

    #[allow(clippy::type_complexity)]
    struct Recorder {
        elements: Vec<(String, String)>,
        attributes: Vec<(String, String, String, Option<String>, Option<String>)>,
        internal_entities: Vec<(String, String)>,
        external_entities: Vec<(String, Option<String>, String)>,
        notations: Vec<(String, Option<String>, Option<String>)>,
        unparsed_entities: Vec<(String, Option<String>, String, String)>,
    }

    impl Recorder {
        fn new() -> Self {
            Self {
                elements: Vec::new(),
                attributes: Vec::new(),
                internal_entities: Vec::new(),
                external_entities: Vec::new(),
                notations: Vec::new(),
                unparsed_entities: Vec::new(),
            }
        }
    }

    impl DeclHandler for Recorder {
        fn element_decl(&mut self, name: &str, model: &str) -> Result<(), SaxError> {
            self.elements.push((name.to_string(), model.to_string()));
            Ok(())
        }

        fn attribute_decl(
            &mut self,
            element_name: &str,
            attribute_name: &str,
            attr_type: &str,
            mode: Option<&str>,
            value: Option<&str>,
        ) -> Result<(), SaxError> {
            self.attributes.push((
                element_name.to_string(),
                attribute_name.to_string(),
                attr_type.to_string(),
                mode.map(str::to_string),
                value.map(str::to_string),
            ));
            Ok(())
        }

        fn internal_entity_decl(&mut self, name: &str, value: &str) -> Result<(), SaxError> {
            self.internal_entities
                .push((name.to_string(), value.to_string()));
            Ok(())
        }

        fn external_entity_decl(
            &mut self,
            name: &str,
            public_id: Option<&str>,
            system_id: &str,
        ) -> Result<(), SaxError> {
            self.external_entities.push((
                name.to_string(),
                public_id.map(str::to_string),
                system_id.to_string(),
            ));
            Ok(())
        }
    }

    impl DtdHandler for Recorder {
        fn notation_decl(
            &mut self,
            name: &str,
            public_id: Option<&str>,
            system_id: Option<&str>,
        ) -> Result<(), SaxError> {
            self.notations.push((
                name.to_string(),
                public_id.map(str::to_string),
                system_id.map(str::to_string),
            ));
            Ok(())
        }

        fn unparsed_entity_decl(
            &mut self,
            name: &str,
            public_id: Option<&str>,
            system_id: &str,
            notation_name: &str,
        ) -> Result<(), SaxError> {
            self.unparsed_entities.push((
                name.to_string(),
                public_id.map(str::to_string),
                system_id.to_string(),
                notation_name.to_string(),
            ));
            Ok(())
        }
    }

    #[test]
    fn dispatches_element_and_attlist_decls() {
        let subset = r#"
            <!ELEMENT root (child)*>
            <!ATTLIST root
                id ID #REQUIRED
                kind (a|b|c) "a"
                lang CDATA #FIXED "en"
                note CDATA #IMPLIED
            >
        "#;
        let mut recorder = Recorder::new();
        dispatch_internal_subset(
            subset,
            None,
            Some(&mut recorder),
            None,
            &mut HashMap::new(),
            XmlVersion::Implicit1_0,
        )
        .expect("valid subset");

        assert_eq!(
            recorder.elements,
            vec![("root".to_string(), "(child)*".to_string())]
        );
        assert_eq!(
            recorder.attributes,
            vec![
                (
                    "root".to_string(),
                    "id".to_string(),
                    "ID".to_string(),
                    Some("#REQUIRED".to_string()),
                    None
                ),
                (
                    "root".to_string(),
                    "kind".to_string(),
                    "(a|b|c)".to_string(),
                    None,
                    Some("a".to_string())
                ),
                (
                    "root".to_string(),
                    "lang".to_string(),
                    "CDATA".to_string(),
                    Some("#FIXED".to_string()),
                    Some("en".to_string())
                ),
                (
                    "root".to_string(),
                    "note".to_string(),
                    "CDATA".to_string(),
                    Some("#IMPLIED".to_string()),
                    None
                ),
            ]
        );
    }

    #[test]
    fn dispatches_internal_and_external_entity_decls() {
        let subset = r#"
            <!ENTITY copy "(c)">
            <!ENTITY logo SYSTEM "logo.png" NDATA png>
            <!ENTITY chapter PUBLIC "-//example//chapter//EN" "chapter.xml">
            <!NOTATION png SYSTEM "image/png">
            <!ENTITY % param "internal only, skipped">
        "#;
        let mut recorder = Recorder::new();
        {
            let dtd: &mut dyn DtdHandler = &mut recorder;
            dispatch_internal_subset(
                subset,
                Some(dtd),
                None,
                None,
                &mut HashMap::new(),
                XmlVersion::Implicit1_0,
            )
            .expect("valid subset");
        }
        assert_eq!(
            recorder.unparsed_entities,
            vec![(
                "logo".to_string(),
                None,
                "logo.png".to_string(),
                "png".to_string()
            )]
        );
        assert_eq!(
            recorder.notations,
            vec![("png".to_string(), None, Some("image/png".to_string()))]
        );

        let mut recorder = Recorder::new();
        {
            let decl: &mut dyn DeclHandler = &mut recorder;
            dispatch_internal_subset(
                subset,
                None,
                Some(decl),
                None,
                &mut HashMap::new(),
                XmlVersion::Implicit1_0,
            )
            .expect("valid subset");
        }
        assert_eq!(
            recorder.internal_entities,
            vec![("copy".to_string(), "(c)".to_string())]
        );
        assert_eq!(
            recorder.external_entities,
            vec![(
                "chapter".to_string(),
                Some("-//example//chapter//EN".to_string()),
                "chapter.xml".to_string()
            )]
        );
    }

    #[test]
    fn skips_comments_and_pis_inside_subset() {
        let subset = r#"
            <!-- a comment with a stray > inside -->
            <?pi some data ?>
            <!ELEMENT root EMPTY>
        "#;
        let mut recorder = Recorder::new();
        dispatch_internal_subset(
            subset,
            None,
            Some(&mut recorder),
            None,
            &mut HashMap::new(),
            XmlVersion::Implicit1_0,
        )
        .expect("valid subset");
        assert_eq!(
            recorder.elements,
            vec![("root".to_string(), "EMPTY".to_string())]
        );
    }
}
