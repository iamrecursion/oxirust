//! A small, tolerant scanner for the document type declaration.
//!
//! The builder needs three things from a `<!DOCTYPE …>`: the declared name and
//! external identifiers (which become the
//! [`DocumentType`](crate::NodeKind::DocumentType) node), the general entities
//! declared in the internal subset (so that `&name;` can be replaced), and the
//! attributes declared with the `ID` type (so that
//! [`Document::get_element_by_id`](crate::Document::get_element_by_id) works).
//!
//! This is *not* a DTD parser: parameter entities are not expanded, conditional
//! sections are not evaluated, and the external subset is never retrieved — an
//! XML processor is explicitly allowed not to read it
//! ([XML 1.0 §5.1](https://www.w3.org/TR/xml/#sec-conditional-sec)). Everything
//! the scanner does not understand is skipped, and the internal subset is kept
//! verbatim so that serializing the document reproduces it exactly.

use oxixml_xml::XmlVersion;

use crate::entity::replacement_text;

/// What the scanner extracted from a `<!DOCTYPE …>` declaration.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct Doctype {
    pub(crate) name: String,
    pub(crate) public_id: Option<String>,
    pub(crate) system_id: Option<String>,
    pub(crate) internal_subset: Option<String>,
}

/// Splits the body of a document type declaration — everything between
/// `<!DOCTYPE` and the final `>` — into its parts.
pub(crate) fn parse_doctype(body: &str) -> Doctype {
    let mut doctype = Doctype::default();
    let bytes = body.as_bytes();
    let mut cursor = skip_space(bytes, 0);

    let start = cursor;
    while cursor < bytes.len() && !is_space(bytes[cursor]) && bytes[cursor] != b'[' {
        cursor += 1;
    }
    doctype.name = body[start..cursor].to_owned();
    cursor = skip_space(bytes, cursor);

    if body[cursor..].starts_with("PUBLIC") {
        cursor = skip_space(bytes, cursor + "PUBLIC".len());
        let (public, next) = read_literal(body, cursor);
        doctype.public_id = public;
        cursor = skip_space(bytes, next);
        let (system, next) = read_literal(body, cursor);
        doctype.system_id = system;
        cursor = skip_space(bytes, next);
    } else if body[cursor..].starts_with("SYSTEM") {
        cursor = skip_space(bytes, cursor + "SYSTEM".len());
        let (system, next) = read_literal(body, cursor);
        doctype.system_id = system;
        cursor = skip_space(bytes, next);
    }

    if bytes.get(cursor) == Some(&b'[') {
        if let Some(end) = body.rfind(']') {
            if end > cursor {
                doctype.internal_subset = Some(body[cursor + 1..end].to_owned());
            }
        }
    }
    doctype
}

/// A declaration found in an internal subset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Declaration {
    /// `<!ENTITY name "replacement text">` — an internal general entity.
    Entity {
        /// The entity name.
        name: String,
        /// The *replacement text* of
        /// [XML 1.0 §4.5](https://www.w3.org/TR/xml/#intern-replacement): the
        /// literal entity value with its character references replaced by the
        /// characters they denote, and its general entity references left as
        /// written, to be resolved wherever the entity is used.
        value: String,
    },
    /// `<!ENTITY name SYSTEM "…">` or `<!ENTITY name PUBLIC "…" "…">` — an
    /// external *parsed* general entity, whose replacement text lives somewhere
    /// else and is only retrieved when the caller supplies a resolver.
    ExternalEntity {
        /// The entity name.
        name: String,
        /// The public identifier, when the declaration used `PUBLIC`.
        public_id: Option<String>,
        /// The system identifier.
        system_id: String,
    },
    /// `<!ATTLIST element attribute ID …>` — an attribute of type `ID`.
    IdAttribute {
        /// The qualified name of the element the declaration applies to.
        element: String,
        /// The qualified name of the attribute.
        attribute: String,
    },
}

/// Scans an internal subset for the declarations the builder cares about.
///
/// `version` selects the `Char` production applied to the character references
/// inside an entity's literal value.
pub(crate) fn scan_internal_subset(subset: &str, version: XmlVersion) -> Vec<Declaration> {
    let mut out = Vec::new();
    let bytes = subset.as_bytes();
    let mut cursor = 0;

    while cursor < bytes.len() {
        if subset[cursor..].starts_with("<!--") {
            cursor = subset[cursor..]
                .find("-->")
                .map_or(bytes.len(), |offset| cursor + offset + 3);
            continue;
        }
        if subset[cursor..].starts_with("<?") {
            cursor = subset[cursor..]
                .find("?>")
                .map_or(bytes.len(), |offset| cursor + offset + 2);
            continue;
        }
        if subset[cursor..].starts_with("<!ENTITY") {
            let (body, next) = read_declaration(subset, cursor + "<!ENTITY".len());
            if let Some(entity) = parse_entity(&body, version) {
                out.push(entity);
            }
            cursor = next;
            continue;
        }
        if subset[cursor..].starts_with("<!ATTLIST") {
            let (body, next) = read_declaration(subset, cursor + "<!ATTLIST".len());
            parse_attlist(&body, &mut out);
            cursor = next;
            continue;
        }
        cursor += next_char_len(bytes, cursor);
    }
    out
}

/// Reads a markup declaration body up to the terminating `>`, honouring quoted
/// literals so that a `>` inside one does not end it. Returns the body and the
/// offset just past the `>`.
fn read_declaration(text: &str, start: usize) -> (String, usize) {
    let bytes = text.as_bytes();
    let mut cursor = start;
    let mut quote: Option<u8> = None;
    while cursor < bytes.len() {
        let byte = bytes[cursor];
        match quote {
            Some(open) => {
                if byte == open {
                    quote = None;
                }
            }
            None => match byte {
                b'"' | b'\'' => quote = Some(byte),
                b'>' => return (text[start..cursor].to_owned(), cursor + 1),
                _ => {}
            },
        }
        cursor += next_char_len(bytes, cursor);
    }
    (text[start..].to_owned(), bytes.len())
}

fn parse_entity(body: &str, version: XmlVersion) -> Option<Declaration> {
    let tokens = tokenize(body);
    let mut iter = tokens.iter();
    let first = iter.next()?;
    // `<!ENTITY % name …>` declares a parameter entity, which this scanner
    // deliberately ignores.
    if first.text.starts_with('%') {
        return None;
    }
    let name = first.text.clone();
    let value = iter.next()?;
    if !value.quoted {
        // `SYSTEM` / `PUBLIC`: an external entity. Its replacement text lives
        // elsewhere, so only its identifiers are recorded here.
        return parse_external_entity(name, value, iter);
    }
    // An unparsed entity (`… NDATA name`) has no replacement text either.
    if iter.next().is_some() {
        return None;
    }
    Some(Declaration::Entity {
        name,
        value: replacement_text(&value.text, version),
    })
}

/// Completes an `<!ENTITY name SYSTEM …>` / `<!ENTITY name PUBLIC …>`
/// declaration, whose keyword the caller has already read.
///
/// An *unparsed* entity — one with an `NDATA` notation — is deliberately not
/// reported: it can only be the value of an `ENTITY` attribute, never the target
/// of a reference, so nothing would ever retrieve it.
fn parse_external_entity<'t>(
    name: String,
    keyword: &Token,
    mut rest: impl Iterator<Item = &'t Token>,
) -> Option<Declaration> {
    let (public_id, system_id) = match keyword.text.as_str() {
        "SYSTEM" => (None, rest.next()?),
        "PUBLIC" => {
            let public = rest.next()?;
            (Some(public.text.clone()), rest.next()?)
        }
        _ => return None,
    };
    if !system_id.quoted {
        return None;
    }
    if rest.next().is_some() {
        // `NDATA name`: an unparsed entity.
        return None;
    }
    Some(Declaration::ExternalEntity {
        name,
        public_id,
        system_id: system_id.text.clone(),
    })
}

fn parse_attlist(body: &str, out: &mut Vec<Declaration>) {
    let tokens = tokenize(body);
    let mut iter = tokens.into_iter();
    let Some(element) = iter.next() else { return };
    let element = element.text;

    loop {
        let Some(attribute) = iter.next() else { return };
        let Some(kind) = iter.next() else { return };
        if kind.text == "NOTATION" {
            // The enumeration group follows the `NOTATION` keyword.
            if iter.next().is_none() {
                return;
            }
        }
        if kind.text == "ID" {
            out.push(Declaration::IdAttribute {
                element: element.clone(),
                attribute: attribute.text,
            });
        }
        let Some(default) = iter.next() else { return };
        if default.text == "#FIXED" && iter.next().is_none() {
            return;
        }
    }
}

/// One token of a markup declaration: a quoted literal, a parenthesized group,
/// or a run of characters delimited by white space.
struct Token {
    text: String,
    quoted: bool,
}

fn tokenize(body: &str) -> Vec<Token> {
    let bytes = body.as_bytes();
    let mut tokens = Vec::new();
    let mut cursor = 0;
    while cursor < bytes.len() {
        cursor = skip_space(bytes, cursor);
        if cursor >= bytes.len() {
            break;
        }
        match bytes[cursor] {
            quote @ (b'"' | b'\'') => {
                let start = cursor + 1;
                let mut end = start;
                while end < bytes.len() && bytes[end] != quote {
                    end += next_char_len(bytes, end);
                }
                tokens.push(Token {
                    text: body[start..end.min(bytes.len())].to_owned(),
                    quoted: true,
                });
                cursor = (end + 1).min(bytes.len());
            }
            b'(' => {
                let start = cursor;
                let mut end = cursor;
                let mut depth = 0_usize;
                while end < bytes.len() {
                    match bytes[end] {
                        b'(' => depth += 1,
                        b')' => {
                            depth -= 1;
                            if depth == 0 {
                                end += 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                    end += next_char_len(bytes, end);
                }
                tokens.push(Token {
                    text: body[start..end.min(bytes.len())].to_owned(),
                    quoted: false,
                });
                cursor = end;
            }
            _ => {
                let start = cursor;
                while cursor < bytes.len()
                    && !is_space(bytes[cursor])
                    && !matches!(bytes[cursor], b'"' | b'\'' | b'(')
                {
                    cursor += next_char_len(bytes, cursor);
                }
                tokens.push(Token {
                    text: body[start..cursor].to_owned(),
                    quoted: false,
                });
            }
        }
    }
    tokens
}

fn read_literal(text: &str, start: usize) -> (Option<String>, usize) {
    let bytes = text.as_bytes();
    let Some(&quote) = bytes.get(start) else {
        return (None, start);
    };
    if quote != b'"' && quote != b'\'' {
        return (None, start);
    }
    let mut end = start + 1;
    while end < bytes.len() && bytes[end] != quote {
        end += next_char_len(bytes, end);
    }
    (
        Some(text[start + 1..end.min(bytes.len())].to_owned()),
        (end + 1).min(bytes.len()),
    )
}

const fn is_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\r' | b'\n')
}

fn skip_space(bytes: &[u8], mut cursor: usize) -> usize {
    while cursor < bytes.len() && is_space(bytes[cursor]) {
        cursor += 1;
    }
    cursor
}

/// Returns the length of the UTF-8 sequence starting at `cursor`, so that byte
/// offsets always land on character boundaries.
fn next_char_len(bytes: &[u8], cursor: usize) -> usize {
    match bytes.get(cursor) {
        None => 1,
        Some(&byte) if byte < 0x80 => 1,
        Some(&byte) if byte < 0xE0 => 2,
        Some(&byte) if byte < 0xF0 => 3,
        Some(_) => 4,
    }
}
