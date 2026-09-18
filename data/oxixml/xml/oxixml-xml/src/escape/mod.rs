//! Manage XML character escapes.
//!
//! This module mirrors quick-xml 0.41 `escape`: [`escape`], [`partial_escape`]
//! and [`minimal_escape`] replace XML special characters with entity references,
//! while [`unescape`] / [`unescape_with`] reverse the process. Predefined XML
//! entities are resolved by [`resolve_predefined_entity`]; with the `escape-html`
//! feature enabled, the full set of HTML5 named references is recognized instead.

use std::borrow::Cow;
use std::fmt;
use std::num::ParseIntError;
use std::ops::Range;
use std::slice::Iter;

use oxixml_support::bytes::{memchr, memchr3};

#[cfg(feature = "escape-html")]
mod html5;
#[cfg(feature = "escape-html")]
#[cfg_attr(docsrs, doc(cfg(feature = "escape-html")))]
pub use html5::resolve_html5_entity;

/// Error while parsing a character reference (`&#<dec-number>;` or `&#x<hex-number>;`).
#[derive(Clone, Debug, PartialEq)]
pub enum ParseCharRefError {
    /// The number contains a sign character (`+` or `-`), which is not allowed.
    UnexpectedSign,
    /// The number cannot be parsed due to non-numeric characters or numeric overflow.
    InvalidNumber(ParseIntError),
    /// The character reference does not represent a valid Unicode code point.
    InvalidCodepoint(u32),
    /// The character reference expanded to a character not permitted in XML.
    ///
    /// Currently only the `0x0` character produces this error.
    IllegalCharacter(u32),
}

impl fmt::Display for ParseCharRefError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::UnexpectedSign => f.write_str("unexpected number sign"),
            Self::InvalidNumber(e) => e.fmt(f),
            Self::InvalidCodepoint(n) => write!(f, "`{}` is not a valid codepoint", n),
            Self::IllegalCharacter(n) => write!(f, "0x{:x} character is not permitted in XML", n),
        }
    }
}

impl std::error::Error for ParseCharRefError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidNumber(e) => Some(e),
            _ => None,
        }
    }
}

/// Error for XML escape / unescape.
#[derive(Clone, Debug, PartialEq)]
pub enum EscapeError {
    /// The referenced entity is unknown to the parser. Contains the byte range of
    /// the entity name and the name itself.
    UnrecognizedEntity(Range<usize>, String),
    /// Cannot find the `;` that terminates an entity started by `&`.
    UnterminatedEntity(Range<usize>),
    /// Parsing a character reference (`&#<dec-number>;` or `&#x<hex-number>;`) was
    /// unsuccessful because not all characters were decimal or hexadecimal digits.
    InvalidCharRef(ParseCharRefError),
    /// More than the maximum number of nested entities were expanded during
    /// attribute-value normalization.
    TooManyNestedEntities,
}

impl fmt::Display for EscapeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::UnrecognizedEntity(rge, res) => {
                write!(f, "at {:?}: unrecognized entity `{}`", rge, res)
            }
            Self::UnterminatedEntity(e) => write!(
                f,
                "Error while escaping character at range {:?}: Cannot find ';' after '&'",
                e
            ),
            Self::InvalidCharRef(e) => write!(f, "invalid character reference: {}", e),
            Self::TooManyNestedEntities => {
                f.write_str("too many nested entities in an attribute value")
            }
        }
    }
}

impl std::error::Error for EscapeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidCharRef(e) => Some(e),
            _ => None,
        }
    }
}

/// Escapes an `&str`, replacing all XML special characters (`<`, `>`, `&`, `'`,
/// `"`) with their corresponding XML escape sequences.
///
/// | Character | Replacement |
/// |-----------|-------------|
/// | `<`       | `&lt;`      |
/// | `>`       | `&gt;`      |
/// | `&`       | `&amp;`     |
/// | `'`       | `&apos;`    |
/// | `"`       | `&quot;`    |
pub fn escape<'a>(raw: impl Into<Cow<'a, str>>) -> Cow<'a, str> {
    _escape(raw, |ch| matches!(ch, b'<' | b'>' | b'&' | b'\'' | b'\"'))
}

/// Escapes an `&str`, replacing the XML special characters `<`, `>` and `&`.
///
/// This should only be used for escaping text content, where the quote characters
/// `"` and `'` may be left unescaped.
pub fn partial_escape<'a>(raw: impl Into<Cow<'a, str>>) -> Cow<'a, str> {
    _escape(raw, |ch| matches!(ch, b'<' | b'>' | b'&'))
}

/// Escapes an `&str`, replacing only the characters (`<` and `&`) that the XML
/// standard [requires] to be escaped in text content and attribute values.
///
/// [requires]: https://www.w3.org/TR/xml11/#syntax
pub fn minimal_escape<'a>(raw: impl Into<Cow<'a, str>>) -> Cow<'a, str> {
    _escape(raw, |ch| matches!(ch, b'<' | b'&'))
}

/// Writes `value[from..to]` followed by the escape sequence for `value[to]`.
pub(crate) fn escape_char<W>(writer: &mut W, value: &str, from: usize, to: usize) -> fmt::Result
where
    W: fmt::Write,
{
    writer.write_str(&value[from..to])?;
    match value.as_bytes()[to] {
        b'<' => writer.write_str("&lt;")?,
        b'>' => writer.write_str("&gt;")?,
        b'\'' => writer.write_str("&apos;")?,
        b'&' => writer.write_str("&amp;")?,
        b'"' => writer.write_str("&quot;")?,

        // These escapes handle characters that must be escaped in elements of
        // `xs:list`s, because they act as delimiters of list elements.
        b'\t' => writer.write_str("&#9;")?,
        b'\n' => writer.write_str("&#10;")?,
        b'\r' => writer.write_str("&#13;")?,
        b' ' => writer.write_str("&#32;")?,
        _ => {
            unreachable!("only '<', '>', '\\'', '&', '\"', '\\t', '\\r', '\\n' and ' ' are escaped")
        }
    }
    Ok(())
}

/// Escapes an `&str`, replacing the subset of XML special characters selected by
/// `escape_chars` with their corresponding XML escape sequences.
fn _escape<'a, F: Fn(u8) -> bool>(raw: impl Into<Cow<'a, str>>, escape_chars: F) -> Cow<'a, str> {
    let raw = raw.into();
    let bytes = raw.as_bytes();
    let mut escaped: Option<String> = None;
    let mut iter = bytes.iter();
    let mut pos = 0;
    while let Some(i) = iter.position(|&b| escape_chars(b)) {
        let escaped = escaped.get_or_insert_with(|| String::with_capacity(raw.len()));
        let new_pos = pos + i;
        // Writing into a `String` is infallible.
        let _ = escape_char(escaped, &raw, pos, new_pos);
        pos = new_pos + 1;
    }

    if let Some(mut escaped) = escaped {
        if let Some(rest) = raw.get(pos..) {
            escaped.push_str(rest);
        }
        Cow::Owned(escaped)
    } else {
        raw
    }
}

/// Unescapes an `&str`, replacing all XML escapes (`&...;`) with their
/// corresponding value.
///
/// With the `escape-html` feature enabled, all [HTML5 escapes] are recognized.
///
/// [HTML5 escapes]: https://dev.w3.org/html5/html-author/charref
pub fn unescape(raw: &str) -> Result<Cow<'_, str>, EscapeError> {
    unescape_with(raw, resolve_predefined_entity)
}

/// Unescapes an `&str`, replacing all XML escapes (`&...;`) with their
/// corresponding value, using `resolve_entity` for named entities.
///
/// Predefined entities are resolved *after* `resolve_entity`, allowing custom
/// dialects to override them. Character references (`&#hh;`) are always resolved
/// before `resolve_entity` is called and cannot be overridden. Entities are not
/// resolved recursively.
///
/// # Example
///
/// ```
/// use oxixml_xml::escape::{resolve_xml_entity, unescape_with};
///
/// let override_named_entities = |entity: &str| match entity {
///     "lt" => Some("FOO"),
///     "gt" => Some("BAR"),
///     "baz" => Some("&lt;"),
///     _ => resolve_xml_entity(entity),
/// };
///
/// assert_eq!(
///     unescape_with("&amp;&lt;test&gt;&baz;", override_named_entities).expect("doc example"),
///     "&FOOtestBAR&lt;"
/// );
/// ```
pub fn unescape_with<'input, 'entity, F>(
    raw: &'input str,
    mut resolve_entity: F,
) -> Result<Cow<'input, str>, EscapeError>
where
    // The lifetime of the output comes from a capture or is `'static`.
    F: FnMut(&str) -> Option<&'entity str>,
{
    let bytes = raw.as_bytes();
    let mut unescaped: Option<String> = None;
    let mut last_end = 0;
    // Positions of every `&` and `;`, in order.
    let mut iter = bytes
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| (b == b'&' || b == b';').then_some(i));
    while let Some(start) = iter.by_ref().find(|p| bytes[*p] == b'&') {
        match iter.next() {
            Some(end) if bytes[end] == b';' => {
                let unescaped = unescaped.get_or_insert_with(|| String::with_capacity(raw.len()));
                unescaped.push_str(&raw[last_end..start]);

                let pat = &raw[start + 1..end];
                if let Some(entity) = pat.strip_prefix('#') {
                    let codepoint = parse_number(entity).map_err(EscapeError::InvalidCharRef)?;
                    unescaped.push_str(codepoint.encode_utf8(&mut [0u8; 4]));
                } else if let Some(value) = resolve_entity(pat) {
                    unescaped.push_str(value);
                } else {
                    return Err(EscapeError::UnrecognizedEntity(
                        start + 1..end,
                        pat.to_string(),
                    ));
                }

                last_end = end + 1;
            }
            _ => return Err(EscapeError::UnterminatedEntity(start..raw.len())),
        }
    }

    if let Some(mut unescaped) = unescaped {
        if let Some(rest) = raw.get(last_end..) {
            unescaped.push_str(rest);
        }
        Ok(Cow::Owned(unescaped))
    } else {
        Ok(Cow::Borrowed(raw))
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Normalizes XML 1.1 end-of-line sequences into single `\n` characters.
pub(crate) fn normalize_xml11_eols(text: &str) -> Cow<'_, str> {
    let bytes = text.as_bytes();

    // The following UTF-8 sequences translate into a single `\n` (U+000A):
    // `0d 0a` (\r\n), `0d c2 85` (\r\x85), `0d` (\r), `c2 85` (\x85),
    // `e2 80 a8` (U+2028).
    if let Some(i) = memchr3(b'\r', 0xC2, 0xE2, bytes) {
        let mut normalized = String::with_capacity(text.len());
        normalized.push_str(&text[0..i]);

        let mut pos = normalize_xml11_eol_step(&mut normalized, text, i, '\n');
        while let Some(i) = memchr3(b'\r', 0xC2, 0xE2, &bytes[pos..]) {
            let index = pos + i;
            normalized.push_str(&text[pos..index]);
            pos = normalize_xml11_eol_step(&mut normalized, text, index, '\n');
        }
        if let Some(rest) = text.get(pos..) {
            normalized.push_str(rest);
        }
        return normalized.into();
    }
    Cow::Borrowed(text)
}

/// Performs one step of XML 1.1 EOL normalization at byte `index`, pushing `ch`
/// in place of the recognized newline sequence. Returns the index of the next
/// unprocessed byte.
fn normalize_xml11_eol_step(normalized: &mut String, text: &str, index: usize, ch: char) -> usize {
    let input = text.as_bytes();
    match input[index] {
        b'\r' => {
            if index + 1 < input.len() {
                let next = input[index + 1];
                if next == b'\n' {
                    normalized.push(ch);
                    return index + 2; // skip \r\n
                }
                if next == 0xC2 {
                    // UTF-8 encoding of U+0085 is `c2 85`.
                    if index + 2 < input.len() && input[index + 2] == 0x85 {
                        normalized.push(ch);
                    } else {
                        normalized.push(ch);
                        normalized.push_str(&text[index + 1..index + 3]);
                    }
                    return index + 3; // skip \r + (c2 xx)
                }
            }
            normalized.push(ch);
            index + 1 // skip \r
        }
        b'\n' => {
            normalized.push(ch);
            index + 1 // skip \n
        }
        // Start of UTF-8 encoding of U+0085 (`c2 85`).
        0xC2 => {
            if index + 1 < input.len() && input[index + 1] == 0x85 {
                normalized.push(ch);
            } else {
                normalized.push_str(&text[index..index + 2]);
            }
            index + 2 // skip (c2 xx)
        }
        // Start of UTF-8 encoding of U+2028 (`e2 80 a8`).
        0xE2 => {
            if index + 2 < input.len() && input[index + 1] == 0x80 && input[index + 2] == 0xA8 {
                normalized.push(ch);
            } else {
                normalized.push_str(&text[index..index + 3]);
            }
            index + 3 // skip (e2 xx xx)
        }
        _ => index + 1,
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Normalizes XML 1.0 end-of-line sequences into single `\n` characters.
pub(crate) fn normalize_xml10_eols(text: &str) -> Cow<'_, str> {
    let bytes = text.as_bytes();

    // The following UTF-8 sequences translate into a single `\n` (U+000A):
    // `0d 0a` (\r\n) and `0d` (\r).
    if let Some(i) = memchr(b'\r', bytes) {
        let mut normalized = String::with_capacity(text.len());
        normalized.push_str(&text[0..i]);

        let mut pos = normalize_xml10_eol_step(&mut normalized, text, i, '\n');
        while let Some(i) = memchr(b'\r', &bytes[pos..]) {
            let index = pos + i;
            normalized.push_str(&text[pos..index]);
            pos = normalize_xml10_eol_step(&mut normalized, text, index, '\n');
        }
        if let Some(rest) = text.get(pos..) {
            normalized.push_str(rest);
        }
        return normalized.into();
    }
    Cow::Borrowed(text)
}

/// Performs one step of XML 1.0 EOL normalization at byte `index`.
fn normalize_xml10_eol_step(normalized: &mut String, text: &str, index: usize, ch: char) -> usize {
    let input = text.as_bytes();
    match input[index] {
        b'\r' => {
            normalized.push(ch);
            if index + 1 < input.len() && input[index + 1] == b'\n' {
                return index + 2; // skip \r\n
            }
            index + 1 // skip \r
        }
        b'\n' => {
            normalized.push(ch);
            index + 1 // skip \n
        }
        _ => index + 1,
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

pub(crate) fn normalize_xml10_attribute_value<'input, 'entity, F>(
    value: &'input str,
    depth: usize,
    resolve_entity: F,
) -> Result<Cow<'input, str>, EscapeError>
where
    F: FnMut(&str) -> Option<&'entity str>,
{
    normalize_attribute_value(
        value,
        depth,
        is_xml10_normalization_char,
        normalize_xml10_eol_step,
        resolve_entity,
    )
}

/// Whether a byte begins a character that XML 1.0 attribute-value normalization
/// rewrites — a callable form of the scan
/// [`normalize_xml10_attribute_value`] performs, for a caller that wants to know
/// whether normalizing would change anything before it pays for the attempt.
pub(crate) const fn is_xml10_normalization_char(b: &u8) -> bool {
    matches!(*b, b'\t' | b'\r' | b'\n' | b'&')
}

pub(crate) fn normalize_xml11_attribute_value<'input, 'entity, F>(
    value: &'input str,
    depth: usize,
    resolve_entity: F,
) -> Result<Cow<'input, str>, EscapeError>
where
    F: FnMut(&str) -> Option<&'entity str>,
{
    normalize_attribute_value(
        value,
        depth,
        is_xml11_normalization_char,
        normalize_xml11_eol_step,
        resolve_entity,
    )
}

const fn is_xml11_normalization_char(b: &u8) -> bool {
    matches!(*b, b'\t' | b'\r' | b'\n' | 0xC2 | 0xE2 | b'&')
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns the attribute value normalized as per [the XML specification], using a
/// custom entity resolver.
///
/// Do not use this method with HTML attributes.
///
/// Escape sequences such as `&gt;` are replaced with their unescaped equivalents,
/// and the characters `\t`, `\r`, `\n` are replaced with a space. This allocates
/// unless the raw attribute value does not require normalization.
///
/// # Parameters
///
/// - `value`: the unnormalized attribute value;
/// - `depth`: maximum number of nested entities that can be expanded before
///   [`EscapeError::TooManyNestedEntities`] is returned;
/// - `is_normalization_char`: predicate reporting whether a byte begins a
///   character requiring normalization (UTF-8 encoding assumed);
/// - `normalize_eol_step`: a function performing EOL normalization of a character;
/// - `resolve_entity`: a function to resolve named entities.
///
/// [the XML specification]: https://www.w3.org/TR/xml11/#AVNormalize
pub fn normalize_attribute_value<'input, 'entity, C, E, F>(
    value: &'input str,
    depth: usize,
    is_normalization_char: C,
    normalize_eol_step: E,
    mut resolve_entity: F,
) -> Result<Cow<'input, str>, EscapeError>
where
    C: Fn(&u8) -> bool,
    E: Fn(&mut String, &str, usize, char) -> usize,
    F: FnMut(&str) -> Option<&'entity str>,
{
    let mut iter = value.as_bytes().iter();

    if let Some(i) = iter.position(&is_normalization_char) {
        let mut normalized = String::with_capacity(value.len());
        let pos = normalize_attr_step(
            &mut normalized,
            &mut iter,
            value,
            0,
            i,
            depth,
            false,
            &is_normalization_char,
            &normalize_eol_step,
            &mut resolve_entity,
        )?;

        normalize_attr_steps(
            &mut normalized,
            &mut iter,
            value,
            pos,
            depth,
            false,
            &is_normalization_char,
            &normalize_eol_step,
            &mut resolve_entity,
        )?;
        return Ok(normalized.into());
    }
    Ok(Cow::Borrowed(value))
}

/// Applies the normalization steps to a whole run of text.
///
/// `in_entity` says whether `input` is an entity's replacement text rather than the attribute's
/// own value; it selects which of the two white-space rules of [XML 1.0 §3.3.3] applies — see
/// [`normalize_entity_whitespace_step`].
///
/// [XML 1.0 §3.3.3]: https://www.w3.org/TR/xml/#AVNormalize
#[allow(clippy::too_many_arguments)]
fn normalize_attr_steps<'entity, C, E, F>(
    normalized: &mut String,
    iter: &mut Iter<u8>,
    input: &str,
    mut pos: usize,
    depth: usize,
    in_entity: bool,
    is_normalization_char: &C,
    normalize_eol_step: &E,
    resolve_entity: &mut F,
) -> Result<(), EscapeError>
where
    C: Fn(&u8) -> bool,
    E: Fn(&mut String, &str, usize, char) -> usize,
    F: FnMut(&str) -> Option<&'entity str>,
{
    while let Some(i) = iter.position(is_normalization_char) {
        pos = normalize_attr_step(
            normalized,
            iter,
            input,
            pos,
            pos + i,
            depth,
            in_entity,
            is_normalization_char,
            normalize_eol_step,
            resolve_entity,
        )?;
    }
    if let Some(rest) = input.get(pos..) {
        normalized.push_str(rest);
    }
    Ok(())
}

/// Performs one step of the [attribute-value normalization algorithm], including
/// the recursive expansion of entity references.
///
/// [attribute-value normalization algorithm]: https://www.w3.org/TR/xml11/#AVNormalize
#[allow(clippy::too_many_arguments)]
fn normalize_attr_step<'entity, C, E, F>(
    normalized: &mut String,
    iter: &mut Iter<u8>,
    input: &str,
    last_pos: usize,
    index: usize,
    depth: usize,
    in_entity: bool,
    is_normalization_char: &C,
    normalize_eol_step: &E,
    resolve_entity: &mut F,
) -> Result<usize, EscapeError>
where
    C: Fn(&u8) -> bool,
    E: Fn(&mut String, &str, usize, char) -> usize,
    F: FnMut(&str) -> Option<&'entity str>,
{
    if depth == 0 {
        return Err(EscapeError::TooManyNestedEntities);
    }
    // For any other character, append the character to the normalized value.
    normalized.push_str(&input[last_pos..index]);

    match input.as_bytes()[index] {
        b'&' => {
            let start = index + 1; // skip `&`
            let end = start
                + match iter.position(|&b| b == b';') {
                    Some(end) => end,
                    None => return Err(EscapeError::UnterminatedEntity(index..input.len())),
                };

            let pat = &input[start..end];
            if let Some(entity) = pat.strip_prefix('#') {
                // For a character reference, append the referenced character.
                let codepoint = parse_number(entity).map_err(EscapeError::InvalidCharRef)?;
                normalized.push_str(codepoint.encode_utf8(&mut [0u8; 4]));
            } else if pat == "amp" {
                // Special case: `&amp;` resolves to `&`; handling it directly
                // avoids a spurious unterminated-entity error.
                normalized.push('&');
            } else if let Some(value) = resolve_entity(pat) {
                // For an entity reference, recursively normalize the replacement text.
                normalize_attr_steps(
                    normalized,
                    &mut value.as_bytes().iter(),
                    value,
                    0,
                    depth.saturating_sub(1),
                    true,
                    is_normalization_char,
                    normalize_eol_step,
                    resolve_entity,
                )?;
            } else {
                return Err(EscapeError::UnrecognizedEntity(start..end, pat.to_string()));
            }
            Ok(end + 1) // skip `;`
        }
        // For a white space character (`#x9`), append a space (`#x20`).
        b'\t' => {
            normalized.push(' ');
            Ok(index + 1) // skip \t
        }
        _ => {
            let pos = if in_entity {
                normalize_entity_whitespace_step(normalized, input, index)
            } else {
                normalize_eol_step(normalized, input, index, ' ')
            };
            // Advance the iterator because several bytes may have been skipped.
            for _ in 0..pos - index - 1 {
                iter.next();
            }
            Ok(pos)
        }
    }
}

/// Applies the white-space rule of [XML 1.0 §3.3.3] to one character of an entity's
/// *replacement text*, and reports the index of the next unprocessed byte.
///
/// Step 1 of the algorithm has the attribute's own text end-of-line normalized first, so a
/// literal `#xD#xA` there is a single line break and contributes a single space — which is what
/// the end-of-line steps do while they normalize. A replacement text is different: it was built
/// from an entity declaration whose literal value had already been end-of-line normalized when
/// the declaration was parsed, so any `#xD` still in it came from a character reference. Those
/// are never line breaks, and step 3 appends one space for *each* white space character it
/// visits, so `<!ENTITY e "&#13;&#10;">` used in an attribute value contributes two spaces
/// rather than one.
///
/// [XML 1.0 §3.3.3]: https://www.w3.org/TR/xml/#AVNormalize
fn normalize_entity_whitespace_step(normalized: &mut String, text: &str, index: usize) -> usize {
    let input = text.as_bytes();
    match input[index] {
        b'\r' | b'\n' => {
            normalized.push(' ');
            index + 1
        }
        // Start of the UTF-8 encoding of U+0085 (`c2 85`), a line break in XML 1.1 only.
        0xC2 => {
            if index + 1 < input.len() && input[index + 1] == 0x85 {
                normalized.push(' ');
            } else {
                normalized.push_str(&text[index..index + 2]);
            }
            index + 2
        }
        // Start of the UTF-8 encoding of U+2028 (`e2 80 a8`), a line break in XML 1.1 only.
        0xE2 => {
            if index + 2 < input.len() && input[index + 1] == 0x80 && input[index + 2] == 0xA8 {
                normalized.push(' ');
            } else {
                normalized.push_str(&text[index..index + 3]);
            }
            index + 3
        }
        _ => index + 1,
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Resolves predefined XML entities, or all HTML5 entities when the `escape-html`
/// feature is enabled.
///
/// Behaves like [`resolve_xml_entity`] without the feature, and like
#[cfg_attr(feature = "escape-html", doc = "[`resolve_html5_entity`] with it.")]
#[cfg_attr(not(feature = "escape-html"), doc = "`resolve_html5_entity` with it.")]
#[inline]
pub const fn resolve_predefined_entity(entity: &str) -> Option<&'static str> {
    #[cfg(not(feature = "escape-html"))]
    {
        resolve_xml_entity(entity)
    }

    #[cfg(feature = "escape-html")]
    {
        resolve_html5_entity(entity)
    }
}

/// Resolves the five predefined XML entities. Returns `None` for any other name.
///
/// ```
/// # use oxixml_xml::escape::resolve_xml_entity;
/// assert_eq!(resolve_xml_entity("lt"), Some("<"));
/// assert_eq!(resolve_xml_entity("gt"), Some(">"));
/// assert_eq!(resolve_xml_entity("amp"), Some("&"));
/// assert_eq!(resolve_xml_entity("apos"), Some("'"));
/// assert_eq!(resolve_xml_entity("quot"), Some("\""));
///
/// assert_eq!(resolve_xml_entity("foo"), None);
/// ```
pub const fn resolve_xml_entity(entity: &str) -> Option<&'static str> {
    // `match` over strings is not allowed in `const fn`, so match on bytes.
    let s = match entity.as_bytes() {
        b"lt" => "<",
        b"gt" => ">",
        b"amp" => "&",
        b"apos" => "'",
        b"quot" => "\"",
        _ => return None,
    };
    Some(s)
}

/// Parses the numeric part of a character reference (the text after `&#`) and
/// returns the referenced character.
pub(crate) fn parse_number(num: &str) -> Result<char, ParseCharRefError> {
    let code = if let Some(hex) = num.strip_prefix('x') {
        from_str_radix(hex, 16)?
    } else {
        from_str_radix(num, 10)?
    };
    if code == 0 {
        return Err(ParseCharRefError::IllegalCharacter(code));
    }
    match std::char::from_u32(code) {
        Some(c) => Ok(c),
        None => Err(ParseCharRefError::InvalidCodepoint(code)),
    }
}

#[inline]
fn from_str_radix(src: &str, radix: u32) -> Result<u32, ParseCharRefError> {
    match src.as_bytes().first().copied() {
        // Sign characters are not allowed, but `u32::from_str_radix` accepts `+`.
        // We handle `-` too for consistent errors.
        Some(b'+') | Some(b'-') => Err(ParseCharRefError::UnexpectedSign),
        _ => u32::from_str_radix(src, radix).map_err(ParseCharRefError::InvalidNumber),
    }
}

#[cfg(test)]
mod tests;
