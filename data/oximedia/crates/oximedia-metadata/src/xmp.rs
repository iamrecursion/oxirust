//! XMP (Extensible Metadata Platform) parsing and writing support.
//!
//! XMP is an Adobe standard for embedding metadata in files using RDF/XML.
//!
//! # Format
//!
//! XMP metadata is stored as RDF/XML with various namespaces:
//! - **dc**: Dublin Core (dc:title, dc:creator, dc:rights, etc.)
//! - **xmp**: XMP Basic (xmp:CreateDate, xmp:ModifyDate, etc.)
//! - **xmpRights**: XMP Rights Management
//! - **photoshop**: Photoshop-specific metadata
//!
//! # Example
//!
//! ```xml
//! <x:xmpmeta xmlns:x="adobe:ns:meta/">
//!   <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
//!     <rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/">
//!       <dc:title>My Title</dc:title>
//!     </rdf:Description>
//!   </rdf:RDF>
//! </x:xmpmeta>
//! ```

use crate::{Error, Metadata, MetadataFormat, MetadataValue};
use quick_xml::events::{BytesEnd, BytesRef, BytesStart, BytesText, Event};
use quick_xml::{Reader, Writer};
use std::borrow::Cow;
use std::io::Cursor;

// ---- XML character-entity decoding ----

/// Resolve a single XML general/character entity reference into its replacement
/// text.
///
/// `name` is the entity body *without* the surrounding `&` and `;` (e.g. `amp`,
/// `lt`, `#38`, `#x41`).
///
/// Handles the five predefined XML entities (`amp`, `lt`, `gt`, `quot`, `apos`)
/// and numeric character references in both decimal (`#NN`) and hexadecimal
/// (`#xHH`) form via [`BytesRef::resolve_char_ref`].
///
/// Unknown or malformed references are preserved verbatim as `&name;` so that no
/// surrounding text is lost and no panic occurs (`char::from_u32`-style failures
/// are handled gracefully).
fn append_entity(out: &mut String, name: &str) {
    match name {
        "amp" => out.push('&'),
        "lt" => out.push('<'),
        "gt" => out.push('>'),
        "quot" => out.push('"'),
        "apos" => out.push('\''),
        _ => {
            // Numeric character reference (`#NN` decimal or `#xHH` hex), or an
            // unknown named entity.  `resolve_char_ref` returns `Ok(Some(ch))`
            // only for valid numeric references; named entities yield
            // `Ok(None)` and invalid numeric references yield `Err`.  In every
            // non-resolvable case we re-emit the reference literally rather than
            // dropping it.
            let bytes = BytesRef::new(name);
            match bytes.resolve_char_ref() {
                Ok(Some(ch)) => out.push(ch),
                _ => {
                    out.push('&');
                    out.push_str(name);
                    out.push(';');
                }
            }
        }
    }
}

/// Accumulator for an element's character data that preserves the zero-copy
/// fast-path.
///
/// `quick_xml` splits element text around entity references, delivering each
/// run of plain text as a separate [`Event::Text`] and each reference as an
/// [`Event::GeneralRef`].  To reconstruct the full value, fragments must be
/// concatenated.
///
/// The common case — a property whose text contains no entities — yields a
/// single borrowed [`Cow::Borrowed`] fragment and no references.  In that case
/// the buffer holds the borrow directly and never allocates an owned `String`.
/// The owned path is taken only when a second fragment or an entity reference
/// actually appears, at which point the previously borrowed slice is promoted to
/// an owned `String` and subsequent data is appended in place.
#[derive(Default)]
struct TextBuf<'a> {
    value: Option<Cow<'a, str>>,
}

impl<'a> TextBuf<'a> {
    /// Append a decoded plain-text fragment, keeping a single fragment borrowed.
    fn push_text(&mut self, text: Cow<'a, str>) {
        match self.value.take() {
            None => self.value = Some(text),
            Some(existing) => {
                let mut owned = existing.into_owned();
                owned.push_str(&text);
                self.value = Some(Cow::Owned(owned));
            }
        }
    }

    /// Append the replacement text of an entity reference, forcing the owned
    /// path (a value containing an entity can never be a pure borrow).
    fn push_entity(&mut self, name: &str) {
        let mut owned = match self.value.take() {
            None => String::new(),
            Some(existing) => existing.into_owned(),
        };
        append_entity(&mut owned, name);
        self.value = Some(Cow::Owned(owned));
    }

    /// Whether any character data has been accumulated.
    fn is_empty(&self) -> bool {
        self.value.is_none()
    }

    /// Take the accumulated value, trimmed of leading/trailing ASCII whitespace,
    /// **preserving the zero-copy borrow** wherever possible.
    ///
    /// Trimming is applied to the *assembled* value rather than per fragment so
    /// that internal whitespace adjacent to an entity (e.g. the spaces in
    /// `Rock &amp; Roll`) is preserved.  Returns `None` if the trimmed value is
    /// empty.
    ///
    /// The returned [`Cow`] retains its borrow state:
    /// - a [`Cow::Borrowed`] value stays borrowed even when surrounding
    ///   whitespace is trimmed, because the trimmed result is a sub-slice of the
    ///   same `&'a str` (still pointing into the original input buffer);
    /// - a [`Cow::Owned`] value (one that required entity unescaping or spanned
    ///   multiple fragments) stays owned, reusing its allocation when no trimming
    ///   is needed and re-allocating only the trimmed slice otherwise.
    fn take_trimmed_cow(&mut self) -> Option<Cow<'a, str>> {
        match self.value.take()? {
            Cow::Borrowed(s) => {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    // `trimmed` is a sub-slice of `s: &'a str`, so it carries the
                    // input lifetime `'a` and remains a true borrow.
                    Some(Cow::Borrowed(trimmed))
                }
            }
            Cow::Owned(s) => {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    None
                } else if trimmed.len() == s.len() {
                    // No leading/trailing whitespace — reuse the allocation.
                    Some(Cow::Owned(s))
                } else {
                    Some(Cow::Owned(trimmed.to_owned()))
                }
            }
        }
    }
}

// ---- XMP Structured Property Types ----

/// XMP structured property kind per the XMP specification.
///
/// - **Seq** (`rdf:Seq`): Ordered array. The order of items is significant.
/// - **Bag** (`rdf:Bag`): Unordered set.  The order of items is not significant.
/// - **Alt** (`rdf:Alt`): Alternative values (typically language alternatives).
///   The first item is the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XmpArrayKind {
    /// Ordered array (`rdf:Seq`).
    Seq,
    /// Unordered set (`rdf:Bag`).
    Bag,
    /// Language alternatives (`rdf:Alt`).
    Alt,
}

impl XmpArrayKind {
    /// The RDF element name for this array kind.
    pub fn rdf_element(self) -> &'static str {
        match self {
            Self::Seq => "rdf:Seq",
            Self::Bag => "rdf:Bag",
            Self::Alt => "rdf:Alt",
        }
    }

    /// Try to detect the array kind from a local element name.
    pub fn from_element(name: &str) -> Option<Self> {
        let local = if let Some(pos) = name.find(':') {
            &name[pos + 1..]
        } else {
            name
        };
        match local {
            "Seq" => Some(Self::Seq),
            "Bag" => Some(Self::Bag),
            "Alt" => Some(Self::Alt),
            _ => None,
        }
    }
}

/// An XMP structured property containing an array of items.
#[derive(Debug, Clone, PartialEq)]
pub struct XmpArray {
    /// The kind of array.
    pub kind: XmpArrayKind,
    /// The items in the array (in order for Seq, unordered for Bag,
    /// first = default for Alt).
    pub items: Vec<String>,
}

impl XmpArray {
    /// Create a new array.
    pub fn new(kind: XmpArrayKind) -> Self {
        Self {
            kind,
            items: Vec::new(),
        }
    }

    /// Add an item.
    pub fn push(&mut self, item: impl Into<String>) {
        self.items.push(item.into());
    }

    /// Get the default / first item (meaningful for Alt arrays).
    pub fn default_item(&self) -> Option<&str> {
        self.items.first().map(|s| s.as_str())
    }

    /// Number of items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the array is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

// ---- Zero-copy borrowed view ----

/// A single XMP property value, borrowing from the input buffer where possible.
///
/// This is the borrowed counterpart of the subset of [`MetadataValue`] that the
/// XMP parser produces ([`MetadataValue::Text`] and [`MetadataValue::TextList`]).
///
/// Each string is a [`Cow<str>`]:
/// - [`Cow::Borrowed`] — the common, zero-copy case, a `&'a str` slice that
///   points directly into the original input buffer (no allocation);
/// - [`Cow::Owned`] — the fallback used only when the value required work that a
///   plain slice cannot represent, namely XML entity unescaping (`&amp;` → `&`)
///   or reassembly of text split across multiple fragments.
#[derive(Debug, Clone, PartialEq)]
pub enum XmpValue<'a> {
    /// A simple text property.
    Text(Cow<'a, str>),
    /// A structured array property (`rdf:Seq` / `rdf:Bag` / `rdf:Alt`).
    TextList(Vec<Cow<'a, str>>),
}

impl<'a> XmpValue<'a> {
    /// Borrow the value as a string slice if it is a [`XmpValue::Text`].
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(value) => Some(value.as_ref()),
            Self::TextList(_) => None,
        }
    }

    /// Borrow the array items if it is a [`XmpValue::TextList`].
    #[must_use]
    pub fn as_text_list(&self) -> Option<&[Cow<'a, str>]> {
        match self {
            Self::TextList(items) => Some(items),
            Self::Text(_) => None,
        }
    }

    /// Whether every string this value holds is borrowed from the input buffer
    /// (i.e. fully zero-copy, no owned allocations).
    ///
    /// Useful for asserting in tests that a clean value did not allocate.
    #[must_use]
    pub fn is_borrowed(&self) -> bool {
        match self {
            Self::Text(value) => matches!(value, Cow::Borrowed(_)),
            Self::TextList(items) => items.iter().all(|item| matches!(item, Cow::Borrowed(_))),
        }
    }

    /// Promote this borrowed value into an owned [`MetadataValue`], allocating as
    /// needed.
    #[must_use]
    pub fn into_owned(self) -> MetadataValue {
        match self {
            Self::Text(value) => MetadataValue::Text(value.into_owned()),
            Self::TextList(items) => {
                MetadataValue::TextList(items.into_iter().map(Cow::into_owned).collect())
            }
        }
    }
}

/// A zero-copy view over parsed XMP metadata.
///
/// Produced by [`parse_borrowed`], an `XmpView` holds the parsed key/value pairs
/// with their strings borrowed directly from the input buffer wherever no XML
/// entity unescaping or fragment reassembly was required.  Keys are always plain
/// `&'a str` slices into the input (XML element names can never contain entity
/// references), while values are [`XmpValue`]s whose strings are borrowed in the
/// common case and owned only when unescaping is needed.
///
/// Document order is preserved.  When the same property key appears more than
/// once, [`XmpView::get`] returns the last occurrence, matching the
/// last-insert-wins semantics of the owned [`parse`] path (which inserts into a
/// `HashMap`).
#[derive(Debug, Clone, PartialEq)]
pub struct XmpView<'a> {
    fields: Vec<(&'a str, XmpValue<'a>)>,
}

impl<'a> XmpView<'a> {
    /// Number of parsed properties.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Whether no properties were parsed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Look up a property value by key.
    ///
    /// Returns the **last** matching property in document order, mirroring the
    /// last-insert-wins behaviour of the owned [`parse`] path.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&XmpValue<'a>> {
        self.fields
            .iter()
            .rev()
            .find(|(field_key, _)| *field_key == key)
            .map(|(_, value)| value)
    }

    /// All parsed properties, in document order.
    #[must_use]
    pub fn fields(&self) -> &[(&'a str, XmpValue<'a>)] {
        &self.fields
    }

    /// Iterate over the parsed properties in document order.
    pub fn iter(&self) -> impl Iterator<Item = (&'a str, &XmpValue<'a>)> + '_ {
        self.fields.iter().map(|(key, value)| (*key, value))
    }

    /// Whether every parsed key and value is borrowed from the input buffer
    /// (i.e. the whole view is fully zero-copy).
    #[must_use]
    pub fn is_fully_borrowed(&self) -> bool {
        self.fields.iter().all(|(_, value)| value.is_borrowed())
    }

    /// Consume the view and materialise an owned [`Metadata`] container.
    ///
    /// This is the bridge used by the owned [`parse`] entry point: every borrowed
    /// slice is copied into an owned `String` exactly once, at the end.
    #[must_use]
    pub fn into_owned_metadata(self) -> Metadata {
        let mut metadata = Metadata::new(MetadataFormat::Xmp);
        for (key, value) in self.fields {
            metadata.insert(key.to_owned(), value.into_owned());
        }
        metadata
    }

    /// Materialise an owned [`Metadata`] container without consuming the view.
    #[must_use]
    pub fn to_metadata(&self) -> Metadata {
        self.clone().into_owned_metadata()
    }
}

/// Re-anchor a sub-slice of `haystack` (identified by `needle`'s bytes) back onto
/// `haystack`'s lifetime, yielding a `&'a str`.
///
/// `quick_xml`'s `BytesStart::name()` only exposes the element-name bytes through
/// a `&self`-bound `QName`, so the slice cannot be stored across the parse loop
/// directly.  However, for a slice-backed reader ([`Reader::from_str`]) those
/// bytes are always a sub-slice of the original input `haystack`.  This function
/// recovers the input-lifetime `&'a str` from the raw byte pointer/length using
/// only safe offset arithmetic and [`str::get`] (no `unsafe`, no panics): the
/// returned slice borrows `haystack`, not the transient `QName`.
///
/// The error arms are defensive — they cannot occur for a `from_str` reader,
/// whose event names always borrow the input — and exist solely to keep the
/// function total without `unwrap`/`panic`.
fn anchor_name<'a>(haystack: &'a str, needle: &[u8]) -> Result<&'a str, Error> {
    let base = haystack.as_ptr() as usize;
    let start = needle.as_ptr() as usize;
    let end = start
        .checked_add(needle.len())
        .ok_or_else(|| Error::Xml("XMP element name length overflow".to_string()))?;
    if start < base || end > base + haystack.len() {
        return Err(Error::Xml(
            "XMP element name is not anchored within the input buffer".to_string(),
        ));
    }
    let offset = start - base;
    haystack
        .get(offset..offset + needle.len())
        .ok_or_else(|| Error::Xml("XMP element name spans a non-UTF-8 boundary".to_string()))
}

/// XMP packet start marker
const XMP_PACKET_START: &str = r#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>"#;

/// XMP packet end marker
const XMP_PACKET_END: &str = r#"<?xpacket end="w"?>"#;

/// Parse XMP metadata from XML data.
///
/// Supports simple text properties and structured array properties
/// (`rdf:Seq`, `rdf:Bag`, `rdf:Alt`).  When an array is encountered,
/// the property value is stored as `MetadataValue::TextList`.
///
/// XML character entities in element text are decoded: the five predefined
/// entities (`&amp;` → `&`, `&lt;` → `<`, `&gt;` → `>`, `&quot;` → `"`,
/// `&apos;` → `'`) as well as decimal (`&#NN;`) and hexadecimal (`&#xHH;`)
/// numeric character references.  Because `quick_xml` reports element text in
/// fragments split around each reference (a [`Event::Text`] for plain runs and
/// an [`Event::GeneralRef`] for each entity), fragments are accumulated per
/// element and committed when the element closes.
///
/// This implementation minimises per-element allocations:
/// - QName bytes are borrowed directly from the event rather than converted
///   to an owned `String` before pattern-matching.
/// - Element text reuses borrowed `Cow<str>` slices from the decoder via
///   `TextBuf`; an owned `String` is materialised only when a value spans
///   multiple fragments or actually contains an entity reference.
///
/// The input is parsed via [`Reader::from_str`] so that the slice-based
/// reader can return borrowed `Cow::Borrowed` slices from the input directly,
/// maximising zero-copy text extraction for pure-ASCII / UTF-8 content.
///
/// This owned entry point is a thin wrapper around the zero-copy
/// [`parse_borrowed`]: it parses once into a borrowed [`XmpView`] and then
/// materialises the owned [`Metadata`] at the very end via
/// [`XmpView::into_owned_metadata`].  There is therefore a single parser; callers
/// that can keep the input buffer alive should prefer [`parse_borrowed`] to avoid
/// the final round of allocations entirely.
///
/// # Errors
///
/// Returns an error if the data is not valid UTF-8 or not valid XMP.
pub fn parse(data: &[u8]) -> Result<Metadata, Error> {
    Ok(parse_borrowed(data)?.into_owned_metadata())
}

/// Parse XMP metadata into a zero-copy [`XmpView`] that borrows from `data`.
///
/// This is the underlying single-pass parser.  Every property key is a `&str`
/// slice into `data`, and every property value is an [`XmpValue`] whose strings
/// borrow from `data` in the common case — no heap allocation occurs for clean,
/// pure-UTF-8 property text.  An owned [`Cow::Owned`] string is materialised only
/// when a value actually requires it:
/// - it contains an XML entity reference that must be unescaped
///   (`&amp;` → `&`, numeric character references, …), or
/// - its character data was split into multiple fragments by `quick_xml` (which
///   happens precisely around entity references) and must be reassembled.
///
/// The returned view borrows `data` for as long as it lives; call
/// [`XmpView::into_owned_metadata`] to detach it into an owned [`Metadata`].
///
/// Structured array properties (`rdf:Seq` / `rdf:Bag` / `rdf:Alt`) are collected
/// into [`XmpValue::TextList`]; simple text properties become [`XmpValue::Text`].
/// XML character entities are decoded exactly as documented on [`parse`].
///
/// # Errors
///
/// Returns an error if the data is not valid UTF-8 or not valid XMP.
pub fn parse_borrowed(data: &[u8]) -> Result<XmpView<'_>, Error> {
    let xml_str = std::str::from_utf8(data)
        .map_err(|e| Error::Xml(format!("XMP data is not valid UTF-8: {e}")))?;

    let mut reader = Reader::from_str(xml_str);
    // Trimming is performed on the assembled value (see `TextBuf::take_trimmed_cow`)
    // rather than per fragment, so reader-level trimming is left disabled: it
    // would otherwise strip the internal whitespace surrounding an entity (e.g.
    // the spaces in `Rock &amp; Roll`).
    reader.config_mut().trim_text(false);

    // Parsed properties, in document order, borrowing from `xml_str`.
    let mut fields: Vec<(&str, XmpValue<'_>)> = Vec::new();

    // The current simple-property element name, borrowed from the input.
    let mut current_tag: Option<&str> = None;
    // Character data accumulated for the current simple property element.
    let mut cur_text = TextBuf::default();

    // State for structured array parsing.
    let mut in_array: Option<(&str, XmpArrayKind)> = None; // (parent tag, kind)
    let mut array_items: Vec<Cow<'_, str>> = Vec::new();
    let mut in_li = false;
    // Character data accumulated for the current `rdf:li` array item.
    let mut li_text = TextBuf::default();

    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) => {
                // Re-anchor the element name onto the input lifetime so it can be
                // stored across the loop as a zero-copy `&str` key.
                let qname = e.name();
                let name = anchor_name(xml_str, qname.as_ref())?;

                // Check if this is an array container element.
                if let Some(kind) = XmpArrayKind::from_element(name) {
                    if let Some(tag) = current_tag {
                        in_array = Some((tag, kind));
                        array_items.clear();
                    }
                } else if (in_array.is_some() && name.ends_with(":li")) || name == "rdf:li" {
                    in_li = true;
                    li_text = TextBuf::default();
                } else if name.contains(':') {
                    // Qualified property name: a borrowed slice into the input.
                    current_tag = Some(name);
                    cur_text = TextBuf::default();
                }
            }
            Ok(Event::Empty(ref e)) => {
                let qname = e.name();
                let name = anchor_name(xml_str, qname.as_ref())?;
                if name.contains(':') {
                    current_tag = Some(name);
                    cur_text = TextBuf::default();
                }
            }
            Ok(Event::Text(e)) => {
                // `decode()` on the str-based reader returns Cow::Borrowed when no
                // charset conversion is required, so no heap allocation occurs for
                // plain UTF-8 text.  Fragments are accumulated and only committed
                // (and trimmed) on the element's End event, because `quick_xml`
                // splits text around entity references into multiple events.
                let cow = e
                    .decode()
                    .map_err(|err| Error::Xml(format!("Failed to decode text: {err}")))?;

                if in_li && in_array.is_some() {
                    li_text.push_text(cow);
                } else if current_tag.is_some() {
                    cur_text.push_text(cow);
                }
            }
            Ok(Event::GeneralRef(ref e)) => {
                // A character/general entity reference such as `&amp;`, `&lt;` or
                // `&#39;`.  Resolve it into its replacement text and append to the
                // active accumulator so that the surrounding text is preserved.
                let name = e
                    .decode()
                    .map_err(|err| Error::Xml(format!("Failed to decode entity: {err}")))?;

                if in_li && in_array.is_some() {
                    li_text.push_entity(&name);
                } else if current_tag.is_some() {
                    cur_text.push_entity(&name);
                }
            }
            Ok(Event::End(ref e)) => {
                // End names are only matched transiently, so the `&self`-bound
                // QName slice is sufficient here (no re-anchoring required).
                let qname = e.name();
                let name_bytes = qname.as_ref();
                let name_str = std::str::from_utf8(name_bytes)
                    .map_err(|err| Error::Xml(format!("Invalid element name encoding: {err}")))?;

                if name_str.ends_with(":li") || name_str == "rdf:li" {
                    if let Some(item) = li_text.take_trimmed_cow() {
                        array_items.push(item);
                    }
                    in_li = false;
                } else if XmpArrayKind::from_element(name_str).is_some() {
                    // End of array container — store collected items.
                    if let Some((parent_tag, _kind)) = in_array.take() {
                        if !array_items.is_empty() {
                            // Move items into the value; re-use the existing Vec.
                            let items = std::mem::take(&mut array_items);
                            fields.push((parent_tag, XmpValue::TextList(items)));
                        }
                    }
                    array_items.clear();
                } else if in_array.is_none() {
                    // End of a simple property element — commit its accumulated,
                    // entity-decoded, trimmed text value.
                    if !cur_text.is_empty() {
                        if let Some(tag) = current_tag {
                            if let Some(text) = cur_text.take_trimmed_cow() {
                                fields.push((tag, XmpValue::Text(text)));
                            }
                        }
                    }
                    current_tag = None;
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(Error::Xml(format!("XML parse error: {e}"))),
            _ => {}
        }
    }

    Ok(XmpView { fields })
}

/// Write XMP metadata to XML data.
///
/// # Errors
///
/// Returns an error if writing fails.
pub fn write(metadata: &Metadata) -> Result<Vec<u8>, Error> {
    let mut result = Vec::new();

    // Write packet start
    result.extend_from_slice(XMP_PACKET_START.as_bytes());
    result.push(b'\n');

    // Create XML writer
    let mut writer = Writer::new(Cursor::new(Vec::new()));

    // Write xmpmeta start
    let mut xmpmeta = BytesStart::new("x:xmpmeta");
    xmpmeta.push_attribute(("xmlns:x", "adobe:ns:meta/"));
    writer
        .write_event(Event::Start(xmpmeta))
        .map_err(|e| Error::Xml(format!("Failed to write xmpmeta start: {e}")))?;

    // Write RDF start
    let mut rdf = BytesStart::new("rdf:RDF");
    rdf.push_attribute(("xmlns:rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"));
    writer
        .write_event(Event::Start(rdf))
        .map_err(|e| Error::Xml(format!("Failed to write RDF start: {e}")))?;

    // Group fields by namespace
    let mut dc_fields = Vec::new();
    let mut xmp_fields = Vec::new();
    let mut other_fields = Vec::new();

    for (key, value) in metadata.fields() {
        if key.starts_with("dc:") {
            dc_fields.push((key, value));
        } else if key.starts_with("xmp:") {
            xmp_fields.push((key, value));
        } else {
            other_fields.push((key, value));
        }
    }

    // Write Dublin Core description
    if !dc_fields.is_empty() {
        write_description(&mut writer, &dc_fields, "http://purl.org/dc/elements/1.1/")?;
    }

    // Write XMP description
    if !xmp_fields.is_empty() {
        write_description(&mut writer, &xmp_fields, "http://ns.adobe.com/xap/1.0/")?;
    }

    // Write other descriptions
    if !other_fields.is_empty() {
        write_description(&mut writer, &other_fields, "")?;
    }

    // Write RDF end
    writer
        .write_event(Event::End(BytesEnd::new("rdf:RDF")))
        .map_err(|e| Error::Xml(format!("Failed to write RDF end: {e}")))?;

    // Write xmpmeta end
    writer
        .write_event(Event::End(BytesEnd::new("x:xmpmeta")))
        .map_err(|e| Error::Xml(format!("Failed to write xmpmeta end: {e}")))?;

    // Get XML data
    let xml_data = writer.into_inner().into_inner();
    result.extend_from_slice(&xml_data);

    // Write packet end
    result.push(b'\n');
    result.extend_from_slice(XMP_PACKET_END.as_bytes());

    Ok(result)
}

/// Write an RDF description with fields.
///
/// Text values are written as simple properties.
/// TextList values are written as `rdf:Bag` structured arrays.
fn write_description(
    writer: &mut Writer<Cursor<Vec<u8>>>,
    fields: &[(&String, &MetadataValue)],
    namespace_uri: &str,
) -> Result<(), Error> {
    // Write Description start
    let mut desc = BytesStart::new("rdf:Description");
    if !namespace_uri.is_empty() {
        // Determine the right xmlns prefix from the first key
        let prefix = fields
            .first()
            .and_then(|(k, _)| k.find(':').map(|p| &k[..p]))
            .unwrap_or("dc");
        let attr_name = format!("xmlns:{prefix}");
        desc.push_attribute((attr_name.as_str(), namespace_uri));
    }
    writer
        .write_event(Event::Start(desc))
        .map_err(|e| Error::Xml(format!("Failed to write Description start: {e}")))?;

    // Write fields
    for (key, value) in fields {
        match value {
            MetadataValue::Text(text) => {
                writer
                    .write_event(Event::Start(BytesStart::new(key.as_str())))
                    .map_err(|e| Error::Xml(format!("Failed to write element start: {e}")))?;
                writer
                    .write_event(Event::Text(BytesText::new(text)))
                    .map_err(|e| Error::Xml(format!("Failed to write text: {e}")))?;
                writer
                    .write_event(Event::End(BytesEnd::new(key.as_str())))
                    .map_err(|e| Error::Xml(format!("Failed to write element end: {e}")))?;
            }
            MetadataValue::TextList(items) => {
                // Write as rdf:Bag
                write_xmp_array(writer, key, XmpArrayKind::Bag, items)?;
            }
            _ => {
                // Skip non-text values for XMP
            }
        }
    }

    // Write Description end
    writer
        .write_event(Event::End(BytesEnd::new("rdf:Description")))
        .map_err(|e| Error::Xml(format!("Failed to write Description end: {e}")))?;

    Ok(())
}

/// Write an XMP structured array property.
///
/// Produces:
/// ```xml
/// <tag>
///   <rdf:Seq|Bag|Alt>
///     <rdf:li>item1</rdf:li>
///     <rdf:li>item2</rdf:li>
///   </rdf:Seq|Bag|Alt>
/// </tag>
/// ```
fn write_xmp_array(
    writer: &mut Writer<Cursor<Vec<u8>>>,
    tag: &str,
    kind: XmpArrayKind,
    items: &[String],
) -> Result<(), Error> {
    let rdf_elem = kind.rdf_element();

    // Open the property element
    writer
        .write_event(Event::Start(BytesStart::new(tag)))
        .map_err(|e| Error::Xml(format!("Failed to write array property start: {e}")))?;

    // Open the array container
    writer
        .write_event(Event::Start(BytesStart::new(rdf_elem)))
        .map_err(|e| Error::Xml(format!("Failed to write {rdf_elem} start: {e}")))?;

    // Write items
    for item in items {
        writer
            .write_event(Event::Start(BytesStart::new("rdf:li")))
            .map_err(|e| Error::Xml(format!("Failed to write rdf:li start: {e}")))?;
        writer
            .write_event(Event::Text(BytesText::new(item)))
            .map_err(|e| Error::Xml(format!("Failed to write rdf:li text: {e}")))?;
        writer
            .write_event(Event::End(BytesEnd::new("rdf:li")))
            .map_err(|e| Error::Xml(format!("Failed to write rdf:li end: {e}")))?;
    }

    // Close the array container
    writer
        .write_event(Event::End(BytesEnd::new(rdf_elem)))
        .map_err(|e| Error::Xml(format!("Failed to write {rdf_elem} end: {e}")))?;

    // Close the property element
    writer
        .write_event(Event::End(BytesEnd::new(tag)))
        .map_err(|e| Error::Xml(format!("Failed to write array property end: {e}")))?;

    Ok(())
}

/// Write an XMP structured array property directly (public API).
///
/// This produces standalone XML for the given array,
/// useful for building XMP documents programmatically.
pub fn write_xmp_array_property(
    metadata: &mut Metadata,
    key: &str,
    kind: XmpArrayKind,
    items: Vec<String>,
) {
    let _ = kind; // kind is used when writing; stored as TextList for now
    metadata.insert(key.to_string(), MetadataValue::TextList(items));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xmp_round_trip() {
        let mut metadata = Metadata::new(MetadataFormat::Xmp);

        metadata.insert(
            "dc:title".to_string(),
            MetadataValue::Text("Test Title".to_string()),
        );
        metadata.insert(
            "dc:creator".to_string(),
            MetadataValue::Text("Test Creator".to_string()),
        );
        metadata.insert(
            "dc:rights".to_string(),
            MetadataValue::Text("Copyright 2024".to_string()),
        );

        let data = write(&metadata).expect("Write failed");
        let parsed = parse(&data).expect("Parse failed");

        assert_eq!(
            parsed.get("dc:title").and_then(|v| v.as_text()),
            Some("Test Title")
        );
        assert_eq!(
            parsed.get("dc:creator").and_then(|v| v.as_text()),
            Some("Test Creator")
        );
        assert_eq!(
            parsed.get("dc:rights").and_then(|v| v.as_text()),
            Some("Copyright 2024")
        );
    }

    #[test]
    fn test_xmp_empty() {
        let metadata = Metadata::new(MetadataFormat::Xmp);
        let data = write(&metadata).expect("Write failed");
        assert!(data.starts_with(XMP_PACKET_START.as_bytes()));
        assert!(data.ends_with(XMP_PACKET_END.as_bytes()));
    }

    // ------- Structured property tests -------

    #[test]
    fn test_xmp_array_kind_element_names() {
        assert_eq!(XmpArrayKind::Seq.rdf_element(), "rdf:Seq");
        assert_eq!(XmpArrayKind::Bag.rdf_element(), "rdf:Bag");
        assert_eq!(XmpArrayKind::Alt.rdf_element(), "rdf:Alt");
    }

    #[test]
    fn test_xmp_array_kind_from_element() {
        assert_eq!(
            XmpArrayKind::from_element("rdf:Seq"),
            Some(XmpArrayKind::Seq)
        );
        assert_eq!(
            XmpArrayKind::from_element("rdf:Bag"),
            Some(XmpArrayKind::Bag)
        );
        assert_eq!(
            XmpArrayKind::from_element("rdf:Alt"),
            Some(XmpArrayKind::Alt)
        );
        assert_eq!(XmpArrayKind::from_element("rdf:Description"), None);
        assert_eq!(XmpArrayKind::from_element("Seq"), Some(XmpArrayKind::Seq));
    }

    #[test]
    fn test_xmp_array_push_and_default() {
        let mut arr = XmpArray::new(XmpArrayKind::Alt);
        assert!(arr.is_empty());
        assert_eq!(arr.len(), 0);
        assert_eq!(arr.default_item(), None);

        arr.push("English");
        arr.push("French");
        assert_eq!(arr.len(), 2);
        assert!(!arr.is_empty());
        assert_eq!(arr.default_item(), Some("English"));
    }

    #[test]
    fn test_parse_xmp_seq_array() {
        let xmp_xml = r#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/">
      <dc:subject>
        <rdf:Seq>
          <rdf:li>landscape</rdf:li>
          <rdf:li>nature</rdf:li>
          <rdf:li>sunset</rdf:li>
        </rdf:Seq>
      </dc:subject>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"#;

        let parsed = parse(xmp_xml.as_bytes()).expect("parse should succeed");
        let subjects = parsed
            .get("dc:subject")
            .and_then(|v| v.as_text_list())
            .expect("should be a text list");
        assert_eq!(subjects.len(), 3);
        assert_eq!(subjects[0], "landscape");
        assert_eq!(subjects[1], "nature");
        assert_eq!(subjects[2], "sunset");
    }

    #[test]
    fn test_parse_xmp_bag_array() {
        let xmp_xml = r#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/">
      <dc:creator>
        <rdf:Bag>
          <rdf:li>Alice</rdf:li>
          <rdf:li>Bob</rdf:li>
        </rdf:Bag>
      </dc:creator>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"#;

        let parsed = parse(xmp_xml.as_bytes()).expect("parse should succeed");
        let creators = parsed
            .get("dc:creator")
            .and_then(|v| v.as_text_list())
            .expect("should be a text list");
        assert_eq!(creators.len(), 2);
        assert_eq!(creators[0], "Alice");
        assert_eq!(creators[1], "Bob");
    }

    #[test]
    fn test_parse_xmp_alt_array() {
        let xmp_xml = r#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/">
      <dc:title>
        <rdf:Alt>
          <rdf:li>My Photo Title</rdf:li>
          <rdf:li>Mon titre de photo</rdf:li>
        </rdf:Alt>
      </dc:title>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"#;

        let parsed = parse(xmp_xml.as_bytes()).expect("parse should succeed");
        let titles = parsed
            .get("dc:title")
            .and_then(|v| v.as_text_list())
            .expect("should be a text list");
        assert_eq!(titles.len(), 2);
        assert_eq!(titles[0], "My Photo Title");
    }

    #[test]
    fn test_xmp_structured_array_write_and_parse_round_trip() {
        let mut metadata = Metadata::new(MetadataFormat::Xmp);

        // Add a text list (will be written as rdf:Bag)
        let keywords = vec!["music".to_string(), "jazz".to_string(), "live".to_string()];
        write_xmp_array_property(&mut metadata, "dc:subject", XmpArrayKind::Bag, keywords);

        // Also add a simple text property
        metadata.insert(
            "dc:title".to_string(),
            MetadataValue::Text("Jazz Concert".to_string()),
        );

        let data = write(&metadata).expect("write should succeed");
        let parsed = parse(&data).expect("parse should succeed");

        // Verify simple text survived
        assert_eq!(
            parsed.get("dc:title").and_then(|v| v.as_text()),
            Some("Jazz Concert")
        );

        // Verify array survived
        let subjects = parsed
            .get("dc:subject")
            .and_then(|v| v.as_text_list())
            .expect("should be text list");
        assert_eq!(subjects.len(), 3);
        assert_eq!(subjects[0], "music");
        assert_eq!(subjects[1], "jazz");
        assert_eq!(subjects[2], "live");
    }

    #[test]
    fn test_xmp_mixed_simple_and_structured() {
        let xmp_xml = r#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/">
      <dc:format>image/jpeg</dc:format>
      <dc:subject>
        <rdf:Bag>
          <rdf:li>tag1</rdf:li>
          <rdf:li>tag2</rdf:li>
        </rdf:Bag>
      </dc:subject>
      <dc:rights>Copyright 2025</dc:rights>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"#;

        let parsed = parse(xmp_xml.as_bytes()).expect("parse should succeed");
        assert_eq!(
            parsed.get("dc:format").and_then(|v| v.as_text()),
            Some("image/jpeg")
        );
        assert_eq!(
            parsed.get("dc:rights").and_then(|v| v.as_text()),
            Some("Copyright 2025")
        );
        let subjects = parsed
            .get("dc:subject")
            .and_then(|v| v.as_text_list())
            .expect("text list");
        assert_eq!(subjects, &["tag1", "tag2"]);
    }

    /// Verify that parsing a large XMP document (50+ properties and array items)
    /// produces byte-identical output to the input data.
    #[test]
    fn test_xmp_large_doc_correctness() {
        // Build an XMP document with 30 simple text properties and a Seq with 25 items.
        let mut builder = String::from(
            r#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:xmp="http://ns.adobe.com/xap/1.0/">"#,
        );

        // 30 simple text properties (split across dc and xmp namespaces).
        for i in 0..15u32 {
            builder.push_str(&format!("\n      <dc:field{i}>value-dc-{i}</dc:field{i}>"));
        }
        for i in 0..15u32 {
            builder.push_str(&format!("\n      <xmp:prop{i}>value-xmp-{i}</xmp:prop{i}>"));
        }

        // A Seq array with 25 items.
        builder.push_str("\n      <dc:subject>\n        <rdf:Seq>");
        for i in 0..25u32 {
            builder.push_str(&format!("\n          <rdf:li>keyword-{i}</rdf:li>"));
        }
        builder.push_str("\n        </rdf:Seq>\n      </dc:subject>");

        builder.push_str(
            r#"
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"#,
        );

        let parsed = parse(builder.as_bytes()).expect("large doc parse should succeed");

        // Verify every simple text property.
        for i in 0..15u32 {
            let key = format!("dc:field{i}");
            let expected = format!("value-dc-{i}");
            assert_eq!(
                parsed.get(&key).and_then(|v| v.as_text()),
                Some(expected.as_str()),
                "missing or wrong value for {key}"
            );
        }
        for i in 0..15u32 {
            let key = format!("xmp:prop{i}");
            let expected = format!("value-xmp-{i}");
            assert_eq!(
                parsed.get(&key).and_then(|v| v.as_text()),
                Some(expected.as_str()),
                "missing or wrong value for {key}"
            );
        }

        // Verify the array.
        let subjects = parsed
            .get("dc:subject")
            .and_then(|v| v.as_text_list())
            .expect("dc:subject should be a text list");
        assert_eq!(subjects.len(), 25, "expected 25 items in dc:subject array");
        for i in 0..25u32 {
            assert_eq!(subjects[i as usize], format!("keyword-{i}"));
        }
    }

    /// Verify that XML-escaped entities are correctly decoded during parsing
    /// and survive a write-then-parse round-trip.
    #[test]
    fn test_xmp_escaped_entity_roundtrip() {
        // Build XMP that contains characters requiring XML escaping.
        let xmp_xml = r#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/">
      <dc:title>Rock &amp; Roll</dc:title>
      <dc:rights>Copyright &lt;2024&gt; COOLJAPAN</dc:rights>
      <dc:description>A song with &quot;quotes&quot; &amp; &apos;apostrophes&apos;</dc:description>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"#;

        let parsed = parse(xmp_xml.as_bytes()).expect("escaped entity parse should succeed");

        // Entities must be decoded in the Metadata values.
        assert_eq!(
            parsed.get("dc:title").and_then(|v| v.as_text()),
            Some("Rock & Roll"),
            "dc:title: &amp; should decode to &"
        );
        assert_eq!(
            parsed.get("dc:rights").and_then(|v| v.as_text()),
            Some("Copyright <2024> COOLJAPAN"),
            "dc:rights: &lt;/&gt; should decode to </>"
        );
        assert_eq!(
            parsed.get("dc:description").and_then(|v| v.as_text()),
            Some("A song with \"quotes\" & 'apostrophes'"),
            "dc:description: all entities should be decoded"
        );

        // Round-trip: write the parsed metadata back to XMP and re-parse.
        let serialized = write(&parsed).expect("write should succeed");
        let reparsed = parse(&serialized).expect("reparsed write output should succeed");

        assert_eq!(
            reparsed.get("dc:title").and_then(|v| v.as_text()),
            Some("Rock & Roll"),
            "round-trip: dc:title value must be preserved"
        );
        assert_eq!(
            reparsed.get("dc:rights").and_then(|v| v.as_text()),
            Some("Copyright <2024> COOLJAPAN"),
            "round-trip: dc:rights value must be preserved"
        );
    }

    // ------- Zero-copy borrowed parse tests -------

    /// Whether the string slice `s` points *into* the byte buffer `input`.
    ///
    /// This proves a value/key is a genuine zero-copy borrow of the original
    /// input buffer rather than an independently-allocated owned `String`.
    fn slice_within(input: &[u8], s: &str) -> bool {
        let base = input.as_ptr() as usize;
        let start = s.as_ptr() as usize;
        let buf_end = base + input.len();
        start >= base && start + s.len() <= buf_end
    }

    const SIMPLE_DOC: &str = r#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/">
      <dc:title>Hello World</dc:title>
      <dc:creator>Jane Roe</dc:creator>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"#;

    #[test]
    fn test_parse_borrowed_values_are_zero_copy() {
        let input = SIMPLE_DOC.as_bytes();
        let view = parse_borrowed(input).expect("borrowed parse should succeed");

        let title = view.get("dc:title").expect("dc:title present");
        assert_eq!(title.as_text(), Some("Hello World"));
        // The value must be a true borrow that points into the input buffer.
        assert!(title.is_borrowed(), "clean value must stay borrowed");
        assert!(
            slice_within(input, title.as_text().expect("text")),
            "value slice must point into the input buffer (zero-copy)"
        );

        let creator = view.get("dc:creator").expect("dc:creator present");
        assert_eq!(creator.as_text(), Some("Jane Roe"));
        assert!(creator.is_borrowed());
        assert!(slice_within(input, creator.as_text().expect("text")));

        assert!(
            view.is_fully_borrowed(),
            "whole clean doc must be zero-copy"
        );
    }

    #[test]
    fn test_parse_borrowed_keys_are_zero_copy() {
        let input = SIMPLE_DOC.as_bytes();
        let view = parse_borrowed(input).expect("borrowed parse should succeed");

        // Every key must be a `&str` slice pointing into the input buffer.
        for (key, _) in view.fields() {
            assert!(
                slice_within(input, key),
                "key {key:?} must point into the input buffer (zero-copy)"
            );
        }
        // And the keys must be the expected ones.
        let mut keys: Vec<&str> = view.fields().iter().map(|(k, _)| *k).collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["dc:creator", "dc:title"]);
    }

    #[test]
    fn test_parse_borrowed_entity_value_is_owned() {
        let input = SIMPLE_DOC.replace("Hello World", "Rock &amp; Roll");
        let bytes = input.as_bytes();
        let view = parse_borrowed(bytes).expect("borrowed parse should succeed");

        let title = view.get("dc:title").expect("dc:title present");
        // The entity must be decoded ...
        assert_eq!(title.as_text(), Some("Rock & Roll"));
        // ... which forces the owned fallback (a plain slice cannot represent
        // unescaped text), so it is NOT borrowed.
        assert!(
            !title.is_borrowed(),
            "entity-bearing value must fall back to Cow::Owned"
        );
        assert!(
            !slice_within(bytes, title.as_text().expect("text")),
            "owned value must not point into the input buffer"
        );
        // The clean sibling stays borrowed even though a sibling is owned.
        let creator = view.get("dc:creator").expect("dc:creator present");
        assert!(creator.is_borrowed());
        assert!(!view.is_fully_borrowed());
    }

    #[test]
    fn test_parse_borrowed_trimmed_value_stays_borrowed() {
        // Surrounding whitespace must be trimmed, yet the result must remain a
        // borrowed *sub-slice* of the input (no allocation).
        let doc = r#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/">
      <dc:rights>
          Copyright 2026
      </dc:rights>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"#;
        let input = doc.as_bytes();
        let view = parse_borrowed(input).expect("borrowed parse should succeed");

        let rights = view.get("dc:rights").expect("dc:rights present");
        assert_eq!(rights.as_text(), Some("Copyright 2026"));
        assert!(
            rights.is_borrowed(),
            "trimmed clean value must remain a borrowed sub-slice"
        );
        assert!(slice_within(input, rights.as_text().expect("text")));
    }

    #[test]
    fn test_parse_borrowed_multibyte_utf8_is_borrowed() {
        // Multi-byte UTF-8 must never be sliced mid-codepoint and must remain a
        // valid zero-copy borrow.
        let doc = r#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/">
      <dc:title>日本語のタイトル — café</dc:title>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"#;
        let input = doc.as_bytes();
        let view = parse_borrowed(input).expect("borrowed parse should succeed");

        let title = view.get("dc:title").expect("dc:title present");
        assert_eq!(title.as_text(), Some("日本語のタイトル — café"));
        assert!(
            title.is_borrowed(),
            "multi-byte clean value must stay borrowed"
        );
        assert!(slice_within(input, title.as_text().expect("text")));
    }

    #[test]
    fn test_parse_borrowed_array_items_are_zero_copy() {
        let input = SIMPLE_DOC.replace(
            "<dc:creator>Jane Roe</dc:creator>",
            "<dc:subject>\n        <rdf:Seq>\n          <rdf:li>landscape</rdf:li>\n          <rdf:li>nature</rdf:li>\n        </rdf:Seq>\n      </dc:subject>",
        );
        let bytes = input.as_bytes();
        let view = parse_borrowed(bytes).expect("borrowed parse should succeed");

        let subjects = view
            .get("dc:subject")
            .and_then(XmpValue::as_text_list)
            .expect("dc:subject should be a list");
        assert_eq!(subjects.len(), 2);
        assert_eq!(subjects[0].as_ref(), "landscape");
        assert_eq!(subjects[1].as_ref(), "nature");
        // Each array item is a borrowed slice into the input buffer.
        for item in subjects {
            assert!(
                matches!(item, Cow::Borrowed(_)),
                "array item must be borrowed"
            );
            assert!(slice_within(bytes, item.as_ref()));
        }
    }

    #[test]
    fn test_parse_borrowed_matches_owned() {
        // The owned `parse` delegates to `parse_borrowed`, so the materialised
        // metadata must agree field-for-field with the borrowed view across a
        // representative mix of simple, entity-bearing and array properties.
        let docs = [
            SIMPLE_DOC.to_string(),
            SIMPLE_DOC.replace("Hello World", "Rock &amp; Roll"),
            SIMPLE_DOC.replace(
                "<dc:creator>Jane Roe</dc:creator>",
                "<dc:subject>\n        <rdf:Bag>\n          <rdf:li>a</rdf:li>\n          <rdf:li>b &amp; c</rdf:li>\n        </rdf:Bag>\n      </dc:subject>",
            ),
        ];

        for doc in &docs {
            let bytes = doc.as_bytes();
            let owned = parse(bytes).expect("owned parse should succeed");
            let view = parse_borrowed(bytes).expect("borrowed parse should succeed");

            // Same number of fields.
            assert_eq!(
                view.len(),
                owned.fields().len(),
                "field counts must agree for {doc}"
            );
            // Borrowed view -> owned conversion must equal the direct owned parse.
            let reowned = view.clone().into_owned_metadata();
            for (key, value) in reowned.fields() {
                assert_eq!(
                    owned.get(key),
                    Some(value),
                    "value for {key} must agree between owned and borrowed parses"
                );
            }
            // And the borrowed view's accessors agree with the owned values.
            for (key, value) in owned.fields() {
                match value {
                    MetadataValue::Text(text) => {
                        assert_eq!(
                            view.get(key).and_then(XmpValue::as_text),
                            Some(text.as_str())
                        );
                    }
                    MetadataValue::TextList(items) => {
                        let borrowed = view
                            .get(key)
                            .and_then(XmpValue::as_text_list)
                            .expect("list");
                        let collected: Vec<&str> = borrowed.iter().map(Cow::as_ref).collect();
                        let expected: Vec<&str> = items.iter().map(String::as_str).collect();
                        assert_eq!(collected, expected);
                    }
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn test_parse_borrowed_empty_and_whitespace_values_skipped() {
        // Empty and whitespace-only properties must produce no field, matching
        // the owned parser's behaviour.
        let doc = r#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/">
      <dc:empty></dc:empty>
      <dc:blank>   </dc:blank>
      <dc:real>value</dc:real>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"#;
        let view = parse_borrowed(doc.as_bytes()).expect("borrowed parse should succeed");
        assert!(
            view.get("dc:empty").is_none(),
            "empty property must be skipped"
        );
        assert!(
            view.get("dc:blank").is_none(),
            "whitespace property must be skipped"
        );
        assert_eq!(
            view.get("dc:real").and_then(XmpValue::as_text),
            Some("value")
        );
        assert_eq!(view.len(), 1);
    }

    #[test]
    fn test_parse_borrowed_get_is_last_wins() {
        // Duplicate keys: `get` returns the last occurrence, matching the
        // HashMap last-insert-wins semantics of the owned parser.
        let doc = SIMPLE_DOC.replace(
            "<dc:creator>Jane Roe</dc:creator>",
            "<dc:title>Second</dc:title>",
        );
        let view = parse_borrowed(doc.as_bytes()).expect("borrowed parse should succeed");
        assert_eq!(
            view.get("dc:title").and_then(XmpValue::as_text),
            Some("Second")
        );

        let owned = parse(doc.as_bytes()).expect("owned parse should succeed");
        assert_eq!(
            owned.get("dc:title").and_then(MetadataValue::as_text),
            Some("Second"),
            "owned parser must agree on last-wins"
        );
    }

    #[test]
    fn test_parse_borrowed_invalid_utf8_errors_cleanly() {
        // Invalid UTF-8 input must surface a clean error, never a panic.
        let bad = [0xFF, 0xFE, 0x00, 0x01];
        let err = parse_borrowed(&bad).expect_err("invalid UTF-8 must error");
        assert!(matches!(err, Error::Xml(_)));
    }

    #[test]
    fn test_parse_borrowed_malformed_xml_errors_cleanly() {
        // A mismatched end tag is rejected by quick_xml's end-name checking and
        // must surface a clean error, never a panic.
        let malformed = b"<a:x><dc:title>v</dc:title></b:y>";
        let err = parse_borrowed(malformed).expect_err("malformed XML must error");
        assert!(matches!(err, Error::Xml(_)));
    }

    #[test]
    fn test_parse_borrowed_missing_namespaces_is_lenient() {
        // No declared namespaces: the parser keys on the prefixed element name
        // regardless, mirroring the owned parser (which does no namespace
        // resolution either).
        let doc = "<dc:title>NoNamespaces</dc:title><dc:author>Anon</dc:author>";
        let view = parse_borrowed(doc.as_bytes()).expect("borrowed parse should succeed");
        assert_eq!(
            view.get("dc:title").and_then(XmpValue::as_text),
            Some("NoNamespaces")
        );
        assert_eq!(
            view.get("dc:author").and_then(XmpValue::as_text),
            Some("Anon")
        );
        assert!(view.is_fully_borrowed());
    }

    #[test]
    fn test_anchor_name_rejects_foreign_slice() {
        // Defensive: a slice that is not part of the haystack must be rejected
        // cleanly (no panic), exercising the non-anchored error path.
        let haystack = "dc:title";
        let foreign = String::from("dc:title");
        let err = anchor_name(haystack, foreign.as_bytes()).expect_err("foreign slice rejected");
        assert!(matches!(err, Error::Xml(_)));

        // And a genuine sub-slice is accepted and re-anchored to the haystack.
        let sub = &haystack.as_bytes()[3..]; // "title"
        let anchored = anchor_name(haystack, sub).expect("sub-slice accepted");
        assert_eq!(anchored, "title");
        assert!(slice_within(haystack.as_bytes(), anchored));
    }
}
