//! Matroska tag parsing and writing support.
//!
//! Matroska (MKV) and WebM use an XML-based tag format defined by the
//! Matroska specification.  The full tag hierarchy is:
//!
//! ```xml
//! <Tags>
//!   <Tag>
//!     <Targets>
//!       <TargetTypeValue>50</TargetTypeValue>   <!-- 50 = Album, 30 = Track -->
//!       <TargetType>ALBUM</TargetType>
//!       <TrackUID>1234</TrackUID>
//!       <EditionUID>0</EditionUID>
//!       <ChapterUID>0</ChapterUID>
//!       <AttachmentUID>0</AttachmentUID>
//!     </Targets>
//!     <SimpleTag>
//!       <Name>TITLE</Name>
//!       <String>My Album</String>
//!       <Language>eng</Language>
//!       <Default>1</Default>
//!       <SimpleTag>                <!-- nested SimpleTag -->
//!         <Name>SORT_WITH</Name>
//!         <String>MyAlbum</String>
//!       </SimpleTag>
//!     </SimpleTag>
//!   </Tag>
//! </Tags>
//! ```
//!
//! # Simple Metadata API
//!
//! For compatibility with the rest of OxiMedia, the `parse()` / `write()`
//! functions flatten the hierarchy into `key → value` pairs in a `Metadata`
//! container.  For full structural access use `MatroskaTags::parse_xml()` and
//! the `MatroskaTag` / `MatroskaSimpleTag` types.
//!
//! # Common Tag Names
//!
//! - **TITLE**: Title
//! - **ARTIST**: Artist or performer
//! - **ALBUM**: Album name
//! - **DATE_RELEASED**: Release date (ISO 8601)
//! - **COMMENT**: Free-form comment
//! - **ENCODER**: Encoder software

use crate::{Error, Metadata, MetadataFormat, MetadataValue};
use quick_xml::events::{BytesEnd, BytesStart, BytesText, Event};
use quick_xml::{Reader, Writer};
use std::io::Cursor;

// ─────────────────────────────────────────────────────────────────────────────
// Target types
// ─────────────────────────────────────────────────────────────────────────────

/// Matroska Target Type Value — the logical level a `<Tag>` applies to.
///
/// Values are from the Matroska Tag specification section 12.1.
///
/// Note: The spec uses 30 for both "Shot/Scene" and "Track/Chapter" depending
/// on context. We represent both as `Track` (value 30) since they share the
/// same numeric discriminant and the spec treats them identically at the
/// serialisation level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetTypeValue {
    /// Track / Chapter / Issue / Volume / Shot / Scene (30)
    Track = 30,
    /// Part / Session (32)
    Part = 32,
    /// Album / Opera / Concert / Movie / Episode (50)
    Album = 50,
    /// Edition / Issue / Volume (60)
    Edition = 60,
    /// Collection (70)
    Collection = 70,
    /// Other / unrecognised value not listed above
    Other = 0,
}

impl TargetTypeValue {
    /// The raw numeric value (may not match `Other`; use `as_u32()` instead).
    #[must_use]
    pub fn numeric_value(self) -> u32 {
        match self {
            Self::Track => 30,
            Self::Part => 32,
            Self::Album => 50,
            Self::Edition => 60,
            Self::Collection => 70,
            Self::Other => 0,
        }
    }

    /// Create from the integer value stored in the XML.
    #[must_use]
    pub fn from_u32(v: u32) -> Self {
        match v {
            30 => Self::Track,
            32 => Self::Part,
            50 => Self::Album,
            60 => Self::Edition,
            70 => Self::Collection,
            _ => Self::Other,
        }
    }

    /// Return the integer value.
    #[must_use]
    pub fn as_u32(self) -> u32 {
        self.numeric_value()
    }
}

impl Default for TargetTypeValue {
    /// Default target type is 50 (Album / Movie) per the Matroska spec.
    fn default() -> Self {
        Self::Album
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Targets
// ─────────────────────────────────────────────────────────────────────────────

/// The `<Targets>` element specifies what a `<Tag>` applies to.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MatroskaTargets {
    /// Logical level (TargetTypeValue).
    pub target_type_value: Option<TargetTypeValue>,
    /// Human-readable name of the target type (e.g. `"ALBUM"`, `"TRACK"`).
    pub target_type: Option<String>,
    /// One or more UIDs identifying specific tracks.
    pub track_uids: Vec<u64>,
    /// One or more UIDs identifying specific editions.
    pub edition_uids: Vec<u64>,
    /// One or more UIDs identifying specific chapters.
    pub chapter_uids: Vec<u64>,
    /// One or more UIDs identifying specific attachments.
    pub attachment_uids: Vec<u64>,
}

impl MatroskaTargets {
    /// Return `true` if no UID is set (tag applies to the whole file).
    #[must_use]
    pub fn is_global(&self) -> bool {
        self.track_uids.is_empty()
            && self.edition_uids.is_empty()
            && self.chapter_uids.is_empty()
            && self.attachment_uids.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SimpleTag
// ─────────────────────────────────────────────────────────────────────────────

/// A `<SimpleTag>` element — a named value with optional language and nesting.
///
/// SimpleTag elements can be nested: a parent SimpleTag can contain child
/// SimpleTags that provide related or translated values.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MatroskaSimpleTag {
    /// The tag name (e.g. `"TITLE"`, `"ARTIST"`).  Required.
    pub name: String,
    /// The string value (empty string if only binary data is present).
    pub string_value: Option<String>,
    /// The binary value (alternative to `string_value`).
    pub binary_value: Option<Vec<u8>>,
    /// BCP-47 language code (default `"und"` = undetermined).
    pub language: String,
    /// Whether this is the default translation (0 or 1).
    pub default: bool,
    /// Nested SimpleTags (e.g. a `SORT_WITH` sub-tag).
    pub children: Vec<MatroskaSimpleTag>,
}

impl MatroskaSimpleTag {
    /// Create a new SimpleTag with the given name and string value.
    #[must_use]
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            string_value: Some(value.into()),
            language: "und".to_string(),
            default: true,
            ..Default::default()
        }
    }

    /// Add a nested child SimpleTag and return `self`.
    #[must_use]
    pub fn with_child(mut self, child: MatroskaSimpleTag) -> Self {
        self.children.push(child);
        self
    }

    /// Recursively collect all `(name, string_value)` pairs into `out`,
    /// with nested names joined by `"."` (e.g. `"TITLE.SORT_WITH"`).
    pub fn collect_flat(&self, prefix: &str, out: &mut Vec<(String, String)>) {
        let full_name = if prefix.is_empty() {
            self.name.clone()
        } else {
            format!("{prefix}.{}", self.name)
        };
        if let Some(ref s) = self.string_value {
            out.push((full_name.clone(), s.clone()));
        }
        for child in &self.children {
            child.collect_flat(&full_name, out);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tag (one <Tag> element)
// ─────────────────────────────────────────────────────────────────────────────

/// One `<Tag>` element in a Matroska file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MatroskaTag {
    /// The `<Targets>` element (which entities this tag applies to).
    pub targets: MatroskaTargets,
    /// The list of `<SimpleTag>` elements inside this `<Tag>`.
    pub simple_tags: Vec<MatroskaSimpleTag>,
}

impl MatroskaTag {
    /// Flatten all SimpleTags (including nested children) into `(name, value)` pairs.
    #[must_use]
    pub fn flatten(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for st in &self.simple_tags {
            st.collect_flat("", &mut out);
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MatroskaTags — structured container
// ─────────────────────────────────────────────────────────────────────────────

/// The complete collection of `<Tags>` from a Matroska file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MatroskaTags {
    /// All `<Tag>` elements.
    pub tags: Vec<MatroskaTag>,
}

impl MatroskaTags {
    /// Parse `<Tags>` XML into a structured `MatroskaTags`.
    ///
    /// # Errors
    ///
    /// Returns an error if the XML is malformed.
    pub fn parse_xml(data: &[u8]) -> Result<Self, Error> {
        let mut reader = Reader::from_reader(Cursor::new(data));
        reader.config_mut().trim_text(true);

        let mut buf = Vec::new();
        let mut result = Self::default();

        // Parser state — we model the nesting with an explicit state machine.
        // `in_tag` / `in_targets` / `simple_tag_stack` track current depth.
        let mut current_tag: Option<MatroskaTag> = None;
        let mut in_targets = false;
        let mut in_targets_field: Option<String> = None;

        // Stack for nested SimpleTags: each entry is a partially-built SimpleTag.
        // When we close a <SimpleTag> we pop it, push as child of the new top,
        // or push into current_tag.simple_tags if the stack is empty.
        let mut simple_tag_stack: Vec<MatroskaSimpleTag> = Vec::new();
        // Track which field inside a SimpleTag we are reading
        let mut in_simple_field: Option<String> = None;

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    let local = String::from_utf8_lossy(e.name().as_ref()).into_owned();
                    match local.as_str() {
                        "Tag" => {
                            current_tag = Some(MatroskaTag::default());
                        }
                        "Targets" if current_tag.is_some() => {
                            in_targets = true;
                        }
                        "TargetTypeValue" | "TargetType" | "TrackUID" | "EditionUID"
                        | "ChapterUID" | "AttachmentUID"
                            if in_targets =>
                        {
                            in_targets_field = Some(local);
                        }
                        "SimpleTag" if current_tag.is_some() => {
                            simple_tag_stack.push(MatroskaSimpleTag {
                                language: "und".to_string(),
                                ..Default::default()
                            });
                        }
                        "Name" | "String" | "Binary" | "Language" | "Default"
                            if !simple_tag_stack.is_empty() =>
                        {
                            in_simple_field = Some(local);
                        }
                        _ => {}
                    }
                }
                Ok(Event::Text(ref e)) => {
                    let text = reader
                        .decoder()
                        .decode(e.as_ref())
                        .map_err(|err| Error::Xml(format!("Decode error: {err}")))?
                        .into_owned();

                    if in_targets {
                        if let (Some(ref field), Some(ref mut tag)) =
                            (&in_targets_field, &mut current_tag)
                        {
                            match field.as_str() {
                                "TargetTypeValue" => {
                                    if let Ok(v) = text.trim().parse::<u32>() {
                                        tag.targets.target_type_value =
                                            Some(TargetTypeValue::from_u32(v));
                                    }
                                }
                                "TargetType" => {
                                    tag.targets.target_type = Some(text.trim().to_string());
                                }
                                "TrackUID" => {
                                    if let Ok(v) = text.trim().parse::<u64>() {
                                        tag.targets.track_uids.push(v);
                                    }
                                }
                                "EditionUID" => {
                                    if let Ok(v) = text.trim().parse::<u64>() {
                                        tag.targets.edition_uids.push(v);
                                    }
                                }
                                "ChapterUID" => {
                                    if let Ok(v) = text.trim().parse::<u64>() {
                                        tag.targets.chapter_uids.push(v);
                                    }
                                }
                                "AttachmentUID" => {
                                    if let Ok(v) = text.trim().parse::<u64>() {
                                        tag.targets.attachment_uids.push(v);
                                    }
                                }
                                _ => {}
                            }
                        }
                    } else if let Some(ref field) = in_simple_field {
                        if let Some(top) = simple_tag_stack.last_mut() {
                            match field.as_str() {
                                "Name" => top.name = text.trim().to_string(),
                                "String" => top.string_value = Some(text.trim().to_string()),
                                "Language" => top.language = text.trim().to_string(),
                                "Default" => top.default = text.trim() == "1",
                                _ => {}
                            }
                        }
                    }
                }
                Ok(Event::End(ref e)) => {
                    let local = String::from_utf8_lossy(e.name().as_ref()).into_owned();
                    match local.as_str() {
                        "Tag" => {
                            if let Some(tag) = current_tag.take() {
                                result.tags.push(tag);
                            }
                            simple_tag_stack.clear();
                        }
                        "Targets" => {
                            in_targets = false;
                            in_targets_field = None;
                        }
                        "TargetTypeValue" | "TargetType" | "TrackUID" | "EditionUID"
                        | "ChapterUID" | "AttachmentUID" => {
                            in_targets_field = None;
                        }
                        "SimpleTag" => {
                            in_simple_field = None;
                            // Pop the completed SimpleTag
                            if let Some(completed) = simple_tag_stack.pop() {
                                if let Some(parent) = simple_tag_stack.last_mut() {
                                    // Nested: add as child of the enclosing SimpleTag
                                    parent.children.push(completed);
                                } else if let Some(ref mut tag) = current_tag {
                                    // Top-level SimpleTag: add to Tag
                                    tag.simple_tags.push(completed);
                                }
                            }
                        }
                        "Name" | "String" | "Binary" | "Language" | "Default" => {
                            in_simple_field = None;
                        }
                        _ => {}
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(Error::Xml(format!("XML parse error: {e}"))),
                _ => {}
            }
            buf.clear();
        }

        Ok(result)
    }

    /// Serialise the structured tags back to XML.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    pub fn write_xml(&self) -> Result<Vec<u8>, Error> {
        let mut writer = Writer::new(Cursor::new(Vec::new()));

        write_start(&mut writer, "Tags")?;

        for tag in &self.tags {
            write_start(&mut writer, "Tag")?;

            // Targets
            write_start(&mut writer, "Targets")?;
            if let Some(ttv) = tag.targets.target_type_value {
                write_text_element(&mut writer, "TargetTypeValue", &ttv.as_u32().to_string())?;
            }
            if let Some(ref tt) = tag.targets.target_type {
                write_text_element(&mut writer, "TargetType", tt)?;
            }
            for uid in &tag.targets.track_uids {
                write_text_element(&mut writer, "TrackUID", &uid.to_string())?;
            }
            for uid in &tag.targets.edition_uids {
                write_text_element(&mut writer, "EditionUID", &uid.to_string())?;
            }
            for uid in &tag.targets.chapter_uids {
                write_text_element(&mut writer, "ChapterUID", &uid.to_string())?;
            }
            for uid in &tag.targets.attachment_uids {
                write_text_element(&mut writer, "AttachmentUID", &uid.to_string())?;
            }
            write_end(&mut writer, "Targets")?;

            // SimpleTags (possibly nested)
            for st in &tag.simple_tags {
                write_simple_tag(&mut writer, st)?;
            }

            write_end(&mut writer, "Tag")?;
        }

        write_end(&mut writer, "Tags")?;

        Ok(writer.into_inner().into_inner())
    }

    /// Convert to a flat `Metadata` container.
    ///
    /// When the same name appears in multiple `<Tag>` elements (e.g. because
    /// they target different entities) the values are joined with `'\n'`.
    #[must_use]
    pub fn to_metadata(&self) -> Metadata {
        let mut metadata = Metadata::new(MetadataFormat::Matroska);
        for tag in &self.tags {
            for (name, value) in tag.flatten() {
                if let Some(existing) = metadata.get(&name) {
                    if let MetadataValue::Text(ref existing_text) = existing.clone() {
                        metadata.insert(
                            name,
                            MetadataValue::Text(format!("{existing_text}\n{value}")),
                        );
                        continue;
                    }
                }
                metadata.insert(name, MetadataValue::Text(value));
            }
        }
        metadata
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// XML writing helpers
// ─────────────────────────────────────────────────────────────────────────────

fn write_start(writer: &mut Writer<Cursor<Vec<u8>>>, name: &str) -> Result<(), Error> {
    writer
        .write_event(Event::Start(BytesStart::new(name)))
        .map_err(|e| Error::Xml(format!("Failed to write <{name}>: {e}")))
}

fn write_end(writer: &mut Writer<Cursor<Vec<u8>>>, name: &str) -> Result<(), Error> {
    writer
        .write_event(Event::End(BytesEnd::new(name)))
        .map_err(|e| Error::Xml(format!("Failed to write </{name}>: {e}")))
}

fn write_text_element(
    writer: &mut Writer<Cursor<Vec<u8>>>,
    name: &str,
    text: &str,
) -> Result<(), Error> {
    write_start(writer, name)?;
    writer
        .write_event(Event::Text(BytesText::new(text)))
        .map_err(|e| Error::Xml(format!("Failed to write text in <{name}>: {e}")))?;
    write_end(writer, name)
}

/// Recursively serialise a `MatroskaSimpleTag` (including its children).
fn write_simple_tag(
    writer: &mut Writer<Cursor<Vec<u8>>>,
    st: &MatroskaSimpleTag,
) -> Result<(), Error> {
    write_start(writer, "SimpleTag")?;

    write_text_element(writer, "Name", &st.name)?;

    if let Some(ref s) = st.string_value {
        write_text_element(writer, "String", s)?;
    }

    if st.language != "und" && !st.language.is_empty() {
        write_text_element(writer, "Language", &st.language)?;
    }

    write_text_element(writer, "Default", if st.default { "1" } else { "0" })?;

    // Nested children
    for child in &st.children {
        write_simple_tag(writer, child)?;
    }

    write_end(writer, "SimpleTag")
}

// ─────────────────────────────────────────────────────────────────────────────
// Public `parse()` / `write()` (flat Metadata API)
// ─────────────────────────────────────────────────────────────────────────────

/// Parse Matroska tags from XML data into a flat `Metadata` container.
///
/// # Errors
///
/// Returns an error if the data is not valid Matroska XML.
pub fn parse(data: &[u8]) -> Result<Metadata, Error> {
    let structured = MatroskaTags::parse_xml(data)?;
    Ok(structured.to_metadata())
}

/// Write a flat `Metadata` container to Matroska XML.
///
/// All fields are written as top-level `<SimpleTag>` elements inside a single
/// `<Tag>` with default targets.
///
/// # Errors
///
/// Returns an error if writing fails.
pub fn write(metadata: &Metadata) -> Result<Vec<u8>, Error> {
    // Build a structured representation and serialise it
    let mut tag = MatroskaTag::default();

    for (key, value) in metadata.fields() {
        let string_value = match value {
            MetadataValue::Text(s) => s.clone(),
            MetadataValue::Integer(i) => i.to_string(),
            MetadataValue::Float(f) => f.to_string(),
            MetadataValue::Boolean(b) => (if *b { "1" } else { "0" }).to_string(),
            _ => continue,
        };
        tag.simple_tags
            .push(MatroskaSimpleTag::new(key, string_value));
    }

    let tags = MatroskaTags { tags: vec![tag] };
    tags.write_xml()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Flat parse / write (backwards-compat) ────────────────────────────

    #[test]
    fn test_matroska_round_trip() {
        let mut metadata = Metadata::new(MetadataFormat::Matroska);

        metadata.insert(
            "TITLE".to_string(),
            MetadataValue::Text("Test Title".to_string()),
        );
        metadata.insert(
            "ARTIST".to_string(),
            MetadataValue::Text("Test Artist".to_string()),
        );
        metadata.insert(
            "ALBUM".to_string(),
            MetadataValue::Text("Test Album".to_string()),
        );

        let data = write(&metadata).expect("Write failed");
        let parsed = parse(&data).expect("Parse failed");

        assert_eq!(
            parsed.get("TITLE").and_then(|v| v.as_text()),
            Some("Test Title")
        );
        assert_eq!(
            parsed.get("ARTIST").and_then(|v| v.as_text()),
            Some("Test Artist")
        );
        assert_eq!(
            parsed.get("ALBUM").and_then(|v| v.as_text()),
            Some("Test Album")
        );
    }

    #[test]
    fn test_matroska_empty() {
        let metadata = Metadata::new(MetadataFormat::Matroska);
        let data = write(&metadata).expect("Write failed");
        let data_str = String::from_utf8_lossy(&data);
        assert!(data_str.contains("<Tags>"));
        assert!(data_str.contains("</Tags>"));
    }

    // ─── TargetTypeValue ──────────────────────────────────────────────────

    #[test]
    fn test_target_type_value_roundtrip() {
        assert_eq!(TargetTypeValue::from_u32(50), TargetTypeValue::Album);
        assert_eq!(TargetTypeValue::Album.as_u32(), 50);
        assert_eq!(TargetTypeValue::from_u32(30), TargetTypeValue::Track);
        assert_eq!(TargetTypeValue::Track.as_u32(), 30);
        assert_eq!(TargetTypeValue::from_u32(99), TargetTypeValue::Other);
        // Other's as_u32() returns the enum discriminant (0), not the input value
        assert_eq!(TargetTypeValue::Other.as_u32(), 0);
    }

    #[test]
    fn test_target_type_value_default_is_album() {
        assert_eq!(TargetTypeValue::default(), TargetTypeValue::Album);
    }

    // ─── MatroskaTargets ──────────────────────────────────────────────────

    #[test]
    fn test_targets_is_global_when_no_uids() {
        let t = MatroskaTargets::default();
        assert!(t.is_global());
    }

    #[test]
    fn test_targets_not_global_when_track_uid() {
        let mut t = MatroskaTargets::default();
        t.track_uids.push(42);
        assert!(!t.is_global());
    }

    // ─── MatroskaSimpleTag ────────────────────────────────────────────────

    #[test]
    fn test_simple_tag_new() {
        let st = MatroskaSimpleTag::new("TITLE", "My Movie");
        assert_eq!(st.name, "TITLE");
        assert_eq!(st.string_value.as_deref(), Some("My Movie"));
        assert_eq!(st.language, "und");
        assert!(st.default);
        assert!(st.children.is_empty());
    }

    #[test]
    fn test_simple_tag_with_child() {
        let child = MatroskaSimpleTag::new("SORT_WITH", "MyMovie");
        let parent = MatroskaSimpleTag::new("TITLE", "My Movie").with_child(child);
        assert_eq!(parent.children.len(), 1);
        assert_eq!(parent.children[0].name, "SORT_WITH");
    }

    #[test]
    fn test_simple_tag_collect_flat_no_children() {
        let st = MatroskaSimpleTag::new("TITLE", "Flat");
        let mut out = Vec::new();
        st.collect_flat("", &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], ("TITLE".to_string(), "Flat".to_string()));
    }

    #[test]
    fn test_simple_tag_collect_flat_with_children() {
        let child = MatroskaSimpleTag::new("SORT_WITH", "Flat");
        let parent = MatroskaSimpleTag::new("TITLE", "My Title").with_child(child);
        let mut out = Vec::new();
        parent.collect_flat("", &mut out);
        // Should produce ("TITLE", "My Title") and ("TITLE.SORT_WITH", "Flat")
        assert_eq!(out.len(), 2);
        assert!(out.iter().any(|(k, _)| k == "TITLE"));
        assert!(out.iter().any(|(k, _)| k == "TITLE.SORT_WITH"));
    }

    // ─── MatroskaTag ─────────────────────────────────────────────────────

    #[test]
    fn test_matroska_tag_flatten_empty() {
        let tag = MatroskaTag::default();
        assert!(tag.flatten().is_empty());
    }

    #[test]
    fn test_matroska_tag_flatten_single() {
        let mut tag = MatroskaTag::default();
        tag.simple_tags
            .push(MatroskaSimpleTag::new("ARTIST", "Beethoven"));
        let flat = tag.flatten();
        assert_eq!(flat.len(), 1);
        assert_eq!(flat[0], ("ARTIST".to_string(), "Beethoven".to_string()));
    }

    // ─── MatroskaTags::parse_xml ──────────────────────────────────────────

    #[test]
    fn test_parse_xml_empty_tags() {
        let xml = b"<Tags></Tags>";
        let result = MatroskaTags::parse_xml(xml).expect("Parse should succeed");
        assert!(result.tags.is_empty());
    }

    #[test]
    fn test_parse_xml_simple_tag_no_targets() {
        let xml = br#"<Tags>
  <Tag>
    <SimpleTag>
      <Name>TITLE</Name>
      <String>My Song</String>
    </SimpleTag>
  </Tag>
</Tags>"#;
        let result = MatroskaTags::parse_xml(xml).expect("Parse should succeed");
        assert_eq!(result.tags.len(), 1);
        let tag = &result.tags[0];
        assert_eq!(tag.simple_tags.len(), 1);
        assert_eq!(tag.simple_tags[0].name, "TITLE");
        assert_eq!(tag.simple_tags[0].string_value.as_deref(), Some("My Song"));
    }

    #[test]
    fn test_parse_xml_targets_target_type_value() {
        let xml = br#"<Tags>
  <Tag>
    <Targets>
      <TargetTypeValue>50</TargetTypeValue>
      <TargetType>ALBUM</TargetType>
    </Targets>
    <SimpleTag>
      <Name>TITLE</Name>
      <String>My Album</String>
    </SimpleTag>
  </Tag>
</Tags>"#;
        let result = MatroskaTags::parse_xml(xml).expect("Parse should succeed");
        let tag = &result.tags[0];
        assert_eq!(tag.targets.target_type_value, Some(TargetTypeValue::Album));
        assert_eq!(tag.targets.target_type.as_deref(), Some("ALBUM"));
    }

    #[test]
    fn test_parse_xml_targets_track_uid() {
        let xml = br#"<Tags>
  <Tag>
    <Targets>
      <TrackUID>123456789</TrackUID>
    </Targets>
    <SimpleTag>
      <Name>ARTIST</Name>
      <String>Mozart</String>
    </SimpleTag>
  </Tag>
</Tags>"#;
        let result = MatroskaTags::parse_xml(xml).expect("Parse should succeed");
        let tag = &result.tags[0];
        assert_eq!(tag.targets.track_uids, vec![123456789u64]);
        assert!(!tag.targets.is_global());
    }

    #[test]
    fn test_parse_xml_nested_simple_tag() {
        let xml = br#"<Tags>
  <Tag>
    <SimpleTag>
      <Name>TITLE</Name>
      <String>My Film</String>
      <SimpleTag>
        <Name>SORT_WITH</Name>
        <String>Film, My</String>
      </SimpleTag>
    </SimpleTag>
  </Tag>
</Tags>"#;
        let result = MatroskaTags::parse_xml(xml).expect("Parse should succeed");
        let tag = &result.tags[0];
        assert_eq!(tag.simple_tags.len(), 1);
        let top = &tag.simple_tags[0];
        assert_eq!(top.name, "TITLE");
        assert_eq!(top.children.len(), 1);
        assert_eq!(top.children[0].name, "SORT_WITH");
        assert_eq!(top.children[0].string_value.as_deref(), Some("Film, My"));
    }

    #[test]
    fn test_parse_xml_language_and_default() {
        // Use ASCII-safe XML for byte string literal; actual non-ASCII
        // would be fine as a &str literal but not inside br#""#.
        let xml = b"<Tags>\n  <Tag>\n    <SimpleTag>\n      <Name>TITLE</Name>\n      <String>Titre francais</String>\n      <Language>fra</Language>\n      <Default>0</Default>\n    </SimpleTag>\n  </Tag>\n</Tags>";
        let result = MatroskaTags::parse_xml(xml).expect("Parse should succeed");
        let st = &result.tags[0].simple_tags[0];
        assert_eq!(st.language, "fra");
        assert!(!st.default);
    }

    #[test]
    fn test_parse_xml_multiple_tags() {
        let xml = br#"<Tags>
  <Tag>
    <SimpleTag><Name>ARTIST</Name><String>Bach</String></SimpleTag>
  </Tag>
  <Tag>
    <Targets><TargetTypeValue>50</TargetTypeValue></Targets>
    <SimpleTag><Name>TITLE</Name><String>Album Title</String></SimpleTag>
  </Tag>
</Tags>"#;
        let result = MatroskaTags::parse_xml(xml).expect("Parse should succeed");
        assert_eq!(result.tags.len(), 2);
    }

    // ─── MatroskaTags::write_xml round-trip ───────────────────────────────

    #[test]
    fn test_write_xml_round_trip_simple() {
        let original = MatroskaTags {
            tags: vec![MatroskaTag {
                targets: MatroskaTargets {
                    target_type_value: Some(TargetTypeValue::Album),
                    target_type: Some("ALBUM".to_string()),
                    ..Default::default()
                },
                simple_tags: vec![
                    MatroskaSimpleTag::new("TITLE", "Round-trip Album"),
                    MatroskaSimpleTag::new("ARTIST", "Test Artist"),
                ],
            }],
        };

        let xml = original.write_xml().expect("Write should succeed");
        let parsed = MatroskaTags::parse_xml(&xml).expect("Parse should succeed");

        assert_eq!(parsed.tags.len(), 1);
        let tag = &parsed.tags[0];
        assert_eq!(tag.targets.target_type_value, Some(TargetTypeValue::Album));
        assert_eq!(tag.simple_tags.len(), 2);
    }

    #[test]
    fn test_write_xml_nested_simple_tag_round_trip() {
        let child = MatroskaSimpleTag::new("SORT_WITH", "Album Sort");
        let parent = MatroskaSimpleTag::new("TITLE", "Album Title").with_child(child);
        let tag = MatroskaTag {
            simple_tags: vec![parent],
            ..Default::default()
        };
        let tags = MatroskaTags { tags: vec![tag] };

        let xml = tags.write_xml().expect("Write should succeed");
        let parsed = MatroskaTags::parse_xml(&xml).expect("Parse should succeed");

        let top = &parsed.tags[0].simple_tags[0];
        assert_eq!(top.name, "TITLE");
        assert_eq!(top.children.len(), 1);
        assert_eq!(top.children[0].name, "SORT_WITH");
    }

    // ─── MatroskaTags::to_metadata ────────────────────────────────────────

    #[test]
    fn test_to_metadata_simple() {
        let tags = MatroskaTags {
            tags: vec![MatroskaTag {
                simple_tags: vec![
                    MatroskaSimpleTag::new("TITLE", "Test Song"),
                    MatroskaSimpleTag::new("ARTIST", "Test Artist"),
                ],
                ..Default::default()
            }],
        };

        let meta = tags.to_metadata();
        assert_eq!(
            meta.get("TITLE").and_then(|v| v.as_text()),
            Some("Test Song")
        );
        assert_eq!(
            meta.get("ARTIST").and_then(|v| v.as_text()),
            Some("Test Artist")
        );
    }

    #[test]
    fn test_to_metadata_nested_flattened() {
        let child = MatroskaSimpleTag::new("SORT_WITH", "Song, Test");
        let parent = MatroskaSimpleTag::new("TITLE", "Test Song").with_child(child);
        let tags = MatroskaTags {
            tags: vec![MatroskaTag {
                simple_tags: vec![parent],
                ..Default::default()
            }],
        };

        let meta = tags.to_metadata();
        assert_eq!(
            meta.get("TITLE").and_then(|v| v.as_text()),
            Some("Test Song")
        );
        // Nested key with dot notation
        assert_eq!(
            meta.get("TITLE.SORT_WITH").and_then(|v| v.as_text()),
            Some("Song, Test")
        );
    }

    #[test]
    fn test_to_metadata_duplicate_keys_joined() {
        // Two Tag elements with the same SimpleTag name
        let tags = MatroskaTags {
            tags: vec![
                MatroskaTag {
                    simple_tags: vec![MatroskaSimpleTag::new("ARTIST", "Alice")],
                    ..Default::default()
                },
                MatroskaTag {
                    simple_tags: vec![MatroskaSimpleTag::new("ARTIST", "Bob")],
                    ..Default::default()
                },
            ],
        };

        let meta = tags.to_metadata();
        let artist = meta.get("ARTIST").and_then(|v| v.as_text()).unwrap_or("");
        // Both artists should be present (joined by newline)
        assert!(artist.contains("Alice"));
        assert!(artist.contains("Bob"));
    }

    // ─── Integration: full structured XML round-trip via parse/write ──────

    #[test]
    fn test_full_structured_parse_write_round_trip() {
        let xml = br#"<Tags>
  <Tag>
    <Targets>
      <TargetTypeValue>50</TargetTypeValue>
      <TargetType>ALBUM</TargetType>
    </Targets>
    <SimpleTag>
      <Name>TITLE</Name>
      <String>Greatest Hits</String>
      <SimpleTag>
        <Name>SORT_WITH</Name>
        <String>Greatest Hits</String>
      </SimpleTag>
    </SimpleTag>
    <SimpleTag>
      <Name>ARTIST</Name>
      <String>Some Artist</String>
    </SimpleTag>
    <SimpleTag>
      <Name>DATE_RELEASED</Name>
      <String>2024-01-15</String>
    </SimpleTag>
  </Tag>
  <Tag>
    <Targets>
      <TargetTypeValue>30</TargetTypeValue>
      <TargetType>TRACK</TargetType>
      <TrackUID>1</TrackUID>
    </Targets>
    <SimpleTag>
      <Name>TITLE</Name>
      <String>Track One</String>
    </SimpleTag>
  </Tag>
</Tags>"#;

        let parsed = MatroskaTags::parse_xml(xml).expect("Initial parse should succeed");
        assert_eq!(parsed.tags.len(), 2);

        // Album tag
        let album_tag = &parsed.tags[0];
        assert_eq!(
            album_tag.targets.target_type_value,
            Some(TargetTypeValue::Album)
        );
        assert_eq!(album_tag.simple_tags.len(), 3);
        let title_st = album_tag
            .simple_tags
            .iter()
            .find(|s| s.name == "TITLE")
            .expect("TITLE should exist");
        assert_eq!(title_st.string_value.as_deref(), Some("Greatest Hits"));
        assert_eq!(title_st.children.len(), 1);
        assert_eq!(title_st.children[0].name, "SORT_WITH");

        // Track tag
        let track_tag = &parsed.tags[1];
        assert_eq!(track_tag.targets.track_uids, vec![1u64]);

        // Re-serialise and parse again
        let xml2 = parsed.write_xml().expect("Write should succeed");
        let parsed2 = MatroskaTags::parse_xml(&xml2).expect("Re-parse should succeed");
        assert_eq!(parsed2.tags.len(), 2);

        // Check nested tag survived the round-trip
        let title_st2 = parsed2.tags[0]
            .simple_tags
            .iter()
            .find(|s| s.name == "TITLE")
            .expect("TITLE should exist in round-tripped data");
        assert_eq!(title_st2.children.len(), 1);
        assert_eq!(title_st2.children[0].name, "SORT_WITH");
    }
}
