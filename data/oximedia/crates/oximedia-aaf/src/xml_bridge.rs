//! AAF-to-XML and XML-to-AAF bridge
//!
//! Provides round-trip serialization between AAF object model and XML, plus
//! a minimal bridge for extracting MXF timecodes from raw KLV byte streams.
//! No external XML crate is used; all parsing is done via string scanning.
//!
//! # Standalone types
//!
//! The `AafComposition`, `AafTrack`, and `AafSlot` types provide a lightweight
//! representation that can be serialised/parsed without depending on the full
//! AAF object model.

use crate::composition::{CompositionMob, Track};
use crate::object_model::MobSlot;
use crate::AafError;
use crate::Result;

// ─── Standalone AAF types ─────────────────────────────────────────────────────

/// Slot type for standalone AAF XML representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AafSlotType {
    /// Video (picture) slot.
    Video,
    /// Audio (sound) slot.
    Audio,
    /// Timecode slot.
    Timecode,
    /// Data / auxiliary slot.
    Data,
}

impl AafSlotType {
    /// Convert to a short XML attribute string.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Video => "Video",
            Self::Audio => "Audio",
            Self::Timecode => "Timecode",
            Self::Data => "Data",
        }
    }

    /// Parse from a string (case-insensitive first character match).
    #[must_use]
    pub fn from_str_loose(s: &str) -> Self {
        let lower = s.to_ascii_lowercase();
        if lower.starts_with('v') || lower.starts_with("pic") {
            Self::Video
        } else if lower.starts_with('a') || lower.starts_with("sou") {
            Self::Audio
        } else if lower.starts_with('t') {
            Self::Timecode
        } else {
            Self::Data
        }
    }
}

/// A slot inside a standalone AAF track.
#[derive(Debug, Clone, PartialEq)]
pub struct AafSlot {
    /// Unique slot identifier within the track.
    pub slot_id: u32,
    /// Slot type.
    pub slot_type: AafSlotType,
    /// Start time in edit units.
    pub start_time: i64,
    /// Duration in edit units.
    pub duration: i64,
    /// Optional reference to a source mob (hex string, UUID, etc.).
    pub source_ref: Option<String>,
}

impl AafSlot {
    /// Create a new slot.
    #[must_use]
    pub fn new(slot_id: u32, slot_type: AafSlotType, start_time: i64, duration: i64) -> Self {
        Self {
            slot_id,
            slot_type,
            start_time,
            duration,
            source_ref: None,
        }
    }

    /// Attach a source reference.
    #[must_use]
    pub fn with_source_ref(mut self, source_ref: impl Into<String>) -> Self {
        self.source_ref = Some(source_ref.into());
        self
    }

    /// End time (start_time + duration).
    #[must_use]
    pub fn end_time(&self) -> i64 {
        self.start_time.saturating_add(self.duration)
    }
}

/// A track inside a standalone AAF composition.
#[derive(Debug, Clone, PartialEq)]
pub struct AafTrack {
    /// Track ID (1-based by convention).
    pub track_id: u32,
    /// Track name (e.g. `"V1"`, `"A1"`).
    pub name: String,
    /// Slots belonging to this track.
    pub slots: Vec<AafSlot>,
}

impl AafTrack {
    /// Create a new track.
    #[must_use]
    pub fn new(track_id: u32, name: impl Into<String>) -> Self {
        Self {
            track_id,
            name: name.into(),
            slots: Vec::new(),
        }
    }

    /// Add a slot to this track.
    pub fn add_slot(&mut self, slot: AafSlot) {
        self.slots.push(slot);
    }

    /// Total duration across all slots.
    #[must_use]
    pub fn total_duration(&self) -> i64 {
        self.slots.iter().map(|s| s.duration).sum()
    }

    /// Number of slots.
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }
}

/// A standalone AAF composition suitable for XML serialization.
///
/// This is a lightweight alternative to `CompositionMob` that carries only the
/// data needed for XML interchange.
#[derive(Debug, Clone, PartialEq)]
pub struct AafComposition {
    /// Composition name.
    pub name: String,
    /// Duration in edit units.
    pub duration: i64,
    /// Edit rate numerator.
    pub edit_rate_num: u32,
    /// Edit rate denominator.
    pub edit_rate_den: u32,
    /// Tracks belonging to this composition.
    pub tracks: Vec<AafTrack>,
}

impl AafComposition {
    /// Create a new composition.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        duration: i64,
        edit_rate_num: u32,
        edit_rate_den: u32,
    ) -> Self {
        Self {
            name: name.into(),
            duration,
            edit_rate_num,
            edit_rate_den,
            tracks: Vec::new(),
        }
    }

    /// Add a track to this composition.
    pub fn add_track(&mut self, track: AafTrack) {
        self.tracks.push(track);
    }

    /// Number of tracks.
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Edit rate as a floating-point value.
    #[must_use]
    pub fn edit_rate_f64(&self) -> f64 {
        if self.edit_rate_den == 0 {
            return 0.0;
        }
        f64::from(self.edit_rate_num) / f64::from(self.edit_rate_den)
    }
}

// ─── Serializer ─────────────────────────────────────────────────────────────

/// Configuration for the AAF XML serializer.
#[derive(Debug, Clone)]
pub struct AafXmlSerializer {
    /// Number of spaces per indentation level.
    pub indent: usize,
    /// Whether to include essence (media) data references in output.
    pub serialize_essence: bool,
}

impl Default for AafXmlSerializer {
    fn default() -> Self {
        Self {
            indent: 2,
            serialize_essence: false,
        }
    }
}

impl AafXmlSerializer {
    /// Create a new serializer with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a serializer with explicit indent size.
    #[must_use]
    pub fn with_indent(mut self, indent: usize) -> Self {
        self.indent = indent;
        self
    }

    /// Toggle essence serialization.
    #[must_use]
    pub fn with_essence(mut self, serialize: bool) -> Self {
        self.serialize_essence = serialize;
        self
    }

    // ── internal helpers ───────────────────────────────────────────────────

    fn pad(&self, level: usize) -> String {
        " ".repeat(self.indent * level)
    }

    fn escape_xml(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;")
    }

    // ── public serialization API ───────────────────────────────────────────

    /// Serialize a slice of standalone `AafComposition`s into a self-contained
    /// AAF XML document.
    ///
    /// Each composition becomes a `<Composition>` element with nested `<Track>`
    /// and `<Slot>` children.
    #[must_use]
    pub fn serialize_to_xml(&self, compositions: &[AafComposition]) -> String {
        let pad0 = self.pad(0);
        let pad1 = self.pad(1);
        let pad2 = self.pad(2);
        let pad3 = self.pad(3);
        let pad4 = self.pad(4);
        let pad5 = self.pad(5);

        let mut out = String::new();
        out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        out.push_str(&format!("{pad0}<AAFFile version=\"1.1\">\n"));

        for comp in compositions {
            let name = Self::escape_xml(&comp.name);
            out.push_str(&format!(
                "{pad1}<Composition name=\"{name}\" duration=\"{}\" \
                 edit_rate_num=\"{}\" edit_rate_den=\"{}\">\n",
                comp.duration, comp.edit_rate_num, comp.edit_rate_den
            ));
            out.push_str(&format!("{pad2}<Tracks>\n"));

            for track in &comp.tracks {
                let track_name = Self::escape_xml(&track.name);
                out.push_str(&format!(
                    "{pad3}<Track id=\"{}\" name=\"{track_name}\">\n",
                    track.track_id
                ));
                out.push_str(&format!("{pad4}<Slots>\n"));

                for slot in &track.slots {
                    let slot_type_str = slot.slot_type.as_str();
                    out.push_str(&format!(
                        "{pad5}<Slot id=\"{}\" type=\"{slot_type_str}\" \
                         start_time=\"{}\" duration=\"{}\"",
                        slot.slot_id, slot.start_time, slot.duration
                    ));
                    if let Some(ref src_ref) = slot.source_ref {
                        let escaped_ref = Self::escape_xml(src_ref);
                        out.push_str(&format!(" source_ref=\"{escaped_ref}\""));
                    }
                    out.push_str("/>\n");
                }

                out.push_str(&format!("{pad4}</Slots>\n"));
                out.push_str(&format!("{pad3}</Track>\n"));
            }

            out.push_str(&format!("{pad2}</Tracks>\n"));
            out.push_str(&format!("{pad1}</Composition>\n"));
        }

        out.push_str(&format!("{pad0}</AAFFile>\n"));
        out
    }

    /// Serialize a single `Track` as a `<Track>` XML element.
    ///
    /// The track is rendered at indentation `level`.
    #[must_use]
    pub fn serialize_track(&self, track: &Track, level: usize) -> String {
        let pad = self.pad(level);
        let pad1 = self.pad(level + 1);
        let name = Self::escape_xml(&track.name);
        let track_type = format!("{:?}", track.track_type);
        let duration = track.duration().unwrap_or(0);
        let edit_num = track.edit_rate.numerator;
        let edit_den = track.edit_rate.denominator;

        let mut out = format!(
            "{pad}<Track id=\"{}\" track_type=\"{track_type}\">\n",
            track.track_id
        );
        out.push_str(&format!("{pad1}<Name>{name}</Name>\n"));
        out.push_str(&format!("{pad1}<Duration>{duration}</Duration>\n"));
        out.push_str(&format!(
            "{pad1}<EditRate numerator=\"{edit_num}\" denominator=\"{edit_den}\"/>\n"
        ));

        let clips = track.source_clips();
        if !clips.is_empty() {
            out.push_str(&format!("{pad1}<SourceClips>\n"));
            let pad2 = self.pad(level + 2);
            let pad3 = self.pad(level + 3);
            for clip in clips {
                out.push_str(&format!("{pad2}<SourceClip>\n"));
                out.push_str(&format!("{pad3}<Length>{}</Length>\n", clip.length));
                out.push_str(&format!(
                    "{pad3}<StartTime>{}</StartTime>\n",
                    clip.start_time.0
                ));
                out.push_str(&format!(
                    "{pad3}<SourceMobId>{}</SourceMobId>\n",
                    clip.source_mob_id
                ));
                out.push_str(&format!(
                    "{pad3}<SourceMobSlotId>{}</SourceMobSlotId>\n",
                    clip.source_mob_slot_id
                ));
                out.push_str(&format!("{pad2}</SourceClip>\n"));
            }
            out.push_str(&format!("{pad1}</SourceClips>\n"));
        }

        out.push_str(&format!("{pad}</Track>\n"));
        out
    }

    /// Serialize a `MobSlot` (from `object_model`) as a `<MobSlot>` element.
    ///
    /// The `id` attribute is a sequential index; `slot_id` is the AAF slot ID.
    #[must_use]
    pub fn serialize_mob_slot(&self, slot: &MobSlot, id: u32, level: usize) -> String {
        let pad = self.pad(level);
        let pad1 = self.pad(level + 1);
        let name = Self::escape_xml(&slot.name);
        let edit_num = slot.edit_rate.numerator;
        let edit_den = slot.edit_rate.denominator;

        let mut out = format!(
            "{pad}<MobSlot id=\"{id}\" slot_id=\"{}\" name=\"{name}\">\n",
            slot.slot_id
        );
        out.push_str(&format!(
            "{pad1}<EditRate numerator=\"{edit_num}\" denominator=\"{edit_den}\"/>\n"
        ));
        out.push_str(&format!("{pad1}<Origin>{}</Origin>\n", slot.origin.0));
        out.push_str(&format!("{pad}</MobSlot>\n"));
        out
    }

    /// Serialize a full `CompositionMob` as an `<AAFFile version="1.1">` document.
    ///
    /// The output is a self-contained XML string.
    #[must_use]
    pub fn serialize_composition(&self, comp: &CompositionMob) -> String {
        let pad0 = self.pad(0);
        let pad1 = self.pad(1);
        let pad2 = self.pad(2);
        let pad3 = self.pad(3);

        let name = Self::escape_xml(comp.name());
        let duration = comp.duration().unwrap_or(0);
        let (edit_num, edit_den) = comp
            .edit_rate()
            .map(|r| (r.numerator, r.denominator))
            .unwrap_or((25, 1));

        let mut out = String::new();
        out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        out.push_str(&format!("{pad0}<AAFFile version=\"1.1\">\n"));
        out.push_str(&format!(
            "{pad1}<Composition name=\"{name}\" duration=\"{duration}\" \
             edit_rate_num=\"{edit_num}\" edit_rate_den=\"{edit_den}\">\n"
        ));
        out.push_str(&format!("{pad2}<Tracks>\n"));

        for (idx, track) in comp.tracks().iter().enumerate() {
            // Render each track at level 3
            let _ = idx;
            out.push_str(&self.serialize_track(track, 3));
        }

        out.push_str(&format!("{pad2}</Tracks>\n"));
        out.push_str(&format!("{pad1}</Composition>\n"));
        out.push_str(&format!("{pad0}</AAFFile>\n"));

        // suppress unused-variable warning for pad3 if indent=0
        let _ = pad3;
        out
    }
}

// ─── XML Parser ──────────────────────────────────────────────────────────────

/// A single XML attribute parsed from a tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XmlAttribute {
    /// Attribute name.
    pub name: String,
    /// Attribute value (unescaped).
    pub value: String,
}

impl XmlAttribute {
    /// Create a new attribute.
    #[must_use]
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }
}

/// Parse `attr1="val1" attr2="val2"` string into a `Vec<XmlAttribute>`.
///
/// Supports both `"` and `'` delimiters.
#[must_use]
pub fn parse_element_attrs(s: &str) -> Vec<XmlAttribute> {
    let mut attrs = Vec::new();
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut i = 0usize;

    while i < len {
        // Skip whitespace
        while i < len && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= len {
            break;
        }

        // Read name (up to '=')
        let name_start = i;
        while i < len && bytes[i] != b'=' && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= len {
            break;
        }
        let name = &s[name_start..i];

        // Skip whitespace then '='
        while i < len && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= len || bytes[i] != b'=' {
            break;
        }
        i += 1; // consume '='

        // Skip whitespace
        while i < len && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= len {
            break;
        }

        // Read quoted value
        let quote = bytes[i];
        if quote != b'"' && quote != b'\'' {
            break;
        }
        i += 1; // consume opening quote
        let val_start = i;
        while i < len && bytes[i] != quote {
            i += 1;
        }
        let value = &s[val_start..i];
        if i < len {
            i += 1; // consume closing quote
        }

        attrs.push(XmlAttribute::new(name, unescape_xml(value)));
    }

    attrs
}

/// Unescape basic XML entities.
fn unescape_xml(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

/// Find the content between `<tag>` and `</tag>` in `xml`.
///
/// Returns `None` if either tag is absent.  Only the first occurrence is found.
#[must_use]
pub fn find_element_content<'a>(xml: &'a str, tag: &str) -> Option<&'a str> {
    // Look for opening tag (with possible attributes)
    let open_pattern = format!("<{tag}");
    let close_pattern = format!("</{tag}>");

    let open_pos = xml.find(&open_pattern)?;
    // Find end of opening tag (could be <tag> or <tag attr="...">)
    let tag_close = xml[open_pos..].find('>')?;
    let content_start = open_pos + tag_close + 1;

    let close_pos = xml[content_start..].find(&close_pattern)?;
    Some(&xml[content_start..content_start + close_pos])
}

/// Minimal XML parser that reconstructs `CompositionMob` instances.
///
/// Only attributes carried in `<Composition>` are parsed; tracks are listed
/// but not fully reconstituted (that would require essence data).
pub struct AafXmlParser;

impl AafXmlParser {
    /// Create a new parser.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Parse an XML string and return full `AafComposition` structs with nested
    /// tracks and slots.
    ///
    /// This method is the inverse of `AafXmlSerializer::serialize_to_xml`.
    pub fn parse_compositions(&self, xml: &str) -> Result<Vec<AafComposition>> {
        let mut compositions = Vec::new();
        let mut search_from = 0usize;

        while let Some(rel) = xml[search_from..].find("<Composition") {
            let comp_start = search_from + rel;

            let after_open = match xml[comp_start..].find('>') {
                Some(p) => comp_start + p + 1,
                None => {
                    return Err(AafError::ParseError(
                        "Unterminated <Composition> tag".to_string(),
                    ))
                }
            };

            let tag_body_start = comp_start + "<Composition".len();
            let tag_body_end = after_open - 1;
            let attr_str = if tag_body_end > tag_body_start {
                &xml[tag_body_start..tag_body_end]
            } else {
                ""
            };
            let attrs = parse_element_attrs(attr_str);

            let name = attrs
                .iter()
                .find(|a| a.name == "name")
                .map(|a| a.value.clone())
                .unwrap_or_default();
            let duration = attrs
                .iter()
                .find(|a| a.name == "duration")
                .and_then(|a| a.value.parse::<i64>().ok())
                .unwrap_or(0);
            let edit_rate_num = attrs
                .iter()
                .find(|a| a.name == "edit_rate_num")
                .and_then(|a| a.value.parse::<u32>().ok())
                .unwrap_or(25);
            let edit_rate_den = attrs
                .iter()
                .find(|a| a.name == "edit_rate_den")
                .and_then(|a| a.value.parse::<u32>().ok())
                .unwrap_or(1);

            let close_tag = "</Composition>";
            let close_pos = match xml[after_open..].find(close_tag) {
                Some(p) => after_open + p,
                None => {
                    return Err(AafError::ParseError(
                        "Missing </Composition> closing tag".to_string(),
                    ))
                }
            };

            let inner = &xml[after_open..close_pos];
            let tracks = self.parse_tracks_from_xml(inner)?;

            let mut comp = AafComposition::new(name, duration, edit_rate_num, edit_rate_den);
            comp.tracks = tracks;
            compositions.push(comp);

            search_from = close_pos + close_tag.len();
        }

        Ok(compositions)
    }

    /// Parse `<Track>` elements from an inner XML fragment.
    fn parse_tracks_from_xml(&self, inner: &str) -> Result<Vec<AafTrack>> {
        let mut tracks = Vec::new();
        let mut search_from = 0usize;

        while let Some(rel) = inner[search_from..].find("<Track ") {
            let track_start = search_from + rel;
            let after_open = match inner[track_start..].find('>') {
                Some(p) => track_start + p + 1,
                None => break,
            };

            let tag_body_start = track_start + "<Track ".len();
            let tag_body_end = after_open - 1;
            let attr_str = if tag_body_end > tag_body_start {
                &inner[tag_body_start..tag_body_end]
            } else {
                ""
            };
            let attrs = parse_element_attrs(attr_str);

            let track_id = attrs
                .iter()
                .find(|a| a.name == "id")
                .and_then(|a| a.value.parse::<u32>().ok())
                .unwrap_or(0);
            let track_name = attrs
                .iter()
                .find(|a| a.name == "name")
                .map(|a| a.value.clone())
                .unwrap_or_default();

            let close_tag = "</Track>";
            let close_pos = match inner[after_open..].find(close_tag) {
                Some(p) => after_open + p,
                None => break,
            };

            let track_inner = &inner[after_open..close_pos];
            let slots = self.parse_slots_from_xml(track_inner);

            let mut track = AafTrack::new(track_id, track_name);
            track.slots = slots;
            tracks.push(track);

            search_from = close_pos + close_tag.len();
        }

        Ok(tracks)
    }

    /// Parse `<Slot ... />` self-closing elements from a track's inner XML.
    fn parse_slots_from_xml(&self, inner: &str) -> Vec<AafSlot> {
        let mut slots = Vec::new();
        let mut search_from = 0usize;

        while let Some(rel) = inner[search_from..].find("<Slot ") {
            let slot_start = search_from + rel;

            // Find the end of the self-closing tag or normal closing
            let tag_end = match inner[slot_start..].find("/>") {
                Some(p) => slot_start + p + 2,
                None => match inner[slot_start..].find('>') {
                    Some(p) => slot_start + p + 1,
                    None => break,
                },
            };

            let tag_body_start = slot_start + "<Slot ".len();
            let tag_body_end = if inner[slot_start..tag_end].contains("/>") {
                tag_end - 2
            } else {
                tag_end - 1
            };
            let attr_str = if tag_body_end > tag_body_start {
                &inner[tag_body_start..tag_body_end]
            } else {
                ""
            };
            let attrs = parse_element_attrs(attr_str);

            let slot_id = attrs
                .iter()
                .find(|a| a.name == "id")
                .and_then(|a| a.value.parse::<u32>().ok())
                .unwrap_or(0);
            let slot_type = attrs
                .iter()
                .find(|a| a.name == "type")
                .map(|a| AafSlotType::from_str_loose(&a.value))
                .unwrap_or(AafSlotType::Data);
            let start_time = attrs
                .iter()
                .find(|a| a.name == "start_time")
                .and_then(|a| a.value.parse::<i64>().ok())
                .unwrap_or(0);
            let duration = attrs
                .iter()
                .find(|a| a.name == "duration")
                .and_then(|a| a.value.parse::<i64>().ok())
                .unwrap_or(0);
            let source_ref = attrs
                .iter()
                .find(|a| a.name == "source_ref")
                .map(|a| a.value.clone());

            let mut slot = AafSlot::new(slot_id, slot_type, start_time, duration);
            slot.source_ref = source_ref;
            slots.push(slot);

            search_from = tag_end;
        }

        slots
    }

    /// Parse an XML string and return summary structs for each `<Composition>`.
    ///
    /// Returns an error if the XML is structurally invalid.
    pub fn parse(&self, xml: &str) -> Result<Vec<ParsedComposition>> {
        let mut compositions = Vec::new();

        // Find all <Composition ...> blocks
        let mut search_from = 0usize;
        while let Some(rel) = xml[search_from..].find("<Composition") {
            let comp_start = search_from + rel;

            // Find the end of the opening tag
            let after_open = match xml[comp_start..].find('>') {
                Some(p) => comp_start + p + 1,
                None => {
                    return Err(AafError::ParseError(
                        "Unterminated <Composition> tag".to_string(),
                    ))
                }
            };

            // Extract attribute string
            let tag_body_start = comp_start + "<Composition".len();
            let tag_body_end = after_open - 1; // before '>'
            let attr_str = if tag_body_end > tag_body_start {
                &xml[tag_body_start..tag_body_end]
            } else {
                ""
            };
            let attrs = parse_element_attrs(attr_str);

            // Pull well-known attributes
            let name = attrs
                .iter()
                .find(|a| a.name == "name")
                .map(|a| a.value.clone())
                .unwrap_or_default();
            let duration = attrs
                .iter()
                .find(|a| a.name == "duration")
                .and_then(|a| a.value.parse::<i64>().ok())
                .unwrap_or(0);
            let edit_rate_num = attrs
                .iter()
                .find(|a| a.name == "edit_rate_num")
                .and_then(|a| a.value.parse::<u32>().ok())
                .unwrap_or(25);
            let edit_rate_den = attrs
                .iter()
                .find(|a| a.name == "edit_rate_den")
                .and_then(|a| a.value.parse::<u32>().ok())
                .unwrap_or(1);

            // Find closing tag
            let close_tag = "</Composition>";
            let close_pos = match xml[after_open..].find(close_tag) {
                Some(p) => after_open + p,
                None => {
                    return Err(AafError::ParseError(
                        "Missing </Composition> closing tag".to_string(),
                    ))
                }
            };

            // Count tracks inside
            let inner = &xml[after_open..close_pos];
            let track_count = count_occurrences(inner, "<Track ");

            compositions.push(ParsedComposition {
                name,
                duration,
                edit_rate_num,
                edit_rate_den,
                track_count,
            });

            search_from = close_pos + close_tag.len();
        }

        Ok(compositions)
    }
}

impl Default for AafXmlParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Lightweight representation of a parsed `<Composition>` element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedComposition {
    /// Composition name.
    pub name: String,
    /// Duration in edit units.
    pub duration: i64,
    /// Edit rate numerator.
    pub edit_rate_num: u32,
    /// Edit rate denominator.
    pub edit_rate_den: u32,
    /// Number of `<Track>` elements found inside.
    pub track_count: usize,
}

/// Count non-overlapping occurrences of `needle` in `haystack`.
fn count_occurrences(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    let mut count = 0usize;
    let mut pos = 0usize;
    while let Some(p) = haystack[pos..].find(needle) {
        count += 1;
        pos += p + needle.len();
    }
    count
}

// ─── MXF Bridge ──────────────────────────────────────────────────────────────

/// Bridge that extracts timecode information embedded in raw MXF KLV streams.
pub struct AafMxfBridge;

impl AafMxfBridge {
    /// Create a new bridge.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Scan raw MXF header data for a SMPTE 377 KLV essence mark.
    ///
    /// Looks for the 4-byte prefix `0x06 0x0E 0x2B 0x34` (SMPTE UL prefix)
    /// and returns the 16-byte Universal Label as a hex string if found.
    ///
    /// Returns `None` if no KLV marker is present in `data`.
    #[must_use]
    pub fn extract_timecode_from_mxf_header(&self, data: &[u8]) -> Option<String> {
        // SMPTE UL prefix — every SMPTE 377 KLV key starts with these 4 bytes
        const UL_PREFIX: [u8; 4] = [0x06, 0x0E, 0x2B, 0x34];

        // Minimum: 4-byte prefix + 12 remaining bytes for the 16-byte UL
        if data.len() < 16 {
            return None;
        }

        let mut i = 0usize;
        while i + 16 <= data.len() {
            if data[i..i + 4] == UL_PREFIX {
                // Found a UL — format the 16 bytes as hex pairs separated by dots
                let ul_bytes = &data[i..i + 16];
                let hex_str = ul_bytes
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<Vec<_>>()
                    .join(".");
                return Some(hex_str);
            }
            i += 1;
        }

        None
    }
}

impl Default for AafMxfBridge {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition::{
        CompositionMob, Sequence, SequenceComponent, SourceClip, Track, TrackType,
    };
    use crate::dictionary::Auid;
    use crate::timeline::{EditRate, Position};
    use uuid::Uuid;

    fn make_simple_composition() -> CompositionMob {
        let mob_id =
            Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").expect("valid UUID literal");
        let mut comp = CompositionMob::new(mob_id, "Test Composition");
        let mut track = Track::new(1, "V1", EditRate::PAL_25, TrackType::Picture);
        let mut seq = Sequence::new(Auid::PICTURE);
        let clip = SourceClip::new(100, Position::zero(), Uuid::new_v4(), 1);
        seq.add_component(SequenceComponent::SourceClip(clip));
        track.set_sequence(seq);
        comp.add_track(track);
        comp
    }

    // ── Serializer tests ────────────────────────────────────────────────────

    #[test]
    fn test_serialize_composition_contains_aaf_file_tag() {
        let ser = AafXmlSerializer::new();
        let comp = make_simple_composition();
        let xml = ser.serialize_composition(&comp);
        assert!(
            xml.contains("<AAFFile version=\"1.1\">"),
            "Missing AAFFile tag"
        );
    }

    #[test]
    fn test_serialize_composition_contains_composition_tag() {
        let ser = AafXmlSerializer::new();
        let comp = make_simple_composition();
        let xml = ser.serialize_composition(&comp);
        assert!(xml.contains("<Composition"), "Missing Composition tag");
        assert!(xml.contains("name=\"Test Composition\""));
    }

    #[test]
    fn test_serialize_composition_contains_tracks_tag() {
        let ser = AafXmlSerializer::new();
        let comp = make_simple_composition();
        let xml = ser.serialize_composition(&comp);
        assert!(xml.contains("<Tracks>"), "Missing <Tracks>");
        assert!(xml.contains("</Tracks>"), "Missing </Tracks>");
    }

    #[test]
    fn test_serialize_composition_contains_track_elements() {
        let ser = AafXmlSerializer::new();
        let comp = make_simple_composition();
        let xml = ser.serialize_composition(&comp);
        assert!(xml.contains("<Track "), "Missing <Track>");
        assert!(xml.contains("</Track>"), "Missing </Track>");
    }

    #[test]
    fn test_serialize_edit_rate_attribute() {
        let ser = AafXmlSerializer::new();
        let comp = make_simple_composition();
        let xml = ser.serialize_composition(&comp);
        assert!(
            xml.contains("edit_rate_num=\"25\""),
            "Missing edit_rate_num; xml=\n{xml}"
        );
        assert!(xml.contains("edit_rate_den=\"1\""));
    }

    #[test]
    fn test_serialize_duration_attribute() {
        let ser = AafXmlSerializer::new();
        let comp = make_simple_composition();
        let xml = ser.serialize_composition(&comp);
        assert!(xml.contains("duration=\"100\""), "xml=\n{xml}");
    }

    #[test]
    fn test_serialize_source_clips_section() {
        let ser = AafXmlSerializer::new();
        let comp = make_simple_composition();
        let xml = ser.serialize_composition(&comp);
        assert!(xml.contains("<SourceClips>"), "Missing SourceClips section");
        assert!(xml.contains("</SourceClips>"));
    }

    #[test]
    fn test_serialize_escape_special_chars() {
        let mob_id = Uuid::new_v4();
        let comp = CompositionMob::new(mob_id, "A&B <Show>");
        let ser = AafXmlSerializer::new();
        let xml = ser.serialize_composition(&comp);
        assert!(xml.contains("A&amp;B &lt;Show&gt;"));
    }

    #[test]
    fn test_serialize_closing_tags() {
        let ser = AafXmlSerializer::new();
        let comp = make_simple_composition();
        let xml = ser.serialize_composition(&comp);
        assert!(xml.contains("</Composition>"));
        assert!(xml.contains("</AAFFile>"));
    }

    #[test]
    fn test_serialize_xml_declaration() {
        let ser = AafXmlSerializer::new();
        let comp = make_simple_composition();
        let xml = ser.serialize_composition(&comp);
        assert!(xml.starts_with("<?xml version=\"1.0\""));
    }

    #[test]
    fn test_serialize_indentation_4_spaces() {
        let ser = AafXmlSerializer::with_indent(AafXmlSerializer::new(), 4);
        let comp = make_simple_composition();
        let xml = ser.serialize_composition(&comp);
        // At indent=4, first child of root should be indented by 4 spaces
        assert!(xml.contains("    <Composition"), "xml=\n{xml}");
    }

    // ── Attribute parser tests ───────────────────────────────────────────────

    #[test]
    fn test_parse_attrs_single() {
        let attrs = parse_element_attrs("name=\"hello\"");
        assert_eq!(attrs.len(), 1);
        assert_eq!(attrs[0].name, "name");
        assert_eq!(attrs[0].value, "hello");
    }

    #[test]
    fn test_parse_attrs_multiple() {
        let attrs = parse_element_attrs("a=\"1\" b=\"2\" c=\"3\"");
        assert_eq!(attrs.len(), 3);
        assert_eq!(attrs[0].name, "a");
        assert_eq!(attrs[1].value, "2");
    }

    #[test]
    fn test_parse_attrs_single_quotes() {
        let attrs = parse_element_attrs("name='world'");
        assert_eq!(attrs.len(), 1);
        assert_eq!(attrs[0].value, "world");
    }

    #[test]
    fn test_parse_attrs_entity_unescaping() {
        let attrs = parse_element_attrs("name=\"A&amp;B\"");
        assert_eq!(attrs[0].value, "A&B");
    }

    #[test]
    fn test_parse_attrs_empty() {
        let attrs = parse_element_attrs("");
        assert!(attrs.is_empty());
    }

    // ── find_element_content tests ───────────────────────────────────────────

    #[test]
    fn test_find_element_content_simple() {
        let xml = "<Root><Name>Foo</Name></Root>";
        let content = find_element_content(xml, "Name");
        assert_eq!(content, Some("Foo"));
    }

    #[test]
    fn test_find_element_content_nested() {
        let xml = "<A><B>inner</B></A>";
        assert_eq!(find_element_content(xml, "B"), Some("inner"));
        assert_eq!(find_element_content(xml, "A"), Some("<B>inner</B>"));
    }

    #[test]
    fn test_find_element_content_missing() {
        let xml = "<Root></Root>";
        assert!(find_element_content(xml, "Missing").is_none());
    }

    #[test]
    fn test_find_element_content_with_attr() {
        let xml = "<Track id=\"1\">data</Track>";
        assert_eq!(find_element_content(xml, "Track"), Some("data"));
    }

    // ── Parser tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_parser_empty_xml() {
        let parser = AafXmlParser::new();
        let result = parser.parse("<AAFFile version=\"1.1\"></AAFFile>");
        assert!(result.is_ok());
        assert!(result.expect("parse empty AAFFile").is_empty());
    }

    #[test]
    fn test_parser_single_composition() {
        let xml = r#"<?xml version="1.0"?>
<AAFFile version="1.1">
  <Composition name="Scene1" duration="250" edit_rate_num="25" edit_rate_den="1">
    <Tracks>
      <Track id="1" track_type="Picture">
        <Name>V1</Name>
      </Track>
    </Tracks>
  </Composition>
</AAFFile>"#;
        let parser = AafXmlParser::new();
        let comps = parser.parse(xml).expect("parse should succeed");
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].name, "Scene1");
        assert_eq!(comps[0].duration, 250);
        assert_eq!(comps[0].edit_rate_num, 25);
        assert_eq!(comps[0].track_count, 1);
    }

    #[test]
    fn test_parser_multiple_compositions() {
        let xml = r#"<AAFFile version="1.1">
  <Composition name="CompA" duration="100" edit_rate_num="24" edit_rate_den="1">
    <Tracks><Track id="1"><Name>V1</Name></Track></Tracks>
  </Composition>
  <Composition name="CompB" duration="200" edit_rate_num="25" edit_rate_den="1">
    <Tracks>
      <Track id="1"><Name>V1</Name></Track>
      <Track id="2"><Name>A1</Name></Track>
    </Tracks>
  </Composition>
</AAFFile>"#;
        let parser = AafXmlParser::new();
        let comps = parser.parse(xml).expect("parse ok");
        assert_eq!(comps.len(), 2);
        assert_eq!(comps[0].name, "CompA");
        assert_eq!(comps[1].name, "CompB");
        assert_eq!(comps[1].track_count, 2);
    }

    #[test]
    fn test_parser_error_unclosed_tag() {
        let xml = "<AAFFile><Composition name=\"X\" duration=\"0\" edit_rate_num=\"25\" edit_rate_den=\"1\">";
        let parser = AafXmlParser::new();
        let result = parser.parse(xml);
        assert!(result.is_err(), "Expected parse error for unclosed tag");
    }

    // ── MXF Bridge tests ─────────────────────────────────────────────────────

    #[test]
    fn test_mxf_bridge_finds_ul_prefix() {
        let bridge = AafMxfBridge::new();
        // Build a buffer with the SMPTE UL prefix at offset 4.
        // We need 4 (padding) + 16 (UL) = 20 bytes minimum for the scan to
        // find a complete 16-byte UL starting at offset 4.
        let mut data = vec![0u8; 4];
        data.extend_from_slice(&[0x06, 0x0E, 0x2B, 0x34]); // bytes 4-7
        data.extend_from_slice(&[
            0x01, 0x01, 0x01, 0x02, // bytes 8-11
            0x04, 0x01, 0x02, 0x01, // bytes 12-15
            0x01, 0x00, 0x00, 0x00,
        ]); // bytes 16-19
        let result = bridge.extract_timecode_from_mxf_header(&data);
        assert!(result.is_some());
        let hex = result.expect("should extract UL hex from MXF header");
        assert!(hex.starts_with("06.0e.2b.34"), "hex={hex}");
    }

    #[test]
    fn test_mxf_bridge_no_prefix() {
        let bridge = AafMxfBridge::new();
        let data = vec![0x00u8; 32];
        assert!(bridge.extract_timecode_from_mxf_header(&data).is_none());
    }

    #[test]
    fn test_mxf_bridge_empty_data() {
        let bridge = AafMxfBridge::new();
        assert!(bridge.extract_timecode_from_mxf_header(&[]).is_none());
    }

    #[test]
    fn test_mxf_bridge_too_short() {
        let bridge = AafMxfBridge::new();
        let data = [0x06, 0x0E, 0x2B, 0x34, 0x01, 0x02];
        // Only 6 bytes total — not enough for full 16-byte UL
        assert!(bridge.extract_timecode_from_mxf_header(&data).is_none());
    }

    #[test]
    fn test_mxf_bridge_hex_format() {
        let bridge = AafMxfBridge::new();
        let data = vec![
            0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01, 0x02, 0x04, 0x01, 0x02, 0x01, 0x01, 0x00,
            0x00, 0x00,
        ];
        let hex = bridge
            .extract_timecode_from_mxf_header(&data)
            .expect("should extract hex from 16-byte UL");
        // Should be 16 groups of 2-hex digits joined by dots
        let parts: Vec<&str> = hex.split('.').collect();
        assert_eq!(parts.len(), 16);
        for part in &parts {
            assert_eq!(part.len(), 2, "each hex group must be 2 chars");
        }
    }

    // ── Round-trip test ──────────────────────────────────────────────────────

    #[test]
    fn test_serialize_parse_roundtrip() {
        let comp = make_simple_composition();
        let ser = AafXmlSerializer::new();
        let xml = ser.serialize_composition(&comp);

        let parser = AafXmlParser::new();
        let parsed = parser.parse(&xml).expect("roundtrip parse");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "Test Composition");
        assert_eq!(parsed[0].duration, 100);
        assert_eq!(parsed[0].edit_rate_num, 25);
        assert_eq!(parsed[0].track_count, 1);
    }

    // ── Standalone AafComposition tests ────────────────────────────────────

    #[test]
    fn test_aaf_composition_creation() {
        let comp = AafComposition::new("My Comp", 500, 24, 1);
        assert_eq!(comp.name, "My Comp");
        assert_eq!(comp.duration, 500);
        assert_eq!(comp.edit_rate_num, 24);
        assert_eq!(comp.edit_rate_den, 1);
        assert!(comp.tracks.is_empty());
    }

    #[test]
    fn test_aaf_composition_edit_rate_f64() {
        let comp = AafComposition::new("T", 100, 30000, 1001);
        let rate = comp.edit_rate_f64();
        assert!((rate - 29.97).abs() < 0.01);
    }

    #[test]
    fn test_aaf_composition_edit_rate_zero_den() {
        let comp = AafComposition::new("T", 100, 25, 0);
        assert_eq!(comp.edit_rate_f64(), 0.0);
    }

    #[test]
    fn test_aaf_track_creation() {
        let track = AafTrack::new(1, "V1");
        assert_eq!(track.track_id, 1);
        assert_eq!(track.name, "V1");
        assert!(track.slots.is_empty());
    }

    #[test]
    fn test_aaf_track_total_duration() {
        let mut track = AafTrack::new(1, "V1");
        track.add_slot(AafSlot::new(1, AafSlotType::Video, 0, 100));
        track.add_slot(AafSlot::new(2, AafSlotType::Video, 100, 50));
        assert_eq!(track.total_duration(), 150);
    }

    #[test]
    fn test_aaf_slot_creation() {
        let slot = AafSlot::new(1, AafSlotType::Audio, 10, 200);
        assert_eq!(slot.slot_id, 1);
        assert_eq!(slot.start_time, 10);
        assert_eq!(slot.duration, 200);
        assert!(slot.source_ref.is_none());
    }

    #[test]
    fn test_aaf_slot_with_source_ref() {
        let slot = AafSlot::new(1, AafSlotType::Video, 0, 100).with_source_ref("mob-abc-123");
        assert_eq!(slot.source_ref.as_deref(), Some("mob-abc-123"));
    }

    #[test]
    fn test_aaf_slot_end_time() {
        let slot = AafSlot::new(1, AafSlotType::Video, 50, 100);
        assert_eq!(slot.end_time(), 150);
    }

    #[test]
    fn test_aaf_slot_type_roundtrip() {
        for st in &[
            AafSlotType::Video,
            AafSlotType::Audio,
            AafSlotType::Timecode,
            AafSlotType::Data,
        ] {
            let s = st.as_str();
            let parsed = AafSlotType::from_str_loose(s);
            assert_eq!(&parsed, st, "roundtrip failed for {s}");
        }
    }

    #[test]
    fn test_serialize_to_xml_standalone() {
        let mut comp = AafComposition::new("StandaloneComp", 250, 25, 1);
        let mut track = AafTrack::new(1, "V1");
        track.add_slot(AafSlot::new(1, AafSlotType::Video, 0, 250).with_source_ref("ref-001"));
        comp.add_track(track);

        let ser = AafXmlSerializer::new();
        let xml = ser.serialize_to_xml(&[comp]);

        assert!(xml.contains("<?xml version=\"1.0\""));
        assert!(xml.contains("<AAFFile version=\"1.1\">"));
        assert!(xml.contains("name=\"StandaloneComp\""));
        assert!(xml.contains("duration=\"250\""));
        assert!(xml.contains("edit_rate_num=\"25\""));
        assert!(xml.contains("<Track "));
        assert!(xml.contains("<Slot "));
        assert!(xml.contains("source_ref=\"ref-001\""));
        assert!(xml.contains("</AAFFile>"));
    }

    #[test]
    fn test_serialize_to_xml_multiple_compositions() {
        let c1 = AafComposition::new("Comp1", 100, 24, 1);
        let c2 = AafComposition::new("Comp2", 200, 25, 1);
        let ser = AafXmlSerializer::new();
        let xml = ser.serialize_to_xml(&[c1, c2]);
        assert!(xml.contains("name=\"Comp1\""));
        assert!(xml.contains("name=\"Comp2\""));
    }

    #[test]
    fn test_serialize_to_xml_empty() {
        let ser = AafXmlSerializer::new();
        let xml = ser.serialize_to_xml(&[]);
        assert!(xml.contains("<AAFFile"));
        assert!(xml.contains("</AAFFile>"));
    }

    #[test]
    fn test_serialize_to_xml_escapes_special_chars() {
        let comp = AafComposition::new("A&B <C>", 10, 25, 1);
        let ser = AafXmlSerializer::new();
        let xml = ser.serialize_to_xml(&[comp]);
        assert!(xml.contains("A&amp;B &lt;C&gt;"));
    }

    #[test]
    fn test_standalone_roundtrip_serialize_parse() {
        let mut comp = AafComposition::new("RoundTrip", 300, 30, 1);
        let mut t1 = AafTrack::new(1, "V1");
        t1.add_slot(AafSlot::new(1, AafSlotType::Video, 0, 150));
        t1.add_slot(AafSlot::new(2, AafSlotType::Video, 150, 150).with_source_ref("mob-xyz"));
        comp.add_track(t1);

        let mut t2 = AafTrack::new(2, "A1");
        t2.add_slot(AafSlot::new(3, AafSlotType::Audio, 0, 300));
        comp.add_track(t2);

        let ser = AafXmlSerializer::new();
        let xml = ser.serialize_to_xml(&[comp.clone()]);

        let parser = AafXmlParser::new();
        let parsed = parser.parse_compositions(&xml).expect("roundtrip parse");
        assert_eq!(parsed.len(), 1);
        let pc = &parsed[0];
        assert_eq!(pc.name, "RoundTrip");
        assert_eq!(pc.duration, 300);
        assert_eq!(pc.edit_rate_num, 30);
        assert_eq!(pc.edit_rate_den, 1);
        assert_eq!(pc.tracks.len(), 2);
        assert_eq!(pc.tracks[0].track_id, 1);
        assert_eq!(pc.tracks[0].name, "V1");
        assert_eq!(pc.tracks[0].slots.len(), 2);
        assert_eq!(pc.tracks[0].slots[0].slot_type, AafSlotType::Video);
        assert_eq!(pc.tracks[0].slots[1].source_ref.as_deref(), Some("mob-xyz"));
        assert_eq!(pc.tracks[1].slots[0].slot_type, AafSlotType::Audio);
    }

    #[test]
    fn test_standalone_roundtrip_multiple() {
        let c1 = AafComposition::new("First", 100, 24, 1);
        let mut c2 = AafComposition::new("Second", 200, 25, 1);
        let mut t = AafTrack::new(1, "V1");
        t.add_slot(AafSlot::new(1, AafSlotType::Video, 0, 200));
        c2.add_track(t);

        let ser = AafXmlSerializer::new();
        let xml = ser.serialize_to_xml(&[c1, c2]);

        let parser = AafXmlParser::new();
        let parsed = parser.parse_compositions(&xml).expect("parse");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].name, "First");
        assert!(parsed[0].tracks.is_empty());
        assert_eq!(parsed[1].name, "Second");
        assert_eq!(parsed[1].tracks.len(), 1);
    }

    #[test]
    fn test_parse_compositions_error_unclosed() {
        let xml = "<AAFFile><Composition name=\"X\" duration=\"0\">";
        let parser = AafXmlParser::new();
        assert!(parser.parse_compositions(xml).is_err());
    }

    #[test]
    fn test_aaf_track_slot_count() {
        let mut track = AafTrack::new(1, "V1");
        assert_eq!(track.slot_count(), 0);
        track.add_slot(AafSlot::new(1, AafSlotType::Video, 0, 100));
        assert_eq!(track.slot_count(), 1);
    }

    #[test]
    fn test_aaf_composition_track_count() {
        let mut comp = AafComposition::new("C", 100, 25, 1);
        assert_eq!(comp.track_count(), 0);
        comp.add_track(AafTrack::new(1, "V1"));
        comp.add_track(AafTrack::new(2, "A1"));
        assert_eq!(comp.track_count(), 2);
    }
}
