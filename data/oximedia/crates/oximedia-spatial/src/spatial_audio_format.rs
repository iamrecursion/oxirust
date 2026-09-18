//! ADM (Audio Definition Model) BWF metadata I/O.
//!
//! This module implements reading and writing of ADM metadata embedded in
//! Broadcast Wave Format (BWF) RIFF chunks, as specified in:
//!
//! - ITU-R BS.2076-2 — Audio Definition Model
//! - EBU Tech 3285 — Broadcast Wave Format specification
//!
//! # Overview
//!
//! ADM describes the spatial intent of audio content through a hierarchy:
//!
//! ```text
//! audioProgramme ──► audioContent ──► audioObject ──► audioPackFormat
//!                                                           │
//!                                                           └──► audioChannelFormat
//!                                                                       │
//!                                                                       └──► audioBlockFormat
//!                                                                                   │
//!                                                                                   └──► position (X, Y, Z)
//! ```
//!
//! # XML format
//!
//! ADM metadata is stored as UTF-8 XML inside a BWF `axml` chunk.  This module
//! provides a minimal pure-Rust parser (no external XML dependencies) that handles
//! the most common ADM elements.

use crate::SpatialError;

// ─── Types ────────────────────────────────────────────────────────────────────

/// Cartesian position in ADM convention.
///
/// All values are in the range [-1, 1].  The mapping is:
/// - X: lateral position (+1 = right, -1 = left)
/// - Y: depth position (+1 = front, -1 = back)
/// - Z: height position (+1 = top, -1 = bottom)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdmCartesian {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Polar position in ADM convention (degrees).
///
/// - `azimuth`: horizontal rotation (0 = front, +90 = left, -90 = right, ±180 = back)
/// - `elevation`: vertical angle (+90 = top, -90 = bottom)
/// - `distance`: 0..∞ (1.0 = unit sphere)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdmPolar {
    pub azimuth: f32,
    pub elevation: f32,
    pub distance: f32,
}

impl AdmPolar {
    /// Convert ADM polar coordinates to Cartesian.
    pub fn to_cartesian(&self) -> AdmCartesian {
        let az_rad = (-self.azimuth).to_radians(); // ADM: +left → physics CCW
        let el_rad = self.elevation.to_radians();
        let r = self.distance;
        let cos_el = el_rad.cos();
        AdmCartesian {
            x: r * cos_el * az_rad.sin(),
            y: r * cos_el * az_rad.cos(),
            z: r * el_rad.sin(),
        }
    }
}

/// Source position — either Cartesian or Polar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdmPosition {
    Cartesian(AdmCartesian),
    Polar(AdmPolar),
}

/// A time interval expressed in ADM format hh:mm:ss.SSSSS.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdmTime {
    pub hours: u32,
    pub minutes: u32,
    pub seconds: u32,
    /// Fractional seconds (stored as 5-digit integer, e.g. 12345 = 0.12345 s).
    pub frac: u32,
}

impl AdmTime {
    /// Zero time.
    pub fn zero() -> Self {
        Self {
            hours: 0,
            minutes: 0,
            seconds: 0,
            frac: 0,
        }
    }

    /// Total duration in seconds.
    pub fn total_seconds(&self) -> f64 {
        self.hours as f64 * 3600.0
            + self.minutes as f64 * 60.0
            + self.seconds as f64
            + self.frac as f64 / 100_000.0
    }

    /// Parse from ADM time string `hh:mm:ss.SSSSS`.
    pub fn parse(s: &str) -> Result<Self, SpatialError> {
        let err = || SpatialError::ParseError(format!("invalid ADM time: '{s}'"));
        let (hms, frac_str) = s.split_once('.').ok_or_else(err)?;
        let parts: Vec<&str> = hms.split(':').collect();
        if parts.len() != 3 {
            return Err(err());
        }
        let hours = parts[0].parse::<u32>().map_err(|_| err())?;
        let minutes = parts[1].parse::<u32>().map_err(|_| err())?;
        let seconds = parts[2].parse::<u32>().map_err(|_| err())?;
        // Pad or truncate fractional part to 5 digits.
        let frac_padded = format!("{:0<5}", &frac_str[..frac_str.len().min(5)]);
        let frac = frac_padded.parse::<u32>().map_err(|_| err())?;
        Ok(Self {
            hours,
            minutes,
            seconds,
            frac,
        })
    }

    /// Format as ADM time string.
    pub fn to_adm_string(&self) -> String {
        format!(
            "{:02}:{:02}:{:02}.{:05}",
            self.hours, self.minutes, self.seconds, self.frac
        )
    }
}

/// A single `audioBlockFormat` — describes the source position at a point in time.
#[derive(Debug, Clone)]
pub struct AudioBlockFormat {
    /// Block format ID (e.g. `"AB_00031001_00000001"`).
    pub id: String,
    /// Start time of this block within the audio object.
    pub rtime: AdmTime,
    /// Duration of this block.
    pub duration: AdmTime,
    /// Source position.
    pub position: AdmPosition,
    /// Gain (linear).
    pub gain: f32,
    /// Width spread in the horizontal plane (degrees, 0 = point source).
    pub width: f32,
    /// Height spread (degrees).
    pub height: f32,
    /// Depth spread.
    pub depth: f32,
}

/// An `audioChannelFormat` — describes the behaviour of a single audio channel.
#[derive(Debug, Clone)]
pub struct AudioChannelFormat {
    /// Channel format ID (e.g. `"AC_00031001"`).
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Type definition (e.g. `"Objects"`, `"DirectSpeakers"`, `"Binaural"`).
    pub type_definition: String,
    /// Time-ordered list of block formats.
    pub blocks: Vec<AudioBlockFormat>,
}

/// An `audioObject` — a named collection of audio content with timing.
#[derive(Debug, Clone)]
pub struct AdmAudioObject {
    /// Object ID (e.g. `"AO_1001"`).
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Start time in the programme timeline.
    pub start: AdmTime,
    /// Duration.
    pub duration: AdmTime,
    /// References to audio pack format IDs.
    pub audio_pack_format_id_refs: Vec<String>,
    /// References to audio track UID IDs.
    pub audio_track_uid_id_refs: Vec<String>,
}

/// An `audioProgramme` — the top-level container.
#[derive(Debug, Clone)]
pub struct AdmAudioProgramme {
    /// Programme ID.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Language code (e.g. `"en"`).
    pub language: String,
    /// Total duration.
    pub duration: Option<AdmTime>,
}

/// Complete parsed ADM document.
#[derive(Debug, Clone, Default)]
pub struct AdmDocument {
    /// Top-level programmes.
    pub programmes: Vec<AdmAudioProgramme>,
    /// Audio objects.
    pub objects: Vec<AdmAudioObject>,
    /// Channel formats (contain block formats with position data).
    pub channel_formats: Vec<AudioChannelFormat>,
    /// Raw XML for passthrough / round-trip fidelity.
    pub raw_xml: String,
}

// ─── XML attribute helpers ────────────────────────────────────────────────────

/// Extract the value of an XML attribute from an opening tag string.
///
/// Handles both double-quoted and single-quoted values.
fn xml_attr<'a>(tag: &'a str, key: &str) -> Option<&'a str> {
    let dq = format!("{}=\"", key);
    let sq = format!("{}='", key);
    if let Some(start) = tag.find(&dq) {
        let after = &tag[start + dq.len()..];
        let end = after.find('"')?;
        return Some(&after[..end]);
    }
    if let Some(start) = tag.find(&sq) {
        let after = &tag[start + sq.len()..];
        let end = after.find('\'')?;
        return Some(&after[..end]);
    }
    None
}

/// Extract the text content between the first occurrence of `<tag ...>` and `</tag>`.
fn xml_inner_text<'a>(xml: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);
    let start_tag = xml.find(&open)?;
    let gt = xml[start_tag..].find('>')?;
    let content_start = start_tag + gt + 1;
    let end = xml[content_start..].find(&close)?;
    Some(&xml[content_start..content_start + end])
}

/// Iterate over all occurrences of a start tag in `xml`, calling `f` with the full tag string.
fn for_each_tag<F: FnMut(&str)>(xml: &str, tag_name: &str, mut f: F) {
    let open_marker = format!("<{}", tag_name);
    let mut rest = xml;
    while let Some(start) = rest.find(&open_marker) {
        let tag_slice = &rest[start..];
        let end = tag_slice.find('>').unwrap_or(tag_slice.len() - 1);
        let tag_content = &tag_slice[..=end];
        f(tag_content);
        rest = &rest[start + 1..];
    }
}

// ─── Position parsing ─────────────────────────────────────────────────────────

/// Parse `<position coordinate="X">value</position>` blocks within a block format.
fn parse_positions(block_xml: &str) -> AdmPosition {
    let mut x: Option<f32> = None;
    let mut y: Option<f32> = None;
    let mut z: Option<f32> = None;
    let mut az: Option<f32> = None;
    let mut el: Option<f32> = None;
    let mut dist = 1.0_f32;

    for_each_tag(block_xml, "position", |tag| {
        // Find coordinate attribute.
        let coord = xml_attr(tag, "coordinate");
        // Find value — text between > and </position>.
        let gt = tag.find('>');
        if let (Some(coord), Some(_gt_idx)) = (coord, gt) {
            // Look for the value in the block_xml after this tag.
            // We scan for the pattern `coordinate="X">value</position>`.
            let marker = format!("coordinate=\"{}\"", coord);
            if let Some(start) = block_xml.find(&marker) {
                let after_marker = &block_xml[start..];
                if let Some(gt_pos) = after_marker.find('>') {
                    let value_start = start + gt_pos + 1;
                    let remaining = &block_xml[value_start..];
                    if let Some(end) = remaining.find("</position>") {
                        let val_str = remaining[..end].trim();
                        if let Ok(v) = val_str.parse::<f32>() {
                            match coord {
                                "X" => x = Some(v),
                                "Y" => y = Some(v),
                                "Z" => z = Some(v),
                                "azimuth" => az = Some(v),
                                "elevation" => el = Some(v),
                                "distance" => dist = v,
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
    });

    if let (Some(x), Some(y), Some(z)) = (x, y, z) {
        AdmPosition::Cartesian(AdmCartesian { x, y, z })
    } else if let Some(az) = az {
        AdmPosition::Polar(AdmPolar {
            azimuth: az,
            elevation: el.unwrap_or(0.0),
            distance: dist,
        })
    } else {
        // Default to front centre.
        AdmPosition::Cartesian(AdmCartesian {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        })
    }
}

// ─── Parser ───────────────────────────────────────────────────────────────────

/// Parse an ADM XML document into an [`AdmDocument`].
///
/// This is a minimal best-effort parser that does not validate against the full
/// ADM schema.  It extracts:
/// - `audioProgramme` elements (id, name, language)
/// - `audioObject` elements (id, name, start, duration)
/// - `audioChannelFormat` elements with their `audioBlockFormat` children
///   (including Cartesian and Polar position blocks)
///
/// Returns an error if the input is not valid UTF-8 or contains unparseable
/// required attributes.
pub fn parse_adm_xml(xml: &str) -> Result<AdmDocument, SpatialError> {
    let mut doc = AdmDocument::default();
    doc.raw_xml = xml.to_owned();

    // ── Parse audioProgramme ─────────────────────────────────────────────────
    for_each_tag(xml, "audioProgramme", |tag| {
        let id = xml_attr(tag, "audioProgrammeID").unwrap_or("").to_owned();
        let name = xml_attr(tag, "audioProgrammeName").unwrap_or("").to_owned();
        let language = xml_attr(tag, "audioLanguage").unwrap_or("").to_owned();
        doc.programmes.push(AdmAudioProgramme {
            id,
            name,
            language,
            duration: None,
        });
    });

    // ── Parse audioObject ─────────────────────────────────────────────────────
    for_each_tag(xml, "audioObject", |tag| {
        let id = xml_attr(tag, "audioObjectID").unwrap_or("").to_owned();
        let name = xml_attr(tag, "audioObjectName").unwrap_or("").to_owned();
        let start = xml_attr(tag, "start")
            .and_then(|s| AdmTime::parse(s).ok())
            .unwrap_or_else(AdmTime::zero);
        let duration = xml_attr(tag, "duration")
            .and_then(|s| AdmTime::parse(s).ok())
            .unwrap_or_else(AdmTime::zero);
        doc.objects.push(AdmAudioObject {
            id,
            name,
            start,
            duration,
            audio_pack_format_id_refs: Vec::new(),
            audio_track_uid_id_refs: Vec::new(),
        });
    });

    // ── Parse audioChannelFormat + audioBlockFormat ───────────────────────────
    // We look for <audioChannelFormat ...>...</audioChannelFormat> blocks.
    let channel_open = "<audioChannelFormat";
    let channel_close = "</audioChannelFormat>";
    let mut search = xml;
    while let Some(start) = search.find(channel_open) {
        let ch_xml = &search[start..];
        let block_end = match ch_xml.find(channel_close) {
            Some(e) => e + channel_close.len(),
            None => break,
        };
        let ch_content = &ch_xml[..block_end];

        // Extract channel attributes from the opening tag.
        let open_end = ch_content.find('>').unwrap_or(0);
        let ch_tag = &ch_content[..=open_end];

        let ch_id = xml_attr(ch_tag, "audioChannelFormatID")
            .unwrap_or("")
            .to_owned();
        let ch_name = xml_attr(ch_tag, "audioChannelFormatName")
            .unwrap_or("")
            .to_owned();
        let type_def = xml_attr(ch_tag, "typeDefinition")
            .unwrap_or("Objects")
            .to_owned();

        let mut blocks = Vec::new();

        // Parse audioBlockFormat children.
        let block_open = "<audioBlockFormat";
        let block_close = "</audioBlockFormat>";
        let mut block_search = ch_content;
        while let Some(bs) = block_search.find(block_open) {
            let blk_xml = &block_search[bs..];
            let blk_end = match blk_xml.find(block_close) {
                Some(e) => e + block_close.len(),
                None => break,
            };
            let blk_content = &blk_xml[..blk_end];

            // Opening tag for block attributes.
            let blk_open_end = blk_content.find('>').unwrap_or(0);
            let blk_tag = &blk_content[..=blk_open_end];

            let blk_id = xml_attr(blk_tag, "audioBlockFormatID")
                .unwrap_or("")
                .to_owned();
            let rtime = xml_attr(blk_tag, "rtime")
                .and_then(|s| AdmTime::parse(s).ok())
                .unwrap_or_else(AdmTime::zero);
            let dur = xml_attr(blk_tag, "duration")
                .and_then(|s| AdmTime::parse(s).ok())
                .unwrap_or_else(AdmTime::zero);

            let gain = xml_inner_text(blk_content, "gain")
                .and_then(|s| s.trim().parse::<f32>().ok())
                .unwrap_or(1.0);
            let width = xml_inner_text(blk_content, "width")
                .and_then(|s| s.trim().parse::<f32>().ok())
                .unwrap_or(0.0);
            let height = xml_inner_text(blk_content, "height")
                .and_then(|s| s.trim().parse::<f32>().ok())
                .unwrap_or(0.0);
            let depth = xml_inner_text(blk_content, "depth")
                .and_then(|s| s.trim().parse::<f32>().ok())
                .unwrap_or(0.0);

            let position = parse_positions(blk_content);

            blocks.push(AudioBlockFormat {
                id: blk_id,
                rtime,
                duration: dur,
                position,
                gain,
                width,
                height,
                depth,
            });

            block_search = &block_search[bs + 1..];
        }

        doc.channel_formats.push(AudioChannelFormat {
            id: ch_id,
            name: ch_name,
            type_definition: type_def,
            blocks,
        });

        search = &search[start + 1..];
    }

    Ok(doc)
}

// ─── Serialiser ───────────────────────────────────────────────────────────────

/// Serialise an [`AdmDocument`] to a minimal ADM XML string.
///
/// The output is a valid ADM XML fragment that can be embedded in a BWF `axml`
/// chunk.  All elements from the document are serialised; the `raw_xml` field
/// is ignored.
pub fn write_adm_xml(doc: &AdmDocument) -> String {
    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str("<ebuCoreMain xmlns:adm=\"urn:ebu:metadata-schema:ebuCore_2015\">\n");
    out.push_str("  <coreMetadata>\n");
    out.push_str("    <format>\n");
    out.push_str("      <audioFormatExtended version=\"ITU-R_BS.2076-2\">\n");

    // audioProgramme
    for prog in &doc.programmes {
        out.push_str(&format!(
            "        <audioProgramme audioProgrammeID=\"{}\" audioProgrammeName=\"{}\" audioLanguage=\"{}\"/>\n",
            prog.id, prog.name, prog.language
        ));
    }

    // audioObject
    for obj in &doc.objects {
        out.push_str(&format!(
            "        <audioObject audioObjectID=\"{}\" audioObjectName=\"{}\" start=\"{}\" duration=\"{}\"/>\n",
            obj.id, obj.name,
            obj.start.to_adm_string(),
            obj.duration.to_adm_string()
        ));
    }

    // audioChannelFormat
    for ch in &doc.channel_formats {
        out.push_str(&format!(
            "        <audioChannelFormat audioChannelFormatID=\"{}\" audioChannelFormatName=\"{}\" typeDefinition=\"{}\">\n",
            ch.id, ch.name, ch.type_definition
        ));
        for blk in &ch.blocks {
            out.push_str(&format!(
                "          <audioBlockFormat audioBlockFormatID=\"{}\" rtime=\"{}\" duration=\"{}\">\n",
                blk.id,
                blk.rtime.to_adm_string(),
                blk.duration.to_adm_string()
            ));
            match &blk.position {
                AdmPosition::Cartesian(c) => {
                    out.push_str(&format!(
                        "            <position coordinate=\"X\">{}</position>\n",
                        c.x
                    ));
                    out.push_str(&format!(
                        "            <position coordinate=\"Y\">{}</position>\n",
                        c.y
                    ));
                    out.push_str(&format!(
                        "            <position coordinate=\"Z\">{}</position>\n",
                        c.z
                    ));
                }
                AdmPosition::Polar(p) => {
                    out.push_str(&format!(
                        "            <position coordinate=\"azimuth\">{}</position>\n",
                        p.azimuth
                    ));
                    out.push_str(&format!(
                        "            <position coordinate=\"elevation\">{}</position>\n",
                        p.elevation
                    ));
                    out.push_str(&format!(
                        "            <position coordinate=\"distance\">{}</position>\n",
                        p.distance
                    ));
                }
            }
            if (blk.gain - 1.0).abs() > 1e-6 {
                out.push_str(&format!("            <gain>{}</gain>\n", blk.gain));
            }
            out.push_str("          </audioBlockFormat>\n");
        }
        out.push_str("        </audioChannelFormat>\n");
    }

    out.push_str("      </audioFormatExtended>\n");
    out.push_str("    </format>\n");
    out.push_str("  </coreMetadata>\n");
    out.push_str("</ebuCoreMain>\n");
    out
}

/// Write a minimal ADM BWF `axml` chunk payload for a given document.
///
/// The BWF `axml` chunk is a raw UTF-8 XML string prepended with a 4-byte
/// RIFF chunk ID `"axml"` and a 4-byte little-endian size.  Returns the
/// complete chunk bytes.
pub fn write_axml_chunk(doc: &AdmDocument) -> Vec<u8> {
    let xml = write_adm_xml(doc);
    let xml_bytes = xml.as_bytes();
    let size = xml_bytes.len() as u32;

    let mut chunk = Vec::with_capacity(8 + xml_bytes.len());
    chunk.extend_from_slice(b"axml");
    chunk.extend_from_slice(&size.to_le_bytes());
    chunk.extend_from_slice(xml_bytes);

    // Pad to even byte boundary (RIFF requirement).
    if xml_bytes.len() % 2 != 0 {
        chunk.push(0x00);
    }

    chunk
}

/// Parse an `axml` chunk payload (starting after the 8-byte chunk header).
///
/// Strips any trailing null padding and decodes as UTF-8 before passing to
/// [`parse_adm_xml`].
pub fn parse_axml_chunk(chunk_data: &[u8]) -> Result<AdmDocument, SpatialError> {
    // Find the end of UTF-8 content (strip null padding).
    let trimmed = chunk_data
        .iter()
        .rposition(|&b| b != 0x00)
        .map(|pos| &chunk_data[..=pos])
        .unwrap_or(&[]);

    let xml = std::str::from_utf8(trimmed)
        .map_err(|e| SpatialError::ParseError(format!("axml chunk is not valid UTF-8: {e}")))?;

    parse_adm_xml(xml)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_xml() -> &'static str {
        r#"<?xml version="1.0"?>
<ebuCoreMain>
  <coreMetadata>
    <format>
      <audioFormatExtended version="ITU-R_BS.2076-2">
        <audioProgramme audioProgrammeID="APR_1001" audioProgrammeName="Main Mix" audioLanguage="en"/>
        <audioObject audioObjectID="AO_1001" audioObjectName="Violin"
                     start="00:00:00.00000" duration="00:00:05.00000"/>
        <audioChannelFormat audioChannelFormatID="AC_00031001"
                            audioChannelFormatName="Objects_1"
                            typeDefinition="Objects">
          <audioBlockFormat audioBlockFormatID="AB_00031001_00000001"
                            rtime="00:00:00.00000" duration="00:00:05.00000">
            <position coordinate="X">0.5</position>
            <position coordinate="Y">1.0</position>
            <position coordinate="Z">0.0</position>
            <gain>0.8</gain>
          </audioBlockFormat>
        </audioChannelFormat>
      </audioFormatExtended>
    </format>
  </coreMetadata>
</ebuCoreMain>"#
    }

    // ── AdmTime ──────────────────────────────────────────────────────────────

    #[test]
    fn test_adm_time_parse_basic() {
        let t = AdmTime::parse("00:01:30.50000").expect("parse");
        assert_eq!(t.hours, 0);
        assert_eq!(t.minutes, 1);
        assert_eq!(t.seconds, 30);
        assert_eq!(t.frac, 50000);
    }

    #[test]
    fn test_adm_time_total_seconds() {
        let t = AdmTime::parse("00:01:00.00000").expect("parse");
        assert!((t.total_seconds() - 60.0).abs() < 0.001);
    }

    #[test]
    fn test_adm_time_to_string_round_trip() {
        let original = "01:23:45.67890";
        let t = AdmTime::parse(original).expect("parse");
        assert_eq!(t.to_adm_string(), original);
    }

    #[test]
    fn test_adm_time_parse_invalid_fails() {
        assert!(AdmTime::parse("invalid").is_err());
        assert!(AdmTime::parse("00:00:abc.00000").is_err());
    }

    #[test]
    fn test_adm_time_zero() {
        let t = AdmTime::zero();
        assert_eq!(t.total_seconds(), 0.0);
    }

    // ── AdmPolar::to_cartesian ───────────────────────────────────────────────

    #[test]
    fn test_polar_to_cartesian_front() {
        let p = AdmPolar {
            azimuth: 0.0,
            elevation: 0.0,
            distance: 1.0,
        };
        let c = p.to_cartesian();
        assert!((c.x).abs() < 1e-4, "front x should be ~0");
        assert!((c.y - 1.0).abs() < 1e-4, "front y should be ~1");
    }

    #[test]
    fn test_polar_to_cartesian_left() {
        // ADM convention: azimuth +90 = left, X: +1 = right, -1 = left.
        // So a source at azimuth=+90 (left) should have x < -0.5.
        let p = AdmPolar {
            azimuth: 90.0,
            elevation: 0.0,
            distance: 1.0,
        };
        let c = p.to_cartesian();
        assert!(
            c.x < -0.5,
            "left source (azimuth=+90) should have negative x in ADM convention, got x={}",
            c.x
        );
    }

    #[test]
    fn test_polar_to_cartesian_top() {
        let p = AdmPolar {
            azimuth: 0.0,
            elevation: 90.0,
            distance: 1.0,
        };
        let c = p.to_cartesian();
        assert!((c.z - 1.0).abs() < 1e-4, "top source should have z=1");
    }

    // ── parse_adm_xml ─────────────────────────────────────────────────────────

    #[test]
    fn test_parse_programme_id_and_name() {
        let doc = parse_adm_xml(sample_xml()).expect("parse ok");
        assert!(!doc.programmes.is_empty(), "Should parse programme");
        assert_eq!(doc.programmes[0].id, "APR_1001");
        assert_eq!(doc.programmes[0].name, "Main Mix");
        assert_eq!(doc.programmes[0].language, "en");
    }

    #[test]
    fn test_parse_audio_object() {
        let doc = parse_adm_xml(sample_xml()).expect("parse ok");
        assert!(!doc.objects.is_empty());
        assert_eq!(doc.objects[0].id, "AO_1001");
        assert_eq!(doc.objects[0].name, "Violin");
        assert!((doc.objects[0].duration.total_seconds() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_channel_format() {
        let doc = parse_adm_xml(sample_xml()).expect("parse ok");
        assert!(!doc.channel_formats.is_empty());
        let ch = &doc.channel_formats[0];
        assert_eq!(ch.id, "AC_00031001");
        assert_eq!(ch.type_definition, "Objects");
    }

    #[test]
    fn test_parse_block_format_position() {
        let doc = parse_adm_xml(sample_xml()).expect("parse ok");
        let ch = &doc.channel_formats[0];
        assert!(!ch.blocks.is_empty());
        let blk = &ch.blocks[0];
        if let AdmPosition::Cartesian(c) = blk.position {
            assert!((c.x - 0.5).abs() < 0.001, "x should be 0.5");
            assert!((c.y - 1.0).abs() < 0.001, "y should be 1.0");
            assert!((c.z - 0.0).abs() < 0.001, "z should be 0.0");
        } else {
            panic!("Expected Cartesian position");
        }
    }

    #[test]
    fn test_parse_block_format_gain() {
        let doc = parse_adm_xml(sample_xml()).expect("parse ok");
        let blk = &doc.channel_formats[0].blocks[0];
        assert!((blk.gain - 0.8).abs() < 0.001, "gain should be 0.8");
    }

    #[test]
    fn test_parse_raw_xml_preserved() {
        let xml = sample_xml();
        let doc = parse_adm_xml(xml).expect("parse ok");
        assert!(!doc.raw_xml.is_empty(), "raw_xml should be stored");
    }

    #[test]
    fn test_parse_empty_xml() {
        let doc = parse_adm_xml("").expect("empty XML should not error");
        assert!(doc.programmes.is_empty());
        assert!(doc.objects.is_empty());
    }

    // ── write_adm_xml ─────────────────────────────────────────────────────────

    #[test]
    fn test_write_adm_xml_contains_programme() {
        let mut doc = AdmDocument::default();
        doc.programmes.push(AdmAudioProgramme {
            id: "APR_0001".into(),
            name: "Test".into(),
            language: "fr".into(),
            duration: None,
        });
        let xml = write_adm_xml(&doc);
        assert!(xml.contains("APR_0001"), "XML should contain programme ID");
        assert!(xml.contains("Test"), "XML should contain programme name");
    }

    #[test]
    fn test_write_adm_xml_contains_cartesian_position() {
        let mut doc = AdmDocument::default();
        doc.channel_formats.push(AudioChannelFormat {
            id: "AC_0001".into(),
            name: "Test".into(),
            type_definition: "Objects".into(),
            blocks: vec![AudioBlockFormat {
                id: "AB_0001".into(),
                rtime: AdmTime::zero(),
                duration: AdmTime::zero(),
                position: AdmPosition::Cartesian(AdmCartesian {
                    x: 0.5,
                    y: 0.5,
                    z: 0.0,
                }),
                gain: 1.0,
                width: 0.0,
                height: 0.0,
                depth: 0.0,
            }],
        });
        let xml = write_adm_xml(&doc);
        assert!(xml.contains("coordinate=\"X\""), "Should contain X coord");
        assert!(xml.contains("0.5"), "Should contain X value");
    }

    // ── axml chunk round-trip ─────────────────────────────────────────────────

    #[test]
    fn test_axml_chunk_round_trip() {
        let mut doc = AdmDocument::default();
        doc.programmes.push(AdmAudioProgramme {
            id: "APR_9999".into(),
            name: "Round Trip".into(),
            language: "de".into(),
            duration: None,
        });

        let chunk = write_axml_chunk(&doc);
        assert_eq!(&chunk[0..4], b"axml", "Chunk should start with 'axml'");

        // Parse back (skip 8-byte header).
        let doc2 = parse_axml_chunk(&chunk[8..]).expect("round-trip parse ok");
        assert!(!doc2.programmes.is_empty());
        assert_eq!(doc2.programmes[0].id, "APR_9999");
    }

    #[test]
    fn test_axml_chunk_size_field() {
        let doc = AdmDocument::default();
        let chunk = write_axml_chunk(&doc);
        let size = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        // Data section size should match chunk length - 8 header bytes (minus possible pad byte).
        assert!(chunk.len() >= 8 + size as usize);
    }

    #[test]
    fn test_parse_polar_block_format() {
        let xml = r#"<audioChannelFormat audioChannelFormatID="AC_0002"
                                          audioChannelFormatName="Drum"
                                          typeDefinition="Objects">
          <audioBlockFormat audioBlockFormatID="AB_0001" rtime="00:00:00.00000" duration="00:00:01.00000">
            <position coordinate="azimuth">30.0</position>
            <position coordinate="elevation">0.0</position>
            <position coordinate="distance">1.0</position>
          </audioBlockFormat>
        </audioChannelFormat>"#;
        let doc = parse_adm_xml(xml).expect("parse polar");
        let blk = &doc.channel_formats[0].blocks[0];
        if let AdmPosition::Polar(p) = blk.position {
            assert!((p.azimuth - 30.0).abs() < 0.1, "azimuth should be 30°");
        } else {
            panic!("Expected Polar position");
        }
    }

    #[test]
    fn test_xml_attr_double_and_single_quotes() {
        assert_eq!(xml_attr(r#"key="val""#, "key"), Some("val"));
        assert_eq!(xml_attr("key='val'", "key"), Some("val"));
        assert_eq!(xml_attr("other=\"x\"", "key"), None);
    }
}
