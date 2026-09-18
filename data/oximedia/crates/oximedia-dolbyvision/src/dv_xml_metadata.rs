//! Dolby Vision XML metadata format: serialization, parsing, and validation.
//!
//! Implements the DolbyVision_RPU.xml sidecar format used in post-production
//! workflows. Parsing is done with a minimal hand-rolled XML reader to avoid
//! external XML crate dependencies.

#![allow(dead_code)]

use crate::DolbyVisionError;

/// Top-level Dolby Vision XML metadata document.
#[derive(Debug, Clone)]
pub struct DvXmlMetadata {
    /// XML format version string (e.g., "CM v4.0").
    pub version: String,
    /// Per-shot metadata entries.
    pub shots: Vec<DvXmlShot>,
    /// Display targets referenced by the metadata.
    pub display_targets: Vec<DisplayTarget>,
}

impl DvXmlMetadata {
    /// Create a new empty metadata document.
    #[must_use]
    pub fn new(version: impl Into<String>) -> Self {
        Self {
            version: version.into(),
            shots: Vec::new(),
            display_targets: Vec::new(),
        }
    }
}

/// A single shot entry in the XML metadata.
#[derive(Debug, Clone)]
pub struct DvXmlShot {
    /// Unique shot identifier (UUID or monotonic string).
    pub unique_id: String,
    /// Byte offset of the shot in the source bitstream.
    pub file_offset: u64,
    /// Duration in frames.
    pub duration: u64,
    /// Level 1 (frame-level luminance) metadata.
    pub level1: Option<Level1>,
    /// Level 2 (trim) metadata.
    pub level2: Option<Level2>,
    /// Level 6 (static HDR / MaxCLL) metadata.
    pub level6: Option<Level6>,
}

/// Level 1 metadata: frame-level min/avg/max PQ in code values [0, 4095].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Level1 {
    /// Minimum PQ code value.
    pub min_pq: u32,
    /// Maximum PQ code value.
    pub max_pq: u32,
    /// Average PQ code value.
    pub avg_pq: u32,
}

/// Level 2 metadata: per-display trim parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct Level2 {
    /// Target display peak PQ code value.
    pub target_max_pq: u32,
    /// Trim slope (0.0–2.0).
    pub trim_slope: f32,
    /// Trim offset (-1.0–1.0).
    pub trim_offset: f32,
    /// Trim power (0.0–2.0).
    pub trim_power: f32,
    /// Trim saturation gain (0.0–2.0).
    pub trim_saturation_gain: f32,
}

/// Level 6 metadata: static MaxCLL and MaxFALL values in nits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Level6 {
    /// Maximum Content Light Level (nits).
    pub max_cll: u32,
    /// Maximum Frame-Average Light Level (nits).
    pub max_fall: u32,
}

/// A display target referenced in the XML metadata.
#[derive(Debug, Clone)]
pub struct DisplayTarget {
    /// Peak luminance of the display in nits.
    pub nits: u32,
    /// Color primaries label (e.g., "BT.2020", "P3-D65").
    pub primaries: String,
    /// Peak luminance type (e.g., "absolute", "relative").
    pub peak_luminance_type: String,
}

// ── Serialization ─────────────────────────────────────────────────────────────

/// Serialize a `DvXmlMetadata` document to an XML string.
///
/// Uses manual string building to avoid external XML crate dependencies.
#[must_use]
pub fn serialize_dv_xml(metadata: &DvXmlMetadata) -> String {
    let mut out = String::with_capacity(4096);

    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str("<DolbyLabsMDF xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"");
    out.push_str(" version=\"");
    out.push_str(&xml_escape(&metadata.version));
    out.push_str("\">\n");

    // Display targets
    if !metadata.display_targets.is_empty() {
        out.push_str("  <Outputs>\n");
        for target in &metadata.display_targets {
            out.push_str("    <Output>\n");
            write_xml_tag(&mut out, "Nits", &target.nits.to_string(), 6);
            write_xml_tag(&mut out, "Primaries", &target.primaries, 6);
            write_xml_tag(
                &mut out,
                "PeakLuminanceType",
                &target.peak_luminance_type,
                6,
            );
            out.push_str("    </Output>\n");
        }
        out.push_str("  </Outputs>\n");
    }

    // Shots
    if !metadata.shots.is_empty() {
        out.push_str("  <Shots>\n");
        for shot in &metadata.shots {
            out.push_str("    <Shot>\n");
            write_xml_tag(&mut out, "UniqueID", &shot.unique_id, 6);
            write_xml_tag(&mut out, "FileOffset", &shot.file_offset.to_string(), 6);
            write_xml_tag(&mut out, "Duration", &shot.duration.to_string(), 6);

            if let Some(l1) = &shot.level1 {
                out.push_str("      <Level1>\n");
                write_xml_tag(&mut out, "MinPQ", &l1.min_pq.to_string(), 8);
                write_xml_tag(&mut out, "MaxPQ", &l1.max_pq.to_string(), 8);
                write_xml_tag(&mut out, "AvgPQ", &l1.avg_pq.to_string(), 8);
                out.push_str("      </Level1>\n");
            }

            if let Some(l2) = &shot.level2 {
                out.push_str("      <Level2>\n");
                write_xml_tag(&mut out, "TargetMaxPQ", &l2.target_max_pq.to_string(), 8);
                write_xml_tag(&mut out, "TrimSlope", &format!("{:.6}", l2.trim_slope), 8);
                write_xml_tag(&mut out, "TrimOffset", &format!("{:.6}", l2.trim_offset), 8);
                write_xml_tag(&mut out, "TrimPower", &format!("{:.6}", l2.trim_power), 8);
                write_xml_tag(
                    &mut out,
                    "TrimSaturationGain",
                    &format!("{:.6}", l2.trim_saturation_gain),
                    8,
                );
                out.push_str("      </Level2>\n");
            }

            if let Some(l6) = &shot.level6 {
                out.push_str("      <Level6>\n");
                write_xml_tag(&mut out, "MaxCLL", &l6.max_cll.to_string(), 8);
                write_xml_tag(&mut out, "MaxFALL", &l6.max_fall.to_string(), 8);
                out.push_str("      </Level6>\n");
            }

            out.push_str("    </Shot>\n");
        }
        out.push_str("  </Shots>\n");
    }

    out.push_str("</DolbyLabsMDF>\n");
    out
}

fn write_xml_tag(out: &mut String, tag: &str, value: &str, indent: usize) {
    let spaces: String = " ".repeat(indent);
    out.push_str(&spaces);
    out.push('<');
    out.push_str(tag);
    out.push('>');
    out.push_str(&xml_escape(value));
    out.push_str("</");
    out.push_str(tag);
    out.push_str(">\n");
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

// ── Minimal XML Parsing ───────────────────────────────────────────────────────

/// Parse a Dolby Vision XML metadata string into `DvXmlMetadata`.
///
/// Uses a minimal hand-rolled parser: finds tags by name and extracts
/// their text content. Does not handle namespaces or deeply nested structures
/// beyond the known schema.
///
/// # Errors
///
/// Returns an error if required elements are missing or values cannot be parsed.
pub fn parse_dv_xml(xml: &str) -> Result<DvXmlMetadata, DolbyVisionError> {
    let version =
        extract_attr(xml, "DolbyLabsMDF", "version").unwrap_or_else(|| "unknown".to_string());

    let display_targets = parse_display_targets(xml)?;
    let shots = parse_shots(xml)?;

    Ok(DvXmlMetadata {
        version,
        shots,
        display_targets,
    })
}

/// Extract a named attribute from the first occurrence of a given XML tag.
fn extract_attr(xml: &str, tag: &str, attr: &str) -> Option<String> {
    let tag_start = format!("<{}", tag);
    let pos = xml.find(&tag_start)?;
    let remainder = &xml[pos..];

    // Find end of opening tag
    let tag_end = remainder.find('>')?;
    let tag_content = &remainder[..tag_end];

    let attr_search = format!("{}=\"", attr);
    let attr_pos = tag_content.find(&attr_search)?;
    let value_start = attr_pos + attr_search.len();
    let value_end = tag_content[value_start..].find('"')?;
    Some(xml_unescape(
        &tag_content[value_start..value_start + value_end],
    ))
}

/// Extract the text content of the first matching tag inside `parent_xml`.
fn extract_tag_text(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{}>", tag);
    let close = format!("</{}>", tag);
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&close)?;
    Some(xml_unescape(xml[start..start + end].trim()))
}

fn xml_unescape(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

fn parse_display_targets(xml: &str) -> Result<Vec<DisplayTarget>, DolbyVisionError> {
    let mut targets = Vec::new();
    let mut search_start = 0;

    while let Some(start_rel) = xml[search_start..].find("<Output>") {
        let start_abs = search_start + start_rel;
        let end_rel = xml[start_abs..]
            .find("</Output>")
            .ok_or_else(|| DolbyVisionError::Generic("Unclosed <Output> tag".to_string()))?;
        let block = &xml[start_abs..start_abs + end_rel + "</Output>".len()];

        let nits_str = extract_tag_text(block, "Nits").unwrap_or_else(|| "0".to_string());
        let nits = nits_str.parse::<u32>().map_err(|e| {
            DolbyVisionError::Generic(format!("Invalid Nits value '{}': {}", nits_str, e))
        })?;

        let primaries =
            extract_tag_text(block, "Primaries").unwrap_or_else(|| "BT.2020".to_string());
        let peak_luminance_type =
            extract_tag_text(block, "PeakLuminanceType").unwrap_or_else(|| "absolute".to_string());

        targets.push(DisplayTarget {
            nits,
            primaries,
            peak_luminance_type,
        });
        search_start = start_abs + end_rel + "</Output>".len();
    }

    Ok(targets)
}

fn parse_shots(xml: &str) -> Result<Vec<DvXmlShot>, DolbyVisionError> {
    let mut shots = Vec::new();
    let mut search_start = 0;

    while let Some(start_rel) = xml[search_start..].find("<Shot>") {
        let start_abs = search_start + start_rel;
        let end_rel = xml[start_abs..]
            .find("</Shot>")
            .ok_or_else(|| DolbyVisionError::Generic("Unclosed <Shot> tag".to_string()))?;
        let block = &xml[start_abs..start_abs + end_rel + "</Shot>".len()];

        let unique_id =
            extract_tag_text(block, "UniqueID").unwrap_or_else(|| "unknown".to_string());

        let file_offset = parse_u64_tag(block, "FileOffset")?;
        let duration = parse_u64_tag(block, "Duration")?;

        let level1 = parse_level1(block)?;
        let level2 = parse_level2(block)?;
        let level6 = parse_level6(block)?;

        shots.push(DvXmlShot {
            unique_id,
            file_offset,
            duration,
            level1,
            level2,
            level6,
        });

        search_start = start_abs + end_rel + "</Shot>".len();
    }

    Ok(shots)
}

fn parse_u32_tag(xml: &str, tag: &str) -> Result<u32, DolbyVisionError> {
    match extract_tag_text(xml, tag) {
        Some(s) => s.parse::<u32>().map_err(|e| {
            DolbyVisionError::Generic(format!("Invalid <{}> u32 value '{}': {}", tag, s, e))
        }),
        None => Ok(0),
    }
}

fn parse_u64_tag(xml: &str, tag: &str) -> Result<u64, DolbyVisionError> {
    match extract_tag_text(xml, tag) {
        Some(s) => s.parse::<u64>().map_err(|e| {
            DolbyVisionError::Generic(format!("Invalid <{}> u64 value '{}': {}", tag, s, e))
        }),
        None => Ok(0),
    }
}

fn parse_f32_tag(xml: &str, tag: &str) -> Result<f32, DolbyVisionError> {
    match extract_tag_text(xml, tag) {
        Some(s) => s.parse::<f32>().map_err(|e| {
            DolbyVisionError::Generic(format!("Invalid <{}> f32 value '{}': {}", tag, s, e))
        }),
        None => Ok(0.0),
    }
}

fn parse_level1(block: &str) -> Result<Option<Level1>, DolbyVisionError> {
    if !block.contains("<Level1>") {
        return Ok(None);
    }
    let start = block.find("<Level1>").unwrap_or(0);
    let end = block[start..]
        .find("</Level1>")
        .unwrap_or(block.len() - start);
    let l1_block = &block[start..start + end + "</Level1>".len()];

    Ok(Some(Level1 {
        min_pq: parse_u32_tag(l1_block, "MinPQ")?,
        max_pq: parse_u32_tag(l1_block, "MaxPQ")?,
        avg_pq: parse_u32_tag(l1_block, "AvgPQ")?,
    }))
}

fn parse_level2(block: &str) -> Result<Option<Level2>, DolbyVisionError> {
    if !block.contains("<Level2>") {
        return Ok(None);
    }
    let start = block.find("<Level2>").unwrap_or(0);
    let end = block[start..]
        .find("</Level2>")
        .unwrap_or(block.len() - start);
    let l2_block = &block[start..start + end + "</Level2>".len()];

    Ok(Some(Level2 {
        target_max_pq: parse_u32_tag(l2_block, "TargetMaxPQ")?,
        trim_slope: parse_f32_tag(l2_block, "TrimSlope")?,
        trim_offset: parse_f32_tag(l2_block, "TrimOffset")?,
        trim_power: parse_f32_tag(l2_block, "TrimPower")?,
        trim_saturation_gain: parse_f32_tag(l2_block, "TrimSaturationGain")?,
    }))
}

fn parse_level6(block: &str) -> Result<Option<Level6>, DolbyVisionError> {
    if !block.contains("<Level6>") {
        return Ok(None);
    }
    let start = block.find("<Level6>").unwrap_or(0);
    let end = block[start..]
        .find("</Level6>")
        .unwrap_or(block.len() - start);
    let l6_block = &block[start..start + end + "</Level6>".len()];

    Ok(Some(Level6 {
        max_cll: parse_u32_tag(l6_block, "MaxCLL")?,
        max_fall: parse_u32_tag(l6_block, "MaxFALL")?,
    }))
}

// ── Validation ────────────────────────────────────────────────────────────────

/// A single validation error found in a `DvXmlMetadata` document.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationError {
    /// Shot unique ID where the error was found (empty for document-level errors).
    pub shot_id: String,
    /// Human-readable description of the validation failure.
    pub message: String,
}

impl ValidationError {
    fn new(shot_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            shot_id: shot_id.into(),
            message: message.into(),
        }
    }
}

/// Validate a `DvXmlMetadata` document for conformance.
///
/// Checks performed:
/// - PQ values in [0, 4095]
/// - Level 2 trim_slope in [0.0, 2.0]
/// - Level 2 trim_offset in [-1.0, 1.0]
/// - Level 2 trim_power in [0.0, 2.0]
/// - Level 2 trim_saturation_gain in [0.0, 2.0]
/// - Level 6 MaxCLL >= MaxFALL
/// - Shot duration > 0
/// - No empty unique_ids
///
/// Returns a list of `ValidationError` (empty = valid).
#[must_use]
pub fn validate_dv_metadata(metadata: &DvXmlMetadata) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    for shot in &metadata.shots {
        let id = &shot.unique_id;

        if id.is_empty() {
            errors.push(ValidationError::new("", "Shot has empty UniqueID"));
        }

        if shot.duration == 0 {
            errors.push(ValidationError::new(id, "Shot duration must be > 0"));
        }

        if let Some(l1) = &shot.level1 {
            if l1.min_pq > 4095 {
                errors.push(ValidationError::new(
                    id,
                    format!("Level1 MinPQ {} exceeds 4095", l1.min_pq),
                ));
            }
            if l1.max_pq > 4095 {
                errors.push(ValidationError::new(
                    id,
                    format!("Level1 MaxPQ {} exceeds 4095", l1.max_pq),
                ));
            }
            if l1.avg_pq > 4095 {
                errors.push(ValidationError::new(
                    id,
                    format!("Level1 AvgPQ {} exceeds 4095", l1.avg_pq),
                ));
            }
            if l1.min_pq > l1.max_pq {
                errors.push(ValidationError::new(
                    id,
                    format!("Level1 MinPQ {} > MaxPQ {}", l1.min_pq, l1.max_pq),
                ));
            }
        }

        if let Some(l2) = &shot.level2 {
            if l2.target_max_pq > 4095 {
                errors.push(ValidationError::new(
                    id,
                    format!("Level2 TargetMaxPQ {} exceeds 4095", l2.target_max_pq),
                ));
            }
            if !(0.0..=2.0).contains(&l2.trim_slope) {
                errors.push(ValidationError::new(
                    id,
                    format!("Level2 TrimSlope {} not in [0.0, 2.0]", l2.trim_slope),
                ));
            }
            if !(-1.0..=1.0).contains(&l2.trim_offset) {
                errors.push(ValidationError::new(
                    id,
                    format!("Level2 TrimOffset {} not in [-1.0, 1.0]", l2.trim_offset),
                ));
            }
            if !(0.0..=2.0).contains(&l2.trim_power) {
                errors.push(ValidationError::new(
                    id,
                    format!("Level2 TrimPower {} not in [0.0, 2.0]", l2.trim_power),
                ));
            }
            if !(0.0..=2.0).contains(&l2.trim_saturation_gain) {
                errors.push(ValidationError::new(
                    id,
                    format!(
                        "Level2 TrimSaturationGain {} not in [0.0, 2.0]",
                        l2.trim_saturation_gain
                    ),
                ));
            }
        }

        if let Some(l6) = &shot.level6 {
            if l6.max_fall > l6.max_cll {
                errors.push(ValidationError::new(
                    id,
                    format!(
                        "Level6 MaxFALL {} > MaxCLL {} (invalid)",
                        l6.max_fall, l6.max_cll
                    ),
                ));
            }
        }
    }

    errors
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_shot(id: &str, offset: u64, duration: u64) -> DvXmlShot {
        DvXmlShot {
            unique_id: id.to_string(),
            file_offset: offset,
            duration,
            level1: Some(Level1 {
                min_pq: 100,
                max_pq: 2081,
                avg_pq: 800,
            }),
            level2: Some(Level2 {
                target_max_pq: 2081,
                trim_slope: 1.0,
                trim_offset: 0.0,
                trim_power: 1.0,
                trim_saturation_gain: 1.0,
            }),
            level6: Some(Level6 {
                max_cll: 1000,
                max_fall: 400,
            }),
        }
    }

    fn make_display_target(nits: u32) -> DisplayTarget {
        DisplayTarget {
            nits,
            primaries: "BT.2020".to_string(),
            peak_luminance_type: "absolute".to_string(),
        }
    }

    fn make_metadata() -> DvXmlMetadata {
        let mut meta = DvXmlMetadata::new("CM v4.0");
        meta.shots.push(make_shot("shot-001", 0, 24));
        meta.display_targets.push(make_display_target(1000));
        meta
    }

    // ── Serialization ────────────────────────────────────────────────────────

    #[test]
    fn test_serialize_contains_version() {
        let meta = make_metadata();
        let xml = serialize_dv_xml(&meta);
        assert!(xml.contains("CM v4.0"), "version not found in:\n{xml}");
    }

    #[test]
    fn test_serialize_contains_shot_id() {
        let meta = make_metadata();
        let xml = serialize_dv_xml(&meta);
        assert!(xml.contains("shot-001"), "shot id not found in:\n{xml}");
    }

    #[test]
    fn test_serialize_contains_level1_max_pq() {
        let meta = make_metadata();
        let xml = serialize_dv_xml(&meta);
        assert!(
            xml.contains("<MaxPQ>2081</MaxPQ>"),
            "MaxPQ not found in:\n{xml}"
        );
    }

    #[test]
    fn test_serialize_contains_level2_trim_slope() {
        let meta = make_metadata();
        let xml = serialize_dv_xml(&meta);
        assert!(
            xml.contains("<TrimSlope>"),
            "TrimSlope not found in:\n{xml}"
        );
    }

    #[test]
    fn test_serialize_contains_level6_max_cll() {
        let meta = make_metadata();
        let xml = serialize_dv_xml(&meta);
        assert!(
            xml.contains("<MaxCLL>1000</MaxCLL>"),
            "MaxCLL not found in:\n{xml}"
        );
    }

    #[test]
    fn test_serialize_contains_display_target_nits() {
        let meta = make_metadata();
        let xml = serialize_dv_xml(&meta);
        assert!(
            xml.contains("<Nits>1000</Nits>"),
            "Display target nits not found in:\n{xml}"
        );
    }

    #[test]
    fn test_serialize_well_formed_root() {
        let meta = make_metadata();
        let xml = serialize_dv_xml(&meta);
        assert!(
            xml.starts_with("<?xml"),
            "Should start with XML declaration"
        );
        assert!(xml.contains("<DolbyLabsMDF"), "Root element missing");
        assert!(xml.contains("</DolbyLabsMDF>"), "Root close tag missing");
    }

    #[test]
    fn test_serialize_empty_metadata() {
        let meta = DvXmlMetadata::new("v1.0");
        let xml = serialize_dv_xml(&meta);
        assert!(xml.contains("v1.0"));
        assert!(xml.contains("</DolbyLabsMDF>"));
    }

    #[test]
    fn test_serialize_xml_escape_in_version() {
        let meta = DvXmlMetadata::new("v1 & v2");
        let xml = serialize_dv_xml(&meta);
        assert!(
            xml.contains("v1 &amp; v2"),
            "Ampersand not escaped in:\n{xml}"
        );
    }

    // ── Parsing ──────────────────────────────────────────────────────────────

    #[test]
    fn test_parse_roundtrip_version() {
        let meta = make_metadata();
        let xml = serialize_dv_xml(&meta);
        let parsed = parse_dv_xml(&xml).expect("parse should succeed");
        assert_eq!(parsed.version, "CM v4.0");
    }

    #[test]
    fn test_parse_roundtrip_shot_count() {
        let meta = make_metadata();
        let xml = serialize_dv_xml(&meta);
        let parsed = parse_dv_xml(&xml).expect("parse should succeed");
        assert_eq!(parsed.shots.len(), 1);
    }

    #[test]
    fn test_parse_roundtrip_shot_id() {
        let meta = make_metadata();
        let xml = serialize_dv_xml(&meta);
        let parsed = parse_dv_xml(&xml).expect("parse should succeed");
        assert_eq!(parsed.shots[0].unique_id, "shot-001");
    }

    #[test]
    fn test_parse_roundtrip_level1() {
        let meta = make_metadata();
        let xml = serialize_dv_xml(&meta);
        let parsed = parse_dv_xml(&xml).expect("parse should succeed");
        let l1 = parsed.shots[0]
            .level1
            .as_ref()
            .expect("Level1 should be present");
        assert_eq!(l1.max_pq, 2081);
        assert_eq!(l1.min_pq, 100);
    }

    #[test]
    fn test_parse_roundtrip_level2() {
        let meta = make_metadata();
        let xml = serialize_dv_xml(&meta);
        let parsed = parse_dv_xml(&xml).expect("parse should succeed");
        let l2 = parsed.shots[0]
            .level2
            .as_ref()
            .expect("Level2 should be present");
        assert_eq!(l2.target_max_pq, 2081);
        assert!(
            (l2.trim_slope - 1.0).abs() < 1e-4,
            "slope={}",
            l2.trim_slope
        );
    }

    #[test]
    fn test_parse_roundtrip_level6() {
        let meta = make_metadata();
        let xml = serialize_dv_xml(&meta);
        let parsed = parse_dv_xml(&xml).expect("parse should succeed");
        let l6 = parsed.shots[0]
            .level6
            .as_ref()
            .expect("Level6 should be present");
        assert_eq!(l6.max_cll, 1000);
        assert_eq!(l6.max_fall, 400);
    }

    #[test]
    fn test_parse_roundtrip_display_target_count() {
        let meta = make_metadata();
        let xml = serialize_dv_xml(&meta);
        let parsed = parse_dv_xml(&xml).expect("parse should succeed");
        assert_eq!(parsed.display_targets.len(), 1);
    }

    #[test]
    fn test_parse_roundtrip_display_target_nits() {
        let meta = make_metadata();
        let xml = serialize_dv_xml(&meta);
        let parsed = parse_dv_xml(&xml).expect("parse should succeed");
        assert_eq!(parsed.display_targets[0].nits, 1000);
    }

    #[test]
    fn test_parse_multiple_shots() {
        let mut meta = DvXmlMetadata::new("CM v4.0");
        meta.shots.push(make_shot("s001", 0, 24));
        meta.shots.push(make_shot("s002", 1000, 48));
        meta.shots.push(make_shot("s003", 5000, 36));
        let xml = serialize_dv_xml(&meta);
        let parsed = parse_dv_xml(&xml).expect("parse should succeed");
        assert_eq!(parsed.shots.len(), 3);
    }

    #[test]
    fn test_parse_empty_document() {
        let meta = DvXmlMetadata::new("test");
        let xml = serialize_dv_xml(&meta);
        let parsed = parse_dv_xml(&xml).expect("parse should succeed");
        assert!(parsed.shots.is_empty());
        assert!(parsed.display_targets.is_empty());
    }

    // ── Validation ────────────────────────────────────────────────────────────

    #[test]
    fn test_validate_valid_metadata_no_errors() {
        let meta = make_metadata();
        let errors = validate_dv_metadata(&meta);
        assert!(errors.is_empty(), "Expected no errors, got: {errors:?}");
    }

    #[test]
    fn test_validate_pq_above_4095() {
        let mut meta = make_metadata();
        meta.shots[0].level1 = Some(Level1 {
            min_pq: 0,
            max_pq: 5000,
            avg_pq: 1000,
        });
        let errors = validate_dv_metadata(&meta);
        assert!(!errors.is_empty(), "Expected error for MaxPQ > 4095");
        assert!(errors.iter().any(|e| e.message.contains("MaxPQ")));
    }

    #[test]
    fn test_validate_min_pq_above_max_pq() {
        let mut meta = make_metadata();
        meta.shots[0].level1 = Some(Level1 {
            min_pq: 3000,
            max_pq: 1000,
            avg_pq: 2000,
        });
        let errors = validate_dv_metadata(&meta);
        assert!(errors.iter().any(|e| e.message.contains("MinPQ")));
    }

    #[test]
    fn test_validate_trim_slope_out_of_range() {
        let mut meta = make_metadata();
        if let Some(l2) = meta.shots[0].level2.as_mut() {
            l2.trim_slope = 3.0;
        }
        let errors = validate_dv_metadata(&meta);
        assert!(errors.iter().any(|e| e.message.contains("TrimSlope")));
    }

    #[test]
    fn test_validate_trim_offset_out_of_range() {
        let mut meta = make_metadata();
        if let Some(l2) = meta.shots[0].level2.as_mut() {
            l2.trim_offset = -2.0;
        }
        let errors = validate_dv_metadata(&meta);
        assert!(errors.iter().any(|e| e.message.contains("TrimOffset")));
    }

    #[test]
    fn test_validate_max_fall_exceeds_max_cll() {
        let mut meta = make_metadata();
        meta.shots[0].level6 = Some(Level6 {
            max_cll: 500,
            max_fall: 800,
        });
        let errors = validate_dv_metadata(&meta);
        assert!(errors.iter().any(|e| e.message.contains("MaxFALL")));
    }

    #[test]
    fn test_validate_zero_duration() {
        let mut meta = make_metadata();
        meta.shots[0].duration = 0;
        let errors = validate_dv_metadata(&meta);
        assert!(errors.iter().any(|e| e.message.contains("duration")));
    }

    #[test]
    fn test_validate_empty_shot_id() {
        let mut meta = make_metadata();
        meta.shots[0].unique_id = String::new();
        let errors = validate_dv_metadata(&meta);
        assert!(errors.iter().any(|e| e.message.contains("UniqueID")));
    }
}
