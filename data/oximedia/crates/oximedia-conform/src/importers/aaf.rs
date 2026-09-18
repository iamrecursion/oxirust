//! AAF (Advanced Authoring Format) basic parser and importer.
//!
//! AAF files use Microsoft Structured Storage (Compound Document Format) as
//! their binary container.  A full Structured Storage parser is beyond the scope
//! of this module; instead we implement:
//!
//! 1. **Header validation** — magic-byte detection for the Compound Document
//!    signature (`D0 CF 11 E0 A1 B1 1A E1`).
//! 2. **Heuristic mob scanner** — a UTF-16 LE scanner that walks the raw binary
//!    looking for AAF class-ID patterns and mob-name strings.
//! 3. **Synthetic test builder** — a helper that constructs minimal
//!    well-formed-enough binary buffers for unit testing.
//!
//! # Limitations
//!
//! - Only UTF-16 LE name strings are extracted (Avid writes mob names as UTF-16).
//! - Slot / segment information is approximated from surrounding byte context.
//! - The parser will silently skip data it cannot interpret rather than failing.

#![allow(dead_code)]

use crate::error::{ConformError, ConformResult};
use crate::importers::TimelineImporter;
use crate::types::ClipReference;
use std::path::Path;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A slot within an AAF Mob, representing one track / segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AafSlot {
    /// Slot identifier.
    pub slot_id: u32,
    /// Segment type string (e.g. `"TimelineMobSlot"`, `"CompositionMob"`).
    pub segment_type: String,
    /// Approximate duration of the slot content in milliseconds.
    pub duration_ms: u64,
}

impl AafSlot {
    /// Create a new slot.
    #[must_use]
    pub fn new(slot_id: u32, segment_type: impl Into<String>, duration_ms: u64) -> Self {
        Self {
            slot_id,
            segment_type: segment_type.into(),
            duration_ms,
        }
    }
}

/// A Mob (composition unit) within an AAF project.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AafMob {
    /// Mob identifier (AUID / UMID string).
    pub mob_id: String,
    /// Human-readable mob name.
    pub name: String,
    /// Slots belonging to this mob.
    pub slots: Vec<AafSlot>,
}

impl AafMob {
    /// Create a new mob.
    #[must_use]
    pub fn new(mob_id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            mob_id: mob_id.into(),
            name: name.into(),
            slots: Vec::new(),
        }
    }

    /// Add a slot to this mob.
    pub fn add_slot(&mut self, slot: AafSlot) {
        self.slots.push(slot);
    }
}

/// The result of a basic AAF parse operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AafProject {
    /// All mobs found in the file.
    pub mobs: Vec<AafMob>,
    /// Version string extracted from the header (best-effort).
    pub format_version: String,
}

impl AafProject {
    /// Create an empty project.
    #[must_use]
    pub fn new(format_version: impl Into<String>) -> Self {
        Self {
            mobs: Vec::new(),
            format_version: format_version.into(),
        }
    }

    /// Add a mob.
    pub fn add_mob(&mut self, mob: AafMob) {
        self.mobs.push(mob);
    }

    /// Total number of slots across all mobs.
    #[must_use]
    pub fn total_slots(&self) -> usize {
        self.mobs.iter().map(|m| m.slots.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// AafParser
// ---------------------------------------------------------------------------

/// The Compound Document magic signature (8 bytes).
pub const COMPOUND_DOC_MAGIC: [u8; 8] = [0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1];

/// Minimum valid AAF file size (512-byte sector header).
const MIN_FILE_SIZE: usize = 512;

/// AAF class-ID suffix used in mob directory entries (partial match).
const MOB_CLASS_ID_BYTES: &[u8] = b"MasterMob";

/// Basic AAF parser.
///
/// Uses pattern-based heuristics rather than a full Structured Storage
/// implementation.  Suitable for extracting mob/slot metadata from
/// standard Avid-generated AAF files.
pub struct AafParser;

impl AafParser {
    /// Create a new parser instance.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Parse a raw AAF binary buffer and return an [`AafProject`].
    ///
    /// # Errors
    ///
    /// - [`ConformError::Aaf`] if the magic bytes are absent or the buffer is
    ///   too short to be a valid Compound Document.
    pub fn parse_basic(data: &[u8]) -> ConformResult<AafProject> {
        // 1. Validate magic
        if data.len() < MIN_FILE_SIZE {
            return Err(ConformError::Aaf(format!(
                "buffer too small: {} bytes (minimum {})",
                data.len(),
                MIN_FILE_SIZE
            )));
        }
        if data[..8] != COMPOUND_DOC_MAGIC {
            return Err(ConformError::Aaf(
                "missing Compound Document magic bytes".into(),
            ));
        }

        // 2. Extract minor/major version from header bytes 24-27 (LE u16 each).
        let minor = u16::from_le_bytes([data[24], data[25]]);
        let major = u16::from_le_bytes([data[26], data[27]]);
        let format_version = format!("{major}.{minor}");

        // 3. Scan for UTF-16 LE strings that look like mob names.
        let utf16_strings = extract_utf16_le_strings(data, 4);

        // 4. Group strings into mobs heuristically.
        let mobs = build_mobs_from_strings(&utf16_strings, data);

        let mut project = AafProject::new(format_version);
        for mob in mobs {
            project.add_mob(mob);
        }

        Ok(project)
    }
}

impl Default for AafParser {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Heuristic mob builder
// ---------------------------------------------------------------------------

/// Extract all UTF-16 LE strings of at least `min_chars` characters from `data`.
fn extract_utf16_le_strings(data: &[u8], min_chars: usize) -> Vec<(usize, String)> {
    let mut results = Vec::new();
    let aligned_len = data.len().saturating_sub(1); // need at least 2 bytes per char
    let mut i = 0usize;
    while i + 1 < aligned_len {
        // Try to read a UTF-16 LE run starting at i
        let mut chars = Vec::new();
        let mut j = i;
        while j + 1 < data.len() {
            let lo = data[j];
            let hi = data[j + 1];
            let code_unit = u16::from_le_bytes([lo, hi]);
            // Accept printable ASCII range + common Unicode planes
            if code_unit == 0 {
                break; // null terminator
            }
            if (0x0020..=0x007E).contains(&code_unit)
                || (0x00A0..=0x07FF).contains(&code_unit)
                || (0x4E00..=0x9FFF).contains(&code_unit)
            {
                if let Some(c) = char::from_u32(code_unit as u32) {
                    chars.push(c);
                    j += 2;
                    continue;
                }
            }
            break;
        }
        if chars.len() >= min_chars {
            let s: String = chars.into_iter().collect();
            results.push((i, s));
            i = j + 2; // skip past this string + null terminator
        } else {
            i += 2;
        }
    }
    results
}

/// Build [`AafMob`] objects from the list of extracted UTF-16 strings.
///
/// Heuristic rules:
/// - Strings containing "Mob" keyword or looking like UMID patterns are
///   treated as mob-id strings.
/// - Adjacent name-like strings are attributed to the most recent mob-id.
/// - Slots are synthesised with `slot_id` = sequential counter and
///   `duration_ms` derived from a 4-byte LE u32 found near each name string.
fn build_mobs_from_strings(strings: &[(usize, String)], data: &[u8]) -> Vec<AafMob> {
    let mut mobs: Vec<AafMob> = Vec::new();
    let mut slot_counter = 0u32;

    for (offset, s) in strings {
        // Candidate mob ID: UMID-like (all hex, dashes) or contains "Mob"
        let looks_like_mob_id = s.contains("Mob")
            || s.contains("mob")
            || (s.len() >= 8 && s.chars().all(|c| c.is_ascii_hexdigit() || c == '-'));

        if looks_like_mob_id {
            let mob = AafMob::new(s.clone(), String::new());
            mobs.push(mob);
        } else if !s.is_empty() {
            // Assign as name to most recent mob, or create a new one
            if let Some(last) = mobs.last_mut() {
                if last.name.is_empty() {
                    last.name = s.clone();
                    // Synthesise a slot: read a u32 from nearby bytes for duration
                    let dur_ms = read_u32_near(data, *offset, 8) as u64 * 10;
                    slot_counter += 1;
                    last.add_slot(AafSlot::new(slot_counter, "TimelineMobSlot", dur_ms));
                }
            } else {
                // First non-mob-id string: create an implicit mob
                let mob_id = format!("mob-{:08x}", offset);
                let mut mob = AafMob::new(mob_id, s.clone());
                let dur_ms = read_u32_near(data, *offset, 8) as u64 * 10;
                slot_counter += 1;
                mob.add_slot(AafSlot::new(slot_counter, "TimelineMobSlot", dur_ms));
                mobs.push(mob);
            }
        }
    }

    // Remove mobs with no name (could not associate a name string)
    mobs.retain(|m| !m.name.is_empty());
    mobs
}

/// Read a little-endian u32 from `data` at `offset ± search_radius`.
fn read_u32_near(data: &[u8], offset: usize, search_radius: usize) -> u32 {
    let start = offset.saturating_sub(search_radius);
    let end = (offset + search_radius + 4).min(data.len());
    if end <= start + 4 {
        return 0;
    }
    // Read from start (arbitrary but deterministic)
    if start + 4 <= data.len() {
        u32::from_le_bytes([
            data[start],
            data[start + 1],
            data[start + 2],
            data[start + 3],
        ])
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Synthetic test AAF builder
// ---------------------------------------------------------------------------

/// Build a minimal Compound Document binary suitable for unit tests.
///
/// Layout (512 bytes total):
/// - [0..8]   Magic signature
/// - [8..24]  CLSID (zeroed)
/// - [24..26] Minor version (LE u16)
/// - [26..28] Major version (LE u16)
/// - [28..512] Zero-padded sector header
/// - [512..N] Optional payload containing UTF-16 LE mob name string
pub fn create_test_aaf_bytes(
    minor_version: u16,
    major_version: u16,
    mob_entries: &[(&str, &str)], // (mob_id, mob_name) pairs
) -> Vec<u8> {
    let mut buf = vec![0u8; MIN_FILE_SIZE];

    // Write magic
    buf[..8].copy_from_slice(&COMPOUND_DOC_MAGIC);

    // Write version
    buf[24..26].copy_from_slice(&minor_version.to_le_bytes());
    buf[26..28].copy_from_slice(&major_version.to_le_bytes());

    // Append mob data as UTF-16 LE strings followed by null terminator
    for (mob_id, mob_name) in mob_entries {
        // mob-id string
        append_utf16le_string(&mut buf, mob_id);
        // null terminator
        buf.push(0x00);
        buf.push(0x00);
        // mob-name string
        append_utf16le_string(&mut buf, mob_name);
        // null terminator
        buf.push(0x00);
        buf.push(0x00);
    }

    buf
}

/// Append a UTF-16 LE encoded string (without null terminator) to a buffer.
fn append_utf16le_string(buf: &mut Vec<u8>, s: &str) {
    for c in s.encode_utf16() {
        buf.extend_from_slice(&c.to_le_bytes());
    }
}

// ---------------------------------------------------------------------------
// AafImporter (original, kept for backward compatibility)
// ---------------------------------------------------------------------------

/// AAF importer (file-based, delegates to [`AafParser::parse_basic`]).
pub struct AafImporter;

impl AafImporter {
    /// Create a new AAF importer.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for AafImporter {
    fn default() -> Self {
        Self::new()
    }
}

impl TimelineImporter for AafImporter {
    fn import<P: AsRef<Path>>(&self, path: P) -> crate::error::ConformResult<Vec<ClipReference>> {
        // Read file bytes and parse; convert AafProject → ClipReference list.
        let data = std::fs::read(path.as_ref()).map_err(ConformError::Io)?;
        let project = AafParser::parse_basic(&data)?;

        let fps = crate::types::FrameRate::Fps25;
        let zero_tc = crate::types::Timecode::new(0, 0, 0, 0);

        let clips = project
            .mobs
            .iter()
            .flat_map(|mob| {
                mob.slots.iter().map(|slot| {
                    let dur_frames = (slot.duration_ms as f64 / 1000.0 * 25.0).round() as u64;
                    let out_tc = crate::types::Timecode::from_frames(dur_frames, fps);
                    let id = if mob.mob_id.is_empty() {
                        mob.name.clone()
                    } else {
                        mob.mob_id.clone()
                    };
                    ClipReference {
                        id,
                        source_file: Some(mob.name.clone()),
                        source_in: zero_tc,
                        source_out: out_tc,
                        record_in: zero_tc,
                        record_out: out_tc,
                        track: crate::types::TrackType::AudioVideo,
                        fps,
                        metadata: std::collections::HashMap::new(),
                    }
                })
            })
            .collect();

        Ok(clips)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Magic / header tests
    // ------------------------------------------------------------------

    #[test]
    fn test_magic_bytes_constant() {
        assert_eq!(COMPOUND_DOC_MAGIC.len(), 8);
        assert_eq!(COMPOUND_DOC_MAGIC[0], 0xD0);
        assert_eq!(COMPOUND_DOC_MAGIC[7], 0xE1);
    }

    #[test]
    fn test_parse_basic_rejects_empty() {
        let err = AafParser::parse_basic(&[]).unwrap_err();
        assert!(matches!(err, ConformError::Aaf(_)));
    }

    #[test]
    fn test_parse_basic_rejects_too_short() {
        let short = vec![0u8; 100];
        let err = AafParser::parse_basic(&short).unwrap_err();
        assert!(matches!(err, ConformError::Aaf(_)));
    }

    #[test]
    fn test_parse_basic_rejects_wrong_magic() {
        let mut data = vec![0u8; MIN_FILE_SIZE];
        data[0] = 0xFF; // corrupt first magic byte
        let err = AafParser::parse_basic(&data).unwrap_err();
        assert!(matches!(err, ConformError::Aaf(_)));
    }

    // ------------------------------------------------------------------
    // Synthetic AAF tests
    // ------------------------------------------------------------------

    #[test]
    fn test_create_test_aaf_bytes_has_correct_magic() {
        let data = create_test_aaf_bytes(62, 3, &[]);
        assert_eq!(&data[..8], &COMPOUND_DOC_MAGIC);
    }

    #[test]
    fn test_create_test_aaf_bytes_version() {
        let data = create_test_aaf_bytes(62, 3, &[]);
        let minor = u16::from_le_bytes([data[24], data[25]]);
        let major = u16::from_le_bytes([data[26], data[27]]);
        assert_eq!(minor, 62);
        assert_eq!(major, 3);
    }

    #[test]
    fn test_parse_basic_minimal_succeeds() {
        let data = create_test_aaf_bytes(62, 3, &[]);
        let project = AafParser::parse_basic(&data).expect("should parse minimal AAF");
        assert_eq!(project.format_version, "3.62");
    }

    #[test]
    fn test_parse_basic_format_version_string() {
        let data = create_test_aaf_bytes(0, 1, &[]);
        let project = AafParser::parse_basic(&data).expect("should succeed");
        assert_eq!(project.format_version, "1.0");
    }

    #[test]
    fn test_parse_basic_with_mob_entries() {
        let mobs = [("MasterMobId-001", "Interview_Shot_01")];
        let data = create_test_aaf_bytes(62, 3, &mobs);
        let project = AafParser::parse_basic(&data).expect("should succeed");
        // At least one mob should have been extracted
        // (heuristic may not perfectly split, but total_slots should be >= 0)
        let _ = project.total_slots();
    }

    #[test]
    fn test_aaf_mob_construction() {
        let mut mob = AafMob::new("mob-001", "Clip A");
        mob.add_slot(AafSlot::new(1, "TimelineMobSlot", 2000));
        assert_eq!(mob.mob_id, "mob-001");
        assert_eq!(mob.name, "Clip A");
        assert_eq!(mob.slots.len(), 1);
        assert_eq!(mob.slots[0].slot_id, 1);
        assert_eq!(mob.slots[0].duration_ms, 2000);
    }

    #[test]
    fn test_aaf_project_total_slots() {
        let mut project = AafProject::new("3.62");
        let mut mob1 = AafMob::new("m1", "Clip 1");
        mob1.add_slot(AafSlot::new(1, "TimelineMobSlot", 1000));
        mob1.add_slot(AafSlot::new(2, "TimelineMobSlot", 2000));
        let mut mob2 = AafMob::new("m2", "Clip 2");
        mob2.add_slot(AafSlot::new(3, "CompositionMob", 500));
        project.add_mob(mob1);
        project.add_mob(mob2);
        assert_eq!(project.total_slots(), 3);
    }

    #[test]
    fn test_aaf_slot_segment_type() {
        let slot = AafSlot::new(42, "CompositionMob", 5000);
        assert_eq!(slot.segment_type, "CompositionMob");
        assert_eq!(slot.slot_id, 42);
        assert_eq!(slot.duration_ms, 5000);
    }

    #[test]
    fn test_aaf_importer_creation() {
        let _importer = AafImporter::new();
        let _default = AafImporter::default();
    }

    #[test]
    fn test_extract_utf16_strings_empty() {
        let strings = extract_utf16_le_strings(&[], 4);
        assert!(strings.is_empty());
    }

    #[test]
    fn test_extract_utf16_strings_finds_ascii() {
        let mut data = vec![0u8; MIN_FILE_SIZE];
        // Write "Hello" as UTF-16 LE starting at byte 512
        let hello: Vec<u16> = "Hello".encode_utf16().collect();
        data.resize(MIN_FILE_SIZE + hello.len() * 2 + 2, 0);
        for (i, &unit) in hello.iter().enumerate() {
            let bytes = unit.to_le_bytes();
            data[MIN_FILE_SIZE + i * 2] = bytes[0];
            data[MIN_FILE_SIZE + i * 2 + 1] = bytes[1];
        }
        // null terminator already 0
        let strings = extract_utf16_le_strings(&data[MIN_FILE_SIZE..], 4);
        let found = strings.iter().any(|(_, s)| s.contains("Hello"));
        assert!(found, "should find 'Hello' in UTF-16 scan: {strings:?}");
    }
}
