//! Subtitle format conversion utilities.
//!
//! Provides serialization and parsing for SRT and WebVTT subtitle formats,
//! with time-formatting helpers and a shared `SubtitleEntry` type.

/// A single subtitle entry with timing and text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubtitleEntry {
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
    /// Text content (may contain newlines for multi-line subtitles).
    pub text: String,
}

impl SubtitleEntry {
    /// Create a new subtitle entry.
    #[must_use]
    pub fn new(start_ms: u64, end_ms: u64, text: impl Into<String>) -> Self {
        Self {
            start_ms,
            end_ms,
            text: text.into(),
        }
    }

    /// Duration of this entry in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Number of text lines in this entry.
    #[must_use]
    pub fn line_count(&self) -> usize {
        if self.text.is_empty() {
            return 0;
        }
        self.text.lines().count()
    }
}

// ── Time formatting helpers ──────────────────────────────────────────────────

/// Format a millisecond timestamp as an SRT time string `HH:MM:SS,mmm`.
#[must_use]
pub fn format_ms_to_srt(ms: u64) -> String {
    let millis = ms % 1000;
    let total_secs = ms / 1000;
    let secs = total_secs % 60;
    let total_mins = total_secs / 60;
    let mins = total_mins % 60;
    let hours = total_mins / 60;
    format!("{hours:02}:{mins:02}:{secs:02},{millis:03}")
}

/// Format a millisecond timestamp as a WebVTT time string `HH:MM:SS.mmm`.
#[must_use]
pub fn format_ms_to_vtt(ms: u64) -> String {
    let millis = ms % 1000;
    let total_secs = ms / 1000;
    let secs = total_secs % 60;
    let total_mins = total_secs / 60;
    let mins = total_mins % 60;
    let hours = total_mins / 60;
    format!("{hours:02}:{mins:02}:{secs:02}.{millis:03}")
}

/// Parse an SRT time string `HH:MM:SS,mmm` to milliseconds.
///
/// Returns `None` if the string is not a valid SRT timestamp.
fn parse_srt_timestamp(s: &str) -> Option<u64> {
    // Expected: HH:MM:SS,mmm
    let s = s.trim();
    let (time_part, millis_part) = s.split_once(',')?;
    let millis: u64 = millis_part.trim().parse().ok()?;
    let parts: Vec<&str> = time_part.split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    let hours: u64 = parts[0].trim().parse().ok()?;
    let mins: u64 = parts[1].trim().parse().ok()?;
    let secs: u64 = parts[2].trim().parse().ok()?;
    Some(((hours * 3600 + mins * 60 + secs) * 1000) + millis)
}

// ── SRT Serializer ───────────────────────────────────────────────────────────

/// Serializes subtitle entries to the SubRip (SRT) format.
pub struct SrtSerializer;

impl SrtSerializer {
    /// Serialize a slice of `SubtitleEntry` values to an SRT-format string.
    ///
    /// Each block is numbered starting from 1.
    #[must_use]
    pub fn to_srt(entries: &[SubtitleEntry]) -> String {
        let mut out = String::new();
        for (i, entry) in entries.iter().enumerate() {
            let n = i + 1;
            let start = format_ms_to_srt(entry.start_ms);
            let end = format_ms_to_srt(entry.end_ms);
            out.push_str(&format!("{n}\n{start} --> {end}\n{}\n\n", entry.text));
        }
        out
    }
}

// ── SRT Parser ───────────────────────────────────────────────────────────────

/// Parses SubRip (SRT) formatted text into subtitle entries.
pub struct SrtParser;

impl SrtParser {
    /// Parse SRT-formatted text into a `Vec<SubtitleEntry>`.
    ///
    /// # Errors
    ///
    /// Returns a `String` error message if the input cannot be parsed.
    pub fn from_srt(input: &str) -> Result<Vec<SubtitleEntry>, String> {
        let mut entries = Vec::new();

        // Split into blocks separated by blank lines
        let blocks: Vec<&str> = input
            .split("\n\n")
            .map(str::trim)
            .filter(|b| !b.is_empty())
            .collect();

        for block in &blocks {
            let lines: Vec<&str> = block.lines().collect();
            if lines.len() < 3 {
                // Could be a block with only a number and timing but no text; skip
                if lines.len() < 2 {
                    continue;
                }
            }

            // First line should be the sequence number (we parse but don't store it)
            let _seq: u64 = lines[0]
                .trim()
                .parse()
                .map_err(|_| format!("Expected sequence number, got: {:?}", lines[0]))?;

            // Second line should be the timing
            let timing_line = lines[1];
            let parts: Vec<&str> = timing_line.splitn(2, "-->").collect();
            if parts.len() != 2 {
                return Err(format!("Invalid timing line: {timing_line:?}"));
            }
            let start_ms = parse_srt_timestamp(parts[0])
                .ok_or_else(|| format!("Invalid start timestamp: {:?}", parts[0]))?;
            let end_ms = parse_srt_timestamp(parts[1])
                .ok_or_else(|| format!("Invalid end timestamp: {:?}", parts[1]))?;

            // Remaining lines are the text content
            let text = lines[2..].join("\n");

            entries.push(SubtitleEntry {
                start_ms,
                end_ms,
                text,
            });
        }

        Ok(entries)
    }
}

// ── WebVTT Serializer ────────────────────────────────────────────────────────

/// Serializes subtitle entries to the WebVTT format.
pub struct VttSerializer;

impl VttSerializer {
    /// Serialize a slice of `SubtitleEntry` values to a WebVTT-format string.
    #[must_use]
    pub fn to_vtt(entries: &[SubtitleEntry]) -> String {
        let mut out = String::from("WEBVTT\n\n");
        for entry in entries {
            let start = format_ms_to_vtt(entry.start_ms);
            let end = format_ms_to_vtt(entry.end_ms);
            out.push_str(&format!("{start} --> {end}\n{}\n\n", entry.text));
        }
        out
    }
}

// ── WebVTT Parser ────────────────────────────────────────────────────────────

/// Parse a WebVTT timestamp `HH:MM:SS.mmm` (or `MM:SS.mmm`) to milliseconds.
///
/// Returns `None` if the string is not a valid WebVTT timestamp.
fn parse_vtt_timestamp(s: &str) -> Option<u64> {
    let s = s.trim();
    // Accept both `HH:MM:SS.mmm` and `MM:SS.mmm`
    let (time_part, millis_part) = s.split_once('.')?;
    let millis: u64 = millis_part.trim().parse().ok()?;
    let parts: Vec<&str> = time_part.split(':').collect();
    let (hours, mins, secs) = match parts.len() {
        3 => {
            let h: u64 = parts[0].trim().parse().ok()?;
            let m: u64 = parts[1].trim().parse().ok()?;
            let s: u64 = parts[2].trim().parse().ok()?;
            (h, m, s)
        }
        2 => {
            let m: u64 = parts[0].trim().parse().ok()?;
            let s: u64 = parts[1].trim().parse().ok()?;
            (0, m, s)
        }
        _ => return None,
    };
    Some(((hours * 3600 + mins * 60 + secs) * 1000) + millis)
}

/// Parses WebVTT formatted text into subtitle entries.
pub struct VttParser;

impl VttParser {
    /// Parse WebVTT-formatted text into a `Vec<SubtitleEntry>`.
    ///
    /// Skips the `WEBVTT` header, NOTE blocks, and cue settings.
    ///
    /// # Errors
    ///
    /// Returns a `String` error message if the input cannot be parsed.
    pub fn from_vtt(input: &str) -> Result<Vec<SubtitleEntry>, String> {
        let mut entries = Vec::new();

        // Split into blocks separated by blank lines
        let blocks: Vec<&str> = input
            .split("\n\n")
            .map(str::trim)
            .filter(|b| !b.is_empty())
            .collect();

        for block in &blocks {
            // Skip the WEBVTT header block and NOTE blocks
            if block.starts_with("WEBVTT") || block.starts_with("NOTE") {
                continue;
            }

            let lines: Vec<&str> = block.lines().collect();
            if lines.is_empty() {
                continue;
            }

            // Detect whether the first line is a cue identifier (no `-->`)
            let timing_line_idx = if lines[0].contains("-->") { 0 } else { 1 };
            let timing_line = match lines.get(timing_line_idx) {
                Some(l) => l,
                None => continue,
            };
            if !timing_line.contains("-->") {
                continue;
            }

            // Split timing line (may have cue settings after the end timestamp)
            let arrow_pos = timing_line
                .find("-->")
                .ok_or_else(|| format!("missing --> in timing line: {timing_line:?}"))?;
            let start_str = &timing_line[..arrow_pos];
            // End timestamp ends at the first space after `-->` + end-time token
            let after_arrow = timing_line[arrow_pos + 3..].trim();
            let end_str = after_arrow.split_whitespace().next().unwrap_or(after_arrow);

            let start_ms = parse_vtt_timestamp(start_str)
                .ok_or_else(|| format!("invalid WebVTT start: {start_str:?}"))?;
            let end_ms = parse_vtt_timestamp(end_str)
                .ok_or_else(|| format!("invalid WebVTT end: {end_str:?}"))?;

            let text_start = timing_line_idx + 1;
            let text = lines[text_start..].join("\n");

            if text.is_empty() {
                continue;
            }

            entries.push(SubtitleEntry {
                start_ms,
                end_ms,
                text,
            });
        }

        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── SubtitleEntry tests ─────────────────────────────────────────────────

    #[test]
    fn test_entry_duration_ms() {
        let e = SubtitleEntry::new(1000, 4500, "Hello");
        assert_eq!(e.duration_ms(), 3500);
    }

    #[test]
    fn test_entry_duration_saturating() {
        let e = SubtitleEntry::new(5000, 3000, "Bad order");
        assert_eq!(e.duration_ms(), 0);
    }

    #[test]
    fn test_entry_line_count_single() {
        let e = SubtitleEntry::new(0, 1000, "Hello");
        assert_eq!(e.line_count(), 1);
    }

    #[test]
    fn test_entry_line_count_multi() {
        let e = SubtitleEntry::new(0, 1000, "Line one\nLine two");
        assert_eq!(e.line_count(), 2);
    }

    #[test]
    fn test_entry_line_count_empty() {
        let e = SubtitleEntry::new(0, 1000, "");
        assert_eq!(e.line_count(), 0);
    }

    // ── Time formatting tests ───────────────────────────────────────────────

    #[test]
    fn test_format_ms_to_srt_zero() {
        assert_eq!(format_ms_to_srt(0), "00:00:00,000");
    }

    #[test]
    fn test_format_ms_to_srt_one_hour() {
        assert_eq!(format_ms_to_srt(3_600_000), "01:00:00,000");
    }

    #[test]
    fn test_format_ms_to_srt_complex() {
        // 1h 2m 3s 456ms
        let ms = 3600_000 + 2 * 60_000 + 3_000 + 456;
        assert_eq!(format_ms_to_srt(ms), "01:02:03,456");
    }

    #[test]
    fn test_format_ms_to_vtt_zero() {
        assert_eq!(format_ms_to_vtt(0), "00:00:00.000");
    }

    #[test]
    fn test_format_ms_to_vtt_complex() {
        let ms = 3600_000 + 2 * 60_000 + 3_000 + 789;
        assert_eq!(format_ms_to_vtt(ms), "01:02:03.789");
    }

    // ── SRT serializer tests ────────────────────────────────────────────────

    #[test]
    fn test_srt_serializer_basic() {
        let entries = vec![
            SubtitleEntry::new(1_000, 4_000, "Hello, world!"),
            SubtitleEntry::new(5_000, 8_000, "Second line."),
        ];
        let srt = SrtSerializer::to_srt(&entries);
        assert!(srt.contains("1\n"));
        assert!(srt.contains("00:00:01,000 --> 00:00:04,000"));
        assert!(srt.contains("Hello, world!"));
        assert!(srt.contains("2\n"));
        assert!(srt.contains("Second line."));
    }

    #[test]
    fn test_srt_serializer_empty() {
        let srt = SrtSerializer::to_srt(&[]);
        assert!(srt.is_empty());
    }

    // ── SRT parser tests ────────────────────────────────────────────────────

    #[test]
    fn test_srt_parser_roundtrip() {
        let entries = vec![SubtitleEntry::new(1_000, 4_000, "Hello!")];
        let srt = SrtSerializer::to_srt(&entries);
        let parsed = SrtParser::from_srt(&srt).expect("should succeed in test");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].start_ms, 1_000);
        assert_eq!(parsed[0].end_ms, 4_000);
        assert_eq!(parsed[0].text, "Hello!");
    }

    #[test]
    fn test_srt_parser_multi_entry() {
        let input = "1\n00:00:01,000 --> 00:00:04,000\nHello\n\n2\n00:00:05,000 --> 00:00:08,000\nWorld\n\n";
        let entries = SrtParser::from_srt(input).expect("should succeed in test");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].text, "Hello");
        assert_eq!(entries[1].text, "World");
        assert_eq!(entries[1].start_ms, 5_000);
    }

    #[test]
    fn test_srt_parser_invalid_sequence() {
        let bad = "abc\n00:00:01,000 --> 00:00:04,000\nText\n\n";
        let result = SrtParser::from_srt(bad);
        assert!(result.is_err());
    }

    // ── VTT serializer tests ────────────────────────────────────────────────

    #[test]
    fn test_vtt_serializer_header() {
        let entries = vec![SubtitleEntry::new(1_000, 4_000, "VTT line")];
        let vtt = VttSerializer::to_vtt(&entries);
        assert!(vtt.starts_with("WEBVTT"));
    }

    #[test]
    fn test_vtt_serializer_timing_format() {
        let entries = vec![SubtitleEntry::new(1_000, 4_000, "Hello")];
        let vtt = VttSerializer::to_vtt(&entries);
        assert!(vtt.contains("00:00:01.000 --> 00:00:04.000"));
    }

    #[test]
    fn test_vtt_serializer_empty() {
        let vtt = VttSerializer::to_vtt(&[]);
        assert_eq!(vtt, "WEBVTT\n\n");
    }

    // ── VTT parser tests ────────────────────────────────────────────────────

    #[test]
    fn test_vtt_parser_basic() {
        let vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nHello VTT\n\n";
        let entries = VttParser::from_vtt(vtt).expect("parse should succeed");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].start_ms, 1_000);
        assert_eq!(entries[0].end_ms, 4_000);
        assert_eq!(entries[0].text, "Hello VTT");
    }

    #[test]
    fn test_vtt_parser_with_cue_id() {
        let vtt = "WEBVTT\n\ncue1\n00:00:01.000 --> 00:00:04.000\nHello\n\n";
        let entries = VttParser::from_vtt(vtt).expect("parse should succeed");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].start_ms, 1_000);
    }

    #[test]
    fn test_vtt_parser_skips_note_blocks() {
        let vtt = "WEBVTT\n\nNOTE This is a comment\n\n00:00:01.000 --> 00:00:02.000\nText\n\n";
        let entries = VttParser::from_vtt(vtt).expect("parse should succeed");
        assert_eq!(entries.len(), 1);
    }

    // ── Round-trip timing precision tests ──────────────────────────────────

    /// SRT → WebVTT → SRT round-trip must be lossless at millisecond precision.
    #[test]
    fn test_srt_webvtt_roundtrip_ms_precision() {
        // Build 50 cues with varied millisecond-precise timestamps.
        let mut original: Vec<SubtitleEntry> = Vec::with_capacity(50);
        for i in 0u64..50 {
            // Use prime-number offsets to hit non-round millisecond values.
            let start_ms = i * 3_617 + (i * 127) % 1000;
            let end_ms = start_ms + 1_234 + (i * 37) % 999;
            original.push(SubtitleEntry::new(
                start_ms,
                end_ms,
                format!("Cue {i} text"),
            ));
        }

        // Step 1: SRT → serialize
        let srt_text = SrtSerializer::to_srt(&original);

        // Step 2: SRT → parse back
        let from_srt = SrtParser::from_srt(&srt_text).expect("SRT parse should succeed");
        assert_eq!(from_srt.len(), 50, "should recover all 50 entries from SRT");

        // Step 3: → WebVTT → serialize
        let vtt_text = VttSerializer::to_vtt(&from_srt);

        // Step 4: WebVTT → parse back
        let from_vtt = VttParser::from_vtt(&vtt_text).expect("VTT parse should succeed");
        assert_eq!(from_vtt.len(), 50, "should recover all 50 entries from VTT");

        // Verify exact millisecond preservation across the full round-trip.
        for (i, (orig, recovered)) in original.iter().zip(from_vtt.iter()).enumerate() {
            assert_eq!(
                orig.start_ms, recovered.start_ms,
                "start_ms mismatch at cue {i}: original={} recovered={}",
                orig.start_ms, recovered.start_ms
            );
            assert_eq!(
                orig.end_ms, recovered.end_ms,
                "end_ms mismatch at cue {i}: original={} recovered={}",
                orig.end_ms, recovered.end_ms
            );
            assert_eq!(orig.text, recovered.text, "text mismatch at cue {i}");
        }
    }

    #[test]
    fn test_srt_webvtt_roundtrip_boundary_timestamps() {
        // Edge cases: 0ms, exact-second, exact-minute, exact-hour.
        let edge_cases: &[(u64, u64, &str)] = &[
            (0, 1000, "zero start"),
            (1000, 2000, "exact second"),
            (60_000, 61_000, "exact minute"),
            (3_600_000, 3_601_000, "exact hour"),
            (3_599_999, 3_600_001, "straddles hour boundary"),
            (86_399_000, 86_400_000, "24h boundary"),
        ];

        let entries: Vec<SubtitleEntry> = edge_cases
            .iter()
            .map(|(s, e, t)| SubtitleEntry::new(*s, *e, t.to_string()))
            .collect();

        let srt = SrtSerializer::to_srt(&entries);
        let from_srt = SrtParser::from_srt(&srt).expect("SRT parse");
        let vtt = VttSerializer::to_vtt(&from_srt);
        let from_vtt = VttParser::from_vtt(&vtt).expect("VTT parse");

        assert_eq!(from_vtt.len(), entries.len());
        for (orig, got) in entries.iter().zip(from_vtt.iter()) {
            assert_eq!(orig.start_ms, got.start_ms, "start_ms edge: {}", orig.text);
            assert_eq!(orig.end_ms, got.end_ms, "end_ms edge: {}", orig.text);
        }
    }
}
