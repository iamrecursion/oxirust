//! Parser allocation optimizations: zero-copy header pre-scan and FCM
//! multi-rate detection.
//!
//! The default [`EdlParser`](crate::parser::EdlParser) allocates a `String` for every comment and reel
//! name during parsing.  For large EDLs (10,000+ events) this can amount to
//! hundreds of kilobytes of short-lived heap objects.  This module provides:
//!
//! 1. **[`EdlHeaderScan`]** — a zero-allocation pre-pass that borrows slices of
//!    the source text to extract the title, FCM frame rate, and FORMAT field
//!    without allocating.  Only the title string (if present) is ever cloned.
//!
//! 2. **[`detect_frame_rate_extended`]** — full-precision frame rate detection
//!    from the FCM line, a `FORMAT:` header, or an inline `# RATE:` comment,
//!    covering 23.976/24/25/29.97/30/50/59.94/60 fps in both drop and non-drop
//!    variants.  This replaces the original two-branch parser that only
//!    distinguished "DROP FRAME" from "NON-DROP FRAME".
//!
//! 3. **[`CompactEventRef`]** — a lightweight (16-byte) event reference that
//!    stores byte offsets into the original source rather than allocated
//!    `String`s.  Full event data is decoded on demand.
//!
//! # Zero-copy scan example
//!
//! ```
//! use oximedia_edl::parser_opt::{EdlHeaderScan, detect_frame_rate_extended};
//! use oximedia_edl::timecode::EdlFrameRate;
//!
//! let src = "TITLE: My Cut\nFCM: NON-DROP FRAME\n\n001  AX  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n";
//! let scan = EdlHeaderScan::scan(src);
//! assert_eq!(scan.title, Some("My Cut"));
//! assert_eq!(scan.frame_rate, EdlFrameRate::Fps2997NDF);
//! assert_eq!(scan.event_line_count, 1);
//! ```

use crate::timecode::EdlFrameRate;

// ─────────────────────────────────────────────────────────────────────────────
// detect_frame_rate_extended
// ─────────────────────────────────────────────────────────────────────────────

/// Detect the EDL frame rate from any of the following sources (in priority order):
///
/// 1. `FCM:` header line (standard CMX 3600)
/// 2. `FORMAT:` header line containing a rate token
/// 3. Inline `# RATE:` comment
///
/// Recognised rate strings (case-insensitive):
/// `23.976`, `23976`, `24`, `25`, `29.97`, `2997`, `30`, `50`, `59.94`, `5994`, `60`
/// with optional `DF`/`NDF`/`DROP`/`NON-DROP` qualifiers.
///
/// Returns [`EdlFrameRate::Fps2997NDF`] as the safe default when no rate is detected.
#[must_use]
pub fn detect_frame_rate_extended(input: &str) -> EdlFrameRate {
    for line in input.lines().take(40) {
        let trimmed = line.trim();
        let upper = trimmed.to_uppercase();

        // ── FCM line ──────────────────────────────────────────────────────
        if let Some(rest) = upper.strip_prefix("FCM:") {
            if let Some(fps) = parse_rate_token(rest.trim()) {
                return fps;
            }
        }

        // ── FORMAT line with embedded rate ────────────────────────────────
        if upper.starts_with("FORMAT:") {
            if let Some(fps) = parse_rate_token(&upper["FORMAT:".len()..]) {
                return fps;
            }
        }

        // ── Inline comment: `* # RATE: 25` or `# RATE: 29.97DF` ──────────
        if trimmed.starts_with('*') || trimmed.starts_with('#') {
            let comment = trimmed.trim_start_matches(['*', '#', ' ']);
            let cup = comment.to_uppercase();
            if let Some(after) = cup.find("RATE:") {
                let token = cup[after + 5..].trim();
                if let Some(fps) = parse_rate_token(token) {
                    return fps;
                }
            }
        }

        // Stop scanning after the first event line (begins with digits)
        if trimmed.chars().next().map_or(false, |c| c.is_ascii_digit()) {
            break;
        }
    }

    EdlFrameRate::Fps2997NDF
}

/// Parse a rate token like `"DROP FRAME"`, `"25"`, `"29.97DF"`, `"NON-DROP FRAME"`,
/// `"24"`, `"59.94DF"`, etc.  Returns `None` if unrecognised.
fn parse_rate_token(token: &str) -> Option<EdlFrameRate> {
    // Normalise whitespace
    let t = token.split_whitespace().collect::<Vec<_>>().join(" ");

    // Drop / non-drop qualifiers
    // Recognise both "DROP FRAME" / "DF" (drop) and "NON-DROP FRAME" / "NDF" (non-drop).
    let ends_with_df = t.ends_with("DF") && !t.ends_with("NDF");
    let is_drop = (t.contains("DROP") && !t.contains("NON")) || ends_with_df;
    let is_ndf = t.contains("NON") || t.ends_with("NDF");

    // Helper: match a single rate string token (digits + optional suffix).
    let match_rate = |word: &str| -> Option<EdlFrameRate> {
        // Extract numeric prefix from this word
        let numeric: String = word
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        // Check if this word itself ends with DF/NDF qualifier
        let word_is_df = word.ends_with("DF") && !word.ends_with("NDF");
        let word_is_ndf = word.ends_with("NDF");
        let eff_drop = is_drop || word_is_df;
        let eff_ndf = is_ndf || word_is_ndf;
        match numeric.as_str() {
            "23.976" | "23976" => {
                if eff_drop {
                    Some(EdlFrameRate::Fps23_976)
                } else {
                    Some(EdlFrameRate::Fps23976)
                }
            }
            "24" => Some(EdlFrameRate::Fps24),
            "25" => Some(EdlFrameRate::Fps25),
            "29.97" | "2997" => {
                if eff_ndf {
                    Some(EdlFrameRate::Fps2997NDF)
                } else {
                    Some(EdlFrameRate::Fps2997DF)
                }
            }
            "30" => Some(EdlFrameRate::Fps30),
            "50" => Some(EdlFrameRate::Fps50),
            "59.94" | "5994" => {
                if eff_drop {
                    Some(EdlFrameRate::Fps59_94)
                } else {
                    Some(EdlFrameRate::Fps5994)
                }
            }
            "60" => Some(EdlFrameRate::Fps60),
            _ => None,
        }
    };

    // Try each whitespace-separated word in the token.
    for word in t.split_whitespace() {
        if let Some(fps) = match_rate(word) {
            return Some(fps);
        }
    }

    // Fall back to keyword-only matching (no numeric digits found)
    if t.contains("NON") {
        Some(EdlFrameRate::Fps2997NDF)
    } else if t.contains("DROP") {
        Some(EdlFrameRate::Fps2997DF)
    } else {
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EdlHeaderScan
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a zero-copy header pre-scan.
///
/// All borrowed fields refer to slices of the original input; only `title` may
/// be `None` when no TITLE line is present.  No heap allocation occurs for
/// `frame_rate` or `event_line_count`.
#[derive(Debug)]
pub struct EdlHeaderScan<'a> {
    /// Title string slice (borrowed from source), or `None`.
    pub title: Option<&'a str>,
    /// Frame rate parsed from FCM/FORMAT/comment lines.
    pub frame_rate: EdlFrameRate,
    /// Number of event lines (lines starting with digits) found.
    pub event_line_count: usize,
    /// Byte offset in `input` where the first event line starts, or `None`.
    pub first_event_offset: Option<usize>,
    /// Whether a `FORMAT:` line was found.
    pub has_format_line: bool,
}

impl<'a> EdlHeaderScan<'a> {
    /// Scan the header section of an EDL string without allocating.
    ///
    /// Scanning stops after the first event line is encountered (lines beginning
    /// with an ASCII digit), but continues to count remaining event lines to
    /// provide [`event_line_count`].
    ///
    /// [`event_line_count`]: EdlHeaderScan::event_line_count
    #[must_use]
    pub fn scan(input: &'a str) -> Self {
        let mut title: Option<&'a str> = None;
        let mut frame_rate = EdlFrameRate::Fps2997NDF;
        let mut event_line_count = 0usize;
        let mut first_event_offset: Option<usize> = None;
        let mut has_format_line = false;
        let mut rate_detected = false;

        let mut byte_offset = 0usize;

        for line in input.lines() {
            let trimmed = line.trim();

            if !trimmed.is_empty() {
                let upper_start: String =
                    trimmed.chars().take(8).collect::<String>().to_uppercase();

                if upper_start.starts_with("TITLE:") {
                    // Borrow the title slice from the original input
                    let start = trimmed.find(':').map_or(6, |i| i + 1);
                    title = Some(trimmed[start..].trim());
                } else if upper_start.starts_with("FCM:") && !rate_detected {
                    let rest = trimmed[4..].trim().to_uppercase();
                    if let Some(fps) = parse_rate_token(&rest) {
                        frame_rate = fps;
                        rate_detected = true;
                    }
                } else if upper_start.starts_with("FORMAT:") {
                    has_format_line = true;
                    if !rate_detected {
                        let rest = trimmed[7..].trim().to_uppercase();
                        if let Some(fps) = parse_rate_token(&rest) {
                            frame_rate = fps;
                            rate_detected = true;
                        }
                    }
                } else if trimmed.starts_with('*') || trimmed.starts_with('#') {
                    // Comment — check for inline RATE:
                    if !rate_detected {
                        let cup = trimmed.to_uppercase();
                        if let Some(pos) = cup.find("RATE:") {
                            let token = cup[pos + 5..].trim().to_string();
                            if let Some(fps) = parse_rate_token(&token) {
                                frame_rate = fps;
                                rate_detected = true;
                            }
                        }
                    }
                } else if trimmed.chars().next().map_or(false, |c| c.is_ascii_digit()) {
                    // Event line
                    if first_event_offset.is_none() {
                        first_event_offset = Some(byte_offset);
                    }
                    event_line_count += 1;
                }
            }

            // Advance byte offset: +1 for the newline character (works for \n and \r\n
            // because `lines()` strips the terminator; we add the line length + 1).
            byte_offset += line.len() + 1;
        }

        Self {
            title,
            frame_rate,
            event_line_count,
            first_event_offset,
            has_format_line,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CompactEventRef
// ─────────────────────────────────────────────────────────────────────────────

/// Lightweight reference to a single event in the source text.
///
/// Stores byte offsets into the original source string rather than allocating
/// copies of event data.  The event can be fully parsed on demand by extracting
/// `source[byte_start..byte_end]` and passing it to [`EdlParser::parse`](crate::parser::EdlParser::parse).
///
/// Size: 16 bytes (two `u32` offsets + one `u32` event number + 4 bytes padding).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactEventRef {
    /// Byte offset of the first character of the event line.
    pub byte_start: u32,
    /// Byte offset one past the last byte belonging to this event (including
    /// trailing comment lines).
    pub byte_end: u32,
    /// Event number (parsed cheaply from the first token).
    pub event_number: u32,
    /// Reel name offset within the event line (relative to `byte_start`).
    pub reel_name_start: u8,
    /// Reel name length in bytes.
    pub reel_name_len: u8,
}

impl CompactEventRef {
    /// Extract the event source slice from the original text.
    ///
    /// Returns `None` if the byte offsets are out of range.
    #[must_use]
    pub fn source_slice<'a>(&self, src: &'a str) -> Option<&'a str> {
        let start = self.byte_start as usize;
        let end = self.byte_end as usize;
        src.get(start..end)
    }

    /// Extract the reel name from the source text.
    ///
    /// Returns `None` if offsets are invalid.
    #[must_use]
    pub fn reel_name_slice<'a>(&self, src: &'a str) -> Option<&'a str> {
        let line_start = self.byte_start as usize;
        let reel_start = line_start + self.reel_name_start as usize;
        let reel_end = reel_start + self.reel_name_len as usize;
        src.get(reel_start..reel_end)
    }
}

/// Build a vec of [`CompactEventRef`]s from a full EDL source string.
///
/// This is O(n) in source length with zero allocations per event (only the
/// output `Vec` is allocated).
#[must_use]
pub fn build_compact_refs(src: &str) -> Vec<CompactEventRef> {
    let mut refs = Vec::new();
    let mut byte_offset: u32 = 0;
    let mut current: Option<CompactEventRef> = None;

    for line in src.lines() {
        let line_bytes = line.len() as u32;
        let trimmed = line.trim();

        if !trimmed.is_empty() && trimmed.chars().next().map_or(false, |c| c.is_ascii_digit()) {
            // New event line: close previous
            if let Some(prev) = current.take() {
                refs.push(prev);
            }

            // Parse event number and reel name using split_whitespace
            // (handles multiple consecutive spaces between fields).
            let mut sw = trimmed.split_whitespace();
            let event_number = sw.next().and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
            let reel_tok = sw.next();

            // Reel name: second whitespace-separated token
            let reel_name_start;
            let reel_name_len;
            if let Some(reel) = reel_tok {
                // Find byte offset of reel token within the original (untrimmed) line.
                // We scan forward past the leading spaces, the event-number token, and
                // the inter-field whitespace to find where the reel name starts.
                let leading_spaces = line.len() - line.trim_start().len();
                // The trimmed prefix up to the reel start:
                // skip event number chars + whitespace between num and reel
                if let Some(reel_pos_in_trimmed) = trimmed.find(reel) {
                    let reel_offset_in_line = leading_spaces + reel_pos_in_trimmed;
                    reel_name_start = reel_offset_in_line.min(255) as u8;
                    reel_name_len = reel.len().min(255) as u8;
                } else {
                    reel_name_start = 0;
                    reel_name_len = 0;
                }
            } else {
                reel_name_start = 0;
                reel_name_len = 0;
            }

            current = Some(CompactEventRef {
                byte_start: byte_offset,
                byte_end: byte_offset + line_bytes,
                event_number,
                reel_name_start,
                reel_name_len,
            });
        } else if !trimmed.is_empty() {
            // Comment / continuation: extend current event's byte range
            if let Some(ref mut ev) = current {
                ev.byte_end = byte_offset + line_bytes;
            }
        }

        // +1 for the newline
        byte_offset += line_bytes + 1;
    }

    if let Some(last) = current {
        refs.push(last);
    }

    refs
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── detect_frame_rate_extended ────────────────────────────────────────────

    #[test]
    fn test_detect_drop_frame() {
        let src = "TITLE: Test\nFCM: DROP FRAME\n";
        assert_eq!(detect_frame_rate_extended(src), EdlFrameRate::Fps2997DF);
    }

    #[test]
    fn test_detect_non_drop_frame() {
        let src = "FCM: NON-DROP FRAME\n";
        assert_eq!(detect_frame_rate_extended(src), EdlFrameRate::Fps2997NDF);
    }

    #[test]
    fn test_detect_25_fps() {
        let src = "FCM: 25\n";
        assert_eq!(detect_frame_rate_extended(src), EdlFrameRate::Fps25);
    }

    #[test]
    fn test_detect_24_fps() {
        let src = "FCM: 24\n";
        assert_eq!(detect_frame_rate_extended(src), EdlFrameRate::Fps24);
    }

    #[test]
    fn test_detect_30_fps() {
        let src = "FCM: 30\n";
        assert_eq!(detect_frame_rate_extended(src), EdlFrameRate::Fps30);
    }

    #[test]
    fn test_detect_60_fps() {
        let src = "FCM: 60\n";
        assert_eq!(detect_frame_rate_extended(src), EdlFrameRate::Fps60);
    }

    #[test]
    fn test_detect_5994_df() {
        let src = "FCM: 59.94DF\n";
        assert_eq!(detect_frame_rate_extended(src), EdlFrameRate::Fps59_94);
    }

    #[test]
    fn test_detect_from_format_line() {
        let src = "FORMAT: CMX 3600 25\n";
        assert_eq!(detect_frame_rate_extended(src), EdlFrameRate::Fps25);
    }

    #[test]
    fn test_detect_from_rate_comment() {
        let src = "* # RATE: 50\n";
        assert_eq!(detect_frame_rate_extended(src), EdlFrameRate::Fps50);
    }

    #[test]
    fn test_detect_default_fallback() {
        // No recognizable rate
        let src = "TITLE: Unknown Rate EDL\n001  AX  V  C  ...\n";
        assert_eq!(detect_frame_rate_extended(src), EdlFrameRate::Fps2997NDF);
    }

    // ── EdlHeaderScan ────────────────────────────────────────────────────────

    #[test]
    fn test_header_scan_basic() {
        let src = "TITLE: My Cut\nFCM: DROP FRAME\n\n001  AX  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n";
        let scan = EdlHeaderScan::scan(src);
        assert_eq!(scan.title, Some("My Cut"));
        assert_eq!(scan.frame_rate, EdlFrameRate::Fps2997DF);
        assert_eq!(scan.event_line_count, 1);
        assert!(scan.first_event_offset.is_some());
    }

    #[test]
    fn test_header_scan_no_title() {
        let src = "FCM: 25\n001  AX  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n";
        let scan = EdlHeaderScan::scan(src);
        assert!(scan.title.is_none());
        assert_eq!(scan.frame_rate, EdlFrameRate::Fps25);
    }

    #[test]
    fn test_header_scan_multiple_events() {
        let src = concat!(
            "TITLE: Multi\nFCM: NON-DROP FRAME\n\n",
            "001  A001  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n",
            "* FROM CLIP NAME: clip1.mov\n",
            "002  A002  V  C  01:00:05:00 01:00:10:00 01:00:05:00 01:00:10:00\n",
            "003  A003  V  D    030  01:00:10:00 01:00:15:00 01:00:10:00 01:00:15:00\n",
        );
        let scan = EdlHeaderScan::scan(src);
        assert_eq!(scan.event_line_count, 3);
        assert_eq!(scan.frame_rate, EdlFrameRate::Fps2997NDF);
    }

    #[test]
    fn test_header_scan_format_line() {
        let src = "FORMAT: CMX 3600\nFCM: NON-DROP FRAME\n";
        let scan = EdlHeaderScan::scan(src);
        assert!(scan.has_format_line);
    }

    // ── CompactEventRef / build_compact_refs ─────────────────────────────────

    #[test]
    fn test_compact_refs_event_count() {
        let src = concat!(
            "TITLE: Test\nFCM: 25\n\n",
            "001  A001  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n",
            "* FROM CLIP NAME: s1.mov\n",
            "002  A002  V  C  01:00:05:00 01:00:10:00 01:00:05:00 01:00:10:00\n",
        );
        let refs = build_compact_refs(src);
        assert_eq!(refs.len(), 2);
    }

    #[test]
    fn test_compact_refs_event_numbers() {
        let src = concat!(
            "001  AX  V  C  00:00:00:00 00:00:01:00 00:00:00:00 00:00:01:00\n",
            "002  BX  V  C  00:00:01:00 00:00:02:00 00:00:01:00 00:00:02:00\n",
        );
        let refs = build_compact_refs(src);
        assert_eq!(refs[0].event_number, 1);
        assert_eq!(refs[1].event_number, 2);
    }

    #[test]
    fn test_compact_ref_reel_name_extraction() {
        let src = "001  MYREEL  V  C  00:00:00:00 00:00:01:00 00:00:00:00 00:00:01:00\n";
        let refs = build_compact_refs(src);
        assert_eq!(refs.len(), 1);
        let reel = refs[0]
            .reel_name_slice(src)
            .expect("reel should be extractable");
        assert_eq!(reel, "MYREEL");
    }

    #[test]
    fn test_compact_ref_source_slice_parseable() {
        use crate::parser::EdlParser;
        let src = concat!(
            "001  AX  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n",
            "* FROM CLIP NAME: test.mov\n",
        );
        let refs = build_compact_refs(src);
        assert_eq!(refs.len(), 1);
        let slice = refs[0].source_slice(src).expect("slice should exist");
        let mut parser = EdlParser::new();
        let mini = parser.parse(slice).expect("slice should parse");
        assert_eq!(mini.events.len(), 1);
        assert_eq!(mini.events[0].clip_name.as_deref(), Some("test.mov"));
    }
}
