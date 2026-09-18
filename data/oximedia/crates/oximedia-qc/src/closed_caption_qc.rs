//! Closed caption quality control checking.
//!
//! This module validates closed captions for common issues such as
//! reading speed violations, overlapping captions, illegal characters,
//! and statistical analysis of caption coverage.

/// Style information for a caption.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptionStyle {
    /// Whether the text is italic.
    pub italic: bool,
    /// Whether the text is underlined.
    pub underline: bool,
    /// Text color.
    pub color: CaptionColor,
}

impl Default for CaptionStyle {
    fn default() -> Self {
        Self {
            italic: false,
            underline: false,
            color: CaptionColor::White,
        }
    }
}

/// Caption text color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptionColor {
    /// White text.
    White,
    /// Yellow text.
    Yellow,
    /// Green text.
    Green,
    /// Cyan text.
    Cyan,
    /// Blue text.
    Blue,
    /// Red text.
    Red,
    /// Magenta text.
    Magenta,
    /// Black text.
    Black,
}

impl CaptionColor {
    /// Returns the color name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::White => "white",
            Self::Yellow => "yellow",
            Self::Green => "green",
            Self::Cyan => "cyan",
            Self::Blue => "blue",
            Self::Red => "red",
            Self::Magenta => "magenta",
            Self::Black => "black",
        }
    }
}

/// A single caption frame.
#[derive(Debug, Clone)]
pub struct CaptionFrame {
    /// SMPTE timecode string (e.g., "00:01:23;14").
    pub timecode: String,
    /// Caption text content.
    pub text: String,
    /// Row/line position (0-based).
    pub line: u8,
    /// Column position (0-based).
    pub column: u8,
    /// Style information.
    pub style: CaptionStyle,
}

impl CaptionFrame {
    /// Creates a new caption frame.
    #[must_use]
    pub fn new(
        timecode: impl Into<String>,
        text: impl Into<String>,
        line: u8,
        column: u8,
        style: CaptionStyle,
    ) -> Self {
        Self {
            timecode: timecode.into(),
            text: text.into(),
            line,
            column,
            style,
        }
    }

    /// Returns the number of characters in the caption text.
    #[must_use]
    pub fn char_count(&self) -> usize {
        self.text.chars().count()
    }

    /// Returns the word count.
    #[must_use]
    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }
}

/// Issue types for caption QC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptionIssueType {
    /// Caption reading speed exceeds maximum.
    TooFast,
    /// Captions overlap in time.
    Overlap,
    /// Caption is missing a period/terminator.
    MissingPeriod,
    /// A word appears truncated.
    TruncatedWord,
    /// Caption contains illegal characters.
    IllegalChar,
    /// Gap between captions is too short.
    GapTooShort,
}

impl CaptionIssueType {
    /// Returns the issue description.
    #[must_use]
    pub fn description(self) -> &'static str {
        match self {
            Self::TooFast => "Caption reading speed exceeds 17 chars/sec",
            Self::Overlap => "Captions overlap in time",
            Self::MissingPeriod => "Caption text is missing end punctuation",
            Self::TruncatedWord => "Caption appears to contain a truncated word",
            Self::IllegalChar => "Caption contains an illegal character",
            Self::GapTooShort => "Gap between captions is less than 2 frames",
        }
    }
}

/// A quality control issue found in captions.
#[derive(Debug, Clone)]
pub struct CaptionQcIssue {
    /// Type of issue found.
    pub issue_type: CaptionIssueType,
    /// Frame index where the issue was found, if applicable.
    pub frame_idx: Option<u64>,
    /// Human-readable description of this specific issue.
    pub description: String,
}

impl CaptionQcIssue {
    /// Creates a new caption QC issue.
    #[must_use]
    pub fn new(
        issue_type: CaptionIssueType,
        frame_idx: Option<u64>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            issue_type,
            frame_idx,
            description: description.into(),
        }
    }
}

/// Legal characters for closed captions (ASCII printable + common Unicode).
fn is_legal_caption_char(c: char) -> bool {
    // Allow printable ASCII, common accented characters, and common punctuation
    (c.is_ascii_graphic() || c == ' ' || c.is_alphabetic())
        && !matches!(c, '\x00'..='\x1F' | '\x7F')
}

/// Checker for closed caption quality issues.
#[derive(Debug, Clone, Default)]
pub struct CaptionQcChecker;

impl CaptionQcChecker {
    /// Creates a new caption QC checker.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Parses a timecode string into seconds (supports "HH:MM:SS;FF" and "HH:MM:SS:FF").
    fn timecode_to_secs(tc: &str, fps: f32) -> f64 {
        // Try to parse "HH:MM:SS;FF" or "HH:MM:SS:FF"
        let normalized = tc.replace(';', ":");
        let parts: Vec<&str> = normalized.split(':').collect();
        if parts.len() == 4 {
            let h: f64 = parts[0].parse().unwrap_or(0.0);
            let m: f64 = parts[1].parse().unwrap_or(0.0);
            let s: f64 = parts[2].parse().unwrap_or(0.0);
            let f: f64 = parts[3].parse().unwrap_or(0.0);
            h * 3600.0 + m * 60.0 + s + f / f64::from(fps)
        } else if parts.len() == 3 {
            let h: f64 = parts[0].parse().unwrap_or(0.0);
            let m: f64 = parts[1].parse().unwrap_or(0.0);
            let s: f64 = parts[2].parse().unwrap_or(0.0);
            h * 3600.0 + m * 60.0 + s
        } else {
            0.0
        }
    }

    /// Checks caption frames for quality issues.
    ///
    /// - Reading speed maximum: 17 chars/sec
    /// - Minimum gap between captions: 2 frames
    #[must_use]
    pub fn check(captions: &[CaptionFrame], fps: f32) -> Vec<CaptionQcIssue> {
        let mut issues = Vec::new();

        // Parse all timecodes to seconds for comparison
        let times: Vec<f64> = captions
            .iter()
            .map(|c| Self::timecode_to_secs(&c.timecode, fps))
            .collect();

        let min_gap_secs = 2.0 / f64::from(fps);

        for (i, caption) in captions.iter().enumerate() {
            let start_secs = times[i];

            // Check for gap too short between consecutive captions
            if i + 1 < captions.len() {
                let next_start = times[i + 1];
                let gap = next_start - start_secs;
                if gap < min_gap_secs && gap >= 0.0 {
                    issues.push(CaptionQcIssue::new(
                        CaptionIssueType::GapTooShort,
                        Some(i as u64),
                        format!(
                            "Gap of {:.4}s between caption {} and {} is less than 2 frames ({:.4}s)",
                            gap,
                            i,
                            i + 1,
                            min_gap_secs
                        ),
                    ));
                }
                // Check overlap (next starts before current ends)
                // We don't have an explicit end time, so we estimate based on reading speed
                // Assume ~17 chars/sec display time
                let chars = caption.char_count();
                let display_time = if chars == 0 { 0.5 } else { chars as f64 / 17.0 };
                let end_secs = start_secs + display_time;
                if next_start < end_secs {
                    issues.push(CaptionQcIssue::new(
                        CaptionIssueType::Overlap,
                        Some(i as u64),
                        format!(
                            "Caption {} (ends ~{:.3}s) overlaps with caption {} (starts {:.3}s)",
                            i,
                            end_secs,
                            i + 1,
                            next_start
                        ),
                    ));
                }

                // Check reading speed if we know next caption time
                // chars / display_time > 17 → too fast
                if display_time > 0.0 && chars as f64 / display_time > 17.0 {
                    issues.push(CaptionQcIssue::new(
                        CaptionIssueType::TooFast,
                        Some(i as u64),
                        format!(
                            "Caption {} has {chars} chars with only {display_time:.3}s display time ({:.1} chars/sec > 17)",
                            i,
                            chars as f64 / display_time,
                        ),
                    ));
                }
            }

            // Check for illegal characters
            for c in caption.text.chars() {
                if !is_legal_caption_char(c) {
                    issues.push(CaptionQcIssue::new(
                        CaptionIssueType::IllegalChar,
                        Some(i as u64),
                        format!(
                            "Caption {} contains illegal character: {:?} (U+{:04X})",
                            i, c, c as u32
                        ),
                    ));
                    break; // Report once per caption
                }
            }

            // Check for truncated words (ends with a hyphen mid-word)
            let trimmed = caption.text.trim_end();
            if trimmed.ends_with('-') && !trimmed.ends_with("--") {
                issues.push(CaptionQcIssue::new(
                    CaptionIssueType::TruncatedWord,
                    Some(i as u64),
                    format!(
                        "Caption {} appears to end with a truncated word: {:?}",
                        i, caption.text
                    ),
                ));
            }
        }

        issues
    }
}

/// Statistical summary of caption data.
#[derive(Debug, Clone)]
pub struct CaptionStats {
    /// Total number of captions.
    pub caption_count: u64,
    /// Average words per minute.
    pub avg_words_per_min: f32,
    /// Percentage of total duration covered by captions.
    pub coverage_pct: f32,
}

impl CaptionStats {
    /// Computes statistics for a set of captions over a given total duration.
    #[must_use]
    pub fn compute(captions: &[CaptionFrame], total_duration_secs: f64) -> Self {
        let caption_count = captions.len() as u64;

        if caption_count == 0 || total_duration_secs <= 0.0 {
            return Self {
                caption_count,
                avg_words_per_min: 0.0,
                coverage_pct: 0.0,
            };
        }

        let total_words: usize = captions.iter().map(CaptionFrame::word_count).sum();
        let total_mins = total_duration_secs / 60.0;
        let avg_words_per_min = (total_words as f64 / total_mins) as f32;

        // Estimate coverage as fraction of time captions are displayed.
        // Assume each caption is displayed at ~17 chars/sec.
        let total_chars: usize = captions.iter().map(CaptionFrame::char_count).sum();
        let caption_secs = total_chars as f64 / 17.0;
        let coverage_pct = ((caption_secs / total_duration_secs) * 100.0).clamp(0.0, 100.0) as f32;

        Self {
            caption_count,
            avg_words_per_min,
            coverage_pct,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_caption(tc: &str, text: &str) -> CaptionFrame {
        CaptionFrame::new(tc, text, 14, 0, CaptionStyle::default())
    }

    #[test]
    fn test_caption_color_name() {
        assert_eq!(CaptionColor::White.name(), "white");
        assert_eq!(CaptionColor::Yellow.name(), "yellow");
        assert_eq!(CaptionColor::Black.name(), "black");
    }

    #[test]
    fn test_caption_frame_char_count() {
        let c = make_caption("00:00:01:00", "Hello world");
        assert_eq!(c.char_count(), 11);
    }

    #[test]
    fn test_caption_frame_word_count() {
        let c = make_caption("00:00:01:00", "Hello world foo");
        assert_eq!(c.word_count(), 3);
    }

    #[test]
    fn test_issue_type_description() {
        assert!(!CaptionIssueType::TooFast.description().is_empty());
        assert!(!CaptionIssueType::GapTooShort.description().is_empty());
    }

    #[test]
    fn test_no_issues_for_well_formed_captions() {
        let captions = vec![
            make_caption("00:00:01:00", "Hello world."),
            make_caption("00:00:05:00", "This is caption two."),
        ];
        let issues = CaptionQcChecker::check(&captions, 30.0);
        // No gap issues since gap = 4s >> 2 frames
        assert!(
            !issues
                .iter()
                .any(|i| i.issue_type == CaptionIssueType::GapTooShort),
            "Should not flag valid gap"
        );
    }

    #[test]
    fn test_illegal_char_detection() {
        let captions = vec![make_caption("00:00:01:00", "Hello\x07world")];
        let issues = CaptionQcChecker::check(&captions, 30.0);
        assert!(
            issues
                .iter()
                .any(|i| i.issue_type == CaptionIssueType::IllegalChar),
            "Should detect illegal char"
        );
    }

    #[test]
    fn test_truncated_word_detection() {
        let captions = vec![make_caption("00:00:01:00", "extra-")];
        let issues = CaptionQcChecker::check(&captions, 30.0);
        assert!(
            issues
                .iter()
                .any(|i| i.issue_type == CaptionIssueType::TruncatedWord),
            "Should detect truncated word"
        );
    }

    #[test]
    fn test_gap_too_short() {
        // Two captions 1 frame apart (at 30fps = 0.033s gap)
        let captions = vec![
            make_caption("00:00:01:00", "A"),
            make_caption("00:00:01:01", "B"),
        ];
        let issues = CaptionQcChecker::check(&captions, 30.0);
        assert!(
            issues
                .iter()
                .any(|i| i.issue_type == CaptionIssueType::GapTooShort),
            "Should flag 1-frame gap"
        );
    }

    #[test]
    fn test_timecode_parsing() {
        // 00:00:01:15 at 30fps = 1 + 15/30 = 1.5 seconds
        let secs = CaptionQcChecker::timecode_to_secs("00:00:01:15", 30.0);
        assert!((secs - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_caption_stats_empty() {
        let stats = CaptionStats::compute(&[], 60.0);
        assert_eq!(stats.caption_count, 0);
        assert_eq!(stats.avg_words_per_min, 0.0);
        assert_eq!(stats.coverage_pct, 0.0);
    }

    #[test]
    fn test_caption_stats_basic() {
        let captions = vec![
            make_caption("00:00:01:00", "Hello world"), // 2 words, 11 chars
            make_caption("00:00:05:00", "Foo bar baz"), // 3 words, 11 chars
        ];
        let stats = CaptionStats::compute(&captions, 120.0); // 2 minutes
        assert_eq!(stats.caption_count, 2);
        assert!(stats.avg_words_per_min > 0.0);
        assert!(stats.coverage_pct >= 0.0 && stats.coverage_pct <= 100.0);
    }

    #[test]
    fn test_caption_stats_coverage_pct_clamped() {
        // Short duration, many chars → coverage could exceed 100%
        let captions = vec![make_caption("00:00:00:00", &"a".repeat(1000))];
        let stats = CaptionStats::compute(&captions, 1.0); // 1 second
        assert!(stats.coverage_pct <= 100.0);
    }
}
