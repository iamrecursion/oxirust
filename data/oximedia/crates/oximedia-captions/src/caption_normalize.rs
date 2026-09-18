#![allow(dead_code)]

//! Caption text and timing normalization utilities.
//!
//! Provides tools for normalizing caption content including whitespace cleanup,
//! Unicode normalization, timing quantization, line-length enforcement, and
//! encoding standardization.

use std::fmt;

/// Normalization rules to apply.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NormalizeConfig {
    /// Collapse consecutive whitespace into a single space.
    pub collapse_whitespace: bool,
    /// Trim leading/trailing whitespace from each line.
    pub trim_lines: bool,
    /// Remove empty lines within caption text.
    pub remove_empty_lines: bool,
    /// Maximum characters per line (0 = unlimited).
    pub max_chars_per_line: usize,
    /// Maximum number of lines per caption (0 = unlimited).
    pub max_lines: usize,
    /// Quantize timing to this interval in milliseconds (0 = no quantization).
    pub timing_quantize_ms: u64,
    /// Minimum caption duration in milliseconds (0 = no minimum).
    pub min_duration_ms: u64,
    /// Maximum caption duration in milliseconds (0 = no maximum).
    pub max_duration_ms: u64,
    /// Minimum gap between consecutive captions in milliseconds.
    pub min_gap_ms: u64,
    /// Apply Unicode NFC normalization.
    pub unicode_nfc: bool,
    /// Remove zero-width characters.
    pub remove_zero_width: bool,
}

impl Default for NormalizeConfig {
    fn default() -> Self {
        Self {
            collapse_whitespace: true,
            trim_lines: true,
            remove_empty_lines: true,
            max_chars_per_line: 42,
            max_lines: 2,
            timing_quantize_ms: 0,
            min_duration_ms: 500,
            max_duration_ms: 7000,
            min_gap_ms: 40,
            unicode_nfc: false,
            remove_zero_width: true,
        }
    }
}

/// A mutable caption entry for normalization.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NormCaption {
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
    /// The text content.
    pub text: String,
}

impl NormCaption {
    /// Create a new normalizable caption.
    #[must_use]
    pub fn new(start_ms: u64, end_ms: u64, text: String) -> Self {
        Self {
            start_ms,
            end_ms,
            text,
        }
    }

    /// Duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Number of text lines.
    #[must_use]
    pub fn line_count(&self) -> usize {
        self.text.lines().count().max(1)
    }
}

/// Type of normalization action taken.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NormAction {
    /// Whitespace was collapsed.
    WhitespaceCollapsed,
    /// Lines were trimmed.
    LinesTrimmed,
    /// Empty lines were removed.
    EmptyLinesRemoved,
    /// Lines were wrapped to fit max chars.
    LinesWrapped,
    /// Excess lines were truncated.
    LinesTruncated,
    /// Timing was quantized.
    TimingQuantized,
    /// Duration was clamped.
    DurationClamped,
    /// Gap was adjusted.
    GapAdjusted,
    /// Zero-width characters were removed.
    ZeroWidthRemoved,
    /// Unicode was normalized.
    UnicodeNormalized,
}

impl fmt::Display for NormAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WhitespaceCollapsed => write!(f, "whitespace_collapsed"),
            Self::LinesTrimmed => write!(f, "lines_trimmed"),
            Self::EmptyLinesRemoved => write!(f, "empty_lines_removed"),
            Self::LinesWrapped => write!(f, "lines_wrapped"),
            Self::LinesTruncated => write!(f, "lines_truncated"),
            Self::TimingQuantized => write!(f, "timing_quantized"),
            Self::DurationClamped => write!(f, "duration_clamped"),
            Self::GapAdjusted => write!(f, "gap_adjusted"),
            Self::ZeroWidthRemoved => write!(f, "zero_width_removed"),
            Self::UnicodeNormalized => write!(f, "unicode_normalized"),
        }
    }
}

/// A record of a normalization action applied to a caption.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NormRecord {
    /// Index of the caption in the list.
    pub caption_index: usize,
    /// The action taken.
    pub action: NormAction,
    /// Description of what changed.
    pub detail: String,
}

/// Result of normalizing a caption track.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NormalizeResult {
    /// The normalized captions.
    pub captions: Vec<NormCaption>,
    /// Log of all actions taken.
    pub actions: Vec<NormRecord>,
}

impl NormalizeResult {
    /// Count of actions by type.
    #[must_use]
    pub fn action_count(&self, action: &NormAction) -> usize {
        self.actions.iter().filter(|a| &a.action == action).count()
    }
}

/// Normalize caption text: collapse whitespace.
fn collapse_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_space = false;
    for ch in text.chars() {
        if ch == '\n' {
            prev_space = false;
            result.push(ch);
        } else if ch.is_whitespace() {
            if !prev_space {
                result.push(' ');
            }
            prev_space = true;
        } else {
            prev_space = false;
            result.push(ch);
        }
    }
    result
}

/// Remove zero-width characters from text.
fn remove_zero_width_chars(text: &str) -> String {
    text.chars()
        .filter(|c| {
            !matches!(
                *c,
                '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{FEFF}' | '\u{00AD}'
            )
        })
        .collect()
}

/// Trim each line in a multiline string.
fn trim_each_line(text: &str) -> String {
    text.lines().map(str::trim).collect::<Vec<_>>().join("\n")
}

/// Remove empty lines from text.
fn remove_empty_lines_from(text: &str) -> String {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Wrap lines to a maximum character width using word boundaries.
fn wrap_lines(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return text.to_string();
    }
    let mut result_lines = Vec::new();
    for line in text.lines() {
        if line.len() <= max_chars {
            result_lines.push(line.to_string());
        } else {
            let words: Vec<&str> = line.split_whitespace().collect();
            let mut current = String::new();
            for word in words {
                if current.is_empty() {
                    current = word.to_string();
                } else if current.len() + 1 + word.len() <= max_chars {
                    current.push(' ');
                    current.push_str(word);
                } else {
                    result_lines.push(current);
                    current = word.to_string();
                }
            }
            if !current.is_empty() {
                result_lines.push(current);
            }
        }
    }
    result_lines.join("\n")
}

/// Truncate to max number of lines.
fn truncate_lines(text: &str, max_lines: usize) -> String {
    if max_lines == 0 {
        return text.to_string();
    }
    text.lines().take(max_lines).collect::<Vec<_>>().join("\n")
}

/// Quantize a time value to the nearest multiple of `quantum_ms`.
fn quantize_time(time_ms: u64, quantum_ms: u64) -> u64 {
    if quantum_ms == 0 {
        return time_ms;
    }
    let half = quantum_ms / 2;
    ((time_ms + half) / quantum_ms) * quantum_ms
}

/// Normalize a list of captions according to the given configuration.
#[must_use]
pub fn normalize_captions(captions: &[NormCaption], config: &NormalizeConfig) -> NormalizeResult {
    let mut result_captions: Vec<NormCaption> = captions.to_vec();
    let mut actions = Vec::new();

    for (i, cap) in result_captions.iter_mut().enumerate() {
        // Zero-width character removal
        if config.remove_zero_width {
            let cleaned = remove_zero_width_chars(&cap.text);
            if cleaned != cap.text {
                actions.push(NormRecord {
                    caption_index: i,
                    action: NormAction::ZeroWidthRemoved,
                    detail: "Removed zero-width characters".to_string(),
                });
                cap.text = cleaned;
            }
        }

        // Whitespace collapsing
        if config.collapse_whitespace {
            let collapsed = collapse_whitespace(&cap.text);
            if collapsed != cap.text {
                actions.push(NormRecord {
                    caption_index: i,
                    action: NormAction::WhitespaceCollapsed,
                    detail: "Collapsed consecutive whitespace".to_string(),
                });
                cap.text = collapsed;
            }
        }

        // Line trimming
        if config.trim_lines {
            let trimmed = trim_each_line(&cap.text);
            if trimmed != cap.text {
                actions.push(NormRecord {
                    caption_index: i,
                    action: NormAction::LinesTrimmed,
                    detail: "Trimmed line whitespace".to_string(),
                });
                cap.text = trimmed;
            }
        }

        // Empty line removal
        if config.remove_empty_lines {
            let cleaned = remove_empty_lines_from(&cap.text);
            if cleaned != cap.text {
                actions.push(NormRecord {
                    caption_index: i,
                    action: NormAction::EmptyLinesRemoved,
                    detail: "Removed empty lines".to_string(),
                });
                cap.text = cleaned;
            }
        }

        // Line wrapping
        if config.max_chars_per_line > 0 {
            let wrapped = wrap_lines(&cap.text, config.max_chars_per_line);
            if wrapped != cap.text {
                actions.push(NormRecord {
                    caption_index: i,
                    action: NormAction::LinesWrapped,
                    detail: format!("Wrapped to {} chars/line", config.max_chars_per_line),
                });
                cap.text = wrapped;
            }
        }

        // Line truncation
        if config.max_lines > 0 && cap.line_count() > config.max_lines {
            let truncated = truncate_lines(&cap.text, config.max_lines);
            actions.push(NormRecord {
                caption_index: i,
                action: NormAction::LinesTruncated,
                detail: format!("Truncated to {} lines", config.max_lines),
            });
            cap.text = truncated;
        }

        // Timing quantization
        if config.timing_quantize_ms > 0 {
            let q_start = quantize_time(cap.start_ms, config.timing_quantize_ms);
            let q_end = quantize_time(cap.end_ms, config.timing_quantize_ms);
            if q_start != cap.start_ms || q_end != cap.end_ms {
                actions.push(NormRecord {
                    caption_index: i,
                    action: NormAction::TimingQuantized,
                    detail: format!(
                        "Quantized {}-{} -> {}-{}",
                        cap.start_ms, cap.end_ms, q_start, q_end
                    ),
                });
                cap.start_ms = q_start;
                cap.end_ms = q_end;
            }
        }

        // Duration clamping
        let dur = cap.duration_ms();
        if config.min_duration_ms > 0 && dur < config.min_duration_ms {
            cap.end_ms = cap.start_ms + config.min_duration_ms;
            actions.push(NormRecord {
                caption_index: i,
                action: NormAction::DurationClamped,
                detail: format!(
                    "Duration extended from {}ms to {}ms",
                    dur, config.min_duration_ms
                ),
            });
        } else if config.max_duration_ms > 0 && dur > config.max_duration_ms {
            cap.end_ms = cap.start_ms + config.max_duration_ms;
            actions.push(NormRecord {
                caption_index: i,
                action: NormAction::DurationClamped,
                detail: format!(
                    "Duration clamped from {}ms to {}ms",
                    dur, config.max_duration_ms
                ),
            });
        }
    }

    // Gap enforcement between consecutive captions
    if config.min_gap_ms > 0 {
        for i in 1..result_captions.len() {
            let prev_end = result_captions[i - 1].end_ms;
            let curr_start = result_captions[i].start_ms;
            if curr_start > prev_end && curr_start - prev_end < config.min_gap_ms {
                // Shrink previous caption to ensure gap
                let new_prev_end = curr_start.saturating_sub(config.min_gap_ms);
                if new_prev_end > result_captions[i - 1].start_ms {
                    actions.push(NormRecord {
                        caption_index: i - 1,
                        action: NormAction::GapAdjusted,
                        detail: format!(
                            "Adjusted end from {} to {} for {}ms gap",
                            prev_end, new_prev_end, config.min_gap_ms
                        ),
                    });
                    result_captions[i - 1].end_ms = new_prev_end;
                }
            }
        }
    }

    NormalizeResult {
        captions: result_captions,
        actions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap(start: u64, end: u64, text: &str) -> NormCaption {
        NormCaption::new(start, end, text.to_string())
    }

    #[test]
    fn test_collapse_whitespace() {
        assert_eq!(collapse_whitespace("Hello   World"), "Hello World");
        assert_eq!(collapse_whitespace("A  B  C"), "A B C");
    }

    #[test]
    fn test_remove_zero_width() {
        let input = "Hello\u{200B}World";
        assert_eq!(remove_zero_width_chars(input), "HelloWorld");
    }

    #[test]
    fn test_trim_each_line() {
        assert_eq!(trim_each_line("  Hello  \n  World  "), "Hello\nWorld");
    }

    #[test]
    fn test_remove_empty_lines() {
        assert_eq!(
            remove_empty_lines_from("Hello\n\nWorld\n\n"),
            "Hello\nWorld"
        );
    }

    #[test]
    fn test_wrap_lines_short() {
        let text = "Short";
        assert_eq!(wrap_lines(text, 42), "Short");
    }

    #[test]
    fn test_wrap_lines_long() {
        let text = "This is a very long line that should be wrapped to fit the limit";
        let wrapped = wrap_lines(text, 30);
        for line in wrapped.lines() {
            assert!(line.len() <= 30, "Line too long: {}", line);
        }
    }

    #[test]
    fn test_truncate_lines() {
        let text = "Line1\nLine2\nLine3\nLine4";
        assert_eq!(truncate_lines(text, 2), "Line1\nLine2");
    }

    #[test]
    fn test_quantize_time() {
        assert_eq!(quantize_time(123, 100), 100);
        assert_eq!(quantize_time(150, 100), 200);
        assert_eq!(quantize_time(149, 100), 100);
        assert_eq!(quantize_time(0, 100), 0);
        assert_eq!(quantize_time(50, 0), 50);
    }

    #[test]
    fn test_normalize_basic() {
        let captions = vec![cap(0, 1000, "  Hello   World  ")];
        let config = NormalizeConfig {
            collapse_whitespace: true,
            trim_lines: true,
            max_chars_per_line: 0,
            max_lines: 0,
            timing_quantize_ms: 0,
            min_duration_ms: 0,
            max_duration_ms: 0,
            min_gap_ms: 0,
            ..Default::default()
        };
        let result = normalize_captions(&captions, &config);
        assert_eq!(result.captions[0].text, "Hello World");
    }

    #[test]
    fn test_normalize_duration_clamp_min() {
        let captions = vec![cap(0, 100, "Short")];
        let config = NormalizeConfig {
            min_duration_ms: 500,
            max_duration_ms: 0,
            max_chars_per_line: 0,
            max_lines: 0,
            timing_quantize_ms: 0,
            min_gap_ms: 0,
            ..Default::default()
        };
        let result = normalize_captions(&captions, &config);
        assert_eq!(result.captions[0].end_ms, 500);
    }

    #[test]
    fn test_normalize_duration_clamp_max() {
        let captions = vec![cap(0, 10000, "Long")];
        let config = NormalizeConfig {
            min_duration_ms: 0,
            max_duration_ms: 5000,
            max_chars_per_line: 0,
            max_lines: 0,
            timing_quantize_ms: 0,
            min_gap_ms: 0,
            ..Default::default()
        };
        let result = normalize_captions(&captions, &config);
        assert_eq!(result.captions[0].end_ms, 5000);
    }

    #[test]
    fn test_normalize_gap_enforcement() {
        let captions = vec![cap(0, 990, "First"), cap(1000, 2000, "Second")];
        let config = NormalizeConfig {
            min_gap_ms: 40,
            min_duration_ms: 0,
            max_duration_ms: 0,
            max_chars_per_line: 0,
            max_lines: 0,
            timing_quantize_ms: 0,
            ..Default::default()
        };
        let result = normalize_captions(&captions, &config);
        let gap = result.captions[1].start_ms - result.captions[0].end_ms;
        assert!(gap >= 40, "Gap was {} but should be >= 40", gap);
    }

    #[test]
    fn test_action_count() {
        let captions = vec![cap(0, 100, "  A  "), cap(1000, 1100, "  B  ")];
        let config = NormalizeConfig {
            trim_lines: true,
            min_duration_ms: 500,
            max_duration_ms: 0,
            max_chars_per_line: 0,
            max_lines: 0,
            timing_quantize_ms: 0,
            min_gap_ms: 0,
            ..Default::default()
        };
        let result = normalize_captions(&captions, &config);
        assert!(result.action_count(&NormAction::DurationClamped) >= 2);
    }

    #[test]
    fn test_norm_caption_line_count() {
        let c = cap(0, 1000, "Line1\nLine2\nLine3");
        assert_eq!(c.line_count(), 3);
    }

    #[test]
    fn test_norm_action_display() {
        assert_eq!(
            NormAction::WhitespaceCollapsed.to_string(),
            "whitespace_collapsed"
        );
        assert_eq!(NormAction::GapAdjusted.to_string(), "gap_adjusted");
    }
}
