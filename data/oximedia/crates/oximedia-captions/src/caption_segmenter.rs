#![allow(dead_code)]
//! Intelligent caption text segmentation.
//!
//! Splits long caption text into appropriately sized lines for display,
//! respecting linguistic boundaries (clause breaks, conjunctions, punctuation)
//! and maximum line-length constraints (character count and pixel width estimates).

/// Default maximum characters per line for broadcast captions.
const DEFAULT_MAX_CHARS_PER_LINE: usize = 42;

/// Default maximum number of lines per caption block.
const DEFAULT_MAX_LINES: usize = 2;

/// Default target balance ratio (0.0 = all on first line, 1.0 = perfectly balanced).
const DEFAULT_BALANCE_TARGET: f64 = 0.85;

/// Configuration for caption segmentation.
#[derive(Debug, Clone)]
pub struct SegmenterConfig {
    /// Maximum characters per line.
    pub max_chars_per_line: usize,
    /// Maximum number of lines per caption block.
    pub max_lines: usize,
    /// Target balance between lines (0.0 to 1.0).
    pub balance_target: f64,
    /// Prefer breaks after punctuation characters.
    pub prefer_punctuation_breaks: bool,
    /// Prefer breaks before conjunctions (and, but, or, etc.).
    pub prefer_conjunction_breaks: bool,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            max_chars_per_line: DEFAULT_MAX_CHARS_PER_LINE,
            max_lines: DEFAULT_MAX_LINES,
            balance_target: DEFAULT_BALANCE_TARGET,
            prefer_punctuation_breaks: true,
            prefer_conjunction_breaks: true,
        }
    }
}

/// Result of segmenting a caption text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentResult {
    /// The segmented lines.
    pub lines: Vec<String>,
    /// Whether the text had to be truncated due to constraints.
    pub was_truncated: bool,
}

impl SegmentResult {
    /// Total character count across all lines.
    #[must_use]
    pub fn total_chars(&self) -> usize {
        self.lines.iter().map(|l| l.chars().count()).sum()
    }

    /// Number of lines produced.
    #[must_use]
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// The longest line's character count.
    #[must_use]
    pub fn max_line_length(&self) -> usize {
        self.lines
            .iter()
            .map(|l| l.chars().count())
            .max()
            .unwrap_or(0)
    }

    /// Line balance ratio: shortest / longest (1.0 = perfectly balanced).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn balance_ratio(&self) -> f64 {
        if self.lines.len() < 2 {
            return 1.0;
        }
        let lengths: Vec<usize> = self.lines.iter().map(|l| l.chars().count()).collect();
        let min_len = *lengths.iter().min().unwrap_or(&0);
        let max_len = *lengths.iter().max().unwrap_or(&1);
        if max_len == 0 {
            return 1.0;
        }
        min_len as f64 / max_len as f64
    }
}

/// Check whether a word is a conjunction suitable as a break point.
fn is_conjunction(word: &str) -> bool {
    matches!(
        word.to_lowercase().as_str(),
        "and"
            | "but"
            | "or"
            | "nor"
            | "yet"
            | "so"
            | "for"
            | "because"
            | "although"
            | "while"
            | "when"
            | "where"
            | "which"
            | "that"
            | "if"
            | "then"
            | "than"
    )
}

/// Check if a character is a punctuation mark suitable for line breaks.
fn is_break_punctuation(c: char) -> bool {
    matches!(c, ',' | ';' | ':' | '.' | '!' | '?' | '-' | '\u{2014}')
}

/// Score a potential break point. Higher is better.
fn score_break(words_before: &[&str], words_after: &[&str], config: &SegmenterConfig) -> f64 {
    if words_before.is_empty() || words_after.is_empty() {
        return 0.0;
    }

    let mut score = 1.0_f64;

    // Prefer balanced lines
    let len_before: usize = words_before
        .iter()
        .map(|w| w.chars().count())
        .sum::<usize>()
        + words_before.len().saturating_sub(1);
    let len_after: usize = words_after.iter().map(|w| w.chars().count()).sum::<usize>()
        + words_after.len().saturating_sub(1);

    #[allow(clippy::cast_precision_loss)]
    let total = (len_before + len_after) as f64;
    if total > 0.0 {
        #[allow(clippy::cast_precision_loss)]
        let balance = {
            let shorter = len_before.min(len_after) as f64;
            let longer = len_before.max(len_after) as f64;
            if longer > 0.0 {
                shorter / longer
            } else {
                1.0
            }
        };
        score += balance * config.balance_target * 10.0;
    }

    // Prefer breaks after punctuation
    if config.prefer_punctuation_breaks {
        if let Some(last_word) = words_before.last() {
            if let Some(last_char) = last_word.chars().last() {
                if is_break_punctuation(last_char) {
                    score += 5.0;
                }
            }
        }
    }

    // Prefer breaks before conjunctions
    if config.prefer_conjunction_breaks {
        if let Some(first_word) = words_after.first() {
            if is_conjunction(first_word) {
                score += 3.0;
            }
        }
    }

    score
}

/// Segment a text into lines respecting the given configuration.
#[must_use]
pub fn segment_text(text: &str, config: &SegmenterConfig) -> SegmentResult {
    let text = text.trim();
    if text.is_empty() {
        return SegmentResult {
            lines: vec![String::new()],
            was_truncated: false,
        };
    }

    // If it fits on one line, no segmentation needed
    if text.chars().count() <= config.max_chars_per_line {
        return SegmentResult {
            lines: vec![text.to_string()],
            was_truncated: false,
        };
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return SegmentResult {
            lines: vec![String::new()],
            was_truncated: false,
        };
    }

    if config.max_lines < 2 {
        // Must fit on one line — truncate
        let truncated = truncate_to_chars(text, config.max_chars_per_line);
        return SegmentResult {
            lines: vec![truncated],
            was_truncated: text.chars().count() > config.max_chars_per_line,
        };
    }

    // Find the best break point for 2-line segmentation
    let mut best_score = f64::NEG_INFINITY;
    let mut best_break = words.len() / 2;

    for i in 1..words.len() {
        let before = &words[..i];
        let after = &words[i..];

        let line1_len: usize = before.iter().map(|w| w.chars().count()).sum::<usize>()
            + before.len().saturating_sub(1);

        // Skip if line 1 exceeds max length
        if line1_len > config.max_chars_per_line {
            continue;
        }

        let line2_len: usize =
            after.iter().map(|w| w.chars().count()).sum::<usize>() + after.len().saturating_sub(1);

        // Skip if line 2 exceeds max length
        if line2_len > config.max_chars_per_line {
            continue;
        }

        let s = score_break(before, after, config);
        if s > best_score {
            best_score = s;
            best_break = i;
        }
    }

    let line1: String = words[..best_break].join(" ");
    let line2: String = words[best_break..].join(" ");

    let mut was_truncated = false;
    let line1 = if line1.chars().count() > config.max_chars_per_line {
        was_truncated = true;
        truncate_to_chars(&line1, config.max_chars_per_line)
    } else {
        line1
    };
    let line2 = if line2.chars().count() > config.max_chars_per_line {
        was_truncated = true;
        truncate_to_chars(&line2, config.max_chars_per_line)
    } else {
        line2
    };

    let mut lines = vec![line1];
    if !line2.is_empty() {
        lines.push(line2);
    }

    SegmentResult {
        lines,
        was_truncated,
    }
}

/// Truncate a string to at most `max_chars` characters, appending an ellipsis
/// if truncation occurs.
fn truncate_to_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    if max_chars < 4 {
        return text.chars().take(max_chars).collect();
    }
    let mut result: String = text.chars().take(max_chars - 3).collect();
    result.push_str("...");
    result
}

/// Split a list of words into segments of at most `max_chars` total length each,
/// respecting word boundaries.
#[must_use]
pub fn split_into_chunks(words: &[&str], max_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();

    for &word in words {
        let word_len = word.chars().count();
        let cur_len = current.chars().count();

        if cur_len == 0 {
            current = word.to_string();
        } else if cur_len + 1 + word_len <= max_chars {
            current.push(' ');
            current.push_str(word);
        } else {
            chunks.push(current);
            current = word.to_string();
        }
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}

/// Estimate the number of caption blocks needed for a given text.
#[must_use]
pub fn estimate_blocks(text: &str, config: &SegmenterConfig) -> usize {
    let char_count = text.chars().count();
    let chars_per_block = config.max_chars_per_line * config.max_lines;
    if chars_per_block == 0 {
        return 0;
    }
    char_count.div_ceil(chars_per_block)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_text_no_split() {
        let result = segment_text("Hello world", &SegmenterConfig::default());
        assert_eq!(result.lines.len(), 1);
        assert_eq!(result.lines[0], "Hello world");
        assert!(!result.was_truncated);
    }

    #[test]
    fn test_long_text_splits_into_two() {
        let text = "The quick brown fox jumps over the lazy dog and runs away fast";
        let result = segment_text(text, &SegmenterConfig::default());
        assert_eq!(result.lines.len(), 2);
        assert!(!result.was_truncated);
    }

    #[test]
    fn test_max_line_length_respected() {
        let config = SegmenterConfig {
            max_chars_per_line: 30,
            ..Default::default()
        };
        let text = "The quick brown fox jumps over the lazy dog";
        let result = segment_text(text, &config);
        for line in &result.lines {
            assert!(line.chars().count() <= 30, "Line too long: {line}");
        }
    }

    #[test]
    fn test_empty_text() {
        let result = segment_text("", &SegmenterConfig::default());
        assert_eq!(result.lines.len(), 1);
        assert_eq!(result.lines[0], "");
    }

    #[test]
    fn test_single_line_max() {
        let config = SegmenterConfig {
            max_lines: 1,
            max_chars_per_line: 10,
            ..Default::default()
        };
        let result = segment_text("This is a very long sentence that overflows", &config);
        assert_eq!(result.lines.len(), 1);
        assert!(result.was_truncated);
        assert!(result.lines[0].chars().count() <= 10);
    }

    #[test]
    fn test_balance_ratio_perfect() {
        let result = SegmentResult {
            lines: vec!["Hello world".to_string(), "Hello world".to_string()],
            was_truncated: false,
        };
        assert!((result.balance_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_balance_ratio_unbalanced() {
        let result = SegmentResult {
            lines: vec!["Hi".to_string(), "Hello world!".to_string()],
            was_truncated: false,
        };
        let ratio = result.balance_ratio();
        assert!(ratio < 0.5);
    }

    #[test]
    fn test_conjunction_preference() {
        let config = SegmenterConfig {
            max_chars_per_line: 30,
            prefer_conjunction_breaks: true,
            ..Default::default()
        };
        let text = "I went to the store and I bought some apples";
        let result = segment_text(text, &config);
        assert_eq!(result.lines.len(), 2);
        // The second line should start with "and"
        assert!(
            result.lines[1].starts_with("and"),
            "Expected break before 'and', got: {:?}",
            result.lines
        );
    }

    #[test]
    fn test_punctuation_preference() {
        let config = SegmenterConfig {
            max_chars_per_line: 35,
            prefer_punctuation_breaks: true,
            ..Default::default()
        };
        let text = "Welcome to our show, today we discuss climate";
        let result = segment_text(text, &config);
        assert_eq!(result.lines.len(), 2);
        // Should break after comma
        assert!(
            result.lines[0].ends_with(','),
            "Expected break after comma, got: {:?}",
            result.lines
        );
    }

    #[test]
    fn test_split_into_chunks() {
        let words = vec!["hello", "world", "foo", "bar", "baz"];
        let chunks = split_into_chunks(&words, 12);
        for chunk in &chunks {
            assert!(chunk.chars().count() <= 12, "Chunk too long: {chunk}");
        }
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_estimate_blocks() {
        let config = SegmenterConfig {
            max_chars_per_line: 42,
            max_lines: 2,
            ..Default::default()
        };
        // 84 chars per block, 200 chars => ceil(200/84) = 3
        let text = "a".repeat(200);
        assert_eq!(estimate_blocks(&text, &config), 3);
    }

    #[test]
    fn test_truncate_to_chars() {
        let t = truncate_to_chars("Hello, world!", 8);
        assert_eq!(t, "Hello...");
        assert!(t.chars().count() <= 8);
    }

    #[test]
    fn test_total_chars() {
        let result = SegmentResult {
            lines: vec!["Hello".to_string(), "World".to_string()],
            was_truncated: false,
        };
        assert_eq!(result.total_chars(), 10);
    }

    #[test]
    fn test_is_conjunction() {
        assert!(is_conjunction("and"));
        assert!(is_conjunction("AND"));
        assert!(is_conjunction("but"));
        assert!(!is_conjunction("hello"));
    }
}
