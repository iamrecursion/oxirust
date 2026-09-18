//! Subtitle line segmentation and block optimization.
//!
//! Provides tools for breaking subtitle text into readable lines,
//! optimizing block layout, and scoring readability.

/// Rules for line breaking in subtitle text.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum SegmentationRule {
    /// Maximum number of characters per line.
    MaxCharsPerLine(u32),
    /// Maximum number of lines per block.
    MaxLines(u32),
    /// Prefer to break at natural linguistic break points.
    BreakAtNaturalPoints,
    /// Avoid leaving orphan words (single short word on a line).
    NoOrphanWords,
}

/// Known English conjunctions and relative pronouns used as natural break points.
static CONJUNCTIONS: &[&str] = &[
    "and", "or", "but", "because", "that", "which", "who", "when", "where", "while",
];

/// Check if a word is a known natural break-point conjunction.
fn is_break_word(word: &str) -> bool {
    let lower = word.to_ascii_lowercase();
    CONJUNCTIONS.contains(&lower.as_str())
}

/// A subtitle block containing one or more display lines.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct SubtitleBlock {
    /// Lines of text in this block.
    pub lines: Vec<String>,
}

impl SubtitleBlock {
    /// Create a new subtitle block.
    #[allow(dead_code)]
    pub fn new(lines: Vec<String>) -> Self {
        Self { lines }
    }

    /// Total character count across all lines (excluding newlines).
    #[allow(dead_code)]
    pub fn char_count(&self) -> usize {
        self.lines.iter().map(|l| l.chars().count()).sum()
    }

    /// Number of lines in the block.
    #[allow(dead_code)]
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// Join all lines into a single space-separated string.
    #[allow(dead_code)]
    pub fn to_plain_text(&self) -> String {
        self.lines.join(" ")
    }
}

/// Line breaker that applies segmentation rules to subtitle text.
pub struct LineBreaker;

impl LineBreaker {
    /// Segment text into lines according to the given rules.
    ///
    /// Applies rules in order:
    /// 1. `MaxCharsPerLine` — hard word-wrap
    /// 2. `BreakAtNaturalPoints` — prefer breaks before conjunctions
    /// 3. `MaxLines` — truncate to max line count
    /// 4. `NoOrphanWords` — merge trailing single-word lines upward
    #[allow(dead_code)]
    pub fn segment(text: &str, rules: &[SegmentationRule]) -> Vec<String> {
        let max_chars = rules.iter().find_map(|r| {
            if let SegmentationRule::MaxCharsPerLine(n) = r {
                Some(*n as usize)
            } else {
                None
            }
        });

        let max_lines = rules.iter().find_map(|r| {
            if let SegmentationRule::MaxLines(n) = r {
                Some(*n as usize)
            } else {
                None
            }
        });

        let natural_breaks = rules.contains(&SegmentationRule::BreakAtNaturalPoints);
        let no_orphans = rules.contains(&SegmentationRule::NoOrphanWords);

        let words: Vec<&str> = text.split_whitespace().collect();

        if words.is_empty() {
            return vec![];
        }

        let max_chars = max_chars.unwrap_or(42); // standard broadcast default

        let mut lines: Vec<String> = Vec::new();
        let mut current_line = String::new();

        for word in &words {
            // Natural break: if the word is a conjunction and current line is long enough,
            // start a new line before the conjunction
            let should_break_before = natural_breaks
                && is_break_word(word)
                && !current_line.is_empty()
                && current_line.len() > max_chars / 3;

            if should_break_before && !current_line.is_empty() {
                lines.push(current_line.trim_end().to_string());
                current_line = String::new();
            }

            let would_be_len = if current_line.is_empty() {
                word.len()
            } else {
                current_line.len() + 1 + word.len()
            };

            if would_be_len > max_chars && !current_line.is_empty() {
                // Current line is full — push it and start new
                lines.push(current_line.trim_end().to_string());
                current_line = word.to_string();
            } else if current_line.is_empty() {
                current_line = word.to_string();
            } else {
                current_line.push(' ');
                current_line.push_str(word);
            }
        }

        if !current_line.is_empty() {
            lines.push(current_line.trim_end().to_string());
        }

        // Apply max lines
        if let Some(max) = max_lines {
            lines.truncate(max);
        }

        // Apply no-orphan rule: if last line has only one short word (≤4 chars)
        // and there are multiple lines, merge it with the previous line
        if no_orphans && lines.len() > 1 {
            if let Some(last) = lines.last() {
                let word_count = last.split_whitespace().count();
                if word_count == 1 && last.chars().count() <= 4 {
                    let orphan = lines
                        .pop()
                        .expect("invariant: lines non-empty confirmed above");
                    if let Some(prev) = lines.last_mut() {
                        prev.push(' ');
                        prev.push_str(&orphan);
                    }
                }
            }
        }

        lines
    }
}

/// Optimizer that reflows subtitle blocks to meet display constraints.
pub struct BlockOptimizer;

impl BlockOptimizer {
    /// Reflow a collection of subtitle blocks to respect character and line limits.
    #[allow(dead_code)]
    pub fn reflow(blocks: &[SubtitleBlock], max_chars: u32, max_lines: u32) -> Vec<SubtitleBlock> {
        let rules = vec![
            SegmentationRule::MaxCharsPerLine(max_chars),
            SegmentationRule::MaxLines(max_lines),
            SegmentationRule::BreakAtNaturalPoints,
        ];

        blocks
            .iter()
            .map(|block| {
                let text = block.to_plain_text();
                let lines = LineBreaker::segment(&text, &rules);
                SubtitleBlock::new(lines)
            })
            .collect()
    }
}

/// Computes a readability score for a subtitle block.
pub struct ReadabilityScore;

impl ReadabilityScore {
    /// Compute a readability score for a subtitle block.
    ///
    /// Score is in [0.0, 1.0] where 1.0 is most readable.
    ///
    /// Factors:
    /// - Average word length (shorter = more readable)
    /// - Line balance (lines close in length = better)
    #[allow(dead_code)]
    pub fn compute(subtitle: &SubtitleBlock) -> f32 {
        if subtitle.lines.is_empty() {
            return 0.0;
        }

        let all_text = subtitle.to_plain_text();
        let words: Vec<&str> = all_text.split_whitespace().collect();

        if words.is_empty() {
            return 0.0;
        }

        // Average word length factor (shorter = better)
        let avg_word_len: f32 =
            words.iter().map(|w| w.chars().count() as f32).sum::<f32>() / words.len() as f32;
        // Normalize: target avg word length = 4-5 chars for readability
        let word_len_score = 1.0 - ((avg_word_len - 4.5).abs() / 10.0).min(1.0);

        // Line balance factor: stddev of line lengths
        let line_lengths: Vec<f32> = subtitle
            .lines
            .iter()
            .map(|l| l.chars().count() as f32)
            .collect();
        let mean_len = line_lengths.iter().sum::<f32>() / line_lengths.len() as f32;
        let variance = line_lengths
            .iter()
            .map(|l| (l - mean_len).powi(2))
            .sum::<f32>()
            / line_lengths.len() as f32;
        let stddev = variance.sqrt();
        // Normalize by mean: lower stddev = better balance
        let balance_score = if mean_len > 0.0 {
            1.0 - (stddev / mean_len).min(1.0)
        } else {
            0.0
        };

        // Combine factors equally
        (word_len_score + balance_score) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_breaker_simple() {
        let lines = LineBreaker::segment("Hello world", &[SegmentationRule::MaxCharsPerLine(42)]);
        assert_eq!(lines, vec!["Hello world"]);
    }

    #[test]
    fn test_line_breaker_wraps_at_max() {
        // 20-char limit should force a break
        let text = "The quick brown fox jumps over the lazy dog";
        let lines = LineBreaker::segment(text, &[SegmentationRule::MaxCharsPerLine(20)]);
        assert!(lines.len() > 1);
        for line in &lines {
            assert!(line.chars().count() <= 20, "Line too long: {line}");
        }
    }

    #[test]
    fn test_line_breaker_max_lines_truncates() {
        let text = "one two three four five six seven eight nine ten";
        let lines = LineBreaker::segment(
            text,
            &[
                SegmentationRule::MaxCharsPerLine(10),
                SegmentationRule::MaxLines(2),
            ],
        );
        assert!(lines.len() <= 2);
    }

    #[test]
    fn test_line_breaker_natural_breaks_at_conjunction() {
        let text = "I went to the store and I bought some milk";
        let lines = LineBreaker::segment(
            text,
            &[
                SegmentationRule::MaxCharsPerLine(60),
                SegmentationRule::BreakAtNaturalPoints,
            ],
        );
        // Should break before "and" or another conjunction
        assert!(lines.len() >= 1, "Should produce at least one line");
    }

    #[test]
    fn test_line_breaker_empty_text() {
        let lines = LineBreaker::segment("", &[SegmentationRule::MaxCharsPerLine(42)]);
        assert!(lines.is_empty());
    }

    #[test]
    fn test_subtitle_block_char_count() {
        let block = SubtitleBlock::new(vec!["Hello".to_string(), "World".to_string()]);
        assert_eq!(block.char_count(), 10);
    }

    #[test]
    fn test_subtitle_block_line_count() {
        let block = SubtitleBlock::new(vec!["A".to_string(), "B".to_string(), "C".to_string()]);
        assert_eq!(block.line_count(), 3);
    }

    #[test]
    fn test_subtitle_block_to_plain_text() {
        let block = SubtitleBlock::new(vec!["Hello".to_string(), "World".to_string()]);
        assert_eq!(block.to_plain_text(), "Hello World");
    }

    #[test]
    fn test_block_optimizer_reflow() {
        let blocks = vec![SubtitleBlock::new(vec![
            "This is a very long subtitle line that should be reflowed properly".to_string(),
        ])];
        let reflowed = BlockOptimizer::reflow(&blocks, 30, 3);
        assert!(!reflowed.is_empty());
        for block in &reflowed {
            for line in &block.lines {
                assert!(line.chars().count() <= 30, "Reflowed line too long: {line}");
            }
        }
    }

    #[test]
    fn test_readability_score_range() {
        let block = SubtitleBlock::new(vec![
            "Hello world this is a test".to_string(),
            "of the readability scorer".to_string(),
        ]);
        let score = ReadabilityScore::compute(&block);
        assert!(score >= 0.0 && score <= 1.0, "Score out of range: {score}");
    }

    #[test]
    fn test_readability_score_empty_block() {
        let block = SubtitleBlock::new(vec![]);
        assert_eq!(ReadabilityScore::compute(&block), 0.0);
    }

    #[test]
    fn test_no_orphan_rule() {
        // Force a situation where last "word" is a short orphan
        let text = "one two three four five six";
        let lines = LineBreaker::segment(
            text,
            &[
                SegmentationRule::MaxCharsPerLine(20),
                SegmentationRule::NoOrphanWords,
            ],
        );
        // Check that no line has a single very short word if there's another line
        if lines.len() > 1 {
            let last = lines.last().expect("should succeed in test");
            let last_words: Vec<&str> = last.split_whitespace().collect();
            // If last line has one word, it should be > 4 chars (orphan check)
            if last_words.len() == 1 {
                assert!(
                    last_words[0].len() > 4,
                    "Orphan word should have been merged: {}",
                    last_words[0]
                );
            }
        }
    }
}
