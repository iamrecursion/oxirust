//! Automatic line-breaking for subtitle text.
//!
//! Provides character-count–based wrapping, word-wrap, and simple
//! soft-hyphenation for long subtitle lines.

#![allow(dead_code)]

/// Configuration for the line-breaking algorithm.
#[derive(Clone, Debug, PartialEq)]
pub struct LineBreakConfig {
    /// Maximum number of characters per line (inclusive).
    pub max_chars: usize,
    /// Maximum number of output lines.  `None` means unlimited.
    pub max_lines: Option<usize>,
    /// When `true`, hard hyphens may be inserted when a word is longer than
    /// `max_chars`.
    pub allow_hyphenation: bool,
    /// String used to separate output lines.
    pub line_separator: String,
}

impl Default for LineBreakConfig {
    fn default() -> Self {
        Self {
            max_chars: 42,
            max_lines: None,
            allow_hyphenation: false,
            line_separator: "\n".to_string(),
        }
    }
}

impl LineBreakConfig {
    /// Create a config with a custom `max_chars` limit.
    #[must_use]
    pub fn with_max_chars(mut self, max_chars: usize) -> Self {
        self.max_chars = max_chars;
        self
    }

    /// Enable or disable hyphenation.
    #[must_use]
    pub fn with_hyphenation(mut self, allow: bool) -> Self {
        self.allow_hyphenation = allow;
        self
    }

    /// Set the maximum number of output lines.
    #[must_use]
    pub fn with_max_lines(mut self, max_lines: usize) -> Self {
        self.max_lines = Some(max_lines);
        self
    }
}

/// Split `text` into lines that respect `config.max_chars`.
///
/// Words are never split unless `config.allow_hyphenation` is `true` and the
/// word is longer than `max_chars`.
#[must_use]
pub fn wrap_text(text: &str, config: &LineBreakConfig) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();

    for word in text.split_whitespace() {
        // Word itself exceeds max_chars – optionally hyphenate it.
        if word.chars().count() > config.max_chars {
            // Flush the current line first.
            if !current.is_empty() {
                lines.push(current.clone());
                current.clear();
            }
            if config.allow_hyphenation {
                let chunks = split_with_hyphen(word, config.max_chars);
                let last = chunks.len().saturating_sub(1);
                for (i, chunk) in chunks.into_iter().enumerate() {
                    if i == last {
                        current = chunk;
                    } else {
                        lines.push(chunk);
                    }
                }
            } else {
                // Push oversized word as-is.
                lines.push(word.to_string());
            }
            continue;
        }

        let candidate = if current.is_empty() {
            word.to_string()
        } else {
            format!("{current} {word}")
        };

        if candidate.chars().count() <= config.max_chars {
            current = candidate;
        } else {
            lines.push(current.clone());
            current = word.to_string();
        }
    }

    if !current.is_empty() {
        lines.push(current);
    }

    if let Some(max) = config.max_lines {
        lines.truncate(max);
    }

    lines
}

/// Join wrapped lines back into a single string using `config.line_separator`.
#[must_use]
pub fn wrap_and_join(text: &str, config: &LineBreakConfig) -> String {
    wrap_text(text, config).join(&config.line_separator)
}

/// Split a single word into chunks of `max_chars - 1` chars plus a hyphen,
/// except for the final chunk.
fn split_with_hyphen(word: &str, max_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let chars: Vec<char> = word.chars().collect();
    let chunk_size = max_chars.saturating_sub(1).max(1);
    let mut start = 0;

    while start < chars.len() {
        let end = (start + chunk_size).min(chars.len());
        let is_last = end == chars.len();
        let chunk: String = chars[start..end].iter().collect();
        if is_last {
            chunks.push(chunk);
        } else {
            chunks.push(format!("{chunk}-"));
        }
        start = end;
    }

    chunks
}

/// Count the number of characters in the longest line of `text`
/// (splitting on `\n`).
#[must_use]
pub fn max_line_length(text: &str) -> usize {
    text.lines().map(|l| l.chars().count()).max().unwrap_or(0)
}

/// Returns `true` if every line in `text` fits within `max_chars`.
#[must_use]
pub fn fits_within(text: &str, max_chars: usize) -> bool {
    text.lines().all(|l| l.chars().count() <= max_chars)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrap_short_text_unchanged() {
        let cfg = LineBreakConfig::default().with_max_chars(50);
        let result = wrap_text("Hello world", &cfg);
        assert_eq!(result, vec!["Hello world"]);
    }

    #[test]
    fn test_wrap_splits_long_line() {
        let cfg = LineBreakConfig::default().with_max_chars(10);
        let text = "Hello world foo bar";
        let lines = wrap_text(text, &cfg);
        assert!(lines.len() > 1);
        for line in &lines {
            assert!(line.chars().count() <= 10, "line too long: {line:?}");
        }
    }

    #[test]
    fn test_wrap_empty_string() {
        let cfg = LineBreakConfig::default();
        let result = wrap_text("", &cfg);
        assert!(result.is_empty());
    }

    #[test]
    fn test_wrap_single_word_no_hyphenation() {
        let cfg = LineBreakConfig::default().with_max_chars(5);
        let result = wrap_text("Hello", &cfg);
        assert_eq!(result, vec!["Hello"]);
    }

    #[test]
    fn test_wrap_oversized_word_no_hyphenation() {
        let cfg = LineBreakConfig::default().with_max_chars(5);
        let result = wrap_text("Supercalifragilistic", &cfg);
        // Pushed as-is when hyphenation disabled
        assert_eq!(result, vec!["Supercalifragilistic"]);
    }

    #[test]
    fn test_wrap_oversized_word_with_hyphenation() {
        let cfg = LineBreakConfig::default()
            .with_max_chars(5)
            .with_hyphenation(true);
        let result = wrap_text("Supercalifragilistic", &cfg);
        // Each chunk (except last) ends with hyphen and is at most 5 chars
        for line in &result {
            assert!(line.chars().count() <= 5, "chunk too long: {line:?}");
        }
    }

    #[test]
    fn test_wrap_max_lines_truncation() {
        let cfg = LineBreakConfig::default()
            .with_max_chars(5)
            .with_max_lines(2);
        let result = wrap_text("one two three four five six", &cfg);
        assert!(result.len() <= 2);
    }

    #[test]
    fn test_wrap_and_join_uses_separator() {
        let cfg = LineBreakConfig {
            max_chars: 10,
            line_separator: " | ".to_string(),
            ..Default::default()
        };
        let joined = wrap_and_join("Hello world foo bar", &cfg);
        assert!(joined.contains(" | "));
    }

    #[test]
    fn test_max_line_length_single() {
        assert_eq!(max_line_length("Hello"), 5);
    }

    #[test]
    fn test_max_line_length_multiline() {
        assert_eq!(max_line_length("Hello\nWorld!\nHi"), 6);
    }

    #[test]
    fn test_max_line_length_empty() {
        assert_eq!(max_line_length(""), 0);
    }

    #[test]
    fn test_fits_within_true() {
        assert!(fits_within("Hello\nWorld", 10));
    }

    #[test]
    fn test_fits_within_false() {
        assert!(!fits_within("Hello\nThis is a very long line indeed", 20));
    }

    #[test]
    fn test_split_with_hyphen_exact_multiple() {
        let chunks = split_with_hyphen("ABCDEFGH", 4);
        // max_chars=4 → chunk_size=3
        // "ABC-", "DEF-", "GH"
        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].ends_with('-'));
        assert!(chunks[1].ends_with('-'));
        assert!(!chunks[2].ends_with('-'));
    }

    #[test]
    fn test_default_config_max_chars() {
        let cfg = LineBreakConfig::default();
        assert_eq!(cfg.max_chars, 42);
    }
}
