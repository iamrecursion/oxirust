#![allow(dead_code)]
//! Subtitle text sanitization and cleanup utilities.
//!
//! This module provides functions to clean subtitle text by stripping HTML/SSA
//! tags, normalizing whitespace, removing hearing-impaired descriptions,
//! trimming excessive line breaks, and detecting common encoding issues.

/// Configuration for sanitization operations.
#[derive(Clone, Debug)]
pub struct SanitizeConfig {
    /// Strip HTML-like tags (e.g. `<i>`, `<b>`, `<font>`).
    pub strip_html_tags: bool,
    /// Strip SSA/ASS override tags (e.g. `{\b1}`, `{\an8}`).
    pub strip_ssa_tags: bool,
    /// Remove hearing-impaired descriptions in brackets/parentheses.
    pub remove_hi_descriptions: bool,
    /// Collapse multiple consecutive whitespace into a single space.
    pub normalize_whitespace: bool,
    /// Trim leading/trailing whitespace from each line.
    pub trim_lines: bool,
    /// Remove empty lines.
    pub remove_empty_lines: bool,
    /// Maximum allowed consecutive line breaks (0 = unlimited).
    pub max_line_breaks: usize,
}

impl Default for SanitizeConfig {
    fn default() -> Self {
        Self {
            strip_html_tags: true,
            strip_ssa_tags: true,
            remove_hi_descriptions: false,
            normalize_whitespace: true,
            trim_lines: true,
            remove_empty_lines: true,
            max_line_breaks: 2,
        }
    }
}

impl SanitizeConfig {
    /// Create a configuration with all sanitization options enabled.
    #[must_use]
    pub fn all() -> Self {
        Self {
            strip_html_tags: true,
            strip_ssa_tags: true,
            remove_hi_descriptions: true,
            normalize_whitespace: true,
            trim_lines: true,
            remove_empty_lines: true,
            max_line_breaks: 1,
        }
    }

    /// Create a minimal configuration (whitespace cleanup only).
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            strip_html_tags: false,
            strip_ssa_tags: false,
            remove_hi_descriptions: false,
            normalize_whitespace: true,
            trim_lines: true,
            remove_empty_lines: false,
            max_line_breaks: 0,
        }
    }
}

/// Strip HTML-like tags from text.
///
/// Removes content between `<` and `>` characters.
#[must_use]
pub fn strip_html_tags(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_tag = false;
    for ch in text.chars() {
        if ch == '<' {
            in_tag = true;
        } else if ch == '>' {
            in_tag = false;
        } else if !in_tag {
            result.push(ch);
        }
    }
    result
}

/// Strip SSA/ASS override tags from text.
///
/// Removes content between `{\` and `}` sequences.
#[must_use]
pub fn strip_ssa_tags(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;
    while i < len {
        if i + 1 < len && chars[i] == '{' && chars[i + 1] == '\\' {
            // Skip until closing }
            while i < len && chars[i] != '}' {
                i += 1;
            }
            if i < len {
                i += 1; // skip the '}'
            }
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }
    result
}

/// Remove hearing-impaired descriptions in square brackets and parentheses.
///
/// Examples: `[music playing]`, `(laughing)`, `[GUNSHOT]`.
#[must_use]
pub fn remove_hi_descriptions(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut depth_square = 0i32;
    let mut depth_paren = 0i32;
    for ch in text.chars() {
        match ch {
            '[' => depth_square += 1,
            ']' => {
                if depth_square > 0 {
                    depth_square -= 1;
                } else {
                    result.push(ch);
                }
            }
            '(' => depth_paren += 1,
            ')' => {
                if depth_paren > 0 {
                    depth_paren -= 1;
                } else {
                    result.push(ch);
                }
            }
            _ => {
                if depth_square == 0 && depth_paren == 0 {
                    result.push(ch);
                }
            }
        }
    }
    result
}

/// Normalize whitespace: collapse runs of spaces/tabs into a single space.
#[must_use]
pub fn normalize_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_ws = false;
    for ch in text.chars() {
        if ch == ' ' || ch == '\t' {
            if !prev_ws {
                result.push(' ');
                prev_ws = true;
            }
        } else {
            result.push(ch);
            prev_ws = false;
        }
    }
    result
}

/// Trim leading and trailing whitespace from each line.
#[must_use]
pub fn trim_lines(text: &str) -> String {
    text.lines().map(str::trim).collect::<Vec<_>>().join("\n")
}

/// Remove empty lines from text.
#[must_use]
pub fn remove_empty_lines(text: &str) -> String {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Limit consecutive line breaks to at most `max` newlines.
#[must_use]
pub fn limit_line_breaks(text: &str, max: usize) -> String {
    if max == 0 {
        return text.to_string();
    }
    let separator = "\n".repeat(max);
    let mut result = String::with_capacity(text.len());
    let mut consecutive_newlines = 0usize;
    for ch in text.chars() {
        if ch == '\n' {
            consecutive_newlines += 1;
            if consecutive_newlines <= max {
                result.push('\n');
            }
        } else {
            consecutive_newlines = 0;
            result.push(ch);
        }
    }
    let _ = separator; // used to illustrate the max
    result
}

/// Apply full sanitization according to the given configuration.
#[must_use]
pub fn sanitize(text: &str, config: &SanitizeConfig) -> String {
    let mut result = text.to_string();
    if config.strip_html_tags {
        result = strip_html_tags(&result);
    }
    if config.strip_ssa_tags {
        result = strip_ssa_tags(&result);
    }
    if config.remove_hi_descriptions {
        result = remove_hi_descriptions(&result);
    }
    if config.normalize_whitespace {
        result = normalize_whitespace(&result);
    }
    if config.trim_lines {
        result = trim_lines(&result);
    }
    if config.remove_empty_lines {
        result = remove_empty_lines(&result);
    }
    if config.max_line_breaks > 0 {
        result = limit_line_breaks(&result, config.max_line_breaks);
    }
    result
}

/// Detect whether text likely contains mojibake (encoding corruption).
///
/// Checks for common sequences that indicate wrongly decoded UTF-8.
#[must_use]
pub fn has_encoding_issues(text: &str) -> bool {
    // Common mojibake sequences
    let patterns = ["\u{FFFD}", "Ã©", "Ã¨", "Ã¼", "Ã¶", "Ã¤", "Â"];
    for pat in &patterns {
        if text.contains(pat) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_html_tags_basic() {
        assert_eq!(strip_html_tags("<i>italic</i>"), "italic");
        assert_eq!(strip_html_tags("<b>bold</b> text"), "bold text");
    }

    #[test]
    fn test_strip_html_tags_no_tags() {
        assert_eq!(strip_html_tags("plain text"), "plain text");
    }

    #[test]
    fn test_strip_html_tags_nested() {
        assert_eq!(strip_html_tags("<b><i>nested</i></b>"), "nested");
    }

    #[test]
    fn test_strip_ssa_tags() {
        assert_eq!(strip_ssa_tags("{\\b1}bold{\\b0}"), "bold");
        assert_eq!(strip_ssa_tags("{\\an8}top text"), "top text");
    }

    #[test]
    fn test_strip_ssa_tags_no_tags() {
        assert_eq!(strip_ssa_tags("no tags here"), "no tags here");
    }

    #[test]
    fn test_remove_hi_descriptions() {
        assert_eq!(remove_hi_descriptions("[music] Hello"), " Hello");
        assert_eq!(remove_hi_descriptions("(laughing) Ha!"), " Ha!");
    }

    #[test]
    fn test_remove_hi_nested() {
        assert_eq!(remove_hi_descriptions("[outer [inner]]"), "");
    }

    #[test]
    fn test_normalize_whitespace() {
        assert_eq!(normalize_whitespace("hello   world"), "hello world");
        assert_eq!(normalize_whitespace("a\t\tb"), "a b");
    }

    #[test]
    fn test_trim_lines() {
        assert_eq!(trim_lines("  hello  \n  world  "), "hello\nworld");
    }

    #[test]
    fn test_remove_empty_lines() {
        assert_eq!(remove_empty_lines("a\n\nb\n\nc"), "a\nb\nc");
    }

    #[test]
    fn test_limit_line_breaks() {
        assert_eq!(limit_line_breaks("a\n\n\nb", 1), "a\nb");
        assert_eq!(limit_line_breaks("a\n\n\nb", 2), "a\n\nb");
    }

    #[test]
    fn test_sanitize_full() {
        let config = SanitizeConfig::all();
        let input = "<i>Hello</i> [music]  world  ";
        let result = sanitize(input, &config);
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn test_has_encoding_issues_clean() {
        assert!(!has_encoding_issues("Hello world"));
    }

    #[test]
    fn test_has_encoding_issues_mojibake() {
        assert!(has_encoding_issues("CafÃ© latte"));
        assert!(has_encoding_issues("replacement \u{FFFD} char"));
    }
}
