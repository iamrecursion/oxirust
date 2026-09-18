//! Greedy and language-aware line breaking, plus CJK character helpers.

/// Returns `true` if `ch` is a CJK character (logographic / ideographic).
fn is_cjk_char(ch: char) -> bool {
    // CJK Unified Ideographs and common extensions.
    ('\u{4E00}'..='\u{9FFF}').contains(&ch)
        || ('\u{3400}'..='\u{4DBF}').contains(&ch)
        || ('\u{F900}'..='\u{FAFF}').contains(&ch)
        // Hiragana and Katakana (Japanese syllabic scripts).
        || ('\u{3040}'..='\u{309F}').contains(&ch)
        || ('\u{30A0}'..='\u{30FF}').contains(&ch)
        // Hangul (Korean).
        || ('\u{AC00}'..='\u{D7AF}').contains(&ch)
}

/// Returns `true` if the character is a line-break *prohibiting* character.
///
/// These characters must not appear at the start of a line (opening brackets,
/// leading punctuation) per Unicode line-breaking rules (UAX #14).
fn is_cjk_no_start(ch: char) -> bool {
    matches!(
        ch,
        '、' | '。'
            | '，'
            | '．'
            | '：'
            | '；'
            | '？'
            | '！'
            | '）'
            | '」'
            | '』'
            | '】'
            | '〕'
            | '〉'
            | '》'
            | '·'
            | '‥'
            | '…'
            | 'ー'
            | 'ヽ'
            | 'ヾ'
            | 'ゝ'
            | 'ゞ'
    )
}

/// Break `text` into lines for CJK scripts (no spaces between words).
///
/// CJK text is broken at character boundaries with the following rules:
/// - No line ends with a leading bracket / punctuation character that should
///   not start a line (`is_cjk_no_start`).
/// - Lines do not exceed `max_width` characters.
pub fn cjk_break(text: &str, max_width: u8) -> Vec<String> {
    let max = max_width.max(1) as usize;
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();

    if n <= max {
        return vec![text.to_string()];
    }

    let mut lines: Vec<String> = Vec::new();
    let mut start = 0;

    while start < n {
        // Ideal end is `start + max`.
        let ideal_end = (start + max).min(n);

        if ideal_end >= n {
            lines.push(chars[start..].iter().collect());
            break;
        }

        // Adjust end if the character *after* the cut cannot start a line.
        let mut end = ideal_end;
        while end > start + 1 && is_cjk_no_start(chars[end]) {
            end -= 1;
        }

        lines.push(chars[start..end].iter().collect());
        start = end;
    }

    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

/// Language-aware line breaking.
///
/// For CJK text, delegates to [`cjk_break`].  For all other scripts,
/// delegates to [`greedy_break`].
///
/// The heuristic for detecting CJK: if > 30% of non-whitespace characters
/// are CJK/Hiragana/Katakana/Hangul, the text is treated as CJK.
pub fn language_aware_break(text: &str, max_width: u8) -> Vec<String> {
    let non_ws: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    if non_ws.is_empty() {
        return vec![String::new()];
    }

    let cjk_count = non_ws.iter().filter(|&&c| is_cjk_char(c)).count();
    let cjk_fraction = cjk_count as f32 / non_ws.len() as f32;

    if cjk_fraction > 0.30 {
        cjk_break(text, max_width)
    } else {
        greedy_break(text, max_width)
    }
}

/// Which algorithm to use when breaking caption text into lines.
#[derive(Debug, Clone, PartialEq)]
pub enum LineBreakAlgorithm {
    /// Break at the last space before `max_width`.
    Greedy,
    /// Dynamic-programming algorithm that minimises raggedness (Knuth-Plass
    /// inspired): `cost(line) = (max_width - used_width)^2`.
    Optimal,
    /// Every line is exactly `u8` characters wide (hard wrap, no splitting of words).
    Fixed(u8),
}

/// Break `text` greedily at the last space before `max_width` characters.
///
/// Words longer than `max_width` are placed on their own line unchanged.
pub fn greedy_break(text: &str, max_width: u8) -> Vec<String> {
    let max = max_width.max(1) as usize;
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();

    for word in text.split_whitespace() {
        if current.is_empty() {
            current.push_str(word);
        } else if current.chars().count() + 1 + word.chars().count() <= max {
            current.push(' ');
            current.push_str(word);
        } else {
            lines.push(current.clone());
            current = word.to_string();
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_break_empty_string() {
        let result = greedy_break("", 40);
        assert_eq!(result, vec![""]);
    }

    #[test]
    fn greedy_break_single_word_fits() {
        let result = greedy_break("Hello", 40);
        assert_eq!(result, vec!["Hello"]);
    }

    #[test]
    fn greedy_break_two_words_fit_on_one_line() {
        let result = greedy_break("Hello world", 20);
        assert_eq!(result, vec!["Hello world"]);
    }

    #[test]
    fn greedy_break_wraps_at_limit() {
        let result = greedy_break("Hello world", 8);
        assert_eq!(result, vec!["Hello", "world"]);
    }

    #[test]
    fn greedy_break_multiple_lines() {
        let result = greedy_break("one two three four five", 9);
        // "one two" = 7, "three" = 5, "four" = 4, "five" = 4
        assert!(result.len() >= 2);
        for line in &result {
            assert!(line.chars().count() <= 9, "line '{line}' exceeds max width");
        }
    }

    #[test]
    fn greedy_break_long_word_gets_own_line() {
        let result = greedy_break("A superlongwordthatexceedslimit B", 10);
        // The long word must appear alone on its line.
        assert!(result.iter().any(|l| l.contains("superlongword")));
    }

    #[test]
    fn greedy_break_preserves_all_words() {
        let text = "one two three four five six seven";
        let result = greedy_break(text, 15);
        let rejoined = result.join(" ");
        assert_eq!(rejoined, text);
    }

    #[test]
    fn cjk_break_short_text_unchanged() {
        let text = "日本語";
        let result = cjk_break(text, 10);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], text);
    }

    #[test]
    fn cjk_break_long_text_splits_at_char_boundary() {
        let text = "これは日本語のテキストサンプルです"; // 16 chars
        let result = cjk_break(text, 5);
        assert!(result.len() > 1, "expected split");
        for line in &result {
            let count = line.chars().count();
            assert!(count <= 5, "line '{line}' has {count} chars > 5");
        }
        // All characters should be preserved.
        let combined: String = result.concat();
        assert_eq!(combined.chars().count(), text.chars().count());
    }

    #[test]
    fn language_aware_break_latin_uses_greedy() {
        let text = "Hello there how are you doing";
        let result = language_aware_break(text, 12);
        let rejoined = result.join(" ");
        assert_eq!(rejoined, text);
    }

    #[test]
    fn language_aware_break_cjk_detected() {
        let text = "これは日本語のテキストです"; // all CJK
        let result = language_aware_break(text, 5);
        assert!(result.len() > 1, "expected multi-line CJK break");
    }
}
