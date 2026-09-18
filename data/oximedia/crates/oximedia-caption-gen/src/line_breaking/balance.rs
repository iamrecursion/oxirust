//! Reading-speed helpers and line-balance utilities.

use super::optimal::optimal_break;

/// Compute reading speed in characters per second.
///
/// Returns 0.0 if `duration_ms` is zero.
pub fn compute_cps(text: &str, duration_ms: u64) -> f32 {
    if duration_ms == 0 {
        return 0.0;
    }
    let char_count = text.chars().count() as f32;
    char_count / (duration_ms as f32 / 1000.0)
}

/// Returns `true` when the reading speed of `text` over `duration_ms` does not
/// exceed `max_cps`.
pub fn reading_speed_ok(text: &str, duration_ms: u64, max_cps: f32) -> bool {
    compute_cps(text, duration_ms) <= max_cps
}

/// Compute the minimum display duration required to read `text` at `max_cps`,
/// but never shorter than `min_ms`.
///
/// Formula: `max(min_ms, ceil(char_count * 1000 / max_cps))`.
pub fn adjust_duration_for_reading(text: &str, min_ms: u32, max_cps: f32) -> u32 {
    if max_cps <= 0.0 {
        return min_ms;
    }
    let char_count = text.chars().count() as f32;
    let required_ms = (char_count * 1000.0 / max_cps).ceil() as u32;
    required_ms.max(min_ms)
}

/// Statistics and scoring for caption line balance.
pub struct LineBalance;

impl LineBalance {
    /// Compute a balance factor in [0.0, 1.0]:
    /// - `0.0` = perfectly balanced (all lines same length).
    /// - `1.0` = maximally unbalanced.
    ///
    /// Uses the standard deviation of line lengths normalised by the mean.
    /// Returns `0.0` for 0 or 1 lines.
    pub fn balance_factor(lines: &[String]) -> f32 {
        if lines.len() <= 1 {
            return 0.0;
        }
        let lengths: Vec<f32> = lines.iter().map(|l| l.chars().count() as f32).collect();
        let mean = lengths.iter().sum::<f32>() / lengths.len() as f32;
        if mean < 1e-6 {
            return 0.0;
        }
        let variance =
            lengths.iter().map(|&l| (l - mean).powi(2)).sum::<f32>() / lengths.len() as f32;
        let std_dev = variance.sqrt();
        // Normalise by mean so the result is dimensionless; cap at 1.0.
        (std_dev / mean).min(1.0)
    }
}

/// Redistribute words across lines to minimise [`LineBalance::balance_factor`].
///
/// Internally calls [`optimal_break`] with a `max_width` derived from the
/// average line length, then returns the result if it is better balanced than
/// the input, otherwise returns the input unchanged.
pub fn rebalance_lines(lines: Vec<String>, max_width: u8) -> Vec<String> {
    if lines.len() <= 1 {
        return lines;
    }

    let original_factor = LineBalance::balance_factor(&lines);
    let combined = lines.join(" ");
    let rebroken = optimal_break(&combined, max_width);
    let new_factor = LineBalance::balance_factor(&rebroken);

    if new_factor < original_factor {
        rebroken
    } else {
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_cps_basic() {
        // 10 chars over 2000ms = 5 cps.
        let cps = compute_cps("Hello wrld", 2000);
        assert!((cps - 5.0).abs() < 0.01, "expected ~5.0, got {cps}");
    }

    #[test]
    fn compute_cps_zero_duration_returns_zero() {
        assert_eq!(compute_cps("Hello", 0), 0.0);
    }

    #[test]
    fn compute_cps_empty_text() {
        assert_eq!(compute_cps("", 1000), 0.0);
    }

    #[test]
    fn reading_speed_ok_slow_enough() {
        // 5 chars at 1 second = 5 cps < 17 → ok.
        assert!(reading_speed_ok("Hello", 1000, 17.0));
    }

    #[test]
    fn reading_speed_ok_too_fast() {
        // 50 chars at 1 second = 50 cps > 17.
        let long_text = "A".repeat(50);
        assert!(!reading_speed_ok(&long_text, 1000, 17.0));
    }

    #[test]
    fn adjust_duration_respects_min() {
        // 5 chars at 17 cps needs ~295ms, but min is 1000ms.
        let d = adjust_duration_for_reading("Hello", 1000, 17.0);
        assert_eq!(d, 1000);
    }

    #[test]
    fn adjust_duration_extends_for_long_text() {
        // 170 chars at 17 cps needs 10000ms; min is 1000ms.
        let text = "A".repeat(170);
        let d = adjust_duration_for_reading(&text, 1000, 17.0);
        assert_eq!(d, 10000);
    }

    #[test]
    fn adjust_duration_zero_max_cps_returns_min() {
        let d = adjust_duration_for_reading("Hello world", 500, 0.0);
        assert_eq!(d, 500);
    }

    #[test]
    fn balance_factor_single_line_is_zero() {
        let lines = vec!["Hello world".to_string()];
        assert_eq!(LineBalance::balance_factor(&lines), 0.0);
    }

    #[test]
    fn balance_factor_equal_lines_is_zero() {
        let lines = vec!["Hello".to_string(), "World".to_string()];
        assert!((LineBalance::balance_factor(&lines)).abs() < 1e-5);
    }

    #[test]
    fn balance_factor_unequal_lines_nonzero() {
        let lines = vec!["A".to_string(), "A much longer line here".to_string()];
        assert!(LineBalance::balance_factor(&lines) > 0.0);
    }

    #[test]
    fn balance_factor_empty_lines_is_zero() {
        assert_eq!(LineBalance::balance_factor(&[]), 0.0);
    }

    #[test]
    fn rebalance_lines_single_line_unchanged() {
        let lines = vec!["Hello world".to_string()];
        let result = rebalance_lines(lines.clone(), 40);
        assert_eq!(result, lines);
    }

    #[test]
    fn rebalance_lines_produces_at_most_same_balance_factor() {
        let lines = vec![
            "Hi".to_string(),
            "This is a much longer second line here".to_string(),
        ];
        let original_factor = LineBalance::balance_factor(&lines);
        let result = rebalance_lines(lines, 40);
        let new_factor = LineBalance::balance_factor(&result);
        assert!(new_factor <= original_factor + 0.01);
    }

    #[test]
    fn rebalance_lines_preserves_all_words() {
        let lines = vec!["one two".to_string(), "three four five six".to_string()];
        let original_words: std::collections::HashSet<String> = lines
            .iter()
            .flat_map(|l| l.split_whitespace())
            .map(|w| w.to_string())
            .collect();
        let result = rebalance_lines(lines, 20);
        let result_words: std::collections::HashSet<String> = result
            .iter()
            .flat_map(|l| l.split_whitespace())
            .map(|w| w.to_string())
            .collect();
        assert_eq!(original_words, result_words);
    }
}
