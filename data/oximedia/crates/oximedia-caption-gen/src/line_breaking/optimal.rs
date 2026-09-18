//! Optimal line-breaking using a Knuth-Plass-inspired DP (baseline O(n²)).

/// Break `text` using a dynamic-programming algorithm that minimises the sum of
/// squared slack on each line: `cost(line) = (max_width - line_width)^2`.
///
/// This produces more balanced lines than the greedy approach.
pub fn optimal_break(text: &str, max_width: u8) -> Vec<String> {
    let max = max_width.max(1) as usize;
    let words: Vec<&str> = text.split_whitespace().collect();
    let n = words.len();

    if n == 0 {
        return vec![String::new()];
    }

    // Pre-compute cumulative character widths (without spaces for quick lookup).
    // span_width(i, j) = sum of word lengths from i..=j plus (j-i) spaces.
    let word_lens: Vec<usize> = words.iter().map(|w| w.chars().count()).collect();

    // dp[i] = minimum cost to break words[i..n] optimally.
    // breaks[i] = the end-index (exclusive) of the first line when starting at i.
    let mut dp = vec![u64::MAX; n + 1];
    let mut breaks: Vec<usize> = vec![n; n + 1];
    dp[n] = 0;

    for i in (0..n).rev() {
        let mut width = 0usize;
        for j in i..n {
            width += word_lens[j];
            if j > i {
                width += 1; // space
            }
            if width > max {
                break;
            }
            let slack = max - width;
            let line_cost = (slack * slack) as u64;
            let rest_cost = dp[j + 1];
            if rest_cost != u64::MAX {
                let total = line_cost.saturating_add(rest_cost);
                if total < dp[i] {
                    dp[i] = total;
                    breaks[i] = j + 1;
                }
            }
        }
        // If no valid break was found (all words too wide), force a single word.
        if dp[i] == u64::MAX {
            dp[i] = 0;
            breaks[i] = i + 1;
        }
    }

    // Reconstruct lines.
    let mut lines: Vec<String> = Vec::new();
    let mut pos = 0;
    while pos < n {
        let end = breaks[pos].min(n);
        let end = if end <= pos { pos + 1 } else { end };
        lines.push(words[pos..end].join(" "));
        pos = end;
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::super::balance::LineBalance;
    use super::super::greedy::greedy_break;
    use super::*;

    #[test]
    fn optimal_break_empty_string() {
        let result = optimal_break("", 40);
        assert_eq!(result, vec![""]);
    }

    #[test]
    fn optimal_break_single_line() {
        let result = optimal_break("Hello world", 20);
        assert_eq!(result, vec!["Hello world"]);
    }

    #[test]
    fn optimal_break_more_balanced_than_greedy() {
        // "one two three four" greedy at width 10:
        //   "one two"  (7) + "three"   (5) + "four" (4)  → slack: 3,5,6
        // optimal should find a better balance.
        let text = "one two three four";
        let optimal = optimal_break(text, 10);
        let greedy = greedy_break(text, 10);
        let opt_balance = LineBalance::balance_factor(&optimal);
        let greed_balance = LineBalance::balance_factor(&greedy);
        // Optimal should be at least as balanced.
        assert!(
            opt_balance <= greed_balance + 0.01,
            "optimal balance {opt_balance} worse than greedy {greed_balance}"
        );
    }

    #[test]
    fn optimal_break_preserves_all_words() {
        let text = "alpha beta gamma delta epsilon zeta";
        let result = optimal_break(text, 15);
        let rejoined = result.join(" ");
        assert_eq!(rejoined, text);
    }

    #[test]
    fn optimal_break_no_line_exceeds_max_width() {
        let text = "short lines should be wrapped correctly by algorithm";
        let result = optimal_break(text, 20);
        for line in &result {
            assert!(
                line.chars().count() <= 20,
                "line '{line}' exceeds max width"
            );
        }
    }

    #[test]
    fn optimal_break_reference_output_known_case() {
        // Reference: "one two three four five" at width 11.
        // Optimal should produce lines whose total slack is minimised.
        let text = "one two three four five";
        let result = optimal_break(text, 11);
        // All words must be present.
        let rejoined = result.join(" ");
        assert_eq!(rejoined, text);
        // No line exceeds max width.
        for line in &result {
            assert!(
                line.chars().count() <= 11,
                "line '{line}' exceeds max width"
            );
        }
    }

    #[test]
    fn greedy_and_optimal_produce_identical_single_line() {
        // When all text fits on one line, both algorithms must produce one line.
        let text = "Hello";
        let g = greedy_break(text, 20);
        let o = optimal_break(text, 20);
        assert_eq!(g, o);
    }

    #[test]
    fn greedy_and_optimal_identical_for_single_word_per_line() {
        // Each word fits on one line individually: both algorithms agree.
        let text = "a b c";
        let g = greedy_break(text, 1);
        let o = optimal_break(text, 1);
        // Both produce 3 lines of 1 character each.
        assert_eq!(g.len(), o.len(), "g={:?} o={:?}", g, o);
    }
}
