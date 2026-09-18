//! Fuzzy matching for keyword and clip name searches.
//!
//! This module implements Levenshtein-distance-based fuzzy matching for
//! clip names and keywords. Scores are normalised to the range `[0.0, 1.0]`
//! where `1.0` means an exact match and `0.0` means completely dissimilar.

use crate::clip::Clip;

/// Fuzzy text matcher using normalised Levenshtein distance.
#[derive(Debug, Clone, Default)]
pub struct FuzzyMatcher;

impl FuzzyMatcher {
    /// Creates a new `FuzzyMatcher`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Computes a similarity score between `pattern` and `text`.
    ///
    /// The score is `1.0 - (edit_distance / max_len)` where `max_len` is
    /// `max(pattern.len(), text.len())`.  Both strings are compared
    /// case-insensitively after lowercasing.
    ///
    /// Returns `1.0` if both strings are empty.
    #[must_use]
    pub fn score(pattern: &str, text: &str) -> f32 {
        let p = pattern.to_lowercase();
        let t = text.to_lowercase();

        let p_len = p.chars().count();
        let t_len = t.chars().count();

        if p_len == 0 && t_len == 0 {
            return 1.0;
        }

        let max_len = p_len.max(t_len);
        let dist = levenshtein(&p, &t);

        1.0 - (dist as f32 / max_len as f32)
    }

    /// Returns matched clips sorted by descending score (highest first).
    ///
    /// Only clips whose best field score (name, keywords) meets `min_score`
    /// are included.
    #[must_use]
    pub fn match_clips<'a>(
        clips: &'a [Clip],
        pattern: &str,
        min_score: f32,
    ) -> Vec<(f32, &'a Clip)> {
        let mut results: Vec<(f32, &Clip)> = clips
            .iter()
            .filter_map(|clip| {
                let best = clip_best_score(clip, pattern);
                if best >= min_score {
                    Some((best, clip))
                } else {
                    None
                }
            })
            .collect();

        // Stable descending sort so equal-score clips preserve insertion order.
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Convenience: checks whether `pattern` fuzzy-matches `text` at or above
    /// `min_score`.
    #[must_use]
    pub fn is_match(pattern: &str, text: &str, min_score: f32) -> bool {
        Self::score(pattern, text) >= min_score
    }
}

/// Returns the highest fuzzy score across all searchable fields of a clip.
fn clip_best_score(clip: &Clip, pattern: &str) -> f32 {
    let mut best = FuzzyMatcher::score(pattern, &clip.name);

    for kw in &clip.keywords {
        let s = FuzzyMatcher::score(pattern, kw);
        if s > best {
            best = s;
        }
    }

    if let Some(desc) = &clip.description {
        let s = FuzzyMatcher::score(pattern, desc);
        if s > best {
            best = s;
        }
    }

    best
}

/// Compute the Levenshtein edit distance between two strings.
///
/// Uses the Wagner-Fischer DP algorithm with a single rolling row to keep
/// memory usage proportional to `O(min(m, n))`.
fn levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Use a and b so that the row vector tracks the shorter dimension.
    let (row_src, col_src) = if m <= n {
        (&a_chars[..], &b_chars[..])
    } else {
        (&b_chars[..], &a_chars[..])
    };

    let row_len = row_src.len();
    let col_len = col_src.len();

    // prev[j] = distance(row_src[0..i], col_src[0..j])
    let mut prev: Vec<usize> = (0..=row_len).collect();
    let mut curr = vec![0usize; row_len + 1];

    for i in 1..=col_len {
        curr[0] = i;
        for j in 1..=row_len {
            let cost = if col_src[i - 1] == row_src[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[row_len]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // ---- levenshtein ----

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(levenshtein("kitten", "kitten"), 0);
    }

    #[test]
    fn test_levenshtein_empty_a() {
        assert_eq!(levenshtein("", "abc"), 3);
    }

    #[test]
    fn test_levenshtein_empty_b() {
        assert_eq!(levenshtein("abc", ""), 3);
    }

    #[test]
    fn test_levenshtein_classic() {
        // "kitten" → "sitting" = 3 edits
        assert_eq!(levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_levenshtein_single_insertion() {
        assert_eq!(levenshtein("abc", "abcd"), 1);
    }

    #[test]
    fn test_levenshtein_single_deletion() {
        assert_eq!(levenshtein("abcd", "abc"), 1);
    }

    // ---- FuzzyMatcher::score ----

    #[test]
    fn test_score_exact_match() {
        let s = FuzzyMatcher::score("interview", "interview");
        assert!((s - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_score_case_insensitive() {
        let s = FuzzyMatcher::score("Interview", "interview");
        assert!((s - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_score_completely_different() {
        let s = FuzzyMatcher::score("aaa", "zzz");
        // 3 substitutions / max(3,3) = 0
        assert!((s - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_score_empty_both() {
        let s = FuzzyMatcher::score("", "");
        assert!((s - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_score_partial_match() {
        let s = FuzzyMatcher::score("interv", "interview");
        // 3 deletions / max(6,9) = 1 - 3/9 ≈ 0.666
        assert!(s > 0.5 && s < 1.0);
    }

    // ---- FuzzyMatcher::is_match ----

    #[test]
    fn test_is_match_true() {
        assert!(FuzzyMatcher::is_match("inteview", "interview", 0.7));
    }

    #[test]
    fn test_is_match_false() {
        assert!(!FuzzyMatcher::is_match("xyz", "interview", 0.9));
    }

    // ---- FuzzyMatcher::match_clips ----

    fn make_clip(name: &str, keywords: &[&str]) -> Clip {
        let mut c = Clip::new(PathBuf::from(format!("/test/{name}.mov")));
        c.set_name(name);
        for kw in keywords {
            c.add_keyword(*kw);
        }
        c
    }

    #[test]
    fn test_match_clips_by_name() {
        let clips = vec![
            make_clip("Interview John", &[]),
            make_clip("B-Roll Outdoor", &[]),
        ];
        let results = FuzzyMatcher::match_clips(&clips, "interview", 0.5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1.name, "Interview John");
    }

    #[test]
    fn test_match_clips_by_keyword() {
        let clips = vec![
            make_clip("Shot A", &["outdoor", "sunny"]),
            make_clip("Shot B", &["indoor", "dark"]),
        ];
        let results = FuzzyMatcher::match_clips(&clips, "outdoor", 0.8);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1.name, "Shot A");
    }

    #[test]
    fn test_match_clips_sorted_descending() {
        let clips = vec![
            make_clip("cat", &[]),      // closer to "bat"
            make_clip("elephant", &[]), // further
            make_clip("bat", &[]),      // exact
        ];
        let results = FuzzyMatcher::match_clips(&clips, "bat", 0.0);
        assert_eq!(results.len(), 3);
        // First result must have score >= second
        assert!(results[0].0 >= results[1].0);
        assert!(results[1].0 >= results[2].0);
    }

    #[test]
    fn test_match_clips_min_score_filter() {
        let clips = vec![
            make_clip("hello", &[]),
            make_clip("world", &[]),
            make_clip("help", &[]),
        ];
        let high = FuzzyMatcher::match_clips(&clips, "hello", 0.8);
        // "hello" exact = 1.0; "help" = 1 - 2/5 = 0.6 → filtered out
        assert_eq!(high.len(), 1);
    }

    #[test]
    fn test_match_clips_empty_list() {
        let results = FuzzyMatcher::match_clips(&[], "interview", 0.5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_match_clips_empty_pattern_matches_all_at_zero_threshold() {
        let clips = vec![make_clip("A", &[]), make_clip("B", &[])];
        // empty pattern vs "A": score = 1 - 1/1 = 0.0 >= 0.0
        let results = FuzzyMatcher::match_clips(&clips, "", 0.0);
        assert_eq!(results.len(), 2);
    }
}
