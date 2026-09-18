#![allow(dead_code)]
//! Spell-correction and suggestion engine for search queries.

use std::collections::HashMap;

/// A single spell-correction candidate.
#[derive(Debug, Clone)]
pub struct SpellCorrection {
    /// The suggested corrected term.
    pub suggestion: String,
    /// Confidence in the suggestion, in `[0.0, 1.0]`.
    pub score: f32,
    /// Original misspelled term.
    pub original: String,
}

impl SpellCorrection {
    /// Create a new correction.
    pub fn new(original: impl Into<String>, suggestion: impl Into<String>, score: f32) -> Self {
        Self {
            original: original.into(),
            suggestion: suggestion.into(),
            score: score.clamp(0.0, 1.0),
        }
    }

    /// Returns the confidence score.
    #[must_use]
    pub fn confidence(&self) -> f32 {
        self.score
    }
}

/// Cache for suggestion lookups.
#[derive(Debug, Default)]
pub struct SuggestionCache {
    inner: HashMap<String, Vec<SpellCorrection>>,
}

impl SuggestionCache {
    /// Create an empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if the cache contains an entry for `term`.
    #[must_use]
    pub fn cache_hit(&self, term: &str) -> bool {
        self.inner.contains_key(term)
    }

    /// Insert a list of corrections for `term`.
    pub fn insert(&mut self, term: impl Into<String>, corrections: Vec<SpellCorrection>) {
        self.inner.insert(term.into(), corrections);
    }

    /// Retrieve cached corrections for `term`.
    #[must_use]
    pub fn get(&self, term: &str) -> Option<&Vec<SpellCorrection>> {
        self.inner.get(term)
    }

    /// Returns the number of cached terms.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// Computes the edit distance (Levenshtein) between two strings.
#[allow(clippy::cast_precision_loss)]
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (m, n) = (a.len(), b.len());
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            dp[i][j] = if a[i - 1] == b[j - 1] {
                dp[i - 1][j - 1]
            } else {
                1 + dp[i - 1][j - 1].min(dp[i - 1][j]).min(dp[i][j - 1])
            };
        }
    }
    dp[m][n]
}

/// Suggests corrections for misspelled query terms.
#[derive(Debug)]
pub struct SpellSuggester {
    /// Known-correct vocabulary.
    vocabulary: Vec<String>,
    /// Max edit distance to consider a word a candidate.
    max_distance: usize,
    cache: SuggestionCache,
}

impl SpellSuggester {
    /// Create a suggester from a vocabulary list.
    #[must_use]
    pub fn new(vocabulary: Vec<String>) -> Self {
        Self {
            vocabulary,
            max_distance: 2,
            cache: SuggestionCache::new(),
        }
    }

    /// Set the maximum edit distance threshold.
    #[must_use]
    pub fn with_max_distance(mut self, d: usize) -> Self {
        self.max_distance = d;
        self
    }

    /// Return up to `n` correction candidates for `term`, sorted by confidence.
    #[allow(clippy::cast_precision_loss)]
    pub fn suggest(&mut self, term: &str) -> Vec<SpellCorrection> {
        let term_lower = term.to_lowercase();

        // Check cache first.
        if self.cache.cache_hit(&term_lower) {
            return self.cache.get(&term_lower).cloned().unwrap_or_default();
        }

        let mut candidates: Vec<SpellCorrection> = self
            .vocabulary
            .iter()
            .filter_map(|w| {
                let dist = edit_distance(&term_lower, &w.to_lowercase());
                if dist == 0 {
                    // Exact match – no correction needed.
                    return None;
                }
                if dist <= self.max_distance {
                    let max_len = term_lower.len().max(w.len()) as f32;
                    let score = if max_len > 0.0 {
                        1.0 - (dist as f32 / max_len)
                    } else {
                        0.0
                    };
                    Some(SpellCorrection::new(term, w.clone(), score))
                } else {
                    None
                }
            })
            .collect();

        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.cache.insert(term_lower, candidates.clone());
        candidates
    }

    /// Return the best single correction, or the original term if none found.
    pub fn auto_correct(&mut self, term: &str) -> String {
        let suggestions = self.suggest(term);
        suggestions
            .into_iter()
            .next()
            .map_or_else(|| term.to_string(), |c| c.suggestion)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> Vec<String> {
        vec![
            "video".to_string(),
            "audio".to_string(),
            "media".to_string(),
            "search".to_string(),
            "codec".to_string(),
            "encode".to_string(),
            "decode".to_string(),
        ]
    }

    #[test]
    fn test_spell_correction_confidence() {
        let c = SpellCorrection::new("vidoe", "video", 0.9);
        assert!((c.confidence() - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_spell_correction_clamped() {
        let c = SpellCorrection::new("x", "y", 1.5);
        assert_eq!(c.confidence(), 1.0);
    }

    #[test]
    fn test_cache_miss() {
        let cache = SuggestionCache::new();
        assert!(!cache.cache_hit("video"));
    }

    #[test]
    fn test_cache_insert_and_hit() {
        let mut cache = SuggestionCache::new();
        cache.insert("vidoe", vec![SpellCorrection::new("vidoe", "video", 0.9)]);
        assert!(cache.cache_hit("vidoe"));
    }

    #[test]
    fn test_cache_len() {
        let mut cache = SuggestionCache::new();
        cache.insert("a", vec![]);
        cache.insert("b", vec![]);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_cache_is_empty() {
        let cache = SuggestionCache::new();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_edit_distance_same() {
        assert_eq!(edit_distance("video", "video"), 0);
    }

    #[test]
    fn test_edit_distance_one_sub() {
        assert_eq!(edit_distance("vidoe", "video"), 2); // transposition = 2 ops
    }

    #[test]
    fn test_suggest_finds_close_word() {
        let mut s = SpellSuggester::new(vocab());
        let suggestions = s.suggest("vidoe");
        // "video" should be in suggestions
        assert!(suggestions.iter().any(|c| c.suggestion == "video"));
    }

    #[test]
    fn test_suggest_sorted_by_confidence() {
        let mut s = SpellSuggester::new(vocab());
        let suggestions = s.suggest("vidoe");
        for w in suggestions.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn test_suggest_exact_match_not_returned() {
        let mut s = SpellSuggester::new(vocab());
        let suggestions = s.suggest("video");
        // Exact match should produce no correction candidates.
        assert!(suggestions.is_empty() || suggestions.iter().all(|c| c.suggestion != "video"));
    }

    #[test]
    fn test_auto_correct_returns_best() {
        let mut s = SpellSuggester::new(vocab());
        let corrected = s.auto_correct("auido");
        assert_eq!(corrected, "audio");
    }

    #[test]
    fn test_auto_correct_no_match_returns_original() {
        let mut s = SpellSuggester::new(vocab());
        // "xyzqrs" is far from all vocab words
        let corrected = s.auto_correct("xyzqrs");
        assert_eq!(corrected, "xyzqrs");
    }

    #[test]
    fn test_suggest_caches_result() {
        let mut s = SpellSuggester::new(vocab());
        s.suggest("auido");
        assert!(s.cache.cache_hit("auido"));
    }

    #[test]
    fn test_suggest_uses_cache_on_second_call() {
        let mut s = SpellSuggester::new(vocab());
        let first = s.suggest("auido");
        let second = s.suggest("auido");
        assert_eq!(first.len(), second.len());
    }
}
