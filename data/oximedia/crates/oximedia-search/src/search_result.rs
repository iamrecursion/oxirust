#![allow(dead_code)]

//! Search result formatting, highlighting, and snippet extraction.
//!
//! This module provides utilities to format raw search results into
//! user-friendly output, including context-aware snippet extraction,
//! keyword highlighting, and result grouping.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Snippet extraction
// ---------------------------------------------------------------------------

/// Configuration for snippet extraction from search results.
#[derive(Debug, Clone)]
pub struct SnippetConfig {
    /// Maximum number of characters in a snippet.
    pub max_length: usize,
    /// Number of context characters around each matched keyword.
    pub context_chars: usize,
    /// Maximum number of snippets per result.
    pub max_snippets: usize,
    /// Opening tag for highlighting matched text.
    pub highlight_pre: String,
    /// Closing tag for highlighting matched text.
    pub highlight_post: String,
    /// Separator between non-contiguous snippet fragments.
    pub fragment_separator: String,
}

impl Default for SnippetConfig {
    fn default() -> Self {
        Self {
            max_length: 300,
            context_chars: 40,
            max_snippets: 3,
            highlight_pre: "<em>".to_string(),
            highlight_post: "</em>".to_string(),
            fragment_separator: " ... ".to_string(),
        }
    }
}

/// A single highlighted snippet extracted from a document field.
#[derive(Debug, Clone)]
pub struct Snippet {
    /// The formatted snippet text (with highlight tags).
    pub text: String,
    /// The source field name this snippet came from.
    pub field: String,
    /// Byte offset in original text where the snippet starts.
    pub offset: usize,
    /// Number of keyword matches in this snippet.
    pub match_count: usize,
}

/// Extracts snippets from text by locating keyword positions and
/// returning context windows around them.
#[derive(Debug, Clone)]
pub struct SnippetExtractor {
    /// Configuration for snippet extraction.
    config: SnippetConfig,
}

impl SnippetExtractor {
    /// Create a new snippet extractor with the given configuration.
    #[must_use]
    pub fn new(config: SnippetConfig) -> Self {
        Self { config }
    }

    /// Create a snippet extractor with default settings.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self {
            config: SnippetConfig::default(),
        }
    }

    /// Extract snippets from the given `text` for the given `keywords`.
    ///
    /// Returns a list of [`Snippet`] values, up to `config.max_snippets`.
    #[must_use]
    pub fn extract(&self, text: &str, keywords: &[&str], field: &str) -> Vec<Snippet> {
        if text.is_empty() || keywords.is_empty() {
            return Vec::new();
        }

        let lower_text = text.to_lowercase();
        let mut positions: Vec<(usize, usize)> = Vec::new();

        for kw in keywords {
            let kw_lower = kw.to_lowercase();
            let mut start = 0;
            while let Some(pos) = lower_text[start..].find(&kw_lower) {
                let abs = start + pos;
                positions.push((abs, kw_lower.len()));
                start = abs + 1;
            }
        }

        positions.sort_by_key(|&(pos, _)| pos);
        positions.dedup();

        // Merge overlapping windows
        let windows = self.merge_windows(&positions, text.len());
        let mut snippets = Vec::new();

        for (win_start, win_end, match_count) in windows.into_iter().take(self.config.max_snippets)
        {
            let raw = &text[win_start..win_end];
            let highlighted = self.highlight_in_range(text, win_start, win_end, &positions);
            snippets.push(Snippet {
                text: highlighted,
                field: field.to_string(),
                offset: win_start,
                match_count,
            });
            let _ = raw; // satisfy usage
        }

        snippets
    }

    /// Merge overlapping context windows into contiguous ranges.
    fn merge_windows(
        &self,
        positions: &[(usize, usize)],
        text_len: usize,
    ) -> Vec<(usize, usize, usize)> {
        if positions.is_empty() {
            return Vec::new();
        }

        let ctx = self.config.context_chars;
        let mut merged: Vec<(usize, usize, usize)> = Vec::new();

        for &(pos, kw_len) in positions {
            let start = pos.saturating_sub(ctx);
            let end = (pos + kw_len + ctx).min(text_len);

            if let Some(last) = merged.last_mut() {
                if start <= last.1 {
                    last.1 = last.1.max(end);
                    last.2 += 1;
                    continue;
                }
            }
            merged.push((start, end, 1));
        }

        merged
    }

    /// Apply highlight markers to keywords within a given range.
    fn highlight_in_range(
        &self,
        text: &str,
        win_start: usize,
        win_end: usize,
        positions: &[(usize, usize)],
    ) -> String {
        let sub = &text[win_start..win_end];
        let mut result = String::with_capacity(sub.len() + 64);
        let mut cursor = 0usize;

        for &(abs_pos, kw_len) in positions {
            if abs_pos < win_start || abs_pos + kw_len > win_end {
                continue;
            }
            let rel = abs_pos - win_start;
            if rel > cursor {
                result.push_str(&sub[cursor..rel]);
            }
            result.push_str(&self.config.highlight_pre);
            result.push_str(&sub[rel..rel + kw_len]);
            result.push_str(&self.config.highlight_post);
            cursor = rel + kw_len;
        }

        if cursor < sub.len() {
            result.push_str(&sub[cursor..]);
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Result grouping
// ---------------------------------------------------------------------------

/// Strategy for grouping search results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupStrategy {
    /// No grouping.
    None,
    /// Group by a specific field value.
    ByField,
    /// Group by date (day granularity).
    ByDate,
    /// Group by score ranges (buckets).
    ByScoreBucket,
}

/// A single group of search results sharing a common attribute.
#[derive(Debug, Clone)]
pub struct ResultGroup {
    /// Label for this group.
    pub label: String,
    /// Number of results in this group.
    pub count: usize,
    /// Sum of relevance scores in this group.
    pub total_score: f64,
    /// Top result score in this group.
    pub max_score: f64,
}

/// Groups search results by a specified strategy.
#[derive(Debug)]
pub struct ResultGrouper {
    /// The strategy to use.
    strategy: GroupStrategy,
    /// Number of score buckets (used for `ByScoreBucket`).
    bucket_count: usize,
}

impl ResultGrouper {
    /// Create a new result grouper with the given strategy.
    #[must_use]
    pub fn new(strategy: GroupStrategy) -> Self {
        Self {
            strategy,
            bucket_count: 5,
        }
    }

    /// Create a result grouper for score buckets with a custom bucket count.
    #[must_use]
    pub fn with_buckets(bucket_count: usize) -> Self {
        Self {
            strategy: GroupStrategy::ByScoreBucket,
            bucket_count: bucket_count.max(1),
        }
    }

    /// Group items by field values. Each item is represented as
    /// `(field_value, score)`.
    #[must_use]
    pub fn group_by_field(&self, items: &[(&str, f64)]) -> Vec<ResultGroup> {
        let mut map: HashMap<&str, (usize, f64, f64)> = HashMap::new();
        for &(field, score) in items {
            let entry = map.entry(field).or_insert((0, 0.0, f64::NEG_INFINITY));
            entry.0 += 1;
            entry.1 += score;
            if score > entry.2 {
                entry.2 = score;
            }
        }

        let mut groups: Vec<ResultGroup> = map
            .into_iter()
            .map(|(label, (count, total, max))| ResultGroup {
                label: label.to_string(),
                count,
                total_score: total,
                max_score: max,
            })
            .collect();
        groups.sort_by(|a, b| {
            b.max_score
                .partial_cmp(&a.max_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        groups
    }

    /// Group scores into fixed-width buckets between 0.0 and 1.0.
    #[must_use]
    pub fn group_by_score(&self, scores: &[f64]) -> Vec<ResultGroup> {
        let n = self.bucket_count;
        let width = 1.0 / n as f64;
        let mut buckets: Vec<(usize, f64, f64)> = vec![(0, 0.0, f64::NEG_INFINITY); n];

        for &s in scores {
            let clamped = s.clamp(0.0, 1.0 - f64::EPSILON);
            let idx = (clamped / width) as usize;
            let idx = idx.min(n - 1);
            buckets[idx].0 += 1;
            buckets[idx].1 += s;
            if s > buckets[idx].2 {
                buckets[idx].2 = s;
            }
        }

        buckets
            .into_iter()
            .enumerate()
            .map(|(i, (count, total, max))| {
                let lo = i as f64 * width;
                let hi = lo + width;
                ResultGroup {
                    label: format!("{lo:.2}-{hi:.2}"),
                    count,
                    total_score: total,
                    max_score: if max == f64::NEG_INFINITY { 0.0 } else { max },
                }
            })
            .collect()
    }

    /// Return the active strategy.
    #[must_use]
    pub fn strategy(&self) -> GroupStrategy {
        self.strategy
    }
}

// ---------------------------------------------------------------------------
// Result deduplication
// ---------------------------------------------------------------------------

/// Strategy for deduplicating near-identical search results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DedupStrategy {
    /// Keep the first occurrence.
    KeepFirst,
    /// Keep the highest-scoring occurrence.
    KeepHighestScore,
}

/// Deduplicator that removes near-identical results based on a key.
#[derive(Debug)]
pub struct ResultDeduplicator {
    /// Strategy to apply.
    strategy: DedupStrategy,
}

impl ResultDeduplicator {
    /// Create a new deduplicator with the given strategy.
    #[must_use]
    pub fn new(strategy: DedupStrategy) -> Self {
        Self { strategy }
    }

    /// Deduplicate items represented as `(key, score)`.
    /// Returns indices of items to keep.
    #[must_use]
    pub fn deduplicate(&self, items: &[(&str, f64)]) -> Vec<usize> {
        let mut seen: HashMap<&str, (usize, f64)> = HashMap::new();

        for (i, &(key, score)) in items.iter().enumerate() {
            match self.strategy {
                DedupStrategy::KeepFirst => {
                    seen.entry(key).or_insert((i, score));
                }
                DedupStrategy::KeepHighestScore => {
                    let entry = seen.entry(key).or_insert((i, score));
                    if score > entry.1 {
                        *entry = (i, score);
                    }
                }
            }
        }

        let mut indices: Vec<usize> = seen.values().map(|&(i, _)| i).collect();
        indices.sort_unstable();
        indices
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- SnippetExtractor tests --

    #[test]
    fn test_extract_empty_text() {
        let extractor = SnippetExtractor::with_defaults();
        let snippets = extractor.extract("", &["hello"], "body");
        assert!(snippets.is_empty());
    }

    #[test]
    fn test_extract_no_keywords() {
        let extractor = SnippetExtractor::with_defaults();
        let snippets = extractor.extract("some text", &[], "body");
        assert!(snippets.is_empty());
    }

    #[test]
    fn test_extract_single_keyword() {
        let extractor = SnippetExtractor::new(SnippetConfig {
            context_chars: 5,
            ..Default::default()
        });
        let text = "The quick brown fox jumps over the lazy dog";
        let snippets = extractor.extract(text, &["fox"], "body");
        assert!(!snippets.is_empty());
        assert!(snippets[0].text.contains("<em>fox</em>"));
        assert_eq!(snippets[0].match_count, 1);
    }

    #[test]
    fn test_extract_case_insensitive() {
        let extractor = SnippetExtractor::new(SnippetConfig {
            context_chars: 5,
            ..Default::default()
        });
        let text = "Hello World";
        let snippets = extractor.extract(text, &["hello"], "title");
        assert!(!snippets.is_empty());
        // The original casing is preserved in the highlighted text
        assert!(snippets[0].text.contains("<em>Hello</em>"));
    }

    #[test]
    fn test_extract_multiple_keywords() {
        let extractor = SnippetExtractor::new(SnippetConfig {
            context_chars: 3,
            max_snippets: 5,
            ..Default::default()
        });
        let text = "alpha beta gamma delta epsilon zeta eta theta";
        let snippets = extractor.extract(text, &["alpha", "theta"], "body");
        assert!(!snippets.is_empty());
    }

    #[test]
    fn test_extract_respects_max_snippets() {
        let extractor = SnippetExtractor::new(SnippetConfig {
            context_chars: 2,
            max_snippets: 1,
            ..Default::default()
        });
        let text = "aaa bbb aaa bbb aaa bbb aaa bbb aaa bbb aaa bbb";
        let snippets = extractor.extract(text, &["aaa"], "body");
        assert!(snippets.len() <= 1);
    }

    #[test]
    fn test_snippet_field_name() {
        let extractor = SnippetExtractor::with_defaults();
        let snippets = extractor.extract("hello world", &["world"], "description");
        assert!(!snippets.is_empty());
        assert_eq!(snippets[0].field, "description");
    }

    // -- ResultGrouper tests --

    #[test]
    fn test_group_by_field_empty() {
        let grouper = ResultGrouper::new(GroupStrategy::ByField);
        let groups = grouper.group_by_field(&[]);
        assert!(groups.is_empty());
    }

    #[test]
    fn test_group_by_field_single() {
        let grouper = ResultGrouper::new(GroupStrategy::ByField);
        let items = vec![("video", 0.9)];
        let groups = grouper.group_by_field(&items);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].label, "video");
        assert_eq!(groups[0].count, 1);
    }

    #[test]
    fn test_group_by_field_multiple() {
        let grouper = ResultGrouper::new(GroupStrategy::ByField);
        let items = vec![
            ("video", 0.9),
            ("audio", 0.7),
            ("video", 0.5),
            ("image", 0.3),
        ];
        let groups = grouper.group_by_field(&items);
        let video_group = groups
            .iter()
            .find(|g| g.label == "video")
            .expect("should succeed in test");
        assert_eq!(video_group.count, 2);
        assert!((video_group.total_score - 1.4).abs() < 1e-9);
    }

    #[test]
    fn test_group_by_score_buckets() {
        let grouper = ResultGrouper::with_buckets(4);
        let scores = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let groups = grouper.group_by_score(&scores);
        assert_eq!(groups.len(), 4);
        let total: usize = groups.iter().map(|g| g.count).sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn test_group_strategy_accessor() {
        let grouper = ResultGrouper::new(GroupStrategy::ByDate);
        assert_eq!(grouper.strategy(), GroupStrategy::ByDate);
    }

    // -- ResultDeduplicator tests --

    #[test]
    fn test_dedup_keep_first() {
        let dedup = ResultDeduplicator::new(DedupStrategy::KeepFirst);
        let items = vec![("a", 0.5), ("b", 0.8), ("a", 0.9)];
        let kept = dedup.deduplicate(&items);
        assert_eq!(kept.len(), 2);
        assert!(kept.contains(&0)); // first "a"
        assert!(kept.contains(&1)); // "b"
    }

    #[test]
    fn test_dedup_keep_highest() {
        let dedup = ResultDeduplicator::new(DedupStrategy::KeepHighestScore);
        let items = vec![("a", 0.5), ("b", 0.8), ("a", 0.9)];
        let kept = dedup.deduplicate(&items);
        assert_eq!(kept.len(), 2);
        assert!(kept.contains(&2)); // highest "a" at index 2
        assert!(kept.contains(&1)); // "b"
    }

    #[test]
    fn test_dedup_empty() {
        let dedup = ResultDeduplicator::new(DedupStrategy::KeepFirst);
        let items: Vec<(&str, f64)> = Vec::new();
        let kept = dedup.deduplicate(&items);
        assert!(kept.is_empty());
    }

    #[test]
    fn test_dedup_all_unique() {
        let dedup = ResultDeduplicator::new(DedupStrategy::KeepFirst);
        let items = vec![("a", 0.5), ("b", 0.6), ("c", 0.7)];
        let kept = dedup.deduplicate(&items);
        assert_eq!(kept.len(), 3);
    }
}
