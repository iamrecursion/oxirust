#![allow(dead_code)]
//! Full-text search across subtitle cues.
//!
//! This module provides utilities for searching subtitle text by keyword,
//! pattern matching, time range filtering, and building simple inverted
//! indexes for fast lookups in large subtitle collections.

/// A single search hit referencing a subtitle by index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchHit {
    /// Index of the matching subtitle in the original collection.
    pub index: usize,
    /// Byte offset of the match within the subtitle text.
    pub byte_offset: usize,
    /// Length of the matched substring in bytes.
    pub match_len: usize,
}

/// Options for subtitle search operations.
#[derive(Clone, Debug)]
pub struct SearchOptions {
    /// Whether the search should be case-insensitive.
    pub case_insensitive: bool,
    /// If set, only match whole words (surrounded by non-alphanumeric chars).
    pub whole_word: bool,
    /// Maximum number of results to return (0 = unlimited).
    pub max_results: usize,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            case_insensitive: true,
            whole_word: false,
            max_results: 0,
        }
    }
}

impl SearchOptions {
    /// Create new search options with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set case sensitivity.
    #[must_use]
    pub fn case_insensitive(mut self, val: bool) -> Self {
        self.case_insensitive = val;
        self
    }

    /// Set whole-word matching.
    #[must_use]
    pub fn whole_word(mut self, val: bool) -> Self {
        self.whole_word = val;
        self
    }

    /// Set maximum result count (0 = unlimited).
    #[must_use]
    pub fn max_results(mut self, val: usize) -> Self {
        self.max_results = val;
        self
    }
}

/// Check if a byte position is at a word boundary in the given text.
fn is_word_boundary(text: &str, pos: usize, match_len: usize) -> bool {
    let before_ok = if pos == 0 {
        true
    } else {
        text.as_bytes()
            .get(pos - 1)
            .map_or(true, |&b| !b.is_ascii_alphanumeric())
    };
    let after_ok = {
        let end = pos + match_len;
        if end >= text.len() {
            true
        } else {
            text.as_bytes()
                .get(end)
                .map_or(true, |&b| !b.is_ascii_alphanumeric())
        }
    };
    before_ok && after_ok
}

/// Search a single text for all occurrences of a query string.
fn find_all_in_text(text: &str, query: &str, opts: &SearchOptions) -> Vec<(usize, usize)> {
    let mut hits = Vec::new();
    if query.is_empty() {
        return hits;
    }

    let (haystack, needle) = if opts.case_insensitive {
        (text.to_lowercase(), query.to_lowercase())
    } else {
        (text.to_string(), query.to_string())
    };

    let mut start = 0;
    while let Some(pos) = haystack[start..].find(&needle) {
        let abs_pos = start + pos;
        if !opts.whole_word || is_word_boundary(text, abs_pos, needle.len()) {
            hits.push((abs_pos, needle.len()));
        }
        start = abs_pos + 1;
    }
    hits
}

/// Subtitle entry used for search (text plus timing).
#[derive(Clone, Debug)]
pub struct SearchableSubtitle {
    /// The subtitle text.
    pub text: String,
    /// Start time in milliseconds.
    pub start_ms: i64,
    /// End time in milliseconds.
    pub end_ms: i64,
}

impl SearchableSubtitle {
    /// Create a new searchable subtitle entry.
    #[must_use]
    pub fn new(text: impl Into<String>, start_ms: i64, end_ms: i64) -> Self {
        Self {
            text: text.into(),
            start_ms,
            end_ms,
        }
    }
}

/// Search a collection of subtitles for a query string.
///
/// Returns a list of search hits with subtitle index and match positions.
#[must_use]
pub fn search_subtitles(
    subtitles: &[SearchableSubtitle],
    query: &str,
    opts: &SearchOptions,
) -> Vec<SearchHit> {
    let mut results = Vec::new();
    for (idx, sub) in subtitles.iter().enumerate() {
        let matches = find_all_in_text(&sub.text, query, opts);
        for (offset, len) in matches {
            results.push(SearchHit {
                index: idx,
                byte_offset: offset,
                match_len: len,
            });
            if opts.max_results > 0 && results.len() >= opts.max_results {
                return results;
            }
        }
    }
    results
}

/// Search subtitles within a specific time range.
#[must_use]
pub fn search_in_time_range(
    subtitles: &[SearchableSubtitle],
    query: &str,
    opts: &SearchOptions,
    range_start_ms: i64,
    range_end_ms: i64,
) -> Vec<SearchHit> {
    let mut results = Vec::new();
    for (idx, sub) in subtitles.iter().enumerate() {
        if sub.end_ms < range_start_ms || sub.start_ms > range_end_ms {
            continue;
        }
        let matches = find_all_in_text(&sub.text, query, opts);
        for (offset, len) in matches {
            results.push(SearchHit {
                index: idx,
                byte_offset: offset,
                match_len: len,
            });
            if opts.max_results > 0 && results.len() >= opts.max_results {
                return results;
            }
        }
    }
    results
}

/// Count total occurrences of a query across all subtitles.
#[must_use]
pub fn count_occurrences(
    subtitles: &[SearchableSubtitle],
    query: &str,
    case_insensitive: bool,
) -> usize {
    let opts = SearchOptions {
        case_insensitive,
        whole_word: false,
        max_results: 0,
    };
    subtitles
        .iter()
        .map(|sub| find_all_in_text(&sub.text, query, &opts).len())
        .sum()
}

/// A simple inverted index mapping words to subtitle indices.
#[derive(Clone, Debug, Default)]
pub struct SubtitleWordIndex {
    /// Map from lowercased word to list of subtitle indices.
    entries: Vec<(String, Vec<usize>)>,
}

impl SubtitleWordIndex {
    /// Build an inverted word index from a collection of subtitles.
    #[must_use]
    pub fn build(subtitles: &[SearchableSubtitle]) -> Self {
        let mut entries: Vec<(String, Vec<usize>)> = Vec::new();
        for (idx, sub) in subtitles.iter().enumerate() {
            for word in sub.text.split(|c: char| !c.is_alphanumeric()) {
                if word.is_empty() {
                    continue;
                }
                let lower = word.to_lowercase();
                if let Some(entry) = entries.iter_mut().find(|(w, _)| w == &lower) {
                    if entry.1.last() != Some(&idx) {
                        entry.1.push(idx);
                    }
                } else {
                    entries.push((lower, vec![idx]));
                }
            }
        }
        Self { entries }
    }

    /// Look up all subtitle indices containing the given word.
    #[must_use]
    pub fn lookup(&self, word: &str) -> Vec<usize> {
        let lower = word.to_lowercase();
        self.entries
            .iter()
            .find(|(w, _)| w == &lower)
            .map_or_else(Vec::new, |(_, indices)| indices.clone())
    }

    /// Return the number of unique words in the index.
    #[must_use]
    pub fn word_count(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_subs() -> Vec<SearchableSubtitle> {
        vec![
            SearchableSubtitle::new("Hello world", 0, 2000),
            SearchableSubtitle::new("The quick brown fox", 2000, 4000),
            SearchableSubtitle::new("Hello again, world!", 4000, 6000),
            SearchableSubtitle::new("Goodbye everyone", 6000, 8000),
        ]
    }

    #[test]
    fn test_basic_search() {
        let subs = sample_subs();
        let hits = search_subtitles(&subs, "hello", &SearchOptions::default());
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].index, 0);
        assert_eq!(hits[1].index, 2);
    }

    #[test]
    fn test_case_sensitive_search() {
        let subs = sample_subs();
        let opts = SearchOptions::new().case_insensitive(false);
        let hits = search_subtitles(&subs, "hello", &opts);
        assert_eq!(hits.len(), 0);
        let hits2 = search_subtitles(&subs, "Hello", &opts);
        assert_eq!(hits2.len(), 2);
    }

    #[test]
    fn test_whole_word_search() {
        let subs = vec![SearchableSubtitle::new("the cat sat on the mat", 0, 1000)];
        let opts = SearchOptions::new().whole_word(true);
        let hits = search_subtitles(&subs, "the", &opts);
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn test_whole_word_no_partial() {
        let subs = vec![SearchableSubtitle::new("theater", 0, 1000)];
        let opts = SearchOptions::new().whole_word(true);
        let hits = search_subtitles(&subs, "the", &opts);
        assert_eq!(hits.len(), 0);
    }

    #[test]
    fn test_max_results() {
        let subs = sample_subs();
        let opts = SearchOptions::new().max_results(1);
        let hits = search_subtitles(&subs, "hello", &opts);
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn test_empty_query() {
        let subs = sample_subs();
        let hits = search_subtitles(&subs, "", &SearchOptions::default());
        assert!(hits.is_empty());
    }

    #[test]
    fn test_search_in_time_range() {
        let subs = sample_subs();
        let hits = search_in_time_range(&subs, "hello", &SearchOptions::default(), 3000, 7000);
        // Only the third subtitle (4000-6000) is within the time range and matches
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].index, 2);
    }

    #[test]
    fn test_search_time_range_no_match() {
        let subs = sample_subs();
        let hits = search_in_time_range(&subs, "hello", &SearchOptions::default(), 6500, 8000);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_count_occurrences() {
        let subs = sample_subs();
        let count = count_occurrences(&subs, "world", true);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_word_index_build_and_lookup() {
        let subs = sample_subs();
        let index = SubtitleWordIndex::build(&subs);
        let hello_indices = index.lookup("hello");
        assert_eq!(hello_indices, vec![0, 2]);
        let fox_indices = index.lookup("fox");
        assert_eq!(fox_indices, vec![1]);
    }

    #[test]
    fn test_word_index_missing_word() {
        let subs = sample_subs();
        let index = SubtitleWordIndex::build(&subs);
        let indices = index.lookup("nonexistent");
        assert!(indices.is_empty());
    }

    #[test]
    fn test_word_index_word_count() {
        let subs = vec![SearchableSubtitle::new("one two three", 0, 1000)];
        let index = SubtitleWordIndex::build(&subs);
        assert_eq!(index.word_count(), 3);
    }

    #[test]
    fn test_search_hit_byte_offset() {
        let subs = vec![SearchableSubtitle::new("abc hello xyz", 0, 1000)];
        let hits = search_subtitles(&subs, "hello", &SearchOptions::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].byte_offset, 4);
        assert_eq!(hits[0].match_len, 5);
    }
}
