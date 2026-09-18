#![allow(dead_code)]

//! Fuzzy / approximate matching for media deduplication.
//!
//! This module provides edit-distance and similarity metrics that detect
//! near-duplicate media by comparing fingerprints, metadata strings, or
//! byte sequences that may differ slightly due to re-encoding, cropping,
//! or metadata edits.
//!
//! # Key Types
//!
//! - [`EditDistance`] - Levenshtein edit distance calculator
//! - [`FuzzyScore`] - A normalised similarity score (0.0 to 1.0)
//! - [`TokenMatcher`] - Token-based (bag-of-words) similarity
//! - [`BigramSimilarity`] - Character bigram overlap metric

use std::collections::{HashMap, HashSet};
use std::fmt;

/// A normalised similarity score in the range `[0.0, 1.0]`.
///
/// - `1.0` means an exact match.
/// - `0.0` means completely dissimilar.
#[derive(Debug, Clone, Copy)]
pub struct FuzzyScore {
    /// The raw score value.
    value: f64,
}

impl FuzzyScore {
    /// Create a new score, clamping to `[0.0, 1.0]`.
    #[must_use]
    pub fn new(value: f64) -> Self {
        Self {
            value: value.clamp(0.0, 1.0),
        }
    }

    /// Return the score value.
    #[must_use]
    pub fn value(self) -> f64 {
        self.value
    }

    /// Check whether the score meets a given threshold.
    #[must_use]
    pub fn meets_threshold(self, threshold: f64) -> bool {
        self.value >= threshold
    }

    /// Exact match (score == 1.0).
    #[must_use]
    pub fn is_exact(self) -> bool {
        (self.value - 1.0).abs() < f64::EPSILON
    }

    /// Combine two scores by averaging.
    #[must_use]
    pub fn average(self, other: Self) -> Self {
        Self::new((self.value + other.value) / 2.0)
    }

    /// Combine two scores using a weighted average.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn weighted_average(scores: &[(Self, f64)]) -> Self {
        if scores.is_empty() {
            return Self::new(0.0);
        }
        let total_weight: f64 = scores.iter().map(|(_, w)| w).sum();
        if total_weight <= 0.0 {
            return Self::new(0.0);
        }
        let weighted_sum: f64 = scores.iter().map(|(s, w)| s.value * w).sum();
        Self::new(weighted_sum / total_weight)
    }
}

impl fmt::Display for FuzzyScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.value)
    }
}

impl PartialEq for FuzzyScore {
    fn eq(&self, other: &Self) -> bool {
        (self.value - other.value).abs() < 1e-10
    }
}

/// Levenshtein edit distance calculator.
pub struct EditDistance;

impl EditDistance {
    /// Compute the Levenshtein distance between two byte slices.
    #[must_use]
    pub fn bytes(a: &[u8], b: &[u8]) -> usize {
        let m = a.len();
        let n = b.len();

        if m == 0 {
            return n;
        }
        if n == 0 {
            return m;
        }

        // Use single-row optimisation
        let mut prev = vec![0usize; n + 1];
        let mut curr = vec![0usize; n + 1];

        for j in 0..=n {
            prev[j] = j;
        }

        for i in 1..=m {
            curr[0] = i;
            for j in 1..=n {
                let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
                curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
            }
            std::mem::swap(&mut prev, &mut curr);
        }

        prev[n]
    }

    /// Compute the Levenshtein distance between two strings.
    #[must_use]
    pub fn strings(a: &str, b: &str) -> usize {
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

        let mut prev = vec![0usize; n + 1];
        let mut curr = vec![0usize; n + 1];

        for j in 0..=n {
            prev[j] = j;
        }

        for i in 1..=m {
            curr[0] = i;
            for j in 1..=n {
                let cost = if a_chars[i - 1] == b_chars[j - 1] {
                    0
                } else {
                    1
                };
                curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
            }
            std::mem::swap(&mut prev, &mut curr);
        }

        prev[n]
    }

    /// Convert edit distance to a normalised similarity score.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn similarity(a: &str, b: &str) -> FuzzyScore {
        let dist = Self::strings(a, b);
        let max_len = a.chars().count().max(b.chars().count());
        if max_len == 0 {
            return FuzzyScore::new(1.0);
        }
        FuzzyScore::new(1.0 - dist as f64 / max_len as f64)
    }
}

/// Token-based (bag-of-words) similarity.
///
/// Computes the Jaccard index of the token sets extracted from two strings.
pub struct TokenMatcher {
    /// Separator characters used for tokenisation.
    separators: Vec<char>,
    /// Whether to compare case-insensitively.
    case_insensitive: bool,
}

impl TokenMatcher {
    /// Create a new token matcher with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            separators: vec![' ', '-', '_', '.', ',', ';', '/', '\\'],
            case_insensitive: true,
        }
    }

    /// Set whether comparison is case-insensitive.
    #[must_use]
    pub fn case_insensitive(mut self, yes: bool) -> Self {
        self.case_insensitive = yes;
        self
    }

    /// Tokenise a string into a set of tokens.
    fn tokenize(&self, s: &str) -> HashSet<String> {
        let input = if self.case_insensitive {
            s.to_lowercase()
        } else {
            s.to_string()
        };

        let mut tokens = HashSet::new();
        let mut current = String::new();

        for ch in input.chars() {
            if self.separators.contains(&ch) {
                if !current.is_empty() {
                    tokens.insert(std::mem::take(&mut current));
                }
            } else {
                current.push(ch);
            }
        }
        if !current.is_empty() {
            tokens.insert(current);
        }

        tokens
    }

    /// Compute the Jaccard similarity between two strings.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn similarity(&self, a: &str, b: &str) -> FuzzyScore {
        let set_a = self.tokenize(a);
        let set_b = self.tokenize(b);

        if set_a.is_empty() && set_b.is_empty() {
            return FuzzyScore::new(1.0);
        }

        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();

        if union == 0 {
            FuzzyScore::new(0.0)
        } else {
            FuzzyScore::new(intersection as f64 / union as f64)
        }
    }
}

impl Default for TokenMatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Character bigram overlap metric (Dice coefficient).
pub struct BigramSimilarity;

impl BigramSimilarity {
    /// Extract character bigrams from a string.
    fn bigrams(s: &str) -> HashMap<(char, char), usize> {
        let chars: Vec<char> = s.chars().collect();
        let mut map = HashMap::new();
        if chars.len() < 2 {
            return map;
        }
        for pair in chars.windows(2) {
            *map.entry((pair[0], pair[1])).or_insert(0) += 1;
        }
        map
    }

    /// Compute the Dice coefficient between two strings.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn similarity(a: &str, b: &str) -> FuzzyScore {
        let bg_a = Self::bigrams(&a.to_lowercase());
        let bg_b = Self::bigrams(&b.to_lowercase());

        if bg_a.is_empty() && bg_b.is_empty() {
            return FuzzyScore::new(1.0);
        }

        let mut intersection_count: usize = 0;
        for (bigram, count_a) in &bg_a {
            if let Some(count_b) = bg_b.get(bigram) {
                intersection_count += (*count_a).min(*count_b);
            }
        }

        let total_a: usize = bg_a.values().sum();
        let total_b: usize = bg_b.values().sum();
        let denom = total_a + total_b;

        if denom == 0 {
            FuzzyScore::new(0.0)
        } else {
            FuzzyScore::new(2.0 * intersection_count as f64 / denom as f64)
        }
    }
}

/// Hamming distance between two equal-length byte slices.
///
/// Counts the number of positions where corresponding bytes differ.
/// Returns `None` if the slices have different lengths.
#[must_use]
pub fn hamming_distance(a: &[u8], b: &[u8]) -> Option<usize> {
    if a.len() != b.len() {
        return None;
    }
    Some(a.iter().zip(b.iter()).filter(|(x, y)| x != y).count())
}

/// Normalised Hamming similarity (1.0 = identical, 0.0 = all bits differ).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn hamming_similarity(a: &[u8], b: &[u8]) -> Option<FuzzyScore> {
    let dist = hamming_distance(a, b)?;
    let len = a.len();
    if len == 0 {
        return Some(FuzzyScore::new(1.0));
    }
    Some(FuzzyScore::new(1.0 - dist as f64 / len as f64))
}

// ---------------------------------------------------------------------------
// Filename / title matching for media deduplication
// ---------------------------------------------------------------------------

/// Normalized filename / title matcher for media deduplication.
///
/// Strips common media-file noise (resolution tags, codec names, release-group
/// markers, punctuation) and computes a combined similarity from Levenshtein
/// edit distance, token Jaccard, and bigram Dice coefficient.
pub struct FilenameMatcher {
    /// Weight for edit-distance similarity (0.0–1.0).
    edit_weight: f64,
    /// Weight for token Jaccard similarity (0.0–1.0).
    token_weight: f64,
    /// Weight for bigram Dice similarity (0.0–1.0).
    bigram_weight: f64,
    /// Minimum combined score to consider a match.
    threshold: f64,
}

impl FilenameMatcher {
    /// Create a new matcher with default weights.
    #[must_use]
    pub fn new(threshold: f64) -> Self {
        Self {
            edit_weight: 0.4,
            token_weight: 0.35,
            bigram_weight: 0.25,
            threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Create with custom weights.  Weights are normalized internally.
    #[must_use]
    pub fn with_weights(threshold: f64, edit_w: f64, token_w: f64, bigram_w: f64) -> Self {
        let total = edit_w + token_w + bigram_w;
        let (ew, tw, bw) = if total <= 0.0 {
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        } else {
            (edit_w / total, token_w / total, bigram_w / total)
        };
        Self {
            edit_weight: ew,
            token_weight: tw,
            bigram_weight: bw,
            threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Normalize a filename/title for comparison.
    ///
    /// Strips extension, converts to lowercase, removes common noise tokens
    /// (resolution tags, codec names, quality markers), and collapses whitespace.
    #[must_use]
    pub fn normalize(name: &str) -> String {
        // Strip directory components – keep only the filename.
        let base = name.rsplit(['/', '\\']).next().unwrap_or(name);

        // Strip extension
        let stem = base.rsplit_once('.').map_or(base, |(s, _)| s);

        let lower = stem.to_lowercase();

        // Remove common noise tokens (resolution, codec, quality, release)
        let noise: &[&str] = &[
            "1080p", "720p", "480p", "2160p", "4k", "uhd", "hdr", "hdr10", "x264", "x265", "h264",
            "h265", "hevc", "avc", "vp9", "av1", "aac", "ac3", "dts", "flac", "opus", "mp3",
            "bluray", "bdrip", "brrip", "webrip", "web-dl", "webdl", "dvdrip", "remux", "remaster",
            "proper", "repack", "mkv", "mp4", "avi", "mov", "wmv", "webm",
        ];

        let mut cleaned = lower;
        for &tag in noise {
            // Replace noise tokens bounded by non-alphanumerics or start/end
            cleaned = remove_noise_token(&cleaned, tag);
        }

        // Replace common separators with spaces
        let normalized: String = cleaned
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == ' ' {
                    c
                } else {
                    ' '
                }
            })
            .collect();

        // Collapse whitespace
        let parts: Vec<&str> = normalized.split_whitespace().collect();
        parts.join(" ")
    }

    /// Compute the combined similarity score between two filenames/titles.
    #[must_use]
    pub fn similarity(&self, name_a: &str, name_b: &str) -> FuzzyScore {
        let norm_a = Self::normalize(name_a);
        let norm_b = Self::normalize(name_b);

        if norm_a.is_empty() && norm_b.is_empty() {
            return FuzzyScore::new(1.0);
        }

        let edit_sim = EditDistance::similarity(&norm_a, &norm_b);
        let token_sim = TokenMatcher::new().similarity(&norm_a, &norm_b);
        let bigram_sim = BigramSimilarity::similarity(&norm_a, &norm_b);

        let combined = edit_sim.value() * self.edit_weight
            + token_sim.value() * self.token_weight
            + bigram_sim.value() * self.bigram_weight;

        FuzzyScore::new(combined)
    }

    /// Returns `true` if the two filenames are considered matching.
    #[must_use]
    pub fn is_match(&self, name_a: &str, name_b: &str) -> bool {
        self.similarity(name_a, name_b)
            .meets_threshold(self.threshold)
    }

    /// Find all matching pairs within a list of filenames.
    ///
    /// Returns `Vec<(usize, usize, FuzzyScore)>` with `i < j`.
    #[must_use]
    pub fn find_matching_pairs(&self, names: &[&str]) -> Vec<(usize, usize, FuzzyScore)> {
        let mut pairs = Vec::new();
        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                let score = self.similarity(names[i], names[j]);
                if score.meets_threshold(self.threshold) {
                    pairs.push((i, j, score));
                }
            }
        }
        pairs
    }

    /// Return the threshold.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

impl Default for FilenameMatcher {
    fn default() -> Self {
        Self::new(0.80)
    }
}

/// Remove a noise token from a string, respecting word boundaries.
fn remove_noise_token(input: &str, token: &str) -> String {
    let mut result = input.to_string();
    loop {
        let lower = result.to_lowercase();
        if let Some(pos) = lower.find(token) {
            let before_ok = pos == 0
                || !lower
                    .as_bytes()
                    .get(pos - 1)
                    .map_or(false, |b| b.is_ascii_alphanumeric());
            let after_pos = pos + token.len();
            let after_ok = after_pos >= lower.len()
                || !lower
                    .as_bytes()
                    .get(after_pos)
                    .map_or(false, |b| b.is_ascii_alphanumeric());
            if before_ok && after_ok {
                result = format!("{}{}", &result[..pos], &result[after_pos..]);
            } else {
                break;
            }
        } else {
            break;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuzzy_score_clamp() {
        assert!((FuzzyScore::new(1.5).value() - 1.0).abs() < f64::EPSILON);
        assert!((FuzzyScore::new(-0.3).value() - 0.0).abs() < f64::EPSILON);
        assert!((FuzzyScore::new(0.75).value() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fuzzy_score_threshold() {
        let s = FuzzyScore::new(0.85);
        assert!(s.meets_threshold(0.8));
        assert!(s.meets_threshold(0.85));
        assert!(!s.meets_threshold(0.9));
    }

    #[test]
    fn test_fuzzy_score_is_exact() {
        assert!(FuzzyScore::new(1.0).is_exact());
        assert!(!FuzzyScore::new(0.999).is_exact());
    }

    #[test]
    fn test_fuzzy_score_average() {
        let a = FuzzyScore::new(0.6);
        let b = FuzzyScore::new(0.8);
        let avg = a.average(b);
        assert!((avg.value() - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_fuzzy_score_weighted_average() {
        let scores = vec![(FuzzyScore::new(1.0), 3.0), (FuzzyScore::new(0.0), 1.0)];
        let avg = FuzzyScore::weighted_average(&scores);
        assert!((avg.value() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_edit_distance_strings_identical() {
        assert_eq!(EditDistance::strings("hello", "hello"), 0);
    }

    #[test]
    fn test_edit_distance_strings_basic() {
        assert_eq!(EditDistance::strings("kitten", "sitting"), 3);
        assert_eq!(EditDistance::strings("", "abc"), 3);
        assert_eq!(EditDistance::strings("abc", ""), 3);
    }

    #[test]
    fn test_edit_distance_bytes() {
        assert_eq!(EditDistance::bytes(b"abc", b"abc"), 0);
        assert_eq!(EditDistance::bytes(b"abc", b"adc"), 1);
        assert_eq!(EditDistance::bytes(b"", b"xyz"), 3);
    }

    #[test]
    fn test_edit_distance_similarity() {
        let s = EditDistance::similarity("hello", "hello");
        assert!(s.is_exact());

        let s2 = EditDistance::similarity("hello", "hxllo");
        assert!(s2.value() > 0.5);

        let s3 = EditDistance::similarity("", "");
        assert!(s3.is_exact());
    }

    #[test]
    fn test_token_matcher_identical() {
        let matcher = TokenMatcher::new();
        let s = matcher.similarity("hello world", "hello world");
        assert!(s.is_exact());
    }

    #[test]
    fn test_token_matcher_case_insensitive() {
        let matcher = TokenMatcher::new().case_insensitive(true);
        let s = matcher.similarity("Hello World", "hello world");
        assert!(s.is_exact());
    }

    #[test]
    fn test_token_matcher_partial() {
        let matcher = TokenMatcher::new();
        let s = matcher.similarity("the quick brown fox", "the quick red fox");
        assert!(s.value() > 0.5);
        assert!(!s.is_exact());
    }

    #[test]
    fn test_bigram_similarity_identical() {
        let s = BigramSimilarity::similarity("night", "night");
        assert!(s.is_exact());
    }

    #[test]
    fn test_bigram_similarity_similar() {
        let s = BigramSimilarity::similarity("night", "nacht");
        assert!(s.value() > 0.0);
        assert!(!s.is_exact());
    }

    #[test]
    fn test_hamming_distance_equal() {
        assert_eq!(hamming_distance(b"abc", b"abc"), Some(0));
    }

    #[test]
    fn test_hamming_distance_different() {
        assert_eq!(hamming_distance(b"abc", b"axc"), Some(1));
    }

    #[test]
    fn test_hamming_distance_length_mismatch() {
        assert_eq!(hamming_distance(b"ab", b"abc"), None);
    }

    #[test]
    fn test_hamming_similarity() {
        let s = hamming_similarity(b"abcd", b"abcd").expect("operation should succeed");
        assert!(s.is_exact());

        let s2 = hamming_similarity(b"abcd", b"axyd").expect("operation should succeed");
        assert!((s2.value() - 0.5).abs() < f64::EPSILON);
    }

    // ---- FilenameMatcher tests ----

    #[test]
    fn test_filename_normalize_basic() {
        let n = FilenameMatcher::normalize("The.Movie.2024.1080p.x264.mkv");
        assert_eq!(n, "the movie 2024");
    }

    #[test]
    fn test_filename_normalize_strips_extension() {
        let n = FilenameMatcher::normalize("video.mp4");
        assert_eq!(n, "video");
    }

    #[test]
    fn test_filename_normalize_strips_directory() {
        let n = FilenameMatcher::normalize("/path/to/video.mp4");
        assert_eq!(n, "video");
    }

    #[test]
    fn test_filename_normalize_codec_tags() {
        let n = FilenameMatcher::normalize("Movie.2024.h265.AAC.BluRay.mp4");
        assert_eq!(n, "movie 2024");
    }

    #[test]
    fn test_filename_matcher_identical() {
        let m = FilenameMatcher::new(0.8);
        let s = m.similarity("The.Movie.2024.mkv", "The.Movie.2024.mkv");
        assert!(s.is_exact());
    }

    #[test]
    fn test_filename_matcher_same_content_different_codec() {
        let m = FilenameMatcher::new(0.8);
        let s = m.similarity(
            "The.Movie.2024.1080p.x264.mkv",
            "The.Movie.2024.720p.x265.mp4",
        );
        assert!(s.meets_threshold(0.8), "Score was {}", s.value());
    }

    #[test]
    fn test_filename_matcher_different_movies() {
        let m = FilenameMatcher::new(0.8);
        let s = m.similarity("Inception.2010.mkv", "Interstellar.2014.mkv");
        assert!(!s.meets_threshold(0.8));
    }

    #[test]
    fn test_filename_matcher_is_match() {
        let m = FilenameMatcher::new(0.9);
        assert!(m.is_match("movie.1080p.mkv", "movie.720p.mp4"));
    }

    #[test]
    fn test_filename_matcher_find_matching_pairs() {
        let m = FilenameMatcher::new(0.8);
        let names = [
            "The.Movie.2024.1080p.mkv",
            "The.Movie.2024.720p.mp4",
            "Totally.Different.2023.mkv",
        ];
        let pairs = m.find_matching_pairs(&names);
        // First two should match, third should not match either
        assert!(pairs.iter().any(|(i, j, _)| *i == 0 && *j == 1));
        assert!(!pairs.iter().any(|(_, j, _)| *j == 2));
    }

    #[test]
    fn test_filename_matcher_empty_strings() {
        let m = FilenameMatcher::new(0.5);
        let s = m.similarity("", "");
        assert!(s.is_exact());
    }

    #[test]
    fn test_filename_matcher_default() {
        let m = FilenameMatcher::default();
        assert!((m.threshold() - 0.80).abs() < f64::EPSILON);
    }

    #[test]
    fn test_filename_matcher_custom_weights() {
        let m = FilenameMatcher::with_weights(0.7, 1.0, 0.0, 0.0);
        // With 100% edit weight, edit distance should dominate
        let s = m.similarity("hello.mp4", "hello.mp4");
        assert!(s.is_exact());
    }

    #[test]
    fn test_filename_normalize_preserves_year() {
        let n = FilenameMatcher::normalize("Movie.Title.2024.Remaster.mkv");
        // "remaster" is noise, "2024" should remain
        assert!(n.contains("2024"));
        assert!(!n.contains("remaster"));
    }

    #[test]
    fn test_remove_noise_token_boundary() {
        let result = remove_noise_token("test1080pin", "1080p");
        // "1080p" is adjacent to alphanumeric chars so should NOT be removed
        assert_eq!(result, "test1080pin");
    }

    #[test]
    fn test_remove_noise_token_standalone() {
        let result = remove_noise_token("test.1080p.file", "1080p");
        assert!(!result.contains("1080p"));
    }
}
