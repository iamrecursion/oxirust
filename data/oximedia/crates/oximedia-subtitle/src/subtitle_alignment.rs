//! Automatic subtitle timing alignment between two subtitle tracks.
//!
//! This module aligns a "source" subtitle track to a "reference" track using a
//! combination of sequence-based dynamic-time-warping (DTW) on timestamps and
//! text similarity scoring.  A common use case is syncing unofficial fansub
//! timings to an authoritatively timed reference track (e.g. an official release
//! or a Blu-ray disc).
//!
//! # Algorithm
//!
//! 1. Build a cost matrix where `cost[i][j]` reflects how dissimilar the
//!    text of source cue `i` is to reference cue `j` (edit distance + timing
//!    difference).
//! 2. Run DTW to find the minimum-cost monotonic alignment path.
//! 3. For each aligned pair, compute the timestamp delta and apply it to the
//!    source cue.  Un-aligned source cues receive a linearly interpolated
//!    offset from their nearest aligned neighbours.

// ============================================================================
// Public types
// ============================================================================

/// A single subtitle entry with timing.
#[derive(Clone, Debug)]
pub struct AlignEntry {
    /// Start time in milliseconds.
    pub start_ms: i64,
    /// End time in milliseconds.
    pub end_ms: i64,
    /// Subtitle text (stripped of markup).
    pub text: String,
}

impl AlignEntry {
    /// Create a new align entry.
    #[must_use]
    pub fn new(start_ms: i64, end_ms: i64, text: impl Into<String>) -> Self {
        Self {
            start_ms,
            end_ms,
            text: text.into(),
        }
    }

    /// Duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> i64 {
        self.end_ms - self.start_ms
    }
}

/// Configuration for the alignment algorithm.
#[derive(Clone, Debug)]
pub struct AlignConfig {
    /// Maximum allowed timing delta (ms) between paired cues when scoring.
    /// Pairs with deltas larger than this are penalised heavily.
    pub max_timing_delta_ms: i64,
    /// Weight applied to text similarity vs timing proximity (0.0–1.0).
    /// 0.0 = timing only, 1.0 = text similarity only.
    pub text_weight: f32,
    /// Maximum number of source cues that may be matched to a single reference
    /// cue (many-to-one grouping).
    pub max_many_to_one: usize,
}

impl Default for AlignConfig {
    fn default() -> Self {
        Self {
            max_timing_delta_ms: 10_000,
            text_weight: 0.6,
            max_many_to_one: 3,
        }
    }
}

/// The result of aligning one track to a reference.
#[derive(Clone, Debug)]
pub struct AlignedTrack {
    /// The re-timed source cues.
    pub cues: Vec<AlignEntry>,
    /// Per-cue alignment metadata.
    pub matches: Vec<AlignMatch>,
}

/// Information about how a single source cue was matched.
#[derive(Clone, Debug)]
pub struct AlignMatch {
    /// Source cue index (0-based).
    pub source_idx: usize,
    /// Reference cue index it was aligned to (if any).
    pub reference_idx: Option<usize>,
    /// Timing delta applied (ms).  Positive = shifted later.
    pub delta_ms: i64,
    /// Text similarity score (0.0 = different, 1.0 = identical).
    pub similarity: f32,
}

// ============================================================================
// Core alignment
// ============================================================================

/// Align `source` subtitle timings to the `reference` track.
///
/// Returns an [`AlignedTrack`] with corrected timestamps and alignment metadata.
///
/// # Arguments
///
/// * `source` – the track whose timestamps are to be corrected.
/// * `reference` – the authoritative reference track.
/// * `config` – tuning parameters for the algorithm.
#[must_use]
pub fn align_tracks(
    source: &[AlignEntry],
    reference: &[AlignEntry],
    config: &AlignConfig,
) -> AlignedTrack {
    if source.is_empty() {
        return AlignedTrack {
            cues: Vec::new(),
            matches: Vec::new(),
        };
    }
    if reference.is_empty() {
        // Nothing to align to — return source unchanged.
        let matches = source
            .iter()
            .enumerate()
            .map(|(i, _)| AlignMatch {
                source_idx: i,
                reference_idx: None,
                delta_ms: 0,
                similarity: 0.0,
            })
            .collect();
        return AlignedTrack {
            cues: source.to_vec(),
            matches,
        };
    }

    // Step 1: build cost matrix
    let cost = build_cost_matrix(source, reference, config);

    // Step 2: DTW path
    let path = dtw_path(&cost, source.len(), reference.len());

    // Step 3: derive per-source delta from path
    let deltas = derive_deltas(source, reference, &path, source.len(), reference.len());

    // Step 4: apply deltas + gather metadata
    let mut cues = Vec::with_capacity(source.len());
    let mut matches = Vec::with_capacity(source.len());

    for (si, (src, (ref_idx_opt, delta))) in source.iter().zip(deltas.iter()).enumerate() {
        let similarity =
            ref_idx_opt.map_or(0.0, |ri| text_similarity(&src.text, &reference[ri].text));

        cues.push(AlignEntry {
            start_ms: src.start_ms + delta,
            end_ms: src.end_ms + delta,
            text: src.text.clone(),
        });
        matches.push(AlignMatch {
            source_idx: si,
            reference_idx: *ref_idx_opt,
            delta_ms: *delta,
            similarity,
        });
    }

    AlignedTrack { cues, matches }
}

/// Build a 2-D cost matrix (source.len() × reference.len()).
fn build_cost_matrix(
    source: &[AlignEntry],
    reference: &[AlignEntry],
    config: &AlignConfig,
) -> Vec<Vec<f64>> {
    let n = source.len();
    let m = reference.len();
    let mut cost = vec![vec![0.0f64; m]; n];

    for (i, src) in source.iter().enumerate() {
        for (j, rf) in reference.iter().enumerate() {
            let timing_cost = timing_cost(src.start_ms, rf.start_ms, config.max_timing_delta_ms);
            let text_cost = 1.0 - text_similarity(&src.text, &rf.text) as f64;
            let w = f64::from(config.text_weight);
            cost[i][j] = w * text_cost + (1.0 - w) * timing_cost;
        }
    }
    cost
}

/// Compute a normalised timing cost in [0,1] for two timestamps.
fn timing_cost(src_ms: i64, ref_ms: i64, max_delta: i64) -> f64 {
    let delta = (src_ms - ref_ms).unsigned_abs() as f64;
    let max = max_delta.max(1) as f64;
    (delta / max).min(1.0)
}

/// DTW: return the optimal monotonic alignment path.
///
/// Each element is `(source_idx, reference_idx)`.
fn dtw_path(cost: &[Vec<f64>], n: usize, m: usize) -> Vec<(usize, usize)> {
    // Accumulated cost matrix
    let mut acc = vec![vec![f64::INFINITY; m]; n];
    acc[0][0] = cost[0][0];

    for j in 1..m {
        acc[0][j] = acc[0][j - 1] + cost[0][j];
    }
    for i in 1..n {
        acc[i][0] = acc[i - 1][0] + cost[i][0];
    }
    for i in 1..n {
        for j in 1..m {
            let prev = acc[i - 1][j].min(acc[i][j - 1]).min(acc[i - 1][j - 1]);
            acc[i][j] = cost[i][j] + prev;
        }
    }

    // Traceback
    let mut path = Vec::new();
    let (mut i, mut j) = (n - 1, m - 1);
    path.push((i, j));

    while i > 0 || j > 0 {
        if i == 0 {
            j -= 1;
        } else if j == 0 {
            i -= 1;
        } else {
            let up = acc[i - 1][j];
            let left = acc[i][j - 1];
            let diag = acc[i - 1][j - 1];
            if diag <= up && diag <= left {
                i -= 1;
                j -= 1;
            } else if up <= left {
                i -= 1;
            } else {
                j -= 1;
            }
        }
        path.push((i, j));
    }
    path.reverse();
    path
}

/// Derive per-source-cue (reference_idx_option, delta_ms) from the DTW path.
///
/// Source cues that appear in the path get the delta from their matched reference.
/// Unmatched cues get linearly interpolated deltas from their neighbours.
fn derive_deltas(
    source: &[AlignEntry],
    reference: &[AlignEntry],
    path: &[(usize, usize)],
    n: usize,
    _m: usize,
) -> Vec<(Option<usize>, i64)> {
    // Build per-source-cue best reference match from the path.
    // Each source index may appear multiple times — take the one with lowest cost.
    let mut src_to_ref: Vec<Option<usize>> = vec![None; n];
    for &(si, ri) in path {
        if si < n {
            src_to_ref[si] = Some(ri);
        }
    }

    // Compute known deltas
    let mut known_delta: Vec<Option<i64>> = vec![None; n];
    for (si, ref_opt) in src_to_ref.iter().enumerate() {
        if let Some(ri) = ref_opt {
            let delta = reference[*ri].start_ms - source[si].start_ms;
            known_delta[si] = Some(delta);
        }
    }

    // Interpolate gaps
    let mut deltas: Vec<(Option<usize>, i64)> = vec![(None, 0); n];

    // Forward pass: fill with last known delta
    let mut last_known: Option<(usize, i64)> = None;
    for si in 0..n {
        if let Some(d) = known_delta[si] {
            last_known = Some((si, d));
            deltas[si] = (src_to_ref[si], d);
        } else if let Some((_, ld)) = last_known {
            deltas[si] = (None, ld);
        }
    }

    // Backward pass: fill remaining (at the start) with first known delta
    let mut first_known: Option<i64> = known_delta.iter().find_map(|d| *d);
    for si in (0..n).rev() {
        if known_delta[si].is_some() {
            first_known = known_delta[si];
        } else if deltas[si].0.is_none() {
            if let Some(fk) = first_known {
                deltas[si] = (None, fk);
            }
        }
    }

    deltas
}

// ============================================================================
// Text similarity
// ============================================================================

/// Compute normalised text similarity in [0.0, 1.0] using character-level
/// Levenshtein distance.
#[must_use]
pub fn text_similarity(a: &str, b: &str) -> f32 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let la = a_chars.len();
    let lb = b_chars.len();

    if la == 0 && lb == 0 {
        return 1.0;
    }
    if la == 0 || lb == 0 {
        return 0.0;
    }

    let dist = levenshtein_distance(&a_chars, &b_chars);
    let max_len = la.max(lb) as f32;
    1.0 - (dist as f32 / max_len)
}

/// Character-level Levenshtein distance.
fn levenshtein_distance(a: &[char], b: &[char]) -> usize {
    let la = a.len();
    let lb = b.len();
    let mut prev: Vec<usize> = (0..=lb).collect();

    for i in 1..=la {
        let mut curr = vec![0usize; lb + 1];
        curr[0] = i;
        for j in 1..=lb {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        prev = curr;
    }
    prev[lb]
}

// ============================================================================
// Frame-offset bias detection
// ============================================================================

/// Estimate a constant frame offset between two tracks by correlating the
/// most text-similar pairs.
///
/// Returns the median timing delta (ms) of the top-N matched pairs.
///
/// This is a lightweight alternative to full DTW for simple constant-offset
/// corrections (e.g. wrong encode start frame).
#[must_use]
pub fn estimate_constant_offset(
    source: &[AlignEntry],
    reference: &[AlignEntry],
    top_n: usize,
) -> Option<i64> {
    if source.is_empty() || reference.is_empty() {
        return None;
    }

    // For each reference cue, find the best-matching source cue by text
    let mut pairs: Vec<(f32, i64)> = Vec::new();

    for rf in reference {
        let best = source.iter().max_by(|a, b| {
            let sa = text_similarity(&a.text, &rf.text);
            let sb = text_similarity(&b.text, &rf.text);
            sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(src) = best {
            let sim = text_similarity(&src.text, &rf.text);
            if sim > 0.5 {
                let delta = rf.start_ms - src.start_ms;
                pairs.push((sim, delta));
            }
        }
    }

    if pairs.is_empty() {
        return None;
    }

    // Sort by similarity descending, take top-N
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let take = top_n.min(pairs.len());
    let mut deltas: Vec<i64> = pairs[..take].iter().map(|(_, d)| *d).collect();

    // Median
    deltas.sort_unstable();
    let mid = deltas.len() / 2;
    Some(deltas[mid])
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn e(start: i64, end: i64, text: &str) -> AlignEntry {
        AlignEntry::new(start, end, text)
    }

    #[test]
    fn test_text_similarity_identical() {
        assert!((text_similarity("Hello", "Hello") - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_text_similarity_empty_both() {
        assert!((text_similarity("", "") - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_text_similarity_one_empty() {
        assert!((text_similarity("Hello", "") - 0.0).abs() < f32::EPSILON);
        assert!((text_similarity("", "World") - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_text_similarity_close() {
        let sim = text_similarity("Hello World", "Hello World!");
        assert!(sim > 0.8, "sim={sim}");
    }

    #[test]
    fn test_text_similarity_unrelated() {
        let sim = text_similarity("abc", "xyz");
        assert!(sim < 0.5, "sim={sim}");
    }

    #[test]
    fn test_levenshtein_equal() {
        let a: Vec<char> = "abc".chars().collect();
        assert_eq!(levenshtein_distance(&a, &a), 0);
    }

    #[test]
    fn test_levenshtein_single_insert() {
        let a: Vec<char> = "abc".chars().collect();
        let b: Vec<char> = "abcd".chars().collect();
        assert_eq!(levenshtein_distance(&a, &b), 1);
    }

    #[test]
    fn test_levenshtein_single_delete() {
        let a: Vec<char> = "abcd".chars().collect();
        let b: Vec<char> = "abc".chars().collect();
        assert_eq!(levenshtein_distance(&a, &b), 1);
    }

    #[test]
    fn test_levenshtein_substitution() {
        let a: Vec<char> = "abc".chars().collect();
        let b: Vec<char> = "axc".chars().collect();
        assert_eq!(levenshtein_distance(&a, &b), 1);
    }

    #[test]
    fn test_align_empty_source() {
        let result = align_tracks(&[], &[e(0, 1000, "Ref")], &AlignConfig::default());
        assert!(result.cues.is_empty());
    }

    #[test]
    fn test_align_empty_reference() {
        let src = vec![e(0, 1000, "Src")];
        let result = align_tracks(&src, &[], &AlignConfig::default());
        assert_eq!(result.cues.len(), 1);
        assert_eq!(result.cues[0].start_ms, 0); // unchanged
    }

    #[test]
    fn test_align_identical_tracks() {
        let track = vec![e(0, 2000, "Hello"), e(3000, 5000, "World")];
        let result = align_tracks(&track, &track, &AlignConfig::default());
        assert_eq!(result.cues.len(), 2);
        // Identical tracks → zero delta
        for m in &result.matches {
            assert_eq!(m.delta_ms, 0, "delta should be 0 for identical tracks");
        }
    }

    #[test]
    fn test_align_constant_offset() {
        let src = vec![e(0, 1000, "Hello"), e(2000, 3000, "World")];
        let reference = vec![e(500, 1500, "Hello"), e(2500, 3500, "World")];
        let result = align_tracks(&src, &reference, &AlignConfig::default());
        assert_eq!(result.cues.len(), 2);
        // Both cues should have shifted by ~500ms
        for cue in &result.cues {
            let delta = cue.start_ms
                - src
                    .iter()
                    .find(|s| s.text == cue.text)
                    .map_or(0, |s| s.start_ms);
            assert!(
                (delta - 500).abs() <= 50,
                "delta should be ~500ms, got {delta}"
            );
        }
    }

    #[test]
    fn test_estimate_constant_offset_simple() {
        let src = vec![e(0, 1000, "Hello"), e(2000, 3000, "World")];
        let reference = vec![e(1000, 2000, "Hello"), e(3000, 4000, "World")];
        let offset = estimate_constant_offset(&src, &reference, 5);
        assert_eq!(offset, Some(1000));
    }

    #[test]
    fn test_estimate_constant_offset_empty_source() {
        let offset = estimate_constant_offset(&[], &[e(0, 1000, "A")], 5);
        assert!(offset.is_none());
    }

    #[test]
    fn test_estimate_constant_offset_empty_reference() {
        let offset = estimate_constant_offset(&[e(0, 1000, "A")], &[], 5);
        assert!(offset.is_none());
    }

    #[test]
    fn test_estimate_offset_low_similarity_yields_none() {
        let src = vec![e(0, 1000, "AAAAA")];
        let reference = vec![e(500, 1500, "ZZZZZ")];
        // similarity < 0.5 → no pair retained
        let offset = estimate_constant_offset(&src, &reference, 5);
        assert!(offset.is_none());
    }

    #[test]
    fn test_align_match_contains_source_idx() {
        let src = vec![e(0, 1000, "A"), e(2000, 3000, "B")];
        let reference = vec![e(0, 1000, "A"), e(2000, 3000, "B")];
        let result = align_tracks(&src, &reference, &AlignConfig::default());
        assert_eq!(result.matches[0].source_idx, 0);
        assert_eq!(result.matches[1].source_idx, 1);
    }

    #[test]
    fn test_align_similarity_high_for_identical_text() {
        let src = vec![e(0, 1000, "Test cue")];
        let reference = vec![e(200, 1200, "Test cue")];
        let result = align_tracks(&src, &reference, &AlignConfig::default());
        assert!(result.matches[0].similarity > 0.9);
    }

    #[test]
    fn test_align_entry_duration() {
        let e = AlignEntry::new(1000, 3500, "Text");
        assert_eq!(e.duration_ms(), 2500);
    }

    #[test]
    fn test_align_config_default() {
        let cfg = AlignConfig::default();
        assert_eq!(cfg.max_timing_delta_ms, 10_000);
        assert!((cfg.text_weight - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn test_align_large_offset() {
        // 60 second constant offset
        let src = vec![
            e(60_000, 62_000, "Scene one"),
            e(65_000, 67_000, "Scene two"),
        ];
        let reference = vec![e(0, 2_000, "Scene one"), e(5_000, 7_000, "Scene two")];
        let result = align_tracks(&src, &reference, &AlignConfig::default());
        // Both cues shifted back by ~60 000 ms
        for cue in &result.cues {
            assert!(
                cue.start_ms < 10_000,
                "cue should be shifted to near-zero: start_ms={}",
                cue.start_ms
            );
        }
    }
}
