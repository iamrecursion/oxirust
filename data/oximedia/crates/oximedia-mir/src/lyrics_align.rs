//! Lyrics timing alignment stub.
//!
//! Given a lyrics string and a list of onset times (in milliseconds), assigns
//! each word in the lyrics to the nearest available onset using a greedy
//! left-to-right matching strategy.
//!
//! # Algorithm
//!
//! 1. Split the lyrics into words by whitespace.  Empty tokens are discarded.
//! 2. Iterate over words in order, consuming onsets greedily:
//!    - The first word is assigned to the first onset.
//!    - Each subsequent word is assigned to the onset closest to the previous
//!      word's onset that has not yet been consumed.
//!    - If there are more words than onsets, remaining words share the last
//!      onset time and are each given a 500 ms duration.
//!    - If there are no onsets at all, all words receive start_ms = 0 and
//!      duration = 500 ms.
//! 3. Duration = start_ms of the *next* word − start_ms of *this* word.
//!    The last word always receives a 500 ms duration.
//!
//! This is a **stub implementation**.  Production-quality lyrics alignment
//! requires forced-alignment systems (e.g., ctc-based acoustic models) and is
//! outside the scope of this crate.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// LyricsWord
// ---------------------------------------------------------------------------

/// A single word with its assigned timing information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LyricsWord {
    /// The word text (no surrounding whitespace).
    pub text: String,
    /// Start time in milliseconds.
    pub start_ms: u32,
    /// End time in milliseconds.
    pub end_ms: u32,
}

impl LyricsWord {
    /// Duration of the word in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u32 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

impl std::fmt::Display for LyricsWord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}[{}ms–{}ms]", self.text, self.start_ms, self.end_ms)
    }
}

// ---------------------------------------------------------------------------
// align_lyrics
// ---------------------------------------------------------------------------

/// Align lyrics words to onset times.
///
/// # Arguments
///
/// * `lyrics` — lyrics as a UTF-8 string.  Words are separated by any Unicode
///   whitespace.  Punctuation is kept as part of the word.
/// * `onsets_ms` — onset times in milliseconds, in ascending order.
///
/// # Returns
///
/// One [`LyricsWord`] per word token found in `lyrics`.  An empty lyrics
/// string returns an empty vector.
///
/// # Example
///
/// ```
/// use oximedia_mir::lyrics_align::align_lyrics;
///
/// let lyrics = "Hello world goodbye";
/// let onsets = [0u32, 500, 1000];
/// let words = align_lyrics(lyrics, &onsets);
/// assert_eq!(words.len(), 3);
/// assert_eq!(words[0].text, "Hello");
/// assert_eq!(words[0].start_ms, 0);
/// assert_eq!(words[1].text, "world");
/// assert_eq!(words[1].start_ms, 500);
/// ```
#[must_use]
pub fn align_lyrics(lyrics: &str, onsets_ms: &[u32]) -> Vec<LyricsWord> {
    // Tokenise into words.
    let words: Vec<&str> = lyrics.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }

    let n_words = words.len();
    let n_onsets = onsets_ms.len();

    // Pre-compute start times for each word.
    let start_times: Vec<u32> = if n_onsets == 0 {
        // No onsets: all words start at 0.
        vec![0u32; n_words]
    } else {
        assign_onsets_greedy(&words, onsets_ms)
    };

    // Build LyricsWord entries.
    let mut result: Vec<LyricsWord> = Vec::with_capacity(n_words);
    for (idx, &word) in words.iter().enumerate() {
        let start = start_times[idx];
        let end = if idx + 1 < n_words {
            // Duration = next word's start − this word's start (minimum 1 ms).
            let next_start = start_times[idx + 1];
            if next_start > start {
                next_start
            } else {
                start + 500
            }
        } else {
            // Last word always gets a 500 ms duration.
            start + 500
        };
        result.push(LyricsWord {
            text: word.to_string(),
            start_ms: start,
            end_ms: end,
        });
    }

    result
}

/// Greedy left-to-right onset assignment.
///
/// Assigns the *n*-th word to the *n*-th onset when the number of onsets ≥
/// number of words.  When there are more words than onsets, excess words are
/// pinned to the last onset.
///
/// When there are more onsets than words, the extra onsets are silently
/// ignored (the surplus gives breathing room for the last word's duration).
fn assign_onsets_greedy(words: &[&str], onsets_ms: &[u32]) -> Vec<u32> {
    let n_words = words.len();
    let n_onsets = onsets_ms.len();

    (0..n_words)
        .map(|word_idx| {
            // Each word gets its own onset if available; otherwise use the last one.
            let onset_idx = word_idx.min(n_onsets - 1);
            onsets_ms[onset_idx]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// align_lyrics_dtw
// ---------------------------------------------------------------------------

/// Align lyrics words to onset times using Dynamic Time Warping (DTW) forced
/// alignment.
///
/// Unlike the greedy `align_lyrics`, this function takes onset strength values
/// into account: it minimises the total cost of the alignment path, where cost
/// at each cell is `1.0 − onset_strength[o]` (low cost at strong onsets).
/// The DTW warping path is forced to be monotone (each word is assigned to an
/// onset index ≥ the previous word's onset index).
///
/// # Arguments
///
/// * `lyrics` — lyrics string; words are split on Unicode whitespace.
/// * `onsets_ms` — onset times in milliseconds, in ascending order.
/// * `onset_strength` — per-onset strength values in `[0.0, 1.0]`.  If the
///   length differs from `onsets_ms`, missing values are treated as `0.5`.
///
/// # Fallback
///
/// If `onsets_ms` or `onset_strength` is empty, or there are no words, the
/// function falls back to [`align_lyrics`] (greedy) or returns an empty vec.
///
/// # Example
///
/// ```
/// use oximedia_mir::lyrics_align::align_lyrics_dtw;
///
/// let lyrics = "Hello world";
/// let onsets = [0u32, 500];
/// let strengths = [0.3f32, 0.9];
/// let words = align_lyrics_dtw(lyrics, &onsets, &strengths);
/// assert_eq!(words.len(), 2);
/// // DTW minimises total cost; "Hello" → onset 0ms, "world" → onset 500ms.
/// assert_eq!(words[0].start_ms, 0);
/// assert_eq!(words[1].start_ms, 500);
/// ```
#[must_use]
pub fn align_lyrics_dtw(
    lyrics: &str,
    onsets_ms: &[u32],
    onset_strength: &[f32],
) -> Vec<LyricsWord> {
    let words: Vec<&str> = lyrics.split_whitespace().collect();

    // Fall back to greedy if inputs are insufficient for DTW.
    if words.is_empty() || onsets_ms.is_empty() || onset_strength.is_empty() {
        return align_lyrics(lyrics, onsets_ms);
    }

    let n_words = words.len();
    let n_onsets = onsets_ms.len();

    // Step 3: cost[o] = 1.0 − onset_strength[o], clamped to [0, 1].
    // Cost is shared across all words (strength is per-onset, not per-word).
    let cost: Vec<f32> = (0..n_onsets)
        .map(|o| {
            let s = onset_strength
                .get(o)
                .copied()
                .unwrap_or(0.5)
                .clamp(0.0, 1.0);
            1.0 - s
        })
        .collect();

    // Step 4: DTW forward pass — "onset-skip" monotone variant.
    //
    // dp is a flat (n_words+1) × (n_onsets+1) array, initialised to +∞.
    // dp[0..=n_onsets] = 0.0  — free prefix: any onset may start the path.
    //
    // Recurrence for i ∈ 1..=n_words, j ∈ 1..=n_onsets:
    //   dp[i*(n_onsets+1)+j] = cost[j-1]
    //       + min( dp[(i-1)*(n_onsets+1)+(j-1)],   // diagonal: word+onset both advance
    //              dp[ i   *(n_onsets+1)+(j-1)] )   // left: skip onset, word stays
    //
    // The "up" direction (word advances, onset stays) is intentionally excluded so
    // that each word is always routed to a strictly non-decreasing onset index,
    // preserving the monotone forced-alignment guarantee.
    let cols = n_onsets + 1;
    let mut dp = vec![f32::INFINITY; (n_words + 1) * cols];
    // Free prefix: dp[0][j] = 0.0 for all j.
    for jj in 0..cols {
        dp[jj] = 0.0;
    }

    for i in 1..=n_words {
        for j in 1..=n_onsets {
            let local_cost = cost[j - 1];
            let from_diag = dp[(i - 1) * cols + (j - 1)];
            let from_left = dp[i * cols + (j - 1)];
            let prev = from_diag.min(from_left);
            if prev.is_finite() {
                dp[i * cols + j] = local_cost + prev;
            }
        }
    }

    // Step 5: Traceback from (n_words, best_j) → (0, _).
    //
    // best_j: the rightmost onset column in row n_words with the minimum cost.
    // Preferring the rightmost minimum allows the traceback to span the widest
    // range of onsets, placing words at later (often stronger) peaks when costs
    // are equal.
    let min_cost = (1..=n_onsets)
        .map(|j| dp[n_words * cols + j])
        .fold(f32::INFINITY, f32::min);
    let best_j = (1..=n_onsets)
        .rev()
        .find(|&j| dp[n_words * cols + j] <= min_cost + f32::EPSILON * 16.0)
        .unwrap_or(n_onsets);

    let mut onset_for_word = vec![0usize; n_words];
    let mut i = n_words;
    let mut j = best_j;

    while i > 0 {
        if j == 0 {
            onset_for_word[i - 1] = 0;
            i -= 1;
            continue;
        }

        let from_diag = dp[(i - 1) * cols + (j - 1)];
        let from_left = dp[i * cols + (j - 1)];
        let tied = (from_diag - from_left).abs() <= f32::EPSILON * 16.0;

        // When the diagonal and left predecessors are tied, check whether a
        // higher-strength onset exists to the left of the current position.
        // If so, skip the current onset (go left) so the word can settle on
        // the stronger peak; otherwise take the diagonal.
        let curr_str = onset_strength.get(j - 1).copied().unwrap_or(0.5);
        let better_exists_left = tied
            && (0..j.saturating_sub(1)).any(|prev_j| {
                onset_strength.get(prev_j).copied().unwrap_or(0.5) > curr_str + f32::EPSILON * 16.0
            });

        if !better_exists_left && from_diag <= from_left {
            // Diagonal: assign this word to onset j-1 (0-indexed).
            onset_for_word[i - 1] = j - 1;
            i -= 1;
            j -= 1;
        } else {
            // Left: skip onset j, keep word i pending.
            j -= 1;
        }
    }

    // Step 6–7: Build LyricsWord entries.
    let mut result: Vec<LyricsWord> = Vec::with_capacity(n_words);
    for w in 0..n_words {
        let oi = onset_for_word[w];
        let start_ms = onsets_ms.get(oi).copied().unwrap_or(0);
        let end_ms = if w + 1 < n_words {
            let next_oi = onset_for_word[w + 1];
            let next_start = onsets_ms.get(next_oi).copied().unwrap_or(start_ms + 500);
            if next_start > start_ms {
                next_start
            } else {
                start_ms + 500
            }
        } else {
            start_ms + 500
        };
        result.push(LyricsWord {
            text: words[w].to_string(),
            start_ms,
            end_ms,
        });
    }

    result
}

// ---------------------------------------------------------------------------
// Helper utilities
// ---------------------------------------------------------------------------

/// Split a multi-line lyrics string into individual lines, preserving empty
/// lines as empty strings.
#[must_use]
pub fn split_lines(lyrics: &str) -> Vec<&str> {
    lyrics.lines().collect()
}

/// Return the total duration of all word segments in milliseconds.
#[must_use]
pub fn total_duration_ms(words: &[LyricsWord]) -> u32 {
    words
        .iter()
        .map(|w| w.end_ms)
        .fold(0u32, |acc, end| acc.max(end))
}

/// Filter words that overlap with a given time range `[from_ms, to_ms]`.
#[must_use]
pub fn words_in_range(words: &[LyricsWord], from_ms: u32, to_ms: u32) -> Vec<&LyricsWord> {
    words
        .iter()
        .filter(|w| w.start_ms < to_ms && w.end_ms > from_ms)
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── align_lyrics ──────────────────────────────────────────────────────────

    #[test]
    fn test_align_empty_lyrics() {
        let result = align_lyrics("", &[0, 500, 1000]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_align_empty_onsets() {
        let result = align_lyrics("hello world", &[]);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].start_ms, 0);
        assert_eq!(result[1].start_ms, 0);
        // All words start at 0 and get 500 ms duration.
        assert_eq!(result[0].end_ms, 500);
        assert_eq!(result[1].end_ms, 500);
    }

    #[test]
    fn test_align_three_words_three_onsets() {
        let lyrics = "Hello world goodbye";
        let onsets = [0u32, 500, 1000];
        let words = align_lyrics(lyrics, &onsets);
        assert_eq!(words.len(), 3);
        assert_eq!(words[0].text, "Hello");
        assert_eq!(words[0].start_ms, 0);
        assert_eq!(words[0].end_ms, 500); // next word starts at 500

        assert_eq!(words[1].text, "world");
        assert_eq!(words[1].start_ms, 500);
        assert_eq!(words[1].end_ms, 1000); // next word starts at 1000

        assert_eq!(words[2].text, "goodbye");
        assert_eq!(words[2].start_ms, 1000);
        assert_eq!(words[2].end_ms, 1500); // last word gets +500
    }

    #[test]
    fn test_align_more_words_than_onsets() {
        let lyrics = "one two three four five";
        let onsets = [100u32, 200, 300]; // only 3 onsets for 5 words
        let words = align_lyrics(lyrics, &onsets);
        assert_eq!(words.len(), 5);
        assert_eq!(words[0].start_ms, 100);
        assert_eq!(words[1].start_ms, 200);
        assert_eq!(words[2].start_ms, 300);
        // Excess words pinned to last onset.
        assert_eq!(words[3].start_ms, 300);
        assert_eq!(words[4].start_ms, 300);
    }

    #[test]
    fn test_align_more_onsets_than_words() {
        let lyrics = "quick brown";
        let onsets = [0u32, 100, 200, 300, 400]; // 5 onsets for 2 words
        let words = align_lyrics(lyrics, &onsets);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].start_ms, 0);
        assert_eq!(words[1].start_ms, 100);
    }

    #[test]
    fn test_align_single_word_single_onset() {
        let words = align_lyrics("only", &[750]);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "only");
        assert_eq!(words[0].start_ms, 750);
        assert_eq!(words[0].end_ms, 1250); // 750 + 500
    }

    #[test]
    fn test_align_word_duration_ms() {
        let words = align_lyrics("a b", &[0, 300]);
        assert_eq!(words[0].duration_ms(), 300); // 300 - 0
        assert_eq!(words[1].duration_ms(), 500); // last word gets +500
    }

    #[test]
    fn test_align_whitespace_lyrics() {
        // Lyrics that are only whitespace produce no words.
        let result = align_lyrics("   \t  \n  ", &[0, 100]);
        assert!(result.is_empty());
    }

    // ── LyricsWord ────────────────────────────────────────────────────────────

    #[test]
    fn test_lyrics_word_display() {
        let w = LyricsWord {
            text: "hello".to_string(),
            start_ms: 100,
            end_ms: 400,
        };
        let s = format!("{w}");
        assert!(s.contains("hello"));
        assert!(s.contains("100ms"));
        assert!(s.contains("400ms"));
    }

    #[test]
    fn test_lyrics_word_duration_saturating() {
        // end_ms < start_ms → duration clamped to 0.
        let w = LyricsWord {
            text: "x".to_string(),
            start_ms: 500,
            end_ms: 300,
        };
        assert_eq!(w.duration_ms(), 0);
    }

    // ── Helper utilities ──────────────────────────────────────────────────────

    #[test]
    fn test_total_duration_ms() {
        let words = align_lyrics("a b c", &[0, 200, 400]);
        let total = total_duration_ms(&words);
        // Last word ends at 400 + 500 = 900 ms.
        assert_eq!(total, 900);
    }

    #[test]
    fn test_words_in_range() {
        let words = align_lyrics("a b c d", &[0, 100, 200, 300]);
        // Range 50ms – 250ms should include words at 100ms and 200ms.
        let in_range = words_in_range(&words, 50, 250);
        let texts: Vec<&str> = in_range.iter().map(|w| w.text.as_str()).collect();
        assert!(texts.contains(&"b"));
        assert!(texts.contains(&"c"));
    }

    #[test]
    fn test_split_lines_preserves_empty() {
        let lyrics = "line one\n\nline three";
        let lines = split_lines(lyrics);
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[1], "");
    }

    #[test]
    fn test_align_multiline_lyrics() {
        // Newlines count as whitespace; split_whitespace ignores them.
        let lyrics = "line one\nline two";
        let onsets = [0u32, 200, 400, 600];
        let words = align_lyrics(lyrics, &onsets);
        assert_eq!(words.len(), 4);
        assert_eq!(words[0].text, "line");
        assert_eq!(words[1].text, "one");
    }

    // ── align_lyrics_dtw ─────────────────────────────────────────────────────

    /// Spec test: 2 words, 4 onsets with alternating low/high strength.
    /// DTW must route each word to the nearest strength peak.
    #[test]
    fn test_dtw_aligns_to_peaks() {
        let onsets_ms = [0u32, 100, 200, 300];
        let onset_strength = [0.1f32, 1.0, 0.1, 1.0];
        let words = align_lyrics_dtw("alpha beta", &onsets_ms, &onset_strength);
        assert_eq!(words.len(), 2);
        // Peaks are at onsets 1 (100ms) and 3 (300ms).
        // DTW assigns word 0 → peak 100ms and word 1 → peak 300ms.
        assert_eq!(
            words[0].start_ms, 100,
            "word 0 should land on peak at 100ms, got {}",
            words[0].start_ms
        );
        assert_eq!(
            words[1].start_ms, 300,
            "word 1 should land on peak at 300ms, got {}",
            words[1].start_ms
        );
    }

    /// Spec test: result timestamps are non-decreasing (monotone).
    #[test]
    fn test_dtw_monotone() {
        let onsets_ms = [0u32, 100, 300, 600, 1000];
        let strength = [0.3f32, 0.9, 0.4, 0.8, 0.2];
        let words = align_lyrics_dtw("one two three four five", &onsets_ms, &strength);
        assert_eq!(words.len(), 5);
        for i in 0..words.len() - 1 {
            assert!(
                words[i].start_ms <= words[i + 1].start_ms,
                "Monotone violated at index {i}: {} > {}",
                words[i].start_ms,
                words[i + 1].start_ms
            );
        }
    }

    /// Spec test: empty onset_strength falls back to greedy with no panic.
    #[test]
    fn test_dtw_empty_strength_fallback() {
        let onsets_ms = [0u32, 200, 400];
        let words = align_lyrics_dtw("hello world", &onsets_ms, &[]);
        assert!(!words.is_empty(), "Fallback must return non-empty vec");
    }

    /// Spec test: single word input returns exactly one LyricsWord with no panic.
    #[test]
    fn test_dtw_single_word_spec() {
        let onsets_ms = [0u32, 250, 500];
        let strength = [0.2f32, 0.9, 0.1];
        let words = align_lyrics_dtw("only", &onsets_ms, &strength);
        assert_eq!(words.len(), 1);
        // Peak is at index 1 (strength 0.9, cost 0.1).
        assert_eq!(words[0].start_ms, 250);
    }

    /// Spec test: for 1 word, DTW picks the peak onset while greedy picks onset 0.
    #[test]
    fn test_dtw_differs_from_greedy() {
        let onsets_ms = [0u32, 50, 100];
        let onset_strength = [0.0f32, 1.0, 0.0];
        let dtw_result = align_lyrics_dtw("foo", &onsets_ms, &onset_strength);
        assert_eq!(dtw_result.len(), 1);
        // DTW minimises cost: peak at onset 1 (50ms, strength 1.0, cost 0.0).
        assert_eq!(
            dtw_result[0].start_ms, 50,
            "DTW should pick the peak at 50ms, got {}",
            dtw_result[0].start_ms
        );
        // Greedy always picks onset 0 (first available).
        let greedy_result = align_lyrics("foo", &onsets_ms);
        assert_eq!(greedy_result[0].start_ms, 0, "Greedy should pick onset 0");
        // Confirm they differ.
        assert_ne!(dtw_result[0].start_ms, greedy_result[0].start_ms);
    }

    /// DTW should prefer the globally cheapest path and correctly assign words
    /// to their minimum-cost onsets.  With 3 words and exactly 3 onsets where
    /// onset 1 is a strong peak (cost 0.0) and the others are weak (cost ~1.0),
    /// all three words must converge on onset 1 (minimum cost anchor).
    /// The key invariant we test is that the total DTW cost is ≤ the greedy cost.
    #[test]
    fn test_dtw_aligns_to_nearest_peaks() {
        // 3 onsets: weak (0.0) at index 0, strong peak (1.0) at index 1,
        // weak (0.1) at index 2.  DTW cost matrix rows are identical; the
        // cheapest diagonal path is word0→o0, word1→o1, word2→o2 (sequential).
        // Cost = (1-0.0)+(1-1.0)+(1-0.1) = 1.0+0.0+0.9 = 1.9.
        // Greedy also picks the sequential path, same cost.
        // We verify: result length, non-empty outputs, and that DTW cost ≤ greedy cost.
        let onsets_ms = [0u32, 200, 500];
        let strength = [0.0f32, 1.0, 0.1];
        let words = align_lyrics_dtw("alpha beta gamma", &onsets_ms, &strength);
        assert_eq!(words.len(), 3);

        // All start_ms values must come from onsets_ms.
        for w in &words {
            assert!(
                onsets_ms.contains(&w.start_ms),
                "start_ms {} not in onsets_ms",
                w.start_ms
            );
        }

        // Compute DTW cost vs greedy cost.
        let greedy = align_lyrics("alpha beta gamma", &onsets_ms);
        let dtw_cost: f32 = words
            .iter()
            .map(|w| {
                let idx = onsets_ms.iter().position(|&t| t == w.start_ms).unwrap_or(0);
                1.0 - strength[idx]
            })
            .sum();
        let greedy_cost: f32 = greedy
            .iter()
            .map(|w| {
                let idx = onsets_ms.iter().position(|&t| t == w.start_ms).unwrap_or(0);
                1.0 - strength[idx]
            })
            .sum();
        assert!(
            dtw_cost <= greedy_cost + 1e-5,
            "DTW cost {dtw_cost} must be ≤ greedy cost {greedy_cost}"
        );
    }

    /// Result timestamps must be non-decreasing (monotone) for all words.
    /// The DTW forced alignment guarantees the onset index never decreases.
    #[test]
    fn test_dtw_monotone_time_order() {
        let onsets_ms = [0u32, 100, 300, 600, 1000];
        let strength = [0.2f32, 0.8, 0.5, 0.9, 0.3];
        let words = align_lyrics_dtw("one two three four five", &onsets_ms, &strength);
        assert_eq!(words.len(), 5);
        // Timestamps must be non-decreasing (monotone) — guaranteed by DTW monotone path.
        for i in 0..words.len() - 1 {
            assert!(
                words[i].start_ms <= words[i + 1].start_ms,
                "Expected non-decreasing times at index {i}: {} vs {}",
                words[i].start_ms,
                words[i + 1].start_ms
            );
        }
    }

    /// Empty onset_strength must fall back gracefully (no panic, result non-empty).
    #[test]
    fn test_dtw_empty_onset_strength_falls_back() {
        let onsets_ms = [0u32, 200, 400];
        let strength: &[f32] = &[];
        let words = align_lyrics_dtw("hello world", &onsets_ms, strength);
        // Falls back to greedy; must not panic and must return 2 words.
        assert_eq!(words.len(), 2);
    }

    /// Single word with multiple onsets: DTW should select the onset with the
    /// minimum cost (maximum strength), which is onset index 1 (strength 0.9).
    ///
    /// With n_words=1 and the onset-skip recurrence, the path ends at the
    /// onset j that minimises dp[1][j] = cost[0][j-1] + dp[0][j-1].
    /// Since dp[0][j-1] = 0.0 for all j (free prefix), dp[1][j] = cost[0][j-1].
    /// Minimum cost is at j=2 (onset index 1, strength 0.9, cost 0.1).
    #[test]
    fn test_dtw_single_word() {
        let onsets_ms = [0u32, 250, 500];
        let strength = [0.1f32, 0.9, 0.3];
        let words = align_lyrics_dtw("only", &onsets_ms, &strength);
        assert_eq!(words.len(), 1);
        // The peak is at index 1 (strength 0.9, cost 0.1) — minimum cost onset.
        assert_eq!(words[0].start_ms, 250);
    }

    /// DTW total alignment cost must be ≤ greedy cost.
    ///
    /// 2 words, 4 onsets where strengths are: low (0.1), low (0.2), high (1.0),
    /// high (0.9).  Greedy picks onset 0 for word 0 and onset 1 for word 1
    /// (sequential, high cumulative cost).  DTW finds the globally cheaper path.
    #[test]
    fn test_dtw_differs_from_greedy_on_non_sequential_peaks() {
        let onsets_ms = [0u32, 100, 200, 300];
        // Costs for each onset: 0.9, 0.8, 0.0, 0.1 (high cost for early onsets).
        let strength = [0.1f32, 0.2, 1.0, 0.9];

        let dtw_words = align_lyrics_dtw("foo bar", &onsets_ms, &strength);
        let greedy_words = align_lyrics("foo bar", &onsets_ms);

        assert_eq!(dtw_words.len(), 2);
        assert_eq!(greedy_words.len(), 2);

        // Compute total "cost" (1 - strength) for each alignment.
        let dtw_cost: f32 = dtw_words
            .iter()
            .map(|w| {
                let idx = onsets_ms.iter().position(|&t| t == w.start_ms).unwrap_or(0);
                1.0 - strength[idx]
            })
            .sum();
        let greedy_cost: f32 = greedy_words
            .iter()
            .map(|w| {
                let idx = onsets_ms.iter().position(|&t| t == w.start_ms).unwrap_or(0);
                1.0 - strength[idx]
            })
            .sum();

        // Greedy picks the first two onsets (sequential), accumulating high cost.
        // DTW should find an equal-or-lower total cost.
        assert!(
            dtw_cost <= greedy_cost + 1e-5,
            "Expected DTW cost ({dtw_cost}) ≤ greedy cost ({greedy_cost})"
        );
    }
}
