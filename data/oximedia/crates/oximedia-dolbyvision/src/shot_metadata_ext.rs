//! Extended per-shot Dolby Vision metadata with luminance and color statistics.
//!
//! Builds on top of basic shot metadata to provide per-shot analytical data
//! that can inform automatic trim selection and content-adaptive mapping.

#![allow(dead_code)]

use crate::cm_analysis::TrimMode;

// ── Luminance Statistics ──────────────────────────────────────────────────────

/// Per-shot PQ luminance statistics (normalised 0–1 range).
#[derive(Debug, Clone, PartialEq)]
pub struct ShotLuminanceStats {
    /// Minimum PQ value observed across the shot.
    pub min_pq: f32,
    /// Maximum PQ value observed across the shot.
    pub max_pq: f32,
    /// Average (mean) PQ value across the shot.
    pub avg_pq: f32,
    /// 10th-percentile PQ value.
    pub percentile_10: f32,
    /// 90th-percentile PQ value.
    pub percentile_90: f32,
}

impl ShotLuminanceStats {
    /// Dynamic range: max_pq minus min_pq.
    #[must_use]
    pub fn dynamic_range(&self) -> f32 {
        self.max_pq - self.min_pq
    }

    /// Returns `true` if the shot is predominantly dark (avg < 0.2).
    #[must_use]
    pub fn is_dark(&self) -> bool {
        self.avg_pq < 0.2
    }

    /// Returns `true` if the shot contains bright highlights (max_pq > 0.85).
    #[must_use]
    pub fn has_bright_highlights(&self) -> bool {
        self.max_pq > 0.85
    }
}

// ── Color Statistics ──────────────────────────────────────────────────────────

/// Per-shot color statistics derived from IPT-PQ analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct ShotColorStats {
    /// Average hue angle in degrees (0–360).
    pub avg_hue: f32,
    /// Spread (standard deviation) of hue angles in degrees.
    pub hue_spread: f32,
    /// Average chroma saturation magnitude.
    pub avg_saturation: f32,
    /// Dominant color as linear BT.2020 RGB `[r, g, b]`.
    pub dominant_color: [f32; 3],
}

// ── Extended Shot Metadata ────────────────────────────────────────────────────

/// Extended Dolby Vision shot metadata combining luminance, color, and trim info.
#[derive(Debug, Clone)]
pub struct DvShotExtMetadata {
    /// Unique shot identifier.
    pub shot_id: u64,
    /// Inclusive start frame index.
    pub start_frame: u64,
    /// Inclusive end frame index.
    pub end_frame: u64,
    /// Luminance statistics for this shot.
    pub luma: ShotLuminanceStats,
    /// Color statistics for this shot.
    pub color: ShotColorStats,
    /// Suggested trim mode based on content analysis.
    pub suggested_trim: TrimMode,
}

impl DvShotExtMetadata {
    /// Duration of the shot in frames (inclusive, so end - start + 1).
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame) + 1
    }

    /// Returns `true` if `frame` falls within this shot's range.
    #[must_use]
    pub fn contains_frame(&self, frame: u64) -> bool {
        frame >= self.start_frame && frame <= self.end_frame
    }
}

// ── PQ Percentile via Quickselect ────────────────────────────────────────────

/// Partial sort (Quickselect): returns the k-th smallest value (0-indexed) in `arr`.
///
/// Uses Hoare's partition scheme for O(n) average-case performance.
/// Modifies `arr` in place.
pub fn quickselect(arr: &mut Vec<f32>, k: usize) -> f32 {
    if arr.is_empty() {
        return 0.0;
    }
    let len = arr.len();
    let k_clamped = k.min(len - 1);
    qs_recursive(arr, 0, len - 1, k_clamped)
}

fn qs_recursive(arr: &mut Vec<f32>, left: usize, right: usize, k: usize) -> f32 {
    if left >= right {
        return arr[left];
    }
    let pivot_idx = qs_partition(arr, left, right);
    if k == pivot_idx {
        arr[pivot_idx]
    } else if k < pivot_idx {
        qs_recursive(arr, left, pivot_idx.saturating_sub(1), k)
    } else {
        qs_recursive(arr, pivot_idx + 1, right, k)
    }
}

fn qs_partition(arr: &mut Vec<f32>, left: usize, right: usize) -> usize {
    // Median-of-three pivot selection
    let mid = left + (right - left) / 2;
    let pivot_val = median_of_three(arr[left], arr[mid], arr[right]);

    // Move pivot to right
    let pivot_pos = if (arr[right] - pivot_val).abs() < f32::EPSILON {
        right
    } else if (arr[mid] - pivot_val).abs() < f32::EPSILON {
        arr.swap(mid, right);
        right
    } else {
        arr.swap(left, right);
        right
    };

    let _ = pivot_pos; // used implicitly via arr[right]
    let pivot = arr[right];
    let mut store = left;
    for i in left..right {
        if arr[i] <= pivot {
            arr.swap(store, i);
            store += 1;
        }
    }
    arr.swap(store, right);
    store
}

fn median_of_three(a: f32, b: f32, c: f32) -> f32 {
    if (a <= b && b <= c) || (c <= b && b <= a) {
        b
    } else if (b <= a && a <= c) || (c <= a && a <= b) {
        a
    } else {
        c
    }
}

/// Compute the percentile value from a slice of PQ floats.
///
/// `percentile` must be in [0.0, 100.0].
/// Uses quickselect for O(n) average performance.
#[must_use]
pub fn compute_pq_percentile(values: &[f32], percentile: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let p = percentile.clamp(0.0, 100.0);
    let idx = ((p / 100.0) * (values.len() - 1) as f32).round() as usize;
    let idx_clamped = idx.min(values.len() - 1);
    let mut scratch = values.to_vec();
    quickselect(&mut scratch, idx_clamped)
}

// ── Shot PQ Analysis ─────────────────────────────────────────────────────────

/// Analyze PQ statistics from a sequence of frames.
///
/// `frames` is a slice of flat f32 arrays (each of length `frame_width` or a
/// whole frame's worth of PQ values). `frame_width` is used to validate
/// per-frame stride (can be ignored for 1-D arrays by setting to 0).
///
/// Returns `ShotLuminanceStats` for the entire shot.
#[must_use]
pub fn analyze_shot_pq(frames: &[Vec<f32>], _frame_width: u32) -> ShotLuminanceStats {
    // Flatten all frame data into a single allocation
    let all_values: Vec<f32> = frames.iter().flat_map(|f| f.iter().copied()).collect();

    if all_values.is_empty() {
        return ShotLuminanceStats {
            min_pq: 0.0,
            max_pq: 0.0,
            avg_pq: 0.0,
            percentile_10: 0.0,
            percentile_90: 0.0,
        };
    }

    let min_pq = all_values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_pq = all_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let avg_pq = all_values.iter().sum::<f32>() / all_values.len() as f32;

    let percentile_10 = compute_pq_percentile(&all_values, 10.0);
    let percentile_90 = compute_pq_percentile(&all_values, 90.0);

    ShotLuminanceStats {
        min_pq,
        max_pq,
        avg_pq,
        percentile_10,
        percentile_90,
    }
}

// ── Trim Suggestion Heuristics ────────────────────────────────────────────────

/// Suggest a trim mode based on shot luminance statistics and target display peak.
///
/// Heuristics:
/// - If max_pq >> target_display_pq (more than 20% headroom compression needed): `Manual` gain
/// - If dynamic range is low (< 0.15): `Auto`
/// - Otherwise: `Auto`
#[must_use]
pub fn suggest_trim_mode(stats: &ShotLuminanceStats, target_display: f32) -> TrimMode {
    // Convert target_display nits to approximate PQ (simplified)
    let target_pq = (target_display / 10_000.0_f32).powf(0.159_301_757_8_f32);

    if stats.max_pq > target_pq * 1.2 {
        // Need significant highlight compression — use Manual gain
        let gain = (target_pq / stats.max_pq).clamp(0.1, 1.0);
        return TrimMode::Manual {
            lift: 0.0,
            gain,
            gamma: 1.0,
        };
    }

    if stats.dynamic_range() < 0.15 {
        // Low-contrast uniform shot — pure Auto is sufficient
        return TrimMode::Auto;
    }

    TrimMode::Auto
}

// ── Shot Metadata Aggregator ──────────────────────────────────────────────────

/// Aggregates per-shot extended metadata and supports merging contiguous shots.
#[derive(Debug, Clone, Default)]
pub struct ShotMetadataAggregator {
    /// All shots in chronological order.
    pub shots: Vec<DvShotExtMetadata>,
}

impl ShotMetadataAggregator {
    /// Create a new empty aggregator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a shot to the aggregator.
    pub fn add_shot(&mut self, shot: DvShotExtMetadata) {
        self.shots.push(shot);
    }

    /// Total number of shots.
    #[must_use]
    pub fn count(&self) -> usize {
        self.shots.len()
    }

    /// Merge contiguous shots whose luminance statistics differ by less than
    /// `threshold_pq` in both `avg_pq` and `max_pq`.
    ///
    /// Merged shots receive the union frame range and averaged statistics.
    pub fn merge_contiguous_shots(&mut self, threshold_pq: f32) {
        if self.shots.len() < 2 {
            return;
        }

        let mut merged: Vec<DvShotExtMetadata> = Vec::with_capacity(self.shots.len());
        let mut current = self.shots[0].clone();

        for next in self.shots.iter().skip(1) {
            let avg_diff = (current.luma.avg_pq - next.luma.avg_pq).abs();
            let max_diff = (current.luma.max_pq - next.luma.max_pq).abs();

            if avg_diff < threshold_pq && max_diff < threshold_pq {
                // Merge: extend current shot to include next
                current.end_frame = next.end_frame;
                // Average the luma stats
                current.luma.min_pq = current.luma.min_pq.min(next.luma.min_pq);
                current.luma.max_pq = current.luma.max_pq.max(next.luma.max_pq);
                current.luma.avg_pq = (current.luma.avg_pq + next.luma.avg_pq) / 2.0;
                current.luma.percentile_10 =
                    (current.luma.percentile_10 + next.luma.percentile_10) / 2.0;
                current.luma.percentile_90 =
                    (current.luma.percentile_90 + next.luma.percentile_90) / 2.0;
                // Keep current shot_id, update color as average
                current.color.avg_saturation =
                    (current.color.avg_saturation + next.color.avg_saturation) / 2.0;
            } else {
                merged.push(current);
                current = next.clone();
            }
        }
        merged.push(current);
        self.shots = merged;
    }

    /// Find the shot containing the given frame, if any.
    #[must_use]
    pub fn shot_for_frame(&self, frame: u64) -> Option<&DvShotExtMetadata> {
        self.shots.iter().find(|s| s.contains_frame(frame))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_luma(min: f32, max: f32, avg: f32) -> ShotLuminanceStats {
        ShotLuminanceStats {
            min_pq: min,
            max_pq: max,
            avg_pq: avg,
            percentile_10: min + (avg - min) * 0.1,
            percentile_90: avg + (max - avg) * 0.9,
        }
    }

    fn make_color() -> ShotColorStats {
        ShotColorStats {
            avg_hue: 120.0,
            hue_spread: 30.0,
            avg_saturation: 0.5,
            dominant_color: [0.2, 0.7, 0.1],
        }
    }

    fn make_shot(id: u64, start: u64, end: u64, luma: ShotLuminanceStats) -> DvShotExtMetadata {
        DvShotExtMetadata {
            shot_id: id,
            start_frame: start,
            end_frame: end,
            luma: luma.clone(),
            color: make_color(),
            suggested_trim: suggest_trim_mode(&luma, 1000.0),
        }
    }

    // ── ShotLuminanceStats ────────────────────────────────────────────────────

    #[test]
    fn test_shot_luma_dynamic_range() {
        let s = make_luma(0.05, 0.8, 0.4);
        assert!((s.dynamic_range() - 0.75).abs() < 1e-5);
    }

    #[test]
    fn test_shot_luma_is_dark_true() {
        let s = make_luma(0.0, 0.15, 0.1);
        assert!(s.is_dark());
    }

    #[test]
    fn test_shot_luma_is_dark_false() {
        let s = make_luma(0.1, 0.9, 0.5);
        assert!(!s.is_dark());
    }

    #[test]
    fn test_shot_luma_has_bright_highlights() {
        let s = make_luma(0.1, 0.95, 0.5);
        assert!(s.has_bright_highlights());
    }

    #[test]
    fn test_shot_luma_no_bright_highlights() {
        let s = make_luma(0.0, 0.5, 0.25);
        assert!(!s.has_bright_highlights());
    }

    // ── DvShotExtMetadata ────────────────────────────────────────────────────

    #[test]
    fn test_shot_duration_frames() {
        let shot = make_shot(1, 0, 23, make_luma(0.1, 0.8, 0.4));
        assert_eq!(shot.duration_frames(), 24);
    }

    #[test]
    fn test_shot_contains_frame_true() {
        let shot = make_shot(1, 10, 30, make_luma(0.1, 0.5, 0.3));
        assert!(shot.contains_frame(15));
    }

    #[test]
    fn test_shot_contains_frame_false() {
        let shot = make_shot(1, 10, 30, make_luma(0.1, 0.5, 0.3));
        assert!(!shot.contains_frame(31));
    }

    // ── Quickselect / Percentile ──────────────────────────────────────────────

    #[test]
    fn test_quickselect_single() {
        let mut v = vec![0.5_f32];
        let result = quickselect(&mut v, 0);
        assert!((result - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_quickselect_sorted_input() {
        let mut v: Vec<f32> = (0..=10).map(|i| i as f32 / 10.0).collect();
        // k=5 → median = 0.5
        let result = quickselect(&mut v, 5);
        assert!((result - 0.5).abs() < 1e-5, "result={result}");
    }

    #[test]
    fn test_quickselect_reverse_sorted() {
        let mut v: Vec<f32> = (0..=10).rev().map(|i| i as f32 / 10.0).collect();
        let result = quickselect(&mut v, 0);
        assert!((result - 0.0).abs() < 1e-5, "result={result}");
    }

    #[test]
    fn test_quickselect_k_beyond_len_clamped() {
        let mut v = vec![0.1_f32, 0.5, 0.9];
        let result = quickselect(&mut v, 100); // k > len-1
        assert!(result >= 0.0 && result <= 1.0, "result={result}");
    }

    #[test]
    fn test_compute_pq_percentile_empty() {
        let result = compute_pq_percentile(&[], 50.0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_compute_pq_percentile_median() {
        let values: Vec<f32> = (1..=101).map(|i| i as f32 / 100.0).collect();
        let p50 = compute_pq_percentile(&values, 50.0);
        assert!(p50 >= 0.49 && p50 <= 0.52, "p50={p50}");
    }

    #[test]
    fn test_compute_pq_percentile_min() {
        let values = vec![0.1_f32, 0.5, 0.9];
        let p0 = compute_pq_percentile(&values, 0.0);
        assert!((p0 - 0.1).abs() < 1e-5, "p0={p0}");
    }

    #[test]
    fn test_compute_pq_percentile_max() {
        let values = vec![0.1_f32, 0.5, 0.9];
        let p100 = compute_pq_percentile(&values, 100.0);
        assert!((p100 - 0.9).abs() < 1e-5, "p100={p100}");
    }

    // ── analyze_shot_pq ──────────────────────────────────────────────────────

    #[test]
    fn test_analyze_shot_pq_empty_frames() {
        let stats = analyze_shot_pq(&[], 1920);
        assert_eq!(stats.min_pq, 0.0);
        assert_eq!(stats.max_pq, 0.0);
        assert_eq!(stats.avg_pq, 0.0);
    }

    #[test]
    fn test_analyze_shot_pq_single_frame() {
        let frame = vec![0.1_f32, 0.5, 0.9, 0.3, 0.7];
        let stats = analyze_shot_pq(&[frame], 5);
        assert!((stats.min_pq - 0.1).abs() < 1e-5, "min={}", stats.min_pq);
        assert!((stats.max_pq - 0.9).abs() < 1e-5, "max={}", stats.max_pq);
    }

    #[test]
    fn test_analyze_shot_pq_multiple_frames() {
        let f1 = vec![0.2_f32, 0.4, 0.6];
        let f2 = vec![0.1_f32, 0.5, 0.8];
        let stats = analyze_shot_pq(&[f1, f2], 3);
        assert!((stats.min_pq - 0.1).abs() < 1e-5);
        assert!((stats.max_pq - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_analyze_shot_pq_percentiles_ordered() {
        let frame: Vec<f32> = (0..=100).map(|i| i as f32 / 100.0).collect();
        let stats = analyze_shot_pq(&[frame], 101);
        assert!(
            stats.percentile_10 <= stats.percentile_90,
            "p10 should <= p90"
        );
    }

    // ── suggest_trim_mode ────────────────────────────────────────────────────

    #[test]
    fn test_suggest_trim_mode_high_peak_manual() {
        // max_pq = 1.0 (10000 nits), target = 100 nits → needs big compression
        let stats = make_luma(0.0, 1.0, 0.5);
        let mode = suggest_trim_mode(&stats, 100.0);
        assert!(
            matches!(mode, TrimMode::Manual { .. }),
            "expected Manual, got {mode:?}"
        );
    }

    #[test]
    fn test_suggest_trim_mode_low_contrast_auto() {
        // Low dynamic range → Auto
        let stats = make_luma(0.4, 0.42, 0.41);
        let mode = suggest_trim_mode(&stats, 1000.0);
        assert_eq!(mode, TrimMode::Auto);
    }

    #[test]
    fn test_suggest_trim_mode_normal_auto() {
        // Normal range, target is close → Auto
        let stats = make_luma(0.05, 0.75, 0.4);
        let mode = suggest_trim_mode(&stats, 1000.0);
        assert_eq!(mode, TrimMode::Auto);
    }

    // ── ShotMetadataAggregator ───────────────────────────────────────────────

    #[test]
    fn test_aggregator_add_and_count() {
        let mut agg = ShotMetadataAggregator::new();
        agg.add_shot(make_shot(1, 0, 23, make_luma(0.1, 0.8, 0.4)));
        agg.add_shot(make_shot(2, 24, 47, make_luma(0.2, 0.9, 0.5)));
        assert_eq!(agg.count(), 2);
    }

    #[test]
    fn test_aggregator_shot_for_frame_found() {
        let mut agg = ShotMetadataAggregator::new();
        agg.add_shot(make_shot(1, 0, 23, make_luma(0.1, 0.8, 0.4)));
        let s = agg.shot_for_frame(10);
        assert!(s.is_some());
        assert_eq!(s.expect("should find shot").shot_id, 1);
    }

    #[test]
    fn test_aggregator_shot_for_frame_not_found() {
        let mut agg = ShotMetadataAggregator::new();
        agg.add_shot(make_shot(1, 0, 23, make_luma(0.1, 0.8, 0.4)));
        let s = agg.shot_for_frame(100);
        assert!(s.is_none());
    }

    #[test]
    fn test_aggregator_merge_similar_shots() {
        let mut agg = ShotMetadataAggregator::new();
        // Two shots with nearly identical luma → should merge
        agg.add_shot(make_shot(1, 0, 23, make_luma(0.1, 0.5, 0.3)));
        agg.add_shot(make_shot(2, 24, 47, make_luma(0.1, 0.52, 0.31)));
        agg.merge_contiguous_shots(0.05);
        assert_eq!(agg.count(), 1, "expected merge into 1 shot");
    }

    #[test]
    fn test_aggregator_no_merge_different_shots() {
        let mut agg = ShotMetadataAggregator::new();
        // Two shots with very different luma → should NOT merge
        agg.add_shot(make_shot(1, 0, 23, make_luma(0.0, 0.2, 0.1)));
        agg.add_shot(make_shot(2, 24, 47, make_luma(0.5, 0.95, 0.8)));
        agg.merge_contiguous_shots(0.05);
        assert_eq!(agg.count(), 2, "expected no merge");
    }

    #[test]
    fn test_aggregator_merge_extends_frame_range() {
        let mut agg = ShotMetadataAggregator::new();
        agg.add_shot(make_shot(1, 0, 23, make_luma(0.1, 0.5, 0.3)));
        agg.add_shot(make_shot(2, 24, 47, make_luma(0.1, 0.52, 0.31)));
        agg.merge_contiguous_shots(0.05);
        assert_eq!(agg.shots[0].start_frame, 0);
        assert_eq!(agg.shots[0].end_frame, 47);
    }

    #[test]
    fn test_aggregator_empty_no_panic() {
        let mut agg = ShotMetadataAggregator::new();
        agg.merge_contiguous_shots(0.1); // Should not panic
        assert_eq!(agg.count(), 0);
    }
}
