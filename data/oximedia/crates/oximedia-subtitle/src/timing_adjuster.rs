//! Subtitle timing adjuster: offset, FPS scaling, and automatic sync detection.
//!
//! `TimingAdjuster` combines a global millisecond offset with frame-rate
//! conversion into a single composable transform that can be applied to
//! individual timestamps, cue entries, or entire `SubtitleDocument`s.

use crate::format_converter::{SubtitleDocument, SubtitleEntry};

// ── TimingAdjuster ────────────────────────────────────────────────────────────

/// Combines a global time-shift with an optional frame-rate conversion.
///
/// The composed transform is: `ts_out = (ts_in * (dst_fps / src_fps)) + offset_ms`
#[derive(Debug, Clone, PartialEq)]
pub struct TimingAdjuster {
    /// Global shift in milliseconds (positive = later, negative = earlier).
    pub offset_ms: i64,
    /// Source frames per second.
    pub frame_rate_src: f64,
    /// Destination frames per second.
    pub frame_rate_dst: f64,
}

impl TimingAdjuster {
    /// Create a new `TimingAdjuster`.
    ///
    /// Pass `src_fps == dst_fps` (e.g. both `1.0`) for a pure offset-only adjuster.
    #[must_use]
    pub fn new(offset_ms: i64, src_fps: f64, dst_fps: f64) -> Self {
        // Fallback to 1.0/1.0 if values are invalid to avoid division by zero
        let src = if src_fps > 0.0 && src_fps.is_finite() {
            src_fps
        } else {
            1.0
        };
        let dst = if dst_fps > 0.0 && dst_fps.is_finite() {
            dst_fps
        } else {
            1.0
        };
        Self {
            offset_ms,
            frame_rate_src: src,
            frame_rate_dst: dst,
        }
    }

    /// Identity adjuster (no change).
    #[must_use]
    pub fn identity() -> Self {
        Self::new(0, 1.0, 1.0)
    }

    /// NTSC 23.976 → PAL 25.0 fps conversion (scale ≈ 1.04167), no offset.
    #[must_use]
    pub fn ntsc_to_pal() -> Self {
        Self::new(0, 23.976, 25.0)
    }

    /// PAL 25.0 → NTSC 23.976 fps conversion (scale ≈ 0.95904), no offset.
    #[must_use]
    pub fn pal_to_ntsc() -> Self {
        Self::new(0, 25.0, 23.976)
    }

    /// Compute the FPS scale factor `dst_fps / src_fps`.
    #[must_use]
    pub fn fps_scale(&self) -> f64 {
        self.frame_rate_dst / self.frame_rate_src
    }

    /// Apply the full transform to a single timestamp in milliseconds.
    ///
    /// The result is clamped to zero.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_precision_loss)]
    pub fn adjust_ms(&self, ts_ms: u64) -> u64 {
        let scaled = ts_ms as f64 * self.fps_scale();
        let shifted = scaled.round() as i64 + self.offset_ms;
        shifted.max(0) as u64
    }

    /// Apply the transform to a single `SubtitleEntry` (in-place).
    pub fn adjust_entry(&self, entry: &mut SubtitleEntry) {
        entry.start_ms = self.adjust_ms(entry.start_ms);
        entry.end_ms = self.adjust_ms(entry.end_ms);
    }

    /// Apply the transform to every entry in a `SubtitleDocument` (in-place).
    pub fn adjust_document(&self, doc: &mut SubtitleDocument) {
        for entry in &mut doc.entries {
            self.adjust_entry(entry);
        }
    }

    /// Estimate the timing offset between `reference` and `target` documents.
    ///
    /// Tries offsets in the range −10 000 ms … +10 000 ms in 100 ms steps and
    /// returns the offset that maximises the number of cue-start-time overlaps
    /// (i.e., the count of `reference` entries whose `start_ms` falls within
    /// `[target_start − tolerance, target_start + tolerance)` after shifting).
    ///
    /// A tolerance of 500 ms is used when looking for matching starts.
    #[must_use]
    pub fn detect_offset(reference: &SubtitleDocument, target: &SubtitleDocument) -> i64 {
        const TOLERANCE_MS: i64 = 500;
        const STEP_MS: i64 = 100;
        const RANGE_MS: i64 = 10_000;

        let ref_starts: Vec<i64> = reference
            .entries
            .iter()
            .map(|e| e.start_ms as i64)
            .collect();
        let tgt_starts: Vec<i64> = target.entries.iter().map(|e| e.start_ms as i64).collect();

        if ref_starts.is_empty() || tgt_starts.is_empty() {
            return 0;
        }

        let mut best_offset = 0i64;
        let mut best_count = 0usize;

        let mut offset = -RANGE_MS;
        while offset <= RANGE_MS {
            let count = ref_starts
                .iter()
                .filter(|&&rs| {
                    tgt_starts.iter().any(|&ts| {
                        let shifted = ts + offset;
                        (shifted - rs).abs() < TOLERANCE_MS
                    })
                })
                .count();

            // Prefer strictly more matches; on tie prefer smaller absolute offset
            let better = count > best_count
                || (count == best_count && count > 0 && offset.abs() < best_offset.abs());

            if better {
                best_count = count;
                best_offset = offset;
            }
            offset += STEP_MS;
        }

        best_offset
    }
}

// ── Non-Linear Time Remapping ─────────────────────────────────────────────────

/// A keyframe in a non-linear time remap curve.
///
/// Maps `source_ms` (original timeline) → `dest_ms` (remapped timeline).
/// The adjuster linearly interpolates between consecutive keyframes and
/// clamps outside the defined range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RemapKeyframe {
    /// Time in the source (original) timeline, in milliseconds.
    pub source_ms: f64,
    /// Corresponding time in the destination timeline, in milliseconds.
    pub dest_ms: f64,
}

impl RemapKeyframe {
    /// Create a new keyframe.
    #[must_use]
    pub fn new(source_ms: f64, dest_ms: f64) -> Self {
        Self { source_ms, dest_ms }
    }
}

/// Non-linear time remapping adjuster.
///
/// Uses a piecewise-linear curve defined by keyframes to remap subtitle
/// timestamps. Supports speed ramps, variable frame rate correction,
/// and arbitrary time warping.
///
/// # Example
///
/// ```ignore
/// use oximedia_subtitle::timing_adjuster::{NonLinearRemapper, RemapKeyframe};
/// let mut remapper = NonLinearRemapper::new();
/// // First 10s at normal speed, then 2x speed for next 10s
/// remapper.add_keyframe(RemapKeyframe::new(0.0, 0.0));
/// remapper.add_keyframe(RemapKeyframe::new(10_000.0, 10_000.0));
/// remapper.add_keyframe(RemapKeyframe::new(20_000.0, 15_000.0)); // 10s → 5s
/// let remapped = remapper.remap_ms(15_000.0); // midpoint of ramp
/// ```
#[derive(Debug, Clone)]
pub struct NonLinearRemapper {
    /// Keyframes sorted by source_ms.
    keyframes: Vec<RemapKeyframe>,
}

impl NonLinearRemapper {
    /// Create a new empty remapper.
    #[must_use]
    pub fn new() -> Self {
        Self {
            keyframes: Vec::new(),
        }
    }

    /// Create from a speed ramp specification.
    ///
    /// Each segment is `(duration_source_ms, speed_factor)`:
    /// - `speed_factor = 1.0` → normal speed
    /// - `speed_factor = 2.0` → source plays at 2× (dest duration = source / 2)
    /// - `speed_factor = 0.5` → slow-motion (dest duration = source × 2)
    #[must_use]
    pub fn from_speed_ramp(segments: &[(f64, f64)]) -> Self {
        let mut remapper = Self::new();
        let mut src_cursor = 0.0f64;
        let mut dst_cursor = 0.0f64;

        remapper.keyframes.push(RemapKeyframe::new(0.0, 0.0));

        for &(duration_ms, speed) in segments {
            let safe_speed = if speed > 0.0 && speed.is_finite() {
                speed
            } else {
                1.0
            };
            src_cursor += duration_ms;
            dst_cursor += duration_ms / safe_speed;
            remapper
                .keyframes
                .push(RemapKeyframe::new(src_cursor, dst_cursor));
        }

        remapper
    }

    /// Create a variable frame rate (VFR) correction remapper.
    ///
    /// Takes a list of `(source_pts_ms, corrected_pts_ms)` pairs
    /// sampled from the actual frame timing, building a correction curve.
    #[must_use]
    pub fn from_vfr_correction(samples: &[(f64, f64)]) -> Self {
        let mut remapper = Self::new();
        for &(src, dst) in samples {
            remapper.keyframes.push(RemapKeyframe::new(src, dst));
        }
        remapper.keyframes.sort_by(|a, b| {
            a.source_ms
                .partial_cmp(&b.source_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        remapper
    }

    /// Add a keyframe. Keyframes will be sorted before remapping.
    pub fn add_keyframe(&mut self, kf: RemapKeyframe) {
        self.keyframes.push(kf);
        self.keyframes.sort_by(|a, b| {
            a.source_ms
                .partial_cmp(&b.source_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Return the number of keyframes.
    #[must_use]
    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    /// Remap a single timestamp (in milliseconds) through the curve.
    ///
    /// Uses piecewise linear interpolation. Timestamps before the first
    /// keyframe or after the last are extrapolated using the nearest segment's
    /// slope.
    #[must_use]
    pub fn remap_ms(&self, source_ms: f64) -> f64 {
        if self.keyframes.is_empty() {
            return source_ms;
        }
        if self.keyframes.len() == 1 {
            // Single keyframe acts as a pure offset
            let kf = &self.keyframes[0];
            return source_ms + (kf.dest_ms - kf.source_ms);
        }

        // Find the segment containing source_ms
        let idx = self
            .keyframes
            .partition_point(|kf| kf.source_ms <= source_ms);

        if idx == 0 {
            // Before first keyframe: extrapolate from first segment
            let kf0 = &self.keyframes[0];
            let kf1 = &self.keyframes[1];
            return self.interpolate(kf0, kf1, source_ms);
        }

        if idx >= self.keyframes.len() {
            // After last keyframe: extrapolate from last segment
            let kf0 = &self.keyframes[self.keyframes.len() - 2];
            let kf1 = &self.keyframes[self.keyframes.len() - 1];
            return self.interpolate(kf0, kf1, source_ms);
        }

        // Between keyframes idx-1 and idx
        let kf0 = &self.keyframes[idx - 1];
        let kf1 = &self.keyframes[idx];
        self.interpolate(kf0, kf1, source_ms)
    }

    /// Remap a single timestamp (u64 milliseconds), clamped to zero.
    #[must_use]
    pub fn remap_ms_u64(&self, source_ms: u64) -> u64 {
        let result = self.remap_ms(source_ms as f64);
        if result < 0.0 {
            0
        } else {
            result.round() as u64
        }
    }

    /// Apply non-linear remapping to a `SubtitleEntry`.
    pub fn remap_entry(&self, entry: &mut SubtitleEntry) {
        entry.start_ms = self.remap_ms_u64(entry.start_ms);
        entry.end_ms = self.remap_ms_u64(entry.end_ms);
    }

    /// Apply non-linear remapping to all entries in a `SubtitleDocument`.
    pub fn remap_document(&self, doc: &mut SubtitleDocument) {
        for entry in &mut doc.entries {
            self.remap_entry(entry);
        }
    }

    /// Compute the local speed factor at a given source timestamp.
    ///
    /// Returns `dest_speed / source_speed` (i.e. `> 1.0` means destination
    /// timeline is running faster than source at that point).
    #[must_use]
    pub fn speed_at(&self, source_ms: f64) -> f64 {
        if self.keyframes.len() < 2 {
            return 1.0;
        }

        let idx = self
            .keyframes
            .partition_point(|kf| kf.source_ms <= source_ms);

        let (kf0, kf1) = if idx == 0 {
            (&self.keyframes[0], &self.keyframes[1])
        } else if idx >= self.keyframes.len() {
            (
                &self.keyframes[self.keyframes.len() - 2],
                &self.keyframes[self.keyframes.len() - 1],
            )
        } else {
            (&self.keyframes[idx - 1], &self.keyframes[idx])
        };

        let src_delta = kf1.source_ms - kf0.source_ms;
        if src_delta.abs() < 1e-9 {
            return 1.0;
        }

        let dst_delta = kf1.dest_ms - kf0.dest_ms;
        dst_delta / src_delta
    }

    /// Linear interpolation / extrapolation between two keyframes.
    fn interpolate(&self, kf0: &RemapKeyframe, kf1: &RemapKeyframe, source_ms: f64) -> f64 {
        let src_delta = kf1.source_ms - kf0.source_ms;
        if src_delta.abs() < 1e-9 {
            return kf0.dest_ms;
        }
        let t = (source_ms - kf0.source_ms) / src_delta;
        kf0.dest_ms + t * (kf1.dest_ms - kf0.dest_ms)
    }
}

impl Default for NonLinearRemapper {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format_converter::{SubtitleDocument, SubtitleEntry, SubtitleFormat};

    fn make_doc(starts: &[(u64, u64, &str)]) -> SubtitleDocument {
        let entries = starts
            .iter()
            .enumerate()
            .map(|(i, &(s, e, t))| SubtitleEntry::new(i as u32 + 1, s, e, t))
            .collect();
        SubtitleDocument {
            format: SubtitleFormat::Srt,
            entries,
            metadata: Default::default(),
            styles: Vec::new(),
        }
    }

    #[test]
    fn test_new_stores_fields() {
        let adj = TimingAdjuster::new(500, 24.0, 25.0);
        assert_eq!(adj.offset_ms, 500);
        assert!((adj.frame_rate_src - 24.0).abs() < f64::EPSILON);
        assert!((adj.frame_rate_dst - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_identity_no_change() {
        let adj = TimingAdjuster::identity();
        assert_eq!(adj.adjust_ms(5_000), 5_000);
    }

    #[test]
    fn test_adjust_ms_pure_offset() {
        let adj = TimingAdjuster::new(1_000, 1.0, 1.0);
        assert_eq!(adj.adjust_ms(4_000), 5_000);
    }

    #[test]
    fn test_adjust_ms_clamp_to_zero() {
        let adj = TimingAdjuster::new(-5_000, 1.0, 1.0);
        assert_eq!(adj.adjust_ms(1_000), 0);
    }

    #[test]
    fn test_adjust_ms_fps_scale() {
        // 24→25: scale = 25/24 ≈ 1.04167; 24_000 * 1.04167 ≈ 25_000
        let adj = TimingAdjuster::new(0, 24.0, 25.0);
        let result = adj.adjust_ms(24_000);
        assert!((result as i64 - 25_000).abs() <= 5, "result={result}");
    }

    #[test]
    fn test_ntsc_to_pal_scale() {
        let adj = TimingAdjuster::ntsc_to_pal();
        // fps_scale = 25 / 23.976 ≈ 1.04167
        let scale = adj.fps_scale();
        let expected = 25.0 / 23.976;
        assert!((scale - expected).abs() < 1e-6, "scale={scale}");
    }

    #[test]
    fn test_pal_to_ntsc_scale() {
        let adj = TimingAdjuster::pal_to_ntsc();
        let scale = adj.fps_scale();
        let expected = 23.976 / 25.0;
        assert!((scale - expected).abs() < 1e-6, "scale={scale}");
    }

    #[test]
    fn test_adjust_entry() {
        let adj = TimingAdjuster::new(500, 1.0, 1.0);
        let mut entry = SubtitleEntry::new(1, 1_000, 4_000, "hello");
        adj.adjust_entry(&mut entry);
        assert_eq!(entry.start_ms, 1_500);
        assert_eq!(entry.end_ms, 4_500);
    }

    #[test]
    fn test_adjust_document() {
        let adj = TimingAdjuster::new(1_000, 1.0, 1.0);
        let mut doc = make_doc(&[(1_000, 4_000, "a"), (5_000, 8_000, "b")]);
        adj.adjust_document(&mut doc);
        assert_eq!(doc.entries[0].start_ms, 2_000);
        assert_eq!(doc.entries[1].start_ms, 6_000);
    }

    #[test]
    fn test_detect_offset_zero() {
        // Same document — best offset should be 0 (or the smallest abs-value tie)
        let doc = make_doc(&[(1_000, 4_000, "a"), (5_000, 8_000, "b")]);
        let offset = TimingAdjuster::detect_offset(&doc, &doc);
        // Tie-breaking prefers smallest absolute offset; should be 0
        assert_eq!(offset, 0, "identity offset should be 0, got {offset}");
    }

    #[test]
    fn test_detect_offset_shifted() {
        let reference = make_doc(&[(1_000, 4_000, "a"), (5_000, 8_000, "b")]);
        // target is shifted by +2000 ms
        let target = make_doc(&[(3_000, 6_000, "a"), (7_000, 10_000, "b")]);
        let offset = TimingAdjuster::detect_offset(&reference, &target);
        // With tolerance=500ms the best offset is the one that maximises matches
        // and minimises absolute value. Should be within 500ms of -2000.
        assert!((offset + 2_000).abs() <= 500, "offset={offset}");
    }

    #[test]
    fn test_detect_offset_empty_docs() {
        let empty = SubtitleDocument::empty(SubtitleFormat::Srt);
        let doc = make_doc(&[(1_000, 4_000, "a")]);
        assert_eq!(TimingAdjuster::detect_offset(&empty, &doc), 0);
        assert_eq!(TimingAdjuster::detect_offset(&doc, &empty), 0);
    }

    #[test]
    fn test_ntsc_to_pal_adjust_entry() {
        let adj = TimingAdjuster::ntsc_to_pal();
        let mut entry = SubtitleEntry::new(1, 23_976, 47_952, "test");
        adj.adjust_entry(&mut entry);
        // 23_976 * (25/23.976) ≈ 25_000
        assert!(
            (entry.start_ms as i64 - 25_000).abs() <= 10,
            "start={}",
            entry.start_ms
        );
    }

    #[test]
    fn test_invalid_fps_fallback() {
        // Zero or NaN fps should not panic; should fall back to 1.0
        let adj = TimingAdjuster::new(0, 0.0, f64::NAN);
        assert_eq!(adj.adjust_ms(1_000), 1_000);
    }

    // ── Non-linear remapping tests ────────────────────────────────────

    #[test]
    fn test_nonlinear_empty_is_identity() {
        let r = NonLinearRemapper::new();
        assert!((r.remap_ms(5000.0) - 5000.0).abs() < 0.01);
    }

    #[test]
    fn test_nonlinear_single_keyframe_offset() {
        let mut r = NonLinearRemapper::new();
        r.add_keyframe(RemapKeyframe::new(0.0, 1000.0));
        // Single keyframe: offset = 1000
        assert!((r.remap_ms(5000.0) - 6000.0).abs() < 0.01);
    }

    #[test]
    fn test_nonlinear_identity_two_keyframes() {
        let mut r = NonLinearRemapper::new();
        r.add_keyframe(RemapKeyframe::new(0.0, 0.0));
        r.add_keyframe(RemapKeyframe::new(10000.0, 10000.0));
        assert!((r.remap_ms(5000.0) - 5000.0).abs() < 0.01);
    }

    #[test]
    fn test_nonlinear_double_speed() {
        let mut r = NonLinearRemapper::new();
        // Source 0-10s mapped to dest 0-5s (2x speed)
        r.add_keyframe(RemapKeyframe::new(0.0, 0.0));
        r.add_keyframe(RemapKeyframe::new(10000.0, 5000.0));
        // At source 5000ms, should be dest 2500ms
        assert!((r.remap_ms(5000.0) - 2500.0).abs() < 0.01);
    }

    #[test]
    fn test_nonlinear_half_speed() {
        let mut r = NonLinearRemapper::new();
        // Source 0-10s mapped to dest 0-20s (0.5x speed / slow-motion)
        r.add_keyframe(RemapKeyframe::new(0.0, 0.0));
        r.add_keyframe(RemapKeyframe::new(10000.0, 20000.0));
        assert!((r.remap_ms(5000.0) - 10000.0).abs() < 0.01);
    }

    #[test]
    fn test_nonlinear_speed_ramp() {
        // Normal 10s, then 2x for 10s
        let r = NonLinearRemapper::from_speed_ramp(&[(10000.0, 1.0), (10000.0, 2.0)]);
        assert_eq!(r.keyframe_count(), 3);
        // At 5s (in first segment): should be ~5000
        assert!((r.remap_ms(5000.0) - 5000.0).abs() < 0.01);
        // At 15s (midpoint of 2x segment): 10000 + (5000/2) = 12500? No...
        // dest = 10000 + (15000-10000) * (15000-10000)/10000 slope
        // kf1 = (10000, 10000), kf2 = (20000, 15000), slope=0.5
        // remap(15000) = 10000 + (15000-10000)*0.5 = 12500
        assert!((r.remap_ms(15000.0) - 12500.0).abs() < 0.01);
    }

    #[test]
    fn test_nonlinear_extrapolation_before() {
        let mut r = NonLinearRemapper::new();
        r.add_keyframe(RemapKeyframe::new(1000.0, 1000.0));
        r.add_keyframe(RemapKeyframe::new(2000.0, 3000.0)); // 2x slope
                                                            // Before first keyframe: extrapolate with slope 2.0
                                                            // remap(0) = 1000 + (0-1000)*2 = 1000 - 2000 = -1000
        assert!((r.remap_ms(0.0) - (-1000.0)).abs() < 0.01);
    }

    #[test]
    fn test_nonlinear_extrapolation_after() {
        let mut r = NonLinearRemapper::new();
        r.add_keyframe(RemapKeyframe::new(0.0, 0.0));
        r.add_keyframe(RemapKeyframe::new(10000.0, 5000.0)); // 0.5 slope
                                                             // After last keyframe: extrapolate
        assert!((r.remap_ms(20000.0) - 10000.0).abs() < 0.01);
    }

    #[test]
    fn test_nonlinear_remap_ms_u64_clamp() {
        let mut r = NonLinearRemapper::new();
        r.add_keyframe(RemapKeyframe::new(1000.0, 0.0));
        r.add_keyframe(RemapKeyframe::new(2000.0, 1000.0));
        // Before the curve starts, may go negative — should clamp to 0
        let val = r.remap_ms_u64(0);
        assert_eq!(val, 0);
    }

    #[test]
    fn test_nonlinear_speed_at_normal() {
        let mut r = NonLinearRemapper::new();
        r.add_keyframe(RemapKeyframe::new(0.0, 0.0));
        r.add_keyframe(RemapKeyframe::new(10000.0, 10000.0));
        let speed = r.speed_at(5000.0);
        assert!((speed - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_nonlinear_speed_at_double() {
        let mut r = NonLinearRemapper::new();
        r.add_keyframe(RemapKeyframe::new(0.0, 0.0));
        r.add_keyframe(RemapKeyframe::new(10000.0, 20000.0));
        let speed = r.speed_at(5000.0);
        assert!((speed - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_nonlinear_speed_at_empty() {
        let r = NonLinearRemapper::new();
        assert!((r.speed_at(5000.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_nonlinear_vfr_correction() {
        let samples = vec![(0.0, 0.0), (10000.0, 10500.0), (20000.0, 20000.0)];
        let r = NonLinearRemapper::from_vfr_correction(&samples);
        assert_eq!(r.keyframe_count(), 3);
        // At 5000ms: interpolate between (0,0) and (10000,10500): 5250
        assert!((r.remap_ms(5000.0) - 5250.0).abs() < 0.01);
    }

    #[test]
    fn test_nonlinear_remap_entry() {
        let r = NonLinearRemapper::from_speed_ramp(&[(10000.0, 2.0)]);
        let mut entry = SubtitleEntry::new(1, 4000, 8000, "test");
        r.remap_entry(&mut entry);
        // slope = 0.5, so 4000 → 2000, 8000 → 4000
        assert_eq!(entry.start_ms, 2000);
        assert_eq!(entry.end_ms, 4000);
    }

    #[test]
    fn test_nonlinear_remap_document() {
        let r = NonLinearRemapper::from_speed_ramp(&[(10000.0, 1.0)]);
        let mut doc = make_doc(&[(1000, 4000, "a"), (5000, 8000, "b")]);
        r.remap_document(&mut doc);
        // Identity speed: should be unchanged
        assert_eq!(doc.entries[0].start_ms, 1000);
        assert_eq!(doc.entries[1].end_ms, 8000);
    }

    #[test]
    fn test_nonlinear_multi_segment_ramp() {
        // 5s normal, 5s at 0.5x (slow), 5s at 3x (fast)
        let r = NonLinearRemapper::from_speed_ramp(&[(5000.0, 1.0), (5000.0, 0.5), (5000.0, 3.0)]);
        assert_eq!(r.keyframe_count(), 4);
        // kf0=(0,0), kf1=(5000,5000), kf2=(10000,15000), kf3=(15000,15000+5000/3≈16666.7)
        // At 7500ms (mid of slow segment): lerp(5000,5000 -> 10000,15000): 10000
        assert!((r.remap_ms(7500.0) - 10000.0).abs() < 1.0);
    }

    #[test]
    fn test_nonlinear_default_trait() {
        let r = NonLinearRemapper::default();
        assert_eq!(r.keyframe_count(), 0);
    }
}
