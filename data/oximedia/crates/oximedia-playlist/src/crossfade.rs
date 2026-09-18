//! Crossfade transitions: overlap duration, fade curves, and gapless playback.
//!
//! This module provides types and logic for computing precise crossfade
//! schedules between consecutive playlist items, including curve shapes,
//! overlap windows, and gapless audio continuity.
//!
//! # Logarithmic crossfade
//!
//! Human hearing is logarithmic in nature (Weber-Fechner law).  A linear
//! volume ramp from 1.0 → 0.0 sounds like the audio drops sharply at the
//! beginning and then lingers quietly at the end.  The [`FadeCurve::Logarithmic`]
//! and [`FadeCurve::LogarithmicSymmetric`] variants implement perceptually uniform
//! fade curves that map the linear gain domain to decibels and back:
//!
//! ```text
//!   gain_db(t) = (1 - t) × max_db          (outgoing, fade-out)
//!   gain_lin   = 10^(gain_db / 20)
//! ```
//!
//! The `max_db` reference is set to 0 dBFS (unity gain), and the transition
//! threshold is `−∞` (silence), which is approached as a true limit.  In
//! practice we clamp the gain to 0.0 when `gain_db < -96 dB` (noise floor
//! of a 16-bit system).
//!
//! `LogarithmicSymmetric` uses the same curve for both the outgoing and
//! incoming signal (i.e. fade-in is the mirror of fade-out), so the combined
//! level dips slightly in the middle — useful for DJ-style transitions where
//! a brief silence between tracks is desirable.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::time::Duration;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Minimum gain in dB before we clamp to absolute silence.
/// This corresponds to the dynamic range floor of 16-bit PCM (−96 dBFS).
const DB_FLOOR: f64 = -96.0;

/// Convert a dB value to a linear amplitude gain.
#[inline]
fn db_to_linear(db: f64) -> f64 {
    if db <= DB_FLOOR {
        0.0
    } else {
        10.0_f64.powf(db / 20.0)
    }
}

/// Convert a linear amplitude gain to dB.  Returns [`DB_FLOOR`] for zero gain.
#[inline]
fn linear_to_db(gain: f64) -> f64 {
    if gain <= 0.0 {
        DB_FLOOR
    } else {
        20.0 * gain.log10()
    }
}

// ── fade curve ────────────────────────────────────────────────────────────────

/// Shape of the fade curve applied during a crossfade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FadeCurve {
    /// Linear fade: gain changes at a constant rate.
    Linear,
    /// Equal-power (sine/cosine) crossfade: maintains constant perceived loudness.
    EqualPower,
    /// Perceptually uniform logarithmic crossfade (dB-linear ramp).
    ///
    /// The outgoing signal fades from 0 dBFS to −96 dBFS following a straight
    /// line in dB space.  The incoming signal is the complement (0 dBFS fade-in
    /// from −96 dBFS).  This curve sounds natural to human ears because our
    /// auditory system perceives loudness logarithmically.
    Logarithmic,
    /// Symmetric logarithmic crossfade — both signals use the same dB-linear
    /// fade-out curve, meaning the combined level dips ≈ 3 dB at the mid-point.
    /// Useful for DJ and broadcast transitions where a brief breath between tracks
    /// is intentional.
    LogarithmicSymmetric,
    /// Exponential fade-in (slow at start, fast at end).
    Exponential,
    /// S-curve (smooth step) for natural-sounding transitions.
    SCurve,
}

impl FadeCurve {
    /// Compute the gain multiplier for the *outgoing* track at position `t` ∈ [0, 1].
    /// `t = 0` = start of crossfade (full volume), `t = 1` = end (silence).
    #[must_use]
    pub fn fade_out_gain(self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            FadeCurve::Linear => 1.0 - t,
            FadeCurve::EqualPower => (std::f64::consts::FRAC_PI_2 * t).cos(),
            FadeCurve::Logarithmic | FadeCurve::LogarithmicSymmetric => {
                // dB-linear ramp: 0 dBFS → −96 dBFS as t goes 0 → 1.
                let db = DB_FLOOR * t;
                db_to_linear(db)
            }
            FadeCurve::Exponential => (1.0 - t).powi(2),
            FadeCurve::SCurve => {
                let v = 1.0 - t;
                v * v * (3.0 - 2.0 * v)
            }
        }
    }

    /// Compute the gain multiplier for the *incoming* track at position `t` ∈ [0, 1].
    /// `t = 0` = start of crossfade (silence), `t = 1` = end (full volume).
    #[must_use]
    pub fn fade_in_gain(self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            FadeCurve::Linear => t,
            FadeCurve::EqualPower => (std::f64::consts::FRAC_PI_2 * t).sin(),
            FadeCurve::Logarithmic => {
                // Complement of the outgoing dB-linear ramp:
                // at t=0: DB_FLOOR → silence; at t=1: 0 dBFS → unity.
                let db = DB_FLOOR * (1.0 - t);
                db_to_linear(db)
            }
            FadeCurve::LogarithmicSymmetric => {
                // Mirror the fade-out: uses the same dB-linear downward ramp
                // but read in reverse — so the incoming track *also* fades
                // out-style but from silence to full volume.
                // This is equivalent to fade_out_gain(1 - t).
                let db = DB_FLOOR * (1.0 - t);
                db_to_linear(db)
            }
            FadeCurve::Exponential => {
                // Smooth exponential approach to unity.
                // Uses 1 - (1-t)^3 for a convex curve (fast rise, slow finish).
                let v = 1.0 - t;
                1.0 - v * v * v
            }
            FadeCurve::SCurve => t * t * (3.0 - 2.0 * t),
        }
    }

    /// Convert a linear gain to dBFS.  Exposed for diagnostic use.
    #[must_use]
    pub fn linear_to_db(gain: f64) -> f64 {
        linear_to_db(gain)
    }

    /// Convert a dBFS value to linear gain.  Exposed for diagnostic use.
    #[must_use]
    pub fn db_to_linear(db: f64) -> f64 {
        db_to_linear(db)
    }

    /// Compute the combined perceived loudness in dB at crossfade position `t`.
    ///
    /// This sums the power of the two signals and converts to dB, giving the
    /// apparent loudness from the listener's perspective.  For an equal-power
    /// crossfade this should return ≈ 0 dBFS throughout.
    #[must_use]
    pub fn combined_db(self, t: f64) -> f64 {
        let out = self.fade_out_gain(t);
        let inp = self.fade_in_gain(t);
        let power = out * out + inp * inp;
        if power <= 0.0 {
            DB_FLOOR
        } else {
            10.0 * power.log10()
        }
    }

    /// Sample the combined fade pair at `n` evenly-spaced points.
    /// Returns `(fade_out, fade_in)` gain vectors.
    #[must_use]
    pub fn sample_pair(self, n: usize) -> (Vec<f64>, Vec<f64>) {
        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        let out: Vec<f64> = (0..n)
            .map(|i| self.fade_out_gain(i as f64 / (n - 1).max(1) as f64))
            .collect();
        let inp: Vec<f64> = (0..n)
            .map(|i| self.fade_in_gain(i as f64 / (n - 1).max(1) as f64))
            .collect();
        (out, inp)
    }

    /// Sample the dB-space gain for the outgoing signal at `n` points.
    /// Useful for rendering volume automation curves in a DAW-style interface.
    #[must_use]
    pub fn sample_db_out(self, n: usize) -> Vec<f64> {
        if n == 0 {
            return Vec::new();
        }
        (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1).max(1) as f64;
                linear_to_db(self.fade_out_gain(t))
            })
            .collect()
    }

    /// Sample the dB-space gain for the incoming signal at `n` points.
    #[must_use]
    pub fn sample_db_in(self, n: usize) -> Vec<f64> {
        if n == 0 {
            return Vec::new();
        }
        (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1).max(1) as f64;
                linear_to_db(self.fade_in_gain(t))
            })
            .collect()
    }
}

// ── crossfade segment ─────────────────────────────────────────────────────────

/// Describes the overlap segment between two playlist items.
#[derive(Debug, Clone)]
pub struct CrossfadeSegment {
    /// Duration of the overlap.
    pub duration: Duration,
    /// Fade curve applied.
    pub curve: FadeCurve,
    /// Offset from the end of the outgoing item where the crossfade starts.
    pub start_offset_from_end: Duration,
    /// Whether to trim silence from the end of the outgoing item.
    pub trim_trailing_silence: bool,
}

impl CrossfadeSegment {
    /// Create a simple crossfade segment.
    #[must_use]
    pub const fn new(duration: Duration, curve: FadeCurve) -> Self {
        Self {
            duration,
            curve,
            start_offset_from_end: duration,
            trim_trailing_silence: false,
        }
    }

    /// Enable trailing-silence trimming.
    #[must_use]
    pub const fn with_silence_trim(mut self) -> Self {
        self.trim_trailing_silence = true;
        self
    }

    /// The offset from the *start* of the outgoing item at which crossfade begins,
    /// given that item's total duration.
    #[must_use]
    pub fn crossfade_start(&self, item_duration: Duration) -> Duration {
        if item_duration > self.start_offset_from_end {
            item_duration
                .checked_sub(self.start_offset_from_end)
                .unwrap_or(Duration::ZERO)
        } else {
            Duration::ZERO
        }
    }
}

// ── gapless sequencer ─────────────────────────────────────────────────────────

/// Entry in a gapless playlist sequence.
#[derive(Debug, Clone)]
pub struct GaplessEntry {
    /// Track identifier.
    pub track_id: u64,
    /// Track duration.
    pub duration: Duration,
    /// Crossfade segment to apply *after* this entry (before the next).
    /// `None` = hard cut.
    pub crossfade: Option<CrossfadeSegment>,
}

impl GaplessEntry {
    /// Create a gapless entry with no crossfade.
    #[must_use]
    pub const fn new(track_id: u64, duration: Duration) -> Self {
        Self {
            track_id,
            duration,
            crossfade: None,
        }
    }

    /// Set a crossfade segment.
    #[must_use]
    pub fn with_crossfade(mut self, seg: CrossfadeSegment) -> Self {
        self.crossfade = Some(seg);
        self
    }
}

/// Compute playback schedule for a gapless sequence.
/// Returns `(track_id, start_time)` pairs where `start_time` is when
/// each track should begin (relative to sequence start).
#[must_use]
pub fn schedule_gapless(entries: &[GaplessEntry]) -> Vec<(u64, Duration)> {
    let mut schedule = Vec::with_capacity(entries.len());
    let mut cursor = Duration::ZERO;
    for (i, entry) in entries.iter().enumerate() {
        schedule.push((entry.track_id, cursor));
        // Advance cursor by this entry's duration minus the next crossfade overlap.
        let overlap = if i + 1 < entries.len() {
            entry
                .crossfade
                .as_ref()
                .map_or(Duration::ZERO, |cf| cf.duration)
        } else {
            Duration::ZERO
        };
        cursor += entry.duration.saturating_sub(overlap);
    }
    schedule
}

/// Total wall-clock duration of a gapless sequence (accounting for overlaps).
#[must_use]
pub fn sequence_duration(entries: &[GaplessEntry]) -> Duration {
    if entries.is_empty() {
        return Duration::ZERO;
    }
    let mut total = Duration::ZERO;
    for (i, entry) in entries.iter().enumerate() {
        let overlap = if i + 1 < entries.len() {
            entry
                .crossfade
                .as_ref()
                .map_or(Duration::ZERO, |cf| cf.duration)
        } else {
            Duration::ZERO
        };
        total += entry.duration.saturating_sub(overlap);
    }
    // Add final entry's full duration.
    total
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_linear_fade_out_endpoints() {
        assert!(approx_eq(FadeCurve::Linear.fade_out_gain(0.0), 1.0, 1e-9));
        assert!(approx_eq(FadeCurve::Linear.fade_out_gain(1.0), 0.0, 1e-9));
    }

    #[test]
    fn test_linear_fade_in_endpoints() {
        assert!(approx_eq(FadeCurve::Linear.fade_in_gain(0.0), 0.0, 1e-9));
        assert!(approx_eq(FadeCurve::Linear.fade_in_gain(1.0), 1.0, 1e-9));
    }

    #[test]
    fn test_equal_power_energy_constant() {
        // For equal-power, fade_out^2 + fade_in^2 ≈ 1 at all points.
        let curve = FadeCurve::EqualPower;
        for i in 0..=10 {
            let t = i as f64 / 10.0;
            let out = curve.fade_out_gain(t);
            let inp = curve.fade_in_gain(t);
            assert!(
                approx_eq(out * out + inp * inp, 1.0, 1e-9),
                "t={t}, out={out}, in={inp}"
            );
        }
    }

    #[test]
    fn test_s_curve_endpoints() {
        assert!(approx_eq(FadeCurve::SCurve.fade_out_gain(0.0), 1.0, 1e-9));
        assert!(approx_eq(FadeCurve::SCurve.fade_out_gain(1.0), 0.0, 1e-9));
        assert!(approx_eq(FadeCurve::SCurve.fade_in_gain(0.0), 0.0, 1e-9));
        assert!(approx_eq(FadeCurve::SCurve.fade_in_gain(1.0), 1.0, 1e-9));
    }

    #[test]
    fn test_exponential_endpoints() {
        assert!(approx_eq(
            FadeCurve::Exponential.fade_out_gain(0.0),
            1.0,
            1e-9
        ));
        assert!(approx_eq(
            FadeCurve::Exponential.fade_out_gain(1.0),
            0.0,
            1e-9
        ));
    }

    // ── Logarithmic crossfade tests ──────────────────────────────────────────

    #[test]
    fn test_logarithmic_fade_out_endpoints() {
        // At t=0 the outgoing signal should be at unity (0 dBFS → gain 1.0).
        assert!(approx_eq(
            FadeCurve::Logarithmic.fade_out_gain(0.0),
            1.0,
            1e-6
        ));
        // At t=1 the outgoing signal should be at silence.
        assert!(approx_eq(
            FadeCurve::Logarithmic.fade_out_gain(1.0),
            0.0,
            1e-6
        ));
    }

    #[test]
    fn test_logarithmic_fade_in_endpoints() {
        // At t=0 the incoming signal should be silent.
        assert!(approx_eq(
            FadeCurve::Logarithmic.fade_in_gain(0.0),
            0.0,
            1e-6
        ));
        // At t=1 the incoming signal should be at unity.
        assert!(approx_eq(
            FadeCurve::Logarithmic.fade_in_gain(1.0),
            1.0,
            1e-6
        ));
    }

    #[test]
    fn test_logarithmic_fade_out_is_monotonically_decreasing() {
        let curve = FadeCurve::Logarithmic;
        let mut prev = curve.fade_out_gain(0.0);
        for i in 1..=20 {
            let t = i as f64 / 20.0;
            let cur = curve.fade_out_gain(t);
            assert!(cur <= prev + 1e-12, "Not monotone at t={t}: {cur} > {prev}");
            prev = cur;
        }
    }

    #[test]
    fn test_logarithmic_fade_in_is_monotonically_increasing() {
        let curve = FadeCurve::Logarithmic;
        let mut prev = curve.fade_in_gain(0.0);
        for i in 1..=20 {
            let t = i as f64 / 20.0;
            let cur = curve.fade_in_gain(t);
            assert!(cur >= prev - 1e-12, "Not monotone at t={t}: {cur} < {prev}");
            prev = cur;
        }
    }

    #[test]
    fn test_logarithmic_db_space_is_linear() {
        // The defining property: gain_db decreases linearly with t.
        let curve = FadeCurve::Logarithmic;
        // Check that the dB progression is roughly linear between non-floor points.
        let db_at_0 = FadeCurve::linear_to_db(curve.fade_out_gain(0.0)); // ≈ 0 dB
        let db_at_half = FadeCurve::linear_to_db(curve.fade_out_gain(0.5)); // ≈ -48 dB
        let db_at_1 = FadeCurve::linear_to_db(curve.fade_out_gain(1.0)); // = -96 dB (floor)
                                                                         // Midpoint should be near -48 dBFS
        assert!(
            approx_eq(db_at_half, -48.0, 1.0),
            "Expected ≈ -48 dB at t=0.5, got {db_at_half}"
        );
        // Start should be 0 dBFS
        assert!(
            approx_eq(db_at_0, 0.0, 0.1),
            "Expected 0 dB at t=0, got {db_at_0}"
        );
        // End should be at or below floor
        assert!(
            db_at_1 <= -96.0 + 0.1,
            "Expected floor at t=1, got {db_at_1}"
        );
    }

    #[test]
    fn test_logarithmic_gain_values_in_range() {
        let curve = FadeCurve::Logarithmic;
        for i in 0..=100 {
            let t = i as f64 / 100.0;
            let g_out = curve.fade_out_gain(t);
            let g_in = curve.fade_in_gain(t);
            assert!(
                (0.0..=1.0001).contains(&g_out),
                "fade_out out of range at t={t}: {g_out}"
            );
            assert!(
                (0.0..=1.0001).contains(&g_in),
                "fade_in out of range at t={t}: {g_in}"
            );
        }
    }

    #[test]
    fn test_logarithmic_symmetric_endpoints() {
        let curve = FadeCurve::LogarithmicSymmetric;
        assert!(approx_eq(curve.fade_out_gain(0.0), 1.0, 1e-6));
        assert!(approx_eq(curve.fade_out_gain(1.0), 0.0, 1e-6));
        assert!(approx_eq(curve.fade_in_gain(0.0), 0.0, 1e-6));
        assert!(approx_eq(curve.fade_in_gain(1.0), 1.0, 1e-6));
    }

    #[test]
    fn test_logarithmic_symmetric_dip_at_midpoint() {
        // For a symmetric log crossfade the combined power at the midpoint
        // should be less than 1 (i.e., a dip compared to equal-power).
        let curve = FadeCurve::LogarithmicSymmetric;
        let out_mid = curve.fade_out_gain(0.5);
        let in_mid = curve.fade_in_gain(0.5);
        let power_mid = out_mid * out_mid + in_mid * in_mid;
        // Equal-power gives power=1; symmetric log gives power < 1 (a dip).
        assert!(
            power_mid < 0.99,
            "Expected power dip at midpoint, got {power_mid}"
        );
    }

    #[test]
    fn test_sample_db_out_length() {
        let v = FadeCurve::Logarithmic.sample_db_out(8);
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_sample_db_in_length() {
        let v = FadeCurve::Logarithmic.sample_db_in(8);
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_combined_db_equal_power_approximately_constant() {
        let curve = FadeCurve::EqualPower;
        for i in 0..=10 {
            let t = i as f64 / 10.0;
            let db = curve.combined_db(t);
            // Combined power ≈ 1.0 ≡ 0 dBFS for equal-power.
            assert!(
                approx_eq(db, 0.0, 0.1),
                "EqualPower combined_db at t={t}: {db}"
            );
        }
    }

    #[test]
    fn test_crossfade_with_log_curve_in_gapless_sequence() {
        let cf = CrossfadeSegment::new(Duration::from_secs(5), FadeCurve::Logarithmic);
        let entries = vec![
            GaplessEntry::new(1, Duration::from_secs(60)).with_crossfade(cf),
            GaplessEntry::new(2, Duration::from_secs(90)),
        ];
        let sched = schedule_gapless(&entries);
        assert_eq!(sched[1].1, Duration::from_secs(55));
    }

    #[test]
    fn test_fade_clamped_out_of_range() {
        let v = FadeCurve::Linear.fade_out_gain(1.5);
        assert!((0.0..=1.0).contains(&v));
        let v2 = FadeCurve::Linear.fade_in_gain(-0.5);
        assert!((0.0..=1.0).contains(&v2));
    }

    #[test]
    fn test_sample_pair_length() {
        let (out, inp) = FadeCurve::Linear.sample_pair(5);
        assert_eq!(out.len(), 5);
        assert_eq!(inp.len(), 5);
    }

    #[test]
    fn test_sample_pair_empty() {
        let (out, inp) = FadeCurve::Linear.sample_pair(0);
        assert!(out.is_empty());
        assert!(inp.is_empty());
    }

    #[test]
    fn test_crossfade_segment_start_offset() {
        let seg = CrossfadeSegment::new(Duration::from_secs(5), FadeCurve::Linear);
        let item_dur = Duration::from_secs(30);
        assert_eq!(seg.crossfade_start(item_dur), Duration::from_secs(25));
    }

    #[test]
    fn test_crossfade_segment_start_offset_short_item() {
        let seg = CrossfadeSegment::new(Duration::from_secs(10), FadeCurve::Linear);
        let item_dur = Duration::from_secs(5);
        assert_eq!(seg.crossfade_start(item_dur), Duration::ZERO);
    }

    #[test]
    fn test_schedule_gapless_no_overlap() {
        let entries = vec![
            GaplessEntry::new(1, Duration::from_secs(60)),
            GaplessEntry::new(2, Duration::from_secs(90)),
        ];
        let sched = schedule_gapless(&entries);
        assert_eq!(sched[0], (1, Duration::ZERO));
        assert_eq!(sched[1], (2, Duration::from_secs(60)));
    }

    #[test]
    fn test_schedule_gapless_with_overlap() {
        let cf = CrossfadeSegment::new(Duration::from_secs(5), FadeCurve::EqualPower);
        let entries = vec![
            GaplessEntry::new(1, Duration::from_secs(60)).with_crossfade(cf),
            GaplessEntry::new(2, Duration::from_secs(90)),
        ];
        let sched = schedule_gapless(&entries);
        assert_eq!(sched[0].1, Duration::ZERO);
        assert_eq!(sched[1].1, Duration::from_secs(55)); // 60 - 5 overlap
    }

    #[test]
    fn test_schedule_gapless_empty() {
        let sched = schedule_gapless(&[]);
        assert!(sched.is_empty());
    }

    #[test]
    fn test_sequence_duration_no_overlap() {
        let entries = vec![
            GaplessEntry::new(1, Duration::from_secs(60)),
            GaplessEntry::new(2, Duration::from_secs(30)),
        ];
        // Without overlap: 60 + 30 = 90 s.
        assert_eq!(sequence_duration(&entries), Duration::from_secs(90));
    }

    #[test]
    fn test_sequence_duration_empty() {
        assert_eq!(sequence_duration(&[]), Duration::ZERO);
    }

    // ── Wave 27: edge-case hardening ─────────────────────────────────────────

    #[test]
    fn test_schedule_gapless_single_item() {
        // A single entry starts at ZERO; there is no following entry, so the
        // trailing-crossfade overlap is never applied.
        let entries = vec![GaplessEntry::new(7, Duration::from_secs(42))];
        let sched = schedule_gapless(&entries);
        assert_eq!(sched.len(), 1);
        assert_eq!(sched[0], (7, Duration::ZERO));
    }

    #[test]
    fn test_schedule_gapless_single_item_with_crossfade_ignored() {
        // Even when a single entry carries a crossfade segment, the last entry
        // has no successor so its overlap is dropped — it still starts at ZERO.
        let cf = CrossfadeSegment::new(Duration::from_secs(5), FadeCurve::EqualPower);
        let entries = vec![GaplessEntry::new(7, Duration::from_secs(42)).with_crossfade(cf)];
        let sched = schedule_gapless(&entries);
        assert_eq!(sched, vec![(7, Duration::ZERO)]);
    }

    #[test]
    fn test_zero_duration_entry_no_nan() {
        // A zero-duration entry carrying a crossfade longer than itself must
        // not panic or underflow; saturating_sub clamps the advance to ZERO.
        let cf = CrossfadeSegment::new(Duration::from_secs(3), FadeCurve::Logarithmic);
        let entries = vec![
            GaplessEntry::new(1, Duration::ZERO).with_crossfade(cf),
            GaplessEntry::new(2, Duration::from_secs(10)),
        ];
        let sched = schedule_gapless(&entries);
        assert_eq!(sched.len(), 2);
        // First starts at ZERO; second cannot start before it (no underflow).
        assert_eq!(sched[0].1, Duration::ZERO);
        assert_eq!(sched[1].1, Duration::ZERO);
        // sequence_duration is finite and well-defined for the same input.
        let total = sequence_duration(&entries);
        assert_eq!(total, Duration::from_secs(10));
    }

    #[test]
    fn test_fade_curves_no_nan_at_boundaries() {
        // Every curve, at the canonical boundary positions, must yield finite
        // gains within [0, 1] with the documented endpoint conventions:
        //   fade_out(0.0) == 1.0, fade_out(1.0) == 0.0
        //   fade_in(0.0)  == 0.0, fade_in(1.0)  == 1.0
        let curves = [
            FadeCurve::Linear,
            FadeCurve::EqualPower,
            FadeCurve::Logarithmic,
            FadeCurve::LogarithmicSymmetric,
            FadeCurve::Exponential,
            FadeCurve::SCurve,
        ];
        for curve in curves {
            for &t in &[0.0_f64, 0.5, 1.0] {
                let out = curve.fade_out_gain(t);
                let inp = curve.fade_in_gain(t);
                assert!(out.is_finite(), "fade_out NaN/inf for {curve:?} at t={t}");
                assert!(inp.is_finite(), "fade_in NaN/inf for {curve:?} at t={t}");
                assert!(
                    (0.0..=1.0001).contains(&out),
                    "fade_out out of range for {curve:?} at t={t}: {out}"
                );
                assert!(
                    (0.0..=1.0001).contains(&inp),
                    "fade_in out of range for {curve:?} at t={t}: {inp}"
                );
            }
            // Endpoint conventions hold for every curve.
            assert!(
                approx_eq(curve.fade_out_gain(0.0), 1.0, 1e-6),
                "fade_out(0.0) != 1.0 for {curve:?}"
            );
            assert!(
                approx_eq(curve.fade_out_gain(1.0), 0.0, 1e-6),
                "fade_out(1.0) != 0.0 for {curve:?}"
            );
            assert!(
                approx_eq(curve.fade_in_gain(0.0), 0.0, 1e-6),
                "fade_in(0.0) != 0.0 for {curve:?}"
            );
            assert!(
                approx_eq(curve.fade_in_gain(1.0), 1.0, 1e-6),
                "fade_in(1.0) != 1.0 for {curve:?}"
            );
        }
    }
}
