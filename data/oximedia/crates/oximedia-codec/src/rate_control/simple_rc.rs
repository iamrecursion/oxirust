//! Simplified rate control algorithms for video encoding.
//!
//! This module provides straightforward rate control implementations suitable
//! for educational use and lightweight encoders. For production-grade rate
//! control with lookahead, AQ, and two-pass support, see the other submodules.
//!
//! # Modes Supported
//!
//! - **CRF / Constant Quality**: quality-driven QP derivation
//! - **ABR / Average Bitrate**: complexity-weighted bit allocation
//! - **CBR / Constant Bitrate**: ABR with VBV buffer clamping
//! - **VBR / Variable Bitrate**: min/max clamped ABR
//! - **Two-Pass ABR**: complexity-histogram scaling on the second pass

#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::collections::VecDeque;

// ──────────────────────────────────────────────
// Rate control mode
// ──────────────────────────────────────────────

/// Simplified rate control mode for video encoding.
///
/// Each variant carries all parameters needed to drive the corresponding
/// algorithm without requiring a separate configuration struct.
#[derive(Debug, Clone)]
pub enum SimpleRateControlMode {
    /// Constant-quality mode (CRF / CQ).
    ///
    /// The `crf` value (0–63, lower = higher quality) drives a QP calculation.
    /// Bitrate is not constrained; it grows with content complexity.
    ConstantQuality {
        /// Constant Rate Factor (0–63).
        crf: u8,
    },

    /// Average-bitrate mode (ABR).
    ///
    /// The encoder targets `target_kbps` averaged over the whole clip.
    /// Individual frames may deviate based on content complexity.
    AverageBitrate {
        /// Target average bitrate in kbps.
        target_kbps: u32,
    },

    /// Constant-bitrate mode (CBR) with a VBV (Video Buffer Verifier) buffer.
    ///
    /// Per-frame bits are clamped so that the VBV never overflows (fullness > 0.9)
    /// or starves (fullness < 0.3).
    ConstantBitrate {
        /// Target constant bitrate in kbps.
        target_kbps: u32,
        /// VBV buffer size in kilobits.
        vbv_size_kb: u32,
    },

    /// Variable-bitrate mode (VBR).
    ///
    /// Like ABR but the instantaneous bitrate is clamped to `[min_kbps, max_kbps]`.
    VariableBitrate {
        /// Minimum instantaneous bitrate in kbps.
        min_kbps: u32,
        /// Maximum instantaneous bitrate in kbps.
        max_kbps: u32,
        /// Long-term average target in kbps.
        target_kbps: u32,
    },

    /// Two-pass average-bitrate mode.
    ///
    /// On the second pass, bits are allocated proportional to the per-frame
    /// complexity ratios recorded during the first pass.  `first_pass_stats`
    /// is a newline-separated string of `f32` complexity values, one per
    /// frame (e.g. produced by [`SimpleRateController::record_frame`]).
    TwoPass {
        /// Target average bitrate in kbps.
        target_kbps: u32,
        /// Serialised first-pass complexity data (one `f32` per line).
        first_pass_stats: Option<String>,
    },
}

// ──────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────

/// Configuration for [`SimpleRateController`].
#[derive(Debug, Clone)]
pub struct SimpleRateControlConfig {
    /// Rate control algorithm and its parameters.
    pub mode: SimpleRateControlMode,
    /// Frames per second of the encoded video.
    pub fps: f32,
    /// Encoded picture resolution as `(width, height)` in pixels.
    pub resolution: (u32, u32),
    /// IDR / keyframe interval in frames.
    pub keyframe_interval: u32,
    /// Number of B-frames between reference frames.
    pub b_frames: u8,
    /// Scene-change sensitivity (0 = disabled, 100 = most sensitive).
    pub scene_change_sensitivity: u8,
}

impl SimpleRateControlConfig {
    /// Construct a constant-quality configuration.
    pub fn crf(crf: u8, fps: f32, resolution: (u32, u32)) -> Self {
        Self {
            mode: SimpleRateControlMode::ConstantQuality { crf },
            fps,
            resolution,
            keyframe_interval: 120,
            b_frames: 2,
            scene_change_sensitivity: 40,
        }
    }

    /// Construct an average-bitrate configuration.
    pub fn abr(target_kbps: u32, fps: f32, resolution: (u32, u32)) -> Self {
        Self {
            mode: SimpleRateControlMode::AverageBitrate { target_kbps },
            fps,
            resolution,
            keyframe_interval: 120,
            b_frames: 2,
            scene_change_sensitivity: 40,
        }
    }

    /// Construct a constant-bitrate configuration.
    pub fn cbr(target_kbps: u32, vbv_size_kb: u32, fps: f32, resolution: (u32, u32)) -> Self {
        Self {
            mode: SimpleRateControlMode::ConstantBitrate {
                target_kbps,
                vbv_size_kb,
            },
            fps,
            resolution,
            keyframe_interval: 120,
            b_frames: 0,
            scene_change_sensitivity: 20,
        }
    }
}

// ──────────────────────────────────────────────
// Statistics
// ──────────────────────────────────────────────

/// Per-session rate control statistics snapshot.
#[derive(Debug, Clone)]
pub struct SimpleRateControlStats {
    /// Number of frames encoded so far.
    pub frames_encoded: u64,
    /// Total bits produced by the encoder (sum of `actual_bits` passed to
    /// [`SimpleRateController::record_frame`]).
    pub total_bits: u64,
    /// Running mean of bits per frame.
    pub avg_bits_per_frame: f64,
    /// Running mean of the complexity values supplied to the controller.
    pub avg_complexity: f32,
    /// Current VBV buffer fullness in `[0.0, 1.0]`.  0 = empty, 1 = full.
    pub vbv_fullness: f64,
    /// Target bitrate in kbps derived from the mode, or 0 for CRF mode.
    pub target_bitrate_kbps: u32,
    /// Actual measured bitrate in kbps based on bits spent and frames encoded.
    pub actual_bitrate_kbps: f64,
}

// ──────────────────────────────────────────────
// Controller
// ──────────────────────────────────────────────

/// Maximum number of recent complexity samples kept for rolling mean.
const COMPLEXITY_HISTORY_LEN: usize = 64;

/// A simplified, self-contained rate controller.
///
/// # Usage
///
/// ```rust
/// use oximedia_codec::rate_control::{SimpleRateControlConfig, SimpleRateController};
///
/// let cfg = SimpleRateControlConfig::abr(4000, 30.0, (1920, 1080));
/// let mut rc = SimpleRateController::new(cfg);
///
/// for i in 0..300_u32 {
///     let complexity = 0.5_f32;                     // supplied by analyser
///     let bits = rc.allocate_frame_bits(complexity);
///     rc.record_frame(bits, complexity);
/// }
///
/// let stats = rc.stats();
/// assert!(stats.frames_encoded == 300);
/// ```
pub struct SimpleRateController {
    /// Immutable configuration supplied at construction time.
    config: SimpleRateControlConfig,
    /// Running frame counter (incremented in `record_frame`).
    frame_count: u64,
    /// Cumulative bits recorded via `record_frame`.
    bits_spent: u64,
    /// Pre-computed target bits per frame (0 for CRF).
    target_bits: u64,
    /// Current VBV buffer fullness as a fraction `[0.0, 1.0]`.
    vbv_fullness: f64,
    /// Sliding window of recent complexity values for mean estimation.
    complexity_history: VecDeque<f32>,
    /// Decoded first-pass complexity histogram for two-pass ABR.
    /// Each element corresponds to a frame index.
    first_pass_complexities: Vec<f32>,
    /// Cached mean of `first_pass_complexities`.
    first_pass_mean: f32,
}

impl SimpleRateController {
    /// Create a new controller from `config`.
    ///
    /// The VBV buffer starts at 50 % fullness for CBR mode and at 0 for all
    /// other modes.
    pub fn new(config: SimpleRateControlConfig) -> Self {
        let target_bits = compute_target_bits_per_frame(&config);
        let vbv_fullness = match &config.mode {
            SimpleRateControlMode::ConstantBitrate { .. } => 0.5,
            _ => 0.0,
        };

        let (first_pass_complexities, first_pass_mean) = parse_first_pass_stats(&config.mode);

        Self {
            config,
            frame_count: 0,
            bits_spent: 0,
            target_bits,
            vbv_fullness,
            complexity_history: VecDeque::with_capacity(COMPLEXITY_HISTORY_LEN),
            first_pass_complexities,
            first_pass_mean,
        }
    }

    // ── Internal helpers ────────────────────────────────────────────────────

    /// Rolling mean of the complexity history.
    ///
    /// Falls back to `1.0` when the history is empty to avoid division by zero.
    fn avg_complexity(&self) -> f32 {
        if self.complexity_history.is_empty() {
            return 1.0;
        }
        let sum: f32 = self.complexity_history.iter().sum();
        sum / self.complexity_history.len() as f32
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /// Return the pre-computed target bits-per-frame for the current mode.
    ///
    /// CRF mode always returns 0 because bit budget is implicitly derived from
    /// the QP rather than a bitrate target.
    pub fn target_bits_per_frame(&self) -> u64 {
        self.target_bits
    }

    /// Decide how many bits to allocate for the next frame given its
    /// `complexity` estimate (any non-negative float, relative scale).
    ///
    /// The return value is a **recommendation** in bits; the caller feeds the
    /// actual encoded size back via `record_frame`.
    pub fn allocate_frame_bits(&mut self, complexity: f32) -> u32 {
        let complexity = complexity.max(0.0001_f32);
        match &self.config.mode.clone() {
            SimpleRateControlMode::ConstantQuality { crf } => {
                let (w, h) = self.config.resolution;
                let qp = (*crf as f32).max(1.0);
                let bits = (w as f64 * h as f64 / (qp * qp) as f64) as u64;
                bits.min(u32::MAX as u64) as u32
            }

            SimpleRateControlMode::AverageBitrate { .. } => {
                let avg = self.avg_complexity();
                let ratio = (complexity / avg) as f64;
                let allocated = (self.target_bits as f64 * ratio).round() as u64;
                // Allow up to 4× target to absorb complexity peaks.
                allocated.clamp(1, self.target_bits.saturating_mul(4)) as u32
            }

            SimpleRateControlMode::VariableBitrate {
                min_kbps, max_kbps, ..
            } => {
                let min_bits = (*min_kbps as f64 * 1000.0 / self.config.fps as f64) as u64;
                let max_bits = (*max_kbps as f64 * 1000.0 / self.config.fps as f64) as u64;
                let avg = self.avg_complexity();
                let ratio = (complexity / avg) as f64;
                let allocated = (self.target_bits as f64 * ratio).round() as u64;
                allocated.clamp(min_bits.max(1), max_bits.max(1)) as u32
            }

            SimpleRateControlMode::ConstantBitrate { vbv_size_kb, .. } => {
                let avg = self.avg_complexity();
                let ratio = (complexity / avg) as f64;
                let mut allocated = (self.target_bits as f64 * ratio).round() as u64;

                // VBV clamping: reduce bits when buffer is getting full,
                // increase when it is running low.
                let vbv_bits = (*vbv_size_kb as u64).saturating_mul(1000);
                if self.vbv_fullness > 0.9 {
                    // Buffer almost full — cut bits to at most 50 % of target.
                    let cap = (self.target_bits as f64 * 0.5).round() as u64;
                    allocated = allocated.min(cap);
                } else if self.vbv_fullness < 0.3 {
                    // Buffer running low — allow up to 150 % of target.
                    let floor = (self.target_bits as f64 * 1.5).round() as u64;
                    allocated = allocated.max(floor);
                }

                // Hard cap at VBV buffer size to prevent overflow in a single frame.
                allocated.clamp(1, vbv_bits.max(1)) as u32
            }

            SimpleRateControlMode::TwoPass { .. } => {
                let idx = self.frame_count as usize;
                if !self.first_pass_complexities.is_empty() && self.first_pass_mean > 0.0 {
                    let frame_cplx = if idx < self.first_pass_complexities.len() {
                        self.first_pass_complexities[idx]
                    } else {
                        self.first_pass_mean
                    };
                    let ratio = (frame_cplx / self.first_pass_mean) as f64;
                    let allocated = (self.target_bits as f64 * ratio).round() as u64;
                    allocated.clamp(1, self.target_bits.saturating_mul(4)) as u32
                } else {
                    // No first-pass data: fall back to flat allocation.
                    self.target_bits.clamp(1, u32::MAX as u64) as u32
                }
            }
        }
    }

    /// Record the outcome of encoding a frame.
    ///
    /// - `actual_bits`: real encoded size in bits.
    /// - `complexity`: the same complexity estimate passed to
    ///   `allocate_frame_bits` for this frame.
    ///
    /// This updates internal accounting (VBV fullness, bit budget, complexity
    /// history) so that subsequent allocations are informed by history.
    pub fn record_frame(&mut self, actual_bits: u32, complexity: f32) {
        self.bits_spent = self.bits_spent.saturating_add(actual_bits as u64);
        self.frame_count = self.frame_count.saturating_add(1);

        // Update sliding complexity window.
        if self.complexity_history.len() >= COMPLEXITY_HISTORY_LEN {
            self.complexity_history.pop_front();
        }
        self.complexity_history.push_back(complexity.max(0.0));

        // Update VBV buffer for CBR mode.
        if let SimpleRateControlMode::ConstantBitrate {
            target_kbps,
            vbv_size_kb,
        } = &self.config.mode
        {
            let vbv_capacity = (*vbv_size_kb as f64) * 1000.0;
            if vbv_capacity > 0.0 {
                let drained = actual_bits as f64;
                let refilled = (*target_kbps as f64 * 1000.0) / self.config.fps as f64;
                // Buffer is refilled by the channel rate and drained by encoded bits.
                let delta = (refilled - drained) / vbv_capacity;
                self.vbv_fullness = (self.vbv_fullness + delta).clamp(0.0, 1.0);
            }
        }
    }

    /// Current VBV buffer fullness: 0.0 = empty, 1.0 = full.
    ///
    /// For non-CBR modes this always returns 0.0.
    pub fn vbv_status(&self) -> f64 {
        self.vbv_fullness
    }

    /// Return `true` if the current frame index is a keyframe position.
    ///
    /// Uses the `keyframe_interval` from the configuration.  Frame index 0
    /// (the very first frame) is always considered a keyframe.
    pub fn is_keyframe(&self) -> bool {
        let interval = self.config.keyframe_interval.max(1) as u64;
        self.frame_count % interval == 0
    }

    /// Snapshot of accumulated statistics.
    pub fn stats(&self) -> SimpleRateControlStats {
        let target_bitrate_kbps = extract_target_kbps(&self.config.mode);
        let avg_bits_per_frame = if self.frame_count == 0 {
            0.0
        } else {
            self.bits_spent as f64 / self.frame_count as f64
        };
        let actual_bitrate_kbps = avg_bits_per_frame * self.config.fps as f64 / 1000.0;

        SimpleRateControlStats {
            frames_encoded: self.frame_count,
            total_bits: self.bits_spent,
            avg_bits_per_frame,
            avg_complexity: self.avg_complexity(),
            vbv_fullness: self.vbv_fullness,
            target_bitrate_kbps,
            actual_bitrate_kbps,
        }
    }
}

// ──────────────────────────────────────────────
// Free-standing helpers
// ──────────────────────────────────────────────

/// Compute the per-frame bit budget for bitrate-based modes.
///
/// Returns 0 for CRF mode (bit budget is implicit in QP).
fn compute_target_bits_per_frame(config: &SimpleRateControlConfig) -> u64 {
    let fps = config.fps.max(0.001_f32) as f64;
    match &config.mode {
        SimpleRateControlMode::ConstantQuality { .. } => 0,
        SimpleRateControlMode::AverageBitrate { target_kbps }
        | SimpleRateControlMode::ConstantBitrate { target_kbps, .. }
        | SimpleRateControlMode::VariableBitrate { target_kbps, .. }
        | SimpleRateControlMode::TwoPass { target_kbps, .. } => {
            (*target_kbps as f64 * 1000.0 / fps).round() as u64
        }
    }
}

/// Extract a representative `target_kbps` value for stats reporting.
///
/// Returns 0 for CRF because there is no bitrate target.
fn extract_target_kbps(mode: &SimpleRateControlMode) -> u32 {
    match mode {
        SimpleRateControlMode::ConstantQuality { .. } => 0,
        SimpleRateControlMode::AverageBitrate { target_kbps }
        | SimpleRateControlMode::ConstantBitrate { target_kbps, .. }
        | SimpleRateControlMode::VariableBitrate { target_kbps, .. }
        | SimpleRateControlMode::TwoPass { target_kbps, .. } => *target_kbps,
    }
}

/// Parse the `first_pass_stats` string for TwoPass mode.
///
/// Returns `(complexities, mean)`.  Both are empty / zero when the mode is not
/// TwoPass or when the stats string is absent.
fn parse_first_pass_stats(mode: &SimpleRateControlMode) -> (Vec<f32>, f32) {
    if let SimpleRateControlMode::TwoPass {
        first_pass_stats: Some(stats),
        ..
    } = mode
    {
        let values: Vec<f32> = stats
            .lines()
            .filter_map(|l| l.trim().parse::<f32>().ok())
            .collect();
        let mean = if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f32>() / values.len() as f32
        };
        return (values, mean);
    }
    (Vec::new(), 0.0)
}

// ──────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_crf(crf: u8) -> SimpleRateController {
        SimpleRateController::new(SimpleRateControlConfig::crf(crf, 30.0, (1920, 1080)))
    }

    fn make_abr(kbps: u32) -> SimpleRateController {
        SimpleRateController::new(SimpleRateControlConfig::abr(kbps, 30.0, (1920, 1080)))
    }

    fn make_cbr(kbps: u32, vbv_kb: u32) -> SimpleRateController {
        SimpleRateController::new(SimpleRateControlConfig::cbr(
            kbps,
            vbv_kb,
            30.0,
            (1920, 1080),
        ))
    }

    // ── 1. CRF allocation is non-zero ────────────────────────────────────────

    #[test]
    fn crf_allocation_nonzero() {
        let mut rc = make_crf(23);
        let bits = rc.allocate_frame_bits(1.0);
        assert!(bits > 0, "CRF allocation must be positive");
    }

    // ── 2. CRF: lower crf (higher quality) → more bits ──────────────────────

    #[test]
    fn crf_lower_crf_more_bits() {
        let mut hi_q = make_crf(10);
        let mut lo_q = make_crf(40);
        let hi = hi_q.allocate_frame_bits(1.0);
        let lo = lo_q.allocate_frame_bits(1.0);
        assert!(hi > lo, "crf=10 should yield more bits than crf=40");
    }

    // ── 3. target_bits_per_frame is correct for ABR ─────────────────────────

    #[test]
    fn abr_target_bits_per_frame() {
        let rc = make_abr(4000);
        // 4000 kbps / 30 fps = ~133_333 bits/frame
        let expected = (4_000_000_u64 / 30) as f64;
        let got = rc.target_bits_per_frame() as f64;
        assert!(
            (got - expected).abs() / expected < 0.01,
            "ABR target bits/frame off: got {got}, expected ~{expected}"
        );
    }

    // ── 4. CRF: target_bits_per_frame is 0 ──────────────────────────────────

    #[test]
    fn crf_target_bits_zero() {
        let rc = make_crf(28);
        assert_eq!(rc.target_bits_per_frame(), 0);
    }

    // ── 5. ABR: higher complexity → more bits ────────────────────────────────

    #[test]
    fn abr_high_complexity_more_bits() {
        let mut rc = make_abr(4000);
        // Prime the history with moderate complexity.
        for _ in 0..16 {
            rc.record_frame(100_000, 1.0);
        }
        let low = rc.allocate_frame_bits(0.5);
        let high = rc.allocate_frame_bits(2.0);
        assert!(
            high > low,
            "higher complexity should yield more bits in ABR"
        );
    }

    // ── 6. record_frame increments frame_count ───────────────────────────────

    #[test]
    fn record_frame_increments_count() {
        let mut rc = make_abr(2000);
        for _ in 0..10 {
            rc.record_frame(50_000, 1.0);
        }
        assert_eq!(rc.stats().frames_encoded, 10);
    }

    // ── 7. record_frame accumulates total_bits ───────────────────────────────

    #[test]
    fn record_frame_accumulates_bits() {
        let mut rc = make_abr(2000);
        for _ in 0..5 {
            rc.record_frame(10_000, 1.0);
        }
        assert_eq!(rc.stats().total_bits, 50_000);
    }

    // ── 8. CBR: VBV fullness moves within [0, 1] ────────────────────────────

    #[test]
    fn cbr_vbv_fullness_in_range() {
        let mut rc = make_cbr(4000, 8000);
        for i in 0..60 {
            let bits = if i % 5 == 0 { 1_000_000 } else { 50_000 };
            rc.record_frame(bits, 1.0);
            let f = rc.vbv_status();
            assert!((0.0..=1.0).contains(&f), "VBV fullness out of range: {f}");
        }
    }

    // ── 9. CBR: VBV reduces allocation when near-full ───────────────────────

    #[test]
    fn cbr_vbv_reduces_when_full() {
        let mut rc = make_cbr(4000, 1000); // tiny VBV
                                           // Force VBV to near-full by encoding many frames with zero bits.
        for _ in 0..100 {
            rc.record_frame(0, 1.0); // no bits drained, VBV fills up
        }
        let bits_full = rc.allocate_frame_bits(1.0);
        let target = rc.target_bits_per_frame() as u32;
        // Should be at most 50 % of target when full.
        assert!(
            bits_full <= target.saturating_mul(5) / 10 + 1,
            "CBR should reduce bits when VBV is full; got {bits_full}, target {target}"
        );
    }

    // ── 10. is_keyframe at correct intervals ─────────────────────────────────

    #[test]
    fn is_keyframe_at_correct_intervals() {
        let cfg = SimpleRateControlConfig {
            mode: SimpleRateControlMode::AverageBitrate { target_kbps: 2000 },
            fps: 30.0,
            resolution: (1280, 720),
            keyframe_interval: 10,
            b_frames: 0,
            scene_change_sensitivity: 0,
        };
        let mut rc = SimpleRateController::new(cfg);
        // frame_count=0, 0%10==0 → keyframe.
        assert!(rc.is_keyframe(), "frame_count=0 should be keyframe");
        // Advance through frames 1–9 (non-keyframe positions).
        for i in 1..10u64 {
            rc.record_frame(10_000, 1.0); // increments frame_count
            assert_eq!(rc.frame_count, i);
            assert!(!rc.is_keyframe(), "frame_count={i} should NOT be keyframe");
        }
        // One more record brings frame_count to 10 → keyframe.
        rc.record_frame(10_000, 1.0);
        assert_eq!(rc.frame_count, 10);
        assert!(rc.is_keyframe(), "frame_count=10 should be keyframe");
    }

    // ── 11. stats returns correct avg_bits_per_frame ─────────────────────────

    #[test]
    fn stats_avg_bits_per_frame() {
        let mut rc = make_abr(2000);
        rc.record_frame(200_000, 1.0);
        rc.record_frame(100_000, 1.0);
        let s = rc.stats();
        assert!((s.avg_bits_per_frame - 150_000.0).abs() < 1.0);
    }

    // ── 12. VBR: allocation is clamped between min and max ──────────────────

    #[test]
    fn vbr_allocation_clamped() {
        let cfg = SimpleRateControlConfig {
            mode: SimpleRateControlMode::VariableBitrate {
                min_kbps: 1000,
                max_kbps: 8000,
                target_kbps: 4000,
            },
            fps: 30.0,
            resolution: (1920, 1080),
            keyframe_interval: 120,
            b_frames: 2,
            scene_change_sensitivity: 40,
        };
        let mut rc = SimpleRateController::new(cfg);
        // Prime history.
        for _ in 0..16 {
            rc.record_frame(130_000, 1.0);
        }
        let bits_low = rc.allocate_frame_bits(0.01); // almost no complexity
        let bits_high = rc.allocate_frame_bits(100.0); // extreme complexity
        let min_bits = (1000_u64 * 1000 / 30) as u32;
        let max_bits = (8000_u64 * 1000 / 30) as u32;
        assert!(
            bits_low >= min_bits.saturating_sub(1),
            "VBR low allocation {bits_low} below min {min_bits}"
        );
        assert!(
            bits_high <= max_bits + 1,
            "VBR high allocation {bits_high} above max {max_bits}"
        );
    }

    // ── 13. Two-pass uses first-pass stats ───────────────────────────────────

    #[test]
    fn two_pass_uses_first_pass_stats() {
        let stats = "0.5\n1.0\n2.0\n0.5\n1.0";
        let cfg = SimpleRateControlConfig {
            mode: SimpleRateControlMode::TwoPass {
                target_kbps: 4000,
                first_pass_stats: Some(stats.to_owned()),
            },
            fps: 30.0,
            resolution: (1920, 1080),
            keyframe_interval: 120,
            b_frames: 2,
            scene_change_sensitivity: 40,
        };
        let mut rc = SimpleRateController::new(cfg);

        // Frame 0: complexity 0.5 (below mean 1.0) → fewer bits than target.
        let bits_easy = rc.allocate_frame_bits(0.5);
        rc.record_frame(bits_easy, 0.5);

        // Frame 1: complexity 1.0 (at mean) → ~target.
        let bits_avg = rc.allocate_frame_bits(1.0);
        rc.record_frame(bits_avg, 1.0);

        // Frame 2: complexity 2.0 (above mean) → more bits than target.
        let bits_hard = rc.allocate_frame_bits(2.0);

        assert!(
            bits_hard > bits_easy,
            "Two-pass: complex frame should get more bits than easy frame"
        );
    }

    // ── 14. stats actual_bitrate_kbps is non-zero after recording ────────────

    #[test]
    fn stats_actual_bitrate_nonzero() {
        let mut rc = make_abr(4000);
        for _ in 0..30 {
            rc.record_frame(133_333, 1.0);
        }
        let s = rc.stats();
        assert!(
            s.actual_bitrate_kbps > 0.0,
            "actual bitrate should be positive"
        );
    }

    // ── 15. Two-pass without first-pass data falls back to flat allocation ───

    #[test]
    fn two_pass_no_stats_flat_fallback() {
        let cfg = SimpleRateControlConfig {
            mode: SimpleRateControlMode::TwoPass {
                target_kbps: 2000,
                first_pass_stats: None,
            },
            fps: 25.0,
            resolution: (1280, 720),
            keyframe_interval: 50,
            b_frames: 2,
            scene_change_sensitivity: 40,
        };
        let mut rc = SimpleRateController::new(cfg);
        let bits = rc.allocate_frame_bits(1.0);
        assert!(bits > 0, "fallback two-pass allocation must be positive");
        // Should equal target_bits_per_frame (flat).
        let target = rc.target_bits_per_frame() as u32;
        assert_eq!(
            bits, target,
            "no-stats two-pass should produce flat allocation equal to target"
        );
    }
}
