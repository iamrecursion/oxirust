//! Temporal PSNR — sliding-window averaged PSNR over a frame sequence.
//!
//! Per-frame PSNR doesn't capture temporal consistency: a single glitch frame
//! can be invisible in a long clip-average but is highly perceptible.
//! [`TemporalPsnrAccumulator`] maintains a sliding window of per-frame PSNR
//! values and exposes both a current-window average and an all-time average.
//!
//! ## PSNR clamping
//!
//! Identical frames produce MSE = 0 → PSNR = ∞.  The accumulator clamps
//! such frames to `PSNR_MAX_DB` (100 dB) before storing so that averages
//! remain finite and useful.
//!
//! ## Multi-channel support
//!
//! When `channels == 3` the accumulator expects interleaved YUV triplets
//! (`y0 u0 v0 y1 u1 v1 …`).  PSNR is computed per channel and averaged with
//! the standard 4:1:1 luma-weighting (Y×4, U×1, V×1, normalised to 1).

use std::collections::VecDeque;

/// Maximum PSNR value stored when frames are identical (MSE ≈ 0).
pub const PSNR_MAX_DB: f32 = 100.0;

/// Configuration for [`TemporalPsnrAccumulator`].
#[derive(Debug, Clone)]
pub struct TemporalPsnrConfig {
    /// Sliding-window length in frames.  Default: 5.
    pub window_frames: usize,
    /// Number of channels to consider.
    ///   * `1` → Y-only (luma)
    ///   * `3` → YUV (interleaved; PSNR averaged with 4:1:1 weighting)
    ///
    /// Default: 1.
    pub channels: usize,
}

impl Default for TemporalPsnrConfig {
    fn default() -> Self {
        Self {
            window_frames: 5,
            channels: 1,
        }
    }
}

/// Accumulates per-frame PSNR values and provides sliding-window statistics.
///
/// # Example
///
/// ```
/// use oximedia_quality::temporal_psnr::{TemporalPsnrAccumulator, TemporalPsnrConfig};
///
/// let mut acc = TemporalPsnrAccumulator::new(TemporalPsnrConfig::default());
/// let ref_frame = vec![128u8; 64 * 64];
/// let dist_frame = vec![130u8; 64 * 64];
/// acc.push_frame(&ref_frame, &dist_frame, 64, 64);
/// println!("window avg = {:?}", acc.current_avg());
/// ```
pub struct TemporalPsnrAccumulator {
    config: TemporalPsnrConfig,
    /// Ring-buffer of per-frame PSNR values (clamped).
    history: VecDeque<f32>,
    /// Running sum of *all* pushed frames for `all_frames_avg`.
    all_sum: f64,
    /// Total number of frames pushed.
    all_count: usize,
}

impl TemporalPsnrAccumulator {
    /// Creates a new accumulator with the given configuration.
    #[must_use]
    pub fn new(config: TemporalPsnrConfig) -> Self {
        let cap = config.window_frames.max(1);
        Self {
            config,
            history: VecDeque::with_capacity(cap),
            all_sum: 0.0,
            all_count: 0,
        }
    }

    /// Pushes a new frame pair, computing its PSNR and updating the window.
    ///
    /// When the window is full the oldest value is evicted.
    pub fn push_frame(&mut self, ref_frame: &[u8], dist_frame: &[u8], w: u32, h: u32) {
        let psnr = self.compute_frame_psnr(ref_frame, dist_frame, w, h);
        let clamped = psnr.min(PSNR_MAX_DB);

        // Evict oldest if at capacity.
        if self.history.len() == self.config.window_frames.max(1) {
            self.history.pop_front();
        }
        self.history.push_back(clamped);

        self.all_sum += f64::from(clamped);
        self.all_count += 1;
    }

    /// Returns the arithmetic mean PSNR over the current sliding window,
    /// or `None` if no frames have been pushed yet.
    #[must_use]
    pub fn current_avg(&self) -> Option<f32> {
        if self.history.is_empty() {
            return None;
        }
        let sum: f32 = self.history.iter().sum();
        Some(sum / self.history.len() as f32)
    }

    /// Returns the arithmetic mean PSNR over *all* frames pushed so far,
    /// or `None` if no frames have been pushed yet.
    #[must_use]
    pub fn all_frames_avg(&self) -> Option<f32> {
        if self.all_count == 0 {
            return None;
        }
        Some((self.all_sum / self.all_count as f64) as f32)
    }

    /// Returns a snapshot of the current window values (oldest first).
    #[must_use]
    pub fn window_values(&self) -> Vec<f32> {
        self.history.iter().copied().collect()
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn compute_frame_psnr(&self, ref_frame: &[u8], dist_frame: &[u8], w: u32, h: u32) -> f32 {
        match self.config.channels {
            3 => self.psnr_yuv_interleaved(ref_frame, dist_frame, w, h),
            _ => psnr_plane(ref_frame, dist_frame),
        }
    }

    /// PSNR for interleaved YUV triplets with 4:1:1 luma weighting.
    fn psnr_yuv_interleaved(&self, ref_frame: &[u8], dist_frame: &[u8], w: u32, h: u32) -> f32 {
        let n = (w * h) as usize;
        let total = ref_frame.len().min(dist_frame.len());
        if total < 3 {
            return PSNR_MAX_DB;
        }

        let mut mse_y = 0.0f64;
        let mut mse_u = 0.0f64;
        let mut mse_v = 0.0f64;
        let mut count = 0usize;

        let mut i = 0;
        while i + 2 < total {
            let ry = f64::from(ref_frame[i]);
            let ru = f64::from(ref_frame[i + 1]);
            let rv = f64::from(ref_frame[i + 2]);
            let dy = f64::from(dist_frame[i]);
            let du = f64::from(dist_frame[i + 1]);
            let dv = f64::from(dist_frame[i + 2]);
            mse_y += (ry - dy) * (ry - dy);
            mse_u += (ru - du) * (ru - du);
            mse_v += (rv - dv) * (rv - dv);
            count += 1;
            i += 3;
        }

        let _ = n; // used via w * h above
        if count == 0 {
            return PSNR_MAX_DB;
        }

        let cnt = count as f64;
        let mse_y = mse_y / cnt;
        let mse_u = mse_u / cnt;
        let mse_v = mse_v / cnt;

        // 4:1:1 weighting
        let mse_weighted = (4.0 * mse_y + mse_u + mse_v) / 6.0;
        mse_to_psnr(mse_weighted)
    }
}

/// Computes PSNR from MSE (8-bit, MAX = 255).
///
/// Returns `PSNR_MAX_DB` when MSE < 1e-10 (practically identical).
fn mse_to_psnr(mse: f64) -> f32 {
    if mse < 1e-10 {
        return PSNR_MAX_DB;
    }
    let psnr = 10.0 * (255.0 * 255.0 / mse).log10();
    psnr as f32
}

/// PSNR for a packed 8-bit luma plane (Y-only path).
fn psnr_plane(reference: &[u8], distorted: &[u8]) -> f32 {
    let n = reference.len().min(distorted.len());
    if n == 0 {
        return PSNR_MAX_DB;
    }

    let mse: f64 = reference[..n]
        .iter()
        .zip(distorted[..n].iter())
        .map(|(&r, &d)| {
            let diff = f64::from(r) - f64::from(d);
            diff * diff
        })
        .sum::<f64>()
        / n as f64;

    mse_to_psnr(mse)
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene-aware temporal quality accumulator
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`SceneAwareQualityAccumulator`].
#[derive(Debug, Clone)]
pub struct SceneAwareQualityConfig {
    /// Frame-difference threshold (mean absolute pixel difference, normalised
    /// to `[0.0, 1.0]`) above which a scene cut is declared.
    ///
    /// Default: `0.15`.
    pub scene_cut_threshold: f32,
    /// Minimum number of frames a scene must contain for its average PSNR to
    /// be recorded in `scene_averages`.
    ///
    /// Scenes shorter than this are silently discarded when they end.
    /// Default: `5`.
    pub min_scene_frames: usize,
}

impl Default for SceneAwareQualityConfig {
    fn default() -> Self {
        Self {
            scene_cut_threshold: 0.15,
            min_scene_frames: 5,
        }
    }
}

/// Scene-aware PSNR accumulator.
///
/// Each call to [`push_frame`](SceneAwareQualityAccumulator::push_frame) feeds
/// one reference/distorted frame pair.  The accumulator:
///
/// 1. Computes per-frame PSNR (luma-only, 8-bit).
/// 2. Detects scene cuts by comparing the **distorted** frame against the
///    previous distorted frame using mean absolute difference (MAD).  When the
///    MAD exceeds `config.scene_cut_threshold` a scene boundary is declared.
/// 3. At each boundary, if the current scene is long enough
///    (`≥ config.min_scene_frames`) the scene's average PSNR is appended to
///    `scene_averages` and `Some(avg_psnr)` is returned; otherwise `None`.
///
/// Call [`overall_average`](SceneAwareQualityAccumulator::overall_average) for
/// the mean of all completed scene averages, and
/// [`scene_count`](SceneAwareQualityAccumulator::scene_count) for the number
/// of completed (and accepted) scenes.
pub struct SceneAwareQualityAccumulator {
    config: SceneAwareQualityConfig,
    /// PSNR values accumulated for the current (not yet completed) scene.
    current_scene_psnr: Vec<f32>,
    /// Average PSNR for each completed scene that was long enough.
    scene_averages: Vec<f32>,
    /// Previous distorted frame — used for scene-cut detection.
    prev_frame: Option<Vec<u8>>,
}

impl SceneAwareQualityAccumulator {
    /// Create a new accumulator with the given configuration.
    #[must_use]
    pub fn new(config: SceneAwareQualityConfig) -> Self {
        Self {
            config,
            current_scene_psnr: Vec::new(),
            scene_averages: Vec::new(),
            prev_frame: None,
        }
    }

    /// Feed a frame pair and return the completed scene's average PSNR if a
    /// scene boundary was detected.
    ///
    /// Returns `Some(scene_avg)` only when:
    /// - a scene cut is detected on this frame, **and**
    /// - the just-completed scene had `≥ min_scene_frames` frames.
    ///
    /// Otherwise returns `None`.
    pub fn push_frame(
        &mut self,
        ref_frame: &[u8],
        dist_frame: &[u8],
        w: u32,
        h: u32,
    ) -> Option<f32> {
        // Compute per-frame PSNR for the reference/distorted pair.
        let psnr = psnr_plane(ref_frame, dist_frame).min(PSNR_MAX_DB);

        // Scene-cut detection: MAD between current and previous distorted frame.
        let scene_cut = match &self.prev_frame {
            None => false,
            Some(prev) => {
                let mad = mean_abs_diff(prev, dist_frame);
                mad > self.config.scene_cut_threshold
            }
        };

        // Update the stored previous frame (clone current dist_frame).
        let total_pixels = (w as usize) * (h as usize);
        let stored_len = total_pixels.min(dist_frame.len());
        self.prev_frame = Some(dist_frame[..stored_len].to_vec());

        let mut completed_avg: Option<f32> = None;

        if scene_cut {
            // Attempt to close the current scene.
            completed_avg = self.close_current_scene();
            // Start a fresh scene with the current frame's PSNR.
            self.current_scene_psnr.push(psnr);
        } else {
            self.current_scene_psnr.push(psnr);
        }

        completed_avg
    }

    /// Flush the in-progress scene and return its average PSNR if it meets the
    /// minimum frame count.  Call this when the video stream ends to ensure the
    /// final scene is not dropped.
    pub fn flush(&mut self) -> Option<f32> {
        self.close_current_scene()
    }

    /// Mean PSNR over all *completed* (and accepted) scenes, or `None` if no
    /// scene has been completed yet.
    #[must_use]
    pub fn overall_average(&self) -> Option<f32> {
        if self.scene_averages.is_empty() {
            return None;
        }
        let sum: f32 = self.scene_averages.iter().sum();
        Some(sum / self.scene_averages.len() as f32)
    }

    /// Number of completed, accepted scenes.
    #[must_use]
    pub fn scene_count(&self) -> usize {
        self.scene_averages.len()
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Close the current scene: compute average, store if long enough, clear
    /// the buffer.  Returns the average if the scene was accepted.
    fn close_current_scene(&mut self) -> Option<f32> {
        if self.current_scene_psnr.len() < self.config.min_scene_frames {
            self.current_scene_psnr.clear();
            return None;
        }
        let n = self.current_scene_psnr.len() as f32;
        let avg = self.current_scene_psnr.iter().sum::<f32>() / n;
        self.scene_averages.push(avg);
        self.current_scene_psnr.clear();
        Some(avg)
    }
}

/// Mean absolute difference between two byte slices, normalised to `[0.0, 1.0]`.
///
/// Compares only the overlapping prefix (shortest slice length).  Returns `0.0`
/// for empty slices.
fn mean_abs_diff(a: &[u8], b: &[u8]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let sum: u64 = a[..n]
        .iter()
        .zip(b[..n].iter())
        .map(|(&x, &y)| u64::from(x.abs_diff(y)))
        .sum();
    (sum as f32) / (n as f32 * 255.0)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_frame(w: u32, h: u32, v: u8) -> Vec<u8> {
        vec![v; (w * h) as usize]
    }

    // ── Task 2 tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_temporal_psnr_identical() {
        let mut acc = TemporalPsnrAccumulator::new(TemporalPsnrConfig::default());
        let frame = flat_frame(64, 64, 128);
        for _ in 0..5 {
            acc.push_frame(&frame, &frame, 64, 64);
        }
        let avg = acc.current_avg().expect("should have value");
        // Identical frames → clamped to PSNR_MAX_DB; average is finite.
        assert!(
            avg >= 99.0,
            "identical frames PSNR avg must be ≥ 99 dB, got {avg}"
        );
    }

    #[test]
    fn test_temporal_psnr_window() {
        let cfg = TemporalPsnrConfig {
            window_frames: 3,
            channels: 1,
        };
        let mut acc = TemporalPsnrAccumulator::new(cfg);
        let ref_frame = flat_frame(32, 32, 128);

        // Push 10 frames with increasing distortion.
        for i in 0u8..10 {
            let dist = flat_frame(32, 32, 128_u8.saturating_add(i * 5));
            acc.push_frame(&ref_frame, &dist, 32, 32);
        }

        // Window should contain only the last 3 frames.
        let window = acc.window_values();
        assert_eq!(
            window.len(),
            3,
            "window must hold exactly 3 frames, got {}",
            window.len()
        );

        // Current avg must be derived from only those 3.
        let manual_avg = window.iter().sum::<f32>() / 3.0;
        let reported = acc.current_avg().expect("should have value");
        assert!(
            (reported - manual_avg).abs() < 1e-3,
            "current_avg mismatch: reported {reported}, manual {manual_avg}"
        );
    }

    #[test]
    fn test_temporal_psnr_trends() {
        let cfg = TemporalPsnrConfig {
            window_frames: 1, // window=1 so current_avg == last frame's PSNR
            channels: 1,
        };
        let mut acc = TemporalPsnrAccumulator::new(cfg);
        let ref_frame = flat_frame(64, 64, 128);

        let mut prev_avg = f32::INFINITY;

        // Gradually increase distortion: diff 0, 5, 10, 15, 20.
        for step in 0u8..5 {
            let dist = flat_frame(64, 64, 128_u8.saturating_add(step * 5));
            acc.push_frame(&ref_frame, &dist, 64, 64);
            if let Some(avg) = acc.current_avg() {
                assert!(
                    avg <= prev_avg + 0.001,
                    "PSNR should be monotonically non-increasing as distortion grows. \
                     step={step}, prev={prev_avg}, current={avg}"
                );
                prev_avg = avg;
            }
        }
    }

    #[test]
    fn test_temporal_psnr_all_frames_avg() {
        let mut acc = TemporalPsnrAccumulator::new(TemporalPsnrConfig {
            window_frames: 2,
            channels: 1,
        });
        assert!(acc.all_frames_avg().is_none());

        let r = flat_frame(32, 32, 100);
        let d = flat_frame(32, 32, 110);
        acc.push_frame(&r, &d, 32, 32);
        acc.push_frame(&r, &d, 32, 32);
        acc.push_frame(&r, &d, 32, 32); // third exceeds window but is counted in all

        let all_avg = acc
            .all_frames_avg()
            .expect("should have value after pushes");
        // All three frames are identical pairs → same PSNR each.
        let win_avg = acc.current_avg().expect("window should have value");
        assert!(
            (all_avg - win_avg).abs() < 0.5,
            "when all frames are the same distortion, all_avg ≈ current_avg"
        );
    }

    #[test]
    fn test_temporal_psnr_no_frames_returns_none() {
        let acc = TemporalPsnrAccumulator::new(TemporalPsnrConfig::default());
        assert!(
            acc.current_avg().is_none(),
            "empty accumulator → current_avg is None"
        );
        assert!(
            acc.all_frames_avg().is_none(),
            "empty accumulator → all_frames_avg is None"
        );
    }

    #[test]
    fn test_temporal_psnr_yuv_channels() {
        let cfg = TemporalPsnrConfig {
            window_frames: 3,
            channels: 3,
        };
        let mut acc = TemporalPsnrAccumulator::new(cfg);

        // Interleaved YUV frame (32×32 = 1024 pixels → 3072 bytes).
        let ref_frame = vec![128u8; 32 * 32 * 3];
        let dist_frame = vec![138u8; 32 * 32 * 3];
        acc.push_frame(&ref_frame, &dist_frame, 32, 32);

        let avg = acc.current_avg().expect("should have value");
        assert!(
            avg > 20.0 && avg < 100.0,
            "YUV PSNR should be reasonable, got {avg}"
        );
    }

    // ── SceneAwareQualityAccumulator tests ────────────────────────────────────

    use super::{SceneAwareQualityAccumulator, SceneAwareQualityConfig};

    const W: u32 = 16;
    const H: u32 = 16;
    const PIXELS: usize = (W * H) as usize;

    /// Push N identical frame pairs (ref == dist at luma `val`).
    fn push_n_similar(acc: &mut SceneAwareQualityAccumulator, n: usize, val: u8) {
        let ref_f = flat_frame(W, H, val);
        let dist_f = flat_frame(W, H, val);
        for _ in 0..n {
            acc.push_frame(&ref_f, &dist_f, W, H);
        }
    }

    // Test 1: 20 similar frames → no scene cuts detected at all.
    #[test]
    fn test_scene_aware_no_cuts() {
        let cfg = SceneAwareQualityConfig {
            scene_cut_threshold: 0.15,
            min_scene_frames: 5,
        };
        let mut acc = SceneAwareQualityAccumulator::new(cfg);

        // Push 20 frames, all with ref=128, dist=128 → no MAD at all.
        push_n_similar(&mut acc, 20, 128);

        assert_eq!(
            acc.scene_count(),
            0,
            "no scene boundaries expected for 20 identical frames"
        );
    }

    // Test 2: abrupt black-to-white transition → 1 boundary detected → scene_count=1
    //         after the boundary is triggered + the first scene is long enough.
    #[test]
    fn test_scene_aware_cut_detected() {
        let cfg = SceneAwareQualityConfig {
            scene_cut_threshold: 0.10, // fairly sensitive
            min_scene_frames: 5,
        };
        let mut acc = SceneAwareQualityAccumulator::new(cfg);

        // Scene 1: 6 black frames (ref=black, dist=black).
        let black = flat_frame(W, H, 0);
        for _ in 0..6 {
            acc.push_frame(&black, &black, W, H);
        }

        // Scene 2: first white frame triggers the cut.
        let white = flat_frame(W, H, 255);
        let result = acc.push_frame(&white, &white, W, H);

        // The cut should have been detected and the first scene (6 frames ≥ min=5)
        // should be accepted.
        assert!(
            result.is_some(),
            "push_frame should return Some(avg) when a scene boundary is detected"
        );
        assert_eq!(
            acc.scene_count(),
            1,
            "exactly one scene should be completed"
        );
    }

    // Test 3: scene shorter than min_scene_frames → not counted.
    #[test]
    fn test_scene_aware_short_scene_skipped() {
        let cfg = SceneAwareQualityConfig {
            scene_cut_threshold: 0.10,
            min_scene_frames: 5,
        };
        let mut acc = SceneAwareQualityAccumulator::new(cfg);

        // Scene 1: only 3 black frames (below min_scene_frames=5).
        let black = flat_frame(W, H, 0);
        for _ in 0..3 {
            acc.push_frame(&black, &black, W, H);
        }

        // Abrupt cut to white.
        let white = flat_frame(W, H, 255);
        let result = acc.push_frame(&white, &white, W, H);

        // Cut detected but scene was too short → not counted.
        assert!(
            result.is_none(),
            "short scene (3 < 5 frames) must be discarded"
        );
        assert_eq!(acc.scene_count(), 0, "short scene must not be counted");
    }

    // Test 4: two scenes each 10 frames → overall_average is finite and reasonable.
    #[test]
    fn test_scene_aware_overall_average() {
        let cfg = SceneAwareQualityConfig {
            scene_cut_threshold: 0.10,
            min_scene_frames: 5,
        };
        let mut acc = SceneAwareQualityAccumulator::new(cfg);

        // Scene 1: 10 black/black frames → perfect PSNR.
        let black = flat_frame(W, H, 0);
        for _ in 0..10 {
            acc.push_frame(&black, &black, W, H);
        }

        // Trigger cut to white, which closes scene 1.
        let white = flat_frame(W, H, 255);
        acc.push_frame(&white, &white, W, H); // cut + first frame of scene 2

        // Push 9 more white/white frames → scene 2 has 10 frames total.
        for _ in 0..9 {
            acc.push_frame(&white, &white, W, H);
        }

        // Flush the final scene.
        acc.flush();

        assert_eq!(acc.scene_count(), 2, "expected 2 completed scenes");
        let avg = acc.overall_average().expect("overall_average must be Some");
        assert!(
            avg.is_finite() && avg > 0.0,
            "overall_average must be a positive finite number, got {avg}"
        );
    }

    // Test 5: mean_abs_diff helper — identical slices → 0.0.
    #[test]
    fn test_mean_abs_diff_identical() {
        let a = vec![100u8; PIXELS];
        let diff = super::mean_abs_diff(&a, &a);
        assert_eq!(diff, 0.0);
    }

    // Test 6: mean_abs_diff helper — max difference (0 vs 255) → 1.0.
    #[test]
    fn test_mean_abs_diff_max() {
        let a = vec![0u8; PIXELS];
        let b = vec![255u8; PIXELS];
        let diff = super::mean_abs_diff(&a, &b);
        assert!(
            (diff - 1.0).abs() < 1e-4,
            "max MAD should be 1.0, got {diff}"
        );
    }
}
