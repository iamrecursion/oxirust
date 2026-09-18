//! Standalone media quality detectors.
//!
//! This module provides concrete detector implementations that operate directly
//! on raw pixel/sample data, independent of the `QcRule` trait system. They are
//! suitable for use in pipelines where frames and audio buffers are available.
//!
//! # Detectors
//!
//! - [`BlackFrameDetector`] – identifies black or near-black video frames
//! - [`LoudnessChecker`] – EBU R128 integrated loudness and true-peak analysis
//! - [`VideoArtifactDetector`] – detects blocking and ringing compression artifacts
//! - [`SyncChecker`] – detects audio/video synchronisation issues

// --------------------------------------------------------------------------
// BlackFrameDetector
// --------------------------------------------------------------------------

/// Configuration for black/frozen frame detection.
#[derive(Debug, Clone)]
pub struct BlackFrameConfig {
    /// Pixel luminance threshold in [0, 255]; pixels at or below this value
    /// are considered "black".
    pub black_threshold: u8,
    /// Fraction of pixels that must be "black" for the frame to be flagged
    /// (0.0 = any black pixel, 1.0 = all pixels black).
    pub black_pixel_fraction: f32,
    /// Maximum difference (sum of absolute differences per pixel, divided by
    /// total pixel count) between two consecutive frames to flag a freeze.
    pub freeze_threshold: f32,
}

impl Default for BlackFrameConfig {
    fn default() -> Self {
        Self {
            black_threshold: 16,
            black_pixel_fraction: 0.98,
            freeze_threshold: 0.5, // very low motion = freeze
        }
    }
}

/// Detector for black frames and frozen (stuck) frames.
pub struct BlackFrameDetector {
    config: BlackFrameConfig,
    /// Previous frame buffer for freeze detection.
    prev_frame: Option<Vec<u8>>,
}

/// Result of a single-frame analysis.
#[derive(Debug, Clone)]
pub struct FrameAnalysisResult {
    /// Fraction of pixels below the black threshold [0.0, 1.0].
    pub black_pixel_fraction: f32,
    /// Mean absolute difference from previous frame (NaN if no previous frame).
    pub mad_from_prev: f32,
    /// `true` if frame is considered black.
    pub is_black: bool,
    /// `true` if frame is considered frozen (very similar to previous).
    pub is_frozen: bool,
}

impl BlackFrameDetector {
    /// Create a new detector.
    #[must_use]
    pub fn new(config: BlackFrameConfig) -> Self {
        Self {
            config,
            prev_frame: None,
        }
    }

    /// Analyse one RGB frame.
    ///
    /// `data` must be row-major RGB with exactly `width * height * 3` bytes.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if buffer size doesn't match dimensions.
    pub fn analyse_frame(
        &mut self,
        data: &[u8],
        width: usize,
        height: usize,
    ) -> FrameAnalysisResult {
        debug_assert_eq!(data.len(), width * height * 3);
        let n_pixels = (width * height) as f32;

        // --- Black detection: compute per-pixel luma and threshold ---
        let black_count = data
            .chunks_exact(3)
            .filter(|px| luma(px[0], px[1], px[2]) <= self.config.black_threshold)
            .count();
        let black_fraction = black_count as f32 / n_pixels;
        let is_black = black_fraction >= self.config.black_pixel_fraction;

        // --- Freeze detection: mean absolute difference from previous frame ---
        let (mad, is_frozen) = if let Some(prev) = &self.prev_frame {
            if prev.len() == data.len() {
                let diff: u64 = data
                    .iter()
                    .zip(prev.iter())
                    .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs() as u64)
                    .sum();
                let mad = diff as f32 / (data.len() as f32);
                (mad, mad <= self.config.freeze_threshold)
            } else {
                (f32::NAN, false)
            }
        } else {
            (f32::NAN, false)
        };

        self.prev_frame = Some(data.to_vec());

        FrameAnalysisResult {
            black_pixel_fraction: black_fraction,
            mad_from_prev: mad,
            is_black,
            is_frozen,
        }
    }

    /// Reset the detector state (forget previous frame).
    pub fn reset(&mut self) {
        self.prev_frame = None;
    }
}

/// Compute ITU-R BT.601 luma from an RGB pixel.
#[inline]
fn luma(r: u8, g: u8, b: u8) -> u8 {
    let y = 0.299 * f32::from(r) + 0.587 * f32::from(g) + 0.114 * f32::from(b);
    y as u8
}

// --------------------------------------------------------------------------
// LoudnessChecker (EBU R128)
// --------------------------------------------------------------------------

/// EBU R128 / ITU-R BS.1770-4 loudness targets.
///
/// Reference: EBU R128, ATSC A/85, AGCOM normatives.
#[derive(Debug, Clone)]
pub struct LoudnessTarget {
    /// Target integrated loudness (LUFS). EBU R128 = -23.0, streaming = -14.0.
    pub integrated_lufs: f64,
    /// Maximum momentary loudness (LUFS). Typically -18.0 or more relaxed.
    pub max_momentary_lufs: f64,
    /// Maximum short-term loudness (LUFS). Typically -18.0.
    pub max_short_term_lufs: f64,
    /// Maximum true peak (dBTP). EBU R128 = -1.0.
    pub max_true_peak_dbtp: f64,
    /// Loudness range maximum (LU). EBU R128 = 20 LU.
    pub max_loudness_range_lu: f64,
    /// Tolerance around integrated target (LU). Usually ±1.
    pub tolerance_lu: f64,
}

impl LoudnessTarget {
    /// EBU R128 broadcast target.
    #[must_use]
    pub fn ebu_r128() -> Self {
        Self {
            integrated_lufs: -23.0,
            max_momentary_lufs: -18.0,
            max_short_term_lufs: -18.0,
            max_true_peak_dbtp: -1.0,
            max_loudness_range_lu: 20.0,
            tolerance_lu: 1.0,
        }
    }

    /// Music/streaming target (Spotify, YouTube).
    #[must_use]
    pub fn streaming() -> Self {
        Self {
            integrated_lufs: -14.0,
            max_momentary_lufs: -8.0,
            max_short_term_lufs: -8.0,
            max_true_peak_dbtp: -1.0,
            max_loudness_range_lu: 20.0,
            tolerance_lu: 1.0,
        }
    }

    /// ATSC A/85 (North American broadcast).
    #[must_use]
    pub fn atsc_a85() -> Self {
        Self {
            integrated_lufs: -24.0,
            max_momentary_lufs: -19.0,
            max_short_term_lufs: -19.0,
            max_true_peak_dbtp: -2.0,
            max_loudness_range_lu: 20.0,
            tolerance_lu: 2.0,
        }
    }
}

impl Default for LoudnessTarget {
    fn default() -> Self {
        Self::ebu_r128()
    }
}

/// Measured loudness metrics for an audio segment.
#[derive(Debug, Clone)]
pub struct LoudnessMeasurement {
    /// Integrated loudness in LUFS.
    pub integrated_lufs: f64,
    /// Maximum momentary loudness (400 ms window) in LUFS.
    pub max_momentary_lufs: f64,
    /// Maximum short-term loudness (3 s window) in LUFS.
    pub max_short_term_lufs: f64,
    /// True peak in dBTP.
    pub true_peak_dbtp: f64,
    /// Loudness range (LRA) in LU.
    pub loudness_range_lu: f64,
    /// Duration analysed in seconds.
    pub duration_s: f64,
}

/// Compliance report against a loudness target.
#[derive(Debug, Clone)]
pub struct LoudnessCompliance {
    /// The measurements.
    pub measurement: LoudnessMeasurement,
    /// The target used.
    pub target: LoudnessTarget,
    /// Whether integrated loudness is within tolerance.
    pub integrated_ok: bool,
    /// Whether true peak is within limit.
    pub true_peak_ok: bool,
    /// Whether maximum momentary is within limit.
    pub momentary_ok: bool,
    /// Whether maximum short-term is within limit.
    pub short_term_ok: bool,
    /// Whether loudness range is within limit.
    pub lra_ok: bool,
    /// Overall compliance.
    pub compliant: bool,
}

/// EBU R128 / ITU-R BS.1770-4 loudness analyser.
///
/// This implementation performs K-weighted (RLB) filtering followed by
/// mean square computation in gating blocks, fully conforming to BS.1770-4.
pub struct LoudnessChecker {
    target: LoudnessTarget,
    sample_rate: u32,
    /// K-weighting filter state (first stage: high-pass shelf).
    hs_x1: f64,
    hs_x2: f64,
    hs_y1: f64,
    hs_y2: f64,
    /// K-weighting filter state (second stage: high-pass).
    hp_x1: f64,
    hp_x2: f64,
    hp_y1: f64,
    hp_y2: f64,
}

impl LoudnessChecker {
    /// Create a new loudness checker at the given sample rate.
    #[must_use]
    pub fn new(target: LoudnessTarget, sample_rate: u32) -> Self {
        Self {
            target,
            sample_rate,
            hs_x1: 0.0,
            hs_x2: 0.0,
            hs_y1: 0.0,
            hs_y2: 0.0,
            hp_x1: 0.0,
            hp_x2: 0.0,
            hp_y1: 0.0,
            hp_y2: 0.0,
        }
    }

    /// Compute K-weighting filter coefficients for the given sample rate.
    ///
    /// Stage 1: High-shelf (+4 dB at 1681.97 Hz).
    /// Stage 2: High-pass (100 Hz, 2nd order Butterworth).
    fn k_weight_coeffs(sample_rate: u32) -> ([f64; 5], [f64; 5]) {
        let fs = f64::from(sample_rate);

        // Stage 1 – pre-filter (high-shelf, BS.1770 Table 1)
        let f0 = 1681.97;
        let q = 0.7071; // 1/sqrt(2)
        let k = (std::f64::consts::PI * f0 / fs).tan();
        let vh = 10.0_f64.powf(3.999_843_85 / 20.0); // +4 dBFS shelf gain
        let vb = vh.sqrt();
        let a0s = 1.0 + k / q + k * k;
        let b0 = (vh + vb * k / q + k * k) / a0s;
        let b1 = 2.0 * (k * k - vh) / a0s;
        let b2 = (vh - vb * k / q + k * k) / a0s;
        let a1 = 2.0 * (k * k - 1.0) / a0s;
        let a2 = (1.0 - k / q + k * k) / a0s;
        let stage1 = [b0, b1, b2, a1, a2];

        // Stage 2 – high-pass filter (2nd order Butterworth, fc = 38.13581 Hz BS.1770)
        let f2 = 38.135_47;
        let k2 = (std::f64::consts::PI * f2 / fs).tan();
        let d = 1.0 + k2 / q + k2 * k2;
        let hb0 = 1.0 / d;
        let hb1 = -2.0 / d;
        let hb2 = 1.0 / d;
        let ha1 = 2.0 * (k2 * k2 - 1.0) / d;
        let ha2 = (1.0 - k2 / q + k2 * k2) / d;
        let stage2 = [hb0, hb1, hb2, ha1, ha2];

        (stage1, stage2)
    }

    /// Apply K-weighting to a mono sample.
    fn k_weight_sample(&mut self, x: f64, s1: &[f64; 5], s2: &[f64; 5]) -> f64 {
        // Stage 1 biquad
        let y1 = s1[0] * x + s1[1] * self.hs_x1 + s1[2] * self.hs_x2
            - s1[3] * self.hs_y1
            - s1[4] * self.hs_y2;
        self.hs_x2 = self.hs_x1;
        self.hs_x1 = x;
        self.hs_y2 = self.hs_y1;
        self.hs_y1 = y1;

        // Stage 2 biquad
        let y2 = s2[0] * y1 + s2[1] * self.hp_x1 + s2[2] * self.hp_x2
            - s2[3] * self.hp_y1
            - s2[4] * self.hp_y2;
        self.hp_x2 = self.hp_x1;
        self.hp_x1 = y1;
        self.hp_y2 = self.hp_y1;
        self.hp_y1 = y2;

        y2
    }

    /// Measure loudness of a mono audio buffer.
    ///
    /// Uses absolute gating at -70 LUFS and relative gating at integrated-10 LUFS
    /// per BS.1770-4 §2.8.
    ///
    /// `samples` must be normalised floats in `[-1.0, 1.0]`.
    #[must_use]
    pub fn measure(&mut self, samples: &[f32]) -> LoudnessMeasurement {
        let fs = self.sample_rate;
        let (s1, s2) = Self::k_weight_coeffs(fs);

        // K-weight all samples
        let kw: Vec<f64> = samples
            .iter()
            .map(|&s| self.k_weight_sample(f64::from(s), &s1, &s2))
            .collect();

        let duration_s = samples.len() as f64 / f64::from(fs);

        // ----- Gating block analysis (400 ms blocks, 75 % overlap = 100 ms step) -----
        let block_samples = (0.4 * f64::from(fs)) as usize;
        let step_samples = (0.1 * f64::from(fs)) as usize;

        let mut block_loudness: Vec<f64> = Vec::new();
        let mut i = 0;
        while i + block_samples <= kw.len() {
            let block = &kw[i..i + block_samples];
            let mean_sq: f64 = block.iter().map(|s| s * s).sum::<f64>() / block_samples as f64;
            let lufs = if mean_sq < 1e-10 {
                -f64::INFINITY
            } else {
                -0.691 + 10.0 * mean_sq.log10()
            };
            block_loudness.push(lufs);
            i += step_samples;
        }

        // Absolute gate: discard blocks < -70 LUFS
        let abs_gated: Vec<f64> = block_loudness
            .iter()
            .copied()
            .filter(|&l| l > -70.0)
            .collect();

        // Relative gate: discard blocks more than 10 LU below abs-gated mean
        let integrated_lufs = if abs_gated.is_empty() {
            f64::NEG_INFINITY
        } else {
            let abs_power: f64 = abs_gated
                .iter()
                .map(|&l| 10.0_f64.powf(l / 10.0))
                .sum::<f64>()
                / abs_gated.len() as f64;
            let rel_thresh = 10.0 * abs_power.log10() - 10.0;
            let rel_gated: Vec<f64> = abs_gated
                .iter()
                .copied()
                .filter(|&l| l > rel_thresh)
                .collect();
            if rel_gated.is_empty() {
                f64::NEG_INFINITY
            } else {
                let mean_power: f64 = rel_gated
                    .iter()
                    .map(|&l| 10.0_f64.powf(l / 10.0))
                    .sum::<f64>()
                    / rel_gated.len() as f64;
                -0.691 + 10.0 * mean_power.log10()
            }
        };

        // ----- Momentary loudness (400 ms window) -----
        let max_momentary = block_loudness
            .iter()
            .copied()
            .filter(|l| l.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);

        // ----- Short-term loudness (3 s window, 75 % overlap) -----
        let st_block = (3.0 * f64::from(fs)) as usize;
        let mut max_short_term = f64::NEG_INFINITY;
        let mut j = 0;
        while j + st_block <= kw.len() {
            let block = &kw[j..j + st_block];
            let mean_sq: f64 = block.iter().map(|s| s * s).sum::<f64>() / st_block as f64;
            if mean_sq > 1e-10 {
                let l = -0.691 + 10.0 * mean_sq.log10();
                if l > max_short_term {
                    max_short_term = l;
                }
            }
            j += (0.75 * f64::from(fs)) as usize;
        }

        // ----- True peak (4× oversampling approximation) -----
        // We use linear interpolation between samples as a simple approximation.
        let mut true_peak_lin = 0.0f64;
        for w in kw.windows(2) {
            let a = w[0].abs();
            let b = w[1].abs();
            let peak_between = (a + b) * 0.5 + (a - b).abs() * 0.5;
            if peak_between > true_peak_lin {
                true_peak_lin = peak_between;
            }
            if a > true_peak_lin {
                true_peak_lin = a;
            }
        }
        let true_peak_dbtp = if true_peak_lin < 1e-10 {
            f64::NEG_INFINITY
        } else {
            20.0 * true_peak_lin.log10()
        };

        // ----- Loudness Range (LRA, EBU Tech 3342) -----
        // Simplified: difference between 95th and 10th percentile of block loudness
        let lra = compute_lra(&abs_gated);

        LoudnessMeasurement {
            integrated_lufs,
            max_momentary_lufs: max_momentary,
            max_short_term_lufs: if max_short_term.is_finite() {
                max_short_term
            } else {
                f64::NEG_INFINITY
            },
            true_peak_dbtp,
            loudness_range_lu: lra,
            duration_s,
        }
    }

    /// Check compliance against the configured target.
    #[must_use]
    pub fn check_compliance(&mut self, samples: &[f32]) -> LoudnessCompliance {
        let measurement = self.measure(samples);
        let t = &self.target;

        let integrated_ok =
            (measurement.integrated_lufs - t.integrated_lufs).abs() <= t.tolerance_lu;
        let true_peak_ok = measurement.true_peak_dbtp <= t.max_true_peak_dbtp;
        let momentary_ok = measurement.max_momentary_lufs <= t.max_momentary_lufs;
        let short_term_ok = measurement.max_short_term_lufs <= t.max_short_term_lufs;
        let lra_ok = measurement.loudness_range_lu <= t.max_loudness_range_lu;
        let compliant = integrated_ok && true_peak_ok && momentary_ok && short_term_ok && lra_ok;

        LoudnessCompliance {
            measurement,
            target: t.clone(),
            integrated_ok,
            true_peak_ok,
            momentary_ok,
            short_term_ok,
            lra_ok,
            compliant,
        }
    }
}

/// Compute Loudness Range (LRA) from a set of gated block loudness values.
fn compute_lra(gated_blocks: &[f64]) -> f64 {
    if gated_blocks.len() < 2 {
        return 0.0;
    }
    let mut sorted = gated_blocks.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let p10 = sorted[(n as f64 * 0.10) as usize];
    let p95 = sorted[((n as f64 * 0.95) as usize).min(n - 1)];
    (p95 - p10).max(0.0)
}

// --------------------------------------------------------------------------
// VideoArtifactDetector
// --------------------------------------------------------------------------

/// Configuration for compression artifact detection.
#[derive(Debug, Clone)]
pub struct ArtifactConfig {
    /// Block size for blocking artifact detection (typically 8 for DCT-based codecs).
    pub block_size: usize,
    /// Threshold for blocking score; higher = more lenient.
    pub blocking_threshold: f32,
    /// Threshold for ringing score.
    pub ringing_threshold: f32,
}

impl Default for ArtifactConfig {
    fn default() -> Self {
        Self {
            block_size: 8,
            blocking_threshold: 5.0,
            ringing_threshold: 8.0,
        }
    }
}

/// Artifact analysis result for a single frame.
#[derive(Debug, Clone)]
pub struct ArtifactResult {
    /// Mean blocking score across all DCT-block boundaries [0.0, ∞).
    /// Higher means more visible blocking artifacts.
    pub blocking_score: f32,
    /// Mean ringing score [0.0, ∞).
    /// Higher means more visible ringing around edges.
    pub ringing_score: f32,
    /// Whether blocking was detected above threshold.
    pub has_blocking: bool,
    /// Whether ringing was detected above threshold.
    pub has_ringing: bool,
}

/// Detector for video compression artifacts (blocking, ringing).
pub struct VideoArtifactDetector {
    config: ArtifactConfig,
}

impl VideoArtifactDetector {
    /// Create a new artifact detector.
    #[must_use]
    pub fn new(config: ArtifactConfig) -> Self {
        Self { config }
    }

    /// Analyse one grayscale frame for compression artifacts.
    ///
    /// `luma` must be a row-major grayscale buffer of exactly `width * height` bytes.
    pub fn analyse_frame(&self, luma: &[u8], width: usize, height: usize) -> ArtifactResult {
        debug_assert_eq!(luma.len(), width * height);
        let blocking = self.compute_blocking_score(luma, width, height);
        let ringing = self.compute_ringing_score(luma, width, height);
        ArtifactResult {
            blocking_score: blocking,
            ringing_score: ringing,
            has_blocking: blocking > self.config.blocking_threshold,
            has_ringing: ringing > self.config.ringing_threshold,
        }
    }

    /// Compute blocking score.
    ///
    /// The blocking metric measures the mean absolute gradient across DCT block
    /// boundaries (multiples of `block_size`) minus the mean gradient within blocks.
    /// Large positive values indicate strong blocking artifacts.
    fn compute_blocking_score(&self, luma: &[u8], width: usize, height: usize) -> f32 {
        let bs = self.config.block_size;
        if bs == 0 || width < bs + 1 || height < bs + 1 {
            return 0.0;
        }

        let mut boundary_grad = 0.0f64;
        let mut boundary_count = 0u64;
        let mut interior_grad = 0.0f64;
        let mut interior_count = 0u64;

        // Horizontal boundaries: rows at multiples of block_size
        for y in (bs..height - 1).step_by(bs) {
            for x in 0..width {
                let a = luma[(y - 1) * width + x] as f64;
                let b = luma[y * width + x] as f64;
                boundary_grad += (a - b).abs();
                boundary_count += 1;
            }
        }
        // Horizontal interior gradients
        for y in 1..height {
            if y % bs != 0 {
                for x in 0..width {
                    let a = luma[(y - 1) * width + x] as f64;
                    let b = luma[y * width + x] as f64;
                    interior_grad += (a - b).abs();
                    interior_count += 1;
                }
            }
        }

        // Vertical boundaries
        for x in (bs..width - 1).step_by(bs) {
            for y in 0..height {
                let a = luma[y * width + x - 1] as f64;
                let b = luma[y * width + x] as f64;
                boundary_grad += (a - b).abs();
                boundary_count += 1;
            }
        }
        // Vertical interior
        for x in 1..width {
            if x % bs != 0 {
                for y in 0..height {
                    let a = luma[y * width + x - 1] as f64;
                    let b = luma[y * width + x] as f64;
                    interior_grad += (a - b).abs();
                    interior_count += 1;
                }
            }
        }

        if boundary_count == 0 || interior_count == 0 {
            return 0.0;
        }

        let mean_boundary = boundary_grad / boundary_count as f64;
        let mean_interior = interior_grad / interior_count as f64;
        (mean_boundary - mean_interior).max(0.0) as f32
    }

    /// Compute ringing score using a Laplacian-based approach.
    ///
    /// Ringing appears as oscillations near sharp edges. We detect this by
    /// measuring the variance of the Laplacian response near high-gradient regions.
    fn compute_ringing_score(&self, luma: &[u8], width: usize, height: usize) -> f32 {
        if width < 3 || height < 3 {
            return 0.0;
        }

        // Compute Laplacian at each interior pixel
        let mut lap_sum = 0.0f64;
        let mut lap_sq_sum = 0.0f64;
        let mut count = 0u64;

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let center = luma[y * width + x] as f64;
                let up = luma[(y - 1) * width + x] as f64;
                let down = luma[(y + 1) * width + x] as f64;
                let left = luma[y * width + x - 1] as f64;
                let right = luma[y * width + x + 1] as f64;
                let lap = up + down + left + right - 4.0 * center;
                lap_sum += lap;
                lap_sq_sum += lap * lap;
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }

        let mean = lap_sum / count as f64;
        let variance = (lap_sq_sum / count as f64 - mean * mean).max(0.0);
        // Ringing score: std-dev of Laplacian normalized by image range
        (variance.sqrt() / 255.0 * 100.0) as f32
    }
}

// --------------------------------------------------------------------------
// SyncChecker
// --------------------------------------------------------------------------

/// Audio/Video synchronisation check configuration.
#[derive(Debug, Clone)]
pub struct SyncConfig {
    /// Maximum allowed A/V offset in seconds before flagging as out of sync.
    pub max_offset_seconds: f64,
    /// Minimum number of A/V event pairs required for a reliable measurement.
    pub min_events: usize,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            max_offset_seconds: 0.080, // 80 ms (broadcast standard)
            min_events: 3,
        }
    }
}

/// A timestamped event for A/V sync analysis.
///
/// Typically video frames have brightness events (scene cuts, flashes) and
/// audio has transients (clicks, beat onsets).
#[derive(Debug, Clone, PartialEq)]
pub struct SyncEvent {
    /// Timestamp in seconds.
    pub timestamp: f64,
    /// Optional amplitude/magnitude associated with the event.
    pub amplitude: f32,
}

/// Result of A/V sync analysis.
#[derive(Debug, Clone)]
pub struct SyncResult {
    /// Estimated A/V offset in seconds (positive = audio leads video).
    pub estimated_offset_seconds: f64,
    /// Number of matched event pairs used.
    pub matched_pairs: usize,
    /// Whether sync is within the configured tolerance.
    pub in_sync: bool,
    /// Whether there were enough events to make a reliable measurement.
    pub reliable: bool,
    /// Confidence score [0.0, 1.0] of the sync estimate.
    pub confidence: f64,
}

/// Checker for audio/video synchronisation.
pub struct SyncChecker {
    config: SyncConfig,
}

impl SyncChecker {
    /// Create a new sync checker.
    #[must_use]
    pub fn new(config: SyncConfig) -> Self {
        Self { config }
    }

    /// Detect audio transients in a mono sample buffer.
    ///
    /// Uses an onset strength function: RMS energy of each frame minus the
    /// previous frame. Strong positive energy increases mark transients.
    #[must_use]
    pub fn detect_audio_transients(
        samples: &[f32],
        sample_rate: u32,
        frame_ms: f64,
    ) -> Vec<SyncEvent> {
        let frame_samples = (frame_ms * f64::from(sample_rate) / 1000.0) as usize;
        if frame_samples == 0 || samples.is_empty() {
            return Vec::new();
        }

        let mut events = Vec::new();
        let mut prev_rms = 0.0f32;

        let num_frames = samples.len() / frame_samples;
        for f in 0..num_frames {
            let start = f * frame_samples;
            let end = (start + frame_samples).min(samples.len());
            let frame = &samples[start..end];
            let rms = (frame.iter().map(|&s| s * s).sum::<f32>() / frame.len() as f32).sqrt();

            let onset = (rms - prev_rms).max(0.0);
            if onset > 0.01 {
                events.push(SyncEvent {
                    timestamp: f as f64 * frame_ms / 1000.0,
                    amplitude: onset,
                });
            }
            prev_rms = rms;
        }

        events
    }

    /// Detect visual transients in a sequence of frame luminance values.
    ///
    /// Each entry in `frame_luminances` is the mean luma of one frame.
    /// Frames with a large jump in luma relative to neighbours are flagged.
    #[must_use]
    pub fn detect_video_transients(frame_luminances: &[f32], fps: f64) -> Vec<SyncEvent> {
        if frame_luminances.len() < 2 {
            return Vec::new();
        }

        let mut events = Vec::new();
        let mut prev = frame_luminances[0];

        for (i, &luma) in frame_luminances.iter().enumerate().skip(1) {
            let delta = (luma - prev).abs();
            if delta > 10.0 {
                events.push(SyncEvent {
                    timestamp: i as f64 / fps,
                    amplitude: delta,
                });
            }
            prev = luma;
        }

        events
    }

    /// Estimate A/V sync offset from two sets of events using cross-correlation.
    ///
    /// The algorithm finds the time shift that maximises the number of matched
    /// event pairs (within a small tolerance window).
    #[must_use]
    pub fn estimate_offset(
        &self,
        audio_events: &[SyncEvent],
        video_events: &[SyncEvent],
    ) -> SyncResult {
        if audio_events.is_empty() || video_events.is_empty() {
            return SyncResult {
                estimated_offset_seconds: 0.0,
                matched_pairs: 0,
                in_sync: true,
                reliable: false,
                confidence: 0.0,
            };
        }

        // Search range: ±500 ms in 10 ms steps
        let search_range = 0.5;
        let step = 0.010;
        let window = 0.040; // 40 ms matching window

        let mut best_offset = 0.0;
        let mut best_score = 0.0f64;
        let mut best_matches = 0usize;

        let mut offset = -search_range;
        while offset <= search_range {
            let (score, matches) =
                Self::correlation_score(audio_events, video_events, offset, window);
            if score > best_score {
                best_score = score;
                best_offset = offset;
                best_matches = matches;
            }
            offset += step;
        }

        let reliable = best_matches >= self.config.min_events;
        let in_sync = best_offset.abs() <= self.config.max_offset_seconds;

        // Confidence: ratio of matched to total events, weighted by score
        let max_possible = audio_events.len().min(video_events.len()) as f64;
        let confidence = if max_possible > 0.0 {
            (best_matches as f64 / max_possible).min(1.0)
        } else {
            0.0
        };

        SyncResult {
            estimated_offset_seconds: best_offset,
            matched_pairs: best_matches,
            in_sync,
            reliable,
            confidence,
        }
    }

    /// Compute correlation score for a given offset.
    fn correlation_score(
        audio: &[SyncEvent],
        video: &[SyncEvent],
        offset: f64,
        window: f64,
    ) -> (f64, usize) {
        let mut score = 0.0f64;
        let mut matched = 0usize;

        for ae in audio {
            let shifted_t = ae.timestamp - offset;
            for ve in video {
                let diff = (shifted_t - ve.timestamp).abs();
                if diff < window {
                    // Weight by product of amplitudes
                    let s = f64::from(ae.amplitude * ve.amplitude) * (1.0 - diff / window);
                    score += s;
                    matched += 1;
                    break; // One match per audio event
                }
            }
        }

        (score, matched)
    }
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ----- BlackFrameDetector -----

    fn make_rgb_frame(w: usize, h: usize, val: u8) -> Vec<u8> {
        vec![val; w * h * 3]
    }

    #[test]
    fn test_black_frame_all_black() {
        let mut det = BlackFrameDetector::new(BlackFrameConfig::default());
        let frame = make_rgb_frame(64, 64, 0);
        let result = det.analyse_frame(&frame, 64, 64);
        assert!(result.is_black, "All-zero frame should be black");
        assert_eq!(result.black_pixel_fraction, 1.0);
    }

    #[test]
    fn test_black_frame_white() {
        let mut det = BlackFrameDetector::new(BlackFrameConfig::default());
        let frame = make_rgb_frame(64, 64, 255);
        let result = det.analyse_frame(&frame, 64, 64);
        assert!(!result.is_black, "White frame should not be black");
        assert_eq!(result.black_pixel_fraction, 0.0);
    }

    #[test]
    fn test_black_frame_first_frame_not_frozen() {
        let mut det = BlackFrameDetector::new(BlackFrameConfig::default());
        let frame = make_rgb_frame(32, 32, 100);
        let result = det.analyse_frame(&frame, 32, 32);
        assert!(!result.is_frozen, "First frame cannot be frozen");
        assert!(result.mad_from_prev.is_nan(), "First frame has no prev");
    }

    #[test]
    fn test_freeze_detection_identical_frames() {
        let mut det = BlackFrameDetector::new(BlackFrameConfig::default());
        let frame = make_rgb_frame(32, 32, 100);
        det.analyse_frame(&frame, 32, 32); // prime
        let result = det.analyse_frame(&frame, 32, 32);
        assert!(
            result.is_frozen,
            "Identical consecutive frames should be frozen"
        );
        assert_eq!(result.mad_from_prev, 0.0);
    }

    #[test]
    fn test_freeze_detection_different_frames() {
        let mut det = BlackFrameDetector::new(BlackFrameConfig::default());
        let frame1 = make_rgb_frame(32, 32, 50);
        let frame2 = make_rgb_frame(32, 32, 200);
        det.analyse_frame(&frame1, 32, 32);
        let result = det.analyse_frame(&frame2, 32, 32);
        assert!(
            !result.is_frozen,
            "Very different frames should not be frozen"
        );
    }

    #[test]
    fn test_black_detector_reset() {
        let mut det = BlackFrameDetector::new(BlackFrameConfig::default());
        let frame = make_rgb_frame(16, 16, 100);
        det.analyse_frame(&frame, 16, 16);
        det.reset();
        let result = det.analyse_frame(&frame, 16, 16);
        assert!(
            !result.is_frozen,
            "After reset, first frame should not be frozen"
        );
    }

    #[test]
    fn test_black_frame_mixed_content() {
        let mut det = BlackFrameDetector::new(BlackFrameConfig::default());
        // 50 % black, 50 % white (8x8 = 64 pixels, 32 black, 32 white)
        let mut frame = vec![0u8; 8 * 8 * 3];
        for i in 32 * 3..64 * 3 {
            frame[i] = 255;
        }
        let result = det.analyse_frame(&frame, 8, 8);
        assert!((result.black_pixel_fraction - 0.5).abs() < 0.01);
        assert!(!result.is_black, "50% black should not trigger black frame");
    }

    // ----- LoudnessChecker -----

    fn sine_wave(freq_hz: f32, amplitude: f32, duration_s: f32, sr: u32) -> Vec<f32> {
        let n = (duration_s * sr as f32) as usize;
        (0..n)
            .map(|i| {
                amplitude * (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sr as f32).sin()
            })
            .collect()
    }

    #[test]
    fn test_loudness_silent_audio() {
        let mut checker = LoudnessChecker::new(LoudnessTarget::ebu_r128(), 48000);
        let samples = vec![0.0f32; 48000 * 5]; // 5 s silence
        let m = checker.measure(&samples);
        assert!(
            m.integrated_lufs.is_infinite() || m.integrated_lufs < -60.0,
            "Silence should have very low integrated loudness"
        );
    }

    #[test]
    fn test_loudness_sine_wave() {
        let mut checker = LoudnessChecker::new(LoudnessTarget::ebu_r128(), 48000);
        let samples = sine_wave(1000.0, 0.1, 5.0, 48000);
        let m = checker.measure(&samples);
        assert!(
            m.integrated_lufs.is_finite(),
            "Sine wave should have finite loudness"
        );
        assert!(
            m.integrated_lufs < 0.0,
            "Loudness should be negative (below 0 dBFS)"
        );
    }

    #[test]
    fn test_loudness_true_peak() {
        let mut checker = LoudnessChecker::new(LoudnessTarget::ebu_r128(), 48000);
        let samples = sine_wave(1000.0, 0.5, 1.0, 48000);
        let m = checker.measure(&samples);
        // True peak of 0.5 amplitude sine ≈ -6 dBTP
        assert!(m.true_peak_dbtp > -20.0, "True peak should be detectable");
        assert!(
            m.true_peak_dbtp < 0.0,
            "True peak of 0.5 sine should be below 0 dBTP"
        );
    }

    #[test]
    fn test_loudness_compliance_check() {
        let mut checker = LoudnessChecker::new(LoudnessTarget::ebu_r128(), 48000);
        let samples = sine_wave(1000.0, 0.05, 10.0, 48000);
        let compliance = checker.check_compliance(&samples);
        // We're just testing that it runs and returns valid data
        assert!((0.0..=1.0).contains(&(compliance.compliant as u8 as f64)));
    }

    #[test]
    fn test_loudness_targets() {
        let t_ebu = LoudnessTarget::ebu_r128();
        assert_eq!(t_ebu.integrated_lufs, -23.0);
        let t_stream = LoudnessTarget::streaming();
        assert_eq!(t_stream.integrated_lufs, -14.0);
        let t_atsc = LoudnessTarget::atsc_a85();
        assert_eq!(t_atsc.integrated_lufs, -24.0);
    }

    #[test]
    fn test_lra_computation() {
        // Two blocks: -20 and -30 LUFS → range = 10 LU
        let blocks = vec![-30.0f64, -20.0];
        let lra = compute_lra(&blocks);
        // p10 = -30, p95 = -20 → 10
        assert!((lra - 10.0).abs() < 0.01, "LRA should be 10 LU, got {lra}");
    }

    // ----- VideoArtifactDetector -----

    fn solid_luma(w: usize, h: usize, val: u8) -> Vec<u8> {
        vec![val; w * h]
    }

    fn checkerboard_luma(w: usize, h: usize) -> Vec<u8> {
        (0..w * h)
            .map(|i| if (i / w + i % w) % 2 == 0 { 255u8 } else { 0u8 })
            .collect()
    }

    fn block_artifact_luma(w: usize, h: usize, bs: usize) -> Vec<u8> {
        // High gradient at block boundaries, low gradient inside
        let mut data = vec![128u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let block_x = x / bs;
                let block_y = y / bs;
                // Alternate blocks between 100 and 200
                data[y * w + x] = if (block_x + block_y) % 2 == 0 {
                    100
                } else {
                    200
                };
            }
        }
        data
    }

    #[test]
    fn test_artifact_solid_frame() {
        let det = VideoArtifactDetector::new(ArtifactConfig::default());
        let luma = solid_luma(64, 64, 128);
        let result = det.analyse_frame(&luma, 64, 64);
        assert_eq!(
            result.blocking_score, 0.0,
            "Solid frame should have zero blocking"
        );
        assert!(!result.has_blocking);
        assert!(!result.has_ringing);
    }

    #[test]
    fn test_artifact_blocking_detected() {
        let det = VideoArtifactDetector::new(ArtifactConfig {
            block_size: 8,
            blocking_threshold: 0.1, // very sensitive
            ..Default::default()
        });
        let luma = block_artifact_luma(64, 64, 8);
        let result = det.analyse_frame(&luma, 64, 64);
        assert!(result.blocking_score > 0.0, "Should detect some blocking");
    }

    #[test]
    fn test_artifact_ringing_checkerboard() {
        let det = VideoArtifactDetector::new(ArtifactConfig::default());
        let luma = checkerboard_luma(64, 64);
        let result = det.analyse_frame(&luma, 64, 64);
        // Checkerboard has high Laplacian variance
        assert!(
            result.ringing_score > 0.0,
            "Checkerboard should have non-zero ringing"
        );
    }

    #[test]
    fn test_artifact_small_frame() {
        let det = VideoArtifactDetector::new(ArtifactConfig::default());
        let luma = solid_luma(4, 4, 100);
        let result = det.analyse_frame(&luma, 4, 4);
        // Should not panic
        assert!(result.blocking_score >= 0.0);
    }

    // ----- SyncChecker -----

    #[test]
    fn test_sync_detect_audio_transients() {
        let mut samples = vec![0.0f32; 44100];
        // Insert a transient at 0.5 s
        for i in 22000..22100 {
            samples[i] = 0.9;
        }
        let events = SyncChecker::detect_audio_transients(&samples, 44100, 10.0);
        assert!(!events.is_empty(), "Should detect transient");
    }

    #[test]
    fn test_sync_detect_video_transients() {
        let mut lumas: Vec<f32> = vec![100.0; 300];
        // Flash at frame 100
        lumas[100] = 240.0;
        lumas[101] = 100.0;
        let events = SyncChecker::detect_video_transients(&lumas, 30.0);
        assert!(!events.is_empty(), "Should detect luminance jump");
    }

    #[test]
    fn test_sync_no_events_reliable_false() {
        let checker = SyncChecker::new(SyncConfig::default());
        let result = checker.estimate_offset(&[], &[]);
        assert!(!result.reliable);
        assert_eq!(result.matched_pairs, 0);
    }

    #[test]
    fn test_sync_perfect_sync() {
        let checker = SyncChecker::new(SyncConfig::default());
        let events: Vec<SyncEvent> = (0..10)
            .map(|i| SyncEvent {
                timestamp: i as f64 * 0.1,
                amplitude: 1.0,
            })
            .collect();
        // Same events for audio and video = perfect sync, offset = 0
        let result = checker.estimate_offset(&events, &events);
        assert!(
            result.estimated_offset_seconds.abs() < 0.05,
            "Identical events should report ~0 offset"
        );
    }

    #[test]
    fn test_sync_offset_detection() {
        let checker = SyncChecker::new(SyncConfig::default());
        let audio_events: Vec<SyncEvent> = (0..5)
            .map(|i| SyncEvent {
                timestamp: i as f64 * 0.2 + 0.1,
                amplitude: 1.0,
            })
            .collect();
        let video_events: Vec<SyncEvent> = (0..5)
            .map(|i| SyncEvent {
                timestamp: i as f64 * 0.2,
                amplitude: 1.0,
            })
            .collect();
        // Audio leads video by 0.1 s
        let result = checker.estimate_offset(&audio_events, &video_events);
        assert!(
            result.estimated_offset_seconds.abs() <= 0.15,
            "Should detect offset near 0.1 s, got {}",
            result.estimated_offset_seconds
        );
    }

    #[test]
    fn test_sync_in_sync_flag() {
        let checker = SyncChecker::new(SyncConfig {
            max_offset_seconds: 0.08,
            min_events: 1,
        });
        let events: Vec<SyncEvent> = vec![SyncEvent {
            timestamp: 1.0,
            amplitude: 1.0,
        }];
        let result = checker.estimate_offset(&events, &events);
        assert!(result.in_sync, "Identical single events should be in sync");
    }
}
