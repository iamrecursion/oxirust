//! Temporal video quality metrics.
//!
//! Provides tools for measuring flickering, motion blur, and frame-rate consistency
//! across a sequence of video frames, as well as full per-frame quality tracking
//! via [`TemporalQualityAnalyzer`].

#![allow(dead_code)]

use crate::{Frame, PsnrCalculator, SsimCalculator};
use serde::{Deserialize, Serialize};

// ─── Temporal flickering ───────────────────────────────────────────────────────

/// Frame-to-frame luminance variance (flickering) statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalFlickering {
    /// Mean inter-frame luminance change
    pub mean: f32,
    /// Standard deviation of inter-frame luminance changes
    pub std_dev: f32,
    /// Maximum single-frame luminance delta
    pub max_delta: f32,
    /// Number of frames where `|delta| > 2 * std_dev`
    pub flickering_frames: u32,
}

impl TemporalFlickering {
    /// Returns `true` when the flickering standard deviation is below 3.0.
    #[must_use]
    pub fn is_acceptable(&self) -> bool {
        self.std_dev < 3.0
    }

    /// Analyse a sequence of per-frame luma means to compute flickering stats.
    ///
    /// `luma_means` must contain at least two values; otherwise all fields are zero.
    #[must_use]
    pub fn analyze(luma_means: &[f32]) -> Self {
        if luma_means.len() < 2 {
            return Self {
                mean: 0.0,
                std_dev: 0.0,
                max_delta: 0.0,
                flickering_frames: 0,
            };
        }

        let deltas: Vec<f32> = luma_means.windows(2).map(|w| (w[1] - w[0]).abs()).collect();

        let n = deltas.len() as f32;
        let mean = deltas.iter().sum::<f32>() / n;
        let variance = deltas.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / n;
        let std_dev = variance.sqrt();

        let max_delta = deltas.iter().copied().fold(0.0f32, f32::max);

        // If std_dev is zero all deltas are identical — no flickering by definition.
        let flickering_frames = if std_dev < 1e-6 {
            0
        } else {
            let threshold = 2.0 * std_dev;
            deltas.iter().filter(|&&d| d > threshold).count() as u32
        };

        Self {
            mean,
            std_dev,
            max_delta,
            flickering_frames,
        }
    }
}

// ─── Motion blur score ────────────────────────────────────────────────────────

/// Aggregated motion-blur scores across a video.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionBlurScore {
    /// Average blur score across all frames
    pub average: f32,
    /// Index of the sharpest frame (highest Laplacian variance)
    pub sharpest_frame: usize,
    /// Index of the blurriest frame (lowest Laplacian variance)
    pub blurriest_frame: usize,
}

impl MotionBlurScore {
    /// Map the average score to a letter grade (A–F).
    ///
    /// Higher Laplacian variance → sharper → better grade.
    #[must_use]
    pub fn quality_grade(&self) -> char {
        match self.average as u32 {
            s if s >= 500 => 'A',
            s if s >= 300 => 'B',
            s if s >= 150 => 'C',
            s if s >= 50 => 'D',
            s if s >= 10 => 'E',
            _ => 'F',
        }
    }

    /// Build a `MotionBlurScore` from a list of per-frame sharpness scores.
    #[must_use]
    pub fn from_frame_scores(scores: &[f32]) -> Self {
        if scores.is_empty() {
            return Self {
                average: 0.0,
                sharpest_frame: 0,
                blurriest_frame: 0,
            };
        }

        let average = scores.iter().sum::<f32>() / scores.len() as f32;

        let sharpest_frame = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);

        let blurriest_frame = scores
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);

        Self {
            average,
            sharpest_frame,
            blurriest_frame,
        }
    }
}

/// Compute per-frame sharpness scores.
pub struct MotionBlurDetector;

impl MotionBlurDetector {
    /// Compute a sharpness score for a single frame using Laplacian variance.
    ///
    /// Higher variance = sharper image.
    /// `frame` is a slice of normalised luma samples (0.0–1.0), arranged
    /// row-major with dimensions `width × height`.
    #[must_use]
    pub fn compute_score(frame: &[f32], width: u32, height: u32) -> f32 {
        if width < 3 || height < 3 || frame.len() < (width * height) as usize {
            return 0.0;
        }

        let w = width as usize;
        let h = height as usize;

        let mut lap_values = Vec::with_capacity((w - 2) * (h - 2));

        for row in 1..h - 1 {
            for col in 1..w - 1 {
                // 5-point discrete Laplacian
                let c = frame[row * w + col];
                let n = frame[(row - 1) * w + col];
                let s = frame[(row + 1) * w + col];
                let e = frame[row * w + col + 1];
                let ww = frame[row * w + col - 1];
                let lap = (4.0 * c - n - s - e - ww).abs();
                lap_values.push(lap);
            }
        }

        if lap_values.is_empty() {
            return 0.0;
        }

        // Return mean-squared Laplacian * 1000 so typical values are in the hundreds.
        // Using mean-squared rather than variance ensures a non-zero score for any
        // frame that has non-zero gradients (e.g. a checkerboard).
        let n = lap_values.len() as f32;
        let mean_sq = lap_values.iter().map(|v| v * v).sum::<f32>() / n;
        mean_sq * 1000.0
    }
}

// ─── Frame-rate consistency ────────────────────────────────────────────────────

/// Frame-rate consistency measured from presentation timestamps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameRateConsistency {
    /// Expected (nominal) frame rate
    pub nominal_fps: f32,
    /// Measured per-frame instantaneous rates
    pub measured_fps: Vec<f32>,
    /// RMS jitter in milliseconds
    pub jitter_ms: f32,
    /// Number of frames with timing more than 50 % off the nominal interval
    pub dropped_frames: u32,
}

impl FrameRateConsistency {
    /// Analyse a sequence of presentation timestamps (in milliseconds).
    ///
    /// `timestamps_ms` must be sorted ascending with at least two entries.
    #[must_use]
    pub fn analyze(timestamps_ms: &[u64]) -> Self {
        if timestamps_ms.len() < 2 {
            return Self {
                nominal_fps: 0.0,
                measured_fps: Vec::new(),
                jitter_ms: 0.0,
                dropped_frames: 0,
            };
        }

        // Compute inter-frame intervals
        let intervals_ms: Vec<f64> = timestamps_ms
            .windows(2)
            .map(|w| (w[1] - w[0]) as f64)
            .collect();

        let n = intervals_ms.len() as f64;
        let mean_interval = intervals_ms.iter().sum::<f64>() / n;
        let nominal_fps = if mean_interval > 0.0 {
            (1000.0 / mean_interval) as f32
        } else {
            0.0
        };

        // Per-frame measured rates
        let measured_fps: Vec<f32> = intervals_ms
            .iter()
            .map(|&dt| if dt > 0.0 { (1000.0 / dt) as f32 } else { 0.0 })
            .collect();

        // RMS jitter
        let variance = intervals_ms
            .iter()
            .map(|dt| (dt - mean_interval).powi(2))
            .sum::<f64>()
            / n;
        let jitter_ms = variance.sqrt() as f32;

        // Dropped frames: interval > 1.5× nominal
        let threshold = mean_interval * 1.5;
        let dropped_frames = intervals_ms.iter().filter(|&&dt| dt > threshold).count() as u32;

        Self {
            nominal_fps,
            measured_fps,
            jitter_ms,
            dropped_frames,
        }
    }

    /// Whether the frame rate is consistent (jitter < 2 ms and no drops).
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        self.jitter_ms < 2.0 && self.dropped_frames == 0
    }
}

// ─── Temporal quality report ──────────────────────────────────────────────────

/// Combined temporal quality report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalQualityReport {
    /// Flickering analysis
    pub flickering: TemporalFlickering,
    /// Motion-blur analysis
    pub motion_blur: MotionBlurScore,
    /// Frame-rate consistency analysis
    pub frame_rate: FrameRateConsistency,
    /// Overall temporal quality score (0–100)
    pub overall_score: f32,
}

impl TemporalQualityReport {
    /// Build a report from pre-computed sub-metrics.
    #[must_use]
    pub fn new(
        flickering: TemporalFlickering,
        motion_blur: MotionBlurScore,
        frame_rate: FrameRateConsistency,
    ) -> Self {
        let overall_score = compute_overall(&flickering, &motion_blur, &frame_rate);
        Self {
            flickering,
            motion_blur,
            frame_rate,
            overall_score,
        }
    }
}

/// Compute a 0–100 overall temporal quality score.
fn compute_overall(
    flickering: &TemporalFlickering,
    motion_blur: &MotionBlurScore,
    frame_rate: &FrameRateConsistency,
) -> f32 {
    // Flickering component (0–40): perfect when std_dev = 0, 0 when std_dev ≥ 10
    let flicker_score = (1.0 - (flickering.std_dev / 10.0).min(1.0)) * 40.0;

    // Sharpness component (0–30): grade A → 30, F → 0
    let sharpness_score = match motion_blur.quality_grade() {
        'A' => 30.0,
        'B' => 24.0,
        'C' => 18.0,
        'D' => 12.0,
        'E' => 6.0,
        _ => 0.0,
    };

    // Frame-rate component (0–30): jitter < 1 ms and no drops → 30
    let jitter_penalty = (frame_rate.jitter_ms / 10.0).min(1.0) * 15.0;
    let drop_penalty = (frame_rate.dropped_frames as f32 * 3.0).min(15.0);
    let fr_score = (30.0 - jitter_penalty - drop_penalty).max(0.0);

    flicker_score + sharpness_score + fr_score
}

// ─── TemporalQualityAnalyzer ───────────────────────────────────────────────────

/// Quality metrics for a single frame pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameQuality {
    /// Index of this frame in the sequence.
    pub frame_index: usize,
    /// PSNR value (dB).
    pub psnr: f32,
    /// SSIM value in \[0, 1\].
    pub ssim: f32,
    /// FSIM value in \[0, 1\], if computed.
    pub fsim: Option<f32>,
    /// Whether a scene change was detected at this frame.
    pub is_scene_change: bool,
}

/// Statistical summary for a sequence of quality scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityStats {
    /// Arithmetic mean of all scores.
    pub mean: f32,
    /// Minimum score across all frames.
    pub min: f32,
    /// Maximum score across all frames.
    pub max: f32,
    /// Standard deviation of all scores.
    pub std_dev: f32,
    /// Number of frames included in the statistics.
    pub frame_count: usize,
}

impl QualityStats {
    /// Compute statistics from a slice of values.
    ///
    /// Returns all-zero stats for an empty slice.
    #[must_use]
    pub fn from_values(values: &[f32]) -> Self {
        if values.is_empty() {
            return Self {
                mean: 0.0,
                min: 0.0,
                max: 0.0,
                std_dev: 0.0,
                frame_count: 0,
            };
        }
        let n = values.len() as f32;
        let mean = values.iter().sum::<f32>() / n;
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        Self {
            mean,
            min,
            max,
            std_dev: variance.sqrt(),
            frame_count: values.len(),
        }
    }
}

/// Summary report produced by [`TemporalQualityAnalyzer::report`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalQualityAnalysisReport {
    /// Total number of frames assessed.
    pub total_frames: usize,
    /// Aggregate PSNR statistics.
    pub psnr_stats: QualityStats,
    /// Aggregate SSIM statistics.
    pub ssim_stats: QualityStats,
    /// Indices of frames where PSNR dropped significantly.
    pub quality_drops: Vec<usize>,
    /// Index of the frame with the lowest PSNR.
    pub worst_frame: usize,
    /// Index of the frame with the highest PSNR.
    pub best_frame: usize,
    /// Rough VMAF-like perceptual quality estimate (0–100).
    ///
    /// Derived from PSNR and SSIM via the relationship:
    /// `vmaf_est ≈ 46·SSIM_mean + 0.35·(PSNR_mean − 20)`.
    pub overall_vmaf_estimate: f32,
}

/// Analyse quality metrics across a sequence of frame pairs.
///
/// # Example
///
/// ```
/// # use oximedia_quality::{TemporalQualityAnalyzer, Frame};
/// # use oximedia_core::PixelFormat;
/// # let ref_frame = Frame::new(64, 64, PixelFormat::Yuv420p).unwrap();
/// # let dist_frame = ref_frame.clone();
/// let mut analyzer = TemporalQualityAnalyzer::new(5);
/// analyzer.add_frame_pair(&ref_frame, &dist_frame, 0);
/// let report = analyzer.report();
/// assert_eq!(report.total_frames, 1);
/// ```
pub struct TemporalQualityAnalyzer {
    frame_scores: Vec<FrameQuality>,
    /// Number of frames in the moving-average smoothing window.
    window_size: usize,
    psnr_calc: PsnrCalculator,
    ssim_calc: SsimCalculator,
}

impl TemporalQualityAnalyzer {
    /// Create a new analyzer with the given smoothing window size.
    ///
    /// A `window_size` of 1 disables smoothing.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            frame_scores: Vec::new(),
            window_size: window_size.max(1),
            psnr_calc: PsnrCalculator::new(),
            ssim_calc: SsimCalculator::new(),
        }
    }

    /// Add a frame pair at the given sequence index and compute its metrics.
    ///
    /// Metrics that fail to compute (e.g. due to size mismatch) are silently
    /// replaced with 0.0 / `None` so that a single bad frame does not abort
    /// the whole sequence.
    pub fn add_frame_pair(&mut self, reference: &Frame, distorted: &Frame, index: usize) {
        let psnr = self
            .psnr_calc
            .calculate(reference, distorted)
            .map(|s| s.score as f32)
            .unwrap_or(0.0);

        let ssim = self
            .ssim_calc
            .calculate(reference, distorted)
            .map(|s| s.score as f32)
            .unwrap_or(0.0);

        // Scene change: PSNR < 20 dB or SSIM < 0.5 compared to previous frame.
        let is_scene_change = if let Some(prev) = self.frame_scores.last() {
            psnr < 20.0 || ssim < 0.5 || (prev.psnr - psnr).abs() > 15.0
        } else {
            false
        };

        self.frame_scores.push(FrameQuality {
            frame_index: index,
            psnr,
            ssim,
            fsim: None,
            is_scene_change,
        });
    }

    /// Returns the frame score entries (immutable).
    #[must_use]
    pub fn frame_scores(&self) -> &[FrameQuality] {
        &self.frame_scores
    }

    /// Smoothed PSNR curve using a centred moving average.
    #[must_use]
    pub fn smoothed_psnr(&self) -> Vec<f32> {
        let values: Vec<f32> = self.frame_scores.iter().map(|f| f.psnr).collect();
        moving_average(&values, self.window_size)
    }

    /// Smoothed SSIM curve using a centred moving average.
    #[must_use]
    pub fn smoothed_ssim(&self) -> Vec<f32> {
        let values: Vec<f32> = self.frame_scores.iter().map(|f| f.ssim).collect();
        moving_average(&values, self.window_size)
    }

    /// Indices of frames where PSNR drops more than `psnr_drop_threshold` dB
    /// below the rolling mean at that position.
    #[must_use]
    pub fn quality_drops(&self, psnr_drop_threshold: f32) -> Vec<usize> {
        let smoothed = self.smoothed_psnr();
        self.frame_scores
            .iter()
            .zip(smoothed.iter())
            .filter_map(|(fq, &mean)| {
                if mean - fq.psnr > psnr_drop_threshold {
                    Some(fq.frame_index)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Aggregate PSNR statistics across all frames.
    #[must_use]
    pub fn psnr_stats(&self) -> QualityStats {
        let values: Vec<f32> = self.frame_scores.iter().map(|f| f.psnr).collect();
        QualityStats::from_values(&values)
    }

    /// Aggregate SSIM statistics across all frames.
    #[must_use]
    pub fn ssim_stats(&self) -> QualityStats {
        let values: Vec<f32> = self.frame_scores.iter().map(|f| f.ssim).collect();
        QualityStats::from_values(&values)
    }

    /// The `pct`-th percentile PSNR value (0–100).
    ///
    /// Uses linear interpolation between adjacent sorted values.
    /// Returns 0.0 if no frames have been added.
    #[must_use]
    pub fn psnr_percentile(&self, pct: f32) -> f32 {
        percentile_of(
            &self.frame_scores.iter().map(|f| f.psnr).collect::<Vec<_>>(),
            pct,
        )
    }

    /// The `pct`-th percentile SSIM value (0–100).
    #[must_use]
    pub fn ssim_percentile(&self, pct: f32) -> f32 {
        percentile_of(
            &self.frame_scores.iter().map(|f| f.ssim).collect::<Vec<_>>(),
            pct,
        )
    }

    /// Generate a comprehensive quality report.
    #[must_use]
    pub fn report(&self) -> TemporalQualityAnalysisReport {
        let psnr_stats = self.psnr_stats();
        let ssim_stats = self.ssim_stats();

        let quality_drops = self.quality_drops(3.0);

        let (worst_frame, best_frame) = if self.frame_scores.is_empty() {
            (0, 0)
        } else {
            let worst = self
                .frame_scores
                .iter()
                .min_by(|a, b| {
                    a.psnr
                        .partial_cmp(&b.psnr)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|f| f.frame_index)
                .unwrap_or(0);
            let best = self
                .frame_scores
                .iter()
                .max_by(|a, b| {
                    a.psnr
                        .partial_cmp(&b.psnr)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|f| f.frame_index)
                .unwrap_or(0);
            (worst, best)
        };

        // Rough VMAF-like estimate derived empirically from PSNR+SSIM
        let vmaf_from_ssim = 46.0 * ssim_stats.mean;
        let vmaf_from_psnr = 0.35 * (psnr_stats.mean - 20.0).max(0.0);
        let overall_vmaf_estimate = (vmaf_from_ssim + vmaf_from_psnr).clamp(0.0, 100.0);

        TemporalQualityAnalysisReport {
            total_frames: self.frame_scores.len(),
            psnr_stats,
            ssim_stats,
            quality_drops,
            worst_frame,
            best_frame,
            overall_vmaf_estimate,
        }
    }
}

// ─── Scene-Aware Temporal Pooler ──────────────────────────────────────────────

/// Quality statistics for a single detected scene segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneSegmentStats {
    /// Index of the first frame in the segment.
    pub start_frame: usize,
    /// Index of the last frame in the segment (inclusive).
    pub end_frame: usize,
    /// Number of frames in the segment.
    pub frame_count: usize,
    /// Mean PSNR for this scene.
    pub psnr_mean: f32,
    /// Mean SSIM for this scene.
    pub ssim_mean: f32,
    /// Minimum PSNR observed in this scene.
    pub psnr_min: f32,
    /// Minimum SSIM observed in this scene.
    pub ssim_min: f32,
}

impl SceneSegmentStats {
    /// Build stats from a slice of `FrameQuality` entries.
    #[must_use]
    pub fn from_frames(frames: &[FrameQuality]) -> Self {
        if frames.is_empty() {
            return Self {
                start_frame: 0,
                end_frame: 0,
                frame_count: 0,
                psnr_mean: 0.0,
                ssim_mean: 0.0,
                psnr_min: 0.0,
                ssim_min: 0.0,
            };
        }
        let n = frames.len() as f32;
        let psnr_mean = frames.iter().map(|f| f.psnr).sum::<f32>() / n;
        let ssim_mean = frames.iter().map(|f| f.ssim).sum::<f32>() / n;
        let psnr_min = frames.iter().map(|f| f.psnr).fold(f32::INFINITY, f32::min);
        let ssim_min = frames.iter().map(|f| f.ssim).fold(f32::INFINITY, f32::min);
        Self {
            start_frame: frames[0].frame_index,
            end_frame: frames[frames.len() - 1].frame_index,
            frame_count: frames.len(),
            psnr_mean,
            ssim_mean,
            psnr_min,
            ssim_min,
        }
    }
}

/// Scene-aware temporal quality pooler.
///
/// Groups frames into scenes by detecting scene cuts (via `FrameQuality::is_scene_change`)
/// and computes per-scene statistics independently.  This avoids the averaging artefact
/// that occurs when metrics from very different scenes are pooled together.
///
/// # Example
///
/// ```
/// use oximedia_quality::temporal_quality::SceneAwarePooler;
/// use oximedia_quality::TemporalQualityAnalyzer;
/// use oximedia_quality::Frame;
/// use oximedia_core::PixelFormat;
///
/// let ref_frame = Frame::new(32, 32, PixelFormat::Yuv420p).unwrap();
/// let mut analyzer = TemporalQualityAnalyzer::new(1);
/// analyzer.add_frame_pair(&ref_frame, &ref_frame, 0);
///
/// let mut pooler = SceneAwarePooler::new();
/// pooler.ingest(analyzer.frame_scores());
/// let report = pooler.report();
/// assert_eq!(report.scene_count, 1);
/// ```
pub struct SceneAwarePooler {
    /// Collected frame quality entries across all ingested sequences.
    frames: Vec<FrameQuality>,
}

/// Report produced by [`SceneAwarePooler::report`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneAwarePoolingReport {
    /// Number of detected scene segments.
    pub scene_count: usize,
    /// Per-scene statistics.
    pub scenes: Vec<SceneSegmentStats>,
    /// Weighted overall PSNR mean (weighted by scene length).
    pub overall_psnr: f32,
    /// Weighted overall SSIM mean (weighted by scene length).
    pub overall_ssim: f32,
    /// Index of the scene segment with the lowest mean PSNR.
    pub worst_scene_idx: usize,
    /// Index of the scene segment with the highest mean PSNR.
    pub best_scene_idx: usize,
}

impl SceneAwarePooler {
    /// Creates a new pooler with no frames.
    #[must_use]
    pub fn new() -> Self {
        Self { frames: Vec::new() }
    }

    /// Ingest frame quality entries from a [`TemporalQualityAnalyzer`].
    pub fn ingest(&mut self, frames: &[FrameQuality]) {
        self.frames.extend_from_slice(frames);
    }

    /// Reset all ingested frames, enabling reuse.
    pub fn reset(&mut self) {
        self.frames.clear();
    }

    /// Split the accumulated frames into scene segments and compute the report.
    ///
    /// A new segment begins whenever `FrameQuality::is_scene_change` is `true`.
    /// If no frames have been ingested, returns a zero-valued report with 0 scenes.
    #[must_use]
    pub fn report(&self) -> SceneAwarePoolingReport {
        if self.frames.is_empty() {
            return SceneAwarePoolingReport {
                scene_count: 0,
                scenes: Vec::new(),
                overall_psnr: 0.0,
                overall_ssim: 0.0,
                worst_scene_idx: 0,
                best_scene_idx: 0,
            };
        }

        // Partition frames into scene segments.
        let mut segments: Vec<Vec<FrameQuality>> = Vec::new();
        let mut current: Vec<FrameQuality> = Vec::new();

        for fq in &self.frames {
            if fq.is_scene_change && !current.is_empty() {
                segments.push(std::mem::take(&mut current));
            }
            current.push(fq.clone());
        }
        if !current.is_empty() {
            segments.push(current);
        }

        let scenes: Vec<SceneSegmentStats> = segments
            .iter()
            .map(|seg| SceneSegmentStats::from_frames(seg))
            .collect();

        let scene_count = scenes.len();

        // Weighted overall metrics (weight = frame count)
        let total_frames: usize = scenes.iter().map(|s| s.frame_count).sum();
        let (overall_psnr, overall_ssim) = if total_frames == 0 {
            (0.0, 0.0)
        } else {
            let tf = total_frames as f32;
            let p = scenes
                .iter()
                .map(|s| s.psnr_mean * s.frame_count as f32)
                .sum::<f32>()
                / tf;
            let s = scenes
                .iter()
                .map(|s| s.ssim_mean * s.frame_count as f32)
                .sum::<f32>()
                / tf;
            (p, s)
        };

        let worst_scene_idx = scenes
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.psnr_mean
                    .partial_cmp(&b.psnr_mean)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or(0, |(i, _)| i);

        let best_scene_idx = scenes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.psnr_mean
                    .partial_cmp(&b.psnr_mean)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or(0, |(i, _)| i);

        SceneAwarePoolingReport {
            scene_count,
            scenes,
            overall_psnr,
            overall_ssim,
            worst_scene_idx,
            best_scene_idx,
        }
    }
}

impl Default for SceneAwarePooler {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute centred moving average of `values` with given `window`.
fn moving_average(values: &[f32], window: usize) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let half = window / 2;
    (0..values.len())
        .map(|i| {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(values.len());
            let slice = &values[lo..hi];
            slice.iter().sum::<f32>() / slice.len() as f32
        })
        .collect()
}

/// Linear-interpolated percentile of a slice of f32 values.
fn percentile_of(values: &[f32], pct: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let idx_f = (pct / 100.0).clamp(0.0, 1.0) * (n - 1) as f32;
    let lo = idx_f.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = idx_f - lo as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── TemporalFlickering ──────────────────────────────────────────────────

    #[test]
    fn test_flickering_stable_sequence() {
        let lumas = vec![100.0f32; 30]; // constant brightness
        let f = TemporalFlickering::analyze(&lumas);
        assert_eq!(f.mean, 0.0);
        assert_eq!(f.std_dev, 0.0);
        assert_eq!(f.max_delta, 0.0);
        assert_eq!(f.flickering_frames, 0);
        assert!(f.is_acceptable());
    }

    #[test]
    fn test_flickering_single_frame() {
        let lumas = vec![100.0f32];
        let f = TemporalFlickering::analyze(&lumas);
        assert_eq!(f.std_dev, 0.0);
    }

    #[test]
    fn test_flickering_large_variance() {
        let lumas: Vec<f32> = (0..20)
            .map(|i| if i % 2 == 0 { 100.0 } else { 120.0 })
            .collect();
        let f = TemporalFlickering::analyze(&lumas);
        assert!(f.mean > 0.0);
        // std_dev should be very low (all deltas equal) → 0 flickering_frames
        assert_eq!(f.flickering_frames, 0);
        assert!(f.is_acceptable());
    }

    #[test]
    fn test_flickering_is_acceptable_threshold() {
        let f_ok = TemporalFlickering {
            mean: 0.5,
            std_dev: 2.9,
            max_delta: 3.0,
            flickering_frames: 0,
        };
        assert!(f_ok.is_acceptable());

        let f_fail = TemporalFlickering {
            mean: 1.0,
            std_dev: 3.1,
            max_delta: 5.0,
            flickering_frames: 2,
        };
        assert!(!f_fail.is_acceptable());
    }

    #[test]
    fn test_flickering_max_delta() {
        let lumas = vec![100.0, 115.0, 110.0, 100.0];
        let f = TemporalFlickering::analyze(&lumas);
        assert!((f.max_delta - 15.0).abs() < 1e-4);
    }

    // ── MotionBlurDetector ──────────────────────────────────────────────────

    #[test]
    fn test_motion_blur_detector_sharp_frame() {
        // Checkerboard pattern → high Laplacian variance
        let w = 8u32;
        let h = 8u32;
        let frame: Vec<f32> = (0..w * h)
            .map(|i| {
                let row = i / w;
                let col = i % w;
                if (row + col) % 2 == 0 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let score = MotionBlurDetector::compute_score(&frame, w, h);
        assert!(
            score > 0.0,
            "sharp checkerboard should have a positive score"
        );
    }

    #[test]
    fn test_motion_blur_detector_flat_frame() {
        let frame = vec![0.5f32; 16 * 16];
        let score = MotionBlurDetector::compute_score(&frame, 16, 16);
        assert_eq!(
            score, 0.0,
            "uniform frame should have zero Laplacian variance"
        );
    }

    #[test]
    fn test_motion_blur_detector_small_frame() {
        let frame = vec![0.5f32; 4];
        let score = MotionBlurDetector::compute_score(&frame, 2, 2);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_motion_blur_score_grade() {
        assert_eq!(
            MotionBlurScore {
                average: 600.0,
                sharpest_frame: 0,
                blurriest_frame: 0
            }
            .quality_grade(),
            'A'
        );
        assert_eq!(
            MotionBlurScore {
                average: 350.0,
                sharpest_frame: 0,
                blurriest_frame: 0
            }
            .quality_grade(),
            'B'
        );
        assert_eq!(
            MotionBlurScore {
                average: 200.0,
                sharpest_frame: 0,
                blurriest_frame: 0
            }
            .quality_grade(),
            'C'
        );
        assert_eq!(
            MotionBlurScore {
                average: 80.0,
                sharpest_frame: 0,
                blurriest_frame: 0
            }
            .quality_grade(),
            'D'
        );
        assert_eq!(
            MotionBlurScore {
                average: 25.0,
                sharpest_frame: 0,
                blurriest_frame: 0
            }
            .quality_grade(),
            'E'
        );
        assert_eq!(
            MotionBlurScore {
                average: 1.0,
                sharpest_frame: 0,
                blurriest_frame: 0
            }
            .quality_grade(),
            'F'
        );
    }

    #[test]
    fn test_motion_blur_score_from_frame_scores() {
        let scores = vec![100.0f32, 200.0, 300.0, 50.0];
        let s = MotionBlurScore::from_frame_scores(&scores);
        assert!((s.average - 162.5).abs() < 1e-3);
        assert_eq!(s.sharpest_frame, 2);
        assert_eq!(s.blurriest_frame, 3);
    }

    #[test]
    fn test_motion_blur_score_empty() {
        let s = MotionBlurScore::from_frame_scores(&[]);
        assert_eq!(s.average, 0.0);
    }

    // ── FrameRateConsistency ───────────────────────────────────────────────

    #[test]
    fn test_frame_rate_perfect() {
        // Exactly 25 fps: 40 ms per frame
        let ts: Vec<u64> = (0..=25).map(|i| i * 40).collect();
        let fr = FrameRateConsistency::analyze(&ts);
        assert!(
            (fr.nominal_fps - 25.0).abs() < 0.1,
            "fps={}",
            fr.nominal_fps
        );
        assert!(fr.jitter_ms < 1.0);
        assert_eq!(fr.dropped_frames, 0);
        assert!(fr.is_consistent());
    }

    #[test]
    fn test_frame_rate_dropped() {
        // One frame is double-duration (dropped)
        let mut ts: Vec<u64> = (0..10).map(|i| i * 33).collect();
        ts[5] = ts[4] + 66; // simulate drop
        let fr = FrameRateConsistency::analyze(&ts);
        assert!(fr.dropped_frames >= 1);
    }

    #[test]
    fn test_frame_rate_single_ts() {
        let ts = vec![0u64];
        let fr = FrameRateConsistency::analyze(&ts);
        assert_eq!(fr.nominal_fps, 0.0);
    }

    #[test]
    fn test_frame_rate_empty() {
        let fr = FrameRateConsistency::analyze(&[]);
        assert_eq!(fr.nominal_fps, 0.0);
    }

    // ── TemporalQualityReport ──────────────────────────────────────────────

    #[test]
    fn test_temporal_quality_report_construction() {
        let flickering = TemporalFlickering::analyze(&[100.0; 30]);
        let blur_scores = vec![500.0f32; 10];
        let motion_blur = MotionBlurScore::from_frame_scores(&blur_scores);
        let ts: Vec<u64> = (0..=25).map(|i| i * 40).collect();
        let frame_rate = FrameRateConsistency::analyze(&ts);

        let report = TemporalQualityReport::new(flickering, motion_blur, frame_rate);
        assert!(report.overall_score >= 0.0 && report.overall_score <= 100.0);
    }

    #[test]
    fn test_temporal_quality_report_high_quality() {
        let flickering = TemporalFlickering {
            mean: 0.0,
            std_dev: 0.0,
            max_delta: 0.0,
            flickering_frames: 0,
        };
        let motion_blur = MotionBlurScore {
            average: 600.0,
            sharpest_frame: 0,
            blurriest_frame: 0,
        };
        let frame_rate = FrameRateConsistency {
            nominal_fps: 25.0,
            measured_fps: vec![25.0; 25],
            jitter_ms: 0.0,
            dropped_frames: 0,
        };
        let report = TemporalQualityReport::new(flickering, motion_blur, frame_rate);
        assert!(
            report.overall_score > 80.0,
            "score={}",
            report.overall_score
        );
    }

    // ── TemporalQualityAnalyzer ────────────────────────────────────────────

    use crate::Frame as QFrame;
    use oximedia_core::PixelFormat;

    fn make_yuv_frame(width: usize, height: usize, y_val: u8) -> QFrame {
        let mut f =
            QFrame::new(width, height, PixelFormat::Yuv420p).expect("should succeed in test");
        f.planes[0].fill(y_val);
        f.planes[1].fill(128);
        f.planes[2].fill(128);
        f
    }

    #[test]
    fn test_analyzer_empty_report() {
        let analyzer = TemporalQualityAnalyzer::new(5);
        let report = analyzer.report();
        assert_eq!(report.total_frames, 0);
        assert_eq!(report.psnr_stats.frame_count, 0);
    }

    #[test]
    fn test_analyzer_single_identical_frame() {
        let mut analyzer = TemporalQualityAnalyzer::new(1);
        let frame = make_yuv_frame(64, 64, 128);
        analyzer.add_frame_pair(&frame, &frame, 0);
        let report = analyzer.report();
        assert_eq!(report.total_frames, 1);
        // Identical frames → very high PSNR
        assert!(
            report.psnr_stats.mean > 50.0,
            "psnr={}",
            report.psnr_stats.mean
        );
        assert!(
            report.ssim_stats.mean > 0.98,
            "ssim={}",
            report.ssim_stats.mean
        );
    }

    #[test]
    fn test_analyzer_multiple_frames_stats() {
        let mut analyzer = TemporalQualityAnalyzer::new(3);
        let ref_frame = make_yuv_frame(32, 32, 128);
        for i in 0..10usize {
            let dist_frame = make_yuv_frame(32, 32, (128 + i as u8 * 2).min(200));
            analyzer.add_frame_pair(&ref_frame, &dist_frame, i);
        }
        let report = analyzer.report();
        assert_eq!(report.total_frames, 10);
        assert!(report.psnr_stats.mean > 0.0);
        assert!(report.psnr_stats.min <= report.psnr_stats.mean);
        assert!(report.psnr_stats.max >= report.psnr_stats.mean);
    }

    #[test]
    fn test_analyzer_smoothed_psnr_length() {
        let mut analyzer = TemporalQualityAnalyzer::new(5);
        let ref_frame = make_yuv_frame(32, 32, 128);
        for i in 0..20usize {
            let dist = make_yuv_frame(32, 32, (128_u8).wrapping_add(i as u8));
            analyzer.add_frame_pair(&ref_frame, &dist, i);
        }
        let smoothed = analyzer.smoothed_psnr();
        assert_eq!(
            smoothed.len(),
            20,
            "smoothed PSNR length must equal frame count"
        );
    }

    #[test]
    fn test_analyzer_smoothed_ssim_length() {
        let mut analyzer = TemporalQualityAnalyzer::new(3);
        let ref_frame = make_yuv_frame(32, 32, 100);
        for i in 0..8usize {
            let dist = make_yuv_frame(32, 32, (100_u8).wrapping_add(i as u8 * 3));
            analyzer.add_frame_pair(&ref_frame, &dist, i);
        }
        let smoothed = analyzer.smoothed_ssim();
        assert_eq!(smoothed.len(), 8);
    }

    #[test]
    fn test_analyzer_quality_drops_detection() {
        let mut analyzer = TemporalQualityAnalyzer::new(5);
        let ref_frame = make_yuv_frame(64, 64, 128);
        // 8 identical frames
        for i in 0..8usize {
            analyzer.add_frame_pair(&ref_frame, &ref_frame, i);
        }
        // 1 very bad frame
        let bad_frame = make_yuv_frame(64, 64, 0);
        analyzer.add_frame_pair(&ref_frame, &bad_frame, 8);
        // More identical frames
        for i in 9..15usize {
            analyzer.add_frame_pair(&ref_frame, &ref_frame, i);
        }
        let drops = analyzer.quality_drops(5.0);
        assert!(!drops.is_empty(), "should detect quality drop at frame 8");
    }

    #[test]
    fn test_analyzer_worst_best_frame() {
        let mut analyzer = TemporalQualityAnalyzer::new(1);
        let ref_frame = make_yuv_frame(32, 32, 128);
        // Frame 0: close to reference (high PSNR)
        let good = make_yuv_frame(32, 32, 130);
        analyzer.add_frame_pair(&ref_frame, &good, 0);
        // Frame 1: far from reference (low PSNR)
        let bad = make_yuv_frame(32, 32, 50);
        analyzer.add_frame_pair(&ref_frame, &bad, 1);
        let report = analyzer.report();
        assert_eq!(report.best_frame, 0, "best frame should be index 0");
        assert_eq!(report.worst_frame, 1, "worst frame should be index 1");
    }

    #[test]
    fn test_analyzer_psnr_percentile_5th_below_median() {
        let mut analyzer = TemporalQualityAnalyzer::new(1);
        let ref_frame = make_yuv_frame(32, 32, 128);
        for i in 0..20usize {
            // Incrementally worse distortion
            let dist = make_yuv_frame(32, 32, (128_u8).saturating_add(i as u8 * 5));
            analyzer.add_frame_pair(&ref_frame, &dist, i);
        }
        let p5 = analyzer.psnr_percentile(5.0);
        let p50 = analyzer.psnr_percentile(50.0);
        let p95 = analyzer.psnr_percentile(95.0);
        assert!(p5 <= p50, "p5={p5} should be <= p50={p50}");
        assert!(p50 <= p95, "p50={p50} should be <= p95={p95}");
    }

    #[test]
    fn test_analyzer_vmaf_estimate_in_range() {
        let mut analyzer = TemporalQualityAnalyzer::new(3);
        let ref_frame = make_yuv_frame(32, 32, 128);
        let dist_frame = make_yuv_frame(32, 32, 132);
        for i in 0..10usize {
            analyzer.add_frame_pair(&ref_frame, &dist_frame, i);
        }
        let report = analyzer.report();
        assert!(
            report.overall_vmaf_estimate >= 0.0 && report.overall_vmaf_estimate <= 100.0,
            "vmaf_estimate out of range: {}",
            report.overall_vmaf_estimate
        );
    }

    #[test]
    fn test_quality_stats_from_values() {
        let values = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];
        let stats = QualityStats::from_values(&values);
        assert!((stats.mean - 30.0).abs() < 1e-4, "mean={}", stats.mean);
        assert!((stats.min - 10.0).abs() < 1e-4);
        assert!((stats.max - 50.0).abs() < 1e-4);
        assert_eq!(stats.frame_count, 5);
        // std_dev of [10,20,30,40,50] = sqrt(200) ≈ 14.14
        assert!(
            (stats.std_dev - 14.142).abs() < 0.01,
            "std_dev={}",
            stats.std_dev
        );
    }

    #[test]
    fn test_quality_stats_empty() {
        let stats = QualityStats::from_values(&[]);
        assert_eq!(stats.frame_count, 0);
        assert_eq!(stats.mean, 0.0);
    }

    #[test]
    fn test_percentile_edge_cases() {
        let values = vec![5.0f32, 10.0, 15.0, 20.0, 25.0];
        assert!((percentile_of(&values, 0.0) - 5.0).abs() < 1e-4, "p0");
        assert!((percentile_of(&values, 100.0) - 25.0).abs() < 1e-4, "p100");
        // median
        let med = percentile_of(&values, 50.0);
        assert!((med - 15.0).abs() < 1e-3, "median={med}");
    }

    #[test]
    fn test_moving_average_window_one() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let ma = moving_average(&values, 1);
        assert_eq!(ma, values, "window=1 should be identity");
    }

    #[test]
    fn test_moving_average_window_full() {
        let values = vec![1.0f32, 3.0, 5.0, 7.0, 9.0];
        let ma = moving_average(&values, values.len());
        // All values should be within [1, 9] and have the same length
        assert_eq!(ma.len(), values.len());
        // The centre value (index 2) must be closest to the global mean = 5.0
        let centre = ma[2];
        assert!((centre - 5.0).abs() < 0.01, "centre ma={centre}");
        // Each smoothed value must be at least the minimum and at most the maximum
        for v in &ma {
            assert!(*v >= 1.0 && *v <= 9.0, "out of range: {v}");
        }
    }

    // ── SceneAwarePooler tests ─────────────────────────────────────────────

    use super::{SceneAwarePooler, SceneSegmentStats};

    #[test]
    fn test_scene_pooler_empty() {
        let pooler = SceneAwarePooler::new();
        let report = pooler.report();
        assert_eq!(report.scene_count, 0);
        assert_eq!(report.scenes.len(), 0);
        assert_eq!(report.overall_psnr, 0.0);
    }

    #[test]
    fn test_scene_pooler_single_scene_no_cuts() {
        // 5 frames with no scene changes → 1 segment
        let frames: Vec<FrameQuality> = (0..5)
            .map(|i| FrameQuality {
                frame_index: i,
                psnr: 40.0,
                ssim: 0.95,
                fsim: None,
                is_scene_change: false,
            })
            .collect();

        let mut pooler = SceneAwarePooler::new();
        pooler.ingest(&frames);
        let report = pooler.report();

        assert_eq!(report.scene_count, 1);
        assert!((report.overall_psnr - 40.0).abs() < 1e-4);
        assert!((report.overall_ssim - 0.95).abs() < 1e-4);
    }

    #[test]
    fn test_scene_pooler_two_scenes() {
        // First 4 frames: psnr=40, then scene cut, next 4: psnr=30
        let mut frames: Vec<FrameQuality> = (0..4)
            .map(|i| FrameQuality {
                frame_index: i,
                psnr: 40.0,
                ssim: 0.95,
                fsim: None,
                is_scene_change: false,
            })
            .collect();
        // Scene cut at index 4
        frames.push(FrameQuality {
            frame_index: 4,
            psnr: 30.0,
            ssim: 0.80,
            fsim: None,
            is_scene_change: true,
        });
        for i in 5..8 {
            frames.push(FrameQuality {
                frame_index: i,
                psnr: 30.0,
                ssim: 0.80,
                fsim: None,
                is_scene_change: false,
            });
        }

        let mut pooler = SceneAwarePooler::new();
        pooler.ingest(&frames);
        let report = pooler.report();

        assert_eq!(report.scene_count, 2, "expected 2 scenes");
        // Scene 0 (4 frames, psnr=40) and scene 1 (4 frames, psnr=30)
        assert!(
            (report.scenes[0].psnr_mean - 40.0).abs() < 1e-3,
            "scene0 psnr"
        );
        assert!(
            (report.scenes[1].psnr_mean - 30.0).abs() < 1e-3,
            "scene1 psnr"
        );
        // Weighted overall: (4*40 + 4*30) / 8 = 35
        assert!(
            (report.overall_psnr - 35.0).abs() < 1e-3,
            "overall={}",
            report.overall_psnr
        );
        assert_eq!(report.worst_scene_idx, 1, "worst scene is second");
        assert_eq!(report.best_scene_idx, 0, "best scene is first");
    }

    #[test]
    fn test_scene_pooler_reset() {
        let frames: Vec<FrameQuality> = (0..3)
            .map(|i| FrameQuality {
                frame_index: i,
                psnr: 35.0,
                ssim: 0.90,
                fsim: None,
                is_scene_change: false,
            })
            .collect();
        let mut pooler = SceneAwarePooler::new();
        pooler.ingest(&frames);
        pooler.reset();
        let report = pooler.report();
        assert_eq!(report.scene_count, 0);
    }

    #[test]
    fn test_scene_segment_stats_from_empty() {
        let stats = SceneSegmentStats::from_frames(&[]);
        assert_eq!(stats.frame_count, 0);
        assert_eq!(stats.psnr_mean, 0.0);
    }

    #[test]
    fn test_scene_pooler_via_analyzer() {
        // Use TemporalQualityAnalyzer to produce frame scores, then feed to pooler.
        let mut analyzer = TemporalQualityAnalyzer::new(1);
        let ref_frame = make_yuv_frame(32, 32, 128);
        let similar = make_yuv_frame(32, 32, 130);
        let different = make_yuv_frame(32, 32, 0); // triggers scene-change heuristic

        for i in 0..5 {
            analyzer.add_frame_pair(&ref_frame, &similar, i);
        }
        // One very different frame to trigger a scene cut
        analyzer.add_frame_pair(&ref_frame, &different, 5);
        for i in 6..10 {
            analyzer.add_frame_pair(&ref_frame, &similar, i);
        }

        let mut pooler = SceneAwarePooler::new();
        pooler.ingest(analyzer.frame_scores());
        let report = pooler.report();

        // We should get at least 2 scenes due to the bad frame
        assert!(
            report.scene_count >= 1,
            "expected at least 1 scene, got {}",
            report.scene_count
        );
        assert_eq!(
            report.scenes.iter().map(|s| s.frame_count).sum::<usize>(),
            10,
            "all frames must be accounted for"
        );
    }
}
