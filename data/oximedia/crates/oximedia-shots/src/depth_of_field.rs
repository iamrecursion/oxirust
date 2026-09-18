//! Depth-of-field (DOF) estimation for video frames.
//!
//! Classifies shots as *shallow focus*, *normal focus*, or *deep focus* using
//! a purely photometric approach: the spatial distribution of blur (measured
//! via Laplacian variance) across horizontal bands of the frame is the primary
//! signal.
//!
//! # Algorithm overview
//!
//! 1. **Laplacian variance map** – each pixel is assigned a local sharpness
//!    score derived from the discrete Laplacian operator. High variance = sharp,
//!    low variance = blurry.
//! 2. **Band analysis** – the frame is divided into `N` equal horizontal bands
//!    (default 8). The mean sharpness of each band is computed.
//! 3. **Focus zone detection** – the band(s) with the highest sharpness form
//!    the *focus zone*. The ratio of in-focus pixels to total pixels gives the
//!    *sharpness ratio*.
//! 4. **Foreground/background blur** – bands above the focus zone are the
//!    background (higher in frame = further for typical photography); bands
//!    below are closer (out-of-focus foreground in some configurations).
//! 5. **Classification** – thresholds on the sharpness ratio, focus-zone width,
//!    and background blur yield the DOF class and a confidence score.

use crate::error::{ShotError, ShotResult};
use crate::frame_buffer::{FrameBuffer, GrayImage};

// ---------------------------------------------------------------------------
// Public Types
// ---------------------------------------------------------------------------

/// Depth-of-field classification result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DofClass {
    /// Shallow depth of field: subject is sharp, background/foreground blurry.
    /// Typical of portrait lenses at large apertures (f/1.4–f/2.8).
    Shallow,
    /// Normal depth of field: moderate blur separation. Typical of f/4–f/8.
    Normal,
    /// Deep depth of field: most of the frame is sharp (landscape / stop-down).
    Deep,
}

impl DofClass {
    /// Human-readable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Shallow => "Shallow Focus",
            Self::Normal => "Normal Focus",
            Self::Deep => "Deep Focus",
        }
    }
}

/// Detailed DOF estimation result.
#[derive(Debug, Clone)]
pub struct DofEstimate {
    /// Overall DOF classification.
    pub class: DofClass,
    /// Confidence in the classification (0.0–1.0).
    pub confidence: f32,
    /// Mean sharpness score of the in-focus zone (0.0–1.0).
    pub focus_zone_sharpness: f32,
    /// Mean sharpness score of the background zone (0.0–1.0).
    pub background_sharpness: f32,
    /// Fraction of bands classified as in-focus (0.0–1.0).
    pub sharpness_ratio: f32,
    /// Estimated focus band index (0 = top, N-1 = bottom).
    pub focus_band_index: usize,
    /// Background blur amount: ratio of focus-zone sharpness to background
    /// sharpness (values > 1.0 indicate background is blurrier).
    pub background_blur_ratio: f32,
    /// Overall frame sharpness variance across bands (high = strong DOF
    /// separation, low = uniformly sharp/blurry).
    pub band_variance: f32,
}

/// Configuration for the DOF estimator.
#[derive(Debug, Clone)]
pub struct DofConfig {
    /// Number of horizontal bands to divide the frame into.
    pub num_bands: usize,
    /// Laplacian kernel size: 3 (3×3 kernel) or 5 (5×5 kernel).
    /// Larger kernels are more robust to noise.
    pub kernel_size: usize,
    /// Sharpness threshold above which a band is considered "in focus"
    /// (fraction of maximum band sharpness, 0.0–1.0).
    pub in_focus_threshold: f32,
    /// Ratio above which shallow DOF is declared
    /// (focus_zone_sharpness / background_sharpness).
    pub shallow_blur_ratio: f32,
    /// Minimum sharpness ratio for deep DOF classification.
    pub deep_sharpness_ratio: f32,
}

impl Default for DofConfig {
    fn default() -> Self {
        Self {
            num_bands: 8,
            kernel_size: 3,
            in_focus_threshold: 0.60,
            shallow_blur_ratio: 2.5,
            deep_sharpness_ratio: 0.75,
        }
    }
}

// ---------------------------------------------------------------------------
// Estimator
// ---------------------------------------------------------------------------

/// Depth-of-field estimator.
pub struct DofEstimator {
    config: DofConfig,
}

impl Default for DofEstimator {
    fn default() -> Self {
        Self::new(DofConfig::default())
    }
}

impl DofEstimator {
    /// Create a new estimator with the given configuration.
    #[must_use]
    pub fn new(config: DofConfig) -> Self {
        Self { config }
    }

    /// Estimate the depth of field for a single frame.
    ///
    /// # Errors
    ///
    /// Returns `ShotError::InvalidFrame` if the frame has fewer than 3
    /// channels or is too small to analyse.
    pub fn estimate(&self, frame: &FrameBuffer) -> ShotResult<DofEstimate> {
        let (h, w, ch) = frame.dim();
        if ch < 3 {
            return Err(ShotError::InvalidFrame(
                "Frame must have at least 3 channels".to_string(),
            ));
        }
        if h < 8 || w < 8 {
            return Err(ShotError::InvalidFrame(
                "Frame is too small for DOF analysis (minimum 8×8)".to_string(),
            ));
        }

        let gray = self.to_grayscale(frame);
        let sharpness_map = self.laplacian_sharpness(&gray);
        let band_sharpness = self.compute_band_sharpness(&sharpness_map, h, w);

        self.classify_dof(&band_sharpness)
    }

    /// Estimate DOF for multiple frames and return an aggregate result.
    ///
    /// Each frame is estimated independently and the results are averaged.
    /// Returns `None` if the slice is empty.
    ///
    /// # Errors
    ///
    /// Returns error if any frame is invalid.
    pub fn estimate_sequence(&self, frames: &[FrameBuffer]) -> ShotResult<Option<DofEstimate>> {
        if frames.is_empty() {
            return Ok(None);
        }

        let mut estimates = Vec::with_capacity(frames.len());
        for frame in frames {
            estimates.push(self.estimate(frame)?);
        }

        // Aggregate by averaging scalar metrics and choosing the modal class.
        let n = estimates.len() as f32;
        let mut shallow_count = 0_u32;
        let mut normal_count = 0_u32;
        let mut deep_count = 0_u32;
        let mut avg_confidence = 0.0_f32;
        let mut avg_fz_sharp = 0.0_f32;
        let mut avg_bg_sharp = 0.0_f32;
        let mut avg_ratio = 0.0_f32;
        let mut avg_focus_band = 0.0_f32;
        let mut avg_blur_ratio = 0.0_f32;
        let mut avg_band_var = 0.0_f32;

        for est in &estimates {
            match est.class {
                DofClass::Shallow => shallow_count += 1,
                DofClass::Normal => normal_count += 1,
                DofClass::Deep => deep_count += 1,
            }
            avg_confidence += est.confidence;
            avg_fz_sharp += est.focus_zone_sharpness;
            avg_bg_sharp += est.background_sharpness;
            avg_ratio += est.sharpness_ratio;
            avg_focus_band += est.focus_band_index as f32;
            avg_blur_ratio += est.background_blur_ratio;
            avg_band_var += est.band_variance;
        }

        let modal_class = if shallow_count >= normal_count && shallow_count >= deep_count {
            DofClass::Shallow
        } else if deep_count >= normal_count {
            DofClass::Deep
        } else {
            DofClass::Normal
        };

        Ok(Some(DofEstimate {
            class: modal_class,
            confidence: avg_confidence / n,
            focus_zone_sharpness: avg_fz_sharp / n,
            background_sharpness: avg_bg_sharp / n,
            sharpness_ratio: avg_ratio / n,
            focus_band_index: (avg_focus_band / n).round() as usize,
            background_blur_ratio: avg_blur_ratio / n,
            band_variance: avg_band_var / n,
        }))
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &DofConfig {
        &self.config
    }

    // ---- Private helpers ----

    /// Convert an RGB frame to grayscale using BT.601 coefficients.
    fn to_grayscale(&self, frame: &FrameBuffer) -> GrayImage {
        let (h, w, _) = frame.dim();
        let mut gray = GrayImage::zeros(h, w);
        for y in 0..h {
            for x in 0..w {
                let r = f32::from(frame.get(y, x, 0));
                let g = f32::from(frame.get(y, x, 1));
                let b = f32::from(frame.get(y, x, 2));
                gray.set(y, x, (r * 0.299 + g * 0.587 + b * 0.114) as u8);
            }
        }
        gray
    }

    /// Compute a per-pixel Laplacian sharpness map.
    ///
    /// The 3×3 discrete Laplacian kernel is:
    /// ```text
    ///  0  1  0
    ///  1 -4  1
    ///  0  1  0
    /// ```
    /// The squared response is used as the local sharpness measure.
    /// Normalised to [0, 1] with respect to the maximum possible response.
    fn laplacian_sharpness(&self, gray: &GrayImage) -> Vec<f32> {
        let (h, w) = gray.dim();
        let mut sharpness = vec![0.0_f32; h * w];

        let max_response_sq = (4.0_f32 * 255.0).powi(2); // worst-case Laplacian

        for y in 1..(h.saturating_sub(1)) {
            for x in 1..(w.saturating_sub(1)) {
                let center = i32::from(gray.get(y, x));
                let top = i32::from(gray.get(y - 1, x));
                let bottom = i32::from(gray.get(y + 1, x));
                let left = i32::from(gray.get(y, x - 1));
                let right = i32::from(gray.get(y, x + 1));

                let lap = top + bottom + left + right - 4 * center;
                let sq = (lap * lap) as f32;
                sharpness[y * w + x] = (sq / max_response_sq).min(1.0);
            }
        }

        sharpness
    }

    /// Compute the mean sharpness for each horizontal band.
    fn compute_band_sharpness(&self, sharpness: &[f32], h: usize, w: usize) -> Vec<f32> {
        let num_bands = self.config.num_bands.max(1);
        let band_height = (h / num_bands).max(1);
        let mut band_sharpness = vec![0.0_f32; num_bands];

        for band in 0..num_bands {
            let y_start = band * band_height;
            let y_end = ((band + 1) * band_height).min(h);
            let mut sum = 0.0_f32;
            let mut count = 0_u32;

            for y in y_start..y_end {
                for x in 0..w {
                    sum += sharpness[y * w + x];
                    count += 1;
                }
            }

            band_sharpness[band] = if count > 0 { sum / count as f32 } else { 0.0 };
        }

        band_sharpness
    }

    /// Classify DOF from the per-band sharpness vector.
    fn classify_dof(&self, band_sharpness: &[f32]) -> ShotResult<DofEstimate> {
        if band_sharpness.is_empty() {
            return Err(ShotError::InvalidFrame(
                "Band sharpness is empty".to_string(),
            ));
        }

        let max_sharpness = band_sharpness.iter().copied().fold(0.0_f32, f32::max);

        let focus_threshold = if max_sharpness < f32::EPSILON {
            // Uniformly blurry frame
            return Ok(DofEstimate {
                class: DofClass::Shallow,
                confidence: 0.5,
                focus_zone_sharpness: 0.0,
                background_sharpness: 0.0,
                sharpness_ratio: 0.0,
                focus_band_index: 0,
                background_blur_ratio: 1.0,
                band_variance: 0.0,
            });
        } else {
            max_sharpness * self.config.in_focus_threshold
        };

        // Identify focus band (index with maximum sharpness)
        let focus_band_index = band_sharpness
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Classify bands as in-focus vs out-of-focus
        let num_in_focus = band_sharpness
            .iter()
            .filter(|&&s| s >= focus_threshold)
            .count();
        let sharpness_ratio = num_in_focus as f32 / band_sharpness.len() as f32;

        let focus_zone_sharpness = band_sharpness[focus_band_index];

        // Background = all bands above the focus band (lower index = top of frame)
        let background_bands: Vec<f32> = if focus_band_index == 0 {
            // Focus is at the top; background is everything below.
            band_sharpness[1..].to_vec()
        } else {
            band_sharpness[..focus_band_index].to_vec()
        };

        let background_sharpness = if background_bands.is_empty() {
            focus_zone_sharpness
        } else {
            background_bands.iter().sum::<f32>() / background_bands.len() as f32
        };

        let background_blur_ratio = if background_sharpness < f32::EPSILON {
            10.0_f32 // background is very blurry
        } else {
            (focus_zone_sharpness / background_sharpness).min(10.0)
        };

        // Band-to-band variance
        let mean_sharp = band_sharpness.iter().sum::<f32>() / band_sharpness.len() as f32;
        let band_variance = band_sharpness
            .iter()
            .map(|&s| (s - mean_sharp).powi(2))
            .sum::<f32>()
            / band_sharpness.len() as f32;

        // Classify
        let (class, confidence) = if background_blur_ratio >= self.config.shallow_blur_ratio {
            let conf = ((background_blur_ratio - self.config.shallow_blur_ratio) / 5.0)
                .clamp(0.0, 1.0)
                * 0.5
                + 0.5;
            (DofClass::Shallow, conf)
        } else if sharpness_ratio >= self.config.deep_sharpness_ratio {
            let conf =
                ((sharpness_ratio - self.config.deep_sharpness_ratio) / 0.25).clamp(0.0, 1.0) * 0.5
                    + 0.5;
            (DofClass::Deep, conf)
        } else {
            // Normal: neither strongly shallow nor strongly deep
            let conf = 0.5
                + (0.5 - (background_blur_ratio - 1.0).abs() / 5.0 - (sharpness_ratio - 0.5).abs())
                    .clamp(0.0, 0.4);
            (DofClass::Normal, conf)
        };

        Ok(DofEstimate {
            class,
            confidence: confidence.clamp(0.0, 1.0),
            focus_zone_sharpness: focus_zone_sharpness / max_sharpness,
            background_sharpness: background_sharpness / max_sharpness,
            sharpness_ratio,
            focus_band_index,
            background_blur_ratio,
            band_variance,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_buffer::FrameBuffer;

    /// Create a frame where the central `focus_band_frac` of rows are sharp
    /// (high-contrast checkerboard) and the rest are smooth (uniform gray).
    fn make_shallow_focus_frame(h: usize, w: usize, focus_band_frac: f32) -> FrameBuffer {
        let mut frame = FrameBuffer::zeros(h, w, 3);
        let focus_start = ((h as f32 * (0.5 - focus_band_frac / 2.0)) as usize).min(h);
        let focus_end = ((h as f32 * (0.5 + focus_band_frac / 2.0)) as usize).min(h);

        for y in 0..h {
            for x in 0..w {
                if y >= focus_start && y < focus_end {
                    // Sharp checkerboard
                    let val = if (x + y) % 2 == 0 { 255 } else { 0 };
                    for c in 0..3 {
                        frame.set(y, x, c, val);
                    }
                } else {
                    // Smooth uniform gray (blurry simulation)
                    for c in 0..3 {
                        frame.set(y, x, c, 128);
                    }
                }
            }
        }
        frame
    }

    /// Create a uniformly sharp frame (checkerboard everywhere → deep DOF).
    fn make_deep_focus_frame(h: usize, w: usize) -> FrameBuffer {
        let mut frame = FrameBuffer::zeros(h, w, 3);
        for y in 0..h {
            for x in 0..w {
                let val = if (x + y) % 2 == 0 { 255 } else { 0 };
                for c in 0..3 {
                    frame.set(y, x, c, val);
                }
            }
        }
        frame
    }

    // ---- DofClass ----

    #[test]
    fn test_dof_class_name() {
        assert_eq!(DofClass::Shallow.name(), "Shallow Focus");
        assert_eq!(DofClass::Normal.name(), "Normal Focus");
        assert_eq!(DofClass::Deep.name(), "Deep Focus");
    }

    #[test]
    fn test_dof_class_equality() {
        assert_eq!(DofClass::Shallow, DofClass::Shallow);
        assert_ne!(DofClass::Shallow, DofClass::Deep);
    }

    // ---- DofConfig ----

    #[test]
    fn test_config_default() {
        let cfg = DofConfig::default();
        assert_eq!(cfg.num_bands, 8);
        assert_eq!(cfg.kernel_size, 3);
        assert!((cfg.in_focus_threshold - 0.60).abs() < f32::EPSILON);
    }

    // ---- DofEstimator creation ----

    #[test]
    fn test_estimator_default() {
        let est = DofEstimator::default();
        assert_eq!(est.config().num_bands, 8);
    }

    // ---- Error cases ----

    #[test]
    fn test_estimate_invalid_channels() {
        let est = DofEstimator::default();
        let frame = FrameBuffer::zeros(80, 80, 1);
        assert!(est.estimate(&frame).is_err());
    }

    #[test]
    fn test_estimate_too_small_frame() {
        let est = DofEstimator::default();
        let frame = FrameBuffer::zeros(4, 4, 3);
        assert!(est.estimate(&frame).is_err());
    }

    // ---- Deep focus detection ----

    #[test]
    fn test_estimate_deep_focus_checkerboard() {
        let est = DofEstimator::default();
        let frame = make_deep_focus_frame(80, 120);
        let result = est.estimate(&frame).expect("should succeed in test");
        assert_eq!(
            result.class,
            DofClass::Deep,
            "uniformly sharp frame should be Deep focus"
        );
        assert!(result.confidence > 0.5, "confidence should be > 0.5");
        assert!(
            result.sharpness_ratio > 0.5,
            "most bands should be in focus"
        );
    }

    // ---- Shallow focus detection ----

    #[test]
    fn test_estimate_shallow_focus_narrow_band() {
        let est = DofEstimator::default();
        // Only 25% of rows are sharp → shallow DOF
        let frame = make_shallow_focus_frame(120, 160, 0.25);
        let result = est.estimate(&frame).expect("should succeed in test");
        assert_eq!(
            result.class,
            DofClass::Shallow,
            "narrow focus band should yield Shallow DOF"
        );
        assert!(
            result.background_blur_ratio > 1.5,
            "background blur ratio should be elevated for shallow focus"
        );
    }

    // ---- Uniform blurry frame ----

    #[test]
    fn test_estimate_uniform_gray_is_shallow() {
        let est = DofEstimator::default();
        let frame = FrameBuffer::from_elem(80, 80, 3, 128);
        let result = est.estimate(&frame).expect("should succeed in test");
        // Uniform gray → no sharpness anywhere → classified as Shallow
        assert_eq!(result.class, DofClass::Shallow);
    }

    // ---- Score bounds ----

    #[test]
    fn test_estimate_scores_in_range() {
        let est = DofEstimator::default();
        for frame in [
            make_deep_focus_frame(80, 80),
            make_shallow_focus_frame(80, 80, 0.3),
            FrameBuffer::from_elem(80, 80, 3, 128),
        ] {
            let result = est.estimate(&frame).expect("should succeed in test");
            assert!(
                result.confidence >= 0.0 && result.confidence <= 1.0,
                "confidence out of range: {}",
                result.confidence
            );
            assert!(
                result.sharpness_ratio >= 0.0 && result.sharpness_ratio <= 1.0,
                "sharpness_ratio out of range"
            );
            assert!(
                result.focus_zone_sharpness >= 0.0 && result.focus_zone_sharpness <= 1.0,
                "focus_zone_sharpness out of range"
            );
            assert!(
                result.background_sharpness >= 0.0 && result.background_sharpness <= 1.0,
                "background_sharpness out of range"
            );
        }
    }

    // ---- estimate_sequence ----

    #[test]
    fn test_estimate_sequence_empty() {
        let est = DofEstimator::default();
        let result = est.estimate_sequence(&[]).expect("should succeed in test");
        assert!(result.is_none());
    }

    #[test]
    fn test_estimate_sequence_single_frame() {
        let est = DofEstimator::default();
        let frames = vec![make_deep_focus_frame(80, 80)];
        let result = est
            .estimate_sequence(&frames)
            .expect("should succeed in test");
        assert!(result.is_some());
        let agg = result.expect("expected Some");
        assert_eq!(agg.class, DofClass::Deep);
    }

    #[test]
    fn test_estimate_sequence_all_shallow_yields_shallow() {
        let est = DofEstimator::default();
        let frames: Vec<FrameBuffer> = (0..4)
            .map(|_| make_shallow_focus_frame(120, 120, 0.25))
            .collect();
        let result = est
            .estimate_sequence(&frames)
            .expect("should succeed in test")
            .expect("expected Some");
        assert_eq!(result.class, DofClass::Shallow);
    }

    #[test]
    fn test_estimate_sequence_invalid_frame_propagates_error() {
        let est = DofEstimator::default();
        let frames = vec![make_deep_focus_frame(80, 80), FrameBuffer::zeros(80, 80, 1)];
        assert!(est.estimate_sequence(&frames).is_err());
    }

    // ---- Focus band index ----

    #[test]
    fn test_focus_band_index_in_range() {
        let est = DofEstimator::default();
        let frame = make_shallow_focus_frame(80, 80, 0.3);
        let result = est.estimate(&frame).expect("should succeed in test");
        assert!(
            result.focus_band_index < est.config().num_bands,
            "focus_band_index should be within [0, num_bands)"
        );
    }

    // ---- Custom config ----

    #[test]
    fn test_custom_config_more_bands() {
        let cfg = DofConfig {
            num_bands: 16,
            ..DofConfig::default()
        };
        let est = DofEstimator::new(cfg);
        let frame = make_deep_focus_frame(128, 128);
        let result = est.estimate(&frame).expect("should succeed in test");
        assert_eq!(result.class, DofClass::Deep);
    }

    #[test]
    fn test_custom_config_high_threshold_normal() {
        // Very high shallow_blur_ratio means we rarely classify as Shallow.
        let cfg = DofConfig {
            shallow_blur_ratio: 100.0,
            deep_sharpness_ratio: 0.99,
            ..DofConfig::default()
        };
        let est = DofEstimator::new(cfg);
        let frame = make_shallow_focus_frame(80, 80, 0.5);
        let result = est.estimate(&frame).expect("should succeed in test");
        // With extreme thresholds the frame should be Normal
        assert_eq!(result.class, DofClass::Normal);
    }

    // ---- band_variance > 0 for mixed frame ----

    #[test]
    fn test_band_variance_nonzero_for_mixed_frame() {
        let est = DofEstimator::default();
        let frame = make_shallow_focus_frame(80, 80, 0.25);
        let result = est.estimate(&frame).expect("should succeed in test");
        assert!(
            result.band_variance > 0.0,
            "mixed-sharpness frame should have nonzero band variance"
        );
    }

    #[test]
    fn test_band_variance_zero_for_uniform_frame() {
        let est = DofEstimator::default();
        let frame = FrameBuffer::from_elem(80, 80, 3, 128);
        let result = est.estimate(&frame).expect("should succeed in test");
        assert!(
            result.band_variance < 1e-6,
            "uniform frame should have ~zero band variance"
        );
    }
}
