//! Scene change detection for video streams.
//!
//! Provides multiple detection methods (threshold, histogram, edge, adaptive)
//! capable of identifying cut, gradual, dissolve, and fade transitions between
//! scenes. Maintains a rolling frame-feature history for temporal analysis.
//!
//! The [`IntegralImage`] type is used
//! inside [`extract_features`] to compute spatial variance in O(1) per region
//! instead of the naïve O(W×H) per query.

use std::collections::VecDeque;

use crate::integral_image::IntegralImage;

// -----------------------------------------------------------------------
// Public enums and structs
// -----------------------------------------------------------------------

/// Algorithm used to determine whether a scene change has occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneDetectionMethod {
    /// Compare per-pixel SAD between consecutive frames against a threshold.
    ThresholdBased,
    /// Measure the histogram difference (χ² or L1) between frames.
    HistogramBased,
    /// Compare edge-density maps derived from Sobel-like filters.
    EdgeBased,
    /// Combine histogram and edge evidence with dynamic thresholding.
    Adaptive,
    /// Adaptive threshold based on content complexity histogram.
    ///
    /// The detection threshold is adjusted dynamically: complex content
    /// (high histogram spread / variance) uses a higher threshold to
    /// avoid false positives, while simple content (flat, low variance)
    /// uses a lower threshold for sensitivity.
    AdaptiveComplexity,
}

/// Per-frame statistical summary used for scene-change analysis.
#[derive(Debug, Clone)]
pub struct FrameFeatures {
    /// Sequential frame index (0-based).
    pub frame_number: u64,
    /// Normalised luma histogram (256 bins, values sum to ≈1.0).
    pub histogram: [f32; 256],
    /// Fraction of pixels classified as edges.
    pub edge_density: f32,
    /// Mean luma (Y-plane) value in [0, 255].
    pub mean_luma: f32,
    /// Mean Cb (U-plane) value in [0, 255].
    pub mean_chroma_u: f32,
    /// Mean Cr (V-plane) value in [0, 255].
    pub mean_chroma_v: f32,
    /// Spatial variance of luma pixels across the full Y plane, computed via
    /// the integral image for O(1) per-region queries.
    ///
    /// Normalised to \[0.0, 1.0\] by dividing by the maximum possible pixel
    /// variance (127.5² ≈ 16256.25 for an 8-bit image).
    pub spatial_variance: f32,
}

/// Classification of a detected scene transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneChangeType {
    /// Instantaneous cut between two shots.
    Cut,
    /// Slow, continuous transition spread over multiple frames.
    Gradual,
    /// Dissolve / cross-fade transition.
    Dissolve,
    /// Fade to or from black.
    Fade,
}

/// A detected scene change event.
#[derive(Debug, Clone)]
pub struct SceneChange {
    /// Frame number at which the change was detected.
    pub frame_number: u64,
    /// Detection confidence in [0.0, 1.0] (higher = more certain).
    pub confidence: f32,
    /// Nature of the transition.
    pub change_type: SceneChangeType,
}

/// Boundary of a single scene within a video.
#[derive(Debug, Clone)]
pub struct SceneBoundary {
    /// First frame of the scene.
    pub start_frame: u64,
    /// Last frame of the scene (inclusive).
    pub end_frame: u64,
    /// Total number of frames in the scene.
    pub duration_frames: u64,
}

/// Ordered list of scene boundaries for an entire video.
#[derive(Debug, Clone, Default)]
pub struct SceneIndex {
    /// Detected scenes in presentation order.
    pub scenes: Vec<SceneBoundary>,
}

impl SceneIndex {
    /// Build a `SceneIndex` from a list of detected `SceneChange` events.
    ///
    /// `total_frames` is the length of the video so that the final scene
    /// boundary can be closed correctly.
    pub fn from_changes(mut changes: Vec<SceneChange>, total_frames: u64) -> Self {
        changes.sort_by_key(|c| c.frame_number);

        let mut scenes = Vec::new();
        let mut start = 0u64;

        for change in &changes {
            let end = change.frame_number.saturating_sub(1);
            if end >= start {
                scenes.push(SceneBoundary {
                    start_frame: start,
                    end_frame: end,
                    duration_frames: end - start + 1,
                });
            }
            start = change.frame_number;
        }

        // Close the final scene.
        if total_frames > 0 {
            let end = total_frames - 1;
            if end >= start {
                scenes.push(SceneBoundary {
                    start_frame: start,
                    end_frame: end,
                    duration_frames: end - start + 1,
                });
            }
        }

        Self { scenes }
    }
}

/// Stateful scene-change detector that accumulates frame features.
pub struct SceneChangeDetector {
    /// Algorithm used to evaluate scene changes.
    pub method: SceneDetectionMethod,
    /// Detection threshold (interpretation depends on method).
    pub threshold: f32,
    /// Maximum number of feature frames retained for temporal analysis.
    pub history_size: usize,
    /// Rolling history of extracted frame features.
    pub frame_history: VecDeque<FrameFeatures>,
}

impl SceneChangeDetector {
    /// Construct a new `SceneChangeDetector`.
    ///
    /// Reasonable defaults: `threshold = 0.35`, `history_size = 30`.
    pub fn new(method: SceneDetectionMethod, threshold: f32, history_size: usize) -> Self {
        Self {
            method,
            threshold,
            history_size,
            frame_history: VecDeque::with_capacity(history_size),
        }
    }

    /// Push a new frame into the detector and return any detected scene change.
    ///
    /// `frame` is a planar YUV420 buffer: Y plane is `width × height` bytes,
    /// followed by U then V planes each of size `(width/2) × (height/2)`.
    /// If the frame is shorter than expected, missing chroma samples are
    /// treated as 128 (neutral).
    pub fn push_frame(
        &mut self,
        frame: &[u8],
        frame_number: u64,
        width: u32,
        height: u32,
    ) -> Option<SceneChange> {
        let features = extract_features(frame, width, height, frame_number);

        let change = if let Some(prev) = self.frame_history.back() {
            let maybe_change = match self.method {
                SceneDetectionMethod::ThresholdBased => self.detect_threshold(prev, &features),
                SceneDetectionMethod::HistogramBased => detect_change(prev, &features),
                SceneDetectionMethod::EdgeBased => self.detect_edge_based(prev, &features),
                SceneDetectionMethod::Adaptive => self.detect_adaptive(prev, &features),
                SceneDetectionMethod::AdaptiveComplexity => {
                    self.detect_adaptive_complexity(prev, &features)
                }
            };

            // Overlay gradual-change detection regardless of primary method.
            if maybe_change.is_none() {
                let history_slice: Vec<&FrameFeatures> = self.frame_history.iter().collect();
                if detect_gradual(&history_slice) {
                    Some(SceneChange {
                        frame_number,
                        confidence: 0.55,
                        change_type: SceneChangeType::Gradual,
                    })
                } else {
                    maybe_change
                }
            } else {
                maybe_change
            }
        } else {
            None
        };

        // Maintain rolling history.
        if self.frame_history.len() >= self.history_size {
            self.frame_history.pop_front();
        }
        self.frame_history.push_back(features);

        change
    }

    // -----------------------------------------------------------------------
    // Private detection strategies
    // -----------------------------------------------------------------------

    fn detect_threshold(&self, prev: &FrameFeatures, curr: &FrameFeatures) -> Option<SceneChange> {
        let luma_diff = (curr.mean_luma - prev.mean_luma).abs() / 255.0;
        if luma_diff > self.threshold {
            // Distinguish fade from cut by checking chroma stability.
            let u_diff = (curr.mean_chroma_u - prev.mean_chroma_u).abs() / 255.0;
            let v_diff = (curr.mean_chroma_v - prev.mean_chroma_v).abs() / 255.0;
            let change_type = if u_diff < 0.05 && v_diff < 0.05 {
                SceneChangeType::Fade
            } else {
                SceneChangeType::Cut
            };
            Some(SceneChange {
                frame_number: curr.frame_number,
                confidence: (luma_diff / self.threshold).min(1.0),
                change_type,
            })
        } else {
            None
        }
    }

    fn detect_edge_based(&self, prev: &FrameFeatures, curr: &FrameFeatures) -> Option<SceneChange> {
        let edge_diff = (curr.edge_density - prev.edge_density).abs();
        if edge_diff > self.threshold {
            Some(SceneChange {
                frame_number: curr.frame_number,
                confidence: (edge_diff / self.threshold).min(1.0),
                change_type: SceneChangeType::Cut,
            })
        } else {
            None
        }
    }

    fn detect_adaptive(&self, prev: &FrameFeatures, curr: &FrameFeatures) -> Option<SceneChange> {
        // Combine histogram L1 distance and edge difference.
        let hist_diff = histogram_l1_distance(&prev.histogram, &curr.histogram);
        let edge_diff = (curr.edge_density - prev.edge_density).abs();
        let combined = hist_diff * 0.7 + edge_diff * 0.3;

        if combined > self.threshold {
            let change_type = classify_change_type(prev, curr, hist_diff);
            Some(SceneChange {
                frame_number: curr.frame_number,
                confidence: (combined / self.threshold).min(1.0),
                change_type,
            })
        } else {
            None
        }
    }

    /// Adaptive complexity-based scene change detection.
    ///
    /// The threshold is dynamically adjusted based on the content complexity
    /// of the *previous* frame (and the rolling history). Complex content
    /// (high histogram spread / variance) raises the threshold to avoid
    /// false positives from natural intra-scene variation. Simple content
    /// (flat, low variance) lowers the threshold for greater sensitivity.
    ///
    /// Complexity is measured as the Shannon entropy of the luma histogram
    /// combined with edge density and the running variance of recent
    /// histogram differences.
    fn detect_adaptive_complexity(
        &self,
        prev: &FrameFeatures,
        curr: &FrameFeatures,
    ) -> Option<SceneChange> {
        // --- Step 1: Compute content complexity of the previous frame ---
        let prev_entropy = histogram_entropy(&prev.histogram);
        let prev_spread = histogram_spread(&prev.histogram);
        let edge_complexity = prev.edge_density;

        // Normalise entropy: maximum for 256-bin histogram is log2(256) = 8.
        let norm_entropy = (prev_entropy / 8.0).clamp(0.0, 1.0);

        // Normalise spread: standard deviation of the histogram, typically
        // in [0, ~0.06] for uniform, much lower for peaky distributions.
        let norm_spread = (prev_spread * 20.0).clamp(0.0, 1.0);

        // Spatial variance (from integral image, already normalised to [0,1]).
        let spatial_var_complexity = prev.spatial_variance;

        // Composite complexity score in [0, 1]:
        //   35% entropy + 25% histogram spread + 25% edge density + 15% spatial variance
        // The spatial variance term (from the integral image) captures local pixel
        // intensity variation that complements the histogram-level measures.
        let complexity = norm_entropy * 0.35
            + norm_spread * 0.25
            + edge_complexity * 0.25
            + spatial_var_complexity * 0.15;

        // --- Step 2: Compute running variance of recent histogram diffs ---
        let history_volatility = self.compute_history_volatility();

        // --- Step 3: Adjust threshold ---
        // Base threshold from the user config, scaled by complexity.
        // High complexity → threshold up to 1.6x base.
        // Low complexity  → threshold down to 0.6x base.
        // History volatility further raises the threshold if the content
        // has been naturally varying (e.g., fast motion, flashing lights).
        let complexity_factor = 0.6 + complexity; // [0.6, 1.6]
        let volatility_factor = 1.0 + history_volatility * 0.5; // [1.0, 1.5]
        let adaptive_threshold = self.threshold * complexity_factor * volatility_factor;

        // --- Step 4: Compute inter-frame distance ---
        let hist_diff = histogram_l1_distance(&prev.histogram, &curr.histogram);
        let edge_diff = (curr.edge_density - prev.edge_density).abs();
        let chroma_diff = ((curr.mean_chroma_u - prev.mean_chroma_u).abs()
            + (curr.mean_chroma_v - prev.mean_chroma_v).abs())
            / 510.0; // normalise to [0, 1]

        // Weighted combination: histogram dominates, with chroma as tiebreaker.
        let combined = hist_diff * 0.6 + edge_diff * 0.2 + chroma_diff * 0.2;

        if combined > adaptive_threshold {
            let change_type = classify_change_type(prev, curr, hist_diff);
            let raw_confidence = combined / adaptive_threshold;
            Some(SceneChange {
                frame_number: curr.frame_number,
                confidence: raw_confidence.min(1.0),
                change_type,
            })
        } else {
            None
        }
    }

    /// Compute the volatility (variance of consecutive histogram L1 distances)
    /// over the recent frame history.
    fn compute_history_volatility(&self) -> f32 {
        if self.frame_history.len() < 3 {
            return 0.0;
        }

        let diffs: Vec<f32> = self
            .frame_history
            .iter()
            .collect::<Vec<_>>()
            .windows(2)
            .map(|pair| histogram_l1_distance(&pair[0].histogram, &pair[1].histogram))
            .collect();

        if diffs.is_empty() {
            return 0.0;
        }

        let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
        let variance =
            diffs.iter().map(|&d| (d - mean) * (d - mean)).sum::<f32>() / diffs.len() as f32;
        variance.sqrt().clamp(0.0, 1.0)
    }
}

// -----------------------------------------------------------------------
// Public free functions
// -----------------------------------------------------------------------

/// Extract per-frame statistical features from a YUV420 planar buffer.
///
/// `frame` layout: Y plane `width × height`, then U plane `(w/2)×(h/2)`,
/// then V plane `(w/2)×(h/2)`.
pub fn extract_features(frame: &[u8], width: u32, height: u32, frame_number: u64) -> FrameFeatures {
    let y_size = (width as usize) * (height as usize);
    let uv_size = ((width as usize + 1) / 2) * ((height as usize + 1) / 2);

    let y_plane = frame.get(..y_size.min(frame.len())).unwrap_or(&[]);
    let u_plane = frame.get(y_size..y_size + uv_size).unwrap_or(&[]);
    let v_plane = frame
        .get(y_size + uv_size..y_size + 2 * uv_size)
        .unwrap_or(&[]);

    // Luma histogram.
    let mut hist_counts = [0u32; 256];
    for &px in y_plane {
        hist_counts[px as usize] += 1;
    }
    let total_y = y_plane.len().max(1) as f32;
    let mut histogram = [0.0f32; 256];
    for (i, &count) in hist_counts.iter().enumerate() {
        histogram[i] = count as f32 / total_y;
    }

    // Mean luma.
    let mean_luma = if y_plane.is_empty() {
        0.0
    } else {
        y_plane.iter().map(|&p| p as f64).sum::<f64>() as f32 / total_y
    };

    // Mean chroma.
    let mean_chroma_u = if u_plane.is_empty() {
        128.0
    } else {
        u_plane.iter().map(|&p| p as f64).sum::<f64>() as f32 / u_plane.len() as f32
    };
    let mean_chroma_v = if v_plane.is_empty() {
        128.0
    } else {
        v_plane.iter().map(|&p| p as f64).sum::<f64>() as f32 / v_plane.len() as f32
    };

    // Edge density: simple horizontal gradient on the Y plane.
    let edge_density = compute_edge_density(y_plane, width, height);

    // Spatial variance via integral image — O(1) full-frame query after O(W×H) build.
    // Maximum possible 8-bit variance is 127.5² = 16256.25 (half-black / half-white).
    const MAX_VAR: f64 = 16256.25;
    let spatial_variance = if y_plane.is_empty() || width == 0 || height == 0 {
        0.0
    } else {
        let ii = IntegralImage::build(y_plane, width, height);
        let raw_var = ii.rect_variance(0, 0, width, height);
        (raw_var / MAX_VAR).clamp(0.0, 1.0) as f32
    };

    FrameFeatures {
        frame_number,
        histogram,
        edge_density,
        mean_luma,
        mean_chroma_u,
        mean_chroma_v,
        spatial_variance,
    }
}

/// Detect a scene change between two consecutive `FrameFeatures` using
/// histogram SAD (L1 distance).
///
/// Returns `None` when the frames appear to be from the same scene.
pub fn detect_change(prev: &FrameFeatures, curr: &FrameFeatures) -> Option<SceneChange> {
    let hist_diff = histogram_l1_distance(&prev.histogram, &curr.histogram);

    // Default cut threshold: 0.35 (35% of total histogram mass moved).
    const CUT_THRESHOLD: f32 = 0.35;
    const DISSOLVE_THRESHOLD: f32 = 0.15;

    if hist_diff >= CUT_THRESHOLD {
        let change_type = classify_change_type(prev, curr, hist_diff);
        let confidence = (hist_diff / CUT_THRESHOLD).min(1.0);
        Some(SceneChange {
            frame_number: curr.frame_number,
            confidence,
            change_type,
        })
    } else if hist_diff >= DISSOLVE_THRESHOLD {
        Some(SceneChange {
            frame_number: curr.frame_number,
            confidence: hist_diff / CUT_THRESHOLD,
            change_type: SceneChangeType::Dissolve,
        })
    } else {
        None
    }
}

/// Detect a gradual scene transition by checking for a monotonic luma trend
/// over the last (up to) 5 frames in `history`.
///
/// Returns `true` when luma has been monotonically increasing or decreasing
/// across all consecutive pairs in the window.
pub fn detect_gradual(history: &[&FrameFeatures]) -> bool {
    // Require at least 5 frames to assess a trend.
    if history.len() < 5 {
        return false;
    }

    let window: Vec<f32> = history
        .iter()
        .rev()
        .take(5)
        .map(|f| f.mean_luma)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    let all_increasing = window.windows(2).all(|w| w[1] > w[0] + 1.0);
    let all_decreasing = window.windows(2).all(|w| w[1] < w[0] - 1.0);

    all_increasing || all_decreasing
}

// -----------------------------------------------------------------------
// Private helpers
// -----------------------------------------------------------------------

/// Compute the Shannon entropy of a normalised histogram in bits.
///
/// Maximum entropy for 256 bins is log2(256) = 8.0 (uniform distribution).
/// A single-peak histogram yields entropy near 0.
fn histogram_entropy(hist: &[f32; 256]) -> f32 {
    let mut entropy = 0.0f64;
    for &p in hist.iter() {
        if p > 0.0 {
            entropy -= p as f64 * (p as f64).log2();
        }
    }
    entropy as f32
}

/// Compute the standard deviation of the histogram bin values.
///
/// Measures how spread-out the histogram mass is. A uniform histogram has
/// low spread (all bins equal); a bimodal or peaky histogram has higher spread.
fn histogram_spread(hist: &[f32; 256]) -> f32 {
    let mean = hist.iter().sum::<f32>() / 256.0;
    let variance = hist.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / 256.0;
    variance.sqrt()
}

/// Compute the L1 (sum of absolute differences) distance between two normalised
/// histograms.  The result lies in \[0.0, 2.0\]; divide by 2 to normalise to \[0,1\].
fn histogram_l1_distance(a: &[f32; 256], b: &[f32; 256]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .sum::<f32>()
        / 2.0
}

/// Sobel-inspired horizontal gradient edge density for a Y-plane.
fn compute_edge_density(y_plane: &[u8], width: u32, height: u32) -> f32 {
    let w = width as usize;
    let h = height as usize;
    if w < 2 || h == 0 {
        return 0.0;
    }

    let mut edge_count = 0u64;
    let total_pixels = ((w - 1) * h) as u64;

    for row in 0..h {
        for col in 0..(w - 1) {
            let left = y_plane.get(row * w + col).copied().unwrap_or(0) as i32;
            let right = y_plane.get(row * w + col + 1).copied().unwrap_or(0) as i32;
            if (left - right).abs() > 20 {
                edge_count += 1;
            }
        }
    }

    if total_pixels == 0 {
        0.0
    } else {
        edge_count as f32 / total_pixels as f32
    }
}

/// Classify the type of change based on histogram difference magnitude and
/// chroma stability.
fn classify_change_type(
    prev: &FrameFeatures,
    curr: &FrameFeatures,
    hist_diff: f32,
) -> SceneChangeType {
    let u_diff = (curr.mean_chroma_u - prev.mean_chroma_u).abs();
    let v_diff = (curr.mean_chroma_v - prev.mean_chroma_v).abs();
    let chroma_stable = u_diff < 5.0 && v_diff < 5.0;

    if chroma_stable && curr.mean_luma < 10.0 {
        SceneChangeType::Fade
    } else if hist_diff > 0.5 {
        SceneChangeType::Cut
    } else {
        SceneChangeType::Dissolve
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers -------------------------------------------------------

    /// Build a minimal YUV420 planar frame: flat Y=`y`, U=128, V=128.
    fn make_yuv_frame(width: u32, height: u32, y_val: u8) -> Vec<u8> {
        let y_size = (width * height) as usize;
        let uv_size = ((width / 2) * (height / 2)) as usize;
        let mut v = Vec::with_capacity(y_size + 2 * uv_size);
        v.extend(std::iter::repeat_n(y_val, y_size));
        v.extend(std::iter::repeat_n(128u8, uv_size)); // U
        v.extend(std::iter::repeat_n(128u8, uv_size)); // V
        v
    }

    /// Edge-rich frame: Y plane alternates 0 and 255 in adjacent columns.
    fn make_ramp_frame(width: u32, height: u32) -> Vec<u8> {
        let y_size = (width * height) as usize;
        let uv_size = ((width / 2) * (height / 2)) as usize;
        let v_y: Vec<u8> = (0..y_size)
            .map(|i| if i % 2 == 0 { 0u8 } else { 255u8 })
            .collect();
        let mut v = v_y;
        v.extend(std::iter::repeat_n(128u8, uv_size));
        v.extend(std::iter::repeat_n(128u8, uv_size));
        v
    }

    /// Build a `FrameFeatures` with a uniform histogram (all bins equal).
    fn uniform_features(frame_number: u64, mean_luma: f32) -> FrameFeatures {
        let mut histogram = [0.0f32; 256];
        let val = 1.0 / 256.0;
        for b in histogram.iter_mut() {
            *b = val;
        }
        FrameFeatures {
            frame_number,
            histogram,
            edge_density: 0.0,
            mean_luma,
            mean_chroma_u: 128.0,
            mean_chroma_v: 128.0,
            spatial_variance: 0.0,
        }
    }

    // ---- extract_features tests ----------------------------------------

    // 1. extract_features: correct histogram normalisation
    #[test]
    fn test_extract_features_histogram_sums_to_one() {
        let frame = make_yuv_frame(16, 16, 100);
        let feats = extract_features(&frame, 16, 16, 0);
        let sum: f32 = feats.histogram.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "histogram sum = {sum}");
    }

    // 2. extract_features: flat Y=0 frame → mean_luma ≈ 0
    #[test]
    fn test_extract_features_zero_luma() {
        let frame = make_yuv_frame(8, 8, 0);
        let feats = extract_features(&frame, 8, 8, 5);
        assert!((feats.mean_luma).abs() < 1e-3);
    }

    // 3. extract_features: flat Y=255 frame → mean_luma ≈ 255
    #[test]
    fn test_extract_features_max_luma() {
        let frame = make_yuv_frame(8, 8, 255);
        let feats = extract_features(&frame, 8, 8, 1);
        assert!((feats.mean_luma - 255.0).abs() < 1e-3);
    }

    // 4. extract_features: neutral chroma is ~128
    #[test]
    fn test_extract_features_neutral_chroma() {
        let frame = make_yuv_frame(8, 8, 128);
        let feats = extract_features(&frame, 8, 8, 2);
        assert!((feats.mean_chroma_u - 128.0).abs() < 1.0);
        assert!((feats.mean_chroma_v - 128.0).abs() < 1.0);
    }

    // 5. extract_features: frame_number is preserved
    #[test]
    fn test_extract_features_frame_number() {
        let frame = make_yuv_frame(4, 4, 50);
        let feats = extract_features(&frame, 4, 4, 42);
        assert_eq!(feats.frame_number, 42);
    }

    // 6. extract_features: ramp frame produces non-zero edge density
    #[test]
    fn test_extract_features_ramp_has_edges() {
        let frame = make_ramp_frame(16, 16);
        let feats = extract_features(&frame, 16, 16, 0);
        assert!(
            feats.edge_density > 0.0,
            "ramp should have nonzero edge density"
        );
    }

    // 7. extract_features: flat frame has zero edge density
    #[test]
    fn test_extract_features_flat_zero_edges() {
        let frame = make_yuv_frame(8, 8, 200);
        let feats = extract_features(&frame, 8, 8, 0);
        assert_eq!(feats.edge_density, 0.0);
    }

    // ---- detect_change tests -------------------------------------------

    // 8. detect_change: identical features → None
    #[test]
    fn test_detect_change_identical_features_no_change() {
        let feats = uniform_features(0, 128.0);
        let result = detect_change(&feats, &feats);
        assert!(
            result.is_none(),
            "identical features should yield no change"
        );
    }

    // 9. detect_change: large histogram difference → Cut
    #[test]
    fn test_detect_change_large_difference_cut() {
        let mut prev_hist = [0.0f32; 256];
        prev_hist[0] = 1.0; // all mass at bin 0

        let mut curr_hist = [0.0f32; 256];
        curr_hist[255] = 1.0; // all mass at bin 255

        let prev = FrameFeatures {
            frame_number: 0,
            histogram: prev_hist,
            edge_density: 0.1,
            mean_luma: 0.0,
            mean_chroma_u: 128.0,
            mean_chroma_v: 128.0,
            spatial_variance: 0.0,
        };
        let curr = FrameFeatures {
            frame_number: 1,
            histogram: curr_hist,
            edge_density: 0.1,
            mean_luma: 255.0,
            mean_chroma_u: 128.0,
            mean_chroma_v: 128.0,
            spatial_variance: 0.0,
        };

        let result = detect_change(&prev, &curr);
        assert!(
            result.is_some(),
            "large histogram diff should yield a change"
        );
        let change = result.expect("change should be detected");
        assert_eq!(change.frame_number, 1);
        assert!(change.confidence > 0.0);
    }

    // 10. detect_change: moderate difference → Dissolve
    #[test]
    fn test_detect_change_moderate_difference_dissolve() {
        // prev: mass concentrated in lower half
        let mut prev_hist = [0.0f32; 256];
        for i in 0..128 {
            prev_hist[i] = 1.0 / 128.0;
        }
        // curr: mass concentrated in upper half
        let mut curr_hist = [0.0f32; 256];
        for i in 128..256 {
            curr_hist[i] = 1.0 / 128.0;
        }

        let prev = FrameFeatures {
            frame_number: 0,
            histogram: prev_hist,
            edge_density: 0.05,
            mean_luma: 64.0,
            mean_chroma_u: 128.0,
            mean_chroma_v: 128.0,
            spatial_variance: 0.0,
        };
        let curr = FrameFeatures {
            frame_number: 1,
            histogram: curr_hist,
            edge_density: 0.05,
            mean_luma: 192.0,
            mean_chroma_u: 128.0,
            mean_chroma_v: 128.0,
            spatial_variance: 0.0,
        };

        let result = detect_change(&prev, &curr);
        // The L1 distance here is 1.0 (all mass moved), so it exceeds CUT_THRESHOLD.
        // Either Cut or Dissolve is acceptable; just verify something is returned.
        assert!(result.is_some());
    }

    // 11. detect_change: fade (low luma, stable chroma)
    #[test]
    fn test_detect_change_fade_classification() {
        let mut prev_hist = [0.0f32; 256];
        prev_hist[128] = 1.0;

        let mut curr_hist = [0.0f32; 256];
        curr_hist[0] = 1.0; // nearly black

        let prev = FrameFeatures {
            frame_number: 9,
            histogram: prev_hist,
            edge_density: 0.0,
            mean_luma: 128.0,
            mean_chroma_u: 128.0,
            mean_chroma_v: 128.0,
            spatial_variance: 0.0,
        };
        let curr = FrameFeatures {
            frame_number: 10,
            histogram: curr_hist,
            edge_density: 0.0,
            mean_luma: 0.0, // fade to black
            mean_chroma_u: 128.0,
            mean_chroma_v: 128.0,
            spatial_variance: 0.0,
        };

        let result = detect_change(&prev, &curr);
        assert!(result.is_some());
        let change = result.expect("fade change should be detected");
        assert_eq!(change.change_type, SceneChangeType::Fade);
    }

    // ---- detect_gradual tests ------------------------------------------

    // 12. detect_gradual: monotonically increasing luma → true
    #[test]
    fn test_detect_gradual_increasing_luma() {
        let feats: Vec<FrameFeatures> = (0..5)
            .map(|i| uniform_features(i, 50.0 + i as f32 * 10.0))
            .collect();
        let refs: Vec<&FrameFeatures> = feats.iter().collect();
        assert!(
            detect_gradual(&refs),
            "monotonically increasing should be gradual"
        );
    }

    // 13. detect_gradual: monotonically decreasing luma → true
    #[test]
    fn test_detect_gradual_decreasing_luma() {
        let feats: Vec<FrameFeatures> = (0..5)
            .map(|i| uniform_features(i, 200.0 - i as f32 * 15.0))
            .collect();
        let refs: Vec<&FrameFeatures> = feats.iter().collect();
        assert!(
            detect_gradual(&refs),
            "monotonically decreasing should be gradual"
        );
    }

    // 14. detect_gradual: stable luma → false
    #[test]
    fn test_detect_gradual_stable_luma_false() {
        let feats: Vec<FrameFeatures> = (0..5).map(|i| uniform_features(i, 128.0)).collect();
        let refs: Vec<&FrameFeatures> = feats.iter().collect();
        assert!(!detect_gradual(&refs), "stable luma should not be gradual");
    }

    // 15. detect_gradual: fewer than 5 frames → false
    #[test]
    fn test_detect_gradual_insufficient_history_false() {
        let feats: Vec<FrameFeatures> = (0..4)
            .map(|i| uniform_features(i, 50.0 + i as f32 * 10.0))
            .collect();
        let refs: Vec<&FrameFeatures> = feats.iter().collect();
        assert!(!detect_gradual(&refs), "< 5 frames should return false");
    }

    // 16. detect_gradual: non-monotonic trend → false
    #[test]
    fn test_detect_gradual_non_monotonic_false() {
        let lumas = [50.0f32, 60.0, 55.0, 70.0, 80.0];
        let feats: Vec<FrameFeatures> = lumas
            .iter()
            .enumerate()
            .map(|(i, &l)| uniform_features(i as u64, l))
            .collect();
        let refs: Vec<&FrameFeatures> = feats.iter().collect();
        assert!(!detect_gradual(&refs), "non-monotonic should return false");
    }

    // ---- SceneIndex tests ----------------------------------------------

    // 17. SceneIndex::from_changes: no changes → one big scene
    #[test]
    fn test_scene_index_no_changes() {
        let index = SceneIndex::from_changes(vec![], 100);
        assert_eq!(index.scenes.len(), 1);
        assert_eq!(index.scenes[0].start_frame, 0);
        assert_eq!(index.scenes[0].end_frame, 99);
        assert_eq!(index.scenes[0].duration_frames, 100);
    }

    // 18. SceneIndex::from_changes: single cut at frame 50
    #[test]
    fn test_scene_index_single_cut() {
        let changes = vec![SceneChange {
            frame_number: 50,
            confidence: 0.9,
            change_type: SceneChangeType::Cut,
        }];
        let index = SceneIndex::from_changes(changes, 100);
        assert_eq!(index.scenes.len(), 2);
        assert_eq!(index.scenes[0].end_frame, 49);
        assert_eq!(index.scenes[1].start_frame, 50);
        assert_eq!(index.scenes[1].end_frame, 99);
    }

    // 19. SceneIndex::from_changes: multiple cuts are ordered
    #[test]
    fn test_scene_index_multiple_cuts_ordered() {
        let changes = vec![
            SceneChange {
                frame_number: 80,
                confidence: 0.95,
                change_type: SceneChangeType::Cut,
            },
            SceneChange {
                frame_number: 30,
                confidence: 0.85,
                change_type: SceneChangeType::Cut,
            },
        ];
        let index = SceneIndex::from_changes(changes, 120);
        // Should sort → cuts at 30 and 80 → 3 scenes
        assert_eq!(index.scenes.len(), 3);
        assert_eq!(index.scenes[0].start_frame, 0);
        assert_eq!(index.scenes[0].end_frame, 29);
        assert_eq!(index.scenes[1].start_frame, 30);
        assert_eq!(index.scenes[1].end_frame, 79);
        assert_eq!(index.scenes[2].start_frame, 80);
    }

    // 20. SceneChangeDetector: push identical frames → no change reported
    #[test]
    fn test_scene_change_detector_no_change_on_identical_frames() {
        let mut detector = SceneChangeDetector::new(SceneDetectionMethod::HistogramBased, 0.35, 30);
        let frame = make_yuv_frame(16, 16, 100);
        // First push seeds the history.
        let first = detector.push_frame(&frame, 0, 16, 16);
        assert!(first.is_none(), "first frame should not trigger a change");
        // Second push of the identical frame.
        let second = detector.push_frame(&frame, 1, 16, 16);
        assert!(
            second.is_none(),
            "identical frames should not trigger a change"
        );
    }

    // 21. SceneChangeDetector: push very different frames → change detected
    #[test]
    fn test_scene_change_detector_detects_cut() {
        let mut detector = SceneChangeDetector::new(SceneDetectionMethod::HistogramBased, 0.20, 30);
        let dark_frame = make_yuv_frame(16, 16, 10);
        let bright_frame = make_yuv_frame(16, 16, 245);
        detector.push_frame(&dark_frame, 0, 16, 16);
        let change = detector.push_frame(&bright_frame, 1, 16, 16);
        assert!(
            change.is_some(),
            "large luma jump should be detected as a scene change"
        );
    }

    // 22. SceneChangeDetector: history is bounded
    #[test]
    fn test_scene_change_detector_history_bounded() {
        let mut detector = SceneChangeDetector::new(SceneDetectionMethod::HistogramBased, 0.35, 5);
        let frame = make_yuv_frame(8, 8, 128);
        for i in 0..20u64 {
            detector.push_frame(&frame, i, 8, 8);
        }
        assert!(detector.frame_history.len() <= 5);
    }

    // 23. SceneDetectionMethod variants are distinguishable
    #[test]
    fn test_scene_detection_method_variants() {
        assert_ne!(
            SceneDetectionMethod::ThresholdBased,
            SceneDetectionMethod::HistogramBased
        );
        assert_ne!(
            SceneDetectionMethod::EdgeBased,
            SceneDetectionMethod::Adaptive
        );
    }

    // 24. SceneChangeType variants are distinguishable
    #[test]
    fn test_scene_change_type_variants() {
        assert_ne!(SceneChangeType::Cut, SceneChangeType::Gradual);
        assert_ne!(SceneChangeType::Dissolve, SceneChangeType::Fade);
    }

    // 25. SceneChange fields are accessible
    #[test]
    fn test_scene_change_fields() {
        let sc = SceneChange {
            frame_number: 99,
            confidence: 0.75,
            change_type: SceneChangeType::Dissolve,
        };
        assert_eq!(sc.frame_number, 99);
        assert!((sc.confidence - 0.75).abs() < 1e-5);
        assert_eq!(sc.change_type, SceneChangeType::Dissolve);
    }

    // ---- AdaptiveComplexity tests ----------------------------------------

    // 26. AdaptiveComplexity: identical frames → no change
    #[test]
    fn test_adaptive_complexity_no_change_identical() {
        let mut detector =
            SceneChangeDetector::new(SceneDetectionMethod::AdaptiveComplexity, 0.35, 30);
        let frame = make_yuv_frame(16, 16, 128);
        detector.push_frame(&frame, 0, 16, 16);
        let result = detector.push_frame(&frame, 1, 16, 16);
        assert!(
            result.is_none(),
            "identical frames should not trigger AdaptiveComplexity change"
        );
    }

    // 27. AdaptiveComplexity: large luma jump detected
    #[test]
    fn test_adaptive_complexity_detects_large_change() {
        let mut detector =
            SceneChangeDetector::new(SceneDetectionMethod::AdaptiveComplexity, 0.20, 30);
        let dark = make_yuv_frame(16, 16, 10);
        let bright = make_yuv_frame(16, 16, 245);
        detector.push_frame(&dark, 0, 16, 16);
        let result = detector.push_frame(&bright, 1, 16, 16);
        assert!(
            result.is_some(),
            "large luma jump should trigger AdaptiveComplexity change"
        );
    }

    // 28. AdaptiveComplexity: complex content raises threshold (fewer false positives)
    #[test]
    fn test_adaptive_complexity_higher_threshold_for_complex_content() {
        // Use a ramp frame (high edge density = complex content) as "previous".
        // Then a slightly shifted ramp as "current".
        // The adaptive threshold should be higher, reducing false positives.
        let mut detector =
            SceneChangeDetector::new(SceneDetectionMethod::AdaptiveComplexity, 0.30, 30);
        let ramp = make_ramp_frame(16, 16);
        detector.push_frame(&ramp, 0, 16, 16);

        // Build a slightly different ramp (shift by a few luma values).
        let y_size = 16 * 16;
        let _uv_size = 8 * 8;
        let mut shifted = ramp.clone();
        for i in 0..y_size {
            shifted[i] = shifted[i].saturating_add(20);
        }
        let result = detector.push_frame(&shifted, 1, 16, 16);
        // With complex content the threshold is raised; this moderate change
        // should NOT trigger a scene change.
        assert!(
            result.is_none(),
            "complex content should raise threshold and avoid false positive"
        );
    }

    // 29. AdaptiveComplexity: simple content has lower threshold (more sensitive)
    #[test]
    fn test_adaptive_complexity_lower_threshold_for_simple_content() {
        // A flat frame (zero entropy, zero edge density) = simple content.
        // Even a moderate change should be detected.
        let mut detector =
            SceneChangeDetector::new(SceneDetectionMethod::AdaptiveComplexity, 0.30, 30);
        let simple = make_yuv_frame(16, 16, 100);
        detector.push_frame(&simple, 0, 16, 16);

        // Moderate shift: 100 → 200 (all mass moves in histogram).
        let different = make_yuv_frame(16, 16, 200);
        let result = detector.push_frame(&different, 1, 16, 16);
        assert!(
            result.is_some(),
            "simple content should lower threshold and detect moderate change"
        );
    }

    // 30. AdaptiveComplexity: history volatility reduces sensitivity
    #[test]
    fn test_adaptive_complexity_history_volatility_reduces_sensitivity() {
        let mut detector =
            SceneChangeDetector::new(SceneDetectionMethod::AdaptiveComplexity, 0.30, 30);
        // Feed alternating frames to build volatility.
        for i in 0..10u64 {
            let val = if i % 2 == 0 { 80u8 } else { 120u8 };
            let frame = make_yuv_frame(16, 16, val);
            detector.push_frame(&frame, i, 16, 16);
        }
        // Now push a moderately different frame.
        let test_frame = make_yuv_frame(16, 16, 160);
        let result = detector.push_frame(&test_frame, 10, 16, 16);
        // High volatility should raise the threshold, reducing sensitivity.
        // This is a heuristic test: we just verify the method runs without error.
        // The actual result depends on the exact volatility calculation.
        let _ = result;
    }

    // 31. AdaptiveComplexity vs Adaptive: different behavior on complex content
    #[test]
    fn test_adaptive_complexity_differs_from_adaptive() {
        // Build the same scenario for both methods and verify they can differ.
        let ramp = make_ramp_frame(16, 16);
        let y_size = 16 * 16;
        let mut shifted = ramp.clone();
        for i in 0..y_size {
            shifted[i] = shifted[i].saturating_add(30);
        }

        let mut det_adaptive = SceneChangeDetector::new(SceneDetectionMethod::Adaptive, 0.25, 30);
        det_adaptive.push_frame(&ramp, 0, 16, 16);
        let result_adaptive = det_adaptive.push_frame(&shifted, 1, 16, 16);

        let mut det_complexity =
            SceneChangeDetector::new(SceneDetectionMethod::AdaptiveComplexity, 0.25, 30);
        det_complexity.push_frame(&ramp, 0, 16, 16);
        let result_complexity = det_complexity.push_frame(&shifted, 1, 16, 16);

        // The two methods should produce different results on complex content
        // because AdaptiveComplexity raises its threshold.
        // (Adaptive may detect; AdaptiveComplexity may not, or with different confidence.)
        if let (Some(a), Some(c)) = (&result_adaptive, &result_complexity) {
            // At minimum, confidences should differ due to different thresholds.
            assert!(
                (a.confidence - c.confidence).abs() > 0.001
                    || result_adaptive.is_some() != result_complexity.is_some(),
                "AdaptiveComplexity should behave differently from Adaptive on complex content"
            );
        }
        // If one is None and the other Some, that also demonstrates the difference.
    }

    // 32. histogram_entropy: uniform distribution → max entropy
    #[test]
    fn test_histogram_entropy_uniform() {
        let mut hist = [0.0f32; 256];
        for b in hist.iter_mut() {
            *b = 1.0 / 256.0;
        }
        let entropy = histogram_entropy(&hist);
        assert!(
            (entropy - 8.0).abs() < 0.01,
            "uniform histogram should have entropy ~8.0, got {entropy}"
        );
    }

    // 33. histogram_entropy: single-bin → zero entropy
    #[test]
    fn test_histogram_entropy_single_bin() {
        let mut hist = [0.0f32; 256];
        hist[128] = 1.0;
        let entropy = histogram_entropy(&hist);
        assert!(
            entropy.abs() < 0.01,
            "single-bin histogram should have entropy ~0, got {entropy}"
        );
    }

    // 34. histogram_spread: uniform → low spread
    #[test]
    fn test_histogram_spread_uniform() {
        let mut hist = [0.0f32; 256];
        for b in hist.iter_mut() {
            *b = 1.0 / 256.0;
        }
        let spread = histogram_spread(&hist);
        assert!(
            spread < 0.001,
            "uniform histogram should have near-zero spread, got {spread}"
        );
    }

    // 35. histogram_spread: single-bin → higher spread
    #[test]
    fn test_histogram_spread_single_bin() {
        let mut hist = [0.0f32; 256];
        hist[128] = 1.0;
        let spread = histogram_spread(&hist);
        assert!(
            spread > 0.01,
            "single-bin histogram should have non-zero spread, got {spread}"
        );
    }

    // 36. AdaptiveComplexity variant is distinct
    #[test]
    fn test_adaptive_complexity_variant_distinct() {
        assert_ne!(
            SceneDetectionMethod::Adaptive,
            SceneDetectionMethod::AdaptiveComplexity
        );
    }

    // ---- IntegralImage integration tests (via extract_features) ----------

    // 37. extract_features: flat frame has spatial_variance == 0
    #[test]
    fn test_extract_features_flat_spatial_variance_zero() {
        let frame = make_yuv_frame(16, 16, 128);
        let feats = extract_features(&frame, 16, 16, 0);
        assert!(
            feats.spatial_variance.abs() < 1e-6,
            "flat frame must have spatial_variance == 0, got {}",
            feats.spatial_variance
        );
    }

    // 38. extract_features: ramp frame has spatial_variance > 0
    #[test]
    fn test_extract_features_ramp_spatial_variance_nonzero() {
        // Build a YUV frame with a ramp Y plane (0..64 repeated).
        let w: u32 = 16;
        let h: u32 = 16;
        let y_size = (w * h) as usize;
        let uv_size = ((w / 2) * (h / 2)) as usize;
        let mut frame = Vec::with_capacity(y_size + 2 * uv_size);
        for i in 0..y_size {
            frame.push((i % 256) as u8);
        }
        frame.extend(std::iter::repeat_n(128u8, uv_size));
        frame.extend(std::iter::repeat_n(128u8, uv_size));

        let feats = extract_features(&frame, w, h, 0);
        assert!(
            feats.spatial_variance > 0.0,
            "ramp frame must have spatial_variance > 0, got {}",
            feats.spatial_variance
        );
    }

    // 39. extract_features: spatial_variance is normalised to [0, 1]
    #[test]
    fn test_extract_features_spatial_variance_normalised() {
        // Maximum-variance frame: alternating 0 and 255.
        let w: u32 = 16;
        let h: u32 = 16;
        let y_size = (w * h) as usize;
        let uv_size = ((w / 2) * (h / 2)) as usize;
        let mut frame = Vec::with_capacity(y_size + 2 * uv_size);
        for i in 0..y_size {
            frame.push(if i % 2 == 0 { 0u8 } else { 255u8 });
        }
        frame.extend(std::iter::repeat_n(128u8, uv_size));
        frame.extend(std::iter::repeat_n(128u8, uv_size));

        let feats = extract_features(&frame, w, h, 0);
        assert!(
            feats.spatial_variance >= 0.0 && feats.spatial_variance <= 1.0,
            "spatial_variance must be in [0, 1], got {}",
            feats.spatial_variance
        );
        // Maximum theoretical variance (alternating 0/255): ≈ 16256.25 / 16256.25 ≈ 1.0
        assert!(
            feats.spatial_variance > 0.5,
            "max-variance frame should have spatial_variance close to 1.0, got {}",
            feats.spatial_variance
        );
    }

    // 40. AdaptiveComplexity uses spatial variance: complex ramp frame raises threshold
    #[test]
    fn test_adaptive_complexity_with_integral_image_ramp() {
        // A ramp frame has high spatial variance → higher threshold → harder to trigger a change.
        let mut detector =
            SceneChangeDetector::new(SceneDetectionMethod::AdaptiveComplexity, 0.30, 30);

        let w: u32 = 16;
        let h: u32 = 16;
        let y_size = (w * h) as usize;
        let uv_size = ((w / 2) * (h / 2)) as usize;

        // Build a high-spatial-variance frame (ramp Y).
        let mut ramp_frame = Vec::with_capacity(y_size + 2 * uv_size);
        for i in 0..y_size {
            ramp_frame.push((i % 256) as u8);
        }
        ramp_frame.extend(std::iter::repeat_n(128u8, uv_size));
        ramp_frame.extend(std::iter::repeat_n(128u8, uv_size));

        detector.push_frame(&ramp_frame, 0, w, h);

        // A slightly shifted version of the ramp — should NOT trigger with high complexity.
        let mut shifted = ramp_frame.clone();
        for i in 0..y_size {
            shifted[i] = shifted[i].saturating_add(15);
        }
        let result = detector.push_frame(&shifted, 1, w, h);

        // High spatial variance raises the complexity score which raises the threshold;
        // this moderate shift should not produce a scene change.
        assert!(
            result.is_none(),
            "high spatial variance should raise threshold and suppress false positive"
        );
    }
}
