//! Pre-defined neural network models for common media processing tasks.
//!
//! All models are zero-initialised at construction and can be populated with
//! pre-trained weights by directly assigning the public layer fields.  The
//! implementations are designed to be minimal and dependency-free — they are
//! pure-Rust inference pipelines, not training engines.

use crate::activations::{relu, sigmoid, softmax};
use crate::error::NeuralError;
use crate::layers::{Conv2dLayer, LinearLayer};
use crate::tensor::Tensor;

// ──────────────────────────────────────────────────────────────────────────────
// Scene classification
// ──────────────────────────────────────────────────────────────────────────────

/// The ten scene categories recognised by [`SceneClassifier`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneClass {
    /// Mostly static, low-motion footage.
    Static = 0,
    /// High-motion action sequences.
    Action = 1,
    /// Dialogue-heavy content with faces and talking.
    Talking = 2,
    /// Outdoor nature / landscape footage.
    Nature = 3,
    /// Sports content with fast movement and crowds.
    Sports = 4,
    /// Live performance / concert footage.
    Concert = 5,
    /// News broadcast, anchor-style delivery.
    News = 6,
    /// Animated content (2-D or 3-D).
    Animation = 7,
    /// Documentary narration with b-roll.
    Documentary = 8,
    /// Unrecognised / default category.
    Unknown = 9,
}

impl SceneClass {
    /// Converts a raw class index (0-9) to a `SceneClass` variant.
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Static,
            1 => Self::Action,
            2 => Self::Talking,
            3 => Self::Nature,
            4 => Self::Sports,
            5 => Self::Concert,
            6 => Self::News,
            7 => Self::Animation,
            8 => Self::Documentary,
            _ => Self::Unknown,
        }
    }

    /// Returns the number of named (non-Unknown) scene classes.
    pub const fn num_classes() -> usize {
        9
    }
}

/// Lightweight two-layer MLP scene classifier.
///
/// Architecture: `input(128) → Linear(128→64) → ReLU → Linear(64→10) → Softmax`.
pub struct SceneClassifier {
    /// First hidden layer.
    pub hidden: LinearLayer,
    /// Output layer.
    pub output: LinearLayer,
}

impl SceneClassifier {
    /// Expected input feature dimensionality.
    pub const INPUT_DIM: usize = 128;
    /// Hidden layer width.
    pub const HIDDEN_DIM: usize = 64;
    /// Number of output classes.
    pub const NUM_CLASSES: usize = 10;

    /// Creates a zero-initialised `SceneClassifier`.
    pub fn new() -> Result<Self, NeuralError> {
        let hidden = LinearLayer::new(Self::INPUT_DIM, Self::HIDDEN_DIM)?;
        let output = LinearLayer::new(Self::HIDDEN_DIM, Self::NUM_CLASSES)?;
        Ok(Self { hidden, output })
    }

    /// Classifies a 128-dimensional feature vector.
    ///
    /// Returns `(class_index, confidence)` where confidence is the softmax
    /// probability of the predicted class.
    pub fn classify(&self, features: &[f32]) -> Result<(usize, f32), NeuralError> {
        if features.is_empty() {
            return Err(NeuralError::EmptyInput(
                "SceneClassifier::classify: empty features".to_string(),
            ));
        }
        if features.len() != Self::INPUT_DIM {
            return Err(NeuralError::ShapeMismatch(format!(
                "SceneClassifier::classify: expected {} features, got {}",
                Self::INPUT_DIM,
                features.len()
            )));
        }
        let input = Tensor::from_data(features.to_vec(), vec![Self::INPUT_DIM])?;
        let h = self.hidden.forward(&input)?;
        // ReLU activation on hidden layer.
        let h_relu_data: Vec<f32> = h.data().iter().map(|&x| relu(x)).collect();
        let h_relu = Tensor::from_data(h_relu_data, vec![Self::HIDDEN_DIM])?;
        let logits = self.output.forward(&h_relu)?;
        let probs = softmax(logits.data());
        // Find argmax.
        let mut best_idx = 0usize;
        let mut best_prob = f32::NEG_INFINITY;
        for (i, &p) in probs.iter().enumerate() {
            if p > best_prob {
                best_prob = p;
                best_idx = i;
            }
        }
        Ok((best_idx, best_prob))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Thumbnail ranking
// ──────────────────────────────────────────────────────────────────────────────

/// Linear aesthetic-quality ranker for video thumbnails.
///
/// Scores a feature vector in [0, 1], where higher means more aesthetically
/// appealing.  The model is a single affine transform followed by a sigmoid.
pub struct ThumbnailRanker {
    /// Linear projection layer: `[INPUT_DIM → 1]`.
    pub layer: LinearLayer,
}

impl ThumbnailRanker {
    /// Expected input feature dimensionality.
    pub const INPUT_DIM: usize = 64;

    /// Creates a zero-initialised `ThumbnailRanker`.
    pub fn new() -> Result<Self, NeuralError> {
        let layer = LinearLayer::new(Self::INPUT_DIM, 1)?;
        Ok(Self { layer })
    }

    /// Scores a 64-dimensional thumbnail feature vector in [0, 1].
    pub fn score(&self, thumbnail_features: &[f32]) -> Result<f32, NeuralError> {
        if thumbnail_features.is_empty() {
            return Err(NeuralError::EmptyInput(
                "ThumbnailRanker::score: empty features".to_string(),
            ));
        }
        if thumbnail_features.len() != Self::INPUT_DIM {
            return Err(NeuralError::ShapeMismatch(format!(
                "ThumbnailRanker::score: expected {} features, got {}",
                Self::INPUT_DIM,
                thumbnail_features.len()
            )));
        }
        let input = Tensor::from_data(thumbnail_features.to_vec(), vec![Self::INPUT_DIM])?;
        let out = self.layer.forward(&input)?;
        // Single scalar sigmoid.
        Ok(sigmoid(out.data()[0]))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Super-resolution upscaler
// ──────────────────────────────────────────────────────────────────────────────

/// Simplified 2× super-resolution upscaler.
///
/// Pipeline:
/// 1. Bicubic-guided 2× upsampling (bilinear approximation for speed).
/// 2. Three sequential 3×3 convolution passes for detail sharpening.
///
/// The model is single-channel (luminance); for multi-channel images, apply
/// per-channel.
pub struct SrUpscaler {
    /// First sharpening conv (1→8 channels).
    pub conv1: Conv2dLayer,
    /// Second sharpening conv (8→8 channels).
    pub conv2: Conv2dLayer,
    /// Third reconstruction conv (8→1 channel).
    pub conv3: Conv2dLayer,
}

impl SrUpscaler {
    /// Creates a zero-initialised `SrUpscaler`.
    pub fn new() -> Result<Self, NeuralError> {
        // All convolutions use 3×3 kernels with stride 1 and padding 1 to
        // preserve the spatial dimensions.
        let conv1 = Conv2dLayer::new(1, 8, 3, 3, (1, 1), (1, 1))?;
        let conv2 = Conv2dLayer::new(8, 8, 3, 3, (1, 1), (1, 1))?;
        let conv3 = Conv2dLayer::new(8, 1, 3, 3, (1, 1), (1, 1))?;
        Ok(Self {
            conv1,
            conv2,
            conv3,
        })
    }

    /// Upscales a single-channel frame by 2×.
    ///
    /// `frame` is a row-major slice of `height × width` f32 luminance values.
    /// Returns a row-major `(2*height) × (2*width)` slice.
    pub fn upscale_2x(
        &self,
        frame: &[f32],
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>, NeuralError> {
        if frame.is_empty() {
            return Err(NeuralError::EmptyInput(
                "SrUpscaler::upscale_2x: empty frame".to_string(),
            ));
        }
        if frame.len() != height * width {
            return Err(NeuralError::ShapeMismatch(format!(
                "SrUpscaler::upscale_2x: frame length {} != height*width {}",
                frame.len(),
                height * width
            )));
        }
        if height == 0 || width == 0 {
            return Err(NeuralError::InvalidShape(
                "SrUpscaler::upscale_2x: height and width must be > 0".to_string(),
            ));
        }

        let out_h = height * 2;
        let out_w = width * 2;

        // Step 1: bilinear 2× upsampling (approximates bicubic for speed).
        let upsampled = bilinear_upsample_2x(frame, height, width);

        // Step 2: apply sharpening convolution pipeline.
        let t = Tensor::from_data(upsampled, vec![1, out_h, out_w])?;
        let t1 = self.conv1.forward(&t)?;
        // ReLU between layers.
        let t1_act = apply_relu_tensor(&t1);
        let t2 = self.conv2.forward(&t1_act)?;
        let t2_act = apply_relu_tensor(&t2);
        let t3 = self.conv3.forward(&t2_act)?;

        Ok(t3.data().to_vec())
    }
}

/// Bilinear 2× upsampling of a single-channel image.
fn bilinear_upsample_2x(src: &[f32], h: usize, w: usize) -> Vec<f32> {
    let out_h = h * 2;
    let out_w = w * 2;
    let mut dst = vec![0.0_f32; out_h * out_w];

    let clamp = |v: isize, max: usize| -> usize {
        if v < 0 {
            0
        } else if v as usize >= max {
            max - 1
        } else {
            v as usize
        }
    };

    for oy in 0..out_h {
        for ox in 0..out_w {
            // Map output pixel centre to input coordinate.
            let src_y = (oy as f32 + 0.5) / 2.0 - 0.5;
            let src_x = (ox as f32 + 0.5) / 2.0 - 0.5;

            let y0 = src_y.floor() as isize;
            let x0 = src_x.floor() as isize;
            let y1 = y0 + 1;
            let x1 = x0 + 1;

            let fy = src_y - src_y.floor();
            let fx = src_x - src_x.floor();

            let y0c = clamp(y0, h);
            let y1c = clamp(y1, h);
            let x0c = clamp(x0, w);
            let x1c = clamp(x1, w);

            let v00 = src[y0c * w + x0c];
            let v01 = src[y0c * w + x1c];
            let v10 = src[y1c * w + x0c];
            let v11 = src[y1c * w + x1c];

            dst[oy * out_w + ox] = v00 * (1.0 - fy) * (1.0 - fx)
                + v01 * (1.0 - fy) * fx
                + v10 * fy * (1.0 - fx)
                + v11 * fy * fx;
        }
    }
    dst
}

/// Applies ReLU element-wise to a tensor, returning a new tensor.
fn apply_relu_tensor(t: &Tensor) -> Tensor {
    let new_data: Vec<f32> = t.data().iter().map(|&x| relu(x)).collect();
    Tensor::from_data(new_data, t.shape().to_vec())
        .unwrap_or_else(|_| unreachable!("apply_relu_tensor: internal invariant violated"))
}

// ──────────────────────────────────────────────────────────────────────────────
// Feature extractor
// ──────────────────────────────────────────────────────────────────────────────

/// Extracts a 128-dimensional feature vector from an image frame using
/// HOG-like gradient histogram descriptors.
///
/// The image is divided into a 4×4 grid of blocks.  Within each block,
/// an 8-bin gradient histogram is computed, giving 4×4×8 = 128 features.
pub struct FeatureExtractor;

impl FeatureExtractor {
    /// Output feature dimensionality.
    pub const FEATURE_DIM: usize = 128;

    /// Creates a `FeatureExtractor`.
    pub fn new() -> Self {
        Self
    }

    /// Extracts a 128-dimensional feature vector from a single-channel image.
    ///
    /// `frame` is a row-major f32 slice of shape `height × width`.
    /// Values should be normalised to [0, 1].
    pub fn extract(
        &self,
        frame: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, NeuralError> {
        if frame.is_empty() {
            return Err(NeuralError::EmptyInput(
                "FeatureExtractor::extract: empty frame".to_string(),
            ));
        }
        if frame.len() != height * width {
            return Err(NeuralError::ShapeMismatch(format!(
                "FeatureExtractor::extract: frame length {} != height*width {}",
                frame.len(),
                height * width
            )));
        }
        if height < 4 || width < 4 {
            return Err(NeuralError::InvalidShape(
                "FeatureExtractor::extract: image must be at least 4×4".to_string(),
            ));
        }

        // Divide image into a 4×4 grid of blocks.
        const GRID: usize = 4;
        const BINS: usize = 8;

        let block_h = height / GRID;
        let block_w = width / GRID;

        let mut features = Vec::with_capacity(Self::FEATURE_DIM);

        // Compute pixel gradients.
        let grad_mag = compute_gradient_magnitudes(frame, width, height);
        let grad_ang = compute_gradient_angles(frame, width, height);

        for by in 0..GRID {
            for bx in 0..GRID {
                let mut histogram = [0.0_f32; BINS];
                let y_start = by * block_h;
                let x_start = bx * block_w;

                for y in y_start..(y_start + block_h).min(height) {
                    for x in x_start..(x_start + block_w).min(width) {
                        let mag = grad_mag[y * width + x];
                        let ang = grad_ang[y * width + x]; // [0, π)
                                                           // Map angle to bin index.
                        let bin_f = ang / std::f32::consts::PI * BINS as f32;
                        let bin = (bin_f as usize).min(BINS - 1);
                        histogram[bin] += mag;
                    }
                }

                // L2-normalise the histogram.
                let norm_sq: f32 = histogram.iter().map(|&v| v * v).sum();
                let norm = (norm_sq + 1e-6).sqrt();
                for &h_val in histogram.iter() {
                    features.push(h_val / norm);
                }
            }
        }

        Ok(features)
    }
}

/// Computes the gradient magnitude at each pixel using central differences.
fn compute_gradient_magnitudes(frame: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut mag = vec![0.0_f32; height * width];
    for y in 0..height {
        for x in 0..width {
            let gx = if x == 0 {
                frame[y * width + x + 1] - frame[y * width + x]
            } else if x == width - 1 {
                frame[y * width + x] - frame[y * width + x - 1]
            } else {
                (frame[y * width + x + 1] - frame[y * width + x - 1]) * 0.5
            };
            let gy = if y == 0 {
                frame[(y + 1) * width + x] - frame[y * width + x]
            } else if y == height - 1 {
                frame[y * width + x] - frame[(y - 1) * width + x]
            } else {
                (frame[(y + 1) * width + x] - frame[(y - 1) * width + x]) * 0.5
            };
            mag[y * width + x] = (gx * gx + gy * gy).sqrt();
        }
    }
    mag
}

/// Computes the gradient angle (in radians, [0, π)) at each pixel.
fn compute_gradient_angles(frame: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut ang = vec![0.0_f32; height * width];
    for y in 0..height {
        for x in 0..width {
            let gx = if x == 0 {
                frame[y * width + x + 1] - frame[y * width + x]
            } else if x == width - 1 {
                frame[y * width + x] - frame[y * width + x - 1]
            } else {
                (frame[y * width + x + 1] - frame[y * width + x - 1]) * 0.5
            };
            let gy = if y == 0 {
                frame[(y + 1) * width + x] - frame[y * width + x]
            } else if y == height - 1 {
                frame[y * width + x] - frame[(y - 1) * width + x]
            } else {
                (frame[(y + 1) * width + x] - frame[(y - 1) * width + x]) * 0.5
            };
            let mut angle = gy.atan2(gx);
            // Map to [0, π).
            if angle < 0.0 {
                angle += std::f32::consts::PI;
            }
            ang[y * width + x] = angle;
        }
    }
    ang
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    // ── SceneClass ────────────────────────────────────────────────────────────

    #[test]
    fn test_scene_class_from_index_all() {
        for i in 0..SceneClass::num_classes() {
            let cls = SceneClass::from_index(i);
            assert_ne!(cls, SceneClass::Unknown);
        }
    }

    #[test]
    fn test_scene_class_unknown_for_out_of_range() {
        assert_eq!(SceneClass::from_index(99), SceneClass::Unknown);
    }

    #[test]
    fn test_scene_class_num_classes() {
        assert_eq!(SceneClass::num_classes(), 9);
    }

    // ── SceneClassifier ───────────────────────────────────────────────────────

    #[test]
    fn test_scene_classifier_zero_weights_uniform_softmax() {
        let clf = SceneClassifier::new().expect("scene classifier new");
        let features = vec![0.0_f32; SceneClassifier::INPUT_DIM];
        let (idx, conf) = clf.classify(&features).expect("classify");
        // With all-zero weights, all logits are zero → uniform softmax → conf ≈ 0.1
        assert!(close(conf, 0.1));
        assert!(idx < SceneClassifier::NUM_CLASSES);
    }

    #[test]
    fn test_scene_classifier_wrong_dim() {
        let clf = SceneClassifier::new().expect("scene classifier new");
        let features = vec![0.0_f32; 64];
        assert!(clf.classify(&features).is_err());
    }

    #[test]
    fn test_scene_classifier_empty_input() {
        let clf = SceneClassifier::new().expect("scene classifier new");
        assert!(clf.classify(&[]).is_err());
    }

    #[test]
    fn test_scene_classifier_returns_valid_class() {
        let clf = SceneClassifier::new().expect("scene classifier new");
        let features = vec![1.0_f32; SceneClassifier::INPUT_DIM];
        let (idx, conf) = clf.classify(&features).expect("classify");
        assert!(idx < 10);
        assert!((0.0..=1.0).contains(&conf));
    }

    // ── ThumbnailRanker ───────────────────────────────────────────────────────

    #[test]
    fn test_thumbnail_ranker_zero_weights_score_half() {
        let ranker = ThumbnailRanker::new().expect("thumbnail ranker new");
        let features = vec![0.0_f32; ThumbnailRanker::INPUT_DIM];
        let score = ranker.score(&features).expect("score");
        // sigmoid(0) = 0.5
        assert!(close(score, 0.5));
    }

    #[test]
    fn test_thumbnail_ranker_score_in_range() {
        let ranker = ThumbnailRanker::new().expect("thumbnail ranker new");
        let features = vec![1.0_f32; ThumbnailRanker::INPUT_DIM];
        let score = ranker.score(&features).expect("score");
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_thumbnail_ranker_wrong_dim() {
        let ranker = ThumbnailRanker::new().expect("thumbnail ranker new");
        let features = vec![0.0_f32; 32];
        assert!(ranker.score(&features).is_err());
    }

    #[test]
    fn test_thumbnail_ranker_empty_input() {
        let ranker = ThumbnailRanker::new().expect("thumbnail ranker new");
        assert!(ranker.score(&[]).is_err());
    }

    // ── SrUpscaler ────────────────────────────────────────────────────────────

    #[test]
    fn test_sr_upscaler_output_size() {
        let upscaler = SrUpscaler::new().expect("sr upscaler new");
        let frame = vec![0.5_f32; 8 * 8];
        let out = upscaler.upscale_2x(&frame, 8, 8).expect("upscale_2x");
        // Output should be 2× both dimensions.
        assert_eq!(out.len(), 16 * 16);
    }

    #[test]
    fn test_sr_upscaler_empty_frame_error() {
        let upscaler = SrUpscaler::new().expect("sr upscaler new");
        assert!(upscaler.upscale_2x(&[], 0, 0).is_err());
    }

    #[test]
    fn test_sr_upscaler_size_mismatch_error() {
        let upscaler = SrUpscaler::new().expect("sr upscaler new");
        let frame = vec![0.0_f32; 10];
        assert!(upscaler.upscale_2x(&frame, 4, 4).is_err());
    }

    #[test]
    fn test_sr_upscaler_produces_finite_output() {
        let upscaler = SrUpscaler::new().expect("sr upscaler new");
        let frame: Vec<f32> = (0..16).map(|i| i as f32 / 16.0).collect();
        let out = upscaler.upscale_2x(&frame, 4, 4).expect("upscale_2x");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_bilinear_upsample_preserves_corners() {
        let src = vec![1.0_f32, 2.0, 3.0, 4.0];
        let dst = bilinear_upsample_2x(&src, 2, 2);
        assert_eq!(dst.len(), 4 * 4);
        // All values should be in the convex hull [1, 4].
        assert!(dst.iter().all(|&v| v >= 1.0 - 1e-4 && v <= 4.0 + 1e-4));
    }

    // ── FeatureExtractor ──────────────────────────────────────────────────────

    #[test]
    fn test_feature_extractor_output_dim() {
        let extractor = FeatureExtractor::new();
        let frame = vec![0.5_f32; 32 * 32];
        let features = extractor.extract(&frame, 32, 32).expect("extract");
        assert_eq!(features.len(), FeatureExtractor::FEATURE_DIM);
    }

    #[test]
    fn test_feature_extractor_empty_error() {
        let extractor = FeatureExtractor::new();
        assert!(extractor.extract(&[], 0, 0).is_err());
    }

    #[test]
    fn test_feature_extractor_too_small_error() {
        let extractor = FeatureExtractor::new();
        let frame = vec![0.0_f32; 3 * 3];
        assert!(extractor.extract(&frame, 3, 3).is_err());
    }

    #[test]
    fn test_feature_extractor_values_finite() {
        let extractor = FeatureExtractor::new();
        let frame: Vec<f32> = (0..64 * 64).map(|i| (i as f32 % 255.0) / 255.0).collect();
        let features = extractor.extract(&frame, 64, 64).expect("extract");
        assert!(features.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_feature_extractor_zero_image_still_runs() {
        let extractor = FeatureExtractor::new();
        let frame = vec![0.0_f32; 16 * 16];
        // Should succeed (gradients are all zero → histogram is zero / normalised to zero).
        let features = extractor.extract(&frame, 16, 16).expect("extract");
        assert_eq!(features.len(), FeatureExtractor::FEATURE_DIM);
    }

    #[test]
    fn test_feature_extractor_different_images_different_features() {
        let extractor = FeatureExtractor::new();
        let frame1 = vec![0.0_f32; 16 * 16];
        let frame2: Vec<f32> = (0..16 * 16).map(|i| i as f32 / (16.0 * 16.0)).collect();
        let f1 = extractor.extract(&frame1, 16, 16).expect("extract");
        let f2 = extractor.extract(&frame2, 16, 16).expect("extract");
        // The feature vectors should differ.
        let diff: f32 = f1.iter().zip(f2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.0);
    }

    // ── gradient helpers ──────────────────────────────────────────────────────

    #[test]
    fn test_gradient_magnitudes_flat_image_zero() {
        let frame = vec![0.5_f32; 8 * 8];
        let mag = compute_gradient_magnitudes(&frame, 8, 8);
        assert!(mag.iter().all(|&v| v.abs() < 1e-5));
    }

    #[test]
    fn test_gradient_angles_in_range() {
        let frame: Vec<f32> = (0..8 * 8).map(|i| i as f32 / 64.0).collect();
        let ang = compute_gradient_angles(&frame, 8, 8);
        assert!(ang
            .iter()
            .all(|&v| v >= 0.0 && v < std::f32::consts::PI + 1e-4));
    }
}
