//! Pre-configured neural network architectures for common media processing tasks.
//!
//! This module provides a *model zoo* — a catalogue of well-known network
//! architectures expressed as ordered sequences of [`LayerConfig`] descriptors.
//! No weights are included; the configs describe the *shape* of each layer so
//! callers can allocate matching weight tensors or map the architecture to an
//! inference backend.
//!
//! # Quick Start
//!
//! ```rust
//! use oximedia_neural::model_zoo::MediaModelZoo;
//!
//! let (info, layers) = MediaModelZoo::scene_classifier();
//! println!("Model: {}", info.name);
//! println!("Input shape: {:?}", info.input_shape);
//! println!("Layers: {}", layers.len());
//! ```

// ──────────────────────────────────────────────────────────────────────────────
// LayerConfig
// ──────────────────────────────────────────────────────────────────────────────

/// A purely descriptive layer specification — shape and type only, no weights.
///
/// Use this enum to represent the *architecture* of a network without
/// instantiating weight tensors.  Each variant corresponds to a common neural
/// network primitive.
#[derive(Debug, Clone, PartialEq)]
pub enum LayerConfig {
    /// 2-D convolution layer.
    Conv2d {
        /// Number of input feature maps.
        in_channels: usize,
        /// Number of output feature maps.
        out_channels: usize,
        /// Square kernel side length.
        kernel_size: usize,
        /// Horizontal and vertical stride.
        stride: usize,
        /// Zero-padding applied to each spatial edge.
        padding: usize,
    },
    /// Fully-connected (affine) layer.
    Linear {
        /// Input feature dimension.
        in_features: usize,
        /// Output feature dimension.
        out_features: usize,
    },
    /// 2-D batch normalisation.
    BatchNorm2d {
        /// Number of feature channels (must match the preceding Conv2d output).
        num_features: usize,
    },
    /// 2-D max pooling.
    MaxPool2d {
        /// Square kernel side length.
        kernel_size: usize,
        /// Stride (typically equal to `kernel_size` for non-overlapping pools).
        stride: usize,
    },
    /// Global average pooling — collapses H×W to a single scalar per channel.
    GlobalAvgPool,
    /// ReLU activation (`max(0, x)`).
    Relu,
    /// Sigmoid activation (`1 / (1 + exp(-x))`).
    Sigmoid,
    /// Softmax along the last dimension.
    Softmax,
    /// Dropout regularisation (inference no-op).
    Dropout {
        /// Drop probability in [0, 1).
        p: f32,
    },
    /// Flattens all dimensions except the batch dimension into one.
    Flatten,
}

// ──────────────────────────────────────────────────────────────────────────────
// ModelInfo
// ──────────────────────────────────────────────────────────────────────────────

/// Metadata describing a model architecture from the [`MediaModelZoo`].
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Short identifier used with [`MediaModelZoo::get_model`].
    pub name: String,
    /// Human-readable description of the model's purpose.
    pub description: String,
    /// Expected input tensor shape (excluding batch dimension).
    pub input_shape: Vec<usize>,
    /// Expected output tensor shape (excluding batch dimension).
    pub output_shape: Vec<usize>,
    /// Approximate total number of learnable parameters.
    pub parameter_count: usize,
}

impl ModelInfo {
    /// Computes the total parameter count for a sequence of [`LayerConfig`]s.
    ///
    /// The calculation follows standard conventions:
    /// - **Conv2d**: `out_channels × in_channels × kernel_size² + out_channels` (with bias)
    /// - **Linear**: `in_features × out_features + out_features` (with bias)
    /// - **BatchNorm2d**: `num_features × 4` (scale, bias, running mean, running var)
    /// - All other layers: 0 parameters
    pub fn parameter_count_from_layers(layers: &[LayerConfig]) -> usize {
        layers.iter().map(|l| layer_param_count(l)).sum()
    }
}

/// Returns the parameter count for a single layer config.
fn layer_param_count(layer: &LayerConfig) -> usize {
    match layer {
        LayerConfig::Conv2d {
            in_channels,
            out_channels,
            kernel_size,
            ..
        } => out_channels * in_channels * kernel_size * kernel_size + out_channels,
        LayerConfig::Linear {
            in_features,
            out_features,
        } => in_features * out_features + out_features,
        LayerConfig::BatchNorm2d { num_features } => num_features * 4,
        _ => 0,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// MediaModelZoo
// ──────────────────────────────────────────────────────────────────────────────

/// A catalogue of pre-configured neural network architectures for media tasks.
///
/// Each method returns a `(ModelInfo, Vec<LayerConfig>)` pair — metadata and
/// the ordered list of layer descriptors that define the architecture.
pub struct MediaModelZoo;

impl MediaModelZoo {
    // ── scene_classifier ─────────────────────────────────────────────────────

    /// A 3-layer CNN for 10-class scene classification.
    ///
    /// **Input**: `[3, 64, 64]` — RGB image, 64×64 pixels.
    /// **Output**: `[10]` — softmax probabilities over 10 scene classes.
    ///
    /// Architecture:
    /// 1. Conv2d(3→16, k=3, s=1, p=1), BN(16), ReLU, MaxPool(2,2)
    /// 2. Conv2d(16→32, k=3, s=1, p=1), BN(32), ReLU, MaxPool(2,2)
    /// 3. Conv2d(32→64, k=3, s=1, p=1), BN(64), ReLU, GlobalAvgPool
    /// 4. Linear(64→10), Softmax
    pub fn scene_classifier() -> (ModelInfo, Vec<LayerConfig>) {
        let layers = vec![
            // Stage 1
            LayerConfig::Conv2d {
                in_channels: 3,
                out_channels: 16,
                kernel_size: 3,
                stride: 1,
                padding: 1,
            },
            LayerConfig::BatchNorm2d { num_features: 16 },
            LayerConfig::Relu,
            LayerConfig::MaxPool2d {
                kernel_size: 2,
                stride: 2,
            },
            // Stage 2
            LayerConfig::Conv2d {
                in_channels: 16,
                out_channels: 32,
                kernel_size: 3,
                stride: 1,
                padding: 1,
            },
            LayerConfig::BatchNorm2d { num_features: 32 },
            LayerConfig::Relu,
            LayerConfig::MaxPool2d {
                kernel_size: 2,
                stride: 2,
            },
            // Stage 3
            LayerConfig::Conv2d {
                in_channels: 32,
                out_channels: 64,
                kernel_size: 3,
                stride: 1,
                padding: 1,
            },
            LayerConfig::BatchNorm2d { num_features: 64 },
            LayerConfig::Relu,
            LayerConfig::GlobalAvgPool,
            // Classifier head
            LayerConfig::Linear {
                in_features: 64,
                out_features: 10,
            },
            LayerConfig::Softmax,
        ];

        let param_count = ModelInfo::parameter_count_from_layers(&layers);
        let info = ModelInfo {
            name: "scene_classifier".to_string(),
            description: "3-layer CNN for 10-class scene classification".to_string(),
            input_shape: vec![3, 64, 64],
            output_shape: vec![10],
            parameter_count: param_count,
        };
        (info, layers)
    }

    // ── quality_estimator ────────────────────────────────────────────────────

    /// An MLP regressor that predicts Mean Opinion Score (MOS) in [1, 5] from
    /// a 256-dimensional feature vector.
    ///
    /// **Input**: `[256]` — perceptual feature vector.
    /// **Output**: `[1]` — sigmoid-normalised score in (0, 1); scale to [1, 5]
    /// with `1.0 + score * 4.0`.
    ///
    /// Architecture: Linear(256→128), ReLU, Dropout(0.3), Linear(128→64),
    /// ReLU, Linear(64→1), Sigmoid.
    pub fn quality_estimator() -> (ModelInfo, Vec<LayerConfig>) {
        let layers = vec![
            LayerConfig::Linear {
                in_features: 256,
                out_features: 128,
            },
            LayerConfig::Relu,
            LayerConfig::Dropout { p: 0.3 },
            LayerConfig::Linear {
                in_features: 128,
                out_features: 64,
            },
            LayerConfig::Relu,
            LayerConfig::Linear {
                in_features: 64,
                out_features: 1,
            },
            LayerConfig::Sigmoid,
        ];

        let param_count = ModelInfo::parameter_count_from_layers(&layers);
        let info = ModelInfo {
            name: "quality_estimator".to_string(),
            description: "Regressor predicting MOS (1-5) from 256-dim feature vector".to_string(),
            input_shape: vec![256],
            output_shape: vec![1],
            parameter_count: param_count,
        };
        (info, layers)
    }

    // ── shot_boundary_detector ───────────────────────────────────────────────

    /// A binary classifier that detects scene cuts from a pair of frame
    /// feature vectors.
    ///
    /// **Input**: `[512]` — concatenated feature vectors from two adjacent
    /// frames (each frame contributes 256 dimensions).
    /// **Output**: `[2]` — softmax probabilities `[P(no_cut), P(cut)]`.
    ///
    /// Architecture: Linear(512→256), ReLU, Linear(256→64), ReLU,
    /// Linear(64→2), Softmax.
    pub fn shot_boundary_detector() -> (ModelInfo, Vec<LayerConfig>) {
        let layers = vec![
            LayerConfig::Linear {
                in_features: 512,
                out_features: 256,
            },
            LayerConfig::Relu,
            LayerConfig::Linear {
                in_features: 256,
                out_features: 64,
            },
            LayerConfig::Relu,
            LayerConfig::Linear {
                in_features: 64,
                out_features: 2,
            },
            LayerConfig::Softmax,
        ];

        let param_count = ModelInfo::parameter_count_from_layers(&layers);
        let info = ModelInfo {
            name: "shot_boundary_detector".to_string(),
            description: "Binary classifier for scene cut detection from frame feature pairs"
                .to_string(),
            input_shape: vec![512],
            output_shape: vec![2],
            parameter_count: param_count,
        };
        (info, layers)
    }

    // ── object_detector_backbone ─────────────────────────────────────────────

    /// A lightweight feature-extraction backbone for object detection,
    /// inspired by MobileNet-style depthwise separable designs.
    ///
    /// **Input**: `[3, 224, 224]` — RGB image at 224×224.
    /// **Output**: `[1000]` — softmax logits over 1000 ImageNet categories.
    pub fn object_detector_backbone() -> (ModelInfo, Vec<LayerConfig>) {
        let layers = vec![
            LayerConfig::Conv2d {
                in_channels: 3,
                out_channels: 32,
                kernel_size: 3,
                stride: 2,
                padding: 1,
            },
            LayerConfig::BatchNorm2d { num_features: 32 },
            LayerConfig::Relu,
            LayerConfig::Conv2d {
                in_channels: 32,
                out_channels: 64,
                kernel_size: 3,
                stride: 1,
                padding: 1,
            },
            LayerConfig::BatchNorm2d { num_features: 64 },
            LayerConfig::Relu,
            LayerConfig::MaxPool2d {
                kernel_size: 2,
                stride: 2,
            },
            LayerConfig::GlobalAvgPool,
            LayerConfig::Flatten,
            LayerConfig::Linear {
                in_features: 64,
                out_features: 128,
            },
            LayerConfig::Relu,
            LayerConfig::Linear {
                in_features: 128,
                out_features: 1000,
            },
            LayerConfig::Softmax,
        ];

        let param_count = ModelInfo::parameter_count_from_layers(&layers);
        let info = ModelInfo {
            name: "object_detector_backbone".to_string(),
            description: "Lightweight feature extraction backbone for object detection".to_string(),
            input_shape: vec![3, 224, 224],
            output_shape: vec![1000],
            parameter_count: param_count,
        };
        (info, layers)
    }

    // ── catalogue helpers ─────────────────────────────────────────────────────

    /// Returns [`ModelInfo`] for all models in the zoo (without layer configs).
    pub fn list_models() -> Vec<ModelInfo> {
        vec![
            Self::scene_classifier().0,
            Self::quality_estimator().0,
            Self::shot_boundary_detector().0,
            Self::object_detector_backbone().0,
        ]
    }

    /// Looks up a model by name string.
    ///
    /// Returns `None` if the name is not recognised.
    ///
    /// # Recognised names
    ///
    /// - `"scene_classifier"`
    /// - `"quality_estimator"`
    /// - `"shot_boundary_detector"`
    /// - `"object_detector_backbone"`
    pub fn get_model(name: &str) -> Option<(ModelInfo, Vec<LayerConfig>)> {
        match name {
            "scene_classifier" => Some(Self::scene_classifier()),
            "quality_estimator" => Some(Self::quality_estimator()),
            "shot_boundary_detector" => Some(Self::shot_boundary_detector()),
            "object_detector_backbone" => Some(Self::object_detector_backbone()),
            _ => None,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── scene_classifier ─────────────────────────────────────────────────────

    #[test]
    fn test_scene_classifier_layer_count() {
        let (_, layers) = MediaModelZoo::scene_classifier();
        assert_eq!(layers.len(), 14);
    }

    #[test]
    fn test_scene_classifier_input_shape() {
        let (info, _) = MediaModelZoo::scene_classifier();
        assert_eq!(info.input_shape, vec![3, 64, 64]);
    }

    #[test]
    fn test_scene_classifier_output_shape() {
        let (info, _) = MediaModelZoo::scene_classifier();
        assert_eq!(info.output_shape, vec![10]);
    }

    #[test]
    fn test_scene_classifier_first_layer_is_conv() {
        let (_, layers) = MediaModelZoo::scene_classifier();
        assert!(
            matches!(
                layers[0],
                LayerConfig::Conv2d {
                    in_channels: 3,
                    out_channels: 16,
                    ..
                }
            ),
            "first layer should be Conv2d(3→16)"
        );
    }

    #[test]
    fn test_scene_classifier_last_layer_is_softmax() {
        let (_, layers) = MediaModelZoo::scene_classifier();
        assert_eq!(layers.last(), Some(&LayerConfig::Softmax));
    }

    // ── quality_estimator ────────────────────────────────────────────────────

    #[test]
    fn test_quality_estimator_layer_count() {
        let (_, layers) = MediaModelZoo::quality_estimator();
        assert_eq!(layers.len(), 7);
    }

    #[test]
    fn test_quality_estimator_input_shape() {
        let (info, _) = MediaModelZoo::quality_estimator();
        assert_eq!(info.input_shape, vec![256]);
    }

    #[test]
    fn test_quality_estimator_output_shape() {
        let (info, _) = MediaModelZoo::quality_estimator();
        assert_eq!(info.output_shape, vec![1]);
    }

    #[test]
    fn test_quality_estimator_has_dropout() {
        let (_, layers) = MediaModelZoo::quality_estimator();
        let has_dropout = layers
            .iter()
            .any(|l| matches!(l, LayerConfig::Dropout { .. }));
        assert!(
            has_dropout,
            "quality_estimator should contain a Dropout layer"
        );
    }

    #[test]
    fn test_quality_estimator_ends_with_sigmoid() {
        let (_, layers) = MediaModelZoo::quality_estimator();
        assert_eq!(layers.last(), Some(&LayerConfig::Sigmoid));
    }

    // ── shot_boundary_detector ───────────────────────────────────────────────

    #[test]
    fn test_shot_boundary_detector_layer_count() {
        let (_, layers) = MediaModelZoo::shot_boundary_detector();
        assert_eq!(layers.len(), 6);
    }

    #[test]
    fn test_shot_boundary_detector_input_shape() {
        let (info, _) = MediaModelZoo::shot_boundary_detector();
        assert_eq!(info.input_shape, vec![512]);
    }

    #[test]
    fn test_shot_boundary_detector_output_shape() {
        let (info, _) = MediaModelZoo::shot_boundary_detector();
        assert_eq!(info.output_shape, vec![2]);
    }

    // ── parameter counts ─────────────────────────────────────────────────────

    #[test]
    fn test_parameter_count_nonzero_all_main_models() {
        let (sc_info, _) = MediaModelZoo::scene_classifier();
        let (qe_info, _) = MediaModelZoo::quality_estimator();
        let (sb_info, _) = MediaModelZoo::shot_boundary_detector();
        assert!(
            sc_info.parameter_count > 0,
            "scene_classifier param_count should be > 0"
        );
        assert!(
            qe_info.parameter_count > 0,
            "quality_estimator param_count should be > 0"
        );
        assert!(
            sb_info.parameter_count > 0,
            "shot_boundary_detector param_count should be > 0"
        );
    }

    #[test]
    fn test_model_info_parameter_count_from_layers_simple() {
        // Linear(4→2): 4*2 + 2 = 10
        let layers = vec![LayerConfig::Linear {
            in_features: 4,
            out_features: 2,
        }];
        assert_eq!(ModelInfo::parameter_count_from_layers(&layers), 10);
    }

    #[test]
    fn test_model_info_parameter_count_conv2d() {
        // Conv2d(1→8, k=3): 8*1*3*3 + 8 = 72 + 8 = 80
        let layers = vec![LayerConfig::Conv2d {
            in_channels: 1,
            out_channels: 8,
            kernel_size: 3,
            stride: 1,
            padding: 0,
        }];
        assert_eq!(ModelInfo::parameter_count_from_layers(&layers), 80);
    }

    #[test]
    fn test_model_info_parameter_count_batchnorm2d() {
        // BatchNorm2d(16): 16*4 = 64
        let layers = vec![LayerConfig::BatchNorm2d { num_features: 16 }];
        assert_eq!(ModelInfo::parameter_count_from_layers(&layers), 64);
    }

    #[test]
    fn test_model_info_parameter_count_no_param_layers() {
        // Relu, Softmax, Dropout, GlobalAvgPool, Flatten — all contribute 0
        let layers = vec![
            LayerConfig::Relu,
            LayerConfig::Softmax,
            LayerConfig::Dropout { p: 0.5 },
            LayerConfig::GlobalAvgPool,
            LayerConfig::Flatten,
        ];
        assert_eq!(ModelInfo::parameter_count_from_layers(&layers), 0);
    }

    // ── catalogue helpers ─────────────────────────────────────────────────────

    #[test]
    fn test_list_models_returns_all() {
        let models = MediaModelZoo::list_models();
        assert_eq!(models.len(), 4);
    }

    #[test]
    fn test_list_models_names_unique() {
        let models = MediaModelZoo::list_models();
        let mut names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
        let original_len = names.len();
        names.dedup();
        // After sorting + dedup there should be no duplicates.
        let mut sorted = models.iter().map(|m| m.name.clone()).collect::<Vec<_>>();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), original_len, "model names must be unique");
    }

    #[test]
    fn test_get_model_by_name_scene_classifier() {
        let result = MediaModelZoo::get_model("scene_classifier");
        assert!(
            result.is_some(),
            "get_model(\"scene_classifier\") should be Some"
        );
        let (info, layers) = result.expect("some");
        assert_eq!(info.name, "scene_classifier");
        assert!(!layers.is_empty());
    }

    #[test]
    fn test_get_model_by_name_quality_estimator() {
        assert!(MediaModelZoo::get_model("quality_estimator").is_some());
    }

    #[test]
    fn test_get_model_by_name_shot_boundary_detector() {
        assert!(MediaModelZoo::get_model("shot_boundary_detector").is_some());
    }

    #[test]
    fn test_get_model_by_name_backbone() {
        assert!(MediaModelZoo::get_model("object_detector_backbone").is_some());
    }

    #[test]
    fn test_get_model_unknown_name() {
        assert!(
            MediaModelZoo::get_model("nonexistent_model").is_none(),
            "unknown name should return None"
        );
    }

    // ── LayerConfig field verification ────────────────────────────────────────

    #[test]
    fn test_layer_config_conv2d_fields() {
        let layer = LayerConfig::Conv2d {
            in_channels: 3,
            out_channels: 16,
            kernel_size: 3,
            stride: 1,
            padding: 1,
        };
        if let LayerConfig::Conv2d {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        } = layer
        {
            assert_eq!(in_channels, 3);
            assert_eq!(out_channels, 16);
            assert_eq!(kernel_size, 3);
            assert_eq!(stride, 1);
            assert_eq!(padding, 1);
        } else {
            panic!("expected Conv2d variant");
        }
    }

    #[test]
    fn test_layer_config_dropout_p_field() {
        let layer = LayerConfig::Dropout { p: 0.25 };
        if let LayerConfig::Dropout { p } = layer {
            assert!((p - 0.25).abs() < 1e-6);
        } else {
            panic!("expected Dropout variant");
        }
    }

    #[test]
    fn test_object_detector_backbone_input_shape() {
        let (info, _) = MediaModelZoo::object_detector_backbone();
        assert_eq!(info.input_shape, vec![3, 224, 224]);
    }

    #[test]
    fn test_object_detector_backbone_output_shape() {
        let (info, _) = MediaModelZoo::object_detector_backbone();
        assert_eq!(info.output_shape, vec![1000]);
    }
}
