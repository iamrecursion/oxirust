//! # oximedia-neural
//!
//! Lightweight neural network inference for media processing, written entirely
//! in pure Rust with no C/Fortran dependencies.
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`tensor`] | Core n-dimensional f32 tensor and math operations |
//! | [`activations`] | Scalar activation functions and tensor apply helpers |
//! | [`layers`] | Neural network layers: linear, conv2d, batchnorm, pooling |
//! | [`media_models`] | Pre-built media pipelines: scene classifier, thumbnail ranker, SR upscaler, feature extractor |
//! | [`error`] | [`NeuralError`] type used throughout the crate |
//! | [`pool`] | TensorPool and TensorArena for reduced-allocation inference |
//! | [`onnx`] | ONNX model loading and inference |
//! | [`recurrent`] | GRU and LSTM recurrent layers |
//! | [`quantize`] | INT8 tensor quantization for efficient inference |
//! | [`skip_connections`] | ResidualBlock, BottleneckBlock, PixelShuffle, skip helpers |
//! | [`attention`] | Multi-head self/cross-attention, causal mask, positional encoding |
//! | [`graph`] | Declarative model graph builder and sequential executor |
//!
//! ## Quick Start
//!
//! ```rust
//! use oximedia_neural::tensor::Tensor;
//! use oximedia_neural::layers::LinearLayer;
//! use oximedia_neural::activations::{relu, ActivationFn, apply_activation};
//!
//! // Create a small fully-connected layer.
//! let layer = LinearLayer::new(4, 2).unwrap();
//! let input = Tensor::ones(vec![4]).unwrap();
//! let output = layer.forward(&input).unwrap();
//! assert_eq!(output.shape(), &[2]);
//! ```
//!
//! ## Architecture
//!
//! ### Supported Layer Types
//!
//! | Layer | Input shape | Output shape | Notes |
//! |-------|-------------|--------------|-------|
//! | [`LinearLayer`] | `[in_features]` | `[out_features]` | Fully-connected; y = Wx + b |
//! | [`Conv2dLayer`] | `[C_in, H, W]` | `[C_out, H', W']` | NCHW-style 2-D convolution |
//! | [`DepthwiseConv2d`] | `[C, H, W]` | `[C, H', W']` | Per-channel depthwise convolution |
//! | [`ConvTranspose2d`] | `[C_in, H, W]` | `[C_out, H', W']` | Transposed / fractionally-strided conv |
//! | [`BatchNorm1d`] | `[C]` or `[N, C]` | same shape | Normalise feature vectors |
//! | [`BatchNorm2d`] | `[C, H, W]` | same shape | Spatial batch normalisation |
//! | [`MaxPool2d`] | `[C, H, W]` | `[C, H', W']` | Max pooling over kernel window |
//! | [`AvgPool2d`] | `[C, H, W]` | `[C, H', W']` | Average pooling over kernel window |
//! | [`GlobalAvgPool`] | `[C, H, W]` | `[C]` | Channel-wise spatial average |
//! | ReLU / Sigmoid / GELU / Swish | any | same shape | Applied via [`apply_activation`] |
//! | Softmax | `[N]` | `[N]` | Normalised probabilities; see [`softmax`] |
//! | [`MultiHeadAttention`] | `[seq, d_model]` | `[seq, d_model]` | Self- or cross-attention |
//! | GRU / LSTM | `[seq, input_size]` | `[seq, hidden_size]` | Temporal recurrent layers; see [`recurrent`] |
//!
//! ### Media Models
//!
//! Pre-built pipelines in the [`media_models`] module accept raw feature vectors and
//! return structured results with no additional setup:
//!
//! | Model | Input shape | Value range | Output | Use case |
//! |-------|-------------|-------------|--------|----------|
//! | [`SceneClassifier`] | `[128]` f32 feature vector | `[0, 1]` | `(class_index: usize, confidence: f32)` | Shot-type classification (interior / exterior / sky …) |
//! | [`ThumbnailRanker`] | `[64]` f32 feature vector | `[0, 1]` | `f32` score in `[0, 1]` | Select the best-quality thumbnail frame |
//! | [`SrUpscaler`] | `[C, H, W]` pixel values | `[0, 1]` | `[C, 2H, 2W]` tensor | 2× super-resolution via PixelShuffle |
//! | [`FeatureExtractor`] | `[C, H, W]` pixel values | `[0, 1]` | `[128]` embedding | Compact visual embedding for retrieval / dedup |
//!
//! ### Performance
//!
//! Rough single-core throughput estimates on a modern x86-64 laptop (operations
//! per second, approximate order of magnitude):
//!
//! | Operation | Approx. latency | Notes |
//! |-----------|-----------------|-------|
//! | `relu_inplace` on `[3, 224, 224]` | < 1 ms | SIMD-accelerated in-place path |
//! | `SceneClassifier::classify` | < 1 ms | Two-layer MLP; SIMD matmul |
//! | `Conv2dLayer` 3×3 on `[3, 224, 224]` | 50–200 ms | im2col + SIMD matmul |
//! | SIMD matmul 512×512 | 10–50 ms | AVX2 path via [`matmul_simd`] |
//!
//! Enable the `onnx` Cargo feature to load arbitrary `.onnx` models through
//! the `onnx_backend` module (gated behind `#[cfg(feature = "onnx")]`).

pub mod accuracy_tests;
pub mod activations;
pub mod attention;
pub mod augment;
pub mod augmentation;
pub mod clip;
pub mod conv_variants;
pub mod deformable;
pub mod dropout;
pub mod error;
pub mod face_detection;
pub mod feature_extractor;
pub mod graph;
pub mod im2col;
pub mod inference_cache;
pub mod layers;
pub mod media_models;
pub mod metrics;
pub mod model_zoo;
pub mod object_detector;
pub mod onnx;
#[cfg(feature = "onnx")]
pub mod onnx_backend;
pub mod onnx_runtime;
pub mod optical_flow;
pub mod optimizer;
pub mod parallel_conv;
pub mod pool;
pub mod quantization;
pub mod quantize;
pub mod recurrent;
pub mod scheduler;
pub mod serialize;
pub mod skip_connections;
pub mod strided_tensor;
pub mod tensor;
pub mod tensor_batch;
pub mod tiled_matmul;
pub mod training;

// ── Re-exports ────────────────────────────────────────────────────────────────

pub use activations::{
    apply_activation, apply_activation_inplace, gelu, leaky_relu, relu, sigmoid, softmax, swish,
    tanh_act, ActivationFn,
};
pub use attention::{
    add_positional_encoding, apply_rope, causal_mask, causal_mask_rect, flash_attention,
    relative_position_shaw, rope_frequencies, scaled_dot_product_attention,
    sinusoidal_positional_encoding, sliding_window_mask, tiled_matmul_t, MultiHeadAttention,
};
pub use conv_variants::{DepthwiseSeparableConv2d, GroupConv2d, ParallelConv2d};
pub use error::NeuralError;
pub use im2col::{col2im, im2col};
pub use layers::{
    AvgPool2d, BatchNorm1d, BatchNorm2d, Conv2dLayer, ConvTranspose2d, DepthwiseConv2d,
    GlobalAvgPool, LinearLayer, MaxPool2d,
};
pub use media_models::{
    FeatureExtractor, SceneClass, SceneClassifier, SrUpscaler, ThumbnailRanker,
};
pub use model_zoo::{LayerConfig, MediaModelZoo, ModelInfo};
pub use onnx::{inspect_onnx, OnnxModelInfo};
#[cfg(feature = "onnx")]
pub use onnx_backend::OnnxBackend;
pub use onnx_runtime::{
    AddExecutor, ExecutionGraph, ExecutionNode, FallbackExecutor, GemmExecutor, MatMulExecutor,
    NodeExecutor, OnnxRuntime, ReluExecutor, ReshapeExecutor, SigmoidExecutor, TransposeExecutor,
};
pub use quantization::{
    dequantize_tensor, quantize_layer_weights, quantize_tensor, QuantizedLinearLayer,
    QuantizedTensor as SymmetricQuantizedTensor,
};
pub use tensor::{
    add, add_inplace, batched_matmul, broadcast_add, broadcast_mul, clamp_inplace, matmul,
    matmul_simd, matmul_simd_into, matmul_simd_runtime, mul, mul_inplace, relu_inplace,
    scale_inplace, sum_along, Tensor,
};
pub use tensor_batch::{
    batch_apply, batch_max, batch_mean, broadcast_sub, concat, split, split_sizes, TensorView,
    TensorViewOwned,
};
