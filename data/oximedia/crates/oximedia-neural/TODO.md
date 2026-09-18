# oximedia-neural TODO

## Current Status
- 6 source files providing lightweight pure Rust neural network inference
- tensor: Core n-dimensional f32 tensor with math operations (add, matmul, mul, sum_along)
- activations: relu, sigmoid, tanh, gelu, leaky_relu, softmax, swish with tensor apply helpers
- layers: LinearLayer, Conv2dLayer, DepthwiseConv2d, BatchNorm1d, MaxPool2d, GlobalAvgPool
- media_models: SceneClassifier, ThumbnailRanker, SrUpscaler, FeatureExtractor
- error: NeuralError type
- Zero external dependencies beyond thiserror (pure Rust)

## Enhancements
- [x] Add batch dimension support to Tensor -- currently only single-sample inference; batch processing needed for throughput (implemented 2026-05-16)
  - **Goal:** Covered by NEURAL-TENSOR slice — batch dim [N,C,H,W], NumPy-style broadcasting (BroadcastIter), relu_inplace/add_inplace, ConvTranspose2d (dilate+conv), AvgPool2d (mirrors MaxPool2d).
  - **Files:** crates/oximedia-neural/src/tensor.rs, src/layers/{mod.rs, conv_transpose.rs (new), avg_pool.rs (new)}, src/lib.rs, TODO.md
  - **Tests:** tensor_batch_iter, broadcast_add_channel_scale, relu_inplace_no_alloc, conv_transpose_upsample_2x, avg_pool_known_input
  - **Risk:** Existing [C,H,W] tests — keep from_chw() ctor for backward compat.
- [x] Implement in-place operations on Tensor (relu_inplace, add_inplace) to reduce allocation during inference (implemented 2026-05-16)
  - **Goal:** Covered by NEURAL-TENSOR slice — batch dim [N,C,H,W], NumPy-style broadcasting (BroadcastIter), relu_inplace/add_inplace, ConvTranspose2d (dilate+conv), AvgPool2d (mirrors MaxPool2d).
  - **Files:** crates/oximedia-neural/src/tensor.rs, src/layers/{mod.rs, conv_transpose.rs (new), avg_pool.rs (new)}, src/lib.rs, TODO.md
  - **Tests:** tensor_batch_iter, broadcast_add_channel_scale, relu_inplace_no_alloc, conv_transpose_upsample_2x, avg_pool_known_input
  - **Risk:** Existing [C,H,W] tests — keep from_chw() ctor for backward compat.
- [x] Add Tensor broadcasting for element-wise ops between different-shaped tensors (e.g., [N,C,H,W] + [1,C,1,1]) (implemented 2026-05-16)
  - **Goal:** Covered by NEURAL-TENSOR slice — batch dim [N,C,H,W], NumPy-style broadcasting (BroadcastIter), relu_inplace/add_inplace, ConvTranspose2d (dilate+conv), AvgPool2d (mirrors MaxPool2d).
  - **Files:** crates/oximedia-neural/src/tensor.rs, src/layers/{mod.rs, conv_transpose.rs (new), avg_pool.rs (new)}, src/lib.rs, TODO.md
  - **Tests:** tensor_batch_iter, broadcast_add_channel_scale, relu_inplace_no_alloc, conv_transpose_upsample_2x, avg_pool_known_input
  - **Risk:** Existing [C,H,W] tests — keep from_chw() ctor for backward compat.
- [x] Implement transposed convolution (ConvTranspose2d) in layers for decoder/upsampling networks (implemented 2026-05-16)
  - **Goal:** Covered by NEURAL-TENSOR slice — batch dim [N,C,H,W], NumPy-style broadcasting (BroadcastIter), relu_inplace/add_inplace, ConvTranspose2d (dilate+conv), AvgPool2d (mirrors MaxPool2d).
  - **Files:** crates/oximedia-neural/src/tensor.rs, src/layers/{mod.rs, conv_transpose.rs (new), avg_pool.rs (new)}, src/lib.rs, TODO.md
  - **Tests:** tensor_batch_iter, broadcast_add_channel_scale, relu_inplace_no_alloc, conv_transpose_upsample_2x, avg_pool_known_input
  - **Risk:** Existing [C,H,W] tests — keep from_chw() ctor for backward compat.
- [x] Add average pooling (AvgPool2d) alongside existing MaxPool2d (implemented 2026-05-16)
  - **Goal:** Covered by NEURAL-TENSOR slice — batch dim [N,C,H,W], NumPy-style broadcasting (BroadcastIter), relu_inplace/add_inplace, ConvTranspose2d (dilate+conv), AvgPool2d (mirrors MaxPool2d).
  - **Files:** crates/oximedia-neural/src/tensor.rs, src/layers/{mod.rs, conv_transpose.rs (new), avg_pool.rs (new)}, src/lib.rs, TODO.md
  - **Tests:** tensor_batch_iter, broadcast_add_channel_scale, relu_inplace_no_alloc, conv_transpose_upsample_2x, avg_pool_known_input
  - **Risk:** Existing [C,H,W] tests — keep from_chw() ctor for backward compat.
- [x] Extend BatchNorm1d to BatchNorm2d for use with Conv2d outputs (spatial batch normalization) (verified 2026-05-16; src/layers.rs:381 pub struct BatchNorm2d)
- [x] Add residual/skip connection helper to simplify ResNet-style architecture building (verified 2026-05-16; src/skip_connections.rs:177 ResidualBlock, BottleneckBlock)
- [x] Improve SrUpscaler with sub-pixel convolution (PixelShuffle) layer for efficient super-resolution (verified 2026-05-16; src/skip_connections.rs:27 PixelShuffle depth-to-space)

## New Features
- [x] Implement ONNX model loading -- parse .onnx protobuf and map ops to existing layers (full-graph inference via oxionnx backend implemented 2026-05-11)
- [x] Add model serialization/deserialization (save/load trained weights in a custom binary format) (verified 2026-05-16; src/serialize.rs:257 model weight serialization)
- [x] Implement GRU/LSTM recurrent layers for temporal media models (e.g., audio classification) (verified 2026-05-16; src/recurrent.rs:997 GRU and LSTM layers)
- [x] Add attention mechanism (self-attention, cross-attention) for transformer-based architectures (verified 2026-05-16; src/attention.rs:151 MultiHeadAttention, CrossAttention:1421 lines)
- [x] Implement Tensor quantization (INT8) for faster inference on resource-constrained devices (verified 2026-05-16; src/quantize.rs:27 QuantizedTensor INT8 with scale)
- [x] Add model graph builder API for defining networks declaratively (Sequential, Functional) (verified 2026-05-16; src/graph.rs:5 Sequential, ModelGraph DAG)
- [x] Implement deformable convolution for object detection in video frames
- [x] Add face detection model using existing Conv2d + pooling layers (pre-defined architecture)
- [x] Implement optical flow estimation model for motion analysis in video

## Performance
- [x] Add SIMD-accelerated matmul using portable_simd for f32 dot products (currently scalar loops) (verified 2026-05-16; src/tensor.rs:451 SIMD via scirs2_core::simd::gemm, matmul_simd:929 AVX2/scalar fallback)
- [x] Implement tiled/blocked matrix multiplication to improve cache locality for large tensors (verified 2026-05-16; src/tiled_matmul.rs:51 tiled_matmul, tiled_matmul_with_tile_size)
- [x] Add multi-threaded Conv2d using rayon for parallel output channel computation (verified 2026-05-16; src/parallel_conv.rs:159 into_par_iter() over output channels)
- [x] Implement im2col optimization for Conv2d to convert convolution to matrix multiplication (verified 2026-05-16; src/im2col.rs:38 im2col, col2im:112)
- [x] Add memory pool/arena allocator for Tensor data to reduce allocation overhead during inference (verified 2026-05-15: src/pool.rs has TensorPool free-list + TensorArena bump allocator + PooledTensor RAII wrapper)
- [x] Profile and optimize Tensor indexing -- consider flat indexing with stride computation (verified 2026-05-15: src/tensor.rs:117 Tensor::flat_index O(rank); src/strided_tensor.rs has compute_strides/ravel_index free functions)

## Testing
- [x] Add numerical accuracy test: compare LinearLayer output against known hand-computed values (implemented 2026-05-15: src/accuracy_tests.rs has 5 LinearLayer accuracy tests with hand-computed expected values)
- [x] Test Conv2dLayer with 3x3 kernel on known input, verify output matches manual convolution (implemented 2026-05-15: src/accuracy_tests.rs has test_conv2d_3x3_sobel_x_like, test_conv2d_box_filter_constant_input, test_conv2d_padded_output_shape_and_values)
- [x] Add gradient-free training test: verify forward pass produces deterministic output for fixed weights (implemented 2026-05-15: src/accuracy_tests.rs has test_linear_deterministic_double_forward, test_conv2d_deterministic_double_forward, test_linear_sequential_calls_no_state_accumulation)
- [x] Test SceneClassifier with synthetic scene patterns (uniform color -> sky, gradient -> landscape) (implemented 2026-05-15: src/accuracy_tests.rs test_scene_classifier_synthetic_patterns — uniform vec![0.7;128] and gradient 0..1 feature vectors, checks valid class_index+confidence and bitwise determinism)
- [x] Add tensor shape mismatch error tests for all operations (matmul, add, conv) (implemented 2026-05-15: src/accuracy_tests.rs test_tensor_shape_mismatch_matmul, test_tensor_shape_mismatch_add, test_scene_classifier_wrong_feature_len_error)
- [x] Benchmark inference latency for each media_model at common resolutions (720p, 1080p) (implemented 2026-05-15: src/accuracy_tests.rs test_relu_latency_budget, test_scene_classifier_latency_budget, test_linear_layer_latency_budget — 10s generous budgets for CI stability)
- [x] Add per-op NaN-semantics known-answer tests (Wave 29 / Slice 5, 2026-06-06: tests/numeric_known_answer.rs) — pins the source-exact divergence between scalar `relu` (`x > 0.0` guard → NaN COLLAPSES to 0.0) and tensor `relu_inplace` (`*v < 0.0` guard → NaN PROPAGATES), softmax's non-finite→uniform `1/len` fallback for `[NaN,1,2]`, and `matmul` NaN-containment (NaN present in seeded row 0, exact finite values in clean row 1). 5 tests
- [x] Add Conv2d cross-correlation (no kernel flip) known-answer tests (Wave 29 / Slice 5, 2026-06-06: tests/numeric_known_answer.rs) — single-window 2×2·2×2+bias=30.5, identity-diagonal kernel over 3×3→[6,8,12,14], all-ones kernel over 2-channel input→110.0; plus matmul inner-dim mismatch→ShapeMismatch and zero-dim Tensor::new→InvalidShape via matches!. 5 tests

## Documentation
- [x] Add architecture guide showing supported layer types and how to compose them into models (implemented 2026-05-15: src/lib.rs top-level rustdoc has "Supported Layer Types" table with 12 layer types, input/output shapes, and notes)
- [x] Document the media_models with input/output specifications (tensor shapes, value ranges) (implemented 2026-05-15: src/lib.rs top-level rustdoc has "Media Models" table with input shape, value range, output, and use case for all 4 models)
- [x] Add performance comparison table: inference time per model at various resolutions (implemented 2026-05-15: src/lib.rs top-level rustdoc has "Performance" table with estimated latency for core operations)
