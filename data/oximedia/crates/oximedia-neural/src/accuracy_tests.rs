//! Numerical accuracy and determinism tests for core neural network components.
//!
//! This module provides:
//! - Hand-verified numerical accuracy tests for [`LinearLayer`](crate::layers::LinearLayer)
//! - Manual convolution verification for [`Conv2dLayer`](crate::layers::Conv2dLayer)
//! - Deterministic forward-pass tests confirming repeatability without gradient state
//!
//! All expected values are pre-computed by hand or by independent reference
//! implementations, then embedded as constants so the tests remain self-contained.

// Nothing to export — this module is test-only.
#[cfg(test)]
mod tests {
    use crate::layers::{Conv2dLayer, LinearLayer};
    use crate::tensor::Tensor;

    // ─────────────────────────────────────────────────────────────────────────
    // Helper
    // ─────────────────────────────────────────────────────────────────────────

    /// Asserts that two float slices agree to within `tol` (absolute error).
    fn assert_close(got: &[f32], expected: &[f32], tol: f32, label: &str) {
        assert_eq!(
            got.len(),
            expected.len(),
            "{label}: length mismatch ({} vs {})",
            got.len(),
            expected.len()
        );
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() <= tol,
                "{label}: element [{i}] differs: got {g}, expected {e} (tol={tol})"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 1. LinearLayer numerical accuracy
    // ─────────────────────────────────────────────────────────────────────────
    //
    // We construct a 3-in → 2-out layer with known weights and bias, then
    // compare the output of `forward` with values computed by hand.
    //
    //   W = [[1, 2, 3],
    //        [4, 5, 6]]     (shape [2, 3])
    //   b = [0.5, -0.5]    (shape [2])
    //   x = [1, 1, 1]
    //
    //   y[0] = 1·1 + 2·1 + 3·1 + 0.5  = 6.5
    //   y[1] = 4·1 + 5·1 + 6·1 − 0.5  = 14.5

    #[test]
    fn test_linear_layer_accuracy_ones_input() {
        let mut layer = LinearLayer::new(3, 2).expect("create layer");
        // Set weights row by row (out_features × in_features, row-major).
        layer.weight = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
            .expect("weight tensor");
        layer.bias = Tensor::from_data(vec![0.5, -0.5], vec![2]).expect("bias tensor");

        let input = Tensor::from_data(vec![1.0, 1.0, 1.0], vec![3]).expect("input");
        let output = layer.forward(&input).expect("forward");

        assert_eq!(output.shape(), &[2]);
        assert_close(output.data(), &[6.5, 14.5], 1e-5, "ones input");
    }

    /// W · x + b with a non-trivial input vector.
    ///
    ///   W = [[1, -1],
    ///        [2,  3]]
    ///   b = [0, 1]
    ///   x = [2, -1]
    ///
    ///   y[0] = 1·2 + (-1)·(-1) + 0 = 3
    ///   y[1] = 2·2 + 3·(-1)    + 1 = 2
    #[test]
    fn test_linear_layer_accuracy_mixed_signs() {
        let mut layer = LinearLayer::new(2, 2).expect("create layer");
        layer.weight = Tensor::from_data(vec![1.0, -1.0, 2.0, 3.0], vec![2, 2]).expect("weight");
        layer.bias = Tensor::from_data(vec![0.0, 1.0], vec![2]).expect("bias");

        let input = Tensor::from_data(vec![2.0, -1.0], vec![2]).expect("input");
        let output = layer.forward(&input).expect("forward");

        assert_eq!(output.shape(), &[2], "output shape");
        assert_close(output.data(), &[3.0, 2.0], 1e-5, "mixed signs");
    }

    /// Zero weights should always produce a result equal to the bias.
    #[test]
    fn test_linear_layer_zero_weights_equals_bias() {
        let mut layer = LinearLayer::new(4, 3).expect("create");
        // weight is already zero; just set non-trivial bias.
        layer.bias = Tensor::from_data(vec![1.0, -2.0, 3.0], vec![3]).expect("bias");

        let input = Tensor::from_data(vec![5.0, -3.0, 0.5, 100.0], vec![4]).expect("input");
        let output = layer.forward(&input).expect("forward");

        assert_eq!(output.shape(), &[3]);
        assert_close(output.data(), &[1.0, -2.0, 3.0], 1e-5, "zero-weight bias");
    }

    /// Zero bias should leave output equal to W·x.
    #[test]
    fn test_linear_layer_zero_bias_pure_matmul() {
        let mut layer = LinearLayer::new(2, 2).expect("create");
        // Identity-like weight: diagonal 2×2.
        layer.weight = Tensor::from_data(vec![2.0, 0.0, 0.0, 3.0], vec![2, 2]).expect("weight");
        // bias is already zero.

        let input = Tensor::from_data(vec![4.0, 5.0], vec![2]).expect("input");
        let output = layer.forward(&input).expect("forward");

        // y = [2*4 + 0*5, 0*4 + 3*5] = [8, 15]
        assert_close(output.data(), &[8.0, 15.0], 1e-5, "zero bias");
    }

    /// 1→1 layer: scalar multiplication.
    #[test]
    fn test_linear_layer_scalar() {
        let mut layer = LinearLayer::new(1, 1).expect("create");
        layer.weight = Tensor::from_data(vec![3.0], vec![1, 1]).expect("weight");
        layer.bias = Tensor::from_data(vec![1.5], vec![1]).expect("bias");

        let input = Tensor::from_data(vec![4.0], vec![1]).expect("input");
        let output = layer.forward(&input).expect("forward");

        // y = 3*4 + 1.5 = 13.5
        assert_close(output.data(), &[13.5], 1e-5, "scalar");
    }

    /// Batched forward (2 samples through 2→2 layer).
    ///
    ///   W = [[1, 0],
    ///        [0, 1]]   (identity)
    ///   b = [1, -1]
    ///   batch = [[2, 3],
    ///            [5, 7]]
    ///
    ///   sample 0: [2+1, 3-1] = [3, 2]
    ///   sample 1: [5+1, 7-1] = [6, 6]
    #[test]
    fn test_linear_layer_batch_forward() {
        let mut layer = LinearLayer::new(2, 2).expect("create");
        layer.weight = Tensor::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).expect("weight");
        layer.bias = Tensor::from_data(vec![1.0, -1.0], vec![2]).expect("bias");

        let batch = Tensor::from_data(vec![2.0, 3.0, 5.0, 7.0], vec![2, 2]).expect("batch");
        let output = layer.forward_batch(&batch).expect("forward_batch");

        assert_eq!(output.shape(), &[2, 2]);
        assert_close(output.data(), &[3.0, 2.0, 6.0, 6.0], 1e-5, "batch");
    }

    /// Wrong rank input returns an error.
    #[test]
    fn test_linear_layer_wrong_rank_error() {
        let layer = LinearLayer::new(4, 2).expect("create");
        let bad_input = Tensor::from_data(vec![1.0; 8], vec![2, 4]).expect("2d input");
        assert!(
            layer.forward(&bad_input).is_err(),
            "2-D input to 1-D forward must error"
        );
    }

    /// Dimension mismatch returns an error.
    #[test]
    fn test_linear_layer_dim_mismatch_error() {
        let layer = LinearLayer::new(4, 2).expect("create");
        let bad_input = Tensor::from_data(vec![1.0; 3], vec![3]).expect("wrong-len input");
        assert!(
            layer.forward(&bad_input).is_err(),
            "length-3 input to in_features=4 must error"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 2. Conv2dLayer 3×3 kernel numerical accuracy
    // ─────────────────────────────────────────────────────────────────────────
    //
    // We define a 1-channel 5×5 input and a single 1-in/1-out 3×3 kernel, then
    // compare the conv2d output to a manually computed reference.
    //
    // Input (1×5×5):
    //   1  2  3  4  5
    //   6  7  8  9 10
    //  11 12 13 14 15
    //  16 17 18 19 20
    //  21 22 23 24 25
    //
    // Kernel (1×1×3×3):
    //   1 0 -1
    //   1 0 -1
    //   1 0 -1
    //   (Sobel-X-like, zero bias)
    //
    // stride=(1,1), padding=(0,0) → output is 1×3×3.
    //
    // Manual computation for each of the 9 output positions:
    //   out[r,c] = sum over (dr,dc): kernel[dr,dc] * input[r+dr, c+dc]
    //
    //   out[0,0]: cols 0..3, rows 0..3
    //     = (1·1 + 0·2 + -1·3) + (1·6 + 0·7 + -1·8) + (1·11 + 0·12 + -1·13)
    //     = (1-3) + (6-8) + (11-13) = -2 + -2 + -2 = -6
    //
    //   out[0,1]: cols 1..4
    //     = (1·2 + 0·3 + -1·4) + (1·7 + 0·8 + -1·9) + (1·12 + 0·13 + -1·14)
    //     = (2-4) + (7-9) + (12-14) = -2 + -2 + -2 = -6
    //
    //   out[0,2]: cols 2..5
    //     = (1·3 + 0·4 + -1·5) + (1·8 + 0·9 + -1·10) + (1·13 + 0·14 + -1·15)
    //     = (3-5) + (8-10) + (13-15) = -2 + -2 + -2 = -6
    //
    //   Because all rows have constant column differences of 2 (the 5×5 input
    //   has values increasing uniformly), every output cell = -6.
    //
    //   Expected output (1×3×3): all -6.

    #[test]
    fn test_conv2d_3x3_sobel_x_like() {
        let input_data: Vec<f32> = (1..=25).map(|x| x as f32).collect();
        let input = Tensor::from_data(input_data, vec![1, 5, 5]).expect("input");

        // Kernel: Sobel-X columns [1, 0, -1] repeated for each row.
        let kernel_data: Vec<f32> = vec![1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0];
        let mut layer = Conv2dLayer::new(1, 1, 3, 3, (1, 1), (0, 0)).expect("create conv");
        layer.weight = Tensor::from_data(kernel_data, vec![1, 1, 3, 3]).expect("weight");
        // bias stays zero.

        let output = layer.forward(&input).expect("forward");

        assert_eq!(output.shape(), &[1, 3, 3], "output shape");
        // All 9 output values should be -6.
        for (i, &v) in output.data().iter().enumerate() {
            assert!((v - (-6.0_f32)).abs() < 1e-4, "out[{i}] = {v}, expected -6");
        }
    }

    /// 3×3 all-ones kernel (box filter / sum) on a constant input should produce
    /// out_val = 9 · constant_value.
    #[test]
    fn test_conv2d_box_filter_constant_input() {
        // 1-channel 4×4 input filled with 2.0.
        let input = Tensor::from_data(vec![2.0_f32; 16], vec![1, 4, 4]).expect("input");

        let kernel = vec![1.0_f32; 9]; // all-ones 3×3
        let mut layer = Conv2dLayer::new(1, 1, 3, 3, (1, 1), (0, 0)).expect("create");
        layer.weight = Tensor::from_data(kernel, vec![1, 1, 3, 3]).expect("weight");

        let output = layer.forward(&input).expect("forward");

        // out_H = (4 - 3)/1 + 1 = 2,  out_W = 2 → shape [1, 2, 2]
        assert_eq!(output.shape(), &[1, 2, 2], "shape");
        // Every output = 9 * 2 = 18
        for (i, &v) in output.data().iter().enumerate() {
            assert!((v - 18.0_f32).abs() < 1e-4, "out[{i}] = {v}, expected 18");
        }
    }

    /// With padding=(1,1) and all-ones kernel, a 3×3 constant input should
    /// produce the correct values: corners=4·c, edges=6·c, center=9·c.
    #[test]
    fn test_conv2d_padded_output_shape_and_values() {
        let c = 1.0_f32;
        let input = Tensor::from_data(vec![c; 9], vec![1, 3, 3]).expect("input");

        let kernel = vec![1.0_f32; 9];
        let mut layer = Conv2dLayer::new(1, 1, 3, 3, (1, 1), (1, 1)).expect("create");
        layer.weight = Tensor::from_data(kernel, vec![1, 1, 3, 3]).expect("weight");

        let output = layer.forward(&input).expect("forward");

        // With same padding, output shape == input shape: [1, 3, 3]
        assert_eq!(output.shape(), &[1, 3, 3], "padded shape");

        let d = output.data();
        // corner cells see only 4 filled neighbours (rest is zero-padded)
        let corners = [d[0], d[2], d[6], d[8]];
        for &v in &corners {
            assert!((v - 4.0).abs() < 1e-4, "corner {v}, expected 4");
        }
        // edge centres see 6 neighbours
        let edges = [d[1], d[3], d[5], d[7]];
        for &v in &edges {
            assert!((v - 6.0).abs() < 1e-4, "edge {v}, expected 6");
        }
        // centre cell sees all 9 neighbours
        assert!((d[4] - 9.0).abs() < 1e-4, "centre {}, expected 9", d[4]);
    }

    /// Bias is added correctly: output with non-zero bias should equal
    /// zero-bias output + bias value.
    #[test]
    fn test_conv2d_bias_addition() {
        let input = Tensor::from_data(vec![1.0_f32; 9], vec![1, 3, 3]).expect("input");

        // Identity kernel: a single 1 in the centre of a 3×3 kernel.
        let mut kernel = vec![0.0_f32; 9];
        kernel[4] = 1.0; // centre of 3×3

        let mut layer = Conv2dLayer::new(1, 1, 3, 3, (1, 1), (1, 1)).expect("create");
        layer.weight = Tensor::from_data(kernel, vec![1, 1, 3, 3]).expect("weight");
        layer.bias = Tensor::from_data(vec![5.0], vec![1]).expect("bias");

        let output = layer.forward(&input).expect("forward");

        // Identity conv on all-ones input = all-ones; + bias 5 = all 6.
        assert_eq!(output.shape(), &[1, 3, 3]);
        for (i, &v) in output.data().iter().enumerate() {
            assert!((v - 6.0).abs() < 1e-4, "bias test out[{i}] = {v}");
        }
    }

    /// Wrong number of input channels returns an error.
    #[test]
    fn test_conv2d_channel_mismatch_error() {
        let layer = Conv2dLayer::new(2, 1, 3, 3, (1, 1), (0, 0)).expect("create");
        // Feed a 1-channel input to a 2-channel conv.
        let input = Tensor::from_data(vec![1.0_f32; 9], vec![1, 3, 3]).expect("input");
        assert!(
            layer.forward(&input).is_err(),
            "channel mismatch must return error"
        );
    }

    /// Wrong rank input (2-D instead of 3-D) returns an error.
    #[test]
    fn test_conv2d_wrong_rank_error() {
        let layer = Conv2dLayer::new(1, 1, 3, 3, (1, 1), (0, 0)).expect("create");
        let input = Tensor::from_data(vec![1.0_f32; 9], vec![3, 3]).expect("2d input");
        assert!(layer.forward(&input).is_err(), "2-D input must error");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 3. Gradient-free determinism: verify forward pass reproducibility
    // ─────────────────────────────────────────────────────────────────────────
    //
    // For any deterministic layer (no randomness in weights or activations),
    // running `forward` twice on the same input with fixed weights must yield
    // identical results.  We also verify that repeated construction with the
    // same weight data produces the same output, confirming no hidden state.

    /// Linear layer: identical inputs produce identical outputs (call it twice).
    #[test]
    fn test_linear_deterministic_double_forward() {
        let mut layer = LinearLayer::new(3, 2).expect("create");
        layer.weight =
            Tensor::from_data(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], vec![2, 3]).expect("weight");
        layer.bias = Tensor::from_data(vec![0.01, -0.01], vec![2]).expect("bias");

        let input = Tensor::from_data(vec![1.5, -0.5, 2.0], vec![3]).expect("input");

        let out1 = layer.forward(&input).expect("forward 1");
        let out2 = layer.forward(&input).expect("forward 2");

        // Outputs must be bitwise identical (no stochastic elements).
        assert_eq!(out1.data(), out2.data(), "double forward must be identical");
        assert_eq!(out1.shape(), out2.shape());
    }

    /// Conv2d layer: identical inputs produce identical outputs.
    #[test]
    fn test_conv2d_deterministic_double_forward() {
        let input_data: Vec<f32> = (1..=25).map(|x| x as f32 * 0.1).collect();
        let input = Tensor::from_data(input_data, vec![1, 5, 5]).expect("input");

        let kernel_data = vec![0.1_f32, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9];
        let mut layer = Conv2dLayer::new(1, 1, 3, 3, (1, 1), (0, 0)).expect("create");
        layer.weight = Tensor::from_data(kernel_data, vec![1, 1, 3, 3]).expect("weight");
        layer.bias = Tensor::from_data(vec![0.05], vec![1]).expect("bias");

        let out1 = layer.forward(&input).expect("forward 1");
        let out2 = layer.forward(&input).expect("forward 2");

        assert_eq!(out1.data(), out2.data(), "double forward must be identical");
        assert_eq!(out1.shape(), out2.shape());
    }

    /// Different inputs with the same weights must produce different outputs.
    #[test]
    fn test_linear_different_inputs_different_outputs() {
        let mut layer = LinearLayer::new(2, 2).expect("create");
        layer.weight = Tensor::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).expect("weight");
        // bias zero.

        let input_a = Tensor::from_data(vec![1.0, 2.0], vec![2]).expect("a");
        let input_b = Tensor::from_data(vec![3.0, 4.0], vec![2]).expect("b");

        let out_a = layer.forward(&input_a).expect("fwd a");
        let out_b = layer.forward(&input_b).expect("fwd b");

        assert_ne!(
            out_a.data(),
            out_b.data(),
            "different inputs must produce different outputs"
        );
    }

    /// Constructing two identical layers from the same weights and forwarding
    /// the same input must yield identical results (no hidden mutable state).
    #[test]
    fn test_linear_no_hidden_mutable_state() {
        let weight_data = vec![2.0_f32, -1.0, 0.5, 3.0];
        let bias_data = vec![0.25_f32, -0.25];

        let mut layer_a = LinearLayer::new(2, 2).expect("a");
        layer_a.weight = Tensor::from_data(weight_data.clone(), vec![2, 2]).expect("weight a");
        layer_a.bias = Tensor::from_data(bias_data.clone(), vec![2]).expect("bias a");

        let mut layer_b = LinearLayer::new(2, 2).expect("b");
        layer_b.weight = Tensor::from_data(weight_data, vec![2, 2]).expect("weight b");
        layer_b.bias = Tensor::from_data(bias_data, vec![2]).expect("bias b");

        let input = Tensor::from_data(vec![1.0, -1.0], vec![2]).expect("input");

        let out_a = layer_a.forward(&input).expect("fwd a");
        let out_b = layer_b.forward(&input).expect("fwd b");

        assert_eq!(
            out_a.data(),
            out_b.data(),
            "same weights must yield same output"
        );
    }

    /// Multiple sequential forwards through a linear layer accumulate no error.
    #[test]
    fn test_linear_sequential_calls_no_state_accumulation() {
        let mut layer = LinearLayer::new(3, 2).expect("create");
        layer.weight =
            Tensor::from_data(vec![1.0, -1.0, 0.5, 0.0, 2.0, -0.5], vec![2, 3]).expect("weight");
        layer.bias = Tensor::from_data(vec![0.1, -0.1], vec![2]).expect("bias");

        let input = Tensor::from_data(vec![0.3, 0.6, 0.9], vec![3]).expect("input");

        // Run 10 sequential forward passes and verify they all equal the first.
        let reference = layer.forward(&input).expect("ref");
        for i in 1..=10 {
            let out = layer.forward(&input).expect("forward");
            assert_eq!(
                out.data(),
                reference.data(),
                "forward #{i} must equal reference"
            );
        }
    }

    /// A Conv2d with a single all-ones 1×1 kernel is an identity scaling.
    ///
    /// stride=(1,1), pad=(0,0), kernel=1×1 of value k → out = k · input.
    #[test]
    fn test_conv2d_1x1_identity_scaling() {
        let input_data: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let input = Tensor::from_data(input_data.clone(), vec![1, 3, 3]).expect("input");

        let mut layer = Conv2dLayer::new(1, 1, 1, 1, (1, 1), (0, 0)).expect("create");
        layer.weight = Tensor::from_data(vec![3.0], vec![1, 1, 1, 1]).expect("weight");
        // bias zero.

        let output = layer.forward(&input).expect("forward");

        assert_eq!(output.shape(), &[1, 3, 3]);
        for (i, (&got, &src)) in output.data().iter().zip(input_data.iter()).enumerate() {
            assert!(
                (got - src * 3.0).abs() < 1e-4,
                "1×1 scale: out[{i}] = {got}, expected {}",
                src * 3.0
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 4. Tensor shape mismatch error tests
    // ─────────────────────────────────────────────────────────────────────────
    //
    // All tensor operations that require compatible shapes must return `Err`
    // (never panic) when given incompatible shapes.

    /// `matmul` with incompatible inner dimensions must return an error.
    #[test]
    fn test_tensor_shape_mismatch_matmul() {
        // [2,3] × [4,5] → inner dims 3 ≠ 4, must fail.
        let a = Tensor::from_data(vec![1.0_f32; 6], vec![2, 3]).expect("a");
        let b = Tensor::from_data(vec![1.0_f32; 20], vec![4, 5]).expect("b");
        assert!(
            crate::tensor::matmul(&a, &b).is_err(),
            "matmul([2,3], [4,5]) must return Err (inner dims 3 ≠ 4)"
        );
    }

    /// Element-wise `add` with mismatched shapes must return an error.
    #[test]
    fn test_tensor_shape_mismatch_add() {
        let a = Tensor::from_data(vec![1.0_f32; 3], vec![3]).expect("a");
        let b = Tensor::from_data(vec![1.0_f32; 4], vec![4]).expect("b");
        assert!(
            crate::tensor::add(&a, &b).is_err(),
            "add([3], [4]) must return Err (shape mismatch)"
        );
    }

    /// `SceneClassifier::classify` with wrong feature length must return an error.
    #[test]
    fn test_scene_classifier_wrong_feature_len_error() {
        use crate::media_models::SceneClassifier;
        let clf = SceneClassifier::new().expect("new");
        // Feed 64 features to a 128-dim classifier — must be Err, never panic.
        let short_features = vec![0.5_f32; 64];
        assert!(
            clf.classify(&short_features).is_err(),
            "classify with wrong feature length must return Err"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 5. SceneClassifier synthetic pattern tests
    // ─────────────────────────────────────────────────────────────────────────
    //
    // Feed controlled synthetic feature vectors through `SceneClassifier::classify`
    // and verify that the returned (class_index, confidence) pair is valid without
    // panicking.  This confirms the full softmax→argmax pipeline is correct.

    /// Uniform and gradient feature vectors must produce valid, reproducible output.
    #[test]
    fn test_scene_classifier_synthetic_patterns() {
        use crate::media_models::SceneClassifier;
        let clf = SceneClassifier::new().expect("new");

        // Uniform "sky" feature vector — all values equal 0.7.
        let uniform = vec![0.7_f32; SceneClassifier::INPUT_DIM];
        let (idx_u, conf_u) = clf.classify(&uniform).expect("uniform classify");
        assert!(
            idx_u < SceneClassifier::NUM_CLASSES,
            "uniform: class index {idx_u} out of range (max {})",
            SceneClassifier::NUM_CLASSES - 1
        );
        assert!(
            conf_u.is_finite() && (0.0_f32..=1.0_f32).contains(&conf_u),
            "uniform: confidence {conf_u} must be finite and in [0, 1]"
        );

        // Gradient "landscape" feature vector — values ramp linearly from 0 to 1.
        let gradient: Vec<f32> = (0..SceneClassifier::INPUT_DIM)
            .map(|i| i as f32 / SceneClassifier::INPUT_DIM as f32)
            .collect();
        let (idx_g, conf_g) = clf.classify(&gradient).expect("gradient classify");
        assert!(
            idx_g < SceneClassifier::NUM_CLASSES,
            "gradient: class index {idx_g} out of range (max {})",
            SceneClassifier::NUM_CLASSES - 1
        );
        assert!(
            conf_g.is_finite() && (0.0_f32..=1.0_f32).contains(&conf_g),
            "gradient: confidence {conf_g} must be finite and in [0, 1]"
        );

        // Both calls must be deterministic (second run equals first).
        let (idx_u2, conf_u2) = clf.classify(&uniform).expect("uniform 2nd");
        assert_eq!(idx_u, idx_u2, "uniform: class index must be deterministic");
        assert_eq!(
            conf_u.to_bits(),
            conf_u2.to_bits(),
            "uniform: confidence must be bitwise stable across calls"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 6. Inference latency budget tests
    // ─────────────────────────────────────────────────────────────────────────
    //
    // These tests run each operation many times and assert the total wall-clock
    // time stays under a generous budget.  The budget is intentionally large
    // (10 s for sub-millisecond operations) so tests never produce false
    // failures on CI, while still catching catastrophic regressions.

    /// Repeated relu_inplace on a 3×224×224 tensor must finish well under 10 s.
    #[test]
    fn test_relu_latency_budget() {
        use crate::tensor::relu_inplace;
        use std::time::Instant;

        let mut t =
            Tensor::from_data(vec![-1.0_f32; 3 * 224 * 224], vec![3, 224, 224]).expect("tensor");

        let start = Instant::now();
        for _ in 0..10 {
            relu_inplace(&mut t);
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_secs() < 10,
            "10× relu_inplace on [3,224,224] took {:?} — budget is 10 s",
            elapsed
        );
    }

    /// Repeated SceneClassifier inference must finish well under 10 s.
    #[test]
    fn test_scene_classifier_latency_budget() {
        use crate::media_models::SceneClassifier;
        use std::time::Instant;

        let clf = SceneClassifier::new().expect("new");
        let features = vec![0.5_f32; SceneClassifier::INPUT_DIM];

        let start = Instant::now();
        for _ in 0..100 {
            clf.classify(&features).expect("classify");
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_secs() < 10,
            "100× SceneClassifier::classify took {:?} — budget is 10 s",
            elapsed
        );
    }

    /// Repeated LinearLayer(256→128) forward passes must finish well under 30 s.
    #[test]
    fn test_linear_layer_latency_budget() {
        use std::time::Instant;

        let layer = LinearLayer::new(256, 128).expect("create");
        let input = Tensor::from_data(vec![0.1_f32; 256], vec![256]).expect("input");

        let start = Instant::now();
        for _ in 0..1000 {
            layer.forward(&input).expect("forward");
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_secs() < 30,
            "1000× LinearLayer(256→128) forward took {:?} — budget is 30 s",
            elapsed
        );
    }
}
