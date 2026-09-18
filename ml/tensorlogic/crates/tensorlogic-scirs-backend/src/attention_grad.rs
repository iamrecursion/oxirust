//! Backward pass for attention operations.
//!
//! Computes gradients through scaled dot-product attention with respect to
//! query (Q), key (K), and value (V) matrices.

use crate::attention::{AttentionConfig, AttentionError};
use scirs2_core::ndarray::{Array2, Array3, Array4};

/// Gradients of a single-head attention operation.
#[derive(Debug, Clone)]
pub struct AttentionGradients {
    /// Gradient w.r.t. query [seq_q, head_dim]
    pub dq: Array2<f64>,
    /// Gradient w.r.t. key [seq_k, head_dim]
    pub dk: Array2<f64>,
    /// Gradient w.r.t. value [seq_k, d_v]
    pub dv: Array2<f64>,
}

/// Gradients of a multi-head attention operation.
#[derive(Debug, Clone)]
pub struct MultiHeadAttentionGrad {
    /// Gradient w.r.t. query [batch, seq_q, model_dim]
    pub dq: Array3<f64>,
    /// Gradient w.r.t. key [batch, seq_k, model_dim]
    pub dk: Array3<f64>,
    /// Gradient w.r.t. value [batch, seq_k, model_dim]
    pub dv: Array3<f64>,
}

/// Compute the backward pass through softmax.
///
/// Given the softmax output `p` and upstream gradient `dp`:
/// `d_logits[i,j] = p[i,j] * (dp[i,j] - sum_k(dp[i,k] * p[i,k]))`
///
/// This is the standard Jacobian-vector product for the softmax function.
pub fn softmax_backward(p: &Array2<f64>, dp: &Array2<f64>) -> Array2<f64> {
    let nrows = p.nrows();
    let ncols = p.ncols();
    let mut result = Array2::<f64>::zeros((nrows, ncols));

    for i in 0..nrows {
        // Compute dot product: sum_k(dp[i,k] * p[i,k])
        let dot: f64 = (0..ncols).map(|k| dp[[i, k]] * p[[i, k]]).sum();
        // d_logits[i,j] = p[i,j] * (dp[i,j] - dot)
        for j in 0..ncols {
            result[[i, j]] = p[[i, j]] * (dp[[i, j]] - dot);
        }
    }

    result
}

/// Backward pass through scaled dot-product attention.
///
/// # Arguments
/// - `dout`: upstream gradient [seq_q, d_v]
/// - `q`: query [seq_q, d_q]
/// - `k`: key [seq_k, d_k]
/// - `v`: value [seq_k, d_v]
/// - `weights`: attention weights from forward pass [seq_q, seq_k] (softmax output)
/// - `scale`: the same scale factor used in forward pass
///
/// # Returns
/// Gradients w.r.t. Q, K, V
pub fn attention_backward(
    dout: &Array2<f64>,
    q: &Array2<f64>,
    k: &Array2<f64>,
    v: &Array2<f64>,
    weights: &Array2<f64>,
    scale: f64,
) -> Result<AttentionGradients, AttentionError> {
    let (seq_q, d_v) = (dout.nrows(), dout.ncols());
    let (q_rows, d_q) = (q.nrows(), q.ncols());
    let (seq_k, d_k) = (k.nrows(), k.ncols());
    let (v_rows, v_cols) = (v.nrows(), v.ncols());
    let (w_rows, w_cols) = (weights.nrows(), weights.ncols());

    // Validate shapes
    if q_rows != seq_q {
        return Err(AttentionError::ShapeMismatch(format!(
            "dout seq_q {} != Q seq_q {}",
            seq_q, q_rows
        )));
    }
    if d_q != d_k {
        return Err(AttentionError::ShapeMismatch(format!(
            "Q head_dim {} != K head_dim {}",
            d_q, d_k
        )));
    }
    if seq_k != v_rows {
        return Err(AttentionError::ShapeMismatch(format!(
            "K seq {} != V seq {}",
            seq_k, v_rows
        )));
    }
    if v_cols != d_v {
        return Err(AttentionError::ShapeMismatch(format!(
            "V d_v {} != dout d_v {}",
            v_cols, d_v
        )));
    }
    if w_rows != seq_q || w_cols != seq_k {
        return Err(AttentionError::ShapeMismatch(format!(
            "weights shape [{},{}] != expected [{},{}]",
            w_rows, w_cols, seq_q, seq_k
        )));
    }

    // dV = W^T @ dout  [seq_k, d_v]
    let mut dv = Array2::<f64>::zeros((seq_k, d_v));
    for j in 0..seq_k {
        for d in 0..d_v {
            let mut sum = 0.0;
            for i in 0..seq_q {
                sum += weights[[i, j]] * dout[[i, d]];
            }
            dv[[j, d]] = sum;
        }
    }

    // dW = dout @ V^T  [seq_q, seq_k]
    let mut dw = Array2::<f64>::zeros((seq_q, seq_k));
    for i in 0..seq_q {
        for j in 0..seq_k {
            let mut sum = 0.0;
            for d in 0..d_v {
                sum += dout[[i, d]] * v[[j, d]];
            }
            dw[[i, j]] = sum;
        }
    }

    // d_scores = softmax_backward(weights, dW)  [seq_q, seq_k]
    let d_scores = softmax_backward(weights, &dw);

    // d_scaled = d_scores / scale  [seq_q, seq_k]
    // Note: forward did scores = (Q @ K^T) / scale then softmax
    // So gradient through the division by scale is also division by scale
    let inv_scale = if scale.abs() > 0.0 { 1.0 / scale } else { 0.0 };
    let d_scaled = &d_scores * inv_scale;

    // dQ = d_scaled @ K  [seq_q, d_q]
    let mut dq = Array2::<f64>::zeros((seq_q, d_q));
    for i in 0..seq_q {
        for d in 0..d_q {
            let mut sum = 0.0;
            for j in 0..seq_k {
                sum += d_scaled[[i, j]] * k[[j, d]];
            }
            dq[[i, d]] = sum;
        }
    }

    // dK = d_scaled^T @ Q  [seq_k, d_k]
    let mut dk = Array2::<f64>::zeros((seq_k, d_k));
    for j in 0..seq_k {
        for d in 0..d_k {
            let mut sum = 0.0;
            for i in 0..seq_q {
                sum += d_scaled[[i, j]] * q[[i, d]];
            }
            dk[[j, d]] = sum;
        }
    }

    Ok(AttentionGradients { dq, dk, dv })
}

/// Backward pass through multi-head attention.
///
/// # Arguments
/// - `dout`: upstream gradient [batch, seq_q, model_dim]
/// - `query`, `key`, `value`: original inputs [batch, seq, model_dim]
/// - `all_weights`: attention weights [batch, n_heads, seq_q, seq_k] (from `forward_with_weights`)
/// - `config`: same `AttentionConfig` used for forward pass
pub fn multihead_attention_backward(
    dout: &Array3<f64>,
    query: &Array3<f64>,
    key: &Array3<f64>,
    value: &Array3<f64>,
    all_weights: &Array4<f64>,
    config: &AttentionConfig,
) -> Result<MultiHeadAttentionGrad, AttentionError> {
    let batch = dout.shape()[0];
    let seq_q = dout.shape()[1];
    let model_dim = dout.shape()[2];
    let seq_k = key.shape()[1];
    let n_heads = config.n_heads;
    let head_dim = config.head_dim;
    let expected_dim = n_heads * head_dim;

    if model_dim != expected_dim {
        return Err(AttentionError::ShapeMismatch(format!(
            "model_dim {} != n_heads*head_dim {}",
            model_dim, expected_dim
        )));
    }
    if query.shape()[0] != batch || key.shape()[0] != batch || value.shape()[0] != batch {
        return Err(AttentionError::ShapeMismatch(format!(
            "batch size mismatch: dout={}, query={}, key={}, value={}",
            batch,
            query.shape()[0],
            key.shape()[0],
            value.shape()[0]
        )));
    }
    if query.shape()[2] != expected_dim
        || key.shape()[2] != expected_dim
        || value.shape()[2] != expected_dim
    {
        return Err(AttentionError::ShapeMismatch(format!(
            "model_dim mismatch: query={}, key={}, value={}, expected={}",
            query.shape()[2],
            key.shape()[2],
            value.shape()[2],
            expected_dim
        )));
    }
    if all_weights.shape() != [batch, n_heads, seq_q, seq_k] {
        return Err(AttentionError::ShapeMismatch(format!(
            "weights shape {:?} != expected [{},{},{},{}]",
            all_weights.shape(),
            batch,
            n_heads,
            seq_q,
            seq_k
        )));
    }

    let scale = config.effective_scale();
    let mut dq_full = Array3::<f64>::zeros((batch, query.shape()[1], model_dim));
    let mut dk_full = Array3::<f64>::zeros((batch, seq_k, model_dim));
    let mut dv_full = Array3::<f64>::zeros((batch, seq_k, model_dim));

    for b in 0..batch {
        for h in 0..n_heads {
            let h_start = h * head_dim;
            let h_end = h_start + head_dim;

            // Extract per-head slices
            let dout_h = dout
                .slice(scirs2_core::ndarray::s![b, .., h_start..h_end])
                .to_owned();
            let q_h = query
                .slice(scirs2_core::ndarray::s![b, .., h_start..h_end])
                .to_owned();
            let k_h = key
                .slice(scirs2_core::ndarray::s![b, .., h_start..h_end])
                .to_owned();
            let v_h = value
                .slice(scirs2_core::ndarray::s![b, .., h_start..h_end])
                .to_owned();
            let w_h = all_weights
                .slice(scirs2_core::ndarray::s![b, h, .., ..])
                .to_owned();

            let grads = attention_backward(&dout_h, &q_h, &k_h, &v_h, &w_h, scale)?;

            // Accumulate into full gradients
            let mut dq_slice = dq_full.slice_mut(scirs2_core::ndarray::s![b, .., h_start..h_end]);
            dq_slice += &grads.dq;

            let mut dk_slice = dk_full.slice_mut(scirs2_core::ndarray::s![b, .., h_start..h_end]);
            dk_slice += &grads.dk;

            let mut dv_slice = dv_full.slice_mut(scirs2_core::ndarray::s![b, .., h_start..h_end]);
            dv_slice += &grads.dv;
        }
    }

    Ok(MultiHeadAttentionGrad {
        dq: dq_full,
        dk: dk_full,
        dv: dv_full,
    })
}

// ────────────────────────────────────────────────────────────────────────────
// Unit tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::{scaled_dot_product_attention, stable_softmax, AttentionConfig};
    use scirs2_core::ndarray::Array2;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn make_array2(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f64) -> Array2<f64> {
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(f(i, j));
            }
        }
        Array2::from_shape_vec((rows, cols), data).expect("shape ok")
    }

    fn make_array3(
        d0: usize,
        d1: usize,
        d2: usize,
        f: impl Fn(usize, usize, usize) -> f64,
    ) -> Array3<f64> {
        let mut data = Vec::with_capacity(d0 * d1 * d2);
        for b in 0..d0 {
            for i in 0..d1 {
                for j in 0..d2 {
                    data.push(f(b, i, j));
                }
            }
        }
        Array3::from_shape_vec((d0, d1, d2), data).expect("shape ok")
    }

    /// Numerical gradient check helper: computes finite-difference gradient of a scalar function.
    fn numerical_grad_check(
        f: impl Fn(&Array2<f64>) -> f64,
        x: &Array2<f64>,
        eps: f64,
    ) -> Array2<f64> {
        let mut grad = Array2::zeros(x.raw_dim());
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                let mut x_plus = x.clone();
                let mut x_minus = x.clone();
                x_plus[[i, j]] += eps;
                x_minus[[i, j]] -= eps;
                grad[[i, j]] = (f(&x_plus) - f(&x_minus)) / (2.0 * eps);
            }
        }
        grad
    }

    /// Compute the sum of all elements of the attention output (scalar loss)
    /// given Q, with fixed K, V, scale.
    fn attention_loss_wrt_q(q: &Array2<f64>, k: &Array2<f64>, v: &Array2<f64>, scale: f64) -> f64 {
        let (out, _) = scaled_dot_product_attention(q, k, v, scale, false).expect("forward ok");
        out.iter().sum()
    }

    fn attention_loss_wrt_k(q: &Array2<f64>, k: &Array2<f64>, v: &Array2<f64>, scale: f64) -> f64 {
        let (out, _) = scaled_dot_product_attention(q, k, v, scale, false).expect("forward ok");
        out.iter().sum()
    }

    fn attention_loss_wrt_v(q: &Array2<f64>, k: &Array2<f64>, v: &Array2<f64>, scale: f64) -> f64 {
        let (out, _) = scaled_dot_product_attention(q, k, v, scale, false).expect("forward ok");
        out.iter().sum()
    }

    // ── Softmax backward tests ──────────────────────────────────────────────

    #[test]
    fn test_softmax_backward_shape() {
        let p = make_array2(3, 5, |i, j| (i + j) as f64 * 0.2);
        let p = stable_softmax(&p);
        let dp = make_array2(3, 5, |i, j| (i * 5 + j) as f64 * 0.1);
        let result = softmax_backward(&p, &dp);
        assert_eq!(result.shape(), p.shape());
    }

    #[test]
    fn test_softmax_backward_uniform() {
        // Uniform p, all-ones dp => d_logits sums to 0 per row
        let n = 4;
        let p = make_array2(3, n, |_, _| 1.0 / n as f64);
        let dp = make_array2(3, n, |_, _| 1.0);
        let result = softmax_backward(&p, &dp);
        for row in result.rows() {
            let row_sum: f64 = row.iter().sum();
            assert!(
                row_sum.abs() < 1e-12,
                "row sum should be ~0, got {}",
                row_sum
            );
        }
    }

    #[test]
    fn test_softmax_backward_single_element() {
        // 1x1 softmax: p=[[1.0]], dp=[[3.0]]
        // d_logits = p * (dp - dot) = 1.0 * (3.0 - 3.0*1.0) = 0.0
        let p = Array2::from_shape_vec((1, 1), vec![1.0]).expect("ok");
        let dp = Array2::from_shape_vec((1, 1), vec![3.0]).expect("ok");
        let result = softmax_backward(&p, &dp);
        assert!((result[[0, 0]]).abs() < 1e-15);
    }

    #[test]
    fn test_softmax_backward_gradient_check() {
        // Numerical check: perturb logits, compare finite-diff softmax output with analytic
        let logits = make_array2(2, 3, |i, j| (i as f64 * 0.5 + j as f64 * 0.3) - 0.7);
        let eps = 1e-5;

        // We'll check gradient of a scalar: sum of all softmax outputs weighted by dp
        let dp = make_array2(2, 3, |i, j| (i * 3 + j + 1) as f64 * 0.2);
        let p = stable_softmax(&logits);
        let analytic = softmax_backward(&p, &dp);

        // Scalar loss = sum(dp * softmax(logits))
        let loss_fn = |x: &Array2<f64>| -> f64 {
            let s = stable_softmax(x);
            s.iter().zip(dp.iter()).map(|(a, b)| a * b).sum()
        };

        let numerical = numerical_grad_check(loss_fn, &logits, eps);

        for i in 0..2 {
            for j in 0..3 {
                let diff = (analytic[[i, j]] - numerical[[i, j]]).abs();
                assert!(
                    diff < 1e-5,
                    "softmax grad mismatch at [{},{}]: analytic={}, numerical={}",
                    i,
                    j,
                    analytic[[i, j]],
                    numerical[[i, j]]
                );
            }
        }
    }

    // ── Attention backward tests ────────────────────────────────────────────

    #[test]
    fn test_attention_backward_shape() {
        let seq_q = 3;
        let seq_k = 4;
        let d = 5;
        let q = make_array2(seq_q, d, |i, j| (i + j) as f64 * 0.1);
        let k = make_array2(seq_k, d, |i, j| (i + j) as f64 * 0.1);
        let v = make_array2(seq_k, d, |i, j| (i + j) as f64 * 0.1);
        let scale = 1.0 / (d as f64).sqrt();
        let (_, weights) = scaled_dot_product_attention(&q, &k, &v, scale, false).expect("fwd ok");
        let dout = make_array2(seq_q, d, |i, j| (i + j) as f64 * 0.01);

        let grads = attention_backward(&dout, &q, &k, &v, &weights, scale).expect("bwd ok");
        assert_eq!(grads.dq.shape(), &[seq_q, d]);
        assert_eq!(grads.dk.shape(), &[seq_k, d]);
        assert_eq!(grads.dv.shape(), &[seq_k, d]);
    }

    #[test]
    fn test_attention_backward_dv_manual() {
        // Manual 2x2 case: dV = W^T @ dout
        let weights = Array2::from_shape_vec((2, 2), vec![0.6, 0.4, 0.3, 0.7]).expect("ok");
        let dout = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("ok");
        let q = Array2::zeros((2, 2));
        let k = Array2::zeros((2, 2));
        let v = Array2::zeros((2, 2));

        let grads = attention_backward(&dout, &q, &k, &v, &weights, 1.0).expect("bwd ok");

        // dV = W^T @ dout
        // W^T = [[0.6, 0.3], [0.4, 0.7]]
        // dV[0,0] = 0.6*1.0 + 0.3*3.0 = 1.5
        // dV[0,1] = 0.6*2.0 + 0.3*4.0 = 2.4
        // dV[1,0] = 0.4*1.0 + 0.7*3.0 = 2.5
        // dV[1,1] = 0.4*2.0 + 0.7*4.0 = 3.6
        assert!((grads.dv[[0, 0]] - 1.5).abs() < 1e-10);
        assert!((grads.dv[[0, 1]] - 2.4).abs() < 1e-10);
        assert!((grads.dv[[1, 0]] - 2.5).abs() < 1e-10);
        assert!((grads.dv[[1, 1]] - 3.6).abs() < 1e-10);
    }

    #[test]
    fn test_attention_backward_zero_grad() {
        let seq_q = 3;
        let seq_k = 4;
        let d = 5;
        let q = make_array2(seq_q, d, |i, j| (i + j) as f64 * 0.1);
        let k = make_array2(seq_k, d, |i, j| (i + j) as f64 * 0.1);
        let v = make_array2(seq_k, d, |i, j| (i + j) as f64 * 0.1);
        let scale = 1.0 / (d as f64).sqrt();
        let (_, weights) = scaled_dot_product_attention(&q, &k, &v, scale, false).expect("fwd ok");
        let dout = Array2::<f64>::zeros((seq_q, d));

        let grads = attention_backward(&dout, &q, &k, &v, &weights, scale).expect("bwd ok");
        for &val in grads.dq.iter() {
            assert!(val.abs() < 1e-15, "dq should be zero, got {}", val);
        }
        for &val in grads.dk.iter() {
            assert!(val.abs() < 1e-15, "dk should be zero, got {}", val);
        }
        for &val in grads.dv.iter() {
            assert!(val.abs() < 1e-15, "dv should be zero, got {}", val);
        }
    }

    #[test]
    fn test_attention_backward_shape_mismatch() {
        let q = make_array2(3, 4, |_, _| 1.0);
        let k = make_array2(3, 5, |_, _| 1.0); // wrong dim
        let v = make_array2(3, 4, |_, _| 1.0);
        let weights = make_array2(3, 3, |_, _| 1.0 / 3.0);
        let dout = make_array2(3, 4, |_, _| 1.0);

        let result = attention_backward(&dout, &q, &k, &v, &weights, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_attention_backward_numerical_dq() {
        let seq_q = 2;
        let seq_k = 3;
        let d = 4;
        let q = make_array2(seq_q, d, |i, j| (i as f64 * 0.3 + j as f64 * 0.2) - 0.5);
        let k = make_array2(seq_k, d, |i, j| (i as f64 * 0.4 + j as f64 * 0.1) - 0.3);
        let v = make_array2(seq_k, d, |i, j| (i as f64 * 0.2 + j as f64 * 0.3) - 0.1);
        let scale = 1.0 / (d as f64).sqrt();

        let (_, weights) = scaled_dot_product_attention(&q, &k, &v, scale, false).expect("fwd ok");
        // dout = all ones (loss = sum of output)
        let dout = Array2::from_elem((seq_q, d), 1.0);

        let grads = attention_backward(&dout, &q, &k, &v, &weights, scale).expect("bwd ok");

        let k_c = k.clone();
        let v_c = v.clone();
        let numerical_dq = numerical_grad_check(
            |q_perturbed| attention_loss_wrt_q(q_perturbed, &k_c, &v_c, scale),
            &q,
            1e-5,
        );

        for i in 0..seq_q {
            for j in 0..d {
                let diff = (grads.dq[[i, j]] - numerical_dq[[i, j]]).abs();
                assert!(
                    diff < 1e-4,
                    "dQ mismatch at [{},{}]: analytic={}, numerical={}",
                    i,
                    j,
                    grads.dq[[i, j]],
                    numerical_dq[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_attention_backward_numerical_dk() {
        let seq_q = 2;
        let seq_k = 3;
        let d = 4;
        let q = make_array2(seq_q, d, |i, j| (i as f64 * 0.3 + j as f64 * 0.2) - 0.5);
        let k = make_array2(seq_k, d, |i, j| (i as f64 * 0.4 + j as f64 * 0.1) - 0.3);
        let v = make_array2(seq_k, d, |i, j| (i as f64 * 0.2 + j as f64 * 0.3) - 0.1);
        let scale = 1.0 / (d as f64).sqrt();

        let (_, weights) = scaled_dot_product_attention(&q, &k, &v, scale, false).expect("fwd ok");
        let dout = Array2::from_elem((seq_q, d), 1.0);

        let grads = attention_backward(&dout, &q, &k, &v, &weights, scale).expect("bwd ok");

        let q_c = q.clone();
        let v_c = v.clone();
        let numerical_dk = numerical_grad_check(
            |k_perturbed| attention_loss_wrt_k(&q_c, k_perturbed, &v_c, scale),
            &k,
            1e-5,
        );

        for i in 0..seq_k {
            for j in 0..d {
                let diff = (grads.dk[[i, j]] - numerical_dk[[i, j]]).abs();
                assert!(
                    diff < 1e-4,
                    "dK mismatch at [{},{}]: analytic={}, numerical={}",
                    i,
                    j,
                    grads.dk[[i, j]],
                    numerical_dk[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_attention_backward_numerical_dv() {
        let seq_q = 2;
        let seq_k = 3;
        let d = 4;
        let q = make_array2(seq_q, d, |i, j| (i as f64 * 0.3 + j as f64 * 0.2) - 0.5);
        let k = make_array2(seq_k, d, |i, j| (i as f64 * 0.4 + j as f64 * 0.1) - 0.3);
        let v = make_array2(seq_k, d, |i, j| (i as f64 * 0.2 + j as f64 * 0.3) - 0.1);
        let scale = 1.0 / (d as f64).sqrt();

        let (_, weights) = scaled_dot_product_attention(&q, &k, &v, scale, false).expect("fwd ok");
        let dout = Array2::from_elem((seq_q, d), 1.0);

        let grads = attention_backward(&dout, &q, &k, &v, &weights, scale).expect("bwd ok");

        let q_c = q.clone();
        let k_c = k.clone();
        let numerical_dv = numerical_grad_check(
            |v_perturbed| attention_loss_wrt_v(&q_c, &k_c, v_perturbed, scale),
            &v,
            1e-5,
        );

        for i in 0..seq_k {
            for j in 0..d {
                let diff = (grads.dv[[i, j]] - numerical_dv[[i, j]]).abs();
                assert!(
                    diff < 1e-4,
                    "dV mismatch at [{},{}]: analytic={}, numerical={}",
                    i,
                    j,
                    grads.dv[[i, j]],
                    numerical_dv[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_attention_backward_identity_weights() {
        // If weights = identity (no mixing), dV = dout for matching indices
        let n = 3;
        let d = 4;
        let mut weights = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            weights[[i, i]] = 1.0;
        }
        let dout = make_array2(n, d, |i, j| (i * d + j + 1) as f64 * 0.5);
        let q = Array2::zeros((n, d));
        let k = Array2::zeros((n, d));
        let v = Array2::zeros((n, d));

        let grads = attention_backward(&dout, &q, &k, &v, &weights, 1.0).expect("bwd ok");

        // dV = W^T @ dout = I @ dout = dout
        for i in 0..n {
            for j in 0..d {
                assert!(
                    (grads.dv[[i, j]] - dout[[i, j]]).abs() < 1e-12,
                    "dV should equal dout at [{},{}]",
                    i,
                    j
                );
            }
        }
    }

    // ── Multi-head backward tests ───────────────────────────────────────────

    #[test]
    fn test_multihead_backward_shape() {
        let batch = 2;
        let seq_q = 3;
        let seq_k = 3;
        let n_heads = 2;
        let head_dim = 4;
        let model_dim = n_heads * head_dim;

        let query = make_array3(batch, seq_q, model_dim, |b, i, j| (b + i + j) as f64 * 0.1);
        let key = make_array3(batch, seq_k, model_dim, |b, i, j| {
            (b * 2 + i + j) as f64 * 0.1
        });
        let value = make_array3(batch, seq_k, model_dim, |b, i, j| {
            (b + i + j + 1) as f64 * 0.1
        });
        let dout = make_array3(batch, seq_q, model_dim, |b, i, j| (b + i + j) as f64 * 0.01);

        let config = AttentionConfig::new(n_heads, head_dim);
        let mha = crate::attention::MultiHeadAttention::new(config.clone());
        let fwd = mha
            .forward_with_weights(&query, &key, &value)
            .expect("fwd ok");
        let all_weights = fwd.attention_weights.expect("weights present");

        let grads =
            multihead_attention_backward(&dout, &query, &key, &value, &all_weights, &config)
                .expect("bwd ok");

        assert_eq!(grads.dq.shape(), &[batch, seq_q, model_dim]);
        assert_eq!(grads.dk.shape(), &[batch, seq_k, model_dim]);
        assert_eq!(grads.dv.shape(), &[batch, seq_k, model_dim]);
    }

    #[test]
    fn test_multihead_backward_single_head() {
        // With n_heads=1, multihead backward should match single attention_backward
        let batch = 1;
        let seq = 3;
        let head_dim = 4;
        let model_dim = head_dim;

        let query = make_array3(batch, seq, model_dim, |_, i, j| {
            (i as f64 * 0.3 + j as f64 * 0.2) - 0.4
        });
        let key = make_array3(batch, seq, model_dim, |_, i, j| {
            (i as f64 * 0.4 + j as f64 * 0.1) - 0.2
        });
        let value = make_array3(batch, seq, model_dim, |_, i, j| {
            i as f64 * 0.1 + j as f64 * 0.3
        });
        let dout = make_array3(batch, seq, model_dim, |_, i, j| {
            (i * model_dim + j + 1) as f64 * 0.1
        });

        let config = AttentionConfig::new(1, head_dim);
        let scale = config.effective_scale();

        // Single head path
        let q2 = query.slice(scirs2_core::ndarray::s![0, .., ..]).to_owned();
        let k2 = key.slice(scirs2_core::ndarray::s![0, .., ..]).to_owned();
        let v2 = value.slice(scirs2_core::ndarray::s![0, .., ..]).to_owned();
        let dout2 = dout.slice(scirs2_core::ndarray::s![0, .., ..]).to_owned();
        let (_, weights) =
            scaled_dot_product_attention(&q2, &k2, &v2, scale, false).expect("fwd ok");
        let single_grads =
            attention_backward(&dout2, &q2, &k2, &v2, &weights, scale).expect("bwd ok");

        // Multi-head path
        let mha = crate::attention::MultiHeadAttention::new(config.clone());
        let fwd = mha
            .forward_with_weights(&query, &key, &value)
            .expect("fwd ok");
        let all_weights = fwd.attention_weights.expect("weights");
        let multi_grads =
            multihead_attention_backward(&dout, &query, &key, &value, &all_weights, &config)
                .expect("bwd ok");

        // Compare
        let multi_dq = multi_grads
            .dq
            .slice(scirs2_core::ndarray::s![0, .., ..])
            .to_owned();
        for i in 0..seq {
            for j in 0..model_dim {
                assert!(
                    (multi_dq[[i, j]] - single_grads.dq[[i, j]]).abs() < 1e-10,
                    "dQ mismatch at [{},{}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_multihead_backward_zero_grad() {
        let batch = 2;
        let seq = 3;
        let n_heads = 2;
        let head_dim = 4;
        let model_dim = n_heads * head_dim;

        let query = make_array3(batch, seq, model_dim, |b, i, j| (b + i + j) as f64 * 0.1);
        let key = make_array3(batch, seq, model_dim, |b, i, j| (b + i + j) as f64 * 0.1);
        let value = make_array3(batch, seq, model_dim, |b, i, j| (b + i + j) as f64 * 0.1);
        let dout = Array3::<f64>::zeros((batch, seq, model_dim));

        let config = AttentionConfig::new(n_heads, head_dim);
        let mha = crate::attention::MultiHeadAttention::new(config.clone());
        let fwd = mha
            .forward_with_weights(&query, &key, &value)
            .expect("fwd ok");
        let all_weights = fwd.attention_weights.expect("weights");

        let grads =
            multihead_attention_backward(&dout, &query, &key, &value, &all_weights, &config)
                .expect("bwd ok");

        for &val in grads.dq.iter() {
            assert!(val.abs() < 1e-15, "dq should be zero");
        }
        for &val in grads.dk.iter() {
            assert!(val.abs() < 1e-15, "dk should be zero");
        }
        for &val in grads.dv.iter() {
            assert!(val.abs() < 1e-15, "dv should be zero");
        }
    }

    #[test]
    fn test_multihead_backward_batch_independence() {
        let batch = 2;
        let seq = 3;
        let n_heads = 2;
        let head_dim = 4;
        let model_dim = n_heads * head_dim;

        let query = make_array3(batch, seq, model_dim, |b, i, j| {
            (b * 100 + i + j) as f64 * 0.1
        });
        let key = make_array3(batch, seq, model_dim, |b, i, j| {
            (b * 50 + i + j) as f64 * 0.1
        });
        let value = make_array3(batch, seq, model_dim, |b, i, j| {
            (b * 30 + i + j + 1) as f64 * 0.1
        });

        // dout only nonzero for batch 0
        let mut dout = Array3::<f64>::zeros((batch, seq, model_dim));
        for i in 0..seq {
            for j in 0..model_dim {
                dout[[0, i, j]] = (i * model_dim + j + 1) as f64 * 0.1;
            }
        }

        let config = AttentionConfig::new(n_heads, head_dim);
        let mha = crate::attention::MultiHeadAttention::new(config.clone());
        let fwd = mha
            .forward_with_weights(&query, &key, &value)
            .expect("fwd ok");
        let all_weights = fwd.attention_weights.expect("weights");

        let grads =
            multihead_attention_backward(&dout, &query, &key, &value, &all_weights, &config)
                .expect("bwd ok");

        // Batch 1 gradients should be zero since dout[1] is zero
        for i in 0..seq {
            for j in 0..model_dim {
                assert!(
                    grads.dq[[1, i, j]].abs() < 1e-15,
                    "batch 1 dq should be zero at [{},{}]",
                    i,
                    j
                );
                assert!(
                    grads.dk[[1, i, j]].abs() < 1e-15,
                    "batch 1 dk should be zero at [{},{}]",
                    i,
                    j
                );
                assert!(
                    grads.dv[[1, i, j]].abs() < 1e-15,
                    "batch 1 dv should be zero at [{},{}]",
                    i,
                    j
                );
            }
        }

        // Batch 0 gradients should be nonzero
        let has_nonzero_dq = grads
            .dq
            .slice(scirs2_core::ndarray::s![0, .., ..])
            .iter()
            .any(|&v| v.abs() > 1e-10);
        assert!(has_nonzero_dq, "batch 0 dq should have nonzero values");
    }

    #[test]
    fn test_softmax_backward_peaked() {
        // Peaked softmax: one large value should result in near-zero gradient for others
        let logits = make_array2(1, 4, |_, j| if j == 0 { 100.0 } else { 0.0 });
        let p = stable_softmax(&logits);
        let dp = make_array2(1, 4, |_, _| 1.0);
        let result = softmax_backward(&p, &dp);

        // Since p is nearly [1,0,0,0], the gradient for the peaked element should be ~0
        // (because p*(dp - dot) where dot ~ dp[0]*1 = 1, so 1*(1-1) = 0)
        // and for non-peaked elements: p_j * (dp_j - dot) ~ 0 * (1-1) = 0
        for j in 0..4 {
            assert!(
                result[[0, j]].abs() < 1e-6,
                "peaked softmax gradient at [0,{}] = {}, expected ~0",
                j,
                result[[0, j]]
            );
        }
    }

    #[test]
    fn test_attention_backward_scale_factor() {
        let seq = 3;
        let d = 4;
        let q = make_array2(seq, d, |i, j| (i as f64 * 0.3 + j as f64 * 0.2) - 0.4);
        let k = make_array2(seq, d, |i, j| (i as f64 * 0.4 + j as f64 * 0.1) - 0.2);
        let v = make_array2(seq, d, |i, j| i as f64 * 0.1 + j as f64 * 0.3);
        let dout = make_array2(seq, d, |i, j| (i + j + 1) as f64 * 0.1);

        let scale1 = 1.0;
        let scale2 = 0.5;

        let (_, w1) = scaled_dot_product_attention(&q, &k, &v, scale1, false).expect("fwd ok");
        let (_, w2) = scaled_dot_product_attention(&q, &k, &v, scale2, false).expect("fwd ok");

        let grads1 = attention_backward(&dout, &q, &k, &v, &w1, scale1).expect("bwd ok");
        let grads2 = attention_backward(&dout, &q, &k, &v, &w2, scale2).expect("bwd ok");

        // Different scales should produce different gradient magnitudes
        let norm1: f64 = grads1.dq.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = grads2.dq.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (norm1 - norm2).abs() > 1e-6,
            "different scales should produce different gradient magnitudes: {} vs {}",
            norm1,
            norm2
        );
    }

    #[test]
    fn test_attention_backward_symmetric_qk() {
        // When Q == K and dout is uniform, dQ and dK should be approximately equal
        let seq = 3;
        let d = 4;
        let qk = make_array2(seq, d, |i, j| (i as f64 * 0.3 + j as f64 * 0.2) - 0.4);
        let v = make_array2(seq, d, |i, j| i as f64 * 0.1 + j as f64 * 0.3);
        let scale = 1.0 / (d as f64).sqrt();
        let dout = Array2::from_elem((seq, d), 1.0);

        let (_, weights) =
            scaled_dot_product_attention(&qk, &qk, &v, scale, false).expect("fwd ok");
        let grads = attention_backward(&dout, &qk, &qk, &v, &weights, scale).expect("bwd ok");

        // When Q==K and dout is uniform, dQ and dK norms should be similar
        // (not exactly equal because V breaks the symmetry between the Q and K roles)
        let dq_norm: f64 = grads.dq.iter().map(|x| x * x).sum::<f64>().sqrt();
        let dk_norm: f64 = grads.dk.iter().map(|x| x * x).sum::<f64>().sqrt();
        let relative_diff = (dq_norm - dk_norm).abs() / (dq_norm + dk_norm + 1e-15);
        assert!(
            relative_diff < 0.5,
            "dQ and dK norms should be similar when Q==K: dQ_norm={}, dK_norm={}, rel_diff={}",
            dq_norm,
            dk_norm,
            relative_diff
        );
    }

    #[test]
    fn test_attention_gradient_finite_values() {
        let seq_q = 4;
        let seq_k = 5;
        let d = 6;
        let q = make_array2(seq_q, d, |i, j| ((i + 1) * (j + 1)) as f64 * 0.05 - 0.7);
        let k = make_array2(seq_k, d, |i, j| (i as f64 * 0.3 - j as f64 * 0.1) * 0.5);
        let v = make_array2(seq_k, d, |i, j| ((i * 2 + j) as f64).sin());
        let scale = 1.0 / (d as f64).sqrt();
        let dout = make_array2(seq_q, d, |i, j| ((i + j) as f64 * 0.3).cos());

        let (_, weights) = scaled_dot_product_attention(&q, &k, &v, scale, false).expect("fwd ok");
        let grads = attention_backward(&dout, &q, &k, &v, &weights, scale).expect("bwd ok");

        for &val in grads.dq.iter() {
            assert!(val.is_finite(), "dQ contains non-finite value: {}", val);
        }
        for &val in grads.dk.iter() {
            assert!(val.is_finite(), "dK contains non-finite value: {}", val);
        }
        for &val in grads.dv.iter() {
            assert!(val.is_finite(), "dV contains non-finite value: {}", val);
        }
    }
}
