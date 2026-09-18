//! Gated Recurrent Unit (GRU) and Long Short-Term Memory (LSTM) layers.
//!
//! Both layers operate in **inference mode only** — they evaluate a full input
//! sequence and return the output sequence plus the final hidden (and cell, for
//! LSTM) state.  No gradient computation is performed.
//!
//! ## Shapes
//!
//! | Input | Shape |
//! |-------|-------|
//! | Sequence tensor | `[T, input_size]` where T is sequence length |
//! | Initial hidden state | `[hidden_size]` (zero-filled if not provided) |
//! | Initial cell state (LSTM) | `[hidden_size]` (zero-filled if not provided) |
//!
//! All weight matrices are stored in row-major order following the PyTorch
//! convention:
//!
//! * **GRU**: `W_ih [3*H, I]`, `W_hh [3*H, H]`, `b_ih [3*H]`, `b_hh [3*H]`
//! * **LSTM**: `W_ih [4*H, I]`, `W_hh [4*H, H]`, `b_ih [4*H]`, `b_hh [4*H]`
//!
//! where `H` = `hidden_size` and `I` = `input_size`.

use crate::error::NeuralError;
use crate::tensor::Tensor;

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

#[inline]
fn sigmoid_f(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn tanh_f(x: f32) -> f32 {
    x.tanh()
}

/// Dense matrix-vector product: `y = W * x` where W is `[rows, cols]`,
/// `x` is length `cols`, result is length `rows`.
fn matvec(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    let mut out = vec![0.0_f32; rows];
    for r in 0..rows {
        let mut acc = 0.0_f32;
        let row_start = r * cols;
        for c in 0..cols {
            acc += w[row_start + c] * x[c];
        }
        out[r] = acc;
    }
    out
}

/// Xavier uniform initialisation — deterministic LCG, no external crate.
///
/// Produces `n` values uniformly distributed in `[-limit, limit]` where
/// `limit = sqrt(6 / (fan_in + fan_out))`.  The LCG is seeded from the
/// shape parameters so different weight matrices get different values.
fn xavier_uniform(fan_in: usize, fan_out: usize, n: usize) -> Vec<f32> {
    let limit = ((6.0_f32) / (fan_in + fan_out) as f32).sqrt();
    // LCG parameters (Knuth MMIX)
    let mut state: u64 = (fan_in as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(fan_out as u64 ^ (n as u64).wrapping_add(1442695040888963407));
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map high 32 bits to [0, 1)
        let bits = (state >> 32) as u32;
        let frac = bits as f32 / (u32::MAX as f32 + 1.0);
        out.push(frac * 2.0 * limit - limit);
    }
    out
}

/// Element-wise vector add (in-place: `a += b`).
fn vec_add_inplace(a: &mut [f32], b: &[f32]) {
    for (ai, &bi) in a.iter_mut().zip(b.iter()) {
        *ai += bi;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// GruCell — single time-step
// ──────────────────────────────────────────────────────────────────────────────

/// Single GRU time-step: given x_t and h_{t-1}, produces h_t.
///
/// Gate layout (each gate is `hidden_size` wide):
/// * gates 0..H   → reset gate r
/// * gates H..2H  → update gate z
/// * gates 2H..3H → new gate n
fn gru_cell(
    x_t: &[f32],
    h_prev: &[f32],
    w_ih: &[f32],
    w_hh: &[f32],
    b_ih: &[f32],
    b_hh: &[f32],
    input_size: usize,
    hidden_size: usize,
) -> Vec<f32> {
    let h3 = 3 * hidden_size;

    // Linear transforms.
    let mut gates_ih = matvec(w_ih, x_t, h3, input_size);
    vec_add_inplace(&mut gates_ih, b_ih);

    let mut gates_hh = matvec(w_hh, h_prev, h3, hidden_size);
    vec_add_inplace(&mut gates_hh, b_hh);

    // r gate: sigmoid(gates_ih[0..H] + gates_hh[0..H])
    let mut r = vec![0.0_f32; hidden_size];
    for i in 0..hidden_size {
        r[i] = sigmoid_f(gates_ih[i] + gates_hh[i]);
    }

    // z gate: sigmoid(gates_ih[H..2H] + gates_hh[H..2H])
    let mut z = vec![0.0_f32; hidden_size];
    for i in 0..hidden_size {
        z[i] = sigmoid_f(gates_ih[hidden_size + i] + gates_hh[hidden_size + i]);
    }

    // n gate: tanh(gates_ih[2H..3H] + r ⊙ gates_hh[2H..3H])
    let mut n = vec![0.0_f32; hidden_size];
    for i in 0..hidden_size {
        n[i] = tanh_f(gates_ih[2 * hidden_size + i] + r[i] * gates_hh[2 * hidden_size + i]);
    }

    // h_t = (1 - z) ⊙ n + z ⊙ h_prev
    let mut h_new = vec![0.0_f32; hidden_size];
    for i in 0..hidden_size {
        h_new[i] = (1.0 - z[i]) * n[i] + z[i] * h_prev[i];
    }
    h_new
}

// ──────────────────────────────────────────────────────────────────────────────
// GruLayer
// ──────────────────────────────────────────────────────────────────────────────

/// A single-layer GRU operating on a variable-length sequence.
///
/// # Example
///
/// ```rust
/// use oximedia_neural::recurrent::GruLayer;
/// use oximedia_neural::tensor::Tensor;
///
/// let gru = GruLayer::new(4, 8).unwrap();
/// let seq = Tensor::zeros(vec![5, 4]).unwrap(); // 5 timesteps, 4 features
/// let (outputs, h_n) = gru.forward(&seq, None).unwrap();
/// assert_eq!(outputs.shape(), &[5, 8]);
/// assert_eq!(h_n.shape(), &[8]);
/// ```
#[derive(Debug, Clone)]
pub struct GruLayer {
    /// Input-hidden weight matrix, shape `[3*hidden_size, input_size]`.
    pub w_ih: Vec<f32>,
    /// Hidden-hidden weight matrix, shape `[3*hidden_size, hidden_size]`.
    pub w_hh: Vec<f32>,
    /// Input-hidden bias, shape `[3*hidden_size]`.
    pub b_ih: Vec<f32>,
    /// Hidden-hidden bias, shape `[3*hidden_size]`.
    pub b_hh: Vec<f32>,
    /// Number of input features.
    pub input_size: usize,
    /// Number of hidden units.
    pub hidden_size: usize,
}

impl GruLayer {
    /// Creates a Xavier-initialised `GruLayer` (weights) with zero biases.
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self, NeuralError> {
        if input_size == 0 || hidden_size == 0 {
            return Err(NeuralError::InvalidShape(
                "GruLayer: input_size and hidden_size must be > 0".to_string(),
            ));
        }
        let h3 = 3 * hidden_size;
        Ok(Self {
            w_ih: xavier_uniform(input_size, hidden_size, h3 * input_size),
            w_hh: xavier_uniform(hidden_size, hidden_size, h3 * hidden_size),
            b_ih: vec![0.0_f32; h3],
            b_hh: vec![0.0_f32; h3],
            input_size,
            hidden_size,
        })
    }

    /// Forward pass over a sequence.
    ///
    /// * `input` — `[T, input_size]` tensor (T timesteps).
    /// * `h_0` — optional initial hidden state `[hidden_size]`; zero if `None`.
    ///
    /// Returns `(outputs, h_n)`:
    /// * `outputs` — `[T, hidden_size]` tensor of all hidden states.
    /// * `h_n` — `[hidden_size]` final hidden state.
    pub fn forward(
        &self,
        input: &Tensor,
        h_0: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor), NeuralError> {
        if input.ndim() != 2 {
            return Err(NeuralError::InvalidShape(format!(
                "GruLayer::forward: expected 2-D input [T, I], got rank {}",
                input.ndim()
            )));
        }
        let (t, feat) = (input.shape()[0], input.shape()[1]);
        if feat != self.input_size {
            return Err(NeuralError::ShapeMismatch(format!(
                "GruLayer::forward: input feature dim {} != input_size {}",
                feat, self.input_size
            )));
        }

        // Validate / unpack h_0.
        let mut h = match h_0 {
            Some(h) => {
                if h.ndim() != 1 || h.numel() != self.hidden_size {
                    return Err(NeuralError::ShapeMismatch(format!(
                        "GruLayer::forward: h_0 must be [hidden_size={}], got {:?}",
                        self.hidden_size,
                        h.shape()
                    )));
                }
                h.data().to_vec()
            }
            None => vec![0.0_f32; self.hidden_size],
        };

        let mut all_hidden = Vec::with_capacity(t * self.hidden_size);

        for step in 0..t {
            let x_t = &input.data()[step * self.input_size..(step + 1) * self.input_size];
            h = gru_cell(
                x_t,
                &h,
                &self.w_ih,
                &self.w_hh,
                &self.b_ih,
                &self.b_hh,
                self.input_size,
                self.hidden_size,
            );
            all_hidden.extend_from_slice(&h);
        }

        let outputs = Tensor::from_data(all_hidden, vec![t, self.hidden_size])?;
        let h_n = Tensor::from_data(h, vec![self.hidden_size])?;
        Ok((outputs, h_n))
    }

    /// Single time-step GRU forward pass operating on raw slices.
    ///
    /// * `input`  — flat slice of length `input_size`.
    /// * `hidden` — flat slice of length `hidden_size` (h_{t-1}).
    ///
    /// Returns the new hidden state `h_t` as a `Vec<f32>` of length `hidden_size`.
    pub fn forward_step(&self, input: &[f32], hidden: &[f32]) -> Result<Vec<f32>, NeuralError> {
        if input.len() != self.input_size {
            return Err(NeuralError::ShapeMismatch(format!(
                "GruLayer::forward_step: input len {} != input_size {}",
                input.len(),
                self.input_size
            )));
        }
        if hidden.len() != self.hidden_size {
            return Err(NeuralError::ShapeMismatch(format!(
                "GruLayer::forward_step: hidden len {} != hidden_size {}",
                hidden.len(),
                self.hidden_size
            )));
        }
        Ok(gru_cell(
            input,
            hidden,
            &self.w_ih,
            &self.w_hh,
            &self.b_ih,
            &self.b_hh,
            self.input_size,
            self.hidden_size,
        ))
    }

    /// Process a full sequence using raw flat slices.
    ///
    /// * `inputs`         — flat slice of length `seq_len * input_size` (row-major).
    /// * `seq_len`        — number of time steps.
    /// * `initial_hidden` — optional initial hidden state `[hidden_size]`; zero if `None`.
    ///
    /// Returns `(all_hidden_states, final_hidden)` where:
    /// * `all_hidden_states` has length `seq_len * hidden_size`.
    /// * `final_hidden` has length `hidden_size`.
    pub fn forward_sequence(
        &self,
        inputs: &[f32],
        seq_len: usize,
        initial_hidden: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Vec<f32>), NeuralError> {
        let expected = seq_len * self.input_size;
        if inputs.len() != expected {
            return Err(NeuralError::ShapeMismatch(format!(
                "GruLayer::forward_sequence: inputs len {} != seq_len({}) * input_size({})",
                inputs.len(),
                seq_len,
                self.input_size
            )));
        }
        let mut h = match initial_hidden {
            Some(ih) => {
                if ih.len() != self.hidden_size {
                    return Err(NeuralError::ShapeMismatch(format!(
                        "GruLayer::forward_sequence: initial_hidden len {} != hidden_size {}",
                        ih.len(),
                        self.hidden_size
                    )));
                }
                ih.to_vec()
            }
            None => vec![0.0_f32; self.hidden_size],
        };
        let mut all_hidden = Vec::with_capacity(seq_len * self.hidden_size);
        for step in 0..seq_len {
            let x_t = &inputs[step * self.input_size..(step + 1) * self.input_size];
            h = gru_cell(
                x_t,
                &h,
                &self.w_ih,
                &self.w_hh,
                &self.b_ih,
                &self.b_hh,
                self.input_size,
                self.hidden_size,
            );
            all_hidden.extend_from_slice(&h);
        }
        Ok((all_hidden, h))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// LstmCell — single time-step
// ──────────────────────────────────────────────────────────────────────────────

/// Single LSTM time-step: given x_t, h_{t-1}, c_{t-1} → (h_t, c_t).
///
/// Gate layout (each gate is `hidden_size` wide):
/// * 0..H    → input gate i
/// * H..2H   → forget gate f
/// * 2H..3H  → cell gate g
/// * 3H..4H  → output gate o
fn lstm_cell(
    x_t: &[f32],
    h_prev: &[f32],
    c_prev: &[f32],
    w_ih: &[f32],
    w_hh: &[f32],
    b_ih: &[f32],
    b_hh: &[f32],
    input_size: usize,
    hidden_size: usize,
) -> (Vec<f32>, Vec<f32>) {
    let h4 = 4 * hidden_size;

    let mut gates = matvec(w_ih, x_t, h4, input_size);
    let gates_hh = matvec(w_hh, h_prev, h4, hidden_size);
    vec_add_inplace(&mut gates, b_ih);
    vec_add_inplace(&mut gates, &gates_hh);
    vec_add_inplace(&mut gates, b_hh);

    let mut c_new = vec![0.0_f32; hidden_size];
    let mut h_new = vec![0.0_f32; hidden_size];

    for j in 0..hidden_size {
        let ig = sigmoid_f(gates[j]);
        let fg = sigmoid_f(gates[hidden_size + j]);
        let gg = tanh_f(gates[2 * hidden_size + j]);
        let og = sigmoid_f(gates[3 * hidden_size + j]);

        c_new[j] = fg * c_prev[j] + ig * gg;
        h_new[j] = og * tanh_f(c_new[j]);
    }

    (h_new, c_new)
}

// ──────────────────────────────────────────────────────────────────────────────
// LstmLayer
// ──────────────────────────────────────────────────────────────────────────────

/// A single-layer LSTM operating on a variable-length sequence.
///
/// # Example
///
/// ```rust
/// use oximedia_neural::recurrent::LstmLayer;
/// use oximedia_neural::tensor::Tensor;
///
/// let lstm = LstmLayer::new(4, 8).unwrap();
/// let seq = Tensor::zeros(vec![5, 4]).unwrap();
/// let (outputs, h_n, c_n) = lstm.forward(&seq, None, None).unwrap();
/// assert_eq!(outputs.shape(), &[5, 8]);
/// assert_eq!(h_n.shape(), &[8]);
/// assert_eq!(c_n.shape(), &[8]);
/// ```
#[derive(Debug, Clone)]
pub struct LstmLayer {
    /// Input-hidden weight matrix, shape `[4*hidden_size, input_size]`.
    pub w_ih: Vec<f32>,
    /// Hidden-hidden weight matrix, shape `[4*hidden_size, hidden_size]`.
    pub w_hh: Vec<f32>,
    /// Input-hidden bias, shape `[4*hidden_size]`.
    pub b_ih: Vec<f32>,
    /// Hidden-hidden bias, shape `[4*hidden_size]`.
    pub b_hh: Vec<f32>,
    /// Number of input features.
    pub input_size: usize,
    /// Number of hidden units.
    pub hidden_size: usize,
}

impl LstmLayer {
    /// Creates a Xavier-initialised `LstmLayer` (weights) with zero biases.
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self, NeuralError> {
        if input_size == 0 || hidden_size == 0 {
            return Err(NeuralError::InvalidShape(
                "LstmLayer: input_size and hidden_size must be > 0".to_string(),
            ));
        }
        let h4 = 4 * hidden_size;
        Ok(Self {
            w_ih: xavier_uniform(input_size, hidden_size, h4 * input_size),
            w_hh: xavier_uniform(hidden_size, hidden_size, h4 * hidden_size),
            b_ih: vec![0.0_f32; h4],
            b_hh: vec![0.0_f32; h4],
            input_size,
            hidden_size,
        })
    }

    /// Forward pass over a sequence.
    ///
    /// * `input` — `[T, input_size]` tensor.
    /// * `h_0` — optional initial hidden state `[hidden_size]`.
    /// * `c_0` — optional initial cell state `[hidden_size]`.
    ///
    /// Returns `(outputs, h_n, c_n)`:
    /// * `outputs` — `[T, hidden_size]` tensor of all hidden states.
    /// * `h_n` — `[hidden_size]` final hidden state.
    /// * `c_n` — `[hidden_size]` final cell state.
    pub fn forward(
        &self,
        input: &Tensor,
        h_0: Option<&Tensor>,
        c_0: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor), NeuralError> {
        if input.ndim() != 2 {
            return Err(NeuralError::InvalidShape(format!(
                "LstmLayer::forward: expected 2-D input [T, I], got rank {}",
                input.ndim()
            )));
        }
        let (t, feat) = (input.shape()[0], input.shape()[1]);
        if feat != self.input_size {
            return Err(NeuralError::ShapeMismatch(format!(
                "LstmLayer::forward: input feature dim {} != input_size {}",
                feat, self.input_size
            )));
        }

        let unpack_state = |s: Option<&Tensor>, name: &str| -> Result<Vec<f32>, NeuralError> {
            match s {
                Some(t) => {
                    if t.ndim() != 1 || t.numel() != self.hidden_size {
                        return Err(NeuralError::ShapeMismatch(format!(
                            "LstmLayer::forward: {name} must be [hidden_size={}], got {:?}",
                            self.hidden_size,
                            t.shape()
                        )));
                    }
                    Ok(t.data().to_vec())
                }
                None => Ok(vec![0.0_f32; self.hidden_size]),
            }
        };

        let mut h = unpack_state(h_0, "h_0")?;
        let mut c = unpack_state(c_0, "c_0")?;

        let mut all_hidden = Vec::with_capacity(t * self.hidden_size);

        for step in 0..t {
            let x_t = &input.data()[step * self.input_size..(step + 1) * self.input_size];
            let (h_new, c_new) = lstm_cell(
                x_t,
                &h,
                &c,
                &self.w_ih,
                &self.w_hh,
                &self.b_ih,
                &self.b_hh,
                self.input_size,
                self.hidden_size,
            );
            h = h_new;
            c = c_new;
            all_hidden.extend_from_slice(&h);
        }

        let outputs = Tensor::from_data(all_hidden, vec![t, self.hidden_size])?;
        let h_n = Tensor::from_data(h, vec![self.hidden_size])?;
        let c_n = Tensor::from_data(c, vec![self.hidden_size])?;
        Ok((outputs, h_n, c_n))
    }

    /// Single time-step LSTM forward pass operating on raw slices.
    ///
    /// * `input`  — flat slice of length `input_size`.
    /// * `hidden` — flat slice of length `hidden_size` (h_{t-1}).
    /// * `cell`   — flat slice of length `hidden_size` (c_{t-1}).
    ///
    /// Returns `(new_hidden, new_cell)` each of length `hidden_size`.
    pub fn forward_step(
        &self,
        input: &[f32],
        hidden: &[f32],
        cell: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), NeuralError> {
        if input.len() != self.input_size {
            return Err(NeuralError::ShapeMismatch(format!(
                "LstmLayer::forward_step: input len {} != input_size {}",
                input.len(),
                self.input_size
            )));
        }
        if hidden.len() != self.hidden_size {
            return Err(NeuralError::ShapeMismatch(format!(
                "LstmLayer::forward_step: hidden len {} != hidden_size {}",
                hidden.len(),
                self.hidden_size
            )));
        }
        if cell.len() != self.hidden_size {
            return Err(NeuralError::ShapeMismatch(format!(
                "LstmLayer::forward_step: cell len {} != hidden_size {}",
                cell.len(),
                self.hidden_size
            )));
        }
        Ok(lstm_cell(
            input,
            hidden,
            cell,
            &self.w_ih,
            &self.w_hh,
            &self.b_ih,
            &self.b_hh,
            self.input_size,
            self.hidden_size,
        ))
    }

    /// Process a full sequence using raw flat slices.
    ///
    /// * `inputs`        — flat slice of length `seq_len * input_size` (row-major).
    /// * `seq_len`       — number of time steps.
    /// * `initial_hidden` — optional `[hidden_size]`; zero if `None`.
    /// * `initial_cell`  — optional `[hidden_size]`; zero if `None`.
    ///
    /// Returns `(all_hidden_states, final_hidden, final_cell)` where
    /// `all_hidden_states` has length `seq_len * hidden_size`.
    pub fn forward_sequence(
        &self,
        inputs: &[f32],
        seq_len: usize,
        initial_hidden: Option<&[f32]>,
        initial_cell: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), NeuralError> {
        let expected = seq_len * self.input_size;
        if inputs.len() != expected {
            return Err(NeuralError::ShapeMismatch(format!(
                "LstmLayer::forward_sequence: inputs len {} != seq_len({}) * input_size({})",
                inputs.len(),
                seq_len,
                self.input_size
            )));
        }
        let unpack = |s: Option<&[f32]>, name: &str| -> Result<Vec<f32>, NeuralError> {
            match s {
                Some(v) => {
                    if v.len() != self.hidden_size {
                        return Err(NeuralError::ShapeMismatch(format!(
                            "LstmLayer::forward_sequence: {name} len {} != hidden_size {}",
                            v.len(),
                            self.hidden_size
                        )));
                    }
                    Ok(v.to_vec())
                }
                None => Ok(vec![0.0_f32; self.hidden_size]),
            }
        };
        let mut h = unpack(initial_hidden, "initial_hidden")?;
        let mut c = unpack(initial_cell, "initial_cell")?;
        let mut all_hidden = Vec::with_capacity(seq_len * self.hidden_size);
        for step in 0..seq_len {
            let x_t = &inputs[step * self.input_size..(step + 1) * self.input_size];
            let (h_new, c_new) = lstm_cell(
                x_t,
                &h,
                &c,
                &self.w_ih,
                &self.w_hh,
                &self.b_ih,
                &self.b_hh,
                self.input_size,
                self.hidden_size,
            );
            h = h_new;
            c = c_new;
            all_hidden.extend_from_slice(&h);
        }
        Ok((all_hidden, h, c))
    }
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

    // ── GruLayer ──────────────────────────────────────────────────────────────

    #[test]
    fn test_gru_output_shape() {
        let gru = GruLayer::new(4, 8).expect("gru layer new");
        let seq = Tensor::zeros(vec![5, 4]).expect("tensor zeros");
        let (out, h_n) = gru.forward(&seq, None).expect("forward pass");
        assert_eq!(out.shape(), &[5, 8]);
        assert_eq!(h_n.shape(), &[8]);
    }

    #[test]
    fn test_gru_zero_weights_deterministic() {
        // With all-zero weights/biases and zero input, h_t should be all-zero
        // for every step: z=sigmoid(0)=0.5, r=0.5, n=tanh(0)=0, h=(1-0.5)*0+0.5*0=0
        let mut gru = GruLayer::new(2, 4).expect("gru layer new");
        // Explicitly zero out weights so Xavier init does not affect the assertion.
        gru.w_ih.iter_mut().for_each(|v| *v = 0.0);
        gru.w_hh.iter_mut().for_each(|v| *v = 0.0);
        let seq = Tensor::zeros(vec![3, 2]).expect("tensor zeros");
        let (out, h_n) = gru.forward(&seq, None).expect("forward pass");
        assert!(out.data().iter().all(|&v| close(v, 0.0)));
        assert!(h_n.data().iter().all(|&v| close(v, 0.0)));
    }

    #[test]
    fn test_gru_custom_h0() {
        let mut gru = GruLayer::new(2, 4).expect("gru layer new");
        // Zero weights so analysis is tractable: h_t = z * h_{t-1} = 0.5 * 1 = 0.5
        gru.w_ih.iter_mut().for_each(|v| *v = 0.0);
        gru.w_hh.iter_mut().for_each(|v| *v = 0.0);
        let seq = Tensor::zeros(vec![1, 2]).expect("tensor zeros");
        let h0 = Tensor::from_data(vec![1.0; 4], vec![4]).expect("tensor from_data");
        let (_, h_n) = gru.forward(&seq, Some(&h0)).expect("forward pass");
        // With zero weights and non-zero h_0, h_t = z * h_{t-1} = 0.5 * 1 = 0.5
        assert!(h_n.data().iter().all(|&v| close(v, 0.5)));
    }

    #[test]
    fn test_gru_wrong_input_rank() {
        let gru = GruLayer::new(4, 8).expect("gru layer new");
        let bad = Tensor::zeros(vec![5, 4, 1]).expect("tensor zeros");
        assert!(gru.forward(&bad, None).is_err());
    }

    #[test]
    fn test_gru_wrong_input_size() {
        let gru = GruLayer::new(4, 8).expect("gru layer new");
        let bad = Tensor::zeros(vec![5, 3]).expect("tensor zeros");
        assert!(gru.forward(&bad, None).is_err());
    }

    #[test]
    fn test_gru_h0_wrong_shape() {
        let gru = GruLayer::new(4, 8).expect("gru layer new");
        let seq = Tensor::zeros(vec![5, 4]).expect("tensor zeros");
        let bad_h0 = Tensor::zeros(vec![3]).expect("tensor zeros");
        assert!(gru.forward(&seq, Some(&bad_h0)).is_err());
    }

    #[test]
    fn test_gru_zero_sizes_error() {
        assert!(GruLayer::new(0, 8).is_err());
        assert!(GruLayer::new(4, 0).is_err());
    }

    #[test]
    fn test_gru_single_step() {
        let gru = GruLayer::new(1, 1).expect("gru layer new");
        let seq = Tensor::from_data(vec![1.0], vec![1, 1]).expect("tensor from_data");
        let (out, _h_n) = gru.forward(&seq, None).expect("forward pass");
        assert_eq!(out.shape(), &[1, 1]);
        // result is finite
        assert!(out.data()[0].is_finite());
    }

    #[test]
    fn test_gru_output_values_finite() {
        let mut gru = GruLayer::new(3, 5).expect("gru layer new");
        // Simple non-trivial weights.
        for (i, w) in gru.w_ih.iter_mut().enumerate() {
            *w = (i as f32 * 0.01) - 0.05;
        }
        let seq_data: Vec<f32> = (0..10 * 3).map(|i| (i as f32) * 0.1 - 1.5).collect();
        let seq = Tensor::from_data(seq_data, vec![10, 3]).expect("tensor from_data");
        let (out, h_n) = gru.forward(&seq, None).expect("forward pass");
        assert!(out.data().iter().all(|v| v.is_finite()));
        assert!(h_n.data().iter().all(|v| v.is_finite()));
    }

    // ── LstmLayer ─────────────────────────────────────────────────────────────

    #[test]
    fn test_lstm_output_shape() {
        let lstm = LstmLayer::new(4, 8).expect("lstm layer new");
        let seq = Tensor::zeros(vec![5, 4]).expect("tensor zeros");
        let (out, h_n, c_n) = lstm.forward(&seq, None, None).expect("forward pass");
        assert_eq!(out.shape(), &[5, 8]);
        assert_eq!(h_n.shape(), &[8]);
        assert_eq!(c_n.shape(), &[8]);
    }

    #[test]
    fn test_lstm_zero_weights_deterministic() {
        // With all-zero weights: i=sigmoid(0)=0.5, f=0.5, g=tanh(0)=0, o=0.5
        // c_t = f*0 + i*g = 0.5*0 = 0; h_t = o*tanh(0) = 0
        let mut lstm = LstmLayer::new(2, 4).expect("lstm layer new");
        lstm.w_ih.iter_mut().for_each(|v| *v = 0.0);
        lstm.w_hh.iter_mut().for_each(|v| *v = 0.0);
        let seq = Tensor::zeros(vec![3, 2]).expect("tensor zeros");
        let (out, h_n, c_n) = lstm.forward(&seq, None, None).expect("forward pass");
        assert!(out.data().iter().all(|&v| close(v, 0.0)));
        assert!(h_n.data().iter().all(|&v| close(v, 0.0)));
        assert!(c_n.data().iter().all(|&v| close(v, 0.0)));
    }

    #[test]
    fn test_lstm_custom_h0_c0() {
        let lstm = LstmLayer::new(2, 4).expect("lstm layer new");
        let seq = Tensor::zeros(vec![1, 2]).expect("tensor zeros");
        let h0 = Tensor::from_data(vec![0.5; 4], vec![4]).expect("tensor from_data");
        let c0 = Tensor::from_data(vec![1.0; 4], vec![4]).expect("tensor from_data");
        let (_out, _h_n, _c_n) = lstm
            .forward(&seq, Some(&h0), Some(&c0))
            .expect("forward pass");
        // Should succeed without error.
    }

    #[test]
    fn test_lstm_wrong_input_rank() {
        let lstm = LstmLayer::new(4, 8).expect("lstm layer new");
        let bad = Tensor::zeros(vec![5]).expect("tensor zeros");
        assert!(lstm.forward(&bad, None, None).is_err());
    }

    #[test]
    fn test_lstm_wrong_input_size() {
        let lstm = LstmLayer::new(4, 8).expect("lstm layer new");
        let bad = Tensor::zeros(vec![5, 3]).expect("tensor zeros");
        assert!(lstm.forward(&bad, None, None).is_err());
    }

    #[test]
    fn test_lstm_zero_sizes_error() {
        assert!(LstmLayer::new(0, 8).is_err());
        assert!(LstmLayer::new(4, 0).is_err());
    }

    #[test]
    fn test_lstm_output_values_finite() {
        let mut lstm = LstmLayer::new(3, 5).expect("lstm layer new");
        for (i, w) in lstm.w_ih.iter_mut().enumerate() {
            *w = (i as f32 * 0.01) - 0.05;
        }
        let seq_data: Vec<f32> = (0..10 * 3).map(|i| (i as f32) * 0.1 - 1.5).collect();
        let seq = Tensor::from_data(seq_data, vec![10, 3]).expect("tensor from_data");
        let (out, h_n, c_n) = lstm.forward(&seq, None, None).expect("forward pass");
        assert!(out.data().iter().all(|v| v.is_finite()));
        assert!(h_n.data().iter().all(|v| v.is_finite()));
        assert!(c_n.data().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_lstm_cell_forget_gate_memorizes() {
        // Set forget gate bias to a large positive value (forget gate ≈ 1)
        // and input/output gate biases to large values too.
        // Start with c_0 = [1.0]; after one step with large forget gate,
        // c_1 should be close to 1.0 (memory preserved).
        let mut lstm = LstmLayer::new(1, 1).expect("lstm layer new");
        // Zero weights so only biases determine gate activations.
        lstm.w_ih.iter_mut().for_each(|v| *v = 0.0);
        lstm.w_hh.iter_mut().for_each(|v| *v = 0.0);
        // b_ih layout: [i, f, g, o] each of size hidden_size=1
        lstm.b_ih[0] = -10.0; // i → ~0
        lstm.b_ih[1] = 10.0; // f → ~1
        lstm.b_ih[2] = 0.0; // g → 0
        lstm.b_ih[3] = 10.0; // o → ~1

        let seq = Tensor::zeros(vec![1, 1]).expect("tensor zeros");
        let c0 = Tensor::from_data(vec![1.0], vec![1]).expect("tensor from_data");
        let (_out, _h_n, c_n) = lstm.forward(&seq, None, Some(&c0)).expect("forward pass");
        // c_1 = sigmoid(10)*1 + sigmoid(-10)*0 ≈ 1*1 = 1
        let c_val = c_n.data()[0];
        assert!(c_val > 0.9, "forget gate should preserve cell: got {c_val}");
    }

    // ── forward_step / forward_sequence — GruLayer ────────────────────────────

    #[test]
    fn test_gru_forward_step_matches_sequence() {
        // Verify that calling forward_step manually for each step produces
        // the same final hidden state as forward_sequence.
        let gru = GruLayer::new(3, 5).expect("gru layer new");
        let seq_len = 4usize;
        let inputs: Vec<f32> = (0..seq_len * 3).map(|i| (i as f32) * 0.05 - 0.3).collect();

        // Manual step-by-step.
        let mut h = vec![0.0_f32; 5];
        for step in 0..seq_len {
            let x = &inputs[step * 3..(step + 1) * 3];
            h = gru.forward_step(x, &h).expect("forward_step");
        }

        // forward_sequence.
        let (_all_h, final_h) = gru
            .forward_sequence(&inputs, seq_len, None)
            .expect("forward_sequence");

        for (a, b) in h.iter().zip(final_h.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "step vs sequence mismatch: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_gru_forward_sequence_output_length() {
        let gru = GruLayer::new(2, 6).expect("gru layer new");
        let seq_len = 7usize;
        let inputs = vec![0.1_f32; seq_len * 2];
        let (all_h, final_h) = gru
            .forward_sequence(&inputs, seq_len, None)
            .expect("forward_sequence");
        assert_eq!(all_h.len(), seq_len * 6, "all_hidden length mismatch");
        assert_eq!(final_h.len(), 6, "final_hidden length mismatch");
        // final hidden should equal the last frame of all_hidden.
        let last_frame = &all_h[(seq_len - 1) * 6..];
        assert_eq!(last_frame, final_h.as_slice());
    }

    #[test]
    fn test_gru_forward_step_wrong_input_len() {
        let gru = GruLayer::new(4, 8).expect("gru layer new");
        let h = vec![0.0_f32; 8];
        assert!(gru.forward_step(&[0.0; 3], &h).is_err()); // wrong input_size
    }

    #[test]
    fn test_gru_forward_step_wrong_hidden_len() {
        let gru = GruLayer::new(4, 8).expect("gru layer new");
        assert!(gru.forward_step(&[0.0; 4], &[0.0; 5]).is_err()); // wrong hidden_size
    }

    #[test]
    fn test_gru_forward_sequence_wrong_inputs_len() {
        let gru = GruLayer::new(3, 4).expect("gru layer new");
        // 5 steps * 3 features = 15, but we pass 14.
        assert!(gru.forward_sequence(&vec![0.0_f32; 14], 5, None).is_err());
    }

    // ── forward_step / forward_sequence — LstmLayer ───────────────────────────

    #[test]
    fn test_lstm_forward_step_matches_sequence() {
        let lstm = LstmLayer::new(3, 5).expect("lstm layer new");
        let seq_len = 4usize;
        let inputs: Vec<f32> = (0..seq_len * 3).map(|i| (i as f32) * 0.05 - 0.3).collect();

        // Manual step-by-step.
        let mut h = vec![0.0_f32; 5];
        let mut c = vec![0.0_f32; 5];
        for step in 0..seq_len {
            let x = &inputs[step * 3..(step + 1) * 3];
            let (h_new, c_new) = lstm.forward_step(x, &h, &c).expect("forward_step");
            h = h_new;
            c = c_new;
        }

        // forward_sequence.
        let (_all_h, final_h, final_c) = lstm
            .forward_sequence(&inputs, seq_len, None, None)
            .expect("forward_sequence");

        for (a, b) in h.iter().zip(final_h.iter()) {
            assert!((a - b).abs() < 1e-5, "hidden mismatch: {a} vs {b}");
        }
        for (a, b) in c.iter().zip(final_c.iter()) {
            assert!((a - b).abs() < 1e-5, "cell mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_lstm_forward_sequence_output_length() {
        let lstm = LstmLayer::new(2, 6).expect("lstm layer new");
        let seq_len = 7usize;
        let inputs = vec![0.1_f32; seq_len * 2];
        let (all_h, final_h, final_c) = lstm
            .forward_sequence(&inputs, seq_len, None, None)
            .expect("forward_sequence");
        assert_eq!(all_h.len(), seq_len * 6);
        assert_eq!(final_h.len(), 6);
        assert_eq!(final_c.len(), 6);
        // Last frame of all_h equals final_h.
        let last_frame = &all_h[(seq_len - 1) * 6..];
        assert_eq!(last_frame, final_h.as_slice());
    }

    #[test]
    fn test_lstm_forward_step_wrong_input_len() {
        let lstm = LstmLayer::new(4, 8).expect("lstm layer new");
        assert!(lstm.forward_step(&[0.0; 3], &[0.0; 8], &[0.0; 8]).is_err());
    }

    #[test]
    fn test_lstm_forward_step_wrong_hidden_len() {
        let lstm = LstmLayer::new(4, 8).expect("lstm layer new");
        assert!(lstm.forward_step(&[0.0; 4], &[0.0; 5], &[0.0; 8]).is_err());
    }

    #[test]
    fn test_lstm_forward_step_wrong_cell_len() {
        let lstm = LstmLayer::new(4, 8).expect("lstm layer new");
        assert!(lstm.forward_step(&[0.0; 4], &[0.0; 8], &[0.0; 5]).is_err());
    }

    #[test]
    fn test_lstm_forward_sequence_wrong_inputs_len() {
        let lstm = LstmLayer::new(3, 4).expect("lstm layer new");
        assert!(lstm
            .forward_sequence(&vec![0.0_f32; 14], 5, None, None)
            .is_err());
    }

    // ── Xavier initialisation ──────────────────────────────────────────────────

    #[test]
    fn test_xavier_uniform_range() {
        let fan_in = 4usize;
        let fan_out = 8usize;
        let weights = xavier_uniform(fan_in, fan_out, 100);
        let limit = ((6.0_f32) / (fan_in + fan_out) as f32).sqrt();
        for &w in &weights {
            assert!(
                w >= -limit && w <= limit,
                "weight {w} out of Xavier range [-{limit}, {limit}]"
            );
        }
    }

    #[test]
    fn test_xavier_weights_not_all_zero() {
        // GruLayer::new should produce non-zero weights via Xavier init.
        let gru = GruLayer::new(4, 8).expect("gru layer new");
        let any_nonzero = gru.w_ih.iter().any(|&v| v != 0.0);
        assert!(
            any_nonzero,
            "Xavier-initialised weights should not all be zero"
        );
    }
}
