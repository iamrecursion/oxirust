//! Recurrent neural network cells: RNN, LSTM, GRU.
//!
//! Provides cell-level and sequence-level forward pass implementations for the
//! three most common recurrent architectures.  All weights are stored as plain
//! `ndarray` arrays so they can be loaded from external checkpoints or
//! initialised with the built-in deterministic LCG scheme.
//!
//! ## Cell types
//! - [`RnnCell`]  – vanilla tanh-RNN
//! - [`LstmCell`] – Long Short-Term Memory (LSTM)
//! - [`GruCell`]  – Gated Recurrent Unit (GRU)
//!
//! ## Sequence helpers
//! - [`rnn_sequence`]  – run `RnnCell` over a slice of inputs
//! - [`lstm_sequence`] – run `LstmCell` over a slice of inputs
//! - [`gru_sequence`]  – run `GruCell` over a slice of inputs

use scirs2_core::ndarray::{Array1, Array2};

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise from recurrent cell operations.
#[derive(Debug, Clone)]
pub enum RecurrentError {
    /// A matrix or vector had the wrong shape.
    ShapeMismatch {
        /// The shape that was expected.
        expected: Vec<usize>,
        /// The shape that was actually provided.
        got: Vec<usize>,
    },
    /// `hidden_size` was zero or otherwise invalid.
    InvalidHiddenSize(usize),
    /// `input_size` was zero or otherwise invalid.
    InvalidInputSize(usize),
    /// The input sequence had length zero.
    EmptySequence,
    /// The input sequence length was invalid for some reason.
    InvalidSequenceLength {
        /// The problematic length that was encountered.
        got: usize,
    },
}

impl std::fmt::Display for RecurrentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecurrentError::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            RecurrentError::InvalidHiddenSize(s) => {
                write!(f, "invalid hidden_size: {s}")
            }
            RecurrentError::InvalidInputSize(s) => {
                write!(f, "invalid input_size: {s}")
            }
            RecurrentError::EmptySequence => {
                write!(f, "input sequence must not be empty")
            }
            RecurrentError::InvalidSequenceLength { got } => {
                write!(f, "invalid sequence length: {got}")
            }
        }
    }
}

impl std::error::Error for RecurrentError {}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Sigmoid activation: σ(x) = 1 / (1 + e^{-x}).
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Deterministic LCG-based pseudo-random value in [-scale, scale].
///
/// Uses the same constants as the rest of the crate so that different modules
/// produce consistent-looking weight matrices.
#[inline]
fn lcg_value(state: &mut u64, scale: f64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005_u64)
        .wrapping_add(1442695040888963407_u64);
    // Map [0, 2^64) → [-scale, scale]
    let normalised = (*state as f64) / (u64::MAX as f64); // [0, 1]
    (normalised * 2.0 - 1.0) * scale
}

/// Fill an `Array2<f64>` with LCG-generated values in [-scale, scale].
fn lcg_fill_2d(rows: usize, cols: usize, scale: f64, state: &mut u64) -> Array2<f64> {
    let data: Vec<f64> = (0..rows * cols).map(|_| lcg_value(state, scale)).collect();
    // unwrap-free: we construct the vec with exactly rows*cols elements
    Array2::from_shape_vec((rows, cols), data).unwrap_or_else(|_| Array2::zeros((rows, cols)))
}

/// Fill an `Array1<f64>` with LCG-generated values in [-scale, scale].
fn lcg_fill_1d(len: usize, scale: f64, state: &mut u64) -> Array1<f64> {
    let data: Vec<f64> = (0..len).map(|_| lcg_value(state, scale)).collect();
    Array1::from_vec(data)
}

// ─────────────────────────────────────────────────────────────────────────────
// RnnCell
// ─────────────────────────────────────────────────────────────────────────────

/// Vanilla RNN cell.
///
/// Computes: `h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)`
#[derive(Debug, Clone)]
pub struct RnnCell {
    /// Number of input features.
    pub input_size: usize,
    /// Number of hidden units.
    pub hidden_size: usize,
    /// Input-hidden weight matrix `[hidden_size, input_size]`.
    pub w_ih: Array2<f64>,
    /// Hidden-hidden weight matrix `[hidden_size, hidden_size]`.
    pub w_hh: Array2<f64>,
    /// Input-hidden bias `[hidden_size]`.
    pub b_ih: Array1<f64>,
    /// Hidden-hidden bias `[hidden_size]`.
    pub b_hh: Array1<f64>,
}

impl RnnCell {
    /// Construct a new RNN cell with small deterministic random weights.
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self, RecurrentError> {
        if input_size == 0 {
            return Err(RecurrentError::InvalidInputSize(input_size));
        }
        if hidden_size == 0 {
            return Err(RecurrentError::InvalidHiddenSize(hidden_size));
        }
        let scale = 0.1_f64;
        let mut state: u64 = 0xdeadbeef_12345678_u64;
        let w_ih = lcg_fill_2d(hidden_size, input_size, scale, &mut state);
        let w_hh = lcg_fill_2d(hidden_size, hidden_size, scale, &mut state);
        let b_ih = lcg_fill_1d(hidden_size, scale, &mut state);
        let b_hh = lcg_fill_1d(hidden_size, scale, &mut state);
        Ok(Self {
            input_size,
            hidden_size,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
        })
    }

    /// Construct an RNN cell from pre-existing weight arrays.
    pub fn from_weights(
        w_ih: Array2<f64>,
        w_hh: Array2<f64>,
        b_ih: Array1<f64>,
        b_hh: Array1<f64>,
    ) -> Result<Self, RecurrentError> {
        let hidden_size = w_ih.nrows();
        let input_size = w_ih.ncols();
        if hidden_size == 0 {
            return Err(RecurrentError::InvalidHiddenSize(hidden_size));
        }
        if input_size == 0 {
            return Err(RecurrentError::InvalidInputSize(input_size));
        }
        // Validate w_hh shape
        if w_hh.nrows() != hidden_size || w_hh.ncols() != hidden_size {
            return Err(RecurrentError::ShapeMismatch {
                expected: vec![hidden_size, hidden_size],
                got: vec![w_hh.nrows(), w_hh.ncols()],
            });
        }
        if b_ih.len() != hidden_size {
            return Err(RecurrentError::ShapeMismatch {
                expected: vec![hidden_size],
                got: vec![b_ih.len()],
            });
        }
        if b_hh.len() != hidden_size {
            return Err(RecurrentError::ShapeMismatch {
                expected: vec![hidden_size],
                got: vec![b_hh.len()],
            });
        }
        Ok(Self {
            input_size,
            hidden_size,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
        })
    }

    /// Run one step forward.
    ///
    /// # Arguments
    /// * `input`  – shape `[input_size]`
    /// * `hidden` – shape `[hidden_size]`
    ///
    /// # Returns
    /// New hidden state, shape `[hidden_size]`.
    pub fn forward(
        &self,
        input: &Array1<f64>,
        hidden: &Array1<f64>,
    ) -> Result<Array1<f64>, RecurrentError> {
        if input.len() != self.input_size {
            return Err(RecurrentError::ShapeMismatch {
                expected: vec![self.input_size],
                got: vec![input.len()],
            });
        }
        if hidden.len() != self.hidden_size {
            return Err(RecurrentError::ShapeMismatch {
                expected: vec![self.hidden_size],
                got: vec![hidden.len()],
            });
        }
        // h_t = tanh(W_ih @ x + b_ih + W_hh @ h + b_hh)
        let pre_act = self.w_ih.dot(input) + &self.b_ih + self.w_hh.dot(hidden) + &self.b_hh;
        Ok(pre_act.mapv(f64::tanh))
    }

    /// Return an all-zeros initial hidden state.
    pub fn init_hidden(&self) -> Array1<f64> {
        Array1::zeros(self.hidden_size)
    }

    /// Total number of learnable parameters.
    pub fn num_parameters(&self) -> usize {
        self.hidden_size * self.input_size    // w_ih
            + self.hidden_size * self.hidden_size // w_hh
            + self.hidden_size                    // b_ih
            + self.hidden_size // b_hh
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LstmState
// ─────────────────────────────────────────────────────────────────────────────

/// Combined hidden and cell state for an LSTM.
#[derive(Debug, Clone)]
pub struct LstmState {
    /// Hidden state `h`, shape `[hidden_size]`.
    pub h: Array1<f64>,
    /// Cell state `c`, shape `[hidden_size]`.
    pub c: Array1<f64>,
}

impl LstmState {
    /// Create an all-zeros initial state.
    pub fn zeros(hidden_size: usize) -> Self {
        Self {
            h: Array1::zeros(hidden_size),
            c: Array1::zeros(hidden_size),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LstmCell
// ─────────────────────────────────────────────────────────────────────────────

/// LSTM cell with combined gate weight matrices.
///
/// Weight row ordering: `[input_gate, forget_gate, cell_gate, output_gate]`.
///
/// Forward pass:
/// ```text
/// i = σ(W_ii @ x + b_ii + W_hi @ h + b_hi)   [rows 0   .. h]
/// f = σ(W_if @ x + b_if + W_hf @ h + b_hf)   [rows h   .. 2h]
/// g = tanh(W_ig @ x + b_ig + W_hg @ h + b_hg) [rows 2h .. 3h]
/// o = σ(W_io @ x + b_io + W_ho @ h + b_ho)   [rows 3h .. 4h]
/// c' = f ⊙ c + i ⊙ g
/// h' = o ⊙ tanh(c')
/// ```
#[derive(Debug, Clone)]
pub struct LstmCell {
    /// Number of input features.
    pub input_size: usize,
    /// Number of hidden units.
    pub hidden_size: usize,
    /// Combined input-hidden weight matrix `[4*hidden_size, input_size]`.
    pub w_ih: Array2<f64>,
    /// Combined hidden-hidden weight matrix `[4*hidden_size, hidden_size]`.
    pub w_hh: Array2<f64>,
    /// Combined input-hidden bias `[4*hidden_size]`.
    pub b_ih: Array1<f64>,
    /// Combined hidden-hidden bias `[4*hidden_size]`.
    pub b_hh: Array1<f64>,
}

impl LstmCell {
    /// Construct a new LSTM cell with small deterministic random weights.
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self, RecurrentError> {
        if input_size == 0 {
            return Err(RecurrentError::InvalidInputSize(input_size));
        }
        if hidden_size == 0 {
            return Err(RecurrentError::InvalidHiddenSize(hidden_size));
        }
        let scale = 0.1_f64;
        let mut state: u64 = 0xfeedface_abcd1234_u64;
        let gates = 4;
        let w_ih = lcg_fill_2d(gates * hidden_size, input_size, scale, &mut state);
        let w_hh = lcg_fill_2d(gates * hidden_size, hidden_size, scale, &mut state);
        let b_ih = lcg_fill_1d(gates * hidden_size, scale, &mut state);
        let b_hh = lcg_fill_1d(gates * hidden_size, scale, &mut state);
        Ok(Self {
            input_size,
            hidden_size,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
        })
    }

    /// Construct an LSTM cell from pre-existing weight arrays.
    pub fn from_weights(
        w_ih: Array2<f64>,
        w_hh: Array2<f64>,
        b_ih: Array1<f64>,
        b_hh: Array1<f64>,
    ) -> Result<Self, RecurrentError> {
        let input_size = w_ih.ncols();
        if input_size == 0 {
            return Err(RecurrentError::InvalidInputSize(input_size));
        }
        let combined_rows = w_ih.nrows();
        if combined_rows == 0 || !combined_rows.is_multiple_of(4) {
            return Err(RecurrentError::ShapeMismatch {
                expected: vec![0 /* 4*h */, input_size],
                got: vec![combined_rows, input_size],
            });
        }
        let hidden_size = combined_rows / 4;
        // Validate all other tensors
        if w_hh.nrows() != combined_rows || w_hh.ncols() != hidden_size {
            return Err(RecurrentError::ShapeMismatch {
                expected: vec![combined_rows, hidden_size],
                got: vec![w_hh.nrows(), w_hh.ncols()],
            });
        }
        if b_ih.len() != combined_rows {
            return Err(RecurrentError::ShapeMismatch {
                expected: vec![combined_rows],
                got: vec![b_ih.len()],
            });
        }
        if b_hh.len() != combined_rows {
            return Err(RecurrentError::ShapeMismatch {
                expected: vec![combined_rows],
                got: vec![b_hh.len()],
            });
        }
        Ok(Self {
            input_size,
            hidden_size,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
        })
    }

    /// Run one LSTM step.
    ///
    /// # Arguments
    /// * `input` – shape `[input_size]`
    /// * `state` – current `(h, c)` state
    ///
    /// # Returns
    /// Updated `LstmState`.
    pub fn forward(
        &self,
        input: &Array1<f64>,
        state: &LstmState,
    ) -> Result<LstmState, RecurrentError> {
        if input.len() != self.input_size {
            return Err(RecurrentError::ShapeMismatch {
                expected: vec![self.input_size],
                got: vec![input.len()],
            });
        }
        if state.h.len() != self.hidden_size {
            return Err(RecurrentError::ShapeMismatch {
                expected: vec![self.hidden_size],
                got: vec![state.h.len()],
            });
        }
        if state.c.len() != self.hidden_size {
            return Err(RecurrentError::ShapeMismatch {
                expected: vec![self.hidden_size],
                got: vec![state.c.len()],
            });
        }

        // Combined pre-activations: shape [4*hidden]
        let gates_pre = self.w_ih.dot(input) + &self.b_ih + self.w_hh.dot(&state.h) + &self.b_hh;

        let h = self.hidden_size;

        // Slice into per-gate vectors (views → owned)
        let i_pre = gates_pre.slice(scirs2_core::ndarray::s![..h]).to_owned();
        let f_pre = gates_pre
            .slice(scirs2_core::ndarray::s![h..2 * h])
            .to_owned();
        let g_pre = gates_pre
            .slice(scirs2_core::ndarray::s![2 * h..3 * h])
            .to_owned();
        let o_pre = gates_pre
            .slice(scirs2_core::ndarray::s![3 * h..])
            .to_owned();

        let i_gate = i_pre.mapv(sigmoid);
        let f_gate = f_pre.mapv(sigmoid);
        let g_gate = g_pre.mapv(f64::tanh);
        let o_gate = o_pre.mapv(sigmoid);

        // c' = f ⊙ c + i ⊙ g
        let new_c = &f_gate * &state.c + &i_gate * &g_gate;
        // h' = o ⊙ tanh(c')
        let new_h = &o_gate * new_c.mapv(f64::tanh);

        Ok(LstmState { h: new_h, c: new_c })
    }

    /// Return an all-zeros initial state.
    pub fn init_state(&self) -> LstmState {
        LstmState::zeros(self.hidden_size)
    }

    /// Total number of learnable parameters.
    pub fn num_parameters(&self) -> usize {
        let gates = 4;
        gates * self.hidden_size * self.input_size    // w_ih
            + gates * self.hidden_size * self.hidden_size // w_hh
            + gates * self.hidden_size                    // b_ih
            + gates * self.hidden_size // b_hh
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GruCell
// ─────────────────────────────────────────────────────────────────────────────

/// GRU cell with combined gate weight matrices.
///
/// Weight row ordering: `[reset_gate, update_gate, new_gate]`.
///
/// Forward pass:
/// ```text
/// r = σ(W_ir @ x + b_ir + W_hr @ h + b_hr)           [rows 0   .. h]
/// z = σ(W_iz @ x + b_iz + W_hz @ h + b_hz)           [rows h   .. 2h]
/// n = tanh(W_in @ x + b_in + r ⊙ (W_hn @ h + b_hn)) [rows 2h .. 3h]
/// h' = (1 - z) ⊙ n + z ⊙ h
/// ```
#[derive(Debug, Clone)]
pub struct GruCell {
    /// Number of input features.
    pub input_size: usize,
    /// Number of hidden units.
    pub hidden_size: usize,
    /// Combined input-hidden weight matrix `[3*hidden_size, input_size]`.
    pub w_ih: Array2<f64>,
    /// Combined hidden-hidden weight matrix `[3*hidden_size, hidden_size]`.
    pub w_hh: Array2<f64>,
    /// Combined input-hidden bias `[3*hidden_size]`.
    pub b_ih: Array1<f64>,
    /// Combined hidden-hidden bias `[3*hidden_size]`.
    pub b_hh: Array1<f64>,
}

impl GruCell {
    /// Construct a new GRU cell with small deterministic random weights.
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self, RecurrentError> {
        if input_size == 0 {
            return Err(RecurrentError::InvalidInputSize(input_size));
        }
        if hidden_size == 0 {
            return Err(RecurrentError::InvalidHiddenSize(hidden_size));
        }
        let scale = 0.1_f64;
        let mut state: u64 = 0xc0ffee00_87654321_u64;
        let gates = 3;
        let w_ih = lcg_fill_2d(gates * hidden_size, input_size, scale, &mut state);
        let w_hh = lcg_fill_2d(gates * hidden_size, hidden_size, scale, &mut state);
        let b_ih = lcg_fill_1d(gates * hidden_size, scale, &mut state);
        let b_hh = lcg_fill_1d(gates * hidden_size, scale, &mut state);
        Ok(Self {
            input_size,
            hidden_size,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
        })
    }

    /// Construct a GRU cell from pre-existing weight arrays.
    pub fn from_weights(
        w_ih: Array2<f64>,
        w_hh: Array2<f64>,
        b_ih: Array1<f64>,
        b_hh: Array1<f64>,
    ) -> Result<Self, RecurrentError> {
        let input_size = w_ih.ncols();
        if input_size == 0 {
            return Err(RecurrentError::InvalidInputSize(input_size));
        }
        let combined_rows = w_ih.nrows();
        if combined_rows == 0 || !combined_rows.is_multiple_of(3) {
            return Err(RecurrentError::ShapeMismatch {
                expected: vec![0 /* 3*h */, input_size],
                got: vec![combined_rows, input_size],
            });
        }
        let hidden_size = combined_rows / 3;
        if w_hh.nrows() != combined_rows || w_hh.ncols() != hidden_size {
            return Err(RecurrentError::ShapeMismatch {
                expected: vec![combined_rows, hidden_size],
                got: vec![w_hh.nrows(), w_hh.ncols()],
            });
        }
        if b_ih.len() != combined_rows {
            return Err(RecurrentError::ShapeMismatch {
                expected: vec![combined_rows],
                got: vec![b_ih.len()],
            });
        }
        if b_hh.len() != combined_rows {
            return Err(RecurrentError::ShapeMismatch {
                expected: vec![combined_rows],
                got: vec![b_hh.len()],
            });
        }
        Ok(Self {
            input_size,
            hidden_size,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
        })
    }

    /// Run one GRU step.
    ///
    /// # Arguments
    /// * `input`  – shape `[input_size]`
    /// * `hidden` – shape `[hidden_size]`
    ///
    /// # Returns
    /// New hidden state, shape `[hidden_size]`.
    pub fn forward(
        &self,
        input: &Array1<f64>,
        hidden: &Array1<f64>,
    ) -> Result<Array1<f64>, RecurrentError> {
        if input.len() != self.input_size {
            return Err(RecurrentError::ShapeMismatch {
                expected: vec![self.input_size],
                got: vec![input.len()],
            });
        }
        if hidden.len() != self.hidden_size {
            return Err(RecurrentError::ShapeMismatch {
                expected: vec![self.hidden_size],
                got: vec![hidden.len()],
            });
        }

        let h = self.hidden_size;

        // Input-side pre-activations: [3h]
        let x_pre = self.w_ih.dot(input) + &self.b_ih;
        // Hidden-side pre-activations: [3h]
        let h_pre = self.w_hh.dot(hidden) + &self.b_hh;

        // Reset and update gates use the sum of both sides
        let r_pre = x_pre.slice(scirs2_core::ndarray::s![..h]).to_owned()
            + h_pre.slice(scirs2_core::ndarray::s![..h]).to_owned();
        let z_pre = x_pre.slice(scirs2_core::ndarray::s![h..2 * h]).to_owned()
            + h_pre.slice(scirs2_core::ndarray::s![h..2 * h]).to_owned();

        let r_gate = r_pre.mapv(sigmoid);
        let z_gate = z_pre.mapv(sigmoid);

        // New gate: x part + r ⊙ h part
        let n_x = x_pre.slice(scirs2_core::ndarray::s![2 * h..]).to_owned();
        let n_h = h_pre.slice(scirs2_core::ndarray::s![2 * h..]).to_owned();
        let n_pre = n_x + &r_gate * n_h;
        let n_gate = n_pre.mapv(f64::tanh);

        // h' = (1 - z) ⊙ n + z ⊙ h
        let ones = Array1::<f64>::ones(h);
        let new_h = (&ones - &z_gate) * &n_gate + &z_gate * hidden;
        Ok(new_h)
    }

    /// Return an all-zeros initial hidden state.
    pub fn init_hidden(&self) -> Array1<f64> {
        Array1::zeros(self.hidden_size)
    }

    /// Total number of learnable parameters.
    pub fn num_parameters(&self) -> usize {
        let gates = 3;
        gates * self.hidden_size * self.input_size    // w_ih
            + gates * self.hidden_size * self.hidden_size // w_hh
            + gates * self.hidden_size                    // b_ih
            + gates * self.hidden_size // b_hh
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sequence helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Run an [`RnnCell`] over a sequence of inputs.
///
/// # Arguments
/// * `cell`   – configured RNN cell
/// * `inputs` – slice of length `T`, each element shape `[input_size]`
///
/// # Returns
/// `Vec` of `T` hidden states, each shape `[hidden_size]`.
pub fn rnn_sequence(
    cell: &RnnCell,
    inputs: &[Array1<f64>],
) -> Result<Vec<Array1<f64>>, RecurrentError> {
    if inputs.is_empty() {
        return Err(RecurrentError::EmptySequence);
    }
    let mut hidden = cell.init_hidden();
    let mut outputs = Vec::with_capacity(inputs.len());
    for x in inputs {
        hidden = cell.forward(x, &hidden)?;
        outputs.push(hidden.clone());
    }
    Ok(outputs)
}

/// Run an [`LstmCell`] over a sequence of inputs.
///
/// # Arguments
/// * `cell`   – configured LSTM cell
/// * `inputs` – slice of length `T`, each element shape `[input_size]`
///
/// # Returns
/// `(all_hidden_states, final_state)` where `all_hidden_states` has length `T`.
pub fn lstm_sequence(
    cell: &LstmCell,
    inputs: &[Array1<f64>],
) -> Result<(Vec<Array1<f64>>, LstmState), RecurrentError> {
    if inputs.is_empty() {
        return Err(RecurrentError::EmptySequence);
    }
    let mut state = cell.init_state();
    let mut hidden_states = Vec::with_capacity(inputs.len());
    for x in inputs {
        state = cell.forward(x, &state)?;
        hidden_states.push(state.h.clone());
    }
    Ok((hidden_states, state))
}

/// Run a [`GruCell`] over a sequence of inputs.
///
/// # Arguments
/// * `cell`   – configured GRU cell
/// * `inputs` – slice of length `T`, each element shape `[input_size]`
///
/// # Returns
/// `Vec` of `T` hidden states, each shape `[hidden_size]`.
pub fn gru_sequence(
    cell: &GruCell,
    inputs: &[Array1<f64>],
) -> Result<Vec<Array1<f64>>, RecurrentError> {
    if inputs.is_empty() {
        return Err(RecurrentError::EmptySequence);
    }
    let mut hidden = cell.init_hidden();
    let mut outputs = Vec::with_capacity(inputs.len());
    for x in inputs {
        hidden = cell.forward(x, &hidden)?;
        outputs.push(hidden.clone());
    }
    Ok(outputs)
}

// ─────────────────────────────────────────────────────────────────────────────
// RecurrentStats
// ─────────────────────────────────────────────────────────────────────────────

/// Diagnostic statistics for a recurrent cell / sequence run.
#[derive(Debug, Clone)]
pub struct RecurrentStats {
    /// Human-readable cell type label (e.g. `"RNN"`, `"LSTM"`, `"GRU"`).
    pub cell_type: String,
    /// Number of input features.
    pub input_size: usize,
    /// Number of hidden units.
    pub hidden_size: usize,
    /// Total number of learnable parameters.
    pub num_parameters: usize,
    /// Length of the most recently processed sequence, if applicable.
    pub sequence_length: Option<usize>,
}

impl RecurrentStats {
    /// Return a single-line human-readable summary.
    pub fn summary(&self) -> String {
        let seq = match self.sequence_length {
            Some(t) => format!("seq_len={t}"),
            None => "seq_len=n/a".to_string(),
        };
        format!(
            "{} | input={} hidden={} params={} {}",
            self.cell_type, self.input_size, self.hidden_size, self.num_parameters, seq
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    // ── RnnCell ────────────────────────────────────────────────────────────

    #[test]
    fn test_rnn_cell_new() {
        let cell = RnnCell::new(4, 8);
        assert!(cell.is_ok(), "RnnCell::new should succeed");
    }

    #[test]
    fn test_rnn_cell_forward_shape() {
        let cell = RnnCell::new(4, 8).expect("construct rnn");
        let x = Array1::zeros(4);
        let h = cell.init_hidden();
        let h_new = cell.forward(&x, &h).expect("rnn forward");
        assert_eq!(h_new.len(), 8);
    }

    #[test]
    fn test_rnn_cell_init_hidden() {
        let cell = RnnCell::new(3, 5).expect("construct rnn");
        let h = cell.init_hidden();
        assert_eq!(h.len(), 5);
        assert!(h.iter().all(|&v| v == 0.0), "init hidden should be zeros");
    }

    #[test]
    fn test_rnn_cell_num_parameters() {
        let input_size = 4;
        let hidden_size = 8;
        let cell = RnnCell::new(input_size, hidden_size).expect("construct rnn");
        // w_ih: 8*4, w_hh: 8*8, b_ih: 8, b_hh: 8  = 32+64+8+8 = 112
        let expected =
            hidden_size * input_size + hidden_size * hidden_size + hidden_size + hidden_size;
        assert_eq!(cell.num_parameters(), expected);
    }

    // ── LstmCell ───────────────────────────────────────────────────────────

    #[test]
    fn test_lstm_cell_new() {
        let cell = LstmCell::new(4, 8);
        assert!(cell.is_ok(), "LstmCell::new should succeed");
    }

    #[test]
    fn test_lstm_cell_forward_shape() {
        let cell = LstmCell::new(4, 8).expect("construct lstm");
        let x = Array1::zeros(4);
        let state = cell.init_state();
        let new_state = cell.forward(&x, &state).expect("lstm forward");
        assert_eq!(new_state.h.len(), 8);
        assert_eq!(new_state.c.len(), 8);
    }

    #[test]
    fn test_lstm_cell_init_state() {
        let cell = LstmCell::new(3, 6).expect("construct lstm");
        let state = cell.init_state();
        assert_eq!(state.h.len(), 6);
        assert_eq!(state.c.len(), 6);
        assert!(state.h.iter().all(|&v| v == 0.0));
        assert!(state.c.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_lstm_cell_gate_bounds() {
        let cell = LstmCell::new(4, 8).expect("construct lstm");
        let x = Array1::from_elem(4, 0.5);
        let state = cell.init_state();
        let new_state = cell.forward(&x, &state).expect("lstm forward");
        // h' = o ⊙ tanh(c') so each element is in (-1, 1)
        for &v in new_state.h.iter() {
            assert!(v > -1.0 && v < 1.0, "h element out of (-1,1): {v}");
        }
    }

    #[test]
    fn test_lstm_cell_num_parameters() {
        let input_size = 4;
        let hidden_size = 8;
        let cell = LstmCell::new(input_size, hidden_size).expect("construct lstm");
        let gates = 4;
        let expected = gates * hidden_size * input_size
            + gates * hidden_size * hidden_size
            + gates * hidden_size
            + gates * hidden_size;
        assert_eq!(cell.num_parameters(), expected);
    }

    // ── GruCell ────────────────────────────────────────────────────────────

    #[test]
    fn test_gru_cell_new() {
        let cell = GruCell::new(4, 8);
        assert!(cell.is_ok(), "GruCell::new should succeed");
    }

    #[test]
    fn test_gru_cell_forward_shape() {
        let cell = GruCell::new(4, 8).expect("construct gru");
        let x = Array1::zeros(4);
        let h = cell.init_hidden();
        let h_new = cell.forward(&x, &h).expect("gru forward");
        assert_eq!(h_new.len(), 8);
    }

    #[test]
    fn test_gru_cell_hidden_init_zeros() {
        let cell = GruCell::new(3, 5).expect("construct gru");
        let h = cell.init_hidden();
        assert_eq!(h.len(), 5);
        assert!(h.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_gru_cell_num_parameters() {
        let input_size = 4;
        let hidden_size = 8;
        let cell = GruCell::new(input_size, hidden_size).expect("construct gru");
        let gates = 3;
        let expected = gates * hidden_size * input_size
            + gates * hidden_size * hidden_size
            + gates * hidden_size
            + gates * hidden_size;
        assert_eq!(cell.num_parameters(), expected);
    }

    // ── Sequence helpers ───────────────────────────────────────────────────

    #[test]
    fn test_rnn_sequence_length() {
        let cell = RnnCell::new(4, 8).expect("rnn");
        let inputs: Vec<Array1<f64>> = (0..7).map(|_| Array1::zeros(4)).collect();
        let out = rnn_sequence(&cell, &inputs).expect("rnn sequence");
        assert_eq!(out.len(), 7, "T inputs → T outputs");
    }

    #[test]
    fn test_rnn_sequence_empty_error() {
        let cell = RnnCell::new(4, 8).expect("rnn");
        let result = rnn_sequence(&cell, &[]);
        assert!(
            matches!(result, Err(RecurrentError::EmptySequence)),
            "expected EmptySequence error"
        );
    }

    #[test]
    fn test_lstm_sequence_length() {
        let cell = LstmCell::new(4, 8).expect("lstm");
        let inputs: Vec<Array1<f64>> = (0..5).map(|_| Array1::zeros(4)).collect();
        let (hidden_states, _) = lstm_sequence(&cell, &inputs).expect("lstm sequence");
        assert_eq!(hidden_states.len(), 5);
    }

    #[test]
    fn test_lstm_sequence_final_state_nonzero() {
        let cell = LstmCell::new(4, 8).expect("lstm");
        // Non-zero inputs so that the state is driven away from zero
        let inputs: Vec<Array1<f64>> = (0..3).map(|_| Array1::from_elem(4, 1.0)).collect();
        let (_, final_state) = lstm_sequence(&cell, &inputs).expect("lstm sequence");
        let h_norm: f64 = final_state.h.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            h_norm > 1e-12,
            "final h should be non-zero for non-zero inputs"
        );
    }

    #[test]
    fn test_gru_sequence_length() {
        let cell = GruCell::new(4, 8).expect("gru");
        let inputs: Vec<Array1<f64>> = (0..6).map(|_| Array1::zeros(4)).collect();
        let out = gru_sequence(&cell, &inputs).expect("gru sequence");
        assert_eq!(out.len(), 6);
    }

    // ── RecurrentStats ─────────────────────────────────────────────────────

    #[test]
    fn test_recurrent_stats_summary_nonempty() {
        let stats = RecurrentStats {
            cell_type: "LSTM".to_string(),
            input_size: 4,
            hidden_size: 8,
            num_parameters: 416,
            sequence_length: Some(10),
        };
        let s = stats.summary();
        assert!(!s.is_empty(), "summary should not be empty");
        assert!(s.contains("LSTM"));
        assert!(s.contains("416"));
    }

    // ── from_weights shape mismatch ─────────────────────────────────────────

    #[test]
    fn test_lstm_cell_from_weights_shape_mismatch() {
        use scirs2_core::ndarray::Array2;
        // w_ih has 4*hidden=8 rows but w_hh has wrong ncols
        let w_ih = Array2::zeros((8, 4));
        let w_hh = Array2::zeros((8, 3)); // should be (8, 2) for hidden=2
        let b_ih = Array1::zeros(8);
        let b_hh = Array1::zeros(8);
        let result = LstmCell::from_weights(w_ih, w_hh, b_ih, b_hh);
        assert!(result.is_err(), "should fail due to w_hh shape mismatch");
    }
}
