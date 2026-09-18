//! Sparse crosspoint matrix optimised for large dimensions (256x256+).
//!
//! Instead of a dense `HashMap<CrosspointId, CrosspointState>`, this uses
//! compressed sparse row (CSR) storage for O(1) output-lookup and efficient
//! iteration over connected inputs per output.

/// A single non-zero crosspoint entry.
#[derive(Debug, Clone, Copy)]
struct SparseEntry {
    input: usize,
    gain_db: f32,
}

/// Sparse crosspoint matrix using CSR-like storage.
///
/// Internally, connections are grouped by output index and stored in
/// sorted order per output, enabling fast lookup and iteration.
#[derive(Debug, Clone)]
pub struct SparseCrosspointMatrix {
    inputs: usize,
    outputs: usize,
    /// Per-output list of (input, gain_db) entries, sorted by input.
    rows: Vec<Vec<SparseEntry>>,
    /// Total number of active connections.
    nnz: usize,
    /// Input labels.
    input_labels: Vec<String>,
    /// Output labels.
    output_labels: Vec<String>,
}

impl SparseCrosspointMatrix {
    /// Creates a new sparse matrix with the given dimensions.
    pub fn new(inputs: usize, outputs: usize) -> Self {
        Self {
            inputs,
            outputs,
            rows: vec![Vec::new(); outputs],
            nnz: 0,
            input_labels: (0..inputs).map(|i| format!("Input {}", i + 1)).collect(),
            output_labels: (0..outputs).map(|i| format!("Output {}", i + 1)).collect(),
        }
    }

    /// Number of inputs.
    pub fn input_count(&self) -> usize {
        self.inputs
    }

    /// Number of outputs.
    pub fn output_count(&self) -> usize {
        self.outputs
    }

    /// Number of active connections (non-zero entries).
    pub fn active_count(&self) -> usize {
        self.nnz
    }

    /// Density as a fraction of total possible crosspoints.
    pub fn density(&self) -> f64 {
        let total = self.inputs * self.outputs;
        if total == 0 {
            return 0.0;
        }
        self.nnz as f64 / total as f64
    }

    /// Connect an input to an output with the given gain.
    pub fn connect(
        &mut self,
        input: usize,
        output: usize,
        gain_db: f32,
    ) -> Result<(), SparseMatrixError> {
        if input >= self.inputs {
            return Err(SparseMatrixError::InvalidInput(input));
        }
        if output >= self.outputs {
            return Err(SparseMatrixError::InvalidOutput(output));
        }

        let row = &mut self.rows[output];
        match row.binary_search_by_key(&input, |e| e.input) {
            Ok(pos) => {
                // Already connected — update gain
                row[pos].gain_db = gain_db;
            }
            Err(pos) => {
                row.insert(pos, SparseEntry { input, gain_db });
                self.nnz += 1;
            }
        }
        Ok(())
    }

    /// Disconnect an input from an output.
    pub fn disconnect(&mut self, input: usize, output: usize) -> Result<(), SparseMatrixError> {
        if input >= self.inputs {
            return Err(SparseMatrixError::InvalidInput(input));
        }
        if output >= self.outputs {
            return Err(SparseMatrixError::InvalidOutput(output));
        }

        let row = &mut self.rows[output];
        if let Ok(pos) = row.binary_search_by_key(&input, |e| e.input) {
            row.remove(pos);
            self.nnz -= 1;
        }
        Ok(())
    }

    /// Check if an input is connected to an output.
    pub fn is_connected(&self, input: usize, output: usize) -> bool {
        if output >= self.outputs {
            return false;
        }
        self.rows[output]
            .binary_search_by_key(&input, |e| e.input)
            .is_ok()
    }

    /// Get the gain for a crosspoint.
    pub fn get_gain(&self, input: usize, output: usize) -> Option<f32> {
        if output >= self.outputs {
            return None;
        }
        self.rows[output]
            .binary_search_by_key(&input, |e| e.input)
            .ok()
            .map(|pos| self.rows[output][pos].gain_db)
    }

    /// Set gain on an existing connection.
    pub fn set_gain(
        &mut self,
        input: usize,
        output: usize,
        gain_db: f32,
    ) -> Result<(), SparseMatrixError> {
        if output >= self.outputs {
            return Err(SparseMatrixError::InvalidOutput(output));
        }
        let row = &mut self.rows[output];
        match row.binary_search_by_key(&input, |e| e.input) {
            Ok(pos) => {
                row[pos].gain_db = gain_db;
                Ok(())
            }
            Err(_) => Err(SparseMatrixError::NotConnected(input, output)),
        }
    }

    /// Get all inputs connected to a given output (sorted).
    pub fn inputs_for_output(&self, output: usize) -> Vec<(usize, f32)> {
        if output >= self.outputs {
            return Vec::new();
        }
        self.rows[output]
            .iter()
            .map(|e| (e.input, e.gain_db))
            .collect()
    }

    /// Get all outputs connected from a given input (linear scan).
    pub fn outputs_for_input(&self, input: usize) -> Vec<(usize, f32)> {
        let mut results = Vec::new();
        for (output, row) in self.rows.iter().enumerate() {
            if let Ok(pos) = row.binary_search_by_key(&input, |e| e.input) {
                results.push((output, row[pos].gain_db));
            }
        }
        results
    }

    /// Clear all connections.
    pub fn clear_all(&mut self) {
        for row in &mut self.rows {
            row.clear();
        }
        self.nnz = 0;
    }

    /// Compute the summing bus output for a given output channel.
    ///
    /// Given a slice of input samples, sums `sample[input] * linear_gain`
    /// for each connected input.
    pub fn compute_output(&self, output: usize, input_samples: &[f32]) -> f32 {
        if output >= self.outputs {
            return 0.0;
        }
        let mut sum = 0.0_f32;
        for entry in &self.rows[output] {
            if entry.input < input_samples.len() {
                let gain_linear = 10.0_f32.powf(entry.gain_db / 20.0);
                sum += input_samples[entry.input] * gain_linear;
            }
        }
        sum
    }

    /// Compute all output channels at once.
    pub fn compute_all_outputs(&self, input_samples: &[f32]) -> Vec<f32> {
        (0..self.outputs)
            .map(|o| self.compute_output(o, input_samples))
            .collect()
    }

    /// Set input label.
    pub fn set_input_label(
        &mut self,
        input: usize,
        label: String,
    ) -> Result<(), SparseMatrixError> {
        if input >= self.inputs {
            return Err(SparseMatrixError::InvalidInput(input));
        }
        self.input_labels[input] = label;
        Ok(())
    }

    /// Set output label.
    pub fn set_output_label(
        &mut self,
        output: usize,
        label: String,
    ) -> Result<(), SparseMatrixError> {
        if output >= self.outputs {
            return Err(SparseMatrixError::InvalidOutput(output));
        }
        self.output_labels[output] = label;
        Ok(())
    }

    /// Get input label.
    pub fn get_input_label(&self, input: usize) -> Option<&str> {
        self.input_labels.get(input).map(String::as_str)
    }

    /// Get output label.
    pub fn get_output_label(&self, output: usize) -> Option<&str> {
        self.output_labels.get(output).map(String::as_str)
    }
}

/// Errors from sparse matrix operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum SparseMatrixError {
    /// Invalid input index.
    #[error("Invalid input index: {0}")]
    InvalidInput(usize),
    /// Invalid output index.
    #[error("Invalid output index: {0}")]
    InvalidOutput(usize),
    /// Crosspoint not connected.
    #[error("Crosspoint not connected: input {0} to output {1}")]
    NotConnected(usize, usize),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_sparse_matrix() {
        let m = SparseCrosspointMatrix::new(256, 256);
        assert_eq!(m.input_count(), 256);
        assert_eq!(m.output_count(), 256);
        assert_eq!(m.active_count(), 0);
    }

    #[test]
    fn test_connect_disconnect() {
        let mut m = SparseCrosspointMatrix::new(8, 8);
        m.connect(0, 0, -6.0).expect("ok");
        assert!(m.is_connected(0, 0));
        assert_eq!(m.active_count(), 1);

        m.disconnect(0, 0).expect("ok");
        assert!(!m.is_connected(0, 0));
        assert_eq!(m.active_count(), 0);
    }

    #[test]
    fn test_connect_updates_gain() {
        let mut m = SparseCrosspointMatrix::new(8, 8);
        m.connect(0, 0, -6.0).expect("ok");
        m.connect(0, 0, -3.0).expect("ok"); // update
        assert_eq!(m.active_count(), 1); // still 1
        assert!((m.get_gain(0, 0).expect("exists") - (-3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_invalid_input() {
        let mut m = SparseCrosspointMatrix::new(4, 4);
        assert!(m.connect(10, 0, 0.0).is_err());
    }

    #[test]
    fn test_invalid_output() {
        let mut m = SparseCrosspointMatrix::new(4, 4);
        assert!(m.connect(0, 10, 0.0).is_err());
    }

    #[test]
    fn test_get_gain_not_connected() {
        let m = SparseCrosspointMatrix::new(4, 4);
        assert!(m.get_gain(0, 0).is_none());
    }

    #[test]
    fn test_set_gain() {
        let mut m = SparseCrosspointMatrix::new(4, 4);
        m.connect(0, 0, 0.0).expect("ok");
        m.set_gain(0, 0, -12.0).expect("ok");
        assert!((m.get_gain(0, 0).expect("exists") - (-12.0)).abs() < 1e-6);
    }

    #[test]
    fn test_set_gain_not_connected() {
        let mut m = SparseCrosspointMatrix::new(4, 4);
        assert!(m.set_gain(0, 0, -6.0).is_err());
    }

    #[test]
    fn test_inputs_for_output() {
        let mut m = SparseCrosspointMatrix::new(8, 8);
        m.connect(2, 0, -6.0).expect("ok");
        m.connect(5, 0, -3.0).expect("ok");
        m.connect(1, 0, 0.0).expect("ok");

        let inputs = m.inputs_for_output(0);
        assert_eq!(inputs.len(), 3);
        // Should be sorted by input index
        assert_eq!(inputs[0].0, 1);
        assert_eq!(inputs[1].0, 2);
        assert_eq!(inputs[2].0, 5);
    }

    #[test]
    fn test_outputs_for_input() {
        let mut m = SparseCrosspointMatrix::new(8, 8);
        m.connect(0, 1, -6.0).expect("ok");
        m.connect(0, 3, -3.0).expect("ok");

        let outputs = m.outputs_for_input(0);
        assert_eq!(outputs.len(), 2);
    }

    #[test]
    fn test_clear_all() {
        let mut m = SparseCrosspointMatrix::new(8, 8);
        m.connect(0, 0, 0.0).expect("ok");
        m.connect(1, 1, 0.0).expect("ok");
        m.clear_all();
        assert_eq!(m.active_count(), 0);
    }

    #[test]
    fn test_density() {
        let mut m = SparseCrosspointMatrix::new(4, 4);
        m.connect(0, 0, 0.0).expect("ok");
        m.connect(1, 1, 0.0).expect("ok");
        // 2 / 16 = 0.125
        assert!((m.density() - 0.125).abs() < 1e-10);
    }

    #[test]
    fn test_compute_output() {
        let mut m = SparseCrosspointMatrix::new(4, 4);
        // Connect input 0 and 1 to output 0 at 0 dB
        m.connect(0, 0, 0.0).expect("ok");
        m.connect(1, 0, 0.0).expect("ok");

        let samples = [0.5_f32, 0.3, 0.0, 0.0];
        let result = m.compute_output(0, &samples);
        assert!((result - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_compute_output_with_gain() {
        let mut m = SparseCrosspointMatrix::new(4, 4);
        // Connect input 0 to output 0 at -20 dB (gain = 0.1)
        m.connect(0, 0, -20.0).expect("ok");

        let samples = [1.0_f32, 0.0, 0.0, 0.0];
        let result = m.compute_output(0, &samples);
        assert!((result - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_compute_all_outputs() {
        let mut m = SparseCrosspointMatrix::new(4, 4);
        m.connect(0, 0, 0.0).expect("ok");
        m.connect(1, 1, 0.0).expect("ok");

        let samples = [0.5_f32, 0.8, 0.0, 0.0];
        let outputs = m.compute_all_outputs(&samples);
        assert_eq!(outputs.len(), 4);
        assert!((outputs[0] - 0.5).abs() < 1e-5);
        assert!((outputs[1] - 0.8).abs() < 1e-5);
        assert!(outputs[2].abs() < 1e-10);
    }

    #[test]
    fn test_large_matrix_256x256() {
        let mut m = SparseCrosspointMatrix::new(256, 256);
        // Connect diagonal
        for i in 0..256 {
            m.connect(i, i, 0.0).expect("ok");
        }
        assert_eq!(m.active_count(), 256);

        // Verify a few
        assert!(m.is_connected(0, 0));
        assert!(m.is_connected(128, 128));
        assert!(m.is_connected(255, 255));
        assert!(!m.is_connected(0, 1));
    }

    #[test]
    fn test_large_matrix_1024x1024_diagonal() {
        let mut m = SparseCrosspointMatrix::new(1024, 1024);
        for i in 0..1024 {
            m.connect(i, i, 0.0).expect("ok");
        }
        assert_eq!(m.active_count(), 1024);
        assert!(m.is_connected(512, 512));
    }

    #[test]
    fn test_labels() {
        let mut m = SparseCrosspointMatrix::new(2, 2);
        m.set_input_label(0, "Mic 1".to_string()).expect("ok");
        m.set_output_label(1, "Mon R".to_string()).expect("ok");
        assert_eq!(m.get_input_label(0), Some("Mic 1"));
        assert_eq!(m.get_output_label(1), Some("Mon R"));
    }

    #[test]
    fn test_disconnect_not_connected() {
        let mut m = SparseCrosspointMatrix::new(4, 4);
        // Disconnecting a non-existent connection should succeed silently
        m.disconnect(0, 0).expect("ok");
        assert_eq!(m.active_count(), 0);
    }

    #[test]
    fn test_density_zero_matrix() {
        let m = SparseCrosspointMatrix::new(0, 0);
        assert!((m.density()).abs() < 1e-10);
    }
}
