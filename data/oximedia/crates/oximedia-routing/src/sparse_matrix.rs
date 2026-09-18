#![allow(dead_code)]
//! Sparse crosspoint matrix for large routing matrices (256×256 and beyond).
//!
//! A dense `N×M` boolean matrix requires `N*M` bytes of storage.  For a
//! 256×256 matrix that is 65 536 bytes — manageable — but at 1024×1024 it
//! grows to ~1 MiB, and at broadcast-scale sizes (4096×4096) it reaches 16 MiB
//! just for connection flags.  When only a small fraction of crosspoints are
//! active (a typical live production scenario), a sparse representation
//! dramatically reduces memory and improves cache utilisation.
//!
//! [`SparseCrosspointMatrix`] stores only *active* connections in a
//! `HashMap<(usize, usize), bool>`, offering O(1) average insert/lookup and
//! consuming memory proportional to the number of active crosspoints rather
//! than the matrix dimensions.
//!
//! ## Interface parity with `CrosspointMatrix`
//!
//! The public API mirrors the dense [`crate::crosspoint_matrix::RoutingMatrix`]
//! so the two types can be used interchangeably:
//!
//! ```rust
//! use oximedia_routing::sparse_matrix::SparseCrosspointMatrix;
//!
//! let mut m = SparseCrosspointMatrix::new(256, 256);
//! m.connect(0, 0).expect("valid");
//! assert!(m.is_connected(0, 0));
//! assert_eq!(m.active_count(), 1);
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from [`SparseCrosspointMatrix`] operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseCrosspointError {
    /// The supplied input index is out of bounds.
    InputOutOfBounds {
        /// Offending index.
        input: usize,
        /// Maximum valid index (`num_inputs - 1`).
        max: usize,
    },
    /// The supplied output index is out of bounds.
    OutputOutOfBounds {
        /// Offending index.
        output: usize,
        /// Maximum valid index (`num_outputs - 1`).
        max: usize,
    },
}

impl std::fmt::Display for SparseCrosspointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputOutOfBounds { input, max } => {
                write!(f, "input index {input} out of bounds (max {max})")
            }
            Self::OutputOutOfBounds { output, max } => {
                write!(f, "output index {output} out of bounds (max {max})")
            }
        }
    }
}

impl std::error::Error for SparseCrosspointError {}

// ---------------------------------------------------------------------------
// SparseCrosspointMatrix
// ---------------------------------------------------------------------------

/// A sparse `num_inputs × num_outputs` routing matrix.
///
/// Only active (closed) crosspoints are stored in a [`HashMap`], making this
/// structure memory-efficient for large, sparsely-connected matrices.
#[derive(Debug, Clone)]
pub struct SparseCrosspointMatrix {
    /// Number of input channels.
    pub num_inputs: usize,
    /// Number of output channels.
    pub num_outputs: usize,
    /// Sparse storage: key = `(input, output)`, value always `true` when present.
    active: HashMap<(usize, usize), bool>,
}

impl SparseCrosspointMatrix {
    /// Creates a new empty matrix of the given dimensions.
    pub fn new(num_inputs: usize, num_outputs: usize) -> Self {
        Self {
            num_inputs,
            num_outputs,
            active: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Bounds checking helpers
    // -----------------------------------------------------------------------

    fn check_input(&self, input: usize) -> Result<(), SparseCrosspointError> {
        if input >= self.num_inputs {
            Err(SparseCrosspointError::InputOutOfBounds {
                input,
                max: self.num_inputs.saturating_sub(1),
            })
        } else {
            Ok(())
        }
    }

    fn check_output(&self, output: usize) -> Result<(), SparseCrosspointError> {
        if output >= self.num_outputs {
            Err(SparseCrosspointError::OutputOutOfBounds {
                output,
                max: self.num_outputs.saturating_sub(1),
            })
        } else {
            Ok(())
        }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Connects `input` to `output`.
    ///
    /// Idempotent: connecting an already-connected crosspoint succeeds silently.
    pub fn connect(&mut self, input: usize, output: usize) -> Result<(), SparseCrosspointError> {
        self.check_input(input)?;
        self.check_output(output)?;
        self.active.insert((input, output), true);
        Ok(())
    }

    /// Disconnects `input` from `output`.
    ///
    /// Idempotent: disconnecting a crosspoint that is not connected succeeds
    /// silently.
    pub fn disconnect(&mut self, input: usize, output: usize) -> Result<(), SparseCrosspointError> {
        self.check_input(input)?;
        self.check_output(output)?;
        self.active.remove(&(input, output));
        Ok(())
    }

    /// Returns `true` if the crosspoint at `(input, output)` is active.
    pub fn is_connected(&self, input: usize, output: usize) -> bool {
        self.active.get(&(input, output)).copied().unwrap_or(false)
    }

    /// Returns the number of currently active connections.
    pub fn active_count(&self) -> usize {
        self.active.values().filter(|&&v| v).count()
    }

    /// Returns the total capacity of the matrix (`num_inputs * num_outputs`).
    pub fn capacity(&self) -> usize {
        self.num_inputs * self.num_outputs
    }

    /// Converts the sparse matrix to a dense boolean representation.
    ///
    /// Returns a `Vec` of length `num_outputs`; each inner `Vec` has length
    /// `num_inputs`.  `dense[output][input]` is `true` when that crosspoint is
    /// active.
    pub fn to_dense(&self) -> Vec<Vec<bool>> {
        let mut dense = vec![vec![false; self.num_inputs]; self.num_outputs];
        for &(input, output) in self.active.keys() {
            if input < self.num_inputs && output < self.num_outputs {
                dense[output][input] = true;
            }
        }
        dense
    }

    /// Returns all input indices connected to the given `output`.
    pub fn inputs_for(&self, output: usize) -> Vec<usize> {
        self.active
            .keys()
            .filter_map(|&(inp, out)| if out == output { Some(inp) } else { None })
            .collect()
    }

    /// Returns all output indices that `input` is connected to.
    pub fn outputs_for(&self, input: usize) -> Vec<usize> {
        self.active
            .keys()
            .filter_map(|&(inp, out)| if inp == input { Some(out) } else { None })
            .collect()
    }

    /// Removes all active connections from the matrix.
    pub fn clear(&mut self) {
        self.active.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_empty() {
        let m = SparseCrosspointMatrix::new(4, 4);
        assert_eq!(m.active_count(), 0);
    }

    #[test]
    fn test_capacity_4x4() {
        let m = SparseCrosspointMatrix::new(4, 4);
        assert_eq!(m.capacity(), 16);
    }

    // -----------------------------------------------------------------------
    // connect
    // -----------------------------------------------------------------------

    #[test]
    fn test_connect_success() {
        let mut m = SparseCrosspointMatrix::new(4, 4);
        m.connect(0, 0).expect("valid indices");
        assert!(m.is_connected(0, 0));
        assert_eq!(m.active_count(), 1);
    }

    #[test]
    fn test_connect_out_of_bounds_input() {
        let mut m = SparseCrosspointMatrix::new(4, 4);
        let err = m.connect(10, 0);
        assert!(matches!(
            err,
            Err(SparseCrosspointError::InputOutOfBounds { input: 10, .. })
        ));
    }

    #[test]
    fn test_connect_out_of_bounds_output() {
        let mut m = SparseCrosspointMatrix::new(4, 4);
        let err = m.connect(0, 10);
        assert!(matches!(
            err,
            Err(SparseCrosspointError::OutputOutOfBounds { output: 10, .. })
        ));
    }

    #[test]
    fn test_connect_idempotent() {
        let mut m = SparseCrosspointMatrix::new(4, 4);
        m.connect(1, 2).expect("valid");
        m.connect(1, 2).expect("valid"); // second connect should succeed
        assert_eq!(m.active_count(), 1);
    }

    // -----------------------------------------------------------------------
    // disconnect
    // -----------------------------------------------------------------------

    #[test]
    fn test_disconnect_removes_connection() {
        let mut m = SparseCrosspointMatrix::new(4, 4);
        m.connect(1, 2).expect("valid");
        m.disconnect(1, 2).expect("valid");
        assert!(!m.is_connected(1, 2));
        assert_eq!(m.active_count(), 0);
    }

    #[test]
    fn test_disconnect_nonexistent_ok() {
        let mut m = SparseCrosspointMatrix::new(4, 4);
        // Disconnecting a non-connected crosspoint should succeed silently.
        m.disconnect(0, 0).expect("should be fine");
        assert_eq!(m.active_count(), 0);
    }

    #[test]
    fn test_disconnect_out_of_bounds() {
        let mut m = SparseCrosspointMatrix::new(4, 4);
        let err = m.disconnect(20, 0);
        assert!(matches!(
            err,
            Err(SparseCrosspointError::InputOutOfBounds { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // is_connected
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_connected_true() {
        let mut m = SparseCrosspointMatrix::new(4, 4);
        m.connect(2, 3).expect("valid");
        assert!(m.is_connected(2, 3));
    }

    #[test]
    fn test_is_connected_false() {
        let m = SparseCrosspointMatrix::new(4, 4);
        assert!(!m.is_connected(0, 0));
    }

    // -----------------------------------------------------------------------
    // active_count
    // -----------------------------------------------------------------------

    #[test]
    fn test_active_count_increments() {
        let mut m = SparseCrosspointMatrix::new(4, 4);
        m.connect(0, 0).expect("valid");
        m.connect(1, 1).expect("valid");
        m.connect(2, 2).expect("valid");
        assert_eq!(m.active_count(), 3);
    }

    #[test]
    fn test_active_count_decrements_on_disconnect() {
        let mut m = SparseCrosspointMatrix::new(4, 4);
        m.connect(0, 0).expect("valid");
        m.connect(1, 1).expect("valid");
        m.disconnect(0, 0).expect("valid");
        assert_eq!(m.active_count(), 1);
    }

    // -----------------------------------------------------------------------
    // capacity
    // -----------------------------------------------------------------------

    #[test]
    fn test_capacity_256x256() {
        let m = SparseCrosspointMatrix::new(256, 256);
        assert_eq!(m.capacity(), 65_536);
    }

    // -----------------------------------------------------------------------
    // to_dense
    // -----------------------------------------------------------------------

    #[test]
    fn test_to_dense_correct() {
        let mut m = SparseCrosspointMatrix::new(3, 3);
        m.connect(0, 0).expect("valid");
        m.connect(1, 2).expect("valid");
        let dense = m.to_dense();

        assert!(dense[0][0]); // output=0, input=0 connected
        assert!(!dense[0][1]);
        assert!(!dense[0][2]);
        assert!(!dense[1][0]);
        assert!(!dense[1][1]);
        assert!(!dense[1][2]);
        assert!(!dense[2][0]);
        assert!(dense[2][1]); // output=2, input=1 connected
        assert!(!dense[2][2]);
    }

    // -----------------------------------------------------------------------
    // inputs_for / outputs_for
    // -----------------------------------------------------------------------

    #[test]
    fn test_inputs_for() {
        let mut m = SparseCrosspointMatrix::new(4, 4);
        m.connect(0, 2).expect("valid");
        m.connect(1, 2).expect("valid");
        m.connect(3, 2).expect("valid");

        let mut inputs = m.inputs_for(2);
        inputs.sort();
        assert_eq!(inputs, vec![0, 1, 3]);
    }

    #[test]
    fn test_outputs_for() {
        let mut m = SparseCrosspointMatrix::new(4, 4);
        m.connect(0, 1).expect("valid");
        m.connect(0, 3).expect("valid");

        let mut outputs = m.outputs_for(0);
        outputs.sort();
        assert_eq!(outputs, vec![1, 3]);
    }

    // -----------------------------------------------------------------------
    // clear
    // -----------------------------------------------------------------------

    #[test]
    fn test_clear() {
        let mut m = SparseCrosspointMatrix::new(4, 4);
        m.connect(0, 0).expect("valid");
        m.connect(1, 1).expect("valid");
        m.connect(2, 2).expect("valid");
        assert_eq!(m.active_count(), 3);

        m.clear();
        assert_eq!(m.active_count(), 0);
        assert!(!m.is_connected(0, 0));
    }

    // -----------------------------------------------------------------------
    // 256×256 performance test
    // -----------------------------------------------------------------------

    #[test]
    fn test_256x256_performance() {
        let mut m = SparseCrosspointMatrix::new(256, 256);

        // Connect the main diagonal (256 crosspoints).
        for i in 0..256 {
            m.connect(i, i).expect("valid diagonal connect");
        }
        assert_eq!(m.active_count(), 256);

        // Verify each diagonal is connected.
        for i in 0..256 {
            assert!(m.is_connected(i, i), "diagonal {i} should be connected");
        }

        // Disconnect all diagonal crosspoints.
        for i in 0..256 {
            m.disconnect(i, i).expect("valid diagonal disconnect");
        }
        assert_eq!(m.active_count(), 0);
    }

    // -----------------------------------------------------------------------
    // Sparse vs dense equivalence
    // -----------------------------------------------------------------------

    #[test]
    fn test_sparse_vs_dense_equivalence() {
        let n = 8;
        let mut sparse = SparseCrosspointMatrix::new(n, n);
        let mut dense = vec![vec![false; n]; n];

        // Make some arbitrary connections.
        let pairs: &[(usize, usize)] = &[(0, 0), (1, 3), (2, 7), (5, 5), (7, 1), (4, 6)];
        for &(inp, out) in pairs {
            sparse.connect(inp, out).expect("valid");
            dense[out][inp] = true;
        }

        let sparse_dense = sparse.to_dense();
        for out in 0..n {
            for inp in 0..n {
                assert_eq!(
                    sparse_dense[out][inp], dense[out][inp],
                    "mismatch at output={out} input={inp}"
                );
            }
        }
    }
}
