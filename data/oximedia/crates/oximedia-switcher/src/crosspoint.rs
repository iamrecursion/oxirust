//! Crosspoint matrix routing for video switcher signal paths.
//!
//! This module implements a crosspoint matrix that maps input signals to
//! output destinations.  It supports multiple signal layers (video, key,
//! audio), locking individual crosspoints, salvos (preset routing states),
//! reverse lookups (which outputs carry a given input), and atomic batch
//! routing (apply multiple routes at once or roll back on error).
//!
//! ## Performance Characteristics
//!
//! | Operation                           | Complexity |
//! |-------------------------------------|-----------|
//! | Route (set input→output)            | O(1)      |
//! | Query input for a given output      | O(1)      |
//! | Reverse query: outputs for input X  | O(k) where k = number of outputs for X |
//! | Batch route (N routes)              | O(N)      |
//! | Recall salvo (M routes)             | O(M)      |
//!
//! The reverse mapping is maintained as a secondary `HashMap<(layer, input), Vec<output>>`
//! and kept consistent with the primary `(layer, output) → Crosspoint` map on every
//! mutating operation.

use std::collections::HashMap;
use std::fmt;

/// Signal layer within the crosspoint matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SignalLayer {
    /// Video signal.
    Video,
    /// Key / alpha signal.
    Key,
    /// Audio signal.
    Audio,
    /// Metadata / ancillary data.
    Metadata,
}

impl fmt::Display for SignalLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Video => write!(f, "Video"),
            Self::Key => write!(f, "Key"),
            Self::Audio => write!(f, "Audio"),
            Self::Metadata => write!(f, "Metadata"),
        }
    }
}

/// A single crosspoint (input-to-output mapping).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Crosspoint {
    /// Input index (source).
    pub input: usize,
    /// Output index (destination).
    pub output: usize,
    /// Signal layer.
    pub layer: SignalLayer,
    /// Whether this crosspoint is locked.
    pub locked: bool,
}

impl Crosspoint {
    /// Create a new crosspoint.
    pub fn new(input: usize, output: usize, layer: SignalLayer) -> Self {
        Self {
            input,
            output,
            layer,
            locked: false,
        }
    }
}

impl fmt::Display for Crosspoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{} -> {} {}",
            self.layer,
            self.input,
            self.output,
            if self.locked { "[LOCKED]" } else { "" }
        )
    }
}

/// Error type for crosspoint operations.
#[derive(Debug, Clone)]
pub enum CrosspointError {
    /// Input index out of range.
    InputOutOfRange(usize),
    /// Output index out of range.
    OutputOutOfRange(usize),
    /// Crosspoint is locked and cannot be changed.
    Locked {
        /// Output index.
        output: usize,
        /// Signal layer.
        layer: SignalLayer,
    },
    /// Salvo not found.
    SalvoNotFound(String),
}

impl fmt::Display for CrosspointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputOutOfRange(i) => write!(f, "Input {i} out of range"),
            Self::OutputOutOfRange(o) => write!(f, "Output {o} out of range"),
            Self::Locked { output, layer } => {
                write!(f, "Output {output} is locked on layer {layer}")
            }
            Self::SalvoNotFound(name) => write!(f, "Salvo '{name}' not found"),
        }
    }
}

/// A salvo is a named preset of crosspoint routes.
#[derive(Debug, Clone)]
pub struct Salvo {
    /// Name of the salvo.
    pub name: String,
    /// Routes: (layer, output) -> input.
    pub routes: HashMap<(SignalLayer, usize), usize>,
}

impl Salvo {
    /// Create a new empty salvo.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            routes: HashMap::new(),
        }
    }

    /// Add a route to the salvo.
    pub fn add_route(&mut self, layer: SignalLayer, output: usize, input: usize) {
        self.routes.insert((layer, output), input);
    }

    /// Get the number of routes in this salvo.
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }
}

/// Crosspoint matrix configuration.
#[derive(Debug, Clone)]
pub struct CrosspointMatrixConfig {
    /// Number of inputs.
    pub num_inputs: usize,
    /// Number of outputs.
    pub num_outputs: usize,
    /// Enabled signal layers.
    pub layers: Vec<SignalLayer>,
}

impl CrosspointMatrixConfig {
    /// Create a new configuration.
    pub fn new(num_inputs: usize, num_outputs: usize) -> Self {
        Self {
            num_inputs,
            num_outputs,
            layers: vec![SignalLayer::Video],
        }
    }

    /// Create with all standard layers.
    pub fn with_all_layers(num_inputs: usize, num_outputs: usize) -> Self {
        Self {
            num_inputs,
            num_outputs,
            layers: vec![
                SignalLayer::Video,
                SignalLayer::Key,
                SignalLayer::Audio,
                SignalLayer::Metadata,
            ],
        }
    }
}

/// A single routing change request used in batch operations.
///
/// See [`CrosspointMatrix::route_batch`].
#[derive(Debug, Clone, Copy)]
pub struct RouteRequest {
    /// Signal layer.
    pub layer: SignalLayer,
    /// Input index (source).
    pub input: usize,
    /// Output index (destination).
    pub output: usize,
}

impl RouteRequest {
    /// Create a new route request.
    pub fn new(layer: SignalLayer, input: usize, output: usize) -> Self {
        Self {
            layer,
            input,
            output,
        }
    }
}

/// Crosspoint matrix router.
///
/// Maintains two complementary indexes for O(1) forward and fast reverse lookup:
///
/// * `routes`: `(layer, output)` → `Crosspoint`  — primary, used for normal routing.
/// * `reverse`: `(layer, input)` → `Vec<output>` — secondary, updated on every route
///   change; enables "which outputs carry this input?" queries without scanning all routes.
#[derive(Debug)]
pub struct CrosspointMatrix {
    /// Configuration.
    config: CrosspointMatrixConfig,
    /// Primary routes: (layer, output) -> crosspoint.
    routes: HashMap<(SignalLayer, usize), Crosspoint>,
    /// Reverse index: (layer, input) -> list of output indices using that input.
    reverse: HashMap<(SignalLayer, usize), Vec<usize>>,
    /// Saved salvos.
    salvos: HashMap<String, Salvo>,
}

impl CrosspointMatrix {
    /// Create a new crosspoint matrix.
    ///
    /// All outputs are initialised to input 0 on every enabled layer.
    pub fn new(config: CrosspointMatrixConfig) -> Self {
        let mut routes = HashMap::new();
        let mut reverse: HashMap<(SignalLayer, usize), Vec<usize>> = HashMap::new();

        // Initialise all outputs to input 0.
        for &layer in &config.layers {
            for out in 0..config.num_outputs {
                routes.insert((layer, out), Crosspoint::new(0, out, layer));
                reverse.entry((layer, 0)).or_insert_with(Vec::new).push(out);
            }
        }

        Self {
            config,
            routes,
            reverse,
            salvos: HashMap::new(),
        }
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Remove `output` from the reverse index entry for `(layer, old_input)`.
    fn reverse_remove(&mut self, layer: SignalLayer, old_input: usize, output: usize) {
        if let Some(v) = self.reverse.get_mut(&(layer, old_input)) {
            v.retain(|&o| o != output);
        }
    }

    /// Add `output` to the reverse index entry for `(layer, new_input)`.
    fn reverse_insert(&mut self, layer: SignalLayer, new_input: usize, output: usize) {
        self.reverse
            .entry((layer, new_input))
            .or_insert_with(Vec::new)
            .push(output);
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Set a crosspoint route (input → output on a given layer).
    ///
    /// Returns `Err` if:
    /// * `input` ≥ `num_inputs`,
    /// * `output` ≥ `num_outputs`, or
    /// * the crosspoint is locked.
    ///
    /// This is an O(1) operation.
    pub fn route(
        &mut self,
        layer: SignalLayer,
        input: usize,
        output: usize,
    ) -> Result<(), CrosspointError> {
        if input >= self.config.num_inputs {
            return Err(CrosspointError::InputOutOfRange(input));
        }
        if output >= self.config.num_outputs {
            return Err(CrosspointError::OutputOutOfRange(output));
        }

        let key = (layer, output);
        if let Some(cp) = self.routes.get(&key) {
            if cp.locked {
                return Err(CrosspointError::Locked { output, layer });
            }
            // Remove old reverse mapping.
            let old_input = cp.input;
            self.reverse_remove(layer, old_input, output);
        }

        self.routes
            .insert(key, Crosspoint::new(input, output, layer));
        self.reverse_insert(layer, input, output);
        Ok(())
    }

    /// Apply multiple routing changes atomically.
    ///
    /// All requests are validated first (bounds check, lock check).  If any
    /// request would fail the entire batch is rejected and no routes are
    /// changed.  On success all routes are applied in order.
    ///
    /// Returns the number of routes applied (equal to `requests.len()` on
    /// success).
    ///
    /// # Errors
    ///
    /// Returns the first [`CrosspointError`] found during validation.
    pub fn route_batch(&mut self, requests: &[RouteRequest]) -> Result<usize, CrosspointError> {
        // --- Validation pass (no mutations) ---
        for req in requests {
            if req.input >= self.config.num_inputs {
                return Err(CrosspointError::InputOutOfRange(req.input));
            }
            if req.output >= self.config.num_outputs {
                return Err(CrosspointError::OutputOutOfRange(req.output));
            }
            if let Some(cp) = self.routes.get(&(req.layer, req.output)) {
                if cp.locked {
                    return Err(CrosspointError::Locked {
                        output: req.output,
                        layer: req.layer,
                    });
                }
            }
        }

        // --- Apply pass (all validated) ---
        for req in requests {
            // Remove old reverse entry.
            if let Some(cp) = self.routes.get(&(req.layer, req.output)) {
                let old_input = cp.input;
                self.reverse_remove(req.layer, old_input, req.output);
            }
            self.routes.insert(
                (req.layer, req.output),
                Crosspoint::new(req.input, req.output, req.layer),
            );
            self.reverse_insert(req.layer, req.input, req.output);
        }

        Ok(requests.len())
    }

    /// Get the current input routed to an output on a given layer.
    ///
    /// Returns `None` if the output is out of range or the layer is not enabled.
    ///
    /// O(1).
    pub fn get_route(&self, layer: SignalLayer, output: usize) -> Option<usize> {
        self.routes.get(&(layer, output)).map(|cp| cp.input)
    }

    /// Get a crosspoint reference.
    ///
    /// O(1).
    pub fn get_crosspoint(&self, layer: SignalLayer, output: usize) -> Option<&Crosspoint> {
        self.routes.get(&(layer, output))
    }

    /// Reverse lookup: return all output indices that currently carry `input`
    /// on `layer`.
    ///
    /// Returns an empty slice if the input is unused or the layer is not active.
    ///
    /// The returned slice may contain outputs in any order.
    pub fn outputs_for_input(&self, layer: SignalLayer, input: usize) -> &[usize] {
        self.reverse
            .get(&(layer, input))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Lock a crosspoint so it cannot be changed.
    pub fn lock(&mut self, layer: SignalLayer, output: usize) -> Result<(), CrosspointError> {
        if output >= self.config.num_outputs {
            return Err(CrosspointError::OutputOutOfRange(output));
        }
        if let Some(cp) = self.routes.get_mut(&(layer, output)) {
            cp.locked = true;
            Ok(())
        } else {
            Err(CrosspointError::OutputOutOfRange(output))
        }
    }

    /// Unlock a crosspoint.
    pub fn unlock(&mut self, layer: SignalLayer, output: usize) -> Result<(), CrosspointError> {
        if output >= self.config.num_outputs {
            return Err(CrosspointError::OutputOutOfRange(output));
        }
        if let Some(cp) = self.routes.get_mut(&(layer, output)) {
            cp.locked = false;
            Ok(())
        } else {
            Err(CrosspointError::OutputOutOfRange(output))
        }
    }

    /// Save the current state as a salvo.
    pub fn save_salvo(&mut self, name: &str) {
        let mut salvo = Salvo::new(name);
        for (&(layer, output), cp) in &self.routes {
            salvo.add_route(layer, output, cp.input);
        }
        self.salvos.insert(name.to_string(), salvo);
    }

    /// Recall a salvo (apply saved routes, skipping locked crosspoints).
    ///
    /// Returns the number of routes actually applied (locked crosspoints are
    /// silently skipped, invalid indices are also skipped).
    pub fn recall_salvo(&mut self, name: &str) -> Result<usize, CrosspointError> {
        let salvo = self
            .salvos
            .get(name)
            .cloned()
            .ok_or_else(|| CrosspointError::SalvoNotFound(name.to_string()))?;

        let mut applied = 0;
        for (&(layer, output), &input) in &salvo.routes {
            if let Some(cp) = self.routes.get(&(layer, output)) {
                if cp.locked {
                    continue;
                }
            }
            if input < self.config.num_inputs && output < self.config.num_outputs {
                // Update reverse index.
                if let Some(cp) = self.routes.get(&(layer, output)) {
                    let old_input = cp.input;
                    self.reverse_remove(layer, old_input, output);
                }
                self.routes
                    .insert((layer, output), Crosspoint::new(input, output, layer));
                self.reverse_insert(layer, input, output);
                applied += 1;
            }
        }

        Ok(applied)
    }

    /// Get the number of inputs.
    pub fn num_inputs(&self) -> usize {
        self.config.num_inputs
    }

    /// Get the number of outputs.
    pub fn num_outputs(&self) -> usize {
        self.config.num_outputs
    }

    /// Get the list of salvos.
    pub fn salvo_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.salvos.keys().cloned().collect();
        names.sort();
        names
    }

    /// Get total number of active routes.
    pub fn active_route_count(&self) -> usize {
        self.routes.len()
    }

    /// Verify that the forward and reverse indexes are mutually consistent.
    ///
    /// Returns `true` if every `(layer, output)→input` entry in `routes` has a
    /// corresponding entry in the `reverse` index, and vice-versa.
    ///
    /// This is primarily useful as a post-condition check in tests.
    #[cfg(test)]
    pub(crate) fn indexes_consistent(&self) -> bool {
        // Every forward route must appear in the reverse index.
        for (&(layer, output), cp) in &self.routes {
            let rev = self.reverse.get(&(layer, cp.input));
            match rev {
                None => return false,
                Some(v) => {
                    if !v.contains(&output) {
                        return false;
                    }
                }
            }
        }
        // Every reverse entry must correspond to a valid forward route.
        for (&(layer, input), outputs) in &self.reverse {
            for &output in outputs {
                match self.routes.get(&(layer, output)) {
                    None => return false,
                    Some(cp) => {
                        if cp.input != input {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_layer_display() {
        assert_eq!(format!("{}", SignalLayer::Video), "Video");
        assert_eq!(format!("{}", SignalLayer::Audio), "Audio");
    }

    #[test]
    fn test_crosspoint_creation() {
        let cp = Crosspoint::new(3, 1, SignalLayer::Video);
        assert_eq!(cp.input, 3);
        assert_eq!(cp.output, 1);
        assert!(!cp.locked);
    }

    #[test]
    fn test_crosspoint_display() {
        let cp = Crosspoint::new(2, 0, SignalLayer::Video);
        let s = format!("{cp}");
        assert!(s.contains("Video"));
        assert!(s.contains("2"));
    }

    #[test]
    fn test_matrix_creation() {
        let config = CrosspointMatrixConfig::new(8, 4);
        let matrix = CrosspointMatrix::new(config);
        assert_eq!(matrix.num_inputs(), 8);
        assert_eq!(matrix.num_outputs(), 4);
    }

    #[test]
    fn test_matrix_default_routing() {
        let config = CrosspointMatrixConfig::new(8, 4);
        let matrix = CrosspointMatrix::new(config);
        // All outputs default to input 0
        for out in 0..4 {
            assert_eq!(matrix.get_route(SignalLayer::Video, out), Some(0));
        }
    }

    #[test]
    fn test_matrix_route_change() {
        let config = CrosspointMatrixConfig::new(8, 4);
        let mut matrix = CrosspointMatrix::new(config);
        matrix
            .route(SignalLayer::Video, 5, 2)
            .expect("should succeed in test");
        assert_eq!(matrix.get_route(SignalLayer::Video, 2), Some(5));
    }

    #[test]
    fn test_matrix_route_input_out_of_range() {
        let config = CrosspointMatrixConfig::new(4, 2);
        let mut matrix = CrosspointMatrix::new(config);
        let result = matrix.route(SignalLayer::Video, 10, 0);
        assert!(matches!(result, Err(CrosspointError::InputOutOfRange(10))));
    }

    #[test]
    fn test_matrix_route_output_out_of_range() {
        let config = CrosspointMatrixConfig::new(4, 2);
        let mut matrix = CrosspointMatrix::new(config);
        let result = matrix.route(SignalLayer::Video, 0, 10);
        assert!(matches!(result, Err(CrosspointError::OutputOutOfRange(10))));
    }

    #[test]
    fn test_matrix_lock_unlock() {
        let config = CrosspointMatrixConfig::new(8, 4);
        let mut matrix = CrosspointMatrix::new(config);
        matrix
            .route(SignalLayer::Video, 3, 0)
            .expect("should succeed in test");
        matrix
            .lock(SignalLayer::Video, 0)
            .expect("should succeed in test");

        // Routing to a locked output should fail
        let result = matrix.route(SignalLayer::Video, 5, 0);
        assert!(matches!(result, Err(CrosspointError::Locked { .. })));

        // Unlock and try again
        matrix
            .unlock(SignalLayer::Video, 0)
            .expect("should succeed in test");
        assert!(matrix.route(SignalLayer::Video, 5, 0).is_ok());
    }

    #[test]
    fn test_salvo_creation() {
        let mut salvo = Salvo::new("show_a");
        salvo.add_route(SignalLayer::Video, 0, 3);
        salvo.add_route(SignalLayer::Video, 1, 5);
        assert_eq!(salvo.route_count(), 2);
        assert_eq!(salvo.name, "show_a");
    }

    #[test]
    fn test_matrix_save_recall_salvo() {
        let config = CrosspointMatrixConfig::new(8, 4);
        let mut matrix = CrosspointMatrix::new(config);

        // Set up a state
        matrix
            .route(SignalLayer::Video, 3, 0)
            .expect("should succeed in test");
        matrix
            .route(SignalLayer::Video, 5, 1)
            .expect("should succeed in test");
        matrix.save_salvo("preset_1");

        // Change routes
        matrix
            .route(SignalLayer::Video, 0, 0)
            .expect("should succeed in test");
        matrix
            .route(SignalLayer::Video, 0, 1)
            .expect("should succeed in test");

        // Recall salvo
        let applied = matrix
            .recall_salvo("preset_1")
            .expect("should succeed in test");
        assert!(applied > 0);
        assert_eq!(matrix.get_route(SignalLayer::Video, 0), Some(3));
        assert_eq!(matrix.get_route(SignalLayer::Video, 1), Some(5));
    }

    #[test]
    fn test_matrix_recall_salvo_not_found() {
        let config = CrosspointMatrixConfig::new(4, 2);
        let mut matrix = CrosspointMatrix::new(config);
        let result = matrix.recall_salvo("nonexistent");
        assert!(matches!(result, Err(CrosspointError::SalvoNotFound(_))));
    }

    #[test]
    fn test_matrix_recall_salvo_respects_locks() {
        let config = CrosspointMatrixConfig::new(8, 4);
        let mut matrix = CrosspointMatrix::new(config);

        matrix
            .route(SignalLayer::Video, 7, 0)
            .expect("should succeed in test");
        matrix.save_salvo("locked_test");

        matrix
            .route(SignalLayer::Video, 0, 0)
            .expect("should succeed in test");
        matrix
            .lock(SignalLayer::Video, 0)
            .expect("should succeed in test");

        matrix
            .recall_salvo("locked_test")
            .expect("should succeed in test");
        // Output 0 should still be 0 because it's locked
        assert_eq!(matrix.get_route(SignalLayer::Video, 0), Some(0));
    }

    #[test]
    fn test_matrix_salvo_names() {
        let config = CrosspointMatrixConfig::new(4, 2);
        let mut matrix = CrosspointMatrix::new(config);
        matrix.save_salvo("b_salvo");
        matrix.save_salvo("a_salvo");

        let names = matrix.salvo_names();
        assert_eq!(names, vec!["a_salvo", "b_salvo"]);
    }

    #[test]
    fn test_matrix_all_layers() {
        let config = CrosspointMatrixConfig::with_all_layers(4, 2);
        let matrix = CrosspointMatrix::new(config);
        // Should have routes for all 4 layers x 2 outputs = 8 routes
        assert_eq!(matrix.active_route_count(), 8);
    }

    // -----------------------------------------------------------------------
    // Reverse mapping tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reverse_all_outputs_start_on_input_0() {
        let config = CrosspointMatrixConfig::new(8, 4);
        let matrix = CrosspointMatrix::new(config);

        // At creation all 4 outputs are mapped to input 0.
        let outs = matrix.outputs_for_input(SignalLayer::Video, 0);
        assert_eq!(outs.len(), 4, "all 4 outputs must initially carry input 0");
        for &o in outs {
            assert!(o < 4, "output index must be in range [0, 4)");
        }
    }

    #[test]
    fn test_reverse_updates_on_route_change() {
        let config = CrosspointMatrixConfig::new(8, 4);
        let mut matrix = CrosspointMatrix::new(config);

        // Route output 2 to input 5.
        matrix
            .route(SignalLayer::Video, 5, 2)
            .expect("route must succeed");

        // Input 5 should now carry output 2.
        let outs_5 = matrix.outputs_for_input(SignalLayer::Video, 5);
        assert_eq!(outs_5, [2], "input 5 must carry output 2");

        // Input 0 should have dropped output 2.
        let outs_0 = matrix.outputs_for_input(SignalLayer::Video, 0);
        assert!(
            !outs_0.contains(&2),
            "input 0 must no longer carry output 2; got {outs_0:?}"
        );
        assert_eq!(outs_0.len(), 3, "input 0 must have 3 remaining outputs");
    }

    #[test]
    fn test_reverse_multiple_outputs_on_same_input() {
        let config = CrosspointMatrixConfig::new(8, 4);
        let mut matrix = CrosspointMatrix::new(config);

        // Route several outputs to the same input (fan-out).
        matrix.route(SignalLayer::Video, 3, 0).expect("route 0→3");
        matrix.route(SignalLayer::Video, 3, 1).expect("route 1→3");
        matrix.route(SignalLayer::Video, 3, 2).expect("route 2→3");

        let outs_3 = matrix.outputs_for_input(SignalLayer::Video, 3);
        let mut sorted = outs_3.to_vec();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2], "inputs 0,1,2 must be on input 3");
    }

    #[test]
    fn test_reverse_empty_for_unused_input() {
        let config = CrosspointMatrixConfig::new(8, 4);
        let matrix = CrosspointMatrix::new(config);

        // Input 7 has never been routed to.
        let outs = matrix.outputs_for_input(SignalLayer::Video, 7);
        assert!(outs.is_empty(), "unused input should have no outputs");
    }

    #[test]
    fn test_indexes_consistent_after_operations() {
        let config = CrosspointMatrixConfig::new(8, 4);
        let mut matrix = CrosspointMatrix::new(config);
        assert!(
            matrix.indexes_consistent(),
            "fresh matrix must be consistent"
        );

        matrix.route(SignalLayer::Video, 3, 0).expect("route");
        assert!(
            matrix.indexes_consistent(),
            "after route must be consistent"
        );

        matrix.route(SignalLayer::Video, 5, 0).expect("re-route");
        assert!(
            matrix.indexes_consistent(),
            "after re-route must be consistent"
        );

        matrix.route(SignalLayer::Video, 5, 1).expect("route");
        assert!(matrix.indexes_consistent(), "multiple routes consistent");

        matrix.save_salvo("snap");
        matrix.route(SignalLayer::Video, 0, 0).expect("reset");
        matrix.route(SignalLayer::Video, 0, 1).expect("reset");
        matrix.recall_salvo("snap").expect("recall must not fail");
        assert!(
            matrix.indexes_consistent(),
            "after salvo recall must be consistent"
        );
    }

    // -----------------------------------------------------------------------
    // Batch routing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_route_all_at_once() {
        let config = CrosspointMatrixConfig::new(8, 4);
        let mut matrix = CrosspointMatrix::new(config);

        let reqs = vec![
            RouteRequest::new(SignalLayer::Video, 1, 0),
            RouteRequest::new(SignalLayer::Video, 2, 1),
            RouteRequest::new(SignalLayer::Video, 3, 2),
            RouteRequest::new(SignalLayer::Video, 4, 3),
        ];

        let applied = matrix.route_batch(&reqs).expect("batch must succeed");
        assert_eq!(applied, 4, "all 4 requests must be applied");

        assert_eq!(matrix.get_route(SignalLayer::Video, 0), Some(1));
        assert_eq!(matrix.get_route(SignalLayer::Video, 1), Some(2));
        assert_eq!(matrix.get_route(SignalLayer::Video, 2), Some(3));
        assert_eq!(matrix.get_route(SignalLayer::Video, 3), Some(4));
        assert!(
            matrix.indexes_consistent(),
            "indexes must be consistent after batch"
        );
    }

    #[test]
    fn test_batch_route_atomic_reject_on_invalid_input() {
        let config = CrosspointMatrixConfig::new(4, 2);
        let mut matrix = CrosspointMatrix::new(config);

        // Set a known route first.
        matrix.route(SignalLayer::Video, 1, 0).expect("setup");

        // Batch contains one valid and one out-of-range request.
        let reqs = vec![
            RouteRequest::new(SignalLayer::Video, 2, 0),
            RouteRequest::new(SignalLayer::Video, 99, 1), // bad input
        ];

        let err = matrix.route_batch(&reqs).expect_err("batch must fail");
        assert!(
            matches!(err, CrosspointError::InputOutOfRange(99)),
            "must report the out-of-range input"
        );

        // Output 0 must still carry input 1 (batch rolled back).
        assert_eq!(
            matrix.get_route(SignalLayer::Video, 0),
            Some(1),
            "route must not have changed after rejected batch"
        );
    }

    #[test]
    fn test_batch_route_atomic_reject_on_locked() {
        let config = CrosspointMatrixConfig::new(8, 4);
        let mut matrix = CrosspointMatrix::new(config);

        matrix.route(SignalLayer::Video, 2, 0).expect("setup");
        matrix.lock(SignalLayer::Video, 1).expect("lock output 1");

        // Batch with a route targeting the locked output.
        let reqs = vec![
            RouteRequest::new(SignalLayer::Video, 3, 0),
            RouteRequest::new(SignalLayer::Video, 3, 1), // locked
        ];

        let err = matrix.route_batch(&reqs).expect_err("batch must fail");
        assert!(
            matches!(err, CrosspointError::Locked { output: 1, .. }),
            "must report the locked output"
        );

        // Output 0 must be unchanged.
        assert_eq!(matrix.get_route(SignalLayer::Video, 0), Some(2));
    }

    #[test]
    fn test_route_request_construction() {
        let req = RouteRequest::new(SignalLayer::Audio, 3, 7);
        assert!(matches!(req.layer, SignalLayer::Audio));
        assert_eq!(req.input, 3);
        assert_eq!(req.output, 7);
    }
}
