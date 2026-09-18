//! Video routing matrix for professional broadcast switchers.
//!
//! Provides a full any-to-any video crosspoint matrix with locking,
//! salvo-based batch connection, and per-output source tracking.

#![allow(dead_code)]

/// Describes the size of a routing matrix in terms of inputs and outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouterSize {
    /// Number of input sources.
    pub inputs: usize,
    /// Number of output destinations.
    pub outputs: usize,
}

impl RouterSize {
    /// Returns `true` when both inputs and outputs are greater than zero.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.inputs > 0 && self.outputs > 0
    }

    /// Returns the total number of crosspoints (`inputs × outputs`).
    #[must_use]
    pub fn total_crosspoints(&self) -> usize {
        self.inputs * self.outputs
    }
}

/// A video routing matrix that maps outputs to source inputs.
///
/// Each output can be connected to exactly one input, or left disconnected.
/// Outputs can be individually locked to prevent accidental changes.
#[derive(Debug, Clone)]
pub struct VideoRouter {
    /// The size of this routing matrix.
    pub size: RouterSize,
    /// For each output index, the connected input index (if any).
    crosspoints: Vec<Option<usize>>,
    /// Lock state per output.
    locked: Vec<bool>,
}

impl VideoRouter {
    /// Creates a new `VideoRouter` with the given size.
    ///
    /// All outputs start disconnected and unlocked.
    #[must_use]
    pub fn new(size: RouterSize) -> Self {
        let n = size.outputs;
        Self {
            size,
            crosspoints: vec![None; n],
            locked: vec![false; n],
        }
    }

    /// Connects `input` to `output`.
    ///
    /// Returns `false` if the output is locked or indices are out of range.
    pub fn connect(&mut self, input: usize, output: usize) -> bool {
        if output >= self.size.outputs || input >= self.size.inputs {
            return false;
        }
        if self.locked[output] {
            return false;
        }
        self.crosspoints[output] = Some(input);
        true
    }

    /// Disconnects `output` from any input.
    ///
    /// Does nothing if the output is locked or out of range.
    pub fn disconnect(&mut self, output: usize) {
        if output < self.size.outputs && !self.locked[output] {
            self.crosspoints[output] = None;
        }
    }

    /// Returns the source input connected to `output`, or `None`.
    #[must_use]
    pub fn get_source(&self, output: usize) -> Option<usize> {
        self.crosspoints.get(output).copied().flatten()
    }

    /// Returns all output indices that are currently connected to `input`.
    #[must_use]
    pub fn all_outputs_for_input(&self, input: usize) -> Vec<usize> {
        self.crosspoints
            .iter()
            .enumerate()
            .filter_map(
                |(out, src)| {
                    if *src == Some(input) {
                        Some(out)
                    } else {
                        None
                    }
                },
            )
            .collect()
    }

    /// Returns `true` if `output` is currently locked.
    #[must_use]
    pub fn is_locked(&self, output: usize) -> bool {
        self.locked.get(output).copied().unwrap_or(false)
    }

    /// Locks `output`, preventing any connection or disconnection.
    pub fn lock(&mut self, output: usize) {
        if output < self.size.outputs {
            self.locked[output] = true;
        }
    }

    /// Unlocks `output`, allowing connection and disconnection.
    pub fn unlock(&mut self, output: usize) {
        if output < self.size.outputs {
            self.locked[output] = false;
        }
    }
}

/// A salvo is a batch of (input, output) connections to be applied atomically.
///
/// Locked outputs are silently skipped during execution.
#[derive(Debug, Clone, Default)]
pub struct RouterSalvo {
    /// The list of (input, output) pairs to connect.
    pub connections: Vec<(usize, usize)>,
}

impl RouterSalvo {
    /// Creates an empty `RouterSalvo`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a connection to the salvo.
    pub fn add(&mut self, input: usize, output: usize) {
        self.connections.push((input, output));
    }

    /// Executes all connections in the salvo against `router`.
    ///
    /// Locked or out-of-range outputs are silently skipped.
    pub fn execute(&self, router: &mut VideoRouter) {
        for &(input, output) in &self.connections {
            router.connect(input, output);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_router(inputs: usize, outputs: usize) -> VideoRouter {
        VideoRouter::new(RouterSize { inputs, outputs })
    }

    #[test]
    fn test_router_size_is_valid() {
        assert!(RouterSize {
            inputs: 4,
            outputs: 4
        }
        .is_valid());
        assert!(!RouterSize {
            inputs: 0,
            outputs: 4
        }
        .is_valid());
        assert!(!RouterSize {
            inputs: 4,
            outputs: 0
        }
        .is_valid());
        assert!(!RouterSize {
            inputs: 0,
            outputs: 0
        }
        .is_valid());
    }

    #[test]
    fn test_total_crosspoints() {
        let size = RouterSize {
            inputs: 8,
            outputs: 4,
        };
        assert_eq!(size.total_crosspoints(), 32);
    }

    #[test]
    fn test_connect_and_get_source() {
        let mut r = make_router(4, 4);
        assert!(r.connect(2, 1));
        assert_eq!(r.get_source(1), Some(2));
    }

    #[test]
    fn test_disconnect() {
        let mut r = make_router(4, 4);
        r.connect(0, 0);
        r.disconnect(0);
        assert_eq!(r.get_source(0), None);
    }

    #[test]
    fn test_connect_out_of_range_returns_false() {
        let mut r = make_router(4, 4);
        assert!(!r.connect(0, 10));
        assert!(!r.connect(10, 0));
    }

    #[test]
    fn test_get_source_disconnected() {
        let r = make_router(4, 4);
        assert_eq!(r.get_source(0), None);
    }

    #[test]
    fn test_all_outputs_for_input() {
        let mut r = make_router(4, 4);
        r.connect(1, 0);
        r.connect(1, 2);
        r.connect(1, 3);
        let outputs = r.all_outputs_for_input(1);
        assert_eq!(outputs.len(), 3);
        assert!(outputs.contains(&0));
        assert!(outputs.contains(&2));
        assert!(outputs.contains(&3));
    }

    #[test]
    fn test_lock_prevents_connect() {
        let mut r = make_router(4, 4);
        r.connect(0, 0);
        r.lock(0);
        assert!(!r.connect(1, 0));
        assert_eq!(r.get_source(0), Some(0)); // unchanged
    }

    #[test]
    fn test_lock_prevents_disconnect() {
        let mut r = make_router(4, 4);
        r.connect(0, 0);
        r.lock(0);
        r.disconnect(0);
        assert_eq!(r.get_source(0), Some(0)); // unchanged
    }

    #[test]
    fn test_unlock_allows_connect() {
        let mut r = make_router(4, 4);
        r.lock(0);
        assert!(!r.connect(0, 0));
        r.unlock(0);
        assert!(r.connect(0, 0));
    }

    #[test]
    fn test_is_locked() {
        let mut r = make_router(4, 4);
        assert!(!r.is_locked(0));
        r.lock(0);
        assert!(r.is_locked(0));
        r.unlock(0);
        assert!(!r.is_locked(0));
    }

    #[test]
    fn test_salvo_execute() {
        let mut r = make_router(8, 8);
        let mut salvo = RouterSalvo::new();
        salvo.add(0, 0);
        salvo.add(3, 5);
        salvo.add(7, 7);
        salvo.execute(&mut r);
        assert_eq!(r.get_source(0), Some(0));
        assert_eq!(r.get_source(5), Some(3));
        assert_eq!(r.get_source(7), Some(7));
    }

    #[test]
    fn test_salvo_skips_locked() {
        let mut r = make_router(4, 4);
        r.connect(3, 2);
        r.lock(2);
        let mut salvo = RouterSalvo::new();
        salvo.add(0, 2); // should be skipped
        salvo.add(1, 3); // should succeed
        salvo.execute(&mut r);
        assert_eq!(r.get_source(2), Some(3)); // locked, unchanged
        assert_eq!(r.get_source(3), Some(1));
    }
}
