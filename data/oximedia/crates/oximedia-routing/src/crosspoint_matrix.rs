//! Audio/video crosspoint routing matrix.
//!
//! Provides a full `NxM` matrix routing system where any input can be connected
//! to any output, with support for protected crosspoints.

#![allow(dead_code)]

/// The state of a single crosspoint (input/output intersection).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CrosspointState {
    /// No connection — the crosspoint is open.
    #[default]
    Open,
    /// Active connection — signal flows from input to output.
    Closed,
    /// Protected crosspoint — cannot be changed without explicit override.
    Protected,
}

impl CrosspointState {
    /// Returns `true` if a routing connection is active (i.e. `Closed`).
    #[must_use]
    pub fn allows_routing(&self) -> bool {
        matches!(self, Self::Closed)
    }
}

/// A single element in the routing matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Crosspoint {
    /// Input index.
    pub input: u32,
    /// Output index.
    pub output: u32,
    /// Current state of this crosspoint.
    pub state: CrosspointState,
}

impl Crosspoint {
    /// Returns `true` if this crosspoint is actively routing (state is `Closed`).
    #[must_use]
    pub fn is_routed(&self) -> bool {
        self.state.allows_routing()
    }
}

/// Full `num_inputs × num_outputs` routing matrix.
#[derive(Debug, Clone)]
pub struct RoutingMatrix {
    crosspoints: Vec<Crosspoint>,
    /// Number of inputs.
    pub num_inputs: u32,
    /// Number of outputs.
    pub num_outputs: u32,
}

impl RoutingMatrix {
    /// Creates a new matrix with all crosspoints in the `Open` state.
    #[must_use]
    pub fn new(inputs: u32, outputs: u32) -> Self {
        let mut crosspoints = Vec::with_capacity((inputs * outputs) as usize);
        for i in 0..inputs {
            for o in 0..outputs {
                crosspoints.push(Crosspoint {
                    input: i,
                    output: o,
                    state: CrosspointState::Open,
                });
            }
        }
        Self {
            crosspoints,
            num_inputs: inputs,
            num_outputs: outputs,
        }
    }

    fn find_mut(&mut self, input: u32, output: u32) -> Option<&mut Crosspoint> {
        self.crosspoints
            .iter_mut()
            .find(|c| c.input == input && c.output == output)
    }

    fn find(&self, input: u32, output: u32) -> Option<&Crosspoint> {
        self.crosspoints
            .iter()
            .find(|c| c.input == input && c.output == output)
    }

    /// Connects `input` to `output`, setting the crosspoint to `Closed`.
    ///
    /// Returns `true` if the connection was made, `false` if the crosspoint is
    /// `Protected` or the indices are out of range.
    pub fn connect(&mut self, input: u32, output: u32) -> bool {
        if input >= self.num_inputs || output >= self.num_outputs {
            return false;
        }
        match self.find_mut(input, output) {
            Some(cp) if cp.state != CrosspointState::Protected => {
                cp.state = CrosspointState::Closed;
                true
            }
            _ => false,
        }
    }

    /// Opens the crosspoint at `(input, output)`, unless it is `Protected`.
    pub fn disconnect(&mut self, input: u32, output: u32) {
        if let Some(cp) = self.find_mut(input, output) {
            if cp.state != CrosspointState::Protected {
                cp.state = CrosspointState::Open;
            }
        }
    }

    /// Returns the current state of the crosspoint at `(input, output)`.
    /// Returns `Open` if the indices are out of range.
    #[must_use]
    pub fn get_state(&self, input: u32, output: u32) -> CrosspointState {
        self.find(input, output)
            .map(|c| c.state)
            .unwrap_or(CrosspointState::Open)
    }

    /// Returns all input indices currently connected to the given output.
    #[must_use]
    pub fn inputs_for(&self, output: u32) -> Vec<u32> {
        self.crosspoints
            .iter()
            .filter(|c| c.output == output && c.is_routed())
            .map(|c| c.input)
            .collect()
    }

    /// Returns all output indices that the given input is connected to.
    #[must_use]
    pub fn outputs_for(&self, input: u32) -> Vec<u32> {
        self.crosspoints
            .iter()
            .filter(|c| c.input == input && c.is_routed())
            .map(|c| c.output)
            .collect()
    }

    /// Returns the total number of active (`Closed`) connections.
    #[must_use]
    pub fn active_connections(&self) -> usize {
        self.crosspoints.iter().filter(|c| c.is_routed()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crosspoint_state_allows_routing_closed() {
        assert!(CrosspointState::Closed.allows_routing());
    }

    #[test]
    fn test_crosspoint_state_allows_routing_open() {
        assert!(!CrosspointState::Open.allows_routing());
    }

    #[test]
    fn test_crosspoint_state_allows_routing_protected() {
        assert!(!CrosspointState::Protected.allows_routing());
    }

    #[test]
    fn test_crosspoint_is_routed() {
        let cp = Crosspoint {
            input: 0,
            output: 0,
            state: CrosspointState::Closed,
        };
        assert!(cp.is_routed());
    }

    #[test]
    fn test_crosspoint_not_routed_when_open() {
        let cp = Crosspoint {
            input: 0,
            output: 0,
            state: CrosspointState::Open,
        };
        assert!(!cp.is_routed());
    }

    #[test]
    fn test_matrix_new_all_open() {
        let m = RoutingMatrix::new(4, 4);
        assert_eq!(m.active_connections(), 0);
        assert_eq!(m.get_state(0, 0), CrosspointState::Open);
    }

    #[test]
    fn test_matrix_connect_success() {
        let mut m = RoutingMatrix::new(4, 4);
        assert!(m.connect(0, 0));
        assert_eq!(m.get_state(0, 0), CrosspointState::Closed);
    }

    #[test]
    fn test_matrix_connect_out_of_range() {
        let mut m = RoutingMatrix::new(4, 4);
        assert!(!m.connect(10, 0));
    }

    #[test]
    fn test_matrix_disconnect() {
        let mut m = RoutingMatrix::new(4, 4);
        m.connect(1, 2);
        m.disconnect(1, 2);
        assert_eq!(m.get_state(1, 2), CrosspointState::Open);
    }

    #[test]
    fn test_matrix_active_connections() {
        let mut m = RoutingMatrix::new(4, 4);
        m.connect(0, 0);
        m.connect(1, 1);
        assert_eq!(m.active_connections(), 2);
    }

    #[test]
    fn test_matrix_inputs_for() {
        let mut m = RoutingMatrix::new(4, 4);
        m.connect(0, 2);
        m.connect(1, 2);
        let inputs = m.inputs_for(2);
        assert_eq!(inputs.len(), 2);
        assert!(inputs.contains(&0));
        assert!(inputs.contains(&1));
    }

    #[test]
    fn test_matrix_outputs_for() {
        let mut m = RoutingMatrix::new(4, 4);
        m.connect(0, 1);
        m.connect(0, 3);
        let outputs = m.outputs_for(0);
        assert_eq!(outputs.len(), 2);
        assert!(outputs.contains(&1));
        assert!(outputs.contains(&3));
    }

    #[test]
    fn test_matrix_protected_cannot_connect() {
        let mut m = RoutingMatrix::new(4, 4);
        // Manually set a crosspoint to Protected
        if let Some(cp) = m
            .crosspoints
            .iter_mut()
            .find(|c| c.input == 2 && c.output == 2)
        {
            cp.state = CrosspointState::Protected;
        }
        assert!(!m.connect(2, 2));
        assert_eq!(m.get_state(2, 2), CrosspointState::Protected);
    }

    #[test]
    fn test_matrix_protected_cannot_disconnect() {
        let mut m = RoutingMatrix::new(4, 4);
        if let Some(cp) = m
            .crosspoints
            .iter_mut()
            .find(|c| c.input == 3 && c.output == 3)
        {
            cp.state = CrosspointState::Protected;
        }
        m.disconnect(3, 3);
        assert_eq!(m.get_state(3, 3), CrosspointState::Protected);
    }
}
