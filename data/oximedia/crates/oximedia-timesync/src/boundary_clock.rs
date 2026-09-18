//! IEEE 1588 Boundary Clock implementation.
//!
//! A Boundary Clock (BC) has multiple PTP ports and can act as a master on
//! some ports while synchronising as a slave on another port.  This module
//! provides the data structures and state-machine logic for a BC, without
//! any network I/O.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// DelayMechanism
// ---------------------------------------------------------------------------

/// Delay measurement mechanism used by a PTP port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelayMechanism {
    /// End-to-end delay measurement (uses Delay_Req / Delay_Resp).
    E2E,
    /// Peer-to-peer delay measurement (uses Pdelay_Req / Pdelay_Resp).
    P2P,
    /// No delay mechanism is used on this port.
    NoMechanism,
}

impl DelayMechanism {
    /// Returns `true` when the port uses peer-delay (P2P) messages.
    #[must_use]
    pub fn uses_pdelay(&self) -> bool {
        matches!(self, DelayMechanism::P2P)
    }
}

// ---------------------------------------------------------------------------
// PortState
// ---------------------------------------------------------------------------

/// State of a PTP port as defined by IEEE 1588-2019.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortState {
    /// Port is initialising.
    Initializing,
    /// Port has encountered a fault and is not operational.
    Faulty,
    /// Port has been administratively disabled.
    Disabled,
    /// Port is listening for Announce messages.
    Listening,
    /// Port is about to become master.
    PreMaster,
    /// Port is acting as a PTP master.
    Master,
    /// Port is in passive mode (not master, not slave).
    Passive,
    /// Port is in the process of becoming a slave.
    Uncalibrated,
    /// Port is synchronised to a master.
    Slave,
}

impl PortState {
    /// Returns `true` for states where the port participates actively in
    /// the PTP protocol (not Faulty, Disabled, or Initializing).
    #[must_use]
    pub fn is_active(&self) -> bool {
        !matches!(
            self,
            PortState::Faulty | PortState::Disabled | PortState::Initializing
        )
    }

    /// Returns `true` when the port is currently acting as a master.
    #[must_use]
    pub fn is_master(&self) -> bool {
        matches!(self, PortState::Master | PortState::PreMaster)
    }

    /// Returns `true` when the port is synchronised to an upstream master.
    #[must_use]
    pub fn is_slave(&self) -> bool {
        matches!(self, PortState::Slave | PortState::Uncalibrated)
    }
}

// ---------------------------------------------------------------------------
// PtpPort
// ---------------------------------------------------------------------------

/// A single port on a Boundary Clock.
#[derive(Debug, Clone)]
pub struct PtpPort {
    /// Unique port identifier within the clock (1-based).
    pub port_id: u16,
    /// Current port state.
    pub state: PortState,
    /// Delay mechanism used by this port.
    pub delay_mechanism: DelayMechanism,
    /// Log base-2 of the mean message interval (e.g. 0 → 1 s, -3 → 125 ms).
    pub log_message_interval: i8,
}

impl PtpPort {
    /// Create a new port starting in the `Initializing` state.
    #[must_use]
    pub fn new(port_id: u16, delay_mechanism: DelayMechanism) -> Self {
        Self {
            port_id,
            state: PortState::Initializing,
            delay_mechanism,
            log_message_interval: 0,
        }
    }

    /// Attempt to transition the port to `new_state`.
    ///
    /// Returns `true` when the transition is allowed and has been applied.
    /// The allowed transitions follow a simplified subset of the IEEE 1588
    /// state machine:
    ///
    /// - **Initializing** → any state
    /// - **Faulty** → Initializing
    /// - **Disabled** → Initializing
    /// - **Listening** → PreMaster, Master, Slave, Passive, Faulty, Disabled
    /// - **PreMaster** → Master, Listening, Faulty, Disabled
    /// - **Master** → Listening, Passive, Faulty, Disabled
    /// - **Passive** → Listening, Faulty, Disabled
    /// - **Uncalibrated** → Slave, Listening, Faulty, Disabled
    /// - **Slave** → Listening, Uncalibrated, Passive, Faulty, Disabled
    pub fn transition_to(&mut self, new_state: PortState) -> bool {
        let allowed = match self.state {
            PortState::Initializing => true,
            PortState::Faulty | PortState::Disabled => {
                matches!(new_state, PortState::Initializing)
            }
            PortState::Listening => matches!(
                new_state,
                PortState::PreMaster
                    | PortState::Master
                    | PortState::Slave
                    | PortState::Passive
                    | PortState::Faulty
                    | PortState::Disabled
            ),
            PortState::PreMaster => matches!(
                new_state,
                PortState::Master | PortState::Listening | PortState::Faulty | PortState::Disabled
            ),
            PortState::Master => matches!(
                new_state,
                PortState::Listening | PortState::Passive | PortState::Faulty | PortState::Disabled
            ),
            PortState::Passive => matches!(
                new_state,
                PortState::Listening | PortState::Faulty | PortState::Disabled
            ),
            PortState::Uncalibrated => matches!(
                new_state,
                PortState::Slave | PortState::Listening | PortState::Faulty | PortState::Disabled
            ),
            PortState::Slave => matches!(
                new_state,
                PortState::Listening
                    | PortState::Uncalibrated
                    | PortState::Passive
                    | PortState::Faulty
                    | PortState::Disabled
            ),
        };

        if allowed {
            self.state = new_state;
        }
        allowed
    }
}

// ---------------------------------------------------------------------------
// BoundaryClock
// ---------------------------------------------------------------------------

/// IEEE 1588 Boundary Clock.
///
/// A BC terminates the PTP protocol on each port and forwards timing
/// information between domains.
#[derive(Debug, Clone)]
pub struct BoundaryClock {
    /// PTP domain number (0–127).
    pub domain: u8,
    /// All ports belonging to this clock.
    ports: Vec<PtpPort>,
    /// Port ID of the port currently in Slave state (the upstream master port).
    pub selected_master_port: Option<u16>,
}

impl BoundaryClock {
    /// Create a new Boundary Clock for the given `domain`.
    #[must_use]
    pub fn new(domain: u8) -> Self {
        Self {
            domain,
            ports: Vec::new(),
            selected_master_port: None,
        }
    }

    /// Add a port to the clock.
    pub fn add_port(&mut self, port: PtpPort) {
        self.ports.push(port);
    }

    /// Total number of ports.
    #[must_use]
    pub fn port_count(&self) -> usize {
        self.ports.len()
    }

    /// Return a reference to the upstream master port (the Slave-state port),
    /// if one exists.
    #[must_use]
    pub fn master_port(&self) -> Option<&PtpPort> {
        if let Some(id) = self.selected_master_port {
            return self.ports.iter().find(|p| p.port_id == id);
        }
        // Fall back to any port in Slave state.
        self.ports.iter().find(|p| p.state.is_slave())
    }

    /// Return references to all ports that are in Master or PreMaster state.
    #[must_use]
    pub fn slave_ports(&self) -> Vec<&PtpPort> {
        self.ports.iter().filter(|p| p.state.is_master()).collect()
    }

    /// Return a mutable reference to the port with the given `port_id`.
    pub fn port_mut(&mut self, port_id: u16) -> Option<&mut PtpPort> {
        self.ports.iter_mut().find(|p| p.port_id == port_id)
    }

    /// Return an immutable reference to the port with the given `port_id`.
    #[must_use]
    pub fn port(&self, port_id: u16) -> Option<&PtpPort> {
        self.ports.iter().find(|p| p.port_id == port_id)
    }

    /// Number of ports currently in an active state.
    #[must_use]
    pub fn active_port_count(&self) -> usize {
        self.ports.iter().filter(|p| p.state.is_active()).count()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_port(id: u16) -> PtpPort {
        PtpPort::new(id, DelayMechanism::E2E)
    }

    // --- DelayMechanism ---

    #[test]
    fn test_delay_mechanism_p2p_uses_pdelay() {
        assert!(DelayMechanism::P2P.uses_pdelay());
    }

    #[test]
    fn test_delay_mechanism_e2e_no_pdelay() {
        assert!(!DelayMechanism::E2E.uses_pdelay());
    }

    #[test]
    fn test_delay_mechanism_no_mechanism_no_pdelay() {
        assert!(!DelayMechanism::NoMechanism.uses_pdelay());
    }

    // --- PortState ---

    #[test]
    fn test_port_state_faulty_not_active() {
        assert!(!PortState::Faulty.is_active());
    }

    #[test]
    fn test_port_state_disabled_not_active() {
        assert!(!PortState::Disabled.is_active());
    }

    #[test]
    fn test_port_state_initializing_not_active() {
        assert!(!PortState::Initializing.is_active());
    }

    #[test]
    fn test_port_state_master_is_active() {
        assert!(PortState::Master.is_active());
    }

    #[test]
    fn test_port_state_slave_is_active() {
        assert!(PortState::Slave.is_active());
    }

    #[test]
    fn test_port_state_master_is_master() {
        assert!(PortState::Master.is_master());
        assert!(PortState::PreMaster.is_master());
        assert!(!PortState::Slave.is_master());
    }

    #[test]
    fn test_port_state_slave_is_slave() {
        assert!(PortState::Slave.is_slave());
        assert!(PortState::Uncalibrated.is_slave());
        assert!(!PortState::Master.is_slave());
    }

    // --- PtpPort transitions ---

    #[test]
    fn test_port_transition_initializing_to_listening() {
        let mut p = make_port(1);
        assert_eq!(p.state, PortState::Initializing);
        assert!(p.transition_to(PortState::Listening));
        assert_eq!(p.state, PortState::Listening);
    }

    #[test]
    fn test_port_transition_listening_to_master() {
        let mut p = make_port(1);
        p.transition_to(PortState::Listening);
        assert!(p.transition_to(PortState::Master));
        assert_eq!(p.state, PortState::Master);
    }

    #[test]
    fn test_port_transition_listening_to_slave() {
        let mut p = make_port(1);
        p.transition_to(PortState::Listening);
        assert!(p.transition_to(PortState::Slave));
        assert_eq!(p.state, PortState::Slave);
    }

    #[test]
    fn test_port_transition_faulty_only_to_initializing() {
        let mut p = make_port(1);
        p.transition_to(PortState::Listening);
        p.transition_to(PortState::Faulty);
        // Faulty → Listening is NOT allowed
        assert!(!p.transition_to(PortState::Listening));
        // Faulty → Initializing IS allowed
        assert!(p.transition_to(PortState::Initializing));
    }

    #[test]
    fn test_port_transition_slave_to_passive() {
        let mut p = make_port(1);
        p.transition_to(PortState::Listening);
        p.transition_to(PortState::Slave);
        assert!(p.transition_to(PortState::Passive));
    }

    // --- BoundaryClock ---

    #[test]
    fn test_boundary_clock_new() {
        let bc = BoundaryClock::new(0);
        assert_eq!(bc.domain, 0);
        assert_eq!(bc.port_count(), 0);
        assert!(bc.master_port().is_none());
    }

    #[test]
    fn test_boundary_clock_add_port() {
        let mut bc = BoundaryClock::new(0);
        bc.add_port(make_port(1));
        bc.add_port(make_port(2));
        assert_eq!(bc.port_count(), 2);
    }

    #[test]
    fn test_boundary_clock_master_port_via_slave_state() {
        let mut bc = BoundaryClock::new(0);
        let mut p1 = make_port(1);
        p1.transition_to(PortState::Listening);
        p1.transition_to(PortState::Slave);
        bc.add_port(p1);
        bc.add_port(make_port(2));
        let mp = bc.master_port();
        assert!(mp.is_some());
        assert_eq!(mp.expect("should succeed in test").port_id, 1);
    }

    #[test]
    fn test_boundary_clock_slave_ports_returns_masters() {
        let mut bc = BoundaryClock::new(0);
        let mut p1 = make_port(1);
        p1.transition_to(PortState::Listening);
        p1.transition_to(PortState::Master);
        let mut p2 = make_port(2);
        p2.transition_to(PortState::Listening);
        p2.transition_to(PortState::Slave);
        bc.add_port(p1);
        bc.add_port(p2);
        let sp = bc.slave_ports();
        assert_eq!(sp.len(), 1);
        assert_eq!(sp[0].port_id, 1);
    }

    #[test]
    fn test_boundary_clock_active_port_count() {
        let mut bc = BoundaryClock::new(0);
        let mut p1 = make_port(1);
        p1.transition_to(PortState::Listening);
        bc.add_port(p1); // active
        bc.add_port(make_port(2)); // Initializing — not active
        assert_eq!(bc.active_port_count(), 1);
    }
}
