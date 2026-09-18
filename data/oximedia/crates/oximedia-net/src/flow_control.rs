#![allow(dead_code)]
//! TCP-inspired flow and congestion control primitives.
//!
//! Provides a sliding congestion window, ACK/NACK processing, and a
//! high-level [`FlowController`] that drives sending decisions.

/// State of the flow controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowControlState {
    /// Normal operation; sending is unrestricted within window.
    Open,
    /// The window has been reduced due to loss or backpressure.
    Throttled,
    /// Complete halt; no packets may be sent.
    Blocked,
    /// Slow-start phase immediately after a reset.
    SlowStart,
}

impl FlowControlState {
    /// Returns `true` when the state prevents or restricts sending.
    #[must_use]
    pub fn is_throttled(self) -> bool {
        matches!(self, Self::Throttled | Self::Blocked)
    }

    /// Returns `true` when sending is completely stopped.
    #[must_use]
    pub fn is_blocked(self) -> bool {
        self == Self::Blocked
    }
}

/// A sliding congestion window tracking in-flight packets.
#[derive(Debug, Clone)]
pub struct CongestionWindow {
    /// Maximum window size in packets.
    max_window: u32,
    /// Current window size in packets.
    window: u32,
    /// Number of packets currently in-flight.
    in_flight: u32,
    /// Slow-start threshold.
    ssthresh: u32,
}

impl CongestionWindow {
    /// Creates a new window with an initial slow-start size.
    #[must_use]
    pub fn new(initial: u32, max_window: u32) -> Self {
        Self {
            max_window,
            window: initial,
            in_flight: 0,
            ssthresh: max_window / 2,
        }
    }

    /// Returns the current window size.
    #[must_use]
    pub fn window_size(&self) -> u32 {
        self.window
    }

    /// Returns the number of packets currently in-flight.
    #[must_use]
    pub fn in_flight(&self) -> u32 {
        self.in_flight
    }

    /// Returns available sending credits (window − in_flight).
    #[must_use]
    pub fn available(&self) -> u32 {
        self.window.saturating_sub(self.in_flight)
    }

    /// Returns `true` when at least one packet may be sent.
    #[must_use]
    pub fn can_send(&self) -> bool {
        self.available() > 0
    }

    /// Records that `n` additional packets have been sent.
    pub fn on_send(&mut self, n: u32) {
        self.in_flight = self.in_flight.saturating_add(n);
    }

    /// Records receipt of an ACK for `n` packets.
    ///
    /// Grows the congestion window using slow-start / congestion-avoidance.
    pub fn on_ack(&mut self, n: u32) {
        self.in_flight = self.in_flight.saturating_sub(n);
        if self.window < self.ssthresh {
            // Slow start: exponential growth
            self.window = (self.window + n).min(self.max_window);
        } else {
            // Congestion avoidance: additive increase
            self.window = (self.window + 1).min(self.max_window);
        }
    }

    /// Records a NACK / loss event.
    ///
    /// Halves the window and updates the slow-start threshold.
    pub fn on_loss(&mut self) {
        self.ssthresh = (self.window / 2).max(2);
        self.window = self.ssthresh;
        self.in_flight = self.in_flight.min(self.window);
    }
}

/// High-level flow controller combining a congestion window with state
/// machine transitions.
#[derive(Debug, Clone)]
pub struct FlowController {
    state: FlowControlState,
    cwindow: CongestionWindow,
    nack_count: u32,
    /// Consecutive NACKs before entering `Blocked` state.
    nack_limit: u32,
}

impl FlowController {
    /// Creates a new controller with a given initial window and NACK limit.
    #[must_use]
    pub fn new(initial_window: u32, max_window: u32, nack_limit: u32) -> Self {
        Self {
            state: FlowControlState::SlowStart,
            cwindow: CongestionWindow::new(initial_window, max_window),
            nack_count: 0,
            nack_limit,
        }
    }

    /// Returns the current flow control state.
    #[must_use]
    pub fn state(&self) -> FlowControlState {
        self.state
    }

    /// Returns the current congestion window size.
    #[must_use]
    pub fn current_window_size(&self) -> u32 {
        self.cwindow.window_size()
    }

    /// Returns `true` when a packet may be sent right now.
    #[must_use]
    pub fn can_send(&self) -> bool {
        !self.state.is_blocked() && self.cwindow.can_send()
    }

    /// Records `n` ACKs, growing the window and relaxing the state if
    /// throttled.
    pub fn ack(&mut self, n: u32) {
        self.cwindow.on_ack(n);
        self.nack_count = self.nack_count.saturating_sub(1);
        if self.state == FlowControlState::Blocked && self.cwindow.can_send() {
            self.state = FlowControlState::Throttled;
        } else if self.state == FlowControlState::Throttled && self.nack_count == 0 {
            self.state = FlowControlState::Open;
        } else if self.state == FlowControlState::SlowStart && self.cwindow.can_send() {
            self.state = FlowControlState::Open;
        }
    }

    /// Records a NACK (negative acknowledgement / loss indication).
    ///
    /// Shrinks the window; after `nack_limit` consecutive NACKs enters
    /// `Blocked`.
    pub fn nack(&mut self) {
        self.cwindow.on_loss();
        self.nack_count += 1;
        if self.nack_count >= self.nack_limit {
            self.state = FlowControlState::Blocked;
        } else {
            self.state = FlowControlState::Throttled;
        }
    }

    /// Resets the controller to slow-start with the original initial window.
    pub fn reset(&mut self) {
        self.nack_count = 0;
        self.state = FlowControlState::SlowStart;
        self.cwindow.in_flight = 0;
        self.cwindow.window = 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_open_not_throttled() {
        assert!(!FlowControlState::Open.is_throttled());
    }

    #[test]
    fn test_state_throttled_is_throttled() {
        assert!(FlowControlState::Throttled.is_throttled());
    }

    #[test]
    fn test_state_blocked_is_throttled() {
        assert!(FlowControlState::Blocked.is_throttled());
    }

    #[test]
    fn test_state_blocked_is_blocked() {
        assert!(FlowControlState::Blocked.is_blocked());
    }

    #[test]
    fn test_state_open_not_blocked() {
        assert!(!FlowControlState::Open.is_blocked());
    }

    #[test]
    fn test_window_can_send_initial() {
        let w = CongestionWindow::new(4, 64);
        assert!(w.can_send());
    }

    #[test]
    fn test_window_available_after_send() {
        let mut w = CongestionWindow::new(4, 64);
        w.on_send(4);
        assert_eq!(w.available(), 0);
        assert!(!w.can_send());
    }

    #[test]
    fn test_window_grows_on_ack() {
        let mut w = CongestionWindow::new(1, 64);
        w.on_send(1);
        w.on_ack(1);
        assert!(w.window_size() >= 2);
    }

    #[test]
    fn test_window_shrinks_on_loss() {
        let mut w = CongestionWindow::new(16, 64);
        w.on_loss();
        assert!(w.window_size() < 16);
    }

    #[test]
    fn test_controller_initial_state_slow_start() {
        let c = FlowController::new(4, 64, 3);
        assert_eq!(c.state(), FlowControlState::SlowStart);
    }

    #[test]
    fn test_controller_ack_transitions_to_open() {
        let mut c = FlowController::new(4, 64, 3);
        c.ack(1);
        assert_eq!(c.state(), FlowControlState::Open);
    }

    #[test]
    fn test_controller_nack_throttles() {
        let mut c = FlowController::new(4, 64, 5);
        c.ack(1); // enter Open first
        c.nack();
        assert_eq!(c.state(), FlowControlState::Throttled);
    }

    #[test]
    fn test_controller_nack_limit_blocks() {
        let mut c = FlowController::new(4, 64, 2);
        c.ack(1);
        c.nack();
        c.nack();
        assert_eq!(c.state(), FlowControlState::Blocked);
    }

    #[test]
    fn test_controller_reset() {
        let mut c = FlowController::new(4, 64, 2);
        c.nack();
        c.reset();
        assert_eq!(c.state(), FlowControlState::SlowStart);
        assert_eq!(c.nack_count, 0);
    }

    #[test]
    fn test_controller_current_window_size() {
        let c = FlowController::new(8, 64, 3);
        assert_eq!(c.current_window_size(), 8);
    }

    #[test]
    fn test_controller_can_send_in_open() {
        let mut c = FlowController::new(4, 64, 3);
        c.ack(1);
        assert!(c.can_send());
    }
}
