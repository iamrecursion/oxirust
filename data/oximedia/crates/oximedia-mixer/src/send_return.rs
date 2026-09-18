//! Mixer send/return bus routing for `OxiMedia`.
//!
//! This module models auxiliary send and return bus pathways used in
//! professional digital audio workstations for effect bussing (reverb,
//! delay, etc.) and monitor mixes.

#![allow(dead_code)]

/// Whether a send is tapped before or after the channel fader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SendType {
    /// Signal is tapped *before* the channel fader — level changes do not
    /// affect the send.
    Pre,
    /// Signal is tapped *after* the channel fader — the send level tracks
    /// the fader.
    Post,
}

impl SendType {
    /// Returns `true` if this is a pre-fader send.
    #[must_use]
    pub fn is_pre_fader(self) -> bool {
        matches!(self, Self::Pre)
    }
}

/// A single auxiliary send from a channel strip to a destination bus.
#[derive(Debug, Clone)]
pub struct AuxSend {
    /// Index of the destination bus.
    pub destination_bus: u32,
    /// Send level in the range 0.0 (silence) to 1.0 (unity gain).
    pub level: f32,
    /// Pre- or post-fader tap point.
    pub send_type: SendType,
    /// Whether this send is currently active.
    pub enabled: bool,
}

impl AuxSend {
    /// Create a new aux send.
    #[must_use]
    pub fn new(destination_bus: u32, level: f32, send_type: SendType) -> Self {
        Self {
            destination_bus,
            level: level.clamp(0.0, 1.0),
            send_type,
            enabled: true,
        }
    }

    /// Returns the effective send level: `0.0` when disabled, `level` when enabled.
    #[must_use]
    pub fn effective_level(&self) -> f32 {
        if self.enabled {
            self.level
        } else {
            0.0
        }
    }
}

/// A return bus that receives signal from one or more aux sends.
#[derive(Debug, Clone)]
pub struct ReturnBus {
    /// Unique bus identifier.
    pub bus_id: u32,
    /// Return level in the range 0.0 to 1.0.
    pub level: f32,
    /// Pan position: -1.0 = full left, 0.0 = centre, 1.0 = full right.
    pub pan: f32,
    /// Whether this bus is muted.
    pub muted: bool,
}

impl ReturnBus {
    /// Create a new return bus centred and un-muted at unity gain.
    #[must_use]
    pub fn new(bus_id: u32) -> Self {
        Self {
            bus_id,
            level: 1.0,
            pan: 0.0,
            muted: false,
        }
    }

    /// Effective gain after muting: `0.0` when muted, `level` when active.
    #[must_use]
    pub fn effective_gain(&self) -> f32 {
        if self.muted {
            0.0
        } else {
            self.level
        }
    }

    /// Left/right gain pair derived from pan position using the "constant-power" law.
    ///
    /// Returns `(left_gain, right_gain)` both in 0.0–1.0.
    #[must_use]
    pub fn pan_gains(&self) -> (f32, f32) {
        let p = self.pan.clamp(-1.0, 1.0);
        // Map pan → angle in [0, π/2]
        let angle = (p + 1.0) * std::f32::consts::FRAC_PI_4;
        let right = angle.sin();
        let left = angle.cos();
        (left * self.effective_gain(), right * self.effective_gain())
    }
}

/// A matrix of auxiliary sends originating from a single channel strip.
#[derive(Debug, Clone, Default)]
pub struct SendMatrix {
    /// All sends registered on this channel.
    pub sends: Vec<AuxSend>,
}

impl SendMatrix {
    /// Create an empty send matrix.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a send to the matrix.
    pub fn add_send(&mut self, send: AuxSend) {
        self.sends.push(send);
    }

    /// Remove the send going to `destination_bus`.  Does nothing if not found.
    pub fn remove_send(&mut self, destination_bus: u32) {
        self.sends.retain(|s| s.destination_bus != destination_bus);
    }

    /// Sum of all effective send levels across every registered send.
    #[must_use]
    pub fn total_send_level(&self) -> f32 {
        self.sends.iter().map(AuxSend::effective_level).sum()
    }

    /// Number of sends that are currently enabled.
    #[must_use]
    pub fn active_sends(&self) -> usize {
        self.sends.iter().filter(|s| s.enabled).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- SendType ----

    #[test]
    fn test_pre_fader_is_pre() {
        assert!(SendType::Pre.is_pre_fader());
    }

    #[test]
    fn test_post_fader_is_not_pre() {
        assert!(!SendType::Post.is_pre_fader());
    }

    // ---- AuxSend ----

    #[test]
    fn test_aux_send_effective_level_when_enabled() {
        let send = AuxSend::new(1, 0.75, SendType::Post);
        assert!((send.effective_level() - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_aux_send_effective_level_when_disabled() {
        let mut send = AuxSend::new(1, 0.75, SendType::Post);
        send.enabled = false;
        assert!(send.effective_level().abs() < f32::EPSILON);
    }

    #[test]
    fn test_aux_send_level_clamped() {
        let send = AuxSend::new(2, 2.5, SendType::Pre);
        assert!((send.level - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_aux_send_created_enabled() {
        let send = AuxSend::new(3, 0.5, SendType::Post);
        assert!(send.enabled);
    }

    // ---- ReturnBus ----

    #[test]
    fn test_return_bus_new_defaults() {
        let bus = ReturnBus::new(0);
        assert!((bus.level - 1.0).abs() < f32::EPSILON);
        assert!(bus.pan.abs() < f32::EPSILON);
        assert!(!bus.muted);
    }

    #[test]
    fn test_return_bus_effective_gain_muted() {
        let mut bus = ReturnBus::new(0);
        bus.muted = true;
        assert!(bus.effective_gain().abs() < f32::EPSILON);
    }

    #[test]
    fn test_return_bus_effective_gain_active() {
        let bus = ReturnBus::new(0);
        assert!((bus.effective_gain() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_pan_gains_centre() {
        let bus = ReturnBus::new(0);
        let (l, r) = bus.pan_gains();
        // At centre the equal-power law gives equal L/R
        assert!((l - r).abs() < 1e-5, "l={l} r={r}");
    }

    #[test]
    fn test_pan_gains_full_right() {
        let mut bus = ReturnBus::new(0);
        bus.pan = 1.0;
        let (l, r) = bus.pan_gains();
        assert!(r > l, "right should dominate when panned right");
    }

    #[test]
    fn test_pan_gains_full_left() {
        let mut bus = ReturnBus::new(0);
        bus.pan = -1.0;
        let (l, r) = bus.pan_gains();
        assert!(l > r, "left should dominate when panned left");
    }

    // ---- SendMatrix ----

    #[test]
    fn test_send_matrix_add_and_count() {
        let mut matrix = SendMatrix::new();
        matrix.add_send(AuxSend::new(1, 0.5, SendType::Post));
        matrix.add_send(AuxSend::new(2, 0.5, SendType::Pre));
        assert_eq!(matrix.active_sends(), 2);
    }

    #[test]
    fn test_send_matrix_total_send_level() {
        let mut matrix = SendMatrix::new();
        matrix.add_send(AuxSend::new(1, 0.4, SendType::Post));
        matrix.add_send(AuxSend::new(2, 0.6, SendType::Post));
        assert!((matrix.total_send_level() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_send_matrix_remove_send() {
        let mut matrix = SendMatrix::new();
        matrix.add_send(AuxSend::new(1, 0.5, SendType::Post));
        matrix.add_send(AuxSend::new(2, 0.5, SendType::Post));
        matrix.remove_send(1);
        assert_eq!(matrix.sends.len(), 1);
        assert_eq!(matrix.sends[0].destination_bus, 2);
    }

    #[test]
    fn test_send_matrix_active_sends_skips_disabled() {
        let mut matrix = SendMatrix::new();
        let mut send = AuxSend::new(1, 0.8, SendType::Post);
        send.enabled = false;
        matrix.add_send(send);
        matrix.add_send(AuxSend::new(2, 0.8, SendType::Post));
        assert_eq!(matrix.active_sends(), 1);
    }
}
