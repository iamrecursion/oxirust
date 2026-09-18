//! Auxiliary send/return infrastructure for the professional audio mixer.
//!
//! Aux sends let individual channels contribute signal to an auxiliary bus
//! (typically used for effects returns such as reverb or delay) at either
//! pre-fader or post-fader tap points.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Whether an auxiliary send taps the signal before or after the channel fader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuxSendMode {
    /// Signal is tapped before the channel fader (pre-fader).
    PreFader,
    /// Signal is tapped after the channel fader (post-fader).
    PostFader,
    /// Signal is tapped before the channel EQ (pre-EQ).
    PreEq,
    /// Send is disabled (muted).
    Off,
}

impl AuxSendMode {
    /// Returns `true` if the send is active (not `Off`).
    #[must_use]
    pub fn is_active(self) -> bool {
        self != AuxSendMode::Off
    }
}

/// A calibrated send-level value stored in decibels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AuxSendLevel {
    /// Level in dB.  `-f32::INFINITY` represents silence (−∞ dB).
    pub db: f32,
}

impl AuxSendLevel {
    /// Unity gain (0 dB).
    pub const UNITY: Self = Self { db: 0.0 };
    /// Silence (−∞ dB).
    pub const SILENCE: Self = Self {
        db: f32::NEG_INFINITY,
    };

    /// Create an `AuxSendLevel` from a dB value.
    #[must_use]
    pub fn from_db(db: f32) -> Self {
        Self { db }
    }

    /// Convert the dB level to a linear amplitude multiplier (≥ 0).
    #[must_use]
    pub fn to_linear(self) -> f32 {
        if self.db.is_infinite() && self.db < 0.0 {
            0.0
        } else {
            10.0_f32.powf(self.db / 20.0)
        }
    }

    /// Returns whether this level is at or below silence threshold (< −96 dB).
    #[must_use]
    pub fn is_silent(self) -> bool {
        self.db < -96.0 || self.db.is_infinite()
    }
}

impl Default for AuxSendLevel {
    fn default() -> Self {
        Self::UNITY
    }
}

/// A single auxiliary send on a mixer channel.
///
/// Tracks the tap mode, level, and which bus index it feeds.
#[derive(Debug, Clone)]
pub struct AuxSend {
    /// Target auxiliary bus index.
    pub bus_index: usize,
    /// Tap mode (pre-fader, post-fader, etc.).
    pub mode: AuxSendMode,
    /// Send level.
    pub level: AuxSendLevel,
    /// Whether the send is muted independently of mode.
    pub muted: bool,
}

impl AuxSend {
    /// Create a new `AuxSend` targeting `bus_index` in post-fader mode at unity.
    #[must_use]
    pub fn new(bus_index: usize) -> Self {
        Self {
            bus_index,
            mode: AuxSendMode::PostFader,
            level: AuxSendLevel::UNITY,
            muted: false,
        }
    }

    /// Compute the effective pre-fader level.
    ///
    /// Returns the linear level if the mode is `PreFader` and the send is active,
    /// otherwise returns `0.0`.
    #[must_use]
    pub fn pre_fader_level(&self) -> f32 {
        if self.muted || self.mode != AuxSendMode::PreFader {
            return 0.0;
        }
        self.level.to_linear()
    }

    /// Compute the effective post-fader level.
    ///
    /// Returns the linear level if the mode is `PostFader` and the send is active,
    /// otherwise returns `0.0`.
    #[must_use]
    pub fn post_fader_level(&self) -> f32 {
        if self.muted || self.mode != AuxSendMode::PostFader {
            return 0.0;
        }
        self.level.to_linear()
    }

    /// Effective linear send level regardless of tap mode (0.0 if muted).
    #[must_use]
    pub fn effective_level(&self) -> f32 {
        if self.muted || !self.mode.is_active() {
            return 0.0;
        }
        self.level.to_linear()
    }
}

/// An auxiliary bus that accumulates contributions from multiple sends.
#[derive(Debug, Clone)]
pub struct AuxBus {
    /// Human-readable name of this aux bus.
    pub name: String,
    /// Registered sends (one per contributing channel).
    sends: Vec<AuxSend>,
    /// Master level for this bus.
    pub master_level: AuxSendLevel,
}

impl AuxBus {
    /// Create an empty `AuxBus` with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            sends: Vec::new(),
            master_level: AuxSendLevel::UNITY,
        }
    }

    /// Add a send to this bus.
    pub fn add_send(&mut self, send: AuxSend) {
        self.sends.push(send);
    }

    /// Return the number of registered sends.
    #[must_use]
    pub fn send_count(&self) -> usize {
        self.sends.len()
    }

    /// Mix all active sends for a single sample value.
    ///
    /// Each send contributes `input * effective_level` to the bus output.
    /// The total is then multiplied by the `master_level`.
    #[must_use]
    pub fn mix_bus(&self, input: f32) -> f32 {
        let sum: f32 = self.sends.iter().map(|s| input * s.effective_level()).sum();
        sum * self.master_level.to_linear()
    }

    /// Return an iterator over all registered sends.
    pub fn sends(&self) -> impl Iterator<Item = &AuxSend> {
        self.sends.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aux_send_mode_is_active() {
        assert!(AuxSendMode::PreFader.is_active());
        assert!(AuxSendMode::PostFader.is_active());
        assert!(AuxSendMode::PreEq.is_active());
        assert!(!AuxSendMode::Off.is_active());
    }

    #[test]
    fn test_aux_send_level_unity_linear() {
        let level = AuxSendLevel::UNITY;
        assert!((level.to_linear() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_aux_send_level_silence_linear() {
        let level = AuxSendLevel::SILENCE;
        assert_eq!(level.to_linear(), 0.0);
    }

    #[test]
    fn test_aux_send_level_minus6db() {
        let level = AuxSendLevel::from_db(-6.0);
        // 10^(-6/20) ≈ 0.501
        assert!((level.to_linear() - 0.501).abs() < 0.001);
    }

    #[test]
    fn test_aux_send_level_is_silent() {
        assert!(AuxSendLevel::SILENCE.is_silent());
        assert!(AuxSendLevel::from_db(-100.0).is_silent());
        assert!(!AuxSendLevel::UNITY.is_silent());
    }

    #[test]
    fn test_aux_send_pre_fader_level() {
        let mut send = AuxSend::new(0);
        send.mode = AuxSendMode::PreFader;
        // Should return unity linear
        assert!((send.pre_fader_level() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_aux_send_post_fader_returns_zero_for_pre_fader() {
        let mut send = AuxSend::new(0);
        send.mode = AuxSendMode::PreFader;
        assert_eq!(send.post_fader_level(), 0.0);
    }

    #[test]
    fn test_aux_send_muted_returns_zero() {
        let mut send = AuxSend::new(0);
        send.mode = AuxSendMode::PostFader;
        send.muted = true;
        assert_eq!(send.effective_level(), 0.0);
        assert_eq!(send.post_fader_level(), 0.0);
    }

    #[test]
    fn test_aux_send_off_mode_returns_zero() {
        let mut send = AuxSend::new(0);
        send.mode = AuxSendMode::Off;
        assert_eq!(send.effective_level(), 0.0);
    }

    #[test]
    fn test_aux_bus_add_send() {
        let mut bus = AuxBus::new("FX1");
        bus.add_send(AuxSend::new(0));
        bus.add_send(AuxSend::new(1));
        assert_eq!(bus.send_count(), 2);
    }

    #[test]
    fn test_aux_bus_mix_bus_unity() {
        let mut bus = AuxBus::new("FX1");
        let mut send = AuxSend::new(0);
        send.mode = AuxSendMode::PostFader;
        bus.add_send(send);
        // With one unity-gain post-fader send and unity master, output == input
        assert!((bus.mix_bus(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_aux_bus_mix_bus_two_sends() {
        let mut bus = AuxBus::new("FX2");
        for _ in 0..2 {
            let mut send = AuxSend::new(0);
            send.mode = AuxSendMode::PostFader;
            bus.add_send(send);
        }
        // Two unity sends: output = 2 * input
        assert!((bus.mix_bus(1.0) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_aux_bus_mix_bus_muted_send() {
        let mut bus = AuxBus::new("FX3");
        let mut send = AuxSend::new(0);
        send.mode = AuxSendMode::PostFader;
        send.muted = true;
        bus.add_send(send);
        assert_eq!(bus.mix_bus(1.0), 0.0);
    }

    #[test]
    fn test_aux_bus_name() {
        let bus = AuxBus::new("Reverb");
        assert_eq!(bus.name, "Reverb");
    }
}
