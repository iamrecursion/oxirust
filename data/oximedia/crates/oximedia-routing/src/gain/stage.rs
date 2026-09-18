//! Gain staging with metering for professional audio routing.

use serde::{Deserialize, Serialize};

/// Gain stage with metering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GainStage {
    /// Gain value in dB
    pub gain_db: f32,
    /// Mute state
    pub muted: bool,
    /// Solo state
    pub soloed: bool,
    /// Polarity inversion
    pub inverted: bool,
    /// Current peak level (in dBFS)
    pub peak_level_db: f32,
    /// Current RMS level (in dBFS)
    pub rms_level_db: f32,
    /// Peak hold value
    pub peak_hold_db: f32,
    /// Clip indicator
    pub clipping: bool,
}

impl Default for GainStage {
    fn default() -> Self {
        Self::new()
    }
}

impl GainStage {
    /// Create a new gain stage
    #[must_use]
    pub fn new() -> Self {
        Self {
            gain_db: 0.0,
            muted: false,
            soloed: false,
            inverted: false,
            peak_level_db: f32::NEG_INFINITY,
            rms_level_db: f32::NEG_INFINITY,
            peak_hold_db: f32::NEG_INFINITY,
            clipping: false,
        }
    }

    /// Set gain in dB
    pub fn set_gain(&mut self, gain_db: f32) {
        self.gain_db = gain_db.clamp(-60.0, 12.0);
    }

    /// Set gain in linear scale
    pub fn set_gain_linear(&mut self, gain: f32) {
        self.gain_db = 20.0 * gain.max(0.001).log10();
    }

    /// Get gain in linear scale
    #[must_use]
    pub fn gain_linear(&self) -> f32 {
        10.0_f32.powf(self.gain_db / 20.0)
    }

    /// Toggle mute
    pub fn toggle_mute(&mut self) {
        self.muted = !self.muted;
    }

    /// Toggle solo
    pub fn toggle_solo(&mut self) {
        self.soloed = !self.soloed;
    }

    /// Toggle polarity
    pub fn toggle_polarity(&mut self) {
        self.inverted = !self.inverted;
    }

    /// Update metering levels
    pub fn update_meters(&mut self, peak_db: f32, rms_db: f32) {
        self.peak_level_db = peak_db;
        self.rms_level_db = rms_db;

        if peak_db > self.peak_hold_db {
            self.peak_hold_db = peak_db;
        }

        self.clipping = peak_db >= 0.0;
    }

    /// Reset peak hold
    pub fn reset_peak_hold(&mut self) {
        self.peak_hold_db = f32::NEG_INFINITY;
    }

    /// Reset clip indicator
    pub fn reset_clip(&mut self) {
        self.clipping = false;
    }

    /// Get effective gain (considering mute)
    #[must_use]
    pub fn effective_gain_db(&self) -> f32 {
        if self.muted {
            f32::NEG_INFINITY
        } else {
            self.gain_db
        }
    }

    /// Check if signal is present
    #[must_use]
    pub fn has_signal(&self) -> bool {
        self.peak_level_db > -60.0
    }
}

/// Multi-channel gain stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiChannelGainStage {
    /// Individual channel gain stages
    pub channels: Vec<GainStage>,
    /// Master gain
    pub master_gain_db: f32,
    /// Link channels together
    pub linked: bool,
}

impl MultiChannelGainStage {
    /// Create a new multi-channel gain stage
    #[must_use]
    pub fn new(channel_count: usize) -> Self {
        Self {
            channels: vec![GainStage::new(); channel_count],
            master_gain_db: 0.0,
            linked: false,
        }
    }

    /// Set gain for a specific channel
    pub fn set_channel_gain(&mut self, channel: usize, gain_db: f32) -> Result<(), GainError> {
        if let Some(stage) = self.channels.get_mut(channel) {
            stage.set_gain(gain_db);

            if self.linked {
                for ch in &mut self.channels {
                    ch.set_gain(gain_db);
                }
            }

            Ok(())
        } else {
            Err(GainError::InvalidChannel(channel))
        }
    }

    /// Set master gain
    pub fn set_master_gain(&mut self, gain_db: f32) {
        self.master_gain_db = gain_db.clamp(-60.0, 12.0);
    }

    /// Get total gain for a channel (channel + master)
    #[must_use]
    pub fn total_gain_db(&self, channel: usize) -> Option<f32> {
        self.channels
            .get(channel)
            .map(|stage| stage.effective_gain_db() + self.master_gain_db)
    }

    /// Mute all channels
    pub fn mute_all(&mut self) {
        for channel in &mut self.channels {
            channel.muted = true;
        }
    }

    /// Unmute all channels
    pub fn unmute_all(&mut self) {
        for channel in &mut self.channels {
            channel.muted = false;
        }
    }

    /// Reset all peak holds
    pub fn reset_all_peaks(&mut self) {
        for channel in &mut self.channels {
            channel.reset_peak_hold();
        }
    }

    /// Check if any channel is clipping
    #[must_use]
    pub fn any_clipping(&self) -> bool {
        self.channels.iter().any(|ch| ch.clipping)
    }

    /// Get channel count
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }
}

/// Errors that can occur in gain operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum GainError {
    /// Invalid channel index
    #[error("Invalid channel: {0}")]
    InvalidChannel(usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gain_stage_creation() {
        let stage = GainStage::new();
        assert_eq!(stage.gain_db, 0.0);
        assert!(!stage.muted);
        assert!(!stage.soloed);
    }

    #[test]
    fn test_set_gain() {
        let mut stage = GainStage::new();
        stage.set_gain(-6.0);
        assert!((stage.gain_db - (-6.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gain_clamping() {
        let mut stage = GainStage::new();
        stage.set_gain(100.0);
        assert!(stage.gain_db <= 12.0);

        stage.set_gain(-100.0);
        assert!(stage.gain_db >= -60.0);
    }

    #[test]
    fn test_linear_gain() {
        let mut stage = GainStage::new();
        stage.set_gain(-6.0);

        let linear = stage.gain_linear();
        assert!((linear - 0.501_187).abs() < 0.001);
    }

    #[test]
    fn test_mute() {
        let mut stage = GainStage::new();
        assert!(!stage.muted);

        stage.toggle_mute();
        assert!(stage.muted);
        assert_eq!(stage.effective_gain_db(), f32::NEG_INFINITY);
    }

    #[test]
    fn test_metering() {
        let mut stage = GainStage::new();
        stage.update_meters(-6.0, -12.0);

        assert!((stage.peak_level_db - (-6.0)).abs() < f32::EPSILON);
        assert!((stage.rms_level_db - (-12.0)).abs() < f32::EPSILON);
        assert!(!stage.clipping);
    }

    #[test]
    fn test_clipping() {
        let mut stage = GainStage::new();
        stage.update_meters(1.0, -3.0);

        assert!(stage.clipping);
    }

    #[test]
    fn test_peak_hold() {
        let mut stage = GainStage::new();
        stage.update_meters(-10.0, -15.0);
        stage.update_meters(-20.0, -25.0);

        assert!((stage.peak_hold_db - (-10.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_multi_channel_gain_stage() {
        let mut multi = MultiChannelGainStage::new(2);
        assert_eq!(multi.channel_count(), 2);

        multi
            .set_channel_gain(0, -6.0)
            .expect("should succeed in test");
        assert!((multi.channels[0].gain_db - (-6.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_linked_channels() {
        let mut multi = MultiChannelGainStage::new(2);
        multi.linked = true;

        multi
            .set_channel_gain(0, -6.0)
            .expect("should succeed in test");

        // Both channels should have the same gain when linked
        assert!((multi.channels[0].gain_db - (-6.0)).abs() < f32::EPSILON);
        assert!((multi.channels[1].gain_db - (-6.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_master_gain() {
        let mut multi = MultiChannelGainStage::new(2);
        multi
            .set_channel_gain(0, -6.0)
            .expect("should succeed in test");
        multi.set_master_gain(-3.0);

        let total = multi.total_gain_db(0).expect("should succeed in test");
        assert!((total - (-9.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_has_signal() {
        let mut stage = GainStage::new();
        assert!(!stage.has_signal());

        stage.update_meters(-20.0, -30.0);
        assert!(stage.has_signal());
    }
}
