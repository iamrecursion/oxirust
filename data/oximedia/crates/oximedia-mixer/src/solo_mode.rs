//! Solo-in-Place and AFL/PFL modes for the mixer processing pipeline.
//!
//! This module integrates with the existing `SoloBus` and extends the mixer
//! processing to support three standard solo modes:
//!
//! - **SIP (Solo-In-Place)**: Mutes all non-soloed channels in the main mix.
//! - **AFL (After-Fader Listen)**: Routes post-fader signal of soloed channels
//!   to the monitor bus without affecting the main mix.
//! - **PFL (Pre-Fader Listen)**: Routes pre-fader signal of soloed channels
//!   to the monitor bus without affecting the main mix.

use std::collections::HashSet;

use crate::channel::ChannelId;

// ---------------------------------------------------------------------------
// SoloMode enum
// ---------------------------------------------------------------------------

/// Solo mode for the mixer pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoloMode {
    /// Solo-In-Place: non-soloed channels are muted (or dimmed).
    Sip,
    /// After-Fader Listen: soloed channels post-fader signal goes to monitor.
    Afl,
    /// Pre-Fader Listen: soloed channels pre-fader signal goes to monitor.
    Pfl,
}

// ---------------------------------------------------------------------------
// SoloProcessor
// ---------------------------------------------------------------------------

/// Manages solo state and computes per-channel gain adjustments based on the
/// active [`SoloMode`].
///
/// The processor tracks which channels are soloed and provides methods to
/// determine:
/// - The main mix gain for each channel (SIP mode may mute non-soloed).
/// - The monitor bus gain and signal source (AFL/PFL).
#[derive(Debug, Clone)]
pub struct SoloProcessor {
    /// Current solo mode.
    mode: SoloMode,
    /// Set of soloed channel IDs.
    soloed: HashSet<ChannelId>,
    /// Set of solo-safe channel IDs (not muted in SIP mode).
    solo_safe: HashSet<ChannelId>,
    /// Dim level for non-soloed channels in SIP mode (0.0 = full mute).
    sip_dim_level: f32,
    /// Monitor bus gain for AFL/PFL.
    monitor_gain: f32,
}

impl SoloProcessor {
    /// Create a new solo processor with the given mode.
    #[must_use]
    pub fn new(mode: SoloMode) -> Self {
        Self {
            mode,
            soloed: HashSet::new(),
            solo_safe: HashSet::new(),
            sip_dim_level: 0.0,
            monitor_gain: 1.0,
        }
    }

    /// Set the solo mode.
    pub fn set_mode(&mut self, mode: SoloMode) {
        self.mode = mode;
    }

    /// Get the current solo mode.
    #[must_use]
    pub fn mode(&self) -> SoloMode {
        self.mode
    }

    /// Solo a channel.
    pub fn solo(&mut self, channel_id: ChannelId) {
        self.soloed.insert(channel_id);
    }

    /// Unsolo a channel.
    pub fn unsolo(&mut self, channel_id: ChannelId) {
        self.soloed.remove(&channel_id);
    }

    /// Toggle solo state of a channel.
    pub fn toggle_solo(&mut self, channel_id: ChannelId) {
        if self.soloed.contains(&channel_id) {
            self.soloed.remove(&channel_id);
        } else {
            self.soloed.insert(channel_id);
        }
    }

    /// Clear all solos.
    pub fn clear_solos(&mut self) {
        self.soloed.clear();
    }

    /// Check if any channel is soloed.
    #[must_use]
    pub fn any_soloed(&self) -> bool {
        !self.soloed.is_empty()
    }

    /// Check if a specific channel is soloed.
    #[must_use]
    pub fn is_soloed(&self, channel_id: ChannelId) -> bool {
        self.soloed.contains(&channel_id)
    }

    /// Mark a channel as solo-safe (exempt from SIP muting).
    pub fn set_solo_safe(&mut self, channel_id: ChannelId, safe: bool) {
        if safe {
            self.solo_safe.insert(channel_id);
        } else {
            self.solo_safe.remove(&channel_id);
        }
    }

    /// Check if a channel is solo-safe.
    #[must_use]
    pub fn is_solo_safe(&self, channel_id: ChannelId) -> bool {
        self.solo_safe.contains(&channel_id)
    }

    /// Set the SIP dim level (0.0 = full mute, 1.0 = no attenuation).
    pub fn set_sip_dim_level(&mut self, level: f32) {
        self.sip_dim_level = level.clamp(0.0, 1.0);
    }

    /// Get the SIP dim level.
    #[must_use]
    pub fn sip_dim_level(&self) -> f32 {
        self.sip_dim_level
    }

    /// Set the monitor bus gain.
    pub fn set_monitor_gain(&mut self, gain: f32) {
        self.monitor_gain = gain.clamp(0.0, 2.0);
    }

    /// Get the monitor bus gain.
    #[must_use]
    pub fn monitor_gain(&self) -> f32 {
        self.monitor_gain
    }

    /// Compute the main mix gain multiplier for a channel.
    ///
    /// - If no channels are soloed: returns 1.0 for all channels.
    /// - SIP mode: soloed/safe channels get 1.0; others get `sip_dim_level`.
    /// - AFL/PFL mode: main mix is unaffected; returns 1.0 for all.
    #[must_use]
    pub fn main_mix_gain(&self, channel_id: ChannelId) -> f32 {
        if !self.any_soloed() {
            return 1.0;
        }

        match self.mode {
            SoloMode::Sip => {
                if self.soloed.contains(&channel_id) || self.solo_safe.contains(&channel_id) {
                    1.0
                } else {
                    self.sip_dim_level
                }
            }
            SoloMode::Afl | SoloMode::Pfl => {
                // AFL/PFL do not affect the main mix
                1.0
            }
        }
    }

    /// Compute the monitor bus gain for a channel.
    ///
    /// - SIP mode: monitor bus not used (returns 0.0).
    /// - AFL mode: soloed channels get `monitor_gain` (post-fader source).
    /// - PFL mode: soloed channels get `monitor_gain` (pre-fader source).
    ///
    /// The caller is responsible for using the correct signal source (pre/post
    /// fader) based on the mode.
    #[must_use]
    pub fn monitor_bus_gain(&self, channel_id: ChannelId) -> f32 {
        if !self.soloed.contains(&channel_id) {
            return 0.0;
        }

        match self.mode {
            SoloMode::Sip => 0.0, // SIP uses the main mix, no separate monitor
            SoloMode::Afl | SoloMode::Pfl => self.monitor_gain,
        }
    }

    /// Returns whether the monitor bus should use pre-fader signal.
    ///
    /// - PFL: `true` (pre-fader signal)
    /// - AFL: `false` (post-fader signal)
    /// - SIP: `false` (not applicable)
    #[must_use]
    pub fn monitor_uses_pre_fader(&self) -> bool {
        self.mode == SoloMode::Pfl
    }

    /// Process channel buffers through the solo system.
    ///
    /// Applies SIP muting to the main mix output and generates the monitor
    /// bus output for AFL/PFL.
    ///
    /// # Arguments
    /// * `channel_id` — The channel being processed.
    /// * `pre_fader` — Pre-fader signal buffer.
    /// * `post_fader_left` — Post-fader left channel buffer (modified in-place for SIP).
    /// * `post_fader_right` — Post-fader right channel buffer (modified in-place for SIP).
    ///
    /// # Returns
    /// `(monitor_left, monitor_right)` — Monitor bus contribution.
    #[must_use]
    pub fn process_channel(
        &self,
        channel_id: ChannelId,
        pre_fader: &[f32],
        post_fader_left: &[f32],
        post_fader_right: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let n = post_fader_left.len().min(post_fader_right.len());

        // Monitor bus output
        let monitor_gain = self.monitor_bus_gain(channel_id);
        let (mon_l, mon_r) = if monitor_gain.abs() < f32::EPSILON {
            (vec![0.0_f32; n], vec![0.0_f32; n])
        } else if self.monitor_uses_pre_fader() {
            // PFL: mono pre-fader to both channels
            let scaled: Vec<f32> = pre_fader
                .iter()
                .take(n)
                .map(|&s| s * monitor_gain)
                .collect();
            (scaled.clone(), scaled)
        } else {
            // AFL: post-fader stereo
            let l: Vec<f32> = post_fader_left
                .iter()
                .take(n)
                .map(|&s| s * monitor_gain)
                .collect();
            let r: Vec<f32> = post_fader_right
                .iter()
                .take(n)
                .map(|&s| s * monitor_gain)
                .collect();
            (l, r)
        };

        (mon_l, mon_r)
    }

    /// Apply SIP muting to a channel's main mix output in-place.
    pub fn apply_sip_muting(&self, channel_id: ChannelId, left: &mut [f32], right: &mut [f32]) {
        let gain = self.main_mix_gain(channel_id);
        if (gain - 1.0).abs() > f32::EPSILON {
            for s in left.iter_mut() {
                *s *= gain;
            }
            for s in right.iter_mut() {
                *s *= gain;
            }
        }
    }

    /// Get the number of currently soloed channels.
    #[must_use]
    pub fn soloed_count(&self) -> usize {
        self.soloed.len()
    }
}

impl Default for SoloProcessor {
    fn default() -> Self {
        Self::new(SoloMode::Sip)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_ch() -> ChannelId {
        ChannelId(Uuid::new_v4())
    }

    #[test]
    fn test_default_mode() {
        let sp = SoloProcessor::default();
        assert_eq!(sp.mode(), SoloMode::Sip);
        assert!(!sp.any_soloed());
    }

    #[test]
    fn test_solo_unsolo() {
        let mut sp = SoloProcessor::new(SoloMode::Sip);
        let ch = make_ch();
        sp.solo(ch);
        assert!(sp.is_soloed(ch));
        assert!(sp.any_soloed());
        assert_eq!(sp.soloed_count(), 1);
        sp.unsolo(ch);
        assert!(!sp.is_soloed(ch));
        assert!(!sp.any_soloed());
    }

    #[test]
    fn test_toggle_solo() {
        let mut sp = SoloProcessor::new(SoloMode::Sip);
        let ch = make_ch();
        sp.toggle_solo(ch);
        assert!(sp.is_soloed(ch));
        sp.toggle_solo(ch);
        assert!(!sp.is_soloed(ch));
    }

    #[test]
    fn test_clear_solos() {
        let mut sp = SoloProcessor::new(SoloMode::Sip);
        sp.solo(make_ch());
        sp.solo(make_ch());
        sp.clear_solos();
        assert!(!sp.any_soloed());
    }

    #[test]
    fn test_sip_mutes_non_soloed() {
        let mut sp = SoloProcessor::new(SoloMode::Sip);
        let ch_soloed = make_ch();
        let ch_not_soloed = make_ch();
        sp.solo(ch_soloed);

        assert!((sp.main_mix_gain(ch_soloed) - 1.0).abs() < f32::EPSILON);
        assert!((sp.main_mix_gain(ch_not_soloed) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sip_dim_level() {
        let mut sp = SoloProcessor::new(SoloMode::Sip);
        sp.set_sip_dim_level(0.2);
        let ch_soloed = make_ch();
        let ch_not = make_ch();
        sp.solo(ch_soloed);

        assert!((sp.main_mix_gain(ch_not) - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sip_solo_safe() {
        let mut sp = SoloProcessor::new(SoloMode::Sip);
        let ch_soloed = make_ch();
        let ch_safe = make_ch();
        let ch_normal = make_ch();
        sp.solo(ch_soloed);
        sp.set_solo_safe(ch_safe, true);

        assert!((sp.main_mix_gain(ch_soloed) - 1.0).abs() < f32::EPSILON);
        assert!((sp.main_mix_gain(ch_safe) - 1.0).abs() < f32::EPSILON);
        assert!((sp.main_mix_gain(ch_normal) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_afl_does_not_affect_main_mix() {
        let mut sp = SoloProcessor::new(SoloMode::Afl);
        let ch_soloed = make_ch();
        let ch_not = make_ch();
        sp.solo(ch_soloed);

        assert!((sp.main_mix_gain(ch_soloed) - 1.0).abs() < f32::EPSILON);
        assert!((sp.main_mix_gain(ch_not) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_pfl_does_not_affect_main_mix() {
        let mut sp = SoloProcessor::new(SoloMode::Pfl);
        let ch_soloed = make_ch();
        let ch_not = make_ch();
        sp.solo(ch_soloed);

        assert!((sp.main_mix_gain(ch_soloed) - 1.0).abs() < f32::EPSILON);
        assert!((sp.main_mix_gain(ch_not) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_afl_monitor_gain() {
        let mut sp = SoloProcessor::new(SoloMode::Afl);
        let ch = make_ch();
        sp.solo(ch);

        assert!((sp.monitor_bus_gain(ch) - 1.0).abs() < f32::EPSILON);
        assert!(!sp.monitor_uses_pre_fader());
    }

    #[test]
    fn test_pfl_monitor_gain() {
        let mut sp = SoloProcessor::new(SoloMode::Pfl);
        let ch = make_ch();
        sp.solo(ch);

        assert!((sp.monitor_bus_gain(ch) - 1.0).abs() < f32::EPSILON);
        assert!(sp.monitor_uses_pre_fader());
    }

    #[test]
    fn test_sip_no_monitor() {
        let mut sp = SoloProcessor::new(SoloMode::Sip);
        let ch = make_ch();
        sp.solo(ch);

        assert!((sp.monitor_bus_gain(ch) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_no_solo_all_unity() {
        let sp = SoloProcessor::new(SoloMode::Sip);
        let ch = make_ch();
        assert!((sp.main_mix_gain(ch) - 1.0).abs() < f32::EPSILON);
        assert!((sp.monitor_bus_gain(ch) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_process_channel_pfl() {
        let mut sp = SoloProcessor::new(SoloMode::Pfl);
        let ch = make_ch();
        sp.solo(ch);

        let pre_fader = vec![0.8_f32; 32];
        let post_l = vec![0.5_f32; 32];
        let post_r = vec![0.3_f32; 32];

        let (mon_l, mon_r) = sp.process_channel(ch, &pre_fader, &post_l, &post_r);

        // PFL: monitor should use pre-fader signal
        for (i, &m) in mon_l.iter().enumerate() {
            assert!(
                (m - 0.8).abs() < 0.01,
                "PFL monitor L[{i}] should be pre-fader value, got {m}"
            );
        }
        assert_eq!(mon_l, mon_r, "PFL monitor should be mono (L==R)");
    }

    #[test]
    fn test_process_channel_afl() {
        let mut sp = SoloProcessor::new(SoloMode::Afl);
        let ch = make_ch();
        sp.solo(ch);

        let pre_fader = vec![0.8_f32; 32];
        let post_l = vec![0.5_f32; 32];
        let post_r = vec![0.3_f32; 32];

        let (mon_l, mon_r) = sp.process_channel(ch, &pre_fader, &post_l, &post_r);

        // AFL: monitor should use post-fader signal
        for (i, &m) in mon_l.iter().enumerate() {
            assert!(
                (m - 0.5).abs() < 0.01,
                "AFL monitor L[{i}] should be post-fader left, got {m}"
            );
        }
        for (i, &m) in mon_r.iter().enumerate() {
            assert!(
                (m - 0.3).abs() < 0.01,
                "AFL monitor R[{i}] should be post-fader right, got {m}"
            );
        }
    }

    #[test]
    fn test_apply_sip_muting() {
        let mut sp = SoloProcessor::new(SoloMode::Sip);
        let ch_soloed = make_ch();
        let ch_muted = make_ch();
        sp.solo(ch_soloed);

        let mut left = vec![1.0_f32; 16];
        let mut right = vec![1.0_f32; 16];
        sp.apply_sip_muting(ch_muted, &mut left, &mut right);

        for &s in &left {
            assert!(s.abs() < f32::EPSILON, "non-soloed should be muted");
        }
    }

    #[test]
    fn test_non_soloed_no_monitor() {
        let mut sp = SoloProcessor::new(SoloMode::Pfl);
        let ch_soloed = make_ch();
        let ch_not = make_ch();
        sp.solo(ch_soloed);

        let pre = vec![1.0_f32; 16];
        let post_l = vec![0.5_f32; 16];
        let post_r = vec![0.5_f32; 16];

        let (mon_l, _) = sp.process_channel(ch_not, &pre, &post_l, &post_r);
        for &s in &mon_l {
            assert!(
                s.abs() < f32::EPSILON,
                "non-soloed should have no monitor output"
            );
        }
    }

    #[test]
    fn test_set_monitor_gain() {
        let mut sp = SoloProcessor::new(SoloMode::Afl);
        sp.set_monitor_gain(0.5);
        let ch = make_ch();
        sp.solo(ch);
        assert!((sp.monitor_bus_gain(ch) - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_solo_safe_toggle() {
        let mut sp = SoloProcessor::new(SoloMode::Sip);
        let ch = make_ch();
        sp.set_solo_safe(ch, true);
        assert!(sp.is_solo_safe(ch));
        sp.set_solo_safe(ch, false);
        assert!(!sp.is_solo_safe(ch));
    }
}
