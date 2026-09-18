//! Automation playback engine that reads `AutomationLane` data and applies
//! parameter changes (gain, pan, send level) sample-accurately during mixing.
//!
//! The [`AutomationPlayer`] holds a set of automation lanes keyed by
//! [`AutomatedParam`] and, for each processing block, renders per-sample
//! parameter values that the mixer can apply directly.

use std::collections::HashMap;

use crate::automation::{AutomationLane, AutomationMode, AutomationParameter};
use crate::channel::ChannelId;

// ---------------------------------------------------------------------------
// AutomatedParam — which parameter is automated
// ---------------------------------------------------------------------------

/// Identifies a single automatable parameter for the player.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AutomatedParam {
    /// Channel fader gain.
    Gain(ChannelId),
    /// Channel pan position.
    Pan(ChannelId),
    /// Channel send level (by send slot index).
    SendLevel(ChannelId, usize),
}

// ---------------------------------------------------------------------------
// AutomationPlayer
// ---------------------------------------------------------------------------

/// Reads automation lanes and produces per-sample parameter arrays for
/// sample-accurate playback.
///
/// Usage:
/// 1. Register lanes via `add_lane`.
/// 2. Each buffer call `render_block` with the current sample position.
/// 3. Query rendered values via `gain_at`, `pan_at`, `send_level_at`.
#[derive(Debug)]
pub struct AutomationPlayer {
    /// Lanes keyed by the parameter they control.
    lanes: HashMap<AutomatedParam, AutomationLane>,
    /// Rendered per-sample values for the current block.
    rendered: HashMap<AutomatedParam, Vec<f32>>,
    /// Whether automation playback is globally enabled.
    pub enabled: bool,
}

impl AutomationPlayer {
    /// Create a new, empty automation player.
    #[must_use]
    pub fn new() -> Self {
        Self {
            lanes: HashMap::new(),
            rendered: HashMap::new(),
            enabled: true,
        }
    }

    /// Register an automation lane for a parameter.
    ///
    /// If a lane already exists for this parameter, it is replaced.
    pub fn add_lane(&mut self, param: AutomatedParam, lane: AutomationLane) {
        self.lanes.insert(param, lane);
    }

    /// Remove an automation lane.
    pub fn remove_lane(&mut self, param: &AutomatedParam) {
        self.lanes.remove(param);
        self.rendered.remove(param);
    }

    /// Check if a lane exists for the given parameter.
    #[must_use]
    pub fn has_lane(&self, param: &AutomatedParam) -> bool {
        self.lanes.contains_key(param)
    }

    /// Get an immutable reference to a lane.
    #[must_use]
    pub fn get_lane(&self, param: &AutomatedParam) -> Option<&AutomationLane> {
        self.lanes.get(param)
    }

    /// Get a mutable reference to a lane.
    #[must_use]
    pub fn get_lane_mut(&mut self, param: &AutomatedParam) -> Option<&mut AutomationLane> {
        self.lanes.get_mut(param)
    }

    /// Render automation values for a processing block.
    ///
    /// `start_sample` is the absolute sample position of the first sample in
    /// this block.  `block_size` is the number of samples.
    ///
    /// After calling this method, use `gain_at`, `pan_at`, or
    /// `send_level_at` to retrieve per-sample values.
    pub fn render_block(&mut self, start_sample: u64, block_size: usize) {
        if !self.enabled {
            self.rendered.clear();
            return;
        }

        for (param, lane) in &self.lanes {
            // Only render if lane is in Read mode
            if lane.mode != AutomationMode::Read || !lane.enabled {
                continue;
            }

            let mut values = Vec::with_capacity(block_size);
            for i in 0..block_size {
                #[allow(clippy::cast_possible_truncation)]
                let sample_pos = start_sample.saturating_add(i as u64);
                values.push(lane.get_value_at(sample_pos));
            }
            self.rendered.insert(param.clone(), values);
        }
    }

    /// Get the rendered gain value for a channel at sample offset within the
    /// current block.
    ///
    /// Returns `None` if no automation is active for this parameter, in which
    /// case the caller should use the channel's static gain value.
    #[must_use]
    pub fn gain_at(&self, channel_id: ChannelId, sample_offset: usize) -> Option<f32> {
        let param = AutomatedParam::Gain(channel_id);
        self.rendered
            .get(&param)
            .and_then(|values| values.get(sample_offset).copied())
    }

    /// Get the rendered pan value for a channel at sample offset.
    #[must_use]
    pub fn pan_at(&self, channel_id: ChannelId, sample_offset: usize) -> Option<f32> {
        let param = AutomatedParam::Pan(channel_id);
        self.rendered
            .get(&param)
            .and_then(|values| values.get(sample_offset).copied())
    }

    /// Get the rendered send level for a channel/send at sample offset.
    #[must_use]
    pub fn send_level_at(
        &self,
        channel_id: ChannelId,
        send_slot: usize,
        sample_offset: usize,
    ) -> Option<f32> {
        let param = AutomatedParam::SendLevel(channel_id, send_slot);
        self.rendered
            .get(&param)
            .and_then(|values| values.get(sample_offset).copied())
    }

    /// Number of registered lanes.
    #[must_use]
    pub fn lane_count(&self) -> usize {
        self.lanes.len()
    }

    /// Clear all lanes and rendered data.
    pub fn clear(&mut self) {
        self.lanes.clear();
        self.rendered.clear();
    }

    /// Convert an [`AutomatedParam`] to the corresponding [`AutomationParameter`].
    #[must_use]
    pub fn to_automation_parameter(param: &AutomatedParam) -> AutomationParameter {
        match param {
            AutomatedParam::Gain(ch) => AutomationParameter::ChannelGain(*ch),
            AutomatedParam::Pan(ch) => AutomationParameter::ChannelPan(*ch),
            AutomatedParam::SendLevel(ch, slot) => AutomationParameter::ChannelSend {
                channel: *ch,
                send: *slot,
            },
        }
    }
}

impl Default for AutomationPlayer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automation::{AutomationLane, AutomationMode, AutomationParameter, AutomationPoint};
    use uuid::Uuid;

    fn make_ch() -> ChannelId {
        ChannelId(Uuid::new_v4())
    }

    #[test]
    fn test_player_new_empty() {
        let player = AutomationPlayer::new();
        assert_eq!(player.lane_count(), 0);
        assert!(player.enabled);
    }

    #[test]
    fn test_add_and_remove_lane() {
        let mut player = AutomationPlayer::new();
        let ch = make_ch();
        let param = AutomatedParam::Gain(ch);
        let lane = AutomationLane::new(AutomationParameter::ChannelGain(ch), 1.0);
        player.add_lane(param.clone(), lane);
        assert_eq!(player.lane_count(), 1);
        assert!(player.has_lane(&param));
        player.remove_lane(&param);
        assert_eq!(player.lane_count(), 0);
    }

    #[test]
    fn test_render_block_gain() {
        let mut player = AutomationPlayer::new();
        let ch = make_ch();
        let param = AutomatedParam::Gain(ch);
        let mut lane = AutomationLane::new(AutomationParameter::ChannelGain(ch), 1.0);
        lane.mode = AutomationMode::Read;
        lane.add_point(AutomationPoint::new(0, 0.0));
        lane.add_point(AutomationPoint::new(100, 1.0));
        player.add_lane(param, lane);

        player.render_block(0, 101);

        // At sample 0, gain should be ~0.0; at sample 100, ~1.0; at 50, ~0.5
        let g0 = player.gain_at(ch, 0);
        let g50 = player.gain_at(ch, 50);
        let g100 = player.gain_at(ch, 100);
        assert!(g0.is_some());
        assert!((g0.unwrap_or(0.0) - 0.0).abs() < 0.01);
        assert!((g50.unwrap_or(0.0) - 0.5).abs() < 0.02);
        assert!((g100.unwrap_or(0.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_render_block_pan() {
        let mut player = AutomationPlayer::new();
        let ch = make_ch();
        let param = AutomatedParam::Pan(ch);
        let mut lane = AutomationLane::new(AutomationParameter::ChannelPan(ch), 0.0);
        lane.mode = AutomationMode::Read;
        lane.add_point(AutomationPoint::new(0, -1.0));
        lane.add_point(AutomationPoint::new(100, 1.0));
        player.add_lane(param, lane);

        player.render_block(0, 101);

        let p0 = player.pan_at(ch, 0);
        let p100 = player.pan_at(ch, 100);
        assert!(p0.is_some());
        assert!((p0.unwrap_or(0.0) - (-1.0)).abs() < 0.01);
        assert!((p100.unwrap_or(0.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_render_block_send_level() {
        let mut player = AutomationPlayer::new();
        let ch = make_ch();
        let param = AutomatedParam::SendLevel(ch, 0);
        let mut lane = AutomationLane::new(
            AutomationParameter::ChannelSend {
                channel: ch,
                send: 0,
            },
            0.5,
        );
        lane.mode = AutomationMode::Read;
        lane.add_point(AutomationPoint::new(0, 0.0));
        lane.add_point(AutomationPoint::new(50, 1.0));
        player.add_lane(param, lane);

        player.render_block(0, 51);

        let s25 = player.send_level_at(ch, 0, 25);
        assert!(s25.is_some());
        assert!((s25.unwrap_or(0.0) - 0.5).abs() < 0.02);
    }

    #[test]
    fn test_disabled_player_no_render() {
        let mut player = AutomationPlayer::new();
        player.enabled = false;
        let ch = make_ch();
        let param = AutomatedParam::Gain(ch);
        let mut lane = AutomationLane::new(AutomationParameter::ChannelGain(ch), 1.0);
        lane.mode = AutomationMode::Read;
        lane.add_point(AutomationPoint::new(0, 0.5));
        player.add_lane(param, lane);

        player.render_block(0, 64);

        // Should return None since player is disabled
        assert!(player.gain_at(ch, 0).is_none());
    }

    #[test]
    fn test_non_read_mode_not_rendered() {
        let mut player = AutomationPlayer::new();
        let ch = make_ch();
        let param = AutomatedParam::Gain(ch);
        let mut lane = AutomationLane::new(AutomationParameter::ChannelGain(ch), 1.0);
        lane.mode = AutomationMode::Write; // Not Read
        lane.add_point(AutomationPoint::new(0, 0.5));
        player.add_lane(param, lane);

        player.render_block(0, 64);

        assert!(player.gain_at(ch, 0).is_none());
    }

    #[test]
    fn test_clear() {
        let mut player = AutomationPlayer::new();
        let ch = make_ch();
        let param = AutomatedParam::Gain(ch);
        let lane = AutomationLane::new(AutomationParameter::ChannelGain(ch), 1.0);
        player.add_lane(param, lane);
        assert_eq!(player.lane_count(), 1);
        player.clear();
        assert_eq!(player.lane_count(), 0);
    }

    #[test]
    fn test_out_of_range_offset_returns_none() {
        let mut player = AutomationPlayer::new();
        let ch = make_ch();
        let param = AutomatedParam::Gain(ch);
        let mut lane = AutomationLane::new(AutomationParameter::ChannelGain(ch), 1.0);
        lane.mode = AutomationMode::Read;
        lane.add_point(AutomationPoint::new(0, 0.5));
        player.add_lane(param, lane);

        player.render_block(0, 10);

        // Offset 10 is out of range (0..10)
        assert!(player.gain_at(ch, 10).is_none());
    }

    #[test]
    fn test_to_automation_parameter() {
        let ch = make_ch();
        let p1 = AutomatedParam::Gain(ch);
        let ap1 = AutomationPlayer::to_automation_parameter(&p1);
        assert_eq!(ap1, AutomationParameter::ChannelGain(ch));

        let p2 = AutomatedParam::Pan(ch);
        let ap2 = AutomationPlayer::to_automation_parameter(&p2);
        assert_eq!(ap2, AutomationParameter::ChannelPan(ch));

        let p3 = AutomatedParam::SendLevel(ch, 2);
        let ap3 = AutomationPlayer::to_automation_parameter(&p3);
        assert_eq!(
            ap3,
            AutomationParameter::ChannelSend {
                channel: ch,
                send: 2,
            }
        );
    }

    #[test]
    fn test_render_at_offset_start() {
        let mut player = AutomationPlayer::new();
        let ch = make_ch();
        let param = AutomatedParam::Gain(ch);
        let mut lane = AutomationLane::new(AutomationParameter::ChannelGain(ch), 1.0);
        lane.mode = AutomationMode::Read;
        lane.add_point(AutomationPoint::new(100, 0.0));
        lane.add_point(AutomationPoint::new(200, 1.0));
        player.add_lane(param, lane);

        // Render starting at sample 100
        player.render_block(100, 101);

        let g0 = player.gain_at(ch, 0); // corresponds to sample 100
        let g100 = player.gain_at(ch, 100); // corresponds to sample 200
        assert!((g0.unwrap_or(0.0) - 0.0).abs() < 0.01);
        assert!((g100.unwrap_or(0.0) - 1.0).abs() < 0.01);
    }
}
