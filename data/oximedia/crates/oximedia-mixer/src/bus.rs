//! Bus architecture for mixer routing.
//!
//! Provides master, group, auxiliary, and matrix buses for flexible audio routing.

use oximedia_audio::ChannelLayout;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::effects::EffectSlot;
use crate::ChannelId;

/// Unique bus identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BusId(pub Uuid);

impl std::fmt::Display for BusId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Bus type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BusType {
    /// Master output bus.
    Master,
    /// Group/submix bus.
    Group,
    /// Auxiliary send/return bus (for effects).
    Auxiliary,
    /// Matrix bus (for routing flexibility).
    Matrix,
}

/// Bus configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusConfig {
    /// Bus name.
    pub name: String,
    /// Bus type.
    pub bus_type: BusType,
    /// Channel layout.
    #[serde(skip)]
    pub layout: ChannelLayout,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Buffer size in samples.
    pub buffer_size: usize,
    /// Enable effects chain.
    pub enable_effects: bool,
    /// Enable automation.
    pub enable_automation: bool,
}

impl BusConfig {
    /// Create master bus config.
    #[must_use]
    pub fn master(sample_rate: u32, buffer_size: usize) -> Self {
        Self {
            name: "Master".to_string(),
            bus_type: BusType::Master,
            layout: ChannelLayout::Stereo,
            sample_rate,
            buffer_size,
            enable_effects: true,
            enable_automation: true,
        }
    }

    /// Create group bus config.
    #[must_use]
    pub fn group(
        name: String,
        layout: ChannelLayout,
        sample_rate: u32,
        buffer_size: usize,
    ) -> Self {
        Self {
            name,
            bus_type: BusType::Group,
            layout,
            sample_rate,
            buffer_size,
            enable_effects: true,
            enable_automation: true,
        }
    }

    /// Create auxiliary bus config.
    #[must_use]
    pub fn auxiliary(name: String, sample_rate: u32, buffer_size: usize) -> Self {
        Self {
            name,
            bus_type: BusType::Auxiliary,
            layout: ChannelLayout::Stereo,
            sample_rate,
            buffer_size,
            enable_effects: true,
            enable_automation: false,
        }
    }
}

/// Bus state flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct BusState {
    /// Bus is muted.
    pub muted: bool,
    /// Bus is soloed.
    pub soloed: bool,
    /// Bus is selected in UI.
    pub selected: bool,
    /// Bus is processing (active).
    pub active: bool,
}

impl Default for BusState {
    fn default() -> Self {
        Self {
            muted: false,
            soloed: false,
            selected: false,
            active: true,
        }
    }
}

/// Bus routing configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BusRouting {
    /// Input sources (channel IDs or bus IDs).
    pub inputs: Vec<BusInput>,
    /// Output destination (bus ID).
    pub output: Option<BusId>,
    /// Send to other buses.
    pub sends: HashMap<BusId, BusSend>,
}

/// Bus input source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BusInput {
    /// Input from channel.
    Channel(ChannelId),
    /// Input from another bus.
    Bus(BusId),
}

/// Bus send configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusSend {
    /// Target bus ID.
    pub target: BusId,
    /// Send level (0.0 = -inf, 1.0 = 0 dB).
    pub level: f32,
    /// Pre-fader send.
    pub pre_fader: bool,
    /// Send enabled.
    pub enabled: bool,
}

impl BusSend {
    /// Create new bus send.
    #[must_use]
    pub fn new(target: BusId) -> Self {
        Self {
            target,
            level: 1.0,
            pre_fader: false,
            enabled: true,
        }
    }
}

/// Bus pan mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BusPanMode {
    /// Stereo balance.
    Stereo,
    /// LCR (Left-Center-Right).
    Lcr,
    /// Dual mono.
    DualMono,
}

/// Bus monitoring mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MonitoringMode {
    /// Normal stereo monitoring.
    Stereo,
    /// Mono sum (L+R).
    Mono,
    /// Left channel only.
    Left,
    /// Right channel only.
    Right,
    /// Stereo difference (L-R).
    Difference,
    /// Mid/Side sum (M).
    Mid,
    /// Mid/Side difference (S).
    Side,
}

/// Mix bus (master, group, aux, matrix).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bus {
    /// Bus name.
    name: String,

    /// Bus type.
    #[allow(clippy::struct_field_names)]
    bus_type: BusType,

    /// Channel layout.
    #[serde(skip)]
    layout: ChannelLayout,

    /// Sample rate in Hz.
    sample_rate: u32,

    /// Buffer size in samples.
    buffer_size: usize,

    /// Bus state.
    state: BusState,

    /// Fader gain (0.0 = -inf dB, 1.0 = 0 dB).
    gain: f32,

    /// Pan position (-1.0 = left, 0.0 = center, 1.0 = right).
    pan: f32,

    /// Pan mode.
    pan_mode: BusPanMode,

    /// Width (for stereo).
    width: f32,

    /// Effect slots (insert effects).
    effects: Vec<EffectSlot>,

    /// Bus routing.
    routing: BusRouting,

    /// Monitoring mode.
    monitoring_mode: MonitoringMode,

    /// Bus color (RGB hex).
    color: Option<String>,

    /// Bus icon/symbol.
    icon: Option<String>,
}

impl Bus {
    /// Create a new bus.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: String,
        bus_type: BusType,
        layout: ChannelLayout,
        sample_rate: u32,
        buffer_size: usize,
    ) -> Self {
        Self {
            name,
            bus_type,
            layout,
            sample_rate,
            buffer_size,
            state: BusState::default(),
            gain: 1.0,
            pan: 0.0,
            pan_mode: BusPanMode::Stereo,
            width: 1.0,
            effects: Vec::new(),
            routing: BusRouting::default(),
            monitoring_mode: MonitoringMode::Stereo,
            color: None,
            icon: None,
        }
    }

    /// Create from config.
    #[must_use]
    pub fn from_config(config: BusConfig) -> Self {
        Self::new(
            config.name,
            config.bus_type,
            config.layout,
            config.sample_rate,
            config.buffer_size,
        )
    }

    /// Get bus name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Set bus name.
    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    /// Get bus type.
    #[must_use]
    pub fn bus_type(&self) -> BusType {
        self.bus_type
    }

    /// Get channel layout.
    #[must_use]
    pub fn layout(&self) -> &ChannelLayout {
        &self.layout
    }

    /// Get sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get buffer size.
    #[must_use]
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Get bus state.
    #[must_use]
    pub fn state(&self) -> &BusState {
        &self.state
    }

    /// Get mutable bus state.
    #[must_use]
    pub fn state_mut(&mut self) -> &mut BusState {
        &mut self.state
    }

    /// Check if bus is muted.
    #[must_use]
    pub fn is_muted(&self) -> bool {
        self.state.muted
    }

    /// Set mute state.
    pub fn set_muted(&mut self, muted: bool) {
        self.state.muted = muted;
    }

    /// Check if bus is soloed.
    #[must_use]
    pub fn is_soloed(&self) -> bool {
        self.state.soloed
    }

    /// Set solo state.
    pub fn set_soloed(&mut self, soloed: bool) {
        self.state.soloed = soloed;
    }

    /// Get bus gain.
    #[must_use]
    pub fn gain(&self) -> f32 {
        self.gain
    }

    /// Set bus gain (0.0 = -inf dB, 1.0 = 0 dB).
    pub fn set_gain(&mut self, gain: f32) {
        self.gain = gain.clamp(0.0, 2.0); // Allow up to +6dB
    }

    /// Get gain in dB.
    #[must_use]
    pub fn gain_db(&self) -> f32 {
        if self.gain <= 0.0 {
            -f32::INFINITY
        } else {
            20.0 * self.gain.log10()
        }
    }

    /// Set gain in dB.
    pub fn set_gain_db(&mut self, db: f32) {
        if db <= -80.0 {
            self.gain = 0.0;
        } else {
            self.gain = 10.0_f32.powf(db / 20.0).clamp(0.0, 2.0);
        }
    }

    /// Get pan position.
    #[must_use]
    pub fn pan(&self) -> f32 {
        self.pan
    }

    /// Set pan position (-1.0 = left, 0.0 = center, 1.0 = right).
    pub fn set_pan(&mut self, pan: f32) {
        self.pan = pan.clamp(-1.0, 1.0);
    }

    /// Get pan mode.
    #[must_use]
    pub fn pan_mode(&self) -> BusPanMode {
        self.pan_mode
    }

    /// Set pan mode.
    pub fn set_pan_mode(&mut self, mode: BusPanMode) {
        self.pan_mode = mode;
    }

    /// Get stereo width.
    #[must_use]
    pub fn width(&self) -> f32 {
        self.width
    }

    /// Set stereo width (0.0 = mono, 1.0 = normal stereo, >1.0 = wide).
    pub fn set_width(&mut self, width: f32) {
        self.width = width.clamp(0.0, 2.0);
    }

    /// Get effect slots.
    #[must_use]
    pub fn effects(&self) -> &[EffectSlot] {
        &self.effects
    }

    /// Get mutable effect slots.
    #[must_use]
    pub fn effects_mut(&mut self) -> &mut Vec<EffectSlot> {
        &mut self.effects
    }

    /// Add effect slot.
    pub fn add_effect(&mut self, slot: EffectSlot) {
        self.effects.push(slot);
    }

    /// Remove effect at index.
    pub fn remove_effect(&mut self, index: usize) -> Option<EffectSlot> {
        if index < self.effects.len() {
            Some(self.effects.remove(index))
        } else {
            None
        }
    }

    /// Get bus routing.
    #[must_use]
    pub fn routing(&self) -> &BusRouting {
        &self.routing
    }

    /// Get mutable bus routing.
    #[must_use]
    pub fn routing_mut(&mut self) -> &mut BusRouting {
        &mut self.routing
    }

    /// Add input source.
    pub fn add_input(&mut self, input: BusInput) {
        if !self.routing.inputs.contains(&input) {
            self.routing.inputs.push(input);
        }
    }

    /// Remove input source.
    pub fn remove_input(&mut self, input: BusInput) {
        self.routing.inputs.retain(|&i| i != input);
    }

    /// Set output destination.
    pub fn set_output(&mut self, output: Option<BusId>) {
        self.routing.output = output;
    }

    /// Add send to another bus.
    pub fn add_send(&mut self, target: BusId, send: BusSend) {
        self.routing.sends.insert(target, send);
    }

    /// Remove send.
    pub fn remove_send(&mut self, target: BusId) {
        self.routing.sends.remove(&target);
    }

    /// Get send to bus.
    #[must_use]
    pub fn get_send(&self, target: BusId) -> Option<&BusSend> {
        self.routing.sends.get(&target)
    }

    /// Get monitoring mode.
    #[must_use]
    pub fn monitoring_mode(&self) -> MonitoringMode {
        self.monitoring_mode
    }

    /// Set monitoring mode.
    pub fn set_monitoring_mode(&mut self, mode: MonitoringMode) {
        self.monitoring_mode = mode;
    }

    /// Get bus color.
    #[must_use]
    pub fn color(&self) -> Option<&str> {
        self.color.as_deref()
    }

    /// Set bus color (RGB hex string).
    pub fn set_color(&mut self, color: Option<String>) {
        self.color = color;
    }

    /// Get bus icon.
    #[must_use]
    pub fn icon(&self) -> Option<&str> {
        self.icon.as_deref()
    }

    /// Set bus icon.
    pub fn set_icon(&mut self, icon: Option<String>) {
        self.icon = icon;
    }

    /// Process audio through bus.
    ///
    /// This would:
    /// 1. Sum inputs from all sources
    /// 2. Apply insert effects
    /// 3. Apply bus gain and pan
    /// 4. Send to output bus
    /// 5. Process sends to other buses
    #[allow(dead_code, clippy::unused_self)]
    fn process(&mut self, _input: &[f32], _output: &mut [f32]) {
        // Skeleton implementation
        // Full implementation would:
        // 1. Sum all input sources
        // 2. Process through effect chain
        // 3. Apply bus gain
        // 4. Apply pan/width
        // 5. Send to output bus
        // 6. Process all sends
    }
}

/// Matrix mixer for advanced routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixMixer {
    /// Input buses.
    pub inputs: Vec<BusId>,
    /// Output buses.
    pub outputs: Vec<BusId>,
    /// Crosspoint gains (`input_index` -> `output_index` -> gain).
    pub crosspoints: HashMap<(usize, usize), f32>,
}

impl MatrixMixer {
    /// Create new matrix mixer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
            crosspoints: HashMap::new(),
        }
    }

    /// Add input bus.
    pub fn add_input(&mut self, bus_id: BusId) {
        if !self.inputs.contains(&bus_id) {
            self.inputs.push(bus_id);
        }
    }

    /// Add output bus.
    pub fn add_output(&mut self, bus_id: BusId) {
        if !self.outputs.contains(&bus_id) {
            self.outputs.push(bus_id);
        }
    }

    /// Set crosspoint gain.
    pub fn set_crosspoint(&mut self, input_index: usize, output_index: usize, gain: f32) {
        if input_index < self.inputs.len() && output_index < self.outputs.len() {
            self.crosspoints.insert((input_index, output_index), gain);
        }
    }

    /// Get crosspoint gain.
    #[must_use]
    pub fn get_crosspoint(&self, input_index: usize, output_index: usize) -> f32 {
        self.crosspoints
            .get(&(input_index, output_index))
            .copied()
            .unwrap_or(0.0)
    }

    /// Clear all crosspoints.
    pub fn clear(&mut self) {
        self.crosspoints.clear();
    }
}

impl Default for MatrixMixer {
    fn default() -> Self {
        Self::new()
    }
}

/// Bus group for organizing buses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusGroup {
    /// Group ID.
    pub id: Uuid,
    /// Group name.
    pub name: String,
    /// Member buses.
    pub buses: Vec<BusId>,
    /// Group color.
    pub color: Option<String>,
}

impl BusGroup {
    /// Create new bus group.
    #[must_use]
    pub fn new(name: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            buses: Vec::new(),
            color: None,
        }
    }

    /// Add bus to group.
    pub fn add_bus(&mut self, bus_id: BusId) {
        if !self.buses.contains(&bus_id) {
            self.buses.push(bus_id);
        }
    }

    /// Remove bus from group.
    pub fn remove_bus(&mut self, bus_id: BusId) {
        self.buses.retain(|&id| id != bus_id);
    }

    /// Check if group contains bus.
    #[must_use]
    pub fn contains(&self, bus_id: BusId) -> bool {
        self.buses.contains(&bus_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bus_creation() {
        let bus = Bus::new(
            "Test Bus".to_string(),
            BusType::Group,
            ChannelLayout::Stereo,
            48000,
            512,
        );

        assert_eq!(bus.name(), "Test Bus");
        assert_eq!(bus.bus_type(), BusType::Group);
        assert_eq!(bus.gain(), 1.0);
    }

    #[test]
    fn test_bus_config() {
        let config = BusConfig::master(48000, 512);
        assert_eq!(config.bus_type, BusType::Master);
        assert_eq!(config.sample_rate, 48000);

        let bus = Bus::from_config(config);
        assert_eq!(bus.bus_type(), BusType::Master);
    }

    #[test]
    fn test_bus_gain() {
        let mut bus = Bus::new(
            "Test".to_string(),
            BusType::Group,
            ChannelLayout::Stereo,
            48000,
            512,
        );

        bus.set_gain(0.5);
        assert!((bus.gain() - 0.5).abs() < f32::EPSILON);

        bus.set_gain_db(-6.0);
        assert!((bus.gain() - 0.501_187_2).abs() < 0.001);
    }

    #[test]
    fn test_bus_state() {
        let mut bus = Bus::new(
            "Test".to_string(),
            BusType::Group,
            ChannelLayout::Stereo,
            48000,
            512,
        );

        assert!(!bus.is_muted());
        bus.set_muted(true);
        assert!(bus.is_muted());

        assert!(!bus.is_soloed());
        bus.set_soloed(true);
        assert!(bus.is_soloed());
    }

    #[test]
    fn test_bus_routing() {
        let mut bus = Bus::new(
            "Test".to_string(),
            BusType::Group,
            ChannelLayout::Stereo,
            48000,
            512,
        );

        let channel_id = ChannelId(Uuid::new_v4());
        let input = BusInput::Channel(channel_id);

        bus.add_input(input);
        assert_eq!(bus.routing().inputs.len(), 1);

        bus.remove_input(input);
        assert_eq!(bus.routing().inputs.len(), 0);
    }

    #[test]
    fn test_bus_send() {
        let mut bus = Bus::new(
            "Test".to_string(),
            BusType::Group,
            ChannelLayout::Stereo,
            48000,
            512,
        );

        let target = BusId(Uuid::new_v4());
        let send = BusSend::new(target);

        bus.add_send(target, send);
        assert!(bus.get_send(target).is_some());

        bus.remove_send(target);
        assert!(bus.get_send(target).is_none());
    }

    #[test]
    fn test_matrix_mixer() {
        let mut matrix = MatrixMixer::new();

        let input1 = BusId(Uuid::new_v4());
        let output1 = BusId(Uuid::new_v4());

        matrix.add_input(input1);
        matrix.add_output(output1);

        matrix.set_crosspoint(0, 0, 0.8);
        assert!((matrix.get_crosspoint(0, 0) - 0.8).abs() < f32::EPSILON);

        matrix.clear();
        assert_eq!(matrix.get_crosspoint(0, 0), 0.0);
    }

    #[test]
    fn test_bus_group() {
        let mut group = BusGroup::new("Group 1".to_string());
        let bus_id = BusId(Uuid::new_v4());

        assert!(!group.contains(bus_id));
        group.add_bus(bus_id);
        assert!(group.contains(bus_id));
        group.remove_bus(bus_id);
        assert!(!group.contains(bus_id));
    }

    #[test]
    fn test_bus_width() {
        let mut bus = Bus::new(
            "Test".to_string(),
            BusType::Group,
            ChannelLayout::Stereo,
            48000,
            512,
        );

        assert_eq!(bus.width(), 1.0);
        bus.set_width(0.5);
        assert_eq!(bus.width(), 0.5);
    }

    #[test]
    fn test_monitoring_modes() {
        let mut bus = Bus::new(
            "Test".to_string(),
            BusType::Group,
            ChannelLayout::Stereo,
            48000,
            512,
        );

        assert_eq!(bus.monitoring_mode(), MonitoringMode::Stereo);
        bus.set_monitoring_mode(MonitoringMode::Mono);
        assert_eq!(bus.monitoring_mode(), MonitoringMode::Mono);
    }
}
