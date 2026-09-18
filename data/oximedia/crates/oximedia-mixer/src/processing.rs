//! DSP processing pipeline for the audio mixer.
//!
//! Implements the full signal chain:
//!
//! ```text
//! Input → PDC Delay → Insert Effects → Fader (× VCA) → Pan → Sends → Bus → Master
//! ```
//!
//! # Bus Routing
//!
//! Channels can be routed to group or auxiliary buses rather than summing
//! directly to the master bus. Group buses then feed into the master bus
//! (or another group bus) according to the routing graph.
//!
//! # Effect Processing
//!
//! Each channel and bus can hold an `EffectsChain` containing one or more
//! `AudioEffect` instances. The chain processes the mono working buffer
//! in-place before fader and pan stages.
//!
//! # Send / Return
//!
//! Aux sends tap the channel signal (pre- or post-fader) and contribute to
//! aux bus accumulators. Each aux bus can hold its own effects chain (e.g.
//! reverb). The aux bus output is then mixed back into the master bus.
//!
//! # VCA Group Control
//!
//! VCA groups apply a multiplicative gain offset to every linked channel's
//! fader during processing, without altering the stored fader value.
//!
//! # Plugin Delay Compensation (PDC)
//!
//! Effects with lookahead (e.g. limiters) introduce latency. The PDC system
//! computes the maximum latency across all channels and inserts compensating
//! delay lines on the shorter-latency channels so that all signals arrive
//! at the summing point time-aligned.

use std::collections::HashMap;

use crate::bus::{BusId, BusType};
use crate::channel::ChannelId;
use crate::effects_chain::AudioEffect;

// ---------------------------------------------------------------------------
// Channel Effects Chain (runtime DSP)
// ---------------------------------------------------------------------------

/// Runtime effects chain for a channel or bus, holding boxed `AudioEffect`
/// instances that process audio in-place.
pub struct RuntimeEffectsChain {
    effects: Vec<RuntimeEffectSlot>,
}

/// A single slot in the runtime effects chain.
pub struct RuntimeEffectSlot {
    /// The effect processor.
    pub effect: Box<dyn AudioEffect>,
    /// Whether this slot is bypassed.
    pub bypassed: bool,
    /// Wet/dry mix (0.0 = fully dry, 1.0 = fully wet).
    pub mix: f32,
    /// Lookahead latency in samples (for PDC).
    pub latency_samples: usize,
}

impl RuntimeEffectSlot {
    /// Create a new runtime effect slot.
    #[must_use]
    pub fn new(effect: Box<dyn AudioEffect>) -> Self {
        Self {
            effect,
            bypassed: false,
            mix: 1.0,
            latency_samples: 0,
        }
    }

    /// Create a slot with a specified lookahead latency.
    #[must_use]
    pub fn with_latency(effect: Box<dyn AudioEffect>, latency_samples: usize) -> Self {
        Self {
            effect,
            bypassed: false,
            mix: 1.0,
            latency_samples,
        }
    }
}

impl RuntimeEffectsChain {
    /// Create an empty runtime effects chain.
    #[must_use]
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
        }
    }

    /// Append an effect slot.
    pub fn add(&mut self, slot: RuntimeEffectSlot) {
        self.effects.push(slot);
    }

    /// Remove effect at index. Returns `None` if out of range.
    pub fn remove(&mut self, idx: usize) -> Option<RuntimeEffectSlot> {
        if idx < self.effects.len() {
            Some(self.effects.remove(idx))
        } else {
            None
        }
    }

    /// Process the sample buffer through all non-bypassed effects in order.
    ///
    /// For each slot the dry signal is preserved and mixed with the wet
    /// output according to the slot's `mix` parameter.
    pub fn process(&mut self, samples: &mut [f32]) {
        for slot in &mut self.effects {
            if slot.bypassed {
                continue;
            }
            if (slot.mix - 1.0).abs() < f32::EPSILON {
                // Fully wet — process in-place directly.
                slot.effect.process(samples);
            } else if slot.mix.abs() < f32::EPSILON {
                // Fully dry — skip processing.
                continue;
            } else {
                // Partial mix — need a dry copy.
                let dry: Vec<f32> = samples.to_vec();
                slot.effect.process(samples);
                let wet_mix = slot.mix;
                let dry_mix = 1.0 - wet_mix;
                for (s, &d) in samples.iter_mut().zip(dry.iter()) {
                    *s = d * dry_mix + *s * wet_mix;
                }
            }
        }
    }

    /// Total lookahead latency (sum of all non-bypassed slots).
    #[must_use]
    pub fn total_latency(&self) -> usize {
        self.effects
            .iter()
            .filter(|s| !s.bypassed)
            .map(|s| s.latency_samples)
            .sum()
    }

    /// Number of slots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.effects.len()
    }

    /// Returns `true` if the chain is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }
}

impl Default for RuntimeEffectsChain {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Channel Routing Assignment
// ---------------------------------------------------------------------------

/// Where a channel's post-pan stereo output is routed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelOutputTarget {
    /// Route directly to master bus (default).
    Master,
    /// Route to a group bus.
    GroupBus(BusId),
}

impl Default for ChannelOutputTarget {
    fn default() -> Self {
        Self::Master
    }
}

/// Aux send descriptor used during processing.
#[derive(Debug, Clone)]
pub struct ProcessAuxSend {
    /// Target aux bus ID.
    pub bus_id: BusId,
    /// Send level (linear, 0.0–1.0).
    pub level: f32,
    /// Whether the send taps pre-fader.
    pub pre_fader: bool,
    /// Whether the send is active.
    pub active: bool,
}

// ---------------------------------------------------------------------------
// VCA runtime state
// ---------------------------------------------------------------------------

/// A VCA group that applies multiplicative gain to linked channels.
#[derive(Debug, Clone)]
pub struct VcaGroupState {
    /// Group name.
    pub name: String,
    /// VCA fader gain (linear, 0.0–2.0).
    pub gain: f32,
    /// Whether the VCA group is muted.
    pub muted: bool,
    /// Channel IDs belonging to this VCA group.
    pub channels: Vec<ChannelId>,
}

impl VcaGroupState {
    /// Create a new VCA group state.
    #[must_use]
    pub fn new(name: String) -> Self {
        Self {
            name,
            gain: 1.0,
            muted: false,
            channels: Vec::new(),
        }
    }

    /// Effective linear gain (0.0 if muted).
    #[must_use]
    pub fn effective_gain(&self) -> f32 {
        if self.muted {
            0.0
        } else {
            self.gain
        }
    }
}

// ---------------------------------------------------------------------------
// PDC (Plugin Delay Compensation)
// ---------------------------------------------------------------------------

/// Per-channel delay line for plugin delay compensation.
///
/// Uses a simple circular buffer with integer-sample delay, no interpolation
/// needed since PDC delays are always whole-sample values.
#[derive(Debug, Clone)]
pub struct PdcDelayLine {
    buffer_left: Vec<f32>,
    buffer_right: Vec<f32>,
    write_pos: usize,
    delay_samples: usize,
}

impl PdcDelayLine {
    /// Create a new stereo PDC delay line.
    ///
    /// `max_delay` is the maximum compensation in samples.
    #[must_use]
    pub fn new(max_delay: usize) -> Self {
        // +1 because a circular buffer of length N can delay up to N-1 samples,
        // so we need max_delay+1 slots to support exactly max_delay compensation.
        let len = max_delay + 1;
        Self {
            buffer_left: vec![0.0; len],
            buffer_right: vec![0.0; len],
            write_pos: 0,
            delay_samples: 0,
        }
    }

    /// Set the compensation delay in samples.
    pub fn set_delay(&mut self, samples: usize) {
        self.delay_samples = samples.min(self.buffer_left.len() - 1);
    }

    /// Current delay in samples.
    #[must_use]
    pub fn delay(&self) -> usize {
        self.delay_samples
    }

    /// Process one stereo sample pair through the delay line.
    ///
    /// Returns the delayed `(left, right)`.
    pub fn process_sample(&mut self, left: f32, right: f32) -> (f32, f32) {
        if self.delay_samples == 0 {
            return (left, right);
        }
        let len = self.buffer_left.len();
        let read_pos = (self.write_pos + len - self.delay_samples) % len;
        let out_l = self.buffer_left[read_pos];
        let out_r = self.buffer_right[read_pos];
        self.buffer_left[self.write_pos] = left;
        self.buffer_right[self.write_pos] = right;
        self.write_pos = (self.write_pos + 1) % len;
        (out_l, out_r)
    }

    /// Process a block of interleaved stereo samples.
    pub fn process_block(&mut self, left: &mut [f32], right: &mut [f32]) {
        for i in 0..left.len().min(right.len()) {
            let (ol, or) = self.process_sample(left[i], right[i]);
            left[i] = ol;
            right[i] = or;
        }
    }

    /// Clear the delay buffers.
    pub fn clear(&mut self) {
        self.buffer_left.fill(0.0);
        self.buffer_right.fill(0.0);
        self.write_pos = 0;
    }
}

// ---------------------------------------------------------------------------
// Stereo bus accumulator
// ---------------------------------------------------------------------------

/// Accumulator for a stereo bus during one processing block.
#[derive(Debug, Clone)]
pub struct BusAccumulator {
    /// Left channel accumulator.
    pub left: Vec<f32>,
    /// Right channel accumulator.
    pub right: Vec<f32>,
    /// Bus gain (linear).
    pub gain: f32,
    /// Whether bus is muted.
    pub muted: bool,
}

impl BusAccumulator {
    /// Create a zeroed accumulator for the given buffer size.
    #[must_use]
    pub fn new(buffer_size: usize) -> Self {
        Self {
            left: vec![0.0; buffer_size],
            right: vec![0.0; buffer_size],
            gain: 1.0,
            muted: false,
        }
    }

    /// Zero the accumulators for a new processing block.
    pub fn clear(&mut self) {
        self.left.fill(0.0);
        self.right.fill(0.0);
    }

    /// Add a stereo signal into this bus.
    pub fn add(&mut self, left: &[f32], right: &[f32]) {
        let n = self.left.len().min(left.len()).min(right.len());
        for i in 0..n {
            self.left[i] += left[i];
            self.right[i] += right[i];
        }
    }

    /// Apply bus gain and mute.
    pub fn apply_gain(&mut self) {
        if self.muted {
            self.left.fill(0.0);
            self.right.fill(0.0);
            return;
        }
        if (self.gain - 1.0).abs() > f32::EPSILON {
            for s in &mut self.left {
                *s *= self.gain;
            }
            for s in &mut self.right {
                *s *= self.gain;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Processing Engine
// ---------------------------------------------------------------------------

/// The mixer processing engine that orchestrates the full DSP pipeline.
///
/// This struct holds all runtime state needed during `process()`:
/// - Per-channel effects chains
/// - Per-bus effects chains
/// - Channel-to-bus routing assignments
/// - Per-channel aux sends
/// - VCA group states
/// - PDC delay lines
pub struct ProcessingEngine {
    /// Buffer size in samples.
    buffer_size: usize,

    /// Per-channel runtime effects chains.
    pub channel_effects: HashMap<ChannelId, RuntimeEffectsChain>,

    /// Per-bus runtime effects chains (for group / aux buses).
    pub bus_effects: HashMap<BusId, RuntimeEffectsChain>,

    /// Channel output routing (which bus each channel feeds).
    pub channel_routing: HashMap<ChannelId, ChannelOutputTarget>,

    /// Per-channel aux sends.
    pub channel_sends: HashMap<ChannelId, Vec<ProcessAuxSend>>,

    /// VCA groups.
    pub vca_groups: Vec<VcaGroupState>,

    /// PDC delay lines (one per channel).
    pub pdc_delays: HashMap<ChannelId, PdcDelayLine>,

    /// Maximum PDC latency across all channels (recomputed on demand).
    max_latency: usize,

    /// Bus accumulators keyed by BusId (group and aux buses).
    bus_accumulators: HashMap<BusId, BusAccumulator>,

    /// Which buses are group buses vs aux buses.
    pub bus_types: HashMap<BusId, BusType>,
}

impl ProcessingEngine {
    /// Create a new processing engine for the given buffer size.
    #[must_use]
    pub fn new(buffer_size: usize) -> Self {
        Self {
            buffer_size,
            channel_effects: HashMap::new(),
            bus_effects: HashMap::new(),
            channel_routing: HashMap::new(),
            channel_sends: HashMap::new(),
            vca_groups: Vec::new(),
            pdc_delays: HashMap::new(),
            max_latency: 0,
            bus_accumulators: HashMap::new(),
            bus_types: HashMap::new(),
        }
    }

    /// Register a bus for accumulation.
    pub fn register_bus(&mut self, bus_id: BusId, bus_type: BusType) {
        self.bus_accumulators
            .entry(bus_id)
            .or_insert_with(|| BusAccumulator::new(self.buffer_size));
        self.bus_types.insert(bus_id, bus_type);
    }

    /// Set a bus accumulator's gain.
    pub fn set_bus_gain(&mut self, bus_id: BusId, gain: f32) {
        if let Some(acc) = self.bus_accumulators.get_mut(&bus_id) {
            acc.gain = gain;
        }
    }

    /// Set a bus accumulator's mute state.
    pub fn set_bus_muted(&mut self, bus_id: BusId, muted: bool) {
        if let Some(acc) = self.bus_accumulators.get_mut(&bus_id) {
            acc.muted = muted;
        }
    }

    /// Recompute PDC delays based on current effects chain latencies.
    ///
    /// Finds the maximum latency across all channels and sets each channel's
    /// PDC delay to `max_latency - channel_latency`.
    pub fn recompute_pdc(&mut self) {
        // Compute per-channel latency.
        let latencies: HashMap<ChannelId, usize> = self
            .channel_effects
            .iter()
            .map(|(&id, chain)| (id, chain.total_latency()))
            .collect();

        self.max_latency = latencies.values().copied().max().unwrap_or(0);

        // Set PDC delay for each channel.
        for (&ch_id, &lat) in &latencies {
            let compensation = self.max_latency.saturating_sub(lat);
            let pdc = self
                .pdc_delays
                .entry(ch_id)
                .or_insert_with(|| PdcDelayLine::new(self.max_latency.max(1)));
            pdc.set_delay(compensation);
        }
    }

    /// Get the current max PDC latency.
    #[must_use]
    pub fn max_latency(&self) -> usize {
        self.max_latency
    }

    /// Compute the effective VCA gain multiplier for a channel.
    ///
    /// If the channel belongs to multiple VCA groups, their gains are
    /// multiplied together.
    #[must_use]
    pub fn vca_gain_for_channel(&self, channel_id: ChannelId) -> f32 {
        let mut gain = 1.0_f32;
        for vca in &self.vca_groups {
            if vca.channels.contains(&channel_id) {
                gain *= vca.effective_gain();
            }
        }
        gain
    }

    /// Process a single channel through effects, returning pre-fader and
    /// post-fader mono buffers.
    ///
    /// Steps:
    /// 1. Run the channel's effects chain on the working buffer.
    /// 2. Return the processed buffer (caller applies fader, pan, sends).
    pub fn process_channel_effects(&mut self, channel_id: ChannelId, working: &mut [f32]) {
        if let Some(chain) = self.channel_effects.get_mut(&channel_id) {
            chain.process(working);
        }
    }

    /// Process the full mixing pipeline for one buffer.
    ///
    /// # Arguments
    ///
    /// * `channel_data` — per-channel data: `(channel_id, gain, pan, muted,
    ///   input_gain_db, phase_inverted, pan_law)` plus the raw input samples.
    /// * `input_samples` — shared mono input buffer.
    ///
    /// # Returns
    ///
    /// `(master_left, master_right)` output buffers.
    #[allow(clippy::too_many_lines)]
    pub fn process_mix(
        &mut self,
        channels: &[(ChannelId, ChannelProcessParams)],
        input_samples: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let bs = self.buffer_size;
        let mut master_left = vec![0.0_f32; bs];
        let mut master_right = vec![0.0_f32; bs];

        // Clear all bus accumulators.
        for acc in self.bus_accumulators.values_mut() {
            acc.clear();
        }

        // --- Per-channel processing ---
        for (channel_id, params) in channels {
            if params.muted {
                continue;
            }

            // Step 1: Input gain + phase inversion
            let input_gain_linear = db_to_linear(params.input_gain_db);
            let phase_mult: f32 = if params.phase_inverted { -1.0 } else { 1.0 };

            let mut working: Vec<f32> = input_samples
                .iter()
                .take(bs)
                .map(|&s| s * input_gain_linear * phase_mult)
                .collect();
            working.resize(bs, 0.0);

            // Step 2: Effects chain
            self.process_channel_effects(*channel_id, &mut working);

            // Pre-fader tap for pre-fader sends
            let pre_fader_signal = working.clone();

            // Step 3: Fader gain × VCA gain
            let vca_gain = self.vca_gain_for_channel(*channel_id);
            let effective_fader = params.fader_gain * vca_gain;
            for sample in &mut working {
                *sample *= effective_fader;
            }

            // Step 4: Pan
            let (left_gain, right_gain) = compute_stereo_pan(params.pan, params.pan_law);

            let mut ch_left = vec![0.0_f32; bs];
            let mut ch_right = vec![0.0_f32; bs];
            for i in 0..bs {
                ch_left[i] = working[i] * left_gain;
                ch_right[i] = working[i] * right_gain;
            }

            // Step 5: PDC compensation on the panned output
            if let Some(pdc) = self.pdc_delays.get_mut(channel_id) {
                pdc.process_block(&mut ch_left, &mut ch_right);
            }

            // Step 6: Aux sends
            if let Some(sends) = self.channel_sends.get(channel_id) {
                for send in sends {
                    if !send.active || send.level.abs() < f32::EPSILON {
                        continue;
                    }
                    if let Some(acc) = self.bus_accumulators.get_mut(&send.bus_id) {
                        let source = if send.pre_fader {
                            &pre_fader_signal
                        } else {
                            &working
                        };
                        let level = send.level;
                        for i in 0..bs.min(source.len()) {
                            // Mono send → equal L/R contribution
                            let s = source[i] * level;
                            acc.left[i] += s;
                            acc.right[i] += s;
                        }
                    }
                }
            }

            // Step 7: Route to target bus or master
            let target = self
                .channel_routing
                .get(channel_id)
                .copied()
                .unwrap_or_default();

            match target {
                ChannelOutputTarget::Master => {
                    for i in 0..bs {
                        master_left[i] += ch_left[i];
                        master_right[i] += ch_right[i];
                    }
                }
                ChannelOutputTarget::GroupBus(bus_id) => {
                    if let Some(acc) = self.bus_accumulators.get_mut(&bus_id) {
                        acc.add(&ch_left, &ch_right);
                    } else {
                        // Fallback to master if bus not found
                        for i in 0..bs {
                            master_left[i] += ch_left[i];
                            master_right[i] += ch_right[i];
                        }
                    }
                }
            }
        }

        // --- Bus processing ---

        // Collect bus IDs by type for ordered processing.
        let aux_buses: Vec<BusId> = self
            .bus_types
            .iter()
            .filter(|(_, &t)| t == BusType::Auxiliary)
            .map(|(&id, _)| id)
            .collect();

        let group_buses: Vec<BusId> = self
            .bus_types
            .iter()
            .filter(|(_, &t)| t == BusType::Group)
            .map(|(&id, _)| id)
            .collect();

        // Process aux buses: apply effects chain, then add to master.
        for bus_id in &aux_buses {
            // Extract accumulator data to process effects.
            if let Some(acc) = self.bus_accumulators.get_mut(bus_id) {
                acc.apply_gain();
            }

            // Apply bus effects chain (e.g. reverb on aux bus).
            // We process left channel as mono through the effects chain.
            if let Some(chain) = self.bus_effects.get_mut(bus_id) {
                if let Some(acc) = self.bus_accumulators.get_mut(bus_id) {
                    chain.process(&mut acc.left);
                    chain.process(&mut acc.right);
                }
            }

            // Sum aux bus output into master.
            if let Some(acc) = self.bus_accumulators.get(bus_id) {
                if !acc.muted {
                    for i in 0..bs {
                        master_left[i] += acc.left[i];
                        master_right[i] += acc.right[i];
                    }
                }
            }
        }

        // Process group buses: apply gain + effects, then add to master.
        for bus_id in &group_buses {
            if let Some(acc) = self.bus_accumulators.get_mut(bus_id) {
                acc.apply_gain();
            }

            if let Some(chain) = self.bus_effects.get_mut(bus_id) {
                if let Some(acc) = self.bus_accumulators.get_mut(bus_id) {
                    chain.process(&mut acc.left);
                    chain.process(&mut acc.right);
                }
            }

            if let Some(acc) = self.bus_accumulators.get(bus_id) {
                if !acc.muted {
                    for i in 0..bs {
                        master_left[i] += acc.left[i];
                        master_right[i] += acc.right[i];
                    }
                }
            }
        }

        (master_left, master_right)
    }
}

// ---------------------------------------------------------------------------
// Channel processing parameters (passed per-channel into process_mix)
// ---------------------------------------------------------------------------

/// Parameters for processing a single channel through the mix pipeline.
#[derive(Debug, Clone)]
pub struct ChannelProcessParams {
    /// Fader gain (linear, 0.0–2.0).
    pub fader_gain: f32,
    /// Pan position (-1.0 to 1.0).
    pub pan: f32,
    /// Whether the channel is muted.
    pub muted: bool,
    /// Input gain in dB.
    pub input_gain_db: f32,
    /// Phase inverted.
    pub phase_inverted: bool,
    /// Pan law to apply.
    pub pan_law: PanLawType,
}

/// Pan law selection for the processing engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanLawType {
    /// Linear pan law.
    Linear,
    /// -3 dB equal power.
    Minus3dB,
    /// -4.5 dB compromise.
    Minus4Dot5dB,
    /// -6 dB equal gain.
    Minus6dB,
}

impl Default for PanLawType {
    fn default() -> Self {
        Self::Minus3dB
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Convert dB to linear gain.
#[inline]
#[must_use]
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// Compute stereo pan gains.
///
/// Returns `(left_gain, right_gain)`.
#[must_use]
fn compute_stereo_pan(pan: f32, law: PanLawType) -> (f32, f32) {
    let pan_norm = ((pan + 1.0) * 0.5).clamp(0.0, 1.0);

    match law {
        PanLawType::Linear | PanLawType::Minus6dB => (1.0 - pan_norm, pan_norm),
        PanLawType::Minus3dB => {
            let angle = pan_norm * std::f32::consts::FRAC_PI_2;
            (angle.cos(), angle.sin())
        }
        PanLawType::Minus4Dot5dB => {
            let linear_l = 1.0 - pan_norm;
            let linear_r = pan_norm;
            let angle = pan_norm * std::f32::consts::FRAC_PI_2;
            let power_l = angle.cos();
            let power_r = angle.sin();
            (
                0.5 * linear_l + 0.5 * power_l,
                0.5 * linear_r + 0.5 * power_r,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects_chain::DelayEffect;
    use uuid::Uuid;

    fn make_channel_id() -> ChannelId {
        ChannelId(Uuid::new_v4())
    }

    fn make_bus_id() -> BusId {
        BusId(Uuid::new_v4())
    }

    // --- RuntimeEffectsChain ---

    #[test]
    fn test_runtime_chain_empty_passthrough() {
        let mut chain = RuntimeEffectsChain::new();
        let mut samples = [0.5_f32, 0.3, 0.1];
        let original = samples;
        chain.process(&mut samples);
        for (s, o) in samples.iter().zip(original.iter()) {
            assert!((s - o).abs() < 1e-6, "empty chain should pass through");
        }
    }

    #[test]
    fn test_runtime_chain_bypassed_slot() {
        let mut chain = RuntimeEffectsChain::new();
        let mut slot = RuntimeEffectSlot::new(Box::new(DelayEffect::new(10, 0.5, 1.0)));
        slot.bypassed = true;
        chain.add(slot);
        let mut samples = [0.5_f32; 64];
        let original = samples;
        chain.process(&mut samples);
        for (s, o) in samples.iter().zip(original.iter()) {
            assert!((s - o).abs() < 1e-6, "bypassed slot should pass through");
        }
    }

    #[test]
    fn test_runtime_chain_with_effect() {
        let mut chain = RuntimeEffectsChain::new();
        // Delay with mix=0 means fully dry — output == input.
        chain.add(RuntimeEffectSlot::new(Box::new(DelayEffect::new(
            10, 0.0, 0.0,
        ))));
        let mut samples = [0.5_f32; 64];
        let original = samples;
        chain.process(&mut samples);
        for (s, o) in samples.iter().zip(original.iter()) {
            assert!((s - o).abs() < 1e-6);
        }
    }

    #[test]
    fn test_runtime_chain_partial_mix() {
        let mut chain = RuntimeEffectsChain::new();
        let mut slot = RuntimeEffectSlot::new(Box::new(DelayEffect::new(10, 0.0, 0.0)));
        slot.mix = 0.5; // 50% mix of a fully-dry effect = still 100% original
        chain.add(slot);
        let mut samples = [0.8_f32; 32];
        chain.process(&mut samples);
        for s in &samples {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn test_runtime_chain_latency() {
        let mut chain = RuntimeEffectsChain::new();
        chain.add(RuntimeEffectSlot::with_latency(
            Box::new(DelayEffect::new(10, 0.0, 0.0)),
            128,
        ));
        chain.add(RuntimeEffectSlot::with_latency(
            Box::new(DelayEffect::new(10, 0.0, 0.0)),
            256,
        ));
        assert_eq!(chain.total_latency(), 384);
    }

    #[test]
    fn test_runtime_chain_latency_bypassed_excluded() {
        let mut chain = RuntimeEffectsChain::new();
        let mut slot =
            RuntimeEffectSlot::with_latency(Box::new(DelayEffect::new(10, 0.0, 0.0)), 128);
        slot.bypassed = true;
        chain.add(slot);
        assert_eq!(chain.total_latency(), 0);
    }

    #[test]
    fn test_runtime_chain_remove() {
        let mut chain = RuntimeEffectsChain::new();
        chain.add(RuntimeEffectSlot::new(Box::new(DelayEffect::new(
            10, 0.0, 0.0,
        ))));
        chain.add(RuntimeEffectSlot::new(Box::new(DelayEffect::new(
            20, 0.0, 0.0,
        ))));
        assert_eq!(chain.len(), 2);
        assert!(chain.remove(0).is_some());
        assert_eq!(chain.len(), 1);
        assert!(chain.remove(5).is_none());
    }

    // --- PdcDelayLine ---

    #[test]
    fn test_pdc_zero_delay_passthrough() {
        let mut pdc = PdcDelayLine::new(1024);
        pdc.set_delay(0);
        let (l, r) = pdc.process_sample(0.5, 0.3);
        assert!((l - 0.5).abs() < 1e-6);
        assert!((r - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_pdc_delay_introduces_latency() {
        let mut pdc = PdcDelayLine::new(16);
        pdc.set_delay(4);

        // Feed 4 samples of value 1.0
        for _ in 0..4 {
            let (l, _r) = pdc.process_sample(1.0, 1.0);
            // During the first 4 samples, output should be 0 (delay buffer was zeroed)
            assert!(l.abs() < 1e-6, "should be zero during delay period");
        }
        // Sample 5: we should now see the first 1.0 coming out
        let (l, _r) = pdc.process_sample(1.0, 1.0);
        assert!(
            (l - 1.0).abs() < 1e-6,
            "should output 1.0 after delay period"
        );
    }

    #[test]
    fn test_pdc_block_processing() {
        let mut pdc = PdcDelayLine::new(8);
        pdc.set_delay(2);
        let mut left = vec![1.0_f32; 8];
        let mut right = vec![0.5_f32; 8];
        pdc.process_block(&mut left, &mut right);
        // First 2 samples should be 0 (delay compensation)
        assert!(left[0].abs() < 1e-6);
        assert!(left[1].abs() < 1e-6);
        // Sample 2 onward should be 1.0
        assert!((left[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pdc_clear() {
        let mut pdc = PdcDelayLine::new(8);
        pdc.set_delay(2);
        for _ in 0..8 {
            pdc.process_sample(1.0, 1.0);
        }
        pdc.clear();
        let (l, r) = pdc.process_sample(0.0, 0.0);
        assert!(l.abs() < 1e-6);
        assert!(r.abs() < 1e-6);
    }

    // --- BusAccumulator ---

    #[test]
    fn test_bus_accumulator_add() {
        let mut acc = BusAccumulator::new(4);
        acc.add(&[1.0, 2.0, 3.0, 4.0], &[0.5, 0.5, 0.5, 0.5]);
        assert!((acc.left[0] - 1.0).abs() < 1e-6);
        assert!((acc.right[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_bus_accumulator_gain() {
        let mut acc = BusAccumulator::new(4);
        acc.add(&[1.0; 4], &[1.0; 4]);
        acc.gain = 0.5;
        acc.apply_gain();
        for s in &acc.left {
            assert!((*s - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_bus_accumulator_muted() {
        let mut acc = BusAccumulator::new(4);
        acc.add(&[1.0; 4], &[1.0; 4]);
        acc.muted = true;
        acc.apply_gain();
        for s in &acc.left {
            assert!(s.abs() < 1e-6);
        }
    }

    // --- VcaGroupState ---

    #[test]
    fn test_vca_group_effective_gain() {
        let mut vca = VcaGroupState::new("Test".into());
        assert!((vca.effective_gain() - 1.0).abs() < 1e-6);
        vca.gain = 0.5;
        assert!((vca.effective_gain() - 0.5).abs() < 1e-6);
        vca.muted = true;
        assert!(vca.effective_gain().abs() < 1e-6);
    }

    // --- ProcessingEngine integration ---

    #[test]
    fn test_engine_direct_to_master() {
        let bs = 64;
        let mut engine = ProcessingEngine::new(bs);
        let ch_id = make_channel_id();

        let input = vec![1.0_f32; bs];
        let params = ChannelProcessParams {
            fader_gain: 1.0,
            pan: 0.0,
            muted: false,
            input_gain_db: 0.0,
            phase_inverted: false,
            pan_law: PanLawType::Minus3dB,
        };

        let (ml, mr) = engine.process_mix(&[(ch_id, params)], &input);
        // Center pan with equal-power: left ≈ cos(π/4) ≈ 0.707
        for i in 0..bs {
            assert!(
                (ml[i] - mr[i]).abs() < 1e-4,
                "center pan should produce equal L/R"
            );
            assert!(ml[i] > 0.5, "should have signal");
        }
    }

    #[test]
    fn test_engine_muted_channel_silence() {
        let bs = 32;
        let mut engine = ProcessingEngine::new(bs);
        let ch_id = make_channel_id();

        let input = vec![1.0_f32; bs];
        let params = ChannelProcessParams {
            fader_gain: 1.0,
            pan: 0.0,
            muted: true,
            input_gain_db: 0.0,
            phase_inverted: false,
            pan_law: PanLawType::Minus3dB,
        };

        let (ml, mr) = engine.process_mix(&[(ch_id, params)], &input);
        for i in 0..bs {
            assert!(ml[i].abs() < 1e-6);
            assert!(mr[i].abs() < 1e-6);
        }
    }

    #[test]
    fn test_engine_group_bus_routing() {
        let bs = 32;
        let mut engine = ProcessingEngine::new(bs);
        let ch_id = make_channel_id();
        let bus_id = make_bus_id();

        // Register group bus and route channel to it.
        engine.register_bus(bus_id, BusType::Group);
        engine
            .channel_routing
            .insert(ch_id, ChannelOutputTarget::GroupBus(bus_id));

        let input = vec![1.0_f32; bs];
        let params = ChannelProcessParams {
            fader_gain: 1.0,
            pan: 0.0,
            muted: false,
            input_gain_db: 0.0,
            phase_inverted: false,
            pan_law: PanLawType::Linear,
        };

        let (ml, mr) = engine.process_mix(&[(ch_id, params)], &input);
        // With linear pan at center: L=0.5, R=0.5; group bus applies gain 1.0
        for i in 0..bs {
            assert!(
                (ml[i] - 0.5).abs() < 1e-4,
                "group bus should route to master, got {}",
                ml[i]
            );
            assert!((mr[i] - 0.5).abs() < 1e-4);
        }
    }

    #[test]
    fn test_engine_aux_send_return() {
        let bs = 32;
        let mut engine = ProcessingEngine::new(bs);
        let ch_id = make_channel_id();
        let aux_bus_id = make_bus_id();

        // Register aux bus.
        engine.register_bus(aux_bus_id, BusType::Auxiliary);

        // Configure post-fader send at unity level.
        engine.channel_sends.insert(
            ch_id,
            vec![ProcessAuxSend {
                bus_id: aux_bus_id,
                level: 1.0,
                pre_fader: false,
                active: true,
            }],
        );

        let input = vec![0.5_f32; bs];
        let params = ChannelProcessParams {
            fader_gain: 1.0,
            pan: 0.0,
            muted: false,
            input_gain_db: 0.0,
            phase_inverted: false,
            pan_law: PanLawType::Linear,
        };

        let (ml, _mr) = engine.process_mix(&[(ch_id, params)], &input);
        // The channel goes to master (direct) + aux bus goes to master.
        // Channel L = 0.5 * 0.5 = 0.25 (linear pan center)
        // Aux bus receives post-fader mono = 0.5 * 1.0 = 0.5, and adds to master
        // Total L = 0.25 + 0.5 = 0.75
        for i in 0..bs {
            assert!(
                (ml[i] - 0.75).abs() < 0.01,
                "aux send should contribute to master, got {}",
                ml[i]
            );
        }
    }

    #[test]
    fn test_engine_pre_fader_send() {
        let bs = 32;
        let mut engine = ProcessingEngine::new(bs);
        let ch_id = make_channel_id();
        let aux_bus_id = make_bus_id();

        engine.register_bus(aux_bus_id, BusType::Auxiliary);
        engine.channel_sends.insert(
            ch_id,
            vec![ProcessAuxSend {
                bus_id: aux_bus_id,
                level: 1.0,
                pre_fader: true,
                active: true,
            }],
        );

        let input = vec![1.0_f32; bs];
        // Fader at zero — post-fader signal is zero, but pre-fader send
        // should still contribute.
        let params = ChannelProcessParams {
            fader_gain: 0.0,
            pan: 0.0,
            muted: false,
            input_gain_db: 0.0,
            phase_inverted: false,
            pan_law: PanLawType::Linear,
        };

        let (ml, _mr) = engine.process_mix(&[(ch_id, params)], &input);
        // Channel direct output = 0 (fader at 0).
        // Pre-fader send = 1.0 * 1.0 = 1.0 per sample into aux bus.
        // Aux bus adds to master.
        for i in 0..bs {
            assert!(
                ml[i] > 0.5,
                "pre-fader send should still contribute even with fader at 0, got {}",
                ml[i]
            );
        }
    }

    #[test]
    fn test_engine_vca_group() {
        let bs = 32;
        let mut engine = ProcessingEngine::new(bs);
        let ch_id = make_channel_id();

        // Create VCA group at -6 dB
        let mut vca = VcaGroupState::new("Drums".into());
        vca.gain = 0.5;
        vca.channels.push(ch_id);
        engine.vca_groups.push(vca);

        let input = vec![1.0_f32; bs];
        let params = ChannelProcessParams {
            fader_gain: 1.0,
            pan: 0.0,
            muted: false,
            input_gain_db: 0.0,
            phase_inverted: false,
            pan_law: PanLawType::Minus3dB,
        };

        let (ml_vca, _) = engine.process_mix(&[(ch_id, params.clone())], &input);

        // Now without VCA
        engine.vca_groups.clear();
        let (ml_no_vca, _) = engine.process_mix(&[(ch_id, params)], &input);

        // VCA at 0.5 should halve the output.
        for i in 0..bs {
            let ratio = if ml_no_vca[i].abs() > 1e-9 {
                ml_vca[i] / ml_no_vca[i]
            } else {
                1.0
            };
            assert!(
                (ratio - 0.5).abs() < 0.01,
                "VCA should halve the output, ratio = {ratio}"
            );
        }
    }

    #[test]
    fn test_engine_vca_muted() {
        let bs = 16;
        let mut engine = ProcessingEngine::new(bs);
        let ch_id = make_channel_id();

        let mut vca = VcaGroupState::new("Muted".into());
        vca.muted = true;
        vca.channels.push(ch_id);
        engine.vca_groups.push(vca);

        let input = vec![1.0_f32; bs];
        let params = ChannelProcessParams {
            fader_gain: 1.0,
            pan: 0.0,
            muted: false,
            input_gain_db: 0.0,
            phase_inverted: false,
            pan_law: PanLawType::Minus3dB,
        };

        let (ml, mr) = engine.process_mix(&[(ch_id, params)], &input);
        for i in 0..bs {
            assert!(ml[i].abs() < 1e-6, "VCA muted should silence channel");
            assert!(mr[i].abs() < 1e-6);
        }
    }

    #[test]
    fn test_engine_effects_chain_integration() {
        let bs = 64;
        let mut engine = ProcessingEngine::new(bs);
        let ch_id = make_channel_id();

        // Add a delay effect with mix=0 (fully dry) — output should equal input.
        let mut chain = RuntimeEffectsChain::new();
        chain.add(RuntimeEffectSlot::new(Box::new(DelayEffect::new(
            10, 0.0, 0.0,
        ))));
        engine.channel_effects.insert(ch_id, chain);

        let input = vec![0.75_f32; bs];
        let params = ChannelProcessParams {
            fader_gain: 1.0,
            pan: 0.0,
            muted: false,
            input_gain_db: 0.0,
            phase_inverted: false,
            pan_law: PanLawType::Minus3dB,
        };

        let (ml, mr) = engine.process_mix(&[(ch_id, params)], &input);
        // At center pan with equal power, L and R should be equal.
        for i in 0..bs {
            assert!(
                (ml[i] - mr[i]).abs() < 1e-4,
                "should produce equal L/R at center pan"
            );
        }
    }

    #[test]
    fn test_engine_pdc_recompute() {
        let bs = 32;
        let mut engine = ProcessingEngine::new(bs);
        let ch1 = make_channel_id();
        let ch2 = make_channel_id();

        // ch1 has 128 samples latency, ch2 has 0.
        let mut chain1 = RuntimeEffectsChain::new();
        chain1.add(RuntimeEffectSlot::with_latency(
            Box::new(DelayEffect::new(10, 0.0, 0.0)),
            128,
        ));
        engine.channel_effects.insert(ch1, chain1);

        let chain2 = RuntimeEffectsChain::new();
        engine.channel_effects.insert(ch2, chain2);

        engine.recompute_pdc();

        assert_eq!(engine.max_latency(), 128);
        // ch1 should have 0 compensation (it's the longest).
        assert_eq!(engine.pdc_delays.get(&ch1).map_or(0, |p| p.delay()), 0);
        // ch2 should have 128 samples compensation.
        assert_eq!(engine.pdc_delays.get(&ch2).map_or(0, |p| p.delay()), 128);
    }

    #[test]
    fn test_engine_bus_effects() {
        let bs = 32;
        let mut engine = ProcessingEngine::new(bs);
        let ch_id = make_channel_id();
        let group_bus_id = make_bus_id();

        engine.register_bus(group_bus_id, BusType::Group);
        engine
            .channel_routing
            .insert(ch_id, ChannelOutputTarget::GroupBus(group_bus_id));

        // Add a dry delay to the group bus effects chain.
        let mut bus_chain = RuntimeEffectsChain::new();
        bus_chain.add(RuntimeEffectSlot::new(Box::new(DelayEffect::new(
            10, 0.0, 0.0,
        ))));
        engine.bus_effects.insert(group_bus_id, bus_chain);

        let input = vec![1.0_f32; bs];
        let params = ChannelProcessParams {
            fader_gain: 1.0,
            pan: 0.0,
            muted: false,
            input_gain_db: 0.0,
            phase_inverted: false,
            pan_law: PanLawType::Linear,
        };

        let (ml, _mr) = engine.process_mix(&[(ch_id, params)], &input);
        // Should still have signal (dry effect passthrough).
        let has_signal = ml.iter().any(|&s| s.abs() > 0.1);
        assert!(
            has_signal,
            "bus effects should process and pass signal through"
        );
    }

    #[test]
    fn test_engine_multiple_channels_to_same_group() {
        let bs = 16;
        let mut engine = ProcessingEngine::new(bs);
        let ch1 = make_channel_id();
        let ch2 = make_channel_id();
        let bus_id = make_bus_id();

        engine.register_bus(bus_id, BusType::Group);
        engine
            .channel_routing
            .insert(ch1, ChannelOutputTarget::GroupBus(bus_id));
        engine
            .channel_routing
            .insert(ch2, ChannelOutputTarget::GroupBus(bus_id));

        let input = vec![1.0_f32; bs];
        let params = ChannelProcessParams {
            fader_gain: 1.0,
            pan: 0.0,
            muted: false,
            input_gain_db: 0.0,
            phase_inverted: false,
            pan_law: PanLawType::Linear,
        };

        let (ml_both, _) = engine.process_mix(&[(ch1, params.clone()), (ch2, params)], &input);

        // Two channels at linear center pan (L=0.5 each) summed = 1.0.
        for i in 0..bs {
            assert!(
                (ml_both[i] - 1.0).abs() < 0.01,
                "two channels should sum on group bus, got {}",
                ml_both[i]
            );
        }
    }

    #[test]
    fn test_engine_group_bus_gain() {
        let bs = 16;
        let mut engine = ProcessingEngine::new(bs);
        let ch_id = make_channel_id();
        let bus_id = make_bus_id();

        engine.register_bus(bus_id, BusType::Group);
        engine.set_bus_gain(bus_id, 0.5);
        engine
            .channel_routing
            .insert(ch_id, ChannelOutputTarget::GroupBus(bus_id));

        let input = vec![1.0_f32; bs];
        let params = ChannelProcessParams {
            fader_gain: 1.0,
            pan: 0.0,
            muted: false,
            input_gain_db: 0.0,
            phase_inverted: false,
            pan_law: PanLawType::Linear,
        };

        let (ml, _) = engine.process_mix(&[(ch_id, params)], &input);
        // Channel at linear center = L=0.5. Bus gain = 0.5. Result = 0.25.
        for i in 0..bs {
            assert!(
                (ml[i] - 0.25).abs() < 0.01,
                "group bus gain should attenuate, got {}",
                ml[i]
            );
        }
    }

    #[test]
    fn test_pan_hard_left_silence_right() {
        let (l, r) = compute_stereo_pan(-1.0, PanLawType::Linear);
        assert!((l - 1.0).abs() < 1e-6);
        assert!(r.abs() < 1e-6, "hard left should silence right channel");
    }

    #[test]
    fn test_pan_hard_right_silence_left() {
        let (l, r) = compute_stereo_pan(1.0, PanLawType::Linear);
        assert!(l.abs() < 1e-6, "hard right should silence left channel");
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pan_center_equal_power() {
        let (l, r) = compute_stereo_pan(0.0, PanLawType::Minus3dB);
        assert!((l - r).abs() < 1e-4, "center pan should produce equal L/R");
        // cos(π/4) ≈ 0.7071
        assert!((l - 0.7071).abs() < 0.001);
    }

    #[test]
    fn test_gain_half_produces_correct_output() {
        let bs = 16;
        let mut engine = ProcessingEngine::new(bs);
        let ch_id = make_channel_id();

        let input = vec![1.0_f32; bs];
        let params_full = ChannelProcessParams {
            fader_gain: 1.0,
            pan: 0.0,
            muted: false,
            input_gain_db: 0.0,
            phase_inverted: false,
            pan_law: PanLawType::Minus3dB,
        };
        let params_half = ChannelProcessParams {
            fader_gain: 0.5,
            ..params_full.clone()
        };

        let (ml_full, _) = engine.process_mix(&[(ch_id, params_full)], &input);
        let (ml_half, _) = engine.process_mix(&[(ch_id, params_half)], &input);

        for i in 0..bs {
            let ratio = if ml_full[i].abs() > 1e-9 {
                ml_half[i] / ml_full[i]
            } else {
                0.5
            };
            assert!(
                (ratio - 0.5).abs() < 0.01,
                "gain=0.5 should produce half output, ratio = {ratio}"
            );
        }
    }

    #[test]
    fn test_phase_inversion() {
        let bs = 16;
        let mut engine = ProcessingEngine::new(bs);
        let ch_id = make_channel_id();

        let input = vec![1.0_f32; bs];
        let params_normal = ChannelProcessParams {
            fader_gain: 1.0,
            pan: 0.0,
            muted: false,
            input_gain_db: 0.0,
            phase_inverted: false,
            pan_law: PanLawType::Linear,
        };
        let params_inverted = ChannelProcessParams {
            phase_inverted: true,
            ..params_normal.clone()
        };

        let (ml_normal, _) = engine.process_mix(&[(ch_id, params_normal)], &input);
        let (ml_inverted, _) = engine.process_mix(&[(ch_id, params_inverted)], &input);

        for i in 0..bs {
            assert!(
                (ml_normal[i] + ml_inverted[i]).abs() < 1e-6,
                "phase inversion should negate signal"
            );
        }
    }

    #[test]
    fn test_send_inactive_no_contribution() {
        let bs = 16;
        let mut engine = ProcessingEngine::new(bs);
        let ch_id = make_channel_id();
        let aux_bus_id = make_bus_id();

        engine.register_bus(aux_bus_id, BusType::Auxiliary);
        engine.channel_sends.insert(
            ch_id,
            vec![ProcessAuxSend {
                bus_id: aux_bus_id,
                level: 1.0,
                pre_fader: false,
                active: false, // disabled
            }],
        );

        let input = vec![1.0_f32; bs];
        let params = ChannelProcessParams {
            fader_gain: 1.0,
            pan: 0.0,
            muted: false,
            input_gain_db: 0.0,
            phase_inverted: false,
            pan_law: PanLawType::Linear,
        };

        // With send disabled, only direct signal (no aux contribution)
        let (ml_disabled, _) = engine.process_mix(&[(ch_id, params.clone())], &input);

        // Enable send
        if let Some(sends) = engine.channel_sends.get_mut(&ch_id) {
            sends[0].active = true;
        }
        let (ml_enabled, _) = engine.process_mix(&[(ch_id, params)], &input);

        // Enabled should be louder (aux adds to master).
        assert!(
            ml_enabled[0] > ml_disabled[0],
            "active send should add signal"
        );
    }

    #[test]
    fn test_muted_bus_no_output() {
        let bs = 16;
        let mut engine = ProcessingEngine::new(bs);
        let ch_id = make_channel_id();
        let bus_id = make_bus_id();

        engine.register_bus(bus_id, BusType::Group);
        engine.set_bus_muted(bus_id, true);
        engine
            .channel_routing
            .insert(ch_id, ChannelOutputTarget::GroupBus(bus_id));

        let input = vec![1.0_f32; bs];
        let params = ChannelProcessParams {
            fader_gain: 1.0,
            pan: 0.0,
            muted: false,
            input_gain_db: 0.0,
            phase_inverted: false,
            pan_law: PanLawType::Linear,
        };

        let (ml, mr) = engine.process_mix(&[(ch_id, params)], &input);
        for i in 0..bs {
            assert!(ml[i].abs() < 1e-6, "muted group bus should silence output");
            assert!(mr[i].abs() < 1e-6);
        }
    }
}
