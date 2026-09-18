//! Plugin hosting API for external audio effects.
//!
//! Defines a VST3-style pure-Rust interface for hosting audio effect plugins.
//! Plugins implement the [`AudioPlugin`](crate::plugin::AudioPlugin) trait and are managed by a
//! [`PluginHost`](crate::plugin::PluginHost) that handles lifecycle, parameter management, and processing.
//!
//! # Architecture
//!
//! ```text
//! PluginHost
//!   ├── PluginDescriptor  (static metadata)
//!   ├── PluginInstance    (active processing state)
//!   │     ├── AudioPlugin (DSP processing)
//!   │     └── ParameterStore (lock-free parameters)
//!   └── PluginRegistry    (available plugin descriptors)
//! ```
//!
//! # Example
//!
//! ```rust
//! use oximedia_mixer::plugin::{AudioPlugin, ParameterInfo, PluginCategory, PluginDescriptor, PluginHost};
//!
//! struct MyGain;
//!
//! impl AudioPlugin for MyGain {
//!     fn descriptor(&self) -> PluginDescriptor {
//!         PluginDescriptor {
//!             id: "com.example.mygain".to_string(),
//!             name: "My Gain".to_string(),
//!             vendor: "Example".to_string(),
//!             version: "1.0.0".to_string(),
//!             category: PluginCategory::Gain,
//!             num_inputs: 2,
//!             num_outputs: 2,
//!             has_editor: false,
//!         }
//!     }
//!
//!     fn prepare(&mut self, sample_rate: f64, max_block_size: usize) {
//!         let _ = (sample_rate, max_block_size);
//!     }
//!
//!     fn process(&mut self, inputs: &[&[f32]], outputs: &mut [&mut [f32]], num_samples: usize) {
//!         for (inp, out) in inputs.iter().zip(outputs.iter_mut()) {
//!             for i in 0..num_samples {
//!                 out[i] = inp[i] * 0.5;
//!             }
//!         }
//!     }
//!
//!     fn reset(&mut self) {}
//!
//!     fn parameter_count(&self) -> usize { 1 }
//!
//!     fn parameter_info(&self, _idx: usize) -> Option<ParameterInfo> {
//!         Some(ParameterInfo {
//!             id: 0,
//!             name: "Gain".to_string(),
//!             min: 0.0,
//!             max: 2.0,
//!             default: 0.5,
//!             unit: "".to_string(),
//!             automatable: true,
//!         })
//!     }
//!
//!     fn get_parameter(&self, _id: u32) -> f32 { 0.5 }
//!
//!     fn set_parameter(&mut self, _id: u32, _value: f32) {}
//!
//!     fn latency_samples(&self) -> u32 { 0 }
//! }
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Plugin Category
// ---------------------------------------------------------------------------

/// Audio effect plugin category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PluginCategory {
    /// Gain / utility.
    Gain,
    /// Dynamics processing (compressor, limiter, gate, expander).
    Dynamics,
    /// Equalisation.
    Equalizer,
    /// Time-based effects (reverb, delay, chorus, flanger).
    TimeBased,
    /// Modulation effects (phaser, tremolo, vibrato, ring mod).
    Modulation,
    /// Distortion and saturation.
    Distortion,
    /// Pitch shifting and time-stretching.
    PitchShift,
    /// Surround and spatial processing.
    Spatial,
    /// Spectral / FFT-based processing.
    Spectral,
    /// Metering and analysis.
    Analyzer,
    /// Other / unclassified.
    Other,
}

impl std::fmt::Display for PluginCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gain => write!(f, "Gain"),
            Self::Dynamics => write!(f, "Dynamics"),
            Self::Equalizer => write!(f, "Equalizer"),
            Self::TimeBased => write!(f, "Time-Based"),
            Self::Modulation => write!(f, "Modulation"),
            Self::Distortion => write!(f, "Distortion"),
            Self::PitchShift => write!(f, "Pitch Shift"),
            Self::Spatial => write!(f, "Spatial"),
            Self::Spectral => write!(f, "Spectral"),
            Self::Analyzer => write!(f, "Analyzer"),
            Self::Other => write!(f, "Other"),
        }
    }
}

// ---------------------------------------------------------------------------
// PluginDescriptor
// ---------------------------------------------------------------------------

/// Static metadata for a plugin.
#[derive(Debug, Clone)]
pub struct PluginDescriptor {
    /// Unique plugin identifier (reverse-domain, e.g. `com.vendor.plugin`).
    pub id: String,
    /// Human-readable plugin name.
    pub name: String,
    /// Plugin vendor/author.
    pub vendor: String,
    /// Plugin version string.
    pub version: String,
    /// Plugin category.
    pub category: PluginCategory,
    /// Number of audio input channels (0 = instrument).
    pub num_inputs: u32,
    /// Number of audio output channels.
    pub num_outputs: u32,
    /// Whether the plugin supports a graphical editor.
    pub has_editor: bool,
}

// ---------------------------------------------------------------------------
// ParameterInfo
// ---------------------------------------------------------------------------

/// Metadata for a single plugin parameter.
#[derive(Debug, Clone)]
pub struct ParameterInfo {
    /// Numeric parameter ID (used with `get_parameter`/`set_parameter`).
    pub id: u32,
    /// Human-readable parameter name.
    pub name: String,
    /// Minimum value.
    pub min: f32,
    /// Maximum value.
    pub max: f32,
    /// Default value.
    pub default: f32,
    /// Unit label (e.g. `"dB"`, `"Hz"`, `"ms"`).
    pub unit: String,
    /// Whether this parameter can be automated.
    pub automatable: bool,
}

// ---------------------------------------------------------------------------
// AudioPlugin trait
// ---------------------------------------------------------------------------

/// Core trait that all audio effect plugins must implement.
///
/// Implementations must be `Send` so they can be transferred to the audio
/// thread.  They need not be `Sync` since they are only accessed from one
/// thread at a time.
pub trait AudioPlugin: Send {
    /// Return the plugin descriptor.
    fn descriptor(&self) -> PluginDescriptor;

    /// Called before processing starts.  The plugin should allocate internal
    /// buffers here.
    fn prepare(&mut self, sample_rate: f64, max_block_size: usize);

    /// Process a block of audio.
    ///
    /// * `inputs`  — one slice per input channel
    /// * `outputs` — one mutable slice per output channel
    /// * `num_samples` — number of valid samples in each slice
    fn process(&mut self, inputs: &[&[f32]], outputs: &mut [&mut [f32]], num_samples: usize);

    /// Reset all internal state (e.g. delay buffers, filter histories).
    fn reset(&mut self);

    /// Number of parameters exposed by this plugin.
    fn parameter_count(&self) -> usize;

    /// Get parameter metadata by index.  Returns `None` if index is out of
    /// range.
    fn parameter_info(&self, idx: usize) -> Option<ParameterInfo>;

    /// Get the current value of a parameter by its numeric ID.
    fn get_parameter(&self, id: u32) -> f32;

    /// Set a parameter value by its numeric ID.
    fn set_parameter(&mut self, id: u32, value: f32);

    /// Lookahead latency introduced by this plugin (in samples).
    fn latency_samples(&self) -> u32;

    /// Called when processing ends.  Default implementation is a no-op.
    fn release(&mut self) {}

    /// Return the plugin's current tail time in samples.
    ///
    /// The host will continue calling `process` for this many samples after
    /// the input goes silent (e.g. for reverb tails).  Return `0` for
    /// zero-latency plugins.
    fn tail_samples(&self) -> u32 {
        0
    }
}

// ---------------------------------------------------------------------------
// Lock-free parameter store
// ---------------------------------------------------------------------------

/// Lock-free store for plugin parameters using `AtomicU32` bit-casting.
///
/// The UI thread writes values; the audio thread reads them without blocking.
#[derive(Debug)]
pub struct ParameterStore {
    /// Parameter values stored as atomic bit-casts of `f32`.
    values: HashMap<u32, AtomicU32>,
    /// Parameter metadata.
    info: HashMap<u32, ParameterInfo>,
}

impl ParameterStore {
    /// Create a new parameter store populated from a plugin's parameter list.
    #[must_use]
    pub fn from_plugin(plugin: &dyn AudioPlugin) -> Self {
        let count = plugin.parameter_count();
        let mut values = HashMap::with_capacity(count);
        let mut info = HashMap::with_capacity(count);
        for idx in 0..count {
            if let Some(param) = plugin.parameter_info(idx) {
                let current = plugin.get_parameter(param.id);
                values.insert(param.id, AtomicU32::new(current.to_bits()));
                info.insert(param.id, param);
            }
        }
        Self { values, info }
    }

    /// Read a parameter value (relaxed — safe for audio thread polling).
    #[must_use]
    pub fn get(&self, id: u32) -> Option<f32> {
        self.values
            .get(&id)
            .map(|a| f32::from_bits(a.load(Ordering::Relaxed)))
    }

    /// Write a parameter value, clamped to the parameter's valid range.
    ///
    /// Returns `false` if the parameter ID is unknown.
    pub fn set(&self, id: u32, value: f32) -> bool {
        if let Some(atom) = self.values.get(&id) {
            let clamped = if let Some(inf) = self.info.get(&id) {
                value.clamp(inf.min, inf.max)
            } else {
                value
            };
            atom.store(clamped.to_bits(), Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// List all parameter IDs.
    #[must_use]
    pub fn parameter_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.values.keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// Get parameter info.
    #[must_use]
    pub fn info(&self, id: u32) -> Option<&ParameterInfo> {
        self.info.get(&id)
    }
}

// ---------------------------------------------------------------------------
// PluginInstance
// ---------------------------------------------------------------------------

/// Error type for plugin operations.
#[derive(Debug, thiserror::Error)]
pub enum PluginError {
    /// Plugin not found in the registry.
    #[error("Plugin not found: {0}")]
    NotFound(String),

    /// Plugin instance not found.
    #[error("Plugin instance not found: {0}")]
    InstanceNotFound(u32),

    /// Incompatible channel configuration.
    #[error("Incompatible channel count: expected {expected}, got {got}")]
    ChannelMismatch {
        /// Expected channel count.
        expected: u32,
        /// Actual channel count.
        got: u32,
    },

    /// Plugin processing error.
    #[error("Plugin processing error: {0}")]
    ProcessingError(String),
}

/// Result type for plugin operations.
pub type PluginResult<T> = Result<T, PluginError>;

/// A live instance of an audio plugin inside the host.
pub struct PluginInstance {
    /// Unique instance identifier (assigned by the host).
    pub instance_id: u32,
    /// The live plugin.
    pub plugin: Box<dyn AudioPlugin>,
    /// Lock-free parameter store for this instance.
    pub params: Arc<ParameterStore>,
    /// Whether the plugin is bypassed.
    pub bypassed: bool,
    /// Wet/dry mix (0.0 = fully dry, 1.0 = fully wet).
    pub wet_mix: f32,
    /// Whether the plugin is prepared (i.e. `prepare()` has been called).
    prepared: bool,
}

impl PluginInstance {
    /// Create a new plugin instance.
    #[must_use]
    pub fn new(instance_id: u32, mut plugin: Box<dyn AudioPlugin>) -> Self {
        let params = Arc::new(ParameterStore::from_plugin(plugin.as_ref()));
        // Force initial parameter sync into the plugin.
        for id in params.parameter_ids() {
            if let Some(v) = params.get(id) {
                plugin.set_parameter(id, v);
            }
        }
        Self {
            instance_id,
            plugin,
            params,
            bypassed: false,
            wet_mix: 1.0,
            prepared: false,
        }
    }

    /// Prepare for processing.
    pub fn prepare(&mut self, sample_rate: f64, max_block_size: usize) {
        self.plugin.prepare(sample_rate, max_block_size);
        self.prepared = true;
    }

    /// Process audio in-place (stereo interleaved → stereo interleaved).
    ///
    /// `left` and `right` are per-channel buffers that are modified in-place.
    /// If the plugin is bypassed, the buffers are left unchanged.
    ///
    /// # Errors
    ///
    /// Returns `PluginError::ProcessingError` if the plugin is not prepared.
    pub fn process_stereo(&mut self, left: &mut [f32], right: &mut [f32]) -> PluginResult<()> {
        if !self.prepared {
            return Err(PluginError::ProcessingError(
                "Plugin not prepared (call prepare() first)".to_string(),
            ));
        }
        if self.bypassed {
            return Ok(());
        }

        // Sync atomic parameters to the plugin before processing.
        for id in self.params.parameter_ids() {
            if let Some(v) = self.params.get(id) {
                self.plugin.set_parameter(id, v);
            }
        }

        let num_samples = left.len().min(right.len());

        let desc = self.plugin.descriptor();
        let num_out = desc.num_outputs as usize;

        // Prepare input and output buffers.
        let input_left: Vec<f32> = left[..num_samples].to_vec();
        let input_right: Vec<f32> = right[..num_samples].to_vec();

        let in_refs: Vec<&[f32]> = vec![&input_left, &input_right];
        let out_left = vec![0.0_f32; num_samples];
        let out_right = vec![0.0_f32; num_samples];

        let mut out_bufs: Vec<Vec<f32>> = (0..num_out.max(2))
            .map(|i| {
                if i == 0 {
                    out_left.clone()
                } else {
                    out_right.clone()
                }
            })
            .collect();

        {
            let mut out_refs: Vec<&mut [f32]> =
                out_bufs.iter_mut().map(|v| v.as_mut_slice()).collect();
            self.plugin.process(&in_refs, &mut out_refs, num_samples);
        }

        // Copy outputs back, applying wet/dry mix.
        let wet = self.wet_mix;
        let dry = 1.0 - wet;
        for i in 0..num_samples {
            let out_l = out_bufs.first().map_or(0.0, |b| b[i]);
            let out_r = out_bufs.get(1).map_or(0.0, |b| b[i]);
            left[i] = left[i] * dry + out_l * wet;
            right[i] = right[i] * dry + out_r * wet;
        }

        Ok(())
    }

    /// Get the plugin descriptor.
    #[must_use]
    pub fn descriptor(&self) -> PluginDescriptor {
        self.plugin.descriptor()
    }

    /// Get the latency in samples introduced by this plugin.
    #[must_use]
    pub fn latency_samples(&self) -> u32 {
        self.plugin.latency_samples()
    }

    /// Reset internal plugin state.
    pub fn reset(&mut self) {
        self.plugin.reset();
    }
}

// ---------------------------------------------------------------------------
// PluginHost
// ---------------------------------------------------------------------------

/// Plugin host that manages plugin instances for the mixer.
///
/// The host assigns unique instance IDs, handles lifecycle (prepare/reset),
/// and provides a chain-processing interface used by the effects pipeline.
pub struct PluginHost {
    /// Active plugin instances keyed by instance ID.
    instances: HashMap<u32, PluginInstance>,
    /// Next instance ID counter.
    next_id: u32,
    /// Current sample rate.
    sample_rate: f64,
    /// Current maximum block size.
    max_block_size: usize,
}

impl PluginHost {
    /// Create a new plugin host.
    #[must_use]
    pub fn new(sample_rate: f64, max_block_size: usize) -> Self {
        Self {
            instances: HashMap::new(),
            next_id: 1,
            sample_rate,
            max_block_size,
        }
    }

    /// Load a plugin into the host.
    ///
    /// The plugin is prepared immediately with the current sample rate and
    /// block size. Returns the assigned instance ID.
    pub fn load_plugin(&mut self, plugin: Box<dyn AudioPlugin>) -> u32 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1).max(1);

        let mut instance = PluginInstance::new(id, plugin);
        instance.prepare(self.sample_rate, self.max_block_size);

        self.instances.insert(id, instance);
        id
    }

    /// Unload a plugin instance by ID.
    ///
    /// Returns `true` if the instance was found and removed.
    pub fn unload_plugin(&mut self, instance_id: u32) -> bool {
        if let Some(mut inst) = self.instances.remove(&instance_id) {
            inst.plugin.release();
            true
        } else {
            false
        }
    }

    /// Get a reference to a plugin instance.
    ///
    /// # Errors
    ///
    /// Returns `PluginError::InstanceNotFound` if the instance does not exist.
    pub fn get_instance(&self, instance_id: u32) -> PluginResult<&PluginInstance> {
        self.instances
            .get(&instance_id)
            .ok_or(PluginError::InstanceNotFound(instance_id))
    }

    /// Get a mutable reference to a plugin instance.
    ///
    /// # Errors
    ///
    /// Returns `PluginError::InstanceNotFound` if the instance does not exist.
    pub fn get_instance_mut(&mut self, instance_id: u32) -> PluginResult<&mut PluginInstance> {
        self.instances
            .get_mut(&instance_id)
            .ok_or(PluginError::InstanceNotFound(instance_id))
    }

    /// Process a stereo buffer through a list of plugin instances in order.
    ///
    /// Instances are processed in the order given by `chain`. Bypassed
    /// instances and unknown IDs are silently skipped.
    ///
    /// # Errors
    ///
    /// Returns the first `PluginError` encountered during processing.
    pub fn process_chain(
        &mut self,
        chain: &[u32],
        left: &mut [f32],
        right: &mut [f32],
    ) -> PluginResult<()> {
        for &id in chain {
            if let Some(instance) = self.instances.get_mut(&id) {
                instance.process_stereo(left, right)?;
            }
        }
        Ok(())
    }

    /// Update the sample rate and block size.  All existing instances are
    /// re-prepared.
    pub fn set_processing_params(&mut self, sample_rate: f64, max_block_size: usize) {
        self.sample_rate = sample_rate;
        self.max_block_size = max_block_size;
        for instance in self.instances.values_mut() {
            instance.prepare(sample_rate, max_block_size);
        }
    }

    /// Reset all plugin instances.
    pub fn reset_all(&mut self) {
        for instance in self.instances.values_mut() {
            instance.reset();
        }
    }

    /// Get the number of loaded plugin instances.
    #[must_use]
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Get the total latency of a plugin chain (sum of per-instance latencies).
    #[must_use]
    pub fn chain_latency(&self, chain: &[u32]) -> u32 {
        chain
            .iter()
            .filter_map(|&id| self.instances.get(&id))
            .filter(|inst| !inst.bypassed)
            .map(|inst| inst.latency_samples())
            .fold(0_u32, |acc, lat| acc.saturating_add(lat))
    }

    /// Get the shared `ParameterStore` for a plugin instance so the UI thread
    /// can write parameters lock-free.
    ///
    /// # Errors
    ///
    /// Returns `PluginError::InstanceNotFound` if the instance does not exist.
    pub fn parameter_store(&self, instance_id: u32) -> PluginResult<Arc<ParameterStore>> {
        self.instances
            .get(&instance_id)
            .map(|inst| Arc::clone(&inst.params))
            .ok_or(PluginError::InstanceNotFound(instance_id))
    }
}

impl std::fmt::Debug for PluginHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PluginHost")
            .field("instance_count", &self.instances.len())
            .field("sample_rate", &self.sample_rate)
            .field("max_block_size", &self.max_block_size)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Simple pass-through plugin for tests.
    struct PassThroughPlugin {
        gain: f32,
    }

    impl PassThroughPlugin {
        fn new(gain: f32) -> Self {
            Self { gain }
        }
    }

    impl AudioPlugin for PassThroughPlugin {
        fn descriptor(&self) -> PluginDescriptor {
            PluginDescriptor {
                id: "com.test.passthrough".to_string(),
                name: "Pass-Through".to_string(),
                vendor: "Test".to_string(),
                version: "0.1.0".to_string(),
                category: PluginCategory::Gain,
                num_inputs: 2,
                num_outputs: 2,
                has_editor: false,
            }
        }

        fn prepare(&mut self, _sample_rate: f64, _max_block_size: usize) {}

        fn process(&mut self, inputs: &[&[f32]], outputs: &mut [&mut [f32]], num_samples: usize) {
            for (inp, out) in inputs.iter().zip(outputs.iter_mut()) {
                for i in 0..num_samples.min(inp.len()).min(out.len()) {
                    out[i] = inp[i] * self.gain;
                }
            }
        }

        fn reset(&mut self) {}

        fn parameter_count(&self) -> usize {
            1
        }

        fn parameter_info(&self, idx: usize) -> Option<ParameterInfo> {
            if idx == 0 {
                Some(ParameterInfo {
                    id: 0,
                    name: "Gain".to_string(),
                    min: 0.0,
                    max: 2.0,
                    default: 1.0,
                    unit: String::new(),
                    automatable: true,
                })
            } else {
                None
            }
        }

        fn get_parameter(&self, _id: u32) -> f32 {
            self.gain
        }

        fn set_parameter(&mut self, id: u32, value: f32) {
            if id == 0 {
                self.gain = value;
            }
        }

        fn latency_samples(&self) -> u32 {
            0
        }
    }

    #[test]
    fn test_plugin_host_load_unload() {
        let mut host = PluginHost::new(48000.0, 512);
        let id = host.load_plugin(Box::new(PassThroughPlugin::new(1.0)));
        assert_eq!(host.instance_count(), 1);
        assert!(host.unload_plugin(id));
        assert_eq!(host.instance_count(), 0);
    }

    #[test]
    fn test_plugin_process_stereo() {
        let mut host = PluginHost::new(48000.0, 512);
        let id = host.load_plugin(Box::new(PassThroughPlugin::new(0.5)));

        let mut left = vec![1.0_f32; 64];
        let mut right = vec![1.0_f32; 64];

        host.process_chain(&[id], &mut left, &mut right)
            .expect("process_chain should succeed");

        for &v in &left {
            assert!((v - 0.5).abs() < 1e-5, "gain=0.5 should halve signal");
        }
    }

    #[test]
    fn test_plugin_bypass() {
        let mut host = PluginHost::new(48000.0, 512);
        let id = host.load_plugin(Box::new(PassThroughPlugin::new(0.0)));

        // Bypass the plugin — should pass through unchanged.
        host.get_instance_mut(id)
            .expect("instance should exist")
            .bypassed = true;

        let mut left = vec![1.0_f32; 32];
        let mut right = vec![1.0_f32; 32];

        host.process_chain(&[id], &mut left, &mut right)
            .expect("chain should succeed");

        for &v in &left {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "bypassed plugin should pass through"
            );
        }
    }

    #[test]
    fn test_plugin_wet_dry_mix() {
        let mut host = PluginHost::new(48000.0, 512);
        let id = host.load_plugin(Box::new(PassThroughPlugin::new(0.0)));

        // 50% wet/dry: 0.5 * input + 0.5 * 0.0 = 0.5
        host.get_instance_mut(id)
            .expect("instance should exist")
            .wet_mix = 0.5;

        let mut left = vec![1.0_f32; 32];
        let mut right = vec![1.0_f32; 32];

        host.process_chain(&[id], &mut left, &mut right)
            .expect("chain should succeed");

        for &v in &left {
            assert!(
                (v - 0.5).abs() < 1e-5,
                "50% wet/dry should produce 0.5, got {v}"
            );
        }
    }

    #[test]
    fn test_parameter_store_set_get() {
        let plugin = PassThroughPlugin::new(1.0);
        let store = ParameterStore::from_plugin(&plugin);
        store.set(0, 0.75);
        let v = store.get(0).expect("should have param 0");
        assert!((v - 0.75).abs() < 1e-5);
    }

    #[test]
    fn test_parameter_store_clamping() {
        let plugin = PassThroughPlugin::new(1.0);
        let store = ParameterStore::from_plugin(&plugin);
        store.set(0, 10.0); // Max is 2.0
        let v = store.get(0).expect("should have param 0");
        assert!(v <= 2.0, "value should be clamped to max");
    }

    #[test]
    fn test_parameter_store_unknown_id() {
        let plugin = PassThroughPlugin::new(1.0);
        let store = ParameterStore::from_plugin(&plugin);
        assert!(!store.set(99, 0.5));
        assert!(store.get(99).is_none());
    }

    #[test]
    fn test_plugin_host_chain_latency() {
        let mut host = PluginHost::new(48000.0, 512);
        let id1 = host.load_plugin(Box::new(PassThroughPlugin::new(1.0)));
        let id2 = host.load_plugin(Box::new(PassThroughPlugin::new(1.0)));
        // Both have 0 latency; chain latency should be 0.
        assert_eq!(host.chain_latency(&[id1, id2]), 0);
    }

    #[test]
    fn test_plugin_host_get_instance_not_found() {
        let host = PluginHost::new(48000.0, 512);
        assert!(host.get_instance(999).is_err());
    }

    #[test]
    fn test_plugin_host_parameter_store_shared() {
        let mut host = PluginHost::new(48000.0, 512);
        let id = host.load_plugin(Box::new(PassThroughPlugin::new(1.0)));

        let store = host.parameter_store(id).expect("store should exist");
        store.set(0, 0.25);

        // The value in the store should be 0.25.
        let v = store.get(0).expect("should have param");
        assert!((v - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_plugin_host_reset_all() {
        let mut host = PluginHost::new(48000.0, 512);
        host.load_plugin(Box::new(PassThroughPlugin::new(1.0)));
        // Should not panic.
        host.reset_all();
    }

    #[test]
    fn test_plugin_descriptor_fields() {
        let plugin = PassThroughPlugin::new(1.0);
        let desc = plugin.descriptor();
        assert_eq!(desc.id, "com.test.passthrough");
        assert_eq!(desc.name, "Pass-Through");
        assert_eq!(desc.num_inputs, 2);
        assert_eq!(desc.num_outputs, 2);
        assert!(!desc.has_editor);
    }

    #[test]
    fn test_plugin_category_display() {
        assert_eq!(PluginCategory::Dynamics.to_string(), "Dynamics");
        assert_eq!(PluginCategory::Equalizer.to_string(), "Equalizer");
    }
}
