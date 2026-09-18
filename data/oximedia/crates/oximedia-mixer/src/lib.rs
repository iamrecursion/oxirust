//! Professional audio mixer with automation for `OxiMedia`.
//!
//! This crate provides a complete digital audio mixing console with:
//!
//! - **Multi-channel mixing** - Support for 100+ channels with flexible routing
//! - **Channel types** - Mono, Stereo, 5.1, 7.1, and Ambisonics
//! - **Effect processing** - Dynamics, EQ, reverb, delay, modulation, distortion
//! - **Automation system** - Full parameter automation with multiple modes
//! - **Bus architecture** - Master, group, and auxiliary buses
//! - **Professional metering** - Peak, RMS, VU, LUFS, phase correlation
//! - **Session management** - Save/load mixer state with undo/redo
//!
//! # Architecture
//!
//! The mixer follows a professional DAW-style architecture:
//!
//! ```text
//! Input → Channel → Effects → Fader → Pan → Sends → Bus → Master Out
//! ```
//!
//! ## Channels
//!
//! Each channel provides:
//! - Input gain and phase inversion
//! - Insert effect chain (up to 8 slots)
//! - Channel fader with gain control
//! - Pan control (stereo, surround, binaural)
//! - Solo/Mute/Arm states
//! - Pre/post-fader sends to buses
//! - Direct monitoring output
//! - Channel linking for stereo pairs
//!
//! ## Buses
//!
//! Multiple bus types:
//! - **Master Bus** - Final stereo mixdown output
//! - **Group Buses** - Submix multiple channels together
//! - **Auxiliary Buses** - Effect sends/returns (reverb, delay, etc.)
//! - **Matrix Buses** - Advanced routing and monitoring
//!
//! ## Automation
//!
//! Full parameter automation with:
//! - **Read Mode** - Play back recorded automation
//! - **Write Mode** - Record all parameter changes
//! - **Touch Mode** - Record only when touching controls
//! - **Latch Mode** - Continue last value after release
//! - **Trim Mode** - Apply relative changes to existing automation
//!
//! ## Effects
//!
//! Professional effect categories:
//! - **Dynamics** - Compressor, Limiter, Gate, Expander, De-esser
//! - **EQ** - Parametric, Graphic, Shelving, High/Low Pass
//! - **Time-based** - Reverb, Delay, Echo, Chorus, Flanger
//! - **Modulation** - Phaser, Vibrato, Tremolo, Ring Modulator
//! - **Distortion** - Saturation, Overdrive, Bit Crusher, Wave Shaper
//!
//! ## Metering
//!
//! Professional-grade metering:
//! - **Peak Meters** - Sample-accurate peak detection
//! - **RMS Meters** - Average level measurement
//! - **VU Meters** - IEC 60268-10 standard (300ms ballistics)
//! - **LUFS Meters** - EBU R128 loudness metering
//! - **Phase Correlation** - Stereo compatibility checking
//! - **Spectrum Analyzer** - Real-time frequency analysis
//!
//! # Real-time Performance
//!
//! The mixer is optimized for low-latency operation:
//! - Lock-free audio processing path
//! - SIMD optimizations for DSP
//! - Memory-efficient buffer management
//! - Zero-copy audio routing where possible
//! - Target latency: <10ms for 48kHz/512 samples
//!
//! # Example
//!
//! ```rust
//! use oximedia_mixer::{AudioMixer, MixerConfig, ChannelType, MixerResult};
//! use oximedia_audio::ChannelLayout;
//!
//! fn example() -> MixerResult<()> {
//!     // Create mixer with default configuration
//!     let config = MixerConfig {
//!         sample_rate: 48000,
//!         buffer_size: 512,
//!         max_channels: 64,
//!         ..Default::default()
//!     };
//!
//!     let mut mixer = AudioMixer::new(config);
//!
//!     // Add a stereo channel
//!     let channel_id = mixer.add_channel(
//!         "Vocals".to_string(),
//!         ChannelType::Stereo,
//!         ChannelLayout::Stereo,
//!     )?;
//!
//!     // Set channel gain (0.0 = -inf dB, 1.0 = 0 dB)
//!     mixer.set_channel_gain(channel_id, 0.8)?;
//!
//!     // Pan center
//!     mixer.set_channel_pan(channel_id, 0.0)?;
//!
//!     // Process audio
//!     // let output = mixer.process(&input_frame)?;
//!     Ok(())
//! }
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod ambisonics;
pub mod analysis_meter;
pub mod atomic_param;
pub mod automation;
pub mod automation_lane;
pub mod automation_player;
pub mod aux_send;
pub mod bus;
pub mod channel;
pub mod channel_strip;
pub mod crossfade;
pub mod delay_line;
pub mod dynamics;
pub mod effects;
pub mod effects_chain;
pub mod eq_band;
pub mod group_bus;
pub mod insert_chain;
pub mod limiter;
pub mod matrix_mixer;
pub mod meter_bridge;
pub mod metering;
pub mod midi_control;
pub mod mix_bus;
pub mod monitor_mix;
pub mod oversampled_limiter;
pub mod pan_matrix;
pub mod routing;
pub mod scene_recall;
pub mod send_return;
pub mod session;
pub mod sidechain;
/// SIMD-accelerated audio bus summing and gain application.
#[allow(unsafe_code)]
pub mod simd_audio;
pub mod snapshot;
pub mod solo_bus;
pub mod solo_mode;
pub mod surround_panner;
pub mod vca;

/// DSP processing pipeline with bus routing, effects, sends, VCA, and PDC.
pub mod processing;

/// Parallel channel mixing using rayon for high channel counts.
pub mod parallel_mix;

/// DAW-style automation lanes with sample-accurate timing and curve interpolation.
pub mod daw_automation;

/// Sample-accurate automation playback engine with breakpoint interpolation.
pub mod automation_engine;

/// Time-based automation recording with Write/Touch/Latch/Trim modes.
pub mod automation_playback;

/// Offline bounce/render engine: processes mixer graph faster-than-realtime.
pub mod bounce;

/// Lock-free audio buffer pool for reusing `Vec<f32>` allocations.
pub mod buffer_pool;

/// Channel fold/unfold: mono↔stereo↔5.1↔7.1 down/up-mix with ITU-R BS.775.
pub mod channel_fold;

/// Channel folder: stereo↔mono, mid/side encoding for use in the processing chain.
pub mod channel_folder;

/// Pre-allocated channel output buffer slab: zero heap allocation on audio thread.
pub mod channel_prealloc;

/// Look-ahead brickwall limiter and clip detector for each channel/bus.
pub mod clip_guard;

/// PFL/AFL cue bus, cue mix level, and headphone output routing.
pub mod cue_monitor;

/// Fader grouping and linking (absolute, relative, VCA-style).
pub mod fader_group;

/// Sample-accurate gain and pan automation recording (Write/Touch/Latch/Trim).
pub mod gain_automation;

/// Gain staging, dB↔linear conversion, dynamic-range gain computer, and loudness normalizer.
pub mod gain_computer;

/// Mix scene management: capture, recall, diff, and timed transitions.
pub mod mix_scene;

/// Offline bounce/render engine with trait-based AudioSource.
pub mod offline_bounce;

/// Lock-free parameter smoothing (linear ramp, exponential IIR, shared atomic param).
pub mod param_smoother;

/// Plugin hosting API: VST3-style pure-Rust plugin interface and PluginHost.
pub mod plugin;

/// SIP/AFL/PFL solo routing with gain multipliers and monitor bus generation.
pub mod solo_modes;

/// Real-time rolling-window spectrum analyzer (pure-Rust FFT, log-bin grouping).
pub mod spectrum_analyzer;

/// Stem mixing: named submix groups with level/mute/solo and bus routing.
pub mod stem_mixer;

/// 5.1/7.1 surround panning with VBAP-style gain computation and LFE crossover.
pub mod surround_pan;

/// Talkback and communication system for studio monitoring with dim control.
pub mod talkback;

/// VCA group manager: additive-dB master trim over linked channels with automation.
pub mod vca_group;

use std::collections::HashMap;

use oximedia_audio::{AudioFrame, ChannelLayout};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub use analysis_meter::{AnalysisMeter, KWeightingFilter, MeterReadings};
pub use atomic_param::{AtomicF32, AtomicParam, SmoothedParam};
pub use automation::{
    AutomationCurve, AutomationData, AutomationMode, AutomationParameter, AutomationPoint,
};
pub use automation_player::{AutomatedParam, AutomationPlayer};
pub use bus::{Bus, BusConfig, BusId, BusType};
pub use channel::{Channel, ChannelId, ChannelType, PanMode};
pub use daw_automation::{
    AutomationLane as DawAutomationLane, AutomationParam, AutomationPoint as DawAutomationPoint,
    CurveType,
};
pub use effects::{Effect, EffectCategory, EffectId, EffectSlot};
pub use metering::{Meter, MeterType, MeteringData};
pub use midi_control::{
    MidiAction, MidiCcEvent, MidiControlConfig, MidiControlSurface, MidiMapping, MidiMappingTarget,
};
pub use parallel_mix::{
    mix_parallel, ParallelChannelInput, ParallelChannelOutput, ParallelMixConfig,
};
pub use processing::{
    ChannelOutputTarget, ChannelProcessParams, PanLawType, ProcessAuxSend, ProcessingEngine,
    RuntimeEffectSlot, RuntimeEffectsChain, VcaGroupState,
};
pub use session::{MixerSession, SessionData};
pub use solo_bus::{SoloBus, SoloBusConfig, SoloMode};
pub use surround_panner::{
    SurroundFormat, SurroundLayout, SurroundPanPosition, SurroundPanner, SURROUND_51_SPEAKERS,
    SURROUND_71_SPEAKERS,
};

/// Audio mixer error types.
#[derive(Debug, thiserror::Error)]
pub enum MixerError {
    /// Channel not found.
    #[error("Channel not found: {0}")]
    ChannelNotFound(ChannelId),

    /// Bus not found.
    #[error("Bus not found: {0}")]
    BusNotFound(BusId),

    /// Effect not found.
    #[error("Effect not found: {0}")]
    EffectNotFound(EffectId),

    /// Invalid parameter value.
    #[error("Invalid parameter value: {0}")]
    InvalidParameter(String),

    /// Maximum channels exceeded.
    #[error("Maximum channels exceeded: {0}")]
    MaxChannelsExceeded(usize),

    /// Audio processing error.
    #[error("Audio processing error: {0}")]
    ProcessingError(String),

    /// Session error.
    #[error("Session error: {0}")]
    SessionError(String),
}

/// Result type for mixer operations.
pub type MixerResult<T> = Result<T, MixerError>;

/// Mixer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixerConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,

    /// Buffer size in samples.
    pub buffer_size: usize,

    /// Maximum number of channels.
    pub max_channels: usize,

    /// Maximum number of buses.
    pub max_buses: usize,

    /// Maximum number of effects per channel.
    pub max_effects_per_channel: usize,

    /// Enable automation.
    pub enable_automation: bool,

    /// Enable metering.
    pub enable_metering: bool,

    /// Metering update rate in Hz.
    pub metering_rate: u32,
}

impl Default for MixerConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            buffer_size: 512,
            max_channels: 128,
            max_buses: 32,
            max_effects_per_channel: 8,
            enable_automation: true,
            enable_metering: true,
            metering_rate: 30,
        }
    }
}

/// Professional audio mixer.
pub struct AudioMixer {
    config: MixerConfig,
    channels: HashMap<ChannelId, Channel>,
    /// Maintains stable insertion-order index mapping for solo operations.
    channel_order: Vec<ChannelId>,
    buses: HashMap<BusId, Bus>,
    master_bus: Bus,
    session: MixerSession,
    sample_count: u64,
    /// Runtime processing engine for bus routing, effects, sends, VCA, and PDC.
    engine: ProcessingEngine,
    /// Solo bus for SIP, AFL, and PFL solo management.
    solo_bus: SoloBus,
    /// Automation player: renders per-block parameter values from automation lanes.
    automation_player: AutomationPlayer,
    /// Left-channel oversampled lookahead limiter for the master bus.
    master_limiter_l: oversampled_limiter::OversampledLimiter,
    /// Right-channel oversampled lookahead limiter for the master bus.
    master_limiter_r: oversampled_limiter::OversampledLimiter,
    /// Whether the master lookahead limiter is active.
    ///
    /// When `false` (default) the legacy `soft_clip()` is used to match the
    /// previous behaviour.  Enable via [`AudioMixer::set_limiter_enabled`].
    limiter_enabled: bool,
    /// Pool of pre-allocated `f32` buffers sized to `config.buffer_size`.
    ///
    /// Used by `extract_f32_samples` to avoid a fresh `Vec` allocation on
    /// every mix cycle.  The RAII [`PooledBuffer`] guard ensures each buffer
    /// is returned to the pool after the processing pass.
    sample_pool: buffer_pool::AudioBufferPool,
}

impl std::fmt::Debug for AudioMixer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioMixer")
            .field("config", &self.config)
            .field("channels", &self.channels)
            .field("buses", &self.buses)
            .field("master_bus", &self.master_bus)
            .field("sample_count", &self.sample_count)
            .finish_non_exhaustive()
    }
}

impl AudioMixer {
    /// Create a new audio mixer.
    #[must_use]
    pub fn new(config: MixerConfig) -> Self {
        let master_bus = Bus::new(
            "Master".to_string(),
            BusType::Master,
            ChannelLayout::Stereo,
            config.sample_rate,
            config.buffer_size,
        );

        let engine = ProcessingEngine::new(config.buffer_size);

        // Build a stereo pair of oversampled lookahead limiters (default: −0.3 dBFS,
        // 50 ms release, 4× oversampling).  They are only exercised when
        // `limiter_enabled` is true.
        #[allow(clippy::cast_precision_loss)]
        let sample_rate_f32 = config.sample_rate as f32;
        let master_limiter_l =
            oversampled_limiter::OversampledLimiter::new(-0.3, 50.0, 4, sample_rate_f32);
        let master_limiter_r =
            oversampled_limiter::OversampledLimiter::new(-0.3, 50.0, 4, sample_rate_f32);

        // Pre-warm 4 buffers so the first few `process()` calls never need to
        // allocate.  Each buffer holds exactly `buffer_size` f32 samples.
        let sample_pool = buffer_pool::AudioBufferPool::new(config.buffer_size, 4);

        Self {
            config,
            channels: HashMap::new(),
            channel_order: Vec::new(),
            buses: HashMap::new(),
            master_bus,
            session: MixerSession::new(),
            sample_count: 0,
            engine,
            solo_bus: SoloBus::default(),
            automation_player: AutomationPlayer::new(),
            master_limiter_l,
            master_limiter_r,
            limiter_enabled: false,
            sample_pool,
        }
    }

    /// Get mixer configuration.
    #[must_use]
    pub fn config(&self) -> &MixerConfig {
        &self.config
    }

    /// Get current sample count.
    #[must_use]
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Get current time in seconds.
    #[must_use]
    pub fn time_seconds(&self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        {
            self.sample_count as f64 / f64::from(self.config.sample_rate)
        }
    }

    /// Get a shared reference to the automation player.
    ///
    /// Use this to register automation lanes for channels.
    #[must_use]
    pub fn automation_player(&self) -> &AutomationPlayer {
        &self.automation_player
    }

    /// Get a mutable reference to the automation player.
    ///
    /// Use this to add, remove, or modify automation lanes.
    #[must_use]
    pub fn automation_player_mut(&mut self) -> &mut AutomationPlayer {
        &mut self.automation_player
    }

    /// Advance automation playback by `samples` samples and apply rendered
    /// parameter values to all channels with active lanes.
    ///
    /// This is called automatically at the start of every `process` call.
    /// The automation player renders values at the current `sample_count`
    /// position, then for each channel the first rendered sample (offset 0)
    /// is used as the block-level parameter value — providing sample-accurate
    /// automation alignment.  The `sample_count` playhead is advanced
    /// *after* this call returns (inside `process`).
    fn tick_automation(&mut self, samples: usize) {
        if !self.automation_player.enabled || !self.config.enable_automation {
            return;
        }

        // Render all registered lanes for this block starting at the current
        // sample_count position.
        self.automation_player
            .render_block(self.sample_count, samples);

        // Collect channel IDs up-front to avoid borrowing self.channels and
        // self.automation_player simultaneously.
        let channel_ids: Vec<ChannelId> = self.channels.keys().copied().collect();

        for id in channel_ids {
            // Sample offset 0 is the first (and representative) value for
            // this buffer.  Using offset 0 avoids per-sample overhead while
            // still providing buffer-accurate automation resolution.
            if let Some(gain) = self.automation_player.gain_at(id, 0) {
                if let Some(ch) = self.channels.get_mut(&id) {
                    ch.set_gain(gain);
                }
            }
            if let Some(pan) = self.automation_player.pan_at(id, 0) {
                if let Some(ch) = self.channels.get_mut(&id) {
                    ch.set_pan(pan);
                }
            }
        }
    }

    /// Add a new channel.
    ///
    /// # Errors
    ///
    /// Returns `MixerError::MaxChannelsExceeded` if the maximum number of channels is reached.
    pub fn add_channel(
        &mut self,
        name: String,
        channel_type: ChannelType,
        layout: ChannelLayout,
    ) -> MixerResult<ChannelId> {
        if self.channels.len() >= self.config.max_channels {
            return Err(MixerError::MaxChannelsExceeded(self.config.max_channels));
        }

        let id = ChannelId(Uuid::new_v4());
        let channel = Channel::new(
            name,
            channel_type,
            layout,
            self.config.sample_rate,
            self.config.buffer_size,
        );

        self.channels.insert(id, channel);
        self.channel_order.push(id);
        Ok(id)
    }

    /// Remove a channel.
    ///
    /// # Errors
    ///
    /// Returns `MixerError::ChannelNotFound` if the channel does not exist.
    pub fn remove_channel(&mut self, id: ChannelId) -> MixerResult<()> {
        self.channels
            .remove(&id)
            .ok_or(MixerError::ChannelNotFound(id))?;
        self.channel_order.retain(|&cid| cid != id);
        Ok(())
    }

    /// Get a channel.
    ///
    /// # Errors
    ///
    /// Returns `MixerError::ChannelNotFound` if the channel does not exist.
    pub fn get_channel(&self, id: ChannelId) -> MixerResult<&Channel> {
        self.channels
            .get(&id)
            .ok_or(MixerError::ChannelNotFound(id))
    }

    /// Get a mutable channel.
    ///
    /// # Errors
    ///
    /// Returns `MixerError::ChannelNotFound` if the channel does not exist.
    pub fn get_channel_mut(&mut self, id: ChannelId) -> MixerResult<&mut Channel> {
        self.channels
            .get_mut(&id)
            .ok_or(MixerError::ChannelNotFound(id))
    }

    /// Get all channels.
    #[must_use]
    pub fn channels(&self) -> &HashMap<ChannelId, Channel> {
        &self.channels
    }

    /// Set channel gain (0.0 = -inf dB, 1.0 = 0 dB).
    ///
    /// # Errors
    ///
    /// Returns `MixerError::ChannelNotFound` if the channel does not exist.
    pub fn set_channel_gain(&mut self, id: ChannelId, gain: f32) -> MixerResult<()> {
        let channel = self.get_channel_mut(id)?;
        channel.set_gain(gain);
        Ok(())
    }

    /// Set channel pan (-1.0 = full left, 0.0 = center, 1.0 = full right).
    ///
    /// # Errors
    ///
    /// Returns `MixerError::ChannelNotFound` if the channel does not exist.
    pub fn set_channel_pan(&mut self, id: ChannelId, pan: f32) -> MixerResult<()> {
        let channel = self.get_channel_mut(id)?;
        channel.set_pan(pan);
        Ok(())
    }

    /// Process audio for one buffer period.
    ///
    /// The full DSP pipeline:
    /// 0. Tick automation: render all active automation lanes and apply
    ///    gain/pan/mute parameter values to the affected channels
    /// 1. Extract f32 samples from the input frame's raw byte buffer
    /// 2. For each channel: input gain -> effects chain -> fader (× VCA) -> pan -> PDC
    /// 3. Route channel outputs to group/aux buses or master
    /// 4. Process aux sends (pre/post-fader) into aux buses
    /// 5. Apply bus effects chains and sum into master
    /// 6. Apply master bus soft clipping to prevent digital overs
    /// 7. Pack the result back into an `AudioFrame`
    ///
    /// # Errors
    ///
    /// Returns `MixerError::ProcessingError` if audio processing fails.
    pub fn process(&mut self, frame: &AudioFrame) -> MixerResult<AudioFrame> {
        let buffer_size = self.config.buffer_size;

        // Advance automation playback and apply rendered parameter values to
        // channels *before* building channel_params so that automated gain/pan
        // values are captured in this buffer's DSP pass.
        self.tick_automation(buffer_size);

        // Extract f32 samples from the raw byte data in the input frame.
        // `pooled_input` is an RAII guard; it is returned to `self.sample_pool`
        // automatically when it drops at the end of this function.
        let pooled_input = extract_f32_samples(frame, buffer_size, &self.sample_pool);

        // Build per-channel processing parameters from Channel state.
        // When any channel is soloed in SIP mode, non-soloed channels are muted.
        let channel_params: Vec<(ChannelId, ChannelProcessParams)> = self
            .channels
            .iter()
            .map(|(&id, ch)| {
                let pan_law = match ch.pan_law() {
                    channel::PanLaw::Linear => PanLawType::Linear,
                    channel::PanLaw::Minus3dB => PanLawType::Minus3dB,
                    channel::PanLaw::Minus4Dot5dB => PanLawType::Minus4Dot5dB,
                    channel::PanLaw::Minus6dB => PanLawType::Minus6dB,
                };
                // Resolve the channel's insertion-order index for solo gain lookup.
                let solo_gain = self
                    .channel_order
                    .iter()
                    .position(|&cid| cid == id)
                    .map(|idx| self.solo_bus.channel_gain(idx as u32))
                    .unwrap_or(1.0);
                // SIP muting: channel is muted if solo_gain is effectively zero.
                let muted_by_solo = solo_gain < f32::EPSILON;
                (
                    id,
                    ChannelProcessParams {
                        fader_gain: ch.gain(),
                        pan: ch.pan(),
                        muted: ch.is_muted() || muted_by_solo,
                        input_gain_db: ch.input().gain_db,
                        phase_inverted: ch.is_phase_inverted(),
                        pan_law,
                    },
                )
            })
            .collect();

        // Delegate to the processing engine.
        // `PooledBuffer` derefs to `&[f32]`, so no copy is required here.
        let (mut master_left, mut master_right) =
            self.engine.process_mix(&channel_params, &pooled_input);

        // Apply master bus limiting / soft clipping to prevent digital overs.
        if self.limiter_enabled {
            // Use the oversampled lookahead limiter for broadcast-quality peak control.
            // The left and right channels are processed independently to keep the hot
            // path allocation-free.  Stereo-linked gain is desirable in a mastering
            // context but is traded here for simplicity; use an external stereo-linked
            // compressor / limiter if image stability is critical.
            let mut limited_left = vec![0.0_f32; buffer_size];
            let mut limited_right = vec![0.0_f32; buffer_size];
            self.master_limiter_l
                .process_block(&master_left, &mut limited_left);
            self.master_limiter_r
                .process_block(&master_right, &mut limited_right);
            master_left = limited_left;
            master_right = limited_right;
        } else {
            // Legacy soft-clip path — kept for backward compatibility.
            for i in 0..buffer_size {
                master_left[i] = soft_clip(master_left[i]);
                master_right[i] = soft_clip(master_right[i]);
            }
        }

        self.sample_count += buffer_size as u64;

        // Create output frame with interleaved stereo packed as raw bytes.
        let mut output = AudioFrame::new(
            oximedia_core::SampleFormat::F32,
            self.config.sample_rate,
            ChannelLayout::Stereo,
        );

        let mut raw_bytes = Vec::with_capacity(buffer_size * 2 * 4);
        for i in 0..buffer_size {
            raw_bytes.extend_from_slice(&master_left[i].to_le_bytes());
            raw_bytes.extend_from_slice(&master_right[i].to_le_bytes());
        }
        output.samples = oximedia_audio::AudioBuffer::Interleaved(bytes::Bytes::from(raw_bytes));

        Ok(output)
    }

    /// Process audio using parallel channel DSP when channel count exceeds 8.
    ///
    /// This is a simplified fast path that processes the channel strip (input gain,
    /// phase inversion, fader, pan) in parallel using rayon.  Aux sends, bus
    /// routing, PDC, and VCA groups are not applied — use [`Self::process`] when
    /// those features are required.
    ///
    /// Returns interleaved stereo `Vec<f32>` of length `block_size * 2`.
    ///
    /// # Errors
    ///
    /// Returns `MixerError::ProcessingError` if `block_size` is zero.
    pub fn process_parallel(&mut self, block_size: usize) -> MixerResult<Vec<f32>> {
        if block_size == 0 {
            return Err(MixerError::ProcessingError("block_size must be > 0".into()));
        }

        self.tick_automation(block_size);

        let channels: Vec<parallel_mix::ParallelChannelInput> = self
            .channels
            .iter()
            .map(|(&_id, ch)| {
                let pan_law = match ch.pan_law() {
                    channel::PanLaw::Linear => PanLawType::Linear,
                    channel::PanLaw::Minus3dB => PanLawType::Minus3dB,
                    channel::PanLaw::Minus4Dot5dB => PanLawType::Minus4Dot5dB,
                    channel::PanLaw::Minus6dB => PanLawType::Minus6dB,
                };
                parallel_mix::ParallelChannelInput {
                    params: ChannelProcessParams {
                        fader_gain: ch.gain(),
                        pan: ch.pan(),
                        muted: ch.is_muted(),
                        input_gain_db: ch.input().gain_db,
                        phase_inverted: ch.is_phase_inverted(),
                        pan_law,
                    },
                    samples: vec![0.0_f32; block_size],
                }
            })
            .collect();

        let config = parallel_mix::ParallelMixConfig::default();
        Ok(parallel_mix::mix_parallel(&channels, block_size, &config))
    }

    /// Get a reference to the processing engine.
    #[must_use]
    pub fn engine(&self) -> &ProcessingEngine {
        &self.engine
    }

    /// Get a mutable reference to the processing engine.
    #[must_use]
    pub fn engine_mut(&mut self) -> &mut ProcessingEngine {
        &mut self.engine
    }

    /// Route a channel to a group bus.
    ///
    /// # Errors
    ///
    /// Returns `MixerError::ChannelNotFound` if the channel does not exist.
    /// Returns `MixerError::BusNotFound` if the bus does not exist.
    pub fn route_channel_to_bus(
        &mut self,
        channel_id: ChannelId,
        bus_id: BusId,
    ) -> MixerResult<()> {
        if !self.channels.contains_key(&channel_id) {
            return Err(MixerError::ChannelNotFound(channel_id));
        }
        if !self.buses.contains_key(&bus_id) {
            return Err(MixerError::BusNotFound(bus_id));
        }
        self.engine
            .channel_routing
            .insert(channel_id, ChannelOutputTarget::GroupBus(bus_id));
        Ok(())
    }

    /// Route a channel directly to the master bus (default routing).
    ///
    /// # Errors
    ///
    /// Returns `MixerError::ChannelNotFound` if the channel does not exist.
    pub fn route_channel_to_master(&mut self, channel_id: ChannelId) -> MixerResult<()> {
        if !self.channels.contains_key(&channel_id) {
            return Err(MixerError::ChannelNotFound(channel_id));
        }
        self.engine
            .channel_routing
            .insert(channel_id, ChannelOutputTarget::Master);
        Ok(())
    }

    /// Add an aux send from a channel to an aux bus.
    ///
    /// # Errors
    ///
    /// Returns `MixerError::ChannelNotFound` if the channel does not exist.
    /// Returns `MixerError::BusNotFound` if the bus does not exist.
    pub fn add_aux_send(
        &mut self,
        channel_id: ChannelId,
        bus_id: BusId,
        level: f32,
        pre_fader: bool,
    ) -> MixerResult<()> {
        if !self.channels.contains_key(&channel_id) {
            return Err(MixerError::ChannelNotFound(channel_id));
        }
        if !self.buses.contains_key(&bus_id) {
            return Err(MixerError::BusNotFound(bus_id));
        }
        let sends = self.engine.channel_sends.entry(channel_id).or_default();
        sends.push(ProcessAuxSend {
            bus_id,
            level: level.clamp(0.0, 1.0),
            pre_fader,
            active: true,
        });
        Ok(())
    }

    /// Add a channel effects chain slot.
    ///
    /// # Errors
    ///
    /// Returns `MixerError::ChannelNotFound` if the channel does not exist.
    pub fn add_channel_effect(
        &mut self,
        channel_id: ChannelId,
        slot: RuntimeEffectSlot,
    ) -> MixerResult<()> {
        if !self.channels.contains_key(&channel_id) {
            return Err(MixerError::ChannelNotFound(channel_id));
        }
        let chain = self
            .engine
            .channel_effects
            .entry(channel_id)
            .or_insert_with(RuntimeEffectsChain::new);
        chain.add(slot);
        Ok(())
    }

    /// Add a bus effects chain slot.
    ///
    /// # Errors
    ///
    /// Returns `MixerError::BusNotFound` if the bus does not exist.
    pub fn add_bus_effect(&mut self, bus_id: BusId, slot: RuntimeEffectSlot) -> MixerResult<()> {
        if !self.buses.contains_key(&bus_id) {
            return Err(MixerError::BusNotFound(bus_id));
        }
        let chain = self
            .engine
            .bus_effects
            .entry(bus_id)
            .or_insert_with(RuntimeEffectsChain::new);
        chain.add(slot);
        Ok(())
    }

    /// Add a VCA group.
    pub fn add_vca_group(&mut self, name: String, channels: Vec<ChannelId>) -> usize {
        let mut vca = VcaGroupState::new(name);
        vca.channels = channels;
        self.engine.vca_groups.push(vca);
        self.engine.vca_groups.len() - 1
    }

    /// Set VCA group gain.
    pub fn set_vca_gain(&mut self, group_index: usize, gain: f32) {
        if let Some(vca) = self.engine.vca_groups.get_mut(group_index) {
            vca.gain = gain.clamp(0.0, 2.0);
        }
    }

    /// Set VCA group mute.
    pub fn set_vca_muted(&mut self, group_index: usize, muted: bool) {
        if let Some(vca) = self.engine.vca_groups.get_mut(group_index) {
            vca.muted = muted;
        }
    }

    /// Recompute plugin delay compensation across all channels.
    pub fn recompute_pdc(&mut self) {
        self.engine.recompute_pdc();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Solo-In-Place / AFL / PFL solo management
    // ─────────────────────────────────────────────────────────────────────────

    /// Solo a channel by its insertion-order index using the specified mode.
    ///
    /// The `channel_id` parameter is a 0-based index into the ordered list of
    /// channels as added by `add_channel`.  Use `is_soloed` to query state.
    ///
    /// In [`SoloMode::Sip`] mode the solo bus will mute all non-soloed channels
    /// during `process`.  In [`SoloMode::Afl`] and [`SoloMode::Pfl`] modes
    /// the main mix is unaffected and the solo bus output gain is applied
    /// separately.
    ///
    /// # Errors
    ///
    /// Returns [`MixerError::InvalidParameter`] if `channel_id` is out of range.
    pub fn solo_channel(&mut self, channel_id: usize, mode: SoloMode) -> MixerResult<()> {
        if channel_id >= self.channel_order.len() {
            return Err(MixerError::InvalidParameter(format!(
                "solo channel index {channel_id} out of range (have {})",
                self.channel_order.len()
            )));
        }
        self.solo_bus.set_mode(mode);
        self.solo_bus.solo(channel_id as u32);
        Ok(())
    }

    /// Remove the solo state from a channel by its insertion-order index.
    ///
    /// # Errors
    ///
    /// Returns [`MixerError::InvalidParameter`] if `channel_id` is out of range.
    pub fn unsolo_channel(&mut self, channel_id: usize) -> MixerResult<()> {
        if channel_id >= self.channel_order.len() {
            return Err(MixerError::InvalidParameter(format!(
                "unsolo channel index {channel_id} out of range (have {})",
                self.channel_order.len()
            )));
        }
        self.solo_bus.unsolo(channel_id as u32);
        Ok(())
    }

    /// Returns `true` if the channel at the given insertion-order index is soloed.
    ///
    /// Returns `false` for any out-of-range index rather than panicking.
    #[must_use]
    pub fn is_soloed(&self, channel_id: usize) -> bool {
        if channel_id >= self.channel_order.len() {
            return false;
        }
        self.solo_bus.is_soloed(channel_id as u32)
    }

    /// Change the global solo mode without altering which channels are soloed.
    ///
    /// Switching to [`SoloMode::Sip`] will immediately begin muting non-soloed
    /// channels in `process` if any channel is currently soloed.
    pub fn set_solo_mode(&mut self, mode: SoloMode) {
        self.solo_bus.set_mode(mode);
    }

    /// Returns the current solo mode.
    #[must_use]
    pub fn solo_mode(&self) -> SoloMode {
        self.solo_bus.mode()
    }

    /// Returns `true` if any channel is currently soloed.
    #[must_use]
    pub fn any_channel_soloed(&self) -> bool {
        self.solo_bus.any_soloed()
    }

    /// Access the solo bus directly for advanced configuration (dim level, etc.).
    #[must_use]
    pub fn solo_bus(&self) -> &SoloBus {
        &self.solo_bus
    }

    /// Mutable access to the solo bus for advanced configuration.
    #[must_use]
    pub fn solo_bus_mut(&mut self) -> &mut SoloBus {
        &mut self.solo_bus
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Master bus limiter
    // ─────────────────────────────────────────────────────────────────────────

    /// Enable or disable the oversampled lookahead limiter on the master bus.
    ///
    /// When `enabled` is `true` the limiter replaces the legacy `soft_clip()`
    /// saturation.  When `false` (default) the legacy behaviour is preserved.
    pub fn set_limiter_enabled(&mut self, enabled: bool) {
        self.limiter_enabled = enabled;
    }

    /// Returns `true` if the master bus lookahead limiter is active.
    #[must_use]
    pub fn limiter_enabled(&self) -> bool {
        self.limiter_enabled
    }

    /// Immutable reference to the left-channel master limiter.
    #[must_use]
    pub fn master_limiter_l(&self) -> &oversampled_limiter::OversampledLimiter {
        &self.master_limiter_l
    }

    /// Mutable reference to the left-channel master limiter.
    ///
    /// Use this to adjust `threshold_db`, `release_ms`, and call
    /// [`oversampled_limiter::OversampledLimiter::recalculate`] after changing parameters.
    #[must_use]
    pub fn master_limiter_l_mut(&mut self) -> &mut oversampled_limiter::OversampledLimiter {
        &mut self.master_limiter_l
    }

    /// Immutable reference to the right-channel master limiter.
    #[must_use]
    pub fn master_limiter_r(&self) -> &oversampled_limiter::OversampledLimiter {
        &self.master_limiter_r
    }

    /// Mutable reference to the right-channel master limiter.
    #[must_use]
    pub fn master_limiter_r_mut(&mut self) -> &mut oversampled_limiter::OversampledLimiter {
        &mut self.master_limiter_r
    }

    /// Get mixer session.
    #[must_use]
    pub fn session(&self) -> &MixerSession {
        &self.session
    }

    /// Get mutable mixer session.
    #[must_use]
    pub fn session_mut(&mut self) -> &mut MixerSession {
        &mut self.session
    }

    /// Add a new bus.
    ///
    /// # Errors
    ///
    /// Returns `MixerError::MaxChannelsExceeded` if the maximum number of buses is reached.
    pub fn add_bus(
        &mut self,
        name: String,
        bus_type: BusType,
        layout: ChannelLayout,
    ) -> MixerResult<BusId> {
        if self.buses.len() >= self.config.max_buses {
            return Err(MixerError::MaxChannelsExceeded(self.config.max_buses));
        }

        let id = BusId(Uuid::new_v4());
        let bus = Bus::new(
            name,
            bus_type,
            layout,
            self.config.sample_rate,
            self.config.buffer_size,
        );

        // Register bus with the processing engine for accumulation.
        self.engine.register_bus(id, bus_type);

        self.buses.insert(id, bus);
        Ok(id)
    }

    /// Get a bus.
    ///
    /// # Errors
    ///
    /// Returns `MixerError::BusNotFound` if the bus does not exist.
    pub fn get_bus(&self, id: BusId) -> MixerResult<&Bus> {
        self.buses.get(&id).ok_or(MixerError::BusNotFound(id))
    }

    /// Get master bus.
    #[must_use]
    pub fn master_bus(&self) -> &Bus {
        &self.master_bus
    }

    /// Get mutable master bus.
    #[must_use]
    pub fn master_bus_mut(&mut self) -> &mut Bus {
        &mut self.master_bus
    }
}

/// Extract f32 samples from an `AudioFrame` into a pooled buffer.
///
/// Interprets the raw bytes in the frame as little-endian f32 values.
/// Returns a [`buffer_pool::PooledBuffer`] of exactly `max_samples` samples.
/// The buffer is automatically returned to `pool` when the guard is dropped.
///
/// Using the pool avoids a heap allocation on every mix cycle; the buffer is
/// zeroed by the pool before being handed out, so stale samples from a prior
/// cycle can never leak into the current one.
fn extract_f32_samples(
    frame: &AudioFrame,
    max_samples: usize,
    pool: &buffer_pool::AudioBufferPool,
) -> buffer_pool::PooledBuffer {
    // Borrow a pre-zeroed buffer from the pool.  If the pool's block_size
    // doesn't match `max_samples` (e.g. the config changed at runtime) an
    // overflow allocation is transparently returned by the pool.
    let mut buf = pool.checkout();

    let raw_bytes: &[u8] = match &frame.samples {
        oximedia_audio::AudioBuffer::Interleaved(data) => data.as_ref(),
        oximedia_audio::AudioBuffer::Planar(planes) => {
            if let Some(first) = planes.first() {
                first.as_ref()
            } else {
                // No input data — buffer is already zeroed by the pool; resize
                // to max_samples by zero-writing directly into the slice.
                let len = buf.len().min(max_samples);
                for s in buf[..len].iter_mut() {
                    *s = 0.0;
                }
                return buf;
            }
        }
    };

    // Each f32 sample occupies 4 bytes (little-endian IEEE 754).
    let num_f32_samples = raw_bytes.len() / 4;
    let count = num_f32_samples.min(max_samples).min(buf.len());

    for i in 0..count {
        let offset = i * 4;
        // SAFETY: `offset + 4 <= raw_bytes.len()` is guaranteed by the count
        // bound above (count <= num_f32_samples = raw_bytes.len() / 4).
        let bytes: [u8; 4] = [
            raw_bytes[offset],
            raw_bytes[offset + 1],
            raw_bytes[offset + 2],
            raw_bytes[offset + 3],
        ];
        buf[i] = f32::from_le_bytes(bytes);
    }
    // Samples beyond `count` up to `buf.len()` were zeroed by the pool on
    // checkout, so no explicit zero-fill is needed for the tail.

    buf
}

/// Soft clipping function using tanh-like saturation.
///
/// Maps input linearly near zero and smoothly saturates towards +/-1.0.
/// This prevents hard digital clipping artifacts.
fn soft_clip(x: f32) -> f32 {
    if x.abs() < 0.5 {
        x // Linear region for small signals
    } else if x > 0.0 {
        // Soft saturation for positive values
        let t = (x - 0.5) * 2.0;
        0.5 + 0.5 * (1.0 - (-t).exp()) / (1.0 + (-t).exp())
    } else {
        // Soft saturation for negative values
        let t = (-x - 0.5) * 2.0;
        -(0.5 + 0.5 * (1.0 - (-t).exp()) / (1.0 + (-t).exp()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixer_creation() {
        let config = MixerConfig::default();
        let mixer = AudioMixer::new(config);
        assert_eq!(mixer.channels().len(), 0);
    }

    #[test]
    fn test_add_channel() {
        let config = MixerConfig::default();
        let mut mixer = AudioMixer::new(config);

        let id = mixer
            .add_channel(
                "Test".to_string(),
                ChannelType::Stereo,
                ChannelLayout::Stereo,
            )
            .expect("test expectation failed");

        assert!(mixer.get_channel(id).is_ok());
    }

    #[test]
    fn test_channel_gain() {
        let config = MixerConfig::default();
        let mut mixer = AudioMixer::new(config);

        let id = mixer
            .add_channel(
                "Test".to_string(),
                ChannelType::Stereo,
                ChannelLayout::Stereo,
            )
            .expect("test expectation failed");

        mixer
            .set_channel_gain(id, 0.5)
            .expect("set_channel_gain should succeed");
        let channel = mixer.get_channel(id).expect("channel should be valid");
        assert!((channel.gain() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_max_channels() {
        let config = MixerConfig {
            max_channels: 2,
            ..Default::default()
        };
        let mut mixer = AudioMixer::new(config);

        mixer
            .add_channel(
                "Channel 1".to_string(),
                ChannelType::Stereo,
                ChannelLayout::Stereo,
            )
            .expect("test expectation failed");
        mixer
            .add_channel(
                "Channel 2".to_string(),
                ChannelType::Stereo,
                ChannelLayout::Stereo,
            )
            .expect("test expectation failed");

        let result = mixer.add_channel(
            "Channel 3".to_string(),
            ChannelType::Stereo,
            ChannelLayout::Stereo,
        );

        assert!(result.is_err());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Solo channel tests
    // ─────────────────────────────────────────────────────────────────────────

    fn make_3ch_mixer() -> (AudioMixer, usize, usize, usize) {
        let mut mixer = AudioMixer::new(MixerConfig::default());
        let _id0 = mixer
            .add_channel(
                "Ch0".to_string(),
                ChannelType::Stereo,
                ChannelLayout::Stereo,
            )
            .expect("add_channel failed");
        let _id1 = mixer
            .add_channel(
                "Ch1".to_string(),
                ChannelType::Stereo,
                ChannelLayout::Stereo,
            )
            .expect("add_channel failed");
        let _id2 = mixer
            .add_channel(
                "Ch2".to_string(),
                ChannelType::Stereo,
                ChannelLayout::Stereo,
            )
            .expect("add_channel failed");
        (mixer, 0, 1, 2)
    }

    #[test]
    fn test_solo_channel_sip_is_soloed() {
        let (mut mixer, idx0, idx1, _idx2) = make_3ch_mixer();
        mixer
            .solo_channel(idx0, SoloMode::Sip)
            .expect("solo_channel should succeed");
        assert!(mixer.is_soloed(idx0), "channel 0 should be soloed");
        assert!(!mixer.is_soloed(idx1), "channel 1 should not be soloed");
    }

    #[test]
    fn test_unsolo_channel_clears_solo() {
        let (mut mixer, idx0, _idx1, _idx2) = make_3ch_mixer();
        mixer
            .solo_channel(idx0, SoloMode::Sip)
            .expect("solo_channel should succeed");
        assert!(mixer.is_soloed(idx0));
        mixer
            .unsolo_channel(idx0)
            .expect("unsolo_channel should succeed");
        assert!(
            !mixer.is_soloed(idx0),
            "channel 0 should no longer be soloed"
        );
        assert!(!mixer.any_channel_soloed(), "no channels should be soloed");
    }

    #[test]
    fn test_is_soloed_out_of_range_returns_false() {
        let (mixer, _idx0, _idx1, _idx2) = make_3ch_mixer();
        assert!(
            !mixer.is_soloed(99),
            "out-of-range index should return false"
        );
    }

    #[test]
    fn test_solo_channel_out_of_range_is_err() {
        let (mut mixer, _idx0, _idx1, _idx2) = make_3ch_mixer();
        let result = mixer.solo_channel(99, SoloMode::Sip);
        assert!(result.is_err(), "solo with out-of-range index should error");
    }

    #[test]
    fn test_unsolo_channel_out_of_range_is_err() {
        let (mut mixer, _idx0, _idx1, _idx2) = make_3ch_mixer();
        let result = mixer.unsolo_channel(99);
        assert!(
            result.is_err(),
            "unsolo with out-of-range index should error"
        );
    }

    #[test]
    fn test_set_solo_mode_changes_mode() {
        let (mut mixer, _idx0, _idx1, _idx2) = make_3ch_mixer();
        mixer.set_solo_mode(SoloMode::Afl);
        assert_eq!(mixer.solo_mode(), SoloMode::Afl);
        mixer.set_solo_mode(SoloMode::Pfl);
        assert_eq!(mixer.solo_mode(), SoloMode::Pfl);
    }

    #[test]
    fn test_any_channel_soloed() {
        let (mut mixer, idx0, _idx1, _idx2) = make_3ch_mixer();
        assert!(!mixer.any_channel_soloed());
        mixer
            .solo_channel(idx0, SoloMode::Sip)
            .expect("solo_channel should succeed");
        assert!(mixer.any_channel_soloed());
        mixer
            .unsolo_channel(idx0)
            .expect("unsolo_channel should succeed");
        assert!(!mixer.any_channel_soloed());
    }

    #[test]
    fn test_solo_sip_mutes_non_soloed_in_process() {
        // Verify that SIP solo gain is computed correctly (channel_gain returns 0.0
        // for non-soloed channels).  We test through the solo_bus API directly.
        let (mut mixer, idx0, _idx1, _idx2) = make_3ch_mixer();
        mixer
            .solo_channel(idx0, SoloMode::Sip)
            .expect("solo_channel should succeed");
        // Soloed channel gets gain 1.0, others get 0.0 (sip_dim_level default = 0.0).
        assert!(
            (mixer.solo_bus().channel_gain(0) - 1.0).abs() < f32::EPSILON,
            "soloed channel should have gain 1.0"
        );
        assert!(
            mixer.solo_bus().channel_gain(1) < f32::EPSILON,
            "non-soloed channel should have gain 0.0 in SIP mode"
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // Automation playback tests
    // ─────────────────────────────────────────────────────────────────────

    /// Build a silent (all-zero) stereo `AudioFrame` for the given buffer_size.
    fn silent_frame(buffer_size: usize) -> AudioFrame {
        use oximedia_audio::AudioBuffer;
        use oximedia_core::SampleFormat;

        let byte_count = buffer_size * 2 * 4; // stereo × 4 bytes/sample
        let mut frame = AudioFrame::new(SampleFormat::F32, 48000, ChannelLayout::Stereo);
        frame.samples = AudioBuffer::Interleaved(bytes::Bytes::from(vec![0u8; byte_count]));
        frame
    }

    #[test]
    fn test_automation_player_accessible() {
        let mut mixer = AudioMixer::new(MixerConfig::default());
        let _ch = mixer
            .add_channel("A".to_string(), ChannelType::Stereo, ChannelLayout::Stereo)
            .expect("add_channel should succeed");
        assert_eq!(mixer.automation_player().lane_count(), 0);
        assert!(mixer.automation_player_mut().enabled);
    }

    #[test]
    fn test_automation_playback_sets_gain() {
        use crate::automation::{
            AutomationLane, AutomationMode, AutomationParameter, AutomationPoint,
        };
        use crate::automation_player::AutomatedParam;

        let config = MixerConfig {
            buffer_size: 512,
            sample_rate: 48000,
            enable_automation: true,
            ..Default::default()
        };
        let mut mixer = AudioMixer::new(config);
        let ch_id = mixer
            .add_channel(
                "Test".to_string(),
                ChannelType::Stereo,
                ChannelLayout::Stereo,
            )
            .expect("add_channel should succeed");

        // Register a gain lane: constant value 0.3 from sample 0 to 48000.
        let mut lane = AutomationLane::new(AutomationParameter::ChannelGain(ch_id), 1.0);
        lane.mode = AutomationMode::Read;
        lane.add_point(AutomationPoint::new(0, 0.3));
        lane.add_point(AutomationPoint::new(48000, 0.3));

        mixer
            .automation_player_mut()
            .add_lane(AutomatedParam::Gain(ch_id), lane);

        let frame = silent_frame(512);
        let _output = mixer.process(&frame).expect("process should succeed");

        // After process(), tick_automation should have set the channel gain to 0.3.
        let gain = mixer
            .get_channel(ch_id)
            .expect("channel should exist")
            .gain();
        assert!(
            (gain - 0.3).abs() < 0.01,
            "expected gain ~0.3 from automation, got {gain}"
        );
    }

    #[test]
    fn test_automation_playback_sets_pan() {
        use crate::automation::{
            AutomationLane, AutomationMode, AutomationParameter, AutomationPoint,
        };
        use crate::automation_player::AutomatedParam;

        let mut mixer = AudioMixer::new(MixerConfig::default());
        let ch_id = mixer
            .add_channel(
                "PanTest".to_string(),
                ChannelType::Stereo,
                ChannelLayout::Stereo,
            )
            .expect("add_channel should succeed");

        // Register a pan lane: constant -0.5 (hard-left-ish).
        let mut lane = AutomationLane::new(AutomationParameter::ChannelPan(ch_id), 0.0);
        lane.mode = AutomationMode::Read;
        lane.add_point(AutomationPoint::new(0, -0.5));
        lane.add_point(AutomationPoint::new(48000, -0.5));

        mixer
            .automation_player_mut()
            .add_lane(AutomatedParam::Pan(ch_id), lane);

        let frame = silent_frame(512);
        let _output = mixer.process(&frame).expect("process should succeed");

        let pan = mixer
            .get_channel(ch_id)
            .expect("channel should exist")
            .pan();
        assert!(
            (pan - (-0.5)).abs() < 0.01,
            "expected pan ~-0.5 from automation, got {pan}"
        );
    }

    #[test]
    fn test_tick_automation_advances_sample_count() {
        let config = MixerConfig {
            buffer_size: 256,
            ..Default::default()
        };
        let mut mixer = AudioMixer::new(config);
        assert_eq!(mixer.sample_count(), 0);

        let frame = silent_frame(256);
        mixer.process(&frame).expect("process should succeed");
        assert_eq!(
            mixer.sample_count(),
            256,
            "sample_count should advance by buffer_size"
        );

        mixer.process(&frame).expect("process should succeed");
        assert_eq!(
            mixer.sample_count(),
            512,
            "sample_count should advance again"
        );
    }

    #[test]
    fn test_automation_disabled_does_not_change_gain() {
        use crate::automation::{
            AutomationLane, AutomationMode, AutomationParameter, AutomationPoint,
        };
        use crate::automation_player::AutomatedParam;

        let config = MixerConfig {
            enable_automation: false,
            ..Default::default()
        };
        let mut mixer = AudioMixer::new(config);
        let ch_id = mixer
            .add_channel(
                "Disabled".to_string(),
                ChannelType::Stereo,
                ChannelLayout::Stereo,
            )
            .expect("add_channel should succeed");

        mixer
            .set_channel_gain(ch_id, 0.7)
            .expect("set_channel_gain should succeed");

        // Register automation that would change gain to 0.1
        let mut lane = AutomationLane::new(AutomationParameter::ChannelGain(ch_id), 1.0);
        lane.mode = AutomationMode::Read;
        lane.add_point(AutomationPoint::new(0, 0.1));

        mixer
            .automation_player_mut()
            .add_lane(AutomatedParam::Gain(ch_id), lane);

        let frame = silent_frame(512);
        mixer.process(&frame).expect("process should succeed");

        // Gain should remain 0.7 because enable_automation = false
        let gain = mixer
            .get_channel(ch_id)
            .expect("channel should exist")
            .gain();
        assert!(
            (gain - 0.7).abs() < 0.01,
            "automation should not have changed gain when disabled, got {gain}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // Master limiter wiring tests
    // ─────────────────────────────────────────────────────────────────────

    /// Build a stereo `AudioFrame` whose interleaved f32 samples are all `value`.
    fn constant_frame(buffer_size: usize, value: f32) -> AudioFrame {
        use oximedia_audio::AudioBuffer;
        use oximedia_core::SampleFormat;

        let mut frame = AudioFrame::new(SampleFormat::F32, 48000, ChannelLayout::Stereo);
        let mut raw = Vec::with_capacity(buffer_size * 2 * 4);
        for _ in 0..buffer_size * 2 {
            raw.extend_from_slice(&value.to_le_bytes());
        }
        frame.samples = AudioBuffer::Interleaved(bytes::Bytes::from(raw));
        frame
    }

    /// Decode the first `n` left-channel samples from a stereo interleaved output frame.
    fn decode_left_samples(frame: &AudioFrame, n: usize) -> Vec<f32> {
        let raw = match &frame.samples {
            oximedia_audio::AudioBuffer::Interleaved(b) => b.clone(),
            _ => return vec![],
        };
        let mut out = Vec::with_capacity(n);
        let mut i = 0;
        while i + 7 < raw.len() && out.len() < n {
            let bytes: [u8; 4] = [raw[i], raw[i + 1], raw[i + 2], raw[i + 3]];
            out.push(f32::from_le_bytes(bytes));
            i += 8; // skip right channel
        }
        out
    }

    #[test]
    fn test_limiter_enabled_accessor_roundtrip() {
        // Verify enable/disable toggles correctly.
        let mut mixer = AudioMixer::new(MixerConfig::default());
        assert!(
            !mixer.limiter_enabled(),
            "limiter should be disabled by default"
        );
        mixer.set_limiter_enabled(true);
        assert!(
            mixer.limiter_enabled(),
            "limiter should be enabled after set"
        );
        mixer.set_limiter_enabled(false);
        assert!(!mixer.limiter_enabled(), "limiter should be disabled again");
    }

    #[test]
    fn test_master_limiter_caps_output_when_enabled() {
        // Build a one-channel mixer with gain = 4.0 (well above 0 dBFS).
        // Feed a constant-1.0 mono signal through the full pipeline.
        // After the lookahead delay is primed the limiter must keep all
        // output samples at or below 1.0 (0 dBFS ceiling).
        let config = MixerConfig {
            buffer_size: 512,
            sample_rate: 48000,
            ..Default::default()
        };
        let mut mixer = AudioMixer::new(config);
        mixer.set_limiter_enabled(true);

        let ch = mixer
            .add_channel("Hot".to_string(), ChannelType::Mono, ChannelLayout::Mono)
            .expect("add_channel should succeed");
        // Drive the channel hard (+12 dB headroom: amplitude 4 × 0.9 = 3.6).
        mixer
            .set_channel_gain(ch, 4.0)
            .expect("set_channel_gain should succeed");

        let frame = constant_frame(512, 0.9);

        // Prime the limiter over several blocks so the gain envelope settles.
        let lookahead = mixer.master_limiter_l().lookahead_samples;
        let priming_blocks = (lookahead / 512).max(1) + 2;
        for _ in 0..priming_blocks {
            mixer.process(&frame).expect("process should succeed");
        }

        // Now check the settled output.
        let output = mixer.process(&frame).expect("process should succeed");
        let left = decode_left_samples(&output, 512);

        // Allow a tiny tolerance (1e-4) matching the existing oversampled-limiter tests.
        for (i, &s) in left.iter().enumerate() {
            assert!(
                s.abs() <= 1.0 + 1e-4,
                "sample[{i}] = {s:.6} exceeds 0 dBFS after master limiter"
            );
        }
    }

    #[test]
    fn test_master_limiter_disabled_soft_clips_legacy() {
        // When the limiter is disabled the legacy soft_clip path is used.
        // A silent (all-zero) input must produce a silent output regardless.
        let config = MixerConfig {
            buffer_size: 256,
            sample_rate: 48000,
            ..Default::default()
        };
        let mut mixer = AudioMixer::new(config);
        assert!(!mixer.limiter_enabled());

        let frame = silent_frame(256);
        let output = mixer.process(&frame).expect("process should succeed");
        let left = decode_left_samples(&output, 256);

        for (i, &s) in left.iter().enumerate() {
            assert!(
                s.abs() < f32::EPSILON,
                "sample[{i}] = {s} should be zero for silent input"
            );
        }
    }

    #[test]
    fn test_master_limiter_refs_accessible() {
        // Verify the accessor methods compile and return sensible initial state.
        let mut mixer = AudioMixer::new(MixerConfig::default());
        // Both limiters should start at unity gain (no signal processed yet).
        assert!(
            mixer.master_limiter_l().gain_reduction_db().abs() < 1e-5,
            "left limiter gain reduction should be 0 dB initially"
        );
        assert!(
            mixer.master_limiter_r().gain_reduction_db().abs() < 1e-5,
            "right limiter gain reduction should be 0 dB initially"
        );
        // Mutable refs should be obtainable.
        mixer.master_limiter_l_mut().reset();
        mixer.master_limiter_r_mut().reset();
    }

    // ─────────────────────────────────────────────────────────────────────
    // Buffer-pool integration tests
    // ─────────────────────────────────────────────────────────────────────

    /// Build a stereo `AudioFrame` with a known constant value in both channels.
    fn constant_stereo_frame(buffer_size: usize, value: f32) -> AudioFrame {
        use oximedia_audio::AudioBuffer;
        use oximedia_core::SampleFormat;

        let mut frame = AudioFrame::new(SampleFormat::F32, 48000, ChannelLayout::Stereo);
        let mut raw = Vec::with_capacity(buffer_size * 2 * 4);
        for _ in 0..buffer_size * 2 {
            raw.extend_from_slice(&value.to_le_bytes());
        }
        frame.samples = AudioBuffer::Interleaved(bytes::Bytes::from(raw));
        frame
    }

    #[test]
    fn test_pooled_extraction_identical_to_fresh() {
        // Calling process() twice with the same input frame should produce
        // byte-identical output frames.  If the pool were leaking stale samples
        // the second call would differ from the first.
        let config = MixerConfig {
            buffer_size: 128,
            sample_rate: 48000,
            ..Default::default()
        };
        let mut mixer = AudioMixer::new(config);
        let frame = constant_stereo_frame(128, 0.25);

        let out1 = mixer.process(&frame).expect("first process should succeed");
        let out2 = mixer
            .process(&frame)
            .expect("second process should succeed");

        let left1 = decode_left_samples(&out1, 128);
        let left2 = decode_left_samples(&out2, 128);

        assert_eq!(
            left1.len(),
            left2.len(),
            "both outputs should have the same length"
        );
        for (i, (a, b)) in left1.iter().zip(left2.iter()).enumerate() {
            assert!(
                (a - b).abs() < f32::EPSILON,
                "sample[{i}] differs between first ({a}) and second ({b}) call — pool may be corrupting data"
            );
        }
    }

    #[test]
    fn test_pool_reuse_no_stale_samples() {
        // Call process() with two different input levels and verify that the
        // second output reflects ONLY the second input (no stale data from the
        // first cycle leaking through the pool).
        let config = MixerConfig {
            buffer_size: 64,
            sample_rate: 48000,
            ..Default::default()
        };
        let mut mixer = AudioMixer::new(config);

        // First pass: non-zero input.
        let hot_frame = constant_stereo_frame(64, 0.8);
        mixer
            .process(&hot_frame)
            .expect("first process should succeed");

        // Second pass: silent input.  The pooled buffer is zeroed by the pool
        // on checkout, so all output samples must be zero (silence).
        let silent = silent_frame(64);
        let out = mixer
            .process(&silent)
            .expect("second process should succeed");
        let left = decode_left_samples(&out, 64);

        for (i, &s) in left.iter().enumerate() {
            assert!(
                s.abs() < f32::EPSILON,
                "sample[{i}] = {s} is non-zero — stale data from prior cycle leaked through pool"
            );
        }
    }

    #[test]
    fn test_soft_clip_never_exceeds_one() {
        // soft_clip must produce |output| <= 1.0 for all finite inputs and must
        // not panic on edge-case IEEE 754 values.
        let cases: &[f32] = &[
            0.0,
            0.5,
            1.0,
            2.0,
            100.0,
            f32::MAX,
            -0.5,
            -1.0,
            -2.0,
            -100.0,
            f32::NEG_INFINITY,
        ];
        for &x in cases {
            let y = soft_clip(x);
            assert!(y.abs() <= 1.0, "soft_clip({x}) = {y} exceeds ±1.0");
        }
        // NaN: soft_clip must not panic (output may be NaN — that is acceptable).
        let _ = soft_clip(f32::NAN);
    }

    #[test]
    fn test_dynamic_channel_add_remove() {
        // Create a mixer, process a block, add a channel, process again, remove
        // the channel, process a third time.  No panic should occur throughout
        // and the output should always have the correct (stereo) channel layout.
        let config = MixerConfig {
            buffer_size: 32,
            sample_rate: 48000,
            ..Default::default()
        };
        let mut mixer = AudioMixer::new(config);
        let frame = silent_frame(32);

        // Baseline: no channels.
        let out0 = mixer
            .process(&frame)
            .expect("process with no channels should succeed");
        assert!(
            matches!(out0.samples, oximedia_audio::AudioBuffer::Interleaved(_)),
            "output should be interleaved stereo"
        );

        // Add a channel, process.
        let ch_id = mixer
            .add_channel(
                "Dynamic".to_string(),
                ChannelType::Stereo,
                ChannelLayout::Stereo,
            )
            .expect("add_channel should succeed");
        let out1 = mixer
            .process(&frame)
            .expect("process after add should succeed");
        assert!(matches!(
            out1.samples,
            oximedia_audio::AudioBuffer::Interleaved(_)
        ));

        // Remove the channel, process.
        mixer
            .remove_channel(ch_id)
            .expect("remove_channel should succeed");
        let out2 = mixer
            .process(&frame)
            .expect("process after remove should succeed");
        assert!(matches!(
            out2.samples,
            oximedia_audio::AudioBuffer::Interleaved(_)
        ));

        // Channel count must be back to zero.
        assert_eq!(
            mixer.channels().len(),
            0,
            "all channels should have been removed"
        );
    }
}
