//! Dolby audio metering and metadata standards.
//!
//! This module implements Dolby-specific metering standards including:
//! - Dolby Dialogue Intelligence
//! - Dolby Metadata (dialnorm, DRC profiles)
//! - Dolby Loudness measurement (Leq(A), Leq(m))
//! - AC-3/E-AC-3/Atmos metadata generation
//!
//! # Dolby Dialogue Intelligence
//!
//! Sophisticated dialogue detection and gating for accurate dialnorm calculation:
//! - Speech vs. non-speech classification
//! - Dialogue gating algorithm
//! - Dialogue level measurement
//! - Dialnorm calculation for AC-3/E-AC-3
//!
//! # Dolby Metadata
//!
//! - dialnorm: -31 to 0 dBFS (dialogue normalization)
//! - DRC profiles: Film Standard, Film Light, Music Standard, Music Light, Speech
//! - compr and dynrng metadata
//! - LFE level metadata
//! - Room type metadata
//! - Mixing level metadata

#![forbid(unsafe_code)]

use crate::frame::AudioFrame;
use std::collections::VecDeque;

/// Dolby Dialogue Intelligence detector.
///
/// Detects dialogue (speech) segments in audio for accurate dialnorm calculation.
pub struct DialogueDetector {
    /// Sample rate in Hz.
    sample_rate: f64,
    /// Channels.
    channels: usize,
    /// Speech detector.
    speech_detector: SpeechDetector,
    /// Dialogue gate.
    dialogue_gate: DialogueGate,
    /// Dialogue level accumulator.
    dialogue_accumulator: f64,
    /// Dialogue sample count.
    dialogue_samples: usize,
}

/// Speech detector using spectral analysis.
struct SpeechDetector {
    /// Energy threshold for speech detection.
    energy_threshold: f64,
    /// Spectral flatness threshold.
    flatness_threshold: f64,
    /// Zero-crossing rate threshold.
    zcr_threshold: f64,
    /// Frame buffer for analysis.
    frame_buffer: VecDeque<f64>,
    /// Frame size (typically 20-30ms).
    frame_size: usize,
    /// Hop size.
    hop_size: usize,
    /// Sample accumulator.
    sample_count: usize,
}

impl SpeechDetector {
    fn new(sample_rate: f64) -> Self {
        let frame_size = (sample_rate * 0.025) as usize; // 25ms frames
        let hop_size = frame_size / 2;

        Self {
            energy_threshold: 0.001,
            flatness_threshold: 0.3,
            zcr_threshold: 0.15,
            frame_buffer: VecDeque::with_capacity(frame_size),
            frame_size,
            hop_size,
            sample_count: 0,
        }
    }

    fn process(&mut self, sample: f64) -> Option<bool> {
        self.frame_buffer.push_back(sample);
        self.sample_count += 1;

        if self.frame_buffer.len() > self.frame_size {
            self.frame_buffer.pop_front();
        }

        // Process every hop
        if self.sample_count >= self.hop_size && self.frame_buffer.len() == self.frame_size {
            self.sample_count = 0;
            let is_speech = self.detect_speech();
            return Some(is_speech);
        }

        None
    }

    fn detect_speech(&self) -> bool {
        // Energy calculation
        let energy: f64 = self
            .frame_buffer
            .iter()
            .map(|&s| s * s)
            .sum::<f64>()
            / self.frame_size as f64;

        if energy < self.energy_threshold {
            return false; // Too quiet to be speech
        }

        // Zero-crossing rate
        let mut zcr = 0;
        for i in 1..self.frame_buffer.len() {
            if self.frame_buffer[i - 1] * self.frame_buffer[i] < 0.0 {
                zcr += 1;
            }
        }
        let zcr_rate = zcr as f64 / self.frame_size as f64;

        // Spectral flatness (simplified)
        let flatness = self.calculate_spectral_flatness();

        // Speech detection logic
        energy > self.energy_threshold
            && zcr_rate > self.zcr_threshold
            && flatness < self.flatness_threshold
    }

    fn calculate_spectral_flatness(&self) -> f64 {
        // Simplified spectral flatness calculation
        // In a full implementation, would use FFT
        let samples: Vec<f64> = self.frame_buffer.iter().copied().collect();

        let geometric_mean = self.geometric_mean(&samples);
        let arithmetic_mean = self.arithmetic_mean(&samples);

        if arithmetic_mean > 0.0 {
            geometric_mean / arithmetic_mean
        } else {
            0.0
        }
    }

    fn geometric_mean(&self, values: &[f64]) -> f64 {
        let product: f64 = values.iter().map(|&x| x.abs() + 1e-10).product();
        product.powf(1.0 / values.len() as f64)
    }

    fn arithmetic_mean(&self, values: &[f64]) -> f64 {
        values.iter().map(|&x| x.abs()).sum::<f64>() / values.len() as f64
    }

    fn reset(&mut self) {
        self.frame_buffer.clear();
        self.sample_count = 0;
    }
}

/// Dialogue gating algorithm.
struct DialogueGate {
    /// Gate threshold (relative to max level).
    gate_threshold: f64,
    /// Maximum level seen.
    max_level: f64,
    /// Speech confidence history.
    confidence_history: VecDeque<bool>,
    /// History size.
    history_size: usize,
}

impl DialogueGate {
    fn new() -> Self {
        Self {
            gate_threshold: -10.0, // 10 dB below max
            max_level: 0.0,
            confidence_history: VecDeque::with_capacity(50),
            history_size: 50,
        }
    }

    fn is_dialogue(&mut self, level: f64, is_speech: bool) -> bool {
        self.max_level = self.max_level.max(level);

        // Add to confidence history
        self.confidence_history.push_back(is_speech);
        if self.confidence_history.len() > self.history_size {
            self.confidence_history.pop_front();
        }

        // Calculate confidence
        let confidence = self
            .confidence_history
            .iter()
            .filter(|&&x| x)
            .count() as f64
            / self.confidence_history.len() as f64;

        // Gate based on level and confidence
        let level_db = if level > 0.0 {
            20.0 * level.log10()
        } else {
            -100.0
        };
        let max_db = if self.max_level > 0.0 {
            20.0 * self.max_level.log10()
        } else {
            -100.0
        };

        level_db > max_db + self.gate_threshold && confidence > 0.6
    }

    fn reset(&mut self) {
        self.max_level = 0.0;
        self.confidence_history.clear();
    }
}

impl DialogueDetector {
    /// Create a new dialogue detector.
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        Self {
            sample_rate,
            channels,
            speech_detector: SpeechDetector::new(sample_rate),
            dialogue_gate: DialogueGate::new(),
            dialogue_accumulator: 0.0,
            dialogue_samples: 0,
        }
    }

    /// Process audio samples.
    pub fn process(&mut self, samples: &[f64]) {
        let frames = samples.len() / self.channels;

        for frame_idx in 0..frames {
            // Calculate frame energy (mono sum)
            let mut frame_energy = 0.0;
            for ch in 0..self.channels {
                let idx = frame_idx * self.channels + ch;
                if idx < samples.len() {
                    frame_energy += samples[idx].abs();
                }
            }
            frame_energy /= self.channels as f64;

            // Detect speech
            if let Some(is_speech) = self.speech_detector.process(frame_energy) {
                // Check dialogue gate
                if self.dialogue_gate.is_dialogue(frame_energy, is_speech) {
                    // Accumulate dialogue level
                    self.dialogue_accumulator += frame_energy * frame_energy;
                    self.dialogue_samples += 1;
                }
            }
        }
    }

    /// Get dialogue level in dBFS.
    pub fn dialogue_level(&self) -> f64 {
        if self.dialogue_samples > 0 {
            let rms = (self.dialogue_accumulator / self.dialogue_samples as f64).sqrt();
            if rms > 0.0 {
                20.0 * rms.log10()
            } else {
                -100.0
            }
        } else {
            -100.0
        }
    }

    /// Get dialogue percentage.
    pub fn dialogue_percentage(&self) -> f64 {
        if self.dialogue_gate.confidence_history.is_empty() {
            return 0.0;
        }

        self.dialogue_gate
            .confidence_history
            .iter()
            .filter(|&&x| x)
            .count() as f64
            / self.dialogue_gate.confidence_history.len() as f64
            * 100.0
    }

    /// Calculate dialnorm value for AC-3/E-AC-3.
    ///
    /// Dialnorm ranges from -31 to 0 dBFS, indicating the dialogue level.
    pub fn calculate_dialnorm(&self) -> i32 {
        let level = self.dialogue_level();

        // Clamp to valid range
        let clamped = level.max(-31.0).min(0.0);

        // Round to nearest integer
        clamped.round() as i32
    }

    /// Reset the detector.
    pub fn reset(&mut self) {
        self.speech_detector.reset();
        self.dialogue_gate.reset();
        self.dialogue_accumulator = 0.0;
        self.dialogue_samples = 0;
    }
}

/// Dolby Dynamic Range Control (DRC) profiles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DrcProfile {
    /// Film Standard - Full dynamic range.
    FilmStandard,
    /// Film Light - Moderate compression.
    FilmLight,
    /// Music Standard - Minimal compression.
    MusicStandard,
    /// Music Light - Light compression for music.
    MusicLight,
    /// Speech - Optimized for dialogue.
    Speech,
    /// None - No DRC applied.
    None,
}

impl DrcProfile {
    /// Get the compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        match self {
            Self::FilmStandard => 1.0,    // No compression
            Self::FilmLight => 2.0,       // 2:1 compression
            Self::MusicStandard => 1.5,   // 1.5:1 compression
            Self::MusicLight => 2.5,      // 2.5:1 compression
            Self::Speech => 3.0,          // 3:1 compression
            Self::None => 1.0,
        }
    }

    /// Get the compr byte value for AC-3.
    pub fn compr_value(&self) -> u8 {
        match self {
            Self::FilmStandard => 0x00,
            Self::FilmLight => 0x20,
            Self::MusicStandard => 0x10,
            Self::MusicLight => 0x30,
            Self::Speech => 0x40,
            Self::None => 0xFF,
        }
    }

    /// Get the dynrng byte value for AC-3.
    pub fn dynrng_value(&self) -> u8 {
        match self {
            Self::FilmStandard => 0x00,
            Self::FilmLight => 0x20,
            Self::MusicStandard => 0x10,
            Self::MusicLight => 0x30,
            Self::Speech => 0x40,
            Self::None => 0xFF,
        }
    }
}

/// Dolby metadata for AC-3/E-AC-3.
#[derive(Clone, Debug)]
pub struct DolbyMetadata {
    /// Dialogue normalization (-31 to 0 dBFS).
    pub dialnorm: i32,
    /// DRC profile.
    pub drc_profile: DrcProfile,
    /// LFE level (dB).
    pub lfe_level: f64,
    /// Room type.
    pub room_type: RoomType,
    /// Mixing level (dB SPL).
    pub mixing_level: f64,
    /// Copyright bit.
    pub copyright: bool,
    /// Original bitstream bit.
    pub original_bitstream: bool,
}

/// Room type for mixing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RoomType {
    /// Not indicated.
    NotIndicated,
    /// Large room (X-curve).
    LargeRoom,
    /// Small room (flat).
    SmallRoom,
}

impl RoomType {
    /// Get the room type code for AC-3.
    pub fn code(&self) -> u8 {
        match self {
            Self::NotIndicated => 0,
            Self::LargeRoom => 1,
            Self::SmallRoom => 2,
        }
    }
}

impl Default for DolbyMetadata {
    fn default() -> Self {
        Self {
            dialnorm: -27,
            drc_profile: DrcProfile::FilmStandard,
            lfe_level: 0.0,
            room_type: RoomType::NotIndicated,
            mixing_level: 85.0,
            copyright: false,
            original_bitstream: true,
        }
    }
}

impl DolbyMetadata {
    /// Create new Dolby metadata with dialnorm.
    pub fn new(dialnorm: i32) -> Self {
        let clamped_dialnorm = dialnorm.max(-31).min(0);
        Self {
            dialnorm: clamped_dialnorm,
            ..Default::default()
        }
    }

    /// Set DRC profile.
    pub fn with_drc_profile(mut self, profile: DrcProfile) -> Self {
        self.drc_profile = profile;
        self
    }

    /// Set LFE level.
    pub fn with_lfe_level(mut self, level: f64) -> Self {
        self.lfe_level = level;
        self
    }

    /// Set room type.
    pub fn with_room_type(mut self, room_type: RoomType) -> Self {
        self.room_type = room_type;
        self
    }

    /// Set mixing level.
    pub fn with_mixing_level(mut self, level: f64) -> Self {
        self.mixing_level = level;
        self
    }

    /// Validate metadata.
    pub fn validate(&self) -> Result<(), String> {
        if self.dialnorm < -31 || self.dialnorm > 0 {
            return Err(format!(
                "Invalid dialnorm: {} (must be -31 to 0)",
                self.dialnorm
            ));
        }

        if self.mixing_level < 80.0 || self.mixing_level > 111.0 {
            return Err(format!(
                "Invalid mixing level: {} (must be 80-111 dB SPL)",
                self.mixing_level
            ));
        }

        Ok(())
    }
}

/// Dolby Leq(A) meter - A-weighted SPL measurement.
///
/// Measures equivalent continuous sound level with A-weighting.
pub struct LeqAMeter {
    /// Sample rate.
    sample_rate: f64,
    /// Channels.
    channels: usize,
    /// A-weighting filter.
    a_weight: AWeightingFilter,
    /// Accumulator.
    accumulator: f64,
    /// Sample count.
    sample_count: usize,
}

/// A-weighting filter for SPL measurement.
struct AWeightingFilter {
    /// Filter coefficients.
    b: [f64; 7],
    a: [f64; 7],
    /// Filter state per channel.
    states: Vec<Vec<f64>>,
}

impl AWeightingFilter {
    fn new(sample_rate: f64, channels: usize) -> Self {
        // A-weighting filter coefficients (simplified for this implementation)
        // In practice, would use proper IIR filter design for A-weighting
        let b = [
            0.169994948147430,
            0.0,
            -0.509984844442290,
            0.0,
            0.509984844442290,
            0.0,
            -0.169994948147430,
        ];
        let a = [
            1.0,
            -2.12979364760736,
            0.42996125885751,
            1.62132698199721,
            -0.96669962900954,
            0.00121015844426,
            0.04400300696788,
        ];

        Self {
            b,
            a,
            states: vec![vec![0.0; 7]; channels],
        }
    }

    fn process(&mut self, sample: f64, channel: usize) -> f64 {
        let state = &mut self.states[channel];

        // Direct Form II Transposed implementation
        let mut y = self.b[0] * sample + state[0];
        for i in 0..6 {
            state[i] = self.b[i + 1] * sample - self.a[i + 1] * y + state[i + 1];
        }

        y
    }

    fn reset(&mut self) {
        for state in &mut self.states {
            state.fill(0.0);
        }
    }
}

impl LeqAMeter {
    /// Create a new Leq(A) meter.
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        Self {
            sample_rate,
            channels,
            a_weight: AWeightingFilter::new(sample_rate, channels),
            accumulator: 0.0,
            sample_count: 0,
        }
    }

    /// Process audio samples.
    pub fn process(&mut self, samples: &[f64]) {
        let frames = samples.len() / self.channels;

        for frame_idx in 0..frames {
            for ch in 0..self.channels {
                let idx = frame_idx * self.channels + ch;
                if idx < samples.len() {
                    let weighted = self.a_weight.process(samples[idx], ch);
                    self.accumulator += weighted * weighted;
                    self.sample_count += 1;
                }
            }
        }
    }

    /// Get Leq(A) in dB SPL.
    ///
    /// Assumes calibration reference of 94 dB SPL = 1.0 RMS.
    pub fn leq_a(&self) -> f64 {
        if self.sample_count > 0 {
            let rms = (self.accumulator / self.sample_count as f64).sqrt();
            // Convert to dB SPL (assuming 94 dB SPL reference)
            94.0 + 20.0 * rms.log10()
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Reset the meter.
    pub fn reset(&mut self) {
        self.a_weight.reset();
        self.accumulator = 0.0;
        self.sample_count = 0;
    }
}

/// Dolby Leq(m) meter - M-weighted measurement.
///
/// M-weighting is used for cinema and large-venue applications.
pub struct LeqMMeter {
    /// Sample rate.
    sample_rate: f64,
    /// Channels.
    channels: usize,
    /// M-weighting filter.
    m_weight: MWeightingFilter,
    /// Accumulator.
    accumulator: f64,
    /// Sample count.
    sample_count: usize,
}

/// M-weighting filter.
struct MWeightingFilter {
    /// High-pass filter state.
    hp_state: Vec<(f64, f64, f64, f64)>,
    /// Low-pass filter state.
    lp_state: Vec<(f64, f64, f64, f64)>,
    channels: usize,
}

impl MWeightingFilter {
    fn new(_sample_rate: f64, channels: usize) -> Self {
        Self {
            hp_state: vec![(0.0, 0.0, 0.0, 0.0); channels],
            lp_state: vec![(0.0, 0.0, 0.0, 0.0); channels],
            channels,
        }
    }

    fn process(&mut self, sample: f64, channel: usize) -> f64 {
        // Simplified M-weighting (high-pass + low-pass)
        // In practice, M-weighting is more complex
        let hp = self.high_pass(sample, channel);
        self.low_pass(hp, channel)
    }

    fn high_pass(&mut self, x: f64, ch: usize) -> f64 {
        let (x1, x2, y1, y2) = self.hp_state[ch];

        // Simple 2nd-order HP at ~20 Hz
        let b0 = 0.998;
        let b1 = -1.996;
        let b2 = 0.998;
        let a1 = -1.996;
        let a2 = 0.996;

        let y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        self.hp_state[ch] = (x, x1, y, y1);
        y
    }

    fn low_pass(&mut self, x: f64, ch: usize) -> f64 {
        let (x1, x2, y1, y2) = self.lp_state[ch];

        // Simple 2nd-order LP at ~10 kHz
        let b0 = 0.01;
        let b1 = 0.02;
        let b2 = 0.01;
        let a1 = -1.5;
        let a2 = 0.6;

        let y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        self.lp_state[ch] = (x, x1, y, y1);
        y
    }

    fn reset(&mut self) {
        self.hp_state.fill((0.0, 0.0, 0.0, 0.0));
        self.lp_state.fill((0.0, 0.0, 0.0, 0.0));
    }
}

impl LeqMMeter {
    /// Create a new Leq(m) meter.
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        Self {
            sample_rate,
            channels,
            m_weight: MWeightingFilter::new(sample_rate, channels),
            accumulator: 0.0,
            sample_count: 0,
        }
    }

    /// Process audio samples.
    pub fn process(&mut self, samples: &[f64]) {
        let frames = samples.len() / self.channels;

        for frame_idx in 0..frames {
            for ch in 0..self.channels {
                let idx = frame_idx * self.channels + ch;
                if idx < samples.len() {
                    let weighted = self.m_weight.process(samples[idx], ch);
                    self.accumulator += weighted * weighted;
                    self.sample_count += 1;
                }
            }
        }
    }

    /// Get Leq(m) in dB SPL.
    pub fn leq_m(&self) -> f64 {
        if self.sample_count > 0 {
            let rms = (self.accumulator / self.sample_count as f64).sqrt();
            85.0 + 20.0 * rms.log10() // Cinema reference: 85 dB SPL
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Reset the meter.
    pub fn reset(&mut self) {
        self.m_weight.reset();
        self.accumulator = 0.0;
        self.sample_count = 0;
    }
}

/// Unified Dolby meter.
///
/// Combines dialogue detection, metadata generation, and loudness measurement.
pub struct DolbyMeter {
    /// Sample rate.
    sample_rate: f64,
    /// Channels.
    channels: usize,
    /// Dialogue detector.
    dialogue_detector: DialogueDetector,
    /// Leq(A) meter.
    leq_a: LeqAMeter,
    /// Leq(m) meter.
    leq_m: LeqMMeter,
    /// Metadata.
    metadata: DolbyMetadata,
}

impl DolbyMeter {
    /// Create a new Dolby meter.
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        Self {
            sample_rate,
            channels,
            dialogue_detector: DialogueDetector::new(sample_rate, channels),
            leq_a: LeqAMeter::new(sample_rate, channels),
            leq_m: LeqMMeter::new(sample_rate, channels),
            metadata: DolbyMetadata::default(),
        }
    }

    /// Process audio frame.
    pub fn process(&mut self, frame: &AudioFrame) {
        let samples = extract_samples_f64(frame);
        self.dialogue_detector.process(&samples);
        self.leq_a.process(&samples);
        self.leq_m.process(&samples);
    }

    /// Get current Dolby metrics.
    pub fn get_metrics(&mut self) -> DolbyMetrics {
        // Update metadata with calculated dialnorm
        self.metadata.dialnorm = self.dialogue_detector.calculate_dialnorm();

        DolbyMetrics {
            dialogue_level: self.dialogue_detector.dialogue_level(),
            dialogue_percentage: self.dialogue_detector.dialogue_percentage(),
            dialnorm: self.metadata.dialnorm,
            leq_a: self.leq_a.leq_a(),
            leq_m: self.leq_m.leq_m(),
            metadata: self.metadata.clone(),
        }
    }

    /// Get Dolby metadata for encoding.
    pub fn get_metadata(&self) -> &DolbyMetadata {
        &self.metadata
    }

    /// Set DRC profile.
    pub fn set_drc_profile(&mut self, profile: DrcProfile) {
        self.metadata.drc_profile = profile;
    }

    /// Set room type.
    pub fn set_room_type(&mut self, room_type: RoomType) {
        self.metadata.room_type = room_type;
    }

    /// Set mixing level.
    pub fn set_mixing_level(&mut self, level: f64) {
        self.metadata.mixing_level = level;
    }

    /// Reset the meter.
    pub fn reset(&mut self) {
        self.dialogue_detector.reset();
        self.leq_a.reset();
        self.leq_m.reset();
    }
}

/// Dolby measurement metrics.
#[derive(Clone, Debug)]
pub struct DolbyMetrics {
    /// Dialogue level (dBFS).
    pub dialogue_level: f64,
    /// Dialogue percentage (0-100%).
    pub dialogue_percentage: f64,
    /// Calculated dialnorm (-31 to 0).
    pub dialnorm: i32,
    /// Leq(A) measurement (dB SPL).
    pub leq_a: f64,
    /// Leq(m) measurement (dB SPL).
    pub leq_m: f64,
    /// Dolby metadata.
    pub metadata: DolbyMetadata,
}

/// Dolby Atmos dialogue detection.
///
/// Enhanced dialogue detection for immersive audio.
pub struct AtmosDialogueDetector {
    /// Base dialogue detector.
    base_detector: DialogueDetector,
    /// Object-based audio channels.
    objects: Vec<ObjectChannel>,
}

/// Object channel for Atmos.
struct ObjectChannel {
    /// Object ID.
    id: usize,
    /// Position (x, y, z).
    position: (f64, f64, f64),
    /// Dialogue confidence.
    dialogue_confidence: f64,
}

impl AtmosDialogueDetector {
    /// Create a new Atmos dialogue detector.
    pub fn new(sample_rate: f64, channels: usize, num_objects: usize) -> Self {
        let mut objects = Vec::with_capacity(num_objects);
        for i in 0..num_objects {
            objects.push(ObjectChannel {
                id: i,
                position: (0.0, 0.0, 0.0),
                dialogue_confidence: 0.0,
            });
        }

        Self {
            base_detector: DialogueDetector::new(sample_rate, channels),
            objects,
        }
    }

    /// Process audio with object metadata.
    pub fn process(&mut self, samples: &[f64]) {
        self.base_detector.process(samples);

        // Update object dialogue confidence
        for obj in &mut self.objects {
            // Simplified: in practice would analyze per-object audio
            obj.dialogue_confidence = self.base_detector.dialogue_percentage() / 100.0;
        }
    }

    /// Get dialogue level for specific object.
    pub fn object_dialogue_level(&self, object_id: usize) -> Option<f64> {
        self.objects
            .get(object_id)
            .map(|obj| self.base_detector.dialogue_level() * obj.dialogue_confidence)
    }

    /// Get overall dialogue level.
    pub fn dialogue_level(&self) -> f64 {
        self.base_detector.dialogue_level()
    }

    /// Reset detector.
    pub fn reset(&mut self) {
        self.base_detector.reset();
        for obj in &mut self.objects {
            obj.dialogue_confidence = 0.0;
        }
    }
}

/// Extract samples as f64 from AudioFrame.
fn extract_samples_f64(frame: &AudioFrame) -> Vec<f64> {
    match &frame.samples {
        crate::frame::AudioBuffer::Interleaved(data) => {
            let sample_count = data.len() / 4;
            let mut samples = Vec::with_capacity(sample_count);

            for i in 0..sample_count {
                let offset = i * 4;
                if offset + 4 <= data.len() {
                    let bytes_array = [
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ];
                    let sample = f32::from_le_bytes(bytes_array);
                    samples.push(f64::from(sample));
                }
            }

            samples
        }
        crate::frame::AudioBuffer::Planar(planes) => {
            if planes.is_empty() {
                return Vec::new();
            }

            let channels = planes.len();
            let sample_size = std::mem::size_of::<f32>();
            let frames = planes[0].len() / sample_size;
            let mut interleaved = Vec::with_capacity(frames * channels);

            for frame_idx in 0..frames {
                for plane in planes {
                    let offset = frame_idx * sample_size;
                    if offset + 4 <= plane.len() {
                        let bytes_array = [
                            plane[offset],
                            plane[offset + 1],
                            plane[offset + 2],
                            plane[offset + 3],
                        ];
                        let sample = f32::from_le_bytes(bytes_array);
                        interleaved.push(f64::from(sample));
                    }
                }
            }

            interleaved
        }
    }
}
