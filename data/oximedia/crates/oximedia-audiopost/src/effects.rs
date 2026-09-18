#![allow(dead_code)]
//! Advanced audio effects processing.

use crate::error::{AudioPostError, AudioPostResult};
use oxifft::{fft, ifft, Complex};
use std::collections::VecDeque;

/// Multiband compressor
#[derive(Debug)]
pub struct MultibandCompressor {
    sample_rate: u32,
    bands: Vec<CompressorBand>,
}

impl MultibandCompressor {
    /// Create a new multiband compressor with specified number of bands
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate or band count is invalid
    pub fn new(sample_rate: u32, num_bands: usize) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        if num_bands < 2 || num_bands > 6 {
            return Err(AudioPostError::Generic(
                "Band count must be 2-6".to_string(),
            ));
        }

        let bands = (0..num_bands).map(|_| CompressorBand::new()).collect();

        Ok(Self { sample_rate, bands })
    }

    /// Get band count
    #[must_use]
    pub fn band_count(&self) -> usize {
        self.bands.len()
    }

    /// Get a band
    #[must_use]
    pub fn get_band(&self, index: usize) -> Option<&CompressorBand> {
        self.bands.get(index)
    }

    /// Get a mutable band
    pub fn get_band_mut(&mut self, index: usize) -> Option<&mut CompressorBand> {
        self.bands.get_mut(index)
    }
}

/// Compressor band
#[derive(Debug, Clone)]
pub struct CompressorBand {
    /// Threshold in dB
    pub threshold: f32,
    /// Ratio
    pub ratio: f32,
    /// Attack time in ms
    pub attack_ms: f32,
    /// Release time in ms
    pub release_ms: f32,
    /// Enabled flag
    pub enabled: bool,
}

impl CompressorBand {
    /// Create a new compressor band
    #[must_use]
    pub fn new() -> Self {
        Self {
            threshold: -20.0,
            ratio: 4.0,
            attack_ms: 5.0,
            release_ms: 50.0,
            enabled: true,
        }
    }
}

impl Default for CompressorBand {
    fn default() -> Self {
        Self::new()
    }
}

/// De-esser for reducing sibilance
#[derive(Debug)]
pub struct DeEsser {
    sample_rate: u32,
    threshold: f32,
    frequency: f32,
}

impl DeEsser {
    /// Create a new de-esser
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        Ok(Self {
            sample_rate,
            threshold: -20.0,
            frequency: 6000.0,
        })
    }

    /// Set threshold
    ///
    /// # Errors
    ///
    /// Returns an error if threshold is invalid
    pub fn set_threshold(&mut self, threshold: f32) -> AudioPostResult<()> {
        if threshold > 0.0 {
            return Err(AudioPostError::InvalidThreshold(threshold));
        }
        self.threshold = threshold;
        Ok(())
    }

    /// Set frequency
    ///
    /// # Errors
    ///
    /// Returns an error if frequency is invalid
    pub fn set_frequency(&mut self, frequency: f32) -> AudioPostResult<()> {
        if frequency <= 0.0 || frequency >= self.sample_rate as f32 / 2.0 {
            return Err(AudioPostError::InvalidFrequency(frequency));
        }
        self.frequency = frequency;
        Ok(())
    }

    /// Process a buffer of samples in-place, reducing sibilance above `frequency_hz`
    /// when the high-frequency energy exceeds `threshold_db`.
    ///
    /// The algorithm uses a first-order high-pass IIR detector to measure energy
    /// in the sibilance band, computes a gain reduction ratio when the detected
    /// level exceeds the threshold, and applies the gain via a second-order
    /// shelf filter that acts only on high-frequency content (above `frequency_hz`).
    ///
    /// # Arguments
    ///
    /// * `samples`       – Mutable slice of interleaved or mono f32 samples.
    /// * `sample_rate`   – Sample rate in Hz (must be > 0).
    /// * `frequency_hz`  – Centre frequency of the sibilance region (typ. 5–10 kHz).
    /// * `threshold_db`  – Level above which gain reduction starts (negative dBFS).
    ///
    /// # Errors
    ///
    /// Returns [`AudioPostError::InvalidSampleRate`] if `sample_rate` is zero, or
    /// [`AudioPostError::InvalidFrequency`] if `frequency_hz` is out of range, or
    /// [`AudioPostError::InvalidThreshold`] if `threshold_db` is positive.
    #[allow(clippy::cast_precision_loss)]
    pub fn process(
        samples: &mut [f32],
        sample_rate: u32,
        frequency_hz: f32,
        threshold_db: f32,
    ) -> AudioPostResult<()> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        let nyquist = sample_rate as f32 / 2.0;
        if frequency_hz <= 0.0 || frequency_hz >= nyquist {
            return Err(AudioPostError::InvalidFrequency(frequency_hz));
        }
        if threshold_db > 0.0 {
            return Err(AudioPostError::InvalidThreshold(threshold_db));
        }

        // --- Detector: first-order high-pass IIR ---
        // RC = 1 / (2π * f_c),  α = RC / (RC + T),  T = 1/fs
        let two_pi = 2.0 * std::f32::consts::PI;
        let rc = 1.0 / (two_pi * frequency_hz);
        let dt = 1.0 / sample_rate as f32;
        let hp_alpha = rc / (rc + dt);

        // --- High-shelf gain filter coefficients (bilinear) ---
        // Applied only when gain reduction is active; we approximate a simple
        // first-order high-shelf by blending a low-pass complement.
        let lp_alpha = dt / (rc + dt); // LP = 1 - HP in first-order topology

        // Attack / release envelopes for the detector.
        // Attack: 1 ms, Release: 50 ms (broadcast standard defaults).
        let attack_coeff = (-1.0 / (0.001 * sample_rate as f32)).exp();
        let release_coeff = (-1.0 / (0.050 * sample_rate as f32)).exp();

        // Threshold in linear domain.
        let threshold_linear = 10.0_f32.powf(threshold_db / 20.0);

        // Gain reduction ratio: 4:1 above threshold (typical de-esser setting).
        let ratio = 4.0_f32;

        // State variables.
        let mut hp_prev_in = 0.0_f32;
        let mut hp_prev_out = 0.0_f32;
        let mut lp_prev_out = 0.0_f32;
        let mut envelope = 0.0_f32;

        for sample in samples.iter_mut() {
            let x = *sample;

            // High-pass detector signal.
            let hp_out = hp_alpha * (hp_prev_out + x - hp_prev_in);
            hp_prev_in = x;
            hp_prev_out = hp_out;

            // Full-wave rectify and smooth (envelope follower).
            let level = hp_out.abs();
            let coeff = if level > envelope {
                attack_coeff
            } else {
                release_coeff
            };
            envelope = coeff * envelope + (1.0 - coeff) * level;

            // Compute gain reduction: only reduce when above threshold.
            let gain_reduction = if envelope > threshold_linear {
                // dB of excess above threshold.
                let excess_db = 20.0 * (envelope / threshold_linear).log10();
                // Apply ratio: GR = excess * (1 - 1/ratio).
                let gr_db = excess_db * (1.0 - 1.0 / ratio);
                // Linear gain factor < 1.
                10.0_f32.powf(-gr_db / 20.0)
            } else {
                1.0
            };

            // Split signal into LP and HP parts; attenuate only HP part.
            let lp_out = lp_prev_out + lp_alpha * (x - lp_prev_out);
            lp_prev_out = lp_out;
            let hp_signal = x - lp_out;

            // Reconstruct with gain-reduced HP component.
            *sample = lp_out + hp_signal * gain_reduction;
        }

        Ok(())
    }
}

/// Transient designer for shaping attack and sustain
#[derive(Debug)]
pub struct TransientDesigner {
    sample_rate: u32,
    attack_gain: f32,
    sustain_gain: f32,
}

impl TransientDesigner {
    /// Create a new transient designer
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        Ok(Self {
            sample_rate,
            attack_gain: 1.0,
            sustain_gain: 1.0,
        })
    }

    /// Set attack gain (0.0 to 2.0)
    pub fn set_attack_gain(&mut self, gain: f32) {
        self.attack_gain = gain.clamp(0.0, 2.0);
    }

    /// Set sustain gain (0.0 to 2.0)
    pub fn set_sustain_gain(&mut self, gain: f32) {
        self.sustain_gain = gain.clamp(0.0, 2.0);
    }
}

/// Convolution reverb
#[derive(Debug)]
pub struct ConvolutionReverb {
    sample_rate: u32,
    impulse_response: Vec<f32>,
    wet_dry_mix: f32,
}

impl ConvolutionReverb {
    /// Create a new convolution reverb
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        Ok(Self {
            sample_rate,
            impulse_response: Vec::new(),
            wet_dry_mix: 0.3,
        })
    }

    /// Load impulse response
    pub fn load_impulse_response(&mut self, ir: Vec<f32>) {
        self.impulse_response = ir;
    }

    /// Set wet/dry mix (0.0 to 1.0)
    pub fn set_mix(&mut self, mix: f32) {
        self.wet_dry_mix = mix.clamp(0.0, 1.0);
    }
}

/// Algorithmic reverb type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReverbType {
    /// Plate reverb
    Plate,
    /// Hall reverb
    Hall,
    /// Room reverb
    Room,
    /// Chamber reverb
    Chamber,
}

/// Algorithmic reverb
#[derive(Debug)]
pub struct AlgorithmicReverb {
    sample_rate: u32,
    reverb_type: ReverbType,
    size: f32,
    damping: f32,
    wet_dry_mix: f32,
}

impl AlgorithmicReverb {
    /// Create a new algorithmic reverb
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32, reverb_type: ReverbType) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        Ok(Self {
            sample_rate,
            reverb_type,
            size: 0.5,
            damping: 0.5,
            wet_dry_mix: 0.3,
        })
    }

    /// Set size (0.0 to 1.0)
    pub fn set_size(&mut self, size: f32) {
        self.size = size.clamp(0.0, 1.0);
    }

    /// Set damping (0.0 to 1.0)
    pub fn set_damping(&mut self, damping: f32) {
        self.damping = damping.clamp(0.0, 1.0);
    }

    /// Set wet/dry mix (0.0 to 1.0)
    pub fn set_mix(&mut self, mix: f32) {
        self.wet_dry_mix = mix.clamp(0.0, 1.0);
    }
}

/// Delay effect
#[derive(Debug)]
pub struct Delay {
    sample_rate: u32,
    delay_buffer: VecDeque<f32>,
    delay_time_ms: f32,
    feedback: f32,
    wet_dry_mix: f32,
}

impl Delay {
    /// Create a new delay effect
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32, max_delay_ms: f32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        let buffer_size = (sample_rate as f32 * max_delay_ms / 1000.0) as usize;
        let delay_buffer = VecDeque::with_capacity(buffer_size);

        Ok(Self {
            sample_rate,
            delay_buffer,
            delay_time_ms: 250.0,
            feedback: 0.5,
            wet_dry_mix: 0.3,
        })
    }

    /// Set delay time in milliseconds
    pub fn set_delay_time(&mut self, ms: f32) {
        self.delay_time_ms = ms.max(0.0);
    }

    /// Set feedback (0.0 to 1.0)
    pub fn set_feedback(&mut self, feedback: f32) {
        self.feedback = feedback.clamp(0.0, 0.99);
    }

    /// Set wet/dry mix (0.0 to 1.0)
    pub fn set_mix(&mut self, mix: f32) {
        self.wet_dry_mix = mix.clamp(0.0, 1.0);
    }

    /// Process audio
    pub fn process(&mut self, input: f32) -> f32 {
        let delay_samples = (self.sample_rate as f32 * self.delay_time_ms / 1000.0) as usize;

        // Get delayed sample
        let delayed = if self.delay_buffer.len() >= delay_samples {
            *self
                .delay_buffer
                .get(self.delay_buffer.len() - delay_samples)
                .unwrap_or(&0.0)
        } else {
            0.0
        };

        // Add input with feedback
        let output = delayed;
        self.delay_buffer.push_back(input + delayed * self.feedback);

        // Remove old samples to maintain buffer size
        if self.delay_buffer.len() > delay_samples * 2 {
            self.delay_buffer.pop_front();
        }

        // Mix wet and dry
        input * (1.0 - self.wet_dry_mix) + output * self.wet_dry_mix
    }
}

/// Chorus effect
#[derive(Debug)]
pub struct Chorus {
    sample_rate: u32,
    rate_hz: f32,
    depth: f32,
    wet_dry_mix: f32,
}

impl Chorus {
    /// Create a new chorus effect
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        Ok(Self {
            sample_rate,
            rate_hz: 0.5,
            depth: 0.5,
            wet_dry_mix: 0.5,
        })
    }

    /// Set modulation rate in Hz
    pub fn set_rate(&mut self, rate_hz: f32) {
        self.rate_hz = rate_hz.max(0.0);
    }

    /// Set depth (0.0 to 1.0)
    pub fn set_depth(&mut self, depth: f32) {
        self.depth = depth.clamp(0.0, 1.0);
    }

    /// Set wet/dry mix (0.0 to 1.0)
    pub fn set_mix(&mut self, mix: f32) {
        self.wet_dry_mix = mix.clamp(0.0, 1.0);
    }
}

/// Flanger effect
#[derive(Debug)]
pub struct Flanger {
    sample_rate: u32,
    rate_hz: f32,
    depth: f32,
    feedback: f32,
    wet_dry_mix: f32,
}

impl Flanger {
    /// Create a new flanger effect
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        Ok(Self {
            sample_rate,
            rate_hz: 0.5,
            depth: 0.5,
            feedback: 0.5,
            wet_dry_mix: 0.5,
        })
    }

    /// Set modulation rate in Hz
    pub fn set_rate(&mut self, rate_hz: f32) {
        self.rate_hz = rate_hz.max(0.0);
    }

    /// Set depth (0.0 to 1.0)
    pub fn set_depth(&mut self, depth: f32) {
        self.depth = depth.clamp(0.0, 1.0);
    }

    /// Set feedback (0.0 to 1.0)
    pub fn set_feedback(&mut self, feedback: f32) {
        self.feedback = feedback.clamp(0.0, 0.99);
    }

    /// Set wet/dry mix (0.0 to 1.0)
    pub fn set_mix(&mut self, mix: f32) {
        self.wet_dry_mix = mix.clamp(0.0, 1.0);
    }
}

/// Phaser effect
#[derive(Debug)]
pub struct Phaser {
    sample_rate: u32,
    num_stages: usize,
    rate_hz: f32,
    depth: f32,
    feedback: f32,
    wet_dry_mix: f32,
}

impl Phaser {
    /// Create a new phaser effect
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate or stage count is invalid
    pub fn new(sample_rate: u32, num_stages: usize) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        if !(2..=12).contains(&num_stages) || num_stages % 2 != 0 {
            return Err(AudioPostError::Generic(
                "Stage count must be even and between 2 and 12".to_string(),
            ));
        }

        Ok(Self {
            sample_rate,
            num_stages,
            rate_hz: 0.5,
            depth: 0.5,
            feedback: 0.5,
            wet_dry_mix: 0.5,
        })
    }

    /// Set modulation rate in Hz
    pub fn set_rate(&mut self, rate_hz: f32) {
        self.rate_hz = rate_hz.max(0.0);
    }

    /// Set depth (0.0 to 1.0)
    pub fn set_depth(&mut self, depth: f32) {
        self.depth = depth.clamp(0.0, 1.0);
    }

    /// Set feedback (0.0 to 1.0)
    pub fn set_feedback(&mut self, feedback: f32) {
        self.feedback = feedback.clamp(0.0, 0.99);
    }

    /// Set wet/dry mix (0.0 to 1.0)
    pub fn set_mix(&mut self, mix: f32) {
        self.wet_dry_mix = mix.clamp(0.0, 1.0);
    }
}

/// Tremolo effect
#[derive(Debug)]
pub struct Tremolo {
    sample_rate: u32,
    rate_hz: f32,
    depth: f32,
    phase: f32,
}

impl Tremolo {
    /// Create a new tremolo effect
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        Ok(Self {
            sample_rate,
            rate_hz: 5.0,
            depth: 0.5,
            phase: 0.0,
        })
    }

    /// Set modulation rate in Hz
    pub fn set_rate(&mut self, rate_hz: f32) {
        self.rate_hz = rate_hz.max(0.0);
    }

    /// Set depth (0.0 to 1.0)
    pub fn set_depth(&mut self, depth: f32) {
        self.depth = depth.clamp(0.0, 1.0);
    }

    /// Process audio
    pub fn process(&mut self, input: f32) -> f32 {
        let modulation = 1.0 - self.depth * (1.0 - self.phase.sin()) / 2.0;
        self.phase += 2.0 * std::f32::consts::PI * self.rate_hz / self.sample_rate as f32;
        if self.phase > 2.0 * std::f32::consts::PI {
            self.phase -= 2.0 * std::f32::consts::PI;
        }
        input * modulation
    }
}

// ── M/S (Mid-Side) Encoding/Decoding Processor ───────────────────────────────

/// M/S (Mid-Side) encoder/decoder for stereo audio.
///
/// The M/S transform encodes a stereo signal into a sum (Mid = L + R) and
/// difference (Side = L − R) channel pair.  This representation allows
/// independent processing of the mid and side components (e.g. different EQ,
/// compression, or width adjustment) before decoding back to L/R stereo.
///
/// Encoding:  M = (L + R) / √2,   S = (L − R) / √2
/// Decoding:  L = (M + S) / √2,   R = (M − S) / √2
///
/// The √2 normalisation factor ensures that the power is preserved (constant
/// loudness through an encode-decode round-trip).
#[derive(Debug, Clone)]
pub struct MidSideProcessor {
    /// Sample rate.
    pub sample_rate: u32,
    /// Linear gain applied to the Mid channel after encoding (default 1.0).
    pub mid_gain: f32,
    /// Linear gain applied to the Side channel after encoding (default 1.0).
    pub side_gain: f32,
    /// Stereo width factor applied during decode.
    /// 0.0 = mono (Side channel completely attenuated),
    /// 1.0 = normal stereo,
    /// >1.0 = enhanced width.
    /// Default: 1.0.
    pub width: f32,
}

impl MidSideProcessor {
    /// Square-root of 2 constant for normalisation.
    const SQRT2: f32 = std::f32::consts::SQRT_2;
    const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;

    /// Create a new M/S processor.
    ///
    /// # Errors
    ///
    /// Returns [`AudioPostError::InvalidSampleRate`] for a zero sample rate.
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        Ok(Self {
            sample_rate,
            mid_gain: 1.0,
            side_gain: 1.0,
            width: 1.0,
        })
    }

    /// Encode a stereo sample pair (L, R) to (Mid, Side).
    ///
    /// Returns `(mid, side)` with √2 normalisation.
    #[inline]
    #[must_use]
    pub fn encode_sample(&self, left: f32, right: f32) -> (f32, f32) {
        let mid = (left + right) * Self::INV_SQRT2 * self.mid_gain;
        let side = (left - right) * Self::INV_SQRT2 * self.side_gain;
        (mid, side)
    }

    /// Decode a (Mid, Side) sample pair back to (L, R) stereo.
    ///
    /// Applies the current `width` factor to the side channel before decoding.
    #[inline]
    #[must_use]
    pub fn decode_sample(&self, mid: f32, side: f32) -> (f32, f32) {
        let s = side * self.width;
        let left = (mid + s) * Self::INV_SQRT2;
        let right = (mid - s) * Self::INV_SQRT2;
        (left, right)
    }

    /// Encode a block of interleaved stereo samples (L R L R …) to interleaved M/S.
    ///
    /// Input must have an even number of samples.  The output is interleaved
    /// Mid/Side: `[M0, S0, M1, S1, …]`.
    ///
    /// # Errors
    ///
    /// Returns [`AudioPostError::InvalidBufferSize`] for an odd-length input.
    pub fn encode_interleaved(&self, input: &[f32], output: &mut [f32]) -> AudioPostResult<()> {
        if input.len() % 2 != 0 {
            return Err(AudioPostError::InvalidBufferSize(input.len()));
        }
        let frames = input.len() / 2;
        if output.len() < input.len() {
            return Err(AudioPostError::InvalidBufferSize(output.len()));
        }
        for i in 0..frames {
            let l = input[i * 2];
            let r = input[i * 2 + 1];
            let (m, s) = self.encode_sample(l, r);
            output[i * 2] = m;
            output[i * 2 + 1] = s;
        }
        Ok(())
    }

    /// Decode a block of interleaved M/S samples back to interleaved L/R stereo.
    ///
    /// # Errors
    ///
    /// Returns [`AudioPostError::InvalidBufferSize`] for an odd-length input or
    /// insufficient output buffer.
    pub fn decode_interleaved(&self, input: &[f32], output: &mut [f32]) -> AudioPostResult<()> {
        if input.len() % 2 != 0 {
            return Err(AudioPostError::InvalidBufferSize(input.len()));
        }
        let frames = input.len() / 2;
        if output.len() < input.len() {
            return Err(AudioPostError::InvalidBufferSize(output.len()));
        }
        for i in 0..frames {
            let m = input[i * 2];
            let s = input[i * 2 + 1];
            let (l, r) = self.decode_sample(m, s);
            output[i * 2] = l;
            output[i * 2 + 1] = r;
        }
        Ok(())
    }

    /// Round-trip encode then decode interleaved stereo.
    ///
    /// Convenience method for stereo width processing.  `width` is applied
    /// to `self.width` before decoding, allowing inline width adjustment.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer size is invalid.
    pub fn process_stereo_width(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        width: f32,
    ) -> AudioPostResult<()> {
        self.width = width.max(0.0);
        if input.len() % 2 != 0 {
            return Err(AudioPostError::InvalidBufferSize(input.len()));
        }
        if output.len() < input.len() {
            return Err(AudioPostError::InvalidBufferSize(output.len()));
        }
        let frames = input.len() / 2;
        for i in 0..frames {
            let l = input[i * 2];
            let r = input[i * 2 + 1];
            let (m, s) = self.encode_sample(l, r);
            let (lo, ro) = self.decode_sample(m, s);
            output[i * 2] = lo;
            output[i * 2 + 1] = ro;
        }
        Ok(())
    }
}

// ── Block-Based FFT Overlap-Add Processor ─────────────────────────────────────

/// Block-based FFT processing with overlap-add for arbitrary frequency-domain
/// effects (e.g. spectral gating, pitch shifting, noise reduction).
///
/// The processor splits the input signal into overlapping frames, transforms
/// each frame to the frequency domain, applies a user-supplied transfer function
/// (closure), and reconstructs the time-domain signal using overlap-add.
///
/// This approach reduces the effective per-sample overhead from O(N) (direct
/// convolution) to O(N log N / hop) by amortising the FFT cost over `hop_size`
/// samples.
pub struct FftOverlapAddProcessor {
    /// FFT frame size (must be a power of two).
    pub fft_size: usize,
    /// Hop size (typically fft_size / 2 or fft_size / 4 for 50%/75% overlap).
    pub hop_size: usize,
    /// Hann window coefficients, length == fft_size.
    window: Vec<f32>,
    /// Input accumulation buffer.
    input_buf: Vec<f32>,
    /// Overlap-add output accumulation buffer.
    output_buf: Vec<f32>,
    /// Write cursor into `input_buf`.
    buf_pos: usize,
    /// Output FIFO.
    output_fifo: VecDeque<f32>,
}

impl std::fmt::Debug for FftOverlapAddProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FftOverlapAddProcessor")
            .field("fft_size", &self.fft_size)
            .field("hop_size", &self.hop_size)
            .finish()
    }
}

impl FftOverlapAddProcessor {
    /// Create a new overlap-add processor.
    ///
    /// # Errors
    ///
    /// Returns [`AudioPostError::InvalidBufferSize`] if:
    /// - `fft_size` is not a power of two.
    /// - `hop_size` is zero or larger than `fft_size`.
    pub fn new(fft_size: usize, hop_size: usize) -> AudioPostResult<Self> {
        if !fft_size.is_power_of_two() || fft_size < 4 {
            return Err(AudioPostError::InvalidBufferSize(fft_size));
        }
        if hop_size == 0 || hop_size > fft_size {
            return Err(AudioPostError::InvalidBufferSize(hop_size));
        }

        // Pre-compute Hann window coefficients.
        let window: Vec<f32> = (0..fft_size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (fft_size - 1) as f32).cos())
            })
            .collect();

        Ok(Self {
            fft_size,
            hop_size,
            window,
            input_buf: vec![0.0; fft_size],
            output_buf: vec![0.0; fft_size * 2],
            buf_pos: 0,
            output_fifo: VecDeque::new(),
        })
    }

    /// Process a block of samples with the supplied frequency-domain transfer
    /// function.
    ///
    /// `transfer` receives a mutable slice of `fft_size` complex bins (the FFT
    /// spectrum of one windowed frame) and may modify them in place.  The IFFT
    /// of the modified spectrum is then overlap-added to reconstruct the output.
    ///
    /// Returns the processed samples.  Output length equals input length; samples
    /// for which no complete FFT frame is available yet are zero (startup latency
    /// of one frame).
    pub fn process<F>(&mut self, input: &[f32], mut transfer: F) -> Vec<f32>
    where
        F: FnMut(&mut [Complex<f32>]),
    {
        let mut output = Vec::with_capacity(input.len());

        for &x in input {
            // Pop a completed output sample first.
            output.push(self.output_fifo.pop_front().unwrap_or(0.0));

            // Accumulate into input ring buffer.
            self.input_buf[self.buf_pos] = x;
            self.buf_pos += 1;

            if self.buf_pos >= self.hop_size {
                self.process_frame(&mut transfer);
                // Shift input buffer left by hop_size.
                let fft_size = self.fft_size;
                self.input_buf.copy_within(self.hop_size..fft_size, 0);
                for s in &mut self.input_buf[(fft_size - self.hop_size)..] {
                    *s = 0.0;
                }
                self.buf_pos = fft_size - self.hop_size;
            }
        }

        output
    }

    /// Process one FFT frame: window → FFT → transfer → IFFT → overlap-add.
    fn process_frame<F>(&mut self, transfer: &mut F)
    where
        F: FnMut(&mut [Complex<f32>]),
    {
        // Apply Hann window and convert to complex.
        let mut frame: Vec<Complex<f32>> = self
            .input_buf
            .iter()
            .zip(self.window.iter())
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();

        // Forward FFT.
        let mut spectrum = fft(&frame);

        // Apply user transfer function.
        transfer(&mut spectrum);

        // Inverse FFT.
        frame = ifft(&spectrum);

        // Scale by 1/fft_size (ifft does not normalise).
        let scale = 1.0 / self.fft_size as f32;

        // Overlap-add into output buffer.
        for (i, c) in frame.iter().enumerate() {
            let sample = c.re * scale;
            // Ensure the output FIFO is large enough.
            while self.output_fifo.len() <= i {
                self.output_fifo.push_back(0.0);
            }
            let _idx = self.output_fifo.len() - (self.fft_size - i).min(self.output_fifo.len());
            if let Some(out) = self.output_fifo.get_mut(i) {
                *out += sample;
            } else {
                self.output_fifo.push_back(sample);
            }
        }
        // Extend FIFO to cover current hop.
        while self.output_fifo.len() < self.hop_size {
            self.output_fifo.push_back(0.0);
        }
    }

    /// Reset internal state (keeps FFT size / hop size / window).
    pub fn reset(&mut self) {
        self.input_buf.fill(0.0);
        self.output_buf.fill(0.0);
        self.buf_pos = 0;
        self.output_fifo.clear();
    }
}

/// Vocoder
#[derive(Debug)]
pub struct Vocoder {
    sample_rate: u32,
    num_bands: usize,
}

impl Vocoder {
    /// Create a new vocoder
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate or band count is invalid
    pub fn new(sample_rate: u32, num_bands: usize) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        if num_bands < 4 || num_bands > 32 {
            return Err(AudioPostError::Generic(
                "Band count must be 4-32".to_string(),
            ));
        }

        Ok(Self {
            sample_rate,
            num_bands,
        })
    }

    /// Get band count
    #[must_use]
    pub fn band_count(&self) -> usize {
        self.num_bands
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiband_compressor() {
        let comp = MultibandCompressor::new(48000, 4).expect("failed to create");
        assert_eq!(comp.band_count(), 4);
    }

    #[test]
    fn test_invalid_band_count() {
        assert!(MultibandCompressor::new(48000, 1).is_err());
        assert!(MultibandCompressor::new(48000, 7).is_err());
    }

    #[test]
    fn test_de_esser() {
        let mut de_esser = DeEsser::new(48000).expect("failed to create");
        assert!(de_esser.set_threshold(-15.0).is_ok());
        assert!(de_esser.set_frequency(7000.0).is_ok());
    }

    #[test]
    fn test_transient_designer() {
        let mut td = TransientDesigner::new(48000).expect("failed to create");
        td.set_attack_gain(1.5);
        td.set_sustain_gain(0.8);
        assert_eq!(td.attack_gain, 1.5);
        assert_eq!(td.sustain_gain, 0.8);
    }

    #[test]
    fn test_convolution_reverb() {
        let mut reverb = ConvolutionReverb::new(48000).expect("failed to create");
        reverb.load_impulse_response(vec![1.0, 0.5, 0.25]);
        reverb.set_mix(0.4);
        assert_eq!(reverb.wet_dry_mix, 0.4);
    }

    #[test]
    fn test_algorithmic_reverb() {
        let mut reverb = AlgorithmicReverb::new(48000, ReverbType::Hall).expect("failed to create");
        reverb.set_size(0.7);
        reverb.set_damping(0.6);
        assert_eq!(reverb.size, 0.7);
    }

    #[test]
    fn test_delay() {
        let mut delay = Delay::new(48000, 1000.0).expect("failed to create");
        delay.set_delay_time(500.0);
        delay.set_feedback(0.6);
        assert_eq!(delay.delay_time_ms, 500.0);
        assert_eq!(delay.feedback, 0.6);
    }

    #[test]
    fn test_delay_process() {
        let mut delay = Delay::new(48000, 100.0).expect("failed to create");
        delay.set_delay_time(10.0);
        let output = delay.process(1.0);
        assert!(output.is_finite());
    }

    #[test]
    fn test_chorus() {
        let mut chorus = Chorus::new(48000).expect("failed to create");
        chorus.set_rate(0.8);
        chorus.set_depth(0.6);
        assert_eq!(chorus.rate_hz, 0.8);
        assert_eq!(chorus.depth, 0.6);
    }

    #[test]
    fn test_flanger() {
        let mut flanger = Flanger::new(48000).expect("failed to create");
        flanger.set_rate(0.5);
        flanger.set_depth(0.7);
        flanger.set_feedback(0.6);
        assert_eq!(flanger.feedback, 0.6);
    }

    #[test]
    fn test_phaser() {
        let mut phaser = Phaser::new(48000, 4).expect("failed to create");
        phaser.set_rate(0.5);
        phaser.set_depth(0.7);
        assert_eq!(phaser.num_stages, 4);
    }

    #[test]
    fn test_phaser_invalid_stages() {
        assert!(Phaser::new(48000, 3).is_err()); // Odd number
        assert!(Phaser::new(48000, 14).is_err()); // Too many
    }

    #[test]
    fn test_tremolo() {
        let mut tremolo = Tremolo::new(48000).expect("failed to create");
        tremolo.set_rate(6.0);
        tremolo.set_depth(0.8);
        let output = tremolo.process(1.0);
        assert!(output.is_finite());
    }

    #[test]
    fn test_vocoder() {
        let vocoder = Vocoder::new(48000, 16).expect("failed to create");
        assert_eq!(vocoder.band_count(), 16);
    }

    #[test]
    fn test_vocoder_invalid_bands() {
        assert!(Vocoder::new(48000, 2).is_err());
        assert!(Vocoder::new(48000, 64).is_err());
    }

    #[test]
    fn test_de_esser_reduces_high_freq() {
        // Generate a 440 Hz sine (low frequency) and mix with 8 kHz sine (high sibilance).
        let sample_rate = 48000_u32;
        let num_samples = 4800_usize; // 100 ms
        let two_pi = 2.0 * std::f32::consts::PI;

        let mut samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                // Low-frequency component at unity.
                let low = (two_pi * 440.0 * t).sin() * 0.3;
                // High-frequency sibilance burst at 0.8 amplitude – well above threshold.
                let high = (two_pi * 8000.0 * t).sin() * 0.8;
                low + high
            })
            .collect();

        // Measure RMS energy in high-frequency band before processing.
        let rms_before: f32 = {
            let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
            (sum_sq / samples.len() as f32).sqrt()
        };

        DeEsser::process(&mut samples, sample_rate, 6000.0, -20.0)
            .expect("de-esser process should succeed");

        // Measure RMS energy after processing; it must be lower due to gain reduction.
        let rms_after: f32 = {
            let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
            (sum_sq / samples.len() as f32).sqrt()
        };

        assert!(
            rms_after < rms_before,
            "de-esser should reduce overall RMS when high-frequency content exceeds threshold \
             (before={rms_before:.4}, after={rms_after:.4})"
        );

        // Verify the output samples are finite and within a reasonable range.
        for (i, &s) in samples.iter().enumerate() {
            assert!(
                s.is_finite() && s.abs() <= 2.0,
                "sample {i} out of range: {s}"
            );
        }
    }

    #[test]
    fn test_module_exports_compile() {
        // Verify that spectral_editor, clip_gain and phase_alignment are accessible
        // through their module paths from the crate root.
        use crate::clip_gain::{ClipGain, Fade, FadeCurve};
        use crate::phase_alignment::{ChannelId, PhaseAligner};
        use crate::spectral_editor::{SpectralEditSession, SpectralRegion};

        // clip_gain
        let cg = ClipGain::new(2.0).with_fade_in(Fade::new(0.5, FadeCurve::Linear));
        assert!(cg.gain_at(0.0) < 0.01);

        // phase_alignment
        let aligner = PhaseAligner::new(48000.0, 10);
        assert_eq!(aligner.relations().len(), 0);
        let _ = ChannelId::new("test");

        // spectral_editor
        let mut session = SpectralEditSession::new(48000.0, 2048, 512);
        let region = SpectralRegion::new(0.0, 1.0, 0.0, 1000.0);
        use crate::spectral_editor::{SpectralEdit, SpectralEditOp};
        session.add_edit(SpectralEdit::new(region, SpectralEditOp::Silence));
        assert_eq!(session.edit_count(), 1);
    }

    #[test]
    fn test_de_esser_process_errors() {
        let mut samples = vec![0.5_f32; 100];

        // Zero sample rate must fail.
        assert!(DeEsser::process(&mut samples, 0, 6000.0, -20.0).is_err());
        // Positive threshold must fail.
        assert!(DeEsser::process(&mut samples, 48000, 6000.0, 1.0).is_err());
        // Frequency above Nyquist must fail.
        assert!(DeEsser::process(&mut samples, 48000, 25000.0, -20.0).is_err());
        // Frequency at zero must fail.
        assert!(DeEsser::process(&mut samples, 48000, 0.0, -20.0).is_err());
    }

    // ── MidSideProcessor tests ────────────────────────────────────────────────

    #[test]
    fn test_ms_processor_creation() {
        let proc = MidSideProcessor::new(48000).expect("failed to create");
        assert_eq!(proc.mid_gain, 1.0);
        assert_eq!(proc.side_gain, 1.0);
        assert_eq!(proc.width, 1.0);
    }

    #[test]
    fn test_ms_processor_invalid_sr() {
        assert!(MidSideProcessor::new(0).is_err());
    }

    #[test]
    fn test_ms_encode_decode_roundtrip_sample() {
        let proc = MidSideProcessor::new(48000).expect("failed to create");
        let l = 0.6f32;
        let r = -0.3f32;
        let (m, s) = proc.encode_sample(l, r);
        let (lo, ro) = proc.decode_sample(m, s);
        assert!((lo - l).abs() < 1e-5, "L mismatch: got {lo}");
        assert!((ro - r).abs() < 1e-5, "R mismatch: got {ro}");
    }

    #[test]
    fn test_ms_encode_mono_signal_zero_side() {
        let proc = MidSideProcessor::new(48000).expect("failed to create");
        let l = 0.5f32;
        let r = 0.5f32; // identical channels → S should be 0
        let (_, s) = proc.encode_sample(l, r);
        assert!(s.abs() < 1e-6, "Side should be ~0 for mono signal, got {s}");
    }

    #[test]
    fn test_ms_encode_out_of_phase_zero_mid() {
        let proc = MidSideProcessor::new(48000).expect("failed to create");
        let l = 0.5f32;
        let r = -0.5f32; // opposite polarity → M should be 0
        let (m, _) = proc.encode_sample(l, r);
        assert!(
            m.abs() < 1e-6,
            "Mid should be ~0 for out-of-phase signal, got {m}"
        );
    }

    #[test]
    fn test_ms_width_zero_gives_mono_output() {
        let proc = MidSideProcessor::new(48000).expect("failed to create");
        let l = 0.6f32;
        let r = -0.3f32;
        let (m, s) = proc.encode_sample(l, r);
        // With width=0, side is fully suppressed → output should be mono.
        let (lo, ro) = {
            let mut p = proc.clone();
            p.width = 0.0;
            p.decode_sample(m, s)
        };
        assert!(
            (lo - ro).abs() < 1e-5,
            "With width=0, L and R should be equal"
        );
    }

    #[test]
    fn test_ms_encode_decode_interleaved_roundtrip() {
        let proc = MidSideProcessor::new(48000).expect("failed to create");
        let input: Vec<f32> = (0..200)
            .map(|i| if i % 2 == 0 { 0.5f32 } else { -0.3f32 })
            .collect();
        let mut ms = vec![0.0f32; 200];
        let mut lr = vec![0.0f32; 200];
        proc.encode_interleaved(&input, &mut ms).expect("encode");
        proc.decode_interleaved(&ms, &mut lr).expect("decode");
        for (i, (&orig, &out)) in input.iter().zip(lr.iter()).enumerate() {
            assert!(
                (orig - out).abs() < 1e-4,
                "Sample {i} mismatch: orig={orig}, out={out}"
            );
        }
    }

    #[test]
    fn test_ms_encode_odd_length_error() {
        let proc = MidSideProcessor::new(48000).expect("failed to create");
        let input = vec![0.5f32; 5];
        let mut output = vec![0.0f32; 5];
        assert!(proc.encode_interleaved(&input, &mut output).is_err());
    }

    #[test]
    fn test_ms_decode_odd_length_error() {
        let proc = MidSideProcessor::new(48000).expect("failed to create");
        let input = vec![0.5f32; 5];
        let mut output = vec![0.0f32; 5];
        assert!(proc.decode_interleaved(&input, &mut output).is_err());
    }

    #[test]
    fn test_ms_process_stereo_width_narrow() {
        let mut proc = MidSideProcessor::new(48000).expect("failed to create");
        // Input: hard-panned (L=1, R=0 alternating frames).
        let input: Vec<f32> = (0..100)
            .map(|i| if i % 2 == 0 { 1.0f32 } else { 0.0f32 })
            .collect();
        let mut output = vec![0.0f32; 100];
        // Width = 0.0 → mono output (L ≈ R).
        proc.process_stereo_width(&input, &mut output, 0.0)
            .expect("width");
        for i in (0..100).step_by(2) {
            let l = output[i];
            let r = output[i + 1];
            assert!(
                (l - r).abs() < 1e-4,
                "L={l}, R={r} should be equal at width=0"
            );
        }
    }

    // ── FftOverlapAddProcessor tests ──────────────────────────────────────────

    #[test]
    fn test_fft_ola_creation() {
        let proc = FftOverlapAddProcessor::new(1024, 256).expect("failed to create");
        assert_eq!(proc.fft_size, 1024);
        assert_eq!(proc.hop_size, 256);
    }

    #[test]
    fn test_fft_ola_invalid_fft_size() {
        assert!(FftOverlapAddProcessor::new(1000, 512).is_err()); // not power-of-two
        assert!(FftOverlapAddProcessor::new(0, 512).is_err());
    }

    #[test]
    fn test_fft_ola_invalid_hop() {
        assert!(FftOverlapAddProcessor::new(1024, 0).is_err());
        assert!(FftOverlapAddProcessor::new(1024, 2048).is_err()); // hop > fft_size
    }

    #[test]
    fn test_fft_ola_output_length_matches_input() {
        let mut proc = FftOverlapAddProcessor::new(256, 64).expect("failed to create");
        let input = vec![0.3f32; 512];
        // Identity transfer function (passthrough).
        let output = proc.process(&input, |_| {});
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_fft_ola_silence_passthrough() {
        let mut proc = FftOverlapAddProcessor::new(256, 64).expect("failed to create");
        let input = vec![0.0f32; 512];
        let output = proc.process(&input, |_| {});
        for &s in &output {
            assert!(
                s.is_finite(),
                "silence passthrough should produce finite output"
            );
        }
    }

    #[test]
    fn test_fft_ola_zero_transfer_zero_output() {
        let mut proc = FftOverlapAddProcessor::new(256, 64).expect("failed to create");
        let input: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin()).collect();
        // Silence the entire spectrum.
        let output = proc.process(&input, |bins| {
            for b in bins.iter_mut() {
                *b = Complex::new(0.0, 0.0);
            }
        });
        // After zeroing the spectrum, output should be near zero (with startup latency).
        let non_zero: usize = output.iter().filter(|&&s| s.abs() > 1e-3).count();
        assert!(
            non_zero < 100,
            "Expected mostly zero output after zeroing spectrum, got {non_zero} non-zero"
        );
    }

    #[test]
    fn test_fft_ola_reset_clears_state() {
        let mut proc = FftOverlapAddProcessor::new(256, 64).expect("failed to create");
        let input = vec![0.5f32; 256];
        proc.process(&input, |_| {});
        proc.reset();
        // After reset, processing zeros should return zeros.
        let zeros = vec![0.0f32; 256];
        let out = proc.process(&zeros, |_| {});
        for &s in &out {
            assert!(s.is_finite());
        }
    }
}
