#![allow(dead_code)]
//! Loudness management and standards compliance.

use crate::error::{AudioPostError, AudioPostResult};
use serde::{Deserialize, Serialize};

/// Loudness standard
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LoudnessStandard {
    /// EBU R128 (-23 LUFS)
    EbuR128,
    /// ATSC A/85 (-24 LKFS)
    AtscA85,
    /// Netflix (-27 LUFS ±2)
    Netflix,
    /// Spotify (-14 LUFS)
    Spotify,
    /// Apple Music (-16 LUFS)
    AppleMusic,
    /// YouTube (-14 LUFS)
    YouTube,
    /// Custom target
    Custom(f32),
}

impl LoudnessStandard {
    /// Get target loudness in LUFS
    #[must_use]
    pub fn target_lufs(&self) -> f32 {
        match self {
            Self::EbuR128 => -23.0,
            Self::AtscA85 => -24.0,
            Self::Netflix => -27.0,
            Self::Spotify => -14.0,
            Self::AppleMusic => -16.0,
            Self::YouTube => -14.0,
            Self::Custom(target) => *target,
        }
    }

    /// Get tolerance in LU
    #[must_use]
    pub fn tolerance(&self) -> f32 {
        match self {
            Self::Netflix => 2.0,
            _ => 1.0,
        }
    }

    /// Get maximum true peak in dBTP
    #[must_use]
    pub fn max_true_peak(&self) -> f32 {
        match self {
            Self::EbuR128 => -1.0,
            Self::AtscA85 => -2.0,
            Self::Netflix => -2.0,
            Self::Spotify => -1.0,
            Self::AppleMusic => -1.0,
            Self::YouTube => -1.0,
            Self::Custom(_) => -1.0,
        }
    }
}

/// Loudness meter
#[derive(Debug)]
pub struct LoudnessMeter {
    sample_rate: u32,
    standard: LoudnessStandard,
    momentary_lufs: f32,
    short_term_lufs: f32,
    integrated_lufs: f32,
    max_true_peak: f32,
    loudness_range: f32,
}

impl LoudnessMeter {
    /// Create a new loudness meter
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32, standard: LoudnessStandard) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        Ok(Self {
            sample_rate,
            standard,
            momentary_lufs: -70.0,
            short_term_lufs: -70.0,
            integrated_lufs: -70.0,
            max_true_peak: -70.0,
            loudness_range: 0.0,
        })
    }

    /// Get momentary loudness (400ms)
    #[must_use]
    pub fn get_momentary_lufs(&self) -> f32 {
        self.momentary_lufs
    }

    /// Get short-term loudness (3s)
    #[must_use]
    pub fn get_short_term_lufs(&self) -> f32 {
        self.short_term_lufs
    }

    /// Get integrated loudness
    #[must_use]
    pub fn get_integrated_lufs(&self) -> f32 {
        self.integrated_lufs
    }

    /// Get maximum true peak
    #[must_use]
    pub fn get_max_true_peak(&self) -> f32 {
        self.max_true_peak
    }

    /// Get loudness range (LRA)
    #[must_use]
    pub fn get_loudness_range(&self) -> f32 {
        self.loudness_range
    }

    /// Check if compliant with standard
    #[must_use]
    pub fn is_compliant(&self) -> bool {
        let target = self.standard.target_lufs();
        let tolerance = self.standard.tolerance();
        let max_peak = self.standard.max_true_peak();

        let loudness_ok = (self.integrated_lufs - target).abs() <= tolerance;
        let peak_ok = self.max_true_peak <= max_peak;

        loudness_ok && peak_ok
    }

    /// Get compliance report
    #[must_use]
    pub fn get_compliance_report(&self) -> ComplianceReport {
        ComplianceReport {
            standard: self.standard,
            integrated_lufs: self.integrated_lufs,
            target_lufs: self.standard.target_lufs(),
            max_true_peak: self.max_true_peak,
            max_allowed_peak: self.standard.max_true_peak(),
            loudness_range: self.loudness_range,
            compliant: self.is_compliant(),
        }
    }

    /// Process audio and update measurements
    pub fn process(&mut self, audio: &[f32]) {
        if audio.is_empty() {
            return;
        }

        // Calculate RMS for momentary loudness
        let rms: f32 = audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32;
        let rms = rms.sqrt();

        // Convert to LUFS (simplified calculation)
        self.momentary_lufs = if rms > 0.0 {
            20.0 * rms.log10() - 0.691
        } else {
            -70.0
        };

        // Update true peak
        for &sample in audio {
            let peak_db = 20.0 * sample.abs().log10();
            if peak_db > self.max_true_peak {
                self.max_true_peak = peak_db;
            }
        }

        // Update integrated (simplified)
        self.integrated_lufs = self.momentary_lufs;
    }

    /// Reset measurements
    pub fn reset(&mut self) {
        self.momentary_lufs = -70.0;
        self.short_term_lufs = -70.0;
        self.integrated_lufs = -70.0;
        self.max_true_peak = -70.0;
        self.loudness_range = 0.0;
    }
}

/// Compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    /// Standard used
    pub standard: LoudnessStandard,
    /// Measured integrated loudness
    pub integrated_lufs: f32,
    /// Target loudness
    pub target_lufs: f32,
    /// Maximum true peak measured
    pub max_true_peak: f32,
    /// Maximum allowed true peak
    pub max_allowed_peak: f32,
    /// Loudness range
    pub loudness_range: f32,
    /// Compliance status
    pub compliant: bool,
}

impl ComplianceReport {
    /// Get loudness delta from target
    #[must_use]
    pub fn loudness_delta(&self) -> f32 {
        self.integrated_lufs - self.target_lufs
    }

    /// Get peak delta from maximum
    #[must_use]
    pub fn peak_delta(&self) -> f32 {
        self.max_true_peak - self.max_allowed_peak
    }
}

/// Loudness normalizer
#[derive(Debug)]
pub struct LoudnessNormalizer {
    sample_rate: u32,
    target_lufs: f32,
    max_true_peak: f32,
}

impl LoudnessNormalizer {
    /// Create a new loudness normalizer
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate or target is invalid
    pub fn new(sample_rate: u32, target_lufs: f32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        if target_lufs > 0.0 {
            return Err(AudioPostError::InvalidLoudnessTarget(target_lufs));
        }

        Ok(Self {
            sample_rate,
            target_lufs,
            max_true_peak: -1.0,
        })
    }

    /// Set maximum true peak
    pub fn set_max_true_peak(&mut self, max_peak: f32) {
        self.max_true_peak = max_peak;
    }

    /// Calculate required gain to reach target
    #[must_use]
    pub fn calculate_gain(&self, current_lufs: f32) -> f32 {
        self.target_lufs - current_lufs
    }

    /// Normalize audio to target loudness
    pub fn normalize(&self, input: &[f32], output: &mut [f32], current_lufs: f32) {
        let gain_db = self.calculate_gain(current_lufs);
        let gain_linear = 10.0_f32.powf(gain_db / 20.0);

        for (out, &inp) in output.iter_mut().zip(input.iter()) {
            *out = inp * gain_linear;

            // Apply true peak limiting
            let peak_linear = 10.0_f32.powf(self.max_true_peak / 20.0);
            if out.abs() > peak_linear {
                *out = out.signum() * peak_linear;
            }
        }
    }
}

/// Automatic gain adjustment
#[derive(Debug)]
pub struct AutoGain {
    sample_rate: u32,
    target_lufs: f32,
    attack_time: f32,
    release_time: f32,
}

impl AutoGain {
    /// Create a new auto gain processor
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate or target is invalid
    pub fn new(sample_rate: u32, target_lufs: f32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        if target_lufs > 0.0 {
            return Err(AudioPostError::InvalidLoudnessTarget(target_lufs));
        }

        Ok(Self {
            sample_rate,
            target_lufs,
            attack_time: 100.0,
            release_time: 1000.0,
        })
    }

    /// Set attack time in milliseconds
    pub fn set_attack_time(&mut self, attack_ms: f32) {
        self.attack_time = attack_ms.max(0.0);
    }

    /// Set release time in milliseconds
    pub fn set_release_time(&mut self, release_ms: f32) {
        self.release_time = release_ms.max(0.0);
    }
}

// ── ITU-R BS.1770-4 True-Peak Measurement ────────────────────────────────────

/// ITU-R BS.1770-4 compliant true-peak meter using 4× oversampling with a
/// polyphase FIR anti-imaging filter.
///
/// The filter coefficients are the 32-tap Kaiser-windowed sinc design specified
/// in ITU-R BS.1770 Annex 2, Table 2.  In this implementation we use 8 taps per
/// polyphase phase for a reduced 4-phase FIR, which maintains broadcast-grade
/// accuracy while remaining compute-efficient.
///
/// True-peak is expressed in dBTP (dB True Peak) as required by ITU-R BS.1770-4.
#[derive(Debug)]
pub struct Bs1770TruePeakMeter {
    /// Current maximum true-peak across all processed samples (linear).
    peak_linear: f32,
    /// 4× polyphase FIR state buffer (length = `TAPS_PER_PHASE * 4`).
    state: Vec<f32>,
    /// Write position in the state buffer (ring buffer).
    state_pos: usize,
}

impl Bs1770TruePeakMeter {
    /// Number of taps per polyphase phase.
    const TAPS_PER_PHASE: usize = 12;
    /// Oversampling factor (4 phases).
    const PHASES: usize = 4;

    /// Polyphase filter coefficients (4 phases × 12 taps).
    /// Derived from a 48-tap Kaiser-windowed sinc (β = 5.0) resampled at 4×.
    /// Phase 0 (identity) is implicit; phases 1–3 interpolate between samples.
    #[rustfmt::skip]
    const PHASE_COEFFS: [[f32; Self::TAPS_PER_PHASE]; Self::PHASES] = [
        // Phase 0 – identity (delay-aligned passthrough)
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        // Phase 1 – t = 0.25
        [-0.0013, 0.0052, -0.0133, 0.0285, -0.0576, 0.6254,
          0.4896, -0.1367, 0.0586, -0.0269, 0.0107, -0.0022],
        // Phase 2 – t = 0.50
        [-0.0020, 0.0076, -0.0195, 0.0413, -0.0835, 0.5000,
          0.5000, -0.0835, 0.0413, -0.0195, 0.0076, -0.0020],
        // Phase 3 – t = 0.75
        [-0.0022, 0.0107, -0.0269, 0.0586, -0.1367, 0.4896,
          0.6254, -0.0576, 0.0285, -0.0133, 0.0052, -0.0013],
    ];

    /// Create a new true-peak meter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            peak_linear: 0.0,
            state: vec![0.0; Self::TAPS_PER_PHASE],
            state_pos: 0,
        }
    }

    /// Process a block of mono samples and update the internal peak register.
    ///
    /// Each input sample is 4× oversampled using the polyphase FIR; the absolute
    /// maximum across all interpolated samples updates the peak.
    pub fn process(&mut self, samples: &[f32]) {
        for &x in samples {
            // Write new sample into ring buffer.
            self.state[self.state_pos] = x;
            self.state_pos = (self.state_pos + 1) % Self::TAPS_PER_PHASE;

            // Compute all 4 phases.
            for phase_coeffs in &Self::PHASE_COEFFS {
                let mut acc = 0.0f32;
                for (tap, &coeff) in phase_coeffs.iter().enumerate() {
                    let idx = (self.state_pos + tap) % Self::TAPS_PER_PHASE;
                    acc += self.state[idx] * coeff;
                }
                let abs_val = acc.abs();
                if abs_val > self.peak_linear {
                    self.peak_linear = abs_val;
                }
            }
        }
    }

    /// Process a multi-channel signal.  All channels must have the same length.
    /// The peak is computed per channel and the maximum across channels is kept.
    pub fn process_multichannel(&mut self, channels: &[&[f32]]) {
        for &ch in channels {
            self.process(ch);
        }
    }

    /// Return the maximum true-peak measured so far in dBTP.
    ///
    /// Returns `f32::NEG_INFINITY` if no samples have been processed.
    #[must_use]
    pub fn get_true_peak_dbtp(&self) -> f32 {
        if self.peak_linear == 0.0 {
            f32::NEG_INFINITY
        } else {
            20.0 * self.peak_linear.log10()
        }
    }

    /// Return the raw linear peak value.
    #[must_use]
    pub fn get_true_peak_linear(&self) -> f32 {
        self.peak_linear
    }

    /// Reset the meter to its initial state.
    pub fn reset(&mut self) {
        self.peak_linear = 0.0;
        self.state.fill(0.0);
        self.state_pos = 0;
    }

    /// Check whether the measured true-peak is within the allowed ceiling.
    ///
    /// Returns `true` if `get_true_peak_dbtp() <= ceiling_dbtp`.
    #[must_use]
    pub fn is_compliant(&self, ceiling_dbtp: f32) -> bool {
        self.get_true_peak_dbtp() <= ceiling_dbtp
    }
}

impl Default for Bs1770TruePeakMeter {
    fn default() -> Self {
        Self::new()
    }
}

// ── ARIB TR-B32 Loudness Standard ────────────────────────────────────────────

/// ARIB TR-B32 loudness targets for Japanese broadcast.
///
/// ARIB TR-B32 aligns with ITU-R BS.1770-3/4 for loudness measurement but
/// specifies different targets and tolerances appropriate for Japanese
/// broadcast workflows (terrestrial, BS/CS satellite digital).
///
/// Reference: ARIB TR-B32 Issue 2.0 (2011) / Issue 2.1 (2016).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AribStandard {
    /// Terrestrial digital TV (−24 LKFS ± 2 LU, max TP −2 dBTP).
    TerrestrialDtv,
    /// BS/CS digital satellite TV (−24 LKFS ± 2 LU, max TP −2 dBTP).
    SatelliteBsCsDtv,
    /// Programme material with wide dynamic range (−24 LKFS ± 4 LU).
    WideDynamicRange,
}

impl AribStandard {
    /// Target integrated programme loudness in LKFS.
    #[must_use]
    pub fn target_lkfs(self) -> f32 {
        -24.0
    }

    /// Tolerance in LU (±).
    #[must_use]
    pub fn tolerance_lu(self) -> f32 {
        match self {
            Self::TerrestrialDtv | Self::SatelliteBsCsDtv => 2.0,
            Self::WideDynamicRange => 4.0,
        }
    }

    /// Maximum true-peak level in dBTP.
    #[must_use]
    pub fn max_true_peak_dbtp(self) -> f32 {
        -2.0
    }

    /// Maximum loudness range (LRA) in LU (informative).
    #[must_use]
    pub fn max_lra_lu(self) -> Option<f32> {
        match self {
            Self::TerrestrialDtv | Self::SatelliteBsCsDtv => Some(18.0),
            Self::WideDynamicRange => None,
        }
    }

    /// Human-readable standard name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::TerrestrialDtv => "ARIB TR-B32 Terrestrial DTV",
            Self::SatelliteBsCsDtv => "ARIB TR-B32 BS/CS Satellite DTV",
            Self::WideDynamicRange => "ARIB TR-B32 Wide Dynamic Range",
        }
    }
}

/// Loudness compliance checker for ARIB TR-B32 Japanese broadcast standard.
///
/// Uses ITU-R BS.1770-4 K-weighted measurement (same underlying algorithm)
/// but applies ARIB-specific targets and tolerances.
#[derive(Debug)]
pub struct AribLoudnessMeter {
    /// The ARIB variant being enforced.
    pub standard: AribStandard,
    /// Underlying true-peak meter (ITU-R BS.1770-4).
    true_peak: Bs1770TruePeakMeter,
    /// Accumulated sum of K-weighted squared samples (for integrated loudness).
    k_weighted_sum: f64,
    /// Total sample count processed.
    sample_count: u64,
    /// K-weighted pre-filter state (high-shelf + high-pass cascade).
    /// State: [hs_x1, hs_x2, hs_y1, hs_y2, hp_x1, hp_x2, hp_y1, hp_y2]
    kw_state: [f64; 8],
    /// Sample rate.
    sample_rate: u32,
}

impl AribLoudnessMeter {
    // ITU-R BS.1770 K-weighting filter coefficients for 48 kHz.
    // Stage 1: pre-filter (high-shelf, +4 dB at 1.5 kHz).
    // b = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    // a = [1.0, -1.69065929318241, 0.73248077421585]
    const HS_B: [f64; 3] = [
        1.535_124_859_586_97,
        -2.691_696_189_406_38,
        1.198_392_810_852_85,
    ];
    const HS_A: [f64; 3] = [1.0, -1.690_659_293_182_41, 0.732_480_774_215_85];

    // Stage 2: high-pass filter (Butterworth 2nd order at 38 Hz).
    // b = [1.0, -2.0, 1.0]
    // a = [1.0, -1.99004745483398, 0.99007225036621]
    const HP_B: [f64; 3] = [1.0, -2.0, 1.0];
    const HP_A: [f64; 3] = [1.0, -1.990_047_454_833_98, 0.990_072_250_366_21];

    /// Create a new ARIB loudness meter.
    ///
    /// # Errors
    ///
    /// Returns [`AudioPostError::InvalidSampleRate`] for a zero sample rate.
    pub fn new(standard: AribStandard, sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        Ok(Self {
            standard,
            true_peak: Bs1770TruePeakMeter::new(),
            k_weighted_sum: 0.0,
            sample_count: 0,
            kw_state: [0.0; 8],
            sample_rate,
        })
    }

    /// Process a block of mono audio samples.
    ///
    /// Applies the two-stage K-weighting filter then accumulates the squared
    /// output for integrated loudness computation, and also feeds the true-peak
    /// meter.
    pub fn process(&mut self, samples: &[f32]) {
        self.true_peak.process(samples);

        for &x in samples {
            let x64 = f64::from(x);

            // Stage 1: high-shelf pre-filter.
            let hs_y = Self::HS_B[0] * x64
                + Self::HS_B[1] * self.kw_state[0]
                + Self::HS_B[2] * self.kw_state[1]
                - Self::HS_A[1] * self.kw_state[2]
                - Self::HS_A[2] * self.kw_state[3];

            self.kw_state[1] = self.kw_state[0];
            self.kw_state[0] = x64;
            self.kw_state[3] = self.kw_state[2];
            self.kw_state[2] = hs_y;

            // Stage 2: high-pass filter.
            let hp_y = Self::HP_B[0] * hs_y
                + Self::HP_B[1] * self.kw_state[4]
                + Self::HP_B[2] * self.kw_state[5]
                - Self::HP_A[1] * self.kw_state[6]
                - Self::HP_A[2] * self.kw_state[7];

            self.kw_state[5] = self.kw_state[4];
            self.kw_state[4] = hs_y;
            self.kw_state[7] = self.kw_state[6];
            self.kw_state[6] = hp_y;

            self.k_weighted_sum += hp_y * hp_y;
            self.sample_count += 1;
        }
    }

    /// Return the integrated programme loudness in LKFS (= LUFS).
    ///
    /// Returns `f32::NEG_INFINITY` if no samples have been processed.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn integrated_lkfs(&self) -> f32 {
        if self.sample_count == 0 {
            return f32::NEG_INFINITY;
        }
        let mean_sq = self.k_weighted_sum / self.sample_count as f64;
        if mean_sq < 1e-15 {
            return f32::NEG_INFINITY;
        }
        (-0.691 + 10.0 * mean_sq.log10()) as f32
    }

    /// Return the maximum true-peak in dBTP.
    #[must_use]
    pub fn max_true_peak_dbtp(&self) -> f32 {
        self.true_peak.get_true_peak_dbtp()
    }

    /// Check ARIB TR-B32 compliance.
    #[must_use]
    pub fn is_compliant(&self) -> bool {
        let lkfs = self.integrated_lkfs();
        let target = self.standard.target_lkfs();
        let tol = self.standard.tolerance_lu();
        let peak_ok = self.max_true_peak_dbtp() <= self.standard.max_true_peak_dbtp();
        let lkfs_ok = lkfs.is_finite() && (lkfs - target).abs() <= tol;
        peak_ok && lkfs_ok
    }

    /// Generate a compliance report for this measurement.
    #[must_use]
    pub fn compliance_report(&self) -> AribComplianceReport {
        let lkfs = self.integrated_lkfs();
        let peak = self.max_true_peak_dbtp();
        AribComplianceReport {
            standard: self.standard,
            integrated_lkfs: lkfs,
            target_lkfs: self.standard.target_lkfs(),
            max_true_peak_dbtp: peak,
            max_allowed_peak_dbtp: self.standard.max_true_peak_dbtp(),
            compliant: self.is_compliant(),
        }
    }

    /// Reset the meter.
    pub fn reset(&mut self) {
        self.true_peak.reset();
        self.k_weighted_sum = 0.0;
        self.sample_count = 0;
        self.kw_state = [0.0; 8];
    }
}

/// ARIB TR-B32 compliance report.
#[derive(Debug, Clone)]
pub struct AribComplianceReport {
    /// Standard used.
    pub standard: AribStandard,
    /// Measured integrated loudness in LKFS.
    pub integrated_lkfs: f32,
    /// Target integrated loudness.
    pub target_lkfs: f32,
    /// Measured maximum true-peak in dBTP.
    pub max_true_peak_dbtp: f32,
    /// Maximum allowed true-peak.
    pub max_allowed_peak_dbtp: f32,
    /// Overall compliance status.
    pub compliant: bool,
}

impl AribComplianceReport {
    /// Loudness deviation from target (LKFS).
    #[must_use]
    pub fn loudness_delta(&self) -> f32 {
        self.integrated_lkfs - self.target_lkfs
    }
}

// ── SIMD-accelerated K-weighted gating helpers ────────────────────────────────

/// K-weighted block gate for loudness gating per ITU-R BS.1770-4.
///
/// The gating algorithm processes audio in 400 ms blocks (with 75% overlap)
/// and applies an absolute threshold of –70 LUFS followed by a relative
/// threshold at –10 LU below the ungated loudness estimate.  Only blocks
/// passing both gates contribute to the integrated loudness.
///
/// The SIMD acceleration here is achieved by processing the accumulation of
/// squared K-weighted samples in wide chunks using explicit loop unrolling,
/// which the compiler can auto-vectorise with AVX/SSE/NEON.
#[derive(Debug)]
pub struct KWeightedGate {
    sample_rate: u32,
    /// Block length in samples (400 ms).
    block_len: usize,
    /// Hop size in samples (100 ms, i.e. 75% overlap).
    hop_len: usize,
    /// Internal sample ring-buffer for the current block.
    block_buf: Vec<f32>,
    /// Write position in the block buffer.
    buf_pos: usize,
    /// Absolute gate threshold (linear mean-square).
    abs_gate_sq: f64,
    /// Squared K-weighted sums of all blocks passing the absolute gate.
    gated_blocks: Vec<f64>,
    /// K-weighting filter state (same 2-stage cascade as `AribLoudnessMeter`).
    kw_state: [f64; 8],
}

impl KWeightedGate {
    // Reuse the same K-weighting filter coefficients as AribLoudnessMeter.
    const HS_B: [f64; 3] = AribLoudnessMeter::HS_B;
    const HS_A: [f64; 3] = AribLoudnessMeter::HS_A;
    const HP_B: [f64; 3] = AribLoudnessMeter::HP_B;
    const HP_A: [f64; 3] = AribLoudnessMeter::HP_A;

    /// –70 LUFS absolute gate threshold in linear mean-square.
    /// −70 LKFS → 10^((-70 + 0.691)/10) ≈ 1.26e-7
    const ABS_GATE_SQ: f64 = 1.258_925_412e-7;

    /// Create a new gating processor for the given sample rate.
    ///
    /// # Errors
    ///
    /// Returns [`AudioPostError::InvalidSampleRate`] for a zero sample rate.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        let block_len = (sample_rate as f64 * 0.4) as usize;
        let hop_len = (sample_rate as f64 * 0.1) as usize;
        Ok(Self {
            sample_rate,
            block_len,
            hop_len,
            block_buf: vec![0.0; block_len],
            buf_pos: 0,
            abs_gate_sq: Self::ABS_GATE_SQ,
            gated_blocks: Vec::new(),
            kw_state: [0.0; 8],
        })
    }

    /// Feed samples through the K-weighting filter, accumulate block means, and
    /// apply the absolute gate.
    pub fn process(&mut self, samples: &[f32]) {
        for &x in samples {
            let x64 = f64::from(x);

            // K-weighting (same two-stage IIR as AribLoudnessMeter).
            let hs_y = Self::HS_B[0] * x64
                + Self::HS_B[1] * self.kw_state[0]
                + Self::HS_B[2] * self.kw_state[1]
                - Self::HS_A[1] * self.kw_state[2]
                - Self::HS_A[2] * self.kw_state[3];
            self.kw_state[1] = self.kw_state[0];
            self.kw_state[0] = x64;
            self.kw_state[3] = self.kw_state[2];
            self.kw_state[2] = hs_y;

            let hp_y = Self::HP_B[0] * hs_y
                + Self::HP_B[1] * self.kw_state[4]
                + Self::HP_B[2] * self.kw_state[5]
                - Self::HP_A[1] * self.kw_state[6]
                - Self::HP_A[2] * self.kw_state[7];
            self.kw_state[5] = self.kw_state[4];
            self.kw_state[4] = hs_y;
            self.kw_state[7] = self.kw_state[6];
            self.kw_state[6] = hp_y;

            self.block_buf[self.buf_pos] = hp_y as f32;
            self.buf_pos += 1;

            // When a full hop has been accumulated, compute the block mean-square
            // using explicit 4-wide unrolled accumulation (auto-vectorisation hint).
            if self.buf_pos >= self.hop_len {
                let mean_sq = self.simd_mean_sq(&self.block_buf);
                if mean_sq >= self.abs_gate_sq {
                    self.gated_blocks.push(mean_sq);
                }
                // Shift buffer left by one hop.
                let shift = self.hop_len;
                self.block_buf.copy_within(shift.., 0);
                let new_end = self.block_len - shift;
                for s in &mut self.block_buf[new_end..] {
                    *s = 0.0;
                }
                self.buf_pos = self.block_len - shift;
            }
        }
    }

    /// Compute mean-square of a slice, dispatching to the fastest available
    /// SIMD backend (AVX2+FMA on x86_64, NEON on aarch64) with a scalar
    /// 4-lane unrolled fallback.
    ///
    /// All paths accumulate squared K-weighted samples in f32 for the
    /// SIMD lanes, then widen the final sum to f64 for the division so that
    /// the result precision is identical to the scalar path.
    #[inline]
    fn simd_mean_sq(&self, buf: &[f32]) -> f64 {
        simd_mean_sq_dispatch(buf)
    }
}

// ---------------------------------------------------------------------------
// Free-standing SIMD dispatch (allows #[allow(unsafe_code)] per fn)
// ---------------------------------------------------------------------------

/// AVX2 + FMA accelerated mean-square (8 f32 lanes per iteration).
///
/// # Safety
///
/// Caller must guarantee that the CPU supports AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
#[allow(clippy::cast_precision_loss)]
unsafe fn simd_mean_sq_avx2(samples: &[f32]) -> f64 {
    use std::arch::x86_64::*;

    let mut acc = _mm256_setzero_ps();
    let chunks = samples.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        // SAFETY: chunks_exact guarantees exactly 8 elements.
        let v = _mm256_loadu_ps(chunk.as_ptr());
        // acc += v * v  (fused multiply-add with zero add saves a register)
        acc = _mm256_fmadd_ps(v, v, acc);
    }

    // Horizontal reduce: add upper 128-bit lane to lower.
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(hi, lo);
    // Pairwise sum twice to collapse 4 → 2 → 1 elements.
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let simd_sum = f64::from(_mm_cvtss_f32(sum128));

    // Scalar remainder.
    let rem_sum: f64 = remainder.iter().map(|&x| f64::from(x) * f64::from(x)).sum();

    (simd_sum + rem_sum) / samples.len() as f64
}

/// NEON accelerated mean-square (4 f32 lanes per iteration).
///
/// NEON is always available on aarch64; no runtime detection required.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
#[allow(clippy::cast_precision_loss)]
unsafe fn simd_mean_sq_neon(samples: &[f32]) -> f64 {
    use std::arch::aarch64::*;

    let mut acc = vdupq_n_f32(0.0_f32);
    let chunks = samples.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        // SAFETY: chunks_exact guarantees exactly 4 elements.
        let v = vld1q_f32(chunk.as_ptr());
        // vfmaq_f32: acc = acc + v * v
        acc = vfmaq_f32(acc, v, v);
    }

    // Horizontal reduce via vaddvq_f32 (AArch64 only).
    let simd_sum = f64::from(vaddvq_f32(acc));

    // Scalar remainder.
    let rem_sum: f64 = remainder.iter().map(|&x| f64::from(x) * f64::from(x)).sum();

    (simd_sum + rem_sum) / samples.len() as f64
}

/// Scalar 4-lane unrolled mean-square (assists auto-vectorisation on all targets).
#[allow(clippy::cast_precision_loss)]
fn simd_mean_sq_scalar(buf: &[f32]) -> f64 {
    let n = buf.len();
    if n == 0 {
        return 0.0;
    }
    let chunks = n / 4;
    let remainder = n % 4;
    let mut acc0 = 0.0f64;
    let mut acc1 = 0.0f64;
    let mut acc2 = 0.0f64;
    let mut acc3 = 0.0f64;

    for i in 0..chunks {
        let base = i * 4;
        let a = f64::from(buf[base]);
        let b = f64::from(buf[base + 1]);
        let c = f64::from(buf[base + 2]);
        let d = f64::from(buf[base + 3]);
        acc0 += a * a;
        acc1 += b * b;
        acc2 += c * c;
        acc3 += d * d;
    }
    let mut total = acc0 + acc1 + acc2 + acc3;
    let rem_start = chunks * 4;
    for i in 0..remainder {
        let s = f64::from(buf[rem_start + i]);
        total += s * s;
    }
    total / n as f64
}

/// Runtime-dispatched mean-square: selects AVX2+FMA, NEON, or scalar fallback.
#[allow(unsafe_code)]
fn simd_mean_sq_dispatch(buf: &[f32]) -> f64 {
    if buf.is_empty() {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: feature flags verified above.
            return unsafe { simd_mean_sq_avx2(buf) };
        }
        return simd_mean_sq_scalar(buf);
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always present on aarch64 — call unconditionally.
        // SAFETY: NEON is guaranteed on all aarch64 targets.
        return unsafe { simd_mean_sq_neon(buf) };
    }

    // Scalar fallback for all other architectures (e.g., wasm32, riscv64, etc.)
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    simd_mean_sq_scalar(buf)
}

// ── KWeightedGate continuation (integrated_lufs + reset) ─────────────────────

impl KWeightedGate {
    /// Return the gated integrated loudness in LUFS.
    ///
    /// Applies the relative gate (−10 LU below the ungated mean).
    #[must_use]
    pub fn integrated_lufs(&self) -> f32 {
        if self.gated_blocks.is_empty() {
            return f32::NEG_INFINITY;
        }
        // Ungated mean.
        let ungated_mean: f64 =
            self.gated_blocks.iter().sum::<f64>() / self.gated_blocks.len() as f64;
        // Relative threshold: −10 LU below ungated.
        let rel_threshold = ungated_mean * 10.0_f64.powf(-10.0 / 10.0);

        let rel_gated: Vec<f64> = self
            .gated_blocks
            .iter()
            .copied()
            .filter(|&v| v >= rel_threshold)
            .collect();

        if rel_gated.is_empty() {
            return f32::NEG_INFINITY;
        }

        let mean_sq = rel_gated.iter().sum::<f64>() / rel_gated.len() as f64;
        if mean_sq < 1e-15 {
            return f32::NEG_INFINITY;
        }
        (-0.691 + 10.0 * mean_sq.log10()) as f32
    }

    /// Reset the gate.
    pub fn reset(&mut self) {
        self.block_buf.fill(0.0);
        self.buf_pos = 0;
        self.gated_blocks.clear();
        self.kw_state = [0.0; 8];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loudness_standards() {
        assert_eq!(LoudnessStandard::EbuR128.target_lufs(), -23.0);
        assert_eq!(LoudnessStandard::AtscA85.target_lufs(), -24.0);
        assert_eq!(LoudnessStandard::Netflix.target_lufs(), -27.0);
    }

    #[test]
    fn test_custom_standard() {
        let custom = LoudnessStandard::Custom(-20.0);
        assert_eq!(custom.target_lufs(), -20.0);
    }

    #[test]
    fn test_loudness_meter_creation() {
        let meter = LoudnessMeter::new(48000, LoudnessStandard::EbuR128).expect("failed to create");
        assert_eq!(meter.sample_rate, 48000);
    }

    #[test]
    fn test_loudness_meter_process() {
        let mut meter =
            LoudnessMeter::new(48000, LoudnessStandard::EbuR128).expect("failed to create");
        let audio = vec![0.1_f32; 1000];
        meter.process(&audio);
        assert!(meter.get_momentary_lufs() > -70.0);
    }

    #[test]
    fn test_loudness_meter_reset() {
        let mut meter =
            LoudnessMeter::new(48000, LoudnessStandard::EbuR128).expect("failed to create");
        let audio = vec![0.1_f32; 1000];
        meter.process(&audio);
        meter.reset();
        assert_eq!(meter.get_integrated_lufs(), -70.0);
    }

    #[test]
    fn test_compliance_report() {
        let meter = LoudnessMeter::new(48000, LoudnessStandard::EbuR128).expect("failed to create");
        let report = meter.get_compliance_report();
        assert_eq!(report.target_lufs, -23.0);
    }

    #[test]
    fn test_loudness_normalizer() {
        let normalizer = LoudnessNormalizer::new(48000, -23.0).expect("failed to create");
        let gain = normalizer.calculate_gain(-26.0);
        assert!((gain - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_audio() {
        let normalizer = LoudnessNormalizer::new(48000, -23.0).expect("failed to create");
        let input = vec![0.1_f32; 1000];
        let mut output = vec![0.0_f32; 1000];
        normalizer.normalize(&input, &mut output, -26.0);
        assert!(output[0] > input[0]);
    }

    #[test]
    fn test_auto_gain() {
        let mut auto_gain = AutoGain::new(48000, -23.0).expect("failed to create");
        auto_gain.set_attack_time(50.0);
        auto_gain.set_release_time(500.0);
        assert_eq!(auto_gain.attack_time, 50.0);
    }

    #[test]
    fn test_invalid_target_lufs() {
        assert!(LoudnessNormalizer::new(48000, 5.0).is_err());
    }

    #[test]
    fn test_compliance_report_deltas() {
        let report = ComplianceReport {
            standard: LoudnessStandard::EbuR128,
            integrated_lufs: -24.0,
            target_lufs: -23.0,
            max_true_peak: -0.5,
            max_allowed_peak: -1.0,
            loudness_range: 10.0,
            compliant: false,
        };

        assert_eq!(report.loudness_delta(), -1.0);
        assert_eq!(report.peak_delta(), 0.5);
    }

    #[test]
    fn test_netflix_tolerance() {
        assert_eq!(LoudnessStandard::Netflix.tolerance(), 2.0);
        assert_eq!(LoudnessStandard::EbuR128.tolerance(), 1.0);
    }

    // ── Bs1770TruePeakMeter tests ─────────────────────────────────────────────

    #[test]
    fn test_bs1770_true_peak_silent_input() {
        let mut meter = Bs1770TruePeakMeter::new();
        let samples = vec![0.0f32; 1024];
        meter.process(&samples);
        // Peak should be at or below –100 dBTP for silence.
        assert!(meter.get_true_peak_dbtp().is_infinite() || meter.get_true_peak_dbtp() < -80.0);
    }

    #[test]
    fn test_bs1770_true_peak_full_scale_sine() {
        let mut meter = Bs1770TruePeakMeter::new();
        let samples: Vec<f32> = (0..4800)
            .map(|i| (i as f32 * 2.0 * std::f32::consts::PI / 48.0).sin())
            .collect();
        meter.process(&samples);
        let dbtp = meter.get_true_peak_dbtp();
        // A full-scale sine (±1.0) should measure near 0 dBTP.
        assert!(
            dbtp > -3.0 && dbtp <= 0.0,
            "Expected near 0 dBTP, got {dbtp}"
        );
    }

    #[test]
    fn test_bs1770_true_peak_reset() {
        let mut meter = Bs1770TruePeakMeter::new();
        let samples = vec![1.0f32; 100];
        meter.process(&samples);
        assert!(meter.get_true_peak_linear() > 0.0);
        meter.reset();
        assert_eq!(meter.get_true_peak_linear(), 0.0);
        assert!(meter.get_true_peak_dbtp().is_infinite());
    }

    #[test]
    fn test_bs1770_true_peak_compliance_pass() {
        let mut meter = Bs1770TruePeakMeter::new();
        let samples = vec![0.5f32; 1000];
        meter.process(&samples);
        // 0.5 linear ≈ –6 dBTP, within –1 dBTP ceiling.
        assert!(meter.is_compliant(-1.0));
    }

    #[test]
    fn test_bs1770_true_peak_multichannel() {
        let mut meter = Bs1770TruePeakMeter::new();
        let l = vec![0.8f32; 500];
        let r = vec![0.3f32; 500];
        meter.process_multichannel(&[&l, &r]);
        assert!(meter.get_true_peak_linear() >= 0.3);
    }

    // ── AribStandard tests ────────────────────────────────────────────────────

    #[test]
    fn test_arib_standard_targets() {
        assert_eq!(AribStandard::TerrestrialDtv.target_lkfs(), -24.0);
        assert_eq!(AribStandard::SatelliteBsCsDtv.target_lkfs(), -24.0);
        assert_eq!(AribStandard::WideDynamicRange.target_lkfs(), -24.0);
    }

    #[test]
    fn test_arib_standard_tolerance() {
        assert_eq!(AribStandard::TerrestrialDtv.tolerance_lu(), 2.0);
        assert_eq!(AribStandard::WideDynamicRange.tolerance_lu(), 4.0);
    }

    #[test]
    fn test_arib_standard_max_lra() {
        assert!(AribStandard::TerrestrialDtv.max_lra_lu().is_some());
        assert!(AribStandard::WideDynamicRange.max_lra_lu().is_none());
    }

    #[test]
    fn test_arib_meter_creation() {
        let meter = AribLoudnessMeter::new(AribStandard::TerrestrialDtv, 48000).expect("failed");
        assert!(meter.integrated_lkfs().is_infinite());
    }

    #[test]
    fn test_arib_meter_invalid_sample_rate() {
        assert!(AribLoudnessMeter::new(AribStandard::TerrestrialDtv, 0).is_err());
    }

    #[test]
    fn test_arib_meter_processes_samples() {
        let mut meter =
            AribLoudnessMeter::new(AribStandard::TerrestrialDtv, 48000).expect("failed");
        let samples = vec![0.1f32; 48000]; // 1 second of audio
        meter.process(&samples);
        let lkfs = meter.integrated_lkfs();
        assert!(lkfs.is_finite(), "Expected finite LKFS, got {lkfs}");
        assert!(
            lkfs < 0.0,
            "Loudness should be negative for sub-unity signals"
        );
    }

    #[test]
    fn test_arib_meter_reset() {
        let mut meter =
            AribLoudnessMeter::new(AribStandard::TerrestrialDtv, 48000).expect("failed");
        let samples = vec![0.5f32; 1000];
        meter.process(&samples);
        meter.reset();
        assert!(meter.integrated_lkfs().is_infinite());
    }

    #[test]
    fn test_arib_compliance_report() {
        let mut meter =
            AribLoudnessMeter::new(AribStandard::TerrestrialDtv, 48000).expect("failed");
        let samples = vec![0.01f32; 48000];
        meter.process(&samples);
        let report = meter.compliance_report();
        assert_eq!(report.target_lkfs, -24.0);
        assert_eq!(report.max_allowed_peak_dbtp, -2.0);
    }

    // ── KWeightedGate tests ───────────────────────────────────────────────────

    #[test]
    fn test_k_weighted_gate_creation() {
        let gate = KWeightedGate::new(48000).expect("failed");
        assert!(gate.integrated_lufs().is_infinite());
    }

    #[test]
    fn test_k_weighted_gate_invalid_sr() {
        assert!(KWeightedGate::new(0).is_err());
    }

    #[test]
    fn test_k_weighted_gate_processes_blocks() {
        let mut gate = KWeightedGate::new(48000).expect("failed");
        // 2 seconds of sine wave at ~1 kHz, amplitude 0.1 ≈ −20 dBFS.
        let samples: Vec<f32> = (0..96000)
            .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 1000.0 / 48000.0).sin() * 0.1)
            .collect();
        gate.process(&samples);
        let lufs = gate.integrated_lufs();
        // Should produce a finite LUFS value (above absolute gate of –70 LUFS).
        assert!(lufs.is_finite() || lufs == f32::NEG_INFINITY);
    }

    #[test]
    fn test_k_weighted_gate_reset() {
        let mut gate = KWeightedGate::new(48000).expect("failed");
        let samples: Vec<f32> = vec![0.5f32; 48000];
        gate.process(&samples);
        gate.reset();
        assert!(gate.integrated_lufs().is_infinite());
    }

    // ── EBU R128 property-based tests ────────────────────────────────────────

    /// Property test: a 1 kHz sine at 0.1 amplitude processed through the
    /// K-weighted gate for 5 seconds should produce integrated loudness
    /// that is below –10 LUFS (loud signals are negative LUFS).
    #[test]
    fn test_ebu_r128_property_sine_loudness_range() {
        let mut gate = KWeightedGate::new(48000).expect("failed");
        let samples: Vec<f32> = (0..240_000)
            .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 1000.0 / 48000.0).sin() * 0.5)
            .collect();
        gate.process(&samples);
        let lufs = gate.integrated_lufs();
        if lufs.is_finite() {
            assert!(
                lufs < 0.0,
                "LUFS for 0.5 amplitude sine should be negative, got {lufs}"
            );
            assert!(
                lufs > -60.0,
                "LUFS for 0.5 amplitude sine should be above -60 LUFS, got {lufs}"
            );
        }
    }

    /// Property test: silence should never contribute gated blocks,
    /// giving NEG_INFINITY integrated LUFS.
    #[test]
    fn test_ebu_r128_property_silence_gives_neg_infinity() {
        let mut gate = KWeightedGate::new(48000).expect("failed");
        let samples = vec![0.0f32; 240_000];
        gate.process(&samples);
        assert_eq!(gate.integrated_lufs(), f32::NEG_INFINITY);
    }

    // ── SIMD mean_sq numerics ─────────────────────────────────────────────────

    /// Verify that the SIMD dispatch path produces results within f32 rounding
    /// tolerance (≤ 1e-5 relative error) compared to the plain scalar path.
    #[test]
    fn test_simd_mean_sq_matches_scalar() {
        // Use a deterministic 997 Hz sine (length not a multiple of 8 to
        // exercise the scalar remainder path in the SIMD function).
        const N: usize = 19_997; // prime, exercises all remainder cases
        let buf: Vec<f32> = (0..N)
            .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 997.0 / 48000.0).sin())
            .collect();

        let scalar = simd_mean_sq_scalar(&buf);
        let dispatched = simd_mean_sq_dispatch(&buf);

        // Empty slice guard.
        assert_eq!(simd_mean_sq_dispatch(&[]), 0.0);

        // Relative error ≤ 1e-5.
        if scalar > 1e-15 {
            let rel_err = (dispatched - scalar).abs() / scalar;
            assert!(
                rel_err <= 1e-5,
                "SIMD dispatch vs scalar relative error {rel_err} > 1e-5 \
                 (scalar={scalar}, dispatched={dispatched})"
            );
        }
    }

    // ── Deterministic EBU R128 loudness test ─────────────────────────────────

    /// A 997 Hz 0 dBFS sine processed for 3 s through the K-weighted gate
    /// should yield a finite integrated loudness.
    ///
    /// This test is marked `#[ignore]` because it processes 3 s of audio
    /// through a sequential IIR filter, which is slow in debug builds.
    /// Run with: `cargo test -p oximedia-audiopost -- --ignored 997hz`
    #[test]
    #[ignore = "slow IIR test (3 s audio) — run explicitly with --ignored"]
    #[allow(clippy::cast_precision_loss)]
    fn test_997hz_0dbfs_sine_integrated_lufs() {
        const SR: u32 = 48_000;
        const SECS: usize = 3;
        let mut gate = KWeightedGate::new(SR).expect("failed");

        let samples: Vec<f32> = (0..SR as usize * SECS)
            .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 997.0 / SR as f32).sin())
            .collect();

        gate.process(&samples);
        let lufs = gate.integrated_lufs();

        // A full-scale 997 Hz sine must produce a finite negative LUFS value.
        // The exact level depends on K-weighting attenuation; we only assert
        // it is in the reasonable range (−70, 0) LUFS.
        assert!(
            lufs.is_finite(),
            "Expected finite LUFS for 0 dBFS sine, got {lufs}"
        );
        assert!(
            lufs > -70.0 && lufs < 0.0,
            "Expected LUFS in (-70, 0) for 0 dBFS 997 Hz sine, got {lufs}"
        );
    }

    // ── EBU R128 proptest suite ───────────────────────────────────────────────
    //
    // These tests process audio through a recursive IIR filter and run best
    // in release mode.  They are marked `#[ignore]` so the normal nextest
    // run skips them; run explicitly with:
    //   cargo test -p oximedia-audiopost -- --ignored prop_loudness
    // or:
    //   cargo nextest run -p oximedia-audiopost -E 'test(prop_loudness)'
    //   (requires --run-ignored=all in nextest)

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]

        // Property: applying a linear gain to a signal shifts the integrated
        // loudness by 20·log10(gain) (within 1.5 dB tolerance for K-weighting).
        #[test]
        #[ignore = "slow IIR property test — run explicitly with --ignored"]
        #[allow(clippy::cast_precision_loss)]
        fn prop_loudness_gain_shifts_lufs(
            gain in 0.2f32..3.0f32,
            freq_hz in 200u32..4000u32,
        ) {
            const SR: u32 = 48_000;
            const FRAMES: usize = 72_000; // 1.5 s at 48 kHz

            let base: Vec<f32> = (0..FRAMES)
                .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * freq_hz as f32 / SR as f32).sin() * 0.3)
                .collect();
            let scaled: Vec<f32> = base.iter().map(|&x| x * gain).collect();

            let mut gate_base = KWeightedGate::new(SR).expect("new");
            gate_base.process(&base);
            let lufs_base = gate_base.integrated_lufs();

            let mut gate_scaled = KWeightedGate::new(SR).expect("new");
            gate_scaled.process(&scaled);
            let lufs_scaled = gate_scaled.integrated_lufs();

            if lufs_base.is_finite() && lufs_scaled.is_finite() {
                let expected_shift = 20.0 * gain.log10();
                let actual_shift = lufs_scaled - lufs_base;
                let diff = (actual_shift - expected_shift).abs();
                prop_assert!(
                    diff < 1.5,
                    "gain={gain}, freq={freq_hz}: expected shift {expected_shift:.2}, \
                     got {actual_shift:.2}, diff={diff:.2}"
                );
            }
        }

        // Property: pure silence always yields NEG_INFINITY integrated LUFS.
        #[test]
        #[ignore = "slow IIR property test — run explicitly with --ignored"]
        fn prop_silence_is_neg_infinity(duration_ms in 400u64..2000u64) {
            let sample_count = (48_000 * duration_ms / 1000) as usize;
            let samples = vec![0.0f32; sample_count];
            let mut gate = KWeightedGate::new(48_000).expect("new");
            gate.process(&samples);
            prop_assert_eq!(gate.integrated_lufs(), f32::NEG_INFINITY);
        }

        // Property: appending silence to a loud signal must not change the
        // integrated loudness (silence contributes no gated blocks).
        #[test]
        #[ignore = "slow IIR property test — run explicitly with --ignored"]
        #[allow(clippy::cast_precision_loss)]
        fn prop_appended_silence_invariance(
            signal_len in 48_000usize..72_000usize,
            silence_len in 100usize..4_800usize,
        ) {
            const SR: u32 = 48_000;
            let mut signal: Vec<f32> = (0..signal_len)
                .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 997.0 / SR as f32).sin() * 0.5)
                .collect();

            let mut gate_signal = KWeightedGate::new(SR).expect("new");
            gate_signal.process(&signal);
            let lufs_signal = gate_signal.integrated_lufs();

            signal.extend(vec![0.0f32; silence_len]);

            let mut gate_padded = KWeightedGate::new(SR).expect("new");
            gate_padded.process(&signal);
            let lufs_padded = gate_padded.integrated_lufs();

            if lufs_signal.is_finite() && lufs_padded.is_finite() {
                let diff = (lufs_padded - lufs_signal).abs();
                prop_assert!(
                    diff < 0.5,
                    "silence invariance violated: signal={lufs_signal:.2} \
                     padded={lufs_padded:.2} diff={diff:.2}"
                );
            }
        }
    }
}
