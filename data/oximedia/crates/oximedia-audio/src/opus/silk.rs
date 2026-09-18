//! SILK mode decoder skeleton.
//!
//! SILK (Skype Internet Low-bitrate Codec) is the speech codec component of Opus.
//! It provides high-quality speech compression at low bitrates (6-40 kbps).
//!
//! # SILK Features
//!
//! - Optimized for speech (using LP synthesis)
//! - Variable bitrate with DTX support
//! - Frame sizes: 10ms, 20ms, 40ms, 60ms
//! - Bandwidths: narrowband (4kHz), medium (6kHz), wideband (8kHz)
//!
//! # Frame Structure
//!
//! Each SILK frame contains:
//! - LSF coefficients (vocal tract shape)
//! - Pitch parameters (for voiced speech)
//! - Excitation signal (residual)

#![forbid(unsafe_code)]

use crate::AudioError;

/// SILK bandwidth modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SilkBandwidth {
    /// Narrowband: 8kHz sample rate, 4kHz audio bandwidth.
    #[default]
    Narrow,
    /// Medium band: 12kHz sample rate, 6kHz audio bandwidth.
    Medium,
    /// Wideband: 16kHz sample rate, 8kHz audio bandwidth.
    Wide,
}

impl SilkBandwidth {
    /// Get sample rate for this bandwidth.
    #[must_use]
    pub fn sample_rate(self) -> u32 {
        match self {
            SilkBandwidth::Narrow => 8000,
            SilkBandwidth::Medium => 12000,
            SilkBandwidth::Wide => 16000,
        }
    }

    /// Get audio bandwidth in Hz.
    #[must_use]
    pub fn audio_bandwidth_hz(self) -> u32 {
        match self {
            SilkBandwidth::Narrow => 4000,
            SilkBandwidth::Medium => 6000,
            SilkBandwidth::Wide => 8000,
        }
    }

    /// Get number of LSF coefficients for this bandwidth.
    #[must_use]
    pub fn lsf_order(self) -> usize {
        match self {
            SilkBandwidth::Narrow => 10,
            SilkBandwidth::Medium => 12,
            SilkBandwidth::Wide => 16,
        }
    }

    /// Get number of subframes per frame.
    #[must_use]
    #[allow(dead_code)]
    pub fn subframes_per_10ms(self) -> usize {
        // SILK uses 5ms subframes for all bandwidths
        2
    }
}

/// Voice activity detection result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(dead_code)]
pub enum VoiceActivity {
    /// Silence (DTX mode).
    Silent,
    /// Unvoiced speech.
    #[default]
    Unvoiced,
    /// Voiced speech.
    Voiced,
}

/// SILK subframe data.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct SilkSubframe {
    /// Subframe index within frame (0-3 for 20ms frames).
    pub index: u8,
    /// Gain for this subframe (Q16 format).
    pub gain_q16: i32,
    /// Pitch lag in samples (for voiced subframes).
    pub pitch_lag: Option<u16>,
    /// Pitch gain (for voiced subframes).
    pub pitch_gain: Option<i16>,
    /// LTP (Long-Term Prediction) filter coefficients.
    pub ltp_coefficients: [i16; 5],
    /// Excitation samples (after inverse quantization).
    pub excitation: Vec<i16>,
    /// Number of samples in this subframe.
    pub sample_count: usize,
}

impl SilkSubframe {
    /// Create a new empty subframe.
    #[must_use]
    pub fn new(index: u8) -> Self {
        Self {
            index,
            ..Default::default()
        }
    }

    /// Check if this subframe is voiced.
    #[must_use]
    pub fn is_voiced(&self) -> bool {
        self.pitch_lag.is_some()
    }

    /// Get samples per subframe at given sample rate.
    #[must_use]
    pub fn samples_per_subframe(sample_rate: u32) -> usize {
        // SILK uses 5ms subframes
        (sample_rate as usize * 5) / 1000
    }
}

/// LSF (Line Spectral Frequency) coefficients.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct LsfCoefficients {
    /// Quantized LSF values (Q15 format).
    pub values: Vec<i16>,
    /// Interpolation factor for previous frame.
    pub interpolation_factor: u8,
}

impl LsfCoefficients {
    /// Create new LSF coefficients.
    #[must_use]
    pub fn new(order: usize) -> Self {
        Self {
            values: vec![0; order],
            interpolation_factor: 0,
        }
    }

    /// Get the order (number of coefficients).
    #[must_use]
    pub fn order(&self) -> usize {
        self.values.len()
    }

    /// Convert LSF to LPC coefficients.
    ///
    /// This performs the LSF to LPC conversion needed for synthesis.
    #[must_use]
    #[allow(dead_code)]
    pub fn to_lpc(&self) -> Vec<i16> {
        // Skeleton implementation - actual conversion is complex
        vec![0; self.values.len()]
    }
}

/// Pitch parameters for voiced frames.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct PitchParameters {
    /// Primary pitch lag in samples.
    pub lag: u16,
    /// Pitch contour (variation across subframes).
    pub contour_index: u8,
    /// Pitch gains for each subframe.
    pub gains: [i16; 4],
    /// Whether pitch uses prediction from previous frame.
    pub use_previous: bool,
}

impl PitchParameters {
    /// Get minimum pitch lag for bandwidth.
    #[must_use]
    pub fn min_lag(bandwidth: SilkBandwidth) -> u16 {
        match bandwidth {
            SilkBandwidth::Narrow => 16,
            SilkBandwidth::Medium => 24,
            SilkBandwidth::Wide => 32,
        }
    }

    /// Get maximum pitch lag for bandwidth.
    #[must_use]
    pub fn max_lag(bandwidth: SilkBandwidth) -> u16 {
        match bandwidth {
            SilkBandwidth::Narrow => 144,
            SilkBandwidth::Medium => 216,
            SilkBandwidth::Wide => 288,
        }
    }
}

/// SILK frame data.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct SilkFrame {
    /// Bandwidth mode.
    pub bandwidth: SilkBandwidth,
    /// Frame size in samples.
    pub frame_size: usize,
    /// Voice activity for this frame.
    pub voice_activity: VoiceActivity,
    /// LSF coefficients.
    pub lsf: LsfCoefficients,
    /// Pitch parameters (if voiced).
    pub pitch: Option<PitchParameters>,
    /// Subframes.
    pub subframes: Vec<SilkSubframe>,
    /// Number of channels (1 for mono, 2 for stereo).
    pub channels: u8,
    /// Side channel prediction gain (for stereo).
    pub stereo_prediction_gain: Option<i16>,
}

impl SilkFrame {
    /// Create a new SILK frame.
    #[must_use]
    pub fn new(bandwidth: SilkBandwidth, frame_size: usize) -> Self {
        let lsf = LsfCoefficients::new(bandwidth.lsf_order());
        Self {
            bandwidth,
            frame_size,
            lsf,
            channels: 1,
            ..Default::default()
        }
    }

    /// Get number of subframes.
    #[must_use]
    pub fn subframe_count(&self) -> usize {
        // 5ms per subframe
        let samples_per_subframe = SilkSubframe::samples_per_subframe(self.bandwidth.sample_rate());
        self.frame_size
            .checked_div(samples_per_subframe)
            .unwrap_or(0)
    }

    /// Check if this is a voiced frame.
    #[must_use]
    pub fn is_voiced(&self) -> bool {
        self.voice_activity == VoiceActivity::Voiced
    }

    /// Check if this is a stereo frame.
    #[must_use]
    pub fn is_stereo(&self) -> bool {
        self.channels == 2
    }
}

/// SILK decoder state.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct SilkDecoderState {
    /// Bandwidth mode.
    pub bandwidth: SilkBandwidth,
    /// Previous frame's LSF coefficients.
    pub prev_lsf: LsfCoefficients,
    /// Previous frame's pitch lag.
    pub prev_pitch_lag: u16,
    /// LPC synthesis filter state.
    pub lpc_state: Vec<i32>,
    /// Previous gain value.
    pub prev_gain: i32,
    /// Random seed for excitation generation.
    pub random_seed: u32,
    /// Number of consecutive lost frames.
    pub lost_frame_count: u32,
}

impl SilkDecoderState {
    /// Create new decoder state.
    #[must_use]
    pub fn new(bandwidth: SilkBandwidth) -> Self {
        let order = bandwidth.lsf_order();
        Self {
            bandwidth,
            prev_lsf: LsfCoefficients::new(order),
            lpc_state: vec![0; order],
            ..Default::default()
        }
    }

    /// Reset decoder state.
    pub fn reset(&mut self) {
        let order = self.bandwidth.lsf_order();
        self.prev_lsf = LsfCoefficients::new(order);
        self.prev_pitch_lag = 0;
        self.lpc_state.fill(0);
        self.prev_gain = 0;
        self.random_seed = 0;
        self.lost_frame_count = 0;
    }

    /// Handle packet loss concealment.
    #[allow(dead_code)]
    pub fn conceal_frame(&mut self) {
        self.lost_frame_count += 1;
        // Actual PLC would decay gain and randomize excitation
    }
}

// ─────────────────────────────── LSF → LPC ──────────────────────────────────

/// Convert Line Spectral Frequencies (LSF) to LPC coefficients.
///
/// # Algorithm (simplified)
///
/// LSF values are in Q15 format (range [0, π] mapped to [0, 32767]).
/// Actual conversion uses the Chebyshev polynomial method; we use the
/// simplified version: for each LSF pair, form a second-order polynomial and
/// convolve to get the full-order AR polynomial.
///
/// This returns coefficients in Q12 format for use in the LPC synthesis filter.
#[must_use]
fn lsf_to_lpc(lsf: &LsfCoefficients) -> Vec<i32> {
    let order = lsf.order();
    if order == 0 {
        return Vec::new();
    }

    // Convert Q15 LSF to floating-point radians.
    let freqs: Vec<f64> = lsf
        .values
        .iter()
        .map(|&v| f64::from(v) * std::f64::consts::PI / 32767.0)
        .collect();

    // Build symmetric and antisymmetric polynomials P and Q.
    // P: product of (1 - 2*cos(w[2k])*z^-1 + z^-2)   for k = 0, 2, 4, ...
    // Q: product of (1 - 2*cos(w[2k+1])*z^-1 + z^-2) for k = 0, 2, 4, ...
    let half = order / 2;

    let mut p = vec![1.0_f64; 1];
    let mut q = vec![1.0_f64; 1];

    for k in 0..half {
        let wp = freqs.get(2 * k).copied().unwrap_or(0.0);
        let wq = freqs.get(2 * k + 1).copied().unwrap_or(0.0);

        // Convolve p with (1 - 2*cos(wp)*z^-1 + z^-2).
        let old_p = p.clone();
        p = vec![0.0; old_p.len() + 2];
        for (i, &c) in old_p.iter().enumerate() {
            p[i] += c;
            p[i + 1] -= 2.0 * wp.cos() * c;
            p[i + 2] += c;
        }

        // Convolve q with (1 - 2*cos(wq)*z^-1 + z^-2).
        let old_q = q.clone();
        q = vec![0.0; old_q.len() + 2];
        for (i, &c) in old_q.iter().enumerate() {
            q[i] += c;
            q[i + 1] -= 2.0 * wq.cos() * c;
            q[i + 2] += c;
        }
    }

    // LPC = (P + Q) / 2, drop first and last coefficient (they're 1).
    let lpc_order = order;
    let mut lpc = vec![0.0_f64; lpc_order];
    for i in 0..lpc_order {
        let pi = p.get(i + 1).copied().unwrap_or(0.0);
        let qi = q.get(i + 1).copied().unwrap_or(0.0);
        lpc[i] = (pi + qi) / 2.0;
    }

    // Convert to Q12.
    lpc.iter().map(|&c| (c * 4096.0) as i32).collect()
}

// ─────────────────────────────── SILK Decoder ────────────────────────────────

/// Decode a `SilkFrame` to f32 PCM samples using LP synthesis.
///
/// # Algorithm
///
/// 1. Convert LSF coefficients to LPC coefficients.
/// 2. For each subframe:
///    a. Scale excitation by subframe gain.
///    b. Run the LPC synthesis filter.
///    c. Apply LTP (long-term prediction) for voiced frames.
/// 3. Upsample from SILK sample rate to 48kHz (simple linear interpolation).
/// 4. Convert to f32 and normalize.
///
/// # Errors
///
/// Returns `AudioError` if the frame is inconsistent.
pub fn decode_frame(
    frame: &SilkFrame,
    state: &mut SilkDecoderState,
) -> Result<Vec<f32>, AudioError> {
    let sample_rate = frame.bandwidth.sample_rate();
    let lpc = lsf_to_lpc(&frame.lsf);
    let order = lpc.len();

    // Total output samples at SILK sample rate.
    let total_samples = frame.frame_size;
    if total_samples == 0 {
        return Ok(Vec::new());
    }

    let mut output_i32 = vec![0i32; total_samples];
    let mut pos = 0usize;

    // Process each subframe.
    for subframe in &frame.subframes {
        let n_sub = subframe.sample_count;
        if n_sub == 0 || pos + n_sub > total_samples {
            break;
        }

        // Gain: Q16 → scale by 2^-16.
        let gain = subframe.gain_q16 as f64 / 65536.0;

        // Build excitation signal for this subframe.
        let mut excitation = vec![0i32; n_sub];

        if subframe.excitation.is_empty() {
            // No excitation provided: generate from PRNG (comfort noise).
            for e in excitation.iter_mut() {
                state.random_seed = state
                    .random_seed
                    .wrapping_mul(196_314_165)
                    .wrapping_add(907_633_515);
                *e = ((state.random_seed >> 16) as i16) as i32;
            }
        } else {
            // Use provided quantized excitation.
            for (i, e) in excitation.iter_mut().enumerate() {
                if i < subframe.excitation.len() {
                    *e = i32::from(subframe.excitation[i]);
                }
            }
        }

        // Apply LTP (long-term prediction) for voiced subframes.
        if let (Some(lag), Some(pitch_gain)) = (subframe.pitch_lag, subframe.pitch_gain) {
            let lag_usize = usize::from(lag);
            let pg = f64::from(pitch_gain) / 32768.0;
            // Gather previous output for LTP.
            for i in 0..n_sub {
                let global_i = pos + i;
                if global_i >= lag_usize {
                    let ltp_sample = output_i32[global_i - lag_usize];
                    excitation[i] += (ltp_sample as f64 * pg) as i32;
                } else if lag_usize > global_i {
                    // Use state's LPC buffer for samples before this frame.
                    let buf_idx = state.lpc_state.len().saturating_sub(lag_usize - global_i);
                    if buf_idx < state.lpc_state.len() {
                        let ltp_sample = state.lpc_state[buf_idx];
                        excitation[i] += (ltp_sample as f64 * pg) as i32;
                    }
                }
            }
        }

        // LPC synthesis filter: s[n] = e[n] * gain + sum_{k=1}^{order} a[k] * s[n-k].
        for i in 0..n_sub {
            let mut filtered = (excitation[i] as f64 * gain * 256.0) as i32;

            // Add LPC feedback.
            for (k, &ak) in lpc.iter().enumerate() {
                let prev_idx = (pos + i).saturating_sub(k + 1);
                let prev_sample = if pos + i > k {
                    output_i32[prev_idx]
                } else if state.lpc_state.len() > k - (pos + i) {
                    let buf_idx = state.lpc_state.len() - (k - (pos + i) + 1);
                    if buf_idx < state.lpc_state.len() {
                        state.lpc_state[buf_idx]
                    } else {
                        0
                    }
                } else {
                    0
                };
                // ak in Q12, prev_sample scaled. Use Q8 arithmetic: >> 12.
                filtered = filtered
                    .saturating_add(((i64::from(ak) * i64::from(prev_sample)) >> 12) as i32);
            }

            output_i32[pos + i] = filtered;
        }

        pos += n_sub;
    }

    // If no subframes provided, generate silence with comfort noise.
    if pos == 0 {
        for s in output_i32.iter_mut() {
            state.random_seed = state
                .random_seed
                .wrapping_mul(196_314_165)
                .wrapping_add(907_633_515);
            *s = ((state.random_seed >> 16) as i16) as i32;
        }
    }

    // Update LPC state buffer with the last `order` samples.
    state.lpc_state.resize(order.max(1), 0);
    let start = output_i32.len().saturating_sub(order);
    for (i, &s) in output_i32[start..].iter().enumerate() {
        state.lpc_state[i] = s;
    }

    // Upsample from SILK sample rate to 48kHz using linear interpolation.
    let out_48k = upsample_to_48k(&output_i32, sample_rate);

    // Normalize: SILK output is in Q8 integer scale.
    let norm = 1.0 / (32768.0 * 256.0);
    let samples_f32: Vec<f32> = out_48k
        .iter()
        .map(|&s| (s as f32 * norm as f32).clamp(-1.0, 1.0))
        .collect();

    Ok(samples_f32)
}

/// Upsample integer samples from `source_rate` to 48000 Hz using linear interpolation.
#[must_use]
fn upsample_to_48k(input: &[i32], source_rate: u32) -> Vec<i32> {
    if source_rate == 0 || input.is_empty() {
        return Vec::new();
    }
    if source_rate == 48000 {
        return input.to_vec();
    }

    let target_len = (input.len() as u64 * 48000 / source_rate as u64) as usize;
    let mut output = vec![0i32; target_len];

    for i in 0..target_len {
        let pos = i as f64 * source_rate as f64 / 48000.0;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(input.len() - 1);
        let frac = (pos - pos.floor()) as f32;
        let lo_val = input[lo] as f32;
        let hi_val = input[hi] as f32;
        output[i] = (lo_val + (hi_val - lo_val) * frac) as i32;
    }

    output
}

/// Encode f32 PCM samples into a `SilkFrame` (simplified analysis encoder).
///
/// This performs LPC analysis using the autocorrelation method (Levinson-Durbin)
/// and stores the quantized LPC → LSF coefficients plus the excitation residual.
///
/// # Errors
///
/// Returns `AudioError` if the sample count is zero.
pub fn encode_frame(
    samples: &[f32],
    state: &mut SilkDecoderState,
    bandwidth: SilkBandwidth,
) -> Result<SilkFrame, AudioError> {
    if samples.is_empty() {
        return Err(AudioError::InvalidParameter(
            "Cannot encode empty SILK frame".into(),
        ));
    }

    let order = bandwidth.lsf_order();
    let sample_rate = bandwidth.sample_rate();

    // Downsample from 48kHz if needed.
    let input_rate = 48000u32;
    let downsampled: Vec<f32> = if input_rate != sample_rate {
        let target_len =
            (samples.len() as u64 * u64::from(sample_rate) / u64::from(input_rate)) as usize;
        (0..target_len)
            .map(|i| {
                let pos = i as f64 * input_rate as f64 / sample_rate as f64;
                let lo = pos.floor() as usize;
                let hi = (lo + 1).min(samples.len() - 1);
                let frac = (pos - pos.floor()) as f32;
                samples[lo] + (samples[hi] - samples[lo]) * frac
            })
            .collect()
    } else {
        samples.to_vec()
    };

    let frame_size = downsampled.len();

    // ── LPC analysis via autocorrelation ────────────────────────────────
    // Compute autocorrelation for lags 0..order.
    let mut r = vec![0.0f64; order + 1];
    for lag in 0..=order {
        let mut sum = 0.0;
        for i in lag..frame_size {
            sum += downsampled[i] as f64 * downsampled[i - lag] as f64;
        }
        r[lag] = sum;
    }

    // Add white noise floor to improve numerical stability.
    r[0] *= 1.0001;

    // Levinson-Durbin recursion to get LPC coefficients.
    let lpc_f64 = levinson_durbin(&r, order);

    // Convert LPC to LSF frequencies (approximate).
    let lsf_values: Vec<i16> = lpc_f64
        .iter()
        .enumerate()
        .map(|(i, _c)| {
            // Simplified: distribute LSF evenly (actual conversion requires root finding).
            let freq = std::f64::consts::PI * (i + 1) as f64 / (order + 1) as f64;
            (freq / std::f64::consts::PI * 32767.0) as i16
        })
        .collect();

    let lsf = LsfCoefficients {
        values: lsf_values,
        interpolation_factor: 0,
    };

    // ── Compute residual (excitation) ────────────────────────────────────
    let mut subframes = Vec::new();
    let samples_per_subframe = SilkSubframe::samples_per_subframe(sample_rate);
    let num_subframes = frame_size.checked_div(samples_per_subframe).unwrap_or(0);

    for sf_idx in 0..num_subframes {
        let start = sf_idx * samples_per_subframe;
        let end = (start + samples_per_subframe).min(frame_size);
        let sf_samples = &downsampled[start..end];

        // Compute subframe gain as RMS.
        let rms = (sf_samples.iter().map(|&x| x as f64 * x as f64).sum::<f64>()
            / sf_samples.len() as f64)
            .sqrt();
        let gain_q16 = (rms * 65536.0).min(2_147_483_647.0) as i32;

        // Compute excitation residual.
        let mut excitation = Vec::with_capacity(sf_samples.len());
        for (i, &s) in sf_samples.iter().enumerate() {
            let global_i = start + i;
            let mut prediction = 0.0f64;
            for (k, &ak) in lpc_f64.iter().enumerate() {
                if global_i > k {
                    prediction += ak * downsampled[global_i - k - 1] as f64;
                }
            }
            let residual = s as f64 - prediction;
            excitation.push((residual * 128.0).clamp(-32767.0, 32767.0) as i16);
        }

        let mut subframe = SilkSubframe::new(sf_idx as u8);
        subframe.gain_q16 = gain_q16;
        subframe.sample_count = end - start;
        subframe.excitation = excitation;
        subframes.push(subframe);
    }

    // Update state.
    state.prev_lsf = lsf.clone();
    state.prev_gain = if let Some(sf) = subframes.first() {
        sf.gain_q16
    } else {
        0
    };

    Ok(SilkFrame {
        bandwidth,
        frame_size,
        voice_activity: VoiceActivity::Unvoiced,
        lsf,
        pitch: None,
        subframes,
        channels: 1,
        stereo_prediction_gain: None,
    })
}

/// Levinson-Durbin recursion: given autocorrelation vector `r` (length order+1),
/// return LPC coefficients a[1..order] as f64 (standard convention).
#[must_use]
fn levinson_durbin(r: &[f64], order: usize) -> Vec<f64> {
    if r.is_empty() || r[0].abs() < 1e-10 {
        return vec![0.0; order];
    }

    let mut a = vec![0.0f64; order];
    let mut e = r[0];

    for m in 0..order {
        // Reflection coefficient.
        let mut k = r[m + 1];
        for j in 0..m {
            k -= a[j] * r[m - j];
        }
        k /= e;

        // Update AR coefficients.
        let a_prev = a.clone();
        for j in 0..m {
            a[j] -= k * a_prev[m - 1 - j];
        }
        a[m] = k;

        // Update error.
        e *= 1.0 - k * k;
        if e < 1e-10 {
            break;
        }
    }

    a
}

/// Serialize a `SilkFrame` to bytes (simple binary format).
///
/// Format:
///   - 1 byte: bandwidth (0=Narrow, 1=Medium, 2=Wide)
///   - 2 bytes: frame_size (u16 LE)
///   - 1 byte: lsf_order
///   - lsf_order * 2 bytes: LSF values (i16 LE each)
///   - 1 byte: subframe_count
///   - Per subframe:
///     - 4 bytes: gain_q16 (i32 LE)
///     - 2 bytes: sample_count (u16 LE)
///     - 2 bytes: excitation_count (u16 LE)
///     - excitation_count * 2 bytes: excitation samples (i16 LE each)
///
/// # Errors
///
/// Currently infallible.
pub fn serialize_silk_frame(frame: &SilkFrame) -> Result<Vec<u8>, AudioError> {
    let mut out = Vec::new();

    let bw_byte = match frame.bandwidth {
        SilkBandwidth::Narrow => 0u8,
        SilkBandwidth::Medium => 1,
        SilkBandwidth::Wide => 2,
    };
    out.push(bw_byte);

    let fs = frame.frame_size as u16;
    out.push(fs as u8);
    out.push((fs >> 8) as u8);

    let order = frame.lsf.values.len() as u8;
    out.push(order);
    for &v in &frame.lsf.values {
        let vu = v as u16;
        out.push(vu as u8);
        out.push((vu >> 8) as u8);
    }

    let sf_count = frame.subframes.len() as u8;
    out.push(sf_count);
    for sf in &frame.subframes {
        let g = sf.gain_q16;
        out.push(g as u8);
        out.push((g >> 8) as u8);
        out.push((g >> 16) as u8);
        out.push((g >> 24) as u8);

        let sc = sf.sample_count as u16;
        out.push(sc as u8);
        out.push((sc >> 8) as u8);

        let ec = sf.excitation.len() as u16;
        out.push(ec as u8);
        out.push((ec >> 8) as u8);

        for &e in &sf.excitation {
            let eu = e as u16;
            out.push(eu as u8);
            out.push((eu >> 8) as u8);
        }
    }

    Ok(out)
}

/// Deserialize bytes back into a `SilkFrame`.
///
/// # Errors
///
/// Returns `AudioError` if data is truncated or malformed.
pub fn deserialize_silk_frame(data: &[u8]) -> Result<SilkFrame, AudioError> {
    if data.len() < 4 {
        return Err(AudioError::InvalidData("SILK frame too short".into()));
    }

    let mut pos = 0usize;

    let bandwidth = match data[pos] {
        0 => SilkBandwidth::Narrow,
        1 => SilkBandwidth::Medium,
        2 => SilkBandwidth::Wide,
        _ => SilkBandwidth::Narrow,
    };
    pos += 1;

    let frame_size = u16::from(data[pos]) | (u16::from(data[pos + 1]) << 8);
    pos += 2;

    let lsf_order = data[pos] as usize;
    pos += 1;

    if pos + lsf_order * 2 > data.len() {
        return Err(AudioError::InvalidData("SILK LSF data truncated".into()));
    }

    let mut lsf_values = Vec::with_capacity(lsf_order);
    for _ in 0..lsf_order {
        let v = (u16::from(data[pos]) | (u16::from(data[pos + 1]) << 8)) as i16;
        lsf_values.push(v);
        pos += 2;
    }

    if pos >= data.len() {
        return Err(AudioError::InvalidData(
            "SILK subframe count missing".into(),
        ));
    }

    let sf_count = data[pos] as usize;
    pos += 1;

    let mut subframes = Vec::with_capacity(sf_count);
    for sf_idx in 0..sf_count {
        if pos + 8 > data.len() {
            break;
        }
        let g = i32::from(data[pos])
            | (i32::from(data[pos + 1]) << 8)
            | (i32::from(data[pos + 2]) << 16)
            | (i32::from(data[pos + 3]) << 24);
        pos += 4;

        let sc = u16::from(data[pos]) | (u16::from(data[pos + 1]) << 8);
        pos += 2;

        let ec = (u16::from(data[pos]) | (u16::from(data[pos + 1]) << 8)) as usize;
        pos += 2;

        if pos + ec * 2 > data.len() {
            break;
        }

        let mut excitation = Vec::with_capacity(ec);
        for _ in 0..ec {
            let e = (u16::from(data[pos]) | (u16::from(data[pos + 1]) << 8)) as i16;
            excitation.push(e);
            pos += 2;
        }

        let mut sf = SilkSubframe::new(sf_idx as u8);
        sf.gain_q16 = g;
        sf.sample_count = sc as usize;
        sf.excitation = excitation;
        subframes.push(sf);
    }

    Ok(SilkFrame {
        bandwidth,
        frame_size: frame_size as usize,
        voice_activity: VoiceActivity::Unvoiced,
        lsf: LsfCoefficients {
            values: lsf_values,
            interpolation_factor: 0,
        },
        pitch: None,
        subframes,
        channels: 1,
        stereo_prediction_gain: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silk_bandwidth_sample_rate() {
        assert_eq!(SilkBandwidth::Narrow.sample_rate(), 8000);
        assert_eq!(SilkBandwidth::Medium.sample_rate(), 12000);
        assert_eq!(SilkBandwidth::Wide.sample_rate(), 16000);
    }

    #[test]
    fn test_silk_bandwidth_lsf_order() {
        assert_eq!(SilkBandwidth::Narrow.lsf_order(), 10);
        assert_eq!(SilkBandwidth::Medium.lsf_order(), 12);
        assert_eq!(SilkBandwidth::Wide.lsf_order(), 16);
    }

    #[test]
    fn test_silk_subframe() {
        let subframe = SilkSubframe::new(0);
        assert_eq!(subframe.index, 0);
        assert!(!subframe.is_voiced());
    }

    #[test]
    fn test_samples_per_subframe() {
        // 5ms at 8kHz = 40 samples
        assert_eq!(SilkSubframe::samples_per_subframe(8000), 40);
        // 5ms at 16kHz = 80 samples
        assert_eq!(SilkSubframe::samples_per_subframe(16000), 80);
    }

    #[test]
    fn test_silk_frame() {
        let frame = SilkFrame::new(SilkBandwidth::Wide, 320);
        assert_eq!(frame.bandwidth, SilkBandwidth::Wide);
        assert_eq!(frame.frame_size, 320);
        assert_eq!(frame.lsf.order(), 16);
    }

    #[test]
    fn test_silk_frame_subframe_count() {
        // 320 samples at 16kHz = 20ms = 4 subframes (5ms each)
        let frame = SilkFrame::new(SilkBandwidth::Wide, 320);
        assert_eq!(frame.subframe_count(), 4);
    }

    #[test]
    fn test_lsf_coefficients() {
        let lsf = LsfCoefficients::new(10);
        assert_eq!(lsf.order(), 10);
        assert_eq!(lsf.values.len(), 10);
    }

    #[test]
    fn test_pitch_lag_limits() {
        assert_eq!(PitchParameters::min_lag(SilkBandwidth::Narrow), 16);
        assert_eq!(PitchParameters::max_lag(SilkBandwidth::Narrow), 144);
        assert_eq!(PitchParameters::min_lag(SilkBandwidth::Wide), 32);
        assert_eq!(PitchParameters::max_lag(SilkBandwidth::Wide), 288);
    }

    #[test]
    fn test_silk_decoder_state() {
        let state = SilkDecoderState::new(SilkBandwidth::Medium);
        assert_eq!(state.bandwidth, SilkBandwidth::Medium);
        assert_eq!(state.prev_lsf.order(), 12);
        assert_eq!(state.lpc_state.len(), 12);
    }

    #[test]
    fn test_silk_decoder_state_reset() {
        let mut state = SilkDecoderState::new(SilkBandwidth::Wide);
        state.prev_gain = 1000;
        state.lost_frame_count = 5;
        state.reset();
        assert_eq!(state.prev_gain, 0);
        assert_eq!(state.lost_frame_count, 0);
    }

    #[test]
    fn test_voice_activity() {
        let mut frame = SilkFrame::new(SilkBandwidth::Narrow, 80);
        assert!(!frame.is_voiced());
        frame.voice_activity = VoiceActivity::Voiced;
        assert!(frame.is_voiced());
    }
}
