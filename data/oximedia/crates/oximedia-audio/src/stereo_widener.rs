#![allow(dead_code)]
//! Stereo field manipulation: widening, narrowing, and mid/side processing.
//!
//! This module provides tools for manipulating the stereo image of audio:
//!
//! - **Stereo widening/narrowing**: Adjust the perceived width of a stereo mix.
//! - **Mid/Side encoding/decoding**: Convert between L/R and M/S representations.
//! - **Haas effect widener**: Uses a short delay on one channel to create width.
//! - **Correlation monitoring**: Measure stereo phase correlation to detect
//!   mono-compatibility issues.
//! - **Balance control**: Pan the stereo image left or right.

/// Mid/Side encoded stereo pair.
#[derive(Debug, Clone, Copy)]
pub struct MidSide {
    /// Mid (center) component: (L + R) / 2.
    pub mid: f64,
    /// Side (difference) component: (L - R) / 2.
    pub side: f64,
}

impl MidSide {
    /// Creates a new mid/side pair.
    pub fn new(mid: f64, side: f64) -> Self {
        Self { mid, side }
    }

    /// Encodes a left/right stereo pair into mid/side.
    pub fn encode(left: f64, right: f64) -> Self {
        Self {
            mid: (left + right) * 0.5,
            side: (left - right) * 0.5,
        }
    }

    /// Decodes back to left/right stereo.
    pub fn decode(&self) -> (f64, f64) {
        let left = self.mid + self.side;
        let right = self.mid - self.side;
        (left, right)
    }
}

/// Configuration for the stereo widener.
#[derive(Debug, Clone)]
pub struct StereoWidenerConfig {
    /// Width factor: 0.0 = mono, 1.0 = original, >1.0 = widened.
    pub width: f64,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Haas delay in milliseconds (0.0 = disabled).
    pub haas_delay_ms: f64,
    /// Balance: -1.0 = full left, 0.0 = center, 1.0 = full right.
    pub balance: f64,
    /// Bass mono frequency in Hz (below this, signal is summed to mono).
    pub bass_mono_freq: f64,
}

impl Default for StereoWidenerConfig {
    fn default() -> Self {
        Self {
            width: 1.0,
            sample_rate: 48000.0,
            haas_delay_ms: 0.0,
            balance: 0.0,
            bass_mono_freq: 0.0,
        }
    }
}

/// A stereo widener processor.
#[derive(Debug, Clone)]
pub struct StereoWidener {
    /// Configuration.
    config: StereoWidenerConfig,
    /// Delay buffer for Haas effect (right channel delay).
    delay_buffer: Vec<f64>,
    /// Write position in the delay buffer.
    delay_pos: usize,
    /// Delay length in samples.
    delay_samples: usize,
    /// Simple low-pass filter state for bass mono (left).
    lp_state_l: f64,
    /// Simple low-pass filter state for bass mono (right).
    lp_state_r: f64,
    /// Low-pass filter coefficient.
    lp_coeff: f64,
}

impl StereoWidener {
    /// Creates a new stereo widener with the given configuration.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(config: StereoWidenerConfig) -> Self {
        let delay_samples = (config.haas_delay_ms * config.sample_rate / 1000.0) as usize;
        let delay_buffer = vec![0.0; delay_samples.max(1)];
        let lp_coeff = if config.bass_mono_freq > 0.0 && config.sample_rate > 0.0 {
            let rc = 1.0 / (2.0 * std::f64::consts::PI * config.bass_mono_freq);
            let dt = 1.0 / config.sample_rate;
            dt / (rc + dt)
        } else {
            0.0
        };
        Self {
            config,
            delay_buffer,
            delay_pos: 0,
            delay_samples,
            lp_state_l: 0.0,
            lp_state_r: 0.0,
            lp_coeff,
        }
    }

    /// Returns the current width setting.
    pub fn width(&self) -> f64 {
        self.config.width
    }

    /// Sets the width factor.
    pub fn set_width(&mut self, width: f64) {
        self.config.width = width;
    }

    /// Sets the balance.
    pub fn set_balance(&mut self, balance: f64) {
        self.config.balance = balance.clamp(-1.0, 1.0);
    }

    /// Resets the internal state.
    pub fn reset(&mut self) {
        for s in &mut self.delay_buffer {
            *s = 0.0;
        }
        self.delay_pos = 0;
        self.lp_state_l = 0.0;
        self.lp_state_r = 0.0;
    }

    /// Processes interleaved stereo samples [L, R, L, R, ...] in-place.
    pub fn process_interleaved(&mut self, samples: &mut [f64]) {
        let frame_count = samples.len() / 2;
        for i in 0..frame_count {
            let l = samples[i * 2];
            let r = samples[i * 2 + 1];
            let (out_l, out_r) = self.process_sample(l, r);
            samples[i * 2] = out_l;
            samples[i * 2 + 1] = out_r;
        }
    }

    /// Processes a single stereo sample pair.
    pub fn process_sample(&mut self, left: f64, right: f64) -> (f64, f64) {
        let mut l = left;
        let mut r = right;

        // Apply mid/side width
        let ms = MidSide::encode(l, r);
        let adjusted = MidSide::new(ms.mid, ms.side * self.config.width);
        let (wl, wr) = adjusted.decode();
        l = wl;
        r = wr;

        // Apply Haas delay to right channel
        if self.delay_samples > 0 {
            let delayed = self.delay_buffer[self.delay_pos];
            self.delay_buffer[self.delay_pos] = r;
            self.delay_pos = (self.delay_pos + 1) % self.delay_buffer.len();
            r = delayed;
        }

        // Bass mono: sum low frequencies to mono
        if self.lp_coeff > 0.0 {
            self.lp_state_l += self.lp_coeff * (l - self.lp_state_l);
            self.lp_state_r += self.lp_coeff * (r - self.lp_state_r);
            let bass_mono = (self.lp_state_l + self.lp_state_r) * 0.5;
            let hi_l = l - self.lp_state_l;
            let hi_r = r - self.lp_state_r;
            l = bass_mono + hi_l;
            r = bass_mono + hi_r;
        }

        // Apply balance
        if self.config.balance != 0.0 {
            let (bl, br) = apply_balance(l, r, self.config.balance);
            l = bl;
            r = br;
        }

        (l, r)
    }

    /// Processes separate left and right channel buffers.
    pub fn process_split(&mut self, left: &mut [f64], right: &mut [f64]) {
        let len = left.len().min(right.len());
        for i in 0..len {
            let (ol, or) = self.process_sample(left[i], right[i]);
            left[i] = ol;
            right[i] = or;
        }
    }
}

/// Applies a balance control to a stereo pair.
///
/// `balance` ranges from -1.0 (full left) to 1.0 (full right).
/// At 0.0, no change is applied.
pub fn apply_balance(left: f64, right: f64, balance: f64) -> (f64, f64) {
    let balance = balance.clamp(-1.0, 1.0);
    if balance <= 0.0 {
        // Pan left: attenuate right
        let gain_r = 1.0 + balance;
        (left, right * gain_r)
    } else {
        // Pan right: attenuate left
        let gain_l = 1.0 - balance;
        (left * gain_l, right)
    }
}

/// Computes the stereo correlation coefficient for a block of interleaved samples.
///
/// Returns a value from -1.0 (out of phase) through 0.0 (uncorrelated)
/// to 1.0 (perfectly correlated / mono).
#[allow(clippy::cast_precision_loss)]
pub fn stereo_correlation(interleaved: &[f64]) -> f64 {
    let frame_count = interleaved.len() / 2;
    if frame_count == 0 {
        return 0.0;
    }
    let mut sum_lr = 0.0;
    let mut sum_ll = 0.0;
    let mut sum_rr = 0.0;
    for i in 0..frame_count {
        let l = interleaved[i * 2];
        let r = interleaved[i * 2 + 1];
        sum_lr += l * r;
        sum_ll += l * l;
        sum_rr += r * r;
    }
    let denom = (sum_ll * sum_rr).sqrt();
    if denom > 0.0 {
        sum_lr / denom
    } else {
        0.0
    }
}

/// Converts interleaved stereo to mid/side representation in-place.
///
/// After conversion, even indices contain mid, odd indices contain side.
pub fn interleaved_to_mid_side(samples: &mut [f64]) {
    let frame_count = samples.len() / 2;
    for i in 0..frame_count {
        let l = samples[i * 2];
        let r = samples[i * 2 + 1];
        samples[i * 2] = (l + r) * 0.5;
        samples[i * 2 + 1] = (l - r) * 0.5;
    }
}

/// Converts mid/side representation back to interleaved stereo in-place.
pub fn mid_side_to_interleaved(samples: &mut [f64]) {
    let frame_count = samples.len() / 2;
    for i in 0..frame_count {
        let m = samples[i * 2];
        let s = samples[i * 2 + 1];
        samples[i * 2] = m + s;
        samples[i * 2 + 1] = m - s;
    }
}

/// Sums a stereo interleaved buffer to mono (in-place, fills left channel, zeros right).
pub fn sum_to_mono(samples: &mut [f64]) {
    let frame_count = samples.len() / 2;
    for i in 0..frame_count {
        let mono = (samples[i * 2] + samples[i * 2 + 1]) * 0.5;
        samples[i * 2] = mono;
        samples[i * 2 + 1] = mono;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mid_side_encode_decode_roundtrip() {
        let ms = MidSide::encode(0.8, 0.2);
        let (l, r) = ms.decode();
        assert!((l - 0.8).abs() < 1e-10);
        assert!((r - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_mid_side_mono_signal() {
        let ms = MidSide::encode(0.5, 0.5);
        assert!((ms.mid - 0.5).abs() < 1e-10);
        assert!(ms.side.abs() < 1e-10);
    }

    #[test]
    fn test_mid_side_pure_side() {
        let ms = MidSide::encode(0.5, -0.5);
        assert!(ms.mid.abs() < 1e-10);
        assert!((ms.side - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_default_config() {
        let cfg = StereoWidenerConfig::default();
        assert!((cfg.width - 1.0).abs() < 1e-10);
        assert!((cfg.balance - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_widener_unity_passthrough() {
        let config = StereoWidenerConfig {
            width: 1.0,
            haas_delay_ms: 0.0,
            balance: 0.0,
            bass_mono_freq: 0.0,
            sample_rate: 48000.0,
        };
        let mut widener = StereoWidener::new(config);
        let (l, r) = widener.process_sample(0.7, 0.3);
        assert!((l - 0.7).abs() < 1e-10);
        assert!((r - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_widener_mono_collapse() {
        let config = StereoWidenerConfig {
            width: 0.0,
            haas_delay_ms: 0.0,
            balance: 0.0,
            bass_mono_freq: 0.0,
            sample_rate: 48000.0,
        };
        let mut widener = StereoWidener::new(config);
        let (l, r) = widener.process_sample(1.0, 0.0);
        // Width 0 means side=0, so L=R=mid
        assert!((l - r).abs() < 1e-10);
    }

    #[test]
    fn test_widener_double_width() {
        let config = StereoWidenerConfig {
            width: 2.0,
            haas_delay_ms: 0.0,
            balance: 0.0,
            bass_mono_freq: 0.0,
            sample_rate: 48000.0,
        };
        let mut widener = StereoWidener::new(config);
        let (l, r) = widener.process_sample(0.8, 0.2);
        // Mid=0.5, Side=0.3*2=0.6 => L=1.1, R=-0.1
        assert!((l - 1.1).abs() < 1e-10);
        assert!((r - (-0.1)).abs() < 1e-10);
    }

    #[test]
    fn test_widener_reset() {
        let config = StereoWidenerConfig::default();
        let mut widener = StereoWidener::new(config);
        let _ = widener.process_sample(1.0, 1.0);
        widener.reset();
        assert!((widener.lp_state_l - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_balance_center() {
        let (l, r) = apply_balance(1.0, 1.0, 0.0);
        assert!((l - 1.0).abs() < 1e-10);
        assert!((r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_balance_full_left() {
        let (l, r) = apply_balance(1.0, 1.0, -1.0);
        assert!((l - 1.0).abs() < 1e-10);
        assert!((r - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_stereo_correlation_mono() {
        let samples = vec![1.0, 1.0, -1.0, -1.0, 0.5, 0.5];
        let corr = stereo_correlation(&samples);
        assert!((corr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_interleaved_ms_roundtrip() {
        let mut samples = vec![0.8, 0.2, -0.5, 0.5];
        let original = samples.clone();
        interleaved_to_mid_side(&mut samples);
        mid_side_to_interleaved(&mut samples);
        for (a, b) in samples.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sum_to_mono() {
        let mut samples = vec![0.8, 0.2, 0.6, 0.4];
        sum_to_mono(&mut samples);
        assert!((samples[0] - 0.5).abs() < 1e-10);
        assert!((samples[1] - 0.5).abs() < 1e-10);
        assert!((samples[2] - 0.5).abs() < 1e-10);
        assert!((samples[3] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_process_interleaved() {
        let config = StereoWidenerConfig {
            width: 1.0,
            haas_delay_ms: 0.0,
            balance: 0.0,
            bass_mono_freq: 0.0,
            sample_rate: 48000.0,
        };
        let mut widener = StereoWidener::new(config);
        let mut samples = vec![0.5, 0.5, -0.5, -0.5];
        widener.process_interleaved(&mut samples);
        assert!((samples[0] - 0.5).abs() < 1e-10);
        assert!((samples[1] - 0.5).abs() < 1e-10);
    }
}
