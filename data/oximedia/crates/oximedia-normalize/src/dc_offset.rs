//! DC offset removal via high-pass filtering and DC offset estimation/correction.
//!
//! DC offset is a constant bias added to an audio waveform that shifts it away from
//! zero.  Left uncorrected it wastes headroom and can cause clicks.  This module
//! provides:
//!
//! - A first-order IIR high-pass filter for continuous DC blocking.
//! - A DC offset estimator that measures the average (mean) of a block.
//! - A block-based corrector that subtracts the estimated offset.

#![allow(dead_code)]

/// Default high-pass cutoff frequency in Hz (very low, just above DC).
pub const DEFAULT_HPF_CUTOFF_HZ: f64 = 5.0;

/// Compute the IIR coefficient `alpha` for a first-order high-pass filter.
///
/// `alpha = 1 / (1 + 2π·fc·Ts)` where Ts = 1/sample_rate.
pub fn hpf_alpha(cutoff_hz: f64, sample_rate: f64) -> f64 {
    let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff_hz);
    let ts = 1.0 / sample_rate;
    rc / (rc + ts)
}

/// Estimate the DC offset (mean) of a mono sample block.
///
/// Returns 0.0 for an empty slice.
#[allow(clippy::cast_precision_loss)]
pub fn estimate_dc_offset(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().sum();
    sum / samples.len() as f64
}

/// Remove DC offset from a block by subtracting `offset` from every sample.
///
/// Modifies `samples` in-place.
pub fn remove_dc_offset(samples: &mut [f64], offset: f64) {
    for s in samples.iter_mut() {
        *s -= offset;
    }
}

/// Estimate and then remove the DC offset from a mono block in one step.
///
/// Returns the estimated offset that was removed.
pub fn auto_remove_dc(samples: &mut [f64]) -> f64 {
    let offset = estimate_dc_offset(samples);
    remove_dc_offset(samples, offset);
    offset
}

/// First-order high-pass IIR filter for continuous DC blocking.
///
/// Uses the difference equation:
///   `y[n] = alpha * (y[n-1] + x[n] - x[n-1])`
#[derive(Debug, Clone)]
pub struct DcBlockFilter {
    alpha: f64,
    /// Previous input sample.
    x_prev: f64,
    /// Previous output sample.
    y_prev: f64,
}

impl DcBlockFilter {
    /// Create a new filter.
    ///
    /// `cutoff_hz` is the -3 dB frequency; `sample_rate` is the audio sample rate.
    pub fn new(cutoff_hz: f64, sample_rate: f64) -> Self {
        Self {
            alpha: hpf_alpha(cutoff_hz, sample_rate),
            x_prev: 0.0,
            y_prev: 0.0,
        }
    }

    /// Create a filter with the default cutoff (5 Hz at the given sample rate).
    pub fn default_cutoff(sample_rate: f64) -> Self {
        Self::new(DEFAULT_HPF_CUTOFF_HZ, sample_rate)
    }

    /// Process a single sample.
    #[inline]
    pub fn process_sample(&mut self, x: f64) -> f64 {
        let y = self.alpha * (self.y_prev + x - self.x_prev);
        self.x_prev = x;
        self.y_prev = y;
        y
    }

    /// Process a block of samples in-place.
    pub fn process_block(&mut self, samples: &mut [f64]) {
        for s in samples.iter_mut() {
            *s = self.process_sample(*s);
        }
    }

    /// Reset the filter state.
    pub fn reset(&mut self) {
        self.x_prev = 0.0;
        self.y_prev = 0.0;
    }

    /// Return the current alpha coefficient.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

/// Multi-channel DC block filter (one independent filter per channel).
#[derive(Debug)]
pub struct MultiChannelDcBlockFilter {
    filters: Vec<DcBlockFilter>,
}

impl MultiChannelDcBlockFilter {
    /// Create a new multi-channel filter.
    pub fn new(channels: usize, cutoff_hz: f64, sample_rate: f64) -> Self {
        Self {
            filters: (0..channels)
                .map(|_| DcBlockFilter::new(cutoff_hz, sample_rate))
                .collect(),
        }
    }

    /// Process interleaved audio in-place.
    ///
    /// `samples.len()` must be a multiple of `channels`.
    pub fn process_interleaved(&mut self, samples: &mut [f64]) {
        let channels = self.filters.len();
        assert!(
            samples.len() % channels == 0,
            "samples.len() must be a multiple of channels"
        );
        for frame in samples.chunks_exact_mut(channels) {
            for (ch, s) in frame.iter_mut().enumerate() {
                *s = self.filters[ch].process_sample(*s);
            }
        }
    }

    /// Reset all channel filters.
    pub fn reset(&mut self) {
        for f in &mut self.filters {
            f.reset();
        }
    }

    /// Return the number of channels.
    pub fn channels(&self) -> usize {
        self.filters.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hpf_alpha_range() {
        let alpha = hpf_alpha(5.0, 48000.0);
        assert!(alpha > 0.0 && alpha < 1.0, "alpha={alpha} out of range");
    }

    #[test]
    fn test_hpf_alpha_higher_cutoff_lower_alpha() {
        let alpha_low = hpf_alpha(5.0, 48000.0);
        let alpha_high = hpf_alpha(100.0, 48000.0);
        assert!(
            alpha_low > alpha_high,
            "higher cutoff should yield lower alpha"
        );
    }

    #[test]
    fn test_estimate_dc_offset_zero_mean() {
        let samples = vec![-1.0, 1.0, -1.0, 1.0];
        let offset = estimate_dc_offset(&samples);
        assert!((offset - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_estimate_dc_offset_positive() {
        let samples = vec![0.5, 0.5, 0.5, 0.5];
        let offset = estimate_dc_offset(&samples);
        assert!((offset - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_estimate_dc_offset_empty() {
        assert_eq!(estimate_dc_offset(&[]), 0.0);
    }

    #[test]
    fn test_remove_dc_offset_applies_subtraction() {
        let mut samples = vec![0.5, 0.7, 0.3];
        remove_dc_offset(&mut samples, 0.5);
        assert!((samples[0] - 0.0).abs() < 1e-12);
        assert!((samples[1] - 0.2).abs() < 1e-12);
        assert!((samples[2] - (-0.2)).abs() < 1e-12);
    }

    #[test]
    fn test_auto_remove_dc_returns_estimated_offset() {
        let mut samples = vec![0.4; 100];
        let offset = auto_remove_dc(&mut samples);
        assert!((offset - 0.4).abs() < 1e-12);
        let remaining = estimate_dc_offset(&samples);
        assert!(remaining.abs() < 1e-12);
    }

    #[test]
    fn test_dc_block_filter_attenuates_dc() {
        let mut filter = DcBlockFilter::new(5.0, 48000.0);
        // Feed a constant DC signal and check that the steady-state output is near zero
        let mut samples = vec![1.0f64; 48000]; // 1 second of DC
        filter.process_block(&mut samples);
        // After one second the output should be close to 0
        let last = *samples.last().expect("should succeed in test");
        assert!(
            last.abs() < 0.01,
            "DC not sufficiently attenuated: {last:.6}"
        );
    }

    #[test]
    fn test_dc_block_filter_passes_ac() {
        let mut filter = DcBlockFilter::new(5.0, 48000.0);
        // 1 kHz sine should pass mostly unchanged (RMS should be preserved)
        let freq = 1000.0_f64;
        let sr = 48000.0_f64;
        let samples: Vec<f64> = (0..4800)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr).sin())
            .collect();
        let mut filtered = samples.clone();
        filter.process_block(&mut filtered);
        // Skip the initial transient (first 100 samples)
        let rms_in: f64 = {
            let s: f64 = samples[100..].iter().map(|x| x * x).sum();
            (s / (samples.len() - 100) as f64).sqrt()
        };
        let rms_out: f64 = {
            let s: f64 = filtered[100..].iter().map(|x| x * x).sum();
            (s / (filtered.len() - 100) as f64).sqrt()
        };
        // RMS should be within 1% of original
        assert!(
            (rms_out - rms_in).abs() / rms_in < 0.01,
            "AC signal attenuated too much: in={rms_in:.4} out={rms_out:.4}"
        );
    }

    #[test]
    fn test_dc_block_filter_reset() {
        let mut filter = DcBlockFilter::new(5.0, 48000.0);
        filter.process_sample(1.0);
        filter.reset();
        assert!((filter.x_prev - 0.0).abs() < 1e-12);
        assert!((filter.y_prev - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_dc_block_filter_alpha_value() {
        let filter = DcBlockFilter::new(5.0, 48000.0);
        let expected = hpf_alpha(5.0, 48000.0);
        assert!((filter.alpha() - expected).abs() < 1e-12);
    }

    #[test]
    fn test_multi_channel_filter_channel_count() {
        let filter = MultiChannelDcBlockFilter::new(4, 5.0, 48000.0);
        assert_eq!(filter.channels(), 4);
    }

    #[test]
    fn test_multi_channel_filter_attenuates_dc_stereo() {
        let mut filter = MultiChannelDcBlockFilter::new(2, 5.0, 48000.0);
        // Stereo interleaved, both channels at DC=0.8
        let mut samples = vec![0.8f64; 48000 * 2];
        filter.process_interleaved(&mut samples);
        let last_l = samples[samples.len() - 2];
        let last_r = samples[samples.len() - 1];
        assert!(last_l.abs() < 0.01, "L DC not removed: {last_l:.6}");
        assert!(last_r.abs() < 0.01, "R DC not removed: {last_r:.6}");
    }

    #[test]
    fn test_multi_channel_filter_reset() {
        let mut filter = MultiChannelDcBlockFilter::new(2, 5.0, 48000.0);
        let mut samples = vec![1.0f64; 200];
        filter.process_interleaved(&mut samples);
        filter.reset();
        // After reset, processing silence should yield silence
        let mut silence = vec![0.0f64; 10];
        filter.process_interleaved(&mut silence);
        assert!(silence.iter().all(|&s| s.abs() < 1e-12));
    }
}
