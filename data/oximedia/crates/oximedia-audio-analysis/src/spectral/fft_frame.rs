//! Spectral analysis of audio signals.
//!
//! Provides FFT-based spectrum analysis, spectral centroid,
//! flatness, rolloff, and other spectral features.

use std::f64::consts::PI;

use oxifft::Complex;

/// FFT size for spectral analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftSize {
    /// 256-point FFT.
    N256 = 256,
    /// 512-point FFT.
    N512 = 512,
    /// 1024-point FFT.
    N1024 = 1024,
    /// 2048-point FFT.
    N2048 = 2048,
    /// 4096-point FFT.
    N4096 = 4096,
}

impl FftSize {
    /// Return the FFT size as a usize.
    #[must_use]
    pub fn size(self) -> usize {
        self as usize
    }

    /// Frequency resolution in Hz per bin: `sample_rate / fft_size`.
    #[must_use]
    pub fn freq_bin_hz(self, sample_rate: u32) -> f64 {
        f64::from(sample_rate) / self.size() as f64
    }
}

/// Window function applied before FFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFunction {
    /// Rectangular (no windowing).
    Rectangular,
    /// Hann window (cosine-squared).
    Hann,
    /// Hamming window (modified cosine).
    Hamming,
    /// Blackman window (three-term cosine).
    Blackman,
    /// Flat-top window (near-unity passband).
    FlatTop,
}

impl WindowFunction {
    /// Apply window to a buffer in-place.
    pub fn apply(self, buffer: &mut [f32]) {
        let n = buffer.len();
        if n == 0 {
            return;
        }
        match self {
            Self::Rectangular => {
                // No modification needed.
            }
            Self::Hann => {
                for (i, sample) in buffer.iter_mut().enumerate() {
                    let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
                    *sample *= w as f32;
                }
            }
            Self::Hamming => {
                for (i, sample) in buffer.iter_mut().enumerate() {
                    let w = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos();
                    *sample *= w as f32;
                }
            }
            Self::Blackman => {
                for (i, sample) in buffer.iter_mut().enumerate() {
                    let t = 2.0 * PI * i as f64 / (n - 1) as f64;
                    let w = 0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos();
                    *sample *= w as f32;
                }
            }
            Self::FlatTop => {
                for (i, sample) in buffer.iter_mut().enumerate() {
                    let t = 2.0 * PI * i as f64 / (n - 1) as f64;
                    let w = 0.215_578_95 - 0.416_631_58 * t.cos() + 0.277_263_16 * (2.0 * t).cos()
                        - 0.083_578_95 * (3.0 * t).cos()
                        + 0.006_947_36 * (4.0 * t).cos();
                    *sample *= w as f32;
                }
            }
        }
    }

    /// Return the name of this window function.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Rectangular => "Rectangular",
            Self::Hann => "Hann",
            Self::Hamming => "Hamming",
            Self::Blackman => "Blackman",
            Self::FlatTop => "FlatTop",
        }
    }

    /// Compute a single window coefficient for position `i` in a window of
    /// length `n`.  Used by [`FftSpectralAnalyzer`] to fill the pre-allocated
    /// scratch buffer without allocating an intermediate `Vec<f32>`.
    #[must_use]
    #[inline]
    pub fn coefficient(self, i: usize, n: usize) -> f64 {
        use std::f64::consts::PI;
        if n == 0 {
            return 1.0;
        }
        match self {
            Self::Rectangular => 1.0,
            Self::Hann => {
                let t = 2.0 * PI * i as f64 / (n - 1) as f64;
                0.5 * (1.0 - t.cos())
            }
            Self::Hamming => {
                let t = 2.0 * PI * i as f64 / (n - 1) as f64;
                0.54 - 0.46 * t.cos()
            }
            Self::Blackman => {
                let t = 2.0 * PI * i as f64 / (n - 1) as f64;
                0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos()
            }
            Self::FlatTop => {
                let t = 2.0 * PI * i as f64 / (n - 1) as f64;
                0.215_578_95 - 0.416_631_58 * t.cos() + 0.277_263_16 * (2.0 * t).cos()
                    - 0.083_578_95 * (3.0 * t).cos()
                    + 0.006_947_36 * (4.0 * t).cos()
            }
        }
    }
}

/// A single spectral frame (magnitude spectrum).
#[derive(Debug, Clone)]
pub struct SpectralFrame {
    /// Magnitude spectrum, length = `fft_size / 2 + 1`.
    pub magnitudes: Vec<f32>,
    /// Phase spectrum, length = `fft_size / 2 + 1`.
    pub phases: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// FFT size used for this frame.
    pub fft_size: usize,
    /// Hop size used for this frame.
    pub hop_size: usize,
    /// Sequential frame index (0-based).
    pub frame_index: u64,
}

impl SpectralFrame {
    /// Create a new empty spectral frame.
    #[must_use]
    pub fn new(fft_size: usize, sample_rate: u32) -> Self {
        let num_bins = fft_size / 2 + 1;
        Self {
            magnitudes: vec![0.0; num_bins],
            phases: vec![0.0; num_bins],
            sample_rate,
            fft_size,
            hop_size: fft_size / 4,
            frame_index: 0,
        }
    }

    /// Frequency (Hz) of bin `k`: `k * sample_rate / fft_size`.
    #[must_use]
    pub fn bin_frequency(&self, k: usize) -> f64 {
        k as f64 * f64::from(self.sample_rate) / self.fft_size as f64
    }

    /// Index of the bin closest to the given frequency.
    #[must_use]
    pub fn frequency_bin(&self, freq_hz: f64) -> usize {
        let bin = (freq_hz * self.fft_size as f64 / f64::from(self.sample_rate)).round() as usize;
        bin.min(self.magnitudes.len().saturating_sub(1))
    }

    /// Spectral centroid (weighted mean frequency in Hz).
    ///
    /// `centroid = sum(f[k] * m[k]) / sum(m[k])`
    #[must_use]
    pub fn centroid(&self) -> f64 {
        let mut weighted_sum = 0.0f64;
        let mut total_mag = 0.0f64;
        for (k, &mag) in self.magnitudes.iter().enumerate() {
            let freq = self.bin_frequency(k);
            weighted_sum += freq * f64::from(mag);
            total_mag += f64::from(mag);
        }
        if total_mag > 0.0 {
            weighted_sum / total_mag
        } else {
            0.0
        }
    }

    /// Spectral spread (weighted standard deviation around centroid, in Hz).
    #[must_use]
    pub fn spread(&self) -> f64 {
        let c = self.centroid();
        let mut weighted_sq = 0.0f64;
        let mut total_mag = 0.0f64;
        for (k, &mag) in self.magnitudes.iter().enumerate() {
            let freq = self.bin_frequency(k);
            let diff = freq - c;
            weighted_sq += diff * diff * f64::from(mag);
            total_mag += f64::from(mag);
        }
        if total_mag > 0.0 {
            (weighted_sq / total_mag).sqrt()
        } else {
            0.0
        }
    }

    /// Spectral flatness (Wiener entropy): `geometric_mean / arithmetic_mean`.
    ///
    /// Range \[0, 1\]: 0 = pure tone, 1 = white noise.
    #[must_use]
    pub fn flatness(&self) -> f64 {
        let n = self.magnitudes.len();
        if n == 0 {
            return 0.0;
        }

        let epsilon = 1e-10_f64;
        let log_sum: f64 = self
            .magnitudes
            .iter()
            .map(|&m| (f64::from(m) + epsilon).ln())
            .sum();
        let geometric_mean = (log_sum / n as f64).exp();

        let arithmetic_mean: f64 = self
            .magnitudes
            .iter()
            .map(|&m| f64::from(m) + epsilon)
            .sum::<f64>()
            / n as f64;

        if arithmetic_mean > 0.0 {
            (geometric_mean / arithmetic_mean).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Spectral rolloff: frequency below which `percentile * 100`% of energy is concentrated.
    ///
    /// `percentile` should be in \[0.0, 1.0\]; default is 0.85 (85th percentile).
    #[must_use]
    pub fn rolloff(&self, percentile: f64) -> f64 {
        let total: f64 = self
            .magnitudes
            .iter()
            .map(|&m| f64::from(m) * f64::from(m))
            .sum();
        if total <= 0.0 {
            return 0.0;
        }
        let threshold = percentile * total;
        let mut cumulative = 0.0f64;
        for (k, &mag) in self.magnitudes.iter().enumerate() {
            cumulative += f64::from(mag) * f64::from(mag);
            if cumulative >= threshold {
                return self.bin_frequency(k);
            }
        }
        self.bin_frequency(self.magnitudes.len().saturating_sub(1))
    }

    /// Energy in a frequency band `[low_hz, high_hz]`.
    #[must_use]
    pub fn band_energy(&self, low_hz: f64, high_hz: f64) -> f64 {
        let low_bin = self.frequency_bin(low_hz);
        let high_bin = self.frequency_bin(high_hz).min(self.magnitudes.len() - 1);
        self.magnitudes[low_bin..=high_bin]
            .iter()
            .map(|&m| f64::from(m) * f64::from(m))
            .sum()
    }

    /// RMS energy of the entire spectrum.
    #[must_use]
    pub fn total_energy(&self) -> f64 {
        let n = self.magnitudes.len();
        if n == 0 {
            return 0.0;
        }
        let sq_sum: f64 = self
            .magnitudes
            .iter()
            .map(|&m| f64::from(m) * f64::from(m))
            .sum();
        sq_sum
    }
}

/// Configuration for the spectral analyzer.
#[derive(Debug, Clone)]
pub struct SpectralConfig {
    /// FFT size.
    pub fft_size: FftSize,
    /// Hop size (default: `fft_size` / 4 for 75% overlap).
    pub hop_size: usize,
    /// Window function.
    pub window: WindowFunction,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

impl SpectralConfig {
    /// Create a new config with 2048-point Hann window and 75% overlap.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        let fft_size = FftSize::N2048;
        Self {
            fft_size,
            hop_size: fft_size.size() / 4,
            window: WindowFunction::Hann,
            sample_rate,
        }
    }

    /// Set FFT size (and reset `hop_size` to `fft_size` / 4).
    #[must_use]
    pub fn with_fft_size(mut self, size: FftSize) -> Self {
        self.fft_size = size;
        self.hop_size = size.size() / 4;
        self
    }

    /// Set window function.
    #[must_use]
    pub fn with_window(mut self, window: WindowFunction) -> Self {
        self.window = window;
        self
    }

    /// Set hop size explicitly.
    #[must_use]
    pub fn with_hop_size(mut self, hop: usize) -> Self {
        self.hop_size = hop;
        self
    }
}

/// Spectral analyzer that processes audio frames.
///
/// Pre-allocates a complex scratch buffer at construction time so that each
/// call to [`Self::process`] avoids heap allocation in the hot path.
pub struct FftSpectralAnalyzer {
    config: SpectralConfig,
    /// Ring buffer accumulating incoming samples.
    buffer: Vec<f32>,
    /// Pre-allocated complex scratch buffer (length = fft_size).
    fft_scratch: Vec<Complex<f64>>,
    frame_count: u64,
}

impl FftSpectralAnalyzer {
    /// Create a new spectral analyzer.
    ///
    /// Pre-allocates the FFT scratch buffer for the configured FFT size.
    #[must_use]
    pub fn new(config: SpectralConfig) -> Self {
        let fft_size = config.fft_size.size();
        Self {
            buffer: Vec::new(),
            fft_scratch: vec![Complex::new(0.0, 0.0); fft_size],
            frame_count: 0,
            config,
        }
    }

    /// Process a block of samples and emit spectral frames.
    ///
    /// Uses OxiFFT (pure-Rust FFT) for O(N log N) spectral analysis.  The
    /// internal scratch buffer is reused across calls to avoid per-frame
    /// allocation.
    pub fn process(&mut self, samples: &[f32]) -> Vec<SpectralFrame> {
        self.buffer.extend_from_slice(samples);
        let fft_size = self.config.fft_size.size();
        let hop_size = self.config.hop_size;
        let mut frames = Vec::new();

        while self.buffer.len() >= fft_size {
            // Copy samples into scratch buffer and apply window in-place.
            debug_assert_eq!(self.fft_scratch.len(), fft_size);
            for (i, dst) in self.fft_scratch.iter_mut().enumerate() {
                let s = f64::from(self.buffer[i]);
                let w = self.config.window.coefficient(i, fft_size);
                *dst = Complex::new(s * w, 0.0);
            }

            let fft_out = oxifft::fft(&self.fft_scratch);

            let num_bins = fft_size / 2 + 1;
            let magnitudes: Vec<f32> = fft_out[..num_bins]
                .iter()
                .map(|c| c.norm() as f32)
                .collect();
            let phases: Vec<f32> = fft_out[..num_bins]
                .iter()
                .map(|c| c.im.atan2(c.re) as f32)
                .collect();

            let mut frame = SpectralFrame::new(fft_size, self.config.sample_rate);
            frame.magnitudes = magnitudes;
            frame.phases = phases;
            frame.hop_size = hop_size;
            frame.frame_index = self.frame_count;
            self.frame_count += 1;

            frames.push(frame);

            if hop_size >= self.buffer.len() {
                self.buffer.clear();
            } else {
                self.buffer.drain(..hop_size);
            }
        }

        frames
    }

    /// Return a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &SpectralConfig {
        &self.config
    }

    /// Return the number of buffered samples waiting for a full frame.
    #[must_use]
    pub fn buffered_samples(&self) -> usize {
        self.buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_size_size() {
        assert_eq!(FftSize::N1024.size(), 1024);
    }

    #[test]
    fn test_fft_size_freq_bin() {
        let bin_hz = FftSize::N1024.freq_bin_hz(44100);
        // 44100 / 1024 ≈ 43.07
        assert!((bin_hz - 43.07).abs() < 0.1, "Got {bin_hz}");
    }

    #[test]
    fn test_window_hann_endpoints() {
        let mut buf = vec![1.0_f32; 128];
        WindowFunction::Hann.apply(&mut buf);
        assert!(
            buf[0].abs() < 1e-6,
            "First sample should be ~0, got {}",
            buf[0]
        );
        assert!(
            buf[127].abs() < 1e-6,
            "Last sample should be ~0, got {}",
            buf[127]
        );
    }

    #[test]
    fn test_window_rectangular() {
        let original = vec![0.5_f32; 64];
        let mut buf = original.clone();
        WindowFunction::Rectangular.apply(&mut buf);
        for (orig, windowed) in original.iter().zip(buf.iter()) {
            assert!(
                (orig - windowed).abs() < 1e-9,
                "Rectangular window should not modify samples"
            );
        }
    }

    #[test]
    fn test_spectral_frame_bin_frequency() {
        let frame = SpectralFrame::new(1024, 44100);
        let freq = frame.bin_frequency(1);
        // 1 * 44100 / 1024 ≈ 43.07 Hz
        assert!((freq - 43.07).abs() < 0.1, "Got {freq}");
    }

    #[test]
    fn test_spectral_frame_centroid_dc() {
        // Frame with energy only in bin 0 (DC) → centroid ≈ 0 Hz
        let mut frame = SpectralFrame::new(1024, 44100);
        frame.magnitudes[0] = 1.0;
        // All other bins remain 0.0
        let c = frame.centroid();
        assert!(
            c < 1.0,
            "Centroid with DC-only energy should be ~0 Hz, got {c}"
        );
    }

    #[test]
    fn test_spectral_flatness_range() {
        // Test with random-ish values - flatness should always be in [0, 1]
        let mut frame = SpectralFrame::new(256, 44100);
        for (i, m) in frame.magnitudes.iter_mut().enumerate() {
            *m = (i as f32 * 0.1).sin().abs();
        }
        let flatness = frame.flatness();
        assert!(
            (0.0..=1.0).contains(&flatness),
            "Flatness {flatness} not in [0, 1]"
        );

        // Also test with a single spike (very non-flat)
        let mut frame2 = SpectralFrame::new(256, 44100);
        frame2.magnitudes[10] = 1.0;
        let flatness2 = frame2.flatness();
        assert!(
            (0.0..=1.0).contains(&flatness2),
            "Flatness {flatness2} not in [0, 1]"
        );
    }

    #[test]
    fn test_spectral_rolloff_at_100_pct() {
        let mut frame = SpectralFrame::new(1024, 44100);
        // Uniform energy
        for m in frame.magnitudes.iter_mut() {
            *m = 1.0;
        }
        let rolloff = frame.rolloff(1.0);
        let max_freq = frame.bin_frequency(frame.magnitudes.len() - 1);
        assert!(
            (rolloff - max_freq).abs() < frame.bin_frequency(1) + 1.0,
            "100th percentile rolloff {rolloff} should be near max freq {max_freq}"
        );
    }

    #[test]
    fn test_band_energy_full_range() {
        let mut frame = SpectralFrame::new(1024, 44100);
        for m in frame.magnitudes.iter_mut() {
            *m = 1.0;
        }
        let total = frame.total_energy();
        let band = frame.band_energy(0.0, 22050.0);
        // Full range band energy should equal total energy
        assert!(
            (band - total).abs() / (total + 1e-10) < 1e-6,
            "band={band}, total={total}"
        );
    }

    #[test]
    fn test_analyzer_process_empty() {
        let config = SpectralConfig::new(44100);
        let mut analyzer = FftSpectralAnalyzer::new(config);
        let frames = analyzer.process(&[]);
        assert!(frames.is_empty(), "Empty input should yield no frames");
    }
}
