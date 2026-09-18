//! FFT implementation and window functions.

use oxifft::api::{Direction, Flags, Plan};
use oxifft::Complex;
use std::f64::consts::PI;

/// Window function type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WindowFunction {
    /// Rectangular window (no windowing).
    Rectangle,
    /// Hann window (raised cosine).
    #[default]
    Hann,
    /// Hamming window.
    Hamming,
    /// Blackman window.
    Blackman,
    /// Blackman-Harris window.
    BlackmanHarris,
    /// Kaiser window.
    Kaiser(u32),
    /// Tukey window (tapered cosine).
    Tukey(u32),
    /// Bartlett window (triangular).
    Bartlett,
    /// Welch window.
    Welch,
    /// Flat-top window.
    FlatTop,
}

impl WindowFunction {
    /// Generate window coefficients.
    #[must_use]
    pub fn generate(&self, size: usize) -> Vec<f64> {
        match self {
            Self::Rectangle => vec![1.0; size],
            Self::Hann => Self::generate_hann(size),
            Self::Hamming => Self::generate_hamming(size),
            Self::Blackman => Self::generate_blackman(size),
            Self::BlackmanHarris => Self::generate_blackman_harris(size),
            Self::Kaiser(beta) => Self::generate_kaiser(size, f64::from(*beta)),
            Self::Tukey(alpha) => Self::generate_tukey(size, f64::from(*alpha) / 100.0),
            Self::Bartlett => Self::generate_bartlett(size),
            Self::Welch => Self::generate_welch(size),
            Self::FlatTop => Self::generate_flattop(size),
        }
    }

    /// Hann window: 0.5 * (1 - cos(2*pi*n/(N-1))).
    fn generate_hann(size: usize) -> Vec<f64> {
        if size == 0 {
            return Vec::new();
        }
        if size == 1 {
            return vec![1.0];
        }

        (0..size)
            .map(|i| {
                let n = i as f64;
                let n_max = (size - 1) as f64;
                0.5 * (1.0 - (2.0 * PI * n / n_max).cos())
            })
            .collect()
    }

    /// Hamming window: 0.54 - 0.46 * cos(2*pi*n/(N-1)).
    fn generate_hamming(size: usize) -> Vec<f64> {
        if size == 0 {
            return Vec::new();
        }
        if size == 1 {
            return vec![1.0];
        }

        (0..size)
            .map(|i| {
                let n = i as f64;
                let n_max = (size - 1) as f64;
                0.54 - 0.46 * (2.0 * PI * n / n_max).cos()
            })
            .collect()
    }

    /// Blackman window.
    fn generate_blackman(size: usize) -> Vec<f64> {
        if size == 0 {
            return Vec::new();
        }
        if size == 1 {
            return vec![1.0];
        }

        const A0: f64 = 0.42;
        const A1: f64 = 0.5;
        const A2: f64 = 0.08;

        (0..size)
            .map(|i| {
                let n = i as f64;
                let n_max = (size - 1) as f64;
                A0 - A1 * (2.0 * PI * n / n_max).cos() + A2 * (4.0 * PI * n / n_max).cos()
            })
            .collect()
    }

    /// Blackman-Harris window.
    fn generate_blackman_harris(size: usize) -> Vec<f64> {
        if size == 0 {
            return Vec::new();
        }
        if size == 1 {
            return vec![1.0];
        }

        const A0: f64 = 0.35875;
        const A1: f64 = 0.48829;
        const A2: f64 = 0.14128;
        const A3: f64 = 0.01168;

        (0..size)
            .map(|i| {
                let n = i as f64;
                let n_max = (size - 1) as f64;
                A0 - A1 * (2.0 * PI * n / n_max).cos() + A2 * (4.0 * PI * n / n_max).cos()
                    - A3 * (6.0 * PI * n / n_max).cos()
            })
            .collect()
    }

    /// Kaiser window.
    fn generate_kaiser(size: usize, beta: f64) -> Vec<f64> {
        if size == 0 {
            return Vec::new();
        }
        if size == 1 {
            return vec![1.0];
        }

        let i0_beta = Self::bessel_i0(beta);
        let n_max = (size - 1) as f64;

        (0..size)
            .map(|i| {
                let n = i as f64;
                let arg = beta * (1.0 - ((2.0 * n / n_max) - 1.0).powi(2)).sqrt();
                Self::bessel_i0(arg) / i0_beta
            })
            .collect()
    }

    /// Modified Bessel function of the first kind, order 0.
    fn bessel_i0(x: f64) -> f64 {
        let mut sum = 1.0;
        let mut term = 1.0;
        let x_half = x / 2.0;

        for k in 1..=50 {
            term *= (x_half / k as f64).powi(2);
            sum += term;
            if term < 1e-12 * sum {
                break;
            }
        }

        sum
    }

    /// Tukey window (tapered cosine).
    fn generate_tukey(size: usize, alpha: f64) -> Vec<f64> {
        if size == 0 {
            return Vec::new();
        }
        if size == 1 {
            return vec![1.0];
        }

        let alpha = alpha.clamp(0.0, 1.0);
        let n_max = (size - 1) as f64;

        (0..size)
            .map(|i| {
                let n = i as f64;
                let ratio = n / n_max;

                if ratio < alpha / 2.0 {
                    0.5 * (1.0 + (2.0 * PI * ratio / alpha - PI).cos())
                } else if ratio > 1.0 - alpha / 2.0 {
                    0.5 * (1.0 + (2.0 * PI * (1.0 - ratio) / alpha - PI).cos())
                } else {
                    1.0
                }
            })
            .collect()
    }

    /// Bartlett window (triangular).
    fn generate_bartlett(size: usize) -> Vec<f64> {
        if size == 0 {
            return Vec::new();
        }
        if size == 1 {
            return vec![1.0];
        }

        let n_max = (size - 1) as f64;

        (0..size)
            .map(|i| {
                let n = i as f64;
                1.0 - ((n - n_max / 2.0) / (n_max / 2.0)).abs()
            })
            .collect()
    }

    /// Welch window.
    fn generate_welch(size: usize) -> Vec<f64> {
        if size == 0 {
            return Vec::new();
        }
        if size == 1 {
            return vec![1.0];
        }

        let n_max = (size - 1) as f64;

        (0..size)
            .map(|i| {
                let n = i as f64;
                let ratio = (n - n_max / 2.0) / (n_max / 2.0);
                1.0 - ratio * ratio
            })
            .collect()
    }

    /// Flat-top window.
    fn generate_flattop(size: usize) -> Vec<f64> {
        if size == 0 {
            return Vec::new();
        }
        if size == 1 {
            return vec![1.0];
        }

        const A0: f64 = 0.21557895;
        const A1: f64 = 0.41663158;
        const A2: f64 = 0.277263158;
        const A3: f64 = 0.083578947;
        const A4: f64 = 0.006947368;

        (0..size)
            .map(|i| {
                let n = i as f64;
                let n_max = (size - 1) as f64;
                A0 - A1 * (2.0 * PI * n / n_max).cos() + A2 * (4.0 * PI * n / n_max).cos()
                    - A3 * (6.0 * PI * n / n_max).cos()
                    + A4 * (8.0 * PI * n / n_max).cos()
            })
            .collect()
    }

    /// Get the coherent gain of the window (sum of coefficients divided by size).
    #[must_use]
    pub fn coherent_gain(&self, size: usize) -> f64 {
        if size == 0 {
            return 0.0;
        }
        let window = self.generate(size);
        window.iter().sum::<f64>() / size as f64
    }

    /// Get the equivalent noise bandwidth factor.
    #[must_use]
    pub fn enbw(&self, size: usize) -> f64 {
        if size == 0 {
            return 0.0;
        }
        let window = self.generate(size);
        let sum_squares: f64 = window.iter().map(|&x| x * x).sum();
        let sum: f64 = window.iter().sum();
        size as f64 * sum_squares / (sum * sum)
    }
}

/// FFT processor.
pub struct FftProcessor {
    fft_size: usize,
    window: Vec<f64>,
}

impl FftProcessor {
    /// Create a new FFT processor.
    #[must_use]
    pub fn new(fft_size: usize, window_fn: WindowFunction) -> Self {
        Self {
            fft_size,
            window: window_fn.generate(fft_size),
        }
    }

    /// Get FFT size.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.fft_size
    }

    /// Process audio samples and return frequency domain representation.
    pub fn process(&mut self, samples: &[f64]) -> Vec<Complex<f64>> {
        // Apply zero-padding if needed
        let input_size = samples.len().min(self.fft_size);

        // Apply window and convert to complex
        let buffer: Vec<Complex<f64>> = (0..self.fft_size)
            .map(|i| {
                if i < input_size {
                    Complex::new(samples[i] * self.window[i], 0.0)
                } else {
                    Complex::new(0.0, 0.0)
                }
            })
            .collect();

        // Perform FFT using OxiFFT plan API
        let mut output = vec![Complex::zero(); self.fft_size];
        if let Some(plan) = Plan::dft_1d(self.fft_size, Direction::Forward, Flags::ESTIMATE) {
            plan.execute(&buffer, &mut output);
        }

        output
    }

    /// Process and return magnitude spectrum.
    pub fn magnitude_spectrum(&mut self, samples: &[f64]) -> Vec<f64> {
        let spectrum = self.process(samples);
        spectrum.iter().map(|c| c.norm()).collect()
    }

    /// Process and return power spectrum (magnitude squared).
    pub fn power_spectrum(&mut self, samples: &[f64]) -> Vec<f64> {
        let spectrum = self.process(samples);
        spectrum.iter().map(|c| c.norm_sqr()).collect()
    }

    /// Process and return phase spectrum.
    pub fn phase_spectrum(&mut self, samples: &[f64]) -> Vec<f64> {
        let spectrum = self.process(samples);
        spectrum.iter().map(|c| c.arg()).collect()
    }

    /// Convert magnitude to decibels.
    #[must_use]
    pub fn to_db(magnitude: f64, reference: f64) -> f64 {
        if magnitude <= 0.0 {
            -100.0
        } else {
            20.0 * (magnitude / reference).log10()
        }
    }

    /// Convert power to decibels.
    #[must_use]
    pub fn power_to_db(power: f64, reference: f64) -> f64 {
        if power <= 0.0 {
            -100.0
        } else {
            10.0 * (power / reference).log10()
        }
    }

    /// Get frequency bin for a given index.
    #[must_use]
    pub fn bin_frequency(&self, bin: usize, sample_rate: f64) -> f64 {
        bin as f64 * sample_rate / self.fft_size as f64
    }

    /// Get bin index for a given frequency.
    #[must_use]
    pub fn frequency_to_bin(&self, frequency: f64, sample_rate: f64) -> usize {
        ((frequency * self.fft_size as f64) / sample_rate)
            .round()
            .max(0.0) as usize
    }

    /// Get the Nyquist bin index.
    #[must_use]
    pub const fn nyquist_bin(&self) -> usize {
        self.fft_size / 2
    }
}

/// Mel scale conversion utilities.
pub struct MelScale;

impl MelScale {
    /// Convert frequency (Hz) to mel scale.
    #[must_use]
    pub fn hz_to_mel(hz: f64) -> f64 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    /// Convert mel scale to frequency (Hz).
    #[must_use]
    pub fn mel_to_hz(mel: f64) -> f64 {
        700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
    }

    /// Create mel filterbank.
    #[must_use]
    pub fn create_filterbank(
        num_filters: usize,
        fft_size: usize,
        sample_rate: f64,
        min_freq: f64,
        max_freq: f64,
    ) -> Vec<Vec<f64>> {
        let min_mel = Self::hz_to_mel(min_freq);
        let max_mel = Self::hz_to_mel(max_freq);

        // Create equally spaced mel points
        let mel_points: Vec<f64> = (0..=num_filters + 1)
            .map(|i| min_mel + (max_mel - min_mel) * i as f64 / (num_filters + 1) as f64)
            .collect();

        // Convert back to Hz
        let hz_points: Vec<f64> = mel_points.iter().map(|&m| Self::mel_to_hz(m)).collect();

        // Convert to FFT bin numbers
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&f| ((fft_size + 1) as f64 * f / sample_rate).floor() as usize)
            .collect();

        // Create filterbank
        let mut filters = Vec::new();

        for i in 0..num_filters {
            let mut filter = vec![0.0; fft_size / 2 + 1];

            let start = bin_points[i];
            let center = bin_points[i + 1];
            let end = bin_points[i + 2];

            // Rising slope
            for k in start..center {
                if center > start {
                    filter[k] = (k - start) as f64 / (center - start) as f64;
                }
            }

            // Falling slope
            for k in center..end {
                if end > center {
                    filter[k] = (end - k) as f64 / (end - center) as f64;
                }
            }

            filters.push(filter);
        }

        filters
    }

    /// Apply mel filterbank to power spectrum.
    #[must_use]
    pub fn apply_filterbank(spectrum: &[f64], filterbank: &[Vec<f64>]) -> Vec<f64> {
        filterbank
            .iter()
            .map(|filter| {
                spectrum
                    .iter()
                    .zip(filter.iter())
                    .map(|(&s, &f)| s * f)
                    .sum()
            })
            .collect()
    }

    /// Compute Mel Frequency Cepstral Coefficients (MFCC).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_mfcc(mel_spectrum: &[f64], num_coeffs: usize) -> Vec<f64> {
        let n = mel_spectrum.len();

        (0..num_coeffs)
            .map(|i| {
                mel_spectrum
                    .iter()
                    .enumerate()
                    .map(|(k, &val)| {
                        let log_val = if val > 0.0 { val.ln() } else { -100.0 };
                        log_val * (PI * i as f64 * (k as f64 + 0.5) / n as f64).cos()
                    })
                    .sum()
            })
            .collect()
    }
}

/// Overlap-add processing for streaming FFT.
pub struct OverlapAdd {
    fft_size: usize,
    hop_size: usize,
    buffer: Vec<f64>,
}

impl OverlapAdd {
    /// Create a new overlap-add processor.
    #[must_use]
    pub fn new(fft_size: usize, hop_size: usize) -> Self {
        Self {
            fft_size,
            hop_size,
            buffer: Vec::new(),
        }
    }

    /// Add samples to buffer and return available frames.
    pub fn push(&mut self, samples: &[f64]) -> Vec<Vec<f64>> {
        self.buffer.extend_from_slice(samples);

        let mut frames = Vec::new();
        while self.buffer.len() >= self.fft_size {
            frames.push(self.buffer[..self.fft_size].to_vec());
            self.buffer.drain(..self.hop_size);
        }

        frames
    }

    /// Get remaining buffered samples.
    #[must_use]
    pub fn remaining(&self) -> &[f64] {
        &self.buffer
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Get buffer length.
    #[must_use]
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }
}

/// Compute a log-mel spectrogram from raw PCM f32 samples.
///
/// Returns a row-major `Vec<f32>` of shape `n_frames × n_mels`.  Frames are
/// extracted with centre padding (half a window of zeros prepended and
/// appended to `samples`) so that every sample participates in at least one
/// frame.  This matches the Whisper-compatible convention when called with
/// `n_fft = 400`, `hop_length = 160`, and `n_mels = 80`.
///
/// The function is intentionally self-contained: it uses the existing
/// [`FftProcessor`] (backed by `oxifft`) for the windowed power spectrum and
/// [`MelScale`] for the triangular filterbank.  No additional crate
/// dependencies are needed.
///
/// # Arguments
///
/// * `samples`     – mono PCM samples in the range `[-1.0, 1.0]`.
/// * `sample_rate` – sample rate in Hz; used to calibrate filterbank frequencies.
/// * `n_mels`      – number of mel filterbank channels (e.g. 80 for Whisper).
/// * `n_fft`       – FFT window size in samples (e.g. 400 for Whisper).
/// * `hop_length`  – hop between consecutive frames in samples (e.g. 160 for Whisper).
///
/// # Examples
///
/// ```rust
/// use oximedia_audio::spectrum::fft::compute_log_mel_spectrogram;
/// let samples: Vec<f32> = (0..16000)
///     .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 16000.0).sin())
///     .collect();
/// let log_mel = compute_log_mel_spectrogram(&samples, 16000, 80, 400, 160);
/// assert_eq!(log_mel.len(), 101 * 80);
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn compute_log_mel_spectrogram(
    samples: &[f32],
    sample_rate: u32,
    n_mels: usize,
    n_fft: usize,
    hop_length: usize,
) -> Vec<f32> {
    if samples.is_empty() || n_fft == 0 || hop_length == 0 || n_mels == 0 {
        return Vec::new();
    }

    // --- Centre-pad the signal: prepend and append n_fft/2 zeros --------
    let pad = n_fft / 2;
    let padded_len = samples.len() + 2 * pad;
    let mut padded = vec![0.0_f64; padded_len];
    for (i, &s) in samples.iter().enumerate() {
        padded[pad + i] = f64::from(s);
    }

    // --- Compute number of frames ----------------------------------------
    // n_frames = (padded_len - n_fft) / hop_length + 1
    let n_frames = if padded_len >= n_fft {
        (padded_len - n_fft) / hop_length + 1
    } else {
        return Vec::new();
    };

    // --- Build FftProcessor (Hann window applied internally) -------------
    let mut processor = FftProcessor::new(n_fft, WindowFunction::Hann);

    // --- Build mel filterbank once ----------------------------------------
    let sr_f64 = f64::from(sample_rate);
    let filterbank = MelScale::create_filterbank(n_mels, n_fft, sr_f64, 0.0, sr_f64 / 2.0);

    // --- Compute log-mel frame by frame -----------------------------------
    let mut output: Vec<f32> = Vec::with_capacity(n_frames * n_mels);

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_length;
        let end = (start + n_fft).min(padded_len);
        let frame_slice = &padded[start..end];

        // power_spectrum returns |X[k]|² for k in 0..n_fft (full spectrum).
        // Only the positive-frequency half (0..n_fft/2+1) is used; the
        // filterbank was built with exactly n_fft/2+1 bins.
        let power = processor.power_spectrum(frame_slice);
        let pos_power = &power[..n_fft / 2 + 1];

        // Apply mel filterbank.
        let mel_energies = MelScale::apply_filterbank(pos_power, &filterbank);

        // Log-compress: ln(energy + 1e-10).
        for energy in mel_energies {
            output.push((energy + 1e-10).ln() as f32);
        }
    }

    output
}

#[cfg(test)]
mod spectrogram_tests {
    use super::compute_log_mel_spectrogram;
    use std::f32::consts::PI;

    /// Basic smoke test: 1 second of 1 kHz sine at 16 kHz sample rate.
    ///
    /// With centre padding:
    ///   padded_len = 16000 + 2*(400/2) = 16400
    ///   n_frames   = (16400 - 400) / 160 + 1 = 16000/160 + 1 = 100 + 1 = 101
    ///   output.len() = 101 * 80 = 8080
    #[test]
    fn sine_1khz_produces_expected_shape() {
        let samples: Vec<f32> = (0..16000)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / 16000.0).sin())
            .collect();

        let log_mel = compute_log_mel_spectrogram(&samples, 16000, 80, 400, 160);

        // Shape check.
        assert_eq!(
            log_mel.len(),
            101 * 80,
            "expected 101 frames × 80 mels = 8080 elements, got {}",
            log_mel.len()
        );

        // Non-trivial signal: the maximum log-mel value must be above ln(0.01).
        let max_val = log_mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let threshold = (0.01_f32).ln();
        assert!(
            max_val > threshold,
            "max log-mel value {max_val:.4} is not above ln(0.01) = {threshold:.4}; signal may be lost"
        );
    }

    #[test]
    fn empty_input_returns_empty() {
        let out = compute_log_mel_spectrogram(&[], 16000, 80, 400, 160);
        assert!(out.is_empty());
    }

    #[test]
    fn zero_n_fft_returns_empty() {
        let samples = vec![0.0f32; 1000];
        let out = compute_log_mel_spectrogram(&samples, 16000, 80, 0, 160);
        assert!(out.is_empty());
    }
}
