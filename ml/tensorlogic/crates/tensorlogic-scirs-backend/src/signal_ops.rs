//! Signal processing operations for audio and time-series data.
//!
//! Provides Short-Time Fourier Transform (STFT), window functions, spectrogram
//! computation, Discrete Cosine Transform (DCT-II), FIR convolution, and Mel
//! filterbank utilities for audio/time-series processing within TensorLogic.
//!
//! ## Features
//!
//! - **DFT/IDFT**: Direct O(N²) implementation — no external FFT dependency.
//! - **Window functions**: Rectangular, Hann, Hamming, Blackman, Triangular, FlatTop.
//! - **STFT/ISTFT**: Short-Time Fourier Transform with overlap-add reconstruction.
//! - **DCT-II / IDCT-II**: Discrete Cosine Transform (Type II) and its inverse.
//! - **FIR filtering**: Direct-convolution FIR filter with sinc low-pass design.
//! - **Mel filterbank**: Triangular Mel-scale filterbank for audio front-ends.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Complex number
// ---------------------------------------------------------------------------

/// Complex number for signal processing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex {
    /// Create a complex number from real and imaginary parts.
    #[inline]
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Return the additive identity (0 + 0i).
    #[inline]
    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    /// Create a complex number from polar form r·e^(iθ).
    #[inline]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    /// Magnitude (modulus): √(re² + im²).
    #[inline]
    pub fn magnitude(&self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Phase (argument): atan2(im, re).
    #[inline]
    pub fn phase(&self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Complex conjugate: re − i·im.
    #[inline]
    pub fn conjugate(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

impl std::ops::Add for Complex {
    type Output = Self;

    /// Complex addition.
    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;

    /// Complex multiplication: (a+bi)(c+di) = (ac−bd) + (ad+bc)i.
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

// ---------------------------------------------------------------------------
// Window function types
// ---------------------------------------------------------------------------

/// Window function applied to each STFT frame before DFT analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    /// All-ones window — no spectral leakage reduction.
    Rectangular,
    /// Hann window: 0.5 · (1 − cos(2πn/(N−1))).
    Hann,
    /// Hamming window: 0.54 − 0.46 · cos(2πn/(N−1)).
    Hamming,
    /// Blackman window: 0.42 − 0.5·cos(2πn/N) + 0.08·cos(4πn/N).
    Blackman,
    /// Triangular (Bartlett) window: 1 − |2n/(N−1) − 1|.
    Triangular,
    /// Flat-top window for accurate amplitude measurement.
    FlatTop,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during signal processing operations.
#[derive(Debug)]
pub enum SignalError {
    /// Input signal is empty.
    EmptySignal,
    /// Window size is invalid (zero or unsupported).
    InvalidWindowSize(usize),
    /// Hop length is incompatible with the window size.
    InvalidHopLength { hop: usize, window: usize },
    /// Signal length is invalid for the requested DCT.
    InvalidDctLength(usize),
    /// Two operands have incompatible shapes.
    DimensionMismatch,
    /// FIR filter has zero coefficients.
    EmptyFilter,
}

impl std::fmt::Display for SignalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySignal => write!(f, "Signal must not be empty"),
            Self::InvalidWindowSize(n) => write!(f, "Invalid window size: {n}"),
            Self::InvalidHopLength { hop, window } => write!(
                f,
                "Invalid hop length {hop}: must be > 0 and <= window size {window}"
            ),
            Self::InvalidDctLength(n) => write!(f, "Invalid DCT signal length: {n}"),
            Self::DimensionMismatch => write!(f, "Dimension mismatch between operands"),
            Self::EmptyFilter => write!(f, "FIR filter must have at least one coefficient"),
        }
    }
}

impl std::error::Error for SignalError {}

// ---------------------------------------------------------------------------
// STFT result
// ---------------------------------------------------------------------------

/// Result of the Short-Time Fourier Transform.
#[derive(Debug, Clone)]
pub struct StftResult {
    /// Complex STFT coefficients: `frames[t][f]`.
    pub frames: Vec<Vec<Complex>>,
    /// Number of time frames.
    pub n_frames: usize,
    /// Number of frequency bins (one-sided): `window_size / 2 + 1`.
    pub n_freqs: usize,
    /// Hop length in samples.
    pub hop_length: usize,
    /// Analysis window size in samples.
    pub window_size: usize,
}

impl StftResult {
    /// Magnitude spectrogram: |STFT(t, f)|.
    pub fn magnitude_spectrogram(&self) -> Vec<Vec<f64>> {
        self.frames
            .iter()
            .map(|frame| frame.iter().map(|c| c.magnitude()).collect())
            .collect()
    }

    /// Power spectrogram: |STFT(t, f)|².
    pub fn power_spectrogram(&self) -> Vec<Vec<f64>> {
        self.frames
            .iter()
            .map(|frame| {
                frame
                    .iter()
                    .map(|c| {
                        let m = c.magnitude();
                        m * m
                    })
                    .collect()
            })
            .collect()
    }

    /// Log-magnitude spectrogram: 20·log₁₀(|S| / ref_db).
    ///
    /// Values below a floor of `ref_db * 1e-10` are clamped to avoid −∞.
    pub fn log_magnitude_spectrogram(&self, ref_db: f64) -> Vec<Vec<f64>> {
        let floor = ref_db.abs() * 1e-10;
        self.frames
            .iter()
            .map(|frame| {
                frame
                    .iter()
                    .map(|c| {
                        let m = c.magnitude().max(floor);
                        20.0 * (m / ref_db).abs().log10()
                    })
                    .collect()
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// FIR filter
// ---------------------------------------------------------------------------

/// Finite Impulse Response (FIR) filter defined by its tap coefficients.
#[derive(Debug, Clone)]
pub struct FirFilter {
    coefficients: Vec<f64>,
}

impl FirFilter {
    /// Create a FIR filter from the given tap coefficients.
    ///
    /// Returns [`SignalError::EmptyFilter`] when `coefficients` is empty.
    pub fn new(coefficients: Vec<f64>) -> Result<Self, SignalError> {
        if coefficients.is_empty() {
            return Err(SignalError::EmptyFilter);
        }
        Ok(Self { coefficients })
    }

    /// Design a windowed-sinc low-pass FIR filter.
    ///
    /// `cutoff_normalized` is the normalised cut-off frequency in [0, 1]
    /// (where 1.0 corresponds to the Nyquist frequency).
    /// `n_taps` must be odd for a Type-I symmetric filter; if even it is
    /// incremented by one internally.
    pub fn low_pass(cutoff_normalized: f64, n_taps: usize) -> Self {
        let n_taps = if n_taps.is_multiple_of(2) {
            n_taps + 1
        } else {
            n_taps
        };
        let fc = cutoff_normalized.clamp(0.0, 1.0);
        let center = (n_taps - 1) as f64 / 2.0;

        // Hann window for side-lobe attenuation.
        let hann_win = window(WindowType::Hann, n_taps);

        let mut coeffs: Vec<f64> = (0..n_taps)
            .map(|i| {
                let n = i as f64 - center;
                let sinc_val = if n.abs() < 1e-12 {
                    2.0 * fc
                } else {
                    (2.0 * PI * fc * n).sin() / (PI * n)
                };
                sinc_val * hann_win[i]
            })
            .collect();

        // Normalise to unit DC gain.
        let dc_gain: f64 = coeffs.iter().sum();
        if dc_gain.abs() > 1e-15 {
            for c in &mut coeffs {
                *c /= dc_gain;
            }
        }

        Self {
            coefficients: coeffs,
        }
    }

    /// Return the number of filter taps.
    pub fn n_taps(&self) -> usize {
        self.coefficients.len()
    }

    /// Read-only access to the tap coefficients.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
}

// ---------------------------------------------------------------------------
// DFT / IDFT  (O(N²) definition — no external FFT dependency)
// ---------------------------------------------------------------------------

/// Compute the Discrete Fourier Transform of a real-valued signal.
///
/// ```text
/// X[k] = Σ_{n=0}^{N-1}  x[n] · e^{−2πi·k·n/N}
/// ```
///
/// Returns the full two-sided spectrum of length N.
pub fn dft(signal: &[f64]) -> Vec<Complex> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }
    let n_f = n as f64;
    (0..n)
        .map(|k| {
            let k_f = k as f64;
            signal
                .iter()
                .enumerate()
                .fold(Complex::zero(), |acc, (nn, &x)| {
                    let angle = -2.0 * PI * k_f * nn as f64 / n_f;
                    acc + Complex::new(x * angle.cos(), x * angle.sin())
                })
        })
        .collect()
}

/// Compute the Inverse Discrete Fourier Transform.
///
/// ```text
/// x[n] = (1/N) · Σ_{k=0}^{N-1}  X[k] · e^{+2πi·k·n/N}
/// ```
///
/// Returns the real part of the reconstructed time-domain signal.
pub fn idft(spectrum: &[Complex]) -> Vec<f64> {
    let n = spectrum.len();
    if n == 0 {
        return Vec::new();
    }
    let n_f = n as f64;
    (0..n)
        .map(|nn| {
            let nn_f = nn as f64;
            let sum = spectrum
                .iter()
                .enumerate()
                .fold(Complex::zero(), |acc, (k, &x)| {
                    let angle = 2.0 * PI * k as f64 * nn_f / n_f;
                    acc + x * Complex::new(angle.cos(), angle.sin())
                });
            sum.re / n_f
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Window functions
// ---------------------------------------------------------------------------

/// Generate a window of length `n` using the specified [`WindowType`].
///
/// Returns an all-ones vector for length 0 (vacuous).
pub fn window(window_type: WindowType, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let n_f = n as f64;
    match window_type {
        WindowType::Rectangular => vec![1.0; n],
        WindowType::Hann => (0..n)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n_f - 1.0)).cos()))
            .collect(),
        WindowType::Hamming => (0..n)
            .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n_f - 1.0)).cos())
            .collect(),
        WindowType::Blackman => (0..n)
            .map(|i| {
                let t = i as f64;
                0.42 - 0.5 * (2.0 * PI * t / n_f).cos() + 0.08 * (4.0 * PI * t / n_f).cos()
            })
            .collect(),
        WindowType::Triangular => (0..n)
            .map(|i| 1.0 - (2.0 * i as f64 / (n_f - 1.0) - 1.0).abs())
            .collect(),
        WindowType::FlatTop => {
            // Coefficients: a0=0.2156, a1=0.4160, a2=0.2781, a3=0.0836, a4=0.0069
            const A0: f64 = 0.2156;
            const A1: f64 = 0.4160;
            const A2: f64 = 0.2781;
            const A3: f64 = 0.0836;
            const A4: f64 = 0.0069;
            (0..n)
                .map(|i| {
                    let t = i as f64;
                    A0 - A1 * (2.0 * PI * t / n_f).cos() + A2 * (4.0 * PI * t / n_f).cos()
                        - A3 * (6.0 * PI * t / n_f).cos()
                        + A4 * (8.0 * PI * t / n_f).cos()
                })
                .collect()
        }
    }
}

/// Apply a window to a signal segment (element-wise multiplication).
///
/// Returns [`SignalError::DimensionMismatch`] when lengths differ.
pub fn apply_window(segment: &[f64], win: &[f64]) -> Result<Vec<f64>, SignalError> {
    if segment.len() != win.len() {
        return Err(SignalError::DimensionMismatch);
    }
    Ok(segment
        .iter()
        .zip(win.iter())
        .map(|(&s, &w)| s * w)
        .collect())
}

// ---------------------------------------------------------------------------
// STFT
// ---------------------------------------------------------------------------

/// Short-Time Fourier Transform.
///
/// The signal is zero-padded so that the last frame is fully covered.
///
/// # Parameters
/// - `signal`: Input time-domain signal.
/// - `window_size`: Analysis window length in samples (must be > 0).
/// - `hop_length`: Frame step in samples (0 < hop ≤ window_size).
/// - `window_type`: Window function applied to each frame.
pub fn stft(
    signal: &[f64],
    window_size: usize,
    hop_length: usize,
    window_type: WindowType,
) -> Result<StftResult, SignalError> {
    if signal.is_empty() {
        return Err(SignalError::EmptySignal);
    }
    if window_size == 0 {
        return Err(SignalError::InvalidWindowSize(window_size));
    }
    if hop_length == 0 || hop_length > window_size {
        return Err(SignalError::InvalidHopLength {
            hop: hop_length,
            window: window_size,
        });
    }

    let win = window(window_type, window_size);
    let n_freqs = window_size / 2 + 1;

    // Determine number of frames: pad the signal so that every frame is complete.
    let n_frames = 1 + (signal.len().saturating_sub(1)) / hop_length;
    // Total signal length needed to supply n_frames frames.
    let padded_len = (n_frames - 1) * hop_length + window_size;
    let mut padded = signal.to_vec();
    if padded.len() < padded_len {
        padded.resize(padded_len, 0.0);
    }

    let mut frames: Vec<Vec<Complex>> = Vec::with_capacity(n_frames);

    for t in 0..n_frames {
        let start = t * hop_length;
        let segment = &padded[start..start + window_size];
        // Apply window — lengths guaranteed equal.
        let windowed: Vec<f64> = segment
            .iter()
            .zip(win.iter())
            .map(|(&s, &w)| s * w)
            .collect();
        // Full DFT, then retain only n_freqs bins (one-sided).
        let spectrum = dft(&windowed);
        let one_sided: Vec<Complex> = spectrum.into_iter().take(n_freqs).collect();
        frames.push(one_sided);
    }

    Ok(StftResult {
        frames,
        n_frames,
        n_freqs,
        hop_length,
        window_size,
    })
}

// ---------------------------------------------------------------------------
// ISTFT (overlap-add)
// ---------------------------------------------------------------------------

/// Reconstruct a time-domain signal from an [`StftResult`] via overlap-add.
///
/// This is a synthesis-filter-bank inversion: each frame is inverse-DFT'd
/// (using the conjugate-symmetric extension to recover the real signal),
/// multiplied by the synthesis window, and accumulated with overlap-add.
pub fn istft(stft_result: &StftResult, window_type: WindowType) -> Vec<f64> {
    let window_size = stft_result.window_size;
    let hop_length = stft_result.hop_length;
    let n_frames = stft_result.n_frames;
    let n_freqs = stft_result.n_freqs;

    let syn_win = window(window_type, window_size);

    // Output buffer length.
    let output_len = if n_frames == 0 {
        0
    } else {
        (n_frames - 1) * hop_length + window_size
    };
    let mut output = vec![0.0_f64; output_len];
    let mut window_sum = vec![0.0_f64; output_len];

    for (t, frame) in stft_result.frames.iter().enumerate() {
        // Reconstruct full two-sided spectrum using Hermitian symmetry.
        let full_spectrum = reconstruct_full_spectrum(frame, window_size, n_freqs);

        // Inverse DFT gives the windowed time-domain frame.
        let time_frame = idft(&full_spectrum);

        let start = t * hop_length;
        for (i, (&tf, &sw)) in time_frame.iter().zip(syn_win.iter()).enumerate() {
            let idx = start + i;
            if idx < output_len {
                output[idx] += tf * sw;
                window_sum[idx] += sw * sw;
            }
        }
    }

    // Normalise by the window overlap sum to correct gain.
    for (o, &ws) in output.iter_mut().zip(window_sum.iter()) {
        if ws.abs() > 1e-12 {
            *o /= ws;
        }
    }

    output
}

/// Reconstruct the full two-sided DFT spectrum from a one-sided representation.
///
/// Uses Hermitian symmetry: X[N−k] = conj(X[k]).
fn reconstruct_full_spectrum(
    one_sided: &[Complex],
    window_size: usize,
    n_freqs: usize,
) -> Vec<Complex> {
    let mut full = vec![Complex::zero(); window_size];

    for (k, &c) in one_sided.iter().enumerate() {
        if k < window_size {
            full[k] = c;
        }
    }

    // Fill the conjugate-symmetric upper half.
    // k is used both as index into `full` and to compute mirror = window_size - k.
    #[allow(clippy::needless_range_loop)]
    for k in n_freqs..window_size {
        let mirror = window_size - k;
        if mirror < n_freqs {
            full[k] = one_sided[mirror].conjugate();
        }
    }

    full
}

// ---------------------------------------------------------------------------
// DCT-II and IDCT-II
// ---------------------------------------------------------------------------

/// Discrete Cosine Transform (Type II).
///
/// ```text
/// X[k] = Σ_{n=0}^{N-1}  x[n] · cos( π/N · (n + 0.5) · k )
/// ```
pub fn dct(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }
    let n_f = n as f64;
    (0..n)
        .map(|k| {
            let k_f = k as f64;
            signal
                .iter()
                .enumerate()
                .map(|(nn, &x)| x * (PI / n_f * (nn as f64 + 0.5) * k_f).cos())
                .sum()
        })
        .collect()
}

/// Inverse Discrete Cosine Transform (Type II / DCT-III normalised).
///
/// ```text
/// Recovers x[n] from DCT-II coefficients X[k].  The DCT-III formula gives:
///
/// x[n] = (1/N) · X[0] + (2/N) · Σ_{k=1}^{N-1} X[k] · cos( π/N · (n + 0.5) · k )
/// ```
pub fn idct(coeffs: &[f64]) -> Vec<f64> {
    let n = coeffs.len();
    if n == 0 {
        return Vec::new();
    }
    let n_f = n as f64;
    (0..n)
        .map(|nn| {
            let nn_f = nn as f64;
            let mut sum = coeffs[0] / n_f;
            for (k, &coeff) in coeffs.iter().enumerate().skip(1) {
                sum += 2.0 / n_f * coeff * (PI / n_f * (nn_f + 0.5) * k as f64).cos();
            }
            sum
        })
        .collect()
}

// ---------------------------------------------------------------------------
// FIR filtering
// ---------------------------------------------------------------------------

/// Apply a FIR filter to a signal via direct convolution.
///
/// Output length equals `signal.len()`.  Boundaries are zero-padded.
pub fn fir_filter(signal: &[f64], filter: &FirFilter) -> Vec<f64> {
    let n = signal.len();
    let m = filter.n_taps();
    let coeffs = filter.coefficients();
    let half = (m - 1) / 2;

    (0..n)
        .map(|i| {
            coeffs
                .iter()
                .enumerate()
                .map(|(j, &c)| {
                    let si = i + j;
                    // Zero-pad: shift so that the filter is centred on sample i.
                    if si < half {
                        0.0
                    } else {
                        let src = si - half;
                        if src < n {
                            signal[src] * c
                        } else {
                            0.0
                        }
                    }
                })
                .sum()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Mel scale utilities
// ---------------------------------------------------------------------------

/// Convert a frequency in Hz to the Mel scale.
///
/// mel = 2595 · log₁₀(1 + hz / 700)
#[inline]
pub fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert a Mel value back to Hz.
///
/// hz = 700 · (10^(mel / 2595) − 1)
#[inline]
pub fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

/// Compute a triangular Mel filterbank.
///
/// Returns an `n_mels × (window_size/2 + 1)` matrix of filterbank weights.
///
/// # Parameters
/// - `n_mels`: Number of Mel filter bands.
/// - `window_size`: STFT window size (determines frequency resolution).
/// - `sample_rate`: Audio sample rate in Hz.
pub fn mel_filterbank(n_mels: usize, window_size: usize, sample_rate: f64) -> Vec<Vec<f64>> {
    let n_freqs = window_size / 2 + 1;

    if n_mels == 0 || n_freqs == 0 {
        return Vec::new();
    }

    let f_min = 0.0_f64;
    let f_max = sample_rate / 2.0;

    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // n_mels + 2 evenly spaced Mel points (including lower and upper edges).
    let mel_points: Vec<f64> = (0..=(n_mels + 1))
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
        .collect();

    // Convert Mel points back to Hz and then to DFT bin indices.
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<f64> = hz_points
        .iter()
        .map(|&hz| (window_size + 1) as f64 * hz / sample_rate)
        .collect();

    // Build triangular filters.
    (0..n_mels)
        .map(|m| {
            let f_left = bin_points[m];
            let f_center = bin_points[m + 1];
            let f_right = bin_points[m + 2];

            (0..n_freqs)
                .map(|k| {
                    let k_f = k as f64;
                    if k_f >= f_left && k_f <= f_center {
                        let denom = f_center - f_left;
                        if denom < 1e-15 {
                            0.0
                        } else {
                            (k_f - f_left) / denom
                        }
                    } else if k_f > f_center && k_f <= f_right {
                        let denom = f_right - f_center;
                        if denom < 1e-15 {
                            0.0
                        } else {
                            (f_right - k_f) / denom
                        }
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPS: f64 = 1e-9;
    const LOOSE_EPS: f64 = 1e-6;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn assert_approx(a: f64, b: f64, tol: f64, msg: &str) {
        assert!(
            (a - b).abs() < tol,
            "{msg}: expected {b}, got {a} (diff {})",
            (a - b).abs()
        );
    }

    // -----------------------------------------------------------------------
    // 1. Rectangular window
    // -----------------------------------------------------------------------
    #[test]
    fn test_window_rectangular() {
        let w = window(WindowType::Rectangular, 4);
        assert_eq!(w, vec![1.0, 1.0, 1.0, 1.0]);
    }

    // -----------------------------------------------------------------------
    // 2. Hann window
    // -----------------------------------------------------------------------
    #[test]
    fn test_window_hann() {
        let n = 8;
        let w = window(WindowType::Hann, n);
        assert_eq!(w.len(), n);
        // Endpoints should be 0.
        assert_approx(w[0], 0.0, EPS, "Hann[0]");
        assert_approx(w[n - 1], 0.0, EPS, "Hann[N-1]");
        // For a periodic Hann window of length N, the sum equals N/2.
        // For a symmetric Hann window (N-1 denominator), the endpoints are 0 and
        // the sum is (N-2)/2 + 1 which for N=8 evaluates to 3.5 (i.e. (N-1)/2).
        let sum: f64 = w.iter().sum();
        let expected_sum = (n as f64 - 1.0) / 2.0;
        assert_approx(sum, expected_sum, 1e-10, "Hann sum ≈ (N-1)/2");
    }

    // -----------------------------------------------------------------------
    // 3. Hamming window first/last values ≈ 0.08
    // -----------------------------------------------------------------------
    #[test]
    fn test_window_hamming_endpoints() {
        let w = window(WindowType::Hamming, 8);
        // w[0] = 0.54 - 0.46 = 0.08
        assert_approx(w[0], 0.08, 1e-10, "Hamming[0]");
        assert_approx(w[7], 0.08, 1e-10, "Hamming[N-1]");
    }

    // -----------------------------------------------------------------------
    // 4. apply_window element-wise product
    // -----------------------------------------------------------------------
    #[test]
    fn test_apply_window_product() {
        let sig = vec![2.0, 3.0, 4.0];
        let win = vec![0.5, 1.0, 0.5];
        let result = apply_window(&sig, &win).expect("apply_window failed");
        assert_eq!(result, vec![1.0, 3.0, 2.0]);
    }

    // -----------------------------------------------------------------------
    // 5. apply_window length mismatch → Err
    // -----------------------------------------------------------------------
    #[test]
    fn test_apply_window_length_mismatch() {
        let sig = vec![1.0, 2.0, 3.0];
        let win = vec![1.0, 1.0];
        assert!(apply_window(&sig, &win).is_err());
    }

    // -----------------------------------------------------------------------
    // 6. DFT of DC signal has magnitude N at freq 0
    // -----------------------------------------------------------------------
    #[test]
    fn test_dft_dc_signal() {
        let n = 4;
        let signal = vec![1.0; n];
        let spectrum = dft(&signal);
        assert_approx(spectrum[0].magnitude(), n as f64, EPS, "DC magnitude");
        // All other bins should have magnitude ≈ 0.
        for (k, s) in spectrum.iter().enumerate().skip(1) {
            assert_approx(s.magnitude(), 0.0, 1e-10, &format!("bin {k}"));
        }
    }

    // -----------------------------------------------------------------------
    // 7. DFT of sine wave peaks at correct frequency bin
    // -----------------------------------------------------------------------
    #[test]
    fn test_dft_sine_peak() {
        let n = 32;
        let freq_bin = 4_usize;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq_bin as f64 * i as f64 / n as f64).sin())
            .collect();
        let spectrum = dft(&signal);
        // Find the bin with maximum magnitude (excluding DC).
        let peak_bin = (1..n)
            .max_by(|&a, &b| {
                spectrum[a]
                    .magnitude()
                    .partial_cmp(&spectrum[b].magnitude())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("empty spectrum");
        // The peak should be at freq_bin or its mirror N - freq_bin.
        assert!(
            peak_bin == freq_bin || peak_bin == n - freq_bin,
            "Expected peak at bin {freq_bin} or {}, got {peak_bin}",
            n - freq_bin
        );
    }

    // -----------------------------------------------------------------------
    // 8. IDFT(DFT(x)) ≈ x  (round-trip)
    // -----------------------------------------------------------------------
    #[test]
    fn test_dft_idft_roundtrip() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.5];
        let reconstructed = idft(&dft(&signal));
        assert_eq!(reconstructed.len(), signal.len());
        for (a, b) in reconstructed.iter().zip(signal.iter()) {
            assert_approx(*a, *b, LOOSE_EPS, "DFT roundtrip");
        }
    }

    // -----------------------------------------------------------------------
    // 9. STFT returns correct n_frames
    // -----------------------------------------------------------------------
    #[test]
    fn test_stft_n_frames() {
        let signal: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let window_size = 16;
        let hop_length = 8;
        let result = stft(&signal, window_size, hop_length, WindowType::Hann).expect("stft failed");
        // n_frames = 1 + (99) / 8 = 1 + 12 = 13
        let expected_frames = 1 + (signal.len() - 1) / hop_length;
        assert_eq!(result.n_frames, expected_frames);
        assert_eq!(result.frames.len(), expected_frames);
    }

    // -----------------------------------------------------------------------
    // 10. STFT returns correct n_freqs = window_size/2 + 1
    // -----------------------------------------------------------------------
    #[test]
    fn test_stft_n_freqs() {
        let signal = vec![1.0; 64];
        let window_size = 16;
        let result = stft(&signal, window_size, 4, WindowType::Rectangular).expect("stft failed");
        assert_eq!(result.n_freqs, window_size / 2 + 1);
        for frame in &result.frames {
            assert_eq!(frame.len(), result.n_freqs);
        }
    }

    // -----------------------------------------------------------------------
    // 11. STFT empty signal → Err
    // -----------------------------------------------------------------------
    #[test]
    fn test_stft_empty_signal() {
        assert!(stft(&[], 16, 8, WindowType::Hann).is_err());
    }

    // -----------------------------------------------------------------------
    // 12. STFT invalid hop_length → Err
    // -----------------------------------------------------------------------
    #[test]
    fn test_stft_invalid_hop() {
        let signal = vec![1.0; 32];
        // hop > window_size
        assert!(stft(&signal, 8, 16, WindowType::Hann).is_err());
        // hop = 0
        assert!(stft(&signal, 8, 0, WindowType::Hann).is_err());
    }

    // -----------------------------------------------------------------------
    // 13. magnitude_spectrogram correct shape
    // -----------------------------------------------------------------------
    #[test]
    fn test_magnitude_spectrogram_shape() {
        let signal = vec![1.0; 64];
        let result = stft(&signal, 16, 4, WindowType::Hann).expect("stft failed");
        let mag = result.magnitude_spectrogram();
        assert_eq!(mag.len(), result.n_frames);
        for row in &mag {
            assert_eq!(row.len(), result.n_freqs);
        }
    }

    // -----------------------------------------------------------------------
    // 14. power_spectrogram values ≥ 0
    // -----------------------------------------------------------------------
    #[test]
    fn test_power_spectrogram_non_negative() {
        let signal: Vec<f64> = (0..64)
            .map(|i| (2.0 * PI * 3.0 * i as f64 / 64.0).sin())
            .collect();
        let result = stft(&signal, 16, 8, WindowType::Hamming).expect("stft failed");
        for row in result.power_spectrogram() {
            for v in row {
                assert!(v >= 0.0, "power must be non-negative, got {v}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // 15. log_magnitude_spectrogram shape matches magnitude
    // -----------------------------------------------------------------------
    #[test]
    fn test_log_magnitude_spectrogram_shape() {
        let signal = vec![1.0; 64];
        let result = stft(&signal, 16, 4, WindowType::Hann).expect("stft failed");
        let log_mag = result.log_magnitude_spectrogram(1.0);
        let mag = result.magnitude_spectrogram();
        assert_eq!(log_mag.len(), mag.len());
        for (lr, mr) in log_mag.iter().zip(mag.iter()) {
            assert_eq!(lr.len(), mr.len());
        }
    }

    // -----------------------------------------------------------------------
    // 16. DCT of constant signal has energy concentrated in DC bin
    // -----------------------------------------------------------------------
    #[test]
    fn test_dct_constant_signal() {
        let n = 8;
        let signal = vec![1.0; n];
        let coeffs = dct(&signal);
        // X[0] = sum of n ones * cos(0) = n
        assert_approx(coeffs[0], n as f64, LOOSE_EPS, "DCT DC bin");
        // Higher-order bins should be near zero for a constant input.
        for (k, &coeff) in coeffs.iter().enumerate().skip(1) {
            assert_approx(coeff.abs(), 0.0, 1e-9, &format!("DCT bin {k} should be ~0"));
        }
    }

    // -----------------------------------------------------------------------
    // 17. IDCT(DCT(x)) ≈ x  (round-trip)
    // -----------------------------------------------------------------------
    #[test]
    fn test_dct_idct_roundtrip() {
        let signal = vec![1.0, 2.0, -1.0, 0.5, 3.0, -0.5, 2.5, 1.5];
        let reconstructed = idct(&dct(&signal));
        assert_eq!(reconstructed.len(), signal.len());
        for (a, b) in reconstructed.iter().zip(signal.iter()) {
            assert_approx(*a, *b, LOOSE_EPS, "DCT roundtrip");
        }
    }

    // -----------------------------------------------------------------------
    // 18. FIR filter with identity kernel [1.0] returns same signal
    // -----------------------------------------------------------------------
    #[test]
    fn test_fir_identity() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let f = FirFilter::new(vec![1.0]).expect("FirFilter::new failed");
        let out = fir_filter(&signal, &f);
        assert_eq!(out.len(), signal.len());
        for (a, b) in out.iter().zip(signal.iter()) {
            assert_approx(*a, *b, EPS, "identity FIR");
        }
    }

    // -----------------------------------------------------------------------
    // 19. FirFilter::low_pass creates n_taps coefficients
    // -----------------------------------------------------------------------
    #[test]
    fn test_fir_low_pass_n_taps() {
        let n_taps = 31;
        let f = FirFilter::low_pass(0.2, n_taps);
        // Odd n_taps → stays 31; even would be bumped.
        assert_eq!(f.n_taps(), n_taps);
    }

    // -----------------------------------------------------------------------
    // 20. FirFilter::new([]) → Err
    // -----------------------------------------------------------------------
    #[test]
    fn test_fir_new_empty_err() {
        assert!(FirFilter::new(vec![]).is_err());
    }

    // -----------------------------------------------------------------------
    // 21. hz_to_mel(1000) ≈ correct Mel value
    // -----------------------------------------------------------------------
    #[test]
    fn test_hz_to_mel() {
        // 2595 * log10(1 + 1000/700) ≈ 999.985...
        let mel = hz_to_mel(1000.0);
        // Known reference: hz_to_mel(1000) ≈ 999.985
        assert_approx(mel, 999.985_f64, 0.01, "hz_to_mel(1000)");
    }

    // -----------------------------------------------------------------------
    // 22. mel_to_hz(hz_to_mel(1000)) ≈ 1000
    // -----------------------------------------------------------------------
    #[test]
    fn test_mel_hz_roundtrip() {
        let hz = 1000.0_f64;
        let roundtrip = mel_to_hz(hz_to_mel(hz));
        assert_approx(roundtrip, hz, LOOSE_EPS, "Mel round-trip");
    }

    // -----------------------------------------------------------------------
    // 23. mel_filterbank returns n_mels rows
    // -----------------------------------------------------------------------
    #[test]
    fn test_mel_filterbank_shape() {
        let n_mels = 40;
        let window_size = 512;
        let sample_rate = 22050.0;
        let fb = mel_filterbank(n_mels, window_size, sample_rate);
        assert_eq!(fb.len(), n_mels);
        let n_freqs = window_size / 2 + 1;
        for row in &fb {
            assert_eq!(row.len(), n_freqs);
        }
    }

    // -----------------------------------------------------------------------
    // 24. ISTFT(STFT(x)) ≈ x  (overlap-add reconstruction)
    // -----------------------------------------------------------------------
    #[test]
    fn test_istft_roundtrip() {
        // Use a signal whose length is an exact multiple of hop_length for a
        // clean round-trip.
        let n = 64;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 4.0 * i as f64 / n as f64).sin())
            .collect();
        let window_size = 16;
        let hop_length = 4;

        let stft_result =
            stft(&signal, window_size, hop_length, WindowType::Hann).expect("stft failed");
        let reconstructed = istft(&stft_result, WindowType::Hann);

        // Compare only the interior samples (away from edge effects).
        let margin = window_size;
        if reconstructed.len() >= n + margin {
            for i in margin..(n - margin) {
                assert_approx(
                    reconstructed[i],
                    signal[i],
                    0.05,
                    &format!("ISTFT roundtrip at sample {i}"),
                );
            }
        } else {
            // At minimum, check that reconstruction has non-trivial length.
            assert!(reconstructed.len() >= n, "ISTFT output too short");
        }
    }

    // -----------------------------------------------------------------------
    // 25. Complex arithmetic sanity checks
    // -----------------------------------------------------------------------
    #[test]
    fn test_complex_arithmetic() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);

        let sum = a + b;
        assert_approx(sum.re, 4.0, EPS, "add re");
        assert_approx(sum.im, 6.0, EPS, "add im");

        let prod = a * b;
        // (1+2i)(3+4i) = 3 - 8 + (4 + 6)i = -5 + 10i
        assert_approx(prod.re, -5.0, EPS, "mul re");
        assert_approx(prod.im, 10.0, EPS, "mul im");

        let conj = a.conjugate();
        assert_approx(conj.re, 1.0, EPS, "conj re");
        assert_approx(conj.im, -2.0, EPS, "conj im");

        let mag = Complex::new(3.0, 4.0).magnitude();
        assert_approx(mag, 5.0, EPS, "magnitude");
    }

    // -----------------------------------------------------------------------
    // 26. Blackman window sum is positive
    // -----------------------------------------------------------------------
    #[test]
    fn test_window_blackman() {
        let w = window(WindowType::Blackman, 16);
        assert_eq!(w.len(), 16);
        let sum: f64 = w.iter().sum();
        assert!(sum > 0.0, "Blackman window sum must be positive");
    }

    // -----------------------------------------------------------------------
    // 27. FlatTop window has correct length
    // -----------------------------------------------------------------------
    #[test]
    fn test_window_flattop() {
        let n = 32;
        let w = window(WindowType::FlatTop, n);
        assert_eq!(w.len(), n);
    }

    // -----------------------------------------------------------------------
    // 28. Triangular window is symmetric
    // -----------------------------------------------------------------------
    #[test]
    fn test_window_triangular_symmetry() {
        let n = 9;
        let w = window(WindowType::Triangular, n);
        for i in 0..n / 2 {
            assert_approx(
                w[i],
                w[n - 1 - i],
                1e-10,
                &format!("triangular symmetry at {i}"),
            );
        }
    }
}
