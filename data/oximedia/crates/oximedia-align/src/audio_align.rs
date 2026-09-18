//! Audio-to-video alignment utilities.
//!
//! Provides tools for synchronising audio tracks to video using clap detection,
//! waveform cross-correlation, and drift computation.

use serde::{Deserialize, Serialize};

/// Method used to achieve audio/video synchronisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum SyncMethod {
    /// Clapper-board detected in audio.
    Clap,
    /// Timecode embedded in the stream.
    Timecode,
    /// Waveform cross-correlation.
    Waveform,
    /// Manually specified offset.
    Manual,
}

impl std::fmt::Display for SyncMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Clap => write!(f, "Clap"),
            Self::Timecode => write!(f, "Timecode"),
            Self::Waveform => write!(f, "Waveform"),
            Self::Manual => write!(f, "Manual"),
        }
    }
}

/// The result of an audio/video synchronisation analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioVideoSync {
    /// Milliseconds to add to the video presentation time to align it with the
    /// audio.  Negative values mean the video must be shifted earlier.
    pub video_offset_ms: i64,
    /// Confidence in the sync measurement (0.0 – 1.0).
    pub confidence: f64,
    /// The method used to establish synchronisation.
    pub method: SyncMethod,
}

impl AudioVideoSync {
    /// Create a new sync result.
    #[must_use]
    pub fn new(video_offset_ms: i64, confidence: f64, method: SyncMethod) -> Self {
        Self {
            video_offset_ms,
            confidence,
            method,
        }
    }

    /// Returns `true` when the sync confidence exceeds the given threshold.
    #[must_use]
    pub fn is_reliable(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }
}

/// Summary report describing the sync state between audio and video tracks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncReport {
    /// Duration of the video track in milliseconds.
    pub video_duration_ms: u64,
    /// Duration of the audio track in milliseconds.
    pub audio_duration_ms: u64,
    /// Sync offset at the beginning of the clip (milliseconds).
    pub sync_offset_ms: i64,
    /// Drift in parts-per-million between audio and video clocks.
    pub drift_ppm: f64,
}

impl SyncReport {
    /// Create a new sync report.
    #[must_use]
    pub fn new(
        video_duration_ms: u64,
        audio_duration_ms: u64,
        sync_offset_ms: i64,
        drift_ppm: f64,
    ) -> Self {
        Self {
            video_duration_ms,
            audio_duration_ms,
            sync_offset_ms,
            drift_ppm,
        }
    }

    /// `true` when the drift magnitude is small enough to be negligible
    /// (less than 1 ppm absolute).
    #[must_use]
    pub fn is_in_sync(&self) -> bool {
        self.drift_ppm.abs() < 1.0
    }

    /// Difference in duration (audio minus video) in milliseconds.
    #[must_use]
    pub fn duration_delta_ms(&self) -> i64 {
        self.audio_duration_ms as i64 - self.video_duration_ms as i64
    }
}

// ── Clap detection ────────────────────────────────────────────────────────────

/// Detect a sharp transient (clap) in a mono audio stream.
///
/// # Arguments
///
/// * `samples` – Normalised f64 samples in [-1.0, 1.0].
/// * `sample_rate` – Samples per second.
///
/// # Returns
///
/// The timestamp (in milliseconds from the start) of the loudest detected
/// transient, or `None` if the signal is empty or featureless.
#[must_use]
pub fn detect_clap(samples: &[f64], sample_rate: u32) -> Option<u64> {
    if samples.is_empty() || sample_rate == 0 {
        return None;
    }

    let sr = sample_rate as usize;

    // Compute a simple onset strength as the rectified first-order difference
    // between sample absolute values (so-called "spectral flux" on raw samples).
    let window = (sr / 100).max(1); // ~10 ms smoothing window

    // Smooth the absolute signal
    let abs_samples: Vec<f64> = samples.iter().map(|&s| s.abs()).collect();
    let smoothed: Vec<f64> = abs_samples
        .windows(window)
        .map(|w| w.iter().sum::<f64>() / w.len() as f64)
        .collect();

    // Compute first-order difference (onset strength)
    let onset: Vec<f64> = smoothed
        .windows(2)
        .map(|w| (w[1] - w[0]).max(0.0))
        .collect();

    // Find global maximum
    let (best_idx, best_val) = onset
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;

    // Require the transient to be significant relative to the smoothing window.
    // A full-scale step across the window produces onset ≈ 1/window, so set the
    // threshold at ~5% of that.
    let min_onset = 0.05 / window as f64;
    if *best_val < min_onset {
        return None;
    }

    // Convert from smoothed-difference index to original sample index
    let sample_idx = best_idx + window; // approximate
    let ms = (sample_idx as u64 * 1000) / u64::from(sample_rate);
    Some(ms)
}

// ── Cross-correlation ─────────────────────────────────────────────────────────

/// Compute the full (linear) cross-correlation of two f32 arrays.
///
/// The output has length `a.len() + b.len() - 1`.
/// The centre element (index `b.len() - 1`) corresponds to zero lag.
#[must_use]
pub fn cross_correlate_waveforms(a: &[f32], b: &[f32]) -> Vec<f32> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }

    let len = a.len() + b.len() - 1;
    let mut result = vec![0.0_f32; len];

    // Cross-correlation: corr[lag] = sum_n a[n] * b[n - lag + (b.len()-1)]
    // lag index in [0, len-1], where index b.len()-1 is zero-lag
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            // lag_index = j - i + (b.len() - 1)  [corr[lag] = sum_n a[n]*b[n+lag]]
            let lag_index = j as isize - i as isize + (b.len() as isize - 1);
            if lag_index >= 0 && (lag_index as usize) < len {
                result[lag_index as usize] += ai * bj;
            }
        }
    }

    result
}

/// Find the lag (in samples) at which two waveforms are best aligned.
///
/// Returns the lag `d` such that shifting `b` by `d` samples aligns it with
/// `a`.  Positive `d` means `b` starts later than `a`.
///
/// Returns 0 when either slice is empty.
#[must_use]
pub fn find_max_correlation_offset(a: &[f32], b: &[f32]) -> i32 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }

    let corr = cross_correlate_waveforms(a, b);

    // Peak index in the correlation vector
    let peak_idx = corr
        .iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i);

    // The zero-lag index in the full cross-correlation is `b.len() - 1`
    let zero_lag = (b.len() as i32) - 1;
    peak_idx as i32 - zero_lag
}

// ── Drift computation ─────────────────────────────────────────────────────────

/// Compute the clock drift between audio and video in parts-per-million.
///
/// # Arguments
///
/// * `start_offset_ms` – Sync offset measured at the beginning of the clip.
/// * `end_offset_ms` – Sync offset measured at the end of the clip.
/// * `duration_ms` – Duration of the clip in milliseconds.
///
/// # Returns
///
/// Drift in ppm.  A positive value means the audio clock runs faster than the
/// video clock.  Returns 0.0 when `duration_ms` is zero.
#[must_use]
pub fn compute_drift(start_offset_ms: i64, end_offset_ms: i64, duration_ms: u64) -> f64 {
    if duration_ms == 0 {
        return 0.0;
    }

    let delta_ms = (end_offset_ms - start_offset_ms) as f64;
    (delta_ms / duration_ms as f64) * 1_000_000.0
}

// ── Spectral (phase-correlation) audio alignment ──────────────────────────────

/// Configuration for spectral audio alignment.
#[derive(Debug, Clone)]
pub struct SpectralAlignConfig {
    /// FFT size (should be a power of two for efficiency). The input signals
    /// will be zero-padded to at least this length.
    pub fft_size: usize,
    /// Maximum lag to search (in samples). If `None`, the full FFT range is
    /// searched.
    pub max_lag: Option<usize>,
}

impl Default for SpectralAlignConfig {
    fn default() -> Self {
        Self {
            fft_size: 8192,
            max_lag: None,
        }
    }
}

/// Result of spectral alignment.
#[derive(Debug, Clone)]
pub struct SpectralAlignResult {
    /// The detected offset in samples.  Positive means `b` should be shifted
    /// *later* (it starts before `a`).
    pub offset_samples: i32,
    /// Peak normalised cross-power spectrum value (higher = more confident).
    pub peak_value: f64,
    /// Confidence score in [0, 1].
    pub confidence: f64,
}

/// Find the alignment offset between two audio signals using phase correlation
/// in the frequency domain.
///
/// This is the spectral equivalent of time-domain cross-correlation: we compute
/// the normalised cross-power spectrum and inverse-FFT it to get the
/// generalised cross-correlation (GCC-PHAT), then pick the lag with the
/// largest peak.
///
/// Phase correlation is more robust than plain cross-correlation for signals
/// with different amplitude envelopes (e.g. different microphone gains)
/// because it whitens the magnitude spectrum.
///
/// # Arguments
///
/// * `a` -- First audio signal (normalised f32 samples).
/// * `b` -- Second audio signal (normalised f32 samples).
/// * `config` -- Spectral alignment configuration.
///
/// # Returns
///
/// A [`SpectralAlignResult`] with the detected offset.  Returns offset 0 with
/// confidence 0 if either signal is empty.
#[must_use]
pub fn spectral_align(a: &[f32], b: &[f32], config: &SpectralAlignConfig) -> SpectralAlignResult {
    if a.is_empty() || b.is_empty() {
        return SpectralAlignResult {
            offset_samples: 0,
            peak_value: 0.0,
            confidence: 0.0,
        };
    }

    // Determine FFT size: next power-of-two >= max(len_a, len_b, config.fft_size)
    let min_len = a.len().max(b.len()).max(config.fft_size);
    let n = min_len.next_power_of_two();

    // Zero-pad both signals to length n
    let mut ra = vec![0.0_f64; n];
    let mut ia = vec![0.0_f64; n];
    for (i, &v) in a.iter().enumerate() {
        ra[i] = f64::from(v);
    }

    let mut rb = vec![0.0_f64; n];
    let mut ib = vec![0.0_f64; n];
    for (i, &v) in b.iter().enumerate() {
        rb[i] = f64::from(v);
    }

    // Forward FFT of both signals
    fft_in_place(&mut ra, &mut ia, false);
    fft_in_place(&mut rb, &mut ib, false);

    // Compute cross-power spectrum with smoothed phase normalisation.
    // We use a regularised version: R(k) = A(k) * conj(B(k)) / (|A(k)*conj(B(k))| + eps)
    // where eps prevents division by zero for zero-padded regions.
    // This is a mild form of GCC-PHAT that retains some magnitude weighting
    // for better performance with zero-padded signals.
    let mut cr = vec![0.0_f64; n];
    let mut ci = vec![0.0_f64; n];

    // Compute a regularisation threshold based on average magnitude
    let mut sum_mag = 0.0_f64;
    for k in 0..n {
        let xr = ra[k] * rb[k] + ia[k] * ib[k];
        let xi = ia[k] * rb[k] - ra[k] * ib[k];
        sum_mag += (xr * xr + xi * xi).sqrt();
    }
    let eps = (sum_mag / n as f64) * 0.01 + 1e-15;

    for k in 0..n {
        // A * conj(B) = (ra+j*ia)*(rb-j*ib) = (ra*rb+ia*ib) + j*(ia*rb-ra*ib)
        let xr = ra[k] * rb[k] + ia[k] * ib[k];
        let xi = ia[k] * rb[k] - ra[k] * ib[k];
        let mag = (xr * xr + xi * xi).sqrt();
        let denom = mag + eps;
        cr[k] = xr / denom;
        ci[k] = xi / denom;
    }

    // Inverse FFT to get generalised cross-correlation
    fft_in_place(&mut cr, &mut ci, true);

    // Search for peak within the allowed lag range
    let max_lag = config.max_lag.unwrap_or(n / 2);
    let max_lag = max_lag.min(n / 2);

    let mut best_idx = 0usize;
    let mut best_val = f64::NEG_INFINITY;

    // Positive lags: indices 0..max_lag
    for i in 0..max_lag.min(n) {
        if cr[i] > best_val {
            best_val = cr[i];
            best_idx = i;
        }
    }
    // Negative lags: indices n-max_lag..n
    let start = if max_lag < n { n - max_lag } else { 0 };
    for i in start..n {
        if cr[i] > best_val {
            best_val = cr[i];
            best_idx = i;
        }
    }

    // Convert index to signed lag
    let offset = if best_idx <= n / 2 {
        best_idx as i32
    } else {
        best_idx as i32 - n as i32
    };

    // Compute confidence as the peak value relative to the RMS of the
    // correlation (a sharp peak means high confidence).
    let rms = (cr.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
    let confidence = if rms > 1e-15 {
        (best_val / (rms * (n as f64).sqrt())).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Negate so that a positive result means "b is delayed (should be shifted
    // later)" which matches the documented convention.
    SpectralAlignResult {
        offset_samples: -offset,
        peak_value: best_val,
        confidence,
    }
}

// ── Radix-2 Cooley-Tukey FFT ─────────────────────────────────────────────────

/// In-place radix-2 Cooley-Tukey FFT (or inverse FFT when `inverse` is true).
///
/// `re` and `im` must have the same length, which must be a power of two.
fn fft_in_place(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    debug_assert_eq!(n, im.len());
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two());

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 0..n {
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
        let mut m = n >> 1;
        while m >= 1 && j >= m {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // Butterfly stages
    let sign: f64 = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = sign * std::f64::consts::PI * 2.0 / len as f64;
        let wn_r = angle.cos();
        let wn_i = angle.sin();

        let mut start = 0;
        while start < n {
            let mut wr = 1.0_f64;
            let mut wi = 0.0_f64;
            for k in 0..half {
                let even = start + k;
                let odd = start + k + half;
                let tr = wr * re[odd] - wi * im[odd];
                let ti = wr * im[odd] + wi * re[odd];
                re[odd] = re[even] - tr;
                im[odd] = im[even] - ti;
                re[even] += tr;
                im[even] += ti;
                let new_wr = wr * wn_r - wi * wn_i;
                wi = wr * wn_i + wi * wn_r;
                wr = new_wr;
            }
            start += len;
        }
        len <<= 1;
    }

    // For inverse FFT, divide by n
    if inverse {
        let inv_n = 1.0 / n as f64;
        for v in re.iter_mut() {
            *v *= inv_n;
        }
        for v in im.iter_mut() {
            *v *= inv_n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── SyncMethod ────────────────────────────────────────────────────────────

    #[test]
    fn test_sync_method_display() {
        assert_eq!(SyncMethod::Clap.to_string(), "Clap");
        assert_eq!(SyncMethod::Timecode.to_string(), "Timecode");
        assert_eq!(SyncMethod::Waveform.to_string(), "Waveform");
        assert_eq!(SyncMethod::Manual.to_string(), "Manual");
    }

    // ── AudioVideoSync ────────────────────────────────────────────────────────

    #[test]
    fn test_audio_video_sync_is_reliable_pass() {
        let sync = AudioVideoSync::new(100, 0.9, SyncMethod::Clap);
        assert!(sync.is_reliable(0.8));
    }

    #[test]
    fn test_audio_video_sync_is_reliable_fail() {
        let sync = AudioVideoSync::new(100, 0.5, SyncMethod::Waveform);
        assert!(!sync.is_reliable(0.8));
    }

    #[test]
    fn test_audio_video_sync_fields() {
        let sync = AudioVideoSync::new(-250, 0.75, SyncMethod::Timecode);
        assert_eq!(sync.video_offset_ms, -250);
        assert_eq!(sync.method, SyncMethod::Timecode);
    }

    // ── SyncReport ────────────────────────────────────────────────────────────

    #[test]
    fn test_sync_report_duration_delta() {
        let r = SyncReport::new(60_000, 60_033, 0, 0.55);
        assert_eq!(r.duration_delta_ms(), 33);
    }

    #[test]
    fn test_sync_report_is_in_sync_true() {
        let r = SyncReport::new(60_000, 60_000, 0, 0.1);
        assert!(r.is_in_sync());
    }

    #[test]
    fn test_sync_report_is_in_sync_false() {
        let r = SyncReport::new(60_000, 60_000, 0, 5.0);
        assert!(!r.is_in_sync());
    }

    // ── detect_clap ───────────────────────────────────────────────────────────

    #[test]
    fn test_detect_clap_empty() {
        assert!(detect_clap(&[], 48000).is_none());
    }

    #[test]
    fn test_detect_clap_zero_sample_rate() {
        let samples = vec![0.0_f64; 100];
        assert!(detect_clap(&samples, 0).is_none());
    }

    #[test]
    fn test_detect_clap_silent_signal() {
        let samples = vec![0.0_f64; 48000];
        // Silent signal – no significant transient
        assert!(detect_clap(&samples, 48000).is_none());
    }

    #[test]
    fn test_detect_clap_finds_transient() {
        // Place a wide transient spike at 1 second (~500 samples = ~10 ms)
        let mut samples = vec![0.01_f64; 48000 * 2];
        for i in 0..500 {
            samples[48000 + i] = 1.0;
        }
        let ts = detect_clap(&samples, 48000);
        assert!(ts.is_some());
        let ms = ts.expect("ms should be valid");
        // Expect roughly around 1000 ms (within ±200 ms to account for smoothing)
        assert!(ms > 800 && ms < 1300, "timestamp={ms}");
    }

    // ── cross_correlate_waveforms ─────────────────────────────────────────────

    #[test]
    fn test_cross_correlate_empty() {
        assert!(cross_correlate_waveforms(&[], &[1.0]).is_empty());
    }

    #[test]
    fn test_cross_correlate_output_length() {
        let a = vec![1.0_f32; 5];
        let b = vec![1.0_f32; 3];
        let corr = cross_correlate_waveforms(&a, &b);
        assert_eq!(corr.len(), 7); // 5 + 3 - 1
    }

    #[test]
    fn test_cross_correlate_identical_unit_impulse() {
        let a = vec![0.0_f32, 1.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        let corr = cross_correlate_waveforms(&a, &b);
        // Peak should be at index b.len()-1 = 2 (zero-lag)
        let peak = corr
            .iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).expect("partial_cmp should succeed"))
            .expect("test expectation failed");
        assert_eq!(peak.0, 2);
    }

    // ── find_max_correlation_offset ───────────────────────────────────────────

    #[test]
    fn test_find_max_correlation_offset_zero_lag() {
        let a = vec![0.0_f32, 0.0, 1.0, 0.0, 0.0];
        let b = vec![0.0_f32, 0.0, 1.0, 0.0, 0.0];
        let lag = find_max_correlation_offset(&a, &b);
        assert_eq!(lag, 0);
    }

    #[test]
    fn test_find_max_correlation_offset_shifted() {
        // b = a shifted right by 2 samples
        let a = vec![0.0_f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let lag = find_max_correlation_offset(&a, &b);
        // b is 2 samples ahead of where we need it → lag should be +2
        assert_eq!(lag, 2);
    }

    #[test]
    fn test_find_max_correlation_offset_empty() {
        assert_eq!(find_max_correlation_offset(&[], &[]), 0);
    }

    // ── compute_drift ─────────────────────────────────────────────────────────

    #[test]
    fn test_compute_drift_zero_duration() {
        assert_eq!(compute_drift(0, 100, 0), 0.0);
    }

    #[test]
    fn test_compute_drift_no_drift() {
        assert_eq!(compute_drift(50, 50, 60_000), 0.0);
    }

    #[test]
    fn test_compute_drift_known_value() {
        // 100 ms drift over 100_000 ms = 1000 ppm
        let ppm = compute_drift(0, 100, 100_000);
        assert!((ppm - 1000.0).abs() < 1e-6, "ppm={ppm}");
    }

    #[test]
    fn test_compute_drift_negative() {
        let ppm = compute_drift(100, 0, 100_000);
        assert!((ppm + 1000.0).abs() < 1e-6, "ppm={ppm}");
    }

    // ── FFT ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_fft_roundtrip() {
        let n = 16;
        let mut re: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let mut im = vec![0.0_f64; n];
        let original = re.clone();

        fft_in_place(&mut re, &mut im, false);
        fft_in_place(&mut re, &mut im, true);

        for (i, (&orig, &recovered)) in original.iter().zip(re.iter()).enumerate() {
            assert!(
                (orig - recovered).abs() < 1e-10,
                "FFT roundtrip mismatch at {i}: {orig} vs {recovered}"
            );
        }
    }

    #[test]
    fn test_fft_dc_component() {
        let n = 8;
        let mut re = vec![1.0_f64; n];
        let mut im = vec![0.0_f64; n];

        fft_in_place(&mut re, &mut im, false);

        // DC component should be n, all others zero
        assert!((re[0] - n as f64).abs() < 1e-10);
        for i in 1..n {
            assert!(re[i].abs() < 1e-10, "bin {i} should be zero: {}", re[i]);
        }
    }

    // ── Spectral alignment ──────────────────────────────────────────────────

    #[test]
    fn test_spectral_align_empty() {
        let config = SpectralAlignConfig::default();
        let result = spectral_align(&[], &[1.0], &config);
        assert_eq!(result.offset_samples, 0);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_spectral_align_identical_signals() {
        let n = 256;
        let signal: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();

        let config = SpectralAlignConfig {
            fft_size: 512,
            max_lag: Some(64),
        };
        let result = spectral_align(&signal, &signal, &config);
        assert_eq!(
            result.offset_samples, 0,
            "identical signals should have zero offset"
        );
        assert!(result.peak_value > 0.0, "peak should be positive");
    }

    #[test]
    fn test_spectral_align_known_shift() {
        let n = 1024;
        let shift = 10;
        // Generate a rich signal with many frequency components
        let signal: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32;
                (t * 0.05).sin()
                    + 0.5 * (t * 0.13).sin()
                    + 0.3 * (t * 0.21).cos()
                    + 0.2 * (t * 0.37).sin()
            })
            .collect();

        // Create b as a delayed copy of signal (a leads, b is delayed)
        let mut a_sig = vec![0.0_f32; n];
        let mut b_sig = vec![0.0_f32; n];
        for i in 0..n {
            a_sig[i] = signal[i];
        }
        for i in shift..n {
            b_sig[i] = signal[i - shift];
        }

        let config = SpectralAlignConfig {
            fft_size: 2048,
            max_lag: Some(64),
        };
        let result = spectral_align(&a_sig, &b_sig, &config);
        // b is delayed by `shift` relative to a, so offset should be positive
        assert!(
            (result.offset_samples - shift as i32).abs() <= 2,
            "expected offset ~{shift}, got {}",
            result.offset_samples
        );
    }

    #[test]
    fn test_spectral_align_negative_shift() {
        let n = 2048;
        let shift = 8;
        let signal: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32;
                (t * 0.07).sin()
                    + 0.5 * (t * 0.19).cos()
                    + 0.3 * (t * 0.31).sin()
                    + 0.2 * (t * 0.47).cos()
            })
            .collect();

        // Construct both signals with shared interior; avoid edge artefacts
        // by copying the full overlap region into both a and b.
        let mut a_sig = vec![0.0_f32; n];
        let mut b_sig = vec![0.0_f32; n];
        for i in 0..n {
            b_sig[i] = signal[i];
        }
        for i in shift..n {
            a_sig[i] = signal[i - shift];
        }

        let config = SpectralAlignConfig {
            fft_size: 4096,
            max_lag: Some(64),
        };
        let result = spectral_align(&a_sig, &b_sig, &config);
        // a is delayed by shift => offset should be negative
        assert!(
            (result.offset_samples + shift as i32).abs() <= 2,
            "expected offset ~-{shift}, got {}",
            result.offset_samples
        );
    }

    #[test]
    fn test_spectral_align_config_default() {
        let config = SpectralAlignConfig::default();
        assert_eq!(config.fft_size, 8192);
        assert!(config.max_lag.is_none());
    }
}
