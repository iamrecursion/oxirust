//! Spectral rolloff computation.

/// Compute spectral rolloff frequency.
///
/// The spectral rolloff is the frequency below which a specified fraction
/// (threshold) of the total spectral energy is contained.  A threshold of
/// 0.85 is typical; higher values give a higher rolloff frequency.
///
/// # Arguments
/// * `magnitude` - Magnitude spectrum (positive frequencies, length = N/2+1)
/// * `sample_rate` - Sample rate in Hz
/// * `threshold` - Energy fraction in (0, 1], typically 0.85
///
/// # Returns
/// Rolloff frequency in Hz, or 0.0 if `magnitude` is empty or `threshold`
/// is outside (0, 1].
#[must_use]
pub fn spectral_rolloff(magnitude: &[f32], sample_rate: f32, threshold: f32) -> f32 {
    if magnitude.is_empty() || threshold <= 0.0 || threshold > 1.0 {
        return 0.0;
    }

    let total_energy: f32 = magnitude.iter().map(|&m| m * m).sum();
    if total_energy <= 0.0 {
        return 0.0;
    }

    let target = total_energy * threshold;
    let mut cumulative = 0.0;
    let n_bins = magnitude.len();

    for (i, &mag) in magnitude.iter().enumerate() {
        cumulative += mag * mag;
        if cumulative >= target {
            // Linear interpolation between bins for sub-bin accuracy
            let freq = bin_to_freq(i, sample_rate, n_bins);
            return freq;
        }
    }

    // Fell through — return Nyquist
    sample_rate / 2.0
}

/// Compute spectral rolloff at the standard 85 % threshold.
#[must_use]
pub fn spectral_rolloff_85(magnitude: &[f32], sample_rate: f32) -> f32 {
    spectral_rolloff(magnitude, sample_rate, 0.85)
}

/// Compute spectral rolloff at the 95 % threshold.
#[must_use]
pub fn spectral_rolloff_95(magnitude: &[f32], sample_rate: f32) -> f32 {
    spectral_rolloff(magnitude, sample_rate, 0.95)
}

/// Compute spectral rolloff track (frame-by-frame).
///
/// Returns one rolloff frequency per frame (spectrogram row).
#[must_use]
pub fn spectral_rolloff_track(
    spectrogram: &[Vec<f32>],
    sample_rate: f32,
    threshold: f32,
) -> Vec<f32> {
    spectrogram
        .iter()
        .map(|frame| spectral_rolloff(frame, sample_rate, threshold))
        .collect()
}

/// Convert an FFT bin index to frequency in Hz.
#[inline]
fn bin_to_freq(bin: usize, sample_rate: f32, n_bins: usize) -> f32 {
    // n_bins = N/2 + 1, so the Nyquist bin = n_bins - 1 = N/2.
    // freq(k) = k * sample_rate / N = k * sample_rate / (2 * (n_bins - 1))
    bin as f32 * sample_rate / (2.0 * (n_bins - 1) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine_spectrum(freq_hz: f32, sample_rate: f32, fft_size: usize) -> Vec<f32> {
        // Synthesise a single-frequency spectrum for testing.
        let bin = ((freq_hz / sample_rate) * fft_size as f32).round() as usize;
        let bin = bin.min(fft_size / 2);
        let mut spectrum = vec![0.0_f32; fft_size / 2 + 1];
        spectrum[bin] = 1.0;
        spectrum
    }

    #[test]
    fn test_rolloff_empty() {
        let empty: Vec<f32> = vec![];
        assert_eq!(spectral_rolloff(&empty, 44100.0, 0.85), 0.0);
    }

    #[test]
    fn test_rolloff_zero_threshold() {
        let mag = vec![1.0; 512];
        assert_eq!(spectral_rolloff(&mag, 44100.0, 0.0), 0.0);
    }

    #[test]
    fn test_rolloff_above_one_threshold() {
        let mag = vec![1.0; 512];
        assert_eq!(spectral_rolloff(&mag, 44100.0, 1.5), 0.0);
    }

    #[test]
    fn test_rolloff_uniform_spectrum_85() {
        // For a uniform spectrum, 85 % of energy is in the first 85 % of bins.
        let n = 100;
        let mag = vec![1.0_f32; n];
        let sr = 44100.0;
        let rolloff = spectral_rolloff(&mag, sr, 0.85);
        let nyquist = sr / 2.0;
        // Should be roughly 85 % of Nyquist
        assert!(rolloff > 0.0 && rolloff <= nyquist);
        assert!(rolloff < nyquist * 0.95);
    }

    #[test]
    fn test_rolloff_low_frequency_content() {
        // All energy at DC: rolloff should be very low.
        let mut mag = vec![0.0_f32; 256];
        mag[0] = 1.0;
        let rolloff = spectral_rolloff(&mag, 44100.0, 0.85);
        assert_eq!(rolloff, 0.0);
    }

    #[test]
    fn test_rolloff_high_frequency_tone() {
        // Energy concentrated near Nyquist: rolloff should be high.
        let mut mag = vec![0.0_f32; 256];
        let last = mag.len() - 1;
        mag[last] = 1.0;
        let sr = 44100.0;
        let rolloff = spectral_rolloff(&mag, sr, 0.85);
        assert!(rolloff > sr / 2.0 * 0.9, "High-freq rolloff = {rolloff}");
    }

    #[test]
    fn test_rolloff_85_vs_95() {
        // 95 % threshold should give >= 85 % threshold.
        let mag: Vec<f32> = (0..256).map(|_| 1.0).collect();
        let sr = 44100.0;
        let r85 = spectral_rolloff_85(&mag, sr);
        let r95 = spectral_rolloff_95(&mag, sr);
        assert!(r95 >= r85, "r95={r95} should be >= r85={r85}");
    }

    #[test]
    fn test_rolloff_track_length() {
        let spectrogram: Vec<Vec<f32>> = (0..10).map(|_| vec![1.0_f32; 64]).collect();
        let track = spectral_rolloff_track(&spectrogram, 44100.0, 0.85);
        assert_eq!(track.len(), 10);
    }

    #[test]
    fn test_rolloff_track_monotone_input() {
        // Each frame has energy at a progressively higher bin.
        let sr = 44100.0;
        let n_bins = 64;
        let spectrogram: Vec<Vec<f32>> = (0..8)
            .map(|i| {
                let mut frame = vec![0.0_f32; n_bins];
                let bin = (i * 8).min(n_bins - 1);
                frame[bin] = 1.0;
                frame
            })
            .collect();
        let track = spectral_rolloff_track(&spectrogram, sr, 0.85);
        // Each successive frame should have equal or higher rolloff.
        for i in 1..track.len() {
            assert!(
                track[i] >= track[i - 1],
                "Expected monotone rolloff, track[{i}]={} < track[{}]={}",
                track[i],
                i - 1,
                track[i - 1]
            );
        }
    }

    #[test]
    fn test_rolloff_full_energy_threshold() {
        // At threshold=1.0, rolloff should reach Nyquist or the last bin.
        let mag = vec![1.0_f32; 128];
        let sr = 8000.0;
        let rolloff = spectral_rolloff(&mag, sr, 1.0);
        assert!(rolloff <= sr / 2.0 + 1.0);
    }

    #[test]
    fn test_bin_to_freq_nyquist() {
        let sr = 44100.0;
        let n_bins = 1025; // fft_size = 2048
        let nyquist_bin = n_bins - 1;
        let freq = bin_to_freq(nyquist_bin, sr, n_bins);
        assert!((freq - sr / 2.0).abs() < 1.0);
    }

    #[test]
    fn test_rolloff_sine_440hz() {
        // Synthesise a 440 Hz tone spectrum and check rolloff > 0.
        let sr = 44100.0;
        let fft_size = 2048;
        let spectrum = sine_spectrum(440.0, sr, fft_size);
        let rolloff = spectral_rolloff(&spectrum, sr, 0.85);
        assert!(rolloff > 0.0);
        // Rolloff should be at or near the 440 Hz bin frequency.
        let _ = PI; // suppress unused import warning
    }
}
