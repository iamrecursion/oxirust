//! Spectral flux computation.
//!
//! Spectral flux measures the amount of change in the power spectrum
//! between successive frames.  It is widely used as an onset detection
//! function and as a measure of temporal variation.

/// Compute spectral flux between two consecutive magnitude spectra.
///
/// Spectral flux is defined as the sum of the squared differences between
/// the current and previous magnitude spectra:
///
/// ```text
/// flux = Σ_k (|X_n(k)| - |X_{n-1}(k)|)²
/// ```
///
/// # Arguments
/// * `current`  - Magnitude spectrum of the current frame
/// * `previous` - Magnitude spectrum of the previous frame
///
/// # Returns
/// Non-negative flux value.  Returns 0.0 if either slice is empty or
/// they have different lengths.
#[must_use]
pub fn spectral_flux(current: &[f32], previous: &[f32]) -> f32 {
    if current.is_empty() || previous.is_empty() || current.len() != previous.len() {
        return 0.0;
    }

    current
        .iter()
        .zip(previous.iter())
        .map(|(&c, &p)| {
            let diff = c - p;
            diff * diff
        })
        .sum()
}

/// Compute half-wave rectified spectral flux (positive differences only).
///
/// Only penalises spectral *increases*, which makes it more suitable for
/// onset detection (ignores decay).
///
/// ```text
/// flux_hwr = Σ_k max(0, |X_n(k)| - |X_{n-1}(k)|)²
/// ```
#[must_use]
pub fn spectral_flux_hwr(current: &[f32], previous: &[f32]) -> f32 {
    if current.is_empty() || previous.is_empty() || current.len() != previous.len() {
        return 0.0;
    }

    current
        .iter()
        .zip(previous.iter())
        .map(|(&c, &p)| {
            let diff = (c - p).max(0.0);
            diff * diff
        })
        .sum()
}

/// Compute normalised spectral flux divided by the number of bins.
///
/// Dividing by the number of bins makes the value independent of FFT size,
/// enabling comparison across different configurations.
#[must_use]
pub fn spectral_flux_normalised(current: &[f32], previous: &[f32]) -> f32 {
    if current.is_empty() || previous.is_empty() || current.len() != previous.len() {
        return 0.0;
    }

    let raw = spectral_flux(current, previous);
    raw / current.len() as f32
}

/// Compute spectral flux over a spectrogram (frame × bins).
///
/// Returns a vector of length `spectrogram.len() - 1`, where each entry is
/// the flux between consecutive frames.  The first frame has no predecessor
/// and is therefore excluded from the output.
///
/// # Arguments
/// * `spectrogram` - Sequence of magnitude spectrum frames (oldest first)
/// * `normalise`   - If `true`, divide each flux by the number of bins
///
/// # Returns
/// Vector of flux values.  Empty if `spectrogram` has fewer than 2 frames.
#[must_use]
pub fn spectral_flux_track(spectrogram: &[Vec<f32>], normalise: bool) -> Vec<f32> {
    if spectrogram.len() < 2 {
        return Vec::new();
    }

    spectrogram
        .windows(2)
        .map(|w| {
            if normalise {
                spectral_flux_normalised(&w[1], &w[0])
            } else {
                spectral_flux(&w[1], &w[0])
            }
        })
        .collect()
}

/// Compute half-wave-rectified spectral flux over a spectrogram.
#[must_use]
pub fn spectral_flux_hwr_track(spectrogram: &[Vec<f32>]) -> Vec<f32> {
    if spectrogram.len() < 2 {
        return Vec::new();
    }

    spectrogram
        .windows(2)
        .map(|w| spectral_flux_hwr(&w[1], &w[0]))
        .collect()
}

/// Simple peak-picking onset detector based on spectral flux.
///
/// Returns the frame indices where onsets are likely to have occurred,
/// defined as frames where the flux value exceeds `threshold`.
///
/// # Arguments
/// * `flux`      - Spectral flux track from [`spectral_flux_track`]
/// * `threshold` - Minimum flux value to count as an onset
#[must_use]
pub fn detect_onsets_from_flux(flux: &[f32], threshold: f32) -> Vec<usize> {
    flux.iter()
        .enumerate()
        .filter_map(|(i, &f)| if f > threshold { Some(i + 1) } else { None })
        // +1 because flux[0] corresponds to the change between frame 0 and 1
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── spectral_flux ──────────────────────────────────────────────────────────

    #[test]
    fn test_flux_identical_frames() {
        let frame = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(spectral_flux(&frame, &frame), 0.0);
    }

    #[test]
    fn test_flux_empty() {
        let empty: Vec<f32> = vec![];
        assert_eq!(spectral_flux(&empty, &empty), 0.0);
    }

    #[test]
    fn test_flux_different_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        assert_eq!(spectral_flux(&a, &b), 0.0);
    }

    #[test]
    fn test_flux_unit_step() {
        // Single bin changes by 1.0 → flux = 1.0
        let prev = vec![0.0, 0.0, 0.0, 0.0];
        let curr = vec![1.0, 0.0, 0.0, 0.0];
        assert!((spectral_flux(&curr, &prev) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_flux_all_bins_change() {
        // All 4 bins each change by 1.0 → flux = 4.0
        let prev = vec![0.0; 4];
        let curr = vec![1.0; 4];
        assert!((spectral_flux(&curr, &prev) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_flux_symmetry() {
        // flux(a, b) == flux(b, a) — both directions give same squared diff
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![3.0, 1.0, 2.0];
        assert!((spectral_flux(&a, &b) - spectral_flux(&b, &a)).abs() < 1e-6);
    }

    // ── spectral_flux_hwr ─────────────────────────────────────────────────────

    #[test]
    fn test_flux_hwr_ignores_decreases() {
        // Bin decreases: should be ignored by HWR
        let prev = vec![1.0, 0.0, 0.0];
        let curr = vec![0.0, 0.0, 0.0]; // All bins decreased
        assert_eq!(spectral_flux_hwr(&curr, &prev), 0.0);
    }

    #[test]
    fn test_flux_hwr_counts_increases() {
        let prev = vec![0.0, 0.0, 0.0];
        let curr = vec![1.0, 0.0, 2.0];
        // Only positive diffs: 1² + 2² = 5
        assert!((spectral_flux_hwr(&curr, &prev) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_flux_hwr_mixed() {
        let prev = vec![1.0, 0.0, 2.0];
        let curr = vec![3.0, 1.0, 0.0]; // First two up, last one down
                                        // (3-1)² + (1-0)² + max(0,0-2)² = 4 + 1 + 0 = 5
        assert!((spectral_flux_hwr(&curr, &prev) - 5.0).abs() < 1e-6);
    }

    // ── spectral_flux_normalised ──────────────────────────────────────────────

    #[test]
    fn test_flux_normalised_scales_with_bins() {
        let prev4 = vec![0.0; 4];
        let curr4 = vec![1.0; 4];
        let prev8 = vec![0.0; 8];
        let curr8 = vec![1.0; 8];
        // Raw flux: 4 and 8 respectively; normalised: both 1.0
        let n4 = spectral_flux_normalised(&curr4, &prev4);
        let n8 = spectral_flux_normalised(&curr8, &prev8);
        assert!((n4 - 1.0).abs() < 1e-6);
        assert!((n8 - 1.0).abs() < 1e-6);
    }

    // ── spectral_flux_track ───────────────────────────────────────────────────

    #[test]
    fn test_flux_track_length() {
        let spectrogram: Vec<Vec<f32>> = (0..10).map(|_| vec![1.0_f32; 64]).collect();
        let track = spectral_flux_track(&spectrogram, false);
        assert_eq!(track.len(), 9);
    }

    #[test]
    fn test_flux_track_empty_spectrogram() {
        let spectrogram: Vec<Vec<f32>> = vec![];
        let track = spectral_flux_track(&spectrogram, false);
        assert!(track.is_empty());
    }

    #[test]
    fn test_flux_track_single_frame() {
        let spectrogram: Vec<Vec<f32>> = vec![vec![1.0; 64]];
        let track = spectral_flux_track(&spectrogram, false);
        assert!(track.is_empty());
    }

    #[test]
    fn test_flux_track_increasing_energy() {
        // Frames with growing energy should produce non-zero flux.
        let spectrogram: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32 * 0.5; 64]).collect();
        let track = spectral_flux_track(&spectrogram, false);
        assert_eq!(track.len(), 4);
        for &f in &track {
            assert!(f > 0.0, "Expected non-zero flux, got {f}");
        }
    }

    #[test]
    fn test_flux_track_constant_frames() {
        // Constant spectrogram: all fluxes should be zero.
        let spectrogram: Vec<Vec<f32>> = (0..5).map(|_| vec![1.0_f32; 32]).collect();
        let track = spectral_flux_track(&spectrogram, false);
        for &f in &track {
            assert!((f).abs() < 1e-6);
        }
    }

    // ── spectral_flux_hwr_track ───────────────────────────────────────────────

    #[test]
    fn test_flux_hwr_track_length() {
        let spectrogram: Vec<Vec<f32>> = (0..6).map(|_| vec![1.0_f32; 32]).collect();
        let track = spectral_flux_hwr_track(&spectrogram);
        assert_eq!(track.len(), 5);
    }

    // ── detect_onsets_from_flux ───────────────────────────────────────────────

    #[test]
    fn test_detect_onsets_basic() {
        let flux = vec![0.1, 5.0, 0.2, 0.1, 8.0, 0.3];
        let onsets = detect_onsets_from_flux(&flux, 1.0);
        assert_eq!(onsets, vec![2, 5]); // flux[1]=5.0 → frame 2, flux[4]=8.0 → frame 5
    }

    #[test]
    fn test_detect_onsets_no_onsets() {
        let flux = vec![0.1, 0.2, 0.3];
        let onsets = detect_onsets_from_flux(&flux, 1.0);
        assert!(onsets.is_empty());
    }

    #[test]
    fn test_detect_onsets_all_above_threshold() {
        let flux = vec![2.0, 3.0, 4.0];
        let onsets = detect_onsets_from_flux(&flux, 1.0);
        assert_eq!(onsets, vec![1, 2, 3]);
    }
}
