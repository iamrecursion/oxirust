//! Spectral centroid computation.

/// Compute spectral centroid (center of mass of the spectrum).
///
/// The spectral centroid represents the "brightness" of a sound and is
/// calculated as the weighted mean of the frequencies present in the signal.
///
/// # Arguments
/// * `magnitude` - Magnitude spectrum
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// Spectral centroid in Hz
#[must_use]
pub fn spectral_centroid(magnitude: &[f32], sample_rate: f32) -> f32 {
    if magnitude.is_empty() {
        return 0.0;
    }

    let mut weighted_sum = 0.0;
    let mut total_magnitude = 0.0;

    for (i, &mag) in magnitude.iter().enumerate() {
        let freq = i as f32 * sample_rate / (2.0 * (magnitude.len() - 1) as f32);
        weighted_sum += freq * mag;
        total_magnitude += mag;
    }

    if total_magnitude > 0.0 {
        weighted_sum / total_magnitude
    } else {
        0.0
    }
}

/// Compute spectral centroid over time (frame-based).
#[must_use]
pub fn spectral_centroid_track(spectrogram: &[Vec<f32>], sample_rate: f32) -> Vec<f32> {
    spectrogram
        .iter()
        .map(|frame| spectral_centroid(frame, sample_rate))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_centroid() {
        // Test with a simple spectrum
        let magnitude = vec![0.0, 1.0, 0.0, 0.0];
        let sample_rate = 1000.0;
        let centroid = spectral_centroid(&magnitude, sample_rate);

        // Centroid should be at the peak (bin 1 out of 3 bins = 166.67 Hz)
        assert!((centroid - 166.67).abs() < 10.0);
    }

    #[test]
    fn test_centroid_empty() {
        let empty: Vec<f32> = vec![];
        assert_eq!(spectral_centroid(&empty, 44100.0), 0.0);
    }

    #[test]
    fn test_centroid_track() {
        let spectrogram = vec![vec![0.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 0.0]];
        let track = spectral_centroid_track(&spectrogram, 1000.0);
        assert_eq!(track.len(), 2);
        assert!(track[0] < track[1]); // Second frame has higher centroid
    }
}
