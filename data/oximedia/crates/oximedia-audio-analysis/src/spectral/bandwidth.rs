//! Spectral bandwidth computation.

/// Compute spectral bandwidth.
///
/// Spectral bandwidth is the weighted standard deviation of the frequencies
/// around the spectral centroid. It represents the range of frequencies
/// present in the signal.
///
/// # Arguments
/// * `magnitude` - Magnitude spectrum
/// * `sample_rate` - Sample rate in Hz
/// * `centroid` - Spectral centroid in Hz (pre-computed for efficiency)
///
/// # Returns
/// Spectral bandwidth in Hz
#[must_use]
pub fn spectral_bandwidth(magnitude: &[f32], sample_rate: f32, centroid: f32) -> f32 {
    if magnitude.is_empty() {
        return 0.0;
    }

    let mut weighted_sum = 0.0;
    let mut total_magnitude = 0.0;

    for (i, &mag) in magnitude.iter().enumerate() {
        let freq = i as f32 * sample_rate / (2.0 * (magnitude.len() - 1) as f32);
        let deviation = freq - centroid;
        weighted_sum += deviation * deviation * mag;
        total_magnitude += mag;
    }

    if total_magnitude > 0.0 {
        (weighted_sum / total_magnitude).sqrt()
    } else {
        0.0
    }
}

/// Compute spectral spread (normalized bandwidth).
#[must_use]
pub fn spectral_spread(magnitude: &[f32], sample_rate: f32, centroid: f32) -> f32 {
    let bandwidth = spectral_bandwidth(magnitude, sample_rate, centroid);
    if centroid > 0.0 {
        bandwidth / centroid
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_bandwidth() {
        // Narrow bandwidth for concentrated spectrum
        let mut narrow = vec![0.0; 100];
        narrow[49] = 1.0;
        narrow[50] = 1.0;
        narrow[51] = 1.0;
        let centroid = 50.0 * 1000.0 / (2.0 * 99.0);
        let bw = spectral_bandwidth(&narrow, 1000.0, centroid);
        assert!(bw < 20.0);

        // Wide bandwidth for distributed spectrum
        let wide = vec![1.0; 100];
        let centroid_wide = 50.0 * 1000.0 / (2.0 * 99.0);
        let bw_wide = spectral_bandwidth(&wide, 1000.0, centroid_wide);
        assert!(bw_wide > 100.0);
    }

    #[test]
    fn test_spectral_spread() {
        let magnitude = vec![1.0; 100];
        let centroid = 500.0;
        let spread = spectral_spread(&magnitude, 1000.0, centroid);
        assert!(spread > 0.0);
    }
}
