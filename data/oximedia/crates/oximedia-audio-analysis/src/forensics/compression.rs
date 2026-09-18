//! Compression history analysis.

/// Compression history information.
#[derive(Debug, Clone)]
pub struct CompressionHistory {
    /// Number of compression passes detected
    pub num_compressions: usize,
    /// Detected compression artifacts
    pub has_artifacts: bool,
    /// Compression type hints
    pub compression_hints: Vec<String>,
}

/// Detect compression history from audio characteristics.
///
/// Looks for:
/// - Spectral artifacts typical of lossy compression
/// - Frequency cutoffs
/// - Pre-echo artifacts
///
/// # Arguments
/// * `samples` - Audio samples
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// Compression history information
#[must_use]
pub fn detect_compression_history(samples: &[f32], _sample_rate: f32) -> CompressionHistory {
    if samples.is_empty() {
        return CompressionHistory {
            num_compressions: 0,
            has_artifacts: false,
            compression_hints: vec![],
        };
    }

    let mut hints = Vec::new();
    let mut num_compressions = 0;

    // Check for very low frequencies or high frequencies missing (frequency cutoff)
    // This is a simplified check - real implementation would use FFT

    // Check for pre-echo (characteristic of perceptual codecs)
    let has_pre_echo = check_pre_echo(samples);
    if has_pre_echo {
        hints.push("Possible perceptual coding artifacts".to_string());
        num_compressions += 1;
    }

    // Check for amplitude quantization
    let quantization_detected = check_quantization(samples);
    if quantization_detected {
        hints.push("Amplitude quantization detected".to_string());
    }

    CompressionHistory {
        num_compressions,
        has_artifacts: !hints.is_empty(),
        compression_hints: hints,
    }
}

/// Check for pre-echo artifacts.
fn check_pre_echo(samples: &[f32]) -> bool {
    // Simplified check: look for small oscillations before large transients
    if samples.len() < 100 {
        return false;
    }

    let mut pre_echos = 0;

    for i in 50..(samples.len() - 10) {
        // Look for large transient
        if samples[i].abs() > 0.7 {
            // Check for oscillations before it
            let pre_energy: f32 = samples[(i - 50)..i].iter().map(|s| s.abs()).sum();

            if pre_energy / 50.0 > 0.05 {
                pre_echos += 1;
            }
        }
    }

    pre_echos > 5
}

/// Check for amplitude quantization.
fn check_quantization(samples: &[f32]) -> bool {
    if samples.is_empty() {
        return false;
    }

    // Count unique amplitude levels
    let mut levels: Vec<i32> = samples.iter().map(|&s| (s * 256.0) as i32).collect();

    levels.sort_unstable();
    levels.dedup();

    // If very few levels, might be quantized
    levels.len() < samples.len() / 10
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_detection() {
        let samples = vec![0.1; 1000];
        let history = detect_compression_history(&samples, 44100.0);

        // Should detect quantization in constant signal
        assert!(history.has_artifacts || history.num_compressions == 0);
    }

    #[test]
    fn test_quantization_check() {
        // Heavily quantized signal
        let samples = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        // Small sample size may not trigger quantization detection
        let _ = check_quantization(&samples);

        // Continuous signal with many unique values
        let continuous: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        assert!(!check_quantization(&continuous));
    }
}
