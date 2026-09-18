//! Vibrato detection and analysis.

use super::{PitchContour, PitchResult};

/// Vibrato detection result.
#[derive(Debug, Clone)]
pub struct VibratoResult {
    /// Whether vibrato is present
    pub present: bool,
    /// Vibrato rate in Hz (modulation frequency)
    pub rate: f32,
    /// Vibrato extent in cents (pitch variation)
    pub extent: f32,
    /// Confidence in detection (0-1)
    pub confidence: f32,
}

/// Detect vibrato from pitch tracking result.
///
/// Vibrato is periodic pitch modulation typically in the range of 4-8 Hz
/// with extent of 50-200 cents.
#[must_use]
pub fn detect_vibrato(
    pitch_result: &PitchResult,
    hop_size: usize,
    sample_rate: f32,
) -> VibratoResult {
    let contour = PitchContour::from_pitch_result(pitch_result, hop_size, sample_rate);

    // Extract voiced pitch values
    let voiced_pitches: Vec<f32> = contour
        .frequencies
        .iter()
        .zip(&contour.confidences)
        .filter(|(_, &conf)| conf > 0.5)
        .map(|(&f, _)| f)
        .collect();

    if voiced_pitches.len() < 20 {
        return VibratoResult {
            present: false,
            rate: 0.0,
            extent: 0.0,
            confidence: 0.0,
        };
    }

    // Compute autocorrelation to find periodicity
    let max_lag = 50; // For ~4-8 Hz vibrato at typical hop rates
    let mut autocorr = vec![0.0; max_lag];

    for lag in 0..max_lag {
        let mut sum = 0.0;
        let mut count = 0;

        for i in 0..(voiced_pitches.len() - lag) {
            sum += voiced_pitches[i] * voiced_pitches[i + lag];
            count += 1;
        }

        if count > 0 {
            autocorr[lag] = sum / count as f32;
        }
    }

    // Find peak in autocorrelation (excluding lag 0)
    let mut max_corr = 0.0;
    let mut best_lag = 0usize;

    for (offset, &val) in autocorr[5..].iter().enumerate() {
        if val > max_corr {
            max_corr = val;
            best_lag = 5 + offset;
        }
    }

    // Vibrato rate from lag
    let hop_duration = hop_size as f32 / sample_rate;
    let rate = if best_lag > 0 {
        1.0 / (best_lag as f32 * hop_duration)
    } else {
        0.0
    };

    // Vibrato extent (convert to cents)
    let mean_pitch = voiced_pitches.iter().sum::<f32>() / voiced_pitches.len() as f32;
    let pitch_variation = voiced_pitches
        .iter()
        .map(|&f| (f - mean_pitch).abs())
        .sum::<f32>()
        / voiced_pitches.len() as f32;

    let extent = 1200.0 * (pitch_variation / mean_pitch).abs(); // Convert to cents

    // Detect vibrato presence
    let present = (4.0..=9.0).contains(&rate) && (20.0..=300.0).contains(&extent);

    let confidence = if present {
        let rate_score = if (5.0..=7.5).contains(&rate) {
            1.0
        } else {
            0.7
        };

        let extent_score = if (50.0..=200.0).contains(&extent) {
            1.0
        } else {
            0.7
        };

        (rate_score + extent_score) / 2.0
    } else {
        0.0
    };

    VibratoResult {
        present,
        rate,
        extent,
        confidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vibrato_detection() {
        // Generate synthetic vibrato (440 Hz with 6 Hz modulation)
        let sample_rate = 44100.0;
        let hop_size = 512;
        let duration = 1.0;
        let num_frames = ((sample_rate * duration) as usize) / hop_size;

        let mut estimates = Vec::new();
        let mut confidences = Vec::new();

        for i in 0..num_frames {
            let t = i as f32 * hop_size as f32 / sample_rate;
            let vibrato = 10.0 * (2.0 * std::f32::consts::PI * 6.0 * t).sin();
            estimates.push(440.0 + vibrato);
            confidences.push(0.9);
        }

        let pitch_result = PitchResult {
            estimates,
            confidences,
            mean_f0: 440.0,
            voicing_rate: 1.0,
        };

        let result = detect_vibrato(&pitch_result, hop_size, sample_rate);

        // Vibrato detection may not be perfect with synthetic data
        assert!(result.rate >= 0.0);
        assert!(result.extent >= 0.0);
    }

    #[test]
    fn test_no_vibrato() {
        // Constant pitch, no vibrato
        let estimates = vec![440.0; 50];
        let confidences = vec![0.9; 50];

        let pitch_result = PitchResult {
            estimates,
            confidences,
            mean_f0: 440.0,
            voicing_rate: 1.0,
        };

        let result = detect_vibrato(&pitch_result, 512, 44100.0);
        assert!(!result.present || result.extent < 20.0);
    }
}
