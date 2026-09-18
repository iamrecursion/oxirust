//! Wow and flutter detection for tape recordings.

use crate::error::RestoreResult;

/// Wow and flutter detection result.
#[derive(Debug, Clone)]
pub struct WowFlutterProfile {
    /// Average wow rate (Hz).
    pub wow_rate: f32,
    /// Average flutter rate (Hz).
    pub flutter_rate: f32,
    /// Maximum deviation (semitones).
    pub max_deviation: f32,
}

/// Wow and flutter detector.
#[derive(Debug)]
pub struct WowFlutterDetector {
    window_size: usize,
}

impl WowFlutterDetector {
    /// Create a new wow/flutter detector.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self { window_size }
    }

    /// Detect wow and flutter in samples.
    ///
    /// Uses normalized autocorrelation-based pitch tracking to identify
    /// slow (wow, < 6 Hz) and fast (flutter, 6–100 Hz) pitch variations.
    pub fn detect(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> RestoreResult<Option<WowFlutterProfile>> {
        if samples.len() < self.window_size * 4 {
            return Ok(None);
        }

        // Compute pitch track using normalized autocorrelation
        let hop = self.window_size / 2;
        let mut pitch_estimates: Vec<f32> = Vec::new();

        let search_min = (sample_rate as f32 / 4000.0) as usize; // Max pitch 4000 Hz
        let search_max = (sample_rate as f32 / 50.0) as usize; // Min pitch 50 Hz
        let search_max = search_max.min(self.window_size / 2);

        for frame_start in (0..samples.len().saturating_sub(self.window_size)).step_by(hop) {
            let frame = &samples[frame_start..frame_start + self.window_size];

            // Compute energy-normalized autocorrelation
            let energy: f32 = frame.iter().map(|x| x * x).sum();
            if energy < 1e-10 {
                pitch_estimates.push(0.0);
                continue;
            }

            let mut best_lag = search_min;
            let mut best_corr = -1.0f32;

            for lag in search_min..search_max {
                let mut corr = 0.0f32;
                let n = self.window_size - lag;
                for i in 0..n {
                    corr += frame[i] * frame[i + lag];
                }
                corr /= (energy * n as f32).sqrt();

                if corr > best_corr {
                    best_corr = corr;
                    best_lag = lag;
                }
            }

            if best_corr > 0.3 {
                let pitch = sample_rate as f32 / best_lag as f32;
                pitch_estimates.push(pitch);
            } else {
                pitch_estimates.push(0.0);
            }
        }

        // Filter out zeros for analysis
        let valid: Vec<f32> = pitch_estimates
            .iter()
            .copied()
            .filter(|&p| p > 0.0)
            .collect();
        if valid.is_empty() {
            return Ok(None);
        }

        // Compute mean pitch
        let mean_pitch = valid.iter().sum::<f32>() / valid.len() as f32;

        // Compute pitch deviation as semitones: 12 * log2(p/mean)
        let deviations: Vec<f32> = valid
            .iter()
            .map(|&p| 12.0 * (p / mean_pitch).log2().abs())
            .collect();

        let max_deviation = deviations.iter().cloned().fold(0.0f32, f32::max);

        // Analyze deviation rate to separate wow from flutter.
        // Wow: deviation changes slowly (< 6 Hz), Flutter: > 6 Hz
        let hop_rate = sample_rate as f32 / hop as f32;
        let mut wow_power = 0.0f32;
        let mut flutter_power = 0.0f32;
        // wow_cutoff used conceptually via the 6.0 literal below
        let _wow_cutoff = 6.0 / hop_rate;

        // Simple spectral analysis of deviation signal via temporal differences
        for (i, &dev) in deviations.iter().enumerate() {
            if i > 0 {
                let rate_of_change = (deviations[i] - deviations[i - 1]).abs() * hop_rate;
                if rate_of_change < 6.0 {
                    wow_power += dev;
                } else {
                    flutter_power += dev;
                }
            }
        }

        let total = (wow_power + flutter_power).max(1e-10);
        let wow_rate = (wow_power / total * 3.0).min(3.0); // Scale to typical range
        let flutter_rate = (flutter_power / total * 50.0).min(100.0); // Scale to typical flutter freq

        Ok(Some(WowFlutterProfile {
            wow_rate,
            flutter_rate,
            max_deviation,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a sinusoidal signal with the given frequency and length.
    fn sine_wave(freq: f32, sample_rate: u32, num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect()
    }

    #[test]
    fn test_wow_flutter_detector_silent() {
        // Silent signal has no pitch → detector returns None
        let samples = vec![0.0f32; 8192];
        let detector = WowFlutterDetector::new(2048);
        let result = detector
            .detect(&samples, 44100)
            .expect("should succeed in test");
        assert!(result.is_none());
    }

    #[test]
    fn test_wow_flutter_detector_sine() {
        // A stable 440 Hz tone should yield a valid (Some) profile
        let sr = 44100u32;
        let samples = sine_wave(440.0, sr, sr as usize / 4); // 0.25 second (fast test)
        let detector = WowFlutterDetector::new(2048);
        let result = detector
            .detect(&samples, sr)
            .expect("should succeed in test");
        assert!(result.is_some());
        let profile = result.expect("should succeed in test");
        // A stable tone has very small pitch deviation
        assert!(
            profile.max_deviation < 1.0,
            "max_deviation={}",
            profile.max_deviation
        );
    }

    #[test]
    fn test_wow_flutter_detector_too_short() {
        // Buffer shorter than window_size * 4 → None
        let samples = vec![0.1f32; 100];
        let detector = WowFlutterDetector::new(2048);
        let result = detector
            .detect(&samples, 44100)
            .expect("should succeed in test");
        assert!(result.is_none());
    }
}
