//! Robustness testing attacks.
//!
//! This module simulates common signal processing attacks to test
//! watermark robustness.

/// Simulated MP3 compression attack.
#[must_use]
pub fn mp3_compression(samples: &[f32], _bitrate: u32) -> Vec<f32> {
    // Simplified MP3 compression simulation
    // In production, this would use actual MP3 encoding/decoding

    samples
        .chunks(128)
        .flat_map(|chunk| {
            // Simulate DCT-based compression artifacts
            let mut compressed = chunk.to_vec();

            // Add slight quantization noise
            for sample in &mut compressed {
                let quantized = (*sample * 100.0).round() / 100.0;
                *sample = quantized;
            }

            compressed
        })
        .collect()
}

/// Resampling attack.
#[must_use]
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }

    #[allow(clippy::cast_precision_loss)]
    let ratio = to_rate as f32 / from_rate as f32;
    #[allow(clippy::cast_precision_loss)]
    let new_len = (samples.len() as f32 * ratio) as usize;

    (0..new_len)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let src_idx = i as f32 / ratio;
            let idx0 = src_idx.floor() as usize;
            let idx1 = (idx0 + 1).min(samples.len() - 1);
            let frac = src_idx.fract();

            // Linear interpolation
            samples[idx0] * (1.0 - frac) + samples[idx1] * frac
        })
        .collect()
}

/// Low-pass filtering attack.
#[must_use]
pub fn lowpass_filter(samples: &[f32], cutoff_hz: f32, sample_rate: u32) -> Vec<f32> {
    use std::f32::consts::PI;

    // Simple first-order IIR low-pass filter
    #[allow(clippy::cast_precision_loss)]
    let rc = 1.0 / (2.0 * PI * cutoff_hz);
    #[allow(clippy::cast_precision_loss)]
    let dt = 1.0 / sample_rate as f32;
    let alpha = dt / (rc + dt);

    let mut filtered = Vec::with_capacity(samples.len());
    let mut prev = 0.0f32;

    for &sample in samples {
        let output = prev + alpha * (sample - prev);
        filtered.push(output);
        prev = output;
    }

    filtered
}

/// High-pass filtering attack.
#[must_use]
pub fn highpass_filter(samples: &[f32], cutoff_hz: f32, sample_rate: u32) -> Vec<f32> {
    use std::f32::consts::PI;

    #[allow(clippy::cast_precision_loss)]
    let rc = 1.0 / (2.0 * PI * cutoff_hz);
    #[allow(clippy::cast_precision_loss)]
    let dt = 1.0 / sample_rate as f32;
    let alpha = rc / (rc + dt);

    let mut filtered = Vec::with_capacity(samples.len());
    let mut prev_input = 0.0f32;
    let mut prev_output = 0.0f32;

    for &sample in samples {
        let output = alpha * (prev_output + sample - prev_input);
        filtered.push(output);
        prev_input = sample;
        prev_output = output;
    }

    filtered
}

/// Additive white Gaussian noise attack.
#[must_use]
pub fn add_noise(samples: &[f32], snr_db: f32) -> Vec<f32> {
    // Calculate signal power
    let signal_power: f32 = samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32;

    // Calculate noise power from SNR
    let noise_power = signal_power / 10.0f32.powf(snr_db / 10.0);
    let noise_std = noise_power.sqrt();

    let mut rng = scirs2_core::random::Random::seed(0xDEAD_CAFE);

    samples
        .iter()
        .map(|&s| {
            // Box-Muller transform for Gaussian noise
            let u1 = (rng.random_f64() as f32).max(1e-10).min(1.0 - f32::EPSILON);
            let u2 = rng.random_f64() as f32;
            let noise =
                noise_std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            s + noise
        })
        .collect()
}

/// Time stretching attack (change duration without changing pitch).
#[must_use]
pub fn time_stretch(samples: &[f32], factor: f32) -> Vec<f32> {
    #[allow(clippy::cast_precision_loss)]
    let new_len = (samples.len() as f32 * factor) as usize;

    (0..new_len)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let src_idx = i as f32 / factor;
            let idx0 = src_idx.floor() as usize;
            let idx1 = (idx0 + 1).min(samples.len() - 1);
            let frac = src_idx.fract();

            samples[idx0] * (1.0 - frac) + samples[idx1] * frac
        })
        .collect()
}

/// Pitch shifting attack.
#[must_use]
pub fn pitch_shift(samples: &[f32], semitones: f32) -> Vec<f32> {
    // Simplified pitch shift using resampling
    // Real pitch shift would use phase vocoder or similar
    let factor = 2.0f32.powf(semitones / 12.0);
    time_stretch(samples, 1.0 / factor)
}

/// Amplitude scaling attack.
#[must_use]
pub fn amplitude_scale(samples: &[f32], factor: f32) -> Vec<f32> {
    samples.iter().map(|&s| s * factor).collect()
}

/// Dynamic range compression attack.
#[must_use]
pub fn compress(samples: &[f32], threshold: f32, ratio: f32) -> Vec<f32> {
    samples
        .iter()
        .map(|&s| {
            let abs_s = s.abs();
            if abs_s > threshold {
                let excess = abs_s - threshold;
                let compressed = threshold + excess / ratio;
                s.signum() * compressed
            } else {
                s
            }
        })
        .collect()
}

/// Cropping attack (remove beginning).
#[must_use]
pub fn crop(samples: &[f32], start: usize, end: usize) -> Vec<f32> {
    let start = start.min(samples.len());
    let end = end.min(samples.len());
    samples[start..end].to_vec()
}

/// Jitter attack (random sample delays).
#[must_use]
pub fn jitter(samples: &[f32], max_delay: usize) -> Vec<f32> {
    let mut rng = scirs2_core::random::Random::seed(0xBEEF_FACE);
    let mut jittered = samples.to_vec();

    for i in max_delay..samples.len() {
        let delay = (rng.random_f64() * (max_delay + 1) as f64) as usize;
        let delay = delay.min(max_delay);
        if i >= delay {
            jittered[i] = samples[i - delay];
        }
    }

    jittered
}

/// Inversion attack (flip polarity).
#[must_use]
pub fn invert(samples: &[f32]) -> Vec<f32> {
    samples.iter().map(|&s| -s).collect()
}

/// Echo addition attack.
#[must_use]
pub fn add_echo(samples: &[f32], delay: usize, decay: f32) -> Vec<f32> {
    let mut echoed = samples.to_vec();

    for i in delay..samples.len() {
        echoed[i] += samples[i - delay] * decay;
    }

    echoed
}

/// Robustness test suite.
pub struct RobustnessTest {
    attacks: Vec<AttackType>,
}

/// Types of attacks used in watermark robustness testing.
#[derive(Debug, Clone)]
pub enum AttackType {
    /// MP3 compression attack with specified bitrate in kbps.
    Mp3 {
        /// Bitrate in kbps
        bitrate: u32,
    },
    /// Resampling attack that changes the sample rate.
    Resample {
        /// Original sample rate in Hz
        from_rate: u32,
        /// Target sample rate in Hz
        to_rate: u32,
    },
    /// Low-pass filter attack.
    LowPass {
        /// Cutoff frequency in Hz
        cutoff_hz: f32,
        /// Sample rate in Hz
        sample_rate: u32,
    },
    /// High-pass filter attack.
    HighPass {
        /// Cutoff frequency in Hz
        cutoff_hz: f32,
        /// Sample rate in Hz
        sample_rate: u32,
    },
    /// Additive white Gaussian noise attack.
    Noise {
        /// Signal-to-noise ratio in dB
        snr_db: f32,
    },
    /// Time stretching attack.
    TimeStretch {
        /// Time stretch factor (> 1 for slower playback)
        factor: f32,
    },
    /// Pitch shifting attack.
    PitchShift {
        /// Number of semitones to shift
        semitones: f32,
    },
    /// Amplitude scaling attack.
    AmplitudeScale {
        /// Scaling factor
        factor: f32,
    },
    /// Dynamic range compression attack.
    Compress {
        /// Compression threshold
        threshold: f32,
        /// Compression ratio
        ratio: f32,
    },
    /// Cropping attack that removes samples.
    Crop {
        /// Starting sample index
        start: usize,
        /// Ending sample index
        end: usize,
    },
    /// Time jitter attack.
    Jitter {
        /// Maximum jitter delay in samples
        max_delay: usize,
    },
    /// Phase inversion attack.
    Invert,
    /// Echo addition attack.
    Echo {
        /// Echo delay in samples
        delay: usize,
        /// Echo decay factor
        decay: f32,
    },
}

impl RobustnessTest {
    /// Create a new robustness test suite.
    #[must_use]
    pub fn new() -> Self {
        Self {
            attacks: Vec::new(),
        }
    }

    /// Add an attack to the test suite.
    pub fn add_attack(&mut self, attack: AttackType) {
        self.attacks.push(attack);
    }

    /// Run all attacks on samples.
    #[must_use]
    pub fn run(&self, samples: &[f32]) -> Vec<(AttackType, Vec<f32>)> {
        self.attacks
            .iter()
            .map(|attack| {
                let attacked = match attack {
                    AttackType::Mp3 { bitrate } => mp3_compression(samples, *bitrate),
                    AttackType::Resample { from_rate, to_rate } => {
                        resample(samples, *from_rate, *to_rate)
                    }
                    AttackType::LowPass {
                        cutoff_hz,
                        sample_rate,
                    } => lowpass_filter(samples, *cutoff_hz, *sample_rate),
                    AttackType::HighPass {
                        cutoff_hz,
                        sample_rate,
                    } => highpass_filter(samples, *cutoff_hz, *sample_rate),
                    AttackType::Noise { snr_db } => add_noise(samples, *snr_db),
                    AttackType::TimeStretch { factor } => time_stretch(samples, *factor),
                    AttackType::PitchShift { semitones } => pitch_shift(samples, *semitones),
                    AttackType::AmplitudeScale { factor } => amplitude_scale(samples, *factor),
                    AttackType::Compress { threshold, ratio } => {
                        compress(samples, *threshold, *ratio)
                    }
                    AttackType::Crop { start, end } => crop(samples, *start, *end),
                    AttackType::Jitter { max_delay } => jitter(samples, *max_delay),
                    AttackType::Invert => invert(samples),
                    AttackType::Echo { delay, decay } => add_echo(samples, *delay, *decay),
                };

                (attack.clone(), attacked)
            })
            .collect()
    }

    /// Create standard robustness test suite.
    #[must_use]
    pub fn standard_suite(sample_rate: u32) -> Self {
        let mut suite = Self::new();

        // Compression attacks
        suite.add_attack(AttackType::Mp3 { bitrate: 128 });
        suite.add_attack(AttackType::Mp3 { bitrate: 64 });

        // Resampling attacks
        suite.add_attack(AttackType::Resample {
            from_rate: sample_rate,
            to_rate: 48000,
        });
        suite.add_attack(AttackType::Resample {
            from_rate: 48000,
            to_rate: sample_rate,
        });

        // Filtering attacks
        suite.add_attack(AttackType::LowPass {
            cutoff_hz: 8000.0,
            sample_rate,
        });
        suite.add_attack(AttackType::HighPass {
            cutoff_hz: 100.0,
            sample_rate,
        });

        // Noise attacks
        suite.add_attack(AttackType::Noise { snr_db: 30.0 });
        suite.add_attack(AttackType::Noise { snr_db: 20.0 });

        // Time domain attacks
        suite.add_attack(AttackType::TimeStretch { factor: 1.1 });
        suite.add_attack(AttackType::PitchShift { semitones: 2.0 });

        // Amplitude attacks
        suite.add_attack(AttackType::AmplitudeScale { factor: 0.8 });
        suite.add_attack(AttackType::Compress {
            threshold: 0.5,
            ratio: 4.0,
        });

        suite
    }
}

impl Default for RobustnessTest {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mp3_compression() {
        let samples: Vec<f32> = vec![0.5; 1000];
        let compressed = mp3_compression(&samples, 128);
        assert_eq!(compressed.len(), samples.len());
    }

    #[test]
    fn test_resampling() {
        let samples: Vec<f32> = vec![0.5; 44100];
        let resampled = resample(&samples, 44100, 48000);
        assert!(resampled.len() > samples.len());
    }

    #[test]
    fn test_lowpass_filter() {
        let samples: Vec<f32> = vec![0.5; 1000];
        let filtered = lowpass_filter(&samples, 1000.0, 44100);
        assert_eq!(filtered.len(), samples.len());
    }

    #[test]
    fn test_noise_addition() {
        let samples: Vec<f32> = vec![0.5; 1000];
        let noisy = add_noise(&samples, 20.0);
        assert_eq!(noisy.len(), samples.len());
        assert_ne!(noisy, samples);
    }

    #[test]
    fn test_time_stretch() {
        let samples: Vec<f32> = vec![0.5; 1000];
        let stretched = time_stretch(&samples, 1.5);
        assert_eq!(stretched.len(), 1500);
    }

    #[test]
    fn test_amplitude_scale() {
        let samples: Vec<f32> = vec![0.5; 1000];
        let scaled = amplitude_scale(&samples, 2.0);
        assert!(scaled.iter().all(|&s| (s - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_robustness_suite() {
        let suite = RobustnessTest::standard_suite(44100);
        let samples: Vec<f32> = vec![0.5; 44100];

        let results = suite.run(&samples);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_compression() {
        let samples: Vec<f32> = vec![0.8, 0.3, -0.7, 0.2];
        let compressed = compress(&samples, 0.5, 4.0);
        assert_eq!(compressed.len(), samples.len());
    }

    #[test]
    fn test_inversion() {
        let samples: Vec<f32> = vec![0.5, -0.3, 0.8];
        let inverted = invert(&samples);
        assert_eq!(inverted, vec![-0.5, 0.3, -0.8]);
    }
}
