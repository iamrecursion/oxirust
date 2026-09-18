//! Loudness normalization for consistent volume.

use bytes::Bytes;
use oximedia_audio::frame::AudioBuffer;
use oximedia_audio::loudness::filter::KWeightFilter;

use crate::error::AccessResult;

const SAMPLE_RATE: f64 = 48_000.0;

/// Normalizes loudness to target level (EBU R128 / ATSC A/85).
pub struct LoudnessNormalizer {
    target_lufs: f32,
}

impl LoudnessNormalizer {
    /// Create a new loudness normalizer.
    #[must_use]
    pub const fn new(target_lufs: f32) -> Self {
        Self { target_lufs }
    }

    /// Measure current loudness in LUFS using ITU-R BS.1770-4 K-weighting.
    #[must_use]
    pub fn measure_loudness(&self, audio: &AudioBuffer) -> f32 {
        let mut filter = KWeightFilter::new(SAMPLE_RATE);

        let mean_square: f64 = match audio {
            AudioBuffer::Interleaved(data) => {
                let samples = bytes_to_f32_samples(data);
                if samples.is_empty() {
                    return -70.0;
                }
                #[allow(clippy::cast_precision_loss)]
                let count = samples.len() as f64;
                let sum_sq: f64 = samples
                    .into_iter()
                    .map(|s| {
                        let filtered = filter.process(f64::from(s));
                        filtered * filtered
                    })
                    .sum();
                sum_sq / count
            }
            AudioBuffer::Planar(planes) => {
                if planes.is_empty() {
                    return -70.0;
                }
                let mut total_sq = 0.0_f64;
                let mut total_count = 0usize;
                for plane in planes {
                    let mut ch_filter = KWeightFilter::new(SAMPLE_RATE);
                    let samples = bytes_to_f32_samples(plane);
                    total_count += samples.len();
                    for s in samples {
                        let filtered = ch_filter.process(f64::from(s));
                        total_sq += filtered * filtered;
                    }
                }
                if total_count == 0 {
                    return -70.0;
                }
                #[allow(clippy::cast_precision_loss)]
                let count = total_count as f64;
                total_sq / count
            }
        };

        if mean_square <= 0.0 {
            return -70.0;
        }

        #[allow(clippy::cast_possible_truncation)]
        let lufs = (-0.691 + 10.0 * mean_square.log10()) as f32;
        lufs
    }

    /// Normalize audio to target loudness.
    pub fn normalize(&self, audio: &AudioBuffer) -> AccessResult<AudioBuffer> {
        let current_lufs = self.measure_loudness(audio);
        let gain_db = self.target_lufs - current_lufs;
        let linear_gain = 10.0_f32.powf(gain_db / 20.0);

        let result = match audio {
            AudioBuffer::Interleaved(data) => {
                let samples = bytes_to_f32_samples(data);
                let scaled: Vec<u8> = samples
                    .into_iter()
                    .flat_map(|s| (s * linear_gain).clamp(-1.0, 1.0).to_le_bytes())
                    .collect();
                AudioBuffer::Interleaved(Bytes::from(scaled))
            }
            AudioBuffer::Planar(planes) => {
                let scaled_planes: Vec<Bytes> = planes
                    .iter()
                    .map(|plane| {
                        let samples = bytes_to_f32_samples(plane);
                        let scaled: Vec<u8> = samples
                            .into_iter()
                            .flat_map(|s| (s * linear_gain).clamp(-1.0, 1.0).to_le_bytes())
                            .collect();
                        Bytes::from(scaled)
                    })
                    .collect();
                AudioBuffer::Planar(scaled_planes)
            }
        };

        Ok(result)
    }

    /// Calculate required gain adjustment.
    #[must_use]
    pub fn calculate_gain(&self, current_lufs: f32) -> f32 {
        self.target_lufs - current_lufs
    }

    /// Get target LUFS.
    #[must_use]
    pub const fn target_lufs(&self) -> f32 {
        self.target_lufs
    }
}

/// Convert raw bytes to f32 samples (little-endian).
fn bytes_to_f32_samples(data: &Bytes) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

impl Default for LoudnessNormalizer {
    fn default() -> Self {
        Self::new(-23.0) // EBU R128 standard
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalizer_creation() {
        let normalizer = LoudnessNormalizer::new(-24.0);
        assert!((normalizer.target_lufs() + 24.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_calculate_gain() {
        let normalizer = LoudnessNormalizer::new(-23.0);
        let gain = normalizer.calculate_gain(-26.0);
        assert!((gain - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_normalize() {
        let normalizer = LoudnessNormalizer::default();
        let audio = AudioBuffer::Interleaved(Bytes::from(vec![0u8; 48000 * 4]));
        let result = normalizer.normalize(&audio);
        assert!(result.is_ok());
    }
}
