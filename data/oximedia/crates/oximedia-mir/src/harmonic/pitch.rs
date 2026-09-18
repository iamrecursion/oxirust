//! Pitch class profile computation.

use crate::utils::stft;
use crate::MirResult;

/// Pitch class profile computer.
pub struct PitchClassProfile {
    sample_rate: f32,
    window_size: usize,
}

impl PitchClassProfile {
    /// Create a new pitch class profile computer.
    #[must_use]
    pub fn new(sample_rate: f32, window_size: usize) -> Self {
        Self {
            sample_rate,
            window_size,
        }
    }

    /// Compute average pitch class profile from signal.
    ///
    /// # Errors
    ///
    /// Returns error if computation fails.
    pub fn compute(&self, signal: &[f32]) -> MirResult<Vec<f32>> {
        let hop_size = self.window_size / 4;
        let frames = stft(signal, self.window_size, hop_size)?;

        let mut pitch_class_profile = vec![0.0; 12];

        for frame in &frames {
            let chroma = self.frame_to_chroma(frame);
            for (i, &value) in chroma.iter().enumerate() {
                pitch_class_profile[i] += value;
            }
        }

        // Normalize by number of frames
        let num_frames = frames.len() as f32;
        if num_frames > 0.0 {
            for value in &mut pitch_class_profile {
                *value /= num_frames;
            }
        }

        Ok(pitch_class_profile)
    }

    /// Convert FFT frame to 12-bin chroma vector.
    #[allow(clippy::cast_precision_loss)]
    fn frame_to_chroma(&self, frame: &[oxifft::Complex<f32>]) -> [f32; 12] {
        let mut chroma = [0.0; 12];
        let num_bins = frame.len() / 2;
        let ref_freq = 16.35; // C0

        for (bin, complex) in frame[1..num_bins].iter().enumerate() {
            let magnitude = complex.norm();
            let freq = (bin + 1) as f32 * self.sample_rate / self.window_size as f32;

            if freq < 20.0 {
                continue;
            }

            let pitch_class = self.freq_to_pitch_class(freq, ref_freq);
            chroma[pitch_class] += magnitude;
        }

        // Normalize
        let sum: f32 = chroma.iter().sum();
        if sum > 0.0 {
            for c in &mut chroma {
                *c /= sum;
            }
        }

        chroma
    }

    /// Convert frequency to pitch class.
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    fn freq_to_pitch_class(&self, freq: f32, ref_freq: f32) -> usize {
        let semitones = 12.0 * (freq / ref_freq).log2();
        (semitones.round() as i32).rem_euclid(12) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pitch_class_profile_creation() {
        let profile = PitchClassProfile::new(44100.0, 2048);
        assert_eq!(profile.sample_rate, 44100.0);
    }
}
