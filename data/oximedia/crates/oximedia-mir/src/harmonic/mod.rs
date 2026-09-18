//! Harmonic analysis and separation.

pub mod pitch;
pub mod separate;

pub use pitch::PitchClassProfile;
pub use separate::HarmonicSeparator;

use crate::types::HarmonicResult;
use crate::utils::stft;
use crate::MirResult;

/// Harmonic analyzer.
pub struct HarmonicAnalyzer {
    sample_rate: f32,
    window_size: usize,
    hop_size: usize,
}

impl HarmonicAnalyzer {
    /// Create a new harmonic analyzer.
    #[must_use]
    pub fn new(sample_rate: f32, window_size: usize, hop_size: usize) -> Self {
        Self {
            sample_rate,
            window_size,
            hop_size,
        }
    }

    /// Analyze harmonic features.
    ///
    /// # Errors
    ///
    /// Returns error if harmonic analysis fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(&self, signal: &[f32]) -> MirResult<HarmonicResult> {
        // Perform harmonic-percussive separation
        let separator = HarmonicSeparator::new(self.sample_rate, self.window_size, self.hop_size);
        let (harmonic_energy, percussive_energy) = separator.separate(signal)?;

        // Compute HPR ratio
        let total_harmonic: f32 = harmonic_energy.iter().sum();
        let total_percussive: f32 = percussive_energy.iter().sum();

        let hpr_ratio = if total_percussive > 0.0 {
            total_harmonic / (total_harmonic + total_percussive)
        } else {
            1.0
        };

        // Compute pitch class profile
        let pitch_analyzer = PitchClassProfile::new(self.sample_rate, self.window_size);
        let pitch_class_profile = pitch_analyzer.compute(signal)?;

        // Compute chroma features over time
        let chroma = self.compute_chroma_frames(signal)?;

        Ok(HarmonicResult {
            harmonic_energy,
            percussive_energy,
            hpr_ratio,
            pitch_class_profile,
            chroma,
        })
    }

    /// Compute chroma features for each frame.
    fn compute_chroma_frames(&self, signal: &[f32]) -> MirResult<Vec<Vec<f32>>> {
        let frames = stft(signal, self.window_size, self.hop_size)?;

        let mut chroma_frames = Vec::with_capacity(frames.len());

        for frame in &frames {
            let chroma = self.frame_to_chroma(frame);
            chroma_frames.push(chroma);
        }

        Ok(chroma_frames)
    }

    /// Convert FFT frame to 12-bin chroma vector.
    #[allow(clippy::cast_precision_loss)]
    fn frame_to_chroma(&self, frame: &[oxifft::Complex<f32>]) -> Vec<f32> {
        let mut chroma = vec![0.0; 12];
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
    fn test_harmonic_analyzer_creation() {
        let analyzer = HarmonicAnalyzer::new(44100.0, 2048, 512);
        assert_eq!(analyzer.sample_rate, 44100.0);
    }
}
