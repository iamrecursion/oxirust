//! Hiss removal using spectral gating.

use crate::error::RestoreResult;
use crate::utils::spectral::{window_coefficients, FftProcessor, WindowFunction};

/// Hiss remover configuration.
#[derive(Debug, Clone)]
pub struct HissRemoverConfig {
    /// High-pass filter frequency (Hz).
    pub highpass_freq: f32,
    /// Threshold for hiss gating (dB).
    pub threshold_db: f32,
    /// Reduction amount (dB).
    pub reduction_db: f32,
}

impl Default for HissRemoverConfig {
    fn default() -> Self {
        Self {
            highpass_freq: 4000.0,
            threshold_db: -50.0,
            reduction_db: -60.0,
        }
    }
}

/// Hiss remover.
#[derive(Debug)]
pub struct HissRemover {
    config: HissRemoverConfig,
    fft_size: usize,
    hop_size: usize,
}

impl HissRemover {
    /// Create a new hiss remover.
    #[must_use]
    pub fn new(config: HissRemoverConfig, fft_size: usize, hop_size: usize) -> Self {
        Self {
            config,
            fft_size,
            hop_size,
        }
    }

    /// Remove hiss from samples.
    pub fn process(&self, samples: &[f32], sample_rate: u32) -> RestoreResult<Vec<f32>> {
        if samples.len() < self.fft_size {
            return Ok(samples.to_vec());
        }

        let fft = FftProcessor::new(self.fft_size);

        // Precompute the analysis/synthesis window once.  The same Hann window
        // is applied on analysis and synthesis, so the per-sample contribution
        // for a pass-through bin is `w[i]²·x`.  Weighted overlap-add (WOLA)
        // reconstruction therefore normalises by the accumulated `Σ w[i]²`
        // (the window-overlap-squared sum), NOT by the raw frame count — the
        // latter leaves a position-dependent gain error and amplitude
        // modulation.
        let window = window_coefficients(self.fft_size, WindowFunction::Hann);

        let mut output = vec![0.0; samples.len()];
        let mut window_sum = vec![0.0f32; samples.len()];

        #[allow(clippy::cast_precision_loss)]
        let bin_width = sample_rate as f32 / self.fft_size as f32;
        let highpass_bin = (self.config.highpass_freq / bin_width) as usize;

        let mut pos = 0;
        while pos + self.fft_size <= samples.len() {
            let mut frame = samples[pos..pos + self.fft_size].to_vec();
            for (s, &w) in frame.iter_mut().zip(window.iter()) {
                *s *= w;
            }

            let spectrum = fft.forward(&frame)?;
            let mut magnitude = fft.magnitude(&spectrum);
            let phase = fft.phase(&spectrum);

            // Apply spectral gating only to high frequencies.
            //
            // The signal is real → the spectrum is Hermitian (`X[N-k] ==
            // conj(X[k])`).  We must attenuate a high-frequency bin `k` AND its
            // conjugate partner `N-k` by the same factor, otherwise the
            // reconstructed real signal loses half its amplitude at the
            // untouched frequencies.  Operate on the unique half `[0, N/2]` and
            // mirror each change onto `N-k`.
            let reduction = db_to_linear(self.config.reduction_db);
            let half = self.fft_size / 2;
            for i in highpass_bin..=half {
                magnitude[i] *= reduction;
                if i > 0 && i < half {
                    magnitude[self.fft_size - i] *= reduction;
                }
            }

            let processed_spectrum = FftProcessor::from_polar(&magnitude, &phase)?;
            let processed_frame = fft.inverse(&processed_spectrum)?;

            for (i, (&sample, &w)) in processed_frame.iter().zip(window.iter()).enumerate() {
                output[pos + i] += sample * w;
                window_sum[pos + i] += w * w;
            }

            pos += self.hop_size;
        }

        for (i, &wsum) in window_sum.iter().enumerate() {
            if wsum > f32::EPSILON {
                output[i] /= wsum;
            }
        }

        Ok(output)
    }
}

#[must_use]
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hiss_remover() {
        use rand::RngExt;
        let mut rng = rand::rng();

        let samples: Vec<f32> = (0..8192).map(|_| rng.random_range(-0.1..0.1)).collect();

        let remover = HissRemover::new(HissRemoverConfig::default(), 2048, 1024);
        let output = remover
            .process(&samples, 44100)
            .expect("should succeed in test");

        assert_eq!(output.len(), samples.len());
    }
}
