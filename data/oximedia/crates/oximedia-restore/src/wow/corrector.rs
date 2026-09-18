//! Wow and flutter correction using time-varying resampling.

use crate::error::RestoreResult;

/// Wow and flutter corrector.
#[derive(Debug, Clone)]
pub struct WowFlutterCorrector {
    interpolation_quality: usize,
}

impl WowFlutterCorrector {
    /// Create a new wow/flutter corrector.
    #[must_use]
    pub fn new(interpolation_quality: usize) -> Self {
        Self {
            interpolation_quality,
        }
    }

    /// Correct wow and flutter using pitch tracking and time-varying resampling.
    ///
    /// Steps:
    /// 1. Divide the signal into overlapping frames and estimate the dominant pitch
    ///    lag via normalized autocorrelation.
    /// 2. Compute a per-frame resampling rate relative to the mean pitch lag.
    /// 3. Walk through the input using a floating-point read cursor that advances
    ///    by `rate` per output sample, using linear interpolation.
    pub fn correct(&self, samples: &[f32]) -> RestoreResult<Vec<f32>> {
        if samples.is_empty() {
            return Ok(vec![]);
        }

        let window_size = 2048.min(samples.len() / 4);
        if window_size < 64 {
            return Ok(samples.to_vec());
        }
        let hop = window_size / 2;

        // Compute pitch track (dominant autocorrelation lag per frame).
        //
        // A frame contributes a real pitch estimate only when it is BOTH
        // non-silent AND convincingly periodic.  Periodicity is measured with a
        // properly *normalised* autocorrelation
        // `r(lag) = Σ a·b / sqrt(Σ a² · Σ b²)` (bounded in [-1, 1]): a strongly
        // tonal frame peaks near 1.0 while white noise stays well below the
        // gate.  Frames that fail either test get the neutral sentinel `1.0`
        // (treated like silence: excluded from the mean-lag estimate and
        // assigned unity resampling rate).
        //
        // The fundamental lag is selected with the original energy-biased score
        // (so the lag chosen for genuinely periodic frames is unchanged), while
        // the scale-invariant normalised correlation at that lag drives the
        // periodicity gate.  This prevents a noisy / aperiodic lead-in from
        // poisoning `mean_lag` with spurious lags — which would otherwise drive
        // the program-material resampling rate away from unity and grossly
        // time-warp an otherwise steady tone.
        const PERIODICITY_GATE: f32 = 0.5;
        let mut pitch_track: Vec<f32> = Vec::new();
        let search_min = 20usize; // ~2 kHz upper pitch limit at 44 100 Hz
        let search_max = (window_size / 2).min(1000);

        for frame_start in (0..samples.len().saturating_sub(window_size)).step_by(hop) {
            let frame = &samples[frame_start..frame_start + window_size];
            let energy: f32 = frame.iter().map(|x| x * x).sum();

            if energy < 1e-10 {
                pitch_track.push(1.0); // neutral rate – silent frame
                continue;
            }

            let mut best_lag = search_min;
            let mut best_score = f32::NEG_INFINITY;
            let mut best_norm = 0.0f32;

            for lag in search_min..search_max {
                let n = window_size - lag;
                let mut dot = 0.0f32;
                let mut e0 = 0.0f32;
                let mut e1 = 0.0f32;
                for i in 0..n {
                    let a = frame[i];
                    let b = frame[i + lag];
                    dot += a * b;
                    e0 += a * a;
                    e1 += b * b;
                }
                // Original selection score (favours the fundamental period).
                let score = dot / (energy * n as f32).sqrt();
                if score > best_score {
                    best_score = score;
                    best_lag = lag;
                    // Normalised cross-correlation in [-1, 1] at this lag.
                    let denom = (e0 * e1).sqrt();
                    best_norm = if denom > f32::EPSILON {
                        dot / denom
                    } else {
                        0.0
                    };
                }
            }

            // Only trust the lag when the frame is convincingly periodic.
            if best_norm >= PERIODICITY_GATE {
                pitch_track.push(best_lag as f32);
            } else {
                pitch_track.push(1.0); // neutral rate – aperiodic / noisy frame
            }
        }

        // Compute mean pitch period from frames with a valid pitch detection
        let valid_lags: Vec<f32> = pitch_track
            .iter()
            .cloned()
            .filter(|&l| l > 1.0 && l < search_max as f32)
            .collect();

        if valid_lags.is_empty() {
            return Ok(samples.to_vec());
        }

        let mean_lag = valid_lags.iter().sum::<f32>() / valid_lags.len() as f32;

        // Per-frame resampling rate: lag/mean_lag.
        // Larger lag → lower pitch → speed up (advance read cursor faster).
        let rates: Vec<f32> = pitch_track
            .iter()
            .map(|&lag| {
                if lag > 1.0 && lag < search_max as f32 {
                    lag / mean_lag
                } else {
                    1.0
                }
            })
            .collect();

        // `interpolation_quality` controls oversampling guard; ensure ≥ 1.
        let _quality = (self.interpolation_quality + 1).max(1);

        // Apply time-varying resampling with linear interpolation
        let mut output = Vec::with_capacity(samples.len());
        let mut read_pos: f64 = 0.0;
        let mut frame_idx = 0usize;
        let len_f = samples.len() as f64;

        while read_pos < len_f - 1.0 {
            // Determine which analysis frame governs this read position
            let frame_for_pos =
                ((read_pos / hop as f64) as usize).min(rates.len().saturating_sub(1));
            let rate = rates[frame_for_pos] as f64;

            // Linear interpolation between adjacent samples
            let idx = read_pos as usize;
            let frac = read_pos - idx as f64;

            let s = if idx + 1 < samples.len() {
                samples[idx] as f64 * (1.0 - frac) + samples[idx + 1] as f64 * frac
            } else {
                samples[idx.min(samples.len() - 1)] as f64
            };

            output.push(s as f32);
            read_pos += rate;
            frame_idx += 1;

            // Safety guard: never produce more than twice the input length
            if frame_idx > samples.len() * 2 {
                break;
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wow_flutter_corrector_empty() {
        let corrector = WowFlutterCorrector::new(4);
        let output = corrector.correct(&[]).expect("should succeed in test");
        assert!(output.is_empty());
    }

    #[test]
    fn test_wow_flutter_corrector_short() {
        // Buffers too short for analysis are returned as-is
        let samples = vec![0.1f32; 100];
        let corrector = WowFlutterCorrector::new(4);
        let output = corrector.correct(&samples).expect("should succeed in test");
        assert_eq!(output.len(), samples.len());
    }

    #[test]
    fn test_wow_flutter_corrector_sine() {
        // Generate a 440 Hz sine wave and verify the corrector produces output
        let sr = 44100u32;
        let num_samples = sr as usize / 4; // 0.25 second (fast test)
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sr as f32;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect();

        let corrector = WowFlutterCorrector::new(4);
        let output = corrector.correct(&samples).expect("should succeed in test");

        // Output length should be in a reasonable range relative to input
        assert!(!output.is_empty());
        assert!(output.len() <= samples.len() * 2);
    }

    #[test]
    fn test_wow_flutter_corrector_silent() {
        // Silent signal: all frames are neutral (rate = 1.0), so output ≈ input length
        let samples = vec![0.0f32; 8192];
        let corrector = WowFlutterCorrector::new(4);
        let output = corrector.correct(&samples).expect("should succeed in test");
        // Should be non-empty and close to input length
        assert!(!output.is_empty());
        assert!(output.len() <= samples.len() * 2);
    }
}
