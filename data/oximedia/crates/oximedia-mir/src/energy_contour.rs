//! Short-time energy contour extraction and analysis.
//!
//! Computes the frame-wise RMS or peak energy of a mono audio signal and
//! provides utilities for smoothing, normalisation, and activity detection.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// EnergyMode
// ---------------------------------------------------------------------------

/// Method used to compute per-frame energy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnergyMode {
    /// Root-mean-square energy.
    Rms,
    /// Peak absolute sample value in the frame.
    Peak,
}

// ---------------------------------------------------------------------------
// EnergyContourConfig
// ---------------------------------------------------------------------------

/// Configuration for the energy contour extractor.
#[derive(Debug, Clone)]
pub struct EnergyContourConfig {
    /// Sample rate in Hz.
    pub sample_rate: f32,
    /// Analysis window size in samples.
    pub window_size: usize,
    /// Hop size in samples.
    pub hop_size: usize,
    /// Energy computation mode.
    pub mode: EnergyMode,
}

impl Default for EnergyContourConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            window_size: 1024,
            hop_size: 512,
            mode: EnergyMode::Rms,
        }
    }
}

// ---------------------------------------------------------------------------
// EnergyContour
// ---------------------------------------------------------------------------

/// The computed energy contour (one value per frame).
#[derive(Debug, Clone)]
pub struct EnergyContour {
    /// Per-frame energy values (linear scale).
    pub values: Vec<f32>,
    /// Hop size used during extraction.
    pub hop_size: usize,
    /// Sample rate used during extraction.
    pub sample_rate: f32,
}

impl EnergyContour {
    /// Number of frames.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the contour is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Maximum energy value.
    #[must_use]
    pub fn max_energy(&self) -> f32 {
        self.values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Minimum energy value.
    #[must_use]
    pub fn min_energy(&self) -> f32 {
        self.values.iter().copied().fold(f32::INFINITY, f32::min)
    }

    /// Mean energy value.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_energy(&self) -> f32 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f32>() / self.values.len() as f32
    }

    /// Return a normalised copy where the peak is 1.0.
    #[must_use]
    pub fn normalize(&self) -> Self {
        let peak = self.max_energy();
        if peak <= 0.0 {
            return self.clone();
        }
        Self {
            values: self.values.iter().map(|&v| v / peak).collect(),
            hop_size: self.hop_size,
            sample_rate: self.sample_rate,
        }
    }

    /// Apply a simple moving-average smoothing with the given kernel radius.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn smooth(&self, radius: usize) -> Self {
        if radius == 0 || self.values.is_empty() {
            return self.clone();
        }
        let n = self.values.len();
        let mut smoothed = Vec::with_capacity(n);
        for i in 0..n {
            let lo = i.saturating_sub(radius);
            let hi = (i + radius + 1).min(n);
            let sum: f32 = self.values[lo..hi].iter().sum();
            smoothed.push(sum / (hi - lo) as f32);
        }
        Self {
            values: smoothed,
            hop_size: self.hop_size,
            sample_rate: self.sample_rate,
        }
    }

    /// Convert energy to dB scale (relative to 1.0).
    #[must_use]
    pub fn to_db(&self) -> Vec<f32> {
        self.values
            .iter()
            .map(|&v| 20.0 * (v.max(1e-10)).log10())
            .collect()
    }

    /// Detect frames whose energy exceeds a threshold.
    /// Returns indices of active frames.
    #[must_use]
    pub fn activity_frames(&self, threshold: f32) -> Vec<usize> {
        self.values
            .iter()
            .enumerate()
            .filter(|(_, &v)| v >= threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Convert frame index to time in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn frame_to_time(&self, frame: usize) -> f32 {
        frame as f32 * self.hop_size as f32 / self.sample_rate
    }
}

// ---------------------------------------------------------------------------
// extract_energy
// ---------------------------------------------------------------------------

/// Extract the energy contour from a mono audio signal.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn extract_energy(samples: &[f32], config: &EnergyContourConfig) -> EnergyContour {
    if samples.is_empty() || config.hop_size == 0 || config.window_size == 0 {
        return EnergyContour {
            values: Vec::new(),
            hop_size: config.hop_size,
            sample_rate: config.sample_rate,
        };
    }

    let n_frames = samples.len().saturating_sub(config.window_size) / config.hop_size + 1;
    let mut values = Vec::with_capacity(n_frames);

    for i in 0..n_frames {
        let start = i * config.hop_size;
        let end = (start + config.window_size).min(samples.len());
        let frame = &samples[start..end];

        let energy = match config.mode {
            EnergyMode::Rms => {
                let sum_sq: f32 = frame.iter().map(|&s| s * s).sum();
                (sum_sq / frame.len() as f32).sqrt()
            }
            EnergyMode::Peak => frame.iter().map(|s| s.abs()).fold(0.0f32, f32::max),
        };
        values.push(energy);
    }

    EnergyContour {
        values,
        hop_size: config.hop_size,
        sample_rate: config.sample_rate,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> EnergyContourConfig {
        EnergyContourConfig::default()
    }

    #[test]
    fn test_extract_empty() {
        let c = extract_energy(&[], &default_cfg());
        assert!(c.is_empty());
    }

    #[test]
    fn test_extract_silence_rms() {
        let silence = vec![0.0f32; 4096];
        let c = extract_energy(&silence, &default_cfg());
        assert!(!c.is_empty());
        for &v in &c.values {
            assert!(v < 1e-6);
        }
    }

    #[test]
    fn test_extract_peak_mode() {
        let mut sig = vec![0.0f32; 2048];
        sig[100] = 0.75;
        let cfg = EnergyContourConfig {
            mode: EnergyMode::Peak,
            ..default_cfg()
        };
        let c = extract_energy(&sig, &cfg);
        assert!(c.max_energy() >= 0.74);
    }

    #[test]
    fn test_max_min_energy() {
        let sig = vec![0.5f32; 4096];
        let c = extract_energy(&sig, &default_cfg());
        assert!(c.max_energy() > 0.0);
        assert!(c.min_energy() >= 0.0);
    }

    #[test]
    fn test_mean_energy() {
        let sig = vec![1.0f32; 4096];
        let c = extract_energy(&sig, &default_cfg());
        // RMS of constant 1.0 is 1.0
        assert!((c.mean_energy() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_normalize() {
        let sig = vec![0.5f32; 4096];
        let c = extract_energy(&sig, &default_cfg());
        let n = c.normalize();
        assert!((n.max_energy() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_zeros() {
        let c = EnergyContour {
            values: vec![0.0; 3],
            hop_size: 512,
            sample_rate: 44100.0,
        };
        let n = c.normalize();
        assert_eq!(n.values, vec![0.0; 3]);
    }

    #[test]
    fn test_smooth() {
        let c = EnergyContour {
            values: vec![0.0, 0.0, 1.0, 0.0, 0.0],
            hop_size: 512,
            sample_rate: 44100.0,
        };
        let s = c.smooth(1);
        // centre value should be average of [0,1,0] = 0.333..
        assert!((s.values[2] - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_smooth_zero_radius() {
        let c = EnergyContour {
            values: vec![1.0, 2.0],
            hop_size: 512,
            sample_rate: 44100.0,
        };
        let s = c.smooth(0);
        assert_eq!(s.values, c.values);
    }

    #[test]
    fn test_to_db() {
        let c = EnergyContour {
            values: vec![1.0, 0.1],
            hop_size: 512,
            sample_rate: 44100.0,
        };
        let db = c.to_db();
        assert!((db[0] - 0.0).abs() < 0.01); // 20*log10(1) = 0
        assert!((db[1] - (-20.0)).abs() < 0.01); // 20*log10(0.1) = -20
    }

    #[test]
    fn test_activity_frames() {
        let c = EnergyContour {
            values: vec![0.1, 0.5, 0.8, 0.2, 0.9],
            hop_size: 512,
            sample_rate: 44100.0,
        };
        let active = c.activity_frames(0.5);
        assert_eq!(active, vec![1, 2, 4]);
    }

    #[test]
    fn test_frame_to_time() {
        let c = EnergyContour {
            values: vec![0.0; 10],
            hop_size: 512,
            sample_rate: 44100.0,
        };
        let t = c.frame_to_time(1);
        assert!((t - 512.0 / 44100.0).abs() < 1e-5);
    }

    #[test]
    fn test_len_matches_frames() {
        let sig = vec![0.5f32; 8192];
        let c = extract_energy(&sig, &default_cfg());
        assert_eq!(c.len(), c.values.len());
    }

    #[test]
    fn test_contour_len_empty() {
        let c = EnergyContour {
            values: Vec::new(),
            hop_size: 512,
            sample_rate: 44100.0,
        };
        assert_eq!(c.len(), 0);
        assert!(c.is_empty());
    }
}
