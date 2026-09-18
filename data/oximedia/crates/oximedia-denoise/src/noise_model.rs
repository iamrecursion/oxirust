#![allow(dead_code)]
//! Statistical noise modeling and classification.
//!
//! Models different noise distributions (Gaussian, Poisson, salt-and-pepper)
//! and provides tools for characterizing the noise present in a signal so that
//! downstream denoisers can choose the optimal strategy.

use std::fmt;

/// Type of noise distribution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NoiseType {
    /// Additive white Gaussian noise.
    Gaussian,
    /// Poisson (shot) noise -- signal-dependent.
    Poisson,
    /// Salt-and-pepper impulse noise.
    SaltAndPepper,
    /// Speckle (multiplicative) noise.
    Speckle,
    /// Mixed / unknown noise.
    Mixed,
}

impl fmt::Display for NoiseType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gaussian => write!(f, "Gaussian"),
            Self::Poisson => write!(f, "Poisson"),
            Self::SaltAndPepper => write!(f, "Salt-and-Pepper"),
            Self::Speckle => write!(f, "Speckle"),
            Self::Mixed => write!(f, "Mixed"),
        }
    }
}

/// Parameters for a Gaussian noise model.
#[derive(Clone, Debug)]
pub struct GaussianModel {
    /// Mean of the noise (ideally 0 for zero-mean noise).
    pub mean: f64,
    /// Standard deviation.
    pub std_dev: f64,
}

impl GaussianModel {
    /// Create a new Gaussian noise model.
    pub fn new(mean: f64, std_dev: f64) -> Self {
        Self {
            mean,
            std_dev: std_dev.abs(),
        }
    }

    /// Variance (sigma squared).
    pub fn variance(&self) -> f64 {
        self.std_dev * self.std_dev
    }

    /// Signal-to-noise ratio given a signal power.
    pub fn snr(&self, signal_power: f64) -> f64 {
        if self.variance() < 1e-15 {
            return f64::INFINITY;
        }
        signal_power / self.variance()
    }

    /// SNR in decibels.
    #[allow(clippy::cast_precision_loss)]
    pub fn snr_db(&self, signal_power: f64) -> f64 {
        let ratio = self.snr(signal_power);
        if ratio <= 0.0 {
            return f64::NEG_INFINITY;
        }
        10.0 * ratio.log10()
    }
}

/// Parameters for a salt-and-pepper noise model.
#[derive(Clone, Debug)]
pub struct SaltPepperModel {
    /// Probability of a salt (max value) pixel.
    pub salt_prob: f64,
    /// Probability of a pepper (min value) pixel.
    pub pepper_prob: f64,
}

impl SaltPepperModel {
    /// Create a new salt-and-pepper model.
    pub fn new(salt_prob: f64, pepper_prob: f64) -> Self {
        Self {
            salt_prob: salt_prob.clamp(0.0, 1.0),
            pepper_prob: pepper_prob.clamp(0.0, 1.0),
        }
    }

    /// Total impulse probability.
    pub fn total_prob(&self) -> f64 {
        (self.salt_prob + self.pepper_prob).min(1.0)
    }
}

/// Combined noise characterization result.
#[derive(Clone, Debug)]
pub struct NoiseProfile {
    /// Primary noise type detected.
    pub primary_type: NoiseType,
    /// Estimated Gaussian component.
    pub gaussian: GaussianModel,
    /// Estimated salt-and-pepper component (if any).
    pub salt_pepper: Option<SaltPepperModel>,
    /// Overall noise severity (0.0 = clean, 1.0 = very noisy).
    pub severity: f64,
    /// Confidence of the classification (0.0 .. 1.0).
    pub confidence: f64,
}

impl NoiseProfile {
    /// Create a profile indicating clean signal.
    pub fn clean() -> Self {
        Self {
            primary_type: NoiseType::Gaussian,
            gaussian: GaussianModel::new(0.0, 0.0),
            salt_pepper: None,
            severity: 0.0,
            confidence: 1.0,
        }
    }

    /// Whether the signal is considered clean (severity below threshold).
    pub fn is_clean(&self, threshold: f64) -> bool {
        self.severity < threshold
    }
}

/// Estimate Gaussian noise parameters from a flat (uniform) region.
#[allow(clippy::cast_precision_loss)]
pub fn estimate_gaussian(samples: &[f64]) -> GaussianModel {
    if samples.is_empty() {
        return GaussianModel::new(0.0, 0.0);
    }
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let variance = samples
        .iter()
        .map(|&x| (x - mean) * (x - mean))
        .sum::<f64>()
        / n;
    GaussianModel::new(mean, variance.sqrt())
}

/// Estimate salt-and-pepper noise by counting extreme values.
#[allow(clippy::cast_precision_loss)]
pub fn estimate_salt_pepper(samples: &[f64], min_val: f64, max_val: f64) -> SaltPepperModel {
    if samples.is_empty() {
        return SaltPepperModel::new(0.0, 0.0);
    }
    let n = samples.len() as f64;
    let eps = (max_val - min_val) * 0.01;
    let salt = samples
        .iter()
        .filter(|&&v| (v - max_val).abs() < eps)
        .count() as f64
        / n;
    let pepper = samples
        .iter()
        .filter(|&&v| (v - min_val).abs() < eps)
        .count() as f64
        / n;
    SaltPepperModel::new(salt, pepper)
}

/// Classify noise type from basic statistics.
#[allow(clippy::cast_precision_loss)]
pub fn classify_noise(samples: &[f64], min_val: f64, max_val: f64) -> NoiseProfile {
    let gaussian = estimate_gaussian(samples);
    let sp = estimate_salt_pepper(samples, min_val, max_val);

    let range = max_val - min_val;
    let normalized_std = if range > 0.0 {
        gaussian.std_dev / range
    } else {
        0.0
    };

    // Decide primary type
    let (primary_type, confidence) = if sp.total_prob() > 0.05 {
        if normalized_std > 0.05 {
            (NoiseType::Mixed, 0.6)
        } else {
            (NoiseType::SaltAndPepper, 0.8)
        }
    } else if normalized_std > 0.01 {
        (NoiseType::Gaussian, 0.9)
    } else {
        (NoiseType::Gaussian, 1.0)
    };

    let severity = (normalized_std * 5.0 + sp.total_prob()).min(1.0);

    let salt_pepper_opt = if sp.total_prob() > 0.001 {
        Some(sp)
    } else {
        None
    };

    NoiseProfile {
        primary_type,
        gaussian,
        salt_pepper: salt_pepper_opt,
        severity,
        confidence,
    }
}

/// Compute the kurtosis of a sample set (excess kurtosis).
///
/// Gaussian noise has excess kurtosis near 0; heavy-tailed noise is positive.
#[allow(clippy::cast_precision_loss)]
pub fn excess_kurtosis(samples: &[f64]) -> f64 {
    if samples.len() < 4 {
        return 0.0;
    }
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let m2 = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    if m2 < 1e-15 {
        return 0.0;
    }
    let m4 = samples.iter().map(|&x| (x - mean).powi(4)).sum::<f64>() / n;
    m4 / (m2 * m2) - 3.0
}

/// Compute the skewness of a sample set.
#[allow(clippy::cast_precision_loss)]
pub fn skewness(samples: &[f64]) -> f64 {
    if samples.len() < 3 {
        return 0.0;
    }
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let m2 = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    if m2 < 1e-15 {
        return 0.0;
    }
    let m3 = samples.iter().map(|&x| (x - mean).powi(3)).sum::<f64>() / n;
    m3 / m2.powf(1.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_type_display() {
        assert_eq!(NoiseType::Gaussian.to_string(), "Gaussian");
        assert_eq!(NoiseType::SaltAndPepper.to_string(), "Salt-and-Pepper");
        assert_eq!(NoiseType::Mixed.to_string(), "Mixed");
    }

    #[test]
    fn test_gaussian_model() {
        let m = GaussianModel::new(0.0, 5.0);
        assert!((m.variance() - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gaussian_snr() {
        let m = GaussianModel::new(0.0, 1.0);
        assert!((m.snr(100.0) - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gaussian_snr_db() {
        let m = GaussianModel::new(0.0, 1.0);
        let db = m.snr_db(100.0);
        assert!((db - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_gaussian_snr_zero_noise() {
        let m = GaussianModel::new(0.0, 0.0);
        assert!(m.snr(1.0).is_infinite());
    }

    #[test]
    fn test_salt_pepper_model() {
        let sp = SaltPepperModel::new(0.02, 0.03);
        assert!((sp.total_prob() - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn test_salt_pepper_clamp() {
        let sp = SaltPepperModel::new(2.0, -1.0);
        assert!((sp.salt_prob - 1.0).abs() < f64::EPSILON);
        assert!((sp.pepper_prob - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_noise_profile_clean() {
        let p = NoiseProfile::clean();
        assert!(p.is_clean(0.01));
        assert!((p.severity - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_estimate_gaussian_constant() {
        let samples = vec![5.0; 100];
        let m = estimate_gaussian(&samples);
        assert!((m.mean - 5.0).abs() < f64::EPSILON);
        assert!(m.std_dev < 1e-10);
    }

    #[test]
    fn test_estimate_gaussian_spread() {
        let samples: Vec<f64> = (0..100).map(|i| (i % 10) as f64).collect();
        let m = estimate_gaussian(&samples);
        assert!(m.std_dev > 0.0);
    }

    #[test]
    fn test_estimate_salt_pepper() {
        let mut samples = vec![128.0; 100];
        samples[0] = 0.0;
        samples[1] = 0.0;
        samples[98] = 255.0;
        samples[99] = 255.0;
        let sp = estimate_salt_pepper(&samples, 0.0, 255.0);
        assert!((sp.salt_prob - 0.02).abs() < 0.001);
        assert!((sp.pepper_prob - 0.02).abs() < 0.001);
    }

    #[test]
    fn test_classify_noise_clean() {
        let samples = vec![128.0; 1000];
        let profile = classify_noise(&samples, 0.0, 255.0);
        assert!(profile.severity < 0.05);
    }

    #[test]
    fn test_excess_kurtosis_uniform() {
        // Uniform distribution has excess kurtosis ~ -1.2
        let samples: Vec<f64> = (0..10000).map(|i| (i % 100) as f64).collect();
        let k = excess_kurtosis(&samples);
        assert!(k < 0.0); // Should be negative for uniform
    }

    #[test]
    fn test_skewness_symmetric() {
        let samples: Vec<f64> = (-50..=50).map(|i| i as f64).collect();
        let s = skewness(&samples);
        assert!(s.abs() < 0.01);
    }

    #[test]
    fn test_estimate_gaussian_empty() {
        let m = estimate_gaussian(&[]);
        assert!((m.mean - 0.0).abs() < f64::EPSILON);
        assert!((m.std_dev - 0.0).abs() < f64::EPSILON);
    }
}
