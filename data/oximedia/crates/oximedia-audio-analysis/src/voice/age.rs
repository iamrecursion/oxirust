//! Age estimation from voice characteristics.

/// Age group classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgeGroup {
    /// Child (0-12 years)
    Child,
    /// Teenager (13-19 years)
    Teen,
    /// Young adult (20-35 years)
    YoungAdult,
    /// Middle-aged adult (36-55 years)
    MiddleAged,
    /// Senior (56+ years)
    Senior,
    /// Unknown/ambiguous
    Unknown,
}

/// Estimate age group from voice characteristics.
///
/// Age affects voice in several ways:
/// - Children: High F0 (200-400 Hz), high formants
/// - Teens: Changing voice, variable F0
/// - Young adults: Stable F0, low jitter/shimmer
/// - Middle-aged: Slightly increased jitter
/// - Seniors: Increased jitter/shimmer, breathiness
///
/// # Arguments
/// * `f0` - Fundamental frequency in Hz
/// * `formants` - Formant frequencies
/// * `jitter` - Pitch variation (0-1)
/// * `shimmer` - Amplitude variation (0-1)
///
/// # Returns
/// Estimated age group
#[must_use]
pub fn estimate_age(f0: f32, formants: &[f32], jitter: f32, shimmer: f32) -> AgeGroup {
    // Very high F0 suggests child
    if f0 > 250.0 && formants.first().is_some_and(|&f1| f1 > 700.0) {
        return AgeGroup::Child;
    }

    // High jitter and shimmer suggest senior
    if jitter > 0.02 && shimmer > 0.08 {
        return AgeGroup::Senior;
    }

    // Moderate jitter suggests middle-aged
    if jitter > 0.01 && jitter <= 0.02 {
        return AgeGroup::MiddleAged;
    }

    // Low jitter and shimmer suggest young adult
    if jitter < 0.01 && shimmer < 0.05 {
        // F0 range can help distinguish teen from young adult
        if !(100.0..=180.0).contains(&f0) {
            return AgeGroup::Teen;
        }
        return AgeGroup::YoungAdult;
    }

    AgeGroup::Unknown
}

/// Estimate age group with confidence score.
#[must_use]
pub fn estimate_age_with_confidence(
    f0: f32,
    formants: &[f32],
    jitter: f32,
    shimmer: f32,
) -> (AgeGroup, f32) {
    let age_group = estimate_age(f0, formants, jitter, shimmer);

    // Compute confidence based on how clearly features match expectations
    let confidence = match age_group {
        AgeGroup::Child => {
            if f0 > 300.0 {
                0.9
            } else {
                0.6
            }
        }
        AgeGroup::Senior => {
            if jitter > 0.03 && shimmer > 0.1 {
                0.8
            } else {
                0.6
            }
        }
        AgeGroup::YoungAdult => {
            if jitter < 0.008 && shimmer < 0.04 {
                0.7
            } else {
                0.5
            }
        }
        AgeGroup::MiddleAged => 0.6,
        AgeGroup::Teen => 0.5,
        AgeGroup::Unknown => 0.0,
    };

    (age_group, confidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_age_estimation() {
        // Child
        let child = estimate_age(300.0, &[750.0, 2000.0], 0.005, 0.03);
        assert_eq!(child, AgeGroup::Child);

        // Young adult
        let young = estimate_age(150.0, &[500.0, 1500.0], 0.005, 0.03);
        assert_eq!(young, AgeGroup::YoungAdult);

        // Senior
        let senior = estimate_age(140.0, &[500.0, 1500.0], 0.025, 0.10);
        assert_eq!(senior, AgeGroup::Senior);
    }

    #[test]
    fn test_age_with_confidence() {
        let (age, conf) = estimate_age_with_confidence(320.0, &[750.0, 2000.0], 0.005, 0.03);
        assert_eq!(age, AgeGroup::Child);
        assert!(conf > 0.7);
    }
}
