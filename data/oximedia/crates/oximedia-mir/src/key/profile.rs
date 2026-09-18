//! Key profiles for Krumhansl-Schmuckler algorithm.

/// Key profile containing weights for each pitch class.
#[derive(Debug, Clone)]
pub struct KeyProfile {
    /// Root note (0-11, C=0).
    pub root: u8,

    /// Major or minor mode.
    pub is_major: bool,

    /// Weights for each pitch class (0-11).
    pub weights: [f32; 12],
}

/// Krumhansl-Schmuckler key profiles for all 24 major and minor keys.
///
/// Based on empirical studies of key perception.
pub const KEY_PROFILES: [KeyProfile; 24] = [
    // C major
    KeyProfile {
        root: 0,
        is_major: true,
        weights: [
            6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
        ],
    },
    // C# major
    KeyProfile {
        root: 1,
        is_major: true,
        weights: [
            2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29,
        ],
    },
    // D major
    KeyProfile {
        root: 2,
        is_major: true,
        weights: [
            2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66,
        ],
    },
    // D# major
    KeyProfile {
        root: 3,
        is_major: true,
        weights: [
            3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39,
        ],
    },
    // E major
    KeyProfile {
        root: 4,
        is_major: true,
        weights: [
            2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19,
        ],
    },
    // F major
    KeyProfile {
        root: 5,
        is_major: true,
        weights: [
            5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52,
        ],
    },
    // F# major
    KeyProfile {
        root: 6,
        is_major: true,
        weights: [
            2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
        ],
    },
    // G major
    KeyProfile {
        root: 7,
        is_major: true,
        weights: [
            4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38,
        ],
    },
    // G# major
    KeyProfile {
        root: 8,
        is_major: true,
        weights: [
            4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33,
        ],
    },
    // A major
    KeyProfile {
        root: 9,
        is_major: true,
        weights: [
            2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48,
        ],
    },
    // A# major
    KeyProfile {
        root: 10,
        is_major: true,
        weights: [
            3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23,
        ],
    },
    // B major
    KeyProfile {
        root: 11,
        is_major: true,
        weights: [
            2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35,
        ],
    },
    // C minor
    KeyProfile {
        root: 0,
        is_major: false,
        weights: [
            6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
        ],
    },
    // C# minor
    KeyProfile {
        root: 1,
        is_major: false,
        weights: [
            3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34,
        ],
    },
    // D minor
    KeyProfile {
        root: 2,
        is_major: false,
        weights: [
            3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69,
        ],
    },
    // D# minor
    KeyProfile {
        root: 3,
        is_major: false,
        weights: [
            2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98,
        ],
    },
    // E minor
    KeyProfile {
        root: 4,
        is_major: false,
        weights: [
            3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75,
        ],
    },
    // F minor
    KeyProfile {
        root: 5,
        is_major: false,
        weights: [
            4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54,
        ],
    },
    // F# minor
    KeyProfile {
        root: 6,
        is_major: false,
        weights: [
            2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
        ],
    },
    // G minor
    KeyProfile {
        root: 7,
        is_major: false,
        weights: [
            3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60,
        ],
    },
    // G# minor
    KeyProfile {
        root: 8,
        is_major: false,
        weights: [
            2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38,
        ],
    },
    // A minor
    KeyProfile {
        root: 9,
        is_major: false,
        weights: [
            5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52,
        ],
    },
    // A# minor
    KeyProfile {
        root: 10,
        is_major: false,
        weights: [
            3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68,
        ],
    },
    // B minor
    KeyProfile {
        root: 11,
        is_major: false,
        weights: [
            2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33,
        ],
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_profiles_count() {
        assert_eq!(KEY_PROFILES.len(), 24);
    }

    #[test]
    fn test_major_minor_balance() {
        let major_count = KEY_PROFILES.iter().filter(|p| p.is_major).count();
        let minor_count = KEY_PROFILES.iter().filter(|p| !p.is_major).count();
        assert_eq!(major_count, 12);
        assert_eq!(minor_count, 12);
    }

    #[test]
    fn test_profile_weights_length() {
        for profile in &KEY_PROFILES {
            assert_eq!(profile.weights.len(), 12);
        }
    }
}
