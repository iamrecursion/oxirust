//! Musical key detection using the Krumhansl-Kessler profile method.
//!
//! Implements the Krumhansl-Schmuckler algorithm: compute a chroma vector
//! from the audio, then correlate with major and minor key profiles to
//! identify the most probable key.

#![allow(dead_code)]

/// 12 pitch classes (C, C#, D, ... B).
pub const NUM_PITCH_CLASSES: usize = 12;

/// Names of the 12 pitch classes.
pub const PITCH_CLASS_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// Krumhansl-Kessler major key profile.
///
/// These values represent the perceived stability/goodness of fit of each
/// scale degree in a major key context.
pub const KK_MAJOR_PROFILE: [f64; 12] = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];

/// Krumhansl-Kessler minor key profile.
pub const KK_MINOR_PROFILE: [f64; 12] = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];

/// A musical key, represented as pitch class + mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MusicalKey {
    /// Root pitch class (0 = C, 1 = C#, …, 11 = B).
    pub root: u8,
    /// Mode: major or minor.
    pub mode: Mode,
}

/// Musical mode (major or minor).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mode {
    /// Major mode (ionian).
    Major,
    /// Natural minor mode (aeolian).
    Minor,
}

impl Mode {
    /// Returns "major" or "minor".
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Major => "major",
            Self::Minor => "minor",
        }
    }
}

impl std::fmt::Display for MusicalKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {}",
            PITCH_CLASS_NAMES[self.root as usize],
            self.mode.name()
        )
    }
}

impl MusicalKey {
    /// Return a human-readable name for this key, e.g. "C major" or "A minor".
    #[must_use]
    pub fn name(&self) -> String {
        self.to_string()
    }

    /// Return the Camelot Wheel code for this key (DJ-friendly notation).
    ///
    /// The Camelot Wheel assigns each key a number (1–12) and a letter:
    /// - `B` suffix for major keys (outer ring)
    /// - `A` suffix for minor keys (inner ring)
    ///
    /// Harmonically adjacent keys share numbers (±1) or the same number
    /// (relative major/minor), making it easy to mix tracks in key.
    #[must_use]
    pub fn camelot_code(&self) -> String {
        // Camelot numbers for each pitch class.
        // Major keys (B suffix): C=8B, G=9B, D=10B, A=11B, E=12B, B=1B,
        //                        F#=2B, Db=3B, Ab=4B, Eb=5B, Bb=6B, F=7B
        // Minor keys (A suffix): Am=8A, Em=9A, Bm=10A, F#m=11A, C#m=12A,
        //                        G#m=1A, Ebm=2A, Bbm=3A, Fm=4A, Cm=5A,
        //                        Gm=6A, Dm=7A
        const MAJOR_CAMELOT: [u8; 12] = [8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1];
        const MINOR_CAMELOT: [u8; 12] = [5, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10];

        let (number, letter) = match self.mode {
            Mode::Major => (MAJOR_CAMELOT[self.root as usize], 'B'),
            Mode::Minor => (MINOR_CAMELOT[self.root as usize], 'A'),
        };
        format!("{number}{letter}")
    }

    /// Return the relative key — same key signature, opposite mode.
    ///
    /// - Major → relative minor is 3 semitones below (root − 3 mod 12)
    /// - Minor → relative major is 3 semitones above (root + 3 mod 12)
    #[must_use]
    pub fn relative_key(&self) -> MusicalKey {
        match self.mode {
            Mode::Major => MusicalKey {
                root: (self.root as i8 - 3).rem_euclid(12) as u8,
                mode: Mode::Minor,
            },
            Mode::Minor => MusicalKey {
                root: (self.root as i8 + 3).rem_euclid(12) as u8,
                mode: Mode::Major,
            },
        }
    }

    /// Return the parallel key — same root pitch, opposite mode.
    ///
    /// E.g. C major → C minor, A minor → A major.
    #[must_use]
    pub fn parallel_key(&self) -> MusicalKey {
        MusicalKey {
            root: self.root,
            mode: match self.mode {
                Mode::Major => Mode::Minor,
                Mode::Minor => Mode::Major,
            },
        }
    }
}

/// Result of key detection.
#[derive(Debug, Clone)]
pub struct KeyDetectionResult {
    /// Best-matching key.
    pub key: MusicalKey,
    /// Pearson correlation coefficient for the best match (higher = more confident).
    pub correlation: f64,
    /// Confidence in [0, 1] normalised from correlation scores.
    pub confidence: f64,
    /// All 24 key scores (12 major + 12 minor).
    pub all_scores: Vec<(MusicalKey, f64)>,
}

/// Compute the mean of a slice.
fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

/// Pearson correlation between two equal-length slices.
#[must_use]
pub fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    let ma = mean(a);
    let mb = mean(b);
    let mut num = 0.0_f64;
    let mut da2 = 0.0_f64;
    let mut db2 = 0.0_f64;
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        num += da * db;
        da2 += da * da;
        db2 += db * db;
    }
    let denom = (da2 * db2).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        num / denom
    }
}

/// Rotate a 12-element array by `shift` positions to the right.
#[must_use]
pub fn rotate_profile(profile: &[f64; 12], shift: usize) -> [f64; 12] {
    let mut out = [0.0_f64; 12];
    for i in 0..12 {
        out[(i + shift) % 12] = profile[i];
    }
    out
}

/// Detect the musical key from a chroma vector using Krumhansl-Kessler profiles.
///
/// # Arguments
///
/// * `chroma` - 12-element chroma energy vector (one value per pitch class).
///
/// The chroma vector is typically computed from a magnitude spectrum by
/// summing energy in each pitch-class bin.
///
/// # Returns
///
/// `KeyDetectionResult` with the best-matching key and all scores.
#[must_use]
pub fn detect_key_from_chroma(chroma: &[f64; 12]) -> KeyDetectionResult {
    let mut scores: Vec<(MusicalKey, f64)> = Vec::with_capacity(24);

    for root in 0_u8..12 {
        let major_profile = rotate_profile(&KK_MAJOR_PROFILE, root as usize);
        let minor_profile = rotate_profile(&KK_MINOR_PROFILE, root as usize);

        let major_corr = pearson_correlation(chroma, &major_profile);
        let minor_corr = pearson_correlation(chroma, &minor_profile);

        scores.push((
            MusicalKey {
                root,
                mode: Mode::Major,
            },
            major_corr,
        ));
        scores.push((
            MusicalKey {
                root,
                mode: Mode::Minor,
            },
            minor_corr,
        ));
    }

    // Find the best key
    let best = scores
        .iter()
        .copied()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((
            MusicalKey {
                root: 0,
                mode: Mode::Major,
            },
            0.0,
        ));

    // Normalise confidence: map correlation from [-1, 1] to [0, 1]
    let confidence = (best.1 + 1.0) / 2.0;

    KeyDetectionResult {
        key: best.0,
        correlation: best.1,
        confidence,
        all_scores: scores,
    }
}

/// Build a simple chroma vector from a power spectrum.
///
/// Maps each FFT bin to its pitch class and accumulates energy.
///
/// # Arguments
///
/// * `spectrum` - Power spectrum (magnitude squared).
/// * `sample_rate` - Audio sample rate in Hz.
/// * `fft_size` - Size of the FFT (number of bins * 2).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn chroma_from_spectrum(spectrum: &[f64], sample_rate: f64, fft_size: usize) -> [f64; 12] {
    let mut chroma = [0.0_f64; 12];
    let hz_per_bin = sample_rate / fft_size as f64;
    let a4_hz = 440.0_f64;

    for (bin, &energy) in spectrum.iter().enumerate() {
        let freq = bin as f64 * hz_per_bin;
        if !(20.0..=5000.0).contains(&freq) {
            continue;
        }
        // Convert frequency to pitch class
        let semitones_from_a4 = 12.0 * (freq / a4_hz).log2();
        // A4 is pitch class 9 (A)
        let pitch_class = ((semitones_from_a4.round() as i64 + 9).rem_euclid(12)) as usize;
        chroma[pitch_class] += energy;
    }
    chroma
}

/// Normalise a chroma vector to unit sum.
#[must_use]
pub fn normalise_chroma(chroma: &[f64; 12]) -> [f64; 12] {
    let sum: f64 = chroma.iter().sum();
    if sum < 1e-12 {
        return *chroma;
    }
    let mut out = *chroma;
    for v in &mut out {
        *v /= sum;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_pitch_class_names_count() {
        assert_eq!(PITCH_CLASS_NAMES.len(), 12);
    }

    #[test]
    fn test_mode_names() {
        assert_eq!(Mode::Major.name(), "major");
        assert_eq!(Mode::Minor.name(), "minor");
    }

    #[test]
    fn test_musical_key_display() {
        let key = MusicalKey {
            root: 0,
            mode: Mode::Major,
        };
        assert_eq!(key.to_string(), "C major");
        let key_am = MusicalKey {
            root: 9,
            mode: Mode::Minor,
        };
        assert_eq!(key_am.to_string(), "A minor");
    }

    #[test]
    fn test_pearson_correlation_identical() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r = pearson_correlation(&a, &a);
        assert!(approx_eq(r, 1.0, 1e-10));
    }

    #[test]
    fn test_pearson_correlation_opposite() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b: Vec<f64> = a.iter().map(|x| 6.0 - x).collect();
        let r = pearson_correlation(&a, &b);
        assert!(approx_eq(r, -1.0, 1e-10));
    }

    #[test]
    fn test_pearson_empty() {
        let r = pearson_correlation(&[], &[]);
        assert!(approx_eq(r, 0.0, 1e-10));
    }

    #[test]
    fn test_rotate_profile_zero() {
        let rotated = rotate_profile(&KK_MAJOR_PROFILE, 0);
        assert_eq!(rotated, KK_MAJOR_PROFILE);
    }

    #[test]
    fn test_rotate_profile_12_is_identity() {
        let rotated = rotate_profile(&KK_MAJOR_PROFILE, 12);
        assert_eq!(rotated, KK_MAJOR_PROFILE);
    }

    #[test]
    fn test_rotate_profile_shifts_correctly() {
        let rotated = rotate_profile(&KK_MAJOR_PROFILE, 1);
        // First element of original should now be at index 1
        assert!(approx_eq(rotated[1], KK_MAJOR_PROFILE[0], 1e-10));
    }

    #[test]
    fn test_detect_key_c_major_profile() {
        // Feed the exact C major profile as the chroma vector
        let result = detect_key_from_chroma(&KK_MAJOR_PROFILE);
        assert_eq!(result.key.root, 0);
        assert_eq!(result.key.mode, Mode::Major);
        assert!(result.correlation > 0.9);
    }

    #[test]
    fn test_detect_key_a_minor_profile() {
        // A minor is root=9
        let a_minor = rotate_profile(&KK_MINOR_PROFILE, 9);
        let result = detect_key_from_chroma(&a_minor);
        assert_eq!(result.key.root, 9);
        assert_eq!(result.key.mode, Mode::Minor);
    }

    #[test]
    fn test_detect_key_all_scores_count() {
        let chroma = [1.0_f64; 12];
        let result = detect_key_from_chroma(&chroma);
        assert_eq!(result.all_scores.len(), 24);
    }

    #[test]
    fn test_normalise_chroma_sums_to_one() {
        let chroma = [1.0, 2.0, 3.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let norm = normalise_chroma(&chroma);
        let sum: f64 = norm.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-10));
    }

    #[test]
    fn test_normalise_zero_chroma_unchanged() {
        let chroma = [0.0_f64; 12];
        let norm = normalise_chroma(&chroma);
        assert_eq!(norm, chroma);
    }

    #[test]
    fn test_confidence_range() {
        let chroma = KK_MAJOR_PROFILE;
        let result = detect_key_from_chroma(&chroma);
        assert!((0.0..=1.0).contains(&result.confidence));
    }

    #[test]
    fn test_chroma_from_spectrum_empty() {
        let spectrum: Vec<f64> = vec![];
        let chroma = chroma_from_spectrum(&spectrum, 44100.0, 2048);
        let sum: f64 = chroma.iter().sum();
        assert!(approx_eq(sum, 0.0, 1e-10));
    }

    // ── MusicalKey new methods ─────────────────────────────────────────────────

    #[test]
    fn test_musical_key_name_c_major() {
        let key = MusicalKey {
            root: 0,
            mode: Mode::Major,
        };
        assert_eq!(key.name(), "C major");
    }

    #[test]
    fn test_musical_key_name_a_minor() {
        let key = MusicalKey {
            root: 9,
            mode: Mode::Minor,
        };
        assert_eq!(key.name(), "A minor");
    }

    #[test]
    fn test_camelot_code_c_major() {
        let key = MusicalKey {
            root: 0,
            mode: Mode::Major,
        };
        assert_eq!(key.camelot_code(), "8B");
    }

    #[test]
    fn test_camelot_code_a_minor() {
        let key = MusicalKey {
            root: 9,
            mode: Mode::Minor,
        };
        assert_eq!(key.camelot_code(), "8A");
    }

    #[test]
    fn test_relative_key_c_major_is_a_minor() {
        let c_major = MusicalKey {
            root: 0,
            mode: Mode::Major,
        };
        let rel = c_major.relative_key();
        assert_eq!(
            rel.root, 9,
            "Relative of C major should be A minor (root=9)"
        );
        assert_eq!(rel.mode, Mode::Minor);
    }

    #[test]
    fn test_relative_key_a_minor_is_c_major() {
        let a_minor = MusicalKey {
            root: 9,
            mode: Mode::Minor,
        };
        let rel = a_minor.relative_key();
        assert_eq!(
            rel.root, 0,
            "Relative of A minor should be C major (root=0)"
        );
        assert_eq!(rel.mode, Mode::Major);
    }

    #[test]
    fn test_parallel_key_c_major_is_c_minor() {
        let c_major = MusicalKey {
            root: 0,
            mode: Mode::Major,
        };
        let par = c_major.parallel_key();
        assert_eq!(
            par.root, 0,
            "Parallel of C major should be C minor (root=0)"
        );
        assert_eq!(par.mode, Mode::Minor);
    }

    #[test]
    fn test_parallel_key_d_major_is_d_minor() {
        let d_major = MusicalKey {
            root: 2,
            mode: Mode::Major,
        };
        let par = d_major.parallel_key();
        assert_eq!(
            par.root, 2,
            "Parallel of D major should be D minor (root=2)"
        );
        assert_eq!(par.mode, Mode::Minor);
    }

    #[test]
    fn test_camelot_code_g_major() {
        let key = MusicalKey {
            root: 7,
            mode: Mode::Major,
        };
        assert_eq!(key.camelot_code(), "9B");
    }

    #[test]
    fn test_relative_then_relative_returns_original() {
        // Relative of relative should return the original key
        let key = MusicalKey {
            root: 5,
            mode: Mode::Major,
        };
        let rel_rel = key.relative_key().relative_key();
        assert_eq!(rel_rel.root, key.root);
        assert_eq!(rel_rel.mode, key.mode);
    }
}
