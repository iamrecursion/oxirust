//! Musical key detection using the Krumhansl-Schmuckler algorithm.
//!
//! Defines pitch classes, key modes, and the `KrumhanslSchmuckler` detector
//! which correlates a 12-element chroma vector with the standard
//! Krumhansl-Kessler tonal hierarchy profiles.

#![allow(dead_code)]

/// The 12 pitch classes of the chromatic scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PitchClass {
    /// C natural
    C,
    /// C sharp / D flat
    Cs,
    /// D natural
    D,
    /// D sharp / E flat
    Ds,
    /// E natural
    E,
    /// F natural
    F,
    /// F sharp / G flat
    Fs,
    /// G natural
    G,
    /// G sharp / A flat
    Gs,
    /// A natural
    A,
    /// A sharp / B flat
    As,
    /// B natural
    B,
}

impl PitchClass {
    /// Semitone index (C = 0, B = 11).
    #[must_use]
    pub fn semitone(self) -> u8 {
        match self {
            Self::C => 0,
            Self::Cs => 1,
            Self::D => 2,
            Self::Ds => 3,
            Self::E => 4,
            Self::F => 5,
            Self::Fs => 6,
            Self::G => 7,
            Self::Gs => 8,
            Self::A => 9,
            Self::As => 10,
            Self::B => 11,
        }
    }

    /// Human-readable note name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::C => "C",
            Self::Cs => "C#",
            Self::D => "D",
            Self::Ds => "D#",
            Self::E => "E",
            Self::F => "F",
            Self::Fs => "F#",
            Self::G => "G",
            Self::Gs => "G#",
            Self::A => "A",
            Self::As => "A#",
            Self::B => "B",
        }
    }

    /// Construct from a semitone index (mod 12).
    #[must_use]
    pub fn from_semitone(s: u8) -> Self {
        match s % 12 {
            0 => Self::C,
            1 => Self::Cs,
            2 => Self::D,
            3 => Self::Ds,
            4 => Self::E,
            5 => Self::F,
            6 => Self::Fs,
            7 => Self::G,
            8 => Self::Gs,
            9 => Self::A,
            10 => Self::As,
            _ => Self::B,
        }
    }
}

/// Musical mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyMode {
    /// Major (ionian) mode.
    Major,
    /// Natural minor (aeolian) mode.
    Minor,
}

impl KeyMode {
    /// Returns `"major"` or `"minor"`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Major => "major",
            Self::Minor => "minor",
        }
    }
}

/// A musical key — root pitch class plus mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MusicalKey {
    /// The tonic pitch class.
    pub root: PitchClass,
    /// Major or minor mode.
    pub mode: KeyMode,
}

impl MusicalKey {
    /// Create a new `MusicalKey`.
    #[must_use]
    pub fn new(root: PitchClass, mode: KeyMode) -> Self {
        Self { root, mode }
    }

    /// Format as `"<note> <mode>"` (e.g. `"C major"`, `"A# minor"`).
    #[must_use]
    pub fn display(&self) -> String {
        format!("{} {}", self.root.name(), self.mode.name())
    }

    /// Return the relative key.
    ///
    /// - For a major key the relative minor is 9 semitones above (3 semitones below).
    /// - For a minor key the relative major is 3 semitones above.
    #[must_use]
    pub fn relative_key(&self) -> MusicalKey {
        match self.mode {
            KeyMode::Major => {
                let semitone = (self.root.semitone() + 9) % 12;
                MusicalKey {
                    root: PitchClass::from_semitone(semitone),
                    mode: KeyMode::Minor,
                }
            }
            KeyMode::Minor => {
                let semitone = (self.root.semitone() + 3) % 12;
                MusicalKey {
                    root: PitchClass::from_semitone(semitone),
                    mode: KeyMode::Major,
                }
            }
        }
    }
}

/// Standard Krumhansl-Kessler major key profile.
pub const KEY_PROFILES_MAJOR: [f32; 12] = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];

/// Standard Krumhansl-Kessler minor key profile.
pub const KEY_PROFILES_MINOR: [f32; 12] = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];

/// Key detector using the Krumhansl-Schmuckler algorithm.
pub struct KrumhanslSchmuckler;

impl KrumhanslSchmuckler {
    /// Detect the musical key by correlating `chroma` with major/minor profiles.
    ///
    /// Evaluates all 24 keys (12 major + 12 minor) and returns the one with
    /// the highest Pearson correlation.
    #[must_use]
    pub fn detect(chroma: &[f32; 12]) -> MusicalKey {
        let mut best_key = MusicalKey::new(PitchClass::C, KeyMode::Major);
        let mut best_corr = f32::NEG_INFINITY;

        for root in 0_u8..12 {
            let major_corr = pearson_f32(
                chroma,
                &rotate_profile_f32(&KEY_PROFILES_MAJOR, root as usize),
            );
            let minor_corr = pearson_f32(
                chroma,
                &rotate_profile_f32(&KEY_PROFILES_MINOR, root as usize),
            );

            if major_corr > best_corr {
                best_corr = major_corr;
                best_key = MusicalKey::new(PitchClass::from_semitone(root), KeyMode::Major);
            }
            if minor_corr > best_corr {
                best_corr = minor_corr;
                best_key = MusicalKey::new(PitchClass::from_semitone(root), KeyMode::Minor);
            }
        }

        best_key
    }
}

/// Rotate a 12-element profile array by `shift` positions.
#[must_use]
fn rotate_profile_f32(profile: &[f32; 12], shift: usize) -> [f32; 12] {
    let mut out = [0.0_f32; 12];
    for i in 0..12 {
        out[(i + shift) % 12] = profile[i];
    }
    out
}

/// Pearson correlation coefficient for two 12-element arrays.
fn pearson_f32(a: &[f32; 12], b: &[f32; 12]) -> f32 {
    let mean_a: f32 = a.iter().sum::<f32>() / 12.0;
    let mean_b: f32 = b.iter().sum::<f32>() / 12.0;
    let mut num = 0.0_f32;
    let mut da2 = 0.0_f32;
    let mut db2 = 0.0_f32;
    for i in 0..12 {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        num += da * db;
        da2 += da * da;
        db2 += db * db;
    }
    let denom = (da2 * db2).sqrt();
    if denom < 1e-9 {
        0.0
    } else {
        num / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── PitchClass ────────────────────────────────────────────────────────────

    #[test]
    fn test_pitch_class_semitones_c() {
        assert_eq!(PitchClass::C.semitone(), 0);
    }

    #[test]
    fn test_pitch_class_semitones_b() {
        assert_eq!(PitchClass::B.semitone(), 11);
    }

    #[test]
    fn test_pitch_class_name_c_sharp() {
        assert_eq!(PitchClass::Cs.name(), "C#");
    }

    #[test]
    fn test_pitch_class_from_semitone_roundtrip() {
        for i in 0_u8..12 {
            let pc = PitchClass::from_semitone(i);
            assert_eq!(pc.semitone(), i);
        }
    }

    #[test]
    fn test_pitch_class_from_semitone_mod12() {
        assert_eq!(PitchClass::from_semitone(12), PitchClass::C);
        assert_eq!(PitchClass::from_semitone(13), PitchClass::Cs);
    }

    // ── KeyMode ───────────────────────────────────────────────────────────────

    #[test]
    fn test_key_mode_names() {
        assert_eq!(KeyMode::Major.name(), "major");
        assert_eq!(KeyMode::Minor.name(), "minor");
    }

    // ── MusicalKey ────────────────────────────────────────────────────────────

    #[test]
    fn test_musical_key_display_c_major() {
        let k = MusicalKey::new(PitchClass::C, KeyMode::Major);
        assert_eq!(k.display(), "C major");
    }

    #[test]
    fn test_musical_key_display_a_minor() {
        let k = MusicalKey::new(PitchClass::A, KeyMode::Minor);
        assert_eq!(k.display(), "A minor");
    }

    #[test]
    fn test_relative_key_c_major_to_a_minor() {
        let c_major = MusicalKey::new(PitchClass::C, KeyMode::Major);
        let rel = c_major.relative_key();
        assert_eq!(rel.root, PitchClass::A);
        assert_eq!(rel.mode, KeyMode::Minor);
    }

    #[test]
    fn test_relative_key_a_minor_to_c_major() {
        let a_minor = MusicalKey::new(PitchClass::A, KeyMode::Minor);
        let rel = a_minor.relative_key();
        assert_eq!(rel.root, PitchClass::C);
        assert_eq!(rel.mode, KeyMode::Major);
    }

    #[test]
    fn test_relative_key_roundtrip() {
        for root in 0_u8..12 {
            let k = MusicalKey::new(PitchClass::from_semitone(root), KeyMode::Major);
            let rel = k.relative_key().relative_key();
            assert_eq!(rel.root, k.root);
            assert_eq!(rel.mode, k.mode);
        }
    }

    // ── KrumhanslSchmuckler::detect ───────────────────────────────────────────

    #[test]
    fn test_detect_c_major_from_profile() {
        // Feed the C-major profile itself — should detect C major
        let result = KrumhanslSchmuckler::detect(&KEY_PROFILES_MAJOR);
        assert_eq!(result.root, PitchClass::C);
        assert_eq!(result.mode, KeyMode::Major);
    }

    #[test]
    fn test_detect_a_minor_from_profile() {
        // A-minor profile = KK_MINOR rotated by 9 semitones
        let mut chroma = [0.0_f32; 12];
        for i in 0..12 {
            chroma[(i + 9) % 12] = KEY_PROFILES_MINOR[i];
        }
        let result = KrumhanslSchmuckler::detect(&chroma);
        assert_eq!(result.root, PitchClass::A);
        assert_eq!(result.mode, KeyMode::Minor);
    }

    #[test]
    fn test_detect_returns_valid_key() {
        let chroma = [1.0_f32; 12];
        let result = KrumhanslSchmuckler::detect(&chroma);
        // Any key is valid — just check it doesn't panic
        let _ = result.display();
    }
}
