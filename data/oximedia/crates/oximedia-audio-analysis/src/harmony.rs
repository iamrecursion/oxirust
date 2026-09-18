//! Harmonic analysis: pitch classes, musical keys, chords, and key detection.
//!
//! Implements a Krumhansl-Schmuckler-inspired key detector and supporting
//! music theory types.

/// The twelve pitch classes in equal temperament.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PitchClass {
    /// C (do)
    C,
    /// C# / Db
    CSharp,
    /// D (re)
    D,
    /// D# / Eb
    DSharp,
    /// E (mi)
    E,
    /// F (fa)
    F,
    /// F# / Gb
    FSharp,
    /// G (sol)
    G,
    /// G# / Ab
    GSharp,
    /// A (la)
    A,
    /// A# / Bb
    ASharp,
    /// B (si)
    B,
}

impl PitchClass {
    /// Returns the number of semitones above C (0 = C, 11 = B).
    #[must_use]
    pub fn semitones(&self) -> u8 {
        match self {
            Self::C => 0,
            Self::CSharp => 1,
            Self::D => 2,
            Self::DSharp => 3,
            Self::E => 4,
            Self::F => 5,
            Self::FSharp => 6,
            Self::G => 7,
            Self::GSharp => 8,
            Self::A => 9,
            Self::ASharp => 10,
            Self::B => 11,
        }
    }

    /// Returns the name of the pitch class as a string slice.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::C => "C",
            Self::CSharp => "C#",
            Self::D => "D",
            Self::DSharp => "D#",
            Self::E => "E",
            Self::F => "F",
            Self::FSharp => "F#",
            Self::G => "G",
            Self::GSharp => "G#",
            Self::A => "A",
            Self::ASharp => "A#",
            Self::B => "B",
        }
    }

    /// Convert a semitone number (mod 12) back to a `PitchClass`.
    #[must_use]
    pub fn from_semitones(semitones: u8) -> Self {
        match semitones % 12 {
            1 => Self::CSharp,
            2 => Self::D,
            3 => Self::DSharp,
            4 => Self::E,
            5 => Self::F,
            6 => Self::FSharp,
            7 => Self::G,
            8 => Self::GSharp,
            9 => Self::A,
            10 => Self::ASharp,
            11 => Self::B,
            _ => Self::C, // 0 and unreachable cases
        }
    }
}

/// A musical key defined by its root pitch class and mode (major/minor).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MusicalKey {
    /// Root pitch class of the key.
    pub root: PitchClass,
    /// `true` for a major key, `false` for a natural minor key.
    pub is_major: bool,
}

impl MusicalKey {
    /// Create a new `MusicalKey`.
    #[must_use]
    pub fn new(root: PitchClass, is_major: bool) -> Self {
        Self { root, is_major }
    }

    /// Returns the name of the key, e.g. "C major" or "A minor".
    #[must_use]
    pub fn name(&self) -> String {
        format!(
            "{} {}",
            self.root.name(),
            if self.is_major { "major" } else { "minor" }
        )
    }

    /// Returns the relative key (major ↔ minor at a distance of 3 semitones).
    ///
    /// The relative minor of C major is A minor (down 3 semitones / up 9).
    /// The relative major of A minor is C major (up 3 semitones).
    #[must_use]
    pub fn relative_key(&self) -> Self {
        let semitones = self.root.semitones();
        let relative_root = if self.is_major {
            // Relative minor is 9 semitones up (or 3 down)
            PitchClass::from_semitones((semitones + 9) % 12)
        } else {
            // Relative major is 3 semitones up
            PitchClass::from_semitones((semitones + 3) % 12)
        };
        Self {
            root: relative_root,
            is_major: !self.is_major,
        }
    }

    /// Returns the number of semitones between the roots of two keys (0 – 11).
    #[must_use]
    pub fn distance_to(&self, other: &MusicalKey) -> u8 {
        let a = self.root.semitones();
        let b = other.root.semitones();
        let diff = if b >= a { b - a } else { 12 - (a - b) };
        diff.min(12 - diff)
    }
}

/// Type of chord.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChordType {
    /// Major triad (1, 3, 5)
    Major,
    /// Minor triad (1, b3, 5)
    Minor,
    /// Diminished triad (1, b3, b5)
    Diminished,
    /// Augmented triad (1, 3, #5)
    Augmented,
    /// Dominant seventh (1, 3, 5, b7)
    Dominant7,
    /// Major seventh (1, 3, 5, 7)
    Major7,
    /// Minor seventh (1, b3, 5, b7)
    Minor7,
}

impl ChordType {
    /// Returns the semitone intervals above the root that make up this chord.
    #[must_use]
    pub fn intervals(&self) -> Vec<u8> {
        match self {
            Self::Major => vec![0, 4, 7],
            Self::Minor => vec![0, 3, 7],
            Self::Diminished => vec![0, 3, 6],
            Self::Augmented => vec![0, 4, 8],
            Self::Dominant7 => vec![0, 4, 7, 10],
            Self::Major7 => vec![0, 4, 7, 11],
            Self::Minor7 => vec![0, 3, 7, 10],
        }
    }
}

/// A chord defined by its root and chord type.
#[derive(Debug, Clone)]
pub struct Chord {
    /// Root note of the chord.
    pub root: PitchClass,
    /// Type of the chord.
    pub chord_type: ChordType,
}

impl Chord {
    /// Create a new `Chord`.
    #[must_use]
    pub fn new(root: PitchClass, chord_type: ChordType) -> Self {
        Self { root, chord_type }
    }

    /// Returns the human-readable name of the chord, e.g. "Cmaj" or "Am7".
    #[must_use]
    pub fn name(&self) -> String {
        let suffix = match self.chord_type {
            ChordType::Major => "maj",
            ChordType::Minor => "min",
            ChordType::Diminished => "dim",
            ChordType::Augmented => "aug",
            ChordType::Dominant7 => "7",
            ChordType::Major7 => "maj7",
            ChordType::Minor7 => "min7",
        };
        format!("{}{}", self.root.name(), suffix)
    }

    /// Returns all pitch classes that make up this chord.
    #[must_use]
    pub fn notes(&self) -> Vec<PitchClass> {
        let root_semi = self.root.semitones();
        self.chord_type
            .intervals()
            .iter()
            .map(|&interval| PitchClass::from_semitones((root_semi + interval) % 12))
            .collect()
    }

    /// Returns `true` if all chord tones belong to the diatonic scale of `key`.
    #[must_use]
    pub fn is_diatonic(&self, key: &MusicalKey) -> bool {
        let scale = diatonic_scale(key);
        self.notes()
            .iter()
            .all(|note| scale.contains(&note.semitones()))
    }
}

/// Returns the set of semitone values present in the diatonic scale of `key`.
fn diatonic_scale(key: &MusicalKey) -> Vec<u8> {
    let root = key.root.semitones();
    // Major: W W H W W W H  → intervals 0,2,4,5,7,9,11
    // Natural minor: W H W W H W W → intervals 0,2,3,5,7,8,10
    let intervals: &[u8] = if key.is_major {
        &[0, 2, 4, 5, 7, 9, 11]
    } else {
        &[0, 2, 3, 5, 7, 8, 10]
    };
    intervals.iter().map(|&i| (root + i) % 12).collect()
}

// ── Krumhansl-Schmuckler key profiles ──────────────────────────────────────

/// Krumhansl-Schmuckler major key profile (C-major oriented).
const KS_MAJOR: [f32; 12] = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];

/// Krumhansl-Schmuckler minor key profile (C-minor oriented).
const KS_MINOR: [f32; 12] = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];

/// Correlate a chroma vector with a pitch-class profile rotated by `root` semitones.
fn correlate(chroma: &[f32; 12], profile: &[f32; 12], root: u8) -> f32 {
    let root = root as usize;
    // Pearson correlation numerator only (we compare relative scores)
    let mean_c: f32 = chroma.iter().sum::<f32>() / 12.0;
    let mean_p: f32 = profile.iter().sum::<f32>() / 12.0;

    let mut num = 0.0_f32;
    let mut denom_c = 0.0_f32;
    let mut denom_p = 0.0_f32;

    for i in 0..12 {
        let c = chroma[i] - mean_c;
        let p = profile[(i + 12 - root) % 12] - mean_p;
        num += c * p;
        denom_c += c * c;
        denom_p += p * p;
    }

    let denom = (denom_c * denom_p).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        num / denom
    }
}

/// Detects the most likely musical key from a 12-element chroma vector.
pub struct KeyDetector;

impl KeyDetector {
    /// Detect the musical key using Krumhansl-Schmuckler correlation.
    ///
    /// Correlates the chroma against all 24 major and minor profiles and
    /// returns the key with the highest correlation.
    #[must_use]
    pub fn detect_from_chroma(chroma: &[f32; 12]) -> MusicalKey {
        let mut best_key = MusicalKey::new(PitchClass::C, true);
        let mut best_score = f32::NEG_INFINITY;

        for root in 0u8..12 {
            let pc = PitchClass::from_semitones(root);

            let major_score = correlate(chroma, &KS_MAJOR, root);
            if major_score > best_score {
                best_score = major_score;
                best_key = MusicalKey::new(pc, true);
            }

            let minor_score = correlate(chroma, &KS_MINOR, root);
            if minor_score > best_score {
                best_score = minor_score;
                best_key = MusicalKey::new(pc, false);
            }
        }

        best_key
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pitch_class_semitones() {
        assert_eq!(PitchClass::C.semitones(), 0);
        assert_eq!(PitchClass::FSharp.semitones(), 6);
        assert_eq!(PitchClass::B.semitones(), 11);
    }

    #[test]
    fn test_pitch_class_name() {
        assert_eq!(PitchClass::C.name(), "C");
        assert_eq!(PitchClass::CSharp.name(), "C#");
        assert_eq!(PitchClass::ASharp.name(), "A#");
    }

    #[test]
    fn test_pitch_class_round_trip() {
        for i in 0u8..12 {
            let pc = PitchClass::from_semitones(i);
            assert_eq!(pc.semitones(), i);
        }
    }

    #[test]
    fn test_musical_key_name_major() {
        let key = MusicalKey::new(PitchClass::C, true);
        assert_eq!(key.name(), "C major");
    }

    #[test]
    fn test_musical_key_name_minor() {
        let key = MusicalKey::new(PitchClass::A, false);
        assert_eq!(key.name(), "A minor");
    }

    #[test]
    fn test_musical_key_relative_major_to_minor() {
        let c_major = MusicalKey::new(PitchClass::C, true);
        let rel = c_major.relative_key();
        assert_eq!(rel.root, PitchClass::A);
        assert!(!rel.is_major);
    }

    #[test]
    fn test_musical_key_relative_minor_to_major() {
        let a_minor = MusicalKey::new(PitchClass::A, false);
        let rel = a_minor.relative_key();
        assert_eq!(rel.root, PitchClass::C);
        assert!(rel.is_major);
    }

    #[test]
    fn test_musical_key_distance() {
        let c = MusicalKey::new(PitchClass::C, true);
        let g = MusicalKey::new(PitchClass::G, true);
        assert_eq!(c.distance_to(&g), 5); // min(7, 5) = 5
    }

    #[test]
    fn test_musical_key_distance_self() {
        let key = MusicalKey::new(PitchClass::D, true);
        assert_eq!(key.distance_to(&key), 0);
    }

    #[test]
    fn test_chord_type_intervals_major() {
        assert_eq!(ChordType::Major.intervals(), vec![0, 4, 7]);
    }

    #[test]
    fn test_chord_type_intervals_minor7() {
        assert_eq!(ChordType::Minor7.intervals(), vec![0, 3, 7, 10]);
    }

    #[test]
    fn test_chord_name() {
        let c_major = Chord::new(PitchClass::C, ChordType::Major);
        assert_eq!(c_major.name(), "Cmaj");

        let a_min7 = Chord::new(PitchClass::A, ChordType::Minor7);
        assert_eq!(a_min7.name(), "Amin7");
    }

    #[test]
    fn test_chord_notes_c_major() {
        let chord = Chord::new(PitchClass::C, ChordType::Major);
        let notes = chord.notes();
        // C, E, G
        assert!(notes.contains(&PitchClass::C));
        assert!(notes.contains(&PitchClass::E));
        assert!(notes.contains(&PitchClass::G));
        assert_eq!(notes.len(), 3);
    }

    #[test]
    fn test_chord_is_diatonic_true() {
        let key = MusicalKey::new(PitchClass::C, true);
        let c_major = Chord::new(PitchClass::C, ChordType::Major);
        assert!(c_major.is_diatonic(&key));
    }

    #[test]
    fn test_chord_is_diatonic_false() {
        let key = MusicalKey::new(PitchClass::C, true);
        // F# major contains F# (6), A# (10), C# (1) — none in C major scale
        let f_sharp_major = Chord::new(PitchClass::FSharp, ChordType::Major);
        assert!(!f_sharp_major.is_diatonic(&key));
    }

    #[test]
    fn test_key_detector_c_major() {
        // Chroma dominated by C major scale tones
        let mut chroma = [0.0f32; 12];
        // C, E, G strongly present
        chroma[0] = 6.0; // C
        chroma[4] = 4.0; // E
        chroma[7] = 5.0; // G
        chroma[2] = 1.0; // D
        chroma[5] = 1.5; // F
        chroma[9] = 1.5; // A
        chroma[11] = 1.0; // B
        let key = KeyDetector::detect_from_chroma(&chroma);
        // The key detected should be major
        assert!(key.is_major);
    }
}
