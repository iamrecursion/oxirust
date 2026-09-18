//! Chord recognition and harmonic analysis for Music Information Retrieval.
//!
//! Provides chord quality enumerations with interval patterns, chord-to-chroma
//! template matching, a timed chord progression container, and a
//! `HarmonicAnalyzer` that identifies common harmonic cadences.

#![allow(dead_code)]

// ── ChordQuality ──────────────────────────────────────────────────────────────

/// Quality (type) of a chord.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChordQuality {
    /// Major triad (root, major 3rd, perfect 5th).
    Major,
    /// Minor triad (root, minor 3rd, perfect 5th).
    Minor,
    /// Diminished triad (root, minor 3rd, tritone).
    Diminished,
    /// Augmented triad (root, major 3rd, augmented 5th).
    Augmented,
    /// Dominant seventh (major triad + minor 7th).
    Dominant7,
    /// Major seventh (major triad + major 7th).
    Major7,
    /// Minor seventh (minor triad + minor 7th).
    Minor7,
}

impl ChordQuality {
    /// Semitone intervals from the root that define this chord quality.
    #[must_use]
    pub fn interval_pattern(self) -> Vec<u8> {
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

    /// Short suffix used in chord names (e.g. `"m"` for minor).
    #[must_use]
    pub fn suffix(self) -> &'static str {
        match self {
            Self::Major => "",
            Self::Minor => "m",
            Self::Diminished => "dim",
            Self::Augmented => "aug",
            Self::Dominant7 => "7",
            Self::Major7 => "maj7",
            Self::Minor7 => "m7",
        }
    }
}

// ── Chord ─────────────────────────────────────────────────────────────────────

/// A chord expressed as root pitch-class index (0 = C … 11 = B) and quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Chord {
    /// Root pitch class (0 = C, 1 = C#, …, 11 = B).
    pub root: u8,
    /// Chord quality.
    pub quality: ChordQuality,
}

impl Chord {
    /// Default pitch-class name table.
    const PC_NAMES: [&'static str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];

    /// Create a new chord.
    #[must_use]
    pub fn new(root: u8, quality: ChordQuality) -> Self {
        Self {
            root: root % 12,
            quality,
        }
    }

    /// Format the chord name using the built-in pitch-class names.
    ///
    /// Example: `Chord::new(0, ChordQuality::Major).display(&[])` → `"C"`.
    ///
    /// If `root_names` is non-empty and long enough its entry is preferred;
    /// otherwise the built-in table is used.
    #[must_use]
    pub fn display(&self, root_names: &[&str]) -> String {
        let root_name = if root_names.len() > self.root as usize {
            root_names[self.root as usize]
        } else {
            Self::PC_NAMES[self.root as usize]
        };
        format!("{}{}", root_name, self.quality.suffix())
    }

    /// Return all pitch classes (mod 12) that belong to this chord.
    #[must_use]
    pub fn notes(&self) -> Vec<u8> {
        self.quality
            .interval_pattern()
            .iter()
            .map(|&i| (self.root + i) % 12)
            .collect()
    }
}

// ── chroma_to_chord ───────────────────────────────────────────────────────────

/// All chord qualities considered during template matching.
const ALL_QUALITIES: &[ChordQuality] = &[
    ChordQuality::Major,
    ChordQuality::Minor,
    ChordQuality::Diminished,
    ChordQuality::Augmented,
    ChordQuality::Dominant7,
    ChordQuality::Major7,
    ChordQuality::Minor7,
];

/// Build a binary chroma template for a chord.
fn chord_template(root: u8, quality: ChordQuality) -> [f32; 12] {
    let mut t = [0.0_f32; 12];
    for &interval in &quality.interval_pattern() {
        t[((root + interval) % 12) as usize] = 1.0;
    }
    t
}

/// Dot product of two 12-element chroma arrays.
fn dot12(a: &[f32; 12], b: &[f32; 12]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Match a 12-element chroma vector against all chord templates.
///
/// Returns the `Chord` (root + quality) whose template has the highest dot
/// product with the input chroma.  Returns `Chord::new(0, ChordQuality::Major)`
/// for an all-zero chroma.
#[must_use]
pub fn chroma_to_chord(chroma: &[f32; 12]) -> Chord {
    let mut best_chord = Chord::new(0, ChordQuality::Major);
    let mut best_score = f32::NEG_INFINITY;

    for &quality in ALL_QUALITIES {
        for root in 0_u8..12 {
            let template = chord_template(root, quality);
            let score = dot12(chroma, &template);
            if score > best_score {
                best_score = score;
                best_chord = Chord::new(root, quality);
            }
        }
    }

    best_chord
}

// ── ChordProgression ──────────────────────────────────────────────────────────

/// A sequence of timed chords.
#[derive(Debug, Clone, Default)]
pub struct ChordProgression {
    /// Chords in order.
    pub chords: Vec<Chord>,
    /// Start timestamps in milliseconds, one per chord.
    pub timestamps_ms: Vec<u64>,
}

impl ChordProgression {
    /// Create an empty progression.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a chord at the given timestamp.
    pub fn add(&mut self, chord: Chord, ts_ms: u64) {
        self.chords.push(chord);
        self.timestamps_ms.push(ts_ms);
    }

    /// Return a reference to the chord active at `ts` milliseconds.
    ///
    /// Returns the last chord whose timestamp is ≤ `ts`, or `None` if
    /// the progression is empty or `ts` is before the first chord.
    #[must_use]
    pub fn at_time_ms(&self, ts: u64) -> Option<&Chord> {
        if self.chords.is_empty() {
            return None;
        }
        let mut idx = None;
        for (i, &t) in self.timestamps_ms.iter().enumerate() {
            if t <= ts {
                idx = Some(i);
            } else {
                break;
            }
        }
        idx.map(|i| &self.chords[i])
    }

    /// Number of chords in the progression.
    #[must_use]
    pub fn chord_count(&self) -> usize {
        self.chords.len()
    }
}

// ── HarmonicAnalyzer ─────────────────────────────────────────────────────────

/// Analyses a chord progression and identifies common harmonic cadences.
pub struct HarmonicAnalyzer;

impl HarmonicAnalyzer {
    /// Analyse consecutive chord pairs in `progression` for cadence patterns.
    ///
    /// Recognised patterns:
    /// - V → I (dominant to tonic, same mode)  → `"perfect cadence"`
    /// - IV → I (subdominant to tonic, same mode) → `"plagal cadence"`
    /// - I → V (tonic to dominant, same mode)  → `"half cadence"`
    /// - V → VI (dominant to submediant, deceptive) → `"deceptive cadence"`
    ///
    /// Returns a `Vec<String>` with one entry per consecutive pair; pairs that
    /// do not match a recognised cadence produce an empty string.
    #[must_use]
    pub fn analyze_progression(progression: &ChordProgression) -> Vec<String> {
        let chords = &progression.chords;
        if chords.len() < 2 {
            return Vec::new();
        }
        let mut results = Vec::with_capacity(chords.len() - 1);
        for pair in chords.windows(2) {
            let prev = &pair[0];
            let curr = &pair[1];
            let label = Self::classify_cadence(prev, curr);
            results.push(label);
        }
        results
    }

    /// Classify a pair of chords as a cadence (or empty string if unknown).
    fn classify_cadence(prev: &Chord, curr: &Chord) -> String {
        #[allow(clippy::cast_possible_wrap)]
        let interval = (curr.root as i8 - prev.root as i8).rem_euclid(12) as u8;
        // Same quality mode (both major or both minor) for most cadences
        let same_quality = prev.quality == curr.quality;

        if interval == 5 && same_quality {
            // V → I: the dominant is 7 semitones above the tonic, so
            // moving DOWN a P5 (== UP a P4 = 5 semitones) is V→I
            return "perfect cadence".to_string();
        }
        if interval == 7 && same_quality {
            // IV → I: subdominant is 5 semitones below the tonic (interval=7 means up a P5)
            return "plagal cadence".to_string();
        }
        if interval == 7
            && curr.quality == ChordQuality::Major
            && prev.quality == ChordQuality::Major
        {
            return "half cadence".to_string();
        }
        if interval == 2 {
            // V → vi: deceptive (e.g. G → Am, interval = 2 semitones up)
            return "deceptive cadence".to_string();
        }

        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ChordQuality::interval_pattern ────────────────────────────────────────

    #[test]
    fn test_major_interval_pattern() {
        assert_eq!(ChordQuality::Major.interval_pattern(), vec![0, 4, 7]);
    }

    #[test]
    fn test_minor_interval_pattern() {
        assert_eq!(ChordQuality::Minor.interval_pattern(), vec![0, 3, 7]);
    }

    #[test]
    fn test_diminished_interval_pattern() {
        assert_eq!(ChordQuality::Diminished.interval_pattern(), vec![0, 3, 6]);
    }

    #[test]
    fn test_augmented_interval_pattern() {
        assert_eq!(ChordQuality::Augmented.interval_pattern(), vec![0, 4, 8]);
    }

    #[test]
    fn test_dominant7_interval_pattern() {
        assert_eq!(
            ChordQuality::Dominant7.interval_pattern(),
            vec![0, 4, 7, 10]
        );
    }

    #[test]
    fn test_major7_interval_pattern() {
        assert_eq!(ChordQuality::Major7.interval_pattern(), vec![0, 4, 7, 11]);
    }

    #[test]
    fn test_minor7_interval_pattern() {
        assert_eq!(ChordQuality::Minor7.interval_pattern(), vec![0, 3, 7, 10]);
    }

    // ── Chord ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_chord_display_c_major() {
        let c = Chord::new(0, ChordQuality::Major);
        assert_eq!(c.display(&[]), "C");
    }

    #[test]
    fn test_chord_display_a_minor() {
        let c = Chord::new(9, ChordQuality::Minor);
        assert_eq!(c.display(&[]), "Am");
    }

    #[test]
    fn test_chord_display_custom_names() {
        let c = Chord::new(0, ChordQuality::Major7);
        let names = ["Do", "Re", "Mi"];
        assert_eq!(c.display(&names), "Domaj7");
    }

    #[test]
    fn test_chord_notes_c_major() {
        let c = Chord::new(0, ChordQuality::Major);
        let notes = c.notes();
        assert_eq!(notes, vec![0, 4, 7]); // C E G
    }

    #[test]
    fn test_chord_notes_wraps_mod12() {
        // B major: B=11, D#=3, F#=6
        let c = Chord::new(11, ChordQuality::Major);
        let notes = c.notes();
        assert!(notes.contains(&11)); // B
        assert!(notes.contains(&3)); // D#
        assert!(notes.contains(&6)); // F#
    }

    // ── chroma_to_chord ───────────────────────────────────────────────────────

    #[test]
    fn test_chroma_to_chord_c_major() {
        let mut chroma = [0.0_f32; 12];
        chroma[0] = 1.0; // C
        chroma[4] = 1.0; // E
        chroma[7] = 1.0; // G
        let chord = chroma_to_chord(&chroma);
        assert_eq!(chord.root, 0);
        assert_eq!(chord.quality, ChordQuality::Major);
    }

    #[test]
    fn test_chroma_to_chord_a_minor() {
        let mut chroma = [0.0_f32; 12];
        chroma[9] = 1.0; // A
        chroma[0] = 1.0; // C
        chroma[4] = 1.0; // E
        let chord = chroma_to_chord(&chroma);
        assert_eq!(chord.root, 9);
        assert_eq!(chord.quality, ChordQuality::Minor);
    }

    // ── ChordProgression ──────────────────────────────────────────────────────

    #[test]
    fn test_chord_progression_empty() {
        let prog = ChordProgression::new();
        assert_eq!(prog.chord_count(), 0);
        assert!(prog.at_time_ms(0).is_none());
    }

    #[test]
    fn test_chord_progression_add_and_count() {
        let mut prog = ChordProgression::new();
        prog.add(Chord::new(0, ChordQuality::Major), 0);
        prog.add(Chord::new(7, ChordQuality::Major), 1000);
        assert_eq!(prog.chord_count(), 2);
    }

    #[test]
    fn test_chord_progression_at_time_ms_first_chord() {
        let mut prog = ChordProgression::new();
        let c = Chord::new(0, ChordQuality::Major);
        prog.add(c, 0);
        assert_eq!(prog.at_time_ms(0), Some(&c));
    }

    #[test]
    fn test_chord_progression_at_time_ms_before_start_returns_none() {
        let mut prog = ChordProgression::new();
        prog.add(Chord::new(0, ChordQuality::Major), 500);
        assert!(prog.at_time_ms(100).is_none());
    }

    #[test]
    fn test_chord_progression_at_time_ms_selects_correct() {
        let mut prog = ChordProgression::new();
        let c = Chord::new(0, ChordQuality::Major);
        let g = Chord::new(7, ChordQuality::Major);
        prog.add(c, 0);
        prog.add(g, 2000);
        assert_eq!(prog.at_time_ms(1500), Some(&c));
        assert_eq!(prog.at_time_ms(2000), Some(&g));
        assert_eq!(prog.at_time_ms(3000), Some(&g));
    }

    // ── HarmonicAnalyzer ──────────────────────────────────────────────────────

    #[test]
    fn test_harmonic_analyzer_empty_progression() {
        let prog = ChordProgression::new();
        let labels = HarmonicAnalyzer::analyze_progression(&prog);
        assert!(labels.is_empty());
    }

    #[test]
    fn test_harmonic_analyzer_single_chord_no_pairs() {
        let mut prog = ChordProgression::new();
        prog.add(Chord::new(0, ChordQuality::Major), 0);
        let labels = HarmonicAnalyzer::analyze_progression(&prog);
        assert!(labels.is_empty());
    }

    #[test]
    fn test_harmonic_analyzer_pair_count() {
        let mut prog = ChordProgression::new();
        for i in 0_u8..4 {
            prog.add(Chord::new(i * 2, ChordQuality::Major), u64::from(i) * 1000);
        }
        let labels = HarmonicAnalyzer::analyze_progression(&prog);
        assert_eq!(labels.len(), 3);
    }

    #[test]
    fn test_harmonic_analyzer_deceptive_cadence() {
        // V (G=7) → VI (A=9): interval 9 → deceptive cadence
        let mut prog = ChordProgression::new();
        prog.add(Chord::new(7, ChordQuality::Major), 0);
        prog.add(Chord::new(9, ChordQuality::Minor), 1000);
        let labels = HarmonicAnalyzer::analyze_progression(&prog);
        assert_eq!(labels[0], "deceptive cadence");
    }
}
