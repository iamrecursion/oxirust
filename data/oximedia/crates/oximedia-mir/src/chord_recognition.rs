//! Chord recognition using chromagram template matching.
//!
//! Extracts a chromagram from audio frames, then matches against
//! major/minor/seventh chord templates to identify chord sequences.

#![allow(dead_code)]

use std::fmt;

/// Number of pitch classes.
const PC: usize = 12;

/// Chord quality enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChordQuality {
    /// Major triad (root, major third, perfect fifth).
    Major,
    /// Minor triad (root, minor third, perfect fifth).
    Minor,
    /// Dominant seventh (major triad + minor seventh).
    Dom7,
    /// Major seventh (major triad + major seventh).
    Maj7,
    /// Minor seventh (minor triad + minor seventh).
    Min7,
    /// Diminished triad (root, minor third, tritone).
    Dim,
    /// Augmented triad (root, major third, augmented fifth).
    Aug,
    /// No chord / silence.
    NoChord,
}

impl ChordQuality {
    /// Human-readable suffix for display.
    #[must_use]
    pub fn suffix(self) -> &'static str {
        match self {
            Self::Major => "",
            Self::Minor => "m",
            Self::Dom7 => "7",
            Self::Maj7 => "maj7",
            Self::Min7 => "m7",
            Self::Dim => "dim",
            Self::Aug => "aug",
            Self::NoChord => "N",
        }
    }

    /// Returns the semitone intervals that define this chord (relative to root).
    #[must_use]
    pub fn intervals(self) -> &'static [usize] {
        match self {
            Self::Major => &[0, 4, 7],
            Self::Minor => &[0, 3, 7],
            Self::Dom7 => &[0, 4, 7, 10],
            Self::Maj7 => &[0, 4, 7, 11],
            Self::Min7 => &[0, 3, 7, 10],
            Self::Dim => &[0, 3, 6],
            Self::Aug => &[0, 4, 8],
            Self::NoChord => &[],
        }
    }
}

/// A specific chord with root and quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Chord {
    /// Root pitch class (0 = C … 11 = B).
    pub root: u8,
    /// Chord quality.
    pub quality: ChordQuality,
}

impl Chord {
    /// Creates a new chord.
    #[must_use]
    pub fn new(root: u8, quality: ChordQuality) -> Self {
        Self { root, quality }
    }

    /// "No chord" sentinel.
    #[must_use]
    pub fn no_chord() -> Self {
        Self {
            root: 0,
            quality: ChordQuality::NoChord,
        }
    }

    /// Pitch class names.
    const PC_NAMES: [&'static str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
}

impl fmt::Display for Chord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.quality == ChordQuality::NoChord {
            return write!(f, "N");
        }
        write!(
            f,
            "{}{}",
            Self::PC_NAMES[self.root as usize],
            self.quality.suffix()
        )
    }
}

/// Build a binary chord template for a given root and quality.
///
/// Returns a 12-element binary vector with 1.0 at active pitch classes.
#[must_use]
pub fn chord_template(root: u8, quality: ChordQuality) -> [f64; PC] {
    let mut t = [0.0_f64; PC];
    for &interval in quality.intervals() {
        t[(root as usize + interval) % PC] = 1.0;
    }
    t
}

/// Dot product of two 12-element vectors.
fn dot(a: &[f64; PC], b: &[f64; PC]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm of a 12-element vector.
fn norm(a: &[f64; PC]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Cosine similarity between two 12-element vectors.
#[must_use]
pub fn cosine_similarity(a: &[f64; PC], b: &[f64; PC]) -> f64 {
    let na = norm(a);
    let nb = norm(b);
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    dot(a, b) / (na * nb)
}

/// Match a chroma frame against all chord templates.
///
/// Returns the best-matching `Chord` and its similarity score.
///
/// The set of evaluated qualities can be controlled via `qualities`.
#[must_use]
pub fn match_chord(chroma: &[f64; PC], qualities: &[ChordQuality]) -> (Chord, f64) {
    let mut best_chord = Chord::no_chord();
    let mut best_score = -1.0_f64;

    for &quality in qualities {
        if quality == ChordQuality::NoChord {
            continue;
        }
        for root in 0_u8..12 {
            let template = chord_template(root, quality);
            let score = cosine_similarity(chroma, &template);
            if score > best_score {
                best_score = score;
                best_chord = Chord::new(root, quality);
            }
        }
    }

    // Check for silence (very low energy)
    let energy: f64 = chroma.iter().sum();
    if energy < 1e-6 {
        return (Chord::no_chord(), 0.0);
    }

    (best_chord, best_score)
}

/// A timed chord event in a chord sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct ChordEvent {
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
    /// The chord at this time.
    pub chord: Chord,
    /// Confidence/similarity score.
    pub score: f64,
}

impl ChordEvent {
    /// Duration of this chord event in seconds.
    #[must_use]
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }
}

/// Simplify a chord sequence by merging consecutive identical chords.
#[must_use]
pub fn merge_chord_sequence(events: &[ChordEvent]) -> Vec<ChordEvent> {
    if events.is_empty() {
        return vec![];
    }
    let mut merged: Vec<ChordEvent> = Vec::new();
    let mut current = events[0].clone();

    for event in events.iter().skip(1) {
        if event.chord == current.chord {
            current.end = event.end;
            // Average score
            current.score = (current.score + event.score) / 2.0;
        } else {
            merged.push(current.clone());
            current = event.clone();
        }
    }
    merged.push(current);
    merged
}

/// Common chord qualities used for recognition.
pub const STANDARD_QUALITIES: &[ChordQuality] = &[
    ChordQuality::Major,
    ChordQuality::Minor,
    ChordQuality::Dom7,
    ChordQuality::Maj7,
    ChordQuality::Min7,
];

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_chord_quality_suffix() {
        assert_eq!(ChordQuality::Major.suffix(), "");
        assert_eq!(ChordQuality::Minor.suffix(), "m");
        assert_eq!(ChordQuality::Dom7.suffix(), "7");
    }

    #[test]
    fn test_chord_display_c_major() {
        let c = Chord::new(0, ChordQuality::Major);
        assert_eq!(c.to_string(), "C");
    }

    #[test]
    fn test_chord_display_a_minor() {
        let c = Chord::new(9, ChordQuality::Minor);
        assert_eq!(c.to_string(), "Am");
    }

    #[test]
    fn test_chord_display_no_chord() {
        let c = Chord::no_chord();
        assert_eq!(c.to_string(), "N");
    }

    #[test]
    fn test_chord_template_c_major() {
        let t = chord_template(0, ChordQuality::Major);
        // C=0, E=4, G=7
        assert!(approx_eq(t[0], 1.0, 1e-10));
        assert!(approx_eq(t[4], 1.0, 1e-10));
        assert!(approx_eq(t[7], 1.0, 1e-10));
        assert!(approx_eq(t[1], 0.0, 1e-10));
    }

    #[test]
    fn test_chord_template_a_minor() {
        let t = chord_template(9, ChordQuality::Minor);
        // A=9, C=0, E=4
        assert!(approx_eq(t[9], 1.0, 1e-10));
        assert!(approx_eq(t[0], 1.0, 1e-10));
        assert!(approx_eq(t[4], 1.0, 1e-10));
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = chord_template(0, ChordQuality::Major);
        let s = cosine_similarity(&a, &a);
        assert!(approx_eq(s, 1.0, 1e-10));
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let mut a = [0.0_f64; 12];
        let mut b = [0.0_f64; 12];
        a[0] = 1.0;
        b[1] = 1.0;
        let s = cosine_similarity(&a, &b);
        assert!(approx_eq(s, 0.0, 1e-10));
    }

    #[test]
    fn test_match_chord_c_major_chroma() {
        // Chroma with energy at C, E, G
        let mut chroma = [0.0_f64; 12];
        chroma[0] = 1.0; // C
        chroma[4] = 1.0; // E
        chroma[7] = 1.0; // G
        let (chord, score) = match_chord(&chroma, STANDARD_QUALITIES);
        assert_eq!(chord.root, 0);
        assert_eq!(chord.quality, ChordQuality::Major);
        assert!(score > 0.9);
    }

    #[test]
    fn test_match_chord_silence_is_no_chord() {
        let chroma = [0.0_f64; 12];
        let (chord, _) = match_chord(&chroma, STANDARD_QUALITIES);
        assert_eq!(chord.quality, ChordQuality::NoChord);
    }

    #[test]
    fn test_merge_chord_sequence_merges_adjacent() {
        let c_major = Chord::new(0, ChordQuality::Major);
        let events = vec![
            ChordEvent {
                start: 0.0,
                end: 1.0,
                chord: c_major,
                score: 0.9,
            },
            ChordEvent {
                start: 1.0,
                end: 2.0,
                chord: c_major,
                score: 0.8,
            },
            ChordEvent {
                start: 2.0,
                end: 3.0,
                chord: Chord::new(7, ChordQuality::Major),
                score: 0.7,
            },
        ];
        let merged = merge_chord_sequence(&events);
        assert_eq!(merged.len(), 2);
        assert!(approx_eq(merged[0].end, 2.0, 1e-10));
    }

    #[test]
    fn test_merge_empty_sequence() {
        let merged = merge_chord_sequence(&[]);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_chord_event_duration() {
        let ev = ChordEvent {
            start: 1.5,
            end: 3.0,
            chord: Chord::new(0, ChordQuality::Major),
            score: 0.8,
        };
        assert!(approx_eq(ev.duration(), 1.5, 1e-10));
    }

    #[test]
    fn test_dim_chord_template() {
        let t = chord_template(0, ChordQuality::Dim);
        // C=0, Eb=3, Gb=6
        assert!(approx_eq(t[0], 1.0, 1e-10));
        assert!(approx_eq(t[3], 1.0, 1e-10));
        assert!(approx_eq(t[6], 1.0, 1e-10));
    }

    #[test]
    fn test_aug_chord_intervals() {
        // Augmented: 0, 4, 8
        let intervals = ChordQuality::Aug.intervals();
        assert_eq!(intervals, &[0, 4, 8]);
    }
}
