//! Common types for MIR analysis.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Tempo analysis result.
    pub tempo: Option<TempoResult>,

    /// Beat tracking result.
    pub beat: Option<BeatResult>,

    /// Key detection result.
    pub key: Option<KeyResult>,

    /// Chord recognition result.
    pub chord: Option<ChordResult>,

    /// Melody extraction result.
    pub melody: Option<MelodyResult>,

    /// Structure analysis result.
    pub structure: Option<StructureResult>,

    /// Genre classification result.
    pub genre: Option<GenreResult>,

    /// Mood detection result.
    pub mood: Option<MoodResult>,

    /// Spectral features result.
    pub spectral: Option<SpectralResult>,

    /// Rhythm features result.
    pub rhythm: Option<RhythmResult>,

    /// Harmonic analysis result.
    pub harmonic: Option<HarmonicResult>,

    /// Sample rate of analyzed audio.
    pub sample_rate: f32,

    /// Duration of analyzed audio in seconds.
    pub duration: f32,
}

/// Tempo detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TempoResult {
    /// Detected BPM (beats per minute).
    pub bpm: f32,

    /// Confidence score (0.0 to 1.0).
    pub confidence: f32,

    /// Tempo stability (0.0 to 1.0, higher = more stable).
    pub stability: f32,

    /// Alternative tempo estimates.
    pub alternatives: Vec<(f32, f32)>, // (BPM, confidence)
}

/// Beat tracking result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeatResult {
    /// Beat times in seconds.
    pub beat_times: Vec<f32>,

    /// Downbeat times in seconds.
    pub downbeat_times: Vec<f32>,

    /// Beat confidence scores.
    pub beat_confidence: Vec<f32>,

    /// Estimated time signature (numerator, denominator).
    pub time_signature: Option<(u8, u8)>,
}

/// Key detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyResult {
    /// Detected key (e.g., "C major", "A minor").
    pub key: String,

    /// Root note (0-11, C=0).
    pub root: u8,

    /// Mode (true = major, false = minor).
    pub is_major: bool,

    /// Confidence score (0.0 to 1.0).
    pub confidence: f32,

    /// Key profile correlations.
    pub profile_correlations: Vec<f32>,
}

/// Chord recognition result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChordResult {
    /// Chord labels with timestamps.
    pub chords: Vec<ChordLabel>,

    /// Chord progression patterns.
    pub progressions: Vec<String>,

    /// Overall harmonic complexity (0.0 to 1.0).
    pub complexity: f32,
}

/// Individual chord label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChordLabel {
    /// Start time in seconds.
    pub start: f32,

    /// End time in seconds.
    pub end: f32,

    /// Chord name (e.g., "C", "Am", "G7").
    pub label: String,

    /// Confidence score (0.0 to 1.0).
    pub confidence: f32,
}

/// Melody extraction result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MelodyResult {
    /// Pitch contour over time (Hz, 0 = no pitch).
    pub pitch_contour: Vec<f32>,

    /// Time points for pitch contour.
    pub time_points: Vec<f32>,

    /// Pitch confidence scores.
    pub confidence: Vec<f32>,

    /// Melodic range (min, max in Hz).
    pub range: (f32, f32),

    /// Melodic contour complexity.
    pub complexity: f32,
}

/// Structure analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureResult {
    /// Structural segments.
    pub segments: Vec<Segment>,

    /// Self-similarity matrix (flattened).
    pub similarity_matrix: Vec<f32>,

    /// Matrix dimensions.
    pub matrix_size: usize,

    /// Overall structural complexity.
    pub complexity: f32,
}

/// Musical segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    /// Start time in seconds.
    pub start: f32,

    /// End time in seconds.
    pub end: f32,

    /// Segment label (e.g., "intro", "verse", "chorus").
    pub label: String,

    /// Confidence score (0.0 to 1.0).
    pub confidence: f32,
}

/// Genre classification result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenreResult {
    /// Genre predictions with confidence scores.
    pub genres: HashMap<String, f32>,

    /// Top genre.
    pub top_genre_name: String,

    /// Top genre confidence.
    pub top_genre_confidence: f32,
}

impl GenreResult {
    /// Get top genre and confidence.
    #[must_use]
    pub fn top_genre(&self) -> (&str, f32) {
        (&self.top_genre_name, self.top_genre_confidence)
    }
}

/// Mood detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodResult {
    /// Valence (negative to positive, -1.0 to 1.0).
    pub valence: f32,

    /// Arousal (calm to energetic, 0.0 to 1.0).
    pub arousal: f32,

    /// Mood labels with confidence.
    pub moods: HashMap<String, f32>,

    /// Emotional intensity (0.0 to 1.0).
    pub intensity: f32,
}

/// Spectral features result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralResult {
    /// Spectral centroid over time (Hz).
    pub centroid: Vec<f32>,

    /// Spectral rolloff over time (Hz).
    pub rolloff: Vec<f32>,

    /// Spectral flux over time.
    pub flux: Vec<f32>,

    /// Spectral contrast over time.
    pub contrast: Vec<Vec<f32>>,

    /// Mean spectral centroid.
    pub mean_centroid: f32,

    /// Mean spectral rolloff.
    pub mean_rolloff: f32,

    /// Mean spectral flux.
    pub mean_flux: f32,
}

/// Rhythm features result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmResult {
    /// Onset strength envelope.
    pub onset_strength: Vec<f32>,

    /// Onset times in seconds.
    pub onset_times: Vec<f32>,

    /// Rhythmic patterns.
    pub patterns: Vec<RhythmPattern>,

    /// Rhythmic complexity (0.0 to 1.0).
    pub complexity: f32,

    /// Syncopation measure (0.0 to 1.0).
    pub syncopation: f32,
}

/// Rhythmic pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmPattern {
    /// Pattern start time in seconds.
    pub start: f32,

    /// Pattern duration in seconds.
    pub duration: f32,

    /// Pattern description.
    pub description: String,

    /// Pattern strength (0.0 to 1.0).
    pub strength: f32,
}

/// Harmonic analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicResult {
    /// Harmonic component energy over time.
    pub harmonic_energy: Vec<f32>,

    /// Percussive component energy over time.
    pub percussive_energy: Vec<f32>,

    /// Harmonic-to-percussive ratio.
    pub hpr_ratio: f32,

    /// Pitch class profile (12 bins, C to B).
    pub pitch_class_profile: Vec<f32>,

    /// Chroma features over time.
    pub chroma: Vec<Vec<f32>>,
}

/// Loudness analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoudnessResult {
    /// Integrated loudness (LUFS approximation).
    pub integrated_loudness: f32,

    /// Loudness range (LRA).
    pub loudness_range: f32,

    /// Peak loudness.
    pub peak_loudness: f32,

    /// True peak value.
    pub true_peak: f32,
}

bitflags::bitflags! {
    /// Feature set flags for selective feature extraction.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct FeatureSet: u32 {
        /// Spectral features.
        const SPECTRAL = 0b0000_0001;
        /// Rhythm features.
        const RHYTHM = 0b0000_0010;
        /// Harmonic features.
        const HARMONIC = 0b0000_0100;
        /// Tempo and beat.
        const TEMPO = 0b0000_1000;
        /// Key detection.
        const KEY = 0b0001_0000;
        /// Chord recognition.
        const CHORD = 0b0010_0000;
        /// Melody extraction.
        const MELODY = 0b0100_0000;
        /// All features.
        const ALL = 0b0111_1111;
    }
}

impl Default for FeatureSet {
    fn default() -> Self {
        Self::ALL
    }
}
