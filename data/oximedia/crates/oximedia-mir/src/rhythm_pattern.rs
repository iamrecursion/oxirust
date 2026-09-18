#![allow(dead_code)]
//! Rhythm pattern analysis and drum-pattern recognition.
//!
//! Extracts onset envelopes, quantises them to a beat grid, and classifies
//! common rhythm patterns (four-on-the-floor, breakbeat, shuffle, etc.).

/// Number of subdivisions per beat used for quantisation.
const SUBDIVISIONS: usize = 16;

/// A quantised rhythm pattern as a binary grid.
///
/// Each element is `true` if an onset falls on that subdivision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RhythmGrid {
    /// Number of beats in the pattern (typically 4 for one bar of 4/4).
    pub beats: usize,
    /// Subdivisions per beat.
    pub subdivisions: usize,
    /// Binary grid: `grid[beat * subdivisions + sub]`.
    pub grid: Vec<bool>,
}

impl RhythmGrid {
    /// Create an empty grid for the given number of beats.
    #[must_use]
    pub fn empty(beats: usize, subdivisions: usize) -> Self {
        Self {
            beats,
            subdivisions,
            grid: vec![false; beats * subdivisions],
        }
    }

    /// Set an onset at a specific beat and subdivision.
    pub fn set(&mut self, beat: usize, sub: usize, value: bool) {
        let idx = beat * self.subdivisions + sub;
        if idx < self.grid.len() {
            self.grid[idx] = value;
        }
    }

    /// Check whether an onset exists at the given position.
    #[must_use]
    pub fn get(&self, beat: usize, sub: usize) -> bool {
        let idx = beat * self.subdivisions + sub;
        self.grid.get(idx).copied().unwrap_or(false)
    }

    /// Return the total number of onsets in the grid.
    #[must_use]
    pub fn onset_count(&self) -> usize {
        self.grid.iter().filter(|&&v| v).count()
    }

    /// Return the pattern density (fraction of slots that are active).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn density(&self) -> f32 {
        if self.grid.is_empty() {
            return 0.0;
        }
        self.onset_count() as f32 / self.grid.len() as f32
    }

    /// Compute Hamming distance to another grid of the same size.
    #[must_use]
    pub fn hamming_distance(&self, other: &Self) -> usize {
        self.grid
            .iter()
            .zip(other.grid.iter())
            .filter(|(a, b)| a != b)
            .count()
    }
}

/// Known rhythm-pattern templates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternKind {
    /// Four-on-the-floor kick pattern.
    FourOnTheFloor,
    /// Breakbeat pattern.
    Breakbeat,
    /// Shuffle / swing pattern.
    Shuffle,
    /// Half-time pattern.
    HalfTime,
    /// No recognised pattern.
    Unknown,
}

/// Build the canonical four-on-the-floor kick grid (4 beats, 16 subs).
#[must_use]
pub fn four_on_the_floor() -> RhythmGrid {
    let mut g = RhythmGrid::empty(4, SUBDIVISIONS);
    for beat in 0..4 {
        g.set(beat, 0, true);
    }
    g
}

/// Build a canonical breakbeat kick grid.
#[must_use]
pub fn breakbeat_pattern() -> RhythmGrid {
    let mut g = RhythmGrid::empty(4, SUBDIVISIONS);
    // Kick on 1 and the "and" of 2
    g.set(0, 0, true);
    g.set(1, 8, true);
    // Snare on 2 and 4
    g.set(1, 0, true);
    g.set(3, 0, true);
    g
}

/// Build a shuffle hi-hat pattern (swung 8ths).
#[must_use]
pub fn shuffle_pattern() -> RhythmGrid {
    let mut g = RhythmGrid::empty(4, SUBDIVISIONS);
    for beat in 0..4 {
        g.set(beat, 0, true); // downbeat
        g.set(beat, 10, true); // swung offbeat (~66%)
    }
    g
}

/// Build a half-time snare grid.
#[must_use]
pub fn half_time_pattern() -> RhythmGrid {
    let mut g = RhythmGrid::empty(4, SUBDIVISIONS);
    g.set(0, 0, true);
    g.set(2, 0, true); // snare on beat 3
    g
}

/// Classify a grid by comparing it to known templates.
#[must_use]
pub fn classify_pattern(grid: &RhythmGrid) -> PatternKind {
    if grid.grid.len() != 4 * SUBDIVISIONS {
        return PatternKind::Unknown;
    }
    let templates: [(PatternKind, RhythmGrid); 4] = [
        (PatternKind::FourOnTheFloor, four_on_the_floor()),
        (PatternKind::Breakbeat, breakbeat_pattern()),
        (PatternKind::Shuffle, shuffle_pattern()),
        (PatternKind::HalfTime, half_time_pattern()),
    ];
    let mut best = PatternKind::Unknown;
    let mut best_dist = usize::MAX;
    for (kind, tmpl) in &templates {
        let d = grid.hamming_distance(tmpl);
        if d < best_dist {
            best_dist = d;
            best = *kind;
        }
    }
    // Only accept if reasonably close.
    if best_dist <= grid.grid.len() / 4 {
        best
    } else {
        PatternKind::Unknown
    }
}

/// Quantise a series of onset times (in seconds) to a `RhythmGrid`.
///
/// * `onset_times` - onset positions in seconds
/// * `bpm` - tempo in beats per minute
/// * `beats` - number of beats in the grid
#[must_use]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn quantise_onsets(onset_times: &[f64], bpm: f64, beats: usize) -> RhythmGrid {
    let beat_dur = 60.0 / bpm;
    let total_dur = beat_dur * beats as f64;
    let mut grid = RhythmGrid::empty(beats, SUBDIVISIONS);

    for &t in onset_times {
        if t < 0.0 || t >= total_dur {
            continue;
        }
        let position = t / beat_dur; // in beats (fractional)
        let beat = position as usize;
        let frac = position - beat as f64;
        let sub = (frac * SUBDIVISIONS as f64).round() as usize;
        let sub = sub.min(SUBDIVISIONS - 1);
        if beat < beats {
            grid.set(beat, sub, true);
        }
    }

    grid
}

/// Compute the syncopation score of a grid (simple off-beat ratio).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn syncopation_score(grid: &RhythmGrid) -> f32 {
    if grid.onset_count() == 0 {
        return 0.0;
    }
    let mut off_beat = 0_usize;
    for (i, &active) in grid.grid.iter().enumerate() {
        if active && (i % grid.subdivisions) != 0 {
            off_beat += 1;
        }
    }
    off_beat as f32 / grid.onset_count() as f32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_grid() {
        let g = RhythmGrid::empty(4, 16);
        assert_eq!(g.onset_count(), 0);
        assert!((g.density() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_set_and_get() {
        let mut g = RhythmGrid::empty(4, 16);
        g.set(0, 0, true);
        assert!(g.get(0, 0));
        assert!(!g.get(0, 1));
    }

    #[test]
    fn test_four_on_the_floor() {
        let g = four_on_the_floor();
        assert_eq!(g.onset_count(), 4);
        for beat in 0..4 {
            assert!(g.get(beat, 0));
        }
    }

    #[test]
    fn test_breakbeat_pattern() {
        let g = breakbeat_pattern();
        assert!(g.onset_count() >= 3);
    }

    #[test]
    fn test_shuffle_pattern() {
        let g = shuffle_pattern();
        // 4 downbeats + 4 swung offbeats
        assert_eq!(g.onset_count(), 8);
    }

    #[test]
    fn test_half_time_pattern() {
        let g = half_time_pattern();
        assert_eq!(g.onset_count(), 2);
    }

    #[test]
    fn test_hamming_distance_identical() {
        let a = four_on_the_floor();
        let b = four_on_the_floor();
        assert_eq!(a.hamming_distance(&b), 0);
    }

    #[test]
    fn test_hamming_distance_different() {
        let a = four_on_the_floor();
        let b = half_time_pattern();
        assert!(a.hamming_distance(&b) > 0);
    }

    #[test]
    fn test_classify_four_on_the_floor() {
        let g = four_on_the_floor();
        assert_eq!(classify_pattern(&g), PatternKind::FourOnTheFloor);
    }

    #[test]
    fn test_classify_breakbeat() {
        let g = breakbeat_pattern();
        assert_eq!(classify_pattern(&g), PatternKind::Breakbeat);
    }

    #[test]
    fn test_quantise_onsets_basic() {
        // 120 BPM -> beat every 0.5 s.  Place onsets on beats 0,1,2,3.
        let onsets = vec![0.0, 0.5, 1.0, 1.5];
        let grid = quantise_onsets(&onsets, 120.0, 4);
        assert_eq!(grid.onset_count(), 4);
        for beat in 0..4 {
            assert!(grid.get(beat, 0));
        }
    }

    #[test]
    fn test_quantise_ignores_out_of_range() {
        let onsets = vec![-1.0, 100.0];
        let grid = quantise_onsets(&onsets, 120.0, 4);
        assert_eq!(grid.onset_count(), 0);
    }

    #[test]
    fn test_syncopation_zero() {
        // All onsets on the downbeat -> syncopation = 0
        let g = four_on_the_floor();
        let s = syncopation_score(&g);
        assert!((s - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_syncopation_high() {
        // All onsets on offbeats
        let mut g = RhythmGrid::empty(4, 16);
        for beat in 0..4 {
            g.set(beat, 8, true); // offbeat 8th
        }
        let s = syncopation_score(&g);
        assert!((s - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_density_full() {
        let mut g = RhythmGrid::empty(1, 4);
        for i in 0..4 {
            g.set(0, i, true);
        }
        assert!((g.density() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_classify_unknown_for_wrong_size() {
        let g = RhythmGrid::empty(2, 8);
        assert_eq!(classify_pattern(&g), PatternKind::Unknown);
    }
}
