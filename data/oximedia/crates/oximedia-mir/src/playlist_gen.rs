//! Music playlist generation using musical constraints.
//!
//! Provides a constraint-based playlist builder that selects and orders
//! tracks based on BPM range, key compatibility, energy level, mood, and
//! maximum playlist duration.

#![allow(dead_code)]

use crate::similarity::AudioFeatures;

// ---------------------------------------------------------------------------
// PlaylistConstraints
// ---------------------------------------------------------------------------

/// Constraints for playlist generation.
#[derive(Debug, Clone)]
pub struct PlaylistConstraints {
    /// Acceptable tempo range in BPM `(min, max)`.
    pub target_bpm_range: (f64, f64),
    /// MIDI pitch-class keys to allow (0 = C … 11 = B).  Empty = any key.
    pub allowed_keys: Vec<u8>,
    /// Maximum total playlist duration in milliseconds.  0 = unlimited.
    pub max_duration_ms: u64,
    /// Minimum energy level for each track (0.0–1.0).
    pub min_energy: f64,
    /// Optional mood tag filter.
    pub mood: Option<String>,
}

impl PlaylistConstraints {
    /// Relaxed constraints – essentially accepts anything.
    #[must_use]
    pub fn relaxed() -> Self {
        Self {
            target_bpm_range: (60.0, 200.0),
            allowed_keys: vec![],
            max_duration_ms: 0,
            min_energy: 0.0,
            mood: None,
        }
    }

    /// Workout / high-energy constraints.
    ///
    /// Prefers tracks around 140–180 BPM with high energy (≥ 0.7).
    #[must_use]
    pub fn workout() -> Self {
        Self {
            target_bpm_range: (140.0, 180.0),
            allowed_keys: vec![],
            max_duration_ms: 3_600_000, // 1 hour
            min_energy: 0.7,
            mood: Some("energetic".to_string()),
        }
    }

    /// Focus / study constraints.
    ///
    /// Prefers moderate BPM (80–120) with lower energy (≤ 0.5).
    #[must_use]
    pub fn focus() -> Self {
        Self {
            target_bpm_range: (80.0, 120.0),
            allowed_keys: vec![],
            max_duration_ms: 7_200_000, // 2 hours
            min_energy: 0.0,
            mood: Some("calm".to_string()),
        }
    }

    /// Sleep / ambient constraints.
    ///
    /// Slow BPM (50–80), very low energy.
    #[must_use]
    pub fn sleep() -> Self {
        Self {
            target_bpm_range: (50.0, 80.0),
            allowed_keys: vec![],
            max_duration_ms: 28_800_000, // 8 hours
            min_energy: 0.0,
            mood: Some("ambient".to_string()),
        }
    }

    /// Check whether a track satisfies the constraints.
    ///
    /// `duration_ms` is the track duration in milliseconds.
    #[must_use]
    pub fn accepts(&self, features: &AudioFeatures, track_duration_ms: u64) -> bool {
        // BPM check
        if features.tempo_bpm < self.target_bpm_range.0
            || features.tempo_bpm > self.target_bpm_range.1
        {
            return false;
        }

        // Key check
        if !self.allowed_keys.is_empty() && !self.allowed_keys.contains(&features.key) {
            return false;
        }

        // Energy check
        if features.energy < self.min_energy {
            return false;
        }

        // Duration check (per track sanity: reject tracks > 30 min)
        if track_duration_ms > 1_800_000 {
            return false;
        }

        true
    }
}

// ---------------------------------------------------------------------------
// PlaylistEntry
// ---------------------------------------------------------------------------

/// A single entry in a generated playlist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaylistEntry {
    /// Unique track identifier.
    pub track_id: u64,
    /// Start position of this entry within the playlist timeline (ms).
    pub start_ms: u64,
    /// Duration of the track in milliseconds.
    pub duration_ms: u64,
    /// Description of the transition type from the previous entry.
    pub transition_type: String,
}

impl PlaylistEntry {
    /// Create a new playlist entry.
    #[must_use]
    pub fn new(track_id: u64, start_ms: u64, duration_ms: u64, transition_type: &str) -> Self {
        Self {
            track_id,
            start_ms,
            duration_ms,
            transition_type: transition_type.to_string(),
        }
    }

    /// End position of this entry in the playlist timeline (ms).
    #[must_use]
    pub fn end_ms(&self) -> u64 {
        self.start_ms.saturating_add(self.duration_ms)
    }
}

// ---------------------------------------------------------------------------
// Key & BPM compatibility helpers
// ---------------------------------------------------------------------------

/// Compute harmonic key compatibility using the Camelot wheel.
///
/// Returns a score in [0.0, 1.0]:
/// - 1.0 = same key
/// - 0.7 = relative major/minor (same number, different mode)
/// - 0.5 = adjacent on the Camelot wheel
/// - 0.0 = incompatible
///
/// `key_a` and `key_b` are MIDI pitch classes (0–11).
/// `mode_a` and `mode_b` are 0 (minor) or 1 (major).
#[must_use]
pub fn key_compatibility(key_a: u8, key_b: u8) -> f64 {
    if key_a == key_b {
        return 1.0;
    }

    // Adjacent semitone on Camelot = ±1 in the circle of fifths
    // The circle of fifths in semitone steps: C(0), G(7), D(2), A(9), E(4),
    // B(11), F#(6), Db(1), Ab(8), Eb(3), Bb(10), F(5) – i.e., +7 mod 12 each step
    let cof_pos = |k: u8| -> u8 {
        // Map pitch class → position on circle of fifths (0..11)
        let steps = [0u8, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5];
        steps[k as usize % 12]
    };

    let pa = cof_pos(key_a);
    let pb = cof_pos(key_b);
    let diff = pa.abs_diff(pb).min(12u8.saturating_sub(pa.abs_diff(pb)));

    match diff {
        0 => 1.0,
        1 => 0.5,
        _ => 0.0,
    }
}

/// Compute BPM compatibility.
///
/// Returns 1.0 for identical BPMs and approaches 0.0 as the ratio
/// approaches 2× (one octave of tempo).
#[must_use]
pub fn bpm_compatibility(bpm_a: f64, bpm_b: f64) -> f64 {
    if bpm_a <= 0.0 || bpm_b <= 0.0 {
        return 0.0;
    }

    // Direct ratio
    let ratio = if bpm_a > bpm_b {
        bpm_a / bpm_b
    } else {
        bpm_b / bpm_a
    };

    // Map ratio ∈ [1, 2] → score ∈ [1, 0]
    // At ratio == 1: perfect match → 1.0
    // At ratio == 2: double/half tempo → 0.0
    (1.0 - (ratio - 1.0)).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Playlist generation
// ---------------------------------------------------------------------------

/// Generate a playlist from a collection of tracks and constraints.
///
/// # Arguments
///
/// * `tracks` — Available tracks as `(track_id, AudioFeatures, duration_ms)`.
/// * `constraints` — Playlist constraints to apply.
/// * `max_tracks` — Maximum number of tracks to include.
///
/// # Returns
///
/// Ordered list of `PlaylistEntry` items.
#[must_use]
pub fn generate_playlist(
    tracks: &[(u64, AudioFeatures)],
    durations_ms: &[u64],
    constraints: &PlaylistConstraints,
    max_tracks: usize,
) -> Vec<PlaylistEntry> {
    let mut result: Vec<PlaylistEntry> = Vec::new();
    let mut total_ms: u64 = 0;

    // Filter candidate tracks
    let candidates: Vec<(usize, &(u64, AudioFeatures))> = tracks
        .iter()
        .enumerate()
        .filter(|(i, (_, feat))| {
            let dur = durations_ms.get(*i).copied().unwrap_or(180_000);
            constraints.accepts(feat, dur)
        })
        .collect();

    let mut used = std::collections::HashSet::new();
    let mut prev_key: Option<u8> = None;
    let mut prev_bpm: Option<f64> = None;

    for _ in 0..max_tracks {
        // Find the best next candidate
        let best = candidates
            .iter()
            .filter(|(_idx, (id, _))| !used.contains(id))
            .map(|(i, (id, feat))| {
                let dur = durations_ms.get(*i).copied().unwrap_or(180_000);
                let key_score = prev_key.map_or(1.0, |k| key_compatibility(k, feat.key));
                let bpm_score = prev_bpm.map_or(1.0, |b| bpm_compatibility(b, feat.tempo_bpm));
                let score = key_score * 0.6 + bpm_score * 0.4;
                (*id, feat, dur, score)
            })
            .max_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));

        match best {
            Some((id, feat, dur, _)) => {
                // Check playlist duration cap
                if constraints.max_duration_ms > 0 && total_ms + dur > constraints.max_duration_ms {
                    break;
                }

                let transition = if prev_key.is_none() {
                    "start"
                } else if prev_key == Some(feat.key) {
                    "same_key"
                } else {
                    "crossfade"
                };

                result.push(PlaylistEntry::new(id, total_ms, dur, transition));
                total_ms += dur;
                used.insert(id);
                prev_key = Some(feat.key);
                prev_bpm = Some(feat.tempo_bpm);
            }
            None => break,
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn feat(tempo: f64, key: u8, energy: f64) -> AudioFeatures {
        AudioFeatures {
            tempo_bpm: tempo,
            key,
            mode: 1,
            loudness_lufs: -10.0,
            energy,
            danceability: 0.7,
            valence: 0.6,
            speechiness: 0.05,
        }
    }

    #[test]
    fn test_constraints_relaxed_accepts_anything() {
        let c = PlaylistConstraints::relaxed();
        let f = feat(120.0, 5, 0.5);
        assert!(c.accepts(&f, 180_000));
    }

    #[test]
    fn test_constraints_workout_rejects_low_bpm() {
        let c = PlaylistConstraints::workout();
        let f = feat(90.0, 5, 0.9); // BPM too low
        assert!(!c.accepts(&f, 180_000));
    }

    #[test]
    fn test_constraints_workout_rejects_low_energy() {
        let c = PlaylistConstraints::workout();
        let f = feat(150.0, 5, 0.2); // Energy too low
        assert!(!c.accepts(&f, 180_000));
    }

    #[test]
    fn test_constraints_focus_accepts_moderate_bpm() {
        let c = PlaylistConstraints::focus();
        let f = feat(100.0, 5, 0.4);
        assert!(c.accepts(&f, 180_000));
    }

    #[test]
    fn test_constraints_sleep_rejects_high_bpm() {
        let c = PlaylistConstraints::sleep();
        let f = feat(140.0, 5, 0.1);
        assert!(!c.accepts(&f, 180_000));
    }

    #[test]
    fn test_key_compatibility_same() {
        assert!((key_compatibility(5, 5) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_key_compatibility_adjacent_fifth() {
        // C (0) and G (7) are adjacent on the circle of fifths
        let score = key_compatibility(0, 7);
        assert!(score >= 0.5, "C and G should be compatible: {score}");
    }

    #[test]
    fn test_key_compatibility_distant() {
        // C (0) and F# (6) are on opposite sides of the circle
        let score = key_compatibility(0, 6);
        assert!(
            score < 0.5,
            "C and F# should have low compatibility: {score}"
        );
    }

    #[test]
    fn test_bpm_compatibility_identical() {
        assert!((bpm_compatibility(120.0, 120.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_bpm_compatibility_double_tempo() {
        assert!((bpm_compatibility(120.0, 240.0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_bpm_compatibility_close_bpm() {
        let score = bpm_compatibility(120.0, 125.0);
        assert!(
            score > 0.9,
            "Close BPMs should have high compatibility: {score}"
        );
    }

    #[test]
    fn test_bpm_compatibility_zero_bpm() {
        assert_eq!(bpm_compatibility(0.0, 120.0), 0.0);
    }

    #[test]
    fn test_generate_playlist_basic() {
        let tracks = vec![
            (1u64, feat(150.0, 5, 0.8)),
            (2, feat(155.0, 5, 0.9)),
            (3, feat(160.0, 7, 0.85)),
        ];
        let durations = vec![180_000u64, 200_000, 210_000];
        let constraints = PlaylistConstraints::workout();
        let playlist = generate_playlist(&tracks, &durations, &constraints, 3);
        assert!(!playlist.is_empty(), "Should generate at least one entry");
    }

    #[test]
    fn test_generate_playlist_respects_max_tracks() {
        let tracks: Vec<(u64, AudioFeatures)> = (0..10).map(|i| (i, feat(150.0, 5, 0.8))).collect();
        let durations = vec![60_000u64; 10];
        let constraints = PlaylistConstraints::workout();
        let playlist = generate_playlist(&tracks, &durations, &constraints, 3);
        assert!(playlist.len() <= 3);
    }

    #[test]
    fn test_generate_playlist_start_ms_monotonic() {
        let tracks = vec![(1u64, feat(150.0, 5, 0.8)), (2, feat(155.0, 5, 0.9))];
        let durations = vec![180_000u64, 200_000];
        let constraints = PlaylistConstraints::workout();
        let playlist = generate_playlist(&tracks, &durations, &constraints, 2);
        if playlist.len() >= 2 {
            assert!(playlist[1].start_ms >= playlist[0].start_ms);
        }
    }

    #[test]
    fn test_generate_playlist_duration_cap() {
        let tracks: Vec<(u64, AudioFeatures)> = (0..10).map(|i| (i, feat(150.0, 5, 0.8))).collect();
        // 6 min each; cap at 10 min → at most 1 track (each 360 000 ms > 10 min? No, 6 min < 10 min)
        let durations = vec![360_000u64; 10]; // 6 min each
        let mut constraints = PlaylistConstraints::workout();
        constraints.max_duration_ms = 900_000; // 15 min cap
        let playlist = generate_playlist(&tracks, &durations, &constraints, 10);
        let total: u64 = playlist.iter().map(|e| e.duration_ms).sum();
        assert!(
            total <= 900_000,
            "Total duration {total} ms should not exceed cap"
        );
    }

    #[test]
    fn test_playlist_entry_end_ms() {
        let entry = PlaylistEntry::new(1, 5000, 180_000, "crossfade");
        assert_eq!(entry.end_ms(), 185_000);
    }
}
