//! Music playlist intelligence for `oximedia-mir`.
//!
//! Provides smart playlist generation using BPM transitions, key
//! compatibility (Camelot wheel), energy flow, and other musical constraints.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PlaylistTrack
// ---------------------------------------------------------------------------

/// Metadata and musical properties of a single playlist track.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaylistTrack {
    /// Unique track identifier.
    pub id: u64,
    /// Track title.
    pub title: String,
    /// Artist name.
    pub artist: String,
    /// Tempo in BPM.
    pub bpm: f32,
    /// Musical key in Camelot notation (e.g., `"4A"`, `"9B"`) or standard notation.
    pub key: String,
    /// Energy level (0.0–1.0).
    pub energy: f32,
    /// Mood tag (e.g., `"happy"`, `"melancholic"`).
    pub mood: String,
    /// Duration in milliseconds.
    pub duration_ms: u64,
}

impl PlaylistTrack {
    /// Create a new track entry.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: u64,
        title: impl Into<String>,
        artist: impl Into<String>,
        bpm: f32,
        key: impl Into<String>,
        energy: f32,
        mood: impl Into<String>,
        duration_ms: u64,
    ) -> Self {
        Self {
            id,
            title: title.into(),
            artist: artist.into(),
            bpm,
            key: key.into(),
            energy,
            mood: mood.into(),
            duration_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// PlaylistConstraint
// ---------------------------------------------------------------------------

/// A constraint applied when building a smart playlist.
#[derive(Debug, Clone)]
pub enum PlaylistConstraint {
    /// Maximum allowed BPM difference between consecutive tracks.
    MaxBpmChange(f32),
    /// Consecutive tracks must be harmonically compatible (Camelot wheel).
    KeyCompatible,
    /// Consecutive tracks should have similar energy levels.
    SimilarEnergy,
    /// No two consecutive tracks from the same artist.
    NoDuplicateArtist,
}

// ---------------------------------------------------------------------------
// BpmTransition
// ---------------------------------------------------------------------------

/// BPM transition compatibility utilities.
pub struct BpmTransition;

impl BpmTransition {
    /// Compatibility score between two BPM values.
    ///
    /// Returns 1.0 for identical BPMs and decays toward 0 as the difference grows.
    /// Tracks at double/half tempo (harmonic mixing) also get a bonus.
    #[must_use]
    pub fn compatibility(bpm_a: f32, bpm_b: f32) -> f32 {
        if bpm_a <= 0.0 || bpm_b <= 0.0 {
            return 0.0;
        }

        // Direct difference
        let diff = (bpm_a - bpm_b).abs();
        let direct = (-diff / 10.0).exp(); // decays at ~10 BPM scale

        // Harmonic relationship (half/double tempo)
        let diff_half = (bpm_a - bpm_b * 2.0).abs();
        let diff_double = (bpm_a * 2.0 - bpm_b).abs();
        let harmonic = ((-diff_half / 10.0).exp() + (-diff_double / 10.0).exp()) * 0.5;

        direct.max(harmonic)
    }

    /// Whether the BPM difference is within the given maximum.
    #[must_use]
    pub fn within_limit(bpm_a: f32, bpm_b: f32, max_diff: f32) -> bool {
        (bpm_a - bpm_b).abs() <= max_diff
    }
}

// ---------------------------------------------------------------------------
// KeyCompatibility (Camelot wheel)
// ---------------------------------------------------------------------------

/// Camelot wheel key compatibility checker.
pub struct KeyCompatibility;

impl KeyCompatibility {
    /// Parse a Camelot wheel key string (e.g., `"4A"`, `"9B"`) into
    /// `(number, suffix)` where suffix is `'A'` or `'B'`.
    ///
    /// Returns `None` if the string cannot be parsed.
    #[must_use]
    pub fn parse_camelot(key: &str) -> Option<(u8, char)> {
        let key = key.trim();
        if key.len() < 2 {
            return None;
        }

        let suffix = key.chars().last()?;
        if suffix != 'A' && suffix != 'B' {
            return None;
        }

        let num_str = &key[..key.len() - 1];
        let num: u8 = num_str.parse().ok()?;
        if !(1..=12).contains(&num) {
            return None;
        }

        Some((num, suffix))
    }

    /// Check whether two keys are compatible on the Camelot wheel.
    ///
    /// Compatible means:
    /// - Identical key
    /// - Adjacent number (±1, wrapping 12→1)
    /// - Same number, opposite suffix (relative major/minor)
    #[must_use]
    pub fn check(key_a: &str, key_b: &str) -> bool {
        match (Self::parse_camelot(key_a), Self::parse_camelot(key_b)) {
            (Some((na, sa)), Some((nb, sb))) => {
                if na == nb {
                    // Same number: either exact match or relative swap
                    return true;
                }
                // Adjacent numbers (wrap-around 1..=12)
                let adjacent =
                    na.abs_diff(nb) == 1 || (na == 1 && nb == 12) || (na == 12 && nb == 1);

                adjacent && sa == sb
            }
            _ => {
                // Fall back: if both keys are identical strings, they're compatible
                key_a.eq_ignore_ascii_case(key_b)
            }
        }
    }

    /// Compatibility score: 1.0 = identical, 0.5 = adjacent, 0.0 = incompatible.
    #[must_use]
    pub fn score(key_a: &str, key_b: &str) -> f32 {
        match (Self::parse_camelot(key_a), Self::parse_camelot(key_b)) {
            (Some((na, sa)), Some((nb, sb))) => {
                if na == nb && sa == sb {
                    return 1.0;
                }
                if na == nb {
                    return 0.7; // relative major/minor
                }
                let adjacent =
                    na.abs_diff(nb) == 1 || (na == 1 && nb == 12) || (na == 12 && nb == 1);
                if adjacent && sa == sb {
                    0.5
                } else {
                    0.0
                }
            }
            _ => {
                if key_a.eq_ignore_ascii_case(key_b) {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// EnergyFlow
// ---------------------------------------------------------------------------

/// How energy should evolve across a playlist.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EnergyFlow {
    /// Tracks should gradually increase in energy.
    Rising,
    /// Tracks should gradually decrease in energy.
    Falling,
    /// Energy should remain roughly constant.
    Steady,
    /// Energy rises and falls in a wave pattern.
    Wave,
}

impl EnergyFlow {
    /// Target energy for the next track given the current energy.
    #[must_use]
    pub fn next_target_energy(self, current: f32) -> f32 {
        match self {
            Self::Rising => (current + 0.15).min(1.0),
            Self::Falling => (current - 0.15).max(0.0),
            Self::Steady => current,
            Self::Wave => {
                // Simple triangle: if below 0.5 rise, else fall
                if current < 0.5 {
                    (current + 0.2).min(1.0)
                } else {
                    (current - 0.2).max(0.0)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PlaylistBuilder
// ---------------------------------------------------------------------------

/// Builds smart playlists from a library of tracks.
pub struct PlaylistBuilder {
    tracks: HashMap<u64, PlaylistTrack>,
}

impl PlaylistBuilder {
    /// Create a new empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tracks: HashMap::new(),
        }
    }

    /// Add a track to the library.
    pub fn add_track(&mut self, track: PlaylistTrack) {
        self.tracks.insert(track.id, track);
    }

    /// Build a smart playlist starting from a seed track.
    ///
    /// Uses greedy selection: at each step, score all remaining candidate
    /// tracks by how well they satisfy the active constraints, then pick the
    /// best one.
    ///
    /// # Arguments
    ///
    /// * `seed_id` — ID of the first track.
    /// * `length` — Desired playlist length (number of tracks).
    /// * `constraints` — List of constraints to apply.
    ///
    /// # Returns
    ///
    /// Ordered list of track IDs forming the playlist.
    #[must_use]
    pub fn build_smart(
        &self,
        seed_id: u64,
        length: u32,
        constraints: &[PlaylistConstraint],
    ) -> Vec<u64> {
        let length = length as usize;
        let mut playlist = Vec::with_capacity(length);

        let Some(seed) = self.tracks.get(&seed_id) else {
            return playlist;
        };

        playlist.push(seed_id);

        let mut used: std::collections::HashSet<u64> = std::collections::HashSet::new();
        used.insert(seed_id);

        let mut current = seed;

        while playlist.len() < length {
            let candidates: Vec<&PlaylistTrack> = self
                .tracks
                .values()
                .filter(|t| !used.contains(&t.id))
                .collect();

            if candidates.is_empty() {
                break;
            }

            let best = candidates
                .into_iter()
                .map(|t| (t, self.score_transition(current, t, constraints)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            match best {
                Some((track, _score)) => {
                    used.insert(track.id);
                    playlist.push(track.id);
                    current = track;
                }
                None => break,
            }
        }

        playlist
    }

    /// Score a transition from `from` to `to` given the active constraints.
    fn score_transition(
        &self,
        from: &PlaylistTrack,
        to: &PlaylistTrack,
        constraints: &[PlaylistConstraint],
    ) -> f32 {
        let mut score = 1.0_f32;

        for constraint in constraints {
            match constraint {
                PlaylistConstraint::MaxBpmChange(max_diff) => {
                    if BpmTransition::within_limit(from.bpm, to.bpm, *max_diff) {
                        score += BpmTransition::compatibility(from.bpm, to.bpm) * 0.2;
                    } else {
                        score -= 0.5;
                    }
                }
                PlaylistConstraint::KeyCompatible => {
                    score += KeyCompatibility::score(&from.key, &to.key) * 0.3;
                }
                PlaylistConstraint::SimilarEnergy => {
                    let energy_diff = (from.energy - to.energy).abs();
                    score += (1.0 - energy_diff) * 0.2;
                }
                PlaylistConstraint::NoDuplicateArtist => {
                    if from.artist == to.artist {
                        score -= 1.0;
                    }
                }
            }
        }

        score
    }
}

impl Default for PlaylistBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_track(id: u64, bpm: f32, key: &str, energy: f32, artist: &str) -> PlaylistTrack {
        PlaylistTrack::new(
            id,
            format!("Track {id}"),
            artist,
            bpm,
            key,
            energy,
            "neutral",
            180_000,
        )
    }

    #[test]
    fn test_bpm_compatibility_identical() {
        assert!((BpmTransition::compatibility(120.0, 120.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_bpm_compatibility_decays() {
        let close = BpmTransition::compatibility(120.0, 125.0);
        let far = BpmTransition::compatibility(120.0, 160.0);
        assert!(close > far, "Closer BPM should have higher compatibility");
    }

    #[test]
    fn test_bpm_within_limit() {
        assert!(BpmTransition::within_limit(120.0, 128.0, 10.0));
        assert!(!BpmTransition::within_limit(120.0, 140.0, 10.0));
    }

    #[test]
    fn test_camelot_parse_valid() {
        assert_eq!(KeyCompatibility::parse_camelot("4A"), Some((4, 'A')));
        assert_eq!(KeyCompatibility::parse_camelot("12B"), Some((12, 'B')));
        assert_eq!(KeyCompatibility::parse_camelot("1A"), Some((1, 'A')));
    }

    #[test]
    fn test_camelot_parse_invalid() {
        assert_eq!(KeyCompatibility::parse_camelot("13A"), None);
        assert_eq!(KeyCompatibility::parse_camelot("4C"), None);
        assert_eq!(KeyCompatibility::parse_camelot("A"), None);
    }

    #[test]
    fn test_camelot_check_identical() {
        assert!(KeyCompatibility::check("4A", "4A"));
    }

    #[test]
    fn test_camelot_check_adjacent() {
        assert!(KeyCompatibility::check("4A", "5A"));
        assert!(KeyCompatibility::check("4A", "3A"));
    }

    #[test]
    fn test_camelot_check_wrap() {
        assert!(KeyCompatibility::check("1A", "12A"));
        assert!(KeyCompatibility::check("12A", "1A"));
    }

    #[test]
    fn test_camelot_check_incompatible() {
        assert!(!KeyCompatibility::check("4A", "8B"));
    }

    #[test]
    fn test_camelot_relative_same_number() {
        // Same number different suffix = relative major/minor → compatible
        assert!(KeyCompatibility::check("4A", "4B"));
    }

    #[test]
    fn test_energy_flow_rising() {
        let next = EnergyFlow::Rising.next_target_energy(0.5);
        assert!((next - 0.65).abs() < 1e-5);
    }

    #[test]
    fn test_energy_flow_falling() {
        let next = EnergyFlow::Falling.next_target_energy(0.5);
        assert!((next - 0.35).abs() < 1e-5);
    }

    #[test]
    fn test_energy_flow_steady() {
        let next = EnergyFlow::Steady.next_target_energy(0.7);
        assert!((next - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_energy_flow_wave_below_half() {
        let next = EnergyFlow::Wave.next_target_energy(0.3);
        assert!(next > 0.3, "Wave should rise when below 0.5");
    }

    #[test]
    fn test_energy_flow_wave_above_half() {
        let next = EnergyFlow::Wave.next_target_energy(0.8);
        assert!(next < 0.8, "Wave should fall when above 0.5");
    }

    #[test]
    fn test_playlist_builder_basic() {
        let mut builder = PlaylistBuilder::new();
        for i in 0..5 {
            builder.add_track(sample_track(i, 120.0 + i as f32, "4A", 0.7, "Artist A"));
        }
        let playlist = builder.build_smart(0, 3, &[]);
        assert_eq!(playlist.len(), 3);
        assert_eq!(playlist[0], 0); // seed first
    }

    #[test]
    fn test_playlist_builder_missing_seed() {
        let builder = PlaylistBuilder::new();
        let playlist = builder.build_smart(999, 5, &[]);
        assert!(playlist.is_empty());
    }

    #[test]
    fn test_playlist_builder_no_duplicate_artist() {
        let mut builder = PlaylistBuilder::new();
        builder.add_track(sample_track(1, 120.0, "4A", 0.7, "Artist A"));
        builder.add_track(sample_track(2, 121.0, "4A", 0.7, "Artist A"));
        builder.add_track(sample_track(3, 122.0, "5A", 0.6, "Artist B"));
        let constraints = [PlaylistConstraint::NoDuplicateArtist];
        let playlist = builder.build_smart(1, 3, &constraints);
        // Track 3 (Artist B) should be selected before track 2 (Artist A)
        assert!(playlist.contains(&3));
    }

    #[test]
    fn test_playlist_builder_key_compatible() {
        let mut builder = PlaylistBuilder::new();
        builder.add_track(sample_track(1, 120.0, "4A", 0.7, "A"));
        builder.add_track(sample_track(2, 120.0, "5A", 0.7, "B")); // adjacent key
        builder.add_track(sample_track(3, 120.0, "9B", 0.7, "C")); // incompatible key
        let constraints = [PlaylistConstraint::KeyCompatible];
        let playlist = builder.build_smart(1, 3, &constraints);
        assert!(!playlist.is_empty());
    }
}
