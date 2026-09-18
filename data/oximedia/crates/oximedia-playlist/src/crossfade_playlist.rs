//! Crossfade-aware playlist management.
//!
//! Tracks crossfade mode and duration per entry, computing accurate total
//! playlist duration by accounting for overlaps between consecutive tracks.

#![allow(dead_code)]

/// The crossfade/transition mode between two playlist entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossfadeMode {
    /// No transition; hard cut between tracks.
    None,
    /// Automatic crossfade duration chosen by the engine.
    Auto,
    /// Fade out the current track then fade in the next.
    FadeOutFadeIn,
    /// Simultaneous fade out + fade in (true crossfade).
    CrossFade,
    /// Crossfade aligned to the musical beat grid.
    BeatMatch,
}

impl CrossfadeMode {
    /// Returns `true` for modes that produce an audio/video transition.
    pub fn uses_transition(&self) -> bool {
        !matches!(self, CrossfadeMode::None)
    }

    /// Returns a sensible default crossfade duration in milliseconds.
    pub fn duration_ms(&self) -> u32 {
        match self {
            CrossfadeMode::None => 0,
            CrossfadeMode::Auto => 3_000,
            CrossfadeMode::FadeOutFadeIn => 4_000,
            CrossfadeMode::CrossFade => 3_000,
            CrossfadeMode::BeatMatch => 2_000,
        }
    }
}

/// Crossfade configuration attached to a playlist entry.
#[derive(Debug, Clone)]
pub struct PlaylistCrossfade {
    /// Crossfade mode.
    pub mode: CrossfadeMode,
    /// Override duration in milliseconds (used instead of `mode.duration_ms()` when set).
    pub duration_ms: u32,
}

impl Default for PlaylistCrossfade {
    fn default() -> Self {
        Self {
            mode: CrossfadeMode::None,
            duration_ms: 0,
        }
    }
}

impl PlaylistCrossfade {
    /// Creates a crossfade with a specific mode, using the mode's default duration.
    pub fn with_mode(mode: CrossfadeMode) -> Self {
        let duration_ms = mode.duration_ms();
        Self { mode, duration_ms }
    }
}

/// A single entry in a crossfade playlist.
#[derive(Debug, Clone)]
pub struct CrossfadeEntry {
    /// Track identifier.
    pub track_id: u64,
    /// Full duration of this track in milliseconds.
    pub duration_ms: u64,
    /// Crossfade applied at the *end* of this entry (overlap with next entry).
    pub crossfade: PlaylistCrossfade,
    /// Playback gain (1.0 = unity).
    pub gain: f32,
}

impl CrossfadeEntry {
    /// Returns the effective end timestamp of this entry when it starts at `start_ms`.
    ///
    /// The end is the track start plus its full duration, minus the crossfade
    /// overlap (so the next track starts `crossfade.duration_ms` before this one ends).
    pub fn effective_end_ms(&self, start_ms: u64) -> u64 {
        let end = start_ms + self.duration_ms;
        let overlap = u64::from(self.crossfade.duration_ms);
        end.saturating_sub(overlap)
    }
}

/// A playlist of crossfade-aware entries.
#[derive(Debug, Clone, Default)]
pub struct CrossfadePlaylist {
    /// Ordered list of entries.
    pub entries: Vec<CrossfadeEntry>,
}

impl CrossfadePlaylist {
    /// Appends an entry to the playlist.
    pub fn add(&mut self, entry: CrossfadeEntry) {
        self.entries.push(entry);
    }

    /// Returns the total playback duration in milliseconds.
    ///
    /// Each crossfade overlap is subtracted once (the overlap between consecutive
    /// entries). The last entry is played to its full duration.
    pub fn total_duration_ms(&self) -> u64 {
        if self.entries.is_empty() {
            return 0;
        }
        let raw_total: u64 = self.entries.iter().map(|e| e.duration_ms).sum();
        // Subtract overlaps from all entries except the last (its crossfade is ignored).
        let overlap: u64 = self.entries[..self.entries.len().saturating_sub(1)]
            .iter()
            .map(|e| u64::from(e.crossfade.duration_ms))
            .sum();
        raw_total.saturating_sub(overlap)
    }

    /// Returns the start time in milliseconds for the entry at `index`.
    ///
    /// Returns 0 for index 0 or if `index >= entry_count()`.
    pub fn track_start_ms(&self, index: usize) -> u64 {
        if index == 0 || index >= self.entries.len() {
            return 0;
        }
        let mut start = 0u64;
        for e in &self.entries[..index] {
            start = start + e.duration_ms - u64::from(e.crossfade.duration_ms);
        }
        start
    }

    /// Returns the number of entries in the playlist.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- CrossfadeMode ---

    #[test]
    fn test_none_no_transition() {
        assert!(!CrossfadeMode::None.uses_transition());
    }

    #[test]
    fn test_auto_uses_transition() {
        assert!(CrossfadeMode::Auto.uses_transition());
    }

    #[test]
    fn test_fade_out_fade_in_uses_transition() {
        assert!(CrossfadeMode::FadeOutFadeIn.uses_transition());
    }

    #[test]
    fn test_crossfade_uses_transition() {
        assert!(CrossfadeMode::CrossFade.uses_transition());
    }

    #[test]
    fn test_beat_match_uses_transition() {
        assert!(CrossfadeMode::BeatMatch.uses_transition());
    }

    #[test]
    fn test_none_duration_zero() {
        assert_eq!(CrossfadeMode::None.duration_ms(), 0);
    }

    #[test]
    fn test_auto_duration_nonzero() {
        assert!(CrossfadeMode::Auto.duration_ms() > 0);
    }

    // --- PlaylistCrossfade ---

    #[test]
    fn test_default_crossfade_is_none() {
        let cf = PlaylistCrossfade::default();
        assert_eq!(cf.mode, CrossfadeMode::None);
        assert_eq!(cf.duration_ms, 0);
    }

    #[test]
    fn test_with_mode_uses_mode_duration() {
        let cf = PlaylistCrossfade::with_mode(CrossfadeMode::CrossFade);
        assert_eq!(cf.duration_ms, CrossfadeMode::CrossFade.duration_ms());
    }

    // --- CrossfadeEntry ---

    #[test]
    fn test_effective_end_no_crossfade() {
        let entry = CrossfadeEntry {
            track_id: 1,
            duration_ms: 10_000,
            crossfade: PlaylistCrossfade::default(),
            gain: 1.0,
        };
        assert_eq!(entry.effective_end_ms(0), 10_000);
    }

    #[test]
    fn test_effective_end_with_crossfade() {
        let entry = CrossfadeEntry {
            track_id: 1,
            duration_ms: 10_000,
            crossfade: PlaylistCrossfade::with_mode(CrossfadeMode::CrossFade), // 3000ms
            gain: 1.0,
        };
        // 0 + 10000 - 3000 = 7000
        assert_eq!(entry.effective_end_ms(0), 7_000);
    }

    // --- CrossfadePlaylist ---

    fn make_playlist() -> CrossfadePlaylist {
        let mut pl = CrossfadePlaylist::default();
        // Track A: 10s, 3s crossfade into B
        pl.add(CrossfadeEntry {
            track_id: 1,
            duration_ms: 10_000,
            crossfade: PlaylistCrossfade::with_mode(CrossfadeMode::CrossFade),
            gain: 1.0,
        });
        // Track B: 8s, no crossfade (last)
        pl.add(CrossfadeEntry {
            track_id: 2,
            duration_ms: 8_000,
            crossfade: PlaylistCrossfade::default(),
            gain: 1.0,
        });
        pl
    }

    #[test]
    fn test_entry_count() {
        let pl = make_playlist();
        assert_eq!(pl.entry_count(), 2);
    }

    #[test]
    fn test_total_duration_with_overlap() {
        let pl = make_playlist();
        // 10000 + 8000 - 3000 (crossfade of first track) = 15000
        assert_eq!(pl.total_duration_ms(), 15_000);
    }

    #[test]
    fn test_total_duration_empty() {
        let pl = CrossfadePlaylist::default();
        assert_eq!(pl.total_duration_ms(), 0);
    }

    #[test]
    fn test_track_start_first() {
        let pl = make_playlist();
        assert_eq!(pl.track_start_ms(0), 0);
    }

    #[test]
    fn test_track_start_second() {
        let pl = make_playlist();
        // Track B starts at 10000 - 3000 = 7000
        assert_eq!(pl.track_start_ms(1), 7_000);
    }

    #[test]
    fn test_track_start_out_of_bounds() {
        let pl = make_playlist();
        assert_eq!(pl.track_start_ms(99), 0);
    }
}
