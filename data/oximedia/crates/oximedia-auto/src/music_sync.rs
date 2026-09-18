//! Music-synchronized automated editing.
//!
//! Generates cut points aligned to a beat grid and paces clips to match
//! the energy of the music.

#![allow(dead_code)]

use crate::narrative::ClipInfo;

/// A grid of beat positions extracted from a music track.
#[derive(Debug, Clone)]
pub struct BeatGrid {
    /// Beats per minute of the track.
    pub bpm: f32,
    /// Timestamps (ms) of downbeats (beat 1 of each bar).
    pub downbeats_ms: Vec<u64>,
    /// Timestamps (ms) of every beat.
    pub beats_ms: Vec<u64>,
}

impl BeatGrid {
    /// Create a beat grid from BPM and downbeat positions.
    pub fn new(bpm: f32, downbeats_ms: Vec<u64>, beats_ms: Vec<u64>) -> Self {
        Self {
            bpm,
            downbeats_ms,
            beats_ms,
        }
    }

    /// Build a simple beat grid by interpolation from BPM alone.
    ///
    /// `duration_ms` is the total track length. `beats_per_bar` is usually 4.
    pub fn from_bpm(bpm: f32, duration_ms: u64, beats_per_bar: u32) -> Self {
        if bpm <= 0.0 || duration_ms == 0 {
            return Self::new(bpm, vec![], vec![]);
        }

        let beat_interval_ms = (60_000.0 / bpm) as u64;
        let mut beats_ms = Vec::new();
        let mut t = 0u64;
        while t < duration_ms {
            beats_ms.push(t);
            t += beat_interval_ms;
        }

        let downbeats_ms = beats_ms
            .iter()
            .enumerate()
            .filter(|(i, _)| i % beats_per_bar as usize == 0)
            .map(|(_, &t)| t)
            .collect();

        Self::new(bpm, downbeats_ms, beats_ms)
    }

    /// Return the timestamp of the beat nearest to `time_ms`.
    pub fn nearest_beat(&self, time_ms: u64) -> u64 {
        if self.beats_ms.is_empty() {
            return time_ms;
        }
        self.beats_ms
            .iter()
            .min_by_key(|&&b| b.abs_diff(time_ms))
            .copied()
            .unwrap_or(time_ms)
    }

    /// Return the number of bars between two timestamps.
    pub fn bars_between(&self, start_ms: u64, end_ms: u64) -> f32 {
        if self.bpm <= 0.0 {
            return 0.0;
        }
        let beat_interval_ms = 60_000.0 / self.bpm;
        let beats_per_bar = 4.0_f32;
        let duration_ms = end_ms.saturating_sub(start_ms) as f32;
        duration_ms / (beat_interval_ms * beats_per_bar)
    }
}

/// The metrical position of a beat within a bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeatPosition {
    /// Beat 1 of the bar (strongest, "downbeat").
    Downbeat,
    /// Beat 3 in 4/4 time (strong off-beat).
    StrongUpbeat,
    /// Beat 2 or 4 in 4/4 time (upbeat / back-beat).
    Upbeat,
    /// Beat subdivision (e.g. "and" between beats in 8th-note mode).
    Subdivision,
}

impl BeatPosition {
    /// Priority weight for edit point selection.
    ///
    /// Higher = more preferred for placing a cut.
    #[must_use]
    pub const fn cut_priority(&self) -> f32 {
        match self {
            Self::Downbeat => 1.0,
            Self::StrongUpbeat => 0.75,
            Self::Upbeat => 0.55,
            Self::Subdivision => 0.30,
        }
    }
}

/// A candidate cut point aligned to the music.
#[derive(Debug, Clone)]
pub struct CutPoint {
    /// Timestamp of this cut in milliseconds.
    pub time_ms: u64,
    /// True if this cut lands on a bar downbeat.
    pub is_downbeat: bool,
    /// Metrical position of this beat within the bar.
    pub beat_position: BeatPosition,
    /// Energy level at this point in the music (0.0–1.0).
    pub energy: f32,
    /// Composite priority for edit selection (higher = more preferred).
    pub priority: f32,
}

impl CutPoint {
    /// Create a new cut point.
    pub fn new(time_ms: u64, is_downbeat: bool, energy: f32) -> Self {
        let beat_position = if is_downbeat {
            BeatPosition::Downbeat
        } else {
            BeatPosition::Upbeat
        };
        let priority = beat_position.cut_priority() * energy;
        Self {
            time_ms,
            is_downbeat,
            beat_position,
            energy,
            priority,
        }
    }

    /// Create a cut point with an explicit beat position.
    pub fn with_beat_position(time_ms: u64, beat_position: BeatPosition, energy: f32) -> Self {
        let is_downbeat = matches!(beat_position, BeatPosition::Downbeat);
        let priority = beat_position.cut_priority() * energy;
        Self {
            time_ms,
            is_downbeat,
            beat_position,
            energy,
            priority,
        }
    }
}

/// Configuration for music-synchronised editing.
#[derive(Debug, Clone)]
pub struct MusicSyncConfig {
    /// If true, snap all cuts to the nearest beat.
    pub snap_to_beat: bool,
    /// If true, prefer cuts on downbeats.
    pub cut_on_downbeat: bool,
    /// Maximum clip length expressed in bars.
    pub max_clip_duration_beats: f32,
    /// If true, match clip energy to beat energy.
    pub energy_match: bool,
    /// Number of beats per bar (default: 4 for 4/4 time).
    pub beats_per_bar: u32,
    /// Only generate cut points on beats with priority >= this threshold (0.0-1.0).
    pub min_cut_priority: f32,
}

impl Default for MusicSyncConfig {
    fn default() -> Self {
        Self {
            snap_to_beat: true,
            cut_on_downbeat: true,
            max_clip_duration_beats: 4.0,
            energy_match: true,
            beats_per_bar: 4,
            min_cut_priority: 0.0,
        }
    }
}

/// Generates music-aligned cut points from a beat grid.
pub struct MusicSyncEditor;

impl MusicSyncEditor {
    /// Generate a list of cut points for a piece of music.
    ///
    /// `cuts_per_bar` controls the cutting density: 1.0 = one cut per bar,
    /// 2.0 = one cut per two beats, 0.5 = one cut per two bars, etc.
    ///
    /// Each [`CutPoint`] now carries a [`BeatPosition`] (downbeat, strong
    /// upbeat, upbeat, or subdivision) and a composite `priority` score that
    /// can be used to select the most musically appropriate edit points.
    pub fn generate_cut_points(
        grid: &BeatGrid,
        duration_ms: u64,
        cuts_per_bar: f32,
    ) -> Vec<CutPoint> {
        Self::generate_cut_points_with_config(
            grid,
            duration_ms,
            cuts_per_bar,
            &MusicSyncConfig::default(),
        )
    }

    /// Generate cut points with explicit configuration.
    ///
    /// This variant assigns [`BeatPosition`] labels and filters by
    /// `config.min_cut_priority`, giving callers fine-grained control over
    /// which beats are promoted to edit points.
    pub fn generate_cut_points_with_config(
        grid: &BeatGrid,
        duration_ms: u64,
        cuts_per_bar: f32,
        config: &MusicSyncConfig,
    ) -> Vec<CutPoint> {
        if grid.beats_ms.is_empty() || cuts_per_bar <= 0.0 {
            return Vec::new();
        }

        let beats_per_bar = config.beats_per_bar.max(1) as usize;
        // Determine the step size in beats
        let step_beats = ((beats_per_bar as f32 / cuts_per_bar).max(1.0)) as usize;

        let downbeat_set: std::collections::HashSet<u64> =
            grid.downbeats_ms.iter().copied().collect();

        grid.beats_ms
            .iter()
            .enumerate()
            .filter(|(i, &t)| i % step_beats == 0 && t < duration_ms)
            .filter_map(|(i, &t)| {
                // Classify beat position within the bar
                let beat_in_bar = i % beats_per_bar;
                let beat_position = if downbeat_set.contains(&t) {
                    BeatPosition::Downbeat
                } else if beat_in_bar == beats_per_bar / 2 {
                    BeatPosition::StrongUpbeat
                } else if beat_in_bar % 2 == 0 {
                    BeatPosition::Upbeat
                } else {
                    BeatPosition::Upbeat
                };

                // Simple ramp: energy increases towards the middle of the track
                let progress = t as f32 / duration_ms.max(1) as f32;
                let energy = 1.0 - (progress - 0.5).abs() * 2.0;
                let energy = energy.clamp(0.0, 1.0);

                let cut = CutPoint::with_beat_position(t, beat_position, energy);

                // Filter by minimum priority if configured
                if cut.priority >= config.min_cut_priority {
                    Some(cut)
                } else {
                    None
                }
            })
            .collect()
    }

    /// From a list of cut points, return only the downbeat cuts.
    #[must_use]
    pub fn downbeat_cuts(cuts: &[CutPoint]) -> Vec<&CutPoint> {
        cuts.iter()
            .filter(|c| c.beat_position == BeatPosition::Downbeat)
            .collect()
    }

    /// From a list of cut points, return only the upbeat cuts.
    #[must_use]
    pub fn upbeat_cuts(cuts: &[CutPoint]) -> Vec<&CutPoint> {
        cuts.iter()
            .filter(|c| !matches!(c.beat_position, BeatPosition::Downbeat))
            .collect()
    }

    /// Select the best `n` cut points by priority score.
    #[must_use]
    pub fn top_cuts(cuts: &[CutPoint], n: usize) -> Vec<&CutPoint> {
        let mut sorted: Vec<&CutPoint> = cuts.iter().collect();
        sorted.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(n);
        // Re-sort by timestamp for chronological output
        sorted.sort_by_key(|c| c.time_ms);
        sorted
    }
}

/// Paces clips against a list of cut points, matching energy.
pub struct ClipPacer;

impl ClipPacer {
    /// Assign clips to cut points, returning `(start_ms, clip)` pairs.
    ///
    /// High-energy cut points receive high-energy clips where possible.
    pub fn pace_clips(clips: &[ClipInfo], cut_points: &[CutPoint]) -> Vec<(u64, ClipInfo)> {
        if clips.is_empty() || cut_points.is_empty() {
            return Vec::new();
        }

        // Sort clips by energy descending so we can pair them with high-energy beats
        let mut sorted_clips: Vec<&ClipInfo> = clips.iter().collect();
        sorted_clips.sort_by(|a, b| {
            b.energy
                .partial_cmp(&a.energy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Sort cut points by energy descending as well
        let mut indexed_cuts: Vec<(usize, &CutPoint)> = cut_points.iter().enumerate().collect();
        indexed_cuts.sort_by(|a, b| {
            b.1.energy
                .partial_cmp(&a.1.energy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Assign clips to cut-point slots by energy rank
        let mut assignments: Vec<(usize, usize)> = Vec::new(); // (cut_idx, clip_rank)
        let n = sorted_clips.len().min(indexed_cuts.len());
        for rank in 0..n {
            assignments.push((indexed_cuts[rank].0, rank));
        }
        // Sort assignments back by cut index for chronological output
        assignments.sort_by_key(|(cut_idx, _)| *cut_idx);

        assignments
            .into_iter()
            .map(|(cut_idx, clip_rank)| {
                let start_ms = cut_points[cut_idx].time_ms;
                let clip = sorted_clips[clip_rank].clone();
                (start_ms, clip)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clip(id: u64, energy: f32) -> ClipInfo {
        ClipInfo::new(id, 3.0, "action", energy, 0.9)
    }

    #[test]
    fn test_beat_grid_from_bpm() {
        let grid = BeatGrid::from_bpm(120.0, 4000, 4);
        // 120 bpm = 500ms per beat; 4 seconds → beats at 0,500,1000,1500,2000,2500,3000,3500
        assert_eq!(grid.beats_ms.len(), 8);
        assert_eq!(grid.beats_ms[0], 0);
        assert_eq!(grid.beats_ms[1], 500);
    }

    #[test]
    fn test_downbeats_extracted() {
        let grid = BeatGrid::from_bpm(120.0, 4000, 4);
        // beats every 500ms, downbeats every 4 beats = every 2000ms → 2 downbeats (0ms, 2000ms)
        assert_eq!(grid.downbeats_ms.len(), 2);
        assert!(grid.downbeats_ms.contains(&0));
        assert!(grid.downbeats_ms.contains(&2000));
    }

    #[test]
    fn test_nearest_beat() {
        let grid = BeatGrid::from_bpm(120.0, 4000, 4);
        assert_eq!(grid.nearest_beat(0), 0);
        assert_eq!(grid.nearest_beat(300), 500);
        assert_eq!(grid.nearest_beat(200), 0);
    }

    #[test]
    fn test_bars_between() {
        let grid = BeatGrid::from_bpm(120.0, 10_000, 4);
        // 120bpm: beat_interval=500ms, bar=2000ms; 4000ms = 2 bars
        let bars = grid.bars_between(0, 4000);
        assert!((bars - 2.0).abs() < 0.05, "Expected ~2 bars, got {bars}");
    }

    #[test]
    fn test_generate_cut_points_count() {
        let grid = BeatGrid::from_bpm(120.0, 8000, 4);
        // 120bpm = 500ms/beat; 8000ms → beats at 0,500,...,7500 → 16 beats
        // cuts_per_bar=1 → step=4 beats → cuts at beat indices 0,4,8,12 → 4 cuts
        let cuts = MusicSyncEditor::generate_cut_points(&grid, 8000, 1.0);
        assert_eq!(cuts.len(), 4);
    }

    #[test]
    fn test_generate_cut_points_downbeat_flag() {
        let grid = BeatGrid::from_bpm(120.0, 8000, 4);
        let cuts = MusicSyncEditor::generate_cut_points(&grid, 8000, 1.0);
        // All cuts at step=4 should be downbeats (every 4 beats = bar start)
        for cut in &cuts {
            assert!(cut.is_downbeat, "Expected downbeat at {}", cut.time_ms);
        }
    }

    #[test]
    fn test_generate_cut_points_empty_grid() {
        let grid = BeatGrid::new(120.0, vec![], vec![]);
        let cuts = MusicSyncEditor::generate_cut_points(&grid, 8000, 1.0);
        assert!(cuts.is_empty());
    }

    #[test]
    fn test_clip_pacer_basic() {
        let grid = BeatGrid::from_bpm(120.0, 8000, 4);
        let cuts = MusicSyncEditor::generate_cut_points(&grid, 8000, 2.0);
        let clips = vec![make_clip(1, 0.9), make_clip(2, 0.4)];
        let paced = ClipPacer::pace_clips(&clips, &cuts);
        assert_eq!(paced.len(), 2);
        // First result should be chronologically first cut
        assert!(paced[0].0 <= paced[1].0);
    }

    #[test]
    fn test_clip_pacer_empty_clips() {
        let cuts = vec![CutPoint::new(0, true, 0.8)];
        let paced = ClipPacer::pace_clips(&[], &cuts);
        assert!(paced.is_empty());
    }

    #[test]
    fn test_clip_pacer_empty_cuts() {
        let clips = vec![make_clip(1, 0.8)];
        let paced = ClipPacer::pace_clips(&clips, &[]);
        assert!(paced.is_empty());
    }

    #[test]
    fn test_cut_point_energy_range() {
        let grid = BeatGrid::from_bpm(120.0, 10_000, 4);
        let cuts = MusicSyncEditor::generate_cut_points(&grid, 10_000, 1.0);
        for cut in &cuts {
            assert!(cut.energy >= 0.0 && cut.energy <= 1.0);
        }
    }
}
