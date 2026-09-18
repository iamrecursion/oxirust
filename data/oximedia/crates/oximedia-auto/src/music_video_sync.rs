//! Advanced music-to-video synchronization.
//!
//! Provides a beat grid, synchronization strategies, and tools for planning
//! and scoring video cuts aligned to music beats, bars, and phrases.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Beat grid
// ---------------------------------------------------------------------------

/// A regular grid of beat timestamps for a music track.
#[derive(Debug, Clone)]
pub struct BeatGrid {
    /// Beats per minute of the track.
    pub bpm: f32,
    /// Timestamp in milliseconds of the first beat.
    pub first_beat_ms: i64,
    /// All beat timestamps in milliseconds, in ascending order.
    pub beats: Vec<i64>,
}

impl BeatGrid {
    /// Generate a beat grid from BPM, total duration, and first-beat offset.
    ///
    /// # Arguments
    ///
    /// * `bpm` — beats per minute (must be > 0)
    /// * `duration_ms` — total track length in milliseconds
    /// * `first_beat_offset_ms` — timestamp of the first beat (may be negative
    ///   if the grid starts before position 0)
    pub fn generate(bpm: f32, duration_ms: i64, first_beat_offset_ms: i64) -> Self {
        if bpm <= 0.0 || duration_ms <= 0 {
            return Self {
                bpm,
                first_beat_ms: first_beat_offset_ms,
                beats: Vec::new(),
            };
        }

        let beat_interval_ms = (60_000.0 / bpm) as i64;
        let mut beats = Vec::new();
        let mut t = first_beat_offset_ms;

        while t < duration_ms {
            if t >= 0 {
                beats.push(t);
            }
            t += beat_interval_ms;
        }

        Self {
            bpm,
            first_beat_ms: first_beat_offset_ms,
            beats,
        }
    }

    /// Number of beats in the grid.
    pub fn beat_count(&self) -> usize {
        self.beats.len()
    }

    /// Beat interval in milliseconds.
    pub fn beat_interval_ms(&self) -> f64 {
        if self.bpm <= 0.0 {
            return 0.0;
        }
        60_000.0 / self.bpm as f64
    }

    /// Bar interval in milliseconds (assuming 4/4 time).
    pub fn bar_interval_ms(&self) -> f64 {
        self.beat_interval_ms() * 4.0
    }

    /// Return the timestamp of the beat nearest to `time_ms`.
    pub fn nearest_beat(&self, time_ms: i64) -> Option<i64> {
        if self.beats.is_empty() {
            return None;
        }
        self.beats
            .iter()
            .min_by_key(|&&b| (b - time_ms).unsigned_abs())
            .copied()
    }

    /// Return the index of the nearest beat to `time_ms`.
    pub fn nearest_beat_index(&self, time_ms: i64) -> Option<usize> {
        if self.beats.is_empty() {
            return None;
        }
        self.beats
            .iter()
            .enumerate()
            .min_by_key(|(_, &b)| (b - time_ms).unsigned_abs())
            .map(|(i, _)| i)
    }

    /// Return all downbeat (bar-1) timestamps assuming 4/4 time.
    pub fn downbeats(&self) -> Vec<i64> {
        self.beats
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 4 == 0)
            .map(|(_, &t)| t)
            .collect()
    }

    /// Return timestamps that start a phrase of `bars_per_phrase` bars.
    pub fn phrase_starts(&self, bars_per_phrase: u8) -> Vec<i64> {
        if bars_per_phrase == 0 {
            return Vec::new();
        }
        let beats_per_phrase = bars_per_phrase as usize * 4;
        self.beats
            .iter()
            .enumerate()
            .filter(|(i, _)| i % beats_per_phrase == 0)
            .map(|(_, &t)| t)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Sync strategy and cut types
// ---------------------------------------------------------------------------

/// Determines at which musical positions cuts are placed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncStrategy {
    /// Cut on every downbeat (beat 1 of each bar).
    OnDownbeat,
    /// Cut on every single beat.
    OnBeat,
    /// Cut on every bar (same as `OnDownbeat` in 4/4).
    OnBar,
    /// Cut every N bars.
    OnPhrase(u8),
    /// Cut at audio transient positions (handled externally).
    OnTransient,
}

/// Direction of a wipe transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WipeDirection {
    /// Wipe from right to left.
    Left,
    /// Wipe from left to right.
    Right,
    /// Wipe from bottom to top.
    Top,
    /// Wipe from top to bottom.
    Bottom,
    /// Diagonal wipe.
    Diagonal,
}

/// The type of cut/transition applied at a sync point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutType {
    /// Instantaneous cut with no transition.
    HardCut,
    /// Gradual dissolve/cross-fade.
    ///
    /// The wrapped `u32` is the transition duration in frames.
    Dissolve(u32),
    /// Wipe transition in a given direction.
    Wipe(WipeDirection),
}

/// A planned video cut aligned to a music sync point.
#[derive(Debug, Clone)]
pub struct VideoSyncPoint {
    /// Timestamp of this cut in milliseconds.
    pub beat_ms: i64,
    /// Type of cut/transition to use at this point.
    pub cut_type: CutType,
}

impl VideoSyncPoint {
    /// Create a new hard-cut sync point.
    pub fn hard_cut(beat_ms: i64) -> Self {
        Self {
            beat_ms,
            cut_type: CutType::HardCut,
        }
    }

    /// Create a dissolve sync point.
    pub fn dissolve(beat_ms: i64, frames: u32) -> Self {
        Self {
            beat_ms,
            cut_type: CutType::Dissolve(frames),
        }
    }
}

// ---------------------------------------------------------------------------
// MusicVideoSyncer
// ---------------------------------------------------------------------------

/// Plans video cuts synchronized to a beat grid using a chosen strategy.
#[derive(Debug, Clone)]
pub struct MusicVideoSyncer {
    /// The beat grid to sync against.
    pub beat_grid: BeatGrid,
    /// The synchronization strategy.
    pub strategy: SyncStrategy,
    /// Tolerance in milliseconds: how far from the ideal beat a cut may land
    /// before it is considered out of sync.
    pub cut_tolerance_ms: u32,
}

impl MusicVideoSyncer {
    /// Create a new syncer.
    pub fn new(beat_grid: BeatGrid, strategy: SyncStrategy, cut_tolerance_ms: u32) -> Self {
        Self {
            beat_grid,
            strategy,
            cut_tolerance_ms,
        }
    }

    /// Plan a list of hard cuts for the full `video_duration_ms`.
    ///
    /// Positions are taken from the beat grid according to the chosen strategy.
    pub fn plan_cuts(&self, video_duration_ms: i64) -> Vec<VideoSyncPoint> {
        let cut_positions: Vec<i64> = match self.strategy {
            SyncStrategy::OnDownbeat | SyncStrategy::OnBar => self.beat_grid.downbeats(),
            SyncStrategy::OnBeat => self.beat_grid.beats.clone(),
            SyncStrategy::OnPhrase(n) => self.beat_grid.phrase_starts(n),
            SyncStrategy::OnTransient => {
                // Transients are external; return empty — caller should supply them
                Vec::new()
            }
        };

        cut_positions
            .into_iter()
            .filter(|&t| t >= 0 && t < video_duration_ms)
            .map(VideoSyncPoint::hard_cut)
            .collect()
    }

    /// Snap each cut in `cuts` to the nearest beat within `cut_tolerance_ms`.
    ///
    /// Cuts that fall outside the tolerance are left unchanged.
    pub fn re_align(&self, cuts: &mut Vec<VideoSyncPoint>) {
        let tolerance = self.cut_tolerance_ms as i64;
        for cut in cuts.iter_mut() {
            if let Some(nearest) = self.beat_grid.nearest_beat(cut.beat_ms) {
                if (nearest - cut.beat_ms).abs() <= tolerance {
                    cut.beat_ms = nearest;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

/// Score how well a set of cut timestamps align with a beat grid.
///
/// Returns a value in `[0, 1]` where `1.0` means every cut lands exactly on a
/// beat and `0.0` means cuts are maximally misaligned (half a beat interval
/// away on average).
pub fn score_cut_alignment(cuts: &[i64], beat_grid: &BeatGrid) -> f32 {
    if cuts.is_empty() || beat_grid.beats.is_empty() {
        return 0.0;
    }

    let half_interval = (beat_grid.beat_interval_ms() / 2.0).max(1.0);
    let mut total_dist = 0.0f64;

    for &cut_ms in cuts {
        let nearest = beat_grid
            .beats
            .iter()
            .min_by_key(|&&b| (b - cut_ms).unsigned_abs())
            .copied()
            .unwrap_or(cut_ms);
        total_dist += (nearest - cut_ms).unsigned_abs() as f64;
    }

    let mean_dist = total_dist / cuts.len() as f64;
    // Normalize: 0 dist → 1.0, half_interval dist → 0.0
    let normalized = 1.0 - (mean_dist / half_interval).min(1.0);
    normalized as f32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- BeatGrid::generate tests --

    #[test]
    fn test_beat_grid_generate_120bpm() {
        // 120 bpm = 500ms/beat; 4000ms duration → beats at 0,500,1000,1500,2000,2500,3000,3500
        let grid = BeatGrid::generate(120.0, 4000, 0);
        assert_eq!(grid.beats.len(), 8);
        assert_eq!(grid.beats[0], 0);
        assert_eq!(grid.beats[1], 500);
    }

    #[test]
    fn test_beat_grid_generate_zero_bpm() {
        let grid = BeatGrid::generate(0.0, 4000, 0);
        assert!(grid.beats.is_empty());
    }

    #[test]
    fn test_beat_grid_generate_negative_duration() {
        let grid = BeatGrid::generate(120.0, -1, 0);
        assert!(grid.beats.is_empty());
    }

    #[test]
    fn test_beat_grid_generate_with_offset() {
        // First beat at 250ms; 120bpm = 500ms/beat; duration 2000ms
        let grid = BeatGrid::generate(120.0, 2000, 250);
        assert_eq!(grid.beats[0], 250);
        assert_eq!(grid.beats[1], 750);
    }

    #[test]
    fn test_beat_grid_downbeats_4beat_bar() {
        let grid = BeatGrid::generate(120.0, 8000, 0);
        // Beats: 0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500
        // Downbeats (every 4): 0,2000,4000,6000
        let db = grid.downbeats();
        assert_eq!(db, vec![0, 2000, 4000, 6000]);
    }

    #[test]
    fn test_beat_grid_phrase_starts() {
        let grid = BeatGrid::generate(120.0, 16_000, 0);
        // phrase = 2 bars = 8 beats = 4000ms
        let phrases = grid.phrase_starts(2);
        assert_eq!(phrases[0], 0);
        assert_eq!(phrases[1], 4000);
    }

    #[test]
    fn test_nearest_beat_exact() {
        let grid = BeatGrid::generate(120.0, 4000, 0);
        assert_eq!(grid.nearest_beat(500), Some(500));
    }

    #[test]
    fn test_nearest_beat_between() {
        let grid = BeatGrid::generate(120.0, 4000, 0);
        // 600ms is closer to 500 than to 1000 (diff 100 vs 400)
        assert_eq!(grid.nearest_beat(600), Some(500));
    }

    #[test]
    fn test_nearest_beat_empty_grid() {
        let grid = BeatGrid {
            bpm: 120.0,
            first_beat_ms: 0,
            beats: Vec::new(),
        };
        assert_eq!(grid.nearest_beat(500), None);
    }

    // -- SyncStrategy / plan_cuts tests --

    #[test]
    fn test_plan_cuts_on_downbeat() {
        let grid = BeatGrid::generate(120.0, 8000, 0);
        let syncer = MusicVideoSyncer::new(grid, SyncStrategy::OnDownbeat, 50);
        let cuts = syncer.plan_cuts(8000);
        // downbeats at 0, 2000, 4000, 6000 — all < 8000
        assert_eq!(cuts.len(), 4);
        assert_eq!(cuts[0].beat_ms, 0);
        assert_eq!(cuts[1].beat_ms, 2000);
    }

    #[test]
    fn test_plan_cuts_on_beat() {
        let grid = BeatGrid::generate(120.0, 4000, 0);
        let syncer = MusicVideoSyncer::new(grid, SyncStrategy::OnBeat, 50);
        let cuts = syncer.plan_cuts(4000);
        assert_eq!(cuts.len(), 8);
    }

    #[test]
    fn test_plan_cuts_on_phrase() {
        let grid = BeatGrid::generate(120.0, 16_000, 0);
        let syncer = MusicVideoSyncer::new(grid, SyncStrategy::OnPhrase(2), 50);
        let cuts = syncer.plan_cuts(16_000);
        // phrase of 2 bars = 8 beats = 4000ms; cuts at 0,4000,8000,12000
        assert_eq!(cuts.len(), 4);
    }

    #[test]
    fn test_plan_cuts_on_transient_empty() {
        let grid = BeatGrid::generate(120.0, 8000, 0);
        let syncer = MusicVideoSyncer::new(grid, SyncStrategy::OnTransient, 50);
        let cuts = syncer.plan_cuts(8000);
        assert!(cuts.is_empty());
    }

    #[test]
    fn test_plan_cuts_respects_duration() {
        let grid = BeatGrid::generate(120.0, 10_000, 0);
        let syncer = MusicVideoSyncer::new(grid, SyncStrategy::OnBeat, 50);
        // Only cuts < 5000 should be included
        let cuts = syncer.plan_cuts(5000);
        for cut in &cuts {
            assert!(cut.beat_ms < 5000);
        }
    }

    // -- re_align tests --

    #[test]
    fn test_re_align_snaps_within_tolerance() {
        let grid = BeatGrid::generate(120.0, 4000, 0);
        let syncer = MusicVideoSyncer::new(grid, SyncStrategy::OnBeat, 100);
        let mut cuts = vec![VideoSyncPoint::hard_cut(480)]; // 20ms from beat at 500
        syncer.re_align(&mut cuts);
        assert_eq!(cuts[0].beat_ms, 500);
    }

    #[test]
    fn test_re_align_leaves_outside_tolerance() {
        let grid = BeatGrid::generate(120.0, 4000, 0);
        let syncer = MusicVideoSyncer::new(grid, SyncStrategy::OnBeat, 50);
        let mut cuts = vec![VideoSyncPoint::hard_cut(650)]; // 150ms from 500, 350ms from 1000
        syncer.re_align(&mut cuts);
        // 650ms: nearest beat is 500 (diff=150) but tolerance=50 → not snapped
        assert_eq!(cuts[0].beat_ms, 650);
    }

    // -- score_cut_alignment tests --

    #[test]
    fn test_score_perfect_alignment() {
        let grid = BeatGrid::generate(120.0, 4000, 0);
        // All cuts land exactly on beats
        let cuts: Vec<i64> = grid.beats.clone();
        let score = score_cut_alignment(&cuts, &grid);
        assert!((score - 1.0).abs() < 1e-5, "expected 1.0, got {score}");
    }

    #[test]
    fn test_score_empty_cuts() {
        let grid = BeatGrid::generate(120.0, 4000, 0);
        let score = score_cut_alignment(&[], &grid);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_score_empty_grid() {
        let grid = BeatGrid {
            bpm: 120.0,
            first_beat_ms: 0,
            beats: Vec::new(),
        };
        let score = score_cut_alignment(&[500, 1000], &grid);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_score_half_interval_misalignment() {
        let grid = BeatGrid::generate(120.0, 4000, 0);
        // 120bpm → 500ms interval; cuts at 250ms offset from every beat
        let cuts: Vec<i64> = grid.beats.iter().map(|&b| b + 250).collect();
        let score = score_cut_alignment(&cuts, &grid);
        // Mean distance = 250 = half interval → score ≈ 0.0
        assert!(score < 0.1, "expected near-zero, got {score}");
    }

    #[test]
    fn test_score_in_valid_range() {
        let grid = BeatGrid::generate(90.0, 10_000, 0);
        let cuts = vec![0, 333, 667, 1000, 1500];
        let score = score_cut_alignment(&cuts, &grid);
        assert!(score >= 0.0 && score <= 1.0, "score out of range: {score}");
    }

    // -- VideoSyncPoint convenience constructors --

    #[test]
    fn test_video_sync_point_hard_cut() {
        let p = VideoSyncPoint::hard_cut(1000);
        assert_eq!(p.beat_ms, 1000);
        assert_eq!(p.cut_type, CutType::HardCut);
    }

    #[test]
    fn test_video_sync_point_dissolve() {
        let p = VideoSyncPoint::dissolve(2000, 12);
        assert_eq!(p.beat_ms, 2000);
        assert!(matches!(p.cut_type, CutType::Dissolve(12)));
    }

    #[test]
    fn test_beat_interval_ms() {
        let grid = BeatGrid::generate(120.0, 4000, 0);
        assert!((grid.beat_interval_ms() - 500.0).abs() < 1e-6);
    }

    #[test]
    fn test_bar_interval_ms() {
        let grid = BeatGrid::generate(120.0, 4000, 0);
        assert!((grid.bar_interval_ms() - 2000.0).abs() < 1e-6);
    }
}
