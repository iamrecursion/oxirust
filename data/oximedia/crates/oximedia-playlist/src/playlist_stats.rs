//! Play statistics tracking for playlists and individual tracks.
//!
//! In addition to per-track play event accounting ([`PlaylistStats`]),
//! this module provides [`PlaylistSummaryStats`] for computing aggregate
//! metadata over a collection of playlist tracks — including total duration,
//! track count, BPM statistics, and genre distribution — without requiring
//! any external crates.

use serde::Serialize;
use std::collections::HashMap;
use std::time::Duration;

// ── Playlist-level summary statistics ─────────────────────────────────────────

/// Metadata for a single track used by [`PlaylistSummaryStats`].
#[derive(Debug, Clone)]
pub struct TrackInfo {
    /// Track URI or identifier.
    pub id: String,
    /// Track duration (exact; `None` if unknown).
    pub duration: Option<Duration>,
    /// Tempo in beats per minute (`None` if unknown).
    pub bpm: Option<f64>,
    /// Genre tag (optional).
    pub genre: Option<String>,
}

impl TrackInfo {
    /// Creates a [`TrackInfo`] with only the id populated.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            duration: None,
            bpm: None,
            genre: None,
        }
    }

    /// Sets the duration.
    #[must_use]
    pub const fn with_duration(mut self, d: Duration) -> Self {
        self.duration = Some(d);
        self
    }

    /// Sets the BPM.
    #[must_use]
    pub fn with_bpm(mut self, bpm: f64) -> Self {
        self.bpm = Some(bpm.max(0.0));
        self
    }

    /// Sets the genre.
    #[must_use]
    pub fn with_genre(mut self, genre: impl Into<String>) -> Self {
        self.genre = Some(genre.into());
        self
    }
}

/// Aggregated summary statistics over a collection of playlist tracks.
///
/// # Example
///
/// ```
/// use oximedia_playlist::playlist_stats::{PlaylistSummaryStats, TrackInfo};
/// use std::time::Duration;
///
/// let tracks = vec![
///     TrackInfo::new("t1").with_duration(Duration::from_secs(210)).with_bpm(128.0),
///     TrackInfo::new("t2").with_duration(Duration::from_secs(195)).with_bpm(132.0),
/// ];
/// let stats = PlaylistSummaryStats::compute(&tracks);
/// assert_eq!(stats.track_count, 2);
/// assert!((stats.average_bpm.unwrap() - 130.0).abs() < 1.0);
/// ```
#[derive(Debug, Clone, Default, Serialize)]
pub struct PlaylistSummaryStats {
    /// Total number of tracks in the playlist.
    pub track_count: usize,
    /// Sum of all known track durations.
    pub total_duration: Duration,
    /// Number of tracks with a known duration.
    pub tracks_with_duration: usize,
    /// Average track duration (computed over tracks with known duration).
    pub average_duration: Option<Duration>,
    /// Average BPM across tracks with a known BPM value.
    pub average_bpm: Option<f64>,
    /// Minimum BPM among tracks with a known BPM.
    pub min_bpm: Option<f64>,
    /// Maximum BPM among tracks with a known BPM.
    pub max_bpm: Option<f64>,
    /// Number of tracks with a known BPM.
    pub tracks_with_bpm: usize,
    /// Genre distribution: genre → track count.
    pub genre_distribution: HashMap<String, usize>,
    /// Number of tracks with no genre tag.
    pub untagged_count: usize,
}

impl PlaylistSummaryStats {
    /// Compute statistics over a slice of [`TrackInfo`] records.
    #[must_use]
    pub fn compute(tracks: &[TrackInfo]) -> Self {
        let track_count = tracks.len();
        let mut total_millis: u128 = 0;
        let mut tracks_with_duration: usize = 0;
        let mut bpm_sum: f64 = 0.0;
        let mut bpm_min: f64 = f64::INFINITY;
        let mut bpm_max: f64 = f64::NEG_INFINITY;
        let mut tracks_with_bpm: usize = 0;
        let mut genre_distribution: HashMap<String, usize> = HashMap::new();
        let mut untagged_count: usize = 0;

        for track in tracks {
            if let Some(dur) = track.duration {
                total_millis += dur.as_millis();
                tracks_with_duration += 1;
            }
            if let Some(bpm) = track.bpm {
                if bpm > 0.0 {
                    bpm_sum += bpm;
                    if bpm < bpm_min {
                        bpm_min = bpm;
                    }
                    if bpm > bpm_max {
                        bpm_max = bpm;
                    }
                    tracks_with_bpm += 1;
                }
            }
            match &track.genre {
                Some(g) => {
                    *genre_distribution.entry(g.clone()).or_insert(0) += 1;
                }
                None => {
                    untagged_count += 1;
                }
            }
        }

        let total_duration = Duration::from_millis(total_millis as u64);

        let average_duration = if tracks_with_duration > 0 {
            Some(Duration::from_millis(
                (total_millis / tracks_with_duration as u128) as u64,
            ))
        } else {
            None
        };

        let average_bpm = if tracks_with_bpm > 0 {
            Some(bpm_sum / tracks_with_bpm as f64)
        } else {
            None
        };

        let min_bpm = if bpm_min.is_finite() {
            Some(bpm_min)
        } else {
            None
        };
        let max_bpm = if bpm_max.is_finite() {
            Some(bpm_max)
        } else {
            None
        };

        Self {
            track_count,
            total_duration,
            tracks_with_duration,
            average_duration,
            average_bpm,
            min_bpm,
            max_bpm,
            tracks_with_bpm,
            genre_distribution,
            untagged_count,
        }
    }

    /// Alias for `compute`; used by the cache layer in `PlaylistStats`.
    #[must_use]
    pub fn from_tracks(tracks: &[TrackInfo]) -> Self {
        Self::compute(tracks)
    }

    /// Returns the total duration as fractional seconds.
    #[must_use]
    pub fn total_duration_secs(&self) -> f64 {
        self.total_duration.as_secs_f64()
    }

    /// Returns the most common genre, or `None` if no genres are tagged.
    #[must_use]
    pub fn dominant_genre(&self) -> Option<&str> {
        self.genre_distribution
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(genre, _)| genre.as_str())
    }

    /// Returns `true` if all tracks have a known BPM.
    #[must_use]
    pub fn all_bpm_known(&self) -> bool {
        self.track_count > 0 && self.tracks_with_bpm == self.track_count
    }

    /// Returns `true` if all tracks have a known duration.
    #[must_use]
    pub fn all_durations_known(&self) -> bool {
        self.track_count > 0 && self.tracks_with_duration == self.track_count
    }
}

/// Statistics for a single track across all play events.
#[derive(Debug, Clone, Default, Serialize)]
pub struct PlayStats {
    /// Number of times the track was played to completion.
    pub complete_plays: u32,
    /// Number of times the track was skipped before completion.
    pub skips: u32,
    /// Cumulative seconds of playback time for this track.
    pub total_duration_secs: f64,
}

impl PlayStats {
    /// Creates a new zero-initialised `PlayStats`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the total number of play events (complete + skipped).
    pub fn total_plays(&self) -> u32 {
        self.complete_plays + self.skips
    }

    /// Returns the average duration per play event in seconds.
    ///
    /// Returns `0.0` if there have been no play events.
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_duration_secs(&self) -> f64 {
        let total = self.total_plays();
        if total == 0 {
            0.0
        } else {
            self.total_duration_secs / f64::from(total)
        }
    }

    /// Returns the completion rate as a value in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` if there have been no play events.
    pub fn completion_rate(&self) -> f64 {
        let total = self.total_plays();
        if total == 0 {
            0.0
        } else {
            f64::from(self.complete_plays) / f64::from(total)
        }
    }
}

/// An individual play-event record.
#[derive(Debug, Clone)]
pub struct PlayEvent {
    /// Track URI or identifier.
    pub track_id: String,
    /// Duration actually played in seconds.
    pub played_secs: f64,
    /// Whether the track was played to completion.
    pub completed: bool,
}

impl PlayEvent {
    /// Creates a completed play event.
    pub fn completed(track_id: impl Into<String>, played_secs: f64) -> Self {
        Self {
            track_id: track_id.into(),
            played_secs,
            completed: true,
        }
    }

    /// Creates a skipped play event.
    pub fn skipped(track_id: impl Into<String>, played_secs: f64) -> Self {
        Self {
            track_id: track_id.into(),
            played_secs,
            completed: false,
        }
    }
}

/// A serialisable row for JSON export.
#[derive(Debug, Clone, Serialize)]
struct TrackStatsRow {
    track_id: String,
    plays: u32,
    completion_rate: f64,
    skips: u32,
}

/// Top-level export payload for `PlaylistStats::to_json`.
#[derive(Debug, Serialize)]
struct PlaylistStatsExport {
    tracks: Vec<TrackStatsRow>,
    total_plays: usize,
    unique_tracks: usize,
}

/// Aggregated play statistics for an entire playlist.
#[derive(Debug, Default)]
pub struct PlaylistStats {
    /// Per-track statistics keyed by track ID.
    track_stats: HashMap<String, PlayStats>,
    /// Chronological history of play events.
    events: Vec<PlayEvent>,
    /// Cached summary statistics; `None` when `is_dirty` is true.
    cached_summary: Option<PlaylistSummaryStats>,
    /// Set to `true` whenever the track stats change.
    is_dirty: bool,
}

impl PlaylistStats {
    /// Creates a new empty `PlaylistStats`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a play event and updates the underlying per-track statistics.
    pub fn record_play(&mut self, event: PlayEvent) {
        let stats = self.track_stats.entry(event.track_id.clone()).or_default();
        if event.completed {
            stats.complete_plays += 1;
        } else {
            stats.skips += 1;
        }
        stats.total_duration_secs += event.played_secs;
        self.events.push(event);
        self.is_dirty = true;
        self.cached_summary = None;
    }

    /// Returns the total number of play events recorded.
    pub fn total_plays(&self) -> usize {
        self.events.len()
    }

    /// Returns the number of unique tracks that have been played.
    pub fn unique_tracks(&self) -> usize {
        self.track_stats.len()
    }

    /// Returns the track ID with the most play events, or `None` if empty.
    pub fn most_played(&self) -> Option<&str> {
        self.track_stats
            .iter()
            .max_by_key(|(_, s)| s.total_plays())
            .map(|(id, _)| id.as_str())
    }

    /// Returns the `PlayStats` for a specific track, or `None` if not found.
    pub fn stats_for(&self, track_id: &str) -> Option<&PlayStats> {
        self.track_stats.get(track_id)
    }

    /// Returns the track with the highest completion rate, or `None` if empty.
    pub fn best_completion_rate_track(&self) -> Option<&str> {
        self.track_stats
            .iter()
            .filter(|(_, s)| s.total_plays() > 0)
            .max_by(|(_, a), (_, b)| {
                a.completion_rate()
                    .partial_cmp(&b.completion_rate())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(id, _)| id.as_str())
    }

    /// Returns the total number of skips across all tracks.
    pub fn total_skips(&self) -> u32 {
        self.track_stats.values().map(|s| s.skips).sum()
    }

    /// Clears all recorded statistics.
    pub fn reset(&mut self) {
        self.track_stats.clear();
        self.events.clear();
        self.is_dirty = true;
        self.cached_summary = None;
    }

    // ── Cached summary ──────────────────────────────────────────────────────

    /// Build `TrackInfo` records from the current play stats so that
    /// `PlaylistSummaryStats::from_tracks` can be called.
    fn build_track_infos(&self) -> Vec<TrackInfo> {
        self.track_stats
            .keys()
            .map(|id| TrackInfo::new(id))
            .collect()
    }

    /// Return aggregate summary statistics, recomputing only when the
    /// underlying data has changed since the last call.
    pub fn summary(&mut self) -> PlaylistSummaryStats {
        if self.is_dirty || self.cached_summary.is_none() {
            let infos = self.build_track_infos();
            self.cached_summary = Some(PlaylistSummaryStats::from_tracks(&infos));
            self.is_dirty = false;
        }
        self.cached_summary.clone().unwrap_or_default()
    }

    // ── Export methods ──────────────────────────────────────────────────────

    /// Serialise play statistics to a pretty-printed JSON string.
    ///
    /// # Errors
    ///
    /// Returns a [`serde_json::Error`] if serialisation fails (in practice
    /// this should never happen for this type).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        let tracks: Vec<TrackStatsRow> = self
            .track_stats
            .iter()
            .map(|(id, stats)| TrackStatsRow {
                track_id: id.clone(),
                plays: stats.total_plays(),
                completion_rate: stats.completion_rate(),
                skips: stats.skips,
            })
            .collect();

        let export = PlaylistStatsExport {
            total_plays: self.events.len(),
            unique_tracks: self.track_stats.len(),
            tracks,
        };

        serde_json::to_string_pretty(&export)
    }

    /// Serialise play statistics to CSV format.
    ///
    /// The returned string begins with a header row and contains one data row
    /// per tracked track.
    pub fn to_csv(&self) -> String {
        let mut out = String::from("track_id,plays,completion_rate,skips\n");
        for (id, stats) in &self.track_stats {
            // Escape commas in the track ID by quoting.
            let safe_id = if id.contains(',') {
                format!("\"{id}\"")
            } else {
                id.clone()
            };
            out.push_str(&format!(
                "{},{},{:.3},{}\n",
                safe_id,
                stats.total_plays(),
                stats.completion_rate(),
                stats.skips
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_play_stats_default_zero() {
        let s = PlayStats::new();
        assert_eq!(s.total_plays(), 0);
        assert_eq!(s.avg_duration_secs(), 0.0);
        assert_eq!(s.completion_rate(), 0.0);
    }

    #[test]
    fn test_play_stats_total_plays() {
        let s = PlayStats {
            complete_plays: 3,
            skips: 2,
            total_duration_secs: 0.0,
        };
        assert_eq!(s.total_plays(), 5);
    }

    #[test]
    fn test_avg_duration_secs() {
        let s = PlayStats {
            complete_plays: 2,
            skips: 0,
            total_duration_secs: 300.0,
        };
        assert!((s.avg_duration_secs() - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_completion_rate_full() {
        let s = PlayStats {
            complete_plays: 4,
            skips: 0,
            total_duration_secs: 0.0,
        };
        assert!((s.completion_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_completion_rate_partial() {
        let s = PlayStats {
            complete_plays: 1,
            skips: 3,
            total_duration_secs: 0.0,
        };
        assert!((s.completion_rate() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_record_play_completed() {
        let mut ps = PlaylistStats::new();
        ps.record_play(PlayEvent::completed("track_a", 180.0));
        assert_eq!(ps.total_plays(), 1);
        assert_eq!(ps.unique_tracks(), 1);
        let s = ps.stats_for("track_a").expect("should succeed in test");
        assert_eq!(s.complete_plays, 1);
        assert_eq!(s.skips, 0);
    }

    #[test]
    fn test_record_play_skipped() {
        let mut ps = PlaylistStats::new();
        ps.record_play(PlayEvent::skipped("track_b", 30.0));
        let s = ps.stats_for("track_b").expect("should succeed in test");
        assert_eq!(s.skips, 1);
        assert_eq!(s.complete_plays, 0);
    }

    #[test]
    fn test_unique_tracks() {
        let mut ps = PlaylistStats::new();
        ps.record_play(PlayEvent::completed("t1", 100.0));
        ps.record_play(PlayEvent::completed("t2", 200.0));
        ps.record_play(PlayEvent::skipped("t1", 50.0));
        assert_eq!(ps.unique_tracks(), 2);
    }

    #[test]
    fn test_most_played() {
        let mut ps = PlaylistStats::new();
        ps.record_play(PlayEvent::completed("popular", 100.0));
        ps.record_play(PlayEvent::completed("popular", 100.0));
        ps.record_play(PlayEvent::completed("rare", 100.0));
        assert_eq!(ps.most_played(), Some("popular"));
    }

    #[test]
    fn test_most_played_empty() {
        let ps = PlaylistStats::new();
        assert!(ps.most_played().is_none());
    }

    #[test]
    fn test_total_skips() {
        let mut ps = PlaylistStats::new();
        ps.record_play(PlayEvent::skipped("t1", 10.0));
        ps.record_play(PlayEvent::skipped("t1", 5.0));
        ps.record_play(PlayEvent::completed("t2", 60.0));
        assert_eq!(ps.total_skips(), 2);
    }

    #[test]
    fn test_reset() {
        let mut ps = PlaylistStats::new();
        ps.record_play(PlayEvent::completed("t1", 60.0));
        ps.reset();
        assert_eq!(ps.total_plays(), 0);
        assert_eq!(ps.unique_tracks(), 0);
    }

    #[test]
    fn test_stats_for_unknown_track() {
        let ps = PlaylistStats::new();
        assert!(ps.stats_for("nonexistent").is_none());
    }

    #[test]
    fn test_best_completion_rate_track() {
        let mut ps = PlaylistStats::new();
        // t1: 2 complete, 2 skip → 50%
        ps.record_play(PlayEvent::completed("t1", 60.0));
        ps.record_play(PlayEvent::completed("t1", 60.0));
        ps.record_play(PlayEvent::skipped("t1", 10.0));
        ps.record_play(PlayEvent::skipped("t1", 10.0));
        // t2: 3 complete, 0 skip → 100%
        ps.record_play(PlayEvent::completed("t2", 90.0));
        ps.record_play(PlayEvent::completed("t2", 90.0));
        ps.record_play(PlayEvent::completed("t2", 90.0));
        assert_eq!(ps.best_completion_rate_track(), Some("t2"));
    }

    // ── PlaylistSummaryStats tests ───────────────────────────────────────────

    #[test]
    fn test_summary_stats_empty() {
        let stats = PlaylistSummaryStats::compute(&[]);
        assert_eq!(stats.track_count, 0);
        assert_eq!(stats.total_duration, Duration::ZERO);
        assert!(stats.average_bpm.is_none());
        assert!(stats.average_duration.is_none());
    }

    #[test]
    fn test_summary_stats_track_count() {
        let tracks: Vec<TrackInfo> = (0..5).map(|i| TrackInfo::new(format!("t{i}"))).collect();
        let stats = PlaylistSummaryStats::compute(&tracks);
        assert_eq!(stats.track_count, 5);
    }

    #[test]
    fn test_summary_stats_total_duration() {
        let tracks = vec![
            TrackInfo::new("t1").with_duration(Duration::from_secs(200)),
            TrackInfo::new("t2").with_duration(Duration::from_secs(300)),
        ];
        let stats = PlaylistSummaryStats::compute(&tracks);
        assert_eq!(stats.total_duration, Duration::from_secs(500));
        assert_eq!(stats.tracks_with_duration, 2);
    }

    #[test]
    fn test_summary_stats_average_duration() {
        let tracks = vec![
            TrackInfo::new("t1").with_duration(Duration::from_secs(100)),
            TrackInfo::new("t2").with_duration(Duration::from_secs(200)),
            TrackInfo::new("t3").with_duration(Duration::from_secs(300)),
        ];
        let stats = PlaylistSummaryStats::compute(&tracks);
        let avg = stats.average_duration.expect("should have average");
        // Average = 200 s (±1 ms due to integer millis)
        let diff_ms = avg.as_millis() as i64 - 200_000;
        assert!(
            diff_ms.unsigned_abs() <= 1,
            "avg_duration off by {diff_ms} ms"
        );
    }

    #[test]
    fn test_summary_stats_average_bpm() {
        let tracks = vec![
            TrackInfo::new("t1").with_bpm(120.0),
            TrackInfo::new("t2").with_bpm(130.0),
            TrackInfo::new("t3").with_bpm(140.0),
        ];
        let stats = PlaylistSummaryStats::compute(&tracks);
        let avg = stats.average_bpm.expect("should have bpm");
        assert!((avg - 130.0).abs() < 1e-6, "expected 130 bpm, got {avg}");
    }

    #[test]
    fn test_summary_stats_min_max_bpm() {
        let tracks = vec![
            TrackInfo::new("t1").with_bpm(80.0),
            TrackInfo::new("t2").with_bpm(160.0),
            TrackInfo::new("t3").with_bpm(120.0),
        ];
        let stats = PlaylistSummaryStats::compute(&tracks);
        assert!((stats.min_bpm.expect("min") - 80.0).abs() < 1e-6);
        assert!((stats.max_bpm.expect("max") - 160.0).abs() < 1e-6);
    }

    #[test]
    fn test_summary_stats_genre_distribution() {
        let tracks = vec![
            TrackInfo::new("t1").with_genre("rock"),
            TrackInfo::new("t2").with_genre("jazz"),
            TrackInfo::new("t3").with_genre("rock"),
            TrackInfo::new("t4"),
        ];
        let stats = PlaylistSummaryStats::compute(&tracks);
        assert_eq!(stats.genre_distribution.get("rock").copied(), Some(2));
        assert_eq!(stats.genre_distribution.get("jazz").copied(), Some(1));
        assert_eq!(stats.untagged_count, 1);
    }

    #[test]
    fn test_summary_stats_dominant_genre() {
        let tracks = vec![
            TrackInfo::new("t1").with_genre("pop"),
            TrackInfo::new("t2").with_genre("pop"),
            TrackInfo::new("t3").with_genre("rock"),
        ];
        let stats = PlaylistSummaryStats::compute(&tracks);
        assert_eq!(stats.dominant_genre(), Some("pop"));
    }

    #[test]
    fn test_summary_stats_all_bpm_known() {
        let tracks = vec![
            TrackInfo::new("t1").with_bpm(120.0),
            TrackInfo::new("t2").with_bpm(130.0),
        ];
        let stats = PlaylistSummaryStats::compute(&tracks);
        assert!(stats.all_bpm_known());
    }

    #[test]
    fn test_summary_stats_not_all_bpm_known() {
        let tracks = vec![
            TrackInfo::new("t1").with_bpm(120.0),
            TrackInfo::new("t2"), // no BPM
        ];
        let stats = PlaylistSummaryStats::compute(&tracks);
        assert!(!stats.all_bpm_known());
    }

    #[test]
    fn test_summary_stats_total_duration_secs() {
        let tracks = vec![
            TrackInfo::new("t1").with_duration(Duration::from_secs(60)),
            TrackInfo::new("t2").with_duration(Duration::from_secs(120)),
        ];
        let stats = PlaylistSummaryStats::compute(&tracks);
        assert!((stats.total_duration_secs() - 180.0).abs() < 0.001);
    }

    #[test]
    fn test_summary_stats_partial_durations() {
        let tracks = vec![
            TrackInfo::new("t1").with_duration(Duration::from_secs(100)),
            TrackInfo::new("t2"), // no duration
        ];
        let stats = PlaylistSummaryStats::compute(&tracks);
        assert_eq!(stats.tracks_with_duration, 1);
        assert_eq!(stats.total_duration, Duration::from_secs(100));
    }

    // ── PlaylistStats cache tests ────────────────────────────────────────────

    #[test]
    fn test_stats_cache_same_as_fresh() {
        let mut ps = PlaylistStats::new();
        ps.record_play(PlayEvent::completed("t1", 180.0));
        ps.record_play(PlayEvent::skipped("t1", 30.0));
        ps.record_play(PlayEvent::completed("t2", 200.0));

        let cached = ps.summary();
        // The cached summary must report the same track count as a fresh compute.
        assert_eq!(cached.track_count, 2);
    }

    #[test]
    fn test_stats_cache_invalidates_on_record_play() {
        let mut ps = PlaylistStats::new();
        ps.record_play(PlayEvent::completed("t1", 60.0));

        let first = ps.summary();
        assert_eq!(first.track_count, 1);

        // Add a second track and check that the summary reflects it.
        ps.record_play(PlayEvent::completed("t2", 60.0));
        let second = ps.summary();
        assert_eq!(second.track_count, 2);
    }

    // ── Export method tests ──────────────────────────────────────────────────

    #[test]
    fn test_to_json_roundtrip() {
        let mut ps = PlaylistStats::new();
        ps.record_play(PlayEvent::completed("track_a", 120.0));
        ps.record_play(PlayEvent::skipped("track_b", 30.0));

        let json = ps.to_json().expect("to_json should succeed");
        let value: serde_json::Value = serde_json::from_str(&json).expect("JSON should parse back");

        let tracks = value.get("tracks").expect("tracks key missing");
        assert!(tracks.is_array());
        assert!(!tracks.as_array().expect("is array").is_empty());
    }

    #[test]
    fn test_to_csv_format() {
        let mut ps = PlaylistStats::new();
        ps.record_play(PlayEvent::completed("alpha", 100.0));
        ps.record_play(PlayEvent::skipped("beta", 20.0));

        let csv = ps.to_csv();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines[0], "track_id,plays,completion_rate,skips");
        // One header + two data rows
        assert_eq!(lines.len(), 3, "expected header + 2 data rows");
        // Each data row must have 4 comma-separated fields.
        for line in &lines[1..] {
            let fields: Vec<&str> = line.splitn(4, ',').collect();
            assert_eq!(fields.len(), 4, "row '{line}' does not have 4 fields");
        }
    }

    #[test]
    fn test_export_empty_stats() {
        let ps = PlaylistStats::new();

        let json = ps.to_json().expect("to_json on empty stats should succeed");
        let value: serde_json::Value = serde_json::from_str(&json).expect("JSON should parse back");
        let tracks = value.get("tracks").expect("tracks key missing");
        assert!(
            tracks.as_array().map(Vec::is_empty).unwrap_or(false),
            "tracks should be empty array"
        );

        let csv = ps.to_csv();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 1, "empty stats should have header only");
        assert_eq!(lines[0], "track_id,plays,completion_rate,skips");
    }
}
