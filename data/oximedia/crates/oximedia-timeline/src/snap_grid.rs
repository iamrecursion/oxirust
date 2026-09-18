//! Snap-to-grid functionality for the timeline editor.
//!
//! Provides configurable snap points (clip edges, markers, bar/beat grid,
//! playhead) and the logic to find the nearest one for an arbitrary frame
//! position.

/// A single point on the timeline that the cursor can snap to.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapPoint {
    /// Frame position of this snap point.
    pub frame: u64,
    /// Human-readable description (e.g. "Clip start", "Bar 4").
    pub label: String,
    /// Category of this snap point.
    pub kind: SnapKind,
}

impl SnapPoint {
    /// Create a new snap point.
    #[must_use]
    pub fn new(frame: u64, label: impl Into<String>, kind: SnapKind) -> Self {
        Self {
            frame,
            label: label.into(),
            kind,
        }
    }
}

/// The type of snap point.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapKind {
    /// Leading edge of a clip.
    ClipStart,
    /// Trailing edge of a clip.
    ClipEnd,
    /// A user-placed marker.
    Marker,
    /// A bar or beat boundary in the musical grid.
    BeatGrid,
    /// The current playhead position.
    Playhead,
    /// A chapter point.
    Chapter,
}

impl SnapKind {
    /// Returns `true` for structural points (clip edges / chapters).
    #[must_use]
    pub fn is_structural(&self) -> bool {
        matches!(self, Self::ClipStart | Self::ClipEnd | Self::Chapter)
    }
}

/// Which categories of snap points are active.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SnapConfig {
    /// Snap to clip start/end edges.
    pub snap_to_clips: bool,
    /// Snap to markers.
    pub snap_to_markers: bool,
    /// Snap to beat-grid points.
    pub snap_to_beats: bool,
    /// Snap to the playhead position.
    pub snap_to_playhead: bool,
    /// Maximum distance (in frames) within which snapping activates.
    pub snap_radius: u64,
}

impl Default for SnapConfig {
    fn default() -> Self {
        Self {
            snap_to_clips: true,
            snap_to_markers: true,
            snap_to_beats: false,
            snap_to_playhead: true,
            snap_radius: 5,
        }
    }
}

impl SnapConfig {
    /// Returns `true` if the given snap kind is enabled.
    #[must_use]
    pub fn is_enabled(&self, kind: SnapKind) -> bool {
        match kind {
            SnapKind::ClipStart | SnapKind::ClipEnd => self.snap_to_clips,
            SnapKind::Marker | SnapKind::Chapter => self.snap_to_markers,
            SnapKind::BeatGrid => self.snap_to_beats,
            SnapKind::Playhead => self.snap_to_playhead,
        }
    }
}

/// Manages a collection of snap points and resolves snap queries.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct SnapGrid {
    points: Vec<SnapPoint>,
    config: SnapConfig,
}

impl SnapGrid {
    /// Create a new, empty snap grid with default config.
    #[must_use]
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            config: SnapConfig::default(),
        }
    }

    /// Create a snap grid with custom configuration.
    #[must_use]
    pub fn with_config(config: SnapConfig) -> Self {
        Self {
            points: Vec::new(),
            config,
        }
    }

    /// Add a snap point.
    pub fn add(&mut self, point: SnapPoint) {
        self.points.push(point);
    }

    /// Add a beat-grid at a regular interval starting from `origin`.
    pub fn add_beat_grid(&mut self, origin: u64, interval: u64, count: u32) {
        for i in 0..count {
            let frame = origin + interval * u64::from(i);
            self.add(SnapPoint::new(
                frame,
                format!("Beat {}", i + 1),
                SnapKind::BeatGrid,
            ));
        }
    }

    /// Total number of snap points registered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns `true` when there are no snap points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Clear all snap points.
    pub fn clear(&mut self) {
        self.points.clear();
    }

    /// Find the nearest enabled snap point within the snap radius.
    ///
    /// Returns `None` when no snap point is close enough.
    #[must_use]
    pub fn nearest(&self, frame: u64) -> Option<&SnapPoint> {
        self.points
            .iter()
            .filter(|p| self.config.is_enabled(p.kind))
            .filter_map(|p| {
                let dist = p.frame.abs_diff(frame);
                if dist <= self.config.snap_radius {
                    Some((dist, p))
                } else {
                    None
                }
            })
            .min_by_key(|(dist, _)| *dist)
            .map(|(_, p)| p)
    }

    /// Snap `frame` to the nearest enabled point, or return `frame` unchanged.
    #[must_use]
    pub fn snap(&self, frame: u64) -> u64 {
        self.nearest(frame).map_or(frame, |p| p.frame)
    }

    /// All snap points of the given kind.
    pub fn by_kind(&self, kind: SnapKind) -> impl Iterator<Item = &SnapPoint> {
        self.points.iter().filter(move |p| p.kind == kind)
    }

    /// Current snap configuration.
    #[must_use]
    pub fn config(&self) -> &SnapConfig {
        &self.config
    }

    /// Update the snap configuration.
    pub fn set_config(&mut self, config: SnapConfig) {
        self.config = config;
    }

    /// Populates snap points from clip boundaries on a timeline.
    ///
    /// Scans all tracks and adds `ClipStart` and `ClipEnd` snap points for
    /// every clip. Existing clip snap points are removed first.
    pub fn populate_from_clips(&mut self, clips: &[(u64, u64, String)]) {
        // Remove existing clip snap points
        self.points
            .retain(|p| !matches!(p.kind, SnapKind::ClipStart | SnapKind::ClipEnd));

        for (start, end, name) in clips {
            self.points.push(SnapPoint::new(
                *start,
                format!("{name} start"),
                SnapKind::ClipStart,
            ));
            self.points.push(SnapPoint::new(
                *end,
                format!("{name} end"),
                SnapKind::ClipEnd,
            ));
        }
    }

    /// Sets the playhead snap position.
    ///
    /// Removes any existing playhead snap point and adds a new one at `frame`.
    pub fn set_playhead(&mut self, frame: u64) {
        self.points.retain(|p| p.kind != SnapKind::Playhead);
        self.points
            .push(SnapPoint::new(frame, "Playhead", SnapKind::Playhead));
    }

    /// Populates snap points from marker positions.
    ///
    /// Existing marker snap points are removed first.
    pub fn populate_from_markers(&mut self, markers: &[(u64, String)]) {
        self.points
            .retain(|p| !matches!(p.kind, SnapKind::Marker | SnapKind::Chapter));

        for (frame, label) in markers {
            self.points
                .push(SnapPoint::new(*frame, label.clone(), SnapKind::Marker));
        }
    }

    /// Magnetic snap: finds the nearest snap point with magnetic strength.
    ///
    /// Unlike `snap()`, magnetic snap uses a strength value (0.0-1.0) that
    /// scales the effective snap radius. Additionally, structural snap points
    /// (clip edges, chapters) have higher priority over non-structural ones
    /// (markers, beat grid) when equidistant.
    ///
    /// Returns `Some((snapped_frame, snap_point))` or `None`.
    #[must_use]
    pub fn magnetic_snap(&self, frame: u64, strength: f32) -> Option<(u64, &SnapPoint)> {
        let strength = strength.clamp(0.0, 1.0);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let effective_radius = (self.config.snap_radius as f32 * strength).round() as u64;
        if effective_radius == 0 {
            return None;
        }

        self.points
            .iter()
            .filter(|p| self.config.is_enabled(p.kind))
            .filter_map(|p| {
                let dist = p.frame.abs_diff(frame);
                if dist <= effective_radius {
                    // Priority: structural points get a bonus (lower distance)
                    let priority_dist = if p.kind.is_structural() {
                        // Structural points are preferred: reduce effective distance
                        dist.saturating_sub(1)
                    } else {
                        dist
                    };
                    Some((priority_dist, dist, p))
                } else {
                    None
                }
            })
            .min_by_key(|(priority_dist, actual_dist, _)| (*priority_dist, *actual_dist))
            .map(|(_, _, p)| (p.frame, p))
    }

    /// Snaps a range (e.g., a clip being dragged) to the nearest snap point.
    ///
    /// Both the start and end of the range are tested against snap points.
    /// Returns the offset that should be applied to the range start to
    /// achieve the snap, or 0 if no snap was found.
    #[must_use]
    pub fn snap_range(&self, range_start: u64, range_end: u64) -> i64 {
        let start_snap = self.nearest(range_start);
        let end_snap = self.nearest(range_end);

        match (start_snap, end_snap) {
            (Some(sp), Some(ep)) => {
                let start_dist = sp.frame.abs_diff(range_start);
                let end_dist = ep.frame.abs_diff(range_end);
                if start_dist <= end_dist {
                    sp.frame as i64 - range_start as i64
                } else {
                    ep.frame as i64 - range_end as i64
                }
            }
            (Some(sp), None) => sp.frame as i64 - range_start as i64,
            (None, Some(ep)) => ep.frame as i64 - range_end as i64,
            (None, None) => 0,
        }
    }

    /// Returns all snap points within the given frame range.
    #[must_use]
    pub fn points_in_range(&self, start: u64, end: u64) -> Vec<&SnapPoint> {
        self.points
            .iter()
            .filter(|p| p.frame >= start && p.frame <= end && self.config.is_enabled(p.kind))
            .collect()
    }

    /// Returns all snap points, regardless of configuration.
    #[must_use]
    pub fn all_points(&self) -> &[SnapPoint] {
        &self.points
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid() -> SnapGrid {
        let mut g = SnapGrid::new();
        g.add(SnapPoint::new(0, "Clip start", SnapKind::ClipStart));
        g.add(SnapPoint::new(100, "Clip end", SnapKind::ClipEnd));
        g.add(SnapPoint::new(50, "Marker", SnapKind::Marker));
        g
    }

    #[test]
    fn test_snap_to_exact() {
        let g = make_grid();
        assert_eq!(g.snap(100), 100);
    }

    #[test]
    fn test_snap_within_radius() {
        let g = make_grid();
        // frame 3 is within default radius (5) of snap point 0
        assert_eq!(g.snap(3), 0);
    }

    #[test]
    fn test_snap_outside_radius_unchanged() {
        let g = make_grid();
        // frame 20 is far from all points (nearest is 0, distance 20)
        assert_eq!(g.snap(20), 20);
    }

    #[test]
    fn test_nearest_returns_closest() {
        let g = make_grid();
        // frame 48 is 2 away from 50 (Marker) and 48 away from 0
        let p = g.nearest(48).expect("should succeed in test");
        assert_eq!(p.frame, 50);
    }

    #[test]
    fn test_nearest_none_when_out_of_range() {
        let g = make_grid();
        assert!(g.nearest(30).is_none());
    }

    #[test]
    fn test_disabled_kind_not_snapped() {
        let config = SnapConfig {
            snap_to_markers: false,
            snap_radius: 10,
            ..SnapConfig::default()
        };
        let mut g = SnapGrid::with_config(config);
        g.add(SnapPoint::new(50, "Marker", SnapKind::Marker));
        // Marker is disabled, so no snap
        assert_eq!(g.snap(50), 50);
    }

    #[test]
    fn test_beat_grid_added() {
        let mut g = SnapGrid::new();
        g.add_beat_grid(0, 24, 4);
        assert_eq!(g.len(), 4);
        let frames: Vec<u64> = g.by_kind(SnapKind::BeatGrid).map(|p| p.frame).collect();
        assert_eq!(frames, vec![0, 24, 48, 72]);
    }

    #[test]
    fn test_clear() {
        let mut g = make_grid();
        g.clear();
        assert!(g.is_empty());
    }

    #[test]
    fn test_len() {
        let g = make_grid();
        assert_eq!(g.len(), 3);
    }

    #[test]
    fn test_by_kind_filter() {
        let g = make_grid();
        let clips: Vec<_> = g.by_kind(SnapKind::ClipStart).collect();
        assert_eq!(clips.len(), 1);
        assert_eq!(clips[0].frame, 0);
    }

    #[test]
    fn test_snap_kind_is_structural() {
        assert!(SnapKind::ClipStart.is_structural());
        assert!(SnapKind::ClipEnd.is_structural());
        assert!(SnapKind::Chapter.is_structural());
        assert!(!SnapKind::Marker.is_structural());
        assert!(!SnapKind::BeatGrid.is_structural());
    }

    #[test]
    fn test_config_is_enabled() {
        let c = SnapConfig::default();
        assert!(c.is_enabled(SnapKind::ClipStart));
        assert!(!c.is_enabled(SnapKind::BeatGrid));
    }

    #[test]
    fn test_set_config() {
        let mut g = SnapGrid::new();
        let cfg = SnapConfig {
            snap_radius: 20,
            ..SnapConfig::default()
        };
        g.set_config(cfg);
        assert_eq!(g.config().snap_radius, 20);
    }

    // --- Magnetic snap tests ---

    #[test]
    fn test_populate_from_clips() {
        let mut g = SnapGrid::new();
        g.populate_from_clips(&[
            (0, 100, "Clip1".to_string()),
            (200, 350, "Clip2".to_string()),
        ]);
        let clip_starts: Vec<u64> = g.by_kind(SnapKind::ClipStart).map(|p| p.frame).collect();
        assert_eq!(clip_starts, vec![0, 200]);
        let clip_ends: Vec<u64> = g.by_kind(SnapKind::ClipEnd).map(|p| p.frame).collect();
        assert_eq!(clip_ends, vec![100, 350]);
    }

    #[test]
    fn test_populate_from_clips_replaces_existing() {
        let mut g = SnapGrid::new();
        g.populate_from_clips(&[(0, 100, "Old".to_string())]);
        assert_eq!(g.by_kind(SnapKind::ClipStart).count(), 1);

        g.populate_from_clips(&[(50, 150, "New".to_string())]);
        // Old clip points should be replaced
        let starts: Vec<u64> = g.by_kind(SnapKind::ClipStart).map(|p| p.frame).collect();
        assert_eq!(starts, vec![50]);
    }

    #[test]
    fn test_set_playhead() {
        let mut g = SnapGrid::new();
        g.set_playhead(100);
        let playheads: Vec<u64> = g.by_kind(SnapKind::Playhead).map(|p| p.frame).collect();
        assert_eq!(playheads, vec![100]);

        // Setting again should replace, not add
        g.set_playhead(200);
        let playheads: Vec<u64> = g.by_kind(SnapKind::Playhead).map(|p| p.frame).collect();
        assert_eq!(playheads, vec![200]);
    }

    #[test]
    fn test_populate_from_markers() {
        let mut g = SnapGrid::new();
        g.populate_from_markers(&[(50, "Intro".to_string()), (200, "Chorus".to_string())]);
        let markers: Vec<u64> = g.by_kind(SnapKind::Marker).map(|p| p.frame).collect();
        assert_eq!(markers, vec![50, 200]);
    }

    #[test]
    fn test_magnetic_snap_full_strength() {
        let mut g = SnapGrid::new();
        g.add(SnapPoint::new(100, "Clip end", SnapKind::ClipEnd));

        // Full strength (1.0): radius = 5
        let result = g.magnetic_snap(103, 1.0);
        assert!(result.is_some());
        let (frame, point) = result.expect("should snap");
        assert_eq!(frame, 100);
        assert_eq!(point.kind, SnapKind::ClipEnd);
    }

    #[test]
    fn test_magnetic_snap_zero_strength() {
        let mut g = SnapGrid::new();
        g.add(SnapPoint::new(100, "Clip end", SnapKind::ClipEnd));

        // Zero strength: no snap
        let result = g.magnetic_snap(103, 0.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_magnetic_snap_half_strength() {
        let config = SnapConfig {
            snap_radius: 10,
            ..SnapConfig::default()
        };
        let mut g = SnapGrid::with_config(config);
        g.add(SnapPoint::new(100, "Clip end", SnapKind::ClipEnd));

        // Half strength: effective radius = 5
        // Distance 4: should snap
        let result = g.magnetic_snap(104, 0.5);
        assert!(result.is_some());

        // Distance 6: should not snap (6 > 5)
        let result = g.magnetic_snap(106, 0.5);
        assert!(result.is_none());
    }

    #[test]
    fn test_magnetic_snap_structural_priority() {
        let config = SnapConfig {
            snap_radius: 10,
            ..SnapConfig::default()
        };
        let mut g = SnapGrid::with_config(config);
        // Marker at 102 and ClipEnd at 101, query at 103
        // Both at similar distance (1 vs 2), but ClipEnd is structural
        // so gets priority_dist = 1 (actual 2 - 1 bonus), vs marker priority_dist = 1
        // When priority distances tie, actual distance breaks tie: ClipEnd at dist 2 vs Marker at dist 1
        // So to demonstrate structural priority, we need same actual distance
        g.add(SnapPoint::new(101, "Clip end", SnapKind::ClipEnd));
        g.add(SnapPoint::new(103, "Marker", SnapKind::Marker));

        // Query at 102: ClipEnd at dist 1 (priority 0), Marker at dist 1 (priority 1)
        let result = g.magnetic_snap(102, 1.0);
        let (frame, point) = result.expect("should snap");
        // Structural point at 101 gets priority bonus: effective dist = 0
        // vs marker at 103 with effective dist = 1
        assert_eq!(frame, 101);
        assert_eq!(point.kind, SnapKind::ClipEnd);
    }

    #[test]
    fn test_snap_range_start_wins() {
        let config = SnapConfig {
            snap_radius: 5,
            ..SnapConfig::default()
        };
        let mut g = SnapGrid::with_config(config);
        g.add(SnapPoint::new(100, "Snap", SnapKind::ClipEnd));

        // Range [98, 200]: start is closer to snap point 100 (dist 2)
        let offset = g.snap_range(98, 200);
        assert_eq!(offset, 2); // Should shift right by 2
    }

    #[test]
    fn test_snap_range_end_wins() {
        let config = SnapConfig {
            snap_radius: 5,
            ..SnapConfig::default()
        };
        let mut g = SnapGrid::with_config(config);
        g.add(SnapPoint::new(200, "Snap", SnapKind::ClipEnd));

        // Range [0, 198]: end is closer to snap point 200 (dist 2)
        let offset = g.snap_range(0, 198);
        assert_eq!(offset, 2); // Should shift right by 2
    }

    #[test]
    fn test_snap_range_no_snap() {
        let g = SnapGrid::new();
        let offset = g.snap_range(100, 200);
        assert_eq!(offset, 0);
    }

    #[test]
    fn test_points_in_range() {
        let g = make_grid();
        let points = g.points_in_range(0, 60);
        assert_eq!(points.len(), 2); // ClipStart at 0, Marker at 50
    }

    #[test]
    fn test_points_in_range_empty() {
        let g = make_grid();
        let points = g.points_in_range(200, 300);
        assert!(points.is_empty());
    }

    #[test]
    fn test_all_points() {
        let g = make_grid();
        assert_eq!(g.all_points().len(), 3);
    }

    #[test]
    fn test_playhead_snap_integration() {
        let config = SnapConfig {
            snap_to_playhead: true,
            snap_radius: 5,
            ..SnapConfig::default()
        };
        let mut g = SnapGrid::with_config(config);
        g.set_playhead(100);

        // Near playhead: should snap
        assert_eq!(g.snap(102), 100);
        // Far from playhead: should not snap
        assert_eq!(g.snap(110), 110);
    }

    #[test]
    fn test_clip_boundaries_and_markers_combined() {
        let config = SnapConfig {
            snap_radius: 5,
            snap_to_clips: true,
            snap_to_markers: true,
            ..SnapConfig::default()
        };
        let mut g = SnapGrid::with_config(config);
        g.populate_from_clips(&[
            (0, 100, "Clip1".to_string()),
            (100, 200, "Clip2".to_string()),
        ]);
        g.populate_from_markers(&[(150, "Marker".to_string())]);
        g.set_playhead(75);

        // Near clip boundary at 100
        assert_eq!(g.snap(98), 100);
        // Near marker at 150
        assert_eq!(g.snap(148), 150);
        // Near playhead at 75
        assert_eq!(g.snap(73), 75);
    }
}
