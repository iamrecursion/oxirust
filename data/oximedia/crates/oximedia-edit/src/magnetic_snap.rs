//! Magnetic timeline snapping for clip alignment.
//!
//! Provides magnetic snap points that attract clip edges during drag
//! operations. Snap targets include other clip edges, the playhead,
//! markers, and regular grid intervals (e.g., beat-aligned snapping).

use crate::clip::ClipId;
use crate::timeline::Timeline;

/// The type of snap target that caused a snap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapTargetType {
    /// Snap to the start or end of another clip.
    ClipEdge,
    /// Snap to the playhead position.
    Playhead,
    /// Snap to a marker.
    Marker,
    /// Snap to a grid line (e.g., beat, frame boundary).
    Grid,
    /// Snap to in/out point.
    InOutPoint,
}

/// A snap target with its position and metadata.
#[derive(Debug, Clone)]
pub struct SnapTarget {
    /// Timeline position of this snap target.
    pub position: i64,
    /// Type of snap target.
    pub target_type: SnapTargetType,
    /// Optional label (e.g., marker name, clip name).
    pub label: Option<String>,
    /// Snap strength multiplier (1.0 = normal, higher = stronger attraction).
    pub strength: f64,
}

impl SnapTarget {
    /// Create a new snap target.
    #[must_use]
    pub fn new(position: i64, target_type: SnapTargetType) -> Self {
        Self {
            position,
            target_type,
            label: None,
            strength: 1.0,
        }
    }

    /// Create a snap target with a label.
    #[must_use]
    pub fn with_label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }

    /// Create a snap target with custom strength.
    #[must_use]
    pub fn with_strength(mut self, strength: f64) -> Self {
        self.strength = strength.max(0.0);
        self
    }
}

/// Result of a snap operation.
#[derive(Debug, Clone)]
pub struct SnapResult {
    /// The snapped position (may differ from the input if snapping occurred).
    pub position: i64,
    /// Whether snapping actually occurred.
    pub snapped: bool,
    /// The snap target that caused the snap (if any).
    pub target: Option<SnapTarget>,
    /// The distance from the original position to the snap target.
    pub distance: i64,
}

impl SnapResult {
    /// Create a result indicating no snap occurred.
    #[must_use]
    pub fn no_snap(position: i64) -> Self {
        Self {
            position,
            snapped: false,
            target: None,
            distance: 0,
        }
    }

    /// Create a result indicating a snap occurred.
    #[must_use]
    pub fn snapped_to(original: i64, target: SnapTarget) -> Self {
        let distance = (original - target.position).abs();
        Self {
            position: target.position,
            snapped: true,
            distance,
            target: Some(target),
        }
    }
}

/// Configuration for magnetic snapping behaviour.
#[derive(Debug, Clone)]
pub struct MagneticSnapConfig {
    /// Whether snapping is enabled.
    pub enabled: bool,
    /// Snap threshold in timebase units (how close the cursor must be).
    pub threshold: i64,
    /// Snap to other clip edges.
    pub snap_to_clips: bool,
    /// Snap to the playhead.
    pub snap_to_playhead: bool,
    /// Snap to markers.
    pub snap_to_markers: bool,
    /// Snap to grid lines.
    pub snap_to_grid: bool,
    /// Snap to in/out points.
    pub snap_to_in_out: bool,
    /// Grid interval in timebase units (0 = disabled).
    pub grid_interval: i64,
    /// Clips to exclude from snap targets (e.g., the clip being dragged).
    pub excluded_clips: Vec<ClipId>,
}

impl Default for MagneticSnapConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 10,
            snap_to_clips: true,
            snap_to_playhead: true,
            snap_to_markers: true,
            snap_to_grid: false,
            snap_to_in_out: true,
            grid_interval: 0,
            excluded_clips: Vec::new(),
        }
    }
}

impl MagneticSnapConfig {
    /// Create a config with all snap types enabled.
    #[must_use]
    pub fn all_enabled(threshold: i64) -> Self {
        Self {
            enabled: true,
            threshold,
            snap_to_clips: true,
            snap_to_playhead: true,
            snap_to_markers: true,
            snap_to_grid: true,
            snap_to_in_out: true,
            grid_interval: 33, // ~30fps frame
            excluded_clips: Vec::new(),
        }
    }

    /// Create a config with snapping disabled.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }
}

/// Magnetic snap engine that computes snap targets and performs snapping.
#[derive(Debug)]
pub struct MagneticSnapEngine {
    /// Snap configuration.
    pub config: MagneticSnapConfig,
}

impl MagneticSnapEngine {
    /// Create a new snap engine.
    #[must_use]
    pub fn new(config: MagneticSnapConfig) -> Self {
        Self { config }
    }

    /// Collect all snap targets from the timeline.
    #[must_use]
    pub fn collect_targets(&self, timeline: &Timeline) -> Vec<SnapTarget> {
        let mut targets = Vec::new();

        if !self.config.enabled {
            return targets;
        }

        // Clip edges
        if self.config.snap_to_clips {
            for track in &timeline.tracks {
                for clip in &track.clips {
                    if self.config.excluded_clips.contains(&clip.id) {
                        continue;
                    }
                    targets.push(
                        SnapTarget::new(clip.timeline_start, SnapTargetType::ClipEdge)
                            .with_label(format!("clip_{}_start", clip.id)),
                    );
                    targets.push(
                        SnapTarget::new(clip.timeline_end(), SnapTargetType::ClipEdge)
                            .with_label(format!("clip_{}_end", clip.id)),
                    );
                }
            }
        }

        // Playhead
        if self.config.snap_to_playhead {
            targets.push(
                SnapTarget::new(timeline.playhead, SnapTargetType::Playhead).with_strength(1.5), // playhead is slightly stronger
            );
        }

        // Markers
        if self.config.snap_to_markers {
            for marker in timeline.markers.all() {
                targets.push(
                    SnapTarget::new(marker.position, SnapTargetType::Marker)
                        .with_label(marker.name.clone()),
                );
            }
        }

        // In/Out points
        if self.config.snap_to_in_out {
            if let Some(in_pos) = timeline.in_out.in_point {
                targets.push(SnapTarget::new(in_pos, SnapTargetType::InOutPoint));
            }
            if let Some(out_pos) = timeline.in_out.out_point {
                targets.push(SnapTarget::new(out_pos, SnapTargetType::InOutPoint));
            }
        }

        // Grid
        if self.config.snap_to_grid && self.config.grid_interval > 0 {
            // Generate grid points around the timeline
            let interval = self.config.grid_interval;
            let max_pos = timeline.duration + interval;
            let mut pos = 0i64;
            while pos <= max_pos {
                targets.push(SnapTarget::new(pos, SnapTargetType::Grid));
                pos += interval;
            }
        }

        targets
    }

    /// Find the best snap target for a given position.
    #[must_use]
    pub fn snap(&self, position: i64, targets: &[SnapTarget]) -> SnapResult {
        if !self.config.enabled || targets.is_empty() {
            return SnapResult::no_snap(position);
        }

        let mut best_target: Option<&SnapTarget> = None;
        let mut best_distance = i64::MAX;

        for target in targets {
            let distance = (position - target.position).abs();
            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let effective_threshold = (self.config.threshold as f64 * target.strength) as i64;

            if distance <= effective_threshold && distance < best_distance {
                best_distance = distance;
                best_target = Some(target);
            }
        }

        match best_target {
            Some(target) => SnapResult::snapped_to(position, target.clone()),
            None => SnapResult::no_snap(position),
        }
    }

    /// Convenience: collect targets from timeline and snap in one call.
    #[must_use]
    pub fn snap_on_timeline(&self, position: i64, timeline: &Timeline) -> SnapResult {
        let targets = self.collect_targets(timeline);
        self.snap(position, &targets)
    }

    /// Snap both the start and end of a clip-sized region, returning
    /// the best overall snap.
    #[must_use]
    pub fn snap_clip_region(
        &self,
        start: i64,
        duration: i64,
        targets: &[SnapTarget],
    ) -> SnapResult {
        if !self.config.enabled {
            return SnapResult::no_snap(start);
        }

        let snap_start = self.snap(start, targets);
        let snap_end = self.snap(start + duration, targets);

        // Pick the closer snap
        if snap_start.snapped && snap_end.snapped {
            if snap_start.distance <= snap_end.distance {
                snap_start
            } else {
                // Adjust so the entire clip shifts
                let adjusted_start = snap_end.position - duration;
                SnapResult {
                    position: adjusted_start,
                    snapped: true,
                    target: snap_end.target,
                    distance: snap_end.distance,
                }
            }
        } else if snap_start.snapped {
            snap_start
        } else if snap_end.snapped {
            let adjusted_start = snap_end.position - duration;
            SnapResult {
                position: adjusted_start,
                snapped: true,
                target: snap_end.target,
                distance: snap_end.distance,
            }
        } else {
            SnapResult::no_snap(start)
        }
    }

    /// Collect targets from all tracks (multi-track snap).
    ///
    /// Unlike `collect_targets` which already does multi-track, this method
    /// collects targets with per-track metadata for cross-track alignment.
    #[must_use]
    pub fn collect_multitrack_targets(&self, timeline: &Timeline) -> Vec<MultiTrackSnapTarget> {
        let mut targets = Vec::new();
        if !self.config.enabled || !self.config.snap_to_clips {
            return targets;
        }

        for (track_idx, track) in timeline.tracks.iter().enumerate() {
            for clip in &track.clips {
                if self.config.excluded_clips.contains(&clip.id) {
                    continue;
                }
                targets.push(MultiTrackSnapTarget {
                    position: clip.timeline_start,
                    track_index: track_idx,
                    clip_id: clip.id,
                    edge: SnapEdge::Start,
                });
                targets.push(MultiTrackSnapTarget {
                    position: clip.timeline_end(),
                    track_index: track_idx,
                    clip_id: clip.id,
                    edge: SnapEdge::End,
                });
            }
        }
        targets
    }

    /// Snap across tracks: find the best alignment across all tracks.
    #[must_use]
    pub fn snap_multitrack(
        &self,
        position: i64,
        source_track: usize,
        targets: &[MultiTrackSnapTarget],
    ) -> SnapResult {
        if !self.config.enabled || targets.is_empty() {
            return SnapResult::no_snap(position);
        }

        let mut best_target: Option<&MultiTrackSnapTarget> = None;
        let mut best_distance = i64::MAX;

        for target in targets {
            // Skip targets on the same track if we only want cross-track snapping.
            if target.track_index == source_track {
                continue;
            }
            let distance = (position - target.position).abs();
            if distance <= self.config.threshold && distance < best_distance {
                best_distance = distance;
                best_target = Some(target);
            }
        }

        match best_target {
            Some(t) => {
                let snap_target = SnapTarget::new(t.position, SnapTargetType::ClipEdge)
                    .with_label(format!("track{}_{:?}_{}", t.track_index, t.edge, t.clip_id));
                SnapResult::snapped_to(position, snap_target)
            }
            None => SnapResult::no_snap(position),
        }
    }
}

/// Edge type for multi-track snap targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapEdge {
    /// Start edge of a clip.
    Start,
    /// End edge of a clip.
    End,
}

/// A snap target with track information for cross-track alignment.
#[derive(Debug, Clone)]
pub struct MultiTrackSnapTarget {
    /// Timeline position.
    pub position: i64,
    /// Track index this target belongs to.
    pub track_index: usize,
    /// Clip ID this target belongs to.
    pub clip_id: ClipId,
    /// Which edge (start or end).
    pub edge: SnapEdge,
}

/// A beat position for beat-based snapping.
#[derive(Debug, Clone)]
pub struct BeatPosition {
    /// Timeline position of this beat.
    pub position: i64,
    /// Beat number (0-indexed within a bar).
    pub beat_number: u32,
    /// Bar number (0-indexed).
    pub bar_number: u32,
    /// Whether this is a downbeat (first beat of bar).
    pub is_downbeat: bool,
}

/// Beat grid for audio-synchronized snapping.
#[derive(Debug, Clone)]
pub struct BeatGrid {
    /// BPM (beats per minute).
    pub bpm: f64,
    /// Beats per bar (time signature numerator).
    pub beats_per_bar: u32,
    /// Timebase units per second (for position calculation).
    pub timebase_rate: i64,
    /// Offset of the first beat from timeline position 0.
    pub offset: i64,
    /// Cached beat positions.
    beats: Vec<BeatPosition>,
}

impl BeatGrid {
    /// Create a new beat grid.
    ///
    /// # Arguments
    /// - `bpm`: beats per minute
    /// - `beats_per_bar`: time signature numerator (e.g. 4 for 4/4)
    /// - `timebase_rate`: timebase units per second (e.g. 1000 for ms)
    /// - `offset`: offset of first beat in timebase units
    #[must_use]
    pub fn new(bpm: f64, beats_per_bar: u32, timebase_rate: i64, offset: i64) -> Self {
        Self {
            bpm: bpm.max(1.0),
            beats_per_bar: beats_per_bar.max(1),
            timebase_rate: timebase_rate.max(1),
            offset,
            beats: Vec::new(),
        }
    }

    /// Compute beat interval in timebase units.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn beat_interval(&self) -> i64 {
        ((60.0 / self.bpm) * self.timebase_rate as f64) as i64
    }

    /// Generate beat positions up to a given timeline duration.
    pub fn generate_beats(&mut self, duration: i64) {
        self.beats.clear();
        let interval = self.beat_interval();
        if interval <= 0 {
            return;
        }

        let mut pos = self.offset;
        let mut beat_num = 0u32;
        let mut bar_num = 0u32;

        while pos <= duration {
            if pos >= 0 {
                self.beats.push(BeatPosition {
                    position: pos,
                    beat_number: beat_num,
                    bar_number: bar_num,
                    is_downbeat: beat_num == 0,
                });
            }
            beat_num += 1;
            if beat_num >= self.beats_per_bar {
                beat_num = 0;
                bar_num += 1;
            }
            pos += interval;
        }
    }

    /// Get all beat positions.
    #[must_use]
    pub fn beats(&self) -> &[BeatPosition] {
        &self.beats
    }

    /// Convert beats to snap targets with downbeats getting stronger attraction.
    #[must_use]
    pub fn to_snap_targets(&self) -> Vec<SnapTarget> {
        self.beats
            .iter()
            .map(|beat| {
                let strength = if beat.is_downbeat { 1.5 } else { 1.0 };
                SnapTarget::new(beat.position, SnapTargetType::Grid)
                    .with_label(format!("bar{}:beat{}", beat.bar_number, beat.beat_number))
                    .with_strength(strength)
            })
            .collect()
    }

    /// Beat count.
    #[must_use]
    pub fn beat_count(&self) -> usize {
        self.beats.len()
    }
}

/// A named timeline marker for marker-based snapping.
#[derive(Debug, Clone)]
pub struct SnapMarker {
    /// Position on the timeline.
    pub position: i64,
    /// Marker name.
    pub name: String,
    /// Color label for UI.
    pub color: [u8; 3],
    /// Extra snap strength for this marker.
    pub strength: f64,
}

impl SnapMarker {
    /// Create a new snap marker.
    #[must_use]
    pub fn new(position: i64, name: String) -> Self {
        Self {
            position,
            name,
            color: [255, 200, 0],
            strength: 1.0,
        }
    }

    /// Set the color.
    #[must_use]
    pub fn with_color(mut self, color: [u8; 3]) -> Self {
        self.color = color;
        self
    }

    /// Set custom strength.
    #[must_use]
    pub fn with_strength(mut self, strength: f64) -> Self {
        self.strength = strength.max(0.0);
        self
    }

    /// Convert to a `SnapTarget`.
    #[must_use]
    pub fn to_snap_target(&self) -> SnapTarget {
        SnapTarget::new(self.position, SnapTargetType::Marker)
            .with_label(self.name.clone())
            .with_strength(self.strength)
    }
}

/// Snap strength zone: stronger snap force near exact alignment.
#[derive(Debug, Clone)]
pub struct SnapStrengthZone {
    /// Inner radius (full strength within this distance).
    pub inner_radius: i64,
    /// Outer radius (linearly decreasing strength to this distance).
    pub outer_radius: i64,
    /// Maximum strength multiplier at inner radius.
    pub max_strength: f64,
    /// Minimum strength multiplier at outer radius.
    pub min_strength: f64,
}

impl SnapStrengthZone {
    /// Create a new strength zone.
    #[must_use]
    pub fn new(inner_radius: i64, outer_radius: i64) -> Self {
        Self {
            inner_radius: inner_radius.max(0),
            outer_radius: outer_radius.max(inner_radius.max(0) + 1),
            max_strength: 3.0,
            min_strength: 0.5,
        }
    }

    /// Compute the effective strength for a given distance from the snap target.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn strength_at_distance(&self, distance: i64) -> f64 {
        let dist = distance.abs();
        if dist <= self.inner_radius {
            self.max_strength
        } else if dist >= self.outer_radius {
            0.0
        } else {
            // Linear interpolation between inner and outer.
            let range = (self.outer_radius - self.inner_radius) as f64;
            let t = (dist - self.inner_radius) as f64 / range;
            self.max_strength + t * (self.min_strength - self.max_strength)
        }
    }
}

impl Default for SnapStrengthZone {
    fn default() -> Self {
        Self::new(3, 15)
    }
}

/// A gap between two clips on a track.
#[derive(Debug, Clone)]
pub struct TimelineGap {
    /// Start position of the gap.
    pub start: i64,
    /// End position of the gap.
    pub end: i64,
    /// Track index.
    pub track_index: usize,
    /// Duration of the gap.
    pub duration: i64,
}

impl TimelineGap {
    /// Create a new gap.
    #[must_use]
    pub fn new(start: i64, end: i64, track_index: usize) -> Self {
        Self {
            start,
            end,
            track_index,
            duration: (end - start).max(0),
        }
    }

    /// Create snap targets for filling this gap (snap to start and end).
    #[must_use]
    pub fn to_snap_targets(&self) -> Vec<SnapTarget> {
        vec![
            SnapTarget::new(self.start, SnapTargetType::ClipEdge)
                .with_label(format!("gap_start_t{}", self.track_index))
                .with_strength(1.2),
            SnapTarget::new(self.end, SnapTargetType::ClipEdge)
                .with_label(format!("gap_end_t{}", self.track_index))
                .with_strength(1.2),
        ]
    }
}

/// Detects gaps between clips on timeline tracks.
#[must_use]
pub fn detect_gaps(timeline: &Timeline) -> Vec<TimelineGap> {
    let mut gaps = Vec::new();
    for (track_idx, track) in timeline.tracks.iter().enumerate() {
        let mut sorted_clips: Vec<&crate::clip::Clip> = track.clips.iter().collect();
        sorted_clips.sort_by_key(|c| c.timeline_start);

        let mut prev_end = 0i64;
        for clip in &sorted_clips {
            if clip.timeline_start > prev_end {
                gaps.push(TimelineGap::new(prev_end, clip.timeline_start, track_idx));
            }
            let clip_end = clip.timeline_end();
            if clip_end > prev_end {
                prev_end = clip_end;
            }
        }
    }
    gaps
}

/// Find the best gap that a clip of given duration could fill exactly.
#[must_use]
pub fn find_fitting_gap(gaps: &[TimelineGap], clip_duration: i64) -> Option<&TimelineGap> {
    gaps.iter()
        .filter(|g| g.duration >= clip_duration)
        .min_by_key(|g| (g.duration - clip_duration).abs())
}

/// Extended snap engine that combines all snap capabilities.
pub struct ExtendedSnapEngine {
    /// Base snap engine.
    pub engine: MagneticSnapEngine,
    /// Beat grid (optional).
    pub beat_grid: Option<BeatGrid>,
    /// Custom markers.
    pub markers: Vec<SnapMarker>,
    /// Strength zone config.
    pub strength_zone: SnapStrengthZone,
    /// Whether multi-track snap is enabled.
    pub multitrack_enabled: bool,
    /// Whether gap detection snap is enabled.
    pub gap_snap_enabled: bool,
}

impl ExtendedSnapEngine {
    /// Create a new extended snap engine.
    #[must_use]
    pub fn new(config: MagneticSnapConfig) -> Self {
        Self {
            engine: MagneticSnapEngine::new(config),
            beat_grid: None,
            markers: Vec::new(),
            strength_zone: SnapStrengthZone::default(),
            multitrack_enabled: true,
            gap_snap_enabled: true,
        }
    }

    /// Add a beat grid.
    pub fn set_beat_grid(&mut self, grid: BeatGrid) {
        self.beat_grid = Some(grid);
    }

    /// Add a snap marker.
    pub fn add_marker(&mut self, marker: SnapMarker) {
        self.markers.push(marker);
    }

    /// Collect all targets including beats, markers, gaps, and standard targets.
    #[must_use]
    pub fn collect_all_targets(&self, timeline: &Timeline) -> Vec<SnapTarget> {
        let mut targets = self.engine.collect_targets(timeline);

        // Add beat targets.
        if let Some(ref grid) = self.beat_grid {
            targets.extend(grid.to_snap_targets());
        }

        // Add custom markers.
        for marker in &self.markers {
            targets.push(marker.to_snap_target());
        }

        // Add gap targets.
        if self.gap_snap_enabled {
            let gaps = detect_gaps(timeline);
            for gap in &gaps {
                targets.extend(gap.to_snap_targets());
            }
        }

        targets
    }

    /// Snap with strength zones applied.
    #[must_use]
    pub fn snap_with_zones(&self, position: i64, targets: &[SnapTarget]) -> SnapResult {
        if !self.engine.config.enabled || targets.is_empty() {
            return SnapResult::no_snap(position);
        }

        let mut best_target: Option<&SnapTarget> = None;
        let mut best_score = f64::MIN;

        for target in targets {
            let distance = (position - target.position).abs();
            let zone_strength = self.strength_zone.strength_at_distance(distance);
            if zone_strength <= 0.0 {
                continue;
            }

            // Score: higher is better (closer + stronger)
            #[allow(clippy::cast_precision_loss)]
            let score = zone_strength * target.strength - (distance as f64 * 0.01);

            if score > best_score {
                best_score = score;
                best_target = Some(target);
            }
        }

        match best_target {
            Some(target) => SnapResult::snapped_to(position, target.clone()),
            None => SnapResult::no_snap(position),
        }
    }

    /// Full snap: collect all targets and snap with zones.
    #[must_use]
    pub fn snap_full(&self, position: i64, timeline: &Timeline) -> SnapResult {
        let targets = self.collect_all_targets(timeline);
        self.snap_with_zones(position, &targets)
    }

    /// Number of custom markers.
    #[must_use]
    pub fn marker_count(&self) -> usize {
        self.markers.len()
    }
}

impl Default for MagneticSnapEngine {
    fn default() -> Self {
        Self::new(MagneticSnapConfig::default())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{Clip, ClipType};
    use crate::timeline::{Timeline, TrackType};
    use oximedia_core::Rational;

    fn make_timeline() -> Timeline {
        let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
        let vt = tl.add_track(TrackType::Video);
        let c1 = Clip::new(0, ClipType::Video, 0, 5000);
        let c2 = Clip::new(0, ClipType::Video, 5000, 3000);
        let _ = tl.add_clip(vt, c1);
        let _ = tl.add_clip(vt, c2);
        tl.set_playhead(2500);
        tl
    }

    #[test]
    fn test_snap_target_creation() {
        let t = SnapTarget::new(1000, SnapTargetType::ClipEdge)
            .with_label("test".to_string())
            .with_strength(2.0);
        assert_eq!(t.position, 1000);
        assert_eq!(t.target_type, SnapTargetType::ClipEdge);
        assert_eq!(t.label.as_deref(), Some("test"));
        assert!((t.strength - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_snap_result_no_snap() {
        let r = SnapResult::no_snap(100);
        assert_eq!(r.position, 100);
        assert!(!r.snapped);
    }

    #[test]
    fn test_snap_result_snapped() {
        let target = SnapTarget::new(100, SnapTargetType::Playhead);
        let r = SnapResult::snapped_to(105, target);
        assert_eq!(r.position, 100);
        assert!(r.snapped);
        assert_eq!(r.distance, 5);
    }

    #[test]
    fn test_disabled_engine_returns_no_snap() {
        let engine = MagneticSnapEngine::new(MagneticSnapConfig::disabled());
        let targets = vec![SnapTarget::new(100, SnapTargetType::ClipEdge)];
        let r = engine.snap(100, &targets);
        assert!(!r.snapped);
    }

    #[test]
    fn test_snap_within_threshold() {
        let config = MagneticSnapConfig {
            enabled: true,
            threshold: 10,
            ..Default::default()
        };
        let engine = MagneticSnapEngine::new(config);
        let targets = vec![SnapTarget::new(100, SnapTargetType::ClipEdge)];

        // Within threshold
        let r = engine.snap(105, &targets);
        assert!(r.snapped);
        assert_eq!(r.position, 100);

        // Outside threshold
        let r2 = engine.snap(120, &targets);
        assert!(!r2.snapped);
    }

    #[test]
    fn test_snap_picks_closest() {
        let config = MagneticSnapConfig {
            enabled: true,
            threshold: 20,
            ..Default::default()
        };
        let engine = MagneticSnapEngine::new(config);
        let targets = vec![
            SnapTarget::new(100, SnapTargetType::ClipEdge),
            SnapTarget::new(200, SnapTargetType::ClipEdge),
        ];

        let r = engine.snap(108, &targets);
        assert!(r.snapped);
        assert_eq!(r.position, 100);

        let r2 = engine.snap(195, &targets);
        assert!(r2.snapped);
        assert_eq!(r2.position, 200);
    }

    #[test]
    fn test_snap_strength_multiplier() {
        let config = MagneticSnapConfig {
            enabled: true,
            threshold: 5,
            ..Default::default()
        };
        let engine = MagneticSnapEngine::new(config);
        // Strong target at 100, threshold 5 * strength 3.0 = 15 effective
        let targets = vec![SnapTarget::new(100, SnapTargetType::Playhead).with_strength(3.0)];

        let r = engine.snap(112, &targets);
        assert!(r.snapped, "should snap within effective threshold of 15");
    }

    #[test]
    fn test_collect_targets_from_timeline() {
        let tl = make_timeline();
        let config = MagneticSnapConfig {
            enabled: true,
            snap_to_clips: true,
            snap_to_playhead: true,
            snap_to_markers: false,
            snap_to_grid: false,
            snap_to_in_out: false,
            ..Default::default()
        };
        let engine = MagneticSnapEngine::new(config);
        let targets = engine.collect_targets(&tl);

        // 2 clips * 2 edges + 1 playhead = 5
        assert_eq!(targets.len(), 5);
    }

    #[test]
    fn test_collect_targets_excludes_clips() {
        let tl = make_timeline();
        let config = MagneticSnapConfig {
            enabled: true,
            snap_to_clips: true,
            snap_to_playhead: false,
            snap_to_markers: false,
            snap_to_grid: false,
            snap_to_in_out: false,
            excluded_clips: vec![1], // exclude first clip
            ..Default::default()
        };
        let engine = MagneticSnapEngine::new(config);
        let targets = engine.collect_targets(&tl);

        // Only second clip (2 edges)
        assert_eq!(targets.len(), 2);
    }

    #[test]
    fn test_snap_on_timeline() {
        let tl = make_timeline();
        let config = MagneticSnapConfig {
            enabled: true,
            threshold: 50,
            snap_to_clips: true,
            snap_to_playhead: false,
            snap_to_markers: false,
            snap_to_grid: false,
            snap_to_in_out: false,
            ..Default::default()
        };
        let engine = MagneticSnapEngine::new(config);
        let r = engine.snap_on_timeline(4990, &tl);
        assert!(r.snapped);
        assert_eq!(r.position, 5000);
    }

    #[test]
    fn test_snap_clip_region() {
        let config = MagneticSnapConfig {
            enabled: true,
            threshold: 10,
            ..Default::default()
        };
        let engine = MagneticSnapEngine::new(config);
        let targets = vec![
            SnapTarget::new(0, SnapTargetType::ClipEdge),
            SnapTarget::new(5000, SnapTargetType::ClipEdge),
        ];

        // Clip start near 0
        let r = engine.snap_clip_region(3, 2000, &targets);
        assert!(r.snapped);
        assert_eq!(r.position, 0);

        // Clip end near 5000 (start=2998, dur=2000, end=4998)
        let r2 = engine.snap_clip_region(2998, 2000, &targets);
        assert!(r2.snapped);
        // Should snap end to 5000, so start = 5000 - 2000 = 3000
        assert_eq!(r2.position, 3000);
    }

    #[test]
    fn test_grid_snapping() {
        let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
        let vt = tl.add_track(TrackType::Video);
        // Add a clip to give the timeline some duration
        let c = Clip::new(0, ClipType::Video, 0, 10000);
        let _ = tl.add_clip(vt, c);

        let config = MagneticSnapConfig {
            enabled: true,
            threshold: 5,
            snap_to_clips: false,
            snap_to_playhead: false,
            snap_to_markers: false,
            snap_to_grid: true,
            snap_to_in_out: false,
            grid_interval: 100,
            ..Default::default()
        };
        let engine = MagneticSnapEngine::new(config);
        let targets = engine.collect_targets(&tl);

        let r = engine.snap(302, &targets);
        assert!(r.snapped);
        assert_eq!(r.position, 300);
    }

    #[test]
    fn test_default_config() {
        let config = MagneticSnapConfig::default();
        assert!(config.enabled);
        assert_eq!(config.threshold, 10);
        assert!(config.snap_to_clips);
        assert!(config.snap_to_playhead);
    }

    #[test]
    fn test_all_enabled_config() {
        let config = MagneticSnapConfig::all_enabled(20);
        assert!(config.enabled);
        assert_eq!(config.threshold, 20);
        assert!(config.snap_to_grid);
        assert_eq!(config.grid_interval, 33);
    }

    // ── Multi-track snap tests ─────────────────────────────────────────

    #[test]
    fn test_multitrack_target_collection() {
        let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
        let vt = tl.add_track(TrackType::Video);
        let at = tl.add_track(TrackType::Audio);
        let _ = tl.add_clip(vt, Clip::new(0, ClipType::Video, 0, 5000));
        let _ = tl.add_clip(at, Clip::new(0, ClipType::Audio, 1000, 3000));

        let engine = MagneticSnapEngine::new(MagneticSnapConfig::all_enabled(20));
        let targets = engine.collect_multitrack_targets(&tl);
        // 2 clips * 2 edges = 4
        assert_eq!(targets.len(), 4);
        // Verify track indices are correct
        let track0_count = targets.iter().filter(|t| t.track_index == 0).count();
        let track1_count = targets.iter().filter(|t| t.track_index == 1).count();
        assert_eq!(track0_count, 2);
        assert_eq!(track1_count, 2);
    }

    #[test]
    fn test_snap_multitrack_cross_track() {
        let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
        let vt = tl.add_track(TrackType::Video);
        let at = tl.add_track(TrackType::Audio);
        let _ = tl.add_clip(vt, Clip::new(0, ClipType::Video, 0, 5000));
        let _ = tl.add_clip(at, Clip::new(0, ClipType::Audio, 5000, 3000));

        let engine = MagneticSnapEngine::new(MagneticSnapConfig {
            enabled: true,
            threshold: 20,
            ..Default::default()
        });
        let targets = engine.collect_multitrack_targets(&tl);

        // Snap from track 0, position 4990 → should find audio clip start at 5000 on track 1
        let r = engine.snap_multitrack(4990, 0, &targets);
        assert!(r.snapped);
        assert_eq!(r.position, 5000);
    }

    #[test]
    fn test_snap_multitrack_ignores_same_track() {
        let config = MagneticSnapConfig {
            enabled: true,
            threshold: 20,
            ..Default::default()
        };
        let engine = MagneticSnapEngine::new(config);
        let targets = vec![MultiTrackSnapTarget {
            position: 100,
            track_index: 0,
            clip_id: 1,
            edge: SnapEdge::Start,
        }];
        // Same track as source → no cross-track snap
        let r = engine.snap_multitrack(105, 0, &targets);
        assert!(!r.snapped);
    }

    // ── Beat grid tests ────────────────────────────────────────────────

    #[test]
    fn test_beat_grid_interval() {
        let grid = BeatGrid::new(120.0, 4, 1000, 0);
        // 120 BPM = 500ms per beat
        assert_eq!(grid.beat_interval(), 500);
    }

    #[test]
    fn test_beat_grid_generate() {
        let mut grid = BeatGrid::new(120.0, 4, 1000, 0);
        grid.generate_beats(2500);
        // Beats at 0, 500, 1000, 1500, 2000, 2500 = 6
        assert_eq!(grid.beat_count(), 6);
        // First beat should be downbeat
        assert!(grid.beats()[0].is_downbeat);
        // Beat 1 is not downbeat
        assert!(!grid.beats()[1].is_downbeat);
        // Beat 4 (bar 1, beat 0) is downbeat
        assert!(grid.beats()[4].is_downbeat);
    }

    #[test]
    fn test_beat_grid_to_snap_targets() {
        let mut grid = BeatGrid::new(120.0, 4, 1000, 0);
        grid.generate_beats(1000);
        let targets = grid.to_snap_targets();
        assert!(!targets.is_empty());
        // Downbeat targets should have strength 1.5
        assert!((targets[0].strength - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_beat_grid_with_offset() {
        let mut grid = BeatGrid::new(120.0, 4, 1000, 200);
        grid.generate_beats(1200);
        let beats = grid.beats();
        assert!(!beats.is_empty());
        assert_eq!(beats[0].position, 200);
    }

    // ── Snap marker tests ──────────────────────────────────────────────

    #[test]
    fn test_snap_marker_creation() {
        let m = SnapMarker::new(5000, "chapter1".to_string())
            .with_color([255, 0, 0])
            .with_strength(2.0);
        assert_eq!(m.position, 5000);
        assert_eq!(m.color, [255, 0, 0]);
        let target = m.to_snap_target();
        assert_eq!(target.position, 5000);
        assert_eq!(target.target_type, SnapTargetType::Marker);
        assert!((target.strength - 2.0).abs() < 1e-9);
    }

    // ── Strength zone tests ────────────────────────────────────────────

    #[test]
    fn test_strength_zone_at_center() {
        let zone = SnapStrengthZone::new(5, 20);
        // At center: max strength
        assert!((zone.strength_at_distance(0) - zone.max_strength).abs() < 1e-9);
        assert!((zone.strength_at_distance(3) - zone.max_strength).abs() < 1e-9);
    }

    #[test]
    fn test_strength_zone_at_boundary() {
        let zone = SnapStrengthZone::new(5, 20);
        // Beyond outer: 0
        assert!((zone.strength_at_distance(25)).abs() < 1e-9);
    }

    #[test]
    fn test_strength_zone_interpolation() {
        let zone = SnapStrengthZone::new(0, 10);
        let mid = zone.strength_at_distance(5);
        assert!(mid > zone.min_strength);
        assert!(mid < zone.max_strength);
    }

    // ── Gap detection tests ────────────────────────────────────────────

    #[test]
    fn test_detect_gaps() {
        let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
        let vt = tl.add_track(TrackType::Video);
        let _ = tl.add_clip(vt, Clip::new(0, ClipType::Video, 1000, 2000));
        let _ = tl.add_clip(vt, Clip::new(0, ClipType::Video, 5000, 1000));

        let gaps = detect_gaps(&tl);
        // Gap from 0–1000 and 3000–5000
        assert_eq!(gaps.len(), 2);
        assert_eq!(gaps[0].start, 0);
        assert_eq!(gaps[0].end, 1000);
        assert_eq!(gaps[1].start, 3000);
        assert_eq!(gaps[1].end, 5000);
    }

    #[test]
    fn test_detect_gaps_no_gaps() {
        let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
        let vt = tl.add_track(TrackType::Video);
        let _ = tl.add_clip(vt, Clip::new(0, ClipType::Video, 0, 5000));
        let _ = tl.add_clip(vt, Clip::new(0, ClipType::Video, 5000, 3000));

        let gaps = detect_gaps(&tl);
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_find_fitting_gap() {
        let gaps = vec![
            TimelineGap::new(0, 1000, 0),
            TimelineGap::new(3000, 5000, 0),
        ];
        // Find gap that fits 1000 duration
        let best = find_fitting_gap(&gaps, 1000);
        assert!(best.is_some());
        assert_eq!(best.map(|g| g.start), Some(0));

        // Find gap that fits 1500 → picks 2000 gap
        let best = find_fitting_gap(&gaps, 1500);
        assert!(best.is_some());
        assert_eq!(best.map(|g| g.duration), Some(2000));
    }

    #[test]
    fn test_gap_to_snap_targets() {
        let gap = TimelineGap::new(100, 500, 0);
        let targets = gap.to_snap_targets();
        assert_eq!(targets.len(), 2);
        assert_eq!(targets[0].position, 100);
        assert_eq!(targets[1].position, 500);
    }

    // ── Extended engine tests ──────────────────────────────────────────

    #[test]
    fn test_extended_engine_with_beats() {
        let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
        let vt = tl.add_track(TrackType::Video);
        let _ = tl.add_clip(vt, Clip::new(0, ClipType::Video, 0, 10000));

        let mut ext = ExtendedSnapEngine::new(MagneticSnapConfig {
            enabled: true,
            threshold: 15,
            snap_to_clips: false,
            snap_to_playhead: false,
            snap_to_markers: false,
            snap_to_grid: false,
            snap_to_in_out: false,
            ..Default::default()
        });
        let mut grid = BeatGrid::new(120.0, 4, 1000, 0);
        grid.generate_beats(10000);
        ext.set_beat_grid(grid);

        let targets = ext.collect_all_targets(&tl);
        assert!(!targets.is_empty());
    }

    #[test]
    fn test_extended_engine_snap_with_zones() {
        let ext = ExtendedSnapEngine::new(MagneticSnapConfig {
            enabled: true,
            threshold: 20,
            ..Default::default()
        });
        let targets = vec![
            SnapTarget::new(100, SnapTargetType::ClipEdge),
            SnapTarget::new(200, SnapTargetType::ClipEdge),
        ];
        let r = ext.snap_with_zones(102, &targets);
        assert!(r.snapped);
        assert_eq!(r.position, 100);
    }

    #[test]
    fn test_extended_engine_markers() {
        let mut ext = ExtendedSnapEngine::new(MagneticSnapConfig::all_enabled(20));
        ext.add_marker(SnapMarker::new(5000, "ch1".to_string()));
        ext.add_marker(SnapMarker::new(10000, "ch2".to_string()));
        assert_eq!(ext.marker_count(), 2);
    }

    #[test]
    fn test_snap_edge_enum() {
        assert_ne!(SnapEdge::Start, SnapEdge::End);
        let _s = format!("{:?}", SnapEdge::Start);
    }
}
