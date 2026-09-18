#![allow(dead_code)]
//! Shot grouping and clustering engine.
//!
//! Groups shots into logical clusters based on visual similarity,
//! temporal proximity, narrative structure, or custom rules.
//! Supports hierarchical grouping, automatic scene segmentation,
//! and tag-based organization.

use std::collections::HashMap;

/// The strategy used to group shots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupingStrategy {
    /// Group consecutive shots by visual similarity.
    VisualSimilarity,
    /// Group shots by temporal proximity (close in time).
    TemporalProximity,
    /// Group by shot type label.
    ShotTypeLabel,
    /// Group by camera angle label.
    CameraAngleLabel,
    /// Group by user-supplied tags.
    TagBased,
}

/// Metadata for a shot to be grouped.
#[derive(Debug, Clone)]
pub struct ShotInfo {
    /// Unique shot identifier.
    pub id: u64,
    /// Shot type label (e.g. "CU", "MS", "LS").
    pub shot_type: String,
    /// Camera angle label.
    pub camera_angle: String,
    /// Start time in frames.
    pub start_frame: u64,
    /// End time in frames.
    pub end_frame: u64,
    /// Average luminance (0.0..1.0).
    pub avg_luminance: f64,
    /// Average color (R, G, B, each 0.0..1.0).
    pub avg_color: (f64, f64, f64),
    /// User-assigned tags.
    pub tags: Vec<String>,
}

impl ShotInfo {
    /// Create a new shot info.
    pub fn new(id: u64, shot_type: &str, start_frame: u64, end_frame: u64) -> Self {
        Self {
            id,
            shot_type: shot_type.to_string(),
            camera_angle: String::new(),
            start_frame,
            end_frame,
            avg_luminance: 0.5,
            avg_color: (0.5, 0.5, 0.5),
            tags: Vec::new(),
        }
    }

    /// Set camera angle.
    pub fn with_angle(mut self, angle: &str) -> Self {
        self.camera_angle = angle.to_string();
        self
    }

    /// Set average luminance.
    pub fn with_luminance(mut self, lum: f64) -> Self {
        self.avg_luminance = lum;
        self
    }

    /// Set average color.
    pub fn with_color(mut self, r: f64, g: f64, b: f64) -> Self {
        self.avg_color = (r, g, b);
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    /// Duration in frames.
    pub fn duration(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame)
    }
}

/// A group of related shots.
#[derive(Debug, Clone)]
pub struct ShotGroup {
    /// Group identifier.
    pub group_id: u64,
    /// Group label or name.
    pub label: String,
    /// Shot IDs belonging to this group.
    pub shot_ids: Vec<u64>,
    /// Strategy that produced this group.
    pub strategy: GroupingStrategy,
    /// Average luminance across group shots.
    pub avg_luminance: f64,
    /// Total duration in frames.
    pub total_duration: u64,
}

impl ShotGroup {
    /// Create a new empty group.
    pub fn new(group_id: u64, label: &str, strategy: GroupingStrategy) -> Self {
        Self {
            group_id,
            label: label.to_string(),
            shot_ids: Vec::new(),
            strategy,
            avg_luminance: 0.0,
            total_duration: 0,
        }
    }

    /// Add a shot to this group.
    pub fn add_shot(&mut self, shot_id: u64) {
        self.shot_ids.push(shot_id);
    }

    /// Return the number of shots.
    pub fn len(&self) -> usize {
        self.shot_ids.len()
    }

    /// Check if group is empty.
    pub fn is_empty(&self) -> bool {
        self.shot_ids.is_empty()
    }
}

/// Result of a grouping operation.
#[derive(Debug, Clone)]
pub struct GroupingResult {
    /// All groups produced.
    pub groups: Vec<ShotGroup>,
    /// Number of ungrouped shots (if any).
    pub ungrouped_count: usize,
    /// Strategy used.
    pub strategy: GroupingStrategy,
}

/// Shot grouping engine.
#[derive(Debug, Clone)]
pub struct ShotGrouper {
    /// Active grouping strategy.
    strategy: GroupingStrategy,
    /// Temporal gap threshold (in frames) for temporal proximity grouping.
    temporal_gap_threshold: u64,
    /// Visual similarity threshold for visual grouping.
    visual_similarity_threshold: f64,
}

impl ShotGrouper {
    /// Create a new shot grouper.
    pub fn new(strategy: GroupingStrategy) -> Self {
        Self {
            strategy,
            temporal_gap_threshold: 30,
            visual_similarity_threshold: 0.7,
        }
    }

    /// Set the temporal gap threshold.
    pub fn with_temporal_gap(mut self, frames: u64) -> Self {
        self.temporal_gap_threshold = frames;
        self
    }

    /// Set the visual similarity threshold.
    pub fn with_visual_threshold(mut self, threshold: f64) -> Self {
        self.visual_similarity_threshold = threshold;
        self
    }

    /// Group the given shots.
    #[allow(clippy::cast_precision_loss)]
    pub fn group(&self, shots: &[ShotInfo]) -> GroupingResult {
        if shots.is_empty() {
            return GroupingResult {
                groups: Vec::new(),
                ungrouped_count: 0,
                strategy: self.strategy,
            };
        }

        let groups = match self.strategy {
            GroupingStrategy::ShotTypeLabel => self.group_by_label(shots, |s| s.shot_type.clone()),
            GroupingStrategy::CameraAngleLabel => {
                self.group_by_label(shots, |s| s.camera_angle.clone())
            }
            GroupingStrategy::TagBased => self.group_by_tags(shots),
            GroupingStrategy::TemporalProximity => self.group_temporal(shots),
            GroupingStrategy::VisualSimilarity => self.group_visual(shots),
        };

        let grouped_count: usize = groups.iter().map(|g| g.len()).sum();
        let ungrouped_count = shots.len().saturating_sub(grouped_count);

        GroupingResult {
            groups,
            ungrouped_count,
            strategy: self.strategy,
        }
    }

    /// Group by a label-extracting function.
    #[allow(clippy::cast_precision_loss)]
    fn group_by_label<F>(&self, shots: &[ShotInfo], label_fn: F) -> Vec<ShotGroup>
    where
        F: Fn(&ShotInfo) -> String,
    {
        let mut map: HashMap<String, Vec<&ShotInfo>> = HashMap::new();
        for shot in shots {
            let label = label_fn(shot);
            map.entry(label).or_default().push(shot);
        }

        let mut groups = Vec::new();
        for (idx, (label, members)) in map.into_iter().enumerate() {
            let mut group = ShotGroup::new(idx as u64, &label, self.strategy);
            let mut lum_sum = 0.0_f64;
            let mut dur_sum = 0_u64;
            for shot in &members {
                group.add_shot(shot.id);
                lum_sum += shot.avg_luminance;
                dur_sum += shot.duration();
            }
            if !members.is_empty() {
                group.avg_luminance = lum_sum / members.len() as f64;
            }
            group.total_duration = dur_sum;
            groups.push(group);
        }

        groups.sort_by_key(|g| g.group_id);
        groups
    }

    /// Group by user-supplied tags (each tag creates a group).
    #[allow(clippy::cast_precision_loss)]
    fn group_by_tags(&self, shots: &[ShotInfo]) -> Vec<ShotGroup> {
        let mut tag_map: HashMap<String, Vec<&ShotInfo>> = HashMap::new();
        for shot in shots {
            for tag in &shot.tags {
                tag_map.entry(tag.clone()).or_default().push(shot);
            }
        }

        let mut groups = Vec::new();
        for (idx, (tag, members)) in tag_map.into_iter().enumerate() {
            let mut group = ShotGroup::new(idx as u64, &tag, self.strategy);
            let mut dur_sum = 0_u64;
            let mut lum_sum = 0.0_f64;
            for shot in &members {
                group.add_shot(shot.id);
                lum_sum += shot.avg_luminance;
                dur_sum += shot.duration();
            }
            if !members.is_empty() {
                group.avg_luminance = lum_sum / members.len() as f64;
            }
            group.total_duration = dur_sum;
            groups.push(group);
        }
        groups
    }

    /// Group by temporal proximity: consecutive shots within gap threshold.
    #[allow(clippy::cast_precision_loss)]
    fn group_temporal(&self, shots: &[ShotInfo]) -> Vec<ShotGroup> {
        let mut sorted: Vec<&ShotInfo> = shots.iter().collect();
        sorted.sort_by_key(|s| s.start_frame);

        let mut groups = Vec::new();
        let mut current_group =
            ShotGroup::new(0, "temporal_0", GroupingStrategy::TemporalProximity);
        let mut last_end = 0_u64;
        let mut lum_sum = 0.0_f64;
        let mut member_count = 0_usize;
        let mut group_idx = 0_u64;

        for (i, shot) in sorted.iter().enumerate() {
            if i > 0 && shot.start_frame.saturating_sub(last_end) > self.temporal_gap_threshold {
                // Finalize current group
                if member_count > 0 {
                    current_group.avg_luminance = lum_sum / member_count as f64;
                }
                groups.push(current_group);
                group_idx += 1;
                let label = format!("temporal_{group_idx}");
                current_group =
                    ShotGroup::new(group_idx, &label, GroupingStrategy::TemporalProximity);
                lum_sum = 0.0;
                member_count = 0;
            }

            current_group.add_shot(shot.id);
            current_group.total_duration += shot.duration();
            lum_sum += shot.avg_luminance;
            member_count += 1;
            last_end = shot.end_frame;
        }

        // Finalize last group
        if !current_group.is_empty() {
            if member_count > 0 {
                current_group.avg_luminance = lum_sum / member_count as f64;
            }
            groups.push(current_group);
        }

        groups
    }

    /// Group by visual similarity using simple color distance.
    #[allow(clippy::cast_precision_loss)]
    fn group_visual(&self, shots: &[ShotInfo]) -> Vec<ShotGroup> {
        let mut assigned = vec![false; shots.len()];
        let mut groups = Vec::new();
        let mut group_idx = 0_u64;

        for i in 0..shots.len() {
            if assigned[i] {
                continue;
            }
            assigned[i] = true;
            let label = format!("visual_{group_idx}");
            let mut group = ShotGroup::new(group_idx, &label, GroupingStrategy::VisualSimilarity);
            group.add_shot(shots[i].id);
            let mut lum_sum = shots[i].avg_luminance;
            let mut dur_sum = shots[i].duration();
            let mut count = 1_usize;

            for j in (i + 1)..shots.len() {
                if assigned[j] {
                    continue;
                }
                let dist = self.color_distance(&shots[i], &shots[j]);
                if dist < (1.0 - self.visual_similarity_threshold) {
                    assigned[j] = true;
                    group.add_shot(shots[j].id);
                    lum_sum += shots[j].avg_luminance;
                    dur_sum += shots[j].duration();
                    count += 1;
                }
            }

            group.avg_luminance = lum_sum / count as f64;
            group.total_duration = dur_sum;
            groups.push(group);
            group_idx += 1;
        }

        groups
    }

    /// Euclidean distance in RGB color space (normalized to 0..1).
    fn color_distance(&self, a: &ShotInfo, b: &ShotInfo) -> f64 {
        let dr = a.avg_color.0 - b.avg_color.0;
        let dg = a.avg_color.1 - b.avg_color.1;
        let db = a.avg_color.2 - b.avg_color.2;
        ((dr * dr + dg * dg + db * db) / 3.0).sqrt()
    }
}

impl Default for ShotGrouper {
    fn default() -> Self {
        Self::new(GroupingStrategy::ShotTypeLabel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_shot(id: u64, stype: &str, start: u64, end: u64) -> ShotInfo {
        ShotInfo::new(id, stype, start, end)
    }

    #[test]
    fn test_group_by_shot_type() {
        let grouper = ShotGrouper::new(GroupingStrategy::ShotTypeLabel);
        let shots = vec![
            make_shot(1, "CU", 0, 30),
            make_shot(2, "LS", 30, 90),
            make_shot(3, "CU", 90, 120),
        ];
        let result = grouper.group(&shots);
        assert_eq!(result.groups.len(), 2);
        let cu_group = result
            .groups
            .iter()
            .find(|g| g.label == "CU")
            .expect("should succeed in test");
        assert_eq!(cu_group.len(), 2);
    }

    #[test]
    fn test_group_by_camera_angle() {
        let grouper = ShotGrouper::new(GroupingStrategy::CameraAngleLabel);
        let shots = vec![
            make_shot(1, "CU", 0, 30).with_angle("eye"),
            make_shot(2, "LS", 30, 90).with_angle("high"),
            make_shot(3, "MS", 90, 120).with_angle("eye"),
        ];
        let result = grouper.group(&shots);
        let eye_group = result
            .groups
            .iter()
            .find(|g| g.label == "eye")
            .expect("should succeed in test");
        assert_eq!(eye_group.len(), 2);
    }

    #[test]
    fn test_group_by_tags() {
        let grouper = ShotGrouper::new(GroupingStrategy::TagBased);
        let shots = vec![
            make_shot(1, "CU", 0, 30).with_tag("action"),
            make_shot(2, "LS", 30, 60)
                .with_tag("dialogue")
                .with_tag("action"),
            make_shot(3, "MS", 60, 90).with_tag("dialogue"),
        ];
        let result = grouper.group(&shots);
        let action_group = result
            .groups
            .iter()
            .find(|g| g.label == "action")
            .expect("should succeed in test");
        assert_eq!(action_group.len(), 2);
    }

    #[test]
    fn test_temporal_grouping_consecutive() {
        let grouper = ShotGrouper::new(GroupingStrategy::TemporalProximity).with_temporal_gap(10);
        let shots = vec![
            make_shot(1, "CU", 0, 30),
            make_shot(2, "MS", 35, 60),   // gap=5, within threshold
            make_shot(3, "LS", 100, 130), // gap=40, new group
        ];
        let result = grouper.group(&shots);
        assert_eq!(result.groups.len(), 2);
        assert_eq!(result.groups[0].len(), 2);
        assert_eq!(result.groups[1].len(), 1);
    }

    #[test]
    fn test_temporal_grouping_all_close() {
        let grouper = ShotGrouper::new(GroupingStrategy::TemporalProximity).with_temporal_gap(100);
        let shots = vec![
            make_shot(1, "CU", 0, 30),
            make_shot(2, "MS", 50, 80),
            make_shot(3, "LS", 90, 120),
        ];
        let result = grouper.group(&shots);
        assert_eq!(result.groups.len(), 1);
        assert_eq!(result.groups[0].len(), 3);
    }

    #[test]
    fn test_visual_grouping_similar_colors() {
        let grouper =
            ShotGrouper::new(GroupingStrategy::VisualSimilarity).with_visual_threshold(0.7);
        let shots = vec![
            make_shot(1, "CU", 0, 30).with_color(0.8, 0.1, 0.1),
            make_shot(2, "MS", 30, 60).with_color(0.82, 0.12, 0.09),
            make_shot(3, "LS", 60, 90).with_color(0.1, 0.1, 0.9),
        ];
        let result = grouper.group(&shots);
        // Shots 1 and 2 are similar (red), shot 3 is different (blue)
        assert!(result.groups.len() >= 2);
    }

    #[test]
    fn test_empty_shots() {
        let grouper = ShotGrouper::new(GroupingStrategy::ShotTypeLabel);
        let result = grouper.group(&[]);
        assert!(result.groups.is_empty());
        assert_eq!(result.ungrouped_count, 0);
    }

    #[test]
    fn test_shot_info_duration() {
        let shot = make_shot(1, "CU", 10, 40);
        assert_eq!(shot.duration(), 30);
    }

    #[test]
    fn test_shot_group_operations() {
        let mut group = ShotGroup::new(0, "test", GroupingStrategy::ShotTypeLabel);
        assert!(group.is_empty());
        group.add_shot(1);
        group.add_shot(2);
        assert_eq!(group.len(), 2);
        assert!(!group.is_empty());
    }

    #[test]
    fn test_grouping_result_strategy() {
        let grouper = ShotGrouper::new(GroupingStrategy::TagBased);
        let shots = vec![make_shot(1, "CU", 0, 30).with_tag("test")];
        let result = grouper.group(&shots);
        assert_eq!(result.strategy, GroupingStrategy::TagBased);
    }

    #[test]
    fn test_group_total_duration() {
        let grouper = ShotGrouper::new(GroupingStrategy::ShotTypeLabel);
        let shots = vec![make_shot(1, "CU", 0, 30), make_shot(2, "CU", 30, 60)];
        let result = grouper.group(&shots);
        let cu_group = result
            .groups
            .iter()
            .find(|g| g.label == "CU")
            .expect("should succeed in test");
        assert_eq!(cu_group.total_duration, 60);
    }

    #[test]
    fn test_default_grouper() {
        let grouper = ShotGrouper::default();
        assert_eq!(grouper.strategy, GroupingStrategy::ShotTypeLabel);
    }
}
