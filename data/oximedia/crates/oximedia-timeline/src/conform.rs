//! Timeline conform and conforming operations.
//!
//! Conform operations re-timestamp or resample clips to match a target
//! sequence format (frame rate, resolution).

/// Settings that describe the target format for a conform operation.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ConformSettings {
    /// Target frames per second.
    pub target_fps: f64,
    /// Target frame width in pixels.
    pub target_width: u32,
    /// Target frame height in pixels.
    pub target_height: u32,
    /// Whether to retimestamp clips after fps conform.
    pub retimestamp: bool,
}

impl ConformSettings {
    /// Create new conform settings.
    #[must_use]
    pub fn new(target_fps: f64, target_width: u32, target_height: u32) -> Self {
        Self {
            target_fps,
            target_width,
            target_height,
            retimestamp: true,
        }
    }

    /// Disable retimestamping.
    #[must_use]
    pub fn without_retimestamp(mut self) -> Self {
        self.retimestamp = false;
        self
    }
}

/// The type of change recorded during a conform operation.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum ConformChangeType {
    /// A speed/time-remap was applied to match the target frame rate.
    SpeedChange,
    /// The clip resolution was scaled to match the target.
    ResolutionChange,
    /// The frame rate of the clip was changed.
    FpsChange,
    /// A timecode offset was applied.
    TimecodeOffset,
}

/// A record of a single change applied during conform.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ConformChange {
    /// ID of the clip that was changed.
    pub clip_id: u64,
    /// The kind of change.
    pub change_type: ConformChangeType,
    /// String representation of the old value.
    pub old_value: String,
    /// String representation of the new value.
    pub new_value: String,
}

impl ConformChange {
    /// Create a new change record.
    #[must_use]
    pub fn new(
        clip_id: u64,
        change_type: ConformChangeType,
        old_value: impl Into<String>,
        new_value: impl Into<String>,
    ) -> Self {
        Self {
            clip_id,
            change_type,
            old_value: old_value.into(),
            new_value: new_value.into(),
        }
    }
}

/// Lightweight description of a clip used for conform calculations.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub struct ClipInfo {
    /// Clip identifier.
    pub id: u64,
    /// Native frame rate of the clip's source media.
    pub fps: f64,
    /// Native width of the clip's source media.
    pub width: u32,
    /// Native height of the clip's source media.
    pub height: u32,
    /// Start frame on the timeline.
    pub start_frame: u64,
    /// End frame on the timeline (exclusive).
    pub end_frame: u64,
}

impl ClipInfo {
    /// Create a new clip info record.
    #[must_use]
    pub fn new(
        id: u64,
        fps: f64,
        width: u32,
        height: u32,
        start_frame: u64,
        end_frame: u64,
    ) -> Self {
        Self {
            id,
            fps,
            width,
            height,
            start_frame,
            end_frame: end_frame.max(start_frame),
        }
    }

    /// Duration in frames.
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame)
    }

    /// Returns `true` if the clip's fps matches `target`.
    #[must_use]
    pub fn matches_fps(&self, target: f64) -> bool {
        (self.fps - target).abs() < 0.01
    }

    /// Returns `true` if the clip's resolution matches the target.
    #[must_use]
    pub fn matches_resolution(&self, width: u32, height: u32) -> bool {
        self.width == width && self.height == height
    }
}

/// Compute the set of conform changes required to bring `clips` into
/// compliance with `settings`.
#[must_use]
pub fn compute_conform_changes(
    clips: &[ClipInfo],
    settings: &ConformSettings,
) -> Vec<ConformChange> {
    let mut changes = Vec::new();

    for clip in clips {
        // Check frame rate.
        if !clip.matches_fps(settings.target_fps) {
            changes.push(ConformChange::new(
                clip.id,
                ConformChangeType::FpsChange,
                format!("{:.4}", clip.fps),
                format!("{:.4}", settings.target_fps),
            ));

            if settings.retimestamp {
                // Compute the speed ratio as old_value.
                let speed = clip.fps / settings.target_fps;
                changes.push(ConformChange::new(
                    clip.id,
                    ConformChangeType::SpeedChange,
                    "1.0000",
                    format!("{speed:.4}"),
                ));
            }
        }

        // Check resolution.
        if !clip.matches_resolution(settings.target_width, settings.target_height) {
            changes.push(ConformChange::new(
                clip.id,
                ConformChangeType::ResolutionChange,
                format!("{}x{}", clip.width, clip.height),
                format!("{}x{}", settings.target_width, settings.target_height),
            ));
        }
    }

    changes
}

/// Apply fps conform to a single clip, returning a new `ClipInfo` with the
/// start/end frames rescaled to the target frame rate.
#[must_use]
pub fn apply_fps_conform(clip: &ClipInfo, target_fps: f64) -> ClipInfo {
    if target_fps <= 0.0 || clip.fps <= 0.0 {
        return clip.clone();
    }

    let ratio = target_fps / clip.fps;
    let new_start = (clip.start_frame as f64 * ratio).round() as u64;
    let new_end = (clip.end_frame as f64 * ratio).round() as u64;

    ClipInfo {
        id: clip.id,
        fps: target_fps,
        width: clip.width,
        height: clip.height,
        start_frame: new_start,
        end_frame: new_end.max(new_start),
    }
}

/// Apply conform settings to a mutable list of clips in-place.
///
/// Returns all conform changes that were applied.
pub fn batch_conform(clips: &mut Vec<ClipInfo>, settings: &ConformSettings) -> Vec<ConformChange> {
    let changes = compute_conform_changes(clips, settings);

    for clip in clips.iter_mut() {
        // Apply fps conform.
        if !clip.matches_fps(settings.target_fps) {
            *clip = apply_fps_conform(clip, settings.target_fps);
        }
        // Apply resolution conform.
        if !clip.matches_resolution(settings.target_width, settings.target_height) {
            clip.width = settings.target_width;
            clip.height = settings.target_height;
        }
    }

    changes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip_24fps() -> ClipInfo {
        ClipInfo::new(1, 24.0, 1920, 1080, 0, 240)
    }

    fn clip_30fps() -> ClipInfo {
        ClipInfo::new(2, 30.0, 1280, 720, 0, 300)
    }

    fn settings_25fps_1080p() -> ConformSettings {
        ConformSettings::new(25.0, 1920, 1080)
    }

    // --- ClipInfo tests ---

    #[test]
    fn test_clip_info_duration() {
        assert_eq!(clip_24fps().duration_frames(), 240);
    }

    #[test]
    fn test_clip_matches_fps() {
        assert!(clip_24fps().matches_fps(24.0));
        assert!(!clip_24fps().matches_fps(25.0));
    }

    #[test]
    fn test_clip_matches_resolution() {
        assert!(clip_24fps().matches_resolution(1920, 1080));
        assert!(!clip_24fps().matches_resolution(1280, 720));
    }

    // --- compute_conform_changes tests ---

    #[test]
    fn test_no_changes_when_already_conformed() {
        let clips = vec![ClipInfo::new(1, 25.0, 1920, 1080, 0, 250)];
        let settings = ConformSettings::new(25.0, 1920, 1080);
        let changes = compute_conform_changes(&clips, &settings);
        assert!(changes.is_empty());
    }

    #[test]
    fn test_fps_change_detected() {
        let clips = vec![clip_24fps()];
        let settings = settings_25fps_1080p();
        let changes = compute_conform_changes(&clips, &settings);
        assert!(
            changes
                .iter()
                .any(|c| c.change_type == ConformChangeType::FpsChange),
            "Expected FpsChange"
        );
    }

    #[test]
    fn test_resolution_change_detected() {
        let clips = vec![clip_30fps()]; // 1280x720 vs target 1920x1080
        let settings = settings_25fps_1080p();
        let changes = compute_conform_changes(&clips, &settings);
        assert!(
            changes
                .iter()
                .any(|c| c.change_type == ConformChangeType::ResolutionChange),
            "Expected ResolutionChange"
        );
    }

    #[test]
    fn test_speed_change_included_with_retimestamp() {
        let clips = vec![clip_24fps()];
        let settings = settings_25fps_1080p(); // retimestamp = true by default
        let changes = compute_conform_changes(&clips, &settings);
        assert!(
            changes
                .iter()
                .any(|c| c.change_type == ConformChangeType::SpeedChange),
            "Expected SpeedChange when retimestamp is enabled"
        );
    }

    #[test]
    fn test_no_speed_change_without_retimestamp() {
        let clips = vec![clip_24fps()];
        let settings = ConformSettings::new(25.0, 1920, 1080).without_retimestamp();
        let changes = compute_conform_changes(&clips, &settings);
        assert!(
            !changes
                .iter()
                .any(|c| c.change_type == ConformChangeType::SpeedChange),
            "No SpeedChange expected when retimestamp is disabled"
        );
    }

    // --- apply_fps_conform tests ---

    #[test]
    fn test_fps_conform_rescales_frames() {
        let clip = ClipInfo::new(1, 24.0, 1920, 1080, 0, 240);
        let conformed = apply_fps_conform(&clip, 48.0);
        assert_eq!(conformed.fps as u32, 48);
        assert_eq!(conformed.end_frame, 480); // doubled
    }

    #[test]
    fn test_fps_conform_noop_for_same_rate() {
        let clip = clip_24fps();
        let conformed = apply_fps_conform(&clip, 24.0);
        assert_eq!(conformed.start_frame, clip.start_frame);
        assert_eq!(conformed.end_frame, clip.end_frame);
    }

    #[test]
    fn test_fps_conform_zero_target_returns_clone() {
        let clip = clip_24fps();
        let conformed = apply_fps_conform(&clip, 0.0);
        assert_eq!(conformed, clip);
    }

    // --- batch_conform tests ---

    #[test]
    fn test_batch_conform_updates_clips_in_place() {
        let mut clips = vec![clip_30fps()];
        let settings = ConformSettings::new(25.0, 1920, 1080);
        batch_conform(&mut clips, &settings);
        assert!((clips[0].fps - 25.0).abs() < 0.01);
        assert_eq!(clips[0].width, 1920);
        assert_eq!(clips[0].height, 1080);
    }

    #[test]
    fn test_batch_conform_returns_changes() {
        let mut clips = vec![clip_30fps()];
        let settings = settings_25fps_1080p();
        let changes = batch_conform(&mut clips, &settings);
        assert!(!changes.is_empty());
    }

    #[test]
    fn test_conform_change_values() {
        let change = ConformChange::new(1, ConformChangeType::FpsChange, "24.0000", "25.0000");
        assert_eq!(change.old_value, "24.0000");
        assert_eq!(change.new_value, "25.0000");
    }
}
