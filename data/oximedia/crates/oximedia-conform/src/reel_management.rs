//! Multi-reel project management.
//!
//! Provides types for organising multi-reel film/video projects, validating
//! reel continuity, and detecting gaps/overlaps/naming conflicts.

#![allow(dead_code)]

/// Newtype wrapper for a reel identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ReelId(pub u32);

impl std::fmt::Display for ReelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Reel-{:04}", self.0)
    }
}

/// A clip within a reel.
#[derive(Debug, Clone)]
pub struct ReelClip {
    /// Unique clip identifier (e.g. from the conform session).
    pub clip_id: String,
    /// The reel this clip belongs to.
    pub reel_id: ReelId,
    /// Start position within the reel in frames.
    pub position_frames: u64,
    /// Duration of this clip in frames.
    pub duration_frames: u64,
}

impl ReelClip {
    /// Create a new reel clip.
    #[must_use]
    pub fn new(
        clip_id: impl Into<String>,
        reel_id: ReelId,
        position_frames: u64,
        duration_frames: u64,
    ) -> Self {
        Self {
            clip_id: clip_id.into(),
            reel_id,
            position_frames,
            duration_frames,
        }
    }

    /// Last frame (exclusive) occupied by this clip.
    #[must_use]
    pub fn end_frame(&self) -> u64 {
        self.position_frames + self.duration_frames
    }
}

/// A single reel.
#[derive(Debug, Clone)]
pub struct Reel {
    /// Reel identifier.
    pub id: ReelId,
    /// Human-readable reel name.
    pub name: String,
    /// Total reel duration in frames.
    pub duration_frames: u64,
    /// Nominal frame rate.
    pub fps: f32,
    /// Resolution (width, height) in pixels.
    pub resolution: (u32, u32),
    /// Clips recorded on this reel.
    pub clips: Vec<ReelClip>,
}

impl Reel {
    /// Create a new reel with no clips.
    #[must_use]
    pub fn new(
        id: ReelId,
        name: impl Into<String>,
        duration_frames: u64,
        fps: f32,
        resolution: (u32, u32),
    ) -> Self {
        Self {
            id,
            name: name.into(),
            duration_frames,
            fps,
            resolution,
            clips: Vec::new(),
        }
    }

    /// Reel duration in seconds.
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        if self.fps <= 0.0 {
            return 0.0;
        }
        self.duration_frames as f64 / f64::from(self.fps)
    }

    /// Return the clip that occupies `frame`, or `None` if no clip is present.
    #[must_use]
    pub fn clip_at_frame(&self, frame: u64) -> Option<&ReelClip> {
        self.clips
            .iter()
            .find(|c| frame >= c.position_frames && frame < c.end_frame())
    }

    /// Add a clip to this reel (sorted by position).
    pub fn add_clip(&mut self, clip: ReelClip) {
        self.clips.push(clip);
        self.clips.sort_by_key(|c| c.position_frames);
    }
}

/// A set of reels forming a complete project.
#[derive(Debug)]
pub struct ReelSet {
    /// All reels in this project.
    pub reels: Vec<Reel>,
    /// Project name.
    pub project_name: String,
}

impl ReelSet {
    /// Create an empty reel set.
    #[must_use]
    pub fn new(project_name: impl Into<String>) -> Self {
        Self {
            reels: Vec::new(),
            project_name: project_name.into(),
        }
    }

    /// Total duration across all reels in frames.
    #[must_use]
    pub fn total_duration_frames(&self) -> u64 {
        self.reels.iter().map(|r| r.duration_frames).sum()
    }

    /// Number of reels.
    #[must_use]
    pub fn reel_count(&self) -> usize {
        self.reels.len()
    }

    /// Find a reel by `ReelId`.
    #[must_use]
    pub fn find_reel(&self, id: ReelId) -> Option<&Reel> {
        self.reels.iter().find(|r| r.id == id)
    }

    /// Add a reel.
    pub fn add_reel(&mut self, reel: Reel) {
        self.reels.push(reel);
    }
}

/// A single validation issue on a reel.
#[derive(Debug, Clone)]
pub struct ReelIssue {
    /// Reel this issue belongs to.
    pub reel_id: ReelId,
    /// Type of issue.
    pub issue_type: ReelIssueType,
    /// Human-readable description.
    pub description: String,
}

/// Classification of reel validation issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReelIssueType {
    /// Gap between two consecutive clips (missing coverage).
    Gap,
    /// Two clips overlap in time.
    Overlap,
    /// Two or more reels share the same name.
    NamingConflict,
    /// A clip's duration exceeds the reel duration.
    DurationMismatch,
}

/// Reel set validator.
pub struct ReelValidator;

impl ReelValidator {
    /// Validate a slice of reels and return all detected issues.
    #[must_use]
    pub fn check(reels: &[Reel]) -> Vec<ReelIssue> {
        let mut issues = Vec::new();

        // Check each reel individually
        for reel in reels {
            issues.extend(Self::check_reel(reel));
        }

        // Check for naming conflicts across reels
        issues.extend(Self::check_naming_conflicts(reels));

        issues
    }

    fn check_reel(reel: &Reel) -> Vec<ReelIssue> {
        let mut issues = Vec::new();

        // Sort clips by position (they should already be sorted)
        let mut sorted = reel.clips.clone();
        sorted.sort_by_key(|c| c.position_frames);

        for i in 0..sorted.len() {
            let clip = &sorted[i];

            // Duration mismatch: clip extends past reel end
            if clip.end_frame() > reel.duration_frames {
                issues.push(ReelIssue {
                    reel_id: reel.id,
                    issue_type: ReelIssueType::DurationMismatch,
                    description: format!(
                        "Clip '{}' ends at frame {} but reel duration is {}",
                        clip.clip_id,
                        clip.end_frame(),
                        reel.duration_frames
                    ),
                });
            }

            if i + 1 < sorted.len() {
                let next = &sorted[i + 1];

                // Gap: space between clip end and next clip start
                if next.position_frames > clip.end_frame() {
                    issues.push(ReelIssue {
                        reel_id: reel.id,
                        issue_type: ReelIssueType::Gap,
                        description: format!(
                            "Gap between clips '{}' and '{}': frames {}–{}",
                            clip.clip_id,
                            next.clip_id,
                            clip.end_frame(),
                            next.position_frames
                        ),
                    });
                }

                // Overlap: next clip starts before current clip ends
                if next.position_frames < clip.end_frame() {
                    issues.push(ReelIssue {
                        reel_id: reel.id,
                        issue_type: ReelIssueType::Overlap,
                        description: format!(
                            "Clips '{}' and '{}' overlap at frame {}",
                            clip.clip_id, next.clip_id, next.position_frames
                        ),
                    });
                }
            }
        }

        issues
    }

    fn check_naming_conflicts(reels: &[Reel]) -> Vec<ReelIssue> {
        let mut issues = Vec::new();
        let mut seen: std::collections::HashMap<&str, ReelId> = std::collections::HashMap::new();

        for reel in reels {
            if let Some(&existing_id) = seen.get(reel.name.as_str()) {
                issues.push(ReelIssue {
                    reel_id: reel.id,
                    issue_type: ReelIssueType::NamingConflict,
                    description: format!(
                        "Reel '{}' (id={}) shares name with reel id={}",
                        reel.name, reel.id, existing_id
                    ),
                });
            } else {
                seen.insert(&reel.name, reel.id);
            }
        }

        issues
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_reel(id: u32, name: &str, duration: u64) -> Reel {
        Reel::new(ReelId(id), name, duration, 24.0, (1920, 1080))
    }

    fn make_clip(clip_id: &str, reel_id: u32, pos: u64, dur: u64) -> ReelClip {
        ReelClip::new(clip_id, ReelId(reel_id), pos, dur)
    }

    #[test]
    fn test_reel_id_display() {
        let id = ReelId(3);
        assert_eq!(format!("{id}"), "Reel-0003");
    }

    #[test]
    fn test_reel_duration_secs() {
        let reel = make_reel(1, "A001", 240);
        assert!((reel.duration_secs() - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_clip_end_frame() {
        let clip = make_clip("clip1", 1, 100, 50);
        assert_eq!(clip.end_frame(), 150);
    }

    #[test]
    fn test_clip_at_frame_found() {
        let mut reel = make_reel(1, "A001", 500);
        reel.add_clip(make_clip("c1", 1, 0, 100));
        reel.add_clip(make_clip("c2", 1, 100, 100));
        assert_eq!(
            reel.clip_at_frame(50).map(|c| c.clip_id.as_str()),
            Some("c1")
        );
        assert_eq!(
            reel.clip_at_frame(100).map(|c| c.clip_id.as_str()),
            Some("c2")
        );
    }

    #[test]
    fn test_clip_at_frame_not_found() {
        let reel = make_reel(1, "A001", 500);
        assert!(reel.clip_at_frame(300).is_none());
    }

    #[test]
    fn test_reel_set_total_duration() {
        let mut rs = ReelSet::new("TestProject");
        rs.add_reel(make_reel(1, "A001", 240));
        rs.add_reel(make_reel(2, "A002", 360));
        assert_eq!(rs.total_duration_frames(), 600);
    }

    #[test]
    fn test_reel_set_find_reel() {
        let mut rs = ReelSet::new("Proj");
        rs.add_reel(make_reel(1, "A001", 100));
        assert!(rs.find_reel(ReelId(1)).is_some());
        assert!(rs.find_reel(ReelId(99)).is_none());
    }

    #[test]
    fn test_validator_no_issues() {
        let mut reel = make_reel(1, "A001", 500);
        reel.add_clip(make_clip("c1", 1, 0, 100));
        reel.add_clip(make_clip("c2", 1, 100, 200));
        let issues = ReelValidator::check(&[reel]);
        assert!(issues.is_empty(), "unexpected issues: {issues:?}");
    }

    #[test]
    fn test_validator_detects_gap() {
        let mut reel = make_reel(1, "A001", 500);
        reel.add_clip(make_clip("c1", 1, 0, 50));
        reel.add_clip(make_clip("c2", 1, 100, 50));
        let issues = ReelValidator::check(&[reel]);
        assert!(issues.iter().any(|i| i.issue_type == ReelIssueType::Gap));
    }

    #[test]
    fn test_validator_detects_overlap() {
        let mut reel = make_reel(1, "A001", 500);
        reel.add_clip(make_clip("c1", 1, 0, 100));
        reel.add_clip(make_clip("c2", 1, 50, 100)); // overlaps with c1
        let issues = ReelValidator::check(&[reel]);
        assert!(issues
            .iter()
            .any(|i| i.issue_type == ReelIssueType::Overlap));
    }

    #[test]
    fn test_validator_detects_duration_mismatch() {
        let mut reel = make_reel(1, "A001", 100);
        reel.add_clip(make_clip("c1", 1, 80, 50)); // ends at 130 > 100
        let issues = ReelValidator::check(&[reel]);
        assert!(issues
            .iter()
            .any(|i| i.issue_type == ReelIssueType::DurationMismatch));
    }

    #[test]
    fn test_validator_detects_naming_conflict() {
        let r1 = make_reel(1, "A001", 100);
        let r2 = make_reel(2, "A001", 200); // same name
        let issues = ReelValidator::check(&[r1, r2]);
        assert!(issues
            .iter()
            .any(|i| i.issue_type == ReelIssueType::NamingConflict));
    }
}
