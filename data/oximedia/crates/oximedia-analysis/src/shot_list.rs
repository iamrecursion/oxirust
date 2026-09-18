//! Shot list generation from analysis data.
//!
//! This module provides data structures and logic for generating
//! a professional shot list from video analysis results.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// The type of a shot in a production.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ShotType {
    /// Wide establishing shot that sets the scene.
    Establishing,
    /// Master/wide shot covering the full scene.
    MasterShot,
    /// Close-up shot focusing on a subject.
    Closeup,
    /// Insert shot of a detail or object.
    Insert,
    /// Reaction shot of a character responding.
    Reaction,
    /// Cutaway to a related but different subject.
    Cutaway,
}

impl ShotType {
    /// Returns true if this shot type typically requires dialogue.
    pub fn requires_dialogue(&self) -> bool {
        matches!(self, Self::MasterShot | Self::Reaction)
    }

    /// Returns true if this shot type is a coverage shot.
    pub fn is_coverage(&self) -> bool {
        matches!(
            self,
            Self::MasterShot | Self::Closeup | Self::Reaction | Self::Cutaway
        )
    }
}

/// A single entry in a shot list.
#[derive(Debug, Clone)]
pub struct ShotListEntry {
    /// Shot number in the sequence.
    pub shot_number: u32,
    /// Timecode in (frame number).
    pub timecode_in: u64,
    /// Timecode out (frame number).
    pub timecode_out: u64,
    /// Type of the shot.
    pub shot_type: ShotType,
    /// Location description.
    pub location: String,
    /// Shot description.
    pub description: String,
}

impl ShotListEntry {
    /// Create a new shot list entry.
    pub fn new(
        shot_number: u32,
        timecode_in: u64,
        timecode_out: u64,
        shot_type: ShotType,
        location: String,
        description: String,
    ) -> Self {
        Self {
            shot_number,
            timecode_in,
            timecode_out,
            shot_type,
            location,
            description,
        }
    }

    /// Returns the duration in frames.
    pub fn duration_frames(&self) -> u64 {
        self.timecode_out.saturating_sub(self.timecode_in)
    }

    /// Returns true if the shot duration exceeds the given threshold.
    pub fn is_long(&self, threshold_frames: u64) -> bool {
        self.duration_frames() > threshold_frames
    }
}

/// A complete shot list for a production.
#[derive(Debug, Clone)]
pub struct ShotList {
    /// All shot entries.
    pub entries: Vec<ShotListEntry>,
    /// Frame rate for this shot list.
    pub frame_rate: f32,
}

impl ShotList {
    /// Create an empty shot list.
    pub fn new(frame_rate: f32) -> Self {
        Self {
            entries: Vec::new(),
            frame_rate,
        }
    }

    /// Add a shot entry to the list.
    pub fn add(&mut self, entry: ShotListEntry) {
        self.entries.push(entry);
    }

    /// Returns the total duration across all shots in frames.
    pub fn total_duration_frames(&self) -> u64 {
        self.entries
            .iter()
            .map(ShotListEntry::duration_frames)
            .sum()
    }

    /// Returns all shots of a specific type.
    pub fn shots_by_type(&self, t: &ShotType) -> Vec<&ShotListEntry> {
        self.entries.iter().filter(|e| &e.shot_type == t).collect()
    }

    /// Returns the average shot duration in frames.
    pub fn average_duration_frames(&self) -> f32 {
        if self.entries.is_empty() {
            return 0.0;
        }
        self.total_duration_frames() as f32 / self.entries.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(num: u32, tc_in: u64, tc_out: u64, shot_type: ShotType) -> ShotListEntry {
        ShotListEntry::new(
            num,
            tc_in,
            tc_out,
            shot_type,
            "INT. STUDIO - DAY".to_string(),
            "Test shot description".to_string(),
        )
    }

    #[test]
    fn test_shot_type_requires_dialogue() {
        assert!(!ShotType::Establishing.requires_dialogue());
        assert!(ShotType::MasterShot.requires_dialogue());
        assert!(!ShotType::Closeup.requires_dialogue());
        assert!(!ShotType::Insert.requires_dialogue());
        assert!(ShotType::Reaction.requires_dialogue());
        assert!(!ShotType::Cutaway.requires_dialogue());
    }

    #[test]
    fn test_shot_type_is_coverage() {
        assert!(!ShotType::Establishing.is_coverage());
        assert!(ShotType::MasterShot.is_coverage());
        assert!(ShotType::Closeup.is_coverage());
        assert!(!ShotType::Insert.is_coverage());
        assert!(ShotType::Reaction.is_coverage());
        assert!(ShotType::Cutaway.is_coverage());
    }

    #[test]
    fn test_entry_duration_frames() {
        let entry = make_entry(1, 100, 250, ShotType::Establishing);
        assert_eq!(entry.duration_frames(), 150);
    }

    #[test]
    fn test_entry_duration_frames_zero() {
        let entry = make_entry(1, 50, 50, ShotType::Insert);
        assert_eq!(entry.duration_frames(), 0);
    }

    #[test]
    fn test_entry_is_long() {
        let entry = make_entry(1, 0, 300, ShotType::MasterShot);
        assert!(entry.is_long(100));
        assert!(!entry.is_long(300));
        assert!(!entry.is_long(500));
    }

    #[test]
    fn test_shot_list_empty() {
        let list = ShotList::new(25.0);
        assert_eq!(list.total_duration_frames(), 0);
        assert_eq!(list.average_duration_frames(), 0.0);
        assert!(list.shots_by_type(&ShotType::Closeup).is_empty());
    }

    #[test]
    fn test_shot_list_add() {
        let mut list = ShotList::new(24.0);
        list.add(make_entry(1, 0, 100, ShotType::Establishing));
        list.add(make_entry(2, 100, 200, ShotType::Closeup));
        assert_eq!(list.entries.len(), 2);
    }

    #[test]
    fn test_shot_list_total_duration() {
        let mut list = ShotList::new(25.0);
        list.add(make_entry(1, 0, 100, ShotType::Establishing));
        list.add(make_entry(2, 100, 250, ShotType::MasterShot));
        list.add(make_entry(3, 250, 300, ShotType::Closeup));
        assert_eq!(list.total_duration_frames(), 300);
    }

    #[test]
    fn test_shot_list_shots_by_type() {
        let mut list = ShotList::new(25.0);
        list.add(make_entry(1, 0, 100, ShotType::Establishing));
        list.add(make_entry(2, 100, 200, ShotType::Closeup));
        list.add(make_entry(3, 200, 300, ShotType::Closeup));
        list.add(make_entry(4, 300, 400, ShotType::Reaction));

        let closeups = list.shots_by_type(&ShotType::Closeup);
        assert_eq!(closeups.len(), 2);

        let reactions = list.shots_by_type(&ShotType::Reaction);
        assert_eq!(reactions.len(), 1);

        let inserts = list.shots_by_type(&ShotType::Insert);
        assert!(inserts.is_empty());
    }

    #[test]
    fn test_shot_list_average_duration() {
        let mut list = ShotList::new(25.0);
        list.add(make_entry(1, 0, 100, ShotType::MasterShot));
        list.add(make_entry(2, 100, 200, ShotType::Closeup));
        // Both shots are 100 frames each
        assert!((list.average_duration_frames() - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_shot_list_average_duration_varied() {
        let mut list = ShotList::new(30.0);
        list.add(make_entry(1, 0, 60, ShotType::Establishing));
        list.add(make_entry(2, 60, 180, ShotType::MasterShot));
        // 60 + 120 = 180, avg = 90
        assert!((list.average_duration_frames() - 90.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_entry_fields() {
        let entry = ShotListEntry::new(
            5,
            1000,
            1200,
            ShotType::Cutaway,
            "EXT. PARK - DAY".to_string(),
            "Cutaway to birds".to_string(),
        );
        assert_eq!(entry.shot_number, 5);
        assert_eq!(entry.timecode_in, 1000);
        assert_eq!(entry.timecode_out, 1200);
        assert_eq!(entry.location, "EXT. PARK - DAY");
        assert_eq!(entry.description, "Cutaway to birds");
    }

    #[test]
    fn test_shot_list_frame_rate() {
        let list = ShotList::new(29.97);
        assert!((list.frame_rate - 29.97).abs() < 0.01);
    }
}
