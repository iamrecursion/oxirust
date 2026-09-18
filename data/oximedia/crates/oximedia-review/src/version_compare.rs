//! Version comparison and diff utilities for review workflows.
//!
//! Provides segment-level diffing between timeline versions and a version
//! history ledger keyed by author and version number.

/// A discrete segment of a timeline identified by a clip reference.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub struct TimelineSegment {
    /// Unique identifier for this segment.
    pub id: u64,
    /// Start position in milliseconds.
    pub start_ms: u64,
    /// End position in milliseconds (exclusive).
    pub end_ms: u64,
    /// Identifier of the clip placed at this position.
    pub clip_id: String,
}

impl TimelineSegment {
    /// Returns the duration of this segment in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Returns `true` if this segment's time range overlaps with `other`.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start_ms < other.end_ms && other.start_ms < self.end_ms
    }
}

/// Represents the differences between two versions of a timeline.
///
/// Each entry is a `(id, id)` tuple identifying the segments involved.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct VersionDiff {
    /// Segments present in v2 but not in v1.
    pub added_segments: Vec<(u64, u64)>,
    /// Segments present in v1 but not in v2.
    pub removed_segments: Vec<(u64, u64)>,
    /// Segments whose clip reference changed between v1 and v2.
    pub modified_segments: Vec<(u64, u64)>,
}

impl VersionDiff {
    /// Returns `true` if v1 and v2 are identical (no additions, removals or modifications).
    #[must_use]
    pub fn is_identical(&self) -> bool {
        self.added_segments.is_empty()
            && self.removed_segments.is_empty()
            && self.modified_segments.is_empty()
    }

    /// Returns the total number of changed segments across all categories.
    #[must_use]
    pub fn total_changes(&self) -> usize {
        self.added_segments.len() + self.removed_segments.len() + self.modified_segments.len()
    }
}

/// Compare two slices of `TimelineSegment` and return a `VersionDiff`.
///
/// Segments are matched by ID.  A segment that exists in both versions but
/// whose `clip_id` differs is classified as *modified*.  Segments present
/// only in `v1` are *removed*; segments only in `v2` are *added*.
#[must_use]
pub fn compare_timelines(v1: &[TimelineSegment], v2: &[TimelineSegment]) -> VersionDiff {
    let mut diff = VersionDiff::default();

    for seg1 in v1 {
        match v2.iter().find(|s| s.id == seg1.id) {
            Some(seg2) if seg2.clip_id != seg1.clip_id => {
                diff.modified_segments.push((seg1.id, seg2.id));
            }
            None => {
                diff.removed_segments.push((seg1.id, seg1.id));
            }
            _ => {}
        }
    }

    for seg2 in v2 {
        if !v1.iter().any(|s| s.id == seg2.id) {
            diff.added_segments.push((seg2.id, seg2.id));
        }
    }

    diff
}

/// A single entry in a version history.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct VersionEntry {
    /// Monotonically increasing version number.
    pub version_number: u32,
    /// Creation timestamp in milliseconds since epoch.
    pub created_ms: u64,
    /// Author who created this version.
    pub author: String,
    /// Human-readable description of the changes.
    pub comment: String,
    /// A hash of the timeline content at this version.
    pub timeline_hash: u64,
}

impl VersionEntry {
    /// Returns how many milliseconds have elapsed since this version was created.
    #[must_use]
    pub fn age_ms(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.created_ms)
    }
}

/// An ordered log of `VersionEntry` records for a single media project.
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct VersionHistory {
    /// Entries in insertion order (earliest first).
    pub entries: Vec<VersionEntry>,
}

impl VersionHistory {
    /// Create an empty `VersionHistory`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Append a new version entry.
    pub fn add(&mut self, entry: VersionEntry) {
        self.entries.push(entry);
    }

    /// Return the most recently added entry, if any.
    #[must_use]
    pub fn latest(&self) -> Option<&VersionEntry> {
        self.entries.last()
    }

    /// Look up an entry by version number.
    #[must_use]
    pub fn find_version(&self, num: u32) -> Option<&VersionEntry> {
        self.entries.iter().find(|e| e.version_number == num)
    }

    /// Return all entries authored by the given author.
    #[must_use]
    pub fn versions_by_author(&self, author: &str) -> Vec<&VersionEntry> {
        self.entries.iter().filter(|e| e.author == author).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(id: u64, start: u64, end: u64, clip: &str) -> TimelineSegment {
        TimelineSegment {
            id,
            start_ms: start,
            end_ms: end,
            clip_id: clip.to_string(),
        }
    }

    fn entry(version_number: u32, created_ms: u64, author: &str) -> VersionEntry {
        VersionEntry {
            version_number,
            created_ms,
            author: author.to_string(),
            comment: "auto".to_string(),
            timeline_hash: version_number as u64 * 1_000,
        }
    }

    // --- TimelineSegment ---

    #[test]
    fn test_segment_duration() {
        let s = seg(1, 1_000, 4_000, "clip-a");
        assert_eq!(s.duration_ms(), 3_000);
    }

    #[test]
    fn test_segment_duration_zero_when_inverted() {
        let s = seg(2, 5_000, 3_000, "clip-b");
        assert_eq!(s.duration_ms(), 0);
    }

    #[test]
    fn test_segment_overlaps_true() {
        let a = seg(1, 0, 1_000, "clip-a");
        let b = seg(2, 500, 1_500, "clip-b");
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_segment_overlaps_false_adjacent() {
        let a = seg(1, 0, 1_000, "clip-a");
        let b = seg(2, 1_000, 2_000, "clip-b");
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_segment_no_overlap_when_disjoint() {
        let a = seg(1, 0, 500, "clip-a");
        let b = seg(2, 600, 1_000, "clip-b");
        assert!(!a.overlaps(&b));
    }

    // --- VersionDiff ---

    #[test]
    fn test_diff_identical_timelines() {
        let v1 = vec![seg(1, 0, 1_000, "clip-a"), seg(2, 1_000, 2_000, "clip-b")];
        let v2 = v1.clone();
        let diff = compare_timelines(&v1, &v2);
        assert!(diff.is_identical());
        assert_eq!(diff.total_changes(), 0);
    }

    #[test]
    fn test_diff_added_segment() {
        let v1 = vec![seg(1, 0, 1_000, "clip-a")];
        let v2 = vec![seg(1, 0, 1_000, "clip-a"), seg(2, 1_000, 2_000, "clip-b")];
        let diff = compare_timelines(&v1, &v2);
        assert_eq!(diff.added_segments.len(), 1);
        assert_eq!(diff.total_changes(), 1);
    }

    #[test]
    fn test_diff_removed_segment() {
        let v1 = vec![seg(1, 0, 1_000, "clip-a"), seg(2, 1_000, 2_000, "clip-b")];
        let v2 = vec![seg(1, 0, 1_000, "clip-a")];
        let diff = compare_timelines(&v1, &v2);
        assert_eq!(diff.removed_segments.len(), 1);
    }

    #[test]
    fn test_diff_modified_segment() {
        let v1 = vec![seg(1, 0, 1_000, "clip-a")];
        let v2 = vec![seg(1, 0, 1_000, "clip-x")];
        let diff = compare_timelines(&v1, &v2);
        assert_eq!(diff.modified_segments.len(), 1);
        assert!(diff.added_segments.is_empty());
    }

    // --- VersionHistory ---

    #[test]
    fn test_history_latest_none_when_empty() {
        let h = VersionHistory::new();
        assert!(h.latest().is_none());
    }

    #[test]
    fn test_history_latest_returns_last_added() {
        let mut h = VersionHistory::new();
        h.add(entry(1, 100, "alice"));
        h.add(entry(2, 200, "bob"));
        assert_eq!(
            h.latest().expect("should succeed in test").version_number,
            2
        );
    }

    #[test]
    fn test_history_find_version() {
        let mut h = VersionHistory::new();
        h.add(entry(1, 100, "alice"));
        h.add(entry(2, 200, "bob"));
        assert!(h.find_version(1).is_some());
        assert!(h.find_version(99).is_none());
    }

    #[test]
    fn test_history_versions_by_author() {
        let mut h = VersionHistory::new();
        h.add(entry(1, 100, "alice"));
        h.add(entry(2, 200, "bob"));
        h.add(entry(3, 300, "alice"));
        let by_alice = h.versions_by_author("alice");
        assert_eq!(by_alice.len(), 2);
    }

    #[test]
    fn test_version_entry_age_ms() {
        let e = entry(1, 1_000, "carol");
        assert_eq!(e.age_ms(4_500), 3_500);
    }

    #[test]
    fn test_version_entry_age_ms_no_underflow() {
        let e = entry(1, 5_000, "dave");
        // now < created -> saturating_sub -> 0
        assert_eq!(e.age_ms(1_000), 0);
    }
}
