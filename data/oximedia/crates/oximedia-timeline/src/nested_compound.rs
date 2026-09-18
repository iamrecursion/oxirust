//! Nested and compound clip management.
//!
//! Provides `NestedClip`, `NestingLevel`, `NestedClipRegistry`, and
//! `FlattenedTimeline` for working with hierarchically grouped clips.

/// A compound clip that wraps a group of inner clip IDs into a single entity.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct NestedClip {
    /// Unique identifier for this compound clip.
    pub id: u64,
    /// Human-readable label.
    pub name: String,
    /// Ordered list of inner clip IDs included in this compound.
    pub inner_clips: Vec<u64>,
    /// Frame position on the outer timeline where this compound starts.
    pub start_frame: u64,
    /// Number of frames this compound occupies on the outer timeline.
    pub duration_frames: u32,
}

impl NestedClip {
    /// Returns the number of inner clips this compound contains.
    #[must_use]
    pub fn clip_count(&self) -> usize {
        self.inner_clips.len()
    }

    /// Returns the exclusive end frame of this compound on the outer timeline.
    #[must_use]
    pub fn end_frame(&self) -> u64 {
        self.start_frame + u64::from(self.duration_frames)
    }

    /// Returns `true` if this compound contains the clip with the given ID.
    #[must_use]
    pub fn contains_clip(&self, id: u64) -> bool {
        self.inner_clips.contains(&id)
    }
}

/// Tracks the depth of compound-clip nesting to enforce a maximum recursion limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub struct NestingLevel {
    /// Current nesting depth (0 = top-level).
    pub depth: u32,
    /// Maximum permitted depth.
    pub max_depth: u32,
}

impl NestingLevel {
    /// Create a new `NestingLevel` at depth 0 with the given maximum.
    #[must_use]
    pub fn new(max_depth: u32) -> Self {
        Self {
            depth: 0,
            max_depth,
        }
    }

    /// Returns `true` if another level of nesting is permitted.
    #[must_use]
    pub fn can_nest(&self) -> bool {
        self.depth < self.max_depth
    }

    /// Return a new `NestingLevel` with the depth incremented by one.
    ///
    /// The depth is capped at `max_depth` to prevent overflow.
    #[must_use]
    pub fn increment(&self) -> Self {
        Self {
            depth: self.depth.saturating_add(1).min(self.max_depth),
            max_depth: self.max_depth,
        }
    }
}

/// Registry for `NestedClip` instances, with creation and lookup helpers.
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct NestedClipRegistry {
    /// All registered compounds.
    pub compounds: Vec<NestedClip>,
    /// Next ID to assign when creating a compound.
    next_id: u64,
}

impl NestedClipRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            compounds: Vec::new(),
            next_id: 1,
        }
    }

    /// Create and register a new compound clip.
    ///
    /// Returns the ID of the newly created compound.
    pub fn create(&mut self, name: &str, clips: Vec<u64>, start: u64, dur: u32) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.compounds.push(NestedClip {
            id,
            name: name.to_string(),
            inner_clips: clips,
            start_frame: start,
            duration_frames: dur,
        });
        id
    }

    /// Look up a compound clip by its ID.
    #[must_use]
    pub fn find(&self, id: u64) -> Option<&NestedClip> {
        self.compounds.iter().find(|c| c.id == id)
    }

    /// Return the inner clip IDs of the compound with `id`, or an empty list
    /// if the compound does not exist.
    #[must_use]
    pub fn flatten(&self, id: u64) -> Vec<u64> {
        self.find(id)
            .map(|c| c.inner_clips.clone())
            .unwrap_or_default()
    }

    /// Return the total number of compound clips in the registry.
    #[must_use]
    pub fn total_compounds(&self) -> usize {
        self.compounds.len()
    }
}

/// The result of flattening one or more compound clips into a linear list of
/// `(compound_id, clip_id)` pairs.
#[derive(Debug, Default, Clone)]
#[allow(dead_code)]
pub struct FlattenedTimeline {
    /// Ordered list of `(compound_id, inner_clip_id)` pairs.
    pub clips: Vec<(u64, u64)>,
}

impl FlattenedTimeline {
    /// Returns the number of entries in the flattened timeline.
    #[must_use]
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_nested(id: u64, inner: Vec<u64>, start: u64, dur: u32) -> NestedClip {
        NestedClip {
            id,
            name: format!("Compound-{id}"),
            inner_clips: inner,
            start_frame: start,
            duration_frames: dur,
        }
    }

    // --- NestedClip ---

    #[test]
    fn test_clip_count() {
        let c = make_nested(1, vec![10, 20, 30], 0, 150);
        assert_eq!(c.clip_count(), 3);
    }

    #[test]
    fn test_clip_count_empty() {
        let c = make_nested(2, vec![], 0, 0);
        assert_eq!(c.clip_count(), 0);
    }

    #[test]
    fn test_end_frame() {
        let c = make_nested(3, vec![1], 100, 50);
        assert_eq!(c.end_frame(), 150);
    }

    #[test]
    fn test_end_frame_zero_duration() {
        let c = make_nested(4, vec![], 200, 0);
        assert_eq!(c.end_frame(), 200);
    }

    #[test]
    fn test_contains_clip_true() {
        let c = make_nested(5, vec![7, 8, 9], 0, 100);
        assert!(c.contains_clip(8));
    }

    #[test]
    fn test_contains_clip_false() {
        let c = make_nested(6, vec![1, 2, 3], 0, 100);
        assert!(!c.contains_clip(99));
    }

    // --- NestingLevel ---

    #[test]
    fn test_nesting_can_nest_at_zero() {
        let nl = NestingLevel::new(4);
        assert!(nl.can_nest());
    }

    #[test]
    fn test_nesting_cannot_nest_at_max() {
        let nl = NestingLevel {
            depth: 4,
            max_depth: 4,
        };
        assert!(!nl.can_nest());
    }

    #[test]
    fn test_nesting_increment_increases_depth() {
        let nl = NestingLevel::new(4);
        let nl2 = nl.increment();
        assert_eq!(nl2.depth, 1);
    }

    #[test]
    fn test_nesting_increment_caps_at_max() {
        let nl = NestingLevel {
            depth: 4,
            max_depth: 4,
        };
        let nl2 = nl.increment();
        assert_eq!(nl2.depth, 4);
    }

    // --- NestedClipRegistry ---

    #[test]
    fn test_registry_create_returns_sequential_ids() {
        let mut reg = NestedClipRegistry::new();
        let id1 = reg.create("A", vec![1, 2], 0, 100);
        let id2 = reg.create("B", vec![3], 100, 50);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[test]
    fn test_registry_find_existing() {
        let mut reg = NestedClipRegistry::new();
        reg.create("Test", vec![10, 20], 0, 80);
        assert!(reg.find(1).is_some());
    }

    #[test]
    fn test_registry_find_nonexistent() {
        let reg = NestedClipRegistry::new();
        assert!(reg.find(999).is_none());
    }

    #[test]
    fn test_registry_flatten_returns_inner_clips() {
        let mut reg = NestedClipRegistry::new();
        reg.create("X", vec![5, 6, 7], 0, 120);
        let inner = reg.flatten(1);
        assert_eq!(inner, vec![5, 6, 7]);
    }

    #[test]
    fn test_registry_flatten_nonexistent_returns_empty() {
        let reg = NestedClipRegistry::new();
        assert!(reg.flatten(99).is_empty());
    }

    #[test]
    fn test_registry_total_compounds() {
        let mut reg = NestedClipRegistry::new();
        reg.create("A", vec![1], 0, 50);
        reg.create("B", vec![2], 50, 50);
        assert_eq!(reg.total_compounds(), 2);
    }

    // --- FlattenedTimeline ---

    #[test]
    fn test_flattened_clip_count() {
        let ft = FlattenedTimeline {
            clips: vec![(1, 10), (1, 11), (2, 20)],
        };
        assert_eq!(ft.clip_count(), 3);
    }

    #[test]
    fn test_flattened_clip_count_empty() {
        let ft = FlattenedTimeline::default();
        assert_eq!(ft.clip_count(), 0);
    }
}
