//! Nested sequence (sub-sequence) support for timeline editing.
//!
//! Allows timelines to contain references to other timelines, enabling
//! hierarchical composition, reusable sequence templates, and
//! non-destructive editing of complex projects.

use std::collections::HashMap;
use std::fmt;

use oximedia_core::Rational;

// ---------------------------------------------------------------------------
// Sequence identity
// ---------------------------------------------------------------------------

/// Unique identifier for a sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SequenceId(pub u64);

impl fmt::Display for SequenceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "seq-{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Nested sequence
// ---------------------------------------------------------------------------

/// How a nested sequence is rendered when its frame rate differs from the
/// parent timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConformMethod {
    /// Drop or duplicate frames to match the parent rate.
    NearestFrame,
    /// Blend adjacent frames for smoother conversion.
    FrameBlend,
    /// Use optical flow for highest quality (computationally expensive).
    OpticalFlow,
}

impl fmt::Display for ConformMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NearestFrame => write!(f, "nearest"),
            Self::FrameBlend => write!(f, "blend"),
            Self::OpticalFlow => write!(f, "optical-flow"),
        }
    }
}

/// A reference to a sub-sequence placed on a parent timeline.
#[derive(Debug, Clone)]
pub struct NestedSequenceRef {
    /// ID of the child sequence being referenced.
    pub sequence_id: SequenceId,
    /// Position on the parent timeline (timebase units). May be negative for pre-roll.
    pub position: i64,
    /// In-point within the child sequence.
    pub in_point: i64,
    /// Out-point within the child sequence.
    pub out_point: i64,
    /// Speed multiplier (1.0 = normal).
    pub speed: f64,
    /// How to conform mismatched frame rates.
    pub conform: ConformMethod,
    /// Whether the nested ref is flattened (baked) or live.
    pub flattened: bool,
}

impl NestedSequenceRef {
    /// Create a new nested sequence reference.
    #[must_use]
    pub fn new(sequence_id: SequenceId, position: i64, in_point: i64, out_point: i64) -> Self {
        Self {
            sequence_id,
            position,
            in_point,
            out_point,
            speed: 1.0,
            conform: ConformMethod::NearestFrame,
            flattened: false,
        }
    }

    /// Builder: set speed.
    #[must_use]
    pub fn with_speed(mut self, speed: f64) -> Self {
        self.speed = speed;
        self
    }

    /// Builder: set conform method.
    #[must_use]
    pub fn with_conform(mut self, method: ConformMethod) -> Self {
        self.conform = method;
        self
    }

    /// Duration of this reference on the parent timeline.
    ///
    /// Returns the number of parent timebase units occupied by this reference,
    /// computed from the source range scaled by speed. Always non-negative.
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    #[must_use]
    pub fn duration(&self) -> i64 {
        let source_len = (self.out_point - self.in_point).max(0);
        if self.speed.abs() < f64::EPSILON {
            return 0;
        }
        (source_len as f64 / self.speed).round().max(0.0) as i64
    }

    /// End position on the parent timeline.
    #[must_use]
    pub fn end_position(&self) -> i64 {
        self.position.saturating_add(self.duration())
    }
}

// ---------------------------------------------------------------------------
// Sequence definition
// ---------------------------------------------------------------------------

/// A sequence (timeline) that can be nested inside other sequences.
#[derive(Debug, Clone)]
pub struct NestedSequence {
    /// Unique identifier.
    pub id: SequenceId,
    /// Human-readable name.
    pub name: String,
    /// Frame rate of this sequence, expressed as a `Rational` (e.g. `Rational::new(24, 1)`).
    pub frame_rate: Rational,
    /// Total duration in timebase units.
    pub duration: i64,
    /// Resolution in pixels: `(width, height)`. Defaults to `(1920, 1080)`.
    pub resolution: (u32, u32),
    /// Number of tracks in this sequence.
    pub track_count: u32,
    /// Nested references *within* this sequence (sub-sub-sequences).
    pub nested_refs: Vec<NestedSequenceRef>,
}

impl NestedSequence {
    /// Create a new sequence with default 1920×1080 resolution.
    #[must_use]
    pub fn new(id: SequenceId, name: impl Into<String>, frame_rate: Rational) -> Self {
        Self {
            id,
            name: name.into(),
            frame_rate,
            duration: 0,
            resolution: (1920, 1080),
            track_count: 0,
            nested_refs: Vec::new(),
        }
    }

    /// Builder: set duration.
    #[must_use]
    pub fn with_duration(mut self, duration: i64) -> Self {
        self.duration = duration;
        self
    }

    /// Builder: set resolution.
    #[must_use]
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.resolution = (width, height);
        self
    }

    /// Builder: set track count.
    #[must_use]
    pub fn with_track_count(mut self, count: u32) -> Self {
        self.track_count = count;
        self
    }

    /// Add a nested reference.
    pub fn add_ref(&mut self, r: NestedSequenceRef) {
        self.nested_refs.push(r);
    }

    /// Remove a nested reference by child sequence ID.
    pub fn remove_ref(&mut self, child_id: SequenceId) -> usize {
        let before = self.nested_refs.len();
        self.nested_refs.retain(|r| r.sequence_id != child_id);
        before - self.nested_refs.len()
    }

    /// Returns `true` if this sequence contains a reference to `child_id`.
    #[must_use]
    pub fn references(&self, child_id: SequenceId) -> bool {
        self.nested_refs.iter().any(|r| r.sequence_id == child_id)
    }

    /// Compute the duration of this sequence in an outer (parent) timebase.
    ///
    /// Uses integer rounding (nearest-frame) to convert from this sequence's
    /// own `frame_rate` into `outer_rate`. All three [`ConformMethod`] variants
    /// produce the same duration accounting; `FrameBlend` and `OpticalFlow`
    /// differ only in render quality, not timeline length.
    ///
    /// # Returns
    ///
    /// The number of `outer_rate` units required to represent this sequence's
    /// full duration. Returns `0` if either rate has a zero component.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_core::Rational;
    /// use oximedia_edit::nested_sequence::{NestedSequence, SequenceId};
    ///
    /// // 24 inner frames at 24 fps = 1 second = 60 frames at 60 fps
    /// let seq = NestedSequence::new(SequenceId(1), "Clip", Rational::new(24, 1))
    ///     .with_duration(24);
    /// let outer = seq.duration_in_outer_timebase(Rational::new(60, 1));
    /// assert_eq!(outer, 60);
    /// ```
    #[must_use]
    pub fn duration_in_outer_timebase(&self, outer_rate: Rational) -> i64 {
        // Guard against zero denominators / numerators.
        if self.frame_rate.num == 0
            || self.frame_rate.den == 0
            || outer_rate.num == 0
            || outer_rate.den == 0
        {
            return 0;
        }
        // Convert:
        //   inner_duration (ticks at inner_rate) → outer_duration (ticks at outer_rate)
        //
        //   outer_duration = inner_duration * (outer_rate / inner_rate)
        //                  = inner_duration * outer_rate.num * inner_rate.den
        //                    ─────────────────────────────────────────────────
        //                           outer_rate.den * inner_rate.num
        //
        // Rounding: add half of the denominator before integer division to
        // implement "round to nearest frame" (NearestFrame semantics).
        // FrameBlend and OpticalFlow share the same timeline duration.
        let numerator: i128 =
            self.duration as i128 * outer_rate.num as i128 * self.frame_rate.den as i128;
        let denominator: i128 = outer_rate.den as i128 * self.frame_rate.num as i128;
        if denominator == 0 {
            return 0;
        }
        // Round: (numerator + denominator/2) / denominator
        // Using the sign-aware rounding: for positive values add half-denom.
        let half = denominator.abs() / 2;
        let rounded = if numerator >= 0 {
            (numerator + half) / denominator
        } else {
            (numerator - half) / denominator
        };
        rounded as i64
    }
}

// ---------------------------------------------------------------------------
// Sequence registry
// ---------------------------------------------------------------------------

/// Manages all sequences in a project and detects circular references.
#[derive(Debug, Clone)]
pub struct SequenceRegistry {
    /// All known sequences.
    sequences: HashMap<SequenceId, NestedSequence>,
    /// Auto-increment counter.
    next_id: u64,
}

impl SequenceRegistry {
    /// Create a new, empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sequences: HashMap::new(),
            next_id: 1,
        }
    }

    /// Create a new sequence and return its ID.
    pub fn create(&mut self, name: impl Into<String>, frame_rate: Rational) -> SequenceId {
        let id = SequenceId(self.next_id);
        self.next_id += 1;
        let seq = NestedSequence::new(id, name, frame_rate);
        self.sequences.insert(id, seq);
        id
    }

    /// Get a sequence by ID.
    #[must_use]
    pub fn get(&self, id: SequenceId) -> Option<&NestedSequence> {
        self.sequences.get(&id)
    }

    /// Get a mutable reference.
    pub fn get_mut(&mut self, id: SequenceId) -> Option<&mut NestedSequence> {
        self.sequences.get_mut(&id)
    }

    /// Remove a sequence and all references to it from other sequences.
    pub fn remove(&mut self, id: SequenceId) -> Option<NestedSequence> {
        let removed = self.sequences.remove(&id);
        // Clean up references in remaining sequences.
        for seq in self.sequences.values_mut() {
            seq.remove_ref(id);
        }
        removed
    }

    /// Returns the number of sequences.
    #[must_use]
    pub fn count(&self) -> usize {
        self.sequences.len()
    }

    /// Returns `true` if there are no sequences.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Check if adding a reference from `parent` to `child` would create a
    /// cycle. Returns `true` if a cycle would be created.
    #[must_use]
    pub fn would_create_cycle(&self, parent: SequenceId, child: SequenceId) -> bool {
        if parent == child {
            return true;
        }
        // DFS from child's nested refs to see if we can reach parent
        let mut stack = vec![child];
        let mut visited = std::collections::HashSet::new();
        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            if let Some(seq) = self.sequences.get(&current) {
                for r in &seq.nested_refs {
                    if r.sequence_id == parent {
                        return true;
                    }
                    stack.push(r.sequence_id);
                }
            }
        }
        false
    }

    /// Compute the nesting depth of a sequence (1 = no nesting).
    #[must_use]
    pub fn depth(&self, id: SequenceId) -> usize {
        let seq = match self.sequences.get(&id) {
            Some(s) => s,
            None => return 0,
        };
        if seq.nested_refs.is_empty() {
            return 1;
        }
        let max_child = seq
            .nested_refs
            .iter()
            .map(|r| self.depth(r.sequence_id))
            .max()
            .unwrap_or(0);
        1 + max_child
    }
}

impl Default for SequenceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_id_display() {
        assert_eq!(SequenceId(7).to_string(), "seq-7");
    }

    #[test]
    fn test_rational_frame_rate_display() {
        // Rational::Display shows "num/den" — verify 24/1 and NTSC.
        let fr = Rational::new(24, 1);
        assert_eq!(fr.to_string(), "24/1");
        let ntsc = Rational::new(30000, 1001);
        assert_eq!(ntsc.to_string(), "30000/1001");
    }

    #[test]
    fn test_conform_method_display() {
        assert_eq!(ConformMethod::NearestFrame.to_string(), "nearest");
        assert_eq!(ConformMethod::OpticalFlow.to_string(), "optical-flow");
    }

    #[test]
    fn test_nested_ref_duration_normal_speed() {
        let r = NestedSequenceRef::new(SequenceId(1), 0, 0, 1000);
        assert_eq!(r.duration(), 1000);
    }

    #[test]
    fn test_nested_ref_duration_double_speed() {
        let r = NestedSequenceRef::new(SequenceId(1), 0, 0, 1000).with_speed(2.0);
        assert_eq!(r.duration(), 500);
    }

    #[test]
    fn test_nested_ref_duration_zero_speed() {
        let r = NestedSequenceRef::new(SequenceId(1), 0, 0, 1000).with_speed(0.0);
        assert_eq!(r.duration(), 0);
    }

    #[test]
    fn test_nested_ref_end_position() {
        let r = NestedSequenceRef::new(SequenceId(1), 500, 0, 1000);
        assert_eq!(r.end_position(), 1500);
    }

    #[test]
    fn test_nested_ref_negative_position() {
        // i64 positions allow pre-roll before timeline start.
        let r = NestedSequenceRef::new(SequenceId(1), -24, 0, 48);
        assert_eq!(r.position, -24_i64);
        assert_eq!(r.end_position(), 24); // -24 + 48 = 24
    }

    #[test]
    fn test_nested_sequence_add_and_remove_ref() {
        let mut seq = NestedSequence::new(SequenceId(1), "Main", Rational::new(24, 1));
        seq.add_ref(NestedSequenceRef::new(SequenceId(2), 0, 0, 100));
        assert!(seq.references(SequenceId(2)));
        let removed = seq.remove_ref(SequenceId(2));
        assert_eq!(removed, 1);
        assert!(!seq.references(SequenceId(2)));
    }

    #[test]
    fn test_nested_sequence_default_resolution() {
        let seq = NestedSequence::new(SequenceId(1), "S", Rational::new(24, 1));
        assert_eq!(seq.resolution, (1920, 1080));
    }

    #[test]
    fn test_nested_sequence_with_resolution() {
        let seq = NestedSequence::new(SequenceId(1), "S", Rational::new(24, 1))
            .with_resolution(1280, 720);
        assert_eq!(seq.resolution, (1280, 720));
    }

    #[test]
    fn test_registry_create_and_get() {
        let mut reg = SequenceRegistry::new();
        let id = reg.create("Timeline 1", Rational::new(30, 1));
        assert!(reg.get(id).is_some());
        assert_eq!(reg.count(), 1);
    }

    #[test]
    fn test_registry_remove_cleans_refs() {
        let mut reg = SequenceRegistry::new();
        let parent = reg.create("Parent", Rational::new(24, 1));
        let child = reg.create("Child", Rational::new(24, 1));
        reg.get_mut(parent)
            .expect("test expectation failed")
            .add_ref(NestedSequenceRef::new(child, 0, 0, 100));
        reg.remove(child);
        assert!(!reg
            .get(parent)
            .expect("get should succeed")
            .references(child));
    }

    #[test]
    fn test_registry_cycle_detection_self() {
        let mut reg = SequenceRegistry::new();
        let id = reg.create("S", Rational::new(24, 1));
        assert!(reg.would_create_cycle(id, id));
    }

    #[test]
    fn test_registry_cycle_detection_indirect() {
        let mut reg = SequenceRegistry::new();
        let a = reg.create("A", Rational::new(24, 1));
        let b = reg.create("B", Rational::new(24, 1));
        let c = reg.create("C", Rational::new(24, 1));
        reg.get_mut(a)
            .expect("test expectation failed")
            .add_ref(NestedSequenceRef::new(b, 0, 0, 100));
        reg.get_mut(b)
            .expect("test expectation failed")
            .add_ref(NestedSequenceRef::new(c, 0, 0, 100));
        // c -> a would create a cycle
        assert!(reg.would_create_cycle(c, a));
        // a -> c already exists, not a cycle from that direction
        assert!(!reg.would_create_cycle(a, c));
    }

    #[test]
    fn test_registry_depth() {
        let mut reg = SequenceRegistry::new();
        let a = reg.create("A", Rational::new(24, 1));
        let b = reg.create("B", Rational::new(24, 1));
        reg.get_mut(a)
            .expect("test expectation failed")
            .add_ref(NestedSequenceRef::new(b, 0, 0, 100));
        assert_eq!(reg.depth(a), 2);
        assert_eq!(reg.depth(b), 1);
    }

    #[test]
    fn test_registry_default() {
        let reg = SequenceRegistry::default();
        assert!(reg.is_empty());
    }

    #[test]
    fn test_nested_sequence_builders() {
        let seq = NestedSequence::new(SequenceId(1), "S", Rational::new(25, 1))
            .with_duration(5000)
            .with_track_count(4);
        assert_eq!(seq.duration, 5000);
        assert_eq!(seq.track_count, 4);
    }

    #[test]
    fn test_nested_ref_with_conform() {
        let r = NestedSequenceRef::new(SequenceId(1), 0, 0, 100)
            .with_conform(ConformMethod::FrameBlend);
        assert_eq!(r.conform, ConformMethod::FrameBlend);
    }

    // --- duration_in_outer_timebase tests ---

    #[test]
    fn test_duration_in_outer_same_rate() {
        // 100 frames at 25 fps → 100 frames at 25 fps
        let seq = NestedSequence::new(SequenceId(1), "S", Rational::new(25, 1)).with_duration(100);
        assert_eq!(seq.duration_in_outer_timebase(Rational::new(25, 1)), 100);
    }

    #[test]
    fn test_duration_24fps_to_60fps() {
        // 24 inner frames at 24 fps = 1 second = 60 frames at 60 fps
        let seq = NestedSequence::new(SequenceId(1), "S", Rational::new(24, 1)).with_duration(24);
        assert_eq!(seq.duration_in_outer_timebase(Rational::new(60, 1)), 60);
    }

    #[test]
    fn test_duration_60fps_to_24fps() {
        // 60 inner frames at 60 fps = 1 second = 24 frames at 24 fps
        let seq = NestedSequence::new(SequenceId(1), "S", Rational::new(60, 1)).with_duration(60);
        assert_eq!(seq.duration_in_outer_timebase(Rational::new(24, 1)), 24);
    }

    #[test]
    fn test_duration_ntsc_to_integer_fps() {
        // 30000/1001 fps (NTSC) — 30030 inner ticks ≈ 1001 * 30 = 1 second
        // outer: 30 fps integer — expect 30 outer frames (1 s)
        let seq = NestedSequence::new(SequenceId(1), "S", Rational::new(30000, 1001))
            .with_duration(30000); // 30000 / (30000/1001) = 1001/1 seconds... hmm
                                   // Actually 30000 inner ticks at 30000/1001 fps:
                                   //   duration_secs = 30000 * 1001 / 30000 = 1001 s   (not useful for a unit test)
                                   // Use a smaller value: 30000/1001 fps, duration = 1001 ticks:
                                   //   1001 * (1001/30000) s * (30/1) fps = 1001 * 1001 / 30000 ≈ 33.4 outer frames
                                   // Let's use an exact case: 30000 ticks at 30000/1001 fps →
                                   //   30000 / (30000/1001) = 1001 s  → at 30fps = 30030 outer
        let seq2 = NestedSequence::new(SequenceId(1), "S", Rational::new(30000, 1001))
            .with_duration(30000);
        // outer 30/1:  30000 * 30 * 1001 / (1 * 30000) = 30030
        assert_eq!(seq2.duration_in_outer_timebase(Rational::new(30, 1)), 30030);
        drop(seq); // silence unused warning
    }

    #[test]
    fn test_duration_zero_inner_duration() {
        let seq = NestedSequence::new(SequenceId(1), "S", Rational::new(24, 1)).with_duration(0);
        assert_eq!(seq.duration_in_outer_timebase(Rational::new(60, 1)), 0);
    }
}
