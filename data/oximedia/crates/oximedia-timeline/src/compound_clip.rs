//! Compound clips and nested timelines.
//!
//! A `CompoundClip` wraps another (inner) timeline, exposing it as a single
//! clip on an outer timeline with optional speed remapping.

/// A compound clip that references an inner timeline.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CompoundClip {
    /// Unique identifier.
    pub id: u64,
    /// Human-readable name.
    pub name: String,
    /// Identifier of the inner timeline this clip represents.
    pub inner_timeline_id: u64,
    /// First frame of the clip on the outer timeline (inclusive).
    pub start_frame: u64,
    /// Last frame of the clip on the outer timeline (exclusive).
    pub end_frame: u64,
    /// Source offset: the inner-timeline frame to begin reading from.
    pub offset_frames: u64,
    /// Playback speed multiplier (1.0 = normal, 2.0 = double, -1.0 = reversed).
    pub speed: f64,
}

impl CompoundClip {
    /// Create a new compound clip with normal speed.
    #[must_use]
    pub fn new(id: u64, name: &str, timeline_id: u64, start: u64, end: u64) -> Self {
        Self {
            id,
            name: name.to_string(),
            inner_timeline_id: timeline_id,
            start_frame: start,
            end_frame: end.max(start),
            offset_frames: 0,
            speed: 1.0,
        }
    }

    /// Set the playback speed and return `self` (builder pattern).
    #[must_use]
    pub fn with_speed(mut self, speed: f64) -> Self {
        self.speed = speed;
        self
    }

    /// Set the source offset in frames and return `self`.
    #[must_use]
    pub fn with_offset(mut self, offset_frames: u64) -> Self {
        self.offset_frames = offset_frames;
        self
    }

    /// Duration of this clip on the outer timeline in frames.
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame)
    }

    /// Map an output frame (on the outer timeline) to the corresponding
    /// source frame (on the inner timeline).
    ///
    /// The result is a floating-point frame number to account for non-integer
    /// speed values.
    #[must_use]
    pub fn source_frame(&self, output_frame: u64) -> f64 {
        let relative = output_frame.saturating_sub(self.start_frame) as f64;
        let src = self.offset_frames as f64 + relative * self.speed;
        src.max(0.0)
    }

    /// Returns `true` if playback is reversed (speed < 0.0).
    #[must_use]
    pub fn is_reversed(&self) -> bool {
        self.speed < 0.0
    }

    /// Returns `true` if this clip has a non-standard speed.
    #[must_use]
    pub fn is_retimed(&self) -> bool {
        (self.speed - 1.0).abs() > f64::EPSILON
    }

    /// Return whether a given outer frame falls within this clip.
    #[must_use]
    pub fn contains_frame(&self, frame: u64) -> bool {
        frame >= self.start_frame && frame < self.end_frame
    }
}

/// A library of compound clips that can be looked up by id or name.
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct CompoundClipLibrary {
    /// All registered compound clips.
    pub clips: Vec<CompoundClip>,
}

impl CompoundClipLibrary {
    /// Create a new, empty library.
    #[must_use]
    pub fn new() -> Self {
        Self { clips: Vec::new() }
    }

    /// Add a compound clip to the library.
    pub fn add(&mut self, clip: CompoundClip) {
        self.clips.push(clip);
    }

    /// Find a clip by id.
    #[must_use]
    pub fn find(&self, id: u64) -> Option<&CompoundClip> {
        self.clips.iter().find(|c| c.id == id)
    }

    /// Find a clip by name (returns the first match).
    #[must_use]
    pub fn find_by_name(&self, name: &str) -> Option<&CompoundClip> {
        self.clips.iter().find(|c| c.name == name)
    }

    /// Remove a clip by id.
    ///
    /// Returns `true` if the clip was found and removed.
    pub fn remove(&mut self, id: u64) -> bool {
        if let Some(pos) = self.clips.iter().position(|c| c.id == id) {
            self.clips.remove(pos);
            true
        } else {
            false
        }
    }

    /// Return the number of clips in the library.
    #[must_use]
    pub fn len(&self) -> usize {
        self.clips.len()
    }

    /// Return `true` if the library contains no clips.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.clips.is_empty()
    }

    /// Return all clips that reference the given inner timeline.
    #[must_use]
    pub fn clips_for_timeline(&self, timeline_id: u64) -> Vec<&CompoundClip> {
        self.clips
            .iter()
            .filter(|c| c.inner_timeline_id == timeline_id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_clip() -> CompoundClip {
        CompoundClip::new(1, "Scene A", 42, 0, 240)
    }

    // --- CompoundClip tests ---

    #[test]
    fn test_clip_duration() {
        let c = sample_clip();
        assert_eq!(c.duration_frames(), 240);
    }

    #[test]
    fn test_clip_source_frame_normal_speed() {
        let c = sample_clip();
        // Output frame 0 maps to inner frame 0 + offset(0) = 0.
        assert!((c.source_frame(0) - 0.0).abs() < f64::EPSILON);
        // Output frame 10 (relative = 10) * speed(1.0) = 10.
        assert!((c.source_frame(10) - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clip_source_frame_double_speed() {
        let c = CompoundClip::new(2, "Fast", 1, 0, 120).with_speed(2.0);
        // Relative frame 5 * speed 2.0 = inner frame 10.
        assert!((c.source_frame(5) - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clip_source_frame_with_offset() {
        let c = CompoundClip::new(3, "Offset", 1, 0, 100).with_offset(50);
        // Frame 0 -> offset(50) + 0 * 1.0 = 50.
        assert!((c.source_frame(0) - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clip_is_not_reversed_by_default() {
        assert!(!sample_clip().is_reversed());
    }

    #[test]
    fn test_clip_is_reversed() {
        let c = CompoundClip::new(4, "Rev", 1, 0, 100).with_speed(-1.0);
        assert!(c.is_reversed());
    }

    #[test]
    fn test_clip_is_retimed() {
        let c = CompoundClip::new(5, "Slow", 1, 0, 200).with_speed(0.5);
        assert!(c.is_retimed());
    }

    #[test]
    fn test_clip_not_retimed_at_normal_speed() {
        assert!(!sample_clip().is_retimed());
    }

    #[test]
    fn test_clip_contains_frame() {
        let c = sample_clip(); // start=0, end=240
        assert!(c.contains_frame(0));
        assert!(c.contains_frame(239));
        assert!(!c.contains_frame(240));
    }

    // --- CompoundClipLibrary tests ---

    #[test]
    fn test_library_add_and_len() {
        let mut lib = CompoundClipLibrary::new();
        lib.add(sample_clip());
        assert_eq!(lib.len(), 1);
        assert!(!lib.is_empty());
    }

    #[test]
    fn test_library_find_by_id() {
        let mut lib = CompoundClipLibrary::new();
        lib.add(sample_clip());
        assert!(lib.find(1).is_some());
        assert!(lib.find(999).is_none());
    }

    #[test]
    fn test_library_find_by_name() {
        let mut lib = CompoundClipLibrary::new();
        lib.add(sample_clip());
        assert!(lib.find_by_name("Scene A").is_some());
        assert!(lib.find_by_name("Unknown").is_none());
    }

    #[test]
    fn test_library_remove_existing() {
        let mut lib = CompoundClipLibrary::new();
        lib.add(sample_clip());
        assert!(lib.remove(1));
        assert!(lib.is_empty());
    }

    #[test]
    fn test_library_remove_nonexistent() {
        let mut lib = CompoundClipLibrary::new();
        assert!(!lib.remove(999));
    }

    #[test]
    fn test_library_clips_for_timeline() {
        let mut lib = CompoundClipLibrary::new();
        lib.add(CompoundClip::new(10, "A", 42, 0, 100));
        lib.add(CompoundClip::new(11, "B", 42, 100, 200));
        lib.add(CompoundClip::new(12, "C", 99, 0, 50));
        let clips = lib.clips_for_timeline(42);
        assert_eq!(clips.len(), 2);
    }
}
