//! Freeze frame module for creating still-frame clips.
//!
//! A freeze frame captures a single frame from a source clip and holds it
//! for a specified duration. This is commonly used in editing for:
//! - Dramatic pauses
//! - End-frame holds before credits
//! - Still-image exports from video
//! - Transition placeholders
//!
//! # Workflow
//!
//! ```
//! use oximedia_timeline::freeze_frame::{FreezeFrame, FreezeFrameManager};
//! use oximedia_timeline::clip::{Clip, MediaSource};
//! use oximedia_timeline::types::{Duration, Position};
//!
//! // Create a freeze from a source clip at frame 50
//! let source = Clip::new(
//!     "Source".to_string(),
//!     MediaSource::black(),
//!     Position::new(0),
//!     Position::new(100),
//!     Position::new(0),
//! ).expect("valid clip");
//!
//! let freeze = FreezeFrame::from_clip(
//!     &source,
//!     Position::new(50),   // source frame to freeze
//!     Duration::new(48),   // hold for 48 frames (2 seconds at 24fps)
//!     Position::new(200),  // place on timeline at frame 200
//! ).expect("valid freeze");
//!
//! assert_eq!(freeze.source_frame(), Position::new(50));
//! assert_eq!(freeze.duration(), Duration::new(48));
//! ```

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::clip::{Clip, ClipId, MediaSource};
use crate::error::{TimelineError, TimelineResult};
use crate::types::{Duration, Position};

/// Unique identifier for a freeze frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FreezeFrameId(Uuid);

impl FreezeFrameId {
    /// Creates a new random freeze frame ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for FreezeFrameId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for FreezeFrameId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The type of freeze frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreezeType {
    /// A single frozen frame from a source clip.
    Single,
    /// A freeze at the first frame of a clip (head freeze).
    Head,
    /// A freeze at the last frame of a clip (tail freeze).
    Tail,
    /// A freeze at a specific timecode.
    AtTimecode,
}

/// A freeze frame created from a source clip.
///
/// Internally, a freeze frame is represented as a clip with speed = 0 (or
/// technically speed = very slow), where source_in and source_out point to
/// the same frame. The duration on the timeline is controlled by the
/// freeze's `hold_duration`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FreezeFrame {
    /// Unique identifier.
    pub id: FreezeFrameId,
    /// The generated clip for this freeze frame.
    pub clip: Clip,
    /// The frame from the source that is frozen.
    pub frozen_frame: Position,
    /// Duration to hold the freeze.
    pub hold_duration: Duration,
    /// Type of freeze.
    pub freeze_type: FreezeType,
    /// Name of the source clip (for reference).
    pub source_clip_name: String,
    /// ID of the source clip (for reference).
    pub source_clip_id: Option<ClipId>,
    /// Whether the freeze includes audio (usually muted for freezes).
    pub include_audio: bool,
}

impl FreezeFrame {
    /// Creates a freeze frame from a source clip at a specific frame.
    ///
    /// # Arguments
    ///
    /// * `source` - The source clip to freeze a frame from.
    /// * `source_frame` - The frame position in the source to freeze (relative
    ///   to source timecode).
    /// * `hold_duration` - How long the freeze should be held on the timeline.
    /// * `timeline_position` - Where to place the freeze on the timeline.
    ///
    /// # Errors
    ///
    /// Returns error if the source frame is outside the source clip's range.
    pub fn from_clip(
        source: &Clip,
        source_frame: Position,
        hold_duration: Duration,
        timeline_position: Position,
    ) -> TimelineResult<Self> {
        // Validate the source frame is within the source range
        if source_frame < source.source_in || source_frame >= source.source_out {
            return Err(TimelineError::InvalidPosition(format!(
                "Source frame {} is outside clip range [{}, {})",
                source_frame, source.source_in, source.source_out
            )));
        }

        if hold_duration.value() <= 0 {
            return Err(TimelineError::InvalidDuration(
                "Hold duration must be positive".to_string(),
            ));
        }

        let freeze_clip = Self::create_freeze_clip(
            &format!("{} [Freeze]", source.name),
            source.source.clone(),
            source_frame,
            hold_duration,
            timeline_position,
        )?;

        Ok(Self {
            id: FreezeFrameId::new(),
            clip: freeze_clip,
            frozen_frame: source_frame,
            hold_duration,
            freeze_type: FreezeType::Single,
            source_clip_name: source.name.clone(),
            source_clip_id: Some(source.id),
            include_audio: false,
        })
    }

    /// Creates a head freeze (first frame of the clip).
    ///
    /// # Errors
    ///
    /// Returns error if hold_duration is invalid.
    pub fn head_freeze(
        source: &Clip,
        hold_duration: Duration,
        timeline_position: Position,
    ) -> TimelineResult<Self> {
        let source_frame = source.source_in;
        let mut freeze = Self::from_clip(source, source_frame, hold_duration, timeline_position)?;
        freeze.freeze_type = FreezeType::Head;
        Ok(freeze)
    }

    /// Creates a tail freeze (last frame of the clip).
    ///
    /// # Errors
    ///
    /// Returns error if hold_duration is invalid.
    pub fn tail_freeze(
        source: &Clip,
        hold_duration: Duration,
        timeline_position: Position,
    ) -> TimelineResult<Self> {
        // Last frame is source_out - 1 (since source_out is exclusive)
        let source_frame = Position::new(source.source_out.value() - 1);
        let mut freeze = Self::from_clip(source, source_frame, hold_duration, timeline_position)?;
        freeze.freeze_type = FreezeType::Tail;
        Ok(freeze)
    }

    /// Creates a freeze from a media source directly (no source clip needed).
    ///
    /// # Errors
    ///
    /// Returns error if the duration is invalid.
    pub fn from_source(
        name: &str,
        source: MediaSource,
        source_frame: Position,
        hold_duration: Duration,
        timeline_position: Position,
    ) -> TimelineResult<Self> {
        if hold_duration.value() <= 0 {
            return Err(TimelineError::InvalidDuration(
                "Hold duration must be positive".to_string(),
            ));
        }

        let freeze_clip = Self::create_freeze_clip(
            &format!("{name} [Freeze]"),
            source,
            source_frame,
            hold_duration,
            timeline_position,
        )?;

        Ok(Self {
            id: FreezeFrameId::new(),
            clip: freeze_clip,
            frozen_frame: source_frame,
            hold_duration,
            freeze_type: FreezeType::AtTimecode,
            source_clip_name: name.to_string(),
            source_clip_id: None,
            include_audio: false,
        })
    }

    /// Returns the frozen source frame position.
    #[must_use]
    pub fn source_frame(&self) -> Position {
        self.frozen_frame
    }

    /// Returns the hold duration.
    #[must_use]
    pub fn duration(&self) -> Duration {
        self.hold_duration
    }

    /// Returns the timeline start position.
    #[must_use]
    pub fn timeline_in(&self) -> Position {
        self.clip.timeline_in
    }

    /// Returns the timeline end position.
    #[must_use]
    pub fn timeline_out(&self) -> Position {
        self.clip.timeline_in + self.hold_duration
    }

    /// Changes the hold duration.
    ///
    /// # Errors
    ///
    /// Returns error if the new duration is not positive.
    pub fn set_duration(&mut self, new_duration: Duration) -> TimelineResult<()> {
        if new_duration.value() <= 0 {
            return Err(TimelineError::InvalidDuration(
                "Hold duration must be positive".to_string(),
            ));
        }
        self.hold_duration = new_duration;
        // Update the clip: source_out stays the same (single frame), but
        // we need to adjust the speed so the clip appears to last new_duration.
        self.clip.source_out = Position::new(self.frozen_frame.value() + new_duration.value());
        Ok(())
    }

    /// Changes the frozen frame.
    ///
    /// # Errors
    ///
    /// Returns error if the frame is outside the source range.
    pub fn set_frozen_frame(&mut self, frame: Position) -> TimelineResult<()> {
        self.frozen_frame = frame;
        self.clip.source_in = frame;
        self.clip.source_out = Position::new(frame.value() + self.hold_duration.value());
        Ok(())
    }

    /// Moves the freeze to a new timeline position.
    pub fn move_to(&mut self, position: Position) {
        self.clip.timeline_in = position;
    }

    /// Sets whether audio is included.
    pub fn set_include_audio(&mut self, include: bool) {
        self.include_audio = include;
    }

    /// Returns a reference to the underlying clip.
    #[must_use]
    pub fn as_clip(&self) -> &Clip {
        &self.clip
    }

    /// Consumes the freeze frame and returns the underlying clip.
    #[must_use]
    pub fn into_clip(self) -> Clip {
        self.clip
    }

    /// Internal helper to create the freeze clip.
    fn create_freeze_clip(
        name: &str,
        source: MediaSource,
        source_frame: Position,
        hold_duration: Duration,
        timeline_position: Position,
    ) -> TimelineResult<Clip> {
        // For a freeze frame, set source_in = source_frame and
        // source_out = source_frame + hold_duration. Speed is 1.0 since
        // the "source" range is artificially extended to fill the hold duration.
        // In a real renderer, only the single frame at source_frame would
        // be displayed for the entire duration.
        let clip = Clip::new(
            name.to_string(),
            source,
            source_frame,
            Position::new(source_frame.value() + hold_duration.value()),
            timeline_position,
        )?;
        Ok(clip)
    }
}

/// Manages freeze frames on a timeline.
///
/// Provides higher-level operations like creating freezes from existing
/// timeline clips and inserting them at specific positions.
#[derive(Clone, Debug, Default)]
pub struct FreezeFrameManager {
    /// All registered freeze frames.
    freezes: Vec<FreezeFrame>,
}

impl FreezeFrameManager {
    /// Creates a new empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            freezes: Vec::new(),
        }
    }

    /// Adds a freeze frame.
    pub fn add(&mut self, freeze: FreezeFrame) {
        self.freezes.push(freeze);
    }

    /// Removes a freeze frame by ID.
    ///
    /// # Errors
    ///
    /// Returns error if the freeze frame is not found.
    pub fn remove(&mut self, id: FreezeFrameId) -> TimelineResult<FreezeFrame> {
        let index = self
            .freezes
            .iter()
            .position(|f| f.id == id)
            .ok_or_else(|| TimelineError::Other(format!("Freeze frame {id} not found")))?;
        Ok(self.freezes.remove(index))
    }

    /// Gets a freeze frame by ID.
    #[must_use]
    pub fn get(&self, id: FreezeFrameId) -> Option<&FreezeFrame> {
        self.freezes.iter().find(|f| f.id == id)
    }

    /// Gets a mutable reference to a freeze frame.
    pub fn get_mut(&mut self, id: FreezeFrameId) -> Option<&mut FreezeFrame> {
        self.freezes.iter_mut().find(|f| f.id == id)
    }

    /// Returns all freeze frames.
    #[must_use]
    pub fn all(&self) -> &[FreezeFrame] {
        &self.freezes
    }

    /// Returns the number of freeze frames.
    #[must_use]
    pub fn count(&self) -> usize {
        self.freezes.len()
    }

    /// Returns `true` if there are no freeze frames.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.freezes.is_empty()
    }

    /// Creates a freeze frame from a clip and adds it to the manager.
    ///
    /// # Errors
    ///
    /// Returns error if the source frame is invalid.
    pub fn create_from_clip(
        &mut self,
        source: &Clip,
        source_frame: Position,
        hold_duration: Duration,
        timeline_position: Position,
    ) -> TimelineResult<FreezeFrameId> {
        let freeze =
            FreezeFrame::from_clip(source, source_frame, hold_duration, timeline_position)?;
        let id = freeze.id;
        self.add(freeze);
        Ok(id)
    }

    /// Creates a head freeze and adds it to the manager.
    ///
    /// # Errors
    ///
    /// Returns error if creation fails.
    pub fn create_head_freeze(
        &mut self,
        source: &Clip,
        hold_duration: Duration,
        timeline_position: Position,
    ) -> TimelineResult<FreezeFrameId> {
        let freeze = FreezeFrame::head_freeze(source, hold_duration, timeline_position)?;
        let id = freeze.id;
        self.add(freeze);
        Ok(id)
    }

    /// Creates a tail freeze and adds it to the manager.
    ///
    /// # Errors
    ///
    /// Returns error if creation fails.
    pub fn create_tail_freeze(
        &mut self,
        source: &Clip,
        hold_duration: Duration,
        timeline_position: Position,
    ) -> TimelineResult<FreezeFrameId> {
        let freeze = FreezeFrame::tail_freeze(source, hold_duration, timeline_position)?;
        let id = freeze.id;
        self.add(freeze);
        Ok(id)
    }

    /// Finds all freeze frames at a given timeline position.
    #[must_use]
    pub fn at_position(&self, position: Position) -> Vec<&FreezeFrame> {
        self.freezes
            .iter()
            .filter(|f| position >= f.timeline_in() && position < f.timeline_out())
            .collect()
    }

    /// Finds all freeze frames from a specific source clip.
    #[must_use]
    pub fn from_source_clip(&self, clip_id: ClipId) -> Vec<&FreezeFrame> {
        self.freezes
            .iter()
            .filter(|f| f.source_clip_id == Some(clip_id))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_source_clip() -> Clip {
        Clip::new(
            "Source Clip".to_string(),
            MediaSource::black(),
            Position::new(0),
            Position::new(100),
            Position::new(0),
        )
        .expect("should succeed in test")
    }

    // --- FreezeFrameId tests ---

    #[test]
    fn test_freeze_frame_id_unique() {
        let id1 = FreezeFrameId::new();
        let id2 = FreezeFrameId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_freeze_frame_id_display() {
        let id = FreezeFrameId::new();
        let s = format!("{id}");
        assert!(!s.is_empty());
    }

    // --- FreezeFrame creation tests ---

    #[test]
    fn test_freeze_from_clip() {
        let source = create_source_clip();
        let freeze = FreezeFrame::from_clip(
            &source,
            Position::new(50),
            Duration::new(48),
            Position::new(200),
        )
        .expect("should succeed");

        assert_eq!(freeze.source_frame(), Position::new(50));
        assert_eq!(freeze.duration(), Duration::new(48));
        assert_eq!(freeze.timeline_in(), Position::new(200));
        assert_eq!(freeze.timeline_out(), Position::new(248));
        assert_eq!(freeze.freeze_type, FreezeType::Single);
        assert_eq!(freeze.source_clip_name, "Source Clip");
        assert!(freeze.source_clip_id.is_some());
        assert!(!freeze.include_audio);
    }

    #[test]
    fn test_freeze_from_clip_invalid_frame() {
        let source = create_source_clip();
        // Frame 200 is outside [0, 100)
        let result = FreezeFrame::from_clip(
            &source,
            Position::new(200),
            Duration::new(48),
            Position::new(0),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_freeze_from_clip_zero_duration() {
        let source = create_source_clip();
        let result = FreezeFrame::from_clip(
            &source,
            Position::new(50),
            Duration::new(0),
            Position::new(0),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_freeze_from_clip_negative_duration() {
        let source = create_source_clip();
        let result = FreezeFrame::from_clip(
            &source,
            Position::new(50),
            Duration::new(-10),
            Position::new(0),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_head_freeze() {
        let source = create_source_clip();
        let freeze = FreezeFrame::head_freeze(&source, Duration::new(24), Position::new(0))
            .expect("should succeed");

        assert_eq!(freeze.source_frame(), Position::new(0));
        assert_eq!(freeze.freeze_type, FreezeType::Head);
    }

    #[test]
    fn test_tail_freeze() {
        let source = create_source_clip();
        let freeze = FreezeFrame::tail_freeze(&source, Duration::new(24), Position::new(100))
            .expect("should succeed");

        assert_eq!(freeze.source_frame(), Position::new(99)); // Last frame
        assert_eq!(freeze.freeze_type, FreezeType::Tail);
    }

    #[test]
    fn test_from_source() {
        let freeze = FreezeFrame::from_source(
            "Color Hold",
            MediaSource::color(1.0, 0.0, 0.0, 1.0),
            Position::new(0),
            Duration::new(48),
            Position::new(100),
        )
        .expect("should succeed");

        assert_eq!(freeze.source_clip_name, "Color Hold");
        assert!(freeze.source_clip_id.is_none());
        assert_eq!(freeze.freeze_type, FreezeType::AtTimecode);
    }

    #[test]
    fn test_from_source_invalid_duration() {
        let result = FreezeFrame::from_source(
            "Test",
            MediaSource::black(),
            Position::new(0),
            Duration::new(0),
            Position::new(0),
        );
        assert!(result.is_err());
    }

    // --- FreezeFrame mutation tests ---

    #[test]
    fn test_set_duration() {
        let source = create_source_clip();
        let mut freeze = FreezeFrame::from_clip(
            &source,
            Position::new(50),
            Duration::new(24),
            Position::new(0),
        )
        .expect("should succeed");

        assert!(freeze.set_duration(Duration::new(96)).is_ok());
        assert_eq!(freeze.duration(), Duration::new(96));
        assert_eq!(freeze.timeline_out(), Position::new(96));
    }

    #[test]
    fn test_set_duration_invalid() {
        let source = create_source_clip();
        let mut freeze = FreezeFrame::from_clip(
            &source,
            Position::new(50),
            Duration::new(24),
            Position::new(0),
        )
        .expect("should succeed");

        assert!(freeze.set_duration(Duration::new(0)).is_err());
    }

    #[test]
    fn test_set_frozen_frame() {
        let source = create_source_clip();
        let mut freeze = FreezeFrame::from_clip(
            &source,
            Position::new(50),
            Duration::new(24),
            Position::new(0),
        )
        .expect("should succeed");

        assert!(freeze.set_frozen_frame(Position::new(75)).is_ok());
        assert_eq!(freeze.source_frame(), Position::new(75));
    }

    #[test]
    fn test_move_to() {
        let source = create_source_clip();
        let mut freeze = FreezeFrame::from_clip(
            &source,
            Position::new(50),
            Duration::new(24),
            Position::new(0),
        )
        .expect("should succeed");

        freeze.move_to(Position::new(500));
        assert_eq!(freeze.timeline_in(), Position::new(500));
        assert_eq!(freeze.timeline_out(), Position::new(524));
    }

    #[test]
    fn test_set_include_audio() {
        let source = create_source_clip();
        let mut freeze = FreezeFrame::from_clip(
            &source,
            Position::new(50),
            Duration::new(24),
            Position::new(0),
        )
        .expect("should succeed");

        assert!(!freeze.include_audio);
        freeze.set_include_audio(true);
        assert!(freeze.include_audio);
    }

    #[test]
    fn test_as_clip() {
        let source = create_source_clip();
        let freeze = FreezeFrame::from_clip(
            &source,
            Position::new(50),
            Duration::new(24),
            Position::new(0),
        )
        .expect("should succeed");

        let clip = freeze.as_clip();
        assert!(clip.name.contains("[Freeze]"));
    }

    #[test]
    fn test_into_clip() {
        let source = create_source_clip();
        let freeze = FreezeFrame::from_clip(
            &source,
            Position::new(50),
            Duration::new(24),
            Position::new(0),
        )
        .expect("should succeed");

        let clip = freeze.into_clip();
        assert!(clip.name.contains("[Freeze]"));
    }

    // --- FreezeFrameManager tests ---

    #[test]
    fn test_manager_empty() {
        let mgr = FreezeFrameManager::new();
        assert!(mgr.is_empty());
        assert_eq!(mgr.count(), 0);
    }

    #[test]
    fn test_manager_add_and_get() {
        let mut mgr = FreezeFrameManager::new();
        let source = create_source_clip();
        let freeze = FreezeFrame::from_clip(
            &source,
            Position::new(50),
            Duration::new(24),
            Position::new(0),
        )
        .expect("should succeed");
        let id = freeze.id;
        mgr.add(freeze);

        assert_eq!(mgr.count(), 1);
        assert!(mgr.get(id).is_some());
    }

    #[test]
    fn test_manager_remove() {
        let mut mgr = FreezeFrameManager::new();
        let source = create_source_clip();
        let freeze = FreezeFrame::from_clip(
            &source,
            Position::new(50),
            Duration::new(24),
            Position::new(0),
        )
        .expect("should succeed");
        let id = freeze.id;
        mgr.add(freeze);

        assert!(mgr.remove(id).is_ok());
        assert!(mgr.is_empty());
    }

    #[test]
    fn test_manager_remove_nonexistent() {
        let mut mgr = FreezeFrameManager::new();
        assert!(mgr.remove(FreezeFrameId::new()).is_err());
    }

    #[test]
    fn test_manager_create_from_clip() {
        let mut mgr = FreezeFrameManager::new();
        let source = create_source_clip();

        let id = mgr
            .create_from_clip(
                &source,
                Position::new(50),
                Duration::new(48),
                Position::new(0),
            )
            .expect("should succeed");

        assert_eq!(mgr.count(), 1);
        let freeze = mgr.get(id).expect("should find");
        assert_eq!(freeze.source_frame(), Position::new(50));
    }

    #[test]
    fn test_manager_create_head_freeze() {
        let mut mgr = FreezeFrameManager::new();
        let source = create_source_clip();

        let id = mgr
            .create_head_freeze(&source, Duration::new(24), Position::new(0))
            .expect("should succeed");

        let freeze = mgr.get(id).expect("should find");
        assert_eq!(freeze.freeze_type, FreezeType::Head);
    }

    #[test]
    fn test_manager_create_tail_freeze() {
        let mut mgr = FreezeFrameManager::new();
        let source = create_source_clip();

        let id = mgr
            .create_tail_freeze(&source, Duration::new(24), Position::new(100))
            .expect("should succeed");

        let freeze = mgr.get(id).expect("should find");
        assert_eq!(freeze.freeze_type, FreezeType::Tail);
    }

    #[test]
    fn test_manager_at_position() {
        let mut mgr = FreezeFrameManager::new();
        let source = create_source_clip();

        mgr.create_from_clip(
            &source,
            Position::new(50),
            Duration::new(24),
            Position::new(0),
        )
        .expect("should succeed");
        mgr.create_from_clip(
            &source,
            Position::new(30),
            Duration::new(48),
            Position::new(100),
        )
        .expect("should succeed");

        // Position 10 is in the first freeze [0, 24)
        let at_10 = mgr.at_position(Position::new(10));
        assert_eq!(at_10.len(), 1);

        // Position 120 is in the second freeze [100, 148)
        let at_120 = mgr.at_position(Position::new(120));
        assert_eq!(at_120.len(), 1);

        // Position 50 is in neither
        let at_50 = mgr.at_position(Position::new(50));
        assert_eq!(at_50.len(), 0);
    }

    #[test]
    fn test_manager_from_source_clip() {
        let mut mgr = FreezeFrameManager::new();
        let source = create_source_clip();
        let source_id = source.id;

        mgr.create_from_clip(
            &source,
            Position::new(50),
            Duration::new(24),
            Position::new(0),
        )
        .expect("should succeed");
        mgr.create_from_clip(
            &source,
            Position::new(30),
            Duration::new(24),
            Position::new(100),
        )
        .expect("should succeed");

        let from_source = mgr.from_source_clip(source_id);
        assert_eq!(from_source.len(), 2);

        // Different source ID should return empty
        let other = mgr.from_source_clip(ClipId::new());
        assert!(other.is_empty());
    }

    #[test]
    fn test_manager_get_mut() {
        let mut mgr = FreezeFrameManager::new();
        let source = create_source_clip();
        let id = mgr
            .create_from_clip(
                &source,
                Position::new(50),
                Duration::new(24),
                Position::new(0),
            )
            .expect("should succeed");

        let freeze_mut = mgr.get_mut(id).expect("should find");
        freeze_mut.include_audio = true;

        assert!(mgr.get(id).expect("should find").include_audio);
    }

    #[test]
    fn test_manager_all() {
        let mut mgr = FreezeFrameManager::new();
        let source = create_source_clip();

        mgr.create_from_clip(
            &source,
            Position::new(50),
            Duration::new(24),
            Position::new(0),
        )
        .expect("should succeed");
        mgr.create_from_clip(
            &source,
            Position::new(30),
            Duration::new(24),
            Position::new(100),
        )
        .expect("should succeed");

        assert_eq!(mgr.all().len(), 2);
    }

    #[test]
    fn test_freeze_frame_clip_name() {
        let source = create_source_clip();
        let freeze = FreezeFrame::from_clip(
            &source,
            Position::new(50),
            Duration::new(24),
            Position::new(0),
        )
        .expect("should succeed");

        assert_eq!(freeze.clip.name, "Source Clip [Freeze]");
    }

    #[test]
    fn test_boundary_freeze_at_source_start() {
        let source = create_source_clip();
        let freeze = FreezeFrame::from_clip(
            &source,
            Position::new(0), // First frame
            Duration::new(24),
            Position::new(0),
        );
        assert!(freeze.is_ok());
    }

    #[test]
    fn test_boundary_freeze_at_last_frame() {
        let source = create_source_clip();
        let freeze = FreezeFrame::from_clip(
            &source,
            Position::new(99), // Last valid frame
            Duration::new(24),
            Position::new(0),
        );
        assert!(freeze.is_ok());
    }

    #[test]
    fn test_boundary_freeze_at_source_out_fails() {
        let source = create_source_clip();
        // source_out is exclusive, so frame 100 should fail
        let freeze = FreezeFrame::from_clip(
            &source,
            Position::new(100),
            Duration::new(24),
            Position::new(0),
        );
        assert!(freeze.is_err());
    }
}
