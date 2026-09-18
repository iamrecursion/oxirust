//! Clip types and management.
//!
//! A clip represents a segment of media (video, audio, or subtitle) that appears
//! on a timeline track. Clips can be trimmed, moved, and have effects applied.

use oximedia_core::Rational;
use std::path::PathBuf;
use std::sync::Arc;

use crate::effect::EffectStack;
use crate::error::EditResult;

/// Unique identifier for clips.
pub type ClipId = u64;

/// A media clip on the timeline.
#[derive(Clone, Debug)]
pub struct Clip {
    /// Unique clip identifier.
    pub id: ClipId,
    /// Clip type (video, audio, or subtitle).
    pub clip_type: ClipType,
    /// Source file path (if from file).
    pub source: Option<PathBuf>,
    /// Start position on timeline (in timebase units).
    pub timeline_start: i64,
    /// Duration on timeline (in timebase units).
    pub timeline_duration: i64,
    /// Timebase for this clip.
    pub timebase: Rational,
    /// Source in point (trim start, in source timebase units).
    pub source_in: i64,
    /// Source out point (trim end, in source timebase units).
    pub source_out: i64,
    /// Source timebase.
    pub source_timebase: Rational,
    /// Playback speed multiplier (1.0 = normal, 2.0 = 2x speed, 0.5 = slow motion).
    pub speed: f64,
    /// Reverse playback.
    pub reverse: bool,
    /// Effect stack applied to this clip.
    pub effects: EffectStack,
    /// Clip opacity/volume (0.0-1.0).
    pub opacity: f32,
    /// Clip is muted.
    pub muted: bool,
    /// Clip is locked (cannot be edited).
    pub locked: bool,
    /// User metadata.
    pub metadata: ClipMetadata,
}

impl Clip {
    /// Create a new clip.
    #[must_use]
    pub fn new(id: ClipId, clip_type: ClipType, timeline_start: i64, duration: i64) -> Self {
        let timebase = Rational::new(1, 1000); // Default to milliseconds
        Self {
            id,
            clip_type,
            source: None,
            timeline_start,
            timeline_duration: duration,
            timebase,
            source_in: 0,
            source_out: duration,
            source_timebase: timebase,
            speed: 1.0,
            reverse: false,
            effects: EffectStack::new(),
            opacity: 1.0,
            muted: false,
            locked: false,
            metadata: ClipMetadata::default(),
        }
    }

    /// Create a clip from a source file.
    #[must_use]
    pub fn from_source(
        id: ClipId,
        clip_type: ClipType,
        source: PathBuf,
        timeline_start: i64,
        duration: i64,
    ) -> Self {
        let mut clip = Self::new(id, clip_type, timeline_start, duration);
        clip.source = Some(source);
        clip
    }

    /// Get timeline end position.
    #[must_use]
    pub fn timeline_end(&self) -> i64 {
        self.timeline_start + self.timeline_duration
    }

    /// Check if this clip overlaps with a time range.
    #[must_use]
    pub fn overlaps(&self, start: i64, end: i64) -> bool {
        !(self.timeline_end() <= start || self.timeline_start >= end)
    }

    /// Check if this clip contains a timeline position.
    #[must_use]
    pub fn contains(&self, position: i64) -> bool {
        position >= self.timeline_start && position < self.timeline_end()
    }

    /// Trim the in point of the clip.
    pub fn trim_in(&mut self, delta: i64) -> EditResult<()> {
        let new_in = self.source_in + delta;
        if new_in < 0 || new_in >= self.source_out {
            return Err(crate::error::EditError::InvalidEdit(
                "Trim would make clip invalid".to_string(),
            ));
        }
        self.source_in = new_in;
        self.timeline_start += delta;
        self.timeline_duration -= delta;
        Ok(())
    }

    /// Trim the out point of the clip.
    pub fn trim_out(&mut self, delta: i64) -> EditResult<()> {
        let new_out = self.source_out + delta;
        if new_out <= self.source_in || new_out > self.max_source_duration() {
            return Err(crate::error::EditError::InvalidEdit(
                "Trim would make clip invalid".to_string(),
            ));
        }
        self.source_out = new_out;
        self.timeline_duration += delta;
        Ok(())
    }

    /// Get maximum source duration (accounting for speed).
    #[must_use]
    pub fn max_source_duration(&self) -> i64 {
        // This would normally come from the actual source media
        // For now, we'll use the source_out as a proxy
        self.source_out
    }

    /// Split the clip at a timeline position.
    ///
    /// Returns a new clip representing the second half.
    pub fn split_at(&mut self, position: i64, new_id: ClipId) -> EditResult<Clip> {
        if !self.contains(position) {
            return Err(crate::error::EditError::InvalidEdit(
                "Split position not in clip".to_string(),
            ));
        }

        let offset = position - self.timeline_start;
        let mut second_half = self.clone();
        second_half.id = new_id;
        second_half.timeline_start = position;
        second_half.timeline_duration = self.timeline_duration - offset;
        second_half.source_in = self.source_in + offset;

        // Adjust this clip
        self.timeline_duration = offset;
        self.source_out = self.source_in + offset;

        Ok(second_half)
    }

    /// Convert timeline position to source position.
    #[must_use]
    pub fn timeline_to_source(&self, timeline_pos: i64) -> i64 {
        if timeline_pos < self.timeline_start {
            return self.source_in;
        }
        if timeline_pos >= self.timeline_end() {
            return self.source_out;
        }

        let offset = timeline_pos - self.timeline_start;
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_precision_loss)]
        let source_offset = (offset as f64 * self.speed) as i64;

        if self.reverse {
            self.source_out - source_offset
        } else {
            self.source_in + source_offset
        }
    }

    /// Get the source duration affected by this clip.
    #[must_use]
    pub fn source_duration(&self) -> i64 {
        self.source_out - self.source_in
    }

    /// Check if this is a video clip.
    #[must_use]
    pub fn is_video(&self) -> bool {
        matches!(self.clip_type, ClipType::Video)
    }

    /// Check if this is an audio clip.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        matches!(self.clip_type, ClipType::Audio)
    }

    /// Check if this is a subtitle clip.
    #[must_use]
    pub fn is_subtitle(&self) -> bool {
        matches!(self.clip_type, ClipType::Subtitle)
    }
}

/// Type of media in a clip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClipType {
    /// Video clip.
    Video,
    /// Audio clip.
    Audio,
    /// Subtitle clip.
    Subtitle,
}

/// Reference-counted clip for sharing.
#[derive(Clone, Debug)]
pub struct ClipRef {
    inner: Arc<Clip>,
}

impl ClipRef {
    /// Create a new clip reference.
    #[must_use]
    pub fn new(clip: Clip) -> Self {
        Self {
            inner: Arc::new(clip),
        }
    }

    /// Get a reference to the clip.
    #[must_use]
    pub fn clip(&self) -> &Clip {
        &self.inner
    }

    /// Try to get mutable access to the clip.
    #[must_use]
    pub fn try_unwrap(self) -> Option<Clip> {
        Arc::try_unwrap(self.inner).ok()
    }

    /// Make a mutable copy if needed.
    #[must_use]
    pub fn make_mut(self) -> Clip {
        match Arc::try_unwrap(self.inner) {
            Ok(clip) => clip,
            Err(arc) => (*arc).clone(),
        }
    }

    /// Get the reference count.
    #[must_use]
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }
}

impl From<Clip> for ClipRef {
    fn from(clip: Clip) -> Self {
        Self::new(clip)
    }
}

/// Metadata for a clip.
#[derive(Clone, Debug, Default)]
pub struct ClipMetadata {
    /// Clip name.
    pub name: Option<String>,
    /// Clip color (for UI).
    pub color: Option<String>,
    /// User notes.
    pub notes: Option<String>,
    /// Custom tags.
    pub tags: Vec<String>,
}

/// Clip selection for editing operations.
#[derive(Clone, Debug)]
pub struct ClipSelection {
    /// Selected clips by ID.
    pub clips: Vec<ClipId>,
}

impl ClipSelection {
    /// Create a new empty selection.
    #[must_use]
    pub fn new() -> Self {
        Self { clips: Vec::new() }
    }

    /// Add a clip to the selection.
    pub fn add(&mut self, clip_id: ClipId) {
        if !self.clips.contains(&clip_id) {
            self.clips.push(clip_id);
        }
    }

    /// Remove a clip from the selection.
    pub fn remove(&mut self, clip_id: ClipId) {
        self.clips.retain(|&id| id != clip_id);
    }

    /// Clear the selection.
    pub fn clear(&mut self) {
        self.clips.clear();
    }

    /// Check if a clip is selected.
    #[must_use]
    pub fn contains(&self, clip_id: ClipId) -> bool {
        self.clips.contains(&clip_id)
    }

    /// Check if the selection is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.clips.is_empty()
    }

    /// Get the number of selected clips.
    #[must_use]
    pub fn len(&self) -> usize {
        self.clips.len()
    }
}

impl Default for ClipSelection {
    fn default() -> Self {
        Self::new()
    }
}

/// Clipboard for cut/copy/paste operations.
#[derive(Clone, Debug, Default)]
pub struct Clipboard {
    /// Clips in the clipboard.
    pub clips: Vec<Clip>,
}

impl Clipboard {
    /// Create a new empty clipboard.
    #[must_use]
    pub fn new() -> Self {
        Self { clips: Vec::new() }
    }

    /// Copy clips to clipboard.
    pub fn copy(&mut self, clips: Vec<Clip>) {
        self.clips = clips;
    }

    /// Cut clips to clipboard (same as copy, but caller removes from timeline).
    pub fn cut(&mut self, clips: Vec<Clip>) {
        self.copy(clips);
    }

    /// Check if clipboard is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.clips.is_empty()
    }

    /// Clear the clipboard.
    pub fn clear(&mut self) {
        self.clips.clear();
    }

    /// Get the time range of clips in the clipboard.
    #[must_use]
    pub fn time_range(&self) -> Option<(i64, i64)> {
        if self.clips.is_empty() {
            return None;
        }

        let min_start = self.clips.iter().map(|c| c.timeline_start).min()?;
        let max_end = self.clips.iter().map(Clip::timeline_end).max()?;
        Some((min_start, max_end))
    }
}
