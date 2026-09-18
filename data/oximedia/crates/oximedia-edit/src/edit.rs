//! Edit operations for the timeline.
//!
//! This module provides various editing operations including cut, copy, paste,
//! split, trim, and advanced edits like ripple, roll, slip, and slide.

use crate::clip::{Clip, ClipId, Clipboard};
use crate::error::{EditError, EditResult};
use crate::timeline::Timeline;

/// Edit mode for different types of editing operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditMode {
    /// Normal edit - no ripple.
    Normal,
    /// Ripple edit - all clips after the edit point move.
    Ripple,
    /// Roll edit - adjust the boundary between two clips.
    Roll,
    /// Slip edit - adjust in/out points without changing position.
    Slip,
    /// Slide edit - move clip without changing duration.
    Slide,
}

/// Timeline editing operations.
pub struct TimelineEditor {
    /// Clipboard for cut/copy/paste.
    clipboard: Clipboard,
    /// Edit history for undo/redo.
    history: EditHistory,
}

impl TimelineEditor {
    /// Create a new timeline editor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            clipboard: Clipboard::new(),
            history: EditHistory::new(),
        }
    }

    /// Cut selected clips to clipboard.
    pub fn cut(&mut self, timeline: &mut Timeline) -> EditResult<()> {
        let selected_ids: Vec<ClipId> = timeline.selection.clips.clone();
        if selected_ids.is_empty() {
            return Ok(());
        }

        let mut clips = Vec::new();
        for clip_id in &selected_ids {
            let clip = timeline.remove_clip(*clip_id)?;
            clips.push(clip);
        }

        self.clipboard.cut(clips);
        timeline.selection.clear();

        Ok(())
    }

    /// Copy selected clips to clipboard.
    pub fn copy(&mut self, timeline: &Timeline) -> EditResult<()> {
        let selected_ids: Vec<ClipId> = timeline.selection.clips.clone();
        if selected_ids.is_empty() {
            return Ok(());
        }

        let mut clips = Vec::new();
        for clip_id in &selected_ids {
            if let Some(clip) = timeline.get_clip(*clip_id) {
                clips.push(clip.clone());
            }
        }

        self.clipboard.copy(clips);

        Ok(())
    }

    /// Paste clips from clipboard at playhead position.
    pub fn paste(&mut self, timeline: &mut Timeline) -> EditResult<Vec<ClipId>> {
        if self.clipboard.is_empty() {
            return Ok(Vec::new());
        }

        let paste_position = timeline.playhead;
        let mut new_clip_ids = Vec::new();

        // Calculate offset based on first clip in clipboard
        let time_range = self.clipboard.time_range();
        let offset = if let Some((min_start, _)) = time_range {
            paste_position - min_start
        } else {
            0
        };

        // Group clips by track and find appropriate tracks
        let clipboard_clips = self.clipboard.clips.clone();
        for mut clip in clipboard_clips {
            clip.timeline_start += offset;

            // Find or create appropriate track
            let track_index = timeline
                .tracks
                .iter()
                .position(|t| t.track_type.matches_clip(clip.clip_type))
                .ok_or_else(|| {
                    EditError::InvalidEdit("No suitable track for clip type".to_string())
                })?;

            let clip_id = timeline.add_clip(track_index, clip)?;
            new_clip_ids.push(clip_id);
        }

        // Select pasted clips
        timeline.selection.clear();
        for clip_id in &new_clip_ids {
            timeline.selection.add(*clip_id);
        }

        Ok(new_clip_ids)
    }

    /// Split clip at playhead position.
    pub fn split_at_playhead(&mut self, timeline: &mut Timeline) -> EditResult<Vec<ClipId>> {
        let position = timeline.playhead;
        let mut new_clip_ids = Vec::new();

        // Find all clips at playhead position
        let clips_at_playhead: Vec<(usize, ClipId)> = timeline
            .tracks
            .iter()
            .enumerate()
            .filter_map(|(track_idx, track)| {
                track.get_clip_at(position).map(|clip| (track_idx, clip.id))
            })
            .collect();

        // Split each clip
        for (_track_idx, clip_id) in clips_at_playhead {
            let new_clip_id = self.split_clip(timeline, clip_id, position)?;
            new_clip_ids.push(new_clip_id);
        }

        Ok(new_clip_ids)
    }

    /// Split a specific clip at a position.
    pub fn split_clip(
        &mut self,
        timeline: &mut Timeline,
        clip_id: ClipId,
        position: i64,
    ) -> EditResult<ClipId> {
        let (track_index, _) = timeline
            .clip_map
            .get(&clip_id)
            .copied()
            .ok_or(EditError::ClipNotFound(clip_id))?;

        // Get next clip ID and increment before mutable borrowing
        let new_id = timeline.next_clip_id;
        timeline.next_clip_id += 1;

        // Get the clip and split it
        let clip = timeline
            .get_clip_mut(clip_id)
            .ok_or(EditError::ClipNotFound(clip_id))?;

        let second_half = clip.split_at(position, new_id)?;

        // Add the second half to the timeline
        timeline.add_clip(track_index, second_half)?;

        Ok(new_id)
    }

    /// Delete selected clips.
    pub fn delete_selection(&mut self, timeline: &mut Timeline) -> EditResult<()> {
        let selected_ids: Vec<ClipId> = timeline.selection.clips.clone();
        for clip_id in selected_ids {
            timeline.remove_clip(clip_id)?;
        }
        timeline.selection.clear();
        Ok(())
    }

    /// Trim clip in point.
    pub fn trim_in(
        &mut self,
        timeline: &mut Timeline,
        clip_id: ClipId,
        delta: i64,
    ) -> EditResult<()> {
        let clip = timeline
            .get_clip_mut(clip_id)
            .ok_or(EditError::ClipNotFound(clip_id))?;
        clip.trim_in(delta)?;
        Ok(())
    }

    /// Trim clip out point.
    pub fn trim_out(
        &mut self,
        timeline: &mut Timeline,
        clip_id: ClipId,
        delta: i64,
    ) -> EditResult<()> {
        let clip = timeline
            .get_clip_mut(clip_id)
            .ok_or(EditError::ClipNotFound(clip_id))?;
        clip.trim_out(delta)?;
        Ok(())
    }

    /// Ripple delete - delete clips and move following clips backward.
    pub fn ripple_delete(
        &mut self,
        timeline: &mut Timeline,
        track_index: usize,
        clip_id: ClipId,
    ) -> EditResult<()> {
        let track = timeline
            .get_track(track_index)
            .ok_or(EditError::InvalidTrackIndex(
                track_index,
                timeline.tracks.len(),
            ))?;

        let clip = track
            .clips
            .iter()
            .find(|c| c.id == clip_id)
            .ok_or(EditError::ClipNotFound(clip_id))?;

        let clip_start = clip.timeline_start;
        let clip_duration = clip.timeline_duration;

        // Get the track count before getting a mutable reference
        let track_count = timeline.tracks.len();

        // Remove the clip
        timeline.remove_clip(clip_id)?;

        // Move all following clips on this track backward
        let track = timeline
            .get_track_mut(track_index)
            .ok_or(EditError::InvalidTrackIndex(track_index, track_count))?;

        for clip in &mut track.clips {
            if clip.timeline_start >= clip_start {
                clip.timeline_start -= clip_duration;
            }
        }

        timeline.rebuild_clip_map();
        Ok(())
    }

    /// Ripple trim - trim a clip and move following clips.
    pub fn ripple_trim(
        &mut self,
        timeline: &mut Timeline,
        clip_id: ClipId,
        delta: i64,
        trim_end: bool,
    ) -> EditResult<()> {
        let (track_index, _) = timeline
            .clip_map
            .get(&clip_id)
            .copied()
            .ok_or(EditError::ClipNotFound(clip_id))?;

        let clip = timeline
            .get_clip_mut(clip_id)
            .ok_or(EditError::ClipNotFound(clip_id))?;

        let clip_end = clip.timeline_end();

        if trim_end {
            clip.trim_out(delta)?;
        } else {
            clip.trim_in(delta)?;
        }

        // Move following clips
        if trim_end {
            let track_count = timeline.tracks.len();
            let track = timeline
                .get_track_mut(track_index)
                .ok_or(EditError::InvalidTrackIndex(track_index, track_count))?;

            for clip in &mut track.clips {
                if clip.timeline_start >= clip_end {
                    clip.timeline_start += delta;
                }
            }
        }

        timeline.rebuild_clip_map();
        Ok(())
    }

    /// Roll edit - adjust the boundary between two adjacent clips.
    #[allow(clippy::similar_names)]
    pub fn roll_edit(
        &mut self,
        timeline: &mut Timeline,
        clip_a_id: ClipId,
        clip_b_id: ClipId,
        delta: i64,
    ) -> EditResult<()> {
        // Get both clips
        let clip_a = timeline
            .get_clip(clip_a_id)
            .ok_or(EditError::ClipNotFound(clip_a_id))?;
        let clip_b = timeline
            .get_clip(clip_b_id)
            .ok_or(EditError::ClipNotFound(clip_b_id))?;

        // Verify clips are adjacent
        if clip_a.timeline_end() != clip_b.timeline_start {
            return Err(EditError::InvalidEdit("Clips are not adjacent".to_string()));
        }

        // Trim first clip out point
        let clip_a = timeline
            .get_clip_mut(clip_a_id)
            .ok_or(EditError::ClipNotFound(clip_a_id))?;
        clip_a.trim_out(delta)?;

        // Trim second clip in point
        let clip_b = timeline
            .get_clip_mut(clip_b_id)
            .ok_or(EditError::ClipNotFound(clip_b_id))?;
        clip_b.trim_in(-delta)?;

        Ok(())
    }

    /// Slip edit - adjust in/out points without changing timeline position.
    pub fn slip_edit(
        &mut self,
        timeline: &mut Timeline,
        clip_id: ClipId,
        delta: i64,
    ) -> EditResult<()> {
        let clip = timeline
            .get_clip_mut(clip_id)
            .ok_or(EditError::ClipNotFound(clip_id))?;

        let new_in = clip.source_in + delta;
        let new_out = clip.source_out + delta;

        // Validate new range
        if new_in < 0 || new_out > clip.max_source_duration() {
            return Err(EditError::InvalidEdit(
                "Slip would exceed source bounds".to_string(),
            ));
        }

        clip.source_in = new_in;
        clip.source_out = new_out;

        Ok(())
    }

    /// Slide edit - move clip without changing duration, adjusting adjacent clips.
    pub fn slide_edit(
        &mut self,
        timeline: &mut Timeline,
        track_index: usize,
        clip_id: ClipId,
        delta: i64,
    ) -> EditResult<()> {
        let track = timeline
            .get_track(track_index)
            .ok_or(EditError::InvalidTrackIndex(
                track_index,
                timeline.tracks.len(),
            ))?;

        let clip_index = track
            .clips
            .iter()
            .position(|c| c.id == clip_id)
            .ok_or(EditError::ClipNotFound(clip_id))?;

        let clip = &track.clips[clip_index];
        let new_start = clip.timeline_start + delta;
        let new_end = clip.timeline_end() + delta;

        // Check if we can slide (need adjacent clips with enough room)
        if delta < 0 {
            // Sliding left
            if clip_index > 0 {
                let prev_clip = &track.clips[clip_index - 1];
                if prev_clip.timeline_end() > new_start {
                    return Err(EditError::InvalidEdit(
                        "Cannot slide: not enough room".to_string(),
                    ));
                }
            }
        } else if delta > 0 {
            // Sliding right
            if clip_index < track.clips.len() - 1 {
                let next_clip = &track.clips[clip_index + 1];
                if next_clip.timeline_start < new_end {
                    return Err(EditError::InvalidEdit(
                        "Cannot slide: not enough room".to_string(),
                    ));
                }
            }
        }

        // Perform the slide
        let clip = timeline
            .get_clip_mut(clip_id)
            .ok_or(EditError::ClipNotFound(clip_id))?;
        clip.timeline_start = new_start;

        Ok(())
    }

    /// Set clip speed.
    pub fn set_speed(
        &mut self,
        timeline: &mut Timeline,
        clip_id: ClipId,
        speed: f64,
    ) -> EditResult<()> {
        if speed <= 0.0 {
            return Err(EditError::InvalidEdit("Speed must be positive".to_string()));
        }

        let clip = timeline
            .get_clip_mut(clip_id)
            .ok_or(EditError::ClipNotFound(clip_id))?;

        clip.speed = speed;

        // Adjust timeline duration based on speed
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_precision_loss)]
        let new_duration = (clip.source_duration() as f64 / speed) as i64;
        clip.timeline_duration = new_duration;

        Ok(())
    }

    /// Reverse clip playback.
    pub fn reverse_clip(&mut self, timeline: &mut Timeline, clip_id: ClipId) -> EditResult<()> {
        let clip = timeline
            .get_clip_mut(clip_id)
            .ok_or(EditError::ClipNotFound(clip_id))?;
        clip.reverse = !clip.reverse;
        Ok(())
    }

    /// Get clipboard.
    #[must_use]
    pub fn clipboard(&self) -> &Clipboard {
        &self.clipboard
    }

    /// Get edit history.
    #[must_use]
    pub fn history(&self) -> &EditHistory {
        &self.history
    }
}

impl Default for TimelineEditor {
    fn default() -> Self {
        Self::new()
    }
}

/// Edit history for undo/redo operations.
#[derive(Debug)]
pub struct EditHistory {
    /// History stack.
    history: Vec<EditAction>,
    /// Current position in history.
    current: usize,
    /// Maximum history size.
    max_size: usize,
}

impl EditHistory {
    /// Create a new edit history.
    #[must_use]
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            current: 0,
            max_size: 100,
        }
    }

    /// Add an action to the history.
    pub fn push(&mut self, action: EditAction) {
        // Remove any actions after current position
        self.history.truncate(self.current);

        // Add new action
        self.history.push(action);
        self.current += 1;

        // Limit history size
        if self.history.len() > self.max_size {
            self.history.remove(0);
            self.current -= 1;
        }
    }

    /// Check if undo is available.
    #[must_use]
    pub fn can_undo(&self) -> bool {
        self.current > 0
    }

    /// Check if redo is available.
    #[must_use]
    pub fn can_redo(&self) -> bool {
        self.current < self.history.len()
    }

    /// Get the action to undo.
    pub fn undo(&mut self) -> Option<&EditAction> {
        if self.can_undo() {
            self.current -= 1;
            Some(&self.history[self.current])
        } else {
            None
        }
    }

    /// Get the action to redo.
    pub fn redo(&mut self) -> Option<&EditAction> {
        if self.can_redo() {
            let action = &self.history[self.current];
            self.current += 1;
            Some(action)
        } else {
            None
        }
    }

    /// Clear history.
    pub fn clear(&mut self) {
        self.history.clear();
        self.current = 0;
    }

    /// Get history size.
    #[must_use]
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Check if history is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }
}

impl Default for EditHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// An edit action that can be undone/redone.
#[derive(Clone, Debug)]
pub enum EditAction {
    /// Add clip.
    AddClip {
        /// Track index.
        track: usize,
        /// Clip data.
        clip: Clip,
    },
    /// Remove clip.
    RemoveClip {
        /// Track index.
        track: usize,
        /// Clip ID.
        clip_id: ClipId,
    },
    /// Move clip.
    MoveClip {
        /// Clip ID.
        clip_id: ClipId,
        /// Old position.
        old_start: i64,
        /// New position.
        new_start: i64,
    },
    /// Trim clip.
    TrimClip {
        /// Clip ID.
        clip_id: ClipId,
        /// Old in/out points.
        old_in: i64,
        /// Old out point.
        old_out: i64,
        /// New in point.
        new_in: i64,
        /// New out point.
        new_out: i64,
    },
    /// Split clip.
    SplitClip {
        /// Original clip ID.
        original_id: ClipId,
        /// New clip ID.
        new_id: ClipId,
        /// Split position.
        position: i64,
    },
}

/// Snap settings for timeline editing.
#[derive(Clone, Debug)]
pub struct SnapSettings {
    /// Enable snapping.
    pub enabled: bool,
    /// Snap to playhead.
    pub snap_to_playhead: bool,
    /// Snap to clip edges.
    pub snap_to_clips: bool,
    /// Snap to markers.
    pub snap_to_markers: bool,
    /// Snap threshold (in timebase units).
    pub threshold: i64,
}

impl Default for SnapSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            snap_to_playhead: true,
            snap_to_clips: true,
            snap_to_markers: true,
            threshold: 5,
        }
    }
}

impl SnapSettings {
    /// Check if a position should snap to a target.
    #[must_use]
    pub fn should_snap(&self, position: i64, target: i64) -> bool {
        if !self.enabled {
            return false;
        }
        (position - target).abs() <= self.threshold
    }

    /// Get snap position if within threshold.
    #[must_use]
    pub fn snap_position(&self, position: i64, targets: &[i64]) -> i64 {
        if !self.enabled {
            return position;
        }

        for &target in targets {
            if self.should_snap(position, target) {
                return target;
            }
        }

        position
    }
}
