//! Undo/redo stack for playlist editing operations.
//!
//! This module provides a command pattern implementation that records
//! playlist mutations and allows reversing or re-applying them with full
//! fidelity.  The `UndoStack` has a configurable capacity so that very old
//! history entries are automatically discarded.

use super::item::{ItemMetadata, PlaylistItem};
use super::manager::Playlist;
use crate::{PlaylistError, Result};
use std::collections::VecDeque;

// ── TrackEntry ────────────────────────────────────────────────────────────────

/// A lightweight representation of a track used in undo/redo commands.
///
/// Keeping a full clone of `PlaylistItem` here is intentional: the undo
/// system must be able to reconstruct exact prior states without relying on
/// any external storage.
#[derive(Debug, Clone)]
pub struct TrackEntry {
    /// The full `PlaylistItem` snapshot captured at command creation time.
    pub item: PlaylistItem,
}

impl TrackEntry {
    /// Creates a new `TrackEntry` from a `PlaylistItem`.
    #[must_use]
    pub fn new(item: PlaylistItem) -> Self {
        Self { item }
    }
}

// ── PlaylistCommand ───────────────────────────────────────────────────────────

/// An atomic, reversible playlist editing operation.
///
/// Each variant stores exactly the information required both to *apply* the
/// operation and to *undo* it.  The `execute` / `undo` logic lives on
/// [`PlaylistEditor`] rather than here so that the command types remain pure
/// data.
#[derive(Debug, Clone)]
pub enum PlaylistCommand {
    /// Append a track to the end of the playlist.
    Add(TrackEntry),

    /// Remove the track at `index`.  The removed `TrackEntry` is stored here
    /// when the command is pushed onto the undo stack so it can be
    /// re-inserted on undo.
    Remove {
        /// Target index.
        index: usize,
        /// The item that was removed (populated by `execute`).
        removed: Option<TrackEntry>,
    },

    /// Move the track at `from` to position `to`.
    Move {
        /// Original index.
        from: usize,
        /// Destination index.
        to: usize,
    },

    /// Replace the metadata of the track at `index`.
    UpdateMetadata {
        /// Target track index.
        index: usize,
        /// New metadata to apply.
        new_metadata: ItemMetadata,
        /// Snapshot of the old metadata (populated by `execute`).
        old_metadata: Option<ItemMetadata>,
    },

    /// Enable or disable the track at `index`.
    SetEnabled {
        /// Target track index.
        index: usize,
        /// New enabled state.
        enabled: bool,
        /// Previous enabled state (populated by `execute`).
        previous: Option<bool>,
    },
}

impl PlaylistCommand {
    /// Convenience constructor for an `Add` command.
    #[must_use]
    pub fn add(item: PlaylistItem) -> Self {
        Self::Add(TrackEntry::new(item))
    }

    /// Convenience constructor for a `Remove` command.
    #[must_use]
    pub fn remove(index: usize) -> Self {
        Self::Remove {
            index,
            removed: None,
        }
    }

    /// Convenience constructor for a `Move` command.
    #[must_use]
    pub fn move_track(from: usize, to: usize) -> Self {
        Self::Move { from, to }
    }

    /// Convenience constructor for an `UpdateMetadata` command.
    #[must_use]
    pub fn update_metadata(index: usize, new_metadata: ItemMetadata) -> Self {
        Self::UpdateMetadata {
            index,
            new_metadata,
            old_metadata: None,
        }
    }

    /// Convenience constructor for a `SetEnabled` command.
    #[must_use]
    pub fn set_enabled(index: usize, enabled: bool) -> Self {
        Self::SetEnabled {
            index,
            enabled,
            previous: None,
        }
    }
}

// ── UndoStack ─────────────────────────────────────────────────────────────────

/// Fixed-capacity double-ended queue holding applied commands for undo/redo.
///
/// When the capacity is reached the oldest undo entry is silently discarded.
#[derive(Debug)]
pub struct UndoStack {
    /// Commands that have been executed (oldest at front, newest at back).
    past: VecDeque<PlaylistCommand>,
    /// Commands that have been undone and can be re-done (most-recent first).
    future: VecDeque<PlaylistCommand>,
    /// Maximum number of commands retained in *each* of `past` and `future`.
    capacity: usize,
}

impl UndoStack {
    /// Creates a new `UndoStack` with the given capacity.
    ///
    /// `capacity` is the maximum number of undo steps retained.  When the
    /// limit is reached the oldest step is evicted.  A capacity of `0`
    /// effectively disables undo/redo.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            past: VecDeque::with_capacity(capacity.min(4096)),
            future: VecDeque::with_capacity(capacity.min(4096)),
            capacity,
        }
    }

    /// Returns the number of available undo steps.
    #[must_use]
    pub fn undo_depth(&self) -> usize {
        self.past.len()
    }

    /// Returns the number of available redo steps.
    #[must_use]
    pub fn redo_depth(&self) -> usize {
        self.future.len()
    }

    /// Returns `true` if there are operations that can be undone.
    #[must_use]
    pub fn can_undo(&self) -> bool {
        !self.past.is_empty()
    }

    /// Returns `true` if there are operations that can be re-done.
    #[must_use]
    pub fn can_redo(&self) -> bool {
        !self.future.is_empty()
    }

    /// Push a command onto the undo history after a fresh edit (not a redo).
    ///
    /// Clears the redo stack because a new edit invalidates future history,
    /// just like most text editors behave.
    fn push_past_invalidating_redo(&mut self, cmd: PlaylistCommand) {
        self.future.clear();
        self.past.push_back(cmd);
        if self.capacity > 0 && self.past.len() > self.capacity {
            self.past.pop_front();
        }
    }

    /// Push a command onto the undo history after a redo operation.
    ///
    /// Unlike [`push_past_invalidating_redo`], this does **not** clear the
    /// redo stack because the remaining redo entries are still valid.
    fn push_past_from_redo(&mut self, cmd: PlaylistCommand) {
        self.past.push_back(cmd);
        if self.capacity > 0 && self.past.len() > self.capacity {
            self.past.pop_front();
        }
    }

    /// Pop the most-recently applied command for undoing.
    fn pop_past(&mut self) -> Option<PlaylistCommand> {
        self.past.pop_back()
    }

    /// Push a command onto the redo stack (after an undo).
    fn push_future(&mut self, cmd: PlaylistCommand) {
        self.future.push_back(cmd);
        if self.capacity > 0 && self.future.len() > self.capacity {
            self.future.pop_front();
        }
    }

    /// Pop the most-recently undone command for re-doing.
    fn pop_future(&mut self) -> Option<PlaylistCommand> {
        self.future.pop_back()
    }

    /// Clear all undo and redo history.
    pub fn clear(&mut self) {
        self.past.clear();
        self.future.clear();
    }
}

// ── PlaylistEditor ────────────────────────────────────────────────────────────

/// A wrapper around [`Playlist`] that records every mutation so it can be
/// undone or re-done.
///
/// All editing operations go through `execute`, which applies the command to
/// the inner playlist *and* stores a reversible snapshot on the undo stack.
///
/// # Example
///
/// ```
/// use oximedia_playlist::playlist::{Playlist, PlaylistItem, PlaylistType};
/// use oximedia_playlist::playlist::undo::{PlaylistCommand, PlaylistEditor};
/// use std::time::Duration;
///
/// let playlist = Playlist::new("demo", PlaylistType::Linear);
/// let mut editor = PlaylistEditor::new(playlist, 100);
///
/// let item = PlaylistItem::new("track1.mxf").with_duration(Duration::from_secs(60));
/// editor.execute(PlaylistCommand::add(item)).expect("execute add");
///
/// assert_eq!(editor.playlist().len(), 1);
/// editor.undo().expect("undo add");
/// assert_eq!(editor.playlist().len(), 0);
/// ```
#[derive(Debug)]
pub struct PlaylistEditor {
    playlist: Playlist,
    stack: UndoStack,
}

impl PlaylistEditor {
    /// Creates a new editor wrapping `playlist` with an undo history of
    /// `capacity` steps.
    #[must_use]
    pub fn new(playlist: Playlist, capacity: usize) -> Self {
        Self {
            playlist,
            stack: UndoStack::new(capacity),
        }
    }

    /// Returns an immutable reference to the underlying playlist.
    #[must_use]
    pub fn playlist(&self) -> &Playlist {
        &self.playlist
    }

    /// Returns a mutable reference to the underlying playlist.
    ///
    /// Mutations made through this reference are **not** recorded on the
    /// undo stack.  Prefer [`execute`](Self::execute) for tracked edits.
    #[must_use]
    pub fn playlist_mut(&mut self) -> &mut Playlist {
        &mut self.playlist
    }

    /// Returns a reference to the undo/redo stack (read-only).
    #[must_use]
    pub fn stack(&self) -> &UndoStack {
        &self.stack
    }

    /// Applies `command` to the playlist and records it on the undo stack.
    ///
    /// # Errors
    ///
    /// Returns `Err` when the command cannot be applied (e.g. index out of
    /// bounds).  In that case the playlist is left unchanged and nothing is
    /// pushed onto the stack.
    pub fn execute(&mut self, command: PlaylistCommand) -> Result<()> {
        let recorded = apply_command(&mut self.playlist, command)?;
        self.stack.push_past_invalidating_redo(recorded);
        Ok(())
    }

    /// Reverts the most recently applied command.
    ///
    /// # Errors
    ///
    /// Returns `Err` if there is nothing to undo or if reversal fails.
    pub fn undo(&mut self) -> Result<()> {
        let cmd = self
            .stack
            .pop_past()
            .ok_or_else(|| PlaylistError::InvalidItem("Nothing to undo".to_string()))?;

        let reverted = reverse_command(&mut self.playlist, cmd)?;
        self.stack.push_future(reverted);
        Ok(())
    }

    /// Re-applies the most recently undone command.
    ///
    /// # Errors
    ///
    /// Returns `Err` if there is nothing to redo or if re-application fails.
    pub fn redo(&mut self) -> Result<()> {
        let cmd = self
            .stack
            .pop_future()
            .ok_or_else(|| PlaylistError::InvalidItem("Nothing to redo".to_string()))?;

        let recorded = apply_command(&mut self.playlist, cmd)?;
        // Use push_past_from_redo so we do NOT clear the remaining redo entries.
        self.stack.push_past_from_redo(recorded);
        Ok(())
    }

    /// Returns `true` if there are operations that can be undone.
    #[must_use]
    pub fn can_undo(&self) -> bool {
        self.stack.can_undo()
    }

    /// Returns `true` if there are operations that can be re-done.
    #[must_use]
    pub fn can_redo(&self) -> bool {
        self.stack.can_redo()
    }
}

// ── Internal command application ──────────────────────────────────────────────

/// Apply `command` to `playlist`, filling in any reverse-data fields, and
/// return the enriched command suitable for storage in the undo stack.
fn apply_command(playlist: &mut Playlist, command: PlaylistCommand) -> Result<PlaylistCommand> {
    match command {
        PlaylistCommand::Add(entry) => {
            playlist.add_item(entry.item.clone());
            Ok(PlaylistCommand::Add(entry))
        }

        PlaylistCommand::Remove { index, .. } => {
            let item = playlist.remove_item(index)?;
            Ok(PlaylistCommand::Remove {
                index,
                removed: Some(TrackEntry::new(item)),
            })
        }

        PlaylistCommand::Move { from, to } => {
            move_item(playlist, from, to)?;
            Ok(PlaylistCommand::Move { from, to })
        }

        PlaylistCommand::UpdateMetadata {
            index,
            new_metadata,
            ..
        } => {
            let item = playlist.items.get_mut(index).ok_or_else(|| {
                PlaylistError::InvalidItem(format!("Index {index} out of bounds"))
            })?;
            let old_metadata = item.metadata.clone();
            item.metadata = new_metadata.clone();
            Ok(PlaylistCommand::UpdateMetadata {
                index,
                new_metadata,
                old_metadata: Some(old_metadata),
            })
        }

        PlaylistCommand::SetEnabled { index, enabled, .. } => {
            let item = playlist.items.get_mut(index).ok_or_else(|| {
                PlaylistError::InvalidItem(format!("Index {index} out of bounds"))
            })?;
            let previous = item.enabled;
            item.enabled = enabled;
            Ok(PlaylistCommand::SetEnabled {
                index,
                enabled,
                previous: Some(previous),
            })
        }
    }
}

/// Reverse `command` (which was previously applied) and apply the reversal to
/// `playlist`.  Returns the reversed form for pushing onto the redo stack.
fn reverse_command(playlist: &mut Playlist, command: PlaylistCommand) -> Result<PlaylistCommand> {
    match command {
        PlaylistCommand::Add(_) => {
            // Undo an Add = remove the last item that was appended.
            let last_index = playlist.items.len().checked_sub(1).ok_or_else(|| {
                PlaylistError::InvalidItem("Playlist is empty on undo".to_string())
            })?;
            let item = playlist.remove_item(last_index)?;
            // Return as Add so that re-applying it (redo) appends again.
            Ok(PlaylistCommand::Add(TrackEntry::new(item)))
        }

        PlaylistCommand::Remove { index, removed } => {
            // Undo a Remove = re-insert the stored item at the original index.
            let entry = removed.ok_or_else(|| {
                PlaylistError::InvalidItem("Remove command has no stored item".to_string())
            })?;
            playlist.insert_item(index, entry.item.clone())?;
            // Return without the stored item so re-applying removes it again.
            Ok(PlaylistCommand::Remove {
                index,
                removed: Some(entry),
            })
        }

        PlaylistCommand::Move { from, to } => {
            // Undo a Move(from→to) = Move(to→from).
            move_item(playlist, to, from)?;
            Ok(PlaylistCommand::Move { from: to, to: from })
        }

        PlaylistCommand::UpdateMetadata {
            index,
            new_metadata,
            old_metadata,
        } => {
            let prev = old_metadata.ok_or_else(|| {
                PlaylistError::InvalidItem("UpdateMetadata command has no old metadata".to_string())
            })?;
            let item = playlist.items.get_mut(index).ok_or_else(|| {
                PlaylistError::InvalidItem(format!("Index {index} out of bounds"))
            })?;
            // Swap: restore old and keep new for potential redo.
            item.metadata = prev.clone();
            Ok(PlaylistCommand::UpdateMetadata {
                index,
                new_metadata: prev,
                old_metadata: Some(new_metadata),
            })
        }

        PlaylistCommand::SetEnabled {
            index,
            enabled,
            previous,
        } => {
            let prev = previous.ok_or_else(|| {
                PlaylistError::InvalidItem("SetEnabled command has no previous state".to_string())
            })?;
            let item = playlist.items.get_mut(index).ok_or_else(|| {
                PlaylistError::InvalidItem(format!("Index {index} out of bounds"))
            })?;
            item.enabled = prev;
            Ok(PlaylistCommand::SetEnabled {
                index,
                enabled: prev,
                previous: Some(enabled),
            })
        }
    }
}

/// Move item at `from` to position `to` in the playlist.
///
/// The semantics match "remove then insert-before": after the operation the
/// item that was at `from` will be at index `to` (if `to > from` the index is
/// taken *before* the removal to avoid off-by-one surprises).
fn move_item(playlist: &mut Playlist, from: usize, to: usize) -> Result<()> {
    let len = playlist.items.len();
    if from >= len {
        return Err(PlaylistError::InvalidItem(format!(
            "Move 'from' index {from} out of bounds (len={len})"
        )));
    }
    if to >= len {
        return Err(PlaylistError::InvalidItem(format!(
            "Move 'to' index {to} out of bounds (len={len})"
        )));
    }
    if from == to {
        return Ok(());
    }
    let item = playlist.items.remove(from);
    // After removal the effective insertion index may shift.
    let insert_at = if to > from { to - 1 } else { to };
    // Clamp to the (now shorter) length.
    let insert_at = insert_at.min(playlist.items.len());
    playlist.items.insert(insert_at, item);
    Ok(())
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::playlist::{Playlist, PlaylistItem, PlaylistType};
    use std::time::Duration;

    fn make_editor(capacity: usize) -> PlaylistEditor {
        PlaylistEditor::new(Playlist::new("test", PlaylistType::Linear), capacity)
    }

    fn item(name: &str, secs: u64) -> PlaylistItem {
        PlaylistItem::new(format!("{name}.mxf")).with_duration(Duration::from_secs(secs))
    }

    // ── Undo add ──────────────────────────────────────────────────────────────

    #[test]
    fn test_undo_add_removes_item() {
        let mut ed = make_editor(50);
        ed.execute(PlaylistCommand::add(item("track1", 60)))
            .expect("execute add");
        assert_eq!(ed.playlist().len(), 1);

        ed.undo().expect("undo add");
        assert_eq!(ed.playlist().len(), 0);
    }

    // ── Redo add ──────────────────────────────────────────────────────────────

    #[test]
    fn test_redo_restores_item() {
        let mut ed = make_editor(50);
        ed.execute(PlaylistCommand::add(item("track2", 120)))
            .expect("execute add");
        ed.undo().expect("undo add");
        assert_eq!(ed.playlist().len(), 0);

        ed.redo().expect("redo add");
        assert_eq!(ed.playlist().len(), 1);
    }

    // ── Undo remove ───────────────────────────────────────────────────────────

    #[test]
    fn test_undo_remove_restores_item() {
        let mut ed = make_editor(50);
        ed.execute(PlaylistCommand::add(item("track3", 30)))
            .expect("add");
        ed.execute(PlaylistCommand::remove(0)).expect("remove");
        assert_eq!(ed.playlist().len(), 0);

        ed.undo().expect("undo remove");
        assert_eq!(ed.playlist().len(), 1);
    }

    // ── Undo move ─────────────────────────────────────────────────────────────

    #[test]
    fn test_undo_move_restores_order() {
        let mut ed = make_editor(50);
        for i in 0..3_u64 {
            ed.execute(PlaylistCommand::add(item(&format!("t{i}"), 10 * (i + 1))))
                .expect("add");
        }
        // Original: [t0, t1, t2].  Move index 0 → index 2.
        ed.execute(PlaylistCommand::move_track(0, 2)).expect("move");

        let names_after_move: Vec<String> = ed
            .playlist()
            .items
            .iter()
            .map(|it| it.name.clone())
            .collect();

        ed.undo().expect("undo move");

        let names_after_undo: Vec<String> = ed
            .playlist()
            .items
            .iter()
            .map(|it| it.name.clone())
            .collect();

        // After undo the list should differ from the post-move state.
        assert_ne!(names_after_move, names_after_undo);
    }

    // ── Undo update metadata ──────────────────────────────────────────────────

    #[test]
    fn test_undo_update_metadata() {
        let mut ed = make_editor(50);
        ed.execute(PlaylistCommand::add(item("track4", 60)))
            .expect("add");

        let mut new_meta = ItemMetadata::default();
        new_meta.title = Some("New Title".to_string());

        ed.execute(PlaylistCommand::update_metadata(0, new_meta))
            .expect("update metadata");

        assert_eq!(
            ed.playlist().items[0].metadata.title.as_deref(),
            Some("New Title")
        );

        ed.undo().expect("undo metadata");
        // Original metadata had no title.
        assert!(ed.playlist().items[0].metadata.title.is_none());
    }

    // ── Undo set_enabled ──────────────────────────────────────────────────────

    #[test]
    fn test_undo_set_enabled() {
        let mut ed = make_editor(50);
        ed.execute(PlaylistCommand::add(item("track5", 60)))
            .expect("add");
        assert!(ed.playlist().items[0].enabled);

        ed.execute(PlaylistCommand::set_enabled(0, false))
            .expect("disable");
        assert!(!ed.playlist().items[0].enabled);

        ed.undo().expect("undo disable");
        assert!(ed.playlist().items[0].enabled);
    }

    // ── Multi-step undo/redo cycle ────────────────────────────────────────────

    #[test]
    fn test_multi_step_undo_redo() {
        let mut ed = make_editor(50);
        ed.execute(PlaylistCommand::add(item("a", 10))).expect("a");
        ed.execute(PlaylistCommand::add(item("b", 20))).expect("b");
        ed.execute(PlaylistCommand::add(item("c", 30))).expect("c");

        // Undo all three adds.
        ed.undo().expect("undo c");
        ed.undo().expect("undo b");
        ed.undo().expect("undo a");
        assert_eq!(ed.playlist().len(), 0);

        // Redo all three adds.
        ed.redo().expect("redo a");
        ed.redo().expect("redo b");
        ed.redo().expect("redo c");
        assert_eq!(ed.playlist().len(), 3);
    }

    // ── New edit clears redo ───────────────────────────────────────────────────

    #[test]
    fn test_new_edit_clears_redo() {
        let mut ed = make_editor(50);
        ed.execute(PlaylistCommand::add(item("x", 10))).expect("x");
        ed.undo().expect("undo x");
        assert!(ed.can_redo());

        // New edit should clear the redo stack.
        ed.execute(PlaylistCommand::add(item("y", 20))).expect("y");
        assert!(!ed.can_redo());
    }

    // ── Capacity eviction ─────────────────────────────────────────────────────

    #[test]
    fn test_capacity_limits_undo_depth() {
        let mut ed = make_editor(3);
        for i in 0..10_u64 {
            ed.execute(PlaylistCommand::add(item(&format!("t{i}"), 10)))
                .expect("add");
        }
        // Only the last 3 commands should be retained.
        assert!(ed.stack().undo_depth() <= 3);
    }

    // ── Error on empty undo ────────────────────────────────────────────────────

    #[test]
    fn test_undo_on_empty_stack_returns_err() {
        let mut ed = make_editor(50);
        assert!(ed.undo().is_err());
    }

    // ── Error on empty redo ────────────────────────────────────────────────────

    #[test]
    fn test_redo_on_empty_stack_returns_err() {
        let mut ed = make_editor(50);
        assert!(ed.redo().is_err());
    }

    // ── can_undo / can_redo flags ─────────────────────────────────────────────

    #[test]
    fn test_can_undo_can_redo_flags() {
        let mut ed = make_editor(50);
        assert!(!ed.can_undo());
        assert!(!ed.can_redo());

        ed.execute(PlaylistCommand::add(item("z", 5))).expect("add");
        assert!(ed.can_undo());
        assert!(!ed.can_redo());

        ed.undo().expect("undo");
        assert!(!ed.can_undo());
        assert!(ed.can_redo());
    }
}
