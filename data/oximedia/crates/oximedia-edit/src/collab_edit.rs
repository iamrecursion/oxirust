//! Collaborative editing support for the timeline editor.
//!
//! Provides operational-transform-based concurrent edit merging across multiple
//! peers sharing a single timeline session.

use std::time::SystemTime;

use crate::clip::ClipId;
use crate::error::{EditError, EditResult};

/// Unique identifier for a collaborating peer.
pub type PeerId = String;

// ─────────────────────────────────────────────────────────────────────────────
// EditOpType
// ─────────────────────────────────────────────────────────────────────────────

/// The kind of atomic edit operation applied to the shared timeline.
#[derive(Clone, Debug, PartialEq)]
pub enum EditOpType {
    /// Move a clip to a new timeline start position.
    MoveClip {
        /// Target clip.
        clip_id: ClipId,
        /// New timeline start position (timebase units).
        new_start: i64,
    },
    /// Trim a clip's source in/out points.
    TrimClip {
        /// Target clip.
        clip_id: ClipId,
        /// New source-in point.
        new_in: i64,
        /// New source-out point.
        new_out: i64,
    },
    /// Delete a clip from the timeline.
    DeleteClip {
        /// Target clip.
        clip_id: ClipId,
    },
    /// Insert a new clip at the given track and position.
    InsertClip {
        /// New clip identifier.
        clip_id: ClipId,
        /// Target track index.
        track_index: usize,
        /// Timeline start position.
        start: i64,
        /// Clip duration.
        duration: i64,
    },
    /// No-op placeholder used when an operation is suppressed by OT.
    NoOp,
}

// ─────────────────────────────────────────────────────────────────────────────
// EditOperation
// ─────────────────────────────────────────────────────────────────────────────

/// A single atomic edit operation carrying authorship and revision metadata.
#[derive(Clone, Debug)]
pub struct EditOperation {
    /// Globally unique operation identifier.
    pub op_id: String,
    /// The peer that originated this operation.
    pub peer_id: PeerId,
    /// Revision of the shared state this operation was generated against.
    pub revision: u64,
    /// The actual edit to apply.
    pub op_type: EditOpType,
    /// Wall-clock time at which the operation was created.
    pub timestamp: SystemTime,
}

impl EditOperation {
    /// Create a new edit operation.
    #[must_use]
    pub fn new(
        op_id: impl Into<String>,
        peer_id: PeerId,
        revision: u64,
        op_type: EditOpType,
    ) -> Self {
        Self {
            op_id: op_id.into(),
            peer_id,
            revision,
            op_type,
            timestamp: SystemTime::now(),
        }
    }

    /// Returns `true` when this operation is a no-op.
    #[must_use]
    pub fn is_noop(&self) -> bool {
        matches!(self.op_type, EditOpType::NoOp)
    }

    /// Clip ID targeted by this operation, if any.
    #[must_use]
    pub fn affected_clip(&self) -> Option<ClipId> {
        match &self.op_type {
            EditOpType::MoveClip { clip_id, .. }
            | EditOpType::TrimClip { clip_id, .. }
            | EditOpType::DeleteClip { clip_id }
            | EditOpType::InsertClip { clip_id, .. } => Some(*clip_id),
            EditOpType::NoOp => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SharedEditState
// ─────────────────────────────────────────────────────────────────────────────

/// Shared state tracking the authoritative operation log and revision counter.
#[derive(Debug)]
pub struct SharedEditState {
    /// Monotonically increasing revision number.
    revision: u64,
    /// Committed operations in order.
    operations: Vec<EditOperation>,
    /// Wall-clock time of the last applied operation.
    last_modified: SystemTime,
}

impl SharedEditState {
    /// Create a fresh shared state at revision 0.
    #[must_use]
    pub fn new() -> Self {
        Self {
            revision: 0,
            operations: Vec::new(),
            last_modified: SystemTime::now(),
        }
    }

    /// Apply an operation, advancing the revision.
    pub fn apply(&mut self, op: &EditOperation) {
        self.revision += 1;
        self.operations.push(op.clone());
        self.last_modified = SystemTime::now();
    }

    /// Current revision number.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Committed operation log.
    #[must_use]
    pub fn operations(&self) -> &[EditOperation] {
        &self.operations
    }

    /// Wall-clock time of the most recent applied operation.
    #[must_use]
    pub fn last_modified(&self) -> SystemTime {
        self.last_modified
    }
}

impl Default for SharedEditState {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CollabEditEvent
// ─────────────────────────────────────────────────────────────────────────────

/// Events emitted by a [`CollabSession`] to notify subscribers of activity.
#[derive(Clone, Debug)]
pub enum CollabEditEvent {
    /// A new peer joined the session.
    PeerJoined(PeerId),
    /// A peer left the session.
    PeerLeft(PeerId),
    /// An operation was successfully applied to the shared state.
    EditApplied(EditOperation),
    /// Two concurrent operations were merged by operational transform.
    ConflictResolved {
        /// The local operation.
        local: EditOperation,
        /// The incoming remote operation.
        remote: EditOperation,
        /// The resolved operation that was actually applied.
        result: EditOperation,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// CollabSession
// ─────────────────────────────────────────────────────────────────────────────

/// A collaborative editing session shared between multiple peers.
///
/// Uses operational transform (OT) to merge concurrent edits that were
/// generated against the same revision of the shared state.
pub struct CollabSession {
    /// Human-readable session identifier.
    pub session_id: String,
    /// Connected peers.
    pub peers: Vec<PeerId>,
    /// Authoritative shared edit state.
    pub shared_state: SharedEditState,
    /// The peer representing the local user.
    pub local_peer: PeerId,
    /// Operations submitted locally but not yet acknowledged by the server.
    pub pending_ops: Vec<EditOperation>,
    /// Operations that have been applied to the shared state.
    pub acknowledged_ops: Vec<EditOperation>,
    /// Pending events to be consumed by the caller.
    pending_events: Vec<CollabEditEvent>,
}

impl CollabSession {
    /// Create a new collaborative session.
    #[must_use]
    pub fn new(session_id: impl Into<String>, local_peer: PeerId) -> Self {
        Self {
            session_id: session_id.into(),
            peers: Vec::new(),
            shared_state: SharedEditState::new(),
            local_peer,
            pending_ops: Vec::new(),
            acknowledged_ops: Vec::new(),
            pending_events: Vec::new(),
        }
    }

    /// Add a peer to the session.
    pub fn add_peer(&mut self, peer: PeerId) {
        if !self.peers.contains(&peer) {
            self.pending_events
                .push(CollabEditEvent::PeerJoined(peer.clone()));
            self.peers.push(peer);
        }
    }

    /// Remove a peer from the session.
    pub fn remove_peer(&mut self, peer: &PeerId) {
        if let Some(pos) = self.peers.iter().position(|p| p == peer) {
            self.pending_events
                .push(CollabEditEvent::PeerLeft(peer.clone()));
            self.peers.remove(pos);
        }
    }

    /// Apply an incoming operation to the shared state.
    ///
    /// If the incoming operation was generated against the same revision as any
    /// pending local operations, OT is applied to resolve the conflict.
    /// Returns the list of transformed operations actually committed.
    pub fn apply_edit(&mut self, op: EditOperation) -> EditResult<Vec<EditOperation>> {
        // OT: if there are pending local ops at the same revision, transform them
        let mut op_to_apply = op.clone();
        let mut conflicts_resolved = false;

        // Collect indices of pending ops at the same revision to transform
        let mut transformed_pending: Vec<EditOperation> = Vec::new();
        for pending in &self.pending_ops {
            if pending.revision == op.revision {
                let resolved = self.ot_transform(pending.clone(), op.clone())?;
                transformed_pending.push(resolved);
                conflicts_resolved = true;
            }
        }

        // If we resolved conflicts, synthesise a merged op
        if conflicts_resolved {
            if let Some(last_resolved) = transformed_pending.last() {
                // The remote op becomes the result of the last resolution
                let result = last_resolved.clone();
                self.pending_events.push(CollabEditEvent::ConflictResolved {
                    local: self
                        .pending_ops
                        .last()
                        .cloned()
                        .unwrap_or_else(|| op.clone()),
                    remote: op.clone(),
                    result: result.clone(),
                });
                op_to_apply = result;
            }
        }

        if !op_to_apply.is_noop() {
            self.shared_state.apply(&op_to_apply);
            self.acknowledged_ops.push(op_to_apply.clone());
            self.pending_events
                .push(CollabEditEvent::EditApplied(op_to_apply.clone()));
        }

        Ok(vec![op_to_apply])
    }

    /// Merge two concurrent operations using operational transform.
    ///
    /// Both `local` and `remote` must have been generated against the same
    /// revision.  Returns the transformed remote operation that can be applied
    /// on top of the local one.
    pub fn merge_concurrent(
        &mut self,
        local: EditOperation,
        remote: EditOperation,
    ) -> EditResult<EditOperation> {
        self.ot_transform(local, remote)
    }

    /// Drain and return pending events.
    pub fn broadcast(&mut self) -> Vec<CollabEditEvent> {
        std::mem::take(&mut self.pending_events)
    }

    /// Peek at pending events without draining them.
    #[must_use]
    pub fn pending_events(&self) -> &[CollabEditEvent] {
        &self.pending_events
    }

    /// Acknowledge a locally-generated operation (moves it from pending to
    /// acknowledged and commits it to the shared state).
    pub fn acknowledge_local(&mut self, op_id: &str) -> EditResult<()> {
        let pos = self
            .pending_ops
            .iter()
            .position(|o| o.op_id == op_id)
            .ok_or_else(|| EditError::InvalidEdit(format!("op {op_id} not in pending list")))?;
        let op = self.pending_ops.remove(pos);
        self.shared_state.apply(&op);
        self.acknowledged_ops.push(op.clone());
        self.pending_events.push(CollabEditEvent::EditApplied(op));
        Ok(())
    }

    /// Enqueue a locally-generated operation (does not apply it yet).
    pub fn enqueue_local(&mut self, op: EditOperation) {
        self.pending_ops.push(op);
    }

    // ── Operational-transform core ────────────────────────────────────────────

    /// Core OT function: transform `remote` so it can be applied after `local`.
    fn ot_transform(
        &self,
        local: EditOperation,
        remote: EditOperation,
    ) -> EditResult<EditOperation> {
        // Only OT for concurrent ops (same base revision)
        if local.revision != remote.revision {
            return Ok(remote);
        }

        let transformed_type = match (&local.op_type, &remote.op_type) {
            // Two moves on the same clip: remote wins (idempotent last-writer-wins)
            (
                EditOpType::MoveClip { clip_id: l_id, .. },
                EditOpType::MoveClip {
                    clip_id: r_id,
                    new_start,
                },
            ) if l_id == r_id => EditOpType::MoveClip {
                clip_id: *r_id,
                new_start: *new_start,
            },

            // Delete beats Move on same clip — suppress the Move
            (
                EditOpType::DeleteClip { clip_id: l_id },
                EditOpType::MoveClip { clip_id: r_id, .. },
            ) if l_id == r_id => EditOpType::NoOp,

            // Move beats Delete: suppress the Delete if local already moved it
            (
                EditOpType::MoveClip { clip_id: l_id, .. },
                EditOpType::DeleteClip { clip_id: r_id },
            ) if l_id == r_id => EditOpType::NoOp,

            // Concurrent trims on same clip: local wins, suppress remote
            (
                EditOpType::TrimClip { clip_id: l_id, .. },
                EditOpType::TrimClip { clip_id: r_id, .. },
            ) if l_id == r_id => EditOpType::NoOp,

            // InsertClip id collision: bump the incoming clip_id via a simple hash
            (
                EditOpType::InsertClip { clip_id: l_id, .. },
                EditOpType::InsertClip {
                    clip_id: r_id,
                    track_index,
                    start,
                    duration,
                },
            ) if l_id == r_id => {
                // Use a deterministic but distinct id
                let new_id = r_id.wrapping_add(0x8000_0000);
                EditOpType::InsertClip {
                    clip_id: new_id,
                    track_index: *track_index,
                    start: *start,
                    duration: *duration,
                }
            }

            // Default: compose (keep remote as-is)
            _ => remote.op_type.clone(),
        };

        Ok(EditOperation {
            op_id: remote.op_id.clone(),
            peer_id: remote.peer_id.clone(),
            revision: remote.revision + 1,
            op_type: transformed_type,
            timestamp: remote.timestamp,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_op(id: &str, peer: &str, rev: u64, op_type: EditOpType) -> EditOperation {
        EditOperation::new(id, peer.to_string(), rev, op_type)
    }

    #[test]
    fn test_new_session() {
        let session = CollabSession::new("session-1", "peer-A".to_string());
        assert_eq!(session.session_id, "session-1");
        assert_eq!(session.local_peer, "peer-A");
        assert_eq!(session.shared_state.revision(), 0);
        assert!(session.peers.is_empty());
    }

    #[test]
    fn test_add_remove_peer() {
        let mut session = CollabSession::new("s1", "A".to_string());
        session.add_peer("B".to_string());
        session.add_peer("C".to_string());
        assert_eq!(session.peers.len(), 2);

        // Duplicate add is ignored
        session.add_peer("B".to_string());
        assert_eq!(session.peers.len(), 2);

        session.remove_peer(&"B".to_string());
        assert_eq!(session.peers.len(), 1);
        assert_eq!(session.peers[0], "C");
    }

    #[test]
    fn test_apply_edit_increments_revision() {
        let mut session = CollabSession::new("s1", "A".to_string());
        let op = make_op(
            "op1",
            "B",
            0,
            EditOpType::MoveClip {
                clip_id: 1,
                new_start: 500,
            },
        );
        session.apply_edit(op).expect("apply_edit should succeed");
        assert_eq!(session.shared_state.revision(), 1);
    }

    #[test]
    fn test_broadcast_drains_events() {
        let mut session = CollabSession::new("s1", "A".to_string());
        session.add_peer("B".to_string());
        let events = session.broadcast();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], CollabEditEvent::PeerJoined(_)));
        // Second broadcast should be empty
        assert!(session.broadcast().is_empty());
    }

    #[test]
    fn test_ot_delete_wins_over_move() {
        let mut session = CollabSession::new("s1", "A".to_string());
        let local = make_op("l1", "A", 0, EditOpType::DeleteClip { clip_id: 42 });
        let remote = make_op(
            "r1",
            "B",
            0,
            EditOpType::MoveClip {
                clip_id: 42,
                new_start: 1000,
            },
        );
        let result = session.merge_concurrent(local, remote).expect("merge ok");
        assert!(result.is_noop(), "delete should suppress concurrent move");
    }

    #[test]
    fn test_ot_concurrent_trims_suppressed() {
        let mut session = CollabSession::new("s1", "A".to_string());
        let local = make_op(
            "l1",
            "A",
            5,
            EditOpType::TrimClip {
                clip_id: 7,
                new_in: 0,
                new_out: 100,
            },
        );
        let remote = make_op(
            "r1",
            "B",
            5,
            EditOpType::TrimClip {
                clip_id: 7,
                new_in: 10,
                new_out: 90,
            },
        );
        let result = session.merge_concurrent(local, remote).expect("merge ok");
        assert!(
            result.is_noop(),
            "local trim should suppress concurrent remote trim"
        );
    }

    #[test]
    fn test_ot_non_concurrent_ops_pass_through() {
        let mut session = CollabSession::new("s1", "A".to_string());
        let local = make_op(
            "l1",
            "A",
            3,
            EditOpType::MoveClip {
                clip_id: 1,
                new_start: 0,
            },
        );
        let remote = make_op(
            "r1",
            "B",
            5,
            EditOpType::MoveClip {
                clip_id: 1,
                new_start: 200,
            },
        );
        let result = session.merge_concurrent(local, remote).expect("merge ok");
        // Different revisions → pass-through, no transform
        assert!(!result.is_noop());
        assert_eq!(result.revision, 5);
    }

    #[test]
    fn test_shared_edit_state_operations() {
        let mut state = SharedEditState::new();
        assert_eq!(state.revision(), 0);
        let op = make_op("o1", "A", 0, EditOpType::NoOp);
        state.apply(&op);
        assert_eq!(state.revision(), 1);
        assert_eq!(state.operations().len(), 1);
    }
}
