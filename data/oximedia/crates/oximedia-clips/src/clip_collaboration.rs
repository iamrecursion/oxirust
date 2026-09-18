//! Multi-user clip annotation with Operational-Transform-based conflict resolution.
//!
//! This module provides a collaborative annotation system that allows multiple
//! users to annotate clips concurrently.  Concurrent edits are reconciled using
//! a last-writer-wins (LWW) strategy per field combined with Operational
//! Transform (OT) for free-text notes, ensuring convergence without data loss.
//!
//! # Design
//!
//! - Each user holds a `CollaborationSession` identified by `UserId`.
//! - Annotations are stored in `SharedAnnotationBoard` — a central in-memory
//!   structure that all sessions write to.
//! - Every change is wrapped in an `AnnotationOp` with a monotonically
//!   increasing `revision` counter.
//! - Conflicts are detected when two ops target the same `(clip_id, field)` at
//!   the same `base_revision`.  The op with the **higher** `user_id` (lexicographic)
//!   wins; the loser's op is rebased onto the winner's text.

use crate::clip::ClipId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Identifiers ─────────────────────────────────────────────────────────────

/// Opaque user identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct UserId(pub String);

impl UserId {
    /// Creates a new user ID from any string.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl std::fmt::Display for UserId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Opaque annotation operation ID.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OpId(pub u64);

// ─── Annotation fields ───────────────────────────────────────────────────────

/// The specific annotation field being edited.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnnotationField {
    /// Free-text note.
    Note,
    /// Comma-separated list of tags.
    Tags,
    /// Single-field text such as a scene description.
    SceneDescription,
    /// Custom named field.
    Custom(String),
}

// ─── Operations ──────────────────────────────────────────────────────────────

/// A text-level edit operation (insert or delete).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TextOp {
    /// Insert `text` at byte position `pos`.
    Insert {
        /// Byte position.
        pos: usize,
        /// Text to insert.
        text: String,
    },
    /// Delete `len` bytes starting at byte position `pos`.
    Delete {
        /// Start byte position.
        pos: usize,
        /// Number of bytes to delete.
        len: usize,
    },
    /// Replace the entire field with `text` (used for non-OT fields).
    Replace {
        /// New text.
        text: String,
    },
}

impl TextOp {
    /// Applies this operation to `current` text, returning the new text.
    ///
    /// Invalid positions are clamped to the string boundaries.
    #[must_use]
    pub fn apply(&self, current: &str) -> String {
        match self {
            Self::Insert { pos, text } => {
                let pos = (*pos).min(current.len());
                let mut s = current.to_string();
                s.insert_str(pos, text);
                s
            }
            Self::Delete { pos, len } => {
                let pos = (*pos).min(current.len());
                let end = (pos + len).min(current.len());
                let mut s = current.to_string();
                s.replace_range(pos..end, "");
                s
            }
            Self::Replace { text } => text.clone(),
        }
    }
}

/// A single collaborative annotation operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationOp {
    /// Unique operation ID (assigned by the board).
    pub id: OpId,
    /// The user who submitted this op.
    pub user_id: UserId,
    /// Target clip.
    pub clip_id: ClipId,
    /// Target field.
    pub field: AnnotationField,
    /// The text edit.
    pub op: TextOp,
    /// Document revision at which this op was submitted.
    pub base_revision: u64,
    /// Wall-clock time of submission.
    pub timestamp: DateTime<Utc>,
}

// ─── Annotation state ────────────────────────────────────────────────────────

/// Per-field annotation state stored in the board.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FieldState {
    /// Current text content.
    pub text: String,
    /// Revision counter for this field.
    pub revision: u64,
    /// Last user to touch this field.
    pub last_editor: Option<UserId>,
    /// History of operations applied to this field.
    pub history: Vec<AnnotationOp>,
}

impl FieldState {
    /// Applies an op to this field state, incrementing the revision.
    pub fn apply(&mut self, op: AnnotationOp) {
        self.text = op.op.apply(&self.text);
        self.revision += 1;
        self.last_editor = Some(op.user_id.clone());
        self.history.push(op);
    }
}

// ─── Conflict ────────────────────────────────────────────────────────────────

/// A conflict record created when two ops targeted the same field at the same
/// base revision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conflict {
    /// The op that "won" (was applied).
    pub winner: AnnotationOp,
    /// The op that "lost" and was rebased.
    pub loser_original: AnnotationOp,
    /// The rebased version of the losing op (actually applied after the winner).
    pub loser_rebased: AnnotationOp,
    /// When the conflict was detected.
    pub detected_at: DateTime<Utc>,
}

// ─── Board ───────────────────────────────────────────────────────────────────

/// Central shared annotation board.
///
/// Thread-safety note: this struct is not `Send + Sync` by default.  For
/// concurrent access wrap it in `Arc<Mutex<SharedAnnotationBoard>>`.
#[derive(Debug, Default)]
pub struct SharedAnnotationBoard {
    /// Map of (clip_id, field) → field state.
    fields: HashMap<(ClipId, String), FieldState>,
    /// Global operation counter.
    next_op_id: u64,
    /// Conflicts detected so far.
    pub conflicts: Vec<Conflict>,
}

impl SharedAnnotationBoard {
    /// Creates an empty board.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the current text for `(clip_id, field)`.
    #[must_use]
    pub fn get(&self, clip_id: &ClipId, field: &AnnotationField) -> Option<&str> {
        self.fields
            .get(&(*clip_id, Self::field_key(field)))
            .map(|fs| fs.text.as_str())
    }

    /// Returns the current revision for `(clip_id, field)`.
    #[must_use]
    pub fn revision(&self, clip_id: &ClipId, field: &AnnotationField) -> u64 {
        self.fields
            .get(&(*clip_id, Self::field_key(field)))
            .map_or(0, |fs| fs.revision)
    }

    /// Submits an operation to the board.
    ///
    /// Returns `Ok(applied_op)` if the op was applied cleanly, or
    /// `Ok(rebased_op)` if the op was rebased after a conflict.
    pub fn submit(&mut self, mut op: AnnotationOp) -> AnnotationOp {
        let key = (op.clip_id, Self::field_key(&op.field));
        let field_state = self.fields.entry(key.clone()).or_default();

        // Assign unique ID.
        op.id = OpId(self.next_op_id);
        self.next_op_id += 1;

        let current_rev = field_state.revision;

        if op.base_revision == current_rev {
            // Clean apply — no conflict.
            field_state.apply(op.clone());
            return op;
        }

        // Conflict: another op was applied since `base_revision`.
        // Last op in history whose revision > base_revision is the winner.
        let winner = field_state
            .history
            .iter()
            .rev()
            .find(|h| h.base_revision >= op.base_revision)
            .cloned();

        if let Some(winner_op) = winner {
            // Determine winner/loser by user ID (lexicographic).
            let (winner_final, mut loser_final) = if winner_op.user_id >= op.user_id {
                (winner_op.clone(), op.clone())
            } else {
                (op.clone(), winner_op.clone())
            };

            // Rebase the loser: adjust its op position based on winner's edit.
            let rebased_text_op = rebase_op(&loser_final.op, &winner_final.op);
            loser_final.op = rebased_text_op;
            loser_final.base_revision = current_rev;

            self.conflicts.push(Conflict {
                winner: winner_final,
                loser_original: op.clone(),
                loser_rebased: loser_final.clone(),
                detected_at: Utc::now(),
            });

            field_state.apply(loser_final.clone());
            return loser_final;
        }

        // No prior history to conflict with — apply directly.
        field_state.apply(op.clone());
        op
    }

    /// Returns all annotations for a clip as a `HashMap<field_key, text>`.
    #[must_use]
    pub fn annotations_for_clip(&self, clip_id: &ClipId) -> HashMap<String, String> {
        self.fields
            .iter()
            .filter(|((cid, _), _)| cid == clip_id)
            .map(|((_, field), state)| (field.clone(), state.text.clone()))
            .collect()
    }

    /// Returns the full history of ops for a `(clip_id, field)` pair.
    #[must_use]
    pub fn history(&self, clip_id: &ClipId, field: &AnnotationField) -> Vec<&AnnotationOp> {
        self.fields
            .get(&(*clip_id, Self::field_key(field)))
            .map_or(Vec::new(), |fs| fs.history.iter().collect())
    }

    fn field_key(field: &AnnotationField) -> String {
        match field {
            AnnotationField::Note => "note".to_string(),
            AnnotationField::Tags => "tags".to_string(),
            AnnotationField::SceneDescription => "scene_description".to_string(),
            AnnotationField::Custom(name) => format!("custom:{name}"),
        }
    }
}

// ─── OT rebase ───────────────────────────────────────────────────────────────

/// Rebases `loser` against `winner` using simple insertion-point adjustment.
///
/// This is a deliberately minimal OT implementation:
/// - If `winner` inserted text before `loser`'s position, shift `loser`'s
///   position forward by the inserted length.
/// - If `winner` deleted text before `loser`'s position, shift backward.
/// - For `Replace` ops, the rebase is a no-op (LWW).
fn rebase_op(loser: &TextOp, winner: &TextOp) -> TextOp {
    match (loser, winner) {
        (TextOp::Insert { pos: lp, text: lt }, TextOp::Insert { pos: wp, text: wt }) => {
            let new_pos = if *wp <= *lp { lp + wt.len() } else { *lp };
            TextOp::Insert {
                pos: new_pos,
                text: lt.clone(),
            }
        }
        (TextOp::Insert { pos: lp, text: lt }, TextOp::Delete { pos: wp, len: wl }) => {
            let new_pos = if *wp < *lp {
                lp.saturating_sub(*wl)
            } else {
                *lp
            };
            TextOp::Insert {
                pos: new_pos,
                text: lt.clone(),
            }
        }
        (TextOp::Delete { pos: lp, len: ll }, TextOp::Insert { pos: wp, text: wt }) => {
            let new_pos = if *wp <= *lp { lp + wt.len() } else { *lp };
            TextOp::Delete {
                pos: new_pos,
                len: *ll,
            }
        }
        (TextOp::Delete { pos: lp, len: ll }, TextOp::Delete { pos: wp, len: wl }) => {
            let new_pos = if *wp < *lp {
                lp.saturating_sub(*wl)
            } else {
                *lp
            };
            TextOp::Delete {
                pos: new_pos,
                len: *ll,
            }
        }
        // For Replace ops, last writer wins — return loser unchanged.
        _ => loser.clone(),
    }
}

// ─── Session ─────────────────────────────────────────────────────────────────

/// A collaborative editing session for a single user.
pub struct CollaborationSession {
    /// User owning this session.
    pub user_id: UserId,
    // The shared board is accessed externally via SharedAnnotationBoard.
}

impl CollaborationSession {
    /// Creates a new session for the given user.
    #[must_use]
    pub fn new(user_id: UserId) -> Self {
        Self { user_id }
    }

    /// Builds an `AnnotationOp` for a text insert operation.
    #[must_use]
    pub fn insert_op(
        &self,
        clip_id: ClipId,
        field: AnnotationField,
        pos: usize,
        text: impl Into<String>,
        base_revision: u64,
    ) -> AnnotationOp {
        AnnotationOp {
            id: OpId(0), // assigned by board
            user_id: self.user_id.clone(),
            clip_id,
            field,
            op: TextOp::Insert {
                pos,
                text: text.into(),
            },
            base_revision,
            timestamp: Utc::now(),
        }
    }

    /// Builds an `AnnotationOp` for a replace operation.
    #[must_use]
    pub fn replace_op(
        &self,
        clip_id: ClipId,
        field: AnnotationField,
        text: impl Into<String>,
        base_revision: u64,
    ) -> AnnotationOp {
        AnnotationOp {
            id: OpId(0),
            user_id: self.user_id.clone(),
            clip_id,
            field,
            op: TextOp::Replace { text: text.into() },
            base_revision,
            timestamp: Utc::now(),
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn new_clip() -> ClipId {
        crate::clip::Clip::new(PathBuf::from("/test.mov")).id
    }

    #[test]
    fn test_text_op_insert() {
        let op = TextOp::Insert {
            pos: 5,
            text: "XYZ".to_string(),
        };
        let result = op.apply("hello world");
        assert_eq!(result, "helloXYZ world");
    }

    #[test]
    fn test_text_op_delete() {
        let op = TextOp::Delete { pos: 5, len: 6 };
        let result = op.apply("hello world");
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_text_op_replace() {
        let op = TextOp::Replace {
            text: "brand new".to_string(),
        };
        let result = op.apply("old text");
        assert_eq!(result, "brand new");
    }

    #[test]
    fn test_board_clean_apply() {
        let mut board = SharedAnnotationBoard::new();
        let clip_id = new_clip();
        let session = CollaborationSession::new(UserId::new("alice"));

        let op = session.replace_op(clip_id, AnnotationField::Note, "Hello", 0);
        board.submit(op);

        assert_eq!(board.get(&clip_id, &AnnotationField::Note), Some("Hello"));
        assert_eq!(board.revision(&clip_id, &AnnotationField::Note), 1);
    }

    #[test]
    fn test_board_two_sequential_inserts() {
        let mut board = SharedAnnotationBoard::new();
        let clip_id = new_clip();

        let alice = CollaborationSession::new(UserId::new("alice"));
        let bob = CollaborationSession::new(UserId::new("bob"));

        // Alice sets the note.
        let op1 = alice.replace_op(clip_id, AnnotationField::Note, "Hello", 0);
        board.submit(op1);

        // Bob appends.
        let rev = board.revision(&clip_id, &AnnotationField::Note);
        let op2 = bob.insert_op(clip_id, AnnotationField::Note, 5, " World", rev);
        board.submit(op2);

        assert_eq!(
            board.get(&clip_id, &AnnotationField::Note),
            Some("Hello World")
        );
    }

    #[test]
    fn test_board_conflict_detection() {
        let mut board = SharedAnnotationBoard::new();
        let clip_id = new_clip();

        let alice = CollaborationSession::new(UserId::new("alice"));
        let bob = CollaborationSession::new(UserId::new("bob"));

        // Set initial note.
        let op0 = alice.replace_op(clip_id, AnnotationField::Note, "Base text", 0);
        board.submit(op0);

        // Both alice and bob submit ops at revision 1 (conflict).
        let op_alice = alice.replace_op(clip_id, AnnotationField::Note, "Alice edit", 1);
        let op_bob = bob.replace_op(clip_id, AnnotationField::Note, "Bob edit", 1);

        board.submit(op_alice);
        board.submit(op_bob);

        // There should be at least one conflict recorded.
        assert!(!board.conflicts.is_empty());
    }

    #[test]
    fn test_board_history_recorded() {
        let mut board = SharedAnnotationBoard::new();
        let clip_id = new_clip();
        let alice = CollaborationSession::new(UserId::new("alice"));

        let op1 = alice.replace_op(clip_id, AnnotationField::Note, "v1", 0);
        let op2 = alice.replace_op(clip_id, AnnotationField::Note, "v2", 1);

        board.submit(op1);
        let rev = board.revision(&clip_id, &AnnotationField::Note);
        let mut op2 = op2;
        op2.base_revision = rev;
        board.submit(op2);

        let hist = board.history(&clip_id, &AnnotationField::Note);
        assert_eq!(hist.len(), 2);
    }

    #[test]
    fn test_annotations_for_clip() {
        let mut board = SharedAnnotationBoard::new();
        let clip_id = new_clip();
        let alice = CollaborationSession::new(UserId::new("alice"));

        let op_note = alice.replace_op(clip_id, AnnotationField::Note, "Great take", 0);
        let op_tags = alice.replace_op(clip_id, AnnotationField::Tags, "interview,day", 0);

        board.submit(op_note);
        board.submit(op_tags);

        let annots = board.annotations_for_clip(&clip_id);
        assert_eq!(annots.get("note").map(|s| s.as_str()), Some("Great take"));
        assert_eq!(
            annots.get("tags").map(|s| s.as_str()),
            Some("interview,day")
        );
    }

    #[test]
    fn test_rebase_insert_after_prior_insert() {
        let winner = TextOp::Insert {
            pos: 0,
            text: "AAA".to_string(),
        };
        let loser = TextOp::Insert {
            pos: 3,
            text: "BBB".to_string(),
        };
        let rebased = rebase_op(&loser, &winner);
        // winner inserted 3 chars at pos 0 before loser's pos 3
        // so loser's position shifts to 3+3=6
        match rebased {
            TextOp::Insert { pos, .. } => assert_eq!(pos, 6),
            _ => panic!("Expected Insert"),
        }
    }
}
