//! Delta-based real-time synchronisation for annotation updates.
//!
//! Instead of broadcasting the full annotation state on every change,
//! this module tracks a compact sequence of [`AnnotationDelta`] operations
//! and transmits only the mutations that occurred since each client's last
//! acknowledged sequence number.
//!
//! # Architecture
//!
//! ```text
//! Client A                 DeltaBroadcaster               Client B
//!   |-- patch op -------->  |                               |
//!                           |-- DeltaMessage(seq=42) -----> |
//!                           |                               | (ack seq=42)
//!                           |<-- AckMessage(seq=42) --------|
//! ```
//!
//! The [`DeltaLog`] stores an append-only ring of at most `capacity` entries.
//! Clients that fall behind by more than `capacity` entries receive a
//! `DeltaMessage::Resync` signal, instructing them to fetch the full snapshot.

use crate::drawing::Annotation;
use crate::error::{ReviewError, ReviewResult};
use crate::DrawingId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Monotonically increasing sequence number for delta log entries.
pub type Seq = u64;

/// Identifier for a connected client.
pub type ClientId = String;

/// A single mutation applied to the annotation state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnnotationDelta {
    /// A new annotation was added.
    Add {
        /// Unique annotation identifier.
        annotation_id: String,
        /// Serialised annotation payload (e.g. JSON).
        payload: String,
        /// The user who performed the action.
        author: String,
    },
    /// An existing annotation was updated in-place.
    Update {
        /// Annotation being modified.
        annotation_id: String,
        /// Field that changed.
        field: String,
        /// New value (serialised).
        value: String,
        /// The user who performed the action.
        author: String,
    },
    /// An annotation was removed.
    Remove {
        /// Annotation being removed.
        annotation_id: String,
        /// The user who performed the action.
        author: String,
    },
    /// A comment was resolved.
    Resolve {
        /// Annotation / comment identifier.
        annotation_id: String,
        /// The user who resolved it.
        resolver: String,
    },
}

/// An entry in the delta log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaEntry {
    /// Log sequence number (monotonically increasing, 1-based).
    pub seq: Seq,
    /// When the mutation was recorded.
    pub timestamp: DateTime<Utc>,
    /// The mutation itself.
    pub delta: AnnotationDelta,
}

// ---------------------------------------------------------------------------
// DeltaLog
// ---------------------------------------------------------------------------

/// Append-only, bounded ring of delta entries.
///
/// Entries with sequence numbers older than `head_seq - capacity` are
/// evicted.  Clients that have fallen behind receive a `Resync` signal.
#[derive(Debug)]
pub struct DeltaLog {
    entries: Vec<DeltaEntry>,
    capacity: usize,
    next_seq: Seq,
}

impl DeltaLog {
    /// Create a new log with the given maximum capacity.
    ///
    /// `capacity` must be at least 1.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
            next_seq: 1,
        }
    }

    /// Append a delta and return its assigned sequence number.
    pub fn push(&mut self, delta: AnnotationDelta) -> Seq {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.entries.push(DeltaEntry {
            seq,
            timestamp: Utc::now(),
            delta,
        });
        // Evict entries beyond capacity (keep the most recent `capacity` entries).
        if self.entries.len() > self.capacity {
            let excess = self.entries.len() - self.capacity;
            self.entries.drain(..excess);
        }
        seq
    }

    /// Return the sequence number of the most recent entry, or 0 if empty.
    #[must_use]
    pub fn latest_seq(&self) -> Seq {
        self.entries.last().map(|e| e.seq).unwrap_or(0)
    }

    /// Return the sequence number of the oldest retained entry, or 0 if empty.
    #[must_use]
    pub fn oldest_seq(&self) -> Seq {
        self.entries.first().map(|e| e.seq).unwrap_or(0)
    }

    /// Return all entries with `seq > since`, or `None` if `since` is older
    /// than the oldest retained entry (signalling a required resync).
    #[must_use]
    pub fn since(&self, since: Seq) -> Option<&[DeltaEntry]> {
        if self.entries.is_empty() {
            return Some(&[]);
        }
        let oldest = self.oldest_seq();
        // If the client's last-ack is *before* the oldest retained entry it
        // cannot catch up incrementally — the caller must resync.
        if since > 0 && since < oldest.saturating_sub(1) {
            return None;
        }
        let pos = self.entries.partition_point(|e| e.seq <= since);
        Some(&self.entries[pos..])
    }

    /// Total number of entries currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the log contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Client cursor tracking
// ---------------------------------------------------------------------------

/// Tracks the last-acknowledged sequence number for each connected client.
#[derive(Debug, Default)]
pub struct ClientCursors {
    cursors: HashMap<ClientId, Seq>,
}

impl ClientCursors {
    /// Create an empty cursor map.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new client at sequence `seq` (typically 0 for a fresh join).
    pub fn register(&mut self, client_id: impl Into<ClientId>, seq: Seq) {
        self.cursors.insert(client_id.into(), seq);
    }

    /// Record that `client_id` has acknowledged up to `seq`.
    ///
    /// Returns `false` if the client was not previously registered.
    pub fn ack(&mut self, client_id: &str, seq: Seq) -> bool {
        match self.cursors.get_mut(client_id) {
            Some(current) => {
                if seq > *current {
                    *current = seq;
                }
                true
            }
            None => false,
        }
    }

    /// Unregister a client (e.g. on disconnect).
    pub fn remove(&mut self, client_id: &str) {
        self.cursors.remove(client_id);
    }

    /// Return the last-acknowledged seq for `client_id`.
    #[must_use]
    pub fn cursor_for(&self, client_id: &str) -> Option<Seq> {
        self.cursors.get(client_id).copied()
    }

    /// Return the minimum cursor across all registered clients.
    ///
    /// Useful for safe log truncation: entries before this seq are not
    /// needed by any client.
    #[must_use]
    pub fn min_cursor(&self) -> Seq {
        self.cursors.values().copied().min().unwrap_or(0)
    }

    /// Number of registered clients.
    #[must_use]
    pub fn client_count(&self) -> usize {
        self.cursors.len()
    }
}

// ---------------------------------------------------------------------------
// DeltaBroadcaster
// ---------------------------------------------------------------------------

/// Outcome of a [`DeltaBroadcaster::fetch`] call.
#[derive(Debug, Clone)]
pub enum FetchResult<'a> {
    /// The client is up to date (no new entries).
    UpToDate,
    /// New entries the client has not yet seen.
    Deltas(&'a [DeltaEntry]),
    /// Client cursor is too old; it must fetch a full snapshot and re-register.
    ResyncRequired,
}

/// Coordinates delta generation and per-client delivery.
#[derive(Debug)]
pub struct DeltaBroadcaster {
    log: DeltaLog,
    cursors: ClientCursors,
}

impl DeltaBroadcaster {
    /// Create a broadcaster with the given log capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            log: DeltaLog::new(capacity),
            cursors: ClientCursors::new(),
        }
    }

    /// Apply a delta and notify the log; returns the assigned sequence number.
    pub fn apply(&mut self, delta: AnnotationDelta) -> Seq {
        self.log.push(delta)
    }

    /// Register a new client.  Use `seq = 0` for a client that has just
    /// performed a full snapshot fetch, or the client's last-known seq.
    pub fn connect(&mut self, client_id: impl Into<ClientId>, seq: Seq) {
        self.cursors.register(client_id, seq);
    }

    /// Disconnect a client.
    pub fn disconnect(&mut self, client_id: &str) {
        self.cursors.remove(client_id);
    }

    /// Fetch deltas for `client_id` since its last acknowledged sequence.
    pub fn fetch<'a>(&'a self, client_id: &str) -> FetchResult<'a> {
        let since = match self.cursors.cursor_for(client_id) {
            Some(s) => s,
            None => return FetchResult::ResyncRequired,
        };
        match self.log.since(since) {
            None => FetchResult::ResyncRequired,
            Some([]) => FetchResult::UpToDate,
            Some(entries) => FetchResult::Deltas(entries),
        }
    }

    /// Acknowledge that `client_id` has processed up to `seq`.
    ///
    /// Returns `false` if the client is unknown.
    pub fn ack(&mut self, client_id: &str, seq: Seq) -> bool {
        self.cursors.ack(client_id, seq)
    }

    /// Number of entries in the log.
    #[must_use]
    pub fn log_len(&self) -> usize {
        self.log.len()
    }

    /// Latest sequence number assigned.
    #[must_use]
    pub fn latest_seq(&self) -> Seq {
        self.log.latest_seq()
    }

    /// Number of currently connected clients.
    #[must_use]
    pub fn client_count(&self) -> usize {
        self.cursors.client_count()
    }
}

// ---------------------------------------------------------------------------
// Annotation diff / apply bridge
// ---------------------------------------------------------------------------
//
// The [`DeltaBroadcaster`] above operates on opaque [`AnnotationDelta`]
// values.  This section bridges the typed [`Annotation`] state used by the
// review API to those deltas: [`diff_annotations`] computes the minimal set of
// operations turning `prev` into `curr`, and [`apply_delta`] / [`apply_message`]
// replay them onto a `HashMap<DrawingId, Annotation>`.  All functions here are
// pure (no I/O, no globals) so they are usable on both native and `wasm32`
// targets.

/// Field-name constants shared by [`diff_annotations`] and [`apply_delta`].
///
/// Centralising these strings guarantees the producer (diff) and consumer
/// (apply) agree on the exact wire identifiers; a mismatch would silently drop
/// updates, so both sides reference the same `const`.
pub mod field {
    /// The annotation's display label (`Option<String>`).
    pub const LABEL: &str = "label";
    /// The annotation's visibility flag (`bool`).
    pub const VISIBLE: &str = "visible";
    /// The annotation's locked flag (`bool`).
    pub const LOCKED: &str = "locked";
    /// The annotation's z-order layer index (`usize`).
    pub const LAYER: &str = "layer";
    /// The underlying drawing shape (`drawing.shape`).
    pub const SHAPE: &str = "drawing.shape";
    /// The underlying drawing stroke style (`drawing.style`).
    pub const STYLE: &str = "drawing.style";
    /// The underlying drawing frame number (`drawing.frame`).
    pub const FRAME: &str = "drawing.frame";
}

/// A self-describing message exchanged between collaborating clients.
///
/// This is the payload carried by the realtime transport.  It supersedes the
/// raw `&[DeltaEntry]` slice returned by [`DeltaBroadcaster::fetch`] by also
/// covering the back-compatible full-state ([`DeltaMessage::Snapshot`]) and
/// resync ([`DeltaMessage::Resync`]) paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeltaMessage {
    /// Incremental updates relative to `base_seq` (the recipient's last-known
    /// sequence number). Replay `entries` in order to advance state.
    Deltas {
        /// Sequence number the receiver is expected to be at before applying.
        base_seq: Seq,
        /// Ordered mutations to replay.
        entries: Vec<DeltaEntry>,
    },
    /// Full annotation state at `seq`. Replaces the recipient's state wholesale
    /// (back-compat path, also used when a client first joins or resyncs).
    Snapshot {
        /// Sequence number this snapshot is current as of.
        seq: Seq,
        /// Complete annotation set.
        annotations: Vec<Annotation>,
    },
    /// The recipient's cursor is too old to catch up incrementally; it must
    /// request a fresh [`DeltaMessage::Snapshot`].
    Resync,
}

/// Index an annotation slice by [`DrawingId`] for set-difference comparison.
fn index_by_id(annotations: &[Annotation]) -> HashMap<DrawingId, &Annotation> {
    annotations.iter().map(|a| (a.id, a)).collect()
}

/// Emit field-level [`AnnotationDelta::Update`]s for every changed field of an
/// annotation that exists in both `prev` and `curr`.
///
/// Only the mutable, user-facing fields are considered; `created_at` and
/// `updated_at` are intentionally skipped (they are bookkeeping timestamps,
/// not collaborative state). Returns a serialisation error only for the
/// drawing sub-fields (which require JSON encoding); the primitive fields
/// never fail.
fn diff_fields(
    prev: &Annotation,
    curr: &Annotation,
    author: &str,
    out: &mut Vec<AnnotationDelta>,
) -> ReviewResult<()> {
    let id = curr.id.to_string();
    let mut push = |field: &str, value: String| {
        out.push(AnnotationDelta::Update {
            annotation_id: id.clone(),
            field: field.to_string(),
            value,
            author: author.to_string(),
        });
    };

    if prev.label != curr.label {
        push(field::LABEL, serde_json::to_string(&curr.label)?);
    }
    if prev.visible != curr.visible {
        push(field::VISIBLE, serde_json::to_string(&curr.visible)?);
    }
    if prev.locked != curr.locked {
        push(field::LOCKED, serde_json::to_string(&curr.locked)?);
    }
    if prev.layer != curr.layer {
        push(field::LAYER, serde_json::to_string(&curr.layer)?);
    }
    if prev.drawing.shape != curr.drawing.shape {
        push(field::SHAPE, serde_json::to_string(&curr.drawing.shape)?);
    }
    if prev.drawing.style != curr.drawing.style {
        push(field::STYLE, serde_json::to_string(&curr.drawing.style)?);
    }
    if prev.drawing.frame != curr.drawing.frame {
        push(field::FRAME, serde_json::to_string(&curr.drawing.frame)?);
    }
    Ok(())
}

/// Compute the minimal set of [`AnnotationDelta`]s transforming `prev` into
/// `curr`, attributing every mutation to `author`.
///
/// - Annotations in `curr` but not `prev` produce [`AnnotationDelta::Add`].
/// - Annotations in `prev` but not `curr` produce [`AnnotationDelta::Remove`].
/// - Annotations in both produce field-level [`AnnotationDelta::Update`]s for
///   each changed field (see [`field`]); unchanged annotations produce nothing.
///
/// If an annotation cannot be serialised, the offending `Add`/`Update` is
/// skipped (the lossy, best-effort variant). Callers that need to know a
/// serialisation failure occurred — so they can fall back to broadcasting a
/// full [`DeltaMessage::Snapshot`] — should use [`try_diff_annotations`].
#[must_use]
pub fn diff_annotations(
    prev: &[Annotation],
    curr: &[Annotation],
    author: &str,
) -> Vec<AnnotationDelta> {
    let prev_map = index_by_id(prev);
    let curr_map = index_by_id(curr);
    let mut deltas = Vec::new();

    // Removals: in prev, gone from curr.
    for (id, _) in &prev_map {
        if !curr_map.contains_key(id) {
            deltas.push(AnnotationDelta::Remove {
                annotation_id: id.to_string(),
                author: author.to_string(),
            });
        }
    }

    // Additions and updates.
    for (id, curr_ann) in &curr_map {
        match prev_map.get(id) {
            None => {
                // New annotation: serialise the whole thing as the payload.
                // On failure, skip (lossy diff) — the strict variant surfaces
                // this so the caller can force a snapshot instead.
                if let Ok(payload) = serde_json::to_string(curr_ann) {
                    deltas.push(AnnotationDelta::Add {
                        annotation_id: id.to_string(),
                        payload,
                        author: author.to_string(),
                    });
                }
            }
            Some(prev_ann) => {
                // Field-level updates; ignore serialisation errors (lossy).
                let _ = diff_fields(prev_ann, curr_ann, author, &mut deltas);
            }
        }
    }

    deltas
}

/// Strict primitive behind [`diff_annotations`]: identical semantics, but
/// returns `Err` (rather than silently skipping) if any annotation fails to
/// serialise. The realtime bridge uses this to decide whether to fall back to
/// a full [`DeltaMessage::Snapshot`].
///
/// # Errors
///
/// Returns [`ReviewError::Serialization`] if an added annotation or an updated
/// drawing sub-field cannot be encoded as JSON.
pub fn try_diff_annotations(
    prev: &[Annotation],
    curr: &[Annotation],
    author: &str,
) -> ReviewResult<Vec<AnnotationDelta>> {
    let prev_map = index_by_id(prev);
    let curr_map = index_by_id(curr);
    let mut deltas = Vec::new();

    for (id, _) in &prev_map {
        if !curr_map.contains_key(id) {
            deltas.push(AnnotationDelta::Remove {
                annotation_id: id.to_string(),
                author: author.to_string(),
            });
        }
    }

    for (id, curr_ann) in &curr_map {
        match prev_map.get(id) {
            None => {
                let payload = serde_json::to_string(curr_ann)?;
                deltas.push(AnnotationDelta::Add {
                    annotation_id: id.to_string(),
                    payload,
                    author: author.to_string(),
                });
            }
            Some(prev_ann) => {
                diff_fields(prev_ann, curr_ann, author, &mut deltas)?;
            }
        }
    }

    Ok(deltas)
}

/// Parse a [`DrawingId`] from its `Display` form (a UUID string).
fn parse_drawing_id(s: &str) -> ReviewResult<DrawingId> {
    let uuid = uuid::Uuid::parse_str(s)
        .map_err(|e| ReviewError::Other(format!("invalid annotation id '{s}': {e}")))?;
    // DrawingId wraps a Uuid; reconstruct it via its serde representation so we
    // do not depend on a private constructor.
    serde_json::from_value(serde_json::Value::String(uuid.to_string()))
        .map_err(ReviewError::Serialization)
}

/// Apply a single [`AnnotationDelta`] to a keyed annotation state.
///
/// - [`AnnotationDelta::Add`] deserialises the payload and inserts it.
/// - [`AnnotationDelta::Update`] patches the named field in place; an
///   unrecognised field name yields [`ReviewError::Other`] (never panics).
/// - [`AnnotationDelta::Remove`] drops the entry.
/// - [`AnnotationDelta::Resolve`] is a no-op here (resolution is tracked by the
///   comment subsystem, not the annotation geometry state).
///
/// # Errors
///
/// Returns an error if the annotation id is malformed, a payload/value cannot
/// be deserialised, the target of an update does not exist, or the update field
/// is unknown.
pub fn apply_delta(
    state: &mut HashMap<DrawingId, Annotation>,
    delta: &AnnotationDelta,
) -> ReviewResult<()> {
    match delta {
        AnnotationDelta::Add {
            annotation_id,
            payload,
            ..
        } => {
            let ann: Annotation = serde_json::from_str(payload)?;
            let id = parse_drawing_id(annotation_id)?;
            state.insert(id, ann);
            Ok(())
        }
        AnnotationDelta::Update {
            annotation_id,
            field: field_name,
            value,
            ..
        } => {
            let id = parse_drawing_id(annotation_id)?;
            let ann = state.get_mut(&id).ok_or_else(|| {
                ReviewError::DrawingNotFound(format!(
                    "cannot update unknown annotation '{annotation_id}'"
                ))
            })?;
            apply_field_update(ann, field_name, value)
        }
        AnnotationDelta::Remove { annotation_id, .. } => {
            let id = parse_drawing_id(annotation_id)?;
            state.remove(&id);
            Ok(())
        }
        // Resolution is a comment-thread concern, not annotation geometry; the
        // delta is carried for audit but applies no geometric change here.
        AnnotationDelta::Resolve { .. } => Ok(()),
    }
}

/// Patch a single field of `ann` from its serialised `value`.
///
/// Field names must match the [`field`] constants used by [`diff_annotations`];
/// any other name is rejected so a typo can never silently corrupt state.
fn apply_field_update(ann: &mut Annotation, field_name: &str, value: &str) -> ReviewResult<()> {
    match field_name {
        field::LABEL => ann.label = serde_json::from_str(value)?,
        field::VISIBLE => ann.visible = serde_json::from_str(value)?,
        field::LOCKED => ann.locked = serde_json::from_str(value)?,
        field::LAYER => ann.layer = serde_json::from_str(value)?,
        field::SHAPE => ann.drawing.shape = serde_json::from_str(value)?,
        field::STYLE => ann.drawing.style = serde_json::from_str(value)?,
        field::FRAME => ann.drawing.frame = serde_json::from_str(value)?,
        other => {
            return Err(ReviewError::Other(format!(
                "unknown annotation update field '{other}'"
            )))
        }
    }
    Ok(())
}

/// Apply a whole [`DeltaMessage`] to a keyed annotation state.
///
/// - [`DeltaMessage::Deltas`] replays each entry in order via [`apply_delta`].
/// - [`DeltaMessage::Snapshot`] replaces the state wholesale.
/// - [`DeltaMessage::Resync`] clears the state and signals (via the returned
///   `Ok(())` with an emptied map) that the caller should request a snapshot.
///
/// # Errors
///
/// Propagates any error from [`apply_delta`] while replaying a `Deltas`
/// message.
pub fn apply_message(
    state: &mut HashMap<DrawingId, Annotation>,
    msg: &DeltaMessage,
) -> ReviewResult<()> {
    match msg {
        DeltaMessage::Deltas { entries, .. } => {
            for entry in entries {
                apply_delta(state, &entry.delta)?;
            }
            Ok(())
        }
        DeltaMessage::Snapshot { annotations, .. } => {
            state.clear();
            for ann in annotations {
                state.insert(ann.id, ann.clone());
            }
            Ok(())
        }
        DeltaMessage::Resync => {
            // The local copy is unrecoverable; drop it and await a snapshot.
            state.clear();
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_add(id: &str) -> AnnotationDelta {
        AnnotationDelta::Add {
            annotation_id: id.to_string(),
            payload: format!("{{\"id\":\"{id}\"}}"),
            author: "alice".to_string(),
        }
    }

    fn make_update(id: &str) -> AnnotationDelta {
        AnnotationDelta::Update {
            annotation_id: id.to_string(),
            field: "text".to_string(),
            value: "updated".to_string(),
            author: "bob".to_string(),
        }
    }

    fn make_remove(id: &str) -> AnnotationDelta {
        AnnotationDelta::Remove {
            annotation_id: id.to_string(),
            author: "carol".to_string(),
        }
    }

    // 1. Empty log returns empty slice for seq=0.
    #[test]
    fn test_delta_log_empty_since_zero() {
        let log = DeltaLog::new(10);
        let result = log.since(0);
        assert!(matches!(result, Some(entries) if entries.is_empty()));
    }

    // 2. Pushing entries increments seq monotonically.
    #[test]
    fn test_delta_log_seq_increments() {
        let mut log = DeltaLog::new(10);
        let s1 = log.push(make_add("a1"));
        let s2 = log.push(make_add("a2"));
        let s3 = log.push(make_add("a3"));
        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
        assert_eq!(s3, 3);
        assert_eq!(log.latest_seq(), 3);
    }

    // 3. `since` returns only entries after the given cursor.
    #[test]
    fn test_delta_log_since_returns_tail() {
        let mut log = DeltaLog::new(10);
        log.push(make_add("a1"));
        log.push(make_add("a2"));
        log.push(make_add("a3"));
        let entries = log.since(1).expect("should return entries");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].seq, 2);
        assert_eq!(entries[1].seq, 3);
    }

    // 4. Capacity eviction triggers resync for stale clients.
    #[test]
    fn test_delta_log_eviction_triggers_resync() {
        let mut log = DeltaLog::new(3);
        for i in 0..5u32 {
            log.push(make_add(&i.to_string()));
        }
        // oldest seq retained should be 3 (5 - 3 + 1)
        assert_eq!(log.oldest_seq(), 3);
        // client that last saw seq=1 is now behind the oldest retained entry
        let result = log.since(1);
        assert!(result.is_none(), "stale client should get None (resync)");
    }

    // 5. Client cursor registration and ack.
    #[test]
    fn test_client_cursors_register_and_ack() {
        let mut cursors = ClientCursors::new();
        cursors.register("c1", 0);
        assert_eq!(cursors.cursor_for("c1"), Some(0));
        let ok = cursors.ack("c1", 5);
        assert!(ok);
        assert_eq!(cursors.cursor_for("c1"), Some(5));
    }

    // 6. Ack for unknown client returns false.
    #[test]
    fn test_client_cursors_ack_unknown_returns_false() {
        let mut cursors = ClientCursors::new();
        assert!(!cursors.ack("ghost", 10));
    }

    // 7. min_cursor returns lowest cursor across all clients.
    #[test]
    fn test_client_cursors_min_cursor() {
        let mut cursors = ClientCursors::new();
        cursors.register("c1", 5);
        cursors.register("c2", 10);
        cursors.register("c3", 3);
        assert_eq!(cursors.min_cursor(), 3);
    }

    // 8. DeltaBroadcaster: apply, connect, fetch, ack round-trip.
    #[test]
    fn test_broadcaster_full_round_trip() {
        let mut bc = DeltaBroadcaster::new(100);
        // Client connects with no prior history.
        bc.connect("alice", 0);
        // Producer pushes two deltas.
        let s1 = bc.apply(make_add("ann-1"));
        let s2 = bc.apply(make_update("ann-1"));
        // Fetch for alice should yield both deltas.
        let entries = match bc.fetch("alice") {
            FetchResult::Deltas(e) => e.to_vec(),
            other => panic!("expected Deltas, got {other:?}"),
        };
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].seq, s1);
        assert_eq!(entries[1].seq, s2);
        // Alice acks the latest.
        bc.ack("alice", s2);
        // Now fetch should return UpToDate.
        assert!(matches!(bc.fetch("alice"), FetchResult::UpToDate));
    }

    // 9. Disconnected client receives ResyncRequired.
    #[test]
    fn test_broadcaster_disconnected_client_resync() {
        let mut bc = DeltaBroadcaster::new(10);
        bc.connect("bob", 0);
        bc.disconnect("bob");
        assert!(matches!(bc.fetch("bob"), FetchResult::ResyncRequired));
    }

    // 10. Remove delta is recorded correctly.
    #[test]
    fn test_broadcaster_remove_delta() {
        let mut bc = DeltaBroadcaster::new(10);
        bc.connect("dave", 0);
        bc.apply(make_add("x"));
        let seq = bc.apply(make_remove("x"));
        let entries = match bc.fetch("dave") {
            FetchResult::Deltas(e) => e.to_vec(),
            other => panic!("expected Deltas, got {other:?}"),
        };
        let last = entries.last().expect("must have last entry");
        assert_eq!(last.seq, seq);
        assert!(matches!(
            &last.delta,
            AnnotationDelta::Remove { annotation_id, .. } if annotation_id == "x"
        ));
    }

    // 11. Resolve delta is serialisable.
    #[test]
    fn test_resolve_delta_serialize() {
        let delta = AnnotationDelta::Resolve {
            annotation_id: "ann-99".to_string(),
            resolver: "manager".to_string(),
        };
        let json = serde_json::to_string(&delta).expect("serialize");
        assert!(json.contains("ann-99"));
        let back: AnnotationDelta = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, delta);
    }

    // 12. log_len and client_count reflect state.
    #[test]
    fn test_broadcaster_metadata() {
        let mut bc = DeltaBroadcaster::new(50);
        bc.connect("u1", 0);
        bc.connect("u2", 0);
        assert_eq!(bc.client_count(), 2);
        bc.apply(make_add("z"));
        bc.apply(make_add("y"));
        assert_eq!(bc.log_len(), 2);
        assert_eq!(bc.latest_seq(), 2);
    }

    // -----------------------------------------------------------------------
    // Annotation diff / apply bridge tests
    // -----------------------------------------------------------------------

    use crate::drawing::{Circle, Color, Drawing, DrawingTool, Point, Shape, StrokeStyle};
    use crate::SessionId;

    /// Build a deterministic annotation with the given id and label.
    fn make_annotation(id: DrawingId, label: Option<&str>) -> Annotation {
        let drawing = Drawing {
            id,
            session_id: SessionId::new(),
            frame: 100,
            tool: DrawingTool::Circle,
            shape: Shape::Circle(Circle::new(Point::new(0.5, 0.5), 0.2)),
            style: StrokeStyle::solid(Color::red(), 2.0),
            author: "author".to_string(),
        };
        let mut ann = Annotation::new(drawing);
        ann.label = label.map(str::to_string);
        ann
    }

    /// Apply an ordered slice of deltas onto a fresh state map.
    fn replay(prev: &[Annotation], deltas: &[AnnotationDelta]) -> HashMap<DrawingId, Annotation> {
        let mut state: HashMap<DrawingId, Annotation> =
            prev.iter().map(|a| (a.id, a.clone())).collect();
        for delta in deltas {
            apply_delta(&mut state, delta).expect("apply must succeed");
        }
        state
    }

    // 13. diff: add + remove + single field update, exact counts.
    // prev{A,B}, curr{A',C} where A' has a changed label.
    #[test]
    fn test_diff_add_update_remove_exact() {
        let id_a = DrawingId::new();
        let id_b = DrawingId::new();
        let id_c = DrawingId::new();

        let a = make_annotation(id_a, Some("original"));
        let b = make_annotation(id_b, None);
        let mut a_prime = a.clone();
        a_prime.label = Some("changed".to_string());
        let c = make_annotation(id_c, Some("new"));

        let prev = vec![a, b];
        let curr = vec![a_prime, c];

        let deltas = diff_annotations(&prev, &curr, "alice");

        let removes = deltas
            .iter()
            .filter(|d| matches!(d, AnnotationDelta::Remove { .. }))
            .count();
        let adds = deltas
            .iter()
            .filter(|d| matches!(d, AnnotationDelta::Add { .. }))
            .count();
        let updates: Vec<_> = deltas
            .iter()
            .filter(|d| matches!(d, AnnotationDelta::Update { .. }))
            .collect();

        assert_eq!(removes, 1, "exactly one Remove (B)");
        assert_eq!(adds, 1, "exactly one Add (C)");
        assert_eq!(updates.len(), 1, "exactly one Update (A.label)");

        // Verify the Remove targets B, the Add targets C, the Update targets A.label.
        assert!(deltas.iter().any(|d| matches!(
            d, AnnotationDelta::Remove { annotation_id, .. } if *annotation_id == id_b.to_string()
        )));
        assert!(deltas.iter().any(|d| matches!(
            d, AnnotationDelta::Add { annotation_id, .. } if *annotation_id == id_c.to_string()
        )));
        match updates[0] {
            AnnotationDelta::Update {
                annotation_id,
                field: f,
                ..
            } => {
                assert_eq!(*annotation_id, id_a.to_string());
                assert_eq!(f, field::LABEL);
            }
            other => panic!("expected Update, got {other:?}"),
        }
    }

    // 14. field-level diff emits ONLY the changed field (visible flipped → one
    //     Update; no label/locked/layer updates).
    #[test]
    fn test_diff_field_level_only_changed() {
        let id = DrawingId::new();
        let prev_ann = make_annotation(id, Some("same"));
        let mut curr_ann = prev_ann.clone();
        curr_ann.visible = !curr_ann.visible; // flip only visibility

        let deltas = diff_annotations(&[prev_ann], &[curr_ann], "bob");

        assert_eq!(deltas.len(), 1, "exactly one field update");
        match &deltas[0] {
            AnnotationDelta::Update { field: f, .. } => {
                assert_eq!(f, field::VISIBLE, "only the visible field changed");
            }
            other => panic!("expected Update, got {other:?}"),
        }
        // No other field name should appear.
        assert!(!deltas.iter().any(|d| matches!(
            d, AnnotationDelta::Update { field: f, .. }
                if f == field::LABEL || f == field::LOCKED || f == field::LAYER
        )));
    }

    // 15. no change → empty delta set.
    #[test]
    fn test_diff_no_change_empty() {
        let id = DrawingId::new();
        let ann = make_annotation(id, Some("stable"));
        let state = std::slice::from_ref(&ann);
        let deltas = diff_annotations(state, state, "carol");
        assert!(deltas.is_empty(), "identical state yields no deltas");
    }

    // 16. applying the diff in order reconstructs curr exactly.
    #[test]
    fn test_apply_reconstructs_curr() {
        let id_a = DrawingId::new();
        let id_b = DrawingId::new();
        let id_c = DrawingId::new();

        let a = make_annotation(id_a, Some("a-orig"));
        let b = make_annotation(id_b, Some("b"));
        let mut a_prime = a.clone();
        a_prime.label = Some("a-new".to_string());
        a_prime.visible = false;
        a_prime.layer = 7;
        let c = make_annotation(id_c, Some("c"));

        let prev = vec![a, b];
        let curr = vec![a_prime, c];

        let deltas = diff_annotations(&prev, &curr, "alice");
        let reconstructed = replay(&prev, &deltas);

        let expected: HashMap<DrawingId, Annotation> =
            curr.iter().map(|a| (a.id, a.clone())).collect();

        assert_eq!(reconstructed.len(), expected.len());
        for (id, exp) in &expected {
            let got = reconstructed
                .get(id)
                .expect("annotation present after apply");
            assert_eq!(got.label, exp.label);
            assert_eq!(got.visible, exp.visible);
            assert_eq!(got.layer, exp.layer);
            assert_eq!(got.locked, exp.locked);
            assert_eq!(got.drawing.frame, exp.drawing.frame);
        }
        assert!(
            !reconstructed.contains_key(&id_b),
            "removed annotation B must be gone"
        );
    }

    // 17. apply_message Snapshot replaces existing state wholesale.
    #[test]
    fn test_apply_message_snapshot_replaces() {
        let id_old = DrawingId::new();
        let id_new = DrawingId::new();
        let mut state: HashMap<DrawingId, Annotation> = HashMap::new();
        state.insert(id_old, make_annotation(id_old, Some("old")));

        let snapshot = DeltaMessage::Snapshot {
            seq: 5,
            annotations: vec![make_annotation(id_new, Some("new"))],
        };
        apply_message(&mut state, &snapshot).expect("snapshot apply");

        assert_eq!(state.len(), 1);
        assert!(!state.contains_key(&id_old), "old state cleared");
        assert!(state.contains_key(&id_new), "snapshot content present");
    }

    // 17b. apply_message Resync clears state.
    #[test]
    fn test_apply_message_resync_clears() {
        let id = DrawingId::new();
        let mut state: HashMap<DrawingId, Annotation> = HashMap::new();
        state.insert(id, make_annotation(id, None));
        apply_message(&mut state, &DeltaMessage::Resync).expect("resync apply");
        assert!(state.is_empty(), "resync drops the local copy");
    }

    // 17c. apply_delta rejects unknown field without panicking.
    #[test]
    fn test_apply_delta_unknown_field_errors() {
        let id = DrawingId::new();
        let mut state: HashMap<DrawingId, Annotation> = HashMap::new();
        state.insert(id, make_annotation(id, None));
        let bad = AnnotationDelta::Update {
            annotation_id: id.to_string(),
            field: "not_a_real_field".to_string(),
            value: "\"x\"".to_string(),
            author: "alice".to_string(),
        };
        assert!(apply_delta(&mut state, &bad).is_err());
    }

    // 18. DeltaMessage serde round-trip for all three variants.
    #[test]
    fn test_delta_message_serde_round_trip() {
        let id = DrawingId::new();
        let now = Utc::now();

        let deltas = DeltaMessage::Deltas {
            base_seq: 3,
            entries: vec![DeltaEntry {
                seq: 4,
                timestamp: now,
                delta: AnnotationDelta::Remove {
                    annotation_id: id.to_string(),
                    author: "alice".to_string(),
                },
            }],
        };
        let snapshot = DeltaMessage::Snapshot {
            seq: 9,
            annotations: vec![make_annotation(id, Some("snap"))],
        };
        let resync = DeltaMessage::Resync;

        for msg in [deltas, snapshot, resync] {
            let json = serde_json::to_string(&msg).expect("serialize message");
            let back: DeltaMessage = serde_json::from_str(&json).expect("deserialize message");
            // Re-encode and compare strings to assert structural equality
            // without requiring PartialEq on DeltaMessage.
            let json2 = serde_json::to_string(&back).expect("re-serialize");
            assert_eq!(json, json2, "round-trip must be stable");
        }
    }

    // 19. try_diff strict variant succeeds on well-formed input and matches the
    //     lossy variant's operation counts.
    #[test]
    fn test_try_diff_matches_lossy() {
        let id_a = DrawingId::new();
        let id_b = DrawingId::new();
        let a = make_annotation(id_a, Some("a"));
        let b = make_annotation(id_b, Some("b"));
        let mut a2 = a.clone();
        a2.locked = true;

        let prev = vec![a];
        let curr = vec![a2, b];

        let lossy = diff_annotations(&prev, &curr, "alice");
        let strict = try_diff_annotations(&prev, &curr, "alice").expect("strict diff ok");
        assert_eq!(lossy.len(), strict.len());
        // One Add (B) + one Update (A.locked).
        assert_eq!(strict.len(), 2);
    }

    // 20. Stale-cursor: driving the DeltaBroadcaster past capacity makes fetch
    //     return ResyncRequired; the bridge must then emit a Snapshot (not
    //     Deltas) so the client converges.
    #[test]
    fn test_stale_cursor_bridge_emits_snapshot() {
        // Capacity 2: a client that only saw seq=1 falls behind after 4 pushes.
        let mut bc = DeltaBroadcaster::new(2);
        bc.connect("late", 1);
        for i in 0..4u32 {
            bc.apply(make_add(&i.to_string()));
        }

        // The broadcaster reports the client cannot catch up incrementally.
        let fetch = bc.fetch("late");
        assert!(
            matches!(fetch, FetchResult::ResyncRequired),
            "stale client must require resync"
        );

        // The bridge's policy: on ResyncRequired, build a full Snapshot rather
        // than Deltas. Simulate the current full state and verify the message.
        let id = DrawingId::new();
        let full_state = vec![make_annotation(id, Some("full"))];
        let msg = match fetch {
            FetchResult::ResyncRequired => DeltaMessage::Snapshot {
                seq: bc.latest_seq(),
                annotations: full_state.clone(),
            },
            FetchResult::Deltas(entries) => DeltaMessage::Deltas {
                base_seq: 1,
                entries: entries.to_vec(),
            },
            FetchResult::UpToDate => DeltaMessage::Deltas {
                base_seq: 1,
                entries: Vec::new(),
            },
        };
        assert!(
            matches!(msg, DeltaMessage::Snapshot { .. }),
            "bridge must emit Snapshot for a stale cursor, not Deltas"
        );

        // And the snapshot reconstructs the full state on the receiver.
        let mut state: HashMap<DrawingId, Annotation> = HashMap::new();
        apply_message(&mut state, &msg).expect("apply snapshot");
        assert_eq!(state.len(), 1);
        assert!(state.contains_key(&id));
    }
}
