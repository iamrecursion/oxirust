//! Error types for the timeline editor.

#![allow(missing_docs)]

use oximedia_core::OxiError;

/// Result type for timeline editing operations.
pub type EditResult<T> = Result<T, EditError>;

/// Errors that can occur during timeline editing.
#[derive(Debug, thiserror::Error)]
pub enum EditError {
    /// Track index out of bounds.
    #[error("Track index {0} out of bounds (total tracks: {1})")]
    InvalidTrackIndex(usize, usize),

    /// Clip not found.
    #[error("Clip with ID {0} not found")]
    ClipNotFound(u64),

    /// Invalid time range.
    #[error("Invalid time range: {start} to {end}")]
    InvalidTimeRange { start: i64, end: i64 },

    /// Invalid transition parameters.
    #[error("Invalid transition: {0}")]
    InvalidTransition(String),

    /// Clip overlap detected.
    #[error("Clip overlap at time {0} on track {1}")]
    ClipOverlap(i64, usize),

    /// Clip type does not match the track type.
    #[error("Track type mismatch: track expects {expected:?} clip but got {got:?}")]
    TrackTypeMismatch {
        /// The clip type the track accepts.
        expected: crate::clip::ClipType,
        /// The clip type that was provided.
        got: crate::clip::ClipType,
    },

    /// Invalid edit operation.
    #[error("Invalid edit operation: {0}")]
    InvalidEdit(String),

    /// Invalid operation (e.g., unsupported format or capability).
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    /// Keyframe error.
    #[error("Keyframe error: {0}")]
    KeyframeError(String),

    /// Render error.
    #[error("Render error: {0}")]
    RenderError(String),

    /// Codec error.
    #[error("Codec error: {0}")]
    CodecError(#[from] OxiError),

    /// I/O error.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Graph error.
    #[error("Filter graph error: {0}")]
    GraphError(#[from] oximedia_graph::GraphError),
}
