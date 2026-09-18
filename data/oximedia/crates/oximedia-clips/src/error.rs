//! Error types for clip management.

use std::path::PathBuf;

/// Result type for clip operations.
pub type ClipResult<T> = Result<T, ClipError>;

/// Errors that can occur during clip operations.
#[derive(Debug, thiserror::Error)]
pub enum ClipError {
    /// Database error.
    #[cfg(not(target_arch = "wasm32"))]
    #[error("Database error: {0}")]
    Database(#[from] oxisql_core::OxiSqlError),

    /// Clip not found.
    #[error("Clip not found: {0}")]
    ClipNotFound(String),

    /// Bin not found.
    #[error("Bin not found: {0}")]
    BinNotFound(String),

    /// Folder not found.
    #[error("Folder not found: {0}")]
    FolderNotFound(String),

    /// Collection not found.
    #[error("Collection not found: {0}")]
    CollectionNotFound(String),

    /// Marker not found.
    #[error("Marker not found: {0}")]
    MarkerNotFound(String),

    /// Note not found.
    #[error("Note not found: {0}")]
    NoteNotFound(String),

    /// Take not found.
    #[error("Take not found: {0}")]
    TakeNotFound(String),

    /// Invalid timecode.
    #[error("Invalid timecode: {0}")]
    InvalidTimecode(String),

    /// Invalid rating.
    #[error("Invalid rating: {0}")]
    InvalidRating(i32),

    /// Invalid proxy quality.
    #[error("Invalid proxy quality: {0}")]
    InvalidProxyQuality(String),

    /// File not found.
    #[error("File not found: {0}")]
    FileNotFound(PathBuf),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Invalid smart collection rule.
    #[error("Invalid smart collection rule: {0}")]
    InvalidSmartRule(String),

    /// Export error.
    #[error("Export error: {0}")]
    Export(String),

    /// Import error.
    #[error("Import error: {0}")]
    Import(String),
}
