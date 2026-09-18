//! Error types for the search engine.

use std::io;

/// Result type for search operations
pub type SearchResult<T> = Result<T, SearchError>;

/// Error types for search operations
#[derive(Debug, thiserror::Error)]
pub enum SearchError {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    /// Tantivy error
    #[cfg(feature = "search-engine")]
    #[error("Index error: {0}")]
    Index(#[from] tantivy::TantivyError),

    /// Query parsing error
    #[cfg(feature = "search-engine")]
    #[error("Query parsing error: {0}")]
    QueryParse(#[from] tantivy::query::QueryParserError),

    /// Invalid query
    #[error("Invalid query: {0}")]
    InvalidQuery(String),

    /// Feature extraction error
    #[error("Feature extraction error: {0}")]
    FeatureExtraction(String),

    /// Audio fingerprinting error
    #[error("Audio fingerprinting error: {0}")]
    AudioFingerprint(String),

    /// Visual indexing error
    #[error("Visual indexing error: {0}")]
    VisualIndex(String),

    /// Face detection error
    #[error("Face detection error: {0}")]
    FaceDetection(String),

    /// OCR error
    #[error("OCR error: {0}")]
    Ocr(String),

    /// Color analysis error
    #[error("Color analysis error: {0}")]
    ColorAnalysis(String),

    /// Index not found
    #[error("Index not found at path: {0}")]
    IndexNotFound(String),

    /// Document not found
    #[error("Document not found: {0}")]
    DocumentNotFound(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Generic error
    #[error("{0}")]
    Other(String),
}
