//! Low Latency DASH (LL-DASH) support.
//!
//! Implements DASH low latency extensions for sub-second latency.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// LL-DASH configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlDashConfig {
    /// Enable LL-DASH.
    pub enabled: bool,

    /// Chunk duration.
    pub chunk_duration: Duration,

    /// Enable chunked encoding.
    pub enable_chunked_encoding: bool,

    /// Enable chunked transfer.
    pub enable_chunked_transfer: bool,

    /// Enable availability time offset.
    pub enable_availability_time_offset: bool,

    /// Availability time offset.
    pub availability_time_offset: Duration,

    /// Enable resync.
    pub enable_resync: bool,
}

impl Default for LlDashConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            chunk_duration: Duration::from_millis(500),
            enable_chunked_encoding: true,
            enable_chunked_transfer: true,
            enable_availability_time_offset: true,
            availability_time_offset: Duration::from_secs(1),
            enable_resync: true,
        }
    }
}

/// Chunk information for LL-DASH.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Chunk sequence number.
    pub sequence: u64,

    /// Chunk duration.
    pub duration: Duration,

    /// Chunk URI.
    pub uri: String,

    /// Chunk size in bytes.
    pub size: usize,
}

impl Chunk {
    /// Creates a new chunk.
    #[must_use]
    pub fn new(sequence: u64, duration: Duration, uri: impl Into<String>, size: usize) -> Self {
        Self {
            sequence,
            duration,
            uri: uri.into(),
            size,
        }
    }
}
