//! Low Latency HLS (LL-HLS) support.
//!
//! Implements Apple's LL-HLS protocol for sub-second latency.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// LL-HLS configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlHlsConfig {
    /// Enable LL-HLS.
    pub enabled: bool,

    /// Part duration (typically 0.5-2 seconds).
    pub part_duration: Duration,

    /// Enable partial segments.
    pub enable_parts: bool,

    /// Enable blocking playlist reload.
    pub enable_blocking_reload: bool,

    /// Enable rendition reports.
    pub enable_rendition_reports: bool,

    /// Enable preload hints.
    pub enable_preload_hints: bool,

    /// Part hold back (how far from live edge).
    pub part_hold_back: Duration,
}

impl Default for LlHlsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            part_duration: Duration::from_millis(500),
            enable_parts: true,
            enable_blocking_reload: true,
            enable_rendition_reports: true,
            enable_preload_hints: true,
            part_hold_back: Duration::from_secs(2),
        }
    }
}

/// Partial segment (part) for LL-HLS.
#[derive(Debug, Clone)]
pub struct Part {
    /// Part duration.
    pub duration: Duration,

    /// Part URI.
    pub uri: String,

    /// Is independent (can be decoded independently).
    pub independent: bool,
}

impl Part {
    /// Creates a new part.
    #[must_use]
    pub fn new(duration: Duration, uri: impl Into<String>) -> Self {
        Self {
            duration,
            uri: uri.into(),
            independent: false,
        }
    }

    /// Marks as independent.
    #[must_use]
    pub const fn independent(mut self) -> Self {
        self.independent = true;
        self
    }
}
