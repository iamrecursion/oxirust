//! Container probing — format detection, stream analysis, and integrity checking.
//!
//! Split from the original `container_probe.rs` (1956 lines) via splitrs.

pub(crate) mod functions;
pub mod integrity;
pub mod multi_format;
pub mod stats;
pub mod types;

// Re-export all public types and functions
pub use integrity::*;
pub use multi_format::*;
pub use stats::*;
pub use types::*;

#[cfg(test)]
mod tests;
