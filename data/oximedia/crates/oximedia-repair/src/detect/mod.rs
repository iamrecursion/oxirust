//! Media corruption detection and analysis.
//!
//! This module provides tools for detecting various types of corruption
//! and issues in media files.
//!
//! # Detection pipeline
//!
//! ```text
//! [File]
//!   │
//!   ▼
//! corruption_check  (detect::corruption)
//!   ├── Magic-byte / header validation
//!   ├── Format-specific checks (MP4 boxes, MKV EBML, AVI chunks, MPEG sync)
//!   └── Emits: CorruptedHeader, high-confidence issues
//!   │
//!   ▼
//! analyze_structure  (detect::analyze)
//!   ├── analyze_container_structure — box/cluster/chunk layout
//!   ├── analyze_timestamps          — PTS/DTS continuity checks
//!   ├── analyze_indices             — seek-table presence and validity
//!   └── analyze_metadata            — tag/atom field sanity
//!   │
//!   ▼
//! deep_scan  (detect::scan)  ← only when severity ≥ High
//!   ├── Streaming or mmap I/O depending on file size
//!   ├── Zero-run detection (corruption artefact)
//!   ├── Sync-byte gap detection (MPEG stream losses)
//!   └── Packet-size sanity checks
//!   │
//!   ▼
//! [Issues]  (Vec<Issue> with type, severity, location, confidence)
//! ```
//!
//! Each stage adds issues to a shared accumulator.  The deep-scan stage is
//! only entered when the earlier stages detect at least one `High`-or-above
//! severity issue, keeping the common case fast.

pub mod analyze;
pub mod corruption;
pub mod scan;
