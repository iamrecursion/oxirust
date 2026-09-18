//! Fuzz target: feed arbitrary bytes into the JSON adapter input parser.
//!
//! This harness exercises the public JSON deserialization boundary of
//! tensorlogic-quantrs-hooks.  The invariant is: **arbitrary byte input must
//! never cause a panic** — all parse errors must surface as `Err`.
//!
//! Targets exercised:
//! - `ModelExport` deserialization (primary adapter input format)
//! - `DistributionExport` deserialization (hook payload boundary)
//! - `QuantRSAssignment` deserialization (variable-assignment payloads)
//! - UTF-8 string path via `serde_json::from_str`

#![no_main]

use libfuzzer_sys::fuzz_target;
use tensorlogic_quantrs_hooks::{DistributionExport, ModelExport, QuantRSAssignment};

fuzz_target!(|data: &[u8]| {
    // Path 1: direct serde_json deserialization of ModelExport from raw bytes.
    // This exercises the primary adapter input parser boundary.
    let _ = serde_json::from_slice::<ModelExport>(data);

    // Path 2: DistributionExport — the hook payload boundary.
    let _ = serde_json::from_slice::<DistributionExport>(data);

    // Path 3: QuantRSAssignment — variable-assignment payloads.
    let _ = serde_json::from_slice::<QuantRSAssignment>(data);

    // Path 4: try UTF-8 interpretation and parse as JSON value for structural
    // coverage of the JSON parser's string handling.
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = serde_json::from_str::<ModelExport>(s);
        let _ = serde_json::from_str::<DistributionExport>(s);
    }
});
