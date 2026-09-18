//! Fuzz target: DistributionExport and Factor conversion boundary.
//!
//! Exercises the full deserialization → conversion pipeline:
//! arbitrary bytes → `DistributionExport` → `Factor::from_quantrs_distribution`.
//! The invariant: **no panic at any stage**; all errors must be `Err`.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tensorlogic_quantrs_hooks::{DistributionExport, Factor, QuantRSDistribution};

fuzz_target!(|data: &[u8]| {
    // Attempt to deserialize a DistributionExport from raw bytes.
    if let Ok(dist) = serde_json::from_slice::<DistributionExport>(data) {
        // If deserialization succeeded, try to convert to a Factor.
        // This exercises the shape-validation logic in Factor::new.
        let _ = Factor::from_quantrs_distribution(&dist);
    }
});
