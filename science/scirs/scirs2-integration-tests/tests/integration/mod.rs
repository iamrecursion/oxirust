// Cross-crate integration tests for SciRS2 v0.2.0
// Tests module interactions, data flow, and API compatibility

// Common utilities and test helpers
pub mod common;
pub mod fixtures;

// Cross-module integration tests
pub mod fft_signal;
pub mod ndimage_vision;
pub mod neural_optimize;
pub mod sparse_linalg;
pub mod stats_datasets;

// Wave 42 — new end-to-end pipelines
pub mod graph_pipeline;
pub mod scientific_pipeline;
pub mod vision_pipeline;

// Performance and memory integration tests
pub mod performance;

// Wave 42 batch 2 — ML, signal, and NLP pipelines
pub mod ml_pipeline;
pub mod nlp_pipeline;
pub mod signal_pipeline;

// Wave 44 — cross-crate numerical consistency tests
pub mod numerical_crosscrate;

// Wave 44 — statistical accuracy validation against known reference values
pub mod numerical_validation;
