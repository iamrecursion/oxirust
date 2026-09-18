//! Audio forensics module for authenticity verification.

pub mod authenticity;
pub mod compression;
pub mod edit;
/// ENF (Electrical Network Frequency) analysis for recording authentication.
pub mod enf;
pub mod noise;

pub use authenticity::{AuthenticityResult, AuthenticityVerifier};
pub use compression::{detect_compression_history, CompressionHistory};
pub use edit::{EditDetector, EditResult};
pub use enf::{analyze_enf, EnfAnalyzer, EnfConfig, EnfFrameEstimate, EnfResult, MainsFrequency};
pub use noise::{NoiseAnalyzer, NoiseConsistency};
