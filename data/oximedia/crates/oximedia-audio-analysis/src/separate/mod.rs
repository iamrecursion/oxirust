//! Source separation module using harmonic-percussive decomposition.

pub mod drums;
pub mod instrumental;
pub mod sources;
pub mod vocal;

pub use drums::separate_drums;
pub use instrumental::separate_instrumental;
pub use sources::{SeparationResult, SourceSeparator};
pub use vocal::separate_vocals;
