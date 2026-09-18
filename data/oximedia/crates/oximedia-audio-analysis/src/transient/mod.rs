//! Transient detection module.

pub mod detect;
pub mod envelope;

pub use detect::{TransientDetector, TransientResult};
pub use envelope::{compute_envelope, EnvelopeCharacteristics};
