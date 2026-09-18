//! Utility modules for audio effects.
//!
//! Provides common building blocks used by various effects:
//! - LFO (Low-Frequency Oscillator) for modulation
//! - Envelope followers for dynamics processing
//! - Delay lines for time-based effects
//! - Parameter smoothing for zipper-free real-time control

pub mod delay_line;
pub mod envelope;
pub mod lfo;

// Re-exports
pub use delay_line::{AllPassFilter, DelayLine, FractionalDelayLine, InterpolationMode};
pub use envelope::{EnvelopeFollower, PeakDetector, RmsDetector};
pub use lfo::{Lfo, LfoWaveform, ParameterSmoother, SmoothedParameter, SmoothingMode, StereoLfo};
