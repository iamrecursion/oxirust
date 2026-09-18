//! Modulation effects module.
//!
//! Provides classic modulation effects:
//!
//! - **Chorus** - Multi-voice ensemble effect
//! - **Tremolo** - Amplitude modulation
//! - **Vibrato** - Pitch/frequency modulation
//! - **Ring Modulation** - Frequency multiplication effect

pub mod chorus;
pub mod ringmod;
pub mod tremolo;
pub mod vibrato;

// Re-exports
pub use chorus::{ChorusConfig, StereoChorus, WavetableChorus, MAX_VOICES, MIN_VOICES};
pub use ringmod::{RingModConfig, RingModulator};
pub use tremolo::{StereoTremolo, TremoloConfig};
pub use vibrato::{StereoVibrato, VibratoConfig};
