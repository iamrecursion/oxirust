//! Pitch and time manipulation effects.
//!
//! Provides:
//!
//! - **Pitch Shifter** - Change pitch without affecting tempo
//! - **Time Stretcher** - Change tempo without affecting pitch
//! - **Auto-tune** - Pitch correction to musical scale

pub mod autotune;
pub mod block_fft_shifter;
pub mod shifter;
pub mod stretch;

// Re-exports
pub use autotune::{AutoTune, AutoTuneConfig, Key, Scale};
pub use block_fft_shifter::{BlockFftConfig, BlockFftShifter};
pub use shifter::{
    AdvancedPitchShifter, FormantPreserver, PitchAlgorithm, PitchShifter, PitchShifterConfig,
};
pub use stretch::{TimeStretchConfig, TimeStretcher};
