//! Key detection using Krumhansl-Schmuckler algorithm.

pub mod detect;
pub mod modulation;
pub mod profile;

pub use detect::KeyDetector;
pub use modulation::{
    KeyRegion, Modulation, ModulationConfig, ModulationDetector, ModulationResult,
};
pub use profile::{KeyProfile, KEY_PROFILES};
