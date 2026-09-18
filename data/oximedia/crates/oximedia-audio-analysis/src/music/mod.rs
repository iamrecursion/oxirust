//! Music analysis module (extends oximedia-mir).

pub mod harmony;
pub mod instrument;
pub mod rhythm;
pub mod timbre;

pub use harmony::{detect_key, detect_key_from_audio, HarmonyAnalyzer, HarmonyResult};
pub use instrument::{detect_instrument, Instrument};
pub use rhythm::{RhythmAnalyzer, RhythmFeatures};
pub use timbre::{TimbralAnalyzer, TimbralFeatures};
