//! Gain staging and trim controls module.

pub mod stage;
pub mod trim;

pub use stage::{GainError, GainStage, MultiChannelGainStage};
pub use trim::TrimControl;
