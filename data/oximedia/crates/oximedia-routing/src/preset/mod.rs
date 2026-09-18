//! Preset management module.

pub mod manager;
pub mod save;

pub use manager::{PresetError, PresetId, PresetManager};
pub use save::{MonitorSettings, PresetData, RoutingPreset};
