//! MADI (Multi-channel Audio Digital Interface) module.

pub mod interface;

pub use interface::{ConnectionType, FrameMode, MadiChannel, MadiError, MadiInterface};
