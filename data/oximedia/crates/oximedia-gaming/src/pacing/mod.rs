//! Frame pacing and timing control.
//!
//! This module provides precise frame timing and buffer management:
//!
//! - Frame pacing for consistent delivery
//! - Buffer management for smooth playback
//! - Jitter reduction
//! - Adaptive timing

pub mod buffer;
pub mod frame;

pub use buffer::{BufferConfig, FrameBuffer};
pub use frame::{FramePacer, FrameTimingInfo, PacingMode};
