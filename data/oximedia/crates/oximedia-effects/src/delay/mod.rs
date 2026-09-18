//! Delay and echo effects.
//!
//! Provides various delay-based effects:
//!
//! - **Basic Delay** - Simple delay with feedback and tone control
//! - **Multi-tap Delay** - Multiple delay taps with independent controls
//! - **Ping-pong Delay** - Stereo bouncing delay effect

#[allow(clippy::module_inception)]
pub mod delay;
pub mod multitap;
pub mod pingpong;

// Re-exports
pub use delay::{DelayConfig, MonoDelay, StereoDelay};
pub use multitap::{DelayTap, MultiTapDelay, MAX_TAPS};
pub use pingpong::{PingPongConfig, PingPongDelay};
