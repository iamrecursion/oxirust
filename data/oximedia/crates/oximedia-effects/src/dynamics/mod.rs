//! Dynamics processing effects.
//!
//! Provides dynamics control:
//!
//! - **Gate** - Noise gate with hysteresis
//! - **Expander** - Upward and downward expansion
//! - **Lookahead Limiter** - Broadcast-quality true-peak limiter
//! - **Multi-band Compressor** - Three-band independent compression
//! - **LUFS Meter** - EBU R128 integrated loudness metering effect

pub mod expander;
pub mod gate;
pub mod lookahead_limiter;
pub mod lufs_meter;
pub mod multiband;

// Re-exports
pub use expander::{Expander, ExpanderConfig, ExpanderType};
pub use gate::{Gate, GateConfig};
pub use lookahead_limiter::{LookaheadLimiter, LookaheadLimiterConfig};
pub use lufs_meter::LufsMeter;
pub use multiband::{MultibandCompressor, MultibandCompressorConfig};
