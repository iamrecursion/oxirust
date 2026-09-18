//! Audio filters for the filter graph.
//!
//! This module provides audio processing filters including:
//!
//! - [`AudioPassthrough`] - Passes audio frames through unchanged
//! - [`ResampleFilter`] - High-quality sample rate conversion
//! - [`ChannelMixFilter`] - Channel mixing and routing
//! - [`VolumeFilter`] - Volume control with fade support
//! - [`NormalizeFilter`] - Audio normalization (Peak, RMS, EBU R128)
//! - [`EqualizerFilter`] - Multi-band parametric equalizer
//! - [`CompressorFilter`] - Dynamics compression
//! - [`LimiterFilter`] - Brickwall and soft limiting
//! - [`DelayFilter`] - Delay effect with feedback
//! - [`TrimFilter`] - Audio trimming with fades

pub mod compressor;
pub mod delay;
pub mod eq;
pub mod limiter;
pub mod mixer;
pub mod normalize;
mod passthrough;
pub mod resample;
pub mod trim;
pub mod volume;

pub use compressor::{CompressorConfig, CompressorFilter, KneeType};
pub use delay::{DelayConfig, DelayFilter, DelayMode};
pub use eq::{BandType, EqBand, EqualizerConfig, EqualizerFilter, MAX_BANDS};
pub use limiter::{LimiterConfig, LimiterFilter, LimiterMode};
pub use mixer::{ChannelMixConfig, ChannelMixFilter, CrossfadeConfig, MixMatrix, MAX_CHANNELS};
pub use normalize::{NormalizationMode, NormalizeConfig, NormalizeFilter};
pub use passthrough::AudioPassthrough;
pub use resample::{ResampleConfig, ResampleFilter, ResampleQuality};
pub use trim::{TrimConfig, TrimFilter, TrimMode};
pub use volume::{FadeConfig, FadeDirection, VolumeConfig, VolumeFilter};
