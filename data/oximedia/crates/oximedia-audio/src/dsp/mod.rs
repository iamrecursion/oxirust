//! Digital Signal Processing (DSP) implementations for audio.
//!
//! This module provides standalone DSP implementations for common audio effects:
//!
//! - **Biquad Filters** - Second-order IIR filters (low-pass, high-pass, etc.)
//! - **Parametric Equalizer** - Multi-band EQ using biquad filters
//! - **Dynamics Compressor** - Full-featured compressor with threshold, ratio, attack, release, knee, and makeup gain
//! - **Limiter** - Brickwall peak limiter with lookahead
//! - **Delay** - Delay effects with feedback, ping-pong, and multi-tap modes
//! - **Reverb** - Schroeder reverb using parallel comb filters and series all-pass filters
//!
//! # Example
//!
//! ```ignore
//! use oximedia_audio::dsp::{Equalizer, EqualizerConfig, EqBand};
//!
//! // Create a 3-band EQ
//! let config = EqualizerConfig::three_band(3.0, 0.0, -2.0);
//! let mut eq = Equalizer::new(config, 48000.0, 2);
//!
//! // Process audio
//! let mut samples = vec![vec![0.0; 1024]; 2];
//! eq.process_planar(&mut samples);
//! ```

#![forbid(unsafe_code)]

pub mod biquad;
pub mod compressor;
pub mod delay;
pub mod eq;
pub mod limiter;
pub mod reverb;

// Re-exports for convenience
pub use biquad::{BiquadCoefficients, BiquadState, BiquadType, CascadedBiquad};
pub use compressor::{Compressor, CompressorConfig, KneeType};
pub use delay::{DelayConfig, DelayLine, DelayMode, MonoDelay, MultiTapDelay, StereoDelay};
pub use eq::{EqBand, Equalizer, EqualizerConfig, MAX_EQ_BANDS};
pub use limiter::{Limiter, LimiterConfig, TruePeakLimiter};
pub use reverb::{MonoReverb, Reverb, ReverbConfig};
