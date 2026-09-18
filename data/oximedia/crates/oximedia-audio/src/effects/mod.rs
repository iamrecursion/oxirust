//! Advanced audio modulation effects for `OxiMedia`.
//!
//! This module provides production-quality implementations of classic
//! modulation effects used in music production and audio processing:
//!
//! # Effects
//!
//! - **Chorus** - Multi-voice chorus with LFO modulation for thicker, richer sounds
//! - **Flanger** - Short delay with feedback and LFO for jet-like sweeping sounds
//! - **Phaser** - All-pass filter cascade with LFO for frequency-dependent phase shifts
//!
//! # Common Infrastructure
//!
//! - **LFO** - Low-frequency oscillator with multiple waveforms (sine, triangle, sawtooth, square, random)
//! - **Delay Lines** - Fractional delay with interpolation for smooth modulation
//! - **All-pass Filters** - Phase-shifting filters for phaser effects
//! - **Parameter Smoothing** - Smooth parameter changes to avoid audio artifacts
//!
//! # Examples
//!
//! ## Chorus Effect
//!
//! ```ignore
//! use oximedia_audio::effects::{StereoChorus, ChorusConfig};
//!
//! let config = ChorusConfig::lush(); // 6-voice lush chorus preset
//! let mut chorus = StereoChorus::new(config, 48000.0);
//!
//! // Process stereo audio (planar format)
//! let mut left = vec![0.0; 1024];
//! let mut right = vec![0.0; 1024];
//! chorus.process_planar(&mut left, &mut right);
//! ```
//!
//! ## Flanger Effect
//!
//! ```ignore
//! use oximedia_audio::effects::{StereoFlanger, FlangerConfig};
//!
//! let config = FlangerConfig::jet(); // Fast, high-feedback jet preset
//! let mut flanger = StereoFlanger::new(config, 48000.0);
//!
//! // Process stereo audio (interleaved format)
//! let mut samples = vec![0.0; 2048];
//! flanger.process_interleaved(&mut samples, 1024);
//! ```
//!
//! ## Phaser Effect
//!
//! ```ignore
//! use oximedia_audio::effects::{StereoPhaser, PhaserConfig, LfoWaveform};
//!
//! let config = PhaserConfig::new(8, 0.5, 1.0) // 8 stages, 0.5 Hz, full depth
//!     .with_resonance(0.7)
//!     .with_waveform(LfoWaveform::Triangle);
//! let mut phaser = StereoPhaser::new(config, 48000.0);
//!
//! // Process mono audio
//! let mut samples = vec![0.0; 1024];
//! phaser.process(&mut samples);
//! ```
//!
//! ## Custom LFO
//!
//! ```ignore
//! use oximedia_audio::effects::{Lfo, LfoWaveform};
//!
//! let mut lfo = Lfo::new(2.0, 48000.0, LfoWaveform::Sine);
//!
//! for _ in 0..100 {
//!     let value = lfo.next(); // Value between -1.0 and 1.0
//!     println!("LFO: {}", value);
//! }
//! ```
//!
//! # Technical Details
//!
//! ## Chorus
//!
//! The chorus effect works by creating multiple delayed copies (voices) of the
//! input signal, each with a slightly different delay time that varies with an LFO.
//! This simulates the natural variations that occur when multiple musicians
//! play the same part together.
//!
//! Key features:
//! - 2-8 voices with independent LFO phases
//! - Configurable base delay (10-50ms) and modulation depth (0-20ms)
//! - Stereo spread control for wider stereo image
//! - Optional feedback for more pronounced effect
//! - Multiple waveform types for different modulation characteristics
//!
//! ## Flanger
//!
//! The flanger uses a very short delay line (0.5-10ms) with feedback, modulated
//! by an LFO. When the delayed signal is mixed with the original, it creates
//! a comb filter with moving notches that produce the characteristic sweeping sound.
//!
//! Key features:
//! - Short delay times for distinctive flanging sound
//! - Positive and negative feedback modes
//! - High feedback creates more pronounced resonant peaks
//! - Linear interpolation for smooth delay modulation
//! - Stereo processing with independent LFO phases
//!
//! ## Phaser
//!
//! The phaser uses a cascade of all-pass filters whose cutoff frequencies are
//! modulated by an LFO. Each all-pass filter introduces a frequency-dependent
//! phase shift. When the phase-shifted signal is mixed with the original,
//! notches appear in the frequency spectrum at points where the signals are
//! out of phase.
//!
//! Key features:
//! - Configurable number of stages (2-12, even numbers only)
//! - First-order all-pass filters for efficient processing
//! - Feedback/resonance control for more pronounced notches
//! - Wide frequency range (200-5000 Hz center, ±3000 Hz range)
//! - Positive and negative feedback modes
//!
//! ## LFO (Low-Frequency Oscillator)
//!
//! The LFO generates periodic control signals used to modulate effect parameters.
//! It supports multiple waveforms for different modulation characteristics:
//!
//! - **Sine**: Smooth, natural modulation
//! - **Triangle**: Linear rise and fall
//! - **Sawtooth**: Linear rise with instant reset (good for "barber pole" effects)
//! - **Square**: Instant switching (trill-like effects)
//! - **Random**: Stepped random values (sample-and-hold)
//!
//! ## Delay Lines
//!
//! The fractional delay line implementation supports reading at non-integer
//! delay times using interpolation:
//!
//! - **None**: Nearest sample (no interpolation)
//! - **Linear**: Linear interpolation between two samples
//! - **Cubic**: Hermite interpolation using four samples (higher quality)
//!
//! Linear interpolation provides a good balance between quality and performance
//! for most modulation effects.
//!
//! ## Parameter Smoothing
//!
//! All effects use parameter smoothers to avoid audio artifacts (clicks, pops)
//! when parameters change. The smoother uses a one-pole lowpass filter with
//! a configurable time constant (typically 10ms).
//!
//! # Performance
//!
//! All effects are optimized for real-time processing:
//!
//! - No dynamic allocations during processing
//! - Efficient circular buffer implementation for delay lines
//! - SIMD-friendly processing (single sample at a time or buffer-based)
//! - Minimal branching in inner loops
//!
//! # Thread Safety
//!
//! Effect processors are not `Send` or `Sync` by default. For multi-threaded
//! use, create separate effect instances per thread.

#![forbid(unsafe_code)]

pub mod chorus;
pub mod delay;
pub mod flanger;
pub mod lfo;
pub mod phaser;

// Re-exports for convenience
pub use chorus::{ChorusConfig, MonoChorus, StereoChorus, MAX_CHORUS_VOICES, MIN_CHORUS_VOICES};
pub use delay::{
    AllPassFilter, FirstOrderAllPass, FractionalDelayLine, InterpolationMode, ModulatedDelayLine,
};
pub use flanger::{FeedbackMode, FlangerConfig, MonoFlanger, StereoFlanger};
pub use lfo::{Lfo, LfoWaveform, ParameterSmoother, StereoLfo};
pub use phaser::{
    MonoPhaser, PhaserConfig, PhaserFeedbackMode, StereoPhaser, MAX_PHASER_STAGES,
    MIN_PHASER_STAGES,
};
