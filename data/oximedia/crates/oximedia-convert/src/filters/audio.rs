// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Audio filters.

use serde::{Deserialize, Serialize};

/// Audio filter type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AudioFilter {
    /// Adjust volume (0.0-2.0, 1.0 = no change)
    VolumeAdjust(f64),
    /// Normalize audio
    Normalize(NormalizeMode),
    /// Apply equalizer
    Equalizer(Vec<EqualizerBand>),
    /// Remove DC offset
    DcRemove,
    /// Apply compressor
    Compressor(CompressorParams),
    /// Apply limiter
    Limiter(LimiterParams),
    /// Apply fade in
    FadeIn(f64),
    /// Apply fade out
    FadeOut(f64),
    /// Remove silence
    SilenceRemove(SilenceParams),
}

/// Normalization mode.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NormalizeMode {
    /// Peak normalization (target dBFS)
    Peak(f64),
    /// RMS normalization (target dBFS)
    Rms(f64),
    /// EBU R128 loudness normalization (target LUFS)
    EbuR128(f64),
}

/// Equalizer band.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EqualizerBand {
    /// Center frequency in Hz
    pub frequency: f64,
    /// Gain in dB
    pub gain: f64,
    /// Q factor (bandwidth)
    pub q: f64,
}

/// Compressor parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CompressorParams {
    /// Threshold in dBFS
    pub threshold: f64,
    /// Ratio (e.g., 4.0 means 4:1)
    pub ratio: f64,
    /// Attack time in milliseconds
    pub attack_ms: f64,
    /// Release time in milliseconds
    pub release_ms: f64,
    /// Knee width in dB
    pub knee_db: f64,
}

impl Default for CompressorParams {
    fn default() -> Self {
        Self {
            threshold: -20.0,
            ratio: 4.0,
            attack_ms: 5.0,
            release_ms: 50.0,
            knee_db: 2.5,
        }
    }
}

/// Limiter parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LimiterParams {
    /// Ceiling level in dBFS
    pub ceiling: f64,
    /// Attack time in milliseconds
    pub attack_ms: f64,
    /// Release time in milliseconds
    pub release_ms: f64,
}

impl Default for LimiterParams {
    fn default() -> Self {
        Self {
            ceiling: -0.1,
            attack_ms: 1.0,
            release_ms: 100.0,
        }
    }
}

/// Silence removal parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SilenceParams {
    /// Threshold in dBFS
    pub threshold: f64,
    /// Minimum silence duration in seconds
    pub min_duration: f64,
}

impl Default for SilenceParams {
    fn default() -> Self {
        Self {
            threshold: -50.0,
            min_duration: 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volume_adjust() {
        let filter = AudioFilter::VolumeAdjust(1.5);
        match filter {
            AudioFilter::VolumeAdjust(v) => assert_eq!(v, 1.5),
            _ => panic!("Wrong filter type"),
        }
    }

    #[test]
    fn test_compressor_default() {
        let params = CompressorParams::default();
        assert_eq!(params.threshold, -20.0);
        assert_eq!(params.ratio, 4.0);
    }

    #[test]
    fn test_limiter_default() {
        let params = LimiterParams::default();
        assert_eq!(params.ceiling, -0.1);
    }

    #[test]
    fn test_silence_params_default() {
        let params = SilenceParams::default();
        assert_eq!(params.threshold, -50.0);
        assert_eq!(params.min_duration, 0.5);
    }

    #[test]
    fn test_equalizer_band() {
        let band = EqualizerBand {
            frequency: 1000.0,
            gain: 3.0,
            q: 1.0,
        };
        assert_eq!(band.frequency, 1000.0);
        assert_eq!(band.gain, 3.0);
    }
}
