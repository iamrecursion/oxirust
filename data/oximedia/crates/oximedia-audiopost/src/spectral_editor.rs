#![allow(dead_code)]
//! Spectral editing operations for audio post-production.
//!
//! Provides frequency-domain selection, attenuation, and transplant tools
//! that allow precise editing of audio in the spectral domain. Useful for
//! isolating or removing specific frequency content, such as hum removal,
//! breath attenuation, or surgical noise removal.

use std::f64::consts::PI;

/// A rectangular region in the time-frequency plane.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralRegion {
    /// Start time in seconds.
    pub time_start: f64,
    /// End time in seconds.
    pub time_end: f64,
    /// Lower frequency bound in Hz.
    pub freq_low: f64,
    /// Upper frequency bound in Hz.
    pub freq_high: f64,
}

impl SpectralRegion {
    /// Create a new spectral region.
    pub fn new(time_start: f64, time_end: f64, freq_low: f64, freq_high: f64) -> Self {
        Self {
            time_start,
            time_end,
            freq_low,
            freq_high,
        }
    }

    /// Duration of the region in seconds.
    pub fn duration(&self) -> f64 {
        (self.time_end - self.time_start).max(0.0)
    }

    /// Bandwidth of the region in Hz.
    pub fn bandwidth(&self) -> f64 {
        (self.freq_high - self.freq_low).max(0.0)
    }

    /// Check if a time-frequency point falls within this region.
    pub fn contains(&self, time: f64, freq: f64) -> bool {
        time >= self.time_start
            && time <= self.time_end
            && freq >= self.freq_low
            && freq <= self.freq_high
    }

    /// Check if two regions overlap.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.time_start < other.time_end
            && self.time_end > other.time_start
            && self.freq_low < other.freq_high
            && self.freq_high > other.freq_low
    }
}

/// Type of spectral edit operation.
#[derive(Debug, Clone, PartialEq)]
pub enum SpectralEditOp {
    /// Attenuate the selected region by the given amount in dB (positive = reduce).
    Attenuate(f64),
    /// Boost the selected region by the given amount in dB.
    Boost(f64),
    /// Replace the content with silence.
    Silence,
    /// Apply a fade-in across the region's time axis.
    FadeIn,
    /// Apply a fade-out across the region's time axis.
    FadeOut,
    /// Interpolate from surrounding frequency content.
    Interpolate,
}

/// A single spectral edit: a region paired with an operation.
#[derive(Debug, Clone)]
pub struct SpectralEdit {
    /// The region to edit.
    pub region: SpectralRegion,
    /// The operation to apply.
    pub operation: SpectralEditOp,
    /// Feather radius in Hz for smooth transitions.
    pub feather_hz: f64,
    /// Feather radius in seconds for smooth transitions.
    pub feather_time: f64,
}

impl SpectralEdit {
    /// Create a new spectral edit.
    pub fn new(region: SpectralRegion, operation: SpectralEditOp) -> Self {
        Self {
            region,
            operation,
            feather_hz: 0.0,
            feather_time: 0.0,
        }
    }

    /// Set feathering for smooth transitions.
    pub fn with_feather(mut self, hz: f64, time: f64) -> Self {
        self.feather_hz = hz.max(0.0);
        self.feather_time = time.max(0.0);
        self
    }

    /// Compute the gain factor for a given time-frequency point.
    #[allow(clippy::cast_precision_loss)]
    pub fn gain_at(&self, time: f64, freq: f64) -> f64 {
        let inside = self.region.contains(time, freq);

        // Compute feather weight if outside but within feather range.
        let weight = if inside {
            1.0
        } else if self.feather_hz > 0.0 || self.feather_time > 0.0 {
            self.feather_weight(time, freq)
        } else {
            0.0
        };

        if weight <= 0.0 {
            return 1.0; // No change outside feathered region.
        }

        let raw_gain = match &self.operation {
            SpectralEditOp::Attenuate(db) => db_to_linear(-db),
            SpectralEditOp::Boost(db) => db_to_linear(*db),
            SpectralEditOp::Silence => 0.0,
            SpectralEditOp::FadeIn => {
                let t = if self.region.duration() > 0.0 {
                    ((time - self.region.time_start) / self.region.duration()).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                t
            }
            SpectralEditOp::FadeOut => {
                let t = if self.region.duration() > 0.0 {
                    1.0 - ((time - self.region.time_start) / self.region.duration()).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                t
            }
            SpectralEditOp::Interpolate => {
                // Simplified: reduce magnitude by 50% (real implementation would use
                // surrounding bins).
                0.5
            }
        };

        // Blend between unity gain and the edit gain based on feather weight.
        1.0 + weight * (raw_gain - 1.0)
    }

    /// Compute feather weight (0..1) for points outside the main region.
    fn feather_weight(&self, time: f64, freq: f64) -> f64 {
        let dt = if time < self.region.time_start {
            self.region.time_start - time
        } else if time > self.region.time_end {
            time - self.region.time_end
        } else {
            0.0
        };

        let df = if freq < self.region.freq_low {
            self.region.freq_low - freq
        } else if freq > self.region.freq_high {
            freq - self.region.freq_high
        } else {
            0.0
        };

        let t_weight = if self.feather_time > 0.0 {
            (1.0 - dt / self.feather_time).max(0.0)
        } else if dt > 0.0 {
            0.0
        } else {
            1.0
        };

        let f_weight = if self.feather_hz > 0.0 {
            (1.0 - df / self.feather_hz).max(0.0)
        } else if df > 0.0 {
            0.0
        } else {
            1.0
        };

        t_weight * f_weight
    }
}

/// Convert dB to linear gain.
#[allow(clippy::cast_precision_loss)]
fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// Convert linear gain to dB.
#[allow(clippy::cast_precision_loss)]
fn linear_to_db(linear: f64) -> f64 {
    if linear <= 0.0 {
        return f64::NEG_INFINITY;
    }
    20.0 * linear.log10()
}

/// A Hann window function for spectral analysis.
#[allow(clippy::cast_precision_loss)]
pub fn hann_window(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / size as f64).cos()))
        .collect()
}

/// Compute the frequency bin index for a given frequency.
#[allow(clippy::cast_precision_loss)]
pub fn freq_to_bin(freq: f64, sample_rate: f64, fft_size: usize) -> usize {
    let bin = (freq * fft_size as f64 / sample_rate).round() as usize;
    bin.min(fft_size / 2)
}

/// Compute the frequency for a given bin index.
#[allow(clippy::cast_precision_loss)]
pub fn bin_to_freq(bin: usize, sample_rate: f64, fft_size: usize) -> f64 {
    bin as f64 * sample_rate / fft_size as f64
}

/// A collection of spectral edits applied to a single audio clip.
#[derive(Debug, Clone)]
pub struct SpectralEditSession {
    /// Sample rate of the audio.
    pub sample_rate: f64,
    /// FFT size used for analysis.
    pub fft_size: usize,
    /// Hop size (overlap).
    pub hop_size: usize,
    /// List of edits in this session.
    edits: Vec<SpectralEdit>,
}

impl SpectralEditSession {
    /// Create a new spectral edit session.
    pub fn new(sample_rate: f64, fft_size: usize, hop_size: usize) -> Self {
        Self {
            sample_rate,
            fft_size,
            hop_size,
            edits: Vec::new(),
        }
    }

    /// Add an edit to the session.
    pub fn add_edit(&mut self, edit: SpectralEdit) {
        self.edits.push(edit);
    }

    /// Return the number of edits.
    pub fn edit_count(&self) -> usize {
        self.edits.len()
    }

    /// Compute the combined gain at a time-frequency point across all edits.
    pub fn combined_gain_at(&self, time: f64, freq: f64) -> f64 {
        let mut gain = 1.0;
        for edit in &self.edits {
            gain *= edit.gain_at(time, freq);
        }
        gain
    }

    /// Clear all edits.
    pub fn clear(&mut self) {
        self.edits.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_region_creation() {
        let r = SpectralRegion::new(0.0, 1.0, 100.0, 5000.0);
        assert_eq!(r.duration(), 1.0);
        assert_eq!(r.bandwidth(), 4900.0);
    }

    #[test]
    fn test_spectral_region_contains() {
        let r = SpectralRegion::new(1.0, 2.0, 200.0, 400.0);
        assert!(r.contains(1.5, 300.0));
        assert!(!r.contains(0.5, 300.0));
        assert!(!r.contains(1.5, 500.0));
    }

    #[test]
    fn test_spectral_region_overlaps() {
        let r1 = SpectralRegion::new(0.0, 2.0, 100.0, 500.0);
        let r2 = SpectralRegion::new(1.0, 3.0, 400.0, 800.0);
        assert!(r1.overlaps(&r2));

        let r3 = SpectralRegion::new(3.0, 4.0, 100.0, 500.0);
        assert!(!r1.overlaps(&r3));
    }

    #[test]
    fn test_attenuate_gain() {
        let region = SpectralRegion::new(0.0, 1.0, 0.0, 1000.0);
        let edit = SpectralEdit::new(region, SpectralEditOp::Attenuate(6.0));
        let gain = edit.gain_at(0.5, 500.0);
        // -6dB is ~0.5012
        assert!((gain - 0.5012).abs() < 0.01);
    }

    #[test]
    fn test_boost_gain() {
        let region = SpectralRegion::new(0.0, 1.0, 0.0, 1000.0);
        let edit = SpectralEdit::new(region, SpectralEditOp::Boost(6.0));
        let gain = edit.gain_at(0.5, 500.0);
        // +6dB is ~1.995
        assert!((gain - 1.995).abs() < 0.01);
    }

    #[test]
    fn test_silence_gain() {
        let region = SpectralRegion::new(0.0, 1.0, 0.0, 1000.0);
        let edit = SpectralEdit::new(region, SpectralEditOp::Silence);
        assert_eq!(edit.gain_at(0.5, 500.0), 0.0);
        // Outside the region: unity.
        assert_eq!(edit.gain_at(2.0, 500.0), 1.0);
    }

    #[test]
    fn test_fade_in() {
        let region = SpectralRegion::new(0.0, 1.0, 0.0, 1000.0);
        let edit = SpectralEdit::new(region, SpectralEditOp::FadeIn);
        let gain_start = edit.gain_at(0.0, 500.0);
        let gain_mid = edit.gain_at(0.5, 500.0);
        let gain_end = edit.gain_at(1.0, 500.0);
        assert!(gain_start < gain_mid);
        assert!(gain_mid < gain_end);
    }

    #[test]
    fn test_fade_out() {
        let region = SpectralRegion::new(0.0, 1.0, 0.0, 1000.0);
        let edit = SpectralEdit::new(region, SpectralEditOp::FadeOut);
        let gain_start = edit.gain_at(0.0, 500.0);
        let gain_end = edit.gain_at(1.0, 500.0);
        assert!(gain_start > gain_end);
    }

    #[test]
    fn test_db_conversion() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-10);
        assert!((db_to_linear(-6.0) - 0.5012).abs() < 0.001);
        assert!((linear_to_db(1.0)).abs() < 1e-10);
        assert_eq!(linear_to_db(0.0), f64::NEG_INFINITY);
    }

    #[test]
    fn test_hann_window() {
        let w = hann_window(4);
        assert_eq!(w.len(), 4);
        // Hann window of size 4:
        //   w[0] = 0.5*(1 - cos(0))       = 0.0  (near 0)
        //   w[1] = 0.5*(1 - cos(π/2))     = 0.5
        //   w[2] = 0.5*(1 - cos(π))       = 1.0  (peak, near 1)
        //   w[3] = 0.5*(1 - cos(3π/2))    = 0.5
        assert!(w[0].abs() < 1e-10, "w[0] should be near 0, got {}", w[0]);
        assert!(
            (w[2] - 1.0).abs() < 1e-10,
            "w[2] should be near 1, got {}",
            w[2]
        );
    }

    #[test]
    fn test_freq_bin_conversion() {
        let sr = 48000.0;
        let fft = 1024;
        let bin = freq_to_bin(1000.0, sr, fft);
        let freq = bin_to_freq(bin, sr, fft);
        assert!((freq - 1000.0).abs() < 50.0); // within one bin
    }

    #[test]
    fn test_spectral_edit_session() {
        let mut session = SpectralEditSession::new(48000.0, 2048, 512);
        assert_eq!(session.edit_count(), 0);

        let region = SpectralRegion::new(0.0, 1.0, 0.0, 500.0);
        session.add_edit(SpectralEdit::new(region, SpectralEditOp::Attenuate(12.0)));
        assert_eq!(session.edit_count(), 1);

        // Gain within edit region should be less than 1.
        let gain = session.combined_gain_at(0.5, 250.0);
        assert!(gain < 1.0);

        session.clear();
        assert_eq!(session.edit_count(), 0);
    }

    #[test]
    fn test_feathered_edit() {
        let region = SpectralRegion::new(1.0, 2.0, 200.0, 400.0);
        let edit = SpectralEdit::new(region, SpectralEditOp::Silence).with_feather(50.0, 0.1);
        // Inside region: gain = 0.
        assert_eq!(edit.gain_at(1.5, 300.0), 0.0);
        // Just outside feather range: gain = 1.
        assert_eq!(edit.gain_at(3.0, 300.0), 1.0);
        // Within feather zone (time): gain between 0 and 1.
        let g = edit.gain_at(2.05, 300.0);
        assert!(g > 0.0 && g < 1.0);
    }
}
