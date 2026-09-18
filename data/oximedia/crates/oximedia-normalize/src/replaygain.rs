//! ReplayGain calculation and tagging.
//!
//! Implements ReplayGain 2.0 specification for track and album gain calculation.

use crate::{NormalizeError, NormalizeResult};
use oximedia_metering::{LoudnessMeter, MeterConfig, Standard};

/// ReplayGain reference level in dB SPL.
///
/// The ReplayGain specification defines 89 dB SPL as the reference playback level,
/// which corresponds to approximately -18 LUFS.
pub const REPLAYGAIN_REFERENCE_LUFS: f64 = -18.0;

/// ReplayGain values for a track or album.
#[derive(Clone, Debug)]
pub struct ReplayGainValues {
    /// Track gain in dB.
    pub track_gain: f64,

    /// Track peak (linear, 0.0-1.0+).
    pub track_peak: f64,

    /// Album gain in dB (if calculated).
    pub album_gain: Option<f64>,

    /// Album peak (if calculated).
    pub album_peak: Option<f64>,

    /// Reference loudness used for calculation.
    pub reference_lufs: f64,
}

impl ReplayGainValues {
    /// Create new ReplayGain values with track data only.
    pub fn new(track_gain: f64, track_peak: f64) -> Self {
        Self {
            track_gain,
            track_peak,
            album_gain: None,
            album_peak: None,
            reference_lufs: REPLAYGAIN_REFERENCE_LUFS,
        }
    }

    /// Set album gain and peak.
    pub fn with_album(mut self, album_gain: f64, album_peak: f64) -> Self {
        self.album_gain = Some(album_gain);
        self.album_peak = Some(album_peak);
        self
    }

    /// Format as ID3v2 TXXX frames.
    pub fn to_id3v2_tags(&self) -> Vec<(String, String)> {
        let mut tags = vec![
            (
                "REPLAYGAIN_TRACK_GAIN".to_string(),
                format!("{:+.2} dB", self.track_gain),
            ),
            (
                "REPLAYGAIN_TRACK_PEAK".to_string(),
                format!("{:.6}", self.track_peak),
            ),
        ];

        if let Some(album_gain) = self.album_gain {
            tags.push((
                "REPLAYGAIN_ALBUM_GAIN".to_string(),
                format!("{album_gain:+.2} dB"),
            ));
        }

        if let Some(album_peak) = self.album_peak {
            tags.push((
                "REPLAYGAIN_ALBUM_PEAK".to_string(),
                format!("{album_peak:.6}"),
            ));
        }

        tags.push((
            "REPLAYGAIN_REFERENCE_LOUDNESS".to_string(),
            format!("{:.1} LUFS", self.reference_lufs),
        ));

        tags
    }

    /// Format as Vorbis comments.
    pub fn to_vorbis_comments(&self) -> Vec<(String, String)> {
        self.to_id3v2_tags() // Same format
    }

    /// Format as APEv2 tags.
    pub fn to_apev2_tags(&self) -> Vec<(String, String)> {
        self.to_id3v2_tags() // Same format
    }
}

/// ReplayGain calculator.
///
/// Calculates ReplayGain values according to the ReplayGain 2.0 specification.
pub struct ReplayGainCalculator {
    meter: LoudnessMeter,
}

impl ReplayGainCalculator {
    /// Create a new ReplayGain calculator.
    pub fn new(sample_rate: f64, channels: usize) -> NormalizeResult<Self> {
        // ReplayGain uses a custom target
        let standard = Standard::Custom {
            target_lufs: REPLAYGAIN_REFERENCE_LUFS,
            max_peak_dbtp: -1.0,
            tolerance_lu: 1.0,
        };

        let config = MeterConfig::new(standard, sample_rate, channels);
        let meter = LoudnessMeter::new(config)?;

        Ok(Self { meter })
    }

    /// Process f32 audio samples.
    pub fn process_f32(&mut self, samples: &[f32]) {
        self.meter.process_f32(samples);
    }

    /// Process f64 audio samples.
    pub fn process_f64(&mut self, samples: &[f64]) {
        self.meter.process_f64(samples);
    }

    /// Calculate ReplayGain values for the processed audio.
    pub fn calculate(&mut self) -> NormalizeResult<ReplayGainValues> {
        let metrics = self.meter.metrics();

        if !metrics.integrated_lufs.is_finite() {
            return Err(NormalizeError::InsufficientData(
                "Not enough audio data to calculate ReplayGain".to_string(),
            ));
        }

        // Track gain is the difference between reference and measured loudness
        let track_gain = REPLAYGAIN_REFERENCE_LUFS - metrics.integrated_lufs;

        // Track peak is the maximum true peak (linear scale)
        let track_peak = metrics.true_peak_linear;

        Ok(ReplayGainValues::new(track_gain, track_peak))
    }

    /// Reset the calculator for a new track.
    pub fn reset(&mut self) {
        self.meter.reset();
    }

    /// Get the underlying loudness meter.
    pub fn meter(&self) -> &LoudnessMeter {
        &self.meter
    }

    /// Get the underlying loudness meter (mutable).
    pub fn meter_mut(&mut self) -> &mut LoudnessMeter {
        &mut self.meter
    }
}

/// Album ReplayGain calculator.
///
/// Calculates album-wide ReplayGain by analyzing multiple tracks.
pub struct AlbumReplayGainCalculator {
    tracks: Vec<ReplayGainValues>,
}

impl AlbumReplayGainCalculator {
    /// Create a new album calculator.
    pub fn new() -> Self {
        Self { tracks: Vec::new() }
    }

    /// Add a track's ReplayGain values.
    pub fn add_track(&mut self, track: ReplayGainValues) {
        self.tracks.push(track);
    }

    /// Calculate album ReplayGain values.
    ///
    /// Album gain is calculated by averaging the track loudness values,
    /// and album peak is the maximum of all track peaks.
    pub fn calculate(&self) -> NormalizeResult<f64> {
        if self.tracks.is_empty() {
            return Err(NormalizeError::InsufficientData(
                "No tracks added to album calculator".to_string(),
            ));
        }

        // Calculate average loudness across all tracks
        // Album gain is calculated from the average of the individual track loudnesses,
        // not the average of the track gains
        let total_loudness: f64 = self
            .tracks
            .iter()
            .map(|t| REPLAYGAIN_REFERENCE_LUFS - t.track_gain)
            .sum();
        let avg_loudness = total_loudness / self.tracks.len() as f64;
        let album_gain = REPLAYGAIN_REFERENCE_LUFS - avg_loudness;

        Ok(album_gain)
    }

    /// Calculate album peak.
    pub fn album_peak(&self) -> f64 {
        self.tracks.iter().map(|t| t.track_peak).fold(0.0, f64::max)
    }

    /// Apply album gain and peak to all tracks.
    pub fn finalize_tracks(&mut self) -> Vec<ReplayGainValues> {
        let album_gain = match self.calculate() {
            Ok(gain) => gain,
            Err(_) => return Vec::new(),
        };

        let album_peak = self.album_peak();

        self.tracks
            .iter()
            .map(|track| {
                let mut updated = track.clone();
                updated.album_gain = Some(album_gain);
                updated.album_peak = Some(album_peak);
                updated
            })
            .collect()
    }

    /// Get the number of tracks.
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Clear all tracks.
    pub fn clear(&mut self) {
        self.tracks.clear();
    }
}

impl Default for AlbumReplayGainCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replaygain_values() {
        let rg = ReplayGainValues::new(3.5, 0.9);
        assert_eq!(rg.track_gain, 3.5);
        assert_eq!(rg.track_peak, 0.9);
        assert!(rg.album_gain.is_none());

        let rg_album = rg.with_album(2.0, 0.95);
        assert_eq!(rg_album.album_gain, Some(2.0));
        assert_eq!(rg_album.album_peak, Some(0.95));
    }

    #[test]
    fn test_id3v2_tags() {
        let rg = ReplayGainValues::new(3.5, 0.9).with_album(2.0, 0.95);
        let tags = rg.to_id3v2_tags();

        assert!(tags.iter().any(|(k, _)| k == "REPLAYGAIN_TRACK_GAIN"));
        assert!(tags.iter().any(|(k, _)| k == "REPLAYGAIN_TRACK_PEAK"));
        assert!(tags.iter().any(|(k, _)| k == "REPLAYGAIN_ALBUM_GAIN"));
        assert!(tags.iter().any(|(k, _)| k == "REPLAYGAIN_ALBUM_PEAK"));
    }

    #[test]
    fn test_calculator_creation() {
        let calc = ReplayGainCalculator::new(48000.0, 2);
        assert!(calc.is_ok());
    }

    #[test]
    fn test_album_calculator() {
        let mut album = AlbumReplayGainCalculator::new();
        assert_eq!(album.track_count(), 0);

        album.add_track(ReplayGainValues::new(3.0, 0.9));
        album.add_track(ReplayGainValues::new(2.0, 0.8));
        assert_eq!(album.track_count(), 2);

        let album_gain = album.calculate().expect("should succeed in test");
        assert!(album_gain.is_finite());

        let album_peak = album.album_peak();
        assert_eq!(album_peak, 0.9);
    }

    #[test]
    fn test_album_finalize() {
        let mut album = AlbumReplayGainCalculator::new();
        album.add_track(ReplayGainValues::new(3.0, 0.9));
        album.add_track(ReplayGainValues::new(2.0, 0.8));

        let finalized = album.finalize_tracks();
        assert_eq!(finalized.len(), 2);
        assert!(finalized[0].album_gain.is_some());
        assert!(finalized[1].album_gain.is_some());
    }
}
