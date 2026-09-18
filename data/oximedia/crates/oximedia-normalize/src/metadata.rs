//! Loudness metadata writing.
//!
//! This module handles writing loudness metadata to various audio file formats.

use crate::{AnalysisResult, NormalizeResult, ReplayGainValues};
use oximedia_metering::Standard;

/// Loudness metadata for tagging.
#[derive(Clone, Debug)]
pub struct LoudnessMetadata {
    /// Integrated loudness in LUFS.
    pub integrated_lufs: f64,

    /// Loudness range in LU.
    pub loudness_range: f64,

    /// True peak in dBTP.
    pub true_peak_dbtp: f64,

    /// Applied gain in dB.
    pub applied_gain_db: f64,

    /// Target standard.
    pub standard: Standard,

    /// ReplayGain values (if calculated).
    pub replay_gain: Option<ReplayGainValues>,

    /// EBU R128 metadata.
    pub r128_metadata: Option<R128Metadata>,

    /// iTunes Sound Check metadata.
    pub soundcheck: Option<SoundCheckMetadata>,
}

impl LoudnessMetadata {
    /// Create from analysis result.
    pub fn from_analysis(analysis: AnalysisResult, applied_gain_db: f64) -> Self {
        Self {
            integrated_lufs: analysis.integrated_lufs,
            loudness_range: analysis.loudness_range,
            true_peak_dbtp: analysis.true_peak_dbtp,
            applied_gain_db,
            standard: analysis.standard,
            replay_gain: None,
            r128_metadata: Some(R128Metadata::from_analysis(&analysis)),
            soundcheck: Some(SoundCheckMetadata::from_analysis(&analysis)),
        }
    }

    /// Set ReplayGain values.
    pub fn with_replay_gain(mut self, rg: ReplayGainValues) -> Self {
        self.replay_gain = Some(rg);
        self
    }

    /// Generate all metadata tags.
    pub fn to_tags(&self) -> MetadataTags {
        let mut tags = MetadataTags::new();

        // Add basic loudness tags
        tags.add(
            "LOUDNESS_INTEGRATED",
            format!("{:.2} LUFS", self.integrated_lufs),
        );
        tags.add("LOUDNESS_RANGE", format!("{:.2} LU", self.loudness_range));
        tags.add("TRUE_PEAK", format!("{:.2} dBTP", self.true_peak_dbtp));
        tags.add(
            "NORMALIZATION_GAIN",
            format!("{:+.2} dB", self.applied_gain_db),
        );
        tags.add("NORMALIZATION_STANDARD", self.standard.name().to_string());

        // Add ReplayGain tags
        if let Some(ref rg) = self.replay_gain {
            for (key, value) in rg.to_id3v2_tags() {
                tags.add(&key, value);
            }
        }

        // Add R128 tags
        if let Some(ref r128) = self.r128_metadata {
            for (key, value) in r128.to_tags() {
                tags.add(&key, value);
            }
        }

        // Add iTunes Sound Check
        if let Some(ref sc) = self.soundcheck {
            tags.add("iTunNORM", sc.to_itunes_string());
        }

        tags
    }
}

/// EBU R128 metadata.
#[derive(Clone, Debug)]
pub struct R128Metadata {
    /// Track loudness in LUFS.
    pub track_loudness: f64,

    /// Loudness range in LU.
    pub loudness_range: f64,

    /// Maximum true peak in dBTP.
    pub max_true_peak: f64,

    /// Maximum momentary loudness.
    pub max_momentary: f64,

    /// Maximum short-term loudness.
    pub max_short_term: f64,
}

impl R128Metadata {
    /// Create from analysis result.
    pub fn from_analysis(analysis: &AnalysisResult) -> Self {
        Self {
            track_loudness: analysis.integrated_lufs,
            loudness_range: analysis.loudness_range,
            max_true_peak: analysis.true_peak_dbtp,
            max_momentary: analysis.metrics.max_momentary,
            max_short_term: analysis.metrics.max_short_term,
        }
    }

    /// Convert to metadata tags.
    pub fn to_tags(&self) -> Vec<(String, String)> {
        vec![
            (
                "R128_TRACK_LOUDNESS".to_string(),
                format!("{:.2} LUFS", self.track_loudness),
            ),
            (
                "R128_LOUDNESS_RANGE".to_string(),
                format!("{:.2} LU", self.loudness_range),
            ),
            (
                "R128_TRUE_PEAK".to_string(),
                format!("{:.2} dBTP", self.max_true_peak),
            ),
            (
                "R128_MAX_MOMENTARY".to_string(),
                format!("{:.2} LUFS", self.max_momentary),
            ),
            (
                "R128_MAX_SHORT_TERM".to_string(),
                format!("{:.2} LUFS", self.max_short_term),
            ),
        ]
    }
}

/// iTunes Sound Check metadata.
///
/// Sound Check uses a proprietary format stored in the iTunNORM tag.
#[derive(Clone, Debug)]
pub struct SoundCheckMetadata {
    /// Gain adjustment (linear).
    pub gain: f64,

    /// Peak value (linear).
    pub peak: f64,
}

impl SoundCheckMetadata {
    /// Create from analysis result.
    pub fn from_analysis(analysis: &AnalysisResult) -> Self {
        // iTunes Sound Check reference is approximately -16 LUFS
        let target_lufs = -16.0;
        let gain_db = target_lufs - analysis.integrated_lufs;
        let gain = 10.0_f64.powf(gain_db / 20.0);
        let peak = analysis.metrics.true_peak_linear;

        Self { gain, peak }
    }

    /// Create from ReplayGain values.
    pub fn from_replay_gain(rg: &ReplayGainValues) -> Self {
        let gain = 10.0_f64.powf(rg.track_gain / 20.0);
        Self {
            gain,
            peak: rg.track_peak,
        }
    }

    /// Convert to iTunes iTunNORM string.
    ///
    /// Format: 10 space-separated 8-character hex values
    pub fn to_itunes_string(&self) -> String {
        // Convert gain to iTunes format (inverted and scaled)
        let itunes_gain = (1.0 / self.gain * 1000.0) as u32;
        let itunes_peak = (self.peak * 32768.0) as u32;

        format!(
            "{:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X}",
            itunes_gain,
            itunes_gain,
            itunes_peak,
            itunes_peak,
            0,
            0,
            itunes_gain,
            itunes_gain,
            itunes_peak,
            itunes_peak
        )
    }

    /// Parse from iTunes iTunNORM string.
    pub fn from_itunes_string(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts.len() != 10 {
            return None;
        }

        let gain_hex = u32::from_str_radix(parts[0], 16).ok()?;
        let peak_hex = u32::from_str_radix(parts[2], 16).ok()?;

        let gain = 1.0 / (f64::from(gain_hex) / 1000.0);
        let peak = f64::from(peak_hex) / 32768.0;

        Some(Self { gain, peak })
    }
}

/// Metadata writer.
///
/// Handles writing loudness metadata to audio files.
pub struct MetadataWriter {
    metadata: LoudnessMetadata,
}

impl MetadataWriter {
    /// Create a new metadata writer.
    pub fn new(metadata: LoudnessMetadata) -> Self {
        Self { metadata }
    }

    /// Get all tags as key-value pairs.
    pub fn tags(&self) -> MetadataTags {
        self.metadata.to_tags()
    }

    /// Write to ID3v2 (MP3).
    pub fn write_id3v2(&self) -> NormalizeResult<Vec<(String, String)>> {
        Ok(self.tags().entries)
    }

    /// Write to Vorbis comments (FLAC, OGG).
    pub fn write_vorbis(&self) -> NormalizeResult<Vec<(String, String)>> {
        Ok(self.tags().entries)
    }

    /// Write to APEv2 (APE, Musepack).
    pub fn write_apev2(&self) -> NormalizeResult<Vec<(String, String)>> {
        Ok(self.tags().entries)
    }

    /// Write to MP4/M4A.
    pub fn write_mp4(&self) -> NormalizeResult<Vec<(String, String)>> {
        Ok(self.tags().entries)
    }
}

/// Metadata tags collection.
#[derive(Clone, Debug, Default)]
pub struct MetadataTags {
    /// Tag entries as (key, value) pairs.
    pub entries: Vec<(String, String)>,
}

impl MetadataTags {
    /// Create a new tags collection.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a tag.
    pub fn add(&mut self, key: &str, value: String) {
        self.entries.push((key.to_string(), value));
    }

    /// Get a tag value.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Check if a tag exists.
    pub fn contains(&self, key: &str) -> bool {
        self.entries.iter().any(|(k, _)| k == key)
    }

    /// Get the number of tags.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_r128_metadata() {
        let analysis = create_test_analysis();
        let r128 = R128Metadata::from_analysis(&analysis);
        assert_eq!(r128.track_loudness, -20.0);

        let tags = r128.to_tags();
        assert!(!tags.is_empty());
    }

    #[test]
    fn test_soundcheck_itunes_string() {
        let sc = SoundCheckMetadata {
            gain: 1.0,
            peak: 0.5,
        };

        let itunes_str = sc.to_itunes_string();
        assert!(!itunes_str.is_empty());

        let parsed = SoundCheckMetadata::from_itunes_string(&itunes_str);
        assert!(parsed.is_some());
    }

    #[test]
    fn test_metadata_tags() {
        let mut tags = MetadataTags::new();
        tags.add("TEST_KEY", "test_value".to_string());

        assert!(tags.contains("TEST_KEY"));
        assert_eq!(tags.get("TEST_KEY"), Some("test_value"));
        assert_eq!(tags.len(), 1);
    }

    fn create_test_analysis() -> AnalysisResult {
        use oximedia_metering::{ComplianceResult, LoudnessMetrics};

        AnalysisResult {
            integrated_lufs: -20.0,
            loudness_range: 10.0,
            true_peak_dbtp: -3.0,
            target_lufs: -23.0,
            max_peak_dbtp: -1.0,
            recommended_gain_db: -3.0,
            safe_gain_db: 2.0,
            is_compliant: false,
            compliance: ComplianceResult {
                standard: Standard::EbuR128,
                loudness_compliant: false,
                peak_compliant: true,
                lra_acceptable: true,
                integrated_lufs: -20.0,
                true_peak_dbtp: -3.0,
                loudness_range: 10.0,
                target_lufs: -23.0,
                max_peak_dbtp: -1.0,
                deviation_lu: 3.0,
            },
            metrics: LoudnessMetrics::default(),
            standard: Standard::EbuR128,
        }
    }
}
