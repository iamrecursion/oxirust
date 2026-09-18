//! ReplayGain metadata support.
//!
//! ReplayGain is a standard for measuring and storing perceived loudness of
//! audio content, enabling players to normalize playback volume without
//! destructive modification to the audio data.
//!
//! # Overview
//!
//! ReplayGain specifies two types of gain values:
//!
//! - **Track gain**: Normalizes individual tracks to a reference level.
//! - **Album gain**: Normalizes all tracks in an album to preserve
//!   relative dynamics between tracks.
//!
//! Each gain type is accompanied by a *peak* value (the maximum sample
//! amplitude) used to prevent clipping.
//!
//! The reference level is **-18 LUFS** (equivalent to 89 dB SPL).
//!
//! # Format Mapping
//!
//! | Field              | Vorbis Comment              | ID3v2 (TXXX)                | APEv2              |
//! |--------------------|-----------------------------|-----------------------------|---------------------|
//! | Track Gain         | `REPLAYGAIN_TRACK_GAIN`     | `REPLAYGAIN_TRACK_GAIN`     | `REPLAYGAIN_TRACK_GAIN` |
//! | Track Peak         | `REPLAYGAIN_TRACK_PEAK`     | `REPLAYGAIN_TRACK_PEAK`     | `REPLAYGAIN_TRACK_PEAK` |
//! | Album Gain         | `REPLAYGAIN_ALBUM_GAIN`     | `REPLAYGAIN_ALBUM_GAIN`     | `REPLAYGAIN_ALBUM_GAIN` |
//! | Album Peak         | `REPLAYGAIN_ALBUM_PEAK`     | `REPLAYGAIN_ALBUM_PEAK`     | `REPLAYGAIN_ALBUM_PEAK` |
//! | Reference Loudness | `REPLAYGAIN_REFERENCE_LOUDNESS` | — | — |
//!
//! # Example
//!
//! ```
//! use oximedia_metadata::replaygain::ReplayGain;
//!
//! let rg = ReplayGain::new()
//!     .with_track_gain(-6.5)
//!     .with_track_peak(0.95)
//!     .with_album_gain(-7.2)
//!     .with_album_peak(0.98);
//!
//! assert_eq!(rg.track_gain_db(), Some(-6.5));
//! assert_eq!(rg.track_gain_formatted(), Some("-6.50 dB".to_string()));
//! ```

use crate::{Error, Metadata, MetadataFormat, MetadataValue};

/// Default reference loudness in LUFS (per ReplayGain 2.0).
const DEFAULT_REFERENCE_LOUDNESS: f64 = -18.0;

/// ReplayGain metadata container.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayGain {
    /// Track gain in dB (e.g., -6.50 means reduce by 6.5 dB).
    track_gain_db: Option<f64>,
    /// Track peak amplitude (0.0 to 1.0+, linear scale).
    track_peak: Option<f64>,
    /// Album gain in dB.
    album_gain_db: Option<f64>,
    /// Album peak amplitude (0.0 to 1.0+, linear scale).
    album_peak: Option<f64>,
    /// Reference loudness in LUFS (default: -18.0).
    reference_loudness: f64,
}

impl Default for ReplayGain {
    fn default() -> Self {
        Self {
            track_gain_db: None,
            track_peak: None,
            album_gain_db: None,
            album_peak: None,
            reference_loudness: DEFAULT_REFERENCE_LOUDNESS,
        }
    }
}

impl ReplayGain {
    /// Create a new empty ReplayGain container.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set track gain in dB.
    pub fn with_track_gain(mut self, gain_db: f64) -> Self {
        self.track_gain_db = Some(gain_db);
        self
    }

    /// Set track peak (linear amplitude, typically 0.0..1.0).
    pub fn with_track_peak(mut self, peak: f64) -> Self {
        self.track_peak = Some(peak);
        self
    }

    /// Set album gain in dB.
    pub fn with_album_gain(mut self, gain_db: f64) -> Self {
        self.album_gain_db = Some(gain_db);
        self
    }

    /// Set album peak.
    pub fn with_album_peak(mut self, peak: f64) -> Self {
        self.album_peak = Some(peak);
        self
    }

    /// Set the reference loudness in LUFS.
    pub fn with_reference_loudness(mut self, lufs: f64) -> Self {
        self.reference_loudness = lufs;
        self
    }

    // ---- Getters ----

    /// Track gain in dB.
    pub fn track_gain_db(&self) -> Option<f64> {
        self.track_gain_db
    }

    /// Track peak amplitude.
    pub fn track_peak(&self) -> Option<f64> {
        self.track_peak
    }

    /// Album gain in dB.
    pub fn album_gain_db(&self) -> Option<f64> {
        self.album_gain_db
    }

    /// Album peak amplitude.
    pub fn album_peak(&self) -> Option<f64> {
        self.album_peak
    }

    /// Reference loudness in LUFS.
    pub fn reference_loudness(&self) -> f64 {
        self.reference_loudness
    }

    // ---- Formatted strings (for tag writing) ----

    /// Format track gain as a ReplayGain string (e.g., "-6.50 dB").
    pub fn track_gain_formatted(&self) -> Option<String> {
        self.track_gain_db.map(format_gain)
    }

    /// Format album gain as a ReplayGain string.
    pub fn album_gain_formatted(&self) -> Option<String> {
        self.album_gain_db.map(format_gain)
    }

    /// Format track peak as a string (e.g., "0.950000").
    pub fn track_peak_formatted(&self) -> Option<String> {
        self.track_peak.map(format_peak)
    }

    /// Format album peak as a string.
    pub fn album_peak_formatted(&self) -> Option<String> {
        self.album_peak.map(format_peak)
    }

    // ---- Computed values ----

    /// Compute the linear gain factor for track playback.
    ///
    /// Returns `10^(gain_db / 20)`, which can be multiplied with audio
    /// samples to apply the gain.
    pub fn track_gain_linear(&self) -> Option<f64> {
        self.track_gain_db.map(db_to_linear)
    }

    /// Compute the linear gain factor for album playback.
    pub fn album_gain_linear(&self) -> Option<f64> {
        self.album_gain_db.map(db_to_linear)
    }

    /// Check if applying track gain would clip (peak * gain > 1.0).
    pub fn track_would_clip(&self) -> bool {
        match (self.track_gain_db, self.track_peak) {
            (Some(gain), Some(peak)) => {
                let linear_gain = db_to_linear(gain);
                peak * linear_gain > 1.0
            }
            _ => false,
        }
    }

    /// Check if applying album gain would clip.
    pub fn album_would_clip(&self) -> bool {
        match (self.album_gain_db, self.album_peak) {
            (Some(gain), Some(peak)) => {
                let linear_gain = db_to_linear(gain);
                peak * linear_gain > 1.0
            }
            _ => false,
        }
    }

    /// Compute safe track gain (clamped to avoid clipping).
    ///
    /// If applying the gain would cause the peak to exceed 1.0, the gain
    /// is reduced to keep peak * linear_gain <= 1.0.
    pub fn safe_track_gain_db(&self) -> Option<f64> {
        match (self.track_gain_db, self.track_peak) {
            (Some(gain), Some(peak)) if peak > 0.0 => {
                let max_gain = linear_to_db(1.0 / peak);
                Some(gain.min(max_gain))
            }
            (Some(gain), _) => Some(gain),
            _ => None,
        }
    }

    /// Compute safe album gain (clamped to avoid clipping).
    pub fn safe_album_gain_db(&self) -> Option<f64> {
        match (self.album_gain_db, self.album_peak) {
            (Some(gain), Some(peak)) if peak > 0.0 => {
                let max_gain = linear_to_db(1.0 / peak);
                Some(gain.min(max_gain))
            }
            (Some(gain), _) => Some(gain),
            _ => None,
        }
    }

    /// Returns true if any ReplayGain data is present.
    pub fn has_data(&self) -> bool {
        self.track_gain_db.is_some()
            || self.track_peak.is_some()
            || self.album_gain_db.is_some()
            || self.album_peak.is_some()
    }

    // ---- Metadata integration ----

    /// Extract ReplayGain data from a `Metadata` container.
    ///
    /// Looks for standard ReplayGain field names across Vorbis Comments,
    /// ID3v2 TXXX, and APEv2 conventions.
    pub fn from_metadata(metadata: &Metadata) -> Self {
        let mut rg = ReplayGain::new();

        // Try standard keys (used across Vorbis, APE, ID3v2 TXXX)
        if let Some(val) = get_text(metadata, "REPLAYGAIN_TRACK_GAIN") {
            rg.track_gain_db = parse_gain(val);
        }
        if let Some(val) = get_text(metadata, "REPLAYGAIN_TRACK_PEAK") {
            rg.track_peak = parse_peak(val);
        }
        if let Some(val) = get_text(metadata, "REPLAYGAIN_ALBUM_GAIN") {
            rg.album_gain_db = parse_gain(val);
        }
        if let Some(val) = get_text(metadata, "REPLAYGAIN_ALBUM_PEAK") {
            rg.album_peak = parse_peak(val);
        }
        if let Some(val) = get_text(metadata, "REPLAYGAIN_REFERENCE_LOUDNESS") {
            if let Some(lufs) = parse_gain(val) {
                rg.reference_loudness = lufs;
            }
        }

        rg
    }

    /// Write ReplayGain data into a `Metadata` container.
    ///
    /// Uses the standard uppercase key names.
    pub fn to_metadata(&self, metadata: &mut Metadata) {
        if let Some(gain) = self.track_gain_formatted() {
            metadata.insert(
                "REPLAYGAIN_TRACK_GAIN".to_string(),
                MetadataValue::Text(gain),
            );
        }
        if let Some(peak) = self.track_peak_formatted() {
            metadata.insert(
                "REPLAYGAIN_TRACK_PEAK".to_string(),
                MetadataValue::Text(peak),
            );
        }
        if let Some(gain) = self.album_gain_formatted() {
            metadata.insert(
                "REPLAYGAIN_ALBUM_GAIN".to_string(),
                MetadataValue::Text(gain),
            );
        }
        if let Some(peak) = self.album_peak_formatted() {
            metadata.insert(
                "REPLAYGAIN_ALBUM_PEAK".to_string(),
                MetadataValue::Text(peak),
            );
        }

        // Only write reference loudness if non-default
        if (self.reference_loudness - DEFAULT_REFERENCE_LOUDNESS).abs() > f64::EPSILON {
            metadata.insert(
                "REPLAYGAIN_REFERENCE_LOUDNESS".to_string(),
                MetadataValue::Text(format_gain(self.reference_loudness)),
            );
        }
    }

    /// Validate the ReplayGain data for reasonable ranges.
    ///
    /// Returns a list of warnings (empty = valid).
    pub fn validate(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if let Some(gain) = self.track_gain_db {
            if gain < -51.0 || gain > 51.0 {
                warnings.push(format!(
                    "Track gain {gain:.2} dB is outside expected range [-51, +51]"
                ));
            }
        }

        if let Some(gain) = self.album_gain_db {
            if gain < -51.0 || gain > 51.0 {
                warnings.push(format!(
                    "Album gain {gain:.2} dB is outside expected range [-51, +51]"
                ));
            }
        }

        if let Some(peak) = self.track_peak {
            if peak < 0.0 {
                warnings.push(format!("Track peak {peak} is negative"));
            }
        }

        if let Some(peak) = self.album_peak {
            if peak < 0.0 {
                warnings.push(format!("Album peak {peak} is negative"));
            }
        }

        warnings
    }
}

/// Compute album-level ReplayGain from a slice of per-track ReplayGain values.
///
/// The album gain is the average of the individual track gains (energy-weighted).
/// The album peak is the maximum peak across all tracks.
///
/// # Errors
///
/// Returns an error if the input is empty or tracks lack gain data.
pub fn compute_album_gain(tracks: &[ReplayGain]) -> Result<ReplayGain, Error> {
    if tracks.is_empty() {
        return Err(Error::ParseError(
            "Cannot compute album gain from empty track list".to_string(),
        ));
    }

    // Collect track gains and compute the energy-weighted average
    let mut sum_linear_energy = 0.0f64;
    let mut count = 0u32;
    let mut max_peak = 0.0f64;

    for track in tracks {
        if let Some(gain_db) = track.track_gain_db {
            let linear = db_to_linear(gain_db);
            // Energy is proportional to amplitude squared
            sum_linear_energy += linear * linear;
            count += 1;
        }
        if let Some(peak) = track.track_peak {
            if peak > max_peak {
                max_peak = peak;
            }
        }
    }

    if count == 0 {
        return Err(Error::ParseError("No tracks have gain data".to_string()));
    }

    let avg_energy = sum_linear_energy / f64::from(count);
    let album_gain = linear_to_db(avg_energy.sqrt());

    let mut rg = ReplayGain::new().with_album_gain(album_gain);

    if max_peak > 0.0 {
        rg = rg.with_album_peak(max_peak);
    }

    Ok(rg)
}

// ---- Internal helpers ----

fn format_gain(gain_db: f64) -> String {
    if gain_db >= 0.0 {
        format!("+{gain_db:.2} dB")
    } else {
        format!("{gain_db:.2} dB")
    }
}

fn format_peak(peak: f64) -> String {
    format!("{peak:.6}")
}

fn db_to_linear(db: f64) -> f64 {
    10.0f64.powf(db / 20.0)
}

fn linear_to_db(linear: f64) -> f64 {
    if linear > 0.0 {
        20.0 * linear.log10()
    } else {
        f64::NEG_INFINITY
    }
}

/// Parse a gain string like "-6.50 dB" or "+3.21 dB" into dB value.
fn parse_gain(s: &str) -> Option<f64> {
    let trimmed = s.trim();
    let numeric_part = trimmed
        .strip_suffix(" dB")
        .or_else(|| trimmed.strip_suffix("dB"))
        .or_else(|| trimmed.strip_suffix(" LUFS"))
        .unwrap_or(trimmed)
        .trim();
    numeric_part.parse::<f64>().ok()
}

/// Parse a peak string like "0.950000" into a float.
fn parse_peak(s: &str) -> Option<f64> {
    s.trim().parse::<f64>().ok()
}

fn get_text<'a>(metadata: &'a Metadata, key: &str) -> Option<&'a str> {
    metadata.get(key).and_then(|v| v.as_text())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replaygain_new_default() {
        let rg = ReplayGain::new();
        assert_eq!(rg.track_gain_db(), None);
        assert_eq!(rg.track_peak(), None);
        assert_eq!(rg.album_gain_db(), None);
        assert_eq!(rg.album_peak(), None);
        assert!((rg.reference_loudness() - (-18.0)).abs() < f64::EPSILON);
        assert!(!rg.has_data());
    }

    #[test]
    fn test_replaygain_with_builders() {
        let rg = ReplayGain::new()
            .with_track_gain(-6.5)
            .with_track_peak(0.95)
            .with_album_gain(-7.2)
            .with_album_peak(0.98)
            .with_reference_loudness(-23.0);

        assert_eq!(rg.track_gain_db(), Some(-6.5));
        assert_eq!(rg.track_peak(), Some(0.95));
        assert_eq!(rg.album_gain_db(), Some(-7.2));
        assert_eq!(rg.album_peak(), Some(0.98));
        assert!((rg.reference_loudness() - (-23.0)).abs() < f64::EPSILON);
        assert!(rg.has_data());
    }

    #[test]
    fn test_replaygain_formatted() {
        let rg = ReplayGain::new()
            .with_track_gain(-6.5)
            .with_track_peak(0.95)
            .with_album_gain(3.2)
            .with_album_peak(0.12345);

        assert_eq!(rg.track_gain_formatted(), Some("-6.50 dB".to_string()));
        assert_eq!(rg.album_gain_formatted(), Some("+3.20 dB".to_string()));
        assert_eq!(rg.track_peak_formatted(), Some("0.950000".to_string()));
        assert_eq!(rg.album_peak_formatted(), Some("0.123450".to_string()));
    }

    #[test]
    fn test_replaygain_linear_conversion() {
        let rg = ReplayGain::new().with_track_gain(0.0);
        let linear = rg.track_gain_linear().expect("should have value");
        assert!((linear - 1.0).abs() < 1e-10); // 0 dB = 1.0 linear

        let rg2 = ReplayGain::new().with_track_gain(-20.0);
        let linear2 = rg2.track_gain_linear().expect("should have value");
        assert!((linear2 - 0.1).abs() < 1e-10); // -20 dB = 0.1 linear

        let rg3 = ReplayGain::new().with_album_gain(20.0);
        let linear3 = rg3.album_gain_linear().expect("should have value");
        assert!((linear3 - 10.0).abs() < 1e-10); // +20 dB = 10.0 linear
    }

    #[test]
    fn test_replaygain_clip_detection() {
        // Peak 0.9, gain +3 dB => linear ~1.413, 0.9 * 1.413 = 1.27 > 1.0 => clips
        let rg = ReplayGain::new().with_track_gain(3.0).with_track_peak(0.9);
        assert!(rg.track_would_clip());

        // Peak 0.5, gain +3 dB => 0.5 * 1.413 = 0.71 < 1.0 => no clip
        let rg2 = ReplayGain::new().with_track_gain(3.0).with_track_peak(0.5);
        assert!(!rg2.track_would_clip());

        // No data => no clip
        let rg3 = ReplayGain::new();
        assert!(!rg3.track_would_clip());
    }

    #[test]
    fn test_replaygain_album_clip_detection() {
        let rg = ReplayGain::new().with_album_gain(6.0).with_album_peak(0.8);
        // 0.8 * 10^(6/20) = 0.8 * ~1.995 = ~1.596 > 1.0
        assert!(rg.album_would_clip());
    }

    #[test]
    fn test_replaygain_safe_gain() {
        let rg = ReplayGain::new().with_track_gain(6.0).with_track_peak(0.9);

        let safe = rg.safe_track_gain_db().expect("should have value");
        // Max gain for peak 0.9: 20*log10(1/0.9) ≈ 0.915 dB
        assert!(safe < 6.0);
        assert!(safe < 1.0); // should be clamped significantly
    }

    #[test]
    fn test_replaygain_safe_gain_no_clip() {
        let rg = ReplayGain::new().with_track_gain(-6.0).with_track_peak(0.5);

        let safe = rg.safe_track_gain_db().expect("should have value");
        // -6.0 dB is well within safe range, should remain unchanged
        assert!((safe - (-6.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_replaygain_safe_album_gain() {
        let rg = ReplayGain::new()
            .with_album_gain(10.0)
            .with_album_peak(0.99);

        let safe = rg.safe_album_gain_db().expect("should have value");
        assert!(safe < 10.0);
    }

    #[test]
    fn test_replaygain_validate_valid() {
        let rg = ReplayGain::new()
            .with_track_gain(-6.5)
            .with_track_peak(0.95);
        assert!(rg.validate().is_empty());
    }

    #[test]
    fn test_replaygain_validate_out_of_range() {
        let rg = ReplayGain::new()
            .with_track_gain(-60.0)
            .with_track_peak(-0.1);
        let warnings = rg.validate();
        assert_eq!(warnings.len(), 2);
    }

    #[test]
    fn test_replaygain_metadata_round_trip() {
        let original = ReplayGain::new()
            .with_track_gain(-6.5)
            .with_track_peak(0.95)
            .with_album_gain(-7.2)
            .with_album_peak(0.98);

        let mut metadata = Metadata::new(MetadataFormat::VorbisComments);
        original.to_metadata(&mut metadata);

        let restored = ReplayGain::from_metadata(&metadata);

        assert!((restored.track_gain_db().expect("track gain") - (-6.5)).abs() < 0.01);
        assert!((restored.track_peak().expect("track peak") - 0.95).abs() < 0.000001);
        assert!((restored.album_gain_db().expect("album gain") - (-7.2)).abs() < 0.01);
        assert!((restored.album_peak().expect("album peak") - 0.98).abs() < 0.000001);
    }

    #[test]
    fn test_replaygain_metadata_with_custom_reference() {
        let rg = ReplayGain::new().with_reference_loudness(-23.0);
        let mut metadata = Metadata::new(MetadataFormat::VorbisComments);
        rg.to_metadata(&mut metadata);

        assert!(metadata.contains("REPLAYGAIN_REFERENCE_LOUDNESS"));
    }

    #[test]
    fn test_replaygain_metadata_default_reference_not_written() {
        let rg = ReplayGain::new().with_track_gain(-5.0);
        let mut metadata = Metadata::new(MetadataFormat::VorbisComments);
        rg.to_metadata(&mut metadata);

        assert!(!metadata.contains("REPLAYGAIN_REFERENCE_LOUDNESS"));
    }

    #[test]
    fn test_parse_gain_various_formats() {
        assert!((parse_gain("-6.50 dB").expect("parse") - (-6.5)).abs() < f64::EPSILON);
        assert!((parse_gain("+3.21 dB").expect("parse") - 3.21).abs() < f64::EPSILON);
        assert!((parse_gain("-6.50dB").expect("parse") - (-6.5)).abs() < f64::EPSILON);
        assert!((parse_gain(" -6.50 dB ").expect("parse") - (-6.5)).abs() < f64::EPSILON);
        assert!((parse_gain("-18.00 LUFS").expect("parse") - (-18.0)).abs() < f64::EPSILON);
        assert!((parse_gain("0.0").expect("parse") - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_peak() {
        assert!((parse_peak("0.950000").expect("parse") - 0.95).abs() < 0.000001);
        assert!((parse_peak("1.000000").expect("parse") - 1.0).abs() < 0.000001);
        assert!(parse_peak("not-a-number").is_none());
    }

    #[test]
    fn test_compute_album_gain_basic() {
        let tracks = vec![
            ReplayGain::new().with_track_gain(-6.0).with_track_peak(0.8),
            ReplayGain::new().with_track_gain(-6.0).with_track_peak(0.9),
            ReplayGain::new().with_track_gain(-6.0).with_track_peak(0.7),
        ];

        let album = compute_album_gain(&tracks).expect("should succeed");
        // All tracks have the same gain, so album gain should be very close
        assert!((album.album_gain_db().expect("gain") - (-6.0)).abs() < 0.01);
        // Album peak = max(0.8, 0.9, 0.7) = 0.9
        assert!((album.album_peak().expect("peak") - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_album_gain_mixed() {
        let tracks = vec![
            ReplayGain::new().with_track_gain(-3.0).with_track_peak(0.8),
            ReplayGain::new()
                .with_track_gain(-9.0)
                .with_track_peak(0.95),
        ];

        let album = compute_album_gain(&tracks).expect("should succeed");
        // The energy average of -3 dB and -9 dB
        // linear(-3) = 0.7079, linear(-9) = 0.3548
        // avg energy = (0.7079^2 + 0.3548^2) / 2 = (0.5011 + 0.1259) / 2 = 0.3135
        // album_gain = 20*log10(sqrt(0.3135)) = 20*log10(0.5599) ≈ -5.04 dB
        let album_gain = album.album_gain_db().expect("gain");
        assert!(album_gain < -4.0 && album_gain > -6.0);
        assert!((album.album_peak().expect("peak") - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_album_gain_empty() {
        let result = compute_album_gain(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_album_gain_no_data() {
        let tracks = vec![ReplayGain::new(), ReplayGain::new()];
        let result = compute_album_gain(&tracks);
        assert!(result.is_err());
    }

    #[test]
    fn test_db_to_linear_and_back() {
        let db = -6.0;
        let linear = db_to_linear(db);
        let back = linear_to_db(linear);
        assert!((db - back).abs() < 1e-10);
    }

    #[test]
    fn test_linear_to_db_zero() {
        let db = linear_to_db(0.0);
        assert!(db.is_infinite() && db < 0.0);
    }

    #[test]
    fn test_replaygain_from_metadata_empty() {
        let metadata = Metadata::new(MetadataFormat::VorbisComments);
        let rg = ReplayGain::from_metadata(&metadata);
        assert!(!rg.has_data());
    }

    #[test]
    fn test_replaygain_no_gain_no_clip() {
        // If no gain set, safe_track_gain_db returns None
        let rg = ReplayGain::new().with_track_peak(0.5);
        assert_eq!(rg.safe_track_gain_db(), None);
    }

    #[test]
    fn test_replaygain_gain_no_peak_safe() {
        // If no peak, safe gain = raw gain
        let rg = ReplayGain::new().with_track_gain(10.0);
        assert_eq!(rg.safe_track_gain_db(), Some(10.0));
    }
}
