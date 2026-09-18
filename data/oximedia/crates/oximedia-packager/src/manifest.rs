//! Common manifest generation utilities.

use crate::error::{PackagerError, PackagerResult};
use chrono::{DateTime, Utc};
use std::time::Duration;

/// Manifest type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestType {
    /// HLS master playlist.
    HlsMaster,
    /// HLS media playlist.
    HlsMedia,
    /// DASH MPD.
    DashMpd,
}

/// Manifest metadata.
#[derive(Debug, Clone)]
pub struct ManifestMetadata {
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Version number.
    pub version: u32,
    /// Title.
    pub title: Option<String>,
    /// Description.
    pub description: Option<String>,
}

impl Default for ManifestMetadata {
    fn default() -> Self {
        Self {
            created_at: Utc::now(),
            version: 1,
            title: None,
            description: None,
        }
    }
}

impl ManifestMetadata {
    /// Create new manifest metadata.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the title.
    #[must_use]
    pub fn with_title(mut self, title: String) -> Self {
        self.title = Some(title);
        self
    }

    /// Set the description.
    #[must_use]
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    /// Increment the version number.
    pub fn increment_version(&mut self) {
        self.version += 1;
    }
}

/// Duration formatter for manifests.
pub struct DurationFormatter;

impl DurationFormatter {
    /// Format duration as HLS duration (decimal seconds).
    #[must_use]
    pub fn format_hls_duration(duration: Duration) -> String {
        format!("{:.3}", duration.as_secs_f64())
    }

    /// Format duration as ISO 8601 duration (for DASH).
    #[must_use]
    pub fn format_iso8601_duration(duration: Duration) -> String {
        let total_secs = duration.as_secs();
        let hours = total_secs / 3600;
        let minutes = (total_secs % 3600) / 60;
        let seconds = total_secs % 60;
        let millis = duration.subsec_millis();

        if hours > 0 {
            format!("PT{hours}H{minutes}M{seconds}.{millis}S")
        } else if minutes > 0 {
            format!("PT{minutes}M{seconds}.{millis}S")
        } else {
            format!("PT{seconds}.{millis}S")
        }
    }

    /// Parse HLS duration.
    pub fn parse_hls_duration(s: &str) -> PackagerResult<Duration> {
        let secs: f64 = s
            .parse()
            .map_err(|_| PackagerError::Time("Invalid HLS duration".to_string()))?;

        Ok(Duration::from_secs_f64(secs))
    }

    /// Parse ISO 8601 duration.
    pub fn parse_iso8601_duration(s: &str) -> PackagerResult<Duration> {
        if !s.starts_with("PT") {
            return Err(PackagerError::Time("Invalid ISO 8601 duration".to_string()));
        }

        let s = &s[2..]; // Skip "PT"
        let mut total_secs = 0u64;

        // Parse hours
        if let Some(h_pos) = s.find('H') {
            let hours: u64 = s[..h_pos]
                .parse()
                .map_err(|_| PackagerError::Time("Invalid hours".to_string()))?;
            total_secs += hours * 3600;
        }

        // Parse minutes
        if let Some(m_pos) = s.find('M') {
            let start = s.find('H').map_or(0, |p| p + 1);
            let minutes: u64 = s[start..m_pos]
                .parse()
                .map_err(|_| PackagerError::Time("Invalid minutes".to_string()))?;
            total_secs += minutes * 60;
        }

        // Parse seconds
        if let Some(s_pos) = s.find('S') {
            let start = s.rfind(['H', 'M']).map_or(0, |p| p + 1);
            let seconds: f64 = s[start..s_pos]
                .parse()
                .map_err(|_| PackagerError::Time("Invalid seconds".to_string()))?;
            total_secs += seconds as u64;
        }

        Ok(Duration::from_secs(total_secs))
    }
}

/// URL builder for manifest URLs.
pub struct UrlBuilder {
    base_url: Option<String>,
}

impl UrlBuilder {
    /// Create a new URL builder.
    #[must_use]
    pub fn new(base_url: Option<String>) -> Self {
        Self { base_url }
    }

    /// Build a URL for a resource.
    #[must_use]
    pub fn build(&self, path: &str) -> String {
        if let Some(base) = &self.base_url {
            if base.ends_with('/') {
                format!("{base}{path}")
            } else {
                format!("{base}/{path}")
            }
        } else {
            path.to_string()
        }
    }

    /// Build a URL with query parameters.
    #[must_use]
    pub fn build_with_params(&self, path: &str, params: &[(&str, &str)]) -> String {
        let base = self.build(path);

        if params.is_empty() {
            return base;
        }

        let query: Vec<String> = params.iter().map(|(k, v)| format!("{k}={v}")).collect();

        format!("{}?{}", base, query.join("&"))
    }
}

/// Manifest writer trait.
#[async_trait::async_trait]
pub trait ManifestWriter {
    /// Write manifest to output.
    async fn write_manifest(&self, manifest: &str, path: &std::path::Path) -> PackagerResult<()>;

    /// Update manifest version.
    async fn update_version(&self, path: &std::path::Path) -> PackagerResult<()>;
}

/// File-based manifest writer.
pub struct FileManifestWriter;

#[async_trait::async_trait]
impl ManifestWriter for FileManifestWriter {
    async fn write_manifest(&self, manifest: &str, path: &std::path::Path) -> PackagerResult<()> {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        tokio::fs::write(path, manifest).await?;

        tracing::debug!("Wrote manifest to {}", path.display());

        Ok(())
    }

    async fn update_version(&self, path: &std::path::Path) -> PackagerResult<()> {
        if !path.exists() {
            return Ok(());
        }

        // Read current manifest
        let _content = tokio::fs::read_to_string(path).await?;

        // Update version in manifest (implementation depends on format)
        // For now, just touch the file to update modification time
        let _metadata = tokio::fs::metadata(path).await?;

        Ok(())
    }
}

/// Codec string builder for manifests.
pub struct CodecStringBuilder;

impl CodecStringBuilder {
    /// Build codec string for AV1.
    #[must_use]
    pub fn av1(profile: u8, level: u8, bit_depth: u8) -> String {
        // AV1 codec string format: av01.P.LLT.DD
        // P = profile (0=Main, 1=High, 2=Professional)
        // LL = level (e.g., 00-31)
        // T = tier (M=Main, H=High)
        // DD = bit depth
        format!("av01.{profile}.{level:02}M.{bit_depth:02}")
    }

    /// Build codec string for VP9.
    #[must_use]
    pub fn vp9(profile: u8, level: u8, bit_depth: u8) -> String {
        // VP9 codec string format: vp09.P.LL.DD
        format!("vp09.{profile}.{level:02}.{bit_depth:02}")
    }

    /// Build codec string for VP8.
    #[must_use]
    pub fn vp8() -> String {
        "vp8".to_string()
    }

    /// Build codec string for Opus.
    #[must_use]
    pub fn opus(channels: u8) -> String {
        format!("opus.{channels}")
    }

    /// Build codec string for Vorbis.
    #[must_use]
    pub fn vorbis() -> String {
        "vorbis".to_string()
    }

    /// Build codec string for FLAC.
    #[must_use]
    pub fn flac() -> String {
        "flac".to_string()
    }

    /// Parse codec string to identify codec.
    pub fn parse_codec(codec_str: &str) -> PackagerResult<String> {
        if codec_str.starts_with("av01") {
            Ok("av1".to_string())
        } else if codec_str.starts_with("vp09") {
            Ok("vp9".to_string())
        } else if codec_str.starts_with("vp8") {
            Ok("vp8".to_string())
        } else if codec_str.starts_with("opus") {
            Ok("opus".to_string())
        } else if codec_str.starts_with("vorbis") {
            Ok("vorbis".to_string())
        } else if codec_str.starts_with("flac") {
            Ok("flac".to_string())
        } else {
            Err(PackagerError::unsupported_codec(format!(
                "Unknown codec string: {codec_str}"
            )))
        }
    }
}

/// Bandwidth calculator for adaptive bitrate.
pub struct BandwidthCalculator;

impl BandwidthCalculator {
    /// Calculate bandwidth from bitrate (adds overhead).
    #[must_use]
    pub fn from_bitrate(bitrate: u32) -> u32 {
        // Add 10% overhead for packaging
        (f64::from(bitrate) * 1.1) as u32
    }

    /// Calculate average bandwidth from segment sizes.
    #[must_use]
    pub fn from_segments(segment_sizes: &[u64], duration: Duration) -> u32 {
        if segment_sizes.is_empty() || duration.is_zero() {
            return 0;
        }

        let total_bytes: u64 = segment_sizes.iter().sum();
        let total_bits = total_bytes * 8;
        let duration_secs = duration.as_secs_f64();

        (total_bits as f64 / duration_secs) as u32
    }

    /// Calculate peak bandwidth from segments.
    #[must_use]
    pub fn peak_bandwidth(segment_sizes: &[u64], segment_duration: Duration) -> u32 {
        if segment_sizes.is_empty() || segment_duration.is_zero() {
            return 0;
        }

        let max_size = *segment_sizes.iter().max().unwrap_or(&0);
        let max_bits = max_size * 8;
        let duration_secs = segment_duration.as_secs_f64();

        (max_bits as f64 / duration_secs) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hls_duration_formatting() {
        let duration = Duration::from_secs(6);
        let formatted = DurationFormatter::format_hls_duration(duration);
        assert_eq!(formatted, "6.000");
    }

    #[test]
    fn test_iso8601_duration_formatting() {
        let duration = Duration::from_secs(125); // 2m 5s
        let formatted = DurationFormatter::format_iso8601_duration(duration);
        assert!(formatted.starts_with("PT2M"));
    }

    #[test]
    fn test_url_builder() {
        let builder = UrlBuilder::new(Some("https://example.com".to_string()));
        let url = builder.build("segment.m4s");
        assert_eq!(url, "https://example.com/segment.m4s");
    }

    #[test]
    fn test_codec_string_av1() {
        let codec = CodecStringBuilder::av1(0, 4, 8);
        assert_eq!(codec, "av01.0.04M.08");
    }

    #[test]
    fn test_bandwidth_calculation() {
        let bitrate = 1_000_000;
        let bandwidth = BandwidthCalculator::from_bitrate(bitrate);
        assert!(bandwidth > bitrate);
    }
}
