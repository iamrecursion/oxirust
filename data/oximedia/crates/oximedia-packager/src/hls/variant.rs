//! HLS variant stream handling.

use crate::config::BitrateEntry;
use crate::error::{PackagerError, PackagerResult};
use crate::hls::playlist::{MediaPlaylistBuilder, VariantStream};
use crate::segment::SegmentInfo;
use std::time::Duration;
use tracing::debug;

/// Variant stream configuration.
#[derive(Debug, Clone)]
pub struct VariantConfig {
    /// Bitrate entry.
    pub bitrate: BitrateEntry,
    /// Variant name/ID.
    pub name: String,
    /// Media playlist filename.
    pub playlist_filename: String,
    /// Segment filename template.
    pub segment_template: String,
}

impl VariantConfig {
    /// Create a new variant configuration.
    #[must_use]
    pub fn new(bitrate: BitrateEntry, name: String) -> Self {
        let playlist_filename = format!("{name}.m3u8");
        let segment_template = format!("{name}_segment_{{index}}.m4s");

        Self {
            bitrate,
            name,
            playlist_filename,
            segment_template,
        }
    }

    /// Get segment filename for index.
    #[must_use]
    pub fn segment_filename(&self, index: u64) -> String {
        self.segment_template.replace("{index}", &index.to_string())
    }

    /// Get variant directory name.
    #[must_use]
    pub fn directory_name(&self) -> String {
        format!("{}_{}", self.name, self.bitrate.bitrate)
    }
}

/// Variant stream manager.
pub struct VariantManager {
    config: VariantConfig,
    segments: Vec<SegmentInfo>,
    current_sequence: u64,
    target_duration: Duration,
}

impl VariantManager {
    /// Create a new variant manager.
    #[must_use]
    pub fn new(config: VariantConfig, target_duration: Duration) -> Self {
        Self {
            config,
            segments: Vec::new(),
            current_sequence: 0,
            target_duration,
        }
    }

    /// Add a segment to this variant.
    pub fn add_segment(&mut self, segment: SegmentInfo) {
        debug!(
            "Adding segment {} to variant {}",
            segment.index, self.config.name
        );
        self.segments.push(segment);
    }

    /// Get the variant configuration.
    #[must_use]
    pub fn config(&self) -> &VariantConfig {
        &self.config
    }

    /// Generate media playlist for this variant.
    pub fn generate_playlist(&self) -> PackagerResult<String> {
        let mut builder = MediaPlaylistBuilder::new(self.target_duration)
            .with_media_sequence(self.current_sequence);

        for segment in &self.segments {
            builder.add_segment(segment.clone());
        }

        builder.build()
    }

    /// Generate media playlist with end marker (for VOD).
    pub fn generate_vod_playlist(&self) -> PackagerResult<String> {
        let mut builder = MediaPlaylistBuilder::new(self.target_duration)
            .with_media_sequence(0)
            .with_playlist_type("VOD".to_string())
            .with_end_list();

        for segment in &self.segments {
            builder.add_segment(segment.clone());
        }

        builder.build()
    }

    /// Update media sequence (for live streaming).
    pub fn update_sequence(&mut self, sequence: u64) {
        self.current_sequence = sequence;
    }

    /// Get current media sequence.
    #[must_use]
    pub fn sequence(&self) -> u64 {
        self.current_sequence
    }

    /// Get number of segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Remove old segments (for live streaming).
    pub fn remove_old_segments(&mut self, max_segments: usize) {
        if self.segments.len() > max_segments {
            let to_remove = self.segments.len() - max_segments;
            self.segments.drain(0..to_remove);
            self.current_sequence += to_remove as u64;

            debug!(
                "Removed {} old segments from variant {}, new sequence: {}",
                to_remove, self.config.name, self.current_sequence
            );
        }
    }

    /// Clear all segments.
    pub fn clear_segments(&mut self) {
        self.segments.clear();
        self.current_sequence = 0;
    }

    /// Get total duration of all segments.
    #[must_use]
    pub fn total_duration(&self) -> Duration {
        self.segments.iter().map(|s| s.duration).sum()
    }

    /// Get average segment size.
    #[must_use]
    pub fn average_segment_size(&self) -> u64 {
        if self.segments.is_empty() {
            return 0;
        }

        let total_size: u64 = self.segments.iter().map(|s| s.size).sum();
        total_size / self.segments.len() as u64
    }

    /// Convert to variant stream for master playlist.
    pub fn to_variant_stream(&self, base_url: Option<&str>) -> PackagerResult<VariantStream> {
        let uri = if let Some(base) = base_url {
            format!("{}/{}", base, self.config.playlist_filename)
        } else {
            self.config.playlist_filename.clone()
        };

        let codec_str = match self.config.bitrate.codec.as_str() {
            "av1" => "av01.0.04M.08".to_string(),
            "vp9" => "vp09.0.40.08".to_string(),
            "vp8" => "vp8".to_string(),
            _ => {
                return Err(PackagerError::unsupported_codec(format!(
                    "Unsupported codec: {}",
                    self.config.bitrate.codec
                )))
            }
        };

        let avg_bandwidth = if self.segments.is_empty() {
            None
        } else {
            let total_size: u64 = self.segments.iter().map(|s| s.size).sum();
            let total_duration = self.total_duration();
            if total_duration.is_zero() {
                None
            } else {
                Some((total_size * 8 / total_duration.as_secs()) as u32)
            }
        };

        let mut variant = VariantStream::new(self.config.bitrate.bitrate, codec_str, uri)
            .with_resolution(self.config.bitrate.width, self.config.bitrate.height);

        if let Some(fps) = self.config.bitrate.framerate {
            variant = variant.with_frame_rate(fps);
        }

        if let Some(avg) = avg_bandwidth {
            variant = variant.with_average_bandwidth(avg);
        }

        Ok(variant)
    }
}

/// Variant set manager for handling multiple variants.
pub struct VariantSet {
    variants: Vec<VariantManager>,
    target_duration: Duration,
}

impl VariantSet {
    /// Create a new variant set.
    #[must_use]
    pub fn new(target_duration: Duration) -> Self {
        Self {
            variants: Vec::new(),
            target_duration,
        }
    }

    /// Add a variant.
    pub fn add_variant(&mut self, config: VariantConfig) {
        let manager = VariantManager::new(config, self.target_duration);
        self.variants.push(manager);
    }

    /// Get variant by name.
    pub fn get_variant_mut(&mut self, name: &str) -> Option<&mut VariantManager> {
        self.variants.iter_mut().find(|v| v.config.name == name)
    }

    /// Get variant by name (immutable).
    #[must_use]
    pub fn get_variant(&self, name: &str) -> Option<&VariantManager> {
        self.variants.iter().find(|v| v.config.name == name)
    }

    /// Get all variants.
    #[must_use]
    pub fn variants(&self) -> &[VariantManager] {
        &self.variants
    }

    /// Get mutable variants.
    pub fn variants_mut(&mut self) -> &mut [VariantManager] {
        &mut self.variants
    }

    /// Generate master playlist.
    pub fn generate_master_playlist(&self, base_url: Option<&str>) -> PackagerResult<String> {
        use crate::hls::playlist::MasterPlaylistBuilder;

        let mut builder = MasterPlaylistBuilder::new();

        for variant in &self.variants {
            let variant_stream = variant.to_variant_stream(base_url)?;
            builder.add_variant(variant_stream);
        }

        builder.build()
    }

    /// Remove old segments from all variants.
    pub fn remove_old_segments(&mut self, max_segments: usize) {
        for variant in &mut self.variants {
            variant.remove_old_segments(max_segments);
        }
    }

    /// Get variant count.
    #[must_use]
    pub fn count(&self) -> usize {
        self.variants.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BitrateEntry;

    #[test]
    fn test_variant_config_creation() {
        let bitrate = BitrateEntry::new(1_000_000, 1280, 720, "av1");
        let config = VariantConfig::new(bitrate, "720p".to_string());

        assert_eq!(config.name, "720p");
        assert_eq!(config.playlist_filename, "720p.m3u8");
        assert!(config.segment_template.contains("720p"));
    }

    #[test]
    fn test_segment_filename_generation() {
        let bitrate = BitrateEntry::new(1_000_000, 1280, 720, "av1");
        let config = VariantConfig::new(bitrate, "720p".to_string());

        let filename = config.segment_filename(5);
        assert!(filename.contains("5"));
        assert!(filename.contains("720p"));
    }

    #[test]
    fn test_variant_manager_sequence() {
        let bitrate = BitrateEntry::new(1_000_000, 1280, 720, "av1");
        let config = VariantConfig::new(bitrate, "720p".to_string());
        let mut manager = VariantManager::new(config, Duration::from_secs(6));

        assert_eq!(manager.sequence(), 0);

        manager.update_sequence(10);
        assert_eq!(manager.sequence(), 10);
    }

    #[test]
    fn test_variant_set_management() {
        let mut set = VariantSet::new(Duration::from_secs(6));

        let bitrate = BitrateEntry::new(1_000_000, 1280, 720, "av1");
        let config = VariantConfig::new(bitrate, "720p".to_string());

        set.add_variant(config);

        assert_eq!(set.count(), 1);
        assert!(set.get_variant("720p").is_some());
    }
}
