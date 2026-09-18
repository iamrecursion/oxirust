//! HLS (HTTP Live Streaming) support.

use crate::{
    config::Config,
    error::{ServerError, ServerResult},
    models::media::Media,
};
use std::path::PathBuf;

/// HLS manifest and segment generator.
pub struct HlsGenerator {
    config: Config,
}

impl HlsGenerator {
    /// Creates a new HLS generator.
    #[must_use]
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// Generates an HLS master playlist.
    ///
    /// # Errors
    ///
    /// Returns an error if the media file is invalid.
    pub fn generate_master_playlist(&self, media: &Media) -> ServerResult<String> {
        if !media.is_video() {
            return Err(ServerError::BadRequest(
                "HLS only supported for video files".to_string(),
            ));
        }

        let width = media.width.unwrap_or(1920);
        let height = media.height.unwrap_or(1080);
        let bitrate = media.bitrate.unwrap_or(5_000_000);

        // Generate variants at different bitrates
        let variants = vec![
            (width, height, bitrate, "high"),
            (width * 3 / 4, height * 3 / 4, bitrate * 3 / 4, "medium"),
            (width / 2, height / 2, bitrate / 2, "low"),
        ];

        let mut playlist = String::from("#EXTM3U\n#EXT-X-VERSION:3\n\n");

        for (w, h, bps, name) in variants {
            playlist.push_str(&format!(
                "#EXT-X-STREAM-INF:BANDWIDTH={},RESOLUTION={}x{}\n",
                bps, w, h
            ));
            playlist.push_str(&format!("{}/playlist.m3u8\n\n", name));
        }

        Ok(playlist)
    }

    /// Generates an HLS media playlist for a variant.
    ///
    /// # Errors
    ///
    /// Returns an error if the media file is invalid.
    pub fn generate_media_playlist(&self, media: &Media, _variant: &str) -> ServerResult<String> {
        let duration = media
            .duration
            .ok_or_else(|| ServerError::BadRequest("Media duration not available".to_string()))?;

        let segment_duration = self.config.hls_segment_duration as f64;
        let num_segments = (duration / segment_duration).ceil() as usize;

        let mut playlist = String::from("#EXTM3U\n");
        playlist.push_str("#EXT-X-VERSION:3\n");
        playlist.push_str(&format!(
            "#EXT-X-TARGETDURATION:{}\n",
            segment_duration.ceil()
        ));
        playlist.push_str("#EXT-X-MEDIA-SEQUENCE:0\n\n");

        for i in 0..num_segments {
            let seg_dur = if i == num_segments - 1 {
                duration - (i as f64 * segment_duration)
            } else {
                segment_duration
            };

            playlist.push_str(&format!("#EXTINF:{:.3},\n", seg_dur));
            playlist.push_str(&format!("segment{}.ts\n", i));
        }

        playlist.push_str("#EXT-X-ENDLIST\n");

        Ok(playlist)
    }

    /// Gets the path to a segment file.
    #[must_use]
    pub fn get_segment_path(&self, media_id: &str, variant: &str, segment: usize) -> PathBuf {
        self.config
            .media_dir
            .join("hls")
            .join(media_id)
            .join(variant)
            .join(format!("segment{}.ts", segment))
    }

    /// Checks if HLS segments exist for a media file.
    #[must_use]
    pub fn segments_exist(&self, media_id: &str, variant: &str) -> bool {
        let hls_dir = self
            .config
            .media_dir
            .join("hls")
            .join(media_id)
            .join(variant);
        hls_dir.exists() && hls_dir.is_dir()
    }
}
