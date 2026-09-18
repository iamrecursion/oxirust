//! DASH (Dynamic Adaptive Streaming over HTTP) support.

use crate::{
    config::Config,
    error::{ServerError, ServerResult},
    models::media::Media,
};
use std::path::PathBuf;

/// DASH manifest and segment generator.
pub struct DashGenerator {
    config: Config,
}

impl DashGenerator {
    /// Creates a new DASH generator.
    #[must_use]
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// Generates a DASH MPD (Media Presentation Description) manifest.
    ///
    /// # Errors
    ///
    /// Returns an error if the media file is invalid.
    #[allow(clippy::too_many_lines)]
    pub fn generate_manifest(&self, media: &Media) -> ServerResult<String> {
        if !media.is_video() {
            return Err(ServerError::BadRequest(
                "DASH only supported for video files".to_string(),
            ));
        }

        let duration = media
            .duration
            .ok_or_else(|| ServerError::BadRequest("Media duration not available".to_string()))?;

        let width = media.width.unwrap_or(1920);
        let height = media.height.unwrap_or(1080);
        let bitrate = media.bitrate.unwrap_or(5_000_000);
        let framerate = media.framerate.unwrap_or(30.0);

        let segment_duration = self.config.dash_segment_duration;

        // Generate variants
        let variants = vec![
            (width, height, bitrate, "high"),
            (width * 3 / 4, height * 3 / 4, bitrate * 3 / 4, "medium"),
            (width / 2, height / 2, bitrate / 2, "low"),
        ];

        let mut mpd = String::from(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" type="static" mediaPresentationDuration=""#,
        );

        mpd.push_str(&format!(
            r#"PT{:.3}S" minBufferTime="PT{}S" profiles="urn:mpeg:dash:profile:isoff-live:2011">
  <Period>
    <AdaptationSet mimeType="video/mp4" codecs="{}" frameRate="{:.3}" segmentAlignment="true" startWithSAP="1">
"#,
            duration,
            segment_duration,
            media.codec_video.as_deref().unwrap_or("av01.0.04M.08"),
            framerate
        ));

        for (w, h, bps, name) in &variants {
            mpd.push_str(&format!(
                r#"      <Representation id="{}" bandwidth="{}" width="{}" height="{}">
        <SegmentTemplate timescale="1000" initialization="{}/init.mp4" media="{}/segment$Number$.m4s" startNumber="0">
          <SegmentTimeline>
"#,
                name, bps, w, h, name, name
            ));

            let num_segments = (duration / (segment_duration as f64)).ceil() as usize;
            for i in 0..num_segments {
                let seg_dur = if i == num_segments - 1 {
                    ((duration - (i as f64 * (segment_duration as f64))) * 1000.0) as u64
                } else {
                    segment_duration * 1000
                };

                mpd.push_str(&format!(
                    r#"            <S d="{}"/>
"#,
                    seg_dur
                ));
            }

            mpd.push_str(
                r#"          </SegmentTimeline>
        </SegmentTemplate>
      </Representation>
"#,
            );
        }

        mpd.push_str(
            r#"    </AdaptationSet>
  </Period>
</MPD>
"#,
        );

        Ok(mpd)
    }

    /// Gets the path to a DASH initialization segment.
    #[must_use]
    pub fn get_init_path(&self, media_id: &str, variant: &str) -> PathBuf {
        self.config
            .media_dir
            .join("dash")
            .join(media_id)
            .join(variant)
            .join("init.mp4")
    }

    /// Gets the path to a DASH media segment.
    #[must_use]
    pub fn get_segment_path(&self, media_id: &str, variant: &str, segment: usize) -> PathBuf {
        self.config
            .media_dir
            .join("dash")
            .join(media_id)
            .join(variant)
            .join(format!("segment{}.m4s", segment))
    }

    /// Checks if DASH segments exist for a media file.
    #[must_use]
    pub fn segments_exist(&self, media_id: &str, variant: &str) -> bool {
        let dash_dir = self
            .config
            .media_dir
            .join("dash")
            .join(media_id)
            .join(variant);
        dash_dir.exists() && dash_dir.is_dir()
    }
}
