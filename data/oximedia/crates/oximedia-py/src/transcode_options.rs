//! Python-bindable transcode options.
//!
//! Provides builder-style configuration types for video/audio transcoding
//! that can be easily exposed to Python through PyO3.

#![allow(dead_code)]

/// Video-specific encoding options.
#[derive(Clone, Debug)]
pub struct PyVideoOptions {
    /// Target output width in pixels, or `None` to keep source width.
    pub width: Option<u32>,
    /// Target output height in pixels, or `None` to keep source height.
    pub height: Option<u32>,
    /// Target frames-per-second, or `None` to keep source frame rate.
    pub fps: Option<f32>,
    /// Constant Rate Factor for quality-based encoding (codec-dependent).
    pub crf: Option<u8>,
    /// Target bitrate in kbit/s for bitrate-based encoding.
    pub bitrate_kbps: Option<u32>,
}

impl PyVideoOptions {
    /// Creates a passthrough (all fields `None`) video options set.
    #[must_use]
    pub fn default() -> Self {
        Self {
            width: None,
            height: None,
            fps: None,
            crf: None,
            bitrate_kbps: None,
        }
    }

    /// Returns `true` if no encoding parameters are set (i.e. copy/passthrough).
    #[must_use]
    pub fn is_passthrough(&self) -> bool {
        self.width.is_none()
            && self.height.is_none()
            && self.fps.is_none()
            && self.crf.is_none()
            && self.bitrate_kbps.is_none()
    }
}

/// Audio-specific encoding options.
#[derive(Clone, Debug)]
pub struct PyAudioOptions {
    /// Target sample rate in Hz, or `None` to keep the source rate.
    pub sample_rate: Option<u32>,
    /// Target channel count, or `None` to keep the source layout.
    pub channels: Option<u8>,
    /// Target bitrate in kbit/s, or `None` to use the codec default.
    pub bitrate_kbps: Option<u32>,
}

impl PyAudioOptions {
    /// Creates a passthrough (all fields `None`) audio options set.
    #[must_use]
    pub fn default() -> Self {
        Self {
            sample_rate: None,
            channels: None,
            bitrate_kbps: None,
        }
    }
}

/// Full transcode job configuration.
#[derive(Clone, Debug)]
pub struct PyTranscodeOptions {
    /// Input file path.
    pub input: String,
    /// Output file path.
    pub output: String,
    /// Video encoding options.
    pub video: PyVideoOptions,
    /// Audio encoding options.
    pub audio: PyAudioOptions,
    /// Whether to overwrite the output file if it already exists.
    pub overwrite: bool,
}

impl PyTranscodeOptions {
    /// Creates a new transcode job with passthrough video/audio options.
    #[must_use]
    pub fn new(input: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            video: PyVideoOptions::default(),
            audio: PyAudioOptions::default(),
            overwrite: false,
        }
    }

    /// Sets the video CRF and returns `self` for chaining.
    #[must_use]
    pub fn with_video_crf(mut self, crf: u8) -> Self {
        self.video.crf = Some(crf);
        self
    }

    /// Sets the audio bitrate in kbit/s and returns `self` for chaining.
    #[must_use]
    pub fn with_audio_bitrate(mut self, kbps: u32) -> Self {
        self.audio.bitrate_kbps = Some(kbps);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── PyVideoOptions ──────────────────────────────────────────────────────

    #[test]
    fn test_video_default_is_passthrough() {
        assert!(PyVideoOptions::default().is_passthrough());
    }

    #[test]
    fn test_video_not_passthrough_when_crf_set() {
        let mut v = PyVideoOptions::default();
        v.crf = Some(23);
        assert!(!v.is_passthrough());
    }

    #[test]
    fn test_video_not_passthrough_when_width_set() {
        let mut v = PyVideoOptions::default();
        v.width = Some(1280);
        assert!(!v.is_passthrough());
    }

    #[test]
    fn test_video_not_passthrough_when_fps_set() {
        let mut v = PyVideoOptions::default();
        v.fps = Some(30.0);
        assert!(!v.is_passthrough());
    }

    #[test]
    fn test_video_not_passthrough_when_bitrate_set() {
        let mut v = PyVideoOptions::default();
        v.bitrate_kbps = Some(4_000);
        assert!(!v.is_passthrough());
    }

    // ── PyAudioOptions ──────────────────────────────────────────────────────

    #[test]
    fn test_audio_default_all_none() {
        let a = PyAudioOptions::default();
        assert!(a.sample_rate.is_none());
        assert!(a.channels.is_none());
        assert!(a.bitrate_kbps.is_none());
    }

    #[test]
    fn test_audio_store_sample_rate() {
        let mut a = PyAudioOptions::default();
        a.sample_rate = Some(48_000);
        assert_eq!(a.sample_rate, Some(48_000));
    }

    // ── PyTranscodeOptions ──────────────────────────────────────────────────

    #[test]
    fn test_new_stores_paths() {
        let opts = PyTranscodeOptions::new("in.mkv", "out.mkv");
        assert_eq!(opts.input, "in.mkv");
        assert_eq!(opts.output, "out.mkv");
    }

    #[test]
    fn test_new_default_no_overwrite() {
        let opts = PyTranscodeOptions::new("a", "b");
        assert!(!opts.overwrite);
    }

    #[test]
    fn test_new_video_passthrough_by_default() {
        let opts = PyTranscodeOptions::new("a", "b");
        assert!(opts.video.is_passthrough());
    }

    #[test]
    fn test_with_video_crf_sets_crf() {
        let opts = PyTranscodeOptions::new("a", "b").with_video_crf(28);
        assert_eq!(opts.video.crf, Some(28));
    }

    #[test]
    fn test_with_video_crf_still_passthrough_for_other_fields() {
        let opts = PyTranscodeOptions::new("a", "b").with_video_crf(28);
        // only crf is set, so is_passthrough() should return false
        assert!(!opts.video.is_passthrough());
    }

    #[test]
    fn test_with_audio_bitrate_sets_bitrate() {
        let opts = PyTranscodeOptions::new("a", "b").with_audio_bitrate(192);
        assert_eq!(opts.audio.bitrate_kbps, Some(192));
    }

    #[test]
    fn test_chaining_both_options() {
        let opts = PyTranscodeOptions::new("in.mp4", "out.webm")
            .with_video_crf(30)
            .with_audio_bitrate(128);
        assert_eq!(opts.video.crf, Some(30));
        assert_eq!(opts.audio.bitrate_kbps, Some(128));
    }

    #[test]
    fn test_input_string_owned() {
        let input = String::from("input_file.mkv");
        let opts = PyTranscodeOptions::new(input, "output.mkv");
        assert_eq!(opts.input, "input_file.mkv");
    }
}
