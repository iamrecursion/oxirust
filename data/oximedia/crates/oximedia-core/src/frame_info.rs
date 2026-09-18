//! Frame metadata and descriptor types.
//!
//! Provides lightweight descriptors for video and audio frames,
//! including resolution, frame type (I/P/B), and presentation timing.

#![allow(dead_code)]

/// Type of a compressed video frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameType {
    /// Intra-coded frame (key frame / IDR).
    Intra,
    /// Predicted frame (references the previous frame).
    Predicted,
    /// Bi-directionally predicted frame.
    BiPredicted,
    /// Unknown / not yet determined.
    Unknown,
}

impl FrameType {
    /// Returns `true` if this is a key frame (intra-coded).
    #[must_use]
    pub fn is_key_frame(self) -> bool {
        self == Self::Intra
    }

    /// Returns a single-character abbreviation (`I`, `P`, `B`, `?`).
    #[must_use]
    pub fn abbreviation(self) -> char {
        match self {
            Self::Intra => 'I',
            Self::Predicted => 'P',
            Self::BiPredicted => 'B',
            Self::Unknown => '?',
        }
    }
}

/// Colour space primaries of a video frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorPrimaries {
    /// BT.709 – standard HDTV.
    Bt709,
    /// BT.2020 – UHD / HDR.
    Bt2020,
    /// BT.601 – SDTV.
    Bt601,
    /// sRGB / IEC 61966-2-1.
    Srgb,
    /// Unknown.
    Unknown,
}

impl ColorPrimaries {
    /// Returns the ITU-T H.273 code point for these primaries.
    #[must_use]
    pub fn code_point(self) -> u8 {
        match self {
            Self::Bt709 => 1,
            Self::Bt601 => 5,
            Self::Srgb => 13,
            Self::Bt2020 => 9,
            Self::Unknown => 0,
        }
    }

    /// Returns `true` if these primaries are suitable for HDR content.
    #[must_use]
    pub fn is_hdr_capable(self) -> bool {
        self == Self::Bt2020
    }
}

/// Metadata describing a single video frame.
#[derive(Debug, Clone)]
pub struct VideoFrameInfo {
    /// Presentation timestamp in the stream's time base.
    pub pts: i64,
    /// Decode timestamp.
    pub dts: i64,
    /// Duration of the frame in the stream's time base.
    pub duration: i64,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Compressed frame type.
    pub frame_type: FrameType,
    /// Colour primaries.
    pub color_primaries: ColorPrimaries,
    /// Compressed size in bytes (0 if unknown / uncompressed).
    pub encoded_size: usize,
}

impl VideoFrameInfo {
    /// Creates a new `VideoFrameInfo`.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pts: i64,
        dts: i64,
        duration: i64,
        width: u32,
        height: u32,
        frame_type: FrameType,
        color_primaries: ColorPrimaries,
        encoded_size: usize,
    ) -> Self {
        Self {
            pts,
            dts,
            duration,
            width,
            height,
            frame_type,
            color_primaries,
            encoded_size,
        }
    }

    /// Returns the total pixel count for this frame.
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Returns `true` if this is a key frame.
    #[must_use]
    pub fn is_key_frame(&self) -> bool {
        self.frame_type.is_key_frame()
    }

    /// Returns the aspect ratio as `width / height`.
    ///
    /// Returns `0.0` if the height is zero.
    #[must_use]
    pub fn aspect_ratio(&self) -> f64 {
        if self.height == 0 {
            return 0.0;
        }
        f64::from(self.width) / f64::from(self.height)
    }

    /// Returns bits per pixel if `encoded_size > 0`, otherwise `0.0`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn bits_per_pixel(&self) -> f64 {
        let pixels = self.pixel_count();
        if pixels == 0 || self.encoded_size == 0 {
            return 0.0;
        }
        (self.encoded_size as f64 * 8.0) / pixels as f64
    }
}

/// Metadata describing a single audio frame (packet).
#[derive(Debug, Clone)]
pub struct AudioFrameInfo {
    /// Presentation timestamp.
    pub pts: i64,
    /// Duration in time-base units.
    pub duration: i64,
    /// Number of audio samples in this frame.
    pub nb_samples: u32,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u8,
    /// Encoded size in bytes.
    pub encoded_size: usize,
}

impl AudioFrameInfo {
    /// Creates a new `AudioFrameInfo`.
    #[must_use]
    pub fn new(
        pts: i64,
        duration: i64,
        nb_samples: u32,
        sample_rate: u32,
        channels: u8,
        encoded_size: usize,
    ) -> Self {
        Self {
            pts,
            duration,
            nb_samples,
            sample_rate,
            channels,
            encoded_size,
        }
    }

    /// Duration of this frame in seconds.
    ///
    /// Returns `0.0` if `sample_rate` is zero.
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        f64::from(self.nb_samples) / f64::from(self.sample_rate)
    }

    /// Total samples across all channels.
    #[must_use]
    pub fn total_samples(&self) -> u64 {
        u64::from(self.nb_samples) * u64::from(self.channels)
    }

    /// Bytes per second assuming this frame is representative of the stream.
    ///
    /// Returns `0.0` if `nb_samples` is zero.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn bytes_per_second(&self) -> f64 {
        if self.nb_samples == 0 {
            return 0.0;
        }
        let dur = self.duration_secs();
        if dur == 0.0 {
            return 0.0;
        }
        self.encoded_size as f64 / dur
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. FrameType::is_key_frame
    #[test]
    fn test_intra_is_key_frame() {
        assert!(FrameType::Intra.is_key_frame());
        assert!(!FrameType::Predicted.is_key_frame());
        assert!(!FrameType::BiPredicted.is_key_frame());
    }

    // 2. FrameType::abbreviation
    #[test]
    fn test_frame_type_abbreviation() {
        assert_eq!(FrameType::Intra.abbreviation(), 'I');
        assert_eq!(FrameType::Predicted.abbreviation(), 'P');
        assert_eq!(FrameType::BiPredicted.abbreviation(), 'B');
        assert_eq!(FrameType::Unknown.abbreviation(), '?');
    }

    // 3. ColorPrimaries::code_point
    #[test]
    fn test_color_primaries_code_points() {
        assert_eq!(ColorPrimaries::Bt709.code_point(), 1);
        assert_eq!(ColorPrimaries::Bt2020.code_point(), 9);
        assert_eq!(ColorPrimaries::Srgb.code_point(), 13);
        assert_eq!(ColorPrimaries::Unknown.code_point(), 0);
    }

    // 4. ColorPrimaries::is_hdr_capable
    #[test]
    fn test_hdr_capable() {
        assert!(ColorPrimaries::Bt2020.is_hdr_capable());
        assert!(!ColorPrimaries::Bt709.is_hdr_capable());
    }

    // 5. VideoFrameInfo::pixel_count
    #[test]
    fn test_pixel_count() {
        let f = VideoFrameInfo::new(
            0,
            0,
            1001,
            1920,
            1080,
            FrameType::Intra,
            ColorPrimaries::Bt709,
            50_000,
        );
        assert_eq!(f.pixel_count(), 1920 * 1080);
    }

    // 6. VideoFrameInfo::is_key_frame
    #[test]
    fn test_video_is_key_frame() {
        let f = VideoFrameInfo::new(
            0,
            0,
            1001,
            1280,
            720,
            FrameType::Intra,
            ColorPrimaries::Bt709,
            0,
        );
        assert!(f.is_key_frame());
    }

    // 7. VideoFrameInfo::aspect_ratio
    #[test]
    fn test_aspect_ratio_16_9() {
        let f = VideoFrameInfo::new(
            0,
            0,
            1,
            1920,
            1080,
            FrameType::Intra,
            ColorPrimaries::Bt709,
            0,
        );
        let ar = f.aspect_ratio();
        assert!((ar - 16.0 / 9.0).abs() < 1e-9);
    }

    #[test]
    fn test_aspect_ratio_zero_height() {
        let f = VideoFrameInfo::new(
            0,
            0,
            1,
            1920,
            0,
            FrameType::Unknown,
            ColorPrimaries::Unknown,
            0,
        );
        assert_eq!(f.aspect_ratio(), 0.0);
    }

    // 8. VideoFrameInfo::bits_per_pixel
    #[test]
    fn test_bits_per_pixel() {
        // 1920*1080 = 2_073_600 pixels; encoded_size = 259_200 bytes → 1.0 bpp
        let f = VideoFrameInfo::new(
            0,
            0,
            1,
            1920,
            1080,
            FrameType::Predicted,
            ColorPrimaries::Bt709,
            259_200,
        );
        assert!((f.bits_per_pixel() - 1.0).abs() < 1e-6);
    }

    // 9. AudioFrameInfo::duration_secs
    #[test]
    fn test_audio_duration_secs() {
        let f = AudioFrameInfo::new(0, 1024, 1024, 48_000, 2, 4096);
        let d = f.duration_secs();
        assert!((d - 1024.0 / 48_000.0).abs() < 1e-9);
    }

    #[test]
    fn test_audio_duration_zero_rate() {
        let f = AudioFrameInfo::new(0, 1, 1024, 0, 2, 0);
        assert_eq!(f.duration_secs(), 0.0);
    }

    // 10. AudioFrameInfo::total_samples
    #[test]
    fn test_total_samples() {
        let f = AudioFrameInfo::new(0, 1, 512, 48_000, 8, 0);
        assert_eq!(f.total_samples(), 512 * 8);
    }

    // 11. AudioFrameInfo::bytes_per_second
    #[test]
    fn test_bytes_per_second() {
        // 1024 samples at 48 kHz = ~21.33 ms; 4096 bytes / 0.02133 s ≈ 192_000 B/s
        let f = AudioFrameInfo::new(0, 1024, 1024, 48_000, 2, 4096);
        let bps = f.bytes_per_second();
        // 4096 / (1024/48000) = 192_000
        assert!((bps - 192_000.0).abs() < 1.0);
    }

    // 12. AudioFrameInfo::bytes_per_second – zero samples
    #[test]
    fn test_bytes_per_second_zero_samples() {
        let f = AudioFrameInfo::new(0, 0, 0, 48_000, 2, 1000);
        assert_eq!(f.bytes_per_second(), 0.0);
    }

    // 13. VideoFrameInfo::bits_per_pixel – zero pixels
    #[test]
    fn test_bpp_zero_pixels() {
        let f = VideoFrameInfo::new(
            0,
            0,
            1,
            0,
            1080,
            FrameType::Intra,
            ColorPrimaries::Bt709,
            1000,
        );
        assert_eq!(f.bits_per_pixel(), 0.0);
    }

    // 14. BiPredicted is not a key frame
    #[test]
    fn test_b_frame_not_key() {
        let f = VideoFrameInfo::new(
            100,
            80,
            1,
            1920,
            1080,
            FrameType::BiPredicted,
            ColorPrimaries::Bt2020,
            10_000,
        );
        assert!(!f.is_key_frame());
    }
}
