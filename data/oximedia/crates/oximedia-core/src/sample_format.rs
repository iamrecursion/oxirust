//! Audio and video sample format definitions.
//!
//! This module provides [`VideoPixelFormat`], [`AudioSampleFormat`], and
//! [`SampleFormatInfo`] for describing the pixel/sample layouts of media frames.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Pixel format for video frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VideoPixelFormat {
    /// YUV 4:2:0 planar (3 planes: Y, Cb, Cr).
    Yuv420p,
    /// YUV 4:2:2 planar (3 planes: Y, Cb, Cr).
    Yuv422p,
    /// YUV 4:4:4 planar (3 planes: Y, Cb, Cr).
    Yuv444p,
    /// Packed RGBA (4 bytes per pixel).
    Rgba,
    /// Packed RGB 24-bit (3 bytes per pixel).
    Rgb24,
    /// Semi-planar YUV 4:2:0 (2 planes: Y, interleaved `CbCr`).
    Nv12,
    /// 10-bit semi-planar YUV 4:2:0 (P010 little-endian, 2 planes).
    P010,
}

impl VideoPixelFormat {
    /// Returns the number of bits used per pixel.
    ///
    /// For planar formats this is the average over all planes.
    #[must_use]
    pub fn bits_per_pixel(&self) -> u8 {
        match self {
            Self::Yuv420p | Self::Nv12 => 12,
            Self::Yuv422p => 16,
            Self::Yuv444p | Self::Rgb24 => 24,
            Self::Rgba => 32,
            Self::P010 => 15, // 10-bit 4:2:0 → 15 bits on average
        }
    }

    /// Returns `true` if the format uses separate planes for each component.
    #[must_use]
    pub fn is_planar(&self) -> bool {
        matches!(
            self,
            Self::Yuv420p | Self::Yuv422p | Self::Yuv444p | Self::Nv12 | Self::P010
        )
    }

    /// Returns `(horizontal, vertical)` chroma subsampling factors.
    ///
    /// A factor of 2 means chroma is halved in that dimension.
    /// Packed RGB formats return `(1, 1)` (no subsampling).
    #[must_use]
    pub fn chroma_subsampling(&self) -> (u8, u8) {
        match self {
            Self::Yuv420p | Self::Nv12 | Self::P010 => (2, 2),
            Self::Yuv422p => (2, 1),
            Self::Yuv444p | Self::Rgba | Self::Rgb24 => (1, 1),
        }
    }
}

/// Sample format for audio frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioSampleFormat {
    /// Signed 16-bit integer, interleaved.
    S16,
    /// Signed 32-bit integer, interleaved.
    S32,
    /// 32-bit float, interleaved.
    F32,
    /// 64-bit float, interleaved.
    F64,
    /// Signed 16-bit integer, planar.
    S16P,
    /// Signed 32-bit integer, planar.
    S32P,
    /// 32-bit float, planar.
    F32P,
}

impl AudioSampleFormat {
    /// Returns the number of bytes per audio sample.
    #[must_use]
    pub fn bytes_per_sample(&self) -> u8 {
        match self {
            Self::S16 | Self::S16P => 2,
            Self::S32 | Self::F32 | Self::S32P | Self::F32P => 4,
            Self::F64 => 8,
        }
    }

    /// Returns `true` if the format stores samples in separate per-channel planes.
    #[must_use]
    pub fn is_planar(&self) -> bool {
        matches!(self, Self::S16P | Self::S32P | Self::F32P)
    }

    /// Returns `true` if the samples are floating-point values.
    #[must_use]
    pub fn is_float(&self) -> bool {
        matches!(self, Self::F32 | Self::F64 | Self::F32P)
    }
}

/// Combined format descriptor that may carry video, audio, or both.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleFormatInfo {
    /// Optional video pixel format.
    pub video: Option<VideoPixelFormat>,
    /// Optional audio sample format.
    pub audio: Option<AudioSampleFormat>,
}

impl SampleFormatInfo {
    /// Creates a new `SampleFormatInfo`.
    #[must_use]
    pub fn new(video: Option<VideoPixelFormat>, audio: Option<AudioSampleFormat>) -> Self {
        Self { video, audio }
    }

    /// Returns `true` if a video format is present.
    #[must_use]
    pub fn has_video(&self) -> bool {
        self.video.is_some()
    }

    /// Returns `true` if an audio format is present.
    #[must_use]
    pub fn has_audio(&self) -> bool {
        self.audio.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- VideoPixelFormat ---

    #[test]
    fn test_yuv420p_bits_per_pixel() {
        assert_eq!(VideoPixelFormat::Yuv420p.bits_per_pixel(), 12);
    }

    #[test]
    fn test_rgba_bits_per_pixel() {
        assert_eq!(VideoPixelFormat::Rgba.bits_per_pixel(), 32);
    }

    #[test]
    fn test_rgb24_bits_per_pixel() {
        assert_eq!(VideoPixelFormat::Rgb24.bits_per_pixel(), 24);
    }

    #[test]
    fn test_yuv422p_bits_per_pixel() {
        assert_eq!(VideoPixelFormat::Yuv422p.bits_per_pixel(), 16);
    }

    #[test]
    fn test_nv12_is_planar() {
        assert!(VideoPixelFormat::Nv12.is_planar());
    }

    #[test]
    fn test_rgba_is_not_planar() {
        assert!(!VideoPixelFormat::Rgba.is_planar());
    }

    #[test]
    fn test_yuv420p_chroma_subsampling() {
        assert_eq!(VideoPixelFormat::Yuv420p.chroma_subsampling(), (2, 2));
    }

    #[test]
    fn test_yuv422p_chroma_subsampling() {
        assert_eq!(VideoPixelFormat::Yuv422p.chroma_subsampling(), (2, 1));
    }

    #[test]
    fn test_yuv444p_chroma_subsampling() {
        assert_eq!(VideoPixelFormat::Yuv444p.chroma_subsampling(), (1, 1));
    }

    #[test]
    fn test_p010_bits_and_subsampling() {
        assert_eq!(VideoPixelFormat::P010.bits_per_pixel(), 15);
        assert_eq!(VideoPixelFormat::P010.chroma_subsampling(), (2, 2));
    }

    // --- AudioSampleFormat ---

    #[test]
    fn test_s16_bytes_per_sample() {
        assert_eq!(AudioSampleFormat::S16.bytes_per_sample(), 2);
    }

    #[test]
    fn test_f64_bytes_per_sample() {
        assert_eq!(AudioSampleFormat::F64.bytes_per_sample(), 8);
    }

    #[test]
    fn test_f32p_is_planar_and_float() {
        assert!(AudioSampleFormat::F32P.is_planar());
        assert!(AudioSampleFormat::F32P.is_float());
    }

    #[test]
    fn test_s32_is_not_planar_not_float() {
        assert!(!AudioSampleFormat::S32.is_planar());
        assert!(!AudioSampleFormat::S32.is_float());
    }

    // --- SampleFormatInfo ---

    #[test]
    fn test_sample_format_info_has_video() {
        let info = SampleFormatInfo::new(Some(VideoPixelFormat::Yuv420p), None);
        assert!(info.has_video());
        assert!(!info.has_audio());
    }

    #[test]
    fn test_sample_format_info_has_audio() {
        let info = SampleFormatInfo::new(None, Some(AudioSampleFormat::F32));
        assert!(!info.has_video());
        assert!(info.has_audio());
    }

    #[test]
    fn test_sample_format_info_both() {
        let info =
            SampleFormatInfo::new(Some(VideoPixelFormat::Nv12), Some(AudioSampleFormat::S16P));
        assert!(info.has_video());
        assert!(info.has_audio());
    }

    #[test]
    fn test_sample_format_info_neither() {
        let info = SampleFormatInfo::new(None, None);
        assert!(!info.has_video());
        assert!(!info.has_audio());
    }
}
