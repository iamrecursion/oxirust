//! AAF MediaDescriptor module
//!
//! Provides `ContainerDef`, `VideoDescriptor`, `AudioDescriptor`, and `MediaDescriptor`
//! for describing the format of essence data in an AAF file.

#[allow(dead_code)]
/// Container format for essence data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerDef {
    /// Material Exchange Format
    Mxf,
    /// Apple QuickTime
    QuickTime,
    /// Avid proprietary container
    Avid,
    /// Broadcast Wave Format
    Bwf,
    /// Microsoft Wave
    Wave,
}

impl ContainerDef {
    /// Returns the conventional file extension for this container
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            ContainerDef::Mxf => "mxf",
            ContainerDef::QuickTime => "mov",
            ContainerDef::Avid => "omf",
            ContainerDef::Bwf => "bwf",
            ContainerDef::Wave => "wav",
        }
    }

    /// Returns `true` when the container is MXF
    #[must_use]
    pub fn is_mxf(&self) -> bool {
        matches!(self, ContainerDef::Mxf)
    }
}

#[allow(dead_code)]
/// Descriptor for video essence
#[derive(Debug, Clone)]
pub struct VideoDescriptor {
    /// Codec name (e.g. "XDCAM HD 422", "ProRes 422")
    pub codec: String,
    /// Frame width in pixels
    pub width: u32,
    /// Frame height in pixels
    pub height: u32,
    /// Frame rate numerator
    pub frame_rate_num: u32,
    /// Frame rate denominator
    pub frame_rate_den: u32,
    /// Bit depth (e.g. 8, 10, 12)
    pub bit_depth: u8,
    /// Color space (e.g. "YCbCr", "RGB")
    pub color_space: String,
}

impl VideoDescriptor {
    /// Create a new `VideoDescriptor`
    #[must_use]
    pub fn new(
        codec: String,
        width: u32,
        height: u32,
        frame_rate_num: u32,
        frame_rate_den: u32,
        bit_depth: u8,
        color_space: String,
    ) -> Self {
        Self {
            codec,
            width,
            height,
            frame_rate_num,
            frame_rate_den,
            bit_depth,
            color_space,
        }
    }

    /// Compute the frame rate as a floating-point value
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn frame_rate(&self) -> f64 {
        if self.frame_rate_den == 0 {
            return 0.0;
        }
        self.frame_rate_num as f64 / self.frame_rate_den as f64
    }

    /// Returns `true` when the resolution qualifies as HD (1280x720 or 1920x1080)
    #[must_use]
    pub fn is_hd(&self) -> bool {
        (self.width >= 1280 && self.height >= 720) && !self.is_4k()
    }

    /// Returns `true` when the resolution qualifies as 4K (width >= 3840)
    #[must_use]
    pub fn is_4k(&self) -> bool {
        self.width >= 3840
    }
}

#[allow(dead_code)]
/// Descriptor for audio essence
#[derive(Debug, Clone)]
pub struct AudioDescriptor {
    /// Codec name (e.g. "PCM", "AAC")
    pub codec: String,
    /// Sample rate in Hz (e.g. 48000)
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u8,
    /// Bit depth (e.g. 16, 24, 32)
    pub bit_depth: u8,
}

impl AudioDescriptor {
    /// Create a new `AudioDescriptor`
    #[must_use]
    pub fn new(codec: String, sample_rate: u32, channels: u8, bit_depth: u8) -> Self {
        Self {
            codec,
            sample_rate,
            channels,
            bit_depth,
        }
    }

    /// Returns `true` when the codec is PCM (case-insensitive)
    #[must_use]
    pub fn is_pcm(&self) -> bool {
        self.codec.to_uppercase() == "PCM"
    }
}

#[allow(dead_code)]
/// Base descriptor carrying container info and essence length
#[derive(Debug, Clone)]
pub struct MediaDescriptor {
    /// Container format
    pub container: ContainerDef,
    /// Length of essence data in bytes
    pub essence_length: u64,
}

impl MediaDescriptor {
    /// Create a new `MediaDescriptor`
    #[must_use]
    pub fn new(container: ContainerDef, essence_length: u64) -> Self {
        Self {
            container,
            essence_length,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ContainerDef tests ---

    #[test]
    fn test_container_mxf_extension() {
        assert_eq!(ContainerDef::Mxf.extension(), "mxf");
    }

    #[test]
    fn test_container_quicktime_extension() {
        assert_eq!(ContainerDef::QuickTime.extension(), "mov");
    }

    #[test]
    fn test_container_avid_extension() {
        assert_eq!(ContainerDef::Avid.extension(), "omf");
    }

    #[test]
    fn test_container_bwf_extension() {
        assert_eq!(ContainerDef::Bwf.extension(), "bwf");
    }

    #[test]
    fn test_container_wave_extension() {
        assert_eq!(ContainerDef::Wave.extension(), "wav");
    }

    #[test]
    fn test_container_is_mxf_true() {
        assert!(ContainerDef::Mxf.is_mxf());
    }

    #[test]
    fn test_container_is_mxf_false() {
        assert!(!ContainerDef::QuickTime.is_mxf());
    }

    // --- VideoDescriptor tests ---

    fn hd_video() -> VideoDescriptor {
        VideoDescriptor::new("ProRes 422".into(), 1920, 1080, 25, 1, 8, "YCbCr".into())
    }

    fn uhd_video() -> VideoDescriptor {
        VideoDescriptor::new("H.265".into(), 3840, 2160, 24, 1, 10, "YCbCr".into())
    }

    fn sd_video() -> VideoDescriptor {
        VideoDescriptor::new("DV".into(), 720, 576, 25, 1, 8, "YCbCr".into())
    }

    #[test]
    fn test_video_descriptor_frame_rate() {
        let v = hd_video();
        assert!((v.frame_rate() - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_video_descriptor_frame_rate_zero_den() {
        let v = VideoDescriptor::new("X".into(), 1920, 1080, 25, 0, 8, "YCbCr".into());
        assert_eq!(v.frame_rate(), 0.0);
    }

    #[test]
    fn test_video_descriptor_is_hd_true() {
        assert!(hd_video().is_hd());
    }

    #[test]
    fn test_video_descriptor_is_hd_false_sd() {
        assert!(!sd_video().is_hd());
    }

    #[test]
    fn test_video_descriptor_is_hd_false_4k() {
        // 4K is not HD
        assert!(!uhd_video().is_hd());
    }

    #[test]
    fn test_video_descriptor_is_4k_true() {
        assert!(uhd_video().is_4k());
    }

    #[test]
    fn test_video_descriptor_is_4k_false() {
        assert!(!hd_video().is_4k());
    }

    // --- AudioDescriptor tests ---

    #[test]
    fn test_audio_descriptor_is_pcm_true() {
        let a = AudioDescriptor::new("PCM".into(), 48000, 2, 24);
        assert!(a.is_pcm());
    }

    #[test]
    fn test_audio_descriptor_is_pcm_case_insensitive() {
        let a = AudioDescriptor::new("pcm".into(), 48000, 2, 24);
        assert!(a.is_pcm());
    }

    #[test]
    fn test_audio_descriptor_is_pcm_false() {
        let a = AudioDescriptor::new("AAC".into(), 44100, 2, 16);
        assert!(!a.is_pcm());
    }

    // --- MediaDescriptor tests ---

    #[test]
    fn test_media_descriptor_fields() {
        let md = MediaDescriptor::new(ContainerDef::Mxf, 1_000_000);
        assert!(md.container.is_mxf());
        assert_eq!(md.essence_length, 1_000_000);
    }
}
