//! Codec capability and information descriptors.
//!
//! Provides lightweight descriptors for codec properties such as supported
//! profiles, levels, bit-depth, and chroma subsampling without depending on
//! any external codec libraries.

#![allow(dead_code)]

/// Chroma subsampling format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChromaSubsampling {
    /// 4:2:0 – half horizontal and vertical chroma resolution.
    Yuv420,
    /// 4:2:2 – half horizontal chroma resolution.
    Yuv422,
    /// 4:4:4 – full chroma resolution.
    Yuv444,
    /// Monochrome (luma-only).
    Monochrome,
}

impl ChromaSubsampling {
    /// Returns the horizontal chroma subsampling factor (1 or 2).
    #[must_use]
    pub fn horizontal_factor(self) -> u8 {
        match self {
            Self::Yuv420 | Self::Yuv422 => 2,
            Self::Yuv444 | Self::Monochrome => 1,
        }
    }

    /// Returns the vertical chroma subsampling factor (1 or 2).
    #[must_use]
    pub fn vertical_factor(self) -> u8 {
        match self {
            Self::Yuv420 => 2,
            Self::Yuv422 | Self::Yuv444 | Self::Monochrome => 1,
        }
    }

    /// Average bits per pixel for the given bit-depth.
    #[must_use]
    pub fn average_bpp(self, bit_depth: u8) -> f32 {
        let luma = f32::from(bit_depth);
        let chroma = match self {
            Self::Yuv420 => f32::from(bit_depth) * 0.5,
            Self::Yuv422 => f32::from(bit_depth),
            Self::Yuv444 => f32::from(bit_depth) * 2.0,
            Self::Monochrome => 0.0,
        };
        luma + chroma
    }
}

/// Video codec profile identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VideoProfile {
    /// AV1 Main profile.
    Av1Main,
    /// AV1 High profile.
    Av1High,
    /// AV1 Professional profile.
    Av1Professional,
    /// VP9 Profile 0 (8-bit 4:2:0).
    Vp9Profile0,
    /// VP9 Profile 1 (8-bit 4:2:2 / 4:4:4).
    Vp9Profile1,
    /// VP9 Profile 2 (10/12-bit 4:2:0).
    Vp9Profile2,
    /// VP9 Profile 3 (10/12-bit 4:2:2 / 4:4:4).
    Vp9Profile3,
    /// Theora.
    Theora,
    /// Generic / unknown profile.
    Unknown,
}

impl VideoProfile {
    /// Returns `true` if the profile supports high bit depth (10/12-bit).
    #[must_use]
    pub fn supports_high_bit_depth(self) -> bool {
        matches!(
            self,
            Self::Av1High | Self::Av1Professional | Self::Vp9Profile2 | Self::Vp9Profile3
        )
    }

    /// Returns `true` if the profile supports 4:4:4 chroma.
    #[must_use]
    pub fn supports_444(self) -> bool {
        matches!(
            self,
            Self::Av1High | Self::Av1Professional | Self::Vp9Profile1 | Self::Vp9Profile3
        )
    }

    /// Returns a short name string for the profile.
    #[must_use]
    pub fn short_name(self) -> &'static str {
        match self {
            Self::Av1Main => "AV1 Main",
            Self::Av1High => "AV1 High",
            Self::Av1Professional => "AV1 Professional",
            Self::Vp9Profile0 => "VP9 P0",
            Self::Vp9Profile1 => "VP9 P1",
            Self::Vp9Profile2 => "VP9 P2",
            Self::Vp9Profile3 => "VP9 P3",
            Self::Theora => "Theora",
            Self::Unknown => "Unknown",
        }
    }
}

/// Capabilities descriptor for a single video codec.
#[derive(Debug, Clone)]
pub struct VideoCodecInfo {
    /// Codec name (e.g. `"AV1"`, `"VP9"`).
    pub name: String,
    /// Active profile.
    pub profile: VideoProfile,
    /// Bit depth in use (8, 10, or 12).
    pub bit_depth: u8,
    /// Chroma subsampling format.
    pub chroma: ChromaSubsampling,
    /// Maximum frame width (pixels).
    pub max_width: u32,
    /// Maximum frame height (pixels).
    pub max_height: u32,
    /// Whether hardware acceleration is available.
    pub hw_accel: bool,
}

impl VideoCodecInfo {
    /// Creates a new `VideoCodecInfo`.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        profile: VideoProfile,
        bit_depth: u8,
        chroma: ChromaSubsampling,
        max_width: u32,
        max_height: u32,
        hw_accel: bool,
    ) -> Self {
        Self {
            name: name.into(),
            profile,
            bit_depth,
            chroma,
            max_width,
            max_height,
            hw_accel,
        }
    }

    /// Returns the maximum pixel count for this codec.
    #[must_use]
    pub fn max_pixels(&self) -> u64 {
        u64::from(self.max_width) * u64::from(self.max_height)
    }

    /// Returns `true` if the given resolution fits within the codec's limits.
    #[must_use]
    pub fn supports_resolution(&self, width: u32, height: u32) -> bool {
        width <= self.max_width && height <= self.max_height
    }

    /// Average bits per pixel at the codec's bit depth and chroma.
    #[must_use]
    pub fn average_bpp(&self) -> f32 {
        self.chroma.average_bpp(self.bit_depth)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. ChromaSubsampling::horizontal_factor
    #[test]
    fn test_chroma_horiz_factor_420() {
        assert_eq!(ChromaSubsampling::Yuv420.horizontal_factor(), 2);
    }

    #[test]
    fn test_chroma_horiz_factor_444() {
        assert_eq!(ChromaSubsampling::Yuv444.horizontal_factor(), 1);
    }

    // 2. ChromaSubsampling::vertical_factor
    #[test]
    fn test_chroma_vert_factor_420() {
        assert_eq!(ChromaSubsampling::Yuv420.vertical_factor(), 2);
    }

    #[test]
    fn test_chroma_vert_factor_422() {
        assert_eq!(ChromaSubsampling::Yuv422.vertical_factor(), 1);
    }

    // 3. ChromaSubsampling::average_bpp
    #[test]
    fn test_average_bpp_420_8bit() {
        // 8 (luma) + 8*0.5 (chroma) = 12.0
        assert!((ChromaSubsampling::Yuv420.average_bpp(8) - 12.0).abs() < 1e-4);
    }

    #[test]
    fn test_average_bpp_monochrome() {
        assert!((ChromaSubsampling::Monochrome.average_bpp(8) - 8.0).abs() < 1e-4);
    }

    // 4. VideoProfile::supports_high_bit_depth
    #[test]
    fn test_av1_main_no_high_bit_depth() {
        assert!(!VideoProfile::Av1Main.supports_high_bit_depth());
    }

    #[test]
    fn test_vp9_p2_high_bit_depth() {
        assert!(VideoProfile::Vp9Profile2.supports_high_bit_depth());
    }

    // 5. VideoProfile::supports_444
    #[test]
    fn test_vp9_p0_no_444() {
        assert!(!VideoProfile::Vp9Profile0.supports_444());
    }

    #[test]
    fn test_av1_high_supports_444() {
        assert!(VideoProfile::Av1High.supports_444());
    }

    // 6. VideoProfile::short_name
    #[test]
    fn test_short_name_av1_main() {
        assert_eq!(VideoProfile::Av1Main.short_name(), "AV1 Main");
    }

    #[test]
    fn test_short_name_unknown() {
        assert_eq!(VideoProfile::Unknown.short_name(), "Unknown");
    }

    // 7. VideoCodecInfo::max_pixels
    #[test]
    fn test_max_pixels() {
        let info = VideoCodecInfo::new(
            "AV1",
            VideoProfile::Av1Main,
            8,
            ChromaSubsampling::Yuv420,
            3840,
            2160,
            false,
        );
        assert_eq!(info.max_pixels(), 3840 * 2160);
    }

    // 8. VideoCodecInfo::supports_resolution – true
    #[test]
    fn test_supports_resolution_true() {
        let info = VideoCodecInfo::new(
            "VP9",
            VideoProfile::Vp9Profile0,
            8,
            ChromaSubsampling::Yuv420,
            1920,
            1080,
            false,
        );
        assert!(info.supports_resolution(1280, 720));
    }

    // 9. VideoCodecInfo::supports_resolution – false (too wide)
    #[test]
    fn test_supports_resolution_false() {
        let info = VideoCodecInfo::new(
            "VP9",
            VideoProfile::Vp9Profile0,
            8,
            ChromaSubsampling::Yuv420,
            1920,
            1080,
            false,
        );
        assert!(!info.supports_resolution(3840, 1080));
    }

    // 10. VideoCodecInfo::average_bpp
    #[test]
    fn test_codec_info_average_bpp() {
        let info = VideoCodecInfo::new(
            "AV1",
            VideoProfile::Av1Main,
            10,
            ChromaSubsampling::Yuv420,
            7680,
            4320,
            true,
        );
        // 10 + 10*0.5 = 15.0
        assert!((info.average_bpp() - 15.0).abs() < 1e-4);
    }

    // 11. hw_accel flag propagation
    #[test]
    fn test_hw_accel_flag() {
        let hw = VideoCodecInfo::new(
            "AV1",
            VideoProfile::Av1Main,
            8,
            ChromaSubsampling::Yuv420,
            1920,
            1080,
            true,
        );
        assert!(hw.hw_accel);
        let sw = VideoCodecInfo::new(
            "AV1",
            VideoProfile::Av1Main,
            8,
            ChromaSubsampling::Yuv420,
            1920,
            1080,
            false,
        );
        assert!(!sw.hw_accel);
    }

    // 12. average_bpp 444 at 8-bit
    #[test]
    fn test_average_bpp_444_8bit() {
        // 8 + 16 = 24
        assert!((ChromaSubsampling::Yuv444.average_bpp(8) - 24.0).abs() < 1e-4);
    }

    // 13. Theora short name
    #[test]
    fn test_theora_short_name() {
        assert_eq!(VideoProfile::Theora.short_name(), "Theora");
    }

    // 14. Monochrome factors
    #[test]
    fn test_monochrome_factors() {
        assert_eq!(ChromaSubsampling::Monochrome.horizontal_factor(), 1);
        assert_eq!(ChromaSubsampling::Monochrome.vertical_factor(), 1);
    }
}
