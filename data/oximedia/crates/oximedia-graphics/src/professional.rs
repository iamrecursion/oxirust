//! Professional broadcast features

use crate::color::Color;
use crate::error::Result;

/// Aspect ratio
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AspectRatio {
    /// 16:9 (HD, Full HD, 4K)
    Ratio16x9,
    /// 4:3 (SD)
    Ratio4x3,
    /// 1:1 (Square, social media)
    Ratio1x1,
    /// 21:9 (Ultrawide)
    Ratio21x9,
    /// 9:16 (Vertical, mobile)
    Ratio9x16,
    /// Custom ratio
    Custom(f32),
}

impl AspectRatio {
    /// Get aspect ratio as float
    #[must_use]
    pub fn as_float(&self) -> f32 {
        match self {
            Self::Ratio16x9 => 16.0 / 9.0,
            Self::Ratio4x3 => 4.0 / 3.0,
            Self::Ratio1x1 => 1.0,
            Self::Ratio21x9 => 21.0 / 9.0,
            Self::Ratio9x16 => 9.0 / 16.0,
            Self::Custom(ratio) => *ratio,
        }
    }

    /// Calculate dimensions for given width
    #[must_use]
    pub fn dimensions_from_width(&self, width: u32) -> (u32, u32) {
        let height = (width as f32 / self.as_float()) as u32;
        (width, height)
    }

    /// Calculate dimensions for given height
    #[must_use]
    pub fn dimensions_from_height(&self, height: u32) -> (u32, u32) {
        let width = (height as f32 * self.as_float()) as u32;
        (width, height)
    }
}

/// Broadcast-safe color limiter
pub struct BroadcastLimiter {
    /// Min luma (typically 16 for legal range)
    pub min_luma: u8,
    /// Max luma (typically 235 for legal range)
    pub max_luma: u8,
    /// Min chroma (typically 16)
    pub min_chroma: u8,
    /// Max chroma (typically 240)
    pub max_chroma: u8,
}

impl BroadcastLimiter {
    /// Create a new broadcast limiter with legal range
    #[must_use]
    pub fn legal_range() -> Self {
        Self {
            min_luma: 16,
            max_luma: 235,
            min_chroma: 16,
            max_chroma: 240,
        }
    }

    /// Create a new broadcast limiter with full range
    #[must_use]
    pub fn full_range() -> Self {
        Self {
            min_luma: 0,
            max_luma: 255,
            min_chroma: 0,
            max_chroma: 255,
        }
    }

    /// Limit color to broadcast-safe values
    #[must_use]
    pub fn limit_color(&self, color: Color) -> Color {
        let (y, cb, cr) = color.to_ycbcr();

        let y = y.clamp(self.min_luma, self.max_luma);
        let cb = cb.clamp(self.min_chroma, self.max_chroma);
        let cr = cr.clamp(self.min_chroma, self.max_chroma);

        Color::from_ycbcr(y, cb, cr, color.a)
    }

    /// Limit entire frame to broadcast-safe values
    pub fn limit_frame(&self, frame: &mut [u8]) -> Result<()> {
        for pixel in frame.chunks_exact_mut(4) {
            let color = Color::new(pixel[0], pixel[1], pixel[2], pixel[3]);
            let limited = self.limit_color(color);
            pixel[0] = limited.r;
            pixel[1] = limited.g;
            pixel[2] = limited.b;
            pixel[3] = limited.a;
        }
        Ok(())
    }
}

impl Default for BroadcastLimiter {
    fn default() -> Self {
        Self::legal_range()
    }
}

/// Anti-flicker filter for interlaced video
pub struct AntiFlickerFilter {
    /// Filter strength (0.0 to 1.0)
    pub strength: f32,
}

impl AntiFlickerFilter {
    /// Create a new anti-flicker filter
    #[must_use]
    pub fn new(strength: f32) -> Self {
        Self {
            strength: strength.clamp(0.0, 1.0),
        }
    }

    /// Apply anti-flicker filter to frame
    pub fn apply(&self, frame: &mut [u8], width: u32, height: u32) -> Result<()> {
        if frame.len() != (width * height * 4) as usize {
            return Err(crate::error::GraphicsError::InvalidParameter(
                "Frame size mismatch".to_string(),
            ));
        }

        // Simple vertical blur to reduce interlacing artifacts
        let mut temp = frame.to_vec();

        for y in 1..(height - 1) {
            for x in 0..width {
                let idx = ((y * width + x) * 4) as usize;
                let idx_above = (((y - 1) * width + x) * 4) as usize;
                let idx_below = (((y + 1) * width + x) * 4) as usize;

                for c in 0..3 {
                    // RGB only, not alpha
                    let above = f32::from(frame[idx_above + c]);
                    let current = f32::from(frame[idx + c]);
                    let below = f32::from(frame[idx_below + c]);

                    let blurred = (above + current * 2.0 + below) / 4.0;
                    let mixed = current * (1.0 - self.strength) + blurred * self.strength;

                    temp[idx + c] = mixed as u8;
                }
            }
        }

        frame.copy_from_slice(&temp);
        Ok(())
    }
}

impl Default for AntiFlickerFilter {
    fn default() -> Self {
        Self::new(0.3)
    }
}

/// Color space converter
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    /// BT.601 (SD)
    BT601,
    /// BT.709 (HD)
    BT709,
    /// BT.2020 (UHD)
    BT2020,
}

impl ColorSpace {
    /// Get RGB to YCbCr matrix coefficients
    #[must_use]
    pub fn rgb_to_ycbcr_coeffs(&self) -> (f32, f32, f32) {
        match self {
            Self::BT601 => (0.299, 0.587, 0.114),
            Self::BT709 => (0.2126, 0.7152, 0.0722),
            Self::BT2020 => (0.2627, 0.678, 0.0593),
        }
    }
}

/// Resolution preset
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionPreset {
    /// 720p HD (1280x720)
    HD720,
    /// 1080p Full HD (1920x1080)
    FullHD,
    /// 4K UHD (3840x2160)
    UHD4K,
    /// 8K UHD (7680x4320)
    UHD8K,
    /// SD NTSC (720x480)
    SDNTSC,
    /// SD PAL (720x576)
    SDPAL,
}

impl ResolutionPreset {
    /// Get width and height
    #[must_use]
    pub fn dimensions(&self) -> (u32, u32) {
        match self {
            Self::HD720 => (1280, 720),
            Self::FullHD => (1920, 1080),
            Self::UHD4K => (3840, 2160),
            Self::UHD8K => (7680, 4320),
            Self::SDNTSC => (720, 480),
            Self::SDPAL => (720, 576),
        }
    }

    /// Get aspect ratio
    #[must_use]
    pub fn aspect_ratio(&self) -> AspectRatio {
        match self {
            Self::HD720 | Self::FullHD | Self::UHD4K | Self::UHD8K => AspectRatio::Ratio16x9,
            Self::SDNTSC | Self::SDPAL => AspectRatio::Ratio4x3,
        }
    }
}

/// Framerate preset
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Framerate {
    /// 23.976 fps (film)
    Film23976,
    /// 24 fps
    Film24,
    /// 25 fps (PAL)
    PAL25,
    /// 29.97 fps (NTSC)
    NTSC2997,
    /// 30 fps
    FPS30,
    /// 50 fps (PAL progressive)
    PAL50,
    /// 59.94 fps (NTSC progressive)
    NTSC5994,
    /// 60 fps
    FPS60,
}

impl Framerate {
    /// Get framerate as float
    #[must_use]
    pub fn as_float(&self) -> f32 {
        match self {
            Self::Film23976 => 23.976,
            Self::Film24 => 24.0,
            Self::PAL25 => 25.0,
            Self::NTSC2997 => 29.97,
            Self::FPS30 => 30.0,
            Self::PAL50 => 50.0,
            Self::NTSC5994 => 59.94,
            Self::FPS60 => 60.0,
        }
    }

    /// Get frame duration in milliseconds
    #[must_use]
    pub fn frame_duration_ms(&self) -> f32 {
        1000.0 / self.as_float()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aspect_ratio() {
        let ar = AspectRatio::Ratio16x9;
        assert_eq!(ar.as_float(), 16.0 / 9.0);

        let (w, h) = ar.dimensions_from_width(1920);
        assert_eq!(w, 1920);
        assert_eq!(h, 1080);
    }

    #[test]
    fn test_aspect_ratio_4x3() {
        let ar = AspectRatio::Ratio4x3;
        let (w, h) = ar.dimensions_from_width(720);
        assert_eq!(w, 720);
        assert_eq!(h, 540);
    }

    #[test]
    fn test_broadcast_limiter() {
        let limiter = BroadcastLimiter::legal_range();
        let color = Color::rgb(255, 255, 255);
        let limited = limiter.limit_color(color);

        // White should be limited to legal range
        assert!(limited.r <= 235);
    }

    #[test]
    fn test_broadcast_limiter_frame() {
        let limiter = BroadcastLimiter::legal_range();
        let mut frame = vec![255u8; 100 * 100 * 4];

        let result = limiter.limit_frame(&mut frame);
        assert!(result.is_ok());
    }

    #[test]
    fn test_anti_flicker_filter() {
        let filter = AntiFlickerFilter::new(0.5);
        assert_eq!(filter.strength, 0.5);

        let mut frame = vec![128u8; 100 * 100 * 4];
        let result = filter.apply(&mut frame, 100, 100);
        assert!(result.is_ok());
    }

    #[test]
    fn test_color_space() {
        let cs = ColorSpace::BT709;
        let (kr, kg, kb) = cs.rgb_to_ycbcr_coeffs();
        assert_eq!(kr, 0.2126);
        assert_eq!(kg, 0.7152);
        assert_eq!(kb, 0.0722);
    }

    #[test]
    fn test_resolution_preset() {
        let res = ResolutionPreset::FullHD;
        assert_eq!(res.dimensions(), (1920, 1080));
        assert_eq!(res.aspect_ratio(), AspectRatio::Ratio16x9);
    }

    #[test]
    fn test_resolution_preset_4k() {
        let res = ResolutionPreset::UHD4K;
        assert_eq!(res.dimensions(), (3840, 2160));
    }

    #[test]
    fn test_framerate() {
        let fps = Framerate::FPS60;
        assert_eq!(fps.as_float(), 60.0);
        assert!((fps.frame_duration_ms() - 16.666667).abs() < 0.001);
    }

    #[test]
    fn test_framerate_film() {
        let fps = Framerate::Film23976;
        assert!((fps.as_float() - 23.976).abs() < 0.001);
    }
}
