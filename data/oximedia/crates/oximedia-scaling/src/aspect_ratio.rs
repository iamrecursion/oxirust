//! Aspect ratio management and conversion.
//!
//! Provides common aspect ratio constants, letterboxing/pillarboxing utilities,
//! and mode-aware aspect ratio conversion.

/// An aspect ratio expressed as integer width and height components.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AspectRatio {
    /// Width component.
    pub width: u32,
    /// Height component.
    pub height: u32,
}

impl AspectRatio {
    /// Create a new aspect ratio and immediately reduce it.
    #[must_use]
    #[allow(dead_code)]
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }.reduce()
    }

    /// Reduce the aspect ratio to its lowest terms using GCD.
    #[must_use]
    #[allow(dead_code)]
    pub fn reduce(self) -> Self {
        let g = gcd(self.width, self.height);
        if g == 0 {
            return self;
        }
        Self {
            width: self.width / g,
            height: self.height / g,
        }
    }

    /// Convert to a floating-point ratio (width / height).
    #[must_use]
    #[allow(dead_code)]
    pub fn to_float(self) -> f32 {
        if self.height == 0 {
            return f32::INFINITY;
        }
        self.width as f32 / self.height as f32
    }

    /// Return `true` if this is a widescreen ratio (width/height > 1.5).
    #[must_use]
    #[allow(dead_code)]
    pub fn is_widescreen(self) -> bool {
        self.to_float() > 1.5
    }
}

/// Compute the greatest common divisor of two u32 values.
#[allow(dead_code)]
fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

// Common aspect ratio constants
/// 4:3 standard definition ratio.
#[allow(dead_code)]
pub const AR_4_3: AspectRatio = AspectRatio {
    width: 4,
    height: 3,
};
/// 16:9 HD widescreen ratio.
#[allow(dead_code)]
pub const AR_16_9: AspectRatio = AspectRatio {
    width: 16,
    height: 9,
};
/// 21:9 ultra-widescreen ratio.
#[allow(dead_code)]
pub const AR_21_9: AspectRatio = AspectRatio {
    width: 21,
    height: 9,
};
/// 1:1 square ratio.
#[allow(dead_code)]
pub const AR_1_1: AspectRatio = AspectRatio {
    width: 1,
    height: 1,
};
/// 9:16 vertical (portrait) ratio.
#[allow(dead_code)]
pub const AR_9_16: AspectRatio = AspectRatio {
    width: 9,
    height: 16,
};
/// 2.39:1 cinema scope ratio (239:100).
#[allow(dead_code)]
pub const AR_2_39_1: AspectRatio = AspectRatio {
    width: 239,
    height: 100,
};

/// Scaling mode for aspect ratio conversion.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleMode {
    /// Scale to fit entirely within destination; may leave bars.
    Fit,
    /// Scale to fill destination entirely; may crop edges.
    Fill,
    /// Stretch to exact destination dimensions ignoring aspect ratio.
    Stretch,
    /// Crop to destination aspect ratio centered on source.
    Crop,
}

impl ScaleMode {
    /// Compute the output rectangle (x, y, w, h) within a `(dst_w, dst_h)` canvas
    /// for content of aspect ratio `src` scaled to `dst`.
    ///
    /// Returns `(x_offset, y_offset, scaled_width, scaled_height)`.
    #[must_use]
    #[allow(dead_code)]
    pub fn compute_output_rect(self, src: AspectRatio, dst: AspectRatio) -> (u32, u32, u32, u32) {
        match self {
            Self::Stretch => (0, 0, dst.width, dst.height),
            Self::Fit => {
                let src_f = src.to_float();
                let dst_f = dst.to_float();
                if src_f >= dst_f {
                    // Source wider: fit width
                    let h = (dst.width as f32 / src_f).round() as u32;
                    let y = (dst.height.saturating_sub(h)) / 2;
                    (0, y, dst.width, h)
                } else {
                    // Source taller: fit height
                    let w = (dst.height as f32 * src_f).round() as u32;
                    let x = (dst.width.saturating_sub(w)) / 2;
                    (x, 0, w, dst.height)
                }
            }
            Self::Fill | Self::Crop => {
                let src_f = src.to_float();
                let dst_f = dst.to_float();
                if src_f >= dst_f {
                    // Source wider: fill height, crop sides
                    let w = (dst.height as f32 * src_f).round() as u32;
                    let x = (w.saturating_sub(dst.width)) / 2;
                    (x, 0, dst.width, dst.height)
                } else {
                    // Source taller: fill width, crop top/bottom
                    let h = (dst.width as f32 / src_f).round() as u32;
                    let y = (h.saturating_sub(dst.height)) / 2;
                    (0, y, dst.width, dst.height)
                }
            }
        }
    }
}

/// Configuration for letterboxing (horizontal bars on top and bottom).
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct LetterboxConfig {
    /// Bar color in linear float RGB.
    pub bar_color: [f32; 3],
    /// Bar opacity (0.0 = transparent, 1.0 = opaque).
    pub bar_opacity: f32,
}

impl Default for LetterboxConfig {
    fn default() -> Self {
        Self {
            bar_color: [0.0, 0.0, 0.0],
            bar_opacity: 1.0,
        }
    }
}

impl LetterboxConfig {
    /// Apply letterboxing to produce a `dst_w x dst_h` image containing `src`.
    ///
    /// Source is assumed to be a single-channel (luma) image.
    /// Output is a 3-channel (R, G, B) interleaved image.
    #[must_use]
    #[allow(dead_code)]
    pub fn apply(&self, src: &[f32], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<f32> {
        let dw = dst_w as usize;
        let dh = dst_h as usize;
        let sw = src_w as usize;
        let sh = src_h as usize;

        // Compute scaled content region
        let src_ar = AspectRatio::new(src_w, src_h);
        let dst_ar = AspectRatio::new(dst_w, dst_h);
        let (x_off, y_off, content_w, content_h) =
            ScaleMode::Fit.compute_output_rect(src_ar, dst_ar);

        let x_off = x_off as usize;
        let y_off = y_off as usize;
        let cw = content_w as usize;
        let ch = content_h as usize;

        let mut out = vec![0.0f32; dw * dh * 3];

        // Fill bars with bar_color
        let [br, bg, bb] = self.bar_color;
        for pixel_idx in 0..dw * dh {
            out[pixel_idx * 3] = br * self.bar_opacity;
            out[pixel_idx * 3 + 1] = bg * self.bar_opacity;
            out[pixel_idx * 3 + 2] = bb * self.bar_opacity;
        }

        // Blit scaled source
        for cy in 0..ch {
            for cx in 0..cw {
                let sx = (cx * sw / cw.max(1)).min(sw.saturating_sub(1));
                let sy = (cy * sh / ch.max(1)).min(sh.saturating_sub(1));
                let src_val = src[sy * sw + sx];
                let dst_x = x_off + cx;
                let dst_y = y_off + cy;
                if dst_x < dw && dst_y < dh {
                    let idx = (dst_y * dw + dst_x) * 3;
                    out[idx] = src_val;
                    out[idx + 1] = src_val;
                    out[idx + 2] = src_val;
                }
            }
        }

        out
    }
}

/// Configuration for pillarboxing (vertical bars on left and right).
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PillarboxConfig {
    /// Bar color in linear float RGB.
    pub bar_color: [f32; 3],
    /// Bar opacity (0.0 = transparent, 1.0 = opaque).
    pub bar_opacity: f32,
}

impl Default for PillarboxConfig {
    fn default() -> Self {
        Self {
            bar_color: [0.0, 0.0, 0.0],
            bar_opacity: 1.0,
        }
    }
}

impl PillarboxConfig {
    /// Apply pillarboxing to produce a `dst_w x dst_h` image containing `src`.
    #[must_use]
    #[allow(dead_code)]
    pub fn apply(&self, src: &[f32], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<f32> {
        // Reuse letterbox but with Fill mode and pillar orientation
        let dw = dst_w as usize;
        let dh = dst_h as usize;
        let sw = src_w as usize;
        let sh = src_h as usize;

        let src_ar = AspectRatio::new(src_w, src_h);
        let dst_ar = AspectRatio::new(dst_w, dst_h);
        let (x_off, y_off, content_w, content_h) =
            ScaleMode::Fit.compute_output_rect(src_ar, dst_ar);

        let x_off = x_off as usize;
        let y_off = y_off as usize;
        let cw = content_w as usize;
        let ch = content_h as usize;

        let mut out = vec![0.0f32; dw * dh * 3];

        let [br, bg, bb] = self.bar_color;
        for pixel_idx in 0..dw * dh {
            out[pixel_idx * 3] = br * self.bar_opacity;
            out[pixel_idx * 3 + 1] = bg * self.bar_opacity;
            out[pixel_idx * 3 + 2] = bb * self.bar_opacity;
        }

        for cy in 0..ch {
            for cx in 0..cw {
                let sx = (cx * sw / cw.max(1)).min(sw.saturating_sub(1));
                let sy = (cy * sh / ch.max(1)).min(sh.saturating_sub(1));
                let src_val = src[sy * sw + sx];
                let dst_x = x_off + cx;
                let dst_y = y_off + cy;
                if dst_x < dw && dst_y < dh {
                    let idx = (dst_y * dw + dst_x) * 3;
                    out[idx] = src_val;
                    out[idx + 1] = src_val;
                    out[idx + 2] = src_val;
                }
            }
        }

        out
    }
}

/// Aspect ratio converter.
pub struct AspectRatioConverter;

impl AspectRatioConverter {
    /// Convert a single-channel source image to a new aspect ratio using the given mode.
    #[must_use]
    #[allow(dead_code)]
    pub fn convert(
        src: &[f32],
        src_ar: AspectRatio,
        dst_ar: AspectRatio,
        mode: ScaleMode,
    ) -> Vec<f32> {
        let (x_off, y_off, content_w, content_h) = mode.compute_output_rect(src_ar, dst_ar);

        let dw = dst_ar.width as usize;
        let dh = dst_ar.height as usize;
        let sw = src_ar.width as usize;
        let sh = src_ar.height as usize;
        let cw = content_w as usize;
        let ch = content_h as usize;

        let mut out = vec![0.0f32; dw * dh];

        match mode {
            ScaleMode::Stretch => {
                // Simple nearest-neighbor stretch
                for dy in 0..dh {
                    for dx in 0..dw {
                        let sx = (dx * sw / dw.max(1)).min(sw.saturating_sub(1));
                        let sy = (dy * sh / dh.max(1)).min(sh.saturating_sub(1));
                        out[dy * dw + dx] = src[sy * sw + sx];
                    }
                }
            }
            ScaleMode::Fit => {
                let x_off = x_off as usize;
                let y_off = y_off as usize;
                for cy in 0..ch {
                    for cx in 0..cw {
                        let sx = (cx * sw / cw.max(1)).min(sw.saturating_sub(1));
                        let sy = (cy * sh / ch.max(1)).min(sh.saturating_sub(1));
                        let dst_x = x_off + cx;
                        let dst_y = y_off + cy;
                        if dst_x < dw && dst_y < dh {
                            out[dst_y * dw + dst_x] = src[sy * sw + sx];
                        }
                    }
                }
            }
            ScaleMode::Fill | ScaleMode::Crop => {
                // x_off/y_off represent crop offset into source space
                let crop_x = x_off as usize;
                let crop_y = y_off as usize;
                for dy in 0..dh {
                    for dx in 0..dw {
                        let sx = (crop_x + dx * cw / dw.max(1)).min(sw.saturating_sub(1));
                        let sy = (crop_y + dy * ch / dh.max(1)).min(sh.saturating_sub(1));
                        out[dy * dw + dx] = src[sy * sw + sx];
                    }
                }
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aspect_ratio_reduce() {
        let ar = AspectRatio::new(1920, 1080);
        assert_eq!(ar.width, 16);
        assert_eq!(ar.height, 9);
    }

    #[test]
    fn test_aspect_ratio_to_float() {
        let ar = AR_16_9;
        let f = ar.to_float();
        assert!((f - 16.0 / 9.0).abs() < 0.001);
    }

    #[test]
    fn test_is_widescreen_16_9() {
        assert!(AR_16_9.is_widescreen());
    }

    #[test]
    fn test_is_widescreen_4_3() {
        assert!(!AR_4_3.is_widescreen());
    }

    #[test]
    fn test_is_widescreen_1_1() {
        assert!(!AR_1_1.is_widescreen());
    }

    #[test]
    fn test_scale_mode_stretch() {
        let (x, y, w, h) = ScaleMode::Stretch.compute_output_rect(AR_4_3, AR_16_9);
        assert_eq!((x, y), (0, 0));
        assert_eq!(w, AR_16_9.width);
        assert_eq!(h, AR_16_9.height);
    }

    #[test]
    fn test_scale_mode_fit_wider_source() {
        // Source 16:9, Dst 4:3 → fit width, bars on top/bottom
        let (x, _y, w, _h) = ScaleMode::Fit.compute_output_rect(AR_16_9, AR_4_3);
        assert_eq!(x, 0);
        assert_eq!(w, AR_4_3.width);
    }

    #[test]
    fn test_letterbox_apply_size() {
        let src = vec![0.5f32; 16]; // 4x4
        let cfg = LetterboxConfig::default();
        let dst = cfg.apply(&src, 4, 4, 8, 6);
        assert_eq!(dst.len(), 8 * 6 * 3);
    }

    #[test]
    fn test_pillarbox_apply_size() {
        let src = vec![0.5f32; 12]; // 4x3
        let cfg = PillarboxConfig::default();
        let dst = cfg.apply(&src, 4, 3, 6, 6);
        assert_eq!(dst.len(), 6 * 6 * 3);
    }

    #[test]
    fn test_aspect_ratio_converter_stretch() {
        let src: Vec<f32> = (0..16).map(|i| i as f32 / 16.0).collect();
        let dst = AspectRatioConverter::convert(&src, AR_4_3, AR_16_9, ScaleMode::Stretch);
        assert_eq!(dst.len(), (AR_16_9.width * AR_16_9.height) as usize);
    }

    #[test]
    fn test_aspect_ratio_21_9_widescreen() {
        assert!(AR_21_9.is_widescreen());
    }

    #[test]
    fn test_aspect_ratio_9_16_not_widescreen() {
        assert!(!AR_9_16.is_widescreen());
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(16, 9), 1);
        assert_eq!(gcd(0, 5), 5);
    }

    #[test]
    fn test_aspect_ratio_converter_fit() {
        let src = vec![0.5f32; 16];
        let dst = AspectRatioConverter::convert(&src, AR_4_3, AR_16_9, ScaleMode::Fit);
        assert_eq!(dst.len(), (AR_16_9.width * AR_16_9.height) as usize);
    }
}
