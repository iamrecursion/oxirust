//! NDI alpha channel support for keying and compositing workflows.
//!
//! NDI sources can carry an alpha plane alongside the standard YUV video,
//! enabling downstream compositing without a separate luma-key or chroma-key
//! step.  This module defines the alpha mode, alpha frame type, and the
//! blending math required to premultiply/demultiply alpha for transport and
//! render.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// AlphaMode
// ---------------------------------------------------------------------------

/// How the alpha channel is encoded in an NDI video frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlphaMode {
    /// No alpha channel.  The frame is fully opaque.
    None,
    /// Straight (unassociated) alpha — color channels are NOT premultiplied.
    Straight,
    /// Premultiplied (associated) alpha — color channels are already scaled
    /// by the corresponding alpha value.  Preferred for compositing.
    Premultiplied,
}

impl AlphaMode {
    /// Returns `true` if the frame carries an alpha channel.
    pub fn has_alpha(self) -> bool {
        !matches!(self, Self::None)
    }

    /// Returns a human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Straight => "straight",
            Self::Premultiplied => "premultiplied",
        }
    }
}

impl Default for AlphaMode {
    fn default() -> Self {
        Self::None
    }
}

// ---------------------------------------------------------------------------
// AlphaFrame
// ---------------------------------------------------------------------------

/// A video frame augmented with a separate alpha plane.
///
/// The alpha plane has the same width and height as the luminance plane.
/// Each sample is an 8-bit value in [0, 255] where 0 is fully transparent
/// and 255 is fully opaque.
#[derive(Debug, Clone)]
pub struct AlphaFrame {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// RGBA or YUV pixel data (without embedded alpha).
    pub pixels: Vec<u8>,
    /// Alpha plane — one byte per pixel, row-major order.
    pub alpha: Vec<u8>,
    /// How the alpha is encoded.
    pub mode: AlphaMode,
}

impl AlphaFrame {
    /// Create a new `AlphaFrame` with the given dimensions and alpha mode.
    ///
    /// `pixels` must have at least `width * height * 3` bytes (RGB).
    /// `alpha` must have exactly `width * height` bytes.
    pub fn new(width: u32, height: u32, pixels: Vec<u8>, alpha: Vec<u8>, mode: AlphaMode) -> Self {
        Self {
            width,
            height,
            pixels,
            alpha,
            mode,
        }
    }

    /// Create an opaque frame (alpha = 255 everywhere).
    pub fn opaque(width: u32, height: u32, pixels: Vec<u8>) -> Self {
        let alpha = vec![255u8; (width * height) as usize];
        Self::new(width, height, pixels, alpha, AlphaMode::Straight)
    }

    /// Create a fully transparent frame (alpha = 0 everywhere).
    pub fn transparent(width: u32, height: u32) -> Self {
        let size = (width * height) as usize;
        let pixels = vec![0u8; size * 3];
        let alpha = vec![0u8; size];
        Self::new(width, height, pixels, alpha, AlphaMode::Straight)
    }

    /// Return the number of pixels in the frame.
    pub fn pixel_count(&self) -> usize {
        (self.width * self.height) as usize
    }

    /// Returns the average alpha value across all pixels.
    pub fn mean_alpha(&self) -> f64 {
        if self.alpha.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.alpha.iter().map(|&a| u64::from(a)).sum();
        sum as f64 / self.alpha.len() as f64
    }

    /// Convert from straight alpha to premultiplied alpha in place.
    ///
    /// `pixels` is treated as tightly-packed RGB triplets.
    pub fn premultiply(&mut self) {
        if self.mode != AlphaMode::Straight {
            return;
        }
        let n = self.pixel_count();
        for i in 0..n {
            let a = f32::from(self.alpha[i]) / 255.0;
            let base = i * 3;
            if base + 2 < self.pixels.len() {
                self.pixels[base] = (f32::from(self.pixels[base]) * a) as u8;
                self.pixels[base + 1] = (f32::from(self.pixels[base + 1]) * a) as u8;
                self.pixels[base + 2] = (f32::from(self.pixels[base + 2]) * a) as u8;
            }
        }
        self.mode = AlphaMode::Premultiplied;
    }

    /// Convert from premultiplied alpha back to straight alpha in place.
    ///
    /// Pixels where alpha is 0 are left unchanged (division by zero avoided).
    pub fn unpremultiply(&mut self) {
        if self.mode != AlphaMode::Premultiplied {
            return;
        }
        let n = self.pixel_count();
        for i in 0..n {
            let a = self.alpha[i];
            if a == 0 {
                continue;
            }
            let inv = 255.0 / f32::from(a);
            let base = i * 3;
            if base + 2 < self.pixels.len() {
                self.pixels[base] = (f32::from(self.pixels[base]) * inv).min(255.0) as u8;
                self.pixels[base + 1] = (f32::from(self.pixels[base + 1]) * inv).min(255.0) as u8;
                self.pixels[base + 2] = (f32::from(self.pixels[base + 2]) * inv).min(255.0) as u8;
            }
        }
        self.mode = AlphaMode::Straight;
    }
}

// ---------------------------------------------------------------------------
// Alpha compositing operations
// ---------------------------------------------------------------------------

/// Composite `src` over `dst` using standard Porter-Duff "over" operation.
///
/// Both frames must have the same dimensions and use straight alpha.
/// Returns the composited frame in straight alpha.
///
/// # Errors
///
/// Returns `None` if the frame dimensions do not match.
pub fn composite_over(src: &AlphaFrame, dst: &AlphaFrame) -> Option<AlphaFrame> {
    if src.width != dst.width || src.height != dst.height {
        return None;
    }
    let n = src.pixel_count();
    let mut out_pixels = vec![0u8; n * 3];
    let mut out_alpha = vec![0u8; n];

    for i in 0..n {
        let sa = f32::from(src.alpha[i]) / 255.0;
        let da = f32::from(dst.alpha[i]) / 255.0;
        // Porter-Duff "over": out_a = sa + da*(1-sa)
        let out_a = sa + da * (1.0 - sa);
        out_alpha[i] = (out_a * 255.0).min(255.0) as u8;

        let base = i * 3;
        if base + 2 < src.pixels.len() && base + 2 < dst.pixels.len() {
            for c in 0..3 {
                let sc = f32::from(src.pixels[base + c]);
                let dc = f32::from(dst.pixels[base + c]);
                // out_c = (sc*sa + dc*da*(1-sa)) / out_a
                let out_c = if out_a > 0.0 {
                    (sc * sa + dc * da * (1.0 - sa)) / out_a
                } else {
                    0.0
                };
                out_pixels[base + c] = out_c.min(255.0) as u8;
            }
        }
    }

    Some(AlphaFrame::new(
        src.width,
        src.height,
        out_pixels,
        out_alpha,
        AlphaMode::Straight,
    ))
}

/// Serialise the alpha plane into a compact NDI metadata XML string for
/// transmission over the NDI metadata channel.
///
/// The alpha data is base64-encoded and wrapped in a `<ndi_alpha>` element.
pub fn encode_alpha_metadata(frame: &AlphaFrame) -> String {
    // Simple hex encoding for deterministic, readable output (no external deps)
    let hex: String = frame
        .alpha
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<Vec<_>>()
        .join("");
    format!(
        "<ndi_alpha width=\"{}\" height=\"{}\" mode=\"{}\">{}</ndi_alpha>",
        frame.width,
        frame.height,
        frame.mode.label(),
        hex
    )
}

/// Decode an alpha plane from an NDI metadata XML string produced by
/// [`encode_alpha_metadata`].
///
/// Returns `None` if the XML is malformed or the hex data is invalid.
pub fn decode_alpha_metadata(xml: &str) -> Option<Vec<u8>> {
    // Extract hex content between the tags
    let start = xml.find('>')?;
    let end = xml.rfind('<')?;
    if end <= start + 1 {
        return None;
    }
    let hex = &xml[start + 1..end];
    if hex.len() % 2 != 0 {
        return None;
    }
    let mut bytes = Vec::with_capacity(hex.len() / 2);
    let mut chars = hex.chars();
    loop {
        let hi = chars.next()?;
        let lo = chars.next()?;
        let hi_val = hi.to_digit(16)? as u8;
        let lo_val = lo.to_digit(16)? as u8;
        bytes.push((hi_val << 4) | lo_val);
        if bytes.len() == hex.len() / 2 {
            break;
        }
    }
    Some(bytes)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rgb_frame(w: u32, h: u32, fill: u8, alpha: u8) -> AlphaFrame {
        let n = (w * h) as usize;
        AlphaFrame::new(w, h, vec![fill; n * 3], vec![alpha; n], AlphaMode::Straight)
    }

    #[test]
    fn test_alpha_mode_has_alpha() {
        assert!(!AlphaMode::None.has_alpha());
        assert!(AlphaMode::Straight.has_alpha());
        assert!(AlphaMode::Premultiplied.has_alpha());
    }

    #[test]
    fn test_alpha_mode_labels() {
        assert_eq!(AlphaMode::None.label(), "none");
        assert_eq!(AlphaMode::Straight.label(), "straight");
        assert_eq!(AlphaMode::Premultiplied.label(), "premultiplied");
    }

    #[test]
    fn test_opaque_frame() {
        let frame = AlphaFrame::opaque(4, 4, vec![128u8; 48]);
        assert_eq!(frame.alpha.len(), 16);
        assert!(frame.alpha.iter().all(|&a| a == 255));
        assert_eq!(frame.mode, AlphaMode::Straight);
    }

    #[test]
    fn test_transparent_frame() {
        let frame = AlphaFrame::transparent(2, 2);
        assert!(frame.alpha.iter().all(|&a| a == 0));
    }

    #[test]
    fn test_pixel_count() {
        let frame = make_rgb_frame(3, 5, 100, 200);
        assert_eq!(frame.pixel_count(), 15);
    }

    #[test]
    fn test_mean_alpha_full() {
        let frame = make_rgb_frame(2, 2, 100, 255);
        assert!((frame.mean_alpha() - 255.0).abs() < 0.01);
    }

    #[test]
    fn test_mean_alpha_half() {
        let frame = make_rgb_frame(2, 2, 100, 128);
        assert!((frame.mean_alpha() - 128.0).abs() < 0.01);
    }

    #[test]
    fn test_premultiply() {
        // Single pixel: R=100, G=200, B=50, A=128 (~50%)
        let mut frame = AlphaFrame::new(1, 1, vec![100, 200, 50], vec![128], AlphaMode::Straight);
        frame.premultiply();
        assert_eq!(frame.mode, AlphaMode::Premultiplied);
        // Expected: R=100*(128/255)≈50, G≈100, B≈25
        assert!(frame.pixels[0] < 60);
        assert!(frame.pixels[1] > 90 && frame.pixels[1] < 110);
        assert!(frame.pixels[2] < 35);
    }

    #[test]
    fn test_premultiply_full_alpha_no_change() {
        let orig = vec![100u8, 150, 200];
        let mut frame = AlphaFrame::new(1, 1, orig.clone(), vec![255], AlphaMode::Straight);
        frame.premultiply();
        assert_eq!(frame.pixels[0], 100);
        assert_eq!(frame.pixels[1], 150);
        assert_eq!(frame.pixels[2], 200);
    }

    #[test]
    fn test_premultiply_zero_alpha_results_in_black() {
        let mut frame = AlphaFrame::new(1, 1, vec![200, 200, 200], vec![0], AlphaMode::Straight);
        frame.premultiply();
        assert_eq!(frame.pixels[0], 0);
        assert_eq!(frame.pixels[1], 0);
        assert_eq!(frame.pixels[2], 0);
    }

    #[test]
    fn test_unpremultiply_roundtrip() {
        // Start with straight, premultiply, then unpremultiply — should be close to original
        let orig = vec![200u8, 100, 50];
        let mut frame = AlphaFrame::new(1, 1, orig.clone(), vec![200], AlphaMode::Straight);
        frame.premultiply();
        frame.unpremultiply();
        // Allow ±2 rounding error
        assert!((frame.pixels[0] as i32 - orig[0] as i32).abs() <= 2);
        assert!((frame.pixels[1] as i32 - orig[1] as i32).abs() <= 2);
        assert!((frame.pixels[2] as i32 - orig[2] as i32).abs() <= 2);
    }

    #[test]
    fn test_composite_over_dimension_mismatch() {
        let src = make_rgb_frame(4, 4, 100, 255);
        let dst = make_rgb_frame(2, 2, 50, 128);
        assert!(composite_over(&src, &dst).is_none());
    }

    #[test]
    fn test_composite_over_fully_opaque_src() {
        // Fully opaque src should completely replace dst
        let src = make_rgb_frame(2, 2, 200, 255);
        let dst = make_rgb_frame(2, 2, 50, 255);
        let out = composite_over(&src, &dst).expect("composite should succeed");
        // Output pixels should be very close to src (200)
        assert!(out.pixels[0] >= 195);
        // Output alpha should be 255
        assert_eq!(out.alpha[0], 255);
    }

    #[test]
    fn test_composite_over_fully_transparent_src() {
        // Fully transparent src should not change dst
        let src = make_rgb_frame(2, 2, 200, 0);
        let dst = make_rgb_frame(2, 2, 50, 255);
        let out = composite_over(&src, &dst).expect("composite should succeed");
        assert_eq!(out.pixels[0], 50);
    }

    #[test]
    fn test_encode_decode_alpha_metadata() {
        let frame = make_rgb_frame(2, 2, 100, 128);
        let xml = encode_alpha_metadata(&frame);
        assert!(xml.contains("ndi_alpha"));
        assert!(xml.contains("width=\"2\""));
        assert!(xml.contains("height=\"2\""));

        let decoded = decode_alpha_metadata(&xml).expect("decode should succeed");
        assert_eq!(decoded, frame.alpha);
    }

    #[test]
    fn test_encode_alpha_metadata_mode_label() {
        let frame = make_rgb_frame(1, 1, 0, 255);
        let xml = encode_alpha_metadata(&frame);
        assert!(xml.contains("mode=\"straight\""));
    }

    #[test]
    fn test_decode_alpha_metadata_empty() {
        assert!(decode_alpha_metadata("").is_none());
    }
}
