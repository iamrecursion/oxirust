//! Chroma keying VFX module.
//!
//! Distance-based green/blue-screen keying with spill suppression.

/// Pre-defined or custom key colour for chroma keying.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum KeyColor {
    /// Standard green-screen colour.
    GreenScreen,
    /// Standard blue-screen colour.
    BlueScreen,
    /// Caller-specified RGB colour (components in [0.0, 1.0]).
    Custom([f32; 3]),
}

impl KeyColor {
    /// Return the key colour as a normalised `[R, G, B]` array.
    #[must_use]
    pub fn rgb(&self) -> [f32; 3] {
        match self {
            Self::GreenScreen => [0.0, 1.0, 0.0],
            Self::BlueScreen => [0.0, 0.0, 1.0],
            Self::Custom(c) => *c,
        }
    }
}

/// Parameters controlling the chroma keying algorithm.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ChromaKeyParams {
    /// Target key colour.
    pub key_color: KeyColor,
    /// How similar a pixel must be to the key to be fully keyed (0–1).
    pub tolerance: f32,
    /// Transition softness around the tolerance boundary (0–1).
    pub softness: f32,
    /// How aggressively to suppress colour spill (0 = none, 1 = full).
    pub spill_suppress: f32,
}

impl ChromaKeyParams {
    /// Sensible defaults for green-screen work.
    #[must_use]
    pub fn green_screen_default() -> Self {
        Self {
            key_color: KeyColor::GreenScreen,
            tolerance: 0.35,
            softness: 0.1,
            spill_suppress: 0.5,
        }
    }

    /// Sensible defaults for blue-screen work.
    #[must_use]
    pub fn blue_screen_default() -> Self {
        Self {
            key_color: KeyColor::BlueScreen,
            tolerance: 0.35,
            softness: 0.1,
            spill_suppress: 0.5,
        }
    }
}

/// The result of keying a single pixel.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AlphaResult {
    /// Alpha value: 0.0 = fully keyed (transparent), 1.0 = fully opaque.
    pub alpha: f32,
    /// RGB colour after spill suppression, normalised to [0.0, 1.0].
    pub spill_corrected: [f32; 3],
}

impl AlphaResult {
    /// Returns `true` when the alpha is below `threshold` (pixel is keyed).
    #[must_use]
    pub fn is_keyed(&self, threshold: f32) -> bool {
        self.alpha < threshold
    }
}

/// Compute the keying result for a single pixel.
///
/// The algorithm:
/// 1. Measure chroma distance between the pixel and the key colour.
/// 2. Convert distance to an alpha in `[0, 1]` using tolerance and softness.
/// 3. Suppress spill by blending the key channel towards the average of
///    the other two channels.
#[must_use]
pub fn compute_alpha(pixel: [f32; 3], params: &ChromaKeyParams) -> AlphaResult {
    let key = params.key_color.rgb();

    // Euclidean distance in RGB space
    let dist =
        ((pixel[0] - key[0]).powi(2) + (pixel[1] - key[1]).powi(2) + (pixel[2] - key[2]).powi(2))
            .sqrt();

    let tol = params.tolerance.clamp(0.001, 1.0);
    let soft = params.softness.clamp(0.0, 1.0);
    let outer = tol + soft;

    // Alpha: 0 inside tolerance, ramps up through softness region, 1 outside
    let alpha = if dist < tol {
        0.0f32
    } else if dist < outer {
        (dist - tol) / soft.max(1e-6)
    } else {
        1.0
    }
    .clamp(0.0, 1.0);

    // Spill suppression: desaturate the key channel towards luminance
    let mut corrected = pixel;
    let key_channel_idx = key
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(1, |(i, _)| i);

    let other_avg = {
        let sum: f32 = pixel
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != key_channel_idx)
            .map(|(_, v)| v)
            .sum();
        sum / 2.0
    };

    let spill = params.spill_suppress.clamp(0.0, 1.0);
    corrected[key_channel_idx] = corrected[key_channel_idx] * (1.0 - spill) + other_avg * spill;

    AlphaResult {
        alpha,
        spill_corrected: corrected,
    }
}

/// Apply chroma keying to a flat RGB byte buffer, returning an RGBA buffer.
///
/// `pixels` must be `width * height * 3` bytes (RGB).
/// Output is `width * height * 4` bytes (RGBA).
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub fn apply_chroma_key(
    pixels: &[u8],
    width: u32,
    height: u32,
    params: &ChromaKeyParams,
) -> Vec<u8> {
    let n = (width * height) as usize;
    let mut output = Vec::with_capacity(n * 4);

    for i in 0..n {
        let base = i * 3;
        let pixel = [
            pixels[base] as f32 / 255.0,
            pixels[base + 1] as f32 / 255.0,
            pixels[base + 2] as f32 / 255.0,
        ];

        let result = compute_alpha(pixel, params);

        output.push(
            (result.spill_corrected[0] * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8,
        );
        output.push(
            (result.spill_corrected[1] * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8,
        );
        output.push(
            (result.spill_corrected[2] * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8,
        );
        output.push((result.alpha * 255.0).round().clamp(0.0, 255.0) as u8);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------ //
    // KeyColor
    // ------------------------------------------------------------------ //

    #[test]
    fn test_green_screen_rgb() {
        let c = KeyColor::GreenScreen.rgb();
        assert_eq!(c, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_blue_screen_rgb() {
        let c = KeyColor::BlueScreen.rgb();
        assert_eq!(c, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_custom_rgb_passthrough() {
        let c = KeyColor::Custom([0.5, 0.3, 0.1]).rgb();
        assert_eq!(c, [0.5, 0.3, 0.1]);
    }

    // ------------------------------------------------------------------ //
    // ChromaKeyParams
    // ------------------------------------------------------------------ //

    #[test]
    fn test_green_screen_default_key_color() {
        let p = ChromaKeyParams::green_screen_default();
        assert_eq!(p.key_color, KeyColor::GreenScreen);
    }

    #[test]
    fn test_blue_screen_default_key_color() {
        let p = ChromaKeyParams::blue_screen_default();
        assert_eq!(p.key_color, KeyColor::BlueScreen);
    }

    // ------------------------------------------------------------------ //
    // AlphaResult
    // ------------------------------------------------------------------ //

    #[test]
    fn test_alpha_result_is_keyed_low_alpha() {
        let r = AlphaResult {
            alpha: 0.05,
            spill_corrected: [0.0; 3],
        };
        assert!(r.is_keyed(0.5));
    }

    #[test]
    fn test_alpha_result_not_keyed_high_alpha() {
        let r = AlphaResult {
            alpha: 0.9,
            spill_corrected: [0.0; 3],
        };
        assert!(!r.is_keyed(0.5));
    }

    // ------------------------------------------------------------------ //
    // compute_alpha
    // ------------------------------------------------------------------ //

    #[test]
    fn test_compute_alpha_exact_key_is_zero() {
        let params = ChromaKeyParams::green_screen_default();
        let result = compute_alpha([0.0, 1.0, 0.0], &params);
        assert_eq!(result.alpha, 0.0);
    }

    #[test]
    fn test_compute_alpha_far_from_key_is_one() {
        let params = ChromaKeyParams::green_screen_default();
        // Pure red is far from green screen
        let result = compute_alpha([1.0, 0.0, 0.0], &params);
        assert!((result.alpha - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_alpha_border_pixel_partial() {
        let params = ChromaKeyParams {
            key_color: KeyColor::GreenScreen,
            tolerance: 0.3,
            softness: 0.3,
            spill_suppress: 0.0,
        };
        // Pixel at distance ~0.45 from green: inside softness region
        let result = compute_alpha([0.3, 0.8, 0.3], &params);
        assert!(result.alpha > 0.0 && result.alpha < 1.0);
    }

    #[test]
    fn test_compute_alpha_spill_suppression_reduces_green() {
        let params = ChromaKeyParams {
            key_color: KeyColor::GreenScreen,
            tolerance: 0.0,
            softness: 0.0,
            spill_suppress: 1.0,
        };
        // Near-green pixel: green should be reduced towards avg(R, B)
        let pixel = [0.1, 0.9, 0.1];
        let result = compute_alpha(pixel, &params);
        // With full spill suppress, green channel → average of red and blue
        let expected_g = (pixel[0] + pixel[2]) / 2.0;
        assert!((result.spill_corrected[1] - expected_g).abs() < 1e-5);
    }

    #[test]
    fn test_compute_alpha_no_spill_unchanged() {
        let params = ChromaKeyParams {
            key_color: KeyColor::GreenScreen,
            tolerance: 0.0,
            softness: 0.0,
            spill_suppress: 0.0,
        };
        let pixel = [0.5, 0.8, 0.3];
        let result = compute_alpha(pixel, &params);
        // No spill: corrected should equal original
        assert!((result.spill_corrected[0] - pixel[0]).abs() < 1e-5);
        assert!((result.spill_corrected[1] - pixel[1]).abs() < 1e-5);
        assert!((result.spill_corrected[2] - pixel[2]).abs() < 1e-5);
    }

    // ------------------------------------------------------------------ //
    // apply_chroma_key
    // ------------------------------------------------------------------ //

    #[test]
    fn test_apply_output_length_rgba() {
        let params = ChromaKeyParams::green_screen_default();
        let pixels = vec![0u8; 3 * 4 * 4]; // 4x4 black
        let out = apply_chroma_key(&pixels, 4, 4, &params);
        assert_eq!(out.len(), 4 * 4 * 4); // RGBA
    }

    #[test]
    fn test_apply_green_pixels_are_transparent() {
        let params = ChromaKeyParams::green_screen_default();
        // 2×2 solid green image
        let pixels = vec![0u8, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0];
        let out = apply_chroma_key(&pixels, 2, 2, &params);
        // Alpha channel (index 3, 7, 11, 15) should be near 0
        for i in 0..4 {
            let alpha = out[i * 4 + 3];
            assert!(alpha < 10, "Expected near-zero alpha, got {}", alpha);
        }
    }

    #[test]
    fn test_apply_red_pixels_are_opaque() {
        let params = ChromaKeyParams::green_screen_default();
        // 1×1 pure red
        let pixels = vec![255u8, 0, 0];
        let out = apply_chroma_key(&pixels, 1, 1, &params);
        assert_eq!(out[3], 255); // alpha should be fully opaque
    }

    #[test]
    fn test_apply_blue_screen_keys_blue() {
        let params = ChromaKeyParams::blue_screen_default();
        // 1×1 pure blue
        let pixels = vec![0u8, 0, 255];
        let out = apply_chroma_key(&pixels, 1, 1, &params);
        let alpha = out[3];
        assert!(
            alpha < 10,
            "Expected keyed (near-zero alpha), got {}",
            alpha
        );
    }
}
