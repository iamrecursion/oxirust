//! Chroma and luma keying effects for compositing workflows.
//!
//! All pixel values are linear-light `f32` in the range [0.0, 1.0].

#![allow(dead_code)]

/// Configuration for a chroma key operation.
#[derive(Debug, Clone)]
pub struct ChromaKeyConfig {
    /// The key colour (e.g. pure green or blue) in linear RGB.
    pub key_color: [f32; 3],
    /// Colour-similarity tolerance radius in RGB space [0.0, 1.0].
    pub tolerance: f32,
    /// Softness of the key edge — widens the transition zone.
    pub softness: f32,
}

impl ChromaKeyConfig {
    /// Preset for a pure green screen key.
    #[must_use]
    pub fn green_screen() -> Self {
        Self {
            key_color: [0.0, 1.0, 0.0],
            tolerance: 0.3,
            softness: 0.1,
        }
    }

    /// Preset for a pure blue screen key.
    #[must_use]
    pub fn blue_screen() -> Self {
        Self {
            key_color: [0.0, 0.0, 1.0],
            tolerance: 0.3,
            softness: 0.1,
        }
    }
}

/// Compute the chroma key mask for a pixel.
///
/// Returns 0.0 when the pixel is fully keyed out (transparent),
/// and 1.0 when it is fully opaque.
#[must_use]
pub fn chroma_key_mask(pixel: [f32; 3], config: &ChromaKeyConfig) -> f32 {
    // Euclidean distance in RGB space
    let dr = pixel[0] - config.key_color[0];
    let dg = pixel[1] - config.key_color[1];
    let db = pixel[2] - config.key_color[2];
    let dist = (dr * dr + dg * dg + db * db).sqrt();

    let inner = config.tolerance;
    let outer = (config.tolerance + config.softness.max(0.0)).max(inner + 1e-6);

    if dist <= inner {
        0.0 // fully keyed
    } else if dist >= outer {
        1.0 // fully opaque
    } else {
        // Linear ramp through the softness zone
        (dist - inner) / (outer - inner)
    }
}

/// Configuration for a luma key operation.
#[derive(Debug, Clone)]
pub struct LumaKeyConfig {
    /// Low luma threshold (pixels below this are affected first).
    pub low: f32,
    /// High luma threshold.
    pub high: f32,
    /// If `true`, the key selects bright areas instead of dark areas.
    pub invert: bool,
}

impl LumaKeyConfig {
    /// Apply the luma key to a single luma value, returning the opacity [0.0, 1.0].
    ///
    /// Without inversion: values inside [`low`, `high`] are keyed out (0.0).
    /// With inversion: values outside [`low`, `high`] are keyed out (0.0).
    #[must_use]
    pub fn apply(&self, luma: f32) -> f32 {
        let lo = self.low.min(self.high);
        let hi = self.low.max(self.high);

        let inside = luma >= lo && luma <= hi;
        let keyed_out = inside != self.invert; // XOR with invert flag

        if keyed_out {
            0.0
        } else {
            1.0
        }
    }
}

/// Compute the Rec. 709 luma value from a linear-light RGB pixel.
#[must_use]
#[inline]
pub fn luma_from_rgb(rgb: [f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

/// Spill suppression — reduces the influence of the key colour bleeding into
/// the subject.
pub struct SpillSuppressor;

impl SpillSuppressor {
    /// Suppress key-colour spill in `pixel` towards neutral grey.
    ///
    /// `key_color` defines the hue to suppress; `strength` ∈ [0.0, 1.0]
    /// controls how aggressively the spill is removed.
    #[must_use]
    pub fn suppress(pixel: [f32; 3], key_color: [f32; 3], strength: f32) -> [f32; 3] {
        // Find which channel is dominant in the key colour
        let key_max_idx = {
            let mut idx = 0;
            for i in 1..3 {
                if key_color[i] > key_color[idx] {
                    idx = i;
                }
            }
            idx
        };

        let mut out = pixel;
        let dominated = out[key_max_idx];

        // Average of the two non-dominant channels
        let others: [usize; 2] = match key_max_idx {
            0 => [1, 2],
            1 => [0, 2],
            _ => [0, 1],
        };
        let avg_others = (out[others[0]] + out[others[1]]) * 0.5;

        // Reduce the dominant channel towards the average
        let target = dominated.min(avg_others);
        out[key_max_idx] = dominated + (target - dominated) * strength.clamp(0.0, 1.0);

        out
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ChromaKeyConfig presets ────────────────────────────────────────────────

    #[test]
    fn test_green_screen_preset() {
        let cfg = ChromaKeyConfig::green_screen();
        assert_eq!(cfg.key_color, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_blue_screen_preset() {
        let cfg = ChromaKeyConfig::blue_screen();
        assert_eq!(cfg.key_color, [0.0, 0.0, 1.0]);
    }

    // ── chroma_key_mask ────────────────────────────────────────────────────────

    #[test]
    fn test_chroma_key_exact_match_is_zero() {
        let cfg = ChromaKeyConfig::green_screen();
        let mask = chroma_key_mask([0.0, 1.0, 0.0], &cfg);
        assert!((mask).abs() < 1e-5, "mask: {mask}");
    }

    #[test]
    fn test_chroma_key_white_is_opaque() {
        let cfg = ChromaKeyConfig::green_screen();
        let mask = chroma_key_mask([1.0, 1.0, 1.0], &cfg);
        assert!((mask - 1.0).abs() < 1e-5, "mask: {mask}");
    }

    #[test]
    fn test_chroma_key_red_is_opaque() {
        let cfg = ChromaKeyConfig::green_screen();
        let mask = chroma_key_mask([1.0, 0.0, 0.0], &cfg);
        assert!((mask - 1.0).abs() < 1e-5, "mask: {mask}");
    }

    #[test]
    fn test_chroma_key_softness_gives_partial() {
        let cfg = ChromaKeyConfig {
            key_color: [0.0, 1.0, 0.0],
            tolerance: 0.2,
            softness: 0.2,
        };
        // A pixel at distance 0.25 (between inner 0.2 and outer 0.4) → partial
        let pixel = [0.0, 0.75, 0.0]; // dist from green ≈ 0.25
        let mask = chroma_key_mask(pixel, &cfg);
        assert!(mask > 0.0 && mask < 1.0, "mask should be partial: {mask}");
    }

    // ── LumaKeyConfig ──────────────────────────────────────────────────────────

    #[test]
    fn test_luma_key_inside_range_keyed_out() {
        let cfg = LumaKeyConfig {
            low: 0.3,
            high: 0.7,
            invert: false,
        };
        assert!((cfg.apply(0.5) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_luma_key_outside_range_opaque() {
        let cfg = LumaKeyConfig {
            low: 0.3,
            high: 0.7,
            invert: false,
        };
        assert!((cfg.apply(0.1) - 1.0).abs() < 1e-5);
        assert!((cfg.apply(0.9) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_luma_key_inverted() {
        let cfg = LumaKeyConfig {
            low: 0.3,
            high: 0.7,
            invert: true,
        };
        // With invert, outside range is now keyed out
        assert!((cfg.apply(0.1) - 0.0).abs() < 1e-5);
        assert!((cfg.apply(0.5) - 1.0).abs() < 1e-5);
    }

    // ── luma_from_rgb ──────────────────────────────────────────────────────────

    #[test]
    fn test_luma_white_is_one() {
        let luma = luma_from_rgb([1.0, 1.0, 1.0]);
        assert!((luma - 1.0).abs() < 1e-5, "luma: {luma}");
    }

    #[test]
    fn test_luma_black_is_zero() {
        let luma = luma_from_rgb([0.0, 0.0, 0.0]);
        assert!((luma).abs() < 1e-5, "luma: {luma}");
    }

    #[test]
    fn test_luma_rec709_coefficients() {
        // Pure red: luma should be 0.2126
        let luma = luma_from_rgb([1.0, 0.0, 0.0]);
        assert!((luma - 0.2126).abs() < 1e-5, "luma: {luma}");
    }

    // ── SpillSuppressor ────────────────────────────────────────────────────────

    #[test]
    fn test_spill_suppress_green_channel_reduced() {
        let pixel = [0.1, 0.8, 0.1];
        let key = [0.0, 1.0, 0.0]; // green key
        let out = SpillSuppressor::suppress(pixel, key, 1.0);
        // The green channel should be reduced
        assert!(
            out[1] <= pixel[1],
            "green should be reduced: {} -> {}",
            pixel[1],
            out[1]
        );
    }

    #[test]
    fn test_spill_suppress_zero_strength() {
        let pixel = [0.1, 0.8, 0.1];
        let key = [0.0, 1.0, 0.0];
        let out = SpillSuppressor::suppress(pixel, key, 0.0);
        // Zero strength → no change
        for i in 0..3 {
            assert!((out[i] - pixel[i]).abs() < 1e-6, "channel {i}: {}", out[i]);
        }
    }

    #[test]
    fn test_spill_suppress_other_channels_unchanged() {
        let pixel = [0.3, 0.9, 0.2];
        let key = [0.0, 1.0, 0.0]; // green key → only G changes
        let out = SpillSuppressor::suppress(pixel, key, 1.0);
        // R and B should be unmodified
        assert!((out[0] - pixel[0]).abs() < 1e-6, "R changed: {}", out[0]);
        assert!((out[2] - pixel[2]).abs() < 1e-6, "B changed: {}", out[2]);
    }
}
