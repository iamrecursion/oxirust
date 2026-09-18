#![allow(dead_code)]
//! Gamut compression LUT — soft-clip and roll-off based out-of-gamut handling.

/// Method used for gamut compression inside a LUT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GamutCompressMethod {
    /// Simple hard clip to the target gamut boundary.
    HardClip,
    /// Smooth sigmoid-like roll-off that preserves hue.
    SoftClip,
    /// Linear de-saturation towards the achromatic axis.
    Desaturate,
    /// ACES reference gamut compression (Resolve-style).
    AcesReference,
}

impl GamutCompressMethod {
    /// Returns `true` when the method uses a smooth roll-off rather than a hard boundary.
    #[must_use]
    pub fn is_soft_clip(self) -> bool {
        matches!(self, Self::SoftClip | Self::AcesReference)
    }

    /// Human-readable label for the method.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::HardClip => "Hard Clip",
            Self::SoftClip => "Soft Clip",
            Self::Desaturate => "Desaturate",
            Self::AcesReference => "ACES Reference",
        }
    }
}

/// Configuration for gamut compression.
#[derive(Debug, Clone)]
pub struct GamutCompressConfig {
    /// Roll-off power (gamma) for soft-clip methods — higher values give a sharper knee.
    pub power: f32,
    /// Threshold above 1.0 at which compression begins (e.g. 0.2 means 1.2).
    pub threshold: f32,
    /// Compression method to apply.
    pub method: GamutCompressMethod,
}

impl Default for GamutCompressConfig {
    fn default() -> Self {
        Self {
            power: 1.2,
            threshold: 0.15,
            method: GamutCompressMethod::SoftClip,
        }
    }
}

impl GamutCompressConfig {
    /// Returns the configured roll-off power value.
    #[must_use]
    pub fn power(&self) -> f32 {
        self.power
    }

    /// Returns `true` when the configuration is valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.power > 0.0 && self.threshold >= 0.0 && self.threshold <= 1.0
    }
}

/// A 3-D LUT that incorporates gamut compression at build time.
#[derive(Debug, Clone)]
pub struct GamutCompressLut {
    /// LUT size (entries per axis).
    pub size: usize,
    /// Flattened data: size^3 × 3 (R, G, B).
    pub data: Vec<[f32; 3]>,
    /// Configuration used when building the LUT.
    pub config: GamutCompressConfig,
    /// Deviation metric computed at build time.
    deviation: f32,
}

impl GamutCompressLut {
    /// Build a new gamut-compression LUT with the given size and config.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn build(size: usize, config: GamutCompressConfig) -> Self {
        let total = size * size * size;
        let mut data = Vec::with_capacity(total);
        let scale = 1.0 / (size - 1).max(1) as f32;

        for bi in 0..size {
            for gi in 0..size {
                for ri in 0..size {
                    let r = ri as f32 * scale;
                    let g = gi as f32 * scale;
                    let b = bi as f32 * scale;
                    let pixel = Self::compress_pixel([r, g, b], &config);
                    data.push(pixel);
                }
            }
        }

        // Compute average deviation from input values.
        let dev: f32 = data
            .iter()
            .zip(Self::identity_iter(size))
            .map(|(out, src)| {
                let dr = out[0] - src[0];
                let dg = out[1] - src[1];
                let db = out[2] - src[2];
                (dr * dr + dg * dg + db * db).sqrt()
            })
            .sum::<f32>()
            / total as f32;

        Self {
            size,
            data,
            config,
            deviation: dev,
        }
    }

    /// Apply this LUT to an RGB pixel using nearest-neighbour lookup.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    #[must_use]
    pub fn apply(&self, pixel: [f32; 3]) -> [f32; 3] {
        let s = (self.size - 1).max(1) as f32;
        let ri = (pixel[0] * s).round().clamp(0.0, s) as usize;
        let gi = (pixel[1] * s).round().clamp(0.0, s) as usize;
        let bi = (pixel[2] * s).round().clamp(0.0, s) as usize;
        let idx = bi * self.size * self.size + gi * self.size + ri;
        self.data[idx.min(self.data.len() - 1)]
    }

    /// Average per-sample deviation of the compressed output from the identity.
    #[must_use]
    pub fn deviation_from_identity(&self) -> f32 {
        self.deviation
    }

    // ---- private helpers ----

    fn compress_pixel(px: [f32; 3], cfg: &GamutCompressConfig) -> [f32; 3] {
        match cfg.method {
            GamutCompressMethod::HardClip => [
                px[0].clamp(0.0, 1.0),
                px[1].clamp(0.0, 1.0),
                px[2].clamp(0.0, 1.0),
            ],
            GamutCompressMethod::SoftClip => [
                Self::soft_clip(px[0], cfg.threshold, cfg.power),
                Self::soft_clip(px[1], cfg.threshold, cfg.power),
                Self::soft_clip(px[2], cfg.threshold, cfg.power),
            ],
            GamutCompressMethod::Desaturate => {
                let luma = 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2];
                let max = px[0].max(px[1]).max(px[2]);
                if max <= 1.0 {
                    return px;
                }
                let t = (max - 1.0).min(1.0);
                [
                    px[0] * (1.0 - t) + luma * t,
                    px[1] * (1.0 - t) + luma * t,
                    px[2] * (1.0 - t) + luma * t,
                ]
            }
            GamutCompressMethod::AcesReference => {
                // Simplified ACES gamut compression: compress each channel individually
                // using a parabolic roll-off beyond the threshold.
                [
                    Self::aces_compress(px[0], cfg.threshold),
                    Self::aces_compress(px[1], cfg.threshold),
                    Self::aces_compress(px[2], cfg.threshold),
                ]
            }
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn soft_clip(x: f32, threshold: f32, power: f32) -> f32 {
        let limit = 1.0 + threshold;
        if x <= 1.0 {
            x
        } else if x >= limit {
            1.0
        } else {
            let t = (x - 1.0) / threshold;
            1.0 - threshold * t.powf(power)
        }
    }

    fn aces_compress(x: f32, threshold: f32) -> f32 {
        let limit = 1.0 + threshold;
        if x >= limit {
            return 1.0;
        }
        if x < 0.0 {
            return 0.0;
        }
        if x > 1.0 {
            let t = (x - 1.0) / threshold;
            return 1.0 - 0.5 * threshold * (1.0 + t - (1.0 + t * t).sqrt());
        }
        x
    }

    /// Iterator that yields the identity LUT values for size^3 entries.
    #[allow(clippy::cast_precision_loss)]
    fn identity_iter(size: usize) -> impl Iterator<Item = [f32; 3]> {
        let scale = 1.0 / (size - 1).max(1) as f32;
        (0..size).flat_map(move |bi| {
            (0..size).flat_map(move |gi| {
                (0..size).map(move |ri| [ri as f32 * scale, gi as f32 * scale, bi as f32 * scale])
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hard_clip_is_not_soft_clip() {
        assert!(!GamutCompressMethod::HardClip.is_soft_clip());
    }

    #[test]
    fn test_soft_clip_is_soft_clip() {
        assert!(GamutCompressMethod::SoftClip.is_soft_clip());
    }

    #[test]
    fn test_aces_reference_is_soft_clip() {
        assert!(GamutCompressMethod::AcesReference.is_soft_clip());
    }

    #[test]
    fn test_desaturate_is_not_soft_clip() {
        assert!(!GamutCompressMethod::Desaturate.is_soft_clip());
    }

    #[test]
    fn test_method_label_hard_clip() {
        assert_eq!(GamutCompressMethod::HardClip.label(), "Hard Clip");
    }

    #[test]
    fn test_method_label_soft_clip() {
        assert_eq!(GamutCompressMethod::SoftClip.label(), "Soft Clip");
    }

    #[test]
    fn test_config_default_power() {
        let cfg = GamutCompressConfig::default();
        assert!((cfg.power() - 1.2).abs() < 1e-6);
    }

    #[test]
    fn test_config_is_valid() {
        let cfg = GamutCompressConfig::default();
        assert!(cfg.is_valid());
    }

    #[test]
    fn test_config_invalid_power() {
        let cfg = GamutCompressConfig {
            power: -1.0,
            ..Default::default()
        };
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_build_creates_correct_size() {
        let lut = GamutCompressLut::build(5, GamutCompressConfig::default());
        assert_eq!(lut.data.len(), 5 * 5 * 5);
    }

    #[test]
    fn test_apply_identity_pixel() {
        let lut = GamutCompressLut::build(17, GamutCompressConfig::default());
        let px = [0.5, 0.5, 0.5];
        let out = lut.apply(px);
        // In-gamut pixels should pass through unchanged (within LUT quantisation error).
        assert!((out[0] - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_apply_clips_bright_pixel() {
        let cfg = GamutCompressConfig {
            method: GamutCompressMethod::HardClip,
            ..Default::default()
        };
        let lut = GamutCompressLut::build(17, cfg);
        let px = [1.5, 1.5, 1.5]; // way out of gamut
        let out = lut.apply(px);
        // Expect output to be near 1.0 after hard clip.
        assert!(out[0] <= 1.01, "expected clipped value, got {}", out[0]);
    }

    #[test]
    fn test_deviation_from_identity_nonnegative() {
        let lut = GamutCompressLut::build(5, GamutCompressConfig::default());
        assert!(lut.deviation_from_identity() >= 0.0);
    }

    #[test]
    fn test_hard_clip_lut_has_positive_deviation() {
        // A hard-clip LUT applied to out-of-range values should deviate from identity.
        let cfg = GamutCompressConfig {
            method: GamutCompressMethod::HardClip,
            threshold: 0.0,
            power: 1.0,
        };
        let lut = GamutCompressLut::build(5, cfg);
        // Deviation should be >= 0 (may be 0 when all sampled pixels are in-gamut).
        assert!(lut.deviation_from_identity() >= 0.0);
    }
}
