//! Colour restoration for aged or degraded video.
//!
//! This module provides tools for diagnosing and correcting colour fading,
//! colour cast, and saturation loss commonly found in archival footage.

#![allow(dead_code)]

/// Type of colour fade observed in the source material.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FadeType {
    /// Overall brightness reduction across all channels.
    GlobalFade,
    /// Colour cast shifted toward red (common in old print film).
    RedShift,
    /// Colour cast shifted toward cyan.
    CyanShift,
    /// Saturation loss with colours trending toward grey.
    SaturationLoss,
    /// Yellowing of whites and highlights.
    Yellowing,
    /// Uneven fading with more damage in shadows.
    ShadowFade,
}

/// RGB triplet in normalised `[0.0, 1.0]` space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rgb {
    /// Red channel.
    pub r: f32,
    /// Green channel.
    pub g: f32,
    /// Blue channel.
    pub b: f32,
}

impl Rgb {
    /// Construct a new `Rgb` value, clamping each channel to `[0.0, 1.0]`.
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self {
            r: r.clamp(0.0, 1.0),
            g: g.clamp(0.0, 1.0),
            b: b.clamp(0.0, 1.0),
        }
    }

    /// Compute perceived luminance using the BT.709 coefficients.
    pub fn luma(&self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }
}

/// Configuration for the colour restoration pass.
#[derive(Debug, Clone)]
pub struct ColorRestoreConfig {
    /// The type of fade to correct.
    pub fade_type: FadeType,
    /// Correction strength in the range `[0.0, 1.0]`.
    pub strength: f32,
    /// Target white point (reference neutral).
    pub white_point: Rgb,
    /// Whether to apply gamma correction after the colour fix.
    pub apply_gamma: bool,
    /// Gamma exponent (typical value: 2.2 for sRGB).
    pub gamma: f32,
}

impl Default for ColorRestoreConfig {
    fn default() -> Self {
        Self {
            fade_type: FadeType::GlobalFade,
            strength: 0.5,
            white_point: Rgb::new(1.0, 1.0, 1.0),
            apply_gamma: false,
            gamma: 2.2,
        }
    }
}

/// Corrects colour fade and cast in a sequence of frames.
#[derive(Debug)]
pub struct ColorRestorer {
    config: ColorRestoreConfig,
}

impl ColorRestorer {
    /// Create a new `ColorRestorer` with default settings.
    pub fn new() -> Self {
        Self {
            config: ColorRestoreConfig::default(),
        }
    }

    /// Create a new `ColorRestorer` with custom configuration.
    pub fn with_config(config: ColorRestoreConfig) -> Self {
        Self { config }
    }

    /// Restore a single pixel.
    ///
    /// Applies the configured correction and clamps the result to `[0.0, 1.0]`.
    pub fn restore_pixel(&self, pixel: Rgb) -> Rgb {
        let s = self.config.strength;
        let corrected = match self.config.fade_type {
            FadeType::GlobalFade => Rgb::new(
                pixel.r + s * (1.0 - pixel.r),
                pixel.g + s * (1.0 - pixel.g),
                pixel.b + s * (1.0 - pixel.b),
            ),
            FadeType::RedShift => Rgb::new(
                pixel.r - s * pixel.r * 0.3,
                pixel.g + s * 0.05,
                pixel.b + s * 0.1,
            ),
            FadeType::CyanShift => {
                Rgb::new(pixel.r + s * 0.15, pixel.g - s * 0.05, pixel.b - s * 0.1)
            }
            FadeType::SaturationLoss => {
                let luma = pixel.luma();
                Rgb::new(
                    luma + (pixel.r - luma) * (1.0 + s),
                    luma + (pixel.g - luma) * (1.0 + s),
                    luma + (pixel.b - luma) * (1.0 + s),
                )
            }
            FadeType::Yellowing => Rgb::new(pixel.r, pixel.g, pixel.b + s * (1.0 - pixel.b) * 0.5),
            FadeType::ShadowFade => {
                // Lift shadows proportional to (1 - luma)
                let lift = s * (1.0 - pixel.luma()) * 0.2;
                Rgb::new(pixel.r + lift, pixel.g + lift, pixel.b + lift)
            }
        };

        if self.config.apply_gamma {
            self.apply_gamma(corrected)
        } else {
            corrected
        }
    }

    /// Apply gamma correction to a pixel.
    #[allow(clippy::cast_precision_loss)]
    fn apply_gamma(&self, pixel: Rgb) -> Rgb {
        let exp = 1.0 / self.config.gamma;
        Rgb::new(pixel.r.powf(exp), pixel.g.powf(exp), pixel.b.powf(exp))
    }

    /// Process an entire frame stored as a flat slice of `Rgb` pixels.
    pub fn restore_frame(&self, pixels: &[Rgb]) -> Vec<Rgb> {
        pixels.iter().map(|&p| self.restore_pixel(p)).collect()
    }

    /// Analyse a frame and return the dominant fade type detected.
    ///
    /// This is a simple heuristic based on per-channel mean values.
    pub fn detect_fade(pixels: &[Rgb]) -> Option<FadeType> {
        if pixels.is_empty() {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        let n = pixels.len() as f32;
        let mean_r: f32 = pixels.iter().map(|p| p.r).sum::<f32>() / n;
        let mean_g: f32 = pixels.iter().map(|p| p.g).sum::<f32>() / n;
        let mean_b: f32 = pixels.iter().map(|p| p.b).sum::<f32>() / n;
        let mean_all = (mean_r + mean_g + mean_b) / 3.0;

        if mean_r > mean_g + 0.1 && mean_r > mean_b + 0.1 {
            return Some(FadeType::RedShift);
        }
        if mean_b < mean_g - 0.1 && mean_b < mean_r - 0.1 {
            return Some(FadeType::Yellowing);
        }
        if mean_all < 0.4 {
            return Some(FadeType::ShadowFade);
        }
        if mean_all < 0.6 {
            return Some(FadeType::GlobalFade);
        }
        Some(FadeType::SaturationLoss)
    }

    /// Current configuration.
    pub fn config(&self) -> &ColorRestoreConfig {
        &self.config
    }
}

impl Default for ColorRestorer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grey(v: f32) -> Rgb {
        Rgb::new(v, v, v)
    }

    #[test]
    fn test_global_fade_brightens() {
        let r = ColorRestorer::new(); // strength=0.5, GlobalFade
        let p = grey(0.4);
        let out = r.restore_pixel(p);
        assert!(
            out.r > p.r,
            "GlobalFade should brighten: {} -> {}",
            p.r,
            out.r
        );
    }

    #[test]
    fn test_output_clamped() {
        let r = ColorRestorer::with_config(ColorRestoreConfig {
            fade_type: FadeType::GlobalFade,
            strength: 1.0,
            ..Default::default()
        });
        let p = Rgb::new(0.99, 0.99, 0.99);
        let out = r.restore_pixel(p);
        assert!(out.r <= 1.0 && out.g <= 1.0 && out.b <= 1.0);
    }

    #[test]
    fn test_red_shift_reduces_red() {
        let r = ColorRestorer::with_config(ColorRestoreConfig {
            fade_type: FadeType::RedShift,
            strength: 0.5,
            ..Default::default()
        });
        let p = Rgb::new(0.9, 0.5, 0.5);
        let out = r.restore_pixel(p);
        assert!(out.r < p.r, "RedShift should reduce red");
    }

    #[test]
    fn test_saturation_loss_increases_saturation() {
        let r = ColorRestorer::with_config(ColorRestoreConfig {
            fade_type: FadeType::SaturationLoss,
            strength: 0.5,
            ..Default::default()
        });
        // Slightly de-saturated pixel
        let p = Rgb::new(0.6, 0.5, 0.45);
        let out = r.restore_pixel(p);
        // The red channel should move further from grey
        let luma = p.luma();
        assert!((out.r - luma).abs() > (p.r - luma).abs());
    }

    #[test]
    fn test_yellowing_raises_blue() {
        let r = ColorRestorer::with_config(ColorRestoreConfig {
            fade_type: FadeType::Yellowing,
            strength: 0.8,
            ..Default::default()
        });
        let p = Rgb::new(0.8, 0.8, 0.4);
        let out = r.restore_pixel(p);
        assert!(out.b > p.b, "Yellowing fix should raise blue");
    }

    #[test]
    fn test_shadow_fade_lifts_shadows() {
        let r = ColorRestorer::with_config(ColorRestoreConfig {
            fade_type: FadeType::ShadowFade,
            strength: 1.0,
            ..Default::default()
        });
        let p = grey(0.1); // dark pixel
        let out = r.restore_pixel(p);
        assert!(out.r > p.r, "ShadowFade should lift dark pixels");
    }

    #[test]
    fn test_restore_frame_length() {
        let r = ColorRestorer::new();
        let pixels: Vec<Rgb> = (0..100).map(|_| grey(0.5)).collect();
        let out = r.restore_frame(&pixels);
        assert_eq!(out.len(), 100);
    }

    #[test]
    fn test_detect_fade_red_shift() {
        let pixels: Vec<Rgb> = vec![Rgb::new(0.9, 0.5, 0.5); 50];
        let fade = ColorRestorer::detect_fade(&pixels);
        assert_eq!(fade, Some(FadeType::RedShift));
    }

    #[test]
    fn test_detect_fade_empty_returns_none() {
        let fade = ColorRestorer::detect_fade(&[]);
        assert_eq!(fade, None);
    }

    #[test]
    fn test_gamma_correction_applied() {
        let r = ColorRestorer::with_config(ColorRestoreConfig {
            fade_type: FadeType::GlobalFade,
            strength: 0.0,
            apply_gamma: true,
            gamma: 2.2,
            ..Default::default()
        });
        // strength=0 means no colour change, but gamma is still applied
        let p = Rgb::new(0.5, 0.5, 0.5);
        let out = r.restore_pixel(p);
        let expected = 0.5_f32.powf(1.0 / 2.2);
        assert!(
            (out.r - expected).abs() < 1e-5,
            "gamma not applied correctly"
        );
    }

    #[test]
    fn test_rgb_new_clamps() {
        let p = Rgb::new(2.0, -1.0, 0.5);
        assert_eq!(p.r, 1.0);
        assert_eq!(p.g, 0.0);
        assert_eq!(p.b, 0.5);
    }

    #[test]
    fn test_luma_grey() {
        let p = grey(0.5);
        let luma = p.luma();
        assert!((luma - 0.5).abs() < 1e-5, "luma of grey should be 0.5");
    }

    #[test]
    fn test_cyan_shift_raises_red() {
        let r = ColorRestorer::with_config(ColorRestoreConfig {
            fade_type: FadeType::CyanShift,
            strength: 0.5,
            ..Default::default()
        });
        let p = Rgb::new(0.4, 0.6, 0.7);
        let out = r.restore_pixel(p);
        assert!(out.r > p.r, "CyanShift should raise red channel");
    }

    #[test]
    fn test_zero_strength_no_change() {
        let r = ColorRestorer::with_config(ColorRestoreConfig {
            fade_type: FadeType::GlobalFade,
            strength: 0.0,
            apply_gamma: false,
            ..Default::default()
        });
        let p = Rgb::new(0.4, 0.5, 0.6);
        let out = r.restore_pixel(p);
        assert!((out.r - p.r).abs() < 1e-6);
        assert!((out.g - p.g).abs() < 1e-6);
        assert!((out.b - p.b).abs() < 1e-6);
    }
}
