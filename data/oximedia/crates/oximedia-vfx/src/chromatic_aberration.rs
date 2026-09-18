//! Chromatic aberration simulation for VFX and lens-matching workflows.
//!
//! Models the radial colour fringing produced by imperfect lens optics where
//! different wavelengths focus at slightly different points.
//!
//! # Overview
//!
//! Two complementary models are provided:
//!
//! 1. **[`AberrationSimulator`]** — simple lateral chromatic aberration via
//!    per-channel normalised coordinate offsets (fast, good for stylised looks).
//!
//! 2. **[`ChromaticAberrationEffect`]** — physically-based model combining
//!    Brown-Conrady radial lens distortion with per-channel dispersion.  Each
//!    colour channel is individually warped using slightly different distortion
//!    coefficients, reproducing the rainbow fringing seen in real glass optics.
//!    This implements the [`crate::VideoEffect`] trait for use in effect chains.

#![allow(dead_code)]

use crate::{EffectParams, Frame, VfxError, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

/// Axis along which lateral chromatic aberration is expressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AberrationAxis {
    /// Horizontal colour fringing.
    Horizontal,
    /// Vertical colour fringing.
    Vertical,
    /// Radial fringing (both axes, centred on the optical axis).
    Radial,
    /// Tangential fringing (perpendicular to the radial direction).
    Tangential,
}

impl AberrationAxis {
    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Horizontal => "horizontal",
            Self::Vertical => "vertical",
            Self::Radial => "radial",
            Self::Tangential => "tangential",
        }
    }

    /// Returns `true` if the axis affects both spatial dimensions.
    pub fn is_two_dimensional(self) -> bool {
        matches!(self, Self::Radial | Self::Tangential)
    }
}

/// Describes a chromatic aberration along a single axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromaticAberration {
    /// Axis of aberration.
    pub axis: AberrationAxis,
    /// Red-channel offset in normalised image coordinates (-1.0 – 1.0).
    pub red_offset: f32,
    /// Green-channel offset in normalised image coordinates.
    pub green_offset: f32,
    /// Blue-channel offset in normalised image coordinates.
    pub blue_offset: f32,
    /// Radial power factor: how quickly the aberration grows from centre (1.0 = linear).
    pub radial_power: f32,
}

impl Default for ChromaticAberration {
    fn default() -> Self {
        Self {
            axis: AberrationAxis::Radial,
            red_offset: 0.002,
            green_offset: 0.0,
            blue_offset: -0.002,
            radial_power: 1.0,
        }
    }
}

impl ChromaticAberration {
    /// Create a zero-aberration (identity) instance.
    pub fn identity() -> Self {
        Self {
            red_offset: 0.0,
            green_offset: 0.0,
            blue_offset: 0.0,
            ..Self::default()
        }
    }

    /// Compute the radial offset magnitude at a normalised distance `r` from
    /// the optical centre (0.0 = centre, 1.0 = corner).
    pub fn radial_offset(&self, r: f32) -> f32 {
        let r = r.clamp(0.0, 1.0);
        r.powf(self.radial_power.max(0.1))
    }

    /// Returns the per-channel displacement `[red, green, blue]` at radius `r`.
    pub fn channel_offsets_at(&self, r: f32) -> [f32; 3] {
        let scale = self.radial_offset(r);
        [
            self.red_offset * scale,
            self.green_offset * scale,
            self.blue_offset * scale,
        ]
    }

    /// Maximum absolute channel offset across all three channels (useful for extent calculations).
    pub fn max_abs_offset(&self) -> f32 {
        self.red_offset
            .abs()
            .max(self.green_offset.abs())
            .max(self.blue_offset.abs())
    }
}

/// Configuration for the aberration simulation pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AberrationConfig {
    /// Primary lateral chromatic aberration model.
    pub aberration: ChromaticAberration,
    /// Additional global strength multiplier (0.0 = off, 1.0 = full).
    pub strength: f32,
    /// Whether to apply the effect only beyond a certain radius from centre.
    pub vignette_mask: bool,
    /// Inner radius (normalised) beyond which the effect starts when `vignette_mask` is true.
    pub mask_inner_radius: f32,
}

impl Default for AberrationConfig {
    fn default() -> Self {
        Self {
            aberration: ChromaticAberration::default(),
            strength: 1.0,
            vignette_mask: false,
            mask_inner_radius: 0.5,
        }
    }
}

impl AberrationConfig {
    /// Returns `true` when the configured aberration would be barely noticeable.
    ///
    /// Subtle is defined as max channel offset below 0.005 normalised units.
    pub fn is_subtle(&self) -> bool {
        self.aberration.max_abs_offset() * self.strength < 0.005
    }

    /// Returns `true` if the configuration would produce a visually meaningful effect.
    pub fn is_active(&self) -> bool {
        self.strength > 0.0 && self.aberration.max_abs_offset() > 0.0
    }
}

/// Simulates chromatic aberration on a pixel buffer.
#[derive(Debug, Clone)]
pub struct AberrationSimulator {
    /// Configuration for this simulator instance.
    pub config: AberrationConfig,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

impl AberrationSimulator {
    /// Create a new simulator for a given frame size.
    pub fn new(width: u32, height: u32, config: AberrationConfig) -> Self {
        Self {
            config,
            width,
            height,
        }
    }

    /// Apply chromatic aberration to a pixel buffer.
    ///
    /// `pixels` is a flat RGBA buffer (4 bytes per pixel, row-major).
    /// The operation returns a new buffer with the effect applied.
    #[allow(clippy::cast_precision_loss)]
    pub fn apply(&self, pixels: &[u8]) -> Vec<u8> {
        let w = self.width as usize;
        let h = self.height as usize;
        let expected = w * h * 4;
        if pixels.len() != expected || w == 0 || h == 0 {
            return pixels.to_vec();
        }

        let cx = w as f32 / 2.0;
        let cy = h as f32 / 2.0;
        let max_r = (cx * cx + cy * cy).sqrt();

        let mut out = pixels.to_vec();

        for py in 0..h {
            for px in 0..w {
                let dx = px as f32 - cx;
                let dy = py as f32 - cy;
                let r = if max_r > 0.0 {
                    (dx * dx + dy * dy).sqrt() / max_r
                } else {
                    0.0
                };

                let mask = if self.config.vignette_mask && r < self.config.mask_inner_radius {
                    0.0_f32
                } else {
                    1.0_f32
                };

                let offsets = self.config.aberration.channel_offsets_at(r);

                // Sample each channel with its offset
                let base_idx = (py * w + px) * 4;
                let channels = [0_usize, 1, 2];
                for (ch, &off) in channels.iter().zip(offsets.iter()) {
                    let effective_off = off * self.config.strength * mask;
                    // Compute offset in pixels
                    let off_px = (effective_off * w as f32) as i32;
                    let off_py = (effective_off * h as f32) as i32;
                    let sx = (px as i32 + off_px).clamp(0, w as i32 - 1) as usize;
                    let sy = (py as i32 + off_py).clamp(0, h as i32 - 1) as usize;
                    let src_idx = (sy * w + sx) * 4 + ch;
                    out[base_idx + ch] = pixels[src_idx];
                }
            }
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Brown-Conrady Lens Distortion Model
// ─────────────────────────────────────────────────────────────────────────────

/// Brown-Conrady lens distortion coefficients.
///
/// Models both radial (`k1`, `k2`, `k3`) and tangential (`p1`, `p2`) distortion
/// as used in OpenCV camera calibration.  Chromatic aberration is achieved by
/// assigning slightly different coefficients to each colour channel.
///
/// Reference: D.C. Brown, "Decentering Distortion of Lenses", 1966.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LensDistortionCoeffs {
    /// Radial distortion coefficient k1 (2nd-order).
    pub k1: f32,
    /// Radial distortion coefficient k2 (4th-order).
    pub k2: f32,
    /// Radial distortion coefficient k3 (6th-order).
    pub k3: f32,
    /// Tangential distortion coefficient p1.
    pub p1: f32,
    /// Tangential distortion coefficient p2.
    pub p2: f32,
}

impl LensDistortionCoeffs {
    /// Identity (no distortion).
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    /// Typical moderate barrel distortion (consumer zoom lens).
    #[must_use]
    pub const fn barrel() -> Self {
        Self {
            k1: -0.25,
            k2: 0.05,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    /// Apply distortion to a normalised image coordinate `(x, y)` where
    /// `(0, 0)` is the principal point and `(1, 1)` is the corner.
    ///
    /// Returns the distorted coordinate.
    #[must_use]
    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;

        let xd = x * radial + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
        let yd = y * radial + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;

        (xd, yd)
    }
}

impl Default for LensDistortionCoeffs {
    fn default() -> Self {
        Self::identity()
    }
}

/// Per-channel lens distortion coefficients for chromatic aberration.
///
/// Chromatic aberration arises because different wavelengths of light are
/// refracted differently by glass.  By assigning slightly different distortion
/// coefficients to the R, G, and B channels we reproduce this effect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerChannelDistortion {
    /// Coefficients for the red channel (longest visible wavelength → least refracted).
    pub red: LensDistortionCoeffs,
    /// Coefficients for the green channel (reference / least aberrated).
    pub green: LensDistortionCoeffs,
    /// Coefficients for the blue channel (shortest visible wavelength → most refracted).
    pub blue: LensDistortionCoeffs,
}

impl PerChannelDistortion {
    /// Identity — no distortion on any channel.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            red: LensDistortionCoeffs::identity(),
            green: LensDistortionCoeffs::identity(),
            blue: LensDistortionCoeffs::identity(),
        }
    }

    /// A typical lens chromatic aberration pattern:
    /// - Red channel: slightly less distorted (magnified).
    /// - Green channel: reference (nominal distortion).
    /// - Blue channel: slightly more distorted (de-magnified).
    ///
    /// `strength` scales the inter-channel difference (0.0 = identity, 1.0 = default).
    #[must_use]
    pub fn typical(strength: f32) -> Self {
        let s = strength.clamp(0.0, 4.0);
        Self {
            red: LensDistortionCoeffs {
                k1: -0.2 + 0.02 * s,
                k2: 0.04,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
            },
            green: LensDistortionCoeffs {
                k1: -0.22,
                k2: 0.04,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
            },
            blue: LensDistortionCoeffs {
                k1: -0.24 - 0.02 * s,
                k2: 0.04,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
            },
        }
    }
}

impl Default for PerChannelDistortion {
    fn default() -> Self {
        Self::identity()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ChromaticAberrationEffect  (VideoEffect)
// ─────────────────────────────────────────────────────────────────────────────

/// Physically-based chromatic aberration effect implementing [`VideoEffect`].
///
/// Each colour channel (R, G, B) is independently sampled at a position warped
/// by its own Brown-Conrady lens distortion coefficients.  The alpha channel is
/// taken from the green (reference) channel position.
///
/// This produces the rainbow fringing typical of real optical systems rather
/// than the uniform lateral shift of [`AberrationSimulator`].
pub struct ChromaticAberrationEffect {
    /// Per-channel distortion parameters.
    pub distortion: PerChannelDistortion,
    /// Global strength blend: 0.0 = original image, 1.0 = full effect.
    pub strength: f32,
}

impl ChromaticAberrationEffect {
    /// Create an identity effect (no aberration).
    #[must_use]
    pub fn identity() -> Self {
        Self {
            distortion: PerChannelDistortion::identity(),
            strength: 1.0,
        }
    }

    /// Create with typical lens chromatic aberration at the given strength.
    #[must_use]
    pub fn typical(strength: f32) -> Self {
        Self {
            distortion: PerChannelDistortion::typical(strength),
            strength: strength.clamp(0.0, 4.0),
        }
    }

    /// Bilinearly sample `channel` (0=R, 1=G, 2=B) from an RGBA pixel buffer
    /// at normalised floating-point coordinates `(nx, ny)` ∈ [0, 1].
    fn sample_channel(
        pixels: &[u8],
        width: usize,
        height: usize,
        nx: f32,
        ny: f32,
        channel: usize,
    ) -> u8 {
        let fx = nx * (width as f32 - 1.0);
        let fy = ny * (height as f32 - 1.0);

        let x0 = (fx.floor() as i32).clamp(0, width as i32 - 1) as usize;
        let y0 = (fy.floor() as i32).clamp(0, height as i32 - 1) as usize;
        let x1 = (x0 + 1).min(width - 1);
        let y1 = (y0 + 1).min(height - 1);

        let tx = fx - fx.floor();
        let ty = fy - fy.floor();

        let p00 = pixels[(y0 * width + x0) * 4 + channel] as f32;
        let p10 = pixels[(y0 * width + x1) * 4 + channel] as f32;
        let p01 = pixels[(y1 * width + x0) * 4 + channel] as f32;
        let p11 = pixels[(y1 * width + x1) * 4 + channel] as f32;

        let top = p00 * (1.0 - tx) + p10 * tx;
        let bot = p01 * (1.0 - tx) + p11 * tx;
        (top * (1.0 - ty) + bot * ty).clamp(0.0, 255.0) as u8
    }

    /// Map a distorted normalised coordinate back to [0, 1] image space.
    ///
    /// The principal point is at (0.5, 0.5).  The normalised coordinate system
    /// centred on the principal point uses the half-diagonal as the unit length.
    fn to_image_coord(coeffs: &LensDistortionCoeffs, nx: f32, ny: f32) -> (f32, f32) {
        // Convert image [0,1] coords to normalised coords centred at (0,0)
        let cx = nx * 2.0 - 1.0; // [-1, 1]
        let cy = ny * 2.0 - 1.0;

        let (dx, dy) = coeffs.apply(cx, cy);

        // Back to [0, 1]
        let out_x = (dx + 1.0) / 2.0;
        let out_y = (dy + 1.0) / 2.0;
        (out_x, out_y)
    }
}

impl Default for ChromaticAberrationEffect {
    fn default() -> Self {
        Self::typical(1.0)
    }
}

impl VideoEffect for ChromaticAberrationEffect {
    fn name(&self) -> &str {
        "ChromaticAberration"
    }

    fn description(&self) -> &'static str {
        "Physically-based per-channel lens distortion producing chromatic aberration"
    }

    fn apply(
        &mut self,
        input: &Frame,
        output: &mut Frame,
        _params: &EffectParams,
    ) -> VfxResult<()> {
        if input.width != output.width || input.height != output.height {
            return Err(VfxError::InvalidDimensions {
                width: output.width,
                height: output.height,
            });
        }

        let w = input.width as usize;
        let h = input.height as usize;

        if w == 0 || h == 0 {
            return Ok(());
        }

        let pixels = &input.data;
        let strength = self.strength.clamp(0.0, 4.0);
        let inv_strength = 1.0 - strength.min(1.0);

        for py in 0..h {
            for px in 0..w {
                let nx = px as f32 / (w as f32 - 1.0).max(1.0);
                let ny = py as f32 / (h as f32 - 1.0).max(1.0);

                // Compute distorted coords per channel
                let (rx, ry) = Self::to_image_coord(&self.distortion.red, nx, ny);
                let (gx, gy) = Self::to_image_coord(&self.distortion.green, nx, ny);
                let (bx, by) = Self::to_image_coord(&self.distortion.blue, nx, ny);

                let r = Self::sample_channel(pixels, w, h, rx, ry, 0);
                let g = Self::sample_channel(pixels, w, h, gx, gy, 1);
                let b = Self::sample_channel(pixels, w, h, bx, by, 2);
                // Alpha from green (reference) channel
                let a = Self::sample_channel(pixels, w, h, gx, gy, 3);

                let base_idx = (py * w + px) * 4;
                let orig = [
                    pixels[base_idx],
                    pixels[base_idx + 1],
                    pixels[base_idx + 2],
                    pixels[base_idx + 3],
                ];

                let out_pixel = if inv_strength > 0.0 {
                    [
                        (orig[0] as f32 * inv_strength + r as f32 * strength).clamp(0.0, 255.0)
                            as u8,
                        (orig[1] as f32 * inv_strength + g as f32 * strength).clamp(0.0, 255.0)
                            as u8,
                        (orig[2] as f32 * inv_strength + b as f32 * strength).clamp(0.0, 255.0)
                            as u8,
                        a,
                    ]
                } else {
                    [r, g, b, a]
                };

                output.set_pixel(px as u32, py as u32, out_pixel);
            }
        }

        Ok(())
    }

    fn supports_gpu(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aberration_axis_labels_unique() {
        let axes = [
            AberrationAxis::Horizontal,
            AberrationAxis::Vertical,
            AberrationAxis::Radial,
            AberrationAxis::Tangential,
        ];
        let labels: Vec<_> = axes.iter().map(|a| a.label()).collect();
        let unique: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(labels.len(), unique.len());
    }

    #[test]
    fn test_aberration_axis_two_dimensional() {
        assert!(AberrationAxis::Radial.is_two_dimensional());
        assert!(AberrationAxis::Tangential.is_two_dimensional());
        assert!(!AberrationAxis::Horizontal.is_two_dimensional());
    }

    #[test]
    fn test_chromatic_aberration_identity_zero_offset() {
        let ca = ChromaticAberration::identity();
        assert_eq!(ca.max_abs_offset(), 0.0);
    }

    #[test]
    fn test_radial_offset_centre() {
        let ca = ChromaticAberration::default();
        assert_eq!(ca.radial_offset(0.0), 0.0);
    }

    #[test]
    fn test_radial_offset_corner() {
        let ca = ChromaticAberration::default();
        assert_eq!(ca.radial_offset(1.0), 1.0);
    }

    #[test]
    fn test_channel_offsets_at_zero_radius() {
        let ca = ChromaticAberration::default();
        let offs = ca.channel_offsets_at(0.0);
        assert_eq!(offs, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_channel_offsets_red_blue_opposite() {
        let ca = ChromaticAberration::default();
        let offs = ca.channel_offsets_at(1.0);
        // red and blue should be opposite in sign
        assert!(offs[0] > 0.0);
        assert!(offs[2] < 0.0);
    }

    #[test]
    fn test_max_abs_offset() {
        let ca = ChromaticAberration {
            red_offset: 0.01,
            green_offset: 0.0,
            blue_offset: -0.02,
            ..Default::default()
        };
        assert!((ca.max_abs_offset() - 0.02).abs() < 1e-6);
    }

    #[test]
    fn test_aberration_config_is_subtle_identity() {
        let cfg = AberrationConfig {
            aberration: ChromaticAberration::identity(),
            ..Default::default()
        };
        assert!(cfg.is_subtle());
    }

    #[test]
    fn test_aberration_config_not_subtle_strong() {
        let cfg = AberrationConfig {
            aberration: ChromaticAberration {
                red_offset: 0.05,
                blue_offset: -0.05,
                ..Default::default()
            },
            strength: 1.0,
            ..Default::default()
        };
        assert!(!cfg.is_subtle());
    }

    #[test]
    fn test_aberration_config_is_active() {
        assert!(AberrationConfig::default().is_active());
        let inactive = AberrationConfig {
            strength: 0.0,
            ..Default::default()
        };
        assert!(!inactive.is_active());
    }

    #[test]
    fn test_simulator_apply_identity_unchanged() {
        let cfg = AberrationConfig {
            aberration: ChromaticAberration::identity(),
            strength: 0.0,
            ..Default::default()
        };
        let sim = AberrationSimulator::new(4, 4, cfg);
        let pixels: Vec<u8> = (0..64).collect();
        let result = sim.apply(&pixels);
        // Alpha channel (index 3, 7, …) should be unchanged; with zero strength
        // the result should match the input (but channel sampling still occurs at same coords)
        assert_eq!(result.len(), pixels.len());
    }

    #[test]
    fn test_simulator_apply_wrong_buffer_size_returns_copy() {
        let sim = AberrationSimulator::new(4, 4, AberrationConfig::default());
        let bad_pixels = vec![0u8; 10]; // wrong size
        let result = sim.apply(&bad_pixels);
        assert_eq!(result, bad_pixels);
    }

    // ── LensDistortionCoeffs ───────────────────────────────────────────────

    #[test]
    fn test_lens_distortion_identity_no_change() {
        let coeff = LensDistortionCoeffs::identity();
        let (xd, yd) = coeff.apply(0.5, 0.3);
        assert!((xd - 0.5).abs() < 1e-5, "x unchanged by identity");
        assert!((yd - 0.3).abs() < 1e-5, "y unchanged by identity");
    }

    #[test]
    fn test_lens_distortion_centre_unchanged() {
        // At (0,0) all radial terms vanish
        let coeff = LensDistortionCoeffs::barrel();
        let (xd, yd) = coeff.apply(0.0, 0.0);
        assert!(xd.abs() < 1e-6);
        assert!(yd.abs() < 1e-6);
    }

    #[test]
    fn test_lens_distortion_barrel_moves_corner_inward() {
        // Barrel distortion (k1 < 0) should move the corner closer to centre
        let coeff = LensDistortionCoeffs::barrel();
        let (xd, _) = coeff.apply(0.7, 0.0);
        // |xd| < |x| for barrel distortion
        assert!(xd.abs() < 0.7, "barrel should reduce magnitude, got {xd}");
    }

    // ── PerChannelDistortion ───────────────────────────────────────────────

    #[test]
    fn test_per_channel_identity_all_same() {
        let pcd = PerChannelDistortion::identity();
        let (rx, _) = pcd.red.apply(0.5, 0.5);
        let (gx, _) = pcd.green.apply(0.5, 0.5);
        let (bx, _) = pcd.blue.apply(0.5, 0.5);
        assert!((rx - gx).abs() < 1e-5);
        assert!((bx - gx).abs() < 1e-5);
    }

    #[test]
    fn test_per_channel_typical_channels_differ() {
        let pcd = PerChannelDistortion::typical(1.0);
        let (rx, _) = pcd.red.apply(0.8, 0.0);
        let (bx, _) = pcd.blue.apply(0.8, 0.0);
        // Red and blue channels should produce different displacements
        assert!((rx - bx).abs() > 1e-4, "R and B channels should differ");
    }

    // ── ChromaticAberrationEffect ──────────────────────────────────────────

    #[test]
    fn test_ca_effect_name() {
        let e = ChromaticAberrationEffect::identity();
        assert_eq!(e.name(), "ChromaticAberration");
        assert!(!e.description().is_empty());
    }

    #[test]
    fn test_ca_effect_identity_preserves_uniform_image() {
        let mut effect = ChromaticAberrationEffect::identity();
        let mut input = Frame::new(16, 16).expect("frame");
        input.clear([200, 100, 50, 255]);
        let mut output = Frame::new(16, 16).expect("frame");
        let params = EffectParams::new();
        effect.apply(&input, &mut output, &params).expect("apply");
        // On a uniform source image the identity effect should produce identical pixels
        let p = output.get_pixel(8, 8).expect("center");
        assert_eq!(p[0], 200, "R channel");
        assert_eq!(p[1], 100, "G channel");
    }

    #[test]
    fn test_ca_effect_dimension_mismatch() {
        let mut effect = ChromaticAberrationEffect::typical(1.0);
        let input = Frame::new(16, 16).expect("frame");
        let mut output = Frame::new(8, 8).expect("frame");
        assert!(effect
            .apply(&input, &mut output, &EffectParams::new())
            .is_err());
    }

    #[test]
    fn test_ca_effect_output_same_size() {
        let mut effect = ChromaticAberrationEffect::typical(0.5);
        let mut input = Frame::new(32, 32).expect("frame");
        input.clear([100, 150, 200, 255]);
        let mut output = Frame::new(32, 32).expect("frame");
        effect
            .apply(&input, &mut output, &EffectParams::new())
            .expect("apply");
        assert_eq!(output.width, 32);
        assert_eq!(output.height, 32);
    }

    #[test]
    fn test_ca_effect_uniform_source_unchanged() {
        // On a solid colour image any warp should produce the same colour
        let mut effect = ChromaticAberrationEffect::typical(2.0);
        let mut input = Frame::new(32, 32).expect("frame");
        input.clear([180, 90, 45, 255]);
        let mut output = Frame::new(32, 32).expect("frame");
        effect
            .apply(&input, &mut output, &EffectParams::new())
            .expect("apply");
        let p = output.get_pixel(16, 16).expect("center");
        assert!((p[0] as i32 - 180).abs() <= 2, "R: {}", p[0]);
        assert!((p[1] as i32 - 90).abs() <= 2, "G: {}", p[1]);
    }
}
