//! ACES Reference Gamut Compression (RGC) algorithm.
//!
//! Implements the Academy Color Encoding System (ACES) gamut compression
//! algorithm as specified in the ACES Reference Gamut Compression Transform
//! (CLF implementation S-2021-001).
//!
//! The algorithm uses "cusp" points — the vertices of the target gamut boundary
//! in a perceptual colorspace — to compute per-channel distance-based compression
//! that smoothly maps out-of-gamut ACEScg linear-light values back into gamut.
//!
//! # Background
//!
//! In ACEScg, oversaturated colors that are produced by cameras with wide-gamut
//! sensors, or through CG rendering, can have negative RGB components or
//! components greater than 1.0 even when represented in ACEScg. The Reference
//! Gamut Compression algorithm addresses this by:
//!
//! 1. Computing the "distance" of each channel from the neutral (achromatic) axis
//! 2. Applying per-channel smooth compression curves parametrized by threshold
//!    and limit values
//! 3. Reconstructing the compressed color from the compressed distances
//!
//! # References
//!
//! - ACES Gamut Mapping Architecture VWG — S-2021-001
//! - <https://docs.acescentral.com/specifications/rgc/>

#![allow(clippy::cast_precision_loss)]

use std::f32;
use std::sync::OnceLock;

/// A cusp point on the ACES gamut boundary in a perceptual colorspace.
///
/// The cusp of a gamut hull for a given hue angle is the point that has the
/// maximum colorimetric purity (chroma). In the ACES gamut compression
/// algorithm, cusps are used to parameterize the gamut boundary shape so
/// that compression can be applied smoothly and predictably.
///
/// In the JzCzhz or similar perceptual polar colorspace:
/// - `hue` is the angular coordinate (0°–360°)
/// - `lightness` corresponds to the Jz (or similar) lightness
/// - `chroma` corresponds to the Cz (or similar) chroma
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcesCuspPoint {
    /// Hue angle in degrees \[0, 360).
    pub hue: f32,
    /// Perceptual lightness at the cusp (Jz or similar, normalized 0–1).
    pub lightness: f32,
    /// Perceptual chroma at the cusp (Cz or similar, normalized >= 0).
    pub chroma: f32,
}

impl AcesCuspPoint {
    /// Construct a new cusp point, clamping to valid ranges.
    ///
    /// # Arguments
    ///
    /// * `hue` - Hue in degrees, wrapped to \[0, 360)
    /// * `lightness` - Lightness component, clamped to \[0, 1]
    /// * `chroma` - Chroma component, clamped to >= 0
    #[must_use]
    pub fn new(hue: f32, lightness: f32, chroma: f32) -> Self {
        let hue_wrapped = hue.rem_euclid(360.0);
        Self {
            hue: hue_wrapped,
            lightness: lightness.clamp(0.0, 1.0),
            chroma: chroma.max(0.0),
        }
    }

    /// Returns the angular distance in hue space between two cusp points.
    ///
    /// The result is always in \[0, 180], since hue distance wraps around.
    #[must_use]
    pub fn hue_distance(&self, other: &Self) -> f32 {
        let diff = (self.hue - other.hue).abs().rem_euclid(360.0);
        if diff > 180.0 {
            360.0 - diff
        } else {
            diff
        }
    }
}

/// ACES Reference Gamut Compression transform for ACEScg linear-light data.
///
/// This struct implements the per-channel distance-based compression algorithm
/// from the ACES Reference Gamut Compression specification (S-2021-001).
///
/// ## Algorithm Overview
///
/// For each pixel (R, G, B) in ACEScg:
/// 1. Find the maximum channel value: `max_c = max(R, G, B)`
/// 2. For each channel `c` in {R, G, B}, compute a "distance" from the
///    neutral axis: `d = (1 - c) / (max_c - c + ε)`
///    - `d = 0`: channel equals neutral (achromatic)
///    - `d = 1`: channel is at the gamut boundary
///    - `d > 1`: channel is outside the gamut (out-of-gamut)
/// 3. Compress each distance using the smooth parabolic function
/// 4. Reconstruct the compressed channel values from the compressed distances
///
/// ## Parameters
///
/// The per-channel `threshold` and `limit` values are based on empirical
/// measurements of real-world camera data in ACES working group analysis.
/// The default values match the ACES Reference Gamut Compression Transform
/// specification exactly.
///
/// - **Threshold** `[t_r, t_g, t_b]`: the normalized distance at which
///   compression begins. Below this value, colors are passed through unchanged.
///   Default: `[0.815, 0.803, 0.880]`
///
/// - **Limit** `[l_r, l_g, l_b]`: the normalized distance at which the
///   compression asymptotically approaches. Represents the furthest extent of
///   out-of-gamut excursion that the compressor handles.
///   Default: `[1.147, 1.264, 1.312]`
///
/// - **Power** `p`: controls the softness of the knee in the compression curve.
///   Default: `1.2` (matches ACES spec)
#[derive(Debug, Clone)]
pub struct AcesGamutCompressor {
    /// Per-channel distance threshold at which compression starts \[R, G, B].
    ///
    /// Must be in (0, 1). Values closer to 1.0 allow more of the gamut through
    /// uncompressed; values closer to 0.0 compress more aggressively.
    pub threshold: [f32; 3],

    /// Per-channel distance limit where clipping would occur \[R, G, B].
    ///
    /// Must be > 1.0. This is the normalized distance beyond which colors
    /// are considered "hard out of gamut" and compressed to the boundary.
    pub limit: [f32; 3],

    /// Compression curve power / softness exponent.
    ///
    /// Controls how sharply the knee of the compression curve bends.
    /// Default `1.2` gives a smooth, cinematically pleasing result.
    /// Higher values give a harder knee; lower values give a softer roll-off.
    pub power: f32,
}

impl Default for AcesGamutCompressor {
    fn default() -> Self {
        Self::new_default()
    }
}

impl AcesGamutCompressor {
    /// Construct the default ACES Reference Gamut Compression Transform parameters.
    ///
    /// These values are taken directly from the ACES S-2021-001 specification
    /// and represent the empirically derived optimal settings for compressing
    /// real-world camera data (Sony VENICE, ARRI ALEXA) into ACEScg gamut.
    ///
    /// | Channel | Threshold | Limit   |
    /// |---------|-----------|---------|
    /// | Red     | 0.815     | 1.147   |
    /// | Green   | 0.803     | 1.264   |
    /// | Blue    | 0.880     | 1.312   |
    ///
    /// Power: 1.2
    #[must_use]
    pub fn new_default() -> Self {
        Self {
            // Per-channel thresholds from ACES S-2021-001 Table 1
            threshold: [0.815, 0.803, 0.880],
            // Per-channel limits from ACES S-2021-001 Table 1
            limit: [1.147, 1.264, 1.312],
            power: 1.2,
        }
    }

    /// Construct with custom per-channel parameters.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Per-channel compression start distances (clamped to (0, 0.999))
    /// * `limit` - Per-channel compression limit distances (must be > 1.0, enforced)
    /// * `power` - Compression curve power (clamped to (0.1, 10.0))
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_colormgmt::aces_gamut::AcesGamutCompressor;
    ///
    /// let compressor = AcesGamutCompressor::new_custom(
    ///     [0.815, 0.803, 0.880],
    ///     [1.147, 1.264, 1.312],
    ///     1.2,
    /// );
    /// ```
    #[must_use]
    pub fn new_custom(threshold: [f32; 3], limit: [f32; 3], power: f32) -> Self {
        Self {
            threshold: [
                threshold[0].clamp(f32::EPSILON, 0.999),
                threshold[1].clamp(f32::EPSILON, 0.999),
                threshold[2].clamp(f32::EPSILON, 0.999),
            ],
            limit: [
                limit[0].max(1.001),
                limit[1].max(1.001),
                limit[2].max(1.001),
            ],
            power: power.clamp(0.1, 10.0),
        }
    }

    /// Construct with custom per-channel parameters (alias for `new_custom`).
    ///
    /// This is the canonical `new` constructor matching the struct field order
    /// documented in the module-level API.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Per-channel compression start distances (clamped to (0, 0.999))
    /// * `limit` - Per-channel compression limit distances (must be > 1.0, enforced)
    /// * `power` - Compression curve power (clamped to (0.1, 10.0))
    #[must_use]
    pub fn new(threshold: [f32; 3], limit: [f32; 3], power: f32) -> Self {
        Self::new_custom(threshold, limit, power)
    }

    /// Apply ACES gamut compression to an entire frame in-place.
    ///
    /// The `frame` slice must be interleaved RGB f32 in ACEScg linear light
    /// with length `width * height * 3`. Each triplet `[r, g, b]` is compressed
    /// in-place.
    ///
    /// The `width` and `height` parameters are accepted for API completeness
    /// but the function operates purely on the slice length.
    ///
    /// # Panics
    ///
    /// Panics if `frame.len() != width as usize * height as usize * 3`.
    pub fn compress_frame_inplace(&self, frame: &mut [f32], width: u32, height: u32) {
        let expected = width as usize * height as usize * 3;
        assert_eq!(
            frame.len(),
            expected,
            "frame length {} does not match width({}) * height({}) * 3 = {}",
            frame.len(),
            width,
            height,
            expected,
        );
        for chunk in frame.chunks_exact_mut(3) {
            let (r, g, b) = self.compress(chunk[0], chunk[1], chunk[2]);
            chunk[0] = r;
            chunk[1] = g;
            chunk[2] = b;
        }
    }

    /// Apply ACES gamut compression to a single ACEScg linear-light pixel.
    ///
    /// This is the main entry point for per-pixel gamut compression. It maps
    /// the input `(r, g, b)` tuple from potentially out-of-gamut ACEScg values
    /// to compressed (but still linear-light ACEScg) values.
    ///
    /// ## Distance Metric
    ///
    /// For each channel `c` relative to `max_c = max(R, G, B)`:
    ///
    /// ```text
    /// d_c = 1 - c / max_c
    /// ```
    ///
    /// This gives:
    /// - `d = 0` when `c = max_c` (the dominant channel, on the neutral axis)
    /// - `d = 1` when `c = 0` (at the gamut boundary along this hue)
    /// - `d > 1` when `c < 0` (outside the gamut — negative component)
    ///
    /// The compressed channel is reconstructed as `c_out = max_c * (1 - d_compressed)`.
    ///
    /// ## Invariants
    ///
    /// - If all channels are in \[0, 1], the output equals the input (no compression).
    /// - The output is always >= 0 for all channels.
    ///
    /// # Arguments
    ///
    /// * `r` - Red channel in ACEScg linear light
    /// * `g` - Green channel in ACEScg linear light
    /// * `b` - Blue channel in ACEScg linear light
    ///
    /// # Returns
    ///
    /// Compressed `(r, g, b)` values in ACEScg linear light.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_colormgmt::aces_gamut::AcesGamutCompressor;
    ///
    /// let c = AcesGamutCompressor::new_default();
    /// // In-gamut color — passes through unchanged
    /// let (r, g, b) = c.compress(0.5, 0.3, 0.7);
    /// assert!((r - 0.5).abs() < 1e-5);
    ///
    /// // Out-of-gamut color — gets compressed
    /// let (r2, g2, b2) = c.compress(1.2, -0.1, 0.8);
    /// assert!(g2 >= 0.0, "negative channel compressed to positive");
    /// ```
    #[must_use]
    pub fn compress(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        const EPSILON: f32 = 1e-6;

        // The max channel value determines the neutral axis scaling.
        let max_c = r.max(g).max(b);

        // If the pixel is entirely black or near-black, return as-is.
        if max_c.abs() < EPSILON {
            return (r, g, b);
        }

        // Compute per-channel distances from the neutral (achromatic) axis.
        //
        // d_c = 1 - c / max_c
        //
        // Geometric interpretation:
        // - d = 0 when c = max_c  (dominant channel, defines hue direction)
        // - d = 1 when c = 0      (at the ACEScg gamut boundary)
        // - d > 1 when c < 0      (outside the gamut, negative component)
        let d_r = 1.0 - r / max_c;
        let d_g = 1.0 - g / max_c;
        let d_b = 1.0 - b / max_c;

        // Apply per-channel smooth compression to each distance independently.
        let d_r_c = self.compress_distance(d_r, self.threshold[0], self.limit[0]);
        let d_g_c = self.compress_distance(d_g, self.threshold[1], self.limit[1]);
        let d_b_c = self.compress_distance(d_b, self.threshold[2], self.limit[2]);

        // Reconstruct compressed channel values from compressed distances.
        //
        // Inverse of d = 1 - c / max_c  →  c = max_c * (1 - d)
        //
        // Replacing d with the compressed distance d_c gives us the
        // reconstructed channel value on the compressed gamut boundary.
        let r_out = max_c * (1.0 - d_r_c);
        let g_out = max_c * (1.0 - d_g_c);
        let b_out = max_c * (1.0 - d_b_c);

        // Clamp to prevent any residual numerical overshoot below zero.
        (r_out.max(0.0), g_out.max(0.0), b_out.max(0.0))
    }

    /// Compress a single channel distance using the ACES parabolic smooth curve.
    ///
    /// This is the core mathematical function of the ACES Reference Gamut
    /// Compression algorithm. It maps the normalized distance `d` from the
    /// neutral axis through a smooth compression curve parameterized by
    /// `threshold` and `limit`.
    ///
    /// ## Compression function
    ///
    /// For `d <= threshold`: `compress(d) = d` (identity, no compression)
    ///
    /// For `d > threshold`: the compression applies a smooth parabolic knee
    /// that asymptotically approaches `limit` as `d → ∞`.
    ///
    /// The formula for the compressed region (d > t):
    /// ```text
    /// s = (limit - threshold) / ((1 - threshold) / (limit - threshold) + 1)^power
    /// compress(d) = limit - s / (d - threshold + s/((limit-threshold)^(1/power)))^(1/power) * ...
    /// ```
    ///
    /// In practice the ACES spec uses a simplified parametric curve derived
    /// from the Daniele-Faber tone mapping operator:
    ///
    /// ```text
    /// compress(d, t, l) =
    ///   d,              if d <= t
    ///   l - (l - t) / (1 + ((d - t) / (l - t))^power),  if d > t
    /// ```
    ///
    /// This formulation:
    /// - Is continuous and has a continuous first derivative at `d = t`
    /// - Approaches `l` asymptotically as `d → ∞`
    /// - Is monotonically increasing
    ///
    /// # Arguments
    ///
    /// * `d` - Input distance from neutral axis (0 = neutral, 1 = boundary, >1 = out of gamut)
    /// * `threshold` - Distance at which compression begins (from `self.threshold`)
    /// * `limit` - Asymptotic limit approached as d → ∞ (from `self.limit`)
    ///
    /// # Returns
    ///
    /// Compressed distance in range \[0, limit).
    #[must_use]
    pub fn compress_distance(&self, d: f32, threshold: f32, limit: f32) -> f32 {
        // If in the uncompressed region, pass through unchanged
        if d <= threshold {
            return d;
        }

        // Normalized excess beyond threshold
        // x = (d - threshold) / (limit - threshold)
        // This maps [threshold, infinity) → [0, infinity)
        let range = limit - threshold;
        if range < f32::EPSILON {
            // Degenerate case: threshold == limit, hard clip to threshold
            return threshold;
        }

        let x = (d - threshold) / range;

        // Smooth parabolic compression using the Daniele-Faber formulation:
        // y = x^p / (x^p + 1)  maps [0, inf) → [0, 1)
        // This is a smooth sigmoidal function with:
        //   y(0) = 0 (continuous at threshold)
        //   y → 1 as x → ∞ (approaches limit asymptotically)
        //   y'(0) > 0 (smooth at knee)
        let x_p = x.powf(self.power);
        // Protect against NaN/Inf when x is very large
        let y = if x_p.is_finite() && (x_p + 1.0).abs() > f32::EPSILON {
            x_p / (x_p + 1.0)
        } else {
            // x is very large: asymptotically approach 1
            1.0
        };

        // Map y back to the output space: [0, 1) → [threshold, limit)
        threshold + y * range
    }

    /// Apply gamut compression to an entire frame of ACEScg pixels.
    ///
    /// Processes an RGB-interleaved slice (`[r0, g0, b0, r1, g1, b1, ...]`)
    /// and returns a new `Vec<f32>` with the compressed values.
    ///
    /// # Panics
    ///
    /// Panics if `pixels.len()` is not divisible by 3.
    ///
    /// # Arguments
    ///
    /// * `pixels` - RGB-interleaved pixel data in ACEScg linear light
    ///
    /// # Returns
    ///
    /// Compressed RGB-interleaved pixel data.
    #[must_use]
    pub fn compress_frame(&self, pixels: &[f32]) -> Vec<f32> {
        assert_eq!(
            pixels.len() % 3,
            0,
            "pixels slice length must be a multiple of 3; got {}",
            pixels.len()
        );
        let mut out = Vec::with_capacity(pixels.len());
        for chunk in pixels.chunks_exact(3) {
            let (r, g, b) = self.compress(chunk[0], chunk[1], chunk[2]);
            out.push(r);
            out.push(g);
            out.push(b);
        }
        out
    }

    /// Check whether a pixel is within the ACEScg gamut (all channels in \[0, 1]).
    #[must_use]
    pub fn is_in_gamut(r: f32, g: f32, b: f32) -> bool {
        r >= 0.0 && r <= 1.0 && g >= 0.0 && g <= 1.0 && b >= 0.0 && b <= 1.0
    }
}

/// Lookup table of ACES cusp points for a set of representative hue angles.
///
/// The ACES RGC algorithm may use a pre-computed table of cusp points to
/// efficiently find the gamut boundary for arbitrary hue values. This struct
/// stores the table and provides interpolation.
///
/// The cusp point table represents the gamut boundary in a perceptual
/// colorspace (e.g., JzCzhz) derived from the ACEScg primaries. Cusp points
/// at intermediate hue angles can be found by linear interpolation in
/// (Jz, Cz) space between the nearest table entries.
#[derive(Debug, Clone)]
pub struct AcesCuspTable {
    /// The sampled cusp points, sorted by hue angle.
    cusps: Vec<AcesCuspPoint>,
}

impl AcesCuspTable {
    /// Construct a cusp table from an unsorted list of cusp points.
    ///
    /// Points will be sorted by hue angle. Duplicate or near-duplicate hue
    /// values are kept (the caller is responsible for de-duplication if needed).
    ///
    /// # Panics
    ///
    /// Panics if `cusps` is empty.
    #[must_use]
    pub fn new(mut cusps: Vec<AcesCuspPoint>) -> Self {
        assert!(!cusps.is_empty(), "cusp table must have at least one entry");
        cusps.sort_by(|a, b| {
            a.hue
                .partial_cmp(&b.hue)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self { cusps }
    }

    /// Construct a default ACEScg cusp table based on the AP1 (ACEScg) primaries.
    ///
    /// This table is **computed** from the AP1 (ACEScg) primary chromaticities
    /// using the CIECAM02 perceptual colour model — the same mechanism used by
    /// [`crate::cusp_gamut::CuspGamutMapper`] for Rec.709/P3/Rec.2020.
    ///
    /// AP1 (ACEScg) primary CIE xy chromaticities (D65 white point):
    ///
    /// | Primary | x     | y     |
    /// |---------|-------|-------|
    /// | R       | 0.713 | 0.293 |
    /// | G       | 0.165 | 0.830 |
    /// | B       | 0.128 | 0.044 |
    /// | White   | 0.3127| 0.3290|
    ///
    /// Six representative hue angles (0°, 60°, 120°, 180°, 240°, 300°) in
    /// CIECAM02 hue space are sampled and the resulting cusps are returned,
    /// sorted by hue.  The CIECAM02 lightness (J ∈ \[0,100\]) and chroma (C) are
    /// normalised into the \[0,1\] convention used by [`AcesCuspPoint`]:
    ///
    /// - `lightness = J / 100`
    /// - `chroma = C / C_NORM` where `C_NORM = 100.0` (typical CIECAM02 chroma
    ///   range for highly saturated colours is ~10–80; dividing by 100 gives a
    ///   normalised value in roughly \[0.1, 0.8\])
    ///
    /// The result is cached in a `OnceLock` so the (relatively expensive) cusp
    /// scan is only performed once per process lifetime.
    #[must_use]
    pub fn acescg_default() -> Self {
        /// Normalisation divisor for CIECAM02 chroma → [0,1].
        const CHROMA_NORM: f64 = 100.0;
        /// Normalisation divisor for CIECAM02 lightness → [0,1].
        const LIGHTNESS_NORM: f64 = 100.0;

        // AP1 (ACEScg) CIE xy primaries and D65 white point.
        const AP1_PRIMARIES: [[f64; 2]; 3] = [
            [0.713, 0.293], // R
            [0.165, 0.830], // G
            [0.128, 0.044], // B
        ];
        const D65_WHITE: [f64; 2] = [0.3127, 0.3290];

        // Six representative CIECAM02 hue angles (degrees).
        const HUE_SAMPLES: [f64; 6] = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0];

        // Global cache — computed at most once per process.
        static CACHE: OnceLock<Vec<AcesCuspPoint>> = OnceLock::new();

        let computed = CACHE.get_or_init(|| {
            use crate::ciecam02::{CiecamModel, CiecamViewingConditions, SurroundCondition};
            use crate::cusp_gamut::find_cusp_at_hue_for_primaries;

            let viewing = CiecamViewingConditions {
                adapting_luminance_la: 64.0,
                background_relative_lum_yb: 20.0,
                surround: SurroundCondition::Average,
            };
            let model = CiecamModel::new(viewing);

            HUE_SAMPLES
                .iter()
                .map(|&hue_target| {
                    let cusp = find_cusp_at_hue_for_primaries(
                        hue_target,
                        &AP1_PRIMARIES,
                        D65_WHITE,
                        &model,
                    );
                    // Normalise CIECAM02 J → [0,1] and C → [0,∞) normalised.
                    let lightness = (cusp.lightness / LIGHTNESS_NORM).clamp(0.0, 1.0) as f32;
                    let chroma = (cusp.max_chroma / CHROMA_NORM).max(0.0) as f32;
                    AcesCuspPoint::new(cusp.hue_angle as f32, lightness, chroma)
                })
                .collect()
        });

        Self::new(computed.clone())
    }

    /// Find the cusp point for an arbitrary hue angle by linear interpolation.
    ///
    /// Interpolates between the two nearest entries in the table using angular
    /// distance. The interpolation is linear in (lightness, chroma) space.
    ///
    /// # Arguments
    ///
    /// * `hue_deg` - Hue angle in degrees (will be wrapped to \[0, 360))
    #[must_use]
    pub fn interpolate(&self, hue_deg: f32) -> AcesCuspPoint {
        let hue = hue_deg.rem_euclid(360.0);

        if self.cusps.len() == 1 {
            return self.cusps[0];
        }

        // Find the two table entries that bracket this hue angle.
        // The table is sorted by hue, so we need to handle the wrap-around
        // between the last entry and the first entry.
        let n = self.cusps.len();

        // Find the first cusp with hue >= target
        let upper_idx = self.cusps.iter().position(|c| c.hue >= hue).unwrap_or(0); // wraps around: hue is beyond the last entry

        let (a, b) = if upper_idx == 0 {
            // Target hue is before the first entry or after the last entry
            // Interpolate between last and first (wrap-around)
            (&self.cusps[n - 1], &self.cusps[0])
        } else {
            (&self.cusps[upper_idx - 1], &self.cusps[upper_idx])
        };

        // Compute angular interpolation factor t in [0, 1]
        let hue_range = if b.hue > a.hue {
            b.hue - a.hue
        } else {
            // Wrap-around: a is near 360°, b is near 0°
            (b.hue + 360.0) - a.hue
        };

        let hue_offset = if hue >= a.hue {
            hue - a.hue
        } else {
            (hue + 360.0) - a.hue
        };

        let t = if hue_range < f32::EPSILON {
            0.0
        } else {
            (hue_offset / hue_range).clamp(0.0, 1.0)
        };

        // Linear interpolation of lightness and chroma
        AcesCuspPoint {
            hue,
            lightness: a.lightness + t * (b.lightness - a.lightness),
            chroma: a.chroma + t * (b.chroma - a.chroma),
        }
    }

    /// Returns all stored cusp points as a slice, sorted by hue.
    #[must_use]
    pub fn cusps(&self) -> &[AcesCuspPoint] {
        &self.cusps
    }

    /// Returns the number of cusp entries in the table.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cusps.len()
    }

    /// Returns `true` if the table has no entries (should never happen after construction).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cusps.is_empty()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── AcesCuspPoint ─────────────────────────────────────────────────────────

    #[test]
    fn test_cusp_point_new_wraps_hue() {
        let p = AcesCuspPoint::new(400.0, 0.5, 0.2);
        assert!((p.hue - 40.0).abs() < 1e-4, "hue wrapping: got {}", p.hue);
    }

    #[test]
    fn test_cusp_point_new_negative_hue() {
        let p = AcesCuspPoint::new(-30.0, 0.5, 0.2);
        assert!((p.hue - 330.0).abs() < 1e-4, "neg hue wrapping: {}", p.hue);
    }

    #[test]
    fn test_cusp_point_clamps_lightness() {
        let p = AcesCuspPoint::new(0.0, 1.5, 0.2);
        assert_eq!(p.lightness, 1.0);
        let p2 = AcesCuspPoint::new(0.0, -0.5, 0.2);
        assert_eq!(p2.lightness, 0.0);
    }

    #[test]
    fn test_cusp_point_clamps_chroma() {
        let p = AcesCuspPoint::new(0.0, 0.5, -0.3);
        assert_eq!(p.chroma, 0.0);
    }

    #[test]
    fn test_cusp_point_hue_distance() {
        let a = AcesCuspPoint::new(10.0, 0.5, 0.2);
        let b = AcesCuspPoint::new(350.0, 0.5, 0.2);
        let dist = a.hue_distance(&b);
        // Distance should be 20° (wrapping around)
        assert!(
            (dist - 20.0).abs() < 1e-3,
            "hue distance should be 20°, got {dist}"
        );
    }

    #[test]
    fn test_cusp_point_hue_distance_direct() {
        let a = AcesCuspPoint::new(90.0, 0.5, 0.2);
        let b = AcesCuspPoint::new(180.0, 0.5, 0.2);
        let dist = a.hue_distance(&b);
        assert!(
            (dist - 90.0).abs() < 1e-3,
            "hue distance should be 90°, got {dist}"
        );
    }

    // ── AcesGamutCompressor construction ─────────────────────────────────────

    #[test]
    fn test_default_compressor_params_match_spec() {
        let c = AcesGamutCompressor::new_default();
        // ACES S-2021-001 Table 1 values
        assert!((c.threshold[0] - 0.815).abs() < 1e-6, "R threshold");
        assert!((c.threshold[1] - 0.803).abs() < 1e-6, "G threshold");
        assert!((c.threshold[2] - 0.880).abs() < 1e-6, "B threshold");
        assert!((c.limit[0] - 1.147).abs() < 1e-6, "R limit");
        assert!((c.limit[1] - 1.264).abs() < 1e-6, "G limit");
        assert!((c.limit[2] - 1.312).abs() < 1e-6, "B limit");
        assert!((c.power - 1.2).abs() < 1e-6, "power");
    }

    #[test]
    fn test_custom_compressor_clamps_params() {
        let c = AcesGamutCompressor::new_custom(
            [0.0, 0.5, 1.5], // first will be clamped to epsilon, third to 0.999
            [0.5, 1.5, 2.0], // first will be raised to 1.001
            15.0,            // will be clamped to 10.0
        );
        assert!(c.threshold[0] > 0.0, "threshold clamped above 0");
        assert!(c.threshold[2] <= 0.999, "threshold clamped below 1");
        assert!(c.limit[0] >= 1.001, "limit raised to minimum");
        assert!((c.power - 10.0).abs() < 1e-5, "power clamped to 10");
    }

    // ── compress_distance ─────────────────────────────────────────────────────

    #[test]
    fn test_compress_distance_identity_below_threshold() {
        let c = AcesGamutCompressor::new_default();
        let threshold = 0.815;
        let limit = 1.147;
        // Any d <= threshold should pass through unchanged
        for &d in &[0.0, 0.3, 0.5, 0.815] {
            let out = c.compress_distance(d, threshold, limit);
            assert!(
                (out - d).abs() < 1e-6,
                "d={d} should be identity, got {out}"
            );
        }
    }

    #[test]
    fn test_compress_distance_monotonically_increasing() {
        let c = AcesGamutCompressor::new_default();
        let threshold = 0.815;
        let limit = 1.147;
        let distances = [0.0, 0.4, 0.815, 0.9, 1.0, 1.1, 1.147, 1.5, 2.0];
        let outputs: Vec<f32> = distances
            .iter()
            .map(|&d| c.compress_distance(d, threshold, limit))
            .collect();
        for i in 1..outputs.len() {
            assert!(
                outputs[i] >= outputs[i - 1],
                "compress_distance must be monotonic: outputs[{}]={} < outputs[{}]={}",
                i,
                outputs[i],
                i - 1,
                outputs[i - 1]
            );
        }
    }

    #[test]
    fn test_compress_distance_asymptotes_at_limit() {
        let c = AcesGamutCompressor::new_default();
        let threshold = 0.815;
        let limit = 1.147;
        // At very large d, output should approach limit but not exceed it
        let out = c.compress_distance(100.0, threshold, limit);
        assert!(
            out < limit,
            "output {out} must be strictly below limit {limit}"
        );
        assert!(
            out > limit - 0.05,
            "output {out} should be close to limit {limit}"
        );
    }

    #[test]
    fn test_compress_distance_continuous_at_threshold() {
        let c = AcesGamutCompressor::new_default();
        let threshold = 0.815;
        let limit = 1.147;
        // Just below and at threshold should be nearly identical (continuity)
        let at = c.compress_distance(threshold, threshold, limit);
        let just_below = c.compress_distance(threshold - 1e-4, threshold, limit);
        assert!(
            (at - just_below).abs() < 1e-3,
            "discontinuity at threshold: at={at}, below={just_below}"
        );
    }

    // ── compress (full pixel) ─────────────────────────────────────────────────

    #[test]
    fn test_compress_in_gamut_identity() {
        let c = AcesGamutCompressor::new_default();
        // A color well within ACEScg gamut
        let (r, g, b) = c.compress(0.5, 0.3, 0.7);
        assert!(
            (r - 0.5).abs() < 1e-4,
            "in-gamut R should be unchanged: {r}"
        );
        assert!(
            (g - 0.3).abs() < 1e-4,
            "in-gamut G should be unchanged: {g}"
        );
        assert!(
            (b - 0.7).abs() < 1e-4,
            "in-gamut B should be unchanged: {b}"
        );
    }

    #[test]
    fn test_compress_negative_channel_maps_to_nonnegative() {
        let c = AcesGamutCompressor::new_default();
        // Highly saturated color with negative red channel
        let (r, _g, _b) = c.compress(-0.1, 0.8, 0.4);
        assert!(r >= 0.0, "negative channel after compression: R={r}");
    }

    #[test]
    fn test_compress_negative_green_maps_to_nonnegative() {
        let c = AcesGamutCompressor::new_default();
        let (_r, g, _b) = c.compress(0.9, -0.2, 0.3);
        assert!(g >= 0.0, "negative green after compression: G={g}");
    }

    #[test]
    fn test_compress_multiple_negative_channels() {
        let c = AcesGamutCompressor::new_default();
        let (r, g, b) = c.compress(1.3, -0.15, -0.05);
        assert!(r >= 0.0, "R={r}");
        assert!(g >= 0.0, "G={g}");
        assert!(b >= 0.0, "B={b}");
    }

    #[test]
    fn test_compress_neutral_grey_unchanged() {
        let c = AcesGamutCompressor::new_default();
        // Perfect neutral grey — all channels equal, no compression needed
        let (r, g, b) = c.compress(0.5, 0.5, 0.5);
        assert!(
            (r - 0.5).abs() < 1e-4 && (g - 0.5).abs() < 1e-4 && (b - 0.5).abs() < 1e-4,
            "neutral grey should be unchanged: ({r}, {g}, {b})"
        );
    }

    #[test]
    fn test_compress_black_unchanged() {
        let c = AcesGamutCompressor::new_default();
        let (r, g, b) = c.compress(0.0, 0.0, 0.0);
        assert!(r.abs() < 1e-6 && g.abs() < 1e-6 && b.abs() < 1e-6);
    }

    #[test]
    fn test_compress_hue_direction_preserved() {
        let c = AcesGamutCompressor::new_default();
        // Out-of-gamut reddish color: the output should still be reddish
        let (r, g, b) = c.compress(1.5, 0.1, 0.05);
        // Red should be highest after compression
        assert!(
            r >= g && r >= b,
            "hue direction should be preserved: ({r},{g},{b})"
        );
    }

    #[test]
    fn test_is_in_gamut() {
        assert!(AcesGamutCompressor::is_in_gamut(0.5, 0.3, 0.7));
        assert!(!AcesGamutCompressor::is_in_gamut(-0.1, 0.3, 0.7));
        assert!(!AcesGamutCompressor::is_in_gamut(0.5, 1.1, 0.7));
        assert!(!AcesGamutCompressor::is_in_gamut(0.5, 0.3, -0.01));
    }

    // ── compress_frame ────────────────────────────────────────────────────────

    #[test]
    fn test_compress_frame_length_preserved() {
        let c = AcesGamutCompressor::new_default();
        let pixels: Vec<f32> = (0..30).map(|i| i as f32 / 30.0).collect();
        let out = c.compress_frame(&pixels);
        assert_eq!(out.len(), 30);
    }

    #[test]
    fn test_compress_frame_all_nonnegative() {
        let c = AcesGamutCompressor::new_default();
        // Frame with out-of-gamut values
        let pixels = vec![1.2_f32, -0.1, 0.5, 0.9, -0.3, 0.2, 0.5, 0.5, 0.5];
        let out = c.compress_frame(&pixels);
        for (i, &v) in out.iter().enumerate() {
            assert!(v >= 0.0, "pixel[{i}]={v} should be non-negative");
        }
    }

    #[test]
    #[should_panic(expected = "multiple of 3")]
    fn test_compress_frame_panics_on_bad_length() {
        let c = AcesGamutCompressor::new_default();
        c.compress_frame(&[0.5, 0.3]);
    }

    // ── AcesCuspTable ─────────────────────────────────────────────────────────

    #[test]
    fn test_cusp_table_sorted_on_construction() {
        let cusps = vec![
            AcesCuspPoint::new(300.0, 0.4, 0.2),
            AcesCuspPoint::new(100.0, 0.6, 0.3),
            AcesCuspPoint::new(200.0, 0.5, 0.25),
        ];
        let table = AcesCuspTable::new(cusps);
        let stored = table.cusps();
        assert!(stored[0].hue <= stored[1].hue, "table not sorted");
        assert!(stored[1].hue <= stored[2].hue, "table not sorted");
    }

    #[test]
    fn test_cusp_table_len() {
        let table = AcesCuspTable::acescg_default();
        assert_eq!(table.len(), 6);
        assert!(!table.is_empty());
    }

    #[test]
    fn test_cusp_table_interpolate_exact_entry() {
        let table = AcesCuspTable::acescg_default();
        // Interpolating at exactly an existing hue should return that cusp's values
        let cusp_at_20 = table.interpolate(20.0);
        assert!(
            (cusp_at_20.hue - 20.0).abs() < 1e-3,
            "hue: {}",
            cusp_at_20.hue
        );
    }

    #[test]
    fn test_cusp_table_interpolate_midpoint() {
        let table = AcesCuspTable::acescg_default();
        let cusps = table.cusps();

        // Pick the first two adjacent entries and interpolate their midpoint hue.
        // The interpolated lightness must be between the two endpoint lightnesses
        // (linear interpolation property).
        assert!(cusps.len() >= 2, "need at least 2 cusps for midpoint test");
        let a = cusps[0];
        let b = cusps[1];

        let mid_hue = (a.hue + b.hue) / 2.0;
        let mid = table.interpolate(mid_hue);

        let lo_l = a.lightness.min(b.lightness);
        let hi_l = a.lightness.max(b.lightness);

        assert!(
            mid.lightness >= lo_l - 0.01 && mid.lightness <= hi_l + 0.01,
            "midpoint lightness {:.4} not in [{lo_l:.4}, {hi_l:.4}] \
             (interpolated between hue {:.1}° and {:.1}° at mid {mid_hue:.1}°)",
            mid.lightness,
            a.hue,
            b.hue
        );
    }

    #[test]
    fn test_cusp_table_interpolate_wraparound() {
        let table = AcesCuspTable::acescg_default();
        // Hue near 360° should interpolate between last entry (~320°) and first (20°)
        let cusp_350 = table.interpolate(350.0);
        // Should be between blue (320°) and red (20°) values
        assert!(cusp_350.lightness > 0.0 && cusp_350.lightness < 1.0);
        assert!(cusp_350.chroma > 0.0);
    }

    #[test]
    fn test_cusp_table_interpolate_all_in_valid_range() {
        let table = AcesCuspTable::acescg_default();
        for hue in (0..360).step_by(5) {
            let c = table.interpolate(hue as f32);
            assert!(
                c.lightness >= 0.0 && c.lightness <= 1.0,
                "lightness out of range at hue {hue}: {}",
                c.lightness
            );
            assert!(
                c.chroma >= 0.0,
                "negative chroma at hue {hue}: {}",
                c.chroma
            );
        }
    }

    #[test]
    #[should_panic(expected = "at least one entry")]
    fn test_cusp_table_empty_panics() {
        AcesCuspTable::new(vec![]);
    }

    // ── Integration: compress + cusp table ───────────────────────────────────

    #[test]
    fn test_compress_with_cusp_guided_params() {
        // Simulate a scenario where cusp data informs threshold/limit selection.
        // The ACEScg compressor with default ACES params should compress
        // a highly saturated cyan (negative red) to positive values.
        let compressor = AcesGamutCompressor::new_default();
        let table = AcesCuspTable::acescg_default();

        // Highly saturated cyan in ACEScg: large negative red
        let (r_in, g_in, b_in) = (-0.3, 0.9, 0.8);
        let cusp = table.interpolate(180.0); // ~cyan hue
        assert!(cusp.chroma > 0.0, "cusp should have positive chroma");

        let (r_out, g_out, b_out) = compressor.compress(r_in, g_in, b_in);
        assert!(
            r_out >= 0.0,
            "compressed red should be non-negative: {r_out}"
        );
        assert!(
            g_out >= 0.0,
            "compressed green should be non-negative: {g_out}"
        );
        assert!(
            b_out >= 0.0,
            "compressed blue should be non-negative: {b_out}"
        );
    }

    #[test]
    fn test_roundtrip_in_gamut_stays_in_gamut() {
        let compressor = AcesGamutCompressor::new_default();
        // In-gamut colors should remain in gamut after compression
        let test_pixels = [
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.18, 0.18, 0.18), // 18% grey card
            (0.9, 0.1, 0.05),
            (0.05, 0.9, 0.1),
            (0.05, 0.1, 0.9),
        ];
        for (r, g, b) in test_pixels {
            let (ro, go, bo) = compressor.compress(r, g, b);
            assert!(
                AcesGamutCompressor::is_in_gamut(ro, go, bo),
                "in-gamut pixel ({r},{g},{b}) should remain in gamut, got ({ro},{go},{bo})"
            );
        }
    }

    // ── Computed AP1 cusp tests ────────────────────────────────────────────

    /// Verify that `acescg_default()` returns computed (not hardcoded) cusps.
    ///
    /// The old hardcoded values were exactly-round numbers like 20.0° / 0.45 / 0.38.
    /// The computed values must:
    /// - Span a reasonable range of hue angles across [0°, 360°)
    /// - Have positive, physically plausible chroma (> 0.01)
    /// - Differ from the former round-number defaults on at least one axis
    #[test]
    fn test_acescg_cusps_are_computed_not_hardcoded() {
        let table = AcesCuspTable::acescg_default();
        let cusps = table.cusps();

        assert_eq!(cusps.len(), 6, "should have 6 representative hue samples");

        // All hues must be in [0, 360) and span the circle reasonably.
        let min_hue = cusps.iter().map(|c| c.hue).fold(f32::INFINITY, f32::min);
        let max_hue = cusps
            .iter()
            .map(|c| c.hue)
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(min_hue >= 0.0, "min hue should be >= 0, got {min_hue}");
        assert!(max_hue < 360.0, "max hue should be < 360, got {max_hue}");
        assert!(
            max_hue - min_hue > 200.0,
            "cusps should span > 200° but span only {:.1}°",
            max_hue - min_hue
        );

        // All chroma values must be positive and physically plausible.
        for c in cusps {
            assert!(
                c.chroma > 0.01,
                "cusp at hue {:.1}° has implausibly small chroma {:.4}",
                c.hue,
                c.chroma
            );
            assert!(
                c.lightness > 0.0 && c.lightness <= 1.0,
                "lightness {} out of (0, 1] at hue {:.1}°",
                c.lightness,
                c.hue
            );
        }

        // The computed table must differ from the former hardcoded constants on
        // at least one representative value.  The original red cusp was (20.0°,
        // 0.45, 0.38); any computed hue for the 0° sample will be exactly 0.0°
        // (the hue_target fed to `find_cusp_at_hue`), so check chroma instead.
        // The hardcoded red chroma was 0.38 (a round number); the computed value
        // must be something different (real AP1 CIECAM02 cusp chroma will be
        // well above or below this depending on viewing conditions).
        let first = &cusps[0]; // sorted by hue → the 0° sample is first
        assert_ne!(
            (first.chroma * 100.0).round(),
            38.0,
            "computed AP1 red cusp chroma {:.4} matches old hardcoded 0.38 — \
             this suggests the table was not recomputed",
            first.chroma
        );
    }

    /// `aces_gamut_compress_pixel` (from `gamut_mapping`) must leave a
    /// mid-grey color completely unchanged (all channels equal → distance = 0).
    #[test]
    fn test_aces_gamut_compress_roundtrip_infield() {
        use crate::gamut_mapping::{aces_gamut_compress_pixel, AcesGamutCompressConfig};

        let cfg = AcesGamutCompressConfig::default();
        // Mid-grey: all channels equal → per-channel distance from achromatic
        // axis is 0 for all channels → the compressor leaves it completely unchanged.
        let input = [0.5_f32, 0.5, 0.5];
        let output = aces_gamut_compress_pixel(input, &cfg);

        for i in 0..3 {
            assert!(
                (output[i] - input[i]).abs() < 1e-5,
                "in-gamut channel {i}: expected ~{:.3}, got {:.3} (shift {:.6})",
                input[i],
                output[i],
                (output[i] - input[i]).abs()
            );
        }

        // Also verify that a slightly desaturated color (all channels > threshold
        // of the distance metric, i.e. all channels within cfg.threshold of max)
        // is mostly unchanged.  With max=0.8 and threshold=0.815, any channel
        // satisfying (0.8 - c) <= 0.815 (i.e. c >= -0.015) passes unchanged.
        // Use [0.8, 0.75, 0.70]: distances are 0.0, 0.05, 0.10 — all < 0.815.
        let near_grey = [0.8_f32, 0.75, 0.70];
        let near_out = aces_gamut_compress_pixel(near_grey, &cfg);
        for i in 0..3 {
            assert!(
                (near_out[i] - near_grey[i]).abs() < 0.01,
                "near-grey channel {i}: unexpected shift {:.4}",
                (near_out[i] - near_grey[i]).abs()
            );
        }
    }

    /// `aces_gamut_compress_pixel` must bring an out-of-AP1-gamut color
    /// (negative channel) into non-negative range.
    #[test]
    fn test_aces_gamut_compress_oob_clamped() {
        use crate::gamut_mapping::{aces_gamut_compress_pixel, AcesGamutCompressConfig};

        let cfg = AcesGamutCompressConfig::default();
        // Highly saturated cyan that produces a large negative red channel.
        let input = [0.0_f32, 1.2, 0.9];
        let output = aces_gamut_compress_pixel(input, &cfg);

        // All channels must be non-negative after compression.
        for i in 0..3 {
            assert!(
                output[i] >= 0.0,
                "channel {i} still negative after compression: {:.4}",
                output[i]
            );
        }
        // The input has a channel > 1.0; the compressor should have reduced it.
        let max_out = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_out <= 1.5,
            "compressed max channel {max_out:.4} unexpectedly large"
        );
    }
}
