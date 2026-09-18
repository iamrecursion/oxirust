//! Gamut boundary analysis and out-of-gamut pixel detection.

/// Identifies which color gamut boundary to check against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GamutBoundary {
    /// ITU-R BT.709 / sRGB gamut (standard HD).
    Rec709,
    /// DCI-P3 / Display P3 gamut (digital cinema / consumer HDR).
    P3,
    /// ITU-R BT.2020 gamut (wide-color Ultra-HD).
    Rec2020,
}

impl GamutBoundary {
    /// Returns the approximate coverage percentage of this gamut relative to Rec2020.
    ///
    /// Values are based on CIE 1931 xy chromaticity area ratios.
    #[must_use]
    pub fn coverage_pct_of_rec2020(&self) -> f64 {
        match self {
            Self::Rec709 => 35.9,
            Self::P3 => 53.6,
            Self::Rec2020 => 100.0,
        }
    }

    /// Rec2020 primary chromaticities (Rx,Ry, Gx,Gy, Bx,By).
    #[must_use]
    pub fn primaries_xy(&self) -> [(f64, f64); 3] {
        match self {
            Self::Rec709 => [(0.640, 0.330), (0.300, 0.600), (0.150, 0.060)],
            Self::P3 => [(0.680, 0.320), (0.265, 0.690), (0.150, 0.060)],
            Self::Rec2020 => [(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)],
        }
    }
}

/// A pixel that lies outside a given gamut boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct OutOfGamutPixel {
    /// Pixel column.
    pub x: u32,
    /// Pixel row.
    pub y: u32,
    /// Normalised linear RGB (0.0–1.0 range, negative or >1.0 = out of gamut).
    pub rgb: [f64; 3],
    /// The gamut this pixel violates.
    pub boundary: GamutBoundary,
}

impl OutOfGamutPixel {
    /// Creates a new `OutOfGamutPixel`.
    #[must_use]
    pub fn new(x: u32, y: u32, rgb: [f64; 3], boundary: GamutBoundary) -> Self {
        Self {
            x,
            y,
            rgb,
            boundary,
        }
    }

    /// Clips the RGB values to the [0.0, 1.0] cube (hard clip).
    ///
    /// Returns a new RGB array with each channel clamped.
    #[must_use]
    pub fn clip_to_gamut(&self) -> [f64; 3] {
        [
            self.rgb[0].clamp(0.0, 1.0),
            self.rgb[1].clamp(0.0, 1.0),
            self.rgb[2].clamp(0.0, 1.0),
        ]
    }

    /// Returns `true` if any channel is negative or exceeds 1.0.
    #[must_use]
    pub fn is_out_of_gamut(&self) -> bool {
        self.rgb.iter().any(|&c| !(0.0..=1.0).contains(&c))
    }

    /// Excess magnitude: sum of out-of-range distances.
    #[must_use]
    pub fn excess_magnitude(&self) -> f64 {
        self.rgb
            .iter()
            .map(|&c| {
                if c < 0.0 {
                    -c
                } else if c > 1.0 {
                    c - 1.0
                } else {
                    0.0
                }
            })
            .sum()
    }
}

/// Analyses a frame of pixels for gamut violations.
#[derive(Debug, Clone)]
pub struct GamutChecker {
    boundary: GamutBoundary,
    /// Tolerance — values within ±tolerance of the boundary are not flagged.
    pub tolerance: f64,
}

impl GamutChecker {
    /// Creates a new `GamutChecker` for the given boundary with zero tolerance.
    #[must_use]
    pub fn new(boundary: GamutBoundary) -> Self {
        Self {
            boundary,
            tolerance: 0.0,
        }
    }

    /// Creates a checker with a small tolerance to avoid false positives from rounding.
    #[must_use]
    pub fn with_tolerance(boundary: GamutBoundary, tolerance: f64) -> Self {
        Self {
            boundary,
            tolerance,
        }
    }

    /// Returns the gamut boundary being checked.
    #[must_use]
    pub fn boundary(&self) -> GamutBoundary {
        self.boundary
    }

    /// Analyses a flat slice of interleaved RGB pixels (row-major, 3 `f64` per pixel).
    ///
    /// `width` and `height` describe the frame dimensions.
    /// Returns a list of out-of-gamut pixels.
    #[must_use]
    pub fn analyze_frame(&self, pixels: &[f64], width: u32, height: u32) -> Vec<OutOfGamutPixel> {
        let mut results = Vec::new();
        let pixel_count = (width * height) as usize;
        let limit = pixel_count.min(pixels.len() / 3);
        let tol = self.tolerance;

        for idx in 0..limit {
            let base = idx * 3;
            let r = pixels[base];
            let g = pixels[base + 1];
            let b = pixels[base + 2];
            let out =
                r < -tol || r > 1.0 + tol || g < -tol || g > 1.0 + tol || b < -tol || b > 1.0 + tol;
            if out {
                let px = (idx as u32) % width;
                let py = (idx as u32) / width;
                results.push(OutOfGamutPixel::new(px, py, [r, g, b], self.boundary));
            }
        }
        results
    }

    /// Returns the percentage of pixels that are outside the gamut (0.0–100.0).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn out_of_gamut_pct(&self, pixels: &[f64], width: u32, height: u32) -> f64 {
        let total = (width * height) as usize;
        if total == 0 {
            return 0.0;
        }
        let oog = self.analyze_frame(pixels, width, height).len();
        (oog as f64 / total as f64) * 100.0
    }

    /// Produce a per-pixel gamut boundary distance map for the given frame.
    ///
    /// Each pixel is classified as:
    /// - In-gamut: `delta_e_to_boundary = 0.0`, `in_gamut = true`.
    /// - Out-of-gamut: `in_gamut = false`, `delta_e_to_boundary` = approximate
    ///   Euclidean distance in linear RGB space from the pixel to the nearest
    ///   in-gamut boundary point (computed via clamp-then-distance).
    ///
    /// # Arguments
    ///
    /// * `pixels` – Flat interleaved RGB slice, 3 `f64` per pixel, row-major.
    /// * `width` – Frame width in pixels.
    /// * `height` – Frame height in pixels.
    #[must_use]
    pub fn visualize_gamut_boundary(
        &self,
        pixels: &[f64],
        width: u32,
        height: u32,
    ) -> GamutVisualization {
        let pixel_count = (width * height) as usize;
        let limit = pixel_count.min(pixels.len() / 3);
        let tol = self.tolerance;

        let mut result_pixels = Vec::with_capacity(limit);

        for idx in 0..limit {
            let base = idx * 3;
            let r = pixels[base];
            let g = pixels[base + 1];
            let b = pixels[base + 2];

            let out_of_gamut =
                r < -tol || r > 1.0 + tol || g < -tol || g > 1.0 + tol || b < -tol || b > 1.0 + tol;

            let delta_e = if out_of_gamut {
                // Approximate delta-E to gamut boundary: clamp RGB to [0,1] then
                // compute the Euclidean RGB-space distance to that boundary point.
                let r_clamped = r.clamp(0.0, 1.0);
                let g_clamped = g.clamp(0.0, 1.0);
                let b_clamped = b.clamp(0.0, 1.0);
                let dr = r - r_clamped;
                let dg = g - g_clamped;
                let db = b - b_clamped;
                // Scale to a perceptual-like range (×100 to approximate Lab delta-E).
                ((dr * dr + dg * dg + db * db).sqrt() * 100.0).max(0.0)
            } else {
                0.0
            };

            result_pixels.push(GamutPixel {
                in_gamut: !out_of_gamut,
                delta_e_to_boundary: delta_e,
            });
        }

        GamutVisualization {
            width,
            height,
            pixels: result_pixels,
        }
    }
}

// ── Gamut visualization types ─────────────────────────────────────────────────

/// Per-pixel gamut classification result.
#[derive(Debug, Clone, PartialEq)]
pub struct GamutPixel {
    /// `true` if the pixel lies within the gamut boundary.
    pub in_gamut: bool,
    /// Approximate delta-E distance to the nearest gamut boundary point.
    ///
    /// `0.0` for in-gamut pixels; positive for out-of-gamut pixels.
    pub delta_e_to_boundary: f64,
}

/// Per-pixel gamut distance map for an entire frame.
#[derive(Debug, Clone)]
pub struct GamutVisualization {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// One `GamutPixel` per pixel, in row-major order.
    pub pixels: Vec<GamutPixel>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- GamutBoundary ---

    #[test]
    fn rec2020_coverage_is_100() {
        assert_eq!(GamutBoundary::Rec2020.coverage_pct_of_rec2020(), 100.0);
    }

    #[test]
    fn rec709_coverage_less_than_p3() {
        assert!(
            GamutBoundary::Rec709.coverage_pct_of_rec2020()
                < GamutBoundary::P3.coverage_pct_of_rec2020()
        );
    }

    #[test]
    fn p3_coverage_less_than_rec2020() {
        assert!(
            GamutBoundary::P3.coverage_pct_of_rec2020()
                < GamutBoundary::Rec2020.coverage_pct_of_rec2020()
        );
    }

    #[test]
    fn primaries_have_three_entries() {
        assert_eq!(GamutBoundary::Rec709.primaries_xy().len(), 3);
        assert_eq!(GamutBoundary::P3.primaries_xy().len(), 3);
        assert_eq!(GamutBoundary::Rec2020.primaries_xy().len(), 3);
    }

    // --- OutOfGamutPixel ---

    #[test]
    fn clip_to_gamut_clamps_channels() {
        let p = OutOfGamutPixel::new(0, 0, [-0.1, 1.2, 0.5], GamutBoundary::Rec709);
        let clipped = p.clip_to_gamut();
        assert_eq!(clipped, [0.0, 1.0, 0.5]);
    }

    #[test]
    fn is_out_of_gamut_negative_channel() {
        let p = OutOfGamutPixel::new(0, 0, [-0.1, 0.5, 0.5], GamutBoundary::P3);
        assert!(p.is_out_of_gamut());
    }

    #[test]
    fn is_out_of_gamut_over_one() {
        let p = OutOfGamutPixel::new(0, 0, [1.1, 0.5, 0.5], GamutBoundary::Rec2020);
        assert!(p.is_out_of_gamut());
    }

    #[test]
    fn in_gamut_pixel_not_flagged() {
        let p = OutOfGamutPixel::new(0, 0, [0.5, 0.5, 0.5], GamutBoundary::Rec709);
        assert!(!p.is_out_of_gamut());
    }

    #[test]
    fn excess_magnitude_correct() {
        let p = OutOfGamutPixel::new(0, 0, [-0.2, 1.3, 0.5], GamutBoundary::Rec709);
        let mag = p.excess_magnitude();
        assert!((mag - 0.5).abs() < 1e-10); // 0.2 + 0.3
    }

    // --- GamutChecker ---

    #[test]
    fn analyze_frame_empty_pixels() {
        let checker = GamutChecker::new(GamutBoundary::Rec709);
        let result = checker.analyze_frame(&[], 0, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn analyze_frame_all_in_gamut() {
        let checker = GamutChecker::new(GamutBoundary::Rec709);
        let pixels = vec![0.5, 0.5, 0.5, 0.1, 0.9, 0.0];
        let result = checker.analyze_frame(&pixels, 2, 1);
        assert!(result.is_empty());
    }

    #[test]
    fn analyze_frame_detects_oog() {
        let checker = GamutChecker::new(GamutBoundary::Rec709);
        let pixels = vec![1.2, 0.5, 0.5]; // one pixel, R > 1
        let result = checker.analyze_frame(&pixels, 1, 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].x, 0);
        assert_eq!(result[0].y, 0);
    }

    #[test]
    fn out_of_gamut_pct_all_in() {
        let checker = GamutChecker::new(GamutBoundary::P3);
        let pixels = vec![0.5, 0.5, 0.5, 0.2, 0.8, 0.3];
        let pct = checker.out_of_gamut_pct(&pixels, 2, 1);
        assert!((pct - 0.0).abs() < 1e-9);
    }

    #[test]
    fn out_of_gamut_pct_half_out() {
        let checker = GamutChecker::new(GamutBoundary::P3);
        // 2 pixels: one in, one out
        let pixels = vec![0.5, 0.5, 0.5, 1.5, 0.5, 0.5];
        let pct = checker.out_of_gamut_pct(&pixels, 2, 1);
        assert!((pct - 50.0).abs() < 1e-9);
    }

    #[test]
    fn tolerance_suppresses_near_boundary() {
        let checker = GamutChecker::with_tolerance(GamutBoundary::Rec709, 0.05);
        // 1.03 is within tolerance of 0.05 above 1.0
        let pixels = vec![1.03, 0.5, 0.5];
        let result = checker.analyze_frame(&pixels, 1, 1);
        assert!(result.is_empty());
    }

    #[test]
    fn checker_boundary_accessor() {
        let checker = GamutChecker::new(GamutBoundary::Rec2020);
        assert_eq!(checker.boundary(), GamutBoundary::Rec2020);
    }

    // ── GamutVisualization tests ───────────────────────────────────────────────

    /// All in-gamut pixels must yield delta_e_to_boundary == 0.0.
    #[test]
    fn test_gamut_visualization_in_gamut_zero() {
        let checker = GamutChecker::new(GamutBoundary::Rec709);
        // 3 fully in-gamut pixels.
        let pixels = vec![
            0.0, 0.0, 0.0, // black
            1.0, 1.0, 1.0, // white
            0.5, 0.25, 0.75, // mid-gray
        ];
        let viz = checker.visualize_gamut_boundary(&pixels, 3, 1);

        assert_eq!(viz.width, 3);
        assert_eq!(viz.height, 1);
        assert_eq!(viz.pixels.len(), 3);

        for (i, px) in viz.pixels.iter().enumerate() {
            assert!(
                px.in_gamut,
                "pixel {i} should be in-gamut but in_gamut = false"
            );
            assert!(
                px.delta_e_to_boundary.abs() < 1e-10,
                "pixel {i} in-gamut delta_e should be 0.0, got {}",
                px.delta_e_to_boundary
            );
        }
    }

    /// Out-of-gamut pixels must have delta_e_to_boundary > 0.0 and in_gamut == false.
    #[test]
    fn test_gamut_visualization_out_of_gamut_positive() {
        let checker = GamutChecker::new(GamutBoundary::Rec709);
        // Two pixels: one in-gamut, one clearly out-of-gamut (R >> 1.0).
        let pixels = vec![
            0.5, 0.5, 0.5, // in gamut
            1.5, 0.5, 0.5, // R = 1.5 → OOG
        ];
        let viz = checker.visualize_gamut_boundary(&pixels, 2, 1);

        // First pixel: in gamut.
        assert!(viz.pixels[0].in_gamut, "pixel 0 should be in-gamut");
        assert!(
            viz.pixels[0].delta_e_to_boundary.abs() < 1e-10,
            "pixel 0 delta_e should be 0"
        );

        // Second pixel: out of gamut.
        assert!(!viz.pixels[1].in_gamut, "pixel 1 should be out-of-gamut");
        assert!(
            viz.pixels[1].delta_e_to_boundary > 0.0,
            "pixel 1 OOG delta_e should be > 0, got {}",
            viz.pixels[1].delta_e_to_boundary
        );

        // The OOG distance for R=1.5 clamped to R=1.0: dr=0.5 → sqrt(0.25)*100=50.
        let expected_delta_e = 0.5_f64 * 100.0; // = 50.0
        assert!(
            (viz.pixels[1].delta_e_to_boundary - expected_delta_e).abs() < 1e-6,
            "expected ~{expected_delta_e}, got {}",
            viz.pixels[1].delta_e_to_boundary
        );
    }

    /// `GamutVisualization` with negative channel values should also be out-of-gamut.
    #[test]
    fn test_gamut_visualization_negative_channel_oog() {
        let checker = GamutChecker::new(GamutBoundary::P3);
        let pixels = vec![-0.2, 0.5, 0.5]; // R < 0
        let viz = checker.visualize_gamut_boundary(&pixels, 1, 1);
        assert!(!viz.pixels[0].in_gamut);
        assert!(viz.pixels[0].delta_e_to_boundary > 0.0);
    }
}
