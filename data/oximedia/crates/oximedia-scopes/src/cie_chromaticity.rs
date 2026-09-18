//! CIE 1931 chromaticity diagram renderer.
//!
//! Renders the classic CIE horseshoe on a black canvas with optional:
//! - Spectral locus outline (the horseshoe boundary of all visible colours)
//! - Gamut triangle for Rec.709 / DCI-P3 / Rec.2020 / ACES colour spaces
//! - D65 white-point marker
//! - Arbitrary (x, y) sample point overlay
//!
//! # Coordinate System
//!
//! CIE xy chromaticity coordinates range roughly:
//! - x: 0.0 … 0.80
//! - y: 0.0 … 0.90
//!
//! The renderer maps this to the pixel canvas so that (0.0, 0.0) appears at
//! the bottom-left (after flipping the Y axis for screen-space).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

// ─── Colour gamut ─────────────────────────────────────────────────────────────

/// Well-known colour gamuts / primaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorGamut {
    /// ITU-R BT.709 (HDTV).
    Rec709,
    /// DCI-P3 (Digital Cinema).
    P3,
    /// ITU-R BT.2020 (UHDTV).
    Rec2020,
    /// ACES AP0 (Academy Color Encoding System).
    Aces,
}

impl ColorGamut {
    /// Returns the CIE xy chromaticity coordinates for the three primaries
    /// `[Red, Green, Blue]`.
    #[must_use]
    pub fn primaries(self) -> [(f32, f32); 3] {
        match self {
            Self::Rec709 => [
                (0.6400, 0.3300), // Red
                (0.3000, 0.6000), // Green
                (0.1500, 0.0600), // Blue
            ],
            Self::P3 => [
                (0.6800, 0.3200), // Red
                (0.2650, 0.6900), // Green
                (0.1500, 0.0600), // Blue
            ],
            Self::Rec2020 => [
                (0.7080, 0.2920), // Red
                (0.1700, 0.7970), // Green
                (0.1310, 0.0460), // Blue
            ],
            Self::Aces => [
                (0.7347, 0.2653),  // Red  (AP0)
                (0.0000, 1.0000),  // Green
                (0.0001, -0.0770), // Blue (clamped during rendering)
            ],
        }
    }

    /// Returns the D65 white point chromaticity `(x, y)` used by this gamut.
    /// ACES uses its own white point (D60 ≈ 0.3217, 0.3290).
    #[must_use]
    pub fn white_point(self) -> (f32, f32) {
        match self {
            Self::Aces => (0.3217, 0.3290),
            _ => (0.3127, 0.3290), // D65
        }
    }
}

// ─── Spectral locus ───────────────────────────────────────────────────────────

/// Approximate CIE 1931 spectral locus xy coordinates for selected wavelengths
/// (380 nm – 700 nm), plus the purple line back to 380 nm.
///
/// Source: CIE 1931 standard observer tables (widely published values).
const SPECTRAL_LOCUS: &[(f32, f32)] = &[
    (0.1741, 0.0050), // 380 nm
    (0.1740, 0.0050), // 385 nm (approx)
    (0.1738, 0.0049), // 390 nm (approx)
    (0.1733, 0.0048), // 395 nm (approx)
    (0.1669, 0.0086), // 400 nm (approx 420 nm in brief table)
    (0.1585, 0.0230), // 430 nm (approx)
    (0.1440, 0.1126), // 460 nm
    (0.1241, 0.2401), // 470 nm (approx)
    (0.0913, 0.3855), // 480 nm (approx)
    (0.0455, 0.4950), // 490 nm (approx)
    (0.0082, 0.5384), // 500 nm
    (0.0139, 0.6475), // 510 nm (approx)
    (0.0743, 0.8338), // 520 nm
    (0.1547, 0.8059), // 530 nm (approx)
    (0.2296, 0.7543), // 540 nm
    (0.3016, 0.6923), // 550 nm (approx)
    (0.3731, 0.6245), // 560 nm
    (0.4441, 0.5547), // 570 nm (approx)
    (0.5125, 0.4866), // 580 nm
    (0.5752, 0.4240), // 590 nm (approx)
    (0.6270, 0.3725), // 600 nm
    (0.6658, 0.3340), // 610 nm (approx)
    (0.6927, 0.3075), // 620 nm
    (0.7079, 0.2920), // 630 nm (approx)
    (0.7270, 0.2730), // 640 nm
    (0.7347, 0.2653), // 650 nm
    (0.7347, 0.2653), // 660 nm
    (0.7347, 0.2653), // 670 nm
    (0.7347, 0.2653), // 680 nm
    (0.7347, 0.2653), // 690 nm
    (0.7347, 0.2653), // 700 nm
];

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for [`CieChromaticityDiagram`].
#[derive(Debug, Clone)]
pub struct CieChromaticityConfig {
    /// Output frame width in pixels.
    pub width: u32,
    /// Output frame height in pixels.
    pub height: u32,
    /// Whether to draw the gamut triangle overlay.
    pub show_gamut: bool,
    /// Which gamut to draw the triangle for.
    pub gamut: ColorGamut,
    /// Whether to draw the spectral locus outline.
    pub show_spectral_locus: bool,
    /// Whether to draw the D65/D60 white-point marker.
    pub show_white_point: bool,
    /// Background RGBA colour (default: opaque black).
    pub background_color: [u8; 4],
}

impl Default for CieChromaticityConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            show_gamut: true,
            gamut: ColorGamut::Rec709,
            show_spectral_locus: true,
            show_white_point: true,
            background_color: [0, 0, 0, 255],
        }
    }
}

// ─── Renderer ────────────────────────────────────────────────────────────────

/// Range of the CIE xy space covered by the diagram.
const CIE_X_MIN: f32 = 0.0;
const CIE_X_MAX: f32 = 0.80;
const CIE_Y_MIN: f32 = 0.0;
const CIE_Y_MAX: f32 = 0.90;

/// CIE 1931 chromaticity diagram renderer.
#[derive(Debug, Clone)]
pub struct CieChromaticityDiagram {
    /// Rendering configuration.
    pub config: CieChromaticityConfig,
}

impl CieChromaticityDiagram {
    /// Creates a new diagram renderer with the given configuration.
    #[must_use]
    pub fn new(config: CieChromaticityConfig) -> Self {
        Self { config }
    }

    /// Maps a CIE (x, y) chromaticity coordinate to pixel (px, py).
    ///
    /// The Y axis is flipped so that higher `y` values appear higher on screen.
    /// Returns `(i32, i32)` so callers can test for out-of-bounds before casting.
    #[must_use]
    pub fn chromaticity_to_pixel(&self, x: f32, y: f32) -> (i32, i32) {
        let w = self.config.width as f32;
        let h = self.config.height as f32;
        let tx = (x - CIE_X_MIN) / (CIE_X_MAX - CIE_X_MIN);
        let ty = (y - CIE_Y_MIN) / (CIE_Y_MAX - CIE_Y_MIN);
        let px = (tx * (w - 1.0)).round() as i32;
        let py = ((1.0 - ty) * (h - 1.0)).round() as i32; // flip Y
        (px, py)
    }

    /// Renders the chromaticity diagram.
    ///
    /// `xy_samples` is a slice of (x, y) chromaticity coordinates that will be
    /// plotted as bright yellow dots on the diagram.
    ///
    /// Returns `width × height × 4` bytes (RGBA, row-major, top-left origin).
    #[must_use]
    pub fn render(&self, xy_samples: &[(f32, f32)]) -> Vec<u8> {
        let w = self.config.width as usize;
        let h = self.config.height as usize;
        let bg = self.config.background_color;
        let mut pixels = vec![bg[0], bg[1], bg[2], bg[3]]
            .into_iter()
            .cycle()
            .take(w * h * 4)
            .collect::<Vec<u8>>();

        // ── Spectral locus ─────────────────────────────────────────────────
        if self.config.show_spectral_locus {
            // Draw line segments connecting consecutive locus points
            for pair in SPECTRAL_LOCUS.windows(2) {
                let (x0, y0) = pair[0];
                let (x1, y1) = pair[1];
                self.draw_line(&mut pixels, w, h, x0, y0, x1, y1, [200, 200, 200, 255]);
            }
            // Close the purple line (700 nm back to 380 nm)
            if let (Some(&last), Some(&first)) = (SPECTRAL_LOCUS.last(), SPECTRAL_LOCUS.first()) {
                self.draw_line(
                    &mut pixels,
                    w,
                    h,
                    last.0,
                    last.1,
                    first.0,
                    first.1,
                    [160, 50, 160, 255],
                );
            }
        }

        // ── Gamut triangle ─────────────────────────────────────────────────
        if self.config.show_gamut {
            let primaries = self.config.gamut.primaries();
            let colour = [255, 255, 255, 255];
            let (rx, ry) = primaries[0];
            let (gx, gy) = primaries[1];
            let (bx, by) = primaries[2];
            self.draw_line(&mut pixels, w, h, rx, ry, gx, gy, colour);
            self.draw_line(&mut pixels, w, h, gx, gy, bx, by, colour);
            self.draw_line(&mut pixels, w, h, bx, by, rx, ry, colour);
        }

        // ── White-point marker ─────────────────────────────────────────────
        if self.config.show_white_point {
            let (wx, wy) = self.config.gamut.white_point();
            let (px, py) = self.chromaticity_to_pixel(wx, wy);
            // Draw a small + cross
            let cross_r: i32 = 4;
            let wp_colour = [255, 255, 200, 255];
            for d in -cross_r..=cross_r {
                self.put_pixel_i32(&mut pixels, w, h, px + d, py, wp_colour);
                self.put_pixel_i32(&mut pixels, w, h, px, py + d, wp_colour);
            }
        }

        // ── Sample points ──────────────────────────────────────────────────
        let sample_colour = [255, 220, 0, 255]; // yellow
        for &(sx, sy) in xy_samples {
            let (px, py) = self.chromaticity_to_pixel(sx, sy);
            // 3×3 dot
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    self.put_pixel_i32(&mut pixels, w, h, px + dx, py + dy, sample_colour);
                }
            }
        }

        pixels
    }

    // ─── Internal helpers ─────────────────────────────────────────────────

    /// Draws a line between two CIE (x,y) points using Bresenham's algorithm.
    fn draw_line(
        &self,
        pixels: &mut Vec<u8>,
        w: usize,
        h: usize,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
        colour: [u8; 4],
    ) {
        let (px0, py0) = self.chromaticity_to_pixel(x0, y0);
        let (px1, py1) = self.chromaticity_to_pixel(x1, y1);
        bresenham_line(pixels, w, h, px0, py0, px1, py1, colour);
    }

    /// Writes an RGBA colour to a pixel position, bounds-checked.
    fn put_pixel_i32(
        &self,
        pixels: &mut Vec<u8>,
        w: usize,
        h: usize,
        px: i32,
        py: i32,
        colour: [u8; 4],
    ) {
        if px < 0 || py < 0 || px >= w as i32 || py >= h as i32 {
            return;
        }
        let idx = (py as usize * w + px as usize) * 4;
        pixels[idx] = colour[0];
        pixels[idx + 1] = colour[1];
        pixels[idx + 2] = colour[2];
        pixels[idx + 3] = colour[3];
    }
}

// ─── Bresenham line ───────────────────────────────────────────────────────────

fn bresenham_line(
    pixels: &mut Vec<u8>,
    w: usize,
    h: usize,
    mut x0: i32,
    mut y0: i32,
    x1: i32,
    y1: i32,
    colour: [u8; 4],
) {
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx = if x0 < x1 { 1i32 } else { -1i32 };
    let sy = if y0 < y1 { 1i32 } else { -1i32 };
    let mut err = dx - dy;

    loop {
        if x0 >= 0 && y0 >= 0 && (x0 as usize) < w && (y0 as usize) < h {
            let idx = (y0 as usize * w + x0 as usize) * 4;
            pixels[idx] = colour[0];
            pixels[idx + 1] = colour[1];
            pixels[idx + 2] = colour[2];
            pixels[idx + 3] = colour[3];
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x0 += sx;
        }
        if e2 < dx {
            err += dx;
            y0 += sy;
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_diagram() -> CieChromaticityDiagram {
        CieChromaticityDiagram::new(CieChromaticityConfig::default())
    }

    // ── Output dimensions ─────────────────────────────────────────────────

    #[test]
    fn test_render_correct_size_default() {
        let d = default_diagram();
        let out = d.render(&[]);
        assert_eq!(out.len(), 512 * 512 * 4);
    }

    #[test]
    fn test_render_correct_size_custom() {
        let cfg = CieChromaticityConfig {
            width: 200,
            height: 150,
            ..Default::default()
        };
        let d = CieChromaticityDiagram::new(cfg);
        let out = d.render(&[]);
        assert_eq!(out.len(), 200 * 150 * 4);
    }

    // ── Spectral locus ────────────────────────────────────────────────────

    #[test]
    fn test_spectral_locus_produces_nonblack_pixels() {
        let cfg = CieChromaticityConfig {
            show_spectral_locus: true,
            show_gamut: false,
            show_white_point: false,
            ..Default::default()
        };
        let d = CieChromaticityDiagram::new(cfg);
        let out = d.render(&[]);
        let nonblack = out
            .chunks_exact(4)
            .filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0)
            .count();
        assert!(
            nonblack > 50,
            "spectral locus should render many non-black pixels, got {nonblack}"
        );
    }

    #[test]
    fn test_no_spectral_locus_all_background() {
        let cfg = CieChromaticityConfig {
            show_spectral_locus: false,
            show_gamut: false,
            show_white_point: false,
            background_color: [0, 0, 0, 255],
            ..Default::default()
        };
        let d = CieChromaticityDiagram::new(cfg);
        let out = d.render(&[]);
        let nonblack = out
            .chunks_exact(4)
            .filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0)
            .count();
        assert_eq!(nonblack, 0, "no locus/gamut/wp → all background");
    }

    // ── Gamut triangle ────────────────────────────────────────────────────

    #[test]
    fn test_gamut_triangle_visible() {
        let cfg = CieChromaticityConfig {
            show_spectral_locus: false,
            show_gamut: true,
            show_white_point: false,
            gamut: ColorGamut::Rec709,
            ..Default::default()
        };
        let d = CieChromaticityDiagram::new(cfg);
        let out = d.render(&[]);
        let white = out
            .chunks_exact(4)
            .filter(|p| p[0] > 200 && p[1] > 200 && p[2] > 200)
            .count();
        assert!(
            white > 10,
            "gamut triangle should draw white lines, got {white}"
        );
    }

    #[test]
    fn test_different_gamuts_produce_different_outputs() {
        let make = |g: ColorGamut| {
            let cfg = CieChromaticityConfig {
                show_spectral_locus: false,
                show_gamut: true,
                show_white_point: false,
                gamut: g,
                width: 128,
                height: 128,
                background_color: [0, 0, 0, 255],
            };
            CieChromaticityDiagram::new(cfg).render(&[])
        };
        let rec709 = make(ColorGamut::Rec709);
        let rec2020 = make(ColorGamut::Rec2020);
        assert_ne!(
            rec709, rec2020,
            "different gamuts should produce different diagrams"
        );
    }

    // ── White point ───────────────────────────────────────────────────────

    #[test]
    fn test_white_point_marker_nonblack() {
        let cfg = CieChromaticityConfig {
            show_spectral_locus: false,
            show_gamut: false,
            show_white_point: true,
            ..Default::default()
        };
        let d = CieChromaticityDiagram::new(cfg);
        let out = d.render(&[]);
        let (wx, wy) = (0.3127f32, 0.3290f32);
        let (px, py) = d.chromaticity_to_pixel(wx, wy);
        let idx = (py as usize * 512 + px as usize) * 4;
        let any = out[idx] > 0 || out[idx + 1] > 0 || out[idx + 2] > 0;
        assert!(any, "white point pixel should be non-zero");
    }

    // ── Sample points ──────────────────────────────────────────────────────

    #[test]
    fn test_sample_points_plotted() {
        let cfg = CieChromaticityConfig {
            show_spectral_locus: false,
            show_gamut: false,
            show_white_point: false,
            ..Default::default()
        };
        let d = CieChromaticityDiagram::new(cfg);
        // D65 white point as a sample
        let out = d.render(&[(0.3127, 0.3290)]);
        let nonblack = out
            .chunks_exact(4)
            .filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0)
            .count();
        assert!(nonblack >= 1, "sample dot should be visible");
    }

    // ── chromaticity_to_pixel ──────────────────────────────────────────────

    #[test]
    fn test_chromaticity_to_pixel_origin() {
        let d = default_diagram();
        let (px, py) = d.chromaticity_to_pixel(CIE_X_MIN, CIE_Y_MIN);
        // Bottom-left in CIE = bottom-left on canvas but Y is flipped → py = height-1
        assert_eq!(px, 0);
        assert_eq!(py, 511); // height - 1
    }

    #[test]
    fn test_chromaticity_to_pixel_top_right() {
        let d = default_diagram();
        let (px, py) = d.chromaticity_to_pixel(CIE_X_MAX, CIE_Y_MAX);
        assert_eq!(px, 511); // width - 1
        assert_eq!(py, 0); // top of canvas
    }

    // ── ColorGamut primaries ──────────────────────────────────────────────

    #[test]
    fn test_rec709_primaries() {
        let p = ColorGamut::Rec709.primaries();
        // Red primary
        assert!((p[0].0 - 0.640).abs() < 0.001);
        assert!((p[0].1 - 0.330).abs() < 0.001);
    }

    #[test]
    fn test_rec2020_primaries_wider_than_rec709() {
        let r709 = ColorGamut::Rec709.primaries()[0];
        let r2020 = ColorGamut::Rec2020.primaries()[0];
        // Rec.2020 red has higher x → further right on the diagram
        assert!(r2020.0 > r709.0);
    }

    #[test]
    fn test_p3_primaries() {
        let p = ColorGamut::P3.primaries();
        assert!((p[0].0 - 0.680).abs() < 0.001); // Red x
        assert!((p[1].1 - 0.690).abs() < 0.001); // Green y
    }
}
