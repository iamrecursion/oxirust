#![allow(dead_code)]
//! Color gamut scope for visualizing how pixel colors map within a target gamut.
//!
//! This module renders a 2-D chromaticity-based gamut scope that plots each
//! pixel's color onto a CIE xy (or u'v') diagram with the gamut triangle
//! overlay for Rec.709, Rec.2020, or DCI-P3. Pixels falling outside the
//! target gamut are highlighted in a warning color.

/// Target color gamut for the scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetGamut {
    /// ITU-R BT.709 (standard HD).
    Rec709,
    /// ITU-R BT.2020 (UHD / HDR).
    Rec2020,
    /// DCI-P3 (Digital Cinema).
    DciP3,
    /// sRGB (identical primaries to Rec.709 but different transfer).
    Srgb,
}

/// Diagram type used for the gamut scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagramKind {
    /// CIE 1931 xy chromaticity diagram.
    CieXy,
    /// CIE 1976 u'v' uniform chromaticity scale diagram.
    CieUpVp,
}

/// Configuration for the gamut scope renderer.
#[derive(Debug, Clone)]
pub struct GamutScopeConfig {
    /// Output scope image width in pixels.
    pub width: u32,
    /// Output scope image height in pixels.
    pub height: u32,
    /// Target gamut to display.
    pub target_gamut: TargetGamut,
    /// Diagram kind (CIE xy or u'v').
    pub diagram_kind: DiagramKind,
    /// Whether to draw the gamut triangle boundary.
    pub show_gamut_triangle: bool,
    /// Whether to highlight out-of-gamut pixels.
    pub highlight_out_of_gamut: bool,
    /// Intensity of pixel dots on the scope (0.0 - 1.0).
    pub dot_intensity: f64,
    /// Whether to draw the spectral locus outline.
    pub show_spectral_locus: bool,
    /// Background brightness (0 = black, 255 = white).
    pub background_level: u8,
}

impl Default for GamutScopeConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            target_gamut: TargetGamut::Rec709,
            diagram_kind: DiagramKind::CieXy,
            show_gamut_triangle: true,
            highlight_out_of_gamut: true,
            dot_intensity: 0.8,
            show_spectral_locus: true,
            background_level: 16,
        }
    }
}

/// CIE xy chromaticity coordinate.
#[derive(Debug, Clone, Copy)]
pub struct ChromaXy {
    /// x coordinate.
    pub x: f64,
    /// y coordinate.
    pub y: f64,
}

impl ChromaXy {
    /// Create a new chromaticity coordinate.
    #[must_use]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// A gamut triangle defined by three primary chromaticities and a white point.
#[derive(Debug, Clone, Copy)]
pub struct GamutTriangle {
    /// Red primary.
    pub red: ChromaXy,
    /// Green primary.
    pub green: ChromaXy,
    /// Blue primary.
    pub blue: ChromaXy,
    /// White point (D65 typically).
    pub white: ChromaXy,
}

impl GamutTriangle {
    /// Get the Rec.709 gamut triangle.
    #[must_use]
    pub fn rec709() -> Self {
        Self {
            red: ChromaXy::new(0.64, 0.33),
            green: ChromaXy::new(0.30, 0.60),
            blue: ChromaXy::new(0.15, 0.06),
            white: ChromaXy::new(0.3127, 0.3290),
        }
    }

    /// Get the Rec.2020 gamut triangle.
    #[must_use]
    pub fn rec2020() -> Self {
        Self {
            red: ChromaXy::new(0.708, 0.292),
            green: ChromaXy::new(0.170, 0.797),
            blue: ChromaXy::new(0.131, 0.046),
            white: ChromaXy::new(0.3127, 0.3290),
        }
    }

    /// Get the DCI-P3 gamut triangle.
    #[must_use]
    pub fn dci_p3() -> Self {
        Self {
            red: ChromaXy::new(0.680, 0.320),
            green: ChromaXy::new(0.265, 0.690),
            blue: ChromaXy::new(0.150, 0.060),
            white: ChromaXy::new(0.314, 0.351),
        }
    }

    /// Get the triangle for a given target gamut.
    #[must_use]
    pub fn for_gamut(gamut: TargetGamut) -> Self {
        match gamut {
            TargetGamut::Rec709 | TargetGamut::Srgb => Self::rec709(),
            TargetGamut::Rec2020 => Self::rec2020(),
            TargetGamut::DciP3 => Self::dci_p3(),
        }
    }

    /// Test whether a chromaticity point is inside this gamut triangle
    /// using barycentric coordinates.
    #[must_use]
    pub fn contains(&self, p: &ChromaXy) -> bool {
        let d1 = sign(p, &self.red, &self.green);
        let d2 = sign(p, &self.green, &self.blue);
        let d3 = sign(p, &self.blue, &self.red);

        let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
        let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);

        !(has_neg && has_pos)
    }

    /// Compute the area of this gamut triangle (in xy units).
    #[must_use]
    pub fn area(&self) -> f64 {
        0.5 * ((self.green.x - self.red.x) * (self.blue.y - self.red.y)
            - (self.blue.x - self.red.x) * (self.green.y - self.red.y))
            .abs()
    }
}

/// Helper for triangle containment test (2-D cross product sign).
fn sign(p: &ChromaXy, a: &ChromaXy, b: &ChromaXy) -> f64 {
    (p.x - b.x) * (a.y - b.y) - (a.x - b.x) * (p.y - b.y)
}

/// Convert linear RGB (0.0 - 1.0) to CIE XYZ using the Rec.709 matrix.
#[must_use]
pub fn linear_rgb_to_xyz(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
    (x, y, z)
}

/// Convert CIE XYZ to CIE xy chromaticity.
#[must_use]
pub fn xyz_to_xy(x: f64, y: f64, z: f64) -> ChromaXy {
    let sum = x + y + z;
    if sum <= 0.0 {
        return ChromaXy::new(0.3127, 0.3290); // D65 white for black pixels
    }
    ChromaXy::new(x / sum, y / sum)
}

/// Convert CIE XYZ to CIE 1976 u'v' coordinates.
#[must_use]
pub fn xyz_to_upvp(x: f64, y: f64, z: f64) -> (f64, f64) {
    let denom = x + 15.0 * y + 3.0 * z;
    if denom <= 0.0 {
        return (0.1978, 0.4683); // D65
    }
    let up = 4.0 * x / denom;
    let vp = 9.0 * y / denom;
    (up, vp)
}

/// Remove sRGB gamma to get linear light.
#[must_use]
pub fn srgb_to_linear(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Result of analyzing a frame against a gamut.
#[derive(Debug, Clone)]
pub struct GamutAnalysis {
    /// Total number of pixels analyzed.
    pub total_pixels: u64,
    /// Number of pixels outside the target gamut.
    pub out_of_gamut_pixels: u64,
    /// Percentage of pixels outside gamut (0.0 - 100.0).
    pub out_of_gamut_pct: f64,
    /// Average chromaticity x.
    pub avg_x: f64,
    /// Average chromaticity y.
    pub avg_y: f64,
}

/// Analyze an RGB24 frame against a target gamut.
///
/// * `frame` — RGB24 pixel data (3 bytes per pixel: R, G, B)
/// * `width` — frame width
/// * `height` — frame height
/// * `gamut` — target gamut to test against
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn analyze_gamut(frame: &[u8], width: u32, height: u32, gamut: TargetGamut) -> GamutAnalysis {
    let triangle = GamutTriangle::for_gamut(gamut);
    let num_pixels = (width as u64) * (height as u64);
    let expected_len = num_pixels as usize * 3;

    if frame.len() < expected_len || num_pixels == 0 {
        return GamutAnalysis {
            total_pixels: num_pixels,
            out_of_gamut_pixels: 0,
            out_of_gamut_pct: 0.0,
            avg_x: 0.3127,
            avg_y: 0.3290,
        };
    }

    let mut out_count = 0_u64;
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;

    for i in 0..num_pixels as usize {
        let r_lin = srgb_to_linear(f64::from(frame[i * 3]) / 255.0);
        let g_lin = srgb_to_linear(f64::from(frame[i * 3 + 1]) / 255.0);
        let b_lin = srgb_to_linear(f64::from(frame[i * 3 + 2]) / 255.0);

        let (x, y, z) = linear_rgb_to_xyz(r_lin, g_lin, b_lin);
        let chroma = xyz_to_xy(x, y, z);

        sum_x += chroma.x;
        sum_y += chroma.y;

        if !triangle.contains(&chroma) {
            out_count += 1;
        }
    }

    let n = num_pixels as f64;
    GamutAnalysis {
        total_pixels: num_pixels,
        out_of_gamut_pixels: out_count,
        out_of_gamut_pct: if num_pixels > 0 {
            (out_count as f64 / n) * 100.0
        } else {
            0.0
        },
        avg_x: sum_x / n,
        avg_y: sum_y / n,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CIE diagram renderer with gamut triangle overlays
// ─────────────────────────────────────────────────────────────────────────────

/// RGBA color for a gamut triangle outline (Rec.709 = green, P3 = yellow, 2020 = red, sRGB = cyan).
fn gamut_line_color(gamut: TargetGamut) -> [u8; 4] {
    match gamut {
        TargetGamut::Rec709 => [0, 220, 0, 220],
        TargetGamut::Srgb => [0, 220, 220, 200],
        TargetGamut::DciP3 => [220, 220, 0, 200],
        TargetGamut::Rec2020 => [220, 0, 0, 200],
    }
}

/// Map a CIE xy chromaticity coordinate to a pixel position within the scope.
///
/// The visible CIE chromaticity locus fits roughly in x ∈ [0.0, 0.8], y ∈ [0.0, 0.9].
/// We map that range to the output image dimensions with a small margin.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
fn xy_to_pixel(x: f64, y: f64, width: u32, height: u32) -> (i32, i32) {
    // CIE locus viewport: x ∈ [0.0, 0.8], y ∈ [0.0, 0.9]
    let margin = 0.05;
    let x_min = -margin;
    let x_max = 0.80 + margin;
    let y_min = -margin;
    let y_max = 0.90 + margin;

    let px = ((x - x_min) / (x_max - x_min) * (width as f64 - 1.0)).round() as i32;
    // Y axis is inverted: higher y → lower pixel row
    let py = ((1.0 - (y - y_min) / (y_max - y_min)) * (height as f64 - 1.0)).round() as i32;
    (px, py)
}

/// Draw a Bresenham line between two CIE xy coordinates onto an RGBA buffer.
#[allow(clippy::cast_possible_wrap)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
fn draw_chroma_line(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    ax: f64,
    ay: f64,
    bx: f64,
    by: f64,
    color: [u8; 4],
) {
    let w = width as usize;
    let h = height as usize;
    let (mut x0, mut y0) = xy_to_pixel(ax, ay, width, height);
    let (x1, y1) = xy_to_pixel(bx, by, width, height);

    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx: i32 = if x0 < x1 { 1 } else { -1 };
    let sy: i32 = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        if x0 >= 0 && y0 >= 0 && (x0 as usize) < w && (y0 as usize) < h {
            let idx = (y0 as usize * w + x0 as usize) * 4;
            let a = color[3] as f32 / 255.0;
            let ia = 1.0 - a;
            rgba[idx] = (color[0] as f32 * a + rgba[idx] as f32 * ia) as u8;
            rgba[idx + 1] = (color[1] as f32 * a + rgba[idx + 1] as f32 * ia) as u8;
            rgba[idx + 2] = (color[2] as f32 * a + rgba[idx + 2] as f32 * ia) as u8;
            rgba[idx + 3] = 255;
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

/// Draw a gamut triangle outline for a given `TargetGamut` onto an RGBA buffer.
///
/// The triangle is drawn as three connected line segments between the red, green,
/// and blue primaries of the target color space in CIE xy space.
///
/// # Arguments
///
/// * `rgba` — RGBA pixel buffer (length must be `width * height * 4`).
/// * `width` / `height` — buffer dimensions.
/// * `gamut` — gamut whose triangle to draw.
/// * `color` — optional RGBA line color; if `None`, the default per-gamut color is used.
pub fn draw_gamut_triangle(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    gamut: TargetGamut,
    color: Option<[u8; 4]>,
) {
    let tri = GamutTriangle::for_gamut(gamut);
    let col = color.unwrap_or_else(|| gamut_line_color(gamut));

    // Draw R→G, G→B, B→R edges
    draw_chroma_line(
        rgba,
        width,
        height,
        tri.red.x,
        tri.red.y,
        tri.green.x,
        tri.green.y,
        col,
    );
    draw_chroma_line(
        rgba,
        width,
        height,
        tri.green.x,
        tri.green.y,
        tri.blue.x,
        tri.blue.y,
        col,
    );
    draw_chroma_line(
        rgba, width, height, tri.blue.x, tri.blue.y, tri.red.x, tri.red.y, col,
    );
}

/// Render a CIE 1931 xy gamut scope with triangle overlays for multiple gamuts.
///
/// The output is an RGBA image of size `config.width × config.height` containing:
/// - Scatter-plotted pixel chromaticities sampled from `frame`.
/// - Gamut triangle outlines for each gamut in `overlays`.
/// - Optional spectral locus (horizon curve).
///
/// # Arguments
///
/// * `frame` — RGB24 frame data (3 bytes per pixel).
/// * `width` / `height` — frame dimensions.
/// * `config` — scope configuration.
/// * `overlays` — list of gamuts whose triangles to overlay (may be empty).
///
/// # Errors
///
/// Returns an error if `frame.len() < width * height * 3`.
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn render_gamut_scope(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &GamutScopeConfig,
    overlays: &[TargetGamut],
) -> oximedia_core::OxiResult<Vec<u8>> {
    let expected = (width as usize) * (height as usize) * 3;
    if frame.len() < expected {
        return Err(oximedia_core::OxiError::InvalidData(format!(
            "Frame too small: need {expected}, got {}",
            frame.len()
        )));
    }

    let out_w = config.width as usize;
    let out_h = config.height as usize;
    // Dark background
    let bg = config.background_level;
    let mut rgba: Vec<u8> = (0..out_w * out_h)
        .flat_map(|_| [bg, bg, bg, 255u8])
        .collect();

    // Plot pixel chromaticities
    let num_pixels = (width as usize) * (height as usize);
    let intensity = (config.dot_intensity.clamp(0.0, 1.0) * 200.0) as u8;
    let oob_color: [u8; 4] = [220, 60, 60, intensity];
    let in_color: [u8; 4] = [60, 200, 100, intensity];

    let triangle = if config.highlight_out_of_gamut {
        Some(GamutTriangle::for_gamut(config.target_gamut))
    } else {
        None
    };

    for i in 0..num_pixels {
        let r_lin = srgb_to_linear(f64::from(frame[i * 3]) / 255.0);
        let g_lin = srgb_to_linear(f64::from(frame[i * 3 + 1]) / 255.0);
        let b_lin = srgb_to_linear(f64::from(frame[i * 3 + 2]) / 255.0);

        let (x, y, z) = linear_rgb_to_xyz(r_lin, g_lin, b_lin);
        let chroma = match config.diagram_kind {
            DiagramKind::CieXy => xyz_to_xy(x, y, z),
            DiagramKind::CieUpVp => {
                let (up, vp) = xyz_to_upvp(x, y, z);
                // u'v' visible locus ≈ u' ∈ [0, 0.62], v' ∈ [0, 0.60] — treat as x,y
                ChromaXy::new(up, vp)
            }
        };

        let is_oob = triangle.as_ref().map_or(false, |t| !t.contains(&chroma));
        let color = if is_oob { oob_color } else { in_color };

        let (px, py) = xy_to_pixel(chroma.x, chroma.y, config.width, config.height);
        if px >= 0 && py >= 0 && (px as usize) < out_w && (py as usize) < out_h {
            let idx = (py as usize * out_w + px as usize) * 4;
            // Additive blending for brightness accumulation
            rgba[idx] = rgba[idx].saturating_add(color[0] / 8);
            rgba[idx + 1] = rgba[idx + 1].saturating_add(color[1] / 8);
            rgba[idx + 2] = rgba[idx + 2].saturating_add(color[2] / 8);
            rgba[idx + 3] = 255;
        }
    }

    // Draw gamut triangle overlays
    if config.show_gamut_triangle {
        draw_gamut_triangle(
            &mut rgba,
            config.width,
            config.height,
            config.target_gamut,
            None,
        );
    }
    for &gamut in overlays {
        draw_gamut_triangle(&mut rgba, config.width, config.height, gamut, None);
    }

    Ok(rgba)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamut_triangle_rec709() {
        let t = GamutTriangle::rec709();
        assert!((t.red.x - 0.64).abs() < 1e-6);
        assert!((t.green.y - 0.60).abs() < 1e-6);
    }

    #[test]
    fn test_gamut_triangle_rec2020() {
        let t = GamutTriangle::rec2020();
        assert!(t.area() > GamutTriangle::rec709().area());
    }

    #[test]
    fn test_gamut_triangle_contains_white() {
        let t = GamutTriangle::rec709();
        assert!(t.contains(&t.white));
    }

    #[test]
    fn test_gamut_triangle_contains_interior() {
        let t = GamutTriangle::rec709();
        // Midpoint of triangle should be inside
        let mid = ChromaXy::new(
            (t.red.x + t.green.x + t.blue.x) / 3.0,
            (t.red.y + t.green.y + t.blue.y) / 3.0,
        );
        assert!(t.contains(&mid));
    }

    #[test]
    fn test_gamut_triangle_outside() {
        let t = GamutTriangle::rec709();
        let outside = ChromaXy::new(0.0, 0.0);
        assert!(!t.contains(&outside));
    }

    #[test]
    fn test_gamut_area_positive() {
        let t = GamutTriangle::rec709();
        assert!(t.area() > 0.0);
    }

    #[test]
    fn test_linear_rgb_to_xyz_black() {
        let (x, y, z) = linear_rgb_to_xyz(0.0, 0.0, 0.0);
        assert!(x.abs() < 1e-10);
        assert!(y.abs() < 1e-10);
        assert!(z.abs() < 1e-10);
    }

    #[test]
    fn test_xyz_to_xy_d65_white() {
        let (x, y, z) = linear_rgb_to_xyz(1.0, 1.0, 1.0);
        let chroma = xyz_to_xy(x, y, z);
        // Should be near D65 white point
        assert!((chroma.x - 0.3127).abs() < 0.01);
        assert!((chroma.y - 0.3290).abs() < 0.01);
    }

    #[test]
    fn test_xyz_to_xy_black_returns_d65() {
        let chroma = xyz_to_xy(0.0, 0.0, 0.0);
        assert!((chroma.x - 0.3127).abs() < 1e-6);
    }

    #[test]
    fn test_srgb_to_linear_zero() {
        assert!(srgb_to_linear(0.0).abs() < 1e-10);
    }

    #[test]
    fn test_srgb_to_linear_one() {
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_xyz_to_upvp_d65() {
        let (x, y, z) = linear_rgb_to_xyz(1.0, 1.0, 1.0);
        let (up, vp) = xyz_to_upvp(x, y, z);
        assert!((up - 0.1978).abs() < 0.01);
        assert!((vp - 0.4683).abs() < 0.01);
    }

    #[test]
    fn test_analyze_gamut_all_black() {
        // 2x2 black frame — all pixels should be in-gamut (mapped to D65)
        let frame = vec![0u8; 2 * 2 * 3];
        let result = analyze_gamut(&frame, 2, 2, TargetGamut::Rec709);
        assert_eq!(result.total_pixels, 4);
        // D65 is inside Rec709
        assert_eq!(result.out_of_gamut_pixels, 0);
    }

    #[test]
    fn test_analyze_gamut_pure_red() {
        // 1x1 pure red pixel
        let frame = vec![255, 0, 0];
        let result = analyze_gamut(&frame, 1, 1, TargetGamut::Rec709);
        assert_eq!(result.total_pixels, 1);
        // Pure Rec.709 red should be on the gamut boundary
    }

    #[test]
    fn test_gamut_scope_config_default() {
        let cfg = GamutScopeConfig::default();
        assert_eq!(cfg.width, 512);
        assert_eq!(cfg.target_gamut, TargetGamut::Rec709);
        assert!(cfg.show_gamut_triangle);
    }

    #[test]
    fn test_for_gamut_srgb_equals_rec709() {
        let srgb = GamutTriangle::for_gamut(TargetGamut::Srgb);
        let r709 = GamutTriangle::for_gamut(TargetGamut::Rec709);
        assert!((srgb.red.x - r709.red.x).abs() < 1e-10);
    }
}
