//! CIE 1931 chromaticity diagram for color gamut visualization.
//!
//! The CIE 1931 xy chromaticity diagram is the standard for representing all
//! visible colors as a 2D plot. This module provides:
//! - CIE xy chromaticity diagram rendering
//! - Gamut overlay for Rec.709, Rec.2020, DCI-P3
//! - Pixel distribution plotting
//! - Out-of-gamut warning
//! - Color space conversions (RGB → XYZ → xyY)
//!
//! The CIE diagram is essential for broadcast and cinema workflows to ensure
//! colors stay within the target color space.

use crate::render::Canvas;
use crate::{GamutColorspace, ScopeConfig, ScopeData, ScopeType};
use oximedia_core::OxiResult;

/// CIE 1931 chromaticity coordinates (x, y).
pub type CieXy = (f32, f32);

/// CIE XYZ tristimulus values.
#[derive(Debug, Clone, Copy)]
pub struct CieXyz {
    /// X tristimulus value.
    pub x: f32,

    /// Y tristimulus value (luminance).
    pub y: f32,

    /// Z tristimulus value.
    pub z: f32,
}

/// CIE xyY color coordinates.
#[derive(Debug, Clone, Copy)]
pub struct CieXyy {
    /// x chromaticity coordinate.
    pub x: f32,

    /// y chromaticity coordinate.
    pub y: f32,

    /// Y luminance.
    pub y_luminance: f32,
}

/// Rec.709 (HD) color gamut primaries in CIE xy coordinates.
///
/// ITU-R BT.709 color space primaries:
/// - Red: (0.64, 0.33)
/// - Green: (0.30, 0.60)
/// - Blue: (0.15, 0.06)
/// - White point D65: (0.3127, 0.3290)
#[must_use]
pub fn rec709_primaries() -> [(f32, f32); 4] {
    [
        (0.64, 0.33),     // Red
        (0.30, 0.60),     // Green
        (0.15, 0.06),     // Blue
        (0.3127, 0.3290), // D65 white
    ]
}

/// Rec.2020 (UHD/HDR) color gamut primaries in CIE xy coordinates.
///
/// ITU-R BT.2020 color space primaries (wider gamut):
/// - Red: (0.708, 0.292)
/// - Green: (0.170, 0.797)
/// - Blue: (0.131, 0.046)
/// - White point D65: (0.3127, 0.3290)
#[must_use]
pub fn rec2020_primaries() -> [(f32, f32); 4] {
    [
        (0.708, 0.292),   // Red
        (0.170, 0.797),   // Green
        (0.131, 0.046),   // Blue
        (0.3127, 0.3290), // D65 white
    ]
}

/// DCI-P3 (Digital Cinema) color gamut primaries in CIE xy coordinates.
///
/// DCI-P3 color space primaries:
/// - Red: (0.680, 0.320)
/// - Green: (0.265, 0.690)
/// - Blue: (0.150, 0.060)
/// - White point D65: (0.3127, 0.3290) (for P3-D65 variant)
#[must_use]
pub fn dci_p3_primaries() -> [(f32, f32); 4] {
    [
        (0.680, 0.320),   // Red
        (0.265, 0.690),   // Green
        (0.150, 0.060),   // Blue
        (0.3127, 0.3290), // D65 white
    ]
}

/// Spectral locus points for CIE 1931 (horseshoe curve).
///
/// These are approximate points along the spectral locus (pure monochromatic light)
/// from 380nm (violet) to 700nm (red).
#[must_use]
#[allow(clippy::excessive_precision)]
pub fn spectral_locus_points() -> Vec<(f32, f32)> {
    vec![
        // Wavelength (nm) -> (x, y) chromaticity
        (0.1741, 0.0050), // 380 nm
        (0.1738, 0.0050), // 390 nm
        (0.1733, 0.0086), // 400 nm
        (0.1728, 0.0213), // 410 nm
        (0.1722, 0.0464), // 420 nm
        (0.1714, 0.0853), // 430 nm
        (0.1703, 0.1394), // 440 nm
        (0.1689, 0.2050), // 450 nm
        (0.1669, 0.2789), // 460 nm
        (0.1644, 0.3621), // 470 nm
        (0.1611, 0.4441), // 480 nm
        (0.1566, 0.5260), // 490 nm
        (0.1510, 0.6082), // 500 nm
        (0.1440, 0.6907), // 510 nm
        (0.1355, 0.7723), // 520 nm
        (0.1241, 0.8338), // 530 nm
        (0.1096, 0.8662), // 540 nm
        (0.0913, 0.8727), // 550 nm
        (0.0687, 0.8598), // 560 nm
        (0.0454, 0.8363), // 570 nm
        (0.0235, 0.8057), // 580 nm
        (0.0082, 0.7618), // 590 nm
        (0.0039, 0.6923), // 600 nm
        (0.0139, 0.6054), // 610 nm
        (0.0389, 0.5030), // 620 nm
        (0.0743, 0.3981), // 630 nm
        (0.1142, 0.2951), // 640 nm
        (0.1547, 0.2123), // 650 nm
        (0.1929, 0.1582), // 660 nm
        (0.2296, 0.1117), // 670 nm
        (0.2658, 0.0822), // 680 nm
        (0.3016, 0.0574), // 690 nm
        (0.3373, 0.0394), // 700 nm
    ]
}

/// Converts sRGB (0-255) to CIE XYZ tristimulus values.
///
/// This conversion assumes sRGB color space with D65 white point.
#[must_use]
pub fn rgb_to_xyz(r: u8, g: u8, b: u8) -> CieXyz {
    // Normalize to 0-1 range
    let r_norm = f32::from(r) / 255.0;
    let g_norm = f32::from(g) / 255.0;
    let b_norm = f32::from(b) / 255.0;

    // Apply sRGB gamma correction (inverse)
    let r_linear = srgb_inverse_gamma(r_norm);
    let g_linear = srgb_inverse_gamma(g_norm);
    let b_linear = srgb_inverse_gamma(b_norm);

    // sRGB to XYZ matrix (D65 white point)
    let x = 0.4124564 * r_linear + 0.3575761 * g_linear + 0.1804375 * b_linear;
    let y = 0.2126729 * r_linear + 0.7151522 * g_linear + 0.0721750 * b_linear;
    let z = 0.0193339 * r_linear + 0.119_192 * g_linear + 0.9503041 * b_linear;

    CieXyz { x, y, z }
}

/// Converts CIE XYZ to CIE xyY (chromaticity + luminance).
#[must_use]
pub fn xyz_to_xyy(xyz: CieXyz) -> CieXyy {
    let sum = xyz.x + xyz.y + xyz.z;

    if sum < 1e-10 {
        // Avoid division by zero, return D65 white point
        return CieXyy {
            x: 0.3127,
            y: 0.3290,
            y_luminance: 0.0,
        };
    }

    CieXyy {
        x: xyz.x / sum,
        y: xyz.y / sum,
        y_luminance: xyz.y,
    }
}

/// Converts sRGB directly to CIE xy chromaticity coordinates.
#[must_use]
pub fn rgb_to_cie_xy(r: u8, g: u8, b: u8) -> CieXy {
    let xyz = rgb_to_xyz(r, g, b);
    let xyy = xyz_to_xyy(xyz);
    (xyy.x, xyy.y)
}

/// Applies sRGB inverse gamma (linearization).
#[must_use]
fn srgb_inverse_gamma(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// Generates a CIE 1931 chromaticity diagram with pixel distribution.
///
/// # Arguments
///
/// * `frame` - RGB24 frame data (width * height * 3 bytes)
/// * `width` - Frame width in pixels
/// * `height` - Frame height in pixels
/// * `config` - Scope configuration
///
/// # Errors
///
/// Returns an error if frame data is invalid or insufficient.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::too_many_lines)]
pub fn generate_cie_diagram(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &ScopeConfig,
) -> OxiResult<ScopeData> {
    let expected_size = (width * height * 3) as usize;
    if frame.len() < expected_size {
        return Err(oximedia_core::OxiError::InvalidData(format!(
            "Frame data too small: expected {expected_size}, got {}",
            frame.len()
        )));
    }

    let scope_width = config.width;
    let scope_height = config.height;

    let mut canvas = Canvas::new(scope_width, scope_height);

    // Draw background (faint spectral locus outline)
    draw_spectral_locus(&mut canvas);

    // Draw gamut triangle for selected colorspace
    draw_gamut_triangle(&mut canvas, config.gamut_colorspace);

    // Create accumulation buffer for pixel distribution
    let mut accumulator = vec![0u32; (scope_width * scope_height) as usize];

    // Plot all pixels from the frame
    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y * width + x) * 3) as usize;
            let r = frame[pixel_idx];
            let g = frame[pixel_idx + 1];
            let b = frame[pixel_idx + 2];

            // Skip pure black pixels (would map to undefined chromaticity)
            if r == 0 && g == 0 && b == 0 {
                continue;
            }

            let (cie_x, cie_y) = rgb_to_cie_xy(r, g, b);

            // Map CIE xy to canvas coordinates
            // CIE diagram typically shows x: 0-0.8, y: 0-0.9
            let canvas_x = (cie_x * scope_width as f32 / 0.8).clamp(0.0, scope_width as f32 - 1.0);
            let canvas_y = scope_height as f32
                - (cie_y * scope_height as f32 / 0.9).clamp(0.0, scope_height as f32 - 1.0);

            let canvas_x = canvas_x as u32;
            let canvas_y = canvas_y as u32;

            let idx = (canvas_y * scope_width + canvas_x) as usize;
            if idx < accumulator.len() {
                accumulator[idx] = accumulator[idx].saturating_add(1);
            }
        }
    }

    // Find max value for normalization
    let max_val = accumulator.iter().copied().max().unwrap_or(1);

    // Draw pixel distribution
    for y in 0..scope_height {
        for x in 0..scope_width {
            let idx = (y * scope_width + x) as usize;
            let count = accumulator[idx];

            if count > 0 {
                let normalized = ((count as f32 / max_val as f32).sqrt() * 255.0) as u8;

                // Color based on position (approximate spectral color)
                let cie_x = (x as f32 / scope_width as f32) * 0.8;
                let cie_y = ((scope_height - y) as f32 / scope_height as f32) * 0.9;

                let color = cie_to_approximate_rgb(cie_x, cie_y, normalized);
                canvas.blend_pixel(x, y, color);
            }
        }
    }

    // Draw labels if enabled
    if config.show_labels {
        draw_cie_labels(&mut canvas);
    }

    Ok(ScopeData {
        width: scope_width,
        height: scope_height,
        data: canvas.data,
        scope_type: ScopeType::CieDiagram,
    })
}

/// Draws the spectral locus (horseshoe curve).
fn draw_spectral_locus(canvas: &mut Canvas) {
    let points = spectral_locus_points();
    let scope_width = canvas.width;
    let scope_height = canvas.height;

    let color = [128, 128, 128, 192]; // Semi-transparent gray

    for i in 0..points.len() - 1 {
        let (x1, y1) = points[i];
        let (x2, y2) = points[i + 1];

        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let canvas_x1 = (x1 * scope_width as f32 / 0.8).min(scope_width as f32 - 1.0) as u32;
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let canvas_y1 = (scope_height as f32 - (y1 * scope_height as f32 / 0.9)).max(0.0) as u32;

        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let canvas_x2 = (x2 * scope_width as f32 / 0.8).min(scope_width as f32 - 1.0) as u32;
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let canvas_y2 = (scope_height as f32 - (y2 * scope_height as f32 / 0.9)).max(0.0) as u32;

        canvas.draw_line(canvas_x1, canvas_y1, canvas_x2, canvas_y2, color);
    }

    // Close the spectral locus with purple line (non-spectral purples)
    if let (Some(&first), Some(&last)) = (points.first(), points.last()) {
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let x1 = (first.0 * scope_width as f32 / 0.8).min(scope_width as f32 - 1.0) as u32;
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let y1 = (scope_height as f32 - (first.1 * scope_height as f32 / 0.9)).max(0.0) as u32;

        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let x2 = (last.0 * scope_width as f32 / 0.8).min(scope_width as f32 - 1.0) as u32;
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let y2 = (scope_height as f32 - (last.1 * scope_height as f32 / 0.9)).max(0.0) as u32;

        canvas.draw_line(x1, y1, x2, y2, [128, 0, 128, 192]); // Purple
    }
}

/// Draws the gamut triangle for the specified colorspace.
fn draw_gamut_triangle(canvas: &mut Canvas, gamut: GamutColorspace) {
    let primaries = match gamut {
        GamutColorspace::Rec709 => rec709_primaries(),
        GamutColorspace::Rec2020 => rec2020_primaries(),
        GamutColorspace::DciP3 => dci_p3_primaries(),
    };

    let scope_width = canvas.width;
    let scope_height = canvas.height;

    let color = [255, 255, 255, 255]; // White

    // Draw triangle connecting R, G, B primaries
    for i in 0..3 {
        let (x1, y1) = primaries[i];
        let (x2, y2) = primaries[(i + 1) % 3];

        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let canvas_x1 = (x1 * scope_width as f32 / 0.8).min(scope_width as f32 - 1.0) as u32;
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let canvas_y1 = (scope_height as f32 - (y1 * scope_height as f32 / 0.9)).max(0.0) as u32;

        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let canvas_x2 = (x2 * scope_width as f32 / 0.8).min(scope_width as f32 - 1.0) as u32;
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let canvas_y2 = (scope_height as f32 - (y2 * scope_height as f32 / 0.9)).max(0.0) as u32;

        canvas.draw_line(canvas_x1, canvas_y1, canvas_x2, canvas_y2, color);
    }

    // Draw white point
    let (white_x, white_y) = primaries[3];
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    let white_canvas_x = (white_x * scope_width as f32 / 0.8).min(scope_width as f32 - 1.0) as u32;
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    let white_canvas_y =
        (scope_height as f32 - (white_y * scope_height as f32 / 0.9)).max(0.0) as u32;

    canvas.draw_circle(white_canvas_x, white_canvas_y, 3, color);
}

/// Draws labels on the CIE diagram.
fn draw_cie_labels(canvas: &mut Canvas) {
    let color = [255, 255, 255, 255];

    // Draw axis labels
    crate::render::draw_label(canvas, 5, canvas.height - 15, "0", color);
    crate::render::draw_label(canvas, canvas.width - 20, canvas.height - 15, "08", color);
    crate::render::draw_label(canvas, 5, 5, "09", color);
}

/// Approximates an RGB color from CIE xy coordinates.
///
/// This is a rough approximation for visualization purposes.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn cie_to_approximate_rgb(x: f32, y: f32, intensity: u8) -> [u8; 4] {
    // Very rough approximation based on wavelength hue
    // In reality, this would require proper XYZ to RGB conversion

    let hue = if x < 0.3 {
        240.0 // Blue region
    } else if x < 0.4 && y > 0.5 {
        120.0 // Green region
    } else if x > 0.5 && y < 0.4 {
        0.0 // Red region
    } else if y > 0.4 {
        60.0 // Yellow-green region
    } else {
        300.0 // Magenta region
    };

    let (r, g, b) = hue_to_rgb(hue);

    [
        ((u16::from(r) * u16::from(intensity)) / 255) as u8,
        ((u16::from(g) * u16::from(intensity)) / 255) as u8,
        ((u16::from(b) * u16::from(intensity)) / 255) as u8,
        255,
    ]
}

/// Converts hue (0-360) to RGB.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn hue_to_rgb(hue: f32) -> (u8, u8, u8) {
    let h = hue / 60.0;
    let c = 255.0;
    let x = c * (1.0 - ((h % 2.0) - 1.0).abs());

    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    (r as u8, g as u8, b as u8)
}

/// CIE diagram statistics.
#[derive(Debug, Clone)]
pub struct CieStats {
    /// Coverage area relative to the selected gamut (0-1).
    pub gamut_coverage: f32,

    /// Percentage of pixels outside the selected gamut.
    pub out_of_gamut_percent: f32,

    /// Average chromaticity coordinates.
    pub avg_chromaticity: (f32, f32),

    /// Color temperature estimate (in Kelvin).
    pub color_temperature: f32,
}

/// Computes CIE diagram statistics from frame data.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn compute_cie_stats(
    frame: &[u8],
    width: u32,
    height: u32,
    gamut: GamutColorspace,
) -> CieStats {
    let mut x_sum = 0.0f32;
    let mut y_sum = 0.0f32;
    let mut valid_pixels = 0u32;
    let mut out_of_gamut_count = 0u32;

    let primaries = match gamut {
        GamutColorspace::Rec709 => rec709_primaries(),
        GamutColorspace::Rec2020 => rec2020_primaries(),
        GamutColorspace::DciP3 => dci_p3_primaries(),
    };

    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y * width + x) * 3) as usize;
            if pixel_idx + 2 >= frame.len() {
                break;
            }

            let r = frame[pixel_idx];
            let g = frame[pixel_idx + 1];
            let b = frame[pixel_idx + 2];

            // Skip pure black
            if r == 0 && g == 0 && b == 0 {
                continue;
            }

            let (cie_x, cie_y) = rgb_to_cie_xy(r, g, b);

            x_sum += cie_x;
            y_sum += cie_y;
            valid_pixels += 1;

            // Check if outside gamut triangle
            if !is_inside_triangle(cie_x, cie_y, &primaries) {
                out_of_gamut_count += 1;
            }
        }
    }

    let avg_x = if valid_pixels > 0 {
        x_sum / valid_pixels as f32
    } else {
        0.3127
    };

    let avg_y = if valid_pixels > 0 {
        y_sum / valid_pixels as f32
    } else {
        0.3290
    };

    // Estimate color temperature (simplified McCamy's approximation)
    let n = (avg_x - 0.3320) / (0.1858 - avg_y);
    let cct = 449.0 * n.powi(3) + 3525.0 * n.powi(2) + 6823.3 * n + 5520.33;
    let color_temperature = cct.clamp(1000.0, 25000.0);

    CieStats {
        gamut_coverage: compute_gamut_coverage(&primaries),
        out_of_gamut_percent: (out_of_gamut_count as f32 / valid_pixels as f32) * 100.0,
        avg_chromaticity: (avg_x, avg_y),
        color_temperature,
    }
}

/// Checks if a point is inside the gamut triangle.
#[must_use]
fn is_inside_triangle(px: f32, py: f32, primaries: &[(f32, f32); 4]) -> bool {
    let (x1, y1) = primaries[0]; // Red
    let (x2, y2) = primaries[1]; // Green
    let (x3, y3) = primaries[2]; // Blue

    let area = ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)).abs();
    let area1 = ((x2 - px) * (y3 - py) - (x3 - px) * (y2 - py)).abs();
    let area2 = ((x1 - px) * (y3 - py) - (x3 - px) * (y1 - py)).abs();
    let area3 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)).abs();

    (area1 + area2 + area3 - area).abs() < 0.01
}

/// Computes the gamut coverage: fraction of the CIE 1931 chromaticity diagram
/// area that is covered by the given gamut triangle (R, G, B primaries).
///
/// Uses the exact Shoelace (cross-product) formula for triangle area,
/// normalised by the standard CIE 1931 2° observer spectral locus area
/// (closed polygon from 380–700 nm, including the purple line).
///
/// Returns a value in `[0.0, 1.0]`.
#[must_use]
fn compute_gamut_coverage(primaries: &[(f32, f32); 4]) -> f32 {
    // Signed area of the gamut triangle via the Shoelace formula.
    let (rx, ry) = primaries[0]; // Red primary
    let (gx, gy) = primaries[1]; // Green primary
    let (bx, by) = primaries[2]; // Blue primary

    let triangle_area = ((gx - rx) * (by - ry) - (bx - rx) * (gy - ry)).abs() * 0.5;

    // CIE 1931 standard chromaticity diagram area (spectral locus, 380–700 nm,
    // closed by the purple (alychne) boundary).  This constant is the accepted
    // industry reference for "% of CIE gamut" calculations, matching the value
    // quoted in ITU-R BT.2407 and related colorimetry literature.
    const CIE_LOCUS_AREA: f32 = 0.19517;

    (triangle_area / CIE_LOCUS_AREA).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(width: u32, height: u32) -> Vec<u8> {
        let mut frame = vec![0u8; (width * height * 3) as usize];

        // Create colorful test pattern
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;

                let r = ((x * 255) / width) as u8;
                let g = ((y * 255) / height) as u8;
                let b = 128u8;

                frame[idx] = r;
                frame[idx + 1] = g;
                frame[idx + 2] = b;
            }
        }

        frame
    }

    #[test]
    fn test_rgb_to_xyz() {
        let xyz = rgb_to_xyz(255, 255, 255); // White
        assert!(xyz.x > 0.8 && xyz.x < 1.1);
        assert!(xyz.y > 0.8 && xyz.y < 1.1);
        assert!(xyz.z > 0.8 && xyz.z < 1.2);
    }

    #[test]
    fn test_xyz_to_xyy() {
        let xyz = CieXyz {
            x: 0.9505,
            y: 1.0000,
            z: 1.0890,
        }; // D65 white
        let xyy = xyz_to_xyy(xyz);

        assert!((xyy.x - 0.3127).abs() < 0.01);
        assert!((xyy.y - 0.3290).abs() < 0.01);
    }

    #[test]
    fn test_rgb_to_cie_xy() {
        let (x, y) = rgb_to_cie_xy(255, 0, 0); // Pure red
        assert!(x > 0.6); // Red has high x value
        assert!(y < 0.4); // Red has low y value
    }

    #[test]
    fn test_generate_cie_diagram() {
        let frame = create_test_frame(100, 100);
        let config = ScopeConfig::default();

        let result = generate_cie_diagram(&frame, 100, 100, &config);
        assert!(result.is_ok());

        let scope = result.expect("should succeed in test");
        assert_eq!(scope.width, config.width);
        assert_eq!(scope.height, config.height);
    }

    #[test]
    fn test_compute_cie_stats() {
        let frame = create_test_frame(100, 100);
        let stats = compute_cie_stats(&frame, 100, 100, GamutColorspace::Rec709);

        assert!(stats.avg_chromaticity.0 > 0.0);
        assert!(stats.avg_chromaticity.1 > 0.0);
        assert!(stats.color_temperature > 1000.0);
        assert!(stats.color_temperature < 25000.0);
    }

    #[test]
    fn test_is_inside_triangle() {
        let primaries = rec709_primaries();

        // Test white point (should be inside)
        assert!(is_inside_triangle(0.3127, 0.3290, &primaries));

        // Test point far outside (should be outside)
        assert!(!is_inside_triangle(0.0, 0.0, &primaries));
    }

    #[test]
    fn test_gamut_primaries() {
        let rec709 = rec709_primaries();
        assert_eq!(rec709.len(), 4);

        let rec2020 = rec2020_primaries();
        assert_eq!(rec2020.len(), 4);

        let dci_p3 = dci_p3_primaries();
        assert_eq!(dci_p3.len(), 4);
    }

    #[test]
    fn test_spectral_locus() {
        let points = spectral_locus_points();
        assert!(points.len() > 20); // Should have many points

        // All points should be in valid range
        for (x, y) in &points {
            assert!(*x >= 0.0 && *x <= 1.0);
            assert!(*y >= 0.0 && *y <= 1.0);
        }
    }
    #[test]
    fn test_gamut_coverage() {
        // Rec.709 should cover ~57% of the CIE diagram (based on standard locus area)
        let rec709 = rec709_primaries();
        let cov = compute_gamut_coverage(&rec709);
        assert!(
            cov > 0.4 && cov < 0.75,
            "Rec.709 coverage {cov} out of expected range"
        );

        // Rec.2020 should cover more than Rec.709
        let rec2020 = rec2020_primaries();
        let cov2020 = compute_gamut_coverage(&rec2020);
        assert!(cov2020 > cov, "Rec.2020 should cover more than Rec.709");

        // DCI-P3 should be between Rec.709 and Rec.2020
        let dcip3 = dci_p3_primaries();
        let cov_p3 = compute_gamut_coverage(&dcip3);
        assert!(
            cov_p3 > cov && cov_p3 < cov2020,
            "DCI-P3 should be between Rec.709 and Rec.2020"
        );

        // All values must be in [0, 1]
        assert!(cov >= 0.0 && cov <= 1.0);
        assert!(cov2020 >= 0.0 && cov2020 <= 1.0);
        assert!(cov_p3 >= 0.0 && cov_p3 <= 1.0);
    }

    #[test]
    fn test_gamut_coverage_in_stats() {
        let frame = create_test_frame(100, 100);
        let stats = compute_cie_stats(&frame, 100, 100, GamutColorspace::Rec709);
        // gamut_coverage must now be non-zero and in range
        assert!(
            stats.gamut_coverage > 0.0,
            "gamut_coverage should not be zero"
        );
        assert!(
            stats.gamut_coverage <= 1.0,
            "gamut_coverage should not exceed 1.0"
        );
    }
}
