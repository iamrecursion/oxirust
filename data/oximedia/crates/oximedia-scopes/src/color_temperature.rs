#![allow(dead_code)]
//! Color temperature analysis scope for video frames.
//!
//! Provides tools to analyze and visualize the correlated color temperature (CCT)
//! of a video image. Useful for white balance verification, matching shots, and
//! ensuring consistent color temperature across a production. Outputs CCT in
//! Kelvin along with tint (green/magenta) deviation.

/// Minimum color temperature in Kelvin.
const MIN_CCT: f64 = 1667.0;
/// Maximum color temperature in Kelvin.
const MAX_CCT: f64 = 25000.0;

/// A CIE xy chromaticity coordinate.
#[derive(Debug, Clone, Copy)]
pub struct CieXy {
    /// x chromaticity coordinate.
    pub x: f64,
    /// y chromaticity coordinate.
    pub y: f64,
}

/// Result of a color temperature analysis.
#[derive(Debug, Clone)]
pub struct ColorTemperatureResult {
    /// Estimated correlated color temperature in Kelvin.
    pub cct_kelvin: f64,
    /// Tint deviation from the Planckian locus (positive = green, negative = magenta).
    pub tint: f64,
    /// Average CIE xy chromaticity of the analyzed region.
    pub avg_chromaticity: CieXy,
    /// Standard deviation of CCT across the frame.
    pub cct_std_dev: f64,
    /// Minimum CCT found in the frame.
    pub cct_min: f64,
    /// Maximum CCT found in the frame.
    pub cct_max: f64,
    /// Per-region CCT values (if spatial analysis was performed).
    pub region_ccts: Vec<f64>,
}

/// Configuration for color temperature analysis.
#[derive(Debug, Clone)]
pub struct ColorTemperatureConfig {
    /// Number of horizontal regions for spatial analysis.
    pub grid_cols: u32,
    /// Number of vertical regions for spatial analysis.
    pub grid_rows: u32,
    /// Minimum pixel brightness to include (0..255).
    pub min_brightness: u8,
    /// Maximum pixel brightness to include (0..255).
    pub max_brightness: u8,
    /// Whether to weight by pixel brightness.
    pub brightness_weighted: bool,
}

impl Default for ColorTemperatureConfig {
    fn default() -> Self {
        Self {
            grid_cols: 4,
            grid_rows: 4,
            min_brightness: 30,
            max_brightness: 230,
            brightness_weighted: true,
        }
    }
}

/// Converts linear sRGB (0..1) to CIE XYZ.
#[allow(clippy::cast_precision_loss)]
fn srgb_to_xyz(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    // sRGB to XYZ (D65) matrix
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
    (x, y, z)
}

/// Converts CIE XYZ to CIE xy chromaticity.
fn xyz_to_xy(x: f64, y: f64, z: f64) -> CieXy {
    let sum = x + y + z;
    if sum < 1e-10 {
        return CieXy {
            x: 0.3127,
            y: 0.3290,
        }; // D65 white point
    }
    CieXy {
        x: x / sum,
        y: y / sum,
    }
}

/// Estimates CCT from CIE xy coordinates using McCamy's approximation.
///
/// McCamy (1992): CCT = 449 * n^3 + 3525 * n^2 + 6823.3 * n + 5520.33
/// where n = (x - 0.3320) / (0.1858 - y)
fn estimate_cct_mccamy(xy: CieXy) -> f64 {
    let denom = 0.1858 - xy.y;
    if denom.abs() < 1e-10 {
        return 6500.0; // Default to D65
    }
    let n = (xy.x - 0.3320) / denom;
    let cct = 449.0 * n * n * n + 3525.0 * n * n + 6823.3 * n + 5520.33;
    cct.clamp(MIN_CCT, MAX_CCT)
}

/// Computes the Planckian locus xy for a given CCT (Kelvin).
///
/// Uses the CIE daylight approximation for 4000K-25000K.
fn planckian_xy(cct: f64) -> CieXy {
    let t = cct;
    let t2 = t * t;
    let t3 = t2 * t;

    let x = if t <= 7000.0 {
        -4.6070e9 / t3 + 2.9678e6 / t2 + 0.09911e3 / t + 0.244063
    } else {
        -2.0064e9 / t3 + 1.9018e6 / t2 + 0.24748e3 / t + 0.237040
    };

    let y = -3.0 * x * x + 2.87 * x - 0.275;
    CieXy { x, y }
}

/// Estimates the tint (deviation from Planckian locus).
fn compute_tint(xy: CieXy, cct: f64) -> f64 {
    let planckian = planckian_xy(cct);
    // Tint is perpendicular distance from the Planckian locus
    // Positive = above (green), Negative = below (magenta)
    let dy = xy.y - planckian.y;
    // Scale to a useful range (roughly -150 to +150)
    dy * 1000.0
}

/// Linearizes an sRGB gamma-encoded value (0..255) to linear (0..1).
#[allow(clippy::cast_precision_loss)]
fn srgb_gamma_to_linear(v: u8) -> f64 {
    let s = v as f64 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// Analyzes the color temperature of an RGB24 video frame.
///
/// # Arguments
/// * `frame` - RGB24 frame data (3 bytes per pixel, row-major)
/// * `width` - Frame width in pixels
/// * `height` - Frame height in pixels
/// * `config` - Analysis configuration
///
/// # Returns
/// Color temperature analysis result.
#[allow(clippy::cast_precision_loss)]
pub fn analyze_color_temperature(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &ColorTemperatureConfig,
) -> ColorTemperatureResult {
    let expected_len = (width * height * 3) as usize;
    if frame.len() < expected_len || width == 0 || height == 0 {
        return ColorTemperatureResult {
            cct_kelvin: 6500.0,
            tint: 0.0,
            avg_chromaticity: CieXy {
                x: 0.3127,
                y: 0.3290,
            },
            cct_std_dev: 0.0,
            cct_min: 6500.0,
            cct_max: 6500.0,
            region_ccts: Vec::new(),
        };
    }

    let cols = config.grid_cols.max(1) as usize;
    let rows = config.grid_rows.max(1) as usize;
    let region_w = width as usize / cols;
    let region_h = height as usize / rows;

    let mut region_ccts = Vec::with_capacity(cols * rows);
    let mut total_x = 0.0_f64;
    let mut total_y = 0.0_f64;
    let mut total_z = 0.0_f64;
    let mut total_weight = 0.0_f64;

    for ry in 0..rows {
        for rx in 0..cols {
            let mut sum_x = 0.0_f64;
            let mut sum_y = 0.0_f64;
            let mut sum_z = 0.0_f64;
            let mut count = 0u64;

            let y_start = ry * region_h;
            let y_end = ((ry + 1) * region_h).min(height as usize);
            let x_start = rx * region_w;
            let x_end = ((rx + 1) * region_w).min(width as usize);

            for py in y_start..y_end {
                for px in x_start..x_end {
                    let idx = (py * width as usize + px) * 3;
                    if idx + 2 >= frame.len() {
                        continue;
                    }
                    let r = frame[idx];
                    let g = frame[idx + 1];
                    let b = frame[idx + 2];

                    // Brightness check
                    let brightness = ((r as u16 + g as u16 + b as u16) / 3) as u8;
                    if brightness < config.min_brightness || brightness > config.max_brightness {
                        continue;
                    }

                    let rl = srgb_gamma_to_linear(r);
                    let gl = srgb_gamma_to_linear(g);
                    let bl = srgb_gamma_to_linear(b);

                    let (cx, cy, cz) = srgb_to_xyz(rl, gl, bl);

                    let w = if config.brightness_weighted {
                        cy.max(0.001)
                    } else {
                        1.0
                    };

                    sum_x += cx * w;
                    sum_y += cy * w;
                    sum_z += cz * w;
                    count += 1;
                }
            }

            if count > 0 {
                let avg_xy = xyz_to_xy(sum_x, sum_y, sum_z);
                let cct = estimate_cct_mccamy(avg_xy);
                region_ccts.push(cct);

                total_x += sum_x;
                total_y += sum_y;
                total_z += sum_z;
                total_weight += count as f64;
                let _ = total_weight;
            }
        }
    }

    let overall_xy = if total_x + total_y > 1e-10 {
        xyz_to_xy(total_x, total_y, total_z)
    } else {
        CieXy {
            x: 0.3127,
            y: 0.3290,
        }
    };

    let overall_cct = estimate_cct_mccamy(overall_xy);
    let tint = compute_tint(overall_xy, overall_cct);

    let (cct_min, cct_max, cct_std) = if region_ccts.is_empty() {
        (overall_cct, overall_cct, 0.0)
    } else {
        let min = region_ccts.iter().copied().fold(f64::INFINITY, f64::min);
        let max = region_ccts
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mean = region_ccts.iter().sum::<f64>() / region_ccts.len() as f64;
        let variance =
            region_ccts.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / region_ccts.len() as f64;
        (min, max, variance.sqrt())
    };

    ColorTemperatureResult {
        cct_kelvin: overall_cct,
        tint,
        avg_chromaticity: overall_xy,
        cct_std_dev: cct_std,
        cct_min,
        cct_max,
        region_ccts,
    }
}

/// Describes the color temperature in human-readable terms.
#[must_use]
pub fn describe_cct(kelvin: f64) -> &'static str {
    if kelvin < 2700.0 {
        "Warm (Candlelight)"
    } else if kelvin < 3500.0 {
        "Warm (Tungsten/Halogen)"
    } else if kelvin < 4500.0 {
        "Neutral (Fluorescent)"
    } else if kelvin < 5500.0 {
        "Daylight (Direct Sun)"
    } else if kelvin < 6500.0 {
        "Daylight (Overcast)"
    } else if kelvin < 8000.0 {
        "Cool (Shade/Cloudy)"
    } else {
        "Very Cool (Blue Sky)"
    }
}

/// Converts a CCT value to an approximate sRGB color for visualization.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn cct_to_rgb(kelvin: f64) -> (u8, u8, u8) {
    let temp = kelvin / 100.0;
    let r;
    let g;
    let b;

    if temp <= 66.0 {
        r = 255.0;
        g = (99.4708025861 * temp.ln() - 161.1195681661).clamp(0.0, 255.0);
    } else {
        r = (329.698727446 * (temp - 60.0).powf(-0.1332047592)).clamp(0.0, 255.0);
        g = (288.1221695283 * (temp - 60.0).powf(-0.0755148492)).clamp(0.0, 255.0);
    }

    if temp >= 66.0 {
        b = 255.0;
    } else if temp <= 19.0 {
        b = 0.0;
    } else {
        b = (138.5177312231 * (temp - 10.0).ln() - 305.0447927307).clamp(0.0, 255.0);
    }

    (r as u8, g as u8, b as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srgb_to_xyz_black() {
        let (x, y, z) = srgb_to_xyz(0.0, 0.0, 0.0);
        assert!(x.abs() < 1e-10);
        assert!(y.abs() < 1e-10);
        assert!(z.abs() < 1e-10);
    }

    #[test]
    fn test_srgb_to_xyz_white() {
        let (x, y, z) = srgb_to_xyz(1.0, 1.0, 1.0);
        assert!((x - 0.9505).abs() < 0.01);
        assert!((y - 1.0).abs() < 0.01);
        assert!((z - 1.089).abs() < 0.02);
    }

    #[test]
    fn test_xyz_to_xy_d65() {
        // D65 white point
        let xy = xyz_to_xy(0.9505, 1.0, 1.089);
        assert!((xy.x - 0.3127).abs() < 0.001);
        assert!((xy.y - 0.3290).abs() < 0.001);
    }

    #[test]
    fn test_xyz_to_xy_zero() {
        let xy = xyz_to_xy(0.0, 0.0, 0.0);
        // Should return D65 default
        assert!((xy.x - 0.3127).abs() < 0.001);
    }

    #[test]
    fn test_mccamy_d65() {
        // D65 chromaticity should give ~6500K
        let xy = CieXy {
            x: 0.3127,
            y: 0.3290,
        };
        let cct = estimate_cct_mccamy(xy);
        assert!((cct - 6500.0).abs() < 500.0);
    }

    #[test]
    fn test_mccamy_tungsten() {
        // Approximate tungsten (~3200K) chromaticity
        let xy = CieXy {
            x: 0.4476,
            y: 0.4074,
        };
        let cct = estimate_cct_mccamy(xy);
        assert!(cct > 2500.0 && cct < 4000.0);
    }

    #[test]
    fn test_srgb_gamma_to_linear() {
        assert!(srgb_gamma_to_linear(0).abs() < 1e-10);
        assert!((srgb_gamma_to_linear(255) - 1.0).abs() < 0.01);
        // Mid gray
        let mid = srgb_gamma_to_linear(128);
        assert!(mid > 0.1 && mid < 0.5);
    }

    #[test]
    fn test_analyze_empty_frame() {
        let config = ColorTemperatureConfig::default();
        let result = analyze_color_temperature(&[], 0, 0, &config);
        assert!((result.cct_kelvin - 6500.0).abs() < 1.0);
    }

    #[test]
    fn test_analyze_white_frame() {
        let width = 64_u32;
        let height = 64_u32;
        let frame = vec![200u8; (width * height * 3) as usize];
        let config = ColorTemperatureConfig::default();
        let result = analyze_color_temperature(&frame, width, height, &config);
        // White should be near 6500K
        assert!(result.cct_kelvin > 4000.0 && result.cct_kelvin < 10000.0);
    }

    #[test]
    fn test_analyze_warm_frame() {
        let width = 32_u32;
        let height = 32_u32;
        let mut frame = vec![0u8; (width * height * 3) as usize];
        for i in 0..(width * height) as usize {
            frame[i * 3] = 200; // Red
            frame[i * 3 + 1] = 150; // Green
            frame[i * 3 + 2] = 100; // Blue
        }
        let config = ColorTemperatureConfig::default();
        let result = analyze_color_temperature(&frame, width, height, &config);
        // Warm image should be below 5000K
        assert!(result.cct_kelvin < 5500.0);
    }

    #[test]
    fn test_describe_cct() {
        assert_eq!(describe_cct(2000.0), "Warm (Candlelight)");
        assert_eq!(describe_cct(3000.0), "Warm (Tungsten/Halogen)");
        assert_eq!(describe_cct(4000.0), "Neutral (Fluorescent)");
        assert_eq!(describe_cct(5200.0), "Daylight (Direct Sun)");
        assert_eq!(describe_cct(6000.0), "Daylight (Overcast)");
        assert_eq!(describe_cct(7000.0), "Cool (Shade/Cloudy)");
        assert_eq!(describe_cct(10000.0), "Very Cool (Blue Sky)");
    }

    #[test]
    fn test_cct_to_rgb_tungsten() {
        let (r, g, b) = cct_to_rgb(3200.0);
        // Warm light should be reddish
        assert!(r > g);
        assert!(g > b);
    }

    #[test]
    fn test_cct_to_rgb_daylight() {
        let (r, g, b) = cct_to_rgb(6500.0);
        // Daylight should be approximately white
        assert!(r > 200);
        assert!(g > 200);
        assert!(b > 200);
    }

    #[test]
    fn test_compute_tint() {
        let xy = CieXy {
            x: 0.3127,
            y: 0.3290,
        };
        let tint = compute_tint(xy, 6500.0);
        // For D65, tint should be close to 0
        assert!(tint.abs() < 50.0);
    }

    #[test]
    fn test_planckian_xy() {
        let xy = planckian_xy(6500.0);
        // Should be close to D65
        assert!((xy.x - 0.313).abs() < 0.02);
    }

    #[test]
    fn test_color_temp_config_default() {
        let config = ColorTemperatureConfig::default();
        assert_eq!(config.grid_cols, 4);
        assert_eq!(config.grid_rows, 4);
        assert!(config.brightness_weighted);
    }
}
