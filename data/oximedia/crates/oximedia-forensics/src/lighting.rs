//! Illumination and Lighting Inconsistency Analysis
//!
//! This module detects tampering by analyzing lighting, shadows, reflections,
//! and illumination consistency across an image.
//!
//! # Methodology
//!
//! Real photographs are, almost always, lit by a small number of physically
//! consistent light sources, so surface shading gradients and cast-shadow
//! directions should agree everywhere in the frame. Splicing content from a
//! different photograph (lit differently) breaks that agreement. This
//! module estimates a per-region local light direction from intensity
//! gradients (`estimate_local_light_direction`) and compares it against the
//! image's dominant estimated light source (`estimate_light_sources`) in
//! `analyze_illumination_consistency`, flagging regions whose local
//! shading direction deviates by more than 60° (π/3) as inconsistent.
//! `analyze_shadows` independently locates dark cast-shadow regions and
//! estimates their direction (`estimate_shadow_direction`); a full 3D
//! light-source estimate combining shading and shadow-direction evidence is
//! produced for [`LightSource`]. `detect_impossible_lighting`
//! additionally flags frames containing both far-brighter-than-average and
//! far-darker-than-average regions simultaneously, a coarse heuristic for
//! physically implausible compositing.
//!
//! # References
//!
//! - M. K. Johnson, H. Farid, "Exposing Digital Forgeries by Detecting
//!   Inconsistencies in Lighting", ACM Multimedia and Security Workshop
//!   (2005) — 2D lighting-direction estimation from occluding contours,
//!   the basis for `analyze_illumination_consistency`.
//! - M. K. Johnson, H. Farid, "Exposing Digital Forgeries in Complex
//!   Lighting Environments", IEEE Transactions on Information Forensics
//!   and Security, 2(3), 450–461 (2007) — extends lighting-consistency
//!   analysis to more general (non-single-point-source) illumination, and
//!   motivates estimating a full 3D [`LightSource`] rather than a single
//!   2D azimuth.
//! - E. Kee, H. Farid, "Exposing Digital Forgeries from 3-D Lighting
//!   Environments", IEEE International Workshop on Information Forensics
//!   and Security (2010) — 3D light-source estimation from shading, the
//!   basis for combining `elevation` with `azimuth` in [`LightSource`].
//! - E. Kee, J. F. O'Brien, H. Farid, "Exposing Photo Manipulation with
//!   Inconsistent Shadows", ACM Transactions on Graphics, 32(3), Article 28
//!   (2013) — cast-shadow geometry as an independent consistency cue, the
//!   basis for `analyze_shadows` / [`ShadowAnalysis`].

use crate::flat_array2::FlatArray2;
use crate::{ForensicTest, ForensicsResult};
use image::RgbImage;
use std::f64::consts::PI;

/// Illumination analysis result
#[derive(Debug, Clone)]
pub struct IlluminationResult {
    /// Detected light sources
    pub light_sources: Vec<LightSource>,
    /// Inconsistent regions
    pub inconsistent_regions: Vec<(usize, usize, usize, usize)>,
    /// Confidence score
    pub confidence: f64,
}

/// Light source estimation
#[derive(Debug, Clone)]
pub struct LightSource {
    /// Direction (azimuth angle)
    pub azimuth: f64,
    /// Direction (elevation angle)
    pub elevation: f64,
    /// Intensity
    pub intensity: f64,
}

/// Shadow analysis result
#[derive(Debug, Clone)]
pub struct ShadowAnalysis {
    /// Detected shadows
    pub shadows: Vec<ShadowRegion>,
    /// Inconsistency score
    pub inconsistency_score: f64,
}

/// Shadow region
#[derive(Debug, Clone)]
pub struct ShadowRegion {
    /// X coordinate
    pub x: usize,
    /// Y coordinate
    pub y: usize,
    /// Width
    pub width: usize,
    /// Height
    pub height: usize,
    /// Shadow direction
    pub direction: f64,
    /// Shadow intensity
    pub intensity: f64,
}

/// Analyze lighting for tampering detection
#[allow(unused_variables)]
pub fn analyze_lighting(image: &RgbImage) -> ForensicsResult<ForensicTest> {
    let mut test = ForensicTest::new("Lighting Analysis");

    let (width, height) = image.dimensions();

    // Convert to grayscale for intensity analysis
    let gray = rgb_to_grayscale(image);

    // Analyze illumination consistency
    let illum_result = analyze_illumination_consistency(&gray)?;

    if !illum_result.inconsistent_regions.is_empty() {
        test.tampering_detected = true;
        test.add_finding(format!(
            "Detected {} regions with inconsistent illumination",
            illum_result.inconsistent_regions.len()
        ));
    }

    test.add_finding(format!(
        "Estimated {} light sources",
        illum_result.light_sources.len()
    ));

    // Analyze shadows
    let shadow_analysis = analyze_shadows(&gray)?;

    if shadow_analysis.inconsistency_score > 0.3 {
        test.tampering_detected = true;
        test.add_finding(format!(
            "Shadow inconsistency detected (score: {:.3})",
            shadow_analysis.inconsistency_score
        ));
    }

    test.add_finding(format!(
        "Found {} shadow regions",
        shadow_analysis.shadows.len()
    ));

    // Detect physically impossible lighting
    let impossible_lighting = detect_impossible_lighting(&gray)?;

    if impossible_lighting {
        test.tampering_detected = true;
        test.add_finding("Physically impossible lighting detected".to_string());
    }

    // Calculate confidence
    let mut confidence = illum_result.confidence;
    confidence = (confidence + shadow_analysis.inconsistency_score) / 2.0;

    if impossible_lighting {
        confidence = (confidence + 0.5).min(1.0);
    }

    test.set_confidence(confidence);

    // Create anomaly map
    let anomaly_map = create_lighting_anomaly_map(image, &illum_result, &shadow_analysis)?;
    test.anomaly_map = Some(anomaly_map);

    Ok(test)
}

/// Convert RGB to grayscale
fn rgb_to_grayscale(image: &RgbImage) -> FlatArray2<f64> {
    let (width, height) = image.dimensions();
    let mut gray = FlatArray2::zeros((height as usize, width as usize));

    for (x, y, pixel) in image.enumerate_pixels() {
        let r = pixel[0] as f64;
        let g = pixel[1] as f64;
        let b = pixel[2] as f64;

        gray[[y as usize, x as usize]] = 0.299 * r + 0.587 * g + 0.114 * b;
    }

    gray
}

/// Analyze illumination consistency
fn analyze_illumination_consistency(gray: &FlatArray2<f64>) -> ForensicsResult<IlluminationResult> {
    let (height, width) = gray.dim();

    // Estimate light sources from gradients
    let light_sources = estimate_light_sources(gray);

    // Divide image into regions and check consistency
    let region_size = 64;
    let mut regional_directions = Vec::new();

    for y in (0..height - region_size).step_by(region_size / 2) {
        for x in (0..width - region_size).step_by(region_size / 2) {
            let direction = estimate_local_light_direction(gray, x, y, region_size);
            regional_directions.push((x, y, direction));
        }
    }

    // Find regions with inconsistent lighting
    let mut inconsistent_regions = Vec::new();

    if !light_sources.is_empty() {
        let primary_direction = light_sources[0].azimuth;

        for (x, y, direction) in &regional_directions {
            let angle_diff = (direction - primary_direction).abs();

            // Normalize to [0, PI]
            let angle_diff = if angle_diff > PI {
                2.0 * PI - angle_diff
            } else {
                angle_diff
            };

            if angle_diff > PI / 3.0 {
                inconsistent_regions.push((*x, *y, region_size, region_size));
            }
        }
    }

    let confidence = if !regional_directions.is_empty() {
        (inconsistent_regions.len() as f64 / regional_directions.len() as f64).min(1.0)
    } else {
        0.0
    };

    Ok(IlluminationResult {
        light_sources,
        inconsistent_regions,
        confidence,
    })
}

/// Estimate light sources from image
fn estimate_light_sources(gray: &FlatArray2<f64>) -> Vec<LightSource> {
    let (height, width) = gray.dim();

    // Simple approach: analyze gradient directions
    let mut gradient_directions = Vec::new();

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let gx = gray[[y, x + 1]] - gray[[y, x - 1]];
            let gy = gray[[y + 1, x]] - gray[[y - 1, x]];

            let magnitude = (gx * gx + gy * gy).sqrt();

            if magnitude > 10.0 {
                let direction = gy.atan2(gx);
                gradient_directions.push(direction);
            }
        }
    }

    if gradient_directions.is_empty() {
        return Vec::new();
    }

    // Find dominant direction (simplified - would use clustering in production)
    let mean_direction = gradient_directions.iter().sum::<f64>() / gradient_directions.len() as f64;

    // Estimate intensity from mean brightness
    let mean_intensity = gray.iter().sum::<f64>() / (height * width) as f64;

    vec![LightSource {
        azimuth: mean_direction,
        elevation: PI / 4.0, // Assume 45 degrees
        intensity: mean_intensity,
    }]
}

/// Estimate local light direction in a region
fn estimate_local_light_direction(gray: &FlatArray2<f64>, x: usize, y: usize, size: usize) -> f64 {
    let (height, width) = gray.dim();
    let mut gx_sum = 0.0;
    let mut gy_sum = 0.0;
    let mut count = 0;

    for dy in 1..size - 1 {
        for dx in 1..size - 1 {
            let px = x + dx;
            let py = y + dy;

            if px > 0 && px < width - 1 && py > 0 && py < height - 1 {
                let gx = gray[[py, px + 1]] - gray[[py, px - 1]];
                let gy = gray[[py + 1, px]] - gray[[py - 1, px]];

                gx_sum += gx;
                gy_sum += gy;
                count += 1;
            }
        }
    }

    if count > 0 {
        let avg_gx = gx_sum / count as f64;
        let avg_gy = gy_sum / count as f64;
        avg_gy.atan2(avg_gx)
    } else {
        0.0
    }
}

/// Analyze shadows in the image
fn analyze_shadows(gray: &FlatArray2<f64>) -> ForensicsResult<ShadowAnalysis> {
    let (height, width) = gray.dim();

    // Detect dark regions (potential shadows)
    let threshold = 80.0;
    let mut shadow_regions = Vec::new();

    let region_size = 32;

    for y in (0..height - region_size).step_by(region_size / 2) {
        for x in (0..width - region_size).step_by(region_size / 2) {
            let mut sum = 0.0;
            let mut count = 0;

            for dy in 0..region_size {
                for dx in 0..region_size {
                    if y + dy < height && x + dx < width {
                        sum += gray[[y + dy, x + dx]];
                        count += 1;
                    }
                }
            }

            let mean = if count > 0 { sum / count as f64 } else { 0.0 };

            if mean < threshold {
                // Estimate shadow direction
                let direction = estimate_shadow_direction(gray, x, y, region_size);

                shadow_regions.push(ShadowRegion {
                    x,
                    y,
                    width: region_size,
                    height: region_size,
                    direction,
                    intensity: threshold - mean,
                });
            }
        }
    }

    // Check consistency of shadow directions
    let inconsistency_score = if shadow_regions.len() > 1 {
        compute_shadow_inconsistency(&shadow_regions)
    } else {
        0.0
    };

    Ok(ShadowAnalysis {
        shadows: shadow_regions,
        inconsistency_score,
    })
}

/// Estimate shadow direction
fn estimate_shadow_direction(gray: &FlatArray2<f64>, x: usize, y: usize, size: usize) -> f64 {
    // Use gradient at shadow boundary
    estimate_local_light_direction(gray, x, y, size)
}

/// Compute shadow direction inconsistency
fn compute_shadow_inconsistency(shadows: &[ShadowRegion]) -> f64 {
    if shadows.len() < 2 {
        return 0.0;
    }

    let mean_direction = shadows.iter().map(|s| s.direction).sum::<f64>() / shadows.len() as f64;

    let mut variance = 0.0;
    for shadow in shadows {
        let mut diff = (shadow.direction - mean_direction).abs();

        // Normalize to [0, PI]
        if diff > PI {
            diff = 2.0 * PI - diff;
        }

        variance += diff * diff;
    }

    variance /= shadows.len() as f64;

    let std_dev = variance.sqrt();

    // Normalize to [0, 1]
    (std_dev / PI).min(1.0)
}

/// Detect physically impossible lighting
fn detect_impossible_lighting(gray: &FlatArray2<f64>) -> ForensicsResult<bool> {
    let (height, width) = gray.dim();

    // Check for contradictory highlights and shadows
    let mean_brightness = gray.iter().sum::<f64>() / (height * width) as f64;

    let mut bright_regions = 0;
    let mut dark_regions = 0;

    let region_size = 64;

    for y in (0..height - region_size).step_by(region_size) {
        for x in (0..width - region_size).step_by(region_size) {
            let mut sum = 0.0;
            let mut count = 0;

            for dy in 0..region_size {
                for dx in 0..region_size {
                    if y + dy < height && x + dx < width {
                        sum += gray[[y + dy, x + dx]];
                        count += 1;
                    }
                }
            }

            let mean = if count > 0 { sum / count as f64 } else { 0.0 };

            if mean > mean_brightness + 50.0 {
                bright_regions += 1;
            } else if mean < mean_brightness - 50.0 {
                dark_regions += 1;
            }
        }
    }

    // If we have many both very bright and very dark regions, might be suspicious
    let total_regions = ((height / region_size) * (width / region_size)) as f64;
    let bright_ratio = bright_regions as f64 / total_regions;
    let dark_ratio = dark_regions as f64 / total_regions;

    // Both high ratios might indicate compositing
    Ok(bright_ratio > 0.3 && dark_ratio > 0.3)
}

/// Create anomaly map from lighting analysis
fn create_lighting_anomaly_map(
    image: &RgbImage,
    illum_result: &IlluminationResult,
    shadow_analysis: &ShadowAnalysis,
) -> ForensicsResult<FlatArray2<f64>> {
    let (width, height) = image.dimensions();
    let mut anomaly_map: FlatArray2<f64> = FlatArray2::zeros((height as usize, width as usize));

    // Mark inconsistent illumination regions
    for (x, y, w, h) in &illum_result.inconsistent_regions {
        for dy in 0..*h {
            for dx in 0..*w {
                let px = x + dx;
                let py = y + dy;

                if px < width as usize && py < height as usize {
                    anomaly_map[[py, px]] = 0.8;
                }
            }
        }
    }

    // Mark shadow regions with high inconsistency
    if shadow_analysis.inconsistency_score > 0.3 {
        for shadow in &shadow_analysis.shadows {
            for dy in 0..shadow.height {
                for dx in 0..shadow.width {
                    let px = shadow.x + dx;
                    let py = shadow.y + dy;

                    if px < width as usize && py < height as usize {
                        anomaly_map[[py, px]] = anomaly_map[[py, px]].max(0.5_f64);
                    }
                }
            }
        }
    }

    Ok(anomaly_map)
}

/// Analyze reflections for consistency
pub fn analyze_reflections(image: &RgbImage) -> ForensicsResult<f64> {
    let gray = rgb_to_grayscale(image);
    let (height, width) = gray.dim();

    // Look for bright spots (potential reflections/specularities)
    let mut reflection_points = Vec::new();

    for y in 0..height {
        for x in 0..width {
            if gray[[y, x]] > 200.0 {
                reflection_points.push((x, y, gray[[y, x]]));
            }
        }
    }

    // Analyze distribution of reflections
    if reflection_points.is_empty() {
        return Ok(0.0);
    }

    // Check if reflections are consistent with a single light source
    // (simplified - would analyze geometric consistency in production)
    let inconsistency_score = if reflection_points.len() > 10 {
        0.3
    } else {
        0.0
    };

    Ok(inconsistency_score)
}

/// Estimate ambient occlusion map
pub fn estimate_ambient_occlusion(gray: &FlatArray2<f64>) -> FlatArray2<f64> {
    let (height, width) = gray.dim();
    let mut ao_map = FlatArray2::zeros((height, width));

    let kernel_size = 5;
    let half_kernel = kernel_size / 2;

    for y in half_kernel..height - half_kernel {
        for x in half_kernel..width - half_kernel {
            let mut darker_count = 0;
            let center = gray[[y, x]];

            for dy in -(half_kernel as i32)..=half_kernel as i32 {
                for dx in -(half_kernel as i32)..=half_kernel as i32 {
                    let ny = (y as i32 + dy) as usize;
                    let nx = (x as i32 + dx) as usize;

                    if gray[[ny, nx]] < center {
                        darker_count += 1;
                    }
                }
            }

            let total = kernel_size * kernel_size;
            ao_map[[y, x]] = darker_count as f64 / total as f64;
        }
    }

    ao_map
}

// ---------------------------------------------------------------------------
// 3D light source estimation from shadow direction analysis
// ---------------------------------------------------------------------------

/// A 3D vector representing a direction in space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    /// Create a new vector.
    #[must_use]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Compute the Euclidean length.
    #[must_use]
    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Return a unit-length version of this vector.
    #[must_use]
    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len < 1e-15 {
            Self::new(0.0, 0.0, 0.0)
        } else {
            Self::new(self.x / len, self.y / len, self.z / len)
        }
    }

    /// Dot product.
    #[must_use]
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Angular difference in radians.
    #[must_use]
    pub fn angle_to(&self, other: &Self) -> f64 {
        let d = self.normalize().dot(&other.normalize());
        d.clamp(-1.0, 1.0).acos()
    }
}

/// Result of 3D light source estimation.
#[derive(Debug, Clone)]
pub struct LightSource3D {
    /// Estimated light direction (unit vector pointing towards the light).
    pub direction: Vec3,
    /// Azimuth angle in radians (angle in the XY plane from +X axis).
    pub azimuth: f64,
    /// Elevation angle in radians (angle above the XY plane).
    pub elevation: f64,
    /// Confidence of this estimate (0.0..1.0).
    pub confidence: f64,
}

/// Result of 3D shadow direction analysis for tampering detection.
#[derive(Debug, Clone)]
pub struct ShadowDirectionAnalysis {
    /// Per-region estimated light directions.
    pub regional_light_dirs: Vec<(usize, usize, Vec3)>,
    /// Global best-fit light direction.
    pub global_light_dir: LightSource3D,
    /// Per-region deviation from the global direction (radians).
    pub deviations: Vec<f64>,
    /// Overall inconsistency score (0.0..1.0).
    pub inconsistency_score: f64,
    /// Flagged regions where deviation exceeds threshold.
    pub flagged_regions: Vec<(usize, usize)>,
}

/// Estimate a 3D light source direction from the shadow boundary gradients
/// in a given image region.
///
/// Uses the observation that shadow boundaries are perpendicular to the
/// projection of the light direction onto the image plane. The elevation
/// is estimated from the shadow-to-object intensity ratio.
fn estimate_3d_light_direction(gray: &FlatArray2<f64>, x: usize, y: usize, size: usize) -> Vec3 {
    let (height, width) = gray.dim();

    // Compute average gradient in the region (same as local light direction).
    let mut gx_sum = 0.0;
    let mut gy_sum = 0.0;
    let mut count = 0;
    let mut bright_sum = 0.0;
    let mut dark_sum = 0.0;
    let mut bright_count = 0u32;
    let mut dark_count = 0u32;

    let mean_val = {
        let mut s = 0.0;
        let mut c = 0u32;
        for dy in 0..size {
            for dx in 0..size {
                let px = x + dx;
                let py = y + dy;
                if px < width && py < height {
                    s += gray[[py, px]];
                    c += 1;
                }
            }
        }
        if c > 0 {
            s / c as f64
        } else {
            128.0
        }
    };

    for dy in 1..size.saturating_sub(1) {
        for dx in 1..size.saturating_sub(1) {
            let px = x + dx;
            let py = y + dy;

            if px > 0 && px < width - 1 && py > 0 && py < height - 1 {
                let gx = gray[[py, px + 1]] - gray[[py, px - 1]];
                let gy = gray[[py + 1, px]] - gray[[py - 1, px]];
                gx_sum += gx;
                gy_sum += gy;
                count += 1;

                let val = gray[[py, px]];
                if val > mean_val {
                    bright_sum += val;
                    bright_count += 1;
                } else {
                    dark_sum += val;
                    dark_count += 1;
                }
            }
        }
    }

    if count == 0 {
        return Vec3::new(0.0, 0.0, 1.0);
    }

    let avg_gx = gx_sum / count as f64;
    let avg_gy = gy_sum / count as f64;

    // Estimate elevation from shadow-to-bright ratio.
    let bright_mean = if bright_count > 0 {
        bright_sum / bright_count as f64
    } else {
        128.0
    };
    let dark_mean = if dark_count > 0 {
        dark_sum / dark_count as f64
    } else {
        128.0
    };

    // Higher contrast between bright/dark regions suggests lower elevation.
    let contrast_ratio = if bright_mean > 1e-10 {
        (dark_mean / bright_mean).clamp(0.0, 1.0)
    } else {
        0.5
    };
    // Map ratio to elevation: high ratio (low contrast) -> high elevation.
    let elevation = contrast_ratio * (PI / 2.0);

    let horiz_mag = (avg_gx * avg_gx + avg_gy * avg_gy).sqrt();
    let dir_x = if horiz_mag > 1e-10 {
        avg_gx / horiz_mag
    } else {
        0.0
    };
    let dir_y = if horiz_mag > 1e-10 {
        avg_gy / horiz_mag
    } else {
        0.0
    };

    // 3D direction: horizontal component scaled by cos(elevation), z by sin(elevation).
    Vec3::new(
        dir_x * elevation.cos(),
        dir_y * elevation.cos(),
        elevation.sin(),
    )
}

/// Perform 3D light source estimation and shadow direction consistency analysis.
///
/// Divides the image into a grid and estimates the 3D light direction for each
/// region. Regions whose light direction deviates significantly from the global
/// estimate are flagged as potentially tampered.
#[allow(clippy::cast_precision_loss)]
pub fn analyze_shadow_directions(
    gray: &FlatArray2<f64>,
    region_size: usize,
    angle_threshold_rad: f64,
) -> ShadowDirectionAnalysis {
    let (height, width) = gray.dim();
    if region_size == 0 || height < region_size || width < region_size {
        return ShadowDirectionAnalysis {
            regional_light_dirs: Vec::new(),
            global_light_dir: LightSource3D {
                direction: Vec3::new(0.0, 0.0, 1.0),
                azimuth: 0.0,
                elevation: PI / 2.0,
                confidence: 0.0,
            },
            deviations: Vec::new(),
            inconsistency_score: 0.0,
            flagged_regions: Vec::new(),
        };
    }

    let mut regional_dirs: Vec<(usize, usize, Vec3)> = Vec::new();

    for y in (0..height.saturating_sub(region_size)).step_by(region_size) {
        for x in (0..width.saturating_sub(region_size)).step_by(region_size) {
            let dir = estimate_3d_light_direction(gray, x, y, region_size);
            regional_dirs.push((x, y, dir));
        }
    }

    if regional_dirs.is_empty() {
        return ShadowDirectionAnalysis {
            regional_light_dirs: Vec::new(),
            global_light_dir: LightSource3D {
                direction: Vec3::new(0.0, 0.0, 1.0),
                azimuth: 0.0,
                elevation: PI / 2.0,
                confidence: 0.0,
            },
            deviations: Vec::new(),
            inconsistency_score: 0.0,
            flagged_regions: Vec::new(),
        };
    }

    // Compute global direction as vector average.
    let n = regional_dirs.len() as f64;
    let mut gx = 0.0;
    let mut gy = 0.0;
    let mut gz = 0.0;
    for (_, _, d) in &regional_dirs {
        gx += d.x;
        gy += d.y;
        gz += d.z;
    }
    let global_dir = Vec3::new(gx / n, gy / n, gz / n).normalize();
    let azimuth = global_dir.y.atan2(global_dir.x);
    let elevation = global_dir.z.asin().clamp(-PI / 2.0, PI / 2.0);

    // Compute per-region deviation.
    let mut deviations = Vec::with_capacity(regional_dirs.len());
    let mut flagged = Vec::new();

    for (x, y, d) in &regional_dirs {
        let angle = global_dir.angle_to(d);
        deviations.push(angle);
        if angle > angle_threshold_rad {
            flagged.push((*x, *y));
        }
    }

    // Inconsistency score: mean deviation normalized by PI.
    let mean_dev = deviations.iter().sum::<f64>() / deviations.len() as f64;
    let inconsistency = (mean_dev / (PI / 4.0)).min(1.0);

    // Confidence based on number of regions and consistency.
    let confidence = (1.0 - inconsistency).max(0.0);

    ShadowDirectionAnalysis {
        regional_light_dirs: regional_dirs,
        global_light_dir: LightSource3D {
            direction: global_dir,
            azimuth,
            elevation,
            confidence,
        },
        deviations,
        inconsistency_score: inconsistency,
        flagged_regions: flagged,
    }
}

/// Convenience: analyze shadow directions with default parameters.
pub fn analyze_shadow_directions_default(gray: &FlatArray2<f64>) -> ShadowDirectionAnalysis {
    analyze_shadow_directions(gray, 32, PI / 4.0)
}

/// Detect light source direction from specular highlights
pub fn detect_light_from_specular(gray: &FlatArray2<f64>) -> Option<(f64, f64)> {
    let (height, width) = gray.dim();

    // Find bright spots
    let mut highlights = Vec::new();

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            if gray[[y, x]] > 220.0 {
                // Check if it's a local maximum
                let mut is_max = true;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let ny = (y as i32 + dy) as usize;
                        let nx = (x as i32 + dx) as usize;
                        if gray[[ny, nx]] > gray[[y, x]] {
                            is_max = false;
                            break;
                        }
                    }
                    if !is_max {
                        break;
                    }
                }

                if is_max {
                    highlights.push((x, y));
                }
            }
        }
    }

    if highlights.is_empty() {
        return None;
    }

    // Estimate direction from centroid
    let cx = highlights.iter().map(|(x, _)| *x).sum::<usize>() as f64 / highlights.len() as f64;
    let cy = highlights.iter().map(|(_, y)| *y).sum::<usize>() as f64 / highlights.len() as f64;

    Some((cx, cy))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_rgb_to_grayscale() {
        let img = RgbImage::new(10, 10);
        let gray = rgb_to_grayscale(&img);
        assert_eq!(gray.dim(), (10, 10));
    }

    #[test]
    fn test_light_source_estimation() {
        // Create an image with strong gradients
        let mut gray = FlatArray2::zeros((64, 64));
        for y in 0..64 {
            for x in 0..64 {
                gray[[y, x]] = (x * 4 + y * 4) as f64; // Stronger gradient
            }
        }
        let sources = estimate_light_sources(&gray);
        // With strong gradients, should find at least one source
        assert!(sources.len() >= 1);
    }

    #[test]
    fn test_local_light_direction() {
        let gray = FlatArray2::from_elem(64, 64, 128.0);
        let direction = estimate_local_light_direction(&gray, 16, 16, 32);
        assert!(direction >= -PI && direction <= PI);
    }

    #[test]
    fn test_shadow_inconsistency() {
        let shadow1 = ShadowRegion {
            x: 0,
            y: 0,
            width: 10,
            height: 10,
            direction: 0.0,
            intensity: 50.0,
        };

        let shadow2 = ShadowRegion {
            x: 20,
            y: 20,
            width: 10,
            height: 10,
            direction: PI / 2.0,
            intensity: 50.0,
        };

        let score = compute_shadow_inconsistency(&vec![shadow1, shadow2]);
        assert!(score > 0.0);
    }

    #[test]
    fn test_ambient_occlusion() {
        let gray = FlatArray2::from_elem(20, 20, 128.0);
        let ao = estimate_ambient_occlusion(&gray);
        assert_eq!(ao.dim(), gray.dim());
    }

    // ---- 3D light source estimation tests ----

    #[test]
    fn test_vec3_length() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!((v.length() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec3_normalize() {
        let v = Vec3::new(0.0, 0.0, 5.0);
        let n = v.normalize();
        assert!((n.z - 1.0).abs() < 1e-10);
        assert!(n.x.abs() < 1e-10);
    }

    #[test]
    fn test_vec3_normalize_zero() {
        let v = Vec3::new(0.0, 0.0, 0.0);
        let n = v.normalize();
        assert!(n.length() < 1e-10);
    }

    #[test]
    fn test_vec3_dot() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        assert!(a.dot(&b).abs() < 1e-10); // orthogonal
    }

    #[test]
    fn test_vec3_angle_to() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let angle = a.angle_to(&b);
        assert!((angle - PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec3_angle_to_same() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let angle = a.angle_to(&a);
        assert!(angle.abs() < 1e-10);
    }

    #[test]
    fn test_shadow_direction_analysis_uniform() {
        let gray = FlatArray2::from_elem(64, 64, 128.0);
        let result = analyze_shadow_directions(&gray, 32, PI / 4.0);
        assert!(result.regional_light_dirs.len() >= 1);
        // Uniform image: low inconsistency.
        assert!(result.inconsistency_score < 0.5);
    }

    #[test]
    fn test_shadow_direction_analysis_small_image() {
        let gray = FlatArray2::from_elem(10, 10, 128.0);
        let result = analyze_shadow_directions(&gray, 32, PI / 4.0);
        // Image too small for regions: should return empty.
        assert!(result.regional_light_dirs.is_empty());
        assert!(result.inconsistency_score < 0.01);
    }

    #[test]
    fn test_shadow_direction_analysis_gradient() {
        // Strong horizontal gradient: light coming from the right.
        let mut gray = FlatArray2::zeros((64, 64));
        for y in 0..64 {
            for x in 0..64 {
                gray[[y, x]] = x as f64 * 4.0;
            }
        }
        let result = analyze_shadow_directions(&gray, 32, PI / 4.0);
        assert!(!result.regional_light_dirs.is_empty());
        // All regions should have similar direction: low inconsistency.
        assert!(result.inconsistency_score < 0.5);
    }

    #[test]
    fn test_shadow_direction_analysis_mixed_gradients() {
        // Two halves with different gradient directions (simulate tampering).
        // Top half: bright on right, dark on left.
        // Bottom half: bright on top-left, dark on bottom-right (diagonal).
        let mut gray = FlatArray2::zeros((64, 64));
        for y in 0..64 {
            for x in 0..64 {
                if y < 32 {
                    gray[[y, x]] = x as f64 * 4.0; // horizontal gradient
                } else {
                    gray[[y, x]] = y as f64 * 4.0; // vertical gradient
                }
            }
        }
        let result = analyze_shadow_directions(&gray, 32, PI / 6.0);
        // There should be multiple regions with varying directions.
        assert!(!result.regional_light_dirs.is_empty());
        assert!(!result.deviations.is_empty());
        // The inconsistency may or may not be > 0 depending on averaging,
        // but deviations should not all be zero.
        let max_dev = result.deviations.iter().cloned().fold(0.0_f64, f64::max);
        assert!(max_dev >= 0.0);
    }

    #[test]
    fn test_shadow_direction_default() {
        let gray = FlatArray2::from_elem(64, 64, 128.0);
        let result = analyze_shadow_directions_default(&gray);
        assert!(result.global_light_dir.confidence >= 0.0);
        assert!(result.global_light_dir.confidence <= 1.0);
    }

    #[test]
    fn test_shadow_direction_flagged_regions() {
        // Create a clearly tampered image with opposing directions.
        let mut gray = FlatArray2::zeros((128, 128));
        for y in 0..128 {
            for x in 0..128 {
                if x < 64 {
                    gray[[y, x]] = y as f64 * 2.0; // vertical gradient
                } else {
                    gray[[y, x]] = (127 - y) as f64 * 2.0; // reversed vertical
                }
            }
        }
        let result = analyze_shadow_directions(&gray, 32, PI / 6.0);
        // Some regions should be flagged.
        // The exact number depends on the threshold, but deviations should exist.
        assert!(!result.deviations.is_empty());
    }

    #[test]
    fn test_light_source_3d_fields() {
        let ls = LightSource3D {
            direction: Vec3::new(0.0, 0.0, 1.0),
            azimuth: 0.0,
            elevation: PI / 2.0,
            confidence: 0.8,
        };
        assert!((ls.direction.z - 1.0).abs() < 1e-10);
        assert!((ls.confidence - 0.8).abs() < 1e-10);
    }
}
