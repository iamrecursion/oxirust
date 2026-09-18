#![allow(dead_code)]
//! Shadow consistency analysis for image tampering detection.
//!
//! This module analyzes shadow directions, intensities, and penumbra characteristics
//! to detect inconsistencies that indicate compositing or manipulation.
//!
//! # Features
//!
//! - **Shadow direction estimation** from gradient analysis
//! - **Shadow consistency scoring** across image regions
//! - **Penumbra width analysis** for detecting pasted shadows
//! - **Light source estimation** from shadow geometry
//! - **Shadow boundary detection** using intensity gradients

/// A 2D direction vector representing estimated shadow direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadowDirection {
    /// Horizontal component of the shadow direction (normalized).
    pub dx: f64,
    /// Vertical component of the shadow direction (normalized).
    pub dy: f64,
    /// Confidence of the direction estimate (0.0 to 1.0).
    pub confidence: f64,
}

impl ShadowDirection {
    /// Create a new shadow direction.
    #[must_use]
    pub fn new(dx: f64, dy: f64, confidence: f64) -> Self {
        let mag = (dx * dx + dy * dy).sqrt();
        if mag > 1e-15 {
            Self {
                dx: dx / mag,
                dy: dy / mag,
                confidence: confidence.clamp(0.0, 1.0),
            }
        } else {
            Self {
                dx: 0.0,
                dy: 0.0,
                confidence: 0.0,
            }
        }
    }

    /// Compute the angle of the shadow direction in radians.
    #[must_use]
    pub fn angle(&self) -> f64 {
        self.dy.atan2(self.dx)
    }

    /// Compute the angular difference between two shadow directions in radians.
    #[must_use]
    pub fn angular_difference(&self, other: &Self) -> f64 {
        let dot = self.dx * other.dx + self.dy * other.dy;
        dot.clamp(-1.0, 1.0).acos()
    }

    /// Check if two shadow directions are consistent (within threshold radians).
    #[must_use]
    pub fn is_consistent(&self, other: &Self, threshold_radians: f64) -> bool {
        self.angular_difference(other) < threshold_radians
    }
}

/// A detected shadow region in an image.
#[derive(Debug, Clone)]
pub struct ShadowRegion {
    /// Region identifier.
    pub id: u32,
    /// Bounding box: (x, y, width, height).
    pub bounds: (u32, u32, u32, u32),
    /// Estimated shadow direction.
    pub direction: ShadowDirection,
    /// Average shadow intensity (0.0 to 1.0, where 0 is darkest).
    pub avg_intensity: f64,
    /// Penumbra width in pixels (soft shadow edge width).
    pub penumbra_width: f64,
    /// Area in pixels.
    pub area: u32,
}

impl ShadowRegion {
    /// Create a new shadow region.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: u32,
        bounds: (u32, u32, u32, u32),
        direction: ShadowDirection,
        avg_intensity: f64,
        penumbra_width: f64,
        area: u32,
    ) -> Self {
        Self {
            id,
            bounds,
            direction,
            avg_intensity,
            penumbra_width,
            area,
        }
    }

    /// Compute the center of this region.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn center(&self) -> (f64, f64) {
        (
            self.bounds.0 as f64 + self.bounds.2 as f64 / 2.0,
            self.bounds.1 as f64 + self.bounds.3 as f64 / 2.0,
        )
    }
}

/// An estimated light source position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightSource {
    /// Estimated direction to light (azimuth) in radians.
    pub azimuth: f64,
    /// Estimated elevation angle in radians (0 = horizon, pi/2 = directly overhead).
    pub elevation: f64,
    /// Confidence of the estimate.
    pub confidence: f64,
}

impl LightSource {
    /// Create a new light source estimate.
    #[must_use]
    pub fn new(azimuth: f64, elevation: f64, confidence: f64) -> Self {
        Self {
            azimuth,
            elevation: elevation.clamp(0.0, std::f64::consts::FRAC_PI_2),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Compute angular distance between two light source estimates.
    #[must_use]
    pub fn angular_distance(&self, other: &Self) -> f64 {
        let da = (self.azimuth - other.azimuth).abs();
        let da = da.min(2.0 * std::f64::consts::PI - da);
        let de = (self.elevation - other.elevation).abs();
        (da * da + de * de).sqrt()
    }
}

/// Configuration for shadow analysis.
#[derive(Debug, Clone)]
pub struct ShadowAnalysisConfig {
    /// Threshold for shadow detection (intensity ratio below this is considered shadow).
    pub shadow_threshold: f64,
    /// Angular consistency threshold in radians.
    pub direction_threshold: f64,
    /// Minimum shadow region area in pixels.
    pub min_region_area: u32,
    /// Number of gradient directions to sample.
    pub num_gradient_bins: u32,
    /// Penumbra consistency threshold (ratio).
    pub penumbra_threshold: f64,
}

impl Default for ShadowAnalysisConfig {
    fn default() -> Self {
        Self {
            shadow_threshold: 0.4,
            direction_threshold: 0.35, // ~20 degrees
            min_region_area: 100,
            num_gradient_bins: 36,
            penumbra_threshold: 2.0,
        }
    }
}

/// A detected shadow inconsistency.
#[derive(Debug, Clone)]
pub struct ShadowInconsistency {
    /// First region involved.
    pub region_a: u32,
    /// Second region involved.
    pub region_b: u32,
    /// Type of inconsistency.
    pub inconsistency_type: ShadowInconsistencyType,
    /// Angular or magnitude difference.
    pub difference: f64,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f64,
    /// Description.
    pub description: String,
}

/// Types of shadow inconsistency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowInconsistencyType {
    /// Shadow directions are inconsistent.
    DirectionMismatch,
    /// Penumbra widths are inconsistent (hard vs soft shadows).
    PenumbraMismatch,
    /// Shadow intensities are inconsistent.
    IntensityMismatch,
    /// Light source direction does not match shadow geometry.
    LightSourceMismatch,
}

/// Result of shadow analysis.
#[derive(Debug, Clone)]
pub struct ShadowAnalysisReport {
    /// Detected shadow regions.
    pub regions: Vec<ShadowRegion>,
    /// Estimated primary light source.
    pub primary_light: Option<LightSource>,
    /// Detected inconsistencies.
    pub inconsistencies: Vec<ShadowInconsistency>,
    /// Overall consistency score (1.0 = perfectly consistent, 0.0 = highly inconsistent).
    pub consistency_score: f64,
    /// Whether tampering is suspected based on shadow analysis.
    pub tampering_suspected: bool,
}

/// Shadow consistency analyzer.
#[derive(Debug, Clone)]
pub struct ShadowAnalyzer {
    /// Configuration.
    config: ShadowAnalysisConfig,
}

impl ShadowAnalyzer {
    /// Create a new shadow analyzer.
    #[must_use]
    pub fn new(config: ShadowAnalysisConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self {
            config: ShadowAnalysisConfig::default(),
        }
    }

    /// Analyze shadow consistency given pre-detected shadow regions.
    #[must_use]
    pub fn analyze_regions(&self, regions: &[ShadowRegion]) -> ShadowAnalysisReport {
        let mut inconsistencies = Vec::new();

        // Estimate primary light direction from shadow regions
        let primary_light = self.estimate_light_source(regions);

        // Check pairwise direction consistency
        for i in 0..regions.len() {
            for j in (i + 1)..regions.len() {
                // Direction consistency
                let angle_diff = regions[i]
                    .direction
                    .angular_difference(&regions[j].direction);
                if angle_diff > self.config.direction_threshold
                    && regions[i].direction.confidence > 0.3
                    && regions[j].direction.confidence > 0.3
                {
                    let confidence = (angle_diff / std::f64::consts::PI).min(1.0)
                        * regions[i].direction.confidence
                        * regions[j].direction.confidence;
                    inconsistencies.push(ShadowInconsistency {
                        region_a: regions[i].id,
                        region_b: regions[j].id,
                        inconsistency_type: ShadowInconsistencyType::DirectionMismatch,
                        difference: angle_diff,
                        confidence,
                        description: format!(
                            "Shadow direction differs by {:.1} degrees between regions {} and {}",
                            angle_diff.to_degrees(),
                            regions[i].id,
                            regions[j].id
                        ),
                    });
                }

                // Penumbra consistency
                let pa = regions[i].penumbra_width;
                let pb = regions[j].penumbra_width;
                if pa > 0.0 && pb > 0.0 {
                    let ratio = if pa > pb { pa / pb } else { pb / pa };
                    if ratio > self.config.penumbra_threshold {
                        inconsistencies.push(ShadowInconsistency {
                            region_a: regions[i].id,
                            region_b: regions[j].id,
                            inconsistency_type: ShadowInconsistencyType::PenumbraMismatch,
                            difference: ratio,
                            confidence: ((ratio - 1.0) / self.config.penumbra_threshold).min(1.0),
                            description: format!(
                                "Penumbra width ratio {:.1}x between regions {} and {}",
                                ratio, regions[i].id, regions[j].id
                            ),
                        });
                    }
                }
            }
        }

        // Compute overall consistency
        let consistency_score = self.compute_consistency_score(regions, &inconsistencies);
        let tampering_suspected = consistency_score < 0.5;

        ShadowAnalysisReport {
            regions: regions.to_vec(),
            primary_light,
            inconsistencies,
            consistency_score,
            tampering_suspected,
        }
    }

    /// Detect shadow regions in a grayscale image.
    ///
    /// Returns a list of detected shadow regions based on intensity thresholding
    /// and connected component analysis.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn detect_shadows(&self, data: &[u8], width: u32, height: u32) -> Vec<ShadowRegion> {
        let mut regions = Vec::new();

        if data.is_empty() || width == 0 || height == 0 {
            return regions;
        }

        // Compute global average intensity
        let total: f64 = data.iter().map(|&p| f64::from(p)).sum();
        let avg = total / data.len() as f64;
        let shadow_limit = avg * self.config.shadow_threshold;

        // Simple grid-based region detection
        let grid_size = 64u32;
        let mut region_id = 0u32;

        let cols = (width + grid_size - 1) / grid_size;
        let rows = (height + grid_size - 1) / grid_size;

        for gy in 0..rows {
            for gx in 0..cols {
                let x0 = gx * grid_size;
                let y0 = gy * grid_size;
                let x1 = (x0 + grid_size).min(width);
                let y1 = (y0 + grid_size).min(height);

                let mut shadow_count = 0u32;
                let mut intensity_sum = 0.0f64;
                let mut pixel_count = 0u32;

                for y in y0..y1 {
                    for x in x0..x1 {
                        let idx = (y * width + x) as usize;
                        if idx < data.len() {
                            let val = f64::from(data[idx]);
                            pixel_count += 1;
                            if val < shadow_limit {
                                shadow_count += 1;
                                intensity_sum += val;
                            }
                        }
                    }
                }

                if shadow_count >= self.config.min_region_area.min(pixel_count / 2)
                    && shadow_count > 0
                {
                    let avg_shadow_intensity = intensity_sum / f64::from(shadow_count) / 255.0;

                    // Estimate local gradient direction
                    let direction =
                        self.estimate_local_direction(data, width, height, x0, y0, x1, y1);

                    // Estimate penumbra width
                    let penumbra =
                        self.estimate_penumbra(data, width, x0, y0, x1, y1, shadow_limit);

                    regions.push(ShadowRegion::new(
                        region_id,
                        (x0, y0, x1 - x0, y1 - y0),
                        direction,
                        avg_shadow_intensity,
                        penumbra,
                        shadow_count,
                    ));
                    region_id += 1;
                }
            }
        }

        regions
    }

    /// Estimate local shadow direction from intensity gradients in a region.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::cast_precision_loss)]
    fn estimate_local_direction(
        &self,
        data: &[u8],
        width: u32,
        height: u32,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
    ) -> ShadowDirection {
        let mut gx_sum = 0.0f64;
        let mut gy_sum = 0.0f64;
        let mut count = 0u32;

        for y in (y0 + 1)..(y1.min(height - 1)) {
            for x in (x0 + 1)..(x1.min(width - 1)) {
                let _idx = (y * width + x) as usize;
                let left = (y * width + x - 1) as usize;
                let right = (y * width + x + 1) as usize;
                let up = ((y - 1) * width + x) as usize;
                let down = ((y + 1) * width + x) as usize;

                if down < data.len() && right < data.len() {
                    let gx = f64::from(data[right]) - f64::from(data[left]);
                    let gy = f64::from(data[down]) - f64::from(data[up]);
                    gx_sum += gx;
                    gy_sum += gy;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return ShadowDirection::new(0.0, 0.0, 0.0);
        }

        let avg_gx = gx_sum / f64::from(count);
        let avg_gy = gy_sum / f64::from(count);
        let mag = (avg_gx * avg_gx + avg_gy * avg_gy).sqrt();

        let confidence = (mag / 50.0).min(1.0);
        ShadowDirection::new(avg_gx, avg_gy, confidence)
    }

    /// Estimate penumbra width along the boundary of a shadow region.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::cast_precision_loss)]
    fn estimate_penumbra(
        &self,
        data: &[u8],
        width: u32,
        x0: u32,
        y0: u32,
        x1: u32,
        _y1: u32,
        shadow_limit: f64,
    ) -> f64 {
        // Sample along the top edge of the region to estimate transition width
        let y = y0;
        let mut transitions = Vec::new();

        for x in x0..x1 {
            let idx = (y * width + x) as usize;
            if idx < data.len() {
                let val = f64::from(data[idx]);
                // Find pixels near the shadow boundary
                if (val - shadow_limit).abs() < 30.0 {
                    transitions.push(val);
                }
            }
        }

        if transitions.len() < 2 {
            return 1.0;
        }

        // Estimate width from the range of transition values
        let min_t = transitions.iter().cloned().fold(f64::MAX, f64::min);
        let max_t = transitions.iter().cloned().fold(f64::MIN, f64::max);
        let range = max_t - min_t;

        // Convert intensity range to approximate pixel width
        (range / 10.0).max(1.0)
    }

    /// Estimate the primary light source from shadow regions.
    #[allow(clippy::cast_precision_loss)]
    fn estimate_light_source(&self, regions: &[ShadowRegion]) -> Option<LightSource> {
        if regions.is_empty() {
            return None;
        }

        // Average the shadow directions (weighted by confidence)
        let mut weighted_dx = 0.0;
        let mut weighted_dy = 0.0;
        let mut total_weight = 0.0;

        for r in regions {
            let w = r.direction.confidence * r.area as f64;
            weighted_dx += r.direction.dx * w;
            weighted_dy += r.direction.dy * w;
            total_weight += w;
        }

        if total_weight < 1e-10 {
            return None;
        }

        let avg_dx = weighted_dx / total_weight;
        let avg_dy = weighted_dy / total_weight;

        // Light is opposite to shadow direction
        let azimuth = (-avg_dy).atan2(-avg_dx);
        // Estimate elevation from shadow intensity (darker shadows = lower sun)
        let avg_intensity: f64 =
            regions.iter().map(|r| r.avg_intensity).sum::<f64>() / regions.len() as f64;
        let elevation = (1.0 - avg_intensity) * std::f64::consts::FRAC_PI_2;

        let confidence = (avg_dx * avg_dx + avg_dy * avg_dy).sqrt().min(1.0);

        Some(LightSource::new(azimuth, elevation, confidence))
    }

    /// Compute overall consistency score from regions and inconsistencies.
    #[allow(clippy::cast_precision_loss)]
    fn compute_consistency_score(
        &self,
        regions: &[ShadowRegion],
        inconsistencies: &[ShadowInconsistency],
    ) -> f64 {
        if regions.len() < 2 {
            return 1.0; // Cannot determine inconsistency with fewer than 2 regions
        }

        let num_pairs = regions.len() * (regions.len() - 1) / 2;
        if num_pairs == 0 {
            return 1.0;
        }

        let total_inconsistency: f64 = inconsistencies.iter().map(|i| i.confidence).sum();
        let score = 1.0 - (total_inconsistency / num_pairs as f64).min(1.0);
        score.max(0.0)
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &ShadowAnalysisConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shadow_direction_creation() {
        let dir = ShadowDirection::new(3.0, 4.0, 0.9);
        assert!((dir.dx * dir.dx + dir.dy * dir.dy - 1.0).abs() < 1e-10);
        assert!((dir.confidence - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_shadow_direction_zero() {
        let dir = ShadowDirection::new(0.0, 0.0, 0.5);
        assert!((dir.dx).abs() < f64::EPSILON);
        assert!((dir.dy).abs() < f64::EPSILON);
        assert!((dir.confidence).abs() < f64::EPSILON);
    }

    #[test]
    fn test_shadow_direction_angle() {
        let dir = ShadowDirection::new(1.0, 0.0, 1.0);
        assert!((dir.angle()).abs() < 1e-10);

        let dir_up = ShadowDirection::new(0.0, 1.0, 1.0);
        assert!((dir_up.angle() - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn test_shadow_direction_consistency() {
        let d1 = ShadowDirection::new(1.0, 0.0, 1.0);
        let d2 = ShadowDirection::new(1.0, 0.1, 1.0);
        assert!(d1.is_consistent(&d2, 0.2));

        let d3 = ShadowDirection::new(-1.0, 0.0, 1.0);
        assert!(!d1.is_consistent(&d3, 0.5));
    }

    #[test]
    fn test_shadow_direction_angular_difference() {
        let d1 = ShadowDirection::new(1.0, 0.0, 1.0);
        let d2 = ShadowDirection::new(0.0, 1.0, 1.0);
        let diff = d1.angular_difference(&d2);
        assert!((diff - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn test_shadow_region_center() {
        let region = ShadowRegion::new(
            0,
            (100, 200, 50, 60),
            ShadowDirection::new(1.0, 0.0, 0.8),
            0.3,
            2.0,
            500,
        );
        let (cx, cy) = region.center();
        assert!((cx - 125.0).abs() < f64::EPSILON);
        assert!((cy - 230.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_light_source_creation() {
        let ls = LightSource::new(1.0, 0.5, 0.8);
        assert!((ls.azimuth - 1.0).abs() < f64::EPSILON);
        assert!((ls.elevation - 0.5).abs() < f64::EPSILON);
        assert!((ls.confidence - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_light_source_distance() {
        let ls1 = LightSource::new(0.0, 0.5, 1.0);
        let ls2 = LightSource::new(0.0, 0.5, 1.0);
        assert!((ls1.angular_distance(&ls2)).abs() < 1e-10);
    }

    #[test]
    fn test_config_default() {
        let config = ShadowAnalysisConfig::default();
        assert!((config.shadow_threshold - 0.4).abs() < f64::EPSILON);
        assert_eq!(config.min_region_area, 100);
    }

    #[test]
    fn test_analyze_empty_regions() {
        let analyzer = ShadowAnalyzer::with_defaults();
        let report = analyzer.analyze_regions(&[]);
        assert!(report.inconsistencies.is_empty());
        assert!((report.consistency_score - 1.0).abs() < f64::EPSILON);
        assert!(!report.tampering_suspected);
    }

    #[test]
    fn test_analyze_consistent_regions() {
        let analyzer = ShadowAnalyzer::with_defaults();
        let regions = vec![
            ShadowRegion::new(
                0,
                (0, 0, 64, 64),
                ShadowDirection::new(1.0, 0.0, 0.9),
                0.3,
                2.0,
                200,
            ),
            ShadowRegion::new(
                1,
                (100, 0, 64, 64),
                ShadowDirection::new(1.0, 0.05, 0.9),
                0.3,
                2.5,
                200,
            ),
        ];
        let report = analyzer.analyze_regions(&regions);
        assert!(report.consistency_score > 0.5);
    }

    #[test]
    fn test_analyze_inconsistent_directions() {
        let analyzer = ShadowAnalyzer::new(ShadowAnalysisConfig {
            direction_threshold: 0.3,
            ..ShadowAnalysisConfig::default()
        });
        let regions = vec![
            ShadowRegion::new(
                0,
                (0, 0, 64, 64),
                ShadowDirection::new(1.0, 0.0, 0.9),
                0.3,
                2.0,
                200,
            ),
            ShadowRegion::new(
                1,
                (100, 0, 64, 64),
                ShadowDirection::new(-1.0, 0.0, 0.9),
                0.3,
                2.0,
                200,
            ),
        ];
        let report = analyzer.analyze_regions(&regions);
        let dir_mismatches: Vec<_> = report
            .inconsistencies
            .iter()
            .filter(|i| i.inconsistency_type == ShadowInconsistencyType::DirectionMismatch)
            .collect();
        assert!(!dir_mismatches.is_empty());
    }

    #[test]
    fn test_detect_shadows_empty() {
        let analyzer = ShadowAnalyzer::with_defaults();
        let regions = analyzer.detect_shadows(&[], 0, 0);
        assert!(regions.is_empty());
    }
}
