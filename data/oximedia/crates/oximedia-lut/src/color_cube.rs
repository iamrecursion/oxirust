#![allow(dead_code)]
//! Color cube sampling and operations.
//!
//! Provides tools for working with 3D color cubes, including
//! sampling, slicing, distance computations, and gamut boundary
//! analysis within the RGB unit cube.

use std::fmt;

/// A point within the RGB unit cube, each channel in `[0.0, 1.0]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CubePoint {
    /// Red channel.
    pub r: f64,
    /// Green channel.
    pub g: f64,
    /// Blue channel.
    pub b: f64,
}

impl CubePoint {
    /// Create a new cube point.
    #[must_use]
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    /// Create a cube point clamped to `[0.0, 1.0]`.
    #[must_use]
    pub fn clamped(r: f64, g: f64, b: f64) -> Self {
        Self {
            r: r.clamp(0.0, 1.0),
            g: g.clamp(0.0, 1.0),
            b: b.clamp(0.0, 1.0),
        }
    }

    /// Euclidean distance to another point.
    #[must_use]
    pub fn distance(&self, other: &Self) -> f64 {
        let dr = self.r - other.r;
        let dg = self.g - other.g;
        let db = self.b - other.b;
        (dr * dr + dg * dg + db * db).sqrt()
    }

    /// Check if this point is inside the unit cube (all channels in `[0, 1]`).
    #[must_use]
    pub fn is_in_gamut(&self) -> bool {
        self.r >= 0.0
            && self.r <= 1.0
            && self.g >= 0.0
            && self.g <= 1.0
            && self.b >= 0.0
            && self.b <= 1.0
    }

    /// Linearly interpolate between this point and another.
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
        }
    }

    /// Convert to an array `[r, g, b]`.
    #[must_use]
    pub fn to_array(self) -> [f64; 3] {
        [self.r, self.g, self.b]
    }

    /// Create from an array `[r, g, b]`.
    #[must_use]
    pub fn from_array(a: [f64; 3]) -> Self {
        Self {
            r: a[0],
            g: a[1],
            b: a[2],
        }
    }

    /// Compute the luminance using Rec.709 coefficients.
    #[must_use]
    pub fn luminance(&self) -> f64 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }
}

impl fmt::Display for CubePoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4}, {:.4})", self.r, self.g, self.b)
    }
}

/// A slice through a color cube at a fixed channel value.
#[derive(Clone, Debug)]
pub enum CubeSliceAxis {
    /// Slice at a fixed red value.
    Red(f64),
    /// Slice at a fixed green value.
    Green(f64),
    /// Slice at a fixed blue value.
    Blue(f64),
}

/// A 2D grid of samples from a cube slice.
#[derive(Clone, Debug)]
pub struct CubeSlice {
    /// The axis and value at which the slice was taken.
    pub axis: CubeSliceAxis,
    /// Number of samples along each dimension.
    pub resolution: usize,
    /// Sampled color values (row-major, resolution x resolution).
    pub samples: Vec<CubePoint>,
}

impl CubeSlice {
    /// Create a slice through the cube at a fixed axis value.
    #[must_use]
    pub fn sample(axis: CubeSliceAxis, resolution: usize) -> Self {
        let mut samples = Vec::with_capacity(resolution * resolution);
        let step = if resolution > 1 {
            1.0 / (resolution as f64 - 1.0)
        } else {
            0.0
        };

        for row in 0..resolution {
            for col in 0..resolution {
                let u = col as f64 * step;
                let v = row as f64 * step;
                let point = match &axis {
                    CubeSliceAxis::Red(val) => CubePoint::new(*val, u, v),
                    CubeSliceAxis::Green(val) => CubePoint::new(u, *val, v),
                    CubeSliceAxis::Blue(val) => CubePoint::new(u, v, *val),
                };
                samples.push(point);
            }
        }

        Self {
            axis,
            resolution,
            samples,
        }
    }

    /// Get a sample at grid coordinates.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Option<&CubePoint> {
        if row < self.resolution && col < self.resolution {
            Some(&self.samples[row * self.resolution + col])
        } else {
            None
        }
    }

    /// Total number of samples.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

/// Uniform sampling of the 3D color cube.
#[derive(Clone, Debug)]
pub struct CubeSampler {
    /// Number of samples per axis.
    pub resolution: usize,
}

impl CubeSampler {
    /// Create a sampler with the given resolution per axis.
    #[must_use]
    pub fn new(resolution: usize) -> Self {
        Self {
            resolution: resolution.max(2),
        }
    }

    /// Generate all sample points.
    #[must_use]
    pub fn generate(&self) -> Vec<CubePoint> {
        let n = self.resolution;
        let step = 1.0 / (n as f64 - 1.0);
        let mut points = Vec::with_capacity(n * n * n);
        for b in 0..n {
            for g in 0..n {
                for r in 0..n {
                    points.push(CubePoint::new(
                        r as f64 * step,
                        g as f64 * step,
                        b as f64 * step,
                    ));
                }
            }
        }
        points
    }

    /// Total number of sample points.
    #[must_use]
    pub fn total_samples(&self) -> usize {
        self.resolution * self.resolution * self.resolution
    }
}

/// Compute the bounding box of a set of cube points.
#[must_use]
pub fn bounding_box(points: &[CubePoint]) -> Option<(CubePoint, CubePoint)> {
    if points.is_empty() {
        return None;
    }
    let mut min_r = f64::MAX;
    let mut min_g = f64::MAX;
    let mut min_b = f64::MAX;
    let mut max_r = f64::MIN;
    let mut max_g = f64::MIN;
    let mut max_b = f64::MIN;

    for p in points {
        min_r = min_r.min(p.r);
        min_g = min_g.min(p.g);
        min_b = min_b.min(p.b);
        max_r = max_r.max(p.r);
        max_g = max_g.max(p.g);
        max_b = max_b.max(p.b);
    }

    Some((
        CubePoint::new(min_r, min_g, min_b),
        CubePoint::new(max_r, max_g, max_b),
    ))
}

/// Compute the centroid (average) of a set of cube points.
#[must_use]
pub fn centroid(points: &[CubePoint]) -> Option<CubePoint> {
    if points.is_empty() {
        return None;
    }
    let n = points.len() as f64;
    let sum_r: f64 = points.iter().map(|p| p.r).sum();
    let sum_g: f64 = points.iter().map(|p| p.g).sum();
    let sum_b: f64 = points.iter().map(|p| p.b).sum();
    Some(CubePoint::new(sum_r / n, sum_g / n, sum_b / n))
}

/// Count how many points are outside the unit cube gamut.
#[must_use]
pub fn count_out_of_gamut(points: &[CubePoint]) -> usize {
    points.iter().filter(|p| !p.is_in_gamut()).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cube_point_new() {
        let p = CubePoint::new(0.5, 0.3, 0.7);
        assert!((p.r - 0.5).abs() < f64::EPSILON);
        assert!((p.g - 0.3).abs() < f64::EPSILON);
        assert!((p.b - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cube_point_clamped() {
        let p = CubePoint::clamped(-0.1, 1.5, 0.5);
        assert!((p.r - 0.0).abs() < f64::EPSILON);
        assert!((p.g - 1.0).abs() < f64::EPSILON);
        assert!((p.b - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cube_point_distance() {
        let a = CubePoint::new(0.0, 0.0, 0.0);
        let b = CubePoint::new(1.0, 0.0, 0.0);
        assert!((a.distance(&b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cube_point_is_in_gamut() {
        assert!(CubePoint::new(0.0, 0.5, 1.0).is_in_gamut());
        assert!(!CubePoint::new(-0.01, 0.5, 1.0).is_in_gamut());
        assert!(!CubePoint::new(0.0, 0.5, 1.01).is_in_gamut());
    }

    #[test]
    fn test_cube_point_lerp() {
        let a = CubePoint::new(0.0, 0.0, 0.0);
        let b = CubePoint::new(1.0, 1.0, 1.0);
        let mid = a.lerp(&b, 0.5);
        assert!((mid.r - 0.5).abs() < 1e-10);
        assert!((mid.g - 0.5).abs() < 1e-10);
        assert!((mid.b - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_cube_point_to_from_array() {
        let p = CubePoint::new(0.1, 0.2, 0.3);
        let arr = p.to_array();
        let p2 = CubePoint::from_array(arr);
        assert!((p.r - p2.r).abs() < f64::EPSILON);
        assert!((p.g - p2.g).abs() < f64::EPSILON);
        assert!((p.b - p2.b).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cube_point_luminance() {
        let white = CubePoint::new(1.0, 1.0, 1.0);
        let lum = white.luminance();
        assert!((lum - 1.0).abs() < 1e-10);

        let black = CubePoint::new(0.0, 0.0, 0.0);
        assert!(black.luminance().abs() < 1e-10);
    }

    #[test]
    fn test_cube_point_display() {
        let p = CubePoint::new(0.0, 0.5, 1.0);
        let s = format!("{p}");
        assert!(s.contains("0.0000"));
        assert!(s.contains("0.5000"));
        assert!(s.contains("1.0000"));
    }

    #[test]
    fn test_cube_slice_sample_count() {
        let slice = CubeSlice::sample(CubeSliceAxis::Red(0.5), 10);
        assert_eq!(slice.sample_count(), 100);
    }

    #[test]
    fn test_cube_slice_get() {
        let slice = CubeSlice::sample(CubeSliceAxis::Blue(0.0), 5);
        let p = slice.get(0, 0);
        assert!(p.is_some());
        assert!(slice.get(5, 0).is_none());
    }

    #[test]
    fn test_cube_sampler_total() {
        let sampler = CubeSampler::new(5);
        assert_eq!(sampler.total_samples(), 125);
        let points = sampler.generate();
        assert_eq!(points.len(), 125);
    }

    #[test]
    fn test_bounding_box() {
        let points = vec![
            CubePoint::new(0.1, 0.2, 0.3),
            CubePoint::new(0.9, 0.8, 0.7),
            CubePoint::new(0.5, 0.5, 0.5),
        ];
        let (min_p, max_p) = bounding_box(&points).expect("should succeed in test");
        assert!((min_p.r - 0.1).abs() < 1e-10);
        assert!((max_p.r - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_bounding_box_empty() {
        assert!(bounding_box(&[]).is_none());
    }

    #[test]
    fn test_centroid() {
        let points = vec![CubePoint::new(0.0, 0.0, 0.0), CubePoint::new(1.0, 1.0, 1.0)];
        let c = centroid(&points).expect("should succeed in test");
        assert!((c.r - 0.5).abs() < 1e-10);
        assert!((c.g - 0.5).abs() < 1e-10);
        assert!((c.b - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_centroid_empty() {
        assert!(centroid(&[]).is_none());
    }

    #[test]
    fn test_count_out_of_gamut() {
        let points = vec![
            CubePoint::new(0.5, 0.5, 0.5),
            CubePoint::new(-0.1, 0.5, 0.5),
            CubePoint::new(0.5, 0.5, 1.1),
        ];
        assert_eq!(count_out_of_gamut(&points), 2);
    }

    #[test]
    fn test_sampler_min_resolution() {
        let sampler = CubeSampler::new(0);
        assert_eq!(sampler.resolution, 2);
    }
}
