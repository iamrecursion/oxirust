//! Assisted rotoscoping tools.

use super::bezier::{BezierCurve, BezierPoint};
use crate::{Frame, VfxResult};

/// Edge detection for assisted rotoscoping.
#[derive(Debug, Clone)]
pub struct EdgeDetector {
    /// Edge detection threshold.
    pub threshold: f32,
    /// Blur amount before detection.
    pub blur: f32,
}

impl EdgeDetector {
    /// Create a new edge detector.
    #[must_use]
    pub const fn new(threshold: f32) -> Self {
        Self {
            threshold,
            blur: 1.0,
        }
    }

    /// Detect edges in frame.
    pub fn detect(&self, input: &Frame, output: &mut Frame) -> VfxResult<()> {
        // Sobel edge detection
        let sobel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
        let sobel_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

        for y in 1..(input.height - 1) {
            for x in 1..(input.width - 1) {
                let mut gx = 0.0;
                let mut gy = 0.0;

                for ky in 0_usize..3 {
                    for kx in 0_usize..3 {
                        let px = x + kx as u32 - 1;
                        let py = y + ky as u32 - 1;
                        let pixel = input.get_pixel(px, py).unwrap_or([0, 0, 0, 0]);
                        let gray = f32::from(pixel[0]) * 0.299
                            + f32::from(pixel[1]) * 0.587
                            + f32::from(pixel[2]) * 0.114;

                        gx += gray * sobel_x[ky][kx];
                        gy += gray * sobel_y[ky][kx];
                    }
                }

                let magnitude = (gx * gx + gy * gy).sqrt();
                let edge = if magnitude > self.threshold { 255 } else { 0 };
                output.set_pixel(x, y, [edge, edge, edge, 255]);
            }
        }

        Ok(())
    }

    /// Snap point to nearest edge.
    #[must_use]
    pub fn snap_to_edge(&self, edges: &Frame, x: f32, y: f32, search_radius: u32) -> (f32, f32) {
        let ix = x as i32;
        let iy = y as i32;
        let mut best_x = x;
        let mut best_y = y;
        let mut best_strength = 0.0;

        for dy in -(search_radius as i32)..=(search_radius as i32) {
            for dx in -(search_radius as i32)..=(search_radius as i32) {
                let px = (ix + dx).max(0).min((edges.width - 1) as i32) as u32;
                let py = (iy + dy).max(0).min((edges.height - 1) as i32) as u32;

                if let Some(pixel) = edges.get_pixel(px, py) {
                    let strength = f32::from(pixel[0]);
                    if strength > best_strength {
                        best_strength = strength;
                        best_x = px as f32;
                        best_y = py as f32;
                    }
                }
            }
        }

        (best_x, best_y)
    }
}

/// Auto-trace tool to automatically create mask from edges.
#[derive(Debug, Clone)]
pub struct AutoTrace {
    /// Tolerance for tracing.
    pub tolerance: f32,
    /// Minimum edge strength.
    pub min_edge_strength: f32,
}

impl AutoTrace {
    /// Create a new auto-trace tool.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            tolerance: 10.0,
            min_edge_strength: 50.0,
        }
    }

    /// Trace contour from seed point.
    pub fn trace(&self, edges: &Frame, seed_x: u32, seed_y: u32) -> VfxResult<BezierCurve> {
        let mut points = Vec::new();

        // Simplified contour tracing
        // In production would implement proper contour following algorithm
        let mut visited = vec![vec![false; edges.width as usize]; edges.height as usize];
        let mut stack = vec![(seed_x, seed_y)];

        while let Some((x, y)) = stack.pop() {
            if x >= edges.width || y >= edges.height || visited[y as usize][x as usize] {
                continue;
            }

            visited[y as usize][x as usize] = true;

            if let Some(pixel) = edges.get_pixel(x, y) {
                if f32::from(pixel[0]) >= self.min_edge_strength {
                    points.push(BezierPoint::new(x as f32, y as f32));

                    // Add neighbors
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let nx = (x as i32 + dx).max(0) as u32;
                            let ny = (y as i32 + dy).max(0) as u32;
                            if nx < edges.width && ny < edges.height {
                                stack.push((nx, ny));
                            }
                        }
                    }
                }
            }
        }

        // Simplify points
        let simplified = self.simplify_points(&points);
        Ok(BezierCurve::from_points(simplified))
    }

    fn simplify_points(&self, points: &[BezierPoint]) -> Vec<BezierPoint> {
        if points.len() < 3 {
            return points.to_vec();
        }

        // Douglas-Peucker simplification
        self.douglas_peucker(points, 0, points.len() - 1, self.tolerance)
    }

    fn douglas_peucker(
        &self,
        points: &[BezierPoint],
        start: usize,
        end: usize,
        tolerance: f32,
    ) -> Vec<BezierPoint> {
        if end <= start + 1 {
            return vec![points[start]];
        }

        let mut max_dist = 0.0;
        let mut max_idx = start;

        for i in (start + 1)..end {
            let dist = self.perpendicular_distance(
                points[i].x,
                points[i].y,
                points[start].x,
                points[start].y,
                points[end].x,
                points[end].y,
            );
            if dist > max_dist {
                max_dist = dist;
                max_idx = i;
            }
        }

        if max_dist > tolerance {
            let mut result1 = self.douglas_peucker(points, start, max_idx, tolerance);
            let result2 = self.douglas_peucker(points, max_idx, end, tolerance);
            result1.extend(result2);
            result1
        } else {
            vec![points[start], points[end]]
        }
    }

    fn perpendicular_distance(&self, px: f32, py: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
        let dx = x2 - x1;
        let dy = y2 - y1;
        let len_sq = dx * dx + dy * dy;

        if len_sq == 0.0 {
            return ((px - x1) * (px - x1) + (py - y1) * (py - y1)).sqrt();
        }

        let t = ((px - x1) * dx + (py - y1) * dy) / len_sq;
        let t = t.clamp(0.0, 1.0);

        let proj_x = x1 + t * dx;
        let proj_y = y1 + t * dy;

        ((px - proj_x) * (px - proj_x) + (py - proj_y) * (py - proj_y)).sqrt()
    }
}

impl Default for AutoTrace {
    fn default() -> Self {
        Self::new()
    }
}

/// Propagator for forward/backward mask propagation.
#[derive(Debug, Clone)]
pub struct Propagator {
    /// Propagation strength.
    pub strength: f32,
}

impl Propagator {
    /// Create a new propagator.
    #[must_use]
    pub const fn new() -> Self {
        Self { strength: 1.0 }
    }

    /// Propagate mask forward one frame using optical flow.
    pub fn propagate_forward(
        &self,
        _mask: &BezierCurve,
        _from: &Frame,
        _to: &Frame,
    ) -> VfxResult<BezierCurve> {
        // Simplified - would use optical flow in production
        Ok(_mask.clone())
    }

    /// Propagate mask backward one frame.
    pub fn propagate_backward(
        &self,
        _mask: &BezierCurve,
        _from: &Frame,
        _to: &Frame,
    ) -> VfxResult<BezierCurve> {
        // Simplified - would use optical flow in production
        Ok(_mask.clone())
    }
}

impl Default for Propagator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_detector() -> VfxResult<()> {
        let detector = EdgeDetector::new(50.0);
        let input = Frame::new(100, 100)?;
        let mut output = Frame::new(100, 100)?;
        detector.detect(&input, &mut output)?;
        Ok(())
    }

    #[test]
    fn test_auto_trace() {
        let tracer = AutoTrace::new();
        assert_eq!(tracer.tolerance, 10.0);
    }

    #[test]
    fn test_propagator() {
        let propagator = Propagator::new();
        assert_eq!(propagator.strength, 1.0);
    }
}
