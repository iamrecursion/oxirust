//! Visual attention and saliency analysis.
//!
//! This module provides types and logic for computing visual saliency,
//! weighted attention centers, and focus metrics for video frames.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// A weighted point of visual interest, with normalized coordinates (0.0–1.0).
#[derive(Debug, Clone)]
pub struct SaliencyPoint {
    /// Horizontal position (0.0 = left, 1.0 = right).
    pub x: f32,
    /// Vertical position (0.0 = top, 1.0 = bottom).
    pub y: f32,
    /// Attention weight (higher = more salient).
    pub weight: f32,
}

impl SaliencyPoint {
    /// Create a new saliency point.
    pub fn new(x: f32, y: f32, weight: f32) -> Self {
        Self { x, y, weight }
    }

    /// Returns true if the point lies within the broadcast-safe area
    /// (x: 0.1–0.9, y: 0.1–0.9).
    pub fn is_in_safe_area(&self) -> bool {
        self.x >= 0.1 && self.x <= 0.9 && self.y >= 0.1 && self.y <= 0.9
    }
}

/// A saliency map composed of weighted interest points for a single frame.
#[derive(Debug, Clone)]
pub struct SaliencyMap {
    /// All saliency points in this map.
    pub points: Vec<SaliencyPoint>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

impl SaliencyMap {
    /// Create an empty saliency map.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            points: Vec::new(),
            width,
            height,
        }
    }

    /// Add a saliency point.
    pub fn add(&mut self, point: SaliencyPoint) {
        self.points.push(point);
    }

    /// Returns the weighted center of attention as (x, y).
    ///
    /// Returns (0.5, 0.5) if there are no points or total weight is zero.
    pub fn weighted_center(&self) -> (f32, f32) {
        let total_weight: f32 = self.points.iter().map(|p| p.weight).sum();
        if total_weight == 0.0 || self.points.is_empty() {
            return (0.5, 0.5);
        }
        let cx = self.points.iter().map(|p| p.x * p.weight).sum::<f32>() / total_weight;
        let cy = self.points.iter().map(|p| p.y * p.weight).sum::<f32>() / total_weight;
        (cx, cy)
    }

    /// Returns the top-N most salient points (sorted descending by weight).
    pub fn top_n(&self, n: usize) -> Vec<&SaliencyPoint> {
        let mut sorted: Vec<&SaliencyPoint> = self.points.iter().collect();
        sorted.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(n);
        sorted
    }

    /// Returns all points that fall within the region of interest [x0, x1] x [y0, y1].
    pub fn in_roi(&self, x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<&SaliencyPoint> {
        self.points
            .iter()
            .filter(|p| p.x >= x0 && p.x <= x1 && p.y >= y0 && p.y <= y1)
            .collect()
    }
}

/// Aggregate focus metric derived from a saliency map.
#[derive(Debug, Clone)]
pub struct FocusMetric {
    /// X coordinate of the attention center (0.0–1.0).
    pub center_x: f32,
    /// Y coordinate of the attention center (0.0–1.0).
    pub center_y: f32,
    /// Spread of the attention (0.0 = fully concentrated, 1.0 = fully scattered).
    pub spread: f32,
}

impl FocusMetric {
    /// Create a `FocusMetric` from a `SaliencyMap`.
    pub fn from_saliency_map(map: &SaliencyMap) -> Self {
        let (cx, cy) = map.weighted_center();
        let spread = if map.points.is_empty() {
            0.0
        } else {
            let total_weight: f32 = map.points.iter().map(|p| p.weight).sum();
            if total_weight == 0.0 {
                0.0
            } else {
                let variance_x = map
                    .points
                    .iter()
                    .map(|p| p.weight * (p.x - cx).powi(2))
                    .sum::<f32>()
                    / total_weight;
                let variance_y = map
                    .points
                    .iter()
                    .map(|p| p.weight * (p.y - cy).powi(2))
                    .sum::<f32>()
                    / total_weight;
                (variance_x + variance_y).sqrt()
            }
        };
        Self {
            center_x: cx,
            center_y: cy,
            spread,
        }
    }

    /// Returns true if the attention center is near the middle of the frame
    /// (within 0.15 of the center in both axes).
    pub fn is_centered(&self) -> bool {
        (self.center_x - 0.5).abs() <= 0.15 && (self.center_y - 0.5).abs() <= 0.15
    }

    /// Returns true if the spread exceeds the given threshold.
    pub fn is_scattered(&self, threshold: f32) -> bool {
        self.spread > threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_saliency_point_is_in_safe_area_yes() {
        let p = SaliencyPoint::new(0.5, 0.5, 1.0);
        assert!(p.is_in_safe_area());
    }

    #[test]
    fn test_saliency_point_is_in_safe_area_edges() {
        let p_left = SaliencyPoint::new(0.1, 0.5, 1.0);
        let p_right = SaliencyPoint::new(0.9, 0.5, 1.0);
        let p_top = SaliencyPoint::new(0.5, 0.1, 1.0);
        let p_bottom = SaliencyPoint::new(0.5, 0.9, 1.0);
        assert!(p_left.is_in_safe_area());
        assert!(p_right.is_in_safe_area());
        assert!(p_top.is_in_safe_area());
        assert!(p_bottom.is_in_safe_area());
    }

    #[test]
    fn test_saliency_point_is_not_in_safe_area() {
        let p_left = SaliencyPoint::new(0.05, 0.5, 1.0);
        let p_right = SaliencyPoint::new(0.95, 0.5, 1.0);
        let p_top = SaliencyPoint::new(0.5, 0.05, 1.0);
        let p_bottom = SaliencyPoint::new(0.5, 0.95, 1.0);
        assert!(!p_left.is_in_safe_area());
        assert!(!p_right.is_in_safe_area());
        assert!(!p_top.is_in_safe_area());
        assert!(!p_bottom.is_in_safe_area());
    }

    #[test]
    fn test_saliency_map_empty_weighted_center() {
        let map = SaliencyMap::new(1920, 1080);
        let (cx, cy) = map.weighted_center();
        assert!((cx - 0.5).abs() < 1e-5);
        assert!((cy - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_saliency_map_weighted_center_single() {
        let mut map = SaliencyMap::new(1920, 1080);
        map.add(SaliencyPoint::new(0.2, 0.8, 1.0));
        let (cx, cy) = map.weighted_center();
        assert!((cx - 0.2).abs() < 1e-5);
        assert!((cy - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_saliency_map_weighted_center_multiple() {
        let mut map = SaliencyMap::new(1920, 1080);
        map.add(SaliencyPoint::new(0.0, 0.0, 1.0));
        map.add(SaliencyPoint::new(1.0, 1.0, 1.0));
        let (cx, cy) = map.weighted_center();
        assert!((cx - 0.5).abs() < 1e-5);
        assert!((cy - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_saliency_map_top_n() {
        let mut map = SaliencyMap::new(1920, 1080);
        map.add(SaliencyPoint::new(0.1, 0.1, 0.3));
        map.add(SaliencyPoint::new(0.5, 0.5, 0.9));
        map.add(SaliencyPoint::new(0.8, 0.8, 0.6));
        map.add(SaliencyPoint::new(0.3, 0.7, 0.1));

        let top2 = map.top_n(2);
        assert_eq!(top2.len(), 2);
        assert!((top2[0].weight - 0.9).abs() < 1e-5);
        assert!((top2[1].weight - 0.6).abs() < 1e-5);
    }

    #[test]
    fn test_saliency_map_top_n_more_than_available() {
        let mut map = SaliencyMap::new(1280, 720);
        map.add(SaliencyPoint::new(0.5, 0.5, 1.0));
        let top10 = map.top_n(10);
        assert_eq!(top10.len(), 1);
    }

    #[test]
    fn test_saliency_map_in_roi() {
        let mut map = SaliencyMap::new(1920, 1080);
        map.add(SaliencyPoint::new(0.1, 0.1, 1.0));
        map.add(SaliencyPoint::new(0.5, 0.5, 1.0));
        map.add(SaliencyPoint::new(0.9, 0.9, 1.0));

        let roi = map.in_roi(0.0, 0.0, 0.6, 0.6);
        assert_eq!(roi.len(), 2);
    }

    #[test]
    fn test_focus_metric_from_empty_map() {
        let map = SaliencyMap::new(1920, 1080);
        let fm = FocusMetric::from_saliency_map(&map);
        assert!((fm.center_x - 0.5).abs() < 1e-5);
        assert!((fm.center_y - 0.5).abs() < 1e-5);
        assert!((fm.spread).abs() < 1e-5);
    }

    #[test]
    fn test_focus_metric_is_centered() {
        let fm = FocusMetric {
            center_x: 0.5,
            center_y: 0.5,
            spread: 0.1,
        };
        assert!(fm.is_centered());
    }

    #[test]
    fn test_focus_metric_is_not_centered() {
        let fm = FocusMetric {
            center_x: 0.0,
            center_y: 0.0,
            spread: 0.1,
        };
        assert!(!fm.is_centered());
    }

    #[test]
    fn test_focus_metric_is_scattered() {
        let fm = FocusMetric {
            center_x: 0.5,
            center_y: 0.5,
            spread: 0.5,
        };
        assert!(fm.is_scattered(0.3));
        assert!(!fm.is_scattered(0.6));
    }
}
