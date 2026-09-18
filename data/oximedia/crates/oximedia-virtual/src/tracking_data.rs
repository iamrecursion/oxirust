#![allow(dead_code)]
//! Tracking data structures for virtual production marker systems.

/// A single tracked point in 2-D image space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackingPoint {
    /// Horizontal position in pixels.
    pub x: f64,
    /// Vertical position in pixels.
    pub y: f64,
    /// Tracking confidence in [0, 1].
    pub confidence: f64,
}

impl TrackingPoint {
    /// Create a new tracking point.
    #[must_use]
    pub fn new(x: f64, y: f64, confidence: f64) -> Self {
        Self { x, y, confidence }
    }

    /// Euclidean distance to another tracking point.
    #[must_use]
    pub fn distance_to(&self, other: &TrackingPoint) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Return `true` if the point is considered reliable.
    #[must_use]
    pub fn is_reliable(&self) -> bool {
        self.confidence >= 0.5
    }
}

/// A collection of tracking points captured in one frame.
#[derive(Debug, Clone)]
pub struct TrackingFrame {
    /// Frame index (zero-based).
    pub frame_index: u64,
    /// Points tracked in this frame.
    pub points: Vec<TrackingPoint>,
}

impl TrackingFrame {
    /// Create a new empty tracking frame.
    #[must_use]
    pub fn new(frame_index: u64) -> Self {
        Self {
            frame_index,
            points: Vec::new(),
        }
    }

    /// Add a tracking point to this frame.
    pub fn add_point(&mut self, point: TrackingPoint) {
        self.points.push(point);
    }

    /// Return the centroid (mean position) of all points.
    /// Returns `None` if there are no points.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn centroid(&self) -> Option<(f64, f64)> {
        if self.points.is_empty() {
            return None;
        }
        let n = self.points.len() as f64;
        let sum_x: f64 = self.points.iter().map(|p| p.x).sum();
        let sum_y: f64 = self.points.iter().map(|p| p.y).sum();
        Some((sum_x / n, sum_y / n))
    }

    /// Return the number of points in this frame.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Return the average confidence across all points.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn avg_confidence(&self) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.points.iter().map(|p| p.confidence).sum();
        sum / self.points.len() as f64
    }
}

/// Complete tracking data for a sequence of frames.
#[derive(Debug, Default)]
pub struct TrackingData {
    frames: Vec<TrackingFrame>,
}

impl TrackingData {
    /// Create a new, empty tracking data container.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a frame to the tracking data.
    pub fn add_frame(&mut self, frame: TrackingFrame) {
        self.frames.push(frame);
    }

    /// Return the total number of frames.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Return the average number of tracked points per frame.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn avg_track_count(&self) -> f64 {
        if self.frames.is_empty() {
            return 0.0;
        }
        let total: usize = self.frames.iter().map(TrackingFrame::point_count).sum();
        total as f64 / self.frames.len() as f64
    }

    /// Access all frames.
    #[must_use]
    pub fn frames(&self) -> &[TrackingFrame] {
        &self.frames
    }

    /// Access a specific frame by index.
    #[must_use]
    pub fn frame(&self, index: usize) -> Option<&TrackingFrame> {
        self.frames.get(index)
    }

    /// Return the frame with the most tracked points.
    #[must_use]
    pub fn best_frame(&self) -> Option<&TrackingFrame> {
        self.frames.iter().max_by_key(|f| f.point_count())
    }

    /// Compute the overall average confidence across all frames and points.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn overall_avg_confidence(&self) -> f64 {
        let all_points: Vec<&TrackingPoint> = self.frames.iter().flat_map(|f| &f.points).collect();
        if all_points.is_empty() {
            return 0.0;
        }
        let sum: f64 = all_points.iter().map(|p| p.confidence).sum();
        sum / all_points.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracking_point_distance_zero() {
        let p = TrackingPoint::new(5.0, 5.0, 1.0);
        assert!((p.distance_to(&p) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_tracking_point_distance_3_4_5() {
        let a = TrackingPoint::new(0.0, 0.0, 1.0);
        let b = TrackingPoint::new(3.0, 4.0, 1.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_tracking_point_distance_symmetric() {
        let a = TrackingPoint::new(1.0, 2.0, 0.9);
        let b = TrackingPoint::new(4.0, 6.0, 0.8);
        assert!((a.distance_to(&b) - b.distance_to(&a)).abs() < 1e-10);
    }

    #[test]
    fn test_tracking_point_reliable() {
        let p = TrackingPoint::new(0.0, 0.0, 0.8);
        assert!(p.is_reliable());
    }

    #[test]
    fn test_tracking_point_unreliable() {
        let p = TrackingPoint::new(0.0, 0.0, 0.3);
        assert!(!p.is_reliable());
    }

    #[test]
    fn test_tracking_frame_centroid_empty() {
        let frame = TrackingFrame::new(0);
        assert!(frame.centroid().is_none());
    }

    #[test]
    fn test_tracking_frame_centroid_single() {
        let mut frame = TrackingFrame::new(0);
        frame.add_point(TrackingPoint::new(10.0, 20.0, 1.0));
        let (cx, cy) = frame.centroid().expect("should succeed in test");
        assert!((cx - 10.0).abs() < 1e-10);
        assert!((cy - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_tracking_frame_centroid_two_points() {
        let mut frame = TrackingFrame::new(0);
        frame.add_point(TrackingPoint::new(0.0, 0.0, 1.0));
        frame.add_point(TrackingPoint::new(10.0, 10.0, 1.0));
        let (cx, cy) = frame.centroid().expect("should succeed in test");
        assert!((cx - 5.0).abs() < 1e-10);
        assert!((cy - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_tracking_frame_avg_confidence() {
        let mut frame = TrackingFrame::new(0);
        frame.add_point(TrackingPoint::new(0.0, 0.0, 0.8));
        frame.add_point(TrackingPoint::new(1.0, 1.0, 0.6));
        let avg = frame.avg_confidence();
        assert!((avg - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_tracking_data_empty() {
        let data = TrackingData::new();
        assert_eq!(data.frame_count(), 0);
        assert!((data.avg_track_count() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_tracking_data_add_frame() {
        let mut data = TrackingData::new();
        data.add_frame(TrackingFrame::new(0));
        data.add_frame(TrackingFrame::new(1));
        assert_eq!(data.frame_count(), 2);
    }

    #[test]
    fn test_tracking_data_avg_track_count() {
        let mut data = TrackingData::new();
        let mut f0 = TrackingFrame::new(0);
        f0.add_point(TrackingPoint::new(1.0, 2.0, 1.0));
        f0.add_point(TrackingPoint::new(3.0, 4.0, 1.0));
        let mut f1 = TrackingFrame::new(1);
        f1.add_point(TrackingPoint::new(5.0, 6.0, 1.0));
        data.add_frame(f0);
        data.add_frame(f1);
        // (2 + 1) / 2 = 1.5
        assert!((data.avg_track_count() - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_tracking_data_best_frame() {
        let mut data = TrackingData::new();
        let mut f0 = TrackingFrame::new(0);
        f0.add_point(TrackingPoint::new(0.0, 0.0, 1.0));
        let mut f1 = TrackingFrame::new(1);
        f1.add_point(TrackingPoint::new(1.0, 1.0, 1.0));
        f1.add_point(TrackingPoint::new(2.0, 2.0, 1.0));
        data.add_frame(f0);
        data.add_frame(f1);
        let best = data.best_frame().expect("should succeed in test");
        assert_eq!(best.frame_index, 1);
    }

    #[test]
    fn test_tracking_data_overall_avg_confidence() {
        let mut data = TrackingData::new();
        let mut f = TrackingFrame::new(0);
        f.add_point(TrackingPoint::new(0.0, 0.0, 1.0));
        f.add_point(TrackingPoint::new(1.0, 1.0, 0.0));
        data.add_frame(f);
        assert!((data.overall_avg_confidence() - 0.5).abs() < 1e-10);
    }
}
