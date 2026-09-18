//! Mask tracking for rotoscoping.

use super::point::{PointTracker, TrackPoint, TrackingConfig};
use crate::{Frame, VfxResult};
use serde::{Deserialize, Serialize};

/// A point in a mask with control handles.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MaskPoint {
    /// Point position.
    pub position: TrackPoint,
    /// In-tangent handle.
    pub tangent_in: (f32, f32),
    /// Out-tangent handle.
    pub tangent_out: (f32, f32),
}

impl MaskPoint {
    /// Create a new mask point.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self {
            position: TrackPoint {
                x,
                y,
                confidence: 1.0,
            },
            tangent_in: (0.0, 0.0),
            tangent_out: (0.0, 0.0),
        }
    }

    /// Create with tangents.
    #[must_use]
    pub const fn with_tangents(mut self, tangent_in: (f32, f32), tangent_out: (f32, f32)) -> Self {
        self.tangent_in = tangent_in;
        self.tangent_out = tangent_out;
        self
    }
}

/// Tracker for mask/rotoscope points.
#[derive(Debug, Clone)]
pub struct MaskTracker {
    point_trackers: Vec<PointTracker>,
    mask_points: Vec<MaskPoint>,
}

impl MaskTracker {
    /// Create a new mask tracker.
    #[must_use]
    pub fn new(config: TrackingConfig, points: Vec<MaskPoint>) -> Self {
        let point_trackers = vec![PointTracker::new(config); points.len()];
        Self {
            point_trackers,
            mask_points: points,
        }
    }

    /// Initialize tracking with reference frame.
    pub fn initialize(&mut self, frame: &Frame) -> VfxResult<()> {
        for (i, point) in self.mask_points.iter().enumerate() {
            self.point_trackers[i].initialize(frame, point.position)?;
        }
        Ok(())
    }

    /// Track all mask points in new frame.
    pub fn track(&self, frame: &Frame) -> VfxResult<Vec<MaskPoint>> {
        let mut tracked_points = Vec::with_capacity(self.mask_points.len());

        for (i, tracker) in self.point_trackers.iter().enumerate() {
            let result = tracker.track(frame)?;
            let original = &self.mask_points[i];

            // Calculate offset
            let offset_x = result.point.x - original.position.x;
            let offset_y = result.point.y - original.position.y;

            // Apply offset to tangents as well
            let new_point = MaskPoint {
                position: result.point,
                tangent_in: (
                    original.tangent_in.0 + offset_x,
                    original.tangent_in.1 + offset_y,
                ),
                tangent_out: (
                    original.tangent_out.0 + offset_x,
                    original.tangent_out.1 + offset_y,
                ),
            };

            tracked_points.push(new_point);
        }

        Ok(tracked_points)
    }

    /// Propagate mask tracking forward through frames.
    pub fn propagate_forward(&mut self, frames: &[Frame]) -> VfxResult<Vec<Vec<MaskPoint>>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_points = Vec::with_capacity(frames.len());

        // Initialize with first frame
        self.initialize(&frames[0])?;
        all_points.push(self.mask_points.clone());

        // Track through remaining frames
        for frame in &frames[1..] {
            let tracked = self.track(frame)?;

            // Update trackers with new positions for next frame
            for (i, point) in tracked.iter().enumerate() {
                self.point_trackers[i].update_template(frame, point.position)?;
            }

            self.mask_points = tracked.clone();
            all_points.push(tracked);
        }

        Ok(all_points)
    }

    /// Propagate mask tracking backward through frames.
    pub fn propagate_backward(&mut self, frames: &[Frame]) -> VfxResult<Vec<Vec<MaskPoint>>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_points = Vec::with_capacity(frames.len());

        // Initialize with last frame
        let last_idx = frames.len() - 1;
        self.initialize(&frames[last_idx])?;
        all_points.push(self.mask_points.clone());

        // Track backward through frames
        for frame in frames[..last_idx].iter().rev() {
            let tracked = self.track(frame)?;

            // Update trackers
            for (i, point) in tracked.iter().enumerate() {
                self.point_trackers[i].update_template(frame, point.position)?;
            }

            self.mask_points = tracked.clone();
            all_points.push(tracked);
        }

        // Reverse to get chronological order
        all_points.reverse();
        Ok(all_points)
    }

    /// Get number of points in mask.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.mask_points.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_point_creation() {
        let point = MaskPoint::new(10.0, 20.0);
        assert_eq!(point.position.x, 10.0);
        assert_eq!(point.position.y, 20.0);
    }

    #[test]
    fn test_mask_point_with_tangents() {
        let point = MaskPoint::new(10.0, 20.0).with_tangents((1.0, 2.0), (3.0, 4.0));
        assert_eq!(point.tangent_in, (1.0, 2.0));
        assert_eq!(point.tangent_out, (3.0, 4.0));
    }

    #[test]
    fn test_mask_tracker_creation() {
        let points = vec![
            MaskPoint::new(10.0, 10.0),
            MaskPoint::new(20.0, 10.0),
            MaskPoint::new(20.0, 20.0),
            MaskPoint::new(10.0, 20.0),
        ];
        let tracker = MaskTracker::new(TrackingConfig::default(), points);
        assert_eq!(tracker.point_count(), 4);
    }
}
