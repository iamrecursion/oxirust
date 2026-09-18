//! In-Camera VFX frustum tracking with motion smoothing and latency compensation.
//!
//! Provides real-time camera frustum data management, EWMA smoothing,
//! and frame-delay latency compensation for in-camera visual effects workflows.

#![allow(dead_code)]

/// Lens data captured from a follow-focus or lens encoder system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LensData {
    /// Focal length in millimeters.
    pub focal_length_mm: f32,
    /// Focus distance in meters.
    pub focus_distance_m: f32,
    /// Iris (T-stop or f-stop value).
    pub iris: f32,
    /// Zoom percentage (0..=100).
    pub zoom_pct: f32,
}

impl LensData {
    /// Creates a new `LensData`.
    #[must_use]
    pub const fn new(
        focal_length_mm: f32,
        focus_distance_m: f32,
        iris: f32,
        zoom_pct: f32,
    ) -> Self {
        Self {
            focal_length_mm,
            focus_distance_m,
            iris,
            zoom_pct,
        }
    }
}

impl Default for LensData {
    fn default() -> Self {
        Self {
            focal_length_mm: 35.0,
            focus_distance_m: 3.0,
            iris: 2.8,
            zoom_pct: 0.0,
        }
    }
}

/// Camera frustum data for a single frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrustumData {
    /// Camera world-space position (x, y, z) in meters.
    pub camera_pos: (f32, f32, f32),
    /// Camera rotation (pitch, yaw, roll) in degrees.
    pub camera_rot: (f32, f32, f32),
    /// Horizontal field-of-view in degrees.
    pub fov_h: f32,
    /// Vertical field-of-view in degrees.
    pub fov_v: f32,
    /// Near clip plane distance in meters.
    pub near: f32,
    /// Far clip plane distance in meters.
    pub far: f32,
}

impl FrustumData {
    /// Creates a new `FrustumData`.
    #[must_use]
    pub const fn new(
        camera_pos: (f32, f32, f32),
        camera_rot: (f32, f32, f32),
        fov_h: f32,
        fov_v: f32,
        near: f32,
        far: f32,
    ) -> Self {
        Self {
            camera_pos,
            camera_rot,
            fov_h,
            fov_v,
            near,
            far,
        }
    }
}

impl Default for FrustumData {
    fn default() -> Self {
        Self {
            camera_pos: (0.0, 0.0, 0.0),
            camera_rot: (0.0, 0.0, 0.0),
            fov_h: 60.0,
            fov_v: 33.75,
            near: 0.1,
            far: 1000.0,
        }
    }
}

/// A motion control data packet combining frustum and lens metadata.
#[derive(Debug, Clone)]
pub struct MoCoDataPacket {
    /// Wall-clock timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// Camera frustum data for this frame.
    pub frustum: FrustumData,
    /// Lens data for this frame.
    pub lens_data: LensData,
}

impl MoCoDataPacket {
    /// Creates a new motion control data packet.
    #[must_use]
    pub const fn new(timestamp_ms: u64, frustum: FrustumData, lens_data: LensData) -> Self {
        Self {
            timestamp_ms,
            frustum,
            lens_data,
        }
    }
}

/// Ring-buffer frustum tracker holding the last N frames of frustum data.
///
/// Capacity is fixed at 60 frames (one second at 60fps).
pub struct FrustumTracker {
    /// Ring buffer storing historical frustum data.
    buffer: Vec<FrustumData>,
    /// Current write position in the ring buffer.
    head: usize,
    /// Number of valid entries stored so far.
    count: usize,
    /// Fixed capacity.
    capacity: usize,
}

impl FrustumTracker {
    /// Ring buffer capacity in frames.
    pub const CAPACITY: usize = 60;

    /// Creates a new `FrustumTracker`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: vec![FrustumData::default(); Self::CAPACITY],
            head: 0,
            count: 0,
            capacity: Self::CAPACITY,
        }
    }

    /// Updates the tracker with a new frustum measurement.
    pub fn update(&mut self, data: FrustumData) {
        self.buffer[self.head] = data;
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    /// Returns a slice view of historical data (most recent first).
    ///
    /// The returned slice contains up to `CAPACITY` entries.
    #[must_use]
    pub fn history(&self) -> &[FrustumData] {
        &self.buffer[..self.count]
    }

    /// Returns the number of valid entries in the history buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns true if no data has been recorded yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns the most recent frustum data, or `None` if empty.
    #[must_use]
    pub fn latest(&self) -> Option<FrustumData> {
        if self.count == 0 {
            return None;
        }
        let latest_idx = if self.head == 0 {
            self.capacity - 1
        } else {
            self.head - 1
        };
        Some(self.buffer[latest_idx])
    }
}

impl Default for FrustumTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Exponential Weighted Moving Average (EWMA) smoother for frustum data.
///
/// Applied independently to each scalar field of `FrustumData`.
pub struct FrustumSmoother;

impl FrustumSmoother {
    /// Smooths a history of frustum data using EWMA.
    ///
    /// `alpha` controls the smoothing weight (0.0 = maximum smoothing, 1.0 = no smoothing).
    /// Returns the EWMA-smoothed value computed over the history slice.
    /// Uses the oldest sample as the initial value for the EMA, then applies forward EWMA.
    #[must_use]
    pub fn smooth(history: &[FrustumData], alpha: f32) -> FrustumData {
        if history.is_empty() {
            return FrustumData::default();
        }

        let alpha = alpha.clamp(0.0, 1.0);
        let one_minus_alpha = 1.0 - alpha;

        let mut acc = history[0];

        for sample in history.iter().skip(1) {
            acc.camera_pos = (
                one_minus_alpha * acc.camera_pos.0 + alpha * sample.camera_pos.0,
                one_minus_alpha * acc.camera_pos.1 + alpha * sample.camera_pos.1,
                one_minus_alpha * acc.camera_pos.2 + alpha * sample.camera_pos.2,
            );
            acc.camera_rot = (
                one_minus_alpha * acc.camera_rot.0 + alpha * sample.camera_rot.0,
                one_minus_alpha * acc.camera_rot.1 + alpha * sample.camera_rot.1,
                one_minus_alpha * acc.camera_rot.2 + alpha * sample.camera_rot.2,
            );
            acc.fov_h = one_minus_alpha * acc.fov_h + alpha * sample.fov_h;
            acc.fov_v = one_minus_alpha * acc.fov_v + alpha * sample.fov_v;
            acc.near = one_minus_alpha * acc.near + alpha * sample.near;
            acc.far = one_minus_alpha * acc.far + alpha * sample.far;
        }

        acc
    }
}

/// Latency compensator that shifts frustum data back by N frames.
///
/// This compensates for pipeline delays (e.g., render engine latency)
/// by returning the frustum data from `frame_delay` frames ago.
pub struct IcvfxLatencyCompensator {
    /// Number of frames to look back in history.
    pub frame_delay: u32,
}

impl IcvfxLatencyCompensator {
    /// Creates a new latency compensator with the given frame delay.
    #[must_use]
    pub const fn new(frame_delay: u32) -> Self {
        Self { frame_delay }
    }

    /// Returns the compensated frustum data from `frame_delay` frames ago.
    ///
    /// If the history is shorter than `frame_delay`, returns the oldest available entry.
    /// Falls back to `current` if history is empty.
    #[must_use]
    pub fn compensate(&self, current: FrustumData, history: &[FrustumData]) -> FrustumData {
        if history.is_empty() {
            return current;
        }

        let delay = self.frame_delay as usize;
        let n = history.len();

        if delay == 0 || n == 0 {
            return current;
        }

        // history[n-1] is most recent, history[0] is oldest
        // We want the entry `delay` frames back from the latest
        if delay >= n {
            history[0]
        } else {
            history[n - 1 - delay]
        }
    }
}

impl Default for IcvfxLatencyCompensator {
    fn default() -> Self {
        Self::new(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lens_data_default() {
        let ld = LensData::default();
        assert_eq!(ld.focal_length_mm, 35.0);
        assert_eq!(ld.iris, 2.8);
    }

    #[test]
    fn test_frustum_data_default() {
        let fd = FrustumData::default();
        assert_eq!(fd.camera_pos, (0.0, 0.0, 0.0));
        assert_eq!(fd.fov_h, 60.0);
    }

    #[test]
    fn test_moco_data_packet() {
        let pkt = MoCoDataPacket::new(1_000, FrustumData::default(), LensData::default());
        assert_eq!(pkt.timestamp_ms, 1_000);
    }

    #[test]
    fn test_frustum_tracker_new() {
        let tracker = FrustumTracker::new();
        assert!(tracker.is_empty());
        assert_eq!(tracker.len(), 0);
        assert_eq!(tracker.history().len(), 0);
    }

    #[test]
    fn test_frustum_tracker_update_single() {
        let mut tracker = FrustumTracker::new();
        let fd = FrustumData {
            camera_pos: (1.0, 2.0, 3.0),
            ..FrustumData::default()
        };
        tracker.update(fd);
        assert_eq!(tracker.len(), 1);
        assert!(!tracker.is_empty());

        let latest = tracker.latest().expect("should succeed in test");
        assert_eq!(latest.camera_pos, (1.0, 2.0, 3.0));
    }

    #[test]
    fn test_frustum_tracker_capacity() {
        let mut tracker = FrustumTracker::new();
        for i in 0..70 {
            tracker.update(FrustumData {
                camera_pos: (i as f32, 0.0, 0.0),
                ..FrustumData::default()
            });
        }
        // Should not exceed capacity
        assert_eq!(tracker.len(), FrustumTracker::CAPACITY);
    }

    #[test]
    fn test_frustum_tracker_latest_after_wrap() {
        let mut tracker = FrustumTracker::new();
        for i in 0..65 {
            tracker.update(FrustumData {
                camera_pos: (i as f32, 0.0, 0.0),
                ..FrustumData::default()
            });
        }
        let latest = tracker.latest().expect("should succeed in test");
        assert_eq!(latest.camera_pos.0, 64.0);
    }

    #[test]
    fn test_frustum_smoother_single_sample() {
        let history = [FrustumData {
            camera_pos: (5.0, 0.0, 0.0),
            ..FrustumData::default()
        }];
        let result = FrustumSmoother::smooth(&history, 0.5);
        assert_eq!(result.camera_pos.0, 5.0);
    }

    #[test]
    fn test_frustum_smoother_empty() {
        let result = FrustumSmoother::smooth(&[], 0.5);
        assert_eq!(result, FrustumData::default());
    }

    #[test]
    fn test_frustum_smoother_convergence() {
        // All samples the same value: smoothed result should equal that value
        let val = FrustumData {
            camera_pos: (10.0, 20.0, 30.0),
            fov_h: 70.0,
            ..FrustumData::default()
        };
        let history = vec![val; 20];
        let result = FrustumSmoother::smooth(&history, 0.1);
        assert!((result.camera_pos.0 - 10.0).abs() < 0.1);
        assert!((result.fov_h - 70.0).abs() < 0.1);
    }

    #[test]
    fn test_frustum_smoother_alpha_clamped() {
        let history = [FrustumData::default()];
        // Should not panic with out-of-range alpha
        let _ = FrustumSmoother::smooth(&history, -1.0);
        let _ = FrustumSmoother::smooth(&history, 2.0);
    }

    #[test]
    fn test_latency_compensator_no_history() {
        let comp = IcvfxLatencyCompensator::new(2);
        let current = FrustumData {
            camera_pos: (1.0, 2.0, 3.0),
            ..FrustumData::default()
        };
        let result = comp.compensate(current, &[]);
        assert_eq!(result.camera_pos, (1.0, 2.0, 3.0));
    }

    #[test]
    fn test_latency_compensator_exact_delay() {
        let comp = IcvfxLatencyCompensator::new(2);
        let history: Vec<FrustumData> = (0..5)
            .map(|i| FrustumData {
                camera_pos: (i as f32, 0.0, 0.0),
                ..FrustumData::default()
            })
            .collect();
        // history[4] = most recent (pos.x=4), delay=2 → should return history[2] (pos.x=2)
        let current = FrustumData::default();
        let result = comp.compensate(current, &history);
        assert_eq!(result.camera_pos.0, 2.0);
    }

    #[test]
    fn test_latency_compensator_delay_exceeds_history() {
        let comp = IcvfxLatencyCompensator::new(10);
        let history = vec![FrustumData {
            camera_pos: (99.0, 0.0, 0.0),
            ..FrustumData::default()
        }];
        let current = FrustumData::default();
        let result = comp.compensate(current, &history);
        // Falls back to oldest = history[0]
        assert_eq!(result.camera_pos.0, 99.0);
    }

    #[test]
    fn test_latency_compensator_zero_delay() {
        let comp = IcvfxLatencyCompensator::new(0);
        let history = vec![FrustumData {
            camera_pos: (5.0, 0.0, 0.0),
            ..FrustumData::default()
        }];
        let current = FrustumData {
            camera_pos: (42.0, 0.0, 0.0),
            ..FrustumData::default()
        };
        let result = comp.compensate(current, &history);
        assert_eq!(result.camera_pos.0, 42.0); // returns current when delay=0
    }
}
