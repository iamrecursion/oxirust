//! Camera stabilization using tracking data.

use super::point::{TrackPoint, TrackingConfig, TwoPointTracker};
use crate::{Frame, VfxError, VfxResult};
use serde::{Deserialize, Serialize};

/// Stabilization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StabilizationMode {
    /// Stabilize position only.
    Position,
    /// Stabilize position and rotation.
    PositionRotation,
    /// Stabilize position, rotation, and scale.
    PositionRotationScale,
}

/// Stabilization parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StabilizationParams {
    /// Smoothing strength (0.0 - 1.0).
    pub smoothing: f32,
    /// Maximum translation correction in pixels.
    pub max_translation: f32,
    /// Maximum rotation correction in degrees.
    pub max_rotation: f32,
    /// Maximum scale correction (ratio).
    pub max_scale: f32,
}

impl Default for StabilizationParams {
    fn default() -> Self {
        Self {
            smoothing: 0.5,
            max_translation: 100.0,
            max_rotation: 10.0,
            max_scale: 0.1,
        }
    }
}

/// Transform for stabilization.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StabilizeTransform {
    /// Translation X.
    pub translate_x: f32,
    /// Translation Y.
    pub translate_y: f32,
    /// Rotation in degrees.
    pub rotation: f32,
    /// Scale factor.
    pub scale: f32,
}

impl Default for StabilizeTransform {
    fn default() -> Self {
        Self {
            translate_x: 0.0,
            translate_y: 0.0,
            rotation: 0.0,
            scale: 1.0,
        }
    }
}

/// Camera stabilizer using motion tracking.
#[derive(Debug, Clone)]
pub struct Stabilizer {
    mode: StabilizationMode,
    params: StabilizationParams,
    tracker: Option<Box<TwoPointTracker>>,
    reference_point1: Option<TrackPoint>,
    reference_point2: Option<TrackPoint>,
    transforms: Vec<StabilizeTransform>,
}

impl Stabilizer {
    /// Create a new stabilizer.
    #[must_use]
    pub fn new(mode: StabilizationMode, params: StabilizationParams) -> Self {
        Self {
            mode,
            params,
            tracker: None,
            reference_point1: None,
            reference_point2: None,
            transforms: Vec::new(),
        }
    }

    /// Initialize stabilizer with tracking points.
    pub fn initialize(
        &mut self,
        frame: &Frame,
        point1: TrackPoint,
        point2: Option<TrackPoint>,
    ) -> VfxResult<()> {
        let config = TrackingConfig::default();

        match self.mode {
            StabilizationMode::Position => {
                // Single point tracking
                self.reference_point1 = Some(point1);
            }
            StabilizationMode::PositionRotation | StabilizationMode::PositionRotationScale => {
                let point2 = point2.ok_or_else(|| {
                    VfxError::ProcessingError(
                        "Two points required for rotation stabilization".to_string(),
                    )
                })?;

                let mut tracker = TwoPointTracker::new(config);
                tracker.initialize(frame, point1, point2)?;

                self.tracker = Some(Box::new(tracker));
                self.reference_point1 = Some(point1);
                self.reference_point2 = Some(point2);
            }
        }

        Ok(())
    }

    /// Analyze motion across frames and compute stabilization transforms.
    pub fn analyze(&mut self, frames: &[Frame]) -> VfxResult<()> {
        if frames.is_empty() {
            return Ok(());
        }

        self.transforms.clear();
        self.transforms.reserve(frames.len());

        match self.mode {
            StabilizationMode::Position => {
                self.analyze_position(frames)?;
            }
            StabilizationMode::PositionRotation => {
                self.analyze_position_rotation(frames)?;
            }
            StabilizationMode::PositionRotationScale => {
                self.analyze_position_rotation_scale(frames)?;
            }
        }

        // Apply smoothing
        self.smooth_transforms();

        Ok(())
    }

    fn analyze_position(&mut self, _frames: &[Frame]) -> VfxResult<()> {
        // Simplified - would track single point through frames
        for _ in 0.._frames.len() {
            self.transforms.push(StabilizeTransform::default());
        }
        Ok(())
    }

    fn analyze_position_rotation(&mut self, frames: &[Frame]) -> VfxResult<()> {
        let tracker = self
            .tracker
            .as_ref()
            .ok_or_else(|| VfxError::ProcessingError("Tracker not initialized".to_string()))?;

        let ref1 = self
            .reference_point1
            .ok_or_else(|| VfxError::ProcessingError("Reference point not set".to_string()))?;
        let ref2 = self
            .reference_point2
            .ok_or_else(|| VfxError::ProcessingError("Reference point 2 not set".to_string()))?;

        let ref_dx = ref2.x - ref1.x;
        let ref_dy = ref2.y - ref1.y;
        let ref_angle = ref_dy.atan2(ref_dx);

        for frame in frames {
            let (result1, result2) = tracker.track(frame)?;

            // Calculate current angle
            let cur_dx = result2.point.x - result1.point.x;
            let cur_dy = result2.point.y - result1.point.y;
            let cur_angle = cur_dy.atan2(cur_dx);

            // Calculate transform
            let rotation = (ref_angle - cur_angle).to_degrees();
            let translate_x = ref1.x - result1.point.x;
            let translate_y = ref1.y - result1.point.y;

            self.transforms.push(StabilizeTransform {
                translate_x: translate_x
                    .clamp(-self.params.max_translation, self.params.max_translation),
                translate_y: translate_y
                    .clamp(-self.params.max_translation, self.params.max_translation),
                rotation: rotation.clamp(-self.params.max_rotation, self.params.max_rotation),
                scale: 1.0,
            });
        }

        Ok(())
    }

    fn analyze_position_rotation_scale(&mut self, frames: &[Frame]) -> VfxResult<()> {
        let tracker = self
            .tracker
            .as_ref()
            .ok_or_else(|| VfxError::ProcessingError("Tracker not initialized".to_string()))?;

        let ref1 = self
            .reference_point1
            .ok_or_else(|| VfxError::ProcessingError("Reference point not set".to_string()))?;
        let ref2 = self
            .reference_point2
            .ok_or_else(|| VfxError::ProcessingError("Reference point 2 not set".to_string()))?;

        let ref_dx = ref2.x - ref1.x;
        let ref_dy = ref2.y - ref1.y;
        let ref_dist = (ref_dx * ref_dx + ref_dy * ref_dy).sqrt();
        let ref_angle = ref_dy.atan2(ref_dx);

        for frame in frames {
            let (result1, result2) = tracker.track(frame)?;

            // Calculate scale
            let cur_dx = result2.point.x - result1.point.x;
            let cur_dy = result2.point.y - result1.point.y;
            let cur_dist = (cur_dx * cur_dx + cur_dy * cur_dy).sqrt();
            let scale = if cur_dist > 0.0 {
                ref_dist / cur_dist
            } else {
                1.0
            };

            // Calculate rotation
            let cur_angle = cur_dy.atan2(cur_dx);
            let rotation = (ref_angle - cur_angle).to_degrees();

            // Calculate translation
            let translate_x = ref1.x - result1.point.x;
            let translate_y = ref1.y - result1.point.y;

            self.transforms.push(StabilizeTransform {
                translate_x: translate_x
                    .clamp(-self.params.max_translation, self.params.max_translation),
                translate_y: translate_y
                    .clamp(-self.params.max_translation, self.params.max_translation),
                rotation: rotation.clamp(-self.params.max_rotation, self.params.max_rotation),
                scale: scale.clamp(1.0 - self.params.max_scale, 1.0 + self.params.max_scale),
            });
        }

        Ok(())
    }

    fn smooth_transforms(&mut self) {
        if self.transforms.len() < 3 {
            return;
        }

        let smoothing = self.params.smoothing;
        let kernel_size = ((smoothing * 10.0) as usize).max(3) | 1; // Odd number
        let half_kernel = kernel_size / 2;

        let original = self.transforms.clone();

        for i in 0..self.transforms.len() {
            let mut sum_tx = 0.0;
            let mut sum_ty = 0.0;
            let mut sum_rot = 0.0;
            let mut sum_scale = 0.0;
            let mut count = 0;

            for j in 0..kernel_size {
                let idx = (i as i32 + j as i32 - half_kernel as i32)
                    .max(0)
                    .min((original.len() - 1) as i32) as usize;

                sum_tx += original[idx].translate_x;
                sum_ty += original[idx].translate_y;
                sum_rot += original[idx].rotation;
                sum_scale += original[idx].scale;
                count += 1;
            }

            self.transforms[i].translate_x = sum_tx / count as f32;
            self.transforms[i].translate_y = sum_ty / count as f32;
            self.transforms[i].rotation = sum_rot / count as f32;
            self.transforms[i].scale = sum_scale / count as f32;
        }
    }

    /// Apply stabilization transform to a frame.
    pub fn apply_transform(
        &self,
        input: &Frame,
        output: &mut Frame,
        frame_index: usize,
    ) -> VfxResult<()> {
        if frame_index >= self.transforms.len() {
            return Err(VfxError::ProcessingError(format!(
                "Frame index {} out of range (have {} transforms)",
                frame_index,
                self.transforms.len()
            )));
        }

        let transform = &self.transforms[frame_index];

        let cx = input.width as f32 / 2.0;
        let cy = input.height as f32 / 2.0;
        let rad = transform.rotation.to_radians();
        let cos = rad.cos();
        let sin = rad.sin();

        for y in 0..output.height {
            for x in 0..output.width {
                // Transform output coordinates to input coordinates
                let mut src_x = x as f32 - cx;
                let mut src_y = y as f32 - cy;

                // Apply inverse scale
                src_x /= transform.scale;
                src_y /= transform.scale;

                // Apply inverse rotation
                let tmp_x = src_x * cos + src_y * sin;
                let tmp_y = -src_x * sin + src_y * cos;
                src_x = tmp_x;
                src_y = tmp_y;

                // Apply inverse translation
                src_x = src_x + cx - transform.translate_x;
                src_y = src_y + cy - transform.translate_y;

                let pixel = if src_x >= 0.0
                    && src_x < input.width as f32
                    && src_y >= 0.0
                    && src_y < input.height as f32
                {
                    self.bilinear_sample(input, src_x, src_y)
                } else {
                    [0, 0, 0, 0]
                };

                output.set_pixel(x, y, pixel);
            }
        }

        Ok(())
    }

    fn bilinear_sample(&self, frame: &Frame, x: f32, y: f32) -> [u8; 4] {
        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(frame.width - 1);
        let y1 = (y0 + 1).min(frame.height - 1);

        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        let p00 = frame.get_pixel(x0, y0).unwrap_or([0, 0, 0, 0]);
        let p10 = frame.get_pixel(x1, y0).unwrap_or([0, 0, 0, 0]);
        let p01 = frame.get_pixel(x0, y1).unwrap_or([0, 0, 0, 0]);
        let p11 = frame.get_pixel(x1, y1).unwrap_or([0, 0, 0, 0]);

        let mut result = [0u8; 4];
        for i in 0..4 {
            let v0 = f32::from(p00[i]) * (1.0 - fx) + f32::from(p10[i]) * fx;
            let v1 = f32::from(p01[i]) * (1.0 - fx) + f32::from(p11[i]) * fx;
            result[i] = (v0 * (1.0 - fy) + v1 * fy) as u8;
        }

        result
    }

    /// Get number of frames analyzed.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.transforms.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stabilization_params_default() {
        let params = StabilizationParams::default();
        assert_eq!(params.smoothing, 0.5);
        assert!(params.max_translation > 0.0);
    }

    #[test]
    fn test_stabilizer_creation() {
        let stabilizer =
            Stabilizer::new(StabilizationMode::Position, StabilizationParams::default());
        assert_eq!(stabilizer.frame_count(), 0);
    }

    #[test]
    fn test_stabilize_transform_default() {
        let transform = StabilizeTransform::default();
        assert_eq!(transform.translate_x, 0.0);
        assert_eq!(transform.translate_y, 0.0);
        assert_eq!(transform.rotation, 0.0);
        assert_eq!(transform.scale, 1.0);
    }
}
