//! Motion blur synthesis.
//!
//! This module provides algorithms for creating realistic motion blur effects.

use super::{
    DepthMap, MotionBlurConfig, MotionVector, MotionVectorField, QualityMode, RollingShutterParams,
};
use crate::error::{CvError, CvResult};
use crate::tracking::{FlowField, FlowMethod, OpticalFlow};

/// Motion blur synthesizer.
///
/// Creates realistic motion blur effects from input frames and motion information.
///
/// # Examples
///
/// ```
/// use oximedia_cv::motion_blur::{MotionBlurSynthesizer, QualityMode};
///
/// let synthesizer = MotionBlurSynthesizer::new()
///     .with_shutter_angle(180.0)
///     .with_samples(16)
///     .with_quality(QualityMode::HighQuality);
/// ```
#[derive(Debug, Clone)]
pub struct MotionBlurSynthesizer {
    config: MotionBlurConfig,
}

impl MotionBlurSynthesizer {
    /// Create a new motion blur synthesizer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: MotionBlurConfig::default(),
        }
    }

    /// Set shutter angle in degrees (0-360).
    #[must_use]
    pub fn with_shutter_angle(mut self, angle: f32) -> Self {
        self.config.shutter_angle = angle;
        self
    }

    /// Set number of accumulation samples.
    #[must_use]
    pub fn with_samples(mut self, samples: usize) -> Self {
        self.config.samples = samples;
        self
    }

    /// Set quality mode.
    #[must_use]
    pub fn with_quality(mut self, quality: QualityMode) -> Self {
        self.config.quality = quality;
        self
    }

    /// Enable depth-aware blur.
    #[must_use]
    pub fn with_depth_aware(mut self, enabled: bool) -> Self {
        self.config.depth_aware = enabled;
        self
    }

    /// Enable rolling shutter compensation.
    #[must_use]
    pub fn with_rolling_shutter(mut self, enabled: bool) -> Self {
        self.config.rolling_shutter = enabled;
        self
    }

    /// Apply motion blur to an RGB image using motion vectors.
    ///
    /// # Arguments
    ///
    /// * `image` - RGB image data (width * height * 3)
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `motion` - Motion vector field
    pub fn apply_blur(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        motion: &MotionVectorField,
    ) -> CvResult<Vec<u8>> {
        self.config.validate()?;

        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected_size = (width * height * 3) as usize;
        if image.len() != expected_size {
            return Err(CvError::insufficient_data(expected_size, image.len()));
        }

        match self.config.quality {
            QualityMode::Fast => self.apply_directional_blur(image, width, height, motion),
            QualityMode::Balanced => self.apply_flow_blur(image, width, height, motion),
            QualityMode::HighQuality => self.apply_accumulation_blur(image, width, height, motion),
        }
    }

    /// Apply motion blur with depth information.
    pub fn apply_blur_with_depth(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        motion: &MotionVectorField,
        depth: &DepthMap,
    ) -> CvResult<Vec<u8>> {
        self.config.validate()?;

        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected_size = (width * height * 3) as usize;
        if image.len() != expected_size {
            return Err(CvError::insufficient_data(expected_size, image.len()));
        }

        // Apply depth-scaled motion blur
        self.apply_depth_aware_blur(image, width, height, motion, depth)
    }

    /// Fast directional blur using box filter.
    fn apply_directional_blur(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        motion: &MotionVectorField,
    ) -> CvResult<Vec<u8>> {
        let mut output = vec![0u8; image.len()];
        let motion_scale = self.config.motion_scale();

        for y in 0..height {
            for x in 0..width {
                let motion_vec = motion.get(x, y);
                let scaled_dx = motion_vec.dx * motion_scale;
                let scaled_dy = motion_vec.dy * motion_scale;

                // Skip if motion is negligible
                if scaled_dx.abs() < 0.5 && scaled_dy.abs() < 0.5 {
                    let src_idx = ((y * width + x) * 3) as usize;
                    let dst_idx = src_idx;
                    output[dst_idx] = image[src_idx];
                    output[dst_idx + 1] = image[src_idx + 1];
                    output[dst_idx + 2] = image[src_idx + 2];
                    continue;
                }

                // Sample along motion direction
                let samples = self.config.samples;
                let mut r_sum = 0u32;
                let mut g_sum = 0u32;
                let mut b_sum = 0u32;
                let mut count = 0u32;

                for i in 0..samples {
                    let t = i as f32 / samples as f32 - 0.5;
                    let sample_x = x as f32 + scaled_dx * t;
                    let sample_y = y as f32 + scaled_dy * t;

                    if let Some((r, g, b)) = sample_rgb(image, width, height, sample_x, sample_y) {
                        r_sum += r as u32;
                        g_sum += g as u32;
                        b_sum += b as u32;
                        count += 1;
                    }
                }

                let dst_idx = ((y * width + x) * 3) as usize;
                if let Some(r_avg) = r_sum.checked_div(count) {
                    output[dst_idx] = r_avg as u8;
                    output[dst_idx + 1] = g_sum.checked_div(count).unwrap_or(0) as u8;
                    output[dst_idx + 2] = b_sum.checked_div(count).unwrap_or(0) as u8;
                } else {
                    let src_idx = dst_idx;
                    output[dst_idx] = image[src_idx];
                    output[dst_idx + 1] = image[src_idx + 1];
                    output[dst_idx + 2] = image[src_idx + 2];
                }
            }
        }

        Ok(output)
    }

    /// Flow-based blur with optical flow refinement.
    fn apply_flow_blur(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        motion: &MotionVectorField,
    ) -> CvResult<Vec<u8>> {
        // Similar to directional but with motion smoothing
        let mut smoothed_motion = motion.clone();
        smoothed_motion.median_filter(2);

        self.apply_directional_blur(image, width, height, &smoothed_motion)
    }

    /// High-quality accumulation buffer blur.
    fn apply_accumulation_blur(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        motion: &MotionVectorField,
    ) -> CvResult<Vec<u8>> {
        let mut accumulator = AccumulationBuffer::new(width, height);
        let motion_scale = self.config.motion_scale();
        let samples = self.config.samples;

        // Accumulate samples along motion path
        for sample in 0..samples {
            let t = sample as f32 / samples as f32 - 0.5;

            for y in 0..height {
                for x in 0..width {
                    let motion_vec = motion.get(x, y);
                    let offset_x = motion_vec.dx * motion_scale * t;
                    let offset_y = motion_vec.dy * motion_scale * t;

                    let sample_x = x as f32 + offset_x;
                    let sample_y = y as f32 + offset_y;

                    if let Some((r, g, b)) = sample_rgb(image, width, height, sample_x, sample_y) {
                        accumulator.add_sample(x, y, r, g, b, 1.0);
                    }
                }
            }
        }

        Ok(accumulator.to_image())
    }

    /// Depth-aware blur with depth-dependent blur amount.
    fn apply_depth_aware_blur(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        motion: &MotionVectorField,
        depth: &DepthMap,
    ) -> CvResult<Vec<u8>> {
        let mut accumulator = AccumulationBuffer::new(width, height);
        let motion_scale = self.config.motion_scale();
        let samples = self.config.samples;

        for y in 0..height {
            for x in 0..width {
                let depth_val = depth.get(x, y);
                let motion_vec = motion.get(x, y);

                // Scale motion by depth (closer objects = more blur)
                let depth_scale = 1.0 - depth_val;
                let scaled_motion = motion_vec.scale(depth_scale);

                let mut r_sum = 0u32;
                let mut g_sum = 0u32;
                let mut b_sum = 0u32;
                let mut count = 0u32;

                for sample in 0..samples {
                    let t = sample as f32 / samples as f32 - 0.5;
                    let offset_x = scaled_motion.dx * motion_scale * t;
                    let offset_y = scaled_motion.dy * motion_scale * t;

                    let sample_x = x as f32 + offset_x;
                    let sample_y = y as f32 + offset_y;

                    if let Some((r, g, b)) = sample_rgb(image, width, height, sample_x, sample_y) {
                        r_sum += r as u32;
                        g_sum += g as u32;
                        b_sum += b as u32;
                        count += 1;
                    }
                }

                if let Some(r_avg) = r_sum.checked_div(count) {
                    accumulator.add_sample(
                        x,
                        y,
                        r_avg as u8,
                        g_sum.checked_div(count).unwrap_or(0) as u8,
                        b_sum.checked_div(count).unwrap_or(0) as u8,
                        1.0,
                    );
                }
            }
        }

        Ok(accumulator.to_image())
    }

    /// Estimate motion from two consecutive frames.
    pub fn estimate_motion(
        &self,
        frame1: &[u8],
        frame2: &[u8],
        width: u32,
        height: u32,
    ) -> CvResult<MotionVectorField> {
        // Convert to grayscale if needed (assuming input is already Y plane or grayscale)
        let flow_estimator = OpticalFlow::new(FlowMethod::LucasKanade)
            .with_window_size(21)
            .with_max_level(3);

        let flow = flow_estimator.compute(frame1, frame2, width, height)?;
        Ok(MotionVectorField::from_flow_field(&flow))
    }
}

impl Default for MotionBlurSynthesizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Accumulation buffer for high-quality motion blur.
pub struct AccumulationBuffer {
    width: u32,
    height: u32,
    r_buffer: Vec<f32>,
    g_buffer: Vec<f32>,
    b_buffer: Vec<f32>,
    weight_buffer: Vec<f32>,
}

impl AccumulationBuffer {
    /// Create a new accumulation buffer.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width * height) as usize;
        Self {
            width,
            height,
            r_buffer: vec![0.0; size],
            g_buffer: vec![0.0; size],
            b_buffer: vec![0.0; size],
            weight_buffer: vec![0.0; size],
        }
    }

    /// Add a sample to the accumulation buffer.
    pub fn add_sample(&mut self, x: u32, y: u32, r: u8, g: u8, b: u8, weight: f32) {
        if x >= self.width || y >= self.height {
            return;
        }

        let idx = (y * self.width + x) as usize;
        if idx < self.r_buffer.len() {
            self.r_buffer[idx] += r as f32 * weight;
            self.g_buffer[idx] += g as f32 * weight;
            self.b_buffer[idx] += b as f32 * weight;
            self.weight_buffer[idx] += weight;
        }
    }

    /// Convert accumulation buffer to final image.
    #[must_use]
    pub fn to_image(&self) -> Vec<u8> {
        let mut output = vec![0u8; (self.width * self.height * 3) as usize];

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = (y * self.width + x) as usize;
                let out_idx = idx * 3;

                if idx < self.weight_buffer.len() && self.weight_buffer[idx] > 0.0 {
                    let weight = self.weight_buffer[idx];
                    output[out_idx] = (self.r_buffer[idx] / weight).min(255.0) as u8;
                    output[out_idx + 1] = (self.g_buffer[idx] / weight).min(255.0) as u8;
                    output[out_idx + 2] = (self.b_buffer[idx] / weight).min(255.0) as u8;
                }
            }
        }

        output
    }

    /// Clear the accumulation buffer.
    pub fn clear(&mut self) {
        for i in 0..self.r_buffer.len() {
            self.r_buffer[i] = 0.0;
            self.g_buffer[i] = 0.0;
            self.b_buffer[i] = 0.0;
            self.weight_buffer[i] = 0.0;
        }
    }
}

/// Camera shake effect generator.
pub struct CameraShake {
    /// Shake amplitude in pixels.
    pub amplitude: f32,
    /// Shake frequency in Hz.
    pub frequency: f32,
    /// Random seed for shake pattern.
    pub seed: u64,
}

impl CameraShake {
    /// Create a new camera shake generator.
    #[must_use]
    pub const fn new(amplitude: f32, frequency: f32) -> Self {
        Self {
            amplitude,
            frequency,
            seed: 12345,
        }
    }

    /// Generate camera shake motion for a frame.
    #[must_use]
    pub fn generate_motion(&self, frame_time: f32) -> MotionVector {
        // Simple sinusoidal shake with some randomness
        let phase = frame_time * self.frequency * 2.0 * std::f32::consts::PI;
        let noise_x = self.perlin_noise(frame_time, 0.0);
        let noise_y = self.perlin_noise(frame_time, 100.0);

        let dx = self.amplitude * (phase.sin() + noise_x * 0.5);
        let dy = self.amplitude * (phase.cos() + noise_y * 0.5);

        MotionVector::new(dx, dy)
    }

    /// Apply camera shake to a motion field.
    pub fn apply_to_field(&self, field: &mut MotionVectorField, frame_time: f32) {
        let shake = self.generate_motion(frame_time);

        for vector in &mut field.vectors {
            vector.dx += shake.dx;
            vector.dy += shake.dy;
        }
    }

    /// Simple Perlin-like noise function.
    fn perlin_noise(&self, x: f32, offset: f32) -> f32 {
        let xi = ((x + offset) * 10.0) as i32;
        let seed = self.seed.wrapping_add(xi as u64);
        let hash = seed.wrapping_mul(2_654_435_761);
        let normalized = (hash & 0xFFFF) as f32 / 65_535.0;
        normalized * 2.0 - 1.0
    }
}

impl Default for CameraShake {
    fn default() -> Self {
        Self::new(2.0, 5.0)
    }
}

/// Per-object motion blur.
pub struct PerObjectBlur {
    /// Object masks (one per object, 0=background, 1=foreground).
    pub masks: Vec<Vec<u8>>,
    /// Motion vectors per object.
    pub object_motions: Vec<MotionVector>,
}

impl PerObjectBlur {
    /// Create a new per-object blur processor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            masks: Vec::new(),
            object_motions: Vec::new(),
        }
    }

    /// Add an object with its mask and motion.
    pub fn add_object(&mut self, mask: Vec<u8>, motion: MotionVector) {
        self.masks.push(mask);
        self.object_motions.push(motion);
    }

    /// Generate motion field from per-object motions.
    #[must_use]
    pub fn generate_motion_field(&self, width: u32, height: u32) -> MotionVectorField {
        let mut field = MotionVectorField::new(width, height);

        for (mask, motion) in self.masks.iter().zip(self.object_motions.iter()) {
            for y in 0..height {
                for x in 0..width {
                    let idx = (y * width + x) as usize;
                    if idx < mask.len() && mask[idx] > 128 {
                        field.set(x, y, *motion);
                    }
                }
            }
        }

        field
    }
}

impl Default for PerObjectBlur {
    fn default() -> Self {
        Self::new()
    }
}

/// Motion trail effect generator.
pub struct MotionTrail {
    /// Trail length in frames.
    pub length: usize,
    /// Trail decay factor (0.0 to 1.0).
    pub decay: f32,
}

impl MotionTrail {
    /// Create a new motion trail generator.
    #[must_use]
    pub const fn new(length: usize, decay: f32) -> Self {
        Self { length, decay }
    }

    /// Generate motion trail effect.
    pub fn apply(
        &self,
        frames: &[Vec<u8>],
        width: u32,
        height: u32,
        motion_fields: &[MotionVectorField],
    ) -> CvResult<Vec<u8>> {
        if frames.is_empty() {
            return Err(CvError::insufficient_data(1, 0));
        }

        let expected_size = (width * height * 3) as usize;
        if frames[0].len() != expected_size {
            return Err(CvError::insufficient_data(expected_size, frames[0].len()));
        }

        let mut accumulator = AccumulationBuffer::new(width, height);
        let trail_len = self.length.min(frames.len());

        // Accumulate past frames with decay
        for i in 0..trail_len {
            let frame_idx = frames.len() - 1 - i;
            let weight = self.decay.powi(i as i32);

            if frame_idx < frames.len() {
                let frame = &frames[frame_idx];

                for y in 0..height {
                    for x in 0..width {
                        let idx = ((y * width + x) * 3) as usize;
                        if idx + 2 < frame.len() {
                            accumulator.add_sample(
                                x,
                                y,
                                frame[idx],
                                frame[idx + 1],
                                frame[idx + 2],
                                weight,
                            );
                        }
                    }
                }
            }
        }

        Ok(accumulator.to_image())
    }
}

impl Default for MotionTrail {
    fn default() -> Self {
        Self::new(5, 0.7)
    }
}

/// Speed ramp effect for variable motion blur.
pub struct SpeedRamp {
    /// Speed curve (time -> speed multiplier).
    pub curve: Vec<(f32, f32)>,
}

impl SpeedRamp {
    /// Create a new speed ramp effect.
    #[must_use]
    pub fn new() -> Self {
        Self {
            curve: vec![(0.0, 1.0), (1.0, 1.0)],
        }
    }

    /// Add a control point to the speed curve.
    pub fn add_point(&mut self, time: f32, speed: f32) {
        self.curve.push((time, speed));
        self.curve
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Get speed multiplier at a given time.
    #[must_use]
    pub fn get_speed(&self, time: f32) -> f32 {
        if self.curve.is_empty() {
            return 1.0;
        }

        // Find surrounding control points
        let mut before = &self.curve[0];
        let mut after = &self.curve[self.curve.len() - 1];

        for i in 0..self.curve.len() - 1 {
            if time >= self.curve[i].0 && time <= self.curve[i + 1].0 {
                before = &self.curve[i];
                after = &self.curve[i + 1];
                break;
            }
        }

        // Linear interpolation
        if (after.0 - before.0).abs() < f32::EPSILON {
            return before.1;
        }

        let t = (time - before.0) / (after.0 - before.0);
        before.1 + (after.1 - before.1) * t
    }

    /// Apply speed ramp to motion field.
    pub fn apply_to_field(&self, field: &mut MotionVectorField, time: f32) {
        let speed = self.get_speed(time);
        field.scale(speed);
    }
}

impl Default for SpeedRamp {
    fn default() -> Self {
        Self::new()
    }
}

/// Sample RGB value with bilinear interpolation.
fn sample_rgb(image: &[u8], width: u32, height: u32, x: f32, y: f32) -> Option<(u8, u8, u8)> {
    if x < 0.0 || y < 0.0 || x >= (width - 1) as f32 || y >= (height - 1) as f32 {
        return None;
    }

    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let idx00 = ((y0 * width + x0) * 3) as usize;
    let idx10 = ((y0 * width + x1) * 3) as usize;
    let idx01 = ((y1 * width + x0) * 3) as usize;
    let idx11 = ((y1 * width + x1) * 3) as usize;

    if idx00 + 2 >= image.len()
        || idx10 + 2 >= image.len()
        || idx01 + 2 >= image.len()
        || idx11 + 2 >= image.len()
    {
        return None;
    }

    let mut result = [0u8; 3];
    for c in 0..3 {
        let v00 = image[idx00 + c] as f32;
        let v10 = image[idx10 + c] as f32;
        let v01 = image[idx01 + c] as f32;
        let v11 = image[idx11 + c] as f32;

        let v0 = v00 + (v10 - v00) * fx;
        let v1 = v01 + (v11 - v01) * fx;
        result[c] = (v0 + (v1 - v0) * fy).min(255.0) as u8;
    }

    Some((result[0], result[1], result[2]))
}

/// Apply rolling shutter effect to motion field.
pub fn apply_rolling_shutter(field: &mut MotionVectorField, params: &RollingShutterParams) {
    for y in 0..field.height {
        let time_offset = params.time_offset(y, field.height);

        for x in 0..field.width {
            let idx = (y * field.width + x) as usize;
            if idx < field.vectors.len() {
                let vector = &mut field.vectors[idx];
                vector.dx *= time_offset;
                vector.dy *= time_offset;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_blur_synthesizer_new() {
        let synth = MotionBlurSynthesizer::new();
        assert_eq!(synth.config.shutter_angle, 180.0);
    }

    #[test]
    fn test_accumulation_buffer() {
        let mut buffer = AccumulationBuffer::new(10, 10);
        buffer.add_sample(5, 5, 100, 150, 200, 1.0);
        let image = buffer.to_image();
        assert_eq!(image.len(), 300);
    }

    #[test]
    fn test_camera_shake() {
        let shake = CameraShake::new(5.0, 10.0);
        let motion = shake.generate_motion(0.0);
        assert!(motion.magnitude() <= shake.amplitude * 2.0);
    }

    #[test]
    fn test_per_object_blur() {
        let mut blur = PerObjectBlur::new();
        let mask = vec![255u8; 100];
        let motion = MotionVector::new(5.0, 10.0);
        blur.add_object(mask, motion);
        assert_eq!(blur.masks.len(), 1);
    }

    #[test]
    fn test_motion_trail() {
        let trail = MotionTrail::new(5, 0.8);
        assert_eq!(trail.length, 5);
        assert!((trail.decay - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_speed_ramp() {
        let mut ramp = SpeedRamp::new();
        ramp.add_point(0.5, 2.0);
        let speed = ramp.get_speed(0.5);
        assert!((speed - 2.0).abs() < 0.001);
    }
}
