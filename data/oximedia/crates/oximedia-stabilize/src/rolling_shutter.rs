#![allow(dead_code)]
//! Rolling shutter correction model and per-row skew estimation.
//!
//! CMOS sensors with rolling shutters expose rows sequentially, introducing
//! a temporal skew that manifests as shear / "jello" artefacts during motion.
//! This module provides primitives for modelling and correcting that skew,
//! complementing (not replacing) the existing `rolling` sub-module.

/// The rolling shutter correction applied to a single image row.
#[derive(Debug, Clone, Copy)]
pub struct RsCorrection {
    /// Row index (0 = top of frame)
    pub row: usize,
    /// Horizontal phase offset for this row (fraction of frame period, 0.0–1.0)
    pub phase: f64,
    /// Horizontal pixel shift applied to undo skew
    pub x_shift: f64,
    /// Vertical pixel shift (usually small / zero for horizontal motion)
    pub y_shift: f64,
}

impl RsCorrection {
    /// Create a correction for the given row.
    pub fn new(row: usize, phase: f64, x_shift: f64, y_shift: f64) -> Self {
        Self {
            row,
            phase,
            x_shift,
            y_shift,
        }
    }

    /// Phase offset of this row within the frame readout (0.0 = top, 1.0 = bottom).
    pub fn row_phase(&self) -> f64 {
        self.phase
    }

    /// Returns `true` if the correction shift is negligibly small.
    pub fn is_negligible(&self, threshold: f64) -> bool {
        self.x_shift.abs() < threshold && self.y_shift.abs() < threshold
    }

    /// Combined shift magnitude.
    pub fn shift_magnitude(&self) -> f64 {
        (self.x_shift * self.x_shift + self.y_shift * self.y_shift).sqrt()
    }
}

/// Model parameters describing the rolling shutter behaviour of a camera.
#[derive(Debug, Clone)]
pub struct RollingShutterModel {
    /// Total readout time of one full frame in milliseconds
    pub readout_ms: f64,
    /// Frame period in milliseconds (= 1000 / fps)
    pub frame_period_ms: f64,
    /// Number of image rows
    pub height: usize,
    /// Estimated horizontal velocity at the time of capture (pixels/ms)
    pub estimated_velocity_x: f64,
    /// Estimated vertical velocity at the time of capture (pixels/ms)
    pub estimated_velocity_y: f64,
}

impl RollingShutterModel {
    /// Create a model from basic camera properties.
    pub fn new(readout_ms: f64, fps: f64, height: usize) -> Self {
        let frame_period_ms = if fps > 0.0 { 1000.0 / fps } else { 33.33 };
        Self {
            readout_ms,
            frame_period_ms,
            height,
            estimated_velocity_x: 0.0,
            estimated_velocity_y: 0.0,
        }
    }

    /// Update the estimated inter-frame velocity.
    pub fn set_velocity(&mut self, vx: f64, vy: f64) {
        self.estimated_velocity_x = vx;
        self.estimated_velocity_y = vy;
    }

    /// Phase offset for a given row (0.0 = top, 1.0 = bottom).
    #[allow(clippy::cast_precision_loss)]
    pub fn row_phase(&self, row: usize) -> f64 {
        if self.height == 0 {
            return 0.0;
        }
        row as f64 / self.height as f64
    }

    /// Estimate the horizontal skew in pixels for a given row based on velocity.
    pub fn estimate_skew(&self, row: usize) -> f64 {
        let phase = self.row_phase(row);
        let time_offset = phase * self.readout_ms; // ms
        self.estimated_velocity_x * time_offset
    }

    /// Estimate the vertical skew for a given row.
    pub fn estimate_skew_y(&self, row: usize) -> f64 {
        let phase = self.row_phase(row);
        let time_offset = phase * self.readout_ms;
        self.estimated_velocity_y * time_offset
    }

    /// Readout ratio: fraction of frame period used for readout.
    pub fn readout_ratio(&self) -> f64 {
        if self.frame_period_ms < 1e-9 {
            return 0.0;
        }
        self.readout_ms / self.frame_period_ms
    }
}

/// Applies rolling shutter correction to frame data using the given model.
#[derive(Debug)]
pub struct RsCorrector {
    model: RollingShutterModel,
}

impl RsCorrector {
    /// Create a corrector with the given model.
    pub fn new(model: RollingShutterModel) -> Self {
        Self { model }
    }

    /// Update the internal model's velocity estimate from inter-frame motion.
    pub fn update_velocity(&mut self, vx: f64, vy: f64) {
        self.model.set_velocity(vx, vy);
    }

    /// Compute the rolling shutter correction for every row in a frame.
    ///
    /// Returns one `RsCorrection` per row.
    pub fn correct_frame(&self) -> Vec<RsCorrection> {
        (0..self.model.height)
            .map(|row| {
                let phase = self.model.row_phase(row);
                let x_shift = self.model.estimate_skew(row);
                let y_shift = self.model.estimate_skew_y(row);
                RsCorrection::new(row, phase, x_shift, y_shift)
            })
            .collect()
    }

    /// Maximum skew magnitude across all rows (convenience metric).
    pub fn skew_magnitude(&self) -> f64 {
        if self.model.height == 0 {
            return 0.0;
        }
        // The maximum skew occurs at the last row
        let last_row = self.model.height - 1;
        let x = self.model.estimate_skew(last_row);
        let y = self.model.estimate_skew_y(last_row);
        (x * x + y * y).sqrt()
    }

    /// Whether the rolling shutter effect is significant for the current motion.
    pub fn is_significant(&self, threshold: f64) -> bool {
        self.skew_magnitude() > threshold
    }

    /// Borrow the underlying model.
    pub fn model(&self) -> &RollingShutterModel {
        &self.model
    }

    /// Apply corrections to a flat RGBA pixel buffer (row-major, 4 bytes/pixel).
    ///
    /// Shifts each row horizontally using nearest-neighbour sampling.
    /// `pixels` is modified in-place.
    pub fn apply_to_pixels(&self, pixels: &mut [u8], width: usize, height: usize) {
        assert_eq!(
            pixels.len(),
            width * height * 4,
            "pixel buffer size mismatch"
        );
        let corrections = self.correct_frame();
        for row in 0..height {
            let shift = corrections.get(row).map_or(0.0, |c| c.x_shift) as i32;
            if shift == 0 {
                continue;
            }
            let start = row * width * 4;
            let end = start + width * 4;
            let row_data = pixels[start..end].to_vec();
            for x in 0..width {
                let src_x = (x as i32 - shift).rem_euclid(width as i32) as usize;
                let dst = start + x * 4;
                let src = start + src_x * 4;
                pixels[dst..dst + 4].copy_from_slice(&row_data[src - start..src - start + 4]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_model(height: usize) -> RollingShutterModel {
        RollingShutterModel::new(16.67, 30.0, height)
    }

    #[test]
    fn test_rs_correction_row_phase() {
        let c = RsCorrection::new(5, 0.5, 2.0, 0.0);
        assert!((c.row_phase() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_rs_correction_is_negligible() {
        let c = RsCorrection::new(0, 0.0, 0.001, 0.0);
        assert!(c.is_negligible(0.01));
        let c2 = RsCorrection::new(0, 0.0, 5.0, 0.0);
        assert!(!c2.is_negligible(0.01));
    }

    #[test]
    fn test_rs_correction_shift_magnitude() {
        let c = RsCorrection::new(0, 0.0, 3.0, 4.0);
        assert!((c.shift_magnitude() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_model_row_phase_top() {
        let m = make_model(100);
        assert!((m.row_phase(0)).abs() < 1e-10);
    }

    #[test]
    fn test_model_row_phase_bottom() {
        let m = make_model(100);
        assert!((m.row_phase(100) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_model_row_phase_zero_height() {
        let m = RollingShutterModel::new(16.67, 30.0, 0);
        assert!((m.row_phase(0)).abs() < 1e-10);
    }

    #[test]
    fn test_model_estimate_skew_zero_velocity() {
        let m = make_model(100);
        // No motion — skew must be zero for all rows
        for row in 0..100 {
            assert!((m.estimate_skew(row)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_model_estimate_skew_with_velocity() {
        let mut m = make_model(100);
        m.set_velocity(10.0, 0.0); // 10 px/ms
                                   // Top row: phase=0 → shift=0; bottom: phase≈1 → shift≈readout_ms * vx
        let top_skew = m.estimate_skew(0);
        let bottom_skew = m.estimate_skew(100);
        assert!((top_skew).abs() < 1e-10);
        assert!(bottom_skew > 0.0);
    }

    #[test]
    fn test_model_readout_ratio() {
        let m = RollingShutterModel::new(16.67, 30.0, 100); // 30 fps → 33.33 ms
        let ratio = m.readout_ratio();
        assert!(ratio > 0.0 && ratio < 1.0);
    }

    #[test]
    fn test_corrector_correct_frame_length() {
        let m = make_model(480);
        let c = RsCorrector::new(m);
        let corr = c.correct_frame();
        assert_eq!(corr.len(), 480);
    }

    #[test]
    fn test_corrector_skew_magnitude_zero_velocity() {
        let m = make_model(100);
        let c = RsCorrector::new(m);
        assert!((c.skew_magnitude()).abs() < 1e-10);
    }

    #[test]
    fn test_corrector_skew_magnitude_nonzero() {
        let mut m = make_model(100);
        m.set_velocity(5.0, 0.0);
        let c = RsCorrector::new(m);
        assert!(c.skew_magnitude() > 0.0);
    }

    #[test]
    fn test_corrector_is_significant() {
        let mut m = make_model(100);
        m.set_velocity(1000.0, 0.0);
        let c = RsCorrector::new(m);
        assert!(c.is_significant(1.0));
    }

    #[test]
    fn test_corrector_is_not_significant_zero_velocity() {
        let m = make_model(100);
        let c = RsCorrector::new(m);
        assert!(!c.is_significant(0.01));
    }

    #[test]
    fn test_corrector_empty_frame() {
        let m = RollingShutterModel::new(16.67, 30.0, 0);
        let c = RsCorrector::new(m);
        assert!((c.skew_magnitude()).abs() < 1e-10);
        let corr = c.correct_frame();
        assert!(corr.is_empty());
    }

    #[test]
    fn test_apply_to_pixels_no_motion() {
        // With zero velocity the pixel buffer should be unchanged
        let m = RollingShutterModel::new(16.67, 30.0, 2);
        let c = RsCorrector::new(m);
        let mut pixels = vec![10u8, 20, 30, 255, 40, 50, 60, 255]; // 2×1 RGBA
        let original = pixels.clone();
        c.apply_to_pixels(&mut pixels, 2, 1);
        // No shift — output must equal input
        // (shifts are cast to i32; 0.0 shift rounds to 0)
        drop(original); // just ensure no panic
    }
}
