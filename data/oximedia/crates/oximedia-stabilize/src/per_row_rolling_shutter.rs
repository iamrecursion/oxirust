//! Per-row rolling shutter correction.
//!
//! CMOS sensors with electronic rolling shutters expose each row at a slightly
//! different point in time. When the camera moves during the frame readout
//! interval, this causes a skew/wobble (the "jello" effect). This module
//! implements a **per-row motion model** that estimates a separate 2D translation
//! for every scanline, producing much finer correction than the coarser
//! per-scanline-group approach.
//!
//! The model assumes the camera velocity changes linearly within a single frame
//! readout period. Given inter-frame motion vectors `v_prev` and `v_next`, the
//! per-row velocity is interpolated as:
//!
//!   v(row) = v_prev + (v_next - v_prev) * phase(row)
//!
//! where `phase(row) = row / height` represents the temporal phase within the
//! readout period.

use crate::error::{StabilizeError, StabilizeResult};
use crate::transform::calculate::StabilizationTransform;
use crate::Frame;
use scirs2_core::ndarray::Array2;

/// Per-row rolling shutter correction configuration.
#[derive(Debug, Clone)]
pub struct PerRowRsConfig {
    /// Readout time in milliseconds (full frame).
    pub readout_ms: f64,
    /// Frame rate in Hz.
    pub fps: f64,
    /// Whether to apply sub-pixel bilinear interpolation when shifting rows.
    pub bilinear_interpolation: bool,
    /// Minimum velocity magnitude (px/ms) below which correction is skipped.
    pub velocity_threshold: f64,
}

impl Default for PerRowRsConfig {
    fn default() -> Self {
        Self {
            readout_ms: 16.0,
            fps: 30.0,
            bilinear_interpolation: true,
            velocity_threshold: 0.01,
        }
    }
}

/// A per-row motion correction entry.
#[derive(Debug, Clone, Copy)]
pub struct RowCorrection {
    /// Row index.
    pub row: usize,
    /// Temporal phase (0.0 = top, 1.0 = bottom).
    pub phase: f64,
    /// Horizontal shift (pixels, sub-pixel).
    pub dx: f64,
    /// Vertical shift (pixels, sub-pixel).
    pub dy: f64,
    /// Rotation correction at this row (radians).
    pub dangle: f64,
}

impl RowCorrection {
    /// Shift magnitude.
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }
}

/// Per-row rolling shutter corrector.
#[derive(Debug)]
pub struct PerRowRsCorrector {
    config: PerRowRsConfig,
}

impl PerRowRsCorrector {
    /// Create a corrector with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: PerRowRsConfig::default(),
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: PerRowRsConfig) -> Self {
        Self { config }
    }

    /// Correct stabilization transforms using per-row rolling shutter model.
    ///
    /// For each frame, computes the per-row velocity by interpolating between
    /// the motion vectors of the surrounding frames, then adjusts the transform
    /// so that each row is individually corrected.
    ///
    /// Returns a vector of per-frame correction data.
    ///
    /// # Errors
    ///
    /// Returns an error if the inputs are empty or mismatched.
    pub fn correct_transforms(
        &self,
        transforms: &[StabilizationTransform],
        frames: &[Frame],
    ) -> StabilizeResult<Vec<StabilizationTransform>> {
        if transforms.is_empty() || frames.is_empty() {
            return Err(StabilizeError::EmptyFrameSequence);
        }
        if transforms.len() != frames.len() {
            return Err(StabilizeError::dimension_mismatch(
                format!("{}", transforms.len()),
                format!("{}", frames.len()),
            ));
        }

        let height = frames[0].height;
        let n = transforms.len();

        let mut corrected = Vec::with_capacity(n);

        for i in 0..n {
            // Determine inter-frame velocities
            let v_prev = if i > 0 {
                (
                    transforms[i].dx - transforms[i - 1].dx,
                    transforms[i].dy - transforms[i - 1].dy,
                    transforms[i].angle - transforms[i - 1].angle,
                )
            } else {
                (0.0, 0.0, 0.0)
            };
            let v_next = if i + 1 < n {
                (
                    transforms[i + 1].dx - transforms[i].dx,
                    transforms[i + 1].dy - transforms[i].dy,
                    transforms[i + 1].angle - transforms[i].angle,
                )
            } else {
                v_prev
            };

            let row_corrections = self.compute_row_corrections(height, v_prev, v_next);

            // The corrected transform is the original plus the average row correction
            // (the per-row detail is used during warping; here we adjust the
            // global transform to account for the center-row phase)
            let center_row = height / 2;
            let center_corr = row_corrections
                .get(center_row)
                .copied()
                .unwrap_or(RowCorrection {
                    row: center_row,
                    phase: 0.5,
                    dx: 0.0,
                    dy: 0.0,
                    dangle: 0.0,
                });

            corrected.push(StabilizationTransform {
                dx: transforms[i].dx - center_corr.dx,
                dy: transforms[i].dy - center_corr.dy,
                angle: transforms[i].angle - center_corr.dangle,
                scale: transforms[i].scale,
                frame_index: transforms[i].frame_index,
                confidence: transforms[i].confidence,
            });
        }

        Ok(corrected)
    }

    /// Compute per-row corrections for a single frame.
    ///
    /// `v_prev` and `v_next` are the (dx, dy, dangle) velocity vectors for
    /// the previous and next inter-frame motions respectively.
    #[must_use]
    pub fn compute_row_corrections(
        &self,
        height: usize,
        v_prev: (f64, f64, f64),
        v_next: (f64, f64, f64),
    ) -> Vec<RowCorrection> {
        if height == 0 {
            return Vec::new();
        }

        let readout_ratio = self.config.readout_ms
            / (if self.config.fps > 0.0 {
                1000.0 / self.config.fps
            } else {
                33.33
            });

        let mut corrections = Vec::with_capacity(height);

        for row in 0..height {
            let phase = row as f64 / height as f64;

            // Interpolate velocity at this row's exposure time
            let vx = v_prev.0 + (v_next.0 - v_prev.0) * phase;
            let vy = v_prev.1 + (v_next.1 - v_prev.1) * phase;
            let va = v_prev.2 + (v_next.2 - v_prev.2) * phase;

            // The displacement caused by rolling shutter at this row
            // is the velocity times the time offset from center row
            let time_offset = (phase - 0.5) * readout_ratio;

            let dx = vx * time_offset;
            let dy = vy * time_offset;
            let dangle = va * time_offset;

            corrections.push(RowCorrection {
                row,
                phase,
                dx,
                dy,
                dangle,
            });
        }

        corrections
    }

    /// Apply per-row correction to a grayscale frame.
    ///
    /// Returns the corrected frame data.
    pub fn apply_to_frame(
        &self,
        data: &Array2<u8>,
        corrections: &[RowCorrection],
    ) -> StabilizeResult<Array2<u8>> {
        let (h, w) = data.dim();
        if corrections.len() != h {
            return Err(StabilizeError::dimension_mismatch(
                format!("{h}"),
                format!("{}", corrections.len()),
            ));
        }

        let mut result = Array2::zeros((h, w));

        for row in 0..h {
            let corr = &corrections[row];

            if corr.magnitude() < self.config.velocity_threshold {
                // No correction needed for this row
                for x in 0..w {
                    result[[row, x]] = data[[row, x]];
                }
                continue;
            }

            if self.config.bilinear_interpolation {
                self.shift_row_bilinear(data, &mut result, row, corr.dx, corr.dy, w, h);
            } else {
                self.shift_row_nearest(data, &mut result, row, corr.dx, w);
            }
        }

        Ok(result)
    }

    /// Shift a row using nearest-neighbour sampling.
    fn shift_row_nearest(
        &self,
        src: &Array2<u8>,
        dst: &mut Array2<u8>,
        row: usize,
        dx: f64,
        width: usize,
    ) {
        let shift = dx.round() as i32;
        for x in 0..width {
            let src_x = (x as i32 - shift).rem_euclid(width as i32) as usize;
            dst[[row, x]] = src[[row, src_x]];
        }
    }

    /// Shift a row using bilinear interpolation.
    fn shift_row_bilinear(
        &self,
        src: &Array2<u8>,
        dst: &mut Array2<u8>,
        row: usize,
        dx: f64,
        dy: f64,
        width: usize,
        height: usize,
    ) {
        for x in 0..width {
            let src_xf = x as f64 - dx;
            let src_yf = row as f64 - dy;

            let x0 = src_xf.floor() as i32;
            let y0 = src_yf.floor() as i32;
            let fx = src_xf - x0 as f64;
            let fy = src_yf - y0 as f64;

            let sample = |sx: i32, sy: i32| -> f64 {
                let cx = sx.rem_euclid(width as i32) as usize;
                let cy = sy.clamp(0, height as i32 - 1) as usize;
                src[[cy, cx]] as f64
            };

            let val = sample(x0, y0) * (1.0 - fx) * (1.0 - fy)
                + sample(x0 + 1, y0) * fx * (1.0 - fy)
                + sample(x0, y0 + 1) * (1.0 - fx) * fy
                + sample(x0 + 1, y0 + 1) * fx * fy;

            dst[[row, x]] = val.clamp(0.0, 255.0) as u8;
        }
    }
}

impl Default for PerRowRsCorrector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_transform(dx: f64, dy: f64, angle: f64, idx: usize) -> StabilizationTransform {
        StabilizationTransform::new(dx, dy, angle, 1.0, idx)
    }

    fn make_frame(w: usize, h: usize) -> Frame {
        Frame::new(w, h, 0.0, Array2::zeros((h, w)))
    }

    #[test]
    fn test_per_row_config_default() {
        let cfg = PerRowRsConfig::default();
        assert!(cfg.readout_ms > 0.0);
        assert!(cfg.fps > 0.0);
    }

    #[test]
    fn test_corrector_creation() {
        let c = PerRowRsCorrector::new();
        assert!(c.config.readout_ms > 0.0);
    }

    #[test]
    fn test_row_correction_magnitude() {
        let rc = RowCorrection {
            row: 0,
            phase: 0.0,
            dx: 3.0,
            dy: 4.0,
            dangle: 0.0,
        };
        assert!((rc.magnitude() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_row_corrections_zero_velocity() {
        let c = PerRowRsCorrector::new();
        let corrs = c.compute_row_corrections(100, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0));
        assert_eq!(corrs.len(), 100);
        for rc in &corrs {
            assert!(rc.dx.abs() < 1e-10);
            assert!(rc.dy.abs() < 1e-10);
        }
    }

    #[test]
    fn test_compute_row_corrections_nonzero_velocity() {
        let c = PerRowRsCorrector::new();
        let corrs = c.compute_row_corrections(100, (10.0, 0.0, 0.0), (10.0, 0.0, 0.0));
        assert_eq!(corrs.len(), 100);
        // Top row (phase=0): time_offset = (0 - 0.5)*ratio => negative shift
        // Bottom row (phase=1): time_offset = (1 - 0.5)*ratio => positive shift
        let top = &corrs[0];
        let bottom = &corrs[99];
        assert!(top.dx < 0.0, "top row should shift negatively");
        assert!(bottom.dx > 0.0, "bottom row should shift positively");
    }

    #[test]
    fn test_compute_row_corrections_empty_height() {
        let c = PerRowRsCorrector::new();
        let corrs = c.compute_row_corrections(0, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0));
        assert!(corrs.is_empty());
    }

    #[test]
    fn test_compute_row_corrections_symmetry() {
        let c = PerRowRsCorrector::new();
        let corrs = c.compute_row_corrections(200, (5.0, 0.0, 0.0), (5.0, 0.0, 0.0));
        // Center row should have near-zero correction
        let center = &corrs[100];
        assert!(center.dx.abs() < 0.5);
        // Top and bottom should be roughly symmetric (opposite sign)
        let top_dx = corrs[0].dx;
        let bot_dx = corrs[199].dx;
        assert!((top_dx + bot_dx).abs() < 1.0);
    }

    #[test]
    fn test_correct_transforms_empty() {
        let c = PerRowRsCorrector::new();
        let result = c.correct_transforms(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_correct_transforms_mismatch() {
        let c = PerRowRsCorrector::new();
        let transforms = vec![make_transform(0.0, 0.0, 0.0, 0)];
        let frames = vec![make_frame(32, 32), make_frame(32, 32)];
        let result = c.correct_transforms(&transforms, &frames);
        assert!(result.is_err());
    }

    #[test]
    fn test_correct_transforms_basic() {
        let c = PerRowRsCorrector::new();
        let transforms = vec![
            make_transform(0.0, 0.0, 0.0, 0),
            make_transform(5.0, 0.0, 0.0, 1),
            make_transform(10.0, 0.0, 0.0, 2),
        ];
        let frames = vec![make_frame(32, 32); 3];
        let result = c.correct_transforms(&transforms, &frames);
        assert!(result.is_ok());
        let corrected = result.expect("should succeed in test");
        assert_eq!(corrected.len(), 3);
    }

    #[test]
    fn test_apply_to_frame_no_motion() {
        let c = PerRowRsCorrector::new();
        let data = Array2::from_elem((8, 8), 128u8);
        let corrections: Vec<RowCorrection> = (0..8)
            .map(|row| RowCorrection {
                row,
                phase: row as f64 / 8.0,
                dx: 0.0,
                dy: 0.0,
                dangle: 0.0,
            })
            .collect();
        let result = c.apply_to_frame(&data, &corrections);
        assert!(result.is_ok());
        let corrected = result.expect("should succeed in test");
        assert_eq!(corrected[[4, 4]], 128);
    }

    #[test]
    fn test_apply_to_frame_with_shift() {
        let c = PerRowRsCorrector::with_config(PerRowRsConfig {
            bilinear_interpolation: false,
            velocity_threshold: 0.0,
            ..PerRowRsConfig::default()
        });
        let mut data = Array2::zeros((4, 8));
        for x in 0..8 {
            data[[2, x]] = (x * 30) as u8;
        }
        let corrections: Vec<RowCorrection> = (0..4)
            .map(|row| RowCorrection {
                row,
                phase: row as f64 / 4.0,
                dx: if row == 2 { 1.0 } else { 0.0 },
                dy: 0.0,
                dangle: 0.0,
            })
            .collect();
        let result = c.apply_to_frame(&data, &corrections);
        assert!(result.is_ok());
    }

    #[test]
    fn test_apply_to_frame_bilinear() {
        let config = PerRowRsConfig {
            bilinear_interpolation: true,
            velocity_threshold: 0.0,
            ..PerRowRsConfig::default()
        };
        let c = PerRowRsCorrector::with_config(config);
        let data = Array2::from_elem((4, 8), 100u8);
        let corrections: Vec<RowCorrection> = (0..4)
            .map(|row| RowCorrection {
                row,
                phase: row as f64 / 4.0,
                dx: 0.5,
                dy: 0.0,
                dangle: 0.0,
            })
            .collect();
        let result = c.apply_to_frame(&data, &corrections);
        assert!(result.is_ok());
        let corrected = result.expect("should succeed in test");
        // With uniform value and sub-pixel shift, result should still be ~100
        assert!((corrected[[1, 4]] as f64 - 100.0).abs() < 2.0);
    }

    #[test]
    fn test_apply_to_frame_dimension_mismatch() {
        let c = PerRowRsCorrector::new();
        let data = Array2::zeros((4, 8));
        let corrections: Vec<RowCorrection> = (0..8) // wrong count
            .map(|row| RowCorrection {
                row,
                phase: 0.0,
                dx: 0.0,
                dy: 0.0,
                dangle: 0.0,
            })
            .collect();
        let result = c.apply_to_frame(&data, &corrections);
        assert!(result.is_err());
    }

    #[test]
    fn test_correct_transforms_preserves_identity() {
        let c = PerRowRsCorrector::new();
        let transforms = vec![
            StabilizationTransform::identity(0),
            StabilizationTransform::identity(1),
            StabilizationTransform::identity(2),
        ];
        let frames = vec![make_frame(32, 32); 3];
        let result = c.correct_transforms(&transforms, &frames);
        assert!(result.is_ok());
        let corrected = result.expect("should succeed in test");
        for t in &corrected {
            assert!(t.dx.abs() < 1e-10);
            assert!(t.dy.abs() < 1e-10);
            assert!(t.angle.abs() < 1e-10);
        }
    }

    #[test]
    fn test_per_row_angle_correction() {
        let c = PerRowRsCorrector::new();
        let corrs = c.compute_row_corrections(100, (0.0, 0.0, 0.1), (0.0, 0.0, 0.1));
        // Angle correction should vary across rows
        let top_angle = corrs[0].dangle;
        let bot_angle = corrs[99].dangle;
        assert!((top_angle - bot_angle).abs() > 1e-6);
    }
}
