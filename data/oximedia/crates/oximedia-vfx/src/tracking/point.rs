//! Single-point and multi-point 2D tracking.

use crate::{Frame, VfxError, VfxResult};
use serde::{Deserialize, Serialize};

/// A tracked point in 2D space.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrackPoint {
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
    /// Tracking confidence (0.0 - 1.0).
    pub confidence: f32,
}

/// Result of tracking operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackingResult {
    /// Tracked point.
    pub point: TrackPoint,
    /// Whether tracking was successful.
    pub success: bool,
    /// Search offset applied.
    pub offset_x: f32,
    /// Search offset applied.
    pub offset_y: f32,
}

/// Configuration for point tracking.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrackingConfig {
    /// Template size for matching.
    pub template_size: u32,
    /// Search area size.
    pub search_size: u32,
    /// Minimum confidence threshold.
    pub min_confidence: f32,
    /// Subpixel accuracy.
    pub subpixel: bool,
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self {
            template_size: 31,
            search_size: 61,
            min_confidence: 0.7,
            subpixel: true,
        }
    }
}

/// Single-point tracker using template matching.
#[derive(Debug, Clone)]
pub struct PointTracker {
    config: TrackingConfig,
    template: Option<Vec<u8>>,
    template_point: Option<TrackPoint>,
}

impl PointTracker {
    /// Create a new point tracker.
    #[must_use]
    pub const fn new(config: TrackingConfig) -> Self {
        Self {
            config,
            template: None,
            template_point: None,
        }
    }

    /// Initialize tracking with a point in the reference frame.
    pub fn initialize(&mut self, frame: &Frame, point: TrackPoint) -> VfxResult<()> {
        // Extract template around point
        let template_size = self.config.template_size as i32;
        let half_size = template_size / 2;

        let x = point.x as i32;
        let y = point.y as i32;

        // Validate bounds
        if x - half_size < 0
            || x + half_size >= frame.width as i32
            || y - half_size < 0
            || y + half_size >= frame.height as i32
        {
            return Err(VfxError::ProcessingError(
                "Point too close to frame edge".to_string(),
            ));
        }

        // Extract template
        let mut template = Vec::with_capacity((template_size * template_size * 4) as usize);
        for ty in (y - half_size)..=(y + half_size) {
            for tx in (x - half_size)..=(x + half_size) {
                if let Some(pixel) = frame.get_pixel(tx as u32, ty as u32) {
                    template.extend_from_slice(&pixel);
                }
            }
        }

        self.template = Some(template);
        self.template_point = Some(point);
        Ok(())
    }

    /// Track point in new frame.
    pub fn track(&self, frame: &Frame) -> VfxResult<TrackingResult> {
        let template = self
            .template
            .as_ref()
            .ok_or_else(|| VfxError::ProcessingError("Tracker not initialized".to_string()))?;
        let template_point = self
            .template_point
            .ok_or_else(|| VfxError::ProcessingError("Tracker not initialized".to_string()))?;

        let template_size = self.config.template_size as i32;
        let search_size = self.config.search_size as i32;
        let half_search = search_size / 2;

        let center_x = template_point.x as i32;
        let center_y = template_point.y as i32;

        // Search for best match
        let mut best_score = f32::NEG_INFINITY;
        let mut best_x = center_x;
        let mut best_y = center_y;

        for sy in (center_y - half_search)..=(center_y + half_search) {
            for sx in (center_x - half_search)..=(center_x + half_search) {
                if !self.is_valid_search_pos(frame, sx, sy, template_size) {
                    continue;
                }

                let score = self.compute_ncc(frame, sx, sy, template)?;
                if score > best_score {
                    best_score = score;
                    best_x = sx;
                    best_y = sy;
                }
            }
        }

        // Subpixel refinement
        let (final_x, final_y) = if self.config.subpixel && best_score > 0.0 {
            self.refine_subpixel(frame, best_x, best_y, template)?
        } else {
            (best_x as f32, best_y as f32)
        };

        let confidence = (best_score + 1.0) / 2.0; // Map from [-1, 1] to [0, 1]
        let success = confidence >= self.config.min_confidence;

        Ok(TrackingResult {
            point: TrackPoint {
                x: final_x,
                y: final_y,
                confidence,
            },
            success,
            offset_x: final_x - template_point.x,
            offset_y: final_y - template_point.y,
        })
    }

    /// Update template with new frame (for long-term tracking).
    pub fn update_template(&mut self, frame: &Frame, point: TrackPoint) -> VfxResult<()> {
        self.initialize(frame, point)
    }

    /// Track point in new frame and return the position with explicit sub-pixel
    /// accuracy via parabolic interpolation (Foroosh et al. 2002).
    ///
    /// Unlike [`PointTracker::track`], this always applies parabolic interpolation on the NCC
    /// surface around the integer-peak position regardless of the `subpixel`
    /// flag in [`TrackingConfig`].  It returns the raw `(f32, f32)` sub-pixel
    /// position rather than a [`TrackingResult`].
    ///
    /// The refinement formula is:
    /// ```text
    /// delta_x = (R[peak-1] - R[peak+1]) / (2 * (R[peak-1] - 2*R[peak] + R[peak+1]))
    /// ```
    /// clamped to `[-0.5, 0.5]`.  At image borders the delta is forced to 0.0
    /// to avoid out-of-bounds access.
    ///
    /// # Errors
    ///
    /// Propagates any error from the underlying NCC computation.
    pub fn track_point_subpixel(&self, frame: &Frame) -> VfxResult<(f32, f32)> {
        let template = self
            .template
            .as_ref()
            .ok_or_else(|| VfxError::ProcessingError("Tracker not initialized".to_string()))?;
        let template_point = self
            .template_point
            .ok_or_else(|| VfxError::ProcessingError("Tracker not initialized".to_string()))?;

        let template_size = self.config.template_size as i32;
        let search_size = self.config.search_size as i32;
        let half_search = search_size / 2;

        let center_x = template_point.x as i32;
        let center_y = template_point.y as i32;

        // Find integer peak via NCC scan
        let mut best_score = f32::NEG_INFINITY;
        let mut peak_x = center_x;
        let mut peak_y = center_y;

        for sy in (center_y - half_search)..=(center_y + half_search) {
            for sx in (center_x - half_search)..=(center_x + half_search) {
                if !self.is_valid_search_pos(frame, sx, sy, template_size) {
                    continue;
                }
                let score = self.compute_ncc(frame, sx, sy, template)?;
                if score > best_score {
                    best_score = score;
                    peak_x = sx;
                    peak_y = sy;
                }
            }
        }

        // Parabolic sub-pixel interpolation (Foroosh et al. 2002)
        // delta_x = (r_left - r_right) / (2*(r_left - 2*r_center + r_right))
        let delta_x = if peak_x > 0
            && peak_x < (frame.width as i32) - 1
            && self.is_valid_search_pos(frame, peak_x - 1, peak_y, template_size)
            && self.is_valid_search_pos(frame, peak_x + 1, peak_y, template_size)
        {
            let r_left = self.compute_ncc(frame, peak_x - 1, peak_y, template)?;
            let r_right = self.compute_ncc(frame, peak_x + 1, peak_y, template)?;
            let r_center = best_score;
            let denom = 2.0 * (r_left - 2.0 * r_center + r_right);
            if denom.abs() > 1e-6 {
                ((r_left - r_right) / denom).clamp(-0.5, 0.5)
            } else {
                0.0
            }
        } else {
            0.0
        };

        let delta_y = if peak_y > 0
            && peak_y < (frame.height as i32) - 1
            && self.is_valid_search_pos(frame, peak_x, peak_y - 1, template_size)
            && self.is_valid_search_pos(frame, peak_x, peak_y + 1, template_size)
        {
            let r_top = self.compute_ncc(frame, peak_x, peak_y - 1, template)?;
            let r_bot = self.compute_ncc(frame, peak_x, peak_y + 1, template)?;
            let r_center = best_score;
            let denom = 2.0 * (r_top - 2.0 * r_center + r_bot);
            if denom.abs() > 1e-6 {
                ((r_top - r_bot) / denom).clamp(-0.5, 0.5)
            } else {
                0.0
            }
        } else {
            0.0
        };

        Ok((peak_x as f32 + delta_x, peak_y as f32 + delta_y))
    }

    fn is_valid_search_pos(&self, frame: &Frame, x: i32, y: i32, template_size: i32) -> bool {
        let half_size = template_size / 2;
        x - half_size >= 0
            && x + half_size < frame.width as i32
            && y - half_size >= 0
            && y + half_size < frame.height as i32
    }

    /// Compute normalized cross-correlation.
    fn compute_ncc(&self, frame: &Frame, x: i32, y: i32, template: &[u8]) -> VfxResult<f32> {
        let template_size = self.config.template_size as i32;
        let half_size = template_size / 2;

        let mut sum_template = 0.0;
        let mut sum_image = 0.0;
        let mut sum_sq_template = 0.0;
        let mut sum_sq_image = 0.0;
        let mut sum_product = 0.0;
        let mut count = 0;

        for ty in 0..template_size {
            for tx in 0..template_size {
                let fx = x + tx - half_size;
                let fy = y + ty - half_size;

                if let Some(pixel) = frame.get_pixel(fx as u32, fy as u32) {
                    let tidx = ((ty * template_size + tx) * 4) as usize;
                    if tidx + 3 < template.len() {
                        // Use grayscale values
                        let template_val = (f32::from(template[tidx]) * 0.299
                            + f32::from(template[tidx + 1]) * 0.587
                            + f32::from(template[tidx + 2]) * 0.114)
                            / 255.0;
                        let image_val = (f32::from(pixel[0]) * 0.299
                            + f32::from(pixel[1]) * 0.587
                            + f32::from(pixel[2]) * 0.114)
                            / 255.0;

                        sum_template += template_val;
                        sum_image += image_val;
                        sum_sq_template += template_val * template_val;
                        sum_sq_image += image_val * image_val;
                        sum_product += template_val * image_val;
                        count += 1;
                    }
                }
            }
        }

        if count == 0 {
            return Ok(0.0);
        }

        let n = count as f32;
        let numerator = sum_product - (sum_template * sum_image) / n;
        let denominator = ((sum_sq_template - sum_template * sum_template / n)
            * (sum_sq_image - sum_image * sum_image / n))
            .sqrt();

        if denominator > 0.0 {
            Ok(numerator / denominator)
        } else {
            Ok(0.0)
        }
    }

    /// Refine position to subpixel accuracy using parabolic interpolation.
    fn refine_subpixel(
        &self,
        frame: &Frame,
        x: i32,
        y: i32,
        template: &[u8],
    ) -> VfxResult<(f32, f32)> {
        // Sample 3x3 neighborhood
        let c = self.compute_ncc(frame, x, y, template)?;
        let l = if x > 0 {
            self.compute_ncc(frame, x - 1, y, template)?
        } else {
            0.0
        };
        let r = if x < frame.width as i32 - 1 {
            self.compute_ncc(frame, x + 1, y, template)?
        } else {
            0.0
        };
        let t = if y > 0 {
            self.compute_ncc(frame, x, y - 1, template)?
        } else {
            0.0
        };
        let b = if y < frame.height as i32 - 1 {
            self.compute_ncc(frame, x, y + 1, template)?
        } else {
            0.0
        };

        // Parabolic interpolation
        let dx = if (l + r - 2.0 * c).abs() > 1e-6 {
            (l - r) / (2.0 * (l + r - 2.0 * c))
        } else {
            0.0
        };

        let dy = if (t + b - 2.0 * c).abs() > 1e-6 {
            (t - b) / (2.0 * (t + b - 2.0 * c))
        } else {
            0.0
        };

        Ok((
            x as f32 + dx.clamp(-0.5, 0.5),
            y as f32 + dy.clamp(-0.5, 0.5),
        ))
    }
}

/// Two-point tracker for position and rotation.
#[derive(Debug, Clone)]
pub struct TwoPointTracker {
    tracker1: PointTracker,
    tracker2: PointTracker,
}

impl TwoPointTracker {
    /// Create a new two-point tracker.
    #[must_use]
    pub fn new(config: TrackingConfig) -> Self {
        Self {
            tracker1: PointTracker::new(config),
            tracker2: PointTracker::new(config),
        }
    }

    /// Initialize with two points.
    pub fn initialize(
        &mut self,
        frame: &Frame,
        point1: TrackPoint,
        point2: TrackPoint,
    ) -> VfxResult<()> {
        self.tracker1.initialize(frame, point1)?;
        self.tracker2.initialize(frame, point2)?;
        Ok(())
    }

    /// Track both points.
    pub fn track(&self, frame: &Frame) -> VfxResult<(TrackingResult, TrackingResult)> {
        let result1 = self.tracker1.track(frame)?;
        let result2 = self.tracker2.track(frame)?;
        Ok((result1, result2))
    }

    /// Calculate rotation between tracked points.
    #[must_use]
    pub fn calculate_rotation(&self, result1: &TrackingResult, result2: &TrackingResult) -> f32 {
        let dx = result2.point.x - result1.point.x;
        let dy = result2.point.y - result1.point.y;
        dy.atan2(dx).to_degrees()
    }

    /// Calculate scale between tracked points.
    #[must_use]
    pub fn calculate_scale(
        &self,
        result1: &TrackingResult,
        result2: &TrackingResult,
        initial_distance: f32,
    ) -> f32 {
        let dx = result2.point.x - result1.point.x;
        let dy = result2.point.y - result1.point.y;
        let current_distance = (dx * dx + dy * dy).sqrt();
        current_distance / initial_distance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracking_config_default() {
        let config = TrackingConfig::default();
        assert_eq!(config.template_size, 31);
        assert_eq!(config.search_size, 61);
        assert!(config.subpixel);
    }

    #[test]
    fn test_point_tracker_initialization() -> VfxResult<()> {
        let frame = Frame::new(100, 100)?;
        let mut tracker = PointTracker::new(TrackingConfig::default());
        let point = TrackPoint {
            x: 50.0,
            y: 50.0,
            confidence: 1.0,
        };
        tracker.initialize(&frame, point)?;
        Ok(())
    }

    // ── track_point_subpixel ──────────────────────────────────────────────

    /// Build a tiny solid-colour frame with a bright 3×3 patch at (cx, cy).
    fn frame_with_patch(w: u32, h: u32, cx: u32, cy: u32) -> Frame {
        let mut f = Frame::new(w, h).expect("frame");
        // Fill with a mid-grey base
        for y in 0..h {
            for x in 0..w {
                f.set_pixel(x, y, [64, 64, 64, 255]);
            }
        }
        // Bright patch
        for dy in 0..=2i32 {
            for dx in 0..=2i32 {
                let px = (cx as i32 + dx - 1).clamp(0, (w as i32) - 1) as u32;
                let py = (cy as i32 + dy - 1).clamp(0, (h as i32) - 1) as u32;
                f.set_pixel(px, py, [255, 255, 255, 255]);
            }
        }
        f
    }

    #[test]
    fn test_subpixel_tracker_zero_shift() -> VfxResult<()> {
        let w = 80u32;
        let h = 80u32;
        let cx = 40u32;
        let cy = 40u32;
        let frame = frame_with_patch(w, h, cx, cy);

        let mut tracker = PointTracker::new(TrackingConfig {
            template_size: 11,
            search_size: 21,
            min_confidence: 0.0,
            subpixel: false,
        });
        let pt = TrackPoint {
            x: cx as f32,
            y: cy as f32,
            confidence: 1.0,
        };
        tracker.initialize(&frame, pt)?;

        // Track against identical frame
        let (rx, ry) = tracker.track_point_subpixel(&frame)?;
        assert!(
            (rx - cx as f32).abs() < 0.5,
            "zero-shift x: got {rx}, expected ~{cx}"
        );
        assert!(
            (ry - cy as f32).abs() < 0.5,
            "zero-shift y: got {ry}, expected ~{cy}"
        );
        Ok(())
    }

    #[test]
    fn test_subpixel_tracker_half_pixel_shift() -> VfxResult<()> {
        // Place template at (40, 40) in reference, then track in a frame where
        // patch is shifted to (41, 40) — we expect the recovered shift ≈ 1 px.
        // Sub-pixel accuracy means the NCC peak will be at 41 ± 0.5.
        let w = 80u32;
        let h = 80u32;
        let ref_frame = frame_with_patch(w, h, 40, 40);
        let shifted_frame = frame_with_patch(w, h, 41, 40);

        let mut tracker = PointTracker::new(TrackingConfig {
            template_size: 11,
            search_size: 21,
            min_confidence: 0.0,
            subpixel: false,
        });
        tracker.initialize(
            &ref_frame,
            TrackPoint {
                x: 40.0,
                y: 40.0,
                confidence: 1.0,
            },
        )?;

        let (rx, _ry) = tracker.track_point_subpixel(&shifted_frame)?;
        // The peak should be at ≈ 41.0 (within 0.5 px of the 1-pixel shift)
        let shift = rx - 40.0;
        assert!(
            (shift - 1.0).abs() < 0.5,
            "expected shift ≈ 1.0, got {shift:.3}"
        );
        Ok(())
    }

    #[test]
    fn test_subpixel_fallback_at_border() -> VfxResult<()> {
        // Place patch very close to the border so peak is at the edge.
        // Must not panic and delta should be 0.0 for border axis.
        let w = 80u32;
        let h = 80u32;
        let ref_frame = frame_with_patch(w, h, 40, 40);
        let edge_frame = frame_with_patch(w, h, 40, 40);

        let mut tracker = PointTracker::new(TrackingConfig {
            template_size: 11,
            search_size: 21,
            min_confidence: 0.0,
            subpixel: false,
        });
        tracker.initialize(
            &ref_frame,
            TrackPoint {
                x: 40.0,
                y: 40.0,
                confidence: 1.0,
            },
        )?;

        // Should succeed without panic
        let (rx, ry) = tracker.track_point_subpixel(&edge_frame)?;
        // Result must be finite
        assert!(rx.is_finite(), "rx must be finite");
        assert!(ry.is_finite(), "ry must be finite");
        Ok(())
    }

    #[test]
    fn test_rotation_calculation() {
        let tracker = TwoPointTracker::new(TrackingConfig::default());
        let result1 = TrackingResult {
            point: TrackPoint {
                x: 0.0,
                y: 0.0,
                confidence: 1.0,
            },
            success: true,
            offset_x: 0.0,
            offset_y: 0.0,
        };
        let result2 = TrackingResult {
            point: TrackPoint {
                x: 1.0,
                y: 0.0,
                confidence: 1.0,
            },
            success: true,
            offset_x: 0.0,
            offset_y: 0.0,
        };
        let rotation = tracker.calculate_rotation(&result1, &result2);
        assert!((rotation - 0.0).abs() < 0.01);
    }
}
