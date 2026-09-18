//! Scratch and artifact repair: detection, inpainting approximation, and temporal consistency.

#![allow(dead_code)]

/// A scratch or blemish region in a video frame.
#[derive(Debug, Clone, PartialEq)]
pub struct ScratchRegion {
    /// X coordinate of the top-left corner.
    pub x: u32,
    /// Y coordinate of the top-left corner.
    pub y: u32,
    /// Width of the scratch in pixels.
    pub width: u32,
    /// Height of the scratch in pixels.
    pub height: u32,
    /// Confidence score that this is a scratch (0.0–1.0).
    pub confidence: f32,
}

impl ScratchRegion {
    /// Create a new scratch region.
    pub fn new(x: u32, y: u32, width: u32, height: u32, confidence: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Area of the scratch region in pixels.
    pub fn area(&self) -> u32 {
        self.width * self.height
    }

    /// Whether the scratch is sufficiently likely to act on.
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}

/// Sensitivity settings for scratch detection.
#[derive(Debug, Clone)]
pub struct DetectionParams {
    /// Minimum brightness delta to flag as a scratch.
    pub brightness_threshold: f32,
    /// Minimum run-length in pixels to be considered a scratch.
    pub min_run_length: u32,
    /// Maximum width of scratches to detect.
    pub max_scratch_width: u32,
    /// Confidence threshold for reporting.
    pub confidence_threshold: f32,
}

impl Default for DetectionParams {
    fn default() -> Self {
        Self {
            brightness_threshold: 0.6,
            min_run_length: 4,
            max_scratch_width: 8,
            confidence_threshold: 0.5,
        }
    }
}

/// Detect scratch regions in a single-channel (luma) frame.
///
/// `pixels` is a row-major flat buffer of normalised luma values (0.0–1.0).
/// `width` and `height` describe the frame dimensions.
pub fn detect_scratches(
    pixels: &[f32],
    width: usize,
    height: usize,
    params: &DetectionParams,
) -> Vec<ScratchRegion> {
    let mut regions = Vec::new();
    if pixels.len() != width * height || width == 0 || height == 0 {
        return regions;
    }

    for col in 0..width {
        let mut run_start = None;
        let mut run_len = 0u32;

        for row in 0..height {
            let val = pixels[row * width + col];
            if val >= params.brightness_threshold {
                if run_start.is_none() {
                    run_start = Some(row);
                    run_len = 0;
                }
                run_len += 1;
            } else {
                if run_len >= params.min_run_length {
                    if let Some(rs) = run_start {
                        #[allow(clippy::cast_precision_loss)]
                        let confidence =
                            (run_len as f32 / params.min_run_length as f32).min(1.0) * 0.8;
                        if confidence >= params.confidence_threshold {
                            regions.push(ScratchRegion::new(
                                col as u32, rs as u32, 1, run_len, confidence,
                            ));
                        }
                    }
                }
                run_start = None;
                run_len = 0;
            }
        }
        // Handle run that reaches bottom edge
        if run_len >= params.min_run_length {
            if let Some(rs) = run_start {
                #[allow(clippy::cast_precision_loss)]
                let confidence = (run_len as f32 / params.min_run_length as f32).min(1.0) * 0.8;
                if confidence >= params.confidence_threshold {
                    regions.push(ScratchRegion::new(
                        col as u32, rs as u32, 1, run_len, confidence,
                    ));
                }
            }
        }
    }
    regions
}

/// Inpainting approximation: fill a detected scratch region using
/// column-neighbour averaging.
///
/// For each pixel in the scratch region, the repaired value is the average
/// of the nearest non-scratch pixels to the left and right in the same row.
pub fn inpaint_scratch(pixels: &mut [f32], width: usize, height: usize, region: &ScratchRegion) {
    let x = region.x as usize;
    let y_start = region.y as usize;
    let y_end = (region.y + region.height) as usize;

    for row in y_start..y_end.min(height) {
        // Find left neighbour
        let left_val = if x > 0 {
            pixels[row * width + x - 1]
        } else {
            0.0
        };
        // Find right neighbour (skip the scratch width)
        let right_x = x + region.width as usize;
        let right_val = if right_x < width {
            pixels[row * width + right_x]
        } else {
            0.0
        };

        let fill = (left_val + right_val) / 2.0;

        for col in x..(x + region.width as usize).min(width) {
            pixels[row * width + col] = fill;
        }
    }
}

/// Check temporal consistency of a pixel across a window of frames.
///
/// Returns `true` if the pixel value in `current_frame` is consistent with
/// the values in `reference_frames` (all within `tolerance`).
pub fn is_temporally_consistent(
    current_frame: &[f32],
    reference_frames: &[&[f32]],
    pixel_idx: usize,
    tolerance: f32,
) -> bool {
    if pixel_idx >= current_frame.len() {
        return false;
    }
    let current = current_frame[pixel_idx];
    reference_frames.iter().all(|frame| {
        if pixel_idx >= frame.len() {
            return false;
        }
        (frame[pixel_idx] - current).abs() <= tolerance
    })
}

/// Restore temporal consistency for pixels in a scratch region by blending
/// the repaired frame with reference frames.
pub fn blend_temporal(
    repaired: &mut [f32],
    reference: &[f32],
    width: usize,
    height: usize,
    region: &ScratchRegion,
    blend_alpha: f32,
) {
    let alpha = blend_alpha.clamp(0.0, 1.0);
    let x = region.x as usize;
    let y_start = region.y as usize;
    let y_end = (region.y + region.height) as usize;

    for row in y_start..y_end.min(height) {
        for col in x..(x + region.width as usize).min(width) {
            let idx = row * width + col;
            if idx < repaired.len() && idx < reference.len() {
                repaired[idx] = alpha * repaired[idx] + (1.0 - alpha) * reference[idx];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// New types: ScratchType, ScratchDetection, detect_vertical_scratch,
// repair_vertical_scratch, ScratchRepairer
// ---------------------------------------------------------------------------

/// The orientation or nature of a film scratch.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScratchType {
    /// A scratch running along a column (top-to-bottom).
    Vertical,
    /// A scratch running along a row (left-to-right).
    Horizontal,
    /// A scratch running at an angle across the frame.
    Diagonal,
    /// Random noise speckling rather than a linear scratch.
    RandomNoise,
}

impl ScratchType {
    /// Returns `true` if the scratch is linear (vertical, horizontal, or diagonal).
    #[must_use]
    pub fn is_linear(&self) -> bool {
        matches!(self, Self::Vertical | Self::Horizontal | Self::Diagonal)
    }
}

/// Information about a single detected scratch on a video frame.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ScratchDetection {
    /// Index of the frame in which the scratch was found.
    pub frame_idx: u64,
    /// Column index (for vertical scratches) or row index (for horizontal).
    pub column_or_row: u32,
    /// Length of the scratch in pixels.
    pub length_px: u32,
    /// Type of scratch.
    pub scratch_type: ScratchType,
    /// Detection confidence in [0.0, 1.0].
    pub confidence: f32,
}

impl ScratchDetection {
    /// Returns `true` if the scratch spans the entire frame height (vertical
    /// scratch) or frame width (horizontal scratch).
    #[must_use]
    pub fn spans_frame(&self, frame_height: u32) -> bool {
        self.length_px >= frame_height
    }
}

/// Detect vertical scratches in a raw `u8` luma frame buffer.
///
/// The function compares each column's mean brightness to the average of
/// adjacent columns; columns that deviate by more than `threshold` are
/// flagged as scratches.
///
/// `frame` is a row-major `u8` buffer (`width * height` bytes).
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn detect_vertical_scratch(
    frame: &[u8],
    width: usize,
    height: usize,
    threshold: u8,
) -> Vec<ScratchDetection> {
    let mut results = Vec::new();
    if frame.len() != width * height || width < 3 || height == 0 {
        return results;
    }

    // Compute per-column mean brightness.
    let col_mean: Vec<f32> = (0..width)
        .map(|col| {
            let sum: u64 = (0..height).map(|row| frame[row * width + col] as u64).sum();
            sum as f32 / height as f32
        })
        .collect();

    for col in 1..width - 1 {
        let neighbour_avg = (col_mean[col - 1] + col_mean[col + 1]) / 2.0;
        let deviation = (col_mean[col] - neighbour_avg).abs();
        if deviation >= threshold as f32 {
            let confidence = (deviation / 255.0).min(1.0);
            results.push(ScratchDetection {
                frame_idx: 0,
                column_or_row: col as u32,
                length_px: height as u32,
                scratch_type: ScratchType::Vertical,
                confidence,
            });
        }
    }
    results
}

/// Repair a vertical scratch at `column` in a raw `u8` luma frame buffer by
/// replacing each affected pixel with the weighted average of neighbouring
/// columns within `radius` pixels.
///
/// `frame` must be a row-major `u8` buffer (`width * height` bytes).
#[allow(clippy::cast_precision_loss)]
pub fn repair_vertical_scratch(
    frame: &mut [u8],
    width: usize,
    height: usize,
    column: u32,
    radius: u32,
) {
    let col = column as usize;
    if frame.len() != width * height || col >= width {
        return;
    }
    let r = radius as usize;
    for row in 0..height {
        let mut sum = 0.0f32;
        let mut count = 0u32;
        for offset in 1..=r {
            if col >= offset {
                sum += frame[row * width + col - offset] as f32;
                count += 1;
            }
            if col + offset < width {
                sum += frame[row * width + col + offset] as f32;
                count += 1;
            }
        }
        if count > 0 {
            frame[row * width + col] = (sum / count as f32).round() as u8;
        }
    }
}

/// A configurable scratch detector and repairer for `u8` luma frames.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ScratchRepairer {
    /// Brightness deviation threshold for detection.
    pub threshold: u8,
    /// Number of neighbouring columns to average during repair.
    pub repair_radius: u32,
}

impl ScratchRepairer {
    /// Create a new `ScratchRepairer`.
    #[must_use]
    pub fn new(threshold: u8, repair_radius: u32) -> Self {
        Self {
            threshold,
            repair_radius,
        }
    }

    /// Detect and repair scratches in-place, returning the list of detections.
    pub fn process_frame(
        &self,
        frame: &mut [u8],
        width: usize,
        height: usize,
    ) -> Vec<ScratchDetection> {
        let detections = detect_vertical_scratch(frame, width, height, self.threshold);
        for d in &detections {
            repair_vertical_scratch(frame, width, height, d.column_or_row, self.repair_radius);
        }
        detections
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scratch_region_area() {
        let r = ScratchRegion::new(0, 0, 3, 10, 0.9);
        assert_eq!(r.area(), 30);
    }

    #[test]
    fn test_scratch_region_confidence_clamped() {
        let r = ScratchRegion::new(0, 0, 1, 1, 1.5);
        assert_eq!(r.confidence, 1.0);
        let r2 = ScratchRegion::new(0, 0, 1, 1, -0.5);
        assert_eq!(r2.confidence, 0.0);
    }

    #[test]
    fn test_scratch_region_is_confident() {
        let r = ScratchRegion::new(0, 0, 1, 1, 0.8);
        assert!(r.is_confident(0.7));
        assert!(!r.is_confident(0.9));
    }

    #[test]
    fn test_detect_scratches_empty_pixels() {
        let params = DetectionParams::default();
        let result = detect_scratches(&[], 0, 0, &params);
        assert!(result.is_empty());
    }

    #[test]
    fn test_detect_scratches_mismatched_size() {
        let params = DetectionParams::default();
        let pixels = vec![1.0f32; 10];
        let result = detect_scratches(&pixels, 5, 3, &params); // 5*3 = 15, not 10
        assert!(result.is_empty());
    }

    #[test]
    fn test_detect_scratches_no_scratches() {
        let params = DetectionParams::default();
        // All pixels at 0.0 — no scratches
        let pixels = vec![0.0f32; 100];
        let result = detect_scratches(&pixels, 10, 10, &params);
        assert!(result.is_empty());
    }

    #[test]
    fn test_detect_scratches_finds_bright_column_run() {
        let mut pixels = vec![0.0f32; 10 * 20]; // 10 wide, 20 tall
                                                // Column 3 has a bright run of 8 pixels starting at row 5
        for row in 5..13 {
            pixels[row * 10 + 3] = 0.9;
        }
        let params = DetectionParams {
            brightness_threshold: 0.7,
            min_run_length: 4,
            confidence_threshold: 0.5,
            ..Default::default()
        };
        let result = detect_scratches(&pixels, 10, 20, &params);
        assert!(!result.is_empty());
        assert_eq!(result[0].x, 3);
    }

    #[test]
    fn test_inpaint_scratch_fills_region() {
        let width = 10;
        let height = 5;
        let mut pixels = vec![0.5f32; width * height];
        // Set scratch column 5 to 1.0
        for row in 0..height {
            pixels[row * width + 5] = 1.0;
        }
        let region = ScratchRegion::new(5, 0, 1, height as u32, 0.9);
        inpaint_scratch(&mut pixels, width, height, &region);
        // After inpainting, column 5 should be averaged with neighbours (0.5 each)
        for row in 0..height {
            assert!((pixels[row * width + 5] - 0.5).abs() < 0.01);
        }
    }

    #[test]
    fn test_is_temporally_consistent_true() {
        let current = vec![0.5f32; 10];
        let ref1 = vec![0.5f32; 10];
        let ref2 = vec![0.52f32; 10];
        assert!(is_temporally_consistent(&current, &[&ref1, &ref2], 3, 0.1));
    }

    #[test]
    fn test_is_temporally_consistent_false() {
        let current = vec![0.5f32; 10];
        let ref1 = vec![0.9f32; 10]; // far from 0.5
        assert!(!is_temporally_consistent(&current, &[&ref1], 3, 0.1));
    }

    #[test]
    fn test_is_temporally_consistent_out_of_bounds() {
        let current = vec![0.5f32; 5];
        let ref1 = vec![0.5f32; 5];
        assert!(!is_temporally_consistent(&current, &[&ref1], 10, 0.1));
    }

    #[test]
    fn test_blend_temporal_blends() {
        let width = 4;
        let height = 4;
        let mut repaired = vec![1.0f32; width * height];
        let reference = vec![0.0f32; width * height];
        let region = ScratchRegion::new(1, 1, 2, 2, 0.9);
        blend_temporal(&mut repaired, &reference, width, height, &region, 0.5);
        // Blended = 0.5*1.0 + 0.5*0.0 = 0.5
        assert!((repaired[1 * width + 1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_detection_params_default() {
        let p = DetectionParams::default();
        assert!(p.brightness_threshold > 0.0);
        assert!(p.min_run_length > 0);
    }

    // --- ScratchType tests ---

    #[test]
    fn test_scratch_type_is_linear_vertical() {
        assert!(ScratchType::Vertical.is_linear());
    }

    #[test]
    fn test_scratch_type_is_linear_horizontal() {
        assert!(ScratchType::Horizontal.is_linear());
    }

    #[test]
    fn test_scratch_type_is_linear_diagonal() {
        assert!(ScratchType::Diagonal.is_linear());
    }

    #[test]
    fn test_scratch_type_is_linear_noise() {
        assert!(!ScratchType::RandomNoise.is_linear());
    }

    // --- ScratchDetection::spans_frame ---

    #[test]
    fn test_scratch_detection_spans_frame_true() {
        let d = ScratchDetection {
            frame_idx: 0,
            column_or_row: 5,
            length_px: 480,
            scratch_type: ScratchType::Vertical,
            confidence: 0.9,
        };
        assert!(d.spans_frame(480));
    }

    #[test]
    fn test_scratch_detection_spans_frame_false() {
        let d = ScratchDetection {
            frame_idx: 0,
            column_or_row: 5,
            length_px: 100,
            scratch_type: ScratchType::Vertical,
            confidence: 0.7,
        };
        assert!(!d.spans_frame(480));
    }

    // --- detect_vertical_scratch ---

    #[test]
    fn test_detect_vertical_scratch_empty_returns_empty() {
        let result = detect_vertical_scratch(&[], 0, 0, 20);
        assert!(result.is_empty());
    }

    #[test]
    fn test_detect_vertical_scratch_too_narrow_returns_empty() {
        let frame = vec![128u8; 2 * 10]; // width = 2, needs >= 3
        let result = detect_vertical_scratch(&frame, 2, 10, 10);
        assert!(result.is_empty());
    }

    #[test]
    fn test_detect_vertical_scratch_uniform_frame_no_scratches() {
        let frame = vec![100u8; 8 * 6]; // uniform frame
        let result = detect_vertical_scratch(&frame, 8, 6, 20);
        assert!(result.is_empty());
    }

    #[test]
    fn test_detect_vertical_scratch_finds_bright_column() {
        let width = 8;
        let height = 6;
        let mut frame = vec![100u8; width * height];
        // Column 4 is much brighter than neighbours (200 vs 100).
        for row in 0..height {
            frame[row * width + 4] = 200;
        }
        let result = detect_vertical_scratch(&frame, width, height, 30);
        assert!(!result.is_empty());
        // Column 4 should be among the detected scratches
        assert!(result.iter().any(|d| d.column_or_row == 4));
        assert!(result
            .iter()
            .all(|d| d.scratch_type == ScratchType::Vertical));
    }

    // --- repair_vertical_scratch ---

    #[test]
    fn test_repair_vertical_scratch_replaces_column() {
        let width = 5;
        let height = 4;
        let mut frame = vec![100u8; width * height];
        // Column 2 has value 200.
        for row in 0..height {
            frame[row * width + 2] = 200;
        }
        repair_vertical_scratch(&mut frame, width, height, 2, 1);
        // After repair, column 2 should be averaged with neighbours (100 each).
        for row in 0..height {
            assert_eq!(frame[row * width + 2], 100);
        }
    }

    #[test]
    fn test_repair_vertical_scratch_out_of_bounds_no_panic() {
        let mut frame = vec![128u8; 4 * 4];
        repair_vertical_scratch(&mut frame, 4, 4, 99, 1); // column 99 out of range
                                                          // Just verify no panic occurred.
    }

    // --- ScratchRepairer ---

    #[test]
    fn test_scratch_repairer_process_frame_detects_and_repairs() {
        let width = 8;
        let height = 6;
        let mut frame = vec![100u8; width * height];
        // Column 4 is much brighter than neighbours.
        for row in 0..height {
            frame[row * width + 4] = 200;
        }
        let repairer = ScratchRepairer::new(30, 1);
        let detections = repairer.process_frame(&mut frame, width, height);
        assert!(!detections.is_empty());
        // Column 4 should now be closer to 100.
        for row in 0..height {
            assert!(frame[row * width + 4] <= 150);
        }
    }

    #[test]
    fn test_scratch_repairer_new_stores_params() {
        let r = ScratchRepairer::new(25, 2);
        assert_eq!(r.threshold, 25);
        assert_eq!(r.repair_radius, 2);
    }
}
