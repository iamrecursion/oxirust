//! Field analysis for interlaced video.
//!
//! This module provides utilities for analyzing individual fields in interlaced
//! content, including field separation, comparison, and field order detection.

use crate::error::{CvError, CvResult};
use oximedia_codec::VideoFrame;

use super::metrics::FieldOrder;

/// Field analyzer for interlaced video.
pub struct FieldAnalyzer {
    /// Threshold for field difference detection.
    diff_threshold: u8,
    /// Number of sample points for field order detection.
    sample_points: usize,
}

impl FieldAnalyzer {
    /// Creates a new field analyzer with default settings.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            diff_threshold: 10,
            sample_points: 100,
        }
    }

    /// Creates a new field analyzer with custom settings.
    #[must_use]
    pub const fn with_config(diff_threshold: u8, sample_points: usize) -> Self {
        Self {
            diff_threshold,
            sample_points,
        }
    }

    /// Separates a frame into its two fields.
    ///
    /// Returns (top_field, bottom_field) where each field contains only the
    /// even or odd lines from the original frame.
    pub fn separate_fields(&self, frame: &VideoFrame) -> CvResult<(Field, Field)> {
        if frame.planes.is_empty() {
            return Err(CvError::insufficient_data(1, 0));
        }

        let width = frame.width as usize;
        let height = frame.height as usize;

        if height < 2 {
            return Err(CvError::invalid_dimensions(frame.width, frame.height));
        }

        let field_height = height / 2;

        // Extract top field (even lines: 0, 2, 4, ...)
        let mut top_field_data = Vec::with_capacity(width * field_height);
        for y in (0..height).step_by(2) {
            let row = frame.planes[0].row(y);
            if row.len() >= width {
                top_field_data.extend_from_slice(&row[..width]);
            }
        }

        // Extract bottom field (odd lines: 1, 3, 5, ...)
        let mut bottom_field_data = Vec::with_capacity(width * field_height);
        for y in (1..height).step_by(2) {
            let row = frame.planes[0].row(y);
            if row.len() >= width {
                bottom_field_data.extend_from_slice(&row[..width]);
            }
        }

        let top_field = Field {
            data: top_field_data,
            width,
            height: field_height,
            is_top: true,
        };

        let bottom_field = Field {
            data: bottom_field_data,
            width,
            height: field_height,
            is_top: false,
        };

        Ok((top_field, bottom_field))
    }

    /// Detects the field order of interlaced content.
    ///
    /// Analyzes motion between fields to determine whether top field first (TFF)
    /// or bottom field first (BFF) ordering is being used.
    pub fn detect_field_order(&self, frames: &[VideoFrame]) -> CvResult<FieldOrder> {
        if frames.len() < 2 {
            return Ok(FieldOrder::Unknown);
        }

        let mut tff_score = 0;
        let mut bff_score = 0;

        // Analyze multiple frame pairs
        for i in 0..frames.len() - 1 {
            let curr = &frames[i];
            let next = &frames[i + 1];

            if curr.width != next.width || curr.height != next.height {
                continue;
            }

            let order = self.detect_field_order_pair(curr, next)?;
            match order {
                FieldOrder::TopFieldFirst => tff_score += 1,
                FieldOrder::BottomFieldFirst => bff_score += 1,
                FieldOrder::Unknown => {}
            }
        }

        Ok(match tff_score.cmp(&bff_score) {
            std::cmp::Ordering::Greater => FieldOrder::TopFieldFirst,
            std::cmp::Ordering::Less => FieldOrder::BottomFieldFirst,
            std::cmp::Ordering::Equal => FieldOrder::Unknown,
        })
    }

    /// Detects field order from a pair of consecutive frames.
    fn detect_field_order_pair(
        &self,
        curr: &VideoFrame,
        next: &VideoFrame,
    ) -> CvResult<FieldOrder> {
        if curr.planes.is_empty() || next.planes.is_empty() {
            return Ok(FieldOrder::Unknown);
        }

        let width = curr.width as usize;
        let height = curr.height as usize;

        if height < 4 {
            return Ok(FieldOrder::Unknown);
        }

        let curr_plane = &curr.planes[0];
        let next_plane = &next.planes[0];

        // Sample motion at various points
        let step_x = width / 10;
        let step_y = height / 10;

        let mut tff_motion = 0.0;
        let mut bff_motion = 0.0;

        for y in (2..height - 2).step_by(step_y.max(2)) {
            for x in (2..width - 2).step_by(step_x.max(2)) {
                // Compare top field temporal consistency
                let curr_top = curr_plane.row(y);
                let next_top = next_plane.row(y - 1);

                if curr_top.len() > x && next_top.len() > x {
                    let diff_top = (i32::from(curr_top[x]) - i32::from(next_top[x])).abs();
                    tff_motion += f64::from(diff_top);
                }

                // Compare bottom field temporal consistency
                let curr_bot = curr_plane.row(y);
                let next_bot = next_plane.row(y + 1);

                if curr_bot.len() > x && next_bot.len() > x {
                    let diff_bot = (i32::from(curr_bot[x]) - i32::from(next_bot[x])).abs();
                    bff_motion += f64::from(diff_bot);
                }
            }
        }

        // Lower motion indicates better temporal alignment
        // TFF: current top field aligns with next top field
        // BFF: current bottom field aligns with next bottom field
        if tff_motion < bff_motion * 0.9 {
            Ok(FieldOrder::TopFieldFirst)
        } else if bff_motion < tff_motion * 0.9 {
            Ok(FieldOrder::BottomFieldFirst)
        } else {
            Ok(FieldOrder::Unknown)
        }
    }

    /// Calculates the difference between two fields.
    pub fn calculate_field_difference(&self, field1: &Field, field2: &Field) -> CvResult<f64> {
        if field1.width != field2.width || field1.height != field2.height {
            return Err(CvError::invalid_dimensions(
                field1.width as u32,
                field1.height as u32,
            ));
        }

        if field1.data.len() != field2.data.len() {
            return Err(CvError::insufficient_data(
                field1.data.len(),
                field2.data.len(),
            ));
        }

        let mut diff_sum = 0i64;
        let pixel_count = field1.data.len();

        for i in 0..pixel_count {
            let diff = (i32::from(field1.data[i]) - i32::from(field2.data[i])).abs();
            diff_sum += i64::from(diff);
        }

        // Normalize to 0.0-1.0 range
        let avg_diff = diff_sum as f64 / pixel_count as f64;
        Ok(avg_diff / 255.0)
    }

    /// Calculates the motion between fields.
    ///
    /// Higher values indicate more motion between fields.
    pub fn calculate_field_motion(&self, field1: &Field, field2: &Field) -> CvResult<f64> {
        let diff = self.calculate_field_difference(field1, field2)?;

        // Motion is simply the difference, but we could apply additional processing
        Ok(diff)
    }

    /// Detects if fields have different motion characteristics.
    ///
    /// This is useful for detecting mixed progressive/interlaced content.
    pub fn detect_field_motion_mismatch(
        &self,
        top_field_curr: &Field,
        top_field_prev: &Field,
        bottom_field_curr: &Field,
        bottom_field_prev: &Field,
    ) -> CvResult<bool> {
        let top_motion = self.calculate_field_motion(top_field_curr, top_field_prev)?;
        let bottom_motion = self.calculate_field_motion(bottom_field_curr, bottom_field_prev)?;

        // If one field has significantly more motion than the other, it's a mismatch
        let ratio = if top_motion > bottom_motion && bottom_motion > 0.0 {
            top_motion / bottom_motion
        } else if bottom_motion > top_motion && top_motion > 0.0 {
            bottom_motion / top_motion
        } else {
            1.0
        };

        Ok(ratio > 2.0)
    }

    /// Reconstructs a frame from two fields using simple line doubling.
    pub fn reconstruct_from_fields(
        &self,
        top_field: &Field,
        bottom_field: &Field,
    ) -> CvResult<Vec<u8>> {
        if top_field.width != bottom_field.width {
            return Err(CvError::invalid_dimensions(
                top_field.width as u32,
                bottom_field.width as u32,
            ));
        }

        let width = top_field.width;
        let height = top_field.height + bottom_field.height;
        let mut frame_data = vec![0u8; width * height];

        // Interleave the fields
        for y in 0..top_field.height {
            let top_offset = y * width;
            let frame_offset = (y * 2) * width;

            if top_offset + width <= top_field.data.len()
                && frame_offset + width <= frame_data.len()
            {
                frame_data[frame_offset..frame_offset + width]
                    .copy_from_slice(&top_field.data[top_offset..top_offset + width]);
            }
        }

        for y in 0..bottom_field.height {
            let bottom_offset = y * width;
            let frame_offset = (y * 2 + 1) * width;

            if bottom_offset + width <= bottom_field.data.len()
                && frame_offset + width <= frame_data.len()
            {
                frame_data[frame_offset..frame_offset + width]
                    .copy_from_slice(&bottom_field.data[bottom_offset..bottom_offset + width]);
            }
        }

        Ok(frame_data)
    }

    /// Reconstructs a frame from two fields using interpolation.
    ///
    /// This provides better quality than simple line doubling by interpolating
    /// between fields.
    pub fn reconstruct_interpolated(
        &self,
        top_field: &Field,
        bottom_field: &Field,
    ) -> CvResult<Vec<u8>> {
        if top_field.width != bottom_field.width {
            return Err(CvError::invalid_dimensions(
                top_field.width as u32,
                bottom_field.width as u32,
            ));
        }

        let width = top_field.width;
        let height = (top_field.height + bottom_field.height).max(2);
        let mut frame_data = vec![0u8; width * height];

        // First line from top field
        if width <= top_field.data.len() {
            frame_data[..width].copy_from_slice(&top_field.data[..width]);
        }

        // Interleave and interpolate
        for y in 0..top_field.height.min(bottom_field.height) {
            let top_offset = y * width;
            let bottom_offset = y * width;

            // Top field line (even)
            let even_line = (y * 2) * width;
            if top_offset + width <= top_field.data.len() && even_line + width <= frame_data.len() {
                frame_data[even_line..even_line + width]
                    .copy_from_slice(&top_field.data[top_offset..top_offset + width]);
            }

            // Interpolated line (odd)
            let odd_line = (y * 2 + 1) * width;
            if y + 1 < top_field.height
                && bottom_offset + width <= bottom_field.data.len()
                && top_offset + width * 2 <= top_field.data.len()
                && odd_line + width <= frame_data.len()
            {
                for x in 0..width {
                    let top_curr = u16::from(top_field.data[top_offset + x]);
                    let top_next = u16::from(
                        top_field
                            .data
                            .get(top_offset + width + x)
                            .copied()
                            .unwrap_or(top_field.data[top_offset + x]),
                    );
                    let bottom_curr = u16::from(bottom_field.data[bottom_offset + x]);

                    // Weighted average for interpolation
                    let interpolated = ((top_curr + top_next + bottom_curr * 2) / 4) as u8;
                    frame_data[odd_line + x] = interpolated;
                }
            }
        }

        Ok(frame_data)
    }

    /// Analyzes field parity (whether fields are from the same frame or different frames).
    pub fn analyze_field_parity(&self, frames: &[VideoFrame]) -> CvResult<Vec<FieldParity>> {
        if frames.len() < 2 {
            return Ok(Vec::new());
        }

        let mut parities = Vec::with_capacity(frames.len() - 1);

        for i in 0..frames.len() - 1 {
            let (top1, bottom1) = self.separate_fields(&frames[i])?;
            let (top2, bottom2) = self.separate_fields(&frames[i + 1])?;

            // Compare bottom of frame i with top of frame i+1
            let inter_frame_diff = self.calculate_field_difference(&bottom1, &top2)?;

            // Compare fields within same frame
            let intra_frame_diff = self.calculate_field_difference(&top1, &bottom1)?;

            let parity = if inter_frame_diff < intra_frame_diff * 0.8 {
                FieldParity::Different
            } else {
                FieldParity::Same
            };

            parities.push(parity);
        }

        Ok(parities)
    }
}

impl Default for FieldAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents a single field from an interlaced frame.
#[derive(Debug, Clone)]
pub struct Field {
    /// Pixel data (luma only).
    pub data: Vec<u8>,
    /// Field width in pixels.
    pub width: usize,
    /// Field height in lines.
    pub height: usize,
    /// True if this is the top field, false for bottom.
    pub is_top: bool,
}

impl Field {
    /// Gets a row from the field.
    #[must_use]
    pub fn row(&self, y: usize) -> &[u8] {
        let start = y * self.width;
        let end = start + self.width;
        if end <= self.data.len() {
            &self.data[start..end]
        } else {
            &[]
        }
    }

    /// Calculates the average pixel value of the field.
    #[must_use]
    pub fn average_value(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }

        let sum: u64 = self.data.iter().map(|&x| u64::from(x)).sum();
        sum as f64 / self.data.len() as f64
    }

    /// Calculates the variance of pixel values in the field.
    #[must_use]
    pub fn variance(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }

        let avg = self.average_value();
        let sum_sq: f64 = self
            .data
            .iter()
            .map(|&x| {
                let diff = f64::from(x) - avg;
                diff * diff
            })
            .sum();

        sum_sq / self.data.len() as f64
    }
}

/// Field parity relationship between frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldParity {
    /// Fields are from the same temporal moment.
    Same,
    /// Fields are from different temporal moments (typical in telecine).
    Different,
}

impl FieldParity {
    /// Returns true if fields are from the same temporal moment.
    #[must_use]
    pub const fn is_same(&self) -> bool {
        matches!(self, Self::Same)
    }

    /// Returns true if fields are from different temporal moments.
    #[must_use]
    pub const fn is_different(&self) -> bool {
        matches!(self, Self::Different)
    }
}
