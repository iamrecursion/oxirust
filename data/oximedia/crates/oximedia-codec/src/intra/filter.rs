//! Intra edge filter implementations.
//!
//! Intra edge filtering smooths the neighbor samples before prediction
//! to reduce blocking artifacts. The filter strength depends on the
//! prediction angle and block size.
//!
//! # Filter Types
//!
//! - **Weak filter**: 3-tap [1, 2, 1] / 4
//! - **Strong filter**: 5-tap [1, 2, 2, 2, 1] / 8
//! - **Adaptive filter**: Selects based on edge strength
//!
//! # Application
//!
//! Edge filtering is typically applied to:
//! - Directional modes at steep angles
//! - Larger block sizes where artifacts are more visible

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_rem_euclid)]

use super::{BitDepth, BlockDimensions, IntraPredContext, MAX_NEIGHBOR_SAMPLES};

/// Filter strength levels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FilterStrength {
    /// No filtering.
    #[default]
    None,
    /// Weak 3-tap filter.
    Weak,
    /// Strong 5-tap filter.
    Strong,
}

impl FilterStrength {
    /// Determine filter strength based on angle and block size.
    #[must_use]
    pub fn from_angle_and_size(angle: i16, width: usize, height: usize) -> Self {
        // Angles close to 45, 135, 225, or 315 degrees benefit from filtering
        let is_steep = is_steep_angle(angle);

        // Larger blocks need stronger filtering
        let min_dim = width.min(height);

        if !is_steep {
            Self::None
        } else if min_dim >= 16 {
            Self::Strong
        } else if min_dim >= 8 {
            Self::Weak
        } else {
            Self::None
        }
    }
}

/// Check if an angle is steep (close to diagonal).
#[must_use]
fn is_steep_angle(angle: i16) -> bool {
    // Normalize to 0-360
    let angle = ((angle % 360) + 360) % 360;

    // Steep angles are within 22.5 degrees of diagonals (45, 135, 225, 315)
    let diagonals = [45, 135, 225, 315];
    diagonals.iter().any(|&d| (angle - d).abs() < 23)
}

/// Intra edge filter.
#[derive(Clone, Copy, Debug, Default)]
pub struct IntraEdgeFilter {
    /// Filter strength.
    strength: FilterStrength,
    /// Bit depth for clamping.
    bit_depth: BitDepth,
}

impl IntraEdgeFilter {
    /// Create a new intra edge filter.
    #[must_use]
    pub const fn new(strength: FilterStrength, bit_depth: BitDepth) -> Self {
        Self {
            strength,
            bit_depth,
        }
    }

    /// Create with automatic strength selection.
    #[must_use]
    pub fn auto(angle: i16, dims: BlockDimensions, bit_depth: BitDepth) -> Self {
        let strength = FilterStrength::from_angle_and_size(angle, dims.width, dims.height);
        Self {
            strength,
            bit_depth,
        }
    }

    /// Get the filter strength.
    #[must_use]
    pub const fn strength(&self) -> FilterStrength {
        self.strength
    }

    /// Apply filter to top samples.
    pub fn filter_top(&self, samples: &mut [u16], count: usize) {
        match self.strength {
            FilterStrength::None => {}
            FilterStrength::Weak => self.apply_weak_filter(samples, count),
            FilterStrength::Strong => self.apply_strong_filter(samples, count),
        }
    }

    /// Apply filter to left samples.
    pub fn filter_left(&self, samples: &mut [u16], count: usize) {
        // Same filter, different orientation
        match self.strength {
            FilterStrength::None => {}
            FilterStrength::Weak => self.apply_weak_filter(samples, count),
            FilterStrength::Strong => self.apply_strong_filter(samples, count),
        }
    }

    /// Apply weak 3-tap filter [1, 2, 1] / 4.
    fn apply_weak_filter(&self, samples: &mut [u16], count: usize) {
        if count < 3 {
            return;
        }

        let max_val = self.bit_depth.max_value();
        let mut filtered = [0u16; MAX_NEIGHBOR_SAMPLES];

        // First sample unchanged
        filtered[0] = samples[0];

        // Apply 3-tap filter to middle samples
        for i in 1..count.saturating_sub(1) {
            let sum =
                u32::from(samples[i - 1]) + 2 * u32::from(samples[i]) + u32::from(samples[i + 1]);
            let val = (sum + 2) / 4;
            filtered[i] = val.min(u32::from(max_val)) as u16;
        }

        // Last sample unchanged
        if count > 1 {
            filtered[count - 1] = samples[count - 1];
        }

        // Copy back
        samples[..count].copy_from_slice(&filtered[..count]);
    }

    /// Apply strong 5-tap filter [1, 2, 2, 2, 1] / 8.
    fn apply_strong_filter(&self, samples: &mut [u16], count: usize) {
        if count < 5 {
            // Fall back to weak filter for small arrays
            self.apply_weak_filter(samples, count);
            return;
        }

        let max_val = self.bit_depth.max_value();
        let mut filtered = [0u16; MAX_NEIGHBOR_SAMPLES];

        // First two samples get special treatment
        filtered[0] = samples[0];
        if count > 1 {
            let sum = u32::from(samples[0]) + 2 * u32::from(samples[1]) + u32::from(samples[2]);
            filtered[1] = ((sum + 2) / 4).min(u32::from(max_val)) as u16;
        }

        // Apply 5-tap filter to middle samples
        for i in 2..count.saturating_sub(2) {
            let sum = u32::from(samples[i - 2])
                + 2 * u32::from(samples[i - 1])
                + 2 * u32::from(samples[i])
                + 2 * u32::from(samples[i + 1])
                + u32::from(samples[i + 2]);
            let val = (sum + 4) / 8;
            filtered[i] = val.min(u32::from(max_val)) as u16;
        }

        // Last two samples get special treatment
        if count > 2 {
            let i = count - 2;
            let sum =
                u32::from(samples[i - 1]) + 2 * u32::from(samples[i]) + u32::from(samples[i + 1]);
            filtered[i] = ((sum + 2) / 4).min(u32::from(max_val)) as u16;
        }
        if count > 1 {
            filtered[count - 1] = samples[count - 1];
        }

        // Copy back
        samples[..count].copy_from_slice(&filtered[..count]);
    }
}

/// Apply intra filter to prediction context.
pub fn apply_intra_filter(ctx: &mut IntraPredContext, angle: i16, dims: BlockDimensions) {
    let filter = IntraEdgeFilter::auto(angle, dims, ctx.bit_depth());

    if filter.strength() == FilterStrength::None {
        return;
    }

    // Get mutable references to samples and filter them
    let top_count = dims.width + dims.height;
    let left_count = dims.height + dims.width;

    ctx.filter_top_samples(|samples| {
        filter.filter_top(samples, top_count.min(samples.len()));
    });

    ctx.filter_left_samples(|samples| {
        filter.filter_left(samples, left_count.min(samples.len()));
    });
}

/// Recursive intra prediction helper.
///
/// Applies intra prediction using a recursive filter approach
/// that considers previously predicted samples.
pub struct RecursiveIntraHelper {
    bit_depth: BitDepth,
}

impl RecursiveIntraHelper {
    /// Create a new recursive intra helper.
    #[must_use]
    pub const fn new(bit_depth: BitDepth) -> Self {
        Self { bit_depth }
    }

    /// Apply recursive filtering to predicted samples.
    ///
    /// This smooths the prediction by considering previously predicted
    /// samples in the current block.
    pub fn apply_recursive_filter(
        &self,
        output: &mut [u16],
        stride: usize,
        dims: BlockDimensions,
        filter_type: RecursiveFilterType,
    ) {
        match filter_type {
            RecursiveFilterType::None => {}
            RecursiveFilterType::Horizontal => {
                self.filter_horizontal(output, stride, dims);
            }
            RecursiveFilterType::Vertical => {
                self.filter_vertical(output, stride, dims);
            }
            RecursiveFilterType::Both => {
                self.filter_horizontal(output, stride, dims);
                self.filter_vertical(output, stride, dims);
            }
        }
    }

    /// Apply horizontal recursive filter.
    fn filter_horizontal(&self, output: &mut [u16], stride: usize, dims: BlockDimensions) {
        let max_val = self.bit_depth.max_value();

        for y in 0..dims.height {
            let row_start = y * stride;
            for x in 1..dims.width {
                let prev = u32::from(output[row_start + x - 1]);
                let curr = u32::from(output[row_start + x]);
                let filtered = (prev + curr + 1) / 2;
                output[row_start + x] = filtered.min(u32::from(max_val)) as u16;
            }
        }
    }

    /// Apply vertical recursive filter.
    fn filter_vertical(&self, output: &mut [u16], stride: usize, dims: BlockDimensions) {
        let max_val = self.bit_depth.max_value();

        for x in 0..dims.width {
            for y in 1..dims.height {
                let prev = u32::from(output[(y - 1) * stride + x]);
                let curr = u32::from(output[y * stride + x]);
                let filtered = (prev + curr + 1) / 2;
                output[y * stride + x] = filtered.min(u32::from(max_val)) as u16;
            }
        }
    }
}

/// Recursive filter type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RecursiveFilterType {
    /// No recursive filtering.
    #[default]
    None,
    /// Horizontal recursive filter.
    Horizontal,
    /// Vertical recursive filter.
    Vertical,
    /// Both horizontal and vertical.
    Both,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_strength_selection() {
        // Diagonal angle, large block -> strong
        let strength = FilterStrength::from_angle_and_size(45, 16, 16);
        assert_eq!(strength, FilterStrength::Strong);

        // Diagonal angle, medium block -> weak
        let strength = FilterStrength::from_angle_and_size(45, 8, 8);
        assert_eq!(strength, FilterStrength::Weak);

        // Diagonal angle, small block -> none
        let strength = FilterStrength::from_angle_and_size(45, 4, 4);
        assert_eq!(strength, FilterStrength::None);

        // Non-diagonal angle -> none
        let strength = FilterStrength::from_angle_and_size(90, 16, 16);
        assert_eq!(strength, FilterStrength::None);
    }

    #[test]
    fn test_is_steep_angle() {
        assert!(is_steep_angle(45));
        assert!(is_steep_angle(50));
        assert!(is_steep_angle(135));
        assert!(is_steep_angle(315));

        assert!(!is_steep_angle(0));
        assert!(!is_steep_angle(90));
        assert!(!is_steep_angle(180));
        assert!(!is_steep_angle(270));
    }

    #[test]
    fn test_weak_filter() {
        let filter = IntraEdgeFilter::new(FilterStrength::Weak, BitDepth::Bits8);
        let mut samples = [100u16, 150, 200, 150, 100];

        filter.apply_weak_filter(&mut samples, 5);

        // First and last unchanged
        assert_eq!(samples[0], 100);
        assert_eq!(samples[4], 100);

        // Middle samples smoothed
        // samples[1] = (100 + 2*150 + 200 + 2) / 4 = 150
        // samples[2] = (150 + 2*200 + 150 + 2) / 4 = 175
        // samples[3] = (200 + 2*150 + 100 + 2) / 4 = 150
        assert!(samples[1] >= 140 && samples[1] <= 160);
        assert!(samples[2] >= 170 && samples[2] <= 180);
        assert!(samples[3] >= 140 && samples[3] <= 160);
    }

    #[test]
    fn test_strong_filter() {
        let filter = IntraEdgeFilter::new(FilterStrength::Strong, BitDepth::Bits8);
        let mut samples = [100u16, 110, 200, 190, 100, 110, 100];

        filter.apply_strong_filter(&mut samples, 7);

        // First unchanged
        assert_eq!(samples[0], 100);
        // Last unchanged
        assert_eq!(samples[6], 100);

        // Middle samples should be smoothed more than weak filter
        // All values should be reasonable (between 100 and 200)
        for sample in &samples {
            assert!(*sample >= 100 && *sample <= 200);
        }
    }

    #[test]
    fn test_filter_clipping() {
        let filter = IntraEdgeFilter::new(FilterStrength::Weak, BitDepth::Bits8);
        let mut samples = [250u16, 255, 255, 255, 250];

        filter.apply_weak_filter(&mut samples, 5);

        // All values should be <= 255
        for sample in &samples {
            assert!(*sample <= 255);
        }
    }

    #[test]
    fn test_recursive_helper_horizontal() {
        let helper = RecursiveIntraHelper::new(BitDepth::Bits8);
        let mut output = vec![100u16, 200, 100, 200];
        let dims = BlockDimensions::new(4, 1);

        helper.filter_horizontal(&mut output, 4, dims);

        // Each sample averaged with previous
        // [100, 150, 125, 162] approximately
        assert_eq!(output[0], 100);
        assert!(output[1] > 100 && output[1] < 200);
    }

    #[test]
    fn test_recursive_helper_vertical() {
        let helper = RecursiveIntraHelper::new(BitDepth::Bits8);
        let mut output = vec![100u16, 100, 200, 200, 100, 100, 200, 200];
        let dims = BlockDimensions::new(2, 4);

        helper.filter_vertical(&mut output, 2, dims);

        // First row unchanged
        assert_eq!(output[0], 100);
        assert_eq!(output[1], 100);

        // Subsequent rows averaged with previous
        assert!(output[2] > 100 && output[2] < 200);
    }

    #[test]
    fn test_auto_filter_creation() {
        let filter = IntraEdgeFilter::auto(45, BlockDimensions::new(16, 16), BitDepth::Bits8);
        assert_eq!(filter.strength(), FilterStrength::Strong);

        let filter = IntraEdgeFilter::auto(90, BlockDimensions::new(16, 16), BitDepth::Bits8);
        assert_eq!(filter.strength(), FilterStrength::None);
    }
}
