//! Intra prediction context.
//!
//! The prediction context manages neighbor samples and availability
//! information needed for intra prediction. It provides a unified
//! interface for accessing top, left, and top-left samples.
//!
//! # Sample Layout
//!
//! For a block at position (x, y):
//! ```text
//!     TL | T0 T1 T2 T3 ...
//!     ---+----------------
//!     L0 | P00 P01 P02 P03
//!     L1 | P10 P11 P12 P13
//!     L2 | P20 P21 P22 P23
//!     L3 | P30 P31 P32 P33
//! ```
//!
//! Where:
//! - TL = Top-Left sample
//! - T0..Tn = Top samples
//! - L0..Ln = Left samples
//! - Pxy = Predicted samples

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]

use super::{BitDepth, MAX_NEIGHBOR_SAMPLES};

/// Top samples array type.
pub type TopSamples = [u16; MAX_NEIGHBOR_SAMPLES];

/// Left samples array type.
pub type LeftSamples = [u16; MAX_NEIGHBOR_SAMPLES];

/// Neighbor availability flags.
#[derive(Clone, Copy, Debug, Default)]
pub struct NeighborAvailability {
    /// Top neighbor is available.
    pub top: bool,
    /// Left neighbor is available.
    pub left: bool,
    /// Top-left neighbor is available.
    pub top_left: bool,
    /// Top-right neighbor is available.
    pub top_right: bool,
    /// Bottom-left neighbor is available.
    pub bottom_left: bool,
}

impl NeighborAvailability {
    /// All neighbors available.
    pub const ALL: Self = Self {
        top: true,
        left: true,
        top_left: true,
        top_right: true,
        bottom_left: true,
    };

    /// No neighbors available.
    pub const NONE: Self = Self {
        top: false,
        left: false,
        top_left: false,
        top_right: false,
        bottom_left: false,
    };

    /// Check if any neighbor is available.
    #[must_use]
    pub const fn any(&self) -> bool {
        self.top || self.left || self.top_left || self.top_right || self.bottom_left
    }

    /// Check if top row is available (for vertical prediction).
    #[must_use]
    pub const fn has_top(&self) -> bool {
        self.top
    }

    /// Check if left column is available (for horizontal prediction).
    #[must_use]
    pub const fn has_left(&self) -> bool {
        self.left
    }
}

/// Intra prediction context.
///
/// Holds neighbor samples and availability information for intra prediction.
#[derive(Clone, Debug)]
pub struct IntraPredContext {
    /// Top row samples (above the block).
    top: TopSamples,
    /// Left column samples (to the left of the block).
    left: LeftSamples,
    /// Top-left corner sample.
    top_left: u16,
    /// Block width.
    width: usize,
    /// Block height.
    height: usize,
    /// Bit depth.
    bit_depth: BitDepth,
    /// Neighbor availability.
    availability: NeighborAvailability,
}

impl IntraPredContext {
    /// Create a new prediction context.
    #[must_use]
    pub fn new(width: usize, height: usize, bit_depth: BitDepth) -> Self {
        let midpoint = bit_depth.midpoint();
        Self {
            top: [midpoint; MAX_NEIGHBOR_SAMPLES],
            left: [midpoint; MAX_NEIGHBOR_SAMPLES],
            top_left: midpoint,
            width,
            height,
            bit_depth,
            availability: NeighborAvailability::NONE,
        }
    }

    /// Create with specific neighbor availability.
    #[must_use]
    pub fn with_availability(
        width: usize,
        height: usize,
        bit_depth: BitDepth,
        availability: NeighborAvailability,
    ) -> Self {
        let mut ctx = Self::new(width, height, bit_depth);
        ctx.availability = availability;
        ctx
    }

    /// Get the bit depth.
    #[must_use]
    pub const fn bit_depth(&self) -> BitDepth {
        self.bit_depth
    }

    /// Get block width.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Get block height.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Check if top neighbor is available.
    #[must_use]
    pub const fn has_top(&self) -> bool {
        self.availability.top
    }

    /// Check if left neighbor is available.
    #[must_use]
    pub const fn has_left(&self) -> bool {
        self.availability.left
    }

    /// Check if top-left neighbor is available.
    #[must_use]
    pub const fn has_top_left(&self) -> bool {
        self.availability.top_left
    }

    /// Get neighbor availability.
    #[must_use]
    pub const fn availability(&self) -> NeighborAvailability {
        self.availability
    }

    /// Set neighbor availability.
    pub fn set_availability(&mut self, has_top: bool, has_left: bool) {
        self.availability.top = has_top;
        self.availability.left = has_left;
        self.availability.top_left = has_top && has_left;
    }

    /// Set full neighbor availability.
    pub fn set_full_availability(&mut self, availability: NeighborAvailability) {
        self.availability = availability;
    }

    /// Get top samples slice.
    #[must_use]
    pub fn top_samples(&self) -> &[u16] {
        &self.top[..self.width.min(MAX_NEIGHBOR_SAMPLES)]
    }

    /// Get left samples slice.
    #[must_use]
    pub fn left_samples(&self) -> &[u16] {
        &self.left[..self.height.min(MAX_NEIGHBOR_SAMPLES)]
    }

    /// Get extended top samples (including top-right).
    #[must_use]
    pub fn extended_top_samples(&self) -> &[u16] {
        let count = (self.width * 2).min(MAX_NEIGHBOR_SAMPLES);
        &self.top[..count]
    }

    /// Get extended left samples (including bottom-left).
    #[must_use]
    pub fn extended_left_samples(&self) -> &[u16] {
        let count = (self.height * 2).min(MAX_NEIGHBOR_SAMPLES);
        &self.left[..count]
    }

    /// Get top-left sample.
    #[must_use]
    pub const fn top_left_sample(&self) -> u16 {
        self.top_left
    }

    /// Set a top sample.
    pub fn set_top_sample(&mut self, idx: usize, value: u16) {
        if idx < MAX_NEIGHBOR_SAMPLES {
            self.top[idx] = value;
        }
    }

    /// Set a left sample.
    pub fn set_left_sample(&mut self, idx: usize, value: u16) {
        if idx < MAX_NEIGHBOR_SAMPLES {
            self.left[idx] = value;
        }
    }

    /// Set top-left sample.
    pub fn set_top_left_sample(&mut self, value: u16) {
        self.top_left = value;
    }

    /// Set all top samples from a slice.
    pub fn set_top_samples(&mut self, samples: &[u16]) {
        let count = samples.len().min(MAX_NEIGHBOR_SAMPLES);
        self.top[..count].copy_from_slice(&samples[..count]);
    }

    /// Set all left samples from a slice.
    pub fn set_left_samples(&mut self, samples: &[u16]) {
        let count = samples.len().min(MAX_NEIGHBOR_SAMPLES);
        self.left[..count].copy_from_slice(&samples[..count]);
    }

    /// Apply a filter function to top samples.
    pub fn filter_top_samples<F>(&mut self, filter: F)
    where
        F: FnOnce(&mut [u16]),
    {
        filter(&mut self.top);
    }

    /// Apply a filter function to left samples.
    pub fn filter_left_samples<F>(&mut self, filter: F)
    where
        F: FnOnce(&mut [u16]),
    {
        filter(&mut self.left);
    }

    /// Reconstruct neighbors from a frame buffer.
    ///
    /// # Arguments
    /// * `frame` - Frame sample buffer
    /// * `frame_stride` - Frame row stride
    /// * `block_x` - Block X position in samples
    /// * `block_y` - Block Y position in samples
    /// * `frame_width` - Frame width in samples
    /// * `frame_height` - Frame height in samples
    #[allow(clippy::too_many_arguments)]
    pub fn reconstruct_neighbors(
        &mut self,
        frame: &[u16],
        frame_stride: usize,
        block_x: usize,
        block_y: usize,
        frame_width: usize,
        frame_height: usize,
    ) {
        // Determine availability
        let has_top = block_y > 0;
        let has_left = block_x > 0;
        let has_top_right = has_top && (block_x + self.width * 2 <= frame_width);
        let has_bottom_left = has_left && (block_y + self.height * 2 <= frame_height);

        self.availability = NeighborAvailability {
            top: has_top,
            left: has_left,
            top_left: has_top && has_left,
            top_right: has_top_right,
            bottom_left: has_bottom_left,
        };

        // Copy top samples
        if has_top {
            let top_y = block_y - 1;
            let top_row_start = top_y * frame_stride;

            // Copy regular top samples
            for x in 0..self.width {
                let frame_x = block_x + x;
                if frame_x < frame_width {
                    self.top[x] = frame[top_row_start + frame_x];
                }
            }

            // Copy top-right samples if available
            if has_top_right {
                for x in self.width..(self.width * 2) {
                    let frame_x = block_x + x;
                    if frame_x < frame_width {
                        self.top[x] = frame[top_row_start + frame_x];
                    }
                }
            } else {
                // Replicate last top sample
                let last = self.top[self.width.saturating_sub(1)];
                for x in self.width..(self.width * 2) {
                    self.top[x] = last;
                }
            }
        }

        // Copy left samples
        if has_left {
            let left_x = block_x - 1;

            // Copy regular left samples
            for y in 0..self.height {
                let frame_y = block_y + y;
                if frame_y < frame_height {
                    self.left[y] = frame[frame_y * frame_stride + left_x];
                }
            }

            // Copy bottom-left samples if available
            if has_bottom_left {
                for y in self.height..(self.height * 2) {
                    let frame_y = block_y + y;
                    if frame_y < frame_height {
                        self.left[y] = frame[frame_y * frame_stride + left_x];
                    }
                }
            } else {
                // Replicate last left sample
                let last = self.left[self.height.saturating_sub(1)];
                for y in self.height..(self.height * 2) {
                    self.left[y] = last;
                }
            }
        }

        // Copy top-left sample
        if has_top && has_left {
            self.top_left = frame[(block_y - 1) * frame_stride + (block_x - 1)];
        } else if has_top {
            self.top_left = self.top[0];
        } else if has_left {
            self.top_left = self.left[0];
        }
    }

    /// Check if the block is at the frame edge.
    #[must_use]
    pub const fn is_at_frame_edge(&self) -> bool {
        !self.availability.top || !self.availability.left
    }

    /// Fill unavailable samples with the midpoint value.
    pub fn fill_unavailable(&mut self) {
        let midpoint = self.bit_depth.midpoint();

        if !self.availability.top {
            self.top.fill(midpoint);
        }

        if !self.availability.left {
            self.left.fill(midpoint);
        }

        if !self.availability.top_left {
            self.top_left = midpoint;
        }
    }

    /// Get a sample at an extended position (can be negative for top-left region).
    #[must_use]
    pub fn get_extended_sample(&self, x: i32, y: i32) -> u16 {
        if x < 0 && y < 0 {
            // Top-left region
            self.top_left
        } else if y < 0 {
            // Top row
            let idx = x as usize;
            if idx < self.top.len() {
                self.top[idx]
            } else {
                self.top[self.top.len() - 1]
            }
        } else if x < 0 {
            // Left column
            let idx = y as usize;
            if idx < self.left.len() {
                self.left[idx]
            } else {
                self.left[self.left.len() - 1]
            }
        } else {
            // Should not happen for neighbor access
            self.bit_depth.midpoint()
        }
    }
}

impl Default for IntraPredContext {
    fn default() -> Self {
        Self::new(4, 4, BitDepth::Bits8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let ctx = IntraPredContext::new(8, 8, BitDepth::Bits8);
        assert_eq!(ctx.width(), 8);
        assert_eq!(ctx.height(), 8);
        assert_eq!(ctx.bit_depth(), BitDepth::Bits8);

        // Should be initialized with midpoint
        assert_eq!(ctx.top_left_sample(), 128);
        assert!(ctx.top_samples().iter().all(|&s| s == 128));
        assert!(ctx.left_samples().iter().all(|&s| s == 128));
    }

    #[test]
    fn test_availability() {
        let mut ctx = IntraPredContext::new(4, 4, BitDepth::Bits8);

        assert!(!ctx.has_top());
        assert!(!ctx.has_left());

        ctx.set_availability(true, true);
        assert!(ctx.has_top());
        assert!(ctx.has_left());
        assert!(ctx.has_top_left());
    }

    #[test]
    fn test_sample_setting() {
        let mut ctx = IntraPredContext::new(4, 4, BitDepth::Bits8);

        ctx.set_top_sample(0, 100);
        ctx.set_top_sample(1, 110);
        ctx.set_left_sample(0, 90);
        ctx.set_top_left_sample(95);

        assert_eq!(ctx.top_samples()[0], 100);
        assert_eq!(ctx.top_samples()[1], 110);
        assert_eq!(ctx.left_samples()[0], 90);
        assert_eq!(ctx.top_left_sample(), 95);
    }

    #[test]
    fn test_bulk_sample_setting() {
        let mut ctx = IntraPredContext::new(4, 4, BitDepth::Bits8);

        let top = [100u16, 110, 120, 130];
        let left = [90u16, 100, 110, 120];

        ctx.set_top_samples(&top);
        ctx.set_left_samples(&left);

        assert_eq!(ctx.top_samples()[..4], [100, 110, 120, 130]);
        assert_eq!(ctx.left_samples()[..4], [90, 100, 110, 120]);
    }

    #[test]
    fn test_reconstruct_neighbors() {
        // Create a simple 16x16 frame
        let frame_width = 16;
        let frame_height = 16;
        let mut frame = vec![0u16; frame_width * frame_height];

        // Fill with gradient
        for y in 0..frame_height {
            for x in 0..frame_width {
                frame[y * frame_width + x] = ((x + y) * 10) as u16;
            }
        }

        let mut ctx = IntraPredContext::new(4, 4, BitDepth::Bits8);

        // Reconstruct at position (4, 4)
        ctx.reconstruct_neighbors(&frame, frame_width, 4, 4, frame_width, frame_height);

        assert!(ctx.has_top());
        assert!(ctx.has_left());
        assert!(ctx.has_top_left());

        // Top row should be from y=3, x=4..8
        // Values: (4+3)*10=70, (5+3)*10=80, etc.
        assert_eq!(ctx.top_samples()[0], 70);
        assert_eq!(ctx.top_samples()[1], 80);

        // Left column should be from x=3, y=4..8
        // Values: (3+4)*10=70, (3+5)*10=80, etc.
        assert_eq!(ctx.left_samples()[0], 70);
        assert_eq!(ctx.left_samples()[1], 80);

        // Top-left should be from (3, 3)
        assert_eq!(ctx.top_left_sample(), 60);
    }

    #[test]
    fn test_reconstruct_at_edge() {
        let frame_width = 16;
        let frame_height = 16;
        let frame = vec![100u16; frame_width * frame_height];

        let mut ctx = IntraPredContext::new(4, 4, BitDepth::Bits8);

        // Reconstruct at position (0, 0) - top-left corner
        ctx.reconstruct_neighbors(&frame, frame_width, 0, 0, frame_width, frame_height);

        assert!(!ctx.has_top());
        assert!(!ctx.has_left());
        assert!(!ctx.has_top_left());
    }

    #[test]
    fn test_extended_sample_access() {
        let mut ctx = IntraPredContext::new(4, 4, BitDepth::Bits8);

        ctx.set_top_samples(&[10, 20, 30, 40]);
        ctx.set_left_samples(&[15, 25, 35, 45]);
        ctx.set_top_left_sample(5);

        // Top-left region
        assert_eq!(ctx.get_extended_sample(-1, -1), 5);

        // Top row
        assert_eq!(ctx.get_extended_sample(0, -1), 10);
        assert_eq!(ctx.get_extended_sample(1, -1), 20);

        // Left column
        assert_eq!(ctx.get_extended_sample(-1, 0), 15);
        assert_eq!(ctx.get_extended_sample(-1, 1), 25);
    }

    #[test]
    fn test_neighbor_availability_constants() {
        let all = NeighborAvailability::ALL;
        assert!(all.top);
        assert!(all.left);
        assert!(all.top_left);
        assert!(all.any());

        let none = NeighborAvailability::NONE;
        assert!(!none.top);
        assert!(!none.left);
        assert!(!none.any());
    }

    #[test]
    fn test_fill_unavailable() {
        let mut ctx = IntraPredContext::new(4, 4, BitDepth::Bits8);
        ctx.set_top_samples(&[200, 200, 200, 200]);
        ctx.availability.top = false;

        ctx.fill_unavailable();

        // Top should be filled with midpoint (128)
        assert!(ctx.top_samples().iter().all(|&s| s == 128));
    }

    #[test]
    fn test_bit_depth_10() {
        let ctx = IntraPredContext::new(4, 4, BitDepth::Bits10);
        assert_eq!(ctx.bit_depth(), BitDepth::Bits10);
        assert_eq!(ctx.top_left_sample(), 512); // 10-bit midpoint
    }

    #[test]
    fn test_extended_samples() {
        let mut ctx = IntraPredContext::new(4, 4, BitDepth::Bits8);

        // Set extended top samples (for top-right)
        for i in 0..8 {
            ctx.set_top_sample(i, (i * 10) as u16);
        }

        let extended = ctx.extended_top_samples();
        assert_eq!(extended.len(), 8);
        assert_eq!(extended[0], 0);
        assert_eq!(extended[7], 70);
    }
}
