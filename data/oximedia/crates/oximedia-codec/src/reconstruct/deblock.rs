//! Deblocking filter for reducing blocking artifacts.
//!
//! The deblocking filter is applied at block boundaries to reduce
//! artifacts caused by block-based transform coding. It uses adaptive
//! filtering based on boundary strength (bS) calculations.

#![forbid(unsafe_code)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::identity_op)]
#![allow(clippy::range_plus_one)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::single_match_else)]
#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::unused_self)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]

use super::pipeline::FrameContext;
use super::{FrameBuffer, PlaneBuffer, ReconstructResult};

// =============================================================================
// Constants
// =============================================================================

/// Maximum boundary strength.
pub const MAX_BOUNDARY_STRENGTH: u8 = 4;

/// Minimum block size for deblocking.
pub const MIN_DEBLOCK_SIZE: usize = 4;

/// Filter tap count for normal filter.
pub const NORMAL_FILTER_TAPS: usize = 4;

/// Filter tap count for strong filter.
pub const STRONG_FILTER_TAPS: usize = 8;

// =============================================================================
// Filter Strength
// =============================================================================

/// Boundary strength (bS) for deblocking.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FilterStrength {
    /// Boundary strength value (0-4).
    pub bs: u8,
    /// Alpha threshold parameter.
    pub alpha: u8,
    /// Beta threshold parameter.
    pub beta: u8,
    /// tc0 clipping parameter.
    pub tc0: u8,
}

impl FilterStrength {
    /// Create a new filter strength.
    #[must_use]
    pub const fn new(bs: u8) -> Self {
        Self {
            bs,
            alpha: 0,
            beta: 0,
            tc0: 0,
        }
    }

    /// Create from quantization parameter.
    #[must_use]
    pub fn from_qp(qp: u8, bs: u8) -> Self {
        let idx = qp.min(51) as usize;

        // Alpha table (indexed by QP)
        const ALPHA: [u8; 52] = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15,
            17, 20, 22, 25, 28, 32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 144, 162,
            182, 203, 226, 255, 255,
        ];

        // Beta table (indexed by QP)
        const BETA: [u8; 52] = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7,
            7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
        ];

        // tc0 table (indexed by QP and bS)
        const TC0: [[u8; 3]; 52] = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 2],
            [1, 1, 2],
            [1, 1, 2],
            [1, 1, 2],
            [1, 2, 3],
            [1, 2, 3],
            [2, 2, 3],
            [2, 2, 4],
            [2, 3, 4],
            [2, 3, 4],
            [3, 3, 5],
            [3, 4, 6],
            [3, 4, 6],
            [4, 5, 7],
            [4, 5, 8],
            [4, 6, 9],
            [5, 7, 10],
            [6, 8, 11],
            [6, 8, 13],
            [7, 10, 14],
            [8, 11, 16],
            [9, 12, 18],
            [10, 13, 20],
            [11, 15, 23],
            [13, 17, 25],
        ];

        let tc0 = if bs > 0 && bs < 4 {
            TC0[idx][(bs - 1) as usize]
        } else {
            0
        };

        Self {
            bs,
            alpha: ALPHA[idx],
            beta: BETA[idx],
            tc0,
        }
    }

    /// Check if filtering should be applied.
    #[must_use]
    pub const fn should_filter(&self) -> bool {
        self.bs > 0
    }

    /// Check if this is a strong filter (bS = 4).
    #[must_use]
    pub const fn is_strong(&self) -> bool {
        self.bs >= 4
    }

    /// Get the tc parameter for filtering.
    #[must_use]
    pub const fn tc(&self) -> i16 {
        if self.bs >= 4 {
            0 // Not used for strong filter
        } else {
            self.tc0 as i16
        }
    }
}

// =============================================================================
// Deblock Parameters
// =============================================================================

/// Parameters for deblocking filter.
#[derive(Clone, Debug, Default)]
pub struct DeblockParams {
    /// Deblocking filter disabled.
    pub disable_deblock: bool,
    /// Alpha offset (slice level).
    pub alpha_offset: i8,
    /// Beta offset (slice level).
    pub beta_offset: i8,
    /// Quantization parameter.
    pub qp: u8,
}

impl DeblockParams {
    /// Create new deblock parameters.
    #[must_use]
    pub fn new(qp: u8) -> Self {
        Self {
            qp,
            ..Default::default()
        }
    }

    /// Set alpha offset.
    #[must_use]
    pub const fn with_alpha_offset(mut self, offset: i8) -> Self {
        self.alpha_offset = offset;
        self
    }

    /// Set beta offset.
    #[must_use]
    pub const fn with_beta_offset(mut self, offset: i8) -> Self {
        self.beta_offset = offset;
        self
    }

    /// Get effective QP for alpha calculation.
    #[must_use]
    pub fn effective_qp_alpha(&self) -> u8 {
        ((self.qp as i16) + i16::from(self.alpha_offset)).clamp(0, 51) as u8
    }

    /// Get effective QP for beta calculation.
    #[must_use]
    pub fn effective_qp_beta(&self) -> u8 {
        ((self.qp as i16) + i16::from(self.beta_offset)).clamp(0, 51) as u8
    }

    /// Create filter strength for an edge.
    #[must_use]
    pub fn create_strength(&self, bs: u8) -> FilterStrength {
        FilterStrength::from_qp(self.qp, bs)
    }
}

// =============================================================================
// Boundary Strength Calculator
// =============================================================================

/// Block information for boundary strength calculation.
#[derive(Clone, Copy, Debug, Default)]
pub struct BlockInfo {
    /// Block is intra-coded.
    pub is_intra: bool,
    /// Has non-zero coefficients.
    pub has_coeffs: bool,
    /// Reference frame index.
    pub ref_frame: u8,
    /// Motion vector x component (in quarter-pel).
    pub mv_x: i16,
    /// Motion vector y component (in quarter-pel).
    pub mv_y: i16,
}

impl BlockInfo {
    /// Create info for an intra block.
    #[must_use]
    pub const fn intra() -> Self {
        Self {
            is_intra: true,
            has_coeffs: true,
            ref_frame: 0,
            mv_x: 0,
            mv_y: 0,
        }
    }

    /// Create info for an inter block.
    #[must_use]
    pub const fn inter(ref_frame: u8, mv_x: i16, mv_y: i16, has_coeffs: bool) -> Self {
        Self {
            is_intra: false,
            has_coeffs,
            ref_frame,
            mv_x,
            mv_y,
        }
    }
}

/// Calculate boundary strength between two adjacent blocks.
#[must_use]
pub fn calculate_boundary_strength(p: &BlockInfo, q: &BlockInfo) -> u8 {
    // If either block is intra, bS = 4
    if p.is_intra || q.is_intra {
        return 4;
    }

    // If either block has non-zero coefficients, bS = 2
    if p.has_coeffs || q.has_coeffs {
        return 2;
    }

    // Check reference frame difference
    if p.ref_frame != q.ref_frame {
        return 1;
    }

    // Check motion vector difference (threshold is 4 quarter-pels = 1 full pel)
    if (p.mv_x - q.mv_x).abs() >= 4 || (p.mv_y - q.mv_y).abs() >= 4 {
        return 1;
    }

    // No filtering needed
    0
}

// =============================================================================
// Deblocking Filter
// =============================================================================

/// Deblocking filter implementation.
#[derive(Debug)]
pub struct DeblockFilter {
    /// Filter parameters.
    params: DeblockParams,
    /// Block size.
    block_size: usize,
}

impl Default for DeblockFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl DeblockFilter {
    /// Create a new deblocking filter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            params: DeblockParams::default(),
            block_size: 8,
        }
    }

    /// Create with specific parameters.
    #[must_use]
    pub fn with_params(params: DeblockParams) -> Self {
        Self {
            params,
            block_size: 8,
        }
    }

    /// Set filter parameters.
    pub fn set_params(&mut self, params: DeblockParams) {
        self.params = params;
    }

    /// Get current parameters.
    #[must_use]
    pub fn params(&self) -> &DeblockParams {
        &self.params
    }

    /// Apply deblocking filter to a frame.
    ///
    /// # Errors
    ///
    /// Returns error if filtering fails.
    pub fn apply(
        &mut self,
        frame: &mut FrameBuffer,
        _context: &FrameContext,
    ) -> ReconstructResult<()> {
        if self.params.disable_deblock {
            return Ok(());
        }

        let bd = frame.bit_depth();

        // Filter Y plane
        self.filter_plane(frame.y_plane_mut(), bd, false)?;

        // Filter chroma planes (use default bS = 1 for simplicity)
        if let Some(u) = frame.u_plane_mut() {
            self.filter_plane(u, bd, true)?;
        }
        if let Some(v) = frame.v_plane_mut() {
            self.filter_plane(v, bd, true)?;
        }

        Ok(())
    }

    /// Filter a single plane.
    fn filter_plane(
        &self,
        plane: &mut PlaneBuffer,
        bd: u8,
        is_chroma: bool,
    ) -> ReconstructResult<()> {
        let width = plane.width() as usize;
        let height = plane.height() as usize;
        let block_size = if is_chroma { 4 } else { self.block_size };

        // Use a default boundary strength for demonstration
        // In a full implementation, this would come from block info
        let strength = self.params.create_strength(2);

        // Filter vertical edges
        for by in 0..(height / block_size) {
            for bx in 1..(width / block_size) {
                let x = (bx * block_size) as u32;
                let y = (by * block_size) as u32;
                self.filter_edge_vertical(plane, x, y, block_size, &strength, bd);
            }
        }

        // Filter horizontal edges
        for by in 1..(height / block_size) {
            for bx in 0..(width / block_size) {
                let x = (bx * block_size) as u32;
                let y = (by * block_size) as u32;
                self.filter_edge_horizontal(plane, x, y, block_size, &strength, bd);
            }
        }

        Ok(())
    }

    /// Filter a vertical edge.
    fn filter_edge_vertical(
        &self,
        plane: &mut PlaneBuffer,
        x: u32,
        y: u32,
        length: usize,
        strength: &FilterStrength,
        bd: u8,
    ) {
        if !strength.should_filter() {
            return;
        }

        for i in 0..length {
            let py = y + i as u32;

            // Get samples
            let p2 = plane.get(x.saturating_sub(3), py);
            let p1 = plane.get(x.saturating_sub(2), py);
            let p0 = plane.get(x.saturating_sub(1), py);
            let q0 = plane.get(x, py);
            let q1 = plane.get(x + 1, py);
            let q2 = plane.get(x + 2, py);

            // Check filter condition
            if !self.should_filter_edge(p0, p1, q0, q1, strength) {
                continue;
            }

            // Apply filter
            let (new_p0, new_p1, new_q0, new_q1) = if strength.is_strong() {
                self.strong_filter(p0, p1, p2, q0, q1, q2, bd)
            } else {
                self.normal_filter(p0, p1, q0, q1, strength, bd)
            };

            // Write back
            plane.set(x.saturating_sub(1), py, new_p0);
            plane.set(x.saturating_sub(2), py, new_p1);
            plane.set(x, py, new_q0);
            plane.set(x + 1, py, new_q1);
        }
    }

    /// Filter a horizontal edge.
    fn filter_edge_horizontal(
        &self,
        plane: &mut PlaneBuffer,
        x: u32,
        y: u32,
        length: usize,
        strength: &FilterStrength,
        bd: u8,
    ) {
        if !strength.should_filter() {
            return;
        }

        for i in 0..length {
            let px = x + i as u32;

            // Get samples
            let p2 = plane.get(px, y.saturating_sub(3));
            let p1 = plane.get(px, y.saturating_sub(2));
            let p0 = plane.get(px, y.saturating_sub(1));
            let q0 = plane.get(px, y);
            let q1 = plane.get(px, y + 1);
            let q2 = plane.get(px, y + 2);

            // Check filter condition
            if !self.should_filter_edge(p0, p1, q0, q1, strength) {
                continue;
            }

            // Apply filter
            let (new_p0, new_p1, new_q0, new_q1) = if strength.is_strong() {
                self.strong_filter(p0, p1, p2, q0, q1, q2, bd)
            } else {
                self.normal_filter(p0, p1, q0, q1, strength, bd)
            };

            // Write back
            plane.set(px, y.saturating_sub(1), new_p0);
            plane.set(px, y.saturating_sub(2), new_p1);
            plane.set(px, y, new_q0);
            plane.set(px, y + 1, new_q1);
        }
    }

    /// Check if edge should be filtered.
    fn should_filter_edge(
        &self,
        p0: i16,
        p1: i16,
        q0: i16,
        q1: i16,
        strength: &FilterStrength,
    ) -> bool {
        let alpha = i16::from(strength.alpha);
        let beta = i16::from(strength.beta);

        // Check thresholds
        (p0 - q0).abs() < alpha && (p1 - p0).abs() < beta && (q1 - q0).abs() < beta
    }

    /// Apply normal (4-tap) filter.
    fn normal_filter(
        &self,
        p0: i16,
        p1: i16,
        q0: i16,
        q1: i16,
        strength: &FilterStrength,
        bd: u8,
    ) -> (i16, i16, i16, i16) {
        let max_val = (1i16 << bd) - 1;
        let tc = strength.tc();

        // Compute delta
        let delta0 = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);

        let new_p0 = (p0 + delta0).clamp(0, max_val);
        let new_q0 = (q0 - delta0).clamp(0, max_val);

        // P1 and Q1 filtering (only if tc0 > 0)
        let (new_p1, new_q1) = if tc > 0 {
            let delta_p1 = ((p2_avg(p0, p1) - p1 + delta0) >> 1).clamp(-tc, tc);
            let delta_q1 = ((p2_avg(q0, q1) - q1 - delta0) >> 1).clamp(-tc, tc);
            (
                (p1 + delta_p1).clamp(0, max_val),
                (q1 + delta_q1).clamp(0, max_val),
            )
        } else {
            (p1, q1)
        };

        (new_p0, new_p1, new_q0, new_q1)
    }

    /// Apply strong (8-tap) filter.
    fn strong_filter(
        &self,
        p0: i16,
        p1: i16,
        p2: i16,
        q0: i16,
        q1: i16,
        q2: i16,
        bd: u8,
    ) -> (i16, i16, i16, i16) {
        let max_val = (1i16 << bd) - 1;

        // Strong filter uses weighted average
        let new_p0 = ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3).clamp(0, max_val);
        let new_p1 = ((p2 + p1 + p0 + q0 + 2) >> 2).clamp(0, max_val);
        let new_q0 = ((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3).clamp(0, max_val);
        let new_q1 = ((p0 + q0 + q1 + q2 + 2) >> 2).clamp(0, max_val);

        (new_p0, new_p1, new_q0, new_q1)
    }
}

/// Helper for p2 average calculation.
#[inline]
fn p2_avg(p0: i16, p1: i16) -> i16 {
    (p0 + p1 + 1) >> 1
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reconstruct::ChromaSubsampling;

    #[test]
    fn test_filter_strength_new() {
        let strength = FilterStrength::new(2);
        assert_eq!(strength.bs, 2);
        assert!(strength.should_filter());
        assert!(!strength.is_strong());

        let strength_zero = FilterStrength::new(0);
        assert!(!strength_zero.should_filter());
    }

    #[test]
    fn test_filter_strength_from_qp() {
        let strength = FilterStrength::from_qp(26, 2);
        assert_eq!(strength.bs, 2);
        assert!(strength.alpha > 0);
        assert!(strength.beta > 0);

        let strength_strong = FilterStrength::from_qp(26, 4);
        assert!(strength_strong.is_strong());
    }

    #[test]
    fn test_deblock_params() {
        let params = DeblockParams::new(26)
            .with_alpha_offset(2)
            .with_beta_offset(-2);

        assert_eq!(params.qp, 26);
        assert_eq!(params.effective_qp_alpha(), 28);
        assert_eq!(params.effective_qp_beta(), 24);
    }

    #[test]
    fn test_block_info() {
        let intra = BlockInfo::intra();
        assert!(intra.is_intra);

        let inter = BlockInfo::inter(1, 10, 20, true);
        assert!(!inter.is_intra);
        assert_eq!(inter.ref_frame, 1);
        assert_eq!(inter.mv_x, 10);
    }

    #[test]
    fn test_boundary_strength_calculation() {
        let intra = BlockInfo::intra();
        let inter = BlockInfo::inter(0, 0, 0, false);

        // Intra edge
        assert_eq!(calculate_boundary_strength(&intra, &inter), 4);

        // Two inter blocks with different refs
        let inter2 = BlockInfo::inter(1, 0, 0, false);
        assert_eq!(calculate_boundary_strength(&inter, &inter2), 1);

        // Two inter blocks with same ref but different MV
        let inter3 = BlockInfo::inter(0, 10, 0, false);
        assert_eq!(calculate_boundary_strength(&inter, &inter3), 1);

        // Two identical inter blocks
        assert_eq!(calculate_boundary_strength(&inter, &inter), 0);
    }

    #[test]
    fn test_boundary_strength_with_coeffs() {
        let inter_coeffs = BlockInfo::inter(0, 0, 0, true);
        let inter_no_coeffs = BlockInfo::inter(0, 0, 0, false);

        assert_eq!(
            calculate_boundary_strength(&inter_coeffs, &inter_no_coeffs),
            2
        );
    }

    #[test]
    fn test_deblock_filter_creation() {
        let filter = DeblockFilter::new();
        assert!(!filter.params().disable_deblock);
    }

    #[test]
    fn test_deblock_filter_with_params() {
        let params = DeblockParams::new(26);
        let filter = DeblockFilter::with_params(params);
        assert_eq!(filter.params().qp, 26);
    }

    #[test]
    fn test_deblock_filter_apply_disabled() {
        let mut frame = FrameBuffer::new(64, 64, 8, ChromaSubsampling::Cs420);
        let context = FrameContext::new(64, 64);

        let mut params = DeblockParams::new(26);
        params.disable_deblock = true;
        let mut filter = DeblockFilter::with_params(params);

        let result = filter.apply(&mut frame, &context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_deblock_filter_apply() {
        let mut frame = FrameBuffer::new(64, 64, 8, ChromaSubsampling::Cs420);

        // Create an artificial edge
        for y in 0..64 {
            for x in 0..8 {
                frame.y_plane_mut().set(x, y as u32, 100);
            }
            for x in 8..64 {
                frame.y_plane_mut().set(x as u32, y as u32, 150);
            }
        }

        let context = FrameContext::new(64, 64);
        let params = DeblockParams::new(26);
        let mut filter = DeblockFilter::with_params(params);

        let result = filter.apply(&mut frame, &context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_strong_filter() {
        let filter = DeblockFilter::new();

        let (new_p0, _new_p1, new_q0, _new_q1) =
            filter.strong_filter(100, 95, 90, 150, 155, 160, 8);

        // After strong filtering, edge should be smoother
        assert!((new_p0 - new_q0).abs() < (100 - 150i16).abs());
        assert!(new_p0 >= 0 && new_p0 <= 255);
        assert!(new_q0 >= 0 && new_q0 <= 255);
    }

    #[test]
    fn test_normal_filter() {
        let filter = DeblockFilter::new();
        let strength = FilterStrength::from_qp(26, 2);

        let (new_p0, _new_p1, new_q0, _new_q1) =
            filter.normal_filter(100, 95, 150, 155, &strength, 8);

        // After normal filtering, values should be closer
        assert!((new_p0 - new_q0).abs() <= (100 - 150i16).abs());
    }

    #[test]
    fn test_constants() {
        assert_eq!(MAX_BOUNDARY_STRENGTH, 4);
        assert_eq!(MIN_DEBLOCK_SIZE, 4);
        assert_eq!(NORMAL_FILTER_TAPS, 4);
        assert_eq!(STRONG_FILTER_TAPS, 8);
    }
}
