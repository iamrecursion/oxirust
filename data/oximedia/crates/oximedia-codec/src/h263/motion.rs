//! Motion compensation and estimation for H.263.
//!
//! This module implements motion vector prediction, motion compensation
//! with half-pixel precision, and motion estimation algorithms.

use crate::CodecError;

/// Motion vector with half-pixel precision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MotionVector {
    /// Horizontal component (in half-pixels).
    pub x: i16,
    /// Vertical component (in half-pixels).
    pub y: i16,
}

impl MotionVector {
    /// Create a new motion vector.
    #[must_use]
    pub const fn new(x: i16, y: i16) -> Self {
        Self { x, y }
    }

    /// Create a zero motion vector.
    #[must_use]
    pub const fn zero() -> Self {
        Self { x: 0, y: 0 }
    }

    /// Add another motion vector.
    #[must_use]
    pub const fn add(&self, other: &Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }

    /// Subtract another motion vector.
    #[must_use]
    pub const fn sub(&self, other: &Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }

    /// Clamp motion vector to valid range.
    #[must_use]
    pub fn clamp(&self, min: i16, max: i16) -> Self {
        Self {
            x: self.x.clamp(min, max),
            y: self.y.clamp(min, max),
        }
    }

    /// Get magnitude squared.
    #[must_use]
    pub const fn magnitude_sq(&self) -> i32 {
        (self.x as i32) * (self.x as i32) + (self.y as i32) * (self.y as i32)
    }
}

/// Motion vector predictor.
///
/// Predicts motion vectors based on neighboring macroblocks.
pub struct MotionVectorPredictor {
    /// MV storage for previous row.
    prev_row: Vec<MotionVector>,
    /// Current row MVs.
    curr_row: Vec<MotionVector>,
    /// Width in macroblocks.
    mb_width: usize,
}

impl MotionVectorPredictor {
    /// Create a new motion vector predictor.
    ///
    /// # Arguments
    ///
    /// * `mb_width` - Width in macroblocks
    #[must_use]
    pub fn new(mb_width: usize) -> Self {
        Self {
            prev_row: vec![MotionVector::zero(); mb_width + 1],
            curr_row: vec![MotionVector::zero(); mb_width + 1],
            mb_width,
        }
    }

    /// Predict motion vector for a macroblock.
    ///
    /// Uses median prediction from left, top, and top-right neighbors.
    ///
    /// # Arguments
    ///
    /// * `mb_x` - Macroblock X position
    /// * `mb_y` - Macroblock Y position
    ///
    /// # Returns
    ///
    /// Predicted motion vector.
    #[must_use]
    pub fn predict(&self, mb_x: usize, mb_y: usize) -> MotionVector {
        let mv_left = if mb_x > 0 {
            self.curr_row[mb_x - 1]
        } else {
            MotionVector::zero()
        };

        let mv_top = if mb_y > 0 {
            self.prev_row[mb_x]
        } else {
            MotionVector::zero()
        };

        let mv_top_right = if mb_y > 0 && mb_x < self.mb_width - 1 {
            self.prev_row[mb_x + 1]
        } else {
            MotionVector::zero()
        };

        // Median prediction
        let pred_x = median3(mv_left.x, mv_top.x, mv_top_right.x);
        let pred_y = median3(mv_left.y, mv_top.y, mv_top_right.y);

        MotionVector::new(pred_x, pred_y)
    }

    /// Update motion vector for a macroblock.
    ///
    /// # Arguments
    ///
    /// * `mb_x` - Macroblock X position
    /// * `mv` - Motion vector for this macroblock
    pub fn update(&mut self, mb_x: usize, mv: MotionVector) {
        self.curr_row[mb_x] = mv;
    }

    /// Advance to next row.
    pub fn next_row(&mut self) {
        std::mem::swap(&mut self.prev_row, &mut self.curr_row);
        self.curr_row.fill(MotionVector::zero());
    }
}

/// Calculate median of three values.
#[must_use]
fn median3(a: i16, b: i16, c: i16) -> i16 {
    if a > b {
        if b > c {
            b
        } else if a > c {
            c
        } else {
            a
        }
    } else if a > c {
        a
    } else if b > c {
        c
    } else {
        b
    }
}

/// Motion compensate a macroblock.
///
/// Performs motion compensation with half-pixel precision.
///
/// # Arguments
///
/// * `ref_frame` - Reference frame data
/// * `ref_stride` - Reference frame stride
/// * `dst` - Destination buffer
/// * `dst_stride` - Destination stride
/// * `mv` - Motion vector (half-pixel precision)
/// * `mb_x` - Macroblock X position
/// * `mb_y` - Macroblock Y position
/// * `width` - Frame width
/// * `height` - Frame height
///
/// # Errors
///
/// Returns error if motion vector is out of bounds.
#[allow(clippy::too_many_arguments)]
pub fn motion_compensate_mb(
    ref_frame: &[u8],
    ref_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    mv: MotionVector,
    mb_x: usize,
    mb_y: usize,
    width: usize,
    height: usize,
) -> Result<(), CodecError> {
    let mb_size = 16;
    let src_x = (mb_x * mb_size) as i32 + (mv.x / 2) as i32;
    let src_y = (mb_y * mb_size) as i32 + (mv.y / 2) as i32;

    // Check bounds
    if src_x < 0
        || src_y < 0
        || src_x + mb_size as i32 > width as i32
        || src_y + mb_size as i32 > height as i32
    {
        return Err(CodecError::InvalidData(
            "Motion vector out of bounds".into(),
        ));
    }

    let half_pel_x = (mv.x & 1) != 0;
    let half_pel_y = (mv.y & 1) != 0;

    if !half_pel_x && !half_pel_y {
        // Full-pixel motion compensation
        copy_block(
            ref_frame,
            ref_stride,
            src_x as usize,
            src_y as usize,
            dst,
            dst_stride,
            0,
            0,
            mb_size,
            mb_size,
        );
    } else if half_pel_x && !half_pel_y {
        // Half-pixel horizontal
        interpolate_horizontal(
            ref_frame,
            ref_stride,
            src_x as usize,
            src_y as usize,
            dst,
            dst_stride,
            0,
            0,
            mb_size,
            mb_size,
        );
    } else if !half_pel_x && half_pel_y {
        // Half-pixel vertical
        interpolate_vertical(
            ref_frame,
            ref_stride,
            src_x as usize,
            src_y as usize,
            dst,
            dst_stride,
            0,
            0,
            mb_size,
            mb_size,
        );
    } else {
        // Half-pixel both directions
        interpolate_diagonal(
            ref_frame,
            ref_stride,
            src_x as usize,
            src_y as usize,
            dst,
            dst_stride,
            0,
            0,
            mb_size,
            mb_size,
        );
    }

    Ok(())
}

/// Copy a block from source to destination.
#[allow(clippy::too_many_arguments)]
fn copy_block(
    src: &[u8],
    src_stride: usize,
    src_x: usize,
    src_y: usize,
    dst: &mut [u8],
    dst_stride: usize,
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
) {
    for y in 0..height {
        let src_offset = (src_y + y) * src_stride + src_x;
        let dst_offset = (dst_y + y) * dst_stride + dst_x;

        if src_offset + width <= src.len() && dst_offset + width <= dst.len() {
            dst[dst_offset..dst_offset + width]
                .copy_from_slice(&src[src_offset..src_offset + width]);
        }
    }
}

/// Interpolate horizontally (half-pixel between pixels).
#[allow(clippy::too_many_arguments)]
fn interpolate_horizontal(
    src: &[u8],
    src_stride: usize,
    src_x: usize,
    src_y: usize,
    dst: &mut [u8],
    dst_stride: usize,
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
) {
    for y in 0..height {
        for x in 0..width {
            let src_offset = (src_y + y) * src_stride + src_x + x;
            let dst_offset = (dst_y + y) * dst_stride + dst_x + x;

            if src_offset + 1 < src.len() && dst_offset < dst.len() {
                let a = src[src_offset] as u16;
                let b = src[src_offset + 1] as u16;
                dst[dst_offset] = ((a + b + 1) / 2) as u8;
            }
        }
    }
}

/// Interpolate vertically (half-pixel between rows).
#[allow(clippy::too_many_arguments)]
fn interpolate_vertical(
    src: &[u8],
    src_stride: usize,
    src_x: usize,
    src_y: usize,
    dst: &mut [u8],
    dst_stride: usize,
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
) {
    for y in 0..height {
        for x in 0..width {
            let src_offset = (src_y + y) * src_stride + src_x + x;
            let dst_offset = (dst_y + y) * dst_stride + dst_x + x;

            if src_offset + src_stride < src.len() && dst_offset < dst.len() {
                let a = src[src_offset] as u16;
                let b = src[src_offset + src_stride] as u16;
                dst[dst_offset] = ((a + b + 1) / 2) as u8;
            }
        }
    }
}

/// Interpolate diagonally (half-pixel in both directions).
#[allow(clippy::too_many_arguments)]
fn interpolate_diagonal(
    src: &[u8],
    src_stride: usize,
    src_x: usize,
    src_y: usize,
    dst: &mut [u8],
    dst_stride: usize,
    dst_x: usize,
    dst_y: usize,
    width: usize,
    height: usize,
) {
    for y in 0..height {
        for x in 0..width {
            let src_offset = (src_y + y) * src_stride + src_x + x;
            let dst_offset = (dst_y + y) * dst_stride + dst_x + x;

            if src_offset + src_stride + 1 < src.len() && dst_offset < dst.len() {
                let a = src[src_offset] as u16;
                let b = src[src_offset + 1] as u16;
                let c = src[src_offset + src_stride] as u16;
                let d = src[src_offset + src_stride + 1] as u16;
                dst[dst_offset] = ((a + b + c + d + 2) / 4) as u8;
            }
        }
    }
}

/// Motion estimation search algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchAlgorithm {
    /// Full search (exhaustive).
    Full,
    /// Three-step search.
    ThreeStep,
    /// Diamond search.
    Diamond,
    /// Hexagon search.
    Hexagon,
}

/// Motion estimator.
pub struct MotionEstimator {
    /// Search algorithm.
    algorithm: SearchAlgorithm,
    /// Search range (in pixels).
    search_range: i16,
}

impl MotionEstimator {
    /// Create a new motion estimator.
    ///
    /// # Arguments
    ///
    /// * `algorithm` - Search algorithm
    /// * `search_range` - Search range in pixels
    #[must_use]
    pub const fn new(algorithm: SearchAlgorithm, search_range: i16) -> Self {
        Self {
            algorithm,
            search_range,
        }
    }

    /// Estimate motion vector for a macroblock.
    ///
    /// # Arguments
    ///
    /// * `cur_frame` - Current frame
    /// * `cur_stride` - Current frame stride
    /// * `ref_frame` - Reference frame
    /// * `ref_stride` - Reference frame stride
    /// * `mb_x` - Macroblock X position
    /// * `mb_y` - Macroblock Y position
    /// * `width` - Frame width
    /// * `height` - Frame height
    ///
    /// # Returns
    ///
    /// Best motion vector.
    #[allow(clippy::too_many_arguments)]
    pub fn estimate(
        &self,
        cur_frame: &[u8],
        cur_stride: usize,
        ref_frame: &[u8],
        ref_stride: usize,
        mb_x: usize,
        mb_y: usize,
        width: usize,
        height: usize,
    ) -> MotionVector {
        match self.algorithm {
            SearchAlgorithm::Full => self.full_search(
                cur_frame, cur_stride, ref_frame, ref_stride, mb_x, mb_y, width, height,
            ),
            SearchAlgorithm::Diamond => self.diamond_search(
                cur_frame, cur_stride, ref_frame, ref_stride, mb_x, mb_y, width, height,
            ),
            SearchAlgorithm::Hexagon => self.hexagon_search(
                cur_frame, cur_stride, ref_frame, ref_stride, mb_x, mb_y, width, height,
            ),
            SearchAlgorithm::ThreeStep => self.three_step_search(
                cur_frame, cur_stride, ref_frame, ref_stride, mb_x, mb_y, width, height,
            ),
        }
    }

    /// Full search motion estimation.
    #[allow(clippy::too_many_arguments)]
    fn full_search(
        &self,
        cur_frame: &[u8],
        cur_stride: usize,
        ref_frame: &[u8],
        ref_stride: usize,
        mb_x: usize,
        mb_y: usize,
        width: usize,
        height: usize,
    ) -> MotionVector {
        let mut best_mv = MotionVector::zero();
        let mut best_sad = u32::MAX;

        for dy in -self.search_range..=self.search_range {
            for dx in -self.search_range..=self.search_range {
                let mv = MotionVector::new(dx * 2, dy * 2);
                if let Ok(sad) = self.calculate_sad(
                    cur_frame, cur_stride, ref_frame, ref_stride, mb_x, mb_y, mv, width, height,
                ) {
                    if sad < best_sad {
                        best_sad = sad;
                        best_mv = mv;
                    }
                }
            }
        }

        best_mv
    }

    /// Diamond search motion estimation.
    #[allow(clippy::too_many_arguments)]
    fn diamond_search(
        &self,
        cur_frame: &[u8],
        cur_stride: usize,
        ref_frame: &[u8],
        ref_stride: usize,
        mb_x: usize,
        mb_y: usize,
        width: usize,
        height: usize,
    ) -> MotionVector {
        const LDSP: [(i16, i16); 9] = [
            (0, 0),
            (0, -2),
            (-1, -1),
            (1, -1),
            (-2, 0),
            (2, 0),
            (-1, 1),
            (1, 1),
            (0, 2),
        ];

        const SDSP: [(i16, i16); 5] = [(0, 0), (0, -1), (-1, 0), (1, 0), (0, 1)];

        let mut center = MotionVector::zero();
        let mut best_sad = self
            .calculate_sad(
                cur_frame, cur_stride, ref_frame, ref_stride, mb_x, mb_y, center, width, height,
            )
            .unwrap_or(u32::MAX);

        // Large diamond search
        loop {
            let mut improved = false;

            for &(dx, dy) in &LDSP {
                let mv = MotionVector::new(center.x + dx * 2, center.y + dy * 2);
                if let Ok(sad) = self.calculate_sad(
                    cur_frame, cur_stride, ref_frame, ref_stride, mb_x, mb_y, mv, width, height,
                ) {
                    if sad < best_sad {
                        best_sad = sad;
                        center = mv;
                        improved = true;
                    }
                }
            }

            if !improved {
                break;
            }
        }

        // Small diamond search
        loop {
            let mut improved = false;

            for &(dx, dy) in &SDSP {
                let mv = MotionVector::new(center.x + dx * 2, center.y + dy * 2);
                if let Ok(sad) = self.calculate_sad(
                    cur_frame, cur_stride, ref_frame, ref_stride, mb_x, mb_y, mv, width, height,
                ) {
                    if sad < best_sad {
                        best_sad = sad;
                        center = mv;
                        improved = true;
                    }
                }
            }

            if !improved {
                break;
            }
        }

        center
    }

    /// Hexagon search motion estimation.
    #[allow(clippy::too_many_arguments)]
    fn hexagon_search(
        &self,
        cur_frame: &[u8],
        cur_stride: usize,
        ref_frame: &[u8],
        ref_stride: usize,
        mb_x: usize,
        mb_y: usize,
        width: usize,
        height: usize,
    ) -> MotionVector {
        const HEXAGON: [(i16, i16); 7] =
            [(0, 0), (-2, 0), (-1, -2), (1, -2), (2, 0), (1, 2), (-1, 2)];

        let mut center = MotionVector::zero();
        let mut best_sad = self
            .calculate_sad(
                cur_frame, cur_stride, ref_frame, ref_stride, mb_x, mb_y, center, width, height,
            )
            .unwrap_or(u32::MAX);

        loop {
            let mut improved = false;

            for &(dx, dy) in &HEXAGON {
                let mv = MotionVector::new(center.x + dx * 2, center.y + dy * 2);
                if let Ok(sad) = self.calculate_sad(
                    cur_frame, cur_stride, ref_frame, ref_stride, mb_x, mb_y, mv, width, height,
                ) {
                    if sad < best_sad {
                        best_sad = sad;
                        center = mv;
                        improved = true;
                    }
                }
            }

            if !improved {
                break;
            }
        }

        center
    }

    /// Three-step search motion estimation.
    #[allow(clippy::too_many_arguments)]
    fn three_step_search(
        &self,
        cur_frame: &[u8],
        cur_stride: usize,
        ref_frame: &[u8],
        ref_stride: usize,
        mb_x: usize,
        mb_y: usize,
        width: usize,
        height: usize,
    ) -> MotionVector {
        let mut center = MotionVector::zero();
        let mut best_sad = self
            .calculate_sad(
                cur_frame, cur_stride, ref_frame, ref_stride, mb_x, mb_y, center, width, height,
            )
            .unwrap_or(u32::MAX);

        let mut step = self.search_range.max(4);

        while step >= 1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }

                    let mv = MotionVector::new(center.x + dx * step * 2, center.y + dy * step * 2);
                    if let Ok(sad) = self.calculate_sad(
                        cur_frame, cur_stride, ref_frame, ref_stride, mb_x, mb_y, mv, width, height,
                    ) {
                        if sad < best_sad {
                            best_sad = sad;
                            center = mv;
                        }
                    }
                }
            }

            step /= 2;
        }

        center
    }

    /// Calculate Sum of Absolute Differences (SAD).
    #[allow(clippy::too_many_arguments)]
    fn calculate_sad(
        &self,
        cur_frame: &[u8],
        cur_stride: usize,
        ref_frame: &[u8],
        ref_stride: usize,
        mb_x: usize,
        mb_y: usize,
        mv: MotionVector,
        width: usize,
        height: usize,
    ) -> Result<u32, CodecError> {
        let mb_size = 16;
        let ref_x = (mb_x * mb_size) as i32 + (mv.x / 2) as i32;
        let ref_y = (mb_y * mb_size) as i32 + (mv.y / 2) as i32;

        // Check bounds
        if ref_x < 0
            || ref_y < 0
            || ref_x + mb_size as i32 > width as i32
            || ref_y + mb_size as i32 > height as i32
        {
            return Err(CodecError::InvalidData("MV out of bounds".into()));
        }

        let mut sad = 0u32;

        for y in 0..mb_size {
            for x in 0..mb_size {
                let cur_offset = (mb_y * mb_size + y) * cur_stride + mb_x * mb_size + x;
                let ref_offset = ((ref_y as usize) + y) * ref_stride + (ref_x as usize) + x;

                if cur_offset < cur_frame.len() && ref_offset < ref_frame.len() {
                    let diff = (cur_frame[cur_offset] as i32) - (ref_frame[ref_offset] as i32);
                    sad += diff.unsigned_abs();
                }
            }
        }

        Ok(sad)
    }
}

/// Loop filter (deblocking) for H.263.
pub struct LoopFilter {
    /// Filter strength.
    strength: u8,
}

impl LoopFilter {
    /// Create a new loop filter.
    ///
    /// # Arguments
    ///
    /// * `strength` - Filter strength (0-31)
    #[must_use]
    pub const fn new(strength: u8) -> Self {
        Self { strength }
    }

    /// Apply deblocking filter to a macroblock boundary.
    ///
    /// # Arguments
    ///
    /// * `frame` - Frame data
    /// * `stride` - Frame stride
    /// * `mb_x` - Macroblock X position
    /// * `mb_y` - Macroblock Y position
    /// * `horizontal` - True for horizontal edge, false for vertical
    pub fn filter_mb_edge(
        &self,
        frame: &mut [u8],
        stride: usize,
        mb_x: usize,
        mb_y: usize,
        horizontal: bool,
    ) {
        let mb_size = 16;

        if horizontal && mb_y > 0 {
            // Filter horizontal edge at top of macroblock
            let y = mb_y * mb_size;
            for x in 0..mb_size {
                let offset = y * stride + mb_x * mb_size + x;
                self.filter_edge_vertical(frame, offset, stride);
            }
        }

        if !horizontal && mb_x > 0 {
            // Filter vertical edge at left of macroblock
            let x = mb_x * mb_size;
            for y in 0..mb_size {
                let offset = (mb_y * mb_size + y) * stride + x;
                self.filter_edge_horizontal(frame, offset);
            }
        }
    }

    /// Filter a vertical edge (pixels above and below).
    fn filter_edge_vertical(&self, frame: &mut [u8], offset: usize, stride: usize) {
        if offset < stride || offset >= frame.len() {
            return;
        }

        let p1 = frame[offset - stride] as i16;
        let p0 = frame[offset] as i16;

        let delta = ((p1 - p0) * self.strength as i16) / 32;
        let delta = delta.clamp(-128, 127);

        frame[offset - stride] = (p1 - delta / 2).clamp(0, 255) as u8;
        frame[offset] = (p0 + delta / 2).clamp(0, 255) as u8;
    }

    /// Filter a horizontal edge (pixels left and right).
    fn filter_edge_horizontal(&self, frame: &mut [u8], offset: usize) {
        if offset == 0 || offset >= frame.len() {
            return;
        }

        let p1 = frame[offset - 1] as i16;
        let p0 = frame[offset] as i16;

        let delta = ((p1 - p0) * self.strength as i16) / 32;
        let delta = delta.clamp(-128, 127);

        frame[offset - 1] = (p1 - delta / 2).clamp(0, 255) as u8;
        frame[offset] = (p0 + delta / 2).clamp(0, 255) as u8;
    }
}
