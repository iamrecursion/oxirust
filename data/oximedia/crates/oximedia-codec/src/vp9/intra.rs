//! VP9 Intra prediction types and functions.
//!
//! This module provides intra-prediction modes and context structures for
//! VP9 decoding. Intra prediction uses neighboring pixels from the current
//! frame to predict block values without reference frames.
//!
//! VP9 supports 10 intra prediction modes:
//! - DC: Average of above and left neighbors
//! - V (Vertical): Extend above neighbors downward
//! - H (Horizontal): Extend left neighbors rightward
//! - D45-D207: Directional modes at various angles
//! - TM (True Motion): Gradient-based prediction

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::similar_names)]
#![allow(clippy::fn_params_excessive_bools)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::bool_to_int_with_if)]
#![allow(clippy::comparison_chain)]

use super::partition::{BlockSize, TxSize};

/// Number of intra prediction modes.
pub const INTRA_MODES: usize = 10;

/// Number of directional intra modes.
pub const DIRECTIONAL_MODES: usize = 8;

/// Maximum block size for intra prediction.
pub const MAX_INTRA_SIZE: usize = 32;

/// Intra prediction mode.
///
/// VP9 supports 10 intra prediction modes including DC, directional,
/// and true motion prediction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[repr(u8)]
pub enum IntraMode {
    /// DC prediction: average of above and left neighbors.
    #[default]
    Dc = 0,
    /// Vertical prediction: extend above neighbors downward.
    V = 1,
    /// Horizontal prediction: extend left neighbors rightward.
    H = 2,
    /// Diagonal 45 degree prediction (up-right).
    D45 = 3,
    /// Diagonal 135 degree prediction (up-left).
    D135 = 4,
    /// Diagonal 117 degree prediction.
    D117 = 5,
    /// Diagonal 153 degree prediction.
    D153 = 6,
    /// Diagonal 207 degree prediction (down-left).
    D207 = 7,
    /// Diagonal 63 degree prediction (steep up-right).
    D63 = 8,
    /// True motion prediction: gradient-based.
    Tm = 9,
}

impl IntraMode {
    /// All intra modes in order.
    pub const ALL: [IntraMode; INTRA_MODES] = [
        IntraMode::Dc,
        IntraMode::V,
        IntraMode::H,
        IntraMode::D45,
        IntraMode::D135,
        IntraMode::D117,
        IntraMode::D153,
        IntraMode::D207,
        IntraMode::D63,
        IntraMode::Tm,
    ];

    /// Directional modes only.
    pub const DIRECTIONAL: [IntraMode; DIRECTIONAL_MODES] = [
        IntraMode::V,
        IntraMode::H,
        IntraMode::D45,
        IntraMode::D135,
        IntraMode::D117,
        IntraMode::D153,
        IntraMode::D207,
        IntraMode::D63,
    ];

    /// Converts from u8 value to `IntraMode`.
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Dc),
            1 => Some(Self::V),
            2 => Some(Self::H),
            3 => Some(Self::D45),
            4 => Some(Self::D135),
            5 => Some(Self::D117),
            6 => Some(Self::D153),
            7 => Some(Self::D207),
            8 => Some(Self::D63),
            9 => Some(Self::Tm),
            _ => None,
        }
    }

    /// Returns the index of this mode.
    #[must_use]
    pub const fn index(&self) -> usize {
        *self as usize
    }

    /// Returns true if this is DC prediction.
    #[must_use]
    pub const fn is_dc(&self) -> bool {
        matches!(self, Self::Dc)
    }

    /// Returns true if this is a directional mode.
    #[must_use]
    pub const fn is_directional(&self) -> bool {
        matches!(
            self,
            Self::V
                | Self::H
                | Self::D45
                | Self::D135
                | Self::D117
                | Self::D153
                | Self::D207
                | Self::D63
        )
    }

    /// Returns true if this is true motion prediction.
    #[must_use]
    pub const fn is_tm(&self) -> bool {
        matches!(self, Self::Tm)
    }

    /// Returns true if this mode requires above neighbors.
    #[must_use]
    pub const fn needs_above(&self) -> bool {
        matches!(
            self,
            Self::Dc
                | Self::V
                | Self::D45
                | Self::D135
                | Self::D117
                | Self::D153
                | Self::D63
                | Self::Tm
        )
    }

    /// Returns true if this mode requires left neighbors.
    #[must_use]
    pub const fn needs_left(&self) -> bool {
        matches!(
            self,
            Self::Dc | Self::H | Self::D135 | Self::D117 | Self::D153 | Self::D207 | Self::Tm
        )
    }

    /// Returns true if this mode requires the above-left corner pixel.
    #[must_use]
    pub const fn needs_above_left(&self) -> bool {
        matches!(self, Self::D135 | Self::D117 | Self::D153 | Self::Tm)
    }

    /// Returns true if this mode requires above-right neighbors.
    #[must_use]
    pub const fn needs_above_right(&self) -> bool {
        matches!(self, Self::D45 | Self::D63)
    }

    /// Returns true if this mode requires below-left neighbors.
    #[must_use]
    pub const fn needs_below_left(&self) -> bool {
        matches!(self, Self::D207)
    }

    /// Returns the prediction angle in degrees (for directional modes).
    ///
    /// Returns `None` for non-directional modes (DC, TM).
    #[must_use]
    pub const fn angle(&self) -> Option<u16> {
        match self {
            Self::V => Some(90),
            Self::H => Some(180),
            Self::D45 => Some(45),
            Self::D135 => Some(135),
            Self::D117 => Some(117),
            Self::D153 => Some(153),
            Self::D207 => Some(207),
            Self::D63 => Some(63),
            Self::Dc | Self::Tm => None,
        }
    }
}

impl From<IntraMode> for u8 {
    fn from(value: IntraMode) -> Self {
        value as u8
    }
}

/// Intra prediction context.
///
/// This structure holds context information and neighbor data needed
/// for intra prediction of a block.
#[derive(Clone, Debug)]
pub struct IntraPredContext {
    /// Block size.
    pub block_size: BlockSize,
    /// Transform size for prediction.
    pub tx_size: TxSize,
    /// Block row position in 4x4 units.
    pub mi_row: usize,
    /// Block column position in 4x4 units.
    pub mi_col: usize,
    /// The intra prediction mode.
    pub mode: IntraMode,
    /// Whether above neighbors are available.
    pub has_above: bool,
    /// Whether left neighbors are available.
    pub has_left: bool,
    /// Whether above-right neighbors are available.
    pub has_above_right: bool,
    /// Whether below-left neighbors are available.
    pub has_below_left: bool,
    /// Above neighbor samples (up to 64 + 1 for corner).
    above: [u8; 65],
    /// Left neighbor samples (up to 64 + 1 for corner).
    left: [u8; 65],
    /// Number of valid above samples.
    above_count: usize,
    /// Number of valid left samples.
    left_count: usize,
}

impl Default for IntraPredContext {
    fn default() -> Self {
        Self::new()
    }
}

impl IntraPredContext {
    /// Creates a new intra prediction context.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            block_size: BlockSize::Block4x4,
            tx_size: TxSize::Tx4x4,
            mi_row: 0,
            mi_col: 0,
            mode: IntraMode::Dc,
            has_above: false,
            has_left: false,
            has_above_right: false,
            has_below_left: false,
            above: [128; 65],
            left: [128; 65],
            above_count: 0,
            left_count: 0,
        }
    }

    /// Creates a context for a specific block and transform size.
    #[must_use]
    pub fn with_sizes(block_size: BlockSize, tx_size: TxSize) -> Self {
        Self {
            block_size,
            tx_size,
            ..Self::new()
        }
    }

    /// Sets the block position.
    pub fn set_position(&mut self, mi_row: usize, mi_col: usize) {
        self.mi_row = mi_row;
        self.mi_col = mi_col;
    }

    /// Sets the prediction mode.
    pub fn set_mode(&mut self, mode: IntraMode) {
        self.mode = mode;
    }

    /// Sets neighbor availability.
    pub fn set_availability(
        &mut self,
        has_above: bool,
        has_left: bool,
        has_above_right: bool,
        has_below_left: bool,
    ) {
        self.has_above = has_above;
        self.has_left = has_left;
        self.has_above_right = has_above_right;
        self.has_below_left = has_below_left;
    }

    /// Sets the above neighbor samples.
    pub fn set_above(&mut self, samples: &[u8]) {
        let count = samples.len().min(65);
        self.above[..count].copy_from_slice(&samples[..count]);
        self.above_count = count;
    }

    /// Sets the left neighbor samples.
    pub fn set_left(&mut self, samples: &[u8]) {
        let count = samples.len().min(65);
        self.left[..count].copy_from_slice(&samples[..count]);
        self.left_count = count;
    }

    /// Returns the above-left corner pixel.
    #[must_use]
    pub const fn above_left(&self) -> u8 {
        self.above[0]
    }

    /// Returns an above neighbor sample.
    #[must_use]
    pub const fn above(&self, index: usize) -> u8 {
        if index + 1 < 65 {
            self.above[index + 1]
        } else {
            128
        }
    }

    /// Returns a left neighbor sample.
    #[must_use]
    pub const fn left(&self, index: usize) -> u8 {
        if index + 1 < 65 {
            self.left[index + 1]
        } else {
            128
        }
    }

    /// Returns the prediction size in pixels.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.tx_size.size()
    }

    /// Returns true if all required neighbors are available for the current mode.
    #[must_use]
    pub const fn neighbors_available(&self) -> bool {
        let needs_above = self.mode.needs_above();
        let needs_left = self.mode.needs_left();
        let needs_above_right = self.mode.needs_above_right();
        let needs_below_left = self.mode.needs_below_left();

        (!needs_above || self.has_above)
            && (!needs_left || self.has_left)
            && (!needs_above_right || self.has_above_right)
            && (!needs_below_left || self.has_below_left)
    }

    /// Fills missing above samples by extending the last available sample.
    pub fn extend_above(&mut self, total_needed: usize) {
        if self.above_count < total_needed && self.above_count > 0 {
            let last = self.above[self.above_count - 1];
            for i in self.above_count..total_needed.min(65) {
                self.above[i] = last;
            }
            self.above_count = total_needed.min(65);
        }
    }

    /// Fills missing left samples by extending the last available sample.
    pub fn extend_left(&mut self, total_needed: usize) {
        if self.left_count < total_needed && self.left_count > 0 {
            let last = self.left[self.left_count - 1];
            for i in self.left_count..total_needed.min(65) {
                self.left[i] = last;
            }
            self.left_count = total_needed.min(65);
        }
    }

    /// Fills all neighbors with a default value when none are available.
    pub fn fill_unavailable(&mut self, value: u8) {
        self.above.fill(value);
        self.left.fill(value);
        self.above_count = 65;
        self.left_count = 65;
    }
}

/// Performs DC prediction for a block.
///
/// DC prediction computes the average of available above and left neighbors.
///
/// # Arguments
///
/// * `ctx` - The intra prediction context with neighbor data
/// * `output` - Output buffer to write predicted samples
/// * `stride` - Output buffer stride (bytes between rows)
#[allow(clippy::cast_possible_truncation)]
pub fn predict_dc(ctx: &IntraPredContext, output: &mut [u8], stride: usize) {
    let size = ctx.size();
    let mut sum: u32 = 0;
    let mut count: u32 = 0;

    // Sum above neighbors
    if ctx.has_above {
        for i in 0..size {
            sum += u32::from(ctx.above(i));
        }
        count += size as u32;
    }

    // Sum left neighbors
    if ctx.has_left {
        for i in 0..size {
            sum += u32::from(ctx.left(i));
        }
        count += size as u32;
    }

    // Compute average
    let dc_value = count
        .checked_div(2)
        .and_then(|half| (sum + half).checked_div(count))
        .map_or(128u8, |v| v as u8);

    // Fill the block with the DC value
    for row in 0..size {
        let row_start = row * stride;
        for col in 0..size {
            output[row_start + col] = dc_value;
        }
    }
}

/// Performs vertical prediction for a block.
///
/// Vertical prediction extends above neighbors downward.
///
/// # Arguments
///
/// * `ctx` - The intra prediction context with neighbor data
/// * `output` - Output buffer to write predicted samples
/// * `stride` - Output buffer stride (bytes between rows)
pub fn predict_vertical(ctx: &IntraPredContext, output: &mut [u8], stride: usize) {
    let size = ctx.size();

    for row in 0..size {
        let row_start = row * stride;
        for col in 0..size {
            output[row_start + col] = ctx.above(col);
        }
    }
}

/// Performs horizontal prediction for a block.
///
/// Horizontal prediction extends left neighbors rightward.
///
/// # Arguments
///
/// * `ctx` - The intra prediction context with neighbor data
/// * `output` - Output buffer to write predicted samples
/// * `stride` - Output buffer stride (bytes between rows)
pub fn predict_horizontal(ctx: &IntraPredContext, output: &mut [u8], stride: usize) {
    let size = ctx.size();

    for row in 0..size {
        let left_val = ctx.left(row);
        let row_start = row * stride;
        for col in 0..size {
            output[row_start + col] = left_val;
        }
    }
}

/// Performs true motion (TM) prediction for a block.
///
/// TM prediction uses the formula: `pred = above + left - above_left`
/// clamped to valid pixel range.
///
/// # Arguments
///
/// * `ctx` - The intra prediction context with neighbor data
/// * `output` - Output buffer to write predicted samples
/// * `stride` - Output buffer stride (bytes between rows)
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn predict_tm(ctx: &IntraPredContext, output: &mut [u8], stride: usize) {
    let size = ctx.size();
    let top_left = i16::from(ctx.above_left());

    for row in 0..size {
        let left_val = i16::from(ctx.left(row));
        let row_start = row * stride;
        for col in 0..size {
            let above_val = i16::from(ctx.above(col));
            let pred = left_val + above_val - top_left;
            output[row_start + col] = pred.clamp(0, 255) as u8;
        }
    }
}

/// Performs D45 (diagonal 45 degree) prediction for a block.
///
/// D45 predicts samples along 45-degree lines from the upper-right.
///
/// # Arguments
///
/// * `ctx` - The intra prediction context with neighbor data
/// * `output` - Output buffer to write predicted samples
/// * `stride` - Output buffer stride (bytes between rows)
#[allow(clippy::cast_possible_truncation)]
pub fn predict_d45(ctx: &IntraPredContext, output: &mut [u8], stride: usize) {
    let size = ctx.size();

    for row in 0..size {
        let row_start = row * stride;
        for col in 0..size {
            let idx = row + col;
            if idx + 1 < size * 2 {
                // Average of two above samples
                let a = i16::from(ctx.above(idx));
                let b = i16::from(ctx.above(idx + 1));
                output[row_start + col] = ((a + b + 1) >> 1) as u8;
            } else {
                output[row_start + col] = ctx.above(size - 1);
            }
        }
    }
}

/// Performs D135 (diagonal 135 degree) prediction for a block.
///
/// D135 predicts samples along 135-degree lines from the upper-left.
///
/// # Arguments
///
/// * `ctx` - The intra prediction context with neighbor data
/// * `output` - Output buffer to write predicted samples
/// * `stride` - Output buffer stride (bytes between rows)
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn predict_d135(ctx: &IntraPredContext, output: &mut [u8], stride: usize) {
    let size = ctx.size();

    for row in 0..size {
        let row_start = row * stride;
        for col in 0..size {
            let diff = col as i32 - row as i32;

            let pred = if diff > 0 {
                // Use above samples
                let idx = (diff - 1) as usize;
                if idx + 1 < size {
                    let a = i16::from(ctx.above(idx));
                    let b = i16::from(ctx.above(idx + 1));
                    ((a + b + 1) >> 1) as u8
                } else {
                    ctx.above(size - 1)
                }
            } else if diff == 0 {
                ctx.above_left()
            } else {
                // Use left samples
                let idx = ((-diff) - 1) as usize;
                if idx + 1 < size {
                    let a = i16::from(ctx.left(idx));
                    let b = i16::from(ctx.left(idx + 1));
                    ((a + b + 1) >> 1) as u8
                } else {
                    ctx.left(size - 1)
                }
            };

            output[row_start + col] = pred;
        }
    }
}

/// Performs D117 prediction for a block.
///
/// D117 predicts samples along 117-degree lines (close to vertical).
///
/// # Arguments
///
/// * `ctx` - The intra prediction context with neighbor data
/// * `output` - Output buffer to write predicted samples
/// * `stride` - Output buffer stride (bytes between rows)
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn predict_d117(ctx: &IntraPredContext, output: &mut [u8], stride: usize) {
    let size = ctx.size();

    for row in 0..size {
        let row_start = row * stride;
        for col in 0..size {
            // D117: approximate formula
            let idx = col as i32 - row as i32 / 2;

            let pred = if idx >= 0 {
                let a = i16::from(ctx.above(idx as usize));
                let b = i16::from(ctx.above((idx + 1) as usize));
                ((a + b + 1) >> 1) as u8
            } else {
                let left_idx = ((-idx) * 2 - 1) as usize;
                if left_idx < size {
                    ctx.left(left_idx)
                } else {
                    ctx.left(size - 1)
                }
            };

            output[row_start + col] = pred;
        }
    }
}

/// Performs D153 prediction for a block.
///
/// D153 predicts samples along 153-degree lines (close to horizontal).
///
/// # Arguments
///
/// * `ctx` - The intra prediction context with neighbor data
/// * `output` - Output buffer to write predicted samples
/// * `stride` - Output buffer stride (bytes between rows)
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn predict_d153(ctx: &IntraPredContext, output: &mut [u8], stride: usize) {
    let size = ctx.size();

    for row in 0..size {
        let row_start = row * stride;
        for col in 0..size {
            // D153: approximate formula
            let idx = row as i32 - col as i32 / 2;

            let pred = if idx >= 0 {
                let a = i16::from(ctx.left(idx as usize));
                let b = i16::from(ctx.left((idx + 1) as usize));
                ((a + b + 1) >> 1) as u8
            } else {
                let above_idx = ((-idx) * 2 - 1) as usize;
                if above_idx < size {
                    ctx.above(above_idx)
                } else {
                    ctx.above(size - 1)
                }
            };

            output[row_start + col] = pred;
        }
    }
}

/// Performs D207 prediction for a block.
///
/// D207 predicts samples along 207-degree lines (down-left).
///
/// # Arguments
///
/// * `ctx` - The intra prediction context with neighbor data
/// * `output` - Output buffer to write predicted samples
/// * `stride` - Output buffer stride (bytes between rows)
#[allow(clippy::cast_possible_truncation)]
pub fn predict_d207(ctx: &IntraPredContext, output: &mut [u8], stride: usize) {
    let size = ctx.size();

    for row in 0..size {
        let row_start = row * stride;
        for col in 0..size {
            let idx = row + col;
            if idx + 1 < size * 2 {
                let a = i16::from(ctx.left(idx));
                let b = i16::from(ctx.left(idx + 1));
                output[row_start + col] = ((a + b + 1) >> 1) as u8;
            } else {
                output[row_start + col] = ctx.left(size - 1);
            }
        }
    }
}

/// Performs D63 prediction for a block.
///
/// D63 predicts samples along 63-degree lines (steep up-right).
///
/// # Arguments
///
/// * `ctx` - The intra prediction context with neighbor data
/// * `output` - Output buffer to write predicted samples
/// * `stride` - Output buffer stride (bytes between rows)
#[allow(clippy::cast_possible_truncation)]
pub fn predict_d63(ctx: &IntraPredContext, output: &mut [u8], stride: usize) {
    let size = ctx.size();

    for row in 0..size {
        let row_start = row * stride;
        for col in 0..size {
            // D63: 2:1 slope (approximately 63 degrees)
            let idx = col + row / 2;

            if idx + 1 < size * 2 {
                let a = i16::from(ctx.above(idx));
                let b = i16::from(ctx.above(idx + 1));
                output[row_start + col] = ((a + b + 1) >> 1) as u8;
            } else {
                output[row_start + col] = ctx.above(size * 2 - 2);
            }
        }
    }
}

/// Applies intra prediction for a block.
///
/// This function dispatches to the appropriate prediction function based
/// on the mode specified in the context.
///
/// # Arguments
///
/// * `ctx` - The intra prediction context with neighbor data and mode
/// * `output` - Output buffer to write predicted samples
/// * `stride` - Output buffer stride (bytes between rows)
pub fn apply_intra_prediction(ctx: &IntraPredContext, output: &mut [u8], stride: usize) {
    match ctx.mode {
        IntraMode::Dc => predict_dc(ctx, output, stride),
        IntraMode::V => predict_vertical(ctx, output, stride),
        IntraMode::H => predict_horizontal(ctx, output, stride),
        IntraMode::D45 => predict_d45(ctx, output, stride),
        IntraMode::D135 => predict_d135(ctx, output, stride),
        IntraMode::D117 => predict_d117(ctx, output, stride),
        IntraMode::D153 => predict_d153(ctx, output, stride),
        IntraMode::D207 => predict_d207(ctx, output, stride),
        IntraMode::D63 => predict_d63(ctx, output, stride),
        IntraMode::Tm => predict_tm(ctx, output, stride),
    }
}

/// Intra mode context for probability selection.
#[derive(Clone, Copy, Debug, Default)]
pub struct IntraModeContext {
    /// Context for y-mode selection.
    pub y_mode_context: u8,
    /// Context for uv-mode selection.
    pub uv_mode_context: u8,
}

impl IntraModeContext {
    /// Creates a new intra mode context.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            y_mode_context: 0,
            uv_mode_context: 0,
        }
    }

    /// Creates context from above and left neighbor modes.
    #[must_use]
    pub fn from_neighbors(above_mode: Option<IntraMode>, left_mode: Option<IntraMode>) -> Self {
        // Y-mode context is based on neighboring intra modes
        let y_mode_context = match (above_mode, left_mode) {
            (Some(a), Some(l)) => {
                let above_ctx = if a.is_directional() { 1 } else { 0 };
                let left_ctx = if l.is_directional() { 1 } else { 0 };
                above_ctx + left_ctx
            }
            (Some(a), None) | (None, Some(a)) => {
                if a.is_directional() {
                    1
                } else {
                    0
                }
            }
            (None, None) => 0,
        };

        Self {
            y_mode_context,
            uv_mode_context: 0,
        }
    }

    /// Sets the UV mode context based on the Y mode.
    pub fn set_uv_context(&mut self, y_mode: IntraMode) {
        self.uv_mode_context = y_mode as u8;
    }
}

/// Sub-block intra mode (for 4x4 blocks within 8x8).
#[derive(Clone, Copy, Debug, Default)]
pub struct SubBlockModes {
    /// Modes for each 4x4 sub-block (in raster scan order).
    pub modes: [IntraMode; 4],
}

impl SubBlockModes {
    /// Creates new sub-block modes with all DC prediction.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            modes: [IntraMode::Dc; 4],
        }
    }

    /// Sets the mode for a specific sub-block.
    pub fn set(&mut self, index: usize, mode: IntraMode) {
        if index < 4 {
            self.modes[index] = mode;
        }
    }

    /// Gets the mode for a specific sub-block.
    #[must_use]
    pub const fn get(&self, index: usize) -> IntraMode {
        if index < 4 {
            self.modes[index]
        } else {
            IntraMode::Dc
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intra_mode() {
        assert_eq!(IntraMode::Dc.index(), 0);
        assert_eq!(IntraMode::Tm.index(), 9);

        assert!(IntraMode::Dc.is_dc());
        assert!(!IntraMode::V.is_dc());

        assert!(IntraMode::V.is_directional());
        assert!(IntraMode::D45.is_directional());
        assert!(!IntraMode::Dc.is_directional());
        assert!(!IntraMode::Tm.is_directional());

        assert!(IntraMode::Tm.is_tm());
        assert!(!IntraMode::Dc.is_tm());
    }

    #[test]
    fn test_intra_mode_from_u8() {
        assert_eq!(IntraMode::from_u8(0), Some(IntraMode::Dc));
        assert_eq!(IntraMode::from_u8(9), Some(IntraMode::Tm));
        assert_eq!(IntraMode::from_u8(10), None);
    }

    #[test]
    fn test_intra_mode_needs() {
        // DC needs both above and left
        assert!(IntraMode::Dc.needs_above());
        assert!(IntraMode::Dc.needs_left());

        // V needs only above
        assert!(IntraMode::V.needs_above());
        assert!(!IntraMode::V.needs_left());

        // H needs only left
        assert!(!IntraMode::H.needs_above());
        assert!(IntraMode::H.needs_left());

        // D45 needs above and above-right
        assert!(IntraMode::D45.needs_above());
        assert!(IntraMode::D45.needs_above_right());

        // TM needs above, left, and above-left
        assert!(IntraMode::Tm.needs_above());
        assert!(IntraMode::Tm.needs_left());
        assert!(IntraMode::Tm.needs_above_left());
    }

    #[test]
    fn test_intra_mode_angle() {
        assert_eq!(IntraMode::V.angle(), Some(90));
        assert_eq!(IntraMode::H.angle(), Some(180));
        assert_eq!(IntraMode::D45.angle(), Some(45));
        assert_eq!(IntraMode::Dc.angle(), None);
        assert_eq!(IntraMode::Tm.angle(), None);
    }

    #[test]
    fn test_intra_pred_context_new() {
        let ctx = IntraPredContext::new();
        assert_eq!(ctx.block_size, BlockSize::Block4x4);
        assert_eq!(ctx.tx_size, TxSize::Tx4x4);
        assert_eq!(ctx.mode, IntraMode::Dc);
        assert!(!ctx.has_above);
        assert!(!ctx.has_left);
    }

    #[test]
    fn test_intra_pred_context_neighbors() {
        let mut ctx = IntraPredContext::new();

        let above = [100, 110, 120, 130, 140];
        let left = [90, 95, 100, 105, 110];

        ctx.set_above(&above);
        ctx.set_left(&left);

        assert_eq!(ctx.above_left(), 100);
        assert_eq!(ctx.above(0), 110);
        assert_eq!(ctx.above(3), 140);
        assert_eq!(ctx.left(0), 95);
        assert_eq!(ctx.left(3), 110);
    }

    #[test]
    fn test_predict_dc() {
        let mut ctx = IntraPredContext::with_sizes(BlockSize::Block4x4, TxSize::Tx4x4);
        ctx.has_above = true;
        ctx.has_left = true;

        // Set neighbors
        ctx.set_above(&[0, 100, 100, 100, 100]);
        ctx.set_left(&[0, 100, 100, 100, 100]);

        let mut output = [0u8; 16];
        predict_dc(&ctx, &mut output, 4);

        // All values should be 100 (average of 100s)
        for val in &output {
            assert_eq!(*val, 100);
        }
    }

    #[test]
    fn test_predict_vertical() {
        let mut ctx = IntraPredContext::with_sizes(BlockSize::Block4x4, TxSize::Tx4x4);
        ctx.has_above = true;

        ctx.set_above(&[0, 10, 20, 30, 40]);

        let mut output = [0u8; 16];
        predict_vertical(&ctx, &mut output, 4);

        // Each column should match the corresponding above value
        for row in 0..4 {
            assert_eq!(output[row * 4], 10);
            assert_eq!(output[row * 4 + 1], 20);
            assert_eq!(output[row * 4 + 2], 30);
            assert_eq!(output[row * 4 + 3], 40);
        }
    }

    #[test]
    fn test_predict_horizontal() {
        let mut ctx = IntraPredContext::with_sizes(BlockSize::Block4x4, TxSize::Tx4x4);
        ctx.has_left = true;

        ctx.set_left(&[0, 10, 20, 30, 40]);

        let mut output = [0u8; 16];
        predict_horizontal(&ctx, &mut output, 4);

        // Each row should have the same value as corresponding left sample
        for col in 0..4 {
            assert_eq!(output[col], 10);
            assert_eq!(output[4 + col], 20);
            assert_eq!(output[8 + col], 30);
            assert_eq!(output[12 + col], 40);
        }
    }

    #[test]
    fn test_predict_tm() {
        let mut ctx = IntraPredContext::with_sizes(BlockSize::Block4x4, TxSize::Tx4x4);
        ctx.has_above = true;
        ctx.has_left = true;

        // above_left = 100, above = [110, 120, 130, 140], left = [105, 115, 125, 135]
        ctx.set_above(&[100, 110, 120, 130, 140]);
        ctx.set_left(&[100, 105, 115, 125, 135]);

        let mut output = [0u8; 16];
        predict_tm(&ctx, &mut output, 4);

        // pred[r][c] = left[r] + above[c] - above_left
        // pred[0][0] = 105 + 110 - 100 = 115
        assert_eq!(output[0], 115);
        // pred[0][1] = 105 + 120 - 100 = 125
        assert_eq!(output[1], 125);
    }

    #[test]
    fn test_apply_intra_prediction() {
        let mut ctx = IntraPredContext::with_sizes(BlockSize::Block4x4, TxSize::Tx4x4);
        ctx.has_above = true;
        ctx.set_above(&[0, 50, 50, 50, 50]);

        let mut output = [0u8; 16];

        ctx.set_mode(IntraMode::V);
        apply_intra_prediction(&ctx, &mut output, 4);

        for row in 0..4 {
            for col in 0..4 {
                assert_eq!(output[row * 4 + col], 50);
            }
        }
    }

    #[test]
    fn test_intra_mode_context() {
        let ctx = IntraModeContext::from_neighbors(Some(IntraMode::V), Some(IntraMode::H));
        assert_eq!(ctx.y_mode_context, 2);

        let ctx2 = IntraModeContext::from_neighbors(Some(IntraMode::Dc), Some(IntraMode::Dc));
        assert_eq!(ctx2.y_mode_context, 0);
    }

    #[test]
    fn test_sub_block_modes() {
        let mut modes = SubBlockModes::new();
        modes.set(0, IntraMode::V);
        modes.set(1, IntraMode::H);
        modes.set(2, IntraMode::D45);
        modes.set(3, IntraMode::Tm);

        assert_eq!(modes.get(0), IntraMode::V);
        assert_eq!(modes.get(1), IntraMode::H);
        assert_eq!(modes.get(2), IntraMode::D45);
        assert_eq!(modes.get(3), IntraMode::Tm);
        assert_eq!(modes.get(4), IntraMode::Dc); // Out of bounds
    }

    #[test]
    fn test_intra_pred_context_extend() {
        let mut ctx = IntraPredContext::new();
        ctx.set_above(&[128, 100, 110]);
        ctx.extend_above(5);

        // Should extend with last value (110)
        assert_eq!(ctx.above(3), 110);
    }

    #[test]
    fn test_intra_pred_context_fill_unavailable() {
        let mut ctx = IntraPredContext::new();
        ctx.fill_unavailable(128);

        assert_eq!(ctx.above_left(), 128);
        assert_eq!(ctx.above(0), 128);
        assert_eq!(ctx.left(0), 128);
    }
}
