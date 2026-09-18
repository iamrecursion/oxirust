//! AV1 transform-coefficient decoding for intra frames.
//!
//! Exact port of the `coeffs()` syntax structure (spec 5.11.39), the
//! `transform_type()`/`compute_tx_type()`/`get_tx_set()`/`get_scan()`
//! functions (spec 5.11.40/47/48/41), and the CDF selection contexts of
//! spec 8.3.2 (`all_zero`, `eob_pt_*`, `eob_extra`, `coeff_base`,
//! `coeff_base_eob`, `coeff_br`, `dc_sign`).

use super::cdfs::CdfCtx;
use super::consts::{
    BR_CDF_SIZE, COEFF_BASE_RANGE, DCT_DCT, H_ADST, H_DCT, H_FLIPADST, IDTX, NUM_BASE_LEVELS,
    SIG_COEF_CONTEXTS, SIG_COEF_CONTEXTS_2D, SIG_COEF_CONTEXTS_EOB, TX_16X16, TX_16X32, TX_16X4,
    TX_16X64, TX_16X8, TX_32X16, TX_32X32, TX_32X8, TX_4X16, TX_4X4, TX_4X8, TX_64X16, TX_64X64,
    TX_8X16, TX_8X32, TX_8X4, TX_8X8, TX_CLASS_HORIZ, TX_CLASS_VERT, TX_SET_DCTONLY,
    TX_SET_INTRA_1, V_ADST, V_DCT, V_FLIPADST,
};
use super::msac::Msac;
use super::tables_conv::{
    ADJUSTED_TX_SIZE, MODE_TO_TXFM, SIG_REF_DIFF_OFFSET, SUBSAMPLED_SIZE, TX_HEIGHT,
    TX_HEIGHT_LOG2, TX_SIZE_SQR, TX_SIZE_SQR_UP, TX_TYPE_IN_SET_INTRA, TX_WIDTH, TX_WIDTH_LOG2,
};
use super::tables_scan as scan_t;

/// `Coeff_Base_Ctx_Offset[ TX_SIZES_ALL ][ 5 ][ 5 ]` (spec 8.3.2, coeff_base).
#[rustfmt::skip]
const COEFF_BASE_CTX_OFFSET: [[[u8; 5]; 5]; 19] = [
    [[0, 1, 6, 6, 0], [1, 6, 6, 21, 0], [6, 6, 21, 21, 0], [6, 21, 21, 21, 0], [0, 0, 0, 0, 0]],
    [[0, 1, 6, 6, 21], [1, 6, 6, 21, 21], [6, 6, 21, 21, 21], [6, 21, 21, 21, 21], [21, 21, 21, 21, 21]],
    [[0, 1, 6, 6, 21], [1, 6, 6, 21, 21], [6, 6, 21, 21, 21], [6, 21, 21, 21, 21], [21, 21, 21, 21, 21]],
    [[0, 1, 6, 6, 21], [1, 6, 6, 21, 21], [6, 6, 21, 21, 21], [6, 21, 21, 21, 21], [21, 21, 21, 21, 21]],
    [[0, 1, 6, 6, 21], [1, 6, 6, 21, 21], [6, 6, 21, 21, 21], [6, 21, 21, 21, 21], [21, 21, 21, 21, 21]],
    [[0, 11, 11, 11, 0], [11, 11, 11, 11, 0], [6, 6, 21, 21, 0], [6, 21, 21, 21, 0], [21, 21, 21, 21, 0]],
    [[0, 16, 6, 6, 21], [16, 16, 6, 21, 21], [16, 16, 21, 21, 21], [16, 16, 21, 21, 21], [0, 0, 0, 0, 0]],
    [[0, 11, 11, 11, 11], [11, 11, 11, 11, 11], [6, 6, 21, 21, 21], [6, 21, 21, 21, 21], [21, 21, 21, 21, 21]],
    [[0, 16, 6, 6, 21], [16, 16, 6, 21, 21], [16, 16, 21, 21, 21], [16, 16, 21, 21, 21], [16, 16, 21, 21, 21]],
    [[0, 11, 11, 11, 11], [11, 11, 11, 11, 11], [6, 6, 21, 21, 21], [6, 21, 21, 21, 21], [21, 21, 21, 21, 21]],
    [[0, 16, 6, 6, 21], [16, 16, 6, 21, 21], [16, 16, 21, 21, 21], [16, 16, 21, 21, 21], [16, 16, 21, 21, 21]],
    [[0, 11, 11, 11, 11], [11, 11, 11, 11, 11], [6, 6, 21, 21, 21], [6, 21, 21, 21, 21], [21, 21, 21, 21, 21]],
    [[0, 16, 6, 6, 21], [16, 16, 6, 21, 21], [16, 16, 21, 21, 21], [16, 16, 21, 21, 21], [16, 16, 21, 21, 21]],
    [[0, 11, 11, 11, 0], [11, 11, 11, 11, 0], [6, 6, 21, 21, 0], [6, 21, 21, 21, 0], [21, 21, 21, 21, 0]],
    [[0, 16, 6, 6, 21], [16, 16, 6, 21, 21], [16, 16, 21, 21, 21], [16, 16, 21, 21, 21], [0, 0, 0, 0, 0]],
    [[0, 11, 11, 11, 11], [11, 11, 11, 11, 11], [6, 6, 21, 21, 21], [6, 21, 21, 21, 21], [21, 21, 21, 21, 21]],
    [[0, 16, 6, 6, 21], [16, 16, 6, 21, 21], [16, 16, 21, 21, 21], [16, 16, 21, 21, 21], [16, 16, 21, 21, 21]],
    [[0, 11, 11, 11, 11], [11, 11, 11, 11, 11], [6, 6, 21, 21, 21], [6, 21, 21, 21, 21], [21, 21, 21, 21, 21]],
    [[0, 16, 6, 6, 21], [16, 16, 6, 21, 21], [16, 16, 21, 21, 21], [16, 16, 21, 21, 21], [16, 16, 21, 21, 21]],
];

/// `Coeff_Base_Pos_Ctx_Offset[ 3 ]` (spec 8.3.2).
const COEFF_BASE_POS_CTX_OFFSET: [usize; 3] = [
    SIG_COEF_CONTEXTS_2D,
    SIG_COEF_CONTEXTS_2D + 5,
    SIG_COEF_CONTEXTS_2D + 10,
];

/// `Mag_Ref_Offset_With_Tx_Class[ 3 ][ 3 ][ 2 ]` (spec 8.3.2, coeff_br).
const MAG_REF_OFFSET_WITH_TX_CLASS: [[[i32; 2]; 3]; 3] = [
    [[0, 1], [1, 0], [1, 1]],
    [[0, 1], [1, 0], [0, 2]],
    [[0, 1], [1, 0], [2, 0]],
];

/// `Filter_Intra_Mode_To_Intra_Dir` (spec 8.3.2): DC, V, H, D157, DC.
pub const FILTER_INTRA_MODE_TO_INTRA_DIR: [usize; 5] = [0, 1, 2, 6, 0];

/// Per-tile coefficient level/DC sign contexts.
pub struct LevelCtxs {
    /// `AboveLevelContext[plane][x4]` (plane-subsampled 4x4 columns).
    pub above_level: [Vec<u8>; 3],
    /// `AboveDcContext[plane][x4]`.
    pub above_dc: [Vec<u8>; 3],
    /// `LeftLevelContext[plane][y4]`.
    pub left_level: [Vec<u8>; 3],
    /// `LeftDcContext[plane][y4]`.
    pub left_dc: [Vec<u8>; 3],
}

impl LevelCtxs {
    pub fn new(mi_cols: usize, mi_rows: usize) -> Self {
        Self {
            above_level: [vec![0; mi_cols], vec![0; mi_cols], vec![0; mi_cols]],
            above_dc: [vec![0; mi_cols], vec![0; mi_cols], vec![0; mi_cols]],
            left_level: [vec![0; mi_rows], vec![0; mi_rows], vec![0; mi_rows]],
            left_dc: [vec![0; mi_rows], vec![0; mi_rows], vec![0; mi_rows]],
        }
    }

    /// `clear_above_context` (spec 7.19).
    pub fn clear_above(&mut self) {
        for p in 0..3 {
            self.above_level[p].fill(0);
            self.above_dc[p].fill(0);
        }
    }

    /// `clear_left_context` (spec 7.19).
    pub fn clear_left(&mut self) {
        for p in 0..3 {
            self.left_level[p].fill(0);
            self.left_dc[p].fill(0);
        }
    }

    /// `reset_block_context` (spec 5.11.30) for skipped blocks.
    pub fn reset_block(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        bw4: usize,
        bh4: usize,
        has_chroma: bool,
        sub_x: bool,
        sub_y: bool,
    ) {
        let nplanes = if has_chroma { 3 } else { 1 };
        for plane in 0..nplanes {
            let (sx, sy) = if plane > 0 {
                (usize::from(sub_x), usize::from(sub_y))
            } else {
                (0, 0)
            };
            for i in (mi_col >> sx)..((mi_col + bw4) >> sx) {
                if i < self.above_level[plane].len() {
                    self.above_level[plane][i] = 0;
                    self.above_dc[plane][i] = 0;
                }
            }
            for i in (mi_row >> sy)..((mi_row + bh4) >> sy) {
                if i < self.left_level[plane].len() {
                    self.left_level[plane][i] = 0;
                    self.left_dc[plane][i] = 0;
                }
            }
        }
    }
}

/// Everything `coeffs()` needs to know about the current block.
pub struct CoefBlock {
    pub plane: usize,
    /// Plane-coordinate top-left of the transform block.
    pub start_x: usize,
    pub start_y: usize,
    pub tx_sz: usize,
    /// `MiSize` of the containing block.
    pub mi_size: usize,
    pub lossless: bool,
    /// `(segmentation_enabled ? get_qindex(1, segment_id) : base_q_idx) > 0`.
    pub qindex_gt0: bool,
    pub reduced_tx_set: bool,
    /// intraDir for intra_tx_type CDF selection (YMode, or the filter-intra
    /// mapped direction when use_filter_intra).
    pub intra_dir: usize,
    /// UVMode (for chroma tx type derivation).
    pub uv_mode: usize,
    pub sub_x: bool,
    pub sub_y: bool,
    pub mi_cols: usize,
    pub mi_rows: usize,
}

/// `get_tx_class( txType )` (spec 8.3.2).
#[inline]
pub fn get_tx_class(tx_type: usize) -> usize {
    if tx_type == V_DCT || tx_type == V_ADST || tx_type == V_FLIPADST {
        TX_CLASS_VERT
    } else if tx_type == H_DCT || tx_type == H_ADST || tx_type == H_FLIPADST {
        TX_CLASS_HORIZ
    } else {
        0 // TX_CLASS_2D
    }
}

/// `get_tx_set( txSz )` (spec 5.11.48), intra variant.
pub fn get_tx_set_intra(tx_sz: usize, reduced_tx_set: bool) -> usize {
    let tx_sz_sqr = usize::from(TX_SIZE_SQR[tx_sz]);
    let tx_sz_sqr_up = usize::from(TX_SIZE_SQR_UP[tx_sz]);
    if tx_sz_sqr_up > TX_32X32 {
        return TX_SET_DCTONLY;
    }
    if tx_sz_sqr_up == TX_32X32 {
        TX_SET_DCTONLY
    } else if reduced_tx_set || tx_sz_sqr == TX_16X16 {
        2 // TX_SET_INTRA_2
    } else {
        TX_SET_INTRA_1
    }
}

/// `Tx_Type_Intra_Inv_Set1/2` (spec 5.11.47).
const TX_TYPE_INTRA_INV_SET1: [usize; 7] = [IDTX, DCT_DCT, V_DCT, H_DCT, 3, 1, 2];
const TX_TYPE_INTRA_INV_SET2: [usize; 5] = [IDTX, DCT_DCT, 3, 1, 2];

/// `transform_type( x4, y4, txSz )` (spec 5.11.47) for intra blocks; writes
/// the TxTypes grid and returns the chosen type.
#[allow(clippy::too_many_arguments)]
pub fn read_transform_type(
    msac: &mut Msac,
    cdfs: &mut CdfCtx,
    tx_types: &mut [u8],
    mi_cols: usize,
    x4: usize,
    y4: usize,
    b: &CoefBlock,
) -> usize {
    let set = get_tx_set_intra(b.tx_sz, b.reduced_tx_set);
    let tx_type = if set > 0 && b.qindex_gt0 {
        let sqr = usize::from(TX_SIZE_SQR[b.tx_sz]);
        if set == TX_SET_INTRA_1 {
            let sym = msac.read_symbol(&mut cdfs.intra_tx_type_set1[sqr][b.intra_dir]);
            TX_TYPE_INTRA_INV_SET1[sym]
        } else {
            let sym = msac.read_symbol(&mut cdfs.intra_tx_type_set2[sqr][b.intra_dir]);
            TX_TYPE_INTRA_INV_SET2[sym]
        }
    } else {
        DCT_DCT
    };
    let w4 = usize::from(TX_WIDTH[b.tx_sz]) >> 2;
    let h4 = usize::from(TX_HEIGHT[b.tx_sz]) >> 2;
    for i in 0..w4 {
        for j in 0..h4 {
            if y4 + j < b.mi_rows && x4 + i < mi_cols {
                tx_types[(y4 + j) * mi_cols + (x4 + i)] = tx_type as u8;
            }
        }
    }
    tx_type
}

/// `compute_tx_type( plane, txSz, blockX, blockY )` (spec 5.11.40), intra.
pub fn compute_tx_type(
    tx_types: &[u8],
    mi_cols: usize,
    b: &CoefBlock,
    block_x: usize,
    block_y: usize,
) -> usize {
    let tx_sz_sqr_up = usize::from(TX_SIZE_SQR_UP[b.tx_sz]);
    if b.lossless || tx_sz_sqr_up > TX_32X32 {
        return DCT_DCT;
    }
    let tx_set = get_tx_set_intra(b.tx_sz, b.reduced_tx_set);
    if b.plane == 0 {
        return usize::from(tx_types[block_y * mi_cols + block_x]);
    }
    // Intra chroma: derive from UV mode, constrained to the set.
    let tx_type = usize::from(MODE_TO_TXFM[b.uv_mode]);
    if TX_TYPE_IN_SET_INTRA[tx_set][tx_type] == 0 {
        return DCT_DCT;
    }
    tx_type
}

/// `get_scan( txSz )` (spec 5.11.41) given the plane tx type.
pub fn get_scan(tx_sz: usize, plane_tx_type: usize) -> &'static [u16] {
    if tx_sz == TX_16X64 {
        return &scan_t::DEFAULT_SCAN_16X32;
    }
    if tx_sz == TX_64X16 {
        return &scan_t::DEFAULT_SCAN_32X16;
    }
    if usize::from(TX_SIZE_SQR_UP[tx_sz]) == TX_64X64 {
        return &scan_t::DEFAULT_SCAN_32X32;
    }
    if plane_tx_type == IDTX {
        return default_scan(tx_sz);
    }
    let prefer_row =
        plane_tx_type == V_DCT || plane_tx_type == V_ADST || plane_tx_type == V_FLIPADST;
    let prefer_col =
        plane_tx_type == H_DCT || plane_tx_type == H_ADST || plane_tx_type == H_FLIPADST;
    if prefer_row {
        mrow_scan(tx_sz)
    } else if prefer_col {
        mcol_scan(tx_sz)
    } else {
        default_scan(tx_sz)
    }
}

fn default_scan(tx_sz: usize) -> &'static [u16] {
    match tx_sz {
        t if t == TX_4X4 => &scan_t::DEFAULT_SCAN_4X4,
        t if t == TX_4X8 => &scan_t::DEFAULT_SCAN_4X8,
        t if t == TX_8X4 => &scan_t::DEFAULT_SCAN_8X4,
        t if t == TX_8X8 => &scan_t::DEFAULT_SCAN_8X8,
        t if t == TX_8X16 => &scan_t::DEFAULT_SCAN_8X16,
        t if t == TX_16X8 => &scan_t::DEFAULT_SCAN_16X8,
        t if t == TX_16X16 => &scan_t::DEFAULT_SCAN_16X16,
        t if t == TX_16X32 => &scan_t::DEFAULT_SCAN_16X32,
        t if t == TX_32X16 => &scan_t::DEFAULT_SCAN_32X16,
        t if t == TX_4X16 => &scan_t::DEFAULT_SCAN_4X16,
        t if t == TX_16X4 => &scan_t::DEFAULT_SCAN_16X4,
        t if t == TX_8X32 => &scan_t::DEFAULT_SCAN_8X32,
        t if t == TX_32X8 => &scan_t::DEFAULT_SCAN_32X8,
        _ => &scan_t::DEFAULT_SCAN_32X32,
    }
}

fn mrow_scan(tx_sz: usize) -> &'static [u16] {
    match tx_sz {
        t if t == TX_4X4 => &scan_t::MROW_SCAN_4X4,
        t if t == TX_4X8 => &scan_t::MROW_SCAN_4X8,
        t if t == TX_8X4 => &scan_t::MROW_SCAN_8X4,
        t if t == TX_8X8 => &scan_t::MROW_SCAN_8X8,
        t if t == TX_8X16 => &scan_t::MROW_SCAN_8X16,
        t if t == TX_16X8 => &scan_t::MROW_SCAN_16X8,
        t if t == TX_16X16 => &scan_t::MROW_SCAN_16X16,
        t if t == TX_4X16 => &scan_t::MROW_SCAN_4X16,
        _ => &scan_t::MROW_SCAN_16X4,
    }
}

fn mcol_scan(tx_sz: usize) -> &'static [u16] {
    match tx_sz {
        t if t == TX_4X4 => &scan_t::MCOL_SCAN_4X4,
        t if t == TX_4X8 => &scan_t::MCOL_SCAN_4X8,
        t if t == TX_8X4 => &scan_t::MCOL_SCAN_8X4,
        t if t == TX_8X8 => &scan_t::MCOL_SCAN_8X8,
        t if t == TX_8X16 => &scan_t::MCOL_SCAN_8X16,
        t if t == TX_16X8 => &scan_t::MCOL_SCAN_16X8,
        t if t == TX_16X16 => &scan_t::MCOL_SCAN_16X16,
        t if t == TX_4X16 => &scan_t::MCOL_SCAN_4X16,
        _ => &scan_t::MCOL_SCAN_16X4,
    }
}

/// `get_plane_residual_size( subsize, plane )` (spec 5.11.38).
#[inline]
pub fn get_plane_residual_size(subsize: usize, plane: usize, sub_x: bool, sub_y: bool) -> usize {
    let subx = if plane > 0 { usize::from(sub_x) } else { 0 };
    let suby = if plane > 0 { usize::from(sub_y) } else { 0 };
    usize::from(SUBSAMPLED_SIZE[subsize][subx][suby])
}

/// `get_coeff_base_ctx` (spec 8.3.2).
#[allow(clippy::too_many_arguments)]
fn get_coeff_base_ctx(
    quant: &[i32],
    tx_sz: usize,
    pos: usize,
    c: usize,
    is_eob: bool,
    tx_class: usize,
) -> usize {
    let adj_tx_sz = usize::from(ADJUSTED_TX_SIZE[tx_sz]);
    let bwl = u32::from(TX_WIDTH_LOG2[adj_tx_sz]);
    let width = 1usize << bwl;
    let height = usize::from(TX_HEIGHT[adj_tx_sz]);
    if is_eob {
        if c == 0 {
            return SIG_COEF_CONTEXTS - 4;
        }
        if c <= (height << bwl) / 8 {
            return SIG_COEF_CONTEXTS - 3;
        }
        if c <= (height << bwl) / 4 {
            return SIG_COEF_CONTEXTS - 2;
        }
        return SIG_COEF_CONTEXTS - 1;
    }
    let row = pos >> bwl;
    let col = pos - (row << bwl);
    let mut mag = 0i32;
    for offsets in &SIG_REF_DIFF_OFFSET[tx_class] {
        let ref_row = row as i32 + i32::from(offsets[0]);
        let ref_col = col as i32 + i32::from(offsets[1]);
        if ref_row >= 0 && ref_col >= 0 && (ref_row as usize) < height && (ref_col as usize) < width
        {
            let q = quant[((ref_row as usize) << bwl) + ref_col as usize];
            mag += core::cmp::min(q.abs(), 3);
        }
    }
    let ctx = core::cmp::min(((mag + 1) >> 1) as usize, 4);
    if tx_class == 0 {
        // TX_CLASS_2D
        if row == 0 && col == 0 {
            return 0;
        }
        return ctx
            + usize::from(
                COEFF_BASE_CTX_OFFSET[tx_sz][core::cmp::min(row, 4)][core::cmp::min(col, 4)],
            );
    }
    let idx = if tx_class == TX_CLASS_VERT { row } else { col };
    ctx + COEFF_BASE_POS_CTX_OFFSET[core::cmp::min(idx, 2)]
}

/// coeff_br context (spec 8.3.2).
fn get_br_ctx(quant: &[i32], tx_sz: usize, pos: usize, tx_class: usize) -> usize {
    let adj_tx_sz = usize::from(ADJUSTED_TX_SIZE[tx_sz]);
    let bwl = u32::from(TX_WIDTH_LOG2[adj_tx_sz]);
    let txw = usize::from(TX_WIDTH[adj_tx_sz]);
    let txh = usize::from(TX_HEIGHT[adj_tx_sz]);
    let row = pos >> bwl;
    let col = pos - (row << bwl);
    let mut mag = 0i32;
    for offsets in &MAG_REF_OFFSET_WITH_TX_CLASS[tx_class] {
        let ref_row = row as i32 + offsets[0];
        let ref_col = col as i32 + offsets[1];
        if ref_row >= 0
            && ref_col >= 0
            && (ref_row as usize) < txh
            && (ref_col as usize) < (1usize << bwl)
        {
            let q = quant[(ref_row as usize) * txw + ref_col as usize];
            mag += core::cmp::min(q, (COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1) as i32);
        }
    }
    let mag = core::cmp::min(((mag + 1) >> 1) as usize, 6);
    if pos == 0 {
        return mag;
    }
    if tx_class == 0 {
        if row < 2 && col < 2 {
            return mag + 7;
        }
        return mag + 14;
    }
    if tx_class == TX_CLASS_HORIZ {
        if col == 0 {
            return mag + 7;
        }
        return mag + 14;
    }
    if row == 0 {
        return mag + 7;
    }
    mag + 14
}

/// `all_zero` context (spec 8.3.2).
#[allow(clippy::too_many_lines)]
fn all_zero_ctx(lc: &LevelCtxs, b: &CoefBlock, x4: usize, y4: usize) -> usize {
    let mut max_x4 = b.mi_cols;
    let mut max_y4 = b.mi_rows;
    if b.plane > 0 {
        max_x4 >>= usize::from(b.sub_x);
        max_y4 >>= usize::from(b.sub_y);
    }
    let w = usize::from(TX_WIDTH[b.tx_sz]);
    let h = usize::from(TX_HEIGHT[b.tx_sz]);
    let w4 = w >> 2;
    let h4 = h >> 2;
    let bsize = get_plane_residual_size(b.mi_size, b.plane, b.sub_x, b.sub_y);
    let bw = block_width(bsize);
    let bh = block_height(bsize);

    if b.plane == 0 {
        let mut top = 0u32;
        let mut left = 0u32;
        for k in 0..w4 {
            if x4 + k < max_x4 {
                top = top.max(u32::from(lc.above_level[0][x4 + k]));
            }
        }
        for k in 0..h4 {
            if y4 + k < max_y4 {
                left = left.max(u32::from(lc.left_level[0][y4 + k]));
            }
        }
        let top = top.min(255);
        let left = left.min(255);
        if bw == w && bh == h {
            0
        } else if top == 0 && left == 0 {
            1
        } else if top == 0 || left == 0 {
            2 + usize::from(top.max(left) > 3)
        } else if top.max(left) <= 3 {
            4
        } else if top.min(left) <= 3 {
            5
        } else {
            6
        }
    } else {
        let mut above = 0u8;
        let mut left = 0u8;
        for i in 0..w4 {
            if x4 + i < max_x4 {
                above |= lc.above_level[b.plane][x4 + i];
                above |= lc.above_dc[b.plane][x4 + i];
            }
        }
        for i in 0..h4 {
            if y4 + i < max_y4 {
                left |= lc.left_level[b.plane][y4 + i];
                left |= lc.left_dc[b.plane][y4 + i];
            }
        }
        let mut ctx = usize::from(above != 0) + usize::from(left != 0);
        ctx += 7;
        if bw * bh > w * h {
            ctx += 3;
        }
        ctx
    }
}

/// `Block_Width[bsize]` / `Block_Height[bsize]` via Num_4x4 tables.
#[inline]
fn block_width(bsize: usize) -> usize {
    usize::from(super::tables_conv::NUM_4X4_BLOCKS_WIDE[bsize]) * 4
}
#[inline]
fn block_height(bsize: usize) -> usize {
    usize::from(super::tables_conv::NUM_4X4_BLOCKS_HIGH[bsize]) * 4
}

/// dc_sign context (spec 8.3.2).
fn dc_sign_ctx(lc: &LevelCtxs, b: &CoefBlock, x4: usize, y4: usize) -> usize {
    let mut max_x4 = b.mi_cols;
    let mut max_y4 = b.mi_rows;
    if b.plane > 0 {
        max_x4 >>= usize::from(b.sub_x);
        max_y4 >>= usize::from(b.sub_y);
    }
    let w4 = usize::from(TX_WIDTH[b.tx_sz]) >> 2;
    let h4 = usize::from(TX_HEIGHT[b.tx_sz]) >> 2;
    let mut dc_sign = 0i32;
    for k in 0..w4 {
        if x4 + k < max_x4 {
            let sign = lc.above_dc[b.plane][x4 + k];
            if sign == 1 {
                dc_sign -= 1;
            } else if sign == 2 {
                dc_sign += 1;
            }
        }
    }
    for k in 0..h4 {
        if y4 + k < max_y4 {
            let sign = lc.left_dc[b.plane][y4 + k];
            if sign == 1 {
                dc_sign -= 1;
            } else if sign == 2 {
                dc_sign += 1;
            }
        }
    }
    match dc_sign.cmp(&0) {
        core::cmp::Ordering::Less => 1,
        core::cmp::Ordering::Greater => 2,
        core::cmp::Ordering::Equal => 0,
    }
}

/// Decodes the coefficients of one transform block (`coeffs()`, spec
/// 5.11.39). `quant` receives the signed quantized levels indexed by
/// position (`pos = scan[c]`, row-major within the adjusted 32x32 corner).
/// Returns `(eob, plane_tx_type)`.
///
/// # Errors
///
/// Returns `InvalidBitstream` if the entropy decoder is exhausted inside an
/// unbounded syntax loop (corrupt data).
#[allow(clippy::too_many_lines)]
pub fn decode_coeffs(
    msac: &mut Msac,
    cdfs: &mut CdfCtx,
    lc: &mut LevelCtxs,
    tx_types: &mut [u8],
    b: &CoefBlock,
    quant: &mut [i32; 1024],
) -> crate::error::CodecResult<(usize, usize)> {
    let x4 = b.start_x >> 2;
    let y4 = b.start_y >> 2;
    let w4 = usize::from(TX_WIDTH[b.tx_sz]) >> 2;
    let h4 = usize::from(TX_HEIGHT[b.tx_sz]) >> 2;
    let tx_sz_ctx =
        (usize::from(TX_SIZE_SQR[b.tx_sz]) + usize::from(TX_SIZE_SQR_UP[b.tx_sz]) + 1) >> 1;
    let ptype = usize::from(b.plane > 0);
    quant.fill(0);

    let mut eob;
    let mut cul_level = 0i32;
    let mut dc_category = 0u8;
    let mut plane_tx_type = DCT_DCT;

    let azc = all_zero_ctx(lc, b, x4, y4);
    let all_zero = msac.read_symbol(&mut cdfs.txb_skip[tx_sz_ctx][azc]) != 0;
    if all_zero {
        eob = 0;
        if b.plane == 0 {
            for i in 0..w4 {
                for j in 0..h4 {
                    if y4 + j < b.mi_rows && x4 + i < b.mi_cols {
                        tx_types[(y4 + j) * b.mi_cols + (x4 + i)] = DCT_DCT as u8;
                    }
                }
            }
        }
    } else {
        if b.plane == 0 {
            read_transform_type(msac, cdfs, tx_types, b.mi_cols, x4, y4, b);
        }
        plane_tx_type = compute_tx_type(tx_types, b.mi_cols, b, x4, y4);
        let scan = get_scan(b.tx_sz, plane_tx_type);
        let tx_class = get_tx_class(plane_tx_type);

        // eob_pt
        let log2w = usize::from(TX_WIDTH_LOG2[b.tx_sz]);
        let log2h = usize::from(TX_HEIGHT_LOG2[b.tx_sz]);
        let eob_multisize = core::cmp::min(log2w, 5) + core::cmp::min(log2h, 5) - 4;
        let eob_ctx = usize::from(tx_class != 0);
        let eob_pt = 1 + match eob_multisize {
            0 => msac.read_symbol(&mut cdfs.eob_pt_16[ptype][eob_ctx]),
            1 => msac.read_symbol(&mut cdfs.eob_pt_32[ptype][eob_ctx]),
            2 => msac.read_symbol(&mut cdfs.eob_pt_64[ptype][eob_ctx]),
            3 => msac.read_symbol(&mut cdfs.eob_pt_128[ptype][eob_ctx]),
            4 => msac.read_symbol(&mut cdfs.eob_pt_256[ptype][eob_ctx]),
            5 => msac.read_symbol(&mut cdfs.eob_pt_512[ptype]),
            _ => msac.read_symbol(&mut cdfs.eob_pt_1024[ptype]),
        };
        eob = if eob_pt < 2 {
            eob_pt
        } else {
            (1 << (eob_pt - 2)) + 1
        };
        let eob_shift = eob_pt as i32 - 3;
        if eob_shift >= 0 {
            let eob_extra =
                msac.read_symbol(&mut cdfs.eob_extra[tx_sz_ctx][ptype][eob_pt - 3]) != 0;
            if eob_extra {
                eob += 1 << eob_shift;
            }
            for i in 1..eob_pt.saturating_sub(2) {
                let eob_shift = eob_pt - 2 - 1 - i;
                if msac.read_literal(1) != 0 {
                    eob += 1 << eob_shift;
                }
            }
        }

        // Levels, reverse scan order.
        for c in (0..eob).rev() {
            let pos = usize::from(scan[c]);
            let mut level;
            if c == eob - 1 {
                // (ctx - SIG_COEF_CONTEXTS + SIG_COEF_CONTEXTS_EOB) with the
                // addition first to stay in unsigned range.
                let ctx = get_coeff_base_ctx(quant, b.tx_sz, pos, c, true, tx_class)
                    + SIG_COEF_CONTEXTS_EOB
                    - SIG_COEF_CONTEXTS;
                level = msac.read_symbol(&mut cdfs.coeff_base_eob[tx_sz_ctx][ptype][ctx]) + 1;
            } else {
                let ctx = get_coeff_base_ctx(quant, b.tx_sz, pos, c, false, tx_class);
                level = msac.read_symbol(&mut cdfs.coeff_base[tx_sz_ctx][ptype][ctx]);
            }
            if level > NUM_BASE_LEVELS {
                let br_ctx = get_br_ctx(quant, b.tx_sz, pos, tx_class);
                let br_tx_sz_ctx = core::cmp::min(tx_sz_ctx, TX_32X32);
                for _ in 0..COEFF_BASE_RANGE / (BR_CDF_SIZE - 1) {
                    let coeff_br =
                        msac.read_symbol(&mut cdfs.coeff_br[br_tx_sz_ctx][ptype][br_ctx]);
                    level += coeff_br;
                    if coeff_br < BR_CDF_SIZE - 1 {
                        break;
                    }
                }
            }
            #[allow(clippy::cast_possible_wrap)]
            {
                quant[pos] = level as i32;
            }
        }

        // Signs and golomb suffixes, forward scan order.
        for c in 0..eob {
            let pos = usize::from(scan[c]);
            let sign = if quant[pos] != 0 {
                if c == 0 {
                    let ctx = dc_sign_ctx(lc, b, x4, y4);
                    msac.read_symbol(&mut cdfs.dc_sign[ptype][ctx]) as u32
                } else {
                    msac.read_literal(1)
                }
            } else {
                0
            };
            if quant[pos] > (NUM_BASE_LEVELS + COEFF_BASE_RANGE) as i32 {
                // Exp-Golomb suffix (spec 5.11.39 golomb_length_bit loop).
                // The spec loop is unbounded; on corrupt data an exhausted
                // symbol decoder yields zero bits forever, so abort once the
                // decoder is past the conformance padding limit (any
                // conformant stream terminates well before that point).
                let mut length = 0u64;
                loop {
                    length += 1;
                    if msac.read_literal(1) != 0 {
                        break;
                    }
                    if msac.exhausted() {
                        return Err(crate::error::CodecError::InvalidBitstream(
                            "AV1: unterminated exp-Golomb coefficient suffix \
                             (entropy decoder exhausted)"
                                .into(),
                        ));
                    }
                }
                // Accumulate modulo 2^32: the final value is masked to
                // 0xFFFFF (spec), and 2^20 divides 2^32, so this preserves
                // the spec result while avoiding shift overflow.
                let mut x: u64 = 1;
                for _ in 0..length.saturating_sub(1) {
                    x = ((x << 1) | u64::from(msac.read_literal(1))) & 0xFFFF_FFFF;
                }
                #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                {
                    quant[pos] =
                        ((x + (COEFF_BASE_RANGE + NUM_BASE_LEVELS) as u64) & 0xFFFFF) as i32;
                }
            }
            if pos == 0 && quant[pos] > 0 {
                dc_category = if sign != 0 { 1 } else { 2 };
            }
            quant[pos] &= 0xFFFFF;
            cul_level += quant[pos];
            if sign != 0 {
                quant[pos] = -quant[pos];
            }
        }
        cul_level = core::cmp::min(63, cul_level);
    }

    // Context updates (also for the all_zero case, which stores 0).
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let cul = cul_level as u8;
    for i in 0..w4 {
        if x4 + i < lc.above_level[b.plane].len() {
            lc.above_level[b.plane][x4 + i] = cul;
            lc.above_dc[b.plane][x4 + i] = dc_category;
        }
    }
    for i in 0..h4 {
        if y4 + i < lc.left_level[b.plane].len() {
            lc.left_level[b.plane][y4 + i] = cul;
            lc.left_dc[b.plane][y4 + i] = dc_category;
        }
    }
    Ok((eob, plane_tx_type))
}
