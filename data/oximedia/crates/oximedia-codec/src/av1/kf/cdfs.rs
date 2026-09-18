//! Live (adapting) CDF context for one tile.
//!
//! Per AV1 spec 8.2.2 (`init_symbol`), each tile starts from a copy of the
//! frame-level CDF state. Intra frames always have
//! `primary_ref_frame == PRIMARY_REF_NONE`, so the frame-level state is the
//! spec defaults: `init_non_coeff_cdfs()` plus `init_coeff_cdfs()` with the
//! quantizer context chosen from `base_q_idx`
//! (spec 7.20/`init_coeff_cdfs` semantics: <=20 -> 0, <=60 -> 1,
//! <=120 -> 2, else 3).

use super::tables_cdf_coef as tc;
use super::tables_cdf_mode as tm;

/// All CDF arrays used by the intra tile decode, by spec Tile* name.
#[derive(Clone)]
pub struct CdfCtx {
    pub intra_frame_y_mode: [[[u16; 14]; 5]; 5],
    pub uv_mode_cfl_not_allowed: [[u16; 14]; 13],
    pub uv_mode_cfl_allowed: [[u16; 15]; 13],
    pub angle_delta: [[u16; 8]; 8],
    pub intrabc: [u16; 3],
    pub partition_w8: [[u16; 5]; 4],
    pub partition_w16: [[u16; 11]; 4],
    pub partition_w32: [[u16; 11]; 4],
    pub partition_w64: [[u16; 11]; 4],
    pub partition_w128: [[u16; 9]; 4],
    pub segment_id: [[u16; 9]; 3],
    pub tx_8x8: [[u16; 3]; 3],
    pub tx_16x16: [[u16; 4]; 3],
    pub tx_32x32: [[u16; 4]; 3],
    pub tx_64x64: [[u16; 4]; 3],
    pub skip: [[u16; 3]; 3],
    pub delta_q: [u16; 5],
    pub delta_lf: [u16; 5],
    pub delta_lf_multi: [[u16; 5]; 4],
    pub filter_intra_mode: [u16; 6],
    pub filter_intra: [[u16; 3]; 22],
    pub cfl_sign: [u16; 9],
    pub cfl_alpha: [[u16; 17]; 6],
    pub palette_y_mode: [[[u16; 3]; 3]; 7],
    pub palette_uv_mode: [[u16; 3]; 2],
    pub intra_tx_type_set1: [[[u16; 8]; 13]; 2],
    pub intra_tx_type_set2: [[[u16; 6]; 13]; 3],
    pub use_wiener: [u16; 3],
    pub use_sgrproj: [u16; 3],
    pub restoration_type: [u16; 4],
    // Coefficient CDFs (quantizer-context slice of the defaults).
    pub txb_skip: [[[u16; 3]; 13]; 5],
    pub eob_pt_16: [[[u16; 6]; 2]; 2],
    pub eob_pt_32: [[[u16; 7]; 2]; 2],
    pub eob_pt_64: [[[u16; 8]; 2]; 2],
    pub eob_pt_128: [[[u16; 9]; 2]; 2],
    pub eob_pt_256: [[[u16; 10]; 2]; 2],
    pub eob_pt_512: [[u16; 11]; 2],
    pub eob_pt_1024: [[u16; 12]; 2],
    pub eob_extra: [[[[u16; 3]; 9]; 2]; 5],
    pub dc_sign: [[[u16; 3]; 3]; 2],
    pub coeff_base_eob: [[[[u16; 4]; 4]; 2]; 5],
    pub coeff_base: [[[[u16; 5]; 42]; 2]; 5],
    pub coeff_br: [[[[u16; 5]; 21]; 2]; 5],
}

impl CdfCtx {
    /// Builds the frame-initial CDF state for `base_q_idx`.
    #[must_use]
    pub fn new(base_q_idx: u32) -> Self {
        let idx = if base_q_idx <= 20 {
            0
        } else if base_q_idx <= 60 {
            1
        } else if base_q_idx <= 120 {
            2
        } else {
            3
        };
        Self {
            intra_frame_y_mode: tm::DEFAULT_INTRA_FRAME_Y_MODE_CDF,
            uv_mode_cfl_not_allowed: tm::DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF,
            uv_mode_cfl_allowed: tm::DEFAULT_UV_MODE_CFL_ALLOWED_CDF,
            angle_delta: tm::DEFAULT_ANGLE_DELTA_CDF,
            intrabc: tm::DEFAULT_INTRABC_CDF,
            partition_w8: tm::DEFAULT_PARTITION_W8_CDF,
            partition_w16: tm::DEFAULT_PARTITION_W16_CDF,
            partition_w32: tm::DEFAULT_PARTITION_W32_CDF,
            partition_w64: tm::DEFAULT_PARTITION_W64_CDF,
            partition_w128: tm::DEFAULT_PARTITION_W128_CDF,
            segment_id: tm::DEFAULT_SEGMENT_ID_CDF,
            tx_8x8: tm::DEFAULT_TX_8X8_CDF,
            tx_16x16: tm::DEFAULT_TX_16X16_CDF,
            tx_32x32: tm::DEFAULT_TX_32X32_CDF,
            tx_64x64: tm::DEFAULT_TX_64X64_CDF,
            skip: tm::DEFAULT_SKIP_CDF,
            delta_q: tm::DEFAULT_DELTA_Q_CDF,
            delta_lf: tm::DEFAULT_DELTA_LF_CDF,
            delta_lf_multi: [tm::DEFAULT_DELTA_LF_CDF; 4],
            filter_intra_mode: tm::DEFAULT_FILTER_INTRA_MODE_CDF,
            filter_intra: tm::DEFAULT_FILTER_INTRA_CDF,
            cfl_sign: tm::DEFAULT_CFL_SIGN_CDF,
            cfl_alpha: tm::DEFAULT_CFL_ALPHA_CDF,
            palette_y_mode: tm::DEFAULT_PALETTE_Y_MODE_CDF,
            palette_uv_mode: tm::DEFAULT_PALETTE_UV_MODE_CDF,
            intra_tx_type_set1: tm::DEFAULT_INTRA_TX_TYPE_SET1_CDF,
            intra_tx_type_set2: tm::DEFAULT_INTRA_TX_TYPE_SET2_CDF,
            use_wiener: tm::DEFAULT_USE_WIENER_CDF,
            use_sgrproj: tm::DEFAULT_USE_SGRPROJ_CDF,
            restoration_type: tm::DEFAULT_RESTORATION_TYPE_CDF,
            txb_skip: tc::DEFAULT_TXB_SKIP_CDF[idx],
            eob_pt_16: tc::DEFAULT_EOB_PT_16_CDF[idx],
            eob_pt_32: tc::DEFAULT_EOB_PT_32_CDF[idx],
            eob_pt_64: tc::DEFAULT_EOB_PT_64_CDF[idx],
            eob_pt_128: tc::DEFAULT_EOB_PT_128_CDF[idx],
            eob_pt_256: tc::DEFAULT_EOB_PT_256_CDF[idx],
            eob_pt_512: tc::DEFAULT_EOB_PT_512_CDF[idx],
            eob_pt_1024: tc::DEFAULT_EOB_PT_1024_CDF[idx],
            eob_extra: tc::DEFAULT_EOB_EXTRA_CDF[idx],
            dc_sign: tc::DEFAULT_DC_SIGN_CDF[idx],
            coeff_base_eob: tc::DEFAULT_COEFF_BASE_EOB_CDF[idx],
            coeff_base: tc::DEFAULT_COEFF_BASE_CDF[idx],
            coeff_br: tc::DEFAULT_COEFF_BR_CDF[idx],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every live CDF row must respect the spec invariants: terminal
    /// probability 32768 and a zero adaptation counter.
    #[test]
    fn default_rows_are_well_formed() {
        let c = CdfCtx::new(80);
        fn check(row: &[u16]) {
            let n = row.len() - 1;
            assert_eq!(row[n - 1], 32768, "terminator");
            assert_eq!(row[n], 0, "counter");
        }
        for a in &c.intra_frame_y_mode {
            for row in a {
                check(row);
            }
        }
        for row in &c.partition_w8 {
            check(row);
        }
        for p in &c.coeff_base {
            for rows in p {
                for row in rows {
                    check(row);
                }
            }
        }
        check(&c.intrabc);
        check(&c.delta_q);
    }
}
