//! AV1 sequence-header and frame-header parsing for the keyframe decoder.
//!
//! Exact implementation of the AV1 spec syntax structures (section 5):
//! `sequence_header_obu`, `color_config`, `uncompressed_header`,
//! `frame_size`/`render_size`/`superres_params`/`compute_image_size`,
//! `tile_info`, `quantization_params`, `segmentation_params`,
//! `delta_q_params`/`delta_lf_params`, `loop_filter_params`, `cdef_params`,
//! `lr_params`, `read_tx_mode`, and `film_grain_params` — the full field
//! surface needed to keep bit positions exact for intra frames.

#![allow(clippy::struct_excessive_bools)]

use super::bits::{tile_log2, BitRdr};
use super::consts::{
    FRAME_LF_COUNT, MAX_SEGMENTS, MAX_TILE_AREA, MAX_TILE_COLS, MAX_TILE_ROWS, MAX_TILE_WIDTH,
    NUM_REF_FRAMES, REFS_PER_FRAME, RESTORATION_TILESIZE_MAX, RESTORE_NONE, SEG_LVL_ALT_Q,
    SEG_LVL_MAX, SEG_LVL_REF_FRAME, SELECT_INTEGER_MV, SELECT_SCREEN_CONTENT_TOOLS,
    SUPERRES_DENOM_BITS, SUPERRES_DENOM_MIN, SUPERRES_NUM, TOTAL_REFS_PER_FRAME,
};
use super::tables_conv::{REMAP_LR_TYPE, SEGMENTATION_FEATURE_MAX, SEGMENTATION_FEATURE_SIGNED};
use crate::error::{CodecError, CodecResult};

/// `Segmentation_Feature_Bits` (spec 5.9.14). Kept next to the parser since
/// the extracted tables carry Signed/Max; Bits appears in the same block.
const SEGMENTATION_FEATURE_BITS: [u32; SEG_LVL_MAX] = [8, 6, 6, 6, 6, 3, 0, 0];

/// Frame type values (spec 6.8.2).
pub const KEY_FRAME: u32 = 0;
pub const INTER_FRAME: u32 = 1;
pub const INTRA_ONLY_FRAME: u32 = 2;
pub const SWITCH_FRAME: u32 = 3;

/// Parsed sequence header (subset of state, all fields bit-exactly consumed).
#[derive(Clone, Debug, Default)]
pub struct SeqHdr {
    pub seq_profile: u32,
    pub still_picture: bool,
    pub reduced_still_picture_header: bool,
    pub decoder_model_info_present_flag: bool,
    pub equal_picture_interval: bool,
    pub buffer_delay_length: u32,
    pub buffer_removal_time_length: u32,
    pub frame_presentation_time_length: u32,
    pub operating_points_cnt: u32,
    pub operating_point_idc: [u32; 32],
    pub decoder_model_present_for_this_op: [bool; 32],
    pub frame_width_bits: u32,
    pub frame_height_bits: u32,
    pub max_frame_width: u32,
    pub max_frame_height: u32,
    pub frame_id_numbers_present_flag: bool,
    pub delta_frame_id_length: u32,
    pub additional_frame_id_length: u32,
    pub use_128x128_superblock: bool,
    pub enable_filter_intra: bool,
    pub enable_intra_edge_filter: bool,
    pub enable_order_hint: bool,
    pub order_hint_bits: u32,
    pub seq_force_screen_content_tools: u32,
    pub seq_force_integer_mv: u32,
    pub enable_superres: bool,
    pub enable_cdef: bool,
    pub enable_restoration: bool,
    // color_config
    pub bit_depth: u32,
    pub mono_chrome: bool,
    pub num_planes: u32,
    pub color_range: bool,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub separate_uv_delta_q: bool,
    pub film_grain_params_present: bool,
}

impl SeqHdr {
    /// `sequence_header_obu()` (spec 5.5.1).
    pub fn parse(payload: &[u8]) -> CodecResult<Self> {
        let r = &mut BitRdr::new(payload);
        let mut s = Self {
            seq_profile: r.f(3)?,
            ..Self::default()
        };
        s.still_picture = r.flag()?;
        s.reduced_still_picture_header = r.flag()?;
        if s.reduced_still_picture_header {
            s.operating_points_cnt = 1;
            s.operating_point_idc[0] = 0;
            let _seq_level_idx0 = r.f(5)?;
        } else {
            let timing_info_present_flag = r.flag()?;
            if timing_info_present_flag {
                // timing_info()
                let _num_units_in_display_tick = r.f(32)?;
                let _time_scale = r.f(32)?;
                s.equal_picture_interval = r.flag()?;
                if s.equal_picture_interval {
                    let _num_ticks_per_picture_minus_1 = r.uvlc()?;
                }
                s.decoder_model_info_present_flag = r.flag()?;
                if s.decoder_model_info_present_flag {
                    // decoder_model_info()
                    s.buffer_delay_length = r.f(5)? + 1;
                    let _num_units_in_decoding_tick = r.f(32)?;
                    s.buffer_removal_time_length = r.f(5)? + 1;
                    s.frame_presentation_time_length = r.f(5)? + 1;
                }
            }
            let initial_display_delay_present_flag = r.flag()?;
            let operating_points_cnt_minus_1 = r.f(5)?;
            s.operating_points_cnt = operating_points_cnt_minus_1 + 1;
            for i in 0..=operating_points_cnt_minus_1 as usize {
                s.operating_point_idc[i] = r.f(12)?;
                let seq_level_idx = r.f(5)?;
                if seq_level_idx > 7 {
                    let _seq_tier = r.f(1)?;
                }
                if s.decoder_model_info_present_flag {
                    s.decoder_model_present_for_this_op[i] = r.flag()?;
                    if s.decoder_model_present_for_this_op[i] {
                        // operating_parameters_info( i )
                        let n = s.buffer_delay_length;
                        let _decoder_buffer_delay = r.f(n)?;
                        let _encoder_buffer_delay = r.f(n)?;
                        let _low_delay_mode_flag = r.flag()?;
                    }
                }
                if initial_display_delay_present_flag {
                    let present = r.flag()?;
                    if present {
                        let _initial_display_delay_minus_1 = r.f(4)?;
                    }
                }
            }
        }
        s.frame_width_bits = r.f(4)? + 1;
        s.frame_height_bits = r.f(4)? + 1;
        s.max_frame_width = r.f(s.frame_width_bits)? + 1;
        s.max_frame_height = r.f(s.frame_height_bits)? + 1;
        s.frame_id_numbers_present_flag = if s.reduced_still_picture_header {
            false
        } else {
            r.flag()?
        };
        if s.frame_id_numbers_present_flag {
            s.delta_frame_id_length = r.f(4)? + 2;
            s.additional_frame_id_length = r.f(3)? + 1;
        }
        s.use_128x128_superblock = r.flag()?;
        s.enable_filter_intra = r.flag()?;
        s.enable_intra_edge_filter = r.flag()?;
        if s.reduced_still_picture_header {
            s.seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS as u32;
            s.seq_force_integer_mv = SELECT_INTEGER_MV as u32;
            s.order_hint_bits = 0;
        } else {
            let _enable_interintra_compound = r.flag()?;
            let _enable_masked_compound = r.flag()?;
            let _enable_warped_motion = r.flag()?;
            let _enable_dual_filter = r.flag()?;
            s.enable_order_hint = r.flag()?;
            if s.enable_order_hint {
                let _enable_jnt_comp = r.flag()?;
                let _enable_ref_frame_mvs = r.flag()?;
            }
            let seq_choose_screen_content_tools = r.flag()?;
            s.seq_force_screen_content_tools = if seq_choose_screen_content_tools {
                SELECT_SCREEN_CONTENT_TOOLS as u32
            } else {
                r.f(1)?
            };
            if s.seq_force_screen_content_tools > 0 {
                let seq_choose_integer_mv = r.flag()?;
                s.seq_force_integer_mv = if seq_choose_integer_mv {
                    SELECT_INTEGER_MV as u32
                } else {
                    r.f(1)?
                };
            } else {
                s.seq_force_integer_mv = SELECT_INTEGER_MV as u32;
            }
            if s.enable_order_hint {
                s.order_hint_bits = r.f(3)? + 1;
            }
        }
        s.enable_superres = r.flag()?;
        s.enable_cdef = r.flag()?;
        s.enable_restoration = r.flag()?;
        s.parse_color_config(r)?;
        s.film_grain_params_present = r.flag()?;
        Ok(s)
    }

    /// `color_config()` (spec 5.5.2).
    fn parse_color_config(&mut self, r: &mut BitRdr<'_>) -> CodecResult<()> {
        let high_bitdepth = r.flag()?;
        if self.seq_profile == 2 && high_bitdepth {
            let twelve_bit = r.flag()?;
            self.bit_depth = if twelve_bit { 12 } else { 10 };
        } else if self.seq_profile <= 2 {
            self.bit_depth = if high_bitdepth { 10 } else { 8 };
        } else {
            return Err(CodecError::InvalidBitstream(
                "AV1: reserved seq_profile 3".into(),
            ));
        }
        self.mono_chrome = if self.seq_profile == 1 {
            false
        } else {
            r.flag()?
        };
        self.num_planes = if self.mono_chrome { 1 } else { 3 };
        let color_description_present_flag = r.flag()?;
        // H.273 code points (spec 6.4.2): CP_BT_709=1, TC_SRGB=13, MC_IDENTITY=0.
        let (cp, tc, mc) = if color_description_present_flag {
            (r.f(8)?, r.f(8)?, r.f(8)?)
        } else {
            (2, 2, 2) // CP/TC/MC_UNSPECIFIED
        };
        if self.mono_chrome {
            self.color_range = r.flag()?;
            self.subsampling_x = true;
            self.subsampling_y = true;
            self.separate_uv_delta_q = false;
            return Ok(());
        } else if cp == 1 && tc == 13 && mc == 0 {
            self.color_range = true;
            self.subsampling_x = false;
            self.subsampling_y = false;
        } else {
            self.color_range = r.flag()?;
            if self.seq_profile == 0 {
                self.subsampling_x = true;
                self.subsampling_y = true;
            } else if self.seq_profile == 1 {
                self.subsampling_x = false;
                self.subsampling_y = false;
            } else if self.bit_depth == 12 {
                self.subsampling_x = r.flag()?;
                self.subsampling_y = if self.subsampling_x { r.flag()? } else { false };
            } else {
                self.subsampling_x = true;
                self.subsampling_y = false;
            }
            if self.subsampling_x && self.subsampling_y {
                let _chroma_sample_position = r.f(2)?;
            }
        }
        self.separate_uv_delta_q = r.flag()?;
        Ok(())
    }
}

/// Per-frame segmentation state.
#[derive(Clone, Debug, Default)]
pub struct Segmentation {
    pub enabled: bool,
    pub feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub feature_data: [[i32; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub seg_id_pre_skip: bool,
    pub last_active_seg_id: u32,
}

/// Tile layout produced by `tile_info()`.
#[derive(Clone, Debug, Default)]
pub struct TileLayout {
    pub tile_cols_log2: u32,
    pub tile_rows_log2: u32,
    pub tile_cols: u32,
    pub tile_rows: u32,
    /// `MiColStarts[0..=TileCols]`.
    pub mi_col_starts: Vec<u32>,
    /// `MiRowStarts[0..=TileRows]`.
    pub mi_row_starts: Vec<u32>,
    pub context_update_tile_id: u32,
    pub tile_size_bytes: u32,
}

/// Loop-filter frame parameters.
#[derive(Clone, Debug, Default)]
pub struct LfParams {
    pub level: [u32; 4],
    pub sharpness: u32,
    pub delta_enabled: bool,
    pub ref_deltas: [i32; TOTAL_REFS_PER_FRAME],
    pub mode_deltas: [i32; 2],
}

/// CDEF frame parameters.
#[derive(Clone, Debug, Default)]
pub struct CdefParams {
    pub damping: u32,
    pub bits: u32,
    pub y_pri_strength: [u32; 8],
    pub y_sec_strength: [u32; 8],
    pub uv_pri_strength: [u32; 8],
    pub uv_sec_strength: [u32; 8],
}

/// Loop-restoration frame parameters.
#[derive(Clone, Debug, Default)]
pub struct LrParams {
    /// `FrameRestorationType[plane]` (RESTORE_* values).
    pub frame_restoration_type: [usize; 3],
    pub uses_lr: bool,
    /// `LoopRestorationSize[plane]`.
    pub loop_restoration_size: [u32; 3],
}

/// Parsed uncompressed frame header for an intra frame.
#[derive(Clone, Debug, Default)]
pub struct FrameHdr {
    pub show_existing_frame: bool,
    pub frame_to_show_map_idx: u32,
    pub frame_type: u32,
    pub frame_is_intra: bool,
    pub show_frame: bool,
    pub showable_frame: bool,
    pub error_resilient_mode: bool,
    pub disable_cdf_update: bool,
    pub allow_screen_content_tools: bool,
    pub force_integer_mv: bool,
    pub frame_size_override_flag: bool,
    pub order_hint: u32,
    pub refresh_frame_flags: u32,
    pub allow_intrabc: bool,
    pub use_superres: bool,
    pub superres_denom: u32,
    pub frame_width: u32,
    pub frame_height: u32,
    pub upscaled_width: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub mi_cols: u32,
    pub mi_rows: u32,
    pub disable_frame_end_update_cdf: bool,
    pub tiles: TileLayout,
    // quantization_params
    pub base_q_idx: u32,
    pub delta_q_y_dc: i32,
    pub delta_q_u_dc: i32,
    pub delta_q_u_ac: i32,
    pub delta_q_v_dc: i32,
    pub delta_q_v_ac: i32,
    pub using_qmatrix: bool,
    pub qm_y: u32,
    pub qm_u: u32,
    pub qm_v: u32,
    pub seg: Segmentation,
    // delta q / lf
    pub delta_q_present: bool,
    pub delta_q_res: u32,
    pub delta_lf_present: bool,
    pub delta_lf_res: u32,
    pub delta_lf_multi: bool,
    pub coded_lossless: bool,
    pub all_lossless: bool,
    pub lossless_array: [bool; MAX_SEGMENTS],
    pub lf: LfParams,
    pub cdef: CdefParams,
    pub lr: LrParams,
    /// `TxMode`: 0 = ONLY_4X4, 3 = TX_MODE_SELECT, 4 = TX_MODE_LARGEST.
    pub tx_mode_select: bool,
    pub reduced_tx_set: bool,
    /// Film grain must be applied to the output (`apply_grain`).
    pub apply_grain: bool,
    /// Total bits consumed by the uncompressed header (for OBU_FRAME).
    pub header_bits: usize,
}

impl FrameHdr {
    /// `get_qindex( ignoreDeltaQ = 1, segmentId )` (spec 7.12.2), i.e. the
    /// frame-header-time variant with `CurrentQIndex` not yet in play.
    pub fn qindex_ignoring_deltaq(&self, segment_id: usize) -> u32 {
        let mut qindex = i64::from(self.base_q_idx);
        if self.seg.enabled && self.seg.feature_enabled[segment_id][SEG_LVL_ALT_Q] {
            qindex = i64::from(self.base_q_idx)
                + i64::from(self.seg.feature_data[segment_id][SEG_LVL_ALT_Q]);
        }
        qindex.clamp(0, 255) as u32
    }

    /// `uncompressed_header()` (spec 5.9.2) for the intra decode path.
    ///
    /// # Errors
    ///
    /// Returns honest `UnsupportedFeature` errors for inter frames and other
    /// surfaces outside the keyframe decoder, and `InvalidBitstream` for
    /// malformed data.
    #[allow(clippy::too_many_lines)]
    pub fn parse(payload: &[u8], seq: &SeqHdr) -> CodecResult<Self> {
        let r = &mut BitRdr::new(payload);
        let mut h = Self::default();

        let id_len = if seq.frame_id_numbers_present_flag {
            seq.additional_frame_id_length + seq.delta_frame_id_length
        } else {
            0
        };
        let all_frames = (1u32 << NUM_REF_FRAMES) - 1;

        if seq.reduced_still_picture_header {
            h.show_existing_frame = false;
            h.frame_type = KEY_FRAME;
            h.frame_is_intra = true;
            h.show_frame = true;
            h.showable_frame = false;
            h.error_resilient_mode = true; // KEY_FRAME && show_frame
        } else {
            h.show_existing_frame = r.flag()?;
            if h.show_existing_frame {
                h.frame_to_show_map_idx = r.f(3)?;
                if seq.decoder_model_info_present_flag && !seq.equal_picture_interval {
                    let _frame_presentation_time = r.f(seq.frame_presentation_time_length)?;
                }
                if seq.frame_id_numbers_present_flag {
                    let _display_frame_id = r.f(id_len)?;
                }
                h.header_bits = r.position();
                return Ok(h);
            }
            h.frame_type = r.f(2)?;
            h.frame_is_intra = h.frame_type == INTRA_ONLY_FRAME || h.frame_type == KEY_FRAME;
            h.show_frame = r.flag()?;
            if h.show_frame && seq.decoder_model_info_present_flag && !seq.equal_picture_interval {
                let _frame_presentation_time = r.f(seq.frame_presentation_time_length)?;
            }
            if h.show_frame {
                h.showable_frame = h.frame_type != KEY_FRAME;
            } else {
                h.showable_frame = r.flag()?;
            }
            if h.frame_type == SWITCH_FRAME || (h.frame_type == KEY_FRAME && h.show_frame) {
                h.error_resilient_mode = true;
            } else {
                h.error_resilient_mode = r.flag()?;
            }
        }

        if !h.frame_is_intra {
            // TODO(0.2.x): inter frame decode — motion vectors, reference
            // MVs, compound prediction, motion compensation, CDF forwarding.
            return Err(CodecError::UnsupportedFeature(
                "AV1 inter frame decode not implemented (keyframe/intra-only decode is)"
                    .to_string(),
            ));
        }

        h.disable_cdf_update = r.flag()?;
        h.allow_screen_content_tools =
            if seq.seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS as u32 {
                r.flag()?
            } else {
                seq.seq_force_screen_content_tools != 0
            };
        if h.allow_screen_content_tools && seq.seq_force_integer_mv == SELECT_INTEGER_MV as u32 {
            let _force_integer_mv = r.flag()?;
        }
        h.force_integer_mv = true; // FrameIsIntra
        if seq.frame_id_numbers_present_flag {
            let _current_frame_id = r.f(id_len)?;
        }
        h.frame_size_override_flag = if h.frame_type == SWITCH_FRAME {
            true
        } else if seq.reduced_still_picture_header {
            false
        } else {
            r.flag()?
        };
        h.order_hint = r.f(seq.order_hint_bits)?;
        // FrameIsIntra => primary_ref_frame = PRIMARY_REF_NONE (no bits read).
        if seq.decoder_model_info_present_flag {
            let buffer_removal_time_present_flag = r.flag()?;
            if buffer_removal_time_present_flag {
                for op_num in 0..seq.operating_points_cnt as usize {
                    if seq.decoder_model_present_for_this_op[op_num] {
                        let op_pt_idc = seq.operating_point_idc[op_num];
                        // temporal_id = spatial_id = 0 for the frames we
                        // decode (no OBU extension in this path).
                        let in_temporal_layer = op_pt_idc & 1;
                        let in_spatial_layer = (op_pt_idc >> 8) & 1;
                        if op_pt_idc == 0 || (in_temporal_layer != 0 && in_spatial_layer != 0) {
                            let _buffer_removal_time = r.f(seq.buffer_removal_time_length)?;
                        }
                    }
                }
            }
        }
        h.refresh_frame_flags =
            if h.frame_type == SWITCH_FRAME || (h.frame_type == KEY_FRAME && h.show_frame) {
                all_frames
            } else {
                r.f(8)?
            };
        if (!h.frame_is_intra || h.refresh_frame_flags != all_frames)
            && h.error_resilient_mode
            && seq.enable_order_hint
        {
            for _ in 0..NUM_REF_FRAMES {
                let _ref_order_hint = r.f(seq.order_hint_bits)?;
            }
        }
        // FrameIsIntra branch: frame_size(), render_size(), allow_intrabc.
        h.parse_frame_size(r, seq)?;
        h.parse_render_size(r)?;
        if h.allow_screen_content_tools && h.upscaled_width == h.frame_width {
            h.allow_intrabc = r.flag()?;
        }
        h.disable_frame_end_update_cdf = if seq.reduced_still_picture_header || h.disable_cdf_update
        {
            true
        } else {
            r.flag()?
        };
        // primary_ref_frame == PRIMARY_REF_NONE: init_non_coeff_cdfs() +
        // setup_past_independence() happen at decode time.
        h.parse_tile_info(r, seq)?;
        h.parse_quantization_params(r, seq)?;
        h.parse_segmentation_params(r)?;
        // delta_q_params()
        if h.base_q_idx > 0 {
            h.delta_q_present = r.flag()?;
        }
        if h.delta_q_present {
            h.delta_q_res = r.f(2)?;
        }
        // delta_lf_params()
        if h.delta_q_present {
            if !h.allow_intrabc {
                h.delta_lf_present = r.flag()?;
            }
            if h.delta_lf_present {
                h.delta_lf_res = r.f(2)?;
                h.delta_lf_multi = r.flag()?;
            }
        }
        // CodedLossless / LosslessArray (init_coeff_cdfs at decode time).
        h.coded_lossless = true;
        for segment_id in 0..MAX_SEGMENTS {
            let qindex = h.qindex_ignoring_deltaq(segment_id);
            h.lossless_array[segment_id] = qindex == 0
                && h.delta_q_y_dc == 0
                && h.delta_q_u_ac == 0
                && h.delta_q_u_dc == 0
                && h.delta_q_v_ac == 0
                && h.delta_q_v_dc == 0;
            if !h.lossless_array[segment_id] {
                h.coded_lossless = false;
            }
        }
        h.all_lossless = h.coded_lossless && (h.frame_width == h.upscaled_width);
        h.parse_loop_filter_params(r, seq)?;
        h.parse_cdef_params(r, seq)?;
        h.parse_lr_params(r, seq)?;
        // read_tx_mode()
        if !h.coded_lossless {
            h.tx_mode_select = r.flag()?;
        }
        // frame_reference_mode(): FrameIsIntra => reference_select = 0.
        // skip_mode_params(): FrameIsIntra => skipModeAllowed = 0, no bits.
        // allow_warped_motion: FrameIsIntra => 0, no bits.
        h.reduced_tx_set = r.flag()?;
        // global_motion_params(): FrameIsIntra => returns after defaults.
        h.parse_film_grain_params(r, seq)?;
        h.header_bits = r.position();
        Ok(h)
    }

    /// `frame_size()` + `superres_params()` + `compute_image_size()`.
    fn parse_frame_size(&mut self, r: &mut BitRdr<'_>, seq: &SeqHdr) -> CodecResult<()> {
        if self.frame_size_override_flag {
            self.frame_width = r.f(seq.frame_width_bits)? + 1;
            self.frame_height = r.f(seq.frame_height_bits)? + 1;
        } else {
            self.frame_width = seq.max_frame_width;
            self.frame_height = seq.max_frame_height;
        }
        // superres_params()
        self.use_superres = if seq.enable_superres {
            r.flag()?
        } else {
            false
        };
        self.superres_denom = if self.use_superres {
            r.f(SUPERRES_DENOM_BITS as u32)? + SUPERRES_DENOM_MIN as u32
        } else {
            SUPERRES_NUM as u32
        };
        self.upscaled_width = self.frame_width;
        self.frame_width = (self.upscaled_width * SUPERRES_NUM as u32 + (self.superres_denom / 2))
            / self.superres_denom;
        // compute_image_size()
        self.mi_cols = 2 * ((self.frame_width + 7) >> 3);
        self.mi_rows = 2 * ((self.frame_height + 7) >> 3);
        Ok(())
    }

    /// `render_size()`.
    fn parse_render_size(&mut self, r: &mut BitRdr<'_>) -> CodecResult<()> {
        let render_and_frame_size_different = r.flag()?;
        if render_and_frame_size_different {
            self.render_width = r.f(16)? + 1;
            self.render_height = r.f(16)? + 1;
        } else {
            self.render_width = self.upscaled_width;
            self.render_height = self.frame_height;
        }
        Ok(())
    }

    /// `tile_info()` (spec 5.9.15).
    fn parse_tile_info(&mut self, r: &mut BitRdr<'_>, seq: &SeqHdr) -> CodecResult<()> {
        let t = &mut self.tiles;
        let (sb_cols, sb_rows, sb_shift) = if seq.use_128x128_superblock {
            ((self.mi_cols + 31) >> 5, (self.mi_rows + 31) >> 5, 5u32)
        } else {
            ((self.mi_cols + 15) >> 4, (self.mi_rows + 15) >> 4, 4u32)
        };
        let sb_size = sb_shift + 2;
        let max_tile_width_sb = MAX_TILE_WIDTH as u32 >> sb_size;
        let mut max_tile_area_sb = (MAX_TILE_AREA as u64 >> (2 * sb_size)) as u32;
        let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
        let max_log2_tile_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS as u32));
        let max_log2_tile_rows = tile_log2(1, sb_rows.min(MAX_TILE_ROWS as u32));
        let min_log2_tiles = min_log2_tile_cols.max(tile_log2(max_tile_area_sb, sb_rows * sb_cols));

        let uniform_tile_spacing_flag = r.flag()?;
        if uniform_tile_spacing_flag {
            t.tile_cols_log2 = min_log2_tile_cols;
            while t.tile_cols_log2 < max_log2_tile_cols {
                if r.flag()? {
                    t.tile_cols_log2 += 1;
                } else {
                    break;
                }
            }
            let tile_width_sb = (sb_cols + (1 << t.tile_cols_log2) - 1) >> t.tile_cols_log2;
            t.mi_col_starts.clear();
            let mut start_sb = 0;
            while start_sb < sb_cols {
                t.mi_col_starts.push(start_sb << sb_shift);
                start_sb += tile_width_sb;
            }
            t.tile_cols = t.mi_col_starts.len() as u32;
            t.mi_col_starts.push(self.mi_cols);

            let min_log2_tile_rows = min_log2_tiles.saturating_sub(t.tile_cols_log2);
            t.tile_rows_log2 = min_log2_tile_rows;
            while t.tile_rows_log2 < max_log2_tile_rows {
                if r.flag()? {
                    t.tile_rows_log2 += 1;
                } else {
                    break;
                }
            }
            let tile_height_sb = (sb_rows + (1 << t.tile_rows_log2) - 1) >> t.tile_rows_log2;
            t.mi_row_starts.clear();
            let mut start_sb = 0;
            while start_sb < sb_rows {
                t.mi_row_starts.push(start_sb << sb_shift);
                start_sb += tile_height_sb;
            }
            t.tile_rows = t.mi_row_starts.len() as u32;
            t.mi_row_starts.push(self.mi_rows);
        } else {
            let mut widest_tile_sb = 0u32;
            let mut start_sb = 0u32;
            t.mi_col_starts.clear();
            while start_sb < sb_cols {
                t.mi_col_starts.push(start_sb << sb_shift);
                let max_width = (sb_cols - start_sb).min(max_tile_width_sb);
                let width_in_sbs_minus_1 = r.ns(max_width)?;
                let size_sb = width_in_sbs_minus_1 + 1;
                widest_tile_sb = widest_tile_sb.max(size_sb);
                start_sb += size_sb;
            }
            t.tile_cols = t.mi_col_starts.len() as u32;
            t.mi_col_starts.push(self.mi_cols);
            t.tile_cols_log2 = tile_log2(1, t.tile_cols);

            if min_log2_tiles > 0 {
                max_tile_area_sb = (sb_rows * sb_cols) >> (min_log2_tiles + 1);
            } else {
                max_tile_area_sb = sb_rows * sb_cols;
            }
            let max_tile_height_sb = (max_tile_area_sb / widest_tile_sb).max(1);

            let mut start_sb = 0u32;
            t.mi_row_starts.clear();
            while start_sb < sb_rows {
                t.mi_row_starts.push(start_sb << sb_shift);
                let max_height = (sb_rows - start_sb).min(max_tile_height_sb);
                let height_in_sbs_minus_1 = r.ns(max_height)?;
                start_sb += height_in_sbs_minus_1 + 1;
            }
            t.tile_rows = t.mi_row_starts.len() as u32;
            t.mi_row_starts.push(self.mi_rows);
            t.tile_rows_log2 = tile_log2(1, t.tile_rows);
        }
        if t.tile_cols_log2 > 0 || t.tile_rows_log2 > 0 {
            t.context_update_tile_id = r.f(t.tile_rows_log2 + t.tile_cols_log2)?;
            t.tile_size_bytes = r.f(2)? + 1;
        } else {
            t.context_update_tile_id = 0;
        }
        Ok(())
    }

    /// `quantization_params()` (spec 5.9.12).
    fn parse_quantization_params(&mut self, r: &mut BitRdr<'_>, seq: &SeqHdr) -> CodecResult<()> {
        self.base_q_idx = r.f(8)?;
        self.delta_q_y_dc = read_delta_q(r)?;
        if seq.num_planes > 1 {
            let diff_uv_delta = if seq.separate_uv_delta_q {
                r.flag()?
            } else {
                false
            };
            self.delta_q_u_dc = read_delta_q(r)?;
            self.delta_q_u_ac = read_delta_q(r)?;
            if diff_uv_delta {
                self.delta_q_v_dc = read_delta_q(r)?;
                self.delta_q_v_ac = read_delta_q(r)?;
            } else {
                self.delta_q_v_dc = self.delta_q_u_dc;
                self.delta_q_v_ac = self.delta_q_u_ac;
            }
        }
        self.using_qmatrix = r.flag()?;
        if self.using_qmatrix {
            self.qm_y = r.f(4)?;
            self.qm_u = r.f(4)?;
            self.qm_v = if seq.separate_uv_delta_q {
                r.f(4)?
            } else {
                self.qm_u
            };
        }
        Ok(())
    }

    /// `segmentation_params()` (spec 5.9.14). Intra frames always have
    /// `primary_ref_frame == PRIMARY_REF_NONE`, so the update flags are
    /// implied (update_map = 1, temporal_update = 0, update_data = 1).
    fn parse_segmentation_params(&mut self, r: &mut BitRdr<'_>) -> CodecResult<()> {
        let s = &mut self.seg;
        s.enabled = r.flag()?;
        if s.enabled {
            // primary_ref_frame == PRIMARY_REF_NONE on intra frames.
            for i in 0..MAX_SEGMENTS {
                for j in 0..SEG_LVL_MAX {
                    let feature_enabled = r.flag()?;
                    s.feature_enabled[i][j] = feature_enabled;
                    let mut clipped_value = 0i32;
                    if feature_enabled {
                        let bits_to_read = SEGMENTATION_FEATURE_BITS[j];
                        let limit = i32::from(SEGMENTATION_FEATURE_MAX[j]);
                        if SEGMENTATION_FEATURE_SIGNED[j] == 1 {
                            let feature_value = r.su(1 + bits_to_read)?;
                            clipped_value = feature_value.clamp(-limit, limit);
                        } else {
                            let feature_value = r.f(bits_to_read)? as i32;
                            clipped_value = feature_value.clamp(0, limit);
                        }
                    }
                    s.feature_data[i][j] = clipped_value;
                }
            }
        }
        s.seg_id_pre_skip = false;
        s.last_active_seg_id = 0;
        for i in 0..MAX_SEGMENTS {
            for j in 0..SEG_LVL_MAX {
                if s.feature_enabled[i][j] {
                    s.last_active_seg_id = i as u32;
                    if j >= SEG_LVL_REF_FRAME {
                        s.seg_id_pre_skip = true;
                    }
                }
            }
        }
        Ok(())
    }

    /// `loop_filter_params()` (spec 5.9.11).
    fn parse_loop_filter_params(&mut self, r: &mut BitRdr<'_>, seq: &SeqHdr) -> CodecResult<()> {
        let lf = &mut self.lf;
        if self.coded_lossless || self.allow_intrabc {
            lf.level = [0; 4];
            lf.ref_deltas = [1, 0, 0, 0, -1, 0, -1, -1];
            lf.mode_deltas = [0, 0];
            return Ok(());
        }
        // Defaults set by setup_past_independence (spec 7.20): the same
        // values as the lossless branch above; update_* only overrides.
        lf.ref_deltas = [1, 0, 0, 0, -1, 0, -1, -1];
        lf.mode_deltas = [0, 0];
        lf.level[0] = r.f(6)?;
        lf.level[1] = r.f(6)?;
        if seq.num_planes > 1 && (lf.level[0] != 0 || lf.level[1] != 0) {
            lf.level[2] = r.f(6)?;
            lf.level[3] = r.f(6)?;
        }
        lf.sharpness = r.f(3)?;
        lf.delta_enabled = r.flag()?;
        if lf.delta_enabled {
            let delta_update = r.flag()?;
            if delta_update {
                for i in 0..TOTAL_REFS_PER_FRAME {
                    if r.flag()? {
                        lf.ref_deltas[i] = r.su(1 + 6)?;
                    }
                }
                for i in 0..2 {
                    if r.flag()? {
                        lf.mode_deltas[i] = r.su(1 + 6)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// `cdef_params()` (spec 5.9.19).
    fn parse_cdef_params(&mut self, r: &mut BitRdr<'_>, seq: &SeqHdr) -> CodecResult<()> {
        let c = &mut self.cdef;
        if self.coded_lossless || self.allow_intrabc || !seq.enable_cdef {
            c.bits = 0;
            c.y_pri_strength[0] = 0;
            c.y_sec_strength[0] = 0;
            c.uv_pri_strength[0] = 0;
            c.uv_sec_strength[0] = 0;
            c.damping = 3;
            return Ok(());
        }
        c.damping = r.f(2)? + 3;
        c.bits = r.f(2)?;
        for i in 0..(1usize << c.bits) {
            c.y_pri_strength[i] = r.f(4)?;
            c.y_sec_strength[i] = r.f(2)?;
            if c.y_sec_strength[i] == 3 {
                c.y_sec_strength[i] += 1;
            }
            if seq.num_planes > 1 {
                c.uv_pri_strength[i] = r.f(4)?;
                c.uv_sec_strength[i] = r.f(2)?;
                if c.uv_sec_strength[i] == 3 {
                    c.uv_sec_strength[i] += 1;
                }
            }
        }
        Ok(())
    }

    /// `lr_params()` (spec 5.9.20).
    fn parse_lr_params(&mut self, r: &mut BitRdr<'_>, seq: &SeqHdr) -> CodecResult<()> {
        let lr = &mut self.lr;
        if self.all_lossless || self.allow_intrabc || !seq.enable_restoration {
            lr.frame_restoration_type = [RESTORE_NONE; 3];
            lr.uses_lr = false;
            return Ok(());
        }
        lr.uses_lr = false;
        let mut uses_chroma_lr = false;
        for i in 0..seq.num_planes as usize {
            let lr_type = r.f(2)? as usize;
            lr.frame_restoration_type[i] = REMAP_LR_TYPE[lr_type] as usize;
            if lr.frame_restoration_type[i] != RESTORE_NONE {
                lr.uses_lr = true;
                if i > 0 {
                    uses_chroma_lr = true;
                }
            }
        }
        if lr.uses_lr {
            let mut lr_unit_shift;
            if seq.use_128x128_superblock {
                lr_unit_shift = r.f(1)? + 1;
            } else {
                lr_unit_shift = r.f(1)?;
                if lr_unit_shift != 0 {
                    lr_unit_shift += r.f(1)?;
                }
            }
            lr.loop_restoration_size[0] = RESTORATION_TILESIZE_MAX as u32 >> (2 - lr_unit_shift);
            let lr_uv_shift = if seq.subsampling_x && seq.subsampling_y && uses_chroma_lr {
                r.f(1)?
            } else {
                0
            };
            lr.loop_restoration_size[1] = lr.loop_restoration_size[0] >> lr_uv_shift;
            lr.loop_restoration_size[2] = lr.loop_restoration_size[0] >> lr_uv_shift;
        }
        Ok(())
    }

    /// `film_grain_params()` (spec 5.9.30) — parsed exactly; application is
    /// gated at decode time.
    #[allow(clippy::too_many_lines)]
    fn parse_film_grain_params(&mut self, r: &mut BitRdr<'_>, seq: &SeqHdr) -> CodecResult<()> {
        if !seq.film_grain_params_present || (!self.show_frame && !self.showable_frame) {
            return Ok(());
        }
        self.apply_grain = r.flag()?;
        if !self.apply_grain {
            return Ok(());
        }
        let _grain_seed = r.f(16)?;
        // frame_type is intra here, so update_grain == 1 (no bit read).
        let num_y_points = r.f(4)?;
        for _ in 0..num_y_points {
            let _point_y_value = r.f(8)?;
            let _point_y_scaling = r.f(8)?;
        }
        let chroma_scaling_from_luma = if seq.mono_chrome { false } else { r.flag()? };
        let (num_cb_points, num_cr_points) = if seq.mono_chrome
            || chroma_scaling_from_luma
            || (seq.subsampling_x && seq.subsampling_y && num_y_points == 0)
        {
            (0, 0)
        } else {
            let num_cb = r.f(4)?;
            for _ in 0..num_cb {
                let _v = r.f(8)?;
                let _s = r.f(8)?;
            }
            let num_cr = r.f(4)?;
            for _ in 0..num_cr {
                let _v = r.f(8)?;
                let _s = r.f(8)?;
            }
            (num_cb, num_cr)
        };
        let _grain_scaling_minus_8 = r.f(2)?;
        let ar_coeff_lag = r.f(2)?;
        let num_pos_luma = 2 * ar_coeff_lag * (ar_coeff_lag + 1);
        let num_pos_chroma = if num_y_points > 0 {
            for _ in 0..num_pos_luma {
                let _c = r.f(8)?;
            }
            num_pos_luma + 1
        } else {
            num_pos_luma
        };
        if chroma_scaling_from_luma || num_cb_points > 0 {
            for _ in 0..num_pos_chroma {
                let _c = r.f(8)?;
            }
        }
        if chroma_scaling_from_luma || num_cr_points > 0 {
            for _ in 0..num_pos_chroma {
                let _c = r.f(8)?;
            }
        }
        let _ar_coeff_shift_minus_6 = r.f(2)?;
        let _grain_scale_shift = r.f(2)?;
        if num_cb_points > 0 {
            let _cb_mult = r.f(8)?;
            let _cb_luma_mult = r.f(8)?;
            let _cb_offset = r.f(9)?;
        }
        if num_cr_points > 0 {
            let _cr_mult = r.f(8)?;
            let _cr_luma_mult = r.f(8)?;
            let _cr_offset = r.f(9)?;
        }
        let _overlap_flag = r.flag()?;
        let _clip_to_restricted_range = r.flag()?;
        Ok(())
    }
}

/// `read_delta_q()` (spec 5.9.13).
fn read_delta_q(r: &mut BitRdr<'_>) -> CodecResult<i32> {
    let delta_coded = r.flag()?;
    if delta_coded {
        r.su(1 + 6)
    } else {
        Ok(0)
    }
}
