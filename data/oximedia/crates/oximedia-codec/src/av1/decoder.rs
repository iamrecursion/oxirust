//! AV1 decoder implementation.
//!
//! Keyframes and intra-only frames decode to real pixels via the bit-exact
//! intra pipeline in [`crate::av1::kf`] (verified against dav1d/aomdec);
//! inter frames and not-yet-implemented surfaces return an honest
//! [`CodecError::UnsupportedFeature`] — never a fabricated frame.

#![forbid(unsafe_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::cast_possible_truncation)]

use super::cdef::CdefParams;
use super::frame_header::{FrameHeader, FrameType as Av1FrameType};
use super::kf::{self, TuOutcome};
use super::loop_filter::LoopFilterParams;
use super::obu::{ObuIterator, ObuType};
use super::quantization::QuantizationParams;
use super::sequence::SequenceHeader;
use super::tile::TileInfo;
use crate::error::{CodecError, CodecResult};
use crate::frame::{FrameType, Plane, VideoFrame};
use crate::traits::{DecoderConfig, VideoDecoder};
use oximedia_core::{CodecId, PixelFormat, Rational, Timestamp};

/// AV1 decoder probe state (headers parsed by the legacy metadata parser;
/// kept for the public inspection API, never used for pixel decode).
#[derive(Clone, Debug, Default)]
struct DecoderState {
    /// Current frame header (if parsed).
    frame_header: Option<FrameHeader>,
    /// Current loop filter parameters.
    loop_filter: LoopFilterParams,
    /// Current CDEF parameters.
    cdef: CdefParams,
    /// Current quantization parameters.
    quantization: QuantizationParams,
    /// Current tile info.
    tile_info: Option<TileInfo>,
    /// Frame is intra-only.
    frame_is_intra: bool,
}

impl DecoderState {
    fn new() -> Self {
        Self::default()
    }

    fn reset(&mut self) {
        self.frame_header = None;
        self.tile_info = None;
    }
}

/// AV1 decoder.
///
/// Scope: keyframe / intra-only reconstruction is implemented for 8-bit
/// 4:2:0 (profile 0) and verified bit-exact against dav1d and aomdec
/// (including deblocking and CDEF); inter frames and other profiles fail
/// honestly.
#[derive(Debug)]
pub struct Av1Decoder {
    /// Decoder configuration.
    config: DecoderConfig,
    /// Current sequence header (legacy metadata parser, for probing).
    sequence_header: Option<SequenceHeader>,
    /// Spec-exact sequence header state for the real decode path.
    kf_seq: Option<kf::hdr::SeqHdr>,
    /// Decoded frame output queue.
    output_queue: Vec<VideoFrame>,
    /// Reference frame slots (`refresh_frame_flags` targets).
    ref_frames: [Option<VideoFrame>; 8],
    /// Decoder is in flush mode.
    flushing: bool,
    /// Frame counter.
    frame_count: u64,
    /// Probe state.
    state: DecoderState,
}

impl Av1Decoder {
    /// Create a new AV1 decoder.
    ///
    /// # Errors
    ///
    /// Returns error if decoder initialization fails.
    pub fn new(config: DecoderConfig) -> CodecResult<Self> {
        let mut decoder = Self {
            config,
            sequence_header: None,
            kf_seq: None,
            output_queue: Vec::new(),
            ref_frames: Default::default(),
            flushing: false,
            frame_count: 0,
            state: DecoderState::new(),
        };

        if let Some(extradata) = decoder.config.extradata.clone() {
            decoder.parse_extradata(&extradata)?;
        }

        Ok(decoder)
    }

    /// Parse codec extradata (usually a sequence header OBU).
    fn parse_extradata(&mut self, data: &[u8]) -> CodecResult<()> {
        for obu_result in ObuIterator::new(data) {
            let (header, payload) = obu_result?;
            if header.obu_type == ObuType::SequenceHeader {
                self.sequence_header = Some(SequenceHeader::parse(payload)?);
                self.kf_seq = Some(kf::hdr::SeqHdr::parse(payload)?);
                break;
            }
        }
        Ok(())
    }

    /// Best-effort legacy header probing: keeps `current_frame_header()`,
    /// `loop_filter_params()` etc. observable. Parse failures here are
    /// non-fatal — the real decode path performs its own exact parsing.
    fn update_probe_state(&mut self, data: &[u8]) {
        for obu_result in ObuIterator::new(data) {
            let Ok((header, payload)) = obu_result else {
                return;
            };
            match header.obu_type {
                ObuType::SequenceHeader => {
                    if let Ok(seq) = SequenceHeader::parse(payload) {
                        self.sequence_header = Some(seq);
                    }
                }
                ObuType::FrameHeader | ObuType::Frame => {
                    if let Some(ref seq) = self.sequence_header {
                        if let Ok(frame_header) = FrameHeader::parse(payload, seq) {
                            self.state.frame_is_intra = frame_header.frame_is_intra;
                            self.state.loop_filter = frame_header.loop_filter.clone();
                            self.state.cdef = frame_header.cdef.clone();
                            self.state.quantization = frame_header.quantization.clone();
                            self.state.tile_info = Some(frame_header.tile_info.clone());
                            self.state.frame_header = Some(frame_header);
                        }
                    }
                    return;
                }
                _ => {}
            }
        }
    }

    /// Decode a temporal unit to pixels via the keyframe/intra pipeline.
    fn decode_temporal_unit(&mut self, data: &[u8], pts: i64) -> CodecResult<()> {
        self.state.reset();
        self.update_probe_state(data);

        match kf::decode_temporal_unit(data, &mut self.kf_seq)? {
            TuOutcome::Frame(decoded, fh) => {
                let mut frame = Self::planes_to_video_frame(&decoded);
                // Intra-only frames are also independently decodable; the
                // public FrameType enum has no separate intra-only variant.
                frame.frame_type = FrameType::Key;
                debug_assert!(fh.frame_is_intra);
                frame.timestamp = Timestamp::new(pts, Rational::new(1, 1000));
                self.frame_count += 1;

                for (i, slot) in self.ref_frames.iter_mut().enumerate() {
                    if fh.refresh_frame_flags & (1 << i) != 0 {
                        *slot = Some(frame.clone());
                    }
                }
                if fh.show_frame {
                    self.output_queue.push(frame);
                }
                Ok(())
            }
            TuOutcome::ShowExisting(idx) => {
                if let Some(ref frame) = self.ref_frames[idx as usize] {
                    let mut output = frame.clone();
                    output.timestamp = Timestamp::new(pts, Rational::new(1, 1000));
                    self.output_queue.push(output);
                    Ok(())
                } else {
                    Err(CodecError::InvalidBitstream(format!(
                        "AV1 show_existing_frame: reference slot {idx} holds no decoded frame"
                    )))
                }
            }
            TuOutcome::NoFrame => Ok(()),
        }
    }

    /// Crops the MI-aligned reconstruction planes to display size and
    /// assembles a [`VideoFrame`].
    fn planes_to_video_frame(decoded: &kf::DecodedIntraFrame) -> VideoFrame {
        let w = decoded.width;
        let h = decoded.height;
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);

        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, w as u32, h as u32);
        let dims = [(w, h), (cw, ch), (cw, ch)];
        for (plane, &(pw, ph)) in decoded.planes.iter().zip(dims.iter()) {
            let mut data = Vec::with_capacity(pw * ph);
            for row in 0..ph {
                let start = row * plane.stride;
                data.extend_from_slice(&plane.data[start..start + pw]);
            }
            frame
                .planes
                .push(Plane::with_dimensions(data, pw, pw as u32, ph as u32));
        }
        frame
    }

    /// Determine pixel format from sequence header.
    fn determine_pixel_format(seq: &SequenceHeader) -> PixelFormat {
        let cc = &seq.color_config;
        if cc.mono_chrome {
            return PixelFormat::Gray8;
        }
        match (cc.bit_depth, cc.subsampling_x, cc.subsampling_y) {
            (8, true, false) => PixelFormat::Yuv422p,
            (8, false, false) => PixelFormat::Yuv444p,
            (10, true, true) => PixelFormat::Yuv420p10le,
            (12, true, true) => PixelFormat::Yuv420p12le,
            // Default to YUV420p for 8-bit 4:2:0 and any other unhandled cases
            _ => PixelFormat::Yuv420p,
        }
    }

    /// Get the current frame header if available.
    #[must_use]
    pub fn current_frame_header(&self) -> Option<&FrameHeader> {
        self.state.frame_header.as_ref()
    }

    /// Get the current sequence header if available.
    #[must_use]
    pub fn current_sequence_header(&self) -> Option<&SequenceHeader> {
        self.sequence_header.as_ref()
    }

    /// Get the current loop filter parameters.
    #[must_use]
    pub fn loop_filter_params(&self) -> &LoopFilterParams {
        &self.state.loop_filter
    }

    /// Get the current CDEF parameters.
    #[must_use]
    pub fn cdef_params(&self) -> &CdefParams {
        &self.state.cdef
    }

    /// Get the current quantization parameters.
    #[must_use]
    pub fn quantization_params(&self) -> &QuantizationParams {
        &self.state.quantization
    }

    /// Get the current tile info if available.
    #[must_use]
    pub fn tile_info(&self) -> Option<&TileInfo> {
        self.state.tile_info.as_ref()
    }

    /// Get decoded frame count.
    #[must_use]
    pub const fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

impl VideoDecoder for Av1Decoder {
    fn codec(&self) -> CodecId {
        CodecId::Av1
    }

    fn send_packet(&mut self, data: &[u8], pts: i64) -> CodecResult<()> {
        if self.flushing {
            return Err(CodecError::InvalidParameter(
                "Cannot send packet while flushing".to_string(),
            ));
        }
        self.decode_temporal_unit(data, pts)
    }

    fn receive_frame(&mut self) -> CodecResult<Option<VideoFrame>> {
        if self.output_queue.is_empty() {
            if self.flushing {
                return Err(CodecError::Eof);
            }
            return Ok(None);
        }
        Ok(Some(self.output_queue.remove(0)))
    }

    fn flush(&mut self) -> CodecResult<()> {
        self.flushing = true;
        Ok(())
    }

    fn reset(&mut self) {
        self.output_queue.clear();
        self.ref_frames = Default::default();
        self.flushing = false;
        self.frame_count = 0;
        self.state.reset();
    }

    fn output_format(&self) -> Option<PixelFormat> {
        self.sequence_header
            .as_ref()
            .map(Self::determine_pixel_format)
    }

    fn dimensions(&self) -> Option<(u32, u32)> {
        if let Some(ref seq) = self.kf_seq {
            return Some((seq.max_frame_width, seq.max_frame_height));
        }
        self.sequence_header
            .as_ref()
            .map(|seq| (seq.max_frame_width(), seq.max_frame_height()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let config = DecoderConfig::default();
        let decoder = Av1Decoder::new(config);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_decoder_codec_id() {
        let config = DecoderConfig::default();
        let decoder = Av1Decoder::new(config).expect("should succeed");
        assert_eq!(decoder.codec(), CodecId::Av1);
    }

    #[test]
    fn test_decoder_flush() {
        let config = DecoderConfig::default();
        let mut decoder = Av1Decoder::new(config).expect("should succeed");
        assert!(decoder.flush().is_ok());
    }

    #[test]
    fn test_send_while_flushing() {
        let config = DecoderConfig::default();
        let mut decoder = Av1Decoder::new(config).expect("should succeed");
        decoder.flush().expect("should succeed");
        let result = decoder.send_packet(&[], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_decoder_reset() {
        let config = DecoderConfig::default();
        let mut decoder = Av1Decoder::new(config).expect("should succeed");
        decoder.flush().expect("should succeed");
        decoder.reset();
        assert_eq!(decoder.frame_count(), 0);
        assert!(decoder.send_packet(&[], 0).is_ok());
    }

    #[test]
    fn test_initial_state() {
        let config = DecoderConfig::default();
        let decoder = Av1Decoder::new(config).expect("should succeed");
        assert!(decoder.current_frame_header().is_none());
        assert!(decoder.current_sequence_header().is_none());
        assert!(decoder.tile_info().is_none());
    }

    #[test]
    fn test_loop_filter_params() {
        let config = DecoderConfig::default();
        let decoder = Av1Decoder::new(config).expect("should succeed");
        let lf = decoder.loop_filter_params();
        assert!(!lf.is_enabled());
    }

    #[test]
    fn test_cdef_params() {
        let config = DecoderConfig::default();
        let decoder = Av1Decoder::new(config).expect("should succeed");
        let cdef = decoder.cdef_params();
        assert!(!cdef.is_enabled());
    }

    #[test]
    fn test_quantization_params() {
        let config = DecoderConfig::default();
        let decoder = Av1Decoder::new(config).expect("should succeed");
        let qp = decoder.quantization_params();
        assert_eq!(qp.base_q_idx, 0);
    }

    /// A hand-built, spec-valid AV1 sequence-header OBU payload:
    /// profile=0, still_picture=1, reduced_still_picture_header=1,
    /// seq_level_idx[0]=0, frame_width_bits=6, frame_height_bits=6,
    /// max_frame_width_minus_1=63, max_frame_height_minus_1=63 (64x64),
    /// 8-bit color, not monochrome, no color description, limited range,
    /// 4:2:0 (implied by profile 0), no separate UV delta Q, no film grain.
    const AV1_SEQ_HEADER_64X64: [u8; 5] = [0x18, 0x15, 0x7F, 0xFC, 0x00];

    /// A hand-built, spec-valid AV1 frame-header OBU payload matching
    /// [`AV1_SEQ_HEADER_64X64`]: render size same as frame size, base_q_idx=0
    /// with no coded deltas (=> `coded_lossless`), no segmentation,
    /// reduced_tx_set=0, uniform tile spacing with exactly one 64x64
    /// superblock.
    const AV1_FRAME_HEADER_64X64: [u8; 2] = [0x00, 0x01];

    /// Builds a temporal unit: `SequenceHeader` OBU followed by a
    /// `FrameHeader` OBU (deliberately WITHOUT any tile group data).
    fn build_headers_only_temporal_unit() -> Vec<u8> {
        use crate::av1::obu::{encode_leb128, ObuHeader, ObuType};

        let mut data = Vec::new();

        let seq_obu = ObuHeader {
            obu_type: ObuType::SequenceHeader,
            has_extension: false,
            has_size: true,
            temporal_id: 0,
            spatial_id: 0,
        };
        data.extend(seq_obu.to_bytes());
        data.extend(encode_leb128(AV1_SEQ_HEADER_64X64.len() as u64));
        data.extend(&AV1_SEQ_HEADER_64X64);

        let frame_header_obu = ObuHeader {
            obu_type: ObuType::FrameHeader,
            has_extension: false,
            has_size: true,
            temporal_id: 0,
            spatial_id: 0,
        };
        data.extend(frame_header_obu.to_bytes());
        data.extend(encode_leb128(AV1_FRAME_HEADER_64X64.len() as u64));
        data.extend(&AV1_FRAME_HEADER_64X64);

        data
    }

    /// Regression test for the blank-frame fabrication bug: a frame header
    /// with NO tile data must produce an honest error, not a zeroed frame.
    #[test]
    fn test_headers_without_tiles_error_honestly_no_blank_frame() {
        let data = build_headers_only_temporal_unit();

        let config = DecoderConfig::default();
        let mut decoder = Av1Decoder::new(config).expect("should succeed");

        let result = decoder.send_packet(&data, 0);
        assert!(
            matches!(result, Err(CodecError::InvalidBitstream(_))),
            "expected honest error for a frame header without tile data, got {result:?}"
        );

        // No fabricated blank frame may be emitted.
        let frame = decoder.receive_frame().expect("should succeed");
        assert!(frame.is_none(), "no fabricated blank frame may be output");
    }

    /// Header/metadata probing keeps working even when pixel decode fails:
    /// dimensions and headers parsed before the error stay observable.
    #[test]
    fn test_headers_without_tiles_still_update_header_state() {
        let data = build_headers_only_temporal_unit();

        let config = DecoderConfig::default();
        let mut decoder = Av1Decoder::new(config).expect("should succeed");
        let result = decoder.send_packet(&data, 0);
        assert!(result.is_err());

        assert_eq!(
            decoder.dimensions(),
            Some((64, 64)),
            "sequence header dimensions should still be captured before the honest error"
        );
        assert!(
            decoder.current_sequence_header().is_some(),
            "sequence header should still be captured before the honest error"
        );
        assert!(
            decoder.current_frame_header().is_some(),
            "frame header should still be captured before the honest error"
        );
    }

    /// Builds a headers-only temporal unit from a REAL encoder stream by
    /// re-wrapping the frame OBU's uncompressed header (without its tile
    /// data) as a standalone FrameHeader OBU.
    fn build_real_headers_only_temporal_unit() -> Vec<u8> {
        use crate::av1::obu::{encode_leb128, ObuHeader};

        let tu: &[u8] = include_bytes!("kf/testdata/s1_svt64.obu");
        let mut out = Vec::new();
        let mut seq: Option<kf::hdr::SeqHdr> = None;
        for obu in ObuIterator::new(tu) {
            let (header, payload) = obu.expect("obu parse");
            match header.obu_type {
                ObuType::SequenceHeader => {
                    seq = Some(kf::hdr::SeqHdr::parse(payload).expect("seq"));
                    let hdr = ObuHeader {
                        obu_type: ObuType::SequenceHeader,
                        has_extension: false,
                        has_size: true,
                        temporal_id: 0,
                        spatial_id: 0,
                    };
                    out.extend(hdr.to_bytes());
                    out.extend(encode_leb128(payload.len() as u64));
                    out.extend_from_slice(payload);
                }
                ObuType::Frame => {
                    let s = seq.as_ref().expect("seq before frame");
                    let fh = kf::hdr::FrameHdr::parse(payload, s).expect("frame hdr");
                    let header_bytes = fh.header_bits.div_ceil(8);
                    let hdr = ObuHeader {
                        obu_type: ObuType::FrameHeader,
                        has_extension: false,
                        has_size: true,
                        temporal_id: 0,
                        spatial_id: 0,
                    };
                    out.extend(hdr.to_bytes());
                    out.extend(encode_leb128(header_bytes as u64));
                    out.extend_from_slice(&payload[..header_bytes]);
                }
                _ => {}
            }
        }
        out
    }

    #[test]
    fn test_error_message_names_the_gap() {
        let data = build_real_headers_only_temporal_unit();

        let config = DecoderConfig::default();
        let mut decoder = Av1Decoder::new(config).expect("should succeed");
        let err = match decoder.send_packet(&data, 0) {
            Err(e) => e,
            Ok(()) => panic!("send_packet must not pretend to decode pixels"),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("without tile group data"),
            "error must state the limitation clearly, got: {msg}"
        );
    }

    /// End-to-end: a real SVT-AV1 keyframe temporal unit decodes to a real
    /// 64x64 YUV420 frame through the public decoder API.
    #[test]
    fn test_real_keyframe_decodes_to_pixels() {
        let data: &[u8] = include_bytes!("kf/testdata/s1_svt64.obu");
        let reference: &[u8] = include_bytes!("kf/testdata/s1_svt64.yuv");

        let config = DecoderConfig::default();
        let mut decoder = Av1Decoder::new(config).expect("should succeed");
        decoder.send_packet(data, 0).expect("keyframe must decode");
        let frame = decoder
            .receive_frame()
            .expect("should succeed")
            .expect("one frame");
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 64);
        assert_eq!(frame.frame_type, FrameType::Key);
        assert_eq!(frame.planes.len(), 3);
        // Y plane must match the dav1d/aomdec reference bit-exactly.
        assert_eq!(frame.planes[0].data.as_slice(), &reference[..64 * 64]);
        assert_eq!(decoder.frame_count(), 1);
    }

    /// Truncated real streams must never produce frames.
    #[test]
    fn test_truncated_real_stream_errors() {
        let data: &[u8] = include_bytes!("kf/testdata/s1_svt64.obu");
        for cut in [3usize, 10, 40, data.len() - 5] {
            let config = DecoderConfig::default();
            let mut decoder = Av1Decoder::new(config).expect("should succeed");
            let _ = decoder.send_packet(&data[..cut], 0);
            let frame = decoder.receive_frame().unwrap_or(None);
            assert!(
                frame.is_none(),
                "truncated stream (len {cut}) must not output a frame"
            );
        }
    }
}
