//! Vorbis audio decoder.
//!
//! Decodes audio packets produced by this crate's [`VorbisEncoder`]
//! (`crate::vorbis::encoder`) back to floating-point PCM samples.
//!
//! # Honest status: full Vorbis I decode is NOT implemented
//!
//! Per `docs/codec_status.md`, full Vorbis I decoding (setup-header
//! codebook/floor/residue/mapping/mode parsing, reverse codebook decode,
//! floor-curve reconstruction, channel coupling, window switching) is
//! **not implemented** and is deferred to 0.2.0. This decoder handles only
//! the *OxiMedia simplified packet format* emitted by `VorbisEncoder`
//! (fixed 2048-sample long blocks, raw little-endian i16 floor amplitudes
//! and residue codes per channel).
//!
//! Streams produced by standards-compliant encoders (libvorbis etc.) are
//! rejected with an honest [`CodecError::UnsupportedFeature`] error at the
//! setup header — instead of silently discarding the setup data and then
//! misinterpreting real Vorbis audio packets as the simplified format
//! (which would produce garbage output).
//!
//! [`VorbisEncoder`]: crate::vorbis::encoder::VorbisEncoder

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]

use super::mdct::MdctTwiddles;
use super::residue::{ResidueConfig, ResidueDecoder, ResidueType};
use crate::error::{CodecError, CodecResult};

// =============================================================================
// Vorbis header types
// =============================================================================

/// Parsed Vorbis identification header fields.
#[derive(Clone, Debug)]
pub struct VorbisIdHeader {
    /// Number of audio channels.
    pub channels: u8,
    /// Audio sample rate in Hz.
    pub sample_rate: u32,
    /// Maximum bitrate hint (-1 = unset).
    pub bitrate_max: i32,
    /// Nominal bitrate (-1 = unset).
    pub bitrate_nominal: i32,
    /// Minimum bitrate hint (-1 = unset).
    pub bitrate_min: i32,
    /// log2 of the short block size.
    pub blocksize_0: u8,
    /// log2 of the long block size.
    pub blocksize_1: u8,
}

/// Parsed Vorbis comment header.
#[derive(Clone, Debug, Default)]
pub struct VorbisCommentHeader {
    /// Encoder vendor string.
    pub vendor: String,
    /// Comment tags as (KEY, value) pairs.
    pub comments: Vec<(String, String)>,
}

/// Discriminant returned by `process_header_packet`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VorbisHeaderType {
    /// Packet type 1 — identification.
    Identification,
    /// Packet type 3 — comment.
    Comment,
    /// Packet type 5 — setup.
    Setup,
}

/// State of a VorbisDecoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecoderState {
    /// Waiting for identification header.
    AwaitingIdHeader,
    /// Waiting for comment header.
    AwaitingCommentHeader,
    /// Waiting for setup header.
    AwaitingSetupHeader,
    /// Ready to decode audio packets.
    ReadyForAudio,
}

/// Vorbis audio decoder.
pub struct VorbisDecoder {
    /// Sample rate (filled from ID header).
    pub sample_rate: u32,
    /// Channel count (filled from ID header).
    pub channels: u8,
    /// Long block size.
    block_size: usize,
    /// MDCT for long block.
    mdct: MdctTwiddles,
    /// Residue decoder.
    residue_dec: ResidueDecoder,
    /// Number of floor x-positions (from setup).
    floor_x_count: usize,
    /// Overlap-add accumulation buffer (per channel).
    ola_buf: Vec<Vec<f64>>,
    /// Current decoder state.
    pub state: DecoderState,
    /// Number of audio packets decoded.
    pub packets_decoded: u64,
    /// Stored identification header (after parsing).
    stored_id_header: Option<VorbisIdHeader>,
    /// Stored comment header (after parsing).
    stored_comment_header: Option<VorbisCommentHeader>,
}

impl VorbisDecoder {
    /// Create a new, uninitialised decoder.
    ///
    /// Feed the three header packets via `decode_packet` before audio.
    #[must_use]
    pub fn new() -> Self {
        let block_size = 2048;
        Self {
            sample_rate: 0,
            channels: 0,
            block_size,
            mdct: MdctTwiddles::new(block_size),
            residue_dec: ResidueDecoder::new(
                ResidueConfig {
                    residue_type: ResidueType::Type1,
                    begin: 0,
                    end: 1024,
                    partition_size: 32,
                    classifications: 4,
                    classbook: 0,
                },
                0.2, // default step — overridden when we parse setup
            ),
            floor_x_count: 16,
            ola_buf: Vec::new(),
            state: DecoderState::AwaitingIdHeader,
            packets_decoded: 0,
            stored_id_header: None,
            stored_comment_header: None,
        }
    }

    /// Decode one Vorbis packet.
    ///
    /// - Header packets are parsed for configuration.
    /// - Audio packets in the *OxiMedia simplified format* (as produced by
    ///   [`VorbisEncoder`](crate::vorbis::encoder::VorbisEncoder)) are
    ///   decoded to interleaved f32 PCM.
    ///
    /// Returns `None` for header packets, `Some(Vec<f32>)` for audio.
    ///
    /// Standards-compliant Vorbis streams are rejected with an honest
    /// [`CodecError::UnsupportedFeature`] at the setup header — see the
    /// module documentation (full Vorbis I decode is deferred to 0.2.0).
    ///
    /// # Errors
    ///
    /// Returns `CodecError` if the packet is malformed or requires the
    /// unimplemented full Vorbis I decode path.
    pub fn decode_packet(&mut self, data: &[u8]) -> CodecResult<Option<Vec<f32>>> {
        if data.is_empty() {
            return Err(CodecError::InvalidBitstream(
                "Empty Vorbis packet".to_string(),
            ));
        }

        let packet_type = data[0];

        match packet_type {
            1 => self.parse_id_header(data),
            3 => self.parse_comment_header(data),
            5 => self.parse_setup_header(data),
            0 => self.decode_audio(data),
            _ => Err(CodecError::InvalidBitstream(format!(
                "Unknown Vorbis packet type: {packet_type}"
            ))),
        }
    }

    // ------------------------------------------------------------------
    // Structured header API
    // ------------------------------------------------------------------

    /// Process a Vorbis header packet and return which header type was parsed.
    ///
    /// This is an alternative entry-point to `decode_packet` for callers that
    /// only need the structured header information without PCM output.
    ///
    /// # Errors
    ///
    /// Returns `CodecError` if the packet is malformed or received out of order.
    pub fn process_header_packet(&mut self, data: &[u8]) -> CodecResult<VorbisHeaderType> {
        if data.is_empty() {
            return Err(CodecError::InvalidBitstream(
                "Empty Vorbis header packet".to_string(),
            ));
        }
        match data[0] {
            1 => {
                self.parse_id_header(data)?;
                Ok(VorbisHeaderType::Identification)
            }
            3 => {
                self.parse_comment_header_full(data)?;
                Ok(VorbisHeaderType::Comment)
            }
            5 => {
                self.parse_setup_header(data)?;
                Ok(VorbisHeaderType::Setup)
            }
            t => Err(CodecError::InvalidBitstream(format!(
                "Not a Vorbis header packet (type byte {t})"
            ))),
        }
    }

    /// Return a reference to the parsed identification header, if available.
    pub fn id_header(&self) -> Option<&VorbisIdHeader> {
        self.stored_id_header.as_ref()
    }

    /// Return a reference to the parsed comment header, if available.
    pub fn comment_header(&self) -> Option<&VorbisCommentHeader> {
        self.stored_comment_header.as_ref()
    }

    /// Returns `true` once all three Vorbis header packets have been processed
    /// and the decoder is ready for audio packets.
    pub fn is_ready(&self) -> bool {
        self.state == DecoderState::ReadyForAudio
    }

    /// Validate a Vorbis audio packet structurally, then report the decode
    /// gap honestly.
    ///
    /// This checks that the decoder is ready and that the packet starts with
    /// the audio-packet type byte (`0x00`), then **always returns an honest
    /// [`CodecError::UnsupportedFeature`] error**: a stateless (`&self`)
    /// audio decode is not implemented — full Vorbis I decode is deferred to
    /// 0.2.0, and even the OxiMedia simplified format requires the stateful
    /// MDCT/overlap-add context that only [`Self::decode_packet`] holds.
    ///
    /// An earlier revision returned `Ok(Vec::new())` here, silently
    /// pretending a packet decoded to zero samples; that misleading no-op
    /// has been replaced by this explicit error. Use
    /// [`Self::decode_packet`] for actual (simplified-format) decoding.
    ///
    /// # Errors
    ///
    /// Returns `CodecError::InvalidBitstream` if the decoder is not yet
    /// ready, the packet is empty, or the packet-type byte is not `0x00`;
    /// otherwise returns `CodecError::UnsupportedFeature` as described
    /// above.
    pub fn decode_audio_packet(&self, data: &[u8]) -> CodecResult<Vec<f32>> {
        if self.state != DecoderState::ReadyForAudio {
            return Err(CodecError::InvalidBitstream(
                "Audio packet received before all headers".to_string(),
            ));
        }
        if data.is_empty() {
            return Err(CodecError::InvalidBitstream(
                "Empty audio packet".to_string(),
            ));
        }
        if data[0] != 0x00 {
            return Err(CodecError::InvalidBitstream(format!(
                "Expected audio packet type 0x00, got {:#04x}",
                data[0]
            )));
        }
        Err(CodecError::UnsupportedFeature(
            "Vorbis full decode not implemented: stateless audio-packet decode is \
             unsupported (deferred to 0.2.0); use decode_packet(), which decodes the \
             OxiMedia simplified packet format with its stateful MDCT/overlap-add context"
                .to_string(),
        ))
    }

    // ------------------------------------------------------------------
    // Header parsing
    // ------------------------------------------------------------------

    fn parse_id_header(&mut self, data: &[u8]) -> CodecResult<Option<Vec<f32>>> {
        if self.state != DecoderState::AwaitingIdHeader {
            return Err(CodecError::InvalidBitstream(
                "Unexpected ID header".to_string(),
            ));
        }
        if data.len() < 23 || &data[1..7] != b"vorbis" {
            return Err(CodecError::InvalidBitstream(
                "Invalid Vorbis ID header".to_string(),
            ));
        }

        // Bytes: [0]=type, [1..7]=magic, [7..11]=version, [11]=channels,
        //        [12..16]=sample_rate, [16..28]=bitrates, [28]=block sizes, [29]=framing
        if data.len() < 29 {
            return Err(CodecError::InvalidBitstream(
                "ID header too short".to_string(),
            ));
        }

        self.channels = data[11];
        self.sample_rate = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);

        if self.channels == 0 || self.channels > 8 {
            return Err(CodecError::InvalidBitstream(format!(
                "Invalid channel count: {}",
                self.channels
            )));
        }

        let bitrate_max = i32::from_le_bytes([data[16], data[17], data[18], data[19]]);
        let bitrate_nominal = i32::from_le_bytes([data[20], data[21], data[22], data[23]]);
        let bitrate_min = i32::from_le_bytes([data[24], data[25], data[26], data[27]]);

        let block_sizes_byte = data[28];
        let log2_short = block_sizes_byte & 0x0F;
        let log2_long = (block_sizes_byte >> 4) & 0x0F;
        let short_sz = 1usize << log2_short;
        let long_sz = 1usize << log2_long;

        if long_sz < short_sz || long_sz < 4 {
            return Err(CodecError::InvalidBitstream(
                "Invalid block sizes in ID header".to_string(),
            ));
        }

        // Framing bit validation (byte [29] must have bit 0 set).
        if data.len() > 29 && (data[29] & 0x01) == 0 {
            return Err(CodecError::InvalidBitstream(
                "Vorbis ID header framing bit not set".to_string(),
            ));
        }

        self.stored_id_header = Some(VorbisIdHeader {
            channels: self.channels,
            sample_rate: self.sample_rate,
            bitrate_max,
            bitrate_nominal,
            bitrate_min,
            blocksize_0: log2_short,
            blocksize_1: log2_long,
        });

        self.block_size = long_sz;
        self.mdct = MdctTwiddles::new(long_sz);
        self.ola_buf = vec![vec![0.0f64; long_sz]; self.channels as usize];

        let _ = short_sz; // short block MDCT could be initialised here

        self.state = DecoderState::AwaitingCommentHeader;
        Ok(None)
    }

    fn parse_comment_header(&mut self, data: &[u8]) -> CodecResult<Option<Vec<f32>>> {
        self.parse_comment_header_full(data)?;
        Ok(None)
    }

    /// Full parse of the comment header packet (type byte 0x03).
    ///
    /// Stores the result in `self.stored_comment_header`.
    fn parse_comment_header_full(&mut self, data: &[u8]) -> CodecResult<()> {
        if self.state != DecoderState::AwaitingCommentHeader {
            return Err(CodecError::InvalidBitstream(
                "Unexpected comment header".to_string(),
            ));
        }
        // Minimum: 7 bytes magic + 4 bytes vendor length
        if data.len() < 11 || &data[0..7] != b"\x03vorbis" {
            return Err(CodecError::InvalidBitstream(
                "Invalid Vorbis comment header magic".to_string(),
            ));
        }

        let mut offset = 7usize;

        // Vendor string length (u32 LE)
        if offset + 4 > data.len() {
            return Err(CodecError::InvalidBitstream(
                "Comment header truncated at vendor length".to_string(),
            ));
        }
        let vendor_len = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        if offset + vendor_len > data.len() {
            return Err(CodecError::InvalidBitstream(
                "Comment header truncated in vendor string".to_string(),
            ));
        }
        let vendor = String::from_utf8_lossy(&data[offset..offset + vendor_len]).into_owned();
        offset += vendor_len;

        // Number of user comment fields (u32 LE)
        if offset + 4 > data.len() {
            return Err(CodecError::InvalidBitstream(
                "Comment header truncated at comment list length".to_string(),
            ));
        }
        let comment_count = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        let mut comments = Vec::with_capacity(comment_count);
        for _ in 0..comment_count {
            if offset + 4 > data.len() {
                return Err(CodecError::InvalidBitstream(
                    "Comment header truncated in comment length field".to_string(),
                ));
            }
            let field_len = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;

            if offset + field_len > data.len() {
                return Err(CodecError::InvalidBitstream(
                    "Comment header truncated in comment value".to_string(),
                ));
            }
            let raw = String::from_utf8_lossy(&data[offset..offset + field_len]).into_owned();
            offset += field_len;

            // Split on the first '=' to produce (KEY, value); KEY is uppercased per spec.
            let (key, value) = if let Some(eq_pos) = raw.find('=') {
                let k = raw[..eq_pos].to_ascii_uppercase();
                let v = raw[eq_pos + 1..].to_string();
                (k, v)
            } else {
                (raw.to_ascii_uppercase(), String::new())
            };
            comments.push((key, value));
        }

        self.stored_comment_header = Some(VorbisCommentHeader { vendor, comments });
        self.state = DecoderState::AwaitingSetupHeader;
        Ok(())
    }

    /// Parse the Vorbis setup header (packet type 5).
    ///
    /// # Honest limitation
    ///
    /// Real Vorbis I setup-header parsing (codebooks, floors, residues,
    /// mappings, modes — Vorbis I spec §4.2.4) is **not implemented**
    /// (deferred to 0.2.0). The only setup packet accepted is the minimal
    /// placeholder emitted by
    /// [`VorbisEncoder`](crate::vorbis::encoder::VorbisEncoder), which
    /// marks the stream as using the OxiMedia simplified packet format.
    ///
    /// Any other setup header — i.e. any standards-compliant Vorbis stream
    /// carrying real codebook data — is rejected with
    /// [`CodecError::UnsupportedFeature`]. An earlier revision silently
    /// discarded the setup payload and flipped to `ReadyForAudio`, which
    /// made subsequent real Vorbis audio packets decode to garbage; this
    /// explicit error replaces that silent no-op.
    fn parse_setup_header(&mut self, data: &[u8]) -> CodecResult<Option<Vec<f32>>> {
        if self.state != DecoderState::AwaitingSetupHeader {
            return Err(CodecError::InvalidBitstream(
                "Unexpected setup header".to_string(),
            ));
        }
        if data.len() < 7 || &data[0..7] != b"\x05vorbis" {
            return Err(CodecError::InvalidBitstream(
                "Invalid Vorbis setup header magic".to_string(),
            ));
        }
        // The OxiMedia encoder emits exactly:
        //   type (0x05) + "vorbis" + codebook_count_minus_1 (0x00) + framing (0x01)
        if data[7..] != [0x00, 0x01] {
            return Err(CodecError::UnsupportedFeature(
                "Vorbis full decode not implemented: setup-header codebook/floor/residue \
                 parsing is unsupported (deferred to 0.2.0); only the simplified setup \
                 packet produced by OxiMedia's VorbisEncoder is accepted"
                    .to_string(),
            ));
        }
        self.state = DecoderState::ReadyForAudio;
        Ok(None)
    }

    // ------------------------------------------------------------------
    // Audio decode
    // ------------------------------------------------------------------

    /// Decode an audio packet in the *OxiMedia simplified format*.
    ///
    /// This is only reachable after `parse_setup_header` accepted the
    /// OxiMedia placeholder setup packet, which guarantees the stream was
    /// produced by this crate's `VorbisEncoder` (raw little-endian i16
    /// floor amplitudes + residue codes per channel). It is NOT a
    /// standards-compliant Vorbis I audio decode — see the module docs.
    fn decode_audio(&mut self, data: &[u8]) -> CodecResult<Option<Vec<f32>>> {
        if self.state != DecoderState::ReadyForAudio {
            return Err(CodecError::InvalidBitstream(
                "Audio packet received before headers".to_string(),
            ));
        }

        let ch = self.channels as usize;
        let n = self.block_size;
        let m = n / 2; // MDCT output size
        let floor_x = self.floor_x_count;

        // Each channel: floor_x * 2 bytes (i16 amplitudes) + m * 2 bytes (i16 codes)
        let per_channel = floor_x * 2 + m * 2;
        let expected = 1 + ch * per_channel; // +1 for packet type byte

        if data.len() < expected {
            return Err(CodecError::InvalidBitstream(format!(
                "Audio packet too short: expected >= {expected}, got {}",
                data.len()
            )));
        }

        let mut interleaved = Vec::with_capacity(m * ch);
        let mut offset = 1usize; // skip packet type byte

        for _c in 0..ch {
            // Read floor amplitudes
            let mut amps = Vec::with_capacity(floor_x);
            for _ in 0..floor_x {
                if offset + 2 > data.len() {
                    return Err(CodecError::InvalidBitstream(
                        "Truncated floor data".to_string(),
                    ));
                }
                let a = i16::from_le_bytes([data[offset], data[offset + 1]]);
                amps.push(a);
                offset += 2;
            }

            // Read residue codes
            let mut codes = Vec::with_capacity(m);
            for _ in 0..m {
                if offset + 2 > data.len() {
                    return Err(CodecError::InvalidBitstream(
                        "Truncated residue data".to_string(),
                    ));
                }
                let c = i16::from_le_bytes([data[offset], data[offset + 1]]);
                codes.push(c);
                offset += 2;
            }

            // Reconstruct coefficients = floor + residue
            let residue = self.residue_dec.decode(&codes);

            // Simplified floor reconstruction: linear interpolation
            let coeffs: Vec<f64> = residue
                .iter()
                .enumerate()
                .map(|(i, &r)| {
                    // Use a flat floor approximation for simplicity
                    let floor_val = if !amps.is_empty() {
                        let idx = (i * floor_x / m).min(floor_x - 1);
                        f64::from(amps[idx]) * 0.1 // scale down floor
                    } else {
                        0.0
                    };
                    r + floor_val
                })
                .collect();

            // Inverse MDCT
            let mut time_samples = self.mdct.inverse(&coeffs);

            // Apply synthesis window
            self.mdct.apply_synthesis_window(&mut time_samples);

            // Overlap-add with previous block
            let ola = &mut self.ola_buf[_c];
            let output_start = n / 4; // output the centre `n/2` samples
            for i in 0..n {
                let combined = time_samples[i] + ola[i];
                if i >= output_start && i < output_start + m {
                    interleaved.push(combined as f32);
                }
                // Update OLA buffer for next block
                ola[i] = if i + m < n { time_samples[i + m] } else { 0.0 };
            }
        }

        self.packets_decoded += 1;

        // Re-interleave channels
        if ch > 1 {
            let samples_per_ch = interleaved.len() / ch;
            let mut reinterleaved = Vec::with_capacity(interleaved.len());
            for s in 0..samples_per_ch {
                for c in 0..ch {
                    reinterleaved.push(interleaved[c * samples_per_ch + s]);
                }
            }
            Ok(Some(reinterleaved))
        } else {
            Ok(Some(interleaved))
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vorbis::encoder::{VorbisConfig, VorbisEncoder, VorbisQuality};

    fn make_enc_dec() -> (VorbisEncoder, VorbisDecoder) {
        let cfg = VorbisConfig {
            sample_rate: 44100,
            channels: 2,
            quality: VorbisQuality::Q5,
        };
        let enc = VorbisEncoder::new(cfg).expect("encoder");
        let dec = VorbisDecoder::new();
        (enc, dec)
    }

    #[test]
    fn test_vorbis_decoder_new_state() {
        let dec = VorbisDecoder::new();
        assert_eq!(dec.state, DecoderState::AwaitingIdHeader);
        assert_eq!(dec.packets_decoded, 0);
    }

    #[test]
    fn test_vorbis_decoder_parse_headers() {
        let (mut enc, mut dec) = make_enc_dec();
        let headers = enc.headers();
        for h in &headers {
            let result = dec.decode_packet(&h.data).expect("header decode");
            assert!(result.is_none(), "Header decode should return None");
        }
        assert_eq!(dec.state, DecoderState::ReadyForAudio);
    }

    #[test]
    fn test_vorbis_decoder_id_header_channels() {
        let (mut enc, mut dec) = make_enc_dec();
        let headers = enc.headers();
        dec.decode_packet(&headers[0].data).expect("id header");
        assert_eq!(dec.channels, 2);
        assert_eq!(dec.sample_rate, 44100);
    }

    #[test]
    fn test_vorbis_decoder_wrong_order_errors() {
        let (mut enc, mut dec) = make_enc_dec();
        let headers = enc.headers();
        // Try to feed comment header before ID header
        let result = dec.decode_packet(&headers[1].data);
        assert!(result.is_err(), "Should error on out-of-order header");
    }

    #[test]
    fn test_vorbis_decoder_empty_packet_errors() {
        let mut dec = VorbisDecoder::new();
        let result = dec.decode_packet(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_vorbis_decoder_audio_before_headers_errors() {
        let mut dec = VorbisDecoder::new();
        // Try to decode an audio packet (type=0) before headers
        let result = dec.decode_packet(&[0u8, 1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_vorbis_encoder_decoder_pipeline() {
        let (mut enc, mut dec) = make_enc_dec();

        // Feed headers to decoder
        let headers = enc.headers();
        for h in &headers {
            dec.decode_packet(&h.data).expect("header");
        }

        // Encode one full block of silence
        let silence = vec![0.0f32; 2048 * 2]; // 2048 stereo samples
        let pkts = enc.encode_interleaved(&silence).expect("encode");

        // Try decoding each packet
        for pkt in &pkts {
            let result = dec.decode_packet(&pkt.data);
            // May succeed or fail (truncated packet check) — just no panic
            let _ = result;
        }
    }

    #[test]
    fn test_vorbis_packets_decoded_counter() {
        let (mut enc, mut dec) = make_enc_dec();
        let headers = enc.headers();
        for h in &headers {
            dec.decode_packet(&h.data).expect("header");
        }
        assert_eq!(dec.packets_decoded, 0); // headers don't count
    }

    // ------------------------------------------------------------------
    // Tests for the new structured header API
    // ------------------------------------------------------------------

    #[test]
    fn test_process_header_packet_returns_correct_types() {
        let (mut enc, mut dec) = make_enc_dec();
        let headers = enc.headers();
        assert_eq!(headers.len(), 3);

        let t0 = dec
            .process_header_packet(&headers[0].data)
            .expect("id header via process_header_packet");
        assert_eq!(t0, VorbisHeaderType::Identification);

        let t1 = dec
            .process_header_packet(&headers[1].data)
            .expect("comment header via process_header_packet");
        assert_eq!(t1, VorbisHeaderType::Comment);

        let t2 = dec
            .process_header_packet(&headers[2].data)
            .expect("setup header via process_header_packet");
        assert_eq!(t2, VorbisHeaderType::Setup);
    }

    #[test]
    fn test_id_header_populated_after_processing() {
        let (mut enc, mut dec) = make_enc_dec();
        let headers = enc.headers();

        assert!(dec.id_header().is_none());

        dec.process_header_packet(&headers[0].data)
            .expect("process id header");

        let id = dec
            .id_header()
            .expect("id_header should be Some after parsing");
        assert_eq!(id.channels, 2);
        assert_eq!(id.sample_rate, 44100);
        // blocksize_1 >= blocksize_0 per spec
        assert!(id.blocksize_1 >= id.blocksize_0);
    }

    #[test]
    fn test_comment_header_populated_after_processing() {
        let (mut enc, mut dec) = make_enc_dec();
        let headers = enc.headers();

        dec.process_header_packet(&headers[0].data).expect("id");
        assert!(dec.comment_header().is_none());
        dec.process_header_packet(&headers[1].data)
            .expect("comment");

        let ch = dec.comment_header().expect("comment_header should be Some");
        // vendor string may be empty but must not panic
        let _ = ch.vendor.len();
    }

    #[test]
    fn test_is_ready_only_after_all_three_headers() {
        let (mut enc, mut dec) = make_enc_dec();
        let headers = enc.headers();

        assert!(!dec.is_ready());
        dec.process_header_packet(&headers[0].data).expect("id");
        assert!(!dec.is_ready());
        dec.process_header_packet(&headers[1].data)
            .expect("comment");
        assert!(!dec.is_ready());
        dec.process_header_packet(&headers[2].data).expect("setup");
        assert!(dec.is_ready());
    }

    #[test]
    fn test_decode_audio_packet_before_headers_errors() {
        let dec = VorbisDecoder::new();
        let result = dec.decode_audio_packet(&[0x00u8, 0x01, 0x02]);
        assert!(result.is_err(), "should error before headers are processed");
    }

    #[test]
    fn test_decode_audio_packet_wrong_type_byte_errors() {
        let (mut enc, mut dec) = make_enc_dec();
        let headers = enc.headers();
        for h in &headers {
            dec.process_header_packet(&h.data).expect("header");
        }
        // Type byte 0x01 is not an audio packet
        let result = dec.decode_audio_packet(&[0x01u8, 0x00]);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_audio_packet_returns_honest_unsupported_error() {
        let (mut enc, mut dec) = make_enc_dec();
        let headers = enc.headers();
        for h in &headers {
            dec.process_header_packet(&h.data).expect("header");
        }
        // Structurally valid audio packet: the stateless entry point must
        // report the decode gap honestly instead of pretending the packet
        // decoded to zero samples (the old silent `Ok(Vec::new())` no-op).
        let result = dec.decode_audio_packet(&[0x00u8, 0x00, 0x00]);
        match result {
            Err(CodecError::UnsupportedFeature(msg)) => {
                assert!(
                    msg.contains("not implemented"),
                    "error must state the limitation clearly, got: {msg}"
                );
            }
            other => panic!("expected honest UnsupportedFeature error, got {other:?}"),
        }
    }

    #[test]
    fn test_real_vorbis_setup_header_rejected_honestly() {
        let (mut enc, mut dec) = make_enc_dec();
        let headers = enc.headers();
        dec.process_header_packet(&headers[0].data).expect("id");
        dec.process_header_packet(&headers[1].data)
            .expect("comment");

        // A standards-compliant setup header carries real codebook data
        // after the magic (each codebook starts with the 24-bit sync
        // pattern 0x564342 = "BCV"). The decoder cannot parse it, so it
        // must say so honestly rather than silently discarding it.
        let mut pkt = Vec::new();
        pkt.push(0x05);
        pkt.extend_from_slice(b"vorbis");
        pkt.push(3); // vorbis_codebook_count - 1
        pkt.extend_from_slice(&[0x42, 0x43, 0x56]); // "BCV" codebook sync
        pkt.extend_from_slice(&[0u8; 32]); // (truncated) codebook payload

        let result = dec.process_header_packet(&pkt);
        match result {
            Err(CodecError::UnsupportedFeature(msg)) => {
                assert!(
                    msg.contains("not implemented"),
                    "error must state the limitation clearly, got: {msg}"
                );
            }
            other => panic!("expected honest UnsupportedFeature error, got {other:?}"),
        }
        assert!(
            !dec.is_ready(),
            "decoder must not claim readiness after rejecting the setup header"
        );
    }

    #[test]
    fn test_setup_header_bad_magic_rejected() {
        let (mut enc, mut dec) = make_enc_dec();
        let headers = enc.headers();
        dec.process_header_packet(&headers[0].data).expect("id");
        dec.process_header_packet(&headers[1].data)
            .expect("comment");

        // Correct type byte but wrong magic string.
        let pkt = [0x05u8, b'x', b'o', b'r', b'b', b'i', b's', 0x00, 0x01];
        let result = dec.process_header_packet(&pkt);
        assert!(
            matches!(result, Err(CodecError::InvalidBitstream(_))),
            "bad setup magic must be InvalidBitstream, got {result:?}"
        );
        assert!(!dec.is_ready());
    }

    #[test]
    fn test_process_header_packet_empty_errors() {
        let mut dec = VorbisDecoder::new();
        assert!(dec.process_header_packet(&[]).is_err());
    }

    #[test]
    fn test_process_header_packet_non_header_type_errors() {
        let mut dec = VorbisDecoder::new();
        // Type byte 0x00 is an audio packet, not a header
        assert!(dec.process_header_packet(&[0x00u8, 0x01]).is_err());
    }

    /// Build a minimal but spec-compliant Vorbis ID header for unit testing.
    fn build_minimal_id_header(channels: u8, sample_rate: u32) -> Vec<u8> {
        let mut pkt = Vec::new();
        pkt.push(0x01); // type
        pkt.extend_from_slice(b"vorbis");
        pkt.extend_from_slice(&0u32.to_le_bytes()); // version
        pkt.push(channels);
        pkt.extend_from_slice(&sample_rate.to_le_bytes());
        pkt.extend_from_slice(&(-1i32).to_le_bytes()); // bitrate_max
        pkt.extend_from_slice(&128_000i32.to_le_bytes()); // bitrate_nominal
        pkt.extend_from_slice(&(-1i32).to_le_bytes()); // bitrate_min
                                                       // blocksize byte: low nibble = 8 (256), high nibble = 11 (2048)
        pkt.push((11u8 << 4) | 8u8);
        pkt.push(0x01); // framing bit set
        pkt
    }

    #[test]
    fn test_id_header_bitrates_parsed() {
        let pkt = build_minimal_id_header(1, 22050);
        let mut dec = VorbisDecoder::new();
        dec.process_header_packet(&pkt).expect("id header");
        let id = dec.id_header().expect("id_header");
        assert_eq!(id.channels, 1);
        assert_eq!(id.sample_rate, 22050);
        assert_eq!(id.bitrate_max, -1);
        assert_eq!(id.bitrate_nominal, 128_000);
        assert_eq!(id.bitrate_min, -1);
        assert_eq!(id.blocksize_0, 8);
        assert_eq!(id.blocksize_1, 11);
    }

    #[test]
    fn test_id_header_framing_bit_required() {
        let mut pkt = build_minimal_id_header(2, 44100);
        // Clear framing bit
        let last = pkt.last_mut().expect("pkt not empty");
        *last = 0x00;
        let mut dec = VorbisDecoder::new();
        let result = dec.process_header_packet(&pkt);
        assert!(result.is_err(), "missing framing bit must be an error");
    }
}
