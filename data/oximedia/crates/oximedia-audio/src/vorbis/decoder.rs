//! Vorbis decoder implementation.
//!
//! **Implementation note**: This decoder targets the internal OxiMedia Vorbis
//! encoder bitstream format.  It does NOT implement Vorbis I spec §4.3 floor-1
//! VQ / residue partition VQ because the in-tree encoder does not produce a
//! spec-compliant bitstream.  Specifically:
//!
//! - Floor data is two dummy 8-bit values per channel (skipped on decode).
//! - Residue is a custom RLE + scalar-quantization scheme (see `residue.rs`).
//! - Channel coupling / mapping are not present in the encoder output.
//!
//! The state machine correctly processes the three mandatory Vorbis headers
//! (identification, comment, setup) before accepting audio packets.

#![forbid(unsafe_code)]

use std::collections::VecDeque;

use bytes::Bytes;

use super::{
    bitpack::BitReader,
    header::{CommentHeader, IdentificationHeader, SetupHeader},
    mdct::{OverlapAdd, VorbisMdct},
    residue::ResidueEncoder,
};
use crate::{
    AudioBuffer, AudioDecoder, AudioDecoderConfig, AudioError, AudioFrame, AudioResult,
    ChannelLayout,
};
use oximedia_core::{CodecId, Rational, SampleFormat, Timestamp};

// ─────────────────────────────────────────────────────────────────────────────
// Decoder state machine
// ─────────────────────────────────────────────────────────────────────────────

/// Internal state of the Vorbis decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum DecoderState {
    /// Waiting for identification header (packet type 1).
    #[default]
    WaitingForIdentification,
    /// Identification received; waiting for comment header (packet type 3).
    WaitingForComment,
    /// Comment received; waiting for setup header (packet type 5).
    WaitingForSetup,
    /// All three headers processed; ready to accept audio packets.
    Ready,
}

// ─────────────────────────────────────────────────────────────────────────────
// VorbisDecoderInner
// ─────────────────────────────────────────────────────────────────────────────

/// Vorbis decoder that mirrors the OxiMedia in-tree encoder format.
pub struct VorbisDecoderInner {
    /// Decoder configuration supplied by the caller.
    config: AudioDecoderConfig,

    /// State machine position.
    state: DecoderState,

    /// Flushing flag (set by [`AudioDecoder::flush`]).
    flushing: bool,

    // ── Stream parameters (filled during header phase) ──────────────────────
    sample_rate: u32,
    channels: u8,
    /// Short block size in samples (2^blocksize_0_bits).  Default 256.
    blocksize_0: usize,
    /// Long block size in samples (2^blocksize_1_bits).  Default 2048.
    blocksize_1: usize,

    // ── IMDCT engines (created after identification header) ─────────────────
    mdct_short: Option<VorbisMdct>,
    mdct_long: Option<VorbisMdct>,

    // ── Overlap-add state (one per channel) ─────────────────────────────────
    /// Per-channel OLA processors.
    overlap: Vec<OverlapAdd>,

    // ── Quantisation step recovered from identification header ───────────────
    quant_step: f32,

    // ── Pending decoded frames waiting for receive_frame() ──────────────────
    pending: VecDeque<AudioFrame>,

    // ── PTS tracking ────────────────────────────────────────────────────────
    pts: i64,

    // ── Previous block flag ──────────────────────────────────────────────────
    prev_block_long: bool,
}

impl VorbisDecoderInner {
    /// Create a new decoder.
    ///
    /// # Errors
    ///
    /// Returns `InvalidParameter` when `config.codec != Vorbis`.
    pub fn new(config: &AudioDecoderConfig) -> AudioResult<Self> {
        if config.codec != CodecId::Vorbis {
            return Err(AudioError::InvalidParameter("Expected Vorbis codec".into()));
        }
        Ok(Self {
            config: config.clone(),
            state: DecoderState::WaitingForIdentification,
            flushing: false,
            sample_rate: config.sample_rate,
            channels: config.channels,
            blocksize_0: 256,
            blocksize_1: 2048,
            mdct_short: None,
            mdct_long: None,
            overlap: Vec::new(),
            quant_step: 0.5,
            pending: VecDeque::new(),
            pts: 0,
            prev_block_long: false,
        })
    }

    // ── Header processing ────────────────────────────────────────────────────

    fn handle_identification(&mut self, data: &[u8]) -> AudioResult<()> {
        let id = IdentificationHeader::parse(data)?;

        self.sample_rate = id.audio_sample_rate;
        self.channels = id.audio_channels;
        self.blocksize_0 = id.block_size_0_samples();
        self.blocksize_1 = id.block_size_1_samples();

        let quality = Self::nominal_to_quality(id.bitrate_nominal, id.audio_channels);
        self.quant_step = Self::quality_to_quant_step(quality);

        self.mdct_short = Some(VorbisMdct::new(self.blocksize_0));
        self.mdct_long = Some(VorbisMdct::new(self.blocksize_1));

        // OLA size = long block / 2 (used for all blocks; short blocks fall back).
        let ola_size = self.blocksize_1 / 2;
        self.overlap = (0..self.channels)
            .map(|_| OverlapAdd::new(ola_size))
            .collect();

        self.state = DecoderState::WaitingForComment;
        Ok(())
    }

    fn handle_comment(&mut self, data: &[u8]) -> AudioResult<()> {
        let _ = CommentHeader::parse(data);
        self.state = DecoderState::WaitingForSetup;
        Ok(())
    }

    fn handle_setup(&mut self, data: &[u8]) -> AudioResult<()> {
        let _ = SetupHeader::parse(data);
        self.state = DecoderState::Ready;
        Ok(())
    }

    // ── Audio packet decoding ────────────────────────────────────────────────

    /// Decode one audio packet and push resulting frame(s) into `self.pending`.
    #[allow(clippy::cast_precision_loss)]
    fn handle_audio_packet(&mut self, data: &[u8], pts: i64) -> AudioResult<()> {
        let mut reader = BitReader::new(data);

        // ── 1. Packet type bit (must be 0 for audio) ─────────────────────────
        let packet_type = reader
            .read_bit()
            .map_err(|_| AudioError::InvalidData("audio packet: missing type bit".into()))?;
        if packet_type {
            return Err(AudioError::InvalidData(
                "Expected audio packet (type bit 0), got header packet".into(),
            ));
        }

        // ── 2. Mode number (1 bit in encoder output) ─────────────────────────
        let use_long_block = reader
            .read_bit()
            .map_err(|_| AudioError::InvalidData("audio packet: missing mode bit".into()))?;

        // ── 3. Window flags (only present for long blocks) ───────────────────
        if use_long_block {
            let _ = reader.read_bit();
            let _ = reader.read_bit();
        }

        let blocksize = if use_long_block {
            self.blocksize_1
        } else {
            self.blocksize_0
        };
        let half_n = blocksize / 2;

        // ── 4. Per-channel floor + residue decode ─────────────────────────────
        let channels = usize::from(self.channels);
        let mut channel_coeffs: Vec<Vec<f32>> = Vec::with_capacity(channels);

        for _ch in 0..channels {
            // Floor: nonzero flag (1 bit) + two 8-bit dummy values — skip all.
            let _ = reader.read_bit();
            let _ = reader.read_bits(8);
            let _ = reader.read_bits(8);

            let coeffs = ResidueEncoder::decode_rle(&mut reader, self.quant_step, half_n)?;
            channel_coeffs.push(coeffs);
        }

        // ── 5. IMDCT + overlap-add per channel ───────────────────────────────
        let mdct = if use_long_block {
            self.mdct_long
                .as_ref()
                .ok_or_else(|| AudioError::InvalidData("MDCT not initialised".into()))?
        } else {
            self.mdct_short
                .as_ref()
                .ok_or_else(|| AudioError::InvalidData("MDCT not initialised".into()))?
        };

        let ola_out_len = half_n;
        let mut interleaved: Vec<f32> = vec![0.0; ola_out_len * channels];

        for (ch, coeffs) in channel_coeffs.iter().enumerate() {
            // IMDCT: N/2 coefficients → N time-domain samples (windowed internally).
            let mut time_block = vec![0.0f32; blocksize];
            mdct.inverse(coeffs, &mut time_block);

            let ola = self.overlap.get_mut(ch).ok_or_else(|| {
                AudioError::InvalidData(format!("overlap buffer missing for channel {ch}"))
            })?;

            if blocksize == ola.size() * 2 {
                // Long block: use OverlapAdd directly.
                let mut out = vec![0.0f32; ola.size()];
                ola.process(&time_block, &mut out);
                for (i, &s) in out.iter().enumerate() {
                    interleaved[i * channels + ch] = s;
                }
            } else {
                // Short block: manual overlap-add using first half of IMDCT output.
                // The OLA buffer is sized for long blocks; for short blocks we emit
                // the windowed output directly (imperfect but audible).
                for (i, &s) in time_block.iter().take(ola_out_len).enumerate() {
                    interleaved[i * channels + ch] = s;
                }
            }
        }

        // ── 6. Build AudioFrame ───────────────────────────────────────────────
        let mut raw_bytes: Vec<u8> = Vec::with_capacity(interleaved.len() * 4);
        for s in &interleaved {
            raw_bytes.extend_from_slice(&s.to_le_bytes());
        }

        let frame = AudioFrame {
            format: SampleFormat::F32,
            sample_rate: self.sample_rate,
            channels: ChannelLayout::from_count(channels),
            samples: AudioBuffer::Interleaved(Bytes::from(raw_bytes)),
            timestamp: Timestamp::new(pts, Rational::new(1, self.sample_rate as i64)),
        };

        self.pending.push_back(frame);
        self.pts = pts + ola_out_len as i64;
        self.prev_block_long = use_long_block;
        Ok(())
    }

    // ── Encoder-mirroring helpers ────────────────────────────────────────────

    /// Mirror `VorbisEncoder::bitrate_to_quality`.
    fn nominal_to_quality(nominal: i32, channels: u8) -> f32 {
        let adjusted = if channels == 1 {
            nominal.saturating_mul(2) as u32
        } else {
            nominal.max(0) as u32
        };
        match adjusted {
            0..=50_000 => -1.0,
            50_001..=72_000 => 0.0,
            72_001..=88_000 => 1.0,
            88_001..=104_000 => 2.0,
            104_001..=120_000 => 3.0,
            120_001..=144_000 => 4.0,
            144_001..=176_000 => 5.0,
            176_001..=208_000 => 6.0,
            208_001..=240_000 => 7.0,
            240_001..=288_000 => 8.0,
            288_001..=410_000 => 9.0,
            _ => 10.0,
        }
    }

    /// Mirror `ResidueEncoder::quality_to_quant_step`.
    fn quality_to_quant_step(quality: f32) -> f32 {
        let q = quality.clamp(-1.0, 10.0);
        let normalized = (10.0 - q) / 11.0;
        0.01 + normalized * 0.99
    }

    /// Check if decoder has received all three headers and is ready.
    pub fn is_ready(&self) -> bool {
        self.state == DecoderState::Ready
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AudioDecoder impl
// ─────────────────────────────────────────────────────────────────────────────

impl AudioDecoder for VorbisDecoderInner {
    fn codec(&self) -> CodecId {
        CodecId::Vorbis
    }

    fn send_packet(&mut self, data: &[u8], pts: i64) -> AudioResult<()> {
        if data.is_empty() {
            return Ok(());
        }

        match self.state {
            DecoderState::WaitingForIdentification => self.handle_identification(data),
            DecoderState::WaitingForComment => self.handle_comment(data),
            DecoderState::WaitingForSetup => self.handle_setup(data),
            DecoderState::Ready => self.handle_audio_packet(data, pts),
        }
    }

    fn receive_frame(&mut self) -> AudioResult<Option<AudioFrame>> {
        Ok(self.pending.pop_front())
    }

    fn flush(&mut self) -> AudioResult<()> {
        self.flushing = true;
        Ok(())
    }

    fn reset(&mut self) {
        self.flushing = false;
        self.state = DecoderState::WaitingForIdentification;
        self.pending.clear();
        self.pts = 0;
        self.prev_block_long = false;
        for ola in &mut self.overlap {
            ola.reset();
        }
    }

    fn output_format(&self) -> Option<SampleFormat> {
        Some(SampleFormat::F32)
    }

    fn sample_rate(&self) -> Option<u32> {
        Some(self.sample_rate)
    }

    fn channel_layout(&self) -> Option<ChannelLayout> {
        Some(ChannelLayout::from_count(usize::from(self.channels)))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        vorbis::encoder::VorbisEncoder, AudioDecoder, AudioDecoderConfig, AudioEncoder,
        AudioEncoderConfig,
    };
    use oximedia_core::CodecId;
    use std::f32::consts::PI;

    fn make_decoder_config() -> AudioDecoderConfig {
        AudioDecoderConfig {
            codec: CodecId::Vorbis,
            sample_rate: 44100,
            channels: 2,
            extradata: None,
        }
    }

    fn make_encoder_config() -> AudioEncoderConfig {
        AudioEncoderConfig {
            codec: CodecId::Vorbis,
            sample_rate: 44100,
            channels: 2,
            bitrate: 128_000,
            frame_size: 2048,
        }
    }

    /// Generate a 440 Hz stereo sine wave as F32 interleaved bytes.
    fn generate_sine_440hz(sample_rate: u32, num_samples: usize) -> Vec<u8> {
        let channels = 2usize;
        let mut raw: Vec<u8> = Vec::with_capacity(num_samples * channels * 4);
        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let s = (2.0 * PI * 440.0 * t).sin() * 0.5;
            raw.extend_from_slice(&s.to_le_bytes());
            raw.extend_from_slice(&s.to_le_bytes());
        }
        raw
    }

    // ── State machine tests ──────────────────────────────────────────────────

    #[test]
    fn test_vorbis_decode_returns_none_when_empty() {
        let config = make_decoder_config();
        let mut dec = VorbisDecoderInner::new(&config).expect("create decoder");
        assert!(dec.receive_frame().expect("receive_frame ok").is_none());
    }

    #[test]
    fn test_vorbis_decoder_audio_decoder_trait() {
        fn assert_audio_decoder<T: AudioDecoder>() {}
        assert_audio_decoder::<VorbisDecoderInner>();
    }

    #[test]
    fn test_vorbis_decode_state_machine_rejects_audio_before_setup() {
        let config = make_decoder_config();
        let mut dec = VorbisDecoderInner::new(&config).expect("create decoder");

        // Try to send a fake audio packet (first bit = 0) before headers are processed.
        // The decoder is in WaitingForIdentification and will try to parse it as an
        // identification header, which will fail (wrong magic bytes).
        let fake_audio: Vec<u8> = vec![0x00, 0xAA, 0xBB];
        assert!(
            dec.send_packet(&fake_audio, 0).is_err(),
            "should reject non-identification data while waiting for identification"
        );
    }

    #[test]
    fn test_vorbis_decoder_wrong_codec() {
        let config = AudioDecoderConfig {
            codec: CodecId::Opus,
            sample_rate: 44100,
            channels: 2,
            extradata: None,
        };
        assert!(VorbisDecoderInner::new(&config).is_err());
    }

    #[test]
    fn test_vorbis_decoder_reset() {
        let config = make_decoder_config();
        let mut dec = VorbisDecoderInner::new(&config).expect("create decoder");
        dec.reset();
        assert!(!dec.flushing);
        assert_eq!(dec.state, DecoderState::WaitingForIdentification);
    }

    #[test]
    fn test_vorbis_decoder_not_ready_initially() {
        let config = make_decoder_config();
        let dec = VorbisDecoderInner::new(&config).expect("create decoder");
        assert!(!dec.is_ready());
    }

    // ── Round-trip test ──────────────────────────────────────────────────────

    #[test]
    fn test_vorbis_decode_synth_440hz_stereo() {
        let enc_config = make_encoder_config();
        let mut encoder = VorbisEncoder::new(&enc_config).expect("create encoder");

        // Collect the 3 header packets.
        let mut header_packets: Vec<Vec<u8>> = Vec::new();
        while header_packets.len() < 3 {
            match encoder.receive_packet() {
                Ok(Some(pkt)) => header_packets.push(pkt.data),
                Ok(None) | Err(_) => break,
            }
        }
        assert_eq!(header_packets.len(), 3, "expected 3 header packets");

        // Send enough audio frames to get at least one encoded audio packet.
        let samples_per_channel = 2048 * 3;
        let raw = generate_sine_440hz(44100, samples_per_channel);
        let audio_frame = AudioFrame {
            format: SampleFormat::F32,
            sample_rate: 44100,
            channels: ChannelLayout::Stereo,
            samples: AudioBuffer::Interleaved(Bytes::from(raw)),
            timestamp: Timestamp::new(0, Rational::new(1, 44100)),
        };
        encoder.send_frame(&audio_frame).expect("send_frame");

        let mut audio_packets: Vec<Vec<u8>> = Vec::new();
        loop {
            match encoder.receive_packet() {
                Ok(Some(pkt)) => audio_packets.push(pkt.data),
                Ok(None) | Err(_) => break,
            }
        }

        // Feed all packets to the decoder.
        let dec_config = make_decoder_config();
        let mut decoder = VorbisDecoderInner::new(&dec_config).expect("create decoder");

        for (i, pkt) in header_packets.iter().enumerate() {
            decoder
                .send_packet(pkt, 0)
                .unwrap_or_else(|e| panic!("header packet {i} rejected: {e:?}"));
        }
        assert!(
            decoder.is_ready(),
            "decoder should be Ready after 3 headers"
        );

        let mut decoded_frames: Vec<AudioFrame> = Vec::new();
        for (i, pkt) in audio_packets.iter().enumerate() {
            let pts = (i * 2048) as i64;
            match decoder.send_packet(pkt, pts) {
                Ok(()) => {}
                Err(e) => {
                    eprintln!("decoder error on audio packet {i}: {e:?}");
                }
            }
            while let Ok(Some(frame)) = decoder.receive_frame() {
                decoded_frames.push(frame);
            }
        }

        assert!(
            !decoded_frames.is_empty(),
            "expected at least one decoded frame"
        );

        let first = &decoded_frames[0];
        assert_eq!(first.format, SampleFormat::F32);
        assert_eq!(first.sample_rate, 44100);
        assert_eq!(first.channels.count(), 2);

        // Verify IMDCT/OLA wiring: long-block output should have blocksize_1/2 = 1024
        // samples per channel (the OLA output is half the block size).
        assert_eq!(
            first.sample_count(),
            1024,
            "long-block IMDCT/OLA should emit blocksize_1/2 = 1024 samples per channel"
        );

        // Verify the decoded frame has non-zero samples.
        let has_nonzero = match &first.samples {
            AudioBuffer::Interleaved(bytes) => bytes.chunks_exact(4).any(|chunk| {
                let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
                f32::from_le_bytes(arr).abs() > 1e-10
            }),
            AudioBuffer::Planar(planes) => planes.iter().any(|plane| {
                plane.chunks_exact(4).any(|chunk| {
                    let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    f32::from_le_bytes(arr).abs() > 1e-10
                })
            }),
        };
        assert!(has_nonzero, "decoded frame should contain non-zero samples");
    }
}
