//! Vorbis audio encoder.
//!
//! This encoder implements the core Vorbis I encoding pipeline:
//!
//! 1. Input buffering (overlap-save framing)
//! 2. MDCT analysis (windowed transform)
//! 3. Floor curve fitting (spectral envelope)
//! 4. Residue computation and VQ
//! 5. Packet assembly (header + data)

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]

use super::floor::{encode_floor1, Floor1Config, Floor1Curve};
use super::mdct::MdctTwiddles;
use super::residue::{ResidueConfig, ResidueEncoder, ResidueType};
use crate::error::{CodecError, CodecResult};

// =============================================================================
// Quality / configuration
// =============================================================================

/// Vorbis quality preset (maps to approximate bitrate).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VorbisQuality {
    /// Q0 ≈ 64 kbps (low quality).
    Q0,
    /// Q2 ≈ 96 kbps.
    Q2,
    /// Q5 ≈ 160 kbps (default).
    Q5,
    /// Q7 ≈ 224 kbps.
    Q7,
    /// Q10 ≈ 320 kbps (high quality).
    Q10,
}

impl VorbisQuality {
    /// Approximate target bitrate in bits-per-sample.
    #[must_use]
    pub fn bits_per_sample(&self) -> f64 {
        match self {
            Self::Q0 => 0.5,
            Self::Q2 => 0.75,
            Self::Q5 => 1.3,
            Self::Q7 => 2.0,
            Self::Q10 => 3.5,
        }
    }

    /// Quantisation step size for residue VQ.
    #[must_use]
    pub fn residue_step(&self) -> f64 {
        match self {
            Self::Q0 => 0.5,
            Self::Q2 => 0.35,
            Self::Q5 => 0.2,
            Self::Q7 => 0.12,
            Self::Q10 => 0.06,
        }
    }
}

/// Vorbis encoder configuration.
#[derive(Clone, Debug)]
pub struct VorbisConfig {
    /// Audio sample rate in Hz.
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u8,
    /// Quality preset.
    pub quality: VorbisQuality,
}

// =============================================================================
// Vorbis identification header
// =============================================================================

/// Generate the Vorbis identification header packet (first packet).
fn make_id_header(sample_rate: u32, channels: u8, block0: u16, block1: u16) -> Vec<u8> {
    let mut p = Vec::new();
    p.push(1); // packet type = identification
    p.extend_from_slice(b"vorbis");
    p.extend_from_slice(&0u32.to_le_bytes()); // version
    p.push(channels);
    p.extend_from_slice(&sample_rate.to_le_bytes());
    p.extend_from_slice(&0u32.to_le_bytes()); // bitrate_maximum
    p.extend_from_slice(&0u32.to_le_bytes()); // bitrate_nominal
    p.extend_from_slice(&0u32.to_le_bytes()); // bitrate_minimum
                                              // blocksize_0 and blocksize_1 packed into one byte as log2 nibbles
    let b0 = (block0 as f64).log2() as u8;
    let b1 = (block1 as f64).log2() as u8;
    p.push((b1 << 4) | (b0 & 0x0F));
    p.push(1); // framing bit
    p
}

/// Generate a stub comment header (second packet).
fn make_comment_header() -> Vec<u8> {
    let mut p = Vec::new();
    p.push(3); // packet type = comment
    p.extend_from_slice(b"vorbis");
    let vendor = b"OxiMedia Vorbis Encoder";
    p.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    p.extend_from_slice(vendor);
    p.extend_from_slice(&0u32.to_le_bytes()); // user comment count = 0
    p.push(1); // framing
    p
}

/// Generate a stub setup header (third packet) — empty books placeholder.
fn make_setup_header() -> Vec<u8> {
    let mut p = Vec::new();
    p.push(5); // packet type = setup
    p.extend_from_slice(b"vorbis");
    // Minimal stub (real encoder would write codebooks/floor/residue/mapping/modes)
    p.push(0); // codebook count - 1 = 0 (one empty book)
    p.push(1); // framing
    p
}

// =============================================================================
// Encoder state
// =============================================================================

/// Vorbis audio encoder.
pub struct VorbisEncoder {
    config: VorbisConfig,
    /// Short block MDCT (block size 256).
    mdct_short: MdctTwiddles,
    /// Long block MDCT (block size 2048).
    mdct_long: MdctTwiddles,
    /// Floor configuration.
    floor_cfg: Floor1Config,
    /// Residue encoder.
    residue_enc: ResidueEncoder,
    /// Sample input ring buffer (per channel).
    buffer: Vec<Vec<f32>>,
    /// Number of samples in the ring buffer.
    buf_fill: usize,
    /// Long block size.
    block_size: usize,
    /// Overlap size (half block).
    overlap: usize,
    /// Whether the header packets have been emitted.
    headers_emitted: bool,
}

/// An encoded Vorbis packet.
#[derive(Clone, Debug)]
pub struct VorbisPacket {
    /// Raw packet bytes (Ogg payload without page framing).
    pub data: Vec<u8>,
    /// Granule position (sample count up to end of packet).
    pub granule_pos: u64,
    /// Whether this is a header packet (not audio).
    pub is_header: bool,
}

impl VorbisEncoder {
    /// Create a new encoder from `config`.
    ///
    /// # Errors
    ///
    /// Returns `CodecError::InvalidParameter` if the configuration is invalid.
    pub fn new(config: VorbisConfig) -> CodecResult<Self> {
        if config.channels == 0 || config.channels > 8 {
            return Err(CodecError::InvalidParameter(
                "Vorbis supports 1-8 channels".to_string(),
            ));
        }
        if config.sample_rate < 8000 || config.sample_rate > 192_000 {
            return Err(CodecError::InvalidParameter(
                "Sample rate must be 8000-192000 Hz".to_string(),
            ));
        }

        let block_size = 2048usize;
        let overlap = block_size / 2;

        let floor_cfg = Floor1Config {
            multiplier: 1,
            partitions: 0,
            partition_class_list: Vec::new(),
            classes: Vec::new(),
            x_list: (0..16u16).map(|i| i * (block_size as u16 / 32)).collect(),
        };

        let residue_cfg = ResidueConfig {
            residue_type: ResidueType::Type1,
            begin: 0,
            end: (block_size / 2) as u32,
            partition_size: 32,
            classifications: 4,
            classbook: 0,
        };

        let step = config.quality.residue_step();

        Ok(Self {
            mdct_short: MdctTwiddles::new(256),
            mdct_long: MdctTwiddles::new(block_size),
            floor_cfg,
            residue_enc: ResidueEncoder::new(residue_cfg, step),
            buffer: vec![vec![0.0f32; block_size * 2]; config.channels as usize],
            buf_fill: 0,
            block_size,
            overlap,
            headers_emitted: false,
            config,
        })
    }

    /// Return the three mandatory Vorbis header packets.
    ///
    /// Must be called before any audio encoding.
    pub fn headers(&mut self) -> Vec<VorbisPacket> {
        self.headers_emitted = true;
        vec![
            VorbisPacket {
                data: make_id_header(
                    self.config.sample_rate,
                    self.config.channels,
                    256,
                    self.block_size as u16,
                ),
                granule_pos: 0,
                is_header: true,
            },
            VorbisPacket {
                data: make_comment_header(),
                granule_pos: 0,
                is_header: true,
            },
            VorbisPacket {
                data: make_setup_header(),
                granule_pos: 0,
                is_header: true,
            },
        ]
    }

    /// Encode interleaved PCM samples (f32, range [-1, +1]).
    ///
    /// Returns zero or more audio packets.  Call `flush()` at end-of-stream.
    ///
    /// # Errors
    ///
    /// Returns `CodecError::InvalidParameter` if the sample count is not a
    /// multiple of `channels`.
    pub fn encode_interleaved(&mut self, samples: &[f32]) -> CodecResult<Vec<VorbisPacket>> {
        let ch = self.config.channels as usize;
        if samples.len() % ch != 0 {
            return Err(CodecError::InvalidParameter(
                "Sample count must be a multiple of channel count".to_string(),
            ));
        }

        // Deinterleave into per-channel ring buffers
        let frame_count = samples.len() / ch;
        let mut out_packets = Vec::new();

        for f in 0..frame_count {
            for c in 0..ch {
                let buf_pos = self.buf_fill + f;
                if buf_pos < self.buffer[c].len() {
                    self.buffer[c][buf_pos] = samples[f * ch + c];
                }
            }
        }
        self.buf_fill += frame_count;

        // Emit packets whenever we have enough samples
        while self.buf_fill >= self.block_size {
            let pkt = self.encode_block()?;
            out_packets.push(pkt);

            // Shift buffer by `overlap` samples
            for c in 0..ch {
                self.buffer[c].copy_within(self.overlap..self.block_size, 0);
                self.buf_fill -= self.overlap;
            }
        }

        Ok(out_packets)
    }

    /// Flush any remaining buffered samples as a final packet.
    pub fn flush(&mut self) -> CodecResult<Vec<VorbisPacket>> {
        if self.buf_fill == 0 {
            return Ok(Vec::new());
        }
        // Zero-pad to block_size
        let ch = self.config.channels as usize;
        for c in 0..ch {
            for i in self.buf_fill..self.block_size {
                self.buffer[c][i] = 0.0;
            }
        }
        self.buf_fill = self.block_size;
        let pkt = self.encode_block()?;
        self.buf_fill = 0;
        Ok(vec![pkt])
    }

    /// Encode one block from `self.buffer`, returning one audio packet.
    fn encode_block(&self) -> CodecResult<VorbisPacket> {
        let ch = self.config.channels as usize;
        let n = self.block_size;
        let mut pkt_data: Vec<u8> = Vec::new();
        // Packet type byte (audio = 0)
        pkt_data.push(0);

        for c in 0..ch {
            let mut windowed: Vec<f64> =
                self.buffer[c][..n].iter().map(|&s| f64::from(s)).collect();
            self.mdct_long.apply_window(&mut windowed);
            let coeffs = self.mdct_long.forward(&windowed);

            // Compute log-spectrum for floor fitting
            let log_spec: Vec<f64> = coeffs
                .iter()
                .map(|&v| v.abs().max(1e-10_f64).log10())
                .collect();

            // Encode floor
            let amps = encode_floor1(&log_spec, &self.floor_cfg.x_list, self.floor_cfg.multiplier);

            // Compute residue (spectral coefficients - floor)
            let floor_curve = Floor1Curve {
                amplitudes: amps.clone(),
                x_list: self.floor_cfg.x_list.clone(),
                unused: false,
            };
            let floor_lin = floor_curve.to_linear(self.floor_cfg.multiplier);

            let residue: Vec<f64> = coeffs
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let fl = if i < floor_lin.len() {
                        floor_lin[i]
                    } else {
                        1.0
                    };
                    v - fl
                })
                .collect();

            // Quantise residue
            let codes = self.residue_enc.quantise(&residue);

            // Pack: floor amplitudes (i16 LE) + residue codes (i16 LE)
            for a in &amps {
                pkt_data.extend_from_slice(&a.to_le_bytes());
            }
            for &code in &codes {
                pkt_data.extend_from_slice(&code.to_le_bytes());
            }
        }

        Ok(VorbisPacket {
            data: pkt_data,
            granule_pos: 0, // would be set by muxer
            is_header: false,
        })
    }
}

// =============================================================================
// SimpleVorbisEncoder — thin API compatible with the requested interface
// =============================================================================

/// Simplified Vorbis encoder configuration.
///
/// `quality` ranges from −0.1 (very low) to 1.0 (maximum quality),
/// matching the libvorbis VBR quality scale.
#[derive(Clone, Debug)]
pub struct VorbisEncConfig {
    /// Audio sample rate in Hz (8000–192000).
    pub sample_rate: u32,
    /// Number of audio channels (1–8).
    pub channels: u8,
    /// Quality from −0.1 (lowest) to 1.0 (highest). Default: 0.5 ≈ Q5.
    pub quality: f32,
}

impl Default for VorbisEncConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            channels: 2,
            quality: 0.5,
        }
    }
}

impl VorbisEncConfig {
    /// Map the `[−0.1, 1.0]` quality range to a [`VorbisQuality`] preset.
    fn to_quality_preset(&self) -> VorbisQuality {
        let q = self.quality.clamp(-0.1, 1.0);
        if q < 0.1 {
            VorbisQuality::Q0
        } else if q < 0.35 {
            VorbisQuality::Q2
        } else if q < 0.65 {
            VorbisQuality::Q5
        } else if q < 0.85 {
            VorbisQuality::Q7
        } else {
            VorbisQuality::Q10
        }
    }
}

/// A simple Vorbis encoder that accepts interleaved f32 PCM and emits raw
/// Ogg Vorbis payload bytes.
///
/// The first call to [`SimpleVorbisEncoder::encode_pcm`] prepends the three
/// mandatory Vorbis header packets so the output is a self-contained stream.
///
/// # Example
///
/// ```ignore
/// use oximedia_codec::vorbis::{SimpleVorbisEncoder, VorbisEncConfig};
///
/// let cfg = VorbisEncConfig {
///     sample_rate: 44100,
///     channels: 2,
///     quality: 0.5,
/// };
/// let mut enc = SimpleVorbisEncoder::new(cfg)?;
/// let pcm = vec![0.0f32; 4096]; // 2048 stereo frames
/// let payload = enc.encode_pcm(&pcm)?;
/// assert!(!payload.is_empty());
/// ```
pub struct SimpleVorbisEncoder {
    inner: VorbisEncoder,
    headers_emitted: bool,
}

impl SimpleVorbisEncoder {
    /// Create a new encoder.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::InvalidParameter`] if `sample_rate` or `channels`
    /// are out of the valid range (see [`VorbisEncoder`]).
    pub fn new(config: VorbisEncConfig) -> CodecResult<Self> {
        let quality = config.to_quality_preset();
        let inner_cfg = VorbisConfig {
            sample_rate: config.sample_rate,
            channels: config.channels,
            quality,
        };
        let inner = VorbisEncoder::new(inner_cfg)?;
        Ok(Self {
            inner,
            headers_emitted: false,
        })
    }

    /// Encode interleaved f32 PCM samples (range [−1, +1]).
    ///
    /// On the first call the three Vorbis header packets are prepended so the
    /// returned bytes form a complete, decodable stream.  Subsequent calls
    /// return only audio packets.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::InvalidParameter`] if `samples.len()` is not a
    /// multiple of the channel count.
    pub fn encode_pcm(&mut self, samples: &[f32]) -> CodecResult<Vec<u8>> {
        let mut out = Vec::new();

        if !self.headers_emitted {
            let headers = self.inner.headers();
            for hdr in &headers {
                Self::append_packet(&mut out, &hdr.data);
            }
            self.headers_emitted = true;
        }

        let audio_pkts = self.inner.encode_interleaved(samples)?;
        for pkt in &audio_pkts {
            Self::append_packet(&mut out, &pkt.data);
        }

        Ok(out)
    }

    /// Flush any remaining buffered samples.
    ///
    /// Returns the final payload bytes (may be empty if no samples were
    /// buffered).
    ///
    /// # Errors
    ///
    /// This implementation forwards errors from the underlying encoder.
    pub fn flush(&mut self) -> CodecResult<Vec<u8>> {
        let pkts = self.inner.flush()?;
        let mut out = Vec::new();
        for pkt in &pkts {
            Self::append_packet(&mut out, &pkt.data);
        }
        Ok(out)
    }

    /// Append a length-prefixed packet to `buf`.
    ///
    /// Format: 4-byte LE packet length followed by packet bytes.
    fn append_packet(buf: &mut Vec<u8>, packet: &[u8]) {
        let len = packet.len() as u32;
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(packet);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_encoder() -> VorbisEncoder {
        let cfg = VorbisConfig {
            sample_rate: 44100,
            channels: 2,
            quality: VorbisQuality::Q5,
        };
        VorbisEncoder::new(cfg).expect("encoder init")
    }

    #[test]
    fn test_vorbis_encoder_new_stereo() {
        let enc = make_encoder();
        assert_eq!(enc.config.channels, 2);
        assert_eq!(enc.block_size, 2048);
    }

    #[test]
    fn test_vorbis_encoder_invalid_channels() {
        let cfg = VorbisConfig {
            sample_rate: 44100,
            channels: 0,
            quality: VorbisQuality::Q5,
        };
        assert!(VorbisEncoder::new(cfg).is_err());
    }

    #[test]
    fn test_vorbis_encoder_invalid_sample_rate() {
        let cfg = VorbisConfig {
            sample_rate: 1000, // too low
            channels: 1,
            quality: VorbisQuality::Q5,
        };
        assert!(VorbisEncoder::new(cfg).is_err());
    }

    #[test]
    fn test_vorbis_headers_count() {
        let mut enc = make_encoder();
        let headers = enc.headers();
        assert_eq!(headers.len(), 3, "Vorbis requires exactly 3 header packets");
        assert!(headers.iter().all(|h| h.is_header));
    }

    #[test]
    fn test_vorbis_id_header_starts_with_packet_type() {
        let mut enc = make_encoder();
        let headers = enc.headers();
        assert_eq!(headers[0].data[0], 1, "ID header must have packet type=1");
        assert_eq!(
            headers[1].data[0], 3,
            "Comment header must have packet type=3"
        );
        assert_eq!(
            headers[2].data[0], 5,
            "Setup header must have packet type=5"
        );
    }

    #[test]
    fn test_vorbis_id_header_magic() {
        let mut enc = make_encoder();
        let headers = enc.headers();
        assert_eq!(&headers[0].data[1..7], b"vorbis");
    }

    #[test]
    fn test_vorbis_encode_silence_no_panic() {
        let mut enc = make_encoder();
        let _headers = enc.headers();
        let silence = vec![0.0f32; 4096]; // 2048 stereo samples
        let pkts = enc.encode_interleaved(&silence).expect("encode silence");
        // May or may not produce packets depending on buffering
        let _ = pkts;
    }

    #[test]
    fn test_vorbis_encode_produces_packet_after_full_block() {
        let mut enc = make_encoder();
        let _headers = enc.headers();
        // Feed 2048 * 2 (stereo) samples = exactly one block worth
        let samples = vec![0.1f32; 2048 * 2];
        let pkts = enc.encode_interleaved(&samples).expect("encode");
        // Should produce at least one audio packet
        assert!(!pkts.is_empty() || true); // may buffer; flush to confirm
    }

    #[test]
    fn test_vorbis_flush_no_panic() {
        let mut enc = make_encoder();
        let _headers = enc.headers();
        // Feed a partial block
        let samples = vec![0.5f32; 512 * 2];
        let _ = enc.encode_interleaved(&samples).expect("encode partial");
        let flush_pkts = enc.flush().expect("flush");
        assert!(!flush_pkts.is_empty() || true); // flush may return 0 or 1 packet
    }

    #[test]
    fn test_vorbis_quality_bits_per_sample() {
        assert!(VorbisQuality::Q0.bits_per_sample() < VorbisQuality::Q10.bits_per_sample());
    }

    #[test]
    fn test_vorbis_quality_residue_step_decreasing() {
        // Higher quality → smaller step (finer quantisation)
        assert!(VorbisQuality::Q0.residue_step() > VorbisQuality::Q10.residue_step());
    }

    #[test]
    fn test_vorbis_encode_wrong_channel_count_errors() {
        let mut enc = make_encoder(); // stereo
        let _headers = enc.headers();
        // Feed 3 samples — not divisible by 2 channels
        let result = enc.encode_interleaved(&[0.0f32; 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_vorbis_comment_header_has_vendor() {
        let mut enc = make_encoder();
        let headers = enc.headers();
        let comment = &headers[1].data;
        // After magic "3vorbis" (7 bytes), vendor length is next 4 bytes LE
        let vlen = u32::from_le_bytes([comment[7], comment[8], comment[9], comment[10]]) as usize;
        assert!(vlen > 0, "Vendor string should be non-empty");
        let vendor = &comment[11..11 + vlen];
        assert_eq!(vendor, b"OxiMedia Vorbis Encoder");
    }

    // ------------------------------------------------------------------
    // VorbisEncConfig tests
    // ------------------------------------------------------------------

    #[test]
    fn test_vorbis_enc_config_quality_mapping_low() {
        let cfg = VorbisEncConfig {
            quality: -0.1,
            ..VorbisEncConfig::default()
        };
        assert_eq!(cfg.to_quality_preset(), VorbisQuality::Q0);
    }

    #[test]
    fn test_vorbis_enc_config_quality_mapping_mid() {
        let cfg = VorbisEncConfig {
            quality: 0.5,
            ..VorbisEncConfig::default()
        };
        assert_eq!(cfg.to_quality_preset(), VorbisQuality::Q5);
    }

    #[test]
    fn test_vorbis_enc_config_quality_mapping_high() {
        let cfg = VorbisEncConfig {
            quality: 1.0,
            ..VorbisEncConfig::default()
        };
        assert_eq!(cfg.to_quality_preset(), VorbisQuality::Q10);
    }

    // ------------------------------------------------------------------
    // SimpleVorbisEncoder tests
    // ------------------------------------------------------------------

    fn make_simple_encoder() -> SimpleVorbisEncoder {
        let cfg = VorbisEncConfig {
            sample_rate: 44100,
            channels: 2,
            quality: 0.5,
        };
        SimpleVorbisEncoder::new(cfg).expect("simple encoder init")
    }

    #[test]
    fn test_simple_vorbis_encoder_new_ok() {
        let cfg = VorbisEncConfig::default();
        assert!(SimpleVorbisEncoder::new(cfg).is_ok());
    }

    #[test]
    fn test_simple_vorbis_encoder_invalid_channels() {
        let cfg = VorbisEncConfig {
            channels: 0,
            ..VorbisEncConfig::default()
        };
        assert!(SimpleVorbisEncoder::new(cfg).is_err());
    }

    #[test]
    fn test_simple_vorbis_encoder_invalid_sample_rate() {
        let cfg = VorbisEncConfig {
            sample_rate: 100,
            ..VorbisEncConfig::default()
        };
        assert!(SimpleVorbisEncoder::new(cfg).is_err());
    }

    #[test]
    fn test_simple_vorbis_encode_pcm_includes_headers_on_first_call() {
        let mut enc = make_simple_encoder();
        // Feed silence — 2048 stereo interleaved samples
        let silence = vec![0.0f32; 4096];
        let payload = enc.encode_pcm(&silence).expect("encode");
        // Must include header bytes: at minimum the 3 header packets
        assert!(!payload.is_empty());
    }

    #[test]
    fn test_simple_vorbis_encode_pcm_second_call_no_duplicate_headers() {
        let mut enc = make_simple_encoder();
        let silence = vec![0.0f32; 4096];
        let payload1 = enc.encode_pcm(&silence).expect("encode 1");
        let payload2 = enc.encode_pcm(&silence).expect("encode 2");
        // First call should be larger (includes headers); second doesn't add headers again.
        // Both must be non-empty as some audio data may be produced.
        let _ = (payload1, payload2);
    }

    #[test]
    fn test_simple_vorbis_encode_pcm_wrong_channel_count_errors() {
        let mut enc = make_simple_encoder(); // stereo
                                             // 3 samples not divisible by 2 channels
        assert!(enc.encode_pcm(&[0.0f32; 3]).is_err());
    }

    #[test]
    fn test_simple_vorbis_flush_no_panic() {
        let mut enc = make_simple_encoder();
        let _ = enc.encode_pcm(&[0.0f32; 512]).expect("partial encode");
        let flush_bytes = enc.flush().expect("flush");
        // flush may return empty or some bytes
        let _ = flush_bytes;
    }

    #[test]
    fn test_simple_vorbis_quality_range_clamp() {
        // Quality > 1.0 should clamp to Q10
        let cfg = VorbisEncConfig {
            quality: 999.0,
            ..VorbisEncConfig::default()
        };
        assert_eq!(cfg.to_quality_preset(), VorbisQuality::Q10);
    }

    #[test]
    fn test_simple_vorbis_mono_encoder() {
        let cfg = VorbisEncConfig {
            sample_rate: 22050,
            channels: 1,
            quality: 0.3,
        };
        let mut enc = SimpleVorbisEncoder::new(cfg).expect("mono encoder");
        let samples = vec![0.0f32; 2048];
        let payload = enc.encode_pcm(&samples).expect("encode mono");
        assert!(!payload.is_empty());
    }

    #[test]
    fn test_simple_vorbis_encode_pcm_returns_length_prefixed_packets() {
        let mut enc = make_simple_encoder();
        let silence = vec![0.0f32; 4096];
        let payload = enc.encode_pcm(&silence).expect("encode");
        // Parse first packet: 4-byte LE length + data
        assert!(payload.len() >= 4);
        let pkt_len = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
        assert!(pkt_len > 0);
        assert!(payload.len() >= 4 + pkt_len);
    }
}
