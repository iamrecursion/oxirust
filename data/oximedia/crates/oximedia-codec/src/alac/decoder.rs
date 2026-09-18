//! ALAC frame decoder.
//!
//! Decodes a single raw ALAC frame (one "audio frame" / packet) into
//! interleaved signed PCM, using the [`AlacSpecificConfig`] magic cookie for
//! geometry and entropy tuning. The frame is a sequence of *syntax elements*
//! (single-channel `SCE` or channel-pair `CPE`) terminated by an `END`
//! element, mirroring the structure of Apple's `ALACDecoder::Decode`.
//!
//! # Element layout
//!
//! Each element begins with a 3-bit tag, then a fixed header:
//!
//! ```text
//! tag            : 3  (0 = SCE, 1 = CPE, 7 = END)
//! reserved       : 12 (zero)
//! partial_frame  : 1
//! bytes_shifted  : 2  (shift = bytes_shifted * 8)
//! escape         : 1  (1 ⇒ samples stored uncompressed)
//! [num_samples]  : 32 (only when partial_frame == 1)
//! ```
//!
//! For the compressed path a `CPE` first stores `mix_bits` (8) and `mix_res`
//! (8, signed). The shifted-off low bits (if any) are stored verbatim, then for
//! each coded channel a predictor sub-header (mode, denshift, pb factor, order,
//! signed 16-bit coefficients) followed by its adaptive-Golomb residual stream.

use super::bitstream::BitReader;
use super::config::AlacSpecificConfig;
use super::lpc::predict_decode;
use super::mix::unmix_stereo;
use super::rice::{decode_residuals, AgState};
use super::{AlacError, AlacResult};

/// Element tag: single-channel element.
pub const TAG_SCE: u32 = 0;
/// Element tag: channel-pair element.
pub const TAG_CPE: u32 = 1;
/// Element tag: end of frame.
pub const TAG_END: u32 = 7;

/// ALAC frame decoder.
pub struct AlacDecoder {
    config: AlacSpecificConfig,
}

impl AlacDecoder {
    /// Create a decoder from a serialized `ALACSpecificConfig` magic cookie.
    pub fn new(magic_cookie: &[u8]) -> AlacResult<Self> {
        let config = AlacSpecificConfig::parse(magic_cookie)?;
        Ok(Self { config })
    }

    /// Create a decoder directly from a parsed config.
    #[must_use]
    pub fn from_config(config: AlacSpecificConfig) -> Self {
        Self { config }
    }

    /// The configuration this decoder was built with.
    #[must_use]
    pub fn config(&self) -> &AlacSpecificConfig {
        &self.config
    }

    /// Decode one ALAC frame into interleaved signed PCM (`i32` per sample).
    ///
    /// The returned vector has `num_samples * num_channels` entries in
    /// channel-interleaved order.
    pub fn decode_packet(&mut self, data: &[u8]) -> AlacResult<Vec<i32>> {
        let mut reader = BitReader::new(data);
        let num_channels = self.config.num_channels as usize;
        // Per-element decode appends each element's channels into `channels`.
        let mut channels: Vec<Vec<i32>> = Vec::with_capacity(num_channels);
        let mut frame_len: Option<usize> = None;

        loop {
            let tag = reader.read_bits(3)?;
            if tag == TAG_END {
                break;
            }
            let pair = match tag {
                TAG_SCE => false,
                TAG_CPE => true,
                other => {
                    return Err(AlacError::InvalidBitstream(format!(
                        "unknown element tag {other}"
                    )));
                }
            };
            let element = self.decode_element(&mut reader, pair)?;
            match frame_len {
                Some(len) if len != element.num_samples => {
                    return Err(AlacError::InvalidBitstream(
                        "inconsistent element sample counts".into(),
                    ));
                }
                _ => frame_len = Some(element.num_samples),
            }
            for ch in element.channels {
                channels.push(ch);
            }
            if channels.len() >= num_channels {
                break;
            }
        }

        if channels.len() != num_channels {
            return Err(AlacError::InvalidBitstream(format!(
                "decoded {} channels, expected {}",
                channels.len(),
                num_channels
            )));
        }
        let num_samples = frame_len.unwrap_or(0);
        interleave(&channels, num_samples)
    }

    /// Decode one frame into planar channels (one `Vec<i32>` per channel).
    pub fn decode_packet_planar(&mut self, data: &[u8]) -> AlacResult<Vec<Vec<i32>>> {
        let interleaved = self.decode_packet(data)?;
        let num_channels = self.config.num_channels as usize;
        if num_channels == 0 {
            return Ok(Vec::new());
        }
        let num_samples = interleaved.len() / num_channels;
        let mut planar = vec![Vec::with_capacity(num_samples); num_channels];
        for frame in interleaved.chunks_exact(num_channels) {
            for (ch, &s) in frame.iter().enumerate() {
                planar[ch].push(s);
            }
        }
        Ok(planar)
    }

    fn decode_element(&self, reader: &mut BitReader, pair: bool) -> AlacResult<DecodedElement> {
        let _reserved = reader.read_bits(12)?;
        let partial_frame = reader.read_bit()?;
        let bytes_shifted = reader.read_bits(2)?;
        let escape = reader.read_bit()?;
        let shift = bytes_shifted * 8;
        if shift >= 32 {
            return Err(AlacError::InvalidBitstream(
                "bytes_shifted too large".into(),
            ));
        }

        let num_samples = if partial_frame {
            reader.read_bits(32)? as usize
        } else {
            self.config.frame_length as usize
        };
        if num_samples == 0 {
            return Err(AlacError::InvalidBitstream("zero-length element".into()));
        }
        if num_samples > MAX_FRAME_SAMPLES {
            return Err(AlacError::InvalidBitstream(format!(
                "element claims {num_samples} samples (>{MAX_FRAME_SAMPLES})"
            )));
        }

        let bit_depth = u32::from(self.config.bit_depth);
        let channel_count = if pair { 2usize } else { 1usize };

        if escape {
            // Uncompressed: raw `bit_depth`-bit samples, per channel interleaved.
            let mut channels = vec![vec![0i32; num_samples]; channel_count];
            for s in 0..num_samples {
                for ch in channels.iter_mut() {
                    ch[s] = reader.read_signed(bit_depth)?;
                }
            }
            return Ok(DecodedElement {
                num_samples,
                channels,
            });
        }

        // Compressed path.
        let (mix_bits, mix_res) = if pair {
            let mb = reader.read_bits(8)?;
            let mr = reader.read_signed(8)?;
            (mb, mr)
        } else {
            (0u32, 0i32)
        };

        // Coded channels may need one extra bit for the stereo "side"/"mid".
        let extra = if pair { 1u32 } else { 0u32 };
        let chan_bits = bit_depth.saturating_sub(shift) + extra;
        if chan_bits == 0 || chan_bits > 32 {
            return Err(AlacError::InvalidBitstream(format!(
                "computed chan_bits {chan_bits} out of range"
            )));
        }

        // Read predictor sub-headers for each coded channel.
        let mut sub_headers = Vec::with_capacity(channel_count);
        for _ in 0..channel_count {
            sub_headers.push(read_sub_header(reader)?);
        }

        // Shifted-off low bits (stored verbatim, per channel interleaved).
        let mut shifted: Vec<Vec<u32>> = Vec::new();
        if shift > 0 {
            shifted = vec![vec![0u32; num_samples]; channel_count];
            for s in 0..num_samples {
                for ch in 0..channel_count {
                    shifted[ch][s] = reader.read_bits(shift)?;
                }
            }
        }

        // Decode each coded channel's residuals → predictor synthesis.
        let mut coded: Vec<Vec<i32>> = Vec::with_capacity(channel_count);
        for header in &sub_headers {
            let mut state = AgState::new(
                scaled_pb(self.config.pb, header.pb_factor),
                self.config.mb,
                self.config.kb,
                chan_bits,
            );
            let residuals = decode_residuals(reader, num_samples, &mut state)?;
            let samples = if header.mode == 0 {
                let mut coefs = header.coefs.clone();
                predict_decode(&residuals, &mut coefs, chan_bits, header.denshift)?
            } else {
                return Err(AlacError::Unsupported(format!(
                    "predictor mode {} (extended) not implemented",
                    header.mode
                )));
            };
            coded.push(samples);
        }

        // Inter-channel recombination.
        let mut channels: Vec<Vec<i32>> = if pair {
            let mut interleaved = vec![0i32; num_samples * 2];
            unmix_stereo(
                &coded[0],
                &coded[1],
                num_samples,
                mix_bits,
                mix_res,
                &mut interleaved,
            );
            let mut left = vec![0i32; num_samples];
            let mut right = vec![0i32; num_samples];
            for j in 0..num_samples {
                left[j] = interleaved[2 * j];
                right[j] = interleaved[2 * j + 1];
            }
            vec![left, right]
        } else {
            vec![coded.into_iter().next().unwrap_or_default()]
        };

        // Re-insert shifted-off low bits.
        if shift > 0 {
            for ch in 0..channel_count {
                for s in 0..num_samples {
                    let high = channels[ch][s];
                    let low = shifted[ch][s];
                    channels[ch][s] = ((high << shift) as u32 | low) as i32;
                }
            }
        }

        Ok(DecodedElement {
            num_samples,
            channels,
        })
    }
}

/// An upper bound on the per-element sample count, guarding against corrupt
/// `partial_frame` headers that would otherwise request huge allocations.
const MAX_FRAME_SAMPLES: usize = 1 << 24;

/// A decoded syntax element's channels and length.
struct DecodedElement {
    num_samples: usize,
    channels: Vec<Vec<i32>>,
}

/// Per-channel predictor sub-header.
pub struct SubHeader {
    /// Prediction mode (0 = standard adaptive FIR).
    pub mode: u32,
    /// Fixed-point denominator shift for the FIR sum.
    pub denshift: u32,
    /// Per-channel `pb` scaling factor (Apple's `pbFactor`, 0..7).
    pub pb_factor: u32,
    /// Predictor coefficients (length = order).
    pub coefs: Vec<i32>,
}

fn read_sub_header(reader: &mut BitReader) -> AlacResult<SubHeader> {
    let mode = reader.read_bits(4)?;
    let denshift = reader.read_bits(4)?;
    let pb_factor = reader.read_bits(3)?;
    let order = reader.read_bits(5)? as usize;
    if order > super::lpc::MAX_COEFS {
        return Err(AlacError::InvalidBitstream(format!(
            "predictor order {order} exceeds {}",
            super::lpc::MAX_COEFS
        )));
    }
    let mut coefs = Vec::with_capacity(order);
    for _ in 0..order {
        coefs.push(reader.read_signed(16)?);
    }
    Ok(SubHeader {
        mode,
        denshift,
        pb_factor,
        coefs,
    })
}

/// Apply Apple's per-channel `pb` scaling: `pb * pb_factor / 4`, with a factor
/// of 0 meaning "use the cookie value unchanged".
#[inline]
pub fn scaled_pb(pb: u8, pb_factor: u32) -> u8 {
    if pb_factor == 0 {
        pb
    } else {
        ((u32::from(pb) * pb_factor) / 4).min(255) as u8
    }
}

/// Interleave planar channels into a single `Vec<i32>`.
fn interleave(channels: &[Vec<i32>], num_samples: usize) -> AlacResult<Vec<i32>> {
    let num_channels = channels.len();
    for ch in channels {
        if ch.len() != num_samples {
            return Err(AlacError::InvalidBitstream(
                "channel length mismatch during interleave".into(),
            ));
        }
    }
    let mut out = vec![0i32; num_samples * num_channels];
    for (c, ch) in channels.iter().enumerate() {
        for (s, &v) in ch.iter().enumerate() {
            out[s * num_channels + c] = v;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaled_pb() {
        assert_eq!(scaled_pb(40, 0), 40);
        assert_eq!(scaled_pb(40, 4), 40);
        assert_eq!(scaled_pb(40, 2), 20);
    }

    #[test]
    fn test_truncated_frame_errs() {
        let cfg = AlacSpecificConfig::new(4096, 44_100, 1, 16);
        let mut dec = AlacDecoder::from_config(cfg);
        // A 1-byte frame cannot contain a valid element.
        let res = dec.decode_packet(&[0x00]);
        assert!(res.is_err());
    }

    #[test]
    fn test_empty_frame_errs() {
        let cfg = AlacSpecificConfig::new(4096, 44_100, 1, 16);
        let mut dec = AlacDecoder::from_config(cfg);
        assert!(dec.decode_packet(&[]).is_err());
    }
}
