//! ALAC frame encoder.
//!
//! Encodes one block of interleaved signed PCM into a raw ALAC frame, mirroring
//! the decoder forward: inter-channel decorrelation → adaptive FIR prediction →
//! adaptive-Golomb residual coding. The bitstream layout produced here is the
//! exact one [`super::decoder::AlacDecoder`] expects, so encode → decode is
//! byte-exact for 16/20/24-bit input (32-bit best-effort).
//!
//! Per element the encoder also assembles the uncompressed *escape* form and
//! keeps whichever representation is smaller, so high-entropy input never
//! expands.

use super::bitstream::BitWriter;
use super::config::AlacSpecificConfig;
use super::decoder::{TAG_CPE, TAG_END, TAG_SCE};
use super::lpc::predict_encode;
use super::mix::mix_stereo;
use super::rice::{encode_residuals, AgState};
use super::{AlacError, AlacResult};

/// Default predictor order used by the encoder.
const DEFAULT_ORDER: usize = 8;
/// Default fixed-point denominator shift for the FIR predictor.
const DEFAULT_DENSHIFT: u32 = 4;
/// Candidate stereo mix weights tried by the encoder (`mix_res`).
const MIX_RES_CANDIDATES: [i32; 5] = [0, 1, 2, 3, 4];
/// Fixed `mix_bits` shift used for stereo decorrelation.
const MIX_BITS: u32 = 2;

/// Configuration for the ALAC encoder.
#[derive(Clone, Copy, Debug)]
pub struct AlacEncoderConfig {
    /// Samples per channel per frame.
    pub frame_length: u32,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u8,
    /// Source PCM bit depth (16, 20, 24, or 32).
    pub bit_depth: u8,
}

impl Default for AlacEncoderConfig {
    fn default() -> Self {
        Self {
            frame_length: 4096,
            sample_rate: 44_100,
            channels: 2,
            bit_depth: 16,
        }
    }
}

/// ALAC frame encoder.
pub struct AlacEncoder {
    config: AlacSpecificConfig,
    /// Number of bytes shifted off as uncompressed low bits (0 by default).
    bytes_shifted: u32,
}

impl AlacEncoder {
    /// Create an encoder from an [`AlacEncoderConfig`].
    pub fn new(config: AlacEncoderConfig) -> AlacResult<Self> {
        let spec = AlacSpecificConfig::new(
            config.frame_length,
            config.sample_rate,
            config.channels,
            config.bit_depth,
        );
        spec.validate()?;
        Ok(Self {
            config: spec,
            bytes_shifted: 0,
        })
    }

    /// Create an encoder from a pre-built [`AlacSpecificConfig`].
    pub fn from_config(config: AlacSpecificConfig) -> AlacResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            bytes_shifted: 0,
        })
    }

    /// Set the number of low bytes to split off uncompressed (0..=2).
    ///
    /// `0` (the default) compresses the full bit depth. Larger values move the
    /// low `8 * bytes_shifted` bits of every sample into a verbatim region,
    /// matching Apple's `mBytesShifted` path; the round-trip stays lossless.
    pub fn set_bytes_shifted(&mut self, bytes_shifted: u32) -> AlacResult<()> {
        let depth = u32::from(self.config.bit_depth);
        if bytes_shifted > 2 || bytes_shifted * 8 >= depth {
            return Err(AlacError::InvalidConfig(format!(
                "bytes_shifted {bytes_shifted} invalid for bit_depth {depth}"
            )));
        }
        self.bytes_shifted = bytes_shifted;
        Ok(())
    }

    /// The serialized magic cookie describing this encoder's output.
    #[must_use]
    pub fn magic_cookie(&self) -> Vec<u8> {
        self.config.serialize()
    }

    /// The parsed configuration backing this encoder.
    #[must_use]
    pub fn config(&self) -> &AlacSpecificConfig {
        &self.config
    }

    /// Encode one interleaved PCM block into a raw ALAC frame.
    ///
    /// `pcm` must contain `num_samples * channels` interleaved samples, where
    /// `num_samples <= frame_length`.
    pub fn encode_frame(&mut self, pcm: &[i32]) -> AlacResult<Vec<u8>> {
        let num_channels = self.config.num_channels as usize;
        if num_channels == 0 {
            return Err(AlacError::InvalidConfig("num_channels is zero".into()));
        }
        if pcm.len() % num_channels != 0 {
            return Err(AlacError::InvalidInput(format!(
                "pcm length {} not divisible by channels {}",
                pcm.len(),
                num_channels
            )));
        }
        let num_samples = pcm.len() / num_channels;
        if num_samples == 0 {
            return Err(AlacError::InvalidInput("empty PCM block".into()));
        }
        if num_samples > self.config.frame_length as usize {
            return Err(AlacError::InvalidInput(format!(
                "block of {num_samples} samples exceeds frame_length {}",
                self.config.frame_length
            )));
        }

        let mut writer = BitWriter::new();
        // Group channels into CPE pairs, with a trailing SCE for an odd channel.
        let mut ch = 0usize;
        while ch < num_channels {
            if ch + 1 < num_channels {
                self.encode_element(&mut writer, pcm, num_channels, ch, true, num_samples)?;
                ch += 2;
            } else {
                self.encode_element(&mut writer, pcm, num_channels, ch, false, num_samples)?;
                ch += 1;
            }
        }
        // End-of-frame marker.
        writer.write_bits(TAG_END, 3);
        Ok(writer.finish())
    }

    /// Encode one syntax element (SCE or CPE) starting at channel `ch0`.
    fn encode_element(
        &self,
        writer: &mut BitWriter,
        pcm: &[i32],
        stride: usize,
        ch0: usize,
        pair: bool,
        num_samples: usize,
    ) -> AlacResult<()> {
        let bit_depth = u32::from(self.config.bit_depth);
        let channel_count = if pair { 2usize } else { 1usize };
        let partial = num_samples != self.config.frame_length as usize;
        let shift = self.bytes_shifted * 8;

        // Build the uncompressed escape candidate's exact bit length.
        let escape_bits =
            element_header_bits(partial) + num_samples * channel_count * bit_depth as usize;

        // Build the compressed candidate into a scratch writer to measure size.
        let compressed =
            self.build_compressed_element(pcm, stride, ch0, pair, num_samples, shift, partial);

        let use_compressed = match &compressed {
            Some((_, bits)) => *bits <= escape_bits,
            None => false,
        };

        if use_compressed {
            if let Some((bytes, bits)) = compressed {
                writer.append_bits(&bytes, bits);
            }
        } else {
            self.write_escape_element(
                writer,
                pcm,
                stride,
                ch0,
                channel_count,
                num_samples,
                partial,
            );
        }
        Ok(())
    }

    /// Build the compressed form of an element, returning its zero-padded bytes
    /// and exact bit length, or `None` if the geometry is degenerate.
    fn build_compressed_element(
        &self,
        pcm: &[i32],
        stride: usize,
        ch0: usize,
        pair: bool,
        num_samples: usize,
        shift: u32,
        partial: bool,
    ) -> Option<(Vec<u8>, usize)> {
        let bit_depth = u32::from(self.config.bit_depth);
        let channel_count = if pair { 2usize } else { 1usize };
        let extra = if pair { 1u32 } else { 0u32 };
        let chan_bits = bit_depth.saturating_sub(shift) + extra;
        if chan_bits == 0 || chan_bits > 32 {
            return None;
        }
        let low_mask: u32 = if shift == 0 { 0 } else { (1u32 << shift) - 1 };

        // Gather per-channel high parts (and the shifted low bits).
        let mut highs: Vec<Vec<i32>> = Vec::with_capacity(channel_count);
        let mut lows: Vec<Vec<u32>> = Vec::with_capacity(channel_count);
        for c in 0..channel_count {
            let mut high = vec![0i32; num_samples];
            let mut low = vec![0u32; num_samples];
            for s in 0..num_samples {
                let sample = pcm[s * stride + ch0 + c];
                if shift > 0 {
                    low[s] = (sample as u32) & low_mask;
                    high[s] = sample >> shift;
                } else {
                    high[s] = sample;
                }
            }
            highs.push(high);
            lows.push(low);
        }

        // Decorrelate (stereo only).
        let (mix_bits, mix_res, coded) = if pair {
            let mut interleaved = vec![0i32; num_samples * 2];
            for s in 0..num_samples {
                interleaved[2 * s] = highs[0][s];
                interleaved[2 * s + 1] = highs[1][s];
            }
            let (mr, u, v) = self.choose_mix(&interleaved, num_samples);
            (MIX_BITS, mr, vec![u, v])
        } else {
            (0u32, 0i32, vec![highs[0].clone()])
        };

        let mut writer = BitWriter::new();
        // Header.
        writer.write_bits(if pair { TAG_CPE } else { TAG_SCE }, 3);
        writer.write_bits(0, 12); // reserved
        writer.write_bit(partial);
        writer.write_bits(self.bytes_shifted, 2);
        writer.write_bit(false); // escape = 0 (compressed)
        if partial {
            writer.write_bits(num_samples as u32, 32);
        }
        if pair {
            writer.write_bits(mix_bits, 8);
            writer.write_signed(mix_res, 8);
        }

        let order = DEFAULT_ORDER.min(num_samples.saturating_sub(1));
        let denshift = DEFAULT_DENSHIFT;
        let init_coefs = initial_coefs(order, denshift);

        // Write predictor sub-headers (one per coded channel).
        for _ in 0..channel_count {
            writer.write_bits(0, 4); // mode 0
            writer.write_bits(denshift, 4);
            writer.write_bits(0, 3); // pb_factor 0 ⇒ use cookie pb
            writer.write_bits(order as u32, 5);
            for &c in &init_coefs {
                writer.write_signed(c, 16);
            }
        }

        // Shifted low bits, per channel interleaved.
        if shift > 0 {
            for s in 0..num_samples {
                for c in 0..channel_count {
                    writer.write_bits(lows[c][s], shift);
                }
            }
        }

        // Residual streams per coded channel.
        for samples in &coded {
            let mut coefs = init_coefs.clone();
            let residuals = predict_encode(samples, &mut coefs, chan_bits, denshift);
            let mut state = AgState::new(self.config.pb, self.config.mb, self.config.kb, chan_bits);
            encode_residuals(&mut writer, &residuals, &mut state);
        }

        Some(writer.finish_with_len())
    }

    /// Pick the stereo `mix_res` minimising the summed magnitude of the two
    /// coded channels (a cheap proxy for residual size).
    fn choose_mix(&self, interleaved: &[i32], num_samples: usize) -> (i32, Vec<i32>, Vec<i32>) {
        let mut best_res = 0i32;
        let mut best_cost = u64::MAX;
        let mut best_u = vec![0i32; num_samples];
        let mut best_v = vec![0i32; num_samples];
        let mut u = vec![0i32; num_samples];
        let mut v = vec![0i32; num_samples];
        for &res in &MIX_RES_CANDIDATES {
            mix_stereo(interleaved, num_samples, MIX_BITS, res, &mut u, &mut v);
            let cost: u64 = u
                .iter()
                .chain(v.iter())
                .map(|&x| u64::from((x as i64).unsigned_abs() as u32))
                .sum();
            if cost < best_cost {
                best_cost = cost;
                best_res = res;
                best_u.copy_from_slice(&u);
                best_v.copy_from_slice(&v);
            }
        }
        (best_res, best_u, best_v)
    }

    /// Write the uncompressed (escape) form of an element.
    fn write_escape_element(
        &self,
        writer: &mut BitWriter,
        pcm: &[i32],
        stride: usize,
        ch0: usize,
        channel_count: usize,
        num_samples: usize,
        partial: bool,
    ) {
        let bit_depth = u32::from(self.config.bit_depth);
        writer.write_bits(if channel_count == 2 { TAG_CPE } else { TAG_SCE }, 3);
        writer.write_bits(0, 12); // reserved
        writer.write_bit(partial);
        writer.write_bits(0, 2); // bytes_shifted 0 in escape form
        writer.write_bit(true); // escape = 1
        if partial {
            writer.write_bits(num_samples as u32, 32);
        }
        for s in 0..num_samples {
            for c in 0..channel_count {
                writer.write_signed(pcm[s * stride + ch0 + c], bit_depth);
            }
        }
    }
}

/// Number of header bits before the per-sample payload of an element.
fn element_header_bits(partial: bool) -> usize {
    // tag(3) + reserved(12) + partial(1) + bytes_shifted(2) + escape(1)
    let base = 3 + 12 + 1 + 2 + 1;
    base + if partial { 32 } else { 0 }
}

/// Initial FIR coefficients: a first-order predictor (predict the previous
/// sample), which the sign-LMS adaptation then refines.
fn initial_coefs(order: usize, denshift: u32) -> Vec<i32> {
    let mut coefs = vec![0i32; order];
    if order > 0 {
        coefs[0] = 1i32 << denshift;
    }
    coefs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_coefs_first_order() {
        let c = initial_coefs(8, 4);
        assert_eq!(c[0], 16);
        assert!(c[1..].iter().all(|&x| x == 0));
    }

    #[test]
    fn test_element_header_bits() {
        assert_eq!(element_header_bits(false), 19);
        assert_eq!(element_header_bits(true), 51);
    }

    #[test]
    fn test_encoder_rejects_misaligned_pcm() {
        let mut enc = AlacEncoder::new(AlacEncoderConfig {
            channels: 2,
            ..Default::default()
        })
        .expect("enc");
        // Odd length for stereo.
        assert!(enc.encode_frame(&[0i32; 5]).is_err());
    }
}
