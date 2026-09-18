//! JPEG 2000 encoder — top-level pipeline for both the 5-3 reversible
//! (lossless) and the CDF 9/7 irreversible (lossy) paths.
//!
//! Produces a raw J2K codestream that the existing [`super::Jpeg2000Decoder`]
//! reconstructs **byte-exact** (lossless 5-3) or **within the quantiser
//! tolerance** (lossy 9-7). The pipelines mirror the decoder in reverse:
//!
//! ```text
//! samples (one i32 plane per component)
//!     │  (no DC level shift — the decoder applies none either)
//!     ▼  forward 5-3 DWT  (decompose_levels, `levels` decompositions)
//!     │  OR forward CDF 9/7 DWT (decompose_levels_97) when `lossless = false`
//!     │  → SubbandTree (LL + HL/LH/HH per level)
//!     │  → (lossy only) per-subband mid-tread quantisation to i32 via
//!     │    `quantize_subband_97(coeffs, step_size, num_bit_planes)`
//!     ▼  subband → code-block partition (xcb/ycb), scanned in the decoder's order
//!     ▼  Tier-1 encode each code-block (forward EBCOT → MQ stream)
//!     ▼  Tier-2 single-layer packet (header + byte-aligned block bodies)
//!     ▼  marker assembly (SOC, SIZ, COD, QCD, SOT, SOD, <tile data>, EOC)
//! ```
//!
//! ## Scope (matches what the decoder reconstructs)
//!
//! - 5-3 reversible wavelet (lossless) — byte-exact round-trip.
//! - CDF 9/7 irreversible wavelet (lossy) — quantiser within ε/μ tolerance.
//!   Wave 10 Slice 2 ships ε = 8, μ = 0 for all subbands, which activates the
//!   decoder's lossless dequantisation shortcut and yields rounding-error-only
//!   loss on `bit_depth`-bounded inputs (see
//!   [`super::quantize_fwd`] for the mid-tread quantiser).
//! - Single quality layer, LRCP, no MCT (`mct = 0`).
//! - 8-bit and 16-bit unsigned components.
//! - 0..N decomposition levels (the decoder's `reconstruct_levels`); for
//!   multi-level decode the decoder reconstructs intermediate resolutions by
//!   doubling the detail-subband dimensions, so multi-level lossless round-trips
//!   are guaranteed for even / power-of-two image dimensions.
//! - Code-block coefficients must fit in `bit_depth` magnitude bit-planes (the
//!   fixed plane count the decoder uses); the encoder returns an error otherwise
//!   rather than silently truncating.
//!
//! ## Multi-component note
//!
//! The companion decoder decodes the *same* tile-data bytes independently for
//! every component (it re-reads from the tile start per component), so it cannot
//! carry distinct data per component within one tile. The encoder therefore
//! emits one tile body (built from component 0) and declares all components in
//! the SIZ marker; a multi-component image round-trips byte-exact when its
//! component planes are identical (e.g. greyscale replicated across RGB).

use super::marker_write::{
    write_cod, write_cod_lossy, write_eoc, write_qcd, write_qcd_lossy, write_siz, write_soc,
    write_sod, write_sot, ComponentSpec,
};
use super::markers::QcdMarker;
use super::quantize_fwd::quantize_subband_97;
use super::tier1_encode::encode_code_block;
use super::tier2_encode::assemble_packet;
use super::wavelet::{decompose_levels, decompose_levels_97, SubbandTree, SubbandTree97};
use super::{Jp2Error, Jp2Result};

/// Wave 10 Slice 2 simplified ε for the lossy QCD (style 2 expounded).
///
/// With μ = 0 and the decoder's `R_b = bit_depth`, choosing `ε = 8` yields
/// `Δ_b = 2^(bit_depth − 8) · 1` — equal to 1.0 for the 8-bit reference path,
/// which activates the lossless dequantisation shortcut in
/// [`super::tier1::CodeBlock::dequantize`]. See
/// [`super::quantize_fwd::quantize_subband_97`] for the forward direction.
const LOSSY_EPSILON: u8 = 8;

/// Configuration for the JPEG 2000 encoder.
///
/// Selects between the 5-3 reversible (lossless) and the CDF 9/7 irreversible
/// (lossy) pipelines via the [`lossless`](Self::lossless) flag. Both paths
/// share the same Tier-1/Tier-2 / marker assembly stages.
#[derive(Debug, Clone, Copy)]
pub struct Jpeg2000EncoderConfig {
    /// Number of wavelet decomposition levels (0 = none; the image is the
    /// LL band directly).
    pub levels: u8,
    /// Code-block width exponent (`xcb`): block width = `2^(xcb + 2)`. Valid 0..8.
    pub xcb: u8,
    /// Code-block height exponent (`ycb`): block height = `2^(ycb + 2)`. Valid 0..8.
    pub ycb: u8,
    /// Component bit depth (1..=16). All components share this depth.
    pub bit_depth: u8,
    /// `true`: 5-3 reversible (lossless) path. `false`: CDF 9/7 irreversible
    /// (lossy) path.
    pub lossless: bool,
}

impl Default for Jpeg2000EncoderConfig {
    fn default() -> Self {
        Self {
            levels: 1,
            xcb: 6, // 64-wide code-blocks
            ycb: 6, // 64-tall code-blocks
            bit_depth: 8,
            lossless: true,
        }
    }
}

impl Jpeg2000EncoderConfig {
    /// Code-block width in samples.
    #[must_use]
    fn cb_width(&self) -> usize {
        1usize << (usize::from(self.xcb) + 2)
    }
    /// Code-block height in samples.
    #[must_use]
    fn cb_height(&self) -> usize {
        1usize << (usize::from(self.ycb) + 2)
    }
}

/// JPEG 2000 lossless (5-3) encoder.
pub struct Jpeg2000Encoder {
    config: Jpeg2000EncoderConfig,
}

impl Jpeg2000Encoder {
    /// Create a new encoder with the given configuration.
    #[must_use]
    pub fn new(config: Jpeg2000EncoderConfig) -> Self {
        Self { config }
    }

    /// Create an encoder with default settings for a single 8-bit grey component
    /// and one decomposition level.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(Jpeg2000EncoderConfig::default())
    }

    /// Encode a single greyscale plane (`width × height`, row-major, one i32 per
    /// pixel) into a raw J2K codestream.
    pub fn encode_greyscale(
        &self,
        samples: &[i32],
        width: usize,
        height: usize,
    ) -> Jp2Result<Vec<u8>> {
        self.encode_planes(&[samples], width, height)
    }

    /// Encode one or more component planes into a raw J2K codestream.
    ///
    /// Each entry of `planes` is a full `width × height` plane (row-major). All
    /// planes share `config.bit_depth`. See the module note on the decoder's
    /// per-component behaviour: for byte-exact multi-component round-trips the
    /// planes must be identical.
    pub fn encode_planes(
        &self,
        planes: &[&[i32]],
        width: usize,
        height: usize,
    ) -> Jp2Result<Vec<u8>> {
        if planes.is_empty() {
            return Err(Jp2Error::InternalError(
                "at least one component plane is required".to_string(),
            ));
        }
        if width == 0 || height == 0 {
            return Err(Jp2Error::Unsupported("zero-dimension image".to_string()));
        }
        let bit_depth = self.config.bit_depth;
        if bit_depth == 0 || bit_depth > 16 {
            return Err(Jp2Error::Unsupported(format!(
                "unsupported bit depth {bit_depth} (1..=16)"
            )));
        }
        let num_levels = usize::from(self.config.levels);
        for (i, plane) in planes.iter().enumerate() {
            if plane.len() < width * height {
                return Err(Jp2Error::InternalError(format!(
                    "component {i} plane too small: expected {}, got {}",
                    width * height,
                    plane.len()
                )));
            }
            let max_val = if bit_depth >= 31 {
                i32::MAX
            } else {
                (1i32 << bit_depth) - 1
            };
            for &v in &plane[..width * height] {
                if v < 0 || v > max_val {
                    return Err(Jp2Error::Unsupported(format!(
                        "sample {v} out of range for {bit_depth}-bit unsigned [0,{max_val}]"
                    )));
                }
            }
        }

        // Build the tile body from component 0 (the decoder reads it for every
        // component — see the module note).
        let tile_data = self.encode_tile_component(planes[0], width, height)?;

        // Assemble the codestream.
        let mut out = Vec::with_capacity(tile_data.len() + 64);
        write_soc(&mut out);
        let comps: Vec<ComponentSpec> = (0..planes.len())
            .map(|_| ComponentSpec::unsigned(bit_depth))
            .collect();
        // Single tile covering the whole image.
        write_siz(
            &mut out,
            width as u32,
            height as u32,
            width as u32,
            height as u32,
            &comps,
        )?;
        if self.config.lossless {
            write_cod(
                &mut out,
                self.config.levels,
                self.config.xcb,
                self.config.ycb,
            );
            write_qcd(&mut out, self.config.levels)?;
        } else {
            write_cod_lossy(
                &mut out,
                self.config.levels,
                self.config.xcb,
                self.config.ycb,
            );
            // Wave 10 Slice 2 simplified §E.1 scheme: ε = 8, μ = 0 for every
            // subband. With the decoder's `R_b = bit_depth`, this gives
            // `step_size = 2^(bit_depth − 8)`, which equals 1.0 for the
            // 8-bit reference path and triggers the decoder's lossless
            // dequantisation shortcut (cast i32 → f64).
            let num_subbands = 1usize + 3 * usize::from(self.config.levels);
            let pairs: Vec<(u8, u16)> = (0..num_subbands).map(|_| (LOSSY_EPSILON, 0u16)).collect();
            write_qcd_lossy(&mut out, self.config.levels, &pairs)?;
        }
        // Psot = bytes from the start of the SOT marker to the end of the tile
        // data, i.e. SOT segment (12) + SOD marker (2) + tile data. Setting it
        // lets the decoder delimit the tile by length instead of scanning for a
        // marker (the entropy data may contain marker-like byte pairs).
        let psot_usize = 12 + 2 + tile_data.len();
        let psot = u32::try_from(psot_usize)
            .map_err(|_| Jp2Error::InternalError("tile-part too long for Psot".to_string()))?;
        write_sot(&mut out, 0, psot, 0, 1);
        write_sod(&mut out);
        out.extend_from_slice(&tile_data);
        write_eoc(&mut out);

        let _ = num_levels;
        Ok(out)
    }

    /// Forward-transform one component plane and produce its tile body (Tier-2
    /// packet: header + byte-aligned code-block bodies in the decoder's order).
    ///
    /// Dispatches between the lossless 5-3 and the lossy 9-7 sub-pipelines on
    /// `config.lossless`.
    fn encode_tile_component(
        &self,
        plane: &[i32],
        width: usize,
        height: usize,
    ) -> Jp2Result<Vec<u8>> {
        let num_levels = usize::from(self.config.levels);
        let block_streams = if self.config.lossless {
            let tree = decompose_levels(plane, width, height, num_levels)?;
            self.encode_subbands(&tree, width, height)?
        } else {
            // Forward CDF 9/7 path: cast samples to f64, decompose, quantise.
            let f64_plane: Vec<f64> = plane[..width * height].iter().map(|&v| v as f64).collect();
            let tree97 = decompose_levels_97(&f64_plane, width, height, num_levels)?;
            self.encode_subbands_97(&tree97)?
        };
        assemble_packet(&block_streams)
    }

    /// Partition every subband into code-blocks (in the exact scan order the
    /// decoder consumes: LL first, then HL/LH/HH per level coarsest→finest, each
    /// subband block-row major) and Tier-1 encode each block.
    ///
    /// Returns one MQ byte stream per code-block; an all-zero block yields an
    /// empty stream (marking it excluded from the packet).
    fn encode_subbands(
        &self,
        tree: &SubbandTree,
        _width: usize,
        _height: usize,
    ) -> Jp2Result<Vec<Vec<u8>>> {
        let cb_w = self.config.cb_width();
        let cb_h = self.config.cb_height();
        let num_bit_planes = self.config.bit_depth;
        let mut streams: Vec<Vec<u8>> = Vec::new();

        // LL subband.
        self.encode_subband_blocks(
            &tree.ll,
            tree.ll_width,
            tree.ll_height,
            cb_w,
            cb_h,
            num_bit_planes,
            &mut streams,
        )?;

        // Detail subbands, coarsest (index 0) to finest, HL/LH/HH per level.
        for level in &tree.levels {
            for subband in [&level.hl, &level.lh, &level.hh] {
                self.encode_subband_blocks(
                    subband,
                    level.width,
                    level.height,
                    cb_w,
                    cb_h,
                    num_bit_planes,
                    &mut streams,
                )?;
            }
        }

        Ok(streams)
    }

    /// Lossy (9-7) counterpart of [`encode_subbands`]: quantise the f64
    /// wavelet subbands to i32 with the QCD-derived step sizes (one per
    /// subband, in the LL → HL/LH/HH coarse-to-fine QCD order) and then
    /// route through the existing forward Tier-1 / Tier-2 path.
    fn encode_subbands_97(&self, tree: &SubbandTree97) -> Jp2Result<Vec<Vec<u8>>> {
        let cb_w = self.config.cb_width();
        let cb_h = self.config.cb_height();
        let num_bit_planes = self.config.bit_depth;
        let mut streams: Vec<Vec<u8>> = Vec::new();

        // Build the QCD-style step size for each subband from the current
        // ε = 8 / μ = 0 policy. We use the parser to mirror exactly what the
        // decoder will compute so that the encoder and decoder share a single
        // source of truth for step sizes.
        let num_subbands = 1usize + 3 * tree.levels.len();
        let pairs: Vec<u16> = (0..num_subbands)
            .map(|_| (u16::from(LOSSY_EPSILON) << 11) | 0)
            .collect();
        let qcd_for_steps = QcdMarker {
            sqcd: 0x02, // guard bits = 0, style = 2 (expounded)
            step_sizes: pairs,
        };

        // LL subband: subband index 0.
        let step_ll = qcd_for_steps.step_size_for_subband(0, self.config.bit_depth);
        let ll_q = quantize_subband_97(&tree.ll, step_ll, num_bit_planes);
        self.encode_subband_blocks(
            &ll_q,
            tree.ll_width,
            tree.ll_height,
            cb_w,
            cb_h,
            num_bit_planes,
            &mut streams,
        )?;

        // Detail subbands, coarsest (index 0) to finest, HL / LH / HH per level.
        let mut qcd_idx = 1usize;
        for level in &tree.levels {
            for subband in [&level.hl, &level.lh, &level.hh] {
                let step = qcd_for_steps.step_size_for_subband(qcd_idx, self.config.bit_depth);
                let q = quantize_subband_97(subband, step, num_bit_planes);
                self.encode_subband_blocks(
                    &q,
                    level.width,
                    level.height,
                    cb_w,
                    cb_h,
                    num_bit_planes,
                    &mut streams,
                )?;
                qcd_idx += 1;
            }
        }

        Ok(streams)
    }

    /// Encode the code-blocks of one subband (block-row major) into `streams`.
    #[allow(clippy::too_many_arguments)]
    fn encode_subband_blocks(
        &self,
        coeffs: &[i32],
        sub_w: usize,
        sub_h: usize,
        cb_w: usize,
        cb_h: usize,
        num_bit_planes: u8,
        streams: &mut Vec<Vec<u8>>,
    ) -> Jp2Result<()> {
        if sub_w == 0 || sub_h == 0 {
            return Ok(());
        }
        if coeffs.len() < sub_w * sub_h {
            return Err(Jp2Error::InternalError(format!(
                "subband buffer too small: expected {}, got {}",
                sub_w * sub_h,
                coeffs.len()
            )));
        }
        let num_cb_h = sub_w.div_ceil(cb_w);
        let num_cb_v = sub_h.div_ceil(cb_h);
        let mag_limit: i64 = 1i64 << num_bit_planes;
        let lossless = self.config.lossless;
        // Lossy mode clamps overflowing magnitudes to (mag_limit − 1) instead
        // of erroring — the resulting decode error is bounded by the quantiser
        // step size at the affected coefficients.
        let mag_clip = (mag_limit - 1) as i32;

        for block_row in 0..num_cb_v {
            for block_col in 0..num_cb_h {
                let cur_cb_w = cb_w.min(sub_w - block_col * cb_w);
                let cur_cb_h = cb_h.min(sub_h - block_row * cb_h);

                // Extract the block's coefficients (row-major within the block).
                let mut block = vec![0i32; cur_cb_w * cur_cb_h];
                let mut all_zero = true;
                for r in 0..cur_cb_h {
                    let src_row = (block_row * cb_h + r) * sub_w + block_col * cb_w;
                    for c in 0..cur_cb_w {
                        let v = coeffs[src_row + c];
                        let stored = if i64::from(v.abs()) >= mag_limit {
                            if lossless {
                                return Err(Jp2Error::Unsupported(format!(
                                    "coefficient magnitude {} exceeds {num_bit_planes} bit-planes;\
                                     \nlossless round-trip would not be exact",
                                    v.abs()
                                )));
                            }
                            if v < 0 {
                                -mag_clip
                            } else {
                                mag_clip
                            }
                        } else {
                            v
                        };
                        block[r * cur_cb_w + c] = stored;
                        if stored != 0 {
                            all_zero = false;
                        }
                    }
                }

                if all_zero {
                    // Excluded block — empty stream.
                    streams.push(Vec::new());
                } else {
                    let stream = encode_code_block(&block, cur_cb_w, cur_cb_h, num_bit_planes)?;
                    streams.push(stream);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg2000::decoder::Jpeg2000Decoder;

    fn decode(bytes: &[u8]) -> crate::jpeg2000::decoder::DecodedImage {
        Jpeg2000Decoder::decode(bytes).expect("decode")
    }

    #[test]
    fn config_defaults() {
        let c = Jpeg2000EncoderConfig::default();
        assert_eq!(c.cb_width(), 256);
        assert_eq!(c.cb_height(), 256);
        assert!(c.lossless);
    }

    #[test]
    fn roundtrip_constant_16x16_one_level() {
        let w = 16;
        let h = 16;
        let plane = vec![137i32; w * h];
        let cfg = Jpeg2000EncoderConfig {
            levels: 1,
            xcb: 4,
            ycb: 4,
            bit_depth: 8,
            lossless: true,
        };
        let enc = Jpeg2000Encoder::new(cfg);
        let bytes = enc.encode_greyscale(&plane, w, h).expect("encode");
        let img = decode(&bytes);
        assert_eq!(img.width as usize, w);
        assert_eq!(img.height as usize, h);
        for (i, &s) in img.samples[0].iter().enumerate() {
            assert_eq!(s, 137u16, "pixel {i}");
        }
    }

    #[test]
    fn roundtrip_gradient_32x32_one_level() {
        let w = 32;
        let h = 32;
        let plane: Vec<i32> = (0..w * h)
            .map(|i| ((i % w) * 255 / (w - 1)) as i32)
            .collect();
        let cfg = Jpeg2000EncoderConfig {
            levels: 1,
            xcb: 5,
            ycb: 5,
            bit_depth: 8,
            lossless: true,
        };
        let enc = Jpeg2000Encoder::new(cfg);
        let bytes = enc.encode_greyscale(&plane, w, h).expect("encode");
        let img = decode(&bytes);
        for (i, (&orig, &dec)) in plane.iter().zip(img.samples[0].iter()).enumerate() {
            assert_eq!(orig, i32::from(dec), "pixel {i}");
        }
    }
}
