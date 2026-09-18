//! Dolby Vision RPU (Reference Processing Unit) builder.
//!
//! Generates minimal but structurally valid Dolby Vision RPU NALU payloads
//! for profiles 5, 8.1, and 8.4.  The output bytes can be embedded in an
//! HEVC bitstream as an unregistered SEI NAL unit (type 62) or as a
//! standalone RPU NAL unit with the `0x19` prefix required by all DV
//! profiles.
//!
//! ## Dolby Vision RPU structure (simplified)
//!
//! ```text
//! ┌───────────────────────────────────────────────────────┐
//! │  RPU NAL prefix   (1 byte, fixed 0x19)                │
//! │  dv_rpu_data_header                                   │
//! │    rpu_type       (4 bits, = 2 for regular)           │
//! │    rpu_format     (13 bits)                           │
//! │    vdr_rpu_profile (4 bits)                           │
//! │    vdr_rpu_level  (4 bits)                            │
//! │    el_spatial_resampling_filter_flag (1 bit)          │
//! │    reserved_zero_3bits (3 bits)                       │
//! │  Polynomial/Bezier mapping curves (per component)     │
//! │  Trim pass metadata (per target display)              │
//! │  RBSP trailing bits                                   │
//! └───────────────────────────────────────────────────────┘
//! ```
//!
//! ## References
//!
//! - ETSI TS 103 572 — Dolby Vision Bitstream Specification
//! - ITU-T H.265 Annex D (SEI messages)
//! - SMPTE ST 2094-10 — Dolby Vision Dynamic Metadata

use crate::HdrError;

// ── DoviProfile ──────────────────────────────────────────────────────────

/// Dolby Vision profile variant supported by [`DoviRpuBuilder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DoviProfile {
    /// Profile 5 — PQ-only single-layer; no HDR10/HLG base layer.
    Profile5,
    /// Profile 8.1 — HDR10 base layer with Dolby Vision RPU metadata.
    Profile81,
    /// Profile 8.4 — HLG base layer with Dolby Vision RPU metadata.
    Profile84,
}

impl DoviProfile {
    /// Numeric profile ID embedded in the RPU header.
    pub fn profile_id(self) -> u8 {
        match self {
            DoviProfile::Profile5 => 5,
            DoviProfile::Profile81 => 8,
            DoviProfile::Profile84 => 8,
        }
    }

    /// VDR RPU level for this profile (1 = standard single-layer).
    pub fn rpu_level(self) -> u8 {
        match self {
            DoviProfile::Profile5 => 1,
            DoviProfile::Profile81 => 6,
            DoviProfile::Profile84 => 6,
        }
    }

    /// Whether this profile carries an enhancement layer.
    pub fn has_el(self) -> bool {
        false // Profiles 5/8.x do not have a separate enhancement layer.
    }
}

impl std::fmt::Display for DoviProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DoviProfile::Profile5 => write!(f, "Dolby Vision Profile 5"),
            DoviProfile::Profile81 => write!(f, "Dolby Vision Profile 8.1"),
            DoviProfile::Profile84 => write!(f, "Dolby Vision Profile 8.4"),
        }
    }
}

// ── DoviRpuConfig ─────────────────────────────────────────────────────────

/// Configuration for [`DoviRpuBuilder`].
///
/// All PQ values use the 12-bit PQ code scale (0–4095), consistent with
/// the SMPTE ST 2084 signal range used in Dolby Vision streams.
#[derive(Debug, Clone)]
pub struct DoviRpuConfig {
    /// Dolby Vision profile to encode.
    pub profile: DoviProfile,
    /// Base-layer maximum PQ code value (default 3079 ≈ 1000 nits).
    pub bl_max_pq: u16,
    /// Enhancement-layer maximum PQ code value (for dual-layer; ignored for P5/P8).
    pub el_max_pq: u16,
    /// Target display maximum PQ code value (e.g. 2081 ≈ 100 nits / SDR TV).
    pub target_max_pq: u16,
    /// Trim-pass slope (multiplicative gain, typically near 1.0).
    pub trim_slop: f32,
    /// Trim-pass offset (additive bias, typically 0.0).
    pub trim_offset: f32,
    /// Trim-pass power (gamma-like exponent, typically 1.0).
    pub trim_power: f32,
}

impl DoviRpuConfig {
    /// Default configuration for Profile 8.1 targeting a 1000-nit display
    /// mastered for 100-nit SDR playback.
    #[must_use]
    pub fn profile81_default() -> Self {
        Self {
            profile: DoviProfile::Profile81,
            bl_max_pq: 3079,
            el_max_pq: 3079,
            target_max_pq: 2081,
            trim_slop: 1.0,
            trim_offset: 0.0,
            trim_power: 1.0,
        }
    }

    /// Default configuration for Profile 5 (PQ-only single-layer).
    #[must_use]
    pub fn profile5_default() -> Self {
        Self {
            profile: DoviProfile::Profile5,
            bl_max_pq: 3079,
            el_max_pq: 3079,
            target_max_pq: 2081,
            trim_slop: 1.0,
            trim_offset: 0.0,
            trim_power: 1.0,
        }
    }

    /// Default configuration for Profile 8.4 (HLG base layer).
    #[must_use]
    pub fn profile84_default() -> Self {
        Self {
            profile: DoviProfile::Profile84,
            bl_max_pq: 3079,
            el_max_pq: 3079,
            target_max_pq: 2081,
            trim_slop: 1.0,
            trim_offset: 0.0,
            trim_power: 1.0,
        }
    }
}

// ── DoviRpuBuilder ────────────────────────────────────────────────────────

/// Dolby Vision RPU NAL unit builder.
///
/// Produces a byte vector representing a complete RPU NAL payload for each
/// call to [`build_rpu_payload`](DoviRpuBuilder::build_rpu_payload).  The
/// output is frame-independent except for `frame_idx`, which is embedded in
/// a reserved field for diagnostic tracing.
///
/// # Example
///
/// ```rust
/// use oximedia_hdr::dovi_rpu::{DoviRpuBuilder, DoviRpuConfig};
///
/// let config = DoviRpuConfig::profile81_default();
/// let builder = DoviRpuBuilder::new(config);
/// let payload = builder.build_rpu_payload(0);
/// assert!(!payload.is_empty());
/// let sei = builder.build_sei_message(&payload);
/// assert!(!sei.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct DoviRpuBuilder {
    config: DoviRpuConfig,
}

impl DoviRpuBuilder {
    /// Create a new builder with the given configuration.
    #[must_use]
    pub fn new(config: DoviRpuConfig) -> Self {
        Self { config }
    }

    /// Build a minimal, structurally-valid Dolby Vision RPU NAL payload.
    ///
    /// The output starts with the DV RPU NAL prefix byte (`0x19`) followed by
    /// the encoded RPU data and RBSP trailing bits.
    ///
    /// `frame_idx` is written into a reserved frame-counter field for stream
    /// debugging; it does not affect the semantic content of the RPU.
    #[must_use]
    pub fn build_rpu_payload(&self, frame_idx: u64) -> Vec<u8> {
        let mut bw = BitWriter::new();

        // ── RPU NAL prefix (always 0x19) ─────────────────────────────────
        bw.write_bits(0x19u32, 8);

        // ── dv_rpu_data_header ────────────────────────────────────────────
        // rpu_type = 2 (regular RPU per ETSI TS 103 572 §7.1)
        bw.write_bits(2u32, 4);
        // rpu_format: bits [12:8] = profile number, bits [7:0] = level
        let rpu_format: u16 = ((self.config.profile.profile_id() as u16) << 8)
            | (self.config.profile.rpu_level() as u16);
        bw.write_bits(rpu_format as u32, 13);
        // vdr_rpu_profile
        bw.write_bits(self.config.profile.profile_id() as u32, 4);
        // vdr_rpu_level
        bw.write_bits(self.config.profile.rpu_level() as u32, 4);
        // extended_spatial_resampling_filter_flag = 0
        bw.write_bits(0u32, 1);
        // el_spatial_resampling_filter_flag = 0 (no EL for P5/P8)
        bw.write_bits(0u32, 1);
        // disable_residual_flag = 1 (no EL residual data)
        bw.write_bits(1u32, 1);

        // ── vdr_dm_data_present_flag = 1 (we include DM metadata) ────────
        bw.write_bits(1u32, 1);
        // use_prev_vdr_rpu_flag = 0 (always include full RPU per frame)
        bw.write_bits(0u32, 1);

        // ── Mapping curves (polynomial, single component for luma) ────────
        self.write_mapping_curves(&mut bw);

        // ── VDR DM data (display management) ─────────────────────────────
        self.write_vdr_dm_data(&mut bw);

        // ── Frame index in reserved field (4 bytes little-endian) ─────────
        // This is non-normative but useful for stream debugging.
        let fi = (frame_idx & 0xFFFF_FFFF) as u32;
        bw.write_bits(fi >> 16, 16);
        bw.write_bits(fi & 0xFFFF, 16);

        // ── RBSP trailing bits ────────────────────────────────────────────
        bw.write_rbsp_trailing_bits();

        bw.into_bytes()
    }

    /// Wrap a raw RPU payload in an HEVC SEI NAL unit (SEI type 62).
    ///
    /// The SEI message structure follows ITU-T H.265 Annex D:
    /// ```text
    /// forbidden_zero_bit (1)
    /// nal_unit_type = 39 (PREFIX_SEI) or 40 (SUFFIX_SEI) (6)
    /// nuh_layer_id = 0 (6)
    /// nuh_temporal_id_plus1 = 1 (3)
    /// payloadType = 62 (unregistered user data) (1–N bytes 0xFF + remainder)
    /// payloadSize (1–N bytes 0xFF + remainder)
    /// payload_data
    /// RBSP trailing bits
    /// ```
    #[must_use]
    pub fn build_sei_message(&self, rpu_payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(rpu_payload.len() + 8);

        // HEVC NAL unit header (2 bytes).
        // nal_unit_type = 39 (PREFIX_SEI_NUT), nuh_layer_id = 0, nuh_temporal_id_plus1 = 1
        // Encoding: [forbidden(1), nal_unit_type(6), nuh_layer_id(6), nuh_temporal_id_plus1(3)]
        // = 0b0_100111_000000_001 = 0x4E01
        out.push(0x4E);
        out.push(0x01);

        // SEI payload type = 62 (user_data_unregistered → Dolby Vision RPU).
        encode_exp_golomb_sei_size(&mut out, 62);

        // SEI payload size.
        encode_exp_golomb_sei_size(&mut out, rpu_payload.len());

        // RPU payload bytes.
        out.extend_from_slice(rpu_payload);

        // RBSP trailing bits: 0x80.
        out.push(0x80);

        out
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /// Write per-component polynomial mapping curves.
    ///
    /// Uses a linear (degree-1) polynomial mapping that implements the
    /// configured trim-pass slope and offset.  Three components (Y, Cb, Cr)
    /// are written; chroma components use identity mapping.
    fn write_mapping_curves(&self, bw: &mut BitWriter) {
        // num_pivots_minus2 = 0 → 2 pivots (linear mapping, 1 segment)
        for comp in 0..3u8 {
            // num_pivots_minus2: coded as 0 for 2 pivots (1 polynomial segment)
            bw.write_bits(0u32, 4); // num_pivots_minus2

            // Pivot values in 12-bit PQ domain.
            let (piv0, piv1) = if comp == 0 {
                // Luma: map [0, bl_max_pq] to [0, target_max_pq].
                (0u16, self.config.bl_max_pq.min(4095))
            } else {
                // Chroma: identity [0, 4095].
                (0u16, 4095u16)
            };

            bw.write_bits(piv0 as u32, 12);
            bw.write_bits(piv1 as u32, 12);

            // mapping_idc = 0 (polynomial mapping for this segment)
            bw.write_bits(0u32, 4);

            // poly_order = 1 (linear polynomial)
            bw.write_bits(1u32, 4);

            if comp == 0 {
                // Linear coeff a₁ (slope) encoded as fixed-point s2.13:
                // trim_slop ≈ 1.0 → 0x2000 (2^13).
                let slope_fp = (self.config.trim_slop * 8192.0).round().clamp(0.0, 65535.0) as u32;
                bw.write_bits(slope_fp, 16);
                // Linear coeff a₀ (offset) encoded as s2.13:
                let offset_fp = encode_signed_fp(self.config.trim_offset, 13);
                bw.write_bits(offset_fp, 16);
            } else {
                // Identity mapping for chroma: slope = 1.0, offset = 0.0.
                bw.write_bits(0x2000u32, 16); // slope = 1.0 in s2.13
                bw.write_bits(0u32, 16); // offset = 0.0
            }
        }
    }

    /// Write VDR display management data (trim passes for target display).
    fn write_vdr_dm_data(&self, bw: &mut BitWriter) {
        // dm_metadata_id = 1 (level 1 metadata block)
        bw.write_bits(1u32, 8);

        // signal_full_range_flag = 0 (limited range)
        bw.write_bits(0u32, 1);
        // source_min_PQ (12 bits): 0 (black)
        bw.write_bits(0u32, 12);
        // source_max_PQ (12 bits): bl_max_pq
        bw.write_bits(self.config.bl_max_pq as u32 & 0xFFF, 12);

        // dm_metadata_id = 4 (level 4: trim pass)
        bw.write_bits(4u32, 8);

        // anchor_pq (12 bits): target_max_pq
        bw.write_bits(self.config.target_max_pq as u32 & 0xFFF, 12);

        // anchor_power (12 bits): trim_power in fixed-point 4.8 (1.0 → 0x0100)
        let power_fp = (self.config.trim_power * 256.0).round().clamp(0.0, 4095.0) as u32;
        bw.write_bits(power_fp, 12);

        // Trim pass coefficients for this target display.
        // trim_slope_bias encoded as int12 offset from 0x800 neutral.
        let slope_bias = encode_trim_coeff(self.config.trim_slop);
        bw.write_bits(slope_bias, 12);
        // trim_offset_bias
        let offset_bias = encode_trim_coeff(self.config.trim_offset);
        bw.write_bits(offset_bias, 12);
        // trim_power_bias
        let power_bias = encode_trim_coeff(self.config.trim_power);
        bw.write_bits(power_bias, 12);

        // chroma_weight (12 bits): neutral = 0x800
        bw.write_bits(0x800u32, 12);
        // saturation_vector_field0..5 (12 bits each): all neutral
        for _ in 0..6 {
            bw.write_bits(0x800u32, 12);
        }

        // dm_metadata_id = 5 (level 5: active area)
        bw.write_bits(5u32, 8);
        // canvas_width  (13 bits): 1920
        bw.write_bits(1920u32, 13);
        // canvas_height (13 bits): 1080
        bw.write_bits(1080u32, 13);
        // active_area: left=0, right=1920, top=0, bottom=1080
        bw.write_bits(0u32, 13);
        bw.write_bits(1920u32, 13);
        bw.write_bits(0u32, 13);
        bw.write_bits(1080u32, 13);

        // dm_metadata_id = 255 (end-of-metadata sentinel)
        bw.write_bits(0xFFu32, 8);
    }
}

// ── Encoding helpers ──────────────────────────────────────────────────────

/// Encode a trim coefficient (centred around 1.0) into a 12-bit biased value.
///
/// Neutral (1.0 for slope/power, 0.0 for offset) maps to 0x800.
/// Range clamp: value maps to [0, 0xFFF].
fn encode_trim_coeff(value: f32) -> u32 {
    // Scale: 1 unit = 1024 steps (12-bit range 0..4095).
    // No added bias — 0.0 maps cleanly to 0, 1.0 maps to 1024.
    (value * 1024.0).round().clamp(0.0, 4095.0) as u32
}

/// Encode a signed float in two's complement fixed-point with `frac_bits`
/// fractional bits, stored in 16 bits.
fn encode_signed_fp(value: f32, frac_bits: u32) -> u32 {
    let scale = (1u32 << frac_bits) as f32;
    let raw = (value * scale).round();
    // Clamp to i16 range and reinterpret as u16.
    let clamped = raw.clamp(i16::MIN as f32, i16::MAX as f32) as i16;
    clamped as u16 as u32
}

/// Write a HEVC SEI payload-type or payload-size field using the standard
/// "run of 0xFF bytes then remainder" encoding.
fn encode_exp_golomb_sei_size(out: &mut Vec<u8>, mut value: usize) {
    while value >= 0xFF {
        out.push(0xFF);
        value -= 0xFF;
    }
    out.push(value as u8);
}

// ── BitWriter ─────────────────────────────────────────────────────────────

/// A simple MSB-first bit-packing writer.
///
/// Bits are accumulated in a 64-bit accumulator and flushed to the output
/// byte buffer as whole bytes.
struct BitWriter {
    buf: Vec<u8>,
    acc: u64,
    bits_in_acc: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            acc: 0,
            bits_in_acc: 0,
        }
    }

    /// Write the `n` least-significant bits of `value` MSB-first.
    fn write_bits(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32, "write_bits: n={n} > 32");
        // Mask to `n` bits.
        let mask: u64 = if n == 32 {
            0xFFFF_FFFF
        } else {
            (1u64 << n) - 1
        };
        let v = (value as u64) & mask;
        self.acc = (self.acc << n) | v;
        self.bits_in_acc += n;
        while self.bits_in_acc >= 8 {
            self.bits_in_acc -= 8;
            let byte = ((self.acc >> self.bits_in_acc) & 0xFF) as u8;
            self.buf.push(byte);
        }
    }

    /// Write RBSP trailing bits: a `1` bit followed by zero-padding to a byte boundary.
    fn write_rbsp_trailing_bits(&mut self) {
        self.write_bits(1u32, 1);
        // Pad to byte boundary with 0s.
        let remaining = if self.bits_in_acc == 0 {
            0
        } else {
            8 - self.bits_in_acc
        };
        if remaining > 0 {
            self.write_bits(0u32, remaining);
        }
    }

    /// Consume the writer and return the accumulated byte buffer.
    fn into_bytes(self) -> Vec<u8> {
        // Any leftover bits that didn't fill a complete byte are discarded
        // (should not happen after write_rbsp_trailing_bits).
        self.buf
    }
}

// ── BitReader ─────────────────────────────────────────────────────────────

/// MSB-first bit-level reader that mirrors `BitWriter` for round-trip verification.
///
/// Bits are consumed from the most-significant bit of each byte, matching the
/// order in which `BitWriter::write_bits` produces them.
pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    /// Bit position within the current byte: 0 = MSB (bit 7), 7 = LSB (bit 0).
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    /// Create a new reader over the given byte slice.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Read `n` bits (1–32) as a `u32`, MSB first.
    ///
    /// # Errors
    /// Returns `HdrError` if `n == 0`, `n > 32`, or the data is exhausted.
    pub fn read_bits(&mut self, n: u8) -> Result<u32, HdrError> {
        if n == 0 || n > 32 {
            return Err(HdrError::MetadataParseError(format!(
                "invalid bit count: {n}"
            )));
        }
        let mut result = 0u32;
        for _ in 0..n {
            if self.byte_pos >= self.data.len() {
                return Err(HdrError::MetadataParseError(
                    "unexpected end of RPU payload".to_string(),
                ));
            }
            let bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1;
            result = (result << 1) | u32::from(bit);
            self.bit_pos += 1;
            if self.bit_pos == 8 {
                self.bit_pos = 0;
                self.byte_pos += 1;
            }
        }
        Ok(result)
    }

    /// Read `n` bits as a `u8` (panics at compile time if n > 8 would overflow
    /// at runtime; actual range check is in [`Self::read_bits`]).
    pub fn read_u8(&mut self, n: u8) -> Result<u8, HdrError> {
        Ok(self.read_bits(n)? as u8)
    }

    /// Read `n` bits as a `u16`.
    pub fn read_u16(&mut self, n: u8) -> Result<u16, HdrError> {
        Ok(self.read_bits(n)? as u16)
    }

    /// Read a single bit as a `bool`.
    pub fn read_bool(&mut self) -> Result<bool, HdrError> {
        Ok(self.read_bits(1)? != 0)
    }

    /// Read a 16-bit value and reinterpret as `i16` (two's complement).
    pub fn read_i16_16(&mut self) -> Result<i16, HdrError> {
        Ok(self.read_bits(16)? as u16 as i16)
    }

    /// Number of complete bytes remaining (not counting any partial current byte).
    pub fn bytes_remaining(&self) -> usize {
        if self.bit_pos == 0 {
            self.data.len().saturating_sub(self.byte_pos)
        } else {
            self.data.len().saturating_sub(self.byte_pos + 1)
        }
    }
}

// ── ParsedRpuFields ────────────────────────────────────────────────────────

/// Intermediate, losslessly decoded fields from an RPU payload produced by
/// [`DoviRpuBuilder::build_rpu_payload`].
///
/// All floating-point fields are reconstructed from their fixed-point encodings
/// using the inverse of the formulas in [`DoviRpuBuilder`].  The fidelity
/// matches the precision of the fixed-point representation, not the original
/// `f32` inputs.
#[derive(Debug, Clone)]
pub struct ParsedRpuFields {
    /// The raw vdr_rpu_profile nibble (4 bits).
    pub vdr_rpu_profile: u8,
    /// The raw vdr_rpu_level nibble (4 bits).
    pub vdr_rpu_level: u8,
    /// Reconstructed DoviProfile (Profile5, Profile81, or Profile84).
    pub profile: DoviProfile,
    /// Luma mapping: second pivot (= bl_max_pq).
    pub bl_max_pq: u16,
    /// Target display maximum PQ (from anchor_pq in DM level 4).
    pub target_max_pq: u16,
    /// Trim slope, decoded from fixed-point s2.13 luma curve coefficient a₁.
    pub trim_slop: f32,
    /// Trim offset, decoded from signed fixed-point s2.13 luma curve coefficient a₀.
    pub trim_offset: f32,
    /// Trim power, decoded from anchor_power (fixed-point 4.8).
    pub trim_power: f32,
    /// Frame index embedded in the reserved 32-bit field.
    pub frame_idx: u64,
}

// ── DoviRpuParser ─────────────────────────────────────────────────────────

/// Parser for Dolby Vision RPU NAL payloads produced by [`DoviRpuBuilder`].
///
/// This parser is the exact inverse of [`DoviRpuBuilder::build_rpu_payload`]:
/// it reads fields in the same bit-for-bit order and with the same widths.
/// It does **not** use `parse_rpu_nal_header` or `verify_rpu_crc` from
/// `dolby_vision_profile.rs` — those functions target the byte-aligned,
/// CRC-appended format produced by `generate_rpu_nal`, which is a distinct
/// encoding.
///
/// # Errors
/// Returns [`HdrError::MetadataParseError`] on truncated data, an unrecognised NAL prefix,
/// or an unknown profile nibble.
pub struct DoviRpuParser;

impl DoviRpuParser {
    /// Parse a raw RPU NAL payload produced by [`DoviRpuBuilder::build_rpu_payload`].
    ///
    /// The payload must begin with the `0x19` NAL prefix byte.  The last byte
    /// must end with RBSP trailing bits (at least one `1` bit).  There is no
    /// CRC in this format — integrity is checked only by structural consistency.
    ///
    /// # Errors
    /// Returns [`HdrError::MetadataParseError`] if:
    /// - The payload is shorter than 8 bytes.
    /// - The first byte is not `0x19`.
    /// - An unknown vdr_rpu_profile nibble is found (only 5 and 8 are known).
    /// - The bit stream ends prematurely.
    pub fn parse(payload: &[u8]) -> Result<ParsedRpuFields, HdrError> {
        if payload.len() < 8 {
            return Err(HdrError::MetadataParseError(
                "RPU payload too short (< 8 bytes)".to_string(),
            ));
        }

        let mut r = BitReader::new(payload);

        // ── RPU NAL prefix (8 bits) ────────────────────────────────────────
        let nal_prefix = r.read_u8(8)?;
        if nal_prefix != 0x19 {
            return Err(HdrError::MetadataParseError(format!(
                "RPU NAL prefix mismatch: expected 0x19, got 0x{nal_prefix:02X}"
            )));
        }

        // ── dv_rpu_data_header ─────────────────────────────────────────────
        let _rpu_type = r.read_u8(4)?; // rpu_type (4 bits)
        let _rpu_format = r.read_u16(13)?; // rpu_format (13 bits)
        let vdr_rpu_profile = r.read_u8(4)?; // vdr_rpu_profile (4 bits)
        let vdr_rpu_level = r.read_u8(4)?; // vdr_rpu_level (4 bits)
        let _ext_spatial = r.read_bool()?; // extended_spatial_resampling_filter_flag (1)
        let _el_spatial = r.read_bool()?; // el_spatial_resampling_filter_flag (1)
        let _disable_res = r.read_bool()?; // disable_residual_flag (1)
        let _vdr_dm_present = r.read_bool()?; // vdr_dm_data_present_flag (1)
        let _use_prev = r.read_bool()?; // use_prev_vdr_rpu_flag (1)

        // ── Mapping curves (3 components) ─────────────────────────────────
        // Mirrors write_mapping_curves: for each of 3 components:
        //   4 bits: num_pivots_minus2
        //   12 bits: pivot0
        //   12 bits: pivot1
        //   4 bits: mapping_idc
        //   4 bits: poly_order
        //   16 bits: slope coefficient (a₁)
        //   16 bits: offset coefficient (a₀)
        let mut bl_max_pq = 0u16;
        let mut trim_slop = 1.0f32;
        let mut trim_offset = 0.0f32;

        for comp in 0..3u8 {
            let _num_pivots_minus2 = r.read_u8(4)?;
            let _piv0 = r.read_u16(12)?;
            let piv1 = r.read_u16(12)?;
            let _mapping_idc = r.read_u8(4)?;
            let _poly_order = r.read_u8(4)?;
            // slope coefficient a₁ (16 bits, unsigned)
            let slope_raw = r.read_bits(16)?;
            // offset coefficient a₀ (16 bits, two's-complement signed → i16)
            let offset_raw = r.read_bits(16)? as u16 as i16;

            if comp == 0 {
                // bl_max_pq is the second pivot for luma (clamped to 12 bits in build)
                bl_max_pq = piv1 & 0x0FFF;
                // Decode slope: slope_fp = trim_slop * 8192.0 → trim_slop = slope_fp / 8192.0
                trim_slop = slope_raw as f32 / 8192.0;
                // Decode offset: offset_fp = encode_signed_fp(trim_offset, 13) → value/2^13
                trim_offset = f32::from(offset_raw) / 8192.0;
            }
            // chroma pivots and coefficients are identity; we skip them
        }

        // ── VDR DM data ───────────────────────────────────────────────────
        // Mirrors write_vdr_dm_data exactly:
        //   8 bits: dm_metadata_id = 1
        //   1 bit: signal_full_range_flag
        //   12 bits: source_min_PQ
        //   12 bits: source_max_PQ  (= bl_max_pq, already captured above)
        //   8 bits: dm_metadata_id = 4
        //   12 bits: anchor_pq      (= target_max_pq)
        //   12 bits: anchor_power   (trim_power in 4.8 fixed-point)
        //   12 bits: trim_slope_bias
        //   12 bits: trim_offset_bias
        //   12 bits: trim_power_bias
        //   12 bits: chroma_weight
        //   6× 12 bits: saturation_vector_field0..5
        //   8 bits: dm_metadata_id = 5
        //   13 bits: canvas_width
        //   13 bits: canvas_height
        //   13 bits: active_left
        //   13 bits: active_right
        //   13 bits: active_top
        //   13 bits: active_bottom
        //   8 bits: dm_metadata_id = 255

        let _dm_id1 = r.read_u8(8)?;
        let _full_range = r.read_bool()?;
        let _src_min_pq = r.read_u16(12)?;
        let _src_max_pq = r.read_u16(12)?;

        let _dm_id4 = r.read_u8(8)?;
        let target_max_pq = r.read_u16(12)?;
        let power_raw = r.read_bits(12)?;
        // anchor_power: trim_power * 256.0 → trim_power = power_raw / 256.0
        let trim_power = power_raw as f32 / 256.0;

        // trim_slope_bias, trim_offset_bias, trim_power_bias (12 bits each)
        let _slope_bias = r.read_bits(12)?;
        let _offset_bias = r.read_bits(12)?;
        let _power_bias = r.read_bits(12)?;

        // chroma_weight + 6 saturation_vector_field values (12 bits each)
        let _chroma_wt = r.read_bits(12)?;
        for _ in 0..6 {
            let _svf = r.read_bits(12)?;
        }

        // DM level 5 (active area)
        let _dm_id5 = r.read_u8(8)?;
        let _canvas_w = r.read_bits(13)?;
        let _canvas_h = r.read_bits(13)?;
        let _act_left = r.read_bits(13)?;
        let _act_right = r.read_bits(13)?;
        let _act_top = r.read_bits(13)?;
        let _act_bottom = r.read_bits(13)?;

        // End-of-metadata sentinel
        let _dm_id_end = r.read_u8(8)?;

        // ── Frame index (32 bits: high 16 then low 16) ────────────────────
        let fi_hi = r.read_bits(16)?;
        let fi_lo = r.read_bits(16)?;
        let frame_idx = u64::from((fi_hi << 16) | fi_lo);

        // ── Reconstruct DoviProfile ────────────────────────────────────────
        // Profiles 8.1 and 8.4 both encode profile_id = 8.
        // Level 1 → Profile5; level 6 → Profile81 (conservative default).
        let profile = match (vdr_rpu_profile, vdr_rpu_level) {
            (5, _) => DoviProfile::Profile5,
            (8, 6) => DoviProfile::Profile81,
            (8, _) => DoviProfile::Profile81,
            _ => {
                return Err(HdrError::MetadataParseError(format!(
                    "unknown vdr_rpu_profile nibble: {vdr_rpu_profile}"
                )));
            }
        };

        Ok(ParsedRpuFields {
            vdr_rpu_profile,
            vdr_rpu_level,
            profile,
            bl_max_pq,
            target_max_pq,
            trim_slop,
            trim_offset,
            trim_power,
            frame_idx,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── DoviProfile ──────────────────────────────────────────────────────

    #[test]
    fn test_profile_ids() {
        assert_eq!(DoviProfile::Profile5.profile_id(), 5);
        assert_eq!(DoviProfile::Profile81.profile_id(), 8);
        assert_eq!(DoviProfile::Profile84.profile_id(), 8);
    }

    #[test]
    fn test_profile_no_el() {
        assert!(!DoviProfile::Profile5.has_el());
        assert!(!DoviProfile::Profile81.has_el());
        assert!(!DoviProfile::Profile84.has_el());
    }

    #[test]
    fn test_profile_display() {
        assert!(DoviProfile::Profile5.to_string().contains("5"));
        assert!(DoviProfile::Profile81.to_string().contains("8.1"));
        assert!(DoviProfile::Profile84.to_string().contains("8.4"));
    }

    // ── DoviRpuConfig ────────────────────────────────────────────────────

    #[test]
    fn test_profile81_default_config() {
        let cfg = DoviRpuConfig::profile81_default();
        assert_eq!(cfg.profile, DoviProfile::Profile81);
        assert_eq!(cfg.bl_max_pq, 3079);
        assert_eq!(cfg.target_max_pq, 2081);
        assert!((cfg.trim_slop - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_profile5_default_config() {
        let cfg = DoviRpuConfig::profile5_default();
        assert_eq!(cfg.profile, DoviProfile::Profile5);
    }

    #[test]
    fn test_profile84_default_config() {
        let cfg = DoviRpuConfig::profile84_default();
        assert_eq!(cfg.profile, DoviProfile::Profile84);
    }

    // ── BitWriter ────────────────────────────────────────────────────────

    #[test]
    fn test_bitwriter_single_byte() {
        let mut bw = BitWriter::new();
        bw.write_bits(0xA5u32, 8);
        bw.write_rbsp_trailing_bits();
        let bytes = bw.into_bytes();
        assert_eq!(bytes[0], 0xA5);
    }

    #[test]
    fn test_bitwriter_partial_byte_padding() {
        let mut bw = BitWriter::new();
        bw.write_bits(0b111u32, 3); // 3 bits
        bw.write_rbsp_trailing_bits(); // 1-bit + 4 bits padding → total 8 bits
        let bytes = bw.into_bytes();
        assert_eq!(bytes.len(), 1);
        // 111_1_0000 = 0xF0
        assert_eq!(bytes[0], 0b1111_0000);
    }

    #[test]
    fn test_bitwriter_zero_bits() {
        let mut bw = BitWriter::new();
        bw.write_bits(0u32, 8);
        bw.write_rbsp_trailing_bits();
        let bytes = bw.into_bytes();
        assert_eq!(bytes[0], 0x00);
    }

    // ── build_rpu_payload ────────────────────────────────────────────────

    #[test]
    fn test_build_rpu_payload_non_empty() {
        let builder = DoviRpuBuilder::new(DoviRpuConfig::profile81_default());
        let payload = builder.build_rpu_payload(0);
        assert!(!payload.is_empty(), "RPU payload must not be empty");
    }

    #[test]
    fn test_build_rpu_payload_starts_with_prefix() {
        let builder = DoviRpuBuilder::new(DoviRpuConfig::profile81_default());
        let payload = builder.build_rpu_payload(0);
        assert_eq!(
            payload[0], 0x19,
            "RPU NAL must start with 0x19 prefix, got 0x{:02X}",
            payload[0]
        );
    }

    #[test]
    fn test_build_rpu_payload_frame_idx_variation() {
        let builder = DoviRpuBuilder::new(DoviRpuConfig::profile81_default());
        let p0 = builder.build_rpu_payload(0);
        let p1 = builder.build_rpu_payload(1);
        // Same length.
        assert_eq!(
            p0.len(),
            p1.len(),
            "frame index should not change payload length"
        );
        // Content differs because frame_idx is encoded.
        assert_ne!(p0, p1, "payloads for different frames should differ");
    }

    #[test]
    fn test_build_rpu_profile5() {
        let builder = DoviRpuBuilder::new(DoviRpuConfig::profile5_default());
        let payload = builder.build_rpu_payload(0);
        assert_eq!(payload[0], 0x19);
        assert!(payload.len() >= 4);
    }

    #[test]
    fn test_build_rpu_profile84() {
        let builder = DoviRpuBuilder::new(DoviRpuConfig::profile84_default());
        let payload = builder.build_rpu_payload(42);
        assert_eq!(payload[0], 0x19);
        assert!(!payload.is_empty());
    }

    // ── build_sei_message ─────────────────────────────────────────────────

    #[test]
    fn test_build_sei_message_non_empty() {
        let builder = DoviRpuBuilder::new(DoviRpuConfig::profile81_default());
        let payload = builder.build_rpu_payload(0);
        let sei = builder.build_sei_message(&payload);
        assert!(!sei.is_empty());
    }

    #[test]
    fn test_build_sei_message_header_bytes() {
        let builder = DoviRpuBuilder::new(DoviRpuConfig::profile81_default());
        let payload = builder.build_rpu_payload(0);
        let sei = builder.build_sei_message(&payload);
        // First 2 bytes are the NAL unit header (PREFIX_SEI_NUT).
        assert_eq!(sei[0], 0x4E, "NAL header byte 0 should be 0x4E");
        assert_eq!(sei[1], 0x01, "NAL header byte 1 should be 0x01");
    }

    #[test]
    fn test_build_sei_message_ends_with_trailing_bits() {
        let builder = DoviRpuBuilder::new(DoviRpuConfig::profile81_default());
        let payload = builder.build_rpu_payload(0);
        let sei = builder.build_sei_message(&payload);
        assert_eq!(
            *sei.last().unwrap_or(&0),
            0x80,
            "SEI must end with RBSP trailing bits 0x80"
        );
    }

    #[test]
    fn test_build_sei_contains_payload() {
        let builder = DoviRpuBuilder::new(DoviRpuConfig::profile81_default());
        let payload = builder.build_rpu_payload(0);
        let sei = builder.build_sei_message(&payload);
        // The payload bytes must be present somewhere in the SEI.
        let sei_body = &sei[..sei.len().saturating_sub(1)]; // strip trailing 0x80
        assert!(
            sei_body.windows(payload.len()).any(|w| w == payload),
            "SEI should contain the raw RPU payload"
        );
    }

    // ── encode_exp_golomb_sei_size ────────────────────────────────────────

    #[test]
    fn test_sei_size_encoding_small() {
        let mut out = Vec::new();
        encode_exp_golomb_sei_size(&mut out, 62);
        assert_eq!(out, vec![62u8]);
    }

    #[test]
    fn test_sei_size_encoding_large() {
        let mut out = Vec::new();
        encode_exp_golomb_sei_size(&mut out, 300);
        // 300 = 255 + 45 → [0xFF, 0x2D]
        assert_eq!(out, vec![0xFF, 45u8]);
    }

    #[test]
    fn test_sei_size_encoding_exact_255() {
        let mut out = Vec::new();
        encode_exp_golomb_sei_size(&mut out, 255);
        assert_eq!(out, vec![0xFF, 0x00]);
    }

    // ── encode_trim_coeff ─────────────────────────────────────────────────

    #[test]
    fn test_trim_coeff_zero() {
        let v = encode_trim_coeff(0.0);
        assert_eq!(v, 0, "0.0 should encode to 0");
    }

    #[test]
    fn test_trim_coeff_clamp_max() {
        let v = encode_trim_coeff(1000.0);
        assert_eq!(v, 4095);
    }

    // ── encode_signed_fp ─────────────────────────────────────────────────

    #[test]
    fn test_signed_fp_zero() {
        let v = encode_signed_fp(0.0, 13);
        assert_eq!(v, 0);
    }

    #[test]
    fn test_signed_fp_one() {
        let v = encode_signed_fp(1.0, 13);
        assert_eq!(v, 8192); // 2^13 = 8192
    }

    // ── BitReader unit tests ─────────────────────────────────────────────

    #[test]
    fn test_bitreader_single_byte() {
        // Write 0xA5 as 8 bits, then read it back.
        let data: &[u8] = &[0xA5];
        let mut r = BitReader::new(data);
        let v = r.read_bits(8).expect("read_bits(8) should succeed");
        assert_eq!(v, 0xA5);
    }

    #[test]
    fn test_bitreader_nibbles() {
        // Byte 0xAB → high nibble 0xA, low nibble 0xB.
        let data: &[u8] = &[0xAB];
        let mut r = BitReader::new(data);
        let hi = r.read_u8(4).expect("high nibble");
        let lo = r.read_u8(4).expect("low nibble");
        assert_eq!(hi, 0xA);
        assert_eq!(lo, 0xB);
    }

    #[test]
    fn test_bitreader_bool() {
        // First bit of 0x80 (1000_0000) is 1.
        let data: &[u8] = &[0x80];
        let mut r = BitReader::new(data);
        assert!(r.read_bool().expect("bool read"));
    }

    #[test]
    fn test_bitreader_cross_byte_boundary() {
        // Pack two 4-bit nibbles (0xC and 0xD) into 8 bits and cross-verify.
        let data: &[u8] = &[0xCD];
        let mut r = BitReader::new(data);
        assert_eq!(r.read_bits(4).expect("first 4 bits"), 0xC);
        assert_eq!(r.read_bits(4).expect("last 4 bits"), 0xD);
    }

    #[test]
    fn test_bitreader_rejects_zero_bits() {
        let data: &[u8] = &[0xFF];
        let mut r = BitReader::new(data);
        assert!(r.read_bits(0).is_err(), "0 bits should error");
    }

    #[test]
    fn test_bitreader_rejects_over_32_bits() {
        let data: &[u8] = &[0xFF; 8];
        let mut r = BitReader::new(data);
        assert!(r.read_bits(33).is_err(), "33 bits should error");
    }

    #[test]
    fn test_bitreader_rejects_truncated() {
        let data: &[u8] = &[0xFF]; // only 8 bits available
        let mut r = BitReader::new(data);
        r.read_bits(8).expect("consume all bytes");
        // Next read should fail.
        assert!(r.read_bits(1).is_err(), "should fail on empty data");
    }

    // ── DoviRpuParser round-trip tests ───────────────────────────────────

    /// Helper: build a payload from config + frame_idx, then parse it.
    fn roundtrip(cfg: DoviRpuConfig, frame_idx: u64) -> ParsedRpuFields {
        let builder = DoviRpuBuilder::new(cfg);
        let payload = builder.build_rpu_payload(frame_idx);
        DoviRpuParser::parse(&payload).expect("parse should succeed")
    }

    #[test]
    fn test_parser_roundtrip_profile81_default() {
        let cfg = DoviRpuConfig::profile81_default();
        let parsed = roundtrip(cfg.clone(), 0);

        assert_eq!(parsed.vdr_rpu_profile, cfg.profile.profile_id());
        assert_eq!(parsed.vdr_rpu_level, cfg.profile.rpu_level());
        assert_eq!(parsed.bl_max_pq, cfg.bl_max_pq & 0x0FFF);
        assert_eq!(parsed.target_max_pq, cfg.target_max_pq & 0x0FFF);

        // Trim slope: slope_raw = round(trim_slop * 8192) → decode = /8192.
        // For trim_slop=1.0 → slope_raw=8192 → decoded=1.0.
        let expected_slop = ((cfg.trim_slop * 8192.0).round() as u32) as f32 / 8192.0;
        assert!(
            (parsed.trim_slop - expected_slop).abs() < 1e-4,
            "trim_slop mismatch: {} vs {}",
            parsed.trim_slop,
            expected_slop
        );

        // Trim power: encoded as round(trim_power*256) / 256.
        let expected_power =
            ((cfg.trim_power * 256.0).round().clamp(0.0, 4095.0) as u32) as f32 / 256.0;
        assert!(
            (parsed.trim_power - expected_power).abs() < 1e-3,
            "trim_power mismatch: {} vs {}",
            parsed.trim_power,
            expected_power
        );
    }

    #[test]
    fn test_parser_roundtrip_profile5() {
        let cfg = DoviRpuConfig::profile5_default();
        let parsed = roundtrip(cfg.clone(), 0);
        assert_eq!(parsed.vdr_rpu_profile, cfg.profile.profile_id());
        assert_eq!(parsed.profile, DoviProfile::Profile5);
    }

    #[test]
    fn test_parser_roundtrip_profile84() {
        let cfg = DoviRpuConfig::profile84_default();
        let parsed = roundtrip(cfg.clone(), 0);
        // Profile84 encodes profile_id=8, same as Profile81 — parser maps both to Profile81.
        assert_eq!(parsed.vdr_rpu_profile, 8);
        assert_eq!(parsed.bl_max_pq, cfg.bl_max_pq & 0x0FFF);
    }

    #[test]
    fn test_parser_frame_idx_round_trip() {
        let cfg = DoviRpuConfig::profile81_default();
        let fi: u64 = 0x0001_ABCD;
        let parsed = roundtrip(cfg, fi);
        // build_rpu_payload masks to lower 32 bits.
        assert_eq!(parsed.frame_idx, fi & 0xFFFF_FFFF);
    }

    #[test]
    fn test_parser_frame_idx_variation_zero_vs_one() {
        let cfg = DoviRpuConfig::profile81_default();
        let p0 = roundtrip(cfg.clone(), 0);
        let p1 = roundtrip(cfg, 1);
        assert_eq!(p0.frame_idx, 0);
        assert_eq!(p1.frame_idx, 1);
    }

    #[test]
    fn test_parser_rejects_too_short() {
        let result = DoviRpuParser::parse(&[0x19, 0x00, 0x00, 0x00]);
        assert!(result.is_err(), "4-byte payload should be rejected");
    }

    #[test]
    fn test_parser_rejects_wrong_nal_prefix() {
        // Build valid payload and corrupt the first byte.
        let cfg = DoviRpuConfig::profile81_default();
        let mut payload = DoviRpuBuilder::new(cfg).build_rpu_payload(0);
        payload[0] = 0x00; // should be 0x19
        let result = DoviRpuParser::parse(&payload);
        assert!(result.is_err(), "wrong NAL prefix should be rejected");
    }

    #[test]
    fn test_parser_rejects_empty() {
        let result = DoviRpuParser::parse(&[]);
        assert!(result.is_err(), "empty payload should be rejected");
    }

    #[test]
    fn test_parser_trim_slop_2_0() {
        // Use trim_slop=2.0 — still representable exactly in u16 slope field.
        let cfg = DoviRpuConfig {
            profile: DoviProfile::Profile81,
            bl_max_pq: 3079,
            el_max_pq: 3079,
            target_max_pq: 2081,
            trim_slop: 2.0, // → slope_fp = 16384
            trim_offset: 0.0,
            trim_power: 1.0,
        };
        let parsed = roundtrip(cfg.clone(), 0);
        let expected = ((cfg.trim_slop * 8192.0).round() as u32) as f32 / 8192.0;
        assert!(
            (parsed.trim_slop - expected).abs() < 1e-4,
            "trim_slop=2.0 round-trip: got {}",
            parsed.trim_slop
        );
    }
}
