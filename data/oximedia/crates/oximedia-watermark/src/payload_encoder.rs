//! Payload encoding and decoding for watermark embedding.
//!
//! Provides a [`PayloadFormat`] taxonomy, an [`EncodedPayload`] wrapper, and
//! a [`PayloadEncoder`] that serialises/deserialises arbitrary byte payloads
//! into a bit-stream suitable for watermark embedding algorithms.
//!
//! Additionally provides [`ReedSolomonConfig`] for configurable Reed-Solomon
//! error-correction parameters that can be wired into the payload pipeline.

#![allow(dead_code)]

// ──────────────────────────────────────────────────────────────────────────────
// Reed-Solomon configuration
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for Reed-Solomon forward error correction over GF(2⁸).
///
/// | Parameter | Meaning |
/// |-----------|---------|
/// | `n`       | Total codeword length in symbols (data + parity).  Must be ≤ 255 for GF(2⁸). |
/// | `k`       | Number of pure data symbols. Must satisfy `k < n`. |
///
/// The number of parity symbols (and thus the error-correction capability) is
/// `n − k`.  A codeword can correct up to `⌊(n−k)/2⌋` symbol errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReedSolomonConfig {
    /// Total codeword length (data + parity symbols).  Must be ≤ 255.
    pub n: usize,
    /// Number of data symbols.  Must satisfy `k < n`.
    pub k: usize,
}

impl Default for ReedSolomonConfig {
    /// Default matches the historical hardcoded values: `n = 16`, `k = 8`.
    fn default() -> Self {
        Self { n: 16, k: 8 }
    }
}

/// Errors produced when validating or constructing a [`ReedSolomonConfig`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RsConfigError {
    /// `k` must be strictly less than `n`.
    KNotLessThanN {
        /// Supplied `n`.
        n: usize,
        /// Supplied `k`.
        k: usize,
    },
    /// `n` exceeds the GF(2⁸) symbol limit of 255.
    NExceedsGfLimit {
        /// Supplied `n`.
        n: usize,
    },
    /// `k` must be at least 1.
    KIsZero,
}

impl std::fmt::Display for RsConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KNotLessThanN { n, k } => {
                write!(f, "Reed-Solomon: k ({k}) must be < n ({n})")
            }
            Self::NExceedsGfLimit { n } => {
                write!(f, "Reed-Solomon: n ({n}) must be ≤ 255 (GF(2⁸) limit)")
            }
            Self::KIsZero => write!(f, "Reed-Solomon: k must be ≥ 1"),
        }
    }
}

impl std::error::Error for RsConfigError {}

impl ReedSolomonConfig {
    /// Validate the configuration, returning an error if any constraint is violated.
    ///
    /// Constraints:
    /// * `k ≥ 1`
    /// * `k < n`
    /// * `n ≤ 255` (GF(2⁸) symbol limit for the `reed-solomon-erasure` crate)
    ///
    /// # Errors
    ///
    /// Returns [`RsConfigError`] if any constraint is violated.
    pub fn validate(&self) -> Result<(), RsConfigError> {
        if self.k == 0 {
            return Err(RsConfigError::KIsZero);
        }
        if self.k >= self.n {
            return Err(RsConfigError::KNotLessThanN {
                n: self.n,
                k: self.k,
            });
        }
        if self.n > 255 {
            return Err(RsConfigError::NExceedsGfLimit { n: self.n });
        }
        Ok(())
    }

    /// Number of parity symbols (`n − k`).
    #[must_use]
    pub fn parity_symbols(&self) -> usize {
        self.n - self.k
    }

    /// Maximum number of correctable symbol errors (`⌊(n−k)/2⌋`).
    #[must_use]
    pub fn max_correctable_errors(&self) -> usize {
        self.parity_symbols() / 2
    }
}

/// Payload serialisation format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PayloadFormat {
    /// Raw bytes — no framing overhead, maximum capacity.
    Raw,
    /// Length-prefixed: a 2-byte big-endian length header precedes the data.
    LengthPrefixed,
    /// NUL-terminated byte string.
    NulTerminated,
    /// Base-128 variable-length integer encoding (little-endian groups of 7 bits).
    Varint,
}

impl PayloadFormat {
    /// Human-readable name for logging and diagnostics.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Raw => "Raw",
            Self::LengthPrefixed => "LengthPrefixed",
            Self::NulTerminated => "NulTerminated",
            Self::Varint => "Varint",
        }
    }

    /// Minimum overhead in bytes introduced by the format (excluding the data itself).
    #[must_use]
    pub fn overhead_bytes(self) -> usize {
        match self {
            Self::Raw => 0,
            Self::LengthPrefixed => 2,
            Self::NulTerminated => 1,
            Self::Varint => 1, // at least one varint byte for the length
        }
    }
}

/// A byte sequence that has been formatted for watermark embedding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedPayload {
    /// The serialised bytes ready for embedding.
    pub bytes: Vec<u8>,
    /// Format used to produce these bytes.
    pub format: PayloadFormat,
}

impl EncodedPayload {
    /// Number of encoded bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the encoded payload is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Number of bits in the encoded payload.
    #[must_use]
    pub fn bit_count(&self) -> usize {
        self.bytes.len() * 8
    }

    /// Extract the `n`-th bit (MSB-first within each byte).
    ///
    /// Returns `None` when `n >= bit_count()`.
    #[must_use]
    pub fn bit(&self, n: usize) -> Option<bool> {
        if n >= self.bit_count() {
            return None;
        }
        let byte_idx = n / 8;
        let bit_idx = 7 - (n % 8); // MSB first
        Some((self.bytes[byte_idx] >> bit_idx) & 1 == 1)
    }
}

/// Error type for payload encoding/decoding operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PayloadEncodeError {
    /// Data is too large for the chosen format (e.g. exceeds 16-bit length field).
    DataTooLarge(usize),
    /// Missing length header during decode.
    MissingLengthHeader,
    /// Payload data was truncated (fewer bytes than the header claimed).
    TruncatedData {
        /// Number of bytes claimed by the header.
        expected: usize,
        /// Number of bytes actually available.
        found: usize,
    },
    /// Missing NUL terminator during decode.
    MissingNulTerminator,
    /// Varint overflow — decoded length exceeded 32 bits.
    VarintOverflow,
}

impl std::fmt::Display for PayloadEncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DataTooLarge(n) => write!(f, "data too large: {n} bytes"),
            Self::MissingLengthHeader => write!(f, "missing length header"),
            Self::TruncatedData { expected, found } => {
                write!(f, "truncated: expected {expected} bytes, found {found}")
            }
            Self::MissingNulTerminator => write!(f, "missing NUL terminator"),
            Self::VarintOverflow => write!(f, "varint length overflow"),
        }
    }
}

/// Encodes and decodes byte payloads according to a chosen [`PayloadFormat`].
///
/// An optional [`ReedSolomonConfig`] can be attached to document the RS
/// parameters that will be used by the surrounding pipeline.  The encoder
/// itself does *not* run RS coding — that is handled by [`crate::payload::PayloadCodec`].
pub struct PayloadEncoder {
    /// Format used by this encoder instance.
    format: PayloadFormat,
    /// Optional Reed-Solomon configuration associated with this encoder.
    rs_config: Option<ReedSolomonConfig>,
}

impl PayloadEncoder {
    /// Create a new encoder for the given format with no associated RS config.
    #[must_use]
    pub fn new(format: PayloadFormat) -> Self {
        Self {
            format,
            rs_config: None,
        }
    }

    /// Attach a [`ReedSolomonConfig`] to this encoder, returning the updated encoder.
    ///
    /// The config is validated immediately.  Returns an error if the parameters
    /// are out of range (see [`ReedSolomonConfig::validate`]).
    ///
    /// # Errors
    ///
    /// Returns [`RsConfigError`] if the RS configuration is invalid.
    pub fn with_rs_config(mut self, config: ReedSolomonConfig) -> Result<Self, RsConfigError> {
        config.validate()?;
        self.rs_config = Some(config);
        Ok(self)
    }

    /// Return the attached Reed-Solomon configuration, if any.
    #[must_use]
    pub fn rs_config(&self) -> Option<&ReedSolomonConfig> {
        self.rs_config.as_ref()
    }

    /// Active format.
    #[must_use]
    pub fn format(&self) -> PayloadFormat {
        self.format
    }

    /// Encode `data` into an [`EncodedPayload`].
    ///
    /// # Errors
    ///
    /// Returns [`PayloadEncodeError`] if the data is too large for the format.
    pub fn encode(&self, data: &[u8]) -> Result<EncodedPayload, PayloadEncodeError> {
        let bytes = match self.format {
            PayloadFormat::Raw => data.to_vec(),
            PayloadFormat::LengthPrefixed => {
                let len = data.len();
                if len > u16::MAX as usize {
                    return Err(PayloadEncodeError::DataTooLarge(len));
                }
                #[allow(clippy::cast_possible_truncation)]
                let len_u16 = len as u16;
                let mut out = Vec::with_capacity(2 + len);
                out.extend_from_slice(&len_u16.to_be_bytes());
                out.extend_from_slice(data);
                out
            }
            PayloadFormat::NulTerminated => {
                let mut out = Vec::with_capacity(data.len() + 1);
                out.extend_from_slice(data);
                out.push(0u8);
                out
            }
            PayloadFormat::Varint => {
                let mut out = Vec::with_capacity(data.len() + 2);
                // Encode the length as a varint.
                let mut n = data.len();
                loop {
                    let mut byte = (n & 0x7F) as u8;
                    n >>= 7;
                    if n != 0 {
                        byte |= 0x80;
                    }
                    out.push(byte);
                    if n == 0 {
                        break;
                    }
                }
                out.extend_from_slice(data);
                out
            }
        };

        Ok(EncodedPayload {
            bytes,
            format: self.format,
        })
    }

    /// Decode an [`EncodedPayload`] back to the original bytes.
    ///
    /// # Errors
    ///
    /// Returns [`PayloadEncodeError`] if the encoded bytes are malformed.
    pub fn decode(&self, payload: &EncodedPayload) -> Result<Vec<u8>, PayloadEncodeError> {
        self.decode_bytes(&payload.bytes)
    }

    /// Decode raw bytes (without an [`EncodedPayload`] wrapper).
    ///
    /// # Errors
    ///
    /// Returns [`PayloadEncodeError`] if the bytes are malformed.
    pub fn decode_bytes(&self, bytes: &[u8]) -> Result<Vec<u8>, PayloadEncodeError> {
        match self.format {
            PayloadFormat::Raw => Ok(bytes.to_vec()),
            PayloadFormat::LengthPrefixed => {
                if bytes.len() < 2 {
                    return Err(PayloadEncodeError::MissingLengthHeader);
                }
                let len = u16::from_be_bytes([bytes[0], bytes[1]]) as usize;
                let data = &bytes[2..];
                if data.len() < len {
                    return Err(PayloadEncodeError::TruncatedData {
                        expected: len,
                        found: data.len(),
                    });
                }
                Ok(data[..len].to_vec())
            }
            PayloadFormat::NulTerminated => {
                if let Some(pos) = bytes.iter().position(|&b| b == 0) {
                    Ok(bytes[..pos].to_vec())
                } else {
                    Err(PayloadEncodeError::MissingNulTerminator)
                }
            }
            PayloadFormat::Varint => {
                // Decode the varint length prefix.
                let mut len: u64 = 0;
                let mut shift = 0u32;
                let mut cursor = 0;
                loop {
                    if cursor >= bytes.len() {
                        return Err(PayloadEncodeError::MissingLengthHeader);
                    }
                    let byte = bytes[cursor];
                    cursor += 1;
                    let value = u64::from(byte & 0x7F);
                    len |= value << shift;
                    shift = shift.saturating_add(7);
                    if shift > 35 {
                        return Err(PayloadEncodeError::VarintOverflow);
                    }
                    if byte & 0x80 == 0 {
                        break;
                    }
                }
                let len = len as usize;
                let data = &bytes[cursor..];
                if data.len() < len {
                    return Err(PayloadEncodeError::TruncatedData {
                        expected: len,
                        found: data.len(),
                    });
                }
                Ok(data[..len].to_vec())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- PayloadFormat ---

    #[test]
    fn test_format_names() {
        assert_eq!(PayloadFormat::Raw.name(), "Raw");
        assert_eq!(PayloadFormat::LengthPrefixed.name(), "LengthPrefixed");
        assert_eq!(PayloadFormat::NulTerminated.name(), "NulTerminated");
        assert_eq!(PayloadFormat::Varint.name(), "Varint");
    }

    #[test]
    fn test_format_overhead_bytes() {
        assert_eq!(PayloadFormat::Raw.overhead_bytes(), 0);
        assert_eq!(PayloadFormat::LengthPrefixed.overhead_bytes(), 2);
        assert_eq!(PayloadFormat::NulTerminated.overhead_bytes(), 1);
        assert_eq!(PayloadFormat::Varint.overhead_bytes(), 1);
    }

    // --- EncodedPayload helpers ---

    #[test]
    fn test_encoded_payload_bit_count() {
        let ep = EncodedPayload {
            bytes: vec![0xFF, 0x00],
            format: PayloadFormat::Raw,
        };
        assert_eq!(ep.bit_count(), 16);
    }

    #[test]
    fn test_encoded_payload_bit_msb_first() {
        let ep = EncodedPayload {
            bytes: vec![0b1010_0000],
            format: PayloadFormat::Raw,
        };
        assert_eq!(ep.bit(0), Some(true));
        assert_eq!(ep.bit(1), Some(false));
        assert_eq!(ep.bit(2), Some(true));
        assert_eq!(ep.bit(3), Some(false));
    }

    #[test]
    fn test_encoded_payload_bit_out_of_range() {
        let ep = EncodedPayload {
            bytes: vec![0xFF],
            format: PayloadFormat::Raw,
        };
        assert_eq!(ep.bit(8), None);
    }

    // --- PayloadEncoder: Raw ---

    #[test]
    fn test_raw_roundtrip() {
        let enc = PayloadEncoder::new(PayloadFormat::Raw);
        let data = b"Hello, World!";
        let encoded = enc.encode(data).expect("should succeed in test");
        let decoded = enc.decode(&encoded).expect("should succeed in test");
        assert_eq!(decoded, data);
    }

    // --- PayloadEncoder: LengthPrefixed ---

    #[test]
    fn test_length_prefixed_roundtrip() {
        let enc = PayloadEncoder::new(PayloadFormat::LengthPrefixed);
        let data = b"Copyright 2024 Acme";
        let encoded = enc.encode(data).expect("should succeed in test");
        // Check header bytes encode the data length.
        let header_len = u16::from_be_bytes([encoded.bytes[0], encoded.bytes[1]]) as usize;
        assert_eq!(header_len, data.len());
        let decoded = enc.decode(&encoded).expect("should succeed in test");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_length_prefixed_missing_header_error() {
        let enc = PayloadEncoder::new(PayloadFormat::LengthPrefixed);
        let ep = EncodedPayload {
            bytes: vec![0x00],
            format: PayloadFormat::LengthPrefixed,
        };
        assert_eq!(
            enc.decode(&ep),
            Err(PayloadEncodeError::MissingLengthHeader)
        );
    }

    #[test]
    fn test_length_prefixed_truncated_data_error() {
        let enc = PayloadEncoder::new(PayloadFormat::LengthPrefixed);
        // Header says 10 bytes but only 3 follow.
        let ep = EncodedPayload {
            bytes: vec![0x00, 0x0A, 0x01, 0x02, 0x03],
            format: PayloadFormat::LengthPrefixed,
        };
        assert!(matches!(
            enc.decode(&ep),
            Err(PayloadEncodeError::TruncatedData { .. })
        ));
    }

    // --- PayloadEncoder: NulTerminated ---

    #[test]
    fn test_nul_terminated_roundtrip() {
        let enc = PayloadEncoder::new(PayloadFormat::NulTerminated);
        let data = b"OxiMedia";
        let encoded = enc.encode(data).expect("should succeed in test");
        assert_eq!(*encoded.bytes.last().expect("should succeed in test"), 0u8);
        let decoded = enc.decode(&encoded).expect("should succeed in test");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_nul_terminated_missing_terminator() {
        let enc = PayloadEncoder::new(PayloadFormat::NulTerminated);
        let ep = EncodedPayload {
            bytes: vec![0x41, 0x42, 0x43], // "ABC" with no NUL
            format: PayloadFormat::NulTerminated,
        };
        assert_eq!(
            enc.decode(&ep),
            Err(PayloadEncodeError::MissingNulTerminator)
        );
    }

    // --- PayloadEncoder: Varint ---

    #[test]
    fn test_varint_roundtrip_small() {
        let enc = PayloadEncoder::new(PayloadFormat::Varint);
        let data = b"tiny";
        let encoded = enc.encode(data).expect("should succeed in test");
        let decoded = enc.decode(&encoded).expect("should succeed in test");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_varint_roundtrip_larger_data() {
        let enc = PayloadEncoder::new(PayloadFormat::Varint);
        let data: Vec<u8> = (0u8..200).collect();
        let encoded = enc.encode(&data).expect("should succeed in test");
        let decoded = enc.decode(&encoded).expect("should succeed in test");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_encoder_format_accessor() {
        let enc = PayloadEncoder::new(PayloadFormat::LengthPrefixed);
        assert_eq!(enc.format(), PayloadFormat::LengthPrefixed);
    }

    // ── Item 1: ReedSolomonConfig ─────────────────────────────────────────────

    #[test]
    fn test_rs_config_custom_params() {
        // n=32, k=16 → 16 parity symbols, 8 correctable errors
        let cfg = ReedSolomonConfig { n: 32, k: 16 };
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.parity_symbols(), 16);
        assert_eq!(cfg.max_correctable_errors(), 8);

        // Attach to encoder via builder
        let enc = PayloadEncoder::new(PayloadFormat::Raw)
            .with_rs_config(cfg.clone())
            .expect("config should be valid in test");
        let attached = enc.rs_config().expect("rs_config should be set in test");
        assert_eq!(attached.n, 32);
        assert_eq!(attached.k, 16);

        // Encoded payload is still plain bytes
        let data = b"Hello RS";
        let ep = enc.encode(data).expect("should succeed in test");
        let decoded = enc.decode(&ep).expect("should succeed in test");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_rs_config_invalid_params_rejected() {
        // k >= n
        let err = ReedSolomonConfig { n: 8, k: 8 }.validate();
        assert!(matches!(err, Err(RsConfigError::KNotLessThanN { .. })));

        let err = ReedSolomonConfig { n: 4, k: 7 }.validate();
        assert!(matches!(err, Err(RsConfigError::KNotLessThanN { .. })));

        // n > 255
        let err = ReedSolomonConfig { n: 256, k: 128 }.validate();
        assert!(matches!(err, Err(RsConfigError::NExceedsGfLimit { .. })));

        // k == 0
        let err = ReedSolomonConfig { n: 8, k: 0 }.validate();
        assert!(matches!(err, Err(RsConfigError::KIsZero)));

        // builder should propagate the error
        let result = PayloadEncoder::new(PayloadFormat::Raw)
            .with_rs_config(ReedSolomonConfig { n: 8, k: 8 });
        assert!(result.is_err());
    }

    #[test]
    fn test_rs_config_default() {
        let cfg = ReedSolomonConfig::default();
        assert_eq!(cfg.n, 16);
        assert_eq!(cfg.k, 8);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_rs_config_boundary_n255() {
        // n == 255 is exactly the GF(2^8) limit — should be accepted
        let cfg = ReedSolomonConfig { n: 255, k: 128 };
        assert!(cfg.validate().is_ok());
    }
}
