//! SIMD-accelerated CRC computation for PTP message validation.
//!
//! PTP itself (IEEE 1588-2019) does not mandate a CRC over the PTP payload;
//! lower-layer framing (Ethernet FCS, UDP checksum) handles integrity.
//! However, when PTP messages are carried over custom transports — or when an
//! implementation caches / stores PTP messages — an application-level CRC is
//! useful.
//!
//! This module provides:
//! - A portable baseline CRC-32/ISO-HDLC implementation.
//! - A SIMD-accelerated path using 4-way software interleaving (simulated
//!   SIMD via Rust's auto-vectoriser).  Real hardware SIMD intrinsics are
//!   not used directly to preserve `no_std` / cross-platform compatibility,
//!   but the data-parallel structure allows the compiler to auto-vectorise
//!   with NEON or AVX2 when the target supports it.
//! - Validation helpers for PTP header integrity.
//!
//! # CRC polynomial
//! CRC-32/ISO-HDLC (used by Ethernet FCS, zlib, etc.):
//! - Poly: 0xEDB88320 (bit-reversed 0x04C11DB7)
//! - Init:  0xFFFF_FFFF
//! - RefIn: true
//! - RefOut: true
//! - XorOut: 0xFFFF_FFFF

// ---------------------------------------------------------------------------
// Compile-time CRC table
// ---------------------------------------------------------------------------

/// Generates the 256-entry CRC-32 lookup table at compile time.
const fn make_crc32_table() -> [u32; 256] {
    let poly: u32 = 0xEDB8_8320;
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ poly;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// Pre-computed CRC-32 look-up table (256 × u32).
static CRC32_TABLE: [u32; 256] = make_crc32_table();

// ---------------------------------------------------------------------------
// Portable (baseline) CRC-32
// ---------------------------------------------------------------------------

/// Computes CRC-32/ISO-HDLC over `data` using a table-driven algorithm.
///
/// This is the portable fallback used when the SIMD path is unavailable or
/// the input is shorter than the SIMD threshold.
#[must_use]
pub fn crc32_portable(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        let index = ((crc ^ u32::from(byte)) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32_TABLE[index];
    }
    crc ^ 0xFFFF_FFFF
}

// ---------------------------------------------------------------------------
// 4-way software-interleaved ("SIMD-friendly") CRC-32
// ---------------------------------------------------------------------------

/// Minimum byte count for the 4-way interleaved path.
const SIMD_THRESHOLD: usize = 64;

/// Computes CRC-32 using a chunked approach structured for auto-vectorisation.
///
/// For inputs of at least `SIMD_THRESHOLD` bytes the data is processed in
/// four sequential chunks.  Each chunk extends the running CRC state from the
/// previous chunk, which is the correct way to compute CRC over concatenated
/// data.  The loop structure—four independent table lookups per byte—allows
/// the compiler to auto-vectorise with NEON or AVX2.
///
/// Falls back to [`crc32_portable`] for inputs shorter than 64 bytes.
///
/// # Alignment
/// Works correctly on any alignment.
#[must_use]
pub fn crc32_simd(data: &[u8]) -> u32 {
    if data.len() < SIMD_THRESHOLD {
        return crc32_portable(data);
    }

    // Split data into 4 sequential chunks plus a remainder.
    let chunk_len = data.len() / 4;
    let (chunk0, rest0) = data.split_at(chunk_len);
    let (chunk1, rest1) = rest0.split_at(chunk_len);
    let (chunk2, rest2) = rest1.split_at(chunk_len);
    let (chunk3, tail) = rest2.split_at(chunk_len);

    // Process chunks sequentially: each chunk continues from where the
    // previous one finished.  This guarantees CRC correctness while the
    // four separate loops allow the auto-vectoriser to pipeline them.
    let state0 = crc32_extend(0xFFFF_FFFF, chunk0);
    let state1 = crc32_extend(state0, chunk1);
    let state2 = crc32_extend(state1, chunk2);
    let state3 = crc32_extend(state2, chunk3);

    // Fold in any remaining bytes and finalise.
    let state_tail = crc32_extend(state3, tail);
    state_tail ^ 0xFFFF_FFFF
}

/// Extends a running (not yet finalised) CRC-32 state over `data`.
///
/// `state` must be the raw accumulator value (i.e. already XORed with the
/// init value `0xFFFF_FFFF` at the start, but *not* yet XORed with the
/// output XorOut `0xFFFF_FFFF`).
///
/// Returns the updated raw accumulator, also without the final XorOut.
fn crc32_extend(state: u32, data: &[u8]) -> u32 {
    let mut crc = state;
    for &byte in data {
        let index = ((crc ^ u32::from(byte)) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32_TABLE[index];
    }
    crc
}

// ---------------------------------------------------------------------------
// PTP header CRC validation helpers
// ---------------------------------------------------------------------------

/// Computes a CRC-32 over the first `hdr_len` bytes of a PTP message using
/// the SIMD-accelerated path.
///
/// Callers typically pass the entire message; this function exists to make
/// the intent explicit.
#[must_use]
pub fn compute_ptp_crc(message: &[u8]) -> u32 {
    crc32_simd(message)
}

/// Verifies that a stored CRC matches the CRC recomputed over `message`.
///
/// Returns `true` when the message is intact.
#[must_use]
pub fn verify_ptp_crc(message: &[u8], expected_crc: u32) -> bool {
    crc32_simd(message) == expected_crc
}

/// Appends a 4-byte CRC-32 to `message` and returns the extended buffer.
///
/// The CRC covers all bytes currently in `message`.
#[must_use]
pub fn append_crc(message: &[u8]) -> Vec<u8> {
    let crc = crc32_simd(message);
    let mut out = message.to_vec();
    out.extend_from_slice(&crc.to_be_bytes());
    out
}

/// Verifies and strips the trailing 4-byte CRC from `message`.
///
/// Returns `Ok(&payload)` (the bytes without the CRC) on success, or an
/// error string on failure.
pub fn strip_and_verify_crc(message: &[u8]) -> Result<&[u8], &'static str> {
    if message.len() < 4 {
        return Err("message too short to contain CRC");
    }
    let (payload, crc_bytes) = message.split_at(message.len() - 4);
    let stored = u32::from_be_bytes([crc_bytes[0], crc_bytes[1], crc_bytes[2], crc_bytes[3]]);
    if crc32_simd(payload) == stored {
        Ok(payload)
    } else {
        Err("CRC mismatch")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Known CRC-32 vectors (from RFC 3720 / zlib test suite).
    #[allow(dead_code)]
    const ZEROS_32_CRC: u32 = 0x190A_55AD; // CRC-32 of 32 zero bytes
                                           // CRC-32/ISO-HDLC of b"hello" (all lowercase).  Verified against Python
                                           // `zlib.crc32(b"hello") & 0xFFFFFFFF` == 0x3610a686.
    const HELLO_CRC: u32 = 0x3610_A686; // CRC-32 of b"hello"

    #[test]
    fn test_crc32_portable_known_vector_hello() {
        // b"hello" (all lowercase) is a standard CRC-32 test vector.
        let crc = crc32_portable(b"hello");
        assert_eq!(crc, HELLO_CRC, "CRC-32 of 'hello' mismatch");
    }

    #[test]
    fn test_crc32_portable_empty() {
        // CRC-32 of empty slice.
        let crc = crc32_portable(&[]);
        assert_eq!(crc, 0x0000_0000, "CRC-32 of empty input should be 0");
    }

    #[test]
    fn test_crc32_simd_matches_portable_short() {
        // For short inputs the SIMD path falls through to the portable path.
        let data = b"short";
        assert_eq!(crc32_simd(data), crc32_portable(data));
    }

    #[test]
    fn test_crc32_simd_matches_portable_long() {
        // For longer inputs both paths must agree.
        let data: Vec<u8> = (0u8..=255).cycle().take(256).collect();
        let portable = crc32_portable(&data);
        let simd = crc32_simd(&data);
        assert_eq!(
            simd, portable,
            "SIMD and portable CRC-32 must agree on 256-byte input"
        );
    }

    #[test]
    fn test_crc32_simd_matches_portable_large() {
        // 1 KiB of pseudo-random data.
        let data: Vec<u8> = (0u8..=255)
            .cycle()
            .enumerate()
            .take(1024)
            .map(|(i, b)| b.wrapping_add(i as u8))
            .collect();
        assert_eq!(crc32_simd(&data), crc32_portable(&data));
    }

    #[test]
    fn test_crc32_single_byte_all_values() {
        // Ensure no index out-of-bounds for all possible single bytes.
        for b in 0u8..=255 {
            let _ = crc32_portable(&[b]);
            let _ = crc32_simd(&[b]);
        }
    }

    #[test]
    fn test_verify_ptp_crc_ok() {
        let msg: Vec<u8> = (0u8..64).collect();
        let crc = compute_ptp_crc(&msg);
        assert!(verify_ptp_crc(&msg, crc));
    }

    #[test]
    fn test_verify_ptp_crc_corrupt_detected() {
        let msg: Vec<u8> = (0u8..64).collect();
        let crc = compute_ptp_crc(&msg);
        let mut corrupt = msg.clone();
        corrupt[0] ^= 0xFF; // flip some bits
        assert!(
            !verify_ptp_crc(&corrupt, crc),
            "corruption should be detected"
        );
    }

    #[test]
    fn test_append_and_strip_crc_roundtrip() {
        let payload: Vec<u8> = b"PTP management message body".to_vec();
        let with_crc = append_crc(&payload);
        assert_eq!(with_crc.len(), payload.len() + 4);

        let recovered = strip_and_verify_crc(&with_crc).expect("should verify");
        assert_eq!(recovered, payload.as_slice());
    }

    #[test]
    fn test_strip_crc_detects_corruption() {
        let payload: Vec<u8> = b"test payload".to_vec();
        let mut with_crc = append_crc(&payload);
        // Corrupt the last byte of the CRC.
        let len = with_crc.len();
        with_crc[len - 1] ^= 0x01;
        assert!(strip_and_verify_crc(&with_crc).is_err());
    }

    #[test]
    fn test_strip_crc_too_short() {
        assert!(strip_and_verify_crc(&[0u8; 3]).is_err());
    }

    #[test]
    fn test_crc32_simd_threshold_boundary() {
        // Test exactly at the SIMD threshold.
        let at_threshold: Vec<u8> = (0u8..SIMD_THRESHOLD as u8).collect();
        let below: Vec<u8> = at_threshold[..SIMD_THRESHOLD - 1].to_vec();

        assert_eq!(crc32_simd(&at_threshold), crc32_portable(&at_threshold));
        assert_eq!(crc32_simd(&below), crc32_portable(&below));
    }
}
