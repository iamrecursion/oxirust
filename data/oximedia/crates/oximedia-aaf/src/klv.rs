//! KLV (Key-Length-Value) encoder/decoder for AAF Local Sets.
//!
//! AAF property streams are serialised as a sequence of KLV triplets where:
//! - **Key** is a 16-byte AUID (SMPTE Universal Label / SMPTE ST 379-1).
//! - **Length** is a BER (Basic Encoding Rules) length prefix.
//! - **Value** is raw bytes — possibly a nested Local Set.
//!
//! BER length encoding:
//! - Short form (length < 128): one byte `0xxxxxxx`.
//! - Long form: first byte `1xxxxxxx` with `xxxxxxx` = number of length
//!   bytes (1..=126), followed by that many big-endian bytes.
//!
//! This module implements **only** the framing layer. The semantics of which
//! AUIDs map to which property values live in `local_set_encode` /
//! `local_set_decode`.

use crate::dictionary::Auid;
use crate::{AafError, Result};

/// Encode a BER length prefix into `out`.
///
/// Lengths below 128 use the short form (1 byte).  Lengths above use the
/// long form: `0x80 | n` followed by `n` big-endian bytes.  Up to 8 length
/// bytes are emitted, supporting values up to `u64::MAX`.
pub fn encode_ber_length(len: u64, out: &mut Vec<u8>) {
    if len < 128 {
        out.push(len as u8);
        return;
    }
    // Choose the smallest number of bytes that represent `len` big-endian.
    let mut needed = 0u8;
    let mut tmp = len;
    while tmp > 0 {
        tmp >>= 8;
        needed += 1;
    }
    debug_assert!((1..=8).contains(&needed));
    out.push(0x80 | needed);
    let be = len.to_be_bytes();
    out.extend_from_slice(&be[8 - needed as usize..]);
}

/// Decode a BER length prefix, returning `(length, bytes_consumed)`.
///
/// # Errors
/// Returns [`AafError::ParseError`] if the buffer is too short or the
/// declared long-form length cannot fit in `u64`.
pub fn decode_ber_length(buf: &[u8]) -> Result<(u64, usize)> {
    let first = *buf
        .first()
        .ok_or_else(|| AafError::ParseError("BER length: empty buffer".into()))?;
    if first < 0x80 {
        return Ok((u64::from(first), 1));
    }
    let n = (first & 0x7F) as usize;
    if n == 0 || n > 8 {
        return Err(AafError::ParseError(format!(
            "BER length: unsupported indicator 0x{first:02X}"
        )));
    }
    if buf.len() < 1 + n {
        return Err(AafError::ParseError("BER length: truncated".into()));
    }
    let mut value: u64 = 0;
    for &b in &buf[1..1 + n] {
        value = (value << 8) | u64::from(b);
    }
    Ok((value, 1 + n))
}

/// Encode one KLV triple (16-byte key + BER length + value bytes).
pub fn encode_klv(key: &Auid, value: &[u8], out: &mut Vec<u8>) {
    out.extend_from_slice(key.as_bytes());
    encode_ber_length(value.len() as u64, out);
    out.extend_from_slice(value);
}

/// Decode one KLV triple, returning `(key, value, total_bytes_consumed)`.
///
/// # Errors
/// Returns [`AafError::ParseError`] if the buffer is too short.
pub fn decode_klv(buf: &[u8]) -> Result<(Auid, &[u8], usize)> {
    if buf.len() < 16 {
        return Err(AafError::ParseError("KLV: missing key".into()));
    }
    let mut key_bytes = [0u8; 16];
    key_bytes.copy_from_slice(&buf[..16]);
    let key = Auid::from_bytes(&key_bytes);
    let (len, len_bytes) = decode_ber_length(&buf[16..])?;
    let header_size = 16 + len_bytes;
    let total = header_size
        .checked_add(len as usize)
        .ok_or_else(|| AafError::ParseError("KLV: length overflow".into()))?;
    if buf.len() < total {
        return Err(AafError::ParseError(format!(
            "KLV: truncated value (need {total}, have {})",
            buf.len()
        )));
    }
    Ok((key, &buf[header_size..total], total))
}

/// Encode a Local Set: a sequence of `(AUID, value)` pairs into one byte
/// buffer.  The result has no outer wrapping — the caller decides whether
/// to wrap it in another KLV envelope or store as-is in a CFB stream.
pub fn encode_local_set(entries: &[(Auid, Vec<u8>)]) -> Vec<u8> {
    // Pre-size: 16 + up to 9 length bytes + value, per entry.
    let mut out = Vec::with_capacity(entries.iter().map(|(_, v)| 16 + 9 + v.len()).sum::<usize>());
    for (key, value) in entries {
        encode_klv(key, value, &mut out);
    }
    out
}

/// Decode all KLV triples in `buf` into `(AUID, Vec<u8>)` pairs.
///
/// Stops when the buffer is consumed.  Trailing zero bytes are accepted as
/// padding — AAF allows zero-padding to align local-set streams to a sector.
///
/// # Errors
/// Returns [`AafError::ParseError`] if any KLV record is malformed.
pub fn decode_local_set(buf: &[u8]) -> Result<Vec<(Auid, Vec<u8>)>> {
    let mut entries = Vec::new();
    let mut cursor = 0;
    while cursor < buf.len() {
        // Skip null-AUID padding entries (16 zero bytes followed by 0x00 length).
        if buf[cursor..].iter().take(17).all(|&b| b == 0) {
            break;
        }
        let (key, value, consumed) = decode_klv(&buf[cursor..])?;
        entries.push((key, value.to_vec()));
        cursor += consumed;
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ber_short_form_boundary() {
        let mut out = Vec::new();
        encode_ber_length(0, &mut out);
        assert_eq!(out, vec![0x00]);
        out.clear();
        encode_ber_length(127, &mut out);
        assert_eq!(out, vec![0x7F]);
    }

    #[test]
    fn ber_long_form_boundary() {
        let mut out = Vec::new();
        encode_ber_length(128, &mut out);
        assert_eq!(out, vec![0x81, 0x80]);
        out.clear();
        encode_ber_length(255, &mut out);
        assert_eq!(out, vec![0x81, 0xFF]);
        out.clear();
        encode_ber_length(256, &mut out);
        assert_eq!(out, vec![0x82, 0x01, 0x00]);
        out.clear();
        encode_ber_length(16_383, &mut out);
        assert_eq!(out, vec![0x82, 0x3F, 0xFF]);
        out.clear();
        encode_ber_length(16_384, &mut out);
        assert_eq!(out, vec![0x82, 0x40, 0x00]);
        out.clear();
        encode_ber_length(0xFFFF_FFFF, &mut out);
        assert_eq!(out, vec![0x84, 0xFF, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn ber_round_trip_random_lengths() {
        for n in [
            0u64,
            1,
            127,
            128,
            255,
            256,
            1_000,
            65_535,
            65_536,
            1_000_000,
            u32::MAX as u64,
            u64::MAX,
        ] {
            let mut out = Vec::new();
            encode_ber_length(n, &mut out);
            let (decoded, consumed) = decode_ber_length(&out).expect("decode");
            assert_eq!(decoded, n, "round-trip {n}");
            assert_eq!(consumed, out.len(), "consumed {n}");
        }
    }

    #[test]
    fn klv_round_trip() {
        let key = Auid::CLASS_COMPOSITION_MOB;
        let value = b"hello".to_vec();
        let mut buf = Vec::new();
        encode_klv(&key, &value, &mut buf);
        let (k2, v2, consumed) = decode_klv(&buf).expect("decode");
        assert_eq!(k2, key);
        assert_eq!(v2, value.as_slice());
        assert_eq!(consumed, buf.len());
    }

    #[test]
    fn klv_round_trip_long_value() {
        let key = Auid::CLASS_SEQUENCE;
        let value: Vec<u8> = (0..2000u32).map(|i| (i & 0xFF) as u8).collect();
        let mut buf = Vec::new();
        encode_klv(&key, &value, &mut buf);
        let (k2, v2, consumed) = decode_klv(&buf).expect("decode");
        assert_eq!(k2, key);
        assert_eq!(v2.len(), value.len());
        assert_eq!(v2, value.as_slice());
        assert_eq!(consumed, buf.len());
    }

    #[test]
    fn local_set_round_trip_multiple_entries() {
        let entries = vec![
            (Auid::CLASS_HEADER, vec![1u8, 2, 3]),
            (Auid::CLASS_COMPOSITION_MOB, vec![]),
            (
                Auid::CLASS_SEQUENCE,
                (0..200u32).map(|i| i as u8).collect::<Vec<u8>>(),
            ),
        ];
        let buf = encode_local_set(&entries);
        let decoded = decode_local_set(&buf).expect("decode");
        assert_eq!(decoded.len(), entries.len());
        for ((ek, ev), (dk, dv)) in entries.iter().zip(decoded.iter()) {
            assert_eq!(ek, dk);
            assert_eq!(ev, dv);
        }
    }

    #[test]
    fn local_set_skips_padding() {
        let mut buf = encode_local_set(&[(Auid::CLASS_HEADER, b"abc".to_vec())]);
        // Append some padding zeros — must be accepted as terminator.
        buf.extend(vec![0u8; 50]);
        let decoded = decode_local_set(&buf).expect("decode");
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].1, b"abc".to_vec());
    }

    #[test]
    fn ber_truncated_long_form_returns_error() {
        // Says 4 length bytes but only 2 follow.
        let buf = [0x84u8, 0x01, 0x02];
        assert!(decode_ber_length(&buf).is_err());
    }

    #[test]
    fn klv_missing_key_returns_error() {
        let buf = [0u8; 10];
        assert!(decode_klv(&buf).is_err());
    }
}
