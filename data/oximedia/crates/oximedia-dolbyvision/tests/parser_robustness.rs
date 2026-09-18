//! Regression guard for parser robustness against malformed RPU bitstreams.
//!
//! The Dolby Vision RPU parser must never panic on arbitrary input: every read
//! is `io::Result`-propagated and all loop counts are bounded (<=15 metadata
//! blocks), so a malformed or truncated buffer should always yield a `Result`
//! rather than aborting the process. These tests feed deterministic random,
//! tiny, truncated, and length-swept buffers through the public passthrough
//! [`DolbyVisionRpu::parse_from_bitstream`] (which forwards identical bytes to
//! the private `parser::parse_rpu_bitstream` with no NAL/SEI pre-processing).
//!
//! Randomness uses an inline linear-congruential generator so the suite runs
//! under stable `cargo nextest` with no `rand`/`proptest`/cargo-fuzz dependency,
//! and reproduces bit-for-bit from a fixed seed.

use oximedia_dolbyvision::DolbyVisionRpu;

/// Minimal deterministic linear-congruential generator (LCG).
///
/// Uses the constants from Knuth's MMIX (`6364136223846793005`,
/// `1442695040888963407`) and takes the high bits, which have the best
/// statistical quality for an LCG.
struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// 10,000 deterministic random buffers (length 0..512) must each yield a
/// `Result` without panicking.
#[test]
fn fuzz_random_buffers_never_panic() {
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
    for _ in 0..10_000 {
        let len = (rng.next_u32() % 512) as usize;
        let mut buf = Vec::with_capacity(len);
        for _ in 0..len {
            buf.push(rng.next_u32() as u8);
        }
        let res = DolbyVisionRpu::parse_from_bitstream(&buf);
        assert!(matches!(res, Ok(_) | Err(_)));
    }
}

/// Empty and very small buffers, including all-`0x00` and all-`0xFF` runs of
/// length 0..=8, must each return a `Result` without panicking.
#[test]
fn fuzz_empty_and_tiny_buffers() {
    let fixed: &[&[u8]] = &[&[], &[0x00], &[0xFF], &[0x00, 0x00], &[0xFF; 512]];
    for buf in fixed {
        let res = DolbyVisionRpu::parse_from_bitstream(buf);
        assert!(matches!(res, Ok(_) | Err(_)));
    }

    for len in 0..=8usize {
        let zeros = vec![0x00u8; len];
        let res = DolbyVisionRpu::parse_from_bitstream(&zeros);
        assert!(matches!(res, Ok(_) | Err(_)));

        let ffs = vec![0xFFu8; len];
        let res = DolbyVisionRpu::parse_from_bitstream(&ffs);
        assert!(matches!(res, Ok(_) | Err(_)));
    }
}

/// Hand-crafted buffers that begin a header then truncate mid-field must each
/// return a `Result` without panicking.
#[test]
fn fuzz_crafted_truncated_headers() {
    let crafted: &[&[u8]] = &[
        &[0x00],
        &[0x00, 0x00],
        &[0x00, 0x00, 0x00],
        &[0x00, 0x00, 0x00, 0x00],
        &[0x19, 0x00],
        &[0x19, 0x00, 0x00, 0x00],
        &[0x7F, 0xFF, 0xFF],
        &[0x01, 0x02, 0x03, 0x04, 0x05],
    ];
    for buf in crafted {
        let res = DolbyVisionRpu::parse_from_bitstream(buf);
        assert!(matches!(res, Ok(_) | Err(_)));
    }
}

/// For every length 0..512, an all-`0x00` and an all-`0xFF` buffer must parse
/// without panicking.
#[test]
fn fuzz_length_sweep_all_zero_and_all_ff() {
    for len in 0..512usize {
        let zeros = vec![0x00u8; len];
        let res = DolbyVisionRpu::parse_from_bitstream(&zeros);
        assert!(matches!(res, Ok(_) | Err(_)));

        let ffs = vec![0xFFu8; len];
        let res = DolbyVisionRpu::parse_from_bitstream(&ffs);
        assert!(matches!(res, Ok(_) | Err(_)));
    }
}
