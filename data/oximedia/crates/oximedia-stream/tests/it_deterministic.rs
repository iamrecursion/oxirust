//! Deterministic known-answer integration tests for `oximedia-stream`.
//!
//! These tests pin exact, reproducible behaviour of three subsystems whose
//! output must be byte- or count-stable across runs:
//!
//! * **SCTE-35** (`scte35`) — `encode_splice_null` produces a fixed 16-byte
//!   section, and the encode → parse → re-encode cycle is byte-identical for a
//!   range of tier values.
//! * **HLS manifest** (`manifest_builder`) — the LL-HLS `#EXT-X-SKIP` tag line
//!   is rendered verbatim, and `build_media_playlist` auto-couples the
//!   `#EXT-X-VERSION` (9 with a skip, 3 without).
//! * **Multi-CDN weighted round-robin** (`multi_cdn`) — selection is fully
//!   deterministic (no RNG); a weight vector `[1, 2, 1]` yields a stable
//!   25/50/25 histogram over 100 picks and identical sequences across runs.
//!
//! All asserts are exact known-answer oracles; nothing here depends on timing,
//! threading, or randomness.

#![allow(clippy::expect_used)]
#![allow(clippy::unwrap_used)]

use oximedia_stream::manifest_builder::{build_media_playlist, HlsManifest, HlsSegment, HlsSkip};
use oximedia_stream::multi_cdn::{CdnProvider, MultiCdnRouter, RoutingStrategy};
use oximedia_stream::scte35::{encode_splice_null, parse_splice_info, SpliceCommand};

// ───────────────────────────────────────────────────────────────────────────
// (a) SCTE-35 splice_null known-answer + round-trip
// ───────────────────────────────────────────────────────────────────────────

/// `encode_splice_null(0)` must emit exactly the 16-byte section:
/// `[0xFC, 0xB0, 0x0F, 0x00 × 13]`.
///
/// Byte 0 = `table_id` (0xFC), byte 1 = `0xB0` (section_syntax + private +
/// reserved + length-hi nibble), byte 2 = `section_length` low byte
/// (`7 + 2 + 0 + 2 + 0 + 4 = 15 = 0x0F`), and the remaining 13 bytes — protocol
/// version, encryption byte, cw_index, tier (0), splice_command_length (0),
/// splice_command_type (`SpliceNull = 0x00`), descriptor_loop_length (`0x0000`)
/// and the 4-byte CRC32 placeholder — are all zero.
#[test]
fn scte35_encode_splice_null_zero_tier_exact_bytes() {
    let bytes = encode_splice_null(0);
    // [0xFC, 0xB0, 0x0F] followed by 13 zero bytes.
    let mut expected = vec![0x00u8; 16];
    expected[0] = 0xFC;
    expected[1] = 0xB0;
    expected[2] = 0x0F;
    assert_eq!(bytes.len(), 16, "splice_null section must be 16 bytes");
    assert_eq!(
        bytes, expected,
        "splice_null(0) must equal [0xFC,0xB0,0x0F, 0x00 x13]"
    );
}

/// The splice command type byte (offset 9) for splice_null must be `0x00`,
/// and the tier nibbles (bytes 6-7) must be zero for tier 0.
#[test]
fn scte35_splice_null_command_type_byte_is_null() {
    let bytes = encode_splice_null(0);
    assert_eq!(
        bytes[9], 0x00,
        "splice_command_type byte must be 0x00 (Null)"
    );
    assert_eq!(bytes[6], 0x00, "tier high byte zero for tier 0");
    assert_eq!(
        bytes[7], 0x00,
        "tier low nibble + cmd-len-hi zero for tier 0"
    );
}

/// `encode_splice_null` → `parse_splice_info` → re-`encode_splice_null` must be
/// byte-identical for a spread of tier values, and the parsed command must be
/// `SpliceCommand::Null` carrying no descriptors.
#[test]
fn scte35_splice_null_round_trip_byte_identical() {
    for &tier in &[0x000u16, 0x123, 0x0FFF] {
        let encoded = encode_splice_null(tier);

        let section = parse_splice_info(&encoded)
            .unwrap_or_else(|e| panic!("parse_splice_info(tier={tier:#X}) failed: {e:?}"));

        // The parsed command must be Null.
        assert!(
            matches!(section.splice_command, SpliceCommand::Null),
            "tier {tier:#X}: expected SpliceCommand::Null, got {:?}",
            section.splice_command
        );
        // splice_null carries no descriptors.
        assert!(
            section.descriptors.is_empty(),
            "tier {tier:#X}: splice_null must have no descriptors, got {}",
            section.descriptors.len()
        );
        // The 12-bit tier must round-trip exactly.
        assert_eq!(
            section.tier, tier,
            "tier {tier:#X}: tier field must round-trip"
        );
        // Not encrypted, protocol version 0.
        assert!(
            !section.encrypted_packet,
            "tier {tier:#X}: must not be encrypted"
        );
        assert_eq!(
            section.protocol_version, 0,
            "tier {tier:#X}: protocol_version 0"
        );

        // Re-encode from the same tier; the encoder is deterministic, so bytes
        // must match the original encoding exactly.
        let re_encoded = encode_splice_null(section.tier);
        assert_eq!(
            encoded, re_encoded,
            "tier {tier:#X}: re-encoded splice_null must be byte-identical"
        );
    }
}

/// Distinct tiers must produce distinct byte sequences (the tier is encoded in
/// bytes 6-7), confirming the tier is not silently dropped.
#[test]
fn scte35_splice_null_distinct_tiers_distinct_bytes() {
    let a = encode_splice_null(0x000);
    let b = encode_splice_null(0x123);
    let c = encode_splice_null(0x0FFF);
    assert_ne!(a, b, "tier 0x000 and 0x123 must differ");
    assert_ne!(b, c, "tier 0x123 and 0x0FFF must differ");
    assert_ne!(a, c, "tier 0x000 and 0x0FFF must differ");
    // All three are the same fixed length regardless of tier.
    assert_eq!(a.len(), 16);
    assert_eq!(b.len(), 16);
    assert_eq!(c.len(), 16);
}

// ───────────────────────────────────────────────────────────────────────────
// (b) HLS manifest: EXT-X-SKIP rendering + version auto-coupling
// ───────────────────────────────────────────────────────────────────────────

/// `HlsSkip` with no recently-removed date-ranges renders exactly
/// `#EXT-X-SKIP:SKIPPED-SEGMENTS=N`.
#[test]
fn hls_skip_tag_line_exact() {
    let skip = HlsSkip {
        skipped_segments: 5,
        recently_removed_dateranges: Vec::new(),
    };
    assert_eq!(skip.to_tag_line(), "#EXT-X-SKIP:SKIPPED-SEGMENTS=5");
}

/// A different skipped-segment count renders the matching integer.
#[test]
fn hls_skip_tag_line_other_count() {
    let skip = HlsSkip {
        skipped_segments: 42,
        recently_removed_dateranges: Vec::new(),
    };
    assert_eq!(skip.to_tag_line(), "#EXT-X-SKIP:SKIPPED-SEGMENTS=42");
}

fn make_segment(duration: f64, n: u64) -> HlsSegment {
    HlsSegment {
        duration,
        uri: format!("seg{n:04}.ts"),
        byte_range: None,
        discontinuity: false,
        program_date_time: None,
        date_range: None,
    }
}

/// A media playlist built WITH a skip must declare `#EXT-X-VERSION:9` and emit
/// an `#EXT-X-SKIP:` line — the auto-coupling behaviour at the top of
/// `build_media_playlist`.
#[test]
fn hls_media_playlist_with_skip_is_version_9() {
    let manifest = HlsManifest {
        target_duration: 6,
        media_sequence: 10,
        segments: (10..13).map(|i| make_segment(6.0, i)).collect(),
        is_endlist: false,
        allow_cache: false,
        skip: Some(HlsSkip {
            skipped_segments: 9,
            recently_removed_dateranges: Vec::new(),
        }),
    };
    let out = build_media_playlist(&manifest);

    assert!(
        out.contains("#EXT-X-VERSION:9"),
        "delta playlist with skip must be version 9; got:\n{out}"
    );
    assert!(
        out.contains("#EXT-X-SKIP:SKIPPED-SEGMENTS=9"),
        "delta playlist must contain the skip line; got:\n{out}"
    );
    // It must NOT be version 3.
    assert!(
        !out.contains("#EXT-X-VERSION:3"),
        "delta playlist must not also be version 3; got:\n{out}"
    );
}

/// A media playlist built WITHOUT a skip must declare `#EXT-X-VERSION:3` and
/// emit no `#EXT-X-SKIP:` line.
#[test]
fn hls_media_playlist_without_skip_is_version_3() {
    let manifest = HlsManifest {
        target_duration: 6,
        media_sequence: 0,
        segments: (0..5).map(|i| make_segment(6.0, i)).collect(),
        is_endlist: false,
        allow_cache: false,
        skip: None,
    };
    let out = build_media_playlist(&manifest);

    assert!(
        out.contains("#EXT-X-VERSION:3"),
        "non-delta playlist must be version 3; got:\n{out}"
    );
    assert!(
        !out.contains("#EXT-X-SKIP"),
        "non-delta playlist must not contain a skip line; got:\n{out}"
    );
    assert!(
        !out.contains("#EXT-X-VERSION:9"),
        "non-delta playlist must not be version 9; got:\n{out}"
    );
}

/// The skip line emitted inside the playlist matches the standalone
/// `to_tag_line()` rendering (the playlist embeds the exact tag line).
#[test]
fn hls_media_playlist_skip_line_matches_tag_line() {
    let skip = HlsSkip {
        skipped_segments: 7,
        recently_removed_dateranges: Vec::new(),
    };
    let tag_line = skip.to_tag_line();
    let manifest = HlsManifest {
        target_duration: 4,
        media_sequence: 7,
        segments: (7..9).map(|i| make_segment(4.0, i)).collect(),
        is_endlist: false,
        allow_cache: false,
        skip: Some(skip),
    };
    let out = build_media_playlist(&manifest);
    assert!(
        out.contains(&tag_line),
        "playlist must embed the exact skip tag line `{tag_line}`; got:\n{out}"
    );
}

// ───────────────────────────────────────────────────────────────────────────
// (c) Multi-CDN weighted round-robin: deterministic histogram
// ───────────────────────────────────────────────────────────────────────────

/// Build a 3-provider router with the given weights, ids `A`, `B`, `C`.
fn make_weighted_router(weights: [u32; 3]) -> MultiCdnRouter {
    let providers = vec![
        CdnProvider::new("A", "https://a.cdn.example", 0, weights[0], 100),
        CdnProvider::new("B", "https://b.cdn.example", 1, weights[1], 100),
        CdnProvider::new("C", "https://c.cdn.example", 2, weights[2], 100),
    ];
    MultiCdnRouter::new(providers)
}

/// Pick `n` providers via the weighted round-robin strategy and return their
/// names in order.
fn wrr_sequence(router: &mut MultiCdnRouter, n: usize) -> Vec<String> {
    (0..n)
        .map(|_| {
            router
                .select_provider(&RoutingStrategy::WeightedRoundRobin)
                .expect("a provider must be available")
                .name
                .clone()
        })
        .collect()
}

/// With weights `[1, 2, 1]`, 100 weighted-round-robin picks must produce the
/// exact histogram A=25, B=50, C=25 (one full period is 4 picks; 100 / 4 = 25
/// complete periods, so the counts are exact, not approximate).
#[test]
fn multi_cdn_wrr_histogram_is_exact_25_50_25() {
    let mut router = make_weighted_router([1, 2, 1]);
    let seq = wrr_sequence(&mut router, 100);

    let count_a = seq.iter().filter(|n| n.as_str() == "A").count();
    let count_b = seq.iter().filter(|n| n.as_str() == "B").count();
    let count_c = seq.iter().filter(|n| n.as_str() == "C").count();

    assert_eq!(count_a, 25, "A (weight 1) must win 25/100; seq={seq:?}");
    assert_eq!(count_b, 50, "B (weight 2) must win 50/100; seq={seq:?}");
    assert_eq!(count_c, 25, "C (weight 1) must win 25/100; seq={seq:?}");
    // Sanity: every pick is one of the three providers.
    assert_eq!(count_a + count_b + count_c, 100);
}

/// The weighted round-robin selection is fully deterministic: two independently
/// constructed routers with identical setup must emit the identical 100-pick
/// sequence (no RNG, no global state).
#[test]
fn multi_cdn_wrr_is_deterministic_across_runs() {
    let mut r1 = make_weighted_router([1, 2, 1]);
    let mut r2 = make_weighted_router([1, 2, 1]);

    let seq1 = wrr_sequence(&mut r1, 100);
    let seq2 = wrr_sequence(&mut r2, 100);

    assert_eq!(
        seq1, seq2,
        "identical setup must yield identical WRR sequences"
    );
}

/// The weighted round-robin pattern is periodic with period == total weight
/// (here 4): picks `[k]` and `[k + 4]` must be the same provider for every `k`.
#[test]
fn multi_cdn_wrr_pattern_is_period_four() {
    let mut router = make_weighted_router([1, 2, 1]);
    let seq = wrr_sequence(&mut router, 100);

    for k in 0..(seq.len() - 4) {
        assert_eq!(
            seq[k],
            seq[k + 4],
            "WRR pattern must repeat with period 4: index {k} vs {}",
            k + 4
        );
    }

    // Within any single period of 4, B (weight 2) appears exactly twice while
    // A and C (weight 1) appear exactly once each.
    let first_period = &seq[0..4];
    assert_eq!(
        first_period.iter().filter(|n| n.as_str() == "B").count(),
        2,
        "B must appear twice per period; period={first_period:?}"
    );
    assert_eq!(
        first_period.iter().filter(|n| n.as_str() == "A").count(),
        1,
        "A must appear once per period; period={first_period:?}"
    );
    assert_eq!(
        first_period.iter().filter(|n| n.as_str() == "C").count(),
        1,
        "C must appear once per period; period={first_period:?}"
    );
}

/// Equal weights `[1, 1, 1]` over 99 picks must distribute exactly evenly
/// (33 each), confirming the cycle is a pure round-robin when weights match.
#[test]
fn multi_cdn_wrr_equal_weights_even_split() {
    let mut router = make_weighted_router([1, 1, 1]);
    let seq = wrr_sequence(&mut router, 99);

    for name in ["A", "B", "C"] {
        let count = seq.iter().filter(|n| n.as_str() == name).count();
        assert_eq!(count, 33, "{name} must win exactly 33/99; seq={seq:?}");
    }
}
