//! Determinism / bit-identity tests for the parallel-encoding seams.
//!
//! The AV1 tile encoder is a *placeholder* RLE compressor (full AV1 codec
//! integration is out of scope here), so these tests assert **determinism**
//! and **framing**, not codec correctness:
//!
//! * Tile count (thread count) must not change the produced bytes.
//! * `assemble_av1_tile_bitstream` is order-independent (sorts by index).
//! * Running the same `TranscodeContext` under different rayon pool sizes
//!   yields identical `PassStats` and identical concatenated encoder output.

mod common;

use std::sync::{Arc, Mutex};

use common::{captured_payloads, make_yuv420, ChecksumEncoder, MockDecoder};

use oximedia_transcode::{
    assemble_av1_tile_bitstream, Av1TileConfig, Av1TileParallelEncoder, FilterGraph, Frame,
    TranscodeContext,
};

/// Builds an RGBA gradient frame whose bytes vary deterministically with
/// position, so the placeholder RLE compressor produces non-trivial output.
fn make_rgba_gradient(width: u32, height: u32) -> Vec<u8> {
    let mut data = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            // R varies horizontally, G vertically, B diagonally; A opaque.
            data.push((x % 256) as u8);
            data.push((y % 256) as u8);
            data.push(((x + y) % 256) as u8);
            data.push(255);
        }
    }
    data
}

/// Builds a synthetic audio frame with interleaved i16 PCM samples (2 ch).
fn make_audio(n_samples: usize, sample_val: i16, pts_ms: i64) -> Frame {
    let mut data = Vec::with_capacity(n_samples * 4);
    for _ in 0..n_samples {
        data.extend_from_slice(&sample_val.to_le_bytes()); // left
        data.extend_from_slice(&sample_val.to_le_bytes()); // right
    }
    Frame::audio(data, pts_ms)
}

// ── Test 3: tile thread-count does not change encoded bytes ────────────────────

/// Encoding one 512×512 RGBA gradient frame with a 2×2 tile grid must produce
/// byte-identical output whether 4 worker threads or 1 are configured.  The
/// per-tile work is pure (RLE over extracted tiles) and assembly is sorted by
/// tile index, so parallelism must not perturb the bitstream.
#[test]
fn test_av1_tile_encode_byte_identical_across_thread_counts() {
    let gradient = make_rgba_gradient(512, 512);

    let cfg_multi = Av1TileConfig::new(1, 1, 4).expect("4-thread 2×2 tile config");
    let cfg_single = Av1TileConfig::new(1, 1, 1).expect("1-thread 2×2 tile config");

    let mut enc_multi =
        Av1TileParallelEncoder::new(cfg_multi, 512, 512).expect("multi-thread encoder");
    let mut enc_single =
        Av1TileParallelEncoder::new(cfg_single, 512, 512).expect("single-thread encoder");

    let bs_multi = enc_multi
        .encode_frame_rgba(&gradient)
        .expect("multi-thread encode");
    let bs_single = enc_single
        .encode_frame_rgba(&gradient)
        .expect("single-thread encode");

    assert!(!bs_multi.is_empty(), "bitstream must be non-empty");
    assert_eq!(
        bs_multi, bs_single,
        "AV1 tile bitstream must be byte-identical regardless of thread count"
    );

    // Both encoders must report the same tile count (4 = 2×2).
    assert_eq!(enc_multi.stats().tiles_encoded, 4);
    assert_eq!(enc_single.stats().tiles_encoded, 4);
    assert_eq!(
        enc_multi.stats().compressed_bytes,
        enc_single.stats().compressed_bytes,
        "compressed byte totals must match across thread counts"
    );

    // Header sanity: first u32 LE is the tile count.
    let tile_count = u32::from_le_bytes([bs_multi[0], bs_multi[1], bs_multi[2], bs_multi[3]]);
    assert_eq!(tile_count, 4, "framing header must encode 4 tiles");
}

// ── Test 4: assemble_av1_tile_bitstream is order-independent ───────────────────

/// `assemble_av1_tile_bitstream` sorts tiles by index, so feeding the same
/// `(index, data)` set in any permutation must yield the identical byte
/// stream.  We check an out-of-order input against a shuffled permutation of
/// the same tuples.
#[test]
fn test_assemble_tile_bitstream_order_independent() {
    let a = vec![0xA1u8, 0xA2, 0xA3];
    let b = vec![0xB1u8, 0xB2];
    let c = vec![0xC1u8, 0xC2, 0xC3, 0xC4];

    // Out-of-order: (2, a), (0, b), (1, c).
    let order1 = vec![
        (2usize, a.clone()),
        (0usize, b.clone()),
        (1usize, c.clone()),
    ];
    // A different permutation of the very same tuples.
    let order2 = vec![
        (0usize, b.clone()),
        (1usize, c.clone()),
        (2usize, a.clone()),
    ];
    // A third permutation for good measure.
    let order3 = vec![
        (1usize, c.clone()),
        (2usize, a.clone()),
        (0usize, b.clone()),
    ];

    let bs1 = assemble_av1_tile_bitstream(order1);
    let bs2 = assemble_av1_tile_bitstream(order2);
    let bs3 = assemble_av1_tile_bitstream(order3);

    assert_eq!(
        bs1, bs2,
        "assembly must be permutation-independent (1 vs 2)"
    );
    assert_eq!(
        bs1, bs3,
        "assembly must be permutation-independent (1 vs 3)"
    );

    // Verify the framing: count header == 3, then entries in index order 0,1,2.
    let count = u32::from_le_bytes([bs1[0], bs1[1], bs1[2], bs1[3]]);
    assert_eq!(count, 3, "tile count header must be 3");

    // First entry index must be 0 (sorted), and its length must match `b`.
    let idx0 = u32::from_le_bytes([bs1[4], bs1[5], bs1[6], bs1[7]]);
    let len0 = u32::from_le_bytes([bs1[8], bs1[9], bs1[10], bs1[11]]);
    assert_eq!(idx0, 0, "first assembled entry must be index 0");
    assert_eq!(len0 as usize, b.len(), "entry 0 length must match tile b");
    assert_eq!(&bs1[12..12 + b.len()], &b[..], "entry 0 payload must be b");
}

// ── Test 5: TranscodeContext is pool-size-invariant (stats + output bytes) ─────

/// Running the *same* frame sequence through `TranscodeContext` inside a
/// 4-thread rayon pool vs a 1-thread pool must yield identical `PassStats`
/// and identical concatenated encoder output.  `TranscodeContext::execute` is
/// itself sequential, but wrapping it in `pool.install` confirms the seam is
/// not sensitive to the ambient rayon pool size.
#[test]
fn test_transcode_context_pool_size_invariant() {
    fn build_frames() -> Vec<Frame> {
        let mut frames: Vec<Frame> = Vec::new();
        for i in 0..10u8 {
            frames.push(make_yuv420(8, 8, 20 + i * 5, i64::from(i) * 40));
        }
        // Interleave a few audio frames so audio_frames is also exercised.
        for i in 0..4i16 {
            frames.push(make_audio(64, 800 + i, i64::from(i) * 21));
        }
        frames
    }

    fn run_in_pool(threads: usize) -> (oximedia_transcode::PassStats, Vec<u8>) {
        let captured = Arc::new(Mutex::new(Vec::<Vec<u8>>::new()));
        let decoder = Box::new(MockDecoder::with_frames(build_frames()));
        let encoder = Box::new(ChecksumEncoder::new(Arc::clone(&captured)));
        let mut ctx = TranscodeContext::new(decoder, FilterGraph::new(), encoder);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("rayon pool should build");

        let stats = pool.install(|| ctx.execute().expect("pipeline should succeed"));

        // Concatenate captured payloads into a single output buffer.
        let mut concatenated = Vec::new();
        for payload in captured_payloads(&captured) {
            concatenated.extend_from_slice(&payload);
        }
        (stats.pass, concatenated)
    }

    let (stats_multi, out_multi) = run_in_pool(4);
    let (stats_single, out_single) = run_in_pool(1);

    // PassStats field-by-field equality (PassStats is not PartialEq).
    assert_eq!(stats_multi.input_frames, stats_single.input_frames);
    assert_eq!(stats_multi.output_frames, stats_single.output_frames);
    assert_eq!(stats_multi.input_bytes, stats_single.input_bytes);
    assert_eq!(stats_multi.output_bytes, stats_single.output_bytes);
    assert_eq!(stats_multi.video_frames, stats_single.video_frames);
    assert_eq!(stats_multi.audio_frames, stats_single.audio_frames);

    // Sanity on the counts: 10 video + 4 audio = 14 frames.
    assert_eq!(stats_multi.input_frames, 14);
    assert_eq!(stats_multi.video_frames, 10);
    assert_eq!(stats_multi.audio_frames, 4);
    assert_eq!(stats_multi.output_frames, 14);

    // Concatenated encoder output must be byte-identical across pool sizes.
    assert_eq!(
        out_multi, out_single,
        "concatenated encoder output must be identical regardless of pool size"
    );
    assert!(!out_multi.is_empty(), "output must be non-empty");
}
