//! Proves that the AV1 decoder in oximedia-codec 0.1.5 does not reconstruct
//! pixel data from a real AV1 bitstream.
//!
//! This test is the executable counterpart to the documentation demotion of
//! AV1 decode from "Stable" to "Bitstream-parsing" in `docs/codec_status.md`.
//! It is `#[ignore]` by default so it does not run in CI without a fixture
//! and does not require any binary fixture in the repository.
//!
//! # Running
//!
//! Provide a real AV1 bitstream (e.g., an `.obu`/`.ivf` payload extracted from
//! a WebM/MP4) via the `OXIMEDIA_AV1_FIXTURE` environment variable:
//!
//! ```bash
//! export OXIMEDIA_AV1_FIXTURE=/path/to/sample.av1
//! cargo test -p oximedia-codec --test av1_real_bitstream -- --ignored --nocapture
//! ```
//!
//! If the env var is not set, the test prints a skip notice and returns Ok(()).
//!
//! # What it checks
//!
//! The decoder is fed the fixture bytes via `send_packet`, then drained via
//! `receive_frame` until it stops producing frames. For each frame the test
//! computes the variance of the Y plane. If every returned frame has zero
//! Y-plane variance (all samples equal), the decoder is not actually
//! reconstructing pixel data and the test panics. When AV1 pixel
//! reconstruction is implemented, this test will pass without modification.
//!
//! Tracked by GitHub issue #9.

use std::error::Error;
use std::fs;
use std::path::PathBuf;

use oximedia_codec::{Av1Decoder, DecoderConfig, VideoDecoder};
use oximedia_core::CodecId;

/// Compute the population variance of a `u8` plane as `f64`.
///
/// Returns 0.0 for empty planes so the test's "all-zero variance" assertion
/// will still fail loudly (an empty plane is not a decoded frame either).
fn u8_plane_variance(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let n = data.len() as f64;
    let mean = data.iter().map(|&b| f64::from(b)).sum::<f64>() / n;
    let sq = data
        .iter()
        .map(|&b| {
            let d = f64::from(b) - mean;
            d * d
        })
        .sum::<f64>();
    sq / n
}

#[test]
#[ignore = "requires real AV1 bitstream fixture via OXIMEDIA_AV1_FIXTURE env var"]
fn av1_real_bitstream_produces_pixels() -> Result<(), Box<dyn Error>> {
    let Some(path) = std::env::var_os("OXIMEDIA_AV1_FIXTURE") else {
        eprintln!(
            "SKIP: OXIMEDIA_AV1_FIXTURE is not set. \
             Point it at a real AV1 bitstream to run this test."
        );
        return Ok(());
    };
    let fixture_path = PathBuf::from(path);
    let bytes = fs::read(&fixture_path)?;
    assert!(
        !bytes.is_empty(),
        "fixture file {} is empty",
        fixture_path.display()
    );

    let config = DecoderConfig {
        codec: CodecId::Av1,
        extradata: None,
        threads: 0,
        low_latency: false,
    };
    let mut decoder = Av1Decoder::new(config)?;

    decoder.send_packet(&bytes, 0)?;

    let mut frames_seen: usize = 0;
    let mut nonzero_variance_frames: usize = 0;
    let mut max_variance: f64 = 0.0;

    while let Some(frame) = decoder.receive_frame()? {
        frames_seen += 1;
        let y_plane = frame.plane(0);
        let var = u8_plane_variance(&y_plane.data);
        if var > 0.0 {
            nonzero_variance_frames += 1;
        }
        if var > max_variance {
            max_variance = var;
        }
        eprintln!(
            "frame {}: {}x{} format={:?} Y-plane variance={:.4}",
            frames_seen, frame.width, frame.height, frame.format, var
        );
    }

    assert!(
        frames_seen > 0,
        "decoder produced zero frames from fixture {}",
        fixture_path.display()
    );

    assert!(
        nonzero_variance_frames > 0,
        "decoder produced {frames_seen} frame(s) but every Y plane had zero variance \
         (max variance observed: {max_variance}). This is the AV1 pixel-reconstruction \
         gap tracked by GitHub issue #9: the bitstream parses but no pixels are \
         reconstructed. See docs/codec_status.md for the taxonomy."
    );

    Ok(())
}

#[test]
fn u8_plane_variance_sanity() {
    assert_eq!(u8_plane_variance(&[]), 0.0);
    assert_eq!(u8_plane_variance(&[128, 128, 128, 128]), 0.0);
    let var = u8_plane_variance(&[0, 255, 0, 255]);
    assert!(var > 0.0, "variance of alternating 0/255 must be positive");
}
