//! Container probing example — honest demonstration of what OxiMedia decoders can do today.
//!
//! This example used to pretend to show a full AV1/VP9 decoding pipeline. That
//! was dishonest: OxiMedia 0.1.5 parses AV1/VP9/VP8/Theora/Vorbis/AVIF bitstreams
//! but does not reconstruct pixel or sample data end-to-end for those codecs.
//! See [`docs/codec_status.md`](../docs/codec_status.md) and GitHub issue #9 for
//! the full, per-decoder status.
//!
//! What this example does:
//!
//! - Probes a synthetic WebM (Matroska) header and reports the detected format
//! - Lists which decoders in `oximedia-codec` and `oximedia-audio` actually
//!   produce decoded samples today and which only parse the bitstream
//!
//! # Usage
//!
//! ```bash
//! cargo run --example decode_video
//! ```

use oximedia::prelude::*;

/// Probe a synthetic WebM (Matroska) header and print the detected format.
fn probe_synthetic_webm() -> OxiResult<()> {
    println!("Step 1: Container format detection");
    println!("----------------------------------\n");

    // Minimal EBML / Matroska signature (header only — not a full file).
    let webm_header: [u8; 16] = [
        0x1A, 0x45, 0xDF, 0xA3, // EBML magic
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F, 0x42, 0x86, 0x81, 0x01,
    ];

    let probe_result = probe_format(&webm_header)?;
    println!("  Detected format: {:?}", probe_result.format);
    println!("  Confidence:      {:.1}%", probe_result.confidence * 100.0);
    println!();

    Ok(())
}

/// Print the honest decoder-status matrix for codecs advertised by `oximedia-codec`
/// and `oximedia-audio`. The column meanings are documented in
/// [`docs/codec_status.md`](../docs/codec_status.md) and in the top-level README.
fn print_decoder_status_matrix() {
    println!("Step 2: Decoder status matrix (oximedia-codec + oximedia-audio)");
    println!("---------------------------------------------------------------\n");

    println!("  Legend:");
    println!("    Verified          end-to-end decode matches a reference on external fixtures");
    println!("    Functional        real reconstruction path present and self-consistent");
    println!("    Bitstream-parsing headers/syntax parsed; no pixel/sample reconstruction");
    println!("    Experimental      API sketch; not intended to decode");
    println!();

    println!("  Video:");
    println!("    AV1      Bitstream-parsing");
    println!("    VP9      Bitstream-parsing");
    println!("    VP8      Bitstream-parsing");
    println!(
        "    Theora   Bitstream-parsing (decode hand-off fixed in 0.1.7; \
         encoder↔decoder bitstream alignment outstanding — see docs/codec_status.md)"
    );
    println!("    AVIF     Bitstream-parsing (depends on AV1)");
    println!("    MJPEG    Functional");
    println!("    FFV1     Functional");
    println!("    JPEG-XL  Functional");
    println!("    H.263    Functional");
    println!("    APV      Functional");
    println!("    PNG/APNG Functional");
    println!("    GIF      Functional");
    println!("    WebP     Functional (VP8L lossless only; no VP8 lossy decoder)");
    println!();

    println!("  Audio:");
    println!("    Vorbis   Bitstream-parsing");
    println!("    Opus     Functional (CELT only; SILK/hybrid are placeholder)");
    println!("    MP3      Functional (oximedia-audio, patents expired 2017)");
    println!("    FLAC     Functional / Verified (CRC-16 round-trip)");
    println!("    PCM      Verified");
    println!();
}

/// Print container formats that can be muxed and demuxed today.
fn print_container_support() {
    println!("Step 3: Container formats");
    println!("-------------------------\n");
    println!("  WebM       Web-optimised Matroska (demux + mux)");
    println!("  Matroska   Flexible open container (.mkv)");
    println!("  MP4        ISOBMFF");
    println!("  Ogg        Xiph.org container");
    println!("  MPEG-TS    Broadcast transport stream");
    println!("  AVI        MJPEG-only, ≤1 GB RIFF (see oximedia-avi)");
    println!();
}

fn main() -> OxiResult<()> {
    println!("OxiMedia decoder status example (0.1.5)");
    println!("=======================================\n");

    probe_synthetic_webm()?;
    print_decoder_status_matrix();
    print_container_support();

    println!("Notes");
    println!("-----\n");
    println!("  Codecs marked Bitstream-parsing parse valid streams but return empty,");
    println!("  constant, or raw-passthrough buffers. They are useful for format");
    println!("  inspection, metadata, and container work — not for playback.");
    println!();
    println!("  See docs/codec_status.md for the full per-decoder status, what is");
    println!("  missing, and the effort required to close each gap. GitHub issue #9");
    println!("  tracks the AV1 pixel-reconstruction gap specifically.");
    println!();

    Ok(())
}
