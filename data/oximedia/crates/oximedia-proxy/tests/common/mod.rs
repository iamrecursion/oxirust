//! Shared test fixtures for `oximedia-proxy` integration tests.
//!
//! Builds real, byte-accurate Matroska container files on disk (via the
//! production `oximedia_container::mux::MatroskaMuxer`) so integration tests
//! can exercise the real `oximedia_transcode::TranscodePipeline` demux/mux
//! path instead of faking file contents.
//!
//! All helpers write to `std::env::temp_dir()`, per project policy.

use bytes::Bytes;
use oximedia_container::{
    mux::{MatroskaMuxer, MuxerConfig},
    Muxer, Packet, PacketFlags, StreamInfo,
};
use oximedia_core::{CodecId, Rational, Timestamp};
use oximedia_io::MemorySource;
use std::path::Path;

/// Build a minimal-but-real single-video-track Matroska file at `path`.
///
/// Writes `n_packets` synthetic (non-decodable placeholder payload) video
/// packets spaced `frame_ms` milliseconds apart. Since no `video_codec`
/// override is required by callers that want stream-copy behaviour, the
/// packet payload bytes do not need to be valid VP9 — the demuxer/muxer only
/// care about container framing, not codec semantics, in stream-copy mode.
pub async fn write_synthetic_video_mkv(
    path: &Path,
    n_packets: u64,
    frame_ms: i64,
) -> std::io::Result<()> {
    let in_buf = MemorySource::new_writable(64 * 1024);
    let mut muxer = MatroskaMuxer::new(in_buf, MuxerConfig::new());

    let mut video = StreamInfo::new(0, CodecId::Vp9, Rational::new(1, 1000));
    video.codec_params.width = Some(320);
    video.codec_params.height = Some(240);
    muxer
        .add_stream(video)
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    muxer
        .write_header()
        .await
        .map_err(|e| std::io::Error::other(e.to_string()))?;

    for i in 0..n_packets {
        let data = vec![0x42u8, 0x00, (i & 0xFF) as u8, 0x01];
        let pkt = Packet::new(
            0,
            Bytes::from(data),
            Timestamp::new(i as i64 * frame_ms, Rational::new(1, 1000)),
            PacketFlags::KEYFRAME,
        );
        muxer
            .write_packet(&pkt)
            .await
            .map_err(|e| std::io::Error::other(e.to_string()))?;
    }
    muxer
        .write_trailer()
        .await
        .map_err(|e| std::io::Error::other(e.to_string()))?;

    let sink = muxer.into_sink();
    let mkv_bytes = sink.written_data().to_vec();
    tokio::fs::write(path, &mkv_bytes).await
}

/// Returns a process-unique temp subdirectory for a test, creating it.
pub fn unique_temp_dir(label: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!(
        "oximedia_proxy_{label}_{}_{nanos}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&dir);
    dir
}
