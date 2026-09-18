//! `OxiMedia` Container Layer
//!
//! Container format handling with resilient parsing for:
//! - Matroska (.mkv) / `WebM` (.webm)
//! - Ogg (.ogg, .opus, .oga)
//! - FLAC (.flac)
//! - WAV (.wav)
//! - MP4 (.mp4) - AV1/VP9 only
//! - MPEG-TS (.ts, .m2ts) - AV1/VP9/VP8/Opus/FLAC only
//! - YUV4MPEG2 (.y4m) - Raw uncompressed video
//! - FLV (.flv) - Adobe Flash Video muxer (MP3/PCM audio, H.263 video)
//!
//! # Overview
//!
//! This crate provides demuxers and muxers for media container formats.
//! A demuxer reads a container file and extracts compressed packets,
//! while a muxer combines compressed packets into a container file.
//!
//! # Key Types
//!
//! - [`ContainerFormat`] - Enumeration of supported container formats
//! - [`probe_format`] - Detect container format from magic bytes
//! - [`Packet`] - Compressed media packet with timestamps
//! - [`StreamInfo`] - Information about a stream (codec, dimensions, etc.)
//! - [`Demuxer`] - Trait for container demuxers
//! - [`Muxer`] - Trait for container muxers
//!
//! # Demuxing Example
//!
//! ```ignore
//! use oximedia_container::{probe_format, demux::MatroskaDemuxer, Demuxer};
//! use oximedia_io::FileSource;
//!
//! // Detect format from file header
//! let mut source = FileSource::open("video.mkv").await?;
//! let mut buf = [0u8; 12];
//! source.read(&mut buf).await?;
//! let format = probe_format(&buf)?;
//! println!("Format: {:?}", format.format);
//!
//! // Demux the file
//! source.seek(std::io::SeekFrom::Start(0)).await?;
//! let mut demuxer = MatroskaDemuxer::new(source);
//! demuxer.probe().await?;
//!
//! for stream in demuxer.streams() {
//!     println!("Stream {}: {:?}", stream.index, stream.codec);
//! }
//!
//! while let Ok(packet) = demuxer.read_packet().await {
//!     println!("Packet: stream={}, size={}, keyframe={}",
//!              packet.stream_index, packet.size(), packet.is_keyframe());
//! }
//! ```
//!
//! # Muxing Example
//!
//! ```ignore
//! use oximedia_container::mux::{MatroskaMuxer, Muxer, MuxerConfig};
//!
//! let config = MuxerConfig::new()
//!     .with_title("My Video");
//!
//! let mut muxer = MatroskaMuxer::new(sink, config);
//! muxer.add_stream(video_info)?;
//! muxer.add_stream(audio_info)?;
//!
//! muxer.write_header().await?;
//!
//! for packet in packets {
//!     muxer.write_packet(&packet).await?;
//! }
//!
//! muxer.write_trailer().await?;
//! ```
//!
//! # Metadata Editing Example
//!
//! ```ignore
//! use oximedia_container::metadata::MetadataEditor;
//!
//! let mut editor = MetadataEditor::open("audio.flac").await?;
//!
//! // Read tags
//! if let Some(title) = editor.get_text("TITLE") {
//!     println!("Title: {}", title);
//! }
//!
//! // Modify tags
//! editor.set("TITLE", "New Title");
//! editor.set("ARTIST", "New Artist");
//!
//! // Save changes
//! editor.save().await?;
//! ```
//!
//! # Wave 4 API Additions
//!
//! ## MP4 Fragment Mode — `Mp4FragmentMode`
//!
//! [`mux::mp4::Mp4FragmentMode`] controls how the MP4 muxer arranges sample data:
//!
//! | Variant | Description |
//! |---------|-------------|
//! | `Progressive` | Classic MP4: single `moov` + `mdat`, optimal for download |
//! | `Fragmented { fragment_duration_ms }` | ISOBMFF fragments; each fragment is a self-contained `moof`+`mdat` pair |
//!
//! `Mp4Mode` is a backward-compatible type alias for `Mp4FragmentMode`.
//!
//! ```ignore
//! use oximedia_container::mux::mp4::{Mp4Muxer, Mp4Config, Mp4FragmentMode};
//!
//! // Progressive (default)
//! let config = Mp4Config::new().with_mode(Mp4FragmentMode::Progressive);
//!
//! // Fragmented — 4-second fragments for DASH/HLS delivery
//! let config = Mp4Config::new()
//!     .with_mode(Mp4FragmentMode::Fragmented { fragment_duration_ms: 4000 });
//! ```
//!
//! ## Sample-Accurate Seek Cursor — [`DecodeSkipCursor`]
//!
//! [`DecodeSkipCursor`] is returned by the `seek_sample_accurate()` methods on the
//! Matroska, MP4, and AVI demuxers. It locates the nearest keyframe at or before a
//! target PTS and records how many decoded samples must be discarded to reach the
//! precise presentation position.
//!
//! | Field | Type | Description |
//! |-------|------|-------------|
//! | `byte_offset` | `u64` | File offset where decoding should start |
//! | `sample_index` | `usize` | 0-based index of the keyframe sample |
//! | `skip_samples` | `u32` | Samples to decode-and-discard after seeking |
//! | `target_pts` | `i64` | Requested PTS in track timescale units |
//!
//! ```ignore
//! use oximedia_container::demux::MatroskaDemuxer;
//! use oximedia_io::FileSource;
//!
//! let mut demuxer = MatroskaDemuxer::new(FileSource::open("video.mkv").await?);
//! demuxer.probe().await?;
//!
//! // Seek to exactly 30 seconds (in track timescale units)
//! let cursor = demuxer.seek_sample_accurate(2_700_000).await?;
//! println!("Start decode at byte {}, skip {} samples",
//!          cursor.byte_offset, cursor.skip_samples);
//! ```
//!
//! ## CMAF Chunked Transfer — `CmafChunkMode` / `CmafChunkedConfig`
//!
//! [`streaming::mux::CmafChunkMode`] and [`streaming::mux::CmafChunkedConfig`] implement
//! chunked CMAF delivery as defined in ISO/IEC 23000-19, enabling sub-segment delivery
//! for LL-HLS and LL-DASH workflows.
//!
//! | Mode | Description |
//! |------|-------------|
//! | `Standard` | Whole-segment delivery (default, no chunking) |
//! | `Chunked` | Each chunk is one or more complete `moof`+`mdat` pairs |
//! | `LowLatencyChunked` | Each chunk is exactly one sample (minimum latency) |
//!
//! `CmafChunkedConfig` carries additional settings: `chunk_duration_ms`,
//! `max_samples_per_chunk`, `include_mfra`, `signal_low_latency` (writes `cmfl`
//! compatible brand in the `styp` box), and `part_target_duration_ms` for LL-HLS.
//!
//! ```ignore
//! use oximedia_container::streaming::mux::{CmafChunkedConfig, CmafChunkMode};
//!
//! let config = CmafChunkedConfig::new()
//!     .with_mode(CmafChunkMode::LowLatencyChunked)
//!     .with_low_latency(true);
//! ```
//!
//! ## Matroska v4 Block Addition Mapping — `BlockAdditionMapping`
//!
//! `demux::matroska::matroska_v4::BlockAdditionMapping` represents a Matroska v4
//! `BlockAdditionMapping` element (EBML ID 0x41CB), which carries auxiliary per-block
//! data channels such as HDR10+ metadata, Dolby Vision RPU data, or depth maps.
//!
//! | Field | Description |
//! |-------|-------------|
//! | `id_name` | Human-readable channel name (e.g., `"hdr10plus"`, `"dovi_rpu"`) |
//! | `id_type` | Numeric type per the Matroska Block Addition Mapping Registry |
//! | `id_extra_data` | Codec-specific configuration payload |
//!
//! Access via `StreamInfo::block_addition_mappings` after probing a Matroska track.
//!
//! ---
//!
//! # Container Format Support Matrix
//!
//! The table below documents which video codecs, audio codecs, and subtitle formats
//! are supported (demux **D** / mux **M** / both **DM**) in each container.
//! All entries listed are patent-free or whose relevant patents have expired.
//!
//! | Container | Extensions | Video codecs | Audio codecs | Subtitles | Notes |
//! |-----------|-----------|--------------|--------------|-----------|-------|
//! | **Matroska** | .mkv .mka .mks | AV1 DM · VP9 DM · VP8 DM · Theora DM · MJPEG DM · APV DM | Opus DM · Vorbis DM · FLAC DM · PCM DM | WebVTT DM · ASS/SSA DM | Full feature set: chapters, attachments, HDR10+/Dolby Vision side-data (v4), SCTE-35 |
//! | **WebM** | .webm | AV1 DM · VP9 DM · VP8 DM | Opus DM · Vorbis DM | — | Matroska subset optimised for web delivery; no chapters or attachments |
//! | **MP4 / ISOBMFF** | .mp4 .m4a .m4v .mov | AV1 DM · VP9 DM | Opus DM | WebVTT D | Patent-free codecs only; progressive and fragmented (fMP4) modes; CMAF-compatible |
//! | **MPEG-TS** | .ts .m2ts .mts | AV1 DM · VP9 DM · VP8 DM | Opus DM · FLAC DM | DVB text D | Broadcast transport stream; PCR timing; SCTE-35 ad markers DM |
//! | **Ogg** | .ogg .opus .oga .ogv | Theora DM · VP8 D | Opus DM · Vorbis DM · FLAC DM | — | Xiph.org envelope; VP8 demux only via Skeleton |
//! | **FLAC** | .flac | — | FLAC DM | — | Native FLAC container; seek table; ID3v2 & Vorbis comment metadata |
//! | **WAV / RIFF** | .wav .wave | — | PCM DM · ADPCM DM | — | RIFF envelope; up to 4 GiB without RF64 extension |
//! | **CAF** | .caf | — | FLAC DM · PCM DM · ALAC DM | — | Apple Core Audio Format; big-endian box structure |
//! | **FLV** | .flv | H.263/Sorenson DM | MP3 DM · PCM DM | — | Adobe Flash Video; RTMP ingest compatible; ≤ 2³² byte offset |
//! | **YUV4MPEG2** | .y4m | Raw YUV DM | — | — | Uncompressed video pipe format; 4:2:0 · 4:2:2 · 4:4:4 · mono |
//! | **WebVTT** | .vtt .webvtt | — | — | WebVTT DM | Text subtitle container |
//! | **SubRip** | .srt | — | — | SRT DM | Simple text subtitle container |
//!
//! ### Querying support programmatically
//!
//! [`ContainerFormat`] exposes `supports_video()`, `supports_audio()`, and
//! `supports_subtitles()` helper methods so you can filter formats at runtime:
//!
//! ```
//! use oximedia_container::ContainerFormat;
//!
//! let video_containers: Vec<_> = [
//!     ContainerFormat::Matroska,
//!     ContainerFormat::WebM,
//!     ContainerFormat::Mp4,
//!     ContainerFormat::MpegTs,
//!     ContainerFormat::Ogg,
//!     ContainerFormat::Flac,
//!     ContainerFormat::Wav,
//!     ContainerFormat::Y4m,
//!     ContainerFormat::Flv,
//! ]
//! .into_iter()
//! .filter(|f| f.supports_video())
//! .collect();
//!
//! assert!(video_containers.contains(&ContainerFormat::Matroska));
//! assert!(video_containers.contains(&ContainerFormat::WebM));
//! assert!(!video_containers.contains(&ContainerFormat::Flac));
//! ```
//!
//! ---
//!
//! # Seeking Strategies
//!
//! `oximedia-container` exposes three complementary seeking strategies that trade
//! accuracy against CPU cost.  All three are driven by [`SeekTarget`] combined with
//! [`SeekFlags`], or through the higher-level [`SeekIndex`] / [`SeekAccuracy`] API.
//!
//! ## 1 — Keyframe-approximate seeking (fastest)
//!
//! The demuxer repositions to the nearest sync sample (keyframe) at or before the
//! requested time.  No pre-roll decoding is required; the first packet read after the
//! seek is immediately displayable.
//!
//! Typical cost: one binary search in the index (`O(log n)`), one `seek()` syscall.
//!
//! Use [`SeekTarget::time`] — the default flags include `SeekFlags::KEYFRAME |
//! SeekFlags::BACKWARD`:
//!
//! ```
//! use oximedia_container::{SeekTarget, SeekFlags};
//!
//! // Seek to 30 seconds (snaps to the nearest keyframe before t=30).
//! let target = SeekTarget::time(30.0);
//! assert!(target.flags.contains(SeekFlags::KEYFRAME));
//! ```
//!
//! Using [`SeekIndex`] directly:
//!
//! ```
//! use oximedia_container::{SeekIndex, SeekIndexEntry, SeekAccuracy};
//!
//! let mut idx = SeekIndex::new(90_000); // 90 kHz timescale
//! idx.add_entry(SeekIndexEntry::keyframe(0,     0,     0,       4096, 3000, 0));
//! idx.add_entry(SeekIndexEntry::keyframe(90_000, 90_000, 12_000, 3840, 3000, 1));
//! idx.add_entry(SeekIndexEntry::keyframe(180_000, 180_000, 24_000, 4000, 3000, 2));
//!
//! let plan = idx.plan_seek(95_000, SeekAccuracy::Keyframe).unwrap();
//! // Nearest keyframe before t=95 000 ticks is at t=90 000.
//! assert_eq!(plan.keyframe_entry.pts, 90_000);
//! assert_eq!(plan.discard_count, 0);
//! ```
//!
//! ## 2 — Sample-accurate seeking (exact, higher CPU cost)
//!
//! The demuxer locates the keyframe before the target, then instructs the decoder to
//! decode-and-discard the intermediate frames until the exact presentation timestamp is
//! reached.  [`SeekPlan::discard_count`] tells the caller how many decoded output
//! frames to drop before the first presentable output.
//!
//! Use [`SeekTarget::sample_accurate`] or [`SeekAccuracy::SampleAccurate`]:
//!
//! ```
//! use oximedia_container::{SeekIndex, SeekIndexEntry, SeekAccuracy, SeekTarget, SeekFlags};
//!
//! let mut idx = SeekIndex::new(90_000);
//! idx.add_entry(SeekIndexEntry::keyframe(     0,      0,     0, 3000, 3000, 0));
//! idx.add_entry(SeekIndexEntry::non_keyframe(3000,  3000, 12_000, 2500, 3000, 1));
//! idx.add_entry(SeekIndexEntry::non_keyframe(6000,  6000, 24_000, 2400, 3000, 2));
//! idx.add_entry(SeekIndexEntry::keyframe(   9000,  9000, 36_000, 3200, 3000, 3));
//!
//! // Seek to exactly t=6000 ticks (a non-keyframe sample).
//! let plan = idx.plan_seek(6000, SeekAccuracy::SampleAccurate).unwrap();
//! assert_eq!(plan.keyframe_entry.pts, 0);      // decode from keyframe at t=0
//! assert_eq!(plan.target_entry.pts, 6000);     // present starting at t=6000
//! // discard_count = frames strictly *between* keyframe and target (t=3000 only).
//! assert_eq!(plan.discard_count, 1);
//!
//! // SeekTarget variant (used with per-format demuxer seek methods):
//! let sa_target = SeekTarget::sample_accurate(6000.0 / 90_000.0);
//! assert!(sa_target.flags.contains(SeekFlags::FRAME_ACCURATE));
//! ```
//!
//! ## 3 — Byte-offset seeking (raw positioning)
//!
//! The demuxer repositions to an absolute byte offset in the file without consulting
//! any timestamp index.  This is useful for scripted split/join workflows where the
//! byte boundary is already known.
//!
//! Use [`SeekTarget::byte`]:
//!
//! ```
//! use oximedia_container::{SeekTarget, SeekFlags};
//!
//! // Seek to byte offset 1 048 576 (1 MiB into the file).
//! let target = SeekTarget::byte(1_048_576);
//! assert!(target.flags.contains(SeekFlags::BYTE));
//! assert!(!target.flags.contains(SeekFlags::KEYFRAME));
//! ```
//!
//! ### Tolerance-based seeking
//!
//! [`SeekAccuracy::WithinTolerance`] provides a middle ground: it first tries
//! sample-accurate positioning and falls back to keyframe positioning if the target
//! falls within the specified tolerance window (in timescale ticks).
//!
//! ```
//! use oximedia_container::{SeekIndex, SeekIndexEntry, SeekAccuracy};
//!
//! let mut idx = SeekIndex::new(1000); // 1 kHz timescale
//! idx.add_entry(SeekIndexEntry::keyframe(0, 0, 0, 1024, 100, 0));
//! idx.add_entry(SeekIndexEntry::keyframe(100, 100, 4096, 1024, 100, 1));
//!
//! // Tolerance of 50 ticks: target=80 is within 50 ticks of keyframe at t=100.
//! let plan = idx.plan_seek(80, SeekAccuracy::WithinTolerance(50));
//! // plan may be None or snap to a keyframe — implementation-defined
//! let _ = plan;
//! ```
//!
//! ---
//!
//! # Streaming Output Modes
//!
//! `oximedia-container` supports three industry-standard adaptive-streaming output
//! modes.  All three use ISOBMFF fragmented MP4 as the media container; they differ
//! in how segments are packaged and how the manifest/playlist is generated.
//!
//! | Mode | Manifest | Segment format | Latency target |
//! |------|----------|----------------|----------------|
//! | **HLS** (HTTP Live Streaming) | `.m3u8` playlist | CMAF `.m4s` + init `.mp4` | ≥ 2 × target-duration (typ. 6 s) |
//! | **DASH** (Dynamic Adaptive Streaming) | MPD XML | ISOBMFF `.m4s` segments | ≥ segment duration (typ. 4 s) |
//! | **CMAF LL** (Low-Latency CMAF) | MPD or `.m3u8` | Chunked `moof+mdat` within each segment | < 1 s (per-chunk delivery) |
//!
//! ## HLS output
//!
//! HLS delivery uses [`fragment::SegmentWriter`] to write `.mp4` init segments and
//! `.m4s` media segments to disk, then generates an HLS `.m3u8` playlist automatically
//! when `with_playlist_generation(true)` is set.
//!
//! ```no_run
//! use oximedia_container::fragment::{SegmentWriter, SegmentWriterConfig};
//!
//! # async fn example() -> oximedia_core::OxiResult<()> {
//! let config = SegmentWriterConfig::new("/var/hls/stream/")
//!     .with_filename_pattern("seg_%05d.m4s")
//!     .with_playlist_generation(true)
//!     .with_delete_old_segments(true)
//!     .with_max_segments(6);  // HLS rolling window: keep 6 segments
//!
//! let mut writer = SegmentWriter::new(config).await?;
//! // writer.write_init_segment(&init_fragment).await?;
//! // writer.write_segment(&media_fragment).await?;
//! // Playlist at /var/hls/stream/playlist.m3u8 is updated after each segment.
//! # Ok(())
//! # }
//! ```
//!
//! ## DASH output
//!
//! DASH manifests (MPDs) are generated via [`dash::emit_mpd`] /
//! [`dash::DashManifestConfig`].  Media segments are the same CMAF `.m4s` files
//! produced by the fMP4 muxer or [`fragment::SegmentWriter`].
//!
//! ```
//! use oximedia_container::dash::{
//!     emit_mpd, DashManifestConfig, DashAdaptationSet, DashRepresentation,
//!     DashSegmentTemplate,
//! };
//!
//! let config = DashManifestConfig {
//!     media_presentation_duration: "PT3600S".to_string(),
//!     min_buffer_time: "PT4S".to_string(),
//!     base_url: Some("https://cdn.example.com/stream/".to_string()),
//!     adaptation_sets: vec![
//!         DashAdaptationSet {
//!             id: 1,
//!             content_type: "video".to_string(),
//!             mime_type: "video/mp4".to_string(),
//!             codecs: "av01.0.08M.08".to_string(),
//!             representations: vec![
//!                 DashRepresentation {
//!                     id: "1080p".to_string(),
//!                     bandwidth: 4_000_000,
//!                     width: Some(1920),
//!                     height: Some(1080),
//!                     frame_rate: Some("25".to_string()),
//!                     audio_sampling_rate: None,
//!                     segment_template: DashSegmentTemplate {
//!                         timescale: 90_000,
//!                         duration: Some(360_000), // 4-second segments
//!                         initialization: "1080p/init.mp4".to_string(),
//!                         media: "1080p/seg$Number$.m4s".to_string(),
//!                         start_number: 1,
//!                         segment_timeline: None,
//!                     },
//!                 },
//!             ],
//!         },
//!     ],
//! };
//!
//! let mpd_xml = emit_mpd(&config);
//! assert!(mpd_xml.contains("<MPD"));
//! assert!(mpd_xml.contains("SegmentTemplate"));
//! assert!(mpd_xml.contains("av01.0.08M.08"));
//! ```
//!
//! ## CMAF Low-Latency Chunked output
//!
//! CMAF chunked transfer (ISO/IEC 23000-19) enables sub-second latency by delivering
//! each chunk as an independent `moof+mdat` pair within the HTTP response body, before
//! the segment boundary is reached.  The [`streaming::mux::CmafChunkedEncoder`]
//! manages chunk assembly; [`streaming::mux::CmafChunkMode`] selects the strategy.
//!
//! ```
//! use oximedia_container::streaming::mux::{
//!     CmafChunkedConfig, CmafChunkMode, CmafChunkedEncoder, ChunkSample,
//! };
//!
//! // LL-HLS / LL-DASH: one chunk per sample (minimum latency).
//! let ll_config = CmafChunkedConfig::new()
//!     .with_mode(CmafChunkMode::LowLatencyChunked)
//!     .with_low_latency_signal(true)
//!     .with_part_target_duration_ms(200);
//!
//! // Standard DASH: chunk every 500 ms, max 5 samples per chunk.
//! let dash_config = CmafChunkedConfig::new()
//!     .with_mode(CmafChunkMode::Chunked)
//!     .with_chunk_duration_ms(500)
//!     .with_max_samples_per_chunk(5);
//!
//! // Feed samples into the encoder (90 kHz timescale).
//! let mut encoder = CmafChunkedEncoder::new(ll_config, 90_000);
//!
//! // Each call to push_sample() may produce one or more chunks.
//! encoder.push_sample(ChunkSample {
//!     pts: 0,
//!     dts: 0,
//!     duration: 3000,      // 1/30 s at 90 kHz
//!     data: vec![0u8; 64], // encoded frame bytes
//!     is_keyframe: true,
//!     track_id: 1,
//! });
//!
//! assert_eq!(encoder.total_chunks_produced(), 1);
//! ```
//!
//! ### Choosing the right mode
//!
//! - **HLS (VOD)**: use `SegmentWriter` with `generate_playlist = true`; segments of
//!   6–10 seconds; plain-MP4 or CMAF `Standard` mode.
//! - **DASH (live)**: use `emit_mpd` with `type="dynamic"`; fMP4 segments of 2–4 s;
//!   CMAF `Standard` mode with `include_mfra = true` for random-access.
//! - **LL-HLS / LL-DASH**: use `CmafChunkMode::LowLatencyChunked`; set
//!   `part_target_duration_ms` ≤ 200 ms; signal `cmfl` brand via
//!   `with_low_latency_signal(true)`.

pub mod attach;
pub mod bitrate_stats;
pub mod box_header;
pub mod caf;
pub mod chapters;
pub mod chunk_map;
pub mod container_probe;
pub(crate) mod container_probe_parsers;
pub mod cue;
pub mod dash;
pub mod data;
pub mod demux;
pub mod edit;
pub mod edit_list;
mod format;
pub mod fragment;
pub mod media_header;
pub mod metadata;
pub mod mkv_cluster;
#[cfg(feature = "mmap")]
pub mod mmap_source;
pub mod multi_angle;
pub mod mux;
pub mod ogg_page;
mod packet;
pub mod preroll;
mod probe;
pub mod pts_dts;
pub mod pts_dts_batch;
pub mod riff;
pub mod sample_entry;
pub mod sample_table;
mod seek;
pub mod segment_index;
mod stream;
pub mod stream_index;
pub mod streaming;
pub mod subtitle_mux;
pub mod timecode;
pub mod track_header;
pub mod track_info;
pub mod tracks;

// Re-export Matroska block-level addition types at crate root
pub use demux::matroska::matroska_v4::{BlockAddIdType, BlockMore};

// Re-export main types at crate root
pub use container_probe::{
    percentile, probe_detailed, DetailedContainerInfo, DetailedStreamInfo, DetailedStreamStats,
    MultiFormatProber,
};
pub use dash::{
    emit_mpd, DashAdaptationSet, DashManifestConfig, DashRepresentation, DashSegmentTemplate,
    DashSegmentTimeline, DashSegmentTimelineEntry,
};
pub use demux::mpegts::scte35::{
    parse_splice_info_section, BreakDuration, Scte35Config, Scte35Parser, SpliceCommand,
    SpliceDescriptor, SpliceInfoSection, SpliceInsert, SpliceTime, SCTE35_DEFAULT_PID,
};
pub use demux::Demuxer;
pub use demux::{
    BufferStats, BufferedPacket, PacketBuffer, ReadAheadBuffer, DEFAULT_READ_AHEAD_SIZE,
};
pub use format::ContainerFormat;
pub use metadata::batch::{BatchMetadataUpdate, BatchResult};
pub use mux::mpegts::scte35::{
    emit_splice_insert, emit_splice_null, emit_time_signal, SpliceInsertConfig,
};
pub use mux::{Muxer, MuxerConfig};
pub use packet::{Packet, PacketFlags};
pub use probe::{probe_format, ProbeResult};
pub use seek::{
    ClosedLoopSeekError, DecodeSkipCursor, MultiTrackSeeker, MultiTrackSeekerError, PtsSeekResult,
    SampleAccurateSeeker, SampleIndex, SampleIndexEntry, SeekAccuracy, SeekFlags, SeekIndex,
    SeekIndexEntry, SeekMode, SeekPlan, SeekResult, SeekTarget, TrackIndex,
};
pub use stream::{CodecParams, Metadata, StreamInfo};
