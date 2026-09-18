# oximedia-container TODO

## Current Status
- 95+ source files; container format demuxing and muxing
- Demuxers: Matroska/WebM, Ogg, FLAC, WAV, MP4, MPEG-TS, SRT, WebVTT, Y4M
- Muxers: Matroska/WebM, Ogg, FLAC, WAV, MPEG-TS, Y4M, CMAF
- Features: format probing (magic bytes), seeking, metadata (read/write/edit), chapters (Matroska/MP4), attachments, cue points, streaming (HLS/DASH), fragmented MP4, PTS/DTS management, track selection
- Modules: demux/, mux/, metadata/, chapters/, fragment/, streaming/, data/ (telemetry/GPS), edit/, tracks/, attach/, cue/, etc.

## Enhancements
- [x] mp4-muxer — muxer exists at mux/mp4/ (5 files: basic.rs, simple.rs, mod.rs, facade.rs, writer.rs); fMP4 fragmented mode tracked by Wave 3 Slice D
- [x] Extend `demux/matroska/parser.rs` with full support for Matroska v4 elements (Block Addition Mappings) (Wave 14: block-level BlockMore/BlockAdditional parsing + BlockAddIdType registry — matroska_v4.rs parse_block_additions/parse_block_more + parser.rs parse_block_group BLOCK_ADDITIONS arm + Block.block_additions field)
- [x] Add sample-accurate seeking in `seek.rs` for all container formats (not just keyframe-based) (SampleAccurateSeeker done)
- [x] Extend `metadata/editor.rs` with batch tag operations (copy all tags between files)
- [x] Improve `streaming/mux.rs` with CMAF low-latency chunked transfer encoding (verified: streaming/mux.rs:376:CmafChunkMode, :384:LowLatencyChunked, :395:CmafChunkedConfig, push_sample)
- [x] Add edit list support in `edit_list.rs` for gapless audio and trimmed video playback (verified 2026-05-16; src/demux/mp4/boxes.rs:550 gapless elst parse, preroll_samples:636; src/edit/list.rs:75 EditList)
- [x] Extend `demux/mpegts/` with SCTE-35 ad insertion marker parsing and mux emission
- [x] Add `container_probe.rs` detailed stream analysis (bitrate distribution, keyframe interval) (verified 2026-05-16; src/container_probe.rs:probe_detailed — DetailedStreamStats with bitrate/keyframe histograms, percentile stats)

## Wave 2 Progress (2026-04-17)
- [x] AVI container: muxer (mux/avi/writer.rs) + demuxer (demux/avi/reader.rs) implemented — MJPEG-only, ≤1 GB. Wave 2 Slice F (2026-04-17). Follow-ups: OpenDML >1GB, audio streams, non-MJPEG codecs.
- [x] MKV MJPEG support: V_MJPEG codec ID now emitted correctly. Wave 2 Slice B.
- [x] MKV APV support: V_MS/VFW/FOURCC with BITMAPINFOHEADER CodecPrivate (biCompression=apv1). Wave 2 Slice B.

## Wave 3 Progress (2026-04-17)
- [x] AVI v3: OpenDML >1 GB super-index, PCM audio track, H264/RGB24 codec FourCCs — Slice C of /ultra Wave 3 (2026-04-17)
- [x] MP4 muxer gap-fill: fMP4 fragmented mode (moof+mdat+sidx), AV1 av1C, MJPEG/APV codec arms — Slice D of /ultra Wave 3 (2026-04-17)
- [x] Matroska+streaming: sample-accurate seek (Cues walk+decode-skip), gapless elst, DASH SegmentTemplate — Slice F of /ultra Wave 3 (2026-04-17)

## Wave 4 Progress (2026-04-18)
- [x] mkv-v4-blockadd: BlockAdditionMappingType 4 (HDR10+) + 5 (DV EL) emit+parse — Wave 4 Slice D
- [x] seek-sample-accurate-all: extend MKV pattern to MP4 (stss+stts+ctts) + AVI (idx1/super-index) — Wave 4 Slice D
- [x] cmaf-ll-chunked: CmafChunkedMode with moof+mdat per chunk, styp=cmfl, prft boxes — Wave 4 Slice D

## 0.1.8 Wave 6 — 2026-05-29
- [x] Split mp4/writer.rs (1985 lines — near policy limit) via splitrs (Slice A, 2026-05-29) (verified 2026-05-29; split complete, all sub-modules < 2000 lines; mod.rs 1219, boxes.rs 362, fragment.rs 154, sample_table.rs 145, types.rs 240; 1372 tests pass, 0 warnings)
  - **Goal:** Module directory mux/mp4/writer/ preserving full public API; no file >= 2000 lines
  - **Files:** crates/oximedia-container/src/mux/mp4/writer.rs → writer/{mod,boxes,fragment,sample_table,types}.rs

## Wave 5 Progress (2026-04-18)
- [x] scte35-parse-emit: `parse_splice_info_section()` free function + `emit_time_signal/splice_null/splice_insert` — demux/mpegts/scte35 + mux/mpegts/scte35, re-exported from crate root — Wave 5 Slice D
- [x] metadata-batch: `BatchMetadataUpdate` builder + `BatchResult` in `metadata/batch.rs`, re-exported from crate root — Wave 5 Slice D
- [x] integration tests: `tests/scte35_parse.rs` (9 tests) + `tests/metadata_batch.rs` (12 tests) smoke-testing public re-exports — Wave 5 Slice D

## New Features
- [x] Implement CAF (Core Audio Format) demuxer/muxer for Apple ecosystem compatibility
- [x] Add MPEG-DASH segment template generation in `streaming/` (verified 2026-05-16; src/streaming/dash.rs:260 SegmentTemplate XML emission)
- [x] Implement sample grouping and random access point indexing in `sample_table.rs` (verified 2026-05-16; src/sample_table.rs:394 lines, random access lookup)
- [x] Add subtitle stream muxing support for Matroska (WebVTT/ASS embedded subtitles)
- [x] Implement `timecode/track.rs` SMPTE timecode track reading for professional workflows (verified 2026-05-16; src/timecode/track.rs:305 lines)
- [x] Add multi-angle support in Matroska via `tracks/selector.rs`
- [x] Implement fragmented MP4 (fMP4) live ingest support in `fragment/` (verified 2026-05-16; src/fragment/mp4.rs:367 FragmentedMp4Ingest, ingest fn:400)
- [x] Add MKV attachment extraction and insertion CLI in `attach/matroska.rs` (verified 2026-05-16; src/attach/matroska.rs:270 lines)

## Performance
- [x] Implement buffered async I/O in `demux/buffer.rs` with configurable read-ahead size (ReadAheadBuffer done)
- [x] Add memory-mapped I/O option for large file demuxing via `mmap` feature gate (MmapDemuxSource done)
- [x] Optimize EBML variable-length integer parsing in `demux/matroska/ebml.rs` with lookup tables (verified: ebml.rs:VINT_LEN_TABLE+VINT_MASK_TABLE)
- [x] Cache cluster/cue positions in `cue/` for faster random-access seeking in Matroska (verified: cue/cache.rs:54:CuePositionCache, find_cluster_before, BTreeMap)
- [x] Parallelize mux writing in `mux/matroska/writer.rs` for multi-stream interleaving (verified: mux/matroska/interleave.rs:thread::scope parallel encode, sort-by-pts)
- [x] Optimize `pts_dts.rs` timestamp rewriting with batch processing (Wave 22: rewrite_timestamps_batch+rebase_timestamps_to_zero free functions in pts_dts.rs; 11 new unit tests)

## Testing
- [x] Add conformance tests for Matroska demuxer against Matroska test suite files (Wave 22: 7 new EBML conformance tests in it_mkv_conformance.rs — Matroska DocType probe/round-trip, hard-coded minimal MKV fixture, malformed/truncated EBML, header-only, wrong DocType; total 15 tests)
- [x] Test MP4 demuxer with fragmented MP4 (fMP4) and progressive MP4 variants
- [x] Add round-trip test: demux -> mux -> demux -> verify packet-level equality for all formats
- [x] Test MPEG-TS demuxer with real broadcast streams containing multiple programs
- [x] Add stress test for `streaming/demux.rs` with simulated live stream input (continuous data)
- [x] Test `metadata/vorbis.rs` Vorbis comment handling with edge cases (empty tags, unicode)

## Documentation
- [x] Document container format support matrix (which codecs in which containers) — rustdoc table in `lib.rs` "Container Format Support Matrix" section (2026-06-24)
- [x] Add seeking strategy documentation (keyframe vs. sample-accurate vs. byte-offset) — rustdoc in `lib.rs` "Seeking Strategies" section with runnable `SeekIndex`/`SeekTarget` examples (2026-06-24)
- [x] Document streaming output modes (HLS, DASH, CMAF) with configuration examples — rustdoc in `lib.rs` "Streaming Output Modes" section with `no_run`/runnable examples for `SegmentWriter`, `emit_mpd`, `CmafChunkedEncoder` (2026-06-24)
