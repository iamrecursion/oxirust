# Wave 5 Deltas (0.1.5)

Summary of what Wave 5 landed in the 0.1.5 cycle. Source pointers are
relative to the repository root.

## Slice A — Transcode pipeline frame-level executor

- `TranscodePipeline::execute()` in
  [`crates/oximedia-transcode/src/pipeline.rs`](../crates/oximedia-transcode/src/pipeline.rs)
  (line 1165). Constructs an internal `Pipeline` from the stored
  `PipelineConfig` and drives it to completion. The companion
  `TranscodePipelineBuilder` supplies the fluent construction API.
- `MultiTrackExecutor<M: Muxer>` in
  [`crates/oximedia-transcode/src/multi_track.rs`](../crates/oximedia-transcode/src/multi_track.rs)
  (line 285) — DTS-ordered multi-track interleaving for synchronous mux of
  video / audio / subtitle tracks.

## Slice B — HW-accel platform probes

Pure-Rust, no FFI:

- [`crates/oximedia-transcode/src/hw_accel/macos.rs`](../crates/oximedia-transcode/src/hw_accel/macos.rs):
  `probe_macos()` reads `sysctl -n hw.optional.arm64` and `machdep.cpu.
  brand_string`, identifies `M1`/`M2`/`M3`/`M4` or Intel Mac, and returns
  a `HwAccelCapabilities` with the VideoToolbox codec matrix (M3/M4 adds
  AV1 decode).
- [`crates/oximedia-transcode/src/hw_accel/linux.rs`](../crates/oximedia-transcode/src/hw_accel/linux.rs):
  `probe_linux()` walks `/sys/class/drm/`, reads PCI vendor IDs
  (`0x8086` Intel, `0x1002` AMD, `0x10de` NVIDIA) via
  `std::fs::read_to_string`, and emits per-card codec capabilities. No
  libva, no DRM ioctls.
- Shared [`capabilities.rs`](../crates/oximedia-transcode/src/hw_accel/capabilities.rs)
  type (`HwAccelCapabilities`, `HwAccelDevice`, `HwKind`) consumed by the
  transcode pipeline to pick hardware codecs.

## Slice C — ABR BBA-1 rate adaptation

- [`crates/oximedia-net/src/abr/bba1.rs`](../crates/oximedia-net/src/abr/bba1.rs)
  implements Huang et al. SIGCOMM 2014 "A Buffer-Based Approach to Rate
  Adaptation" with three-zone variant selection:
  - Reservoir `[0, r]` → lowest variant.
  - Cushion `(r, r+c]` → linear-interpolate variant index.
  - Above-cushion `(r+c, ∞)` → highest variant.
- `BbaParams::default()` (B=30 s, r=10 s, c=20 s) matches the paper;
  `BbaParams::low_latency()` (B=10 s, r=2 s, c=8 s) for live streaming.
- `select_variant(buffer_level, &params, &variants) -> usize` is a
  stateless pure function — no controller object required.

## Slice D — Container v5: SCTE-35 + BatchMetadataUpdate

- [`crates/oximedia-container/src/mux/mpegts/scte35.rs`](../crates/oximedia-container/src/mux/mpegts/scte35.rs)
  emits ad-marker PES payloads with MPEG-2 CRC-32:
  - `emit_splice_null()` (line 182) — heartbeat.
  - `emit_splice_insert(&SpliceInsertConfig)` (line 238) — ad-break
    in/out with PTS.
  - `emit_time_signal(pts: Option<Duration>)` (line 160) — programme
    signalling.
- [`crates/oximedia-container/src/demux/mpegts/scte35.rs`](../crates/oximedia-container/src/demux/mpegts/scte35.rs)
  parser: `Scte35Parser` (line 258) parses the same payloads back into
  typed events.
- [`crates/oximedia-container/src/metadata/batch.rs`](../crates/oximedia-container/src/metadata/batch.rs)
  introduces `BatchMetadataUpdate` (line 111) — container-agnostic
  in-place tag updates on MP4 / MKV / FLAC / Vorbis metadata without
  rewriting the whole file.

## Slice E — Core: ErrorContext + FormatNegotiator

- [`crates/oximedia-core/src/error_context.rs`](../crates/oximedia-core/src/error_context.rs)
  — structured error chains:
  - `ErrorFrame` (line 94) — one frame in the chain (file, line,
    function, message).
  - `ErrorContext` (line 234) — owns a `Vec<ErrorFrame>`, emitted as the
    trace when an error propagates.
  - `ErrorChain` (line 406) — iterator over the full chain.
  - `ErrorContextBuilder` (line 494) — fluent construction.
- [`crates/oximedia-core/src/codec_negotiation.rs`](../crates/oximedia-core/src/codec_negotiation.rs)
  — decoder/encoder format selection:
  - `FormatCost` trait (line 699) — cost function abstraction.
  - `FormatNegotiator<'a, F>` (line 874) — negotiates a preferred
    pixel/sample format pair between source and sink, minimising cost.

## Slice F — This document

Docs round 3: [`codec_status.md`](codec_status.md) (unchanged since
Action A), [`rate_control.md`](rate_control.md),
[`simd_dispatch.md`](simd_dispatch.md), and this file.

## Gate results (2026-04-21)

- `oximedia-transcode`: 1174 tests pass.
- `oximedia-net`: 1606 tests pass.
- `oximedia-container`: 1282 tests pass, 65 skipped.
- `oximedia-core`: 1022 tests pass.
- Zero clippy warnings on all four crates.

No source behaviour changed in Slice F — documentation only.
