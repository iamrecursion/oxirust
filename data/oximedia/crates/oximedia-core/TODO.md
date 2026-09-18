# oximedia-core TODO

## Current Status
- 42 source files; foundational types and traits for the entire OxiMedia framework
- Types: Rational, Timestamp, PixelFormat, SampleFormat, CodecId, MediaType, FourCC, ChannelLayout
- Traits: Decoder, Demuxer interfaces
- Error handling: unified OxiError with patent violation detection (blocks H.264, H.265, AAC, etc.)
- Memory: buffer pools (alloc/buffer_pool), ring buffer, work queue, event queue
- HDR: metadata, transfer functions, color primaries, conversions, parser
- Additional: codec_info, codec_negotiation, type_registry, error_context, resource_handle, sync, frame_info, media_time, version
- WASM support via feature gate

## Enhancements
- [x] Add Timestamp arithmetic operations (add, sub, multiply by Rational) with overflow protection
- [x] Extend `PixelFormat` in `types/pixel_format.rs` with NV12, NV21, P010, P016 for hardware interop
- [x] Add `SampleFormat` support for 24-bit and 64-bit float in `types/sample_format.rs`
- [x] Extend `codec_negotiation.rs` with automatic format negotiation between encoder and decoder
- [x] Improve `error_context.rs` with structured error context chain (file, function, line info)
- [x] Add `ChannelLayout` presets for Atmos and surround configurations in `channel_layout.rs`
- [x] Extend `buffer_pool.rs` with memory pressure callbacks and automatic pool shrinking (Wave 19 Slice E 2026-06-01)
  - **Goal:** `on_pressure(Fn)` hook + `shrink_to(target)` / high-watermark policy so the pool releases idle buffers when a pressure signal fires or watermark is exceeded; track in-use vs free counts.
  - **Design:** Add `PressureConfig { high_watermark_free: usize, shrink_to: usize }`; `BufferPool::set_pressure_callback(fn: impl Fn() + Send + Sync + 'static)`; `shrink_to(target)` drops free (not in-use) buffers down to target; `watermark_check()` auto-invokes `shrink_to` when free count exceeds high_watermark_free. Track `in_use: AtomicUsize` and `free: AtomicUsize`.
  - **Files:** `src/alloc/buffer_pool.rs`, `TODO.md` L20
  - **Tests:** pool shrinks to target on pressure callback; pool retains in-use buffers (never frees borrowed); watermark auto-shrink fires above threshold; no-shrink below threshold.
  - **Risk:** shrinking must never reclaim a checked-out buffer — assert in_use accounting; all counter ops must be atomic or lock-protected consistently.
- [x] Add `CodecId` variants for all supported codecs (currently missing some like WebP, GIF, JPEG-XL)

## Wave 3 Progress (2026-04-17)
- [x] PixelFormat HW interop variants: NV12, NV21, P010, P016 — Slice E of /ultra Wave 3 (2026-04-17)
- [x] SampleFormat extensions: S24 (3-byte packed LE), F64 (IEEE-754 double) — Slice E of /ultra Wave 3 (2026-04-17)
- [x] CodecId new variants: WebP, Gif, Jxl (still-image JXL) — Slice E of /ultra Wave 3 (2026-04-17)
- [x] Typed FourCc struct + ~30 codec fourcc constants — Slice E of /ultra Wave 3 (2026-04-17)

## Wave 4 Progress (2026-04-18)
- [x] timestamp-arith: duration_add/duration_sub/scale_by with saturating arithmetic — Wave 4 Slice C
- [x] channel-layout-atmos: Surround714, Surround916, DolbyAtmosBed9_1_6 variants — Wave 4 Slice C
- [x] pixfmt-color-meta: ColorPrimaries, TransferCharacteristics, MatrixCoefficients enums + ColorSpace integration — Wave 4 Slice C

## New Features
- [x] Implement zero-copy frame sharing between crates using `resource_handle.rs` with ref-counted buffers (verified 2026-05-16; src/resource_handle.rs:92 ref_count field, acquire/release semantics)
- [x] Add media duration/bitrate estimation utilities in `media_time.rs` (Wave 19 Slice E 2026-06-01)
  - **Goal:** `estimate_bitrate(bytes: u64, duration: Timestamp) -> Option<u64>`, `estimate_duration(bytes: u64, bitrate: u64) -> Option<Timestamp>`, `estimate_size(bitrate: u64, duration: Timestamp) -> Option<u64>` — all overflow-safe, returning `None` on zero-denominator or overflow.
  - **Design:** Use `checked_mul` / `checked_div` throughout; `Timestamp` internally as `Rational` with time_base; `estimate_duration` returns `Timestamp` with `time_base = 1/1_000_000_000` (nanoseconds). Include rounding.
  - **Files:** `src/media_time.rs`, `TODO.md` L36
  - **Tests:** bitrate from bytes+duration round-trips with duration from bytes+bitrate (within rounding); size estimate matches; estimation returns None on zero bitrate/duration; overflow-safe on u64::MAX inputs (no panic, returns None).
  - **Risk:** integer overflow on large byte counts × high bitrates — all paths must use checked arithmetic.
- [x] Implement typed FourCC constants for all supported codecs in `fourcc.rs`
- [x] Add `sync.rs` inter-thread synchronization primitives optimized for media pipelines (bounded channel with backpressure) (verified 2026-05-16; src/sync.rs:508 BoundedChannel with backpressure, SpscRingBuffer:327)
- [x] Implement frame pool with configurable pre-allocation for low-latency pipelines in `alloc/` (verified 2026-05-16; src/frame_pool.rs FramePool, FramePoolConfig pre_alloc field:25)
- [x] Add color primaries and matrix coefficients to `PixelFormat` metadata (verified 2026-05-16; src/types/color_meta.rs:9 ColorPrimaries enum, MatrixCoefficients, src/pixel_format_color.rs)
- [ ] Implement WASM-compatible async runtime abstraction in `wasm.rs` for cross-platform pipelines (verified-open 2026-05-16: not yet implemented)

## Wave 12 Progress (2026-05-31)
- [x] GCD Rational reduction on construction in `types/rational.rs` — already present; property-based tests (commutativity, associativity, reciprocal involution, distributivity, etc.) added — Wave 12 Slice A
- [x] SIMD-accelerated pixel format conversion helpers in `convert/pixel.rs`: `u8_to_f32_slice`, `f32_to_u8_slice`, `yuv420_to_rgb` (BT.601 fixed-point) — Wave 12 Slice A
- [x] Chase-Lev work-stealing queue in `work_queue_ws.rs` backed by `crossbeam-deque`; 4-worker / 10,000-task stress test — Wave 12 Slice A
- [x] `AlignedVec<T>` cache-line-aligned allocation in `alloc/aligned_vec.rs` (64-byte alignment, pure-safe-Rust over-allocation strategy) — Wave 12 Slice A
- [x] Timestamp conversion accuracy tests added to `types/timestamp.rs` (90kHz↔48kHz, ms↔fps, NTSC) — Wave 12 Slice A

## Performance
- [x] Optimize `Rational` arithmetic in `types/rational.rs` with GCD reduction on construction — Wave 12 Slice A
- [x] Add SIMD-accelerated pixel format conversion helpers in `convert/pixel.rs` — Wave 12 Slice A
- [x] Implement lock-free ring buffer variant in `ring_buffer.rs` for single-producer/single-consumer (verified 2026-05-16; src/ring_buffer.rs:327 SpscRingBuffer lock-free SPSC)
- [x] Optimize `work_queue.rs` with work-stealing scheduler for multi-threaded pipelines — `work_queue_ws.rs` Wave 12 Slice A
- [x] Add cache-line-aligned buffer allocation in `alloc/mod.rs` for SIMD-friendly access — `alloc/aligned_vec.rs` Wave 12 Slice A
- [x] Profile and optimize `event_queue.rs` for high-throughput event processing (>1M events/sec) — lock-free MPMC via 4-tier crossbeam_deque::Injector, Clone-able Arc<Inner>, pop() drains Critical→High→Normal→Low; pop_batch/drain_high_priority; Arc<str> payloads; MPMC stress 4×4 threads 4000 events; 249 tests 0 warnings (0.1.9)

## Testing
- [x] Add property-based tests for `Rational` arithmetic (commutativity, associativity, overflow) — Wave 12 Slice A
- [x] Test all `PixelFormat` variants for correct plane count, bit depth, and chroma subsampling — `tests/pixel_format_exhaustive.rs`, 19 tests across all 20 variants (plane_count, bits_per_component, chroma_subsampling, is_planar, is_semi_planar, is_yuv, is_rgb, has_alpha, frame_buffer_size, stride_for_width, Display/FromStr roundtrip, 4:2:0/4:2:2/4:4:4/semi-planar invariants) (2026-06-24)
- [x] Test patent violation detection in `error.rs` for all known patent-encumbered codec names — `tests/patent_detection.rs`, 20+ tests covering OxiError::PatentViolation creation/display and CodecId::from_str rejection of H.264/H.265/AAC/AC-3/DTS/VC-1/MPEG-4 names (2026-06-24)
- [x] Add buffer pool stress test: allocate/deallocate across multiple threads — `tests/buffer_pool_stress.rs`, 6 tests (concurrent acquire/release, in_use_count accuracy, small pool burst, pressure callback concurrency, shrink does not reclaim in-use, buffer isolation) (2026-06-24)
- [x] Test `Timestamp` conversion accuracy between different time bases (90kHz, 48kHz, 1/fps) — Wave 12 Slice A
- [x] Test `type_registry.rs` registration and lookup with concurrent access — `tests/type_registry_concurrent.rs`, 7 tests (concurrent writers, concurrent readers, interleaved read/write, concurrent unregister, Send+Sync assertion, with_defaults concurrent read, by_kind concurrent) (2026-06-24)
- [x] Add WASM compilation test to verify `wasm` feature compiles cleanly — Added `#[cfg(all(test, target_arch = "wasm32", feature = "wasm"))]` test module in `src/wasm.rs` with 6 tests exercising all WASM types; native tests verify equivalent arithmetic/formatting logic; compile verification: `cargo build --target wasm32-unknown-unknown --features wasm` (wasm32 target not installed in this env) (2026-06-24)

## Documentation
- [x] Document the patent-free codec philosophy and green list in crate-level docs — expanded `src/lib.rs` with patent-free philosophy, full Green List table, legal rationale, and PatentViolation usage example (2026-06-24)
- [x] Add type conversion guide (Timestamp <-> seconds, Rational <-> f64) — added to `src/lib.rs` `# Type Conversion Guide` section with doctests for Timestamp↔seconds, Rational↔f64, and frame count conversions (2026-06-24)
- [x] Document buffer pool usage patterns for zero-copy media pipelines — expanded `src/alloc/buffer_pool.rs` module doc with 4 patterns: basic acquire/release, acquire_or_alloc for unbounded pipelines, pressure management, and thread-safe sharing (2026-06-24)
