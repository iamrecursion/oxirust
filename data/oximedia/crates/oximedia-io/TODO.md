# oximedia-io TODO

## Current Status
- 27+ modules providing the I/O foundation for OxiMedia
- Sources: FileSource (tokio async), MemorySource, MediaSource trait
- Bit-level: BitReader with Exp-Golomb coding support for H.264-style parsing
- Buffering: buffered_io, buffered_reader, ring_buffer, buffer_pool
- Advanced I/O: aligned_io, async_io, scatter_gather, seekable, mmap, splice_pipe
- Writing: chunked_writer, write_journal (journaled writes for crash safety)
- Utilities: checksum, compression, copy_engine, format_detector, io_stats, rate_limiter
- File ops: file_metadata, file_watch, temp_files, progress_reader, verify_io
- Pipeline: io_pipeline for chaining I/O operations
- Dependencies: oximedia-core, bytes, async-trait, tokio (non-wasm)

## Enhancements
- [x] Add configurable read-ahead buffering to `buffered_reader.rs` with adaptive window sizing
- [x] Extend `ring_buffer.rs` with wait-free SPSC (single producer single consumer) mode for async pipelines
- [x] Add CRC-32C (Castagnoli) alongside existing checksums in `checksum.rs` for modern formats (verified 2026-05-16; src/checksum.rs:219 Crc32c variant, compute fn:252)
- [x] Implement write coalescing in `chunked_writer.rs` to batch small writes into larger I/O operations
- [x] Extend `format_detector.rs` with detection for all OxiMedia-supported container formats (MXF, DPX, EXR)
- [x] Add `rate_limiter.rs` support for both read and write direction with separate bandwidth limits (verified 2026-05-16; src/rate_limiter.rs:371 IoDirection::Read/Write, read_bucket/write_bucket)
- [x] Extend `mmap.rs` with huge page support on Linux for large file mappings (verified 2026-05-16; src/mmap.rs:317 HugePageSize enum, MAP_HUGETLB:320)
- [x] Add `file_watch.rs` support for recursive directory watching with debounce (verified 2026-05-16; src/file_watch.rs:57 debounce Duration, recursive:59)

## New Features
- [x] Add an `http_source.rs` module for streaming media from HTTP/HTTPS URLs with range request support (verified 2026-05-16; src/http_source.rs:559 lines)
- [x] Implement an `s3_source.rs` module for reading from S3-compatible object storage (verified 2026-05-16; src/s3_source.rs:566 lines)
- [x] Add a `pipe_source.rs` module for reading from Unix pipes and stdin (verified 2026-05-16; src/pipe_source.rs:371 lines)
- [x] Implement a `multipart_writer.rs` module for writing large files in parallel segments (verified 2026-05-16; src/multipart_writer.rs:501 lines)
- [x] Add a `prefetch.rs` module for predictive I/O prefetching based on access patterns (verified 2026-05-16; src/prefetch.rs:455 lines)
- [x] Implement a `dedup_writer.rs` module for content-deduplicating writes (hash-based block dedup) (verified 2026-05-16; src/dedup_writer.rs:282 lines)
- [x] Add an `io_metrics.rs` module with Prometheus-compatible I/O throughput and latency metrics (verified 2026-05-16; src/io_metrics.rs:264 lines)
- [x] Implement a `retrying_source.rs` wrapper that retries failed reads with exponential backoff

## Performance
- [x] Add vectored I/O (readv/writev) support to `scatter_gather.rs` for reduced syscall overhead (verified 2026-05-16; src/scatter_gather.rs:1 vectored I/O primitives, ReadVec/test_readvec:237)
- [x] Implement direct I/O (O_DIRECT) option in `aligned_io.rs` for bypassing OS page cache (verified 2026-05-16; src/aligned_io.rs:7 O_DIRECT modeling, alignment-aware:100)
- [x] Add zero-copy sendfile/splice support in `copy_engine.rs` on Linux (planned 2026-06-01)
  - **Goal:** Add `CopyMode::ZeroCopy` that uses kernel-accelerated I/O without any new dependencies.
  - **Design:** `src/copy_engine.rs:14` `CopyMode` enum has Buffered/Sparse/Chunked — no zero-copy path. Add `CopyMode::ZeroCopy` arm in `run()` at :120 using `std::io::copy(&mut File::open(src)?, &mut File::create(dst)?)` — on Linux std auto-selects `copy_file_range`/`sendfile` internally; portable fallback elsewhere. **No new dep — 100% Pure Rust.** (Raw `libc::splice` deferred as a feature-gated follow-up.)
  - **Files:** `src/copy_engine.rs`, `TODO.md`.
  - **Tests:** ZeroCopy produces byte-identical output to Buffered on a temp file (`std::env::temp_dir()`); empty-file case; `CopyMode::ZeroCopy.to_string()` roundtrip.
  - **Risk:** none for the `std::io::copy` path — it is always safe and portable; the kernel optimization is transparent.
- [x] Optimize `BitReader` for batch bit extraction (read 32/64 bits at a time from buffer) (planned 2026-06-01)
  - **Goal:** Eliminate the per-bit loop in `read_bits`/`read_u16`/`read_u32`/`read_u64` to reduce iteration count from N bits to N/8 byte operations.
  - **Design:** `src/bits/reader.rs:128` `read_bits` currently loops bit-by-bit; `read_u16/u32/u64` at :188/:210/:231 all funnel through it (64 iterations for a u64). Add a byte-aligned fast path: when `bit_pos==0` && `n%8==0` && enough bytes remain, read directly via `u32::from_be_bytes`/`u64::from_be_bytes` from `data[byte_pos..]`. For unaligned reads, refill a `u64` accumulator and shift out `n` bits at once. Preserve exact MSB-first semantics throughout.
  - **Files:** `src/bits/reader.rs`, `TODO.md`.
  - **Tests:** batch read == bit-by-bit read for aligned + unaligned cases, all of u16/u32/u64; edge bits at buffer boundary; keep file < 2000 lines (currently 898).
  - **Risk:** endianness and partial-byte edge cases at buffer boundary — assert batch vs slow path on a comprehensive sweep.
- [x] Add io_uring support as an optional backend for `async_io.rs` on Linux (verified 2026-05-16; src/async_io.rs:149 io_uring/IOCP/kqueue data structures modeled)
- [x] Implement double-buffered reading in `buffered_io.rs` for overlapped I/O and processing (planned 2026-06-01)
  - **Goal:** Let the caller process one buffer while the background thread fills the next, eliminating I/O stalls.
  - **Design:** `src/buffered_io.rs` (405L) has `BufferPool`/`ReadAheadBuffer`/`CoalescingWriter` but no overlapped reader. Add `DoubleBufferedReader<R: Read + Send>` owning two heap buffers; a `std::thread` fills the back buffer while the caller consumes the front, swapping via `std::sync::mpsc` channel + `Mutex`/`Condvar` (std only, no crossbeam). On EOF the background thread signals done; `read()` drains the remaining buffer then returns 0.
  - **Files:** `src/buffered_io.rs`, `TODO.md`.
  - **Tests:** `DoubleBufferedReader` yields the same byte stream as a plain `Read` impl; partial final buffer; empty file; buffer larger than source; thread join must not drop the last buffer.
  - **Risk:** thread-join on drop must flush remaining bytes; test EOF mid-buffer.

## Testing
- [x] Add tests for `write_journal.rs` crash recovery by simulating interrupted writes (Wave 27)
  - **Caveat:** `WriteJournal` is in-memory, serialize-only — there is NO persistence/`replay()` API. The test pins the *real* durability surface: the 40-byte `JournalEntry::to_bytes`/`from_bytes` codec. A torn write is modeled by truncating a concatenated entry stream mid-record; decoding in 40-byte frames recovers exactly the 7 intact records and rejects the 17-byte partial tail. Also covers zero-length file → 0 entries, and checkpoint-then-recover (checkpoint clears entries but preserves the seq counter). See `tests/io_reliability.rs`.
- [x] Test `ring_buffer.rs` under concurrent producer/consumer with varying rates (Wave 27 — SPSC, 100k seeded xorshift bytes, partial-push loop + rate-varied consumer, exact FIFO order verified)
- [x] Add `mmap.rs` tests with files larger than available RAM to verify windowed mapping (Wave 27 — `MmapFile` is an in-process simulation, so "larger than RAM" is modeled as 16×4 KiB windows advancing the region index; offset/length/slice/contiguity all asserted)
- [x] Test `format_detector.rs` with truncated files and zero-length inputs (Wave 27 — empty/1-byte JPEG/7-byte PNG/11-byte RIFF-short/3-byte EBML all → Unknown, no panic; positive control full RIFF/WAVE → Wav conf 1.0)
- [x] Add throughput benchmarks for `copy_engine.rs` comparing buffered vs. mmap vs. splice paths (Wave 27 — `benches/copy_engine_bench.rs`: Buffered/Chunked/ZeroCopy over a 4 MiB temp file + `SplicePipe::transfer` over cursors as the crate's portable "splice" dimension; harness=false, no cfg gate)
- [x] Test `progress_reader.rs` callback accuracy at various read granularities (Wave 27 — granularities [1,7,64,4096,16384] all reach bytes_read==10_000 & fraction==1.0; report_interval=2500 fires >=4 times)

## Documentation
- [ ] Add an I/O architecture diagram showing the source -> buffer -> pipeline -> writer flow
- [ ] Document the wasm32 limitations (no FileSource, no mmap, no file_watch)
- [ ] Add performance tuning guide for buffer sizes and I/O strategies per use case
