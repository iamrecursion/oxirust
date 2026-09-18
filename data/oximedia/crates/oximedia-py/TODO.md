# oximedia-py TODO

## Current Status
- 105 source files providing comprehensive Python bindings via PyO3
- 94 public modules covering nearly all OxiMedia crates
- Bindings include: codecs (AV1, VP9, VP8, Opus, Vorbis, FLAC), containers (Matroska, Ogg, MP4), quality assessment, audio analysis, scene detection, filter graph, effects, LUT, EDL, streaming, async pipeline, Jupyter integration, DataFrame export, broadcast, transcoding, color management, timecode, cv2 compatibility, and many more
- Registers classes and functions into single `oximedia` Python module with `cv2` submodule
- Dependencies: pyo3, ndarray, numpy, image, tokio, plus 40+ oximedia workspace crates

## Enhancements
- [x] Add `__repr__` and `__str__` implementations to all pyclass types for better REPL experience (verified 2026-05-16; src/progress_reporting.rs:127 __repr__, 357 total __repr__/__str__ impls across src/)
- [x] Implement Python context manager protocol (`__enter__`/`__exit__`) for resource-holding types (demuxers, encoders) (verified implemented 2026-05-05)
- [x] Add type stub generation (.pyi files) for IDE autocomplete and static type checking (verified implemented 2026-05-05)
- [x] Extend `cv2_compat` with additional OpenCV function coverage (morphological ops, contour detection) (verified implemented 2026-05-05)
- [x] Add numpy array zero-copy frame access in `VideoFrame` and `AudioFrame` via buffer protocol (completed 2026-05-06)
- [x] Implement async/await support for Python 3.10+ coroutines in `async_pipeline` (completed 2026-05-15 — pyo3-async-runtimes 0.28 added; `process_frame_async` and `process_batch_async` on `AsyncPipeline` return native Python coroutines via `future_into_py`; 6 tests in `src/async_pipeline.rs`)
- [x] Add pickle support for serializable types (EncoderConfig, QualityScore, etc.) (verified implemented 2026-05-05)
- [x] Extend `dataframe` module with Apache Arrow export for zero-copy pandas interop (verified implemented 2026-05-05)

## New Features
- [x] Add `oximedia.io` submodule for file read/write operations (open, probe, transcode) (verified implemented 2026-05-05)
- [x] Implement Python logging integration — route Rust tracing events to Python logging module (verified implemented 2026-05-05)
- [x] Add `oximedia.utils` submodule with common media helper functions (duration_to_timecode, fps_to_rational) (verified implemented 2026-05-05)
- [x] Implement streaming iterator protocol for frame-by-frame decoding (`for frame in decoder`) (verified implemented 2026-05-05)
- [x] Add `oximedia.benchmark` module exposing profiler bindings for Python performance analysis (verified implemented 2026-05-05)
- [x] Implement callback-based progress reporting for long-running operations (encode, transcode) (verified implemented 2026-05-05)
- [x] Add CLI wrapper — `python -m oximedia transcode input.mkv output.webm` (completed 2026-05-06)
- [x] Implement `oximedia.test` submodule with test media generators (synthetic video/audio frames) (verified implemented 2026-05-05)

## Performance
- [x] Minimize GIL holding time in CPU-intensive operations (encode, decode, quality assessment) (completed 2026-05-06 — 64 `py.detach` sites across 16 files)
- [x] Use `PyBuffer` protocol for zero-copy data exchange with numpy arrays (completed 2026-05-06 — `__getbuffer__`/`__releasebuffer__` on `VideoFrame.plane(i)`, `AudioFrame.samples`, `Mat`)
- [x] Implement batch processing APIs that release GIL for parallel Rust execution (completed 2026-05-06)
- [x] Add optional memory pool for frame allocation to reduce Python GC pressure (completed 2026-05-15 — `PyFramePool` pyclass in `src/frame_pool.rs`; acquire/release cycle with zero-fill, capacity enforcement, overflow up to 2×cap; 6 tests)
- [ ] Profile and optimize hot paths in `filter_graph` Python-Rust boundary crossings

## Testing
- [x] Add Python-level integration tests using pytest (test import, basic encode/decode roundtrip) (completed 2026-05-06 — 9 test files in `tests/`)
- [ ] Test cv2_compat API parity with actual OpenCV Python — verify compatible function signatures
- [ ] Add memory leak tests for long-running Python scripts using tracemalloc
- [ ] Test Jupyter integration with nbconvert-based notebook execution
- [ ] Verify all pyclass types are importable and constructible from Python

## Documentation
- [ ] Add Python API reference generated from type stubs
- [ ] Create Jupyter notebook tutorials for common workflows (transcode, quality check, scene detect)
- [ ] Document cv2 compatibility layer with migration guide from OpenCV Python

## Deferred Stubs (added 2026-05-04 by /stub-check Wave 3)

- [ ] oximedia-py: `src/timeline_py.rs:689` — Python timeline render()
  - **Approach:** Python wrapper for the edit render pipeline. Depends on render_video_at and render_audio_at in oximedia-edit being fully implemented first (see crates/oximedia-edit/TODO.md). Once those are done: call `oximedia_edit::Renderer::render_at(timestamp)` from the Python binding, convert the resulting VideoFrame to a numpy array (same pattern as frame_converter.rs), and expose progress callbacks via a Python callable.
  - **Scope:** medium (est. ~300 LoC) but blocked on render_video_at/render_audio_at
  - **Prerequisites:** oximedia-edit render_video_at + render_audio_at fully implemented
  - **Risk:** Python GIL contention during long render; async render loop needs careful tokio/pyo3 bridge
