# Changelog — oximedia-py

All notable changes to the OxiMedia Python bindings are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.7] - 2026-05-06

### Added
- PyO3 buffer protocol (`__getbuffer__` / `__releasebuffer__`) on
  `VideoFrame.plane(i)` (returns a new `PyVideoPlaneBuffer` per call),
  `AudioFrame.samples` (interleaved view as well as per-channel views), and the
  `cv2_compat::Mat` type (mutable view). Each view owns its own
  `Py_buffer.internal` block (shape/strides via `Box::into_raw`, format via
  `CString::into_raw`) so multiple concurrent views are safe. NumPy can now
  call `np.asarray(frame.plane(0))` and obtain a true zero-copy view.
- `python -m oximedia` command-line interface (`python/oximedia/__main__.py`)
  with subcommands: `probe`, `transcode`, `quality`, `cv2 cvt-color`,
  `cv2 convert`, `presets`, `codecs`, and `version`. The CLI is shipped inside
  the wheel as a "mixed" maturin layout (`python-source = "python"`).
- pytest harness under `crates/oximedia-py/tests/` with 9 test modules
  (`test_smoke.py`, `test_buffer_protocol.py`, `test_gil_release.py`,
  `test_cv2_parity.py`, `test_quality.py`, `test_stub_coverage.py`,
  `test_context_manager.py`, `test_iterators.py`, plus `conftest.py`) and a
  top-level `Makefile` exposing `make dev` / `make test` / `make clean`.
- Comprehensive PEP 561 type stubs at `python/oximedia/`: `__init__.pyi`
  (top-level pyclasses), `cv2.pyi` (cv2 submodule), plus `io.pyi`,
  `utils.pyi`, `logging.pyi`, `test.pyi`, `benchmark.pyi`, and a `py.typed`
  marker. `mypy --strict` validates cleanly against the stubs.
- `python/oximedia/__init__.py` re-exporter so `import oximedia` resolves to
  the mixed Python/Rust package surface.

### Changed
- 64 CPU-heavy operations now release the Python GIL via `Python::detach`
  across 16 source files (`audio.rs`, `audio_analysis.rs`,
  `audio_normalize.rs`, `cv2_compat/edges|features|filters|hough|image_io|
  morphology.rs`, `dedup_py.rs`, `denoise_py.rs`, `ml_py.rs`, `quality.rs`,
  `stabilize_py.rs`, `transcode_py.rs`, `video.rs`). Codec encode/decode loops
  (Av1/Vp9/Vp8/Opus/Vorbis/Flac), ML inference, quality metrics, denoising,
  stabilisation, audio analysis, transcoding, dedup, and most cv2 routines
  now run with the GIL released, allowing other Python threads to make
  progress.
- `pyproject.toml` switched to maturin "mixed" layout: `python-source =
  "python"`, dotted `module-name = "oximedia.oximedia"`, and an explicit
  `include` list so `.pyi` stubs, `py.typed`, and `__main__.py` ship in every
  built wheel (these were silently omitted previously).
- `src/types.rs` grew from ~1,000 to ~1,600 lines while still under the
  workspace 2,000-line policy, accommodating the new buffer protocol
  implementations.
