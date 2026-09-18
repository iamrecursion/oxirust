# oximedia-web roadmap

Status as of 2026-07-12.

- **M0** — gates + skeleton — *done*. Nested workspace, 5 crates
  (`oximedia-web-core` + 4 module crates), 4 gate scripts (`build.sh`,
  `size-gate.sh`, `dep-gate.sh`, `serve.sh`), `allowed-deps.txt` (23 crates,
  generated from a real `cargo tree`), `deny.toml` (licenses check passes),
  `package.json`, root `Cargo.toml` `exclude = ["fuzz", "web"]`. All 10
  requested verification steps passed at hand-off.
- **M1** — `oximedia-web-scopes` port — *done*. Waveform, vectorscope,
  histogram, false-colour; allocation-free renders after warm-up; three
  known upstream bugs fixed during the port (RGB24→RGBA8 stride, false
  colour output-size mismatch, incomplete font glyph set) plus one wasm-web
  correctness requirement (zero per-frame allocation). 39 tests pass;
  native + wasm32 clippy clean at `-D warnings`; wasm 21,669 B gzip (14% of
  the 153,600 B soft budget); dep-gate passes, no new deps.
- **M2** — OxiScope demo v1 — *done*. `web/demo/` — dark colorist page,
  three input paths (file/webcam/procedural patterns), full grade panel
  wired to `oximedia-web-color`, all four `oximedia-web-scopes` scopes fed
  from the *graded* output, `.cube` export, runtime-measured size badge,
  capabilities readout, hard SIMD-missing error (no software fallback).
  Verified end-to-end in headless Chrome: tone-map roll-off visibly
  provable on the waveform, all four scopes render non-blank, exported
  `.cube` is a valid 35,937-line LUT. Found and locally worked around a bug
  in the shared `_frame.js` SIMD probe (out of the demo agent's scope to
  fix at the source).
- **M3** — `oximedia-web-color` — *done*. Exposure/contrast/saturation,
  Reinhard/Reinhard-extended/Hable/filmic/ACES(Narkowicz)/ACES-ODT
  (`AcesOt2` port) tone mapping with peak-nits control, primaries-aware
  gamut mapping (BT.709/BT.2020/Display-P3), sRGB/PQ/HLG/linear transfer
  functions, `.cube` load/export (trilinear + tetrahedral). 105 unit + 1
  integration test pass; native + wasm32 clippy clean; wasm 59,389 B gzip
  (29% of the 204,800 B soft budget) at the time this milestone shipped.
  **Known shortfall:** the 6 ms/1080p wasm performance target is *not* met
  — measured ~44 ms in the crate's own perf canary at the time (documented
  as a safe-scalar-wasm limitation, not a correctness bug). **Superseded
  by the P1 perf-retune pass below**: current size is 59,636 B gzip and
  current browser-bench-measured cost is ~25 ms/1080p-frame against a
  separately-tracked ≤12 ms bench budget (still not met, ~3.7x faster) —
  see the **P1** entry below and
  [`README.md`'s Limitations](README.md#limitations) for current numbers;
  this bullet is left as the historical record of the M3 milestone.
- **M4** — ship-prep — *done* (publish deliberately withheld). Bench
  harness ([`bench/`](bench/), M4a) built and verified end-to-end via
  headless Chrome; README patent paragraph in place; `package.json` filled
  in (scoped `@cooljapan/oximedia-web`, 4 subpath exports, no deps) but
  **not published to npm** — publish happens only on explicit user
  instruction, per policy, and none has been given yet.
- **M5** — scale + quality — *done*. `oximedia-web-scale` (M5a): Lanczos3 /
  Catmull-Rom / Mitchell-Netravali / bilinear separable resampler
  (corrected an upstream naming bug — `oximedia-scaling`'s "Bicubic" is
  actually Catmull-Rom); 33 tests pass; wasm 17,439 B gzip (14% of the
  122,880 B soft budget). `oximedia-web-quality` (M5b): windowed SSIM
  (single-scale, 11×11 Gaussian) cross-validated against a naive reference,
  PSNR (RGB + BT.709 luma, real `f64::INFINITY` for identical frames); VMAF
  explicitly **not implemented** (documented deferral — three unvalidated
  implementations exist upstream, none suitable to port as-is); 32 tests
  pass (1 `#[ignore]`d perf test); wasm 15,485 B gzip (8% of the 204,800 B
  soft budget). Both native + wasm32 clippy clean, dep-gate passes.
- **X1** — `oximedia-wasm` defect fixes — *partial*. All 7 documented
  defects fixed (f64 data-plane APIs → f32/u8, JSON hot path → typed
  getters, `WasmVp8Decoder`/`WasmAv1Decoder`/`WasmVorbisDecoder` removed,
  4 unused deps + a dead `/tmp` path removed, 3 orphaned modules
  [`hdr_wasm`, `lut_wasm`, `spatial_wasm`] wired in, `wasm-opt` re-enabled,
  npm packaging/README honesty corrected) plus one extra `&[f64]`
  wasm-bindgen violation found and fixed
  (`audiopost_wasm.rs::wasm_mix_audio`). Native check/clippy/test are 100%
  clean. Marked *partial* only because `cargo check -p oximedia-wasm
  --target wasm32-unknown-unknown` remains blocked by a **pre-existing,
  out-of-scope** `wgpu` API mismatch in `crates/oximedia-gpu`
  (`RequestAdapterOptions` missing field `apply_limit_buckets`), reached
  transitively through `oximedia-colormgmt`'s default `gpu-accel` feature —
  unrelated to and untouched by this task; flagged to the owning team.
- **X2** — root workspace tokio feature-unification fix — *done*. Root
  `Cargo.toml`'s `tokio = { features = ["full"] }` was silently unioning
  `"full"` into every workspace member via Cargo feature unification,
  pulling `mio` (net feature) into the wasm32 build through
  `oximedia-graph`. Fixed by pinning the root to
  `default-features = false` and giving every tokio-declaring member an
  explicit feature list (`oximedia-graph` gets the minimal net-free set;
  everything else keeps the prior full-feature behaviour, verified as an
  exact superset with zero behavioural change). `oximedia-graph`'s 814
  tests pass; native `cargo check --workspace` is clean; the `mio` wasm32
  blocker is confirmed gone (the *remaining* wasm32 blocker is the
  unrelated `oximedia-gpu`/`wgpu` issue tracked under X1 above).
- **X3** — docs honesty pass — *done*. `docs/simd_dispatch.md`'s WASM
  SIMD128 section rewritten: it previously and falsely claimed
  `oximedia-simd` has a working WASM tier reusing SSE4.2 paths "exercised
  by an `oximedia-codec` WASM test matrix" — in reality `oximedia-simd` has
  zero wasm32-specific code (wasm32 builds compile the scalar fallback) and
  `oximedia-codec` does not even depend on `oximedia-simd`; the section now
  documents the two real WASM SIMD paths that do exist:
  `oximedia-codec`'s own `core::arch::wasm32` module
  (`crates/oximedia-codec/src/simd/wasm.rs`) and this workspace's
  autovectorization-over-`chunks_exact` approach
  (`-C target-feature=+simd128` in `web/.cargo/config.toml`).
  `docs/codec_status.md` gained a browser-surface note (native codec status
  unchanged; `oximedia-wasm`'s npm surface no longer exports standalone
  VP8/AV1/Vorbis decoder classes). `oximedia-wasm/README.md` was reviewed
  and found already honest (no residual npm-installability claims about
  unpublished packages) — no edit needed there.
- **P1 — WASM kernel perf-retune pass** — *partial*. `scopes`/`color`/
  `scale` per-frame kernels retuned (const-span monomorphised Lanczos
  h-pass + 4-tap-fused v-pass and an opaque-frame premultiply skip in
  `scale`; u64-packed LUT lattice points + branchless tetrahedral select +
  last-pixel memo in `color`; killed per-pixel fn-pointer YCbCr dispatch,
  vectorised row-buffer conversion, run-collapsed scatter accumulation,
  and a new `Scopes.load_frame`/`*_current()` resident-frame API in
  `scopes`) — see the [`CHANGELOG`](../CHANGELOG.md) `[Unreleased]` entry
  and [`README.md`'s Measured performance](README.md#measured-performance)
  section for the itemized changes and full table. Measured via two
  consecutive `web/bench/run.sh` runs (headless Chrome 150,
  `hardwareConcurrency` 8, macOS, 2026-07-12; the two runs agreed within
  ~2.1% on every suite): `scopes` all-four-combined **13.00 / 13.25 ms vs.
  the ≤16 ms budget — MET** (≤8 ms stretch goal not met; worst-case-noise
  input 13.40–13.60 ms); `color` exposure+ACES+LUT33 **24.90 / 25.10 ms
  vs. the ≤12 ms budget — NOT MET** (~3.7x faster than this pass's
  pre-retune baseline); `scale` lanczos3 4K→1080p **51.80 / 52.25 ms vs.
  the ≤40 ms budget — NOT MET** (~5.4x faster). Total wasm+glue gzip size
  (re-measured from a full `scripts/build.sh` clean rebuild;
  `size-gate.sh`/`dep-gate.sh` both pass) held at 150,072 B / 512,000 B
  soft budget (29%) — up from a prior 144,045 B snapshot but still
  comfortably inside budget. Marked *partial* because two of the three ms
  budgets below are not yet met.
  - [ ] `color`: close the gap between the measured 24.90–25.10 ms and the
    ≤12 ms budget. Reported cause (not independently re-verified by this
    pass): `VideoFrame` acquisition/copy + canvas put/get overhead around
    the wasm kernel, not the kernel itself. Next lever to investigate: a
    zero-copy wasm-memory-view path through `js/_frame.js`/`js/color.js`.
  - [ ] `scale`: close the gap between the measured 51.80–52.25 ms and the
    ≤40 ms budget. Reported cause (not independently re-verified by this
    pass): 4K frame copy/boundary-copy cost around an already
    near-native-speed wasm kernel. Next lever to investigate: a
    resident-frame cache pattern like `scopes`'
    `Scopes.load_frame`/`*_current()` API above.
  - [ ] `scopes`: the ≤8 ms stretch goal for all-four-combined is not met
    (13.00–13.25 ms measured, worst-case-noise 13.40–13.60 ms); no plan
    yet for closing this further.
  - [ ] Re-verify the "JS plumbing, not kernel" attribution for the
    `color`/`scale` gaps above with an isolated wasm-kernel-only (no
    `VideoFrame`/canvas acquisition) micro-benchmark before treating it as
    settled fact rather than a hypothesis.
- **Polish-fix statuses** (three defects assigned to a parallel fixes pass
  this round; all three were found *already correctly fixed* in the
  shared working tree before that pass started — verified end-to-end, not
  re-implemented):
  - `web/js/_frame.js` `SIMD_PROBE` bytes — **fixed**. Matches the
    documented 29-byte WASM structure (code-section size at index 20,
    function-body size at index 22); `WebAssembly.validate(new
    Uint8Array(SIMD_PROBE))` returns `true`. This is the same defect M2
    above flagged as a local demo-side workaround; it is now fixed at the
    source, so that workaround is no longer load-bearing (see next item).
    `web/dist/_frame.js` is byte-identical to `web/js/_frame.js`.
  - `web/demo/app.js` — **clean**; no local `detectSimd`/`SIMD_PROBE`
    override remains, `detectCapabilities` is imported directly from
    `../dist/_frame.js`. Headless-Chrome smoke test (`demo/` via
    `scripts/serve.sh`) reports `data-oxiscope="ready"` with no
    simd-unsupported error in the DOM.
  - AV1 decode honesty in `oximedia-wasm/src/media_player.rs` — **fixed**
    (outside `web/`'s own tree, but relevant to `oximedia-web`'s "no fake
    decode" guarantee story since the demo/bench can exercise
    `oximedia-wasm`'s player): `decode_video_packet` now returns an honest
    `Err` for AV1-coded tracks instead of wrapping a decoder that produced
    zero-filled output; regression test `test_next_frame_av1_track_errors`
    added. See `oximedia-wasm/TODO.md`'s "AV1 decode audit (0.1.9, later
    pass)" section for the full defect writeup.

## Notes

- **M1 / M3 / M5 implementation strategy is PORT, not depend.** Algorithms
  are copied and adapted from the native analysis crates
  (`oximedia-scopes`, `oximedia-hdr`, `oximedia-colormgmt`, `oximedia-lut`,
  `oximedia-scaling`, `oximedia-quality`) into `oximedia-web-core` as
  dependency-free kernels, rather than cargo-depending on those crates,
  because the source crates carry `rayon`, `scirs2` (BLAS), `serde_json`
  and `f64` data planes that violate the wasm size and data-plane gates
  (`scripts/size-gate.sh`, `scripts/dep-gate.sh`). The canonical-source
  file list for each port lives in that crate's `lib.rs` doc comment.
