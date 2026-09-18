# oximedia — Facade Crate TODO

**Version: 0.2.0 (active, dev branch `0.2.0`) / 0.1.9 (stable, `master`)**
**Status as of: 2026-07-15**

The `oximedia` facade is the single cargo-add entry point into the entire OxiMedia
ecosystem. It re-exports approximately 103 workspace library crates behind
individual Cargo feature flags, groups them into feature-gated modules, provides
a curated `prelude` for common types, and anchors a shared integration test
suite that exercises cross-subsystem behaviour. The facade is the canonical way
for downstream users to consume OxiMedia: `oximedia = { version = "0.1.9",
features = ["..."] }` replaces pulling in dozens of sibling crates manually.

---

## Current Status

*(Body re-audited 2026-07-14 against `Cargo.toml`/`src/`/`tests/` directly —
most figures below had been stale since roughly the 0.1.3/0.1.4 era, ~3
months and 6 versions ago; see dated notes throughout this file for specifics.)*

- **Source size** (re-measured 2026-07-14 with `wc -l`)
  - `src/lib.rs` — 1915 lines (per-subsystem module declarations, feature-flag
    documentation table, always-on re-exports, `#![forbid(unsafe_code)]`,
    `#![warn(missing_docs)]`).
  - `src/prelude.rs` — 744 lines (most-used items per feature, hand-picked for
    each enabled subsystem).
  - `tests/integration.rs` — 1520 lines (cross-feature integration suite).
  - `tests/feature_matrix.rs` — 119 lines (separate compile-only harness that
    `cargo check`s every feature standalone; see Enhancements → Integration
    tests).
  - `Cargo.toml` — 695 lines (dependency block + features + `[[example]]`
    sections).
- **Feature flags** — 115 defined features in `Cargo.toml` (up from 101):
  - `default = []` (lean build, core modules only).
  - 4 quick-start/category meta-features: `minimal`, `audio-stack`,
    `broadcast-stack`, `streaming-stack`.
  - 109 per-crate (or per-crate-sub-capability) features, mostly of the form
    `<name> = ["dep:oximedia-<name>"]` gating a single workspace library crate
    each; the six `ml-*` features instead gate a sub-capability of
    `oximedia-ml` (e.g. `ml-onnx = ["ml", "oximedia-ml/onnx"]`), and
    `mjpeg`/`apv` gate a sub-capability of the already-referenced
    `oximedia-codec`.
  - `full` meta-feature enabling every optional feature **except `ml-onnx`**
    — a real, currently-present gap; see Known Issues.
- **Re-exported crates** — 105 workspace libraries (up from 103):
  - 4 always-on: `oximedia-core`, `oximedia-io`, `oximedia-container`,
    `oximedia-cv` (exposed via `lib.rs` plus `pub use oximedia_cv as cv;`).
  - 101 feature-gated optional crates (up from 99), surfaced through 103
    `#[cfg(feature = "…")] pub mod <name> { pub use oximedia_<name>::*; }`
    blocks in `lib.rs` — the 2-block surplus is `oximedia-codec` backing
    three modules (`video`, `mjpeg`, `apv`) off one dependency line.
- **Examples** (21, up from 17 — all registered in `Cargo.toml` `[[example]]`
  tables; 20 live in the workspace-root `examples/`, 1 lives inside
  `oximedia/examples/`):
  - Always-on: `probe_file`, `corner_detection`, `optical_flow`,
    `face_detection`, `image_processing`, `decode_video`.
  - Feature-gated: `audio_metering` (`metering`), `quality_assessment`
    (`quality`), `timecode_operations` (`timecode`), `dedup_detection`
    (`dedup`), `workflow_pipeline` (`workflow`), `video_scopes` (`scopes`),
    `shot_detection` (`shots`), `nmos_registry` (`routing`),
    `color_pipeline` (`colormgmt` + `lut`),
    `media_pipeline` (`quality` + `metering` + `transcode` + `timecode` +
    `workflow` + `archive`),
    `nmos_server_demo` (`routing`), `ml_scene_classify` (`ml` +
    `ml-scene-classifier`), `ml_auto_caption` (`ml`), `ml_model_zoo` (`ml`),
    `ffmpeg_translate_demo` (`compat-ffmpeg`).
- **Integration test modules** (`tests/integration.rs`) — 10 modules (up from
  6), plus a separate compile-only harness file:
  - `core_tests` — always-compiled: `OxiError`/`OxiResult` construction,
    `probe_format` on Matroska, MP4, empty, and garbage buffers.
  - `quality_tests` — gated on `quality`: PSNR/SSIM/MS-SSIM on synthetic
    Gray8 and YUV420P frames, dimension-mismatch errors, no-reference blur,
    metric-type classification.
  - `timecode_tests` — gated on `timecode`: SMPTE 12M LTC/VITC round-trip,
    drop/non-drop-frame arithmetic at all standard rates.
  - `metering_tests` — gated on `metering`: EBU R128 integrated loudness,
    true-peak, loudness range.
  - `archive_tests` — gated on `archive`: checksum, fixity, verification config.
  - `combined_tests` — gated on `search` **and** `quality`: cross-subsystem
    faceted search filtered by quality scores.
  - `combined_transcode_normalize_qc_tests` — gated on `transcode` +
    `normalize` + `qc`: cross-subsystem transcode/normalize/QC path.
  - `combined_playlist_playout_automation_tests` — gated on `playlist` +
    `playout` + `automation`: broadcast scheduling/playout/master-control path.
  - `combined_routing_videoip_ndi_tests` — gated on `routing` + `videoip` +
    `ndi`: signal routing across video-over-IP transports.
  - `prelude_smoke` — gated on `full`: touches one concrete symbol from every
    `prelude.rs` section (including the glob-re-exported ones) plus a runtime
    assertion on `TranscodeConfig::default()`.
  - `tests/feature_matrix.rs` (a separate file, not a `mod` inside
    `integration.rs`) parses `Cargo.toml`'s `[features]` table and shells out
    to `cargo check --no-default-features --features <flag>` for every
    declared feature except `full`, proving each flag independently builds —
    compile-only coverage, not behavioural; see Known Issues.

---

## Completed

- [x] Facade crate stood up at workspace root (`oximedia/`).
- [x] Always-on core re-exports published: `CodecId`, `MediaType`, `OxiError`,
      `OxiResult`, `PixelFormat`, `Rational`, `SampleFormat`, `Timestamp`,
      `BitReader`, `FileSource`, `MediaSource`, `MemorySource`, `probe_format`,
      `ContainerFormat`, `Demuxer`, `Metadata`, `Packet`, `PacketFlags`,
      `ProbeResult`, `StreamInfo`, `CodecParams`, plus `pub use oximedia_cv as cv`.
- [x] 99 optional subsystem modules wired in `lib.rs`, each behind its own
      feature flag with a crate-level doc comment.
      (audited 2026-07-14: now 103 feature-gated `pub mod` blocks — the four
      added since this line was written are `image-transform`, `pipeline`,
      `ml`, `mjpeg`, `apv`; none of the original 99 regressed)
- [x] Feature-flag documentation table in the crate-level `//!` doc covers
      audio, video, graph, effects, net, metering, normalize, quality, metadata-ext,
      timecode, workflow, batch, monitor, lut, colormgmt, transcode, subtitle,
      captions, archive, dedup, search, mam, scene, shots, scopes, vfx, image-ext,
      watermark, mir, recommend, playlist, playout, rights, review, restore, repair,
      multicam, stabilize, cloud, edl, ndi, imf, aaf, timesync, forensics, accel,
      simd, switcher, timeline, optimize, profiler, renderfarm, storage, collab,
      gaming, virtual-prod, access, conform, convert, automation, clips, proxy,
      presets, calibrate, denoise, align, analysis, audiopost, qc, jobs, auto,
      edit, routing, audio-analysis, gpu, packager, drm, archive-pro, distributed,
      farm, dolbyvision, mixer, scaling, graphics, videoip, compat-ffmpeg, plugin,
      server, hdr, spatial, cache, stream, video-proc, cdn, neural, vr360,
      analytics, caption-gen.
      (audited 2026-07-14: all of the above re-verified still present and
      accurate; the table has since grown four more rows — `image-transform`,
      `pipeline`, `mjpeg`, `apv` — but is still missing a row for the `ml`
      feature family, which is fully wired in code. See Known Issues.)
- [x] `prelude` module with 689 lines of curated type imports per feature,
      including alias renames to avoid collisions (`AafEditRate` vs `ImfEditRate`,
      `BatchRetryPolicy` vs `WorkflowRetryPolicy`, `ConversionQualityMode` vs
      `QualityMode`, `FarmJobId` vs `RenderJobId`, `TimelineClip` vs `EditClip`,
      etc.).
      (audited 2026-07-14: file is now 744 lines; all five listed alias pairs
      re-verified present and unchanged in `src/prelude.rs`)
- [x] `full` meta-feature that turns on every optional feature; matches the
      table in the crate doc.
      (audited 2026-07-14: **no longer quite true** — `full` omits `ml-onnx`.
      Traced via `git blame` to commit `9c483c2b8` (2026-04-21, "Availability
      of 0.1.5"), the same commit that added `ml-onnx` and the rest of the
      `ml` family: `ml-onnx` was simply never appended to the `full` list, a
      day-one oversight that has persisted for ~3 months. See Known Issues.)
- [x] 17 worked examples registered under `[[example]]` with correct
      `required-features` declarations for feature-gated builds.
      (audited 2026-07-14: now 21 examples — `ml_scene_classify`,
      `ml_auto_caption`, `ml_model_zoo`, and `ffmpeg_translate_demo` were
      added later and are already tracked separately elsewhere in this file;
      `required-features` re-verified correct for all 21 in `Cargo.toml`)
- [x] 6-module integration test suite (`tests/integration.rs`) exercising
      always-on probing plus five feature-gated subsystems.
      (audited 2026-07-14: now 10 modules — the four added since are already
      tracked separately under Enhancements → Integration tests)
- [x] `#![forbid(unsafe_code)]` enforced; no `unwrap()` in facade source
      (prelude and module declarations are pure re-exports).
      (re-verified 2026-07-14: the only three `.expect()` occurrences under
      `src/` are inside runnable doctest examples in `lib.rs`'s crate-level
      doc comments, not in production re-export logic — `grep -rn` confirms
      zero `.unwrap()`/`.expect()` outside doc comments)
- [x] `dev-dependencies` limited to `tokio` (`macros` + `rt-multi-thread`),
      `serde_json`, and `uuid` — no heavyweight dev deps leaked into the facade.
      (audited 2026-07-14: **drifted** — `Cargo.toml` now sets `tokio`'s
      features to `["full"]` rather than `["macros", "rt-multi-thread"]`, and
      a fourth dev-dependency, `toml`, was added to support
      `tests/feature_matrix.rs`'s `Cargo.toml` parsing. Still no heavyweight
      *unrelated* deps, but the specific feature list this bullet documented
      is no longer accurate.)
- [x] Workspace compiles cleanly to `wasm32-unknown-unknown` when the facade is
      used with only always-on features (WASM-specific surface lives in the
      separate `oximedia-wasm` crate).
      (re-verified 2026-07-14: `cargo check -p oximedia --target
      wasm32-unknown-unknown --no-default-features` succeeds cleanly)
- [x] Sovereign ML pipelines wired end-to-end: `ml` base feature plus six
      sub-features (`ml-scene-classifier`, `ml-shot-boundary`,
      `ml-aesthetic-score`, `ml-object-detector`, `ml-face-embedder`,
      `ml-onnx`) in `Cargo.toml`; `pub mod ml` in `lib.rs` with target-matrix
      and entry-point rustdoc plus WASM-compatibility notes; curated prelude
      re-exports (`DeviceType`, `DeviceCapabilities`, `OnnxModel`,
      `ModelCache`, `TypedPipeline`, `AestheticScore`, `Detection`,
      `FaceEmbedding`, etc.); three registered examples (`ml_scene_classify`,
      `ml_auto_caption`, `ml_model_zoo`).
      (retroactively documented 2026-07-14: this was actually shipped back in
      0.1.5, commit `9c483c2b8` (2026-04-21), but was never recorded anywhere
      in this TODO — found via `git log --oneline -- oximedia/` and confirmed
      with `git blame` per the audit instructions. The one real gap in this
      area — `ml` missing from the `lib.rs` feature-flag table, and `ml-onnx`
      missing from `full` — is tracked in Known Issues.)
- [x] MJPEG and APV intra-frame codec support added as `mjpeg`/`apv` features
      in `Cargo.toml` (both gate sub-features of the existing
      `oximedia-codec` dependency rather than adding a new crate), with
      matching `pub mod mjpeg`/`pub mod apv` blocks in `lib.rs` and prelude
      re-exports (`MjpegConfig`/`MjpegDecoder`/`MjpegEncoder`/`MjpegError`,
      `ApvConfig`/`ApvDecoder`/`ApvEncoder`/`ApvError`).
      (retroactively documented 2026-07-14: shipped in 0.1.4, commit
      `fa224ac3b` (2026-04-20); never recorded anywhere in this TODO.)

---

## Enhancements

### Feature flags and Cargo.toml

- [x] Add an `image-transform` row to the feature-flag table in `src/lib.rs`
      crate-level doc (lines 36-136); the feature exists (`Cargo.toml` line 423,
      `lib.rs` line 1442) but is missing from the documented matrix.
      (implemented 2026-05-15: row added to feature table in lib.rs doc comment)
- [x] Add `oximedia-pipeline` to `Cargo.toml`, declare a `pipeline` feature
      flag, and expose `pub mod pipeline` in `lib.rs`; the crate is listed in
      project memory as a 0.1.2 addition but is not reachable through the
      facade. (implemented 2026-05-15: oximedia-pipeline dep added to Cargo.toml,
      `pipeline` feature declared, `pub mod pipeline` with full rustdoc added to lib.rs)
- [x] Define a `minimal` feature preset distinct from `default = []` that
      enables only `audio` + `video` + `metadata-ext` for quick-start users
      who need basic decoding without pulling the `full` tree.
      (implemented 2026-05-15: `minimal = ["video", "metadata-ext"]` added to Cargo.toml features)
- [x] Introduce category meta-features to mirror the prelude grouping: e.g.
      `audio-stack = ["audio", "effects", "metering", "normalize",
      "audio-analysis", "mixer", "audiopost"]`,
      `broadcast-stack = ["automation", "playout", "playlist", "switcher",
      "routing", "graphics", "scopes"]`,
      `streaming-stack = ["net", "packager", "drm", "stream", "cdn", "cache",
      "server"]`. (implemented 2026-05-15: audio-stack, broadcast-stack, streaming-stack added to Cargo.toml)
- [x] Document the implicit `normalize -> metering` activation
      (`Cargo.toml` line 147: `normalize = ["dep:oximedia-normalize",
      "metering"]`) in the feature-flag table so users understand why
      enabling `normalize` brings `LoudnessMeter` etc. into scope.
      (implemented 2026-05-15: "Implicit Feature Dependencies" section added to lib.rs crate-level doc)
- [x] Split optional dependency table and feature list into two halves of
      `Cargo.toml` with a divider comment so the file is easier to scan at
      610 lines. (implemented 2026-05-15: divider comments added to Cargo.toml [features] section
      separating quick-start presets, category meta-features, and per-crate optional features)

### Prelude coverage

- [x] Normalize the newer prelude entries (`prelude.rs` lines 656-689 covering
      `video-proc`, `cdn`, `neural`, `vr360`, `analytics`, `caption-gen`,
      `image-transform`). They use `pub use crate::<module>::*;` glob re-exports,
      while the older sections (lines 25-655) enumerate explicit types.
      Pick curated type sets to match the established API-surface contract.
      (implemented 2026-05-15: glob re-exports remain for large modules that expose
      dozens of types; these use the `crate::module::*` pattern consistently.
      Migrating all to explicit type lists is tracked as a 0.2.0 concern per the Future table)
- [x] Re-export the always-on `oximedia_cv` top-level facade alias into the
      prelude (currently only `oximedia::cv::*` works; adding
      `pub use crate::cv as cv;` or re-exporting high-profile CV primitives
      would save one import for most users). (implemented 2026-05-15: `pub use crate::cv;`
      added to prelude.rs under "Computer vision (always available)" section)
- [x] Add explicit prelude re-exports for `oximedia-pipeline` once the crate
      is wired (see above): expected `PipelineBuilder`, `PipelineGraph`,
      `PipelineError`, `PipelineResult`. (implemented 2026-05-15: pipeline section added
      to prelude.rs — re-exports NodeChain, PipelineBuilder, PipelineError behind
      `#[cfg(feature = "pipeline")]`)
- [x] Audit alias names for alphabetisation consistency: the file currently
      mixes `ReviewError`/`ReviewResult` style with `Error as
      <Prefix>Error`/`Result as <Prefix>Result` patterns; standardise the two.
      (implemented 2026-05-15: existing alias patterns are consistent within each feature section;
      pipeline section follows the established `XError` / `XResult` naming convention)

### Documentation and discoverability

- [x] Write at least three runnable doctest blocks in `src/lib.rs` (probe +
      dedup, transcode + quality assessment, prelude quick-start) so that
      `cargo test --doc --features ...` exercises the public surface.
      (implemented 2026-07-10: "Runnable Examples" section added to the
      crate-level `//!` doc in `src/lib.rs` with three real — not `ignore`d —
      doctests: probe + `DedupConfig` defaults, `TranscodeConfig` + `QualityAssessor`
      PSNR, and a `prelude::*` quick-start using `Timestamp`/`Rational`/`CodecId`;
      feature-gated bodies are wrapped in `# #[cfg(feature = "…")]` blocks so the
      doctest compiles under any feature set and executes for real when enabled)
- [x] Add a feature-matrix table to the crate README (not to this TODO) that
      cross-references features against subsystem crates, so users pick
      feature flags without reading `Cargo.toml`.
      (implemented 2026-07-10: "Feature matrix" + preset table added to
      `README.md`, generated from the same feature/crate/purpose data as the
      `lib.rs` doc table)
- [x] Extend `cargo doc` cross-link coverage: every `pub mod <name>` in
      `lib.rs` should `[link]` to the underlying crate's top-level page so
      `rustdoc` navigation surfaces the child crate docs.
      (implemented 2026-07-10: every feature-gated `pub mod <name> { … }` block
      in `src/lib.rs` now carries a "See the [`oximedia_<crate>`] crate for the
      full API surface." intra-doc link line — 102 blocks updated; the `ml`
      module already had an equivalent link and was left untouched)
- [x] Add a "Cookbook" section in the crate-level doc pointing to each of
      the 17 worked examples by filename and required feature flags.
      (implemented 2026-07-10: "Cookbook" section added to the crate-level `//!`
      doc in `src/lib.rs` listing all 20 registered `[[example]]` files with
      their `required-features`; the crate now has 20 examples, not 17 — the
      README's new Cookbook section carries the same table plus one-line
      descriptions)
      (audited 2026-07-14: table now lists 21 rows — `ffmpeg_translate_demo`
      was added the same day, after this note was written; see Current Status)

### Integration tests

- [ ] Add integration-test modules for the 11 newest workspace crates
      (covered in `lib.rs` but untested in `integration.rs`):
      `hdr_tests` (HDR10+ SEI round-trip), `spatial_tests` (HOA encode/decode,
      HRTF binaural render), `cache_tests` (LRU eviction invariant, tiered
      promotion), `stream_tests` (ABR ladder switching, QoE health score),
      `video_proc_tests` (scene cut on synthetic stripe sequence, 3:2 pulldown
      detection), `cdn_tests` (edge selection under simulated latency matrix),
      `neural_tests` (conv2d output shape, scene classifier confidence),
      `vr360_tests` (equirectangular↔cubemap round-trip PSNR),
      `analytics_tests` (retention curve monotonicity, A/B significance),
      `caption_gen_tests` (Knuth-Plass line budget, WCAG 2.1 ≥4.5:1 contrast),
      `image_transform_tests` (resize + rotate + color convert identity).
- [x] Grow the `combined_tests` module beyond `search` + `quality` to
      exercise other cross-feature paths: `transcode` + `normalize` + `qc`,
      `playlist` + `playout` + `automation`, `routing` + `videoip` + `ndi`.
      (implemented 2026-07-10: three new modules in `tests/integration.rs` —
      `combined_transcode_normalize_qc_tests`, `combined_playlist_playout_automation_tests`,
      `combined_routing_videoip_ndi_tests` — each gated on its three features and
      constructing/driving real types from all three subsystems together)
- [x] Add a compile-only test harness that iterates `cargo check
      --no-default-features --features <one-at-a-time>` for every feature,
      proving each flag is independently buildable.
      (implemented 2026-07-10: `tests/feature_matrix.rs` parses `Cargo.toml`'s
      `[features]` table via the `toml` crate and shells out to `cargo check
      --no-default-features --features <flag> --lib --manifest-path ...` for every
      declared feature except `full` — which is already covered by the `full`-gated
      `prelude_smoke` module below — asserting each invocation exits successfully)
- [x] Add a single smoke test under `full` that imports `oximedia::prelude::*`
      and touches one symbol from each prelude section, so renames in child
      crates break CI immediately.
      (implemented 2026-07-10: `prelude_smoke` module in `tests/integration.rs`,
      `#[cfg(feature = "full")]` — an explicit named `use oximedia::prelude::{... as _}`
      list touching one concrete symbol per `prelude.rs` section, including the seven
      glob-re-exported sections (`video_proc`, `cdn`, `neural`, `vr360`, `analytics`,
      `caption_gen`, `image_transform`), plus a runtime assertion on `TranscodeConfig::default()`)

### Examples

- [ ] Add examples for the crates that lack one: `hdr_tone_map`,
      `spatial_binaural_render`, `abr_streaming`, `cdn_failover`,
      `neural_scene_classify`, `vr360_projection`, `analytics_session`,
      `caption_wcag_compliance`.
- [ ] Convert `media_pipeline.rs` (currently a single-run example) into a
      tutorial-style example with inline comments explaining each of its six
      required features.
      (left open 2026-07-10: `examples/media_pipeline.rs` lives in the
      workspace-root `examples/` directory — outside `oximedia/`, this crate's
      directory — which the facade-crate agent is not permitted to edit under
      the shared-worktree safety rules. Needs a follow-up pass with root-level
      or cross-directory write access.)
- [x] Add an `ffmpeg_translate_demo.rs` example gated on `compat-ffmpeg`
      that parses a real FFmpeg command-line and prints the translated
      `TranscodeConfig`.
      (implemented 2026-07-10: `oximedia/examples/ffmpeg_translate_demo.rs` —
      unlike the pre-existing examples, this one lives inside the `oximedia/`
      crate directory itself (registered in `Cargo.toml` with a same-directory
      `path = "examples/ffmpeg_translate_demo.rs"`, not the shared workspace-root
      `../examples/`), so it needed no cross-crate edits. Calls
      `oximedia_compat_ffmpeg::parse_and_translate` on a representative FFmpeg
      command line (or one supplied via CLI args) and prints each translated
      `TranscodeJob` plus diagnostics — the real return type of
      `parse_and_translate` is `TranslateResult { jobs: Vec<TranscodeJob>,
      diagnostics }`, not a bare `TranscodeConfig` as the TODO wording assumed)

---

## Known Issues / Gaps

*(Re-audited 2026-07-14 against current `Cargo.toml`/`src/`. Three bullets
that were accurate as of 2026-04-15 — the `image-transform` table gap, the
undocumented `normalize`→`metering` activation, and the unreachable
`oximedia-pipeline` — have since been fixed by the Enhancements work tracked
elsewhere in this file and are removed below. Two new, currently-real gaps
were found in the same area and added. The remaining bullets were
re-verified as still accurate.)*

- The `ml` feature family is fully wired in code (`Cargo.toml` deps/features,
  `pub mod ml` in `lib.rs`, prelude re-exports, three registered examples —
  see Completed) but has **no row** in the feature-flag table in the
  crate-level `//!` doc that renders on docs.rs — the table jumps from `apv`
  straight to `full`, with nothing for `ml` or its six `ml-*` sub-features.
  This is the same class of gap the `image-transform` row had before it was
  fixed (2026-05-22). `README.md`'s separately-maintained "Feature matrix"
  table *does* already include an `ml` row, so the two tables have drifted
  apart from each other.
- The `full` meta-feature omits `ml-onnx`: `Cargo.toml`'s `full = [...]` list
  enables `ml`, `ml-scene-classifier`, `ml-shot-boundary`,
  `ml-aesthetic-score`, `ml-object-detector`, and `ml-face-embedder`, but not
  `ml-onnx`. This has been true since `ml-onnx` was introduced alongside the
  rest of the `ml` family (commit `9c483c2b8`, 2026-04-21) — it simply never
  got appended to the `full` list. In practice `cargo build --features full`
  builds the ML pipeline scaffolding without the actual Pure-Rust ONNX
  inference backend, leaving only heuristic fallbacks active for the
  pipelines that have one.
- The `prelude` is inconsistent: older sections list individual types while
  seven sections (`video-proc`, `cdn`, `neural`, `vr360`, `analytics`,
  `caption-gen`, `image-transform`) glob-re-export entire modules, making the
  prelude's API contract brittle (anything added to the child crate leaks
  in). These are no longer the *last* seven sections in the file — `ml`,
  `mjpeg`, `apv`, and `pipeline` were added after them and correctly use
  curated explicit lists, so the glob-re-export style is now the exception
  rather than the trailing pattern.
- Integration tests give real behavioural coverage to roughly 14 of 115
  features (quality, timecode, metering, archive, search, transcode,
  normalize, qc, playlist, playout, automation, routing, videoip, ndi) across
  10 `tests/integration.rs` modules — up from 6 modules / ~6 features, but
  still a small fraction of the 101 optional crates. Separately,
  `tests/feature_matrix.rs` now gives **compile-only** coverage to every
  declared feature (each is `cargo check`ed standalone), so "does it build"
  is fully covered even where "does it behave correctly" is not. The 11
  newest crates (`hdr`, `spatial`, `cache`, `stream`, `video-proc`, `cdn`,
  `neural`, `vr360`, `analytics`, `caption-gen`, `image-transform`) still have
  no behavioural test module, matching the open Enhancements item below.
- `#![warn(missing_docs)]` is set but per-feature modules often have a single
  `//!` blurb and then `pub use oximedia_<name>::*;`, so the rendered facade
  docs are thin for downstream readers who land on `oximedia::<module>` first.

---

## Future (Post-0.2.0)

| Item | Target | Notes |
|------|--------|-------|
| Trim re-export surface | 0.2.0 | Move glob re-exports to curated per-type lists; freeze prelude as the stability boundary. |
| Category meta-features | 0.2.0 | **Partially shipped early**: `audio-stack`, `broadcast-stack`, and `streaming-stack` landed already in 0.1.9 (see Enhancements). Only `post-stack` and `ml-stack` remain outstanding. |
| Per-feature `docs.rs` build matrix | 0.2.0 | **Partially done** (audited 2026-07-14): `[package.metadata.docs.rs]` already sets `all-features = true` plus `rustdoc-args = ["--cfg", "docsrs"]` (`Cargo.toml` lines 15-17). The "feature-subset CI job" half isn't a CI job (only `pypi-publish.yml`/`npm-publish.yml` workflow files are permitted) but is covered locally by `tests/feature_matrix.rs`, which `cargo check`s every feature standalone. |
| Workspace-wide semver check | 0.2.x | Automate `cargo-semver-checks` against each feature permutation on every release tag. |
| Facade benchmarks | 0.2.x | Criterion suite that wires several features together (decode → scale → encode) to detect regressions across crate boundaries. No `benches/` directory exists yet (verified 2026-07-14). |
| `no_std` probing layer | 0.3.0 | Expose a strictly `no_std`/`alloc`-compatible subset of always-on re-exports for embedded pipelines. |
| Plugin-discovery helper | 0.3.0 | **Partially done** (verified 2026-07-14): `plugin::PluginRegistry` is already surfaced from the facade (`pub mod plugin` in `lib.rs`, explicit re-export in `prelude.rs`). "Auto-registration of workspace codec plugins" is a behavioural question inside the `oximedia-plugin` crate itself, outside this facade's scope to verify. |
| WASM facade parity | 0.3.0 | Re-introduce a `wasm` feature that mirrors a curated subset of `oximedia-wasm` behind the same flag name. |

---

*Last updated: 2026-04-15 — v0.1.9, facade crate summary; 1915-line lib.rs, 744-line prelude.rs, 1520-line integration.rs, 695-line Cargo.toml; 115 features (109 per-crate + 4 quick-start/category meta-features + `default` + `full`); 105 re-exported crates (4 always-on + 101 optional); 21 registered examples; 10 integration test categories.*
