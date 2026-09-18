# Changelog — oximedia-cli

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.7] - 2026-05-06

### Added
- **Per-subcommand `--help` smoke harness** (`tests/cli_help_per_command.rs`) —
  exercises every variant of `Commands` (90 + `convert`/`ff` aliases = 92
  tests) by spawning `oximedia <subcmd> --help` via `assert_cmd` and asserting
  exit 0.
- **Listing-style smoke tests** (`tests/cli_smoke_listings.rs`) — 10 end-to-end
  smokes for `preset list`, `plugin list`, `loudness standards`, `quality
  list`, `info`, `version`, `doctor` (plain + `--json`), `man-page`, and
  `completions bash`.
- **Root-level smoke tests** (`tests/cli_help.rs`, rewritten) — 5 tests for
  `--help`, `--version`, no-args usage, and the global flag surface.
- **`oximedia-ff` golden translation suite** — 71 JSON fixtures under
  `tests/ff_golden/*.json` driven by `tests/ff_golden.rs`. Covers 12 patent
  substitutions, 4 direct codecs + copy, 8 quality flags, 15 single-chain
  filters, 3 `filter_complex` graphs, 5 stream selectors, 4 seek/duration
  flags, 3 metadata/movflags, 2 two-pass phases, 3 hwaccel backends, 4
  loglevel/banner flags, 2 format aliases, 1 overwrite, 2 combo invocations,
  and 2 hard-error cases. Partial-match expectations keep the suite
  resilient to translator field additions.
- **`probe --format json` snapshot tests** (`tests/probe_format_json_snapshot.rs`)
  — 8 snapshots under `tests/probe_snapshots/*.json` covering 4 containers
  (MP4 / WebM / Ogg / FLAC) and the `--streams` / `--chapters` / `--metadata`
  / all-flags schemas. Tiny in-process magic-byte fixtures are written into
  per-test `tempfile::TempDir`; output is normalized for path/size/f32
  confidence before comparison. Re-baseline with
  `OXIMEDIA_UPDATE_SNAPSHOTS=1`.
- **`--json` flag on the `oximedia-ff` binary** (`src/bin/oximedia-ff.rs`) —
  emits the structured `TranslateResult` (jobs, diagnostics, warnings,
  unsupported) as JSON on stdout and exits without execution. Used as the
  surface for the golden suite.
- **`examples/` directory** — 8 reference shell-script workflows plus a
  README: `dailies-ingest.sh`, `quality-check.sh`, `abr-package.sh`,
  `loudness-normalize.sh`, `forensics-investigate.sh`, `restore-degraded.sh`,
  `live-broadcast.sh`, `cv2-pipeline.sh`. All scripts pass `bash -n` and use
  only verified-wired flags.

### Changed
- **`commands.rs` (1480 lines) split** into `commands/` module — `mod.rs`
  (1118 lines, the single 90-variant `Commands` enum + `parse_key_val`),
  `infrastructure.rs` (102, `MonitorCommand`), `video.rs` (118,
  `RestoreCommand`), `subtitle.rs` (137, `CaptionsCommand`), and `compat.rs`
  (75, `PresetCommand`). The top-level enum stays in one file because
  `clap_derive::Subcommand` requires all variants in a single definition;
  the four nested sub-enums were the only relocatable units. Import paths
  (`crate::commands::Foo`) are preserved via re-exports so `handlers.rs` is
  unchanged.
- **`oximedia-cv2 --list-constants`** now iterates the auto-generated
  `LIST_CONSTANTS` table from the `oximedia-compat-cv2` build.rs reflection
  pass; the previous hand-maintained constant table was deleted, removing a
  maintenance hazard between `src/constants.rs` and the binary.
- Added `serde` and `serde_json` to `[dev-dependencies]` to support the
  `--json` golden runner.

### Validated
- `cargo nextest run -p oximedia-cli --all-features` passes (647+ tests).
- `cargo clippy -p oximedia-cli --all-features --all-targets -- -D warnings`
  is clean.
- `cargo doc --no-deps -p oximedia-cli` is clean.

### Notes
- This entry covers only the `oximedia-cli` crate. The companion changes
  to `oximedia-compat-cv2` (build.rs constants reflection + ORB BRIEF
  pipeline + Hamming brute-force matcher) are recorded under that crate's
  own `CHANGELOG.md`.

### Run 4 (2026-05-06)

#### Added
- **`oximedia doctor --full` flag** (`src/doctor_cmd.rs`, `tests/doctor_full.rs`) —
  three new diagnostic sections appended only when `--full` is passed: codec
  capability matrix, `OXIMEDIA_PLUGIN_PATH` directory validation (per-path
  exists/readable/dylib-count), and `OXICUDA_HOME` probe (path + version.txt
  parse). Default-output JSON schema unchanged so existing consumers stay
  compatible.
- **10 new `oximedia-cv2` subcommands** (`src/bin/oximedia-cv2.rs`,
  `tests/cv2_ops.rs`, `tests/cv2_orb.rs`, `tests/cv2_dnn.rs`) — `flip`, `rotate`,
  `sobel`, `laplacian`, `adaptive-threshold`, `morphology-ex`, `fast-corners`,
  `harris-corners`, `orb-detect`, and `dnn-forward` (gated `#[cfg(feature = "dnn")]`).
  Surfaces work that already lives in `oximedia-compat-cv2`. 14 new tests.
- **`dnn = ["oximedia-compat-cv2/dnn"]` feature passthrough** (`Cargo.toml`) —
  enables the cv2 dnn module + the binary's `dnn-forward` subcommand when built
  with `--features dnn`.

#### Changed
- **`handlers.rs` (927 lines) split into `handlers/` submodule** — `mod.rs`
  (re-exports), `logging.rs` (`init_color`/`init_logging`), `inspect.rs`
  (`probe_file`), `dispatch.rs` (monitor/restore/captions thin dispatchers),
  `preset_ui.rs` (heavyweight preset rendering), `reference.rs` (`show_info`,
  `show_version`, `rustc_version_str`). Zero behavior change; all existing
  imports preserved via `pub use` re-exports.

#### Validated
- `cargo nextest run -p oximedia-cli --all-features` — 679 tests pass.
- `cargo nextest run -p oximedia-cli --features dnn` — 25 cv2-dnn tests pass.
- `cargo nextest run -p oximedia-cli` (no features) — 23 default tests pass.
- `cargo clippy -p oximedia-cli --all-features --all-targets -- -D warnings` —
  clean.

#### Notes
- The companion `oximedia-compat-cv2` Run 4 changes (new `dnn` module, ORB
  follow-ups: `BFMatcher`/`knn_match`/mask support) are recorded under that
  crate's own `CHANGELOG.md`.
- Refinement 5 (real-ffmpeg golden) closed as WONT-FIX — depending on system
  ffmpeg violates the project's Pure Rust Policy. The 71-fixture
  `TranslateResult` golden suite from Run 3 Slice B is sufficient.
