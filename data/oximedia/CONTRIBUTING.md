# Contributing to OxiMedia

Thank you for your interest in contributing to OxiMedia — the Sovereign Media
Framework, a patent-free, memory-safe, Pure Rust reconstruction of FFmpeg and
OpenCV. This document explains how the workspace is organized, what is
expected of a change, and how to get it merged.

By participating in this project you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Table of Contents

- [Ways to Contribute](#ways-to-contribute)
- [Project Layout](#project-layout)
- [How to Build](#how-to-build)
- [How to Run Tests](#how-to-run-tests)
- [Coding Standards](#coding-standards)
- [Branch Policy](#branch-policy)
- [Commit and Pull Request Conventions](#commit-and-pull-request-conventions)
- [License and Contribution Terms](#license-and-contribution-terms)

## Ways to Contribute

- **Bug reports** — open a GitHub issue using the "Bug Report" template.
  Include a minimal reproduction, the OxiMedia version/commit, and OS/Rust
  toolchain versions.
- **Feature requests** — use the "Feature Request" template. Explain the use
  case, not just the API you'd like.
- **Codec / container work** — `docs/codec_status.md` is the single source of
  truth for what is honestly *Verified*, *Functional*, *Bitstream-parsing
  only*, or *Experimental* per codec. It also lists effort estimates
  (small/medium/large/specialist) for closing each gap — a good place to find
  a well-scoped task.
- **Documentation** — module-level rustdoc, the crate `README.md` files under
  `crates/*/`, and the guides under `docs/` (e.g. `docs/ml_guide.md`,
  `docs/simd_dispatch.md`, `docs/rate_control.md`) all welcome improvements.
- **Refactors** — if you notice a source file creeping past ~2000 lines, a
  split is welcome (see [Coding Standards](#coding-standards)).

Before starting significant work (a new codec, a new crate, an architectural
change), please open an issue first to discuss the approach — it avoids
wasted effort on both sides.

## Project Layout

OxiMedia is a single Cargo workspace (`resolver = "2"`) rooted at this
repository, with over 100 member crates under `crates/`, plus:

- `oximedia/` — the facade crate. It re-exports the full ecosystem behind
  Cargo features (`hdr`, `spatial`, `cache`, `stream`, `video-proc`, `cdn`,
  `neural`, `vr360`, `analytics`, `caption-gen`, `full`, ...). Most
  downstream users depend on `oximedia = { version = "...", features = [...] }`
  rather than on individual `oximedia-*` crates directly.
- `oximedia-cli/` — the `oximedia` command-line binary.
- `oximedia-wasm/` — the WebAssembly bindings (`wasm32-unknown-unknown`,
  synchronous API surface only, no `std::fs`).
- `crates/oximedia-py/` — the PyO3 Python extension module. It is
  intentionally excluded from `default-members` in the root `Cargo.toml`
  because its cdylib output name collides with the CLI and facade crate
  builds; build it explicitly with `maturin` or
  `cargo build -p oximedia-py`.
- `crates/oximedia-core`, `crates/oximedia-codec`, `crates/oximedia-container`,
  etc. — the domain crates (codecs, containers, CV, streaming, DRM, and so
  on). Shared workspace metadata (version, edition, license, MSRV, authors,
  repository) lives once in the root `Cargo.toml` under
  `[workspace.package]` and is inherited per crate via `x.workspace = true`.
- `benches/` — Criterion benchmark targets.
- `fuzz/` — cargo-fuzz targets (excluded from the main workspace via
  `exclude = ["fuzz"]`; build/run it from within `fuzz/`).
- `examples/` — runnable example programs.
- `compat/` — FFmpeg/OpenCV compatibility shim crates.

## How to Build

The MSRV is pinned via `rust-version.workspace = true` (currently the value
in `[workspace.package]` in the root `Cargo.toml`); the toolchain used by CI
tooling locally is `stable` with the `rustfmt` and `clippy` components (see
`rust-toolchain.toml`).

```bash
# Build the default workspace members (excludes oximedia-py, see above)
cargo build

# Build everything with every feature enabled, per crate
cargo build --workspace --all-features

# Build just the facade crate with a feature bundle
cargo build -p oximedia --features full

# Build the CLI
cargo build -p oximedia-cli

# Format and lint (must be clean — see Coding Standards)
cargo fmt --all
cargo clippy --workspace --all-features --all-targets
```

`unsafe_code` is `deny` at the workspace level
(`[workspace.lints.rust]` in the root `Cargo.toml`). A small number of
crates that need SIMD intrinsics, `mmap`, or lock-free primitives
(currently `oximedia-accel`, `oximedia-gpu`, `oximedia-simd`,
`oximedia-routing`, `oximedia-switcher`) opt back in to `unsafe_code = "allow"`
at the crate level, and `oximedia-automation` / `oximedia-presets` tighten it
further to `forbid`. If your change needs `unsafe`, it almost certainly
belongs in one of the already-gated crates — do not casually loosen the lint
elsewhere; open an issue first if you believe a new crate genuinely needs it.

## How to Run Tests

Tests run under [cargo-nextest](https://nexte.st/):

```bash
cargo install cargo-nextest --locked   # one-time setup
cargo nextest run --workspace --all-features
```

Plain `cargo test` also works, but nextest is what the project's own
development workflow assumes, and it is what `.config/nextest.toml`
configures.

`.config/nextest.toml` defines a `serial-latency` test group
(`max-threads = 1`) for tests that assert on wall-clock timing budgets or do
heavy mmap/I/O and would produce false-positive failures if run concurrently
with the rest of the suite (e.g. `perf_budget`/`latency_budget` tests,
`oximedia-repair` deep-scan and mmap-scan tests, the
`oximedia-videoip` precise-sleeper spin test, the
`oximedia-transcode` EBU R128 conformance tests, and a handful of other
named tests). If you add a new test that measures wall-clock time or
depends on being the sole user of a shared resource, add a matching
`[[profile.default.overrides]]` filter to `.config/nextest.toml` rather than
hand-tuning thresholds to survive contention — see the existing entries for
the pattern and the rationale comments above each one.

There is currently no automated test CI workflow in `.github/workflows/`
(only `pypi-publish.yml` and `npm-publish.yml`, which handle Python/npm
package publishing). Until that changes, **please run `cargo fmt --all`,
`cargo clippy --workspace --all-features --all-targets`, and
`cargo nextest run --workspace --all-features` locally before opening a
PR** — reviewers will expect a clean run and may ask for the output.

For temporary files needed by tests, use `std::env::temp_dir()` (or the
`tempfile` crate where the workspace already depends on it) rather than
hardcoding a path — this keeps tests portable across CI runners and
contributors' machines.

## Coding Standards

These are enforced by convention and code review, not (yet) by CI, so please
hold yourself to them:

- **Pure Rust by default.** OxiMedia's entire reason to exist is to be a
  memory-safe, dependency-light alternative to C/C++/Fortran-based FFmpeg
  and OpenCV. Default features must not pull in a C/C++/Fortran toolchain
  dependency. If a C dependency is genuinely unavoidable (e.g. certain
  hardware vendor SDKs), it must sit behind an explicit, non-default Cargo
  feature.
  - **Use the COOLJAPAN ecosystem instead of these forbidden crates:**
    - `zip`, `flate2`, `zstd`, `bzip2`, `lz4`, `tar`, `snap`, `brotli`,
      `miniz_oxide` → use `oxiarc-archive`, `oxiarc-deflate`, `oxiarc-lz4`,
      `oxiarc-zstd` (the workspace already depends on all four).
    - `rusqlite` → use `oxisql-sqlite-compat` (workspace dep) /
      `oxisql-core`.
    - `bincode` → use `oxicode`.
    - `rustfft` → use `oxifft`.
    - `openblas` (or any BLAS requiring a system/Fortran library) → use
      `oxiblas`.
  - If you're unsure whether a crate you want to add is Pure Rust, check
    `cargo tree -i <crate>` for `-sys` crates or `links = "..."` build
    dependencies, and check the COOLJAPAN ecosystem at
    `https://github.com/cool-japan/` for an existing replacement first.
- **No `unwrap()` / `expect()` in production code.** Use `Result<T, E>` and
  `?`, or an explicit `match`/`if let` with a real fallback. `unwrap`/`expect`
  are acceptable in `#[test]` functions and doctests, where a panic on
  failure is the desired behavior.
- **Zero warnings.** `cargo build` and `cargo clippy` (both run with
  `--all-features`) must be silent. The workspace enables
  `clippy::all` and `clippy::pedantic` at `warn` (see
  `[workspace.lints.clippy]` in the root `Cargo.toml`); individual crates
  may allow specific pedantic lints where they're not useful for that
  crate's domain (documentation lints like `missing_errors_doc`, API-shape
  lints like `must_use_candidate`, etc.) — follow the existing per-crate
  `[lints.clippy]` block conventions rather than adding blanket
  `#[allow]`s in source files.
- **Keep files under ~2000 lines.** If a module is creeping past that,
  split it. The project uses and recommends
  [SplitRS](https://github.com/cool-japan/splitrs) for mechanical,
  behavior-preserving file splits; `splitrs --help` documents the options.
  Prior splits in this repo (e.g. `crates/oximedia-container/src/mp4/writer.rs`,
  `crates/oximedia-caption-gen/src/line_breaking/`) are good references for
  the expected module shape (a `mod.rs`/`lib.rs` re-export hub plus focused
  submodules).
- **Workspace dependency policy.** Shared dependencies are declared once in
  `[workspace.dependencies]` in the root `Cargo.toml` and consumed per crate
  with `some_dep.workspace = true` (or
  `some_dep = { workspace = true, features = [...] }` to add crate-local
  features on top). Likewise, shared package metadata
  (`version`, `edition`, `license`, `rust-version`, `authors`, `homepage`,
  `repository`) is inherited via `field.workspace = true` in each crate's
  `[package]` section — do not pin per-crate versions for dependencies that
  already exist in `[workspace.dependencies]`, and do not hardcode a crate's
  own version/edition/license independently of the workspace. The two fields
  that *are* expected to differ per crate are `description`, `keywords`, and
  `categories`, since each crate serves a different niche on crates.io.
- **Naming.** Follow standard Rust naming conventions: `snake_case` for
  variables, functions, and modules; `UpperCamelCase` for types and traits;
  `SCREAMING_SNAKE_CASE` for constants. `cargo fmt` and `clippy::pedantic`
  will catch most deviations.
- **Tests use `std::env::temp_dir()`**, not a hardcoded path, for any
  scratch file I/O.
- **Honesty in status claims.** If you implement part of a codec/format and
  it isn't end-to-end correct yet (e.g. header parsing works but pixel
  reconstruction is a stub), update `docs/codec_status.md` to reflect the
  real state using the existing taxonomy (*Verified* / *Functional* /
  *Bitstream-parsing* / *Experimental*) rather than leaving stale or
  optimistic claims in the README.

## Branch Policy

- Active development happens on version branches named `0.x.y` (e.g. the
  current development branch). Cargo.toml's workspace version is bumped when
  the branch changes — don't bump the version on every commit.
- `master` (the stable/remote branch) receives **only** release commits, of
  the form `Availability of X.Y.Z`, once a version branch's work is ready to
  ship. Do not open PRs against `master` with feature/fix work; target the
  current version branch instead.
- Please rebase (rather than merge) your branch on top of the target branch
  before opening a PR when practical, to keep history readable — though a
  clean merge commit is acceptable if a rebase would be disruptive.

## Commit and Pull Request Conventions

- Commit messages should be short, imperative, and specific about *what*
  changed (e.g. `Fix AV1 non-square TX block coefficient decoding`,
  `Add SplitRS-based split for mp4/writer.rs`) — see `git log` for the house
  style. A one-line summary is fine for small changes; add a body for
  anything that needs "why" context.
- Every PR should:
  1. Pass `cargo fmt --all --check`, `cargo clippy --workspace --all-features
     --all-targets`, and `cargo nextest run --workspace --all-features`
     locally (there is no CI gate that does this for you yet).
  2. Include or update tests for the behavior you changed.
  3. Fill out `.github/PULL_REQUEST_TEMPLATE.md` (used automatically when
     you open a PR on GitHub) — in particular the Patent/Legal Compliance
     and Dependencies checklists, since patent-freedom and the Pure Rust
     policy are core project invariants, not style preferences.
  4. Update `CHANGELOG.md` under `[Unreleased]` for user-visible changes,
     and `docs/codec_status.md` for codec/container status changes.
  5. Keep the diff focused — prefer several small, reviewable PRs over one
     large one, especially across crate boundaries.
- New dependencies should be justified in the PR description: what it's for,
  why an existing workspace dependency (or the COOLJAPAN ecosystem) doesn't
  already cover it, and confirmation it's Apache-2.0 compatible and,
  ideally, Pure Rust.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md), v2.1.
Please read it before participating.

## License and Contribution Terms

OxiMedia is licensed under the [Apache License, Version 2.0](LICENSE). By
submitting a contribution, you agree that it is licensed under the same
terms (Apache-2.0 §5, "Submission of Contributions"), and that you have the
right to submit it under that license (the PR template's "Patent/Legal
Compliance" checklist asks you to confirm this explicitly, given the
project's patent-freedom goals — please do not port in code from
patent-encumbered or non-Apache-2.0-compatible codebases). Signing off your
commits (`git commit -s`) to record a Developer Certificate of Origin-style
attestation is welcomed, though not currently enforced by tooling.

## Questions

Open a [GitHub Discussion](https://github.com/cool-japan/oximedia/discussions)
or a blank issue. Do **not** use public issues for security reports — see
[SECURITY.md](SECURITY.md) for responsible disclosure instructions.
