# Contributing to OxiNum

OxiNum (`oxinum-core`, `oxinum-int`, `oxinum-float`, `oxinum-rational`,
`oxinum-complex`, `oxinum`) is part of the **COOLJAPAN ecosystem** — a family
of Pure-Rust replacements for common C/C++/Fortran-backed libraries,
maintained by COOLJAPAN OU (Team Kitasan).

## Building and Testing

```sh
cargo build --workspace
cargo nextest run --workspace           # test runner (preferred over `cargo test`)
cargo clippy --all-targets -- -D warnings
cargo doc --workspace --no-deps
```

All four commands must succeed with zero warnings before a change is
considered ready for review.

## Project Rules

- **Pure Rust by default.** No C/C++/Fortran dependency (or non-default C
  feature) without an explicit, feature-gated opt-in and clear justification.
- **No `unwrap()`/`expect()`/`panic!`/`unreachable!`/`assert!` on data
  derived from untrusted input** in non-test code. Prefer `OxiNumError` or a
  `checked_*`/`try_*` API; a `panic!` is only acceptable for a genuinely
  impossible internal invariant and must carry a rustdoc `# Panics` section.
- **Zero clippy warnings** — `cargo clippy --all-targets -- -D warnings`
  must pass cleanly.
- **Workspace dependency inheritance.** Shared dependencies live in the
  root `[workspace.dependencies]` table via `dep.workspace = true`; do not
  pin ad-hoc versions in a member `Cargo.toml`. Prefer the latest
  crates.io release when adding a dependency.
- **Keep files under 2000 lines** — split oversized modules instead.
- **Match existing style:** `snake_case` naming, the established error
  enum/variants, and surrounding comment density/doc format.
- Use `std::env::temp_dir()` for temporary files in tests — never a
  hardcoded absolute path.

## Submitting Changes

Open a pull request against the crate's current `0.x` development branch
(not `main`, which is reserved for tagged releases). Describe what changed
and why, and make sure the checklist above passes locally first.
