# phop-cli — TODO

> Command-line front end: `phop discover data.csv`. Thin wrapper over `phop-core` —
> parse args, load data, run the `Discoverer`, print the Pareto front. No engine logic here.

**Crate type:** `bin` (binary name **`phop`**). **Deps:** `phop-core` (path),
`clap = "4"` (derive), `anyhow`/`thiserror`, `serde`, optional `indicatif` (progress).

Legend: `[ ]` todo · `[~]` in progress · `[x]` done.

---

## M0 — skeleton
- [x] `Cargo.toml` with `[[bin]] name = "phop"`.
- [x] `main.rs` with `clap` derive; `--help` renders cleanly (workspace verification gate).
- [x] `phop --version` wired to crate version.

## Core command — `phop discover`
- [x] `phop discover <DATA>` where `<DATA>` is CSV (later: parquet/arrow via `phop-core`).
- [x] Flags mirroring `Config`:
  - [x] `--population <N>` (default 256)
  - [x] `--max-depth <D>`
  - [x] `--max-epochs <N>`
  - [x] `--top-k <K>` (default 5)
  - [x] `--target <COL>` (0-based index OR header name; default: last col = target)
  - [x] `--features <COLS>` — comma-separated feature subset (indices or names); via
        `DataSet::from_csv_columns`.
  - [x] `--gpu <cpu|cuda>` — maps to `Config.backend`; `cuda` needs a `--features gpu-cuda` build
        (warns + falls back to CPU otherwise). Metal pending the core backend.
  - [x] `--lambda-complexity / --lambda-sparsity / --lambda-parsimony`
  - [x] `--seed <N>` for reproducibility
  - [x] `--format <table|latex|json|rust>` output selector
- [x] Load `DataSet::from_csv` / `from_csv_with_target`, build `Config`, run `Discoverer::fit`.
- [x] Print Pareto front (`pareto_top(k)`): each row → LaTeX, complexity, MSE
      (and Rust codegen with `--format rust`).
- [x] `--quiet` / `--verbose` (verbose prints resolved config to stderr).

## UX / robustness
- [x] Helpful errors for bad CSV / shape mismatch / unknown columns (map errors → exit codes).
- [x] `--output <FILE>` to write results (json/latex/etc.) instead of stdout.
- [ ] Respect `NO_COLOR`; pretty table via a small formatter (no heavy dep).

## Tests
- [x] `assert_cmd` / `predicates` tests: `--help`, `discover` on a tiny fixture CSV,
      `--format json`, `--target 0` / by name, unknown column, missing CSV, `--quiet`.
- [ ] Snapshot of Kepler demo output once the engine works (M1+).

### Definition of done
`cargo run -p phop-cli -- discover examples/kepler.csv --top-k 5` prints a Pareto front;
`--help` is clean; tests green; no warnings.
