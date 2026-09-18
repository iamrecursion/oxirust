# phop-examples — TODO

> The **killer demos** — the proof that phop *discovers physics*. Each is a small, self-contained
> program over `phop-core` that recovers a known law from data within a time budget (design §4.1).
> These are the M8 launch artifacts and the README's first impression.

**Crate type:** `bin`/`examples`. **Deps:** `phop-core` (path), `csv`/`serde`, `anyhow`.
Each example: a header comment stating the **target law**, **data source**, and **time budget**;
prints the recovered Pareto front + the canonical LaTeX.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done.

---

## Demos (all must land by M6)
- [x] **`kepler.rs`** — Kepler's 3rd law `T² ∝ a³` (recovered as `T = k · a^(3/2)`).
      Data: synthetic `(a, T)` (matches bundled `examples/data/kepler.csv`). Budget: **15 s**. *This is the README headline.*
- [x] **`planck.rs`** — Planck radiation `B(λ,T) = (2hc²/λ⁵) / (exp(hc/λkT) − 1)`.
      Data: synthetic normalized spectral curve `B(u) = u^5/(exp(u)-1)`. Budget: **60 s**. (Approximation expected today.)
- [x] **`michaelis_menten.rs`** — `v = V_max·[S] / (K_m + [S])`. Data: synthetic `[S]` vs `v` (matches bundled CSV). Budget: **10 s**.
- [x] **`black_scholes.rs`** — Black-Scholes core (synthetic 2-feature call-price surface, r=0, K=1). Budget: **90 s**. (Approximation expected.)
- [ ] (batch) **Feynman Top-50** — lives mainly in `phop-bench`; add a thin example entry point.

## Shared scaffolding
- [x] Tiny bundled CSVs (or downloader) for each demo under `examples/data/` (kepler.csv, michaelis_menten.csv, 40 rows each).
- [x] Common helper: load → `Config` → `Discoverer::fit` → print `pareto_top(k)` (LaTeX + MSE + complexity). Shared data-gen fns in `src/lib.rs`.
- [x] `--time-budget` wall-clock assertion so demos double as integration tests of the budgets (`tests/budgets.rs`).

## Quality / docs
- [x] Each example runnable via `cargo run -p phop-examples --example kepler`.
- [x] Recovery assertion: `tests/budgets.rs::exp_growth_recovers_exactly` checks the exact case
      (mse < 1e-6, renders as an exponential). Approximated laws stay budget/non-empty asserted.
- [x] README embeds a 30-second runnable demo (the exp-growth quick start) + CLI usage on the
      bundled `examples/data/*.csv`.

### Definition of done
All four named demos recover their target law within budget on CPU; outputs match the canonical
forms; examples run clean and warning-free; Kepler powers the README.
