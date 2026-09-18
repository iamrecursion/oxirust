# phop-bench — TODO

> Benchmark harness: the evidence for the paper's central claims — **recovery rate ≥ PySR** at
> **≥ 5× speed** (design §4 M6, Risk T5). Measures `phop-core`; ships no engine logic.

**Crate type:** `lib` + benches (and a small bin for full-suite runs). **Deps:**
`phop-core` (path), `criterion` (wall-clock), `serde`/`csv`, `anyhow`, optional `polars`/`arrow`
for dataset loading.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done.

---

## Datasets
- [ ] **Feynman Symbolic Regression Benchmark** (Udrescu & Tegmark 2020, 100 equations).
      - [ ] fetch + convert to a common internal format (action item, due ~W4).
      - [ ] loader exposing `(name, X, y, ground_truth_expr)` per equation.
- [ ] **PMLB** regression subset (secondary).
- [ ] Synthetic stress sets: noise levels, sample sizes, variable counts.

## Metrics
- [ ] **Recovery**: symbolic equivalence of distilled expr vs ground truth
      (normalize via `scirs2-symbolic`; tolerance on constants). Report Top-1 / Top-3 / Top-50.
- [~] **Accuracy**: R² / MSE on held-out points.
      (MSE + R² computed for the best solution per case via `eval_tree`; held-out split TODO.)
- [ ] **Wall-clock**: time-to-solution; epochs-to-recovery; throughput (points/s).
- [ ] **Complexity**: node count of the recovered expression (Pareto context).
- [ ] **Robustness**: recovery vs noise/sample-size sweeps (supports the T5 "operator-agnostic
      robustness" narrative — phop has no operator set to mis-specify).

## Comparison harness
- [ ] Side-by-side vs **PySR** (run externally; ingest its outputs/timings into the same tables).
- [ ] Reproducible config matrix (population, depth, seed); fixed hardware notes (RTX 4090 for M4/M6).
- [x] Emit machine-readable results (json/csv) **and** LaTeX tables for `phop.tex` §Experiments.
      (`results_to_csv`, `results_to_json`, `results_to_latex_table` in `src/lib.rs`; the
      `recovery` bin prints the LaTeX table after the human-readable one.)

## Criterion micro-benches
- [ ] Forward-eval throughput vs population size and depth.
- [~] CPU vs CUDA forward throughput — `src/bin/gpu_throughput.rs` (feature `gpu-cuda`) times
      GPU `CudaEmlEngine::eval_tree` vs CPU `eval_tree` across batch sizes. On an RTX 3060: ~8×
      at 100k–1M rows; 1M-row forward in ~15 ms end-to-end (< 100 ms). Metal/ROCm + the full
      GPU optimization-loop epoch gate remain.
- [ ] Memory footprint vs `pop × batch × depth` (validates T3 mitigations).

## Reporting
- [ ] One-command full run producing the headline table (recovery × speed vs PySR).
- [ ] Track results across commits to catch regressions.

### Definition of done
Feynman 100-eq run reproducible end-to-end; Top-3 recovery ≥ PySR at ≥ 5× speed demonstrated and
exported as LaTeX for the paper; benches run under `cargo bench`; no warnings.
