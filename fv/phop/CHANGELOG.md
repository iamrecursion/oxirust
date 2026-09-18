# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - Unreleased

### Added

### Changed

### Fixed

## [0.1.0] - 2026-06-25

### Added

**Core engine & discovery**

- Tensorized EML forest with guarded bottom-up forward evaluation over the `eml(x,y) = exp(x) − ln(y)` operator (level-order/heap encoding), numerically matching `oxieml` and staying finite on extreme inputs (exp overflow, ln of non-positive).
- Topology-enumeration discovery pipeline: enumerate → gradient constant fit (Adam) → Levenberg–Marquardt polish → named-constant snapping → Pareto front → code distillation, deterministic for a fixed seed.
- Gumbel-Softmax differentiable leaf-topology search (`discover_gumbel`) with linear/cosine temperature annealing, argmax hardening, and post-hardening constant re-fit.
- Gated depth learning (`discover_gated`): per-node soft source selection with learnable expand/terminate gates so tree depth and shape are learned under a depth curriculum, plus a warm-start variant (`discover_gated_warm`) seeded from the enumerate solution that recovers `exp(exp(x))`.
- Differentiable structural penalties (complexity, sparsity, parsimony) via `lambda_complexity` / `lambda_sparsity` / `lambda_parsimony`.
- Rich-leaf affine engine (`discover_affine` / `discover_affine_pareto`) with `Linear` (`Σaᵢxᵢ+b`) and log-linear `LogLinear` (`Σaᵢ ln xᵢ+b`) leaves recovering product/power/ratio laws, with iterative AI-Feynman-style rational-exponent and named-constant snapping for symbolic recovery.
- Merged meta-search `discover_auto_all` (CLI `--method auto`) spanning the EML-tree searches and the rich-leaf engine on one Pareto front, plus `oxieml::symreg` GA/beam/MCTS candidates re-scored by phop's evaluator.

**GPU backends**

- CUDA backend (`gpu-cuda`, oxicuda 0.3): forward EML PTX kernel, GPU-resident constant fit, and Gumbel-Softmax topology with gradient-checked reverse-mode analytic gradients (~9.4 ms/epoch at 1M rows on an RTX 3060).
- Apple Metal backend (`gpu-metal`, oxicuda-metal 0.3): forward, GPU-resident fit, and Gumbel topology at full CUDA parity.
- Portable WebGPU/Metal/Vulkan/DX12 forward backend (`gpu-wgpu`) compiling the EML tree to a single WGSL compute shader; `gpu_backend()` selects CUDA → wgpu → CPU with a CPU fallback always available.

**Verification**

- Certified roots (`certified_root`, interval Newton/Krawczyk → `RootCertificate`) and sound range enclosures (`certified_range`).
- SMT property proofs (`smt` feature) via OxiZ — lower/upper bounds, sign, no-root, and equivalence — with counterexamples re-verified by exact forward evaluation.
- Lean proof-carrying tier (`lean` feature) via OxiLean: kernel self-check and machine-checked EML rewrite (`∀x, eml x 1 = exp x`) against a postulated EML theory, with negative controls.

**Analysis & utilities**

- Buckingham-π dimensional analysis (`dimension.rs`): exact-rational nullspace π-groups and `DataSet::to_dimensionless` reduction.
- Governing-equation discovery (`ode::discover_ode`) recovering the RHS of an autonomous scalar ODE `dx/dt = f(x)`.
- NSGA-III multi-objective Pareto ranking (`rank_multiobjective`) over accuracy, complexity, interpretability, and elegance.
- e-graph equality-saturation canonicalization (`egraph` feature) via scirs2-symbolic, and a neuro-symbolic TensorLogic bridge (`tensorlogic` feature) mapping a discovered law into oxieml's TensorLogic IR as a weighted rule/equation.
- Robust losses (MSE / Huber / Trimmed) wired into the LM polish via IRLS, with alternative scirs2-optimize LM and L-BFGS constant-fit backends.
- Symbolic distillation to canonical LaTeX, Rust, NumPy, and SymPy; `Solution::compile_rust` deploy path; CAS analysis (derivative / antiderivative / Maclaurin / limit) via oxieml.
- NUMA-parallel topology sweep (`parallel` feature) via scirs2-core, order-preserving so discovery stays deterministic.

**Interfaces**

- `phop` CLI: `discover` (`--method enumerate|gumbel|gated|gated-warm|auto|rich`, `--format table|latex|rust|json`, `--certify`, `--units`, `--gpu`, `--analyze`, `--lambda-*`) and `predict` (reload a law from JSON and apply to new CSV).
- Python bindings (`phop-py`, PyO3) with the GIL released and `.pyi` stubs.
- WebAssembly bindings (`phop-wasm`): `discover_json` / `capabilities` / `set_panic_hook`, plus `discover_and_verify` running the whole discover → CAS analyze → e-graph canonicalize → certified range/root → SMT proof pipeline (OxiZ SMT solver included) client-side in wasm32; npm package `@cooljapan/phop-wasm`.
- Benchmark harness (`phop-bench`, criterion + Feynman-style recovery suite) and worked examples (`phop-examples`: exp growth, Gumbel exp, Kepler, Michaelis–Menten, Planck, Black–Scholes) with bundled CSVs.
- README quick start and an mdBook (`docs/`) spanning theory → implementation → applications.

**Project-wide**

- Pure-Rust by default (no C/C++/Fortran), organized as a six-crate Cargo workspace.

### Fixed

- Resolved the oxieml 0.1.2 EML→LRA SMT-bridge unsoundness (an `eml` node with a `Const` `ln`-operand could return a spurious `Unsat`) upstream in oxieml 0.1.3 (cool-japan/oxieml#1); `eval_interval` now returns indeterminate rather than a conflict, so `Unsat` is reported only for genuinely-infeasible constraints.

[0.1.1]: https://github.com/cool-japan/phop/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cool-japan/phop/releases/tag/v0.1.0
