# Architecture

phop is layered:

- **Layer A — Tensorized forest** (`forest.rs`): batched, guarded autograd evaluation of EML
  trees over data fed through placeholders. A CUDA path (`gpu.rs`) mirrors it on the GPU.
- **Layer B — Differentiable topology** (`gumbel.rs`): Gumbel-Softmax leaf selection over a
  complete-tree skeleton, trained with Adam and temperature annealing. `gated.rs` adds
  per-node expand/terminate gates that make the tree's **depth/shape** differentiable.
- **Layer C — Multi-objective Pareto** (`pareto.rs`, `solution.rs`): candidates scored on
  accuracy and complexity; the non-dominated front is returned (`Solution::predict` evaluates a
  recovered law on new data).
- **Layer D — Distillation** (`distill.rs`, `codegen.rs`, `polish.rs` + oxieml): constants are
  refined by Levenberg–Marquardt and *snapped* to named constants (π, e, √2, …) where that does
  not worsen the fit; the tree is then rendered to canonical LaTeX plus Rust, NumPy, and SymPy.

The `Discoverer` (`discoverer.rs`) orchestrates structural enumeration → constant fitting → LM
polish → snapping → Pareto. `discover_gumbel` and `discover_gated` provide differentiable
alternatives. `Config::backend` selects CPU or CUDA for the fitting inner loop. Data ingress
(`dataset.rs`) supports CSV target/feature selection, z-score standardization, and shuffled
minibatching; reproducibility comes from a shared SplitMix64 (`rng.rs`). All return a `ParetoFront`.

Built on `scirs2-autograd` (reverse-mode AD), `scirs2-core` (SIMD/CUDA/Metal backends), the
cool-japan `oxicuda` GPU stack (CUDA backend, optional), and `oxieml` (EML AST + CAS). No C/FFI in
the core; the GPU path loads hand-written PTX through the driver JIT.
