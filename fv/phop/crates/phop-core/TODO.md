# phop-core — TODO

> The engine. Everything novel about phop lives here: the **tensorized EML forest** (Layer A),
> **differentiable topology** (Layer B), **multi-objective loss + Pareto** (Layer C), and
> **symbolic distillation** (Layer D), plus the public `Discoverer / Config / DataSet` API.
> Pure Rust, no C/FFI. GPU only through `scirs2-core` backends.

**Crate type:** `lib`. **MSRV:** track stable. **Deps:**
`scirs2-{core,autograd,optimize,symbolic,linalg} = "0.5"`, `oxieml = "0.1"`,
`ndarray = "0.16"`, `rayon = "1.10"`, `thiserror = "2"`, `serde = "1"`.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[?]` open decision.

> Implemented engine features (forest, fit/polish, gumbel, gated, loss, pareto, distill/codegen,
> GPU/Metal backends, the standalone `discover_affine` rich-leaf engine, warm-start,
> post-hardening re-fit) are recorded in `CHANGELOG.md` (0.1.0, 2026-06-25). Only the open
> items below remain.

---

## Open work

### Engine
- [ ] `VectorizedEmlForest` — single dense-tensor *population* forest (whole population held as one
      dense `[population, batch, features]` tensor).
- [ ] Affine variable leaves `a·xᵢ+b` folded **into** the main tensorized-forest / gated autograd
      path (a leaf-parameterization upgrade inside the `EmlTree`/autograd path). Distinct from the
      already-shipped standalone rich-leaf affine engine (`crate::affine` / `discover_affine`); what
      remains is carrying `(a,b)` on the main forest's leaves and rendering the equivalent subtree.
- [ ] Straight-through estimator (discrete forward at inference, soft grads at train).
- [ ] Shape-mixture.
- [ ] `from_parquet` / `from_arrow` data ingress (the "(later)" formats).

### Backends
- [ ] GPU port of `gated`.
- [ ] ROCm native backend (oxicuda exposes a feature gate; needs hardware).

### Testing
- [ ] Advanced property / round-trip tests: forest `encode → decode → encode` round-trip, and
      extreme-input NaN/Inf property tests on loss and grads.

### Known limitations
- [ ] Kepler `a^{3/2}` — a *depth + real-vs-complex evaluation* problem, **not** an EML-expressiveness
      gap. It is representable via `oxieml::Canonical::pow`; levers are deeper search and complex-aware
      eval (phop's `eval_real` guards `ln` to `>0`).
