# phop — Master TODO

> **phop** (Thai: พบ — *to discover, to encounter*) — the first **differentiable
> symbolic-discovery engine**: learns expression *topology* **and** *numeric parameters*
> by gradient descent over a tensorized population of homogeneous EML trees
> (`eml(x, y) = exp(x) − ln(y)`, Odrzywołek 2026), end-to-end in pure Rust.

This is the workspace-level roadmap. Each member crate has its own `crates/phop-*/TODO.md`
with the detailed task breakdown. Engine internals live in `phop-core`; everything else is a
front end, harness, or demo around it.

**Tagline:** *Discover the equation. One operator at a time.*
**License:** Apache-2.0 · **Edition:** 2021 · **Resolver:** 2

---

## Status (2026-06-25)

0.1.0 has been released (see CHANGELOG.md); 0.1.1 is now in development. Everything already implemented (all six crates,
the engine layers, GPU backends, the verified discovery pipeline, the PySR benchmark run, etc.)
is recorded in **CHANGELOG.md**. This TODO lists only what is still **open**.

> Known upstream issue: `scirs2-autograd` 0.5 emits unconditional debug `println!`/`eprintln!`
> from optimizer/tensor-op hot paths; worked around with a scoped stdout silencer (`silence.rs`)
> during the Adam loop; reported upstream as **cool-japan/scirs#128**.

---

## 0. Ground rules (apply to every crate)

- [ ] **No-warnings policy.** `cargo clippy --workspace --all-targets --all-features -- -D warnings`
      must be clean at all times. Enforce centrally via `[workspace.lints]` + `[lints] workspace = true`.
- [ ] **Latest crates always.** Pin to the newest crates.io minor; re-check before each milestone.
- [ ] **`cargo nextest run` continuously.** Keep the suite green; fix as you go.
- [ ] **2000-line ceiling per file.** Split modules before they cross it (e.g. `forest/` submodule tree).
- [ ] **Pure Rust, zero C/FFI** in `phop-core`. GPU only via `scirs2-core` backends.
- [ ] **English everywhere** — code, comments, docs, commit messages.
- [ ] No stray scratch `.md` in the tree; use `/tmp/`. (`TODO.md` / `README.md` / `docs/` excepted.)

---

## 1. Workspace layout (decision: `crates/` directory)

```
phop/                          (git repo / cargo workspace root)
├─ Cargo.toml                  workspace manifest
├─ Cargo.lock
├─ README.md  LICENSE-APACHE  .gitignore  rust-toolchain.toml  deny.toml
├─ TODO.md                     ← this file
├─ crates/
│  ├─ phop-core/               engine — Layers A–D (+ DataSet/Config/Discoverer)
│  ├─ phop-cli/                `phop discover data.csv`
│  ├─ phop-py/                 PyO3 bindings (cdylib)
│  ├─ phop-wasm/               wasm-bindgen, Layer-A subset for in-browser demos
│  ├─ phop-bench/              Feynman / PMLB benchmark + criterion harness
│  └─ phop-examples/           Kepler · Planck · Michaelis–Menten · Black-Scholes
└─ docs/                       mdBook: theory → implementation → applications
```

```toml
[workspace]
members  = ["crates/phop-core", "crates/phop-cli", "crates/phop-py",
            "crates/phop-wasm", "crates/phop-bench", "crates/phop-examples"]
resolver = "2"
```

> Note: the design doc §3.2 drew crates at top level; we nest them under `crates/` per the
> agreed layout. Everything else in §3.2/§3.3 stands.

### Dependency graph
```
        scirs2-{core,autograd,optimize,symbolic,linalg} = "0.5"   oxieml = "0.1"
                                   │
                              ┌────┴─────┐
                              │ phop-core│  (ndarray 0.16, rayon 1.10, thiserror 2, serde 1)
                              └────┬─────┘
        ┌──────────┬──────────┬────┴─────┬───────────┬──────────────┐
   phop-cli    phop-py    phop-wasm   phop-bench  phop-examples   docs/
   (clap)     (pyo3,      (wasm-      (criterion)  (—)            (mdbook)
              maturin)    bindgen)
```

---

## 2. Milestone roadmap (open milestones only)

- [ ] **M7 · W6 · Paper + arXiv.** "Differentiable Symbolic Discovery with a Single Binary Operator."
      *Done when:* arXiv id assigned; `phop.tex` Method/Impl/Experiments sections completed.
- [ ] **M8 · W7+ · Public release.** `phop-py` on PyPI, `phop-wasm` browser demo, `phop` CLI;
      crates.io publish; HN / Reddit launch.

### Known limitations / deferred
- [ ] phop is **not competitive with PySR** on multivariable Feynman laws; its niche is speed +
      exactness on the narrow shallow-EML class (plus the differentiable-structure / EML-IR ecosystem
      story) — not Feynman breadth.
- [ ] **Deeper recovery.** Kepler `a^{3/2}` / Planck remain *approximated* — representable as EML trees
      (`oxieml::Canonical::pow` builds them) but not reached by phop's bounded-depth search (a search/eval
      limit, not an EML-expressiveness limit). Levers: deeper search, affine/richer leaf basis, warm-start
      from the `oxieml` construction, complex-aware/domain-careful eval.

---

## 3. Cross-cutting open questions (resolve before/at the milestone noted)

- [ ] **Q1 [M0] Flat encoding:** level-order/heap (`child(i)=2i+1,2i+2`, trivial vectorized
      bottom-up eval, fixed-shape) **vs** DFS post-order parent-index (compact, irregular).
      *Decision drives the whole forest tensor.* Lean heap for v0.1.
- [ ] **Q2 [M2] Structure relaxation:** Gumbel-Softmax vs Concrete vs straight-through Bernoulli
      vs SparseMAP — empirically compare at M2.
- [ ] **Q3 [M2] Topology init:** random vs warm-start from `oxieml::SymRegEngine` GA vs data-driven (PCA).
- [ ] **Q4 [M3] Pareto maintenance:** NSGA-II archive vs crowding-distance vs ε-dominance.
- [ ] **Q5 [M4] GPU sharding:** data-parallel vs model-parallel (population) vs hybrid.
- [ ] **Q6 [phop-py] Python API philosophy:** sklearn-style `fit/predict` vs PyTorch-style `Module`
      vs PySR-compatible. (Lean sklearn-style; see `crates/phop-py/TODO.md`.)
- [ ] **Q7 [M5] Discovery validation:** how much to auto-check (significance, extrapolation,
      dimensional consistency).

---

## 4. Risk register (design doc §5) — owners track mitigations in crate TODOs

| ID | Risk | Sev | Mitigation home |
|----|------|-----|-----------------|
| T1 | EML numerical instability (`exp` overflow / `ln` of negatives) | High | `phop-core` guarded eval |
| T2 | Gumbel-Softmax non-convergence / local minima | High | `phop-core` annealing + GA warm-start |
| T3 | `pop × batch × depth` memory blow-up | Med | `phop-core` checkpointing / sharding / minibatch |
| T4 | GPU backend divergence (CUDA vs Metal) | Med | `phop-core` via `scirs2-core` abstraction + CPU fallback |
| T5 | Loses to PySR on Feynman | Med | reframe on "operator-agnostic robustness"; `phop-bench` |
| S1 | Competitor fills L3–L5 first (paper is 2026/03) | **Top** | ship M7 ≤ 6 weeks, no exceptions |
| S2 | Direct competition from Odrzywołek / adjacent | High | contact author early; scope phop to tensorized batch learning |
| L2 | "world-first" claim provenance | Med | timestamp first commit; arXiv preprint records L3–L5 |

---

## 5. Outreach

- [ ] Draft contact email to Odrzywołek (S2) *(deferred — outreach)*.

---

## 6. Documentation & community (M8 — see `docs/`)

- [ ] Colab / JupyterLab notebook driving `phop-py` *(deferred — needs a published wheel)*.
- [ ] "Planck discovered in 60 s" timelapse video *(deferred — media production)*.
- [ ] Launch posts: HN / r/rust / r/MachineLearning; Medium "From NAND to EML" *(deferred — M8 launch)*.

---

## 7. Ecosystem synergies — the EML-IR gradient front-end (open follow-ups)

> phop's lasting role is the **gradient-learning front-end to an EML-IR-native ecosystem**: a
> discovered law is a live `oxieml::EmlTree`, so it can be differentiated, integrated, solved,
> dimensionally-checked, SMT-verified, JIT-compiled, and deployed in one Rust type system, no FFI.
> phop's own moat is **differentiable *structure* over a tensorized GPU population**; everything
> downstream is delegated, not reimplemented.

### 7B. Research follow-ups (open)
- [ ] **Governing-equation discovery — multi-variable / weak-PDE forms** via
      `oxieml::symreg::{sindy,pde}` (scalar-ODE `discover_ode` already shipped).
- [ ] **Dimensional-analysis units interop:** `oxieml::Units` unit parsing; target-group (π₀) handling;
      wire into the CLI (Buckingham-π reduction already shipped).
- [ ] **OxiLean tier generalisation:** derive `eq_trans`/`eq_congr` from the builtin `Eq.rec` (drop two
      postulates); generalise the rewrite to arbitrary discovered trees; thread a `LeanProof` onto `Solution`.
- [ ] **SMT `Unsat` as a second `Proven` source** (now-unblocked after the oxieml 0.1.3 relaxation fix —
      it can prove dependency-cancellation interval arithmetic can't).
- [ ] **Fold `scirs2-symbolic`'s beam into `discover_auto`** as an additional contributor / warm-start
      source.

### 7C. Paper narrative (open)
- [ ] Extend the positioning table: phop's contribution = **differentiable structure over a
      tensorized GPU population**; everything else (CAS, certified numerics, alt-search, SINDy,
      JIT) is delegated to the ecosystem — the part competitors can only glue.
- [ ] Method / Implementation / Experiments paper sections remain unwritten (overlaps M7).
