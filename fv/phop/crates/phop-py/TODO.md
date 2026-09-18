# phop-py — TODO

> Python bindings (PyO3). Gives the Python/Julia-accustomed SR community day-one access via
> `pip install phop` (Risk S3 mitigation). Wraps `phop-core`; adds no engine logic.

**Crate type:** `cdylib` (PyO3 extension module). **Build:** `maturin` + `pyproject.toml`.
**Deps:** `phop-core` (path), `pyo3` (latest, `extension-module`), `numpy` (rust-numpy),
optional `pyo3-arrow` for zero-copy.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[?]` open decision.

---

## Build/packaging
- [x] `Cargo.toml`: `[lib] crate-type = ["cdylib", "rlib"]`; `[lints] workspace = true`.
- [ ] **Isolate from the default workspace test loop** so a bare `cargo build`/`nextest` of the
      workspace does **not** require a Python interpreter (feature gate or exclude bin tests).
- [x] `pyproject.toml` (maturin backend), classifiers, Apache-2.0 license metadata.
- [ ] Build wheels for linux/mac/windows; `pip install phop` smoke test (→ M8 PyPI publish).

## API surface (mirror design §3.4)
- [?] **Q6 — API philosophy.** Default to **sklearn-style** (`fit`/`predict`); keep names PySR-ish
      where natural for migration. (PyTorch-`Module` style rejected — EML homogeneity makes it
      unnecessary.)
- [x] `Discoverer(population=256, max_depth=3, max_epochs=1000, seed=0, top_k=5,
      lambda_complexity=, lambda_sparsity=, lambda_parsimony=)`.
- [x] `.fit(X, y)` accepting NumPy arrays / pandas columns (zero-copy where possible).
- [ ] `.predict(X)`.
- [x] `result.top_latex(k)` → list[str]; `result.top_sympy(k)` → list[str];
      `result.top(k)` → list of solution objects exposing
      `.latex / .pretty / .rust_code / .numpy_code / .sympy_code / .complexity / .mse`.
- [ ] `to_sympy()` returning a real SymPy expr (optional; via string round-trip).
- [x] Release the GIL around `fit` (long Rust compute) so it plays nice with async/notebooks.

## Ergonomics
- [x] Type stubs (`.pyi`) for editor completion (`phop.pyi` + `py.typed`, shipped via maturin `include`).
- [x] Map `PhopError` → Python exceptions (`ValueError` / custom `PhopError`).
- [ ] Mirror the design's Python snippet:
      ```python
      import phop, pandas as pd
      df = pd.read_csv("kepler.csv")
      disco = phop.Discoverer(population=256, max_depth=10, gpu="cuda")
      result = disco.fit(df[["a"]], df["T"])
      print(result.top_latex(5))
      ```

## Tests/docs
- [ ] `pytest` round-trip on a tiny dataset; CI builds the wheel.
- [ ] Colab / JupyterLab demo notebook (→ docs, M8).

### Definition of done
`pip install`-able wheel; the design's snippet runs and prints LaTeX; GIL released during fit;
no warnings; workspace `cargo nextest` unaffected by Python presence.
