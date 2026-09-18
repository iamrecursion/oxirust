# Rediscovering physical laws

The `phop-examples` crate ships demos that synthesize data for a known law and run discovery:

```bash
cargo run -p phop-examples --example exp_growth        # y = exp(x)            (exact)
cargo run -p phop-examples --example gumbel_exp        # y = exp(x) via Layer B
cargo run -p phop-examples --example kepler            # T = a^(3/2)           (approx.)
cargo run -p phop-examples --example michaelis_menten  # v = Vmax·S/(Km+S)     (approx.)
cargo run -p phop-examples --example planck            # Planck radiation      (approx.)
cargo run -p phop-examples --example black_scholes     # call price            (approx.)
```

Each demo shares pure data-generating helpers in `phop-examples/src/lib.rs`, and
`phop-examples/tests/budgets.rs` runs them under a wall-clock budget so the demos double as
integration tests. Tiny bundled CSVs live under `examples/data/` (`kepler.csv`,
`michaelis_menten.csv`) for use with the CLI.

`exp_growth` recovers `eml(x, 1) = exp(x)` exactly and renders it as `e^{x}`. Deeper laws such as
`T = a^{3/2}` are currently **approximated** — but this is a *phop search/evaluation* limit, **not**
an EML-expressiveness one. By Odrzywołek's completeness theorem the power is a finite `eml` tree,
and `oxieml` constructs it explicitly: `a^{3/2} = Canonical::pow(a, 3/2) = exp(1.5·ln a)`, where
`mul` and `ln` are themselves `eml` compositions. phop falls short for three reasons: that tree is a
**deep composition** beyond the bounded-depth search; phop's leaves are a single variable or
constant (**no affine leaf** `a·xᵢ+b` — a parameterization choice); and the construction is valid on
the **complex/principal branch** (e.g. `ln a < 0` for `a < 1`), while phop's `eval_real` guards `ln`
to positive arguments. So the levers are deeper search, a richer (affine) leaf basis, warm-starting
from the `oxieml` construction, and complex-aware evaluation — tracked in the project `TODO.md`
files. Levenberg–Marquardt polishing and named-constant snapping (`polish.rs`) sharpen the
constants that the bounded search *does* reach.

## From the command line

The same engine is available as a CLI over CSV data:

```bash
cargo run -p phop-cli -- discover examples/data/kepler.csv --top-k 5 --format latex
```

The last column is the target by default (`--target` picks another; `--features` selects a feature
subset). `--format json` emits each ranked solution with its LaTeX, Rust, NumPy, and SymPy forms.

## Predicting with a recovered law

A discovered `Solution` evaluates on new data through the guarded forward:

```rust
let front = phop_core::Discoverer::new(cfg).fit(&ds)?;
let best = front.best().unwrap();
let y_hat = best.predict(&x_new)?;   // Array1<f64>
println!("{}", best.latex());
```
