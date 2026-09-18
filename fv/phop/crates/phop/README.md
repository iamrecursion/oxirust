# phop

Façade crate for **phop** — a differentiable symbolic-discovery engine on the EML operator
`eml(x, y) = exp(x) − ln(y)`, in pure Rust.

This crate re-exports [`phop-core`](https://crates.io/crates/phop-core) and reserves the `phop`
name for a forthcoming higher-level façade.

- **Library:** `phop` (this crate) or `phop-core`.
- **CLI:** install [`phop-cli`](https://crates.io/crates/phop-cli) — provides the `phop` binary
  (`phop discover data.csv`).

Licensed under Apache-2.0.
