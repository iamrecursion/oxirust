# phop-core

The engine behind [phop](https://github.com/cool-japan/phop) — the first
differentiable symbolic-discovery engine: it learns expression *topology* and
*numeric parameters* by gradient descent over a tensorized population of
homogeneous EML trees (`eml(x, y) = exp(x) − ln(y)`, Odrzywołek 2026),
end-to-end in pure Rust.

## Install

```toml
[dependencies]
phop-core = "0.1"
```

## Usage

```rust
use phop_core::{Config, DataSet, Discoverer};

// Build a DataSet from in-memory arrays (or load one from CSV).
let ds = DataSet::from_xy(&[[1.0], [2.0], [3.0]], &[2.0, 4.0, 6.0]);

let front = Discoverer::new(Config::default()).fit(&ds);
if let Some(best) = front.best() {
    println!("{}", best.latex());
}
```

Key entry points: `Discoverer`, `discover_auto` / `discover_auto_all` (the latter
merges the EML-tree searches with the rich-leaf affine engine into one Pareto
front), `discover_affine` / `discover_affine_pareto` (recovers product / power /
ratio laws), and `Solution::predict` / `latex` / `analyze`.

## Features

Pure-Rust by default (no C/C++/Fortran). Opt-in flags: `smt` (OxiZ SMT proofs),
`lean` (OxiLean proof-carrying), `egraph`, `parallel`, `tensorlogic`,
`gpu-cuda`, `gpu-wgpu`.

Part of the [phop](https://github.com/cool-japan/phop) project.

## License

Apache-2.0
