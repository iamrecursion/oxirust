# Introduction

**phop** (Thai: พบ — *to discover, to encounter*) is a differentiable symbolic-discovery
engine in pure Rust. Given data `(x, y)`, it returns a Pareto front of closed-form expressions
built from a single binary operator, the EML operator `eml(x, y) = exp(x) − ln(y)`.

This book has three parts:

- **Theory** — why a single operator suffices, and how its homogeneity makes expression
  *structure* differentiable.
- **Implementation** — how the engine is built on the SciRS2 autograd ecosystem.
- **Applications** — rediscovering physical laws from data.

For a quick start, see the [README](https://github.com/cool-japan/phop). The fastest demo:

```bash
cargo run -p phop-examples --example exp_growth
```
