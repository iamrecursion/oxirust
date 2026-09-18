# tensorlogic-oxicuda-backend

OxiCUDA GPU execution backend for TensorLogic, built on the COOLJAPAN Pure-Rust CUDA
ecosystem. Implements the `TlExecutor` and `TlAutodiff` traits from `tensorlogic-infer`.

## Feature flags

| Feature  | Default | Effect |
|----------|---------|--------|
| (none)   | yes     | Pure-Rust stub. Compiles everywhere. `OxiCudaExecutor::new()` returns `BackendDisabled`. |
| `gpu`    | no      | Enables `oxicuda-backend`, `oxicuda-blas`, `oxicuda-driver`, `oxicuda-memory`. Requires the NVIDIA driver at runtime. |
| `fft`    | no      | Enables `oxicuda-fft` GPU FFT wrapper (implies `gpu`). Exposes `forward_c2c_1d` / `inverse_c2c_1d`. |

## Quick start

```rust
use tensorlogic_oxicuda_backend::{OxiCudaBackendError, OxiCudaExecutor};

// With default features (no `gpu`), construction returns BackendDisabled.
// With `gpu` on a machine without an NVIDIA GPU, returns OxiCuda runtime error.
match OxiCudaExecutor::new() {
    Ok(_exec) => {
        // gpu feature is enabled and GPU is available.
    }
    Err(OxiCudaBackendError::BackendDisabled) => {
        // pure-Rust default path; expected when `gpu` is off.
    }
    Err(_other) => {
        // GPU feature enabled but no GPU available (e.g. macOS, CI).
    }
}
```

## Supported operations

### Einsum specs

- `ij,jk->ik` — 2-D matrix multiply (SGEMM via `oxicuda-blas`)
- `bij,bjk->bik` — batched 3-D matrix multiply (`gemm_strided_batched`)
- Identity / pure-transpose specs
- Single-sum-axis contraction specs

### Unary ops

Relu, Sigmoid, Gelu, Silu, Tanh, HardSigmoid, HardSwish, Softplus, LeakyRelu,
Neg, Abs, Sqrt, Rsqrt, Exp, Log, Ceil, Floor, Scale, OneMinus.

### Binary ops

Add, Mul, Sub, Div, Pow, Min, Max, Eq (→ 0.0/1.0), Ne, Lt, Gt, Le, Ge,
FusedAddRelu, FusedScaleAdd, OrMax, OrProbSum, Nand, Nor, Xor.

Note on Nor: implemented as `1 − max(a, b)` (conservative approximation;
probabilistic `1 − (a + b − ab)` form is a planned follow-up).

### Reductions

Sum, Max, Min, Product, Mean — all per-axis on N-D tensors via native GPU kernel
(`oxicuda_blas::reduction::axis::reduce_axis`).

### FFT

1-D C2C forward and inverse via `oxicuda-fft`. Requires `--features fft`.

```rust
#[cfg(all(feature = "gpu", feature = "fft"))]
use tensorlogic_oxicuda_backend::{forward_c2c_1d, inverse_c2c_1d};
```

### Autodiff (`TlAutodiff`)

Forward topo-walk + backward Wengert tape.

Gradient coverage: Matmul2D, BatchedMatmul3D, Relu, Sigmoid, OneMinus, Exp, Log,
Sqrt, Neg, Abs, Add, Sub, Mul, Div, ReduceSum, ReduceMax, ReduceMin, ReduceMean.

Unsupported ops return `UnsupportedAutodiffOp`. Host-side broadcast for
ReduceSum/Max gradients and matrix transpose for matmul gradients are documented
perf TODOs; native GPU kernels are a Round 5 follow-up.

## Performance

```bash
cargo bench --features gpu
```

Compares GPU OxiCUDA matmul vs CPU scirs-backend matmul across square shapes
`[64, 256, 1024, 2048]`. GPU group skips gracefully when no NVIDIA driver is present.

## Pure Rust Policy

Default features are 100% pure Rust with no GPU dependency — the crate builds and
tests on any platform. Enabling `--features gpu` only adds Rust dependencies; at
runtime it requires **the NVIDIA driver only** (`libcuda.so` / `nvcuda.dll`).
No CUDA SDK. No `nvcc`. No C/C++ toolchain.

## Status

Research Preview | Version 0.1.x | COOLJAPAN OU

See [CHANGELOG.md](CHANGELOG.md) for release history.
