# Changelog

All notable changes to `tensorlogic-oxicuda-backend` are documented here.

## [Unreleased]

### Added
- FFT sub-feature: `forward_c2c_1d` / `inverse_c2c_1d` via `oxicuda-fft` (enable with `--features fft`).
- `TlAutodiff` implementation: forward topo-walk + backward Wengert tape covering matmul, batched matmul, elementwise, and reduction gradients.
- GPU vs CPU matmul benchmark suite (`cargo bench --features gpu`).

## [0.1.3] - 2026-04-17

### Added
- Native per-axis reduction via `oxicuda_blas::reduction::axis::reduce_axis` — replaces CPU transpose-pack-loop.
- Unary ops: OneMinus, Neg, Abs, Sqrt, Rsqrt, Exp, Log, Ceil, Floor, HardSigmoid, HardSwish, Softplus, LeakyRelu.
- Binary ops: Sub, Div, Pow, Min, Max, Eq, Ne, Lt, Gt, Le, Ge, OrMax, OrProbSum, Nand, Nor, Xor.

### Removed
- CPU transpose-pack-loop fallback in `reduce.rs`.

## [0.1.2] - 2026-04-17 (planned)

### Added
- Einsum dispatch: batched matmul `bij,bjk->bik`; identity / pure-transpose / single-sum-axis specs.
- Elementwise: Relu, Sigmoid, Gelu, Silu, Tanh, HardSigmoid, HardSwish, Softplus, LeakyRelu, Scale (unary); Add, Mul, FusedAddRelu, FusedScaleAdd (binary).

## [0.1.1] - 2026-04-17 (planned)

### Added
- Real SGEMM via `oxicuda_blas::level3::gemm` for `einsum("ij,jk->ik")`.
- Integration test gated behind `TENSORLOGIC_GPU_TESTS=1`.

## [0.1.0] - 2026-04-15

### Added
- Initial crate scaffold registered in workspace.
- `OxiCudaExecutor` / `OxiCudaTensor` / `OxiCudaBackendError` skeleton.
- `TlExecutor` trait implementation; default build is 100% pure Rust (`BackendDisabled` on GPU path).
- `gpu` feature gate enabling `oxicuda-backend` + `oxicuda-blas`.
