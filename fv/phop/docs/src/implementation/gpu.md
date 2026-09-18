# The CUDA backend

phop has an optional CUDA backend (feature `gpu-cuda`) built on the cool-japan
[`oxicuda`](https://crates.io/crates/oxicuda) driver/memory/launch stack — the same ecosystem as
`scirs2` and `oxieml`. It lives in `gpu.rs` and is entirely opt-in: the default build never pulls
`oxicuda`, and every GPU test skips cleanly when no device is present, so the CPU path is always
the fallback.

## How it works

The EML primitive `eml(a, b) = exp(clip(a)) − ln(max(b, ε))` is written as a small **PTX** kernel
and loaded through the driver JIT — so no CUDA toolkit or NVRTC is needed at run time, only the
driver (`libcuda`). `exp`/`ln` use the hardware `ex2.approx`/`lg2.approx` instructions, which makes
the on-device math **single precision**: the GPU is a fast approximate engine for the inner loop,
while exact `f64` scoring and the Levenberg–Marquardt polish stay on the CPU.

Three things run on-device:

1. **Forward** — `CudaEmlEngine::eval_tree` evaluates a tree by composing the elementwise kernel
   bottom-up; matches the CPU `f64` forward to `< 1e-3`.
2. **Constant fit** — `fit_constants` keeps the data resident and, per step, runs one forward
   (with an atomic sum-of-squared-residuals reduction) and one **reverse-mode analytic backward**
   (`constant_grad`), then a host-side Adam update. The gradient is checked against finite
   differences.
3. **Gumbel topology** — `discover_gumbel_cuda` does the per-leaf softmax on the host (the
   parameters are tiny) and the weighted leaf combination, `eml` tree, backprop, and gradient
   dot-products on-device.

`Config::backend = Backend::Cuda` routes the `Discoverer`'s constant-fitting step through the GPU
(GPU coarse fit → CPU LM polish → `f64`), and the CLI exposes it as `--gpu cuda`.

## Running it

```bash
# build with the feature; the CUDA libs must be on the loader path
LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  cargo test -p phop-core --features gpu-cuda
LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  cargo run -p phop-bench --features gpu-cuda --release --bin gpu_throughput
```

On an RTX 3060 the forward is ~8× the CPU at ≥100k rows and the resident constant-fit loop runs at
~9 ms/epoch over 1M rows.
