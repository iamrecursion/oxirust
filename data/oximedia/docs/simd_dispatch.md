# SIMD Dispatch — OxiMedia 0.1.5

`oximedia-simd` runtime-dispatches hot-loop kernels to the best available
SIMD backend per CPU. Paths are relative to
[`crates/oximedia-simd/src/`](../crates/oximedia-simd/src/).

## CPU feature detection

`CpuFeatures` is defined at `src/lib.rs:123`. It carries per-feature flags:

- `avx2`, `avx512f`, `avx512bw`, `avx512vl` — x86-64.
- `sse4_2` — x86-64 fallback tier.
- `neon` — ARM aarch64.

Two entry points:

- `CpuFeatures::detect()` — caches via `OnceLock` on first call; free
  afterwards.
- `detect_cpu_features()` — the same `OnceLock` under the hood.

The cached value is populated once by an `#[cfg(target_arch = "x86_64")]`
branch calling `is_x86_feature_detected!("avx2")` (etc.) or an
`#[cfg(target_arch = "aarch64")]` branch using
`std::arch::is_aarch64_feature_detected!("neon")`. On any other target,
the returned `CpuFeatures` has every flag `false`.

`CpuFeatures::best_simd_width()` returns 512 / 256 / 128 / 64 based on the
widest available register tier.

## Backend tiers

SIMD kernels are organised as one file per architecture or intrinsic
family:

| File | Target | Representative kernels |
|---|---|---|
| `avx512.rs`          | x86-64 AVX-512F/BW/VL/VNNI | `bgra_to_rgba_avx512`, `hsum_f32_avx512`, `scale_i16_avx512`, `sad_8x8_vnni`, `sad_4x4_vnni` |
| `x86.rs`             | x86-64 AVX2 / SSE4.2       | AVX2 bilinear scale, SSE4.2 YUV→RGB BT.601 |
| `neon.rs`            | ARM aarch64 NEON           | `has_neon` probe; NEON kernels |
| `arm.rs`             | ARM scalar / fallback      | Portable ARM wrappers |
| `amx.rs`             | Apple AMX (experimental)   | AMX tile routines |
| `portable.rs`        | `std::simd` portable path  | Cross-platform 128-bit fallbacks |
| `scalar.rs` / `scalar_fallback.rs` | Any target | Reference scalar reductions used for correctness checks |

The `dispatch.rs` module (covered below) picks between these at runtime.

### AVX-512 VNNI tier

`sad_8x8_vnni` (`avx512.rs:1162`) and `sad_4x4_vnni`
(`avx512.rs:1202`) use `_mm512_dpbusd_epi32` to fold sum-of-absolute-
differences into a single dot-product-and-accumulate. Both require
AVX-512F + BW + VNNI; the call sites gate on these three feature flags.

### AVX2 tier (dominant x86 path)

Most x86-64 CPUs support AVX2 but not AVX-512. The AVX2 path in `x86.rs`
handles bilinear scale in 8-pixel batches, YUV→RGB fixed-point in 4-pixel
strides, `_mm256_maddubs_epi16` reductions, and
`_mm_prefetch(_MM_HINT_T0)` cache hints.

### SSE4.2 fallback

Pre-Haswell x86-64 falls to SSE4.2 paths in `x86.rs`. `x86.rs` is gated on
`#[cfg(target_arch = "x86_64")]` (`core::arch::x86_64` intrinsics) and is
never compiled on `wasm32`; see "WASM SIMD128" below for the unrelated,
actual WASM SIMD paths.

### NEON tier

On aarch64 the `neon.rs` module provides NEON implementations.
`neon::has_neon()` (re-exported as `oximedia_simd::has_neon`) returns
`true` on every aarch64 target (NEON is mandatory), `false` elsewhere.

### WASM SIMD128

`oximedia-simd` itself has **no** wasm32-specific code path — there is no
`wasm32` branch anywhere under `crates/oximedia-simd/src/`, and its
`CpuFeatures::detect()` `x86_64`/`aarch64` `cfg` arms simply fall through to
the "every flag `false`" default on `wasm32-unknown-unknown`. `oximedia-simd`
also does not use WASM SIMD128 by way of another crate: `oximedia-codec`,
the workspace's other SIMD-heavy crate, does **not** depend on
`oximedia-simd` at all (`crates/oximedia-codec/Cargo.toml` has no such
dependency). In other words, any WASM SIMD tier this document previously
described as "reusing the SSE4.2 paths, exercised by an `oximedia-codec`
WASM test matrix" does not exist; `oximedia-simd`'s wasm32 builds compile
only the portable scalar fallbacks (`scalar.rs` / `scalar_fallback.rs`),
selected because `detect_cpu_features()` reports no SIMD tier available.

Two real, independent WASM SIMD paths exist elsewhere in the workspace:

- **`oximedia-codec`** has its own hand-written WASM SIMD128 backend at
  [`crates/oximedia-codec/src/simd/wasm.rs`](../crates/oximedia-codec/src/simd/wasm.rs),
  gated on `#[cfg(target_arch = "wasm32")]` and
  `#[target_feature(enable = "simd128")]`, using `core::arch::wasm32`
  intrinsics directly (not anything from `oximedia-simd`). It implements the
  crate's own `SimdOps`/`SimdOpsExt` traits and falls back to
  `ScalarFallback` when `simd128` is unavailable or the target isn't
  `wasm32`.
- **The `web/` workspace** (`oximedia-web-core` and friends, see
  [`web/README.md`](../web/README.md)) achieves WASM SIMD acceleration a
  different way again: no hand-written `core::arch::wasm32` intrinsics at
  all. Kernels are written as ordinary safe Rust over `chunks_exact` /
  `chunks_exact_mut`, compiled with `-C target-feature=+simd128`
  (`web/.cargo/config.toml`) so LLVM autovectorizes the inner loops into
  SIMD128 instructions. This keeps every `web/crates/*` crate at
  `#![forbid(unsafe_code)]` with zero `unsafe` blocks, at the cost of
  depending on the autovectorizer doing a good job rather than hand-tuning
  intrinsics.

## Dispatch module

`src/dispatch.rs` is the public safe façade. Every function there follows
the same three-line pattern:

1. `#[cfg(target_arch = "x86_64")]` block that probes
   `is_x86_feature_detected!` and calls the best `unsafe` intrinsic kernel
   from `avx512.rs` / `x86.rs`.
2. On match, `return` after the unsafe call — no further fallback runs.
3. If no SIMD path matched, fall through to the scalar kernel
   (`avx512::*_scalar` or `scalar::*`).

A minimal example (abridged from `dispatch.rs`):

```rust
pub fn bgra_to_rgba(src: &[u8], dst: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            // SAFETY: avx512f + avx512bw confirmed above.
            unsafe { crate::avx512::bgra_to_rgba_avx512(src, dst) };
            return;
        }
    }
    crate::avx512::bgra_to_rgba_scalar(src, dst);
}
```

Each dispatched function is fully safe in the public API; `unsafe` appears
only inside the `cfg`-gated branch, always guarded by a runtime feature
check.

### Adding a new dispatch entry

1. Implement the scalar reference in `scalar.rs` (or
   `<backend>::*_scalar`).
2. Implement the SIMD-accelerated version in the appropriate backend file
   (`avx512.rs`, `x86.rs`, `neon.rs`, ...). Use `#[target_feature]` on the
   function so the intrinsics compile.
3. Add a safe wrapper in `dispatch.rs` that probes
   `is_x86_feature_detected!` (x86) or
   `std::arch::is_aarch64_feature_detected!` (aarch64) and calls the
   intrinsic under `unsafe { ... }`, then falls back to scalar.
4. Add a property test in `fuzz_targets.rs` or
   `scalar_equivalence.rs` asserting the SIMD result bit-matches the
   scalar reference.

## Benchmarks

Criterion suite at `crates/oximedia-simd/benches/simd_benchmarks.rs`
covers the main dispatched kernels. Workspace-level benches live in
`benches/` (`codec_bench.rs`, `filter_benchmark.rs`,
`quality_metrics.rs`).

## Safety convention

Every `unsafe` intrinsic block in `oximedia-simd`:

1. Lives inside an `#[cfg(target_arch = "...")]` guard matching the
   intrinsic family.
2. Is additionally guarded by a runtime `is_x86_feature_detected!` (or
   `is_aarch64_feature_detected!`) check on the specific feature(s) used.
3. Has a `SAFETY:` comment naming the exact CPU features it relies on.
4. Is the only statement in its `unsafe` block — no unrelated logic.

The `#![deny(unsafe_op_in_unsafe_fn)]` attribute at the top of
`dispatch.rs` enforces the last rule at compile time. A correctness test
in `scalar_equivalence.rs` re-runs every dispatched kernel against its
scalar reference on random inputs, ensuring the fast path never drifts
from the portable path.

## See also

- [`docs/rate_control.md`](rate_control.md) — the encoder hot loops that
  these SIMD kernels back.
- [`docs/codec_status.md`](codec_status.md) — decoder-side status.
- [`docs/wave5_deltas.md`](wave5_deltas.md) — what 0.1.5 added.
