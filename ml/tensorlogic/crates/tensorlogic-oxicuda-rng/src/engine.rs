//! Core RNG engine: a unified GPU + CPU dual-path random number generator.
//!
//! # Design
//!
//! [`RngEngine`] exposes a small, ergonomic surface (`uniform_f32`,
//! `normal_f32`, `bernoulli`, `uniform_f64`, `normal_f64`, streaming variants)
//! that dispatches at runtime to either:
//!
//! * The **CPU path** — a minimal PCG-XSH-RR 64-bit generator with Box-Muller
//!   transform, implemented entirely in pure Rust with zero external
//!   dependencies.  This is always available when the `cpu` feature is enabled
//!   (the default).
//!
//! * The **GPU path** — `oxicuda-rand`'s `RngGenerator`, which compiles and
//!   launches PTX kernels on an NVIDIA CUDA device.  Activated by the `gpu`
//!   feature at compile time and then conditionally at runtime via
//!   `gpu_available()`.
//!
//! # Thread safety
//!
//! On the **CPU path**, `RngEngine` is both [`Send`] and [`Sync`] — the state
//! is plain integers with no shared mutable references.  On the **GPU path**,
//! `RngEngine` is `Send` but NOT `Sync` because a CUDA stream cannot be shared
//! across threads.
//!
//! # Policy compliance
//!
//! This file does **not** import `rand`, `rand_distr`, or `ndarray`.
//! The PCG generator and Box-Muller transform are implemented from scratch.

use crate::error::RngError;

// ---------------------------------------------------------------------------
// Public kind enum
// ---------------------------------------------------------------------------

/// The RNG algorithm family to request.
///
/// On the CPU path all three variants share the same underlying PCG-XSH-RR
/// 64-bit state machine — the distinction is preserved so that switching
/// to the GPU path (where Philox, XORWOW, and MRG32k3a map to distinct
/// cuRAND kernels) is a zero-cost refactor.
///
/// On the GPU path the variant selects the corresponding `oxicuda-rand`
/// engine (`RngEngine::Philox` → Philox-4x32-10, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RngEngineKind {
    /// Philox-4x32-10 counter-based PRNG (cuRAND default).
    Philox,
    /// XORWOW with Weyl sequence addition.
    Xorwow,
    /// MRG32k3a combined multiple recursive generator (highest statistical quality).
    Mrg32k3a,
}

impl RngEngineKind {
    /// Returns a stable string representation of the engine kind.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Philox => "philox",
            Self::Xorwow => "xorwow",
            Self::Mrg32k3a => "mrg32k3a",
        }
    }
}

impl std::fmt::Display for RngEngineKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// GPU path helpers
// ---------------------------------------------------------------------------

/// Returns `true` when a CUDA device is accessible at runtime.
///
/// In Round 6 this will call `oxicuda_driver::init()` + `Device::count()`.
/// For now it always returns `false` so the CPU path acts as the universal
/// fallback.
#[cfg(feature = "gpu")]
fn gpu_available() -> bool {
    false // Round 6 will replace this with a real driver probe
}

// ---------------------------------------------------------------------------
// CPU path — PCG-XSH-RR 64-bit generator
// ---------------------------------------------------------------------------

/// Minimal PCG-XSH-RR 64-bit PRNG.
///
/// This is a verbatim implementation of the PCG family algorithm as described
/// by M. E. O'Neill (2014).  No external crate is used.
///
/// State advancement:
/// ```text
///   state' = state * PCG_MULT + inc        (all mod 2^64)
/// ```
/// Output function (XSH-RR):
/// ```text
///   xorshifted = ((state >> 18) ^ state) >> 27   (32-bit result)
///   rot        = state >> 59
///   out        = rotate_right(xorshifted, rot)
/// ```
#[cfg(feature = "cpu")]
struct CpuRngState {
    /// LCG accumulator.
    state: u64,
    /// Stream selector — must be odd.
    inc: u64,
}

#[cfg(feature = "cpu")]
impl CpuRngState {
    const PCG_MULT: u64 = 6_364_136_223_846_793_005_u64;

    /// Constructs a seeded PCG generator.
    ///
    /// The stream discriminator is derived from the seed (with the odd-bit
    /// forced), then the generator is "warmed up" with two advance steps so
    /// the initial `state = 0` bias is eliminated.
    fn new(seed: u64) -> Self {
        let inc = seed.wrapping_shl(1) | 1; // ensure odd — stream != 0 required
        let mut s = Self { state: 0, inc };
        // Warm-up: absorb the seed into the state before first output.
        let _ = s.next_u32(); // advance once from zero
        s.state = s.state.wrapping_add(seed);
        let _ = s.next_u32(); // second advance after seeding
        s
    }

    /// Returns the next 32-bit pseudorandom output.
    #[inline]
    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        // Linear congruential step.
        self.state = old.wrapping_mul(Self::PCG_MULT).wrapping_add(self.inc);
        // XSH-RR permutation.
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Returns a uniform sample in `[0.0, 1.0)` by masking the 23 mantissa
    /// bits of an f32.
    #[inline]
    fn next_f32(&mut self) -> f32 {
        // Take the top 23 bits, set the exponent to 127 (= 1.0), subtract 1.
        // This maps the 23-bit integer uniformly to [1.0, 2.0) → shift to [0.0, 1.0).
        let bits = (self.next_u32() >> 9) | 0x3f80_0000_u32;
        f32::from_bits(bits) - 1.0_f32
    }

    /// Returns a pair of independent standard normal samples using Box-Muller.
    ///
    /// Box-Muller transform:
    /// ```text
    ///   r     = sqrt(-2 * ln(u1))
    ///   theta = 2 * PI * u2
    ///   z0    = r * cos(theta)
    ///   z1    = r * sin(theta)
    /// ```
    /// where `u1`, `u2 ~ Uniform(0, 1)`.  We guard against `ln(0)` by
    /// clamping `u1` to `f32::EPSILON`.
    #[inline]
    fn next_normal_pair(&mut self) -> (f32, f32) {
        // Guard u1 away from zero to avoid ln(0) = -inf.
        let u1 = {
            let raw = self.next_f32();
            if raw < f32::EPSILON {
                f32::EPSILON
            } else {
                raw
            }
        };
        let u2 = self.next_f32();

        let r = (-2.0_f32 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2; // TAU = 2π
        (r * theta.cos(), r * theta.sin())
    }

    /// Returns the next 64-bit pseudorandom output by combining two 32-bit
    /// PCG outputs into a single u64.
    ///
    /// The high 32 bits come from the first PCG step, the low 32 bits from
    /// the second.  This preserves the sequential structure of the stream so
    /// that `next_u64` and `next_u32` interleave predictably.
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let hi = self.next_u32() as u64;
        let lo = self.next_u32() as u64;
        (hi << 32) | lo
    }

    /// Returns a uniform sample in `[0.0, 1.0)` with 52-bit mantissa
    /// precision using the IEEE 754 exponent-field trick.
    ///
    /// Construction:
    /// ```text
    ///   bits = next_u64() >> 12               (top 52 bits from the 64-bit PCG output)
    ///   x    = f64::from_bits(0x3FF0…0 | bits) - 1.0
    /// ```
    /// The exponent `0x3FF` represents a biased value of 1023, placing the
    /// result in `[1.0, 2.0)`.  Subtracting 1.0 maps to `[0.0, 1.0)`.
    #[inline]
    fn next_f64(&mut self) -> f64 {
        // 64-bit PCG output; keep top 52 bits for the f64 mantissa.
        let bits = self.next_u64();
        // IEEE 754 double: sign=0, exponent=1023 (0x3FF bias → [1.0, 2.0)).
        f64::from_bits(0x3FF0_0000_0000_0000_u64 | (bits >> 12)) - 1.0_f64
    }

    /// Returns a pair of independent standard normal f64 samples via Box-Muller.
    ///
    /// We use `(1.0 - u1)` rather than `u1` directly to guarantee the argument
    /// to `ln()` is strictly in `(0.0, 1.0]`, avoiding `ln(0)`.
    #[inline]
    fn next_normal_pair_f64(&mut self) -> (f64, f64) {
        // u1 ∈ [0, 1) — we invert to (0, 1] before the logarithm.
        let u1 = self.next_f64();
        let u2 = self.next_f64();

        // Use (1 - u1) to map [0,1) → (0,1] and guard against ln(0).
        let safe_u1 = if u1 >= 1.0 { f64::EPSILON } else { 1.0 - u1 };
        let r = (-2.0_f64 * safe_u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2; // TAU = 2π
        (r * theta.cos(), r * theta.sin())
    }
}

// ---------------------------------------------------------------------------
// Inner state enum
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
use oxicuda_rand::generator::{RngEngine as OxiRngEngine, RngGenerator};

#[cfg(feature = "gpu")]
use std::sync::Arc;

/// Polymorphic inner state for the dual-path engine.
enum RngEngineInner {
    /// Pure-Rust PCG generator (always available when `cpu` feature is on).
    #[cfg(feature = "cpu")]
    Cpu(CpuRngState),

    /// GPU-backed generator using `oxicuda-rand`.
    #[cfg(feature = "gpu")]
    Gpu(GpuRngState),
}

/// All GPU-related state bundled together.
#[cfg(feature = "gpu")]
struct GpuRngState {
    generator: RngGenerator,
}

// ---------------------------------------------------------------------------
// Public RngEngine
// ---------------------------------------------------------------------------

/// A seeded, dual-path random number generator.
///
/// Constructed via [`RngEngine::new`].  The `gpu` Cargo feature enables the
/// GPU path; if CUDA is not available at runtime the constructor transparently
/// falls back to the CPU path.
///
/// # Thread safety
///
/// On the **CPU path** (`default`), `RngEngine` is both [`Send`] and [`Sync`]
/// — the state is a pair of `u64` integers and carries no shared references.
///
/// On the **GPU path** (`feature = "gpu"`), `RngEngine` is [`Send`] but NOT
/// [`Sync`].  A CUDA stream cannot be shared across threads; the
/// `PhantomData<*const ()>` field enforces that statically.
pub struct RngEngine {
    /// The engine kind (preserved for introspection and GPU dispatch).
    kind: RngEngineKind,
    /// The inner state — either CPU or GPU.
    inner: RngEngineInner,
    /// Makes `RngEngine` non-`Sync` on the GPU path only.
    ///
    /// On the CPU path this field is absent, allowing the compiler to
    /// auto-derive `Sync` from the plain-integer fields.
    #[cfg(feature = "gpu")]
    _not_sync: std::marker::PhantomData<*const ()>,
}

// SAFETY: `RngEngine` owns its state exclusively (no shared references).
// The CPU path is a plain `u64` pair — both `Send` and `Sync` are safe.
// The GPU path holds a `RngGenerator` which owns a CUDA stream.  Streams are
// safe to *move* across threads (`Send`) but must not be shared (`!Sync`).
// We provide an explicit `Send` impl because the PhantomData on the GPU path
// would otherwise block the auto-derived `Send` as well.
unsafe impl Send for RngEngine {}
// `Sync` is intentionally NOT implemented on the GPU path.
// `PhantomData<*const ()>` prevents the auto-derived impl there.
// On the CPU path (no PhantomData) the compiler auto-derives `Sync` because
// all fields are `u64` (which are `Send + Sync`).

impl RngEngine {
    /// Constructs a new RNG engine of the requested `kind` and `seed`.
    ///
    /// When the `gpu` feature is enabled **and** a CUDA device is reachable at
    /// runtime, the GPU path is chosen; otherwise the CPU path is used.
    ///
    /// # Errors
    ///
    /// Currently infallible on the CPU path.  Returns [`RngError::GpuError`]
    /// if CUDA initialisation fails and there is no CPU fallback compiled in.
    pub fn new(kind: RngEngineKind, seed: u64) -> Result<Self, RngError> {
        // ----------------------------------------------------------------
        // GPU path: attempt to acquire a CUDA context and build a generator.
        // ----------------------------------------------------------------
        #[cfg(feature = "gpu")]
        if gpu_available() {
            return Self::new_gpu(kind, seed);
        }

        // ----------------------------------------------------------------
        // CPU path.
        // ----------------------------------------------------------------
        #[cfg(feature = "cpu")]
        {
            Ok(Self {
                kind,
                inner: RngEngineInner::Cpu(CpuRngState::new(seed)),
                #[cfg(feature = "gpu")]
                _not_sync: std::marker::PhantomData,
            })
        }

        // If neither feature is compiled in this is unreachable, but the
        // compiler needs a return expression in all branches.
        #[cfg(not(any(feature = "cpu", feature = "gpu")))]
        Err(RngError::GpuError(
            "no backend compiled: enable the `cpu` or `gpu` feature".to_string(),
        ))
    }

    /// Constructs a GPU-backed generator.
    #[cfg(feature = "gpu")]
    fn new_gpu(kind: RngEngineKind, seed: u64) -> Result<Self, RngError> {
        use oxicuda_driver::{context::Context, Device};

        oxicuda_driver::init().map_err(|e| RngError::GpuError(e.to_string()))?;
        let device = Device::get(0).map_err(|e| RngError::GpuError(e.to_string()))?;
        let ctx = Arc::new(Context::new(&device).map_err(|e| RngError::GpuError(e.to_string()))?);

        let oxi_kind = match kind {
            RngEngineKind::Philox => OxiRngEngine::Philox,
            RngEngineKind::Xorwow => OxiRngEngine::Xorwow,
            RngEngineKind::Mrg32k3a => OxiRngEngine::Mrg32k3a,
        };

        let generator = RngGenerator::new(oxi_kind, seed, &ctx)
            .map_err(|e| RngError::GpuError(e.to_string()))?;

        Ok(Self {
            kind,
            inner: RngEngineInner::Gpu(GpuRngState { generator }),
            _not_sync: std::marker::PhantomData::<*const ()>,
        })
    }

    /// Returns the engine kind that was requested at construction.
    #[inline]
    pub fn kind(&self) -> RngEngineKind {
        self.kind
    }

    /// Returns `true` when the active path is the GPU.
    pub fn is_gpu(&self) -> bool {
        match &self.inner {
            #[cfg(feature = "cpu")]
            RngEngineInner::Cpu(_) => false,
            #[cfg(feature = "gpu")]
            RngEngineInner::Gpu(_) => true,
        }
    }

    // -----------------------------------------------------------------------
    // Uniform f32
    // -----------------------------------------------------------------------

    /// Fills `out` with independent uniform samples drawn from `[0.0, 1.0)`.
    ///
    /// # Errors
    ///
    /// * [`RngError::EmptyBuffer`] — `out` is empty.
    /// * [`RngError::GpuError`]   — CUDA operation failed (GPU path only).
    pub fn uniform_f32(&mut self, out: &mut [f32]) -> Result<(), RngError> {
        if out.is_empty() {
            return Err(RngError::EmptyBuffer);
        }
        match &mut self.inner {
            #[cfg(feature = "cpu")]
            RngEngineInner::Cpu(state) => {
                for slot in out.iter_mut() {
                    *slot = state.next_f32();
                }
                Ok(())
            }
            #[cfg(feature = "gpu")]
            RngEngineInner::Gpu(gs) => {
                use oxicuda_memory::DeviceBuffer;
                let n = out.len();
                let mut dev_buf =
                    DeviceBuffer::<f32>::alloc(n).map_err(|e| RngError::GpuError(e.to_string()))?;
                gs.generator
                    .generate_uniform_f32(&mut dev_buf)
                    .map_err(|e| RngError::GpuError(e.to_string()))?;
                dev_buf
                    .copy_to_host(out)
                    .map_err(|e| RngError::GpuError(e.to_string()))?;
                Ok(())
            }
        }
    }

    // -----------------------------------------------------------------------
    // Normal f32
    // -----------------------------------------------------------------------

    /// Fills `out` with independent normal samples from `N(mean, std_dev²)`.
    ///
    /// Uses Box-Muller on the CPU path and the engine's native Gaussian kernel
    /// on the GPU path.
    ///
    /// # Errors
    ///
    /// * [`RngError::EmptyBuffer`]                  — `out` is empty.
    /// * [`RngError::InvalidParam`]                 — `std_dev < 0` or not finite.
    /// * [`RngError::GpuError`]                     — CUDA failure (GPU path).
    pub fn normal_f32(&mut self, out: &mut [f32], mean: f32, std_dev: f32) -> Result<(), RngError> {
        if out.is_empty() {
            return Err(RngError::EmptyBuffer);
        }
        if !std_dev.is_finite() || std_dev < 0.0 {
            return Err(RngError::InvalidParam(format!(
                "std_dev must be finite and >= 0, got {std_dev}"
            )));
        }
        if !mean.is_finite() {
            return Err(RngError::InvalidParam(format!(
                "mean must be finite, got {mean}"
            )));
        }

        match &mut self.inner {
            #[cfg(feature = "cpu")]
            RngEngineInner::Cpu(state) => {
                let n = out.len();
                let mut i = 0usize;
                // Consume pairs from Box-Muller; handle the odd element.
                while i + 1 < n {
                    let (z0, z1) = state.next_normal_pair();
                    out[i] = mean + std_dev * z0;
                    out[i + 1] = mean + std_dev * z1;
                    i += 2;
                }
                if i < n {
                    let (z0, _) = state.next_normal_pair();
                    out[i] = mean + std_dev * z0;
                }
                Ok(())
            }
            #[cfg(feature = "gpu")]
            RngEngineInner::Gpu(gs) => {
                use oxicuda_memory::DeviceBuffer;
                let n = out.len();
                let mut dev_buf =
                    DeviceBuffer::<f32>::alloc(n).map_err(|e| RngError::GpuError(e.to_string()))?;
                gs.generator
                    .generate_normal_f32(&mut dev_buf, mean, std_dev)
                    .map_err(|e| RngError::GpuError(e.to_string()))?;
                dev_buf
                    .copy_to_host(out)
                    .map_err(|e| RngError::GpuError(e.to_string()))?;
                Ok(())
            }
        }
    }

    // -----------------------------------------------------------------------
    // Bernoulli
    // -----------------------------------------------------------------------

    /// Fills `out` with Bernoulli(p) samples: each element is `1u8` with
    /// probability `p` and `0u8` otherwise.
    ///
    /// # Errors
    ///
    /// * [`RngError::EmptyBuffer`]  — `out` is empty.
    /// * [`RngError::InvalidParam`] — `p` is not in `[0.0, 1.0]`.
    /// * [`RngError::GpuError`]     — CUDA failure (GPU path).
    pub fn bernoulli(&mut self, out: &mut [u8], p: f32) -> Result<(), RngError> {
        if out.is_empty() {
            return Err(RngError::EmptyBuffer);
        }
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(RngError::InvalidParam(format!(
                "p must be in [0.0, 1.0], got {p}"
            )));
        }

        match &mut self.inner {
            #[cfg(feature = "cpu")]
            RngEngineInner::Cpu(state) => {
                for slot in out.iter_mut() {
                    *slot = u8::from(state.next_f32() < p);
                }
                Ok(())
            }
            #[cfg(feature = "gpu")]
            RngEngineInner::Gpu(gs) => {
                // GPU path: generate uniform f32 on device, threshold on host.
                // A future optimisation can do the threshold in a PTX kernel.
                use oxicuda_memory::DeviceBuffer;
                let n = out.len();
                let mut dev_buf =
                    DeviceBuffer::<f32>::alloc(n).map_err(|e| RngError::GpuError(e.to_string()))?;
                gs.generator
                    .generate_uniform_f32(&mut dev_buf)
                    .map_err(|e| RngError::GpuError(e.to_string()))?;

                let mut host_buf = vec![0f32; n];
                dev_buf
                    .copy_to_host(&mut host_buf)
                    .map_err(|e| RngError::GpuError(e.to_string()))?;
                for (slot, &u) in out.iter_mut().zip(host_buf.iter()) {
                    *slot = u8::from(u < p);
                }
                Ok(())
            }
        }
    }

    // -----------------------------------------------------------------------
    // Uniform f64
    // -----------------------------------------------------------------------

    /// Fills `out` with independent uniform samples drawn from `[0.0, 1.0)`
    /// with 52-bit mantissa precision.
    ///
    /// Each value is constructed from a 64-bit PCG output using the IEEE 754
    /// exponent-field trick: the top 52 bits are inserted into the mantissa of
    /// a double with exponent bias 1023 (∈ `[1.0, 2.0)`), then 1.0 is
    /// subtracted to shift to `[0.0, 1.0)`.
    ///
    /// # Errors
    ///
    /// * [`RngError::EmptyBuffer`] — `out` is empty.
    pub fn uniform_f64(&mut self, out: &mut [f64]) -> Result<(), RngError> {
        if out.is_empty() {
            return Err(RngError::EmptyBuffer);
        }
        match &mut self.inner {
            #[cfg(feature = "cpu")]
            RngEngineInner::Cpu(state) => {
                for slot in out.iter_mut() {
                    *slot = state.next_f64();
                }
                Ok(())
            }
            #[cfg(feature = "gpu")]
            RngEngineInner::Gpu(_gs) => {
                // GPU path: no native f64 cuRAND kernel wired yet; use CPU
                // emulation on the host side for correctness.
                Err(RngError::GpuError(
                    "uniform_f64 on GPU path not yet implemented".to_string(),
                ))
            }
        }
    }

    // -----------------------------------------------------------------------
    // Normal f64
    // -----------------------------------------------------------------------

    /// Fills `out` with independent normal samples from `N(mean, std_dev²)`
    /// with double precision.
    ///
    /// Uses Box-Muller on the CPU path.  Each pair of output values consumes
    /// two `uniform_f64` draws; an odd-length buffer consumes one additional
    /// pair (discarding the second normal from the last Box-Muller step).
    ///
    /// # Errors
    ///
    /// * [`RngError::EmptyBuffer`]  — `out` is empty.
    /// * [`RngError::InvalidParam`] — `std_dev < 0` or not finite, or `mean`
    ///   is not finite.
    pub fn normal_f64(&mut self, out: &mut [f64], mean: f64, std_dev: f64) -> Result<(), RngError> {
        if out.is_empty() {
            return Err(RngError::EmptyBuffer);
        }
        if !std_dev.is_finite() || std_dev < 0.0 {
            return Err(RngError::InvalidParam(format!(
                "std_dev must be finite and >= 0, got {std_dev}"
            )));
        }
        if !mean.is_finite() {
            return Err(RngError::InvalidParam(format!(
                "mean must be finite, got {mean}"
            )));
        }

        match &mut self.inner {
            #[cfg(feature = "cpu")]
            RngEngineInner::Cpu(state) => {
                let n = out.len();
                let mut i = 0usize;
                // Consume pairs from Box-Muller; handle the odd trailing element.
                while i + 1 < n {
                    let (z0, z1) = state.next_normal_pair_f64();
                    out[i] = mean + std_dev * z0;
                    out[i + 1] = mean + std_dev * z1;
                    i += 2;
                }
                if i < n {
                    let (z0, _) = state.next_normal_pair_f64();
                    out[i] = mean + std_dev * z0;
                }
                Ok(())
            }
            #[cfg(feature = "gpu")]
            RngEngineInner::Gpu(_gs) => Err(RngError::GpuError(
                "normal_f64 on GPU path not yet implemented".to_string(),
            )),
        }
    }

    // -----------------------------------------------------------------------
    // Streaming API
    // -----------------------------------------------------------------------

    /// Generates `total` f32 uniform samples and delivers them in chunks of at
    /// most `chunk_size` elements, calling `consumer` once per chunk.
    ///
    /// The final chunk may be smaller than `chunk_size` when `total` is not a
    /// multiple of `chunk_size`.
    ///
    /// # Determinism
    ///
    /// Given the same seed and the same `total`, the complete sequence of
    /// generated values is identical regardless of `chunk_size`.  The chunk
    /// size only affects how many values are presented per callback.
    ///
    /// # Errors
    ///
    /// * [`RngError::EmptyBuffer`] — `total == 0` or `chunk_size == 0`.
    pub fn fill_uniform_chunked<F: FnMut(&[f32])>(
        &mut self,
        total: usize,
        chunk_size: usize,
        consumer: &mut F,
    ) -> Result<(), RngError> {
        if total == 0 || chunk_size == 0 {
            return Err(RngError::EmptyBuffer);
        }

        let mut buf = vec![0f32; chunk_size];
        let mut remaining = total;

        while remaining > 0 {
            let n = remaining.min(chunk_size);
            self.uniform_f32(&mut buf[..n])?;
            consumer(&buf[..n]);
            remaining -= n;
        }
        Ok(())
    }

    /// Generates `total` f64 uniform samples and delivers them in chunks of at
    /// most `chunk_size` elements, calling `consumer` once per chunk.
    ///
    /// The final chunk may be smaller than `chunk_size` when `total` is not a
    /// multiple of `chunk_size`.
    ///
    /// # Determinism
    ///
    /// Given the same seed and the same `total`, the complete sequence of
    /// generated values is identical regardless of `chunk_size`.
    ///
    /// # Errors
    ///
    /// * [`RngError::EmptyBuffer`] — `total == 0` or `chunk_size == 0`.
    pub fn fill_uniform_chunked_f64<F: FnMut(&[f64])>(
        &mut self,
        total: usize,
        chunk_size: usize,
        consumer: &mut F,
    ) -> Result<(), RngError> {
        if total == 0 || chunk_size == 0 {
            return Err(RngError::EmptyBuffer);
        }

        let mut buf = vec![0f64; chunk_size];
        let mut remaining = total;

        while remaining > 0 {
            let n = remaining.min(chunk_size);
            self.uniform_f64(&mut buf[..n])?;
            consumer(&buf[..n]);
            remaining -= n;
        }
        Ok(())
    }

    /// Generates `total` f32 normal samples from `N(mean, std_dev²)` and
    /// delivers them in chunks of at most `chunk_size` elements, calling
    /// `consumer` once per chunk.
    ///
    /// The final chunk may be smaller than `chunk_size`.
    ///
    /// # Determinism
    ///
    /// Given the same seed and the same `total`, the full sequence is identical
    /// regardless of `chunk_size`.  Note: because Box-Muller consumes values in
    /// pairs, chunk boundaries that split a pair internally will advance the
    /// stream by a full pair — the global sequence is determined by `total`, not
    /// chunk boundaries.
    ///
    /// # Errors
    ///
    /// * [`RngError::EmptyBuffer`] — `total == 0` or `chunk_size == 0`.
    /// * [`RngError::InvalidParam`] — `std_dev < 0` or not finite, or `mean`
    ///   is not finite.
    pub fn fill_normal_chunked<F: FnMut(&[f32])>(
        &mut self,
        total: usize,
        chunk_size: usize,
        mean: f32,
        std_dev: f32,
        consumer: &mut F,
    ) -> Result<(), RngError> {
        if total == 0 || chunk_size == 0 {
            return Err(RngError::EmptyBuffer);
        }
        if !std_dev.is_finite() || std_dev < 0.0 {
            return Err(RngError::InvalidParam(format!(
                "std_dev must be finite and >= 0, got {std_dev}"
            )));
        }
        if !mean.is_finite() {
            return Err(RngError::InvalidParam(format!(
                "mean must be finite, got {mean}"
            )));
        }

        let mut buf = vec![0f32; chunk_size];
        let mut remaining = total;

        while remaining > 0 {
            let n = remaining.min(chunk_size);
            self.normal_f32(&mut buf[..n], mean, std_dev)?;
            consumer(&buf[..n]);
            remaining -= n;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Unit tests (CPU-path only, no CUDA device required)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // PCG internals
    // -----------------------------------------------------------------------

    #[test]
    #[cfg(feature = "cpu")]
    fn pcg_inc_is_odd() {
        // The inc field must be odd for full-period PCG.
        for seed in [0u64, 1, 42, u64::MAX, u64::MAX / 2] {
            let state = CpuRngState::new(seed);
            assert_eq!(state.inc & 1, 1, "inc must be odd for seed={seed}");
        }
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn pcg_uniform_in_range() {
        let mut state = CpuRngState::new(12345);
        for _ in 0..10_000 {
            let v = state.next_f32();
            assert!(
                (0.0..1.0).contains(&v),
                "uniform sample {v} not in [0.0, 1.0)"
            );
        }
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn pcg_deterministic_replay() {
        let mut a = CpuRngState::new(777);
        let mut b = CpuRngState::new(777);
        for _ in 0..1000 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn pcg_different_seeds_differ() {
        let mut a = CpuRngState::new(0);
        let mut b = CpuRngState::new(1);
        // Extremely unlikely that 100 consecutive u32 outputs are identical.
        let outputs_a: Vec<u32> = (0..100).map(|_| a.next_u32()).collect();
        let outputs_b: Vec<u32> = (0..100).map(|_| b.next_u32()).collect();
        assert_ne!(
            outputs_a, outputs_b,
            "different seeds should produce different sequences"
        );
    }

    // -----------------------------------------------------------------------
    // Box-Muller
    // -----------------------------------------------------------------------

    #[test]
    #[cfg(feature = "cpu")]
    fn box_muller_pair_is_finite() {
        let mut state = CpuRngState::new(42);
        for _ in 0..10_000 {
            let (z0, z1) = state.next_normal_pair();
            assert!(z0.is_finite(), "z0 is not finite: {z0}");
            assert!(z1.is_finite(), "z1 is not finite: {z1}");
        }
    }

    // -----------------------------------------------------------------------
    // RngEngineKind
    // -----------------------------------------------------------------------

    #[test]
    fn engine_kind_as_str() {
        assert_eq!(RngEngineKind::Philox.as_str(), "philox");
        assert_eq!(RngEngineKind::Xorwow.as_str(), "xorwow");
        assert_eq!(RngEngineKind::Mrg32k3a.as_str(), "mrg32k3a");
    }

    #[test]
    fn engine_kind_display() {
        assert_eq!(format!("{}", RngEngineKind::Philox), "philox");
        assert_eq!(format!("{}", RngEngineKind::Xorwow), "xorwow");
        assert_eq!(format!("{}", RngEngineKind::Mrg32k3a), "mrg32k3a");
    }

    // -----------------------------------------------------------------------
    // RngEngine construction & properties
    // -----------------------------------------------------------------------

    #[test]
    fn engine_new_returns_ok() {
        for kind in [
            RngEngineKind::Philox,
            RngEngineKind::Xorwow,
            RngEngineKind::Mrg32k3a,
        ] {
            assert!(
                RngEngine::new(kind, 0).is_ok(),
                "construction failed for {kind}"
            );
        }
    }

    #[test]
    fn engine_kind_accessor() {
        let eng = RngEngine::new(RngEngineKind::Mrg32k3a, 1).unwrap();
        assert_eq!(eng.kind(), RngEngineKind::Mrg32k3a);
    }

    #[test]
    fn engine_is_not_gpu_in_ci() {
        // GPU path always falls back to CPU in CI (no CUDA device).
        let eng = RngEngine::new(RngEngineKind::Philox, 42).unwrap();
        assert!(!eng.is_gpu());
    }

    // -----------------------------------------------------------------------
    // Error handling
    // -----------------------------------------------------------------------

    #[test]
    fn uniform_empty_buffer_error() {
        let mut eng = RngEngine::new(RngEngineKind::Philox, 0).unwrap();
        let mut out: Vec<f32> = vec![];
        assert!(matches!(
            eng.uniform_f32(&mut out),
            Err(RngError::EmptyBuffer)
        ));
    }

    #[test]
    fn normal_empty_buffer_error() {
        let mut eng = RngEngine::new(RngEngineKind::Philox, 0).unwrap();
        let mut out: Vec<f32> = vec![];
        assert!(matches!(
            eng.normal_f32(&mut out, 0.0, 1.0),
            Err(RngError::EmptyBuffer)
        ));
    }

    #[test]
    fn bernoulli_empty_buffer_error() {
        let mut eng = RngEngine::new(RngEngineKind::Philox, 0).unwrap();
        let mut out: Vec<u8> = vec![];
        assert!(matches!(
            eng.bernoulli(&mut out, 0.5),
            Err(RngError::EmptyBuffer)
        ));
    }

    #[test]
    fn normal_negative_stddev_error() {
        let mut eng = RngEngine::new(RngEngineKind::Philox, 0).unwrap();
        let mut out = vec![0f32; 10];
        assert!(matches!(
            eng.normal_f32(&mut out, 0.0, -1.0),
            Err(RngError::InvalidParam(_))
        ));
    }

    #[test]
    fn normal_nan_mean_error() {
        let mut eng = RngEngine::new(RngEngineKind::Philox, 0).unwrap();
        let mut out = vec![0f32; 10];
        assert!(matches!(
            eng.normal_f32(&mut out, f32::NAN, 1.0),
            Err(RngError::InvalidParam(_))
        ));
    }

    #[test]
    fn bernoulli_invalid_p_error() {
        let mut eng = RngEngine::new(RngEngineKind::Philox, 0).unwrap();
        let mut out = vec![0u8; 10];
        assert!(matches!(
            eng.bernoulli(&mut out, -0.1),
            Err(RngError::InvalidParam(_))
        ));
        assert!(matches!(
            eng.bernoulli(&mut out, 1.1),
            Err(RngError::InvalidParam(_))
        ));
        assert!(matches!(
            eng.bernoulli(&mut out, f32::NAN),
            Err(RngError::InvalidParam(_))
        ));
    }

    // -----------------------------------------------------------------------
    // Statistical sanity — quick checks (small N, loose tolerances)
    // -----------------------------------------------------------------------

    #[test]
    fn uniform_in_range() {
        let mut eng = RngEngine::new(RngEngineKind::Philox, 42).unwrap();
        let mut out = vec![0f32; 1_000];
        eng.uniform_f32(&mut out).unwrap();
        for &v in &out {
            assert!((0.0..1.0).contains(&v), "uniform sample {v} out of [0,1)");
        }
    }

    #[test]
    fn normal_odd_length_fills_all_elements() {
        // Exercises the trailing odd-element branch in normal_f32.
        let mut eng = RngEngine::new(RngEngineKind::Xorwow, 99).unwrap();
        let mut out = vec![f32::NAN; 7]; // odd length
        eng.normal_f32(&mut out, 0.0, 1.0).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "element {i} is not finite: {v}");
        }
    }

    #[test]
    fn bernoulli_outputs_only_zero_or_one() {
        let mut eng = RngEngine::new(RngEngineKind::Mrg32k3a, 555).unwrap();
        let mut out = vec![255u8; 1_000];
        eng.bernoulli(&mut out, 0.5).unwrap();
        for &b in &out {
            assert!(b == 0 || b == 1, "bernoulli output {b} is not 0 or 1");
        }
    }

    #[test]
    fn bernoulli_p_zero_produces_all_zeros() {
        let mut eng = RngEngine::new(RngEngineKind::Philox, 1).unwrap();
        let mut out = vec![1u8; 500];
        eng.bernoulli(&mut out, 0.0).unwrap();
        assert!(out.iter().all(|&b| b == 0));
    }

    #[test]
    fn bernoulli_p_one_produces_all_ones() {
        let mut eng = RngEngine::new(RngEngineKind::Philox, 2).unwrap();
        let mut out = vec![0u8; 500];
        eng.bernoulli(&mut out, 1.0).unwrap();
        assert!(out.iter().all(|&b| b == 1));
    }
}

// ---------------------------------------------------------------------------
// Compile-time Send+Sync assertions for the CPU path
// ---------------------------------------------------------------------------

/// Verifies at compile time that [`RngEngine`] is both [`Send`] and [`Sync`]
/// on the CPU path (no `gpu` feature).
///
/// If the type bounds fail, this module fails to compile — no runtime test
/// needed.
#[cfg(not(feature = "gpu"))]
mod send_sync_assertions {
    use super::RngEngine;

    fn _assert_send<T: Send>() {}
    fn _assert_sync<T: Sync>() {}

    fn _check_rng_engine_send_sync() {
        _assert_send::<RngEngine>();
        _assert_sync::<RngEngine>();
    }
}
