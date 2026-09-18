//! Layer A on the GPU (M4) — CUDA forward evaluation of an EML tree.
//!
//! This is the optional `gpu-cuda` backend, built on the cool-japan **oxicuda** driver/memory/
//! launch stack (the same ecosystem as `scirs2` and `oxieml`). It mirrors the CPU
//! [`crate::forest::eval_tree`]: a tree is evaluated bottom-up, but each `eml` node becomes a
//! single elementwise GPU kernel launch over device buffers of length `batch`.
//!
//! The kernel is the EML primitive itself,
//! `eml(a, b) = exp(clip(a, ±C)) − ln(max(b, ε))`,
//! with the same guards as the CPU path (Risk T1). It is written in PTX and loaded through the
//! driver JIT (no NVRTC / CUDA toolkit needed at runtime — only the driver `libcuda`). `exp` and
//! `ln` use the hardware `ex2.approx` / `lg2.approx` instructions, so this path is **single
//! precision**: it is a fast approximate forward for candidate screening, while exact f64
//! scoring stays on the CPU. All numeric constants are passed as kernel parameters because PTX
//! float immediates must be hex-encoded; passing them keeps the kernel source readable and exact.
//!
//! Build with `--features gpu-cuda`. At runtime the CUDA driver must be on the loader path.
//! [`cuda_available`] reports whether a device can be opened.

use crate::config::Config;
use crate::dataset::DataSet;
use crate::error::{PhopError, Result};
use crate::fit::{collect_consts, substitute_consts};
use crate::pareto::ParetoFront;
use crate::rng::SplitMix64;
use crate::solution::Solution;
use oxieml::{EmlNode, EmlTree};
use scirs2_core::ndarray::{Array1, Array2};
use std::sync::Arc;

use oxicuda::launch;
use oxicuda::prelude::*;

/// Symmetric clamp on the `exp` argument (matches [`crate::forest::EXP_CLAMP`]).
const EXP_CLAMP: f32 = 50.0;
/// Lower clamp on the `ln` argument (matches [`crate::forest::LN_EPS`]).
const LN_EPS: f32 = 1e-12;

/// Two kernels in one module:
/// - `eml_elem`: `out[i] = exp(clip(a[i])) - ln(max(b[i], lnlo))` via approximate base-2
///   transcendental instructions (constants arrive as parameters — PTX float immediates are hex).
/// - `ssr_reduce`: atomically accumulates `sum((pred[i] - y[i])^2)` into a single device scalar,
///   so a forward's error reduces to one float of device→host traffic (the GPU-resident fit loop).
const KERNELS_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry eml_elem(
    .param .u64 p_out, .param .u64 p_a, .param .u64 p_b, .param .u32 p_n,
    .param .f32 p_negclip, .param .f32 p_posclip, .param .f32 p_lnlo,
    .param .f32 p_log2e, .param .f32 p_ln2
)
{
    .reg .pred %p;
    .reg .b32 %r<8>;
    .reg .f32 %f<24>;
    .reg .b64 %rd<12>;

    ld.param.u64 %rd1, [p_out];
    ld.param.u64 %rd2, [p_a];
    ld.param.u64 %rd3, [p_b];
    ld.param.u32 %r1, [p_n];
    ld.param.f32 %f11, [p_negclip];
    ld.param.f32 %f12, [p_posclip];
    ld.param.f32 %f13, [p_lnlo];
    ld.param.f32 %f14, [p_log2e];
    ld.param.f32 %f15, [p_ln2];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p, %r5, %r1;
    @%p bra DONE_EML;

    mul.wide.u32 %rd4, %r5, 4;
    add.s64 %rd5, %rd2, %rd4;
    add.s64 %rd6, %rd3, %rd4;
    add.s64 %rd7, %rd1, %rd4;

    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];

    max.f32 %f3, %f1, %f11;
    min.f32 %f4, %f3, %f12;
    mul.f32 %f5, %f4, %f14;
    ex2.approx.f32 %f6, %f5;

    max.f32 %f7, %f2, %f13;
    lg2.approx.f32 %f8, %f7;
    mul.f32 %f9, %f8, %f15;

    sub.f32 %f10, %f6, %f9;
    st.global.f32 [%rd7], %f10;
DONE_EML:
    ret;
}

.visible .entry ssr_reduce(
    .param .u64 p_acc, .param .u64 p_pred, .param .u64 p_y, .param .u32 p_n
)
{
    .reg .pred %p;
    .reg .b32 %r<6>;
    .reg .f32 %f<6>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [p_acc];
    ld.param.u64 %rd2, [p_pred];
    ld.param.u64 %rd3, [p_y];
    ld.param.u32 %r1, [p_n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p, %r5, %r1;
    @%p bra DONE_SSR;

    mul.wide.u32 %rd4, %r5, 4;
    add.s64 %rd5, %rd2, %rd4;
    add.s64 %rd6, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    sub.f32 %f3, %f1, %f2;
    mul.f32 %f4, %f3, %f3;
    red.global.add.f32 [%rd1], %f4;
DONE_SSR:
    ret;
}

.visible .entry sub_elem(
    .param .u64 p_out, .param .u64 p_a, .param .u64 p_b, .param .u32 p_n
)
{
    .reg .pred %p;
    .reg .b32 %r<6>;
    .reg .f32 %f<4>;
    .reg .b64 %rd<10>;
    ld.param.u64 %rd1, [p_out];
    ld.param.u64 %rd2, [p_a];
    ld.param.u64 %rd3, [p_b];
    ld.param.u32 %r1, [p_n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p, %r5, %r1;
    @%p bra DONE_SUB;
    mul.wide.u32 %rd4, %r5, 4;
    add.s64 %rd5, %rd2, %rd4;
    add.s64 %rd6, %rd3, %rd4;
    add.s64 %rd7, %rd1, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    sub.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;
DONE_SUB:
    ret;
}

.visible .entry sum_reduce(
    .param .u64 p_acc, .param .u64 p_buf, .param .u32 p_n
)
{
    .reg .pred %p;
    .reg .b32 %r<6>;
    .reg .f32 %f<3>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [p_acc];
    ld.param.u64 %rd2, [p_buf];
    ld.param.u32 %r1, [p_n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p, %r5, %r1;
    @%p bra DONE_SUM;
    mul.wide.u32 %rd4, %r5, 4;
    add.s64 %rd5, %rd2, %rd4;
    ld.global.f32 %f1, [%rd5];
    red.global.add.f32 [%rd1], %f1;
DONE_SUM:
    ret;
}

// Reverse-mode local backward of one eml node. Given the upstream gradient g (= dL/dnode) and the
// node's child outputs a, b, write the child gradients:
//   ga = [negclip < a < posclip] * g * exp(clip(a))        (d/da of  exp(clip a))
//   gb = [b > lnlo]              * g * (-1 / max(b, lnlo))  (d/db of -ln(clip b))
.visible .entry eml_back(
    .param .u64 p_ga, .param .u64 p_gb, .param .u64 p_g, .param .u64 p_a, .param .u64 p_b,
    .param .u32 p_n, .param .f32 p_negclip, .param .f32 p_posclip, .param .f32 p_lnlo,
    .param .f32 p_log2e
)
{
    .reg .pred %pq, %pa1, %pa2, %pin, %pb;
    .reg .b32 %r<6>;
    .reg .f32 %f<24>;
    .reg .b64 %rd<14>;

    ld.param.u64 %rd1, [p_ga];
    ld.param.u64 %rd2, [p_gb];
    ld.param.u64 %rd3, [p_g];
    ld.param.u64 %rd4, [p_a];
    ld.param.u64 %rd5, [p_b];
    ld.param.u32 %r1, [p_n];
    ld.param.f32 %f11, [p_negclip];
    ld.param.f32 %f12, [p_posclip];
    ld.param.f32 %f13, [p_lnlo];
    ld.param.f32 %f14, [p_log2e];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %pq, %r5, %r1;
    @%pq bra DONE_BACK;

    mul.wide.u32 %rd6, %r5, 4;
    add.s64 %rd7, %rd3, %rd6;
    add.s64 %rd8, %rd4, %rd6;
    add.s64 %rd9, %rd5, %rd6;
    add.s64 %rd10, %rd1, %rd6;
    add.s64 %rd11, %rd2, %rd6;

    ld.global.f32 %f1, [%rd7];
    ld.global.f32 %f2, [%rd8];
    ld.global.f32 %f3, [%rd9];
    mov.f32 %f4, 0f00000000;
    mov.f32 %f5, 0fBF800000;

    max.f32 %f6, %f2, %f11;
    min.f32 %f7, %f6, %f12;
    mul.f32 %f8, %f7, %f14;
    ex2.approx.f32 %f9, %f8;
    mul.f32 %f10, %f1, %f9;
    setp.gt.f32 %pa1, %f2, %f11;
    setp.lt.f32 %pa2, %f2, %f12;
    and.pred %pin, %pa1, %pa2;
    selp.f32 %f15, %f10, %f4, %pin;
    st.global.f32 [%rd10], %f15;

    max.f32 %f16, %f3, %f13;
    rcp.approx.f32 %f17, %f16;
    mul.f32 %f18, %f17, %f5;
    mul.f32 %f19, %f1, %f18;
    setp.gt.f32 %pb, %f3, %f13;
    selp.f32 %f20, %f19, %f4, %pb;
    st.global.f32 [%rd11], %f20;
DONE_BACK:
    ret;
}

.visible .entry axpy(
    .param .u64 p_out, .param .u64 p_a, .param .f32 p_alpha, .param .u32 p_n
)
{
    .reg .pred %p;
    .reg .b32 %r<6>;
    .reg .f32 %f<5>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [p_out];
    ld.param.u64 %rd2, [p_a];
    ld.param.f32 %f1, [p_alpha];
    ld.param.u32 %r1, [p_n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p, %r5, %r1;
    @%p bra DONE_AXPY;
    mul.wide.u32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    ld.global.f32 %f2, [%rd5];
    ld.global.f32 %f3, [%rd6];
    fma.rn.f32 %f4, %f1, %f3, %f2;
    st.global.f32 [%rd5], %f4;
DONE_AXPY:
    ret;
}

.visible .entry dot_reduce(
    .param .u64 p_acc, .param .u64 p_a, .param .u64 p_b, .param .u32 p_n
)
{
    .reg .pred %p;
    .reg .b32 %r<6>;
    .reg .f32 %f<4>;
    .reg .b64 %rd<10>;
    ld.param.u64 %rd1, [p_acc];
    ld.param.u64 %rd2, [p_a];
    ld.param.u64 %rd3, [p_b];
    ld.param.u32 %r1, [p_n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p, %r5, %r1;
    @%p bra DONE_DOT;
    mul.wide.u32 %rd4, %r5, 4;
    add.s64 %rd5, %rd2, %rd4;
    add.s64 %rd6, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    mul.f32 %f3, %f1, %f2;
    red.global.add.f32 [%rd1], %f3;
DONE_DOT:
    ret;
}
"#;

/// Map an oxicuda error into a [`PhopError::Backend`].
fn be<E: std::fmt::Display>(e: E) -> PhopError {
    PhopError::Backend(e.to_string())
}

/// Whether a CUDA device can be opened on this machine right now.
///
/// Returns `false` (never panics) if the driver is missing or no device is present, so callers
/// can fall back to the CPU path.
#[must_use]
pub fn cuda_available() -> bool {
    oxicuda::init().is_ok() && Device::get(0).is_ok()
}

/// A reusable CUDA evaluator: holds the device context, a stream, and the compiled EML kernel.
///
/// Construct once and evaluate many trees; the PTX is JIT-compiled a single time.
pub struct CudaEmlEngine {
    // Kept alive for the lifetime of the stream/kernels.
    _ctx: Arc<Context>,
    stream: Stream,
    eml: Kernel,
    ssr: Kernel,
    sub: Kernel,
    sum: Kernel,
    back: Kernel,
    axpy: Kernel,
    dot: Kernel,
}

impl CudaEmlEngine {
    /// Initialize the driver, open device 0, and load the EML + reduction + backward kernels.
    ///
    /// # Errors
    /// Returns [`PhopError::Backend`] if the driver/device/module cannot be initialized.
    pub fn new() -> Result<Self> {
        oxicuda::init().map_err(be)?;
        let dev = Device::get(0).map_err(be)?;
        let ctx = Arc::new(Context::new(&dev).map_err(be)?);
        let stream = Stream::new(&ctx).map_err(be)?;
        let module = Arc::new(Module::from_ptx(KERNELS_PTX).map_err(be)?);
        let eml = Kernel::from_module(module.clone(), "eml_elem").map_err(be)?;
        let ssr = Kernel::from_module(module.clone(), "ssr_reduce").map_err(be)?;
        let sub = Kernel::from_module(module.clone(), "sub_elem").map_err(be)?;
        let sum = Kernel::from_module(module.clone(), "sum_reduce").map_err(be)?;
        let back = Kernel::from_module(module.clone(), "eml_back").map_err(be)?;
        let axpy = Kernel::from_module(module.clone(), "axpy").map_err(be)?;
        let dot = Kernel::from_module(module, "dot_reduce").map_err(be)?;
        Ok(Self {
            _ctx: ctx,
            stream,
            eml,
            ssr,
            sub,
            sum,
            back,
            axpy,
            dot,
        })
    }

    /// Evaluate `tree` over `data` (`[batch, n_vars]`) on the GPU, returning `[batch]`
    /// predictions in `f64` (computed in single precision on-device).
    ///
    /// # Errors
    /// Returns [`PhopError::Backend`] on any CUDA failure, or
    /// [`PhopError::NumericalInstability`] if the result contains non-finite values.
    pub fn eval_tree(&self, tree: &EmlTree, data: &Array2<f64>) -> Result<Array1<f64>> {
        let n = data.nrows();
        let n_vars = data.ncols();
        // Host-side f32 feature columns; uploaded on demand at each variable leaf.
        let host_cols: Vec<Vec<f32>> = (0..n_vars)
            .map(|j| data.column(j).iter().map(|&v| v as f32).collect())
            .collect();

        let root = self.eval_node(&tree.root, &host_cols, n)?;
        let mut out = vec![0f32; n];
        root.copy_to_host(&mut out).map_err(be)?;

        let values: Vec<f64> = out.iter().map(|&v| f64::from(v)).collect();
        if values.iter().any(|v| !v.is_finite()) {
            return Err(PhopError::NumericalInstability(
                "GPU forward produced non-finite values".to_string(),
            ));
        }
        Ok(Array1::from(values))
    }

    /// Evaluate `tree` on the GPU and return its mean-squared error against `y`.
    ///
    /// # Errors
    /// Returns [`PhopError`] on any CUDA failure or non-finite output.
    pub fn eval_mse(&self, tree: &EmlTree, data: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let pred = self.eval_tree(tree, data)?;
        Ok(crate::fit::mse(&pred, y))
    }

    /// Evaluate many trees over the same data on the GPU, reusing this engine (and its
    /// JIT-compiled kernel) across all of them. Returns one prediction vector per tree.
    ///
    /// # Errors
    /// Returns the first [`PhopError`] encountered.
    pub fn eval_trees(&self, trees: &[&EmlTree], data: &Array2<f64>) -> Result<Vec<Array1<f64>>> {
        trees.iter().map(|t| self.eval_tree(t, data)).collect()
    }

    /// Recursively evaluate a node into a device buffer of length `n`.
    fn eval_node(
        &self,
        node: &EmlNode,
        host_cols: &[Vec<f32>],
        n: usize,
    ) -> Result<DeviceBuffer<f32>> {
        match node {
            EmlNode::One => self.const_buffer(1.0, n),
            EmlNode::Const(c) => self.const_buffer(*c as f32, n),
            EmlNode::Var(i) => DeviceBuffer::<f32>::from_host(&host_cols[*i]).map_err(be),
            EmlNode::Eml { left, right } => {
                let a = self.eval_node(left, host_cols, n)?;
                let b = self.eval_node(right, host_cols, n)?;
                let out = DeviceBuffer::<f32>::alloc(n).map_err(be)?;
                self.launch_eml(&out, &a, &b, n)?;
                Ok(out)
            }
        }
    }

    /// Allocate a device buffer filled with the constant `v`.
    fn const_buffer(&self, v: f32, n: usize) -> Result<DeviceBuffer<f32>> {
        DeviceBuffer::<f32>::from_host(&vec![v; n]).map_err(be)
    }

    /// Launch the elementwise EML kernel: `out = eml(a, b)`.
    fn launch_eml(
        &self,
        out: &DeviceBuffer<f32>,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        n: usize,
    ) -> Result<()> {
        let block = 256u32;
        let grid = grid_size_for(n as u32, block);
        let args = (
            out.as_device_ptr(),
            a.as_device_ptr(),
            b.as_device_ptr(),
            n as u32,
            -EXP_CLAMP,
            EXP_CLAMP,
            LN_EPS,
            std::f32::consts::LOG2_E,
            std::f32::consts::LN_2,
        );
        launch!(self.eml, grid(grid), block(block), &self.stream, &args).map_err(be)?;
        self.stream.synchronize().map_err(be)?;
        Ok(())
    }

    // ----- GPU-resident constant fitting (M4) -------------------------------------------------

    /// Fit the free `Const` leaves of a fixed-topology `template` to `ds` entirely on the GPU.
    ///
    /// The feature data and targets are uploaded **once** and stay resident; each optimization
    /// step runs one forward (sum-of-squared-residuals reduced on-device) and one reverse pass
    /// that yields the **analytic** gradient of all constants at once (see [`Self::constant_grad`]).
    /// Host-side Adam then updates them. This is the GPU analogue of [`crate::fit::fit_constants`];
    /// it is single precision.
    ///
    /// Returns the fitted tree and its (GPU-measured) MSE. A constant-free template is just
    /// evaluated. `learning_rate` and `max_epochs` mirror [`crate::Config`].
    ///
    /// # Errors
    /// Returns [`PhopError`] on any CUDA failure.
    pub fn fit_constants(
        &self,
        template: &EmlTree,
        ds: &DataSet,
        learning_rate: f64,
        max_epochs: usize,
    ) -> Result<(EmlTree, f64)> {
        let n = ds.len();
        let n_vars = ds.n_vars();
        // Upload feature columns and targets once; they stay resident across all steps.
        let dev_cols: Vec<DeviceBuffer<f32>> = (0..n_vars)
            .map(|j| {
                let col: Vec<f32> = ds.x.column(j).iter().map(|&v| v as f32).collect();
                DeviceBuffer::<f32>::from_host(&col).map_err(be)
            })
            .collect::<Result<_>>()?;
        let y_host: Vec<f32> = ds.y.iter().map(|&v| v as f32).collect();
        let dev_y = DeviceBuffer::<f32>::from_host(&y_host).map_err(be)?;

        let mut theta = Vec::new();
        collect_consts(&template.root, &mut theta);
        let p = theta.len();
        if p == 0 {
            let m = self.forward_mse_resident(&template.root, &dev_cols, &dev_y, n)?;
            return Ok((template.clone(), m));
        }

        // Flat node list (built once); each step rebuilds only constant buffers from `theta`.
        let flat = build_flat(&template.root);

        // Host-side Adam state for the (few) constants.
        let (b1, b2, eps) = (0.9_f64, 0.999_f64, 1e-8_f64);
        let mut m = vec![0.0_f64; p];
        let mut v = vec![0.0_f64; p];

        for t in 1..=max_epochs {
            // One forward + one reverse pass yields the exact (analytic) gradient of all
            // constants at once — no per-constant finite-difference forwards.
            let (_mse, grad) = self.forward_grad(&flat, &theta, &dev_cols, &dev_y, n)?;
            let bc1 = 1.0 - b1.powi(t as i32);
            let bc2 = 1.0 - b2.powi(t as i32);
            for j in 0..p {
                m[j] = b1 * m[j] + (1.0 - b1) * grad[j];
                v[j] = b2 * v[j] + (1.0 - b2) * grad[j] * grad[j];
                let mhat = m[j] / bc1;
                let vhat = v[j] / bc2;
                theta[j] -= learning_rate * mhat / (vhat.sqrt() + eps);
            }
        }

        let fitted = tree_with_consts(template, &theta);
        let m = self.forward_mse_resident(&fitted.root, &dev_cols, &dev_y, n)?;
        Ok((fitted, m))
    }

    /// Return the MSE and the **analytic** (reverse-mode) gradient of every constant leaf of
    /// `template` against `ds`, at the template's current constant values.
    ///
    /// Useful on its own (e.g. for a gradient check) and the engine of [`Self::fit_constants`].
    ///
    /// # Errors
    /// Returns [`PhopError`] on any CUDA failure.
    pub fn constant_grad(&self, template: &EmlTree, ds: &DataSet) -> Result<(f64, Vec<f64>)> {
        let n = ds.len();
        let n_vars = ds.n_vars();
        let dev_cols: Vec<DeviceBuffer<f32>> = (0..n_vars)
            .map(|j| {
                let col: Vec<f32> = ds.x.column(j).iter().map(|&v| v as f32).collect();
                DeviceBuffer::<f32>::from_host(&col).map_err(be)
            })
            .collect::<Result<_>>()?;
        let y_host: Vec<f32> = ds.y.iter().map(|&v| v as f32).collect();
        let dev_y = DeviceBuffer::<f32>::from_host(&y_host).map_err(be)?;
        let mut theta = Vec::new();
        collect_consts(&template.root, &mut theta);
        let flat = build_flat(&template.root);
        self.forward_grad(&flat, &theta, &dev_cols, &dev_y, n)
    }

    /// Forward (storing every node's output) + reverse pass over the resident buffers.
    /// Returns `(mse, grad)` where `grad[j] = dMSE/dθ_j`.
    fn forward_grad(
        &self,
        flat: &[FlatNode],
        theta: &[f64],
        dev_cols: &[DeviceBuffer<f32>],
        dev_y: &DeviceBuffer<f32>,
        n: usize,
    ) -> Result<(f64, Vec<f64>)> {
        let p = theta.len();
        let root = flat.len() - 1;

        // Forward: postorder, so children precede parents.
        let mut vals: Vec<DeviceBuffer<f32>> = Vec::with_capacity(flat.len());
        for node in flat {
            let buf = match node {
                FlatNode::One => self.const_buffer(1.0, n)?,
                FlatNode::Const(j) => self.const_buffer(theta[*j] as f32, n)?,
                FlatNode::Var(i) => {
                    let mut b = DeviceBuffer::<f32>::alloc(n).map_err(be)?;
                    b.copy_from_device(&dev_cols[*i]).map_err(be)?;
                    b
                }
                FlatNode::Eml { left, right } => {
                    let out = DeviceBuffer::<f32>::alloc(n).map_err(be)?;
                    self.launch_eml(&out, &vals[*left], &vals[*right], n)?;
                    out
                }
            };
            vals.push(buf);
        }

        let mse = self.reduce_ssr(&vals[root], dev_y, n)? / n.max(1) as f64;

        // Backward: seed grad[root] = pred - y, then reverse postorder (parents before children).
        let mut grads: Vec<Option<DeviceBuffer<f32>>> = (0..flat.len()).map(|_| None).collect();
        let g_root = DeviceBuffer::<f32>::alloc(n).map_err(be)?;
        self.launch_sub(&g_root, &vals[root], dev_y, n)?;
        grads[root] = Some(g_root);
        for i in (0..flat.len()).rev() {
            if let FlatNode::Eml { left, right } = &flat[i] {
                let g = grads[i].take().expect("parent gradient computed first");
                let ga = DeviceBuffer::<f32>::alloc(n).map_err(be)?;
                let gb = DeviceBuffer::<f32>::alloc(n).map_err(be)?;
                self.launch_back(&ga, &gb, &g, &vals[*left], &vals[*right], n)?;
                grads[*left] = Some(ga);
                grads[*right] = Some(gb);
            }
        }

        // dMSE/dθ_j = (2/n) * Σ_rows grad_at_leaf_j.
        let mut grad = vec![0.0_f64; p];
        for (i, node) in flat.iter().enumerate() {
            if let FlatNode::Const(j) = node {
                if let Some(gbuf) = &grads[i] {
                    grad[*j] = 2.0 * self.reduce_sum(gbuf, n)? / n.max(1) as f64;
                }
            }
        }
        Ok((mse, grad))
    }

    /// Forward-evaluate `node` against resident columns and reduce the MSE on-device.
    fn forward_mse_resident(
        &self,
        node: &EmlNode,
        dev_cols: &[DeviceBuffer<f32>],
        dev_y: &DeviceBuffer<f32>,
        n: usize,
    ) -> Result<f64> {
        let pred = self.eval_node_resident(node, dev_cols, n)?;
        Ok(self.reduce_ssr(&pred, dev_y, n)? / n.max(1) as f64)
    }

    /// Sum of squared residuals `Σ (pred - y)^2`, reduced on-device.
    fn reduce_ssr(
        &self,
        pred: &DeviceBuffer<f32>,
        dev_y: &DeviceBuffer<f32>,
        n: usize,
    ) -> Result<f64> {
        let acc = DeviceBuffer::<f32>::from_host(&[0.0f32]).map_err(be)?;
        let block = 256u32;
        let grid = grid_size_for(n as u32, block);
        let args = (
            acc.as_device_ptr(),
            pred.as_device_ptr(),
            dev_y.as_device_ptr(),
            n as u32,
        );
        launch!(self.ssr, grid(grid), block(block), &self.stream, &args).map_err(be)?;
        self.stream.synchronize().map_err(be)?;
        let mut out = [0f32; 1];
        acc.copy_to_host(&mut out).map_err(be)?;
        Ok(f64::from(out[0]))
    }

    /// Sum of a device buffer `Σ buf`, reduced on-device.
    fn reduce_sum(&self, buf: &DeviceBuffer<f32>, n: usize) -> Result<f64> {
        let acc = DeviceBuffer::<f32>::from_host(&[0.0f32]).map_err(be)?;
        let block = 256u32;
        let grid = grid_size_for(n as u32, block);
        let args = (acc.as_device_ptr(), buf.as_device_ptr(), n as u32);
        launch!(self.sum, grid(grid), block(block), &self.stream, &args).map_err(be)?;
        self.stream.synchronize().map_err(be)?;
        let mut out = [0f32; 1];
        acc.copy_to_host(&mut out).map_err(be)?;
        Ok(f64::from(out[0]))
    }

    /// Launch the elementwise subtraction `out = a - b`.
    fn launch_sub(
        &self,
        out: &DeviceBuffer<f32>,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        n: usize,
    ) -> Result<()> {
        let block = 256u32;
        let grid = grid_size_for(n as u32, block);
        let args = (
            out.as_device_ptr(),
            a.as_device_ptr(),
            b.as_device_ptr(),
            n as u32,
        );
        launch!(self.sub, grid(grid), block(block), &self.stream, &args).map_err(be)?;
        self.stream.synchronize().map_err(be)?;
        Ok(())
    }

    /// Launch the eml backward kernel, writing child gradients `ga`, `gb`.
    fn launch_back(
        &self,
        ga: &DeviceBuffer<f32>,
        gb: &DeviceBuffer<f32>,
        g: &DeviceBuffer<f32>,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        n: usize,
    ) -> Result<()> {
        let block = 256u32;
        let grid = grid_size_for(n as u32, block);
        let args = (
            ga.as_device_ptr(),
            gb.as_device_ptr(),
            g.as_device_ptr(),
            a.as_device_ptr(),
            b.as_device_ptr(),
            n as u32,
            -EXP_CLAMP,
            EXP_CLAMP,
            LN_EPS,
            std::f32::consts::LOG2_E,
        );
        launch!(self.back, grid(grid), block(block), &self.stream, &args).map_err(be)?;
        self.stream.synchronize().map_err(be)?;
        Ok(())
    }

    /// Like [`Self::eval_node`] but variable leaves are device-to-device copies of the resident
    /// feature columns (no host round-trip per step).
    fn eval_node_resident(
        &self,
        node: &EmlNode,
        dev_cols: &[DeviceBuffer<f32>],
        n: usize,
    ) -> Result<DeviceBuffer<f32>> {
        match node {
            EmlNode::One => self.const_buffer(1.0, n),
            EmlNode::Const(c) => self.const_buffer(*c as f32, n),
            EmlNode::Var(i) => {
                let mut out = DeviceBuffer::<f32>::alloc(n).map_err(be)?;
                out.copy_from_device(&dev_cols[*i]).map_err(be)?;
                Ok(out)
            }
            EmlNode::Eml { left, right } => {
                let a = self.eval_node_resident(left, dev_cols, n)?;
                let b = self.eval_node_resident(right, dev_cols, n)?;
                let out = DeviceBuffer::<f32>::alloc(n).map_err(be)?;
                self.launch_eml(&out, &a, &b, n)?;
                Ok(out)
            }
        }
    }

    /// Launch `out += alpha * a` (single precision).
    fn launch_axpy(
        &self,
        out: &DeviceBuffer<f32>,
        alpha: f32,
        a: &DeviceBuffer<f32>,
        n: usize,
    ) -> Result<()> {
        let block = 256u32;
        let grid = grid_size_for(n as u32, block);
        let args = (out.as_device_ptr(), a.as_device_ptr(), alpha, n as u32);
        launch!(self.axpy, grid(grid), block(block), &self.stream, &args).map_err(be)?;
        self.stream.synchronize().map_err(be)?;
        Ok(())
    }

    /// Reduce the dot product `Σ a[i] * b[i]` on-device.
    fn reduce_dot(&self, a: &DeviceBuffer<f32>, b: &DeviceBuffer<f32>, n: usize) -> Result<f64> {
        let acc = DeviceBuffer::<f32>::from_host(&[0.0f32]).map_err(be)?;
        let block = 256u32;
        let grid = grid_size_for(n as u32, block);
        let args = (
            acc.as_device_ptr(),
            a.as_device_ptr(),
            b.as_device_ptr(),
            n as u32,
        );
        launch!(self.dot, grid(grid), block(block), &self.stream, &args).map_err(be)?;
        self.stream.synchronize().map_err(be)?;
        let mut out = [0f32; 1];
        acc.copy_to_host(&mut out).map_err(be)?;
        Ok(f64::from(out[0]))
    }

    // ----- GPU-resident Gumbel-Softmax topology search (Layer B / M4) -------------------------

    /// One forward + reverse pass of the Gumbel-Softmax forest at the given logits/constants and
    /// sampled noise, returning `(mse, dz, dc)` — the gradients w.r.t. the leaf selection logits
    /// `z` (`n_leaves*k`) and leaf constants `c` (`n_leaves`).
    ///
    /// The per-leaf soft selection (a `softmax` over `k = n_vars + 1` sources) is computed on the
    /// host (the parameters are tiny); the per-row heavy work — the weighted source combination,
    /// the `eml` tree, the SSR and the gradient dot-products — runs on the GPU. The structural
    /// penalty (`struct_lambda * Σ variable-weight mass`) is folded into `dz` via `dL/dw`.
    #[allow(clippy::too_many_arguments)]
    fn gumbel_grad(
        &self,
        z: &[f64],
        c: &[f64],
        gumbel: &[f64],
        tau: f64,
        dev_cols: &[DeviceBuffer<f32>],
        dev_y: &DeviceBuffer<f32>,
        n: usize,
        n_vars: usize,
        depth: usize,
        struct_lambda: f64,
    ) -> Result<(f64, Vec<f64>, Vec<f64>)> {
        let k = n_vars + 1;
        let n_leaves = 1usize << depth;
        let internal_count = (1usize << depth) - 1;
        let total = (1usize << (depth + 1)) - 1;
        let inv_tau = 1.0 / tau;

        // Host softmax of the perturbed logits, per leaf.
        let mut w = vec![0.0_f64; n_leaves * k];
        for l in 0..n_leaves {
            let base = l * k;
            let mut mx = f64::NEG_INFINITY;
            for i in 0..k {
                let v = (z[base + i] + gumbel[base + i]) * inv_tau;
                if v > mx {
                    mx = v;
                }
            }
            let mut sum = 0.0;
            for i in 0..k {
                let e = ((z[base + i] + gumbel[base + i]) * inv_tau - mx).exp();
                w[base + i] = e;
                sum += e;
            }
            for i in 0..k {
                w[base + i] /= sum;
            }
        }

        // Forward.
        let mut vals: Vec<Option<DeviceBuffer<f32>>> = (0..total).map(|_| None).collect();
        for l in 0..n_leaves {
            let base = l * k;
            let leaf = self.const_buffer((w[base + (k - 1)] * c[l]) as f32, n)?;
            for i in 0..n_vars {
                self.launch_axpy(&leaf, w[base + i] as f32, &dev_cols[i], n)?;
            }
            vals[internal_count + l] = Some(leaf);
        }
        for i in (0..internal_count).rev() {
            let out = DeviceBuffer::<f32>::alloc(n).map_err(be)?;
            let a = vals[2 * i + 1].as_ref().expect("child");
            let b = vals[2 * i + 2].as_ref().expect("child");
            self.launch_eml(&out, a, b, n)?;
            vals[i] = Some(out);
        }
        let pred = vals[0].as_ref().expect("root");
        let mse = self.reduce_ssr(pred, dev_y, n)? / n.max(1) as f64;

        // Backward through the tree.
        let mut grads: Vec<Option<DeviceBuffer<f32>>> = (0..total).map(|_| None).collect();
        let g0 = DeviceBuffer::<f32>::alloc(n).map_err(be)?;
        self.launch_sub(&g0, vals[0].as_ref().expect("root"), dev_y, n)?;
        grads[0] = Some(g0);
        for i in 0..internal_count {
            let ga = DeviceBuffer::<f32>::alloc(n).map_err(be)?;
            let gb = DeviceBuffer::<f32>::alloc(n).map_err(be)?;
            let g = grads[i].as_ref().expect("parent grad");
            let a = vals[2 * i + 1].as_ref().expect("child");
            let b = vals[2 * i + 2].as_ref().expect("child");
            self.launch_back(&ga, &gb, g, a, b, n)?;
            grads[2 * i + 1] = Some(ga);
            grads[2 * i + 2] = Some(gb);
        }

        // Per-leaf source weight + constant gradients (scaled by 2/n; penalty added to var weights).
        let scale = 2.0 / n.max(1) as f64;
        let mut dw = vec![0.0_f64; n_leaves * k];
        let mut dc = vec![0.0_f64; n_leaves];
        for l in 0..n_leaves {
            let base = l * k;
            let gleaf = grads[internal_count + l].as_ref().expect("leaf grad");
            let sum_g = self.reduce_sum(gleaf, n)?;
            for i in 0..n_vars {
                dw[base + i] = scale * self.reduce_dot(gleaf, &dev_cols[i], n)? + struct_lambda;
            }
            dw[base + (k - 1)] = scale * c[l] * sum_g;
            dc[l] = scale * w[base + (k - 1)] * sum_g;
        }

        // Softmax backward: dz_i = (1/tau) * w_i * (dw_i - Σ_j w_j dw_j).
        let mut dz = vec![0.0_f64; n_leaves * k];
        for l in 0..n_leaves {
            let base = l * k;
            let mut wdot = 0.0;
            for i in 0..k {
                wdot += w[base + i] * dw[base + i];
            }
            for i in 0..k {
                dz[base + i] = inv_tau * w[base + i] * (dw[base + i] - wdot);
            }
        }
        Ok((mse, dz, dc))
    }

    /// Run one Gumbel-Softmax restart on the GPU and return the hardened solution.
    fn gumbel_fit_restart(
        &self,
        ds: &DataSet,
        cfg: &Config,
        depth: usize,
        seed: u64,
    ) -> Result<Solution> {
        let n = ds.len();
        let n_vars = ds.n_vars();
        let k = n_vars + 1;
        let n_leaves = 1usize << depth;
        let internal_count = (1usize << depth) - 1;

        let dev_cols: Vec<DeviceBuffer<f32>> = (0..n_vars)
            .map(|j| {
                let col: Vec<f32> = ds.x.column(j).iter().map(|&v| v as f32).collect();
                DeviceBuffer::<f32>::from_host(&col).map_err(be)
            })
            .collect::<Result<_>>()?;
        let y_host: Vec<f32> = ds.y.iter().map(|&v| v as f32).collect();
        let dev_y = DeviceBuffer::<f32>::from_host(&y_host).map_err(be)?;

        let mut z = vec![0.0_f64; n_leaves * k];
        let mut c = vec![1.0_f64; n_leaves];
        // Adam state for z and c.
        let (b1, b2, eps) = (0.9_f64, 0.999_f64, 1e-8_f64);
        let (mut mz, mut vz) = (vec![0.0; z.len()], vec![0.0; z.len()]);
        let (mut mc, mut vc) = (vec![0.0; c.len()], vec![0.0; c.len()]);
        let struct_lambda =
            cfg.lambda_complexity + cfg.lambda_sparsity + cfg.lambda_parsimony * depth as f64;
        let mut rng = SplitMix64::new(seed);

        for epoch in 0..cfg.max_epochs {
            let tau = cfg
                .temperature(epoch as f64 / cfg.max_epochs.max(1) as f64)
                .max(1e-2);
            let gumbel: Vec<f64> = (0..n_leaves * k).map(|_| rng.gumbel()).collect();
            let (_mse, dz, dc) = self.gumbel_grad(
                &z,
                &c,
                &gumbel,
                tau,
                &dev_cols,
                &dev_y,
                n,
                n_vars,
                depth,
                struct_lambda,
            )?;
            let t = (epoch + 1) as i32;
            let bc1 = 1.0 - b1.powi(t);
            let bc2 = 1.0 - b2.powi(t);
            for j in 0..z.len() {
                mz[j] = b1 * mz[j] + (1.0 - b1) * dz[j];
                vz[j] = b2 * vz[j] + (1.0 - b2) * dz[j] * dz[j];
                z[j] -= cfg.learning_rate * (mz[j] / bc1) / ((vz[j] / bc2).sqrt() + eps);
            }
            for j in 0..c.len() {
                mc[j] = b1 * mc[j] + (1.0 - b1) * dc[j];
                vc[j] = b2 * vc[j] + (1.0 - b2) * dc[j] * dc[j];
                c[j] -= cfg.learning_rate * (mc[j] / bc1) / ((vc[j] / bc2).sqrt() + eps);
            }
        }

        // Harden: argmax each leaf's (noise-free) logits into a concrete source.
        let mut choices = Vec::with_capacity(n_leaves);
        for (l, &cl) in c.iter().enumerate() {
            let base = l * k;
            let best = (0..k)
                .max_by(|&i, &j| {
                    z[base + i]
                        .partial_cmp(&z[base + j])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            if best < n_vars {
                choices.push(LeafChoice::Var(best));
            } else {
                choices.push(LeafChoice::Const(cl));
            }
        }
        let tree = build_complete_tree(0, internal_count, &choices);
        let pred = crate::forest::eval_tree(&tree, &ds.x)?;
        Ok(Solution::new(tree, crate::fit::mse(&pred, &ds.y)))
    }
}

/// A hardened leaf choice from the Gumbel search: a variable column or a learned constant.
enum LeafChoice {
    Var(usize),
    Const(f64),
}

/// Build a concrete EML tree from a complete-tree skeleton (heap indexing) and per-leaf choices.
fn build_complete_tree(node: usize, internal_count: usize, choices: &[LeafChoice]) -> EmlTree {
    if node >= internal_count {
        match &choices[node - internal_count] {
            LeafChoice::Var(j) => EmlTree::var(*j),
            LeafChoice::Const(v) => EmlTree::const_val(*v),
        }
    } else {
        let l = build_complete_tree(2 * node + 1, internal_count, choices);
        let r = build_complete_tree(2 * node + 2, internal_count, choices);
        EmlTree::eml(&l, &r)
    }
}

/// Discover expressions by **GPU** Gumbel-Softmax topology search.
///
/// The GPU analogue of [`crate::discover_gumbel`]: runs up to `min(cfg.population, 16)` independent
/// restarts over a complete tree of depth `min(cfg.max_depth, 4)`, each trained with on-device
/// forward/backward (see [`CudaEmlEngine`]), and returns their Pareto front.
///
/// # Errors
/// Returns [`PhopError`] if CUDA is unavailable, the dataset is empty, or no restart converges.
pub fn discover_gumbel_cuda(ds: &DataSet, cfg: &Config) -> Result<ParetoFront> {
    if ds.is_empty() {
        return Err(PhopError::ShapeMismatch("empty dataset".to_string()));
    }
    let engine = CudaEmlEngine::new()?;
    let depth = cfg.max_depth.clamp(1, 4);
    let restarts = cfg.population.clamp(1, 16);
    let mut sols: Vec<Solution> = Vec::new();
    for r in 0..restarts {
        if let Ok(sol) =
            engine.gumbel_fit_restart(ds, cfg, depth, cfg.seed.wrapping_add(r as u64 + 1))
        {
            if sol.mse.is_finite() {
                sols.push(sol);
            }
        }
    }
    if sols.is_empty() {
        return Err(PhopError::NotConverged(
            "no GPU Gumbel-Softmax restart converged".to_string(),
        ));
    }
    Ok(ParetoFront::from_candidates(sols))
}

/// A node of the flattened tree used by the GPU forward/backward passes.
enum FlatNode {
    /// The constant `1`.
    One,
    /// A free constant leaf, identified by its index into the constant vector.
    Const(usize),
    /// A feature variable leaf (column index).
    Var(usize),
    /// An internal `eml` node referencing its children by flat index.
    Eml { left: usize, right: usize },
}

/// Flatten a tree into a postorder list (children before parents). Constant leaves are numbered in
/// left-to-right order, which matches the pre-order numbering used by
/// [`collect_consts`]/[`substitute_consts`], so the returned gradient aligns with the constant
/// vector.
fn build_flat(root: &EmlNode) -> Vec<FlatNode> {
    fn go(node: &EmlNode, out: &mut Vec<FlatNode>, theta: &mut usize) -> usize {
        match node {
            EmlNode::One => out.push(FlatNode::One),
            EmlNode::Var(i) => out.push(FlatNode::Var(*i)),
            EmlNode::Const(_) => {
                let j = *theta;
                *theta += 1;
                out.push(FlatNode::Const(j));
            }
            EmlNode::Eml { left, right } => {
                let l = go(left, out, theta);
                let r = go(right, out, theta);
                out.push(FlatNode::Eml { left: l, right: r });
            }
        }
        out.len() - 1
    }
    let mut out = Vec::new();
    let mut theta = 0usize;
    go(root, &mut out, &mut theta);
    out
}

/// Rebuild a tree from a flat constant vector (pre-order), substituting the `Const` leaves.
fn tree_with_consts(template: &EmlTree, consts: &[f64]) -> EmlTree {
    let mut idx = 0;
    EmlTree::from_node(substitute_consts(&template.root, consts, &mut idx))
}

/// Convenience: evaluate one tree on the GPU, creating a fresh engine.
///
/// Prefer reusing a [`CudaEmlEngine`] when evaluating many trees (it avoids re-JITing the kernel).
///
/// # Errors
/// Returns [`PhopError::Backend`] if CUDA is unavailable or the launch fails.
pub fn eval_tree_cuda(tree: &EmlTree, data: &Array2<f64>) -> Result<Array1<f64>> {
    CudaEmlEngine::new()?.eval_tree(tree, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_forward_matches_cpu_when_available() {
        if !cuda_available() {
            eprintln!("skipping GPU test: no CUDA device available");
            return;
        }
        let engine = CudaEmlEngine::new().expect("engine");

        // A depth-2 tree: eml(eml(x0, 1), x1) over a small, well-scaled batch.
        let inner = EmlTree::eml(&EmlTree::var(0), &EmlTree::one());
        let tree = EmlTree::eml(&inner, &EmlTree::var(1));

        let rows = 64usize;
        let mut data = Array2::<f64>::zeros((rows, 2));
        for i in 0..rows {
            data[[i, 0]] = i as f64 * 0.03; // x0 in [0, ~1.9]
            data[[i, 1]] = 1.0 + i as f64 * 0.05; // x1 > 0
        }

        let gpu = engine.eval_tree(&tree, &data).expect("gpu eval");
        let cpu = crate::forest::eval_tree(&tree, &data).expect("cpu eval");
        assert_eq!(gpu.len(), cpu.len());
        for i in 0..rows {
            let rel = (gpu[i] - cpu[i]).abs() / (cpu[i].abs() + 1e-6);
            assert!(
                rel < 1e-3,
                "row {i}: gpu={} cpu={} rel={rel:.3e}",
                gpu[i],
                cpu[i]
            );
        }
    }

    #[test]
    fn gpu_eval_mse_and_batch_match_cpu() {
        if !cuda_available() {
            eprintln!("skipping GPU test: no CUDA device available");
            return;
        }
        let engine = CudaEmlEngine::new().expect("engine");

        // y = exp(x0) sampled; the tree eml(x0, 1) = exp(x0) should give ~0 MSE.
        let rows = 50usize;
        let mut data = Array2::<f64>::zeros((rows, 1));
        let mut yv = vec![0.0; rows];
        for i in 0..rows {
            let x = i as f64 * 0.04;
            data[[i, 0]] = x;
            yv[i] = x.exp();
        }
        let y = Array1::from(yv);
        let exact = EmlTree::eml(&EmlTree::var(0), &EmlTree::one());
        let off = EmlTree::eml(&EmlTree::var(0), &EmlTree::const_val(2.0)); // worse fit

        let mse_exact = engine.eval_mse(&exact, &data, &y).expect("mse");
        let mse_off = engine.eval_mse(&off, &data, &y).expect("mse off");
        assert!(mse_exact < 1e-3, "exact tree GPU mse too high: {mse_exact}");
        assert!(
            mse_off > mse_exact,
            "ranking inverted: {mse_off} !> {mse_exact}"
        );

        // Batched reuse returns one prediction vector per tree.
        let preds = engine.eval_trees(&[&exact, &off], &data).expect("batch");
        assert_eq!(preds.len(), 2);
        assert_eq!(preds[0].len(), rows);
    }

    #[test]
    fn gpu_resident_fit_recovers_constant() {
        if !cuda_available() {
            eprintln!("skipping GPU test: no CUDA device available");
            return;
        }
        let engine = CudaEmlEngine::new().expect("engine");

        // Model: y = eml(x, c) = exp(x) - ln(c), true c = 3.0. Start far (c = 1.0) and let the
        // GPU-resident fit loop (forward + SSR reduction on device, FD-grad + Adam) recover it.
        let true_c = 3.0_f64;
        let rows = 24usize;
        let mut data = Array2::<f64>::zeros((rows, 1));
        let mut yv = vec![0.0; rows];
        for i in 0..rows {
            let x = (i + 1) as f64 * 0.1;
            data[[i, 0]] = x;
            yv[i] = x.exp() - true_c.ln();
        }
        let y = Array1::from(yv);
        let ds = DataSet::from_arrays(data, y).unwrap();

        let template = EmlTree::eml(&EmlTree::var(0), &EmlTree::const_val(1.0));
        let (fitted, m) = engine
            .fit_constants(&template, &ds, 0.1, 600)
            .expect("gpu fit");
        let mut consts = Vec::new();
        collect_consts(&fitted.root, &mut consts);
        assert!(
            (consts[0] - true_c).abs() < 0.1,
            "GPU-fit constant {} (want {true_c}), mse = {m}",
            consts[0]
        );
        assert!(m < 1e-2, "GPU-fit mse too high: {m}");
    }

    #[test]
    fn gpu_analytic_grad_matches_finite_difference() {
        if !cuda_available() {
            eprintln!("skipping GPU test: no CUDA device available");
            return;
        }
        let engine = CudaEmlEngine::new().expect("engine");

        // Two-constant tree eml(eml(x0, c0), c1) on a well-scaled batch (no clamp active), so the
        // on-device analytic gradient must agree with a central finite difference of the MSE.
        let rows = 40usize;
        let mut data = Array2::<f64>::zeros((rows, 1));
        let mut yv = vec![0.0; rows];
        for i in 0..rows {
            let x = i as f64 * 0.05;
            data[[i, 0]] = x;
            yv[i] = (x * 0.5).exp() + 0.3 * x; // arbitrary smooth, non-trivial target
        }
        let ds = DataSet::from_arrays(data, Array1::from(yv)).unwrap();

        let theta = [2.0_f64, 1.5_f64];
        let tree = EmlTree::eml(
            &EmlTree::eml(&EmlTree::var(0), &EmlTree::const_val(theta[0])),
            &EmlTree::const_val(theta[1]),
        );

        let (_mse, analytic) = engine.constant_grad(&tree, &ds).expect("grad");
        assert_eq!(analytic.len(), 2);

        // Central finite difference of the MSE via the public GPU forward.
        let mse_at = |t: &[f64]| -> f64 {
            let tt = tree_with_consts(&tree, t);
            engine.eval_mse(&tt, &ds.x, &ds.y).expect("mse")
        };
        for j in 0..2 {
            let h = 1e-2;
            let mut tp = theta.to_vec();
            tp[j] += h;
            let mut tm = theta.to_vec();
            tm[j] -= h;
            let fd = (mse_at(&tp) - mse_at(&tm)) / (2.0 * h);
            let rel = (analytic[j] - fd).abs() / (fd.abs() + 1e-3);
            assert!(
                rel < 5e-2,
                "constant {j}: analytic {} vs finite-diff {} (rel {rel:.3e})",
                analytic[j],
                fd
            );
        }
    }

    #[test]
    fn gpu_gumbel_grad_matches_finite_difference() {
        if !cuda_available() {
            eprintln!("skipping GPU test: no CUDA device available");
            return;
        }
        let engine = CudaEmlEngine::new().expect("engine");

        // Depth-1 skeleton, 1 feature => k=2, n_leaves=2. Check dz/dc vs finite differences of the
        // full Gumbel forward MSE at fixed noise and temperature.
        let rows = 32usize;
        let mut data = Array2::<f64>::zeros((rows, 1));
        let mut yv = vec![0.0; rows];
        for i in 0..rows {
            let x = i as f64 * 0.06;
            data[[i, 0]] = x;
            yv[i] = x.exp();
        }
        let ds = DataSet::from_arrays(data, Array1::from(yv)).unwrap();
        let dev_cols = vec![DeviceBuffer::<f32>::from_host(
            &ds.x.column(0).iter().map(|&v| v as f32).collect::<Vec<_>>(),
        )
        .unwrap()];
        let dev_y =
            DeviceBuffer::<f32>::from_host(&ds.y.iter().map(|&v| v as f32).collect::<Vec<_>>())
                .unwrap();

        let (depth, n_vars) = (1usize, 1usize);
        let k = n_vars + 1;
        let n_leaves = 1usize << depth;
        let tau = 0.7;
        let z = vec![0.3_f64, -0.2, 0.1, 0.4];
        let c = vec![1.2_f64, 0.8];
        let gumbel = vec![0.05_f64, -0.1, 0.2, -0.05];

        // MSE-only helper at given (z, c): reuse gumbel_grad's forward by reading its returned mse.
        let mse_at = |zz: &[f64], cc: &[f64]| -> f64 {
            engine
                .gumbel_grad(
                    zz, cc, &gumbel, tau, &dev_cols, &dev_y, rows, n_vars, depth, 0.0,
                )
                .unwrap()
                .0
        };
        let (_m, dz, dc) = engine
            .gumbel_grad(
                &z, &c, &gumbel, tau, &dev_cols, &dev_y, rows, n_vars, depth, 0.0,
            )
            .unwrap();
        assert_eq!(dz.len(), n_leaves * k);
        assert_eq!(dc.len(), n_leaves);

        let h = 1e-2;
        for j in 0..z.len() {
            let mut zp = z.clone();
            zp[j] += h;
            let mut zm = z.clone();
            zm[j] -= h;
            let fd = (mse_at(&zp, &c) - mse_at(&zm, &c)) / (2.0 * h);
            let rel = (dz[j] - fd).abs() / (fd.abs() + 1e-3);
            assert!(
                rel < 8e-2,
                "dz[{j}] analytic {} vs fd {} (rel {rel:.3e})",
                dz[j],
                fd
            );
        }
        for j in 0..c.len() {
            let mut cp = c.clone();
            cp[j] += h;
            let mut cm = c.clone();
            cm[j] -= h;
            let fd = (mse_at(&z, &cp) - mse_at(&z, &cm)) / (2.0 * h);
            let rel = (dc[j] - fd).abs() / (fd.abs() + 1e-3);
            assert!(
                rel < 8e-2,
                "dc[{j}] analytic {} vs fd {} (rel {rel:.3e})",
                dc[j],
                fd
            );
        }
    }

    #[test]
    fn gpu_gumbel_recovers_exp_structure() {
        if !cuda_available() {
            eprintln!("skipping GPU test: no CUDA device available");
            return;
        }
        // y = exp(x0): the depth-1 GPU Gumbel search should beat the constant predictor.
        let xs: Vec<f64> = (0..40).map(|i| f64::from(i) * 0.08).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.exp()).collect();
        let x = Array2::from_shape_vec((xs.len(), 1), xs).unwrap();
        let ds = DataSet::from_arrays(x, Array1::from(ys.clone())).unwrap();

        let cfg = Config::default()
            .max_depth(1)
            .population(6)
            .max_epochs(800)
            .learning_rate(0.1)
            .seed(3);
        let front = discover_gumbel_cuda(&ds, &cfg).expect("gpu gumbel");
        let mean = ys.iter().sum::<f64>() / ys.len() as f64;
        let var = ys.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / ys.len() as f64;
        let best = front.best().unwrap();
        assert!(
            best.mse < var * 0.5,
            "GPU gumbel best mse {} not below half-variance {} ({})",
            best.mse,
            var * 0.5,
            best.pretty()
        );
    }
}
