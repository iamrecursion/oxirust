//! Layer A on the GPU (M4) — Apple Metal forward evaluation of an EML tree.
//!
//! This is the optional `gpu-metal` backend, built on the cool-japan **oxicuda-metal** device/
//! memory/launch stack (the Metal sibling of the CUDA engine). It mirrors the CPU
//! [`crate::forest::eval_tree`]: a tree is evaluated bottom-up, but each `eml` node becomes a
//! single **fused** elementwise kernel launch over device buffers of length `batch`.
//!
//! The kernel is the EML primitive itself,
//! `eml(a, b) = exp(clip(a, ±C)) − ln(max(b, ε))`,
//! with the same guards as the CPU path (Risk T1). It is written in MSL (the Metal Shading
//! Language) and JIT-compiled by the Metal driver through oxicuda-metal. MSL has no `f64`, so this
//! path is **single precision**: it is a fast approximate forward for candidate screening, while
//! exact f64 scoring stays on the CPU. The guards mirror the CPU/CUDA forward exactly, so the only
//! divergence is the precision of the hardware `exp`/`log`.
//!
//! Build with `--features gpu-metal`; macOS only. [`metal_available`] reports whether a Metal
//! device is present. Unlike CUDA's RAII device buffers, Metal buffer handles are raw `u64`, so
//! every intermediate buffer is freed **explicitly** on each path, including error paths.

use crate::config::Config;
use crate::dataset::DataSet;
use crate::error::{PhopError, Result};
use crate::fit::{collect_consts, substitute_consts};
use crate::pareto::ParetoFront;
use crate::rng::SplitMix64;
use crate::solution::Solution;
use oxicuda_backend::ComputeBackend;
use oxicuda_metal::MetalBackend;
use oxieml::{EmlNode, EmlTree};
use scirs2_core::ndarray::{Array1, Array2};

/// Symmetric clamp on the `exp` argument (matches [`crate::forest::EXP_CLAMP`]).
const EXP_CLAMP: f32 = 50.0;
/// Lower clamp on the `ln` argument (matches [`crate::forest::LN_EPS`]).
const LN_EPS: f32 = 1e-12;
/// Threadgroup width for the on-device reductions; also the number of host-summed partials.
const REDUCE_TG: usize = 256;

/// The fused EML primitive as a single bounds-checked, single-precision MSL kernel:
/// `out[i] = exp(clamp(a[i], neg_clip, pos_clip)) - log(max(b[i], ln_eps))`.
///
/// The clamp/eps constants arrive as `set_bytes` scalars so the source stays exact and readable.
/// The grid is rounded up to whole threadgroups, so the kernel guards `gid` against `n`.
const EML_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void eml_elem(
    device float*       out      [[buffer(0)]],
    device const float* a        [[buffer(1)]],
    device const float* b        [[buffer(2)]],
    constant uint&      n        [[buffer(3)]],
    constant float&     neg_clip [[buffer(4)]],
    constant float&     pos_clip [[buffer(5)]],
    constant float&     ln_eps   [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    float ea = exp(clamp(a[gid], neg_clip, pos_clip));
    float lb = log(max(b[gid], ln_eps));
    out[gid] = ea - lb;
}
"#;

/// The fitting kernels (single precision MSL): the EML backward, an elementwise subtraction, and
/// two threadgroup reductions. Buffer binding follows the same rule as [`EML_MSL`]: `handles[i]`
/// binds to `[[buffer(i)]]`, then `scalar_bytes[j]` binds to `[[buffer(handles.len() + j)]]`.
///
/// `eml_back` is the reverse of `eml_elem`, with matching guards: where `eml_elem` clamps `a` into
/// `[neg_clip, pos_clip]`, the gradient w.r.t. `a` is zero outside that open interval; where it
/// floors `b` at `ln_eps`, the gradient w.r.t. `b` is zero at/below the floor. The two reductions
/// sum into `REDUCE_TG` partials (`scratch[256]` MUST equal `REDUCE_TG`), summed on the host.
const FIT_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void eml_back(
    device float* ga [[buffer(0)]], device float* gb [[buffer(1)]],
    device const float* g [[buffer(2)]], device const float* a [[buffer(3)]],
    device const float* b [[buffer(4)]], constant uint& n [[buffer(5)]],
    constant float& neg_clip [[buffer(6)]], constant float& pos_clip [[buffer(7)]],
    constant float& ln_eps [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= n) { return; }
  float av = a[gid], bv = b[gid], gv = g[gid];
  float ea = exp(clamp(av, neg_clip, pos_clip));
  ga[gid] = (av > neg_clip && av < pos_clip) ? gv * ea : 0.0f;
  float mb = max(bv, ln_eps);
  gb[gid] = (bv > ln_eps) ? gv * (-1.0f / mb) : 0.0f;
}

kernel void sub_elem(device float* out [[buffer(0)]], device const float* a [[buffer(1)]],
    device const float* b [[buffer(2)]], constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= n) { return; }
  out[gid] = a[gid] - b[gid];
}

kernel void reduce_ssr(device float* partials [[buffer(0)]], device const float* pred [[buffer(1)]],
    device const float* y [[buffer(2)]], constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]], uint tsz [[threads_per_threadgroup]],
    uint grp [[threadgroup_position_in_grid]]) {
  threadgroup float scratch[256];           // MUST equal REDUCE_TG
  float acc = 0.0f;
  for (uint i = gid; i < n; i += gsz) { float d = pred[i] - y[i]; acc += d * d; }
  scratch[lid] = acc; threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tsz >> 1; s > 0; s >>= 1) {
    if (lid < s && lid + s < tsz) { scratch[lid] += scratch[lid + s]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) { partials[grp] = scratch[0]; }
}

kernel void reduce_sum(device float* partials [[buffer(0)]], device const float* buf [[buffer(1)]],
    constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]], uint tsz [[threads_per_threadgroup]],
    uint grp [[threadgroup_position_in_grid]]) {
  threadgroup float scratch[256];           // MUST equal REDUCE_TG
  float acc = 0.0f; for (uint i = gid; i < n; i += gsz) { acc += buf[i]; }
  scratch[lid] = acc; threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tsz >> 1; s > 0; s >>= 1) {
    if (lid < s && lid + s < tsz) { scratch[lid] += scratch[lid + s]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) { partials[grp] = scratch[0]; }
}

kernel void axpy_elem(device float* out [[buffer(0)]], device const float* a [[buffer(1)]],
    constant uint& n [[buffer(2)]], constant float& alpha [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= n) { return; } out[gid] = fma(alpha, a[gid], out[gid]); }

kernel void reduce_dot(device float* partials [[buffer(0)]], device const float* a [[buffer(1)]],
    device const float* b [[buffer(2)]], constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]], uint tsz [[threads_per_threadgroup]],
    uint grp [[threadgroup_position_in_grid]]) {
  threadgroup float scratch[256];           // MUST equal REDUCE_TG
  float acc = 0.0f; for (uint i = gid; i < n; i += gsz) { acc += a[i] * b[i]; }
  scratch[lid] = acc; threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tsz >> 1; s > 0; s >>= 1) {
    if (lid < s && lid + s < tsz) { scratch[lid] += scratch[lid + s]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) { partials[grp] = scratch[0]; }
}
"#;

/// Map an oxicuda-metal error into a [`PhopError::Backend`].
fn be<E: std::fmt::Display>(e: E) -> PhopError {
    PhopError::Backend(e.to_string())
}

/// Whether a Metal device can be acquired on this machine right now.
///
/// Returns `false` (never panics) if no Metal device is present or initialization fails, so
/// callers can fall back to the CPU path.
#[must_use]
pub fn metal_available() -> bool {
    let mut b = MetalBackend::new();
    b.init().is_ok()
}

/// A reusable Metal evaluator: holds the initialized backend and its internally cached pipeline.
///
/// Construct once and evaluate many trees; the MSL kernel is JIT-compiled a single time and cached
/// by the backend (keyed by function name and source hash), so repeated launches reuse it.
pub struct MetalEmlEngine {
    backend: MetalBackend,
}

impl MetalEmlEngine {
    /// Create and initialize the Metal backend (acquires the system-default device).
    ///
    /// # Errors
    /// Returns [`PhopError::Backend`] if no Metal device can be acquired (e.g. on non-macOS).
    pub fn new() -> Result<Self> {
        let mut backend = MetalBackend::new();
        backend.init().map_err(be)?;
        Ok(Self { backend })
    }

    /// Evaluate `tree` over `data` (`[batch, n_vars]`) on the GPU, returning `[batch]`
    /// predictions in `f64` (computed in single precision on-device).
    ///
    /// # Errors
    /// Returns [`PhopError::Backend`] on any Metal failure, or
    /// [`PhopError::NumericalInstability`] if the result contains non-finite values.
    pub fn eval_tree(&self, tree: &EmlTree, data: &Array2<f64>) -> Result<Array1<f64>> {
        let n = data.nrows();
        if n == 0 {
            return Ok(Array1::zeros(0));
        }
        let n_vars = data.ncols();
        // Host-side f32 feature columns; uploaded on demand at each variable leaf.
        let host_cols: Vec<Vec<f32>> = (0..n_vars)
            .map(|j| data.column(j).iter().map(|&v| v as f32).collect())
            .collect();

        let root = self.eval_node(&tree.root, &host_cols, n)?;
        let mut bytes = vec![0u8; n * std::mem::size_of::<f32>()];
        let copy = self.backend.copy_dtoh(&mut bytes, root);
        // Free the root buffer on every path before propagating a copy error.
        let _ = self.backend.free(root);
        copy.map_err(be)?;

        let values: Vec<f64> = bytes
            .chunks_exact(4)
            .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect();
        if values.iter().any(|v| !v.is_finite()) {
            return Err(PhopError::NumericalInstability(
                "Metal forward produced non-finite values".to_string(),
            ));
        }
        Ok(Array1::from(values))
    }

    /// Evaluate `tree` on the GPU and return its mean-squared error against `y`.
    ///
    /// # Errors
    /// Returns [`PhopError`] on any Metal failure or non-finite output.
    pub fn eval_mse(&self, tree: &EmlTree, data: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        Ok(crate::fit::mse(&self.eval_tree(tree, data)?, y))
    }

    /// Evaluate many trees over the same data on the GPU, reusing this engine (and its cached
    /// pipeline) across all of them. Returns one prediction vector per tree.
    ///
    /// # Errors
    /// Returns the first [`PhopError`] encountered.
    pub fn eval_trees(&self, trees: &[&EmlTree], data: &Array2<f64>) -> Result<Vec<Array1<f64>>> {
        trees.iter().map(|t| self.eval_tree(t, data)).collect()
    }

    /// The human-readable name of the Metal device backing this engine.
    ///
    /// Falls back to the backend's own name (`"metal"`) when no detailed device list is exposed.
    #[must_use]
    pub fn device_name(&self) -> String {
        if let Ok(devices) = ComputeBackend::available_devices(&self.backend) {
            if let Some(first) = devices.first() {
                if !first.name.is_empty() {
                    return first.name.clone();
                }
            }
        }
        ComputeBackend::name(&self.backend).to_string()
    }

    /// Recursively evaluate a node into a Metal device buffer of length `n`, returning an owned
    /// handle the **caller** must free. Leak-safe: on any error every buffer allocated here (or by
    /// a child) is freed before the error propagates.
    fn eval_node(&self, node: &EmlNode, host_cols: &[Vec<f32>], n: usize) -> Result<u64> {
        match node {
            EmlNode::One => self.const_buffer(1.0, n),
            EmlNode::Const(c) => self.const_buffer(*c as f32, n),
            EmlNode::Var(i) => self.upload_f32(&host_cols[*i]),
            EmlNode::Eml { left, right } => {
                let a = self.eval_node(left, host_cols, n)?;
                let b = match self.eval_node(right, host_cols, n) {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = self.backend.free(a);
                        return Err(e);
                    }
                };
                let out = match self.backend.alloc(n * std::mem::size_of::<f32>()) {
                    Ok(out) => out,
                    Err(e) => {
                        let _ = self.backend.free(a);
                        let _ = self.backend.free(b);
                        return Err(be(e));
                    }
                };
                let launched = self.launch_eml(out, a, b, n);
                let _ = self.backend.free(a);
                let _ = self.backend.free(b);
                match launched {
                    Ok(()) => Ok(out),
                    Err(e) => {
                        let _ = self.backend.free(out);
                        Err(e)
                    }
                }
            }
        }
    }

    /// Allocate a device buffer of length `n` filled with the constant `v`.
    fn const_buffer(&self, v: f32, n: usize) -> Result<u64> {
        self.upload_f32(&vec![v; n])
    }

    /// Upload an `f32` host slice to a fresh device buffer, returning its handle. Leak-safe: the
    /// buffer is freed if the host→device copy fails.
    fn upload_f32(&self, host: &[f32]) -> Result<u64> {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(host));
        for &v in host {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let handle = self.backend.alloc(bytes.len()).map_err(be)?;
        if let Err(e) = self.backend.copy_htod(handle, &bytes) {
            let _ = self.backend.free(handle);
            return Err(be(e));
        }
        Ok(handle)
    }

    /// Launch the fused elementwise EML kernel: `out = eml(a, b)`.
    ///
    /// `handles[i]` binds to MSL `[[buffer(i)]]`; the clamp/eps scalars follow at
    /// `[[buffer(handles.len() + j)]]`.
    fn launch_eml(&self, out: u64, a: u64, b: u64, n: usize) -> Result<()> {
        let n_le = (n as u32).to_le_bytes();
        let neg_le = (-EXP_CLAMP).to_le_bytes();
        let pos_le = EXP_CLAMP.to_le_bytes();
        let eps_le = LN_EPS.to_le_bytes();
        self.backend
            .launch_custom_kernel(
                EML_MSL,
                "eml_elem",
                &[out, a, b],
                &[&n_le, &neg_le, &pos_le, &eps_le],
                n,
            )
            .map_err(be)
    }

    /// Launch the elementwise subtraction `out = a - b`.
    ///
    /// `handles[i]` binds to MSL `[[buffer(i)]]`; the length scalar follows at `[[buffer(3)]]`.
    fn launch_sub(&self, out: u64, a: u64, b: u64, n: usize) -> Result<()> {
        let n_le = (n as u32).to_le_bytes();
        self.backend
            .launch_custom_kernel(FIT_MSL, "sub_elem", &[out, a, b], &[&n_le], n)
            .map_err(be)
    }

    /// Launch the eml backward kernel, writing child gradients `ga`, `gb` from upstream `g` and the
    /// node's child outputs `a`, `b`. Mirrors the guards of [`Self::launch_eml`].
    fn launch_back(&self, ga: u64, gb: u64, g: u64, a: u64, b: u64, n: usize) -> Result<()> {
        let n_le = (n as u32).to_le_bytes();
        let neg = (-EXP_CLAMP).to_le_bytes();
        let pos = EXP_CLAMP.to_le_bytes();
        let eps = LN_EPS.to_le_bytes();
        self.backend
            .launch_custom_kernel(
                FIT_MSL,
                "eml_back",
                &[ga, gb, g, a, b],
                &[&n_le, &neg, &pos, &eps],
                n,
            )
            .map_err(be)
    }

    /// Run a threadgroup reduction kernel (`reduce_ssr`/`reduce_sum`) and sum its `REDUCE_TG`
    /// partials on the host. Leak-safe: the partials buffer is freed on every path.
    fn reduce_partials(&self, function_name: &str, data_handles: &[u64], n: usize) -> Result<f64> {
        let zeros = vec![0.0f32; REDUCE_TG];
        let partials = self.upload_f32(&zeros)?; // alloc + zero in one
        let mut handles = Vec::with_capacity(data_handles.len() + 1);
        handles.push(partials);
        handles.extend_from_slice(data_handles);
        let n_le = (n as u32).to_le_bytes();
        let launched = self.backend.launch_custom_kernel(
            FIT_MSL,
            function_name,
            &handles,
            &[&n_le],
            REDUCE_TG,
        );
        let mut bytes = vec![0u8; REDUCE_TG * 4];
        let read = launched
            .map_err(be)
            .and_then(|()| self.backend.copy_dtoh(&mut bytes, partials).map_err(be));
        let _ = self.backend.free(partials);
        read?;
        let sum: f64 = bytes
            .chunks_exact(4)
            .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .sum();
        Ok(sum)
    }

    /// Sum of squared residuals `Σ (pred - y)^2`, reduced on-device.
    fn reduce_ssr(&self, pred: u64, dev_y: u64, n: usize) -> Result<f64> {
        self.reduce_partials("reduce_ssr", &[pred, dev_y], n)
    }

    /// Sum of a device buffer `Σ buf`, reduced on-device.
    fn reduce_sum(&self, buf: u64, n: usize) -> Result<f64> {
        self.reduce_partials("reduce_sum", &[buf], n)
    }

    /// Launch the in-place AXPY `out += alpha * a` (single precision, via `fma`).
    ///
    /// `handles[i]` binds to MSL `[[buffer(i)]]`; the length scalar follows at `[[buffer(2)]]`
    /// and the `alpha` scalar at `[[buffer(3)]]`.
    fn launch_axpy(&self, out: u64, alpha: f32, a: u64, n: usize) -> Result<()> {
        let n_le = (n as u32).to_le_bytes();
        let alpha_le = alpha.to_le_bytes();
        self.backend
            .launch_custom_kernel(FIT_MSL, "axpy_elem", &[out, a], &[&n_le, &alpha_le], n)
            .map_err(be)
    }

    /// Reduce the dot product `Σ a[i] * b[i]` on-device.
    fn reduce_dot(&self, a: u64, b: u64, n: usize) -> Result<f64> {
        self.reduce_partials("reduce_dot", &[a, b], n)
    }

    /// Forward-evaluate `node` against the host-resident columns (re-uploading each variable leaf)
    /// and reduce the MSE on-device. Reuses the leak-safe [`Self::eval_node`].
    fn forward_mse(
        &self,
        node: &EmlNode,
        host_cols: &[Vec<f32>],
        dev_y: u64,
        n: usize,
    ) -> Result<f64> {
        let pred = self.eval_node(node, host_cols, n)?;
        let m = self.reduce_ssr(pred, dev_y, n);
        let _ = self.backend.free(pred);
        Ok(m? / n.max(1) as f64)
    }

    /// Forward (storing every node's output) + reverse pass over the host-resident columns and the
    /// device-resident target. Returns `(mse, grad)` where `grad[j] = dMSE/dθ_j`. Leak-safe: every
    /// device buffer allocated by the inner pass is freed here on every path.
    fn forward_grad(
        &self,
        flat: &[FlatNode],
        theta: &[f64],
        host_cols: &[Vec<f32>],
        dev_y: u64,
        n: usize,
    ) -> Result<(f64, Vec<f64>)> {
        let mut vals: Vec<u64> = Vec::with_capacity(flat.len());
        let mut grads: Vec<Option<u64>> = (0..flat.len()).map(|_| None).collect();
        let out = self.forward_grad_inner(flat, theta, host_cols, dev_y, n, &mut vals, &mut grads);
        for &h in &vals {
            let _ = self.backend.free(h);
        }
        for g in grads.into_iter().flatten() {
            let _ = self.backend.free(g);
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_grad_inner(
        &self,
        flat: &[FlatNode],
        theta: &[f64],
        host_cols: &[Vec<f32>],
        dev_y: u64,
        n: usize,
        vals: &mut Vec<u64>,
        grads: &mut [Option<u64>],
    ) -> Result<(f64, Vec<f64>)> {
        let p = theta.len();
        let root = flat.len() - 1;
        // Forward postorder: ALLOC then push into `vals` BEFORE any fallible launch, so cleanup frees it.
        for node in flat {
            match node {
                FlatNode::One => {
                    let h = self.const_buffer(1.0, n)?;
                    vals.push(h);
                }
                FlatNode::Const(j) => {
                    let h = self.const_buffer(theta[*j] as f32, n)?;
                    vals.push(h);
                }
                FlatNode::Var(i) => {
                    let h = self.upload_f32(&host_cols[*i])?;
                    vals.push(h);
                }
                FlatNode::Eml { left, right } => {
                    let (l, r) = (vals[*left], vals[*right]);
                    let out = self
                        .backend
                        .alloc(n * std::mem::size_of::<f32>())
                        .map_err(be)?;
                    vals.push(out); // recorded before the launch
                    self.launch_eml(out, l, r, n)?;
                }
            }
        }
        let mse = self.reduce_ssr(vals[root], dev_y, n)? / n.max(1) as f64;
        // Backward: seed g_root = pred - y, recorded before the sub launch.
        let g_root = self
            .backend
            .alloc(n * std::mem::size_of::<f32>())
            .map_err(be)?;
        grads[root] = Some(g_root);
        self.launch_sub(g_root, vals[root], dev_y, n)?;
        for i in (0..flat.len()).rev() {
            if let FlatNode::Eml { left, right } = &flat[i] {
                let g = grads[i].ok_or_else(|| be("missing parent gradient"))?;
                let ga = self
                    .backend
                    .alloc(n * std::mem::size_of::<f32>())
                    .map_err(be)?;
                grads[*left] = Some(ga);
                let gb = self
                    .backend
                    .alloc(n * std::mem::size_of::<f32>())
                    .map_err(be)?;
                grads[*right] = Some(gb);
                self.launch_back(ga, gb, g, vals[*left], vals[*right], n)?;
            }
        }
        let mut grad = vec![0.0_f64; p];
        for (i, node) in flat.iter().enumerate() {
            if let FlatNode::Const(j) = node {
                if let Some(gbuf) = grads[i] {
                    grad[*j] = 2.0 * self.reduce_sum(gbuf, n)? / n.max(1) as f64;
                }
            }
        }
        Ok((mse, grad))
    }

    /// Return the MSE and the **analytic** (reverse-mode) gradient of every constant leaf of
    /// `template` against `ds`, at the template's current constant values.
    ///
    /// The Metal analogue of the CUDA `constant_grad`: feature columns are host-resident and the
    /// target is uploaded once (`dev_y`) and freed on every path. Single precision.
    ///
    /// # Errors
    /// Returns [`PhopError`] on any Metal failure.
    pub fn constant_grad(&self, template: &EmlTree, ds: &DataSet) -> Result<(f64, Vec<f64>)> {
        let n = ds.len();
        let mut theta = Vec::new();
        collect_consts(&template.root, &mut theta);
        let p = theta.len();
        if n == 0 {
            return Ok((0.0, vec![0.0; p]));
        }
        let n_vars = ds.n_vars();
        let host_cols: Vec<Vec<f32>> = (0..n_vars)
            .map(|j| ds.x.column(j).iter().map(|&v| v as f32).collect())
            .collect();
        let y_host: Vec<f32> = ds.y.iter().map(|&v| v as f32).collect();
        let dev_y = self.upload_f32(&y_host)?;
        let flat = build_flat(&template.root);
        let out = self.forward_grad(&flat, &theta, &host_cols, dev_y, n);
        let _ = self.backend.free(dev_y);
        out
    }

    /// Fit the free `Const` leaves of a fixed-topology `template` to `ds` on the GPU via Apple Metal.
    ///
    /// The Metal analogue of [`crate::fit::fit_constants`]: each step runs one forward (sum-of-squared
    /// residuals reduced on-device) and one reverse pass yielding the analytic gradient of all
    /// constants at once, then host-side Adam updates them. Feature columns stay host-resident and are
    /// re-uploaded per variable leaf; only the target is device-resident (`dev_y`). The kernels are
    /// MSL; this path is single precision. A constant-free template is just evaluated. `learning_rate`
    /// and `max_epochs` mirror [`crate::Config`].
    ///
    /// # Errors
    /// Returns [`PhopError`] on any Metal failure.
    pub fn fit_constants(
        &self,
        template: &EmlTree,
        ds: &DataSet,
        learning_rate: f64,
        max_epochs: usize,
    ) -> Result<(EmlTree, f64)> {
        let n = ds.len();
        if n == 0 {
            return Ok((template.clone(), 0.0));
        }
        let n_vars = ds.n_vars();
        let host_cols: Vec<Vec<f32>> = (0..n_vars)
            .map(|j| ds.x.column(j).iter().map(|&v| v as f32).collect())
            .collect();
        let y_host: Vec<f32> = ds.y.iter().map(|&v| v as f32).collect();
        let dev_y = self.upload_f32(&y_host)?;
        let out = self.fit_inner(template, &host_cols, dev_y, n, learning_rate, max_epochs);
        let _ = self.backend.free(dev_y);
        out
    }

    fn fit_inner(
        &self,
        template: &EmlTree,
        host_cols: &[Vec<f32>],
        dev_y: u64,
        n: usize,
        learning_rate: f64,
        max_epochs: usize,
    ) -> Result<(EmlTree, f64)> {
        let mut theta = Vec::new();
        collect_consts(&template.root, &mut theta);
        let p = theta.len();
        if p == 0 {
            let m = self.forward_mse(&template.root, host_cols, dev_y, n)?;
            return Ok((template.clone(), m));
        }
        let flat = build_flat(&template.root);
        let (b1, b2, eps) = (0.9_f64, 0.999_f64, 1e-8_f64);
        let mut m_adam = vec![0.0_f64; p];
        let mut v_adam = vec![0.0_f64; p];
        for t in 1..=max_epochs {
            let (_mse, grad) = self.forward_grad(&flat, &theta, host_cols, dev_y, n)?;
            let bc1 = 1.0 - b1.powi(t as i32);
            let bc2 = 1.0 - b2.powi(t as i32);
            for j in 0..p {
                m_adam[j] = b1 * m_adam[j] + (1.0 - b1) * grad[j];
                v_adam[j] = b2 * v_adam[j] + (1.0 - b2) * grad[j] * grad[j];
                let mhat = m_adam[j] / bc1;
                let vhat = v_adam[j] / bc2;
                theta[j] -= learning_rate * mhat / (vhat.sqrt() + eps);
            }
        }
        let fitted = tree_with_consts(template, &theta);
        let m = self.forward_mse(&fitted.root, host_cols, dev_y, n)?;
        Ok((fitted, m))
    }

    // ----- GPU-resident Gumbel-Softmax topology search (Layer B / M4) -------------------------

    /// One forward + reverse pass of the Gumbel-Softmax forest at the given logits/constants and
    /// sampled noise, returning `(mse, dz, dc)` — the gradients w.r.t. the leaf selection logits
    /// `z` (`n_leaves*k`) and leaf constants `c` (`n_leaves`).
    ///
    /// The per-leaf soft selection (a `softmax` over `k = n_vars + 1` sources) is computed on the
    /// host (the parameters are tiny); the per-row heavy work — the weighted source combination,
    /// the `eml` tree, the SSR and the gradient dot-products — runs on the GPU via Apple Metal. The
    /// structural penalty (`struct_lambda * Σ variable-weight mass`) is folded into `dz` via
    /// `dL/dw`. `dev_cols`/`dev_y` are **borrowed** (the caller owns and frees them); every
    /// intermediate device buffer allocated here is freed on every path.
    #[allow(clippy::too_many_arguments)]
    fn gumbel_grad(
        &self,
        z: &[f64],
        c: &[f64],
        gumbel: &[f64],
        tau: f64,
        dev_cols: &[u64],
        dev_y: u64,
        n: usize,
        n_vars: usize,
        depth: usize,
        struct_lambda: f64,
    ) -> Result<(f64, Vec<f64>, Vec<f64>)> {
        let k = n_vars + 1;
        let n_leaves = 1usize << depth;
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

        let total = (1usize << (depth + 1)) - 1;
        let mut vals: Vec<Option<u64>> = (0..total).map(|_| None).collect();
        let mut grads: Vec<Option<u64>> = (0..total).map(|_| None).collect();
        let out = self.gumbel_grad_inner(
            &w,
            c,
            dev_cols,
            dev_y,
            n,
            n_vars,
            depth,
            struct_lambda,
            inv_tau,
            &mut vals,
            &mut grads,
        );
        for h in vals.into_iter().flatten() {
            let _ = self.backend.free(h);
        }
        for g in grads.into_iter().flatten() {
            let _ = self.backend.free(g);
        }
        out
    }

    /// The GPU forward/backward + final `dw`/`dz`/`dc` assembly behind [`Self::gumbel_grad`].
    ///
    /// Receives the already-softmaxed per-leaf weights `w`, the leaf constants `c`, the borrowed
    /// resident columns/target, and `inv_tau = 1/tau` (needed by the softmax-backward step; the
    /// outer fn consumes `tau` in the host softmax, so it is passed through here). Each device
    /// allocation is recorded into `vals[i]`/`grads[i]` **immediately, before** the consuming
    /// launch, so the outer cleanup frees it on every path. Returns `(mse, dz, dc)`.
    #[allow(clippy::too_many_arguments)]
    fn gumbel_grad_inner(
        &self,
        w: &[f64],
        c: &[f64],
        dev_cols: &[u64],
        dev_y: u64,
        n: usize,
        n_vars: usize,
        depth: usize,
        struct_lambda: f64,
        inv_tau: f64,
        vals: &mut [Option<u64>],
        grads: &mut [Option<u64>],
    ) -> Result<(f64, Vec<f64>, Vec<f64>)> {
        let k = n_vars + 1;
        let n_leaves = 1usize << depth;
        let internal_count = (1usize << depth) - 1;

        // Forward: seed each leaf with its constant mass, then AXPY in each variable source.
        for l in 0..n_leaves {
            let base = l * k;
            let leaf = self.const_buffer((w[base + (k - 1)] * c[l]) as f32, n)?;
            vals[internal_count + l] = Some(leaf);
            for i in 0..n_vars {
                self.launch_axpy(leaf, w[base + i] as f32, dev_cols[i], n)?;
            }
        }
        for i in (0..internal_count).rev() {
            let outb = self
                .backend
                .alloc(n * std::mem::size_of::<f32>())
                .map_err(be)?;
            vals[i] = Some(outb);
            let a = vals[2 * i + 1].ok_or_else(|| be("missing child"))?;
            let b = vals[2 * i + 2].ok_or_else(|| be("missing child"))?;
            self.launch_eml(outb, a, b, n)?;
        }
        let pred = vals[0].ok_or_else(|| be("missing root"))?;
        let mse = self.reduce_ssr(pred, dev_y, n)? / n.max(1) as f64;

        // Backward through the tree: seed g_root = pred - y, then top-down.
        let g0 = self
            .backend
            .alloc(n * std::mem::size_of::<f32>())
            .map_err(be)?;
        grads[0] = Some(g0);
        self.launch_sub(g0, pred, dev_y, n)?;
        for i in 0..internal_count {
            let g = grads[i].ok_or_else(|| be("missing parent grad"))?;
            let ga = self
                .backend
                .alloc(n * std::mem::size_of::<f32>())
                .map_err(be)?;
            grads[2 * i + 1] = Some(ga);
            let gb = self
                .backend
                .alloc(n * std::mem::size_of::<f32>())
                .map_err(be)?;
            grads[2 * i + 2] = Some(gb);
            let a = vals[2 * i + 1].ok_or_else(|| be("missing child"))?;
            let b = vals[2 * i + 2].ok_or_else(|| be("missing child"))?;
            self.launch_back(ga, gb, g, a, b, n)?;
        }

        // Per-leaf source weight + constant gradients (scaled by 2/n; penalty added to var weights).
        let scale = 2.0 / n.max(1) as f64;
        let mut dw = vec![0.0_f64; n_leaves * k];
        let mut dc = vec![0.0_f64; n_leaves];
        for l in 0..n_leaves {
            let base = l * k;
            let gleaf = grads[internal_count + l].ok_or_else(|| be("missing leaf grad"))?;
            let sum_g = self.reduce_sum(gleaf, n)?;
            for i in 0..n_vars {
                dw[base + i] = scale * self.reduce_dot(gleaf, dev_cols[i], n)? + struct_lambda;
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
    ///
    /// Owns the resident buffers: it uploads the feature columns and target once, runs
    /// [`Self::gumbel_fit_inner`], then frees `dev_cols`/`dev_y` on **every** path (success or
    /// error). Leak-safe even when a column upload fails partway through.
    fn gumbel_fit_restart(
        &self,
        ds: &DataSet,
        cfg: &Config,
        depth: usize,
        seed: u64,
    ) -> Result<Solution> {
        let n_vars = ds.n_vars();
        let mut dev_cols = Vec::with_capacity(n_vars);
        for j in 0..n_vars {
            let col: Vec<f32> = ds.x.column(j).iter().map(|&v| v as f32).collect();
            match self.upload_f32(&col) {
                Ok(h) => dev_cols.push(h),
                Err(e) => {
                    for &h in &dev_cols {
                        let _ = self.backend.free(h);
                    }
                    return Err(e);
                }
            }
        }
        let y_host: Vec<f32> = ds.y.iter().map(|&v| v as f32).collect();
        let dev_y = match self.upload_f32(&y_host) {
            Ok(h) => h,
            Err(e) => {
                for &h in &dev_cols {
                    let _ = self.backend.free(h);
                }
                return Err(e);
            }
        };
        let out = self.gumbel_fit_inner(ds, cfg, depth, seed, &dev_cols, dev_y);
        for &h in &dev_cols {
            let _ = self.backend.free(h);
        }
        let _ = self.backend.free(dev_y);
        out
    }

    /// Host-side Adam over the leaf logits `z` and constants `c`, driven by [`Self::gumbel_grad`]
    /// against the borrowed resident buffers, then argmax-hardened into a concrete EML tree.
    ///
    /// A verbatim port of the CUDA restart loop; only the gradient call uses raw `u64` handles.
    #[allow(clippy::too_many_arguments)]
    fn gumbel_fit_inner(
        &self,
        ds: &DataSet,
        cfg: &Config,
        depth: usize,
        seed: u64,
        dev_cols: &[u64],
        dev_y: u64,
    ) -> Result<Solution> {
        let n = ds.len();
        let n_vars = ds.n_vars();
        let k = n_vars + 1;
        let n_leaves = 1usize << depth;
        let internal_count = (1usize << depth) - 1;
        let mut z = vec![0.0_f64; n_leaves * k];
        let mut c = vec![1.0_f64; n_leaves];
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
                dev_cols,
                dev_y,
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

/// Convenience: evaluate one tree on the GPU via Apple Metal, creating a fresh engine.
///
/// Prefer reusing a [`MetalEmlEngine`] when evaluating many trees (it avoids re-compiling and
/// re-caching the kernel).
///
/// # Errors
/// Returns [`PhopError::Backend`] if Metal is unavailable or the launch fails.
pub fn eval_tree_metal(tree: &EmlTree, data: &Array2<f64>) -> Result<Array1<f64>> {
    MetalEmlEngine::new()?.eval_tree(tree, data)
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

/// Discover expressions by **Metal GPU** Gumbel-Softmax topology search.
///
/// The Apple Metal analogue of the CUDA `discover_gumbel_cuda`: runs up to
/// `min(cfg.population, 16)` independent restarts over a complete tree of depth
/// `min(cfg.max_depth, 4)`, each trained with on-device forward/backward, returning their Pareto front.
///
/// # Errors
/// Returns [`PhopError`] if Metal is unavailable, the dataset is empty, or no restart converges.
pub fn discover_gumbel_metal(ds: &DataSet, cfg: &Config) -> Result<ParetoFront> {
    if ds.is_empty() {
        return Err(PhopError::ShapeMismatch("empty dataset".to_string()));
    }
    let engine = MetalEmlEngine::new()?;
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
            "no Metal Gumbel-Softmax restart converged".to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_forward_matches_cpu_when_available() -> Result<()> {
        if !metal_available() {
            eprintln!("skipping Metal test: no Metal device available");
            return Ok(());
        }
        let engine = match MetalEmlEngine::new() {
            Ok(e) => e,
            Err(e) => {
                eprintln!("skipping Metal test: engine init failed: {e}");
                return Ok(());
            }
        };
        eprintln!("Metal device: {}", engine.device_name());

        // Single-precision MSL exp/log: tol kept tight on well-scaled (small, positive) inputs.
        const TOL: f64 = 1e-4;
        let mut max_rel = 0.0_f64;

        // (1) A depth-2 tree: eml(eml(x0, 1), x1) = exp(exp(x0)) - ln(x1) over a well-scaled batch.
        let inner = EmlTree::eml(&EmlTree::var(0), &EmlTree::one());
        let tree = EmlTree::eml(&inner, &EmlTree::var(1));

        let rows = 64usize;
        let mut data = Array2::<f64>::zeros((rows, 2));
        for i in 0..rows {
            data[[i, 0]] = i as f64 * 0.01; // x0 in [0, ~0.63]
            data[[i, 1]] = 1.0 + i as f64 * 0.05; // x1 > 0
        }

        let gpu = engine.eval_tree(&tree, &data)?;
        let cpu = crate::forest::eval_tree(&tree, &data)?;
        assert_eq!(gpu.len(), cpu.len());
        for i in 0..rows {
            let rel = (gpu[i] - cpu[i]).abs() / (cpu[i].abs() + 1e-6);
            max_rel = max_rel.max(rel);
            assert!(
                rel < TOL,
                "tree1 row {i}: gpu={} cpu={} rel={rel:.3e}",
                gpu[i],
                cpu[i]
            );
        }

        // (2) eml(x0, 1) = exp(x0) over a single-column dataset.
        let tree2 = EmlTree::eml(&EmlTree::var(0), &EmlTree::one());
        let mut data2 = Array2::<f64>::zeros((rows, 1));
        for i in 0..rows {
            data2[[i, 0]] = i as f64 * 0.01;
        }
        let gpu2 = engine.eval_tree(&tree2, &data2)?;
        let cpu2 = crate::forest::eval_tree(&tree2, &data2)?;
        assert_eq!(gpu2.len(), cpu2.len());
        for i in 0..rows {
            let rel = (gpu2[i] - cpu2[i]).abs() / (cpu2[i].abs() + 1e-6);
            max_rel = max_rel.max(rel);
            assert!(
                rel < TOL,
                "tree2 row {i}: gpu={} cpu={} rel={rel:.3e}",
                gpu2[i],
                cpu2[i]
            );
        }

        eprintln!("Metal forward max relative error vs CPU: {max_rel:.3e} (tol {TOL:.1e})");
        Ok(())
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "gpu-metal"))]
    fn metal_analytic_grad_matches_finite_difference() -> Result<()> {
        if !metal_available() {
            eprintln!("skipping Metal test: no Metal device available");
            return Ok(());
        }
        let engine = match MetalEmlEngine::new() {
            Ok(e) => e,
            Err(e) => {
                eprintln!("skipping Metal test: engine init failed: {e}");
                return Ok(());
            }
        };

        let rows = 40usize;
        let mut data = Array2::<f64>::zeros((rows, 1));
        let mut yv = vec![0.0; rows];
        for i in 0..rows {
            let x = i as f64 * 0.05;
            data[[i, 0]] = x;
            yv[i] = (x * 0.5).exp() + 0.3 * x;
        }
        let ds = DataSet::from_arrays(data, Array1::from(yv))?;

        let theta = [2.0_f64, 1.5_f64];
        let tree = EmlTree::eml(
            &EmlTree::eml(&EmlTree::var(0), &EmlTree::const_val(theta[0])),
            &EmlTree::const_val(theta[1]),
        );

        let (_mse, analytic) = engine.constant_grad(&tree, &ds)?;
        assert_eq!(analytic.len(), 2);

        let mse_at = |t: &[f64]| -> Result<f64> {
            let tt = tree_with_consts(&tree, t);
            engine.eval_mse(&tt, &ds.x, &ds.y)
        };
        for j in 0..2 {
            let h = 1e-2;
            let mut tp = theta.to_vec();
            tp[j] += h;
            let mut tm = theta.to_vec();
            tm[j] -= h;
            let fd = (mse_at(&tp)? - mse_at(&tm)?) / (2.0 * h);
            let rel = (analytic[j] - fd).abs() / (fd.abs() + 1e-3);
            eprintln!(
                "metal grad const {j}: analytic {} fd {} rel {rel:.3e}",
                analytic[j], fd
            );
            assert!(rel < 5e-2);
        }
        Ok(())
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "gpu-metal"))]
    fn metal_resident_fit_recovers_constant() -> Result<()> {
        if !metal_available() {
            eprintln!("skipping Metal test: no Metal device available");
            return Ok(());
        }
        let engine = match MetalEmlEngine::new() {
            Ok(e) => e,
            Err(e) => {
                eprintln!("skipping Metal test: engine init failed: {e}");
                return Ok(());
            }
        };

        let true_c = 3.0_f64;
        let rows = 24usize;
        let mut data = Array2::<f64>::zeros((rows, 1));
        let mut yv = vec![0.0; rows];
        for i in 0..rows {
            let x = (i + 1) as f64 * 0.1;
            data[[i, 0]] = x;
            yv[i] = x.exp() - true_c.ln();
        }
        let ds = DataSet::from_arrays(data, Array1::from(yv))?;

        let template = EmlTree::eml(&EmlTree::var(0), &EmlTree::const_val(1.0));
        let (fitted, m) = engine.fit_constants(&template, &ds, 0.1, 600)?;

        let mut consts = Vec::new();
        collect_consts(&fitted.root, &mut consts);
        eprintln!(
            "metal recovered constant {} (true {true_c}), mse {m:.3e}",
            consts[0]
        );
        assert!((consts[0] - true_c).abs() < 0.1);
        assert!(m < 1e-2);
        Ok(())
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "gpu-metal"))]
    fn metal_fit_reduces_error_like_cpu() -> Result<()> {
        if !metal_available() {
            eprintln!("skipping Metal test: no Metal device available");
            return Ok(());
        }
        let engine = match MetalEmlEngine::new() {
            Ok(e) => e,
            Err(e) => {
                eprintln!("skipping Metal test: engine init failed: {e}");
                return Ok(());
            }
        };

        let rows = 15usize;
        let mut data = Array2::<f64>::zeros((rows, 1));
        let mut yv = vec![0.0; rows];
        for i in 0..rows {
            let x = (i + 1) as f64 * 0.2;
            data[[i, 0]] = x;
            yv[i] = x.exp() - 5.0_f64.ln();
        }
        let ds = DataSet::from_arrays(data, Array1::from(yv))?;

        let template = EmlTree::eml(&EmlTree::var(0), &EmlTree::const_val(1.0));
        let before = engine.eval_mse(&template, &ds.x, &ds.y)?;
        let (_, after) = engine.fit_constants(&template, &ds, 0.05, 4000)?;
        assert!(after < before * 0.5);

        let cfg = crate::config::Config::default()
            .learning_rate(0.05)
            .max_epochs(4000);
        let (_, cpu_after) = crate::fit::fit_constants(&template, &ds, &cfg)?;
        eprintln!("metal fit before {before:.3e} after {after:.3e} cpu_after {cpu_after:.3e}");
        assert!(after <= cpu_after * 2.0 + 1e-6);
        Ok(())
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "gpu-metal"))]
    fn metal_gumbel_grad_matches_finite_difference() -> Result<()> {
        if !metal_available() {
            eprintln!("skipping Metal test: no Metal device available");
            return Ok(());
        }
        let engine = match MetalEmlEngine::new() {
            Ok(e) => e,
            Err(e) => {
                eprintln!("skipping Metal test: engine init failed: {e}");
                return Ok(());
            }
        };

        let rows = 32usize;
        let mut data = Array2::<f64>::zeros((rows, 1));
        let mut yv = vec![0.0; rows];
        for i in 0..rows {
            let x = i as f64 * 0.06;
            data[[i, 0]] = x;
            yv[i] = x.exp();
        }
        let ds = DataSet::from_arrays(data, Array1::from(yv))?;

        let dev_cols =
            vec![engine.upload_f32(&ds.x.column(0).iter().map(|&v| v as f32).collect::<Vec<_>>())?];
        let dev_y = engine.upload_f32(&ds.y.iter().map(|&v| v as f32).collect::<Vec<_>>())?;

        let (depth, n_vars) = (1usize, 1usize);
        let k = n_vars + 1;
        let n_leaves = 1usize << depth;
        let tau = 0.7;
        let z = vec![0.3_f64, -0.2, 0.1, 0.4];
        let c = vec![1.2_f64, 0.8];
        let gumbel = vec![0.05_f64, -0.1, 0.2, -0.05];

        let mse_at = |zz: &[f64], cc: &[f64]| -> Result<f64> {
            Ok(engine
                .gumbel_grad(
                    zz, cc, &gumbel, tau, &dev_cols, dev_y, rows, n_vars, depth, 0.0,
                )?
                .0)
        };
        let (_m, dz, dc) = engine.gumbel_grad(
            &z, &c, &gumbel, tau, &dev_cols, dev_y, rows, n_vars, depth, 0.0,
        )?;
        assert_eq!(dz.len(), n_leaves * k);
        assert_eq!(dc.len(), n_leaves);

        let h = 1e-2;
        for j in 0..z.len() {
            let mut zp = z.clone();
            zp[j] += h;
            let mut zm = z.clone();
            zm[j] -= h;
            let fd = (mse_at(&zp, &c)? - mse_at(&zm, &c)?) / (2.0 * h);
            let rel = (dz[j] - fd).abs() / (fd.abs() + 1e-3);
            eprintln!("metal dz[{j}] analytic {} fd {} rel {rel:.3e}", dz[j], fd);
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
            let fd = (mse_at(&z, &cp)? - mse_at(&z, &cm)?) / (2.0 * h);
            let rel = (dc[j] - fd).abs() / (fd.abs() + 1e-3);
            eprintln!("metal dc[{j}] analytic {} fd {} rel {rel:.3e}", dc[j], fd);
            assert!(
                rel < 8e-2,
                "dc[{j}] analytic {} vs fd {} (rel {rel:.3e})",
                dc[j],
                fd
            );
        }

        for &h in &dev_cols {
            let _ = engine.backend.free(h);
        }
        let _ = engine.backend.free(dev_y);
        Ok(())
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "gpu-metal"))]
    fn metal_gumbel_recovers_exp_structure() -> Result<()> {
        if !metal_available() {
            eprintln!("skipping Metal test: no Metal device available");
            return Ok(());
        }
        let xs: Vec<f64> = (0..40).map(|i| f64::from(i) * 0.08).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.exp()).collect();
        let x = Array2::from_shape_vec((xs.len(), 1), xs)
            .map_err(|e| PhopError::ShapeMismatch(e.to_string()))?;
        let ds = DataSet::from_arrays(x, Array1::from(ys.clone()))?;
        let cfg = Config::default()
            .max_depth(1)
            .population(6)
            .max_epochs(800)
            .learning_rate(0.1)
            .seed(3);
        let front = discover_gumbel_metal(&ds, &cfg)?;
        let mean = ys.iter().sum::<f64>() / ys.len() as f64;
        let var = ys.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / ys.len() as f64;
        let best = front
            .best()
            .ok_or_else(|| PhopError::NotConverged("empty front".to_string()))?;
        eprintln!(
            "metal gumbel best mse {} half-var {} ({})",
            best.mse,
            var * 0.5,
            best.pretty()
        );
        assert!(
            best.mse < var * 0.5,
            "Metal gumbel best mse {} not below half-variance {} ({})",
            best.mse,
            var * 0.5,
            best.pretty()
        );
        Ok(())
    }
}
