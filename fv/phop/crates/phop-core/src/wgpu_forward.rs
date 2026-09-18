//! Portable GPU forward evaluation of an EML tree via wgpu (opt-in `gpu-wgpu` feature).
//!
//! This is the cross-platform sibling of the CUDA backend ([`crate::gpu`]): it runs the EML forward
//! pass on any wgpu adapter — WebGPU, Metal, Vulkan, or DX12 — so phop's GPU path is no longer
//! NVIDIA-only. It is selected by [`crate::accel::gpu_backend`] (CUDA → wgpu → CPU).
//!
//! **Precision.** wgpu/WGSL has no `f64`, so this path evaluates in **`f32`** (the CUDA and CPU paths
//! stay `f64`). For phop that is fine: the GPU produces a fast coarse forward, and the Levenberg–
//! Marquardt polish re-sharpens constants in `f64` on the CPU.
//!
//! **Approach.** Rather than orchestrate one kernel per node, the EML tree is **compiled to a single
//! WGSL compute shader** (one expression evaluated per data row), then dispatched once. The guards
//! mirror the CPU/CUDA forward exactly: `eml(a, b) = exp(clamp(a, ±50)) − ln(clamp(b, [1e-12, max]))`.
//!
//! We use `scirs2-core`'s low-level [`WebGPUContext`] for adapter/device setup (its high-level
//! `GpuContext(Wgpu)` path does not detect wgpu adapters) and drive the pipeline with raw wgpu, which
//! is why phop depends on the same wgpu 29 that `scirs2-core` does.

use crate::error::{PhopError, Result};
use oxieml::{EmlNode, EmlTree};
use scirs2_core::gpu::backends::WebGPUContext;
use scirs2_core::ndarray::{Array1, Array2};
use std::fmt::Write as _;
use wgpu::util::DeviceExt as _;

/// EML guard constants, mirrored from [`crate::forest`] so the GPU and CPU forwards agree.
const EXP_CLAMP: f32 = 50.0;
const LN_EPS: f32 = 1e-12;

/// Whether a usable wgpu **compute device** can be created at runtime.
///
/// This is stricter than merely probing for an adapter: it attempts full device creation, so a
/// machine whose only adapter is a limited software rasterizer (e.g. `llvmpipe`, which reports zero
/// compute workgroups) reports `false` and [`crate::accel::gpu_backend`] correctly falls back to CPU
/// instead of selecting an unusable backend.
#[must_use]
pub fn wgpu_available() -> bool {
    WebGPUContext::new().is_ok()
}

/// Emit a WGSL `f32` literal for a constant (always decimal, never an `i32`-typed bare integer).
fn fmt_const(c: f64) -> String {
    format!("f32({})", c as f32)
}

/// Recursively emit the WGSL expression for an EML node. Variables read row `r`'s slice of the
/// row-major input buffer at `base = r * n_vars`.
fn emit(node: &EmlNode, out: &mut String) {
    match node {
        EmlNode::One => out.push_str("1.0"),
        EmlNode::Const(c) => out.push_str(&fmt_const(*c)),
        EmlNode::Var(i) => {
            // Ignore the formatting result: writing to a String never fails.
            let _ = write!(out, "input[base + {i}u]");
        }
        EmlNode::Eml { left, right } => {
            out.push_str("(g_exp(");
            emit(left, out);
            out.push_str(") - g_ln(");
            emit(right, out);
            out.push(')');
            out.push(')');
        }
    }
}

/// Build the full WGSL compute shader for `tree`.
fn build_shader(tree: &EmlTree) -> String {
    let mut expr = String::new();
    emit(&tree.root, &mut expr);
    format!(
        r#"
@group(0) @binding(0) var<storage, read>       input  : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;

struct Dims {{ n_rows: u32, n_vars: u32, pad0: u32, pad1: u32 }};
@group(0) @binding(2) var<uniform> dims : Dims;

fn g_exp(x: f32) -> f32 {{ return exp(clamp(x, -{exp_clamp}, {exp_clamp})); }}
fn g_ln(x: f32)  -> f32 {{ return log(clamp(x, {ln_eps}, 3.4e38)); }}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let r = gid.x;
    if (r >= dims.n_rows) {{ return; }}
    let base = r * dims.n_vars;
    output[r] = {expr};
}}
"#,
        exp_clamp = EXP_CLAMP,
        ln_eps = LN_EPS,
        expr = expr,
    )
}

/// Evaluate `tree` over `data` (`[n_rows, n_vars]`) on a wgpu adapter, returning `f32`-precision
/// predictions widened to `f64`.
///
/// # Errors
/// Returns [`PhopError::Eval`] if no adapter is available or any wgpu operation fails, and
/// [`PhopError::NumericalInstability`] if the result contains non-finite values.
pub fn eval_tree_wgpu(tree: &EmlTree, data: &Array2<f64>) -> Result<Array1<f64>> {
    let n_rows = data.nrows();
    let n_vars = data.ncols();
    if n_rows == 0 {
        return Ok(Array1::zeros(0));
    }

    let ctx = WebGPUContext::new().map_err(|e| PhopError::Eval(format!("wgpu init: {e:?}")))?;
    let device = ctx.device();
    let queue = ctx.queue();

    // Row-major f32 input.
    let mut input: Vec<f32> = Vec::with_capacity(n_rows * n_vars);
    for r in 0..n_rows {
        for c in 0..n_vars {
            input.push(data[[r, c]] as f32);
        }
    }
    let input_bytes: Vec<u8> = input.iter().flat_map(|f| f.to_le_bytes()).collect();
    let dims: [u32; 4] = [n_rows as u32, n_vars as u32, 0, 0];
    let dims_bytes: Vec<u8> = dims.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out_size = (n_rows * std::mem::size_of::<f32>()) as u64;

    let shader = build_shader(tree);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("phop-eml-forward"),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("phop-eml-bgl"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, false),
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("phop-eml-layout"),
        bind_group_layouts: &[Some(&bgl)],
        ..Default::default()
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("phop-eml-pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("phop-eml-input"),
        contents: &input_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let dims_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("phop-eml-dims"),
        contents: &dims_bytes,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("phop-eml-output"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("phop-eml-bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: dims_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("phop-eml-encoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("phop-eml-pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let groups = (n_rows as u32).div_ceil(64);
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("phop-eml-staging"),
        size: out_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_size);
    queue.submit(Some(encoder.finish()));

    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| PhopError::Eval(format!("wgpu poll: {e:?}")))?;

    let slice = staging.slice(0..out_size);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| PhopError::Eval(format!("wgpu poll (map): {e:?}")))?;
    rx.recv()
        .map_err(|_| PhopError::Eval("wgpu map channel closed".into()))?
        .map_err(|e| PhopError::Eval(format!("wgpu map_async: {e:?}")))?;

    let mapped = slice.get_mapped_range();
    let out: Vec<f64> = mapped
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)
        .collect();
    drop(mapped);
    staging.unmap();

    if out.iter().any(|v| !v.is_finite()) {
        return Err(PhopError::NumericalInstability(
            "wgpu forward produced non-finite values".to_string(),
        ));
    }
    Ok(Array1::from(out))
}

/// A read-only or read-write storage-buffer bind-group-layout entry at `binding`.
fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_codegen_is_well_formed() {
        // exp(x0): eml(x0, 1) → guarded exp/ln, reads input[base + 0u].
        let tree = EmlTree::eml(&EmlTree::var(0), &EmlTree::one());
        let src = build_shader(&tree);
        assert!(src.contains("g_exp(input[base + 0u])"));
        assert!(src.contains("g_ln(1.0)"));
        assert!(src.contains("@workgroup_size(64)"));
    }

    #[test]
    fn wgpu_forward_matches_cpu_when_available() {
        // Runtime GPU execution is opt-in: it requires a real hardware adapter. Many CI / headless
        // hosts only expose a software rasterizer (llvmpipe) that lacks compute and can even segfault
        // on teardown, so this test runs only when explicitly requested. The pure codegen test above
        // always runs.
        if std::env::var("PHOP_WGPU_RUNTIME_TEST").is_err() {
            eprintln!("skipping wgpu forward runtime test: set PHOP_WGPU_RUNTIME_TEST to enable");
            return;
        }
        if !wgpu_available() {
            eprintln!("skipping wgpu forward test: no usable adapter");
            return;
        }
        // y = exp(x0) over a benign range; f32 GPU must match the CPU within f32 tolerance.
        let xs: Vec<f64> = (0..16).map(|i| f64::from(i) * 0.1).collect();
        let data = Array2::from_shape_vec((xs.len(), 1), xs.clone()).expect("shape");
        let tree = EmlTree::eml(&EmlTree::var(0), &EmlTree::one());
        let gpu = eval_tree_wgpu(&tree, &data).expect("wgpu forward");
        let cpu = crate::forest::eval_tree(&tree, &data).expect("cpu forward");
        for i in 0..xs.len() {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1e-3,
                "row {i}: gpu={} cpu={}",
                gpu[i],
                cpu[i]
            );
        }
    }
}
