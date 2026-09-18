//! Smoke tests for per-axis reduction dispatch.
//!
//! GPU tests are gated behind `TENSORLOGIC_GPU_TESTS=1` and `#[ignore]`.
//!
//! CPU-oracle tests (no GPU required) validate the axis-transpose helper
//! logic indirectly by checking reduction over a small [2,3,4] tensor.
//!
//! Run GPU tests with:
//! ```bash
//! TENSORLOGIC_GPU_TESTS=1 cargo test -p tensorlogic-oxicuda-backend \
//!   --features gpu -- --include-ignored reduce
//! ```

#[cfg(feature = "gpu")]
use tensorlogic_infer::{ReduceOp, TlExecutor};
#[cfg(feature = "gpu")]
use tensorlogic_oxicuda_backend::{OxiCudaExecutor, OxiCudaTensor};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
fn gpu_tests_enabled() -> bool {
    std::env::var("TENSORLOGIC_GPU_TESTS")
        .map(|v| v == "1")
        .unwrap_or(false)
}

#[cfg(feature = "gpu")]
fn try_init_exec() -> Option<OxiCudaExecutor> {
    match OxiCudaExecutor::new() {
        Ok(e) => Some(e),
        Err(err) => {
            eprintln!("GPU executor init failed: {err}; skipping");
            None
        }
    }
}

#[cfg(feature = "gpu")]
fn make_tensor(shape: Vec<usize>, data: Vec<f32>) -> OxiCudaTensor {
    match OxiCudaTensor::new(shape, data) {
        Ok(t) => t,
        Err(e) => panic!("tensor build failed: {e}"),
    }
}

/// Build a [2,3,4] tensor with values 0..23 as f32.
#[cfg(feature = "gpu")]
fn make_2_3_4_tensor() -> OxiCudaTensor {
    let data: Vec<f32> = (0_u32..24).map(|i| i as f32).collect();
    make_tensor(vec![2, 3, 4], data)
}

#[cfg(feature = "gpu")]
fn assert_close(got: &[f32], want: &[f32], eps: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= eps,
            "{label}[{i}]: got {g}, want {w}, eps {eps}"
        );
    }
}

// ---------------------------------------------------------------------------
// CPU oracle: numpy-style sum/max reduction
// ---------------------------------------------------------------------------

/// Compute CPU sum over `axis` for a row-major `f32` tensor with given shape.
fn cpu_sum_axis(data: &[f32], shape: &[usize], axis: usize) -> (Vec<f32>, Vec<usize>) {
    let ndim = shape.len();

    let out_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter(|&(d, _)| d != axis)
        .map(|(_, &s)| s)
        .collect();
    let out_len: usize = out_shape.iter().product::<usize>().max(1);
    let mut out = vec![0.0_f32; out_len];

    let mut src_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        src_strides[i] = src_strides[i + 1] * shape[i + 1];
    }

    let out_strides: Vec<usize> = {
        let mut s = vec![1usize; ndim - 1];
        for i in (0..ndim - 2).rev() {
            s[i] = s[i + 1] * out_shape[i + 1];
        }
        s
    };

    for (flat, &val) in data.iter().enumerate() {
        let mut rem = flat;
        let mut multi: Vec<usize> = vec![0; ndim];
        for (d, m) in multi.iter_mut().enumerate() {
            *m = rem / src_strides[d];
            rem %= src_strides[d];
        }

        let mut out_flat = 0;
        let mut od = 0;
        for (d, &m) in multi.iter().enumerate() {
            if d == axis {
                continue;
            }
            out_flat += m * out_strides[od];
            od += 1;
        }

        out[out_flat] += val;
    }

    (out, out_shape)
}

/// Compute CPU max over `axis` for a row-major `f32` tensor with given shape.
#[allow(dead_code)]
fn cpu_max_axis(data: &[f32], shape: &[usize], axis: usize) -> (Vec<f32>, Vec<usize>) {
    let ndim = shape.len();
    let out_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter(|&(d, _)| d != axis)
        .map(|(_, &s)| s)
        .collect();
    let out_len: usize = out_shape.iter().product::<usize>().max(1);
    let mut out = vec![f32::NEG_INFINITY; out_len];

    let mut src_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        src_strides[i] = src_strides[i + 1] * shape[i + 1];
    }

    let out_strides: Vec<usize> = {
        let mut s = vec![1usize; ndim - 1];
        for i in (0..ndim - 2).rev() {
            s[i] = s[i + 1] * out_shape[i + 1];
        }
        s
    };

    for (flat, &val) in data.iter().enumerate() {
        let mut rem = flat;
        let mut multi: Vec<usize> = vec![0; ndim];
        for (d, m) in multi.iter_mut().enumerate() {
            *m = rem / src_strides[d];
            rem %= src_strides[d];
        }

        let mut out_flat = 0;
        let mut od = 0;
        for (d, &m) in multi.iter().enumerate() {
            if d == axis {
                continue;
            }
            out_flat += m * out_strides[od];
            od += 1;
        }

        if val > out[out_flat] {
            out[out_flat] = val;
        }
    }

    (out, out_shape)
}

// ---------------------------------------------------------------------------
// CPU oracle self-test (no GPU needed)
// ---------------------------------------------------------------------------

#[test]
fn cpu_sum_axis0_correctness() {
    // shape [2, 3]: sum axis 0 → [3]
    let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (out, shape) = cpu_sum_axis(&data, &[2, 3], 0);
    assert_eq!(shape, vec![3]);
    assert_eq!(out, vec![5.0, 7.0, 9.0]);
}

#[test]
fn cpu_sum_axis1_correctness() {
    // shape [2, 3]: sum axis 1 → [2]
    let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (out, shape) = cpu_sum_axis(&data, &[2, 3], 1);
    assert_eq!(shape, vec![2]);
    assert_eq!(out, vec![6.0, 15.0]);
}

// ---------------------------------------------------------------------------
// GPU reduce tests (ignored without env var)
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
#[test]
#[ignore]
fn gpu_reduce_sum_axis0_2x3x4() {
    if !gpu_tests_enabled() {
        eprintln!("TENSORLOGIC_GPU_TESTS != 1; skipping");
        return;
    }
    let mut exec = match try_init_exec() {
        Some(e) => e,
        None => return,
    };

    let t = make_2_3_4_tensor();
    let data = t.data.clone();

    let got = match exec.reduce(ReduceOp::Sum, &t, &[0]) {
        Ok(r) => r,
        Err(e) => panic!("reduce sum axis=0 failed: {e}"),
    };

    let (expected, exp_shape) = cpu_sum_axis(&data, &[2, 3, 4], 0);
    assert_eq!(got.shape, exp_shape, "shape mismatch");
    assert_close(&got.data, &expected, 1e-3, "sum axis=0");
}

#[cfg(feature = "gpu")]
#[test]
#[ignore]
fn gpu_reduce_sum_axis1_2x3x4() {
    if !gpu_tests_enabled() {
        eprintln!("TENSORLOGIC_GPU_TESTS != 1; skipping");
        return;
    }
    let mut exec = match try_init_exec() {
        Some(e) => e,
        None => return,
    };

    let t = make_2_3_4_tensor();
    let data = t.data.clone();

    let got = match exec.reduce(ReduceOp::Sum, &t, &[1]) {
        Ok(r) => r,
        Err(e) => panic!("reduce sum axis=1 failed: {e}"),
    };

    let (expected, exp_shape) = cpu_sum_axis(&data, &[2, 3, 4], 1);
    assert_eq!(got.shape, exp_shape, "shape mismatch");
    assert_close(&got.data, &expected, 1e-3, "sum axis=1");
}

#[cfg(feature = "gpu")]
#[test]
#[ignore]
fn gpu_reduce_sum_axis2_2x3x4() {
    if !gpu_tests_enabled() {
        eprintln!("TENSORLOGIC_GPU_TESTS != 1; skipping");
        return;
    }
    let mut exec = match try_init_exec() {
        Some(e) => e,
        None => return,
    };

    let t = make_2_3_4_tensor();
    let data = t.data.clone();

    let got = match exec.reduce(ReduceOp::Sum, &t, &[2]) {
        Ok(r) => r,
        Err(e) => panic!("reduce sum axis=2 failed: {e}"),
    };

    let (expected, exp_shape) = cpu_sum_axis(&data, &[2, 3, 4], 2);
    assert_eq!(got.shape, exp_shape, "shape mismatch");
    assert_close(&got.data, &expected, 1e-3, "sum axis=2");
}

#[cfg(feature = "gpu")]
#[test]
#[ignore]
fn gpu_reduce_max_axis0_2x3x4() {
    if !gpu_tests_enabled() {
        eprintln!("TENSORLOGIC_GPU_TESTS != 1; skipping");
        return;
    }
    let mut exec = match try_init_exec() {
        Some(e) => e,
        None => return,
    };

    let t = make_2_3_4_tensor();
    let data = t.data.clone();

    let got = match exec.reduce(ReduceOp::Max, &t, &[0]) {
        Ok(r) => r,
        Err(e) => panic!("reduce max axis=0 failed: {e}"),
    };

    let (expected, exp_shape) = cpu_max_axis(&data, &[2, 3, 4], 0);
    assert_eq!(got.shape, exp_shape, "shape mismatch");
    assert_close(&got.data, &expected, 1e-3, "max axis=0");
}

#[cfg(feature = "gpu")]
#[test]
#[ignore]
fn gpu_reduce_identity_empty_axes() {
    if !gpu_tests_enabled() {
        eprintln!("TENSORLOGIC_GPU_TESTS != 1; skipping");
        return;
    }
    let mut exec = match try_init_exec() {
        Some(e) => e,
        None => return,
    };

    let t = make_2_3_4_tensor();
    let data = t.data.clone();

    let got = match exec.reduce(ReduceOp::Sum, &t, &[]) {
        Ok(r) => r,
        Err(e) => panic!("reduce with empty axes failed: {e}"),
    };

    // Empty axes → identity (clone)
    assert_eq!(got.shape, vec![2, 3, 4]);
    assert_eq!(got.data, data);
}
