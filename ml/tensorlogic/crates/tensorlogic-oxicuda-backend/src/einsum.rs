//! Einsum spec parser and kernel dispatcher for the OxiCUDA backend.
//!
//! # Supported specs
//!
//! | Spec pattern                    | Dispatch                          |
//! |---------------------------------|-----------------------------------|
//! | `"ij,jk->ik"`                  | 2-D SGEMM via `gemm_api::gemm`    |
//! | `"bij,bjk->bik"` (batched)     | Strided batched GEMM              |
//! | Identity (`"ij->ij"` etc.)     | Clone the input                   |
//! | Unknown / unsupported           | `UnsupportedSpec` error            |
//!
//! # Design decisions
//!
//! The parser normalises whitespace, then pattern-matches the full spec string.
//! A proper expression parser is follow-up work; at MVP correctness of the
//! three common cases is sufficient.

#[cfg(any(test, feature = "gpu"))]
use crate::error::OxiCudaBackendError;
#[cfg(feature = "gpu")]
use crate::executor::OxiCudaTensor;

#[cfg(feature = "gpu")]
use oxicuda_blas::batched::strided_gemm;
#[cfg(feature = "gpu")]
use oxicuda_blas::handle::BlasHandle;
#[cfg(feature = "gpu")]
use oxicuda_blas::level3::gemm_api;
#[cfg(feature = "gpu")]
use oxicuda_blas::types::{Layout, MatrixDesc, MatrixDescMut, Transpose};
#[cfg(feature = "gpu")]
use oxicuda_driver::ffi::CUdeviceptr;
#[cfg(feature = "gpu")]
use oxicuda_memory::DeviceBuffer;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Normalise a spec string by stripping whitespace and lower-casing.
///
/// Available in both the CPU test path and the GPU dispatch path.
#[cfg(any(test, feature = "gpu"))]
fn normalise(spec: &str) -> String {
    spec.chars()
        .filter(|c| !c.is_whitespace())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// Convert `usize` → `u32`, returning a `DimensionOverflow` error on failure.
#[cfg(feature = "gpu")]
fn to_u32(dim: usize, name: &str) -> Result<u32, OxiCudaBackendError> {
    u32::try_from(dim).map_err(|_| {
        OxiCudaBackendError::DimensionOverflow(format!(
            "einsum: {name} dimension {dim} exceeds u32::MAX"
        ))
    })
}

/// Extract the raw device pointer from a [`DeviceBuffer`] as a [`CUdeviceptr`].
#[cfg(feature = "gpu")]
fn buf_ptr<T: Copy>(buf: &DeviceBuffer<T>) -> CUdeviceptr {
    buf.as_device_ptr()
}

// ---------------------------------------------------------------------------
// 2-D matmul: "ij,jk->ik"
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
fn matmul_2d(
    handle: &BlasHandle,
    inputs: &[OxiCudaTensor],
) -> Result<OxiCudaTensor, OxiCudaBackendError> {
    if inputs.len() != 2 {
        return Err(OxiCudaBackendError::InvalidEinsumSpec(format!(
            "'ij,jk->ik' expects 2 inputs, got {}",
            inputs.len()
        )));
    }
    let a = &inputs[0];
    let b = &inputs[1];
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(OxiCudaBackendError::InvalidShape(format!(
            "matmul expects 2-D tensors, got shapes {:?} and {:?}",
            a.shape, b.shape
        )));
    }
    if a.shape[1] != b.shape[0] {
        return Err(OxiCudaBackendError::InvalidShape(format!(
            "inner dims mismatch: A{:?} · B{:?}",
            a.shape, b.shape
        )));
    }
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];

    let m32 = to_u32(m, "M")?;
    let k32 = to_u32(k, "K")?;
    let n32 = to_u32(n, "N")?;

    let mut d_a = DeviceBuffer::<f32>::from_host(&a.data)?;
    let mut d_b = DeviceBuffer::<f32>::from_host(&b.data)?;
    let mut d_c = DeviceBuffer::<f32>::zeroed(m * n)?;

    let desc_a = MatrixDesc::<f32>::from_buffer(&d_a, m32, k32, Layout::RowMajor)?;
    let desc_b = MatrixDesc::<f32>::from_buffer(&d_b, k32, n32, Layout::RowMajor)?;
    let mut desc_c = MatrixDescMut::<f32>::from_buffer(&mut d_c, m32, n32, Layout::RowMajor)?;

    // Keep d_a / d_b alive (they are owned DeviceBuffers, not raw pointers)
    let _ = &mut d_a;
    let _ = &mut d_b;

    gemm_api::gemm(
        handle,
        Transpose::NoTrans,
        Transpose::NoTrans,
        1.0_f32,
        &desc_a,
        &desc_b,
        0.0_f32,
        &mut desc_c,
    )?;

    handle.stream().synchronize()?;

    let mut host_c = vec![0.0_f32; m * n];
    d_c.copy_to_host(&mut host_c)?;

    OxiCudaTensor::new(vec![m, n], host_c)
}

// ---------------------------------------------------------------------------
// Batched matmul: "bij,bjk->bik"
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
fn matmul_batched(
    handle: &BlasHandle,
    inputs: &[OxiCudaTensor],
) -> Result<OxiCudaTensor, OxiCudaBackendError> {
    if inputs.len() != 2 {
        return Err(OxiCudaBackendError::InvalidEinsumSpec(format!(
            "'bij,bjk->bik' expects 2 inputs, got {}",
            inputs.len()
        )));
    }
    let a = &inputs[0];
    let b = &inputs[1];
    if a.shape.len() != 3 || b.shape.len() != 3 {
        return Err(OxiCudaBackendError::InvalidShape(format!(
            "batched matmul expects 3-D tensors, got {:?} and {:?}",
            a.shape, b.shape
        )));
    }
    if a.shape[0] != b.shape[0] {
        return Err(OxiCudaBackendError::InvalidShape(format!(
            "batch dims mismatch: A.batch={} B.batch={}",
            a.shape[0], b.shape[0]
        )));
    }
    if a.shape[2] != b.shape[1] {
        return Err(OxiCudaBackendError::InvalidShape(format!(
            "inner dims mismatch for batched matmul: A[2]={} B[1]={}",
            a.shape[2], b.shape[1]
        )));
    }

    let batch = a.shape[0];
    let m = a.shape[1];
    let k = a.shape[2];
    let n = b.shape[2];

    let batch32 = to_u32(batch, "batch")?;
    let m32 = to_u32(m, "M")?;
    let k32 = to_u32(k, "K")?;
    let n32 = to_u32(n, "N")?;

    let d_a = DeviceBuffer::<f32>::from_host(&a.data)?;
    let d_b = DeviceBuffer::<f32>::from_host(&b.data)?;
    let d_c_zero = DeviceBuffer::<f32>::zeroed(batch * m * n)?;
    let d_d = DeviceBuffer::<f32>::zeroed(batch * m * n)?;

    // Strides are in element counts (not bytes).
    let stride_a = (m * k) as i64;
    let stride_b = (k * n) as i64;
    let stride_c = (m * n) as i64; // c is read with beta=0 so content doesn't matter
    let stride_d = (m * n) as i64;

    // Leading dimensions for row-major: lda = cols of op(A) (no-trans) = k
    let lda = k32;
    let ldb = n32;
    let ldc = n32;
    let ldd = n32;

    strided_gemm::gemm_strided_batched::<f32>(
        handle,
        Transpose::NoTrans,
        Transpose::NoTrans,
        m32,
        n32,
        k32,
        1.0_f32,
        buf_ptr(&d_a),
        lda,
        stride_a,
        buf_ptr(&d_b),
        ldb,
        stride_b,
        0.0_f32,
        buf_ptr(&d_c_zero),
        ldc,
        stride_c,
        buf_ptr(&d_d),
        ldd,
        stride_d,
        batch32,
    )?;

    handle.stream().synchronize()?;

    let total = batch * m * n;
    let mut host_d = vec![0.0_f32; total];
    d_d.copy_to_host(&mut host_d)?;

    OxiCudaTensor::new(vec![batch, m, n], host_d)
}

// ---------------------------------------------------------------------------
// 2-D GEMM with configurable transpose flags
// ---------------------------------------------------------------------------

/// 2-D GEMM with configurable transpose flags.
///
/// Computes `C = op(A) · op(B)` where op is Identity (NoTrans) or Transpose (Trans).
///
/// `a` has stored shape `[a_rows, a_cols]`.
/// `b` has stored shape `[b_rows, b_cols]`.
/// Output shape is `[M, N]` where:
///   M = if trans_a == NoTrans { a_rows } else { a_cols }
///   N = if trans_b == NoTrans { b_cols } else { b_rows }
///   K = if trans_a == NoTrans { a_cols } else { a_rows }  (must equal the K from B)
#[cfg(feature = "gpu")]
pub(crate) fn matmul_2d_trans_flags(
    handle: &BlasHandle,
    a: &OxiCudaTensor,
    trans_a: Transpose,
    b: &OxiCudaTensor,
    trans_b: Transpose,
) -> Result<OxiCudaTensor, OxiCudaBackendError> {
    if a.shape.len() != 2 {
        return Err(OxiCudaBackendError::InvalidShape(format!(
            "matmul_2d_trans_flags: A must be 2-D, got shape {:?}",
            a.shape
        )));
    }
    if b.shape.len() != 2 {
        return Err(OxiCudaBackendError::InvalidShape(format!(
            "matmul_2d_trans_flags: B must be 2-D, got shape {:?}",
            b.shape
        )));
    }

    // M, K from A (using stored shape)
    let (m, k_from_a) = if trans_a == Transpose::NoTrans {
        (a.shape[0], a.shape[1])
    } else {
        (a.shape[1], a.shape[0])
    };

    // K, N from B (using stored shape)
    let (k_from_b, n) = if trans_b == Transpose::NoTrans {
        (b.shape[0], b.shape[1])
    } else {
        (b.shape[1], b.shape[0])
    };

    if k_from_a != k_from_b {
        return Err(OxiCudaBackendError::InvalidShape(format!(
            "matmul_2d_trans_flags: inner dims mismatch: K_from_A={k_from_a} K_from_B={k_from_b} \
             (A stored {:?} trans={trans_a:?}, B stored {:?} trans={trans_b:?})",
            a.shape, b.shape
        )));
    }

    let a_rows32 = to_u32(a.shape[0], "A_rows")?;
    let a_cols32 = to_u32(a.shape[1], "A_cols")?;
    let b_rows32 = to_u32(b.shape[0], "B_rows")?;
    let b_cols32 = to_u32(b.shape[1], "B_cols")?;
    let m32 = to_u32(m, "M")?;
    let n32 = to_u32(n, "N")?;

    let mut d_a = DeviceBuffer::<f32>::from_host(&a.data)?;
    let mut d_b = DeviceBuffer::<f32>::from_host(&b.data)?;
    let mut d_c = DeviceBuffer::<f32>::zeroed(m * n)?;

    // MatrixDesc uses the STORED shape of the buffer (not the op(A) shape).
    let desc_a = MatrixDesc::<f32>::from_buffer(&d_a, a_rows32, a_cols32, Layout::RowMajor)?;
    let desc_b = MatrixDesc::<f32>::from_buffer(&d_b, b_rows32, b_cols32, Layout::RowMajor)?;
    let mut desc_c = MatrixDescMut::<f32>::from_buffer(&mut d_c, m32, n32, Layout::RowMajor)?;

    // Keep d_a / d_b alive until after the gemm call.
    let _ = &mut d_a;
    let _ = &mut d_b;

    gemm_api::gemm(
        handle,
        trans_a,
        trans_b,
        1.0_f32,
        &desc_a,
        &desc_b,
        0.0_f32,
        &mut desc_c,
    )?;

    handle.stream().synchronize()?;

    let mut host_c = vec![0.0_f32; m * n];
    d_c.copy_to_host(&mut host_c)?;

    OxiCudaTensor::new(vec![m, n], host_c)
}

// ---------------------------------------------------------------------------
// Batched GEMM with configurable transpose flags
// ---------------------------------------------------------------------------

/// Batched 3-D GEMM with configurable transpose flags.
///
/// Computes `D[b] = op(A[b]) · op(B[b])` for each batch slice.
///
/// `a` has stored shape `[batch, a_rows, a_cols]`.
/// `b` has stored shape `[batch, b_rows, b_cols]`.
/// Output shape is `[batch, M, N]` where:
///   M = if trans_a { a_cols } else { a_rows }
///   N = if trans_b { b_rows } else { b_cols }
///   K = if trans_a { a_rows } else { a_cols }  (must equal K from B)
#[cfg(feature = "gpu")]
pub(crate) fn matmul_batched_trans_flags(
    handle: &BlasHandle,
    a: &OxiCudaTensor,
    trans_a: Transpose,
    b: &OxiCudaTensor,
    trans_b: Transpose,
) -> Result<OxiCudaTensor, OxiCudaBackendError> {
    if a.shape.len() != 3 {
        return Err(OxiCudaBackendError::InvalidShape(format!(
            "matmul_batched_trans_flags: A must be 3-D, got shape {:?}",
            a.shape
        )));
    }
    if b.shape.len() != 3 {
        return Err(OxiCudaBackendError::InvalidShape(format!(
            "matmul_batched_trans_flags: B must be 3-D, got shape {:?}",
            b.shape
        )));
    }
    if a.shape[0] != b.shape[0] {
        return Err(OxiCudaBackendError::InvalidShape(format!(
            "matmul_batched_trans_flags: batch dims mismatch: A.batch={} B.batch={}",
            a.shape[0], b.shape[0]
        )));
    }

    let batch = a.shape[0];

    // M, K from A (using stored inner dims)
    let (m, k_from_a) = if trans_a == Transpose::NoTrans {
        (a.shape[1], a.shape[2])
    } else {
        (a.shape[2], a.shape[1])
    };

    // K, N from B (using stored inner dims)
    let (k_from_b, n) = if trans_b == Transpose::NoTrans {
        (b.shape[1], b.shape[2])
    } else {
        (b.shape[2], b.shape[1])
    };

    if k_from_a != k_from_b {
        return Err(OxiCudaBackendError::InvalidShape(format!(
            "matmul_batched_trans_flags: inner dims mismatch: K_from_A={k_from_a} K_from_B={k_from_b} \
             (A stored {:?} trans={trans_a:?}, B stored {:?} trans={trans_b:?})",
            a.shape, b.shape
        )));
    }

    let batch32 = to_u32(batch, "batch")?;
    let m32 = to_u32(m, "M")?;
    let n32 = to_u32(n, "N")?;
    let k32 = to_u32(k_from_a, "K")?;

    // Leading dimensions are always the STORED last dim (row-major stride = last dim).
    let lda = to_u32(a.shape[2], "lda")?;
    let ldb = to_u32(b.shape[2], "ldb")?;
    let ldd = n32;

    // Strides are in element counts per batch slice.
    let stride_a = (a.shape[1] * a.shape[2]) as i64;
    let stride_b = (b.shape[1] * b.shape[2]) as i64;
    let stride_d = (m * n) as i64;

    // A zero-content C buffer (beta=0 so its values are ignored, but the
    // strided_gemm API requires a valid pointer and matching dimensions).
    let ldc = ldd;
    let stride_c = stride_d;

    let d_a = DeviceBuffer::<f32>::from_host(&a.data)?;
    let d_b = DeviceBuffer::<f32>::from_host(&b.data)?;
    let d_c_zero = DeviceBuffer::<f32>::zeroed(batch * m * n)?;
    let d_d = DeviceBuffer::<f32>::zeroed(batch * m * n)?;

    strided_gemm::gemm_strided_batched::<f32>(
        handle,
        trans_a,
        trans_b,
        m32,
        n32,
        k32,
        1.0_f32,
        buf_ptr(&d_a),
        lda,
        stride_a,
        buf_ptr(&d_b),
        ldb,
        stride_b,
        0.0_f32,
        buf_ptr(&d_c_zero),
        ldc,
        stride_c,
        buf_ptr(&d_d),
        ldd,
        stride_d,
        batch32,
    )?;

    handle.stream().synchronize()?;

    let total = batch * m * n;
    let mut host_d = vec![0.0_f32; total];
    d_d.copy_to_host(&mut host_d)?;

    OxiCudaTensor::new(vec![batch, m, n], host_d)
}

// ---------------------------------------------------------------------------
// Identity dispatch: single input, same indices in and out
// ---------------------------------------------------------------------------

/// Return an identity tensor for specs like `"ij->ij"` where no computation
/// is needed — just clone the input.
#[cfg(feature = "gpu")]
fn identity_dispatch(
    inputs: &[OxiCudaTensor],
    spec: &str,
) -> Result<OxiCudaTensor, OxiCudaBackendError> {
    if inputs.len() != 1 {
        return Err(OxiCudaBackendError::InvalidEinsumSpec(format!(
            "identity spec '{spec}' expects 1 input, got {}",
            inputs.len()
        )));
    }
    Ok(inputs[0].clone())
}

// ---------------------------------------------------------------------------
// Public dispatch entry
// ---------------------------------------------------------------------------

/// Parse an einsum spec and dispatch to the correct GPU kernel.
///
/// # Supported forms
///
/// - `"ij,jk->ik"` — 2-D matrix multiplication.
/// - `"bij,bjk->bik"` — Strided batched 3-D matrix multiplication.
/// - Single-input identity specs where all input indices appear verbatim in
///   the output (e.g. `"ij->ij"`, `"abc->abc"`).
/// - Everything else returns [`OxiCudaBackendError::UnsupportedSpec`].
#[cfg(feature = "gpu")]
pub fn dispatch_einsum(
    handle: &BlasHandle,
    spec: &str,
    inputs: &[OxiCudaTensor],
) -> Result<OxiCudaTensor, OxiCudaBackendError> {
    let norm = normalise(spec);
    match norm.as_str() {
        "ij,jk->ik" => matmul_2d(handle, inputs),
        "bij,bjk->bik" => matmul_batched(handle, inputs),
        other => {
            // Check if it is a single-input identity spec
            if is_identity_spec(other) {
                identity_dispatch(inputs, spec)
            } else {
                Err(OxiCudaBackendError::UnsupportedSpec(spec.to_string()))
            }
        }
    }
}

/// Returns `true` when the spec is a single-input identity: one comma-free
/// input, one `->`, and the output indices are exactly the input indices in the
/// same order.
///
/// Examples: `"ij->ij"`, `"abc->abc"`.
#[cfg(any(test, feature = "gpu"))]
fn is_identity_spec(norm: &str) -> bool {
    // Must contain exactly one `->`
    let arrow_count = norm.matches("->").count();
    if arrow_count != 1 {
        return false;
    }
    // Split on `->`
    let parts: Vec<&str> = norm.splitn(2, "->").collect();
    if parts.len() != 2 {
        return false;
    }
    let lhs = parts[0];
    let rhs = parts[1];
    // No comma in lhs (single input) and identical labels
    !lhs.contains(',') && lhs == rhs
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalise_strips_spaces() {
        assert_eq!(normalise("i j , j k -> i k"), "ij,jk->ik");
    }

    #[test]
    fn normalise_lowercases() {
        assert_eq!(normalise("IJ,JK->IK"), "ij,jk->ik");
    }

    #[test]
    fn is_identity_spec_true_for_ij_to_ij() {
        assert!(is_identity_spec("ij->ij"));
        assert!(is_identity_spec("abc->abc"));
        assert!(is_identity_spec("i->i"));
    }

    #[test]
    fn is_identity_spec_false_for_batched_matmul() {
        assert!(!is_identity_spec("bij,bjk->bik"));
    }

    #[test]
    fn is_identity_spec_false_for_permuted() {
        // transpose: not identity
        assert!(!is_identity_spec("ij->ji"));
    }

    #[test]
    fn is_identity_spec_false_for_multi_input() {
        assert!(!is_identity_spec("ij,jk->ik"));
    }

    #[test]
    fn unsupported_spec_error_message() {
        let e = OxiCudaBackendError::UnsupportedSpec("weird->spec".to_string());
        let msg = e.to_string();
        assert!(msg.contains("weird->spec"), "got: {msg}");
    }
}
