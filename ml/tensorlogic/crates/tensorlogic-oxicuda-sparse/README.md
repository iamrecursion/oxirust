# tensorlogic-oxicuda-sparse

GPU-accelerated sparse matrix operations for TensorLogic with pure-Rust CPU fallback.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../../LICENSE)
[![Status](https://img.shields.io/badge/status-Alpha-yellow.svg)]()
[![Tests](https://img.shields.io/badge/tests-27%2F27-brightgreen.svg)]()

Provides Compressed Sparse Row (CSR) matrices with sparse matrix-vector (SpMV) and
sparse matrix-matrix (SpMM) operations. The CPU path is pure-Rust; enabling `--features gpu`
routes to OxiCUDA sparse kernels (NVIDIA driver required, no CUDA SDK).

## Feature flags

| Feature | Default | Effect |
|---------|---------|--------|
| `cpu`   | yes     | Pure-Rust CPU sparse operations. Always compiled as fallback. |
| `gpu`   | no      | Enables `oxicuda-sparse`, `oxicuda-driver`, `oxicuda-memory`, `oxicuda-blas`. Requires an NVIDIA driver at runtime — no CUDA SDK needed. |

## Quick start

```rust
use tensorlogic_oxicuda_sparse::{SparseCsr, spmv, spmm, SparseError};

fn main() -> Result<(), SparseError> {
    // Build a 3×3 sparse matrix from triplets (row, col, val)
    // [[1, 0, 2],
    //  [0, 3, 0],
    //  [4, 0, 5]]
    let rows = vec![0, 0, 1, 2, 2];
    let cols = vec![0, 2, 1, 0, 2];
    let vals = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let mat = SparseCsr::from_triplets(&rows, &cols, &vals, 3, 3)?;

    println!("nnz: {}", mat.nnz());       // 5
    println!("shape: {:?}", mat.shape()); // (3, 3)

    // Sparse matrix-vector multiply: y = mat * x
    let x = vec![1.0f32, 1.0, 1.0];
    let y = spmv(&mat, &x)?;
    // y = [3.0, 3.0, 9.0]

    // Convert to dense for inspection
    let dense = mat.to_dense();
    println!("dense: {:?}", dense);

    Ok(())
}
```

## API

### `SparseCsr`

| Method | Description |
|--------|-------------|
| `from_triplets(rows, cols, vals, nrows, ncols)` | Build CSR matrix from COO triplets |
| `nnz()` | Number of non-zero elements |
| `shape()` | Returns `(nrows, ncols)` |
| `to_dense()` | Convert to row-major dense `Vec<f32>` |

### Top-level functions

| Function | Description |
|----------|-------------|
| `spmv(mat, x)` | Sparse matrix-vector multiply: y = mat · x |
| `spmm(a, b)` | Sparse matrix-matrix multiply: C = A · B (result is CSR) |

### `SparseError`

Error type returned by all fallible operations. Implements `std::error::Error` via `thiserror`.

## Requirements

- CPU features work on all platforms (pure Rust, no native deps)
- GPU features require `--features gpu` and an NVIDIA GPU with CUDA driver at runtime

## License

Apache-2.0 — see [LICENSE](../../LICENSE) for details.
