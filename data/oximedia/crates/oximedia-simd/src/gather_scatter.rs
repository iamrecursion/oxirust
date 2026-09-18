#![allow(dead_code)]
//! SIMD-style gather and scatter memory operations.
//!
//! Provides efficient gather (indexed read) and scatter (indexed write)
//! operations that mimic hardware SIMD gather/scatter instructions in
//! pure scalar Rust. These are critical for:
//! - LUT (Look-Up Table) application with vectorized indices
//! - Sparse matrix operations in video filtering
//! - Indexed pixel access patterns in motion compensation
//! - Palette-based color remapping

/// Error type for gather/scatter operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatherScatterError {
    /// An index is out of bounds for the source/destination buffer.
    IndexOutOfBounds {
        /// The offending index value.
        index: usize,
        /// The buffer length.
        buffer_len: usize,
    },
    /// The indices and values slices have mismatched lengths.
    LengthMismatch {
        /// Length of the indices slice.
        indices_len: usize,
        /// Length of the values slice.
        values_len: usize,
    },
    /// An empty input was provided where non-empty is required.
    EmptyInput,
}

impl std::fmt::Display for GatherScatterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IndexOutOfBounds { index, buffer_len } => {
                write!(
                    f,
                    "index {index} out of bounds for buffer of length {buffer_len}"
                )
            }
            Self::LengthMismatch {
                indices_len,
                values_len,
            } => {
                write!(
                    f,
                    "length mismatch: {indices_len} indices vs {values_len} values"
                )
            }
            Self::EmptyInput => write!(f, "empty input"),
        }
    }
}

impl std::error::Error for GatherScatterError {}

/// Result type for gather/scatter operations.
pub type Result<T> = std::result::Result<T, GatherScatterError>;

/// Gather u8 values from `source` at the positions specified by `indices`.
///
/// # Errors
///
/// Returns `GatherScatterError::IndexOutOfBounds` if any index exceeds `source.len()`.
pub fn gather_u8(source: &[u8], indices: &[usize]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(indices.len());
    for &idx in indices {
        if idx >= source.len() {
            return Err(GatherScatterError::IndexOutOfBounds {
                index: idx,
                buffer_len: source.len(),
            });
        }
        out.push(source[idx]);
    }
    Ok(out)
}

/// Gather u16 values from `source` at the positions specified by `indices`.
///
/// # Errors
///
/// Returns `GatherScatterError::IndexOutOfBounds` if any index exceeds `source.len()`.
pub fn gather_u16(source: &[u16], indices: &[usize]) -> Result<Vec<u16>> {
    let mut out = Vec::with_capacity(indices.len());
    for &idx in indices {
        if idx >= source.len() {
            return Err(GatherScatterError::IndexOutOfBounds {
                index: idx,
                buffer_len: source.len(),
            });
        }
        out.push(source[idx]);
    }
    Ok(out)
}

/// Gather f32 values from `source` at the positions specified by `indices`.
///
/// # Errors
///
/// Returns `GatherScatterError::IndexOutOfBounds` if any index exceeds `source.len()`.
pub fn gather_f32(source: &[f32], indices: &[usize]) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(indices.len());
    for &idx in indices {
        if idx >= source.len() {
            return Err(GatherScatterError::IndexOutOfBounds {
                index: idx,
                buffer_len: source.len(),
            });
        }
        out.push(source[idx]);
    }
    Ok(out)
}

/// Scatter `values` into `dest` at the positions specified by `indices`.
///
/// Each `values[i]` is written to `dest[indices[i]]`. If multiple indices
/// refer to the same position, the last write wins.
///
/// # Errors
///
/// Returns an error if lengths mismatch or any index is out of bounds.
pub fn scatter_u8(dest: &mut [u8], indices: &[usize], values: &[u8]) -> Result<()> {
    if indices.len() != values.len() {
        return Err(GatherScatterError::LengthMismatch {
            indices_len: indices.len(),
            values_len: values.len(),
        });
    }
    for (&idx, &val) in indices.iter().zip(values.iter()) {
        if idx >= dest.len() {
            return Err(GatherScatterError::IndexOutOfBounds {
                index: idx,
                buffer_len: dest.len(),
            });
        }
        dest[idx] = val;
    }
    Ok(())
}

/// Scatter `values` into `dest` at the positions specified by `indices`.
///
/// Each `values[i]` is written to `dest[indices[i]]`. If multiple indices
/// refer to the same position, the last write wins.
///
/// # Errors
///
/// Returns an error if lengths mismatch or any index is out of bounds.
pub fn scatter_f32(dest: &mut [f32], indices: &[usize], values: &[f32]) -> Result<()> {
    if indices.len() != values.len() {
        return Err(GatherScatterError::LengthMismatch {
            indices_len: indices.len(),
            values_len: values.len(),
        });
    }
    for (&idx, &val) in indices.iter().zip(values.iter()) {
        if idx >= dest.len() {
            return Err(GatherScatterError::IndexOutOfBounds {
                index: idx,
                buffer_len: dest.len(),
            });
        }
        dest[idx] = val;
    }
    Ok(())
}

/// Perform a masked gather: only gathers elements where `mask[i]` is `true`.
///
/// Returns a vector with gathered values for `true` mask positions and
/// `default` for `false` positions.
///
/// # Errors
///
/// Returns an error if `indices.len() != mask.len()` or any active index is out of bounds.
pub fn masked_gather_u8(
    source: &[u8],
    indices: &[usize],
    mask: &[bool],
    default: u8,
) -> Result<Vec<u8>> {
    if indices.len() != mask.len() {
        return Err(GatherScatterError::LengthMismatch {
            indices_len: indices.len(),
            values_len: mask.len(),
        });
    }
    let mut out = Vec::with_capacity(indices.len());
    for (&idx, &active) in indices.iter().zip(mask.iter()) {
        if active {
            if idx >= source.len() {
                return Err(GatherScatterError::IndexOutOfBounds {
                    index: idx,
                    buffer_len: source.len(),
                });
            }
            out.push(source[idx]);
        } else {
            out.push(default);
        }
    }
    Ok(out)
}

/// Perform a masked scatter: only writes elements where `mask[i]` is `true`.
///
/// # Errors
///
/// Returns an error if lengths mismatch or any active index is out of bounds.
pub fn masked_scatter_u8(
    dest: &mut [u8],
    indices: &[usize],
    values: &[u8],
    mask: &[bool],
) -> Result<()> {
    if indices.len() != values.len() || indices.len() != mask.len() {
        return Err(GatherScatterError::LengthMismatch {
            indices_len: indices.len(),
            values_len: values.len(),
        });
    }
    for i in 0..indices.len() {
        if mask[i] {
            if indices[i] >= dest.len() {
                return Err(GatherScatterError::IndexOutOfBounds {
                    index: indices[i],
                    buffer_len: dest.len(),
                });
            }
            dest[indices[i]] = values[i];
        }
    }
    Ok(())
}

/// Gather with stride: reads every `stride`-th element starting from `offset`.
///
/// # Errors
///
/// Returns `EmptyInput` if source is empty, or `IndexOutOfBounds` if stride produces
/// an out-of-range access.
pub fn gather_strided_u8(
    source: &[u8],
    offset: usize,
    stride: usize,
    count: usize,
) -> Result<Vec<u8>> {
    if source.is_empty() {
        return Err(GatherScatterError::EmptyInput);
    }
    if stride == 0 {
        return Err(GatherScatterError::EmptyInput);
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let idx = offset + i * stride;
        if idx >= source.len() {
            return Err(GatherScatterError::IndexOutOfBounds {
                index: idx,
                buffer_len: source.len(),
            });
        }
        out.push(source[idx]);
    }
    Ok(out)
}

/// Gather-add: gathers values and accumulates them into an existing output buffer.
///
/// `out[i] += source[indices[i]]` for all i.
///
/// # Errors
///
/// Returns an error if `out.len() != indices.len()` or any index is out of bounds.
#[allow(clippy::cast_precision_loss)]
pub fn gather_add_f32(source: &[f32], indices: &[usize], out: &mut [f32]) -> Result<()> {
    if out.len() != indices.len() {
        return Err(GatherScatterError::LengthMismatch {
            indices_len: indices.len(),
            values_len: out.len(),
        });
    }
    for (o, &idx) in out.iter_mut().zip(indices.iter()) {
        if idx >= source.len() {
            return Err(GatherScatterError::IndexOutOfBounds {
                index: idx,
                buffer_len: source.len(),
            });
        }
        *o += source[idx];
    }
    Ok(())
}

/// Compute histogram by scattering increments: `hist[data[i]] += 1`.
///
/// # Errors
///
/// Returns an error if any value in `data` exceeds `hist_size - 1`.
pub fn scatter_histogram(data: &[u8], hist_size: usize) -> Result<Vec<u32>> {
    let mut hist = vec![0u32; hist_size];
    for &val in data {
        let idx = val as usize;
        if idx >= hist_size {
            return Err(GatherScatterError::IndexOutOfBounds {
                index: idx,
                buffer_len: hist_size,
            });
        }
        hist[idx] += 1;
    }
    Ok(hist)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gather_u8_basic() {
        let source = vec![10, 20, 30, 40, 50];
        let indices = vec![4, 2, 0, 3];
        let result = gather_u8(&source, &indices).expect("should succeed in test");
        assert_eq!(result, vec![50, 30, 10, 40]);
    }

    #[test]
    fn test_gather_u8_out_of_bounds() {
        let source = vec![10, 20, 30];
        let indices = vec![0, 5];
        let err = gather_u8(&source, &indices).unwrap_err();
        assert_eq!(
            err,
            GatherScatterError::IndexOutOfBounds {
                index: 5,
                buffer_len: 3
            }
        );
    }

    #[test]
    fn test_gather_u16_basic() {
        let source = vec![100u16, 200, 300, 400];
        let indices = vec![3, 1];
        let result = gather_u16(&source, &indices).expect("should succeed in test");
        assert_eq!(result, vec![400, 200]);
    }

    #[test]
    fn test_gather_f32_basic() {
        let source = vec![1.0f32, 2.0, 3.0, 4.0];
        let indices = vec![2, 0, 3];
        let result = gather_f32(&source, &indices).expect("should succeed in test");
        assert_eq!(result, vec![3.0, 1.0, 4.0]);
    }

    #[test]
    fn test_scatter_u8_basic() {
        let mut dest = vec![0u8; 5];
        let indices = vec![1, 3, 4];
        let values = vec![10, 30, 40];
        scatter_u8(&mut dest, &indices, &values).expect("should succeed in test");
        assert_eq!(dest, vec![0, 10, 0, 30, 40]);
    }

    #[test]
    fn test_scatter_u8_length_mismatch() {
        let mut dest = vec![0u8; 5];
        let indices = vec![1, 3];
        let values = vec![10];
        let err = scatter_u8(&mut dest, &indices, &values).unwrap_err();
        assert_eq!(
            err,
            GatherScatterError::LengthMismatch {
                indices_len: 2,
                values_len: 1
            }
        );
    }

    #[test]
    fn test_scatter_f32_basic() {
        let mut dest = vec![0.0f32; 4];
        let indices = vec![0, 2];
        let values = vec![1.5, 3.5];
        scatter_f32(&mut dest, &indices, &values).expect("should succeed in test");
        assert_eq!(dest, vec![1.5, 0.0, 3.5, 0.0]);
    }

    #[test]
    fn test_masked_gather_u8() {
        let source = vec![10, 20, 30, 40, 50];
        let indices = vec![0, 1, 2, 3];
        let mask = vec![true, false, true, false];
        let result =
            masked_gather_u8(&source, &indices, &mask, 255).expect("should succeed in test");
        assert_eq!(result, vec![10, 255, 30, 255]);
    }

    #[test]
    fn test_masked_scatter_u8() {
        let mut dest = vec![0u8; 5];
        let indices = vec![0, 1, 2];
        let values = vec![10, 20, 30];
        let mask = vec![true, false, true];
        masked_scatter_u8(&mut dest, &indices, &values, &mask).expect("should succeed in test");
        assert_eq!(dest, vec![10, 0, 30, 0, 0]);
    }

    #[test]
    fn test_gather_strided_u8() {
        let source: Vec<u8> = (0..20).collect();
        let result = gather_strided_u8(&source, 1, 3, 5).expect("should succeed in test");
        assert_eq!(result, vec![1, 4, 7, 10, 13]);
    }

    #[test]
    fn test_gather_strided_u8_zero_stride() {
        let source = vec![1u8, 2, 3];
        let err = gather_strided_u8(&source, 0, 0, 1).unwrap_err();
        assert_eq!(err, GatherScatterError::EmptyInput);
    }

    #[test]
    fn test_gather_add_f32() {
        let source = vec![1.0f32, 2.0, 3.0, 4.0];
        let indices = vec![0, 2, 1];
        let mut out = vec![10.0f32, 20.0, 30.0];
        gather_add_f32(&source, &indices, &mut out).expect("should succeed in test");
        assert_eq!(out, vec![11.0, 23.0, 32.0]);
    }

    #[test]
    fn test_scatter_histogram() {
        let data = vec![0u8, 1, 1, 2, 2, 2, 3];
        let hist = scatter_histogram(&data, 256).expect("should succeed in test");
        assert_eq!(hist[0], 1);
        assert_eq!(hist[1], 2);
        assert_eq!(hist[2], 3);
        assert_eq!(hist[3], 1);
        assert_eq!(hist[4], 0);
    }

    #[test]
    fn test_gather_u8_empty_indices() {
        let source = vec![10, 20, 30];
        let result = gather_u8(&source, &[]).expect("should succeed in test");
        assert!(result.is_empty());
    }

    #[test]
    fn test_scatter_histogram_out_of_bounds() {
        let data = vec![5u8];
        let err = scatter_histogram(&data, 4).unwrap_err();
        assert_eq!(
            err,
            GatherScatterError::IndexOutOfBounds {
                index: 5,
                buffer_len: 4
            }
        );
    }
}
