//! Per-pixel arithmetic and bitwise operations mirroring the `cv2` API.
//!
//! All operations on `CV_8U` types use saturating arithmetic, clamping results
//! to `[0, 255]`.  Size and type mismatches return `Cv2Error::SizeMismatch`.

use crate::error::{Cv2Error, Cv2Result};
use crate::mat::{Mat, MatType, Point, Scalar};

// ── Size / type validation ────────────────────────────────────────────────────

/// Assert that two `Mat`s have identical dimensions and element type.
fn check_same_size(src1: &Mat, src2: &Mat) -> Cv2Result<()> {
    if src1.rows != src2.rows || src1.cols != src2.cols || src1.mat_type != src2.mat_type {
        return Err(Cv2Error::SizeMismatch {
            expected: (src1.rows, src1.cols),
            actual: (src2.rows, src2.cols),
        });
    }
    Ok(())
}

// ── Arithmetic operations ─────────────────────────────────────────────────────

/// Per-pixel addition with saturation: `dst = saturate(src1 + src2)`.
pub fn add(src1: &Mat, src2: &Mat) -> Cv2Result<Mat> {
    check_same_size(src1, src2)?;
    let data: Vec<u8> = src1
        .data
        .iter()
        .zip(src2.data.iter())
        .map(|(&a, &b)| a.saturating_add(b))
        .collect();
    Ok(Mat {
        data,
        rows: src1.rows,
        cols: src1.cols,
        step: src1.step,
        mat_type: src1.mat_type,
    })
}

/// Per-pixel subtraction with saturation: `dst = saturate(src1 - src2)`.
pub fn subtract(src1: &Mat, src2: &Mat) -> Cv2Result<Mat> {
    check_same_size(src1, src2)?;
    let data: Vec<u8> = src1
        .data
        .iter()
        .zip(src2.data.iter())
        .map(|(&a, &b)| a.saturating_sub(b))
        .collect();
    Ok(Mat {
        data,
        rows: src1.rows,
        cols: src1.cols,
        step: src1.step,
        mat_type: src1.mat_type,
    })
}

/// Per-pixel multiplication with saturation: `dst = saturate(src1 * src2 / 255)`.
///
/// The scale factor `/255` matches cv2's `cv2.multiply` default scale=1 semantics
/// for `CV_8U` images (i.e., product is divided by 255 to keep range in \[0, 255\]).
pub fn multiply(src1: &Mat, src2: &Mat) -> Cv2Result<Mat> {
    check_same_size(src1, src2)?;
    let data: Vec<u8> = src1
        .data
        .iter()
        .zip(src2.data.iter())
        .map(|(&a, &b)| ((a as u32 * b as u32 + 127) / 255).min(255) as u8)
        .collect();
    Ok(Mat {
        data,
        rows: src1.rows,
        cols: src1.cols,
        step: src1.step,
        mat_type: src1.mat_type,
    })
}

/// Per-pixel division: `dst = saturate(src1 / src2)`.
///
/// Division by zero produces 0 in the output, matching cv2 behaviour.
pub fn divide(src1: &Mat, src2: &Mat) -> Cv2Result<Mat> {
    check_same_size(src1, src2)?;
    let data: Vec<u8> = src1
        .data
        .iter()
        .zip(src2.data.iter())
        .map(|(&a, &b)| {
            if b == 0 {
                0
            } else {
                (a as u32 * 255 / b as u32).min(255) as u8
            }
        })
        .collect();
    Ok(Mat {
        data,
        rows: src1.rows,
        cols: src1.cols,
        step: src1.step,
        mat_type: src1.mat_type,
    })
}

/// Weighted blend: `dst = saturate(alpha * src1 + beta * src2 + gamma)`.
///
/// Mirrors `cv2.addWeighted(src1, alpha, src2, beta, gamma)`.
pub fn add_weighted(src1: &Mat, alpha: f64, src2: &Mat, beta: f64, gamma: f64) -> Cv2Result<Mat> {
    check_same_size(src1, src2)?;
    let data: Vec<u8> = src1
        .data
        .iter()
        .zip(src2.data.iter())
        .map(|(&a, &b)| (alpha * a as f64 + beta * b as f64 + gamma).clamp(0.0, 255.0) as u8)
        .collect();
    Ok(Mat {
        data,
        rows: src1.rows,
        cols: src1.cols,
        step: src1.step,
        mat_type: src1.mat_type,
    })
}

/// Per-pixel absolute difference: `dst = |src1 - src2|`.
///
/// Mirrors `cv2.absdiff(src1, src2)`.
pub fn abs_diff(src1: &Mat, src2: &Mat) -> Cv2Result<Mat> {
    check_same_size(src1, src2)?;
    let data: Vec<u8> = src1
        .data
        .iter()
        .zip(src2.data.iter())
        .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs() as u8)
        .collect();
    Ok(Mat {
        data,
        rows: src1.rows,
        cols: src1.cols,
        step: src1.step,
        mat_type: src1.mat_type,
    })
}

// ── Bitwise operations ────────────────────────────────────────────────────────

/// Per-pixel bitwise AND: `dst = src1 & src2`.
pub fn bitwise_and(src1: &Mat, src2: &Mat) -> Cv2Result<Mat> {
    check_same_size(src1, src2)?;
    let data: Vec<u8> = src1
        .data
        .iter()
        .zip(src2.data.iter())
        .map(|(&a, &b)| a & b)
        .collect();
    Ok(Mat {
        data,
        rows: src1.rows,
        cols: src1.cols,
        step: src1.step,
        mat_type: src1.mat_type,
    })
}

/// Per-pixel bitwise OR: `dst = src1 | src2`.
pub fn bitwise_or(src1: &Mat, src2: &Mat) -> Cv2Result<Mat> {
    check_same_size(src1, src2)?;
    let data: Vec<u8> = src1
        .data
        .iter()
        .zip(src2.data.iter())
        .map(|(&a, &b)| a | b)
        .collect();
    Ok(Mat {
        data,
        rows: src1.rows,
        cols: src1.cols,
        step: src1.step,
        mat_type: src1.mat_type,
    })
}

/// Per-pixel bitwise XOR: `dst = src1 ^ src2`.
pub fn bitwise_xor(src1: &Mat, src2: &Mat) -> Cv2Result<Mat> {
    check_same_size(src1, src2)?;
    let data: Vec<u8> = src1
        .data
        .iter()
        .zip(src2.data.iter())
        .map(|(&a, &b)| a ^ b)
        .collect();
    Ok(Mat {
        data,
        rows: src1.rows,
        cols: src1.cols,
        step: src1.step,
        mat_type: src1.mat_type,
    })
}

/// Per-pixel bitwise NOT: `dst = ~src`.
pub fn bitwise_not(src: &Mat) -> Cv2Result<Mat> {
    let data: Vec<u8> = src.data.iter().map(|&v| !v).collect();
    Ok(Mat {
        data,
        rows: src.rows,
        cols: src.cols,
        step: src.step,
        mat_type: src.mat_type,
    })
}

// ── Range and comparison ──────────────────────────────────────────────────────

/// Per-pixel range mask: `dst(i) = 255` if all channels of `src` are within
/// `[lower, upper]`, otherwise `0`.
///
/// Output is always `CV_8UC1` (single-channel mask).
/// Mirrors `cv2.inRange(src, lower, upper)`.
pub fn in_range(src: &Mat, lower: Scalar, upper: Scalar) -> Cv2Result<Mat> {
    let ch = src.channels();
    let lo = [lower.0 as u8, lower.1 as u8, lower.2 as u8, lower.3 as u8];
    let hi = [upper.0 as u8, upper.1 as u8, upper.2 as u8, upper.3 as u8];

    let n = src.rows * src.cols;
    let mut data = vec![0u8; n];
    for i in 0..n {
        let in_rng = (0..ch).all(|c| {
            let v = src.data[i * ch + c];
            v >= lo[c] && v <= hi[c]
        });
        if in_rng {
            data[i] = 255;
        }
    }
    Ok(Mat::from_gray_bytes(data, src.rows, src.cols))
}

/// Clamp all pixel values to `[0, value]`: `dst = min(src, value)`.
pub fn mat_min(src: &Mat, value: f64) -> Cv2Result<Mat> {
    let v = value.clamp(0.0, 255.0) as u8;
    let data: Vec<u8> = src.data.iter().map(|&p| p.min(v)).collect();
    Ok(Mat {
        data,
        rows: src.rows,
        cols: src.cols,
        step: src.step,
        mat_type: src.mat_type,
    })
}

/// Clamp all pixel values to `[value, 255]`: `dst = max(src, value)`.
pub fn mat_max(src: &Mat, value: f64) -> Cv2Result<Mat> {
    let v = value.clamp(0.0, 255.0) as u8;
    let data: Vec<u8> = src.data.iter().map(|&p| p.max(v)).collect();
    Ok(Mat {
        data,
        rows: src.rows,
        cols: src.cols,
        step: src.step,
        mat_type: src.mat_type,
    })
}

/// Compute per-channel mean over all pixels.
///
/// Returns `Scalar(ch0_mean, ch1_mean, ch2_mean, ch3_mean)`.
/// Empty or zero-pixel `Mat` returns `Scalar::default()`.
pub fn mean(src: &Mat) -> Scalar {
    let n = src.rows * src.cols;
    if n == 0 {
        return Scalar::default();
    }
    let ch = src.channels();
    let mut sums = [0f64; 4];
    for i in 0..n {
        for c in 0..ch.min(4) {
            sums[c] += src.data[i * ch + c] as f64;
        }
    }
    let nf = n as f64;
    Scalar(sums[0] / nf, sums[1] / nf, sums[2] / nf, sums[3] / nf)
}

/// Per-element comparison, returning a `CV_8UC1` mask.
///
/// Output pixel is `255` where the predicate holds, `0` otherwise.
/// `cmp_op` values: `CMP_EQ=0`, `CMP_GT=1`, `CMP_GE=2`, `CMP_LT=3`, `CMP_LE=4`, `CMP_NE=5`.
///
/// Only the first channel of each pixel is used for comparison (matching
/// cv2 behaviour for single-channel inputs; for multi-channel use, operate
/// on individual channels first).
pub fn compare(src1: &Mat, src2: &Mat, cmp_op: i32) -> Cv2Result<Mat> {
    check_same_size(src1, src2)?;
    let ch = src1.channels();
    let n = src1.rows * src1.cols;
    let mut data = vec![0u8; n];
    for i in 0..n {
        // Compare first channel value
        let a = src1.data[i * ch] as i32;
        let b = src2.data[i * ch] as i32;
        let result = match cmp_op {
            0 => a == b, // CMP_EQ
            1 => a > b,  // CMP_GT
            2 => a >= b, // CMP_GE
            3 => a < b,  // CMP_LT
            4 => a <= b, // CMP_LE
            5 => a != b, // CMP_NE
            _ => false,
        };
        if result {
            data[i] = 255;
        }
    }
    Ok(Mat {
        data,
        rows: src1.rows,
        cols: src1.cols,
        step: src1.cols,
        mat_type: MatType::CV_8UC1,
    })
}

// ── Reduction / statistics helpers ───────────────────────────────────────────

/// Read one component at `byte_offset` as `f64`, respecting `mat_type` depth.
///
/// For 8-bit types reads one byte; for 32-bit float types reads four bytes.
fn component_as_f64(data: &[u8], byte_offset: usize, mat_type: MatType) -> f64 {
    match mat_type {
        MatType::CV_8UC1 | MatType::CV_8UC3 | MatType::CV_8UC4 => data[byte_offset] as f64,
        MatType::CV_32FC1 | MatType::CV_32FC2 | MatType::CV_32FC3 => {
            let bytes = [
                data[byte_offset],
                data[byte_offset + 1],
                data[byte_offset + 2],
                data[byte_offset + 3],
            ];
            f32::from_ne_bytes(bytes) as f64
        }
        MatType::CV_64FC1 => {
            let bytes: [u8; 8] = data[byte_offset..byte_offset + 8]
                .try_into()
                .unwrap_or([0u8; 8]);
            f64::from_ne_bytes(bytes)
        }
    }
}

/// Validate that `mask` (if `Some`) is CV_8UC1 and has identical dimensions to `src`.
fn check_mask(src: &Mat, mask: Option<&Mat>) -> Cv2Result<()> {
    if let Some(m) = mask {
        if m.mat_type != MatType::CV_8UC1 {
            return Err(Cv2Error::UnsupportedDtype {
                mat_type: m.mat_type,
            });
        }
        if m.rows != src.rows || m.cols != src.cols {
            return Err(Cv2Error::SizeMismatch {
                expected: (src.rows, src.cols),
                actual: (m.rows, m.cols),
            });
        }
    }
    Ok(())
}

/// Count pixels where any component is non-zero.
///
/// For single-channel mats a pixel is a single element; for multi-channel mats
/// a pixel is counted if **any** channel is non-zero (matching the natural
/// extension of `cv2.countNonZero`).
///
/// Returns `Cv2Error::UnsupportedDtype` for unsupported element types.
pub fn count_non_zero(src: &Mat) -> Cv2Result<u64> {
    let n = src.rows * src.cols;
    let ch = src.channels();
    let depth = src.mat_type.depth_bytes();
    let mut count = 0u64;
    match src.mat_type {
        MatType::CV_8UC1 | MatType::CV_8UC3 | MatType::CV_8UC4 => {
            for i in 0..n {
                let base = i * ch;
                if (0..ch).any(|c| src.data[base + c] != 0) {
                    count += 1;
                }
            }
        }
        MatType::CV_32FC1 | MatType::CV_32FC2 | MatType::CV_32FC3 => {
            for i in 0..n {
                let base = i * ch * depth;
                let nonzero = (0..ch).any(|c| {
                    let off = base + c * depth;
                    let v = component_as_f64(&src.data, off, src.mat_type);
                    v != 0.0
                });
                if nonzero {
                    count += 1;
                }
            }
        }
        _ => {
            return Err(Cv2Error::UnsupportedDtype {
                mat_type: src.mat_type,
            });
        }
    }
    Ok(count)
}

/// Per-channel sum of all pixel values.
///
/// Returns `[ch0_sum, ch1_sum, ch2_sum, ch3_sum]`; channels beyond the
/// element count are `0.0`.  For `CV_8UC3` (BGR) index 0 is blue, 1 green,
/// 2 red.
pub fn sum_elems(src: &Mat) -> Cv2Result<[f64; 4]> {
    let n = src.rows * src.cols;
    let ch = src.channels();
    let depth = src.mat_type.depth_bytes();
    let mut sums = [0.0f64; 4];
    match src.mat_type {
        MatType::CV_8UC1 | MatType::CV_8UC3 | MatType::CV_8UC4 => {
            for i in 0..n {
                let base = i * ch;
                for c in 0..ch.min(4) {
                    sums[c] += src.data[base + c] as f64;
                }
            }
        }
        MatType::CV_32FC1 => {
            for i in 0..n {
                let off = i * depth;
                sums[0] += component_as_f64(&src.data, off, src.mat_type);
            }
        }
        _ => {
            return Err(Cv2Error::UnsupportedDtype {
                mat_type: src.mat_type,
            });
        }
    }
    Ok(sums)
}

/// Per-channel mean with optional mask.
///
/// If `mask` is `Some`, only pixels where `mask[i] > 0` are included.  If all
/// pixels are masked out the result is `[0.0; 4]`.
pub fn mean_val(src: &Mat, mask: Option<&Mat>) -> Cv2Result<[f64; 4]> {
    check_mask(src, mask)?;
    let n = src.rows * src.cols;
    let ch = src.channels();
    let depth = src.mat_type.depth_bytes();
    let mut sums = [0.0f64; 4];
    let mut count = 0u64;

    for i in 0..n {
        if let Some(m) = mask {
            if m.data[i] == 0 {
                continue;
            }
        }
        count += 1;
        let base = i * ch * depth;
        for c in 0..ch.min(4) {
            let off = base + c * depth;
            sums[c] += component_as_f64(&src.data, off, src.mat_type);
        }
    }

    if count == 0 {
        return Ok([0.0; 4]);
    }
    let nf = count as f64;
    Ok([sums[0] / nf, sums[1] / nf, sums[2] / nf, sums[3] / nf])
}

/// Per-channel mean and population standard deviation (Welford's online algorithm).
///
/// Returns `(means, stddevs)`.  Uses **population** standard deviation
/// (`sqrt(M2 / count)`) matching `cv2.meanStdDev`.
///
/// If `mask` is `Some`, only pixels where `mask[i] > 0` are included.
pub fn mean_std_dev(src: &Mat, mask: Option<&Mat>) -> Cv2Result<([f64; 4], [f64; 4])> {
    check_mask(src, mask)?;
    let n = src.rows * src.cols;
    let ch = src.channels();
    let depth = src.mat_type.depth_bytes();

    match src.mat_type {
        MatType::CV_8UC1 | MatType::CV_8UC3 | MatType::CV_32FC1 => {}
        _ => {
            return Err(Cv2Error::UnsupportedDtype {
                mat_type: src.mat_type,
            });
        }
    }

    let mut means = [0.0f64; 4];
    let mut m2 = [0.0f64; 4];
    let mut counts = [0u64; 4];

    for i in 0..n {
        if let Some(m) = mask {
            if m.data[i] == 0 {
                continue;
            }
        }
        let base = i * ch * depth;
        for c in 0..ch.min(4) {
            let off = base + c * depth;
            let x = component_as_f64(&src.data, off, src.mat_type);
            counts[c] += 1;
            let delta = x - means[c];
            means[c] += delta / counts[c] as f64;
            let delta2 = x - means[c];
            m2[c] += delta * delta2;
        }
    }

    let mut stddevs = [0.0f64; 4];
    for c in 0..4 {
        if counts[c] > 0 {
            stddevs[c] = (m2[c] / counts[c] as f64).sqrt();
        }
    }
    Ok((means, stddevs))
}

/// Compute a scalar norm of `src`.
///
/// `norm_type` values:
/// - `NORM_INF` (1): maximum absolute value
/// - `NORM_L1` (2): sum of absolute values
/// - `NORM_L2` (4): Euclidean norm `sqrt(sum of squares)`
/// - `NORM_L2SQR` (5): sum of squares
pub fn norm(src: &Mat, norm_type: i32) -> Cv2Result<f64> {
    let n_bytes = src.data.len();
    let depth = src.mat_type.depth_bytes();

    // Validate norm_type first
    if !matches!(norm_type, 1 | 2 | 4 | 5) {
        return Err(Cv2Error::UnsupportedFlag {
            name: "norm: unsupported norm_type",
            value: norm_type,
        });
    }

    let elem_count = n_bytes / depth;
    let mut acc = match norm_type {
        1 => f64::NEG_INFINITY, // NORM_INF — will take max
        _ => 0.0f64,
    };

    for i in 0..elem_count {
        let v = component_as_f64(&src.data, i * depth, src.mat_type);
        match norm_type {
            1 => {
                // NORM_INF
                if v.abs() > acc {
                    acc = v.abs();
                }
            }
            2 => acc += v.abs(),   // NORM_L1
            4 | 5 => acc += v * v, // NORM_L2 / NORM_L2SQR
            _ => unreachable!(),
        }
    }

    // For NORM_INF on an empty mat
    if n_bytes == 0 && norm_type == 1 {
        return Ok(0.0);
    }

    Ok(match norm_type {
        4 => acc.sqrt(), // NORM_L2
        _ => acc,
    })
}

/// Norm of the element-wise difference `‖src1 − src2‖`.
///
/// `src1` and `src2` must have the same size and element type.
/// For `CV_8U` types, the difference uses `i64` arithmetic to avoid
/// saturation; for `CV_32FC1` uses `f64` difference.
pub fn norm_diff(src1: &Mat, src2: &Mat, norm_type: i32) -> Cv2Result<f64> {
    check_same_size(src1, src2)?;
    if !matches!(norm_type, 1 | 2 | 4 | 5) {
        return Err(Cv2Error::UnsupportedFlag {
            name: "norm: unsupported norm_type",
            value: norm_type,
        });
    }
    let n_bytes = src1.data.len();
    let depth = src1.mat_type.depth_bytes();
    let elem_count = n_bytes / depth;

    let mut acc = match norm_type {
        1 => f64::NEG_INFINITY,
        _ => 0.0f64,
    };

    for i in 0..elem_count {
        let a = component_as_f64(&src1.data, i * depth, src1.mat_type);
        let b = component_as_f64(&src2.data, i * depth, src2.mat_type);
        let diff = (a - b).abs();
        match norm_type {
            1 => {
                if diff > acc {
                    acc = diff;
                }
            }
            2 => acc += diff,
            4 | 5 => acc += diff * diff,
            _ => unreachable!(),
        }
    }

    if n_bytes == 0 && norm_type == 1 {
        return Ok(0.0);
    }

    Ok(match norm_type {
        4 => acc.sqrt(),
        _ => acc,
    })
}

/// Find the minimum and maximum values and their locations in `src`.
///
/// Only supports single-channel mats (`CV_8UC1`, `CV_32FC1`).  For
/// multi-channel types returns `Cv2Error::UnsupportedDtype`.
///
/// Returns `(min_val, max_val, min_loc, max_loc)`.  If the mask excludes
/// all pixels the result is `(0.0, 0.0, (0,0), (0,0))`.
pub fn min_max_loc(src: &Mat, mask: Option<&Mat>) -> Cv2Result<(f64, f64, Point, Point)> {
    check_mask(src, mask)?;
    match src.mat_type {
        MatType::CV_8UC1 | MatType::CV_32FC1 => {}
        _ => {
            return Err(Cv2Error::UnsupportedDtype {
                mat_type: src.mat_type,
            });
        }
    }
    let n = src.rows * src.cols;
    let depth = src.mat_type.depth_bytes();
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    let mut min_loc = Point { x: 0, y: 0 };
    let mut max_loc = Point { x: 0, y: 0 };
    let mut any = false;

    for i in 0..n {
        if let Some(m) = mask {
            if m.data[i] == 0 {
                continue;
            }
        }
        let v = component_as_f64(&src.data, i * depth, src.mat_type);
        let row = (i / src.cols) as i32;
        let col = (i % src.cols) as i32;
        if !any || v < min_v {
            min_v = v;
            min_loc = Point { x: col, y: row };
        }
        if !any || v > max_v {
            max_v = v;
            max_loc = Point { x: col, y: row };
        }
        any = true;
    }

    if !any {
        return Ok((0.0, 0.0, Point { x: 0, y: 0 }, Point { x: 0, y: 0 }));
    }
    Ok((min_v, max_v, min_loc, max_loc))
}

/// Split a multi-channel `Mat` into a `Vec` of single-channel `Mat`s.
///
/// Channel order is preserved (for `CV_8UC3` the output is
/// `[blue_mat, green_mat, red_mat]`).
pub fn split(src: &Mat) -> Cv2Result<Vec<Mat>> {
    let ch = src.channels();
    let n = src.rows * src.cols;
    let depth = src.mat_type.depth_bytes();

    match src.mat_type {
        MatType::CV_8UC1 | MatType::CV_32FC1 => {
            // Already single-channel — return a deep copy.
            let copy = Mat {
                data: src.data.clone(),
                rows: src.rows,
                cols: src.cols,
                step: src.cols * depth,
                mat_type: src.mat_type,
            };
            return Ok(vec![copy]);
        }
        MatType::CV_8UC3 | MatType::CV_8UC4 => {}
        _ => {
            return Err(Cv2Error::UnsupportedDtype {
                mat_type: src.mat_type,
            });
        }
    }

    let out_type = MatType::CV_8UC1;
    let mut planes: Vec<Vec<u8>> = (0..ch).map(|_| vec![0u8; n]).collect();

    for i in 0..n {
        let base = i * ch;
        for c in 0..ch {
            planes[c][i] = src.data[base + c];
        }
    }

    Ok(planes
        .into_iter()
        .map(|plane| Mat {
            step: src.cols,
            data: plane,
            rows: src.rows,
            cols: src.cols,
            mat_type: out_type,
        })
        .collect())
}

/// Merge multiple single-channel `Mat`s into a single multi-channel `Mat`.
///
/// All input mats must have the same dimensions and element type.
/// Accepts 1 to 4 `CV_8UC1` sources, producing `CV_8UC{1..4}` output.
pub fn merge(srcs: &[&Mat]) -> Cv2Result<Mat> {
    if srcs.is_empty() {
        return Err(Cv2Error::UnsupportedFlag {
            name: "merge: at least 1 source required",
            value: 0,
        });
    }
    if srcs.len() > 4 {
        return Err(Cv2Error::UnsupportedFlag {
            name: "merge: max 4 channels",
            value: srcs.len() as i32,
        });
    }
    let first = srcs[0];
    for s in srcs.iter().skip(1) {
        if s.rows != first.rows || s.cols != first.cols || s.mat_type != first.mat_type {
            return Err(Cv2Error::SizeMismatch {
                expected: (first.rows, first.cols),
                actual: (s.rows, s.cols),
            });
        }
    }
    // Only CV_8UC1 sources are supported.
    if first.mat_type != MatType::CV_8UC1 {
        return Err(Cv2Error::UnsupportedDtype {
            mat_type: first.mat_type,
        });
    }
    let ch = srcs.len();
    let n = first.rows * first.cols;
    let out_type = match ch {
        1 => MatType::CV_8UC1,
        3 => MatType::CV_8UC3,
        4 => MatType::CV_8UC4,
        _ => {
            return Err(Cv2Error::UnsupportedFlag {
                name: "merge: unsupported channel count",
                value: ch as i32,
            });
        }
    };
    let mut data = vec![0u8; n * ch];
    for i in 0..n {
        for c in 0..ch {
            data[i * ch + c] = srcs[c].data[i];
        }
    }
    Ok(Mat {
        data,
        rows: first.rows,
        cols: first.cols,
        step: first.cols * ch,
        mat_type: out_type,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::{Mat, Scalar};

    fn gray(data: Vec<u8>, rows: usize, cols: usize) -> Mat {
        Mat::from_gray_bytes(data, rows, cols)
    }

    #[test]
    fn test_add_saturation() {
        let a = gray(vec![200u8, 100u8], 1, 2);
        let b = gray(vec![100u8, 200u8], 1, 2);
        let c = add(&a, &b).unwrap();
        assert_eq!(c.at_8u1(0, 0), 255, "200+100 saturates to 255");
        assert_eq!(c.at_8u1(0, 1), 255, "100+200 saturates to 255");
    }

    #[test]
    fn test_subtract_saturation() {
        let a = gray(vec![50u8, 200u8], 1, 2);
        let b = gray(vec![100u8, 50u8], 1, 2);
        let c = subtract(&a, &b).unwrap();
        assert_eq!(c.at_8u1(0, 0), 0, "50-100 saturates to 0");
        assert_eq!(c.at_8u1(0, 1), 150, "200-50 = 150");
    }

    #[test]
    fn test_add_weighted_blend() {
        let a = gray(vec![100u8; 4], 2, 2);
        let b = gray(vec![200u8; 4], 2, 2);
        let c = add_weighted(&a, 0.5, &b, 0.5, 0.0).unwrap();
        assert_eq!(c.at_8u1(0, 0), 150, "0.5*100 + 0.5*200 = 150");
    }

    #[test]
    fn test_bitwise_not() {
        let src = gray(vec![0u8, 255u8, 128u8], 1, 3);
        let n = bitwise_not(&src).unwrap();
        assert_eq!(n.at_8u1(0, 0), 255);
        assert_eq!(n.at_8u1(0, 1), 0);
        assert_eq!(n.at_8u1(0, 2), 127);
    }

    #[test]
    fn test_bitwise_and() {
        let a = gray(vec![0xFF, 0x0F], 1, 2);
        let b = gray(vec![0x0F, 0xFF], 1, 2);
        let c = bitwise_and(&a, &b).unwrap();
        assert_eq!(c.at_8u1(0, 0), 0x0F);
        assert_eq!(c.at_8u1(0, 1), 0x0F);
    }

    #[test]
    fn test_abs_diff_symmetric() {
        let a = gray(vec![200u8, 100u8], 1, 2);
        let b = gray(vec![100u8, 200u8], 1, 2);
        let c = abs_diff(&a, &b).unwrap();
        assert_eq!(c.at_8u1(0, 0), 100);
        assert_eq!(c.at_8u1(0, 1), 100);
    }

    #[test]
    fn test_in_range_basic() {
        let src = gray(vec![50u8, 100u8, 200u8], 1, 3);
        let mask = in_range(
            &src,
            Scalar(60.0, 0.0, 0.0, 0.0),
            Scalar(150.0, 255.0, 255.0, 255.0),
        )
        .unwrap();
        assert_eq!(mask.at_8u1(0, 0), 0, "50 < 60 → out of range");
        assert_eq!(mask.at_8u1(0, 1), 255, "100 in [60,150]");
        assert_eq!(mask.at_8u1(0, 2), 0, "200 > 150 → out of range");
    }

    #[test]
    fn test_mean_gray() {
        let src = gray(vec![0u8, 100u8, 200u8], 1, 3);
        let m = mean(&src);
        // mean = (0+100+200)/3 = 100.0
        assert!(
            (m.0 - 100.0).abs() < 0.01,
            "mean should be 100.0, got {}",
            m.0
        );
    }

    #[test]
    fn test_compare_eq() {
        let a = gray(vec![10u8, 20u8, 30u8], 1, 3);
        let b = gray(vec![10u8, 25u8, 30u8], 1, 3);
        let c = compare(&a, &b, 0 /* CMP_EQ */).unwrap();
        assert_eq!(c.at_8u1(0, 0), 255, "10==10 → 255");
        assert_eq!(c.at_8u1(0, 1), 0, "20!=25 → 0");
        assert_eq!(c.at_8u1(0, 2), 255, "30==30 → 255");
    }

    #[test]
    fn test_size_mismatch_returns_error() {
        let a = gray(vec![0u8; 4], 2, 2);
        let b = gray(vec![0u8; 6], 2, 3);
        assert!(add(&a, &b).is_err());
    }

    #[test]
    fn test_mat_min_max() {
        let src = gray(vec![10u8, 50u8, 200u8], 1, 3);
        let clamped = mat_min(&src, 100.0).unwrap();
        assert_eq!(clamped.at_8u1(0, 0), 10);
        assert_eq!(clamped.at_8u1(0, 1), 50);
        assert_eq!(clamped.at_8u1(0, 2), 100);

        let floored = mat_max(&src, 50.0).unwrap();
        assert_eq!(floored.at_8u1(0, 0), 50);
        assert_eq!(floored.at_8u1(0, 1), 50);
        assert_eq!(floored.at_8u1(0, 2), 200);
    }
}
