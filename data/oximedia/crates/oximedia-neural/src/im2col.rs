//! im2col and col2im transforms for efficient convolution via GEMM.
//!
//! `im2col` reshapes a `[C, H, W]` input tensor into a 2-D column matrix
//! `[kernel_h * kernel_w * in_channels, out_h * out_w]` so that a standard
//! matrix-matrix multiply against the weight matrix produces the full
//! convolution output in a single BLAS-style call.
//!
//! `col2im` is the adjoint (transpose) operation used for transposed
//! convolutions and gradient computation: it scatters (accumulates) columns
//! back into a `[C, H, W]` spatial map.

// ──────────────────────────────────────────────────────────────────────────────
// im2col
// ──────────────────────────────────────────────────────────────────────────────

/// Transform a `[C, H, W]` input tensor into a column matrix suitable for
/// convolution via a single matrix multiplication.
///
/// # Layout
/// Input `input` is a flat, **row-major** buffer of length
/// `in_channels * height * width`.
///
/// Output column matrix has shape
/// `[col_rows, col_cols]` where
/// - `col_rows = kernel_h * kernel_w * in_channels`
/// - `col_cols = out_h * out_w`
/// - `out_h = (height + 2*pad_h - kernel_h) / stride_h + 1`
/// - `out_w = (width  + 2*pad_w - kernel_w) / stride_w + 1`
///
/// The returned tuple is `(col_data, col_rows, col_cols)`.
///
/// Out-of-bounds regions introduced by padding are filled with `0.0`.
///
/// # Panics
/// Does not panic; all arithmetic uses checked expressions.  Returns an empty
/// vector `(vec![], 0, 0)` when the output spatial size would be zero (e.g.
/// kernel larger than padded input).
pub fn im2col(
    input: &[f32],
    in_channels: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> (Vec<f32>, usize, usize) {
    debug_assert_eq!(input.len(), in_channels * height * width);
    debug_assert!(stride_h > 0 && stride_w > 0);

    let padded_h = height + 2 * pad_h;
    let padded_w = width + 2 * pad_w;

    if padded_h < kernel_h || padded_w < kernel_w {
        return (vec![], 0, 0);
    }

    let out_h = (padded_h - kernel_h) / stride_h + 1;
    let out_w = (padded_w - kernel_w) / stride_w + 1;

    let col_rows = kernel_h * kernel_w * in_channels;
    let col_cols = out_h * out_w;

    // Output buffer indexed as [col_row, col_col] in row-major order.
    let mut col = vec![0.0_f32; col_rows * col_cols];

    // col_row encodes (channel, kernel_row, kernel_col):
    //   col_row = ic * (kernel_h * kernel_w) + kh_i * kernel_w + kw_i
    // col_col encodes (output_row, output_col):
    //   col_col = oh * out_w + ow
    for ic in 0..in_channels {
        let in_channel_offset = ic * height * width;
        for kh_i in 0..kernel_h {
            for kw_i in 0..kernel_w {
                let col_row = ic * (kernel_h * kernel_w) + kh_i * kernel_w + kw_i;
                let col_row_offset = col_row * col_cols;

                for oh in 0..out_h {
                    // Spatial row in the (zero-padded) input.
                    let ih = oh * stride_h + kh_i;
                    // Check if this row is in the original (non-padded) input.
                    let in_h_valid = ih >= pad_h && ih < pad_h + height;
                    let orig_ih = ih.wrapping_sub(pad_h); // safe to use only when in_h_valid

                    for ow in 0..out_w {
                        let iw = ow * stride_w + kw_i;
                        let in_w_valid = iw >= pad_w && iw < pad_w + width;

                        let val = if in_h_valid && in_w_valid {
                            let orig_iw = iw - pad_w;
                            input[in_channel_offset + orig_ih * width + orig_iw]
                        } else {
                            0.0_f32
                        };

                        col[col_row_offset + oh * out_w + ow] = val;
                    }
                }
            }
        }
    }

    (col, col_rows, col_cols)
}

// ──────────────────────────────────────────────────────────────────────────────
// col2im
// ──────────────────────────────────────────────────────────────────────────────

/// Inverse (adjoint) of [`im2col`]: scatter-accumulate a column matrix back
/// into a `[C, H, W]` spatial buffer.
///
/// `col` must have length `col_rows * col_cols` where
/// - `col_rows = kernel_h * kernel_w * in_channels`
/// - `col_cols = out_h * out_w`
///
/// Returns a flat buffer of length `in_channels * height * width` (row-major,
/// channels-first).  Padding regions are discarded; overlapping contributions
/// (from stride < kernel) are **summed**.
///
/// Returns a zero-filled buffer if the column matrix is empty.
pub fn col2im(
    col: &[f32],
    in_channels: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Vec<f32> {
    debug_assert!(stride_h > 0 && stride_w > 0);

    let padded_h = height + 2 * pad_h;
    let padded_w = width + 2 * pad_w;

    let mut output = vec![0.0_f32; in_channels * height * width];

    if padded_h < kernel_h || padded_w < kernel_w || col.is_empty() {
        return output;
    }

    let out_h = (padded_h - kernel_h) / stride_h + 1;
    let out_w = (padded_w - kernel_w) / stride_w + 1;
    let col_cols = out_h * out_w;

    for ic in 0..in_channels {
        let out_channel_offset = ic * height * width;
        for kh_i in 0..kernel_h {
            for kw_i in 0..kernel_w {
                let col_row = ic * (kernel_h * kernel_w) + kh_i * kernel_w + kw_i;
                let col_row_offset = col_row * col_cols;

                for oh in 0..out_h {
                    let ih = oh * stride_h + kh_i;
                    let in_h_valid = ih >= pad_h && ih < pad_h + height;
                    let orig_ih = ih.wrapping_sub(pad_h);

                    for ow in 0..out_w {
                        if !in_h_valid {
                            continue;
                        }
                        let iw = ow * stride_w + kw_i;
                        if iw < pad_w || iw >= pad_w + width {
                            continue;
                        }
                        let orig_iw = iw - pad_w;
                        output[out_channel_offset + orig_ih * width + orig_iw] +=
                            col[col_row_offset + oh * out_w + ow];
                    }
                }
            }
        }
    }

    output
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive direct convolution for correctness reference.
    /// Input: [C, H, W], Weight: [out_C, in_C, kH, kW], Output: [out_C, out_H, out_W].
    fn naive_conv2d(
        input: &[f32],
        weight: &[f32],
        in_c: usize,
        in_h: usize,
        in_w: usize,
        out_c: usize,
        kh: usize,
        kw: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> Vec<f32> {
        let padded_h = in_h + 2 * pad_h;
        let padded_w = in_w + 2 * pad_w;
        let out_h = (padded_h - kh) / stride_h + 1;
        let out_w = (padded_w - kw) / stride_w + 1;

        // Pad input.
        let mut padded = vec![0.0_f32; in_c * padded_h * padded_w];
        for c in 0..in_c {
            for h in 0..in_h {
                for w in 0..in_w {
                    padded[c * padded_h * padded_w + (h + pad_h) * padded_w + (w + pad_w)] =
                        input[c * in_h * in_w + h * in_w + w];
                }
            }
        }

        let mut out = vec![0.0_f32; out_c * out_h * out_w];
        for oc in 0..out_c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = 0.0_f32;
                    for ic in 0..in_c {
                        for kh_i in 0..kh {
                            for kw_i in 0..kw {
                                let ph = oh * stride_h + kh_i;
                                let pw = ow * stride_w + kw_i;
                                let in_val = padded[ic * padded_h * padded_w + ph * padded_w + pw];
                                let w_idx =
                                    oc * (in_c * kh * kw) + ic * (kh * kw) + kh_i * kw + kw_i;
                                acc += in_val * weight[w_idx];
                            }
                        }
                    }
                    out[oc * out_h * out_w + oh * out_w + ow] = acc;
                }
            }
        }
        out
    }

    /// Compute convolution via im2col + explicit GEMM and compare with naive.
    fn im2col_conv2d(
        input: &[f32],
        weight: &[f32],
        in_c: usize,
        in_h: usize,
        in_w: usize,
        out_c: usize,
        kh: usize,
        kw: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> Vec<f32> {
        let (col, col_rows, col_cols) = im2col(
            input, in_c, in_h, in_w, kh, kw, stride_h, stride_w, pad_h, pad_w,
        );

        // weight shape: [out_c, col_rows]  (each row = one filter flattened)
        // col shape:    [col_rows, col_cols]
        // output:       [out_c, col_cols]
        let out_h_w = col_cols; // out_h * out_w
        let mut out = vec![0.0_f32; out_c * out_h_w];
        for oc in 0..out_c {
            for j in 0..out_h_w {
                let mut acc = 0.0_f32;
                for k in 0..col_rows {
                    acc += weight[oc * col_rows + k] * col[k * col_cols + j];
                }
                out[oc * out_h_w + j] = acc;
            }
        }
        out
    }

    #[test]
    fn test_im2col_1x1_kernel_no_padding() {
        // 1×1 conv is just a channel-wise scale — trivial but exercises the basic path.
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // [C=2, H=2, W=2]
        let _weight = vec![2.0, 0.5]; // [out_c=2, in_c=1, kH=1, kW=1] — but let's use out_c=1
                                      // Simpler: out_c=1, weight shape [1, 2, 1, 1]
        let weight = vec![1.0, -1.0]; // out_c=1, in_c=2, kh=1, kw=1

        let naive = naive_conv2d(&input, &weight, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0);
        let im2c = im2col_conv2d(&input, &weight, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0);

        assert_eq!(naive.len(), im2c.len(), "output length mismatch");
        for (i, (a, b)) in naive.iter().zip(im2c.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "mismatch at index {i}: naive={a}, im2col={b}"
            );
        }
    }

    #[test]
    fn test_im2col_3x3_kernel_no_padding() {
        // [C=1, H=5, W=5] input, [out_c=1, in_c=1, 3, 3] weight, stride=1, pad=0
        let input: Vec<f32> = (1..=25).map(|x| x as f32).collect();
        let weight: Vec<f32> = (1..=9).map(|x| x as f32).collect();

        let naive = naive_conv2d(&input, &weight, 1, 5, 5, 1, 3, 3, 1, 1, 0, 0);
        let im2c = im2col_conv2d(&input, &weight, 1, 5, 5, 1, 3, 3, 1, 1, 0, 0);

        assert_eq!(naive.len(), im2c.len());
        for (i, (a, b)) in naive.iter().zip(im2c.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "index {i}: naive={a}, im2col={b}");
        }
    }

    #[test]
    fn test_im2col_3x3_with_padding() {
        // same-padding (pad=1) to keep spatial size
        let input: Vec<f32> = (1..=25).map(|x| x as f32).collect();
        let weight: Vec<f32> = vec![0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0];

        let naive = naive_conv2d(&input, &weight, 1, 5, 5, 1, 3, 3, 1, 1, 1, 1);
        let im2c = im2col_conv2d(&input, &weight, 1, 5, 5, 1, 3, 3, 1, 1, 1, 1);

        assert_eq!(naive.len(), im2c.len());
        for (i, (a, b)) in naive.iter().zip(im2c.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "index {i}: naive={a}, im2col={b}");
        }
    }

    #[test]
    fn test_im2col_stride2() {
        // [C=1, H=6, W=6], 3×3 kernel, stride=2, pad=0 → out 2×2
        let input: Vec<f32> = (1..=36).map(|x| x as f32).collect();
        let weight: Vec<f32> = vec![1.0; 9];

        let naive = naive_conv2d(&input, &weight, 1, 6, 6, 1, 3, 3, 2, 2, 0, 0);
        let im2c = im2col_conv2d(&input, &weight, 1, 6, 6, 1, 3, 3, 2, 2, 0, 0);

        assert_eq!(naive.len(), im2c.len());
        for (i, (a, b)) in naive.iter().zip(im2c.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "index {i}: naive={a}, im2col={b}");
        }
    }

    #[test]
    fn test_im2col_multichannel() {
        // [C=3, H=4, W=4], out_c=2, 3×3 kernel, stride=1, pad=1
        let input: Vec<f32> = (1..=48).map(|x| x as f32).collect();
        let weight: Vec<f32> = (1..=54).map(|x| x as f32 * 0.1).collect(); // [2, 3, 3, 3]

        let naive = naive_conv2d(&input, &weight, 3, 4, 4, 2, 3, 3, 1, 1, 1, 1);
        let im2c = im2col_conv2d(&input, &weight, 3, 4, 4, 2, 3, 3, 1, 1, 1, 1);

        assert_eq!(naive.len(), im2c.len());
        for (i, (a, b)) in naive.iter().zip(im2c.iter()).enumerate() {
            assert!((a - b).abs() < 1e-3, "index {i}: naive={a}, im2col={b}");
        }
    }

    #[test]
    fn test_col2im_roundtrip_identity_kernel() {
        // For a 1×1 identity kernel with no padding, im2col should simply
        // return the input data reshuffled, and col2im should recover it.
        let input: Vec<f32> = (1..=12).map(|x| x as f32).collect(); // [C=3, H=2, W=2]
        let (col, _col_rows, _col_cols) = im2col(&input, 3, 2, 2, 1, 1, 1, 1, 0, 0);
        let recovered = col2im(&col, 3, 2, 2, 1, 1, 1, 1, 0, 0);

        assert_eq!(input.len(), recovered.len());
        for (i, (a, b)) in input.iter().zip(recovered.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "index {i}: input={a}, recovered={b}");
        }
    }

    #[test]
    fn test_im2col_column_shape() {
        // Verify the returned col_rows / col_cols match expectations.
        let input = vec![0.0_f32; 2 * 8 * 8]; // [C=2, H=8, W=8]
        let (col, col_rows, col_cols) = im2col(&input, 2, 8, 8, 3, 3, 1, 1, 1, 1);

        let expected_col_rows = 3 * 3 * 2; // kh * kw * in_c
        let out_h = (8 + 2 - 3) / 1 + 1; // = 8 (same-pad)
        let out_w = out_h;
        let expected_col_cols = out_h * out_w;

        assert_eq!(col_rows, expected_col_rows);
        assert_eq!(col_cols, expected_col_cols);
        assert_eq!(col.len(), col_rows * col_cols);
    }

    #[test]
    fn test_im2col_empty_when_kernel_too_large() {
        let input = vec![1.0_f32; 1 * 2 * 2]; // [C=1, H=2, W=2]
        let (col, col_rows, col_cols) = im2col(&input, 1, 2, 2, 5, 5, 1, 1, 0, 0);
        assert_eq!(col_rows, 0);
        assert_eq!(col_cols, 0);
        assert!(col.is_empty());
    }
}
