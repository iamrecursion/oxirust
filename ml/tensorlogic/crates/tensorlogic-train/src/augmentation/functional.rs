use scirs2_core::ndarray::{ArrayD, Dimension, IxDyn};

use super::error::AugmentationError;
use super::rng::{sample_beta_symmetric, AugRng};

/// Add element-wise Gaussian noise: output = input + N(0, std²).
pub fn gaussian_noise(
    input: &ArrayD<f64>,
    std: f64,
    rng: &mut AugRng,
) -> Result<ArrayD<f64>, AugmentationError> {
    if std < 0.0 {
        return Err(AugmentationError::InvalidNoise { std });
    }
    if input.is_empty() {
        return Err(AugmentationError::EmptyInput);
    }
    let noisy = input.mapv(|x| x + rng.next_normal() * std);
    Ok(noisy)
}

/// Apply inverted dropout: zero each element with probability `p`; scale survivors by 1/(1−p).
///
/// When `training` is `false` the input is returned unchanged (inference mode).
pub fn dropout(
    input: &ArrayD<f64>,
    p: f64,
    training: bool,
    rng: &mut AugRng,
) -> Result<ArrayD<f64>, AugmentationError> {
    if !(0.0..=1.0).contains(&p) {
        return Err(AugmentationError::InvalidProbability(p));
    }
    if !training {
        return Ok(input.clone());
    }
    let scale = if (p - 1.0).abs() < 1e-12 {
        0.0
    } else {
        1.0 / (1.0 - p)
    };
    let result = input.mapv(|x| if rng.next_bool(p) { 0.0 } else { x * scale });
    Ok(result)
}

/// Generate a binary dropout mask of the given shape.
///
/// Each element is 1.0 with probability (1 − p) and 0.0 with probability p.
pub fn dropout_mask(
    shape: &[usize],
    p: f64,
    rng: &mut AugRng,
) -> Result<ArrayD<f64>, AugmentationError> {
    if !(0.0..=1.0).contains(&p) {
        return Err(AugmentationError::InvalidProbability(p));
    }
    let total: usize = shape.iter().product();
    let data: Vec<f64> = (0..total)
        .map(|_| if rng.next_bool(p) { 0.0 } else { 1.0 })
        .collect();
    ArrayD::from_shape_vec(IxDyn(shape), data).map_err(|_| AugmentationError::EmptyInput)
}

/// Mixup: λ·x1 + (1−λ)·x2 where λ ~ Beta(alpha, alpha).
///
/// Returns `(mixed, lambda)`.
pub fn mixup(
    x1: &ArrayD<f64>,
    x2: &ArrayD<f64>,
    alpha: f64,
    rng: &mut AugRng,
) -> Result<(ArrayD<f64>, f64), AugmentationError> {
    if alpha <= 0.0 {
        return Err(AugmentationError::InvalidAlpha(alpha));
    }
    if x1.shape() != x2.shape() {
        return Err(AugmentationError::ShapeMismatch {
            expected: x1.shape().to_vec(),
            got: x2.shape().to_vec(),
        });
    }
    if x1.is_empty() {
        return Err(AugmentationError::EmptyInput);
    }
    let lambda = sample_beta_symmetric(alpha, rng);
    let mixed = x1.mapv(|v| v * lambda) + x2.mapv(|v| v * (1.0 - lambda));
    Ok((mixed, lambda))
}

/// CutMix: paste a random rectangular region from x2 into x1.
///
/// The patch covers fraction (1 − lambda) of the spatial area.
/// Input must have at least 2 dimensions; last two are treated as (H, W).
/// Returns `(mixed, lambda)` where lambda = fraction of x1 retained.
pub fn cutmix(
    x1: &ArrayD<f64>,
    x2: &ArrayD<f64>,
    alpha: f64,
    rng: &mut AugRng,
) -> Result<(ArrayD<f64>, f64), AugmentationError> {
    if alpha <= 0.0 {
        return Err(AugmentationError::InvalidAlpha(alpha));
    }
    if x1.shape() != x2.shape() {
        return Err(AugmentationError::ShapeMismatch {
            expected: x1.shape().to_vec(),
            got: x2.shape().to_vec(),
        });
    }
    if x1.ndim() < 2 {
        return Err(AugmentationError::ShapeMismatch {
            expected: vec![2],
            got: x1.shape().to_vec(),
        });
    }
    if x1.is_empty() {
        return Err(AugmentationError::EmptyInput);
    }

    let ndim = x1.ndim();
    let h = x1.shape()[ndim - 2];
    let w = x1.shape()[ndim - 1];

    // Sample mixing ratio
    let lambda_raw = sample_beta_symmetric(alpha, rng);
    // Patch area fraction = 1 - lambda_raw
    let cut_ratio = (1.0 - lambda_raw).sqrt();
    let cut_h = ((h as f64 * cut_ratio) as usize).max(1).min(h);
    let cut_w = ((w as f64 * cut_ratio) as usize).max(1).min(w);

    // Random top-left corner
    let top = if h > cut_h {
        rng.next_usize(h - cut_h + 1)
    } else {
        0
    };
    let left = if w > cut_w {
        rng.next_usize(w - cut_w + 1)
    } else {
        0
    };

    let actual_lambda = 1.0 - (cut_h * cut_w) as f64 / (h * w) as f64;

    let mut mixed = x1.clone();

    // Iterate over all indices; replace elements inside the bounding box with x2.
    for (idx, val) in mixed.indexed_iter_mut() {
        let raw = idx.slice();
        let ih = raw[ndim - 2];
        let iw = raw[ndim - 1];
        if ih >= top && ih < top + cut_h && iw >= left && iw < left + cut_w {
            *val = x2[idx.clone()];
        }
    }

    Ok((mixed, actual_lambda))
}

/// Random 2-D crop: extract a sub-array of size [.., crop_h, crop_w] at a random position.
///
/// Input must have at least 2 dimensions. All leading batch/channel dimensions are preserved.
pub fn random_crop_2d(
    input: &ArrayD<f64>,
    crop_h: usize,
    crop_w: usize,
    rng: &mut AugRng,
) -> Result<ArrayD<f64>, AugmentationError> {
    let ndim = input.ndim();
    if ndim < 2 {
        return Err(AugmentationError::InvalidCrop {
            crop_size: crop_h,
            input_size: 0,
        });
    }
    let h = input.shape()[ndim - 2];
    let w = input.shape()[ndim - 1];
    if crop_h > h {
        return Err(AugmentationError::InvalidCrop {
            crop_size: crop_h,
            input_size: h,
        });
    }
    if crop_w > w {
        return Err(AugmentationError::InvalidCrop {
            crop_size: crop_w,
            input_size: w,
        });
    }
    let top = if h > crop_h {
        rng.next_usize(h - crop_h + 1)
    } else {
        0
    };
    let left = if w > crop_w {
        rng.next_usize(w - crop_w + 1)
    } else {
        0
    };

    crop_2d_impl(input, top, left, crop_h, crop_w)
}

/// Center crop: crop `[crop_h, crop_w]` from the center of the last two spatial dims.
pub fn center_crop_2d(
    input: &ArrayD<f64>,
    crop_h: usize,
    crop_w: usize,
) -> Result<ArrayD<f64>, AugmentationError> {
    let ndim = input.ndim();
    if ndim < 2 {
        return Err(AugmentationError::InvalidCrop {
            crop_size: crop_h,
            input_size: 0,
        });
    }
    let h = input.shape()[ndim - 2];
    let w = input.shape()[ndim - 1];
    if crop_h > h {
        return Err(AugmentationError::InvalidCrop {
            crop_size: crop_h,
            input_size: h,
        });
    }
    if crop_w > w {
        return Err(AugmentationError::InvalidCrop {
            crop_size: crop_w,
            input_size: w,
        });
    }
    let top = (h - crop_h) / 2;
    let left = (w - crop_w) / 2;
    crop_2d_impl(input, top, left, crop_h, crop_w)
}

/// Internal helper: extract sub-array given top-left corner and crop dimensions.
fn crop_2d_impl(
    input: &ArrayD<f64>,
    top: usize,
    left: usize,
    crop_h: usize,
    crop_w: usize,
) -> Result<ArrayD<f64>, AugmentationError> {
    let ndim = input.ndim();

    // Build output shape: leading dims unchanged, last two = crop_h, crop_w.
    let mut out_shape = input.shape().to_vec();
    out_shape[ndim - 2] = crop_h;
    out_shape[ndim - 1] = crop_w;

    let total: usize = out_shape.iter().product();
    let mut data = Vec::with_capacity(total);

    // Iterate over all output linear indices, reconstruct multi-dim coords,
    // offset the spatial dims, then index the input.
    for flat in 0..total {
        let mut rem = flat;
        let mut out_idx = vec![0usize; ndim];
        for d in (0..ndim).rev() {
            out_idx[d] = rem % out_shape[d];
            rem /= out_shape[d];
        }
        // Map output spatial coords to input spatial coords.
        let mut src_idx = out_idx.clone();
        src_idx[ndim - 2] += top;
        src_idx[ndim - 1] += left;

        let v = input[IxDyn(&src_idx)];
        data.push(v);
    }

    ArrayD::from_shape_vec(IxDyn(&out_shape), data).map_err(|_| AugmentationError::EmptyInput)
}

/// Random horizontal flip of the last two spatial dimensions with probability `p`.
pub fn random_hflip(
    input: &ArrayD<f64>,
    p: f64,
    rng: &mut AugRng,
) -> Result<ArrayD<f64>, AugmentationError> {
    if !(0.0..=1.0).contains(&p) {
        return Err(AugmentationError::InvalidProbability(p));
    }
    if !rng.next_bool(p) {
        return Ok(input.clone());
    }
    hflip_impl(input)
}

/// Random vertical flip of the last two spatial dimensions with probability `p`.
pub fn random_vflip(
    input: &ArrayD<f64>,
    p: f64,
    rng: &mut AugRng,
) -> Result<ArrayD<f64>, AugmentationError> {
    if !(0.0..=1.0).contains(&p) {
        return Err(AugmentationError::InvalidProbability(p));
    }
    if !rng.next_bool(p) {
        return Ok(input.clone());
    }
    vflip_impl(input)
}

/// Internal horizontal flip (flip along last dim = width).
fn hflip_impl(input: &ArrayD<f64>) -> Result<ArrayD<f64>, AugmentationError> {
    let ndim = input.ndim();
    if ndim < 2 {
        return Err(AugmentationError::InvalidCrop {
            crop_size: 0,
            input_size: 0,
        });
    }
    let w = input.shape()[ndim - 1];
    let shape = input.shape().to_vec();
    let total: usize = shape.iter().product();
    let mut data = vec![0.0f64; total];

    for (flat, val) in input.iter().enumerate() {
        let mut rem = flat;
        let mut idx = vec![0usize; ndim];
        for d in (0..ndim).rev() {
            idx[d] = rem % shape[d];
            rem /= shape[d];
        }
        // Flip the last (width) dimension.
        idx[ndim - 1] = w - 1 - idx[ndim - 1];
        let mut dst_flat = 0usize;
        let mut stride = 1usize;
        for d in (0..ndim).rev() {
            dst_flat += idx[d] * stride;
            stride *= shape[d];
        }
        data[dst_flat] = *val;
    }

    ArrayD::from_shape_vec(IxDyn(&shape), data).map_err(|_| AugmentationError::EmptyInput)
}

/// Internal vertical flip (flip along second-to-last dim = height).
fn vflip_impl(input: &ArrayD<f64>) -> Result<ArrayD<f64>, AugmentationError> {
    let ndim = input.ndim();
    if ndim < 2 {
        return Err(AugmentationError::InvalidCrop {
            crop_size: 0,
            input_size: 0,
        });
    }
    let h = input.shape()[ndim - 2];
    let shape = input.shape().to_vec();
    let total: usize = shape.iter().product();
    let mut data = vec![0.0f64; total];

    for (flat, val) in input.iter().enumerate() {
        let mut rem = flat;
        let mut idx = vec![0usize; ndim];
        for d in (0..ndim).rev() {
            idx[d] = rem % shape[d];
            rem /= shape[d];
        }
        // Flip the second-to-last (height) dimension.
        idx[ndim - 2] = h - 1 - idx[ndim - 2];
        let mut dst_flat = 0usize;
        let mut stride = 1usize;
        for d in (0..ndim).rev() {
            dst_flat += idx[d] * stride;
            stride *= shape[d];
        }
        data[dst_flat] = *val;
    }

    ArrayD::from_shape_vec(IxDyn(&shape), data).map_err(|_| AugmentationError::EmptyInput)
}

/// Normalize input: `(x − mean[c]) / std[c]`.
///
/// For a `[B, C, H, W]` or `[C, H, W]` tensor the normalization is per-channel.
/// If `mean` and `std` have length 1 the same value is applied to all channels.
/// For 1-D or 2-D tensors element-wise normalization uses the first element of `mean`/`std`.
pub fn normalize(
    input: &ArrayD<f64>,
    mean: &[f64],
    std: &[f64],
) -> Result<ArrayD<f64>, AugmentationError> {
    if mean.is_empty() || std.is_empty() {
        return Err(AugmentationError::EmptyInput);
    }
    if input.is_empty() {
        return Err(AugmentationError::EmptyInput);
    }

    let ndim = input.ndim();
    let shape = input.shape().to_vec();

    // Determine which axis is the channel axis: axis 1 for ndim >= 3, else element-wise.
    if ndim >= 3 {
        let num_channels = shape[ndim - 3]; // C dim: ..., C, H, W
                                            // Broadcast mean/std to num_channels length.
        let m: Vec<f64> = broadcast_stats(mean, num_channels)?;
        let s: Vec<f64> = broadcast_stats(std, num_channels)?;

        let mut result = input.clone();
        // Iterate over every element and apply per-channel normalization.
        for (idx, val) in result.indexed_iter_mut() {
            let raw = idx.slice();
            let c = raw[ndim - 3];
            *val = (*val - m[c]) / s[c];
        }
        Ok(result)
    } else {
        // 1-D or 2-D: scalar normalization.
        let m = mean[0];
        let s = std[0];
        Ok(input.mapv(|x| (x - m) / s))
    }
}

/// Denormalize input: `x * std[c] + mean[c]` (inverse of `normalize`).
pub fn denormalize(
    input: &ArrayD<f64>,
    mean: &[f64],
    std: &[f64],
) -> Result<ArrayD<f64>, AugmentationError> {
    if mean.is_empty() || std.is_empty() {
        return Err(AugmentationError::EmptyInput);
    }
    if input.is_empty() {
        return Err(AugmentationError::EmptyInput);
    }

    let ndim = input.ndim();
    let shape = input.shape().to_vec();

    if ndim >= 3 {
        let num_channels = shape[ndim - 3];
        let m: Vec<f64> = broadcast_stats(mean, num_channels)?;
        let s: Vec<f64> = broadcast_stats(std, num_channels)?;

        let mut result = input.clone();
        for (idx, val) in result.indexed_iter_mut() {
            let raw = idx.slice();
            let c = raw[ndim - 3];
            *val = *val * s[c] + m[c];
        }
        Ok(result)
    } else {
        let m = mean[0];
        let s = std[0];
        Ok(input.mapv(|x| x * s + m))
    }
}

/// Broadcast a stats slice to `n` channels.
///
/// If len == 1, replicate; if len == n, use as-is; otherwise error.
fn broadcast_stats(stats: &[f64], n: usize) -> Result<Vec<f64>, AugmentationError> {
    if stats.len() == 1 {
        Ok(vec![stats[0]; n])
    } else if stats.len() == n {
        Ok(stats.to_vec())
    } else {
        Err(AugmentationError::ShapeMismatch {
            expected: vec![n],
            got: vec![stats.len()],
        })
    }
}

/// Clamp all elements to `[min_val, max_val]`.
pub fn clip(input: &ArrayD<f64>, min_val: f64, max_val: f64) -> ArrayD<f64> {
    input.mapv(|x| x.clamp(min_val, max_val))
}
