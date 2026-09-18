//! Residual / skip-connection helpers and sub-pixel convolution (PixelShuffle).
//!
//! ## Residual blocks
//!
//! ResNet-style architectures route the input around one or more layers and add
//! it back to the transformed output:
//!
//! ```text
//! output = F(x) + projection(x)   // general projection residual
//! output = F(x) + x               // identity residual (same shape)
//! ```
//!
//! ## PixelShuffle
//!
//! Sub-pixel convolution rearranges `C*r²` feature channels into a spatially
//! `r×` upsampled tensor with `C` channels, enabling efficient learned
//! super-resolution with no checkerboard artefacts from transposed convolutions.

use crate::error::NeuralError;
use crate::layers::{BatchNorm2d, Conv2dLayer, LinearLayer};
use crate::tensor::{add, Tensor};

// ──────────────────────────────────────────────────────────────────────────────
// PixelShuffle
// ──────────────────────────────────────────────────────────────────────────────

/// Sub-pixel convolution (PixelShuffle / depth-to-space) layer.
///
/// Rearranges a `[C * r², H, W]` (or `[N, C * r², H, W]`) tensor into
/// `[C, H*r, W*r]` (or `[N, C, H*r, W*r]`) by treating groups of `r²` input
/// channels as a 2-D tile of output pixels.
///
/// The channel interleaving follows the standard PyTorch convention:
/// channel index `c * r² + r_y * r + r_x` maps to output pixel
/// `(y * r + r_y, x * r + r_x)` in output channel `c`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelShuffle {
    /// Integer upscale factor `r`.
    pub upscale_factor: usize,
}

impl PixelShuffle {
    /// Creates a `PixelShuffle` with the given upscale factor.
    ///
    /// Returns an error if `upscale_factor == 0`.
    pub fn new(upscale_factor: usize) -> Result<Self, NeuralError> {
        if upscale_factor == 0 {
            return Err(NeuralError::InvalidShape(
                "PixelShuffle: upscale_factor must be > 0".to_string(),
            ));
        }
        Ok(Self { upscale_factor })
    }

    /// Forward pass on a 3-D tensor `[C*r², H, W]` → `[C, H*r, W*r]`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        let r = self.upscale_factor;
        match input.ndim() {
            3 => {
                let (c_times_r2, in_h, in_w) =
                    (input.shape()[0], input.shape()[1], input.shape()[2]);
                let r2 = r * r;
                if c_times_r2 % r2 != 0 {
                    return Err(NeuralError::ShapeMismatch(format!(
                        "PixelShuffle: input channels {} is not divisible by r²={}",
                        c_times_r2, r2
                    )));
                }
                let c_out = c_times_r2 / r2;
                let out_h = in_h * r;
                let out_w = in_w * r;
                pixel_shuffle_3d(input, r, c_out, in_h, in_w, out_h, out_w)
            }
            4 => {
                let (n, c_times_r2, in_h, in_w) = (
                    input.shape()[0],
                    input.shape()[1],
                    input.shape()[2],
                    input.shape()[3],
                );
                let r2 = r * r;
                if c_times_r2 % r2 != 0 {
                    return Err(NeuralError::ShapeMismatch(format!(
                        "PixelShuffle: input channels {} is not divisible by r²={}",
                        c_times_r2, r2
                    )));
                }
                let c_out = c_times_r2 / r2;
                let out_h = in_h * r;
                let out_w = in_w * r;
                let mut out_data = vec![0.0_f32; n * c_out * out_h * out_w];
                let sample_in = c_times_r2 * in_h * in_w;
                let sample_out = c_out * out_h * out_w;
                for b in 0..n {
                    let in_slice = &input.data()[b * sample_in..(b + 1) * sample_in];
                    let out_slice = &mut out_data[b * sample_out..(b + 1) * sample_out];
                    shuffle_slice(in_slice, out_slice, r, c_out, in_h, in_w, out_h, out_w);
                }
                Tensor::from_data(out_data, vec![n, c_out, out_h, out_w])
            }
            other => Err(NeuralError::InvalidShape(format!(
                "PixelShuffle: expected 3-D or 4-D input, got rank {other}"
            ))),
        }
    }
}

/// Shuffle a single sample (flat C*r²×H×W slice into C×H*r×W*r).
fn shuffle_slice(
    inp: &[f32],
    out: &mut [f32],
    r: usize,
    c_out: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) {
    let r2 = r * r;
    for c in 0..c_out {
        for ry in 0..r {
            for rx in 0..r {
                let src_ch = c * r2 + ry * r + rx;
                for iy in 0..in_h {
                    for ix in 0..in_w {
                        let src_idx = src_ch * in_h * in_w + iy * in_w + ix;
                        let oy = iy * r + ry;
                        let ox = ix * r + rx;
                        let dst_idx = c * out_h * out_w + oy * out_w + ox;
                        out[dst_idx] = inp[src_idx];
                    }
                }
            }
        }
    }
}

fn pixel_shuffle_3d(
    input: &Tensor,
    r: usize,
    c_out: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> Result<Tensor, NeuralError> {
    let mut out_data = vec![0.0_f32; c_out * out_h * out_w];
    shuffle_slice(
        input.data(),
        &mut out_data,
        r,
        c_out,
        in_h,
        in_w,
        out_h,
        out_w,
    );
    Tensor::from_data(out_data, vec![c_out, out_h, out_w])
}

// ──────────────────────────────────────────────────────────────────────────────
// ResidualBlock
// ──────────────────────────────────────────────────────────────────────────────

/// A classic BasicBlock residual unit (two 3×3 Conv2d + BN + ReLU, identity shortcut).
///
/// The identity shortcut is used when `in_channels == out_channels` and
/// `stride == (1,1)`.  Otherwise a 1×1 projection convolution is applied to
/// the skip path so that the dimensions match.
///
/// Architecture (with projection shortcut):
/// ```text
/// ┌─ conv1(3×3, stride) ─ BN ─ ReLU ─ conv2(3×3, stride=1) ─ BN ─┐
/// x ───────────────────────────────────────────────────────── proj ─ add ─ ReLU ─ out
/// ```
#[derive(Debug, Clone)]
pub struct ResidualBlock {
    /// First 3×3 convolution.
    pub conv1: Conv2dLayer,
    /// Batch normalisation after conv1.
    pub bn1: BatchNorm2d,
    /// Second 3×3 convolution.
    pub conv2: Conv2dLayer,
    /// Batch normalisation after conv2.
    pub bn2: BatchNorm2d,
    /// Optional 1×1 projection on the skip path (None if identity shortcut).
    pub projection: Option<Conv2dLayer>,
    /// Stride applied to the first convolution.
    pub stride: (usize, usize),
}

impl ResidualBlock {
    /// Creates a zero-initialised `ResidualBlock`.
    ///
    /// * `in_channels` — number of input feature-map channels.
    /// * `out_channels` — number of output feature-map channels.
    /// * `stride` — stride for the first convolution (usually `(1,1)` or
    ///   `(2,2)` for downsampling).
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: (usize, usize),
    ) -> Result<Self, NeuralError> {
        if in_channels == 0 || out_channels == 0 {
            return Err(NeuralError::InvalidShape(
                "ResidualBlock: channel counts must be > 0".to_string(),
            ));
        }
        if stride.0 == 0 || stride.1 == 0 {
            return Err(NeuralError::InvalidShape(
                "ResidualBlock: stride must be > 0 in both dimensions".to_string(),
            ));
        }

        // conv1: in_channels → out_channels, stride, pad=1
        let conv1 = Conv2dLayer::new(in_channels, out_channels, 3, 3, stride, (1, 1))?;
        let bn1 = BatchNorm2d::new(out_channels)?;
        // conv2: out_channels → out_channels, stride=1, pad=1
        let conv2 = Conv2dLayer::new(out_channels, out_channels, 3, 3, (1, 1), (1, 1))?;
        let bn2 = BatchNorm2d::new(out_channels)?;

        // Need a projection shortcut when shape changes.
        let needs_projection = in_channels != out_channels || stride.0 != 1 || stride.1 != 1;
        let projection = if needs_projection {
            Some(Conv2dLayer::new(
                in_channels,
                out_channels,
                1,
                1,
                stride,
                (0, 0),
            )?)
        } else {
            None
        };

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            projection,
            stride,
        })
    }

    /// Forward pass on a 3-D `[C, H, W]` or 4-D `[N, C, H, W]` input tensor.
    ///
    /// Returns a tensor of shape `[out_channels, H', W']` or
    /// `[N, out_channels, H', W']`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        match input.ndim() {
            3 => self.forward_single(input),
            4 => {
                let n = input.shape()[0];
                let c = input.shape()[1];
                let h = input.shape()[2];
                let w = input.shape()[3];
                let sample_size = c * h * w;
                let mut outputs: Vec<f32> = Vec::new();
                let mut out_shape: Option<Vec<usize>> = None;
                for b in 0..n {
                    let slice = Tensor::from_data(
                        input.data()[b * sample_size..(b + 1) * sample_size].to_vec(),
                        vec![c, h, w],
                    )?;
                    let out = self.forward_single(&slice)?;
                    if out_shape.is_none() {
                        out_shape = Some(out.shape().to_vec());
                    }
                    outputs.extend_from_slice(out.data());
                }
                let os = out_shape.unwrap_or_default();
                let full_shape = [vec![n], os].concat();
                Tensor::from_data(outputs, full_shape)
            }
            other => Err(NeuralError::InvalidShape(format!(
                "ResidualBlock: expected 3-D or 4-D input, got rank {other}"
            ))),
        }
    }

    fn forward_single(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        // Main path.
        let h = self.conv1.forward(input)?;
        let h = self.bn1.forward(&h)?;
        let h = apply_relu_t(&h);
        let h = self.conv2.forward(&h)?;
        let h = self.bn2.forward(&h)?;

        // Skip path.
        let skip = match &self.projection {
            Some(proj) => proj.forward(input)?,
            None => input.clone(),
        };

        // Element-wise add then ReLU.
        let out = add(&h, &skip)?;
        Ok(apply_relu_t(&out))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// BottleneckBlock
// ──────────────────────────────────────────────────────────────────────────────

/// ResNet bottleneck residual block (1×1 ─ 3×3 ─ 1×1) used in deeper variants.
///
/// `mid_channels` is the width of the bottleneck (typically `out_channels / 4`
/// in the original ResNet-50/101/152).
#[derive(Debug, Clone)]
pub struct BottleneckBlock {
    /// 1×1 channel reduction.
    pub conv1: Conv2dLayer,
    /// BN after conv1.
    pub bn1: BatchNorm2d,
    /// 3×3 spatial convolution.
    pub conv2: Conv2dLayer,
    /// BN after conv2.
    pub bn2: BatchNorm2d,
    /// 1×1 channel expansion.
    pub conv3: Conv2dLayer,
    /// BN after conv3.
    pub bn3: BatchNorm2d,
    /// Optional projection on the skip path.
    pub projection: Option<Conv2dLayer>,
    /// Stride applied to the 3×3 convolution.
    pub stride: (usize, usize),
}

impl BottleneckBlock {
    /// Creates a zero-initialised `BottleneckBlock`.
    ///
    /// * `in_channels` — channels entering the block.
    /// * `mid_channels` — bottleneck width (the 3×3 channel count).
    /// * `out_channels` — channels leaving the block (after the 1×1 expansion).
    /// * `stride` — stride for the 3×3 convolution.
    pub fn new(
        in_channels: usize,
        mid_channels: usize,
        out_channels: usize,
        stride: (usize, usize),
    ) -> Result<Self, NeuralError> {
        if in_channels == 0 || mid_channels == 0 || out_channels == 0 {
            return Err(NeuralError::InvalidShape(
                "BottleneckBlock: channel counts must be > 0".to_string(),
            ));
        }
        if stride.0 == 0 || stride.1 == 0 {
            return Err(NeuralError::InvalidShape(
                "BottleneckBlock: stride must be > 0".to_string(),
            ));
        }

        let conv1 = Conv2dLayer::new(in_channels, mid_channels, 1, 1, (1, 1), (0, 0))?;
        let bn1 = BatchNorm2d::new(mid_channels)?;
        let conv2 = Conv2dLayer::new(mid_channels, mid_channels, 3, 3, stride, (1, 1))?;
        let bn2 = BatchNorm2d::new(mid_channels)?;
        let conv3 = Conv2dLayer::new(mid_channels, out_channels, 1, 1, (1, 1), (0, 0))?;
        let bn3 = BatchNorm2d::new(out_channels)?;

        let needs_projection = in_channels != out_channels || stride.0 != 1 || stride.1 != 1;
        let projection = if needs_projection {
            Some(Conv2dLayer::new(
                in_channels,
                out_channels,
                1,
                1,
                stride,
                (0, 0),
            )?)
        } else {
            None
        };

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            projection,
            stride,
        })
    }

    /// Forward pass on `[C, H, W]` or `[N, C, H, W]`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        match input.ndim() {
            3 => self.forward_single(input),
            4 => {
                let n = input.shape()[0];
                let c = input.shape()[1];
                let h = input.shape()[2];
                let w = input.shape()[3];
                let sample_size = c * h * w;
                let mut outputs: Vec<f32> = Vec::new();
                let mut out_shape: Option<Vec<usize>> = None;
                for b in 0..n {
                    let slice = Tensor::from_data(
                        input.data()[b * sample_size..(b + 1) * sample_size].to_vec(),
                        vec![c, h, w],
                    )?;
                    let out = self.forward_single(&slice)?;
                    if out_shape.is_none() {
                        out_shape = Some(out.shape().to_vec());
                    }
                    outputs.extend_from_slice(out.data());
                }
                let os = out_shape.unwrap_or_default();
                let full_shape = [vec![n], os].concat();
                Tensor::from_data(outputs, full_shape)
            }
            other => Err(NeuralError::InvalidShape(format!(
                "BottleneckBlock: expected 3-D or 4-D input, got rank {other}"
            ))),
        }
    }

    fn forward_single(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        // Main path: 1×1 → BN → ReLU → 3×3 → BN → ReLU → 1×1 → BN
        let h = apply_relu_t(&self.bn1.forward(&self.conv1.forward(input)?)?);
        let h = apply_relu_t(&self.bn2.forward(&self.conv2.forward(&h)?)?);
        let h = self.bn3.forward(&self.conv3.forward(&h)?)?;

        // Skip path.
        let skip = match &self.projection {
            Some(proj) => proj.forward(input)?,
            None => input.clone(),
        };

        let out = add(&h, &skip)?;
        Ok(apply_relu_t(&out))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SkipAdd helper
// ──────────────────────────────────────────────────────────────────────────────

/// Adds two tensors element-wise (skip / residual connection).
///
/// This is a thin wrapper around [`crate::tensor::add`] that accepts tensors
/// by shared reference and returns a `Result`, making it easy to express skip
/// connections inline in a `?`-chain:
///
/// ```rust
/// use oximedia_neural::skip_connections::skip_add;
/// use oximedia_neural::tensor::Tensor;
///
/// let a = Tensor::ones(vec![4]).unwrap();
/// let b = Tensor::ones(vec![4]).unwrap();
/// let c = skip_add(&a, &b).unwrap();   // [2.0, 2.0, 2.0, 2.0]
/// ```
#[inline]
pub fn skip_add(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuralError> {
    add(a, b)
}

/// Applies a linear projection to `skip` so it matches the shape of `main`,
/// then adds them (identity if no projection needed).
///
/// This is useful when building ad-hoc residual connections where the skip
/// path may have a different channel count than the main path.
pub fn skip_add_projected(
    main: &Tensor,
    skip: &Tensor,
    projection: &LinearLayer,
) -> Result<Tensor, NeuralError> {
    if main.shape() == skip.shape() {
        // Same shape — identity shortcut.
        return add(main, skip);
    }
    // Apply the linear projection to the flattened skip tensor.
    let flat_skip = skip.reshape(vec![skip.numel()])?;
    let projected = projection.forward(&flat_skip)?;
    let reshaped = projected.reshape(main.shape().to_vec())?;
    add(main, &reshaped)
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

#[inline]
fn apply_relu_t(t: &Tensor) -> Tensor {
    let new_data: Vec<f32> = t.data().iter().map(|&x| x.max(0.0)).collect();
    Tensor::from_data(new_data, t.shape().to_vec())
        .unwrap_or_else(|_| unreachable!("apply_relu_t: shape is valid by construction"))
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    // ── PixelShuffle ─────────────────────────────────────────────────────────

    #[test]
    fn test_pixel_shuffle_r2_shape_3d() {
        // [C*4, H, W] → [C, H*2, W*2]
        let ps = PixelShuffle::new(2).expect("pixel shuffle new");
        let input = Tensor::zeros(vec![4, 3, 3]).expect("tensor zeros"); // 1 out-ch, r=2, 3×3 in
        let out = ps.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[1, 6, 6]);
    }

    #[test]
    fn test_pixel_shuffle_r2_values() {
        // 1-channel r=2 PixelShuffle: channels [ch0, ch1, ch2, ch3] → 2×2 spatial tile.
        // pixel interleave: ch=c*r²+ry*r+rx; c=0 always.
        // ch0→(ry=0,rx=0), ch1→(ry=0,rx=1), ch2→(ry=1,rx=0), ch3→(ry=1,rx=1)
        let ps = PixelShuffle::new(2).expect("pixel shuffle new");
        // Single 1×1 spatial, 4 input channels → 2×2 output, 1 channel
        let data = vec![10.0_f32, 20.0, 30.0, 40.0]; // ch0..ch3 at pixel (0,0)
        let input = Tensor::from_data(data, vec![4, 1, 1]).expect("tensor from_data");
        let out = ps.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[1, 2, 2]);
        // Expected: [10, 20, 30, 40] arranged in 2×2
        assert!(close(out.data()[0], 10.0)); // (0,0)
        assert!(close(out.data()[1], 20.0)); // (0,1)
        assert!(close(out.data()[2], 30.0)); // (1,0)
        assert!(close(out.data()[3], 40.0)); // (1,1)
    }

    #[test]
    fn test_pixel_shuffle_r2_batch_shape() {
        let ps = PixelShuffle::new(2).expect("pixel shuffle new");
        // [N=2, C*4=8, H=3, W=3] → [2, 2, 6, 6]
        let input = Tensor::zeros(vec![2, 8, 3, 3]).expect("tensor zeros");
        let out = ps.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[2, 2, 6, 6]);
    }

    #[test]
    fn test_pixel_shuffle_zero_factor_error() {
        assert!(PixelShuffle::new(0).is_err());
    }

    #[test]
    fn test_pixel_shuffle_channel_not_divisible() {
        let ps = PixelShuffle::new(2).expect("pixel shuffle new");
        // 3 channels is not divisible by r²=4
        let input = Tensor::zeros(vec![3, 4, 4]).expect("tensor zeros");
        assert!(ps.forward(&input).is_err());
    }

    #[test]
    fn test_pixel_shuffle_wrong_rank() {
        let ps = PixelShuffle::new(2).expect("pixel shuffle new");
        let input = Tensor::zeros(vec![4, 4]).expect("tensor zeros");
        assert!(ps.forward(&input).is_err());
    }

    #[test]
    fn test_pixel_shuffle_r1_identity() {
        // r=1 → no rearrangement, shape unchanged
        let ps = PixelShuffle::new(1).expect("pixel shuffle new");
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let input = Tensor::from_data(data.clone(), vec![3, 2, 2]).expect("tensor from_data");
        let out = ps.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[3, 2, 2]);
        // Values should be identical (identity permutation)
        for (a, b) in out.data().iter().zip(data.iter()) {
            assert!(close(*a, *b));
        }
    }

    // ── skip_add ─────────────────────────────────────────────────────────────

    #[test]
    fn test_skip_add_basic() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3]).expect("tensor from_data");
        let b = Tensor::from_data(vec![10.0, 20.0, 30.0], vec![3]).expect("tensor from_data");
        let c = skip_add(&a, &b).expect("add");
        assert!(close(c.data()[0], 11.0));
        assert!(close(c.data()[1], 22.0));
        assert!(close(c.data()[2], 33.0));
    }

    #[test]
    fn test_skip_add_shape_mismatch_error() {
        let a = Tensor::zeros(vec![3]).expect("tensor zeros");
        let b = Tensor::zeros(vec![4]).expect("tensor zeros");
        assert!(skip_add(&a, &b).is_err());
    }

    // ── ResidualBlock ─────────────────────────────────────────────────────────

    #[test]
    fn test_residual_block_identity_shortcut_shape() {
        // same in/out channels, stride=1 → identity shortcut
        let block = ResidualBlock::new(4, 4, (1, 1)).expect("residual block new");
        assert!(block.projection.is_none());
        let input = Tensor::zeros(vec![4, 8, 8]).expect("tensor zeros");
        let out = block.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[4, 8, 8]);
    }

    #[test]
    fn test_residual_block_projection_shortcut_shape() {
        // different channels → projection shortcut
        let block = ResidualBlock::new(2, 4, (1, 1)).expect("residual block new");
        assert!(block.projection.is_some());
        let input = Tensor::zeros(vec![2, 8, 8]).expect("tensor zeros");
        let out = block.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[4, 8, 8]);
    }

    #[test]
    fn test_residual_block_stride2_downsamples() {
        let block = ResidualBlock::new(4, 4, (2, 2)).expect("residual block new");
        // stride 2 → H/2, W/2 (with padding=1)
        let input = Tensor::zeros(vec![4, 8, 8]).expect("tensor zeros");
        let out = block.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[4, 4, 4]);
    }

    #[test]
    fn test_residual_block_output_non_negative_due_to_relu() {
        let block = ResidualBlock::new(2, 2, (1, 1)).expect("residual block new");
        let input = Tensor::from_data(
            (0..2 * 4 * 4).map(|i| (i as f32 - 16.0) / 4.0).collect(),
            vec![2, 4, 4],
        )
        .expect("operation should succeed");
        let out = block.forward(&input).expect("forward pass");
        // Output ReLU guarantees all values ≥ 0
        assert!(out.data().iter().all(|&v| v >= -1e-6));
    }

    #[test]
    fn test_residual_block_batch_shape() {
        let block = ResidualBlock::new(2, 4, (1, 1)).expect("residual block new");
        let input = Tensor::zeros(vec![3, 2, 8, 8]).expect("tensor zeros");
        let out = block.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[3, 4, 8, 8]);
    }

    #[test]
    fn test_residual_block_zero_channels_error() {
        assert!(ResidualBlock::new(0, 4, (1, 1)).is_err());
    }

    // ── BottleneckBlock ───────────────────────────────────────────────────────

    #[test]
    fn test_bottleneck_identity_shortcut_shape() {
        // in == out, stride=1
        let block = BottleneckBlock::new(4, 2, 4, (1, 1)).expect("bottleneck block new");
        assert!(block.projection.is_none());
        let input = Tensor::zeros(vec![4, 8, 8]).expect("tensor zeros");
        let out = block.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[4, 8, 8]);
    }

    #[test]
    fn test_bottleneck_projection_shortcut_shape() {
        let block = BottleneckBlock::new(2, 2, 8, (1, 1)).expect("bottleneck block new");
        assert!(block.projection.is_some());
        let input = Tensor::zeros(vec![2, 8, 8]).expect("tensor zeros");
        let out = block.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[8, 8, 8]);
    }

    #[test]
    fn test_bottleneck_batch_shape() {
        let block = BottleneckBlock::new(4, 2, 4, (1, 1)).expect("bottleneck block new");
        let input = Tensor::zeros(vec![3, 4, 6, 6]).expect("tensor zeros");
        let out = block.forward(&input).expect("forward pass");
        assert_eq!(out.shape(), &[3, 4, 6, 6]);
    }
}
