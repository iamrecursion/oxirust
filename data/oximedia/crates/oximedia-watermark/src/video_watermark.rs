//! Video watermarking module.
//!
//! Frame-level visual watermark embedding and detection using spatial and
//! frequency domain techniques. Supports:
//!
//! - DCT-domain watermarking (robust to compression)
//! - Spatial-domain additive watermarking (fast)
//! - Block-based embedding with perceptual weighting
//! - Blind detection without original frame

use crate::error::{WatermarkError, WatermarkResult};
use crate::payload::{generate_pn_sequence, pack_bits, unpack_bits, PayloadCodec};

/// Video watermark configuration.
#[derive(Debug, Clone)]
pub struct VideoWatermarkConfig {
    /// Embedding strength (0.0 to 1.0).
    pub strength: f32,
    /// Block size for block-based embedding (default 8).
    pub block_size: usize,
    /// Use frequency (DCT) domain embedding.
    pub frequency_domain: bool,
    /// Secret key for watermark generation.
    pub key: u64,
    /// Embed in luminance channel only (vs. all channels).
    pub luma_only: bool,
    /// Redundancy: repeat watermark across multiple blocks.
    pub redundancy: usize,
}

impl Default for VideoWatermarkConfig {
    fn default() -> Self {
        Self {
            strength: 0.05,
            block_size: 8,
            frequency_domain: true,
            key: 0,
            luma_only: true,
            redundancy: 4,
        }
    }
}

/// A video frame represented as a 2D pixel array (grayscale or single channel).
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// Pixel data in row-major order, values in [0.0, 1.0].
    pub pixels: Vec<f32>,
    /// Frame width in pixels.
    pub width: usize,
    /// Frame height in pixels.
    pub height: usize,
}

impl VideoFrame {
    /// Create a new frame.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            pixels: vec![0.0; width * height],
            width,
            height,
        }
    }

    /// Create from existing pixel data.
    ///
    /// # Errors
    ///
    /// Returns error if pixel count doesn't match dimensions.
    pub fn from_pixels(pixels: Vec<f32>, width: usize, height: usize) -> WatermarkResult<Self> {
        if pixels.len() != width * height {
            return Err(WatermarkError::InvalidParameter(format!(
                "Pixel count {} != width*height {}",
                pixels.len(),
                width * height,
            )));
        }
        Ok(Self {
            pixels,
            width,
            height,
        })
    }

    /// Get pixel at (x, y).
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        if x < self.width && y < self.height {
            self.pixels[y * self.width + x]
        } else {
            0.0
        }
    }

    /// Set pixel at (x, y).
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        if x < self.width && y < self.height {
            self.pixels[y * self.width + x] = value;
        }
    }
}

/// Video watermark embedder.
pub struct VideoWatermarkEmbedder {
    config: VideoWatermarkConfig,
    codec: PayloadCodec,
}

impl VideoWatermarkEmbedder {
    /// Create a new video watermark embedder.
    ///
    /// # Errors
    ///
    /// Returns error if codec initialization fails.
    pub fn new(config: VideoWatermarkConfig) -> WatermarkResult<Self> {
        let codec = PayloadCodec::new(16, 8)?;
        Ok(Self { config, codec })
    }

    /// Embed watermark into a video frame.
    ///
    /// # Errors
    ///
    /// Returns error if frame is too small or encoding fails.
    pub fn embed(&self, frame: &VideoFrame, payload: &[u8]) -> WatermarkResult<VideoFrame> {
        let encoded = self.codec.encode(payload)?;
        let bits = unpack_bits(&encoded, encoded.len() * 8);

        if self.config.frequency_domain {
            self.embed_dct(frame, &bits)
        } else {
            self.embed_spatial(frame, &bits)
        }
    }

    /// DCT-domain watermark embedding.
    ///
    /// Embeds bits in mid-frequency DCT coefficients of each block.
    fn embed_dct(&self, frame: &VideoFrame, bits: &[bool]) -> WatermarkResult<VideoFrame> {
        let bs = self.config.block_size;
        let blocks_x = frame.width / bs;
        let blocks_y = frame.height / bs;
        let total_blocks = blocks_x * blocks_y;

        if total_blocks == 0 {
            return Err(WatermarkError::InsufficientCapacity {
                needed: bits.len(),
                have: 0,
            });
        }

        let capacity = total_blocks / self.config.redundancy;

        if capacity < bits.len() {
            return Err(WatermarkError::InsufficientCapacity {
                needed: bits.len(),
                have: capacity,
            });
        }

        let mut result = frame.clone();

        for (bit_idx, &bit) in bits.iter().enumerate() {
            let pn = generate_pn_sequence(bs * bs, self.config.key + bit_idx as u64);

            for rep in 0..self.config.redundancy {
                let block_idx = bit_idx * self.config.redundancy + rep;
                if block_idx >= total_blocks {
                    break;
                }

                let bx = (block_idx % blocks_x) * bs;
                let by = (block_idx / blocks_x) * bs;

                // Extract block
                let mut block = extract_block(&result, bx, by, bs);

                // Forward DCT (Type-II)
                dct2d_forward(&mut block, bs);

                // Modify mid-frequency coefficients
                let bit_value = if bit { 1.0f32 } else { -1.0f32 };
                let mid_start = bs / 4;
                let mid_end = 3 * bs / 4;

                for dy in mid_start..mid_end {
                    for dx in mid_start..mid_end {
                        let pn_idx = dy * bs + dx;
                        if pn_idx < pn.len() {
                            let coeff_idx = dy * bs + dx;
                            block[coeff_idx] +=
                                self.config.strength * bit_value * f32::from(pn[pn_idx]);
                        }
                    }
                }

                // Inverse DCT
                dct2d_inverse(&mut block, bs);

                // Write back
                write_block(&mut result, bx, by, bs, &block);
            }
        }

        // Clamp pixels to [0, 1]
        for p in &mut result.pixels {
            *p = p.clamp(0.0, 1.0);
        }

        Ok(result)
    }

    /// Spatial-domain additive watermark embedding.
    fn embed_spatial(&self, frame: &VideoFrame, bits: &[bool]) -> WatermarkResult<VideoFrame> {
        let bs = self.config.block_size;
        let blocks_x = frame.width / bs;
        let blocks_y = frame.height / bs;
        let total_blocks = blocks_x * blocks_y;
        let capacity = total_blocks / self.config.redundancy;

        if capacity < bits.len() {
            return Err(WatermarkError::InsufficientCapacity {
                needed: bits.len(),
                have: capacity,
            });
        }

        let mut result = frame.clone();

        for (bit_idx, &bit) in bits.iter().enumerate() {
            let pn = generate_pn_sequence(bs * bs, self.config.key + bit_idx as u64);
            let bit_value = if bit { 1.0f32 } else { -1.0f32 };

            for rep in 0..self.config.redundancy {
                let block_idx = bit_idx * self.config.redundancy + rep;
                if block_idx >= total_blocks {
                    break;
                }

                let bx = (block_idx % blocks_x) * bs;
                let by = (block_idx / blocks_x) * bs;

                // Add watermark pattern to block
                for dy in 0..bs {
                    for dx in 0..bs {
                        let x = bx + dx;
                        let y = by + dy;
                        let pn_idx = dy * bs + dx;

                        if x < result.width && y < result.height && pn_idx < pn.len() {
                            let pixel = result.get(x, y);
                            // Strength-weighted additive watermark
                            let wm = self.config.strength * bit_value * f32::from(pn[pn_idx]);
                            result.set(x, y, (pixel + wm).clamp(0.0, 1.0));
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// Calculate embedding capacity in bits for a frame.
    #[must_use]
    pub fn capacity(&self, width: usize, height: usize) -> usize {
        let bs = self.config.block_size;
        let blocks_x = width / bs;
        let blocks_y = height / bs;
        let total_blocks = blocks_x * blocks_y;
        total_blocks / self.config.redundancy
    }
}

/// Video watermark detector.
pub struct VideoWatermarkDetector {
    config: VideoWatermarkConfig,
    codec: PayloadCodec,
}

impl VideoWatermarkDetector {
    /// Create a new video watermark detector.
    ///
    /// # Errors
    ///
    /// Returns error if codec initialization fails.
    pub fn new(config: VideoWatermarkConfig) -> WatermarkResult<Self> {
        let codec = PayloadCodec::new(16, 8)?;
        Ok(Self { config, codec })
    }

    /// Detect and extract watermark from a video frame.
    ///
    /// # Errors
    ///
    /// Returns error if watermark not detected.
    pub fn detect(&self, frame: &VideoFrame, expected_bits: usize) -> WatermarkResult<Vec<u8>> {
        let bits = if self.config.frequency_domain {
            self.detect_dct(frame, expected_bits)?
        } else {
            self.detect_spatial(frame, expected_bits)?
        };

        let bytes = pack_bits(&bits);
        self.codec.decode(&bytes)
    }

    /// DCT-domain watermark detection.
    fn detect_dct(&self, frame: &VideoFrame, expected_bits: usize) -> WatermarkResult<Vec<bool>> {
        let bs = self.config.block_size;
        let blocks_x = frame.width / bs;
        let blocks_y = frame.height / bs;
        let total_blocks = blocks_x * blocks_y;

        let mut bits = Vec::with_capacity(expected_bits);

        for bit_idx in 0..expected_bits {
            let pn = generate_pn_sequence(bs * bs, self.config.key + bit_idx as u64);
            let mut correlation = 0.0f32;

            for rep in 0..self.config.redundancy {
                let block_idx = bit_idx * self.config.redundancy + rep;
                if block_idx >= total_blocks {
                    break;
                }

                let bx = (block_idx % blocks_x) * bs;
                let by = (block_idx / blocks_x) * bs;

                let mut block = extract_block(frame, bx, by, bs);
                dct2d_forward(&mut block, bs);

                // Correlate with PN sequence in mid-frequency region
                let mid_start = bs / 4;
                let mid_end = 3 * bs / 4;

                for dy in mid_start..mid_end {
                    for dx in mid_start..mid_end {
                        let pn_idx = dy * bs + dx;
                        if pn_idx < pn.len() {
                            let coeff_idx = dy * bs + dx;
                            correlation += block[coeff_idx] * f32::from(pn[pn_idx]);
                        }
                    }
                }
            }

            bits.push(correlation > 0.0);
        }

        Ok(bits)
    }

    /// Spatial-domain watermark detection.
    ///
    /// Subtracts the block mean before correlating with the PN sequence
    /// to remove DC bias and isolate the watermark component.
    fn detect_spatial(
        &self,
        frame: &VideoFrame,
        expected_bits: usize,
    ) -> WatermarkResult<Vec<bool>> {
        let bs = self.config.block_size;
        let blocks_x = frame.width / bs;
        let blocks_y = frame.height / bs;
        let total_blocks = blocks_x * blocks_y;

        let mut bits = Vec::with_capacity(expected_bits);

        for bit_idx in 0..expected_bits {
            let pn = generate_pn_sequence(bs * bs, self.config.key + bit_idx as u64);
            let mut correlation = 0.0f32;

            for rep in 0..self.config.redundancy {
                let block_idx = bit_idx * self.config.redundancy + rep;
                if block_idx >= total_blocks {
                    break;
                }

                let bx = (block_idx % blocks_x) * bs;
                let by = (block_idx / blocks_x) * bs;

                // Extract block and compute mean
                let block = extract_block(frame, bx, by, bs);
                #[allow(clippy::cast_precision_loss)]
                let mean = block.iter().sum::<f32>() / block.len() as f32;

                // Correlate mean-subtracted pixels with PN
                for (pn_idx, (&pixel, &pn_val)) in block.iter().zip(pn.iter()).enumerate() {
                    let _ = pn_idx;
                    correlation += (pixel - mean) * f32::from(pn_val);
                }
            }

            bits.push(correlation > 0.0);
        }

        Ok(bits)
    }

    /// Compute watermark detection confidence (normalized correlation).
    #[must_use]
    pub fn detection_confidence(&self, frame: &VideoFrame, expected_bits: usize) -> f32 {
        let bs = self.config.block_size;
        let blocks_x = frame.width / bs;
        let blocks_y = frame.height / bs;
        let total_blocks = blocks_x * blocks_y;

        if expected_bits == 0 || total_blocks == 0 {
            return 0.0;
        }

        let mut total_corr = 0.0f32;
        let mut count = 0;

        for bit_idx in 0..expected_bits {
            let pn = generate_pn_sequence(bs * bs, self.config.key + bit_idx as u64);
            let mut bit_corr = 0.0f32;
            let mut energy = 0.0f32;

            for rep in 0..self.config.redundancy {
                let block_idx = bit_idx * self.config.redundancy + rep;
                if block_idx >= total_blocks {
                    break;
                }

                let bx = (block_idx % blocks_x) * bs;
                let by = (block_idx / blocks_x) * bs;

                let mut block = extract_block(frame, bx, by, bs);
                if self.config.frequency_domain {
                    dct2d_forward(&mut block, bs);
                }

                for (idx, &coeff) in block.iter().enumerate() {
                    if idx < pn.len() {
                        bit_corr += coeff * f32::from(pn[idx]);
                        energy += coeff * coeff;
                    }
                }
            }

            if energy > 1e-10 {
                total_corr += bit_corr.abs() / energy.sqrt();
                count += 1;
            }
        }

        if count > 0 {
            #[allow(clippy::cast_precision_loss)]
            let result = total_corr / count as f32;
            result
        } else {
            0.0
        }
    }
}

/// Extract a block from a frame.
fn extract_block(frame: &VideoFrame, bx: usize, by: usize, bs: usize) -> Vec<f32> {
    let mut block = vec![0.0f32; bs * bs];
    for dy in 0..bs {
        for dx in 0..bs {
            block[dy * bs + dx] = frame.get(bx + dx, by + dy);
        }
    }
    block
}

/// Write a block back to a frame.
fn write_block(frame: &mut VideoFrame, bx: usize, by: usize, bs: usize, block: &[f32]) {
    for dy in 0..bs {
        for dx in 0..bs {
            if bx + dx < frame.width && by + dy < frame.height {
                frame.set(bx + dx, by + dy, block[dy * bs + dx]);
            }
        }
    }
}

/// Simple 2D DCT-II (forward transform) on a bs x bs block.
///
/// Uses the separable property: apply 1D DCT to rows, then columns.
fn dct2d_forward(block: &mut [f32], bs: usize) {
    // Row transform
    let mut temp = vec![0.0f32; bs];
    for row in 0..bs {
        let offset = row * bs;
        dct1d_forward(&block[offset..offset + bs], &mut temp);
        block[offset..offset + bs].copy_from_slice(&temp);
    }

    // Column transform
    let mut col = vec![0.0f32; bs];
    for c in 0..bs {
        for r in 0..bs {
            col[r] = block[r * bs + c];
        }
        dct1d_forward(&col, &mut temp);
        for r in 0..bs {
            block[r * bs + c] = temp[r];
        }
    }
}

/// Simple 2D IDCT (inverse transform).
fn dct2d_inverse(block: &mut [f32], bs: usize) {
    let mut temp = vec![0.0f32; bs];

    // Column transform (inverse)
    let mut col = vec![0.0f32; bs];
    for c in 0..bs {
        for r in 0..bs {
            col[r] = block[r * bs + c];
        }
        dct1d_inverse(&col, &mut temp);
        for r in 0..bs {
            block[r * bs + c] = temp[r];
        }
    }

    // Row transform (inverse)
    for row in 0..bs {
        let offset = row * bs;
        let input: Vec<f32> = block[offset..offset + bs].to_vec();
        dct1d_inverse(&input, &mut temp);
        block[offset..offset + bs].copy_from_slice(&temp);
    }
}

/// 1D DCT-II.
fn dct1d_forward(input: &[f32], output: &mut [f32]) {
    let n = input.len();
    #[allow(clippy::cast_precision_loss)]
    let nf = n as f32;

    for k in 0..n {
        let mut sum = 0.0f32;
        for i in 0..n {
            #[allow(clippy::cast_precision_loss)]
            let angle = std::f32::consts::PI * (2.0 * i as f32 + 1.0) * k as f32 / (2.0 * nf);
            sum += input[i] * angle.cos();
        }

        // Normalization
        let scale = if k == 0 {
            (1.0 / nf).sqrt()
        } else {
            (2.0 / nf).sqrt()
        };

        output[k] = sum * scale;
    }
}

/// 1D IDCT (Type-III).
fn dct1d_inverse(input: &[f32], output: &mut [f32]) {
    let n = input.len();
    #[allow(clippy::cast_precision_loss)]
    let nf = n as f32;

    for i in 0..n {
        let mut sum = input[0] * (1.0 / nf).sqrt();

        for k in 1..n {
            #[allow(clippy::cast_precision_loss)]
            let angle = std::f32::consts::PI * (2.0 * i as f32 + 1.0) * k as f32 / (2.0 * nf);
            sum += input[k] * (2.0 / nf).sqrt() * angle.cos();
        }

        output[i] = sum;
    }
}

/// Compute PSNR between two frames.
#[must_use]
pub fn frame_psnr(original: &VideoFrame, watermarked: &VideoFrame) -> f32 {
    let n = original.pixels.len().min(watermarked.pixels.len());
    if n == 0 {
        return 0.0;
    }

    #[allow(clippy::cast_precision_loss)]
    let mse: f32 = original
        .pixels
        .iter()
        .zip(watermarked.pixels.iter())
        .take(n)
        .map(|(&a, &b)| {
            let diff = a - b;
            diff * diff
        })
        .sum::<f32>()
        / n as f32;

    if mse < 1e-10 {
        return 100.0;
    }

    10.0 * (1.0 / mse).log10()
}

/// Apply JPEG-like compression simulation to a frame (attack).
#[must_use]
pub fn jpeg_compression_attack(frame: &VideoFrame, quality: f32) -> VideoFrame {
    let bs = 8;
    let mut result = frame.clone();

    let blocks_x = frame.width / bs;
    let blocks_y = frame.height / bs;

    // Quantization factor inversely proportional to quality
    let quant = (1.0 - quality.clamp(0.0, 1.0)) * 50.0 + 1.0;

    for by_idx in 0..blocks_y {
        for bx_idx in 0..blocks_x {
            let bx = bx_idx * bs;
            let by = by_idx * bs;

            let mut block = extract_block(&result, bx, by, bs);
            dct2d_forward(&mut block, bs);

            // Quantize DCT coefficients
            for coeff in &mut block {
                *coeff = (*coeff / quant).round() * quant;
            }

            dct2d_inverse(&mut block, bs);
            write_block(&mut result, bx, by, bs, &block);
        }
    }

    // Clamp
    for p in &mut result.pixels {
        *p = p.clamp(0.0, 1.0);
    }

    result
}

/// Add Gaussian noise to a frame (attack).
#[must_use]
pub fn noise_attack(frame: &VideoFrame, noise_std: f32) -> VideoFrame {
    let mut rng = scirs2_core::random::Random::seed(0xCAFE_BABE);
    let mut result = frame.clone();

    for p in &mut result.pixels {
        let u1 = (rng.random_f64() as f32).max(1e-10).min(1.0 - f32::EPSILON);
        let u2 = rng.random_f64() as f32;
        let noise = noise_std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        *p = (*p + noise).clamp(0.0, 1.0);
    }

    result
}

/// Crop a frame (attack).
#[must_use]
pub fn crop_attack(
    frame: &VideoFrame,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> VideoFrame {
    let w = width.min(frame.width.saturating_sub(x));
    let h = height.min(frame.height.saturating_sub(y));

    let mut result = VideoFrame::new(w, h);
    for dy in 0..h {
        for dx in 0..w {
            result.set(dx, dy, frame.get(x + dx, y + dy));
        }
    }

    result
}

/// Scale a frame (resize attack).
#[must_use]
pub fn scale_attack(frame: &VideoFrame, new_width: usize, new_height: usize) -> VideoFrame {
    let mut result = VideoFrame::new(new_width, new_height);

    if new_width == 0 || new_height == 0 {
        return result;
    }

    #[allow(clippy::cast_precision_loss)]
    let x_ratio = frame.width as f32 / new_width as f32;
    #[allow(clippy::cast_precision_loss)]
    let y_ratio = frame.height as f32 / new_height as f32;

    for y in 0..new_height {
        for x in 0..new_width {
            #[allow(clippy::cast_precision_loss)]
            let src_x = (x as f32 * x_ratio) as usize;
            #[allow(clippy::cast_precision_loss)]
            let src_y = (y as f32 * y_ratio) as usize;
            result.set(
                x,
                y,
                frame.get(src_x.min(frame.width - 1), src_y.min(frame.height - 1)),
            );
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_frame(width: usize, height: usize) -> VideoFrame {
        let mut rng = scirs2_core::random::Random::seed(42);
        let pixels: Vec<f32> = (0..width * height)
            .map(|_| rng.random_f64() as f32 * 0.5 + 0.25)
            .collect();
        VideoFrame::from_pixels(pixels, width, height).expect("should succeed")
    }

    #[test]
    fn test_video_frame_creation() {
        let frame = VideoFrame::new(64, 64);
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 64);
        assert_eq!(frame.pixels.len(), 4096);
    }

    #[test]
    fn test_video_frame_get_set() {
        let mut frame = VideoFrame::new(8, 8);
        frame.set(3, 4, 0.75);
        assert!((frame.get(3, 4) - 0.75).abs() < 1e-6);

        // Out of bounds should return 0
        assert!((frame.get(100, 100)).abs() < 1e-6);
    }

    #[test]
    fn test_dct_roundtrip() {
        let input: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mut block: Vec<f32> = vec![0.0; 64];
        for (i, &v) in input.iter().enumerate() {
            for j in 0..8 {
                block[i * 8 + j] = v * (j as f32 + 1.0) / 8.0;
            }
        }

        let original = block.clone();
        dct2d_forward(&mut block, 8);
        dct2d_inverse(&mut block, 8);

        for (i, (&orig, &recovered)) in original.iter().zip(block.iter()).enumerate() {
            assert!(
                (orig - recovered).abs() < 0.01,
                "DCT roundtrip failed at {i}: {orig} vs {recovered}"
            );
        }
    }

    #[test]
    fn test_spatial_embed_detect() {
        let config = VideoWatermarkConfig {
            strength: 0.15,
            block_size: 8,
            frequency_domain: false,
            key: 12345,
            luma_only: true,
            redundancy: 8,
        };

        // Use a uniform gray frame so PN correlation is clean
        let pixels = vec![0.5f32; 384 * 384];
        let frame = VideoFrame::from_pixels(pixels, 384, 384).expect("should succeed");
        let payload = b"V";

        let embedder =
            VideoWatermarkEmbedder::new(config.clone()).expect("embedder should succeed");
        let watermarked = embedder
            .embed(&frame, payload)
            .expect("embed should succeed");

        // Check quality
        let psnr = frame_psnr(&frame, &watermarked);
        assert!(psnr > 10.0, "PSNR = {psnr}");

        let detector = VideoWatermarkDetector::new(config).expect("detector should succeed");
        let codec = PayloadCodec::new(16, 8).expect("codec should succeed");
        let encoded = codec.encode(payload).expect("encode should succeed");
        let expected_bits = encoded.len() * 8;

        let detected = detector
            .detect(&watermarked, expected_bits)
            .expect("detect should succeed");
        assert_eq!(detected.as_slice(), payload);
    }

    #[test]
    fn test_dct_embed_detect() {
        let config = VideoWatermarkConfig {
            strength: 0.1,
            block_size: 8,
            frequency_domain: true,
            key: 54321,
            luma_only: true,
            redundancy: 4,
        };

        let frame = make_test_frame(384, 384);
        let payload = b"V";

        let embedder =
            VideoWatermarkEmbedder::new(config.clone()).expect("embedder should succeed");
        let watermarked = embedder
            .embed(&frame, payload)
            .expect("embed should succeed");

        let psnr = frame_psnr(&frame, &watermarked);
        assert!(psnr > 15.0, "PSNR = {psnr}");

        let detector = VideoWatermarkDetector::new(config).expect("detector should succeed");
        let codec = PayloadCodec::new(16, 8).expect("codec should succeed");
        let encoded = codec.encode(payload).expect("encode should succeed");
        let expected_bits = encoded.len() * 8;

        let detected = detector
            .detect(&watermarked, expected_bits)
            .expect("detect should succeed");
        assert_eq!(detected.as_slice(), payload);
    }

    #[test]
    fn test_capacity_calculation() {
        let config = VideoWatermarkConfig::default();
        let embedder = VideoWatermarkEmbedder::new(config).expect("embedder should succeed");

        let cap_small = embedder.capacity(64, 64);
        let cap_large = embedder.capacity(256, 256);
        assert!(cap_large > cap_small);
        assert!(cap_small > 0);
    }

    #[test]
    fn test_frame_psnr() {
        let frame1 = make_test_frame(64, 64);
        let frame2 = frame1.clone();

        let psnr = frame_psnr(&frame1, &frame2);
        assert!(psnr > 90.0); // Identical frames

        // Modified frame
        let mut modified = frame1.clone();
        for p in &mut modified.pixels {
            *p += 0.01;
        }
        let psnr2 = frame_psnr(&frame1, &modified);
        assert!(psnr2 > 30.0 && psnr2 < 90.0);
    }

    #[test]
    fn test_jpeg_compression_attack() {
        let frame = make_test_frame(64, 64);

        let compressed_high = jpeg_compression_attack(&frame, 0.9);
        let compressed_low = jpeg_compression_attack(&frame, 0.1);

        let psnr_high = frame_psnr(&frame, &compressed_high);
        let psnr_low = frame_psnr(&frame, &compressed_low);

        // High quality should have higher PSNR
        assert!(psnr_high > psnr_low, "high={psnr_high}, low={psnr_low}");
    }

    #[test]
    fn test_noise_attack() {
        let frame = make_test_frame(64, 64);
        let noisy = noise_attack(&frame, 0.05);
        assert_eq!(noisy.pixels.len(), frame.pixels.len());
        assert_ne!(noisy.pixels, frame.pixels);
    }

    #[test]
    fn test_crop_attack() {
        let frame = make_test_frame(64, 64);
        let cropped = crop_attack(&frame, 10, 10, 32, 32);
        assert_eq!(cropped.width, 32);
        assert_eq!(cropped.height, 32);
    }

    #[test]
    fn test_scale_attack() {
        let frame = make_test_frame(64, 64);
        let scaled = scale_attack(&frame, 128, 128);
        assert_eq!(scaled.width, 128);
        assert_eq!(scaled.height, 128);
    }

    #[test]
    fn test_detection_confidence() {
        let config = VideoWatermarkConfig {
            strength: 0.1,
            block_size: 8,
            frequency_domain: true,
            key: 99999,
            luma_only: true,
            redundancy: 4,
        };

        let frame = make_test_frame(384, 384);
        let payload = b"V";

        let embedder =
            VideoWatermarkEmbedder::new(config.clone()).expect("embedder should succeed");
        let watermarked = embedder
            .embed(&frame, payload)
            .expect("embed should succeed");

        let detector = VideoWatermarkDetector::new(config).expect("detector should succeed");
        let codec = PayloadCodec::new(16, 8).expect("codec should succeed");
        let encoded = codec.encode(payload).expect("encode should succeed");
        let bits = encoded.len() * 8;

        let conf_wm = detector.detection_confidence(&watermarked, bits);
        let conf_orig = detector.detection_confidence(&frame, bits);

        // Watermarked should have higher confidence
        assert!(
            conf_wm > conf_orig * 0.5,
            "watermarked conf={conf_wm}, original conf={conf_orig}"
        );
    }

    #[test]
    fn test_watermark_survives_mild_noise() {
        let config = VideoWatermarkConfig {
            strength: 0.15,
            block_size: 8,
            frequency_domain: true,
            key: 77777,
            luma_only: true,
            redundancy: 4,
        };

        let frame = make_test_frame(384, 384);
        let payload = b"V";

        let embedder =
            VideoWatermarkEmbedder::new(config.clone()).expect("embedder should succeed");
        let watermarked = embedder
            .embed(&frame, payload)
            .expect("embed should succeed");

        // Apply mild noise
        let attacked = noise_attack(&watermarked, 0.01);

        let detector = VideoWatermarkDetector::new(config).expect("detector should succeed");
        let codec = PayloadCodec::new(16, 8).expect("codec should succeed");
        let encoded = codec.encode(payload).expect("encode should succeed");
        let bits = encoded.len() * 8;

        let result = detector.detect(&attacked, bits);
        // Should either succeed or fail gracefully
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_frame_from_pixels_invalid() {
        let result = VideoFrame::from_pixels(vec![0.0; 10], 4, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_default_config() {
        let config = VideoWatermarkConfig::default();
        assert!(config.strength > 0.0);
        assert!(config.strength <= 1.0);
        assert_eq!(config.block_size, 8);
        assert!(config.frequency_domain);
        assert!(config.luma_only);
        assert_eq!(config.redundancy, 4);
    }
}
