//! DWT-based robust image watermarking.
//!
//! Embeds a binary watermark payload into an image's luminance channel
//! using a single-level Haar DWT decomposition.  The watermark bits are
//! embedded in the LL (approximation) sub-band coefficients using QIM.
//!
//! ## Algorithm Overview
//!
//! 1. Convert RGB image to YCbCr; work on the Y (luma) channel.
//! 2. Apply Haar DWT: split image into 2×2 blocks → LL, LH, HL, HH sub-bands.
//! 3. Embed payload bits in LL sub-band coefficients using QIM with
//!    quantisation step `delta`.
//! 4. Apply inverse DWT.
//! 5. Convert back to RGB.
//!
//! Detection runs the same DWT and reads quantisation decisions from LL
//! coefficients.
//!
//! ## Image Format
//!
//! Images are represented as flat `Vec<u8>` in row-major order with 3 bytes
//! per pixel (R, G, B).  Width and height must both be even.

use crate::error::{WatermarkError, WatermarkResult};
use crate::payload::{pack_bits, unpack_bits, PayloadCodec};

// ---------------------------------------------------------------------------
// ImageBuffer
// ---------------------------------------------------------------------------

/// A flat RGB image buffer.
#[derive(Debug, Clone)]
pub struct ImageBuffer {
    /// Raw pixel data: [R0, G0, B0, R1, G1, B1, …] in row-major order.
    pub data: Vec<u8>,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
}

impl ImageBuffer {
    /// Create a new image buffer.
    ///
    /// # Errors
    ///
    /// Returns error if `data.len() != width * height * 3`.
    pub fn new(data: Vec<u8>, width: usize, height: usize) -> WatermarkResult<Self> {
        if data.len() != width * height * 3 {
            return Err(WatermarkError::InvalidData(format!(
                "Expected {} bytes for {}×{} RGB image, got {}",
                width * height * 3,
                width,
                height,
                data.len()
            )));
        }
        Ok(Self {
            data,
            width,
            height,
        })
    }

    /// Get pixel at (x, y) as (R, G, B).
    #[must_use]
    pub fn get_pixel(&self, x: usize, y: usize) -> (u8, u8, u8) {
        let idx = (y * self.width + x) * 3;
        (self.data[idx], self.data[idx + 1], self.data[idx + 2])
    }

    /// Set pixel at (x, y).
    pub fn set_pixel(&mut self, x: usize, y: usize, r: u8, g: u8, b: u8) {
        let idx = (y * self.width + x) * 3;
        self.data[idx] = r;
        self.data[idx + 1] = g;
        self.data[idx + 2] = b;
    }
}

// ---------------------------------------------------------------------------
// DwtWatermarkConfig
// ---------------------------------------------------------------------------

/// Configuration for DWT-based image watermarking.
#[derive(Debug, Clone)]
pub struct DwtWatermarkConfig {
    /// QIM quantisation step size for LL coefficients.
    pub delta: f32,
    /// Watermark strength multiplier applied to QIM embedding.
    pub strength: f32,
    /// Secret key for randomising coefficient selection order.
    pub key: u64,
}

impl Default for DwtWatermarkConfig {
    fn default() -> Self {
        Self {
            delta: 8.0,
            strength: 1.0,
            key: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// DwtImageEmbedder
// ---------------------------------------------------------------------------

/// Embeds a payload into an image using Haar DWT and QIM.
pub struct DwtImageEmbedder {
    config: DwtWatermarkConfig,
    codec: PayloadCodec,
}

impl DwtImageEmbedder {
    /// Create a new embedder.
    ///
    /// # Errors
    ///
    /// Returns error if codec initialisation fails.
    pub fn new(config: DwtWatermarkConfig) -> WatermarkResult<Self> {
        let codec = PayloadCodec::new(16, 8)?;
        Ok(Self { config, codec })
    }

    /// Embed `payload` into `image`.
    ///
    /// Both `width` and `height` must be even.
    ///
    /// # Errors
    ///
    /// Returns [`WatermarkError::InsufficientCapacity`] if the image is too
    /// small, or [`WatermarkError::InvalidParameter`] if dimensions are odd.
    pub fn embed(&self, image: &ImageBuffer, payload: &[u8]) -> WatermarkResult<ImageBuffer> {
        if image.width % 2 != 0 || image.height % 2 != 0 {
            return Err(WatermarkError::InvalidParameter(
                "Image dimensions must be even for Haar DWT".to_string(),
            ));
        }

        let encoded = self.codec.encode(payload)?;
        let bits = unpack_bits(&encoded, encoded.len() * 8);

        // Extract luma channel.
        let mut luma = rgb_to_luma(&image.data, image.width, image.height);

        // Forward Haar DWT → get LL sub-band.
        let (mut ll, lh, hl, hh) = haar_dwt_2d(&luma, image.width, image.height);
        let ll_w = image.width / 2;
        let ll_h = image.height / 2;
        let capacity = ll_w * ll_h;

        if bits.len() > capacity {
            return Err(WatermarkError::InsufficientCapacity {
                needed: bits.len(),
                have: capacity,
            });
        }

        // Select coefficient indices in pseudo-random order using key.
        let indices = shuffled_indices(ll.len(), self.config.key);

        // QIM embedding.
        let delta = self.config.delta * self.config.strength;
        for (bit_idx, &bit) in bits.iter().enumerate() {
            let coeff_idx = indices[bit_idx];
            ll[coeff_idx] = qim_quantize(ll[coeff_idx], bit, delta);
        }

        // Inverse Haar DWT.
        luma = haar_idwt_2d(&ll, &lh, &hl, &hh, image.width, image.height);

        // Compose result image.
        let result_data = luma_to_rgb(&image.data, &luma, image.width, image.height);
        ImageBuffer::new(result_data, image.width, image.height)
    }

    /// Capacity in bits for an image of given dimensions.
    #[must_use]
    pub fn capacity(&self, width: usize, height: usize) -> usize {
        (width / 2) * (height / 2)
    }
}

// ---------------------------------------------------------------------------
// DwtImageDetector
// ---------------------------------------------------------------------------

/// Detects a DWT-embedded payload from an image.
pub struct DwtImageDetector {
    config: DwtWatermarkConfig,
    codec: PayloadCodec,
}

impl DwtImageDetector {
    /// Create a new detector.
    ///
    /// # Errors
    ///
    /// Returns error if codec initialisation fails.
    pub fn new(config: DwtWatermarkConfig) -> WatermarkResult<Self> {
        let codec = PayloadCodec::new(16, 8)?;
        Ok(Self { config, codec })
    }

    /// Detect and decode the payload from `image`.
    ///
    /// `expected_bits` must match the number of encoded bits used during
    /// embedding.
    ///
    /// # Errors
    ///
    /// Returns error if sync pattern not found or CRC fails.
    pub fn detect(&self, image: &ImageBuffer, expected_bits: usize) -> WatermarkResult<Vec<u8>> {
        let luma = rgb_to_luma(&image.data, image.width, image.height);
        let (ll, _, _, _) = haar_dwt_2d(&luma, image.width, image.height);

        let indices = shuffled_indices(ll.len(), self.config.key);
        let delta = self.config.delta * self.config.strength;

        let mut bits = Vec::with_capacity(expected_bits);
        for i in 0..expected_bits.min(ll.len()) {
            let coeff_idx = indices[i];
            bits.push(qim_detect(ll[coeff_idx], delta));
        }

        let bytes = pack_bits(&bits);
        self.codec.decode(&bytes)
    }
}

// ---------------------------------------------------------------------------
// Haar DWT (2D single-level)
// ---------------------------------------------------------------------------

/// Forward 2D Haar DWT.
///
/// Returns (LL, LH, HL, HH) sub-bands each of size `(w/2) × (h/2)`.
fn haar_dwt_2d(data: &[f32], w: usize, h: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let hw = w / 2;
    let hh = h / 2;
    let n = hw * hh;
    let mut ll = vec![0.0f32; n];
    let mut lh = vec![0.0f32; n];
    let mut hl = vec![0.0f32; n];
    let mut hh_band = vec![0.0f32; n];

    for row in 0..hh {
        for col in 0..hw {
            let r = row * 2;
            let c = col * 2;

            let a = data[r * w + c];
            let b = data[r * w + c + 1];
            let c_ = data[(r + 1) * w + c];
            let d = data[(r + 1) * w + c + 1];

            let idx = row * hw + col;
            ll[idx] = (a + b + c_ + d) * 0.5;
            lh[idx] = (a - b + c_ - d) * 0.5;
            hl[idx] = (a + b - c_ - d) * 0.5;
            hh_band[idx] = (a - b - c_ + d) * 0.5;
        }
    }

    (ll, lh, hl, hh_band)
}

/// Inverse 2D Haar DWT.
fn haar_idwt_2d(ll: &[f32], lh: &[f32], hl: &[f32], hh: &[f32], w: usize, h: usize) -> Vec<f32> {
    let hw = w / 2;
    let mut out = vec![0.0f32; w * h];

    for row in 0..h / 2 {
        for col in 0..hw {
            let idx = row * hw + col;
            let l = ll[idx];
            let lh_ = lh[idx];
            let hl_ = hl[idx];
            let hh_ = hh[idx];

            let r = row * 2;
            let c = col * 2;
            out[r * w + c] = (l + lh_ + hl_ + hh_) * 0.5;
            out[r * w + c + 1] = (l - lh_ + hl_ - hh_) * 0.5;
            out[(r + 1) * w + c] = (l + lh_ - hl_ - hh_) * 0.5;
            out[(r + 1) * w + c + 1] = (l - lh_ - hl_ + hh_) * 0.5;
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Colour space helpers
// ---------------------------------------------------------------------------

/// Extract luma channel from packed RGB bytes (Rec. 601 coefficients).
fn rgb_to_luma(data: &[u8], w: usize, h: usize) -> Vec<f32> {
    (0..w * h)
        .map(|i| {
            let r = data[i * 3] as f32;
            let g = data[i * 3 + 1] as f32;
            let b = data[i * 3 + 2] as f32;
            0.299 * r + 0.587 * g + 0.114 * b
        })
        .collect()
}

/// Substitute modified luma back into the original RGB image.
fn luma_to_rgb(original: &[u8], new_luma: &[f32], w: usize, h: usize) -> Vec<u8> {
    let mut out = original.to_vec();
    for i in 0..w * h {
        let old_y = 0.299 * original[i * 3] as f32
            + 0.587 * original[i * 3 + 1] as f32
            + 0.114 * original[i * 3 + 2] as f32;
        let delta_y = new_luma[i] - old_y;

        out[i * 3] = (original[i * 3] as f32 + delta_y).clamp(0.0, 255.0) as u8;
        out[i * 3 + 1] = (original[i * 3 + 1] as f32 + delta_y).clamp(0.0, 255.0) as u8;
        out[i * 3 + 2] = (original[i * 3 + 2] as f32 + delta_y).clamp(0.0, 255.0) as u8;
    }
    out
}

// ---------------------------------------------------------------------------
// QIM helpers
// ---------------------------------------------------------------------------

/// QIM quantise a coefficient for bit `b` with step `delta`.
fn qim_quantize(value: f32, bit: bool, delta: f32) -> f32 {
    let offset = if bit { delta * 0.5 } else { 0.0 };
    ((value - offset) / delta).round() * delta + offset
}

/// QIM detect: which quantiser does `value` belong to?
fn qim_detect(value: f32, delta: f32) -> bool {
    let dist0 = (value - (value / delta).round() * delta).abs();
    let v1 = value - delta * 0.5;
    let dist1 = (v1 - (v1 / delta).round() * delta).abs();
    dist1 < dist0
}

// ---------------------------------------------------------------------------
// Index shuffler (Fisher-Yates with xorshift64)
// ---------------------------------------------------------------------------

fn shuffled_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    if n <= 1 {
        return indices;
    }
    let mut state = if seed == 0 {
        0xDEAD_BEEF_CAFE_1234u64
    } else {
        seed
    };
    for i in (1..n).rev() {
        state = xorshift64(state);
        let j = (state as usize) % (i + 1);
        indices.swap(i, j);
    }
    indices
}

fn xorshift64(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::payload::PayloadCodec;

    fn make_image(w: usize, h: usize, fill: u8) -> ImageBuffer {
        let data = vec![fill; w * h * 3];
        ImageBuffer::new(data, w, h).expect("make_image buffer should be valid")
    }

    fn gradient_image(w: usize, h: usize) -> ImageBuffer {
        let mut data = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                data[i] = ((x * 255) / w) as u8;
                data[i + 1] = ((y * 255) / h) as u8;
                data[i + 2] = 128;
            }
        }
        ImageBuffer::new(data, w, h).expect("gradient_image buffer should be valid")
    }

    #[test]
    fn test_image_buffer_pixel_access() {
        let mut img = make_image(4, 4, 100);
        img.set_pixel(1, 2, 10, 20, 30);
        let (r, g, b) = img.get_pixel(1, 2);
        assert_eq!((r, g, b), (10, 20, 30));
    }

    #[test]
    fn test_image_buffer_wrong_size_error() {
        let result = ImageBuffer::new(vec![0u8; 10], 4, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_capacity_calculation() {
        let embedder = DwtImageEmbedder::new(DwtWatermarkConfig::default())
            .expect("default embedder creation should succeed");
        // 256×256 image → LL sub-band is 128×128 = 16384 coefficients
        assert_eq!(embedder.capacity(256, 256), 16384);
    }

    #[test]
    fn test_embed_preserves_dimensions() {
        let img = gradient_image(64, 64);
        let embedder = DwtImageEmbedder::new(DwtWatermarkConfig::default())
            .expect("default embedder creation should succeed");
        let watermarked = embedder
            .embed(&img, b"IMG")
            .expect("embedding IMG should succeed");
        assert_eq!(watermarked.width, 64);
        assert_eq!(watermarked.height, 64);
        assert_eq!(watermarked.data.len(), 64 * 64 * 3);
    }

    #[test]
    fn test_embed_roundtrip_basic() {
        let img = gradient_image(128, 128);
        let config = DwtWatermarkConfig {
            delta: 8.0,
            strength: 1.0,
            key: 0xABCD1234,
        };
        let embedder =
            DwtImageEmbedder::new(config.clone()).expect("embedder creation should succeed");
        let detector = DwtImageDetector::new(config).expect("detector creation should succeed");

        let payload = b"W";
        let codec = PayloadCodec::new(16, 8).expect("codec creation should succeed");
        let encoded = codec
            .encode(payload)
            .expect("payload encoding should succeed");
        let expected_bits = encoded.len() * 8;

        let watermarked = embedder
            .embed(&img, payload)
            .expect("embedding should succeed");
        let extracted = detector
            .detect(&watermarked, expected_bits)
            .expect("detection should succeed");
        assert_eq!(extracted, payload.as_slice());
    }

    #[test]
    fn test_embed_odd_dimensions_returns_error() {
        let data = vec![128u8; 3 * 3 * 3];
        let img = ImageBuffer::new(data, 3, 3).expect("3x3 image buffer should be valid");
        let embedder = DwtImageEmbedder::new(DwtWatermarkConfig::default())
            .expect("default embedder should succeed");
        let result = embedder.embed(&img, b"test");
        assert!(result.is_err());
    }

    #[test]
    fn test_embed_insufficient_capacity_error() {
        // 4×4 image has LL sub-band of 2×2 = 4 coefficients, far too small
        // for a full RS-encoded payload (280 bits).
        let img = gradient_image(4, 4);
        let embedder = DwtImageEmbedder::new(DwtWatermarkConfig::default())
            .expect("default embedder should succeed");
        let result = embedder.embed(&img, b"toolong");
        assert!(result.is_err());
    }

    #[test]
    fn test_haar_dwt_idwt_roundtrip() {
        let w = 8;
        let h = 8;
        let original: Vec<f32> = (0..w * h).map(|i| i as f32).collect();
        let (ll, lh, hl, hh) = haar_dwt_2d(&original, w, h);
        let reconstructed = haar_idwt_2d(&ll, &lh, &hl, &hh, w, h);

        for (o, r) in original.iter().zip(reconstructed.iter()) {
            assert!(
                (o - r).abs() < 1e-4,
                "DWT roundtrip error at: orig={o}, recon={r}"
            );
        }
    }

    #[test]
    fn test_qim_quantize_detect_zero() {
        let delta = 8.0;
        let v = 12.3f32;
        let q0 = qim_quantize(v, false, delta);
        assert!(!qim_detect(q0, delta), "bit-0 should be detected as false");
    }

    #[test]
    fn test_qim_quantize_detect_one() {
        let delta = 8.0;
        let v = 12.3f32;
        let q1 = qim_quantize(v, true, delta);
        assert!(qim_detect(q1, delta), "bit-1 should be detected as true");
    }

    #[test]
    fn test_shuffled_indices_length() {
        let idx = shuffled_indices(100, 42);
        assert_eq!(idx.len(), 100);
    }

    #[test]
    fn test_shuffled_indices_is_permutation() {
        let n = 50;
        let mut idx = shuffled_indices(n, 99);
        idx.sort_unstable();
        assert_eq!(idx, (0..n).collect::<Vec<_>>());
    }

    #[test]
    fn test_shuffled_indices_deterministic() {
        let idx1 = shuffled_indices(100, 42);
        let idx2 = shuffled_indices(100, 42);
        assert_eq!(idx1, idx2);
    }

    #[test]
    fn test_embed_changes_image() {
        let img = gradient_image(64, 64);
        let embedder = DwtImageEmbedder::new(DwtWatermarkConfig::default())
            .expect("default embedder should succeed");
        let watermarked = embedder
            .embed(&img, b"A")
            .expect("embedding A should succeed");
        // At least some pixels should differ.
        assert_ne!(img.data, watermarked.data);
    }
}
