//! Transform operations (DCT, FFT, geometric transforms)

use crate::{GpuDevice, GpuError, Result};
use oxifft::Complex;

/// Transform operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformType {
    /// Discrete Cosine Transform (DCT)
    DCT,
    /// Inverse DCT
    IDCT,
    /// Fast Fourier Transform (FFT)
    FFT,
    /// Inverse FFT
    IFFT,
    /// Rotate 90 degrees
    Rotate90,
    /// Rotate 180 degrees
    Rotate180,
    /// Rotate 270 degrees
    Rotate270,
    /// Flip horizontal
    FlipHorizontal,
    /// Flip vertical
    FlipVertical,
    /// Transpose
    Transpose,
    /// Affine transform
    Affine,
    /// Perspective transform
    Perspective,
}

/// Transform kernel for frequency domain and geometric operations
pub struct TransformKernel {
    transform_type: TransformType,
}

impl TransformKernel {
    /// Create a new transform kernel
    #[must_use]
    pub fn new(transform_type: TransformType) -> Self {
        Self { transform_type }
    }

    /// Create a DCT transform kernel
    #[must_use]
    pub fn dct() -> Self {
        Self::new(TransformType::DCT)
    }

    /// Create an IDCT transform kernel
    #[must_use]
    pub fn idct() -> Self {
        Self::new(TransformType::IDCT)
    }

    /// Create a rotate kernel
    #[must_use]
    pub fn rotate(degrees: i32) -> Self {
        let transform_type = match degrees % 360 {
            90 | -270 => TransformType::Rotate90,
            180 | -180 => TransformType::Rotate180,
            270 | -90 => TransformType::Rotate270,
            _ => TransformType::Rotate90, // Default
        };
        Self::new(transform_type)
    }

    /// Create a flip kernel
    #[must_use]
    pub fn flip(horizontal: bool) -> Self {
        let transform_type = if horizontal {
            TransformType::FlipHorizontal
        } else {
            TransformType::FlipVertical
        };
        Self::new(transform_type)
    }

    /// Execute the transform operation (frequency-domain, f32 data).
    ///
    /// Handles DCT and IDCT which operate on `f32` frequency-domain data.
    /// For pixel-level geometric transforms (rotate, flip, transpose) use
    /// [`TransformKernel::execute_u8`] instead.
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `input` - Input data buffer
    /// * `output` - Output data buffer
    /// * `width` - Data width
    /// * `height` - Data height
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails or is not supported for f32 data.
    pub fn execute(
        &self,
        device: &GpuDevice,
        input: &[f32],
        output: &mut [f32],
        width: u32,
        height: u32,
    ) -> Result<()> {
        match self.transform_type {
            TransformType::DCT => {
                crate::ops::TransformOperation::dct_2d(device, input, output, width, height)
            }
            TransformType::IDCT => {
                crate::ops::TransformOperation::idct_2d(device, input, output, width, height)
            }
            // FFT/IFFT: use execute_fft_f32 / execute_ifft_f32 directly.
            // Affine/Perspective: matrix parameters cannot be passed through the
            // unit enum variant — use execute_affine_f32 / execute_perspective_f32
            // directly. These arms return NotSupported to preserve the API contract.
            TransformType::FFT => self.execute_fft_f32(input, output, width, height),
            TransformType::IFFT => self.execute_ifft_f32(input, output, width, height),
            TransformType::Affine => Err(crate::GpuError::NotSupported(
                "Affine requires a matrix — call execute_affine_f32() directly".to_string(),
            )),
            TransformType::Perspective => Err(crate::GpuError::NotSupported(
                "Perspective requires a matrix — call execute_perspective_f32() directly"
                    .to_string(),
            )),
            _ => Err(crate::GpuError::NotSupported(format!(
                "Transform type {:?} requires u8 pixel data — use execute_u8()",
                self.transform_type
            ))),
        }
    }

    /// Execute a geometric pixel transform on an interleaved `u8` image buffer.
    ///
    /// Handles `Rotate90`, `Rotate180`, `Rotate270`, `FlipHorizontal`,
    /// `FlipVertical`, and `Transpose`.  `FFT`, `IFFT`, `Affine`, and
    /// `Perspective` are deliberately left as `NotSupported`.
    ///
    /// The `_device` parameter is accepted for API symmetry but is not used
    /// by the CPU-side implementations (the geometric ops are fully pure-Rust).
    ///
    /// # Arguments
    ///
    /// * `_device` - GPU device (unused; present for API consistency)
    /// * `input` - Input pixel buffer (`width * height * channels` bytes)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `channels` - Bytes per pixel (e.g. 3 for RGB, 4 for RGBA)
    ///
    /// # Errors
    ///
    /// Returns [`crate::GpuError::NotSupported`] for frequency-domain,
    /// `Affine`, and `Perspective` transform types.
    pub fn execute_u8(
        &self,
        _device: &GpuDevice,
        input: &[u8],
        width: u32,
        height: u32,
        channels: u32,
    ) -> Result<Vec<u8>> {
        match self.transform_type {
            TransformType::Rotate90 => Ok(crate::ops::TransformOperation::rotate90(
                input, width, height, channels,
            )),
            TransformType::Rotate180 => Ok(crate::ops::TransformOperation::rotate180(
                input, width, height, channels,
            )),
            TransformType::Rotate270 => Ok(crate::ops::TransformOperation::rotate270(
                input, width, height, channels,
            )),
            TransformType::FlipHorizontal => Ok(crate::ops::TransformOperation::flip_horizontal(
                input, width, height, channels,
            )),
            TransformType::FlipVertical => Ok(crate::ops::TransformOperation::flip_vertical(
                input, width, height, channels,
            )),
            TransformType::Transpose => Ok(crate::ops::TransformOperation::transpose(
                input, width, height, channels,
            )),
            TransformType::FFT | TransformType::IFFT => Err(crate::GpuError::NotSupported(
                "FFT/IFFT operates on f32 data — use execute()".to_string(),
            )),
            // Affine/Perspective u8: matrix parameters cannot be passed through the
            // unit enum variant — use execute_affine_u8 / execute_perspective_u8
            // directly. These arms return NotSupported to preserve the API contract.
            TransformType::Affine => Err(crate::GpuError::NotSupported(
                "Affine requires a matrix — call execute_affine_u8() directly".to_string(),
            )),
            TransformType::Perspective => Err(crate::GpuError::NotSupported(
                "Perspective requires a matrix — call execute_perspective_u8() directly"
                    .to_string(),
            )),
            TransformType::DCT | TransformType::IDCT => {
                Err(crate::GpuError::NotSupported(format!(
                    "Transform type {:?} operates on f32 data — use execute()",
                    self.transform_type
                )))
            }
        }
    }

    /// Get the transform type
    #[must_use]
    pub fn transform_type(&self) -> TransformType {
        self.transform_type
    }

    /// Check if this is a frequency domain transform
    #[must_use]
    pub fn is_frequency_domain(&self) -> bool {
        matches!(
            self.transform_type,
            TransformType::DCT | TransformType::IDCT | TransformType::FFT | TransformType::IFFT
        )
    }

    /// Check if this is a geometric transform
    #[must_use]
    pub fn is_geometric(&self) -> bool {
        matches!(
            self.transform_type,
            TransformType::Rotate90
                | TransformType::Rotate180
                | TransformType::Rotate270
                | TransformType::FlipHorizontal
                | TransformType::FlipVertical
                | TransformType::Transpose
                | TransformType::Affine
                | TransformType::Perspective
        )
    }

    /// Estimate FLOPS for the transform operation
    #[must_use]
    pub fn estimate_flops(width: u32, height: u32, transform_type: TransformType) -> u64 {
        let n = u64::from(width) * u64::from(height);

        match transform_type {
            TransformType::DCT | TransformType::IDCT => {
                // DCT complexity: O(N^2 log N) for 2D
                let log_n = (n as f64).log2().ceil() as u64;
                n * n * log_n
            }
            TransformType::FFT | TransformType::IFFT => {
                // FFT complexity: O(N log N)
                let log_n = (n as f64).log2().ceil() as u64;
                n * log_n * 5 // 5 ops per butterfly
            }
            _ => {
                // Geometric transforms: O(N)
                n
            }
        }
    }

    // -------------------------------------------------------------------------
    // Affine transform — f32 path (CPU fallback, inverse-mapped nearest-neighbour)
    // -------------------------------------------------------------------------

    /// Apply a 2D affine transform to a packed f32 scalar image.
    ///
    /// Matrix layout: `[a, b, c, d, tx, ty]` such that the *forward* mapping
    /// `(x', y') = ([a b; c d] · [x; y]) + [tx; ty]` is inverted before use.
    /// For each output pixel `(ox, oy)` the inverse transform finds the source
    /// coordinate `(sx, sy)` and performs nearest-neighbour sampling (clamped to
    /// border).
    ///
    /// # Arguments
    ///
    /// * `input`  – f32 buffer of `width * height` samples
    /// * `output` – f32 buffer of `width * height` samples (same size as input)
    /// * `width`  – image width in pixels
    /// * `height` – image height in pixels
    /// * `matrix` – forward affine matrix `[a, b, c, d, tx, ty]`
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::InvalidBufferSize`] if buffers are too small, or
    /// [`GpuError::Internal`] if the matrix is singular (det ≈ 0).
    pub fn execute_affine_f32(
        &self,
        input: &[f32],
        output: &mut [f32],
        width: u32,
        height: u32,
        matrix: [f32; 6],
    ) -> Result<()> {
        let expected = (width * height) as usize;
        if input.len() < expected {
            return Err(GpuError::InvalidBufferSize {
                expected,
                actual: input.len(),
            });
        }
        if output.len() < expected {
            return Err(GpuError::InvalidBufferSize {
                expected,
                actual: output.len(),
            });
        }

        // Forward matrix [a, b, c, d, tx, ty]
        let a = matrix[0];
        let b = matrix[1];
        let c = matrix[2];
        let d = matrix[3];
        let tx = matrix[4];
        let ty = matrix[5];

        let det = a * d - b * c;
        if det.abs() < f32::EPSILON {
            return Err(GpuError::Internal("Affine matrix is singular".to_string()));
        }

        // Inverse 2×2 part
        let inv_det = 1.0 / det;
        let ia = d * inv_det;
        let ib = -b * inv_det;
        let ic = -c * inv_det;
        let id = a * inv_det;
        // Inverse translation: inv_t = -M_inv * t
        let itx = -(ia * tx + ib * ty);
        let ity = -(ic * tx + id * ty);

        let w = width as i32;
        let h = height as i32;

        for oy in 0..height {
            for ox in 0..width {
                let fx = ox as f32;
                let fy = oy as f32;
                let sx = ia * fx + ib * fy + itx;
                let sy = ic * fx + id * fy + ity;
                let ix = (sx.floor() as i32).clamp(0, w - 1) as u32;
                let iy = (sy.floor() as i32).clamp(0, h - 1) as u32;
                let out_idx = (oy * width + ox) as usize;
                let in_idx = (iy * width + ix) as usize;
                output[out_idx] = input[in_idx];
            }
        }

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Affine transform — u8 path (delegates to f32)
    // -------------------------------------------------------------------------

    /// Apply a 2D affine transform to a packed `u8` image (any number of
    /// channels per pixel).
    ///
    /// Each pixel is treated as `channels` consecutive bytes. The geometric
    /// mapping is computed in f32 (inverse affine, nearest-neighbour sampling).
    ///
    /// # Arguments
    ///
    /// * `input`    – u8 buffer of `width * height * channels` bytes
    /// * `output`   – u8 buffer of `width * height * channels` bytes
    /// * `width`    – image width in pixels
    /// * `height`   – image height in pixels
    /// * `channels` – bytes per pixel (e.g. 3 for RGB, 4 for RGBA)
    /// * `matrix`   – forward affine matrix `[a, b, c, d, tx, ty]`
    ///
    /// # Errors
    ///
    /// Returns an error if buffers are too small or the matrix is singular.
    pub fn execute_affine_u8(
        &self,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        channels: u32,
        matrix: [f32; 6],
    ) -> Result<()> {
        let expected = (width * height * channels) as usize;
        if input.len() < expected {
            return Err(GpuError::InvalidBufferSize {
                expected,
                actual: input.len(),
            });
        }
        if output.len() < expected {
            return Err(GpuError::InvalidBufferSize {
                expected,
                actual: output.len(),
            });
        }

        let a = matrix[0];
        let b = matrix[1];
        let c = matrix[2];
        let d = matrix[3];
        let tx = matrix[4];
        let ty = matrix[5];

        let det = a * d - b * c;
        if det.abs() < f32::EPSILON {
            return Err(GpuError::Internal("Affine matrix is singular".to_string()));
        }

        let inv_det = 1.0 / det;
        let ia = d * inv_det;
        let ib = -b * inv_det;
        let ic = -c * inv_det;
        let id = a * inv_det;
        let itx = -(ia * tx + ib * ty);
        let ity = -(ic * tx + id * ty);

        let w = width as i32;
        let h = height as i32;
        let ch = channels as usize;

        for oy in 0..height {
            for ox in 0..width {
                let fx = ox as f32;
                let fy = oy as f32;
                let sx = ia * fx + ib * fy + itx;
                let sy = ic * fx + id * fy + ity;
                let ix = (sx.floor() as i32).clamp(0, w - 1) as u32;
                let iy = (sy.floor() as i32).clamp(0, h - 1) as u32;
                let out_off = ((oy * width + ox) as usize) * ch;
                let in_off = ((iy * width + ix) as usize) * ch;
                output[out_off..out_off + ch].copy_from_slice(&input[in_off..in_off + ch]);
            }
        }

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Perspective transform — f32 path (CPU, inverse-mapped nearest-neighbour)
    // -------------------------------------------------------------------------

    /// Apply a 2D perspective (homography) transform to a packed f32 scalar image.
    ///
    /// `matrix` is a row-major 3×3 homography `H` stored as 9 f32 values.  For
    /// each output pixel `(ox, oy)` the inverse `H⁻¹` maps it back to the source
    /// coordinate `(sx/w, sy/w)`, which is sampled with clamped nearest-neighbour.
    ///
    /// # Arguments
    ///
    /// * `input`  – f32 buffer of `width * height` samples
    /// * `output` – f32 buffer of `width * height` samples
    /// * `width`  – image width
    /// * `height` – image height
    /// * `matrix` – 3×3 row-major homography `[h00,h01,h02, h10,h11,h12, h20,h21,h22]`
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Internal`] if the matrix is singular.
    pub fn execute_perspective_f32(
        &self,
        input: &[f32],
        output: &mut [f32],
        width: u32,
        height: u32,
        matrix: [f32; 9],
    ) -> Result<()> {
        let expected = (width * height) as usize;
        if input.len() < expected {
            return Err(GpuError::InvalidBufferSize {
                expected,
                actual: input.len(),
            });
        }
        if output.len() < expected {
            return Err(GpuError::InvalidBufferSize {
                expected,
                actual: output.len(),
            });
        }

        // Compute 3×3 inverse using cofactor expansion (f64 for precision).
        let m = matrix.map(|v| v as f64);
        let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6]);

        if det.abs() < 1e-12 {
            return Err(GpuError::Internal(
                "Perspective matrix is singular".to_string(),
            ));
        }

        let inv_det = 1.0 / det;
        let inv: [f64; 9] = [
            (m[4] * m[8] - m[5] * m[7]) * inv_det,
            (m[2] * m[7] - m[1] * m[8]) * inv_det,
            (m[1] * m[5] - m[2] * m[4]) * inv_det,
            (m[5] * m[6] - m[3] * m[8]) * inv_det,
            (m[0] * m[8] - m[2] * m[6]) * inv_det,
            (m[2] * m[3] - m[0] * m[5]) * inv_det,
            (m[3] * m[7] - m[4] * m[6]) * inv_det,
            (m[1] * m[6] - m[0] * m[7]) * inv_det,
            (m[0] * m[4] - m[1] * m[3]) * inv_det,
        ];

        let w = width as i32;
        let h = height as i32;

        for oy in 0..height {
            for ox in 0..width {
                let x = ox as f64;
                let y = oy as f64;
                let xh = inv[0] * x + inv[1] * y + inv[2];
                let yh = inv[3] * x + inv[4] * y + inv[5];
                let wh = inv[6] * x + inv[7] * y + inv[8];
                if wh.abs() < 1e-12 {
                    // Maps to infinity; use border pixel.
                    output[(oy * width + ox) as usize] = input[0];
                    continue;
                }
                let sx = (xh / wh).round() as i32;
                let sy = (yh / wh).round() as i32;
                let ix = sx.clamp(0, w - 1) as u32;
                let iy = sy.clamp(0, h - 1) as u32;
                let out_idx = (oy * width + ox) as usize;
                let in_idx = (iy * width + ix) as usize;
                output[out_idx] = input[in_idx];
            }
        }

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Perspective transform — u8 path
    // -------------------------------------------------------------------------

    /// Apply a 2D perspective transform to a packed u8 image.
    ///
    /// Same geometry as [`Self::execute_perspective_f32`] but operates on multi-channel
    /// u8 pixel data. `channels` is the number of bytes per pixel.
    ///
    /// # Errors
    ///
    /// Returns an error if buffers are too small or the matrix is singular.
    pub fn execute_perspective_u8(
        &self,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        channels: u32,
        matrix: [f32; 9],
    ) -> Result<()> {
        let expected = (width * height * channels) as usize;
        if input.len() < expected {
            return Err(GpuError::InvalidBufferSize {
                expected,
                actual: input.len(),
            });
        }
        if output.len() < expected {
            return Err(GpuError::InvalidBufferSize {
                expected,
                actual: output.len(),
            });
        }

        let m = matrix.map(|v| v as f64);
        let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6]);

        if det.abs() < 1e-12 {
            return Err(GpuError::Internal(
                "Perspective matrix is singular".to_string(),
            ));
        }

        let inv_det = 1.0 / det;
        let inv: [f64; 9] = [
            (m[4] * m[8] - m[5] * m[7]) * inv_det,
            (m[2] * m[7] - m[1] * m[8]) * inv_det,
            (m[1] * m[5] - m[2] * m[4]) * inv_det,
            (m[5] * m[6] - m[3] * m[8]) * inv_det,
            (m[0] * m[8] - m[2] * m[6]) * inv_det,
            (m[2] * m[3] - m[0] * m[5]) * inv_det,
            (m[3] * m[7] - m[4] * m[6]) * inv_det,
            (m[1] * m[6] - m[0] * m[7]) * inv_det,
            (m[0] * m[4] - m[1] * m[3]) * inv_det,
        ];

        let iw = width as i32;
        let ih = height as i32;
        let ch = channels as usize;

        for oy in 0..height {
            for ox in 0..width {
                let x = ox as f64;
                let y = oy as f64;
                let xh = inv[0] * x + inv[1] * y + inv[2];
                let yh = inv[3] * x + inv[4] * y + inv[5];
                let wh = inv[6] * x + inv[7] * y + inv[8];
                let (ix, iy) = if wh.abs() < 1e-12 {
                    (0u32, 0u32)
                } else {
                    let sx = (xh / wh).round() as i32;
                    let sy = (yh / wh).round() as i32;
                    (sx.clamp(0, iw - 1) as u32, sy.clamp(0, ih - 1) as u32)
                };
                let out_off = ((oy * width + ox) as usize) * ch;
                let in_off = ((iy * width + ix) as usize) * ch;
                output[out_off..out_off + ch].copy_from_slice(&input[in_off..in_off + ch]);
            }
        }

        Ok(())
    }

    // -------------------------------------------------------------------------
    // FFT / IFFT — f32 path (CPU, 2D separable via OxiFFT)
    // -------------------------------------------------------------------------

    /// Compute a 2D forward FFT of an f32 scalar image via row-column separation.
    ///
    /// Input samples are treated as real values; adjacent pairs `(input[2k],
    /// input[2k+1])` are **not** used as complex pairs — instead each f32 sample
    /// is promoted to a complex number with imaginary part 0 before the 1D FFT.
    ///
    /// The result is stored interleaved: `output[2*k] = re`, `output[2*k+1] = im`
    /// for each complex output coefficient.  Therefore `output` must have at least
    /// `2 * width * height` elements.
    ///
    /// The 2D FFT is computed as 1D FFT of every row followed by 1D FFT of every
    /// column (separability property).
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::InvalidBufferSize`] if buffers are undersized.
    pub fn execute_fft_f32(
        &self,
        input: &[f32],
        output: &mut [f32],
        width: u32,
        height: u32,
    ) -> Result<()> {
        let n = (width * height) as usize;
        if input.len() < n {
            return Err(GpuError::InvalidBufferSize {
                expected: n,
                actual: input.len(),
            });
        }
        // Output is interleaved complex: need 2*n f32 slots.
        let out_needed = 2 * n;
        if output.len() < out_needed {
            return Err(GpuError::InvalidBufferSize {
                expected: out_needed,
                actual: output.len(),
            });
        }

        let w = width as usize;
        let h = height as usize;

        // Build complex working buffer: real input → complex with im=0.
        let mut work: Vec<Complex<f64>> = input[..n]
            .iter()
            .map(|&v| Complex::new(v as f64, 0.0))
            .collect();

        // 1D FFT of each row.
        for row in 0..h {
            let start = row * w;
            let row_slice: Vec<Complex<f64>> = work[start..start + w].to_vec();
            let row_fft = oxifft::fft(&row_slice);
            work[start..start + w].copy_from_slice(&row_fft);
        }

        // 1D FFT of each column.
        let mut col_buf = vec![Complex::new(0.0f64, 0.0); h];
        for col in 0..w {
            for row in 0..h {
                col_buf[row] = work[row * w + col];
            }
            let col_fft = oxifft::fft(&col_buf);
            for row in 0..h {
                work[row * w + col] = col_fft[row];
            }
        }

        // Store interleaved (re, im) into output.
        for (k, c) in work.iter().enumerate() {
            output[2 * k] = c.re as f32;
            output[2 * k + 1] = c.im as f32;
        }

        Ok(())
    }

    /// Compute a 2D inverse FFT of a packed complex f32 buffer.
    ///
    /// Input format: interleaved complex `input[2*k] = re`, `input[2*k+1] = im`.
    /// Output format: interleaved complex (same layout as [`Self::execute_fft_f32`]).
    ///
    /// The IFFT is computed as: conjugate → forward FFT → conjugate → divide by N.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::InvalidBufferSize`] if buffers are undersized.
    pub fn execute_ifft_f32(
        &self,
        input: &[f32],
        output: &mut [f32],
        width: u32,
        height: u32,
    ) -> Result<()> {
        let n = (width * height) as usize;
        let in_needed = 2 * n;
        if input.len() < in_needed {
            return Err(GpuError::InvalidBufferSize {
                expected: in_needed,
                actual: input.len(),
            });
        }
        let out_needed = 2 * n;
        if output.len() < out_needed {
            return Err(GpuError::InvalidBufferSize {
                expected: out_needed,
                actual: output.len(),
            });
        }

        let w = width as usize;
        let h = height as usize;

        // Read interleaved complex and conjugate.
        let mut work: Vec<Complex<f64>> = (0..n)
            .map(|k| Complex::new(input[2 * k] as f64, -(input[2 * k + 1] as f64)))
            .collect();

        // Row FFTs.
        for row in 0..h {
            let start = row * w;
            let row_slice: Vec<Complex<f64>> = work[start..start + w].to_vec();
            let row_fft = oxifft::fft(&row_slice);
            work[start..start + w].copy_from_slice(&row_fft);
        }

        // Column FFTs.
        let mut col_buf = vec![Complex::new(0.0f64, 0.0); h];
        for col in 0..w {
            for row in 0..h {
                col_buf[row] = work[row * w + col];
            }
            let col_fft = oxifft::fft(&col_buf);
            for row in 0..h {
                work[row * w + col] = col_fft[row];
            }
        }

        // Conjugate and divide by N.
        let scale = 1.0 / n as f64;
        for (k, c) in work.iter().enumerate() {
            output[2 * k] = (c.re * scale) as f32;
            output[2 * k + 1] = (-c.im * scale) as f32;
        }

        Ok(())
    }
}

/// Affine transformation matrix
#[derive(Debug, Clone, Copy)]
pub struct AffineMatrix {
    /// Matrix elements [a, b, c, d, tx, ty]
    /// [ a  b  tx ]
    /// [ c  d  ty ]
    /// [ 0  0  1  ]
    pub elements: [f32; 6],
}

impl AffineMatrix {
    /// Create an identity matrix
    #[must_use]
    pub fn identity() -> Self {
        Self {
            elements: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        }
    }

    /// Create a translation matrix
    #[must_use]
    pub fn translation(tx: f32, ty: f32) -> Self {
        Self {
            elements: [1.0, 0.0, tx, 0.0, 1.0, ty],
        }
    }

    /// Create a rotation matrix
    #[must_use]
    pub fn rotation(angle_radians: f32) -> Self {
        let cos = angle_radians.cos();
        let sin = angle_radians.sin();
        Self {
            elements: [cos, -sin, 0.0, sin, cos, 0.0],
        }
    }

    /// Create a scaling matrix
    #[must_use]
    pub fn scaling(sx: f32, sy: f32) -> Self {
        Self {
            elements: [sx, 0.0, 0.0, 0.0, sy, 0.0],
        }
    }

    /// Combine two affine transformations
    #[must_use]
    pub fn combine(&self, other: &Self) -> Self {
        let a1 = self.elements;
        let a2 = other.elements;

        Self {
            elements: [
                a1[0] * a2[0] + a1[1] * a2[3],
                a1[0] * a2[1] + a1[1] * a2[4],
                a1[0] * a2[2] + a1[1] * a2[5] + a1[2],
                a1[3] * a2[0] + a1[4] * a2[3],
                a1[3] * a2[1] + a1[4] * a2[4],
                a1[3] * a2[2] + a1[4] * a2[5] + a1[5],
            ],
        }
    }

    /// Get matrix elements
    #[must_use]
    pub fn as_array(&self) -> [f32; 6] {
        self.elements
    }
}

impl Default for AffineMatrix {
    fn default() -> Self {
        Self::identity()
    }
}

/// Warp kernel for geometric transformations
pub struct WarpKernel {
    matrix: AffineMatrix,
}

impl WarpKernel {
    /// Create a new warp kernel
    #[must_use]
    pub fn new(matrix: AffineMatrix) -> Self {
        Self { matrix }
    }

    /// Create a rotation warp
    #[must_use]
    pub fn rotation(angle_degrees: f32, center_x: f32, center_y: f32) -> Self {
        let angle_radians = angle_degrees.to_radians();

        // Translate to origin, rotate, translate back
        let t1 = AffineMatrix::translation(-center_x, -center_y);
        let r = AffineMatrix::rotation(angle_radians);
        let t2 = AffineMatrix::translation(center_x, center_y);

        let matrix = t1.combine(&r).combine(&t2);

        Self::new(matrix)
    }

    /// Create a scaling warp
    #[must_use]
    pub fn scaling(sx: f32, sy: f32, center_x: f32, center_y: f32) -> Self {
        let t1 = AffineMatrix::translation(-center_x, -center_y);
        let s = AffineMatrix::scaling(sx, sy);
        let t2 = AffineMatrix::translation(center_x, center_y);

        let matrix = t1.combine(&s).combine(&t2);

        Self::new(matrix)
    }

    /// Get the transformation matrix
    #[must_use]
    pub fn matrix(&self) -> &AffineMatrix {
        &self.matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_kernel_creation() {
        let kernel = TransformKernel::dct();
        assert_eq!(kernel.transform_type(), TransformType::DCT);
        assert!(kernel.is_frequency_domain());
        assert!(!kernel.is_geometric());

        let kernel = TransformKernel::rotate(90);
        assert_eq!(kernel.transform_type(), TransformType::Rotate90);
        assert!(!kernel.is_frequency_domain());
        assert!(kernel.is_geometric());
    }

    #[test]
    fn test_affine_matrix_identity() {
        let identity = AffineMatrix::identity();
        let elements = identity.as_array();
        assert_eq!(elements, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_affine_matrix_translation() {
        let trans = AffineMatrix::translation(10.0, 20.0);
        let elements = trans.as_array();
        assert_eq!(elements[2], 10.0);
        assert_eq!(elements[5], 20.0);
    }

    #[test]
    fn test_affine_matrix_scaling() {
        let scale = AffineMatrix::scaling(2.0, 3.0);
        let elements = scale.as_array();
        assert_eq!(elements[0], 2.0);
        assert_eq!(elements[4], 3.0);
    }

    #[test]
    fn test_affine_matrix_combination() {
        let t1 = AffineMatrix::translation(10.0, 20.0);
        let s = AffineMatrix::scaling(2.0, 2.0);
        let combined = t1.combine(&s);

        // The result should be a combined transformation
        assert!(combined.elements[0] > 0.0);
    }

    #[test]
    fn test_flops_estimation() {
        let flops_dct = TransformKernel::estimate_flops(64, 64, TransformType::DCT);
        let flops_rotate = TransformKernel::estimate_flops(64, 64, TransformType::Rotate90);

        assert!(flops_dct > 0);
        assert!(flops_rotate > 0);
        assert!(flops_dct > flops_rotate); // DCT should be more expensive
    }

    // -------------------------------------------------------------------------
    // New tests for affine, perspective, FFT, IFFT
    // -------------------------------------------------------------------------

    /// Identity affine: output must equal input.
    #[test]
    fn test_affine_identity() {
        let kernel = TransformKernel::new(TransformType::Affine);
        let width = 4u32;
        let height = 4u32;
        let input: Vec<f32> = (0..(width * height)).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; (width * height) as usize];
        // Identity forward matrix [a=1, b=0, c=0, d=1, tx=0, ty=0]
        let identity = [1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
        kernel
            .execute_affine_f32(&input, &mut output, width, height, identity)
            .expect("affine identity must succeed");
        assert_eq!(input, output, "identity affine must preserve all values");
    }

    /// Affine with 2× uniform scale: inverse maps output → input by dividing
    /// coordinates by 2, so `output[oy][ox]` comes from `input[oy/2][ox/2]`.
    #[test]
    fn test_affine_scale() {
        let kernel = TransformKernel::new(TransformType::Affine);
        let width = 8u32;
        let height = 8u32;
        // Assign unique values: pixel (x, y) = y * width + x as f32.
        let input: Vec<f32> = (0..(width * height)).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; (width * height) as usize];
        // Forward 2× scale: [a=2, b=0, c=0, d=2, tx=0, ty=0]
        // Inverse: maps (ox, oy) → (ox/2, oy/2)
        let mat = [2.0f32, 0.0, 0.0, 2.0, 0.0, 0.0];
        kernel
            .execute_affine_f32(&input, &mut output, width, height, mat)
            .expect("affine 2x scale must succeed");
        // Check several output pixels: output[oy*w+ox] == input[(oy/2)*w + (ox/2)]
        for oy in 0..height {
            for ox in 0..width {
                let expected = input[((oy / 2) * width + (ox / 2)) as usize];
                let got = output[(oy * width + ox) as usize];
                assert!(
                    (got - expected).abs() < 1e-5,
                    "output[{oy}][{ox}]={got}, expected {expected}"
                );
            }
        }
    }

    /// Singular affine matrix must return an error.
    #[test]
    fn test_affine_singular_returns_error() {
        let kernel = TransformKernel::new(TransformType::Affine);
        let width = 4u32;
        let height = 4u32;
        let input = vec![1.0f32; (width * height) as usize];
        let mut output = vec![0.0f32; (width * height) as usize];
        // Singular: det = 0*0 - 0*0 = 0
        let singular = [0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = kernel.execute_affine_f32(&input, &mut output, width, height, singular);
        assert!(result.is_err(), "singular affine must return error");
    }

    /// FFT of a unit impulse (1, 0, 0, …) should have all output magnitudes = 1.
    #[test]
    fn test_fft_impulse() {
        let kernel = TransformKernel::new(TransformType::FFT);
        let width = 4u32;
        let height = 4u32;
        let n = (width * height) as usize;
        let mut input = vec![0.0f32; n];
        input[0] = 1.0; // unit impulse
                        // Output needs 2*n slots (interleaved complex)
        let mut output = vec![0.0f32; 2 * n];
        kernel
            .execute_fft_f32(&input, &mut output, width, height)
            .expect("FFT of impulse must succeed");
        // All magnitudes should be 1.0 (within floating-point tolerance)
        for k in 0..n {
            let re = output[2 * k] as f64;
            let im = output[2 * k + 1] as f64;
            let mag = re.hypot(im);
            assert!(
                (mag - 1.0).abs() < 1e-4,
                "FFT[{k}] magnitude={mag:.6}, expected 1.0"
            );
        }
    }

    /// FFT followed by IFFT must recover the original signal.
    #[test]
    fn test_fft_ifft_roundtrip() {
        let kernel = TransformKernel::new(TransformType::FFT);
        let width = 4u32;
        let height = 4u32;
        let n = (width * height) as usize;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let mut freq = vec![0.0f32; 2 * n];
        kernel
            .execute_fft_f32(&input, &mut freq, width, height)
            .expect("FFT must succeed");
        let mut recovered = vec![0.0f32; 2 * n];
        kernel
            .execute_ifft_f32(&freq, &mut recovered, width, height)
            .expect("IFFT must succeed");
        // Real parts of recovered should match input; imaginary parts ≈ 0.
        for k in 0..n {
            let diff = (recovered[2 * k] - input[k]).abs();
            assert!(
                diff < 1e-4,
                "IFFT roundtrip: idx={k} expected={:.4}, got={:.4}",
                input[k],
                recovered[2 * k]
            );
        }
    }

    /// Perspective with identity homography should be a no-op.
    #[test]
    fn test_perspective_identity_f32() {
        let kernel = TransformKernel::new(TransformType::Perspective);
        let width = 4u32;
        let height = 4u32;
        let input: Vec<f32> = (0..(width * height)).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; (width * height) as usize];
        // Identity 3×3 homography
        let identity = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        kernel
            .execute_perspective_f32(&input, &mut output, width, height, identity)
            .expect("perspective identity must succeed");
        assert_eq!(input, output, "identity perspective must preserve values");
    }

    /// Singular perspective matrix must return an error.
    #[test]
    fn test_perspective_singular() {
        let kernel = TransformKernel::new(TransformType::Perspective);
        let width = 4u32;
        let height = 4u32;
        let input = vec![1.0f32; (width * height) as usize];
        let mut output = vec![0.0f32; (width * height) as usize];
        // All-zero matrix is singular.
        let singular = [0.0f32; 9];
        let result = kernel.execute_perspective_f32(&input, &mut output, width, height, singular);
        assert!(result.is_err(), "singular perspective must return error");
    }

    /// Affine u8 identity: output must equal input.
    #[test]
    fn test_affine_u8_identity() {
        let kernel = TransformKernel::new(TransformType::Affine);
        let width = 4u32;
        let height = 4u32;
        let channels = 3u32;
        let input: Vec<u8> = (0..(width * height * channels) as usize)
            .map(|i| (i % 256) as u8)
            .collect();
        let mut output = vec![0u8; input.len()];
        let identity = [1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
        kernel
            .execute_affine_u8(&input, &mut output, width, height, channels, identity)
            .expect("affine u8 identity must succeed");
        assert_eq!(input, output, "identity affine u8 must preserve all bytes");
    }
}
