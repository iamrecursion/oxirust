//! Richardson-Lucy blind deconvolution for image deblurring.
//!
//! Blind deconvolution jointly estimates both the latent sharp image `u` and
//! the unknown point spread function (PSF) `h` from a single blurry observation
//! `f`, without any prior knowledge of the blur kernel.
//!
//! # Algorithm
//!
//! The Richardson-Lucy (R-L) algorithm maximises the Poisson log-likelihood of
//! the observed image. Given the forward model  `f = h ⊛ u + noise`, each
//! iteration applies:
//!
//! ```text
//! u_{n+1} = u_n · [ h̄  ⊛ ( f / (h ⊛ u_n) ) ]
//! h_{n+1} = h_n · [ ū_n ⊛ ( f / (h_n ⊛ u_n) ) ]
//! ```
//!
//! where `⊛` denotes convolution and `h̄` / `ū_n` are the 180°-rotated
//! (flipped) versions of `h` and `u_n`.  Convolutions are computed in the
//! frequency domain via `oxifft::rfft2d` / `oxifft::irfft2d` for O(N log N)
//! performance.  After every PSF update, `h` is clamped to non-negative values
//! and renormalised so it sums to 1, ensuring it remains a proper blur kernel.
//!
//! # Example
//!
//! ```rust
//! use oximedia_restoration::blind_deconv::{blind_deconvolve, BlindDeconvConfig};
//!
//! let width = 8u32;
//! let height = 8u32;
//! let blurry: Vec<f32> = vec![0.5; (width * height) as usize];
//! let cfg = BlindDeconvConfig { iterations: 5, psf_size: 3, regularization: 1e-6 };
//! let (sharp, psf) = blind_deconvolve(&blurry, width, height, &cfg);
//! assert_eq!(sharp.len(), (width * height) as usize);
//! assert_eq!(psf.len(), ((2 * cfg.psf_size + 1) * (2 * cfg.psf_size + 1)) as usize);
//! ```

use oxifft::{irfft2d, rfft2d, Complex};

// ─── Public API ───────────────────────────────────────────────────────────────

/// Configuration for the Richardson-Lucy blind deconvolution.
#[derive(Debug, Clone)]
pub struct BlindDeconvConfig {
    /// Number of R-L iterations (more → sharper, costlier). Default: 15.
    pub iterations: u32,
    /// PSF half-size in pixels.  The estimated kernel will have side length
    /// `2 * psf_size + 1`.  Default: 5.
    pub psf_size: u32,
    /// Small additive regularisation constant added to denominators to avoid
    /// division by zero.  Default: 1e-6.
    pub regularization: f32,
}

impl Default for BlindDeconvConfig {
    fn default() -> Self {
        Self {
            iterations: 15,
            psf_size: 5,
            regularization: 1e-6,
        }
    }
}

/// Run Richardson-Lucy blind deconvolution on a single-channel float image.
///
/// # Parameters
/// - `blurry` – linearised row-major pixel values in `[0, 1]`.  Length must
///   equal `w * h`.
/// - `w`, `h` – image dimensions in pixels.
/// - `cfg` – algorithm configuration.
///
/// # Returns
/// A pair `(sharp_image, psf)` where:
/// - `sharp_image` has the same length as `blurry` (`w * h`), values clamped
///   to `[0, 1]`.
/// - `psf` has length `(2*psf_size+1)²`, values ≥ 0 summing to 1.
///
/// # Panics
/// Panics if `blurry.len() != w * h`.
pub fn blind_deconvolve(
    blurry: &[f32],
    w: u32,
    h: u32,
    cfg: &BlindDeconvConfig,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(
        blurry.len(),
        (w as usize) * (h as usize),
        "blurry.len() must equal w * h"
    );

    let psf_side = (2 * cfg.psf_size + 1) as usize;
    let img_w = w as usize;
    let img_h = h as usize;
    let reg = cfg.regularization;

    // ── Pad dimensions to next power of two for efficient FFT ────────────────
    let pad_h = next_pow2(img_h + psf_side - 1);
    let pad_w = next_pow2(img_w + psf_side - 1);

    // ── Initialise PSF h as a small Gaussian centred in the PSF window ───────
    let mut psf = init_gaussian_psf(psf_side, psf_side);

    // ── Initialise latent image u as a copy of the blurry input ──────────────
    let mut u: Vec<f32> = blurry.to_vec();

    // ── Precompute padded blurry image F (stays constant) ────────────────────
    let f_padded = pad_image(blurry, img_w, img_h, pad_w, pad_h);

    for _ in 0..cfg.iterations {
        // ── Pad current u and h ───────────────────────────────────────────────
        let u_padded = pad_image(&u, img_w, img_h, pad_w, pad_h);
        let h_padded = embed_psf(&psf, psf_side, psf_side, pad_w, pad_h);

        // ── Step 1: reblur estimate  r = h ⊛ u ───────────────────────────────
        let r_padded = fft_convolve(&u_padded, &h_padded, pad_h, pad_w);

        // ── Build ratio  ratio = f / (r + reg) ───────────────────────────────
        let ratio: Vec<f32> = f_padded
            .iter()
            .zip(r_padded.iter())
            .map(|(&fi, &ri)| fi / (ri + reg))
            .collect();

        // ── Update u: u_new = u · (h̄ ⊛ ratio) ──────────────────────────────
        {
            let h_flip = flip_psf(&psf, psf_side, psf_side);
            let h_flip_padded = embed_psf(&h_flip, psf_side, psf_side, pad_w, pad_h);
            let correction = fft_convolve(&ratio, &h_flip_padded, pad_h, pad_w);
            for (ui, ci) in u
                .iter_mut()
                .zip(crop_image(&correction, img_w, img_h, pad_w, pad_h).iter())
            {
                *ui = (*ui * *ci).clamp(0.0, 1.0);
            }
        }

        // ── Update h: h_new = h · (ū ⊛ ratio) ──────────────────────────────
        {
            let u_padded2 = pad_image(&u, img_w, img_h, pad_w, pad_h);
            let u_flip_padded = flip_padded(&u_padded2, pad_w, pad_h);
            let correction_full = fft_convolve(&ratio, &u_flip_padded, pad_h, pad_w);
            // Extract PSF-sized region from correction
            let corr_psf = crop_image(&correction_full, psf_side, psf_side, pad_w, pad_h);
            for (hi, ci) in psf.iter_mut().zip(corr_psf.iter()) {
                *hi = (*hi * *ci).max(0.0);
            }
            normalise_psf(&mut psf);
        }
    }

    (u, psf)
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Compute the next power of two ≥ `n`.
fn next_pow2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

/// Initialise an `(rows × cols)` Gaussian PSF centred at the middle pixel.
/// Values are positive and sum to 1.
fn init_gaussian_psf(rows: usize, cols: usize) -> Vec<f32> {
    let cy = (rows / 2) as f32;
    let cx = (cols / 2) as f32;
    // Sigma ≈ 1.0 — a slight smooth kernel as initial estimate
    let sigma = 1.0_f32;
    let mut psf = vec![0.0f32; rows * cols];
    let mut sum = 0.0f32;
    for r in 0..rows {
        for c in 0..cols {
            let dy = r as f32 - cy;
            let dx = c as f32 - cx;
            let val = (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
            psf[r * cols + c] = val;
            sum += val;
        }
    }
    if sum > 0.0 {
        for v in &mut psf {
            *v /= sum;
        }
    } else {
        // Degenerate case: delta at centre
        psf[cy as usize * cols + cx as usize] = 1.0;
    }
    psf
}

/// Pad an `(img_w × img_h)` image (row-major) into a `(pad_w × pad_h)` buffer.
/// The image occupies the top-left corner; remaining pixels are replicated from
/// the nearest edge (mirror-pad to reduce ringing artefacts).
fn pad_image(img: &[f32], img_w: usize, img_h: usize, pad_w: usize, pad_h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; pad_h * pad_w];
    for r in 0..pad_h {
        let sr = r.min(img_h - 1);
        for c in 0..pad_w {
            let sc = c.min(img_w - 1);
            out[r * pad_w + c] = img[sr * img_w + sc];
        }
    }
    out
}

/// Embed a small PSF kernel into the top-left corner of a `(pad_w × pad_h)`
/// frequency-domain buffer, with the kernel centre placed at pixel (0,0) so
/// that the FFT-based convolution is a linear (non-circular) correlation.
fn embed_psf(psf: &[f32], psf_w: usize, psf_h: usize, pad_w: usize, pad_h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; pad_h * pad_w];
    let cy = psf_h / 2;
    let cx = psf_w / 2;
    for r in 0..psf_h {
        for c in 0..psf_w {
            // Wrap kernel so its centre is at (0,0) in the padded array
            let pr = (r + pad_h - cy) % pad_h;
            let pc = (c + pad_w - cx) % pad_w;
            out[pr * pad_w + pc] = psf[r * psf_w + c];
        }
    }
    out
}

/// 180°-rotate (flip) a PSF kernel.
fn flip_psf(psf: &[f32], psf_w: usize, psf_h: usize) -> Vec<f32> {
    let n = psf_w * psf_h;
    let mut flipped = vec![0.0f32; n];
    for i in 0..n {
        flipped[n - 1 - i] = psf[i];
    }
    flipped
}

/// 180°-rotate a padded image buffer.
fn flip_padded(buf: &[f32], pad_w: usize, pad_h: usize) -> Vec<f32> {
    let n = pad_w * pad_h;
    let mut flipped = vec![0.0f32; n];
    for i in 0..n {
        flipped[n - 1 - i] = buf[i];
    }
    flipped
}

/// Extract the top-left `(out_w × out_h)` region from a `(pad_w × pad_h)` buffer.
fn crop_image(buf: &[f32], out_w: usize, out_h: usize, pad_w: usize, _pad_h: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(out_h * out_w);
    for r in 0..out_h {
        for c in 0..out_w {
            out.push(buf[r * pad_w + c]);
        }
    }
    out
}

/// Clamp PSF values to ≥ 0 and renormalise so they sum to 1.
fn normalise_psf(psf: &mut [f32]) {
    for v in psf.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    let sum: f32 = psf.iter().sum();
    if sum > 1e-12 {
        for v in psf.iter_mut() {
            *v /= sum;
        }
    } else {
        // Degenerate: reset to uniform
        let n = psf.len();
        let inv = 1.0 / n as f32;
        for v in psf.iter_mut() {
            *v = inv;
        }
    }
}

/// Frequency-domain multiplication of two padded images.
/// Computes the linear convolution `a ⊛ b` in the `(pad_h × pad_w)` domain
/// using rfft2d → complex multiply → irfft2d.
///
/// Both `a` and `b` must have length `pad_h * pad_w`.
fn fft_convolve(a: &[f32], b: &[f32], pad_h: usize, pad_w: usize) -> Vec<f32> {
    // rfft2d returns pad_h * (pad_w/2 + 1) complex values
    let a_freq = rfft2d::<f32>(a, pad_h, pad_w);
    let b_freq = rfft2d::<f32>(b, pad_h, pad_w);

    // Element-wise complex multiply
    let prod: Vec<Complex<f32>> = a_freq
        .iter()
        .zip(b_freq.iter())
        .map(|(af, bf)| Complex {
            re: af.re * bf.re - af.im * bf.im,
            im: af.re * bf.im + af.im * bf.re,
        })
        .collect();

    // irfft2d normalises by 1/(pad_h * pad_w)
    irfft2d::<f32>(&prod, pad_h, pad_w)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Uniform input: all-constant image should come out near-constant after R-L.
    /// A constant image convolved with any normalised PSF is itself, so the
    /// ratio f/(h⊛u) is identically 1 and no update occurs — output ≈ input.
    #[test]
    fn test_blind_deconv_uniform_input() {
        let w = 16u32;
        let h = 16u32;
        let val = 0.5f32;
        let blurry = vec![val; (w * h) as usize];
        let cfg = BlindDeconvConfig {
            iterations: 15,
            psf_size: 5,
            regularization: 1e-6,
        };
        let (sharp, _psf) = blind_deconvolve(&blurry, w, h, &cfg);
        assert_eq!(sharp.len(), (w * h) as usize);
        for &v in &sharp {
            // Uniform image should remain near-constant (within float precision)
            assert!((v - val).abs() < 0.05, "Expected ~{val}, got {v}");
        }
    }

    /// Output shape must match input dimensions; PSF must have side (2*psf_size+1)².
    #[test]
    fn test_blind_deconv_output_shape() {
        let w = 20u32;
        let h = 15u32;
        let cfg = BlindDeconvConfig {
            iterations: 3,
            psf_size: 4,
            regularization: 1e-6,
        };
        let blurry = vec![0.3f32; (w * h) as usize];
        let (sharp, psf) = blind_deconvolve(&blurry, w, h, &cfg);

        assert_eq!(sharp.len(), (w * h) as usize, "Sharp image length mismatch");
        let expected_psf_side = 2 * cfg.psf_size + 1;
        assert_eq!(
            psf.len(),
            (expected_psf_side * expected_psf_side) as usize,
            "PSF length mismatch"
        );
    }

    /// Convergence: deconvolving a box-blurred image should improve PSNR vs the
    /// original sharp reference after 15 iterations, compared to the blurry input.
    #[test]
    fn test_blind_deconv_convergence() {
        let w = 32u32;
        let h = 32u32;
        let n = (w * h) as usize;

        // Create a synthetic sharp image: vertical stripe pattern
        let sharp_ref: Vec<f32> = (0..n)
            .map(|i| {
                let col = (i % w as usize) as f32 / w as f32;
                if col < 0.33 {
                    0.8f32
                } else if col < 0.66 {
                    0.2f32
                } else {
                    0.6f32
                }
            })
            .collect();

        // Apply a 3×3 box blur to create the blurry input
        let blurry = box_blur(&sharp_ref, w as usize, h as usize, 1);

        // Compute blurry PSNR vs sharp reference
        let blurry_psnr = psnr(&blurry, &sharp_ref);

        // Run blind deconvolution
        let cfg = BlindDeconvConfig {
            iterations: 15,
            psf_size: 3,
            regularization: 1e-6,
        };
        let (deblurred, _psf) = blind_deconvolve(&blurry, w, h, &cfg);

        // Compute deblurred PSNR vs sharp reference
        let deblurred_psnr = psnr(&deblurred, &sharp_ref);

        assert!(
            deblurred_psnr > blurry_psnr,
            "Expected deblurred PSNR ({deblurred_psnr:.2} dB) > blurry PSNR ({blurry_psnr:.2} dB)"
        );
    }

    // ─── Test helpers ─────────────────────────────────────────────────────────

    /// Naïve box blur with radius `r` (kernel size 2r+1).
    fn box_blur(img: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; w * h];
        let ksize = (2 * r + 1) as f32;
        let k2 = ksize * ksize;
        for row in 0..h {
            for col in 0..w {
                let mut acc = 0.0f32;
                for dr in 0..=(2 * r) {
                    let sr = (row + dr).saturating_sub(r).min(h - 1);
                    for dc in 0..=(2 * r) {
                        let sc = (col + dc).saturating_sub(r).min(w - 1);
                        acc += img[sr * w + sc];
                    }
                }
                out[row * w + col] = acc / k2;
            }
        }
        out
    }

    /// Compute PSNR (dB) between two images, both in [0,1].
    fn psnr(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let mse: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            / a.len() as f32;
        if mse < 1e-10 {
            return 100.0; // perfect
        }
        10.0 * (1.0_f32 / mse).log10()
    }
}
