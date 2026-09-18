//! Image restoration algorithms for `OxiMedia`.
//!
//! `oximedia-restoration` provides advanced image restoration techniques:
//!
//! - **Blind deconvolution** — Richardson-Lucy iterative algorithm that
//!   simultaneously estimates the latent sharp image and the unknown point
//!   spread function (PSF) from a single blurry observation, using
//!   frequency-domain convolutions via `OxiFFT`.
//!
//! - **Content-aware inpainting** — PatchMatch-based texture synthesis that
//!   fills masked (damaged/missing) regions by finding and blending the
//!   nearest-neighbour patches from the undamaged surroundings.
//!
//! # Quick start
//!
//! ```rust
//! use oximedia_restoration::blind_deconv::{blind_deconvolve, BlindDeconvConfig};
//! use oximedia_restoration::inpaint::{inpaint_patchmatch, InpaintConfig};
//!
//! // --- Blind deconvolution (single-channel f32 image) ---
//! let w = 8u32;
//! let h = 8u32;
//! let blurry = vec![0.5f32; (w * h) as usize];
//! let (sharp, psf) = blind_deconvolve(&blurry, w, h, &BlindDeconvConfig::default());
//! assert_eq!(sharp.len(), (w * h) as usize);
//!
//! // --- Inpainting (packed RGB u8 image) ---
//! let image = vec![128u8; (w * h * 3) as usize];
//! let mask = vec![false; (w * h) as usize]; // no pixels masked
//! let restored = inpaint_patchmatch(&image, &mask, w, h, &InpaintConfig::default());
//! assert_eq!(restored.len(), image.len());
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod blind_deconv;
pub mod inpaint;
