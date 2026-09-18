//! Digital signal processing (DSP) utilities.
//!
//! This module exposes reusable DSP building blocks:
//!
//! - [`block_fft`]: Overlap-add STFT processor using `oxifft`.

pub mod block_fft;

pub use block_fft::{BlockFftProcessor, Window};
