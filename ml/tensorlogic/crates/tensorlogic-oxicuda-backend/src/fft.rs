//! GPU-accelerated FFT operations via OxiCUDA FFT.
//!
//! This module wraps `oxicuda-fft` to provide host-slice-in / host-vec-out
//! 1-D complex-to-complex FFT (both forward and inverse directions).
//!
//! The functions are **only available** when the crate is built with both
//! `--features gpu` and `--features fft`.  Calling them without those
//! features returns `OxiCudaBackendError::FftDisabled`.

#[cfg(all(feature = "gpu", feature = "fft"))]
mod gpu_impl {
    use oxicuda_fft::{types::Complex, FftDirection, FftHandle, FftPlan, FftType};
    use oxicuda_memory::DeviceBuffer;

    use crate::error::OxiCudaBackendError;
    use crate::executor::GpuState;

    /// Execute a 1-D forward C2C FFT on the GPU.
    ///
    /// `signal` is a host slice of complex single-precision samples.
    /// Returns a `Vec<Complex<f32>>` of the same length holding the frequency-domain
    /// representation.
    ///
    /// # Errors
    ///
    /// - [`OxiCudaBackendError::DimensionOverflow`] when `signal.len()` exceeds
    ///   practical GPU limits (guarded conservatively at `u32::MAX` elements).
    /// - [`OxiCudaBackendError::Fft`] for any OxiCUDA FFT or driver error.
    pub fn forward_c2c_1d(
        state: &GpuState,
        signal: &[Complex<f32>],
    ) -> Result<Vec<Complex<f32>>, OxiCudaBackendError> {
        c2c_1d(state, signal, FftDirection::Forward)
    }

    /// Execute a 1-D inverse C2C FFT on the GPU.
    ///
    /// `signal` is a host slice of complex single-precision frequency-domain values.
    /// Returns an unnormalised `Vec<Complex<f32>>` of the same length.  The caller
    /// is responsible for dividing by `N` to obtain the properly scaled time-domain
    /// signal.
    ///
    /// # Errors
    ///
    /// - [`OxiCudaBackendError::DimensionOverflow`] when `signal.len()` exceeds
    ///   practical GPU limits (guarded conservatively at `u32::MAX` elements).
    /// - [`OxiCudaBackendError::Fft`] for any OxiCUDA FFT or driver error.
    pub fn inverse_c2c_1d(
        state: &GpuState,
        signal: &[Complex<f32>],
    ) -> Result<Vec<Complex<f32>>, OxiCudaBackendError> {
        c2c_1d(state, signal, FftDirection::Inverse)
    }

    /// Shared implementation for forward and inverse 1-D C2C FFT.
    ///
    /// Steps:
    /// 1. Validate length fits in a `u32` (conservative safety bound).
    /// 2. Build an `FftPlan` (single-batch, C2C).
    /// 3. Create an `FftHandle` bound to the context.
    /// 4. Upload input to device.
    /// 5. Allocate an output device buffer.
    /// 6. Execute the FFT (passing raw `CUdeviceptr`s).
    /// 7. Synchronise the handle's stream.
    /// 8. Copy output back to host.
    fn c2c_1d(
        state: &GpuState,
        signal: &[Complex<f32>],
        direction: FftDirection,
    ) -> Result<Vec<Complex<f32>>, OxiCudaBackendError> {
        let n = signal.len();

        // Conservative bound: FftPlan uses usize but practical GPU memory limits
        // mean anything beyond u32::MAX is not realistically useful.
        if n > u32::MAX as usize {
            return Err(OxiCudaBackendError::DimensionOverflow(format!(
                "signal length {n} exceeds u32::MAX ({})",
                u32::MAX
            )));
        }

        // Build the FFT plan (batch = 1, single-precision C2C).
        let plan = FftPlan::new_1d(n, FftType::C2C, 1)
            .map_err(|e| OxiCudaBackendError::Fft(e.to_string()))?;

        // Build the FFT handle bound to our existing CUDA context.
        let handle =
            FftHandle::new(state.context()).map_err(|e| OxiCudaBackendError::Fft(e.to_string()))?;

        // Host → device: input buffer.
        let input_buf = DeviceBuffer::<Complex<f32>>::from_host(signal)
            .map_err(|e| OxiCudaBackendError::OxiCuda(e.to_string()))?;

        // Allocate device output buffer (same length as input).
        let output_buf = DeviceBuffer::<Complex<f32>>::alloc(n)
            .map_err(|e| OxiCudaBackendError::OxiCuda(e.to_string()))?;

        // Execute the FFT. `FftHandle::execute` takes raw CUdeviceptr values.
        handle
            .execute(
                &plan,
                input_buf.as_device_ptr(),
                output_buf.as_device_ptr(),
                direction,
            )
            .map_err(|e| OxiCudaBackendError::Fft(e.to_string()))?;

        // Synchronise the handle's internal stream before reading back results.
        handle
            .stream()
            .synchronize()
            .map_err(|e| OxiCudaBackendError::OxiCuda(e.to_string()))?;

        // Device → host: copy results out.
        let mut result = vec![
            Complex {
                re: 0.0_f32,
                im: 0.0_f32
            };
            n
        ];
        output_buf
            .copy_to_host(&mut result)
            .map_err(|e| OxiCudaBackendError::OxiCuda(e.to_string()))?;

        Ok(result)
    }
}

// ---- Public API — compiled only with both `gpu` and `fft` features ----------

/// Execute a 1-D forward C2C FFT on the GPU.
///
/// Requires both the `gpu` and `fft` Cargo features.  When either is absent
/// this symbol is not exported; call sites should gate on the same feature
/// combination and fall back to [`crate::error::OxiCudaBackendError::FftDisabled`].
#[cfg(all(feature = "gpu", feature = "fft"))]
pub use gpu_impl::forward_c2c_1d;

/// Execute a 1-D inverse C2C FFT on the GPU.
///
/// Requires both the `gpu` and `fft` Cargo features.  When either is absent
/// this symbol is not exported; call sites should gate on the same feature
/// combination and fall back to [`crate::error::OxiCudaBackendError::FftDisabled`].
#[cfg(all(feature = "gpu", feature = "fft"))]
pub use gpu_impl::inverse_c2c_1d;
