//! OxiFFT-based FFT backend for high performance
//!
//! Pure Rust FFT implementation using OxiFFT (FFTW-compatible algorithms).
//! Provides high-performance FFT with plan caching via the wisdom system.
//!
//! This replaces the FFTW backend while maintaining Pure Rust Policy compliance.

use crate::error::{FFTError, FFTResult};
use crate::oxifft_plan_cache;
use oxifft::{Complex as OxiComplex, Direction, Flags};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::Complex;

/// Real-to-complex FFT using OxiFFT with plan caching
///
/// Computes the FFT of a real-valued signal, returning complex frequency components.
/// Uses cached plans for better performance on repeated transforms of the same size.
///
/// # Arguments
///
/// * `input` - Real-valued input signal
///
/// # Returns
///
/// * Complex frequency spectrum (N/2+1 components for real input of length N)
///
/// # Errors
///
/// Returns an error if the FFT computation fails.
pub fn rfft_oxifft(input: &ArrayView1<f64>) -> FFTResult<Array1<Complex<f64>>> {
    let n = input.len();

    // Prepare input and output buffers
    let input_vec: Vec<f64> = input.to_vec();
    let output_len = n / 2 + 1;
    let mut output: Vec<OxiComplex<f64>> = vec![OxiComplex::zero(); output_len];

    // Execute with cached plan
    oxifft_plan_cache::execute_r2c(&input_vec, &mut output)?;

    // Convert oxifft Complex to scirs2 Complex<f64>
    let result: Vec<Complex<f64>> = output.iter().map(|c| Complex::new(c.re, c.im)).collect();

    Ok(Array1::from_vec(result))
}

/// Complex-to-complex FFT using OxiFFT with plan caching
///
/// Computes the FFT of a complex-valued signal.
///
/// # Errors
///
/// Returns an error if the FFT computation fails.
pub fn fft_oxifft(input: &ArrayView1<Complex<f64>>) -> FFTResult<Array1<Complex<f64>>> {
    let n = input.len();

    // Convert scirs2 Complex to oxifft Complex
    let input_oxifft: Vec<OxiComplex<f64>> =
        input.iter().map(|c| OxiComplex::new(c.re, c.im)).collect();
    let mut output: Vec<OxiComplex<f64>> = vec![OxiComplex::zero(); n];

    // Execute with cached plan
    oxifft_plan_cache::execute_c2c(&input_oxifft, &mut output, Direction::Forward)?;

    // Convert back
    let result: Vec<Complex<f64>> = output.iter().map(|c| Complex::new(c.re, c.im)).collect();

    Ok(Array1::from_vec(result))
}

/// Inverse complex-to-complex FFT using OxiFFT with plan caching
///
/// # Errors
///
/// Returns an error if the FFT computation fails.
pub fn ifft_oxifft(input: &ArrayView1<Complex<f64>>) -> FFTResult<Array1<Complex<f64>>> {
    let n = input.len();

    // Convert to oxifft format
    let input_oxifft: Vec<OxiComplex<f64>> =
        input.iter().map(|c| OxiComplex::new(c.re, c.im)).collect();
    let mut output: Vec<OxiComplex<f64>> = vec![OxiComplex::zero(); n];

    // Execute with cached plan
    oxifft_plan_cache::execute_c2c(&input_oxifft, &mut output, Direction::Backward)?;

    // OxiFFT normalizes automatically for backward transforms via execute_normalized
    // But we use direct execution, so normalize manually
    let scale = 1.0 / (n as f64);
    let result: Vec<Complex<f64>> = output
        .iter()
        .map(|c| Complex::new(c.re * scale, c.im * scale))
        .collect();

    Ok(Array1::from_vec(result))
}

/// Inverse real FFT using OxiFFT with plan caching
///
/// Converts complex frequency spectrum back to real-valued time-domain signal.
///
/// # Errors
///
/// Returns an error if the FFT computation fails.
pub fn irfft_oxifft(input: &ArrayView1<Complex<f64>>, n: usize) -> FFTResult<Array1<f64>> {
    // Convert to oxifft format
    let input_oxifft: Vec<OxiComplex<f64>> =
        input.iter().map(|c| OxiComplex::new(c.re, c.im)).collect();
    let mut output: Vec<f64> = vec![0.0; n];

    // Execute with cached plan
    oxifft_plan_cache::execute_c2r(&input_oxifft, &mut output, n)?;

    // Normalize (OxiFFT doesn't normalize C2R by default)
    let scale = 1.0 / (n as f64);
    let result: Vec<f64> = output.iter().map(|&x| x * scale).collect();

    Ok(Array1::from_vec(result))
}

// ========================================
// 2D FFT FUNCTIONS
// ========================================

/// 2D complex-to-complex FFT using OxiFFT with plan caching
///
/// # Errors
///
/// Returns an error if the FFT computation fails.
pub fn fft2_oxifft(input: &ArrayView2<Complex<f64>>) -> FFTResult<Array2<Complex<f64>>> {
    let (rows, cols) = input.dim();
    let n = rows * cols;

    // Convert to row-major contiguous Vec
    let input_oxifft: Vec<OxiComplex<f64>> =
        input.iter().map(|c| OxiComplex::new(c.re, c.im)).collect();
    let mut output: Vec<OxiComplex<f64>> = vec![OxiComplex::zero(); n];

    // Execute with cached plan
    oxifft_plan_cache::execute_c2c_2d(&input_oxifft, &mut output, rows, cols, Direction::Forward)?;

    // Convert back and reshape to 2D
    let result: Vec<Complex<f64>> = output.iter().map(|c| Complex::new(c.re, c.im)).collect();

    Array2::from_shape_vec((rows, cols), result)
        .map_err(|e| FFTError::ComputationError(format!("Failed to reshape result: {:?}", e)))
}

/// 2D inverse complex-to-complex FFT using OxiFFT with plan caching
///
/// # Errors
///
/// Returns an error if the FFT computation fails.
pub fn ifft2_oxifft(input: &ArrayView2<Complex<f64>>) -> FFTResult<Array2<Complex<f64>>> {
    let (rows, cols) = input.dim();
    let n = rows * cols;

    // Convert to row-major contiguous Vec
    let input_oxifft: Vec<OxiComplex<f64>> =
        input.iter().map(|c| OxiComplex::new(c.re, c.im)).collect();
    let mut output: Vec<OxiComplex<f64>> = vec![OxiComplex::zero(); n];

    // Execute with cached plan
    oxifft_plan_cache::execute_c2c_2d(&input_oxifft, &mut output, rows, cols, Direction::Backward)?;

    // Normalize
    let scale = 1.0 / (n as f64);
    let result: Vec<Complex<f64>> = output
        .iter()
        .map(|c| Complex::new(c.re * scale, c.im * scale))
        .collect();

    Array2::from_shape_vec((rows, cols), result)
        .map_err(|e| FFTError::ComputationError(format!("Failed to reshape result: {:?}", e)))
}

/// 2D real-to-complex FFT using OxiFFT with plan caching
///
/// # Errors
///
/// Returns an error if the FFT computation fails.
pub fn rfft2_oxifft(input: &ArrayView2<f64>) -> FFTResult<Array2<Complex<f64>>> {
    let (rows, cols) = input.dim();

    // Convert to row-major contiguous Vec
    let input_vec: Vec<f64> = input.iter().cloned().collect();

    // For 2D real FFT, output has shape (rows, cols/2 + 1)
    let out_cols = cols / 2 + 1;
    let mut output: Vec<OxiComplex<f64>> = vec![OxiComplex::zero(); rows * out_cols];

    // Execute with cached plan
    oxifft_plan_cache::execute_r2c_2d(&input_vec, &mut output, rows, cols)?;

    // Convert back and reshape
    let result: Vec<Complex<f64>> = output.iter().map(|c| Complex::new(c.re, c.im)).collect();

    Array2::from_shape_vec((rows, out_cols), result)
        .map_err(|e| FFTError::ComputationError(format!("Failed to reshape result: {:?}", e)))
}

/// 2D inverse real FFT using OxiFFT with plan caching
///
/// # Errors
///
/// Returns an error if the FFT computation fails.
pub fn irfft2_oxifft(
    input: &ArrayView2<Complex<f64>>,
    shape: (usize, usize),
) -> FFTResult<Array2<f64>> {
    let (rows, cols) = shape;
    let (in_rows, in_cols) = input.dim();

    // Validate input dimensions
    if in_rows != rows || in_cols != cols / 2 + 1 {
        return Err(FFTError::ValueError(format!(
            "Input shape ({}, {}) doesn't match expected ({}, {}) for output shape ({}, {})",
            in_rows,
            in_cols,
            rows,
            cols / 2 + 1,
            rows,
            cols
        )));
    }

    // Convert to row-major contiguous Vec
    let input_oxifft: Vec<OxiComplex<f64>> =
        input.iter().map(|c| OxiComplex::new(c.re, c.im)).collect();
    let mut output: Vec<f64> = vec![0.0; rows * cols];

    // Execute with cached plan
    oxifft_plan_cache::execute_c2r_2d(&input_oxifft, &mut output, rows, cols)?;

    // Normalize
    let scale = 1.0 / ((rows * cols) as f64);
    let result: Vec<f64> = output.iter().map(|&x| x * scale).collect();

    Array2::from_shape_vec((rows, cols), result)
        .map_err(|e| FFTError::ComputationError(format!("Failed to reshape result: {:?}", e)))
}

// ========================================
// DCT/DST FUNCTIONS using R2R transforms with plan caching
// ========================================

/// DCT Type II using OxiFFT with plan caching (REDFT10)
/// This is the most commonly used DCT type
///
/// # Errors
///
/// Returns an error if the DCT computation fails.
pub fn dct2_oxifft(input: &ArrayView1<f64>) -> FFTResult<Array1<f64>> {
    let n = input.len();
    let input_vec: Vec<f64> = input.to_vec();
    let mut output: Vec<f64> = vec![0.0; n];

    // Execute with cached plan
    oxifft_plan_cache::execute_dct2(&input_vec, &mut output)?;

    Ok(Array1::from_vec(output))
}

/// IDCT Type II using OxiFFT with plan caching (REDFT01 = DCT-III)
/// DCT-III is the inverse of DCT-II
///
/// # Errors
///
/// Returns an error if the IDCT computation fails.
pub fn idct2_oxifft(input: &ArrayView1<f64>) -> FFTResult<Array1<f64>> {
    let n = input.len();
    let input_vec: Vec<f64> = input.to_vec();
    let mut output: Vec<f64> = vec![0.0; n];

    // Execute with cached plan
    oxifft_plan_cache::execute_idct2(&input_vec, &mut output)?;

    // Normalize - need to scale by 1/(2*N)
    let scale = 1.0 / (2.0 * n as f64);
    let result: Vec<f64> = output.iter().map(|&x| x * scale).collect();

    Ok(Array1::from_vec(result))
}

/// DST Type II using OxiFFT with plan caching (RODFT10)
///
/// # Errors
///
/// Returns an error if the DST computation fails.
pub fn dst2_oxifft(input: &ArrayView1<f64>) -> FFTResult<Array1<f64>> {
    let n = input.len();
    let input_vec: Vec<f64> = input.to_vec();
    let mut output: Vec<f64> = vec![0.0; n];

    // Execute with cached plan
    oxifft_plan_cache::execute_dst2(&input_vec, &mut output)?;

    Ok(Array1::from_vec(output))
}

/// IDST Type II using OxiFFT with plan caching (RODFT01 = DST-III)
///
/// # Errors
///
/// Returns an error if the IDST computation fails.
pub fn idst2_oxifft(input: &ArrayView1<f64>) -> FFTResult<Array1<f64>> {
    let n = input.len();
    let input_vec: Vec<f64> = input.to_vec();
    let mut output: Vec<f64> = vec![0.0; n];

    // Execute with cached plan
    oxifft_plan_cache::execute_idst2(&input_vec, &mut output)?;

    // Normalize
    let scale = 1.0 / (2.0 * n as f64);
    let result: Vec<f64> = output.iter().map(|&x| x * scale).collect();

    Ok(Array1::from_vec(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_fft_oxifft_basic() {
        let input = array![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0)
        ];
        let result = fft_oxifft(&input.view()).expect("FFT failed");
        assert_eq!(result.len(), 4);

        // Sum of inputs should equal DC component
        assert_relative_eq!(result[0].re, 10.0, epsilon = 1e-10);
        assert_relative_eq!(result[0].im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ifft_oxifft_roundtrip() {
        let input = array![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0)
        ];
        let fft_result = fft_oxifft(&input.view()).expect("FFT failed");
        let ifft_result = ifft_oxifft(&fft_result.view()).expect("IFFT failed");

        for (orig, recovered) in input.iter().zip(ifft_result.iter()) {
            assert_relative_eq!(orig.re, recovered.re, epsilon = 1e-10);
            assert_relative_eq!(orig.im, recovered.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_rfft_oxifft_basic() {
        let input = array![1.0, 2.0, 3.0, 4.0];
        let result = rfft_oxifft(&input.view()).expect("RFFT failed");
        // Output length should be n/2 + 1 = 3
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_irfft_oxifft_roundtrip() {
        let input = array![1.0, 2.0, 3.0, 4.0];
        let rfft_result = rfft_oxifft(&input.view()).expect("RFFT failed");
        let irfft_result = irfft_oxifft(&rfft_result.view(), 4).expect("IRFFT failed");

        for (orig, recovered) in input.iter().zip(irfft_result.iter()) {
            assert_relative_eq!(*orig, *recovered, epsilon = 1e-10);
        }
    }
}
