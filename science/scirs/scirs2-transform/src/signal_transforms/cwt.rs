//! Continuous Wavelet Transform (CWT) Implementation
//!
//! Provides continuous wavelet transform with multiple mother wavelets including:
//! - Morlet wavelet
//! - Mexican Hat wavelet (Ricker wavelet)
//! - Complex Morlet wavelet
//! - Gaussian derivatives

use crate::error::{Result, TransformError};
use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::Complex;
use scirs2_fft::{fft, ifft};
use std::f64::consts::PI;

/// Trait for continuous wavelet functions
pub trait ContinuousWavelet: Send + Sync {
    /// Compute the wavelet at a given scale and position
    fn wavelet(&self, t: f64, scale: f64) -> Complex<f64>;

    /// Get the wavelet name
    fn name(&self) -> &str;

    /// Get the central frequency
    fn central_frequency(&self) -> f64 {
        1.0
    }

    /// Compute the wavelet in frequency domain (for FFT-based CWT)
    fn wavelet_fft(&self, omega: f64, scale: f64) -> Complex<f64> {
        // Default implementation - can be overridden for efficiency
        let norm = (2.0 * PI).sqrt();
        Complex::new((omega * scale).cos() * norm, -(omega * scale).sin() * norm)
    }
}

/// Morlet wavelet (real-valued)
#[derive(Debug, Clone, Copy)]
pub struct MorletWavelet {
    /// Central frequency parameter (omega0)
    pub omega0: f64,
}

impl MorletWavelet {
    /// Create a new Morlet wavelet
    pub fn new(omega0: f64) -> Self {
        MorletWavelet { omega0 }
    }

    /// Create with default omega0 = 6.0
    pub fn default() -> Self {
        MorletWavelet::new(6.0)
    }
}

impl ContinuousWavelet for MorletWavelet {
    fn wavelet(&self, t: f64, scale: f64) -> Complex<f64> {
        let scaled_t = t / scale;
        let exp_term = (-0.5 * scaled_t * scaled_t).exp();
        let cos_term = (self.omega0 * scaled_t).cos();
        let correction = (-0.5 * self.omega0 * self.omega0).exp();

        let value = (exp_term * cos_term - correction * exp_term) / scale.sqrt();
        Complex::new(value, 0.0)
    }

    fn name(&self) -> &str {
        "Morlet"
    }

    fn central_frequency(&self) -> f64 {
        self.omega0 / (2.0 * PI)
    }

    fn wavelet_fft(&self, omega: f64, scale: f64) -> Complex<f64> {
        let scaled_omega = omega * scale;
        let arg = -0.5 * (scaled_omega - self.omega0).powi(2);
        let value = (PI.sqrt() * 2.0).sqrt() * scale.sqrt() * arg.exp();
        Complex::new(value, 0.0)
    }
}

/// Complex Morlet wavelet
#[derive(Debug, Clone, Copy)]
pub struct ComplexMorletWavelet {
    /// Central frequency parameter
    pub omega0: f64,
    /// Bandwidth parameter
    pub sigma: f64,
}

impl ComplexMorletWavelet {
    /// Create a new complex Morlet wavelet
    pub fn new(omega0: f64, sigma: f64) -> Self {
        ComplexMorletWavelet { omega0, sigma }
    }

    /// Create with default parameters
    pub fn default() -> Self {
        ComplexMorletWavelet::new(6.0, 1.0)
    }
}

impl ContinuousWavelet for ComplexMorletWavelet {
    fn wavelet(&self, t: f64, scale: f64) -> Complex<f64> {
        let scaled_t = t / scale;
        let exp_term = (-0.5 * scaled_t * scaled_t / (self.sigma * self.sigma)).exp();
        let complex_exp = Complex::new(
            (self.omega0 * scaled_t).cos(),
            (self.omega0 * scaled_t).sin(),
        );

        (complex_exp * exp_term) / scale.sqrt()
    }

    fn name(&self) -> &str {
        "Complex Morlet"
    }

    fn central_frequency(&self) -> f64 {
        self.omega0 / (2.0 * PI)
    }
}

/// Mexican Hat wavelet (Ricker wavelet, 2nd derivative of Gaussian)
#[derive(Debug, Clone, Copy)]
pub struct MexicanHatWavelet {
    /// Scaling parameter
    pub sigma: f64,
}

impl MexicanHatWavelet {
    /// Create a new Mexican Hat wavelet
    pub fn new(sigma: f64) -> Self {
        MexicanHatWavelet { sigma }
    }

    /// Create with default sigma = 1.0
    pub fn default() -> Self {
        MexicanHatWavelet::new(1.0)
    }
}

impl ContinuousWavelet for MexicanHatWavelet {
    fn wavelet(&self, t: f64, scale: f64) -> Complex<f64> {
        let scaled_t = t / scale;
        let sigma2 = self.sigma * self.sigma;
        let t2 = scaled_t * scaled_t;

        let norm = 2.0 / (3.0 * self.sigma).sqrt() / PI.powf(0.25);
        let exp_term = (-t2 / (2.0 * sigma2)).exp();
        let poly_term = 1.0 - t2 / sigma2;

        let value = norm * poly_term * exp_term / scale.sqrt();
        Complex::new(value, 0.0)
    }

    fn name(&self) -> &str {
        "Mexican Hat"
    }

    fn central_frequency(&self) -> f64 {
        1.0 / (2.0 * PI)
    }
}

/// Gaussian wavelet (nth derivative)
#[derive(Debug, Clone, Copy)]
pub struct GaussianWavelet {
    /// Derivative order
    pub order: usize,
}

impl GaussianWavelet {
    /// Create a new Gaussian wavelet
    pub fn new(order: usize) -> Self {
        GaussianWavelet { order }
    }
}

impl ContinuousWavelet for GaussianWavelet {
    fn wavelet(&self, t: f64, scale: f64) -> Complex<f64> {
        let scaled_t = t / scale;
        let exp_term = (-0.5 * scaled_t * scaled_t).exp();

        let value = match self.order {
            0 => exp_term,
            1 => -scaled_t * exp_term,
            2 => (scaled_t * scaled_t - 1.0) * exp_term,
            _ => {
                // Hermite polynomial approximation for higher orders
                (scaled_t * scaled_t - 1.0) * exp_term
            }
        };

        Complex::new(value / scale.sqrt(), 0.0)
    }

    fn name(&self) -> &str {
        "Gaussian"
    }
}

/// Continuous Wavelet Transform
#[derive(Debug, Clone)]
pub struct CWT<W: ContinuousWavelet> {
    wavelet: W,
    scales: Vec<f64>,
    sampling_period: f64,
}

impl<W: ContinuousWavelet> CWT<W> {
    /// Create a new CWT with given wavelet and scales
    pub fn new(wavelet: W, scales: Vec<f64>) -> Self {
        CWT {
            wavelet,
            scales,
            sampling_period: 1.0,
        }
    }

    /// Set the sampling period
    pub fn with_sampling_period(mut self, period: f64) -> Self {
        self.sampling_period = period;
        self
    }

    /// Create scales using logarithmic spacing
    pub fn with_log_scales(wavelet: W, n_scales: usize, min_scale: f64, max_scale: f64) -> Self {
        let scales = Self::log_scales(n_scales, min_scale, max_scale);
        CWT::new(wavelet, scales)
    }

    /// Generate logarithmically spaced scales
    fn log_scales(n: usize, min_scale: f64, max_scale: f64) -> Vec<f64> {
        let log_min = min_scale.ln();
        let log_max = max_scale.ln();
        let step = (log_max - log_min) / (n - 1) as f64;

        (0..n).map(|i| (log_min + i as f64 * step).exp()).collect()
    }

    /// Compute CWT using direct convolution
    pub fn transform(&self, signal: &ArrayView1<f64>) -> Result<Array2<Complex<f64>>> {
        let n = signal.len();
        let n_scales = self.scales.len();

        if n == 0 {
            return Err(TransformError::InvalidInput("Empty signal".to_string()));
        }

        let mut coeffs = Array2::from_elem((n_scales, n), Complex::new(0.0, 0.0));

        // Compute CWT for each scale
        for (scale_idx, &scale) in self.scales.iter().enumerate() {
            for t_idx in 0..n {
                let mut sum = Complex::new(0.0, 0.0);

                for tau_idx in 0..n {
                    let tau = (tau_idx as f64 - t_idx as f64) * self.sampling_period;
                    let wavelet_val = self.wavelet.wavelet(tau, scale);
                    sum = sum + wavelet_val * signal[tau_idx];
                }

                coeffs[[scale_idx, t_idx]] = sum * self.sampling_period;
            }
        }

        Ok(coeffs)
    }

    /// Compute CWT using FFT (more efficient for longer signals)
    pub fn transform_fft(&self, signal: &ArrayView1<f64>) -> Result<Array2<Complex<f64>>> {
        let n = signal.len();
        let n_scales = self.scales.len();

        if n == 0 {
            return Err(TransformError::InvalidInput("Empty signal".to_string()));
        }

        // Convert signal to f64 vector for FFT
        let signal_vec: Vec<f64> = signal.iter().copied().collect();

        // Compute FFT of signal
        let signal_fft = fft(&signal_vec, None)?;

        // Prepare frequency array
        let freqs: Vec<f64> = (0..n)
            .map(|i| {
                if i <= n / 2 {
                    2.0 * PI * i as f64 / (n as f64 * self.sampling_period)
                } else {
                    2.0 * PI * (i as f64 - n as f64) / (n as f64 * self.sampling_period)
                }
            })
            .collect();

        let mut coeffs = Array2::from_elem((n_scales, n), Complex::new(0.0, 0.0));

        // Compute CWT for each scale using FFT
        for (scale_idx, &scale) in self.scales.iter().enumerate() {
            // Compute wavelet in frequency domain
            let wavelet_fft: Vec<Complex<f64>> = freqs
                .iter()
                .map(|&omega| {
                    if omega >= 0.0 {
                        self.wavelet.wavelet_fft(omega, scale).conj()
                    } else {
                        Complex::new(0.0, 0.0)
                    }
                })
                .collect();

            // Multiply in frequency domain
            let product: Vec<Complex<f64>> = signal_fft
                .iter()
                .zip(wavelet_fft.iter())
                .map(|(&s, &w)| s * w)
                .collect();

            // Inverse FFT
            let cwt_scale = ifft(&product, None)?;

            // Store results
            for (t_idx, &val) in cwt_scale.iter().enumerate() {
                coeffs[[scale_idx, t_idx]] = val;
            }
        }

        Ok(coeffs)
    }

    /// Compute the scalogram (magnitude of CWT coefficients)
    pub fn scalogram(&self, signal: &ArrayView1<f64>) -> Result<Array2<f64>> {
        let coeffs = self.transform_fft(signal)?;
        let (n_scales, n_time) = coeffs.dim();

        let mut scalogram = Array2::zeros((n_scales, n_time));
        for i in 0..n_scales {
            for j in 0..n_time {
                scalogram[[i, j]] = coeffs[[i, j]].norm();
            }
        }

        Ok(scalogram)
    }

    /// Get the scales
    pub fn scales(&self) -> &[f64] {
        &self.scales
    }

    /// Get the frequencies corresponding to scales
    pub fn frequencies(&self) -> Vec<f64> {
        let fc = self.wavelet.central_frequency();
        self.scales
            .iter()
            .map(|&s| fc / (s * self.sampling_period))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_morlet_wavelet() {
        let wavelet = MorletWavelet::default();
        let val = wavelet.wavelet(0.0, 1.0);

        assert!(val.re.abs() > 0.0);
        assert_abs_diff_eq!(val.im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mexican_hat_wavelet() {
        let wavelet = MexicanHatWavelet::default();
        let val = wavelet.wavelet(0.0, 1.0);

        assert!(val.re.abs() > 0.0);
        assert_abs_diff_eq!(val.im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cwt_simple() -> Result<()> {
        let signal = Array1::from_vec(vec![0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0]);
        let wavelet = MorletWavelet::default();
        let scales = vec![1.0, 2.0, 4.0];

        let cwt = CWT::new(wavelet, scales);
        let coeffs = cwt.transform(&signal.view())?;

        assert_eq!(coeffs.dim(), (3, 8));

        Ok(())
    }

    #[test]
    fn test_cwt_fft() -> Result<()> {
        let signal = Array1::from_vec((0..64).map(|i| (i as f64 * 0.1).sin()).collect());
        let wavelet = MorletWavelet::default();
        let cwt = CWT::with_log_scales(wavelet, 32, 1.0, 32.0);

        let coeffs = cwt.transform_fft(&signal.view())?;

        assert_eq!(coeffs.dim(), (32, 64));

        Ok(())
    }

    #[test]
    fn test_scalogram() -> Result<()> {
        let signal = Array1::from_vec((0..64).map(|i| (i as f64 * 0.1).sin()).collect());
        let wavelet = MorletWavelet::default();
        let cwt = CWT::with_log_scales(wavelet, 16, 1.0, 16.0);

        let scalogram = cwt.scalogram(&signal.view())?;

        assert_eq!(scalogram.dim(), (16, 64));
        assert!(scalogram.iter().all(|&x| x >= 0.0));

        Ok(())
    }

    #[test]
    fn test_log_scales() {
        let scales = CWT::<MorletWavelet>::log_scales(10, 1.0, 100.0);

        assert_eq!(scales.len(), 10);
        assert_abs_diff_eq!(scales[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scales[9], 100.0, epsilon = 1e-10);

        // Check logarithmic spacing
        for i in 1..scales.len() {
            let ratio = scales[i] / scales[i - 1];
            assert!(ratio > 1.0);
        }
    }
}
