//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{AudioPostError, AudioPostResult};
use oxifft::Complex;

use super::types::{DeclickConfig, SpectralSubtractionConfig};

/// Remove impulsive clicks from `samples` and return the cleaned audio.
///
/// Algorithm:
/// 1. Compute first-differences of the input.
/// 2. Estimate the Median Absolute Deviation (MAD) of those differences.
/// 3. Flag any sample whose first-difference exceeds `config.mad_threshold × MAD`.
/// 4. Replace each flagged sample by cubic Hermite interpolation from the
///    `config.interpolation_radius` nearest non-flagged neighbours.
///
/// # Errors
///
/// Returns `AudioPostError::InvalidBufferSize` if the input is empty.
#[allow(clippy::cast_precision_loss)]
pub fn declick(samples: &[f32], config: &DeclickConfig) -> AudioPostResult<Vec<f32>> {
    if samples.is_empty() {
        return Err(AudioPostError::InvalidBufferSize(0));
    }
    let n = samples.len();
    let mut output = samples.to_vec();
    if n < 3 {
        return Ok(output);
    }
    let mut diffs: Vec<f32> = vec![0.0; n];
    for i in 1..n {
        diffs[i] = samples[i] - samples[i - 1];
    }
    let mut sorted_diffs: Vec<f32> = diffs[1..].iter().map(|&x| x.abs()).collect();
    sorted_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = {
        let m = sorted_diffs.len();
        if m == 0 {
            1e-9_f32
        } else if m % 2 == 1 {
            sorted_diffs[m / 2]
        } else {
            (sorted_diffs[m / 2 - 1] + sorted_diffs[m / 2]) / 2.0
        }
    };
    let mad = median.max(1e-9);
    let threshold = config.mad_threshold * mad;
    let mut flagged = vec![false; n];
    for i in 1..n {
        if diffs[i].abs() > threshold {
            flagged[i] = true;
        }
    }
    let radius = config.interpolation_radius;
    let mut i = 0;
    while i < n {
        if flagged[i] {
            let region_start = i;
            while i < n && flagged[i] {
                i += 1;
            }
            let region_end = i;
            let p0_idx = region_start.saturating_sub(1);
            let p1_idx = region_end.min(n - 1);
            let m0 = if p0_idx > 0 && p0_idx + 1 < n {
                let lo = p0_idx.saturating_sub(radius.min(p0_idx));
                let hi = (p0_idx + radius).min(n - 1);
                (samples[hi] - samples[lo]) / (2.0 * (hi - lo).max(1) as f32)
            } else {
                0.0
            };
            let m1 = if p1_idx + 1 < n {
                let lo = p1_idx.saturating_sub(radius.min(p1_idx));
                let hi = (p1_idx + radius).min(n - 1);
                (samples[hi] - samples[lo]) / (2.0 * (hi - lo).max(1) as f32)
            } else {
                0.0
            };
            let v0 = output[p0_idx];
            let v1 = output[p1_idx];
            let span = (p1_idx - p0_idx).max(1) as f32;
            for j in region_start..region_end {
                let t = (j - p0_idx) as f32 / span;
                let t2 = t * t;
                let t3 = t2 * t;
                let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
                let h10 = t3 - 2.0 * t2 + t;
                let h01 = -2.0 * t3 + 3.0 * t2;
                let h11 = t3 - t2;
                output[j] = h00 * v0 + h10 * m0 * span + h01 * v1 + h11 * m1 * span;
            }
        } else {
            i += 1;
        }
    }
    Ok(output)
}
/// Spectral subtraction noise reduction (Boll, 1979) with Wiener post-filter.
///
/// Signal flow per frame:
/// 1. Apply Hann window.
/// 2. Forward FFT (OxiFFT).
/// 3. Compute per-bin signal PSD.
/// 4. Subtract `α × noisePSD`; floor at `β × noisePSD`.
/// 5. Compute Wiener gain = `signalPSD / (signalPSD + noisePSD)`, clamped to `[0, 1]`.
/// 6. Apply gain to complex spectrum; IFFT; overlap-add.
///
/// Noise PSD is estimated from the quietest `noise_percentile` frames.
///
/// # Errors
///
/// Returns `AudioPostError::InvalidBufferSize` if the buffer size is wrong or
/// `AudioPostError::Generic` on internal failures.
#[allow(clippy::cast_precision_loss)]
pub fn spectral_subtract(
    samples: &[f32],
    config: &SpectralSubtractionConfig,
) -> AudioPostResult<Vec<f32>> {
    if samples.is_empty() {
        return Err(AudioPostError::InvalidBufferSize(0));
    }
    if !config.fft_size.is_power_of_two() || config.fft_size < 4 {
        return Err(AudioPostError::InvalidBufferSize(config.fft_size));
    }
    if config.hop_size == 0 || config.hop_size > config.fft_size {
        return Err(AudioPostError::InvalidBufferSize(config.hop_size));
    }
    let n = samples.len();
    let fft_size = config.fft_size;
    let hop = config.hop_size;
    let hann: Vec<f32> = (0..fft_size)
        .map(|i| {
            let t = i as f32 / (fft_size - 1) as f32;
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * t).cos())
        })
        .collect();
    let num_frames = (n + hop - 1) / hop;
    let mut frame_energies: Vec<(f32, usize)> = Vec::with_capacity(num_frames);
    let mut frame_psds: Vec<Vec<f32>> = Vec::with_capacity(num_frames);
    for frame_idx in 0..num_frames {
        let start = frame_idx * hop;
        let input: Vec<Complex<f32>> = (0..fft_size)
            .map(|j| {
                let sample_idx = start + j;
                let s = if sample_idx < n {
                    samples[sample_idx]
                } else {
                    0.0
                };
                Complex::new(s * hann[j], 0.0)
            })
            .collect();
        let spectrum = oxifft::fft(&input);
        let psd: Vec<f32> = spectrum.iter().map(|c| c.norm_sqr()).collect();
        let energy: f32 = psd.iter().sum();
        frame_energies.push((energy, frame_idx));
        frame_psds.push(psd);
    }
    let mut sorted_energies = frame_energies.clone();
    sorted_energies.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let noise_frames = ((num_frames as f32 * config.noise_percentile).ceil() as usize).max(1);
    let mut noise_psd = vec![0.0_f32; fft_size];
    for &(_, fi) in sorted_energies.iter().take(noise_frames) {
        for (bin, &p) in noise_psd.iter_mut().zip(frame_psds[fi].iter()) {
            *bin += p;
        }
    }
    let noise_scale = 1.0 / noise_frames as f32;
    for p in &mut noise_psd {
        *p *= noise_scale;
    }
    let mut output = vec![0.0_f32; n + fft_size];
    let mut ola_weights = vec![0.0_f32; n + fft_size];
    for frame_idx in 0..num_frames {
        let start = frame_idx * hop;
        let input: Vec<Complex<f32>> = (0..fft_size)
            .map(|j| {
                let sample_idx = start + j;
                let s = if sample_idx < n {
                    samples[sample_idx]
                } else {
                    0.0
                };
                Complex::new(s * hann[j], 0.0)
            })
            .collect();
        let mut spectrum = oxifft::fft(&input);
        for (bin_idx, spec_bin) in spectrum.iter_mut().enumerate() {
            let signal_psd = spec_bin.norm_sqr();
            let noise_p = noise_psd[bin_idx];
            let enhanced_psd = (signal_psd - config.alpha * noise_p).max(config.beta * noise_p);
            let gain = if signal_psd + noise_p > 1e-30 {
                (enhanced_psd / (enhanced_psd + noise_p)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            *spec_bin = *spec_bin * gain;
        }
        let recovered = oxifft::ifft(&spectrum);
        for j in 0..fft_size {
            let out_idx = start + j;
            if out_idx < output.len() {
                output[out_idx] += recovered[j].re * hann[j];
                ola_weights[out_idx] += hann[j] * hann[j];
            }
        }
    }
    for (s, &w) in output.iter_mut().zip(ola_weights.iter()) {
        if w >= 0.25 {
            *s /= w;
        } else {
            *s = 0.0;
        }
    }
    output.truncate(n);
    Ok(output)
}
/// Levinson-Durbin AR coefficient estimation from `data` (order `order`).
/// Returns `None` if the autocorrelation matrix is singular.
#[allow(clippy::cast_precision_loss)]
pub(super) fn levinson_durbin(data: &[f32], order: usize) -> Option<Vec<f32>> {
    let n = data.len();
    if n < order + 1 {
        return None;
    }
    let mut r = vec![0.0_f32; order + 1];
    for lag in 0..=order {
        let mut sum = 0.0_f32;
        for i in 0..(n - lag) {
            sum += data[i] * data[i + lag];
        }
        r[lag] = sum / n as f32;
    }
    if r[0].abs() < 1e-15 {
        return None;
    }
    let mut a = vec![0.0_f32; order];
    let mut prev = vec![0.0_f32; order];
    let mut err = r[0];
    for i in 0..order {
        let num: f32 = r[i + 1]
            + a[..i]
                .iter()
                .zip(r[1..=i].iter().rev())
                .map(|(&aj, &rj)| aj * rj)
                .sum::<f32>();
        if err.abs() < 1e-15 {
            return None;
        }
        let k = (-num / err).clamp(-0.999_999, 0.999_999);
        prev[..=i].copy_from_slice(&a[..=i]);
        a[i] = k;
        for j in 0..i {
            a[j] = prev[j] + k * prev[i - 1 - j];
        }
        err *= 1.0 - k * k;
        if err < 1e-30 {
            err = 1e-30;
        }
    }
    Some(a)
}
#[cfg(test)]
mod tests {
    use super::super::types::{
        ArLpcDeclickConfig, ClickRemover, Declicker, Declipper, DenoiseConfig, Denoiser,
        HissRemover, HumRemover, PhaseCorrector, SpectralNoiseReducer, SpectralRepair,
        StereoEnhancer, VinylClickRemover,
    };
    use super::*;
    /// Verify that oxifft::ifft is normalized (roundtrip recovers input).
    #[test]
    fn debug_oxifft_normalization() {
        let x: Vec<oxifft::Complex<f32>> = (0..8_usize)
            .map(|i| oxifft::Complex::new(i as f32, 0.0))
            .collect();
        let f = oxifft::fft(&x);
        let r = oxifft::ifft(&f);
        eprintln!("input[7]={}, recovered[7]={}", x[7].re, r[7].re);
        let diff = (r[7].re - x[7].re).abs();
        assert!(
            diff < 0.01,
            "oxifft::ifft is not normalized! input[7]={}, recovered[7]={}",
            x[7].re,
            r[7].re
        );
    }
    /// Diagnostic: check actual SNR values in spectral subtraction.
    #[test]
    fn debug_spectral_subtract_snr() {
        use std::f32::consts::PI;
        let sample_rate = 48_000_u32;
        let n = 4096_usize;
        let tone_amp = 0.5_f32;
        let noise_amp = 0.3_f32;
        let tone: Vec<f32> = (0..n)
            .map(|i| tone_amp * (2.0 * PI * 1000.0 * i as f32 / sample_rate as f32).sin())
            .collect();
        let noise: Vec<f32> = (0..n)
            .map(|i| {
                let x = (i as f32 * 12.9898 + 78.233).sin() * 43758.5453;
                (x - x.floor() - 0.5) * 2.0 * noise_amp
            })
            .collect();
        let mixed: Vec<f32> = tone.iter().zip(noise.iter()).map(|(t, s)| t + s).collect();
        let config = SpectralSubtractionConfig {
            fft_size: 1024,
            hop_size: 512,
            noise_percentile: 0.05,
            alpha: 2.0,
            beta: 0.05,
        };
        let result = spectral_subtract(&mixed, &config).expect("spectral_subtract ok");
        let tone_rms = (tone.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
        let result_rms = (result.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
        let residual: Vec<f32> = result.iter().zip(tone.iter()).map(|(r, t)| r - t).collect();
        let residual_power: f32 = residual.iter().map(|x| x * x).sum::<f32>() / n as f32;
        let tone_power: f32 = tone.iter().map(|x| x * x).sum::<f32>() / n as f32;
        let noise_power: f32 = noise.iter().map(|x| x * x).sum::<f32>() / n as f32;
        let input_snr = 10.0 * (tone_power / noise_power).log10();
        let output_snr = 10.0 * (tone_power / residual_power.max(1e-30)).log10();
        eprintln!("tone_rms={tone_rms:.4}, result_rms={result_rms:.4}");
        eprintln!("input_snr={input_snr:.2} dB, output_snr={output_snr:.2} dB");
        eprintln!("result[100..105]: {:?}", &result[100..105]);
        eprintln!("tone[100..105]: {:?}", &tone[100..105]);
    }
    #[test]
    fn test_spectral_noise_reducer() {
        let mut reducer = SpectralNoiseReducer::new(48000, 1024).expect("failed to create");
        let noise = vec![0.01_f32; 2048];
        reducer.capture_noise_profile(&noise);
        reducer.set_reduction_amount(0.7);
        assert_eq!(reducer.reduction_amount, 0.7);
    }
    #[test]
    fn test_hiss_remover() {
        let mut hiss_remover = HissRemover::new(48000).expect("failed to create");
        assert!(hiss_remover.set_threshold(-30.0).is_ok());
        hiss_remover.set_reduction(0.6);
        assert_eq!(hiss_remover.reduction, 0.6);
    }
    #[test]
    fn test_hum_remover() {
        let hum_remover = HumRemover::new(48000, 60.0).expect("failed to create");
        let harmonics = hum_remover.get_harmonic_frequencies();
        assert_eq!(harmonics[0], 60.0);
        assert_eq!(harmonics[1], 120.0);
    }
    #[test]
    fn test_invalid_fundamental_freq() {
        assert!(HumRemover::new(48000, 55.0).is_err());
    }
    #[test]
    fn test_click_remover() {
        let mut click_remover = ClickRemover::new(48000).expect("failed to create");
        click_remover.set_sensitivity(0.7);
        let mut audio = vec![0.0_f32; 100];
        audio[50] = 10.0;
        let clicks = click_remover.detect_clicks(&audio);
        assert!(!clicks.is_empty());
    }
    #[test]
    fn test_click_removal() {
        let click_remover = ClickRemover::new(48000).expect("failed to create");
        let mut input = vec![0.0_f32; 100];
        input[50] = 10.0;
        let mut output = vec![0.0_f32; 100];
        click_remover.process(&input, &mut output);
        assert!(output[50].abs() < input[50].abs());
    }
    #[test]
    fn test_declipper() {
        let mut declipper = Declipper::new(48000).expect("failed to create");
        declipper.set_threshold(0.9);
        let mut audio = vec![0.5_f32; 100];
        audio[50] = 1.0;
        let regions = declipper.detect_clipping(&audio);
        assert!(!regions.is_empty());
    }
    #[test]
    fn test_declipping_process() {
        let declipper = Declipper::new(48000).expect("failed to create");
        let mut input = vec![0.0_f32; 100];
        input[50] = 1.0;
        input[51] = 1.0;
        let mut output = vec![0.0_f32; 100];
        declipper.process(&input, &mut output);
        assert!(output[50] < 1.0);
    }
    #[test]
    fn test_spectral_repair() {
        let repair = SpectralRepair::new(48000, 2048).expect("failed to create");
        assert_eq!(repair.fft_size, 2048);
    }
    #[test]
    fn test_phase_corrector() {
        let corrector = PhaseCorrector::new(48000).expect("failed to create");
        let left = vec![1.0_f32; 100];
        let right = vec![1.0_f32; 100];
        let correlation = corrector.analyze_phase_correlation(&left, &right);
        assert!(correlation > 0.0);
    }
    #[test]
    fn test_stereo_enhancer() {
        let mut enhancer = StereoEnhancer::new(48000).expect("failed to create");
        enhancer.set_width(1.5);
        assert_eq!(enhancer.width, 1.5);
    }
    #[test]
    fn test_stereo_enhancement() {
        let enhancer = StereoEnhancer::new(48000).expect("failed to create");
        let left = vec![1.0_f32; 100];
        let right = vec![-1.0_f32; 100];
        let mut out_left = vec![0.0_f32; 100];
        let mut out_right = vec![0.0_f32; 100];
        enhancer.process(&left, &right, &mut out_left, &mut out_right);
        assert!(out_left[0] != 0.0);
    }
    #[test]
    fn test_invalid_fft_size() {
        assert!(SpectralNoiseReducer::new(48000, 1000).is_err());
    }
    #[test]
    fn test_vinyl_click_remover_creation() {
        let remover = VinylClickRemover::new(48000).expect("failed to create");
        assert_eq!(remover.sensitivity, 6.0);
    }
    #[test]
    fn test_vinyl_click_remover_invalid_sr() {
        assert!(VinylClickRemover::new(0).is_err());
    }
    #[test]
    fn test_vinyl_click_remover_detects_click() {
        let remover = VinylClickRemover::new(48000).expect("failed to create");
        let mut audio: Vec<f32> = (0..512).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        audio[256] += 5.0;
        let clicks = remover.detect_clicks(&audio);
        assert!(!clicks.is_empty(), "Should detect the click");
    }
    #[test]
    fn test_vinyl_click_remover_no_false_positives_on_sine() {
        let mut remover = VinylClickRemover::new(48000).expect("failed to create");
        remover.sensitivity = 10.0;
        let audio: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.05).sin()).collect();
        let clicks = remover.detect_clicks(&audio);
        assert!(
            clicks.len() < 5,
            "Too many false positives: {}",
            clicks.len()
        );
    }
    #[test]
    fn test_vinyl_click_remover_output_reduced_at_click() {
        let remover = VinylClickRemover::new(48000).expect("failed to create");
        let mut input: Vec<f32> = vec![0.0f32; 200];
        input[100] += 8.0;
        let mut output = vec![0.0f32; 200];
        remover.process(&input, &mut output);
        assert!(
            output[100].abs() < input[100].abs(),
            "Click should be reduced; input={}, output={}",
            input[100],
            output[100]
        );
    }
    #[test]
    fn test_vinyl_click_remover_empty_input() {
        let remover = VinylClickRemover::new(48000).expect("failed to create");
        let input: Vec<f32> = vec![];
        let clicks = remover.detect_clicks(&input);
        assert!(clicks.is_empty());
    }
    #[test]
    fn test_vinyl_click_remover_process_inplace() {
        let remover = VinylClickRemover::new(48000).expect("failed to create");
        let mut audio = vec![0.0f32; 100];
        audio[50] += 9.0;
        let original_click = audio[50];
        remover.process_inplace(&mut audio);
        assert!(audio[50].abs() < original_click.abs());
    }
    #[test]
    fn test_restoration_noise_reduction_with_synthetic_profile() {
        let mut reducer = SpectralNoiseReducer::new(48000, 1024).expect("failed to create");
        let noise: Vec<f32> = (0..2048)
            .map(|i| ((i as f32 * 17.3).sin()) * 0.02)
            .collect();
        reducer.capture_noise_profile(&noise);
        reducer.set_reduction_amount(0.9);
        let signal: Vec<f32> = (0..1024)
            .map(|i| (i as f32 * 0.05).sin() * 0.3 + noise[i] * 0.1)
            .collect();
        let mut output = vec![0.0f32; 1024];
        reducer.process(&signal, &mut output);
        assert!(reducer.noise_profile.iter().any(|&v| v > 0.0));
    }
    #[test]
    fn test_declicker_creation() {
        let cfg = ArLpcDeclickConfig {
            sample_rate: 16000,
            ..Default::default()
        };
        assert!(Declicker::new(cfg).is_ok());
    }
    #[test]
    fn test_declicker_creation_invalid_sr() {
        let cfg = ArLpcDeclickConfig {
            sample_rate: 0,
            ..Default::default()
        };
        assert!(Declicker::new(cfg).is_err());
    }
    #[test]
    fn test_declicker_creation_invalid_order() {
        let cfg = ArLpcDeclickConfig {
            ar_order: 0,
            sample_rate: 16000,
            ..Default::default()
        };
        assert!(Declicker::new(cfg).is_err());
    }
    #[test]
    fn declick_removes_impulses() {
        use std::f32::consts::PI;
        let sample_rate = 16_000_u32;
        let freq = 1_000.0_f32;
        let mut samples: Vec<f32> = (0..sample_rate as usize)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();
        for &pos in &[1000_usize, 3000, 5000, 7000, 9000] {
            samples[pos] = 1.0;
        }
        let original_energy: f32 = samples.iter().map(|x| x * x).sum();
        let cfg = ArLpcDeclickConfig {
            sample_rate,
            ..Default::default()
        };
        let declicker = Declicker::new(cfg).expect("ok");
        declicker.process(&mut samples).expect("process ok");
        let processed_energy: f32 = samples.iter().map(|x| x * x).sum();
        assert!(
            processed_energy / original_energy > 0.99,
            "Energy ratio {:.4}",
            processed_energy / original_energy
        );
    }
    #[test]
    fn test_declicker_empty_input() {
        let cfg = ArLpcDeclickConfig {
            sample_rate: 16000,
            ..Default::default()
        };
        let declicker = Declicker::new(cfg).expect("ok");
        let mut empty: Vec<f32> = vec![];
        assert!(declicker.process(&mut empty).is_err());
    }
    #[test]
    fn test_levinson_durbin_basics() {
        let data: Vec<f32> = std::iter::once(1.0_f32)
            .chain(std::iter::repeat_n(0.0_f32, 63))
            .collect();
        let coeffs = levinson_durbin(&data, 4).expect("ok");
        assert_eq!(coeffs.len(), 4);
        for &c in &coeffs {
            assert!(c.abs() < 1e-3, "unexpected coeff {c}");
        }
    }
    #[test]
    fn test_denoiser_creation() {
        let cfg = DenoiseConfig {
            sample_rate: 16000,
            ..Default::default()
        };
        assert!(Denoiser::new(cfg).is_ok());
    }
    #[test]
    fn test_denoiser_creation_invalid_sr() {
        let cfg = DenoiseConfig {
            sample_rate: 0,
            ..Default::default()
        };
        assert!(Denoiser::new(cfg).is_err());
    }
    #[test]
    fn test_denoiser_creation_invalid_fft() {
        let cfg = DenoiseConfig {
            fft_size: 1000,
            sample_rate: 16000,
            ..Default::default()
        };
        assert!(Denoiser::new(cfg).is_err());
    }
    #[test]
    fn test_denoiser_reset() {
        let cfg = DenoiseConfig {
            sample_rate: 16000,
            ..Default::default()
        };
        let mut denoiser = Denoiser::new(cfg).expect("ok");
        let samples = vec![0.1_f32; 2048];
        let _ = denoiser.process(&samples).expect("ok");
        denoiser.reset();
        assert!(!denoiser.floor_ready);
        assert!(denoiser.noise_floor.iter().all(|&v| v == 0.0));
    }
    /// Compute SNR of `noisy` relative to `signal` (in dB).
    fn compute_snr(signal: &[f32], noisy: &[f32]) -> f32 {
        let len = signal.len().min(noisy.len());
        if len == 0 {
            return 0.0;
        }
        let sig_power: f32 = signal[..len].iter().map(|x| x * x).sum::<f32>() / len as f32;
        let noise_power: f32 = signal[..len]
            .iter()
            .zip(noisy[..len].iter())
            .map(|(s, n)| (s - n).powi(2))
            .sum::<f32>()
            / len as f32;
        10.0 * (sig_power / noise_power.max(1e-12)).log10()
    }
    #[test]
    fn denoise_reduces_noise() {
        use std::f32::consts::PI;
        let sample_rate = 16_000_u32;
        let freq = 1_000.0_f32;
        let noise_scale = 0.1_f32;
        let n = sample_rate as usize * 2;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();
        let mut seed = 12345_u64;
        let noisy: Vec<f32> = signal
            .iter()
            .map(|&s| {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let noise = ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 2.0 * noise_scale;
                s + noise
            })
            .collect();
        let snr_before = compute_snr(&signal, &noisy);
        let cfg = DenoiseConfig {
            fft_size: 1024,
            overlap_fraction: 0.5,
            noise_estimation_frames: 10,
            oversubtraction_alpha: 2.0,
            spectral_floor_beta: 0.002,
            sample_rate,
        };
        let mut denoiser = Denoiser::new(cfg).expect("ok");
        let denoised = denoiser.process(&noisy).expect("process ok");
        let min_len = signal.len().min(denoised.len());
        let snr_after = compute_snr(&signal[..min_len], &denoised[..min_len]);
        assert!(
            snr_after > snr_before + 10.0,
            "SNR before={snr_before:.1} dB, after={snr_after:.1} dB; expected +10 dB improvement"
        );
        assert!(
            snr_after > 30.0,
            "SNR after denoising: {snr_after:.1} dB (target > 30 dB)"
        );
    }
    #[test]
    fn test_denoiser_empty_input() {
        let cfg = DenoiseConfig {
            sample_rate: 16000,
            ..Default::default()
        };
        let mut denoiser = Denoiser::new(cfg).expect("ok");
        assert!(denoiser.process(&[]).is_err());
    }
}
