//! Enhanced Frequency Domain Decomposition (EFDD) with damping estimation.
//!
//! EFDD extends the classical FDD algorithm by:
//! 1. Estimating the cross-PSD matrix via the batched Welch estimator.
//! 2. Performing an SVD at each frequency bin to extract singular values and
//!    mode shapes.
//! 3. Peak-picking from the first singular value curve to identify modal
//!    frequencies.
//! 4. Building an SDOF spectral bell around each peak by tracking bins whose
//!    mode shape correlates well (MAC ≥ threshold) with the shape at the peak.
//! 5. Extracting a damping estimate from either:
//!    - **Half-power bandwidth** (fast, frequency-domain), or
//!    - **Exponential decay of the IFFT** correlation function (more accurate
//!      for well-separated modes).
//!
//! # References
//! - Brincker, R., Zhang, L. & Andersen, P. (2001). "Modal identification of
//!   output-only systems using frequency domain decomposition." *Smart Materials
//!   and Structures*, 10(3), 441–445.
//! - Jacobsen, N.J., Andersen, P. & Brincker, R. (2006). "Using EFDD as a
//!   robust technique to harmonic excitation in OMA." *ISMA Proc.*

use crate::error::{SignalError, SignalResult};
use crate::welch_batch::{ifft_real_onesided, BatchedWelch, WelchConfig};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Public configuration
// ---------------------------------------------------------------------------

/// Configuration for the Enhanced FDD algorithm.
#[derive(Debug, Clone)]
pub struct EfddConfig {
    /// Welch PSD configuration used to compute the cross-PSD matrix.
    pub welch_config: WelchConfig,
    /// Number of structural modes to extract.
    pub n_modes: usize,
    /// MAC threshold (0 < threshold ≤ 1) for including a frequency bin in
    /// the SDOF bell around a peak.  Default `0.8`.
    pub mac_threshold: f64,
    /// Half-width of the SDOF frequency window in bins.  The actual search
    /// range is `[peak - sdof_window_bins, peak + sdof_window_bins]`.
    pub sdof_window_bins: usize,
}

impl Default for EfddConfig {
    fn default() -> Self {
        Self {
            welch_config: WelchConfig::default(),
            n_modes: 4,
            mac_threshold: 0.8,
            sdof_window_bins: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Public result types
// ---------------------------------------------------------------------------

/// Modal parameters identified for a single structural mode.
#[derive(Debug, Clone)]
pub struct EfddMode {
    /// Natural frequency in Hz.
    pub natural_frequency: f64,
    /// Viscous damping ratio (0 < ζ < 1).
    pub damping_ratio: f64,
    /// Mode shape as the first left singular vector at the peak frequency,
    /// normalised to unit Euclidean norm.
    pub mode_shape: Vec<f64>,
    /// Zero-based index of this mode in the sorted list.
    pub modal_index: usize,
}

/// Output of the EFDD algorithm.
#[derive(Debug, Clone)]
pub struct EfddResult {
    /// Identified structural modes, sorted by ascending natural frequency.
    pub modes: Vec<EfddMode>,
    /// Frequency axis in Hz corresponding to `singular_values`.
    pub freqs: Vec<f64>,
    /// `singular_values[m][k]` — first singular value at frequency bin `k`
    /// for mode rank `m`.  Index 0 is the dominant (largest) curve.
    pub singular_values: Vec<Vec<f64>>,
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Identify modal parameters from multi-channel ambient vibration data using
/// Enhanced FDD.
///
/// # Arguments
/// * `data` — `data[channel][sample]`, all channels the same length.
/// * `config` — Algorithm configuration (Welch + EFDD parameters).
///
/// # Returns
/// [`EfddResult`] sorted by ascending natural frequency.
///
/// # Errors
/// Returns [`SignalError`] if the data layout is invalid or the signal is
/// too short to produce any Welch segments.
pub fn efdd(data: &[Vec<f64>], config: EfddConfig) -> SignalResult<EfddResult> {
    let n_channels = data.len();
    if n_channels == 0 {
        return Err(SignalError::InvalidArgument(
            "data must contain at least one channel".to_string(),
        ));
    }

    // Compute cross-PSD matrix via batched Welch
    let bw = BatchedWelch::new(config.welch_config.clone())?;
    let welch_res = bw.compute(data)?;

    let freqs = welch_res.freqs.clone();
    let n_bins = freqs.len();

    // Unpack CPSD (guaranteed to be Some because we called compute, not psd_only)
    let cpsd = welch_res
        .cpsd
        .ok_or_else(|| SignalError::ComputationError("CPSD not computed".to_string()))?;

    // At each frequency bin compute the first singular value and vector
    // via power iteration on Re(G(f)) (sufficient for real structural modes).
    let mut sv1 = vec![0.0f64; n_bins];
    let mut u1: Vec<Vec<f64>> = Vec::with_capacity(n_bins);

    for bin in 0..n_bins {
        let (sv, u) = first_sv_power_iter(&cpsd, bin, n_channels, 30);
        sv1[bin] = sv;
        u1.push(u);
    }

    // Peak-picking
    let fs = config.welch_config.fs;
    let f_min = freqs.get(1).copied().unwrap_or(0.0); // skip DC
    let f_max = fs / 2.0;
    let n_peaks = config.n_modes;
    let peak_indices = peak_picking(&sv1, &freqs, n_peaks, f_min, f_max);

    let n_found = peak_indices.len();

    let mut modes: Vec<EfddMode> = Vec::with_capacity(n_found);

    for (m, &peak_idx) in peak_indices.iter().enumerate() {
        let fn_hz = freqs[peak_idx];
        let mode_shape = unit_norm(&u1[peak_idx]);

        // Build SDOF bell: bins within window whose mode shape MAC ≥ threshold
        let sdof = extract_sdof_spectrum(
            &sv1,
            &u1,
            peak_idx,
            config.sdof_window_bins,
            config.mac_threshold,
        );

        // Primary damping estimate: half-power bandwidth
        let damp_hp = estimate_damping_halfpower(&sdof, &freqs, peak_idx);

        // Secondary estimate: IFFT decay (more accurate for clean modes)
        let damp_decay = damping_from_decay(&sdof, config.welch_config.fs, fn_hz);

        // Use the average of both estimates; clamp to physical range
        let damping_ratio = ((damp_hp + damp_decay) / 2.0).clamp(1e-5, 0.5);

        modes.push(EfddMode {
            natural_frequency: fn_hz,
            damping_ratio,
            mode_shape,
            modal_index: m,
        });
    }

    // Sort by ascending natural frequency
    modes.sort_by(|a, b| {
        a.natural_frequency
            .partial_cmp(&b.natural_frequency)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (m, mode) in modes.iter_mut().enumerate() {
        mode.modal_index = m;
    }

    Ok(EfddResult {
        modes,
        freqs,
        singular_values: vec![sv1],
    })
}

// ---------------------------------------------------------------------------
// Algorithm helpers (public for testing / reuse)
// ---------------------------------------------------------------------------

/// Find up to `n_peaks` peaks in the first singular value curve `sv`.
///
/// A bin is a peak if it is strictly larger than both immediate neighbours
/// and is a local maximum within the frequency range `[f_min, f_max]`.
/// Peaks are returned sorted by descending singular value magnitude so that
/// the strongest modes come first; the result is then truncated to `n_peaks`.
///
/// Returns a `Vec` of bin indices into `sv` / `freqs`.
pub fn peak_picking(
    sv: &[f64],
    freqs: &[f64],
    n_peaks: usize,
    f_min: f64,
    f_max: f64,
) -> Vec<usize> {
    if sv.len() < 3 || n_peaks == 0 {
        return vec![];
    }

    let mut candidates: Vec<(usize, f64)> = Vec::new();

    for i in 1..sv.len() - 1 {
        let f = freqs.get(i).copied().unwrap_or(0.0);
        if f < f_min || f > f_max {
            continue;
        }
        if sv[i] > sv[i - 1] && sv[i] > sv[i + 1] {
            candidates.push((i, sv[i]));
        }
    }

    // Sort by descending singular value
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(n_peaks);

    // Return indices in frequency order (ascending)
    let mut indices: Vec<usize> = candidates.into_iter().map(|(idx, _)| idx).collect();
    indices.sort_unstable();
    indices
}

/// Extract the SDOF spectral bell around a peak bin.
///
/// For each frequency bin in `[peak_idx - half_width, peak_idx + half_width]`
/// the MAC between the mode shape at that bin and the mode shape at the peak
/// is computed.  Bins with MAC ≥ `mac_threshold` contribute their singular
/// value; all others are set to zero.
///
/// # Arguments
/// * `sv` — first singular value curve (length `n_bins`).
/// * `u1` — mode shape vectors, one per frequency bin.
/// * `freq_idx` — index of the peak frequency bin.
/// * `half_width` — search half-window in bins.
/// * `mac_threshold` — minimum MAC to include a bin.
pub fn extract_sdof_spectrum(
    sv: &[f64],
    u1: &[Vec<f64>],
    freq_idx: usize,
    half_width: usize,
    mac_threshold: f64,
) -> Vec<f64> {
    let n_bins = sv.len();
    let mut sdof = vec![0.0f64; n_bins];

    let start = freq_idx.saturating_sub(half_width);
    let end = (freq_idx + half_width + 1).min(n_bins);

    let u_peak = &u1[freq_idx];
    for bin in start..end {
        let mac = compute_mac(u_peak, &u1[bin]);
        if mac >= mac_threshold {
            sdof[bin] = sv[bin];
        }
    }
    sdof
}

/// Estimate damping ratio using the half-power bandwidth (–3 dB) method.
///
/// Finds the two bins on either side of the peak where the PSD drops to
/// half the peak value (–3 dB) and computes:
/// `ζ = (f2 - f1) / (2 * fn)`
/// where `f1 < fn < f2` are the half-power frequencies.
///
/// Falls back to a conservative 1 % estimate if insufficient data.
pub fn estimate_damping_halfpower(sdof: &[f64], freqs: &[f64], peak_idx: usize) -> f64 {
    let n = sdof.len();
    if n < 3 || peak_idx >= n {
        return 0.01;
    }

    let peak_val = sdof[peak_idx];
    if peak_val <= 0.0 {
        return 0.01;
    }
    let half_val = peak_val * 0.5; // -3 dB in power

    let fn_hz = freqs.get(peak_idx).copied().unwrap_or(1.0);
    if fn_hz <= 0.0 {
        return 0.01;
    }

    // Find f1: scan left from peak_idx
    let mut f1 = freqs.get(0).copied().unwrap_or(0.0);
    for i in (0..peak_idx).rev() {
        if sdof[i] <= half_val {
            // Interpolate between i and i+1
            let fa = freqs.get(i).copied().unwrap_or(0.0);
            let fb = freqs.get(i + 1).copied().unwrap_or(fn_hz);
            let ya = sdof[i];
            let yb = sdof[i + 1];
            if (yb - ya).abs() > f64::EPSILON {
                f1 = fa + (half_val - ya) / (yb - ya) * (fb - fa);
            } else {
                f1 = (fa + fb) / 2.0;
            }
            break;
        }
    }

    // Find f2: scan right from peak_idx
    let mut f2 = freqs.last().copied().unwrap_or(fn_hz);
    for i in peak_idx + 1..n {
        if sdof[i] <= half_val {
            let fa = freqs.get(i - 1).copied().unwrap_or(fn_hz);
            let fb = freqs.get(i).copied().unwrap_or(fn_hz);
            let ya = sdof[i - 1];
            let yb = sdof[i];
            if (yb - ya).abs() > f64::EPSILON {
                f2 = fa + (half_val - ya) / (yb - ya) * (fb - fa);
            } else {
                f2 = (fa + fb) / 2.0;
            }
            break;
        }
    }

    let delta_f = (f2 - f1).max(0.0);
    let zeta = delta_f / (2.0 * fn_hz);
    zeta.clamp(1e-5, 0.5)
}

/// Estimate damping ratio from the exponential decay of the SDOF correlation
/// function obtained by inverse-FFT of the SDOF spectral bell.
///
/// Algorithm:
/// 1. IFFT the `sdof_spectrum` to get the auto-correlation function `r(t)`.
/// 2. Find positive peaks of `r(t)` (local maxima).
/// 3. Log-linear regression on `ln |r(t_k)|` vs. `t_k` gives the decay rate
///    `α ≈ -ζ * ω_n`.
/// 4. Return `ζ = -α / ω_n`.
///
/// Falls back to 0.01 if fewer than two peaks are found.
pub fn damping_from_decay(sdof_spectrum: &[f64], fs: f64, peak_freq: f64) -> f64 {
    let n_bins = sdof_spectrum.len();
    if n_bins < 4 || peak_freq <= 0.0 || fs <= 0.0 {
        return 0.01;
    }

    // Build one-sided complex spectrum (real-valued = purely magnitude)
    let spectrum: Vec<(f64, f64)> = sdof_spectrum.iter().map(|&v| (v, 0.0)).collect();
    let n_full = 2 * (n_bins - 1);
    if n_full == 0 {
        return 0.01;
    }
    let corr = ifft_real_onesided(&spectrum, n_full);

    // Find positive local maxima in the first half of the correlation function
    let dt = 1.0 / fs;
    let half_len = corr.len() / 2;
    let mut peaks: Vec<(f64, f64)> = Vec::new(); // (t, |r(t)|)

    for i in 1..half_len.saturating_sub(1) {
        let v = corr[i].abs();
        if v > corr[i - 1].abs() && v > corr[i + 1].abs() && v > 1e-30 {
            let t = i as f64 * dt;
            peaks.push((t, v));
        }
    }

    if peaks.len() < 2 {
        return 0.01;
    }

    // Log-linear regression: ln(v) = ln(A) + alpha * t  where alpha = -zeta * omega_n
    let n = peaks.len() as f64;
    let mean_t = peaks.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_lv = peaks.iter().map(|p| p.1.ln()).sum::<f64>() / n;

    let num: f64 = peaks
        .iter()
        .map(|p| (p.0 - mean_t) * (p.1.ln() - mean_lv))
        .sum();
    let den: f64 = peaks.iter().map(|p| (p.0 - mean_t).powi(2)).sum();

    if den.abs() < 1e-30 {
        return 0.01;
    }

    let alpha = num / den; // should be negative for decaying signal
    let omega_n = 2.0 * PI * peak_freq;

    if omega_n.abs() < 1e-10 {
        return 0.01;
    }

    let zeta = (-alpha / omega_n).clamp(1e-5, 0.5);
    zeta
}

/// Modal Assurance Criterion (MAC) between two mode-shape vectors.
///
/// Returns a value in `[0, 1]` where 1 means the vectors are collinear and
/// 0 means they are orthogonal.
pub fn compute_mac(u1: &[f64], u2: &[f64]) -> f64 {
    let n = u1.len().min(u2.len());
    if n == 0 {
        return 0.0;
    }
    let num: f64 = u1.iter().zip(u2.iter()).map(|(a, b)| a * b).sum::<f64>();
    let d1: f64 = u1.iter().map(|v| v * v).sum::<f64>();
    let d2: f64 = u2.iter().map(|v| v * v).sum::<f64>();
    let denom = d1 * d2;
    if denom < f64::EPSILON {
        return 0.0;
    }
    (num * num / denom).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Return the first singular value and left singular vector of the real part
/// of the CPSD matrix at a given frequency bin via power iteration.
fn first_sv_power_iter(
    cpsd: &Vec<Vec<Vec<(f64, f64)>>>,
    bin: usize,
    n_ch: usize,
    n_iter: usize,
) -> (f64, Vec<f64>) {
    if n_ch == 0 {
        return (0.0, vec![]);
    }

    // Initialise with a uniform vector
    let init_val = 1.0 / (n_ch as f64).sqrt();
    let mut v = vec![init_val; n_ch];

    for _ in 0..n_iter {
        let mut w = vec![0.0f64; n_ch];
        for i in 0..n_ch {
            for j in 0..n_ch {
                // Use only Re(G_{ij}) — sufficient for well-separated modes
                let re = cpsd
                    .get(i)
                    .and_then(|row| row.get(j))
                    .and_then(|col| col.get(bin))
                    .map(|c| c.0)
                    .unwrap_or(0.0);
                w[i] += re * v[j];
            }
        }
        let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < f64::EPSILON {
            return (0.0, v);
        }
        for x in v.iter_mut().zip(w.iter()) {
            *x.0 = x.1 / norm;
        }
    }

    // Rayleigh quotient for eigenvalue estimate
    let mut rq_num = 0.0f64;
    for i in 0..n_ch {
        let mut av_i = 0.0f64;
        for j in 0..n_ch {
            let re = cpsd
                .get(i)
                .and_then(|row| row.get(j))
                .and_then(|col| col.get(bin))
                .map(|c| c.0)
                .unwrap_or(0.0);
            av_i += re * v[j];
        }
        rq_num += v[i] * av_i;
    }

    (rq_num.max(0.0), v)
}

/// Normalise a vector to unit Euclidean norm.  Returns the vector unchanged
/// if its norm is numerically zero.
fn unit_norm(v: &[f64]) -> Vec<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < f64::EPSILON {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::welch_batch::WelchConfig;
    use std::f64::consts::PI;

    // -------------------------------------------------------------------
    // Helper: generate a damped multi-DOF vibration signal
    // -------------------------------------------------------------------

    /// x_k(t) = sum_m  phi_m_k * sin(2*pi*fn_m*t) * exp(-zeta_m * omega_m * t)
    fn make_2dof_signal(
        n: usize,
        fs: f64,
        fn1: f64,
        zeta1: f64,
        fn2: f64,
        zeta2: f64,
        phi: [[f64; 2]; 2], // phi[mode][channel]
    ) -> Vec<Vec<f64>> {
        let mut ch1 = vec![0.0f64; n];
        let mut ch2 = vec![0.0f64; n];
        for i in 0..n {
            let t = i as f64 / fs;
            let w1 = 2.0 * PI * fn1;
            let w2 = 2.0 * PI * fn2;
            ch1[i] = phi[0][0] * (w1 * t).sin() * (-zeta1 * w1 * t).exp()
                + phi[1][0] * (w2 * t).sin() * (-zeta2 * w2 * t).exp();
            ch2[i] = phi[0][1] * (w1 * t).sin() * (-zeta1 * w1 * t).exp()
                + phi[1][1] * (w2 * t).sin() * (-zeta2 * w2 * t).exp();
        }
        vec![ch1, ch2]
    }

    /// 2-DOF system: EFDD should find both frequencies within 5%.
    #[test]
    fn test_efdd_2dof_freq_accuracy() {
        let fs = 512.0f64;
        let n = 4096usize;
        let fn1 = 20.0f64;
        let fn2 = 60.0f64;
        let zeta1 = 0.03f64;
        let zeta2 = 0.05f64;
        // Well-separated mode shapes
        let phi = [[1.0, 0.8], [0.6, -1.0]];
        let data = make_2dof_signal(n, fs, fn1, zeta1, fn2, zeta2, phi);

        let cfg = EfddConfig {
            welch_config: WelchConfig {
                nperseg: 256,
                noverlap: 128,
                window: crate::welch_batch::WelchWindow::Hann,
                fs,
                detrend: false,
            },
            n_modes: 2,
            mac_threshold: 0.7,
            sdof_window_bins: 8,
        };

        let result = efdd(&data, cfg).expect("efdd should succeed");
        assert_eq!(result.modes.len(), 2, "should find 2 modes");

        let found_freqs: Vec<f64> = result.modes.iter().map(|m| m.natural_frequency).collect();
        let targets = [fn1, fn2];
        let tol = 0.05; // 5% tolerance
        for &tf in &targets {
            let close = found_freqs.iter().any(|&ff| (ff - tf).abs() / tf <= tol);
            assert!(close, "no mode found near {tf} Hz; found: {found_freqs:?}");
        }
    }

    /// `damping_from_decay` on a clean exponential envelope gives correct ζ.
    ///
    /// This test uses a well-resolved SDOF spectrum: the bandwidth `2*ζ*fn`
    /// spans many frequency bins so the IFFT decay is clearly visible.
    #[test]
    fn test_damping_from_decay_clean_exponential() {
        // Use a large nperseg and moderate fn so the SDOF bell is well-resolved:
        //   fn=10 Hz, zeta=0.05 → bandwidth = 2*0.05*10 = 1.0 Hz
        //   fs=200 Hz, nperseg=2048 → resolution = 200/2048 ≈ 0.098 Hz (≈10 bins per Hz)
        // → ~10 bins per bandwidth: well-resolved.
        let fs = 200.0f64;
        let fn_hz = 10.0f64;
        let zeta_true = 0.05f64;
        let nperseg = 2048usize;
        let n_bins = nperseg / 2 + 1;

        // Lorentzian (SDOF mechanical admittance PSD)
        let omega_n = 2.0 * PI * fn_hz;
        let mut sdof = vec![0.0f64; n_bins];
        for k in 0..n_bins {
            let f = k as f64 * fs / nperseg as f64;
            let w = 2.0 * PI * f;
            let denom =
                (omega_n * omega_n - w * w).powi(2) + (2.0 * zeta_true * omega_n * w).powi(2);
            sdof[k] = if denom > 1e-30 { 1.0 / denom } else { 0.0 };
        }

        let zeta_est = damping_from_decay(&sdof, fs, fn_hz);
        // For a well-resolved SDOF spectrum the IFFT method should be within 10%
        let tol = 0.10;
        assert!(
            (zeta_est - zeta_true).abs() / zeta_true <= tol,
            "damping estimate {zeta_est:.4} too far from true {zeta_true:.4}"
        );
    }

    /// MAC of identical vectors is 1.
    #[test]
    fn test_mac_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let mac = compute_mac(&v, &v);
        assert!(
            (mac - 1.0).abs() < 1e-12,
            "MAC of identical vectors should be 1, got {mac}"
        );
    }

    /// MAC of orthogonal vectors is 0.
    #[test]
    fn test_mac_orthogonal_vectors() {
        let u = vec![1.0, 0.0, 0.0];
        let v = vec![0.0, 1.0, 0.0];
        let mac = compute_mac(&u, &v);
        assert!(
            mac.abs() < 1e-12,
            "MAC of orthogonal vectors should be 0, got {mac}"
        );
    }

    /// `peak_picking` returns the global maximum when there is one clear peak.
    #[test]
    fn test_peak_picking_global_max() {
        // Smooth bell with a single obvious maximum
        let n_bins = 64usize;
        let fs = 256.0f64;
        let peak_bin = 20usize;
        let mut sv = vec![0.0f64; n_bins];
        for k in 0..n_bins {
            let d = k as f64 - peak_bin as f64;
            sv[k] = 1.0 / (1.0 + d * d * 0.2);
        }
        let freqs: Vec<f64> = (0..n_bins)
            .map(|k| k as f64 * fs / (n_bins * 2 - 2) as f64)
            .collect();

        let f_min = freqs[1];
        let f_max = freqs[n_bins - 1];
        let peaks = peak_picking(&sv, &freqs, 1, f_min, f_max);
        assert_eq!(peaks.len(), 1, "should find exactly 1 peak");
        assert_eq!(peaks[0], peak_bin, "peak should be at bin {peak_bin}");
    }

    /// Damping from a purely decaying exponential matches well.
    #[test]
    fn test_damping_from_decay_exponential_decay() {
        // Build a spectrum whose IFFT is a damped cosine: exp(-alpha*t)*cos(omega_n*t)
        // The one-sided PSD corresponds to a Lorentzian centred on fn.
        let fs = 512.0f64;
        let fn_hz = 25.0f64;
        let zeta_true = 0.06f64;
        let nperseg = 512usize;
        let n_bins = nperseg / 2 + 1;
        let omega_n = 2.0 * PI * fn_hz;

        let mut sdof = vec![0.0f64; n_bins];
        for k in 0..n_bins {
            let f = k as f64 * fs / nperseg as f64;
            let w = 2.0 * PI * f;
            let denom =
                (omega_n * omega_n - w * w).powi(2) + (2.0 * zeta_true * omega_n * w).powi(2);
            sdof[k] = if denom > 1e-30 {
                omega_n.powi(2) / denom
            } else {
                0.0
            };
        }

        let zeta_est = damping_from_decay(&sdof, fs, fn_hz);
        // Require within 15% relative error for this test
        let err = (zeta_est - zeta_true).abs() / zeta_true;
        assert!(
            err <= 0.15,
            "decay damping estimate {zeta_est:.4} deviates {:.1}% from true {zeta_true:.4}",
            err * 100.0
        );
    }
}
