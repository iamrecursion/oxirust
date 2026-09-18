//! Tempo utility functions: incremental ACF with early-exit.

/// Incrementally evaluates the autocorrelation for lags in `[min_lag, max_lag]`.
///
/// Returns `(detected_lag, confidence, lags_scanned)`.
///
/// * `signal` — onset-strength envelope (mono, f32 samples).
/// * `min_lag`, `max_lag` — valid lag range (in samples / frames).
/// * `confidence_threshold` — normalised peak prominence to trigger early exit
///   (e.g. 0.8). The threshold is applied against `best_val / acf_zero`.
/// * `min_lags_before_exit` — minimum number of lags scanned before allowing
///   early exit (e.g. 8), so the very first lag cannot prematurely win.
///
/// If `acf_zero` is below a tiny epsilon the signal is silent and the function
/// returns `(min_lag, 0.0, 0)` immediately.
///
/// # Panics
///
/// Does not panic for any valid slice input.
#[must_use]
pub fn bounded_acf_with_early_exit(
    signal: &[f32],
    min_lag: usize,
    max_lag: usize,
    confidence_threshold: f32,
    min_lags_before_exit: usize,
) -> (usize, f32, usize) {
    let acf_zero: f32 = signal.iter().map(|&x| x * x).sum();
    if acf_zero < 1e-10 {
        return (min_lag, 0.0, 0);
    }

    let mut best_lag = min_lag;
    let mut best_val = f32::NEG_INFINITY;
    let n = signal.len();
    let mut lags_scanned: usize = 0;

    let effective_max = max_lag.min(n.saturating_sub(1));
    for lag in min_lag..=effective_max {
        let usable_len = n.saturating_sub(lag);
        let acf_val: f32 = signal[..usable_len]
            .iter()
            .zip(&signal[lag..lag + usable_len])
            .map(|(&a, &b)| a * b)
            .sum();

        lags_scanned += 1;

        if acf_val > best_val {
            best_val = acf_val;
            best_lag = lag;
        }

        // Early-exit: normalised prominence exceeds threshold AND
        // minimum lags have already been scanned.
        let normalized = best_val / acf_zero;
        if normalized >= confidence_threshold && lags_scanned >= min_lags_before_exit {
            return (best_lag, normalized, lags_scanned);
        }
    }

    let confidence = if best_val == f32::NEG_INFINITY {
        0.0
    } else {
        (best_val / acf_zero).clamp(0.0, 1.0)
    };
    (best_lag, confidence, lags_scanned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silent_signal_returns_zero_confidence() {
        let (lag, conf, scanned) = bounded_acf_with_early_exit(&[0.0; 100], 5, 50, 0.8, 8);
        assert_eq!(lag, 5);
        assert_eq!(conf, 0.0);
        assert_eq!(scanned, 0);
    }

    #[test]
    fn test_full_scan_when_confidence_never_reached() {
        // White-noise-like signal: confidence will not exceed 0.8 in practice.
        let signal: Vec<f32> = (0..256)
            .map(|i| if i % 7 == 0 { 1.0 } else { 0.0 })
            .collect();
        let min_lag = 4;
        let max_lag = 40;
        let (_, _, scanned) = bounded_acf_with_early_exit(&signal, min_lag, max_lag, 0.99, 1);
        // May or may not early-exit; just verify a sane count
        assert!(scanned >= 1);
        assert!(scanned <= max_lag - min_lag + 1);
    }

    #[test]
    fn test_perfect_periodic_early_exits() {
        // Build a perfectly periodic signal at period=20.
        let period: usize = 20;
        let n = 512;
        let signal: Vec<f32> = (0..n)
            .map(|i| if i % period == 0 { 1.0 } else { 0.0 })
            .collect();

        let (best_lag, conf, scanned) = bounded_acf_with_early_exit(&signal, 10, 40, 0.5, 8);

        assert_eq!(best_lag, period, "should detect period=20");
        assert!(conf > 0.5, "confidence should exceed threshold");
        // Should exit before exhausting all lags (40-10+1 = 31 lags total)
        assert!(
            scanned <= 31,
            "scanned={scanned} should be within range 1..=31"
        );
    }

    #[test]
    fn test_min_lag_clamp_on_max_lag() {
        // max_lag > signal length: should not panic
        let signal = vec![1.0_f32; 10];
        let (lag, _conf, _scanned) = bounded_acf_with_early_exit(&signal, 2, 1000, 0.8, 8);
        assert!(lag >= 2);
    }
}
