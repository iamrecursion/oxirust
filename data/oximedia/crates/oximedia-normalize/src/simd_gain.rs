//! SIMD-friendly batch gain application.
//!
//! Provides optimized gain application routines that process samples in chunks
//! of 4 or 8, giving the compiler explicit vectorization opportunities.
//! These routines are pure safe Rust that rely on autovectorization hints
//! (tight loops with known bounds and no aliasing).
//!
//! # Usage
//!
//! ```rust
//! use oximedia_normalize::simd_gain::apply_gain_f32_batch;
//!
//! let input = vec![0.5_f32; 1024];
//! let mut output = vec![0.0_f32; 1024];
//! apply_gain_f32_batch(&input, &mut output, 0.8);
//! assert!((output[0] - 0.4).abs() < 1e-6);
//! ```

/// Apply a linear gain to f32 samples, processing 8 at a time.
///
/// The output buffer must be at least as large as the input buffer.
/// Remaining samples (< 8) are processed individually.
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
pub fn apply_gain_f32_batch(input: &[f32], output: &mut [f32], gain: f32) {
    assert!(
        output.len() >= input.len(),
        "output buffer too small: {} < {}",
        output.len(),
        input.len()
    );

    let n = input.len();
    let chunks = n / 8;
    let remainder = n % 8;

    // Process 8 samples at a time (autovectorization hint)
    for c in 0..chunks {
        let base = c * 8;
        let i_chunk = &input[base..base + 8];
        let o_chunk = &mut output[base..base + 8];
        o_chunk[0] = i_chunk[0] * gain;
        o_chunk[1] = i_chunk[1] * gain;
        o_chunk[2] = i_chunk[2] * gain;
        o_chunk[3] = i_chunk[3] * gain;
        o_chunk[4] = i_chunk[4] * gain;
        o_chunk[5] = i_chunk[5] * gain;
        o_chunk[6] = i_chunk[6] * gain;
        o_chunk[7] = i_chunk[7] * gain;
    }

    // Handle remaining samples
    let tail_start = chunks * 8;
    for i in 0..remainder {
        output[tail_start + i] = input[tail_start + i] * gain;
    }
}

/// Apply a linear gain to f32 samples in-place, processing 8 at a time.
pub fn apply_gain_f32_inplace_batch(samples: &mut [f32], gain: f32) {
    let n = samples.len();
    let chunks = n / 8;
    let remainder = n % 8;

    for c in 0..chunks {
        let base = c * 8;
        let chunk = &mut samples[base..base + 8];
        chunk[0] *= gain;
        chunk[1] *= gain;
        chunk[2] *= gain;
        chunk[3] *= gain;
        chunk[4] *= gain;
        chunk[5] *= gain;
        chunk[6] *= gain;
        chunk[7] *= gain;
    }

    let tail_start = chunks * 8;
    for i in 0..remainder {
        samples[tail_start + i] *= gain;
    }
}

/// Apply a linear gain to f64 samples, processing 4 at a time.
///
/// The output buffer must be at least as large as the input buffer.
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
pub fn apply_gain_f64_batch(input: &[f64], output: &mut [f64], gain: f64) {
    assert!(
        output.len() >= input.len(),
        "output buffer too small: {} < {}",
        output.len(),
        input.len()
    );

    let n = input.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // Process 4 samples at a time (autovectorization hint for f64 / 256-bit SIMD)
    for c in 0..chunks {
        let base = c * 4;
        let i_chunk = &input[base..base + 4];
        let o_chunk = &mut output[base..base + 4];
        o_chunk[0] = i_chunk[0] * gain;
        o_chunk[1] = i_chunk[1] * gain;
        o_chunk[2] = i_chunk[2] * gain;
        o_chunk[3] = i_chunk[3] * gain;
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        output[tail_start + i] = input[tail_start + i] * gain;
    }
}

/// Apply a linear gain to f64 samples in-place, processing 4 at a time.
pub fn apply_gain_f64_inplace_batch(samples: &mut [f64], gain: f64) {
    let n = samples.len();
    let chunks = n / 4;
    let remainder = n % 4;

    for c in 0..chunks {
        let base = c * 4;
        let chunk = &mut samples[base..base + 4];
        chunk[0] *= gain;
        chunk[1] *= gain;
        chunk[2] *= gain;
        chunk[3] *= gain;
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        samples[tail_start + i] *= gain;
    }
}

/// Apply per-sample gain from a gain envelope (f32), processing 8 at a time.
///
/// `gains` must have the same length as `samples`.
pub fn apply_gain_envelope_f32(samples: &mut [f32], gains: &[f32]) {
    assert_eq!(
        samples.len(),
        gains.len(),
        "samples and gains must have the same length"
    );

    let n = samples.len();
    let chunks = n / 8;
    let remainder = n % 8;

    for c in 0..chunks {
        let base = c * 8;
        let s = &mut samples[base..base + 8];
        let g = &gains[base..base + 8];
        s[0] *= g[0];
        s[1] *= g[1];
        s[2] *= g[2];
        s[3] *= g[3];
        s[4] *= g[4];
        s[5] *= g[5];
        s[6] *= g[6];
        s[7] *= g[7];
    }

    let tail_start = chunks * 8;
    for i in 0..remainder {
        samples[tail_start + i] *= gains[tail_start + i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_gain_f32_batch_unity() {
        let input = vec![0.5_f32; 100];
        let mut output = vec![0.0_f32; 100];
        apply_gain_f32_batch(&input, &mut output, 1.0);
        for (i, &o) in output.iter().enumerate() {
            assert!((o - 0.5).abs() < 1e-7, "mismatch at {i}: {o}");
        }
    }

    #[test]
    fn test_apply_gain_f32_batch_scale() {
        let input = vec![0.5_f32; 17]; // not a multiple of 8
        let mut output = vec![0.0_f32; 17];
        apply_gain_f32_batch(&input, &mut output, 2.0);
        for (i, &o) in output.iter().enumerate() {
            assert!((o - 1.0).abs() < 1e-6, "mismatch at {i}: {o}");
        }
    }

    #[test]
    fn test_apply_gain_f32_batch_empty() {
        let input: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];
        apply_gain_f32_batch(&input, &mut output, 2.0);
        assert!(output.is_empty());
    }

    #[test]
    fn test_apply_gain_f32_inplace_batch() {
        let mut samples = vec![0.25_f32; 33];
        apply_gain_f32_inplace_batch(&mut samples, 4.0);
        for (i, &s) in samples.iter().enumerate() {
            assert!((s - 1.0).abs() < 1e-6, "mismatch at {i}: {s}");
        }
    }

    #[test]
    fn test_apply_gain_f64_batch_unity() {
        let input = vec![0.5_f64; 100];
        let mut output = vec![0.0_f64; 100];
        apply_gain_f64_batch(&input, &mut output, 1.0);
        for (i, &o) in output.iter().enumerate() {
            assert!((o - 0.5).abs() < 1e-12, "mismatch at {i}: {o}");
        }
    }

    #[test]
    fn test_apply_gain_f64_batch_scale() {
        let input = vec![0.25_f64; 13]; // not a multiple of 4
        let mut output = vec![0.0_f64; 13];
        apply_gain_f64_batch(&input, &mut output, 4.0);
        for (i, &o) in output.iter().enumerate() {
            assert!((o - 1.0).abs() < 1e-12, "mismatch at {i}: {o}");
        }
    }

    #[test]
    fn test_apply_gain_f64_inplace_batch() {
        let mut samples = vec![0.5_f64; 11];
        apply_gain_f64_inplace_batch(&mut samples, 0.5);
        for (i, &s) in samples.iter().enumerate() {
            assert!((s - 0.25).abs() < 1e-12, "mismatch at {i}: {s}");
        }
    }

    #[test]
    fn test_apply_gain_envelope_f32() {
        let mut samples = vec![1.0_f32; 20];
        let gains: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
        apply_gain_envelope_f32(&mut samples, &gains);
        for (i, &s) in samples.iter().enumerate() {
            let expected = i as f32 * 0.1;
            assert!(
                (s - expected).abs() < 1e-6,
                "mismatch at {i}: expected {expected}, got {s}"
            );
        }
    }

    #[test]
    fn test_apply_gain_envelope_f32_empty() {
        let mut samples: Vec<f32> = vec![];
        let gains: Vec<f32> = vec![];
        apply_gain_envelope_f32(&mut samples, &gains);
    }

    #[test]
    fn test_batch_matches_scalar_f32() {
        // Verify batch result matches simple scalar loop
        let input: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let gain = 1.5_f32;

        let expected: Vec<f32> = input.iter().map(|&s| s * gain).collect();

        let mut batch_output = vec![0.0_f32; 100];
        apply_gain_f32_batch(&input, &mut batch_output, gain);

        for (i, (&e, &b)) in expected.iter().zip(batch_output.iter()).enumerate() {
            assert!(
                (e - b).abs() < 1e-7,
                "mismatch at {i}: expected {e}, got {b}"
            );
        }
    }

    #[test]
    fn test_batch_matches_scalar_f64() {
        let input: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
        let gain = 1.5_f64;

        let expected: Vec<f64> = input.iter().map(|&s| s * gain).collect();

        let mut batch_output = vec![0.0_f64; 100];
        apply_gain_f64_batch(&input, &mut batch_output, gain);

        for (i, (&e, &b)) in expected.iter().zip(batch_output.iter()).enumerate() {
            assert!(
                (e - b).abs() < 1e-12,
                "mismatch at {i}: expected {e}, got {b}"
            );
        }
    }

    #[test]
    fn test_apply_gain_f32_batch_exactly_8() {
        let input = vec![0.5_f32; 8];
        let mut output = vec![0.0_f32; 8];
        apply_gain_f32_batch(&input, &mut output, 3.0);
        for &o in &output {
            assert!((o - 1.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_apply_gain_f64_batch_exactly_4() {
        let input = vec![0.5_f64; 4];
        let mut output = vec![0.0_f64; 4];
        apply_gain_f64_batch(&input, &mut output, 3.0);
        for &o in &output {
            assert!((o - 1.5).abs() < 1e-12);
        }
    }
}
