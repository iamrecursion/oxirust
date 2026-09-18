use scirs2_core::Complex64;
use scirs2_fft::{hfft, ihfft};
use std::f64::consts::PI;

#[allow(dead_code)]
fn main() {
    // Part 1: Real to Complex (IHFFT)
    // Create a simple real-valued signal (a cosine wave)
    let n = 16;
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / n as f64).cos())
        .collect();

    println!("Original real signal:");
    for (i, val) in signal.iter().enumerate() {
        println!("  x[{i}] = {val:.6}");
    }

    // Convert to complex spectrum with ihfft, using "backward" normalization
    let spectrum = ihfft(&signal, None, Some("backward")).expect("Operation failed");

    println!("\nComplex spectrum (Hermitian-symmetric):");
    for (i, val) in spectrum.iter().enumerate() {
        let re = if val.re.abs() < 1e-10 { 0.0 } else { val.re };
        let im = if val.im.abs() < 1e-10 { 0.0 } else { val.im };
        println!("  X[{i}] = {re:.6} + {im:.6}i");
    }

    println!("\nVerifying Hermitian symmetry properties:");

    // Check first element is real
    println!("First element real part: {:.6}", spectrum[0].re);
    println!(
        "First element imag part: {:.6} (should be zero)",
        spectrum[0].im
    );

    // Check Hermitian symmetry: X[i] = conj(X[n-i])
    for i in 1..spectrum.len() / 2 {
        let conj_idx = spectrum.len() - i;
        if conj_idx < spectrum.len() {
            println!("X[{}] = {:.6} + {:.6}i", i, spectrum[i].re, spectrum[i].im);
            println!(
                "X[{}] = {:.6} + {:.6}i",
                conj_idx, spectrum[conj_idx].re, spectrum[conj_idx].im
            );
            println!(
                "Conjugate check: {:.6e}",
                (spectrum[i].re - spectrum[conj_idx].re).abs()
                    + (spectrum[i].im + spectrum[conj_idx].im).abs()
            );
            println!();
        }
    }

    // Part 2: Complex to Real (HFFT)
    println!("\n------------- Complex to Real Conversion -------------\n");

    // Create a complex signal with perfect Hermitian symmetry for demonstration
    // For a real signal of length n, the hfft input should have (n/2)+1 elements
    // with proper Hermitian symmetry for elements up to n_input-1 (where n_input is the complex signal length)
    let n_freq = n / 2 + 1;
    let mut complex_signal = vec![Complex64::new(0.0, 0.0); n_freq];

    // Set frequency components
    complex_signal[0] = Complex64::new(0.0, 0.0); // DC component (real)
    complex_signal[1] = Complex64::new(0.5, 0.0); // Frequency 1 (real for this example)

    println!("Input complex signal (Hermitian-symmetric):");
    for (i, val) in complex_signal.iter().enumerate() {
        println!("  X[{}] = {:.6} + {:.6}i", i, val.re, val.im);
    }

    // Apply hfft to get real output, using "backward" normalization
    match hfft(&complex_signal, Some(n), Some("backward")) {
        Ok(real_result) => {
            println!("\nReal output signal:");
            for (i, val) in real_result.iter().enumerate() {
                println!("  x[{i}] = {val:.6}");
            }

            // Expected output is a cosine wave at frequency 1
            let expected: Vec<f64> = (0..n)
                .map(|i| (2.0 * PI * i as f64 / n as f64).cos())
                .collect();

            println!("\nComparison with expected cosine wave:");
            let mut max_error: f64 = 0.0;
            for i in 0..n {
                let error = (real_result[i] - expected[i]).abs();
                max_error = max_error.max(error);
                println!(
                    "  x[{}] = {:.6} (expected: {:.6}, error: {:.6e})",
                    i, real_result[i], expected[i], error
                );
            }
            println!("\nMaximum error: {max_error:.6e}");
        }
        Err(e) => {
            println!("\nError in hfft: {e:?}");
        }
    }
}
