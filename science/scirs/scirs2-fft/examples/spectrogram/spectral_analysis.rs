//! Spectral analysis example using FFT and spectrograms
//!
//! This example demonstrates how to perform spectral analysis on time-domain signals
//! using various FFT functions and visualizing them with spectrograms.

use scirs2_core::Complex64;
use scirs2_fft::{
    fft, fftfreq, fftshift, get_window, spectrogram, stft, window::Window, hilbert,
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Spectral Analysis Example");
    println!("-----------------------\n");

    // Generate a signal with multiple frequency components
    let fs = 1000.0; // Sampling frequency: 1 kHz
    let t_max = 1.0; // 1 second signal
    let n_samples = (t_max * fs) as usize;

    let t: Vec<f64> = (0..n_samples).map(|i| i as f64 / fs).collect();
    
    // Create a signal with multiple frequency components:
    // - 50 Hz sine wave
    // - 150 Hz sine wave with increasing amplitude
    // - 250 Hz sine wave that appears in the middle
    let mut signal = Vec::with_capacity(n_samples);
    
    for (i, &ti) in t.iter().enumerate() {
        let progress = i as f64 / n_samples as f64;
        let freq1 = 50.0;
        let freq2 = 150.0;
        let freq3 = 250.0;
        
        let component1 = (2.0 * PI * freq1 * ti).sin();
        let component2 = (2.0 * PI * freq2 * ti).sin() * progress;
        
        // The third component only appears in the middle half of the signal
        let component3 = if progress > 0.25 && progress < 0.75 {
            // Fade in and out gradually
            let window_pos = (progress - 0.25) * 2.0;
            let window_amplitude = if window_pos <= 0.5 {
                window_pos * 2.0
            } else {
                (1.0 - window_pos) * 2.0
            };
            (2.0 * PI * freq3 * ti).sin() * window_amplitude
        } else {
            0.0
        };
        
        signal.push(component1 + component2 + component3);
    }
    
    println!("Signal generated: {} samples at {} Hz", signal.len(), fs);
    
    // === Basic FFT ===
    println!("\n1. Basic FFT Analysis");
    
    // Compute the FFT of the signal
    let spectrum = fft(&signal, None)?;
    let freqs = fftfreq(n_samples, 1.0 / fs);
    let shifted_freqs = fftshift(&freqs);
    let shifted_spectrum = fftshift(&spectrum);
    
    // Display some FFT results
    println!("  FFT computed: {} frequency bins", spectrum.len());
    println!("  First few frequency bins:");
    for i in 0..5 {
        println!("  - {:.1} Hz: magnitude = {:.3}", 
            freqs[i], 
            spectrum[i].norm());
    }
    
    // Find peaks in the spectrum
    let mut max_magnitude = 0.0;
    let mut max_freq = 0.0;
    
    for (i, &freq) in freqs.iter().enumerate().take(n_samples / 2) {
        let magnitude = spectrum[i].norm();
        if magnitude > max_magnitude {
            max_magnitude = magnitude;
            max_freq = freq;
        }
    }
    
    println!("  Dominant frequency component: {:.1} Hz", max_freq);
    
    // === Hilbert Transform ===
    println!("\n2. Hilbert Transform");
    
    // Compute the analytic signal using Hilbert transform
    let analytic_signal = hilbert(&signal)?;
    
    // Compute instantaneous amplitude (envelope)
    let envelope: Vec<f64> = analytic_signal
        .iter()
        .map(|c| (c.re.powi(2) + c.im.powi(2)).sqrt())
        .collect();
    
    // Compute instantaneous phase
    let inst_phase: Vec<f64> = analytic_signal
        .iter()
        .map(|c| c.im.atan2(c.re))
        .collect();
    
    // Compute instantaneous frequency (derivative of phase)
    let mut inst_freq = Vec::with_capacity(n_samples - 1);
    for i in 0..(n_samples - 1) {
        let mut phase_diff = inst_phase[i + 1] - inst_phase[i];
        
        // Handle phase wrapping
        if phase_diff > PI {
            phase_diff -= 2.0 * PI;
        } else if phase_diff < -PI {
            phase_diff += 2.0 * PI;
        }
        
        // Convert to Hz: phase_diff / (2π * dt)
        inst_freq.push(phase_diff * fs / (2.0 * PI));
    }
    
    println!("  Hilbert transform computed");
    println!("  Signal envelope: min={:.2}, max={:.2}", 
        envelope.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        envelope.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
    
    // === Short-Time Fourier Transform (STFT) ===
    println!("\n3. Short-Time Fourier Transform (STFT)");
    
    // Compute STFT
    let windowsize = 256;
    let hop_length = 128;
    let (stft_freqs, stft_times, stft_result) = stft(
        &signal,
        Window::Hann,
        windowsize,
        Some(windowsize - hop_length),
        None,
        Some(fs),
        None,
        None,
    )?;
    
    println!("  STFT computed: {} frequency bins, {} time frames", 
        stft_freqs.len(), 
        stft_times.len());
    
    // === Spectrogram ===
    println!("\n4. Spectrogram");
    
    // Compute spectrogram (returns power spectrum)
    let (spec_freqs, spec_times, spec_result) = spectrogram(
        &signal,
        Some(fs),
        Some(Window::Hamming),
        Some(256),
        Some(128),
        None,
        None,
        Some(false),
        None,
    )?;
    
    println!("  Spectrogram computed: {} frequency bins, {} time frames", 
        spec_freqs.len(), 
        spec_times.len());
    
    // Display some spectrogram values
    println!("  First few time-frequency bins (dB scale):");
    for i in 0..3 {
        for j in 0..3 {
            let power_db = 10.0 * spec_result[[i, j]].log10();
            println!("  - ({:.1} Hz, {:.3} s): {:.1} dB", 
                spec_freqs[i], 
                spec_times[j],
                power_db);
        }
    }
    
    // === Window Functions ===
    println!("\n5. Window Functions");
    
    let window_length = 64;
    let windows = vec![
        Window::Hann, 
        Window::Hamming, 
        Window::Blackman, 
        Window::Kaiser(2.0),
        Window::Rectangular,
    ];
    
    println!("  Window functions compared:");
    for window_type in windows {
        let window = get_window(window_type, window_length)?;
        
        // Calculate effective bandwidth
        let enbw = window.iter().map(|x| x.powi(2)).sum::<f64>() / 
                  window.iter().sum::<f64>().powi(2) * window_length as f64;
        
        println!("  - {:?}: ENBW = {:.2}", window_type, enbw);
    }
    
    println!("\nSpectral analysis complete!");
    Ok(())
}
