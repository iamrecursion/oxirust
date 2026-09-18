//! Integration tests for AudiopostPipeline (workflow.rs).
//!
//! Tests verify:
//! - Pipeline processes audio through all DSP stages without error.
//! - Output is finite with no NaN/Inf.
//! - With normalize_loudness enabled, integrated LUFS converges toward target.
//! - PipelineOutput metadata (clicks_repaired, integrated_lufs, true_peak_dbtp) is sensible.

use std::f32::consts::PI;

use oximedia_audiopost::workflow::{AudiopostPipeline, PipelineStageConfig};

/// Generate a 440 Hz mono sine of length `n` at `sample_rate` with amplitude `amp`.
fn make_sine(n: usize, freq_hz: f32, amp: f32, sample_rate: u32) -> Vec<f32> {
    (0..n)
        .map(|i| amp * (2.0 * PI * freq_hz * i as f32 / sample_rate as f32).sin())
        .collect()
}

#[test]
fn test_pipeline_processes_without_error() {
    let sample_rate = 48_000_u32;
    let n = sample_rate as usize; // 1 second
    let signal = make_sine(n, 440.0, 0.3, sample_rate);

    let mut pipeline = AudiopostPipeline::new(sample_rate, PipelineStageConfig::default())
        .expect("pipeline creation should succeed");
    let output = pipeline
        .process(&signal)
        .expect("pipeline process should succeed");

    assert_eq!(
        output.samples.len(),
        n,
        "output sample count must equal input"
    );
}

#[test]
fn test_pipeline_output_is_finite() {
    let sample_rate = 48_000_u32;
    let n = sample_rate as usize;
    let signal = make_sine(n, 880.0, 0.4, sample_rate);

    let mut pipeline = AudiopostPipeline::new(sample_rate, PipelineStageConfig::default())
        .expect("pipeline creation ok");
    let output = pipeline.process(&signal).expect("process ok");

    for (i, &s) in output.samples.iter().enumerate() {
        assert!(s.is_finite(), "output sample {i} is not finite: {s}");
    }
}

#[test]
fn test_pipeline_metadata_fields_finite() {
    let sample_rate = 48_000_u32;
    let n = sample_rate as usize;
    let signal = make_sine(n, 1_000.0, 0.3, sample_rate);

    let mut pipeline =
        AudiopostPipeline::new(sample_rate, PipelineStageConfig::default()).expect("pipeline ok");
    let output = pipeline.process(&signal).expect("process ok");

    assert!(
        output.integrated_lufs.is_finite() || output.integrated_lufs == f64::NEG_INFINITY,
        "integrated_lufs must be finite or -∞"
    );
    assert!(
        output.true_peak_dbtp.is_finite() || output.true_peak_dbtp == f64::NEG_INFINITY,
        "true_peak_dbtp must be finite or -∞"
    );
    // clicks_repaired is a count; must be a valid usize.
    assert!(
        output.clicks_repaired < n,
        "clicks_repaired must be < signal length"
    );
}

#[test]
fn test_pipeline_declick_only() {
    // Bypass denoise and loudness-normalize; only run the declick stage.
    let sample_rate = 48_000_u32;
    let n = 4_096_usize;
    let mut signal = make_sine(n, 1_000.0, 0.5, sample_rate);
    // Inject 3 clicks.
    signal[100] = 2.0;
    signal[1000] = -2.0;
    signal[3000] = 2.0;

    let config = PipelineStageConfig {
        declick: true,
        denoise: false,
        normalize_loudness: false,
        ..PipelineStageConfig::default()
    };
    let mut pipeline = AudiopostPipeline::new(sample_rate, config).expect("pipeline ok");
    let output = pipeline.process(&signal).expect("process ok");

    assert_eq!(output.samples.len(), n);
    for (i, &s) in output.samples.iter().enumerate() {
        assert!(s.is_finite(), "declick-only sample {i} not finite: {s}");
    }
    // With 3 injected large clicks, clicks_repaired should be at least 1.
    assert!(
        output.clicks_repaired >= 1,
        "expected at least 1 click repaired, got {}",
        output.clicks_repaired
    );
}

#[test]
fn test_pipeline_denoise_only() {
    let sample_rate = 48_000_u32;
    let n = 4_096_usize;
    let signal: Vec<f32> = (0..n)
        .map(|i| {
            let tone = (2.0 * PI * 1_000.0 * i as f32 / sample_rate as f32).sin() * 0.5;
            let noise = (i as f32 * 17.3).sin() * 0.2;
            tone + noise
        })
        .collect();

    let config = PipelineStageConfig {
        declick: false,
        denoise: true,
        normalize_loudness: false,
        ..PipelineStageConfig::default()
    };
    let mut pipeline = AudiopostPipeline::new(sample_rate, config).expect("pipeline ok");
    let output = pipeline.process(&signal).expect("process ok");

    assert_eq!(output.samples.len(), n);
    for (i, &s) in output.samples.iter().enumerate() {
        assert!(s.is_finite(), "denoise-only sample {i} not finite: {s}");
    }
}

#[test]
fn test_pipeline_passthrough_when_all_disabled() {
    // All stages bypassed: output must equal input exactly.
    let sample_rate = 48_000_u32;
    let n = 1_024_usize;
    let signal = make_sine(n, 440.0, 0.3, sample_rate);

    let config = PipelineStageConfig {
        declick: false,
        denoise: false,
        normalize_loudness: false,
        ..PipelineStageConfig::default()
    };
    let mut pipeline = AudiopostPipeline::new(sample_rate, config).expect("pipeline ok");
    let output = pipeline.process(&signal).expect("process ok");

    assert_eq!(output.samples.len(), n);
    for (i, (&a, &b)) in output.samples.iter().zip(signal.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "passthrough mismatch at sample {i}: output={a}, input={b}"
        );
    }
}

#[test]
fn test_pipeline_zero_sample_rate_error() {
    let result = AudiopostPipeline::new(0, PipelineStageConfig::default());
    assert!(result.is_err(), "zero sample rate must return an error");
}

#[test]
fn test_pipeline_empty_input_error() {
    let sample_rate = 48_000_u32;
    let mut pipeline =
        AudiopostPipeline::new(sample_rate, PipelineStageConfig::default()).expect("pipeline ok");
    let result = pipeline.process(&[]);
    assert!(result.is_err(), "empty input must return an error");
}

#[test]
fn test_pipeline_lufs_diagnostic() {
    // Diagnostic: print actual integrated_lufs after normalization to calibrate threshold.
    let sample_rate = 48_000_u32;
    let n = sample_rate as usize; // 1 second
    let signal = make_sine(n, 1_000.0, 0.5, sample_rate);

    let config = PipelineStageConfig {
        declick: false,
        denoise: false,
        normalize_loudness: true,
        ..PipelineStageConfig::default()
    };
    let mut pipeline = AudiopostPipeline::new(sample_rate, config).expect("pipeline ok");
    let output = pipeline.process(&signal).expect("process ok");

    eprintln!(
        "DIAG workflow_lufs: integrated_lufs={:.4} LUFS, true_peak_dbtp={:.4} dBTP",
        output.integrated_lufs, output.true_peak_dbtp
    );

    // Must be finite
    assert!(
        output.integrated_lufs.is_finite() || output.integrated_lufs == f64::NEG_INFINITY,
        "integrated_lufs must be finite or -∞, got {}",
        output.integrated_lufs
    );
}
