//! Benchmark helper utilities
//!
//! This module provides common utilities for benchmarks:
//! - Test data generation
//! - Sample file paths
//! - Performance measurement helpers

#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// Get the path to benchmark test data directory
pub fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("Failed to get parent directory")
        .join("test_data")
}

/// Get path to a specific test file
pub fn test_file(name: &str) -> PathBuf {
    test_data_dir().join(name)
}

/// Generate synthetic video frame data (YUV 4:2:0)
pub fn generate_yuv420_frame(width: usize, height: usize) -> Vec<u8> {
    let y_size = width * height;
    let uv_size = (width / 2) * (height / 2);
    let total_size = y_size + 2 * uv_size;

    let mut data = vec![0u8; total_size];

    // Fill with a gradient pattern for Y plane
    for y in 0..height {
        for x in 0..width {
            data[y * width + x] = ((x + y) % 256) as u8;
        }
    }

    // Fill U and V planes with mid-gray
    for i in y_size..total_size {
        data[i] = 128;
    }

    data
}

/// Generate synthetic RGB frame data
pub fn generate_rgb_frame(width: usize, height: usize) -> Vec<u8> {
    let total_size = width * height * 3;
    let mut data = vec![0u8; total_size];

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            data[idx] = ((x * 255) / width) as u8; // R
            data[idx + 1] = ((y * 255) / height) as u8; // G
            data[idx + 2] = 128; // B
        }
    }

    data
}

/// Generate synthetic audio samples (sine wave)
pub fn generate_audio_samples(sample_rate: usize, duration_ms: usize, frequency: f32) -> Vec<f32> {
    let num_samples = (sample_rate * duration_ms) / 1000;
    let mut samples = Vec::with_capacity(num_samples);

    let angular_freq = 2.0 * std::f32::consts::PI * frequency / sample_rate as f32;

    for i in 0..num_samples {
        samples.push((i as f32 * angular_freq).sin());
    }

    samples
}

/// Generate synthetic PCM audio data (16-bit signed)
pub fn generate_pcm_i16(sample_rate: usize, duration_ms: usize, frequency: f32) -> Vec<i16> {
    let samples = generate_audio_samples(sample_rate, duration_ms, frequency);
    samples.iter().map(|&s| (s * 32767.0) as i16).collect()
}

/// Create a simple Matroska/WebM header for testing
pub fn create_minimal_webm_header() -> Vec<u8> {
    // Minimal WebM header with EBML header
    vec![
        0x1A, 0x45, 0xDF, 0xA3, // EBML
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F, 0x42, 0x86, 0x81,
        0x01, // EBMLVersion = 1
        0x42, 0xF7, 0x81, 0x01, // EBMLReadVersion = 1
        0x42, 0xF2, 0x81, 0x04, // EBMLMaxIDLength = 4
        0x42, 0xF3, 0x81, 0x08, // EBMLMaxSizeLength = 8
        0x42, 0x82, 0x84, 0x77, 0x65, 0x62, 0x6D, // DocType = "webm"
        0x42, 0x87, 0x81, 0x02, // DocTypeVersion = 2
        0x42, 0x85, 0x81, 0x02, // DocTypeReadVersion = 2
    ]
}

/// Create minimal Ogg page for testing
pub fn create_minimal_ogg_page() -> Vec<u8> {
    let mut page = Vec::new();

    // Ogg page header
    page.extend_from_slice(b"OggS"); // Capture pattern
    page.push(0); // Version
    page.push(0); // Header type
    page.extend_from_slice(&[0; 8]); // Granule position
    page.extend_from_slice(&[0; 4]); // Serial number
    page.extend_from_slice(&[0; 4]); // Page sequence
    page.extend_from_slice(&[0; 4]); // Checksum (would need calculation)
    page.push(1); // Number of segments
    page.push(0); // Segment table (0 bytes)

    page
}

/// Check if a test file exists
pub fn test_file_exists(name: &str) -> bool {
    test_file(name).exists()
}

/// Get file size in bytes
pub fn file_size(path: &Path) -> Option<u64> {
    std::fs::metadata(path).ok().map(|m| m.len())
}

/// Calculate throughput in MB/s
pub fn calculate_throughput_mbs(bytes: u64, elapsed_nanos: u64) -> f64 {
    let bytes_per_sec = (bytes as f64 * 1_000_000_000.0) / elapsed_nanos as f64;
    bytes_per_sec / (1024.0 * 1024.0)
}

/// Calculate frames per second
pub fn calculate_fps(num_frames: u64, elapsed_nanos: u64) -> f64 {
    (num_frames as f64 * 1_000_000_000.0) / elapsed_nanos as f64
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_generate_yuv420_frame() {
        let data = generate_yuv420_frame(1920, 1080);
        let expected_size = 1920 * 1080 + 2 * (960 * 540);
        assert_eq!(data.len(), expected_size);
    }

    #[test]
    fn test_generate_rgb_frame() {
        let data = generate_rgb_frame(1920, 1080);
        assert_eq!(data.len(), 1920 * 1080 * 3);
    }

    #[test]
    fn test_generate_audio_samples() {
        let samples = generate_audio_samples(48000, 1000, 440.0);
        assert_eq!(samples.len(), 48000);
        // Check that we have a valid sine wave
        assert!(samples.iter().all(|&s| s >= -1.0 && s <= 1.0));
    }

    #[test]
    fn test_calculate_throughput() {
        let throughput = calculate_throughput_mbs(1024 * 1024, 1_000_000_000);
        assert!((throughput - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_calculate_fps() {
        let fps = calculate_fps(60, 1_000_000_000);
        assert!((fps - 60.0).abs() < 0.01);
    }
}
