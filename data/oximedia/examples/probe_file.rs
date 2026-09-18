//! Probe media container format.
//!
//! This example demonstrates how to detect media container formats from file headers.
//! It shows how to use the `probe_format` function to identify `WebM`, Matroska, Ogg,
//! FLAC, and WAV files without requiring external files.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example probe_file
//! ```

use oximedia::prelude::*;

/// Generate synthetic `WebM` file header.
///
/// `WebM` files start with EBML header (0x1A 0x45 0xDF 0xA3).
fn create_webm_header() -> Vec<u8> {
    vec![
        0x1A, 0x45, 0xDF, 0xA3, // EBML header
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F, 0x42, 0x86, 0x81,
        0x01, // EBML version
    ]
}

/// Generate synthetic Ogg file header.
///
/// Ogg files start with "`OggS`" magic bytes.
fn create_ogg_header() -> Vec<u8> {
    vec![
        b'O', b'g', b'g', b'S', // Magic bytes
        0x00, // Version
        0x02, // Header type
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Granule position
    ]
}

/// Generate synthetic FLAC file header.
///
/// FLAC files start with "fLaC" magic bytes.
fn create_flac_header() -> Vec<u8> {
    vec![
        b'f', b'L', b'a', b'C', // Magic bytes
        0x00, // Last metadata block flag
        0x00, 0x00, 0x22, // Metadata block length
        0x00, 0x00, 0x00, 0x00,
    ]
}

/// Generate synthetic WAV file header.
///
/// WAV files start with "RIFF" followed by "WAVE".
fn create_wav_header() -> Vec<u8> {
    vec![
        b'R', b'I', b'F', b'F', // RIFF header
        0x24, 0x00, 0x00, 0x00, // File size - 8
        b'W', b'A', b'V', b'E', // WAVE marker
        b'f', b'm', b't', b' ', // fmt chunk
    ]
}

/// Generate unknown format data.
fn create_unknown_header() -> Vec<u8> {
    vec![0xFF; 16]
}

fn main() -> OxiResult<()> {
    println!("OxiMedia Container Format Probe Example");
    println!("========================================\n");

    // Test WebM/Matroska detection
    println!("Testing WebM/Matroska detection:");
    let webm_data = create_webm_header();
    match probe_format(&webm_data) {
        Ok(result) => {
            println!("  Format: {:?}", result.format);
            println!("  Confidence: {:.1}%", result.confidence * 100.0);
        }
        Err(e) => {
            println!("  Error: {e}");
        }
    }
    println!();

    // Test Ogg detection
    println!("Testing Ogg detection:");
    let ogg_data = create_ogg_header();
    match probe_format(&ogg_data) {
        Ok(result) => {
            println!("  Format: {:?}", result.format);
            println!("  Confidence: {:.1}%", result.confidence * 100.0);
        }
        Err(e) => {
            println!("  Error: {e}");
        }
    }
    println!();

    // Test FLAC detection
    println!("Testing FLAC detection:");
    let flac_data = create_flac_header();
    match probe_format(&flac_data) {
        Ok(result) => {
            println!("  Format: {:?}", result.format);
            println!("  Confidence: {:.1}%", result.confidence * 100.0);
        }
        Err(e) => {
            println!("  Error: {e}");
        }
    }
    println!();

    // Test WAV detection
    println!("Testing WAV detection:");
    let wav_data = create_wav_header();
    match probe_format(&wav_data) {
        Ok(result) => {
            println!("  Format: {:?}", result.format);
            println!("  Confidence: {:.1}%", result.confidence * 100.0);
        }
        Err(e) => {
            println!("  Error: {e}");
        }
    }
    println!();

    // Test unknown format
    println!("Testing unknown format:");
    let unknown_data = create_unknown_header();
    match probe_format(&unknown_data) {
        Ok(result) => {
            println!("  Format: {:?}", result.format);
            println!("  Confidence: {:.1}%", result.confidence * 100.0);
        }
        Err(e) => {
            println!("  Error: {e}");
        }
    }
    println!();

    println!("Probe test completed successfully!");
    println!("\nNote: All tests use synthetic headers for demonstration.");
    println!("In production, you would read from actual media files.");

    Ok(())
}
