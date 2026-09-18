//! Corner detection example.
//!
//! This example demonstrates three corner detection algorithms:
//! - Harris corner detector
//! - Shi-Tomasi (Good Features to Track)
//! - FAST (Features from Accelerated Segment Test)
//!
//! The example uses synthetic test images to showcase each algorithm's
//! ability to detect corner features without requiring external files.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example corner_detection
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::unnecessary_wraps)]

use oximedia_cv::detect::{CornerDetector, FastDetector, HarrisDetector, ShiTomasiDetector};

/// Create a synthetic test image with corner-like patterns.
///
/// The image contains a checkerboard pattern which has corners at
/// the intersections of black and white squares.
fn create_test_image() -> (Vec<u8>, u32, u32) {
    let width = 64u32;
    let height = 64u32;
    let mut image = vec![0u8; (width * height) as usize];

    // Create a checkerboard pattern (8x8 squares)
    let square_size = 8;
    for y in 0..height {
        for x in 0..width {
            let square_x = x / square_size;
            let square_y = y / square_size;
            let is_white = (square_x + square_y) % 2 == 0;
            let idx = (y * width + x) as usize;
            image[idx] = if is_white { 255 } else { 0 };
        }
    }

    (image, width, height)
}

/// Create a synthetic test image with a gradient pattern.
///
/// This image contains horizontal and vertical edges which should
/// produce different corner responses than the checkerboard.
fn create_gradient_image() -> (Vec<u8>, u32, u32) {
    let width = 64u32;
    let height = 64u32;
    let mut image = vec![0u8; (width * height) as usize];

    // Create a gradient pattern
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let val_x = (x * 255 / width) as u8;
            let val_y = (y * 255 / height) as u8;
            image[idx] = ((u16::from(val_x) + u16::from(val_y)) / 2) as u8;
        }
    }

    (image, width, height)
}

/// Create a synthetic test image with L-shaped corners.
///
/// This image contains clear L-shaped corners that should be
/// strongly detected by all corner detection algorithms.
fn create_l_corners_image() -> (Vec<u8>, u32, u32) {
    let width = 64u32;
    let height = 64u32;
    let mut image = vec![128u8; (width * height) as usize];

    // Draw L-shaped corners
    // Top-left L
    for y in 8..24 {
        for x in 8..12 {
            let idx = (y * width + x) as usize;
            image[idx] = 255;
        }
    }
    for y in 20..24 {
        for x in 8..24 {
            let idx = (y * width + x) as usize;
            image[idx] = 255;
        }
    }

    // Bottom-right inverted L
    for y in 40..56 {
        for x in 52..56 {
            let idx = (y * width + x) as usize;
            image[idx] = 0;
        }
    }
    for y in 40..44 {
        for x in 40..56 {
            let idx = (y * width + x) as usize;
            image[idx] = 0;
        }
    }

    (image, width, height)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OxiMedia Corner Detection Example");
    println!("==================================\n");

    // Test Harris detector
    println!("1. Harris Corner Detector");
    println!("   Uses corner response R = det(M) - k*trace(M)^2");
    println!();

    let detector = HarrisDetector::new()
        .with_block_size(5)
        .with_k(0.04)
        .with_threshold(0.01);

    let (image, width, height) = create_test_image();
    match detector.detect(&image, width, height) {
        Ok(corners) => {
            println!("   Checkerboard pattern:");
            println!("   Found {} corners", corners.len());
            if !corners.is_empty() {
                println!("   Top 5 corners by response:");
                for (i, corner) in corners.iter().take(5).enumerate() {
                    println!(
                        "     {}: ({:2}, {:2}) response={:.4}",
                        i + 1,
                        corner.x,
                        corner.y,
                        corner.response
                    );
                }
            }
        }
        Err(e) => {
            println!("   Error: {e}");
        }
    }
    println!();

    // Test with L-corners
    let (image2, width2, height2) = create_l_corners_image();
    match detector.detect(&image2, width2, height2) {
        Ok(corners) => {
            println!("   L-shaped corners pattern:");
            println!("   Found {} corners", corners.len());
            if !corners.is_empty() {
                println!("   Top 5 corners by response:");
                for (i, corner) in corners.iter().take(5).enumerate() {
                    println!(
                        "     {}: ({:2}, {:2}) response={:.4}",
                        i + 1,
                        corner.x,
                        corner.y,
                        corner.response
                    );
                }
            }
        }
        Err(e) => {
            println!("   Error: {e}");
        }
    }
    println!();

    // Test Shi-Tomasi detector
    println!("2. Shi-Tomasi Corner Detector (Good Features to Track)");
    println!("   Uses minimum eigenvalue of structure tensor");
    println!();

    let shi_tomasi = ShiTomasiDetector::new()
        .with_block_size(5)
        .with_quality_level(0.01)
        .with_min_distance(10.0);

    match shi_tomasi.detect(&image, width, height) {
        Ok(corners) => {
            println!("   Checkerboard pattern:");
            println!("   Found {} corners", corners.len());
            if !corners.is_empty() {
                println!("   Top 5 corners by response:");
                for (i, corner) in corners.iter().take(5).enumerate() {
                    println!(
                        "     {}: ({:2}, {:2}) response={:.4}",
                        i + 1,
                        corner.x,
                        corner.y,
                        corner.response
                    );
                }
            }
        }
        Err(e) => {
            println!("   Error: {e}");
        }
    }
    println!();

    // Test FAST detector
    println!("3. FAST Corner Detector (Features from Accelerated Segment Test)");
    println!("   Uses circle-based intensity comparison");
    println!();

    let fast = FastDetector::new().with_threshold(20).with_nms(true);

    match fast.detect(&image, width, height) {
        Ok(corners) => {
            println!("   Checkerboard pattern:");
            println!("   Found {} corners", corners.len());
            if !corners.is_empty() {
                println!("   Top 10 corners by response:");
                for (i, corner) in corners.iter().take(10).enumerate() {
                    println!(
                        "     {}: ({:2}, {:2}) response={:.1}",
                        i + 1,
                        corner.x,
                        corner.y,
                        corner.response
                    );
                }
            }
        }
        Err(e) => {
            println!("   Error: {e}");
        }
    }
    println!();

    // Test FAST with gradient image
    let (gradient_img, gw, gh) = create_gradient_image();
    match fast.detect(&gradient_img, gw, gh) {
        Ok(corners) => {
            println!("   Gradient pattern:");
            println!("   Found {} corners", corners.len());
            println!("   (Gradients typically have fewer corners than checkerboards)");
        }
        Err(e) => {
            println!("   Error: {e}");
        }
    }
    println!();

    println!("Corner detection test completed successfully!");
    println!("\nComparison:");
    println!("- Harris: Good general-purpose detector, sensitive to noise");
    println!("- Shi-Tomasi: Better for tracking, selects most stable corners");
    println!("- FAST: Very fast, good for real-time applications");
    println!("\nAll detectors can find corners in the synthetic test images.");

    Ok(())
}
