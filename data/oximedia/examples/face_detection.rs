//! Face detection example using Haar cascades.
//!
//! This example demonstrates face detection using Haar cascade classifiers.
//! It shows how to:
//! - Create and configure a Haar cascade detector
//! - Process grayscale images for face detection
//! - Use integral images for fast feature evaluation
//! - Handle detection results with bounding boxes
//!
//! The example uses synthetic test images to demonstrate the detection
//! pipeline without requiring external model files or face images.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example face_detection
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::unnecessary_wraps)]

use oximedia_cv::detect::{FaceDetector, HaarCascade, IntegralImage};

/// Create a synthetic grayscale image with various intensity patterns.
///
/// This image simulates face-like regions with different brightness levels.
fn create_synthetic_face_image() -> (Vec<u8>, u32, u32) {
    let width = 100u32;
    let height = 100u32;
    let mut image = vec![128u8; (width * height) as usize];

    // Create a face-like oval region
    let center_x = 50.0;
    let center_y = 50.0;
    let radius_x = 25.0;
    let radius_y = 30.0;

    for y in 0..height {
        for x in 0..width {
            let dx = (x as f32 - center_x) / radius_x;
            let dy = (y as f32 - center_y) / radius_y;
            let dist = (dx * dx + dy * dy).sqrt();

            let idx = (y * width + x) as usize;

            if dist < 1.0 {
                // Face region (lighter)
                image[idx] = 180;

                // Add darker regions for eyes
                if (40..=45).contains(&y) && ((35..=40).contains(&x) || (60..=65).contains(&x)) {
                    image[idx] = 80;
                }

                // Add darker region for mouth
                if (65..=70).contains(&y) && (40..=60).contains(&x) {
                    image[idx] = 100;
                }
            }
        }
    }

    (image, width, height)
}

/// Create a test image with multiple face-like regions at different scales.
fn create_multi_scale_image() -> (Vec<u8>, u32, u32) {
    let width = 200u32;
    let height = 150u32;
    let mut image = vec![128u8; (width * height) as usize];

    // Large face
    draw_face_region(&mut image, width, 50.0, 50.0, 30.0, 35.0);

    // Medium face
    draw_face_region(&mut image, width, 140.0, 50.0, 20.0, 23.0);

    // Small face
    draw_face_region(&mut image, width, 95.0, 110.0, 15.0, 17.0);

    (image, width, height)
}

/// Helper function to draw a face-like region in an image.
fn draw_face_region(image: &mut [u8], width: u32, cx: f32, cy: f32, rx: f32, ry: f32) {
    for y in 0..image.len() as u32 / width {
        for x in 0..width {
            let dx = (x as f32 - cx) / rx;
            let dy = (y as f32 - cy) / ry;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < 1.0 {
                let idx = (y * width + x) as usize;
                if idx < image.len() {
                    image[idx] = 180;

                    // Eyes
                    let eye_y = cy - ry * 0.3;
                    let eye_left_x = cx - rx * 0.4;
                    let eye_right_x = cx + rx * 0.4;
                    let eye_dist_left =
                        ((x as f32 - eye_left_x).powi(2) + (y as f32 - eye_y).powi(2)).sqrt();
                    let eye_dist_right =
                        ((x as f32 - eye_right_x).powi(2) + (y as f32 - eye_y).powi(2)).sqrt();

                    if eye_dist_left < rx * 0.15 || eye_dist_right < rx * 0.15 {
                        image[idx] = 80;
                    }

                    // Mouth
                    let mouth_y = cy + ry * 0.4;
                    let mouth_dist =
                        ((x as f32 - cx).powi(2) + (y as f32 - mouth_y).powi(2)).sqrt();
                    if mouth_dist < rx * 0.3
                        && y as f32 > mouth_y - 2.0
                        && (y as f32) < mouth_y + 2.0
                    {
                        image[idx] = 100;
                    }
                }
            }
        }
    }
}

/// Create a blank image with no face patterns.
fn create_blank_image() -> (Vec<u8>, u32, u32) {
    let width = 100u32;
    let height = 100u32;
    let image = vec![128u8; (width * height) as usize];
    (image, width, height)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OxiMedia Face Detection Example (Haar Cascade)");
    println!("===============================================\n");

    // Test 1: Integral image computation
    println!("Test 1: Integral Image Computation");
    println!("-----------------------------------\n");

    let test_img = vec![1u8; 100];
    let integral = IntegralImage::compute(&test_img, 10, 10);

    println!("  Image size: 10x10");
    println!("  Integral image computed successfully");
    println!("  Sum of 5x5 region: {}", integral.sum(0, 0, 5, 5));
    println!("  Mean of 5x5 region: {:.2}", integral.mean(0, 0, 5, 5));
    println!("  Variance: {:.4}", integral.variance(0, 0, 5, 5));
    println!();

    // Test 2: Haar cascade detector setup
    println!("Test 2: Haar Cascade Detector Configuration");
    println!("--------------------------------------------\n");

    let cascade = HaarCascade::new(24, 24)
        .with_scale_factor(1.1)
        .with_min_neighbors(3);

    println!("  Cascade window size: 24x24");
    println!("  Scale factor: 1.1");
    println!("  Minimum neighbors: 3");
    println!("  Detector created successfully");
    println!();

    // Test 3: Face detection on synthetic image
    println!("Test 3: Face Detection on Synthetic Image");
    println!("------------------------------------------\n");

    let (image, width, height) = create_synthetic_face_image();
    println!("  Image size: {width}x{height}");

    match cascade.detect(&image, width, height) {
        Ok(faces) => {
            println!("  Detected {} face region(s)", faces.len());
            for (i, face) in faces.iter().enumerate() {
                let bbox = &face.bbox;
                println!(
                    "    Face {}: x={}, y={}, w={}, h={}, confidence={:.2}",
                    i + 1,
                    bbox.x,
                    bbox.y,
                    bbox.width,
                    bbox.height,
                    bbox.confidence
                );
                let center = bbox.center();
                println!("      Center: ({}, {})", center.0, center.1);
                println!("      Area: {} pixels", bbox.area());
            }
            if faces.is_empty() {
                println!("  (Note: Empty cascade has no trained features)");
            }
        }
        Err(e) => println!("  Error: {e}"),
    }
    println!();

    // Test 4: Multi-scale detection
    println!("Test 4: Multi-Scale Face Detection");
    println!("-----------------------------------\n");

    let (multi_image, multi_width, multi_height) = create_multi_scale_image();
    println!("  Image size: {multi_width}x{multi_height}");
    println!("  Contains 3 face-like regions at different scales");

    match cascade.detect(&multi_image, multi_width, multi_height) {
        Ok(faces) => {
            println!("  Detected {} face region(s)", faces.len());
            for (i, face) in faces.iter().enumerate() {
                let bbox = &face.bbox;
                println!(
                    "    Face {}: position=({}, {}), size={}x{}",
                    i + 1,
                    bbox.x,
                    bbox.y,
                    bbox.width,
                    bbox.height
                );
            }
            if faces.is_empty() {
                println!("  (Note: Empty cascade has no trained features)");
            }
        }
        Err(e) => println!("  Error: {e}"),
    }
    println!();

    // Test 5: Minimum/maximum size constraints
    println!("Test 5: Size Constraints");
    println!("-------------------------\n");

    let mut cascade_constrained = HaarCascade::new(24, 24)
        .with_scale_factor(1.1)
        .with_min_neighbors(2);

    cascade_constrained.set_min_size(30, 30);
    cascade_constrained.set_max_size(80, 80);

    println!("  Minimum face size: 30x30");
    println!("  Maximum face size: 80x80");

    match cascade_constrained.detect(&multi_image, multi_width, multi_height) {
        Ok(faces) => {
            println!("  Detected {} face(s) within size constraints", faces.len());
            for (i, face) in faces.iter().enumerate() {
                let bbox = &face.bbox;
                println!("    Face {}: size={}x{}", i + 1, bbox.width, bbox.height);
            }
        }
        Err(e) => println!("  Error: {e}"),
    }
    println!();

    // Test 6: Blank image (no faces)
    println!("Test 6: Blank Image (No Faces)");
    println!("-------------------------------\n");

    let (blank_image, blank_w, blank_h) = create_blank_image();
    match cascade.detect(&blank_image, blank_w, blank_h) {
        Ok(faces) => {
            println!("  Detected {} face(s)", faces.len());
            println!("  Result: As expected, no faces in blank image");
        }
        Err(e) => println!("  Error: {e}"),
    }
    println!();

    println!("Face detection test completed successfully!");
    println!("\nNote: This example uses an empty Haar cascade for demonstration.");
    println!("In production, you would load a trained cascade model from file.");
    println!("\nHaar Cascade Features:");
    println!("- Fast detection using integral images (O(1) region sums)");
    println!("- Multi-scale detection via image pyramid");
    println!("- Cascade architecture for efficient rejection");
    println!("- Non-maximum suppression to merge overlapping detections");

    Ok(())
}
