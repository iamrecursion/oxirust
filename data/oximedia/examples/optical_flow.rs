//! Optical flow estimation example.
//!
//! This example demonstrates optical flow computation between consecutive frames
//! using multiple algorithms:
//! - Lucas-Kanade pyramidal optical flow
//! - Farneback dense optical flow
//! - Dense RLOF (Robust Local Optical Flow)
//!
//! The example generates synthetic moving patterns to test flow computation
//! without requiring external video files.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example optical_flow
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::unnecessary_wraps)]

use oximedia_cv::tracking::{FlowMethod, OpticalFlow, Point2D};

/// Create a synthetic image with a circle pattern.
///
/// # Arguments
///
/// * `width` - Image width
/// * `height` - Image height
/// * `center_x` - Circle center X coordinate
/// * `center_y` - Circle center Y coordinate
/// * `radius` - Circle radius
fn create_circle_image(
    width: u32,
    height: u32,
    center_x: f32,
    center_y: f32,
    radius: f32,
) -> Vec<u8> {
    let mut image = vec![50u8; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 - center_x;
            let dy = y as f32 - center_y;
            let dist = (dx * dx + dy * dy).sqrt();

            let idx = (y * width + x) as usize;
            if dist < radius {
                image[idx] = 200;
            } else if dist < radius + 5.0 {
                // Add a border for better gradient
                let alpha = (radius + 5.0 - dist) / 5.0;
                image[idx] = (50.0 + 150.0 * alpha) as u8;
            }
        }
    }

    image
}

/// Create a synthetic image with a square pattern.
///
/// # Arguments
///
/// * `width` - Image width
/// * `height` - Image height
/// * `x` - Square top-left X coordinate
/// * `y` - Square top-left Y coordinate
/// * `size` - Square size
fn create_square_image(width: u32, height: u32, x: u32, y: u32, size: u32) -> Vec<u8> {
    let mut image = vec![50u8; (width * height) as usize];

    for py in 0..height {
        for px in 0..width {
            let idx = (py * width + px) as usize;
            if px >= x && px < x + size && py >= y && py < y + size {
                image[idx] = 200;
            }
        }
    }

    image
}

/// Print flow field statistics.
fn print_flow_statistics(
    _method_name: &str,
    field: &oximedia_cv::tracking::FlowField,
    expected_dx: f32,
    expected_dy: f32,
) {
    let avg_mag = field.average_magnitude();
    let max_mag = field.max_magnitude();

    println!("   Average magnitude: {avg_mag:.2} pixels");
    println!("   Maximum magnitude: {max_mag:.2} pixels");

    // Sample flow at center
    let center_x = field.width / 2;
    let center_y = field.height / 2;
    let center_mag = field.magnitude(center_x, center_y);
    let center_dir = field.direction(center_x, center_y);

    println!("   Flow at center ({center_x}, {center_y}):");
    println!("     Magnitude: {center_mag:.2} pixels");
    println!(
        "     Direction: {:.2} radians ({:.1}°)",
        center_dir,
        center_dir.to_degrees()
    );

    println!("   Expected motion: dx={expected_dx:.1}, dy={expected_dy:.1}");
    let expected_mag = (expected_dx * expected_dx + expected_dy * expected_dy).sqrt();
    println!("   Expected magnitude: {expected_mag:.2} pixels");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OxiMedia Optical Flow Estimation Example");
    println!("=========================================\n");

    let width = 80u32;
    let height = 80u32;

    // Test 1: Horizontal motion (circle moving right)
    println!("Test 1: Horizontal Motion (Circle Moving Right)");
    println!("------------------------------------------------\n");

    let prev_frame = create_circle_image(width, height, 30.0, 40.0, 10.0);
    let curr_frame = create_circle_image(width, height, 35.0, 40.0, 10.0);

    // Lucas-Kanade
    println!("1.1 Lucas-Kanade Dense Flow:");
    let lk_flow = OpticalFlow::new(FlowMethod::LucasKanade).with_window_size(9);

    match lk_flow.compute(&prev_frame, &curr_frame, width, height) {
        Ok(field) => {
            print_flow_statistics("Lucas-Kanade", &field, 5.0, 0.0);
        }
        Err(e) => println!("   Error: {e}"),
    }
    println!();

    // Farneback
    println!("1.2 Farneback Dense Flow:");
    let farneback_flow = OpticalFlow::new(FlowMethod::Farneback)
        .with_window_size(9)
        .with_iterations(10);

    match farneback_flow.compute(&prev_frame, &curr_frame, width, height) {
        Ok(field) => {
            print_flow_statistics("Farneback", &field, 5.0, 0.0);
        }
        Err(e) => println!("   Error: {e}"),
    }
    println!();

    // RLOF
    println!("1.3 Dense RLOF:");
    let rlof_flow = OpticalFlow::new(FlowMethod::DenseRlof)
        .with_window_size(9)
        .with_iterations(5);

    match rlof_flow.compute(&prev_frame, &curr_frame, width, height) {
        Ok(field) => {
            print_flow_statistics("Dense RLOF", &field, 5.0, 0.0);
        }
        Err(e) => println!("   Error: {e}"),
    }
    println!();

    // Test 2: Diagonal motion (square moving diagonally)
    println!("Test 2: Diagonal Motion (Square Moving Down-Right)");
    println!("---------------------------------------------------\n");

    let prev_square = create_square_image(width, height, 20, 20, 15);
    let curr_square = create_square_image(width, height, 25, 25, 15);

    println!("2.1 Lucas-Kanade Dense Flow:");
    match lk_flow.compute(&prev_square, &curr_square, width, height) {
        Ok(field) => {
            print_flow_statistics("Lucas-Kanade", &field, 5.0, 5.0);
        }
        Err(e) => println!("   Error: {e}"),
    }
    println!();

    // Test 3: Sparse optical flow (tracking specific points)
    println!("Test 3: Sparse Optical Flow (Point Tracking)");
    println!("---------------------------------------------\n");

    let points = vec![
        Point2D::new(30.0, 40.0), // Center of circle
        Point2D::new(25.0, 40.0), // Left side
        Point2D::new(35.0, 40.0), // Right side
    ];

    println!("3.1 Tracking {} points:", points.len());
    println!("   Initial points:");
    for (i, p) in points.iter().enumerate() {
        println!("     Point {}: ({:.1}, {:.1})", i + 1, p.x, p.y);
    }

    match lk_flow.compute_sparse(&prev_frame, &curr_frame, width, height, &points) {
        Ok(new_points) => {
            println!("\n   Tracked points:");
            for (i, (old, new)) in points.iter().zip(new_points.iter()).enumerate() {
                let dx = new.x - old.x;
                let dy = new.y - old.y;
                let dist = (dx * dx + dy * dy).sqrt();
                println!(
                    "     Point {}: ({:.1}, {:.1}) -> ({:.1}, {:.1})",
                    i + 1,
                    old.x,
                    old.y,
                    new.x,
                    new.y
                );
                println!("       Motion: dx={dx:.2}, dy={dy:.2}, distance={dist:.2}");
            }
        }
        Err(e) => println!("   Error: {e}"),
    }
    println!();

    // Test 4: No motion (static image)
    println!("Test 4: Static Image (No Motion)");
    println!("---------------------------------\n");

    let static_frame = create_circle_image(width, height, 40.0, 40.0, 10.0);

    println!("4.1 Lucas-Kanade Dense Flow:");
    match lk_flow.compute(&static_frame, &static_frame, width, height) {
        Ok(field) => {
            print_flow_statistics("Lucas-Kanade", &field, 0.0, 0.0);
        }
        Err(e) => println!("   Error: {e}"),
    }
    println!();

    println!("Optical flow test completed successfully!");
    println!("\nAlgorithm Comparison:");
    println!("- Lucas-Kanade: Fast, works well for small motions");
    println!("- Farneback: Dense flow, good for smooth motion fields");
    println!("- RLOF: Robust to outliers, iterative refinement");
    println!("\nAll algorithms can estimate motion from synthetic patterns.");

    Ok(())
}
