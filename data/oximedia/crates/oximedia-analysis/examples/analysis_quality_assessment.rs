//! Quality assessment example.
//!
//! This example demonstrates detailed quality assessment of video frames,
//! including blockiness detection, blur measurement, and noise estimation.

use oximedia_analysis::quality::{QualityAssessor, QualityStats};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OxiMedia Quality Assessment Example\n");

    // Create quality assessor
    let mut assessor = QualityAssessor::new();

    let width = 1920;
    let height = 1080;

    println!("Assessing quality of different frame types...\n");

    // Test 1: Perfect quality frame
    println!("1. Perfect Quality Frame (Uniform)");
    let perfect_frame = vec![128u8; width * height];
    assessor.process_frame(&perfect_frame, width, height, 0)?;

    // Test 2: Blurred frame
    println!("2. Blurred Frame (Low Pass Filtered)");
    let blurred_frame = generate_blurred_frame(width, height);
    assessor.process_frame(&blurred_frame, width, height, 1)?;

    // Test 3: Noisy frame
    println!("3. Noisy Frame (Gaussian Noise)");
    let noisy_frame = generate_noisy_frame(width, height);
    assessor.process_frame(&noisy_frame, width, height, 2)?;

    // Test 4: Blocked frame
    println!("4. Blocked Frame (8x8 DCT Artifacts)");
    let blocked_frame = generate_blocked_frame(width, height);
    assessor.process_frame(&blocked_frame, width, height, 3)?;

    // Test 5: Sharp detailed frame
    println!("5. Sharp Frame (High Frequency Content)");
    let sharp_frame = generate_sharp_frame(width, height);
    assessor.process_frame(&sharp_frame, width, height, 4)?;

    // Test 6: Mixed quality frame
    println!("6. Mixed Quality Frame (Multiple Artifacts)");
    let mixed_frame = generate_mixed_quality_frame(width, height);
    assessor.process_frame(&mixed_frame, width, height, 5)?;

    // Get results
    let stats = assessor.finalize();

    // Print detailed results
    println!("\n=== Quality Assessment Results ===\n");
    print_quality_stats(&stats);

    // Print per-frame analysis
    println!("\n=== Per-Frame Analysis ===\n");
    println!(
        "{:<8} {:<12} {:<12} {:<12} {:<12}",
        "Frame", "Blockiness", "Blur", "Noise", "Overall"
    );
    println!("{:-<60}", "");

    for frame_quality in &stats.frame_scores {
        println!(
            "{:<8} {:<12.3} {:<12.3} {:<12.3} {:<12.3}",
            frame_quality.frame,
            frame_quality.blockiness,
            frame_quality.blur,
            frame_quality.noise,
            frame_quality.overall
        );
    }

    // Identify problem frames
    println!("\n=== Problem Frames ===\n");

    let low_quality_frames: Vec<_> = stats
        .frame_scores
        .iter()
        .filter(|f| f.overall < 0.5)
        .collect();

    if low_quality_frames.is_empty() {
        println!("No significant quality issues detected.");
    } else {
        println!("Frames with quality issues (overall < 0.5):");
        for frame in &low_quality_frames {
            println!("  Frame {}: {:.2}", frame.frame, frame.overall);
            if frame.blockiness > 0.3 {
                println!("    - High blockiness detected ({:.3})", frame.blockiness);
            }
            if frame.blur > 0.7 {
                println!("    - Significant blur detected ({:.3})", frame.blur);
            }
            if frame.noise > 0.5 {
                println!("    - High noise level detected ({:.3})", frame.noise);
            }
        }
    }

    // Quality recommendations
    println!("\n=== Recommendations ===\n");
    print_recommendations(&stats);

    Ok(())
}

fn print_quality_stats(stats: &QualityStats) {
    println!("Overall Statistics:");
    println!("  Average Quality Score: {:.2}/1.0", stats.average_score);
    println!("  Average Blockiness:    {:.3}", stats.avg_blockiness);
    println!("  Average Blur:          {:.3}", stats.avg_blur);
    println!("  Average Noise:         {:.3}", stats.avg_noise);
    println!();

    // Quality rating
    let rating = if stats.average_score > 0.8 {
        "Excellent"
    } else if stats.average_score > 0.6 {
        "Good"
    } else if stats.average_score > 0.4 {
        "Fair"
    } else if stats.average_score > 0.2 {
        "Poor"
    } else {
        "Very Poor"
    };

    println!("  Overall Rating: {}", rating);
}

fn print_recommendations(stats: &QualityStats) {
    let mut recommendations = Vec::new();

    if stats.avg_blockiness > 0.3 {
        recommendations.push("High blockiness detected. Consider:");
        recommendations.push("  - Using higher bitrate for encoding");
        recommendations.push("  - Applying deblocking filters");
        recommendations.push("  - Using better quality encoder settings");
    }

    if stats.avg_blur > 0.6 {
        recommendations.push("Significant blur detected. Consider:");
        recommendations.push("  - Checking camera focus settings");
        recommendations.push("  - Reducing motion blur (higher shutter speed)");
        recommendations.push("  - Applying sharpening filters carefully");
    }

    if stats.avg_noise > 0.4 {
        recommendations.push("High noise levels detected. Consider:");
        recommendations.push("  - Using temporal noise reduction");
        recommendations.push("  - Improving lighting conditions");
        recommendations.push("  - Using lower ISO/gain settings");
    }

    if stats.average_score > 0.8 {
        recommendations.push("Video quality is excellent. No improvements needed.");
    }

    if recommendations.is_empty() {
        recommendations.push("Video quality is acceptable.");
    }

    for recommendation in &recommendations {
        println!("{}", recommendation);
    }
}

/// Generate a blurred frame using simple box filter.
fn generate_blurred_frame(width: usize, height: usize) -> Vec<u8> {
    // Start with sharp checkerboard
    let mut frame = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            frame[y * width + x] = if ((x / 64) + (y / 64)) % 2 == 0 {
                50
            } else {
                200
            };
        }
    }

    // Apply simple blur (3x3 average)
    let mut blurred = frame.clone();
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let sum = frame[(y - 1) * width + (x - 1)] as u32
                + frame[(y - 1) * width + x] as u32
                + frame[(y - 1) * width + (x + 1)] as u32
                + frame[y * width + (x - 1)] as u32
                + frame[y * width + x] as u32
                + frame[y * width + (x + 1)] as u32
                + frame[(y + 1) * width + (x - 1)] as u32
                + frame[(y + 1) * width + x] as u32
                + frame[(y + 1) * width + (x + 1)] as u32;
            blurred[y * width + x] = (sum / 9) as u8;
        }
    }

    blurred
}

/// Generate a noisy frame.
fn generate_noisy_frame(width: usize, height: usize) -> Vec<u8> {
    let mut frame = vec![128u8; width * height];

    // Add random noise
    for pixel in &mut frame {
        let noise = (((*pixel as i32) ^ 0x5555) % 41) - 20; // Pseudo-random noise
        *pixel = (*pixel as i32 + noise).clamp(0, 255) as u8;
    }

    frame
}

/// Generate a frame with blocking artifacts.
fn generate_blocked_frame(width: usize, height: usize) -> Vec<u8> {
    let mut frame = vec![128u8; width * height];

    // Create smooth gradient
    for y in 0..height {
        for x in 0..width {
            frame[y * width + x] = ((x + y) * 255 / (width + height)) as u8;
        }
    }

    // Add strong discontinuities at 8x8 block boundaries
    for y in (0..height).step_by(8) {
        for x in 0..width {
            let val = frame[y * width + x] as i32;
            frame[y * width + x] = (val + 30).min(255) as u8;
        }
    }
    for x in (0..width).step_by(8) {
        for y in 0..height {
            let val = frame[y * width + x] as i32;
            frame[y * width + x] = (val + 30).min(255) as u8;
        }
    }

    frame
}

/// Generate a sharp frame with high frequency content.
fn generate_sharp_frame(width: usize, height: usize) -> Vec<u8> {
    let mut frame = vec![128u8; width * height];

    // Create fine checkerboard pattern (high frequency)
    for y in 0..height {
        for x in 0..width {
            frame[y * width + x] = if ((x / 4) + (y / 4)) % 2 == 0 {
                80
            } else {
                180
            };
        }
    }

    frame
}

/// Generate a frame with multiple quality issues.
fn generate_mixed_quality_frame(width: usize, height: usize) -> Vec<u8> {
    // Start with blocked frame
    let mut frame = generate_blocked_frame(width, height);

    // Add noise
    for y in 0..height / 2 {
        for x in 0..width {
            let noise = (((x + y) * 17) % 21) as i32 - 10;
            let val = frame[y * width + x] as i32;
            frame[y * width + x] = (val + noise).clamp(0, 255) as u8;
        }
    }

    // Blur bottom half
    for y in height / 2..height - 1 {
        for x in 1..width - 1 {
            let sum = frame[(y - 1) * width + x] as u32
                + frame[y * width + (x - 1)] as u32
                + frame[y * width + x] as u32
                + frame[y * width + (x + 1)] as u32;
            frame[y * width + x] = (sum / 4) as u8;
        }
    }

    frame
}
