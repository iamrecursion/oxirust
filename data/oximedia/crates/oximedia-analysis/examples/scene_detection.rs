//! Scene detection example.
//!
//! This example demonstrates scene change detection using histogram difference,
//! edge change ratio, and motion analysis.

use oximedia_analysis::scene::SceneDetector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OxiMedia Scene Detection Example\n");

    // Create scene detector with threshold
    let mut detector = SceneDetector::new(0.3);

    let width = 1920;
    let height = 1080;

    println!("Processing video with multiple scene changes...\n");

    // Simulate a video with distinct scenes
    let scenes: Vec<(&str, usize, usize, fn(usize, usize, usize) -> Vec<u8>)> = vec![
        ("Dark indoor scene", 0, 50, generate_dark_scene),
        ("Bright outdoor scene", 50, 100, generate_bright_scene),
        ("Action scene", 100, 150, generate_action_scene),
        ("Close-up portrait", 150, 200, generate_portrait_scene),
        ("Landscape", 200, 250, generate_landscape_scene),
        ("Night scene", 250, 300, generate_night_scene),
    ];

    println!("Generating synthetic scenes:");
    for (name, start, _end, _) in &scenes {
        println!("  Frames {}-{}: {}", start, _end, name);
    }
    println!();

    // Process all frames
    for frame_num in 0..300 {
        // Find current scene
        let frame = if let Some((_, start, _end, generator)) = scenes
            .iter()
            .find(|(_, s, e, _)| frame_num >= *s && frame_num < *e)
        {
            generator(frame_num - start, width, height)
        } else {
            vec![128u8; width * height]
        };

        detector.process_frame(&frame, width, height, frame_num)?;

        if frame_num % 30 == 0 {
            print!("\rProcessing frame {}/300", frame_num);
        }
    }
    println!("\rProcessing frame 300/300  ");

    // Get detected scenes
    let detected_scenes = detector.finalize();

    // Print results
    println!("\n=== Scene Detection Results ===\n");
    println!("Detected {} scene changes\n", detected_scenes.len());

    println!(
        "{:<8} {:<12} {:<12} {:<12} {:<15}",
        "Scene", "Start", "End", "Confidence", "Type"
    );
    println!("{:-<65}", "");

    for (i, scene) in detected_scenes.iter().enumerate() {
        println!(
            "{:<8} {:<12} {:<12} {:<12.2} {:<15?}",
            i + 1,
            scene.start_frame,
            scene.end_frame,
            scene.confidence,
            scene.change_type
        );
    }

    // Analyze detection accuracy
    println!("\n=== Analysis ===\n");

    // Expected scene changes at frames: 50, 100, 150, 200, 250
    let expected_changes: Vec<usize> = vec![50, 100, 150, 200, 250];
    let mut detected_frames: Vec<usize> = detected_scenes.iter().map(|s| s.start_frame).collect();
    detected_frames.sort_unstable();

    println!("Expected scene changes: {:?}", expected_changes);
    println!("Detected scene changes: {:?}\n", detected_frames);

    // Calculate accuracy
    let mut true_positives = 0;
    let tolerance: usize = 5; // Allow ±5 frames tolerance

    for &expected in &expected_changes {
        if detected_frames
            .iter()
            .any(|&detected| expected.abs_diff(detected) <= tolerance)
        {
            true_positives += 1;
        }
    }

    let accuracy = (true_positives as f64 / expected_changes.len() as f64) * 100.0;
    println!(
        "Accuracy: {:.1}% ({}/{})",
        accuracy,
        true_positives,
        expected_changes.len()
    );

    if true_positives == expected_changes.len() {
        println!("Perfect detection!");
    } else if true_positives >= 3 {
        println!("Good detection with minor misses");
    } else {
        println!("Consider adjusting threshold for better detection");
    }

    Ok(())
}

/// Generate a dark indoor scene.
fn generate_dark_scene(_frame_offset: usize, width: usize, height: usize) -> Vec<u8> {
    let mut frame = vec![40u8; width * height];

    // Add some detail/variation
    for y in 0..height {
        for x in 0..width {
            let variation = ((x ^ y) as u8).wrapping_mul(3);
            frame[y * width + x] = frame[y * width + x].saturating_add(variation);
        }
    }

    frame
}

/// Generate a bright outdoor scene.
fn generate_bright_scene(_frame_offset: usize, width: usize, height: usize) -> Vec<u8> {
    let mut frame = vec![220u8; width * height];

    // Add sky-like gradient
    for y in 0..height {
        let gradient = ((y as u16 * 30) / height as u16) as u8;
        for x in 0..width {
            frame[y * width + x] = frame[y * width + x].saturating_sub(gradient);
        }
    }

    frame
}

/// Generate an action scene with high motion.
fn generate_action_scene(frame_offset: usize, width: usize, height: usize) -> Vec<u8> {
    let mut frame = vec![128u8; width * height];

    // Moving objects
    let obj_x = (frame_offset * 40) % width;
    let obj_y = (frame_offset * 20) % height;

    // Draw moving rectangle
    for y in obj_y.saturating_sub(50)..=(obj_y + 50).min(height - 1) {
        for x in obj_x.saturating_sub(80)..=(obj_x + 80).min(width - 1) {
            frame[y * width + x] = 200;
        }
    }

    // Add motion blur effect
    for y in 0..height {
        for x in 1..width - 1 {
            let avg = (frame[y * width + x - 1] as u16
                + frame[y * width + x] as u16
                + frame[y * width + x + 1] as u16)
                / 3;
            frame[y * width + x] = avg as u8;
        }
    }

    frame
}

/// Generate a portrait scene with face-like region.
fn generate_portrait_scene(_frame_offset: usize, width: usize, height: usize) -> Vec<u8> {
    let mut frame = vec![100u8; width * height];

    // Simple face-like oval in center
    let center_x = width / 2;
    let center_y = height / 2;
    let radius_x = width / 6;
    let radius_y = height / 4;

    for y in 0..height {
        for x in 0..width {
            let dx = (x as i32 - center_x as i32).abs();
            let dy = (y as i32 - center_y as i32).abs();

            // Ellipse equation
            let in_oval = (dx * dx * radius_y as i32 * radius_y as i32
                + dy * dy * radius_x as i32 * radius_x as i32)
                < (radius_x * radius_y) as i32 * (radius_x * radius_y) as i32;

            if in_oval {
                frame[y * width + x] = 180; // Brighter for face
            }
        }
    }

    frame
}

/// Generate a landscape scene.
fn generate_landscape_scene(_frame_offset: usize, width: usize, height: usize) -> Vec<u8> {
    let mut frame = vec![0u8; width * height];

    let horizon = height * 2 / 3;

    for y in 0..height {
        for x in 0..width {
            if y < horizon {
                // Sky - bright
                frame[y * width + x] = 200 - (y * 50 / horizon) as u8;
            } else {
                // Ground - darker with texture
                let texture = ((x / 10 + y / 10) % 20) as u8;
                frame[y * width + x] = 80 + texture;
            }
        }
    }

    frame
}

/// Generate a night scene.
fn generate_night_scene(frame_offset: usize, width: usize, height: usize) -> Vec<u8> {
    let mut frame = vec![20u8; width * height];

    // Add some "stars" (bright points)
    for i in 0..100 {
        let x = ((i * 1234 + frame_offset * 5) % width as usize) as usize;
        let y = ((i * 5678) % height as usize) as usize;
        if x < width && y < height {
            frame[y * width + x] = 250;
        }
    }

    frame
}
