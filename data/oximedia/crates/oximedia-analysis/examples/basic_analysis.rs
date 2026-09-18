//! Basic video analysis example.
//!
//! This example demonstrates how to use the oximedia-analysis crate to analyze
//! a simple synthetic video stream and generate reports.

use oximedia_analysis::{AnalysisConfig, Analyzer};
use oximedia_core::types::Rational;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OxiMedia Analysis Example\n");

    // Create analysis configuration
    let config = AnalysisConfig::default()
        .with_scene_detection(true)
        .with_quality_assessment(true)
        .with_black_frame_detection(true)
        .with_content_classification(true)
        .with_thumbnail_generation(10)
        .with_motion_analysis(true)
        .with_color_analysis(true)
        .with_temporal_analysis(true);

    println!("Configuration:");
    println!("  Scene detection: {}", config.scene_detection);
    println!("  Quality assessment: {}", config.quality_assessment);
    println!("  Black frame detection: {}", config.black_detection);
    println!(
        "  Content classification: {}",
        config.content_classification
    );
    println!("  Thumbnail generation: {}", config.thumbnail_generation);
    println!("  Motion analysis: {}", config.motion_analysis);
    println!("  Color analysis: {}", config.color_analysis);
    println!("  Temporal analysis: {}", config.temporal_analysis);
    println!();

    // Create analyzer
    let mut analyzer = Analyzer::new(config);

    // Generate synthetic test content
    println!("Generating synthetic test video...");
    let width = 1920;
    let height = 1080;
    let frame_count = 300; // 10 seconds at 30fps
    let frame_rate = Rational::new(30, 1);

    // Simulate different scenes
    for frame_num in 0..frame_count {
        let (y_plane, u_plane, v_plane) = generate_test_frame(frame_num, width, height);

        analyzer.process_video_frame(&y_plane, &u_plane, &v_plane, width, height, frame_rate)?;

        if frame_num % 30 == 0 {
            print!("\rProcessed frame {}/{}", frame_num, frame_count);
        }
    }
    println!("\rProcessed frame {}/{}  ", frame_count, frame_count);

    // Generate synthetic audio
    println!("\nGenerating synthetic test audio...");
    let sample_rate = 48000;
    let audio_samples = generate_test_audio(10, sample_rate);
    analyzer.process_audio_samples(&audio_samples, sample_rate)?;

    // Finalize and get results
    println!("\nFinalizing analysis...");
    let results = analyzer.finalize();

    // Print summary
    println!("\n=== Analysis Results ===\n");
    println!("Total frames analyzed: {}", results.frame_count);
    println!(
        "Frame rate: {}/{} fps",
        results.frame_rate.num, results.frame_rate.den
    );

    println!("\nScene Detection:");
    println!("  Detected {} scenes", results.scenes.len());
    for (i, scene) in results.scenes.iter().take(5).enumerate() {
        println!(
            "  Scene {}: frames {}-{} ({:?}, confidence: {:.2})",
            i + 1,
            scene.start_frame,
            scene.end_frame,
            scene.change_type,
            scene.confidence
        );
    }
    if results.scenes.len() > 5 {
        println!("  ... and {} more", results.scenes.len() - 5);
    }

    println!("\nBlack Frame Detection:");
    println!("  Detected {} black segments", results.black_frames.len());
    for segment in &results.black_frames {
        println!(
            "  Frames {}-{} (avg luminance: {:.1})",
            segment.start_frame, segment.end_frame, segment.avg_luminance
        );
    }

    println!("\nQuality Assessment:");
    println!(
        "  Average quality score: {:.2}/1.0",
        results.quality_stats.average_score
    );
    println!("  Blockiness: {:.3}", results.quality_stats.avg_blockiness);
    println!("  Blur: {:.3}", results.quality_stats.avg_blur);
    println!("  Noise: {:.3}", results.quality_stats.avg_noise);

    if let Some(ref classification) = results.content_classification {
        println!("\nContent Classification:");
        println!("  Primary type: {:?}", classification.primary_type);
        println!("  Confidence: {:.2}", classification.confidence);
        println!(
            "  Temporal activity: {:.2}",
            classification.stats.avg_temporal_activity
        );
        println!(
            "  Spatial complexity: {:.2}",
            classification.stats.avg_spatial_complexity
        );
    }

    if !results.thumbnails.is_empty() {
        println!("\nThumbnail Selection:");
        println!("  Selected {} thumbnails", results.thumbnails.len());
        for thumb in &results.thumbnails {
            println!("  Frame {}: score {:.2}", thumb.frame, thumb.score);
        }
    }

    if let Some(ref motion) = results.motion_stats {
        println!("\nMotion Analysis:");
        println!("  Camera motion: {:?}", motion.camera_motion);
        println!("  Average motion: {:.2}", motion.avg_motion);
        println!("  Max motion: {:.2}", motion.max_motion);
        println!("  Stability: {:.2}", motion.stability);
    }

    if let Some(ref color) = results.color_analysis {
        println!("\nColor Analysis:");
        println!("  Grading style: {:?}", color.grading_style);
        println!("  Average saturation: {:.2}", color.avg_saturation);
        println!("  Color diversity: {:.2}", color.color_diversity);
        println!("  Dominant colors:");
        for (i, dc) in color.dominant_colors.iter().take(5).enumerate() {
            println!(
                "    {}. RGB({}, {}, {}) - {:.1}%",
                i + 1,
                dc.rgb.0,
                dc.rgb.1,
                dc.rgb.2,
                dc.percentage * 100.0
            );
        }
    }

    if let Some(ref audio) = results.audio_analysis {
        println!("\nAudio Analysis:");
        println!("  Peak level: {:.1} dBFS", audio.peak_dbfs);
        println!("  RMS level: {:.1} dBFS", audio.rms_dbfs);
        println!("  Dynamic range: {:.1} dB", audio.dynamic_range_db);
        if let Some(phase) = audio.phase_correlation {
            println!("  Phase correlation: {:.2}", phase);
        }
        println!("  Silence segments: {}", audio.silence_segments.len());
        println!("  Clipping events: {}", audio.clipping_events.len());
    }

    if let Some(ref temporal) = results.temporal_analysis {
        println!("\nTemporal Analysis:");
        println!("  Consistency: {:.2}", temporal.consistency);
        println!("  Temporal noise: {:.2}", temporal.temporal_noise);
        println!("  Flicker events: {}", temporal.flicker_events.len());
        if let Some(telecine) = temporal.telecine {
            println!("  Telecine pattern: {:?}", telecine);
        }
    }

    // Generate reports
    println!("\n=== Generating Reports ===\n");

    let json_report = results.to_json()?;
    println!("JSON report generated ({} bytes)", json_report.len());

    let html_report = results.to_html()?;
    println!("HTML report generated ({} bytes)", html_report.len());

    // Save reports to files
    std::fs::write("/tmp/oximedia_analysis.json", &json_report)?;
    std::fs::write("/tmp/oximedia_analysis.html", &html_report)?;

    println!("\nReports saved to:");
    println!("  /tmp/oximedia_analysis.json");
    println!("  /tmp/oximedia_analysis.html");

    Ok(())
}

/// Generate a test frame with varying content.
fn generate_test_frame(
    frame_num: usize,
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let uv_width = (width + 1) / 2;
    let uv_height = (height + 1) / 2;

    let mut y_plane = vec![0u8; width * height];
    let mut u_plane = vec![128u8; uv_width * uv_height];
    let mut v_plane = vec![128u8; uv_width * uv_height];

    // Create different scenes every 100 frames
    let scene = frame_num / 100;

    match scene {
        0 => {
            // Scene 1: Bright static content
            for pixel in &mut y_plane {
                *pixel = 200;
            }
        }
        1 => {
            // Scene 2: Moving gradient
            let offset = (frame_num - 100) * 10;
            for y in 0..height {
                for x in 0..width {
                    let val = ((x + offset) % 256) as u8;
                    y_plane[y * width + x] = val;
                }
            }
        }
        2 => {
            // Scene 3: Complex pattern with color
            for y in 0..height {
                for x in 0..width {
                    let checker = ((x / 32) + (y / 32)) % 2;
                    y_plane[y * width + x] = if checker == 0 { 100 } else { 180 };
                }
            }
            // Add color
            for y in 0..uv_height {
                for x in 0..uv_width {
                    u_plane[y * uv_width + x] = (x * 255 / uv_width) as u8;
                    v_plane[y * uv_width + x] = (y * 255 / uv_height) as u8;
                }
            }
        }
        _ => {
            // Default: mid-gray
            for pixel in &mut y_plane {
                *pixel = 128;
            }
        }
    }

    // Add some black frames at the end
    if frame_num >= 290 {
        for pixel in &mut y_plane {
            *pixel = 0;
        }
    }

    (y_plane, u_plane, v_plane)
}

/// Generate test audio with varying characteristics.
fn generate_test_audio(duration_secs: usize, sample_rate: u32) -> Vec<f32> {
    let total_samples = duration_secs * sample_rate as usize;
    let mut samples = Vec::with_capacity(total_samples * 2); // Stereo

    for i in 0..total_samples {
        let t = i as f32 / sample_rate as f32;

        // Generate different audio characteristics
        let freq = 440.0 + (t * 10.0).sin() * 100.0; // Variable frequency
        let amplitude = (t * 0.5).cos() * 0.3 + 0.5; // Variable amplitude

        let sample = (2.0 * std::f32::consts::PI * freq * t).sin() * amplitude;

        // Stereo
        samples.push(sample * 0.7); // Left
        samples.push(sample * 0.9); // Right

        // Add silence in the middle
        if (3..4).contains(&(i / sample_rate as usize)) {
            samples[i * 2] = 0.0;
            samples[i * 2 + 1] = 0.0;
        }
    }

    samples
}
