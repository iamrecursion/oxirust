//! Shot detection and classification example.
//!
//! Demonstrates the `ShotDetector` API using synthetic frame sequences that
//! simulate real-world editing patterns: a stable sequence (single shot), an
//! abrupt hard-cut, and a second stable sequence (second shot).
//!
//! # Usage
//!
//! ```bash
//! cargo run --example shot_detection --features shots -p oximedia
//! ```

use oximedia::prelude::*;
use oximedia_shots::FrameBuffer;

/// Create a sequence of near-identical frames simulating a static shot.
///
/// All channels are set to `base_value`; a tiny per-frame variation keeps
/// the frames distinct without triggering a cut detection.
fn create_static_sequence(count: usize, base_value: u8) -> Vec<FrameBuffer> {
    (0..count)
        .map(|i| {
            // Vary by at most ±1 LSB to stay well below the cut threshold.
            let value = base_value.saturating_add((i % 2) as u8);
            FrameBuffer::from_elem(64, 64, 3, value)
        })
        .collect()
}

/// Create a single "hard cut" transition frame.
///
/// The value is maximally different from the preceding shot so the detector
/// will register an abrupt change.
fn create_cut_frame(after_value: u8) -> FrameBuffer {
    FrameBuffer::from_elem(64, 64, 3, after_value)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OxiMedia Shot Detection Example");
    println!("=================================\n");

    // ── Build synthetic frame sequence ────────────────────────────────────────
    //
    // Layout:
    //   Frames  0–9  : Shot 1 — dark scene (value = 60)
    //   Frame  10    : Hard cut — bright scene starts (value = 200)
    //   Frames 10–19 : Shot 2 — bright scene (value = 200)
    //
    let mut frames: Vec<FrameBuffer> = Vec::new();
    frames.extend(create_static_sequence(10, 60)); // Shot 1
    frames.push(create_cut_frame(200)); // Hard cut frame
    frames.extend(create_static_sequence(9, 200)); // Shot 2 (remaining)

    println!("Synthetic timeline:");
    println!("  Total frames : {}", frames.len());
    println!("  Frames  0–9  : shot 1 (dark,  value=60)");
    println!("  Frame  10    : hard cut -> bright");
    println!("  Frames 10–19 : shot 2 (bright, value=200)");
    println!();

    // ── Configure and run detector ────────────────────────────────────────────
    let config = ShotDetectorConfig {
        enable_cut_detection: true,
        enable_dissolve_detection: false, // Not needed for this demo
        enable_fade_detection: false,
        enable_wipe_detection: false,
        enable_classification: true,
        enable_movement_detection: true,
        enable_composition_analysis: true,
        cut_threshold: 0.3,
        dissolve_threshold: 0.15,
        fade_threshold: 0.2,
    };

    let detector = ShotDetector::new(config);
    let shots = detector.detect_shots(&frames)?;

    // ── Print shot boundaries ─────────────────────────────────────────────────
    println!("--- Detected Shots ---");
    println!("  Total shots detected: {}", shots.len());
    println!();

    for shot in &shots {
        let duration = shot.duration_seconds();
        println!("  Shot #{}", shot.id);
        println!("    Start PTS      : {}", shot.start.pts);
        println!("    End PTS        : {}", shot.end.pts);
        println!(
            "    Duration       : {:.3} s  ({} frames)",
            duration,
            shot.end.pts - shot.start.pts
        );
        println!(
            "    Type           : {} ({})",
            shot.shot_type.name(),
            shot.shot_type.abbreviation()
        );
        println!("    Camera angle   : {}", shot.angle.name());
        println!("    Transition     : {}", shot.transition.name());
        println!("    Confidence     : {:.2}", shot.confidence);

        if shot.movements.is_empty() {
            println!("    Camera movement: none detected");
        } else {
            for mv in &shot.movements {
                println!(
                    "    Camera movement: {} (speed={:.1}, conf={:.2})",
                    mv.movement_type.name(),
                    mv.speed,
                    mv.confidence
                );
            }
        }

        let comp = &shot.composition;
        println!(
            "    Composition    : rule-of-thirds={:.2}  symmetry={:.2}  \
                  balance={:.2}  overall={:.2}",
            comp.rule_of_thirds,
            comp.symmetry,
            comp.balance,
            comp.overall_score()
        );
        println!();
    }

    // ── Scene detection ───────────────────────────────────────────────────────
    let scenes = detector.detect_scenes(&shots);
    println!("--- Scene Detection ---");
    println!("  Scenes found: {}", scenes.len());
    for scene in &scenes {
        println!(
            "  Scene #{} — shots: {}  duration: {:.3} s",
            scene.id,
            scene.shot_count(),
            scene.duration().to_seconds()
        );
    }
    println!();

    // ── Shot statistics ───────────────────────────────────────────────────────
    let stats: ShotStatistics = detector.analyze_shots(&shots);
    println!("--- Shot Statistics ---");
    println!("  Total shots         : {}", stats.total_shots);
    println!(
        "  Average duration    : {:.3} s",
        stats.average_shot_duration
    );
    println!("  Min duration        : {:.3} s", stats.min_shot_duration);
    println!("  Max duration        : {:.3} s", stats.max_shot_duration);

    if !stats.shot_type_distribution.is_empty() {
        println!("  Shot type breakdown:");
        for (shot_type, count) in &stats.shot_type_distribution {
            if *count > 0 {
                println!("    {:>5} — {}", count, shot_type.name());
            }
        }
    }

    println!("\nShot detection completed.");
    Ok(())
}
