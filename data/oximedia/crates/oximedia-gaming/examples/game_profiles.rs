//! Game profile optimization example.

use oximedia_gaming::capture::game::{GameCapture, GameProfile};

fn main() {
    println!("=== OxiMedia Gaming - Game Profile Optimization ===\n");

    // Show recommended settings for different game types
    let profiles = [
        ("Counter-Strike 2", GameProfile::Fps),
        ("League of Legends", GameProfile::Moba),
        ("Street Fighter 6", GameProfile::Fighting),
        ("Forza Horizon", GameProfile::Racing),
        ("Civilization VI", GameProfile::Strategy),
        ("Elden Ring", GameProfile::Rpg),
        ("Celeste", GameProfile::Platformer),
        ("Generic Game", GameProfile::Generic),
    ];

    println!("Recommended settings by game:\n");

    for (game_name, profile) in profiles {
        let capture = GameCapture::new(profile);
        let settings = capture.recommended_settings();

        println!("📱 {}", game_name);
        println!("   Profile: {:?}", profile);
        println!("   Target Latency: {} ms", settings.target_latency_ms);
        println!("   Max FPS: {}", settings.max_framerate);
        println!("   Priority: {:?}", settings.priority);
        println!(
            "   Motion Prediction: {}",
            if settings.motion_prediction {
                "Yes"
            } else {
                "No"
            }
        );
        println!();
    }

    // Show profile comparison
    println!("=== Profile Comparison ===\n");
    println!("| Profile    | Latency | FPS | Priority  | Motion |");
    println!("|------------|---------|-----|-----------|--------|");

    for (_game, profile) in profiles {
        let capture = GameCapture::new(profile);
        let settings = capture.recommended_settings();

        println!(
            "| {:<10} | {:>4} ms | {:>3} | {:<9} | {:<6} |",
            format!("{:?}", profile),
            settings.target_latency_ms,
            settings.max_framerate,
            format!("{:?}", settings.priority),
            if settings.motion_prediction {
                "Yes"
            } else {
                "No"
            }
        );
    }
}
