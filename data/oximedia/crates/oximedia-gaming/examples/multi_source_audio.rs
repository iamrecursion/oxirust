//! Multi-source audio mixing example.

use oximedia_gaming::audio::{AudioMixer, AudioSource, MixerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiMedia Gaming - Multi-Source Audio Mixing ===\n");

    // Create audio mixer
    let config = MixerConfig {
        sample_rate: 48000,
        channels: 2,
        master_volume: 1.0,
    };

    let mut mixer = AudioMixer::new(config)?;
    println!("Audio Mixer created (48kHz, Stereo)\n");

    // Add audio sources
    println!("Adding audio sources...");

    mixer.add_source(AudioSource {
        name: "Game".to_string(),
        volume: 0.8,
        muted: false,
    });
    println!("  ✓ Game audio (80%)");

    mixer.add_source(AudioSource {
        name: "Microphone".to_string(),
        volume: 1.0,
        muted: false,
    });
    println!("  ✓ Microphone (100%)");

    mixer.add_source(AudioSource {
        name: "Music".to_string(),
        volume: 0.3,
        muted: false,
    });
    println!("  ✓ Background music (30%)");

    mixer.add_source(AudioSource {
        name: "Discord".to_string(),
        volume: 0.6,
        muted: false,
    });
    println!("  ✓ Discord chat (60%)\n");

    println!("Total sources: {}\n", mixer.source_count());

    // Adjust volumes
    println!("Adjusting volumes...");
    mixer.set_source_volume("Game", 0.6)?;
    println!("  Game: 80% → 60%");

    mixer.set_source_volume("Music", 0.2)?;
    println!("  Music: 30% → 20%\n");

    // Mute Discord temporarily
    println!("Muting Discord...");
    mixer.set_source_mute("Discord", true)?;
    println!("  ✓ Discord muted\n");

    // Unmute Discord
    println!("Unmuting Discord...");
    mixer.set_source_mute("Discord", false)?;
    println!("  ✓ Discord unmuted\n");

    println!("Audio mixing configured successfully!");

    Ok(())
}
