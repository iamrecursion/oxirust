//! Audio description generation example.

use bytes::Bytes;
use oximedia_access::audio_desc::script::{
    AudioDescriptionEntry, AudioDescriptionScript, DescriptionCategory,
};
use oximedia_access::audio_desc::timing::{DialogueSegment, TimingAnalyzer, TimingConstraints};
use oximedia_access::audio_desc::{
    AudioDescriptionConfig, AudioDescriptionGenerator, AudioDescriptionMixer,
    AudioDescriptionQuality, AudioDescriptionType, MixConfig, MixStrategy,
};
use oximedia_audio::frame::AudioBuffer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Audio Description Generation Example\n");

    // Step 1: Create audio description script
    println!("1. Creating audio description script...");
    let mut script = AudioDescriptionScript::new();

    // Add scene description
    script.add_entry(
        AudioDescriptionEntry::new(
            1000,
            3000,
            "A lone figure walks through a misty forest at dawn.".to_string(),
        )
        .with_category(DescriptionCategory::Scene)
        .with_priority(8),
    );

    // Add character action
    script.add_entry(
        AudioDescriptionEntry::new(
            5000,
            7000,
            "She stops suddenly, noticing something in the distance.".to_string(),
        )
        .with_category(DescriptionCategory::Character)
        .with_priority(9),
    );

    // Add visual effect
    script.add_entry(
        AudioDescriptionEntry::new(
            10000,
            12000,
            "A bright light emanates from behind the trees.".to_string(),
        )
        .with_category(DescriptionCategory::VisualEffect)
        .with_priority(7),
    );

    println!("   Created script with {} entries", script.len());
    println!("   Total duration: {}ms\n", script.total_duration_ms());

    // Step 2: Validate script
    println!("2. Validating script...");
    match script.validate() {
        Ok(()) => println!("   Script validation: OK\n"),
        Err(e) => {
            println!("   Script validation failed: {e}\n");
            return Ok(());
        }
    }

    // Step 3: Analyze timing for placement
    println!("3. Analyzing dialogue timing for AD placement...");
    let dialogue = vec![
        DialogueSegment::new(0, 800),
        DialogueSegment::new(3500, 4500),
        DialogueSegment::new(7500, 9500),
    ];

    let constraints = TimingConstraints::from_quality(AudioDescriptionQuality::Standard);
    let analyzer = TimingAnalyzer::new(constraints);
    let gaps = analyzer.find_gaps(&dialogue);

    println!(
        "   Found {} suitable gaps for audio description",
        gaps.len()
    );
    for (i, gap) in gaps.iter().enumerate() {
        println!(
            "   Gap {}: {}ms - {}ms (available: {}ms)",
            i + 1,
            gap.start_time_ms,
            gap.end_time_ms,
            gap.available_duration_ms
        );
    }
    println!();

    // Step 4: Generate audio description
    println!("4. Generating audio description...");
    let config = AudioDescriptionConfig::new(
        AudioDescriptionType::Standard,
        AudioDescriptionQuality::Standard,
    )
    .with_voice("en-US-Neural-Female".to_string())
    .with_speech_rate(1.0)
    .with_volume(0.9);

    let generator = AudioDescriptionGenerator::new(config);

    // Validate script against quality constraints
    match generator.validate_script(&script) {
        Ok(()) => println!("   Script meets quality requirements"),
        Err(e) => {
            println!("   Script validation failed: {e}");
            return Ok(());
        }
    }

    let segments = generator.generate(&script)?;
    println!("   Generated {} audio segments\n", segments.len());

    // Step 5: Mix into main audio
    println!("5. Mixing audio description into main audio...");

    // Create placeholder main audio (in production, this would be actual audio)
    // 15 seconds at 48kHz, stereo, f32 = 4 bytes per sample
    let sample_count = 48000usize * 15 * 2;
    let main_audio = AudioBuffer::Interleaved(Bytes::from(vec![0u8; sample_count * 4]));

    // Configure mixing with ducking strategy
    let mix_config = MixConfig::for_strategy(MixStrategy::Duck);
    let mixer = AudioDescriptionMixer::new(mix_config.clone());

    println!("   Mix strategy: {:?}", mix_config.strategy);
    println!("   Description volume: {}", mix_config.description_volume);
    println!("   Main audio volume during AD: {}", mix_config.main_volume);
    println!("   Duck attack: {}ms", mix_config.duck_attack_ms);
    println!("   Duck release: {}ms", mix_config.duck_release_ms);

    let output = mixer.mix(&main_audio, &segments, 48000, 2)?;
    println!("   Mixed audio: {} bytes\n", output.size());

    // Step 6: Export script
    println!("6. Exporting script to JSON...");
    let json = script.to_json()?;
    println!("   Script JSON ({} bytes):", json.len());
    println!("{json}\n");

    println!("Audio description generation complete!");

    Ok(())
}
