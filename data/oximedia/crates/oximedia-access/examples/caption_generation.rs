//! Caption generation and synchronization example.

use oximedia_access::caption::style::EdgeStyle;
use oximedia_access::caption::sync::{CaptionSynchronizer, SyncQuality};
use oximedia_access::caption::{
    CaptionConfig, CaptionGenerator, CaptionPosition, CaptionPositioner, CaptionQuality,
    CaptionStyle, CaptionStylePreset, CaptionType,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Caption Generation and Synchronization Example\n");
    println!("===============================================\n");

    // Step 1: Configure caption generation
    println!("1. Configuring caption generator...");

    let config = CaptionConfig::new("en".to_string(), CaptionType::Closed)
        .with_quality(CaptionQuality::Standard)
        .with_max_chars_per_line(42)
        .with_speaker_identification(true);

    config.validate()?;

    println!("   Language: {}", config.language);
    println!("   Type: {:?}", config.caption_type);
    println!("   Quality: {:?}", config.quality);
    println!("   Max chars per line: {}", config.max_chars_per_line);
    println!("   Speaker identification: {}", config.identify_speakers);
    println!("   Sound effects: {}", config.include_sound_effects);
    println!();

    // Step 2: Generate captions from transcript
    println!("2. Generating captions from transcript...");

    let generator = CaptionGenerator::new(config.clone());

    let transcript = "Hello and welcome to this demonstration of caption generation. \
                     We will show you how to create professional captions with proper \
                     timing and formatting. This is an example of multi-line captions \
                     that wrap correctly according to best practices.";

    let timestamps = vec![(0, 3000), (3000, 7000), (7000, 11000), (11000, 15000)];

    let captions = generator.generate_from_transcript(transcript, &timestamps)?;

    println!("   Generated {} caption segments", captions.len());

    for (i, caption) in captions.iter().enumerate() {
        println!("\n   Caption {}:", i + 1);
        println!(
            "   Time: {}ms - {}ms ({}ms duration)",
            caption.start_time(),
            caption.end_time(),
            caption.duration()
        );
        println!("   Text: {}", caption.text());
        println!("   Confidence: {:.2}", caption.confidence);
    }
    println!();

    // Step 3: Apply styling
    println!("3. Applying caption styles...");

    println!("\n   Available style presets:");
    let presets = vec![
        CaptionStylePreset::Standard,
        CaptionStylePreset::HighContrast,
        CaptionStylePreset::YellowOnBlack,
        CaptionStylePreset::BlackOnWhite,
        CaptionStylePreset::Transparent,
    ];

    for preset in presets {
        let style = CaptionStyle::from_preset(preset);
        println!("   - {preset:?}:");
        println!("      Font size: {}", style.font_size);
        println!("      Text color: {:?}", style.text_color);
        println!("      Background: {:?}", style.background_color);
        println!("      Edge style: {:?}", style.edge_style);
    }

    // Create custom style
    let custom_style = CaptionStyle::default()
        .with_font_size(48)
        .with_text_color(255, 255, 255, 255)
        .with_background_color(0, 0, 0, 220)
        .with_edge_style(EdgeStyle::Outline);

    println!("\n   Custom style created:");
    println!("      Font size: {}", custom_style.font_size);
    println!("      Edge style: {:?}", custom_style.edge_style);
    println!();

    // Step 4: Smart positioning
    println!("4. Caption positioning...");

    let positioner = CaptionPositioner::default();

    let positions = vec![
        CaptionPosition::BottomCenter,
        CaptionPosition::TopCenter,
        CaptionPosition::BottomLeft,
        CaptionPosition::Custom(50.0, 75.0),
    ];

    let frame_size = (1920, 1080);

    for position in positions {
        let (x, y) = positioner.get_coordinates(&position, frame_size.0, frame_size.1);
        println!("   Position {position:?}: ({x}, {y})");
    }
    println!();

    // Step 5: Synchronization
    println!("5. Synchronizing captions...");

    let mut captions_for_sync = captions.clone();
    let synchronizer = CaptionSynchronizer::new(SyncQuality::Frame, 24.0);

    println!("   Sync quality: {:?}", SyncQuality::Frame);
    println!("   Frame rate: 24.0 fps");

    // Sync to frame boundaries
    synchronizer.sync_to_frames(&mut captions_for_sync)?;
    println!("   ✓ Synced to frame boundaries");

    // Adjust timing offset
    let offset_ms = 500;
    synchronizer.adjust_offset(&mut captions_for_sync, offset_ms);
    println!("   ✓ Applied {offset_ms}ms offset");

    // Validate timing
    match synchronizer.validate(&captions_for_sync) {
        Ok(()) => println!("   ✓ Timing validation passed"),
        Err(e) => println!("   ✗ Timing validation failed: {e}"),
    }
    println!();

    // Step 6: Quality validation
    println!("6. Validating caption quality...");

    let mut validation_errors = 0;

    for (i, caption) in captions.iter().enumerate() {
        if let Err(e) = generator.validate_caption(caption) {
            println!("   ✗ Caption {}: {}", i + 1, e);
            validation_errors += 1;
        }
    }

    if validation_errors == 0 {
        println!("   ✓ All captions meet quality requirements");
    } else {
        println!("   {validation_errors} validation errors found");
    }
    println!();

    // Step 7: Format text examples
    println!("7. Text formatting examples...");

    let long_text = "This is a very long caption that needs to be split into \
                    multiple lines to ensure proper readability and compliance \
                    with caption standards and best practices for accessibility.";

    println!("   Original text length: {} chars", long_text.len());

    let max_chars_per_line = 35;
    let formatted = format_caption_text(long_text, max_chars_per_line);
    println!("   Formatted text (max {max_chars_per_line} chars/line):");
    for (i, line) in formatted.lines().enumerate() {
        println!("   Line {}: {} ({} chars)", i + 1, line, line.len());
    }
    println!();

    // Step 8: Export examples
    println!("8. Export format examples...");

    println!("\n   WebVTT format:");
    println!("   WEBVTT");
    println!();
    for caption in captions.iter().take(2) {
        let start = format_vtt_time(caption.start_time());
        let end = format_vtt_time(caption.end_time());
        println!("   {start} --> {end}");
        println!("   {}", caption.text());
        println!();
    }

    println!("   SRT format:");
    for (i, caption) in captions.iter().take(2).enumerate() {
        let start = format_srt_time(caption.start_time());
        let end = format_srt_time(caption.end_time());
        println!("   {}", i + 1);
        println!("   {start} --> {end}");
        println!("   {}", caption.text());
        println!();
    }

    println!("Caption generation complete!");

    Ok(())
}

/// Format text according to caption rules (wrapping at max chars per line).
fn format_caption_text(text: &str, max_chars_per_line: usize) -> String {
    let max_lines = 2;
    let mut lines = Vec::new();
    let mut current_line = String::new();

    for word in text.split_whitespace() {
        if current_line.len() + word.len() < max_chars_per_line {
            if !current_line.is_empty() {
                current_line.push(' ');
            }
            current_line.push_str(word);
        } else {
            if !current_line.is_empty() {
                lines.push(current_line.clone());
                current_line.clear();
            }
            current_line.push_str(word);
        }

        if lines.len() >= max_lines {
            break;
        }
    }

    if !current_line.is_empty() && lines.len() < max_lines {
        lines.push(current_line);
    }

    lines.join("\n")
}

fn format_vtt_time(ms: i64) -> String {
    let seconds = ms / 1000;
    let minutes = seconds / 60;
    let hours = minutes / 60;
    let millis = ms % 1000;
    format!(
        "{:02}:{:02}:{:02}.{:03}",
        hours,
        minutes % 60,
        seconds % 60,
        millis
    )
}

fn format_srt_time(ms: i64) -> String {
    let seconds = ms / 1000;
    let minutes = seconds / 60;
    let hours = minutes / 60;
    let millis = ms % 1000;
    format!(
        "{:02}:{:02}:{:02},{:03}",
        hours,
        minutes % 60,
        seconds % 60,
        millis
    )
}
