//! Example demonstrating subtitle rendering.

use oximedia_codec::VideoFrame;
use oximedia_core::{PixelFormat, Rational, Timestamp};
use oximedia_subtitle::{parser::srt, Alignment, Color, OutlineStyle, Position, SubtitleStyle};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example SRT subtitle data
    let srt_data = r"1
00:00:01,000 --> 00:00:04,000
Hello, World!

2
00:00:05,000 --> 00:00:08,000
This is a subtitle example
with multiple lines.

3
00:00:10,000 --> 00:00:13,000
<i>Styled subtitle with HTML tags</i>
";

    // Parse SRT subtitles
    let subtitles = srt::parse(srt_data.as_bytes())?;
    println!("Parsed {} subtitles", subtitles.len());

    for (i, subtitle) in subtitles.iter().enumerate() {
        println!(
            "Subtitle {}: {}ms - {}ms: {}",
            i + 1,
            subtitle.start_time,
            subtitle.end_time,
            subtitle.text
        );
    }

    // Create a sample video frame
    let mut frame = VideoFrame::new(PixelFormat::Rgb24, 1920, 1080);
    frame.allocate();

    // Fill with blue background
    if let Some(plane) = frame.planes.get_mut(0) {
        let data_len = plane.data.len();
        let mut new_data = vec![0u8; data_len];
        for pixel in new_data.chunks_exact_mut(3) {
            pixel[0] = 0; // R
            pixel[1] = 0; // G
            pixel[2] = 100; // B
        }
        plane.data = new_data;
    }

    // Note: In a real application, you would load a font file
    // For this example, we'll show how to create the renderer
    println!("\nTo use the subtitle renderer:");
    println!("1. Load a font: Font::from_file(\"path/to/font.ttf\")?");
    println!("2. Configure style with colors, size, outline, etc.");
    println!("3. Create renderer: SubtitleRenderer::new(font, style)");
    println!("4. Render: renderer.render_subtitle(&subtitle, &mut frame, timestamp)?");

    // Example style configuration
    let style = SubtitleStyle::default()
        .with_font_size(48.0)
        .with_color(255, 255, 255, 255) // White text
        .with_outline(OutlineStyle::new(Color::black(), 2.0))
        .with_position(Position::bottom_center())
        .with_alignment(Alignment::Center)
        .with_margins(40, 40, 40, 80);

    println!(
        "\nStyle configuration: Font size={}, Color=white, Outline=black 2px",
        style.font_size
    );

    // Example timestamps
    let ts1 = Timestamp::new(2000, Rational::new(1, 1000)); // 2 seconds
    let ts2 = Timestamp::new(6000, Rational::new(1, 1000)); // 6 seconds

    // Show which subtitles would be active at different times
    for subtitle in &subtitles {
        #[allow(clippy::cast_possible_truncation)]
        let ts1_ms = (ts1.to_seconds() * 1000.0).round() as i64;
        #[allow(clippy::cast_possible_truncation)]
        let ts2_ms = (ts2.to_seconds() * 1000.0).round() as i64;
        if subtitle.is_active(ts1_ms) {
            println!(
                "\nAt t=2s, active subtitle: '{}'",
                subtitle.text.lines().next().unwrap_or("")
            );
        }
        if subtitle.is_active(ts2_ms) {
            println!(
                "At t=6s, active subtitle: '{}'",
                subtitle.text.lines().next().unwrap_or("")
            );
        }
    }

    // Example: Write subtitles back to SRT format
    let output_srt = srt::write(&subtitles)?;
    println!("\nGenerated SRT output:\n{output_srt}");

    Ok(())
}
