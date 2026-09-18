//! Basic game streaming example.

use oximedia_gaming::{CaptureSource, EncoderPreset, GameStreamer, StreamConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiMedia Gaming - Basic Streaming Example ===\n");

    // Configure streaming
    let config = StreamConfig::builder()
        .source(CaptureSource::PrimaryMonitor)
        .resolution(1920, 1080)
        .framerate(60)
        .encoder_preset(EncoderPreset::LowLatency)
        .bitrate(6000)
        .build()?;

    println!("Configuration:");
    println!(
        "  Resolution: {}x{}",
        config.resolution.0, config.resolution.1
    );
    println!("  Framerate: {} fps", config.framerate);
    println!("  Bitrate: {} kbps\n", config.bitrate);

    // Create streamer
    println!("Creating streamer...");
    let mut streamer = GameStreamer::new(config).await?;

    // Start streaming
    println!("Starting stream...");
    streamer.start().await?;
    println!("Stream started!\n");

    // Simulate streaming for 10 seconds
    for i in 1..=10 {
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

        let stats = streamer.get_stats();
        println!(
            "[{}s] FPS: {}, Bitrate: {} kbps, Latency: {:?}",
            i, stats.fps, stats.bitrate, stats.total_latency
        );
    }

    // Stop streaming
    println!("\nStopping stream...");
    streamer.stop().await?;
    println!("Stream stopped!");

    Ok(())
}
