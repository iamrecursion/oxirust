//! Service discovery example using mDNS.

use oximedia_videoip::discovery::DiscoveryClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Discovering video sources on the network...");
    println!();

    let client = DiscoveryClient::new()?;

    // Discover all sources with a 5-second timeout
    let sources = client.discover_all(5).await?;

    if sources.is_empty() {
        println!("No sources found.");
        println!();
        println!("Make sure a source is broadcasting on the network.");
        return Ok(());
    }

    println!("Found {} source(s):", sources.len());
    println!();

    for (i, source) in sources.iter().enumerate() {
        println!("{}. {}", i + 1, source.name);
        println!("   Address: {}", source.socket_addr());
        println!(
            "   Video: {:?} {}x{} @ {:.2} fps",
            source.video_format.codec,
            source.video_format.resolution.width,
            source.video_format.resolution.height,
            source.video_format.frame_rate.to_float()
        );
        println!(
            "   Audio: {:?} {} Hz, {} channels",
            source.audio_format.codec,
            source.audio_format.sample_rate,
            source.audio_format.channels
        );

        if !source.metadata.is_empty() {
            println!("   Metadata:");
            for (key, value) in &source.metadata {
                println!("     {}: {}", key, value);
            }
        }

        println!();
    }

    // Try to discover a specific source
    println!("Searching for 'Example Camera'...");
    match client.discover_by_name("Example Camera", 3).await {
        Ok(source) => {
            println!("Found: {} at {}", source.name, source.socket_addr());
        }
        Err(_) => {
            println!("'Example Camera' not found");
        }
    }

    Ok(())
}
