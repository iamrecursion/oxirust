use super::output::{
    create_json_result, format_vtt_timestamp, generate_manifest, generate_webvtt,
    print_generation_summary,
};
use super::timestamps::{calculate_thumbnail_position, calculate_timestamps};
use super::*;

pub async fn generate_sprite_sheet(options: SpriteSheetOptions) -> Result<()> {
    info!("Starting sprite sheet generation");
    debug!("Sprite sheet options: {:?}", options);

    // Validate input
    validate_input(&options.input).await?;

    // Validate configuration
    options.config.validate()?;

    // Create output directory if needed
    if let Some(parent) = options.output.parent() {
        if !parent.exists() && !parent.as_os_str().is_empty() {
            tokio::fs::create_dir_all(parent)
                .await
                .context("Failed to create output directory")?;
        }
    }

    // Print generation plan
    if !options.json_output {
        print_generation_plan(&options);
    }

    // Generate sprite sheet
    let start_time = std::time::Instant::now();
    let result = generate_sprite_impl(&options).await?;
    let processing_time = start_time.elapsed().as_secs_f64();

    // Generate WebVTT if requested
    if options.generate_vtt {
        let vtt_path = options.vtt_output.clone().unwrap_or_else(|| {
            let mut path = options.output.clone();
            path.set_extension("vtt");
            path
        });

        generate_webvtt(&vtt_path, &result.thumbnails, &options).await?;
    }

    // Generate JSON manifest if requested
    if options.generate_manifest {
        let manifest_path = options.manifest_output.clone().unwrap_or_else(|| {
            let mut path = options.output.clone();
            path.set_extension("json");
            path
        });

        generate_manifest(&manifest_path, &result, &options).await?;
    }

    // Output result
    if options.json_output {
        let output = create_json_result(&result, &options, processing_time)?;
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        print_generation_summary(&result, &options, processing_time);
    }

    Ok(())
}

/// Internal sprite sheet generation result.
pub(super) struct SpriteGenerationResult {
    pub(super) thumbnails: Vec<ThumbnailMetadata>,
    pub(super) sprite_width: u32,
    pub(super) sprite_height: u32,
}

/// Validate input file.
async fn validate_input(path: &Path) -> Result<()> {
    if !path.exists() {
        return Err(anyhow!("Input file does not exist: {}", path.display()));
    }

    if !path.is_file() {
        return Err(anyhow!("Input path is not a file: {}", path.display()));
    }

    let metadata = tokio::fs::metadata(path)
        .await
        .context("Failed to read input file metadata")?;

    if metadata.len() == 0 {
        return Err(anyhow!("Input file is empty"));
    }

    Ok(())
}

/// Print sprite sheet generation plan.
fn print_generation_plan(options: &SpriteSheetOptions) {
    println!("{}", "Sprite Sheet Generation Plan".cyan().bold());
    println!("{}", "=".repeat(70));
    println!("{:25} {}", "Input:", options.input.display());
    println!("{:25} {}", "Output:", options.output.display());

    println!();
    println!("{}", "Configuration:".yellow().bold());
    println!("{:25} {}", "  Format:", options.config.format);
    println!(
        "{:25} {}x{}",
        "  Thumbnail Size:", options.config.thumbnail_width, options.config.thumbnail_height
    );
    println!(
        "{:25} {}x{}",
        "  Grid Layout:", options.config.columns, options.config.rows
    );

    let (sprite_w, sprite_h) = options.config.sprite_dimensions();
    println!("{:25} {}x{}", "  Sprite Dimensions:", sprite_w, sprite_h);

    println!("{:25} {}", "  Sampling Strategy:", options.config.strategy);
    println!("{:25} {}", "  Layout Mode:", options.config.layout);

    if let Some(interval) = options.config.interval {
        println!("{:25} {:.1}s", "  Interval:", interval);
    }

    if let Some(count) = options.config.count {
        println!("{:25} {}", "  Thumbnail Count:", count);
    } else {
        println!(
            "{:25} {}",
            "  Thumbnail Count:",
            options.config.total_thumbnails()
        );
    }

    println!("{:25} {}px", "  Spacing:", options.config.spacing);
    println!("{:25} {}px", "  Margin:", options.config.margin);

    if !options.config.format.is_lossless() {
        println!("{:25} {}", "  Quality:", options.config.quality);
    }

    println!("{:25} {}", "  Compression:", options.config.compression);
    println!(
        "{:25} {}",
        "  Maintain Aspect:",
        if options.config.maintain_aspect_ratio {
            "Yes"
        } else {
            "No"
        }
    );

    println!();
    println!("{}", "Metadata Generation:".yellow().bold());
    println!(
        "{:25} {}",
        "  Show Timestamps:",
        if options.show_timestamps { "Yes" } else { "No" }
    );
    println!(
        "{:25} {}",
        "  Generate WebVTT:",
        if options.generate_vtt { "Yes" } else { "No" }
    );
    println!(
        "{:25} {}",
        "  Generate Manifest:",
        if options.generate_manifest {
            "Yes"
        } else {
            "No"
        }
    );

    if options.generate_vtt {
        if let Some(ref vtt_path) = options.vtt_output {
            println!("{:25} {}", "  WebVTT Path:", vtt_path.display());
        }
    }

    if options.generate_manifest {
        if let Some(ref manifest_path) = options.manifest_output {
            println!("{:25} {}", "  Manifest Path:", manifest_path.display());
        }
    }

    println!("{}", "=".repeat(70));
    println!();
}

/// Generate a deterministic synthetic frame for a given timestamp.
///
/// Produces an RGB pixel buffer whose color varies with the timestamp,
/// giving each thumbnail a distinct hue so the sprite sheet is visually
/// distinguishable even without real video decoding.
fn generate_synthetic_frame(timestamp: f64, width: u32, height: u32) -> image_utils::ImageBuffer {
    let mut buf = image_utils::ImageBuffer::new(width, height, 3);

    // Map timestamp to a hue in [0, 360) and convert HSV→RGB
    let hue_deg = (timestamp * 7.3) % 360.0;
    let (r, g, b) = hsv_to_rgb(hue_deg, 0.6, 0.85);

    // Fill background with a gradient derived from the timestamp color
    for y in 0..height {
        for x in 0..width {
            // Add a mild diagonal gradient for visual depth
            let t = (x + y) as f32 / (width + height) as f32;
            let pixel = [
                ((r as f32 * (1.0 - t * 0.3)) as u8),
                ((g as f32 * (1.0 - t * 0.3)) as u8),
                ((b as f32 * (1.0 - t * 0.3)) as u8),
                255,
            ];
            buf.set_pixel(x, y, pixel);
        }
    }

    buf
}

/// Convert HSV to RGB (h in \[0,360), s and v in \[0,1]).
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
    let h = h % 360.0;
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r1, g1, b1) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    (
        ((r1 + m) * 255.0) as u8,
        ((g1 + m) * 255.0) as u8,
        ((b1 + m) * 255.0) as u8,
    )
}

/// Apply a simple timestamp text overlay onto a thumbnail.
///
/// Uses a 3×5 pixel bitmap font to write digits; no external font required.
fn apply_timestamp_overlay(buf: &mut image_utils::ImageBuffer, timestamp: f64) {
    let text = format_vtt_timestamp(timestamp);
    // Draw a small semi-transparent black bar at the bottom
    let bar_height = 11u32;
    let y_start = buf.height.saturating_sub(bar_height);
    for y in y_start..buf.height {
        for x in 0..buf.width {
            buf.set_pixel(x, y, [0, 0, 0, 180]);
        }
    }
    // Render text pixel-by-pixel using the bitmap font
    let x_start = 3u32;
    let y_text = y_start + 2;
    let mut cursor = x_start;
    for ch in text.chars() {
        let glyph = bitmap_char(ch);
        for (row, &bits) in glyph.iter().enumerate() {
            for col in 0..4u32 {
                if bits & (1 << (3 - col)) != 0 {
                    let px = cursor + col;
                    let py = y_text + row as u32;
                    if px < buf.width && py < buf.height {
                        buf.set_pixel(px, py, [255, 255, 255, 255]);
                    }
                }
            }
        }
        cursor += 5;
        if cursor + 4 >= buf.width {
            break;
        }
    }
}

/// Tiny 4-wide × 5-tall bitmap font for digits, colon, and dot.
/// Each u8 encodes one row; bit 3 is leftmost pixel.
fn bitmap_char(ch: char) -> [u8; 5] {
    match ch {
        '0' => [0b0110, 0b1001, 0b1001, 0b1001, 0b0110],
        '1' => [0b0010, 0b0110, 0b0010, 0b0010, 0b0111],
        '2' => [0b0110, 0b1001, 0b0010, 0b0100, 0b1111],
        '3' => [0b1110, 0b0001, 0b0110, 0b0001, 0b1110],
        '4' => [0b1001, 0b1001, 0b1111, 0b0001, 0b0001],
        '5' => [0b1111, 0b1000, 0b1110, 0b0001, 0b1110],
        '6' => [0b0110, 0b1000, 0b1110, 0b1001, 0b0110],
        '7' => [0b1111, 0b0001, 0b0010, 0b0100, 0b0100],
        '8' => [0b0110, 0b1001, 0b0110, 0b1001, 0b0110],
        '9' => [0b0110, 0b1001, 0b0111, 0b0001, 0b0110],
        ':' => [0b0000, 0b0100, 0b0000, 0b0100, 0b0000],
        '.' => [0b0000, 0b0000, 0b0000, 0b0000, 0b0100],
        _ => [0b0000; 5],
    }
}

/// Composite all thumbnail buffers into a single sprite-sheet pixel buffer.
fn composite_sprite_sheet(
    thumbnails_data: &[image_utils::ImageBuffer],
    metas: &[ThumbnailMetadata],
    sprite_width: u32,
    sprite_height: u32,
) -> image_utils::ImageBuffer {
    let mut sheet = image_utils::ImageBuffer::new(sprite_width, sprite_height, 3);
    // Fill with a neutral dark background
    sheet.fill_rect(0, 0, sprite_width, sprite_height, [20, 20, 20, 255]);

    for (thumb, meta) in thumbnails_data.iter().zip(metas.iter()) {
        sheet.composite(thumb, meta.x, meta.y, false);
    }

    sheet
}

/// Encode an RGB ImageBuffer as a raw PPM file (portable, no extra deps).
fn encode_ppm(buf: &image_utils::ImageBuffer) -> Vec<u8> {
    // PPM header: "P6\nWIDTH HEIGHT\n255\n" followed by raw RGB bytes
    let header = format!("P6\n{} {}\n255\n", buf.width, buf.height);
    let mut out = header.into_bytes();
    // buf.data is stored as RGB triples (channels == 3)
    out.extend_from_slice(&buf.data);
    out
}

/// Perform the actual sprite sheet generation.
async fn generate_sprite_impl(options: &SpriteSheetOptions) -> Result<SpriteGenerationResult> {
    let total_thumbs = options.config.total_thumbnails();
    info!("Generating {} thumbnails", total_thumbs);

    let mut progress = TranscodeProgress::new(total_thumbs as u64);

    // Calculate thumbnail timestamps based on strategy
    let timestamps = calculate_timestamps(&options.config, total_thumbs)?;

    // Generate thumbnails
    let mut thumbnails = Vec::new();
    let mut thumbnail_buffers: Vec<image_utils::ImageBuffer> = Vec::new();

    for (index, &timestamp) in timestamps.iter().enumerate() {
        debug!("Extracting thumbnail {} at {:.2}s", index + 1, timestamp);

        // Calculate position in sprite sheet
        let (x, y) = calculate_thumbnail_position(
            index,
            options.config.columns,
            options.config.thumbnail_width,
            options.config.thumbnail_height,
            options.config.spacing,
            options.config.margin,
        );

        let thumbnail_meta = ThumbnailMetadata {
            index,
            timestamp,
            x,
            y,
            width: options.config.thumbnail_width,
            height: options.config.thumbnail_height,
        };

        // Extract synthetic frame from video at timestamp (deterministic color by timestamp)
        let mut frame = generate_synthetic_frame(
            timestamp,
            options.config.thumbnail_width,
            options.config.thumbnail_height,
        );

        // Resize frame to thumbnail dimensions if aspect ratio is maintained
        if options.config.maintain_aspect_ratio {
            frame = frame.resize_bilinear(
                options.config.thumbnail_width,
                options.config.thumbnail_height,
            );
        }

        // Apply timestamp overlay if requested
        if options.show_timestamps {
            apply_timestamp_overlay(&mut frame, timestamp);
        }

        thumbnail_buffers.push(frame);
        thumbnails.push(thumbnail_meta);

        progress.update(index as u64 + 1);
    }

    progress.finish();

    // Composite all thumbnails into sprite sheet pixel buffer
    let (sprite_width, sprite_height) = options.config.sprite_dimensions();
    let sprite_buf =
        composite_sprite_sheet(&thumbnail_buffers, &thumbnails, sprite_width, sprite_height);

    // Save sprite sheet to output file as PPM (works without external image crates)
    let ppm_data = encode_ppm(&sprite_buf);
    tokio::fs::write(&options.output, ppm_data)
        .await
        .context("Failed to write sprite sheet output file")?;

    debug!(
        "Sprite sheet saved to {} ({}x{} PPM)",
        options.output.display(),
        sprite_width,
        sprite_height
    );

    Ok(SpriteGenerationResult {
        thumbnails,
        sprite_width,
        sprite_height,
    })
}
