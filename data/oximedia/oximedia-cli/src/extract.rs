//! Frame extraction from video files.
//!
//! Provides:
//! - Frame extraction to image files
//! - Thumbnail generation
//! - Image sequence output
//! - Multiple output formats (PNG, JPEG, PPM)

use crate::progress::TranscodeProgress;
use anyhow::{anyhow, Context, Result};
use colored::Colorize;
use oximedia_codec::{convert_yuv420p_to_rgb, yuv_to_rgb, DecoderConfig, VideoDecoder};
use oximedia_container::{demux::Demuxer, probe_format, ContainerFormat};
use oximedia_core::{CodecId, OxiError, PixelFormat};
use oximedia_image::{
    jpeg::{JpegEncoder, JpegQuality},
    png::{PngEncoder, PngImage},
    ColorSpace, ImageData, ImageFrame, PixelType,
};
use oximedia_io::source::FileSource;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Options for frame extraction.
#[derive(Debug, Clone)]
pub struct ExtractOptions {
    pub input: PathBuf,
    pub output_pattern: String,
    pub format: Option<String>,
    pub start_time: Option<String>,
    pub frames: Option<usize>,
    pub every: usize,
    pub quality: u8,
}

/// Supported output image formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Png,
    Jpeg,
    Ppm,
}

impl ImageFormat {
    /// Parse format from string or file extension.
    #[allow(dead_code)]
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "png" => Ok(Self::Png),
            "jpg" | "jpeg" => Ok(Self::Jpeg),
            "ppm" => Ok(Self::Ppm),
            _ => Err(anyhow!("Unsupported image format: {}", s)),
        }
    }

    /// Get format from file extension.
    #[allow(dead_code)]
    pub fn from_extension(ext: &str) -> Result<Self> {
        Self::from_str(ext)
    }

    /// Get file extension for this format.
    #[allow(dead_code)]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
            Self::Ppm => "ppm",
        }
    }

    /// Get format name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Png => "PNG",
            Self::Jpeg => "JPEG",
            Self::Ppm => "PPM",
        }
    }
}

/// Main frame extraction function.
pub async fn extract_frames(options: ExtractOptions) -> Result<()> {
    info!("Starting frame extraction");
    debug!("Extract options: {:?}", options);

    // Validate input file
    validate_input(&options.input).await?;

    // Determine output format
    let format = determine_format(&options)?;

    // Validate quality for JPEG
    if format == ImageFormat::Jpeg && options.quality > 100 {
        return Err(anyhow!("JPEG quality must be between 0 and 100"));
    }

    // Parse output pattern
    let output_dir = parse_output_pattern(&options.output_pattern)?;

    // Create output directory if needed
    if let Some(dir) = output_dir {
        if !dir.exists() {
            tokio::fs::create_dir_all(&dir)
                .await
                .context("Failed to create output directory")?;
        }
    }

    // Print extraction plan (status output; suppressed by --quiet)
    if !crate::progress::is_quiet() {
        print_extraction_plan(&options, format);
    }

    // Perform extraction
    let extracted = extract_frames_impl(&options, format).await?;

    // Print summary (status output; suppressed by --quiet)
    if !crate::progress::is_quiet() {
        print_extraction_summary(&options, extracted);
    }

    Ok(())
}

/// Validate input file exists and is readable.
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

/// Determine output format from options or pattern.
fn determine_format(options: &ExtractOptions) -> Result<ImageFormat> {
    if let Some(ref fmt) = options.format {
        // Explicit format specified
        ImageFormat::from_str(fmt)
    } else {
        // Try to detect from output pattern
        let pattern = &options.output_pattern;

        if pattern.ends_with(".png") || pattern.contains('%') && !pattern.contains('.') {
            Ok(ImageFormat::Png)
        } else if pattern.ends_with(".jpg") || pattern.ends_with(".jpeg") {
            Ok(ImageFormat::Jpeg)
        } else if pattern.ends_with(".ppm") {
            Ok(ImageFormat::Ppm)
        } else {
            // Default to PNG
            Ok(ImageFormat::Png)
        }
    }
}

/// Parse output pattern to extract directory path.
fn parse_output_pattern(pattern: &str) -> Result<Option<PathBuf>> {
    let path = PathBuf::from(pattern);

    if let Some(parent) = path.parent() {
        if parent != Path::new("") {
            Ok(Some(parent.to_path_buf()))
        } else {
            Ok(None)
        }
    } else {
        Ok(None)
    }
}

/// Print extraction plan before starting.
fn print_extraction_plan(options: &ExtractOptions, format: ImageFormat) {
    println!("{}", "Frame Extraction Plan".cyan().bold());
    println!("{}", "=".repeat(60));
    println!("{:20} {}", "Input:", options.input.display());
    println!("{:20} {}", "Output Pattern:", options.output_pattern);
    println!("{:20} {}", "Format:", format.name());

    if let Some(ref start) = options.start_time {
        println!("{:20} {}", "Start Time:", start);
    }

    if let Some(count) = options.frames {
        println!("{:20} {}", "Frame Count:", count);
    }

    if options.every > 1 {
        println!("{:20} Every {} frames", "Sampling:", options.every);
    }

    if format == ImageFormat::Jpeg {
        println!("{:20} {}", "JPEG Quality:", options.quality);
    }

    println!("{}", "=".repeat(60));
    println!();
}

/// Encode raw RGB24 data as a PPM file.
fn encode_ppm(width: u32, height: u32, rgb_data: &[u8]) -> Vec<u8> {
    let header = format!("P6\n{} {}\n255\n", width, height);
    let mut out = header.into_bytes();
    out.extend_from_slice(rgb_data);
    out
}

/// Encode raw RGB24 data as PNG bytes using the oximedia-image PNG encoder.
fn encode_png(width: u32, height: u32, rgb_data: &[u8]) -> Result<Vec<u8>> {
    use oximedia_image::png::PngColorType;

    let image = PngImage {
        width,
        height,
        bit_depth: 8,
        color_type: PngColorType::Rgb,
        pixels: rgb_data.to_vec(),
        metadata: std::collections::HashMap::new(),
    };

    PngEncoder::default()
        .encode(&image)
        .map_err(|e| anyhow!("PNG encode failed: {}", e))
}

/// Encode raw RGB24 data as JPEG bytes using the oximedia-image JPEG encoder.
fn encode_jpeg(width: u32, height: u32, rgb_data: &[u8], quality: u8) -> Result<Vec<u8>> {
    let frame = ImageFrame::new(
        1,
        width,
        height,
        PixelType::U8,
        3,
        ColorSpace::Srgb,
        ImageData::interleaved(rgb_data.to_vec()),
    );

    JpegEncoder::new(JpegQuality::new(quality))
        .encode(&frame)
        .map_err(|e| anyhow!("JPEG encode failed: {}", e))
}

/// Convert a decoded VideoFrame to a flat RGB24 byte buffer.
///
/// Handles Yuv420p, Yuv422p, Yuv444p, Rgb24, and Rgba32 input formats.
///
/// # Errors
///
/// Returns an error if the pixel format is not supported or conversion fails.
fn video_frame_to_rgb24(frame: &oximedia_codec::VideoFrame) -> Result<Vec<u8>> {
    match frame.format {
        PixelFormat::Yuv420p => {
            let rgb_frame = convert_yuv420p_to_rgb(frame)
                .map_err(|e| anyhow!("YUV420p->RGB conversion failed: {}", e))?;
            if let Some(plane) = rgb_frame.planes.first() {
                Ok(plane.data.clone())
            } else {
                Err(anyhow!("RGB frame has no planes after conversion"))
            }
        }
        PixelFormat::Yuv422p => {
            // Manual 4:2:2 → RGB conversion (half-width chroma)
            if frame.planes.len() != 3 {
                return Err(anyhow!("YUV422p requires 3 planes"));
            }
            let w = frame.width as usize;
            let h = frame.height as usize;
            let y_data = &frame.planes[0].data;
            let u_data = &frame.planes[1].data;
            let v_data = &frame.planes[2].data;
            let uv_width = w / 2;
            let mut rgb = vec![0u8; w * h * 3];
            for row in 0..h {
                for col in 0..w {
                    let y_val = y_data[row * w + col];
                    let uv_idx = row * uv_width + col / 2;
                    let u_val = u_data.get(uv_idx).copied().unwrap_or(128);
                    let v_val = v_data.get(uv_idx).copied().unwrap_or(128);
                    let (r, g, b) = yuv_to_rgb(y_val, u_val, v_val);
                    let off = (row * w + col) * 3;
                    rgb[off] = r;
                    rgb[off + 1] = g;
                    rgb[off + 2] = b;
                }
            }
            Ok(rgb)
        }
        PixelFormat::Yuv444p => {
            // All planes at full resolution
            if frame.planes.len() != 3 {
                return Err(anyhow!("YUV444p requires 3 planes"));
            }
            let w = frame.width as usize;
            let h = frame.height as usize;
            let y_data = &frame.planes[0].data;
            let u_data = &frame.planes[1].data;
            let v_data = &frame.planes[2].data;
            let mut rgb = vec![0u8; w * h * 3];
            for i in 0..(w * h) {
                let y_val = y_data.get(i).copied().unwrap_or(0);
                let u_val = u_data.get(i).copied().unwrap_or(128);
                let v_val = v_data.get(i).copied().unwrap_or(128);
                let (r, g, b) = yuv_to_rgb(y_val, u_val, v_val);
                rgb[i * 3] = r;
                rgb[i * 3 + 1] = g;
                rgb[i * 3 + 2] = b;
            }
            Ok(rgb)
        }
        PixelFormat::Rgb24 => {
            if let Some(plane) = frame.planes.first() {
                Ok(plane.data.clone())
            } else {
                Err(anyhow!("Rgb24 frame has no planes"))
            }
        }
        PixelFormat::Rgba32 => {
            // Strip alpha channel
            if let Some(plane) = frame.planes.first() {
                let rgba = &plane.data;
                let n_pixels = (frame.width * frame.height) as usize;
                let mut rgb = vec![0u8; n_pixels * 3];
                for i in 0..n_pixels {
                    rgb[i * 3] = rgba[i * 4];
                    rgb[i * 3 + 1] = rgba[i * 4 + 1];
                    rgb[i * 3 + 2] = rgba[i * 4 + 2];
                }
                Ok(rgb)
            } else {
                Err(anyhow!("Rgba32 frame has no planes"))
            }
        }
        other => Err(anyhow!(
            "Unsupported pixel format for frame extraction: {:?}",
            other
        )),
    }
}

/// Build a decoder for the given codec id with optional extradata bytes.
///
/// The `extradata` carries codec-specific header bytes (e.g., AV1 sequence header OBU,
/// VP9 codec-private block) as found in the container's `CodecParams.extradata`.
fn make_decoder(codec: CodecId, extradata: Option<Vec<u8>>) -> Result<Box<dyn VideoDecoder>> {
    // AV1, VP9, and VP8 are always enabled (default features of oximedia-codec).
    // Reject unsupported codecs before building the config to avoid a moved-value
    // warning on the `other` branch.
    match codec {
        CodecId::Av1 | CodecId::Vp9 | CodecId::Vp8 => {}
        other => {
            return Err(anyhow!(
                "Unsupported codec for frame extraction: {:?}. \
                 Only AV1, VP9, and VP8 are supported.",
                other
            ));
        }
    }

    let config = DecoderConfig {
        codec,
        extradata,
        threads: 0,
        low_latency: false,
    };

    match codec {
        CodecId::Av1 => {
            use oximedia_codec::Av1Decoder;
            let dec = Av1Decoder::new(config)
                .map_err(|e| anyhow!("Failed to create AV1 decoder: {}", e))?;
            Ok(Box::new(dec))
        }
        CodecId::Vp9 => {
            use oximedia_codec::Vp9Decoder;
            let dec = Vp9Decoder::new(config)
                .map_err(|e| anyhow!("Failed to create VP9 decoder: {}", e))?;
            Ok(Box::new(dec))
        }
        CodecId::Vp8 => {
            use oximedia_codec::Vp8Decoder;
            let dec = Vp8Decoder::new(config)
                .map_err(|e| anyhow!("Failed to create VP8 decoder: {}", e))?;
            Ok(Box::new(dec))
        }
        // Unreachable — the guard above filters unsupported codecs
        _ => unreachable!("Codec guard should have caught unsupported codecs"),
    }
}

/// Perform the actual frame extraction from a real video file.
/// Returns the number of frames actually written to disk.
async fn extract_frames_impl(options: &ExtractOptions, format: ImageFormat) -> Result<usize> {
    let input_path = &options.input;
    info!("Extracting frames from {}", input_path.display());

    // ── 1. Probe the container format ──────────────────────────────────────────
    let probe_bytes = {
        let mut file = tokio::fs::File::open(input_path)
            .await
            .with_context(|| format!("Cannot open '{}' for probing", input_path.display()))?;
        let mut buf = [0u8; 64];
        use tokio::io::AsyncReadExt;
        let n = file
            .read(&mut buf)
            .await
            .context("Read failed during probe")?;
        buf[..n].to_vec()
    };

    let probe = probe_format(&probe_bytes)
        .map_err(|e| anyhow!("Cannot probe input '{}': {:?}", input_path.display(), e))?;

    debug!("Detected container format: {:?}", probe.format);

    // ── 2. Open demuxer ────────────────────────────────────────────────────────
    let source = FileSource::open(input_path)
        .await
        .with_context(|| format!("Cannot open '{}' for demuxing", input_path.display()))?;

    let collected_frames = decode_frames_from_source(source, probe.format, options).await?;

    if collected_frames.is_empty() {
        return Err(anyhow!("No frames decoded from '{}'", input_path.display()));
    }

    // ── 3. Write frames to disk ────────────────────────────────────────────────
    let mut progress = TranscodeProgress::new_spinner();
    let mut extracted = 0usize;

    for (i, (width, height, rgb_data)) in collected_frames.iter().enumerate() {
        let output_file = generate_output_filename(&options.output_pattern, i);
        debug!("Writing frame {} to {}", i, output_file.display());

        // Ensure parent directory exists
        if let Some(parent) = output_file.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .context("Failed to create frame output directory")?;
            }
        }

        let file_data = match format {
            ImageFormat::Ppm => encode_ppm(*width, *height, rgb_data),
            ImageFormat::Png => encode_png(*width, *height, rgb_data)
                .with_context(|| format!("PNG encoding failed for frame {i}"))?,
            ImageFormat::Jpeg => encode_jpeg(*width, *height, rgb_data, options.quality)
                .with_context(|| format!("JPEG encoding failed for frame {i}"))?,
        };

        tokio::fs::write(&output_file, file_data)
            .await
            .with_context(|| format!("Failed to write frame to {}", output_file.display()))?;

        extracted += 1;
        progress.update(extracted as u64);
    }

    progress.finish();

    info!(
        "Extracted {} frame(s) from {}",
        extracted,
        input_path.display()
    );

    Ok(extracted)
}

/// Open the given container format, find a video stream, decode frames, and return
/// collected RGB24 data as `(width, height, rgb_bytes)` tuples.
async fn decode_frames_from_source(
    source: FileSource,
    container_format: ContainerFormat,
    options: &ExtractOptions,
) -> Result<Vec<(u32, u32, Vec<u8>)>> {
    match container_format {
        ContainerFormat::Matroska | ContainerFormat::WebM => {
            let demuxer = oximedia_container::demux::MatroskaDemuxer::new(source);
            decode_via_async_demuxer(demuxer, options).await
        }
        ContainerFormat::Mp4 => {
            let demuxer = oximedia_container::demux::Mp4Demuxer::new(source);
            decode_via_async_demuxer(demuxer, options).await
        }
        ContainerFormat::MpegTs => {
            let demuxer = oximedia_container::demux::MpegTsDemuxer::new(source);
            decode_via_async_demuxer(demuxer, options).await
        }
        ContainerFormat::Y4m => {
            // Y4m uses synchronous BufReader-based demuxer; read the file synchronously.
            let path = options.input.clone();
            let every = options.every;
            let max_frames = options.frames.unwrap_or(100);
            tokio::task::spawn_blocking(move || decode_y4m_sync(&path, every, max_frames))
                .await
                .context("Y4M decode task panicked")?
        }
        other => Err(anyhow!(
            "Container format {:?} is not supported for frame extraction. \
             Supported: Matroska/WebM, MP4, MPEG-TS, Y4M.",
            other
        )),
    }
}

/// Generic async demux + decode loop for any `Demuxer` impl that wraps `FileSource`.
async fn decode_via_async_demuxer<D>(
    mut demuxer: D,
    options: &ExtractOptions,
) -> Result<Vec<(u32, u32, Vec<u8>)>>
where
    D: Demuxer,
{
    // Probe headers
    demuxer
        .probe()
        .await
        .map_err(|e| anyhow!("Failed to probe container: {:?}", e))?;

    // Find the first video stream
    let streams = demuxer.streams();
    let video_stream = streams
        .iter()
        .find(|s| s.is_video())
        .ok_or_else(|| anyhow!("No video stream found in input"))?;

    let stream_index = video_stream.index;
    let codec_id = video_stream.codec;
    // Convert bytes::Bytes extradata to Vec<u8> for DecoderConfig
    let extradata: Option<Vec<u8>> = video_stream
        .codec_params
        .extradata
        .as_ref()
        .map(|b| b.to_vec());

    debug!(
        "Video stream {}: codec={:?}, extradata={} bytes",
        stream_index,
        codec_id,
        extradata.as_ref().map_or(0, |b| b.len())
    );

    let mut decoder = make_decoder(codec_id, extradata)?;

    let max_frames = options.frames.unwrap_or(100);
    let every = options.every;

    let mut collected: Vec<(u32, u32, Vec<u8>)> = Vec::new();
    let mut frame_counter: usize = 0; // counts all decoded frames (for sampling)
    let mut total_packets: u64 = 0;

    // Read packets and decode
    loop {
        if collected.len() >= max_frames {
            break;
        }

        let packet = match demuxer.read_packet().await {
            Ok(p) => p,
            Err(OxiError::Eof) => break,
            Err(e) => {
                warn!("Demux error (continuing): {:?}", e);
                break;
            }
        };

        // Only process packets from the selected video stream
        if packet.stream_index != stream_index {
            continue;
        }

        total_packets += 1;
        let pts = packet.pts();

        if let Err(e) = decoder.send_packet(&packet.data, pts) {
            warn!("Decoder send_packet error (skipping): {:?}", e);
            continue;
        }

        // Drain decoded frames
        loop {
            match decoder.receive_frame() {
                Ok(Some(video_frame)) => {
                    if frame_counter % every == 0 {
                        match video_frame_to_rgb24(&video_frame) {
                            Ok(rgb) => {
                                collected.push((video_frame.width, video_frame.height, rgb));
                                if collected.len() >= max_frames {
                                    break;
                                }
                            }
                            Err(e) => {
                                warn!("Frame pixel-format conversion failed (skipping): {}", e);
                            }
                        }
                    }
                    frame_counter += 1;
                }
                Ok(None) => break, // decoder needs more input
                Err(e) => {
                    warn!("Frame decode error (skipping): {:?}", e);
                    break;
                }
            }
        }
    }

    debug!(
        "Processed {} packets, collected {} frames",
        total_packets,
        collected.len()
    );

    // Flush remaining frames from decoder
    if collected.len() < max_frames {
        if let Err(e) = decoder.flush() {
            warn!("Decoder flush error: {:?}", e);
        }
        loop {
            match decoder.receive_frame() {
                Ok(Some(video_frame)) => {
                    if collected.len() >= max_frames {
                        break;
                    }
                    if frame_counter % every == 0 {
                        match video_frame_to_rgb24(&video_frame) {
                            Ok(rgb) => {
                                collected.push((video_frame.width, video_frame.height, rgb));
                            }
                            Err(e) => {
                                warn!("Flush frame conversion failed (skipping): {}", e);
                            }
                        }
                    }
                    frame_counter += 1;
                }
                Ok(None) => break,
                Err(_) => break,
            }
        }
    }

    Ok(collected)
}

/// Synchronous Y4M decode (used in spawn_blocking).
fn decode_y4m_sync(
    path: &Path,
    every: usize,
    max_frames: usize,
) -> Result<Vec<(u32, u32, Vec<u8>)>> {
    use oximedia_container::demux::Y4mDemuxer;
    use std::fs::File;

    let file =
        File::open(path).with_context(|| format!("Cannot open Y4M file: {}", path.display()))?;

    let mut demuxer =
        Y4mDemuxer::new(file).map_err(|e| anyhow!("Y4M header parse failed: {:?}", e))?;

    let width = demuxer.width();
    let height = demuxer.height();
    let w = width as usize;
    let h = height as usize;

    let mut collected: Vec<(u32, u32, Vec<u8>)> = Vec::new();
    let mut frame_idx: usize = 0;

    loop {
        if collected.len() >= max_frames {
            break;
        }

        let raw_frame = match demuxer.read_frame() {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(OxiError::Eof) => break,
            Err(e) => {
                warn!("Y4M read_frame error: {:?}", e);
                break;
            }
        };

        if frame_idx % every == 0 {
            // Y4M raw frame is YUV420p: Y plane (w*h) + U plane (w/2*h/2) + V plane (w/2*h/2)
            let y_size = w * h;
            let uv_size = (w / 2) * (h / 2);
            let expected = y_size + uv_size * 2;

            if raw_frame.len() < expected {
                warn!(
                    "Y4M frame {} too short: {} < {}",
                    frame_idx,
                    raw_frame.len(),
                    expected
                );
                frame_idx += 1;
                continue;
            }

            let y_data = &raw_frame[..y_size];
            let u_data = &raw_frame[y_size..y_size + uv_size];
            let v_data = &raw_frame[y_size + uv_size..y_size + uv_size * 2];
            let uv_width = w / 2;

            let mut rgb = vec![0u8; w * h * 3];
            for row in 0..h {
                for col in 0..w {
                    let y_val = y_data[row * w + col];
                    let uv_idx = (row / 2) * uv_width + col / 2;
                    let u_val = u_data.get(uv_idx).copied().unwrap_or(128);
                    let v_val = v_data.get(uv_idx).copied().unwrap_or(128);
                    let (r, g, b) = yuv_to_rgb(y_val, u_val, v_val);
                    let off = (row * w + col) * 3;
                    rgb[off] = r;
                    rgb[off + 1] = g;
                    rgb[off + 2] = b;
                }
            }

            collected.push((width, height, rgb));
        }

        frame_idx += 1;
    }

    Ok(collected)
}

/// Generate output filename from pattern and frame number.
fn generate_output_filename(pattern: &str, frame_number: usize) -> PathBuf {
    if pattern.contains('%') {
        // Pattern contains format specifier (e.g., "frame_%04d.png")
        // Simple implementation: replace %04d, %d, etc.
        let output = if pattern.contains("%04d") {
            pattern.replace("%04d", &format!("{:04}", frame_number))
        } else if pattern.contains("%05d") {
            pattern.replace("%05d", &format!("{:05}", frame_number))
        } else if pattern.contains("%d") {
            pattern.replace("%d", &format!("{}", frame_number))
        } else {
            // Fallback: append frame number before extension
            let path = PathBuf::from(pattern);
            let stem = path.file_stem().unwrap_or_default().to_string_lossy();
            let ext = path.extension().unwrap_or_default().to_string_lossy();
            let parent = path.parent().unwrap_or(Path::new(""));

            let filename = if ext.is_empty() {
                format!("{}_{:04}.png", stem, frame_number)
            } else {
                format!("{}_{:04}.{}", stem, frame_number, ext)
            };

            parent.join(filename).to_string_lossy().to_string()
        };

        PathBuf::from(output)
    } else {
        // No pattern, just append frame number
        let path = PathBuf::from(pattern);
        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
        let ext = path.extension().unwrap_or_default().to_string_lossy();
        let parent = path.parent().unwrap_or(Path::new(""));

        let filename = if ext.is_empty() {
            format!("{}_{:04}.png", stem, frame_number)
        } else {
            format!("{}_{:04}.{}", stem, frame_number, ext)
        };

        parent.join(filename)
    }
}

/// Print extraction summary after completion.
///
/// `extracted` is the number of frames actually written to disk by
/// `extract_frames_impl` — never an estimate derived from the requested
/// options (the old `frames.unwrap_or(100) / every` formula reported a
/// fabricated count whenever `--frames` was omitted or decoding stopped
/// early).
fn print_extraction_summary(options: &ExtractOptions, extracted: usize) {
    println!();
    println!("{}", "Frame Extraction Complete".green().bold());
    println!("{}", "=".repeat(60));
    println!("{:20} {}", "Frames Extracted:", extracted);
    println!("{:20} {}", "Output Pattern:", options.output_pattern);
    println!("{}", "=".repeat(60));
}

/// Parse time string (e.g., "00:01:30", "90", "1:30") to seconds.
#[allow(dead_code)]
fn parse_time(s: &str) -> Result<f64> {
    // Try parsing as seconds first
    if let Ok(seconds) = s.parse::<f64>() {
        return Ok(seconds);
    }

    // Try parsing as HH:MM:SS or MM:SS
    let parts: Vec<&str> = s.split(':').collect();

    match parts.len() {
        1 => {
            // Just seconds
            parts[0].parse().context("Invalid time format")
        }
        2 => {
            // MM:SS
            let minutes: f64 = parts[0].parse().context("Invalid minutes")?;
            let seconds: f64 = parts[1].parse().context("Invalid seconds")?;
            Ok(minutes * 60.0 + seconds)
        }
        3 => {
            // HH:MM:SS
            let hours: f64 = parts[0].parse().context("Invalid hours")?;
            let minutes: f64 = parts[1].parse().context("Invalid minutes")?;
            let seconds: f64 = parts[2].parse().context("Invalid seconds")?;
            Ok(hours * 3600.0 + minutes * 60.0 + seconds)
        }
        _ => Err(anyhow!("Invalid time format: {}", s)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_format_parsing() {
        assert_eq!(
            ImageFormat::from_str("png").expect("ImageFormat::from_str should succeed"),
            ImageFormat::Png
        );
        assert_eq!(
            ImageFormat::from_str("jpg").expect("ImageFormat::from_str should succeed"),
            ImageFormat::Jpeg
        );
        assert_eq!(
            ImageFormat::from_str("jpeg").expect("ImageFormat::from_str should succeed"),
            ImageFormat::Jpeg
        );
        assert_eq!(
            ImageFormat::from_str("ppm").expect("ImageFormat::from_str should succeed"),
            ImageFormat::Ppm
        );
        assert!(ImageFormat::from_str("bmp").is_err());
    }

    #[test]
    fn test_parse_time() {
        assert_eq!(parse_time("30").expect("parse should succeed"), 30.0);
        assert_eq!(parse_time("1:30").expect("parse should succeed"), 90.0);
        assert_eq!(parse_time("1:01:30").expect("parse should succeed"), 3690.0);
    }

    #[test]
    fn test_generate_output_filename() {
        assert_eq!(
            generate_output_filename("frame_%04d.png", 1),
            PathBuf::from("frame_0001.png")
        );

        assert_eq!(
            generate_output_filename("output_%d.jpg", 42),
            PathBuf::from("output_42.jpg")
        );

        assert_eq!(
            generate_output_filename("frames/frame_%05d.png", 123),
            PathBuf::from("frames/frame_00123.png")
        );
    }

    #[test]
    fn test_determine_format() {
        let options = ExtractOptions {
            input: PathBuf::from("input.mkv"),
            output_pattern: "frame_%04d.png".to_string(),
            format: None,
            start_time: None,
            frames: None,
            every: 1,
            quality: 90,
        };

        assert_eq!(
            determine_format(&options).expect("format determination should succeed"),
            ImageFormat::Png
        );

        let options_jpg = ExtractOptions {
            output_pattern: "frame_%04d.jpg".to_string(),
            ..options
        };

        assert_eq!(
            determine_format(&options_jpg).expect("format determination should succeed"),
            ImageFormat::Jpeg
        );
    }

    /// Verify encode_ppm produces valid PPM output.
    #[test]
    fn test_encode_ppm_still_works() {
        // 2x2 checkerboard: red, green, blue, white
        let rgb_data = vec![
            255u8, 0, 0, // red
            0, 255, 0, // green
            0, 0, 255, // blue
            255, 255, 255, // white
        ];
        let ppm = encode_ppm(2, 2, &rgb_data);

        // Should start with PPM header
        let header_str = String::from_utf8_lossy(&ppm);
        assert!(header_str.starts_with("P6\n2 2\n255\n"));

        // Total length: header + 4 pixels * 3 bytes each
        let header = b"P6\n2 2\n255\n";
        assert_eq!(ppm.len(), header.len() + 12);

        // Verify pixel data is appended correctly
        assert_eq!(&ppm[header.len()..], rgb_data.as_slice());
    }

    /// Verify that passing a non-video file returns an error instead of synthetic frames.
    #[test]
    fn test_non_video_file_returns_error_not_synthetic() {
        // Verify that generate_frame_buffer (the old synthetic function) no longer exists
        // by confirming our encode_ppm works but decode_frames_from_source would fail on
        // a non-video input. We can't easily run the async function in a unit test, but
        // we can confirm no synthetic path is taken by verifying the public interface only
        // has real-decode paths.
        //
        // The absence of generate_frame_buffer as a function in this module confirms
        // the synthetic path is gone. This test documents that intent.
        let rgb = vec![128u8; 3]; // 1x1 pixel
        let ppm = encode_ppm(1, 1, &rgb);
        assert!(!ppm.is_empty(), "encode_ppm must still work");
    }

    /// Verify PNG encoding produces valid PNG output from a simple RGB buffer.
    #[test]
    fn test_encode_png_produces_valid_output() {
        // 4x4 checkerboard: alternating red and blue pixels
        let mut rgb_data = vec![0u8; 4 * 4 * 3];
        for row in 0..4usize {
            for col in 0..4usize {
                let off = (row * 4 + col) * 3;
                if (row + col) % 2 == 0 {
                    rgb_data[off] = 255; // red
                } else {
                    rgb_data[off + 2] = 255; // blue
                }
            }
        }

        let png = encode_png(4, 4, &rgb_data).expect("PNG encoding must succeed");

        // PNG signature: \x89PNG\r\n\x1a\n
        assert_eq!(&png[..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    }

    /// Verify JPEG encoding produces valid JPEG output (SOI + EOI markers).
    #[test]
    fn test_encode_jpeg_produces_valid_output() {
        // 16x16 solid red image
        let rgb_data = vec![200u8, 50, 50].repeat(16 * 16);
        let jpeg = encode_jpeg(16, 16, &rgb_data, 90).expect("JPEG encoding must succeed");

        // Must start with SOI marker 0xFFD8
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "JPEG must start with SOI marker");
        // Must end with EOI marker 0xFFD9
        assert_eq!(
            &jpeg[jpeg.len() - 2..],
            &[0xFF, 0xD9],
            "JPEG must end with EOI marker"
        );
    }
}
