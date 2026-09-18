//! Video frame extraction utilities for the OxiMedia CLI.
//!
//! Provides helpers for extracting raw RGB24 frames from video files.
//! Y4M (YUV4MPEG2) is supported natively via [`oximedia_container`].
//! Other container formats (MP4, MKV, WebM, MPEG-TS) are supported via
//! the async demuxer + codec pipeline.

use anyhow::{bail, Context, Result};
use oximedia_codec::{Av1Decoder, DecoderConfig, VideoDecoder, VideoFrame, Vp8Decoder, Vp9Decoder};
use oximedia_container::demux::y4m::{Y4mChroma, Y4mDemuxer};
use oximedia_container::demux::{MatroskaDemuxer, Mp4Demuxer, MpegTsDemuxer};
use oximedia_container::Demuxer;
use oximedia_core::{CodecId, PixelFormat};
use oximedia_io::FileSource;
use std::io::Cursor;
use std::path::Path;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Extract a single video frame from `input` as packed RGB24 bytes.
///
/// Returns `(rgb_data, width, height)`.
///
/// # Format support
///
/// - **Y4M (YUV4MPEG2)**: supported for all standard chroma modes (fast path).
/// - **MP4 / MOV / M4V**: supported via Mp4Demuxer + AV1/VP9 decoder.
/// - **MKV / WebM**: supported via MatroskaDemuxer + AV1/VP9/VP8 decoder.
/// - **MPEG-TS / M2TS / MTS**: supported via MpegTsDemuxer + AV1/VP9/VP8 decoder.
/// - **All other formats**: returns a descriptive error.
///
/// # Errors
///
/// Returns an error if the file does not exist, is not a recognised format,
/// does not contain enough frames, or is corrupted.
pub async fn extract_video_frame_rgb(input: &Path, frame_num: u64) -> Result<(Vec<u8>, u32, u32)> {
    if !input.exists() {
        bail!("Input file not found: {}", input.display());
    }

    // Peek at the first 9 bytes to detect Y4M magic.
    let magic = read_magic(input)?;
    if magic.starts_with(b"YUV4MPEG2") {
        return extract_y4m_frame_rgb(input, frame_num);
    }

    // Non-Y4M: use the async container/codec pipeline.
    decode_nth_frame_rgb(input, frame_num).await
}

/// Extract multiple frames from `input` in a single pass.
///
/// `frame_indices` must be **sorted ascending**. Frames not present in the
/// file are silently omitted from the result.
///
/// Returns a `Vec` of `(rgb_data, width, height)` tuples, one per requested
/// frame index that was successfully read.
///
/// # Errors
///
/// See [`extract_video_frame_rgb`].
pub async fn extract_video_frames_rgb(
    input: &Path,
    frame_indices: &[u64],
) -> Result<Vec<(Vec<u8>, u32, u32)>> {
    if frame_indices.is_empty() {
        return Ok(Vec::new());
    }

    if !input.exists() {
        bail!("Input file not found: {}", input.display());
    }

    let magic = read_magic(input)?;
    if magic.starts_with(b"YUV4MPEG2") {
        return extract_y4m_frames_rgb(input, frame_indices);
    }

    // Non-Y4M: use the async container/codec pipeline.
    decode_frames_rgb(input, frame_indices).await
}

// ---------------------------------------------------------------------------
// Internal: Y4M single-frame extraction
// ---------------------------------------------------------------------------

fn extract_y4m_frame_rgb(input: &Path, frame_num: u64) -> Result<(Vec<u8>, u32, u32)> {
    let data = std::fs::read(input).context("Failed to read Y4M file")?;
    let mut demuxer = Y4mDemuxer::new(Cursor::new(data)).context("Failed to parse Y4M header")?;

    let width = demuxer.width();
    let height = demuxer.height();
    let chroma = demuxer.chroma();

    // Walk forward to the requested frame.
    let mut current: u64 = 0;
    loop {
        let raw = demuxer
            .read_frame()
            .with_context(|| format!("I/O error reading frame {current}"))?;

        match raw {
            None => {
                bail!(
                    "Y4M file has fewer than {} frames (only {} found)",
                    frame_num + 1,
                    current
                );
            }
            Some(yuv) => {
                if current == frame_num {
                    let rgb = yuv_to_rgb24(&yuv, width, height, chroma);
                    return Ok((rgb, width, height));
                }
                current += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal: Y4M multi-frame extraction (single pass)
// ---------------------------------------------------------------------------

fn extract_y4m_frames_rgb(input: &Path, frame_indices: &[u64]) -> Result<Vec<(Vec<u8>, u32, u32)>> {
    let data = std::fs::read(input).context("Failed to read Y4M file")?;
    let mut demuxer = Y4mDemuxer::new(Cursor::new(data)).context("Failed to parse Y4M header")?;

    let width = demuxer.width();
    let height = demuxer.height();
    let chroma = demuxer.chroma();

    // We'll collect matching frames in order.
    let mut results: Vec<(Vec<u8>, u32, u32)> = Vec::with_capacity(frame_indices.len());
    let mut idx_iter = frame_indices.iter().peekable();
    let mut current: u64 = 0;

    loop {
        // If all requested indices have been satisfied, stop early.
        let Some(&next_wanted) = idx_iter.peek() else {
            break;
        };

        let raw = demuxer
            .read_frame()
            .with_context(|| format!("I/O error reading frame {current}"))?;

        match raw {
            None => break, // EOF — remaining indices are past the end of file.
            Some(yuv) => {
                if current == *next_wanted {
                    let rgb = yuv_to_rgb24(&yuv, width, height, chroma);
                    results.push((rgb, width, height));
                    idx_iter.next(); // consume this index
                }
                current += 1;
            }
        }
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Internal: container dispatch — detect extension and open demuxer
// ---------------------------------------------------------------------------

async fn decode_nth_frame_rgb(input: &Path, frame_num: u64) -> Result<(Vec<u8>, u32, u32)> {
    let ext = input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "mp4" | "m4v" | "mov" => {
            let source = FileSource::open(input)
                .await
                .with_context(|| format!("Cannot open {}", input.display()))?;
            let demuxer = Mp4Demuxer::new(source);
            let frames = decode_via_demuxer(demuxer, &[frame_num]).await?;
            frames.into_iter().next().ok_or_else(|| {
                anyhow::anyhow!("Frame {} not found in {}", frame_num, input.display())
            })
        }
        "mkv" | "webm" => {
            let source = FileSource::open(input)
                .await
                .with_context(|| format!("Cannot open {}", input.display()))?;
            let demuxer = MatroskaDemuxer::new(source);
            let frames = decode_via_demuxer(demuxer, &[frame_num]).await?;
            frames.into_iter().next().ok_or_else(|| {
                anyhow::anyhow!("Frame {} not found in {}", frame_num, input.display())
            })
        }
        "ts" | "mts" | "m2ts" => {
            let source = FileSource::open(input)
                .await
                .with_context(|| format!("Cannot open {}", input.display()))?;
            let demuxer = MpegTsDemuxer::new(source);
            let frames = decode_via_demuxer(demuxer, &[frame_num]).await?;
            frames.into_iter().next().ok_or_else(|| {
                anyhow::anyhow!("Frame {} not found in {}", frame_num, input.display())
            })
        }
        _ => bail!(
            "Unsupported container extension '{}'; convert to Y4M or use MP4/MKV/WebM/TS. \
             Example: oximedia convert --input {} --output {}.y4m",
            ext,
            input.display(),
            input.file_stem().and_then(|s| s.to_str()).unwrap_or("out")
        ),
    }
}

async fn decode_frames_rgb(
    input: &Path,
    frame_indices: &[u64],
) -> Result<Vec<(Vec<u8>, u32, u32)>> {
    if frame_indices.is_empty() {
        return Ok(Vec::new());
    }

    let ext = input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "mp4" | "m4v" | "mov" => {
            let source = FileSource::open(input)
                .await
                .with_context(|| format!("Cannot open {}", input.display()))?;
            let demuxer = Mp4Demuxer::new(source);
            decode_via_demuxer(demuxer, frame_indices).await
        }
        "mkv" | "webm" => {
            let source = FileSource::open(input)
                .await
                .with_context(|| format!("Cannot open {}", input.display()))?;
            let demuxer = MatroskaDemuxer::new(source);
            decode_via_demuxer(demuxer, frame_indices).await
        }
        "ts" | "mts" | "m2ts" => {
            let source = FileSource::open(input)
                .await
                .with_context(|| format!("Cannot open {}", input.display()))?;
            let demuxer = MpegTsDemuxer::new(source);
            decode_via_demuxer(demuxer, frame_indices).await
        }
        _ => bail!(
            "Unsupported container extension '{}'; convert to Y4M or use MP4/MKV/WebM/TS. \
             Example: oximedia convert --input {} --output {}.y4m",
            ext,
            input.display(),
            input.file_stem().and_then(|s| s.to_str()).unwrap_or("out")
        ),
    }
}

// ---------------------------------------------------------------------------
// Internal: generic async demux+decode loop
// ---------------------------------------------------------------------------

/// Drive a concrete demuxer to extract the requested `frame_indices` as RGB24.
///
/// Frame indices must be sorted ascending. Frames past the end of file are
/// silently omitted from the result (same contract as the Y4M multi-frame path).
async fn decode_via_demuxer<D: Demuxer>(
    mut demuxer: D,
    frame_indices: &[u64],
) -> Result<Vec<(Vec<u8>, u32, u32)>> {
    demuxer.probe().await.context("Container probe failed")?;

    // Find first video stream.
    let (video_stream_idx, codec_id) = {
        let streams = demuxer.streams();
        streams
            .iter()
            .find(|s| s.is_video())
            .map(|s| (s.index, s.codec))
            .ok_or_else(|| anyhow::anyhow!("No video stream found in container"))?
    };

    // Create the appropriate decoder for the codec.
    let mut decoder: Box<dyn VideoDecoder + Send> = match codec_id {
        CodecId::Av1 => {
            let cfg = DecoderConfig {
                codec: CodecId::Av1,
                ..DecoderConfig::default()
            };
            Box::new(Av1Decoder::new(cfg).context("Failed to create AV1 decoder")?)
        }
        CodecId::Vp9 => {
            let cfg = DecoderConfig {
                codec: CodecId::Vp9,
                ..DecoderConfig::default()
            };
            Box::new(Vp9Decoder::new(cfg).context("Failed to create VP9 decoder")?)
        }
        CodecId::Vp8 => {
            let cfg = DecoderConfig {
                codec: CodecId::Vp8,
                ..DecoderConfig::default()
            };
            Box::new(Vp8Decoder::new(cfg).context("Failed to create VP8 decoder")?)
        }
        other => bail!(
            "Video codec {:?} is not yet wired for scopes/thumbnail extraction. \
             Convert to AV1/VP9/VP8 or use Y4M. \
             Example: oximedia convert --input <file> --output out.y4m",
            other
        ),
    };

    let mut results: Vec<(Vec<u8>, u32, u32)> = Vec::with_capacity(frame_indices.len());
    let mut idx_iter = frame_indices.iter().peekable();
    let mut frame_counter: u64 = 0;

    // Packet-read loop.
    loop {
        // If all wanted frames are collected, stop early.
        if idx_iter.peek().is_none() {
            break;
        }

        let packet = match demuxer.read_packet().await {
            Ok(p) => p,
            Err(e) if e.is_eof() => break,
            Err(e) => return Err(anyhow::anyhow!("Demuxer read error: {e}")),
        };

        // Skip packets from other streams.
        if packet.stream_index != video_stream_idx {
            continue;
        }

        let pts = packet.timestamp.pts;
        decoder
            .send_packet(&packet.data, pts)
            .context("Decoder send_packet failed")?;

        // Drain all frames the decoder is ready to yield.
        loop {
            match decoder.receive_frame() {
                Ok(Some(frame)) => {
                    if let Some(&&next_wanted) = idx_iter.peek() {
                        if frame_counter == next_wanted {
                            let rgb = video_frame_to_rgb24(&frame)?;
                            results.push((rgb, frame.width, frame.height));
                            idx_iter.next();
                        }
                    }
                    frame_counter += 1;
                }
                Ok(None) => break,
                Err(e) => {
                    return Err(anyhow::anyhow!(
                        "Decode error on frame {frame_counter}: {e}"
                    ))
                }
            }
        }
    }

    // Flush decoder for any trailing frames.
    decoder.flush().context("Decoder flush failed")?;
    loop {
        if idx_iter.peek().is_none() {
            break;
        }
        match decoder.receive_frame() {
            Ok(Some(frame)) => {
                if let Some(&&next_wanted) = idx_iter.peek() {
                    if frame_counter == next_wanted {
                        let rgb = video_frame_to_rgb24(&frame)?;
                        results.push((rgb, frame.width, frame.height));
                        idx_iter.next();
                    }
                }
                frame_counter += 1;
            }
            Ok(None) => break,
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "Decode flush error on frame {frame_counter}: {e}"
                ))
            }
        }
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Internal: VideoFrame → packed RGB24
// ---------------------------------------------------------------------------

/// Convert a decoded `VideoFrame` to packed RGB24.
///
/// Handles stride-aware plane repacking before conversion. Only 8-bit
/// YUV420p, YUV422p, and YUV444p are supported; other formats return
/// a descriptive error directing the user to re-encode to 8-bit first.
fn video_frame_to_rgb24(frame: &VideoFrame) -> Result<Vec<u8>> {
    let w = frame.width;
    let h = frame.height;

    match frame.format {
        PixelFormat::Yuv420p => {
            if frame.planes.len() < 3 {
                bail!("YUV420p frame has fewer than 3 planes");
            }
            let y = repack_plane(&frame.planes[0]);
            let u = repack_plane(&frame.planes[1]);
            let v = repack_plane(&frame.planes[2]);
            let mut packed = Vec::with_capacity(y.len() + u.len() + v.len());
            packed.extend_from_slice(&y);
            packed.extend_from_slice(&u);
            packed.extend_from_slice(&v);
            Ok(yuv420_to_rgb24(&packed, w, h))
        }
        PixelFormat::Yuv422p => {
            if frame.planes.len() < 3 {
                bail!("YUV422p frame has fewer than 3 planes");
            }
            let y = repack_plane(&frame.planes[0]);
            let u = repack_plane(&frame.planes[1]);
            let v = repack_plane(&frame.planes[2]);
            let mut packed = Vec::with_capacity(y.len() + u.len() + v.len());
            packed.extend_from_slice(&y);
            packed.extend_from_slice(&u);
            packed.extend_from_slice(&v);
            Ok(yuv422_to_rgb24(&packed, w, h))
        }
        PixelFormat::Yuv444p => {
            if frame.planes.len() < 3 {
                bail!("YUV444p frame has fewer than 3 planes");
            }
            let y = repack_plane(&frame.planes[0]);
            let u = repack_plane(&frame.planes[1]);
            let v = repack_plane(&frame.planes[2]);
            let mut packed = Vec::with_capacity(y.len() + u.len() + v.len());
            packed.extend_from_slice(&y);
            packed.extend_from_slice(&u);
            packed.extend_from_slice(&v);
            Ok(yuv444_to_rgb24(&packed, w, h))
        }
        other => bail!(
            "Pixel format {:?} is not supported for RGB extraction. \
             Only 8-bit YUV (420p/422p/444p) is supported. \
             Re-encode to 8-bit AV1/VP9 or use Y4M.",
            other
        ),
    }
}

/// Repack a plane that may have row-stride padding into a tightly-packed buffer.
///
/// If `stride == width`, this is a simple slice copy; otherwise only `width`
/// bytes per row are copied, discarding any row-end padding.
fn repack_plane(plane: &oximedia_codec::Plane) -> Vec<u8> {
    let width = plane.width as usize;
    let height = plane.height as usize;
    let stride = plane.stride;

    if stride == width {
        plane.data[..width * height].to_vec()
    } else {
        let mut out = Vec::with_capacity(width * height);
        for row in 0..height {
            let start = row * stride;
            let end = start + width;
            if end <= plane.data.len() {
                out.extend_from_slice(&plane.data[start..end]);
            } else {
                // Partial / corrupt row — copy what we have and pad with zeros.
                let available = plane.data.len().saturating_sub(start);
                out.extend_from_slice(&plane.data[start..start + available]);
                out.extend(std::iter::repeat_n(0, width - available));
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Internal: magic detection
// ---------------------------------------------------------------------------

fn read_magic(input: &Path) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut f = std::fs::File::open(input).context("Failed to open input file")?;
    let mut buf = [0u8; 9];
    let n = f.read(&mut buf).context("Failed to read file header")?;
    Ok(buf[..n].to_vec())
}

// ---------------------------------------------------------------------------
// YUV → RGB conversion
// ---------------------------------------------------------------------------

/// Convert raw planar YUV data to packed RGB24, dispatching on chroma format.
///
/// BT.601 studio-swing coefficients are used (Y: 16-235, UV: 16-240).
/// For mono frames a grey ramp is produced.
pub fn yuv_to_rgb24(yuv: &[u8], width: u32, height: u32, chroma: Y4mChroma) -> Vec<u8> {
    match chroma {
        Y4mChroma::Mono => yuv_mono_to_rgb24(yuv, width, height),
        Y4mChroma::C420jpeg | Y4mChroma::C420mpeg2 | Y4mChroma::C420paldv => {
            yuv420_to_rgb24(yuv, width, height)
        }
        Y4mChroma::C422 => yuv422_to_rgb24(yuv, width, height),
        Y4mChroma::C444 => yuv444_to_rgb24(yuv, width, height),
        Y4mChroma::C444alpha => {
            // Strip the alpha plane (last w*h bytes) and treat remainder as 444.
            let pixel_count = (width as usize) * (height as usize);
            yuv444_to_rgb24(&yuv[..pixel_count * 3], width, height)
        }
    }
}

/// BT.601 full-range YUV → RGB clamped to [0, 255].
///
/// `y`, `u`, `v` are in the 8-bit range [0, 255] with U/V centred at 128.
#[inline(always)]
fn yuv_pixel_to_rgb(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    let yf = y as f32;
    let uf = u as f32 - 128.0;
    let vf = v as f32 - 128.0;

    let r = (yf + 1.402 * vf).clamp(0.0, 255.0) as u8;
    let g = (yf - 0.344_136 * uf - 0.714_136 * vf).clamp(0.0, 255.0) as u8;
    let b = (yf + 1.772 * uf).clamp(0.0, 255.0) as u8;
    (r, g, b)
}

fn yuv420_to_rgb24(yuv: &[u8], width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    // Y4mChroma::bytes_per_frame uses (w+1)/2 for odd dimensions — match here.
    let chroma_w = (w + 1) / 2;
    let chroma_h = (h + 1) / 2;

    let y_size = w * h;
    let u_size = chroma_w * chroma_h;

    let y_plane = &yuv[..y_size];
    let u_plane = &yuv[y_size..y_size + u_size];
    let v_plane = &yuv[y_size + u_size..];

    let mut rgb = vec![0u8; w * h * 3];
    for row in 0..h {
        for col in 0..w {
            let y = y_plane[row * w + col];
            let u = u_plane[(row / 2) * chroma_w + (col / 2)];
            let v = v_plane[(row / 2) * chroma_w + (col / 2)];
            let (r, g, b) = yuv_pixel_to_rgb(y, u, v);
            let idx = (row * w + col) * 3;
            rgb[idx] = r;
            rgb[idx + 1] = g;
            rgb[idx + 2] = b;
        }
    }
    rgb
}

fn yuv422_to_rgb24(yuv: &[u8], width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let chroma_w = (w + 1) / 2;

    let y_size = w * h;
    let u_size = chroma_w * h;

    let y_plane = &yuv[..y_size];
    let u_plane = &yuv[y_size..y_size + u_size];
    let v_plane = &yuv[y_size + u_size..];

    let mut rgb = vec![0u8; w * h * 3];
    for row in 0..h {
        for col in 0..w {
            let y = y_plane[row * w + col];
            let u = u_plane[row * chroma_w + (col / 2)];
            let v = v_plane[row * chroma_w + (col / 2)];
            let (r, g, b) = yuv_pixel_to_rgb(y, u, v);
            let idx = (row * w + col) * 3;
            rgb[idx] = r;
            rgb[idx + 1] = g;
            rgb[idx + 2] = b;
        }
    }
    rgb
}

fn yuv444_to_rgb24(yuv: &[u8], width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let pixel_count = w * h;

    let y_plane = &yuv[..pixel_count];
    let u_plane = &yuv[pixel_count..pixel_count * 2];
    let v_plane = &yuv[pixel_count * 2..];

    let mut rgb = vec![0u8; pixel_count * 3];
    for i in 0..pixel_count {
        let (r, g, b) = yuv_pixel_to_rgb(y_plane[i], u_plane[i], v_plane[i]);
        rgb[i * 3] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
    }
    rgb
}

fn yuv_mono_to_rgb24(yuv: &[u8], width: u32, height: u32) -> Vec<u8> {
    let pixel_count = (width as usize) * (height as usize);
    let mut rgb = vec![0u8; pixel_count * 3];
    for (i, &luma) in yuv[..pixel_count].iter().enumerate() {
        rgb[i * 3] = luma;
        rgb[i * 3 + 1] = luma;
        rgb[i * 3 + 2] = luma;
    }
    rgb
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_y4m(width: u32, height: u32, frame_count: usize) -> Vec<u8> {
        let mut data: Vec<u8> = Vec::new();
        let header = format!("YUV4MPEG2 W{width} H{height} F25:1 Ip C420jpeg\n");
        data.extend_from_slice(header.as_bytes());
        let frame_size = Y4mChroma::C420jpeg.bytes_per_frame(width, height);
        for i in 0..frame_count {
            data.extend_from_slice(b"FRAME\n");
            let fill = (i & 0xFF) as u8;
            data.extend(std::iter::repeat_n(fill, frame_size));
        }
        data
    }

    #[test]
    fn test_yuv_pixel_to_rgb_grey() {
        // Y=128, U=128, V=128 should give near-grey (128, 128, 128)
        let (r, g, b) = yuv_pixel_to_rgb(128, 128, 128);
        assert!((r as i32 - 128).abs() <= 2);
        assert!((g as i32 - 128).abs() <= 2);
        assert!((b as i32 - 128).abs() <= 2);
    }

    #[tokio::test]
    async fn test_extract_nonexistent_file() {
        let p = std::env::temp_dir().join("oximedia_no_such_file_9999.y4m");
        let result = extract_video_frame_rgb(&p, 0).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_extract_frame_from_y4m_file() {
        let tmp = std::env::temp_dir().join("oximedia_fe_test_single.y4m");
        let data = make_y4m(4, 4, 3);
        std::fs::write(&tmp, &data).expect("write test Y4M");

        let result = extract_video_frame_rgb(&tmp, 0).await;
        assert!(result.is_ok(), "{:?}", result);
        let (rgb, w, h) = result.expect("frame extraction");
        assert_eq!(w, 4);
        assert_eq!(h, 4);
        assert_eq!(rgb.len(), 4 * 4 * 3);

        std::fs::remove_file(&tmp).ok();
    }

    #[tokio::test]
    async fn test_extract_frame_out_of_range() {
        let tmp = std::env::temp_dir().join("oximedia_fe_test_oor.y4m");
        let data = make_y4m(4, 4, 2);
        std::fs::write(&tmp, &data).expect("write test Y4M");

        let result = extract_video_frame_rgb(&tmp, 5).await;
        assert!(result.is_err());

        std::fs::remove_file(&tmp).ok();
    }

    #[tokio::test]
    async fn test_extract_multiple_frames_single_pass() {
        let tmp = std::env::temp_dir().join("oximedia_fe_test_multi.y4m");
        let data = make_y4m(4, 4, 10);
        std::fs::write(&tmp, &data).expect("write test Y4M");

        let indices = vec![0u64, 2, 5, 9];
        let result = extract_video_frames_rgb(&tmp, &indices).await;
        assert!(result.is_ok(), "{:?}", result);
        let frames = result.expect("frame extraction");
        assert_eq!(frames.len(), 4);
        for (rgb, w, h) in &frames {
            assert_eq!(*w, 4);
            assert_eq!(*h, 4);
            assert_eq!(rgb.len(), 4 * 4 * 3);
        }

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_yuv420_odd_dimensions() {
        // 7x7 — chroma planes use (7+1)/2=4 per axis, total frame = 49 + 16 + 16 = 81
        let frame_size = Y4mChroma::C420jpeg.bytes_per_frame(7, 7);
        assert_eq!(frame_size, 81);
        let yuv = vec![128u8; frame_size];
        let rgb = yuv420_to_rgb24(&yuv, 7, 7);
        assert_eq!(rgb.len(), 7 * 7 * 3);
    }

    #[tokio::test]
    async fn test_unsupported_extension_returns_clean_error() {
        let tmp = std::env::temp_dir().join("oximedia_fe_test_unsupported.avi");
        // Write a fake AVI magic
        std::fs::write(&tmp, b"RIFF\x00\x00\x00\x00AVI fake").expect("write test file");

        let result = extract_video_frame_rgb(&tmp, 0).await;
        assert!(result.is_err());
        let msg = result
            .expect_err("expected error")
            .to_string()
            .to_lowercase();
        assert!(
            msg.contains("unsupported") || msg.contains("container") || msg.contains("extension"),
            "unexpected error message: {msg}"
        );

        std::fs::remove_file(&tmp).ok();
    }

    #[tokio::test]
    async fn test_unsupported_extension_multi_returns_clean_error() {
        let tmp = std::env::temp_dir().join("oximedia_fe_test_unsup_multi.avi");
        std::fs::write(&tmp, b"RIFF\x00\x00\x00\x00AVI fake").expect("write test file");

        let result = extract_video_frames_rgb(&tmp, &[0, 1]).await;
        assert!(result.is_err());
        let msg = result
            .expect_err("expected error")
            .to_string()
            .to_lowercase();
        assert!(
            msg.contains("unsupported") || msg.contains("container") || msg.contains("extension"),
            "unexpected error message: {msg}"
        );

        std::fs::remove_file(&tmp).ok();
    }

    #[tokio::test]
    async fn y4m_fast_path_still_works() {
        // Verify that a non-existent Y4M file returns Err cleanly (not a panic).
        let result = extract_video_frame_rgb(Path::new("/nonexistent.y4m"), 0).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn y4m_returns_err_not_panic_on_missing_file() {
        let result = extract_video_frames_rgb(Path::new("/nonexistent.y4m"), &[0, 1, 2]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn unsupported_container_returns_clean_err() {
        // .avi is not supported — either file-not-found or unsupported-extension, both are Err.
        let path_buf = std::env::temp_dir().join("test_oximedia_unsup_check.avi");
        let result = extract_video_frame_rgb(&path_buf, 0).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_repack_plane_no_stride_padding() {
        use oximedia_codec::Plane;
        let plane = Plane {
            data: vec![1, 2, 3, 4, 5, 6],
            stride: 3,
            width: 3,
            height: 2,
        };
        let repacked = repack_plane(&plane);
        assert_eq!(repacked, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_repack_plane_with_stride_padding() {
        use oximedia_codec::Plane;
        // 3-wide, 2-tall, stride=4 (1 padding byte per row)
        let plane = Plane {
            data: vec![1, 2, 3, 0, 4, 5, 6, 0],
            stride: 4,
            width: 3,
            height: 2,
        };
        let repacked = repack_plane(&plane);
        assert_eq!(repacked, vec![1, 2, 3, 4, 5, 6]);
    }
}
