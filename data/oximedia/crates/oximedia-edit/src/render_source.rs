//! Render source resolution for the timeline rendering pipeline.
//!
//! `RenderSource` abstracts over all input media types that can feed into the
//! renderer: decoded image files (PNG, JPEG), decoded WAV audio, deterministic
//! test patterns (SMPTE colour bars, 1 kHz sine), and unsupported/unknown
//! sources.
//!
//! # Caching
//!
//! `from_path` returns an `Arc<RenderSource>` so that multiple clips that
//! reference the same file share a single decoded copy.  The per-`TimelineRenderer`
//! `source_cache: HashMap<PathBuf, Arc<RenderSource>>` prevents repeated
//! I/O and decode work.

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::{EditError, EditResult};

// ─── Decoded image data ───────────────────────────────────────────────────────

/// Decoded image data in RGBA8 packed format (`width * height * 4` bytes).
#[derive(Debug, Clone)]
pub struct DecodedImageData {
    /// RGBA8 pixel bytes, row-major.
    pub pixels: Vec<u8>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl DecodedImageData {
    /// Create a new decoded image (validates dimensions vs. buffer length).
    pub fn new(pixels: Vec<u8>, width: u32, height: u32) -> EditResult<Self> {
        let expected = (width as usize) * (height as usize) * 4;
        if pixels.len() != expected {
            return Err(EditError::InvalidOperation(format!(
                "DecodedImageData: expected {expected} bytes for {width}x{height} RGBA8, got {}",
                pixels.len()
            )));
        }
        Ok(Self {
            pixels,
            width,
            height,
        })
    }
}

// ─── Decoded WAV data ─────────────────────────────────────────────────────────

/// Decoded WAV audio data.
#[derive(Debug, Clone)]
pub struct WavData {
    /// Interleaved f32 samples.  Length = frame_count * channels.
    pub samples: Vec<f32>,
    /// Sample rate reported by the WAV header.
    pub sample_rate: u32,
    /// Number of channels reported by the WAV header.
    pub channels: u16,
}

// ─── RenderSource ─────────────────────────────────────────────────────────────

/// A resolved and decoded media source used by the render pipeline.
///
/// Cheaply cloneable via `Arc`.
#[derive(Debug)]
pub enum RenderSource {
    /// A decoded still image (PNG or JPEG) in RGBA8.
    Image(DecodedImageData),
    /// Decoded WAV audio.
    Wav(WavData),
    /// Deterministic test pattern (SMPTE colour bars / 1 kHz sine).
    TestPattern,
    /// Source file exists but its format is not supported.
    Unsupported {
        /// Path to the unsupported file.
        path: PathBuf,
    },
}

impl RenderSource {
    /// Resolve a file-system path to a `RenderSource`.
    ///
    /// Recognised extensions (case-insensitive): `.png`, `.jpg`/`.jpeg`,
    /// `.wav`.  Everything else becomes [`RenderSource::Unsupported`].
    ///
    /// Returns an `Arc` so callers can share the decoded data cheaply.
    pub fn from_path(path: &Path) -> EditResult<Arc<Self>> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase());

        match ext.as_deref() {
            Some("png") => {
                let data = std::fs::read(path).map_err(|e| {
                    EditError::InvalidOperation(format!("RenderSource: cannot read {path:?}: {e}"))
                })?;
                let source = decode_png(&data)?;
                Ok(Arc::new(RenderSource::Image(source)))
            }
            Some("jpg") | Some("jpeg") => {
                let data = std::fs::read(path).map_err(|e| {
                    EditError::InvalidOperation(format!("RenderSource: cannot read {path:?}: {e}"))
                })?;
                let source = decode_jpeg(&data)?;
                Ok(Arc::new(RenderSource::Image(source)))
            }
            Some("wav") => {
                let data = std::fs::read(path).map_err(|e| {
                    EditError::InvalidOperation(format!("RenderSource: cannot read {path:?}: {e}"))
                })?;
                let source = decode_wav(&data)?;
                Ok(Arc::new(RenderSource::Wav(source)))
            }
            _ => Ok(Arc::new(RenderSource::Unsupported {
                path: path.to_path_buf(),
            })),
        }
    }

    /// Produce an RGBA8 video frame of size `width × height` at `source_pts`.
    ///
    /// * For [`RenderSource::Image`] the image is scaled (nearest-neighbour) to
    ///   fill the requested dimensions.
    /// * For [`RenderSource::TestPattern`] a deterministic SMPTE colour bar
    ///   pattern is generated.
    /// * All other variants return a black frame.
    ///
    /// The returned buffer is `width * height * 4` bytes.
    #[must_use]
    pub fn sample_video(&self, _source_pts: i64, width: u32, height: u32) -> Vec<u8> {
        match self {
            RenderSource::Image(img) => scale_nearest_rgba8(img, width, height),
            RenderSource::TestPattern => generate_smpte_bars(width, height),
            _ => vec![0u8; (width as usize) * (height as usize) * 4],
        }
    }

    /// Produce interleaved f32 audio samples for `num_samples` stereo frames
    /// starting at `source_pts`.
    ///
    /// * For [`RenderSource::Wav`] the decoded samples are sliced/zero-padded at
    ///   the requested offset.  Channel count mismatches are handled by up-mixing
    ///   mono→stereo or truncating to the requested channel count.
    /// * For [`RenderSource::TestPattern`] a 1 kHz sine wave is generated.
    /// * All other variants return silence.
    ///
    /// `num_samples` is per-channel frame count.  The returned buffer has
    /// `num_samples * channels` elements.
    #[must_use]
    pub fn sample_audio(
        &self,
        source_pts: i64,
        num_samples: usize,
        channels: u16,
        sample_rate: u32,
    ) -> Vec<f32> {
        match self {
            RenderSource::Wav(wav) => {
                slice_wav_samples(wav, source_pts, num_samples, channels, sample_rate)
            }
            RenderSource::TestPattern => {
                generate_sine(source_pts, num_samples, channels, sample_rate)
            }
            _ => vec![0.0_f32; num_samples * channels as usize],
        }
    }
}

// ─── PNG decode helper ────────────────────────────────────────────────────────

fn decode_png(data: &[u8]) -> EditResult<DecodedImageData> {
    use oximedia_image::png::PngDecoder;

    let decoder = PngDecoder::new();
    let img = decoder
        .decode(data)
        .map_err(|e| EditError::InvalidOperation(format!("PNG decode error: {e:?}")))?;

    let w = img.width;
    let h = img.height;
    let pixels_rgba8 = convert_png_to_rgba8(&img.pixels, img.color_type, w, h)?;
    DecodedImageData::new(pixels_rgba8, w, h)
}

/// Convert raw PNG pixel bytes to RGBA8.
fn convert_png_to_rgba8(
    raw: &[u8],
    color_type: oximedia_image::png::PngColorType,
    width: u32,
    height: u32,
) -> EditResult<Vec<u8>> {
    use oximedia_image::png::PngColorType;

    let pixel_count = (width as usize) * (height as usize);
    let mut out = Vec::with_capacity(pixel_count * 4);

    match color_type {
        PngColorType::Rgba => {
            // Already RGBA8 — just clone.
            if raw.len() >= pixel_count * 4 {
                out.extend_from_slice(&raw[..pixel_count * 4]);
            } else {
                out.extend_from_slice(raw);
                // zero-pad if short
                out.resize(pixel_count * 4, 0);
            }
        }
        PngColorType::Rgb => {
            // RGB → RGBA8 (alpha = 255).
            let src_len = raw.len().min(pixel_count * 3);
            let src_pixels = src_len / 3;
            for i in 0..src_pixels {
                out.push(raw[i * 3]);
                out.push(raw[i * 3 + 1]);
                out.push(raw[i * 3 + 2]);
                out.push(255);
            }
            // Pad remaining pixels with opaque black.
            for _ in src_pixels..pixel_count {
                out.extend_from_slice(&[0, 0, 0, 255]);
            }
        }
        PngColorType::Grayscale => {
            // L → RGBA8.
            let src_len = raw.len().min(pixel_count);
            for i in 0..src_len {
                let v = raw[i];
                out.extend_from_slice(&[v, v, v, 255]);
            }
            for _ in src_len..pixel_count {
                out.extend_from_slice(&[0, 0, 0, 255]);
            }
        }
        PngColorType::GrayscaleAlpha => {
            let src_len = raw.len().min(pixel_count * 2);
            let src_pixels = src_len / 2;
            for i in 0..src_pixels {
                let v = raw[i * 2];
                let a = raw[i * 2 + 1];
                out.extend_from_slice(&[v, v, v, a]);
            }
            for _ in src_pixels..pixel_count {
                out.extend_from_slice(&[0, 0, 0, 255]);
            }
        }
        PngColorType::Indexed => {
            // No palette expansion here; treat each byte as a luma value.
            let src_len = raw.len().min(pixel_count);
            for i in 0..src_len {
                let v = raw[i];
                out.extend_from_slice(&[v, v, v, 255]);
            }
            for _ in src_len..pixel_count {
                out.extend_from_slice(&[0, 0, 0, 255]);
            }
        }
    }

    Ok(out)
}

// ─── JPEG decode helper ───────────────────────────────────────────────────────

fn decode_jpeg(data: &[u8]) -> EditResult<DecodedImageData> {
    use oximedia_image::jpeg::JpegDecoder;

    let decoder = JpegDecoder::new();
    let frame = decoder
        .decode(data)
        .map_err(|e| EditError::InvalidOperation(format!("JPEG decode error: {e:?}")))?;

    let w = frame.width;
    let h = frame.height;
    let pixel_count = (w as usize) * (h as usize);
    let mut out = Vec::with_capacity(pixel_count * 4);

    match frame.components {
        3 => {
            // RGB → RGBA8.
            let src_len = frame.pixels.len().min(pixel_count * 3);
            let src_pixels = src_len / 3;
            for i in 0..src_pixels {
                out.push(frame.pixels[i * 3]);
                out.push(frame.pixels[i * 3 + 1]);
                out.push(frame.pixels[i * 3 + 2]);
                out.push(255);
            }
            for _ in src_pixels..pixel_count {
                out.extend_from_slice(&[0, 0, 0, 255]);
            }
        }
        1 => {
            // Grayscale → RGBA8.
            let src_len = frame.pixels.len().min(pixel_count);
            for i in 0..src_len {
                let v = frame.pixels[i];
                out.extend_from_slice(&[v, v, v, 255]);
            }
            for _ in src_len..pixel_count {
                out.extend_from_slice(&[0, 0, 0, 255]);
            }
        }
        4 => {
            // CMYK — convert to RGBA8 (rough approximation).
            let src_len = frame.pixels.len().min(pixel_count * 4);
            let src_pixels = src_len / 4;
            for i in 0..src_pixels {
                let c = frame.pixels[i * 4] as f32 / 255.0;
                let m = frame.pixels[i * 4 + 1] as f32 / 255.0;
                let y = frame.pixels[i * 4 + 2] as f32 / 255.0;
                let k = frame.pixels[i * 4 + 3] as f32 / 255.0;
                #[allow(clippy::cast_possible_truncation)]
                #[allow(clippy::cast_sign_loss)]
                let r = ((1.0 - c) * (1.0 - k) * 255.0).round().clamp(0.0, 255.0) as u8;
                #[allow(clippy::cast_possible_truncation)]
                #[allow(clippy::cast_sign_loss)]
                let g = ((1.0 - m) * (1.0 - k) * 255.0).round().clamp(0.0, 255.0) as u8;
                #[allow(clippy::cast_possible_truncation)]
                #[allow(clippy::cast_sign_loss)]
                let b = ((1.0 - y) * (1.0 - k) * 255.0).round().clamp(0.0, 255.0) as u8;
                out.extend_from_slice(&[r, g, b, 255]);
            }
            for _ in src_pixels..pixel_count {
                out.extend_from_slice(&[0, 0, 0, 255]);
            }
        }
        _ => {
            // Unknown component count — black frame.
            out.extend(std::iter::repeat_n(0u8, pixel_count * 4));
        }
    }

    DecodedImageData::new(out, w, h)
}

// ─── WAV decode helper ────────────────────────────────────────────────────────

fn decode_wav(data: &[u8]) -> EditResult<WavData> {
    use oximedia_audio::wav::WavReader;

    let cursor = Cursor::new(data);
    let mut reader = WavReader::new(cursor)
        .map_err(|e| EditError::InvalidOperation(format!("WAV read error: {e:?}")))?;

    let spec = reader.spec();
    let samples = reader
        .read_samples_f32()
        .map_err(|e| EditError::InvalidOperation(format!("WAV sample read error: {e:?}")))?;

    Ok(WavData {
        samples,
        sample_rate: spec.sample_rate,
        channels: spec.channels,
    })
}

// ─── Nearest-neighbour image scaler ──────────────────────────────────────────

/// Scale RGBA8 image to target dimensions using nearest-neighbour.
fn scale_nearest_rgba8(img: &DecodedImageData, target_w: u32, target_h: u32) -> Vec<u8> {
    if img.width == 0 || img.height == 0 {
        return vec![0u8; (target_w as usize) * (target_h as usize) * 4];
    }

    let src_w = img.width as usize;
    let src_h = img.height as usize;
    let dst_w = target_w as usize;
    let dst_h = target_h as usize;
    let mut out = vec![0u8; dst_w * dst_h * 4];

    for dy in 0..dst_h {
        let sy = (dy * src_h / dst_h).min(src_h - 1);
        for dx in 0..dst_w {
            let sx = (dx * src_w / dst_w).min(src_w - 1);
            let src_idx = (sy * src_w + sx) * 4;
            let dst_idx = (dy * dst_w + dx) * 4;
            out[dst_idx..dst_idx + 4].copy_from_slice(&img.pixels[src_idx..src_idx + 4]);
        }
    }

    out
}

// ─── SMPTE colour bar generator ──────────────────────────────────────────────

/// SMPTE EG 1-1990 colour bars (75% saturation, 100% amplitude).
///
/// Returns RGBA8 packed bytes of size `width * height * 4`.
#[must_use]
pub fn generate_smpte_bars(width: u32, height: u32) -> Vec<u8> {
    // Eight standard SMPTE bar colours (RGBA8, linear gamma approximation).
    // Order: grey, yellow, cyan, green, magenta, red, blue, black (bottom row)
    const BARS: [[u8; 4]; 8] = [
        [192, 192, 192, 255], // 75% grey
        [192, 192, 0, 255],   // 75% yellow
        [0, 192, 192, 255],   // 75% cyan
        [0, 192, 0, 255],     // 75% green
        [192, 0, 192, 255],   // 75% magenta
        [192, 0, 0, 255],     // 75% red
        [0, 0, 192, 255],     // 75% blue
        [0, 0, 0, 255],       // black
    ];

    let w = width as usize;
    let h = height as usize;
    let mut out = vec![0u8; w * h * 4];

    // Top 7/8 of frame — seven colour bars.
    let top_rows = (h * 7) / 8;
    for y in 0..top_rows {
        for x in 0..w {
            let bar = (x * 7 / w).min(6);
            let idx = (y * w + x) * 4;
            out[idx..idx + 4].copy_from_slice(&BARS[bar]);
        }
    }
    // Bottom 1/8 — four sub-bars: -I, white, +Q, black.
    let bottom_colours: [[u8; 4]; 4] = [
        [0, 0, 128, 255],     // -I (dark blue)
        [255, 255, 255, 255], // 100% white
        [19, 0, 77, 255],     // +Q (dark purple)
        [0, 0, 0, 255],       // black
    ];
    for y in top_rows..h {
        for x in 0..w {
            let seg = (x * 4 / w).min(3);
            let idx = (y * w + x) * 4;
            out[idx..idx + 4].copy_from_slice(&bottom_colours[seg]);
        }
    }

    out
}

// ─── 1 kHz sine generator ─────────────────────────────────────────────────────

/// Generate `num_samples` interleaved f32 samples of a 1 kHz sine wave.
///
/// `source_pts` is used as the starting sample offset (for determinism across
/// calls for the same clip at different seek positions).
///
/// The returned buffer has `num_samples * channels` elements.
#[must_use]
pub fn generate_sine(
    source_pts: i64,
    num_samples: usize,
    channels: u16,
    sample_rate: u32,
) -> Vec<f32> {
    use std::f64::consts::TAU;

    let sr = sample_rate as f64;
    let freq = 1000.0_f64;
    let amplitude = 0.25_f64; // -12 dBFS to avoid clipping
    let ch = channels as usize;
    let mut out = Vec::with_capacity(num_samples * ch);
    let start_sample = source_pts.max(0) as u64;

    for i in 0..num_samples {
        let t = (start_sample + i as u64) as f64 / sr;
        #[allow(clippy::cast_possible_truncation)]
        let sample = (TAU * freq * t).sin() * amplitude;
        #[allow(clippy::cast_possible_truncation)]
        let sample_f32 = sample as f32;
        for _ in 0..ch {
            out.push(sample_f32);
        }
    }

    out
}

// ─── WAV sample slicer ────────────────────────────────────────────────────────

/// Extract a contiguous slice of audio from a decoded WAV, handling:
/// - Offset past end-of-file (returns silence).
/// - Mono → stereo up-mix.
/// - Sample-rate mismatch (no resampling; just shifts offset scaling).
fn slice_wav_samples(
    wav: &WavData,
    source_pts: i64,
    num_samples: usize,
    target_channels: u16,
    _target_sample_rate: u32,
) -> Vec<f32> {
    let src_ch = wav.channels as usize;
    let tgt_ch = target_channels as usize;
    let total_frames = wav.samples.len() / src_ch.max(1);
    let start_frame = source_pts.max(0) as usize;
    let out_len = num_samples * tgt_ch;

    if start_frame >= total_frames || src_ch == 0 {
        return vec![0.0_f32; out_len];
    }

    let available = (total_frames - start_frame).min(num_samples);
    let mut out = vec![0.0_f32; out_len];

    for i in 0..available {
        let src_frame = start_frame + i;
        for c in 0..tgt_ch {
            let src_c = if src_ch == 1 {
                // Mono up-mix: replicate channel 0.
                0
            } else {
                c.min(src_ch - 1)
            };
            let src_idx = src_frame * src_ch + src_c;
            let dst_idx = i * tgt_ch + c;
            if src_idx < wav.samples.len() {
                out[dst_idx] = wav.samples[src_idx];
            }
        }
    }

    out
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smpte_bars_dimensions() {
        let bars = generate_smpte_bars(16, 8);
        assert_eq!(bars.len(), 16 * 8 * 4);
    }

    #[test]
    fn test_smpte_bars_not_all_black() {
        let bars = generate_smpte_bars(8, 4);
        let all_zero = bars.iter().all(|&b| b == 0);
        assert!(!all_zero, "SMPTE bars should not be all black");
    }

    #[test]
    fn test_generate_sine_length() {
        let samples = generate_sine(0, 480, 2, 48_000);
        assert_eq!(samples.len(), 480 * 2);
    }

    #[test]
    fn test_generate_sine_deterministic() {
        let a = generate_sine(0, 48, 1, 48_000);
        let b = generate_sine(0, 48, 1, 48_000);
        assert_eq!(a, b);
    }

    #[test]
    fn test_generate_sine_amplitude_bounded() {
        let samples = generate_sine(0, 4800, 1, 48_000);
        for s in &samples {
            assert!(s.abs() <= 0.26, "sample {s} exceeds expected amplitude");
        }
    }

    #[test]
    fn test_render_source_test_pattern_video() {
        let src = RenderSource::TestPattern;
        let frame = src.sample_video(0, 8, 4);
        assert_eq!(frame.len(), 8 * 4 * 4);
    }

    #[test]
    fn test_render_source_test_pattern_audio() {
        let src = RenderSource::TestPattern;
        let audio = src.sample_audio(0, 48, 2, 48_000);
        assert_eq!(audio.len(), 48 * 2);
    }

    #[test]
    fn test_render_source_unsupported_video_is_black() {
        let src = RenderSource::Unsupported {
            path: PathBuf::from("foo.xyz"),
        };
        let frame = src.sample_video(0, 4, 2);
        assert!(
            frame.iter().all(|&b| b == 0),
            "unsupported source should produce black frame"
        );
    }

    #[test]
    fn test_render_source_unsupported_audio_is_silence() {
        let src = RenderSource::Unsupported {
            path: PathBuf::from("foo.xyz"),
        };
        let audio = src.sample_audio(0, 48, 2, 48_000);
        assert!(
            audio.iter().all(|&s| s == 0.0),
            "unsupported source should produce silence"
        );
    }

    #[test]
    fn test_scale_nearest_identity() {
        let img = DecodedImageData {
            pixels: vec![255, 0, 0, 255, 0, 255, 0, 255],
            width: 2,
            height: 1,
        };
        let out = scale_nearest_rgba8(&img, 2, 1);
        assert_eq!(out, img.pixels);
    }

    #[test]
    fn test_scale_nearest_upscale() {
        let img = DecodedImageData {
            pixels: vec![255, 0, 0, 255],
            width: 1,
            height: 1,
        };
        let out = scale_nearest_rgba8(&img, 2, 2);
        assert_eq!(out.len(), 2 * 2 * 4);
        // All pixels should be the same red.
        for chunk in out.chunks_exact(4) {
            assert_eq!(chunk, &[255, 0, 0, 255]);
        }
    }

    #[test]
    fn test_slice_wav_silence_when_offset_past_end() {
        let wav = WavData {
            samples: vec![0.5; 16],
            sample_rate: 48_000,
            channels: 1,
        };
        let out = slice_wav_samples(&wav, 9999, 48, 2, 48_000);
        assert!(out.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_slice_wav_mono_to_stereo_upmix() {
        let wav = WavData {
            samples: vec![1.0; 8],
            sample_rate: 48_000,
            channels: 1,
        };
        let out = slice_wav_samples(&wav, 0, 4, 2, 48_000);
        // Each stereo frame: [1.0, 1.0]
        assert_eq!(out.len(), 8);
        for &s in &out {
            assert!((s - 1.0).abs() < 1e-6);
        }
    }
}
