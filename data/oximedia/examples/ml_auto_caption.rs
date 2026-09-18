//! Sovereign ML auto-caption example — Whisper-style speech tokenisation.
//!
//! Demonstrates the [`oximedia::ml`] auto-caption pipeline that shipped in the
//! 0.1.5 ML roadmap. Two operating modes:
//!
//! 1. **Device-probe demo** (no arguments) — enumerates every backend compiled
//!    into the current build and prints capability metadata. This is the path
//!    CI always runs.
//! 2. **Real inference** — given `encoder.onnx`, `decoder.onnx`, `input.wav`
//!    on the command line, the example loads the WAV via a small dependency-
//!    free RIFF/PCM reader, downmixes to mono `f32`, builds an
//!    [`AutoCaptionPipeline`] with Whisper-compatible defaults from
//!    [`AutoCaptionConfig::default`], runs greedy decoding via
//!    [`AutoCaptionPipeline::caption`], and prints the resulting token id
//!    sequence.
//!
//! The WAV input is expected at the configured Whisper sample rate (16 kHz by
//! default); the example surfaces a helpful error when the rates differ. Token
//! ids are model-specific; mapping them to text needs a BPE/SentencePiece
//! tokeniser, handled by `oximedia-captions` in a full caption-emit workflow.
//!
//! # Usage
//!
//! ```bash
//! # Device-probe only (always succeeds):
//! cargo run -p oximedia --example ml_auto_caption --features ml
//!
//! # Real inference (requires the `ml-onnx` feature plus ONNX files):
//! cargo run -p oximedia --example ml_auto_caption --features "ml ml-onnx" \
//!     -- whisper-encoder.onnx whisper-decoder.onnx clip.wav
//! ```

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use oximedia::prelude::*;
use oximedia_ml::pipelines::auto_caption::{AutoCaptionConfig, AutoCaptionPipeline};

fn format_bytes(bytes: Option<u64>) -> String {
    bytes.map_or_else(|| "-".to_string(), |b| format!("{b} B"))
}

/// Parsed `fmt ` chunk metadata for the inline WAV reader below.
struct WavInfo {
    sample_rate: u32,
    channels: u16,
    bits_per_sample: u16,
    /// `true` for IEEE float (format tag 3), `false` for PCM (tag 1).
    is_float: bool,
}

/// Minimal RIFF/WAVE reader supporting PCM (8/16/24/32-bit) and IEEE float
/// (32/64-bit). Returns interleaved `f32` samples in `[-1.0, 1.0]`.
fn read_wav_f32(path: &Path) -> Result<(WavInfo, Vec<f32>), Box<dyn std::error::Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hdr = [0u8; 12];
    reader.read_exact(&mut hdr)?;
    if &hdr[0..4] != b"RIFF" || &hdr[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }

    let mut info: Option<WavInfo> = None;
    let mut samples: Vec<f32> = Vec::new();

    loop {
        let mut chunk_hdr = [0u8; 8];
        match reader.read_exact(&mut chunk_hdr) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(Box::new(e)),
        }
        let id: [u8; 4] = [chunk_hdr[0], chunk_hdr[1], chunk_hdr[2], chunk_hdr[3]];
        let size =
            u32::from_le_bytes([chunk_hdr[4], chunk_hdr[5], chunk_hdr[6], chunk_hdr[7]]) as usize;
        let padded = size + (size & 1);
        let mut buf = vec![0u8; padded];
        reader.read_exact(&mut buf)?;
        let payload = &buf[..size];
        match &id {
            b"fmt " => {
                if payload.len() < 16 {
                    return Err("fmt chunk too short".into());
                }
                let mut tag = u16::from_le_bytes([payload[0], payload[1]]);
                let channels = u16::from_le_bytes([payload[2], payload[3]]);
                let sample_rate =
                    u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]);
                let bits = u16::from_le_bytes([payload[14], payload[15]]);
                // WAVE_FORMAT_EXTENSIBLE: actual tag lives in the GUID prefix.
                if tag == 0xFFFE && payload.len() >= 26 {
                    tag = u16::from_le_bytes([payload[24], payload[25]]);
                }
                info = Some(WavInfo {
                    sample_rate,
                    channels,
                    bits_per_sample: bits,
                    is_float: tag == 3,
                });
            }
            b"data" => {
                let spec = info.as_ref().ok_or("data chunk before fmt")?;
                samples = decode_samples(payload, spec)?;
            }
            _ => {}
        }
    }
    let info = info.ok_or("missing fmt chunk")?;
    Ok((info, samples))
}

fn decode_samples(raw: &[u8], spec: &WavInfo) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bps = (spec.bits_per_sample / 8) as usize;
    if bps == 0 {
        return Err("bits_per_sample must be > 0".into());
    }
    let n = raw.len() / bps;
    let mut out = Vec::with_capacity(n);
    match (spec.is_float, spec.bits_per_sample) {
        (false, 8) => out.extend(raw.iter().map(|&b| (b as f32 - 128.0) / 128.0)),
        (false, 16) => out.extend(raw.chunks_exact(2).map(|c| {
            let v = i16::from_le_bytes([c[0], c[1]]);
            f32::from(v) / 32768.0
        })),
        (false, 24) => out.extend(raw.chunks_exact(3).map(|c| {
            let mut v = i32::from(c[0]) | (i32::from(c[1]) << 8) | (i32::from(c[2]) << 16);
            if v & 0x0080_0000 != 0 {
                v |= !0x00FF_FFFF; // sign-extend
            }
            v as f32 / 8_388_608.0
        })),
        (false, 32) => out.extend(raw.chunks_exact(4).map(|c| {
            let v = i32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            v as f32 / 2_147_483_648.0
        })),
        (true, 32) => out.extend(
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        ),
        (true, 64) => out.extend(raw.chunks_exact(8).map(|c| {
            let bits = u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
            f64::from_bits(bits) as f32
        })),
        (is_float, bits) => {
            return Err(format!("unsupported WAV format: float={is_float} bits={bits}").into())
        }
    }
    Ok(out)
}

/// Downmix interleaved multi-channel `f32` samples to mono by averaging
/// across the channels of each frame. A mono input is returned unchanged.
fn downmix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    let channels = channels.max(1);
    if channels == 1 {
        return interleaved.to_vec();
    }
    let frames = interleaved.len() / channels;
    let inv = 1.0_f32 / channels as f32;
    (0..frames)
        .map(|f| {
            let start = f * channels;
            let sum: f32 = interleaved[start..start + channels].iter().sum();
            sum * inv
        })
        .collect()
}

/// Print the device capability matrix; reused as the always-on preamble.
fn print_device_matrix() {
    let device = DeviceType::auto();
    println!(
        "Auto-selected inference device: {device} ({})",
        device.display_name()
    );

    let capabilities = DeviceCapabilities::probe_all();
    println!("\nDevice capability probe:");
    println!(
        "  {:<10} {:<12} {:<28} {:<10} {:<6} {:<6} {:<6}",
        "Backend", "Status", "Device Name", "Memory", "fp16", "bf16", "int8"
    );
    println!("  {}", "-".repeat(80));
    for caps in &capabilities {
        let status = if caps.is_available {
            "available"
        } else {
            "unavailable"
        };
        println!(
            "  {:<10} {:<12} {:<28} {:<10} {:<6} {:<6} {:<6}",
            caps.device_type.name(),
            status,
            caps.device_name,
            format_bytes(caps.memory_total_bytes),
            caps.supports_fp16,
            caps.supports_bf16,
            caps.supports_int8,
        );
    }

    let best = DeviceCapabilities::best_available();
    println!(
        "\nBest available: {} -- int8:{} fp16:{} bf16:{}",
        best, best.supports_int8, best.supports_fp16, best.supports_bf16,
    );
}

fn run_real_inference(
    encoder: PathBuf,
    decoder: PathBuf,
    audio: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = AutoCaptionConfig::default();
    println!("\nLoading audio: {}", audio.display());
    let (info, interleaved) = read_wav_f32(&audio)?;
    println!(
        "  format: {} Hz, {} ch, {}-bit{}",
        info.sample_rate,
        info.channels,
        info.bits_per_sample,
        if info.is_float { " float" } else { "" }
    );
    if info.sample_rate != cfg.sample_rate {
        return Err(format!(
            "WAV is {} Hz but pipeline expects {} Hz (resample externally with `oximedia-audio` \
             or ffmpeg before invoking this example)",
            info.sample_rate, cfg.sample_rate
        )
        .into());
    }
    let samples = downmix_to_mono(&interleaved, info.channels as usize);
    println!("  downmixed to {} mono samples", samples.len());

    println!("\nLoading encoder: {}", encoder.display());
    println!("Loading decoder: {}", decoder.display());
    let pipeline = match AutoCaptionPipeline::new(&encoder, &decoder, cfg) {
        Ok(p) => p,
        Err(MlError::FeatureDisabled(feat)) => {
            println!(
                "\nAutoCaption requires the '{feat}' feature. \
                 Rebuild with `--features \"ml ml-onnx\"`."
            );
            return Ok(());
        }
        Err(e) => return Err(Box::new(e)),
    };

    println!("\nRunning greedy decode loop...");
    match pipeline.caption(&samples) {
        Ok(tokens) => {
            println!("Generated {} token ids:", tokens.len());
            let preview: Vec<u32> = tokens.iter().take(32).copied().collect();
            println!("  first {}: {:?}", preview.len(), preview);
            if tokens.len() > preview.len() {
                println!("  (+{} more)", tokens.len() - preview.len());
            }
            println!(
                "\nNote: token ids map to text via a model-specific tokenizer\n\
                 (BPE / SentencePiece). The OxiMedia captions crate composes this\n\
                 pipeline with a tokeniser to emit SRT/WebVTT cues."
            );
        }
        Err(MlError::FeatureDisabled(feat)) => {
            println!(
                "Decoding step disabled (missing feature '{feat}'). \
                 Rebuild with `--features \"ml ml-onnx\"`."
            );
        }
        Err(e) => return Err(Box::new(e)),
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OxiMedia -- ML Auto-Caption");
    println!("===========================\n");

    print_device_matrix();

    let args: Vec<String> = std::env::args().collect();
    match (args.get(1), args.get(2), args.get(3)) {
        (Some(enc), Some(dec), Some(wav)) => {
            run_real_inference(PathBuf::from(enc), PathBuf::from(dec), PathBuf::from(wav))?;
        }
        _ => {
            println!(
                "\nNo model paths provided; device-probe demo complete.\n\
                 Pass `encoder.onnx decoder.onnx input.wav` to run inference:\n\
                   cargo run -p oximedia --example ml_auto_caption \\\n\
                       --features \"ml ml-onnx\" -- enc.onnx dec.onnx clip.wav"
            );
        }
    }

    Ok(())
}
