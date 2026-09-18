//! Top-level `oximedia quality` subcommand.
//!
//! Provides dedicated video/image quality assessment using full-reference metrics
//! (PSNR, SSIM, MS-SSIM, VIF, FSIM, VMAF) and no-reference metrics
//! (NIQE, BRISQUE, blockiness, blur, noise).
//!
//! Uses `oximedia-quality` for all metric computation.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use oximedia_core::PixelFormat;
use oximedia_quality::{Frame, MetricType, QualityAssessor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Subcommand enum
// ---------------------------------------------------------------------------

/// Subcommands for `oximedia quality`.
#[derive(Subcommand, Debug)]
pub enum QualityCommand {
    /// Compare reference and distorted video/image files with full-reference metrics
    Compare {
        /// Reference (original) file
        #[arg(long)]
        reference: PathBuf,

        /// Distorted (encoded/processed) file
        #[arg(long)]
        distorted: PathBuf,

        /// Metrics to compute (comma-separated list)
        ///
        /// Full-reference: psnr, ssim, ms-ssim, vif, fsim, vmaf
        /// No-reference: niqe, brisque, blockiness, blur, noise
        #[arg(long, default_value = "psnr,ssim")]
        metrics: String,

        /// Output format: text or json
        #[arg(long, default_value = "text")]
        output_format: String,

        /// Frame dimensions width (pixels) — used for synthetic frame creation
        #[arg(long, default_value = "1920")]
        width: usize,

        /// Frame dimensions height (pixels)
        #[arg(long, default_value = "1080")]
        height: usize,
    },

    /// Analyze a single file with no-reference quality metrics
    Analyze {
        /// Input media file or image
        #[arg(short, long)]
        input: PathBuf,

        /// No-reference metrics to compute (comma-separated)
        ///
        /// Supported: niqe, brisque, blockiness, blur, noise
        #[arg(long, default_value = "brisque,blockiness,blur,noise")]
        metrics: String,

        /// Output format: text or json
        #[arg(long, default_value = "text")]
        output_format: String,
    },

    /// List all available quality metrics and their descriptions
    List,

    /// Explain a specific quality metric in detail
    Explain {
        /// Metric name (psnr, ssim, vmaf, brisque, etc.)
        #[arg(value_name = "METRIC")]
        metric: String,
    },
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Entry point called from `main.rs`.
pub async fn run_quality(command: QualityCommand, json_output: bool, ndjson: bool) -> Result<()> {
    match command {
        QualityCommand::Compare {
            reference,
            distorted,
            metrics,
            output_format,
            width,
            height,
        } => {
            if ndjson {
                colored::control::set_override(false);
                return cmd_compare_ndjson(&reference, &distorted, &metrics, width, height).await;
            }
            let fmt = if json_output { "json" } else { &output_format };
            cmd_compare(&reference, &distorted, &metrics, fmt, width, height).await
        }

        QualityCommand::Analyze {
            input,
            metrics,
            output_format,
        } => {
            if ndjson {
                colored::control::set_override(false);
                return cmd_analyze_ndjson(&input, &metrics).await;
            }
            let fmt = if json_output { "json" } else { &output_format };
            cmd_analyze(&input, &metrics, fmt).await
        }

        QualityCommand::List => cmd_list(json_output),

        QualityCommand::Explain { metric } => cmd_explain(&metric, json_output),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a comma-separated metric list into `Vec<MetricType>`.
fn parse_metrics(metrics_str: &str) -> Result<Vec<MetricType>> {
    let mut result = Vec::new();
    for raw in metrics_str.split(',') {
        let name = raw.trim().to_lowercase();
        let metric = match name.as_str() {
            "psnr" => MetricType::Psnr,
            "ssim" => MetricType::Ssim,
            "ms-ssim" | "msssim" | "ms_ssim" => MetricType::MsSsim,
            "vmaf" => MetricType::Vmaf,
            "vif" => MetricType::Vif,
            "fsim" => MetricType::Fsim,
            "niqe" => MetricType::Niqe,
            "brisque" => MetricType::Brisque,
            "blockiness" | "block" => MetricType::Blockiness,
            "blur" => MetricType::Blur,
            "noise" => MetricType::Noise,
            other => {
                return Err(anyhow::anyhow!(
                    "Unknown metric '{other}'. Use `oximedia quality list` to see all available metrics."
                ))
            }
        };
        result.push(metric);
    }
    if result.is_empty() {
        return Err(anyhow::anyhow!("No metrics specified."));
    }
    Ok(result)
}

/// BT.601 integer luma of one RGB pixel: `y = (299*r + 587*g + 114*b) / 1000`.
///
/// Weights sum to 1000, so the result is already in `0..=255`; the clamp is
/// defensive only. Matches the conversion used by `probe --quality-snapshot`.
fn bt601_luma(r: u8, g: u8, b: u8) -> u8 {
    let y = (299 * i32::from(r) + 587 * i32::from(g) + 114 * i32::from(b)) / 1_000;
    y.clamp(0, 255) as u8
}

/// Convert packed RGB24 bytes into a single-plane `Gray8` quality [`Frame`].
///
/// All video/image quality metrics read `frame.planes[0]` (the luma plane),
/// so a full RGB→YUV conversion is unnecessary.
fn rgb24_to_gray_frame(rgb: &[u8], width: u32, height: u32) -> Result<Frame> {
    if width == 0 || height == 0 {
        return Err(anyhow::anyhow!(
            "decoded frame has zero dimensions ({width}x{height})"
        ));
    }
    let pixel_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| anyhow::anyhow!("frame dimensions overflow: {width}x{height}"))?;
    let expected_len = pixel_count
        .checked_mul(3)
        .ok_or_else(|| anyhow::anyhow!("frame dimensions overflow: {width}x{height}"))?;
    if rgb.len() < expected_len {
        return Err(anyhow::anyhow!(
            "RGB buffer too short: got {} bytes, need {expected_len} for {width}x{height}",
            rgb.len()
        ));
    }

    let mut frame = Frame::new(width as usize, height as usize, PixelFormat::Gray8)
        .map_err(|e| anyhow::anyhow!("failed to allocate Gray8 frame: {e}"))?;
    let luma = frame.luma_mut();
    for (px, out) in rgb[..expected_len].chunks_exact(3).zip(luma.iter_mut()) {
        *out = bt601_luma(px[0], px[1], px[2]);
    }
    Ok(frame)
}

/// Decode frame 0 of `path` and return it as a `Gray8` quality [`Frame`].
///
/// This is a **real decode** of the actual media content via
/// [`crate::frame_extract::extract_video_frame_rgb`] (Y4M natively;
/// MP4/MKV/WebM/TS through the demuxer + AV1/VP9/VP8 pipeline). Audio-only or
/// undecodable input yields an honest error — never a synthetic stand-in.
async fn decode_gray_frame(path: &Path) -> Result<Frame> {
    let (rgb, width, height) = crate::frame_extract::extract_video_frame_rgb(path, 0)
        .await
        .with_context(|| format!("Failed to decode a video frame from {}", path.display()))?;
    rgb24_to_gray_frame(&rgb, width, height)
}

/// Nearest-neighbour resize of a single-plane `Gray8` frame to `dw`x`dh`.
///
/// Full-reference metrics require identical dimensions; when a distorted input
/// was encoded at a different resolution than the reference, it is resampled to
/// the reference size before assessment (standard full-reference practice).
fn resize_gray_frame(src: &Frame, dw: usize, dh: usize) -> Result<Frame> {
    let mut out = Frame::new(dw, dh, PixelFormat::Gray8)
        .map_err(|e| anyhow::anyhow!("failed to allocate resized frame: {e}"))?;
    let (sw, sh) = (src.width, src.height);
    if sw == 0 || sh == 0 {
        return Err(anyhow::anyhow!("source frame has zero dimensions"));
    }
    let src_luma = src.luma();
    let dst = out.luma_mut();
    for y in 0..dh {
        let sy = (y * sh / dh).min(sh - 1);
        for x in 0..dw {
            let sx = (x * sw / dw).min(sw - 1);
            dst[y * dw + x] = src_luma[sy * sw + sx];
        }
    }
    Ok(out)
}

/// Outcome of one metric: a real score, or a reason it could not be computed
/// (e.g. a no-reference size guard on a small frame). Never fabricated.
struct MetricScore {
    metric: MetricType,
    score: Option<f64>,
    reason: Option<String>,
    components: HashMap<String, f64>,
}

/// Result of a full-reference `compare` over two real decoded frames.
struct CompareOutcome {
    width: usize,
    height: usize,
    distorted_resized: bool,
    scores: Vec<MetricScore>,
}

/// Result of a no-reference `analyze` over one real decoded frame.
struct AnalyzeOutcome {
    width: usize,
    height: usize,
    scores: Vec<MetricScore>,
}

/// Decode both inputs and compute the requested full-reference metrics on the
/// ACTUAL pixels. Per-metric failures degrade gracefully (recorded with a
/// reason); a whole-input decode failure returns an honest error.
async fn compare_scores(
    reference: &Path,
    distorted: &Path,
    metrics: &[MetricType],
) -> Result<CompareOutcome> {
    let ref_frame = decode_gray_frame(reference).await.with_context(|| {
        format!(
            "Cannot assess quality: reference {} did not yield a decodable video frame",
            reference.display()
        )
    })?;
    let (width, height) = (ref_frame.width, ref_frame.height);

    let mut dist_frame = decode_gray_frame(distorted).await.with_context(|| {
        format!(
            "Cannot assess quality: distorted {} did not yield a decodable video frame",
            distorted.display()
        )
    })?;

    let mut distorted_resized = false;
    if dist_frame.width != width || dist_frame.height != height {
        dist_frame = resize_gray_frame(&dist_frame, width, height)?;
        distorted_resized = true;
    }

    let assessor = QualityAssessor::new();
    let mut scores = Vec::with_capacity(metrics.len());
    for &metric in metrics {
        match assessor.assess(&ref_frame, &dist_frame, metric) {
            Ok(s) => scores.push(MetricScore {
                metric,
                score: Some(s.score),
                reason: None,
                components: s.components.clone(),
            }),
            Err(e) => scores.push(MetricScore {
                metric,
                score: None,
                reason: Some(e.to_string()),
                components: HashMap::new(),
            }),
        }
    }

    Ok(CompareOutcome {
        width,
        height,
        distorted_resized,
        scores,
    })
}

/// Decode the input and compute the requested no-reference metrics on the
/// ACTUAL frame. Per-metric size-guard failures degrade gracefully.
async fn analyze_scores(input: &Path, metrics: &[MetricType]) -> Result<AnalyzeOutcome> {
    let frame = decode_gray_frame(input).await.with_context(|| {
        format!(
            "Cannot analyze quality: {} did not yield a decodable video frame",
            input.display()
        )
    })?;
    let (width, height) = (frame.width, frame.height);

    let assessor = QualityAssessor::new();
    let mut scores = Vec::with_capacity(metrics.len());
    for &metric in metrics {
        match assessor.assess_no_reference(&frame, metric) {
            Ok(s) => scores.push(MetricScore {
                metric,
                score: Some(s.score),
                reason: None,
                components: s.components.clone(),
            }),
            Err(e) => scores.push(MetricScore {
                metric,
                score: None,
                reason: Some(e.to_string()),
                components: HashMap::new(),
            }),
        }
    }

    Ok(AnalyzeOutcome {
        width,
        height,
        scores,
    })
}

/// Metric display name and scale description.
fn metric_display_info(metric: MetricType) -> (&'static str, &'static str) {
    match metric {
        MetricType::Psnr => ("PSNR", "dB (higher = better; ≥40 dB: excellent)"),
        MetricType::Ssim => ("SSIM", "0–1 (higher = better; ≥0.95: excellent)"),
        MetricType::MsSsim => ("MS-SSIM", "0–1 (higher = better)"),
        MetricType::Vmaf => ("VMAF", "0–100 (higher = better; ≥90: excellent)"),
        MetricType::Vif => ("VIF", "0–1 (higher = better)"),
        MetricType::Fsim => ("FSIM", "0–1 (higher = better)"),
        MetricType::Niqe => ("NIQE", "lower = better (natural images ~3–5)"),
        MetricType::Brisque => ("BRISQUE", "0–100 (lower = better)"),
        MetricType::Blockiness => ("Blockiness", "0–1 (lower = better)"),
        MetricType::Blur => ("Blur", "0–1 (lower = better)"),
        MetricType::Noise => ("Noise", "0–1 (lower = better)"),
        // Forward-compatible: handle any future variants added to the non-exhaustive enum
        _ => ("Unknown", "see oximedia quality list"),
    }
}

// ---------------------------------------------------------------------------
// Compare
// ---------------------------------------------------------------------------

async fn cmd_compare(
    reference: &PathBuf,
    distorted: &PathBuf,
    metrics_str: &str,
    output_format: &str,
    width: usize,
    height: usize,
) -> Result<()> {
    if !reference.exists() {
        return Err(anyhow::anyhow!(
            "Reference file not found: {}",
            reference.display()
        ));
    }
    if !distorted.exists() {
        return Err(anyhow::anyhow!(
            "Distorted file not found: {}",
            distorted.display()
        ));
    }

    let metrics = parse_metrics(metrics_str)?;

    // Validate all metrics are full-reference (or allowed for compare)
    for m in &metrics {
        if m.is_no_reference() {
            return Err(anyhow::anyhow!(
                "Metric '{m:?}' is a no-reference metric and cannot be used with `compare`. \
                 Use `oximedia quality analyze` instead."
            ));
        }
    }

    // Decode a real frame from each input and assess on the ACTUAL pixels.
    let _ = (width, height); // legacy args; real frame dimensions are used now.
    let outcome = compare_scores(reference.as_path(), distorted.as_path(), &metrics).await?;

    if output_format == "json" {
        let results: Vec<serde_json::Value> = outcome
            .scores
            .iter()
            .map(|ms| {
                let (name, scale) = metric_display_info(ms.metric);
                serde_json::json!({
                    "metric": name,
                    "score": ms.score,
                    "unavailable": ms.reason.as_deref(),
                    "scale": scale,
                    "components": ms.components,
                })
            })
            .collect();
        let output = serde_json::json!({
            "command": "quality compare",
            "reference": reference.display().to_string(),
            "distorted": distorted.display().to_string(),
            "frame_dimensions": { "width": outcome.width, "height": outcome.height },
            "distorted_resized_to_reference": outcome.distorted_resized,
            "metrics": results,
        });
        let s = serde_json::to_string_pretty(&output).context("JSON serialization failed")?;
        println!("{s}");
        return Ok(());
    }

    // Human-readable
    println!("{}", "Quality Comparison".green().bold());
    println!("{}", "=".repeat(60));
    println!("{:20} {}", "Reference:", reference.display());
    println!("{:20} {}", "Distorted:", distorted.display());
    println!(
        "{:20} {}×{} (decoded frame 0)",
        "Frame size:", outcome.width, outcome.height
    );
    if outcome.distorted_resized {
        println!(
            "{:20} {}",
            "Note:",
            "distorted frame resized to reference resolution for comparison".dimmed()
        );
    }
    println!();
    println!("{}", "Results".cyan().bold());
    println!("{}", "-".repeat(60));
    println!("{:<12} {:>12}  Scale", "Metric", "Score");
    println!("{}", "-".repeat(60));

    for ms in &outcome.scores {
        let (name, scale) = metric_display_info(ms.metric);
        match ms.score {
            Some(score) => {
                let score_str = format!("{score:.4}").yellow().to_string();
                println!("{:<12} {:>12}  {}", name, score_str, scale.dimmed());
            }
            None => {
                let reason = ms.reason.as_deref().unwrap_or("unavailable");
                println!("{:<12} {:>12}  {}", name, "n/a".dimmed(), reason.dimmed());
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// NDJSON helpers for Compare and Analyze
// ---------------------------------------------------------------------------

/// Emit one NDJSON record per metric for `quality compare`.
async fn cmd_compare_ndjson(
    reference: &PathBuf,
    distorted: &PathBuf,
    metrics_str: &str,
    width: usize,
    height: usize,
) -> Result<()> {
    if !reference.exists() {
        return Err(anyhow::anyhow!(
            "Reference file not found: {}",
            reference.display()
        ));
    }
    if !distorted.exists() {
        return Err(anyhow::anyhow!(
            "Distorted file not found: {}",
            distorted.display()
        ));
    }

    let metrics = parse_metrics(metrics_str)?;
    let _ = (width, height); // legacy args; real frame dimensions are used now.
    let outcome = compare_scores(reference.as_path(), distorted.as_path(), &metrics).await?;

    let mut writer = crate::output::NdjsonWriter::new(std::io::stdout());
    for ms in &outcome.scores {
        let (name, scale) = metric_display_info(ms.metric);
        let record = serde_json::json!({
            "metric": name,
            "score": ms.score,
            "unavailable": ms.reason.as_deref(),
            "scale": scale,
            "reference": reference.display().to_string(),
            "distorted": distorted.display().to_string(),
            "width": outcome.width,
            "height": outcome.height,
        });
        writer
            .emit(&record)
            .context("Failed to write NDJSON quality record")?;
    }
    Ok(())
}

/// Emit one NDJSON record per metric for `quality analyze` (no-reference).
async fn cmd_analyze_ndjson(input: &PathBuf, metrics_str: &str) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    let metrics = parse_metrics(metrics_str)?;
    for m in &metrics {
        if m.requires_reference() {
            return Err(anyhow::anyhow!(
                "Metric '{m:?}' requires a reference file. Use `oximedia quality compare` instead."
            ));
        }
    }

    let outcome = analyze_scores(input.as_path(), &metrics).await?;

    let mut writer = crate::output::NdjsonWriter::new(std::io::stdout());
    for ms in &outcome.scores {
        let (name, scale) = metric_display_info(ms.metric);
        let record = serde_json::json!({
            "metric": name,
            "score": ms.score,
            "unavailable": ms.reason.as_deref(),
            "scale": scale,
            "input": input.display().to_string(),
            "width": outcome.width,
            "height": outcome.height,
        });
        writer
            .emit(&record)
            .context("Failed to write NDJSON quality record")?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Analyze (no-reference)
// ---------------------------------------------------------------------------

async fn cmd_analyze(input: &PathBuf, metrics_str: &str, output_format: &str) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    let metrics = parse_metrics(metrics_str)?;

    // Warn if any full-reference metric was requested
    for m in &metrics {
        if m.requires_reference() {
            return Err(anyhow::anyhow!(
                "Metric '{m:?}' requires a reference file. Use `oximedia quality compare` instead."
            ));
        }
    }

    // Decode a real frame from the input and score no-reference metrics on it.
    let outcome = analyze_scores(input.as_path(), &metrics).await?;

    let file_size = std::fs::metadata(input)
        .with_context(|| format!("Cannot stat: {}", input.display()))
        .map(|m| m.len())
        .unwrap_or(0);

    if output_format == "json" {
        let results: Vec<serde_json::Value> = outcome
            .scores
            .iter()
            .map(|ms| {
                let (name, scale) = metric_display_info(ms.metric);
                serde_json::json!({
                    "metric": name,
                    "score": ms.score,
                    "unavailable": ms.reason.as_deref(),
                    "scale": scale,
                })
            })
            .collect();
        let output = serde_json::json!({
            "command": "quality analyze",
            "input": input.display().to_string(),
            "file_size_bytes": file_size,
            "frame_dimensions": { "width": outcome.width, "height": outcome.height },
            "metrics": results,
        });
        let s = serde_json::to_string_pretty(&output).context("JSON serialization failed")?;
        println!("{s}");
        return Ok(());
    }

    println!("{}", "Quality Analysis (No-Reference)".green().bold());
    println!("{}", "=".repeat(60));
    println!("{:20} {}", "Input:", input.display());
    println!("{:20} {} bytes", "File size:", file_size);
    println!(
        "{:20} {}×{} (decoded frame 0)",
        "Frame size:", outcome.width, outcome.height
    );
    println!();
    println!("{}", "Results".cyan().bold());
    println!("{}", "-".repeat(60));
    println!("{:<14} {:>10}  Scale", "Metric", "Score");
    println!("{}", "-".repeat(60));

    for ms in &outcome.scores {
        let (name, scale) = metric_display_info(ms.metric);
        match ms.score {
            Some(score) => {
                println!(
                    "{:<14} {:>10.4}  {}",
                    name,
                    score.to_string().yellow(),
                    scale.dimmed()
                );
            }
            None => {
                let reason = ms.reason.as_deref().unwrap_or("unavailable");
                println!("{:<14} {:>10}  {}", name, "n/a".dimmed(), reason.dimmed());
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// List
// ---------------------------------------------------------------------------

fn cmd_list(json_output: bool) -> Result<()> {
    let metrics = [
        (
            MetricType::Psnr,
            "Full-reference",
            "Peak Signal-to-Noise Ratio",
        ),
        (
            MetricType::Ssim,
            "Full-reference",
            "Structural Similarity Index",
        ),
        (MetricType::MsSsim, "Full-reference", "Multi-Scale SSIM"),
        (
            MetricType::Vmaf,
            "Full-reference",
            "Video Multi-Method Assessment Fusion",
        ),
        (
            MetricType::Vif,
            "Full-reference",
            "Visual Information Fidelity",
        ),
        (
            MetricType::Fsim,
            "Full-reference",
            "Feature Similarity Index",
        ),
        (
            MetricType::Niqe,
            "No-reference",
            "Natural Image Quality Evaluator",
        ),
        (
            MetricType::Brisque,
            "No-reference",
            "Blind/Referenceless Image Spatial Quality Evaluator",
        ),
        (
            MetricType::Blockiness,
            "No-reference",
            "DCT-based blockiness detection",
        ),
        (
            MetricType::Blur,
            "No-reference",
            "Laplacian variance blur detection",
        ),
        (
            MetricType::Noise,
            "No-reference",
            "Spatial/temporal noise estimation",
        ),
    ];

    if json_output {
        let list: Vec<serde_json::Value> = metrics
            .iter()
            .map(|(m, kind, desc)| {
                let (name, scale) = metric_display_info(*m);
                serde_json::json!({
                    "name": name,
                    "kind": kind,
                    "description": desc,
                    "scale": scale,
                })
            })
            .collect();
        let result = serde_json::json!({ "metrics": list });
        let s = serde_json::to_string_pretty(&result).context("JSON serialization failed")?;
        println!("{s}");
        return Ok(());
    }

    println!("{}", "Available Quality Metrics".green().bold());
    println!("{}", "=".repeat(72));
    println!("{:<14} {:<18} Description", "Name", "Type");
    println!("{}", "-".repeat(72));

    for (m, kind, desc) in &metrics {
        let (name, _) = metric_display_info(*m);
        println!("{:<14} {:<18} {}", name, kind, desc);
    }

    println!();
    println!(
        "{}",
        "Full-reference metrics require both --reference and --distorted.".dimmed()
    );
    println!(
        "{}",
        "No-reference metrics work on a single --input file.".dimmed()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Explain
// ---------------------------------------------------------------------------

fn cmd_explain(metric_str: &str, json_output: bool) -> Result<()> {
    let metric_name = metric_str.trim().to_lowercase();
    let (metric, long_desc) = match metric_name.as_str() {
        "psnr" => (
            MetricType::Psnr,
            "PSNR (Peak Signal-to-Noise Ratio) measures the ratio between the \
             maximum possible power of a signal and the power of corrupting noise \
             that affects the fidelity of its representation. Expressed in decibels \
             (dB). Higher values indicate better quality. Typical values: ≥40 dB = \
             excellent, 30–40 dB = good, 20–30 dB = fair, <20 dB = poor. PSNR \
             correlates well with quality for compression artifacts but less so for \
             blurring or structural distortions.",
        ),
        "ssim" => (
            MetricType::Ssim,
            "SSIM (Structural Similarity Index) measures image quality by comparing \
             luminance, contrast, and structure between a reference and distorted \
             image. Ranges from -1 (inverse) to 1 (identical). Values ≥0.95 are \
             generally considered excellent. SSIM is more perceptually accurate than \
             PSNR for many types of distortion.",
        ),
        "ms-ssim" | "msssim" | "ms_ssim" => (
            MetricType::MsSsim,
            "MS-SSIM (Multi-Scale SSIM) extends SSIM by evaluating structural \
             similarity at multiple spatial scales. This better accounts for \
             the viewing distance and display resolution effects on perceptual quality.",
        ),
        "vmaf" => (
            MetricType::Vmaf,
            "VMAF (Video Multi-Method Assessment Fusion) is a full-reference \
             perceptual video quality metric developed by Netflix. It uses a \
             machine-learning model trained on human opinion scores, combining \
             VIF (Visual Information Fidelity), DLM (Detail Loss Metric), and \
             motion features. Score range: 0–100; ≥90 = excellent, 70–90 = good, \
             <70 = noticeable quality loss.",
        ),
        "vif" => (
            MetricType::Vif,
            "VIF (Visual Information Fidelity) quantifies the amount of visual \
             information preserved in the distorted image relative to the reference, \
             based on a natural scene statistics model in the wavelet domain. Values \
             range from 0 (no information) to 1 (perfect fidelity).",
        ),
        "fsim" => (
            MetricType::Fsim,
            "FSIM (Feature Similarity Index) measures quality by comparing salient \
             features (phase congruency and gradient magnitude) between reference and \
             distorted images. Ranges 0–1; higher is better.",
        ),
        "niqe" => (
            MetricType::Niqe,
            "NIQE (Natural Image Quality Evaluator) is a no-reference metric that \
             measures deviation from the statistical regularities of natural images \
             using a multivariate Gaussian model. Lower scores indicate more natural \
             (higher quality) images. Pristine images typically score 3–5.",
        ),
        "brisque" => (
            MetricType::Brisque,
            "BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) is a \
             no-reference metric that uses a natural scene statistics model on \
             spatial domain features. Score range 0–100; lower is better quality. \
             Typically: <20 = excellent, 20–40 = good, 40–60 = fair, >60 = poor.",
        ),
        "blockiness" | "block" => (
            MetricType::Blockiness,
            "Blockiness detection quantifies DCT-based compression blocking artifacts \
             that appear as visible 8×8 or 16×16 block boundaries in H.264/AV1 \
             encoded video. Score 0–1; lower is better.",
        ),
        "blur" => (
            MetricType::Blur,
            "Blur detection uses Laplacian variance to measure the sharpness of an \
             image. Lower variance indicates more blur. Score 0–1; lower indicates \
             more blurring.",
        ),
        "noise" => (
            MetricType::Noise,
            "Noise estimation quantifies spatial and temporal noise artifacts. \
             Combines measurements of grain-like high-frequency components across \
             frames. Score 0–1; lower indicates less noise.",
        ),
        other => {
            return Err(anyhow::anyhow!(
            "Unknown metric '{other}'. Use `oximedia quality list` to see all available metrics."
        ))
        }
    };

    let (name, scale) = metric_display_info(metric);
    let kind = if metric.requires_reference() {
        "Full-reference"
    } else {
        "No-reference"
    };

    if json_output {
        let result = serde_json::json!({
            "metric": name,
            "kind": kind,
            "scale": scale,
            "description": long_desc,
        });
        let s = serde_json::to_string_pretty(&result).context("JSON serialization failed")?;
        println!("{s}");
        return Ok(());
    }

    println!("{} {}", "Metric:".green().bold(), name.yellow().bold());
    println!("{}", "=".repeat(60));
    println!("{:10} {}", "Type:", kind);
    println!("{:10} {}", "Scale:", scale);
    println!();
    println!("{}", "Description".cyan().bold());
    println!("{}", "-".repeat(60));

    // Word-wrap at 58 chars
    let mut line_buf = String::new();
    for word in long_desc.split_whitespace() {
        if line_buf.len() + word.len() + 1 > 58 {
            println!("  {line_buf}");
            line_buf = word.to_string();
        } else {
            if !line_buf.is_empty() {
                line_buf.push(' ');
            }
            line_buf.push_str(word);
        }
    }
    if !line_buf.is_empty() {
        println!("  {line_buf}");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_metrics_valid() {
        let metrics = parse_metrics("psnr,ssim").expect("should parse");
        assert_eq!(metrics.len(), 2);
        assert!(metrics.contains(&MetricType::Psnr));
        assert!(metrics.contains(&MetricType::Ssim));
    }

    #[test]
    fn test_parse_metrics_no_reference() {
        let metrics = parse_metrics("brisque,blur,noise").expect("should parse");
        assert_eq!(metrics.len(), 3);
        for m in &metrics {
            assert!(m.is_no_reference());
        }
    }

    #[test]
    fn test_parse_metrics_unknown() {
        let result = parse_metrics("psnr,unknown_metric");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_metrics_empty() {
        let result = parse_metrics("  ");
        // space-only splits to one empty token → unknown
        assert!(result.is_err());
    }

    /// Write a single-frame C420jpeg Y4M whose luma is produced by `luma_fn`
    /// and whose chroma is neutral (U=V=128). With neutral chroma the real
    /// YUV→RGB→BT.601-luma round-trip is exact, so the decoded Gray8 frame
    /// equals `luma_fn` — a deterministic real-decode fixture.
    fn write_y4m_frame(
        name: &str,
        w: usize,
        h: usize,
        luma_fn: impl Fn(usize, usize) -> u8,
    ) -> PathBuf {
        let mut data = Vec::new();
        data.extend_from_slice(format!("YUV4MPEG2 W{w} H{h} F25:1 Ip C420jpeg\n").as_bytes());
        data.extend_from_slice(b"FRAME\n");
        for y in 0..h {
            for x in 0..w {
                data.push(luma_fn(x, y));
            }
        }
        let chroma_len = w.div_ceil(2) * h.div_ceil(2);
        data.extend(std::iter::repeat_n(128u8, chroma_len * 2));
        let path = std::env::temp_dir().join(format!(
            "oximedia_quality_{}_{}.y4m",
            name,
            std::process::id()
        ));
        std::fs::write(&path, &data).expect("write Y4M fixture");
        path
    }

    #[test]
    fn test_bt601_luma_grey_identity() {
        for v in [0u8, 1, 64, 128, 200, 255] {
            assert_eq!(bt601_luma(v, v, v), v);
        }
    }

    #[test]
    fn test_rgb24_to_gray_frame_pixel_exact() {
        let (w, h) = (5u32, 3u32);
        let mut rgb = Vec::new();
        for i in 0..(w * h) as usize {
            rgb.extend_from_slice(&[
                (i * 3 % 256) as u8,
                (i * 5 % 256) as u8,
                (i * 7 % 256) as u8,
            ]);
        }
        let frame = rgb24_to_gray_frame(&rgb, w, h).expect("convert");
        assert_eq!(frame.width, 5);
        assert_eq!(frame.height, 3);
        for (i, &luma) in frame.luma().iter().enumerate() {
            let px = &rgb[i * 3..i * 3 + 3];
            assert_eq!(luma, bt601_luma(px[0], px[1], px[2]));
        }
    }

    /// Quality-bar proof: comparing two DIFFERENT real images yields different
    /// scores than comparing identical ones — the metrics run on the ACTUAL
    /// decoded pixels (the old code returned the same scores for any input).
    #[tokio::test]
    async fn test_compare_identical_vs_different_real_frames() {
        let a = write_y4m_frame("cmp_a", 96, 96, |x, y| ((x * 7 + y * 13) % 251) as u8);
        let a_copy = write_y4m_frame("cmp_acopy", 96, 96, |x, y| ((x * 7 + y * 13) % 251) as u8);
        let b = write_y4m_frame("cmp_b", 96, 96, |_, _| 128);

        let metrics = vec![MetricType::Psnr, MetricType::Ssim];
        let identical = compare_scores(&a, &a_copy, &metrics)
            .await
            .expect("compare identical");
        let different = compare_scores(&a, &b, &metrics)
            .await
            .expect("compare different");

        std::fs::remove_file(&a).ok();
        std::fs::remove_file(&a_copy).ok();
        std::fs::remove_file(&b).ok();

        // Real, not hardcoded 1920x1080.
        assert_eq!(identical.width, 96);
        assert_eq!(identical.height, 96);

        let score = |o: &CompareOutcome, m: MetricType| -> f64 {
            o.scores
                .iter()
                .find(|s| s.metric == m)
                .and_then(|s| s.score)
                .expect("metric score present")
        };
        let psnr_id = score(&identical, MetricType::Psnr);
        let psnr_diff = score(&different, MetricType::Psnr);
        let ssim_id = score(&identical, MetricType::Ssim);
        let ssim_diff = score(&different, MetricType::Ssim);

        assert!(
            psnr_id > psnr_diff,
            "PSNR identical ({psnr_id}) must exceed different ({psnr_diff})"
        );
        assert!(
            ssim_id > ssim_diff,
            "SSIM identical ({ssim_id}) must exceed different ({ssim_diff})"
        );
    }

    #[tokio::test]
    async fn test_analyze_real_frame_scores() {
        let img = write_y4m_frame("an_real", 96, 96, |x, y| ((x * 7 + y * 13) % 251) as u8);
        let metrics = vec![MetricType::Blur, MetricType::Noise, MetricType::Blockiness];
        let outcome = analyze_scores(&img, &metrics)
            .await
            .expect("analyze real frame");
        std::fs::remove_file(&img).ok();

        assert_eq!(outcome.width, 96);
        assert_eq!(outcome.height, 96);
        for ms in &outcome.scores {
            let s = ms
                .score
                .unwrap_or_else(|| panic!("{:?} should score: {:?}", ms.metric, ms.reason));
            assert!(s.is_finite(), "{:?} score must be finite", ms.metric);
        }
    }

    #[tokio::test]
    async fn test_analyze_audio_only_errors_honestly() {
        // A WAV carries no video frame — analyze must honestly error, never
        // fabricate a score on a synthetic frame.
        let path =
            std::env::temp_dir().join(format!("oximedia_quality_audio_{}.wav", std::process::id()));
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&36u32.to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&8000u32.to_le_bytes());
        wav.extend_from_slice(&16000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&0u32.to_le_bytes());
        std::fs::write(&path, &wav).expect("write wav");

        let result = analyze_scores(&path, &[MetricType::Blur]).await;
        std::fs::remove_file(&path).ok();
        assert!(
            result.is_err(),
            "audio-only input must not yield fabricated scores"
        );
    }

    #[test]
    fn test_cmd_list_no_panic() {
        assert!(cmd_list(false).is_ok());
        assert!(cmd_list(true).is_ok());
    }

    #[test]
    fn test_cmd_explain_psnr() {
        assert!(cmd_explain("psnr", true).is_ok());
    }

    #[test]
    fn test_cmd_explain_vmaf() {
        assert!(cmd_explain("vmaf", false).is_ok());
    }

    #[test]
    fn test_cmd_explain_unknown() {
        assert!(cmd_explain("xyz_unknown", false).is_err());
    }

    #[tokio::test]
    async fn test_cmd_compare_missing_reference() {
        let result = cmd_compare(
            &std::env::temp_dir().join("nonexistent_ref_12345.mkv"),
            &std::env::temp_dir().join("nonexistent_dist_12345.mkv"),
            "psnr",
            "text",
            1920,
            1080,
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_cmd_compare_no_reference_metric_rejected() {
        let dir = std::env::temp_dir();
        let ref_path = dir.join("oximedia_quality_ref_test.mkv");
        let dist_path = dir.join("oximedia_quality_dist_test.mkv");
        std::fs::write(&ref_path, b"ref").expect("write ok");
        std::fs::write(&dist_path, b"dist").expect("write ok");
        // brisque is no-reference — should be rejected in compare mode
        let result = cmd_compare(&ref_path, &dist_path, "brisque", "text", 1920, 1080).await;
        assert!(result.is_err());
        std::fs::remove_file(&ref_path).ok();
        std::fs::remove_file(&dist_path).ok();
    }

    #[tokio::test]
    async fn test_cmd_compare_psnr_ssim_json() {
        let ref_path =
            write_y4m_frame("cmd_cmp_ref", 96, 96, |x, y| ((x * 7 + y * 13) % 251) as u8);
        let dist_path = write_y4m_frame("cmd_cmp_dist", 96, 96, |_, _| 128);
        let result = cmd_compare(&ref_path, &dist_path, "psnr,ssim", "json", 64, 64).await;
        assert!(result.is_ok(), "unexpected error: {result:?}");
        std::fs::remove_file(&ref_path).ok();
        std::fs::remove_file(&dist_path).ok();
    }

    #[tokio::test]
    async fn test_cmd_analyze_missing_file() {
        let result = cmd_analyze(
            &std::env::temp_dir().join("nonexistent_analyze_12345.mkv"),
            "blur",
            "text",
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_cmd_analyze_no_reference_json() {
        let path = write_y4m_frame("cmd_analyze", 96, 96, |x, y| ((x * 5 + y * 11) % 241) as u8);
        let result = cmd_analyze(&path, "blur,noise", "json").await;
        assert!(result.is_ok(), "unexpected error: {result:?}");
        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_cmd_analyze_full_ref_metric_rejected() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_quality_full_ref_reject.mkv");
        std::fs::write(&path, b"stub").expect("write ok");
        let result = cmd_analyze(&path, "psnr", "text").await;
        assert!(result.is_err());
        std::fs::remove_file(&path).ok();
    }
}
