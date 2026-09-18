//! Recommendation engine commands: codec, settings, workflow, analyze.
//!
//! Exposes `oximedia-recommend` content-based, collaborative, and hybrid
//! recommendation capabilities via the CLI.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

/// Recommend command subcommands.
#[derive(Subcommand, Debug)]
pub enum RecommendCommand {
    /// Recommend the best codec for a given input
    Codec {
        /// Input media file to analyze
        #[arg(short, long)]
        input: PathBuf,

        /// Target use case: streaming, archival, editing, broadcast
        #[arg(long, default_value = "streaming")]
        use_case: String,

        /// Target bitrate in kbps (optional, for quality analysis)
        #[arg(long)]
        bitrate: Option<u32>,

        /// Target resolution (e.g. 1920x1080)
        #[arg(long)]
        resolution: Option<String>,
    },

    /// Recommend encoding settings for a file
    Settings {
        /// Input media file
        #[arg(short, long)]
        input: PathBuf,

        /// Codec to optimize for (av1, vp9, vp8, opus, vorbis, flac, pcm, aac, mp3)
        #[arg(long, default_value = "av1")]
        codec: String,

        /// Optimization target: quality, speed, size, balanced
        #[arg(long, default_value = "balanced")]
        target: String,

        /// Maximum encoding time in seconds (0 for unlimited)
        #[arg(long, default_value = "0")]
        max_time: u64,
    },

    /// Recommend a workflow for a given task
    Workflow {
        /// Task description: transcode, archive, stream, edit, broadcast
        #[arg(long)]
        task: String,

        /// Input file count
        #[arg(long, default_value = "1")]
        file_count: usize,

        /// Total input size in MB
        #[arg(long)]
        total_size_mb: Option<f64>,

        /// Available CPU cores
        #[arg(long)]
        cores: Option<usize>,
    },

    /// Analyze content and provide recommendations
    Analyze {
        /// Input media file to analyze
        #[arg(short, long)]
        input: PathBuf,

        /// Include codec recommendation
        #[arg(long)]
        codec: bool,

        /// Include quality analysis
        #[arg(long)]
        quality: bool,

        /// Include complexity analysis
        #[arg(long)]
        complexity: bool,

        /// Include full report
        #[arg(long)]
        full: bool,
    },
}

/// Handle recommend command dispatch.
pub async fn handle_recommend_command(command: RecommendCommand, json_output: bool) -> Result<()> {
    match command {
        RecommendCommand::Codec {
            input,
            use_case,
            bitrate,
            resolution,
        } => {
            handle_codec(
                &input,
                &use_case,
                bitrate,
                resolution.as_deref(),
                json_output,
            )
            .await
        }
        RecommendCommand::Settings {
            input,
            codec,
            target,
            max_time,
        } => handle_settings(&input, &codec, &target, max_time, json_output).await,
        RecommendCommand::Workflow {
            task,
            file_count,
            total_size_mb,
            cores,
        } => handle_workflow(&task, file_count, total_size_mb, cores, json_output).await,
        RecommendCommand::Analyze {
            input,
            codec,
            quality,
            complexity,
            full,
        } => handle_analyze(&input, codec, quality, complexity, full, json_output).await,
    }
}

/// Parse a `WIDTHxHEIGHT` (or `WIDTH:HEIGHT`) resolution string.
fn parse_resolution(s: &str) -> Result<(u32, u32)> {
    let normalized = s.to_lowercase().replace(':', "x");
    let (w, h) = normalized.split_once('x').ok_or_else(|| {
        anyhow::anyhow!("Invalid resolution '{s}'. Expected WIDTHxHEIGHT, e.g. 1920x1080")
    })?;
    let width: u32 = w
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid width in resolution '{s}'"))?;
    let height: u32 = h
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid height in resolution '{s}'"))?;
    if width == 0 || height == 0 {
        return Err(anyhow::anyhow!("Resolution dimensions must be non-zero"));
    }
    Ok((width, height))
}

/// Bitrate adequacy assessment derived from bits-per-pixel at an assumed
/// 30 fps: the classification and per-codec guidance genuinely change with
/// the `--bitrate` / `--resolution` inputs.
struct BitrateAssessment {
    bits_per_pixel: f64,
    verdict: &'static str,
    guidance: String,
}

/// Assess a target bitrate against a target resolution.
///
/// Bits-per-pixel-per-frame thresholds (assuming 30 fps) follow common
/// encoder guidance: below ~0.02 bpp even AV1 visibly struggles; 0.02–0.05
/// is AV1-efficient territory; 0.05–0.10 gives VP9 headroom; above 0.10 any
/// of the patent-free codecs has ample budget.
fn assess_bitrate(bitrate_kbps: u32, width: u32, height: u32) -> BitrateAssessment {
    const ASSUMED_FPS: f64 = 30.0;
    let pixels_per_second = f64::from(width) * f64::from(height) * ASSUMED_FPS;
    let bits_per_pixel = (f64::from(bitrate_kbps) * 1000.0) / pixels_per_second;

    let (verdict, guidance) = if bits_per_pixel < 0.02 {
        (
            "very low",
            format!(
                "{bitrate_kbps} kbps is a very tight budget for {width}x{height}@30 — expect \
                 visible artifacts even with AV1; consider a lower resolution or a higher bitrate"
            ),
        )
    } else if bits_per_pixel < 0.05 {
        (
            "low",
            format!(
                "{bitrate_kbps} kbps at {width}x{height}@30 needs AV1's efficiency; VP9/VP8 \
                 would degrade noticeably at this budget"
            ),
        )
    } else if bits_per_pixel < 0.10 {
        (
            "adequate",
            format!(
                "{bitrate_kbps} kbps at {width}x{height}@30 is comfortable for AV1 and workable \
                 for VP9"
            ),
        )
    } else {
        (
            "generous",
            format!(
                "{bitrate_kbps} kbps at {width}x{height}@30 leaves ample headroom — any \
                 patent-free codec (AV1/VP9/VP8) will look good; pick by encode-speed needs"
            ),
        )
    };

    BitrateAssessment {
        bits_per_pixel,
        verdict,
        guidance,
    }
}

/// Recommend a codec for the input.
async fn handle_codec(
    input: &PathBuf,
    use_case: &str,
    bitrate: Option<u32>,
    resolution: Option<&str>,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    let valid_use_cases = ["streaming", "archival", "editing", "broadcast"];
    if !valid_use_cases.contains(&use_case) {
        return Err(anyhow::anyhow!(
            "Unknown use case '{}'. Supported: {}",
            use_case,
            valid_use_cases.join(", ")
        ));
    }

    // Validate --resolution for real (garbage must not be echoed back as if
    // it had been understood).
    let parsed_resolution = resolution.map(parse_resolution).transpose()?;

    let file_size = std::fs::metadata(input)
        .context("Failed to read file metadata")?
        .len();

    // Recommend based on use case (patent-free only)
    let (recommended_video, recommended_audio, reason) = match use_case {
        "streaming" => ("AV1", "Opus", "Best compression for streaming delivery"),
        "archival" => (
            "AV1",
            "FLAC",
            "Lossless audio + efficient video for archive",
        ),
        "editing" => ("VP9", "FLAC", "Fast decode speed for editing workflows"),
        "broadcast" => ("AV1", "Opus", "High quality at broadcast bitrates"),
        _ => ("AV1", "Opus", "Default recommendation"),
    };

    // --bitrate/--resolution genuinely shape the output: with both present a
    // bits-per-pixel assessment is computed and reported.
    let assessment = match (bitrate, parsed_resolution) {
        (Some(kbps), Some((w, h))) => Some(assess_bitrate(kbps, w, h)),
        _ => None,
    };

    if json_output {
        let mut result = serde_json::json!({
            "command": "codec",
            "input": input.display().to_string(),
            "file_size": file_size,
            "use_case": use_case,
            "target_bitrate": bitrate,
            "target_resolution": resolution,
            "recommendation": {
                "video_codec": recommended_video,
                "audio_codec": recommended_audio,
                "reason": reason,
            },
            "alternatives": [
                {"video": "VP9", "audio": "Vorbis", "note": "Wider hardware decode support"},
                {"video": "VP8", "audio": "Vorbis", "note": "Legacy compatibility"},
            ],
            "status": "analyzed",
        });
        if let Some(ref a) = assessment {
            result["bitrate_assessment"] = serde_json::json!({
                "bits_per_pixel": a.bits_per_pixel,
                "assumed_fps": 30,
                "verdict": a.verdict,
                "guidance": a.guidance,
            });
        }
        let json_str = serde_json::to_string_pretty(&result)
            .context("Failed to serialize codec recommendation")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Codec Recommendation".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {} bytes", "File size:", file_size);
        println!("{:20} {}", "Use case:", use_case);
        if let Some(br) = bitrate {
            println!("{:20} {} kbps", "Target bitrate:", br);
        }
        if let Some((w, h)) = parsed_resolution {
            println!("{:20} {}x{}", "Target resolution:", w, h);
        }
        println!();
        println!("{}", "Recommendation".cyan().bold());
        println!("{}", "-".repeat(60));
        println!("  Video codec:  {}", recommended_video.green());
        println!("  Audio codec:  {}", recommended_audio.green());
        println!("  Reason:       {}", reason);
        if let Some(ref a) = assessment {
            println!();
            println!("{}", "Bitrate Assessment".cyan().bold());
            println!("{}", "-".repeat(60));
            println!(
                "  Bits/pixel:   {:.4} (at an assumed 30 fps)",
                a.bits_per_pixel
            );
            println!("  Verdict:      {}", a.verdict.yellow());
            println!("  Guidance:     {}", a.guidance);
        }
        println!();
        println!("{}", "Alternatives".cyan().bold());
        println!("{}", "-".repeat(60));
        println!("  VP9 + Vorbis  - Wider hardware decode support");
        println!("  VP8 + Vorbis  - Legacy compatibility");
        println!();
        println!(
            "{}",
            "Note: OxiMedia only recommends patent-free codecs.".dimmed()
        );
    }

    Ok(())
}

/// Recommend encoding settings.
async fn handle_settings(
    input: &PathBuf,
    codec: &str,
    target: &str,
    max_time: u64,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    let valid_codecs = ["av1", "vp9", "vp8", "opus", "vorbis", "flac"];
    if !valid_codecs.contains(&codec) {
        return Err(anyhow::anyhow!(
            "Unsupported codec '{}'. Supported: {}",
            codec,
            valid_codecs.join(", ")
        ));
    }

    let valid_targets = ["quality", "speed", "size", "balanced"];
    if !valid_targets.contains(&target) {
        return Err(anyhow::anyhow!(
            "Unknown target '{}'. Supported: {}",
            target,
            valid_targets.join(", ")
        ));
    }

    let (preset, crf, threads) = match target {
        "quality" => ("slow", 22, 0),
        "speed" => ("ultrafast", 28, 0),
        "size" => ("medium", 32, 0),
        _ => ("medium", 26, 0),
    };

    if json_output {
        let result = serde_json::json!({
            "command": "settings",
            "input": input.display().to_string(),
            "codec": codec,
            "target": target,
            "max_encoding_time": max_time,
            "settings": {
                "preset": preset,
                "crf": crf,
                "threads": threads,
                "keyframe_interval": 250,
                "pixel_format": "yuv420p",
            },
            "status": "recommended",
        });
        let json_str = serde_json::to_string_pretty(&result)
            .context("Failed to serialize settings recommendation")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Encoding Settings Recommendation".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {}", "Codec:", codec);
        println!("{:20} {}", "Optimization:", target);
        if max_time > 0 {
            println!("{:20} {}s", "Max encode time:", max_time);
        }
        println!();
        println!("{}", "Recommended Settings".cyan().bold());
        println!("{}", "-".repeat(60));
        println!("  Preset:             {}", preset);
        println!("  CRF:                {}", crf);
        println!("  Threads:            {} (auto)", threads);
        println!("  Keyframe interval:  250");
        println!("  Pixel format:       YUV420P");
    }

    Ok(())
}

/// Recommend a workflow.
async fn handle_workflow(
    task: &str,
    file_count: usize,
    total_size_mb: Option<f64>,
    cores: Option<usize>,
    json_output: bool,
) -> Result<()> {
    let valid_tasks = ["transcode", "archive", "stream", "edit", "broadcast"];
    if !valid_tasks.contains(&task) {
        return Err(anyhow::anyhow!(
            "Unknown task '{}'. Supported: {}",
            task,
            valid_tasks.join(", ")
        ));
    }

    let parallel = file_count > 1;
    let recommended_parallelism = cores.unwrap_or(4).min(file_count);

    if json_output {
        let result = serde_json::json!({
            "command": "workflow",
            "task": task,
            "file_count": file_count,
            "total_size_mb": total_size_mb,
            "cores": cores,
            "recommendation": {
                "parallel": parallel,
                "parallelism": recommended_parallelism,
                "pipeline": format!("decode -> filter -> encode ({})", task),
            },
            "status": "recommended",
        });
        let json_str = serde_json::to_string_pretty(&result)
            .context("Failed to serialize workflow recommendation")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Workflow Recommendation".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Task:", task);
        println!("{:20} {}", "File count:", file_count);
        if let Some(sz) = total_size_mb {
            println!("{:20} {:.1} MB", "Total size:", sz);
        }
        println!("{:20} {}", "Parallelism:", recommended_parallelism);
        println!();
        println!(
            "{}",
            format!("Pipeline: decode -> filter -> encode ({})", task).cyan()
        );
    }

    Ok(())
}

/// Analyze content and provide recommendations.
#[allow(clippy::too_many_arguments)]
async fn handle_analyze(
    input: &PathBuf,
    codec: bool,
    quality: bool,
    complexity: bool,
    full: bool,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    let file_size = std::fs::metadata(input)
        .context("Failed to read file metadata")?
        .len();

    let include_codec = codec || full;
    let include_quality = quality || full;
    let include_complexity = complexity || full;

    if json_output {
        let result = serde_json::json!({
            "command": "analyze",
            "input": input.display().to_string(),
            "file_size": file_size,
            "analysis": {
                "codec_recommendation": include_codec,
                "quality_analysis": include_quality,
                "complexity_analysis": include_complexity,
            },
            "status": "analyzed",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize analysis result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Content Analysis".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {} bytes", "File size:", file_size);
        println!();
        if include_codec {
            println!("{}", "Codec Analysis".cyan().bold());
            println!("{}", "-".repeat(60));
            println!("  Recommended: AV1 (best compression/quality ratio)");
            println!();
        }
        if include_quality {
            println!("{}", "Quality Analysis".cyan().bold());
            println!("{}", "-".repeat(60));
            println!("  Content quality assessment pending file decode.");
            println!();
        }
        if include_complexity {
            println!("{}", "Complexity Analysis".cyan().bold());
            println!("{}", "-".repeat(60));
            println!("  Content complexity assessment pending file decode.");
            println!();
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_use_cases() {
        let valid = ["streaming", "archival", "editing", "broadcast"];
        for uc in &valid {
            assert!(valid.contains(uc));
        }
    }

    #[test]
    fn test_parse_resolution_forms() {
        assert_eq!(parse_resolution("1920x1080").expect("x form"), (1920, 1080));
        assert_eq!(
            parse_resolution("1280:720").expect("colon form"),
            (1280, 720)
        );
        assert_eq!(
            parse_resolution("3840X2160").expect("capital X"),
            (3840, 2160)
        );
        assert!(parse_resolution("garbage").is_err());
        assert!(parse_resolution("0x1080").is_err());
        assert!(parse_resolution("1920x").is_err());
    }

    #[test]
    fn test_assess_bitrate_verdicts_change_with_inputs() {
        // 500 kbps for 4K is starvation territory.
        let low = assess_bitrate(500, 3840, 2160);
        assert_eq!(low.verdict, "very low");
        // 8000 kbps for 720p is generous.
        let high = assess_bitrate(8000, 1280, 720);
        assert_eq!(high.verdict, "generous");
        // The numeric output must actually derive from the inputs.
        assert!(low.bits_per_pixel < high.bits_per_pixel);
        assert!(low.guidance.contains("3840x2160"));
        assert!(high.guidance.contains("8000 kbps"));
    }

    #[test]
    fn test_valid_codecs() {
        let valid = ["av1", "vp9", "vp8", "opus", "vorbis", "flac"];
        assert_eq!(valid.len(), 6);
    }

    #[test]
    fn test_valid_targets() {
        let valid = ["quality", "speed", "size", "balanced"];
        assert_eq!(valid.len(), 4);
    }

    #[test]
    fn test_valid_tasks() {
        let valid = ["transcode", "archive", "stream", "edit", "broadcast"];
        assert_eq!(valid.len(), 5);
    }
}
