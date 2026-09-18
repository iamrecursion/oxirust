//! Complete media processing pipeline demonstration.
//!
//! Showcases integration of OxiMedia subsystems:
//! 1. Container probing (format detection)
//! 2. Quality assessment (PSNR/SSIM)
//! 3. Loudness metering (EBU R128)
//! 4. Transcoding configuration (AV1, VP9)
//! 5. Workflow orchestration (DAG)
//! 6. Archive verification (checksum)
//! 7. Timecode handling (SMPTE)
//!
//! # Usage
//!
//! ```bash
//! cargo run --example media_pipeline \
//!   --features "quality,metering,transcode,timecode,workflow,archive" \
//!   -p oximedia
//! ```

use oximedia::prelude::*;
use oximedia::transcode::{TranscodeEstimator, TranscodePreset};
use oximedia::workflow::{
    AnalysisType, TaskBuilder, TaskPriority, TaskType, TransferProtocol, WorkflowBuilder,
};
use std::collections::HashMap;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiMedia Complete Pipeline Demo ===\n");

    // ─────────────────────────────────────────────────────────────────────────
    // Stage 1: Format Detection
    // ─────────────────────────────────────────────────────────────────────────
    println!("Stage 1: Format Detection");

    // Synthetic WebM/EBML header (0x1A 0x45 0xDF 0xA3 = EBML magic)
    let webm_header: Vec<u8> = vec![
        0x1A, 0x45, 0xDF, 0xA3, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F, 0x42, 0x86, 0x81,
        0x01,
    ];
    match probe_format(&webm_header) {
        Ok(result) => {
            println!("  Format    : {:?}", result.format);
            println!("  Confidence: {:.1}%", result.confidence * 100.0);
        }
        Err(e) => println!("  Probe error: {e}"),
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Stage 2: Quality Assessment
    // ─────────────────────────────────────────────────────────────────────────
    println!("\nStage 2: Quality Assessment");

    const W: usize = 64;
    const H: usize = 64;

    // YUV420p reference frame with luma gradient
    let mut reference = QualityFrame::new(W, H, PixelFormat::Yuv420p)?;
    for (i, px) in reference.luma_mut().iter_mut().enumerate() {
        *px = ((i * 255) / (W * H)) as u8;
    }
    if reference.chroma().is_some() {
        reference.planes[1].fill(128);
        reference.planes[2].fill(128);
    }

    // Distorted copy: +20 luma offset
    let mut distorted = reference.clone();
    for px in distorted.luma_mut() {
        *px = (*px as i16 + 20).clamp(0, 255) as u8;
    }

    let assessor = QualityAssessor::new();
    let psnr = assessor.assess(&reference, &distorted, MetricType::Psnr)?;
    let ssim = assessor.assess(&reference, &distorted, MetricType::Ssim)?;

    let quality_label = if psnr.score >= 40.0 {
        "Excellent"
    } else if psnr.score >= 30.0 {
        "Good"
    } else {
        "Poor"
    };
    println!("  Frame size : {}x{} YUV420p gradient", W, H);
    println!("  PSNR       : {:.2} dB  → {quality_label}", psnr.score);
    println!("  SSIM       : {:.4}    (1.0 = identical)", ssim.score);

    // ─────────────────────────────────────────────────────────────────────────
    // Stage 3: Audio Loudness Metering
    // ─────────────────────────────────────────────────────────────────────────
    println!("\nStage 3: Audio Loudness Metering");

    let meter_cfg = MeterConfig::new(Standard::EbuR128, 48_000.0, 2);
    let mut meter = LoudnessMeter::new(meter_cfg)?;

    // 2 s of 1 kHz sine at -18 dBFS (stereo, 48 kHz)
    let amplitude: f32 = 10.0_f32.powf(-18.0 / 20.0);
    let angular = 2.0 * std::f32::consts::PI * 1_000.0 / 48_000.0;
    let audio: Vec<f32> = (0..48_000_usize * 2)
        .flat_map(|n| {
            let s = amplitude * (angular * n as f32).sin();
            [s, s]
        })
        .collect();

    meter.process_f32(&audio);
    let metrics: LoudnessMetrics = meter.metrics();
    let compliance = meter.check_compliance();

    println!(
        "  Signal     : 1 kHz sine, 2 s, stereo, 48 kHz, {:.1} dBFS",
        20.0 * amplitude.log10() as f64
    );
    if metrics.integrated_lufs.is_finite() {
        println!("  Integrated : {:.2} LUFS", metrics.integrated_lufs);
    } else {
        println!("  Integrated : (below absolute gate — short duration)");
    }
    if metrics.short_term_lufs.is_finite() {
        println!("  Short-term : {:.2} LUFS", metrics.short_term_lufs);
    }
    println!("  Compliant  : {} (EBU R128)", compliance.is_compliant());

    // ─────────────────────────────────────────────────────────────────────────
    // Stage 4: Transcode Configuration
    // ─────────────────────────────────────────────────────────────────────────
    println!("\nStage 4: Transcode Configuration");

    let yt_cfg = TranscodePreset::YouTubeHd.into_config();
    let duration_s = 3_600.0_f64; // 1-hour programme
    let v_kbps = yt_cfg.video_bitrate.map(|b| (b / 1000) as u32);
    let a_kbps = yt_cfg.audio_bitrate.map(|b| (b / 1000) as u32);
    let size_gib = TranscodeEstimator::estimate_size_bytes(duration_s, v_kbps, a_kbps) as f64
        / (1024.0 * 1024.0 * 1024.0);

    println!(
        "  YouTubeHd  : {}",
        TranscodePreset::YouTubeHd.description()
    );
    println!("  Est. size  : {size_gib:.2} GiB for 1 h");

    let web_cfg = TranscodePreset::WebDelivery.into_config();
    println!(
        "  WebDelivery: {:?} @ {} kbps video",
        web_cfg.video_codec.as_deref().unwrap_or("vp9"),
        web_cfg.video_bitrate.unwrap_or(0) / 1000
    );

    let arch_cfg = TranscodePreset::LosslessArchive.into_config();
    println!(
        "  LosslessArchive: {:?} / {:?}",
        arch_cfg.video_codec.as_deref().unwrap_or("ffv1"),
        arch_cfg.audio_codec.as_deref().unwrap_or("flac")
    );

    // ─────────────────────────────────────────────────────────────────────────
    // Stage 5: Timecode Operations
    // ─────────────────────────────────────────────────────────────────────────
    println!("\nStage 5: Timecode");

    let tc_start = Timecode::new(1, 0, 0, 0, FrameRate::Fps25)?;
    let tc_end = Timecode::from_frames(tc_start.to_frames() + 1500, FrameRate::Fps25)?;
    let tc_df = Timecode::from_frames(tc_start.to_frames() + 1500, FrameRate::Fps2997DF)?;

    println!("  Start      : {tc_start} (25 fps NDF)");
    println!("  +1500 fr   : {tc_end}  (= +60 s at 25 fps)");
    println!("  29.97 DF   : {tc_df}  (same absolute frames)");

    // ─────────────────────────────────────────────────────────────────────────
    // Stage 6: Workflow DAG
    // ─────────────────────────────────────────────────────────────────────────
    println!("\nStage 6: Workflow DAG");

    let tmp = std::env::temp_dir().join("oximedia_pipeline_demo");
    std::fs::create_dir_all(&tmp)?;
    let db_path = tmp.join("pipeline.db");

    let workflow = WorkflowBuilder::new("pipeline-demo")
        .description("Ingest → QC → Transcode → Deliver")
        .max_concurrent_tasks(2)
        .fail_fast(true)
        .add_task(
            TaskBuilder::new(
                "ingest",
                TaskType::Analysis {
                    input: PathBuf::from("/mnt/ingest/source.mxf"),
                    analyses: vec![AnalysisType::AudioLevels, AnalysisType::VideoQuality],
                    output: Some(PathBuf::from("/tmp/report/ingest.json")),
                },
            )
            .named("ingest")
            .priority(TaskPriority::High),
        )
        .add_task(
            TaskBuilder::new(
                "qc",
                TaskType::QualityControl {
                    input: PathBuf::from("/mnt/ingest/source.mxf"),
                    profile: "broadcast_hd".to_string(),
                    rules: vec!["loudness_within_r128".to_string()],
                },
            )
            .named("qc")
            .priority(TaskPriority::High),
        )
        .add_task(
            TaskBuilder::new(
                "transcode",
                TaskType::Transcode {
                    input: PathBuf::from("/mnt/ingest/source.mxf"),
                    output: PathBuf::from("/mnt/proxy/output_av1.mkv"),
                    preset: "av1_broadcast".to_string(),
                    params: {
                        let mut p = HashMap::new();
                        p.insert("crf".to_string(), serde_json::json!(28));
                        p
                    },
                },
            )
            .named("transcode")
            .priority(TaskPriority::Normal),
        )
        .add_task(
            TaskBuilder::new(
                "deliver",
                TaskType::Transfer {
                    source: "/mnt/proxy/output_av1.mkv".to_string(),
                    destination: "s3://media-delivery/output_av1.mkv".to_string(),
                    protocol: TransferProtocol::S3,
                    options: HashMap::new(),
                },
            )
            .named("deliver")
            .priority(TaskPriority::Normal),
        )
        .depends_on("qc", "ingest")?
        .depends_on("transcode", "qc")?
        .depends_on("deliver", "transcode")?
        .build()?;

    match workflow.topological_sort() {
        Ok(order) => {
            let names: Vec<&str> = order
                .iter()
                .filter_map(|id| workflow.get_task(id))
                .map(|t| t.name.as_str())
                .collect();
            println!("  Exec order : {}", names.join(" → "));
        }
        Err(e) => println!("  Topo sort error: {e}"),
    }

    let engine = oximedia::workflow::WorkflowEngine::new(&db_path)?;
    let wf_id = engine.submit_workflow(&workflow).await?;
    println!("  Persisted  : workflow {wf_id}");
    std::fs::remove_dir_all(&tmp).unwrap_or_default();

    // ─────────────────────────────────────────────────────────────────────────
    // Stage 7: Archive Configuration
    // ─────────────────────────────────────────────────────────────────────────
    println!("\nStage 7: Archive Configuration");

    let verify_cfg = oximedia::archive::VerificationConfig {
        enable_blake3: true,
        enable_md5: false,
        enable_sha256: true,
        enable_crc32: true,
        generate_sidecars: true,
        validate_containers: true,
        enable_fixity_checks: true,
        fixity_check_interval_days: 90,
        auto_quarantine: false,
        parallel_threads: 4,
        database_path: PathBuf::from("/archive/verification.db"),
        quarantine_dir: PathBuf::from("/archive/quarantine"),
        enable_premis_logging: true,
        enable_bagit: false,
    };

    println!("  BLAKE3     : {}", verify_cfg.enable_blake3);
    println!("  SHA-256    : {}", verify_cfg.enable_sha256);
    println!(
        "  Fixity     : every {} days",
        verify_cfg.fixity_check_interval_days
    );
    println!("  Threads    : {}", verify_cfg.parallel_threads);
    println!("  PREMIS log : {}", verify_cfg.enable_premis_logging);

    println!("\n=== Pipeline Demo Complete ===");
    println!("All 7 stages executed successfully.");
    Ok(())
}
