//! Top-level `oximedia ml` subcommand — sovereign ML pipeline surface.
//!
//! Provides three verbs:
//!
//! | Verb   | What it does                                                                                   |
//! |--------|------------------------------------------------------------------------------------------------|
//! | `list` | Enumerate every built-in pipeline + the entries registered in `oximedia_ml::ModelZoo::default`. |
//! | `probe`| Print a colored table of every compiled-in device backend (CPU / CUDA / WebGPU / DirectML / CoreML) and whether it is usable at runtime. |
//! | `run`  | Load a pipeline against an ONNX `.onnx` file and an input media path, optionally `--dry-run` to validate inputs without touching the ONNX runtime. |
//!
//! The whole module is compiled unconditionally so the clap dispatcher
//! always accepts `oximedia ml ...`. When the `ml` feature is disabled,
//! the handler body bails with a clear "rebuild with --features ml"
//! message instead of clap reporting an unknown subcommand.

use anyhow::Result;
use clap::Subcommand;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Subcommand enum
// ---------------------------------------------------------------------------

/// Subcommands for `oximedia ml`.
#[derive(Subcommand, Debug)]
pub enum MlCommand {
    /// List built-in ML pipelines and registered model zoo entries.
    List {
        /// Emit machine-readable JSON instead of the colored text table.
        #[arg(long)]
        json: bool,
    },

    /// Probe ML execution device availability (CPU / CUDA / WebGPU / DirectML / CoreML).
    Probe {
        /// Limit the probe to a single device by name (cpu, cuda, webgpu, directml, coreml).
        ///
        /// When omitted, every compiled-in backend is probed and rendered
        /// as a table (or a JSON array with `--json`).
        #[arg(long, value_name = "DEVICE")]
        device: Option<String>,

        /// Emit the probe result as JSON instead of the colored text table.
        #[arg(long)]
        json: bool,
    },

    /// Run a typed pipeline end-to-end against an input media file.
    Run {
        /// Pipeline identifier.
        ///
        /// One of: scene-classifier, shot-boundary, aesthetic-score,
        /// object-detector, face-embedder. Use `oximedia ml list` to see
        /// which are available in the current build.
        #[arg(long, value_name = "NAME")]
        pipeline: String,

        /// Path to the `.onnx` model file backing the pipeline.
        #[arg(long, value_name = "PATH")]
        model: PathBuf,

        /// Path to the input image / video frame / audio file for the pipeline.
        #[arg(long, value_name = "PATH")]
        input: PathBuf,

        /// Execution device (auto, cpu, cuda, webgpu, directml, coreml).
        #[arg(long, default_value = "auto", value_name = "DEVICE")]
        device: String,

        /// Classification top-K (scene-classifier only).
        #[arg(long, value_name = "N")]
        top_k: Option<usize>,

        /// Shot-boundary / detection confidence threshold in [0.0, 1.0].
        #[arg(long, value_name = "F")]
        threshold: Option<f32>,

        /// Emit the result as JSON instead of the colored text summary.
        #[arg(long)]
        json: bool,

        /// Validate inputs + print the pipeline metadata without invoking
        /// the ONNX runtime. Useful for smoke-testing CLI wiring in
        /// builds where the `onnx` feature is off.
        #[arg(long)]
        dry_run: bool,
    },
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Entry point dispatched from `main.rs`.
///
/// `global_json` is the top-level `--json` flag; each subcommand also
/// carries a local `--json` so users can opt in without setting the
/// global flag. We treat either as "emit JSON".
pub async fn run_ml(command: MlCommand, global_json: bool) -> Result<()> {
    match command {
        MlCommand::List { json } => cmd_list(global_json || json),
        MlCommand::Probe { device, json } => cmd_probe(device.as_deref(), global_json || json),
        MlCommand::Run {
            pipeline,
            model,
            input,
            device,
            top_k,
            threshold,
            json,
            dry_run,
        } => cmd_run(
            &pipeline,
            &model,
            &input,
            &device,
            top_k,
            threshold,
            global_json || json,
            dry_run,
        ),
    }
}

// ===========================================================================
// Feature-gated handler bodies
// ===========================================================================

#[cfg(not(feature = "ml"))]
fn cmd_list(_json: bool) -> Result<()> {
    bail_feature_disabled()
}

#[cfg(not(feature = "ml"))]
fn cmd_probe(_device: Option<&str>, _json: bool) -> Result<()> {
    bail_feature_disabled()
}

#[cfg(not(feature = "ml"))]
#[allow(clippy::too_many_arguments)]
fn cmd_run(
    _pipeline: &str,
    _model: &std::path::Path,
    _input: &std::path::Path,
    _device: &str,
    _top_k: Option<usize>,
    _threshold: Option<f32>,
    _json: bool,
    _dry_run: bool,
) -> Result<()> {
    bail_feature_disabled()
}

#[cfg(not(feature = "ml"))]
fn bail_feature_disabled() -> Result<()> {
    anyhow::bail!("ML subcommand requires building with --features ml")
}

// ===========================================================================
// Real implementation (only compiled with --features ml)
// ===========================================================================

#[cfg(feature = "ml")]
mod ml_impl {
    use super::*;
    use colored::Colorize;
    use oximedia_ml::{DeviceCapabilities, DeviceType, ModelZoo, PipelineInfo, PipelineTask};
    use std::path::Path;

    // -----------------------------------------------------------------
    // Pipeline catalogue
    // -----------------------------------------------------------------

    /// Static metadata for a pipeline surfaced by `oximedia ml list`.
    struct PipelineSpec {
        id: &'static str,
        name: &'static str,
        task: PipelineTask,
        input_size: Option<(u32, u32)>,
        available: bool,
        feature: &'static str,
    }

    /// Enumerate every built-in pipeline. Entries whose pipeline feature
    /// is disabled at build time still appear (with `available = false`)
    /// so users can discover what is available across the full zoo.
    ///
    /// The `available` flag reflects whether the `oximedia-ml` crate was
    /// compiled with that pipeline feature on. We currently detect this
    /// at the CLI layer via `ml-all-pipelines`: enable that feature to
    /// turn on every pipeline simultaneously. Finer-grained CLI features
    /// (one per pipeline) can be added later without changing this
    /// function's shape.
    fn built_in_pipelines() -> Vec<PipelineSpec> {
        let all_on = cfg!(feature = "ml-all-pipelines");
        vec![
            PipelineSpec {
                id: "scene-classifier/places365",
                name: "Scene Classifier",
                task: PipelineTask::SceneClassification,
                input_size: Some((224, 224)),
                available: all_on,
                feature: "scene-classifier",
            },
            PipelineSpec {
                id: "shot-boundary/transnet-v2",
                name: "Shot Boundary Detector",
                task: PipelineTask::ShotBoundary,
                input_size: Some((48, 27)),
                available: all_on,
                feature: "shot-boundary",
            },
            PipelineSpec {
                id: "aesthetic-score/nima",
                name: "Aesthetic Scorer (NIMA)",
                task: PipelineTask::AestheticScoring,
                input_size: Some((224, 224)),
                available: all_on,
                feature: "aesthetic-score",
            },
            PipelineSpec {
                id: "object-detector/yolov8",
                name: "Object Detector (YOLOv8)",
                task: PipelineTask::Detection,
                input_size: Some((640, 640)),
                available: all_on,
                feature: "object-detector",
            },
            PipelineSpec {
                id: "face-embedder/arcface",
                name: "Face Embedder (ArcFace)",
                task: PipelineTask::FaceEmbedding,
                input_size: Some((112, 112)),
                available: all_on,
                feature: "face-embedder",
            },
        ]
    }

    pub(super) fn list(json: bool) -> Result<()> {
        let pipelines = built_in_pipelines();
        let zoo = ModelZoo::with_defaults();

        if json {
            let pipelines_json: Vec<serde_json::Value> = pipelines
                .iter()
                .map(|p| {
                    serde_json::json!({
                        "id": p.id,
                        "name": p.name,
                        "task": task_str(p.task),
                        "input_size": p.input_size.map(|(w, h)| serde_json::json!([w, h])),
                        "available": p.available,
                        "feature": p.feature,
                    })
                })
                .collect();

            let zoo_json: Vec<serde_json::Value> = zoo
                .entries()
                .map(|e| {
                    serde_json::json!({
                        "id": e.id,
                        "name": e.name,
                        "task": task_str(e.task),
                        "input_size": e.input_size.map(|(w, h)| serde_json::json!([w, h])),
                        "num_classes": e.num_classes,
                        "notes": e.notes,
                    })
                })
                .collect();

            let out = serde_json::json!({
                "command": "ml list",
                "pipelines": pipelines_json,
                "model_zoo": zoo_json,
            });
            println!("{}", serde_json::to_string_pretty(&out)?);
            return Ok(());
        }

        println!("{}", "Built-in ML Pipelines".bold().cyan());
        println!("{}", "=====================".cyan());
        println!();
        println!(
            "  {:<30} {:<26} {:<12} {:<10} {}",
            "ID".bold(),
            "Task".bold(),
            "Input".bold(),
            "Status".bold(),
            "Feature".bold(),
        );
        for p in &pipelines {
            let input_str = p
                .input_size
                .map(|(w, h)| format!("{}x{}", w, h))
                .unwrap_or_else(|| "—".to_string());
            let status = if p.available {
                "available".green().to_string()
            } else {
                "disabled".dimmed().to_string()
            };
            println!(
                "  {:<30} {:<26} {:<12} {:<10} {}",
                p.id,
                task_str(p.task),
                input_str,
                status,
                p.feature.dimmed(),
            );
        }

        println!();
        println!("{}", "Registered Model Zoo Entries".bold().cyan());
        println!("{}", "============================".cyan());
        println!();
        if zoo.is_empty() {
            println!("  {}", "(no entries)".dimmed());
        } else {
            // Collect into a Vec so output is stable (HashMap iteration order is arbitrary).
            let mut entries: Vec<_> = zoo.entries().collect();
            entries.sort_by_key(|e| e.id);
            for e in entries {
                let input_str = e
                    .input_size
                    .map(|(w, h)| format!("{}x{}", w, h))
                    .unwrap_or_else(|| "—".to_string());
                let classes = e
                    .num_classes
                    .map(|n| format!("{} classes", n))
                    .unwrap_or_else(|| "—".to_string());
                println!(
                    "  {:<30} {:<36} {:<10} {}",
                    e.id.green(),
                    e.name,
                    input_str,
                    classes.dimmed(),
                );
                if !e.notes.is_empty() {
                    println!("      {} {}", "note:".dimmed(), e.notes.dimmed());
                }
            }
        }

        Ok(())
    }

    pub(super) fn probe(device_filter: Option<&str>, json: bool) -> Result<()> {
        let caps: Vec<DeviceCapabilities> = if let Some(name) = device_filter {
            let device = parse_device_name(name)?;
            vec![DeviceCapabilities::probe(device)]
        } else {
            DeviceCapabilities::probe_all()
        };

        if json {
            let payload = serde_json::json!({
                "command": "ml probe",
                "devices": caps,
            });
            println!("{}", serde_json::to_string_pretty(&payload)?);
            return Ok(());
        }

        println!("{}", "ML Execution Devices".bold().cyan());
        println!("{}", "====================".cyan());
        println!();
        println!(
            "  {:<10} {:<8} {:<32} {:<6} {:<6} {:<6}",
            "Device".bold(),
            "Avail".bold(),
            "Name".bold(),
            "FP16".bold(),
            "BF16".bold(),
            "INT8".bold(),
        );
        for c in &caps {
            let avail = if c.is_available {
                "yes".green().to_string()
            } else {
                "no".dimmed().to_string()
            };
            let fp16 = yesno_glyph(c.supports_fp16);
            let bf16 = yesno_glyph(c.supports_bf16);
            let int8 = yesno_glyph(c.supports_int8);
            println!(
                "  {:<10} {:<8} {:<32} {:<6} {:<6} {:<6}",
                c.device_type.name(),
                avail,
                c.device_name,
                fp16,
                bf16,
                int8,
            );
        }

        println!();
        let best = DeviceType::auto();
        println!(
            "  {} {}",
            "auto →".dimmed(),
            best.display_name().bold().green()
        );

        Ok(())
    }

    pub(super) fn run(
        pipeline: &str,
        model: &Path,
        input: &Path,
        device: &str,
        top_k: Option<usize>,
        threshold: Option<f32>,
        json: bool,
        dry_run: bool,
    ) -> Result<()> {
        // 1. Validate pipeline identifier upfront.
        let spec = built_in_pipelines()
            .into_iter()
            .find(|p| {
                let head = p.id.split('/').next().unwrap_or(p.id);
                head.eq_ignore_ascii_case(pipeline) || p.id.eq_ignore_ascii_case(pipeline)
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown pipeline '{pipeline}'. \
                     Use `oximedia ml list` to see available pipelines."
                )
            })?;

        // 2. Resolve device.
        let resolved_device = if device.eq_ignore_ascii_case("auto") {
            DeviceType::auto()
        } else {
            parse_device_name(device)?
        };

        // 3. Validate threshold / top-k ranges.
        if let Some(t) = threshold {
            if !(0.0..=1.0).contains(&t) || !t.is_finite() {
                anyhow::bail!("--threshold must be a finite number in [0.0, 1.0], got {t}");
            }
        }
        if matches!(top_k, Some(0)) {
            anyhow::bail!("--top-k must be >= 1");
        }

        // 4. Validate that the model + input files exist (dry-run still
        //    requires real paths so users catch typos early).
        if !model.exists() {
            anyhow::bail!("model file not found: {}", model.display());
        }
        if !input.exists() {
            anyhow::bail!("input file not found: {}", input.display());
        }

        // 5. Build the PipelineInfo we can statically advertise.
        let info = PipelineInfo {
            id: spec.id,
            name: spec.name,
            task: spec.task,
            input_size: spec.input_size,
        };

        if dry_run {
            if json {
                let out = serde_json::json!({
                    "command": "ml run",
                    "mode": "dry-run",
                    "pipeline": info,
                    "device": resolved_device.name(),
                    "device_display": resolved_device.display_name(),
                    "model": model.display().to_string(),
                    "input": input.display().to_string(),
                    "top_k": top_k,
                    "threshold": threshold,
                });
                println!("{}", serde_json::to_string_pretty(&out)?);
                return Ok(());
            }

            println!("{}", "ML Pipeline Dry-Run".bold().cyan());
            println!("{}", "===================".cyan());
            println!();
            println!("  pipeline : {}", info.id.green());
            println!("  name     : {}", info.name);
            println!("  task     : {}", task_str(info.task));
            if let Some((w, h)) = info.input_size {
                println!("  input    : {}x{}", w, h);
            }
            println!(
                "  device   : {} ({})",
                resolved_device.display_name().green(),
                resolved_device.name()
            );
            println!("  model    : {}", model.display());
            println!("  input    : {}", input.display());
            if let Some(k) = top_k {
                println!("  top-k    : {k}");
            }
            if let Some(t) = threshold {
                println!("  threshold: {t:.3}");
            }
            println!();
            println!(
                "  {}",
                "dry-run: pipeline validated, ONNX runtime not invoked".dimmed()
            );
            return Ok(());
        }

        // Non-dry-run path: we need the ONNX runtime. The typed pipelines
        // live behind their own feature gates in oximedia-ml, and they
        // each need per-format media decoders that are well outside the
        // Wave 5 Slice A scope. Until those adapters ship (Wave 5 Slice B/C,
        // Wave 6), direct non-dry-run from the CLI to a clear error.
        let _ = (info, resolved_device);
        Err(anyhow::anyhow!(
            "non-dry-run execution for pipeline '{}' is not yet wired into the CLI. \
             Re-run with `--dry-run` to validate the pipeline + device, or use the \
             `oximedia-ml` Rust API directly.",
            spec.id
        )
        .context("feature pending: Wave 6 will deliver the full `ml run` data path"))
    }

    // -----------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------

    /// Parse a user-facing device name into a `DeviceType`.
    fn parse_device_name(name: &str) -> Result<DeviceType> {
        match name.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(DeviceType::Cpu),
            "cuda" | "nvidia" | "gpu-cuda" => Ok(DeviceType::Cuda),
            "webgpu" | "wgpu" | "gpu" => Ok(DeviceType::WebGpu),
            "directml" | "dml" => Ok(DeviceType::DirectMl),
            "coreml" | "apple" | "mps" => Ok(DeviceType::CoreMl),
            other => Err(anyhow::anyhow!(
                "unknown device '{other}'. Expected one of: cpu, cuda, webgpu, directml, coreml."
            )),
        }
    }

    /// Convert a `PipelineTask` into a stable kebab-case string used in
    /// CLI / JSON output. Matches the serde rename_all attribute on the
    /// underlying type.
    fn task_str(task: PipelineTask) -> &'static str {
        match task {
            PipelineTask::SceneClassification => "scene-classification",
            PipelineTask::ShotBoundary => "shot-boundary",
            PipelineTask::Detection => "detection",
            PipelineTask::Segmentation => "segmentation",
            PipelineTask::AestheticScoring => "aesthetic-scoring",
            PipelineTask::FaceEmbedding => "face-embedding",
            PipelineTask::Custom => "custom",
        }
    }

    /// Render a boolean as a colored Unicode glyph for terminal output.
    fn yesno_glyph(v: bool) -> String {
        if v {
            "yes".green().to_string()
        } else {
            "-".dimmed().to_string()
        }
    }
}

#[cfg(feature = "ml")]
fn cmd_list(json: bool) -> Result<()> {
    ml_impl::list(json)
}

#[cfg(feature = "ml")]
fn cmd_probe(device: Option<&str>, json: bool) -> Result<()> {
    ml_impl::probe(device, json)
}

#[cfg(feature = "ml")]
#[allow(clippy::too_many_arguments)]
fn cmd_run(
    pipeline: &str,
    model: &std::path::Path,
    input: &std::path::Path,
    device: &str,
    top_k: Option<usize>,
    threshold: Option<f32>,
    json: bool,
    dry_run: bool,
) -> Result<()> {
    ml_impl::run(
        pipeline, model, input, device, top_k, threshold, json, dry_run,
    )
}

// ---------------------------------------------------------------------------
// Unit tests (feature-agnostic validation only)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn ml_command_dispatch_exists() {
        // Sanity: the MlCommand enum is constructable. Actual behavior is
        // covered by integration tests in tests/ml_commands.rs which run
        // the real binary with and without the `ml` feature.
        let cmd = MlCommand::List { json: true };
        // Dispatching a List in a feature-off build must report a useful
        // error; in a feature-on build it succeeds. Either outcome is OK
        // for this unit test.
        let _ = run_ml(cmd, false).await;
    }

    #[cfg(feature = "ml")]
    #[tokio::test]
    async fn probe_with_invalid_device_errors() {
        let cmd = MlCommand::Probe {
            device: Some("bogus-accelerator".to_string()),
            json: false,
        };
        let err = run_ml(cmd, false).await.unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.to_ascii_lowercase().contains("unknown device"),
            "expected 'unknown device' in error, got: {msg}"
        );
    }

    #[cfg(feature = "ml")]
    #[tokio::test]
    async fn run_rejects_unknown_pipeline() {
        let cmd = MlCommand::Run {
            pipeline: "no-such-pipeline".to_string(),
            model: PathBuf::from("/nonexistent/model.onnx"),
            input: PathBuf::from("/nonexistent/input.png"),
            device: "cpu".to_string(),
            top_k: None,
            threshold: None,
            json: false,
            dry_run: true,
        };
        let err = run_ml(cmd, false).await.unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.to_ascii_lowercase().contains("unknown pipeline"),
            "expected 'unknown pipeline' in error, got: {msg}"
        );
    }

    #[cfg(feature = "ml")]
    #[tokio::test]
    async fn run_rejects_bad_threshold() {
        // Use a threshold outside [0, 1] but supply a valid pipeline so we
        // reach the threshold validation branch.
        let cmd = MlCommand::Run {
            pipeline: "scene-classifier".to_string(),
            model: PathBuf::from("/nonexistent/model.onnx"),
            input: PathBuf::from("/nonexistent/input.png"),
            device: "cpu".to_string(),
            top_k: None,
            threshold: Some(1.5),
            json: false,
            dry_run: true,
        };
        let err = run_ml(cmd, false).await.unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("--threshold"),
            "expected --threshold validation error, got: {msg}"
        );
    }

    #[cfg(not(feature = "ml"))]
    #[tokio::test]
    async fn feature_off_reports_rebuild_instructions() {
        let cmd = MlCommand::List { json: false };
        let err = run_ml(cmd, false).await.unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("--features ml"),
            "expected rebuild hint, got: {msg}"
        );
    }
}
