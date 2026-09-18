//! Virtual production commands: create, list, start, stop, configure.
//!
//! Exposes `oximedia-virtual` LED wall, camera tracking, compositing,
//! and genlock synchronization via the CLI.
//!
//! # Session registry
//!
//! Each CLI invocation is a fresh process, so `oximedia-virtual`'s in-memory
//! `VirtualProduction` handle cannot survive between e.g. `virtual create`
//! and a later `virtual start`. To make `create`/`list`/`start`/`stop`/
//! `configure` behave honestly across separate invocations, this module
//! persists a small session registry (name, workflow, fps, ...) as JSON in
//! the platform state directory (see [`default_registry_path`]). `create`
//! only succeeds if the session didn't already exist; `start`/`stop`/
//! `configure` only succeed for sessions that are actually present in the
//! registry, and error out (never fabricate success) otherwise.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::{Path, PathBuf};

/// Virtual production subcommands.
#[derive(Subcommand, Debug)]
pub enum VirtualCommand {
    /// Create a new virtual production session
    Create {
        /// Session name
        #[arg(short, long)]
        name: String,

        /// Workflow type: led-wall, hybrid, green-screen, ar
        #[arg(long, default_value = "led-wall")]
        workflow: String,

        /// Target frames per second
        #[arg(long, default_value = "60")]
        fps: f64,

        /// Number of tracked cameras
        #[arg(long, default_value = "1")]
        cameras: usize,

        /// Quality mode: draft, preview, final
        #[arg(long, default_value = "preview")]
        quality: String,

        /// Synchronization accuracy target in ms
        #[arg(long, default_value = "0.5")]
        sync_ms: f64,
    },

    /// List active virtual production sessions
    List {
        /// Show detailed per-session info
        #[arg(long)]
        detailed: bool,
    },

    /// Start a virtual production session
    Start {
        /// Session name
        #[arg(short, long)]
        name: String,

        /// Enable motion capture integration
        #[arg(long)]
        mocap: bool,

        /// Enable Unreal Engine integration
        #[arg(long)]
        unreal: bool,

        /// Enable lens distortion correction
        #[arg(long)]
        lens_correction: bool,
    },

    /// Stop a virtual production session
    Stop {
        /// Session name
        #[arg(short, long)]
        name: String,

        /// Force stop (skip graceful shutdown)
        #[arg(long)]
        force: bool,
    },

    /// Configure a virtual production session
    Configure {
        /// Session name
        #[arg(short, long)]
        name: String,

        /// Set workflow type
        #[arg(long)]
        workflow: Option<String>,

        /// Set target FPS
        #[arg(long)]
        fps: Option<f64>,

        /// Set quality mode
        #[arg(long)]
        quality: Option<String>,

        /// Enable/disable color calibration
        #[arg(long)]
        color_calibration: Option<bool>,

        /// Set number of cameras
        #[arg(long)]
        cameras: Option<usize>,
    },
}

/// Handle virtual production command dispatch.
pub async fn handle_virtual_command(command: VirtualCommand, json_output: bool) -> Result<()> {
    // Resolve the persisted-session-registry path once. `default_registry_path`
    // always succeeds in practice (its final fallback is `std::env::temp_dir()`,
    // which is infallible), but we still surface a real error instead of
    // silently discarding session state if a future platform ever returns
    // `None` all the way down.
    let registry_path = default_registry_path().ok_or_else(|| {
        anyhow::anyhow!(
            "Could not determine a writable state directory to persist virtual production sessions"
        )
    })?;

    match command {
        VirtualCommand::Create {
            name,
            workflow,
            fps,
            cameras,
            quality,
            sync_ms,
        } => {
            handle_create(
                &name,
                &workflow,
                fps,
                cameras,
                &quality,
                sync_ms,
                json_output,
                &registry_path,
            )
            .await
        }
        VirtualCommand::List { detailed } => {
            handle_list(detailed, json_output, &registry_path).await
        }
        VirtualCommand::Start {
            name,
            mocap,
            unreal,
            lens_correction,
        } => {
            handle_start(
                &name,
                mocap,
                unreal,
                lens_correction,
                json_output,
                &registry_path,
            )
            .await
        }
        VirtualCommand::Stop { name, force } => {
            handle_stop(&name, force, json_output, &registry_path).await
        }
        VirtualCommand::Configure {
            name,
            workflow,
            fps,
            quality,
            color_calibration,
            cameras,
        } => {
            handle_configure(
                &name,
                workflow.as_deref(),
                fps,
                quality.as_deref(),
                color_calibration,
                cameras,
                json_output,
                &registry_path,
            )
            .await
        }
    }
}

/// Parse workflow type string.
fn parse_workflow(s: &str) -> Result<oximedia_virtual::WorkflowType> {
    match s {
        "led-wall" | "ledwall" | "led" => Ok(oximedia_virtual::WorkflowType::LedWall),
        "hybrid" => Ok(oximedia_virtual::WorkflowType::Hybrid),
        "green-screen" | "greenscreen" | "gs" => Ok(oximedia_virtual::WorkflowType::GreenScreen),
        "ar" | "augmented-reality" => Ok(oximedia_virtual::WorkflowType::AugmentedReality),
        other => Err(anyhow::anyhow!(
            "Unknown workflow '{}'. Supported: led-wall, hybrid, green-screen, ar",
            other
        )),
    }
}

/// Parse quality mode string.
fn parse_quality(s: &str) -> Result<oximedia_virtual::QualityMode> {
    match s {
        "draft" => Ok(oximedia_virtual::QualityMode::Draft),
        "preview" => Ok(oximedia_virtual::QualityMode::Preview),
        "final" => Ok(oximedia_virtual::QualityMode::Final),
        other => Err(anyhow::anyhow!(
            "Unknown quality mode '{}'. Supported: draft, preview, final",
            other
        )),
    }
}

// ---------------------------------------------------------------------------
// Session registry (persisted across CLI invocations)
// ---------------------------------------------------------------------------

/// A persisted virtual-production session record.
///
/// `workflow` and `quality` are stored in their canonical `Debug`-formatted
/// form (e.g. `"LedWall"`) so the value is stable regardless of which alias
/// (`led-wall`, `ledwall`, `led`, ...) the user typed.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct SessionRecord {
    name: String,
    workflow: String,
    fps: f64,
    cameras: usize,
    quality: String,
    sync_accuracy_ms: f64,
    status: String,
    motion_capture: bool,
    unreal_integration: bool,
    lens_correction: bool,
    color_calibration: Option<bool>,
    created_at_unix_secs: u64,
}

/// The full on-disk session registry.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
struct SessionRegistry {
    sessions: Vec<SessionRecord>,
}

/// Default location for the persisted session registry: the platform state
/// directory (falling back to the local-data dir, then the OS temp dir --
/// the same fallback chain `tui_cmd.rs` uses for its own persisted state).
fn default_registry_path() -> Option<PathBuf> {
    let base = dirs::state_dir()
        .or_else(dirs::data_local_dir)
        .or_else(|| Some(std::env::temp_dir()))?;
    Some(base.join("oximedia").join("virtual_sessions.json"))
}

/// Load the session registry from `path`. A missing file is treated as an
/// empty registry (first run), not an error.
fn load_registry(path: &Path) -> Result<SessionRegistry> {
    if !path.exists() {
        return Ok(SessionRegistry::default());
    }
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read session registry: {}", path.display()))?;
    serde_json::from_str(&data)
        .with_context(|| format!("Failed to parse session registry: {}", path.display()))
}

/// Persist the session registry to `path`, creating the parent directory if
/// needed.
fn save_registry(path: &Path, registry: &SessionRegistry) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!(
                    "Failed to create session registry directory: {}",
                    parent.display()
                )
            })?;
        }
    }
    let data =
        serde_json::to_string_pretty(registry).context("Failed to serialize session registry")?;
    std::fs::write(path, data)
        .with_context(|| format!("Failed to write session registry: {}", path.display()))?;
    Ok(())
}

/// Current Unix time in seconds (`0` if the clock is somehow before the
/// epoch -- never fails, so callers don't need to handle an error for a
/// purely informational timestamp).
fn now_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Handler: Create
// ---------------------------------------------------------------------------

/// Create a new virtual production session.
#[allow(clippy::too_many_arguments)]
async fn handle_create(
    name: &str,
    workflow: &str,
    fps: f64,
    cameras: usize,
    quality: &str,
    sync_ms: f64,
    json_output: bool,
    registry_path: &Path,
) -> Result<()> {
    if name.trim().is_empty() {
        return Err(anyhow::anyhow!("Session name must not be empty"));
    }

    let wf = parse_workflow(workflow)?;
    let qm = parse_quality(quality)?;

    if fps <= 0.0 || fps > 240.0 {
        return Err(anyhow::anyhow!(
            "FPS must be between 0 and 240, got {}",
            fps
        ));
    }
    if cameras == 0 || cameras > 64 {
        return Err(anyhow::anyhow!(
            "Camera count must be between 1 and 64, got {}",
            cameras
        ));
    }

    let config = oximedia_virtual::VirtualProductionConfig::default()
        .with_workflow(wf)
        .with_target_fps(fps)
        .with_quality(qm)
        .with_sync_accuracy_ms(sync_ms)
        .with_num_cameras(cameras);

    let _vp = oximedia_virtual::VirtualProduction::new(config)
        .map_err(|e| anyhow::anyhow!("Failed to create virtual production session: {}", e))?;

    let mut registry = load_registry(registry_path)?;
    if registry.sessions.iter().any(|s| s.name == name) {
        return Err(anyhow::anyhow!(
            "A virtual production session named '{name}' already exists. \
             Use 'oximedia virtual stop --name {name}' first, or choose a different name."
        ));
    }

    registry.sessions.push(SessionRecord {
        name: name.to_string(),
        workflow: format!("{wf:?}"),
        fps,
        cameras,
        quality: format!("{qm:?}"),
        sync_accuracy_ms: sync_ms,
        status: "created".to_string(),
        motion_capture: false,
        unreal_integration: false,
        lens_correction: false,
        color_calibration: None,
        created_at_unix_secs: now_unix_secs(),
    });
    save_registry(registry_path, &registry)?;

    if json_output {
        let result = serde_json::json!({
            "command": "create",
            "name": name,
            "workflow": format!("{:?}", wf),
            "fps": fps,
            "cameras": cameras,
            "quality": format!("{:?}", qm),
            "sync_accuracy_ms": sync_ms,
            "status": "created",
        });
        let json_str = serde_json::to_string_pretty(&result)
            .context("Failed to serialize virtual production config")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Virtual Production Session Created".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Session:", name);
        println!("{:20} {:?}", "Workflow:", wf);
        println!("{:20} {} fps", "Target FPS:", fps);
        println!("{:20} {}", "Cameras:", cameras);
        println!("{:20} {:?}", "Quality:", qm);
        println!("{:20} {} ms", "Sync accuracy:", sync_ms);
        println!();
        println!(
            "{}",
            "Session registered. Use 'oximedia virtual start' to begin.".dimmed()
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: List
// ---------------------------------------------------------------------------

/// List active virtual production sessions.
async fn handle_list(detailed: bool, json_output: bool, registry_path: &Path) -> Result<()> {
    let registry = load_registry(registry_path)?;

    if json_output {
        let sessions_json: Vec<serde_json::Value> = registry
            .sessions
            .iter()
            .map(|s| {
                serde_json::json!({
                    "name": s.name,
                    "workflow": s.workflow,
                    "fps": s.fps,
                    "cameras": s.cameras,
                    "quality": s.quality,
                    "sync_accuracy_ms": s.sync_accuracy_ms,
                    "status": s.status,
                    "motion_capture": s.motion_capture,
                    "unreal_integration": s.unreal_integration,
                    "lens_correction": s.lens_correction,
                    "color_calibration": s.color_calibration,
                    "created_at_unix_secs": s.created_at_unix_secs,
                })
            })
            .collect();
        let result = serde_json::json!({
            "command": "list",
            "sessions": sessions_json,
            "supported_workflows": ["led-wall", "hybrid", "green-screen", "ar"],
            "supported_qualities": ["draft", "preview", "final"],
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize session list")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Virtual Production Sessions".green().bold());
        println!("{}", "=".repeat(60));
        if registry.sessions.is_empty() {
            println!("  No active sessions.");
        } else {
            for s in &registry.sessions {
                if detailed {
                    println!("  {}", s.name.cyan().bold());
                    println!("    {:18} {}", "Workflow:", s.workflow);
                    println!("    {:18} {}", "Status:", s.status);
                    println!("    {:18} {} fps", "Target FPS:", s.fps);
                    println!("    {:18} {}", "Cameras:", s.cameras);
                    println!("    {:18} {}", "Quality:", s.quality);
                    println!("    {:18} {} ms", "Sync accuracy:", s.sync_accuracy_ms);
                    println!("    {:18} {}", "Motion capture:", s.motion_capture);
                    println!("    {:18} {}", "Unreal Engine:", s.unreal_integration);
                    println!("    {:18} {}", "Lens correction:", s.lens_correction);
                } else {
                    println!("  {:20} {:12} [{}]", s.name.cyan(), s.workflow, s.status);
                }
            }
        }
        println!();
        if detailed {
            println!("{}", "Supported Workflows".cyan().bold());
            println!("{}", "-".repeat(60));
            println!("  led-wall       Full LED volume with camera tracking");
            println!("  hybrid         Mix LED wall and green screen");
            println!("  green-screen   Traditional green screen + real-time compositing");
            println!("  ar             Augmented reality overlay");
            println!();
            println!("{}", "Quality Modes".cyan().bold());
            println!("{}", "-".repeat(60));
            println!("  draft          Setup and rehearsal quality");
            println!("  preview        Monitoring quality");
            println!("  final          Recording quality");
            println!();
        }
        println!(
            "{}",
            "Use 'oximedia virtual create' to create a new session.".dimmed()
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: Start
// ---------------------------------------------------------------------------

/// Start a virtual production session.
async fn handle_start(
    name: &str,
    mocap: bool,
    unreal: bool,
    lens_correction: bool,
    json_output: bool,
    registry_path: &Path,
) -> Result<()> {
    let mut registry = load_registry(registry_path)?;
    let session = registry
        .sessions
        .iter_mut()
        .find(|s| s.name == name)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "No virtual production session named '{name}'. \
                 Use 'oximedia virtual create --name {name}' first."
            )
        })?;

    session.status = "running".to_string();
    session.motion_capture = mocap;
    session.unreal_integration = unreal;
    session.lens_correction = lens_correction;
    save_registry(registry_path, &registry)?;

    if json_output {
        let result = serde_json::json!({
            "command": "start",
            "name": name,
            "motion_capture": mocap,
            "unreal_integration": unreal,
            "lens_correction": lens_correction,
            "status": "started",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize start status")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Starting Virtual Production".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Session:", name);
        println!(
            "{:20} {}",
            "Motion capture:",
            if mocap { "enabled" } else { "disabled" }
        );
        println!(
            "{:20} {}",
            "Unreal Engine:",
            if unreal { "enabled" } else { "disabled" }
        );
        println!(
            "{:20} {}",
            "Lens correction:",
            if lens_correction {
                "enabled"
            } else {
                "disabled"
            }
        );
        println!();
        println!("{}", "Session is now running.".green());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: Stop
// ---------------------------------------------------------------------------

/// Stop a virtual production session.
async fn handle_stop(
    name: &str,
    force: bool,
    json_output: bool,
    registry_path: &Path,
) -> Result<()> {
    let mut registry = load_registry(registry_path)?;
    let idx = registry
        .sessions
        .iter()
        .position(|s| s.name == name)
        .ok_or_else(|| {
            anyhow::anyhow!("No virtual production session named '{name}' is active.")
        })?;
    registry.sessions.remove(idx);
    save_registry(registry_path, &registry)?;

    if json_output {
        let result = serde_json::json!({
            "command": "stop",
            "name": name,
            "force": force,
            "status": "stopped",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize stop status")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Stopping Virtual Production".yellow().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Session:", name);
        println!("{:20} {}", "Force:", force);
        println!();
        if force {
            println!("{}", "Session force-stopped.".yellow());
        } else {
            println!("{}", "Session gracefully stopped.".green());
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: Configure
// ---------------------------------------------------------------------------

/// Configure a virtual production session.
#[allow(clippy::too_many_arguments)]
async fn handle_configure(
    name: &str,
    workflow: Option<&str>,
    fps: Option<f64>,
    quality: Option<&str>,
    color_calibration: Option<bool>,
    cameras: Option<usize>,
    json_output: bool,
    registry_path: &Path,
) -> Result<()> {
    // Validate parameters if provided.
    let parsed_workflow = workflow.map(parse_workflow).transpose()?;
    let parsed_quality = quality.map(parse_quality).transpose()?;
    if let Some(f) = fps {
        if f <= 0.0 || f > 240.0 {
            return Err(anyhow::anyhow!("FPS must be between 0 and 240, got {}", f));
        }
    }
    if let Some(c) = cameras {
        if c == 0 || c > 64 {
            return Err(anyhow::anyhow!(
                "Camera count must be between 1 and 64, got {}",
                c
            ));
        }
    }

    let mut registry = load_registry(registry_path)?;
    let session = registry
        .sessions
        .iter_mut()
        .find(|s| s.name == name)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "No virtual production session named '{name}'. \
                 Use 'oximedia virtual create --name {name}' first."
            )
        })?;

    if let Some(wf) = parsed_workflow {
        session.workflow = format!("{wf:?}");
    }
    if let Some(f) = fps {
        session.fps = f;
    }
    if let Some(qm) = parsed_quality {
        session.quality = format!("{qm:?}");
    }
    if let Some(cc) = color_calibration {
        session.color_calibration = Some(cc);
    }
    if let Some(c) = cameras {
        session.cameras = c;
    }
    save_registry(registry_path, &registry)?;

    if json_output {
        let result = serde_json::json!({
            "command": "configure",
            "name": name,
            "workflow": workflow,
            "fps": fps,
            "quality": quality,
            "color_calibration": color_calibration,
            "cameras": cameras,
            "status": "configured",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize config")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Configure Virtual Production".cyan().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Session:", name);
        if let Some(w) = workflow {
            println!("{:20} {}", "Workflow:", w);
        }
        if let Some(f) = fps {
            println!("{:20} {} fps", "Target FPS:", f);
        }
        if let Some(q) = quality {
            println!("{:20} {}", "Quality:", q);
        }
        if let Some(cc) = color_calibration {
            println!("{:20} {}", "Color calibration:", cc);
        }
        if let Some(c) = cameras {
            println!("{:20} {}", "Cameras:", c);
        }
        println!();
        println!("{}", "Configuration updated.".green());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[test]
    fn test_parse_workflow_variants() {
        assert!(parse_workflow("led-wall").is_ok());
        assert!(parse_workflow("hybrid").is_ok());
        assert!(parse_workflow("green-screen").is_ok());
        assert!(parse_workflow("ar").is_ok());
        assert!(parse_workflow("invalid").is_err());
    }

    #[test]
    fn test_parse_quality_variants() {
        assert!(parse_quality("draft").is_ok());
        assert!(parse_quality("preview").is_ok());
        assert!(parse_quality("final").is_ok());
        assert!(parse_quality("bad").is_err());
    }

    #[test]
    fn test_parse_workflow_aliases() {
        assert!(parse_workflow("ledwall").is_ok());
        assert!(parse_workflow("gs").is_ok());
        assert!(parse_workflow("augmented-reality").is_ok());
    }

    #[test]
    fn test_parse_quality_matches_enum() {
        let draft = parse_quality("draft").expect("should succeed");
        assert_eq!(draft, oximedia_virtual::QualityMode::Draft);
        let final_q = parse_quality("final").expect("should succeed");
        assert_eq!(final_q, oximedia_virtual::QualityMode::Final);
    }

    // ── Session registry tests ──────────────────────────────────────────────
    //
    // Each test gets its own registry file under `std::env::temp_dir()` (a
    // unique path per test via an atomic counter + the test's own TypeId-free
    // name is not available, so we combine PID-independent process time with
    // a counter) so tests can run concurrently without sharing state.

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_registry_path(tag: &str) -> PathBuf {
        let n = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        std::env::temp_dir().join(format!("oximedia_cli_virtual_test_{tag}_{pid}_{n}.json"))
    }

    #[tokio::test]
    async fn test_create_then_list_shows_real_session() {
        let path = temp_registry_path("create_list");

        handle_create(
            "studio-a", "led-wall", 60.0, 2, "preview", 0.5, false, &path,
        )
        .await
        .expect("create should succeed");

        let registry = load_registry(&path).expect("registry should load");
        assert_eq!(registry.sessions.len(), 1);
        assert_eq!(registry.sessions[0].name, "studio-a");
        assert_eq!(registry.sessions[0].status, "created");
        assert_eq!(registry.sessions[0].cameras, 2);

        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_create_duplicate_name_errors() {
        let path = temp_registry_path("dup");

        handle_create(
            "dup-session",
            "led-wall",
            60.0,
            1,
            "preview",
            0.5,
            false,
            &path,
        )
        .await
        .expect("first create should succeed");

        let err = handle_create(
            "dup-session",
            "led-wall",
            60.0,
            1,
            "preview",
            0.5,
            false,
            &path,
        )
        .await
        .expect_err("second create with the same name must fail");
        assert!(
            err.to_string().contains("already exists"),
            "error should mention the duplicate, got: {err}"
        );

        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_start_unknown_session_errors_and_stays_absent() {
        let path = temp_registry_path("start_unknown");

        let err = handle_start("ghost", false, false, false, false, &path)
            .await
            .expect_err("starting a session that was never created must fail");
        assert!(
            err.to_string().contains("No virtual production session"),
            "error should explain the session doesn't exist, got: {err}"
        );

        // No fabricated file/registry should have been created for a pure
        // read-then-fail path over a nonexistent registry.
        let registry = load_registry(&path).expect("missing file loads as empty registry");
        assert!(registry.sessions.is_empty());

        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_create_then_start_updates_status() {
        let path = temp_registry_path("start_ok");

        handle_create("live-set", "hybrid", 59.94, 3, "final", 0.2, false, &path)
            .await
            .expect("create should succeed");

        handle_start("live-set", true, true, false, false, &path)
            .await
            .expect("start should succeed for a real session");

        let registry = load_registry(&path).expect("registry should load");
        let session = registry
            .sessions
            .iter()
            .find(|s| s.name == "live-set")
            .expect("session should still be present");
        assert_eq!(session.status, "running");
        assert!(session.motion_capture);
        assert!(session.unreal_integration);
        assert!(!session.lens_correction);

        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_stop_unknown_session_errors() {
        let path = temp_registry_path("stop_unknown");

        let err = handle_stop("ghost", false, false, &path)
            .await
            .expect_err("stopping a session that was never created must fail");
        assert!(err.to_string().contains("No virtual production session"));

        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_create_start_then_stop_removes_from_list() {
        let path = temp_registry_path("stop_ok");

        handle_create("temp-set", "ar", 30.0, 1, "draft", 1.0, false, &path)
            .await
            .expect("create should succeed");
        handle_start("temp-set", false, false, false, false, &path)
            .await
            .expect("start should succeed");

        let before = load_registry(&path).expect("registry should load");
        assert_eq!(before.sessions.len(), 1);

        handle_stop("temp-set", false, false, &path)
            .await
            .expect("stop should succeed for a real, running session");

        let after = load_registry(&path).expect("registry should load");
        assert!(
            after.sessions.is_empty(),
            "stopped session must no longer be listed"
        );

        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_configure_unknown_session_errors() {
        let path = temp_registry_path("configure_unknown");

        let err = handle_configure(
            "ghost",
            Some("hybrid"),
            None,
            None,
            None,
            None,
            false,
            &path,
        )
        .await
        .expect_err("configuring a session that was never created must fail");
        assert!(err.to_string().contains("No virtual production session"));

        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_configure_updates_real_session() {
        let path = temp_registry_path("configure_ok");

        handle_create("cfg-set", "led-wall", 60.0, 1, "preview", 0.5, false, &path)
            .await
            .expect("create should succeed");

        handle_configure(
            "cfg-set",
            Some("hybrid"),
            Some(120.0),
            Some("final"),
            Some(true),
            Some(4),
            false,
            &path,
        )
        .await
        .expect("configure should succeed for a real session");

        let registry = load_registry(&path).expect("registry should load");
        let session = registry
            .sessions
            .iter()
            .find(|s| s.name == "cfg-set")
            .expect("session should still be present");
        assert_eq!(session.workflow, "Hybrid");
        assert_eq!(session.fps, 120.0);
        assert_eq!(session.quality, "Final");
        assert_eq!(session.color_calibration, Some(true));
        assert_eq!(session.cameras, 4);

        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_list_empty_registry_reports_no_sessions() {
        let path = temp_registry_path("list_empty");
        let registry = load_registry(&path).expect("missing file loads as empty registry");
        assert!(registry.sessions.is_empty());
        // Exercise the real handler too (stdout only; nothing to assert on
        // besides "it doesn't error").
        handle_list(false, false, &path)
            .await
            .expect("listing an empty registry should not error");
    }

    #[tokio::test]
    async fn test_create_invalid_fps_does_not_persist() {
        let path = temp_registry_path("invalid_fps");
        let err = handle_create(
            "bad-fps", "led-wall", 999.0, 1, "preview", 0.5, false, &path,
        )
        .await
        .expect_err("fps out of range must fail");
        assert!(err.to_string().contains("FPS"));
        assert!(
            !path.exists(),
            "no registry file should be created for a validation failure"
        );
    }
}
