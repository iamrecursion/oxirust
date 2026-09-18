//! DAG workflow orchestration example.
//!
//! Demonstrates OxiMedia's workflow engine:
//! - Creates a `WorkflowEngine` with a temp-dir SQLite persistence path
//! - Builds a linear media-processing pipeline using `WorkflowBuilder`:
//!     1. `ingest`          — initial file analysis
//!     2. `transcode`       — AV1 encode (depends on ingest)
//!     3. `quality_check`   — QC validation (depends on transcode)
//!     4. `deliver`         — S3 upload (depends on QC)
//! - Submits the workflow to the engine
//! - Verifies persistence by reloading from the SQLite store
//! - Prints tasks, dependencies, and topological execution order
//! - Cleans up the temp SQLite file
//!
//! # Usage
//!
//! ```bash
//! cargo run --example workflow_pipeline --features workflow -p oximedia
//! ```

use oximedia::workflow::{
    AnalysisType, TaskBuilder, TaskPriority, TaskType, TransferProtocol, Workflow, WorkflowBuilder,
    WorkflowEngine,
};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Build the four-stage media pipeline
// ---------------------------------------------------------------------------

fn build_media_pipeline() -> Result<Workflow, Box<dyn std::error::Error>> {
    // Stage 1 — ingest: analyse the source file
    let ingest = TaskBuilder::new(
        "ingest",
        TaskType::Analysis {
            input: PathBuf::from("/mnt/nfs/ingest/source.mxf"),
            analyses: vec![AnalysisType::AudioLevels, AnalysisType::VideoQuality],
            output: Some(PathBuf::from("/tmp/qc/ingest_report.json")),
        },
    )
    .named("ingest")
    .priority(TaskPriority::High)
    .timeout(Duration::from_secs(300))
    .metadata("stage", "1")
    .metadata(
        "description",
        "Analyse source MXF and generate ingest report",
    );

    // Stage 2 — transcode: encode to AV1 proxy
    let transcode = TaskBuilder::new(
        "transcode",
        TaskType::Transcode {
            input: PathBuf::from("/mnt/nfs/ingest/source.mxf"),
            output: PathBuf::from("/mnt/nfs/proxy/output_av1.mkv"),
            preset: "av1_broadcast".to_string(),
            params: {
                let mut p = HashMap::new();
                p.insert("crf".to_string(), serde_json::json!(28));
                p.insert("speed".to_string(), serde_json::json!(4));
                p
            },
        },
    )
    .named("transcode")
    .priority(TaskPriority::Normal)
    .timeout(Duration::from_secs(7200))
    .metadata("stage", "2")
    .metadata("description", "Encode to AV1 broadcast proxy");

    // Stage 3 — quality_check: validate the encoded output
    let quality_check = TaskBuilder::new(
        "quality_check",
        TaskType::QualityControl {
            input: PathBuf::from("/mnt/nfs/proxy/output_av1.mkv"),
            profile: "broadcast_hd".to_string(),
            rules: vec![
                "loudness_within_r128".to_string(),
                "no_black_frames".to_string(),
                "aspect_ratio_16_9".to_string(),
            ],
        },
    )
    .named("quality_check")
    .priority(TaskPriority::High)
    .timeout(Duration::from_secs(600))
    .metadata("stage", "3")
    .metadata(
        "description",
        "Validate encoded output against broadcast spec",
    );

    // Stage 4 — deliver: transfer approved file to S3
    let deliver = TaskBuilder::new(
        "deliver",
        TaskType::Transfer {
            source: "/mnt/nfs/proxy/output_av1.mkv".to_string(),
            destination: "s3://media-delivery/2026/output_av1.mkv".to_string(),
            protocol: TransferProtocol::S3,
            options: {
                let mut o = HashMap::new();
                o.insert("storage_class".to_string(), "STANDARD".to_string());
                o.insert("multipart_threshold_mb".to_string(), "100".to_string());
                o
            },
        },
    )
    .named("deliver")
    .priority(TaskPriority::Normal)
    .timeout(Duration::from_secs(1800))
    .metadata("stage", "4")
    .metadata("description", "Upload approved proxy to S3 delivery bucket");

    // Linear chain: ingest → transcode → quality_check → deliver
    let workflow = WorkflowBuilder::new("media-pipeline-example")
        .description("Four-stage broadcast media processing pipeline")
        .max_concurrent_tasks(2)
        .fail_fast(true)
        .add_task(ingest)
        .add_task(transcode)
        .add_task(quality_check)
        .add_task(deliver)
        .depends_on("transcode", "ingest")?
        .depends_on("quality_check", "transcode")?
        .depends_on("deliver", "quality_check")?
        .build()?;

    Ok(workflow)
}

// ---------------------------------------------------------------------------
// Print helpers
// ---------------------------------------------------------------------------

fn print_workflow_structure(workflow: &Workflow) {
    println!("Workflow: \"{}\"", workflow.name);
    println!("  ID          : {}", workflow.id);
    println!("  Description : {}", workflow.description);
    println!("  State       : {:?}", workflow.state);
    println!("  Max parallel: {}", workflow.config.max_concurrent_tasks);
    println!("  Fail-fast   : {}", workflow.config.fail_fast);
    println!();

    println!("Tasks ({}):", workflow.tasks.len());
    let mut task_list: Vec<_> = workflow.tasks().collect();
    task_list.sort_by_key(|t| t.metadata.get("stage").cloned().unwrap_or_default());

    for task in &task_list {
        let stage = task
            .metadata
            .get("stage")
            .map(String::as_str)
            .unwrap_or("?");
        let desc = task
            .metadata
            .get("description")
            .map(String::as_str)
            .unwrap_or("");
        println!(
            "  [{stage}] {name:20}  priority={priority:?}  timeout={secs}s",
            name = task.name,
            priority = task.priority,
            secs = task.timeout.as_secs(),
        );
        if !desc.is_empty() {
            println!("       desc: {desc}");
        }
        let deps = workflow.get_dependencies(&task.id);
        if !deps.is_empty() {
            let dep_names: Vec<&str> = deps
                .iter()
                .filter_map(|id| workflow.get_task(id))
                .map(|t| t.name.as_str())
                .collect();
            println!("       deps: {}", dep_names.join(", "));
        }
    }
    println!();

    // Topological execution order
    match workflow.topological_sort() {
        Ok(order) => {
            let names: Vec<&str> = order
                .iter()
                .filter_map(|id| workflow.get_task(id))
                .map(|t| t.name.as_str())
                .collect();
            println!("Execution order (topological):");
            for (i, name) in names.iter().enumerate() {
                println!("  {}. {name}", i + 1);
            }
        }
        Err(e) => println!("Could not compute topological order: {e}"),
    }
    println!();
}

// ---------------------------------------------------------------------------
// Main (async via tokio)
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OxiMedia Workflow Pipeline Example");
    println!("====================================\n");

    // ── 1. Temporary SQLite path ──────────────────────────────────────────
    let tmp: PathBuf = env::temp_dir().join("oximedia_workflow_example");
    fs::create_dir_all(&tmp)?;
    let db_path = tmp.join("workflows.db");

    println!("Persistence DB: {}", db_path.display());
    println!();

    // ── 2. Build the workflow ─────────────────────────────────────────────
    println!("Building four-stage media pipeline...");
    let workflow = build_media_pipeline()?;

    // ── 3. Print structure before submission ─────────────────────────────
    print_workflow_structure(&workflow);

    // ── 4. Create engine and submit the workflow ──────────────────────────
    println!("Creating WorkflowEngine...");
    let engine = WorkflowEngine::new(&db_path)?;

    let workflow_id = engine.submit_workflow(&workflow).await?;
    println!("Workflow submitted successfully.");
    println!("  Workflow ID : {workflow_id}");
    println!();

    // ── 5. Verify persistence by reloading ───────────────────────────────
    let loaded = engine.persistence().load_workflow(workflow_id)?;
    println!(
        "Verified persistence — reloaded workflow: \"{}\"",
        loaded.name
    );
    println!("  Tasks persisted: {}", loaded.tasks.len());
    println!("  Edges persisted: {}", loaded.edges.len());
    println!();

    // ── 6. Cleanup ────────────────────────────────────────────────────────
    fs::remove_dir_all(&tmp).unwrap_or_default();
    println!("Temp directory cleaned up: {}", tmp.display());
    println!("\nExample completed successfully.");

    Ok(())
}
