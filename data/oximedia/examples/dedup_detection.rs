//! Media deduplication detection example.
//!
//! Demonstrates OxiMedia's duplicate detection pipeline:
//! - Configures `DuplicateDetector` with a temp-dir SQLite database
//! - Writes synthetic "media" files with intentional content duplication
//! - Indexes files using `DetectionStrategy::ExactHash`
//! - Retrieves and prints a `DuplicateReport` with groups and space-savings
//! - Cleans up all temporary files on exit
//!
//! # Usage
//!
//! ```bash
//! cargo run --example dedup_detection --features dedup -p oximedia
//! ```

use oximedia::dedup::{DedupConfig, DetectionStrategy, DuplicateDetector};
use std::env;
use std::fs;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Synthetic media content helpers
// ---------------------------------------------------------------------------

/// Minimal synthetic MKV-style payload.  Content is byte-identical between
/// `file_a` and `file_b` so that exact-hash dedup groups them together.
fn synthetic_mkv() -> Vec<u8> {
    let mut data = vec![0x1A, 0x45, 0xDF, 0xA3]; // EBML magic
    data.extend_from_slice(b"synthetic-video-payload-abcdefgh");
    data.extend(vec![0u8; 4096]);
    data
}

/// A different synthetic payload — should NOT appear in any duplicate group.
fn synthetic_ogg() -> Vec<u8> {
    let mut data = vec![b'O', b'g', b'g', b'S']; // Ogg magic
    data.extend_from_slice(b"synthetic-audio-payload-zyxwvuts");
    data.extend(vec![0xFFu8; 2048]);
    data
}

// ---------------------------------------------------------------------------
// Main (async via tokio)
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OxiMedia Duplicate Detection Example");
    println!("=====================================\n");

    // ── 1. Set up temp directory and file paths ──────────────────────────
    let tmp: PathBuf = env::temp_dir().join("oximedia_dedup_example");
    fs::create_dir_all(&tmp)?;

    let db_path = tmp.join("dedup.db");
    let file_a = tmp.join("video_master.mkv");
    let file_b = tmp.join("video_copy.mkv"); // exact duplicate of A
    let file_c = tmp.join("audio_unique.ogg"); // unique content

    // ── 2. Write synthetic files to disk ─────────────────────────────────
    let mkv_data = synthetic_mkv();
    let ogg_data = synthetic_ogg();

    fs::write(&file_a, &mkv_data)?;
    fs::write(&file_b, &mkv_data)?;
    fs::write(&file_c, &ogg_data)?;

    println!("Synthetic media files written:");
    println!("  {} ({} bytes)", file_a.display(), mkv_data.len());
    println!(
        "  {} ({} bytes)  [exact copy]",
        file_b.display(),
        mkv_data.len()
    );
    println!(
        "  {} ({} bytes)  [unique]",
        file_c.display(),
        ogg_data.len()
    );
    println!();

    // ── 3. Configure DuplicateDetector ───────────────────────────────────
    let config = DedupConfig {
        database_path: db_path.clone(),
        perceptual_threshold: 0.95,
        ssim_threshold: 0.90,
        histogram_threshold: 0.85,
        feature_match_threshold: 50,
        audio_threshold: 0.90,
        metadata_threshold: 0.80,
        parallel: false,
        sample_frames: 4,
        chunk_size: 4096,
        ..DedupConfig::default()
    };

    println!(
        "Initialising DuplicateDetector (db: {})...",
        db_path.display()
    );
    let mut detector = DuplicateDetector::new(config).await?;

    // ── 4. Index files ────────────────────────────────────────────────────
    println!("Indexing files...");
    detector.add_file(&file_a).await?;
    detector.add_file(&file_b).await?;
    detector.add_file(&file_c).await?;

    // ── 5. Run duplicate detection ────────────────────────────────────────
    println!("Running duplicate detection (ExactHash strategy)...");
    let report = detector
        .find_duplicates(DetectionStrategy::ExactHash)
        .await?;

    // ── 6. Print DuplicateReport ──────────────────────────────────────────
    println!();
    println!("Duplicate Report");
    println!("----------------");
    println!("  Duplicate groups  : {}", report.groups.len());
    println!("  Duplicate files   : {}", report.total_duplicates);
    println!("  Reclaimable space : {} bytes", report.wasted_space);
    println!();

    if report.groups.is_empty() {
        println!("  No duplicates found.");
    } else {
        for (idx, group) in report.groups.iter().enumerate() {
            println!("  Group {} — {} files", idx + 1, group.files.len());
            for file in &group.files {
                println!("    - {file}");
            }
            for score in &group.scores {
                println!(
                    "    similarity [{method}]: {pct:.1}%",
                    method = score.method,
                    pct = score.score * 100.0,
                );
            }

            if let Some(rec) = group.recommend_action() {
                println!(
                    "    recommendation: keep '{}', remove {} file(s)",
                    rec.keep_file,
                    rec.delete_files.len(),
                );
            }
            println!();
        }
    }

    // ── 7. Cleanup ────────────────────────────────────────────────────────
    fs::remove_dir_all(&tmp).unwrap_or_default();
    println!("Temp directory cleaned up: {}", tmp.display());
    println!("\nExample completed successfully.");

    Ok(())
}
