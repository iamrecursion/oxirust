//! End-to-end proxy workflow test: ingest -> generate proxy -> edit (simulate)
//! -> conform -> verify output.
//!
//! This drives the real production seams together:
//! - A real Matroska file on disk (`common::write_synthetic_video_mkv`).
//! - `ProxyGenerator::generate_with_settings` -> real
//!   `oximedia_transcode::TranscodePipeline` demux/remux (stream-copy mode,
//!   which is exercised here because the frame-level H.264/VP9 encode path
//!   requires real decodable bitstreams that a synthetic fixture cannot
//!   provide — see `oximedia-transcode`'s own
//!   `transcode_roundtrip.rs` module doc for the same constraint).
//! - `ProxyLinkManager` — real JSON-backed link database.
//! - A simulated "edit": a hand-authored CMX 3600 EDL referencing the proxy.
//! - `ConformEngine::conform_from_edl` — the real (currently existence-check
//!   only) conform seam; see inline comments for what it does and does not
//!   yet do.
//! - `WorkflowValidator::validate_all` — real cross-checks that both the
//!   proxy and original files exist, are non-empty, and the link metadata is
//!   self-consistent, standing in for "verify output" in the absence of a
//!   conform step that materializes a new file.
//!
//! All temporary files live under `std::env::temp_dir()`.

mod common;

use common::{unique_temp_dir, write_synthetic_video_mkv};
use oximedia_proxy::{
    ConformEngine, ProxyError, ProxyGenerationSettings, ProxyGenerator, ProxyLinkManager,
    WorkflowValidator,
};
use std::collections::HashMap;

#[tokio::test]
async fn test_e2e_ingest_generate_edit_conform_verify() {
    let dir = unique_temp_dir("e2e_workflow");

    let original_path = dir.join("camera_orig.mkv");
    let proxy_path = dir.join("camera_proxy.mkv");
    let db_path = dir.join("links.json");
    let edl_path = dir.join("edit.edl");
    let conformed_output = dir.join("final_conformed.mkv");

    // ── Step 1: "ingest" — build a real camera-original Matroska file ──────
    write_synthetic_video_mkv(&original_path, 30, 33)
        .await
        .expect("failed to write synthetic original");
    assert!(original_path.exists(), "original file must exist on disk");

    // ── Step 2: generate a real proxy via the production transcode path ────
    // `codec: "copy"` keeps the pipeline in packet-level stream-copy mode
    // (see `oximedia_transcode::pipeline::requires_frame_level`), which is
    // the only mode compatible with a synthetic (non-decodable) fixture.
    let settings = ProxyGenerationSettings {
        codec: "copy".to_string(),
        audio_codec: "copy".to_string(),
        container: "mkv".to_string(),
        ..ProxyGenerationSettings::default()
    };
    let generator = ProxyGenerator::new();
    let encode_result = generator
        .generate_with_settings(&original_path, &proxy_path, settings.clone())
        .await
        .expect("real proxy generation must succeed");

    assert!(proxy_path.exists(), "proxy file must be written to disk");
    assert!(
        encode_result.file_size > 0,
        "proxy file size must be non-zero, got {}",
        encode_result.file_size
    );

    // ── Step 3: link the freshly generated proxy to its original ────────────
    {
        let mut link_manager = ProxyLinkManager::new(&db_path)
            .await
            .expect("link database must open");
        link_manager
            .link_proxy_with_metadata(
                &proxy_path,
                &original_path,
                settings.scale_factor,
                encode_result.codec.clone(),
                encode_result.duration,
                Some("01:00:00:00".to_string()),
                HashMap::new(),
            )
            .await
            .expect("linking proxy to original must succeed");
        assert!(link_manager.has_link(&proxy_path));
    }

    // ── Step 4: "edit (simulate)" — author a CMX 3600 EDL referencing the ──
    // proxy, as an offline editor would produce after cutting on the proxy.
    let edl_text = format!(
        "TITLE: E2E Workflow Test\nFCM: NON-DROP FRAME\n\n\
         001  AX       V     C        01:00:00:00 01:00:00:20 01:00:00:00 01:00:00:20\n\
         * FROM CLIP NAME: {}\n",
        proxy_path.display()
    );
    std::fs::write(&edl_path, edl_text).expect("failed to write simulated EDL");

    // ── Step 5: conform back to the original ────────────────────────────────
    let engine = ConformEngine::new(&db_path)
        .await
        .expect("conform engine must open the same link database");
    let conform_result = engine
        .conform_from_edl(&edl_path, &conformed_output)
        .await
        .expect("conform_from_edl must succeed for an existing EDL");

    assert_eq!(conform_result.output_path, conformed_output);
    assert!(
        conform_result.frame_accurate,
        "conform result must report frame-accurate conforming"
    );
    // NOTE: `EdlConformer::conform` is currently an existence-check-only seam
    // (see `src/conform/edl.rs`) — it validates the EDL exists and returns a
    // fixed result without yet parsing per-clip relink counts. Document the
    // honest current contract rather than asserting fabricated relink counts.
    assert_eq!(conform_result.clips_relinked, 0);
    assert_eq!(conform_result.clips_failed, 0);

    // Conforming with a missing EDL must fail with FileNotFound.
    let missing_edl = dir.join("does_not_exist.edl");
    let err = engine
        .conform_from_edl(&missing_edl, &conformed_output)
        .await
        .expect_err("conforming a missing EDL must error");
    assert!(matches!(err, ProxyError::FileNotFound(_)));

    // ── Step 6: "verify output" — cross-check the link database is sound ───
    let validator = WorkflowValidator::new(engine.link_manager());
    let report = validator
        .validate_all()
        .expect("validate_all must not error");

    assert_eq!(report.total_links, 1);
    assert_eq!(report.valid_links, 1);
    assert!(
        report.errors.is_empty(),
        "expected no validation errors, got: {:?}",
        report.errors
    );

    // ── Cleanup ──────────────────────────────────────────────────────────────
    let _ = std::fs::remove_dir_all(&dir);
}
