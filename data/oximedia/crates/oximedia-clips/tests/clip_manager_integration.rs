//! Integration tests for ClipManager full lifecycle, EDL export, clip_compare,
//! concurrent access, and bin_organizer auto-assignment.
//!
//! These tests use `:memory:` SQLite so no temp files are required for most
//! tests. The concurrent-access test uses a temp-file DB because SQLite
//! in-memory databases are not shared across connections.

#[cfg(not(target_arch = "wasm32"))]
mod integration {
    use oximedia_clips::bin_organizer::{
        BinOrganizer, BinRule, ClipDescriptor, ClipMediaType, OrganizeCriteria,
    };
    use oximedia_clips::clip::Clip;
    use oximedia_clips::clip_compare::{ClipComparer, ComparableClip};
    use oximedia_clips::clip_fingerprint::{ClipFingerprint, FingerprintDb, FrameFingerprint};
    use oximedia_clips::export::EdlExporter;
    use oximedia_clips::{ClipManager, Rating};
    use oximedia_core::types::Rational;
    use std::path::PathBuf;
    use std::sync::Arc;

    // ─── helper ──────────────────────────────────────────────────────────────

    /// Returns a fresh in-memory `ClipManager`.
    async fn make_manager() -> ClipManager {
        ClipManager::new(":memory:")
            .await
            .expect("ClipManager::new should succeed on :memory:")
    }

    // ─── test 1: full lifecycle ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_clip_lifecycle_create_search_update_delete() {
        let manager = make_manager().await;

        // 1. Add a clip
        let mut clip = Clip::new(PathBuf::from("/footage/interview.mov"));
        clip.set_name("Interview Take 1");
        clip.add_keyword("interview");
        clip.add_keyword("john-doe");
        let clip_id = manager
            .add_clip(clip)
            .await
            .expect("add_clip should succeed");

        // 2. Search → one result
        let results = manager
            .search("interview")
            .await
            .expect("search should succeed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Interview Take 1");

        // 3. Update the clip
        let mut updated = manager
            .get_clip(&clip_id)
            .await
            .expect("get_clip should succeed");
        updated.set_name("Interview Take 1 (Selected)");
        updated.set_rating(Rating::FourStars);
        manager
            .update_clip(updated)
            .await
            .expect("update_clip should succeed");

        // 4. Verify updated fields
        let reloaded = manager
            .get_clip(&clip_id)
            .await
            .expect("get_clip should succeed");
        assert_eq!(reloaded.name, "Interview Take 1 (Selected)");
        assert_eq!(reloaded.rating, Rating::FourStars);

        // 5. Delete
        manager
            .delete_clip(&clip_id)
            .await
            .expect("delete_clip should succeed");

        // 6. Search returns empty
        let after_delete = manager
            .search("interview")
            .await
            .expect("search should succeed");
        assert_eq!(after_delete.len(), 0);
    }

    // ─── test 2: EDL export round-trip ───────────────────────────────────────

    #[tokio::test]
    async fn test_export_edl_round_trip() {
        // Create a clip with defined in/out points (24 fps)
        // in_point = 0, out_point = 48 → 2 seconds at 24 fps
        let mut clip = Clip::new(PathBuf::from("/footage/scene01.mov"));
        clip.set_name("Scene 01");
        clip.set_in_point(0);
        clip.set_out_point(48);
        clip.set_duration(48);

        let clips = vec![clip];
        let fps = Rational::new(24, 1);
        let exporter = EdlExporter::new(fps);

        let edl = exporter
            .to_edl(&clips)
            .expect("to_edl should produce valid EDL");

        // Round-trip verification: the EDL must contain the clip name and the
        // 00:00:00:00 in-point timecode.
        assert!(
            edl.contains("TITLE: Clip Export"),
            "EDL must contain title header"
        );
        assert!(edl.contains("Scene 01"), "EDL must embed the clip name");
        // In-point of frame 0 → 00:00:00:00
        assert!(
            edl.contains("00:00:00:00"),
            "EDL must contain in-point timecode 00:00:00:00"
        );
        // Out-point of frame 48 at 24 fps → 00:00:02:00
        assert!(
            edl.contains("00:00:02:00"),
            "EDL must contain out-point timecode 00:00:02:00"
        );
    }

    // ─── test 3: clip_compare similarity ─────────────────────────────────────

    #[tokio::test]
    async fn test_clip_compare_identical_vs_different() {
        let comparer = ClipComparer::new();

        // Identical clips (same ID, same properties)
        let a = ComparableClip {
            id: 1,
            name: "Clip A".to_string(),
            in_point: 0,
            out_point: 240,
            rating: 4,
            keywords: vec!["interview".to_string()],
            codec: "H.264".to_string(),
            width: 1920,
            height: 1080,
            frame_rate: 24.0,
            color_label: String::new(),
            note: String::new(),
        };
        let a_copy = a.clone();

        let result_identical = comparer.compare(&a, &a_copy);
        assert!(
            result_identical.is_identical(),
            "Comparing a clip with itself must return no differences"
        );
        assert_eq!(result_identical.diff_count(), 0);

        // Different clip — different duration and codec
        let b = ComparableClip {
            id: 2,
            name: "Clip B".to_string(),
            in_point: 0,
            out_point: 480,
            rating: 2,
            keywords: vec!["action".to_string()],
            codec: "ProRes 422".to_string(),
            width: 3840,
            height: 2160,
            frame_rate: 29.97,
            color_label: String::new(),
            note: String::new(),
        };

        let result_different = comparer.compare(&a, &b);
        assert!(
            !result_different.is_identical(),
            "Clips with different properties must show differences"
        );
        assert!(
            result_different.diff_count() > 0,
            "Should report at least one difference"
        );
    }

    // ─── test 4: fingerprint similarity scores ────────────────────────────────

    #[test]
    fn test_clip_fingerprint_similarity_range() {
        // Build two identical fingerprints from the same frame data
        let identical_hash = [0xABu8; 32];
        let fp_a = ClipFingerprint {
            clip_id: "clip_a".to_string(),
            frames: vec![FrameFingerprint::new(identical_hash, 0)],
            duration_ms: 1000,
            sample_fps: 1.0,
        };
        let fp_b = ClipFingerprint {
            clip_id: "clip_b".to_string(),
            frames: vec![FrameFingerprint::new(identical_hash, 0)],
            duration_ms: 1000,
            sample_fps: 1.0,
        };

        let mut db = FingerprintDb::new(8.0);
        db.insert(fp_a.clone());

        let results = db.find_similar(&fp_b);
        assert_eq!(results.len(), 1);
        let sim = results[0].similarity;
        assert!(
            (0.0..=1.0).contains(&sim),
            "Similarity must be in [0.0, 1.0], got {sim}"
        );
        // Identical hashes → similarity must be 1.0
        assert!(
            (sim - 1.0).abs() < 1e-9,
            "Identical fingerprints should yield similarity ≈ 1.0, got {sim}"
        );

        // Build a maximally different fingerprint (all bits flipped)
        let different_hash = [0x00u8; 32]; // 0xAB ^ 0xFF ^ ... — use all zeros vs 0xFF for max diff
        let fp_c = ClipFingerprint {
            clip_id: "clip_c".to_string(),
            frames: vec![FrameFingerprint::new([0xFFu8; 32], 0)],
            duration_ms: 1000,
            sample_fps: 1.0,
        };
        let _ = different_hash; // silence lint

        let mut db2 = FingerprintDb::new(8.0);
        db2.insert(fp_c.clone());
        let fp_b_flipped = ClipFingerprint {
            clip_id: "clip_b_flipped".to_string(),
            frames: vec![FrameFingerprint::new([0x00u8; 32], 0)],
            duration_ms: 1000,
            sample_fps: 1.0,
        };
        let results2 = db2.find_similar(&fp_b_flipped);
        assert_eq!(results2.len(), 1);
        let low_sim = results2[0].similarity;
        assert!(
            (0.0..=1.0).contains(&low_sim),
            "Similarity must be in [0.0, 1.0], got {low_sim}"
        );
        assert!(
            low_sim < 0.1,
            "Maximally different fingerprints should yield low similarity, got {low_sim}"
        );
    }

    // ─── test 5: concurrent add_clips ────────────────────────────────────────

    #[tokio::test]
    async fn test_concurrent_add_clips() {
        // Use a file-backed temp DB so multiple `ClipManager` handles can share
        // the same SQLite database through the connection pool.
        let db_path = format!(
            "{}/clips_test_{}.db",
            std::env::temp_dir().display(),
            std::process::id()
        );
        // Ensure cleanup on test exit via a simple guard.
        let _guard = DbCleanupGuard(db_path.clone());

        let manager = Arc::new(
            ClipManager::new(&db_path)
                .await
                .expect("ClipManager::new should succeed"),
        );

        const TASKS: usize = 5;
        const CLIPS_PER_TASK: usize = 10;

        let mut handles = Vec::with_capacity(TASKS);
        for task_idx in 0..TASKS {
            let m = Arc::clone(&manager);
            let handle = tokio::spawn(async move {
                let clips: Vec<Clip> = (0..CLIPS_PER_TASK)
                    .map(|i| {
                        let mut c = Clip::new(PathBuf::from(format!(
                            "/footage/task{task_idx}_clip{i}.mov"
                        )));
                        c.set_name(format!("Task {task_idx} Clip {i}"));
                        c
                    })
                    .collect();

                m.add_clips(clips).await.expect("add_clips should succeed")
            });
            handles.push(handle);
        }

        // Await all tasks
        for handle in handles {
            handle.await.expect("tokio task should not panic");
        }

        let total = manager
            .clip_count()
            .await
            .expect("clip_count should succeed");
        assert_eq!(
            total,
            (TASKS * CLIPS_PER_TASK) as i64,
            "Expected {expected} clips, got {total}",
            expected = TASKS * CLIPS_PER_TASK
        );
    }

    // ─── test 6: bin_organizer auto-assignment by rating ─────────────────────

    #[test]
    fn test_bin_organizer_auto_assignment_by_rating() {
        // Create clip descriptors with different ratings
        let mut clip_high_1 = ClipDescriptor::new(1, "Scene 01 Take A");
        clip_high_1.rating = 5;
        clip_high_1.media_type = ClipMediaType::Video;

        let mut clip_high_2 = ClipDescriptor::new(2, "Scene 01 Take B");
        clip_high_2.rating = 4;
        clip_high_2.media_type = ClipMediaType::Video;

        let mut clip_low = ClipDescriptor::new(3, "Scene 02 Take A");
        clip_low.rating = 2;
        clip_low.media_type = ClipMediaType::Video;

        let clips = vec![clip_high_1, clip_high_2, clip_low];

        // Organize by rating — each distinct rating value becomes its own bin
        let organizer = BinOrganizer::with_criteria(OrganizeCriteria::ByRating);
        let bins = organizer.organize(&clips);

        // There are 3 distinct ratings (5, 4, 2) → 3 bins
        assert_eq!(bins.len(), 3, "Expected 3 rating bins, got {}", bins.len());

        // Verify that each bin contains the correct number of clips
        let total_clips: usize = bins.iter().map(|b| b.clip_count()).sum();
        assert_eq!(
            total_clips, 3,
            "All clips should be assigned to a bin, got {total_clips}"
        );

        // The BinOrganizer formats ratings as "N Stars" (e.g. "5 Stars", "4 Stars").
        // Confirm the "5 Stars" and "4 Stars" bins each have 1 clip and "2 Stars" bin has 1 clip.
        let bin_5 = bins.iter().find(|b| b.name == "5 Stars");
        let bin_4 = bins.iter().find(|b| b.name == "4 Stars");
        let bin_2 = bins.iter().find(|b| b.name == "2 Stars");

        assert!(bin_5.is_some(), "Should have a bin named '5 Stars'");
        assert!(bin_4.is_some(), "Should have a bin named '4 Stars'");
        assert!(bin_2.is_some(), "Should have a bin named '2 Stars'");

        assert_eq!(
            bin_5.expect("bin_5 checked above").clip_count(),
            1,
            "Rating-5 bin should contain 1 clip"
        );
        assert_eq!(
            bin_4.expect("bin_4 checked above").clip_count(),
            1,
            "Rating-4 bin should contain 1 clip"
        );
    }

    // ─── test 7: bin_organizer apply_rules ───────────────────────────────────

    #[test]
    fn test_bin_organizer_apply_rules_by_camera() {
        let clips = vec![
            {
                let mut c = ClipDescriptor::new(1, "Clip A-Cam");
                c.camera = Some("Camera A".to_string());
                c.media_type = ClipMediaType::Video;
                c
            },
            {
                let mut c = ClipDescriptor::new(2, "Clip B-Cam");
                c.camera = Some("Camera B".to_string());
                c.media_type = ClipMediaType::Video;
                c
            },
            {
                let mut c = ClipDescriptor::new(3, "Clip A-Cam 2");
                c.camera = Some("Camera A".to_string());
                c.media_type = ClipMediaType::Video;
                c
            },
        ];

        let mut organizer = BinOrganizer::new();
        organizer.add_rule(BinRule::new(
            "A Camera",
            OrganizeCriteria::ByCamera,
            "Camera A",
        ));

        let bins = organizer.apply_rules(&clips);
        assert_eq!(
            bins.len(),
            1,
            "Only the rule-matching bin should be returned"
        );
        assert_eq!(bins[0].name, "A Camera");
        assert_eq!(bins[0].clip_count(), 2, "Camera A should have 2 clips");
    }

    // ─── helper: temp DB cleanup guard ───────────────────────────────────────

    struct DbCleanupGuard(String);

    impl Drop for DbCleanupGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
}
