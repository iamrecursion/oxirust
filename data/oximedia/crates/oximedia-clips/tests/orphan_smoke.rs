//! Smoke tests for newly-wired orphan modules in oximedia-clips.

// ─── bin ─────────────────────────────────────────────────────────────────────
#[test]
fn test_bin_clip_management() {
    use oximedia_clips::bin::ClipBin;

    let mut root = ClipBin::new("Root");
    let child_idx = root.add_child_bin("Day 1");
    root.add_clip(100);
    root.add_clip(101);

    assert_eq!(root.list_clips(), &[100, 101]);
    assert_eq!(root.child_bin_count(), 1);
    let child = root
        .get_child_bin(child_idx)
        .expect("child bin should exist");
    assert_eq!(child.name(), "Day 1");
}

// ─── clip_ai_tag ─────────────────────────────────────────────────────────────
#[test]
fn test_clip_ai_tag_suggestions() {
    use oximedia_clips::clip::Clip;
    use oximedia_clips::clip_ai_tag::{AiTaggerConfig, ClipAiTagger};
    use std::path::PathBuf;

    let clip = Clip::new(PathBuf::from("/footage/interview_daylight.mov"));

    let tagger = ClipAiTagger::new(AiTaggerConfig::default());
    let result = tagger.tag_clip(&clip);
    // All suggestions have confidence in [0, 1]
    for s in &result.suggestions {
        assert!(
            s.confidence >= 0.0 && s.confidence <= 1.0,
            "confidence must be in [0,1]"
        );
    }
}

// ─── clip_collaboration ──────────────────────────────────────────────────────
#[test]
fn test_clip_collaboration_shared_board() {
    use oximedia_clips::clip_collaboration::{
        AnnotationField, AnnotationOp, SharedAnnotationBoard, TextOp, UserId,
    };
    use oximedia_clips::ClipId;
    use uuid::Uuid;

    let mut board = SharedAnnotationBoard::new();
    let user = UserId("user-1".to_string());
    let clip_id = ClipId::from_uuid(Uuid::nil());

    let op = AnnotationOp {
        id: oximedia_clips::clip_collaboration::OpId(0),
        user_id: user,
        clip_id,
        field: AnnotationField::Note,
        op: TextOp::Replace {
            text: "scene 1 description".to_string(),
        },
        base_revision: 0,
        timestamp: chrono::Utc::now(),
    };
    board.submit(op);
    let text = board.get(&clip_id, &AnnotationField::Note);
    assert!(text.is_some(), "annotation should be stored");
    assert_eq!(text.unwrap(), "scene 1 description");
}

// ─── clip_face_index ─────────────────────────────────────────────────────────
#[test]
fn test_clip_face_index_identity_search() {
    use oximedia_clips::clip_face_index::{ClipFaceIndex, FaceEntry};

    let mut index = ClipFaceIndex::new(4);
    index.add_identity("alice".to_string(), vec![1.0_f32, 0.0, 0.0, 0.0]);

    let entry = FaceEntry {
        clip_id: "clip-01".to_string(),
        frame_number: 120,
        bounding_box: (10, 20, 80, 90),
        embedding: vec![1.0_f32, 0.0, 0.0, 0.0],
        confidence: 0.97,
    };
    index.add_face(entry);

    let clips = index.clips_by_identity("alice", 0.9);
    assert!(!clips.is_empty(), "should find clip containing alice");
    assert!(clips.contains(&"clip-01".to_string()));
}

// ─── clip_favorites ──────────────────────────────────────────────────────────
#[test]
fn test_clip_favorites_and_recent() {
    use oximedia_clips::clip_favorites::{FavoriteCollection, RecentClipList};
    use oximedia_clips::ClipId;
    use uuid::Uuid;

    let id_a = ClipId::from_uuid(Uuid::nil());

    let mut fav = FavoriteCollection::new("My Picks".to_string());
    fav.add(id_a);
    assert_eq!(fav.len(), 1);

    let mut recent = RecentClipList::new(5);
    recent.record_access(id_a);
    assert_eq!(recent.most_recent(), Some(id_a));
}

// ─── clip_scene_detect ───────────────────────────────────────────────────────
#[test]
fn test_clip_scene_detect_threshold_detector() {
    use oximedia_clips::clip_scene_detect::ThresholdSceneDetector;

    let detector = ThresholdSceneDetector::new(0.3);
    let frames: Vec<Vec<f32>> = vec![
        vec![0.0; 16], // frame 0 — black
        vec![0.0; 16], // frame 1 — black (same scene)
        vec![1.0; 16], // frame 2 — white (cut!)
        vec![0.9; 16], // frame 3 — still white
    ];
    let boundaries = detector.detect(&frames);
    assert_eq!(boundaries.len(), 1);
    assert_eq!(boundaries[0].frame_number, 2);
}

// ─── clip_transcode_status ───────────────────────────────────────────────────
#[test]
fn test_clip_transcode_status_lifecycle() {
    use oximedia_clips::clip_transcode_status::{
        TranscodeJob, TranscodeState, TranscodeStatusStore,
    };

    let mut store = TranscodeStatusStore::new();
    let job = TranscodeJob::new("clip-01".to_string(), "proxy-720p".to_string());
    let job_id = job.job_id.clone();
    store.register(job);

    store.update_progress(&job_id, 0.5);
    assert_eq!(
        store.job(&job_id).map(|j| j.state.clone()),
        Some(TranscodeState::Running),
    );

    store.complete(&job_id, "/cache/clip-01-proxy.mov".to_string());
    assert!(
        matches!(
            store.job(&job_id).map(|j| &j.state),
            Some(TranscodeState::Completed { .. })
        ),
        "job should be completed"
    );
}

// ─── clip_versioning ─────────────────────────────────────────────────────────
#[test]
fn test_clip_versioning_undo_redo() {
    use oximedia_clips::clip_versioning::{ClipEditOperation, ClipVersionHistory};

    let mut history = ClipVersionHistory::new("clip-01".to_string());
    history.push(ClipEditOperation::NameChanged {
        before: "raw_take_1.mov".to_string(),
        after: "Interview A".to_string(),
    });
    assert_eq!(history.undo_count(), 1);

    let _op = history.undo().expect("should have undo");
    assert_eq!(history.undo_count(), 0);
    assert_eq!(history.redo_count(), 1);
}

// ─── offline ─────────────────────────────────────────────────────────────────
#[test]
fn test_offline_detector_missing_file() {
    use oximedia_clips::offline::ClipOfflineDetector;

    // A non-existent path should report as offline
    let is_online = ClipOfflineDetector::check(42, "/nonexistent/media/shot_abc.mov");
    assert!(!is_online, "non-existent path should be offline");
}

#[test]
fn test_offline_detector_temp_file() {
    use oximedia_clips::offline::ClipOfflineDetector;
    use std::io::Write;

    // Write a real temp file and check it's online
    let mut path = std::env::temp_dir();
    path.push("oximedia_clips_offline_smoke.tmp");
    {
        let mut f = std::fs::File::create(&path).expect("create temp file");
        f.write_all(b"test").expect("write temp file");
    }
    let is_online = ClipOfflineDetector::check(99, path.to_str().expect("valid path"));
    assert!(is_online, "existing temp file should be online");
    let _ = std::fs::remove_file(&path);
}

// ─── rating_store ────────────────────────────────────────────────────────────
#[test]
fn test_rating_store_stars_and_color() {
    use oximedia_clips::rating_store::ClipRating;

    let mut rating = ClipRating::new();
    rating.set_stars(1, 4);
    rating.set_color(1, "green");

    let (stars, color) = rating.get(1).expect("clip 1 should have a rating");
    assert_eq!(stars, 4);
    assert_eq!(color, "green");
}

// ─── usage ───────────────────────────────────────────────────────────────────
#[test]
fn test_usage_tracker_reverse_lookup() {
    use oximedia_clips::usage::ClipUsageTracker;

    let mut tracker = ClipUsageTracker::new();
    tracker.record_use(10, 1001);
    tracker.record_use(10, 1002);
    tracker.record_use(20, 1001);

    let seqs = tracker.usages_of(10);
    assert!(seqs.contains(&1001));
    assert!(seqs.contains(&1002));
    assert!(!seqs.contains(&9999));
}
