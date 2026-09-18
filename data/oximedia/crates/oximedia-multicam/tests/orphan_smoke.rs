//! Smoke tests for newly-wired orphan modules in oximedia-multicam.

// ── audio_selection ───────────────────────────────────────────────────────────

#[test]
fn test_multi_angle_audio_mixer_follows_video_by_default() {
    use oximedia_multicam::audio_selection::{AudioAngle, MultiAngleAudioMixer};

    let angles = vec![
        AudioAngle {
            angle_id: 0,
            channels: 2,
            sample_rate: 48_000,
        },
        AudioAngle {
            angle_id: 1,
            channels: 4,
            sample_rate: 48_000,
        },
    ];
    let mixer = MultiAngleAudioMixer::new(0, None, angles);
    assert_eq!(mixer.current_audio_angle(), 0);
    assert!(mixer.is_audio_following_video());
}

// ── auto_frame ────────────────────────────────────────────────────────────────

#[test]
fn test_auto_framer_constructs_and_computes_crop() {
    use oximedia_multicam::auto_frame::{AutoFrameConfig, AutoFramer, SubjectPosition};

    let config = AutoFrameConfig::default();
    assert!(config.validate().is_ok());
    let mut framer = AutoFramer::new(config, 1920, 1080).expect("construct ok");
    let subject = SubjectPosition::new(960.0, 540.0);
    let crop = framer.compute_crop(&subject);
    assert!(crop.aspect_ratio() > 0.0);
}

// ── cut_export ────────────────────────────────────────────────────────────────

#[test]
fn test_cut_exporter_to_edl_contains_title() {
    use oximedia_multicam::cut_export::MulticamCutExporter;

    let cuts = [(0u64, 1u64, 0u64), (5000, 2, 5000)];
    let edl = MulticamCutExporter::to_edl(&cuts);
    assert!(edl.contains("TITLE"), "EDL should have a TITLE line");
}

// ── fcp_xml ───────────────────────────────────────────────────────────────────

#[test]
fn test_fcp_xml_exporter_default_config() {
    use oximedia_multicam::fcp_xml::MultiCamXmlExporter;

    let exporter = MultiCamXmlExporter::default_config();
    let _e = exporter; // confirm we can create it
}

// ── framing_suggest ───────────────────────────────────────────────────────────

#[test]
fn test_framing_adjustment_acceptable() {
    use oximedia_multicam::framing_suggest::FramingAdjustment;

    let adj = FramingAdjustment::acceptable();
    assert!(
        adj.magnitude() < 0.05,
        "acceptable adjustment should have near-zero magnitude"
    );
}

// ── genlock ───────────────────────────────────────────────────────────────────

#[test]
fn test_genlock_monitor_camera_management() {
    use oximedia_multicam::genlock::GenlockMonitor;

    let mut monitor = GenlockMonitor::with_defaults().expect("construct ok");
    let id = monitor.add_camera("CAM_A");
    assert_eq!(monitor.camera_count(), 1);
    monitor.remove_camera(id);
    assert_eq!(monitor.camera_count(), 0);
}

// ── proxy_gen (existing) + proxy_generator (new) ─────────────────────────────

#[test]
fn test_multi_angle_proxy_generator_stub() {
    use oximedia_multicam::proxy_generator::MultiAngleProxyGenerator;

    let gen = MultiAngleProxyGenerator::with_defaults();
    let proxy = gen.generate_stub(0, 10);
    assert!(proxy.is_valid());
    assert_eq!(proxy.angle_id, 0);
}

// ── sub_frame_sync ────────────────────────────────────────────────────────────

#[test]
fn test_sub_frame_offset_total_frames() {
    use oximedia_multicam::sub_frame_sync::SubFrameOffset;

    let offset = SubFrameOffset {
        angle_id: 0,
        integer_frames: 5,
        fractional_frames: 0.25,
        confidence: 0.9,
    };
    let total = offset.total_frames();
    assert!((total - 5.25).abs() < 1e-9, "expected 5.25, got {total}");
}

// ── sync_points ───────────────────────────────────────────────────────────────

#[test]
fn test_sync_point_manager_nearest() {
    use oximedia_multicam::sync_points::SyncPointManager;

    let mut mgr = SyncPointManager::new();
    mgr.add(0, "slate");
    mgr.add(30_000, "scene_2");
    mgr.add(90_000, "scene_3_end");

    let nearest = mgr.nearest(31_000).expect("should find nearest");
    assert_eq!(nearest.label, "scene_2");
}

// ── sync_verify_parallel ──────────────────────────────────────────────────────

#[test]
fn test_parallel_sync_verifier_constructs() {
    use oximedia_multicam::sync_verify_parallel::{ParallelSyncVerifier, ParallelVerifyConfig};

    let config = ParallelVerifyConfig::default();
    let verifier = ParallelSyncVerifier::new(config);
    let _ = verifier; // confirm it constructs
}

// ── visca ─────────────────────────────────────────────────────────────────────

#[test]
fn test_visca_command_pan_tilt_drive_builds() {
    use oximedia_multicam::visca::ViscaCommand;

    let cmd = ViscaCommand::pan_tilt_drive(8, 8, 1, 1);
    assert!(!cmd.description.is_empty());
    assert!(!cmd.is_inquiry());
}
