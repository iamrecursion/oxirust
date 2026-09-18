//! Smoke tests for newly-wired orphan modules in oximedia-forensics.

// ─── artifact_analysis ───────────────────────────────────────────────────────
#[test]
fn test_artifact_analysis_solid_frame_low_blockiness() {
    use oximedia_forensics::artifact_analysis::CompressionArtifactAnalyzer;

    // Solid-colour frame — no blockiness
    let frame = vec![128u8; 64 * 64 * 3];
    let score = CompressionArtifactAnalyzer::blockiness(&frame, 64, 64);
    assert!(
        score < 0.5,
        "solid frame should have very low blockiness, got {score}"
    );
}

// ─── audio_forensics ─────────────────────────────────────────────────────────
#[test]
fn test_audio_forensics_returns_results() {
    use oximedia_forensics::audio_forensics::detect_audio_splices;

    // 1 second of constant amplitude signal at 44100 Hz
    let n = 44100usize;
    let signal: Vec<f32> = vec![0.1_f32; n];

    // Should run without panicking; result may or may not contain splices
    let splices = detect_audio_splices(&signal, 44100);
    // A constant signal has no sudden changes — verify result is a valid Vec
    let _ = splices.len(); // just confirm the return type is usable
}

// ─── batch_forensics ─────────────────────────────────────────────────────────
#[test]
fn test_batch_forensics_empty_batch() {
    use oximedia_forensics::batch_forensics::{
        AnalysisFn, BatchForensicAnalyzer, FileForensicResult,
    };

    let analyze_fn: &AnalysisFn = &|_path, _data| {
        FileForensicResult::success("test", false, 0.0, std::collections::HashMap::new(), vec![])
    };
    let analyzer = BatchForensicAnalyzer::new();
    let report = analyzer.analyze_batch(&[], analyze_fn);
    assert_eq!(report.total_files, 0, "empty batch should have 0 total");
}

#[test]
fn test_batch_forensics_single_file() {
    use oximedia_forensics::batch_forensics::{
        AnalysisFn, BatchForensicAnalyzer, FileForensicResult,
    };

    let data: &[u8] = b"fake image data";
    let analyze_fn: &AnalysisFn = &|path, _data| {
        FileForensicResult::success(path, false, 0.1, std::collections::HashMap::new(), vec![])
    };

    let analyzer = BatchForensicAnalyzer::new();
    let report = analyzer.analyze_batch(&[("test.jpg", data)], analyze_fn);
    assert_eq!(report.total_files, 1, "should have 1 result");
}

// ─── chromatic_forensics ─────────────────────────────────────────────────────
#[test]
fn test_chromatic_forensics_config_defaults() {
    use oximedia_forensics::chromatic_forensics::ChromaticConfig;

    let cfg = ChromaticConfig::default();
    assert!(cfg.block_size > 0, "block_size should be positive");
    assert!(cfg.anomaly_sigma > 0.0, "anomaly_sigma should be positive");
}

#[test]
fn test_chromatic_forensics_analyze_solid_image() {
    use oximedia_forensics::chromatic_forensics::{analyze_chromatic_aberration, ChromaticConfig};

    let cfg = ChromaticConfig::default();
    // 64×64 solid-colour RGB frame — extract separate channels as f64
    let channel: Vec<f64> = vec![0.5_f64; 64 * 64];
    let result = analyze_chromatic_aberration(&channel, &channel, &channel, 64, 64, &cfg);
    // Should not panic; consistency score in [0, 1]
    assert!(
        result.consistency_score >= 0.0 && result.consistency_score <= 1.0,
        "consistency score must be in [0, 1]"
    );
}

// ─── custody ─────────────────────────────────────────────────────────────────
#[test]
fn test_custody_chain_verify() {
    use oximedia_forensics::custody::ChainOfCustody;

    let mut chain = ChainOfCustody::new(42);
    chain.add_event("ingest", 1001, 1_700_000_000);
    chain.add_event("transcode", 1002, 1_700_000_100);
    chain.add_event("deliver", 1003, 1_700_000_200);
    assert!(chain.verify(), "valid chain should verify");
}

#[test]
fn test_custody_chain_empty_is_valid() {
    use oximedia_forensics::custody::ChainOfCustody;

    let chain = ChainOfCustody::new(1);
    assert!(chain.verify(), "empty chain should be trivially valid");
}

// ─── deepfake_detect ─────────────────────────────────────────────────────────
#[test]
fn test_deepfake_detect_stable_landmarks_not_flagged() {
    use oximedia_forensics::deepfake_detect::{check_landmark_consistency, FaceLandmarks};

    // 10 frames with identical 5-point landmarks — no jitter
    let frames: Vec<FaceLandmarks> = (0..10)
        .map(|_| {
            FaceLandmarks::new(vec![
                (100.0, 200.0),
                (150.0, 200.0),
                (125.0, 230.0),
                (110.0, 260.0),
                (140.0, 260.0),
            ])
        })
        .collect();

    let consistency = check_landmark_consistency(&frames);
    // Stable landmarks should have low inter-frame deviation
    assert!(
        consistency.inter_frame_deviation < 1.0,
        "stable landmarks should have near-zero deviation, got {}",
        consistency.inter_frame_deviation
    );
    assert!(
        !consistency.blink_anomaly,
        "stable landmarks should not trigger blink anomaly"
    );
}

// ─── double_compression ──────────────────────────────────────────────────────
#[test]
fn test_double_compression_histogram_periodicity() {
    use oximedia_forensics::double_compression::{histogram_periodicity, DctHistogram};

    let mut hist = DctHistogram::new(128);
    // Flat uniform distribution — should have near-zero periodicity
    for v in -128..=128i32 {
        hist.add(v);
    }
    let score = histogram_periodicity(&hist, 10);
    assert!(score >= 0.0, "periodicity score should be non-negative");
}

#[test]
fn test_double_compression_benford_chi_squared() {
    use oximedia_forensics::double_compression::{benford_chi_squared, benford_expected};

    let expected = benford_expected();
    // Using expected as observed → chi-squared should be near 0
    let chi2 = benford_chi_squared(&expected);
    assert!(
        chi2 < 1e-6,
        "chi-squared of expected vs expected should be near 0"
    );
}

// ─── phylogeny ───────────────────────────────────────────────────────────────
#[test]
fn test_phylogeny_tree_root_selection() {
    use oximedia_forensics::phylogeny::{PhylogenyAnalyzer, PhylogenyNode};

    let nodes = vec![
        PhylogenyNode::new("v1".to_string(), 8, 8, vec![0u8; 64]),
        PhylogenyNode::new("v2".to_string(), 8, 8, vec![128u8; 64]),
        PhylogenyNode::new("v3".to_string(), 8, 8, vec![255u8; 64]),
    ];
    let analyzer = PhylogenyAnalyzer::new();
    let tree = analyzer
        .build_tree(nodes)
        .expect("tree construction should succeed");
    // A tree over 3 nodes should have depth >= 1
    assert!(tree.depth() >= 1, "tree should have at least depth 1");
}

// ─── quantization_table ──────────────────────────────────────────────────────
#[test]
fn test_quantization_table_quality_estimate() {
    use oximedia_forensics::quantization_table::{
        estimate_quality_factor, QuantTable, STANDARD_LUMINANCE_Q50,
    };

    let table = QuantTable::new(STANDARD_LUMINANCE_Q50);
    let qf = estimate_quality_factor(&table);
    // Q50 standard table should give QF near 50
    assert!(qf.is_some(), "should estimate quality factor");
    let qf_val = qf.unwrap();
    assert!(
        (qf_val as i32 - 50).abs() <= 5,
        "quality factor should be near 50, got {qf_val}"
    );
}

// ─── tamper_detect ───────────────────────────────────────────────────────────
#[test]
fn test_tamper_detect_mismatch_detected() {
    use oximedia_forensics::tamper_detect::MetadataTamperDetector;
    use std::collections::HashMap;

    let mut embedded = HashMap::new();
    embedded.insert("camera_model".to_string(), "CanonEOS".to_string());
    embedded.insert("timestamp".to_string(), "2024-01-01T00:00:00Z".to_string());

    let mut external = HashMap::new();
    external.insert("camera_model".to_string(), "CanonEOS".to_string());
    external.insert("timestamp".to_string(), "2024-06-15T12:00:00Z".to_string());

    let issues = MetadataTamperDetector::check(&embedded, &external);
    assert!(!issues.is_empty(), "timestamp mismatch should be detected");
}

#[test]
fn test_tamper_detect_identical_metadata_clean() {
    use oximedia_forensics::tamper_detect::MetadataTamperDetector;
    use std::collections::HashMap;

    let mut meta = HashMap::new();
    meta.insert("camera_model".to_string(), "SonyA7".to_string());
    meta.insert("iso".to_string(), "800".to_string());

    let issues = MetadataTamperDetector::check(&meta, &meta.clone());
    assert!(
        issues.is_empty(),
        "identical metadata should have no issues"
    );
}

// ─── video_forensics ─────────────────────────────────────────────────────────
#[test]
fn test_video_forensics_clean_sequence_no_splices() {
    use oximedia_forensics::video_forensics::{
        detect_temporal_splices, FrameStatistics, TemporalSpliceConfig,
    };

    // Slowly varying brightness — no splice
    let frames: Vec<FrameStatistics> = (0..20)
        .map(|i| FrameStatistics {
            frame_index: i,
            mean_brightness: 0.5 + 0.01 * i as f32,
            brightness_std: 0.1,
            noise_estimate: 0.02,
            histogram: [0.0625_f32; 16],
        })
        .collect();

    let cfg = TemporalSpliceConfig::default();
    let result = detect_temporal_splices(&frames, &cfg);
    assert!(
        result.splice_points.is_empty(),
        "smooth sequence should have no splice points"
    );
}
