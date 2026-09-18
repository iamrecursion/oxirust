//! Smoke tests for newly-wired orphan modules in `oximedia-analysis`.

// ── bitrate_analysis ──────────────────────────────────────────────────────────
#[test]
fn bitrate_analyzer_sample_stats() {
    use oximedia_analysis::bitrate_analysis::{BitrateAnalyzer, BitrateSample};
    let mut analyzer = BitrateAnalyzer::with_defaults();
    analyzer.add_sample(BitrateSample::new(0, 4_000_000.0, true));
    analyzer.add_sample(BitrateSample::new(1, 2_000_000.0, false));
    analyzer.add_sample(BitrateSample::new(2, 3_000_000.0, false));
    assert_eq!(analyzer.sample_count(), 3);
    let stats = analyzer.compute_stats();
    assert_eq!(stats.count, 3);
    assert!(stats.mean_bps > 0.0);
}

#[test]
fn bitrate_variability_constant_classification() {
    use oximedia_analysis::bitrate_analysis::{BitrateAnalyzer, BitrateSample, BitrateVariability};
    let mut analyzer = BitrateAnalyzer::with_defaults();
    // All same bitrate → near-zero CV → Constant
    for i in 0..10u64 {
        analyzer.add_sample(BitrateSample::new(i, 2_000_000.0, i == 0));
    }
    let var = analyzer.classify_variability();
    assert!(matches!(
        var,
        BitrateVariability::Constant | BitrateVariability::Moderate
    ));
}

// ── vmaf_estimator ────────────────────────────────────────────────────────────
#[test]
fn vmaf_estimator_identical_frames_near_100() {
    use oximedia_analysis::vmaf_estimator::VmafEstimator;
    let frame = vec![128u8; 64 * 64];
    let score = VmafEstimator::estimate(&frame, &frame, 64, 64);
    assert!(score >= 95.0, "expected near-100 VMAF, got {score}");
}

#[test]
fn vmaf_estimator_severely_distorted_lower_score() {
    use oximedia_analysis::vmaf_estimator::VmafEstimator;
    let reference = vec![200u8; 64 * 64];
    let distorted = vec![0u8; 64 * 64];
    let score = VmafEstimator::estimate(&reference, &distorted, 64, 64);
    assert!(
        score < 95.0,
        "expected lower VMAF for distorted frame, got {score}"
    );
}

// ── frozen_frame ──────────────────────────────────────────────────────────────
#[test]
fn frozen_frame_detector_no_freeze_varying_frames() {
    use oximedia_analysis::frozen_frame::{FrameHash, FrozenFrameDetector};
    let detector = FrozenFrameDetector::default_params();
    // Build 5 distinct frames (different luma values).
    let hashes: Vec<FrameHash> = (0..5u8)
        .map(|i| {
            let luma: Vec<u8> = vec![i * 50; 64];
            FrameHash::from_luma(i as usize, &luma)
        })
        .collect();
    let ranges = detector.detect(&hashes);
    assert!(
        ranges.is_empty(),
        "expected no frozen ranges, got {}",
        ranges.len()
    );
}

// ── commercial_detect ────────────────────────────────────────────────────────
#[test]
fn commercial_detector_empty_input_no_breaks() {
    use oximedia_analysis::commercial_detect::CommercialDetector;
    let detector = CommercialDetector::default_params();
    let breaks = detector.detect_breaks(&[]);
    assert!(breaks.is_empty());
}

// ── gamut_analyzer ───────────────────────────────────────────────────────────
#[test]
fn gamut_standard_primaries_ordered() {
    use oximedia_analysis::gamut_analyzer::Gamut;
    let rec709 = Gamut::rec709();
    let rec2020 = Gamut::rec2020();
    // Rec.2020 has wider gamut (higher max_saturation) than Rec.709
    assert!(rec2020.max_saturation > rec709.max_saturation);
}

#[test]
fn gamut_analyzer_grey_frame_zero_out_of_gamut() {
    use oximedia_analysis::gamut_analyzer::{Gamut, GamutAnalyzer};
    // A perfectly grey frame (R=G=B=128) has zero saturation → all in gamut.
    let frame = vec![128u8; 64 * 64 * 3];
    let ratio = GamutAnalyzer::out_of_gamut_ratio(&frame, 64, 64, &Gamut::rec709());
    assert_eq!(ratio, 0.0, "grey frame should be 100% in gamut");
}

// ── segment_summary ───────────────────────────────────────────────────────────
#[test]
fn segment_summarizer_empty_state() {
    use oximedia_analysis::segment_summary::SegmentSummarizer;
    let summarizer = SegmentSummarizer::with_defaults();
    assert_eq!(summarizer.segment_count(), 0);
    assert_eq!(summarizer.total_frames(), 0);
}

// ── multi_pass ────────────────────────────────────────────────────────────────
#[test]
fn multi_pass_first_pass_empty_returns_error() {
    use oximedia_analysis::multi_pass::MultiPassAnalyzer;
    let analyzer = MultiPassAnalyzer::new();
    let result = analyzer.analyze_first_pass(&[]);
    assert!(result.is_err(), "empty first pass should return an error");
}

#[test]
fn multi_pass_first_pass_single_frame() {
    use oximedia_analysis::multi_pass::MultiPassAnalyzer;
    let analyzer = MultiPassAnalyzer::new();
    let luma = vec![128u8; 64 * 64];
    let frames = [luma.as_slice()];
    let result = analyzer.analyze_first_pass(&frames);
    assert!(result.is_ok());
    let pass1 = result.expect("single frame first pass should succeed");
    assert!(pass1.has_data());
    assert_eq!(pass1.frame_count, 1);
}

// ── complexity_metrics ────────────────────────────────────────────────────────
#[test]
fn complexity_config_default_valid() {
    use oximedia_analysis::complexity_metrics::ComplexityConfig;
    let cfg = ComplexityConfig::default();
    // DCT block size should be a positive power-of-2 (at least 4).
    assert!(cfg.dct_block_size >= 4);
}

// ── composition_analyzer (shot_composition) ───────────────────────────────────
#[test]
fn composition_config_default_valid() {
    use oximedia_analysis::shot_composition::CompositionConfig;
    let cfg = CompositionConfig::default();
    // Grid cols and rows should be >= 2 for rule-of-thirds analysis.
    assert!(cfg.grid_cols >= 2);
    assert!(cfg.grid_rows >= 2);
}

// ── frequency_analysis ────────────────────────────────────────────────────────
#[test]
fn frequency_analysis_config_default() {
    use oximedia_analysis::frequency_analysis::FrequencyAnalysisConfig;
    let cfg = FrequencyAnalysisConfig::default();
    assert!(cfg.window_size > 0);
}

// ── spatial_info ──────────────────────────────────────────────────────────────
#[test]
fn spatial_info_analyzer_initial_empty() {
    use oximedia_analysis::spatial_info::SpatialInfoAnalyzer;
    let analyzer = SpatialInfoAnalyzer::new();
    assert_eq!(analyzer.frame_count(), 0);
    assert!(analyzer.results().is_empty());
}

// ── bitrate_recommender ───────────────────────────────────────────────────────
#[test]
fn bitrate_recommender_returns_positive_rate() {
    use oximedia_analysis::bitrate_recommender::BitrateRecommender;
    let recommender = BitrateRecommender::new();
    // 1080p 30fps medium complexity → should recommend a positive bitrate.
    let rec = recommender.recommend(1920, 1080, 30.0, 0.5);
    assert!(
        rec.bps > 0,
        "expected positive bps recommendation, got {}",
        rec.bps
    );
    assert!(rec.kbps > 0.0);
    assert!(rec.mbps > 0.0);
}
