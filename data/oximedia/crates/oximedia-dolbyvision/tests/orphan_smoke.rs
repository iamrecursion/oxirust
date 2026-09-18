//! Smoke tests for the 25 newly-wired orphan modules in oximedia-dolbyvision.
//!
//! Each test exercises the main public type(s) of one module, verifying at
//! least one meaningful invariant.

use oximedia_dolbyvision::{DolbyVisionRpu, Level1Metadata, Profile};

// ── auto_profile_detect ───────────────────────────────────────────────────────

#[test]
fn test_auto_profile_detect_smoke() {
    use oximedia_dolbyvision::auto_profile_detect::{ProfileDetector, RpuHeaderFields};
    // Use a low confidence threshold so we always get a candidate.
    let detector = ProfileDetector::new(0.0);
    let rpu = DolbyVisionRpu::new(Profile::Profile8);
    let fields = RpuHeaderFields::from_rpu(&rpu);
    let result = detector.detect(&fields);
    // With threshold 0.0 we always get at least one candidate.
    assert!(
        result.is_some(),
        "expected at least one detected profile at threshold=0.0"
    );
    let detected = result.unwrap();
    assert!(
        detected.confidence >= 0.0,
        "confidence must be non-negative"
    );
}

// ── batch_rpu ─────────────────────────────────────────────────────────────────

#[test]
fn test_batch_rpu_smoke() {
    use oximedia_dolbyvision::batch_rpu::{BatchConfig, BatchRpuProcessor, ErrorPolicy};
    let config = BatchConfig {
        error_policy: ErrorPolicy::Skip,
        ..BatchConfig::default()
    };
    let mut processor = BatchRpuProcessor::new(config);
    let rpu = DolbyVisionRpu::new(Profile::Profile8);
    processor
        .process_frame(0, Ok(rpu))
        .expect("process_frame should succeed for valid RPU");
    let result = processor.finish();
    assert_eq!(
        result.stats.total_frames, 1,
        "should have processed 1 frame"
    );
    assert_eq!(result.errors.len(), 0, "no errors expected for valid RPU");
}

// ── conformance ───────────────────────────────────────────────────────────────

#[test]
fn test_conformance_smoke() {
    use oximedia_dolbyvision::conformance;
    // The public helper functions must succeed for a valid Profile 8 RPU.
    conformance::verify_profile8_minimal_roundtrip()
        .expect("Profile 8 conformance round-trip should pass");
    conformance::verify_level1_roundtrip().expect("L1 conformance round-trip should pass");
}

// ── display_mapping ───────────────────────────────────────────────────────────

#[test]
fn test_display_mapping_smoke() {
    use oximedia_dolbyvision::display_mapping::{DisplayMapper, DisplayProfile};
    // Source mastered at 4000 nit.
    let mapper = DisplayMapper::new(DisplayProfile::HDR_4000);
    // Target: 100-nit SDR display.
    let params = mapper.compute_trim(&DisplayProfile::SDR_100);
    // Trim slope must be positive (range compression from 4000→100 nit).
    assert!(
        params.trim_slope > 0.0,
        "trim_slope={} must be > 0.0",
        params.trim_slope
    );
}

// ── dv_compliance ─────────────────────────────────────────────────────────────

#[test]
fn test_dv_compliance_smoke() {
    use oximedia_dolbyvision::dv_compliance::{ComplianceChecker, ComplianceSpec};
    use oximedia_dolbyvision::dv_xml_export::{DvL2Entry, DvShotEntry};
    let checker = ComplianceChecker::new(ComplianceSpec::Permissive);
    let shots = vec![DvShotEntry {
        frame_start: 0,
        frame_end: 23,
        l1_min: 0.001,
        l1_mid: 0.10,
        l1_max: 0.58,
        l2_entries: vec![DvL2Entry::identity(2081)],
    }];
    let report = checker.check_shots(&shots, None);
    // A valid shot under Permissive spec should produce no violations.
    assert!(
        report.violations.is_empty(),
        "expected no compliance violations, got: {:?}",
        report.violations
    );
}

// ── dv_hdr10plus_bridge ───────────────────────────────────────────────────────

#[test]
fn test_dv_hdr10plus_bridge_smoke() {
    use oximedia_dolbyvision::dv_hdr10plus_bridge::DvToHdr10PlusBridge;
    let l1 = Level1Metadata {
        min_pq: 100,
        max_pq: 3000,
        avg_pq: 1500,
    };
    let bridge = DvToHdr10PlusBridge::default();
    let sei = bridge.convert(&l1, None);
    // Targeted system display must be plausible (>= 1 nit coded as 10000x nit).
    assert!(
        sei.targeted_system_display_maximum_luminance >= 1,
        "targeted luminance must be positive"
    );
}

// ── dv_xml_export ─────────────────────────────────────────────────────────────

#[test]
fn test_dv_xml_export_smoke() {
    use oximedia_dolbyvision::dv_xml_export::{
        DvL2Entry, DvShotEntry, DvXmlDocument, DvXmlExporter, DvXmlParser, DvXmlVersion,
    };
    let doc = DvXmlDocument {
        version: DvXmlVersion::V6_0_6,
        shots: vec![DvShotEntry {
            frame_start: 0,
            frame_end: 23,
            l1_min: 0.0,
            l1_mid: 0.1,
            l1_max: 0.58,
            l2_entries: vec![DvL2Entry::identity(2081)],
        }],
        frame_rate: (24, 1),
        total_frames: 24,
    };
    let xml = DvXmlExporter::to_xml(&doc);
    let parsed = DvXmlParser::from_xml(&xml).expect("XML round-trip should succeed");
    assert_eq!(parsed.shots.len(), 1, "should have 1 shot after round-trip");
}

// ── gamut_mapping ─────────────────────────────────────────────────────────────

#[test]
fn test_gamut_mapping_smoke() {
    use oximedia_dolbyvision::gamut_mapping::{ClipStrategy, GamutMapper, GamutSpace};
    let mapper = GamutMapper::new(GamutSpace::Bt2020, GamutSpace::Bt709, ClipStrategy::Clip);
    // Map a D65 white point: should be very close to [1, 1, 1] in BT.709.
    let out = mapper.mapping().convert([1.0f32, 1.0, 1.0]);
    assert!((out[0] - 1.0).abs() < 0.01, "white r={}", out[0]);
    assert!((out[1] - 1.0).abs() < 0.01, "white g={}", out[1]);
    assert!((out[2] - 1.0).abs() < 0.01, "white b={}", out[2]);
}

// ── hdr10plus_bridge ─────────────────────────────────────────────────────────

#[test]
fn test_hdr10plus_bridge_smoke() {
    use oximedia_dolbyvision::hdr10plus_bridge::dv_rpu_to_hdr10p;
    let mut rpu = DolbyVisionRpu::new(Profile::Profile8);
    rpu.level1 = Some(Level1Metadata {
        min_pq: 50,
        max_pq: 2500,
        avg_pq: 1000,
    });
    let meta = dv_rpu_to_hdr10p(&rpu);
    // Average MaxRGB must be positive.
    assert!(
        meta.average_max_rgb > 0.0,
        "average_max_rgb must be positive, got {}",
        meta.average_max_rgb
    );
}

// ── metadata_report ───────────────────────────────────────────────────────────

#[test]
fn test_metadata_report_smoke() {
    use oximedia_dolbyvision::metadata_report::MetadataReporter;
    let rpus = vec![
        {
            let mut r = DolbyVisionRpu::new(Profile::Profile8);
            r.level1 = Some(Level1Metadata {
                min_pq: 0,
                max_pq: 3000,
                avg_pq: 1200,
            });
            r
        },
        {
            let mut r = DolbyVisionRpu::new(Profile::Profile8);
            r.level1 = Some(Level1Metadata {
                min_pq: 10,
                max_pq: 2800,
                avg_pq: 1100,
            });
            r
        },
    ];
    let report = MetadataReporter::generate(&rpus);
    assert_eq!(report.total_frames, 2, "report must reflect 2 frames");
    // At least L1 coverage should be 1.0 since we set level1 on all frames.
    assert!(
        (report.level1_coverage - 1.0).abs() < f64::EPSILON,
        "L1 coverage must be 100%"
    );
}

// ── profile10 ─────────────────────────────────────────────────────────────────

#[test]
fn test_profile10_smoke() {
    use oximedia_dolbyvision::profile10::{
        Av1MetadataObuHeader, Profile10RpuContainer, AV1_METADATA_OBU_TYPE_DV,
    };
    let header = Av1MetadataObuHeader::default();
    assert_eq!(
        header.obu_type, AV1_METADATA_OBU_TYPE_DV,
        "default header must use DV metadata type"
    );
    // Build a container from a short payload.
    let payload = vec![0x19u8, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00];
    let container = Profile10RpuContainer::new(payload.clone());
    assert_eq!(
        container.rpu_payload_size as usize,
        payload.len(),
        "container must record payload size"
    );
}

// ── rpu_merge ────────────────────────────────────────────────────────────────

#[test]
fn test_rpu_merge_smoke() {
    use oximedia_dolbyvision::rpu_merge::{merge_rpus, MergeConfig, MergeStrategy};
    let primary = {
        let mut r = DolbyVisionRpu::new(Profile::Profile8);
        r.level1 = Some(Level1Metadata {
            min_pq: 10,
            max_pq: 3000,
            avg_pq: 1500,
        });
        r
    };
    let secondary = {
        let mut r = DolbyVisionRpu::new(Profile::Profile8);
        r.level1 = Some(Level1Metadata {
            min_pq: 5,
            max_pq: 3500,
            avg_pq: 1800,
        });
        r
    };
    let config = MergeConfig {
        level1_strategy: MergeStrategy::PreferHigherPeak,
        ..MergeConfig::default()
    };
    let merged = merge_rpus(&primary, &secondary, &config).expect("merge should succeed");
    // Higher-peak strategy: merged L1 max_pq should be 3500.
    let l1 = merged.level1.expect("merged RPU must have L1");
    assert_eq!(l1.max_pq, 3500, "merge should select higher peak L1");
}

// ── rpu_stats ────────────────────────────────────────────────────────────────

#[test]
fn test_rpu_stats_smoke() {
    use oximedia_dolbyvision::rpu_stats::RpuStatsBuilder;
    let mut builder = RpuStatsBuilder::new();
    for i in 0u16..5 {
        let l1 = Level1Metadata {
            min_pq: i * 10,
            max_pq: 2000 + i * 50,
            avg_pq: 1000 + i * 30,
        };
        builder.push_frame(i as u64, &l1, i == 0);
    }
    let report = builder.finish();
    assert_eq!(report.scenes.len(), 1, "should have 1 scene");
    // The scene's peak must equal the max of all max_pq values (2200).
    assert_eq!(report.scenes[0].max_pq, 2200, "scene max_pq must be 2200");
}

// ── rpu_statistics ───────────────────────────────────────────────────────────

#[test]
fn test_rpu_statistics_smoke() {
    use oximedia_dolbyvision::dv_xml_export::{DvL2Entry, DvShotEntry};
    use oximedia_dolbyvision::rpu_statistics::RpuStatistics;
    let shots = vec![
        DvShotEntry {
            frame_start: 0,
            frame_end: 23,
            l1_min: 0.001,
            l1_mid: 0.10,
            l1_max: 0.45,
            l2_entries: vec![DvL2Entry::identity(2081)],
        },
        DvShotEntry {
            frame_start: 24,
            frame_end: 47,
            l1_min: 0.002,
            l1_mid: 0.20,
            l1_max: 0.80,
            l2_entries: vec![DvL2Entry::identity(2081)],
        },
    ];
    let stats = RpuStatistics::from_shots(&shots);
    assert_eq!(stats.scene_stats.len(), 2, "should have 2 scenes");
    // Global max L1 should be 0.80.
    assert!(
        (stats.overall.global_max_pq - 0.80).abs() < 1e-5,
        "global_max_pq={:.5}",
        stats.overall.global_max_pq
    );
}

// ── rpu_timeline ─────────────────────────────────────────────────────────────

#[test]
fn test_rpu_timeline_smoke() {
    use oximedia_dolbyvision::rpu_timeline::{RpuTimeline, TimeBase, TimedRpu};
    let mut timeline = RpuTimeline::new(TimeBase::new(1, 90000));
    for pts in [0u64, 3000, 6000] {
        let rpu = DolbyVisionRpu::new(Profile::Profile8);
        timeline.insert(TimedRpu::new(pts, rpu));
    }
    assert_eq!(timeline.len(), 3, "timeline must contain 3 entries");
    // After removing the middle entry, length drops to 2.
    let removed = timeline.delete_at_pts(3000);
    assert!(removed, "delete_at_pts must return true for known PTS");
    assert_eq!(
        timeline.len(),
        2,
        "timeline must have 2 entries after removal"
    );
}

// ── rpu_validate ─────────────────────────────────────────────────────────────

#[test]
fn test_rpu_validate_smoke() {
    use oximedia_dolbyvision::rpu_validate::RpuValidator;
    // Too-short byte slice must produce at least one error.
    let short = [0x19u8, 0x01];
    let errors = RpuValidator::validate(&short);
    assert!(!errors.is_empty(), "short RPU must fail validation");
    // Empty slice must also fail.
    let errors_empty = RpuValidator::validate(&[]);
    assert!(!errors_empty.is_empty(), "empty RPU must fail validation");
}

// ── rpu_validator ─────────────────────────────────────────────────────────────

#[test]
fn test_rpu_validator_smoke() {
    use oximedia_dolbyvision::dv_xml_export::{DvL2Entry, DvShotEntry};
    use oximedia_dolbyvision::rpu_validator::{RpuValidationLevel, RpuValidator};
    let validator = RpuValidator::new(RpuValidationLevel::Basic);
    let shot = DvShotEntry {
        frame_start: 0,
        frame_end: 11,
        l1_min: 0.001,
        l1_mid: 0.10,
        l1_max: 0.58,
        l2_entries: vec![DvL2Entry::identity(2081)],
    };
    let result = validator.validate_shot(&shot, 0);
    // A valid shot under Basic validation must have no issues.
    assert!(
        result.is_empty(),
        "expected no issues for valid shot, got: {result:?}"
    );
}

// ── rpu_visualize ─────────────────────────────────────────────────────────────

#[test]
fn test_rpu_visualize_smoke() {
    use oximedia_dolbyvision::rpu_visualize::{PlotConfig, RpuPlotter};
    let rpus: Vec<DolbyVisionRpu> = (0u16..8)
        .map(|i| {
            let mut r = DolbyVisionRpu::new(Profile::Profile8);
            r.level1 = Some(Level1Metadata {
                min_pq: 100,
                max_pq: 2000 + i * 50,
                avg_pq: 1000 + i * 20,
            });
            r
        })
        .collect();
    let config = PlotConfig::default();
    let plot = RpuPlotter::plot_l1_luminance(&rpus, &config);
    // Plot output must be non-empty.
    assert!(!plot.is_empty(), "plot output must not be empty");
}

// ── scene_stats ───────────────────────────────────────────────────────────────

#[test]
fn test_scene_stats_smoke() {
    use oximedia_dolbyvision::scene_stats::DvSceneStats;
    let mut stats = DvSceneStats::default();
    stats.add_frame(100, 500, 3000);
    stats.add_frame(80, 600, 2800);
    let (scene_min, _scene_avg_mid, scene_max) = stats.scene_summary();
    assert_eq!(scene_max, 3000, "scene_max must be 3000");
    assert_eq!(scene_min, 80, "scene_min must be 80");
}

// ── scene_trim_enhanced ───────────────────────────────────────────────────────

#[test]
fn test_scene_trim_enhanced_smoke() {
    use oximedia_dolbyvision::scene_trim_enhanced::{
        L1Sample, SceneChangeDetector, SceneChangeDetectorConfig,
    };
    let config = SceneChangeDetectorConfig::default();
    let detector = SceneChangeDetector::new(config);
    // Build a sequence with an obvious luminance jump at frame 5.
    let mut samples: Vec<L1Sample> = (0u64..5)
        .map(|i| L1Sample {
            frame_index: i,
            min_pq: 1000,
            max_pq: 1200,
            avg_pq: 2000,
        })
        .collect();
    samples.extend((5u64..10).map(|i| L1Sample {
        frame_index: i,
        min_pq: 100,
        max_pq: 200,
        avg_pq: 500,
    }));
    let result = detector.detect(&samples);
    // Must detect at least one boundary at the luminance jump.
    assert!(
        !result.boundaries.is_empty(),
        "should detect at least one scene boundary"
    );
}

// ── streaming_parser ─────────────────────────────────────────────────────────

#[test]
fn test_streaming_parser_smoke() {
    use oximedia_dolbyvision::streaming_parser::{parse_nal_batch, StreamingRpuParser};
    // A fresh parser should accumulate zero RPUs with zero errors initially.
    let parser = StreamingRpuParser::new();
    assert_eq!(parser.rpu_count(), 0, "new parser should have rpu_count=0");
    assert_eq!(
        parser.error_count(),
        0,
        "new parser should have error_count=0"
    );
    // Parsing empty batch must not panic.
    let (rpus, errors) = parse_nal_batch(&[]);
    assert!(rpus.is_empty(), "no RPUs from empty batch");
    assert!(errors.is_empty(), "no errors from empty batch");
}

// ── tone_map_preview ─────────────────────────────────────────────────────────

#[test]
fn test_tone_map_preview_smoke() {
    use oximedia_dolbyvision::tone_map_preview::{ToneMapConfig, ToneMapTarget, ToneMapper};
    let config = ToneMapConfig {
        target: ToneMapTarget::Sdr(100.0),
        ..ToneMapConfig::default()
    };
    // Tone-map a mid-gray pixel using L1 max_pq = 0.5 (normalised PQ).
    let (r, g, b) = ToneMapper::map_pixel(0.5, 0.5, 0.5, 0.5, &config);
    // Output must be in [0, 1] for SDR target.
    assert!(r >= 0.0 && r <= 1.0, "r={r} out of range");
    assert!(g >= 0.0 && g <= 1.0, "g={g} out of range");
    assert!(b >= 0.0 && b <= 1.0, "b={b} out of range");
}

// ── trim_meta ─────────────────────────────────────────────────────────────────

#[test]
fn test_trim_meta_smoke() {
    use oximedia_dolbyvision::trim_meta::TrimMetadata;
    let meta = TrimMetadata::new(2).expect("level 2 is valid");
    assert_eq!(meta.level, 2, "level must be 2");
    // Level 3 must also be constructable.
    let meta3 = TrimMetadata::new(3).expect("level 3 is valid");
    assert_eq!(meta3.level, 3, "level must be 3");
    // Level 99 must return an error.
    assert!(TrimMetadata::new(99).is_err(), "level 99 must be invalid");
}

// ── xml_manifest ─────────────────────────────────────────────────────────────

#[test]
fn test_xml_manifest_smoke() {
    use oximedia_dolbyvision::xml_manifest::DvManifestBuilder;
    let mut builder = DvManifestBuilder::new()
        .with_title("Test Title")
        .with_frame_rate(24, 1);
    builder.add_shot(0, 23, 2081.0);
    builder.add_shot(24, 47, 3079.0);
    let xml = builder.build_xml();
    // Must contain the title.
    assert!(xml.contains("Test Title"), "manifest must contain title");
    // Must contain at least one shot marker.
    assert!(
        xml.contains("start=\"0\"") || xml.contains("<Shot"),
        "manifest must reference shots"
    );
}
