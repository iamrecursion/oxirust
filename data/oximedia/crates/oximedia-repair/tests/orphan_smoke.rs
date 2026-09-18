//! Smoke tests for newly-wired orphan modules in oximedia-repair.

#[test]
fn test_audio_gap_fill_silence() {
    use oximedia_repair::audio_gap::AudioGapFiller;
    let mut samples = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    AudioGapFiller::fill_silence(&mut samples, 1, 2);
    assert_eq!(samples[1], 0.0);
    assert_eq!(samples[2], 0.0);
    assert_eq!(samples[0], 1.0); // unchanged
}

#[test]
fn test_corruption_heatmap_construct() {
    use oximedia_repair::corruption_heatmap::HeatMapConfig;
    let config = HeatMapConfig::new(1_000_000, 100);
    assert_eq!(config.bucket_count, 100);
}

#[test]
fn test_diagnostic_finding_new() {
    use oximedia_repair::diagnostic::{DiagnosticFinding, DiagnosticSeverity};
    let finding = DiagnosticFinding::new(
        "E001",
        DiagnosticSeverity::Warning,
        "offset mismatch",
        "try repair",
    );
    assert_eq!(finding.severity, DiagnosticSeverity::Warning);
}

#[test]
fn test_frame_sub_strategy() {
    use oximedia_repair::frame_sub::FrameSubstitutor;
    let sub = FrameSubstitutor::new();
    let _ = sub;
}

#[test]
fn test_header_recovery_scan_empty() {
    use oximedia_repair::header_recovery::HeaderRecovery;
    let result = HeaderRecovery::scan_for_sync(&[], b"\x00\x00\x01");
    assert!(result.is_none());
}

#[test]
fn test_progressive_repair_step() {
    use oximedia_repair::progressive_repair::RepairStep;
    let _ = std::mem::size_of::<RepairStep>();
}

#[test]
fn test_recovery_plan_step() {
    use oximedia_repair::recovery_plan::RepairStep;
    let _ = std::mem::size_of::<RepairStep>();
}

#[test]
fn test_repair_diff_byte_summary() {
    use oximedia_repair::repair_diff::ByteDiffSummary;
    let _ = std::mem::size_of::<ByteDiffSummary>();
}

#[test]
fn test_segment_salvage_map() {
    use oximedia_repair::segment_salvage::SalvageMap;
    let map = SalvageMap::new(1_000_000);
    assert_eq!(map.segment_count(), 0);
}
