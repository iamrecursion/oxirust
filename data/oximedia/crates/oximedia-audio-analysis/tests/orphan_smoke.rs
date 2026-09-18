//! Smoke tests for newly-wired orphan modules in oximedia-audio-analysis.

#[test]
fn test_cqt_config_default() {
    use oximedia_audio_analysis::cqt::CqtConfig;
    let config = CqtConfig::default();
    assert!(config.bins_per_octave > 0);
}

#[test]
fn test_event_detection_detected_event() {
    let _ = std::any::type_name::<oximedia_audio_analysis::event_detection::DetectedEvent>();
}

#[test]
fn test_quality_degradation_result_fields() {
    let _ = std::any::type_name::<
        oximedia_audio_analysis::quality_degradation::QualityDegradationResult,
    >();
    // Verify the type can be referenced without panic
}

#[test]
fn test_segmentation_audio_segment() {
    let _ = std::any::type_name::<oximedia_audio_analysis::segmentation::AudioSegment>();
}

#[test]
fn test_singing_detection_result_fields() {
    use oximedia_audio_analysis::singing::SingingDetectionResult;
    let _ = std::mem::size_of::<SingingDetectionResult>();
}
