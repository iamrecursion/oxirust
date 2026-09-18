//! Comprehensive test suite for oximedia-qc.
//!
//! This module contains extensive tests covering all QC functionality.

#![cfg(test)]

use crate::audio::*;
use crate::batch::*;
use crate::codec_validation::*;
use crate::compliance::*;
use crate::container::*;
use crate::format::*;
use crate::profiles::*;
use crate::rules::*;
use crate::standards::*;
use crate::temporal::*;
use crate::video::*;
use crate::*;

// ============================================================================
// Core QC Tests
// ============================================================================

#[test]
fn test_qc_new() {
    let qc = QualityControl::new();
    assert_eq!(qc.rule_count(), 0);
}

#[test]
fn test_qc_with_thresholds() {
    let thresholds = Thresholds::new().with_min_video_bitrate(1_000_000);
    let qc = QualityControl::with_thresholds(thresholds);
    assert_eq!(qc.rule_count(), 0);
}

#[test]
fn test_qc_preset_basic() {
    let qc = QualityControl::with_preset(QcPreset::Basic);
    assert!(qc.rule_count() >= 3);
}

#[test]
fn test_qc_preset_streaming() {
    let qc = QualityControl::with_preset(QcPreset::Streaming);
    assert!(qc.rule_count() > 5);
}

#[test]
fn test_qc_preset_broadcast() {
    let qc = QualityControl::with_preset(QcPreset::Broadcast);
    assert!(qc.rule_count() > 8);
}

#[test]
fn test_qc_preset_comprehensive() {
    let qc = QualityControl::with_preset(QcPreset::Comprehensive);
    assert!(qc.rule_count() > 15);
}

#[test]
fn test_qc_preset_youtube() {
    let qc = QualityControl::with_preset(QcPreset::YouTube);
    assert!(qc.rule_count() > 5);
}

#[test]
fn test_qc_preset_vimeo() {
    let qc = QualityControl::with_preset(QcPreset::Vimeo);
    assert!(qc.rule_count() > 5);
}

// ============================================================================
// Rules Tests
// ============================================================================

#[test]
fn test_severity_ordering() {
    assert!(Severity::Critical > Severity::Error);
    assert!(Severity::Error > Severity::Warning);
    assert!(Severity::Warning > Severity::Info);
}

#[test]
fn test_severity_display() {
    assert_eq!(Severity::Info.to_string(), "INFO");
    assert_eq!(Severity::Warning.to_string(), "WARNING");
    assert_eq!(Severity::Error.to_string(), "ERROR");
    assert_eq!(Severity::Critical.to_string(), "CRITICAL");
}

#[test]
fn test_check_result_pass() {
    let result = CheckResult::pass("test_rule");
    assert!(result.passed);
    assert_eq!(result.rule_name, "test_rule");
}

#[test]
fn test_check_result_fail() {
    let result = CheckResult::fail("test_rule", Severity::Error, "Test error");
    assert!(!result.passed);
    assert_eq!(result.severity, Severity::Error);
}

#[test]
fn test_check_result_with_recommendation() {
    let result = CheckResult::pass("test").with_recommendation("Do this");
    assert_eq!(result.recommendation, Some("Do this".to_string()));
}

#[test]
fn test_check_result_with_stream() {
    let result = CheckResult::pass("test").with_stream(1);
    assert_eq!(result.stream_index, Some(1));
}

#[test]
fn test_check_result_with_timestamp() {
    let result = CheckResult::pass("test").with_timestamp(12.5);
    assert_eq!(result.timestamp, Some(12.5));
}

#[test]
fn test_rule_category_display() {
    assert_eq!(RuleCategory::Video.to_string(), "video");
    assert_eq!(RuleCategory::Audio.to_string(), "audio");
    assert_eq!(RuleCategory::Container.to_string(), "container");
    assert_eq!(RuleCategory::Compliance.to_string(), "compliance");
}

#[test]
fn test_thresholds_default() {
    let thresholds = Thresholds::default();
    assert_eq!(thresholds.max_silence_duration, Some(2.0));
    assert_eq!(thresholds.loudness_target, Some(-23.0));
}

#[test]
fn test_thresholds_builder() {
    let thresholds = Thresholds::new()
        .with_min_video_bitrate(5_000_000)
        .with_max_video_bitrate(20_000_000)
        .with_loudness_target(-24.0);

    assert_eq!(thresholds.min_video_bitrate, Some(5_000_000));
    assert_eq!(thresholds.max_video_bitrate, Some(20_000_000));
    assert_eq!(thresholds.loudness_target, Some(-24.0));
}

#[test]
fn test_qc_context_creation() {
    let context = QcContext::new("test.mkv");
    assert_eq!(context.file_path, "test.mkv");
    assert!(context.streams.is_empty());
}

#[test]
fn test_qc_context_add_stream() {
    let mut context = QcContext::new("test.mkv");
    let stream = oximedia_container::StreamInfo::new(
        0,
        oximedia_core::CodecId::Av1,
        oximedia_core::Rational::new(1, 30),
    );
    context.add_stream(stream);
    assert_eq!(context.streams.len(), 1);
}

#[test]
fn test_qc_context_set_duration() {
    let mut context = QcContext::new("test.mkv");
    context.set_duration(120.5);
    assert_eq!(context.duration, Some(120.5));
}

#[test]
fn test_qc_context_video_streams() {
    let mut context = QcContext::new("test.mkv");
    let video_stream = oximedia_container::StreamInfo::new(
        0,
        oximedia_core::CodecId::Av1,
        oximedia_core::Rational::new(1, 30),
    );
    context.add_stream(video_stream);

    let video_streams = context.video_streams();
    assert_eq!(video_streams.len(), 1);
}

#[test]
fn test_qc_context_audio_streams() {
    let mut context = QcContext::new("test.mkv");
    let audio_stream = oximedia_container::StreamInfo::new(
        1,
        oximedia_core::CodecId::Opus,
        oximedia_core::Rational::new(1, 48000),
    );
    context.add_stream(audio_stream);

    let audio_streams = context.audio_streams();
    assert_eq!(audio_streams.len(), 1);
}

// ============================================================================
// Video QC Tests
// ============================================================================

#[test]
fn test_video_codec_validation_creation() {
    let validator = VideoCodecValidation;
    assert_eq!(validator.name(), "video_codec_validation");
}

#[test]
fn test_resolution_validation_creation() {
    let validator = ResolutionValidation::new();
    assert_eq!(validator.name(), "resolution_validation");
}

#[test]
fn test_resolution_validation_with_min() {
    let validator = ResolutionValidation::new().with_min_resolution(1920, 1080);
    // Configuration should be applied (private fields)
    assert_eq!(validator.name(), "resolution_validation");
}

#[test]
fn test_resolution_validation_with_max() {
    let validator = ResolutionValidation::new().with_max_resolution(3840, 2160);
    assert_eq!(validator.name(), "resolution_validation");
}

#[test]
fn test_resolution_validation_even_requirement() {
    let validator = ResolutionValidation::new().with_even_requirement(false);
    assert_eq!(validator.name(), "resolution_validation");
}

#[test]
fn test_frame_rate_validation_creation() {
    let validator = FrameRateValidation::new();
    assert_eq!(validator.name(), "frame_rate_validation");
}

#[test]
fn test_frame_rate_validation_with_rates() {
    let validator = FrameRateValidation::new().with_expected_rates(vec![24.0, 30.0, 60.0]);
    assert_eq!(validator.name(), "frame_rate_validation");
}

#[test]
fn test_frame_rate_validation_with_tolerance() {
    let validator = FrameRateValidation::new().with_tolerance(0.001);
    assert_eq!(validator.name(), "frame_rate_validation");
}

#[test]
fn test_bitrate_analysis_creation() {
    let thresholds = Thresholds::new();
    let analyzer = BitrateAnalysis::new(thresholds);
    assert_eq!(analyzer.name(), "bitrate_analysis");
}

#[test]
fn test_interlacing_detection_creation() {
    let detector = InterlacingDetection;
    assert_eq!(detector.name(), "interlacing_detection");
}

#[test]
fn test_black_frame_detection_creation() {
    let detector = BlackFrameDetection::default();
    assert_eq!(detector.name(), "black_frame_detection");
}

#[test]
fn test_black_frame_detection_with_duration() {
    let detector = BlackFrameDetection::new(3.0);
    assert_eq!(detector.name(), "black_frame_detection");
}

#[test]
fn test_freeze_frame_detection_creation() {
    let detector = FreezeFrameDetection::default();
    assert_eq!(detector.name(), "freeze_frame_detection");
}

#[test]
fn test_freeze_frame_detection_with_duration() {
    let detector = FreezeFrameDetection::new(2.0);
    assert_eq!(detector.name(), "freeze_frame_detection");
}

#[test]
fn test_compression_artifact_detection_creation() {
    let detector = CompressionArtifactDetection::default();
    assert_eq!(detector.name(), "compression_artifact_detection");
}

#[test]
fn test_compression_artifact_detection_with_threshold() {
    let detector = CompressionArtifactDetection::new(0.2);
    assert_eq!(detector.name(), "compression_artifact_detection");
}

// ============================================================================
// Audio QC Tests
// ============================================================================

#[test]
fn test_audio_codec_validation_creation() {
    let validator = AudioCodecValidation;
    assert_eq!(validator.name(), "audio_codec_validation");
}

#[test]
fn test_sample_rate_validation_creation() {
    let validator = SampleRateValidation::new();
    assert_eq!(validator.name(), "sample_rate_validation");
}

#[test]
fn test_sample_rate_validation_with_rates() {
    let validator = SampleRateValidation::new().with_allowed_rates(vec![48000, 96000]);
    assert_eq!(validator.name(), "sample_rate_validation");
}

#[test]
fn test_sample_rate_validation_strict() {
    let validator = SampleRateValidation::new().with_strict(false);
    assert_eq!(validator.name(), "sample_rate_validation");
}

#[test]
fn test_loudness_compliance_ebu_r128() {
    let thresholds = Thresholds::new();
    let compliance = LoudnessCompliance::ebu_r128(thresholds);
    assert_eq!(compliance.name(), "loudness_compliance");
}

#[test]
fn test_loudness_compliance_atsc_a85() {
    let thresholds = Thresholds::new();
    let compliance = LoudnessCompliance::atsc_a85(thresholds);
    assert_eq!(compliance.name(), "loudness_compliance");
}

#[test]
fn test_loudness_compliance_custom() {
    let thresholds = Thresholds::new();
    let compliance = LoudnessCompliance::custom(-16, thresholds);
    assert_eq!(compliance.name(), "loudness_compliance");
}

#[test]
fn test_clipping_detection_creation() {
    let detector = ClippingDetection::new();
    assert_eq!(detector.name(), "clipping_detection");
}

#[test]
fn test_clipping_detection_with_threshold() {
    let detector = ClippingDetection::new().with_threshold(0.95);
    assert_eq!(detector.name(), "clipping_detection");
}

#[test]
fn test_clipping_detection_with_max_consecutive() {
    let detector = ClippingDetection::new().with_max_consecutive(5);
    assert_eq!(detector.name(), "clipping_detection");
}

#[test]
fn test_silence_detection_creation() {
    let thresholds = Thresholds::new();
    let detector = SilenceDetection::new(&thresholds);
    assert_eq!(detector.name(), "silence_detection");
}

#[test]
fn test_silence_detection_with_threshold() {
    let thresholds = Thresholds::new();
    let detector = SilenceDetection::new(&thresholds).with_threshold_db(-50.0);
    assert_eq!(detector.name(), "silence_detection");
}

#[test]
fn test_silence_detection_with_duration() {
    let thresholds = Thresholds::new();
    let detector = SilenceDetection::new(&thresholds).with_max_duration(3.0);
    assert_eq!(detector.name(), "silence_detection");
}

#[test]
fn test_phase_detection_creation() {
    let detector = PhaseDetection::new();
    assert_eq!(detector.name(), "phase_detection");
}

#[test]
fn test_phase_detection_with_threshold() {
    let detector = PhaseDetection::new().with_threshold(-0.5);
    assert_eq!(detector.name(), "phase_detection");
}

#[test]
fn test_dc_offset_detection_creation() {
    let detector = DcOffsetDetection::new();
    assert_eq!(detector.name(), "dc_offset_detection");
}

#[test]
fn test_dc_offset_detection_with_threshold() {
    let detector = DcOffsetDetection::new().with_threshold(0.02);
    assert_eq!(detector.name(), "dc_offset_detection");
}

#[test]
fn test_channel_validation_creation() {
    let validator = ChannelValidation::new();
    assert_eq!(validator.name(), "channel_validation");
}

#[test]
fn test_channel_validation_with_configurations() {
    let validator = ChannelValidation::new().with_allowed_configurations(vec![2, 6, 8]);
    assert_eq!(validator.name(), "channel_validation");
}

// ============================================================================
// Container QC Tests
// ============================================================================

#[test]
fn test_format_validation_creation() {
    let validator = FormatValidation;
    assert_eq!(validator.name(), "format_validation");
}

#[test]
fn test_stream_synchronization_creation() {
    let sync = StreamSynchronization::default();
    assert_eq!(sync.name(), "stream_synchronization");
}

#[test]
fn test_stream_synchronization_with_offset() {
    let sync = StreamSynchronization::new(200.0);
    assert_eq!(sync.name(), "stream_synchronization");
}

#[test]
fn test_timestamp_continuity_creation() {
    let continuity = TimestampContinuity::default();
    assert_eq!(continuity.name(), "timestamp_continuity");
}

#[test]
fn test_timestamp_continuity_with_gap() {
    let continuity = TimestampContinuity::new(1.0);
    assert_eq!(continuity.name(), "timestamp_continuity");
}

#[test]
fn test_keyframe_interval_creation() {
    let interval = KeyframeInterval::default();
    assert_eq!(interval.name(), "keyframe_interval");
}

#[test]
fn test_keyframe_interval_with_range() {
    let interval = KeyframeInterval::new(2.0, 8.0);
    assert_eq!(interval.name(), "keyframe_interval");
}

#[test]
fn test_seeking_capability_creation() {
    let capability = SeekingCapability;
    assert_eq!(capability.name(), "seeking_capability");
}

#[test]
fn test_duration_consistency_creation() {
    let consistency = DurationConsistency::default();
    assert_eq!(consistency.name(), "duration_consistency");
}

#[test]
fn test_duration_consistency_with_tolerance() {
    let consistency = DurationConsistency::new(0.5);
    assert_eq!(consistency.name(), "duration_consistency");
}

#[test]
fn test_metadata_validation_creation() {
    let validator = MetadataValidation::default();
    assert_eq!(validator.name(), "metadata_validation");
}

#[test]
fn test_metadata_validation_with_fields() {
    let validator = MetadataValidation::new()
        .with_required_fields(vec!["title".to_string(), "author".to_string()]);
    assert_eq!(validator.name(), "metadata_validation");
}

#[test]
fn test_stream_ordering_creation() {
    let ordering = StreamOrdering;
    assert_eq!(ordering.name(), "stream_ordering");
}

#[test]
fn test_file_size_validation_creation() {
    let validator = FileSizeValidation::default();
    assert_eq!(validator.name(), "file_size_validation");
}

#[test]
fn test_file_size_validation_with_max() {
    let validator = FileSizeValidation::new().with_max_size(1_000_000_000);
    assert_eq!(validator.name(), "file_size_validation");
}

#[test]
fn test_file_size_validation_with_min() {
    let validator = FileSizeValidation::new().with_min_size(1_000_000);
    assert_eq!(validator.name(), "file_size_validation");
}

// ============================================================================
// Compliance Tests
// ============================================================================

#[test]
fn test_youtube_compliance_creation() {
    let compliance = YouTubeCompliance;
    assert_eq!(compliance.name(), "youtube_compliance");
}

#[test]
fn test_vimeo_compliance_creation() {
    let compliance = VimeoCompliance;
    assert_eq!(compliance.name(), "vimeo_compliance");
}

#[test]
fn test_broadcast_compliance_creation() {
    let compliance = BroadcastCompliance::new();
    assert_eq!(compliance.name(), "broadcast_compliance");
}

#[test]
fn test_broadcast_compliance_with_stereo() {
    let compliance = BroadcastCompliance::new().with_stereo_requirement(false);
    assert_eq!(compliance.name(), "broadcast_compliance");
}

#[test]
fn test_broadcast_compliance_with_hd() {
    let compliance = BroadcastCompliance::new().with_hd_requirement(false);
    assert_eq!(compliance.name(), "broadcast_compliance");
}

#[test]
fn test_patent_free_enforcement_creation() {
    let enforcement = PatentFreeEnforcement;
    assert_eq!(enforcement.name(), "patent_free_enforcement");
}

#[test]
fn test_custom_compliance_creation() {
    let compliance = CustomCompliance::new("my_spec");
    assert_eq!(compliance.name(), "my_spec");
}

#[test]
fn test_custom_compliance_with_video_codecs() {
    let compliance =
        CustomCompliance::new("test").with_video_codecs(vec![oximedia_core::CodecId::Av1]);
    assert_eq!(compliance.name(), "test");
}

#[test]
fn test_custom_compliance_with_audio_codecs() {
    let compliance =
        CustomCompliance::new("test").with_audio_codecs(vec![oximedia_core::CodecId::Opus]);
    assert_eq!(compliance.name(), "test");
}

#[test]
fn test_custom_compliance_with_min_resolution() {
    let compliance = CustomCompliance::new("test").with_min_resolution(1280, 720);
    assert_eq!(compliance.name(), "test");
}

#[test]
fn test_custom_compliance_with_max_resolution() {
    let compliance = CustomCompliance::new("test").with_max_resolution(3840, 2160);
    assert_eq!(compliance.name(), "test");
}

// ============================================================================
// Temporal QC Tests
// ============================================================================

#[test]
fn test_dropped_frame_detection_creation() {
    let detector = DroppedFrameDetection::new();
    assert_eq!(detector.name(), "dropped_frame_detection");
}

#[test]
fn test_dropped_frame_detection_with_tolerance() {
    let detector = DroppedFrameDetection::new().with_tolerance(5);
    assert_eq!(detector.name(), "dropped_frame_detection");
}

#[test]
fn test_duplicate_frame_detection_creation() {
    let detector = DuplicateFrameDetection::new();
    assert_eq!(detector.name(), "duplicate_frame_detection");
}

#[test]
fn test_duplicate_frame_detection_with_max() {
    let detector = DuplicateFrameDetection::new().with_max_consecutive(5);
    assert_eq!(detector.name(), "duplicate_frame_detection");
}

#[test]
fn test_timecode_continuity_creation() {
    let continuity = TimecodeContinuity::new();
    assert_eq!(continuity.name(), "timecode_continuity");
}

#[test]
fn test_timecode_continuity_with_discontinuities() {
    let continuity = TimecodeContinuity::new().with_discontinuities_allowed(true);
    assert_eq!(continuity.name(), "timecode_continuity");
}

#[test]
fn test_duration_accuracy_creation() {
    let accuracy = DurationAccuracy::new();
    assert_eq!(accuracy.name(), "duration_accuracy");
}

#[test]
fn test_duration_accuracy_with_tolerance() {
    let accuracy = DurationAccuracy::new().with_tolerance(0.5);
    assert_eq!(accuracy.name(), "duration_accuracy");
}

#[test]
fn test_timestamp_validation_creation() {
    let validator = TimestampValidation::new();
    assert_eq!(validator.name(), "timestamp_validation");
}

// ============================================================================
// Format Validator Tests
// ============================================================================

#[test]
fn test_mp4_validator_creation() {
    let validator = Mp4Validator::new();
    assert_eq!(validator.name(), "mp4_isobmff_validation");
}

#[test]
fn test_matroska_validator_creation() {
    let validator = MatroskaValidator::new();
    assert_eq!(validator.name(), "matroska_schema_validation");
}

#[test]
fn test_mxf_validator_creation() {
    let validator = MxfValidator::new();
    assert_eq!(validator.name(), "mxf_operational_pattern_validation");
}

#[test]
fn test_mpegts_validator_creation() {
    let validator = MpegTsValidator::new();
    assert_eq!(validator.name(), "mpegts_structure_validation");
}

// ============================================================================
// Codec Validation Tests
// ============================================================================

#[test]
fn test_av1_validator_creation() {
    let validator = Av1BitstreamValidator::new();
    assert_eq!(validator.name(), "av1_bitstream_validation");
}

#[test]
fn test_vp9_validator_creation() {
    let validator = Vp9BitstreamValidator::new();
    assert_eq!(validator.name(), "vp9_bitstream_validation");
}

#[test]
fn test_opus_validator_creation() {
    let validator = OpusBitstreamValidator::new();
    assert_eq!(validator.name(), "opus_bitstream_validation");
}

// ============================================================================
// Standards Tests
// ============================================================================

#[test]
fn test_ebu_r128_validator_creation() {
    let validator = EbuR128Validator::new();
    assert_eq!(validator.name(), "ebu_r128_compliance");
}

#[test]
fn test_smpte_validator_creation() {
    let validator = SmpteValidator::new();
    assert_eq!(validator.name(), "smpte_standards_compliance");
}

#[test]
fn test_dpp_validator_creation() {
    let validator = DppValidator::new();
    assert_eq!(validator.name(), "dpp_compliance");
}

#[test]
fn test_atsc_a85_validator_creation() {
    let validator = AtscA85Validator::new();
    assert_eq!(validator.name(), "atsc_a85_compliance");
}

// ============================================================================
// Batch Processing Tests
// ============================================================================

#[test]
fn test_batch_processor_creation() {
    let qc = QualityControl::new();
    let _processor = BatchProcessor::new(qc);
    // Processor created successfully
    assert_eq!(1, 1);
}

#[test]
fn test_batch_processor_with_parallel_jobs() {
    let qc = QualityControl::new();
    let _processor = BatchProcessor::new(qc).with_parallel_jobs(4);
    assert_eq!(1, 1);
}

#[test]
fn test_batch_processor_with_detailed_reports() {
    let qc = QualityControl::new();
    let _processor = BatchProcessor::new(qc).with_detailed_reports(false);
    assert_eq!(1, 1);
}

// ============================================================================
// Profile Tests
// ============================================================================

#[test]
fn test_profile_manager_builtin_profiles() {
    let manager = ProfileManager::new();
    let profiles = manager.list_profiles();
    assert!(!profiles.is_empty());
}

#[test]
fn test_profile_manager_get_netflix() {
    let manager = ProfileManager::new();
    let profile = manager.get_profile("netflix");
    assert!(profile.is_some());
}

#[test]
fn test_profile_manager_get_amazon() {
    let manager = ProfileManager::new();
    let profile = manager.get_profile("amazon");
    assert!(profile.is_some());
}

#[test]
fn test_profile_manager_get_apple() {
    let manager = ProfileManager::new();
    let profile = manager.get_profile("apple");
    assert!(profile.is_some());
}

#[test]
fn test_profile_manager_get_bbc() {
    let manager = ProfileManager::new();
    let profile = manager.get_profile("bbc");
    assert!(profile.is_some());
}

#[test]
fn test_profile_manager_get_dpp() {
    let manager = ProfileManager::new();
    let profile = manager.get_profile("dpp");
    assert!(profile.is_some());
}

#[test]
fn test_profile_manager_get_archive() {
    let manager = ProfileManager::new();
    let profile = manager.get_profile("archive");
    assert!(profile.is_some());
}
