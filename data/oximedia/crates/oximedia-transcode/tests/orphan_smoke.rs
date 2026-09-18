//! Smoke tests for the 24 newly-registered orphan modules.
//!
//! Each test verifies basic instantiation and a trivial invariant — proving
//! the module is correctly wired into the crate's module tree.

// ── burn_in_spec ─────────────────────────────────────────────────────────────
#[test]
fn burn_in_spec_safe_area_uniform() {
    use oximedia_transcode::burn_in_spec::SafeArea;
    let sa = SafeArea::uniform(0.05);
    assert!((sa.left - 0.05).abs() < 1e-6);
    assert!((sa.right - 0.05).abs() < 1e-6);
    assert!((sa.top - 0.05).abs() < 1e-6);
    assert!((sa.bottom - 0.05).abs() < 1e-6);
}

// ── chapter_transcode ────────────────────────────────────────────────────────
#[test]
fn chapter_transcode_chapter_info_new() {
    use oximedia_transcode::chapter_transcode::ChapterInfo;
    let ch = ChapterInfo::new("Intro", 0);
    assert_eq!(ch.title, "Intro");
    assert_eq!(ch.start_ms, 0);
    assert!(ch.end_ms.is_none());
}

// ── chroma_selector ──────────────────────────────────────────────────────────
#[test]
fn chroma_selector_subsampling_notation() {
    use oximedia_transcode::chroma_selector::ChromaSubsampling;
    assert_eq!(ChromaSubsampling::Yuv420.notation(), "4:2:0");
    assert_eq!(ChromaSubsampling::Yuv422.notation(), "4:2:2");
    assert_eq!(ChromaSubsampling::Yuv444.notation(), "4:4:4");
}

// ── cmaf ─────────────────────────────────────────────────────────────────────
#[test]
fn cmaf_config_defaults() {
    use oximedia_transcode::cmaf::CmafConfig;
    let cfg = CmafConfig::video_default();
    assert_eq!(cfg.timescale, 90_000);
    assert_eq!(cfg.segment_duration_ms, 2_000);

    let acfg = CmafConfig::audio_default();
    assert_eq!(acfg.timescale, 44_100);
}

// ── codec_compat ─────────────────────────────────────────────────────────────
#[test]
fn codec_compat_matrix_new() {
    use oximedia_transcode::codec_compat::CompatMatrix;
    let matrix = CompatMatrix::new();
    // The matrix is built; spot-check that vp9 is known
    let vp9 = oximedia_transcode::codec_compat::VideoCodec::from_name("vp9");
    assert!(vp9.is_some());
    let _ = matrix; // alive
}

// ── codec_negotiation ────────────────────────────────────────────────────────
#[test]
fn codec_negotiation_media_class_default() {
    use oximedia_transcode::codec_negotiation::MediaClass;
    assert_eq!(MediaClass::default(), MediaClass::Video);
}

// ── downmix_table ────────────────────────────────────────────────────────────
#[test]
fn downmix_table_layout_channel_count() {
    use oximedia_transcode::downmix_table::DownmixLayout;
    assert_eq!(DownmixLayout::Mono.channel_count(), 1);
    assert_eq!(DownmixLayout::Stereo.channel_count(), 2);
    assert_eq!(DownmixLayout::FivePointOne.channel_count(), 6);
    assert_eq!(DownmixLayout::SevenPointOne.channel_count(), 8);
}

// ── encode_ladder_validator ──────────────────────────────────────────────────
#[test]
fn encode_ladder_validator_rung_new() {
    use oximedia_transcode::encode_ladder_validator::{
        EncodeLadder, LadderRung, LadderSpec, LadderValidator,
    };
    let ladder = EncodeLadder::new(vec![
        LadderRung::new(1920, 1080, 4_500_000, 30.0, "vp9"),
        LadderRung::new(1280, 720, 2_500_000, 30.0, "vp9"),
        LadderRung::new(854, 480, 1_000_000, 30.0, "vp9"),
    ]);
    let report = LadderValidator::new(LadderSpec::Hls).validate(&ladder);
    assert!(
        report.is_ok(),
        "HLS ladder validation failed: {:?}",
        report.errors()
    );
}

// ── eta_smoother ─────────────────────────────────────────────────────────────
#[test]
fn eta_smoother_rolling_single_push() {
    use oximedia_transcode::eta_smoother::{EtaSmoother, RollingEtaSmoother};
    use std::time::Duration;
    let mut s = RollingEtaSmoother::new(8);
    assert!(s.smoothed_eta().is_none()); // no samples yet
    s.push(Duration::from_secs(120));
    let eta = s.smoothed_eta().expect("should have eta after one push");
    assert_eq!(eta.as_secs(), 120);
}

// ── gop_optimizer ────────────────────────────────────────────────────────────
#[test]
fn gop_optimizer_default_config() {
    use oximedia_transcode::gop_optimizer::GopConfig;
    let cfg = GopConfig::default();
    assert!(cfg.min_gop < cfg.max_gop);
    assert!(cfg.fixed_period > 0);
}

// ── hdr_format ───────────────────────────────────────────────────────────────
#[test]
fn hdr_format_names() {
    use oximedia_transcode::hdr_format::HdrFormat;
    assert_eq!(HdrFormat::Hdr10.name(), "HDR10");
    assert_eq!(HdrFormat::HlgBt2100.name(), "HLG BT.2100");
    assert!(!HdrFormat::Hdr10.requires_dynamic_metadata());
    assert!(HdrFormat::Hdr10Plus.requires_dynamic_metadata());
}

// ── hdr_metadata_bridge ──────────────────────────────────────────────────────
#[test]
fn hdr_metadata_bridge_policy_default() {
    use oximedia_transcode::hdr_metadata_bridge::HdrPolicy;
    assert_eq!(HdrPolicy::default(), HdrPolicy::Passthrough);
}

// ── lookahead_buffer ─────────────────────────────────────────────────────────
#[test]
fn lookahead_buffer_empty_analysis() {
    use oximedia_transcode::lookahead_buffer::{LookaheadBuffer, LookaheadConfig};
    let buf = LookaheadBuffer::new(LookaheadConfig::default())
        .expect("default config should produce a valid buffer");
    assert_eq!(buf.len(), 0);
    assert!(buf.is_empty());
}

// ── mmap_io ──────────────────────────────────────────────────────────────────
#[test]
fn mmap_io_config_default_access_pattern() {
    use oximedia_transcode::mmap_io::{AccessPattern, LargeFileConfig};
    let cfg = LargeFileConfig::default();
    assert_eq!(cfg.access_pattern, AccessPattern::Sequential);
}

// ── output_validator ─────────────────────────────────────────────────────────
#[test]
fn output_validator_streaming_profile_passes() {
    use oximedia_transcode::output_validator::{
        ActualOutputProperties, OutputSpec, OutputValidator, ValidationProfile,
    };
    let spec = OutputSpec::builder()
        .video_codec("vp9")
        .resolution(1920, 1080)
        .build();
    let actual = ActualOutputProperties {
        video_codec: Some("vp9".to_string()),
        audio_codec: None,
        width: Some(1920),
        height: Some(1080),
        video_bitrate_bps: None,
        audio_bitrate_bps: None,
        duration_secs: None,
        frame_rate_num: None,
        frame_rate_den: None,
        container_format: None,
    };
    let report = OutputValidator::new(ValidationProfile::streaming()).validate(&spec, &actual);
    assert!(report.passed());
}

// ── pipeline_executor ────────────────────────────────────────────────────────
#[test]
fn pipeline_executor_passthrough_stage() {
    use oximedia_transcode::pipeline_executor::{PassthroughStage, PipelineFrame};
    let frame = PipelineFrame::video(vec![0u8; 16], 0, 320, 240);
    assert_eq!(frame.width, 320);
    assert_eq!(frame.height, 240);
    assert!(!frame.is_audio);
    let _ = PassthroughStage; // zero-sized, just prove it exists
}

// ── profile_io ───────────────────────────────────────────────────────────────
#[test]
fn profile_io_round_trip() {
    use oximedia_transcode::profile_io::TranscodeProfileExport;
    let export = TranscodeProfileExport::new("test", "vp9", "opus", "4000k", "128k", "good");
    let json = export.to_json();
    let parsed = TranscodeProfileExport::from_json(&json).expect("round-trip should succeed");
    assert_eq!(parsed, export);
}

// ── retry_backoff ────────────────────────────────────────────────────────────
#[test]
fn retry_backoff_default_policy() {
    use oximedia_transcode::retry_backoff::{BackoffPolicy, RetryScheduler};
    use std::time::Duration;
    let policy = BackoffPolicy::default();
    assert!(policy.multiplier >= 1.0);
    assert_eq!(policy.max_attempts, Some(5));

    let mut sched = RetryScheduler::new(policy, 42);
    // First attempt delay
    let delay = sched.delay_for_attempt(1);
    assert!(delay.is_some());
    let d = delay.expect("first delay should exist");
    assert!(d <= Duration::from_secs(30));
}

// ── spatial_audio_passthrough ────────────────────────────────────────────────
#[test]
fn spatial_audio_passthrough_planner_new() {
    use oximedia_transcode::spatial_audio_passthrough::{
        SpatialAudioPlanner, SpatialPlannerConfig,
    };
    let planner = SpatialAudioPlanner::new(SpatialPlannerConfig::default());
    let _ = planner; // alive
}

// ── thumbnail_strip ──────────────────────────────────────────────────────────
#[test]
fn thumbnail_strip_frame_count() {
    use oximedia_transcode::thumbnail_strip::{ThumbnailStrip, ThumbnailStripConfig};
    let cfg = ThumbnailStripConfig::default_web();
    let strip = ThumbnailStrip::new(cfg, 30.0); // 30 s at 5 s interval → 6 frames
    assert_eq!(strip.frame_count(), 6);
}

// ── transcode_cache ──────────────────────────────────────────────────────────
#[test]
fn transcode_cache_miss_then_insert() {
    use oximedia_transcode::transcode_cache::{
        CacheKey, CacheParams, EvictionPolicy, TranscodeCache, TranscodeCacheConfig,
    };
    let cfg = TranscodeCacheConfig {
        max_entries: 8,
        max_bytes: 1024 * 1024,
        eviction_policy: EvictionPolicy::Lru,
    };
    let mut cache = TranscodeCache::new(cfg);
    let params = CacheParams {
        codec: "vp9".into(),
        bitrate_bps: 4_000_000,
        width: 1920,
        height: 1080,
        extra: Default::default(),
    };
    let key = CacheKey::new(0xdeadbeef_u64, &params);
    assert!(cache.get(&key).is_none());
    let out = std::env::temp_dir()
        .join("oximedia-smoke-transcode-cache.webm")
        .to_string_lossy()
        .to_string();
    cache
        .insert(key.clone(), out, 1024)
        .expect("insert should succeed");
    assert!(cache.get(&key).is_some());
}

// ── transcode_estimator ──────────────────────────────────────────────────────
#[test]
fn transcode_estimator_av1_estimate() {
    use oximedia_transcode::transcode_estimator::{
        EstimateInput, TargetCodec, TranscodeEstimatorV2,
    };
    let input = EstimateInput {
        duration_secs: 60.0,
        width: 1920,
        height: 1080,
        input_bitrate_kbps: 8_000,
        target_bitrate_kbps: 4_000,
        codec: TargetCodec::Av1,
        has_hdr: false,
    };
    let result = TranscodeEstimatorV2::default().estimate(&input, 8);
    assert!(result.estimated_secs > 0.0);
    assert!(result.output_size_bytes_estimated > 0);
}

// ── watch_transcode ──────────────────────────────────────────────────────────
#[test]
fn watch_transcode_watcher_new() {
    use oximedia_transcode::watch_transcode::{
        TranscodeProfile, WatchTranscoder, WatchTranscoderConfig,
    };
    use oximedia_transcode::TranscodeConfig;
    use std::path::PathBuf;
    let profile = TranscodeProfile::new("test_profile", TranscodeConfig::default());
    let cfg = WatchTranscoderConfig::new(
        PathBuf::from(std::env::temp_dir().join("oximedia-watch-in")),
        profile,
    );
    let watcher = WatchTranscoder::new(cfg);
    let _ = watcher;
}

// ── watcher ──────────────────────────────────────────────────────────────────
#[test]
fn watcher_config_new() {
    use oximedia_transcode::watcher::{WatchConfig, WatchedFile};
    let cfg = WatchConfig::new("/tmp/in", "/tmp/out", "web_720p", 500);
    assert_eq!(cfg.poll_interval_ms, 500);

    let wf = WatchedFile::new("/tmp/in/test.mp4", 1024, 0);
    assert_eq!(wf.size_bytes, 1024);
}
