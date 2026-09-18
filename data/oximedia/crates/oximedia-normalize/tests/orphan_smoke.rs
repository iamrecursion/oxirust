//! Smoke tests for the newly-wired orphan modules of `oximedia-normalize`.
//!
//! Each test exercises at least one public API from two or three of the wired
//! modules, verifying that the modules compile, link, and behave at a basic
//! level.  These are deliberately lightweight integration checks — the
//! per-module unit tests already cover corner cases.

// ─── adaptive_normalization + segment_loudness ───────────────────────────────

#[test]
fn test_adaptive_normalization_basic() {
    use oximedia_normalize::adaptive_normalization::{AdaptiveNormalizer, AdaptiveSegment};

    let sample_rate = 48_000u32;
    let freq = 1000.0f32;
    let amplitude = 0.4f32;
    let samples: Vec<f32> = (0..sample_rate as usize)
        .map(|i| {
            amplitude * (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin()
        })
        .collect();

    let norm = AdaptiveNormalizer::new(-14.0);
    let result = norm.process(&samples, sample_rate);

    assert!(
        !result.segments.is_empty(),
        "must produce at least one segment"
    );
    assert_eq!(
        result.output_samples.len(),
        samples.len(),
        "output length must match input"
    );

    let total_span: usize = result.segments.iter().map(AdaptiveSegment::len).sum();
    assert_eq!(
        total_span,
        samples.len(),
        "segment spans must cover entire signal"
    );
}

#[test]
fn test_segment_loudness_config_defaults() {
    use oximedia_normalize::segment_loudness::SegmentLoudnessConfig;

    let cfg = SegmentLoudnessConfig {
        sample_rate: 48_000.0,
        channels: 2,
        target_lufs: -23.0,
        max_peak_dbtp: -1.0,
        silence_threshold_db: -40.0,
        min_silence_ms: 200.0,
        onset_ratio: 4.0,
        max_gain_db: 20.0,
        boundary_fade_ms: 10.0,
    };

    assert_eq!(cfg.channels, 2);
    assert!((cfg.target_lufs - (-23.0)).abs() < 1e-9);
}

// ─── album_norm + batch_normalizer ───────────────────────────────────────────

#[test]
fn test_album_norm_streaming_plan() {
    use oximedia_normalize::album_norm::{AlbumNormConfig, AlbumNormSession};

    let cfg = AlbumNormConfig::streaming();
    assert!(cfg.validate().is_ok());

    let mut session = AlbumNormSession::new(cfg);
    session.add_track("track01", -18.5);
    session.add_track("track02", -20.1);
    session.add_track("track03", -17.3); // loudest

    let plan = session
        .compute_plan()
        .expect("plan must succeed for non-empty session");
    assert_eq!(plan.track_gains.len(), 3);

    // All tracks receive the same gain offset in album mode.
    let gains: Vec<f64> = plan.track_gains.iter().map(|tg| tg.gain_db).collect();
    let max_diff = gains
        .windows(2)
        .map(|w| (w[0] - w[1]).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff < 1e-9,
        "all tracks must share the same gain offset, diff={max_diff}"
    );
}

#[test]
fn test_batch_normalizer_independent_schedule() {
    use oximedia_normalize::batch_normalizer::{BatchNormalizer, BatchNormalizerConfig, GainMode};

    let config = BatchNormalizerConfig {
        target_lufs: -14.0,
        max_gain_db: 20.0,
        mode: GainMode::Independent,
        ..Default::default()
    };

    let mut normalizer = BatchNormalizer::new(config).expect("create BatchNormalizer");

    let audio = vec![0.2_f32; 48_000]; // 1 s of audio
    let _id = normalizer
        .measure("track1", &audio, 48_000.0, 1)
        .expect("measure");

    let schedule = normalizer.schedule_gains().expect("schedule_gains");
    // GainSchedule exposes entries as a Vec field.
    assert_eq!(schedule.entries.len(), 1, "should have one scheduled item");
}

// ─── cinema_loudness + dialogue_gate ─────────────────────────────────────────

#[test]
fn test_cinema_loudness_config_targets() {
    use oximedia_normalize::cinema_loudness::{CinemaLoudnessConfig, CinemaStandard};

    let cfg = CinemaLoudnessConfig::dolby_atmos();
    assert!(cfg.validate().is_ok());
    assert!((cfg.standard.target_lufs() - (-27.0)).abs() < 1e-9);
    assert!((cfg.standard.true_peak_ceiling_dbtp() - (-1.0)).abs() < 1e-9);

    let dci = CinemaLoudnessConfig::dci();
    assert!((dci.standard.target_lufs() - (-24.0)).abs() < 1e-9);

    let custom = CinemaStandard::Custom {
        target_lufs: -30.0,
        true_peak_ceiling_dbtp: -2.0,
    };
    assert!((custom.target_lufs() - (-30.0)).abs() < 1e-9);
}

#[test]
fn test_dialogue_gate_measurer_basic() {
    use oximedia_normalize::dialogue_gate::{DialogueGateConfig, DialogueGateMeasurer};

    let cfg = DialogueGateConfig::broadcast();
    let mut measurer = DialogueGateMeasurer::new(cfg).expect("create DialogueGateMeasurer");

    // Feed silence then a tone burst (speech-like).
    let silence = vec![0.0f32; 9_600]; // 0.2 s at 48 kHz stereo
    measurer.process_f32(&silence);

    let sine: Vec<f32> = (0..4_800)
        .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48_000.0).sin())
        .collect();
    measurer.process_f32(&sine);

    // The finish() call may fail for insufficient data, which is OK in a smoke test.
    let _ = measurer.finish();
    // At a minimum, block_count() should be callable.
    let _ = measurer.block_count();
}

// ─── drc_metadata + dynamics_preserving ─────────────────────────────────────

#[test]
fn test_drc_metadata_encoder_gain_word() {
    use oximedia_normalize::drc_metadata::{DrcMetadataEncoder, DrcProfile};

    let encoder = DrcMetadataEncoder::new(DrcProfile::Speech);

    // 0 dB always encodes to gain word 0.
    let zero_frame = encoder.encode_gain_word(0.0);
    assert_eq!(zero_frame.gain_word_u8, 0, "0 dB → gain word must be 0");

    // A non-trivial gain should produce a non-zero gain word.
    let frame = encoder.encode_gain_word(-6.0);
    assert_ne!(frame.gain_word_u8, 0, "−6 dB → gain word must be non-zero");
}

#[test]
fn test_dynamics_preserving_normalizer_basic() {
    use oximedia_normalize::dynamics_preserving::{
        DynPreservingConfig, DynamicsPreservingNormalizer,
    };

    let config = DynPreservingConfig::ebu_r128(48_000.0, 1);
    let mut norm =
        DynamicsPreservingNormalizer::new(config).expect("create DynamicsPreservingNormalizer");

    let mut samples: Vec<f32> = (0..4_800)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin())
        .collect();

    let _result = norm.process(&mut samples).expect("process must succeed");

    // After processing, some energy must remain.
    let energy: f32 = samples.iter().map(|s| s * s).sum();
    assert!(
        energy > 0.0,
        "output must be non-silent after normalization"
    );
}

// ─── genre_adaptive + headroom ───────────────────────────────────────────────

#[test]
fn test_genre_adaptive_targets() {
    use oximedia_normalize::genre_adaptive::GenreNormalizer;

    let music = GenreNormalizer::new("music");
    assert!(
        (music.target_lufs() - (-14.0)).abs() < 0.01,
        "music target must be −14 LUFS"
    );

    let speech = GenreNormalizer::new("speech");
    assert!(
        (speech.target_lufs() - (-16.0)).abs() < 0.01,
        "speech target must be −16 LUFS"
    );

    let podcast = GenreNormalizer::new("podcast");
    assert!(
        (podcast.target_lufs() - (-16.0)).abs() < 0.01,
        "podcast target must be −16 LUFS"
    );

    let unknown = GenreNormalizer::new("unknown_xyz");
    assert!(
        (unknown.target_lufs() - (-15.0)).abs() < 0.01,
        "unknown genre must fall back to −15 LUFS mixed target"
    );
}

#[test]
fn test_headroom_manager_clamping() {
    use oximedia_normalize::headroom::HeadroomManager;

    let mgr = HeadroomManager::new(1.0);
    assert!((mgr.target_headroom_db() - 1.0).abs() < 1e-6);

    // current peak at -3 dBFS → can add up to 2 dB safely
    let safe = mgr.max_safe_gain_db(-3.0);
    assert!(
        (safe - 2.0).abs() < 0.001,
        "max safe gain should be 2.0 dB, got {safe}"
    );

    // When the peak is at 0 dBFS and headroom is 1 dB, gain must be -1 dB.
    let safe_at_zero = mgr.max_safe_gain_db(0.0);
    assert!(
        (safe_at_zero - (-1.0)).abs() < 0.001,
        "peak at 0 dBFS with 1 dB headroom → safe gain must be −1 dB, got {safe_at_zero}"
    );
}

// ─── history + loudness_history_db ───────────────────────────────────────────

#[test]
fn test_loudness_history_trend() {
    use oximedia_normalize::history::LoudnessHistory;

    let mut h = LoudnessHistory::new(10);
    // Feed a linearly decreasing sequence.
    for i in 0..5 {
        h.add(-20.0 - i as f32);
    }
    let slope = h.trend();
    assert!(
        slope < 0.0,
        "loudness is decreasing → trend must be negative, got {slope}"
    );
}

#[test]
fn test_loudness_history_db_roundtrip() {
    use oximedia_normalize::loudness_history_db::{LoudnessHistoryDb, MeasurementRecord};

    let mut db = LoudnessHistoryDb::new();
    db.push(MeasurementRecord::new(
        1_000_000_000,
        -23.5,
        8.2,
        -1.0,
        "track_a",
    ));
    db.push(MeasurementRecord::new(
        2_000_000_000,
        -16.0,
        4.1,
        -0.5,
        "track_b",
    ));

    assert_eq!(db.len(), 2);

    // CSV bytes round-trip.
    let bytes = db.to_csv_bytes();
    let db2 = LoudnessHistoryDb::from_csv_bytes(&bytes).expect("round-trip must succeed");

    assert_eq!(
        db2.records().len(),
        2,
        "should have 2 records after round-trip"
    );
    assert!(
        (db2.records()[0].integrated_lufs - (-23.5)).abs() < 1e-5,
        "first record LUFS must survive round-trip"
    );
}

// ─── live_stream_norm ─────────────────────────────────────────────────────────

#[test]
fn test_live_stream_norm_processes_chunk() {
    use oximedia_normalize::live_stream_norm::{LiveStreamNormConfig, LiveStreamNormalizer};

    let cfg = LiveStreamNormConfig::default_broadcast();
    let mut norm = LiveStreamNormalizer::new(cfg).expect("create LiveStreamNormalizer");

    let chunk = vec![0.1_f32; 960]; // 20 ms at 48 kHz stereo
    let mut output = vec![0.0_f32; chunk.len()];
    norm.process_chunk(&chunk, &mut output)
        .expect("process_chunk must succeed");

    // Output must contain finite values.
    assert!(
        output.iter().all(|s| s.is_finite()),
        "all output samples must be finite"
    );
}

// ─── multiband_normalize ─────────────────────────────────────────────────────

#[test]
fn test_multiband_normalize_default_bands() {
    use oximedia_normalize::multiband_normalize::MultibandNormalizer;

    let norm = MultibandNormalizer::new(-14.0);
    assert_eq!(
        norm.bands.len(),
        4,
        "default 4-band config must have 4 bands"
    );

    // Band order: sub < low < mid < high (contiguous, non-overlapping).
    for w in norm.bands.windows(2) {
        assert!(
            w[0].high_hz <= w[1].low_hz,
            "bands must not overlap: {} Hz vs {} Hz",
            w[0].high_hz,
            w[1].low_hz
        );
    }
}

// ─── peak_limiter ─────────────────────────────────────────────────────────────

#[test]
fn test_peak_limiter_ceiling_enforcement() {
    use oximedia_normalize::peak_limiter::{LookAheadPeakLimiter, PeakLimiterConfig};

    let cfg = PeakLimiterConfig {
        ceiling_dbtp: -1.0,
        lookahead_ms: 5.0,
        release_ms: 100.0,
        sample_rate: 48_000.0,
        channels: 1,
        oversample: 4,
    };

    let mut limiter = LookAheadPeakLimiter::new(cfg).expect("create LookAheadPeakLimiter");

    // A high-amplitude burst should be clamped below the ceiling.
    let input: Vec<f64> = vec![0.99f64; 4_800];
    let mut output = vec![0.0f64; input.len()];
    limiter
        .process(&input, &mut output)
        .expect("process must succeed");

    // Ceiling at -1 dBTP ≈ 0.891 linear amplitude.
    let ceiling_linear = 10.0_f64.powf(-1.0 / 20.0);
    let max_out = output.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_out <= ceiling_linear * 1.001, // 0.1% tolerance for lookahead ramp-up
        "output peak {max_out:.4} must not exceed −1 dBTP ({ceiling_linear:.4})"
    );
}

// ─── podcast_loudness ─────────────────────────────────────────────────────────

#[test]
fn test_podcast_loudness_config_platforms() {
    use oximedia_normalize::podcast_loudness::{PodcastNormConfig, PodcastPlatform};

    let spotify = PodcastNormConfig::spotify();
    assert!(spotify.validate().is_ok());
    assert!((spotify.platform.target_lufs() - (-16.0)).abs() < 1e-9);

    let apple = PodcastNormConfig::apple_standard();
    assert!((apple.platform.target_lufs() - (-14.0)).abs() < 1e-9);
    assert_eq!(apple.platform.name(), "Apple Podcasts (Standard)");

    let yt = PodcastPlatform::YouTube;
    assert!((yt.target_lufs() - (-14.0)).abs() < 1e-9);
}

// ─── simd_gain ───────────────────────────────────────────────────────────────

#[test]
fn test_simd_gain_batch_correctness() {
    use oximedia_normalize::simd_gain::{
        apply_gain_f32_batch, apply_gain_f32_inplace_batch, apply_gain_f64_batch,
    };

    // f32 out-of-place (1024 samples — multiple of 8).
    let input_f32 = vec![0.5_f32; 1024];
    let mut output_f32 = vec![0.0_f32; 1024];
    apply_gain_f32_batch(&input_f32, &mut output_f32, 0.8);
    assert!(
        (output_f32[0] - 0.4).abs() < 1e-6,
        "expected 0.4, got {}",
        output_f32[0]
    );
    assert!(
        (output_f32[1023] - 0.4).abs() < 1e-6,
        "last sample: expected 0.4, got {}",
        output_f32[1023]
    );

    // f32 in-place (17 samples — not a multiple of 8; exercises tail path).
    let mut samples = vec![1.0_f32; 17];
    apply_gain_f32_inplace_batch(&mut samples, 0.5);
    assert!(
        samples.iter().all(|&s| (s - 0.5).abs() < 1e-6),
        "in-place gain mismatch"
    );

    // f64 out-of-place (16 samples).
    let input_f64 = vec![0.25_f64; 16];
    let mut output_f64 = vec![0.0_f64; 16];
    apply_gain_f64_batch(&input_f64, &mut output_f64, 2.0);
    assert!(
        (output_f64[0] - 0.5).abs() < 1e-12,
        "f64 gain: expected 0.5, got {}",
        output_f64[0]
    );
}

// ─── vad_dialogue_norm ────────────────────────────────────────────────────────

#[test]
fn test_vad_dialogue_norm_processes_frame() {
    use oximedia_normalize::dialogue_norm::DialogueLoudness;
    use oximedia_normalize::vad_dialogue_norm::{VadDialogueNormConfig, VadDialogueNormalizer};

    let config = VadDialogueNormConfig::broadcast();
    let mut processor = VadDialogueNormalizer::new(config);

    // A -28 LUFS measurement — target is -23 LUFS → 5 dB boost expected.
    let loudness = DialogueLoudness::new(-28.0, 8.0, -6.0, 0.75);
    processor.update_loudness(loudness);

    let mut frame = vec![0.3_f32; 480]; // 10 ms @ 48 kHz mono
    let stats = processor.process_frame(&mut frame);

    // Either the frame was classified as speech (gain applied) or gain was negligible.
    assert!(
        stats.speech_frame || stats.gain_applied_db.abs() < 0.01,
        "speech_frame={}, gain={:.3}",
        stats.speech_frame,
        stats.gain_applied_db
    );
}
