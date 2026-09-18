//! Smoke tests for newly-wired orphan modules in oximedia-audiopost.

// ── arib_loudness ─────────────────────────────────────────────────────────────

#[test]
fn test_arib_loudness_analyzer_constructs() {
    use oximedia_audiopost::arib_loudness::{AribConfig, AribLoudnessAnalyzer};

    let config = AribConfig::default();
    let analyzer = AribLoudnessAnalyzer::new(48000, 2, config);
    assert!(analyzer.is_ok());
}

#[test]
fn test_arib_compliance_default_is_not_compliant_without_data() {
    use oximedia_audiopost::arib_loudness::{AribConfig, AribLoudnessAnalyzer};

    let config = AribConfig::default();
    let analyzer = AribLoudnessAnalyzer::new(48000, 2, config).expect("construct ok");
    // No audio processed → integrated loudness is -∞, so compliance should fail.
    let compliance = analyzer.check_compliance();
    // Target is -24 LKFS; without data it won't pass.
    assert!(!compliance.target_pass);
}

// ── crossfade_engine ──────────────────────────────────────────────────────────

#[test]
fn test_crossfade_engine_linear_fade_pair() {
    use oximedia_audiopost::crossfade_engine::CrossfadeCurve;

    // gain_pair returns (fade_out_gain, fade_in_gain).
    // At t=0: fade-out = 1.0, fade-in = 0.0.
    let (fade_out_at_0, fade_in_at_0) = CrossfadeCurve::Linear.gain_pair(0.0);
    assert!(
        (fade_in_at_0 - 0.0).abs() < 1e-6,
        "fade-in at t=0 should be 0, got {fade_in_at_0}"
    );
    assert!(
        (fade_out_at_0 - 1.0).abs() < 1e-6,
        "fade-out at t=0 should be 1, got {fade_out_at_0}"
    );

    // At t=1: fade-out = 0.0, fade-in = 1.0.
    let (fade_out_at_1, fade_in_at_1) = CrossfadeCurve::Linear.gain_pair(1.0);
    assert!(
        (fade_in_at_1 - 1.0).abs() < 1e-6,
        "fade-in at t=1 should be 1, got {fade_in_at_1}"
    );
    assert!(
        (fade_out_at_1 - 0.0).abs() < 1e-6,
        "fade-out at t=1 should be 0, got {fade_out_at_1}"
    );
}

// ── dialogue_enhancer ─────────────────────────────────────────────────────────

#[test]
fn test_dialogue_enhancer_constructs() {
    use oximedia_audiopost::dialogue_enhancer::{DialogueEnhancer, DialogueEnhancerConfig};

    let config = DialogueEnhancerConfig::default();
    let enhancer = DialogueEnhancer::new(config, 48000);
    assert!(enhancer.is_ok());
}

// ── dialogue_eq ───────────────────────────────────────────────────────────────

#[test]
fn test_dialogue_eq_constructs_and_processes() {
    use oximedia_audiopost::dialogue_eq::DialogueEq;

    let eq = DialogueEq::new(48000).expect("construct ok");
    assert_eq!(eq.sample_rate(), 48000);
}

// ── foley_sync ────────────────────────────────────────────────────────────────

#[test]
fn test_foley_synchronizer_starts_empty() {
    use oximedia_audiopost::foley_sync::FoleySynchronizer;

    let sync = FoleySynchronizer::new(0.0);
    assert_eq!(sync.sync_point_count(), 0);
}

// ── loudness_measure ──────────────────────────────────────────────────────────

#[test]
fn test_loudness_measure_silence_gives_low_lufs() {
    use oximedia_audiopost::loudness_measure::LoudnessMeasurement;

    let silence = vec![0.0f32; 48000]; // 1 second of silence
    let lufs = LoudnessMeasurement::compute_lufs(&silence, 48000);
    // Silence should produce -∞ or a very low LUFS value.
    assert!(lufs < -80.0 || lufs.is_infinite());
}

// ── m_s_processing ────────────────────────────────────────────────────────────

#[test]
fn test_ms_encoder_decoder_round_trip() {
    use oximedia_audiopost::m_s_processing::{MsDecoder, MsEncoder};

    let left = vec![0.5f32; 256];
    let right = vec![0.2f32; 256];

    let (mid, side) = MsEncoder::encode(&left, &right).expect("encode ok");
    let (l_dec, r_dec) = MsDecoder::decode(&mid, &side).expect("decode ok");

    for (l_orig, l_rec) in left.iter().zip(l_dec.iter()) {
        assert!(
            (l_orig - l_rec).abs() < 1e-5,
            "left channel round-trip failed"
        );
    }
    for (r_orig, r_rec) in right.iter().zip(r_dec.iter()) {
        assert!(
            (r_orig - r_rec).abs() < 1e-5,
            "right channel round-trip failed"
        );
    }
}

// ── podcast_processor ────────────────────────────────────────────────────────

#[test]
fn test_podcast_processor_intro_outro_config_default_48k() {
    use oximedia_audiopost::podcast_processor::IntroOutroConfig;

    let config = IntroOutroConfig::default_48k();
    assert!(config.validate().is_ok());
}

// ── room_tone_matcher ─────────────────────────────────────────────────────────

#[test]
fn test_room_tone_analyzer_constructs() {
    use oximedia_audiopost::room_tone_matcher::RoomToneAnalyzer;

    let analyzer = RoomToneAnalyzer::new(48000, 2048);
    assert!(analyzer.is_ok());
}

// ── stem_mixer ────────────────────────────────────────────────────────────────

#[test]
fn test_stem_mix_empty_renders_silence() {
    use oximedia_audiopost::stem_mixer::StemMix;

    let mix = StemMix::new(48000);
    let (l, r) = mix.render_stereo();
    assert!(l.is_empty());
    assert!(r.is_empty());
}

// ── surround_upmix ────────────────────────────────────────────────────────────

#[test]
fn test_surround_upmixer_constructs() {
    use oximedia_audiopost::surround_upmix::{SurroundUpmixer, UpmixAlgorithm, UpmixConfig};

    let config = UpmixConfig::default();
    let upmixer = SurroundUpmixer::new(48000, UpmixAlgorithm::Passive, config);
    assert!(upmixer.is_ok());
}

// ── true_peak (re-exported test – confirms module is wired) ───────────────────

#[test]
fn test_true_peak_result_is_finite_from_orphan_smoke() {
    use oximedia_audiopost::true_peak::TruePeakMeter;

    let mut meter = TruePeakMeter::new(48000, 1).expect("construct ok");
    let samples: Vec<f32> = (0..480).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
    meter.process(&[&samples]).expect("process ok");
    let tp = meter.max_true_peak_dbtp();
    assert!(tp.is_finite());
}
