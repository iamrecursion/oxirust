//! Smoke tests for all 15 newly wired access orphan modules.

// ── haptic ───────────────────────────────────────────────────────────────────

#[test]
fn test_haptic_generator_from_beat_onset() {
    use oximedia_access::haptic::{HapticDescriptionGenerator, MediaEvent};
    let gen = HapticDescriptionGenerator::new();
    let pattern = gen.from_event(MediaEvent::BeatOnset { intensity: 0.8 });
    assert!(!pattern.is_empty());
}

#[test]
fn test_haptic_waveform_duration() {
    use oximedia_access::haptic::HapticWaveform;
    // Click is short, Rise is longer
    assert!(
        HapticWaveform::Click.typical_duration_ms() < HapticWaveform::Rise.typical_duration_ms()
    );
}

// ── color_blind_sim ──────────────────────────────────────────────────────────

#[test]
fn test_color_blind_sim_types() {
    use oximedia_access::color_blind_sim::{ColorBlindSimulator, CvdType, Rgba};
    let sim = ColorBlindSimulator::for_type(CvdType::Deuteranopia);
    let pixel = Rgba::rgba(200, 50, 50, 255);
    let simulated = sim.simulate_pixel(pixel);
    // Alpha channel must be preserved
    assert_eq!(simulated.a, 255);
}

#[test]
fn test_color_blind_sim_contrast_ratio() {
    use oximedia_access::color_blind_sim::{contrast_ratio, Rgba};
    let black = Rgba::rgba(0, 0, 0, 255);
    let white = Rgba::rgba(255, 255, 255, 255);
    let ratio = contrast_ratio(black, white);
    // WCAG specifies black-on-white is 21:1
    assert!((ratio - 21.0).abs() < 0.1, "contrast ratio = {ratio}");
}

#[test]
fn test_color_blind_sim_cvd_type_display() {
    use oximedia_access::color_blind_sim::CvdType;
    // Use Display trait (std::fmt::Display is implemented)
    assert!(!format!("{}", CvdType::Deuteranopia).is_empty());
    assert!(!format!("{}", CvdType::Protanopia).is_empty());
    assert!(!format!("{}", CvdType::Tritanopia).is_empty());
}

// ── wcag_checker ─────────────────────────────────────────────────────────────

#[test]
fn test_wcag_checker_criterion_construction() {
    use oximedia_access::wcag_checker::{WcagCriterion, WcagLevel};
    let criterion = WcagCriterion::new("1.4.3", "Contrast (Minimum)", WcagLevel::AA);
    assert_eq!(criterion.id, "1.4.3");
    assert_eq!(criterion.level, WcagLevel::AA);
}

#[test]
fn test_wcag_checker_level_ordering() {
    use oximedia_access::wcag_checker::WcagLevel;
    assert!(WcagLevel::A < WcagLevel::AA);
    assert!(WcagLevel::AA < WcagLevel::AAA);
}

// ── parallel_compliance ──────────────────────────────────────────────────────

#[test]
fn test_parallel_compliance_batch_check() {
    use oximedia_access::parallel_compliance::{
        BatchConfig, MediaAssetInfo, ParallelComplianceChecker,
    };
    let checker = ParallelComplianceChecker::new(BatchConfig::default());
    let assets = vec![
        MediaAssetInfo::new("video_a.mp4").with_has_captions(true),
        MediaAssetInfo::new("video_b.mp4").with_has_captions(false),
    ];
    let report = checker.check_batch(&assets);
    assert_eq!(report.total_assets(), 2);
}

// ── auto_alt_text ────────────────────────────────────────────────────────────

#[test]
fn test_auto_alt_text_generation() {
    use oximedia_access::auto_alt_text::{AltTextGenerator, DominantColor, ImageFeatures};
    let features = ImageFeatures {
        width: 1920,
        height: 1080,
        dominant_color: DominantColor::Blue,
        brightness: 0.65,
        edge_density: 0.3,
        texture_complexity: 0.4,
        face_count: 2,
        text_present: false,
    };
    let gen = AltTextGenerator::default();
    let alt = gen.generate(&features);
    assert!(!alt.text.is_empty());
}

// ── chapter_navigator ────────────────────────────────────────────────────────

#[test]
fn test_chapter_navigator_add_and_find() {
    use oximedia_access::chapter_navigator::{Chapter, ChapterList};
    let mut list = ChapterList::new(120_000); // 2 min total
    list.add_chapter(Chapter::new(1, 0, "Introduction"))
        .expect("add ok");
    list.add_chapter(Chapter::new(2, 60_000, "Part 1"))
        .expect("add ok");
    assert_eq!(list.len(), 2);
    let ch = list.chapter_at(90_000);
    assert!(ch.is_some());
    assert_eq!(ch.unwrap().title, "Part 1");
}

// ── text_to_speech_hints ─────────────────────────────────────────────────────

#[test]
fn test_tts_hints_engine_process() {
    use oximedia_access::text_to_speech_hints::{TtsHintConfig, TtsHintEngine};
    let config = TtsHintConfig {
        use_default_abbreviations: true,
        use_default_phonetics: false,
        ..Default::default()
    };
    let engine = TtsHintEngine::new(config);
    // A basic string should pass through (no crashes)
    let result = engine.process("Hello world.");
    assert!(!result.is_empty());
}

#[test]
fn test_tts_phonetic_hint_spelling_variant() {
    use oximedia_access::text_to_speech_hints::PhoneticHint;
    let hint = PhoneticHint::spelling("GIF", "jif");
    assert_eq!(hint.word, "GIF");
    assert_eq!(hint.pronunciation, "jif");
}

// ── audio_description_meta ───────────────────────────────────────────────────

#[test]
fn test_audio_description_meta_style() {
    use oximedia_access::audio_description_meta::AudioDescriptionStyle;
    assert_eq!(AudioDescriptionStyle::Standard.label(), "Standard");
    assert_eq!(AudioDescriptionStyle::Extended.label(), "Extended");
}

#[test]
fn test_audio_description_priority_ordering() {
    use oximedia_access::audio_description_meta::AdPriority;
    assert!(AdPriority::Primary < AdPriority::Secondary);
    assert!(AdPriority::Secondary < AdPriority::Supplemental);
}

// ── spatial_accessibility ────────────────────────────────────────────────────

#[test]
fn test_spatial_direction_labels() {
    use oximedia_access::spatial_accessibility::SpatialDirection;
    assert_eq!(SpatialDirection::Left.label(), "Left");
    assert_eq!(SpatialDirection::Right.label(), "Right");
    assert_eq!(SpatialDirection::Behind.label(), "Behind");
}

#[test]
fn test_spatial_direction_pan_positions() {
    use oximedia_access::spatial_accessibility::SpatialDirection;
    assert!(SpatialDirection::Left.pan_position() < 0.0);
    assert!(SpatialDirection::Right.pan_position() > 0.0);
    assert_eq!(SpatialDirection::Front.pan_position(), 0.0);
}

// ── precomputed_filter ───────────────────────────────────────────────────────

#[test]
fn test_precomputed_filter_biquad_identity() {
    use oximedia_access::precomputed_filter::BiquadCoefficients;
    let id = BiquadCoefficients::IDENTITY;
    assert!(id.is_identity());
    // Applying identity to a sample should return the same value
    let mut state = [0.0f64; 2];
    let output = id.process_sample(0.5, &mut state);
    assert!((output - 0.5).abs() < 1e-9);
}

// ── audio_profile ────────────────────────────────────────────────────────────

#[test]
fn test_audio_profile_frequency_bands() {
    use oximedia_access::audio_profile::FrequencyBand;
    // Each band should have a non-empty display label
    let band = FrequencyBand::Mid;
    assert!(!format!("{band}").is_empty());
}

#[test]
fn test_audio_profile_hearing_severity_ordering() {
    use oximedia_access::audio_profile::HearingSeverity;
    assert!(HearingSeverity::None < HearingSeverity::Mild);
    assert!(HearingSeverity::Mild < HearingSeverity::Moderate);
    assert!(HearingSeverity::Moderate < HearingSeverity::Severe);
    assert!(HearingSeverity::Severe < HearingSeverity::Profound);
}

// ── adaptive_font ────────────────────────────────────────────────────────────

#[test]
fn test_adaptive_font_config_defaults() {
    use oximedia_access::adaptive_font::AdaptiveFontConfig;
    let cfg = AdaptiveFontConfig::default();
    assert!(cfg.min_size_px > 0);
    assert!(cfg.max_size_px > cfg.min_size_px);
}

#[test]
fn test_adaptive_font_policy_default_is_adaptive() {
    use oximedia_access::adaptive_font::FontSizePolicy;
    assert_eq!(FontSizePolicy::default(), FontSizePolicy::Adaptive);
}

// ── sign_language_metadata ───────────────────────────────────────────────────

#[test]
fn test_sign_language_code_iso() {
    use oximedia_access::sign_language_metadata::SignLanguageCode;
    assert_eq!(SignLanguageCode::Asl.iso_code(), "ase");
    assert_eq!(SignLanguageCode::Bsl.iso_code(), "bfi");
    assert_eq!(SignLanguageCode::Jsl.iso_code(), "jsl");
}

#[test]
fn test_sign_language_signer_positions() {
    use oximedia_access::sign_language_metadata::SignerPosition;
    // There should be at least a BottomRight position
    let pos = SignerPosition::BottomRight;
    assert_eq!(pos, SignerPosition::BottomRight);
}

// ── caption_collision ────────────────────────────────────────────────────────

#[test]
fn test_caption_box_collides() {
    use oximedia_access::caption_collision::CaptionBox;
    let a = CaptionBox {
        id: "a".to_string(),
        x: 0,
        y: 0,
        width: 100,
        height: 50,
        start_ms: 0,
        end_ms: 3000,
        priority: 1,
    };
    let b = CaptionBox {
        id: "b".to_string(),
        x: 50, // overlaps with a spatially
        y: 25,
        width: 100,
        height: 50,
        start_ms: 2000, // overlaps with a temporally
        end_ms: 5000,
        priority: 1,
    };
    assert!(a.collides_with(&b));
}

#[test]
fn test_caption_box_no_temporal_overlap() {
    use oximedia_access::caption_collision::CaptionBox;
    let a = CaptionBox {
        id: "a".to_string(),
        x: 0,
        y: 0,
        width: 100,
        height: 50,
        start_ms: 0,
        end_ms: 2000,
        priority: 1,
    };
    let b = CaptionBox {
        id: "b".to_string(),
        x: 0,
        y: 0,
        width: 100,
        height: 50,
        start_ms: 3000,
        end_ms: 5000,
        priority: 1, // no temporal overlap
    };
    assert!(!a.collides_with(&b));
}

// ── accessibility_report ─────────────────────────────────────────────────────

#[test]
fn test_accessibility_report_criterion_label() {
    use oximedia_access::accessibility_report::Criterion;
    // Each criterion should have a non-empty label
    assert!(!Criterion::CaptionsPresent.label().is_empty());
}

#[test]
fn test_accessibility_report_wcag_level() {
    use oximedia_access::accessibility_report::{Criterion, WcagLevel};
    // CaptionsPresent is a Level A requirement
    assert_eq!(Criterion::CaptionsPresent.minimum_level(), WcagLevel::A);
}

#[test]
fn test_accessibility_report_builder_pass() {
    use oximedia_access::accessibility_report::{Criterion, Finding, ReportBuilder};
    let finding = Finding::pass(Criterion::CaptionsPresent);
    let report = ReportBuilder::new("asset-1")
        .record(finding)
        .build()
        .expect("build ok");
    assert_eq!(report.asset_id, "asset-1");
}
