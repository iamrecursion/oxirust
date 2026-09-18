//! Smoke tests for the 17 newly-wired orphan modules in oximedia-hdr.
//!
//! Each test validates that the module compiles, its primary types can be
//! constructed, and its key invariants hold at a basic level.  Detailed
//! correctness tests live inside each module's `#[cfg(test)]` blocks.

// ── ambient_light_adaptation ──────────────────────────────────────────────────

#[test]
fn ambient_light_adaptation_cinema_to_home() {
    use oximedia_hdr::ambient_light_adaptation::{
        cinema_to_home_adapter, AmbientLightAdapter, ViewingSurround,
    };

    let adapter: AmbientLightAdapter =
        cinema_to_home_adapter().expect("cinema_to_home_adapter must succeed");

    let factor = adapter.adaptation_factor(ViewingSurround::DimHome);
    assert!(
        (0.0..=1.0).contains(&factor),
        "adaptation factor out of range: {factor}"
    );
}

#[test]
fn ambient_light_adaptation_display_brightness_model() {
    use oximedia_hdr::ambient_light_adaptation::DisplayBrightnessModel;

    let model = DisplayBrightnessModel::oled_1000();
    model.validate().expect("OLED-1000 model should be valid");

    let effective = model.effective_peak_nits(0.0);
    assert!(effective > 0.0, "effective peak nits must be positive");
}

// ── display_db ────────────────────────────────────────────────────────────────

#[test]
fn display_db_capability_hdr10_check() {
    use oximedia_hdr::display_db::HdrDisplayCapability;

    let hdr10 = HdrDisplayCapability::new("HDR10 Panel", 600.0, 0.005);
    assert!(
        hdr10.is_hdr10_capable(),
        "600-nit panel should be HDR10-capable"
    );

    let sdr = HdrDisplayCapability::new("SDR Panel", 300.0, 0.05);
    assert!(
        !sdr.is_hdr10_capable(),
        "300-nit panel must not qualify as HDR10"
    );
}

#[test]
fn display_db_registry_add_lookup() {
    use oximedia_hdr::display_db::{DisplayDb, HdrDisplayCapability};

    let mut db = DisplayDb::new();
    db.add(HdrDisplayCapability::new("OLED Pro", 1000.0, 0.0001));
    db.add(HdrDisplayCapability::new("SDR TV", 250.0, 0.1));

    assert_eq!(db.len(), 2);
    let found = db.find("OLED Pro");
    assert!(found.is_some(), "added display must be retrievable by name");
}

// ── dynamic_metadata_validator ────────────────────────────────────────────────

#[test]
fn dynamic_metadata_validator_valid_metadata() {
    use oximedia_hdr::dynamic_metadata::{Hdr10PlusDynamicMetadata, Hdr10PlusWindow};
    use oximedia_hdr::dynamic_metadata_validator::{
        DynMetaValidator, HDR10PLUS_APP_ID, T35_COUNTRY_USA,
    };

    let window = Hdr10PlusWindow {
        window_upper_left: (0, 0),
        window_lower_right: (3840, 2160),
        center_of_ellipse: (1920, 1080),
        rotation_angle: 0,
        semimajor_axis_external: 1080,
        semiminor_axis_external: 960,
        semimajor_axis_internal: 540,
        semiminor_axis_internal: 480,
        overlap_process_option: 0,
        maxscl: [50_000, 60_000, 40_000],
        average_maxrgb: 100,
    };

    let meta = Hdr10PlusDynamicMetadata {
        country_code: T35_COUNTRY_USA,
        terminal_provider_code: 0x003C,
        application_identifier: HDR10PLUS_APP_ID,
        application_version: 0,
        num_windows: 1,
        windows: vec![window],
        targeted_system_display_max_luminance: 10_000, // 1000 nits (units: ×10)
        average_maxrgb: 5_000,
        distribution_values: [100, 200, 300, 400, 500, 600, 700, 800, 900],
        fraction_bright_pixels: 50,
    };

    let report = DynMetaValidator::validate(&meta);
    assert_eq!(
        report.error_count(),
        0,
        "valid HDR10+ metadata must produce no errors"
    );
}

// ── hdr10plus ────────────────────────────────────────────────────────────────

#[test]
fn hdr10plus_analyzer_black_frame() {
    use oximedia_hdr::hdr10plus::Hdr10PlusFrameAnalyzer;

    let a = Hdr10PlusFrameAnalyzer::new();
    let frame = vec![0u8; 8 * 8 * 3];
    let meta = a.analyze(&frame, 8, 8);
    assert_eq!(meta.target_system_display_max_luminance, 0);
    assert_eq!(meta.num_windows, 1);
}

#[test]
fn hdr10plus_analyzer_nonzero_frame() {
    use oximedia_hdr::hdr10plus::Hdr10PlusFrameAnalyzer;

    let a = Hdr10PlusFrameAnalyzer::new();
    let frame = vec![180u8; 4 * 4 * 3];
    let meta = a.analyze(&frame, 4, 4);
    assert!(
        meta.target_system_display_max_luminance > 0,
        "180/255 PQ ≈ 2600 nits, must be nonzero"
    );
    // Max ≥ avg for uniform frame
    assert!(meta.target_system_display_max_luminance >= meta.average_maxrgb);
}

// ── hdr10plus_generator ───────────────────────────────────────────────────────

#[test]
fn hdr10plus_generator_scene_stats() {
    use oximedia_hdr::hdr10plus_generator::{Hdr10PlusMetadataStream, SceneLuminanceStats};

    // pixels_nits: 64 luminance values at 500 nits; high_threshold = 800 nits
    let pixels_nits = vec![500.0f32; 64];
    let stats = SceneLuminanceStats::from_pixels(0, &pixels_nits, 800.0)
        .expect("from_pixels must succeed for valid input");
    stats
        .validate()
        .expect("uniform-luminance stats should pass validation");

    // 2 target tiers → add_shot creates 1 trim pass per tier = 2 total passes
    let mut stream =
        Hdr10PlusMetadataStream::new(4000.0, vec![100.0, 1000.0]).expect("stream creation");
    stream.add_shot(&stats).expect("add_shot must succeed");
    // stream.len() counts trim passes; 1 shot × 2 tiers = 2 passes
    assert_eq!(
        stream.len(),
        2,
        "one shot on 2 tiers should produce 2 trim passes"
    );

    // Verify passes are accessible by tier
    let passes_100 = stream.passes_for_tier(100.0);
    assert!(!passes_100.is_empty(), "must have passes for 100-nit tier");
}

// ── hdr_fingerprint ───────────────────────────────────────────────────────────

#[test]
fn hdr_fingerprint_basic_compute() {
    use oximedia_hdr::hdr_fingerprint::HdrFrameFingerprinter;

    // 8×8 interleaved RGB frame (width * height * 3 elements)
    let pixels = vec![0.5f32; 8 * 8 * 3];
    let fingerprint = HdrFrameFingerprinter::compute(&pixels, 8, 8)
        .expect("fingerprint of uniform frame must succeed");

    // Two identical frames must produce the same fingerprint
    let fp2 = HdrFrameFingerprinter::compute(&pixels, 8, 8).expect("second fingerprint");
    assert_eq!(
        fingerprint.as_u64(),
        fp2.as_u64(),
        "identical frames → identical fingerprint"
    );
}

#[test]
fn hdr_fingerprint_different_frames_differ() {
    use oximedia_hdr::hdr_fingerprint::HdrFrameFingerprinter;

    // Build a 32×32 frame: left half bright (0.9), right half dark (0.1).
    // This creates non-uniform block means so the fingerprint is non-zero
    // and the two halves produce opposite-polarity bits → hashes will differ.
    let w = 32usize;
    let h = 32usize;
    let mut left_bright = Vec::with_capacity(w * h * 3);
    let mut right_bright = Vec::with_capacity(w * h * 3);
    for _row in 0..h {
        for col in 0..w {
            let lv = if col < w / 2 { 0.9f32 } else { 0.1 };
            let rv = if col < w / 2 { 0.1f32 } else { 0.9 };
            left_bright.push(lv);
            left_bright.push(lv);
            left_bright.push(lv);
            right_bright.push(rv);
            right_bright.push(rv);
            right_bright.push(rv);
        }
    }
    let a = HdrFrameFingerprinter::compute(&left_bright, w, h).expect("left-bright fingerprint");
    let b = HdrFrameFingerprinter::compute(&right_bright, w, h).expect("right-bright fingerprint");
    assert_ne!(
        a.as_u64(),
        b.as_u64(),
        "left-bright and right-bright frames must produce different fingerprints"
    );
}

// ── hdr_grading_assistant ─────────────────────────────────────────────────────

#[test]
fn hdr_grading_assistant_neutral_identity() {
    use oximedia_hdr::hdr_grading_assistant::CreativeLook;

    let params = CreativeLook::Neutral.params();
    let input = [0.5f32, 0.4, 0.3];
    let output = params.apply_pq(input);
    for i in 0..3 {
        assert!(
            (input[i] - output[i]).abs() < 1e-4,
            "neutral grade must be identity at channel {i}: in={}, out={}",
            input[i],
            output[i]
        );
    }
}

#[test]
fn hdr_grading_assistant_vivid_boosts_saturation() {
    use oximedia_hdr::hdr_grading_assistant::CreativeLook;

    let neutral = CreativeLook::Neutral.params();
    let vivid = CreativeLook::VividVibrant.params();

    let input = [0.8f32, 0.2, 0.1]; // highly-saturated warm colour
    let n_out = neutral.apply_pq(input);
    let v_out = vivid.apply_pq(input);

    // Vivid should differ from neutral for a non-grey input
    let differ = n_out
        .iter()
        .zip(v_out.iter())
        .any(|(a, b)| (a - b).abs() > 1e-5);
    assert!(differ, "VividVibrant should modify a saturated input");
}

// ── hdr_lut_pipeline ─────────────────────────────────────────────────────────

#[test]
fn hdr_lut_pipeline_identity_passthrough() {
    use oximedia_hdr::hdr_lut_pipeline::HdrLutPipeline;

    let pipeline = HdrLutPipeline::new();
    assert!(pipeline.is_empty(), "new pipeline has no stages");

    let (r, g, b) = pipeline
        .apply(0.5, 0.3, 0.2)
        .expect("apply on empty pipeline");
    assert!((r - 0.5).abs() < 1e-5, "empty pipeline: r passthrough");
    assert!((g - 0.3).abs() < 1e-5, "empty pipeline: g passthrough");
    assert!((b - 0.2).abs() < 1e-5, "empty pipeline: b passthrough");
}

#[test]
fn hdr_lut_pipeline_brightness_stage() {
    use oximedia_hdr::hdr_lut_pipeline::{make_brightness_lut, HdrLutPipeline, LutSize};

    let mut pipeline = HdrLutPipeline::new();
    let stage = make_brightness_lut(LutSize::S17, 2.0);
    pipeline.push(stage);
    assert_eq!(pipeline.len(), 1);

    // 0.4 * 2.0 = 0.8 (within LUT precision)
    let (r, _g, _b) = pipeline.apply(0.4, 0.4, 0.4).expect("brightness apply");
    assert!(r > 0.4, "brightness > 1 should increase r; got {r}");
    assert!(r <= 1.0, "brightness output must not exceed 1.0");
}

// ── hdr_metadata_validator ────────────────────────────────────────────────────

#[test]
fn hdr_metadata_validator_valid_mastering_display() {
    use oximedia_hdr::hdr_metadata_validator::{
        ChromaticityXy, Hdr10MasteringDisplay, HdrMetadataValidator,
    };

    let md = Hdr10MasteringDisplay {
        red: ChromaticityXy { x: 0.708, y: 0.292 },
        green: ChromaticityXy { x: 0.170, y: 0.797 },
        blue: ChromaticityXy { x: 0.131, y: 0.046 },
        white: ChromaticityXy {
            x: 0.3127,
            y: 0.3290,
        },
        min_luminance_nits: 0.005,
        max_luminance_nits: 1000.0,
    };

    let report = HdrMetadataValidator::validate_mastering_display(&md)
        .expect("validate_mastering_display must not error");
    assert!(
        report.is_compliant(),
        "valid BT.2020 mastering display must be compliant"
    );
}

// ── hdr_scene_analysis ────────────────────────────────────────────────────────

#[test]
fn hdr_scene_analysis_empty_finishes_clean() {
    use oximedia_hdr::hdr_scene_analysis::{SceneAnalyzer, SceneAnalyzerConfig};

    let config = SceneAnalyzerConfig::default();
    let mut analyzer = SceneAnalyzer::new(config);
    // Push one frame (below min_scene_frames of 8) to test graceful finish
    analyzer.push_frame(500.0).expect("push frame must succeed");
    let scenes = analyzer.finish().expect("finish must not error");
    // 1 frame < min_scene_frames=8, but finish should flush it
    let _ = scenes;
}

#[test]
fn hdr_scene_analysis_single_scene_detected() {
    use oximedia_hdr::hdr_scene_analysis::{SceneAnalyzer, SceneAnalyzerConfig};

    let config = SceneAnalyzerConfig {
        window_size: 4,
        cut_threshold_ratio: 2.0,
        min_scene_frames: 4,
        max_nits_clamp: 10_000.0,
    };
    let mut analyzer = SceneAnalyzer::new(config);
    for _ in 0..10 {
        analyzer.push_frame(500.0).expect("push");
    }
    let scenes = analyzer.finish().expect("finish must not error");
    assert!(
        !scenes.is_empty(),
        "10 frames should produce at least one scene"
    );
    assert!(scenes[0].peak_nits > 0.0, "peak nits must be positive");
}

// ── hlg_broadcast_constraints ─────────────────────────────────────────────────

#[test]
fn hlg_broadcast_constraints_valid_signal_compliant() {
    use oximedia_hdr::hlg_broadcast_constraints::{HlgBroadcastValidator, HlgSignalParams};

    // Nominal BT.2100 HLG broadcast parameters
    let params = HlgSignalParams {
        peak_signal: 1.0,
        black_signal: 0.0,
        nominal_peak_nits: 1000.0,
        system_gamma: 1.2,
        extended_range: false,
    };

    let report = HlgBroadcastValidator::validate(&params);
    assert_eq!(
        report.error_count(),
        0,
        "valid HLG broadcast signal must produce no errors"
    );
}

// ── hlg_display_gamma ─────────────────────────────────────────────────────────

#[test]
fn hlg_display_gamma_for_1000_nit_display() {
    use oximedia_hdr::hlg_display_gamma::HlgDisplayAdapter;

    let adapter = HlgDisplayAdapter::for_display_nits(1000.0)
        .expect("1000-nit adapter must be constructible");

    let gamma = adapter.system_gamma();
    // BT.2100 reference: gamma for 1000 nit ≈ 1.2
    assert!(
        (1.0..=1.5).contains(&gamma),
        "system gamma for 1000-nit display out of expected range: {gamma}"
    );
}

#[test]
fn hlg_display_gamma_ootf_unity_on_grey() {
    use oximedia_hdr::hlg_display_gamma::HlgDisplayAdapter;

    let adapter = HlgDisplayAdapter::for_display_nits(1000.0).expect("adapter");
    let (r, g, b) = adapter.apply_ootf(0.5, 0.5, 0.5);
    // Grey input: all channels must remain equal after OOTF (achromatic invariant)
    assert!(
        (r - g).abs() < 1e-5 && (g - b).abs() < 1e-5,
        "grey OOTF must preserve achromatic balance: r={r}, g={g}, b={b}"
    );
}

// ── hlg_reference_display ─────────────────────────────────────────────────────

#[test]
fn hlg_reference_display_1000_nit_construction() {
    use oximedia_hdr::hlg_reference_display::{
        AmbientViewingEnvironment, HlgReferenceDisplay, ReferenceDisplayPeak,
    };

    let display =
        HlgReferenceDisplay::new(ReferenceDisplayPeak::Nits1000).expect("1000-nit display");
    let params = display.ootf_params();
    assert!(
        params.gamma > 1.0,
        "gamma must exceed 1.0 for 1000-nit display"
    );

    let _ = HlgReferenceDisplay::with_ambient(
        ReferenceDisplayPeak::Nits1000,
        AmbientViewingEnvironment::Dim,
    )
    .expect("with_ambient must succeed");
}

// ── hlg_to_pq ────────────────────────────────────────────────────────────────

#[test]
fn hlg_to_pq_monotonic_and_bounded() {
    use oximedia_hdr::hlg_to_pq::hlg_to_pq;

    let mut prev = -1.0f32;
    for i in 0..=20u32 {
        let hlg = i as f32 / 20.0;
        let pq = hlg_to_pq(hlg).expect("valid HLG");
        assert!(
            (0.0..=1.0).contains(&pq),
            "PQ out of [0,1] for hlg={hlg}: pq={pq}"
        );
        assert!(
            pq >= prev,
            "not monotonic at hlg={hlg}: pq={pq} < prev={prev}"
        );
        prev = pq;
    }
}

#[test]
fn hlg_to_pq_rejects_out_of_range() {
    use oximedia_hdr::hlg_to_pq::hlg_to_pq;

    assert!(hlg_to_pq(-0.01).is_err(), "negative HLG must be rejected");
    assert!(hlg_to_pq(1.01).is_err(), "HLG > 1 must be rejected");
}

// ── pq_simd ──────────────────────────────────────────────────────────────────

#[test]
fn pq_simd_processor_roundtrip() {
    use oximedia_hdr::pq_simd::PqSimdProcessor;

    // Use scalar-only for deterministic results independent of CPU features.
    let proc = PqSimdProcessor::scalar_only();
    // Linear light values in [0, 1] (normalized to 10 000 nits = 1.0)
    let input: Vec<f32> = (0..=20).map(|i| i as f32 / 20.0).collect();
    let mut encoded = vec![0.0f32; input.len()];
    proc.oetf_batch(&input, &mut encoded).expect("oetf_batch");

    let mut decoded = vec![0.0f32; input.len()];
    proc.eotf_batch(&encoded, &mut decoded).expect("eotf_batch");

    for (i, (&orig, &rt)) in input.iter().zip(decoded.iter()).enumerate() {
        assert!(
            (orig - rt).abs() < 2e-3,
            "PQ round-trip error at index {i}: orig={orig}, rt={rt}"
        );
    }
}

#[test]
fn pq_simd_processor_tier_name_nonempty() {
    use oximedia_hdr::pq_simd::{PqSimdProcessor, SimdTier};

    let tier = SimdTier::detect();
    assert!(!tier.name().is_empty(), "SimdTier::name must be non-empty");

    let scalar = PqSimdProcessor::scalar_only();
    assert_eq!(scalar.tier(), SimdTier::Scalar);
}

// ── sdr_to_hdr ────────────────────────────────────────────────────────────────

#[test]
fn sdr_to_hdr_boost_increases_luminance() {
    use oximedia_hdr::sdr_to_hdr::SdrToHdrConverter;

    let converter = SdrToHdrConverter::new(4.0, 1.0);
    let [r, g, b] = converter.convert([0.25, 0.25, 0.25]);
    assert!(
        (r - 1.0).abs() < 1e-4,
        "0.25 * boost 4.0 should produce 1.0, got {r}"
    );
    let _ = (g, b);
}

#[test]
fn sdr_to_hdr_black_stays_black() {
    use oximedia_hdr::sdr_to_hdr::SdrToHdrConverter;

    let converter = SdrToHdrConverter::new(8.0, 1.5);
    assert_eq!(converter.convert([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
}

// ── soft_clip_gamut ───────────────────────────────────────────────────────────

#[test]
fn soft_clip_gamut_bt2390_in_gamut_passthrough() {
    use oximedia_hdr::soft_clip_gamut::SoftClipGamutMapper;

    let mapper = SoftClipGamutMapper::default_bt2390();
    let (r, g, b) = mapper
        .map_pixel(0.5, 0.5, 0.5)
        .expect("map_pixel must succeed");
    // Values comfortably in gamut should not be clipped significantly
    assert!(
        (0.3..=0.7).contains(&r),
        "in-gamut value r altered too much: {r}"
    );
    let _ = (g, b);
}

#[test]
fn soft_clip_gamut_out_of_gamut_clamped() {
    use oximedia_hdr::soft_clip_gamut::SoftClipGamutMapper;

    let mapper = SoftClipGamutMapper::default_bt2390();
    // Highly out-of-gamut signal
    let (r, _g, _b) = mapper.map_pixel(1.5, 0.2, 0.1).expect("map_pixel");
    assert!(
        r <= 1.0 + 1e-4,
        "out-of-gamut R must be clipped to ≤ 1.0, got {r}"
    );
}
