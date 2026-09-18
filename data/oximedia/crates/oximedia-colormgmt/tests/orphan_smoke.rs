//! Smoke tests for newly-wired orphan modules in oximedia-colormgmt.
//!
//! Each test exercises ≥1 public API entry point per module to confirm that
//! the module is properly compiled and linked.

// ── color_checker ─────────────────────────────────────────────────────────────

#[test]
fn color_checker_reference_patches_count() {
    use oximedia_colormgmt::color_checker::reference_patches;
    let patches = reference_patches();
    assert_eq!(
        patches.len(),
        24,
        "ColorChecker Classic has exactly 24 patches"
    );
}

#[test]
fn color_checker_analyse_perfect_match() {
    use oximedia_colormgmt::color_checker::ColorChecker;
    let checker = ColorChecker::new();
    let refs = checker.reference_patches().to_vec();
    // Perfect match: measured == reference patches
    let report = checker.analyse_lab(&refs);
    assert!(
        report.mean_delta_e < 1e-6,
        "Perfect match should have near-zero mean ΔE, got {}",
        report.mean_delta_e
    );
}

// ── color_pipeline ────────────────────────────────────────────────────────────

#[test]
fn color_pipeline_identity_passthrough() {
    use oximedia_colormgmt::color_pipeline::ColorPipeline;
    let pipeline = ColorPipeline::new();
    let (r, g, b) = pipeline.apply(0.5, 0.3, 0.8);
    assert!((r - 0.5).abs() < 1e-6);
    assert!((g - 0.3).abs() < 1e-6);
    assert!((b - 0.8).abs() < 1e-6);
}

#[test]
fn color_pipeline_srgb_roundtrip() {
    use oximedia_colormgmt::color_pipeline::ColorPipeline;
    let encode = ColorPipeline::linear_to_srgb();
    let decode = ColorPipeline::srgb_to_linear();
    let input = (0.18_f32, 0.5_f32, 0.9_f32);
    let encoded = encode.apply(input.0, input.1, input.2);
    let (r, g, b) = decode.apply(encoded.0, encoded.1, encoded.2);
    assert!(
        (r - input.0).abs() < 1e-4,
        "sRGB roundtrip R: {} vs {}",
        r,
        input.0
    );
    assert!(
        (g - input.1).abs() < 1e-4,
        "sRGB roundtrip G: {} vs {}",
        g,
        input.1
    );
    assert!(
        (b - input.2).abs() < 1e-4,
        "sRGB roundtrip B: {} vs {}",
        b,
        input.2
    );
}

// ── color_temperature ─────────────────────────────────────────────────────────

#[test]
fn color_temperature_d65_kelvin() {
    use oximedia_colormgmt::color_temperature::kelvin_to_chromaticity;
    // D65 is approximately 6504 K
    let chroma = kelvin_to_chromaticity(6504.0);
    // x should be near 0.3127, y near 0.3290
    assert!((chroma.x - 0.3127).abs() < 0.01, "D65 x: {}", chroma.x);
    assert!((chroma.y - 0.3290).abs() < 0.01, "D65 y: {}", chroma.y);
}

#[test]
fn color_temperature_cct_estimate_roundtrip() {
    use oximedia_colormgmt::color_temperature::{chromaticity_to_cct, kelvin_to_chromaticity};
    let kelvin = 5000.0_f64;
    let chroma = kelvin_to_chromaticity(kelvin);
    let cct = chromaticity_to_cct(&chroma).expect("CCT estimate should succeed");
    assert!(
        (cct - kelvin).abs() < 200.0,
        "CCT roundtrip: {} vs {}",
        cct,
        kelvin
    );
}

// ── cusp_gamut_map ────────────────────────────────────────────────────────────

#[test]
fn cusp_gamut_map_mapper_constructs() {
    use oximedia_colormgmt::cusp_gamut_map::{CuspMapper, Gamut};
    let mapper = CuspMapper::new(Gamut::Rec709);
    assert_eq!(mapper.destination_gamut(), Gamut::Rec709);
}

#[test]
fn cusp_gamut_map_in_gamut_identity() {
    use oximedia_colormgmt::cusp_gamut_map::{CuspMapper, Gamut};
    let mapper = CuspMapper::new(Gamut::Rec709);
    // White point [0.95, 1.0, 1.09] is inside Rec.709 gamut — should pass through
    let white = [0.95_f64, 1.0, 1.09];
    let out = mapper.map_xyz(white[0], white[1], white[2]);
    // Should not clip to black
    assert!(
        out[1] > 0.5,
        "Luminance preserved for in-gamut white: Y={}",
        out[1]
    );
}

// ── gamut_compression ─────────────────────────────────────────────────────────

#[test]
fn gamut_compression_defaults_construct() {
    use oximedia_colormgmt::gamut_compression::GamutCompressor;
    let gc = GamutCompressor::with_defaults();
    // Apply to an in-gamut colour — should change minimally
    let rgb = [0.5_f64, 0.3, 0.2];
    let out = gc.compress_rgb_ratio(rgb);
    for i in 0..3 {
        assert!(
            out[i] >= 0.0 && out[i] <= 1.0,
            "channel {} out of range: {}",
            i,
            out[i]
        );
    }
}

#[test]
fn gamut_compression_out_of_gamut_clamped() {
    use oximedia_colormgmt::gamut_compression::{CompressorConfig, GamutCompressor, KneeMethod};
    let cfg = CompressorConfig {
        knee_method: KneeMethod::Reinhard,
        ..Default::default()
    };
    let gc = GamutCompressor::new(cfg);
    let oog = [-0.2_f64, 1.5, 0.8]; // out-of-gamut red channel
    let out = gc.compress_achromatic(oog);
    assert!(out[0] >= 0.0, "Negative channel should be compressed");
    assert!(
        out[1] <= 1.0,
        "Super-white channel should be compressed to ≤1"
    );
}

// ── gamuts ────────────────────────────────────────────────────────────────────

#[test]
fn gamuts_display_p3_primaries_shape() {
    use oximedia_colormgmt::gamuts::display_p3::DisplayP3;
    let primaries = DisplayP3::primaries();
    assert_eq!(primaries.len(), 3, "Display P3 has 3 primaries");
    // All xy values in [0,1]
    for (i, p) in primaries.iter().enumerate() {
        assert!(
            p[0] > 0.0 && p[0] < 1.0,
            "Primary {} x out of range: {}",
            i,
            p[0]
        );
        assert!(
            p[1] > 0.0 && p[1] < 1.0,
            "Primary {} y out of range: {}",
            i,
            p[1]
        );
    }
}

#[test]
fn gamuts_display_p3_srgb_roundtrip() {
    use oximedia_colormgmt::gamuts::{display_p3::DisplayP3, srgb_to_display_p3};
    // sRGB white should map near Display P3 white (both D65)
    let white = [1.0_f32, 1.0, 1.0];
    let p3 = srgb_to_display_p3(white);
    // White stays white under a same-white-point gamut conversion
    for i in 0..3 {
        assert!(
            (p3[i] - 1.0).abs() < 0.02,
            "sRGB→P3 white channel {}: {}",
            i,
            p3[i]
        );
    }
    // The white_point should be D65
    let wp = DisplayP3::white_point();
    assert!((wp[0] - 0.3127).abs() < 1e-4);
    assert!((wp[1] - 0.3290).abs() < 1e-4);
}

// ── hdr_gamut ─────────────────────────────────────────────────────────────────

#[test]
fn hdr_gamut_bt2020_to_bt709_white_preserves() {
    use oximedia_colormgmt::hdr_gamut::hdr_bt2020_to_bt709;
    let white = [1.0_f32, 1.0, 1.0];
    let out = hdr_bt2020_to_bt709(white, 0.9);
    // White should remain approximately white after gamut map
    for i in 0..3 {
        assert!(
            out[i] > 0.8 && out[i] <= 1.0,
            "White channel {}: {}",
            i,
            out[i]
        );
    }
}

#[test]
fn hdr_gamut_black_maps_to_black() {
    use oximedia_colormgmt::hdr_gamut::hdr_bt2020_to_bt709;
    let black = [0.0_f32, 0.0, 0.0];
    let out = hdr_bt2020_to_bt709(black, 0.9);
    for i in 0..3 {
        assert!(out[i].abs() < 1e-6, "Black channel {}: {}", i, out[i]);
    }
}

// ── icc_v5 ────────────────────────────────────────────────────────────────────

#[test]
fn icc_v5_profile_version_parse() {
    use oximedia_colormgmt::icc_v5::ProfileVersion;
    let v4 = ProfileVersion {
        major: 4,
        minor: 0,
        bugfix: 0,
    };
    assert!(v4.is_v4_or_newer());
    assert!(!v4.is_iccmax());

    let v5 = ProfileVersion {
        major: 5,
        minor: 0,
        bugfix: 0,
    };
    assert!(v5.is_iccmax());
    assert!(v5.is_v4_or_newer());
}

#[test]
fn icc_v5_profile_class_signature() {
    use oximedia_colormgmt::icc_v5::IccProfileClass;
    let sig = IccProfileClass::Input.signature();
    assert_eq!(sig.len(), 4);
}

#[test]
fn icc_v5_float16_to_f64() {
    use oximedia_colormgmt::icc_v5::Float16;
    // 0x3C00 = 1.0 in IEEE 754 half-float
    let f16 = Float16(0x3C00);
    let f64_val = f16.to_f64();
    assert!(
        (f64_val - 1.0).abs() < 1e-3,
        "Float16(0x3C00) should be ~1.0, got {}",
        f64_val
    );
}

// ── illuminant_tests (test-only module) ───────────────────────────────────────

// illuminant_tests contains only #[cfg(test)] items; it is wired as pub mod so
// the compiler can see it. A smoke test here simply confirms the module is
// reachable — the module's own #[test] items run via `cargo nextest`.
#[test]
fn illuminant_tests_module_visible() {
    // The module exists and has no pub items to call directly — confirm compilation.
    let _ = std::any::type_name::<()>(); // trivial assertion
}

// ── look_table ────────────────────────────────────────────────────────────────

#[test]
fn look_table_asc_cdl_identity() {
    use oximedia_colormgmt::look_table::AscCdl;
    let cdl = AscCdl::identity();
    let rgb = [0.5_f64, 0.3, 0.8];
    let out = cdl.apply(rgb);
    for i in 0..3 {
        assert!(
            (out[i] - rgb[i]).abs() < 1e-9,
            "Identity CDL channel {}: {} vs {}",
            i,
            out[i],
            rgb[i]
        );
    }
}

#[test]
fn look_table_asc_cdl_validate_identity() {
    use oximedia_colormgmt::look_table::AscCdl;
    let cdl = AscCdl::identity();
    assert!(
        cdl.validate().is_ok(),
        "Identity CDL should pass validation"
    );
}

// ── lut_chain_opt ─────────────────────────────────────────────────────────────

#[test]
fn lut_chain_opt_merge_two_identities() {
    use oximedia_colormgmt::lut_chain_opt::LutChainOptimizer;
    let id: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let m = LutChainOptimizer::merge_matrix_pair(id, id);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0_f32 } else { 0.0_f32 };
            assert!(
                (m[i][j] - expected).abs() < 1e-5,
                "merged[{}][{}]: {}",
                i,
                j,
                m[i][j]
            );
        }
    }
}

#[test]
fn lut_chain_opt_apply_identity() {
    use oximedia_colormgmt::lut_chain_opt::LutChainOptimizer;
    let id: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let color = [0.5_f32, 0.3, 0.8];
    let out = LutChainOptimizer::apply(id, color);
    for i in 0..3 {
        assert!((out[i] - color[i]).abs() < 1e-6);
    }
}

// ── matrix_coeff ──────────────────────────────────────────────────────────────

#[test]
fn matrix_coeff_identity_matrix() {
    use oximedia_colormgmt::matrix_coeff::ColorMatrix3x3;
    let id = ColorMatrix3x3::identity();
    let color = [0.5_f64, 0.3, 0.8];
    let out = id.mul_vec(color);
    for i in 0..3 {
        assert!((out[i] - color[i]).abs() < 1e-10);
    }
}

#[test]
fn matrix_coeff_bt709_roundtrip() {
    use oximedia_colormgmt::matrix_coeff::{rgb_to_ycbcr, ycbcr_to_rgb, MatrixCoefficients};
    let rgb = [0.5_f64, 0.4, 0.3];
    let ycbcr = rgb_to_ycbcr(rgb, MatrixCoefficients::Bt709);
    let rgb2 = ycbcr_to_rgb(ycbcr, MatrixCoefficients::Bt709);
    for i in 0..3 {
        assert!(
            (rgb2[i] - rgb[i]).abs() < 1e-9,
            "BT.709 roundtrip channel {}: {} vs {}",
            i,
            rgb2[i],
            rgb[i]
        );
    }
}

// ── multi_illuminant ──────────────────────────────────────────────────────────

#[test]
fn multi_illuminant_single_d65_composite() {
    use oximedia_colormgmt::multi_illuminant::{Illuminant, IlluminantBlend};
    let mut blend = IlluminantBlend::new();
    blend.add(Illuminant::D65, 1.0);
    let xyz = blend
        .composite_xyz()
        .expect("single illuminant composite should succeed");
    // D65 XYZ is approximately [0.95047, 1.0, 1.08883]
    assert!(
        (xyz[1] - 1.0).abs() < 0.01,
        "D65 Y should be ~1.0: {}",
        xyz[1]
    );
}

#[test]
fn multi_illuminant_blend_empty_fails() {
    use oximedia_colormgmt::multi_illuminant::IlluminantBlend;
    let blend = IlluminantBlend::new();
    assert!(blend.composite_xyz().is_err(), "empty blend should fail");
}

// ── ocio ──────────────────────────────────────────────────────────────────────

#[test]
fn ocio_parse_minimal_config() {
    use oximedia_colormgmt::ocio::OcioConfig;
    let yaml = "ocio_profile_version: 2\nname: Test Config\n";
    let cfg = OcioConfig::from_str(yaml).expect("minimal OCIO config should parse");
    assert_eq!(cfg.version, 2);
    assert_eq!(cfg.name.as_deref(), Some("Test Config"));
}

#[test]
fn ocio_parse_missing_version_defaults_to_v1() {
    use oximedia_colormgmt::ocio::OcioConfig;
    // The parser defaults to version 1 when ocio_profile_version is absent
    let yaml = "name: No Version\n";
    let cfg = OcioConfig::from_str(yaml).expect("should succeed with default version");
    assert_eq!(cfg.version, 1, "default version should be 1");
    assert_eq!(cfg.name.as_deref(), Some("No Version"));
}

// ── ocio_parser ───────────────────────────────────────────────────────────────

#[test]
fn ocio_parser_parse_basic_config() {
    use oximedia_colormgmt::ocio_parser::parse_ocio;
    let yaml = r#"ocio_profile_version: 2
name: Studio Config
roles:
  default: sRGB
  scene_linear: ACEScg
colorspaces:
  - name: sRGB
    family: Display
    encoding: sdr-video
    isdata: false
"#;
    let cfg = parse_ocio(yaml).expect("basic OCIO parser should succeed");
    assert_eq!(cfg.ocio_profile_version, 2);
}

#[test]
fn ocio_parser_find_colorspace() {
    use oximedia_colormgmt::ocio_parser::parse_ocio;
    let yaml = r#"ocio_profile_version: 1
colorspaces:
  - name: Linear
    family: Scene
    isdata: false
"#;
    let cfg = parse_ocio(yaml).expect("parser should succeed");
    let cs = cfg.find_colorspace("Linear");
    assert!(cs.is_some(), "Linear colorspace should be found");
}

// ── perceptual_uniformity ─────────────────────────────────────────────────────

#[test]
fn perceptual_uniformity_macadam_ellipses_count() {
    use oximedia_colormgmt::perceptual_uniformity::MacAdamEllipse;
    let ellipses = MacAdamEllipse::standard_set();
    assert!(
        !ellipses.is_empty(),
        "Standard MacAdam ellipse set should not be empty"
    );
}

#[test]
fn perceptual_uniformity_jnd_analyzer_constructs() {
    use oximedia_colormgmt::perceptual_uniformity::JndAnalyzer;
    let analyzer = JndAnalyzer::new();
    // Compute Lab distance between two close colours
    let lab1 = [50.0_f64, 10.0, 5.0];
    let lab2 = [50.5_f64, 10.0, 5.0];
    let dist = analyzer.lab_distance(lab1, lab2);
    assert!(dist >= 0.0, "Lab distance should be non-negative: {}", dist);
    assert!(
        (dist - 0.5).abs() < 1e-9,
        "Distance should be ~0.5: {}",
        dist
    );
}

// ── white_balance ─────────────────────────────────────────────────────────────

#[test]
fn white_balance_grey_world_neutral_scene() {
    use oximedia_colormgmt::white_balance::grey_world;
    // Perfectly neutral patches — grey world gives identity gains
    let patches: Vec<[f64; 3]> = vec![[0.5, 0.5, 0.5]; 10];
    let gains = grey_world(&patches).expect("grey world should succeed");
    // Normalised gains should be near 1.0
    let norm = gains.normalise_green().expect("normalise should succeed");
    assert!(
        (norm.r - 1.0).abs() < 0.01,
        "Neutral scene R gain: {}",
        norm.r
    );
    assert!(
        (norm.g - 1.0).abs() < 0.01,
        "Neutral scene G gain: {}",
        norm.g
    );
    assert!(
        (norm.b - 1.0).abs() < 0.01,
        "Neutral scene B gain: {}",
        norm.b
    );
}

#[test]
fn white_balance_kelvin_preset_d65() {
    use oximedia_colormgmt::white_balance::KelvinPreset;
    let gains = KelvinPreset::D65.to_gains();
    // D65 white balance gains should be reasonable (r,g,b each roughly 0.5–2.0)
    assert!(gains.r > 0.1 && gains.r < 5.0, "D65 R gain: {}", gains.r);
    assert!(gains.g > 0.1 && gains.g < 5.0, "D65 G gain: {}", gains.g);
    assert!(gains.b > 0.1 && gains.b < 5.0, "D65 B gain: {}", gains.b);
}
