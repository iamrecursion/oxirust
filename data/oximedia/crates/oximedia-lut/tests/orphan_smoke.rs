//! Smoke tests for the 22 orphan modules registered in 0.1.8 Wave 6.
//!
//! Each orphan gets at least one smoke test verifying the public API compiles
//! and behaves correctly at a high level.

// ---------------------------------------------------------------------------
// blend — LUT strength / blend utilities (free-function API, f32 scalars)
// ---------------------------------------------------------------------------

#[test]
fn blend_full_strength_returns_lut_out() {
    let original = [0.1_f32, 0.2, 0.3];
    let lut_out = [0.8_f32, 0.7, 0.6];
    let result = oximedia_lut::blend::blend_lut_result(original, lut_out, 1.0);
    assert!((result[0] - 0.8).abs() < 1e-5);
    assert!((result[1] - 0.7).abs() < 1e-5);
    assert!((result[2] - 0.6).abs() < 1e-5);
}

#[test]
fn blend_zero_strength_returns_original() {
    let original = [0.4_f32, 0.5, 0.6];
    let lut_out = [0.9_f32, 0.9, 0.9];
    let result = oximedia_lut::blend::blend_lut_result(original, lut_out, 0.0);
    assert!((result[0] - 0.4).abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// clf — CLF XML round-trip (1141-line implementation)
// ---------------------------------------------------------------------------

#[test]
fn clf_lut1d_xml_roundtrip() {
    use oximedia_lut::clf::{ClfDocument, ClfLut1d, ClfNode};

    let lut1d = ClfLut1d {
        id: "l0".to_string(),
        name: Some("identity".to_string()),
        size: 3,
        data: vec![[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
    };
    let doc = ClfDocument::new("test", "Identity", vec![ClfNode::Lut1d(lut1d)]);
    let xml = doc.to_xml();
    assert!(xml.contains("LUT1D"), "output XML: {xml}");
    assert!(xml.contains("ProcessList"), "output XML: {xml}");

    // Parse back — CLF round-trip
    let doc2 = ClfDocument::from_xml(&xml).expect("re-parse should succeed");
    assert_eq!(doc2.nodes.len(), 1);
    assert_eq!(doc2.nodes[0].id(), "l0");
}

#[test]
fn clf_lut1d_apply_identity() {
    use oximedia_lut::clf::{ClfLut1d, ClfNode};

    let lut1d = ClfLut1d {
        id: "l0".to_string(),
        name: None,
        size: 3,
        data: vec![[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
    };
    let node = ClfNode::Lut1d(lut1d);
    let out = node.apply([0.5, 0.5, 0.5]);
    assert!((out[0] - 0.5).abs() < 0.01, "out[0]={}", out[0]);
}

// ---------------------------------------------------------------------------
// color_chart — chart analysis & LUT generation
// ---------------------------------------------------------------------------

#[test]
fn color_chart_analyzer_generates_correction_lut() {
    use oximedia_lut::color_chart::{ColorChart, ColorChartAnalyzer};

    let measured = vec![
        [0.18_f64, 0.10, 0.08],
        [0.35, 0.22, 0.17],
        [0.10, 0.17, 0.25],
        [0.18, 0.18, 0.18],
    ];
    let reference = vec![
        [0.20_f64, 0.12, 0.09],
        [0.38, 0.25, 0.19],
        [0.11, 0.19, 0.28],
        [0.18, 0.18, 0.18],
    ];
    let chart = ColorChart::from_patches(measured, reference).expect("build chart");
    let analyzer = ColorChartAnalyzer::new(chart);
    let lut = analyzer.generate_3d_lut(17);
    // 17³ entries
    assert_eq!(lut.len(), 17 * 17 * 17);
}

// ---------------------------------------------------------------------------
// combine — 1-D LUT sequential combination (free-function API)
// ---------------------------------------------------------------------------

#[test]
fn combine_identity_luts_1d_passthrough() {
    use oximedia_lut::combine::combine_luts_1d;

    let identity: Vec<f32> = (0..=16).map(|i| i as f32 / 16.0).collect();
    let combined = combine_luts_1d(&identity, &identity);
    assert_eq!(combined.len(), identity.len());
    for (a, b) in identity.iter().zip(combined.iter()) {
        assert!((a - b).abs() < 1e-4, "a={a} b={b}");
    }
}

#[test]
fn combine_empty_luts_returns_empty() {
    use oximedia_lut::combine::combine_luts_1d;

    let result = combine_luts_1d(&[], &[0.0_f32, 1.0]);
    assert!(result.is_empty());
}

// ---------------------------------------------------------------------------
// creative_grade — film emulation presets
// ---------------------------------------------------------------------------

#[test]
fn creative_grade_kodak_vision3_generates_lut() {
    use oximedia_lut::creative_grade::FilmPreset;

    let preset = FilmPreset::kodak_vision3_250d();
    let lut = preset.bake_lut3d(17);
    assert_eq!(lut.len(), 17 * 17 * 17);
}

#[test]
fn creative_grade_apply_pixel_no_panic() {
    use oximedia_lut::creative_grade::FilmPreset;

    let preset = FilmPreset::sepia_tone();
    let out = preset.apply(&[0.5, 0.5, 0.5]);
    assert!(out[0] >= 0.0 && out[0] <= 1.0, "out[0]={}", out[0]);
}

#[test]
fn creative_grade_identity_preset_is_passthrough() {
    use oximedia_lut::creative_grade::FilmPreset;

    let preset = FilmPreset::identity();
    let out = preset.apply(&[0.6, 0.4, 0.2]);
    assert!((out[0] - 0.6).abs() < 0.01, "r={}", out[0]);
    assert!((out[1] - 0.4).abs() < 0.01, "g={}", out[1]);
    assert!((out[2] - 0.2).abs() < 0.01, "b={}", out[2]);
}

// ---------------------------------------------------------------------------
// davinci — DaVinci Resolve .cube export (format A)
// ---------------------------------------------------------------------------

#[test]
fn davinci_to_davinci_cube_contains_header() {
    let identity_lut = [[[0.5_f32; 3]; 33]; 33];
    let cube_str = oximedia_lut::davinci::to_davinci_cube(&identity_lut, 33);
    assert!(
        cube_str.contains("LUT_3D_SIZE"),
        "should contain LUT_3D_SIZE header"
    );
    assert!(cube_str.contains("33"), "should contain size 33");
}

#[test]
fn davinci_cube_titled_includes_title() {
    let identity_lut = [[[0.5_f32; 3]; 33]; 33];
    let cube_str = oximedia_lut::davinci::to_davinci_cube_titled(&identity_lut, 33, "MyLUT");
    assert!(cube_str.contains("MyLUT"));
}

// ---------------------------------------------------------------------------
// display_calibration — display calibration LUT generation
// ---------------------------------------------------------------------------

#[test]
fn display_calibration_1d_lut_has_correct_size() {
    use oximedia_lut::display_calibration::{CalibrationTarget, DisplayCalibrator};

    // No patches → estimate_display_gamma falls back to default 2.2
    let calibrator = DisplayCalibrator::new(vec![], CalibrationTarget::D65Srgb);
    let lut = calibrator.generate_1d_lut(17);
    assert_eq!(lut.len(), 17);
}

#[test]
fn display_calibration_3d_lut_needs_patches() {
    use oximedia_lut::display_calibration::{
        CalibrationPatch, CalibrationTarget, DisplayCalibrator,
    };

    // Build 4 minimal measurement patches (one per primaries + white)
    let patches = vec![
        CalibrationPatch::new([1.0, 0.0, 0.0], [0.41, 0.21, 0.02]),
        CalibrationPatch::new([0.0, 1.0, 0.0], [0.36, 0.72, 0.12]),
        CalibrationPatch::new([0.0, 0.0, 1.0], [0.18, 0.07, 0.95]),
        CalibrationPatch::new([1.0, 1.0, 1.0], [0.95, 1.00, 1.09]),
    ];
    let calibrator = DisplayCalibrator::new(patches, CalibrationTarget::D65Srgb);
    let result = calibrator.generate_3d_lut(9);
    assert!(result.is_ok(), "3D LUT generation failed: {:?}", result);
    let lut = result.unwrap();
    assert_eq!(lut.len(), 9 * 9 * 9);
}

// ---------------------------------------------------------------------------
// hald_clut — identity CLUT apply (CFIX anchor test)
// ---------------------------------------------------------------------------

#[test]
fn hald_clut_identity_passthrough() {
    use oximedia_lut::hald_clut::HaldClut;

    let h = HaldClut::identity(4);
    let (r, g, b) = h.apply(0.5, 0.3, 0.8);
    assert!((r - 0.5).abs() < 0.02, "r={r}");
    assert!((g - 0.3).abs() < 0.02, "g={g}");
    assert!((b - 0.8).abs() < 0.02, "b={b}");
}

#[test]
fn hald_clut_lut3d_data_identity_lookup() {
    use oximedia_lut::hald_clut::Lut3DData;

    let lut = Lut3DData::identity(8);
    // Corner lookup: black
    let black = lut.lookup(0.0, 0.0, 0.0);
    assert!(black[0].abs() < 1e-4, "black.r={}", black[0]);
    // Corner lookup: white
    let white = lut.lookup(1.0, 1.0, 1.0);
    assert!((white[0] - 1.0).abs() < 1e-4, "white.r={}", white[0]);
    // Mid-grey
    let grey = lut.lookup(0.5, 0.5, 0.5);
    assert!((grey[0] - 0.5).abs() < 0.02, "grey.r={}", grey[0]);
}

// ---------------------------------------------------------------------------
// icc_profile — ICC matrix/TRC profile to LUT
// ---------------------------------------------------------------------------

#[test]
fn icc_profile_srgb_lut3d_has_correct_size() {
    use oximedia_lut::icc_profile::{IccMatrixProfile, IccToLutConverter, IccTrc};

    let profile = IccMatrixProfile {
        description: "sRGB IEC61966-2-1".to_string(),
        matrix_to_xyz_d50: [
            [0.436_065, 0.385_151, 0.143_081],
            [0.222_491, 0.716_888, 0.060_621],
            [0.013_920, 0.097_045, 0.714_136],
        ],
        trc_r: IccTrc::Parametric {
            gamma: 2.4,
            a: 1.0 / 1.055,
            b: 0.055 / 1.055,
            c: 1.0 / 12.92,
            d: 0.04045,
            e: 0.0,
            f: 0.0,
        },
        trc_g: IccTrc::Gamma(2.2),
        trc_b: IccTrc::Gamma(2.2),
    };

    let converter = IccToLutConverter::new(profile);
    let lut = converter.to_lut3d(17);
    assert_eq!(lut.len(), 17 * 17 * 17);
}

#[test]
fn icc_profile_identity_trc_passthrough() {
    use oximedia_lut::icc_profile::IccTrc;

    let trc = IccTrc::Identity;
    assert!((trc.apply(0.5) - 0.5).abs() < 1e-10);
    assert!((trc.apply(0.0)).abs() < 1e-10);
    assert!((trc.apply(1.0) - 1.0).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// invert — 1-D LUT inversion
// ---------------------------------------------------------------------------

#[test]
fn invert_1d_lut_identity_roundtrip() {
    use oximedia_lut::invert::invert_1d_lut;

    let identity: Vec<f32> = (0..=16).map(|i| i as f32 / 16.0).collect();
    let inv = invert_1d_lut(&identity);
    assert_eq!(inv.len(), identity.len());
    // Inversion of identity should be identity
    for (a, b) in identity.iter().zip(inv.iter()) {
        assert!((a - b).abs() < 0.02, "a={a} b={b}");
    }
}

#[test]
fn invert_1d_lut_empty_returns_empty() {
    use oximedia_lut::invert::invert_1d_lut;

    let inv = invert_1d_lut(&[]);
    assert!(inv.is_empty());
}

// ---------------------------------------------------------------------------
// log_to_display — log-to-display LUT generation
// ---------------------------------------------------------------------------

#[test]
fn log_to_display_slog3_srgb_3d_lut_has_correct_size() {
    use oximedia_lut::log_to_display::{
        generate_log_to_display_lut, DisplayTarget, LogCurve, LogToDisplayParams,
    };

    let params = LogToDisplayParams::new(LogCurve::SLog3, DisplayTarget::Srgb);
    let lut = generate_log_to_display_lut(17, &params).expect("slog3 lut");
    assert_eq!(lut.len(), 17 * 17 * 17);
}

#[test]
fn log_to_display_apply_mid_grey_is_finite() {
    use oximedia_lut::log_to_display::{
        process_rgb_log_to_display, DisplayTarget, LogCurve, LogToDisplayParams,
    };

    let params = LogToDisplayParams::new(LogCurve::VLog, DisplayTarget::Rec709);
    let out = process_rgb_log_to_display(&[0.5, 0.5, 0.5], &params);
    assert!(out[0].is_finite(), "r={}", out[0]);
    assert!(out[1].is_finite(), "g={}", out[1]);
    assert!(out[2].is_finite(), "b={}", out[2]);
}

// ---------------------------------------------------------------------------
// lut_blend — struct-based LUT blending API (blend_lut3d / blend_curve)
// ---------------------------------------------------------------------------

#[test]
fn lut_blend_linear_mix_midpoint() {
    use oximedia_lut::lut_blend::{blend_curve, BlendMode};

    // Identity-like: all zeros
    let a = vec![[0.0_f64, 0.0, 0.0]; 3];
    // All-ones
    let b = vec![[1.0_f64, 1.0, 1.0]; 3];
    let result = blend_curve(&a, &b, 0.5, BlendMode::Linear).expect("blend");
    assert_eq!(result.len(), 3);
    for entry in &result {
        assert!((entry[0] - 0.5).abs() < 0.01, "v={}", entry[0]);
    }
}

#[test]
fn lut_blend_3d_identity_blend_produces_mid() {
    use oximedia_lut::lut_blend::{blend_lut3d, BlendMode};

    let size = 2_usize;
    let entries = size * size * size;
    let lut_a = vec![[0.0_f64, 0.0, 0.0]; entries];
    let lut_b = vec![[1.0_f64, 1.0, 1.0]; entries];
    let result = blend_lut3d(&lut_a, &lut_b, size, 0.5, BlendMode::Linear).expect("blend3d");
    assert_eq!(result.len(), entries);
    for entry in &result {
        assert!((entry[0] - 0.5).abs() < 0.01);
    }
}

// ---------------------------------------------------------------------------
// lut_chain_ops — chain two identity LUTs (uses hald_clut::Lut3DData)
// ---------------------------------------------------------------------------

#[test]
fn lut_chain_ops_two_identity_luts_passthrough() {
    use oximedia_lut::hald_clut::Lut3DData;
    use oximedia_lut::lut_chain_ops::{LutChainOps, LutOperation};

    let identity = Lut3DData::identity(17);
    let mut chain = LutChainOps::new();
    chain.push(LutOperation::Apply3D(identity.clone()));
    chain.push(LutOperation::Apply3D(identity));

    let (ro, go, bo) = chain.apply(0.5, 0.5, 0.5);
    assert!((ro - 0.5).abs() < 0.03, "r={ro}");
    assert!((go - 0.5).abs() < 0.03, "g={go}");
    assert!((bo - 0.5).abs() < 0.03, "b={bo}");
}

#[test]
fn lut_chain_ops_bake_and_apply_consistent() {
    use oximedia_lut::lut_chain_ops::{LutChainOps, LutOperation};

    let mut chain = LutChainOps::new();
    chain.push(LutOperation::Saturation(0.5));

    // Apply before baking
    let pre_bake = chain.apply(0.8, 0.2, 0.4);
    chain.bake();
    // Apply after baking — should be within trilinear interpolation tolerance
    let post_bake = chain.apply(0.8, 0.2, 0.4);
    assert!(
        (pre_bake.0 - post_bake.0).abs() < 0.02,
        "pre={pre_bake:?} post={post_bake:?}"
    );
}

// ---------------------------------------------------------------------------
// lut_compress — lossy LUT compression
// ---------------------------------------------------------------------------

#[test]
fn lut_compress_decimate_and_measure() {
    use oximedia_lut::lut_compress::{compress_decimate, compute_metrics};

    let n = 9_usize;
    let source: Vec<[f64; 3]> = {
        let scale = (n - 1) as f64;
        let mut v = Vec::with_capacity(n * n * n);
        for bi in 0..n {
            for gi in 0..n {
                for ri in 0..n {
                    v.push([ri as f64 / scale, gi as f64 / scale, bi as f64 / scale]);
                }
            }
        }
        v
    };
    // Compress to size 5, then up-sample back to 9 for metrics
    let small = compress_decimate(&source, n, 5).expect("compress");
    assert_eq!(small.len(), 5 * 5 * 5);

    let reconstructed = {
        use oximedia_lut::lut_compress::decompress_upsample;
        decompress_upsample(&small, 5, n).expect("decompress")
    };
    assert_eq!(reconstructed.len(), n * n * n);
    let metrics = compute_metrics(&source, &reconstructed).expect("metrics");
    assert!(metrics.psnr_db > 0.0, "PSNR={}", metrics.psnr_db);
    assert_eq!(metrics.original_entries, n * n * n);
}

// ---------------------------------------------------------------------------
// lut_export — LUT export to .drx / .csp / .look
// ---------------------------------------------------------------------------

#[test]
fn lut_export_to_resolve_drx_contains_xml() {
    use oximedia_lut::lut3d::Lut3d;
    use oximedia_lut::lut_export::LutExporter;
    use oximedia_lut::LutSize;

    let lut = Lut3d::identity(LutSize::Size17);
    let drx = LutExporter::to_resolve_drx(&lut);
    assert!(
        drx.contains("<?xml") || drx.contains("<resolve") || drx.contains("drx"),
        "drx output: {drx}"
    );
}

#[test]
fn lut_export_to_csp_contains_header() {
    use oximedia_lut::lut3d::Lut3d;
    use oximedia_lut::lut_export::LutExporter;
    use oximedia_lut::LutSize;

    let lut = Lut3d::identity(LutSize::Size17);
    let csp = LutExporter::to_nuke_csp(&lut);
    assert!(!csp.is_empty(), "CSP output should not be empty");
}

// ---------------------------------------------------------------------------
// lut_preview_html — HTML preview generator
// ---------------------------------------------------------------------------

#[test]
fn lut_preview_html_is_valid_html5() {
    use oximedia_lut::lut3d::Lut3d;
    use oximedia_lut::lut_preview_html::{generate_lut_preview_html, LutPreviewOptions};
    use oximedia_lut::LutSize;

    let lut = Lut3d::identity(LutSize::Size17);
    let opts = LutPreviewOptions::default();
    let html = generate_lut_preview_html(&lut, "Identity LUT", &opts);
    assert!(html.contains("<!DOCTYPE html"), "html should be HTML5");
    assert!(html.contains("Identity LUT"), "title should appear");
}

// ---------------------------------------------------------------------------
// lut_resample — LUT resampling utilities
// ---------------------------------------------------------------------------

#[test]
fn lut_resample_3d_identity_stays_identity() {
    use oximedia_lut::lut_resample::{resample_3d, ResampleMethod};

    let n = 9_usize;
    let scale = (n - 1) as f64;
    // Build identity LUT as flat Vec<[f64; 3]>
    let source: Vec<[f64; 3]> = (0..n * n * n)
        .map(|idx| {
            let ri = (idx % (n * n)) % n;
            let gi = (idx % (n * n)) / n;
            let bi = idx / (n * n);
            [ri as f64 / scale, gi as f64 / scale, bi as f64 / scale]
        })
        .collect();

    let result = resample_3d(&source, n, 5, ResampleMethod::Trilinear);
    assert_eq!(result.target_size, 5);
    assert_eq!(result.data.len(), 5 * 5 * 5);
    // Mid-grey should still be near 0.5 after resampling
    let mid = 2_usize; // index 2 in a 5-point grid (0..4)
    let idx = mid * 5 * 5 + mid * 5 + mid;
    assert!(
        (result.data[idx][0] - 0.5).abs() < 0.02,
        "r={}",
        result.data[idx][0]
    );
}

// ---------------------------------------------------------------------------
// lut_smoother — Gaussian LUT smoothing
// ---------------------------------------------------------------------------

#[test]
fn lut_smoother_identity_remains_near_identity() {
    use oximedia_lut::lut_smoother::{smooth_gaussian, SmootherParams};

    let n = 9_usize;
    let scale = (n - 1) as f64;
    // Build identity LUT in r-major order: index = r * size² + g * size + b
    let data: Vec<[f64; 3]> = (0..n * n * n)
        .map(|idx| {
            let ri = idx / (n * n);
            let gi = (idx / n) % n;
            let bi = idx % n;
            [ri as f64 / scale, gi as f64 / scale, bi as f64 / scale]
        })
        .collect();

    let params = SmootherParams::with_radius(1);
    let smoothed = smooth_gaussian(&data, n, &params).expect("smooth");
    assert_eq!(smoothed.len(), data.len());
    // Mid-point should be approximately 0.5 after smoothing identity
    let mid = n / 2;
    let idx = mid * n * n + mid * n + mid;
    assert!(
        (smoothed[idx][0] - 0.5).abs() < 0.05,
        "mid={}",
        smoothed[idx][0]
    );
}

// ---------------------------------------------------------------------------
// photographic_luts — pre-built photographic presets (uses hald_clut::Lut3DData)
// ---------------------------------------------------------------------------

#[test]
fn photographic_luts_fuji_chrome_generates_lut() {
    use oximedia_lut::photographic_luts::PhotoLutPreset;

    let lut = PhotoLutPreset::FujiChrome.to_lut3d();
    // to_lut3d() always generates a 33³ LUT
    assert_eq!(lut.size, 33);
    assert_eq!(lut.data.len(), 33 * 33 * 33);
}

#[test]
fn photographic_luts_all_presets_apply_no_nan() {
    use oximedia_lut::photographic_luts::PhotoLutPreset;

    let presets = [
        PhotoLutPreset::FilmNoir,
        PhotoLutPreset::Kodachrome,
        PhotoLutPreset::FujiChrome,
        PhotoLutPreset::Vintage,
    ];
    for preset in &presets {
        let (r, g, b) = PhotoLutPreset::apply_to_pixel(0.5, 0.5, 0.5, preset);
        assert!(r.is_finite(), "{:?}: r={r}", preset);
        assert!(g.is_finite(), "{:?}: g={g}", preset);
        assert!(b.is_finite(), "{:?}: b={b}", preset);
    }
}

// ---------------------------------------------------------------------------
// resolve_lut — DaVinci Resolve .drx parser (format B / XML node graph)
// ---------------------------------------------------------------------------

#[test]
fn resolve_lut_parse_drx_version() {
    use oximedia_lut::resolve_lut::ResolveLutParser;

    let drx = r#"<?xml version="1.0" encoding="UTF-8"?>
<resolve_davinci_resolve version="1.0">
  <grade version="1.0">
    <node type="serial" enabled="true">
      <correction type="lift_gamma_gain">
        <lift r="0.05" g="0.05" b="0.05" master="0.05"/>
        <gamma r="1.0" g="1.0" b="1.0" master="1.0"/>
        <gain r="1.2" g="1.1" b="1.0" master="1.1"/>
      </correction>
    </node>
  </grade>
</resolve_davinci_resolve>"#;

    let result = ResolveLutParser::parse_drx(drx).expect("parse");
    assert_eq!(result.version, "1.0");
    assert_eq!(result.nodes.len(), 1);
    let lift_r = result.nodes[0].get_param("lift.r").expect("lift.r");
    assert!((lift_r - 0.05).abs() < 1e-4, "lift.r={lift_r}");
}

// ---------------------------------------------------------------------------
// split_toning — shadow/highlight toning
// ---------------------------------------------------------------------------

#[test]
fn split_toning_zero_strength_is_passthrough() {
    use oximedia_lut::split_toning::{apply_split_toning, SplitToningParams};

    // strength = 0.0 means no toning applied regardless of hue/sat settings
    let params = SplitToningParams {
        shadow_hue: 200.0,
        shadow_saturation: 0.8,
        highlight_hue: 40.0,
        highlight_saturation: 0.8,
        balance: 0.0,
        strength: 0.0,
    };
    let input = [0.5_f64, 0.3, 0.7];
    let out = apply_split_toning(&input, &params);
    // With strength = 0, no toning should be applied — output must equal input
    assert!(
        (out[0] - input[0]).abs() < 1e-9,
        "r: {} vs {}",
        out[0],
        input[0]
    );
    assert!(
        (out[1] - input[1]).abs() < 1e-9,
        "g: {} vs {}",
        out[1],
        input[1]
    );
    assert!(
        (out[2] - input[2]).abs() < 1e-9,
        "b: {} vs {}",
        out[2],
        input[2]
    );
}

#[test]
fn split_toning_generates_lut() {
    use oximedia_lut::split_toning::{generate_split_toning_lut, SplitToningParams};

    let params = SplitToningParams {
        shadow_hue: 220.0,
        shadow_saturation: 0.3,
        highlight_hue: 30.0,
        highlight_saturation: 0.2,
        balance: 0.0,
        strength: 0.5,
    };
    let lut = generate_split_toning_lut(9, &params).expect("lut");
    assert_eq!(lut.len(), 9 * 9 * 9);
}

// ---------------------------------------------------------------------------
// validate — simple 1-D LUT validation predicates
// ---------------------------------------------------------------------------

#[test]
fn validate_monotone_identity_lut_passes() {
    use oximedia_lut::validate::check_1d_monotone;

    let identity: Vec<f32> = (0..=16).map(|i| i as f32 / 16.0).collect();
    assert!(check_1d_monotone(&identity));
}

#[test]
fn validate_non_monotone_lut_fails() {
    use oximedia_lut::validate::check_1d_monotone;

    let lut = [0.0_f32, 0.5, 0.3, 1.0]; // 0.3 < 0.5 → not monotone
    assert!(!check_1d_monotone(&lut));
}

#[test]
fn validate_clip_check_in_range_passes() {
    use oximedia_lut::validate::check_no_clipping;

    let lut = [0.0_f32, 0.25, 0.5, 0.75, 1.0];
    assert!(check_no_clipping(&lut));
}
